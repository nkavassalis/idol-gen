#!/usr/bin/env python
"""Idol: one image -> a 3D Gaussian splat (.ply), with the quality levers exposed as flags.

Examples
  ./idol-gen photo.jpg --mode real                  # photo, paper defaults (50 steps, 262k Gaussians)
  ./idol-gen render.png --preset draft              # quick preview (20 steps, 65k)
  ./idol-gen render.png --preset max                # 100 steps, 1,048,576 Gaussians
  ./idol-gen render.png --steps 80 --num_gaussians 524288 --cfg_strength 2.0
  ./idol-gen renders/ --out out/ --skip-existing    # a whole folder
  ./idol-gen render.png --variants 3                # seeds 42,43,44 side by side

Quality levers, roughly in order of impact:
  --num_gaussians  density of the splat (32,768 .. 1,048,576, multiple of 32)
  --steps          denoising steps; more recover detail, cost is linear
  --cfg_strength   how strictly the object follows the input image (0 = unconditional)
  --rescale_t      noise-schedule shift: higher spends more steps on shape, lower on detail
  --crop_margin    how much empty room the square crop leaves around the object
  --dinov3_dtype   fp32 image encoder costs ~1.7 GB more VRAM
"""
import argparse
import json
import os
import sys
import time


def _reexec_into_venv():
    """Run under the project venv no matter which python started us (deps live there, not in ~/.local)."""
    venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               '.venv', 'bin', 'python')
    if os.path.isfile(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.environ.setdefault('PYTHONNOUSERSITE', '1')
        os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])


_reexec_into_venv()

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')

# name -> (steps, num_gaussians, cfg_strength, rescale_t)
PRESETS = {
    'draft':    dict(steps=20,  num_gaussians=65_536,    cfg_strength=3.0, rescale_t=3.0),
    'standard': dict(steps=50,  num_gaussians=262_144,   cfg_strength=3.0, rescale_t=3.0),
    'high':     dict(steps=75,  num_gaussians=524_288,   cfg_strength=3.0, rescale_t=3.0),
    'max':      dict(steps=100, num_gaussians=1_048_576, cfg_strength=3.0, rescale_t=3.0),
}
MIN_GAUSSIANS, MAX_GAUSSIANS, GAUSSIAN_MULTIPLE = 32_768, 1_048_576, 32


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f'must be a positive integer, got {value}')
    return ivalue


def gaussians(value):
    ivalue = int(value)
    if not MIN_GAUSSIANS <= ivalue <= MAX_GAUSSIANS:
        raise argparse.ArgumentTypeError(
            f'--num_gaussians must be between {MIN_GAUSSIANS:,} and {MAX_GAUSSIANS:,}, got {ivalue:,}')
    if ivalue % GAUSSIAN_MULTIPLE:
        raise argparse.ArgumentTypeError(
            f'--num_gaussians must be a multiple of {GAUSSIAN_MULTIPLE}, got {ivalue:,} '
            f'(nearest valid: {ivalue - ivalue % GAUSSIAN_MULTIPLE:,})')
    return ivalue


def build_parser():
    parser = argparse.ArgumentParser(
        prog='idol-gen', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='presets: ' + ',  '.join(
            f'{name} (steps={p["steps"]}, gaussians={p["num_gaussians"]:,})' for name, p in PRESETS.items()))
    parser.add_argument('image', help='an image file, or a folder of images')
    parser.add_argument('-o', '--out', default='outputs', help='output directory (default: outputs)')
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic',
                        help="'synthetic': renders / generated images of one object (crop 1.2). "
                             "'real': photographs (crop 1.7). Overridden by --crop_margin.")
    parser.add_argument('--preset', choices=sorted(PRESETS), default='standard',
                        help='starting point for steps/gaussians/cfg/rescale_t; explicit flags still win '
                             '(default: standard)')
    # quality levers
    parser.add_argument('--steps', type=positive_int, default=None, help='denoising steps (default: preset)')
    parser.add_argument('--num_gaussians', '--gaussians', type=gaussians, default=None, metavar='N',
                        help=f'{MIN_GAUSSIANS:,}-{MAX_GAUSSIANS:,}, multiple of {GAUSSIAN_MULTIPLE} (default: preset)')
    parser.add_argument('--cfg_strength', '--cfg', type=float, default=None, metavar='W',
                        help='classifier-free guidance: how strictly the object follows the image (default: preset)')
    parser.add_argument('--rescale_t', type=float, default=None, metavar='T',
                        help='time shift: >1 spends more steps on overall shape, <1 on fine detail (default: preset)')
    parser.add_argument('--crop_margin', type=float, default=None, metavar='M',
                        help='override the mode crop: square side = object bounding box x this')
    parser.add_argument('--seed', type=int, default=42, help='fix it to get the same object again')
    parser.add_argument('--variants', type=positive_int, default=1, metavar='N',
                        help='generate N seeds (seed, seed+1, ...) so you can pick the best (default: 1)')
    # precision / speed
    parser.add_argument('--dinov3_dtype', choices=['bf16', 'fp32'], default='bf16',
                        help='fp32 image encoder: sharper features, ~1.7 GB more VRAM')
    parser.add_argument('--no_optimize', dest='optimize', action='store_false',
                        help='turn off the faster sampling (torch.compile, batched guidance, cuDNN attention, TF32)')
    parser.add_argument('--device', default=None, help='torch device, e.g. cuda, cuda:1, cpu (default: auto)')
    parser.add_argument('--weights', default='speridlabs/idol', help='Hugging Face repo or local folder')
    # bookkeeping
    parser.add_argument('--skip-existing', action='store_true', help='skip images whose .ply is already in --out')
    parser.add_argument('--no-preview', dest='save_preview', action='store_false',
                        help='do not save <name>_input.png (the image the model saw)')
    parser.add_argument('--no-camera', dest='save_camera', action='store_false',
                        help='do not save <name>_camera.json (the predicted input camera)')
    parser.add_argument('--dry-run', action='store_true',
                        help='resolve settings, check encoder access and GPU, then exit without generating')
    parser.add_argument('-q', '--quiet', action='store_true', help='less output')
    return parser


def resolve(args, parser):
    """Merge preset with any explicit overrides and report what will actually run."""
    settings = dict(PRESETS[args.preset])
    for name in ('steps', 'num_gaussians', 'cfg_strength', 'rescale_t'):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
            settings.setdefault('_overridden', []).append(name)
    for name, value in settings.items():
        if not name.startswith('_'):
            setattr(args, name, value)
    args.overridden = settings.get('_overridden', [])
    if args.num_gaussians // GAUSSIAN_MULTIPLE < 1:
        parser.error('--num_gaussians is too small to place any anchor points')
    return args


def collect_images(path, out, skip_existing):
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.lower().endswith(IMAGE_EXTENSIONS))
        images = [os.path.join(path, n) for n in names]
    elif os.path.isfile(path):
        images = [path]
    else:
        raise SystemExit(f'error: no such file or folder: {path}')
    if skip_existing:
        kept = [p for p in images
                if not os.path.exists(os.path.join(out, f'{os.path.splitext(os.path.basename(p))[0]}.ply'))]
        skipped = len(images) - len(kept)
        if skipped:
            print(f'skipping {skipped} image(s) that already have a .ply in {out}')
        images = kept
    return images


def report(args, count):
    gpu = ''
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
            gpu = f'  gpu: {name} ({total:.0f} GB)'
    except Exception:
        pass
    print(f'\nIdol  |  {count} image(s) x {args.variants} variant(s)  |  preset={args.preset}'
          f'{" (overrode: " + ", ".join(args.overridden) + ")" if args.overridden else ""}')
    print(f'  steps={args.steps}  gaussians={args.num_gaussians:,}  cfg={args.cfg_strength}  '
          f'rescale_t={args.rescale_t}')
    print(f'  mode={args.mode}  crop_margin={args.crop_margin if args.crop_margin is not None else "from mode"}  '
          f'seed={args.seed}..{args.seed + args.variants - 1}  dinov3={args.dinov3_dtype}  '
          f'optimize={"on" if args.optimize else "off"}')
    if gpu:
        print(gpu)
    print(f'  out: {args.out}\n')


def main(argv=None):
    parser = build_parser()
    args = resolve(parser.parse_args(argv), parser)
    images = collect_images(args.image, args.out, args.skip_existing)
    if not images:
        print('nothing to do')
        return 0

    if args.dry_run:
        report(args, len(images))
        from idol.conditioning import EncoderAccessError, require_encoder_access
        from idol.pipeline import WEIGHT_FILES
        try:
            if os.path.isdir(args.weights):
                with open(os.path.join(args.weights, 'config.json')) as f:
                    encoders = json.load(f)['image_encoders']
            else:
                from huggingface_hub import hf_hub_download
                with open(hf_hub_download(args.weights, 'config.json')) as f:
                    encoders = json.load(f)['image_encoders']
            require_encoder_access([encoders['dinov3'], encoders['flux2_vae']])
        except EncoderAccessError as error:
            print(f'\n{error}\n', file=sys.stderr)
            return 2
        print('weights: ' + (args.weights if os.path.isdir(args.weights)
                             else f'{args.weights} ({", ".join(WEIGHT_FILES)})'))
        print('encoder access: OK (DINOv3 + FLUX.2 VAE)')
        print('dry run: nothing generated')
        return 0

    os.makedirs(args.out, exist_ok=True)
    import torch

    from idol.conditioning import EncoderAccessError
    from idol.pipeline import IdolPipeline

    report(args, len(images))
    t0 = time.time()
    try:
        pipeline = IdolPipeline.from_pretrained(args.weights, device=args.device,
                                                dinov3_dtype=args.dinov3_dtype, optimize=args.optimize)
    except EncoderAccessError as error:
        print(f'\n{error}\n', file=sys.stderr)
        return 2
    print(f'model loaded in {time.time() - t0:.1f}s on {pipeline.device}\n')

    failures = 0
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        for variant in range(args.variants):
            seed = args.seed + variant
            suffix = f'_seed{seed}' if args.variants > 1 else ''
            ply = os.path.join(args.out, f'{name}{suffix}.ply')
            t1 = time.time()
            try:
                output = pipeline(path, mode=args.mode, crop_margin=args.crop_margin, seed=seed,
                                  steps=args.steps, cfg_strength=args.cfg_strength, rescale_t=args.rescale_t,
                                  num_gaussians=args.num_gaussians, verbose=not args.quiet)
            except ValueError as error:
                print(f'{name}: {error}', file=sys.stderr)
                failures += 1
                continue
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'{name}{suffix}: out of memory with {args.num_gaussians:,} Gaussians, '
                      f'{args.steps} steps, dinov3={args.dinov3_dtype}. Try fewer --num_gaussians, '
                      f'--dinov3_dtype bf16, or --preset draft.', file=sys.stderr)
                failures += 1
                continue
            output.gaussian.save_ply(ply)
            if args.save_preview:
                output.preprocessed_image.save(os.path.join(args.out, f'{name}{suffix}_input.png'))
            if args.save_camera:
                with open(os.path.join(args.out, f'{name}{suffix}_camera.json'), 'w') as f:
                    json.dump(output.camera, f, indent=2)
            print(f'{name}{suffix}: {os.path.relpath(ply, args.out)}  '
                  f'({output.gaussian.xyz.shape[0]:,} gaussians, {time.time() - t1:.1f}s)')
    print(f'\n{len(images) * args.variants - failures} saved to {args.out} in {time.time() - t0:.1f}s')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
