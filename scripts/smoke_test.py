#!/usr/bin/env python
"""Smoke test for the Idol install that does NOT need the gated image encoders.

The DINOv3 + FLUX.2 VAE encoders are gated on Hugging Face, so this test swaps the encoder for a stub that
returns random tokens of the right shape. Everything else is the real thing: BiRefNet background removal,
cropping, the DiT sampler, the octree + Gaussian decoders, and the PLY writer.

  scripts/smoke_test.py [--steps 8] [--num_gaussians 32768] [--device cuda]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile


def _reexec_into_venv():
    """Run under the project venv no matter which python started us (deps live there, not in ~/.local)."""
    venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               '.venv', 'bin', 'python')
    if os.path.isfile(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
        os.environ.setdefault('PYTHONNOUSERSITE', '1')
        os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])


_reexec_into_venv()

import torch
from tensordict import TensorDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLE = os.path.join(ROOT, 'idol-sdk', 'demo', 'examples', 'food-donut.webp')
FAILURES = []


def check(name, condition, detail=''):
    print(f'  {"PASS" if condition else "FAIL"}  {name}{": " + detail if detail else ""}')
    if not condition:
        FAILURES.append(name)


class StubImageEncoder:
    """Stand-in for idol.conditioning.ImageEncoder: random tokens with the shapes the denoiser expects
    (feature1 = DINOv3 tokens [B, 4096+5, 1280], feature2 = FLUX.2 VAE tokens [B, 4096+5, 128])."""

    def __init__(self, dinov3_repo, flux2_vae_repo, dinov3_dtype='bf16', device='cuda'):
        self.device = torch.device(device)
        print(f'  (stub encoder in place of {dinov3_repo} + {flux2_vae_repo})')

    @torch.no_grad()
    def __call__(self, images):
        batch = images.shape[0]
        return TensorDict({
            'feature1': torch.randn(batch, 4101, 1280, device=self.device),
            'feature2': torch.randn(batch, 4101, 128, device=self.device),
        }, batch_size=batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--num_gaussians', type=int, default=32_768)
    parser.add_argument('--device', default=None)
    parser.add_argument('--eager', action='store_true',
                        help='disable the optimized sampling path (torch.compile) that the CLI uses by default')
    args = parser.parse_args()

    import idol.pipeline as pipeline_module
    from idol.pipeline import IdolPipeline

    # the two gated encoders are the only thing this test cannot download
    pipeline_module.ImageEncoder = StubImageEncoder
    pipeline_module.require_encoder_access = lambda repos: None

    print('\n== environment')
    check('torch sees a CUDA GPU', torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
    check('example image present', os.path.isfile(EXAMPLE), EXAMPLE)

    print('\n== pipeline (real weights, stubbed image encoder)')
    pipeline = IdolPipeline.from_pretrained('speridlabs/idol', device=args.device, optimize=not args.eager)
    check('optimize flag respected', pipeline.optimize == (not args.eager),
          'compiled/batched guidance' if pipeline.optimize else 'eager')

    print('\n== preprocessing (BiRefNet background removal + crop)')
    import numpy as np

    def object_span(image):
        """Width of the non-black region: how much of the frame the object fills."""
        mask = np.array(image).max(axis=2) > 8
        cols = np.nonzero(mask.any(axis=0))[0]
        return cols[-1] - cols[0] if len(cols) else 0

    prepared = pipeline.preprocess(EXAMPLE, mode='synthetic')
    check('preprocess returns 1024x1024 RGB', prepared.size == (1024, 1024) and prepared.mode == 'RGB',
          f'{prepared.size} {prepared.mode}')
    check('background removed (corners are black)',
          np.array(prepared)[[0, 0, -1, -1], [0, -1, 0, -1]].max() == 0)
    span_synthetic, span_real = object_span(prepared), object_span(pipeline.preprocess(EXAMPLE, mode='real'))
    check('mode real leaves more room than synthetic', span_real < span_synthetic,
          f'object spans {span_real}px (real) vs {span_synthetic}px (synthetic) of 1024')
    span_tight = object_span(pipeline.preprocess(EXAMPLE, crop_margin=1.0))
    check('crop_margin overrides the mode', span_tight > span_synthetic,
          f'crop_margin=1.0 -> {span_tight}px')
    check('alpha-less input gets a background removed', object_span(prepared) > 500)

    print('\n== generation + PLY')
    with tempfile.TemporaryDirectory() as out:
        output = pipeline(EXAMPLE, mode='synthetic', seed=42, steps=args.steps, cfg_strength=3.0,
                          rescale_t=3.0, num_gaussians=args.num_gaussians, verbose=False)
        ply = os.path.join(out, 'donut.ply')
        output.gaussian.save_ply(ply)
        count = output.gaussian.xyz.shape[0]
        check('gaussian count matches request', count == args.num_gaussians, f'{count:,}')
        check('xyz is finite', bool(torch.isfinite(output.gaussian.xyz).all()))
        check('scales are positive', bool((output.gaussian.scaling > 0).all()))
        check('rotations are finite', bool(torch.isfinite(output.gaussian.rotation).all()))
        check('opacity in (0, 1)', bool(((output.gaussian.opacity > 0) & (output.gaussian.opacity < 1)).all()))
        check('PLY written', os.path.getsize(ply) > 1_000_000, f'{os.path.getsize(ply) / 1e6:.1f} MB')

        from plyfile import PlyData
        parsed = PlyData.read(ply)
        attrs = {p.name for p in parsed['vertex'].properties}
        needed = {'x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                  'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'}
        check('PLY has the standard 3DGS attributes', needed <= attrs, f'{len(attrs)} properties')
        check('PLY vertex count matches', parsed['vertex'].count == count, f'{parsed["vertex"].count:,}')

        with open(os.path.join(out, 'donut_camera.json'), 'w') as f:
            json.dump(output.camera, f)
        check('camera has yaw/pitch/radius/fov', {'yaw', 'pitch', 'radius', 'fov'} <= set(output.camera),
              json.dumps(output.camera))

        print('\n== determinism')
        again = pipeline(EXAMPLE, mode='synthetic', seed=42, steps=args.steps, cfg_strength=3.0,
                         rescale_t=3.0, num_gaussians=args.num_gaussians, verbose=False)
        same = torch.allclose(again.gaussian.xyz, output.gaussian.xyz, atol=1e-4)
        check('same seed reproduces the same object', same,
              f'max |dx| = {(again.gaussian.xyz - output.gaussian.xyz).abs().max():.2e}')
        other = pipeline(EXAMPLE, mode='synthetic', seed=7, steps=args.steps, cfg_strength=3.0,
                         rescale_t=3.0, num_gaussians=args.num_gaussians, verbose=False)
        check('a different seed changes the object',
              not torch.allclose(other.gaussian.xyz, output.gaussian.xyz, atol=1e-4))

    print('\n== CLI')
    from generate import PRESETS, build_parser, resolve

    parser = build_parser()
    ns = resolve(parser.parse_args([EXAMPLE, '--preset', 'draft', '--steps', '11']), parser)
    check('preset sets gaussians', ns.num_gaussians == PRESETS['draft']['num_gaussians'],
          f'{ns.num_gaussians:,}')
    check('explicit flag beats preset', ns.steps == 11, f'steps={ns.steps}')
    check('preset leaves mode/crop defaults', ns.mode == 'synthetic' and ns.crop_margin is None)
    ns = resolve(parser.parse_args([EXAMPLE, '--preset', 'max', '--gaussians', '65536']), parser)
    check('--gaussians alias works', ns.num_gaussians == 65_536, f'{ns.num_gaussians:,}')

    ns = resolve(parser.parse_args([EXAMPLE, '--density', '256k']), parser)
    check('--density 256k = 262,144', ns.num_gaussians == 262_144, f'{ns.num_gaussians:,}')
    ns = resolve(parser.parse_args([EXAMPLE, '--preset', 'max', '--density', '32k']), parser)
    check('--density beats the preset', ns.num_gaussians == 32_768 and ns.steps == PRESETS['max']['steps'],
          f'{ns.num_gaussians:,} gaussians at {ns.steps} steps')
    ns = resolve(parser.parse_args([EXAMPLE, '--preset', 'draft', '--density', '1m']), parser)
    check('--density 1m = 1,048,576, preset steps kept',
          ns.num_gaussians == 1_048_576 and ns.steps == PRESETS['draft']['steps'],
          f'{ns.num_gaussians:,} at {ns.steps} steps')
    check('--density is recorded as an override', 'density=1m' in ns.overridden, str(ns.overridden))
    for text, want in (('1m', 1_048_576), ('1M', 1_048_576), ('256k', 262_144), ('0.5m', 524_288),
                       ('262,144', 262_144), ('32_768', 32_768),
                       ('48k', 49_152)):  # free-form shorthand is allowed, unlike --density's fixed rungs
        ns = resolve(parser.parse_args([EXAMPLE, '--gaussians', text]), parser)
        check(f'--gaussians {text} parses', ns.num_gaussians == want, f'{ns.num_gaussians:,}')
    for bad in ('33000', '100', '2000000', 'banana', '48.251k', '1m1m', ''):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args([EXAMPLE, '--num_gaussians', bad])
            check(f'rejects --num_gaussians {bad!r}', False, 'accepted an invalid value')
        except SystemExit:
            check(f'rejects --num_gaussians {bad!r}', True)
    for bad in ('48k', '2m', '256000'):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args([EXAMPLE, '--density', bad])
            check(f'rejects --density {bad!r}', False, 'accepted an invalid rung')
        except SystemExit:
            check(f'rejects --density {bad!r}', True)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args([EXAMPLE, '--density', '256k', '--num_gaussians', '65536'])
        check('rejects --density together with --num_gaussians', False, 'accepted both')
    except SystemExit:
        check('rejects --density together with --num_gaussians', True)

    print('\n== batch handling')
    from generate import collect_images

    with tempfile.TemporaryDirectory() as folder:
        from PIL import Image

        for name in ('a.png', 'b.jpg'):
            Image.open(EXAMPLE).save(os.path.join(folder, name))
        open(os.path.join(folder, 'notes.txt'), 'a').close()
        found = collect_images(folder, folder, skip_existing=False)
        check('folder mode keeps images only, sorted', [os.path.basename(p) for p in found] == ['a.png', 'b.jpg'],
              str([os.path.basename(p) for p in found]))
        open(os.path.join(folder, 'a.ply'), 'w').close()
        kept = collect_images(folder, folder, skip_existing=True)
        check('--skip-existing drops images with a .ply', [os.path.basename(p) for p in kept] == ['b.jpg'],
              str([os.path.basename(p) for p in kept]))
        check('single file mode', collect_images(EXAMPLE, folder, False) == [EXAMPLE])
        try:
            collect_images(os.path.join(folder, 'nope.png'), folder, False)
            check('missing input is rejected', False, 'accepted a missing path')
        except SystemExit:
            check('missing input is rejected', True)

    print('\n' + ('ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    raise SystemExit(main())
