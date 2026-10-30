#!/usr/bin/env python
"""idol-app: generate and view 3D Gaussian splats from the browser — no command line needed.

  idol-app                          # http://localhost:7860, writes to outputs/web
  idol-app --port 8080 --out shots/ --open
  idol-app --image photo.jpg        # start with an image already loaded

One page: drop in a picture, pick a preset or pull the quality levers, hit Generate, and the splat appears
in the 3D viewport next to the exact image the model saw. Everything runs locally; the only network traffic
is downloading model weights. The CLI (idol-gen / idol-view) does the same things and stays the way to batch
a folder — this app is the "just try it" path.
"""
import argparse
import io
import json
import os
import threading
import time

# numpy, PIL and viser are imported where they are used, so `idol-app --help` stays dependency-free.


def _blank_preview():
    import numpy as np

    return np.zeros((1, 1, 3), dtype=np.uint8)


class AppState:
    """What the page knows right now: the uploaded image, the objects generated, and a lock so two
    visitors cannot drive the GPU at the same time."""

    def __init__(self, pipeline, out_dir, ply_convention='y-up'):
        from idol_tools.ply import UP_AXIS

        self.pipeline = pipeline
        self.out_dir = out_dir
        self.ply_convention = ply_convention
        self.up = UP_AXIS.get(ply_convention, '+y')   # what the files we write consider up
        self.uploaded = None            # (stem, PIL.Image)
        self.lock = threading.Lock()
        self.objects = {}               # name -> {'path': str, 'handle': ..., 'checkbox': ..., 'count': int}
        self.last = None
        self.grid = None
        self.ground_height = None       # None = drop the floor onto the lowest point of what is on screen
        self.ground_visible = True

    def settings(self, controls):
        """Resolve the form into pipeline kwargs, mirroring idol-gen's preset/flag precedence.

        The preset dropdown does not live here: picking one pushes its values into the editable levers
        (see apply_preset), so this reads exactly what the page shows."""
        from idol_tools.generate import DENSITY

        density = controls['density'].value
        gaussians = DENSITY[density] if density in DENSITY else int(controls['gaussians'].value)
        crop_margin = float(controls['crop_margin'].value)
        return {
            'mode': controls['mode'].value,
            'seed': int(controls['seed'].value),
            'steps': int(controls['steps'].value),
            'cfg_strength': float(controls['cfg_strength'].value),
            'rescale_t': float(controls['rescale_t'].value),
            'num_gaussians': gaussians,
            'crop_margin': crop_margin if crop_margin > 0 else None,
        }


def apply_preset(controls, preset):
    """Push a preset's values into the editable levers (what the CLI does when nothing overrides them)."""
    from idol_tools.generate import PRESETS

    values = PRESETS[preset]
    controls['steps'].value = values['steps']
    controls['cfg_strength'].value = values['cfg_strength']
    controls['rescale_t'].value = values['rescale_t']
    controls['density'].value = next((rung for rung, count in _density_map().items()
                                      if count == values['num_gaussians']), 'custom')
    controls['gaussians'].value = values['num_gaussians']


def _density_map():
    from idol_tools.generate import DENSITY

    return DENSITY


AUTO_GROUND = -2.0      # slider floor, used as the "sit it on the object" sentinel


def add_object(server, state, name, ply_path, controls=None, visible_others=False):
    """Put a generated .ply into the scene (parsed from the file, so what you see is what you saved)."""
    import numpy as np

    from idol_tools import viewer

    obj = viewer.load_ply(ply_path)
    obj['name'] = name
    with server.atomic():
        if not visible_others:
            for other in state.objects.values():
                other['checkbox'].value = False
        handle = server.scene.add_gaussian_splats(f'/splats/{name}', centers=obj['centers'],
                                                  covariances=obj['covariances'], rgbs=obj['rgbs'],
                                                  opacities=obj['opacities'])
        checkbox = server.gui.add_checkbox(name, initial_value=True)
        state.objects[name] = {'path': ply_path, 'handle': handle, 'checkbox': checkbox, 'count': obj['count'],
                               'mins': np.min(obj['centers'], axis=0)}

        def toggle(_event, handle=handle, checkbox=checkbox):
            handle.visible = bool(checkbox.value)

        checkbox.on_update(toggle)
    state.last = name
    if controls and 'ground_reposition' in controls:
        controls['ground_reposition']()          # the floor follows whatever we just dropped in
    return obj


def build_ui(server, state):
    """Create the side panel and wire it up. Returns the control handles (handy for tests and theming)."""
    from PIL import Image

    from idol_tools.generate import DENSITY, PRESETS

    controls = {}
    status = {'text': 'Upload an image, then press Generate.'}

    def set_status(markdown):
        status['text'] = markdown
        controls['status'].content = markdown

    with server.gui.add_folder('Input'):
        controls['upload'] = server.gui.add_upload_button('Choose an image', mime_type='image/*')
        controls['preview'] = server.gui.add_image(_blank_preview(), label='image the model sees')
        controls['source'] = server.gui.add_markdown('_no image yet_')

    with server.gui.add_folder('Quality'):
        controls['preset'] = server.gui.add_dropdown('Preset', list(PRESETS), initial_value='standard')
        controls['density'] = server.gui.add_dropdown('Density', list(DENSITY) + ['custom'], initial_value='256k')
        controls['gaussians'] = server.gui.add_number('Exact Gaussians', 262_144, min=32_768, max=1_048_576, step=32)
        controls['steps'] = server.gui.add_number('Steps', 50, min=1, max=200, step=1)
        controls['cfg_strength'] = server.gui.add_number('CFG strength', 3.0, min=0.0, max=10.0, step=0.1)
        controls['rescale_t'] = server.gui.add_number('Time shift', 3.0, min=0.1, max=10.0, step=0.1)

    with server.gui.add_folder('Framing and seed'):
        controls['mode'] = server.gui.add_dropdown('Mode', ['synthetic', 'real'], initial_value='synthetic')
        controls['crop_margin'] = server.gui.add_number('Crop margin (0 = from mode)', 0.0, min=0.0, max=4.0, step=0.05)
        controls['seed'] = server.gui.add_number('Seed', 42, min=0, max=2**31 - 1, step=1)
        controls['randomize'] = server.gui.add_checkbox('Randomize seed', initial_value=False)
        controls['download'] = server.gui.add_checkbox('Download the .ply when done', initial_value=False)

    with server.gui.add_folder('Run'):
        controls['generate'] = server.gui.add_button('Generate', color='blue')
        controls['progress'] = server.gui.add_progress_bar(value=0.0, visible=False, animated=True)
        controls['status'] = server.gui.add_markdown(status['text'])
        controls['download_last'] = server.gui.add_button('Download last .ply')

    with server.gui.add_folder('View'):
        controls['up'] = server.gui.add_dropdown('Up axis', ['+y', '-y', '+z', '-z'], initial_value=state.up)
        controls['flip'] = server.gui.add_button('Flip upside down')
        controls['ground'] = server.gui.add_checkbox('Ground grid', initial_value=True)
        controls['ground_height'] = server.gui.add_slider('Ground height', min=AUTO_GROUND, max=1.5, step=0.01,
                                                          initial_value=AUTO_GROUND,
                                                          hint='the lowest slider value means: sit it on the '
                                                               'bottom of the object')

    def on_upload(event):
        import numpy as np

        upload = event.target.value
        name, content = (upload.name, upload.content) if upload else ('upload', b'')
        try:
            image = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception as error:                                  # noqa: BLE001 - any unparsable upload
            set_status(f'Could not read **{name}** as an image ({error}).')
            return
        stem = os.path.splitext(os.path.basename(name))[0] or 'upload'
        state.uploaded = (stem, image)
        controls['preview'].image = np.asarray(image)
        set_status(f'Ready: **{stem}** ({image.size[0]}x{image.size[1]}). Press Generate.')

    def on_generate(_event):
        import numpy as np

        if state.uploaded is None:
            set_status('Upload an image first.')
            return
        if not state.lock.acquire(blocking=False):
            set_status('Already generating — one at a time.')
            return
        settings = state.settings(controls)
        if controls['randomize'].value:
            import random

            settings['seed'] = random.randrange(2 ** 31)
            controls['seed'].value = settings['seed']
        stem, image = state.uploaded
        name = f'{stem}_s{settings["seed"]}'
        controls['progress'].visible = True
        set_status(f'Generating **{name}** ({settings["num_gaussians"]:,} Gaussians, {settings["steps"]} steps)…')
        started = time.time()
        try:
            output = state.pipeline(image, verbose=False, **settings)
            os.makedirs(state.out_dir, exist_ok=True)
            ply = os.path.join(state.out_dir, f'{name}.ply')
            from idol_tools import ply as ply_io

            ply_io.save_ply(output.gaussian, ply, state.ply_convention)
            output.preprocessed_image.save(os.path.join(state.out_dir, f'{name}_input.png'))
            with open(os.path.join(state.out_dir, f'{name}_camera.json'), 'w') as handle:
                json.dump(output.camera, handle, indent=2)
            controls['preview'].image = np.asarray(output.preprocessed_image)
            obj = add_object(server, state, name, ply, controls)
            if controls['download'].value:
                with open(ply, 'rb') as handle:
                    server.send_file_download(f'{name}.ply', handle.read())
            set_status(f'**{name}**: {obj["count"]:,} Gaussians in {time.time() - started:.1f}s -> `{ply}`')
        except Exception as error:                                    # noqa: BLE001 - report to the page, not the log
            set_status(f'`{type(error).__name__}` {error}')
            if os.environ.get('IDOL_DEBUG'):
                import traceback

                traceback.print_exc()                                 # the page stays clean; the terminal can talk
        finally:
            controls['progress'].visible = False
            state.lock.release()

    def on_download_last(_event):
        if not state.last:
            set_status('Nothing to download yet.')
            return
        with open(state.objects[state.last]['path'], 'rb') as handle:
            server.send_file_download(os.path.basename(state.objects[state.last]['path']), handle.read())

    def on_preset(event):
        apply_preset(controls, event.target.value)

    def on_density(event):
        rung = event.target.value
        if rung in _density_map():
            controls['gaussians'].value = _density_map()[rung]

    def set_up(axis):
        """Tell viser which way is up, and lay the ground grid as a floor under the objects."""
        from idol_tools import viewer

        state.up = axis
        server.scene.set_up_direction(axis)
        if state.grid is not None:
            try:
                state.grid.remove()
            except Exception:                                          # noqa: BLE001 - cosmetic only
                pass
        state.grid = viewer.add_ground(server, up=axis, height=state.ground_height,
                                       objects=list(state.objects.values()))
        if state.grid is not None:
            state.grid.visible = bool(state.ground_visible)

    def reposition_ground(_=None):
        set_up(state.up)

    def on_ground_toggle(event):
        state.ground_visible = bool(event.target.value)
        if state.grid is not None:
            state.grid.visible = state.ground_visible

    def on_ground_height(event):
        value = float(event.target.value)
        state.ground_height = None if abs(value - AUTO_GROUND) < 1e-9 else value
        set_up(state.up)

    def on_up(event):
        set_up(event.target.value)

    def on_flip(_event):
        controls['up'].value = {'+y': '-y', '-y': '+y', '+z': '-z', '-z': '+z'}[controls['up'].value]
        set_up(controls['up'].value)

    controls['upload'].on_upload(on_upload)
    controls['generate'].on_click(on_generate)
    controls['download_last'].on_click(on_download_last)
    controls['up'].on_update(on_up)
    controls['flip'].on_click(on_flip)
    controls['ground'].on_update(on_ground_toggle)
    controls['ground_height'].on_update(on_ground_height)
    controls['preset'].on_update(on_preset)
    controls['density'].on_update(on_density)
    controls['ground_reposition'] = reposition_ground

    set_up(state.up)
    return controls, set_status


def load_pipeline(args):
    from idol.conditioning import EncoderAccessError
    from idol.pipeline import IdolPipeline

    try:
        return IdolPipeline.from_pretrained(args.weights, device=args.device, dinov3_dtype=args.dinov3_dtype,
                                            optimize=args.optimize)
    except EncoderAccessError as error:
        raise SystemExit(f'\n{error}\n') from None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='idol-app', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--out', default=os.path.join('outputs', 'web'), help='where to write results')
    parser.add_argument('--host', default='127.0.0.1',
                        help='bind address; 0.0.0.0 to let others on your network in (default: localhost only)')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--image', default=None, help='start with this image already loaded')
    parser.add_argument('--device', default=None, help='torch device, e.g. cuda:1 (default: auto)')
    parser.add_argument('--weights', default='speridlabs/idol', help='Hugging Face repo or local folder')
    parser.add_argument('--dinov3_dtype', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--no_optimize', dest='optimize', action='store_false',
                        help='turn off the faster sampling path (torch.compile, batched guidance, cuDNN, TF32)')
    parser.add_argument('--open', action='store_true', help='open the browser when the server is up')
    parser.add_argument('--no-model', dest='load_weights', action='store_false',
                        help='start the page without loading the model (checks the UI and wiring only)')
    parser.add_argument('--ply_convention', choices=['y-up', 'upstream'], default='y-up',
                        help="'y-up' (default) writes objects the right way up for splat viewers; "
                             "'upstream' matches idol-sdk's save_ply, which comes out upside down")
    args = parser.parse_args(argv)

    try:
        import viser
    except ImportError:
        raise SystemExit('error: viser is not installed. Run scripts/setup.sh, or: pip install viser') from None

    pipeline = load_pipeline(args) if args.load_weights else None
    server = viser.ViserServer(host=args.host, port=args.port)
    state = AppState(pipeline, args.out, args.ply_convention)
    controls, set_status = build_ui(server, state)

    if pipeline is not None:
        set_status('Model ready. Upload an image, then press **Generate**.')
    else:
        set_status('Model not loaded (started with --no-model).')

    if args.image:
        import numpy as np
        from PIL import Image

        image = Image.open(args.image).convert('RGB')
        state.uploaded = (os.path.splitext(os.path.basename(args.image))[0], image)
        controls['preview'].image = np.asarray(image)
        set_status(f'Ready: **{state.uploaded[0]}**. Press Generate.')

    host = 'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host
    print(f'\nIdol app: http://{host}:{args.port}   (results -> {args.out})')
    if args.host == '0.0.0.0':
        print('  bound to 0.0.0.0: anyone on your network can generate on this GPU')
    print('  Ctrl-C to stop\n')
    if args.open:
        import webbrowser

        webbrowser.open(f'http://{host}:{args.port}')
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        server.stop()
    return 0


if __name__ == '__main__':
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from idol_tools._bootstrap import reexec_into_venv

    reexec_into_venv(__file__)
    raise SystemExit(main())
