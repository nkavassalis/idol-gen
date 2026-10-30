#!/usr/bin/env python
"""Smoke test for the Idol project.

  ./idol-test               full: real weights, BiRefNet, sampler, decoders, PLY, viewer, CLI
  ./idol-test --eager       same, on the un-optimized sampling path
  ./idol-test --cli-only    no GPU and no model weights needed: CLI, batch and viewer maths only
                            (this is what CI runs)

The two gated Hugging Face image encoders (DINOv3, FLUX.2 VAE) are replaced by a stub that returns random
tokens of the right shape, so the full test runs offline and without HF access. Everything else is real.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

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

    def __call__(self, images):
        batch = images.shape[0]
        return torch_stub_dict(batch, self.device)


def torch_stub_dict(batch, device):
    from tensordict import TensorDict

    return TensorDict({
        'feature1': torch.randn(batch, 4101, 1280, device=device),
        'feature2': torch.randn(batch, 4101, 128, device=device),
    }, batch_size=batch)


# --------------------------------------------------------------------------- GPU + real weights sections

def test_pipeline(pipeline, example):
    import numpy as np

    print('\n== preprocessing (BiRefNet background removal + crop)')

    def object_span(image):
        """Width of the non-black region: how much of the frame the object fills."""
        mask = np.array(image).max(axis=2) > 8
        cols = np.nonzero(mask.any(axis=0))[0]
        return cols[-1] - cols[0] if len(cols) else 0

    prepared = pipeline.preprocess(example, mode='synthetic')
    check('preprocess returns 1024x1024 RGB', prepared.size == (1024, 1024) and prepared.mode == 'RGB',
          f'{prepared.size} {prepared.mode}')
    check('background removed (corners are black)',
          np.array(prepared)[[0, 0, -1, -1], [0, -1, 0, -1]].max() == 0)
    span_synthetic, span_real = object_span(prepared), object_span(pipeline.preprocess(example, mode='real'))
    check('mode real leaves more room than synthetic', span_real < span_synthetic,
          f'object spans {span_real}px (real) vs {span_synthetic}px (synthetic) of 1024')
    span_tight = object_span(pipeline.preprocess(example, crop_margin=1.0))
    check('crop_margin overrides the mode', span_tight > span_synthetic, f'crop_margin=1.0 -> {span_tight}px')
    check('alpha-less input gets a background removed', object_span(prepared) > 500)

    print('\n== generation + PLY')
    with tempfile.TemporaryDirectory() as out:
        output = pipeline(example, mode='synthetic', seed=42, steps=8, cfg_strength=3.0,
                          rescale_t=3.0, num_gaussians=32_768, verbose=False)
        ply = os.path.join(out, 'donut.ply')
        output.gaussian.save_ply(ply)
        count = output.gaussian.xyz.shape[0]
        check('gaussian count matches request', count == 32_768, f'{count:,}')
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
        again = pipeline(example, mode='synthetic', seed=42, steps=8, cfg_strength=3.0,
                         rescale_t=3.0, num_gaussians=32_768, verbose=False)
        check('same seed reproduces the same object',
              torch.allclose(again.gaussian.xyz, output.gaussian.xyz, atol=1e-4),
              f'max |dx| = {(again.gaussian.xyz - output.gaussian.xyz).abs().max():.2e}')
        other = pipeline(example, mode='synthetic', seed=7, steps=8, cfg_strength=3.0,
                         rescale_t=3.0, num_gaussians=32_768, verbose=False)
        check('a different seed changes the object',
              not torch.allclose(other.gaussian.xyz, output.gaussian.xyz, atol=1e-4))

        print('\n== density ladder')
        for rung in ('32k', '256k'):
            from idol_tools.generate import DENSITY

            dense = pipeline(example, mode='synthetic', seed=42, steps=4, cfg_strength=3.0, rescale_t=3.0,
                             num_gaussians=DENSITY[rung], verbose=False)
            check(f'--density {rung} reaches the pipeline',
                  dense.gaussian.xyz.shape[0] == DENSITY[rung], f'{dense.gaussian.xyz.shape[0]:,}')


# --------------------------------------------------------------------------- viewer sections

_SENTINEL = object()


class FakeEvent:
    """Stands in for viser's GuiEvent."""

    def __init__(self, target):
        self.target = target
        self.client = None
        self.client_id = None


class FakeControl:
    """A viser GUI handle: a value, the props it was created with, and callbacks you can fire()."""

    def __init__(self, value=None, **props):
        self.value = value
        self.props = props
        self.handlers = []

    def on_update(self, func):
        self.handlers.append(func)
        return func

    on_click = on_update
    on_upload = on_update

    def fire(self, value=_SENTINEL):
        if value is not _SENTINEL:
            self.value = value
        for handler in list(self.handlers):
            handler(FakeEvent(self))
        return self


class FakeNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeViser:
    """Records everything the app and viewer do, so wiring is tested without a browser or a port."""

    class _Gui:
        def __init__(self, log):
            self.log = log
            self.controls = {}

        @contextlib.contextmanager
        def add_folder(self, label, **kwargs):
            self.log.append(('folder', label))
            yield

        def _add(self, kind, label, value, **props):
            self.log.append((kind, label))
            control = FakeControl(value=value, **props)
            if label:
                self.controls[label] = control
            return control

        def add_button(self, label, **kwargs):
            return self._add('button', label, None, **kwargs)

        def add_upload_button(self, label, **kwargs):
            return self._add('upload', label, None, **kwargs)

        def add_checkbox(self, label, initial_value, **kwargs):
            return self._add('checkbox', label, initial_value, **kwargs)

        def add_number(self, label, initial_value, **kwargs):
            return self._add('number', label, initial_value, **kwargs)

        def add_slider(self, label, min, max, step, initial_value, **kwargs):
            return self._add('slider', label, initial_value, min=min, max=max, step=step, **kwargs)

        def add_dropdown(self, label, options, initial_value=None, **kwargs):
            return self._add('dropdown', label, initial_value, options=list(options), **kwargs)

        def add_progress_bar(self, value, **kwargs):
            return self._add('progress', None, value, **kwargs)

        def add_image(self, image, **kwargs):
            return self._add('image', kwargs.pop('label', None), image, **kwargs)

        def add_markdown(self, content, **kwargs):
            return self._add('markdown', None, content, **kwargs)

    class _Scene:
        def __init__(self, log):
            self.log = log

        def set_up_direction(self, direction):
            self.log.append(('up', direction))

        def add_grid(self, name, **kwargs):
            self.log.append(('grid', kwargs.get('plane')))

        def add_gaussian_splats(self, name, **kwargs):
            self.log.append(('splats', name, tuple(kwargs['centers'].shape), tuple(kwargs['covariances'].shape),
                             tuple(kwargs['rgbs'].shape), tuple(kwargs['opacities'].shape)))
            return FakeNode(visible=kwargs.get('visible', True), scale=1.0)

    class ViserServer:
        def __init__(self, *args, **kwargs):
            self.log = []
            self.gui = FakeViser._Gui(self.log)
            self.scene = FakeViser._Scene(self.log)

        @contextlib.contextmanager
        def atomic(self):
            yield

        def send_file_download(self, filename, content, **kwargs):
            self.log.append(('download', filename, len(content)))

        def stop(self):
            self.log.append(('stop',))


def test_viewer():
    import numpy as np

    from idol_tools import viewer

    print('\n== viewer maths (quaternions, covariances, colours)')
    identity = viewer.rotation_matrices([[1, 0, 0, 0]])
    check('identity quaternion -> identity matrix', np.allclose(identity, np.eye(3), atol=1e-6))
    q90x = [[np.sqrt(0.5), np.sqrt(0.5), 0, 0]]           # +90 degrees about x
    rotated = viewer.rotation_matrices(q90x) @ np.array([0.0, 1.0, 0.0])
    check('90 deg about x maps +y to +z', np.allclose(rotated, [0, 0, 1], atol=1e-6), str(rotated.round(6)))
    check('rotation matrices are orthonormal',
          np.allclose(viewer.rotation_matrices(np.random.RandomState(0).randn(64, 4))
                      @ np.transpose(viewer.rotation_matrices(np.random.RandomState(0).randn(64, 4)),
                                     (0, 2, 1)), np.eye(3), atol=1e-4))

    scaling = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    cov = viewer.covariances_from(scaling, [[1, 0, 0, 0]])
    check('covariance is diagonal for an unrotated Gaussian',
          np.allclose(cov[0], np.diag([0.01, 0.04, 0.09]), atol=1e-6), str(np.diag(cov[0]).round(6)))
    cov_r = viewer.covariances_from(scaling, q90x)
    check('covariance is symmetric', np.allclose(cov_r, np.transpose(cov_r, (0, 2, 1)), atol=1e-7))
    eigen = np.sort(np.linalg.eigvalsh(cov_r)[0])
    check('rotating preserves the eigenvalues (scales squared)',
          np.allclose(eigen, np.sort([0.01, 0.04, 0.09]), atol=1e-6), str(eigen.round(6)))
    check('covariance is positive semi-definite', bool((np.linalg.eigvalsh(cov_r) >= -1e-9).all()))

    rgb = viewer.sh_dc_to_rgb([[0, 0, 0], [-10, 0, 0], [10, 0, 0]])
    check('SH DC 0 -> mid grey', np.allclose(rgb[0], 0.5))
    check('extreme SH clips into [0, 1]', rgb[1].min() >= 0.0 and rgb[2].max() <= 1.0, str(rgb[[1, 2]].round(3)))

    print('\n== viewer .ply round trip (SDK Gaussian -> PLY -> viser arrays)')
    from idol.gaussian import Gaussian

    generator = torch.Generator().manual_seed(0)
    gaussian = Gaussian(raw_xyz=torch.rand(2048, 3, generator=generator) - 0.5,
                        features_dc=torch.randn(2048, 1, 3, generator=generator) * 0.5,
                        raw_scaling=torch.randn(2048, 3, generator=generator) - 2,
                        raw_rotation=torch.randn(2048, 4, generator=generator),
                        raw_opacity=torch.randn(2048, 1, generator=generator),
                        scaling_bias=0.004, opacity_bias=0.1, minimum_kernel_size=0.0009)
    with tempfile.TemporaryDirectory() as out:
        ply = os.path.join(out, 'synthetic.ply')
        gaussian.save_ply(ply)
        obj = viewer.load_ply(ply)
        check('round trip keeps the Gaussian count', obj['count'] == 2048, f'{obj["count"]:,}')
        expected_rgb = viewer.sh_dc_to_rgb(gaussian.features_dc[:, 0].numpy())
        check('colours match the SH-DC formula', np.allclose(obj['rgbs'], expected_rgb, atol=1e-6))
        check('opacities are probabilities',
              obj['opacities'].shape == (2048, 1) and float(obj['opacities'].min()) > 0
              and float(obj['opacities'].max()) < 1)
        scales = gaussian.scaling.numpy()
        recovered = np.sort(np.linalg.eigvalsh(obj['covariances']), axis=1)
        check('covariance eigenvalues recover the stored scales',
              np.allclose(recovered, np.sort(scales ** 2, axis=1), atol=1e-6),
              f'max err {np.abs(recovered - np.sort(scales ** 2, axis=1)).max():.2e}')
        check('PLY is y-up: save_ply rotates the SDK z-up object for viewers',
              np.allclose(obj['centers'], np.stack([gaussian.xyz.numpy()[:, 0],
                                                    -gaussian.xyz.numpy()[:, 2],
                                                    gaussian.xyz.numpy()[:, 1]], axis=-1), atol=1e-5),
              'x, y, z -> x, -z, y')
        check('load_ply reports the object name and extent',
              obj['name'] == 'synthetic' and len(obj['extent']) == 3, str(obj['extent']))

        print('\n== viewer input handling')
        try:
            viewer.load_ply(os.path.join(out, 'missing.ply'))
            check('missing file is reported', False, 'no error raised')
        except ValueError as error:
            check('missing file is reported', 'no such file' in str(error), str(error))
        junk = os.path.join(out, 'junk.ply')
        open(junk, 'w').write('this is not a ply')
        try:
            viewer.load_ply(junk)
            check('garbage file is reported', False, 'no error raised')
        except ValueError as error:
            check('garbage file is reported', 'not a readable' in str(error), str(error)[:60])
        from plyfile import PlyData, PlyElement

        sparse = np.empty(4, dtype=[(n, 'f4') for n in ('x', 'y', 'z')])
        sparse_path = os.path.join(out, 'sparse.ply')
        PlyData([PlyElement.describe(sparse, 'vertex')]).write(sparse_path)
        try:
            viewer.load_ply(sparse_path)
            check('non-3DGS ply is reported', False, 'no error raised')
        except ValueError as error:
            check('non-3DGS ply is reported', 'missing' in str(error), str(error)[:70])

        folder = os.path.join(out, 'folder')
        os.makedirs(folder)
        for name in ('a.ply', 'b.ply', 'c.txt'):
            open(os.path.join(folder, name), 'w').close()
        import contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            found = viewer.collect([folder, ply])
        check('collect expands folders and files, sorted',
              [os.path.basename(p) for p in found] == ['a.ply', 'b.ply', 'synthetic.ply'],
              str([os.path.basename(p) for p in found]))
        limited = viewer.collect([folder], limit=1)
        check('collect --limit caps the list', len(limited) == 1, str(limited))
        try:
            viewer.collect(['/definitely/not/here'])
            check('collect rejects missing paths', False, 'no error raised')
        except SystemExit:
            check('collect rejects missing paths', True)

    print('\n== viewer scene wiring (fake viser, no port opened)')
    obj = {'name': 'thing', 'count': 10, 'centers': np.zeros((10, 3), np.float32),
           'covariances': np.tile(np.eye(3, dtype=np.float32), (10, 1, 1)),
           'rgbs': np.zeros((10, 3), np.float32), 'opacities': np.ones((10, 1), np.float32),
           'extent': [1, 1, 1], 'path': 'thing.ply'}

    recorded = []

    class RecordingServer:
        def __init__(self):
            self.gui = FakeViser._Gui(recorded)
            self.scene = FakeViser._Scene(recorded)

    handles = viewer.build_scene(RecordingServer(), [obj, dict(obj, name='other')], up='+y')
    check('up axis +y is set for the .ply files Idol writes', ('up', '+y') in recorded, str(recorded[:2]))
    check('grid lies in the xy plane for y-up', ('grid', 'xy') in recorded)
    check('one splat node per object', sum(1 for e in recorded if e[0] == 'splats') == 2,
          str([e[1] for e in recorded if e[0] == 'splats']))
    check('handles returned for every object', set(handles) == {'thing', 'other'}, str(sorted(handles)))
    first = [e for e in recorded if e[0] == 'splats'][0]
    check('arrays have the shapes viser wants',
          first[2:] == ((10, 3), (10, 3, 3), (10, 3), (10, 1)), str(first[2:]))
    check('a checkbox per object for toggling', sum(1 for e in recorded if e[0] == 'checkbox') == 2)
    check('everything starts visible', all(h.visible for h in handles.values()))

    recorded.clear()
    handles2 = viewer.build_scene(RecordingServer(), [obj], up='+z')
    check('--up z switches the axis and puts the grid in xz',
          ('up', '+z') in recorded and ('grid', 'xz') in recorded, str(recorded[:2]))
    check('one object still works', list(handles2) == ['thing'])

    try:
        viewer.serve([])
        check('serving with nothing to view is refused', False, 'no error raised')
    except SystemExit as error:
        check('serving with nothing to view is refused', 'nothing to view' in str(error), str(error))


# --------------------------------------------------------------------------- web app section

def test_app():
    import types

    import numpy as np
    from idol.gaussian import Gaussian
    from PIL import Image

    from idol_tools import app as app_module
    from idol_tools.generate import DENSITY, PRESETS

    print('\n== web app: the page and its controls')
    generator = torch.Generator().manual_seed(5)
    synthetic = Gaussian(raw_xyz=torch.rand(128, 3, generator=generator) - 0.5,
                         features_dc=torch.randn(128, 1, 3, generator=generator) * 0.4,
                         raw_scaling=torch.randn(128, 3, generator=generator) - 3,
                         raw_rotation=torch.randn(128, 4, generator=generator),
                         raw_opacity=torch.randn(128, 1, generator=generator),
                         scaling_bias=0.004, opacity_bias=0.1, minimum_kernel_size=0.0009)

    calls = []

    class FakeOutput:
        def __init__(self):
            self.gaussian = synthetic
            self.preprocessed_image = Image.new('RGB', (16, 16), (9, 9, 9))
            self.camera = {'yaw': 0.1, 'pitch': 0.2, 'radius': 1.0, 'fov': 40.0}

    class FakePipeline:
        def __init__(self, error=None):
            self.error = error

        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            if self.error:
                raise self.error
            return FakeOutput()

    with tempfile.TemporaryDirectory() as out:
        server = FakeViser.ViserServer()
        state = app_module.AppState(FakePipeline(), out)
        controls, set_status = app_module.build_ui(server, state)

        expected = {'upload', 'preview', 'source', 'preset', 'density', 'gaussians', 'steps', 'cfg_strength',
                    'rescale_t', 'mode', 'crop_margin', 'seed', 'randomize', 'download', 'generate',
                    'progress', 'status', 'download_last'}
        check('every control the page needs exists', expected <= set(controls), str(sorted(expected - set(controls))))
        check('the page is y-up with a ground grid', ('up', '+y') in server.log and ('grid', 'xy') in server.log)
        check('preset dropdown offers the CLI presets',
              controls['preset'].props['options'] == list(PRESETS), str(controls['preset'].props['options']))
        check('density dropdown offers the rungs plus custom',
              controls['density'].props['options'] == list(DENSITY) + ['custom'],
              str(controls['density'].props['options']))
        check('Gaussians is bounded to the valid range in steps of 32',
              (controls['gaussians'].props['min'], controls['gaussians'].props['max'],
               controls['gaussians'].props['step']) == (32_768, 1_048_576, 32))
        check('steps is bounded to something sane',
              (controls['steps'].props['min'], controls['steps'].props['max']) == (1, 200))
        check('mode matches the CLI', controls['mode'].props['options'] == ['synthetic', 'real'])

        print('\n== web app: levers behave like the CLI')
        controls['preset'].fire('draft')
        check('picking a preset pushes its values into the levers',
              controls['steps'].value == PRESETS['draft']['steps']
              and controls['gaussians'].value == PRESETS['draft']['num_gaussians']
              and controls['density'].value == '64k',
              f"steps={controls['steps'].value} gaussians={controls['gaussians'].value}")
        controls['density'].fire('1m')
        check('choosing a density sets the exact count', controls['gaussians'].value == 1_048_576,
              f"{controls['gaussians'].value:,}")
        settings = state.settings(controls)
        check('settings() resolves a density rung', settings['num_gaussians'] == 1_048_576,
              f"{settings['num_gaussians']:,}")
        controls['density'].fire('custom')
        controls['gaussians'].value = 100_000
        check('custom density uses the typed count',
              state.settings(controls)['num_gaussians'] == 100_000, str(state.settings(controls)['num_gaussians']))
        controls['crop_margin'].value = 0.0
        check('crop margin 0 means "from mode"', state.settings(controls)['crop_margin'] is None)
        controls['crop_margin'].value = 1.5
        controls['mode'].fire('real')
        settings = state.settings(controls)
        check('crop margin and mode pass through',
              settings['crop_margin'] == 1.5 and settings['mode'] == 'real', str(settings))

        print('\n== web app: upload')
        image = Image.new('RGB', (24, 16), (200, 30, 30))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        controls['upload'].fire(types.SimpleNamespace(name='donut.png', content=buffer.getvalue()))
        preview_shape = np.asarray(controls['preview'].image).shape[:2]
        check('an upload is remembered and previewed',
              state.uploaded is not None and state.uploaded[0] == 'donut' and preview_shape == (16, 24),
              str(state.uploaded and state.uploaded[0]))
        check('the page says it is ready', 'Ready' in controls['status'].content, controls['status'].content)
        controls['upload'].fire(types.SimpleNamespace(name='broken.png', content=b'not an image'))
        check('an unparsable upload is reported on the page', 'Could not read' in controls['status'].content,
              controls['status'].content)
        controls['upload'].fire(types.SimpleNamespace(name='donut.png', content=buffer.getvalue()))

        print('\n== web app: generate')
        empty = FakeViser.ViserServer()
        empty_state = app_module.AppState(FakePipeline(), out)
        empty_controls, _ = app_module.build_ui(empty, empty_state)
        empty_controls['generate'].fire()
        check('generate without an upload asks for one instead of crashing',
              'Upload an image first' in empty_controls['status'].content, empty_controls['status'].content)
        check('and the model was never called', not calls, str(calls))

        controls['preset'].fire('standard')
        controls['density'].fire('32k')
        controls['randomize'].value = False
        controls['generate'].fire()
        ply = os.path.join(out, 'donut_s42.ply')
        check('the .ply, preview and camera are written to disk',
              os.path.exists(ply) and os.path.exists(os.path.join(out, 'donut_s42_input.png'))
              and os.path.exists(os.path.join(out, 'donut_s42_camera.json')))
        check('the pipeline got the resolved settings',
              calls and calls[-1]['num_gaussians'] == 32_768 and calls[-1]['steps'] == PRESETS['standard']['steps']
              and calls[-1]['seed'] == 42 and calls[-1]['verbose'] is False, str(calls[-1]))
        check('the new object lands in the scene',
              any(entry[0] == 'splats' and entry[1].endswith('/donut_s42') for entry in server.log),
              str([e[1] for e in server.log if e[0] == 'splats'][-2:]))
        check('the page reports what it made',
              '128 Gaussians' in controls['status'].content and 'donut_s42' in controls['status'].content,
              controls['status'].content)
        check('the progress bar is hidden again', controls['progress'].visible is False)
        check('the lock is released', not state.lock.locked())
        check('no download unless asked', not any(e[0] == 'download' for e in server.log))

        controls['download'].value = True
        controls['randomize'].value = True
        controls['generate'].fire()
        check('the .ply is offered as a browser download',
              any(e[0] == 'download' and e[1].endswith('.ply') and e[2] > 1000 for e in server.log),
              str([e for e in server.log if e[0] == 'download'][-1:]))
        check('randomize picks a new seed', calls[-1]['seed'] != 42 and controls['seed'].value == calls[-1]['seed'],
              str(calls[-1]['seed']))
        check('older objects are hidden so results do not overlap',
              sum(1 for obj in state.objects.values() if obj['checkbox'].value) == 1,
              str({name: bool(obj['checkbox'].value) for name, obj in state.objects.items()}))

        print('\n== web app: failure handling')
        busy_state = app_module.AppState(FakePipeline(), out)
        busy_controls, _ = app_module.build_ui(FakeViser.ViserServer(), busy_state)
        busy_controls['upload'].fire(types.SimpleNamespace(name='x.png', content=buffer.getvalue()))
        busy_state.lock.acquire()
        busy_controls['generate'].fire()
        busy_state.lock.release()
        check('a second generation while busy is refused politely',
              'Already generating' in busy_controls['status'].content, busy_controls['status'].content)

        boom = app_module.AppState(FakePipeline(RuntimeError('CUDA out of memory')), out)
        boom_controls, _ = app_module.build_ui(FakeViser.ViserServer(), boom)
        boom_controls['upload'].fire(types.SimpleNamespace(name='x.png', content=buffer.getvalue()))
        boom_controls['generate'].fire()
        check('a model failure is shown on the page, not a stack trace',
              'RuntimeError' in boom_controls['status'].content
              and 'out of memory' in boom_controls['status'].content
              and 'Traceback' not in boom_controls['status'].content, boom_controls['status'].content)
        check('the page recovers: the lock is free and progress is hidden',
              not boom.lock.locked() and boom_controls['progress'].visible is False)

        idle = app_module.AppState(FakePipeline(), out)
        idle_controls, _ = app_module.build_ui(FakeViser.ViserServer(), idle)
        idle_controls['download_last'].fire()
        check('downloading before generating is handled', 'Nothing to download' in idle_controls['status'].content,
              idle_controls['status'].content)


# --------------------------------------------------------------------------- CLI sections

def test_cli(example):
    from idol_tools.generate import DENSITY, PRESETS, build_parser, collect_images, resolve

    print('\n== CLI presets and levers')
    parser = build_parser()
    ns = resolve(parser.parse_args([example, '--preset', 'draft', '--steps', '11']), parser)
    check('preset sets gaussians', ns.num_gaussians == PRESETS['draft']['num_gaussians'],
          f'{ns.num_gaussians:,}')
    check('explicit flag beats preset', ns.steps == 11, f'steps={ns.steps}')
    check('preset leaves mode/crop defaults', ns.mode == 'synthetic' and ns.crop_margin is None)
    ns = resolve(parser.parse_args([example, '--preset', 'max', '--gaussians', '65536']), parser)
    check('--gaussians alias works', ns.num_gaussians == 65_536, f'{ns.num_gaussians:,}')

    print('\n== CLI density ladder')
    ns = resolve(parser.parse_args([example, '--density', '256k']), parser)
    check('--density 256k = 262,144', ns.num_gaussians == 262_144, f'{ns.num_gaussians:,}')
    ns = resolve(parser.parse_args([example, '--preset', 'max', '--density', '32k']), parser)
    check('--density beats the preset', ns.num_gaussians == 32_768 and ns.steps == PRESETS['max']['steps'],
          f'{ns.num_gaussians:,} gaussians at {ns.steps} steps')
    ns = resolve(parser.parse_args([example, '--preset', 'draft', '--density', '1m']), parser)
    check('--density 1m = 1,048,576, preset steps kept',
          ns.num_gaussians == 1_048_576 and ns.steps == PRESETS['draft']['steps'],
          f'{ns.num_gaussians:,} at {ns.steps} steps')
    check('--density is recorded as an override', 'density=1m' in ns.overridden, str(ns.overridden))
    check('every rung is a valid multiple of 32 in range',
          all(32_768 <= v <= 1_048_576 and v % 32 == 0 for v in DENSITY.values()), str(sorted(DENSITY)))
    for text, want in (('1m', 1_048_576), ('1M', 1_048_576), ('256k', 262_144), ('0.5m', 524_288),
                       ('262,144', 262_144), ('32_768', 32_768), ('48k', 49_152)):
        ns = resolve(parser.parse_args([example, '--gaussians', text]), parser)
        check(f'--gaussians {text} parses', ns.num_gaussians == want, f'{ns.num_gaussians:,}')
    for bad in ('33000', '100', '2000000', 'banana', '48.251k', '1m1m', ''):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args([example, '--num_gaussians', bad])
            check(f'rejects --num_gaussians {bad!r}', False, 'accepted an invalid value')
        except SystemExit:
            check(f'rejects --num_gaussians {bad!r}', True)
    for bad in ('48k', '2m', '256000'):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args([example, '--density', bad])
            check(f'rejects --density {bad!r}', False, 'accepted an invalid rung')
        except SystemExit:
            check(f'rejects --density {bad!r}', True)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args([example, '--density', '256k', '--num_gaussians', '65536'])
        check('rejects --density together with --num_gaussians', False, 'accepted both')
    except SystemExit:
        check('rejects --density together with --num_gaussians', True)

    print('\n== CLI batch handling')
    with tempfile.TemporaryDirectory() as folder:
        from PIL import Image

        for name in ('a.png', 'b.jpg'):
            Image.open(example).save(os.path.join(folder, name))
        open(os.path.join(folder, 'notes.txt'), 'a').close()
        found = collect_images(folder, folder, skip_existing=False)
        check('folder mode keeps images only, sorted', [os.path.basename(p) for p in found] == ['a.png', 'b.jpg'],
              str([os.path.basename(p) for p in found]))
        open(os.path.join(folder, 'a.ply'), 'w').close()
        with contextlib.redirect_stdout(io.StringIO()):
            kept = collect_images(folder, folder, skip_existing=True)
        check('--skip-existing drops images with a .ply', [os.path.basename(p) for p in kept] == ['b.jpg'],
              str([os.path.basename(p) for p in kept]))
        check('single file mode', collect_images(example, folder, False) == [example])
        try:
            collect_images(os.path.join(folder, 'nope.png'), folder, False)
            check('missing input is rejected', False, 'accepted a missing path')
        except SystemExit:
            check('missing input is rejected', True)

    print('\n== CLI viewer flags')
    ns = parser.parse_args([example, '--serve', '--port', '9123', '--host', '0.0.0.0', '--open'])
    check('--serve/--port/--host/--open parse',
          ns.serve and ns.port == 9123 and ns.host == '0.0.0.0' and ns.open is True)
    ns = parser.parse_args([example])
    check('serving is off by default', not ns.serve and ns.port == 8080 and ns.host == '127.0.0.1')

    print('\n== CLI --serve wiring (fake pipeline, fake viewer)')
    from idol.gaussian import Gaussian
    from PIL import Image

    from idol_tools import generate as gen
    from idol_tools import viewer as viewer_module

    generator = torch.Generator().manual_seed(3)
    fake_gaussian = Gaussian(raw_xyz=torch.rand(64, 3, generator=generator) - 0.5,
                             features_dc=torch.randn(64, 1, 3, generator=generator) * 0.4,
                             raw_scaling=torch.randn(64, 3, generator=generator) - 3,
                             raw_rotation=torch.randn(64, 4, generator=generator),
                             raw_opacity=torch.randn(64, 1, generator=generator),
                             scaling_bias=0.004, opacity_bias=0.1, minimum_kernel_size=0.0009)

    class FakeOutput:
        def __init__(self):
            self.gaussian = fake_gaussian
            self.preprocessed_image = Image.new('RGB', (8, 8))
            self.camera = {'yaw': 0.1, 'pitch': 0.2, 'radius': 1.0, 'fov': 40.0}

    class FakePipeline:
        optimize = False
        device = torch.device('cpu')

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, image, **kwargs):
            return FakeOutput()

    import idol.pipeline

    served = {}
    real_pipeline, real_serve = idol.pipeline.IdolPipeline, viewer_module.serve
    idol.pipeline.IdolPipeline = FakePipeline
    viewer_module.serve = lambda objects, **kwargs: served.update(objects=objects, **kwargs) or 0
    try:
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                code = gen.main([example, '--out', out, '--steps', '4', '--serve', '--port', '9123',
                                 '--host', '0.0.0.0', '--open', '-q'])
            ply = os.path.join(out, f'{os.path.splitext(os.path.basename(example))[0]}.ply')
            check('--serve returns the viewer exit code', code == 0, str(code))
            check('--serve passes the generated .ply to the viewer',
                  [os.path.basename(o['path']) for o in served.get('objects', [])] == [os.path.basename(ply)],
                  str([os.path.basename(o['path']) for o in served.get('objects', [])]))
            check('served objects are already parsed', bool(served.get('objects')) and
                  served['objects'][0]['count'] == 64, str(served.get('objects', [{}])[0].get('count')))
            check('--serve forwards host and port',
                  served.get('host') == '0.0.0.0' and served.get('port') == 9123,
                  f"{served.get('host')}:{served.get('port')}")
            check('--serve opens the browser when asked', served.get('open_browser') is True)
            check('preview and camera json are written alongside',
                  os.path.exists(os.path.join(out, 'food-donut_input.png'))
                  and os.path.exists(os.path.join(out, 'food-donut_camera.json')))
    finally:
        idol.pipeline.IdolPipeline = real_pipeline
        viewer_module.serve = real_serve


def main():
    global torch
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cli-only', action='store_true', help='skip GPU/weights sections (what CI runs)')
    parser.add_argument('--eager', action='store_true',
                        help='disable the optimized sampling path (torch.compile) that the CLI uses by default')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    example = EXAMPLE if os.path.isfile(EXAMPLE) else None

    if args.cli_only:
        print('\n== cli-only mode (no GPU, no model weights)')
        if example is None:
            from PIL import Image

            os.makedirs(os.path.dirname(EXAMPLE), exist_ok=True)
            Image.new('RGB', (256, 256), (120, 90, 40)).save(EXAMPLE)
            print(f'  (created a placeholder example at {EXAMPLE})')
        test_cli(example)
        test_viewer()
        test_app()
    else:
        import idol.pipeline as pipeline_module
        from idol.pipeline import IdolPipeline

        # the two gated encoders are the only thing this test cannot download
        pipeline_module.ImageEncoder = StubImageEncoder
        pipeline_module.require_encoder_access = lambda repos: None

        print('\n== environment')
        check('torch sees a CUDA GPU', torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
        check('example image present', example is not None, str(example))
        if not torch.cuda.is_available() or example is None:
            print('\n' + (f'{len(FAILURES)} FAILED: {FAILURES}' if FAILURES else 'cannot continue'))
            return 1

        print('\n== pipeline (real weights, stubbed image encoder)')
        pipeline = IdolPipeline.from_pretrained('speridlabs/idol', device=args.device, optimize=not args.eager)
        check('optimize flag respected', pipeline.optimize == (not args.eager),
              'compiled/batched guidance' if pipeline.optimize else 'eager')
        test_pipeline(pipeline, example)
        test_cli(example)
        test_viewer()
        test_app()

    print('\n' + ('ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.path.insert(0, ROOT)
    from idol_tools._bootstrap import reexec_into_venv

    reexec_into_venv(__file__)
    raise SystemExit(main())
