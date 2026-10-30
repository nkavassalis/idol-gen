#!/usr/bin/env python
"""View 3D Gaussian splat .ply files in the browser, locally, via viser.

  idol-view outputs/                      # every .ply in the folder, checkboxes to switch between them
  idol-view outputs/donut.ply --port 9000
  python scripts/viewer.py outputs/ --up z

Nothing is uploaded: viser runs a local web server and the browser does the rendering. The .ply files
Idol writes are y-up (the convention splat viewers use), so the default up axis is +y; pass --up z for
files that keep the SDK's native z-up.
"""
import argparse
import os
import sys

SH_C0 = 0.28209479177387814          # degree-0 spherical harmonic constant: rgb = 0.5 + SH_C0 * f_dc
PLY_REQUIRED = ('x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')


def sh_dc_to_rgb(features_dc):
    """[N, 3] degree-0 spherical harmonics -> [N, 3] sRGB in [0, 1]."""
    import numpy as np

    return np.clip(0.5 + SH_C0 * np.asarray(features_dc, dtype=np.float32), 0.0, 1.0)


def rotation_matrices(quaternions):
    """[N, 4] unit quaternions in (w, x, y, z) -> [N, 3, 3] rotation matrices."""
    import numpy as np

    q = np.asarray(quaternions, dtype=np.float32)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if (norms == 0).any():
        raise ValueError('zero-length quaternion')
    w, x, y, z = (q / np.clip(norms, 1e-12, None)).T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)


def covariances_from(scaling, quaternions):
    """3DGS parameters -> the second moments viser wants: Sigma = R diag(s^2) R^T, (N, 3, 3).

    ``scaling`` holds standard deviations (the PLY stores log of them), ``quaternions`` are (w, x, y, z)."""
    import numpy as np

    scaling = np.asarray(scaling, dtype=np.float32)
    rot = rotation_matrices(quaternions)
    scaled = rot * scaling[:, None, :]                      # R @ diag(s), columns scaled
    return np.matmul(scaled, np.transpose(scaled, (0, 2, 1)))


def load_ply(path):
    """Read a standard 3DGS .ply into the arrays viser renders. Raises ValueError if it isn't one."""
    import numpy as np
    from plyfile import PlyData

    if not os.path.isfile(path):
        raise ValueError(f'no such file: {path}')
    try:
        from plyfile import PlyData

        vertex = PlyData.read(path)['vertex'].data
    except Exception as error:            # not a .ply at all, truncated, or no vertex element
        raise ValueError(f'{path} is not a readable 3D Gaussian splat .ply '
                         f'({type(error).__name__}: {error})') from error
    names = set(vertex.dtype.names or ())
    missing = [name for name in PLY_REQUIRED if name not in names]
    if missing:
        raise ValueError(f'{path} is not a standard 3DGS .ply, missing: {", ".join(missing)}')

    count = len(vertex)
    if count == 0:
        raise ValueError(f'{path} has no Gaussians')
    centers = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)
    features_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1).astype(np.float32)
    scaling = np.exp(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)).astype(np.float32)
    quaternions = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']],
                           axis=-1).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-vertex['opacity']))).astype(np.float32)[:, None]
    return {
        'name': os.path.splitext(os.path.basename(path))[0],
        'path': path,
        'count': count,
        'centers': centers,
        'covariances': covariances_from(scaling, quaternions),
        'rgbs': sh_dc_to_rgb(features_dc),
        'opacities': opacities,
        'extent': (centers.max(axis=0) - centers.min(axis=0)).round(3).tolist(),
    }


def collect(paths, limit=None):
    """Expand files/folders into a sorted list of .ply paths (folders are searched non-recursively)."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            found += [os.path.join(path, name) for name in sorted(os.listdir(path))
                      if name.lower().endswith('.ply')]
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise SystemExit(f'error: no such file or folder: {path}')
    if limit is not None and len(found) > limit:
        print(f'{len(found)} .ply files found, showing the first {limit} (raise with --limit)')
        found = found[:limit]
    return found


def build_scene(server, objects, up='+y'):
    """Populate a viser server with splat objects plus a GUI to toggle them. Returns the handles.

    Split out from serve() so it can be tested without opening a port."""
    server.scene.set_up_direction(up)
    server.scene.add_grid('/ground', plane='xy' if up in ('+y', '-y') else 'xz',
                          width=2.0, height=2.0, cell_size=0.1, cell_color=(70, 70, 70),
                          section_size=0.5, section_thickness=3)

    handles, visible = {}, {}
    with server.gui.add_folder('Objects'):
        for obj in objects:
            handles[obj['name']] = server.scene.add_gaussian_splats(
                f'/splats/{obj["name"]}', centers=obj['centers'], covariances=obj['covariances'],
                rgbs=obj['rgbs'], opacities=obj['opacities'])
            visible[obj['name']] = server.gui.add_checkbox(obj['name'], initial_value=True)
        scale_splats = server.gui.add_slider('Splat scale', min=0.1, max=3.0, step=0.05, initial_value=1.0)
        server.gui.add_markdown(f'{len(objects)} object(s), up axis **{up}**')
    with server.gui.add_folder('Details'):
        info = server.gui.add_markdown('')

    objects_by_name = {obj['name']: obj for obj in objects}

    def update(_=None):
        for name, handle in handles.items():
            handle.visible = bool(visible[name].value)
        total = sum(handles[name].visible and objects_by_name[name]['count'] for name in handles)
        info.content = (f'Showing **{total:,}** Gaussians from '
                        f'**{sum(1 for name in handles if handles[name].visible)}** of {len(handles)} object(s)')

    for checkbox in visible.values():
        checkbox.on_update(update)

    @scale_splats.on_update
    def _(_):
        for handle in handles.values():
            handle.scale = float(scale_splats.value)

    update()
    return handles


def serve(objects, host='127.0.0.1', port=8080, up='+y', open_browser=False):
    """Serve splat objects on http://host:port and block until interrupted."""
    try:
        import viser
    except ImportError:
        raise SystemExit('error: viser is not installed. Run scripts/setup.sh, or: pip install viser') from None
    if not objects:
        raise SystemExit('error: nothing to view (no .ply files found)')

    server = viser.ViserServer(host=host, port=port)
    build_scene(server, objects, up=up)
    url = f'http://{"localhost" if host in ("0.0.0.0", "127.0.0.1") else host}:{port}'
    print(f'\nViewing {len(objects)} object(s): {url}')
    print('  drag to orbit, scroll to zoom, right-drag to pan; Ctrl-C to stop\n')
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        server.stop()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='idol-view', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='.ply files and/or folders containing them')
    parser.add_argument('--host', default='127.0.0.1',
                        help='bind address; 0.0.0.0 to let others on your network in (default: localhost only)')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--up', choices=['+y', '-y', '+z', '-z'], default='+y',
                        help='up axis of the files: +y for what idol-gen writes (default); -y matches '
                             'idol-sdk\'s own save_ply, whose objects hang upside down in y-up viewers')
    parser.add_argument('--limit', type=int, default=24, help='max objects to load from folders (default: 24)')
    parser.add_argument('--open', action='store_true', help='open the browser automatically')
    parser.add_argument('--info', action='store_true', help='print what would be loaded and exit')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = collect(args.paths, limit=None if args.info else args.limit)
    if not paths:
        print('no .ply files found')
        return 1
    if args.info:
        for path in paths:
            try:
                obj = load_ply(path)
                print(f'{obj["name"]:24} {obj["count"]:>9,} gaussians  extent {obj["extent"]}')
            except ValueError as error:
                print(f'{os.path.basename(path):24} ERROR: {error}')
        return 0
    objects = []
    for path in paths:
        try:
            objects.append(load_ply(path))
        except ValueError as error:
            print(f'skipping: {error}', file=sys.stderr)
    return serve(objects, host=args.host, port=args.port, up=args.up, open_browser=args.open)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from idol_tools._bootstrap import reexec_into_venv

    reexec_into_venv(__file__)
    raise SystemExit(main())
