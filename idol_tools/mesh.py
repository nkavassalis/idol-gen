"""Turn Gaussian splats into a printable mesh (.stl / .obj / .ply).

Splats are not a surface, so this reconstructs one: every Gaussian contributes its opacity to a voxel grid as
the ellipsoidal Gaussian it actually is (an accumulated "solidity" field, in the spirit of Gaussian Opacity
Fields), then marching cubes takes an isosurface of that field. That matters for real files: in an Idol output
the median Gaussian is *smaller than a voxel* while others are flat sheets several tenths wide, so splatting
centres and blurring would either shred the thin parts or blob the wide ones.

Gaussians are bucketed by kernel footprint before accumulation, which keeps a million-splat file interactive:
most splats touch a 1x1x1 or 2x1x1 block, so one vectorised pass handles a whole bucket.

Quality is a printable proxy, not a scan: expect soft detail and merged thin features. `--resolution` and
`--threshold` are the two knobs that matter, and `--info` prints field statistics that tell you where the
threshold should sit.

numpy is imported inside the functions rather than at module scope so that `idol-mesh --help` works on a bare
interpreter, matching viewer.py and the CI job that checks help screens with nothing installed.
"""
import os

DEFAULT_RESOLUTION = 256
DEFAULT_THRESHOLD = 0.5
DEFAULT_SIGMA = 0.8             # pre-smoothing, in voxels: closes the pinhole noise splats leave behind
DEFAULT_RADIUS = 3.0            # how far out, in sigmas, each Gaussian is accumulated
DEFAULT_MAX_SIDE = 32           # cap on a kernel's voxels per axis, so one huge splat cannot stall the run
DEFAULT_KEEP_SMALL = 100        # drop connected components smaller than this many faces
FORMATS = ('stl', 'obj', 'ply')


def available():
    """True when the optional meshing dependencies are importable. Lets the web app boot and grey out the
    STL controls instead of crashing when someone has not installed the mesh extra."""
    import importlib.util

    return (importlib.util.find_spec('skimage') is not None
            and importlib.util.find_spec('scipy') is not None
            and importlib.util.find_spec('plyfile') is not None)


def _require_mesh_deps():
    try:
        from scipy.ndimage import gaussian_filter
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from skimage.measure import marching_cubes
    except ImportError:
        raise SystemExit(
            'error: meshing needs scikit-image and scipy. Install them with\n'
            "  uv pip install --python .venv/bin/python scikit-image\n"
            "or: scripts/setup.sh --mesh") from None
    return marching_cubes, gaussian_filter, coo_matrix, connected_components


def stl_record_dtype():
    """One binary STL record: 12 float32 (normal + 3 vertices) then a uint16 attribute count = 50 bytes."""
    import numpy as np

    return np.dtype([('normal', '<f4', (3,)), ('a', '<f4', (3,)), ('b', '<f4', (3,)),
                     ('c', '<f4', (3,)), ('attributes', '<u2')])


def solidity(centers, covariances, opacities, resolution=DEFAULT_RESOLUTION, pad=0.06,
             radius=DEFAULT_RADIUS, max_side=DEFAULT_MAX_SIDE, subsample=0, chunk=4096, progress=None):
    """Accumulate splat opacity onto a grid. Returns (field, origin, voxel_size).

    `field` is in units of accumulated opacity: a thickly covered interior reaches several units, bare surface
    voxels sit near 1, floaters stay below 1. `threshold` in mesh_from_solidity picks the surface. Note the
    isosurface sits outside the splat cloud by roughly `radius` sigma, so results are slightly dilated."""
    import numpy as np

    centers = np.asarray(centers, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64).reshape(-1)
    if centers.ndim != 2 or centers.shape[1:] != (3,) or covariances.shape != (centers.shape[0], 3, 3):
        raise ValueError('expected centers (N, 3) and covariances (N, 3, 3)')
    if len(centers) == 0:
        raise ValueError('nothing to mesh: no gaussians')

    keep = (opacities > 0.02) & np.isfinite(centers).all(axis=1) & np.isfinite(covariances).all(axis=(1, 2))
    centers, covariances, opacities = centers[keep], covariances[keep], opacities[keep]
    if len(centers) == 0:
        raise ValueError('nothing to mesh: every Gaussian is transparent or non-finite')
    if subsample and len(centers) > subsample:
        step = int(np.ceil(len(centers) / subsample))
        centers, covariances, opacities = centers[::step], covariances[::step], opacities[::step]

    lo = centers.min(axis=0) - pad
    hi = centers.max(axis=0) + pad
    voxel = float(np.max(hi - lo)) / float(resolution)
    dims = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int64), 1)
    field = np.zeros(dims, dtype=np.float32)

    jitter = 1e-12 * np.eye(3)
    values, vectors = np.linalg.eigh(covariances + jitter)                # (N, 3) variances and rotation columns
    sigmas = np.sqrt(np.maximum(values, 1e-24))
    half = radius * np.einsum('nij,nj->ni', np.abs(vectors), sigmas)      # axis-aligned extent of each ellipsoid
    trace = np.trace(covariances, axis1=1, axis2=2)[:, None, None]
    precision = np.linalg.inv(covariances + jitter * trace / 3.0)

    centre_vox = (centers - lo) / voxel
    low = np.floor(centre_vox - half / voxel).astype(np.int64)
    high = np.ceil(centre_vox + half / voxel).astype(np.int64)
    clipped_low = np.maximum(low, 0)
    clipped_high = np.minimum(high, dims - 1)
    side = clipped_high - clipped_low + 1
    usable = (side > 0).all(axis=1) & (high - low).max(axis=1) <= max_side
    if not usable.all() and progress:
        progress(f'{int((~usable).sum())} Gaussian(s) had kernels over {max_side} voxels wide and were skipped')

    # Bucket by kernel footprint: every member of a bucket shares one offset grid, so the whole bucket is
    # accumulated in a single vectorised scatter instead of one Python iteration per splat.
    buckets = {}
    for idx in np.flatnonzero(usable):
        buckets.setdefault(tuple(side[idx]), []).append(idx)

    done = 0
    for shape, indices in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        dx, dy, dz = shape
        offsets = np.stack(np.meshgrid(np.arange(dx), np.arange(dy), np.arange(dz), indexing='ij'), axis=-1)
        offsets = offsets.reshape(-1, 3).astype(np.int64)
        start = 0
        while start < len(indices):
            group = np.asarray(indices[start:start + chunk])
            start += chunk
            base = clipped_low[group]                                        # (G, 3) first voxel of each box
            pts = (base[:, None, :] + offsets[None, :, :]) * voxel + lo - centers[group][:, None, :]
            q = np.einsum('gki,gij,gkj->gk', pts, precision[group], pts)     # Mahalanobis^2 per sample
            weight = opacities[group][:, None] * np.exp(-0.5 * np.clip(q, 0.0, 60.0))
            inside = (q <= radius * radius) & (weight > 1e-4)
            flat = ((base[:, None, 0] + offsets[None, :, 0]) * dims[1] * dims[2]
                    + (base[:, None, 1] + offsets[None, :, 1]) * dims[2]
                    + (base[:, None, 2] + offsets[None, :, 2]))
            np.add.at(field.reshape(-1), flat[inside].ravel(), weight[inside].ravel())
        done += len(indices)
        if progress:
            progress(f'{done:,}/{len(centers):,} gaussians accumulated')
    return field, lo, voxel


def drop_small_components(vertices, faces, min_faces):
    """Keep the pieces worth printing: floaters and dust are the usual reason a splat mesh fails on a printer."""
    import numpy as np

    if min_faces <= 0 or len(faces) == 0:
        return vertices, faces, 0
    _, _, coo_matrix, connected_components = _require_mesh_deps()

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    count, labels = connected_components(graph, directed=False)
    faces_per_component = np.bincount(labels[faces[:, 0]], minlength=count)
    big = faces_per_component >= min_faces
    kept_faces = faces[big[labels[faces[:, 0]]]]
    if len(kept_faces) == 0:
        return vertices[:0], kept_faces, len(faces)
    # Compact in place, keeping vertices shared. Rebuilding as a triangle soup would detach every triangle
    # from its neighbours and make each one its own component.
    used = np.zeros(len(vertices), dtype=bool)
    used[kept_faces.ravel()] = True
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    return vertices[used], remap[kept_faces], int(len(faces) - len(kept_faces))


def mesh_from_solidity(field, origin, voxel, threshold=DEFAULT_THRESHOLD, smooth=0,
                       min_faces=DEFAULT_KEEP_SMALL, sigma=DEFAULT_SIGMA):
    """Isosurface of the accumulated field. Returns (vertices, faces, stats)."""
    import numpy as np

    marching_cubes, gaussian_filter, _, _ = _require_mesh_deps()

    work = field
    if sigma and sigma > 0:
        work = gaussian_filter(field.astype(np.float32), sigma=sigma)
    if work.max() <= threshold:
        raise ValueError(f'field peaks at {work.max():.3f}, below the threshold {threshold}: nothing would be '
                         'extracted. Lower --threshold (see --info).')

    verts, faces, _normals, _values = marching_cubes(work, level=threshold, spacing=(voxel, voxel, voxel))
    verts = verts + np.asarray(origin, dtype=np.float64)
    before = len(faces)
    verts, faces, dropped = drop_small_components(verts, faces, min_faces)
    if len(faces) == 0:
        raise ValueError(f'every extracted component was smaller than --keep-small ({min_faces} faces)')

    for _ in range(int(smooth)):
        verts = _laplacian_smooth(verts, faces, 0.5)

    # Marching cubes winding is not guaranteed outward, and slicers care: a negative signed volume means the
    # normals point into the object, so flip the winding rather than trusting abs() downstream.
    if _signed_volume(verts, faces) < 0:
        faces = faces[:, ::-1]

    stats = {'vertices': int(len(verts)), 'faces': int(len(faces)), 'dropped_faces': int(before - len(faces)),
             'field_max': float(field.max()), 'threshold': float(threshold),
             'bbox_min': verts.min(axis=0).round(4).tolist(), 'bbox_max': verts.max(axis=0).round(4).tolist(),
             'volume': float(abs(_signed_volume(verts, faces))), 'watertight': _is_closed(faces)}
    return verts.astype(np.float32), faces.astype(np.int64), stats


def _laplacian_smooth(verts, faces, lam):
    import numpy as np

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                            faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]])
    neighbours = np.zeros((len(verts), 3), dtype=np.float64)
    counts = np.zeros(len(verts), dtype=np.float64)
    np.add.at(neighbours, edges[:, 0], verts[edges[:, 1]])
    np.add.at(counts, edges[:, 0], 1.0)
    average = np.divide(neighbours, counts[:, None], out=np.zeros_like(neighbours), where=counts[:, None] > 0)
    return verts + lam * (average - verts)


def _signed_volume(verts, faces):
    """Signed volume of a triangle soup: negative when the winding points the normals inwards."""
    import numpy as np

    if len(faces) == 0:
        return 0.0
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def _mesh_volume(verts, faces):
    return abs(_signed_volume(verts, faces))


def _is_closed(faces):
    import numpy as np

    if len(faces) == 0:
        return False
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = np.sort(edges, axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    return bool((counts == 2).all())


def write_stl(path, vertices, faces):
    """Binary STL: 80-byte header, one uint32 triangle count, then 50 bytes per triangle."""
    import numpy as np

    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)

    records = np.empty(len(faces), dtype=stl_record_dtype())
    records['normal'] = normals.astype(np.float32)
    records['a'] = a.astype(np.float32)
    records['b'] = b.astype(np.float32)
    records['c'] = c.astype(np.float32)
    records['attributes'] = 0
    with open(path, 'wb') as handle:
        handle.write(b'idol-gen'.ljust(80, b'\0'))
        handle.write(np.array(len(faces), dtype=np.uint32).tobytes())
        handle.write(records.tobytes())
    return path


def read_stl(path):
    """Read back a binary STL. Used by the tests, and handy for sanity-checking any slicer input."""
    import numpy as np

    dtype = stl_record_dtype()
    with open(path, 'rb') as handle:
        handle.read(80)
        count = int(np.frombuffer(handle.read(4), dtype=np.uint32)[0])
        records = np.frombuffer(handle.read(count * dtype.itemsize), dtype=dtype)
    if len(records) != count:
        raise ValueError(f'{path}: header claims {count} triangles, file holds {len(records)}')
    vertices = np.stack([records['a'], records['b'], records['c']], axis=1).reshape(-1, 3)
    normal = np.stack([records['normal']] * 3, axis=1).reshape(-1, 3)
    return vertices, normal


def write_obj(path, vertices, faces):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('# generated by idol-gen\n')
        for v in vertices:
            handle.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for f in faces:
            handle.write(f'f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n')
    return path


def write_mesh_ply(path, vertices, faces):
    import numpy as np
    from plyfile import PlyData, PlyElement

    vertex = np.empty(len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'], vertex['y'], vertex['z'] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    face = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = faces
    PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')]).write(path)
    return path


WRITERS = {'stl': write_stl, 'obj': write_obj, 'ply': write_mesh_ply}


def mesh_file(ply_path, out_dir=None, fmt='stl', resolution=DEFAULT_RESOLUTION, threshold=DEFAULT_THRESHOLD,
              radius=DEFAULT_RADIUS, max_side=DEFAULT_MAX_SIDE, subsample=0, smooth=0, sigma=DEFAULT_SIGMA,
              min_faces=DEFAULT_KEEP_SMALL, quiet=False, overwrite=False):
    """One .ply -> one mesh. Returns (out_path, stats)."""
    from idol_tools import viewer

    if fmt not in FORMATS:
        raise ValueError(f'unknown format {fmt!r}; expected one of {list(FORMATS)}')
    if not 16 <= int(resolution) <= 1024:
        raise ValueError(f'--resolution must be between 16 and 1024 (got {resolution})')
    if threshold <= 0:
        raise ValueError('--threshold must be positive')

    out_dir = out_dir or os.path.dirname(os.path.abspath(ply_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(ply_path))[0]
    out_path = os.path.join(out_dir, f'{stem}.{fmt}')
    if os.path.exists(out_path) and not overwrite:
        return out_path, {'skipped': True}

    obj = viewer.load_ply(ply_path)
    if not quiet:
        print(f'  {obj["count"]:,} gaussians -> {resolution}^3 field ...', flush=True)
    field, origin, voxel = solidity(obj['centers'], obj['covariances'], obj['opacities'],
                                    resolution=resolution, radius=radius, max_side=max_side,
                                    subsample=subsample,
                                    progress=(None if quiet else (lambda m: print(f'    {m}', flush=True))))
    vertices, faces, stats = mesh_from_solidity(field, origin, voxel, threshold=threshold, smooth=smooth,
                                                sigma=sigma, min_faces=min_faces)
    WRITERS[fmt](out_path, vertices, faces)
    stats.update({'source': os.path.basename(ply_path), 'format': fmt, 'gaussians': int(obj['count']),
                  'size_bytes': os.path.getsize(out_path), 'voxel': voxel})
    return out_path, stats


def info(path, resolution=DEFAULT_RESOLUTION, threshold=DEFAULT_THRESHOLD):
    """Report the field statistics without writing anything: where the threshold should sit."""
    import numpy as np

    from idol_tools import viewer

    obj = viewer.load_ply(path)
    field, _origin, voxel = solidity(obj['centers'], obj['covariances'], obj['opacities'], resolution=resolution)
    percentiles = np.percentile(field, [50, 90, 99])
    return {'gaussians': int(obj['count']), 'grid': list(field.shape), 'voxel': float(voxel),
            'field_max': float(field.max()), 'field_median': float(percentiles[0]),
            'field_p90': float(percentiles[1]), 'field_p99': float(percentiles[2]),
            'voxels_above_threshold': int((field > threshold).sum()),
            'suggested_threshold': float(max(0.05, 0.5 * float(field.max())))}


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog='idol-mesh', description='Convert Idol .ply splats into printable .stl / .obj / .ply meshes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputs', nargs='+', help='.ply file(s) or folder(s)')
    parser.add_argument('--out', default=None, help='output folder (default: next to each input)')
    parser.add_argument('--format', choices=FORMATS, default='stl', help='mesh format to write')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION,
                        help='voxels along the longest axis; raise it for small parts you will print')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='surface level of the accumulated opacity field (see --info)')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS,
                        help='how far out each Gaussian is accumulated, in sigmas')
    parser.add_argument('--max-side', dest='max_side', type=int, default=DEFAULT_MAX_SIDE,
                        help='skip Gaussians whose kernel spans more than this many voxels per axis')
    parser.add_argument('--subsample', type=int, default=0,
                        help='use at most this many gaussians (0 = all); speeds up 1M-splat files')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA,
                        help='pre-smooth the field with this many voxels of Gaussian blur (softens noise)')
    parser.add_argument('--smooth', type=int, default=0, help='Laplacian smoothing iterations on the mesh')
    parser.add_argument('--keep-small', dest='keep_small', type=int, default=DEFAULT_KEEP_SMALL,
                        help='drop connected components with fewer than this many faces (floaters/dust)')
    parser.add_argument('--info', action='store_true', help='print field statistics and write nothing')
    parser.add_argument('--overwrite', action='store_true', help='re-mesh even if the output exists')
    parser.add_argument('--quiet', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    files = []
    for item in args.inputs:
        if os.path.isdir(item):
            files.extend(sorted(os.path.join(item, name) for name in os.listdir(item)
                                 if name.lower().endswith('.ply')))
        else:
            files.append(item)
    files = [f for f in files if f.lower().endswith('.ply')]
    if not files:
        raise SystemExit('error: no .ply files found')

    if args.info:
        for path in files:
            stats = info(path, resolution=args.resolution, threshold=args.threshold)
            print(f'{os.path.basename(path)}: {stats["gaussians"]:,} gaussians, grid {stats["grid"]}, '
                  f'field max {stats["field_max"]:.2f} p90 {stats["field_p90"]:.2f} '
                  f'-> try --threshold {stats["suggested_threshold"]:.2f}')
        return 0

    print(f'meshing {len(files)} file(s) as {args.format}, resolution {args.resolution}, '
          f'threshold {args.threshold}')
    written, failed = [], 0
    for path in files:
        print(f'\n{os.path.basename(path)}')
        try:
            out_path, stats = mesh_file(path, out_dir=args.out, fmt=args.format, resolution=args.resolution,
                                        threshold=args.threshold, radius=args.radius, max_side=args.max_side,
                                        subsample=args.subsample, smooth=args.smooth, sigma=args.sigma,
                                        min_faces=args.keep_small, quiet=args.quiet, overwrite=args.overwrite)
        except (ValueError, RuntimeError) as error:
            print(f'  skipped: {error}')
            failed += 1
            continue
        if stats.get('skipped'):
            print(f'  exists: {out_path} (use --overwrite)')
            continue
        written.append(out_path)
        print(f'  {out_path}  ({stats["faces"]:,} faces, {stats["vertices"]:,} vertices, '
              f'{stats["size_bytes"] / 1e6:.1f} MB, volume {stats["volume"]:.4f}, '
              f'{"closed" if stats["watertight"] else "open"})')
    print(f'\n{len(written)} mesh(es) written')
    return 1 if not written and failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
