"""Writing 3D Gaussian splat .ply with the up axis viewers actually expect.

The SDK's `Gaussian` is z-up — verified, not assumed: rendering the object from its own predicted camera with
world up +z reproduces the input image (seat on top, foot ring at the bottom), while world up -z stands the
object on its head. The predicted camera pitch is positive for the same images, and those images show the top
of the object, so +z really is up in the model's frame.

Upstream's `Gaussian.save_ply` rotates by **+90 degrees about x**, which maps that up to **-y** — so the file
lands upside down in every y-up viewer (gsplat.tech, SuperSplat, Blender add-ons, viser). Rotating by **-90
degrees about x** instead puts up at **+y**, where a y-up viewer looks for it.

`convention='upstream'` reproduces idol-sdk's bytes exactly, so you can A/B the two or compare against the
official Space. `up_axis()` is the single source of truth for which way is up in what we just wrote, so the
viewer and the web app cannot disagree with the writer.
"""
import math

SQRT_HALF = math.sqrt(0.5)

# Pre-multiplied rotation applied to every Gaussian (and to every centre) as the object leaves the z-up frame.
CONVENTIONS = {
    'y-up': (SQRT_HALF, -SQRT_HALF, 0.0, 0.0),      # R_x(-90): (x, y, z) -> (x,  z, -y); up +z -> +y
    'upstream': (SQRT_HALF, SQRT_HALF, 0.0, 0.0),   # R_x(+90): (x, y, z) -> (x, -z,  y); up +z -> -y
}
UP_AXIS = {'y-up': '+y', 'upstream': '-y'}
ATTRIBUTE_NAMES = ('x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                   'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')


def rotate_centers(xyz, convention='y-up'):
    """The SDK's z-up object -> the frame `convention` writes, as (N, 3)."""
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float32)
    if convention == 'y-up':
        return np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=-1)
    if convention == 'upstream':
        return np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=-1)
    raise ValueError(f'unknown convention {convention!r}; expected one of {sorted(CONVENTIONS)}')


def save_ply(gaussian, path, convention='y-up'):
    """Write a standard 3DGS .ply. Same attributes as upstream, one different rotation.

    Values are clamped before the inverse activations the format wants: a saturated opacity or a zero scale
    would otherwise be written as +/-inf, which some viewers reject."""
    import numpy as np
    from idol.gaussian import inverse_sigmoid, quaternion_multiply
    from plyfile import PlyData, PlyElement

    if convention not in CONVENTIONS:
        raise ValueError(f'unknown convention {convention!r}; expected one of {sorted(CONVENTIONS)}')

    eps = 1e-6
    xyz = rotate_centers(gaussian.xyz.detach().cpu().numpy(), convention)
    normals = np.zeros_like(xyz)
    f_dc = gaussian.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = inverse_sigmoid(gaussian.opacity.clamp(eps, 1 - eps)).detach().cpu().numpy()
    scale = gaussian.scaling.clamp_min(eps).log().detach().cpu().numpy()
    rotation = quaternion_multiply(np.array(CONVENTIONS[convention], dtype=np.float64),
                                   gaussian.rotation.detach().cpu().numpy())

    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1).astype(np.float32)
    elements = np.empty(attributes.shape[0], dtype=[(name, 'f4') for name in ATTRIBUTE_NAMES])
    for column, name in enumerate(ATTRIBUTE_NAMES):
        elements[name] = attributes[:, column]
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    return path
