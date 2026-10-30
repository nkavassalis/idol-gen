"""Idol command line tools: image -> 3D Gaussian splat, plus a local browser viewer.

  idol-gen photo.jpg --mode real      generate a .ply
  idol-view outputs/                  look at it in the browser
  idol-test --cli-only                run the checks that need no GPU

The model itself is the upstream `speridlabs/idol` SDK; this package is the CLI, the presets, the validation
and the viewer around it.
"""

__version__ = '0.1.0'
