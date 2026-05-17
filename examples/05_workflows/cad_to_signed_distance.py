"""
CAD shape to signed-distance field
==================================

A tessellated CAD part drives a signed-distance volume directly
through PyVista's ``compute_implicit_distance``. The result is a
volumetric scalar field useful for collision queries, mesh morphing,
implicit modeling, or feeding marching-cubes-style reconstructions.

The part is a real STEP solid, the NIST AM Bench 2022 LPBF bridge
specimen (public domain, https://doi.org/10.18434/mds2-2607). Its
multi-leg structure gives the SDF interior real shape to slice
through.
"""

# %%

import numpy as np
import pandas as pd
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real STEP part as a surface.

part = pyvista_cad.read_step(downloads.step_part_path()).combine().extract_surface()
part

# %%
# Build a uniform grid wrapping the part with a small margin.

bounds = np.array(part.bounds).reshape(3, 2)
margin = 4.0
dims = np.array((90, 30, 40))
origin = bounds[:, 0] - margin
spacing = (bounds[:, 1] - bounds[:, 0] + 2 * margin) / (dims - 1)
grid = pv.ImageData(dimensions=dims, origin=origin, spacing=spacing)

# %%
# Sample the signed distance into the volume.

grid = grid.compute_implicit_distance(part)
sdf = grid['implicit_distance']

pd.DataFrame(
    {
        'metric': ['SDF min', 'SDF max', 'inside-mass fraction'],
        'value': [sdf.min(), sdf.max(), float((sdf < 0).mean())],
    }
)

# %%
# Slice the SDF through two orthogonal planes to expose the interior
# field.

mid = bounds.mean(axis=1)
slc_z = grid.slice(normal='z', origin=mid)
slc_x = grid.slice(normal='x', origin=mid)
clim = (-float(np.percentile(np.abs(sdf), 95)), float(np.percentile(np.abs(sdf), 95)))

pl = pv.Plotter()
pl.add_mesh(part, color='lightgray', opacity=0.2)
pl.add_mesh(
    slc_z,
    scalars='implicit_distance',
    cmap='RdBu',
    clim=clim,
    scalar_bar_args={'title': 'signed dist [mm]'},
)
pl.add_mesh(slc_x, scalars='implicit_distance', cmap='RdBu', clim=clim, show_scalar_bar=False)
pl.show()
