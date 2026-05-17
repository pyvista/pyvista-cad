"""
Read an IGES file
=================

IGES (``.iges`` / ``.igs``) is one of the older neutral CAD interchange
formats, still common for legacy mechanical hand-offs and
turbomachinery. ``pyvista_cad.read_iges`` wraps `pyiges
<https://github.com/pyvista/pyiges>`_ and tessellates analytic
surfaces with geomdl.

This example uses a real centrifugal impeller exported from SolidWorks
to IGES (MIT, pyiges) and pushes beyond a load+plot demo by computing
surface curvature, the kind of analysis turbomachinery engineers
routinely run on imported IGES geometry.
"""

# %%

import numpy as np
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real impeller. ``surfaces=True`` tessellates every analytic
# B-spline surface; ``lines=False`` drops the construction wires.

iges = pyvista_cad.read_iges(
    downloads.iges_impeller_path(),
    bsplines=True,
    surfaces=True,
    lines=False,
)
iges

# %%
# Compute mean curvature on the impeller surface. Mean curvature
# highlights the convex blade ridges and concave hub fillets: the
# regions a CFD or stress engineer wants to refine first. (Gaussian
# curvature on a near-developable blade is ~0 almost everywhere and
# washes out; mean curvature reads far better.)

surface = iges.extract_surface().compute_normals()
surface['mean_curvature'] = surface.curvature(curv_type='mean')
surface

# %%
# Render with a diverging colormap centred on zero so flat regions sit
# mid-scale and ridges / valleys pop. Symmetric, robust ``clim`` from
# the 90th percentile of the absolute curvature keeps a few extreme
# outliers from flattening the contrast.

q = float(np.nanpercentile(np.abs(surface['mean_curvature']), 90))

pl = pv.Plotter()
pl.cad.add(
    surface,
    scalars='mean_curvature',
    cmap='RdBu_r',
    clim=(-q, q),
    scalar_bar_args={'title': 'Mean curvature'},
)
pl.show()
