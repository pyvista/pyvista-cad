"""
gmsh round-trip
===============

Push a :class:`pyvista.PolyData` surface into a gmsh model, then pull
the result back as a :class:`pyvista.UnstructuredGrid`. Use this when
you need gmsh's remesher, e.g. to clean up a noisy scan or to
prepare a surface for a downstream FEA tet-mesher.

This example takes a real 3D scan, the T-LESS object-5
reconstruction from Primesense RGB-D scans (CC-BY 4.0, Hodaň et al.,
WACV 2017), and runs it through gmsh: cleaning a noisy scan is
exactly the use case this bridge exists for.
"""

# %%

import gmsh
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real scan and decimate it. gmsh's discrete remesher
# chokes on a dense scan, and thinning it first is the normal prep.

_, scan_path = downloads.scan_pair_paths()
surface = pv.read(scan_path).extract_surface().triangulate().decimate(0.9)
surface

# %%
# Push into gmsh and pull back.

gmsh.initialize()
try:
    pyvista_cad.to_gmsh(surface, model_name='scan_demo')
    grid = pyvista_cad.from_gmsh()
finally:
    gmsh.finalize()

grid

# %%
# Render the gmsh-remeshed surface.

grid['Z'] = grid.points[:, 2]

pl = pv.Plotter()
pl.add_mesh(grid, scalars='Z', cmap='inferno', show_edges=True, edge_color='black')
pl.show()
