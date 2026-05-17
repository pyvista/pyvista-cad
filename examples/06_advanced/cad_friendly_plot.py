"""
CAD-friendly plotting
=====================

A generic mesh viewer draws the *triangulation*: with ``show_edges=True``
you see every facet edge, an artifact of the tessellation tolerance that
means nothing to a CAD user. What a CAD user wants is the model's
*topological* edges (the B-rep feature curves) over smoothly shaded
faces, exactly like a CAD application's viewport.

``pyvista-cad`` reproduces that pipeline:

* :func:`pyvista_cad.topods_to_multiblock` tessellates each topological
  face independently and attaches *analytic* surface normals (a coarse
  cylinder still shades perfectly round), plus a sibling edges block of
  the real B-rep curves recovered from the same mesh pass.
* The ``plotter.cad`` component and the ``.cad.plot()`` accessor render
  faces smooth with their triangle edges hidden and the topological
  edges overlaid in the theme edge colour.
"""

# %%
import numpy as np
import pyvista as pv

import pyvista_cad
from pyvista_cad import examples

# %%
# Read a STEP bracket. It comes back as a :class:`pyvista.MultiBlock`
# with the originating ``TopoDS`` cached on each block, so the CAD-view
# helpers can recover exact topology.

mb = pyvista_cad.read_step(examples.bracket_step_path())

# %%
# The generic view: triangle mesh with facet edges. The filleted bend
# looks polygonal and the edges are tessellation noise, not features.

combined = mb.combine().extract_surface()
combined.plot(show_edges=True, color='lightblue')

# %%
# The CAD-friendly view: smooth analytic shading, only the topological
# edges drawn. Same data, but it reads like a CAD viewport.

mb.cad.plot()

# %%
# The data behind the render. ``topods_to_multiblock`` keeps per-face
# identity and classifies every edge curve.
#
# ``.cad.cad_view()`` resolves the read result (recovering each block's
# cached B-rep) into a ``{'faces', 'edges'}`` MultiBlock.
cad = mb.cad.cad_view(linear_deflection=0.2)
edges = cad['edges']
legend = edges.field_data['cad.edge_kind_legend'][0]
kinds, counts = np.unique(edges.cell_data['cad.edge_kind'], return_counts=True)
n_faces = sum(1 for _ in cad['faces'].cad.walk())
print(f'topological faces : {n_faces}')
print(f'topological edges : {edges.n_cells}')
print(f'edge-kind legend  : {legend}')
print(f'edge-kind counts  : {dict(zip(kinds.tolist(), counts.tolist()))}')

# %%
# A silhouette-only outline, handy for drawings and thumbnails.

pl = pv.Plotter()
pl.cad.add(cad, color='white', silhouette=True)
pl.show()
