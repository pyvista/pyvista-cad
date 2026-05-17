"""
Read a 3MF package
==================

3MF is the modern interchange format for additive manufacturing. It
carries per-object geometry, names, colors, and units in a single
archive. This example loads a real 3MF from the 3MF Consortium sample
set (BSD-2-Clause): a colored dodecahedron chain from the material
extension.

Why this matters: slicers like PrusaSlicer and Cura key off the
per-object content inside the 3MF. Reading it straight into a
:class:`pyvista.MultiBlock` keeps a design flow headed for print.
"""

# %%

import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Read the real 3MF package.

mb = pyvista_cad.read_three_mf(downloads.three_mf_colored_path())
mb

# %%
# Render the model.

pl = pv.Plotter()
pl.add_mesh(mb, multi_colors=True, show_edges=True, edge_color='black')
pl.show()
