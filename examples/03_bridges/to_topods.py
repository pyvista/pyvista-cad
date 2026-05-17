"""
PyVista -> TopoDS_Compound
==========================

Send a tessellated :class:`pyvista.PolyData` back into the OCCT world
as a faceted ``TopoDS_Compound``. Useful when downstream OCCT-based
code expects shape objects rather than meshes, e.g. boolean
operations or further OCCT-native exports.

This example loads a real STEP part, the NIST AM Bench 2022 LPBF
bridge specimen (public domain, https://doi.org/10.18434/mds2-2607),
pushes it through ``to_topods``, performs a boolean cut in OCCT
(the kind of operation that only works on a TopoDS shape), then
rounds it back to PyVista.
"""

# %%

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real STEP part.

mesh = pyvista_cad.read_step(downloads.step_part_path()).combine().extract_surface()
mesh

# %%
# Push to OCCT.

shape = pyvista_cad.to_topods(mesh)
type(shape).__name__

# %%
# Boolean-cut against an OCCT primitive: an operation that lives in
# TopoDS-land, not available on a PyVista mesh. The box trims one end
# of the specimen.

b = mesh.bounds
cutter = BRepPrimAPI_MakeBox(
    gp_Pnt(b.x_max - 10.0, b.y_min - 1.0, b.z_min - 1.0),
    20.0,
    (b.y_max - b.y_min) + 2.0,
    (b.z_max - b.z_min) + 2.0,
).Shape()
cut_result = BRepAlgoAPI_Cut(shape, cutter).Shape()
type(cut_result).__name__

# %%
# Round-trip the cut compound back to PyVista.

back = pyvista_cad.from_topods(cut_result)
back

# %%
# Render the post-cut part.

back['X (mm)'] = back.points[:, 0]

pl = pv.Plotter()
pl.cad.add(back, scalars='X (mm)', cmap='inferno')
pl.show()
