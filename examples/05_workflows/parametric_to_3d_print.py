"""
Real CAD part to a 3D-print 3MF plate
=====================================

A common prototyping job: you need many copies of one part, so you
nest them on a build plate and hand a single 3MF to the slicer with
each instance named so the slicer tree mirrors the layout.

The part is a real M3 nyloc lock nut from the FreeCAD parts library
(CC-BY 3.0, FreeCAD-library contributors). We read its native FCStd,
nest a 3x3 grid of instances, and write one 3MF; per-part labels
round-trip into the 3MF as ``<object name>``.
"""

# %%

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Read the real FCStd nut and tessellate it to a single surface.

nut = pyvista_cad.read_fcstd(downloads.fcstd_nut_path()).combine().extract_surface()
nut

# %%
# Nest a 3x3 grid of instances on the build plate, one named block
# per instance.

pitch = (nut.bounds.x_max - nut.bounds.x_min) * 1.5
plate = pv.MultiBlock()
rows = []
for i in range(3):
    for j in range(3):
        name = f'nut_r{i}c{j}'
        inst = nut.copy()
        inst.translate((i * pitch, j * pitch, 0.0), inplace=True)
        inst.field_data['cad.label'] = np.array([name])
        plate.append(inst, name=name)
        rows.append({'label': name, 'cells': inst.n_cells})

pd.DataFrame(rows)

# %%
# Write one 3MF carrying all nine instances, then round-trip it to
# confirm the layout and names survived.

out = Path(tempfile.mkdtemp()) / 'nut_plate.3mf'
pyvista_cad.write_three_mf(plate, out, units='mm')
back = pyvista_cad.read_three_mf(out)
back

# %%
# Render the build plate.

pl = pv.Plotter()
pl.add_mesh(back, multi_colors=True, show_edges=True, edge_color='black')
pl.show()
