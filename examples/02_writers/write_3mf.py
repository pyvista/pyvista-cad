"""
Write a 3MF package
===================

Export a printable part to 3MF. The 3MF format carries per-object
names, units, and (where supported) colors: everything a modern
slicer needs to put the part on the build plate.

This example loads a real STEP part, the NIST AM Bench 2022 LPBF
bridge specimen (public domain, https://doi.org/10.18434/mds2-2607),
writes it to 3MF in millimetres, and verifies the round-trip
preserves geometry and units.
"""

# %%

from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real part and tessellate it to a single surface.

mesh = pyvista_cad.read_step(downloads.step_part_path()).combine().extract_surface()
mesh.field_data['cad.label'] = np.array(['amb2022_bridge'])
mesh

# %%
# Write to 3MF, preserving units explicitly.

path = Path(tempfile.mkdtemp()) / 'amb2022_bridge.3mf'
pyvista_cad.write_three_mf(mesh, path, units='mm')

# %%
# Round-trip through ``read_three_mf`` to confirm the file is valid
# and the units survived.

mb = pyvista_cad.read_three_mf(path)
mb

# %%
# Render the round-tripped part.

surface = mb.combine().extract_surface()
surface['Z (mm)'] = surface.points[:, 2]

pl = pv.Plotter()
pl.add_mesh(surface, scalars='Z (mm)', cmap='viridis', show_edges=True, edge_color='black')
pl.show()
