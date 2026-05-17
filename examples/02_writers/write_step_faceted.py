"""
Write a faceted STEP file
=========================

Send a tessellated mesh back to a STEP file via OCCT's
``STEPControl_Writer``. The result is faceted (triangulated), not a
B-rep reconstruction, ideal for downstream tools that accept STEP
for visualization or CAM but do not require analytical surfaces.

This example takes a real 3D scan, the T-LESS object-5
reconstruction from Primesense RGB-D scans (CC-BY 4.0, Hodaň et al.,
WACV 2017), decimates it to a CAM-friendly facet count, and writes
it to a faceted STEP: the canonical "scanned mesh into a CAD/CAM
pipeline" hand-off.
"""

# %%

from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real scan. A faceted STEP carries one face per triangle, so
# decimation is the routine prep before this hand-off.

_, scan_path = downloads.scan_pair_paths()
src = pv.read(scan_path).extract_surface().triangulate().decimate(0.8)
src.field_data['cad.units'] = np.array(['mm'])
src

# %%
# Write a faceted STEP.

path = Path(tempfile.mkdtemp()) / 'tless_obj5_faceted.step'
pyvista_cad.write_step(src, path)

# %%
# Read back. The output is a faceted STEP; check the cell count
# tracks the source tessellation.

out = pyvista_cad.read_step(path)
out

# %%
# Render colored by Z.

surface = out.combine().extract_surface()
surface['Z'] = surface.points[:, 2]

pl = pv.Plotter()
pl.add_mesh(surface, scalars='Z', cmap='plasma', show_edges=True, edge_color='black')
pl.show()
