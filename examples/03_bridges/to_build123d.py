"""
PyVista -> build123d
====================

Wrap a :class:`pyvista.PolyData` as a build123d ``Compound`` of
faceted triangles, then push it onward to STEP through build123d's
native exporter. This is the canonical "scanned mesh into a CAD
pipeline" flow.

This example uses a real 3D scan: the T-LESS object-5 reconstruction
from real Primesense RGB-D scans (CC-BY 4.0, Hodaň et al., WACV
2017). We decimate it for a tractable facet count, wrap it as a
build123d Compound, query its bounding box, and export a STEP file
fit for downstream CAM.
"""

# %%

from pathlib import Path
import tempfile

import build123d as b3d
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real scan and decimate it to a tractable triangle count.

_, scan_path = downloads.scan_pair_paths()
mesh = pv.read(scan_path).extract_surface().triangulate()
mesh = mesh.decimate(0.85)
mesh

# %%
# Wrap as a build123d Compound and query its bounding box.

compound = pyvista_cad.to_build123d(mesh)
compound.bounding_box()

# %%
# Export the compound straight to STEP: the typical hand-off for CAM.

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / 'tless_obj5_scan.step'
    b3d.export_step(compound, str(out))

# %%
# Round-trip back to PyVista for rendering.

back = pyvista_cad.from_build123d(compound)
back['Z'] = back.points[:, 2]

pl = pv.Plotter()
pl.cad.add(back, scalars='Z', cmap='magma')
pl.show()
