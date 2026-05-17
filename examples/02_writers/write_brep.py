"""
Write a BREP file
=================

Serialize a tessellated :class:`pyvista.PolyData` as a faceted
``TopoDS_Compound`` in OpenCASCADE's native BREP format. BREP is the
canonical handoff to any OCCT-aware tool (FreeCAD, build123d,
cadquery).

This example loads a real STEP fixture, the NIST AM Bench 2022
recoater-guide part (public domain,
https://doi.org/10.18434/mds2-2607), writes it out to BREP, and
reads it back, confirming the round-trip preserves geometry that
downstream OCCT code can consume.
"""

# %%

from pathlib import Path
import tempfile

import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Load the real part and write it to BREP.

src = pyvista_cad.read_step(downloads.step_recoater_path()).combine().extract_surface()
path = Path(tempfile.mkdtemp()) / 'recoater.brep'
pyvista_cad.write_brep(src, path)

# %%
# Read back and verify the tessellation reproduces the source.

poly = pyvista_cad.read_brep(path, linear_deflection=0.1)
poly

# %%
# Render colored by Z.

poly['Z'] = poly.points[:, 2]

pl = pv.Plotter()
pl.cad.add(poly, scalars='Z', cmap='cividis')
pl.show()
