"""
Read a STEP part
================

STEP is the dominant neutral interchange format in mechanical CAD.
This example loads a real STEP part, the NIST AM Bench 2022 LPBF
bridge specimen (public domain, https://doi.org/10.18434/mds2-2607),
and inspects the resulting :class:`pyvista.MultiBlock`. Every STEP
part becomes one block with its OCCT label and color attached as
``cad.label`` / ``cad.color`` field data.
"""

# %%

import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Read the real STEP part.

mb = pyvista_cad.read_step(downloads.step_part_path())
mb

# %%
# Render CAD-friendly: smooth faces coloured by Z with the model's
# topological edges overlaid; the triangle-mesh edges are hidden.

for _, part in mb.cad.walk():
    part['elevation'] = part.points[:, 2]

pl = pv.Plotter()
pl.cad.add(mb, scalars='elevation', cmap='viridis')
pl.show()
