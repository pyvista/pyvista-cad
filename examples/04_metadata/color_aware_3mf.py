"""
Per-object names round-trip through 3MF
=======================================

3MF stores one ``<object>`` per mesh. When you write a
:class:`pyvista.MultiBlock`, every block name and ``cad.label`` is
preserved on the way in and on the way out.

This example loads a real multi-part STEP assembly, the NIST AM
Bench 2022 build-plate layout (public domain,
https://doi.org/10.18434/mds2-2607) with seven named products, writes
it to 3MF, and confirms names, units, and per-block metadata survive
the round-trip.
"""

# %%

from pathlib import Path
import tempfile

import pandas as pd
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Read the real assembly: each part is a named block.

mb_in = pyvista_cad.read_step(downloads.step_assembly_path())
mb_in

# %%
# Write to 3MF, then read it back.

out = Path(tempfile.mkdtemp()) / 'amb2022_plate.3mf'
pyvista_cad.write_three_mf(mb_in, out, units='mm')
mb_out = pyvista_cad.read_three_mf(out)

rows = []
for key in mb_out.keys():  # noqa: SIM118
    block = mb_out[key]
    rows.append(
        {
            'key': key,
            'cells': block.n_cells,
            'guid': str(block.field_data['cad.guid'][0])[:8],
        }
    )
pd.DataFrame(rows)

# %%
# Render the round-tripped assembly.

pl = pv.Plotter()
pl.add_mesh(mb_out, multi_colors=True, show_edges=True, edge_color='black')
pl.show()
