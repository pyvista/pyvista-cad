"""
Walk and filter a real IFC assembly
===================================

The ``.cad`` accessor adds ``walk``, ``find``, and ``flatten`` for
hierarchical CAD assemblies. ``walk`` yields ``(path, block)`` pairs;
``find`` filters by metadata predicates; ``flatten`` collapses every
leaf to a single-level MultiBlock with path-named blocks.

This example walks a real BIM model, the buildingSMART
"Single-family house" sample (CC-BY 4.0, buildingSMART International),
and finds all walls by their IFC type, the canonical first step in
any BIM analysis.
"""

# %%

import pandas as pd
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Read the real IFC building.

root = pyvista_cad.read_ifc(downloads.ifc_building_path())
root

# %%
# Drop stray geo-reference / survey markers placed far from the house.
# ``drop_spatial_outliers`` uses a robust MAD test on block position
# (markers have no reliable IFC class) and preserves the hierarchy, so
# the walk below still shows the real building tree.

root = root.cad.drop_spatial_outliers()

# %%
# Walk the tree. Path, IFC type, and cell count.

rows = []
for path, block in root.cad.walk():
    n_cells = (
        block.n_cells
        if isinstance(block, pv.DataSet)
        else sum(b.n_cells for b in block if isinstance(b, pv.DataSet))
    )
    rows.append(
        {
            'path': str(path),
            'kind': type(block).__name__,
            'ifc_type': block.field_data.get('cad.ifc_type', [b''])[0],
            'cells': n_cells,
        }
    )
pd.DataFrame(rows)

# %%
# Find every wall.

walls = root.cad.find(ifc_type='IfcWall')
pd.DataFrame({'path': [str(p) for p, _ in walls]})

# %%
# Flatten the tree: every leaf becomes a path-named block.

list(root.cad.flatten().keys())

# %%
# Render the building with each element in its own color.

pl = pv.Plotter()
pl.add_mesh(root, multi_colors=True, show_edges=True, edge_color='black')
pl.show()
