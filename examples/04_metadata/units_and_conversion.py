"""
Units and unit conversion
=========================

The ``cad.units`` field-data slot records the document unit on read
and write. ``units_conversion_factor`` returns the scalar to multiply
points by to reach a target unit, so an mm-defined part can move into
meters (for FEA) or inches (for US-customary CAM) without leaving the
pipeline.

This example reads a real mm-defined STEP part, the NIST AM Bench
2022 LPBF bridge specimen (public domain,
https://doi.org/10.18434/mds2-2607), converts it to inches, and
confirms the bounding box and volume scale accordingly.
"""

# %%

import numpy as np
import pandas as pd
import pyvista as pv

import pyvista_cad
from pyvista_cad._metadata import units_conversion_factor
from pyvista_cad.examples import downloads

# %%
# Read the real part. Verify the source unit.

mb = pyvista_cad.read_step(downloads.step_part_path())
mesh = mb.combine().extract_surface()
src_unit = str(mesh.cad.units) if mesh.cad.units else 'mm'
pd.DataFrame({'source_unit': [src_unit], 'bounds': [mesh.bounds]})

# %%
# Convert mm -> inch. The scalar factor is 1/25.4.

factor = units_conversion_factor(src_unit, 'inch')

converted = mesh.copy()
converted.points = converted.points * factor
converted.field_data['cad.units'] = np.array(['inch'])
pd.DataFrame(
    {
        'factor': [factor],
        'bounds_source': [mesh.bounds],
        'bounds_inch': [converted.bounds],
    }
)

# %%
# Verify volumes scale by ``factor**3`` (the volume invariant for
# isotropic scaling).

pd.DataFrame(
    {
        'volume_ratio': [converted.volume / mesh.volume if mesh.volume > 0 else float('nan')],
        'expected': [factor**3],
    }
)

# %%
# Render the converted mesh.

converted['Z (in)'] = converted.points[:, 2]
pl = pv.Plotter()
pl.cad.add(converted, scalars='Z (in)', cmap='viridis')
pl.show()
