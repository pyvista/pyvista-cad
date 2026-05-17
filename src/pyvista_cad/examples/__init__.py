"""Bundled example CAD files.

Provides module-level attributes pointing at small fixture files
shipped with the package:

- :data:`dxf_plate`: a small DXF drawing with two layers.
- :data:`step_cube`: a 10 mm cube exported to STEP via build123d
  (committed; no build123d needed at read time).
- :func:`bracket_step_path`: parametric L-bracket (50x30x10 mm) with
  two M6 through-holes and a 3 mm fillet at the inside corner.
- :func:`drawing_dxf_path`: 2D mechanical drawing of a bracket with
  layered body, holes, centerlines, and annotations.
- :func:`assembly_3mf_path`: two-part 3MF assembly (red base plate +
  four blue standoffs) with per-object colors and per-item transforms.
- :func:`small_building_ifc_path`: IFC4 file with Site > Building >
  Storey > 4 walls + 1 slab and property sets.

Use them with :func:`pyvista.read`:

>>> import pyvista as pv
>>> import pyvista_cad
>>> import os
>>> os.path.basename(pyvista_cad.examples.dxf_plate)
'plate.dxf'

"""

from pathlib import Path

from pyvista_cad.examples import downloads

_DATA_DIR = Path(__file__).parent / 'data'

dxf_plate: str = str(_DATA_DIR / 'plate.dxf')
step_cube: str = str(_DATA_DIR / 'cube.step')


def bracket_step_path() -> str:
    """Return the path to the bundled L-bracket STEP fixture.

    Returns
    -------
    str
        Absolute filesystem path to ``bracket.step``.

    """
    return str(_DATA_DIR / 'bracket.step')


def drawing_dxf_path() -> str:
    """Return the path to the bundled 2D mechanical drawing DXF fixture.

    Returns
    -------
    str
        Absolute filesystem path to ``drawing.dxf``.

    """
    return str(_DATA_DIR / 'drawing.dxf')


def assembly_3mf_path() -> str:
    """Return the path to the bundled two-part 3MF assembly fixture.

    Returns
    -------
    str
        Absolute filesystem path to ``assembly.3mf``.

    """
    return str(_DATA_DIR / 'assembly.3mf')


def small_building_ifc_path() -> str:
    """Return the path to the bundled small building IFC fixture.

    Returns
    -------
    str
        Absolute filesystem path to ``small_building.ifc``.

    """
    return str(_DATA_DIR / 'small_building.ifc')


__all__ = [
    'assembly_3mf_path',
    'bracket_step_path',
    'downloads',
    'drawing_dxf_path',
    'dxf_plate',
    'small_building_ifc_path',
    'step_cube',
]
