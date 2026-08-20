"""IGES backend powered by OCCT via OCP.

OCCT's ``IGESControl_Reader`` parses the file in C++ and hands back a
``TopoDS_Shape``, which the shared :func:`topods_to_polydata` workhorse
tessellates. Two consequences distinguish this path from the pyiges
one:

- It is faster, because neither the parse nor the surface evaluation
  runs in Python. On the 4 MB example impeller the gap is about 5x at
  matched tessellation density and about 36x at the two backends'
  respective defaults, which do not produce the same mesh density. See
  the README for the measurements.
- It honors IGES trimmed-surface entities (type 144): the tessellation
  is clipped to the trimming curves. pyiges dispatches type 128 and has
  no type-144 handler, so it tessellates the full underlying surface
  and can emit geometry outside the part's real boundary.

Per-entity IGES *level* metadata is not recovered here; that stays a
pyiges-backend feature (see ``_pyiges.py``).
"""

import os

import numpy as np
import pyvista as pv

from pyvista_cad._conversion import topods_to_polydata
from pyvista_cad._errors import CadReadError, OptionalDependencyError

_BACKEND_NAME = 'ocp'


def read_iges(
    path: str | os.PathLike[str],
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> pv.PolyData:
    """Read an IGES file via OCCT and tessellate it to a ``PolyData``.

    Parameters
    ----------
    path : str or os.PathLike
        Path to an IGES (``.iges`` / ``.igs``) file.
    linear_deflection : float, default: 0.1
        OCCT chordal tessellation tolerance (model units).
    angular_deflection : float, default: 0.5
        OCCT angular tessellation tolerance (radians).

    Returns
    -------
    pyvista.PolyData
        Tessellated IGES geometry, with the originating
        ``TopoDS_Shape`` cached on the result so ``.cad`` can recover
        topological edges.

    """
    try:
        from OCP.IFSelect import IFSelect_ReturnStatus
        from OCP.IGESControl import IGESControl_Reader
    except ImportError as exc:
        msg = 'OCP not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc

    fname = os.fspath(path)
    if not os.path.exists(fname):
        msg = f'No such file: {fname}'
        raise FileNotFoundError(msg)

    reader = IGESControl_Reader()
    try:
        status = reader.ReadFile(fname)
    except Exception as exc:  # OCCT raises assorted Standard_Failure subclasses
        msg = f'OCCT failed to read IGES {fname}: {exc}'
        raise CadReadError(msg) from exc
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        msg = f'OCCT failed to read IGES {fname}: reader status {status}'
        raise CadReadError(msg)

    # TransferRoots returns the number of roots it mapped to shapes.
    # Zero means the file parsed but carried nothing this reader can
    # turn into geometry: an empty model, or one holding only entities
    # OCCT's IGES processor does not transfer (drafting, annotation,
    # structure). Raising beats handing back an empty PolyData that
    # looks like a successful read of a part with no faces. A file that
    # transfers no roots also yields a null OneShape, so this covers
    # the null case too.
    if reader.TransferRoots() == 0:
        msg = f'OCCT transferred no geometry from IGES {fname}'
        raise CadReadError(msg)

    mesh = topods_to_polydata(
        reader.OneShape(),
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    mesh.field_data['cad.source_format'] = np.array(['iges'])
    mesh.field_data['cad.backend'] = np.array([_BACKEND_NAME])
    return mesh
