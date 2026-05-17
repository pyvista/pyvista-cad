"""BREP (`.brep`, `.brp`) reader and writer (via OCP)."""

import os
from typing import Any

import pyvista as pv

from pyvista_cad._errors import CadReadError, CadWriteError, OptionalDependencyError


def _require_ocp() -> Any:
    try:
        import OCP  # noqa: F401
    except ImportError as exc:
        msg = 'OCP not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc
    return None


@pv.register_reader('.brep')
@pv.register_reader('.brp')
def read_brep(
    path: str | os.PathLike[str],
    /,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    **_: Any,
) -> pv.PolyData:
    """Read a BREP file as a :class:`pyvista.PolyData`.

    BREP is OpenCASCADE's native shape serialization. The file is
    loaded with ``BRepTools.Read_s`` and tessellated with
    ``BRepMesh_IncrementalMesh``.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.brep`` / ``.brp`` file.
    linear_deflection : float, default: 0.1
        OCCT linear deflection (model units).
    angular_deflection : float, default: 0.5
        OCCT angular deflection (radians).
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.PolyData
        Tessellated surface mesh.

    """
    _require_ocp()
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape

    from pyvista_cad._bridges.topods import from_topods

    shape = TopoDS_Shape()
    builder = BRep_Builder()
    try:
        ok = BRepTools.Read_s(shape, str(path), builder)
    except Exception as exc:
        msg = f'OCP failed to read BREP {path}: {exc}'
        raise CadReadError(msg) from exc
    if not ok or shape.IsNull():
        msg = f'OCP could not parse BREP file: {path}'
        raise CadReadError(msg)

    poly = from_topods(
        shape,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    import numpy as np

    # ``from_topods`` -> ``topods_to_polydata`` already registers the
    # originating ``TopoDS_Shape`` in ``_conversion._BREP_CACHE``, so
    # ``get_cached_topods(poly)`` will return the exact-curve shape and
    # downstream consumers (e.g. ``CadShape.tessellate``) can skip the
    # re-tessellation round-trip.
    poly.field_data['cad.source_format'] = np.array(['brep'])
    poly.field_data['cad.backend'] = np.array(['ocp'])
    poly.field_data['cad.has_brep_origin'] = np.array([True])
    return poly


def write_brep(
    mesh: pv.PolyData,
    path: str | os.PathLike[str],
    /,
    **_: Any,
) -> None:
    """Write a :class:`pyvista.PolyData` as a faceted BREP file.

    The mesh is converted to a ``TopoDS_Compound`` of triangular
    faces and serialized with ``BRepTools.Write_s``.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Source mesh.
    path : str or os.PathLike
        Destination ``.brep`` / ``.brp`` path.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Raises
    ------
    pyvista_cad.CadWriteError
        If ``mesh`` is not a :class:`pyvista.PolyData`, or if OCCT's
        ``BRepTools.Write_s`` reports a failed write (for example an
        unwritable destination path): the failure is never silent.
    pyvista_cad.OptionalDependencyError
        If OCP is not installed.

    """
    _require_ocp()
    from OCP.BRepTools import BRepTools

    from pyvista_cad._conversion import polydata_to_topods

    if not isinstance(mesh, pv.PolyData):
        msg = f'write_brep requires a pv.PolyData; got {type(mesh).__name__}'
        raise CadWriteError(msg)

    shape = polydata_to_topods(mesh)
    # ``BRepTools.Write_s`` returns a bool status and never raises on an
    # unwritable destination: it just returns False and writes nothing.
    # Check the status (and that the file actually materialised) so a
    # failed write surfaces as a CadWriteError instead of vanishing.
    ok = BRepTools.Write_s(shape, str(path))
    if not ok or not os.path.exists(path):
        msg = f'OCP failed to write BREP {path} (BRepTools.Write_s returned {ok!r})'
        raise CadWriteError(msg)
