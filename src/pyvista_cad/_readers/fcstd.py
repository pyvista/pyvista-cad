"""FreeCAD document (``.FCStd``) reader."""

import os
from typing import Any

import pyvista as pv


@pv.register_reader('.fcstd')
def read_fcstd(
    path: str | os.PathLike[str],
    /,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    **_: Any,
) -> pv.MultiBlock:
    """Read a FreeCAD document as a :class:`pyvista.MultiBlock`.

    FCStd files are zip archives carrying a ``Document.xml`` manifest
    plus one BREP file per part. The FreeCAD binary is **not**
    required; only OCP is used to tessellate the BREPs.

    Each part becomes one block. ``field_data['cad.label']`` carries
    the FreeCAD label, ``cad.color`` carries the shape color when
    present, and ``cad.source_format`` is ``'fcstd'``.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.FCStd`` file.
    linear_deflection : float, default: 0.1
        OCCT linear deflection (model units).
    angular_deflection : float, default: 0.5
        OCCT angular deflection (radians).
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.MultiBlock
        One block per FreeCAD part.

    """
    from pyvista_cad._backends._freecad import read_fcstd_internal

    return read_fcstd_internal(
        path,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
