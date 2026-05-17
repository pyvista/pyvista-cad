"""IGES (``.iges``, ``.igs``) reader via pyiges."""

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyvista as pv


# Registration glue for IGES lives outside this module: ``pyvista_cad``
# eagerly registers a lightweight ``.iges``/``.igs`` trampoline (with
# ``override=True``) so this metadata-aware reader deterministically
# wins over ``pyiges``'s own ``pyvista.readers`` entry point and no
# multi-provider warning fires. Decorating ``read_iges`` here too would
# re-register the same extension when this module is imported and emit a
# "replaces an existing custom reader" warning.
def read_iges(
    path: str | os.PathLike[str],
    /,
    *,
    bsplines: bool = True,
    surfaces: bool = True,
    lines: bool = False,
    points: bool = False,
    delta: float = 0.025,
    **_: Any,
) -> 'pv.DataSet':
    """Read an IGES file using pyiges and return a pyvista dataset.

    Wraps ``pyiges.read(path).to_vtk(...)``. Requires
    ``pyvista-cad[iges]`` (pyiges + geomdl).

    Parameters
    ----------
    path : str or os.PathLike
        Path to an IGES (``.iges`` / ``.igs``) file.
    bsplines : bool, default: True
        Tessellate BSpline surfaces.
    surfaces : bool, default: True
        Include analytic surfaces.
    lines : bool, default: False
        Include line/curve entities.
    points : bool, default: False
        Include isolated point entities.
    delta : float, default: 0.025
        Tessellation step for geomdl.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.DataSet
        Tessellated dataset (``PolyData`` or ``MultiBlock`` depending
        on the IGES content).

    """
    from pyvista_cad._backends._pyiges import read_iges_internal

    return read_iges_internal(
        path,
        bsplines=bsplines,
        surfaces=surfaces,
        lines=lines,
        points=points,
        delta=delta,
    )
