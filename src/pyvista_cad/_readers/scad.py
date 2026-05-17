"""OpenSCAD source (``.scad``) reader."""

import os
from typing import Any

import pyvista as pv


@pv.register_reader('.scad')
def read_scad(
    path: str | os.PathLike[str],
    /,
    *,
    binary: str | None = None,
    args: tuple[str, ...] = (),
    timeout: float | None = None,
    **_: Any,
) -> pv.PolyData:
    """Compile an OpenSCAD source file and return the resulting mesh.

    The OpenSCAD CLI compiles ``.scad`` to STL in a temp directory;
    the STL is then loaded as a :class:`pyvista.PolyData`. The
    ``openscad`` binary must be installed and on ``PATH`` (or pass
    ``binary=`` to point at it).

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.scad`` source file.
    binary : str or None, optional
        Explicit path to the openscad executable.
    args : tuple of str, optional
        Extra CLI arguments forwarded to openscad (e.g. ``('-D', 'n=20')``).
        The fast ``--backend=manifold`` kernel (OpenSCAD >= 2024) is used
        automatically, with a transparent fallback to the default CGAL
        kernel on older builds. Pass an explicit ``--backend=...`` here to
        override.
    timeout : float or None, optional
        Subprocess timeout in seconds.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.PolyData
        Tessellated mesh.

    """
    from pyvista_cad._backends._openscad import read_scad_internal

    return read_scad_internal(path, binary=binary, args=args, timeout=timeout)
