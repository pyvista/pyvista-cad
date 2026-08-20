"""IGES (``.iges``, ``.igs``) reader.

Two read backends are supported:

- ``'pyiges'`` (default): pure-Python parse plus geomdl tessellation.
  Recovers the per-entity IGES *level*. Requires ``pyvista-cad[iges]``.
- ``'ocp'``: OCCT's ``IGESControl_Reader``. Several times faster on
  large files, and it clips trimmed surfaces (IGES type 144) to their
  trimming curves, which pyiges does not. No level metadata. Requires
  ``pyvista-cad[step]``.

The default stays ``'pyiges'`` so existing reads are unchanged;
``'ocp'`` is opt-in.
"""

import os
from typing import TYPE_CHECKING, Any, Literal

from pyvista_cad._errors import OptionalDependencyError
from pyvista_cad._uri import require_local_path

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
    backend: Literal['pyiges', 'ocp'] = 'pyiges',
    bsplines: bool = True,
    surfaces: bool = True,
    lines: bool = False,
    points: bool = False,
    delta: float = 0.025,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    **_: Any,
) -> 'pv.DataSet':
    """Read an IGES file and return a pyvista dataset.

    Parameters
    ----------
    path : str or os.PathLike
        Path to an IGES (``.iges`` / ``.igs``) file.
    backend : {'pyiges', 'ocp'}, default: 'pyiges'
        Which reader to use. ``'pyiges'`` wraps
        ``pyiges.read(path).to_vtk(...)`` and recovers the per-entity
        IGES level. ``'ocp'`` wraps OCCT's ``IGESControl_Reader``: much
        faster on large files and it respects trimmed-surface
        boundaries, but it recovers no level metadata.
    bsplines : bool, default: True
        Tessellate BSpline surfaces. ``'pyiges'`` backend only.
    surfaces : bool, default: True
        Include analytic surfaces. ``'pyiges'`` backend only.
    lines : bool, default: False
        Include line/curve entities. ``'pyiges'`` backend only.
    points : bool, default: False
        Include isolated point entities. ``'pyiges'`` backend only.
    delta : float, default: 0.025
        Tessellation step for geomdl. ``'pyiges'`` backend only.
    linear_deflection : float, default: 0.1
        OCCT chordal tessellation tolerance (model units). ``'ocp'``
        backend only.
    angular_deflection : float, default: 0.5
        OCCT angular tessellation tolerance (radians). ``'ocp'``
        backend only.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.DataSet
        Tessellated dataset. ``'pyiges'`` returns ``PolyData`` or
        ``MultiBlock`` depending on the IGES content; ``'ocp'`` always
        returns ``PolyData``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    pyvista_cad.CadReadError
        If parsing fails.
    pyvista_cad.OptionalDependencyError
        If the selected backend is not installed, or ``backend`` names
        an unknown reader.

    """
    require_local_path(path)

    if backend == 'pyiges':
        from pyvista_cad._backends._pyiges import read_iges_internal

        return read_iges_internal(
            path,
            bsplines=bsplines,
            surfaces=surfaces,
            lines=lines,
            points=points,
            delta=delta,
        )
    if backend == 'ocp':
        from pyvista_cad._backends._ocp_iges import read_iges as _read_ocp

        return _read_ocp(
            path,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
    msg = f'unknown IGES backend {backend!r}'
    raise OptionalDependencyError(msg)
