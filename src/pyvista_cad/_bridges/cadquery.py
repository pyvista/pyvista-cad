"""Bridge to and from cadquery objects."""

from typing import TYPE_CHECKING, Any

import numpy as np

from pyvista_cad._conversion import (
    polydata_to_topods,
    topods_to_polydata,
    unwrap_to_topods,
)
from pyvista_cad._errors import OptionalDependencyError

if TYPE_CHECKING:
    import pyvista as pv


def from_cadquery(
    workplane: Any,
    /,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> 'pv.PolyData':
    """Convert a :class:`cadquery.Workplane` or ``Compound`` to PyVista.

    The geometry is unwrapped to an OCCT ``TopoDS`` shape and
    tessellated at the requested deflections.

    Parameters
    ----------
    workplane : cadquery.Workplane or cadquery.Compound
        Input geometry.
    linear_deflection : float, default: 0.1
        OCCT linear deflection (model units).
    angular_deflection : float, default: 0.5
        OCCT angular deflection (radians).

    Returns
    -------
    pyvista.PolyData
        Triangulated geometry. The ``cad.source_format`` field array is
        set to ``'cadquery'``.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If :mod:`cadquery` is not installed.

    See Also
    --------
    to_cadquery : Inverse conversion.

    Examples
    --------
    >>> import cadquery as cq
    >>> from pyvista_cad import from_cadquery
    >>> box = cq.Workplane('XY').box(1, 1, 1)
    >>> poly = from_cadquery(box)
    >>> poly.n_cells > 0
    True

    """
    try:
        import cadquery  # noqa: F401
    except ImportError as exc:
        msg = 'from_cadquery requires cadquery; install pyvista-cad[cadquery]'
        raise OptionalDependencyError(msg) from exc

    topods = unwrap_to_topods(workplane)
    poly = topods_to_polydata(
        topods,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    poly.field_data['cad.source_format'] = np.array(['cadquery'])
    return poly


def to_cadquery(mesh: 'pv.PolyData') -> Any:
    """Wrap a :class:`pyvista.PolyData` as a faceted ``cadquery.Workplane``.

    The mesh is converted to an OCCT ``TopoDS`` shell and wrapped in a
    ``cadquery.Compound`` on a default ``XY`` workplane.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh to convert.

    Returns
    -------
    cadquery.Workplane
        Workplane wrapping a faceted ``Compound``. Analytical surface
        information is lost.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If :mod:`cadquery` is not installed.

    See Also
    --------
    from_cadquery : Inverse conversion.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista_cad import to_cadquery
    >>> wp = to_cadquery(pv.Sphere())
    >>> wp.val().Volume() > 0
    True

    """
    try:
        import cadquery as cq
    except ImportError as exc:
        msg = 'to_cadquery requires cadquery; install pyvista-cad[cadquery]'
        raise OptionalDependencyError(msg) from exc

    topods = polydata_to_topods(mesh)
    compound = cq.Compound(topods)
    return cq.Workplane('XY').add(compound)
