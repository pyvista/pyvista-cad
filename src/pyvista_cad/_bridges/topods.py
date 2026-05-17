"""Raw OCCT (``OCP``) bridges.

Tessellates ``OCP.TopoDS.TopoDS_Shape`` to :class:`pyvista.PolyData`
and rebuilds a faceted ``TopoDS_Compound`` from a PolyData.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from pyvista_cad._conversion import polydata_to_topods, topods_to_polydata
from pyvista_cad._errors import OptionalDependencyError

if TYPE_CHECKING:
    import pyvista as pv


def from_topods(
    shape: Any,
    /,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> 'pv.PolyData':
    """Tessellate a raw OCCT ``TopoDS_Shape`` into a :class:`pyvista.PolyData`.

    Parameters
    ----------
    shape : OCP.TopoDS.TopoDS_Shape
        The OCCT shape to tessellate.
    linear_deflection : float, default: 0.1
        OCCT tessellation linear deflection (model units).
    angular_deflection : float, default: 0.5
        OCCT tessellation angular deflection (radians).

    Returns
    -------
    pyvista.PolyData
        Triangulated geometry.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If OCP is not installed.

    Examples
    --------
    Tessellate a raw OCCT shape from a build123d box (requires the
    ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad import from_topods
    >>> shape = b3d.Box(10, 10, 10).wrapped
    >>> mesh = from_topods(shape)
    >>> type(mesh).__name__
    'PolyData'
    >>> [round(b) for b in mesh.bounds]
    [-5, 5, -5, 5, -5, 5]

    """
    try:
        import OCP  # noqa: F401
    except ImportError as exc:
        msg = 'OCP not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc

    poly = topods_to_polydata(
        shape, linear_deflection=linear_deflection, angular_deflection=angular_deflection
    )
    poly.field_data['cad.source_format'] = np.array(['OCP'])
    return poly


def to_topods(mesh: 'pv.PolyData') -> Any:
    """Build a faceted ``TopoDS_Compound`` from a :class:`pyvista.PolyData`.

    Each triangle becomes a planar OCCT face. Analytical surface
    information is lost.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh to convert.

    Returns
    -------
    OCP.TopoDS.TopoDS_Compound
        Faceted compound.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If OCP is not installed.

    Examples
    --------
    Build a faceted OCCT compound from a PyVista sphere (requires the
    ``[step]`` extra):

    >>> import pyvista as pv
    >>> from pyvista_cad import to_topods
    >>> shape = to_topods(pv.Sphere())
    >>> type(shape).__name__
    'TopoDS_Solid'

    """
    try:
        import OCP  # noqa: F401
    except ImportError as exc:
        msg = 'OCP not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc

    return polydata_to_topods(mesh)
