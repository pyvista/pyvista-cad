"""Bridges to and from build123d.

Translates between :mod:`build123d` shapes / compounds and PyVista
datasets, riding on the shared OCCT tessellator in
:mod:`pyvista_cad._conversion`.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_cad._conversion import (
    polydata_to_topods,
    topods_to_polydata,
    unwrap_to_topods,
)
from pyvista_cad._errors import OptionalDependencyError


def _color_rgb(obj: Any) -> tuple[float, ...] | None:
    """Extract an RGB or RGBA color from a build123d object.

    A modern :class:`build123d.Color` is iterable, yielding four
    floats ``(r, g, b, a)`` in ``[0, 1]``; it exposes neither
    ``.r/.g/.b`` attributes nor item access. Legacy color objects
    carrying ``.r/.g/.b`` are still handled for forward compatibility.

    Parameters
    ----------
    obj : Any
        build123d shape or compound child. Its ``color`` attribute
        is read; anything else is ignored.

    Returns
    -------
    tuple of float or None
        ``(r, g, b)`` or ``(r, g, b, a)`` in ``[0, 1]``, matching the
        ``cad.color`` contract. ``None`` when no usable color exists.

    """
    color = getattr(obj, 'color', None)
    if color is None:
        return None

    r = getattr(color, 'r', None)
    g = getattr(color, 'g', None)
    b = getattr(color, 'b', None)
    if r is not None and g is not None and b is not None:
        try:
            return (float(r), float(g), float(b))
        except (TypeError, ValueError):
            return None

    if isinstance(color, Iterable) and not isinstance(color, (str, bytes)):
        try:
            components = [float(c) for c in color]
        except (TypeError, ValueError):
            return None
        if len(components) >= 4:
            return tuple(components[:4])
        if len(components) == 3:
            return tuple(components[:3])
        return None

    return None


def from_build123d(
    shape: Any,
    /,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    pass_data: bool = True,
) -> pv.PolyData | pv.MultiBlock:
    """Convert a :mod:`build123d` ``Shape`` or ``Compound`` to PyVista.

    Parameters
    ----------
    shape : build123d.Shape or build123d.Compound
        Input geometry.
    linear_deflection : float, default: 0.1
        OCCT tessellation linear deflection (model units).
    angular_deflection : float, default: 0.5
        OCCT tessellation angular deflection (radians).
    pass_data : bool, default: True
        When ``True``, propagate build123d color and label onto
        ``cad.color`` and ``cad.label`` field-data.

    Returns
    -------
    pyvista.PolyData or pyvista.MultiBlock
        ``PolyData`` for a single shape; ``MultiBlock`` for a
        ``Compound`` with multiple children.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If :mod:`build123d` is not installed.

    Examples
    --------
    Convert a build123d box to PyVista (requires the ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad import from_build123d
    >>> box = b3d.Box(10, 10, 10)
    >>> mesh = from_build123d(box)
    >>> type(mesh).__name__
    'PolyData'
    >>> [round(b) for b in mesh.bounds]
    [-5, 5, -5, 5, -5, 5]

    """
    try:
        import build123d as b3d
    except ImportError as exc:
        msg = 'build123d not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc

    if isinstance(shape, b3d.Compound):
        children = list(shape.children) if getattr(shape, 'children', None) else []
        if children:
            mb = pv.MultiBlock()
            for child in children:
                topods = unwrap_to_topods(child)
                poly = topods_to_polydata(
                    topods,
                    linear_deflection=linear_deflection,
                    angular_deflection=angular_deflection,
                )
                label = getattr(child, 'label', '') or f'part_{mb.n_blocks}'
                if pass_data:
                    color = _color_rgb(child)
                    if color is not None:
                        poly.field_data['cad.color'] = np.asarray(color, dtype=float)
                    poly.field_data['cad.label'] = np.array([label])
                poly.field_data['cad.source_format'] = np.array(['build123d'])
                mb.append(poly, name=label)
            mb.field_data['cad.source_format'] = np.array(['build123d'])
            return mb

    topods = unwrap_to_topods(shape)
    poly = topods_to_polydata(
        topods,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    if pass_data:
        color = _color_rgb(shape)
        if color is not None:
            poly.field_data['cad.color'] = np.asarray(color, dtype=float)
        label = getattr(shape, 'label', '') or ''
        if label:
            poly.field_data['cad.label'] = np.array([label])
    poly.field_data['cad.source_format'] = np.array(['build123d'])
    return poly


def to_build123d(mesh: pv.PolyData) -> Any:
    """Wrap a :class:`pyvista.PolyData` as a faceted ``build123d.Compound``.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh to convert.

    Returns
    -------
    build123d.Compound
        Faceted compound. Analytical surface data is lost; chained
        fillets / shells will not behave as on parametric geometry.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If :mod:`build123d` is not installed.

    Examples
    --------
    Wrap a PyVista sphere as a build123d compound (requires the
    ``[step]`` extra):

    >>> import pyvista as pv
    >>> from pyvista_cad import to_build123d
    >>> compound = to_build123d(pv.Sphere())
    >>> type(compound).__name__
    'Compound'

    """
    try:
        import build123d as b3d
    except ImportError as exc:
        msg = 'build123d not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc

    topods = polydata_to_topods(mesh)
    return b3d.Compound(topods)
