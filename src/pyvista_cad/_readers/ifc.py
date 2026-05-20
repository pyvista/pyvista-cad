"""IFC (`.ifc`) reader."""

import os
from typing import Any

import pyvista as pv

from pyvista_cad._uri import require_local_path


@pv.register_reader('.ifc')
def read_ifc(
    path: str | os.PathLike[str],
    /,
    *,
    include: tuple[str, ...] | None = None,
    use_world_coords: bool = True,
    **_: Any,
) -> pv.MultiBlock:
    """Read an IFC file as a :class:`pyvista.MultiBlock`.

    Each IFC product with renderable geometry becomes one block.
    Block names follow ``<element name>_<index>`` so they are unique
    even when names collide. Per-element ``cad.guid``,
    ``cad.ifc_type``, ``cad.label``, and ``cad.material`` (with
    optional ``cad.color``) field data are attached.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.ifc`` file (IFC2X3 / IFC4 / IFC4X3).
    include : tuple of str or None, default: None
        Restrict to specific IFC classes, e.g. ``('IfcWall', 'IfcSlab')``.
        When ``None``, every product with geometry is exported.
    use_world_coords : bool, default: True
        Apply each element's local placement so the result is in world
        coordinates.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.MultiBlock
        Tessellated geometry, one block per element.

    Examples
    --------
    Read the bundled small-building IFC fixture:

    >>> import pyvista_cad
    >>> from pyvista_cad import read_ifc
    >>> mb = read_ifc(pyvista_cad.examples.small_building_ifc_path())
    >>> type(mb).__name__
    'MultiBlock'
    >>> str(mb.field_data['cad.source_format'][0])
    'ifc'

    """
    require_local_path(path)
    from pyvista_cad._backends._ifcopenshell import read_ifc_internal

    return read_ifc_internal(path, include=include, use_world_coords=use_world_coords)
