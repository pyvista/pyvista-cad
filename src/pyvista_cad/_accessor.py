"""The ``.cad`` accessor.

Loaded via the ``pyvista.accessors`` entry-point group; importing
:mod:`pyvista_cad` registers the ``.cad`` attribute on every
:class:`pyvista.DataSet` and :class:`pyvista.MultiBlock`.

Two distinct accessor classes are registered:

* :class:`CadDataSetAccessor` for :class:`pyvista.DataSet` subclasses.
* :class:`CadMultiBlockAccessor` for :class:`pyvista.MultiBlock`.

The two classes share the metadata surface (units, label, color,
source-format, ...) by duplication rather than inheritance.
"""

from collections.abc import Callable, Iterator
import os
from typing import Any
import warnings

import numpy as np
import pyvista as pv
from pyvista import DataSet, MultiBlock

from pyvista_cad._metadata import (
    CadMetadata,
    units_conversion_factor,
    validate_units,
)

# --------------------------------------------------------------------------- #
# Module-level helpers (shared between both accessor classes).
# --------------------------------------------------------------------------- #


def _read_str_field(field_data: Any, key: str) -> str | None:
    if key not in field_data:
        return None
    blob = field_data[key]
    if hasattr(blob, '__len__') and len(blob) > 0:
        return str(blob[0])
    return str(blob)


def _block_layers(block: Any) -> set[str]:
    cd = getattr(block, 'cell_data', None)
    if cd is None:
        return set()
    for key in ('cad.layer', 'Layer'):
        if key in cd:
            return {str(x) for x in np.asarray(cd[key])}
    return set()


def _resolve_cell_array(ds: Any, key: str, fallback: str | None = None) -> np.ndarray | None:
    cd = getattr(ds, 'cell_data', None)
    if cd is None:
        return None
    if key in cd:
        return np.asarray(cd[key])
    if fallback is not None and fallback in cd:
        return np.asarray(cd[fallback])
    return None


def _as_polydata(ds: Any) -> pv.PolyData:
    if isinstance(ds, pv.MultiBlock):
        ds = ds.combine(merge_points=True)
    if not isinstance(ds, pv.PolyData):
        ds = ds.extract_surface(algorithm='dataset_surface')
    return ds


def _get_cached_topods(mesh: Any) -> Any | None:
    """Return the cached originating ``TopoDS_Shape`` for ``mesh``, or ``None``.

    A genuinely absent OCCT backend (``ImportError`` on ``OCP``) yields
    ``None``: the mesh has no recoverable B-rep origin and that is
    legitimate. An entry that *exists* but is not a valid, non-null
    ``TopoDS_Shape`` is a corrupt cache and raises
    :class:`~pyvista_cad.CadCacheError` rather than masquerading as a
    plain mesh with no CAD origin.
    """
    from pyvista_cad._conversion import get_cached_topods

    shape = get_cached_topods(mesh)
    if shape is None:
        return None
    # pragma reason: fires only with the OCP C-extension uninstalled;
    # unreachable while OCCT is a hard test dependency.
    try:
        from OCP.TopoDS import TopoDS_Shape
    except ImportError:  # pragma: no cover
        return None
    if not isinstance(shape, TopoDS_Shape) or shape.IsNull():
        from pyvista_cad._errors import CadCacheError

        msg = (
            f'corrupt B-rep cache: cache entry is {type(shape).__name__}, '
            f'not a valid non-null TopoDS_Shape'
        )
        raise CadCacheError(msg)
    return shape


def _collect_cached_topods(mb: Any) -> Any | None:
    """Recover a cached B-rep for a MultiBlock by walking its leaves.

    ``MultiBlock.combine()`` discards the per-leaf B-rep cache, so the
    combined surface alone cannot be re-tessellated. Gather each leaf's
    cached ``TopoDS_Shape`` into a single ``TopoDS_Compound`` so the
    assembly re-tessellates as a whole. Returns ``None`` only when no
    leaf carries a cache entry; a corrupt entry still raises via
    :func:`_get_cached_topods`.
    """
    shapes: list[Any] = []
    for block in mb:
        if block is None:
            continue
        if isinstance(block, pv.MultiBlock):
            nested = _collect_cached_topods(block)
            if nested is not None:
                shapes.append(nested)
            continue
        leaf = _get_cached_topods(block)
        if leaf is not None:
            shapes.append(leaf)
    if not shapes:
        return None
    if len(shapes) == 1:
        return shapes[0]
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shp in shapes:
        builder.Add(compound, shp)
    return compound


def _topods_props(shape: Any) -> Any:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props


# --------------------------------------------------------------------------- #
# Common metadata mixin assembled via direct attribute assignment after
# the class bodies. Both accessor classes get the same property objects
# without subclassing (functionally identical to copy-paste, but DRY).
# --------------------------------------------------------------------------- #


def _make_source_format() -> property:
    def fget(self: Any) -> str | None:
        return _read_str_field(self._dataset.field_data, 'cad.source_format')

    return property(
        fget,
        doc="""Originating CAD format ('STEP', 'DXF', ...) or ``None``.

        Returns
        -------
        str or None
            The ``cad.source_format`` field-data string, or ``None``
            when the dataset carries no origin stamp.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> pv.Sphere().cad.source_format is None
        True

        """,
    )


def _make_units() -> property:
    def fget(self: Any) -> str | None:
        return _read_str_field(self._dataset.field_data, 'cad.units')

    def fset(self: Any, value: str | None) -> None:
        if value is None:
            self._dataset.field_data.pop('cad.units', None)
            return
        validate_units(value)
        self._dataset.field_data['cad.units'] = np.array([value])

    return property(
        fget,
        fset,
        doc="""Canonical unit string or ``None``.

        Returns
        -------
        str or None
            The ``cad.units`` field-data string, or ``None``.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Sphere()
        >>> mesh.cad.units = 'mm'
        >>> mesh.cad.units
        'mm'

        """,
    )


def _make_label() -> property:
    def fget(self: Any) -> str | None:
        return _read_str_field(self._dataset.field_data, 'cad.label')

    def fset(self: Any, value: str | None) -> None:
        if value is None:
            self._dataset.field_data.pop('cad.label', None)
            return
        self._dataset.field_data['cad.label'] = np.array([str(value)])

    return property(
        fget,
        fset,
        doc="""Dataset-level label, or ``None``.

        Returns
        -------
        str or None
            The ``cad.label`` field-data string, or ``None``.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Sphere()
        >>> mesh.cad.label = 'widget'
        >>> mesh.cad.label
        'widget'

        """,
    )


def _make_color() -> property:
    def fget(self: Any) -> tuple[float, float, float] | None:
        fd = self._dataset.field_data
        if 'cad.color' not in fd:
            return None
        arr = np.asarray(fd['cad.color']).ravel()
        if arr.size < 3:
            return None
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    def fset(self: Any, value: tuple[float, float, float] | None) -> None:
        if value is None:
            self._dataset.field_data.pop('cad.color', None)
            return
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != 3:
            msg = f'color must be a 3-tuple of floats; got shape {arr.shape}'
            raise ValueError(msg)
        self._dataset.field_data['cad.color'] = arr

    return property(
        fget,
        fset,
        doc="""Dataset-level RGB color, or ``None``.

        Returns
        -------
        tuple of float or None
            ``(r, g, b)`` in ``[0, 1]``, or ``None``.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Sphere()
        >>> mesh.cad.color = (0.1, 0.2, 0.3)
        >>> mesh.cad.color
        (0.1, 0.2, 0.3)

        """,
    )


def _make_metadata() -> property:
    def fget(self: Any) -> CadMetadata:
        return CadMetadata.from_dataset(self._dataset)

    def fset(self: Any, value: CadMetadata) -> None:
        for key in _METADATA_FIELD_KEYS:
            self._dataset.field_data.pop(key, None)
        value.apply_to(self._dataset)

    return property(
        fget,
        fset,
        doc="""Live :class:`~pyvista_cad.CadMetadata` view.

        Returns
        -------
        pyvista_cad.CadMetadata
            A snapshot built from the dataset's ``cad.*`` field data.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Sphere()
        >>> mesh.cad.set_metadata(units='mm')
        >>> type(mesh.cad.metadata).__name__
        'CadMetadata'

        """,
    )


def _make_layers() -> property:
    def fget(self: Any) -> list[str]:
        ds = self._dataset
        if isinstance(ds, pv.MultiBlock):
            out: set[str] = set()
            for block in ds:
                if block is None:
                    continue
                out.update(_block_layers(block))
            return sorted(out)
        return sorted(_block_layers(ds))

    return property(
        fget,
        doc="""Sorted unique DXF / cell-data layer names.

        Returns
        -------
        list of str
            Sorted distinct ``cad.layer`` (or ``Layer``) cell values;
            an empty list when no layer array is present.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> pv.Sphere().cad.layers
        []

        """,
    )


def _make_has_brep_origin() -> property:
    def fget(self: Any) -> bool:
        ds = self._dataset
        fd = getattr(ds, 'field_data', {})
        if 'cad.has_brep_origin' in fd:
            try:
                return bool(np.asarray(fd['cad.has_brep_origin']).ravel()[0])
            except (IndexError, ValueError):
                return False
        if isinstance(ds, pv.MultiBlock):
            for block in ds:
                if block is None:
                    continue
                if _get_cached_topods(block) is not None:
                    return True
                bfd = getattr(block, 'field_data', {})
                if 'cad.has_brep_origin' in bfd:
                    try:
                        if bool(np.asarray(bfd['cad.has_brep_origin']).ravel()[0]):
                            return True
                    except (IndexError, ValueError):
                        continue
            return False
        return _get_cached_topods(ds) is not None

    return property(
        fget,
        doc="""Whether this dataset has a recoverable B-rep origin.

        ``True`` when a cached originating ``TopoDS_Shape`` is
        associated with the mesh (or, for a MultiBlock, with any leaf
        block), or when a ``cad.has_brep_origin`` field-data flag is
        stamped truthy. ``False`` otherwise. The exact-geometry helpers
        (:meth:`tessellate`, :meth:`exact_volume`,
        :meth:`center_of_mass`) require this to be ``True``.

        Returns
        -------
        bool
            ``True`` if a cached ``TopoDS_Shape`` is recoverable.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> pv.Sphere().cad.has_brep_origin
        False

        """,
    )


def _make_tessellation_quality() -> property:
    def fget(self: Any) -> dict[str, float | None] | None:
        fd = self._dataset.field_data
        lin = fd.get('cad.linear_deflection')
        ang = fd.get('cad.angular_deflection')
        if lin is None and ang is None:
            return None

        def _scalar(arr: Any) -> float | None:
            if arr is None:
                return None
            a = np.asarray(arr).ravel()
            return float(a[0]) if a.size else None

        return {
            'linear_deflection': _scalar(lin),
            'angular_deflection': _scalar(ang),
        }

    return property(
        fget,
        doc="""Tessellation deflections last used to build this mesh.

        :meth:`tessellate` stamps the ``cad.linear_deflection`` and
        ``cad.angular_deflection`` it used into field data; this
        property reads them back.

        Returns
        -------
        dict or None
            ``{'linear_deflection': float | None,
            'angular_deflection': float | None}`` when at least one
            deflection is stamped, otherwise ``None``.

        Examples
        --------
        >>> import pyvista_cad
        >>> from pyvista_cad import examples
        >>> mesh = pyvista_cad.read_step(examples.step_cube)
        >>> fine = mesh.cad.tessellate(linear_deflection=0.05)
        >>> sorted(fine.cad.tessellation_quality)
        ['angular_deflection', 'linear_deflection']

        """,
    )


# --------------------------------------------------------------------------- #
# Shared method implementations as free functions.
# --------------------------------------------------------------------------- #


def _set_units_impl(self: Any, target: str, *, convert: bool = False) -> None:
    validate_units(target)
    if convert:
        current = _read_str_field(self._dataset.field_data, 'cad.units')
        if current is None:
            msg = 'convert=True requires the current units to be set'
            raise ValueError(msg)
        factor = units_conversion_factor(current, target)
        _scale_points(self._dataset, factor)
    self._dataset.field_data['cad.units'] = np.array([target])


_METADATA_FIELD_KEYS: tuple[str, ...] = (
    'cad.source_format',
    'cad.source_format_version',
    'cad.units',
    'cad.label',
    'cad.guid',
    'cad.color',
    'cad.transform',
    'cad.metadata',
)


def _set_metadata_impl(
    self: Any,
    *,
    units: str | None = None,
    label: str | None = None,
    color: tuple[float, ...] | None = None,
    transform: Any | None = None,
    source_format: str | None = None,
    source_format_version: str | None = None,
    guid: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    from pyvista_cad._errors import MetadataError

    if units is not None:
        try:
            validate_units(units)
        except ValueError as exc:
            raise MetadataError(str(exc)) from exc

    rgba: tuple[float, ...] | None = None
    if color is not None:
        arr = np.asarray(color, dtype=float).ravel()
        if arr.size not in (3, 4):
            msg = f'color must be an RGB 3-tuple or RGBA 4-tuple; got shape {arr.shape}'
            raise MetadataError(msg)
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            msg = f'color components must lie in [0, 1]; got {tuple(arr)}'
            raise MetadataError(msg)
        rgba = tuple(float(x) for x in arr)

    xform: np.ndarray | None = None
    if transform is not None:
        xform = np.asarray(transform, dtype=float)
        if xform.shape != (4, 4):
            msg = f'transform must be a 4x4 matrix; got shape {xform.shape}'
            raise MetadataError(msg)

    if metadata is not None and not isinstance(metadata, dict):
        msg = f'metadata must be a dict; got {type(metadata).__name__}'
        raise MetadataError(msg)

    md = CadMetadata(
        source_format=source_format,
        source_format_version=source_format_version,
        units=units,
        color=rgba,
        label=label,
        guid=guid,
        transform=xform,
        extra=dict(metadata) if metadata else {},
    )
    fd = self._dataset.field_data
    for key in _METADATA_FIELD_KEYS:
        fd.pop(key, None)
    md.apply_to(self._dataset)


def _scale_points(ds: Any, factor: float) -> None:
    if isinstance(ds, pv.MultiBlock):
        for block in ds:
            if block is not None and hasattr(block, 'points'):
                block.points = np.asarray(block.points, dtype=float) * factor
        return
    if hasattr(ds, 'points'):
        ds.points = np.asarray(ds.points, dtype=float) * factor


def _tessellate_impl(
    self: Any,
    *,
    linear_deflection: float | None = None,
    angular_deflection: float = 0.5,
) -> pv.PolyData:
    from pyvista_cad import CadError

    ds = self._dataset
    mesh = _as_polydata(ds)
    if linear_deflection is None:
        bounds = np.asarray(mesh.bounds, dtype=float)
        bbox_diag = float(np.linalg.norm(bounds[1::2] - bounds[::2]))
        linear_deflection = 0.001 * bbox_diag if bbox_diag > 0 else 0.1

    if isinstance(ds, pv.MultiBlock):
        shape = _collect_cached_topods(ds)
    else:
        shape = _get_cached_topods(mesh)
    if shape is None:
        msg = (
            'tessellate() requires a CAD-origin mesh — load via '
            'read_step/read_brep/read_iges first'
        )
        raise CadError(msg)

    from OCP.BRepMesh import BRepMesh_IncrementalMesh

    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    from pyvista_cad._bridges.topods import from_topods as _from

    out = _from(
        shape,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    # Stamp tessellation parameters so the ``tessellation_quality``
    # property (see ``_make_tessellation_quality``) reads them back.
    out.field_data['cad.linear_deflection'] = np.array([float(linear_deflection)])
    out.field_data['cad.angular_deflection'] = np.array([float(angular_deflection)])
    return out


def _cad_view_impl(self: Any, **kwargs: Any) -> pv.MultiBlock:
    """Resolve the wrapped dataset to a ``{'faces', 'edges'}`` MultiBlock."""
    from pyvista_cad._cad_view import as_cad_multiblock

    ds = self._dataset
    if isinstance(ds, pv.MultiBlock):
        return as_cad_multiblock(ds, **kwargs)
    mesh = _as_polydata(ds)
    shape = _get_cached_topods(mesh)
    return as_cad_multiblock(shape if shape is not None else mesh, **kwargs)


def _plot_impl(self: Any, **kwargs: Any) -> Any:
    """CAD-friendly one-shot plot of the wrapped dataset.

    Prefers the cached originating ``TopoDS`` (true topological edges +
    analytic normals); falls back to feature-edge extraction for a
    plain mesh with no B-rep origin.
    """
    from pyvista_cad._cad_view import plot_cad

    ds = self._dataset
    if isinstance(ds, pv.MultiBlock):
        # Pass the MultiBlock straight through: ``_resolve_cad`` walks
        # it block-by-block and recovers each block's cached ``TopoDS``,
        # which a ``combine()`` would discard.
        return plot_cad(ds, **kwargs)
    mesh = _as_polydata(ds)
    shape = _get_cached_topods(mesh)
    return plot_cad(shape if shape is not None else mesh, **kwargs)


def _to_dxf_impl(
    self: Any,
    path: str | os.PathLike[str],
    /,
    *,
    layer: str = '0',
    layer_array: str | None = 'cad.layer',
    dxf_version: str = 'R2018',
    units: str | None = None,
    color_index: int | None = None,
    **kwargs: Any,
) -> None:
    from pyvista_cad._readers.dxf import write_dxf

    ds: Any = self._dataset
    if isinstance(ds, pv.MultiBlock):
        ds = ds.combine(merge_points=True)
    if not isinstance(ds, pv.PolyData):
        ds = ds.extract_surface(algorithm='dataset_surface')
    resolved_array = layer_array
    if resolved_array is not None and resolved_array not in ds.cell_data:
        resolved_array = 'Layer' if 'Layer' in ds.cell_data else None
    write_dxf(
        ds,
        path,
        layer=layer,
        layer_array=resolved_array,
        dxf_version=dxf_version,
        units=units,
        color_index=color_index,
        **kwargs,
    )


def _to_3mf_impl(
    self: Any,
    path: str | os.PathLike[str],
    /,
    *,
    units: str | None = None,
) -> None:
    from pyvista_cad._readers.three_mf import write_three_mf

    ds: Any = self._dataset
    if not isinstance(ds, (pv.PolyData, pv.MultiBlock)):
        ds = ds.extract_surface(algorithm='dataset_surface')
    write_three_mf(ds, path, units=units)


def _to_brep_impl(self: Any, path: str | os.PathLike[str], /) -> None:
    from pyvista_cad._readers.brep import write_brep

    write_brep(_as_polydata(self._dataset), path)


def _to_step_impl(self: Any, path: str | os.PathLike[str], /) -> None:
    from pyvista_cad._readers.step import write_step

    ds: Any = self._dataset
    if not isinstance(ds, (pv.PolyData, pv.MultiBlock)):
        ds = ds.extract_surface(algorithm='dataset_surface')
    write_step(ds, path)


def _to_gltf_impl(self: Any, path: str | os.PathLike[str], /) -> None:
    """Write to glTF via trimesh.

    Falls back to PyVista's ``Plotter.export_gltf`` only if trimesh is
    unavailable. ``cad.label`` is propagated onto trimesh metadata so
    glTF node names survive the trip when possible.
    """
    from pyvista_cad._errors import CadError, CadWarning

    ds: Any = self._dataset
    try:
        # trimesh ships no type stubs; optional dependency.
        import trimesh  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on env
        # Last-ditch: a plotter export.
        try:
            pl = pv.Plotter(off_screen=True)
            if isinstance(ds, pv.MultiBlock):
                pl.add_mesh(ds.combine(merge_points=True))
            else:
                pl.add_mesh(ds)
            pl.export_gltf(str(path))
            pl.close()
            return
        except Exception as inner:
            msg = 'to_gltf requires `trimesh` (or PyVista with a working gltf exporter)'
            raise CadError(msg) from inner
        finally:
            del exc

    scene = trimesh.Scene()
    if isinstance(ds, pv.MultiBlock):
        # Walk multiblock and add each leaf as a named node.
        acc = ds.cad
        for path_, block in acc.walk():
            if isinstance(block, pv.MultiBlock):
                continue
            poly = (
                block
                if isinstance(block, pv.PolyData)
                else block.extract_surface(algorithm='dataset_surface')
            )
            try:
                node = pv.to_trimesh(poly, triangulate=True)
            except (ValueError, TypeError) as exc:
                # A leaf with no triangulable faces (point cloud,
                # polyline-only block) makes the trimesh conversion
                # raise. Skip it, but say so loudly: a silently shorter
                # glTF is worse than a named warning.
                name = _read_str_field(poly.field_data, 'cad.label') or path_
                warnings.warn(
                    f'Dropping MultiBlock leaf {name!r} from the glTF '
                    f'export: it is not convertible to a trimesh mesh '
                    f'({exc}).',
                    CadWarning,
                    stacklevel=2,
                )
                continue
            name = _read_str_field(poly.field_data, 'cad.label') or path_
            scene.add_geometry(node, node_name=name, geom_name=name)
    else:
        poly = _as_polydata(ds)
        node = pv.to_trimesh(poly, triangulate=True)
        name = _read_str_field(poly.field_data, 'cad.label') or 'mesh'
        scene.add_geometry(node, node_name=name, geom_name=name)
    scene.export(str(path))


# --------------------------------------------------------------------------- #
# DataSet accessor.
# --------------------------------------------------------------------------- #


# Re-registering the identical accessor class on the same target (which
# happens when ``pytest --cov`` or any double-import re-executes this
# module) is a silent no-op in pyvista >= 0.48.4, so the plain decorator
# is safe under the project's ``filterwarnings=error``.
@pv.register_dataset_accessor('cad', pv.DataSet)
class CadDataSetAccessor:
    """The ``.cad`` accessor on every PyVista :class:`pyvista.DataSet`.

    Surfaces ``cad.*`` field-data and cell-data as Python attributes,
    routes to the CAD readers / writers, and exposes per-mesh helpers
    such as :meth:`tessellate`, :meth:`exact_volume`, and
    :meth:`center_of_mass`.

    Parameters
    ----------
    dataset : pyvista.DataSet
        The dataset this accessor is bound to.

    Examples
    --------
    >>> import pyvista as pv
    >>> import pyvista_cad
    >>> pv.Sphere().cad.source_format is None
    True

    """

    def __init__(self, dataset: DataSet) -> None:
        self._dataset = dataset

    def __repr__(self) -> str:
        """Return a short representation of the accessor.

        Returns
        -------
        str
            Class name plus the wrapped dataset type.

        """
        return f'<CadDataSetAccessor on {type(self._dataset).__name__}>'

    # Metadata properties ----------------------------------------------------
    source_format = _make_source_format()
    units = _make_units()
    label = _make_label()
    color = _make_color()
    metadata = _make_metadata()
    layers = _make_layers()
    has_brep_origin = _make_has_brep_origin()
    tessellation_quality = _make_tessellation_quality()

    # Metadata mutators ------------------------------------------------------
    def set_units(self, target: str, *, convert: bool = False) -> None:
        """Update ``cad.units``, optionally rescaling point coordinates.

        Parameters
        ----------
        target : str
            Target unit string.
        convert : bool, default: False
            When ``True``, rescale point coordinates by the unit ratio.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Sphere()
        >>> mesh.cad.set_units('mm')
        >>> mesh.cad.units
        'mm'

        """
        _set_units_impl(self, target, convert=convert)

    def set_metadata(
        self,
        *,
        units: str | None = None,
        label: str | None = None,
        color: tuple[float, ...] | None = None,
        transform: Any | None = None,
        source_format: str | None = None,
        source_format_version: str | None = None,
        guid: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write the standard ``cad.*`` metadata into this dataset.

        Symmetric write side of the read-only metadata properties
        (:attr:`units`, :attr:`label`, :attr:`color`, ...) and the
        :attr:`metadata` view. Every populated keyword is validated and
        written into field data via :class:`~pyvista_cad.CadMetadata`;
        any previously stored ``cad.*`` schema key is cleared first, so
        this replaces the metadata rather than merging into it.

        Parameters
        ----------
        units : str, optional
            Canonical length-unit string (``'mm'``, ``'inch'``, ...).
        label : str, optional
            Whole-dataset label.
        color : tuple of float, optional
            RGB 3-tuple or RGBA 4-tuple in ``[0, 1]``. Alpha rides on
            the color; there is no separate opacity key.
        transform : array-like, optional
            ``4x4`` assembly-local-to-world transform.
        source_format : str, optional
            Originating CAD format (``'STEP'``, ``'DXF'``, ...).
        source_format_version : str, optional
            Format-specific version stamp (e.g. a DXF release).
        guid : str, optional
            Stable unique identifier (IFC ``GlobalId``, etc.).
        metadata : dict, optional
            Free-form JSON-serializable bag stored under
            ``cad.metadata``.

        Raises
        ------
        pyvista_cad.MetadataError
            If ``units`` is not a known unit, ``color`` is not a 3- or
            4-tuple in ``[0, 1]``, ``transform`` is not ``4x4``, or
            ``metadata`` is not a dict.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Sphere()
        >>> mesh.cad.set_metadata(
        ...     units='mm', label='widget', color=(0.2, 0.4, 0.6, 0.5)
        ... )
        >>> mesh.cad.units
        'mm'
        >>> mesh.cad.metadata.color
        (0.2, 0.4, 0.6, 0.5)

        """
        _set_metadata_impl(
            self,
            units=units,
            label=label,
            color=color,
            transform=transform,
            source_format=source_format,
            source_format_version=source_format_version,
            guid=guid,
            metadata=metadata,
        )

    # Split / filter helpers (cell-array driven) -----------------------------
    def split_by_layer(self, layer_array: str = 'cad.layer') -> pv.MultiBlock:
        """Split into one block per ``cad.layer`` cell value.

        Parameters
        ----------
        layer_array : str, default: 'cad.layer'
            Cell-data array whose distinct values define the partition.

        Returns
        -------
        pyvista.MultiBlock
            One block per unique layer value.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Cube().triangulate()
        >>> mesh.cell_data['cad.layer'] = np.array(['L'] * mesh.n_cells)
        >>> mesh.cad.split_by_layer().keys()
        ['L']

        """
        return _split_by_cell_array(self._dataset, layer_array, fallback='Layer')

    def split_by_label(self, label_array: str = 'cad.label') -> pv.MultiBlock:
        """Split a composite mesh by the values of a per-cell label array.

        Groups cells by the distinct values of the ``label_array``
        cell-data array and returns one block per unique value, named
        by that value. This is a value-based partition of a single
        mesh, not a glob or pattern match. Useful when an importer
        stamps a per-face or per-part label onto one combined mesh.

        Parameters
        ----------
        label_array : str, default: 'cad.label'
            Name of the per-cell array whose distinct values define
            the partition.

        Returns
        -------
        pyvista.MultiBlock
            One block per unique label value (empty when the array is
            absent).

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Cube().triangulate()
        >>> half = mesh.n_cells // 2
        >>> mesh.cell_data['cad.label'] = np.array(
        ...     ['a'] * half + ['b'] * (mesh.n_cells - half)
        ... )
        >>> parts = mesh.cad.split_by_label()
        >>> sorted(parts.keys())
        ['a', 'b']

        """
        return _split_by_cell_array(self._dataset, label_array, fallback=None)

    def split_by_color(self, color_array: str = 'cad.color') -> pv.MultiBlock:
        """Split into one block per per-cell RGB color.

        Parameters
        ----------
        color_array : str, default: 'cad.color'
            ``(N, 3)`` RGB cell-data array defining the partition.

        Returns
        -------
        pyvista.MultiBlock
            One block per unique RGB triplet (empty when absent).

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mesh = pv.Cube().triangulate()
        >>> mesh.cell_data['cad.color'] = np.tile([1.0, 0.0, 0.0], (mesh.n_cells, 1))
        >>> mesh.cad.split_by_color().n_blocks
        1

        """
        ds = self._dataset
        if not hasattr(ds, 'cell_data') or color_array not in ds.cell_data:
            return pv.MultiBlock()
        colors = np.asarray(ds.cell_data[color_array])
        if colors.ndim != 2 or colors.shape[1] != 3:
            msg = f'{color_array!r} must be (N, 3) RGB; got shape {colors.shape}'
            raise ValueError(msg)
        rounded = np.round(colors, 6)
        unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
        inverse = np.ascontiguousarray(inverse, dtype=np.int64)
        sort_idx = np.argsort(inverse, kind='stable')
        counts = np.bincount(inverse, minlength=unique.shape[0])
        starts = np.zeros(unique.shape[0] + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        mb = pv.MultiBlock()
        for i, row in enumerate(unique):
            # np.unique guarantees every row occurs at least once, so
            # bincount yields a strictly positive count and the slice
            # [starts[i], starts[i + 1]) is never empty.
            lo, hi = int(starts[i]), int(starts[i + 1])
            block = _fast_extract_cells_sorted(ds, sort_idx[lo:hi])
            name = f'rgb({row[0]:.3f},{row[1]:.3f},{row[2]:.3f})'
            mb.append(block, name=name)
        return mb

    # Exact (BREP-cache) properties -----------------------------------------
    def exact_volume(self) -> float:
        """Return the exact volume from the cached ``TopoDS_Shape``.

        Returns
        -------
        float
            Volume in cube of the dataset's units, as computed by
            OCCT ``GProp_GProps``.

        Raises
        ------
        pyvista_cad.CadError
            If no cached ``TopoDS_Shape`` is associated with the mesh.

        Examples
        --------
        >>> import pyvista_cad
        >>> from pyvista_cad import examples
        >>> cube = pyvista_cad.read_step(examples.step_cube)[0]
        >>> round(cube.cad.exact_volume())
        1000

        """
        from pyvista_cad import CadError

        shape = _get_cached_topods(_as_polydata(self._dataset))
        if shape is None:
            msg = 'exact_volume requires a CAD-origin mesh'
            raise CadError(msg)
        return float(_topods_props(shape).Mass())

    def center_of_mass(self) -> np.ndarray:
        """Return the exact centre of mass from the cached ``TopoDS_Shape``.

        Returns
        -------
        numpy.ndarray
            ``[x, y, z]`` in dataset coordinates.

        Raises
        ------
        pyvista_cad.CadError
            If no cached ``TopoDS_Shape`` is associated with the mesh.

        Examples
        --------
        >>> import pyvista_cad
        >>> from pyvista_cad import examples
        >>> cube = pyvista_cad.read_step(examples.step_cube)[0]
        >>> [round(x) for x in cube.cad.center_of_mass()]
        [0, 0, 0]

        """
        from pyvista_cad import CadError

        shape = _get_cached_topods(_as_polydata(self._dataset))
        if shape is None:
            msg = 'center_of_mass requires a CAD-origin mesh'
            raise CadError(msg)
        pnt = _topods_props(shape).CentreOfMass()
        return np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=float)

    # Conversion: outgoing --------------------------------------------------
    def to_build123d(self) -> Any:
        """Wrap the dataset as a faceted ``build123d.Compound``.

        Returns
        -------
        build123d.Compound
            Faceted compound wrapping the dataset geometry.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> type(pv.Sphere().cad.to_build123d()).__name__
        'Compound'

        """
        from pyvista_cad._bridges.build123d import to_build123d as _impl

        return _impl(_as_polydata(self._dataset))

    def to_cadquery(self) -> Any:
        """Wrap the dataset as a faceted :class:`cadquery.Workplane`.

        Returns
        -------
        cadquery.Workplane
            Workplane wrapping the faceted dataset geometry.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> type(pv.Sphere().cad.to_cadquery()).__name__
        'Workplane'

        """
        from pyvista_cad._bridges.cadquery import to_cadquery as _impl

        return _impl(_as_polydata(self._dataset))

    def to_topods(self) -> Any:
        """Build a faceted ``TopoDS`` shape from the dataset.

        A watertight mesh yields a ``TopoDS_Solid``; an open mesh
        yields a sewn shell or compound.

        Returns
        -------
        TopoDS_Shape
            ``TopoDS_Solid`` for a watertight mesh, otherwise a sewn
            shell or compound.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> type(pv.Sphere().cad.to_topods()).__name__
        'TopoDS_Solid'

        """
        from pyvista_cad._bridges.topods import to_topods as _impl

        return _impl(_as_polydata(self._dataset))

    def to_gmsh(self, *, model_name: str = 'pyvista_cad') -> None:
        """Install the dataset as the current gmsh model.

        Examples
        --------
        >>> import gmsh
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> gmsh.initialize()
        >>> gmsh.option.setNumber('General.Terminal', 0)
        >>> pv.Sphere().cad.to_gmsh()
        >>> gmsh.model.getCurrent()
        'pyvista_cad'
        >>> gmsh.finalize()

        """
        from pyvista_cad._bridges.gmsh import to_gmsh as _impl

        ds: Any = self._dataset
        if not isinstance(ds, (pv.UnstructuredGrid, pv.PolyData)):
            ds = ds.extract_surface(algorithm='dataset_surface')
        _impl(ds, model_name=model_name)

    def to_dxf(
        self,
        path: str | os.PathLike[str],
        /,
        *,
        layer: str = '0',
        layer_array: str | None = 'cad.layer',
        dxf_version: str = 'R2018',
        units: str | None = None,
        color_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Write this dataset as a DXF file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.dxf')
        >>> pv.Sphere().cad.to_dxf(out)
        >>> os.path.exists(out)
        True

        """
        _to_dxf_impl(
            self,
            path,
            layer=layer,
            layer_array=layer_array,
            dxf_version=dxf_version,
            units=units,
            color_index=color_index,
            **kwargs,
        )

    def to_3mf(
        self,
        path: str | os.PathLike[str],
        /,
        *,
        units: str | None = None,
    ) -> None:
        """Write to ``path`` as a 3MF file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.3mf')
        >>> pv.Sphere().cad.to_3mf(out)
        >>> os.path.exists(out)
        True

        """
        _to_3mf_impl(self, path, units=units)

    def to_brep(self, path: str | os.PathLike[str], /) -> None:
        """Write to ``path`` as a faceted BREP file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.brep')
        >>> pv.Sphere().cad.to_brep(out)
        >>> os.path.exists(out)
        True

        """
        _to_brep_impl(self, path)

    def to_step(self, path: str | os.PathLike[str], /) -> None:
        """Write to ``path`` as a faceted STEP file.

        Examples
        --------
        >>> import contextlib
        >>> import io
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.step')
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     pv.Sphere().cad.to_step(out)
        >>> os.path.exists(out)
        True

        """
        _to_step_impl(self, path)

    def to_gltf(self, path: str | os.PathLike[str], /) -> None:
        """Write to ``path`` as a glTF (``.gltf`` / ``.glb``) file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.glb')
        >>> pv.Sphere().cad.to_gltf(out)
        >>> os.path.exists(out)
        True

        """
        _to_gltf_impl(self, path)

    def tessellate(
        self,
        *,
        linear_deflection: float | None = None,
        angular_deflection: float = 0.5,
    ) -> pv.PolyData:
        """Re-tessellate this mesh from its cached originating B-rep.

        Reuses the exact ``TopoDS_Shape`` stashed when this mesh was
        produced by a CAD reader (STEP, BREP, IGES, build123d bridge)
        and re-meshes it at the requested deflections. This does not
        decimate or remesh the existing triangulation; it goes back to
        the analytic geometry. The chosen deflections are stamped into
        field data and surface as :attr:`tessellation_quality`.

        A mesh with no cached B-rep origin cannot be re-tessellated and
        raises rather than silently returning the existing facets.

        Parameters
        ----------
        linear_deflection : float, optional
            Maximum chordal deviation between a facet and the true
            surface, in dataset units. When ``None``, defaults to
            ``0.001`` of the bounding-box diagonal.
        angular_deflection : float, default: 0.5
            Maximum angle (radians) between adjacent facet normals.

        Returns
        -------
        pyvista.PolyData
            The freshly tessellated surface, carrying
            ``cad.linear_deflection`` / ``cad.angular_deflection``.

        Raises
        ------
        pyvista_cad.CadError
            If no cached originating ``TopoDS_Shape`` is associated
            with the mesh (it was not produced by a CAD reader).

        Examples
        --------
        >>> import pyvista_cad
        >>> from pyvista_cad import examples
        >>> mesh = pyvista_cad.read_step(examples.step_cube)
        >>> retess = mesh.cad.tessellate(linear_deflection=0.02)
        >>> retess.n_cells > 0
        True
        >>> retess.cad.tessellation_quality['linear_deflection']
        0.02

        """
        return _tessellate_impl(
            self,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )

    def plot(self, **kwargs: Any) -> Any:
        """CAD-friendly plot: shaded faces + topological edges.

        Unlike :meth:`pyvista.DataSet.plot`, the overlaid lines are the
        model's *topological* edges (B-rep feature curves), not the
        arbitrary triangle-mesh edges. For a mesh with no cached B-rep
        origin, falls back to crease feature-edge extraction.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pyvista_cad.add_cad` /
            :meth:`pyvista.Plotter.show`.

        Returns
        -------
        Any
            Whatever :meth:`pyvista.Plotter.show` returns.

        Examples
        --------
        >>> import pyvista as pv
        >>> pv.OFF_SCREEN = True
        >>> import pyvista_cad
        >>> cpos = pv.Sphere().cad.plot(return_cpos=True)
        >>> type(cpos).__name__
        'CameraPosition'

        """
        return _plot_impl(self, **kwargs)

    def cad_view(self, **kwargs: Any) -> pv.MultiBlock:
        """Resolve to a ``{'faces', 'edges'}`` CAD-view MultiBlock.

        ``faces`` is a per-face MultiBlock with analytic normals;
        ``edges`` the topological B-rep curves (with ``cad.edge_id`` /
        ``cad.edge_kind`` cell arrays). The data behind
        :meth:`plot` / ``plotter.cad.add``.

        Parameters
        ----------
        **kwargs
            ``linear_deflection``, ``angular_deflection``,
            ``feature_angle`` -- see :func:`pyvista_cad.as_cad_multiblock`.

        Returns
        -------
        pyvista.MultiBlock
            ``{'faces', 'edges'}``.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> view = pv.Sphere().cad.cad_view()
        >>> sorted(view.keys())
        ['edges', 'faces']

        """
        return _cad_view_impl(self, **kwargs)

    def _as_polydata(self) -> pv.PolyData:
        return _as_polydata(self._dataset)


# --------------------------------------------------------------------------- #
# Helper for layer/label split (shared with MultiBlock accessor).
# --------------------------------------------------------------------------- #


def _fast_extract_cells_sorted(ds: Any, cell_ids: np.ndarray) -> Any:
    """Extract a cell-id slice via :meth:`pyvista.DataSet.extract_cells`."""
    return ds.extract_cells(np.ascontiguousarray(cell_ids))


def _split_by_cell_array(
    ds: Any,
    key: str,
    *,
    fallback: str | None,
) -> pv.MultiBlock:
    if not hasattr(ds, 'cell_data'):
        msg = f'split by {key!r} requires a DataSet with cell_data'
        raise TypeError(msg)
    raw = _resolve_cell_array(ds, key, fallback=fallback)
    if raw is None:
        return pv.MultiBlock()
    labels = np.asarray(raw).astype(str)
    unique, inverse = np.unique(labels, return_inverse=True)
    inverse = np.ascontiguousarray(inverse, dtype=np.int64)
    # Group cell indices by `inverse` in a single O(N log N) sort,
    # then slice into contiguous index ranges per unique label.
    sort_idx = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse, minlength=unique.size)
    starts = np.zeros(unique.size + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    mb = pv.MultiBlock()
    for idx in np.argsort(unique):
        # np.unique guarantees every label occurs at least once, so
        # bincount yields a strictly positive count and the slice
        # [starts[idx], starts[idx + 1]) is never empty.
        lo, hi = int(starts[idx]), int(starts[idx + 1])
        block = _fast_extract_cells_sorted(ds, sort_idx[lo:hi])
        mb.append(block, name=str(unique[idx]))
    return mb


# --------------------------------------------------------------------------- #
# MultiBlock accessor.
# --------------------------------------------------------------------------- #


@pv.register_dataset_accessor('cad', pv.MultiBlock)
class CadMultiBlockAccessor:
    """The ``.cad`` accessor on every :class:`pyvista.MultiBlock`.

    Assembly-aware: walks the hierarchy, filters by metadata, flattens,
    and dumps a pure-data view of the tree.

    Parameters
    ----------
    dataset : pyvista.MultiBlock
        MultiBlock this accessor is bound to.

    Examples
    --------
    >>> import pyvista as pv
    >>> import pyvista_cad
    >>> pv.MultiBlock().cad.source_format is None
    True

    """

    def __init__(self, dataset: MultiBlock) -> None:
        self._dataset = dataset

    def __repr__(self) -> str:
        """Return a short repr.

        Returns
        -------
        str
            Class name plus the wrapped MultiBlock type.

        """
        return f'<CadMultiBlockAccessor on {type(self._dataset).__name__}>'

    # Metadata properties (duplicated, not inherited) -----------------------
    source_format = _make_source_format()
    units = _make_units()
    label = _make_label()
    color = _make_color()
    metadata = _make_metadata()
    layers = _make_layers()
    has_brep_origin = _make_has_brep_origin()
    tessellation_quality = _make_tessellation_quality()

    def set_units(self, target: str, *, convert: bool = False) -> None:
        """Update ``cad.units`` on every block, optionally rescaling points.

        Parameters
        ----------
        target : str
            Target unit string.
        convert : bool, default: False
            When ``True``, rescale point coordinates by the unit ratio.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock([pv.Sphere()])
        >>> mb.cad.set_units('mm')
        >>> mb.cad.units
        'mm'

        """
        _set_units_impl(self, target, convert=convert)

    def set_metadata(
        self,
        *,
        units: str | None = None,
        label: str | None = None,
        color: tuple[float, ...] | None = None,
        transform: Any | None = None,
        source_format: str | None = None,
        source_format_version: str | None = None,
        guid: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write the standard ``cad.*`` metadata onto this MultiBlock.

        Writes assembly-level metadata into the MultiBlock's own field
        data (not recursively into the leaf blocks). Symmetric write
        side of the read-only metadata properties and the
        :attr:`metadata` view; any previously stored ``cad.*`` schema
        key is cleared first.

        Parameters
        ----------
        units : str, optional
            Canonical length-unit string (``'mm'``, ``'inch'``, ...).
        label : str, optional
            Assembly-level label.
        color : tuple of float, optional
            RGB 3-tuple or RGBA 4-tuple in ``[0, 1]``. Alpha rides on
            the color; there is no separate opacity key.
        transform : array-like, optional
            ``4x4`` assembly-local-to-world transform.
        source_format : str, optional
            Originating CAD format (``'STEP'``, ``'DXF'``, ...).
        source_format_version : str, optional
            Format-specific version stamp (e.g. a DXF release).
        guid : str, optional
            Stable unique identifier (IFC ``GlobalId``, etc.).
        metadata : dict, optional
            Free-form JSON-serializable bag stored under
            ``cad.metadata``.

        Raises
        ------
        pyvista_cad.MetadataError
            If ``units`` is not a known unit, ``color`` is not a 3- or
            4-tuple in ``[0, 1]``, ``transform`` is not ``4x4``, or
            ``metadata`` is not a dict.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock([pv.Sphere(), pv.Cube()])
        >>> mb.cad.set_metadata(units='m', label='assembly')
        >>> mb.cad.units
        'm'

        """
        _set_metadata_impl(
            self,
            units=units,
            label=label,
            color=color,
            transform=transform,
            source_format=source_format,
            source_format_version=source_format_version,
            guid=guid,
            metadata=metadata,
        )

    # Assembly walk / filter ------------------------------------------------
    def walk(self) -> Iterator[tuple[str, pv.DataSet]]:
        """Walk the MultiBlock tree depth-first.

        Yields
        ------
        tuple of (str, pyvista.DataSet)
            ``(path, block)`` pairs in DFS pre-order.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock({'a': pv.Sphere(), 'b': pv.Cube()})
        >>> [name for name, _ in mb.cad.walk()]
        ['a', 'b']

        """

        def _descend(prefix: str, mb: pv.MultiBlock) -> Iterator[tuple[str, pv.DataSet]]:
            for i in range(mb.n_blocks):
                block = mb[i]
                if block is None:
                    continue
                name = mb.get_block_name(i) or f'block_{i}'
                path = f'{prefix}/{name}' if prefix else name
                yield path, block
                if isinstance(block, pv.MultiBlock):
                    yield from _descend(path, block)

        yield from _descend('', self._dataset)

    def find(
        self,
        predicate: str | Callable[[str, pv.DataSet], bool] | None = None,
        *,
        name: str | None = None,
        ifc_type: str | None = None,
        layer: str | None = None,
        source_format: str | None = None,
    ) -> list[tuple[str, pv.DataSet]]:
        """Filter the MultiBlock tree.

        Parameters
        ----------
        predicate : str, callable, or None, optional
            * ``str``: glob-style pattern matched against block labels
              (``cad.label`` or the trailing path name).
            * ``callable``: ``(path, block) -> bool``.
            * ``None``: ignored — fall back to the keyword predicates.
        name, ifc_type, layer, source_format : str, optional
            Exact-match metadata filters (AND-combined). Each defaults
            to ``None`` (do not filter).

        Returns
        -------
        list of (str, pyvista.DataSet)
            Matching ``(path, block)`` pairs.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock({'wheel': pv.Sphere(), 'axle': pv.Cube()})
        >>> [name for name, _ in mb.cad.find(predicate='whe*')]
        ['wheel']

        """
        import fnmatch

        callable_pred: Callable[[str, pv.DataSet], bool] | None = None
        glob_pred: str | None = None
        if callable(predicate):
            callable_pred = predicate
        elif isinstance(predicate, str):
            glob_pred = predicate

        matches: list[tuple[str, pv.DataSet]] = []
        for path, block in self.walk():
            if callable_pred is not None and not callable_pred(path, block):
                continue
            if glob_pred is not None:
                trail = path.split('/')[-1]
                lbl = _read_str_field(getattr(block, 'field_data', {}), 'cad.label')
                if not (
                    fnmatch.fnmatchcase(trail, glob_pred)
                    or (lbl is not None and fnmatch.fnmatchcase(lbl, glob_pred))
                ):
                    continue
            if name is not None and path.split('/')[-1] != name:
                continue
            if ifc_type is not None:
                got = _read_str_field(getattr(block, 'field_data', {}), 'cad.ifc_type')
                if got != ifc_type:
                    continue
            if source_format is not None:
                got = _read_str_field(getattr(block, 'field_data', {}), 'cad.source_format')
                if got != source_format:
                    continue
            if layer is not None and layer not in _block_layers(block):
                continue
            matches.append((path, block))
        return matches

    @staticmethod
    def _clone_tree(mb: pv.MultiBlock, drop_ids: frozenset[int]) -> pv.MultiBlock:
        """Rebuild ``mb`` minus leaves in ``drop_ids`` and emptied containers.

        Geometry is shared (block objects are reused, not deep-copied);
        only the MultiBlock containers are new. Container field data is
        carried over so ``cad.*`` assembly metadata survives the prune.
        """
        out = pv.MultiBlock()
        for i in range(mb.n_blocks):
            block = mb[i]
            if block is None:
                continue
            name = mb.get_block_name(i) or f'block_{i}'
            if isinstance(block, pv.MultiBlock):
                sub = CadMultiBlockAccessor._clone_tree(block, drop_ids)
                if sub.n_blocks:
                    out.append(sub, name=name)
            elif id(block) not in drop_ids:
                out.append(block, name=name)
        for key in mb.field_data:
            out.field_data[key] = mb.field_data[key]
        return out

    def drop_spatial_outliers(
        self,
        *,
        threshold: float = 3.5,
        min_blocks: int = 4,
    ) -> pv.MultiBlock:
        """Return a copy with spatial-outlier leaf blocks removed.

        Real BIM / CAD exports routinely carry a few non-building
        markers — a geo-reference point, a survey origin, a true-north
        symbol — placed tens to thousands of metres from the model.
        They wreck the camera framing and any bounding-box-driven
        analysis, yet they have no reliable class to filter on (a
        geo-reference is usually a generic ``IfcBuildingElementProxy``,
        and the block name is arbitrary), so a name/type rule is
        brittle.

        This drops a leaf when its centroid is a *robust* outlier
        against the bulk: the modified z-score (Iglewicz & Hoaglin) of
        its distance from the median centroid, scaled by the median
        absolute deviation (MAD), exceeds ``threshold``. MAD-based, so
        a handful of far-flung markers cannot mask one another the way
        a mean / standard-deviation rule would. The assembly hierarchy
        is preserved; containers the prune leaves empty are dropped.
        Geometry is shared, not copied.

        Parameters
        ----------
        threshold : float, default: 3.5
            Modified z-score above which a block is an outlier. 3.5 is
            the Iglewicz & Hoaglin recommendation; lower prunes more
            aggressively.
        min_blocks : int, default: 4
            Below this many usable leaf blocks the robust statistic is
            undefined; the tree is returned structurally unchanged.

        Returns
        -------
        pyvista.MultiBlock
            A new tree with the same nesting minus outlier leaves (and
            any container they emptied).

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> house = pv.MultiBlock(
        ...     {
        ...         'a': pv.Cube(center=(0, 0, 0)),
        ...         'b': pv.Cube(center=(1, 0, 0)),
        ...         'c': pv.Cube(center=(0, 1, 0)),
        ...         'd': pv.Cube(center=(1, 1, 0)),
        ...         'geo-reference': pv.Cube(center=(500, 500, 0)),
        ...     }
        ... )
        >>> sorted(house.cad.drop_spatial_outliers().keys())
        ['a', 'b', 'c', 'd']

        """
        leaves = [b for _, b in self.walk() if isinstance(b, pv.DataSet) and b.n_points > 0]
        if len(leaves) < min_blocks:
            return self._clone_tree(self._dataset, frozenset())

        centroids = np.array([b.center for b in leaves], dtype=np.float64)
        finite = np.asarray(np.isfinite(centroids).all(axis=1), dtype=bool)
        if finite.sum() < min_blocks:
            return self._clone_tree(self._dataset, frozenset())

        median = np.median(centroids[finite], axis=0)
        dist = np.linalg.norm(centroids - median, axis=1)
        dist_med = np.median(dist[finite])
        mad = np.median(np.abs(dist[finite] - dist_med))
        if mad <= 0:
            # Degenerate spread (coincident / collinear centroids): a
            # robust cut-off is undefined, so prune nothing rather than
            # guess at an arbitrary one.
            return self._clone_tree(self._dataset, frozenset())

        # Iglewicz & Hoaglin modified z-score. A non-finite centroid
        # (NaN distance) fails ``z > threshold`` silently, so flag it
        # explicitly via ``ok``.
        mod_z = 0.6745 * (dist - dist_med) / mad
        drop: set[int] = set()
        for idx, block in enumerate(leaves):
            if not bool(finite[idx]) or float(mod_z[idx]) > threshold:
                drop.add(id(block))
        return self._clone_tree(self._dataset, frozenset(drop))

    def flatten(self) -> pv.MultiBlock:
        """Collapse a nested MultiBlock into a flat MultiBlock of leaves.

        Walks the assembly tree depth-first and returns a single-level
        :class:`pyvista.MultiBlock` containing every leaf dataset, keyed
        by its full slash-joined path. Structure is flattened but each
        leaf stays a separate block (geometry is not merged). Contrast
        :meth:`flatten_to_polydata`, which fuses every leaf into one
        :class:`pyvista.PolyData`.

        Returns
        -------
        pyvista.MultiBlock
            One block per leaf, named by its tree path.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> inner = pv.MultiBlock([pv.Sphere(), pv.Cube()])
        >>> tree = pv.MultiBlock([inner, pv.Cone()])
        >>> flat = tree.cad.flatten()
        >>> len(flat)
        3
        >>> all(isinstance(b, pv.DataSet) for b in flat)
        True

        """
        out = pv.MultiBlock()
        for path, block in self.walk():
            if isinstance(block, pv.MultiBlock):
                continue
            out.append(block, name=path)
        return out

    def flatten_to_polydata(self) -> pv.PolyData:
        """Fuse the entire MultiBlock into one :class:`pyvista.PolyData`.

        Merges every leaf into a single surface mesh (points merged,
        surface-extracted). Use this when you want one mesh; use
        :meth:`flatten` when you want to keep the leaves as separate
        blocks in a flat MultiBlock.

        Returns
        -------
        pyvista.PolyData
            Concatenation of every leaf block, surface-extracted.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> tree = pv.MultiBlock([pv.Sphere(), pv.Cube()])
        >>> poly = tree.cad.flatten_to_polydata()
        >>> isinstance(poly, pv.PolyData)
        True

        """
        # MultiBlock.combine is contractually an UnstructuredGrid, so a
        # surface extraction is always required to reach PolyData.
        combined = self._dataset.combine(merge_points=True)
        return combined.extract_surface(algorithm='dataset_surface')

    def assembly_tree(self) -> dict[str, Any]:
        """Return a pure-data view of the MultiBlock hierarchy.

        Returns
        -------
        dict
            Nested ``{name: subtree or None}`` mapping, with ``None``
            marking leaves. Block labels (``cad.label``) override the
            block name when present.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock({'a': pv.Sphere(), 'b': pv.Cube()})
        >>> mb.cad.assembly_tree()
        {'a': None, 'b': None}

        """

        def _walk(mb: pv.MultiBlock) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for i, block in enumerate(mb):
                if block is None:
                    continue
                lbl = _read_str_field(getattr(block, 'field_data', {}), 'cad.label')
                name = lbl or mb.get_block_name(i) or f'block_{i}'
                if isinstance(block, pv.MultiBlock):
                    out[name] = _walk(block)
                else:
                    out[name] = None
            return out

        return _walk(self._dataset)

    # Split helpers (only meaningful on a leaf with cell_data; here we
    # surface them as no-ops on MultiBlock by combining first) ---------------
    def split_by_layer(self, layer_array: str = 'cad.layer') -> pv.MultiBlock:
        """Combine and split into one block per layer.

        Parameters
        ----------
        layer_array : str, default: 'cad.layer'
            Cell-data array whose distinct values define the partition.

        Returns
        -------
        pyvista.MultiBlock
            One block per unique layer value.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> cube = pv.Cube().triangulate()
        >>> cube.cell_data['cad.layer'] = np.array(['L'] * cube.n_cells)
        >>> pv.MultiBlock([cube]).cad.split_by_layer().keys()
        ['L']

        """
        combined = self._dataset.combine(merge_points=True)
        return combined.cad.split_by_layer(layer_array)

    def split_by_label(self, label_array: str = 'cad.label') -> pv.MultiBlock:
        """Combine the assembly, then split by per-cell label value.

        Combines every leaf into one mesh and partitions it by the
        distinct values of the ``label_array`` cell-data array, one
        block per unique value. Value-based partition, not a pattern
        match.

        Parameters
        ----------
        label_array : str, default: 'cad.label'
            Name of the per-cell array whose distinct values define
            the partition.

        Returns
        -------
        pyvista.MultiBlock
            One block per unique label value.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> a, b = pv.Cube().triangulate(), pv.Sphere()
        >>> a.cell_data['cad.label'] = np.array(['a'] * a.n_cells)
        >>> b.cell_data['cad.label'] = np.array(['b'] * b.n_cells)
        >>> parts = pv.MultiBlock([a, b]).cad.split_by_label()
        >>> sorted(parts.keys())
        ['a', 'b']

        """
        combined = self._dataset.combine(merge_points=True)
        return combined.cad.split_by_label(label_array)

    def split_by_color(self, color_array: str = 'cad.color') -> pv.MultiBlock:
        """Combine and split into one block per per-cell color.

        Parameters
        ----------
        color_array : str, default: 'cad.color'
            ``(N, 3)`` RGB cell-data array defining the partition.

        Returns
        -------
        pyvista.MultiBlock
            One block per unique RGB triplet.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> cube = pv.Cube().triangulate()
        >>> cube.cell_data['cad.color'] = np.tile([0.0, 1.0, 0.0], (cube.n_cells, 1))
        >>> pv.MultiBlock([cube]).cad.split_by_color().n_blocks
        1

        """
        combined = self._dataset.combine(merge_points=True)
        return combined.cad.split_by_color(color_array)

    # Exact (BREP cache) — only on leaves; raise on MultiBlock --------------
    def exact_volume(self) -> float:
        """Sum exact volume across all BREP-cached leaves.

        Returns
        -------
        float
            Total volume from every cached ``TopoDS_Shape``.

        Raises
        ------
        pyvista_cad.CadError
            If no cached B-rep leaf is present.

        Examples
        --------
        >>> import pyvista_cad
        >>> from pyvista_cad import examples
        >>> asm = pyvista_cad.read_step(examples.step_cube)
        >>> round(asm.cad.exact_volume())
        1000

        """
        from pyvista_cad import CadError

        total = 0.0
        found = False
        for _, block in self.walk():
            if isinstance(block, pv.MultiBlock):
                continue
            shape = _get_cached_topods(block)
            if shape is None:
                continue
            total += float(_topods_props(shape).Mass())
            found = True
        if not found:
            msg = 'exact_volume requires at least one CAD-origin leaf mesh'
            raise CadError(msg)
        return total

    # Outgoing conversion / writers (re-use the DataSet implementations).
    # Construction from external CAD objects is the module-level
    # ``pyvista_cad.from_build123d`` / ``from_cadquery`` / ``from_topods``
    # / ``from_gmsh`` API; there is deliberately no
    # ``.cad.from_*`` on either accessor (an accessor is bound to an
    # existing dataset, so constructing a new one through it is a leaky
    # idiom).
    def to_build123d(self) -> Any:
        """Wrap as a faceted ``build123d.Compound``.

        Returns
        -------
        build123d.Compound
            Faceted compound wrapping the combined assembly.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock([pv.Sphere()])
        >>> type(mb.cad.to_build123d()).__name__
        'Compound'

        """
        from pyvista_cad._bridges.build123d import to_build123d as _impl

        return _impl(_as_polydata(self._dataset))

    def to_cadquery(self) -> Any:
        """Wrap as a faceted :class:`cadquery.Workplane`.

        Returns
        -------
        cadquery.Workplane
            Workplane wrapping the faceted combined assembly.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock([pv.Sphere()])
        >>> type(mb.cad.to_cadquery()).__name__
        'Workplane'

        """
        from pyvista_cad._bridges.cadquery import to_cadquery as _impl

        return _impl(_as_polydata(self._dataset))

    def to_topods(self) -> Any:
        """Build a faceted ``TopoDS`` shape from the combined assembly.

        Returns
        -------
        TopoDS_Shape
            Faceted shape built from the combined assembly.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock([pv.Sphere()])
        >>> type(mb.cad.to_topods()).__name__
        'TopoDS_Solid'

        """
        from pyvista_cad._bridges.topods import to_topods as _impl

        return _impl(_as_polydata(self._dataset))

    def to_gmsh(self, *, model_name: str = 'pyvista_cad') -> None:
        """Install this MultiBlock as the current gmsh model.

        Examples
        --------
        >>> import gmsh
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> gmsh.initialize()
        >>> gmsh.option.setNumber('General.Terminal', 0)
        >>> pv.MultiBlock([pv.Sphere()]).cad.to_gmsh()
        >>> gmsh.model.getCurrent()
        'pyvista_cad'
        >>> gmsh.finalize()

        """
        from pyvista_cad._bridges.gmsh import to_gmsh as _impl

        # MultiBlock.combine is contractually an UnstructuredGrid, which
        # the gmsh bridge consumes directly.
        ds: Any = self._dataset.combine(merge_points=True)
        _impl(ds, model_name=model_name)

    def to_dxf(
        self,
        path: str | os.PathLike[str],
        /,
        *,
        layer: str = '0',
        layer_array: str | None = 'cad.layer',
        dxf_version: str = 'R2018',
        units: str | None = None,
        color_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Write this MultiBlock as a DXF file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.dxf')
        >>> pv.MultiBlock([pv.Sphere()]).cad.to_dxf(out)
        >>> os.path.exists(out)
        True

        """
        _to_dxf_impl(
            self,
            path,
            layer=layer,
            layer_array=layer_array,
            dxf_version=dxf_version,
            units=units,
            color_index=color_index,
            **kwargs,
        )

    def to_3mf(
        self,
        path: str | os.PathLike[str],
        /,
        *,
        units: str | None = None,
    ) -> None:
        """Write to ``path`` as a 3MF file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.3mf')
        >>> pv.MultiBlock([pv.Sphere()]).cad.to_3mf(out)
        >>> os.path.exists(out)
        True

        """
        _to_3mf_impl(self, path, units=units)

    def to_brep(self, path: str | os.PathLike[str], /) -> None:
        """Write to ``path`` as a faceted BREP file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.brep')
        >>> pv.MultiBlock([pv.Sphere()]).cad.to_brep(out)
        >>> os.path.exists(out)
        True

        """
        _to_brep_impl(self, path)

    def to_step(self, path: str | os.PathLike[str], /) -> None:
        """Write to ``path`` as a faceted STEP file.

        Examples
        --------
        >>> import contextlib
        >>> import io
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.step')
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     pv.MultiBlock([pv.Sphere()]).cad.to_step(out)
        >>> os.path.exists(out)
        True

        """
        _to_step_impl(self, path)

    def to_gltf(self, path: str | os.PathLike[str], /) -> None:
        """Write to ``path`` as a glTF / GLB file.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> out = os.path.join(tempfile.mkdtemp(), 'o.glb')
        >>> pv.MultiBlock([pv.Sphere()]).cad.to_gltf(out)
        >>> os.path.exists(out)
        True

        """
        _to_gltf_impl(self, path)

    def tessellate(
        self,
        *,
        linear_deflection: float | None = None,
        angular_deflection: float = 0.5,
    ) -> pv.PolyData:
        """Re-tessellate the combined assembly from its cached B-rep.

        Combines the assembly to a single surface and re-meshes it from
        the cached originating ``TopoDS_Shape`` at the requested
        deflections, returning fresh geometry rather than the existing
        triangulation. Raises if no cached B-rep origin is present
        instead of silently degrading to the current facets.

        Parameters
        ----------
        linear_deflection : float, optional
            Maximum chordal deviation, in dataset units. When ``None``,
            defaults to ``0.001`` of the bounding-box diagonal.
        angular_deflection : float, default: 0.5
            Maximum angle (radians) between adjacent facet normals.

        Returns
        -------
        pyvista.PolyData
            The freshly tessellated surface, carrying
            ``cad.linear_deflection`` / ``cad.angular_deflection``.

        Raises
        ------
        pyvista_cad.CadError
            If no cached originating ``TopoDS_Shape`` is associated
            with the combined mesh.

        Examples
        --------
        >>> import pyvista_cad
        >>> from pyvista_cad import examples
        >>> asm = pyvista_cad.read_step(examples.step_cube)
        >>> asm.cad.has_brep_origin
        True
        >>> fine = asm.cad.tessellate(linear_deflection=0.05)
        >>> fine.n_cells > 0
        True

        """
        return _tessellate_impl(
            self,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )

    def plot(self, **kwargs: Any) -> Any:
        """CAD-friendly plot: shaded faces + topological edges.

        Renders the assembly with B-rep feature curves overlaid rather
        than triangle-mesh edges. Honours a ``{'faces', 'edges'}`` block
        produced by :func:`pyvista_cad.topods_to_multiblock`; otherwise
        resolves through the cached originating ``TopoDS``.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pyvista_cad.add_cad` /
            :meth:`pyvista.Plotter.show`.

        Returns
        -------
        Any
            Whatever :meth:`pyvista.Plotter.show` returns.

        Examples
        --------
        >>> import pyvista as pv
        >>> pv.OFF_SCREEN = True
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock([pv.Sphere()])
        >>> cpos = mb.cad.plot(return_cpos=True)
        >>> type(cpos).__name__
        'CameraPosition'

        """
        return _plot_impl(self, **kwargs)

    def cad_view(self, **kwargs: Any) -> pv.MultiBlock:
        """Resolve to a ``{'faces', 'edges'}`` CAD-view MultiBlock.

        ``faces`` is a per-face MultiBlock with analytic normals;
        ``edges`` the topological B-rep curves (with ``cad.edge_id`` /
        ``cad.edge_kind`` cell arrays). The data behind
        :meth:`plot` / ``plotter.cad.add``.

        Parameters
        ----------
        **kwargs
            ``linear_deflection``, ``angular_deflection``,
            ``feature_angle`` -- see :func:`pyvista_cad.as_cad_multiblock`.

        Returns
        -------
        pyvista.MultiBlock
            ``{'faces', 'edges'}``.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> mb = pv.MultiBlock([pv.Sphere()])
        >>> view = mb.cad.cad_view()
        >>> sorted(view.keys())
        ['edges', 'faces']

        """
        return _cad_view_impl(self, **kwargs)

    def _as_polydata(self) -> pv.PolyData:
        return _as_polydata(self._dataset)
