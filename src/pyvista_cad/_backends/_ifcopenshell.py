"""IFC backend powered by ifcopenshell."""

from datetime import date, datetime, time
from decimal import Decimal
import json
import os
from typing import Any
import warnings

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadReadError, CadWarning, OptionalDependencyError

_BACKEND = 'ifcopenshell'


class _PsetJSONEncoder(json.JSONEncoder):
    """JSON encoder that preserves numeric precision for IFC psets.

    ``json.dumps(..., default=str)`` round-trips IFC measure wrappers
    (and ``Decimal``) as strings, which collapses precision and breaks
    downstream consumers. This encoder:

    - unwraps ifcopenshell measure objects via ``.wrappedValue`` so the
      underlying ``float`` / ``int`` lands in the JSON as a number;
    - converts ``Decimal`` to ``float`` (full precision preserved);
    - serialises ``datetime`` / ``date`` / ``time`` as ISO 8601 strings;
    - falls back to ``str()`` for anything else (matches old behaviour).
    """

    def default(self, o: Any) -> Any:
        # Unwrap ifcopenshell measure / typed wrappers first so the
        # numeric payload gets handled as a number, not a string.
        wrapped = getattr(o, 'wrappedValue', None)
        if wrapped is not None and not isinstance(wrapped, type(o)):
            return wrapped
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (datetime, date, time)):
            return o.isoformat()
        if isinstance(o, bytes):
            return o.decode('utf-8', errors='replace')
        return str(o)


# Spatial structure traversal order for the nested MultiBlock tree. Entities
# whose ``is_a()`` does not match one of these layers fall through to the
# element pass and are attached to their containing structure.
_SPATIAL_CLASSES = (
    'IfcProject',
    'IfcSite',
    'IfcBuilding',
    'IfcBuildingStorey',
    'IfcSpace',
)


def _require_ifc() -> Any:
    try:
        import ifcopenshell
        import ifcopenshell.geom as geom
        import ifcopenshell.util.element as util_element
        import ifcopenshell.util.unit as util_unit
    except ImportError as exc:
        msg = 'ifcopenshell not installed; install pyvista-cad[ifc]'
        raise OptionalDependencyError(msg) from exc
    return ifcopenshell, geom, util_element, util_unit


# Map IFC SI/conversion unit names to our canonical Units literal.
_SI_NAME_MAP = {
    'METRE': 'm',
    'METER': 'm',
}
_SI_PREFIX_MAP = {
    None: 'm',
    'MILLI': 'mm',
    'CENTI': 'cm',
    'MICRO': 'micron',
}
_CONVERSION_NAME_MAP = {
    'INCH': 'inch',
    'INCHES': 'inch',
    'FOOT': 'foot',
    'FEET': 'foot',
    'MILLIMETRE': 'mm',
    'MILLIMETER': 'mm',
    'CENTIMETRE': 'cm',
    'CENTIMETER': 'cm',
    'MICROMETRE': 'micron',
    'MICROMETER': 'micron',
}


def _resolve_units(model: Any, util_unit: Any) -> str | None:
    # ``get_project_unit`` returns ``None`` (not an exception) when the
    # model declares no length unit, so absence is normal control flow.
    # A raised ``RuntimeError`` / ``AttributeError`` means the unit
    # assignment is structurally broken: the unit is lost, so warn.
    try:
        unit = util_unit.get_project_unit(model, 'LENGTHUNIT')
    except (RuntimeError, AttributeError) as exc:
        warnings.warn(
            f'IFC: the project LENGTHUNIT assignment is malformed ({exc}); '
            'geometry is read in unitless coordinates.',
            CadWarning,
            stacklevel=2,
        )
        return None
    if unit is None:
        return None
    cls = unit.is_a()
    if cls == 'IfcSIUnit':
        prefix = getattr(unit, 'Prefix', None)
        name = (getattr(unit, 'Name', '') or '').upper()
        if name in {'METRE', 'METER'}:
            return _SI_PREFIX_MAP.get(prefix, 'm') if prefix in _SI_PREFIX_MAP else None
        return _SI_NAME_MAP.get(name)
    if cls == 'IfcConversionBasedUnit':
        raw_name = (getattr(unit, 'Name', '') or '').upper()
        return _CONVERSION_NAME_MAP.get(raw_name)
    return None


def _shape_to_polydata(shape: Any) -> pv.PolyData:
    geo = shape.geometry
    verts = np.asarray(geo.verts, dtype=np.float64)
    if verts.size == 0:
        return pv.PolyData()
    pts = verts.reshape(-1, 3)
    tri_idx = np.asarray(geo.faces, dtype=np.int64).reshape(-1, 3)
    n_tri = len(tri_idx)
    faces = np.empty((n_tri, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = tri_idx
    poly = pv.PolyData(pts, faces.ravel())
    mats = list(geo.materials)
    if mats:
        diffuse = getattr(mats[0], 'diffuse', None)
        if diffuse is not None:
            # ifcopenshell's ``colour`` exposes its RGB triple via the
            # ``.components`` sequence. A diffuse object that does not
            # yield three components is a malformed material and its
            # color is lost.
            comps = getattr(diffuse, 'components', None)
            try:
                rgb = [float(c) for c in comps] if comps is not None else []
            except (TypeError, ValueError):
                rgb = []
            if len(rgb) >= 3:
                poly.field_data['cad.color'] = np.array(rgb[:3], dtype=np.float64)
            else:
                guid = getattr(shape, 'guid', '?')
                warnings.warn(
                    f'IFC element {guid!r}: diffuse material color is '
                    f'malformed ({diffuse!r}); the element renders without '
                    'its authored color.',
                    CadWarning,
                    stacklevel=2,
                )
    return poly


def _material_name(product: Any) -> str | None:
    rels = getattr(product, 'HasAssociations', None) or ()
    for rel in rels:
        if rel.is_a() != 'IfcRelAssociatesMaterial':
            continue
        mat = rel.RelatingMaterial
        if mat is None:
            continue
        name = getattr(mat, 'Name', None)
        if name:
            return str(name)
        # MaterialLayerSet / List / Constituent fallbacks
        for attr in ('ForLayerSet', 'Materials', 'MaterialConstituents', 'MaterialLayers'):
            sub = getattr(mat, attr, None)
            if sub:
                first = sub[0] if hasattr(sub, '__getitem__') else next(iter(sub))
                sub_name = getattr(first, 'Name', None)
                if sub_name:
                    return str(sub_name)
    return None


def _stamp_element_fields(
    poly: pv.PolyData,
    product: Any,
    units: str | None,
    util_element: Any,
) -> None:
    poly.field_data['cad.backend'] = np.array([_BACKEND])
    poly.field_data['cad.source_format'] = np.array(['ifc'])
    poly.field_data['cad.guid'] = np.array([str(product.GlobalId)])
    poly.field_data['cad.ifc_type'] = np.array([product.is_a()])
    label = getattr(product, 'Name', None) or product.is_a()
    poly.field_data['cad.label'] = np.array([str(label)])
    if units:
        poly.field_data['cad.units'] = np.array([units])
    mat = _material_name(product)
    if mat:
        poly.field_data['cad.material'] = np.array([mat])
    try:
        psets = util_element.get_psets(product)
    except (RuntimeError, AttributeError, TypeError) as exc:
        warnings.warn(
            f'IFC product {product.GlobalId!r}: property sets could not be '
            f'resolved ({exc}); cad.psets is omitted for this element.',
            CadWarning,
            stacklevel=2,
        )
        psets = None
    if psets:
        poly.field_data['cad.psets'] = np.array([json.dumps(psets, cls=_PsetJSONEncoder)])


def _stamp_container_fields(mb: pv.MultiBlock, entity: Any, units: str | None) -> None:
    mb.field_data['cad.backend'] = np.array([_BACKEND])
    mb.field_data['cad.source_format'] = np.array(['ifc'])
    if units:
        mb.field_data['cad.units'] = np.array([units])
    if entity is not None:
        mb.field_data['cad.guid'] = np.array([str(entity.GlobalId)])
        mb.field_data['cad.ifc_type'] = np.array([entity.is_a()])
        label = getattr(entity, 'Name', None) or entity.is_a()
        mb.field_data['cad.label'] = np.array([str(label)])


def _build_geom_map(
    model: Any,
    geom: Any,
    *,
    include: tuple[str, ...] | None,
    use_world_coords: bool,
    units: str | None,
    util_element: Any,
) -> dict[str, pv.PolyData]:
    settings = geom.settings()
    # Only ``use-world-coords=True`` is a meaningful request; the
    # ifcopenshell default is local coordinates, so a rejected
    # ``False`` is a no-op, not a loss. A rejected ``True`` means the
    # installed ifcopenshell does not honor the option and geometry
    # comes back in local (un-placed) coordinates: warn.
    try:
        settings.set('use-world-coords', use_world_coords)
    except (RuntimeError, TypeError, KeyError) as exc:
        if use_world_coords:
            warnings.warn(
                'IFC: this ifcopenshell build rejected the '
                f'use-world-coords setting ({exc}); geometry is returned '
                'in local coordinates and product placements are not '
                'applied.',
                CadWarning,
                stacklevel=2,
            )

    if include is not None:
        elements = [el for cls in include for el in model.by_type(cls)]
        if not elements:
            return {}
        iterator = geom.iterator(settings, model, include=elements)
    else:
        iterator = geom.iterator(settings, model)

    out: dict[str, pv.PolyData] = {}
    if not iterator.initialize():
        return out
    while True:
        shape = iterator.get()
        try:
            product = model.by_guid(shape.guid)
        except RuntimeError as exc:
            warnings.warn(
                f'IFC: geometry GUID {shape.guid!r} has no matching '
                f'product in the model ({exc}); this element is rendered '
                'without its label, material, units, or property sets.',
                CadWarning,
                stacklevel=2,
            )
            product = None
        poly = _shape_to_polydata(shape)
        if product is not None:
            _stamp_element_fields(poly, product, units, util_element)
        out[str(shape.guid)] = poly
        if not iterator.next():
            break
    return out


def _walk_spatial(
    entity: Any,
    geom_map: dict[str, pv.PolyData],
    units: str | None,
    consumed: set[str],
) -> pv.MultiBlock:
    """Build a nested MultiBlock mirroring the IFC spatial decomposition."""
    container = pv.MultiBlock()
    _stamp_container_fields(container, entity, units)

    # Sub-containers via IsDecomposedBy (aggregation).
    for rel in getattr(entity, 'IsDecomposedBy', None) or ():
        for child in rel.RelatedObjects:
            if child.is_a() in _SPATIAL_CLASSES:
                sub = _walk_spatial(child, geom_map, units, consumed)
                name = getattr(child, 'Name', None) or child.is_a()
                container.append(sub, name=str(name))
            else:
                # Aggregated element (e.g. a column aggregated to a storey).
                _append_element(container, child, geom_map, consumed)

    # Contained elements via ContainsElements.
    for rel in getattr(entity, 'ContainsElements', None) or ():
        for child in rel.RelatedElements:
            _append_element(container, child, geom_map, consumed)

    return container


def _append_element(
    container: pv.MultiBlock,
    element: Any,
    geom_map: dict[str, pv.PolyData],
    consumed: set[str],
) -> None:
    guid = str(getattr(element, 'GlobalId', '') or '')
    poly = geom_map.get(guid)
    if poly is None or poly.n_points == 0:
        return
    consumed.add(guid)
    name = getattr(element, 'Name', None) or element.is_a()
    container.append(poly, name=str(name))


def read_ifc_internal(
    path: str | os.PathLike[str],
    *,
    include: tuple[str, ...] | None = None,
    use_world_coords: bool = True,
) -> pv.MultiBlock:
    ifcopenshell, geom, util_element, util_unit = _require_ifc()
    try:
        f = ifcopenshell.open(str(path))
    except (ifcopenshell.Error, RuntimeError, OSError, ValueError) as exc:
        # ifcopenshell.Error: malformed / non-IFC SPF content.
        # OSError: unreadable path. RuntimeError / ValueError: lower-level
        # parser failures. Re-raised as CadReadError, never swallowed.
        msg = f'ifcopenshell failed to open {path}: {exc}'
        raise CadReadError(msg) from exc

    units = _resolve_units(f, util_unit)
    geom_map = _build_geom_map(
        f,
        geom,
        include=include,
        use_world_coords=use_world_coords,
        units=units,
        util_element=util_element,
    )

    # When an include filter was supplied and matched nothing, return an
    # empty MultiBlock rather than a stamped, structurally-empty tree.
    if include is not None and not geom_map:
        return pv.MultiBlock()

    projects = f.by_type('IfcProject')
    if projects:
        root_proj = projects[0]
        consumed: set[str] = set()
        mb = _walk_spatial(root_proj, geom_map, units, consumed)
        # Append any orphaned geometry (no spatial container) flat.
        for guid, poly in geom_map.items():
            if guid in consumed or poly.n_points == 0:
                continue
            # A by_guid miss here is already diagnosed in _build_geom_map
            # (same guid); fall back to the guid as a stable block name
            # without re-warning.
            try:
                product = f.by_guid(guid)
                name = getattr(product, 'Name', None) or product.is_a()
            except RuntimeError:
                name = guid
            mb.append(poly, name=str(name))
        return mb

    # No project root: fall back to flat structure.
    mb = pv.MultiBlock()
    _stamp_container_fields(mb, None, units)
    for guid, poly in geom_map.items():
        if poly.n_points == 0:
            continue
        # See the orphan-geometry note above: a by_guid miss is already
        # diagnosed in _build_geom_map, so fall back silently here.
        try:
            product = f.by_guid(guid)
            name = getattr(product, 'Name', None) or product.is_a()
        except RuntimeError:
            name = guid
        mb.append(poly, name=str(name))
    return mb
