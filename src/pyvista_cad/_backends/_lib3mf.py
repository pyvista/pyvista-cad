"""3MF backend powered by lib3mf.

Reads ``.3mf`` files into :class:`pyvista.MultiBlock` (one block per
mesh object) and writes :class:`pyvista.PolyData` /
:class:`pyvista.MultiBlock` back as 3MF.
"""

import os
from typing import Any
import warnings

import numpy as np
import pyvista as pv

from pyvista_cad._errors import (
    CadReadError,
    CadWarning,
    CadWriteError,
    OptionalDependencyError,
)


def _require_lib3mf() -> Any:
    try:
        import lib3mf
    except ImportError as exc:
        msg = 'lib3mf not installed; install pyvista-cad[3mf]'
        raise OptionalDependencyError(msg) from exc
    return lib3mf


def _mesh_object_to_polydata(lib3mf: Any, obj: Any) -> pv.PolyData:
    # ``GetVertices`` / ``GetTriangleIndices`` return a Python ``list`` of
    # fixed-layout ctypes structs (``Position`` is one ``c_float[3]``,
    # ``Triangle`` is one ``c_uint[3]``). Reading ``.Coordinates[:]`` per
    # element in a Python list comprehension is a SWIG slice per vertex.
    # Packing the list into a single contiguous ctypes array once and
    # viewing it with ``np.frombuffer`` moves the per-element work into C.
    verts = obj.GetVertices()
    tris = obj.GetTriangleIndices()
    n_vert = len(verts)
    n_tri = len(tris)
    if n_vert:
        vbuf = (lib3mf.Position * n_vert)(*verts)
        pts = np.frombuffer(vbuf, dtype=np.float32).reshape(n_vert, 3).astype(np.float64)
    else:
        pts = np.empty((0, 3), dtype=np.float64)
    if n_tri:
        tbuf = (lib3mf.Triangle * n_tri)(*tris)
        idx = np.frombuffer(tbuf, dtype=np.uint32).reshape(n_tri, 3).astype(np.int64)
        faces = np.empty((n_tri, 4), dtype=np.int64)
        faces[:, 0] = 3
        faces[:, 1:] = idx
        faces_arr: np.ndarray | None = faces.ravel()
    else:
        faces_arr = None
    poly = pv.PolyData(pts, faces_arr)
    name = obj.GetName() or ''
    if name:
        poly.field_data['cad.label'] = np.array([name])
    poly.field_data['cad.source_format'] = np.array(['3mf'])
    poly.field_data['cad.backend'] = np.array(['lib3mf'])
    poly.field_data['cad.guid'] = np.array([obj.GetUUID()[1]])
    return poly


def _rgba_from_color(color: Any) -> np.ndarray:
    """Normalise a lib3mf ``Color`` (0-255 channels) to RGBA float in [0, 1]."""
    return np.array(
        [color.Red / 255.0, color.Green / 255.0, color.Blue / 255.0, color.Alpha / 255.0],
        dtype=np.float64,
    )


def _color_from_color_group(
    lib3mf: Any, model: Any, unique_rid: int, prop_id: int, obj_name: str
) -> np.ndarray | None:
    """Resolve an object-level color from a ColorGroup resource."""
    it = model.GetColorGroups()
    while it.MoveNext():
        cg = it.GetCurrentColorGroup()
        if cg.GetUniqueResourceID() != unique_rid:
            continue
        try:
            color = cg.GetColor(prop_id)
        except lib3mf.ELib3MFException:
            warnings.warn(
                f'3MF object {obj_name!r}: ColorGroup {unique_rid} is '
                'present but its color entry could not be read; the '
                "part's authored color is lost.",
                CadWarning,
                stacklevel=3,
            )
            return None
        return _rgba_from_color(color)
    warnings.warn(
        f'3MF object {obj_name!r}: object-level color references '
        f'ColorGroup {unique_rid}, which is absent from the model; the '
        "part's authored color is lost.",
        CadWarning,
        stacklevel=3,
    )
    return None


def _color_from_base_material(
    lib3mf: Any, model: Any, unique_rid: int, prop_id: int, obj_name: str
) -> np.ndarray | None:
    """Resolve an object-level color from a BaseMaterialGroup's display color.

    A part whose object-level property points at a BaseMaterialGroup
    carries its color in the referenced material's display color, not in
    a ColorGroup. ``prop_id`` is the 1-based material index within the
    group.
    """
    try:
        group = model.GetBaseMaterialGroupByID(unique_rid)
        color = group.GetDisplayColor(prop_id)
    except lib3mf.ELib3MFException:
        warnings.warn(
            f'3MF object {obj_name!r}: BaseMaterialGroup {unique_rid} is '
            'present but its display color could not be read; the '
            "part's authored color is lost.",
            CadWarning,
            stacklevel=3,
        )
        return None
    return _rgba_from_color(color)


def _object_level_color(lib3mf: Any, model: Any, obj: Any) -> np.ndarray | None:
    """Return an RGBA float array in [0, 1] for the object's object-level color, or None.

    Resolves both a ColorGroup property and a BaseMaterialGroup display
    color. Returns ``None`` when the object simply has no object-level
    color resource (the common, non-error case). When a color *is*
    declared but its resource cannot be resolved, the color is lost: a
    :class:`~pyvista_cad.CadWarning` naming the object is emitted rather
    than silently dropping it.
    """
    exc_type = lib3mf.ELib3MFException
    obj_name = obj.GetName() or obj.GetUUID()[1]

    # Absence of an object-level property is normal control flow, not an
    # error: ``has_prop`` is the documented "no property" signal.
    unique_rid, prop_id, has_prop = obj.GetObjectLevelProperty()
    if not has_prop or unique_rid == 0:
        return None

    try:
        ptype = model.GetPropertyTypeByID(unique_rid)
    except exc_type:
        warnings.warn(
            f'3MF object {obj_name!r}: declares an object-level property '
            f'(resource {unique_rid}) whose type is unresolvable; the '
            "part's authored color is lost.",
            CadWarning,
            stacklevel=2,
        )
        return None

    ptype_int = int(ptype)
    if ptype_int == int(lib3mf.PropertyType.Colors):
        return _color_from_color_group(lib3mf, model, unique_rid, prop_id, obj_name)
    if ptype_int == int(lib3mf.PropertyType.BaseMaterial):
        return _color_from_base_material(lib3mf, model, unique_rid, prop_id, obj_name)
    # Any other property type (texture coords, composite, ...) carries no
    # single resolvable color; this is not a failure.
    return None


def _transform_to_matrix(transform: Any) -> np.ndarray:
    """Convert a lib3mf 4x3 affine ``Transform`` to a row-major 4x4 numpy matrix.

    lib3mf stores the transform as rows 0..2 = basis vectors (columns of the
    rotation/scale), row 3 = translation. The standard 4x4 form has those
    basis vectors as the first three columns and the translation as the
    fourth column.
    """
    m = np.eye(4, dtype=np.float64)
    fields = np.array([list(transform.Fields[i]) for i in range(4)], dtype=np.float64)
    # fields[0..2] are basis columns; fields[3] is translation.
    m[:3, 0] = fields[0]
    m[:3, 1] = fields[1]
    m[:3, 2] = fields[2]
    m[:3, 3] = fields[3]
    return m


def _is_identity(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.allclose(matrix, np.eye(4), atol=tol))


def read_three_mf_internal(path: str | os.PathLike[str]) -> pv.MultiBlock:
    lib3mf = _require_lib3mf()
    w = lib3mf.get_wrapper()
    model = w.CreateModel()
    try:
        model.QueryReader('3mf').ReadFromFile(str(path))
    except Exception as exc:
        msg = f'lib3mf failed to read {path}: {exc}'
        raise CadReadError(msg) from exc

    units_idx = model.GetUnit()
    unit_names = {0: 'micron', 1: 'mm', 2: 'cm', 3: 'inch', 4: 'foot', 5: 'm'}
    # The 3MF model unit applies to every object, so propagate it to
    # every leaf block (mission goal 7: per-block units), not only the
    # MultiBlock root.
    model_unit = unit_names.get(int(units_idx), 'mm')

    mb = pv.MultiBlock()

    # Walk build items so per-instance transforms are honored. Each build item
    # references an object plus an optional transform; multiple build items
    # may reference the same object (instancing).
    build_it = model.GetBuildItems()
    have_build_item = False
    idx = 0
    while build_it.MoveNext():
        have_build_item = True
        item = build_it.GetCurrent()
        try:
            obj = item.GetObjectResource()
        except lib3mf.ELib3MFException as exc:
            warnings.warn(
                f'3MF build item {idx}: its object resource could not be '
                f'resolved ({exc}); this part is dropped from the result.',
                CadWarning,
                stacklevel=2,
            )
            idx += 1
            continue
        if not obj.IsMeshObject():
            # Components objects and other non-mesh resources are out of
            # scope for the Wave 2 reader.
            continue
        poly = _mesh_object_to_polydata(lib3mf, obj)
        color = _object_level_color(lib3mf, model, obj)
        if color is not None:
            poly.field_data['cad.color'] = color

        if item.HasObjectTransform():
            matrix = _transform_to_matrix(item.GetObjectTransform())
        else:
            matrix = np.eye(4, dtype=np.float64)
        # Preserve the original 4x4 transform on the round-trip path even
        # when applying it to the points below.
        poly.field_data['cad.transform'] = matrix.astype(np.float64)
        if not _is_identity(matrix):
            poly = poly.transform(matrix, inplace=False)
            # ``transform`` strips field_data on some pyvista versions; restore.
            poly.field_data['cad.transform'] = matrix.astype(np.float64)

        poly.field_data['cad.units'] = np.array([model_unit])
        name = obj.GetName() or f'object_{idx}'
        mb.append(poly, name=name)
        idx += 1

    # Fallback for files with no build items (rare, but defensive).
    if not have_build_item:
        it = model.GetMeshObjects()
        while it.MoveNext():
            obj = it.GetCurrentMeshObject()
            poly = _mesh_object_to_polydata(lib3mf, obj)
            color = _object_level_color(lib3mf, model, obj)
            if color is not None:
                poly.field_data['cad.color'] = color
            poly.field_data['cad.units'] = np.array([model_unit])
            name = obj.GetName() or f'object_{idx}'
            mb.append(poly, name=name)
            idx += 1

    mb.field_data['cad.source_format'] = np.array(['3mf'])
    mb.field_data['cad.backend'] = np.array(['lib3mf'])
    mb.field_data['cad.units'] = np.array([model_unit])
    return mb


def _identity_transform(lib3mf: Any) -> Any:
    t = lib3mf.Transform()
    t.Fields[0][:] = [1.0, 0.0, 0.0]
    t.Fields[1][:] = [0.0, 1.0, 0.0]
    t.Fields[2][:] = [0.0, 0.0, 1.0]
    t.Fields[3][:] = [0.0, 0.0, 0.0]
    return t


def _color_to_lib3mf(lib3mf: Any, rgba: np.ndarray) -> Any:
    """Build a lib3mf ``Color`` (0-255 channels) from an RGBA float array.

    Parameters
    ----------
    lib3mf : module
        The imported ``lib3mf`` module.
    rgba : numpy.ndarray
        RGBA values in ``[0, 1]``. Shorter arrays are padded with an
        opaque alpha.

    Returns
    -------
    lib3mf.Color
        The equivalent 8-bit-per-channel color.
    """
    vals = np.asarray(rgba, dtype=np.float64).ravel()
    if vals.size < 4:
        vals = np.concatenate([vals, np.ones(4 - vals.size, dtype=np.float64)])
    channels = np.clip(np.rint(vals[:4] * 255.0), 0, 255).astype(np.uint8)
    color = lib3mf.Color()
    color.Red = int(channels[0])
    color.Green = int(channels[1])
    color.Blue = int(channels[2])
    color.Alpha = int(channels[3])
    return color


def _assign_object_color(model: Any, lib3mf: Any, obj: Any, rgba: np.ndarray, name: str) -> None:
    """Attach an object-level color to a mesh object via a BaseMaterialGroup.

    The color is stored as a single-material BaseMaterialGroup display
    color and bound through the object-level property. This is exactly
    the representation :func:`_color_from_base_material` reads back, so
    the round-trip is symmetric.
    """
    color = _color_to_lib3mf(lib3mf, rgba)
    group = model.AddBaseMaterialGroup()
    prop_id = group.AddMaterial(name or 'material', color)
    obj.SetObjectLevelProperty(group.GetUniqueResourceID(), prop_id)


def _add_polydata_as_mesh_object(model: Any, lib3mf: Any, mesh: pv.PolyData, name: str) -> Any:
    obj = model.AddMeshObject()
    obj.SetName(name)
    tri = mesh.triangulate()
    points = np.asarray(tri.points, dtype=np.float64)
    if tri.n_cells:
        faces = np.asarray(tri.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    else:
        faces = np.zeros((0, 3), dtype=np.int64)

    positions = []
    for x, y, z in points:
        p = lib3mf.Position()
        p.Coordinates = (float(x), float(y), float(z))
        positions.append(p)
    triangles = []
    for a, b, c in faces:
        t = lib3mf.Triangle()
        t.Indices = (int(a), int(b), int(c))
        triangles.append(t)
    obj.SetGeometry(positions, triangles)
    return obj


def write_three_mf_internal(
    dataset: pv.PolyData | pv.MultiBlock,
    path: str | os.PathLike[str],
    *,
    units: str | None = None,
) -> None:
    lib3mf = _require_lib3mf()
    w = lib3mf.get_wrapper()
    model = w.CreateModel()

    unit_map = {'micron': 0, 'mm': 1, 'cm': 2, 'inch': 3, 'foot': 4, 'm': 5}
    chosen_units = units
    if chosen_units is None and 'cad.units' in dataset.field_data:
        chosen_units = str(dataset.field_data['cad.units'][0])
    if chosen_units in unit_map:
        try:
            model.SetUnit(unit_map[chosen_units])
        except lib3mf.ELib3MFException as exc:
            warnings.warn(
                f'3MF writer: lib3mf rejected unit {chosen_units!r} ({exc}); '
                'the file is written with the lib3mf default unit (mm).',
                CadWarning,
                stacklevel=2,
            )

    def _block_color(block: Any) -> np.ndarray | None:
        # Capture color before any surface extraction, which can strip
        # field_data on some pyvista versions.
        if 'cad.color' in block.field_data:
            return np.asarray(block.field_data['cad.color'], dtype=np.float64)
        return None

    if isinstance(dataset, pv.MultiBlock):
        blocks: list[tuple[str, pv.PolyData, np.ndarray | None]] = []
        for i, block in enumerate(dataset):
            if block is None:
                continue
            name = dataset.get_block_name(i) or f'object_{i}'
            color = _block_color(block)
            if isinstance(block, pv.MultiBlock):
                combined = block.combine(merge_points=True)
                surface = combined.extract_surface(algorithm='dataset_surface')
            elif isinstance(block, pv.PolyData):
                surface = block
            else:
                surface = block.extract_surface(algorithm='dataset_surface')
            blocks.append((name, surface, color))
    elif isinstance(dataset, pv.PolyData):
        blocks = [('object_0', dataset, _block_color(dataset))]
    else:
        msg = (
            f'write_three_mf requires a pv.PolyData or pv.MultiBlock; got {type(dataset).__name__}'
        )
        raise CadWriteError(msg)

    ident = _identity_transform(lib3mf)
    for name, poly, color in blocks:
        obj = _add_polydata_as_mesh_object(model, lib3mf, poly, name)
        if color is not None:
            _assign_object_color(model, lib3mf, obj, color, name)
        model.AddBuildItem(obj, ident)

    try:
        model.QueryWriter('3mf').WriteToFile(str(path))
    except Exception as exc:
        msg = f'lib3mf failed to write {path}: {exc}'
        raise CadWriteError(msg) from exc
