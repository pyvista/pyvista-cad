"""DXF read/write backed by ezdxf.

Implements the DXF reader (every drawable entity → ``pv.PolyData`` with
a ``Layer`` cell-data array) and the DXF writer (vertex/line/face cells
→ DXF primitives).
"""

from collections.abc import Iterable
import os
from typing import Any
import warnings

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadReadError, CadWarning, CadWriteError
from pyvista_cad._metadata import insunits_code, insunits_to_str

_DXF_VERSION_LABEL: dict[str, str] = {
    'AC1009': 'R12',
    'AC1012': 'R13',
    'AC1014': 'R14',
    'AC1015': 'R2000',
    'AC1018': 'R2004',
    'AC1021': 'R2007',
    'AC1024': 'R2010',
    'AC1027': 'R2013',
    'AC1032': 'R2018',
}


def read_dxf_internal(
    path: str | os.PathLike[str],
    *,
    clean: bool,
    keep_text: bool,
    expand_blocks: bool,
    layout: str,
) -> pv.PolyData:
    """Read a DXF file into a single :class:`pyvista.PolyData`.

    Returns
    -------
    pyvista.PolyData
        All drawable entities merged into one mesh, with ``Layer``
        cell data and ``cad.*`` field data.

    """
    import ezdxf
    from ezdxf import recover
    from ezdxf.disassemble import recursive_decompose, to_primitives

    fname = os.fspath(path)
    if not os.path.exists(fname):
        msg = f'No such file: {fname}'
        raise FileNotFoundError(msg)

    try:
        try:
            doc = ezdxf.readfile(fname)
        except (ezdxf.DXFStructureError, StopIteration):
            # StopIteration leaks out of ezdxf's stream parser on truncated
            # files; fall through to the recovery path.
            doc, _auditor = recover.readfile(fname)
    except (ezdxf.DXFError, OSError, StopIteration, ValueError) as exc:
        # Normalise every non-DXF / unparseable-bytes failure into a single
        # ``CadReadError``. ``OSError`` is raised by ezdxf on random bytes
        # ("File is not a DXF file."); ``StopIteration`` on truncated streams;
        # ``ValueError`` on header garbage.
        msg = f'failed to read DXF {fname}: {exc}'
        raise CadReadError(msg) from exc

    space = _select_layout(doc, layout)

    entities = space if not expand_blocks else recursive_decompose(space)
    primitives = to_primitives(entities)

    points_buf: list[tuple[float, float, float]] = []
    verts_idx: list[int] = []
    lines_idx: list[int] = []
    polys_idx: list[int] = []
    layers_verts: list[str] = []
    layers_lines: list[str] = []
    layers_polys: list[str] = []
    text_polys: list[str] = []
    acis_skipped = 0
    dropped_prims = 0

    for prim in primitives:
        try:
            verts_iter = list(prim.vertices())
        except (ezdxf.DXFError, ValueError, TypeError, ArithmeticError):
            # ezdxf raises DXFError on a structurally broken entity and
            # ValueError / TypeError / ArithmeticError on degenerate
            # geometry (e.g. a zero-radius arc). The entity is dropped.
            ent = getattr(prim, 'entity', None)
            if ent is None or ent.dxftype() not in {'REGION', 'BODY', '3DSOLID'}:
                dropped_prims += 1
            verts_iter = []
        if not verts_iter:
            entity = getattr(prim, 'entity', None)
            if entity is not None and entity.dxftype() in {'REGION', 'BODY', '3DSOLID'}:
                acis_skipped += 1
            continue

        layer = _entity_layer(prim)
        pts = [(float(v.x), float(v.y), float(v.z)) for v in verts_iter]

        if len(pts) == 1:
            base = len(points_buf)
            points_buf.append(pts[0])
            verts_idx.extend([1, base])
            layers_verts.append(layer)
            continue

        if _looks_like_face(pts):
            face_pts = pts[:-1] if _closed(pts) else pts
            if len(face_pts) < 3:
                continue
            base = len(points_buf)
            points_buf.extend(face_pts)
            polys_idx.append(len(face_pts))
            polys_idx.extend(range(base, base + len(face_pts)))
            layers_polys.append(layer)
            entity = getattr(prim, 'entity', None)
            if (
                keep_text
                and entity is not None
                and entity.dxftype()
                in {
                    'TEXT',
                    'MTEXT',
                    'ATTRIB',
                }
            ):
                text_polys.append(_entity_text(entity))
            else:
                text_polys.append('')
            continue

        base = len(points_buf)
        points_buf.extend(pts)
        lines_idx.append(len(pts))
        lines_idx.extend(range(base, base + len(pts)))
        layers_lines.append(layer)

    if not points_buf:
        poly = pv.PolyData()
    else:
        points = np.asarray(points_buf, dtype=float)
        poly = pv.PolyData()
        poly.points = points
        if verts_idx:
            poly.verts = np.asarray(verts_idx, dtype=np.int64)
        if lines_idx:
            poly.lines = np.asarray(lines_idx, dtype=np.int64)
        if polys_idx:
            poly.faces = np.asarray(polys_idx, dtype=np.int64)

    layer_array = np.asarray(layers_verts + layers_lines + layers_polys, dtype=object)
    if layer_array.size > 0:
        poly.cell_data['Layer'] = layer_array
        poly.cell_data['cad.layer'] = layer_array

    if keep_text and text_polys and any(t != '' for t in text_polys):
        full = [''] * len(layers_verts) + [''] * len(layers_lines) + text_polys
        poly.cell_data['Text'] = np.asarray(full, dtype=object)

    if clean and poly.n_points:
        poly = poly.clean()

    poly.field_data['cad.source_format'] = np.array(['dxf'])
    poly.field_data['cad.reader'] = np.array(['pyvista_cad.read_dxf'])
    poly.field_data['cad.backend'] = np.array(['ezdxf'])
    poly.field_data['cad.source_format_version'] = np.array(
        [_DXF_VERSION_LABEL.get(doc.dxfversion, doc.dxfversion)]
    )
    insunits = int(doc.header.get('$INSUNITS', 0))
    poly.field_data['cad.units'] = np.array([insunits_to_str(insunits)])

    if acis_skipped:
        warnings.warn(
            f'{acis_skipped} ACIS solid entities in {fname} were skipped; '
            'pyvista-cad does not decode embedded ACIS bodies.',
            CadWarning,
            stacklevel=3,
        )

    if dropped_prims:
        warnings.warn(
            f'{dropped_prims} DXF entities in {fname} could not be converted '
            'to geometry (broken or degenerate definition) and were dropped '
            'from the result.',
            CadWarning,
            stacklevel=3,
        )

    return poly


def write_dxf_internal(
    mesh: pv.PolyData,
    path: str | os.PathLike[str],
    *,
    layer: str,
    layer_array: str | None,
    dxf_version: str,
    units: str | None,
    color_index: int | None,
) -> None:
    """Write a :class:`pyvista.PolyData` as a DXF file."""
    import ezdxf

    if not isinstance(mesh, pv.PolyData):
        msg = f'write_dxf expects a PolyData, got {type(mesh).__name__}'
        raise CadWriteError(msg)

    try:
        doc = ezdxf.new(dxfversion=dxf_version)
    except ezdxf.DXFError as exc:
        msg = f'unsupported DXF version {dxf_version!r}: {exc}'
        raise CadWriteError(msg) from exc

    resolved_units = units
    if resolved_units is None and 'cad.units' in mesh.field_data:
        resolved_units = str(mesh.field_data['cad.units'][0])
    doc.header['$INSUNITS'] = insunits_code(resolved_units)

    msp = doc.modelspace()

    per_cell_layer: np.ndarray | None = None
    if layer_array is not None and layer_array in mesh.cell_data:
        per_cell_layer = np.asarray(mesh.cell_data[layer_array], dtype=object)

    cell_idx = 0
    common_kwargs: dict[str, Any] = {}
    if color_index is not None:
        common_kwargs['color'] = int(color_index)

    points = mesh.points

    for vert_pts in _iter_vert_cells(mesh):
        kw = {**common_kwargs, 'layer': _resolve_layer(layer, per_cell_layer, cell_idx)}
        msp.add_point(tuple(points[vert_pts[0]]), dxfattribs=kw)
        cell_idx += 1

    for line_pts in _iter_line_cells(mesh):
        kw = {**common_kwargs, 'layer': _resolve_layer(layer, per_cell_layer, cell_idx)}
        coords = [tuple(points[i]) for i in line_pts]
        if len(coords) == 2:
            msp.add_line(coords[0], coords[1], dxfattribs=kw)
        else:
            msp.add_polyline3d(coords, dxfattribs=kw)
        cell_idx += 1

    for face_pts in _iter_face_cells(mesh):
        kw = {**common_kwargs, 'layer': _resolve_layer(layer, per_cell_layer, cell_idx)}
        coords = [tuple(points[i]) for i in face_pts]
        if len(coords) == 3:
            msp.add_3dface([*coords, coords[-1]], dxfattribs=kw)
        elif len(coords) == 4:
            msp.add_3dface(coords, dxfattribs=kw)
        else:
            mesh_entity = msp.add_mesh(dxfattribs=kw)
            with mesh_entity.edit_data() as data:
                data.vertices = coords
                data.faces = [list(range(len(coords)))]
        cell_idx += 1

    try:
        doc.saveas(os.fspath(path))
    except OSError as exc:
        msg = f'failed to write DXF {path}: {exc}'
        raise CadWriteError(msg) from exc


def _select_layout(doc: Any, layout: str) -> Any:
    """Return the modelspace, active paperspace, or a named layout."""
    if layout == 'modelspace':
        return doc.modelspace()
    if layout == 'paperspace':
        return doc.layouts.active_layout()
    if layout in doc.layout_names():
        return doc.layouts.get(layout)
    msg = f'unknown DXF layout {layout!r}; known: {sorted(doc.layout_names())}'
    raise CadReadError(msg)


def _entity_layer(prim: Any) -> str:
    entity = getattr(prim, 'entity', None)
    if entity is None:
        return '0'
    dxf_attrs = getattr(entity, 'dxf', None)
    if dxf_attrs is None:
        return '0'
    return str(getattr(dxf_attrs, 'layer', '0'))


def _entity_text(entity: Any) -> str:
    """Extract the display string of a TEXT / MTEXT / ATTRIB entity.

    Returns ``''`` when the entity carries no text attribute. When a text
    accessor (e.g. ``MTEXT.plain_text``) is present but raises while
    decoding inline formatting, the text is lost: a :class:`~pyvista_cad.CadWarning`
    is emitted naming the entity rather than dropping it silently.
    """
    import ezdxf

    saw_accessor = False
    for attr in ('text', 'plain_text', 'value'):
        val = getattr(entity, attr, None)
        if callable(val):
            saw_accessor = True
            try:
                return str(val())
            except (ezdxf.DXFError, ValueError, TypeError):
                continue
        if isinstance(val, str):
            return val
    if saw_accessor:
        warnings.warn(
            f'DXF {entity.dxftype()} entity: its text content could not be '
            'decoded (malformed inline formatting) and is dropped from the '
            'result.',
            CadWarning,
            stacklevel=3,
        )
    return ''


def _closed(pts: list[tuple[float, float, float]]) -> bool:
    if len(pts) < 4:
        return False
    a, b = pts[0], pts[-1]
    return abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12 and abs(a[2] - b[2]) < 1e-12


def _looks_like_face(pts: list[tuple[float, float, float]]) -> bool:
    if len(pts) == 3:
        return True
    return bool(len(pts) == 4 and _closed(pts))


def _iter_vert_cells(mesh: pv.PolyData) -> Iterable[list[int]]:
    if mesh.verts is None or len(mesh.verts) == 0:
        return
    arr = np.asarray(mesh.verts)
    i = 0
    while i < arr.size:
        n = int(arr[i])
        yield [int(x) for x in arr[i + 1 : i + 1 + n]]
        i += 1 + n


def _iter_line_cells(mesh: pv.PolyData) -> Iterable[list[int]]:
    if mesh.lines is None or len(mesh.lines) == 0:
        return
    arr = np.asarray(mesh.lines)
    i = 0
    while i < arr.size:
        n = int(arr[i])
        yield [int(x) for x in arr[i + 1 : i + 1 + n]]
        i += 1 + n


def _iter_face_cells(mesh: pv.PolyData) -> Iterable[list[int]]:
    if mesh.faces is None or len(mesh.faces) == 0:
        return
    arr = np.asarray(mesh.faces)
    i = 0
    while i < arr.size:
        n = int(arr[i])
        yield [int(x) for x in arr[i + 1 : i + 1 + n]]
        i += 1 + n


def _resolve_layer(default: str, per_cell: np.ndarray | None, idx: int) -> str:
    if per_cell is None or idx >= len(per_cell):
        return default
    val = per_cell[idx]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val)
