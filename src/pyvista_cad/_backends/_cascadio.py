"""STEP reader backed by cascadio (lightweight alternative path).

cascadio loads STEP and emits a glTF binary that PyVista can parse via
its native glTF reader. We strip the data back out into a MultiBlock
keyed on cascadio's mesh names.
"""

import json
import os
import pathlib
import re
import struct
import tempfile
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_cad._backends._build123d import _detect_step_units
from pyvista_cad._errors import CadReadError, OptionalDependencyError

_METRES_PER_UNIT = {'mm': 1e-3, 'cm': 1e-2, 'm': 1.0, 'inch': 0.0254, 'foot': 0.3048}

_INDEX_DTYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
_MESH_BLOCK = re.compile(r'Mesh_(\d+)$')


def read_step(
    path: str | os.PathLike[str],
    *,
    linear_deflection: float,
    angular_deflection: float,
    merge: bool,
) -> pv.MultiBlock | pv.PolyData:
    """Read a STEP file via :mod:`cascadio` (read-only, no OCP).

    Returns
    -------
    pyvista.MultiBlock or pyvista.PolyData
        ``MultiBlock`` of parts, or a single ``PolyData`` when
        ``merge`` is true.

    """
    try:
        import cascadio
    except ImportError as exc:
        msg = 'cascadio not installed; install pyvista-cad[step-light]'
        raise OptionalDependencyError(msg) from exc

    fname = os.fspath(path)
    if not os.path.exists(fname):
        msg = f'No such file: {fname}'
        raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory() as tmp:
        glb_path = pathlib.Path(tmp) / 'out.glb'
        try:
            cascadio.step_to_glb(
                fname,
                str(glb_path),
                tol_linear=linear_deflection,
                tol_angular=angular_deflection,
                include_brep=True,
            )
        except Exception as exc:
            msg = f'cascadio failed to read STEP {fname}: {exc}'
            raise CadReadError(msg) from exc
        loaded: Any = pv.read(str(glb_path))
        _tag_face_ids(loaded, _read_face_ids(glb_path))

    # cascadio writes glTF in metres; return the file's declared unit.
    units = _detect_step_units(fname)
    if units not in _METRES_PER_UNIT:
        units = 'm'
    to_file = 1.0 / _METRES_PER_UNIT[units]

    mb = _flatten_to_multiblock(loaded)
    mb.field_data['cad.source_format'] = np.array(['step'])
    mb.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
    mb.field_data['cad.backend'] = np.array(['cascadio'])
    mb.field_data['cad.units'] = np.array([units])
    for i in range(len(mb)):
        block = mb[i]
        if block is None:
            continue
        block.points *= to_file
        block.field_data['cad.source_format'] = np.array(['step'])
        block.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
        block.field_data['cad.backend'] = np.array(['cascadio'])
        block.field_data['cad.units'] = np.array([units])
        name = mb.get_block_name(i)
        if name:
            block.field_data['cad.label'] = np.array([name])

    if merge:
        offset = 0
        for block in mb:
            if block is not None and 'cad.face_id' in block.cell_data:
                block.cell_data['cad.face_id'] += offset
                offset = int(block.cell_data['cad.face_id'].max()) + 1
        combined = mb.combine(merge_points=True).extract_surface(algorithm='dataset_surface')
        combined.field_data['cad.source_format'] = np.array(['step'])
        combined.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
        combined.field_data['cad.backend'] = np.array(['cascadio'])
        combined.field_data['cad.units'] = np.array([units])
        return combined
    return mb


def _read_face_ids(glb_path: pathlib.Path) -> dict[int, list[np.ndarray | None]]:
    """Per-triangle B-rep face indices from cascadio's ``TM_brep_faces`` extension.

    Keyed on glTF mesh index, one entry per primitive (``None`` where the
    primitive carries no face indices). VTK's glTF loader ignores the
    extension, so the accessors are read from the GLB directly.
    """
    data = glb_path.read_bytes()
    (json_len,) = struct.unpack_from('<I', data, 12)
    doc = json.loads(data[20 : 20 + json_len])
    bin_start = 20 + json_len + 8

    out: dict[int, list[np.ndarray | None]] = {}
    for mesh_index, mesh in enumerate(doc.get('meshes', [])):
        prims: list[np.ndarray | None] = []
        for prim in mesh['primitives']:
            ext = prim.get('extensions', {}).get('TM_brep_faces')
            if ext is None:
                prims.append(None)
                continue
            acc = doc['accessors'][ext['faceIndices']]
            view = doc['bufferViews'][acc['bufferView']]
            start = bin_start + view.get('byteOffset', 0) + acc.get('byteOffset', 0)
            dtype = _INDEX_DTYPES[acc['componentType']]
            prims.append(np.frombuffer(data, dtype, acc['count'], start).astype(np.int64))
        out[mesh_index] = prims
    return out


def _tag_face_ids(obj: Any, face_ids: dict[int, list[np.ndarray | None]]) -> None:
    """Attach ``cad.face_id`` to each primitive block of a glTF read.

    VTK names a glTF mesh's group ``Mesh_<index>`` and orders its
    children by primitive, one cell per triangle in index order.
    """
    if not isinstance(obj, pv.MultiBlock):
        return
    for i in range(len(obj)):
        block = obj[i]
        match = _MESH_BLOCK.search(obj.get_block_name(i) or '')
        if match is None or not isinstance(block, pv.MultiBlock):
            _tag_face_ids(block, face_ids)
            continue
        prims = face_ids.get(int(match.group(1)), [])
        for j in range(min(len(block), len(prims))):
            leaf, ids = block[j], prims[j]
            if isinstance(leaf, pv.PolyData) and ids is not None and len(ids) == leaf.n_cells:
                leaf.cell_data['cad.face_id'] = ids


def _flatten_to_multiblock(obj: Any) -> pv.MultiBlock:
    """Coerce a PyVista glTF read result into a flat MultiBlock."""
    if isinstance(obj, pv.MultiBlock):
        out = pv.MultiBlock()
        for i in range(len(obj)):
            block = obj[i]
            name = obj.get_block_name(i) or f'part_{i}'
            if isinstance(block, pv.MultiBlock):
                sub = _flatten_to_multiblock(block)
                for j in range(len(sub)):
                    sub_name = sub.get_block_name(j) or f'part_{i}_{j}'
                    out.append(sub[j], name=f'{name}/{sub_name}')
            elif block is not None:
                out.append(block, name=name)
        return out
    mb = pv.MultiBlock()
    mb.append(obj, name='part_0')
    return mb
