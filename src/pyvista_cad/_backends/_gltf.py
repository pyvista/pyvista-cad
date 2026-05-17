"""Enrichment of glTF datasets with manifest metadata.

PyVista reads ``.gltf`` and ``.glb`` natively via VTK's glTF importer,
but the importer drops node names and ``extras`` payloads. This
module parses the glTF JSON (or GLB chunk) separately and exposes
the per-node metadata so callers can attach ``cad.label`` and
``cad.extras`` to dataset field data. The base geometry is read by
instantiating :class:`pyvista.GLTFReader` directly rather than via
:func:`pyvista.read`, so the enriching reader can claim ``.gltf`` /
``.glb`` without recursing back through the reader registry.
"""

import json
import os
from pathlib import Path
import struct
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadReadError

_GLB_MAGIC = 0x46546C67


def _load_gltf_json(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == '.glb':
        if len(data) < 20:
            msg = f'glb file too short: {path}'
            raise CadReadError(msg)
        magic, version, _length = struct.unpack('<III', data[:12])
        if magic != _GLB_MAGIC:
            msg = f'not a glb file (bad magic): {path}'
            raise CadReadError(msg)
        if version != 2:
            msg = f'unsupported glb version {version}: {path}'
            raise CadReadError(msg)
        chunk_len, chunk_type = struct.unpack('<II', data[12:20])
        if chunk_type != 0x4E4F534A:
            msg = f'first glb chunk is not JSON: {path}'
            raise CadReadError(msg)
        return json.loads(data[20 : 20 + chunk_len])
    try:
        return json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f'failed to parse glTF JSON {path}: {exc}'
        raise CadReadError(msg) from exc


def _quat_to_matrix(q: list[float]) -> np.ndarray:
    """Build a 4x4 rotation matrix from a glTF quaternion ``[x, y, z, w]``."""
    x, y, z, w = q
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y - z * w)
    m[0, 2] = 2 * (x * z + y * w)
    m[1, 0] = 2 * (x * y + z * w)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z - x * w)
    m[2, 0] = 2 * (x * z - y * w)
    m[2, 1] = 2 * (y * z + x * w)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _node_local_matrix(node: dict[str, Any]) -> np.ndarray:
    """Return the node's local 4x4 transform from ``matrix`` or TRS."""
    if 'matrix' in node and node['matrix'] is not None:
        # glTF stores matrix as column-major 16-float; numpy default row-major.
        return np.asarray(node['matrix'], dtype=np.float64).reshape(4, 4, order='F')
    m = np.eye(4, dtype=np.float64)
    if 'scale' in node and node['scale'] is not None:
        s = np.asarray(node['scale'], dtype=np.float64)
        scale = np.diag([s[0], s[1], s[2], 1.0])
        m = scale @ m
    if 'rotation' in node and node['rotation'] is not None:
        m = _quat_to_matrix(list(node['rotation'])) @ m
    if 'translation' in node and node['translation'] is not None:
        t = np.asarray(node['translation'], dtype=np.float64)
        trans = np.eye(4, dtype=np.float64)
        trans[:3, 3] = t
        m = trans @ m
    return m


def gltf_node_metadata(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return per-node metadata pulled directly from the glTF manifest.

    Each entry has ``name`` (node name, possibly empty), ``mesh``
    (mesh index or ``None``), ``extras`` (parsed extras dict or
    ``None``), ``index`` (node index), and ``world_matrix`` (4x4
    accumulated world transform).

    Returns
    -------
    list of dict
        One metadata dict per glTF node.

    """
    p = Path(path)
    if not p.exists():
        msg = f'glTF file not found: {path}'
        raise CadReadError(msg)
    manifest = _load_gltf_json(p)
    nodes = manifest.get('nodes', []) or []
    scenes = manifest.get('scenes', []) or []
    default_scene_idx = manifest.get('scene', 0)
    if scenes:
        scene_roots = scenes[default_scene_idx].get('nodes', []) or []
    else:
        # Fall back to treating every non-child node as a root.
        all_children: set[int] = set()
        for n in nodes:
            for c in n.get('children', []) or []:
                all_children.add(c)
        scene_roots = [i for i in range(len(nodes)) if i not in all_children]

    world: dict[int, np.ndarray] = {}

    def walk(idx: int, parent: np.ndarray) -> None:
        if idx < 0 or idx >= len(nodes):
            return
        local = _node_local_matrix(nodes[idx])
        w = parent @ local
        world[idx] = w
        for child in nodes[idx].get('children', []) or []:
            walk(child, w)

    identity = np.eye(4, dtype=np.float64)
    for root in scene_roots:
        walk(root, identity)

    out: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        out.append(
            {
                'index': i,
                'name': node.get('name', ''),
                'mesh': node.get('mesh'),
                'extras': node.get('extras'),
                'world_matrix': world.get(i, np.eye(4, dtype=np.float64)),
            }
        )
    return out


def enrich_gltf(
    dataset: pv.DataSet,
    path: str | os.PathLike[str],
) -> pv.DataSet:
    """Attach glTF node names / extras to a dataset's field data.

    The first node with a mesh provides ``cad.label``; the document
    becomes ``cad.source_format = 'gltf'``. When *dataset* is a
    :class:`pyvista.MultiBlock`, block names are filled from node
    names in order.

    Parameters
    ----------
    dataset : pyvista.DataSet
        The dataset returned by :meth:`pyvista.GLTFReader.read`.
    path : str or os.PathLike
        Same path that was read; used to pull node metadata.

    Returns
    -------
    pyvista.DataSet
        The input dataset, mutated in place and returned for chaining.

    """
    nodes = gltf_node_metadata(path)
    dataset.field_data['cad.source_format'] = np.array(['gltf'])
    dataset.field_data['cad.backend'] = np.array(['pyvista'])
    named = [n for n in nodes if n.get('name')]
    if named:
        dataset.field_data['cad.label'] = np.array([named[0]['name']])
    extras = [n['extras'] for n in nodes if n.get('extras')]
    if extras:
        dataset.field_data['cad.extras'] = np.array([json.dumps(extras)])

    if isinstance(dataset, pv.MultiBlock):
        meshed = [n for n in nodes if n.get('mesh') is not None]
        for i in range(min(dataset.n_blocks, len(meshed))):
            node = meshed[i]
            name = node.get('name') or f'node_{i}'
            dataset.set_block_name(i, name)
            block = dataset[i]
            if block is None:
                continue
            matrix = np.asarray(node.get('world_matrix'), dtype=np.float64)
            if matrix.shape != (4, 4):
                matrix = np.eye(4, dtype=np.float64)
            # VTK's glTF importer already bakes per-node world transforms
            # into the point coordinates. Applying ``block.transform(matrix)``
            # here would double-transform. We only stamp ``cad.transform``
            # for downstream consumers that want the node-local matrix.
            block.field_data['cad.label'] = np.array([name])
            block.field_data['cad.source_format'] = np.array(['gltf'])
            block.field_data['cad.backend'] = np.array(['pyvista'])
            block.field_data['cad.transform'] = matrix.astype(np.float64)
            block.field_data['cad.gltf_node_index'] = np.array(
                [int(node.get('index', i))], dtype=np.int64
            )
            if node.get('extras'):
                block.field_data['cad.extras'] = np.array([json.dumps(node['extras'])])
    return dataset


def read_gltf_internal(path: str | os.PathLike[str]) -> pv.DataSet:
    """Read a glTF/GLB file with pyvista and enrich it with manifest metadata.

    Uses :class:`pyvista.GLTFReader` directly rather than
    :func:`pyvista.read`. ``pyvista_cad`` registers this reader for
    ``.gltf`` / ``.glb``, so going through ``pyvista.read`` would
    dispatch back here and recurse forever.

    Returns
    -------
    pyvista.DataSet
        The parsed mesh with ``cad.*`` manifest metadata attached.

    """
    dataset = pv.GLTFReader(str(path)).read()
    return enrich_gltf(dataset, path)
