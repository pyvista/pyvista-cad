"""glTF read invariants and node-transform fidelity.

A glTF writer is not implemented; we exercise the reader against a
hand-built GLB containing multiple named nodes with non-identity
transforms.
"""

import json
from pathlib import Path
import struct

import numpy as np
import pyvista as pv

import pyvista_cad


def _build_glb(path: Path, nodes: list[dict]) -> None:
    """Write a multi-node, multi-mesh GLB. Each entry of ``nodes`` is a dict
    with optional ``name``, ``matrix`` (16 floats, column-major) and a
    ``mesh`` index that gets paired one-to-one with a tiny triangle mesh.
    """
    n = len(nodes)
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    bin_pts = triangle.tobytes()
    bin_idx = indices.tobytes()
    if len(bin_idx) % 4:
        bin_idx += b'\x00' * (4 - len(bin_idx) % 4)

    # One mesh per node, each sharing the same vertices/indices buffer views.
    bin_blob = bin_pts + bin_idx

    prim = {'attributes': {'POSITION': 0}, 'indices': 1, 'mode': 4}
    meshes = [{'primitives': [prim]} for _ in range(n)]
    accessors = [
        {
            'bufferView': 0,
            'componentType': 5126,
            'count': 3,
            'type': 'VEC3',
            'min': [0, 0, 0],
            'max': [1, 1, 0],
        },
        {
            'bufferView': 1,
            'componentType': 5123,
            'count': 3,
            'type': 'SCALAR',
        },
    ]
    bufferviews = [
        {'buffer': 0, 'byteOffset': 0, 'byteLength': len(bin_pts)},
        {'buffer': 0, 'byteOffset': len(bin_pts), 'byteLength': len(indices) * 2},
    ]
    gltf_nodes = []
    for i, spec in enumerate(nodes):
        node: dict = {'mesh': i}
        if 'name' in spec:
            node['name'] = spec['name']
        if 'matrix' in spec:
            node['matrix'] = spec['matrix']
        gltf_nodes.append(node)

    gltf = {
        'asset': {'version': '2.0'},
        'scene': 0,
        'scenes': [{'nodes': list(range(n))}],
        'nodes': gltf_nodes,
        'meshes': meshes,
        'accessors': accessors,
        'bufferViews': bufferviews,
        'buffers': [{'byteLength': len(bin_blob)}],
    }
    json_bytes = json.dumps(gltf).encode('utf-8')
    while len(json_bytes) % 4:
        json_bytes += b' '

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<III', 0x46546C67, 2, total))
        fh.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))
        fh.write(json_bytes)
        fh.write(struct.pack('<II', len(bin_blob), 0x004E4942))
        fh.write(bin_blob)


def test_gltf_node_count_preserved(tmp_path: Path) -> None:
    """Each glTF node becomes a MultiBlock entry."""
    glb = tmp_path / 'multi.glb'
    _build_glb(
        glb,
        [
            {'name': 'a'},
            {'name': 'b'},
            {'name': 'c'},
        ],
    )
    dataset = pv.read(str(glb))
    out = pyvista_cad.enrich_gltf(dataset, glb)
    # The dataset is a MultiBlock; the read path attaches a label per node.
    assert isinstance(out, pv.MultiBlock)
    assert out.n_blocks == 3


def test_gltf_node_transforms_preserved(tmp_path: Path) -> None:
    """Per-node ``matrix`` survives on each block's ``cad.transform`` to 1e-6."""
    # glTF stores matrix as column-major flat 16-float.
    src_matrix = np.eye(4, dtype=np.float64)
    src_matrix[:3, 3] = [4.0, -2.0, 7.0]
    # Column-major flatten.
    flat = src_matrix.T.reshape(16).tolist()

    glb = tmp_path / 'xform.glb'
    _build_glb(glb, [{'name': 'shifted', 'matrix': flat}])
    dataset = pv.read(str(glb))
    out = pyvista_cad.enrich_gltf(dataset, glb)
    block = out[0]
    assert 'cad.transform' in block.field_data
    matrix = np.asarray(block.field_data['cad.transform'])
    np.testing.assert_allclose(matrix, src_matrix, atol=1e-6)


def test_gltf_no_double_transform(tmp_path: Path) -> None:
    """VTK already bakes the world transform; ``enrich_gltf`` must not re-apply.

    Triangle ``[(0,0,0), (1,0,0), (0,1,0)]`` translated by ``[4,-2,7]``
    must land with bounds center at ``(4.5, -1.5, 7.0)``. If the
    enrichment re-applied the matrix the center would be ``(8.5, -3.5, 14.0)``.
    """
    src_matrix = np.eye(4, dtype=np.float64)
    src_matrix[:3, 3] = [4.0, -2.0, 7.0]
    flat = src_matrix.T.reshape(16).tolist()

    glb = tmp_path / 'no_double.glb'
    _build_glb(glb, [{'name': 'shifted', 'matrix': flat}])
    dataset = pv.read(str(glb))
    out = pyvista_cad.enrich_gltf(dataset, glb)
    block = out[0]
    np.testing.assert_allclose(block.center, (4.5, -1.5, 7.0), atol=1e-6)
