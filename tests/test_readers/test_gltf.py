"""Tests for glTF metadata enrichment."""

import json
from pathlib import Path
import struct

import numpy as np
import pyvista as pv

import pyvista_cad
from pyvista_cad._backends._gltf import gltf_node_metadata


def _write_minimal_glb(path: Path, *, name: str, extras: dict | None = None) -> None:
    """Write a syntactically valid GLB with one named node referencing one mesh."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2], dtype=np.uint16)
    bin_pts = points.tobytes()
    bin_idx = indices.tobytes()
    if len(bin_idx) % 4:
        bin_idx += b'\x00' * (4 - len(bin_idx) % 4)
    bin_blob = bin_pts + bin_idx

    node: dict = {'name': name, 'mesh': 0}
    if extras is not None:
        node['extras'] = extras

    gltf = {
        'asset': {'version': '2.0'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [node],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1, 'mode': 4}]}],
        'accessors': [
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
        ],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': 0, 'byteLength': len(bin_pts)},
            {'buffer': 0, 'byteOffset': len(bin_pts), 'byteLength': len(indices) * 2},
        ],
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


def test_gltf_node_metadata_extracts_names(tmp_path: Path):
    glb = tmp_path / 'thing.glb'
    _write_minimal_glb(glb, name='widget', extras={'ifc_type': 'IfcWall', 'guid': 'g1'})
    nodes = gltf_node_metadata(glb)
    assert len(nodes) == 1
    assert nodes[0]['name'] == 'widget'
    assert nodes[0]['extras']['ifc_type'] == 'IfcWall'


def test_enrich_gltf_attaches_label(tmp_path: Path):
    glb = tmp_path / 'labeled.glb'
    _write_minimal_glb(glb, name='hinge', extras={'k': 'v'})
    dataset = pv.read(str(glb))
    out = pyvista_cad.enrich_gltf(dataset, glb)
    assert str(out.field_data['cad.source_format'][0]) == 'gltf'
    assert str(out.field_data['cad.label'][0]) == 'hinge'
    extras_json = str(out.field_data['cad.extras'][0])
    assert 'k' in extras_json


def test_read_gltf_round_trip(tmp_path: Path):
    glb = tmp_path / 'r.glb'
    _write_minimal_glb(glb, name='thing')
    dataset = pyvista_cad.read_gltf(glb)
    assert str(dataset.field_data['cad.label'][0]) == 'thing'


def test_read_gltf_invalid_json_raises(tmp_path: Path):
    """A .gltf file whose body is not valid JSON raises ``CadReadError``.

    Exercised through ``gltf_node_metadata`` (the manifest-side surface);
    ``read_gltf`` calls VTK's importer first which throws a less specific
    error before our manifest parser ever sees the bad bytes.
    """
    import pytest  # noqa: PLC0415

    from pyvista_cad import CadReadError  # noqa: PLC0415

    bad = tmp_path / 'broken.gltf'
    bad.write_text('{this is : not, json at all,,,')
    with pytest.raises(CadReadError):
        gltf_node_metadata(bad)


def test_read_gltf_glb_bad_magic_raises(tmp_path: Path):
    """A .glb file with wrong magic header raises ``CadReadError``."""
    import pytest  # noqa: PLC0415

    from pyvista_cad import CadReadError  # noqa: PLC0415

    bad = tmp_path / 'wrong_magic.glb'
    # 20+ bytes so we get past the length check and hit the magic check.
    bad.write_bytes(b'\x00' * 20 + b'extra payload')
    with pytest.raises(CadReadError):
        gltf_node_metadata(bad)


def test_pv_read_dispatches_to_enriching_reader_without_recursion(tmp_path: Path):
    """``pv.read`` on .glb routes through the registered reader.

    The reader reads base geometry via ``pv.GLTFReader`` directly, so
    claiming the extension does not recurse back through ``pv.read``.
    A returned, enriched dataset proves both: dispatch happened and the
    call terminated.
    """
    import warnings  # noqa: PLC0415

    src = tmp_path / 'd.glb'
    _write_minimal_glb(src, name='Widget')
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # builtin-override warning would fail here
        out = pv.read(str(src))
    assert str(out.field_data['cad.source_format'][0]) == 'gltf'
    assert str(out.field_data['cad.label'][0]) == 'Widget'


def test_read_gltf_internal_uses_gltf_reader_not_pv_read(monkeypatch, tmp_path: Path):
    """``read_gltf_internal`` must not call ``pv.read`` (recursion guard)."""
    from pyvista_cad._backends import _gltf  # noqa: PLC0415

    def _boom(*_a, **_k):  # pragma: no cover - only hit on regression
        msg = 'read_gltf_internal must use pv.GLTFReader, not pv.read'
        raise AssertionError(msg)

    monkeypatch.setattr(_gltf.pv, 'read', _boom)
    glb = tmp_path / 'g.glb'
    _write_minimal_glb(glb, name='ok')
    out = _gltf.read_gltf_internal(glb)
    assert str(out.field_data['cad.label'][0]) == 'ok'
