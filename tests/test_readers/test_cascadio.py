"""cascadio STEP backend: end-to-end read plus the flattening helper.

``_flatten_to_multiblock`` is a pure transform and is exercised directly
(no cascadio needed). The end-to-end ``read_step(..., backend='cascadio')``
path requires the optional ``cascadio`` dependency and skips without it.
"""

import json
from pathlib import Path
import struct

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._backends._cascadio import (
    _flatten_to_multiblock,
    _offset_face_ids,
    _read_face_ids,
)
from pyvista_cad._errors import CadWarning

FACE_COLORS_STEP = Path(__file__).parents[1] / 'data' / 'assembly_face_colors.step'
#: B-rep faces per leaf: two Housing instances (colored face, the other five), Plate, Pin.
FACE_COLORS_LEAF_FACES = [1, 5, 1, 5, 6, 3]


def test_flatten_to_multiblock_flattens_nested_paths() -> None:
    """A nested MultiBlock collapses to one flat level with ``parent/child``
    path names; named and unnamed blocks both resolve."""
    inner = pv.MultiBlock()
    inner.append(pv.Sphere(), name='leaf_a')
    inner.append(pv.Cube(), name='leaf_b')

    outer = pv.MultiBlock()
    outer.append(inner, name='group')
    outer.append(pv.Cone(), name='solo')

    flat = _flatten_to_multiblock(outer)
    assert isinstance(flat, pv.MultiBlock)
    names = list(flat.keys())
    assert names == ['group/leaf_a', 'group/leaf_b', 'solo']
    assert all(isinstance(b, pv.PolyData) for b in flat)


def test_flatten_to_multiblock_wraps_bare_polydata() -> None:
    """A bare PolyData becomes a one-block MultiBlock named ``part_0``."""
    flat = _flatten_to_multiblock(pv.Sphere())
    assert isinstance(flat, pv.MultiBlock)
    assert list(flat.keys()) == ['part_0']


def test_read_step_cascadio_backend_e2e() -> None:
    """``read_step(backend='cascadio')`` reads a real STEP fixture to a
    bounded, non-empty mesh with the cascadio backend recorded."""
    pytest.importorskip('cascadio')
    path = Path(pyvista_cad.examples.step_cube)
    out = pyvista_cad.read_step(path, backend='cascadio')
    combined = out.combine() if isinstance(out, pv.MultiBlock) else out
    assert combined.n_cells > 0
    assert combined.n_points > 0
    assert str(out.field_data['cad.backend'][0]) == 'cascadio'
    assert str(out.field_data['cad.source_format'][0]) == 'step'
    assert str(out.field_data['cad.units'][0]) == 'mm'
    assert all(str(b.field_data['cad.units'][0]) == 'mm' for b in out)
    np.testing.assert_allclose(combined.bounds, [-5, 5, -5, 5, -5, 5], atol=1e-6)


@pytest.mark.parametrize(('prefix', 'units', 'half'), [('.CENTI.', 'cm', 5.0), ('$', 'm', 5.0)])
def test_read_step_cascadio_returns_declared_unit(
    tmp_path: Path, prefix: str, units: str, half: float
) -> None:
    """Coordinates come back in the file's declared length unit, not glTF metres."""
    pytest.importorskip('cascadio')
    text = Path(pyvista_cad.examples.step_cube).read_text()
    path = tmp_path / f'cube_{units}.step'
    path.write_text(text.replace('SI_UNIT(.MILLI.,.METRE.)', f'SI_UNIT({prefix},.METRE.)'))
    out = pyvista_cad.read_step(path, backend='cascadio', merge=True)
    assert str(out.field_data['cad.units'][0]) == units
    np.testing.assert_allclose(out.bounds, [-half, half] * 3, atol=1e-6)


def test_read_step_cascadio_tags_brep_faces() -> None:
    """Each triangle carries its B-rep face, so a box reads as 6 faces with
    12 edges and a pin as 3 faces with 2 circular edges and no seam."""
    pytest.importorskip('cascadio')
    path = Path(__file__).parents[1] / 'data' / 'assembly_two_parts.step'
    base, pin = pyvista_cad.read_step(path, backend='cascadio')
    for part, n_faces, n_edges in [(base, 6, 12), (pin, 3, 2)]:
        assert part.cell_data['cad.face_id'].size == part.n_cells
        assert np.unique(part.cell_data['cad.face_id']).size == n_faces
        edges = pyvista_cad.face_id_edges(part)
        assert np.unique(edges.cell_data['cad.edge_id']).size == n_edges


def test_read_step_cascadio_merge_keeps_face_ids_distinct() -> None:
    pytest.importorskip('cascadio')
    path = Path(__file__).parents[1] / 'data' / 'assembly_two_parts.step'
    merged = pyvista_cad.read_step(path, backend='cascadio', merge=True)
    assert np.unique(merged.cell_data['cad.face_id']).size == 9


def test_offset_face_ids_leaves_shared_instance_arrays_alone() -> None:
    """Two instances of one glTF mesh share a buffer; each gets its own range."""
    part = pv.Cube().triangulate()
    part.cell_data['cad.face_id'] = np.repeat(np.arange(6), 2)
    mb = pv.MultiBlock([part, part.copy(deep=False)])
    _offset_face_ids(mb)
    assert mb[0].cell_data['cad.face_id'].tolist() == np.repeat(np.arange(6), 2).tolist()
    assert mb[1].cell_data['cad.face_id'].min() == 6


def test_read_step_cascadio_matches_build123d() -> None:
    """Both STEP backends place a multi-part assembly in the same coordinates."""
    pytest.importorskip('cascadio')
    pytest.importorskip('build123d')
    path = Path(__file__).parents[1] / 'data' / 'assembly_two_parts.step'
    light = pyvista_cad.read_step(path, backend='cascadio')
    full = pyvista_cad.read_step(path, backend='build123d')
    assert str(light.field_data['cad.units'][0]) == str(full.field_data['cad.units'][0])
    np.testing.assert_allclose(light.bounds, full.bounds, atol=1e-6)


@pytest.mark.parametrize('backend', ['cascadio', 'build123d'])
def test_read_step_default_tessellation_follows_model_size(tmp_path: Path, backend: str) -> None:
    """The same shape declared in metres tessellates like its millimetre original."""
    pytest.importorskip(backend, exc_type=ImportError)
    source = Path(pyvista_cad.examples.bracket_step_path())
    path = tmp_path / 'bracket_m.step'
    path.write_text(source.read_text().replace('SI_UNIT(.MILLI.,', 'SI_UNIT($,'))
    small = pyvista_cad.read_step(source, backend=backend, merge=True)
    large = pyvista_cad.read_step(path, backend=backend, merge=True)
    assert large.n_cells == small.n_cells
    fixed = pyvista_cad.read_step(path, backend=backend, merge=True, linear_deflection=0.1)
    assert fixed.n_cells > 2 * large.n_cells


def test_read_step_cascadio_returns_active_normals() -> None:
    """Each part carries the B-rep surface normals as the active ``Normals``."""
    pytest.importorskip('cascadio')
    out = pyvista_cad.read_step(pyvista_cad.examples.bracket_step_path(), backend='cascadio')
    for block in out:
        assert block.point_data.active_normals_name == 'Normals'
        assert 'NORMAL' not in block.point_data
        normals = np.asarray(block.point_data['Normals'])
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


def _glb(
    path: Path,
    primitives: list[list[tuple[int | None, int | None]]],
    triangles: list[int],
    face_arrays: list[list[int]],
) -> Path:
    """Write a GLB of ``(indices, faceIndices)`` accessor pairs, one list per mesh.

    The accessors are one index accessor per ``triangles`` entry, then ``face_arrays``.
    """
    accessors: list[dict] = [{'count': 3 * count} for count in triangles]
    views: list[dict] = []
    body = b''
    for array in face_arrays:
        accessors.append({'bufferView': len(views), 'componentType': 5125, 'count': len(array)})
        views.append({'byteOffset': len(body)})
        body += np.asarray(array, dtype=np.uint32).tobytes()
    meshes = []
    for mesh in primitives:
        prims: list[dict] = []
        for indices, faces in mesh:
            prims.append({} if indices is None else {'indices': indices})
            if faces is not None:
                prims[-1]['extensions'] = {'TM_brep_faces': {'faceIndices': faces}}
        meshes.append({'primitives': prims})
    text = json.dumps({'meshes': meshes, 'accessors': accessors, 'bufferViews': views}).encode()
    text += b' ' * (-len(text) % 4)
    chunks = struct.pack('<II', len(text), 0x4E4F534A) + text
    chunks += struct.pack('<II', len(body), 0x004E4942) + body
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, 12 + len(chunks)) + chunks)
    return path


def _as_lists(face_ids: dict) -> list[list[list[int] | None]]:
    return [[None if a is None else a.tolist() for a in face_ids[i]] for i in sorted(face_ids)]


def test_read_face_ids_repairs_arrays_of_equal_length(tmp_path: Path) -> None:
    """A later primitive shifts the written pairing, and length alone cannot see it.

    The first mesh has two primitives, so cascadio hands the second mesh the
    array of the first mesh's second primitive. All three arrays are one
    length, so only the index accessors tell them apart.
    """
    arrays = [[0, 1], [2, 3], [4, 5]]
    glb = _glb(tmp_path / 'm.glb', [[(0, 3), (1, None)], [(2, 4)]], [2, 2, 2], arrays)
    assert _as_lists(_read_face_ids(glb)) == [[[0, 1], [2, 3]], [[4, 5]]]


def test_read_face_ids_shares_arrays_between_instances(tmp_path: Path) -> None:
    """Meshes that reuse a primitive's index accessor read the same array."""
    primitives = [[(0, 2), (1, None)], [(0, 2), (1, None)]]
    glb = _glb(tmp_path / 'm.glb', primitives, [1, 2], [[7], [8, 9]])
    assert _as_lists(_read_face_ids(glb)) == [[[7], [8, 9]], [[7], [8, 9]]]


def test_read_face_ids_trusts_an_extension_on_every_primitive(tmp_path: Path) -> None:
    """A writer that tags each primitive itself is read as written, in any accessor order."""
    glb = _glb(tmp_path / 'm.glb', [[(0, 3), (1, 2)]], [1, 1], [[5], [6]])
    assert _as_lists(_read_face_ids(glb)) == [[[6], [5]]]


def test_read_face_ids_trusts_single_primitive_meshes(tmp_path: Path) -> None:
    """Without a later primitive nothing shifts, so an untagged mesh stays untagged."""
    glb = _glb(tmp_path / 'm.glb', [[(0, 2)], [(1, None)]], [1, 1], [[5]])
    assert _as_lists(_read_face_ids(glb)) == [[[5]], [None]]


def test_read_face_ids_without_the_extension(tmp_path: Path) -> None:
    """A read with no B-rep data has nothing to repair and nothing to warn about."""
    glb = _glb(tmp_path / 'm.glb', [[(0, None), (1, None)]], [1, 1], [])
    assert _as_lists(_read_face_ids(glb)) == [[None, None]]


@pytest.mark.parametrize(
    'face_arrays',
    [[[1], [2, 3, 4]], [[1], [2, 3], [4]]],
    ids=['array-of-the-wrong-length', 'more-arrays-than-primitives'],
)
def test_read_face_ids_refuses_a_layout_it_cannot_pair(
    tmp_path: Path, face_arrays: list[list[int]]
) -> None:
    """An unknown layout tags nothing and says so, rather than tagging the wrong faces."""
    glb = _glb(tmp_path / 'm.glb', [[(0, 2), (1, None)]], [1, 2], face_arrays)
    with pytest.warns(CadWarning, match='B-rep face'):
        face_ids = _read_face_ids(glb)
    assert _as_lists(face_ids) == [[None, None]]


def test_read_face_ids_refuses_a_primitive_without_indices(tmp_path: Path) -> None:
    """Rank needs an index accessor on every primitive; without one nothing is paired."""
    glb = _glb(tmp_path / 'm.glb', [[(0, 1), (None, None)]], [1], [[1]])
    with pytest.warns(CadWarning, match='B-rep face'):
        face_ids = _read_face_ids(glb)
    assert _as_lists(face_ids) == [[None, None]]


def test_read_step_cascadio_tags_every_primitive() -> None:
    """Every leaf of a multi-color assembly keeps its ids and its own B-rep faces."""
    pytest.importorskip('cascadio')
    leaves = list(pyvista_cad.read_step(FACE_COLORS_STEP, backend='cascadio'))
    assert all('cad.face_id' in leaf.cell_data for leaf in leaves)
    counts = [np.unique(leaf.cell_data['cad.face_id']).size for leaf in leaves]
    assert counts == FACE_COLORS_LEAF_FACES


def test_read_step_cascadio_face_ids_follow_the_geometry() -> None:
    """A box leaf has one normal per face id, which a neighbor's array would not give it."""
    pytest.importorskip('cascadio')
    leaves = list(pyvista_cad.read_step(FACE_COLORS_STEP, backend='cascadio'))
    for leaf in leaves[:5]:
        normals = np.round(leaf.cell_normals, 4)
        ids = leaf.cell_data['cad.face_id']
        for face in np.unique(ids):
            assert np.unique(normals[ids == face], axis=0).shape[0] == 1
        assert np.unique(normals, axis=0).shape[0] == np.unique(ids).size


def test_read_step_cascadio_merge_keeps_every_face() -> None:
    """A merged read has every B-rep face of every leaf under its own id."""
    pytest.importorskip('cascadio')
    merged = pyvista_cad.read_step(FACE_COLORS_STEP, backend='cascadio', merge=True)
    assert np.unique(merged.cell_data['cad.face_id']).size == sum(FACE_COLORS_LEAF_FACES)


def test_read_step_cascadio_edges_after_a_multi_color_part() -> None:
    """Parts after a multi-color one draw their edges: 12 on a box, 2 circles on a pin."""
    pytest.importorskip('cascadio')
    plate, pin = list(pyvista_cad.read_step(FACE_COLORS_STEP, backend='cascadio'))[4:]
    for part, n_edges in [(plate, 12), (pin, 2)]:
        edges = pyvista_cad.face_id_edges(part)
        assert np.unique(edges.cell_data['cad.edge_id']).size == n_edges
