"""cascadio STEP backend: end-to-end read plus the flattening helper.

``_flatten_to_multiblock`` is a pure transform and is exercised directly
(no cascadio needed). The end-to-end ``read_step(..., backend='cascadio')``
path requires the optional ``cascadio`` dependency and skips without it.
"""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._backends._cascadio import _flatten_to_multiblock


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


def test_read_step_cascadio_matches_build123d() -> None:
    """Both STEP backends place a multi-part assembly in the same coordinates."""
    pytest.importorskip('cascadio')
    pytest.importorskip('build123d')
    path = Path(__file__).parents[1] / 'data' / 'assembly_two_parts.step'
    light = pyvista_cad.read_step(path, backend='cascadio')
    full = pyvista_cad.read_step(path, backend='build123d')
    assert str(light.field_data['cad.units'][0]) == str(full.field_data['cad.units'][0])
    np.testing.assert_allclose(light.bounds, full.bounds, atol=1e-6)
