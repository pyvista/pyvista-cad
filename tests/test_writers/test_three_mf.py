"""3MF writer round-trips (INV-025: moved out of tests/test_roundtrip/).

These assert the *writer* contract: ``write_three_mf`` followed by
``read_three_mf`` restores geometry, block names and ``cad.*`` metadata.
The reader-side ground-truth tests (object color / build-item transform
synthesised directly through lib3mf) stay in
tests/test_roundtrip/test_three_mf_roundtrip.py.
"""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('lib3mf')


def _sphere() -> pv.PolyData:
    return pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()


def test_write_three_mf_round_trip_preserves_vertex_count(tmp_path: Path) -> None:
    """Vertex count survives the round-trip exactly (no merging in the writer)."""
    src = _sphere()
    path = tmp_path / 's.3mf'
    pyvista_cad.write_three_mf(src, path)
    rt = pyvista_cad.read_three_mf(path)
    assert rt[0].n_points == src.n_points


def test_write_three_mf_round_trip_preserves_triangle_count(tmp_path: Path) -> None:
    """Triangle count survives exactly."""
    src = _sphere()
    path = tmp_path / 's.3mf'
    pyvista_cad.write_three_mf(src, path)
    rt = pyvista_cad.read_three_mf(path)
    assert rt[0].n_cells == src.n_cells


def test_write_three_mf_round_trip_preserves_metadata(tmp_path: Path) -> None:
    """Units, source format and per-block label survive the writer
    round-trip."""
    mb_in = pv.MultiBlock({'box': pv.Cube().triangulate(), 'cone': pv.Cone().triangulate()})
    path = tmp_path / 'parts.3mf'
    pyvista_cad.write_three_mf(mb_in, path, units='mm')
    rt = pyvista_cad.read_three_mf(path)
    assert rt.n_blocks == 2
    assert sorted(rt.keys()) == ['box', 'cone']
    assert str(rt.field_data['cad.units'][0]) == 'mm'
    assert str(rt.field_data['cad.source_format'][0]) == '3mf'
    assert str(rt['box'].field_data['cad.label'][0]) == 'box'
    assert str(rt['cone'].field_data['cad.label'][0]) == 'cone'


def test_write_three_mf_pv_read_dispatch(tmp_path: Path) -> None:
    """``pv.read`` reads the written 3MF back to a one-block MultiBlock."""
    path = tmp_path / 'cube.3mf'
    pyvista_cad.write_three_mf(pv.Cube().triangulate(), path)
    rt = pv.read(path)
    assert isinstance(rt, pv.MultiBlock)
    assert rt.n_blocks == 1
