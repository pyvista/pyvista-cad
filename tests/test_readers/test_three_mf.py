"""Tests for the 3MF reader and writer."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('lib3mf')


def _src_polydata() -> pv.PolyData:
    return pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()


def test_three_mf_round_trip_polydata(tmp_path: Path) -> None:
    src = _src_polydata()
    path = tmp_path / 'sphere.3mf'
    pyvista_cad.write_three_mf(src, path, units='mm')
    mb = pyvista_cad.read_three_mf(path)
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 1
    assert mb[0].n_cells == src.n_cells
    assert mb[0].n_points == src.n_points
    assert str(mb.field_data['cad.units'][0]) == 'mm'


def test_three_mf_multiblock_preserves_names(tmp_path: Path) -> None:
    a = pv.Cube()
    b = pv.Cone()
    mb_in = pv.MultiBlock({'box': a, 'cone': b})
    path = tmp_path / 'parts.3mf'
    pyvista_cad.write_three_mf(mb_in, path)
    mb_out = pyvista_cad.read_three_mf(path)
    assert mb_out.n_blocks == 2
    assert sorted(mb_out.keys()) == ['box', 'cone']


def test_three_mf_pv_read_dispatch(tmp_path: Path) -> None:
    path = tmp_path / 'cube.3mf'
    pyvista_cad.write_three_mf(pv.Cube(), path)
    mb = pv.read(path)
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 1


def test_read_three_mf_not_a_zip_raises(tmp_path: Path) -> None:
    """A 3MF file that is not a valid ZIP archive raises ``CadReadError``."""
    from pyvista_cad import CadReadError  # noqa: PLC0415

    bad = tmp_path / 'not_a_zip.3mf'
    bad.write_bytes(b'this is plain text, not a zip\n' * 16)
    with pytest.raises(CadReadError):
        pyvista_cad.read_three_mf(bad)


def test_read_multimaterial_3mf_fidelity(multimaterial_3mf_path: Path) -> None:
    """The two-object lib3mf fixture reads as two named, colored cubes.

    Geometry, labels and per-block RGBA colors are pinned from
    ``tests/data/README.md``. ``cad.units`` is surfaced on the MultiBlock
    root (the reader does not re-stamp it per block).
    """
    mb = pyvista_cad.read_three_mf(multimaterial_3mf_path)
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 2
    assert list(mb.keys()) == ['CubeA', 'CubeB']
    assert str(mb.field_data['cad.units'][0]) == 'mm'
    assert str(mb.field_data['cad.backend'][0]) == 'lib3mf'
    assert str(mb.field_data['cad.source_format'][0]) == '3mf'

    cb = mb.combine().bounds
    assert (cb.x_min, cb.x_max) == pytest.approx((0.0, 30.0), abs=1e-6)
    assert (cb.y_min, cb.y_max) == pytest.approx((0.0, 10.0), abs=1e-6)
    assert (cb.z_min, cb.z_max) == pytest.approx((0.0, 10.0), abs=1e-6)

    a = mb['CubeA']
    assert a.n_cells == 12
    assert str(a.field_data['cad.label'][0]) == 'CubeA'
    np.testing.assert_allclose(
        np.asarray(a.field_data['cad.color']), [1.0, 0.0, 0.0, 1.0], atol=1.0 / 255.0
    )

    b = mb['CubeB']
    assert b.n_cells == 12
    assert str(b.field_data['cad.label'][0]) == 'CubeB'
    np.testing.assert_allclose(
        np.asarray(b.field_data['cad.color']), [0.0, 0.0, 1.0, 1.0], atol=1.0 / 255.0
    )


def test_multimaterial_3mf_round_trip_preserves_label_and_units(
    multimaterial_3mf_path: Path, tmp_path: Path
) -> None:
    """Write the fixture back out and re-read: per-block labels and the
    document units survive the writer round-trip."""
    mb = pyvista_cad.read_three_mf(multimaterial_3mf_path)
    out = tmp_path / 'rt.3mf'
    pyvista_cad.write_three_mf(mb, out)
    rt = pyvista_cad.read_three_mf(out)
    assert rt.n_blocks == 2
    assert list(rt.keys()) == ['CubeA', 'CubeB']
    assert str(rt.field_data['cad.units'][0]) == 'mm'
    assert str(rt['CubeA'].field_data['cad.label'][0]) == 'CubeA'
    assert str(rt['CubeB'].field_data['cad.label'][0]) == 'CubeB'


def test_multimaterial_3mf_round_trip_preserves_color(
    multimaterial_3mf_path: Path, tmp_path: Path
) -> None:
    """Per-block color, units, and label survive the writer round-trip."""
    mb = pyvista_cad.read_three_mf(multimaterial_3mf_path)
    out = tmp_path / 'rt.3mf'
    pyvista_cad.write_three_mf(mb, out)
    rt = pyvista_cad.read_three_mf(out)
    assert rt.n_blocks == 2
    assert list(rt.keys()) == ['CubeA', 'CubeB']

    a = rt['CubeA']
    assert str(a.field_data['cad.label'][0]) == 'CubeA'
    assert str(a.field_data['cad.units'][0]) == 'mm'
    np.testing.assert_allclose(
        np.asarray(a.field_data['cad.color']),
        [1.0, 0.0, 0.0, 1.0],
        atol=1.0 / 255.0,
    )

    b = rt['CubeB']
    assert str(b.field_data['cad.label'][0]) == 'CubeB'
    assert str(b.field_data['cad.units'][0]) == 'mm'
    np.testing.assert_allclose(
        np.asarray(b.field_data['cad.color']),
        [0.0, 0.0, 1.0, 1.0],
        atol=1.0 / 255.0,
    )


def test_three_mf_accessor_to_3mf(tmp_path: Path) -> None:
    path = tmp_path / 'sphere.3mf'
    src = _src_polydata()
    src.cad.to_3mf(path)
    assert path.exists()
    mb = pyvista_cad.read_three_mf(path)
    assert mb.n_blocks == 1
