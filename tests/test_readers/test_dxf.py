"""Tests for the DXF reader."""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad import CadReadError


def test_read_dxf_returns_polydata() -> None:
    poly = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0
    assert poly.n_points > 0


def test_read_dxf_attaches_metadata() -> None:
    poly = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    assert str(poly.field_data['cad.source_format'][0]) == 'dxf'
    assert str(poly.field_data['cad.units'][0]) == 'mm'


def test_read_dxf_emits_layer_arrays() -> None:
    poly = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    assert 'Layer' in poly.cell_data
    assert 'cad.layer' in poly.cell_data
    layers = sorted({str(x) for x in poly.cell_data['Layer']})
    assert layers == ['HOLES', 'OUTLINE']


def test_pv_read_routes_to_pyvista_cad() -> None:
    poly = pv.read(pyvista_cad.examples.dxf_plate)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0


def test_read_dxf_truncated_returns_empty_or_raises(tmp_path: Path) -> None:
    """A DXF truncated mid-stream either raises ``CadReadError`` or yields
    an empty PolyData via ``ezdxf.recover``.

    ezdxf's recover path is permissive: it can parse a SECTION header with no
    body as a structurally valid (but empty) document. Either outcome is an
    acceptable contract — what's not acceptable is silently returning a mesh
    with cells that don't exist in the source.
    """
    bad = tmp_path / 'truncated.dxf'
    # Just the SECTION header — no ENDSEC, no EOF.
    bad.write_bytes(b'0\nSECTION\n2\nHEADER\n')
    try:
        poly = pyvista_cad.read_dxf(bad)
    except CadReadError:
        return
    assert poly.n_cells == 0
    assert poly.n_points == 0


def test_read_dxf_garbage_raises(tmp_path: Path) -> None:
    """A random-bytes file with no DXF structure raises ``CadReadError``."""
    bad = tmp_path / 'garbage.dxf'
    bad.write_bytes(b'\x00\x01\x02not a dxf at all\xff\xfe\xfd' * 32)
    with pytest.raises(CadReadError):
        pyvista_cad.read_dxf(bad)


def test_read_dxf_bounds_are_sane() -> None:
    poly = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    b = poly.bounds
    assert b.x_min >= -1.0
    assert b.x_max <= 61.0
    assert b.y_min >= -1.0
    assert b.y_max <= 41.0


def _assert_dxf_fixture_geometry(poly: pv.PolyData) -> None:
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == 5
    assert poly.n_points == 81
    b = poly.bounds
    assert (b.x_min, b.x_max) == pytest.approx((0.0, 100.0), abs=1e-6)
    assert (b.y_min, b.y_max) == pytest.approx((0.0, 50.0), abs=1e-6)
    assert (b.z_min, b.z_max) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert str(poly.field_data['cad.backend'][0]) == 'ezdxf'
    assert str(poly.field_data['cad.source_format'][0]) == 'dxf'
    assert str(poly.field_data['cad.reader'][0]) == 'pyvista_cad.read_dxf'


def test_read_dxf_r2018_fixture(dxf_r2018_path: Path) -> None:
    """R2018 fixture: exact geometry, ``mm`` units, version recorded."""
    poly = pyvista_cad.read_dxf(dxf_r2018_path)
    _assert_dxf_fixture_geometry(poly)
    assert str(poly.field_data['cad.units'][0]) == 'mm'
    assert str(poly.field_data['cad.source_format_version'][0]) == 'R2018'


def test_read_dxf_r12_fixture(dxf_r12_path: Path) -> None:
    """R12 fixture: same geometry; R12 cannot persist ``$INSUNITS`` so units
    are ``unitless`` while the format version is still surfaced."""
    poly = pyvista_cad.read_dxf(dxf_r12_path)
    _assert_dxf_fixture_geometry(poly)
    assert str(poly.field_data['cad.units'][0]) == 'unitless'
    assert str(poly.field_data['cad.source_format_version'][0]) == 'R12'


def test_read_dxf_r12_and_r2018_geometry_match(dxf_r12_path: Path, dxf_r2018_path: Path) -> None:
    """The two DXF versions encode the same drawing; only metadata differs."""
    r12 = pyvista_cad.read_dxf(dxf_r12_path)
    r2018 = pyvista_cad.read_dxf(dxf_r2018_path)
    assert r12.n_cells == r2018.n_cells
    assert r12.n_points == r2018.n_points
    assert r12.bounds == pytest.approx(tuple(r2018.bounds), abs=1e-6)
