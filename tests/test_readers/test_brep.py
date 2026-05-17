"""Tests for the BREP reader and writer."""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('OCP')
b3d = pytest.importorskip('build123d')


def test_brep_round_trip_polydata(tmp_path: Path) -> None:
    src = pyvista_cad.from_build123d(b3d.Box(2, 4, 6))
    path = tmp_path / 'box.brep'
    pyvista_cad.write_brep(src, path)
    assert path.exists()
    poly = pyvista_cad.read_brep(path)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == src.n_cells
    assert str(poly.field_data['cad.source_format'][0]) == 'brep'
    assert str(poly.field_data['cad.backend'][0]) == 'ocp'
    assert bool(poly.field_data['cad.has_brep_origin'][0])
    assert poly.bounds.x_min == pytest.approx(src.bounds.x_min, abs=0.01)


def test_brep_pv_read_dispatch(tmp_path: Path) -> None:
    src = pyvista_cad.from_build123d(b3d.Box(1, 1, 1))
    path = tmp_path / 'cube.brep'
    pyvista_cad.write_brep(src, path)
    poly = pv.read(path)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == src.n_cells


def test_read_brep_empty_file_raises(tmp_path: Path) -> None:
    """An empty .brep file raises ``CadReadError``."""
    from pyvista_cad import CadReadError  # noqa: PLC0415

    bad = tmp_path / 'empty.brep'
    bad.write_bytes(b'')
    with pytest.raises(CadReadError):
        pyvista_cad.read_brep(bad)


def test_brep_accessor_to_brep(tmp_path: Path) -> None:
    src = pyvista_cad.from_build123d(b3d.Box(1, 1, 1))
    path = tmp_path / 'cube.brep'
    src.cad.to_brep(path)
    assert path.exists()


def test_read_external_brep_fixture(external_brep_path: Path) -> None:
    """A BREP produced by ``BRepTools.Write_s`` (not our writer) reads to the
    exact, origin-centred box recorded in ``tests/data/README.md``."""
    poly = pyvista_cad.read_brep(external_brep_path)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == 12
    assert poly.n_points == 24
    b = poly.bounds
    assert (b.x_min, b.x_max) == pytest.approx((-12.5, 12.5), abs=1e-6)
    assert (b.y_min, b.y_max) == pytest.approx((-7.5, 7.5), abs=1e-6)
    assert (b.z_min, b.z_max) == pytest.approx((-4.0, 4.0), abs=1e-6)
    assert str(poly.field_data['cad.source_format'][0]) == 'brep'
    assert str(poly.field_data['cad.backend'][0]) == 'ocp'
    assert bool(poly.field_data['cad.has_brep_origin'][0])


def test_read_external_brep_via_pv_read(external_brep_path: Path) -> None:
    """``pv.read`` dispatch yields the same externally-produced geometry."""
    poly = pv.read(external_brep_path)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == 12
    assert poly.n_points == 24
    assert str(poly.field_data['cad.source_format'][0]) == 'brep'


def test_read_brep_malformed_table_raises(tmp_path: Path) -> None:
    """A BREP whose header declares a ``TShapes`` count with no matching
    entries makes OCCT's ``BRepTools.Read_s`` throw mid-parse; ``read_brep``
    must surface that as ``CadReadError`` (the ``except`` arm), not leak the
    raw OCP ``Standard_OutOfRange``."""
    from pyvista_cad import CadReadError  # noqa: PLC0415

    header = 'DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n'
    body = (
        header + 'Locations 0\n\nCurve2ds 0\n\nCurves 0\n\nPolygon3D 0\n\n'
        'PolygonOnTriangulations 0\n\nSurfaces 0\n\nTriangulations 0\n\n'
        'TShapes 9223372036854775807\n'
    )
    bad = tmp_path / 'malformed.brep'
    bad.write_text(body)
    with pytest.raises(CadReadError, match='failed to read BREP'):
        pyvista_cad.read_brep(bad)
