"""Tests for the DXF writer and round-trip."""

from pathlib import Path

import numpy as np
import pyvista as pv

import pyvista_cad


def test_write_dxf_round_trip_preserves_layers(tmp_path: Path) -> None:
    src = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    out = tmp_path / 'out.dxf'
    pyvista_cad.write_dxf(src, out)
    assert out.exists()
    rt = pyvista_cad.read_dxf(out)
    src_layers = sorted({str(x) for x in src.cell_data['Layer']})
    rt_layers = sorted({str(x) for x in rt.cell_data['Layer']})
    assert src_layers == rt_layers


def test_write_dxf_round_trip_preserves_bounds(tmp_path: Path) -> None:
    src = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    out = tmp_path / 'out.dxf'
    pyvista_cad.write_dxf(src, out)
    rt = pyvista_cad.read_dxf(out)
    np.testing.assert_allclose(rt.bounds, src.bounds, atol=1e-6)


def test_write_dxf_default_units_mm(tmp_path: Path) -> None:
    mesh = pv.PolyData([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], lines=[2, 0, 1])
    mesh.cad.units = 'mm'
    out = tmp_path / 'line.dxf'
    pyvista_cad.write_dxf(mesh, out)
    rt = pyvista_cad.read_dxf(out)
    assert str(rt.field_data['cad.units'][0]) == 'mm'


def test_accessor_to_dxf(tmp_path: Path) -> None:
    src = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    out = tmp_path / 'via-accessor.dxf'
    src.cad.to_dxf(out)
    assert out.exists()
    rt = pyvista_cad.read_dxf(out)
    assert rt.n_cells == src.n_cells
