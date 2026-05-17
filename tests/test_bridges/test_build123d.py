"""Tests for the build123d bridge."""

import pytest
import pyvista as pv

import pyvista_cad

b3d = pytest.importorskip('build123d')


def test_from_build123d_box() -> None:
    mesh = pyvista_cad.from_build123d(b3d.Box(10, 10, 10))
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells == 12
    assert mesh.bounds.x_min == pytest.approx(-5.0, abs=0.01)
    assert mesh.bounds.x_max == pytest.approx(5.0, abs=0.01)
    assert str(mesh.field_data['cad.source_format'][0]) == 'build123d'


def test_from_build123d_module_level() -> None:
    mesh = pyvista_cad.from_build123d(b3d.Box(2, 2, 2))
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells == 12


def test_to_build123d_returns_compound() -> None:
    mesh = pyvista_cad.from_build123d(b3d.Box(1, 1, 1))
    compound = pyvista_cad.to_build123d(mesh)
    assert isinstance(compound, b3d.Compound)


def test_from_build123d_compound_multiblock() -> None:
    box = b3d.Box(1, 1, 1)
    cyl = b3d.Cylinder(0.5, 1)
    box.label = 'box'
    cyl.label = 'cyl'
    compound = b3d.Compound(children=[box, cyl])
    out = pyvista_cad.from_build123d(compound)
    assert isinstance(out, pv.MultiBlock)
    assert out.n_blocks == 2
    labels = sorted(out.keys())
    assert labels == ['box', 'cyl']
