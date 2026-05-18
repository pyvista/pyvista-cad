"""Tests for the cadquery bridge."""

import pytest
import pyvista as pv

import pyvista_cad

cq = pytest.importorskip('cadquery', exc_type=ImportError)


def test_from_cadquery_box() -> None:
    wp = cq.Workplane('XY').box(2, 4, 6)
    mesh = pyvista_cad.from_cadquery(wp)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells > 0
    assert str(mesh.field_data['cad.source_format'][0]) == 'cadquery'
    assert mesh.bounds.x_min == pytest.approx(-1.0, abs=0.05)
    assert mesh.bounds.x_max == pytest.approx(1.0, abs=0.05)


def test_to_cadquery_returns_workplane() -> None:
    src = pv.Sphere(theta_resolution=8, phi_resolution=8)
    wp = pyvista_cad.to_cadquery(src)
    assert isinstance(wp, cq.Workplane)
