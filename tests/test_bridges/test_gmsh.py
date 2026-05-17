"""Tests for the gmsh bridge."""

import pytest
import pyvista as pv

import pyvista_cad

gmsh = pytest.importorskip('gmsh')


@pytest.fixture
def gmsh_session() -> None:
    gmsh.initialize()
    yield
    gmsh.finalize()


def test_to_gmsh_from_polydata(gmsh_session: None) -> None:
    src = pv.Sphere(theta_resolution=8, phi_resolution=8)
    pyvista_cad.to_gmsh(src, model_name='sphere_test')
    grid = pyvista_cad.from_gmsh()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells > 0
    assert grid.n_points > 0
