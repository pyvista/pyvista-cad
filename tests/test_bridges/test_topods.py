"""Tests for the raw OCP / TopoDS bridge."""

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('OCP', exc_type=ImportError)
b3d = pytest.importorskip('build123d', exc_type=ImportError)


def test_from_topods_box() -> None:
    box = b3d.Box(4, 4, 4)
    mesh = pyvista_cad.from_topods(box.wrapped)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells == 12
    assert str(mesh.field_data['cad.source_format'][0]) == 'OCP'


def test_to_topods_sews_into_solid() -> None:
    """A unit-radius sphere sews into a TopoDS shape whose volume matches
    ``4/3 pi r^3`` within 1% (allowing for tessellation discretisation)."""
    from OCP.BRepGProp import BRepGProp  # noqa: PLC0415
    from OCP.GProp import GProp_GProps  # noqa: PLC0415

    radius = 1.0
    # Use a fine tessellation so the discrete volume converges on the
    # analytical sphere volume within 1%.
    mesh = pv.Sphere(theta_resolution=60, phi_resolution=60, radius=radius)
    ts = pyvista_cad.to_topods(mesh)
    assert type(ts).__name__ in {'TopoDS_Solid', 'TopoDS_Shell', 'TopoDS_Shape'}

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(ts, props)
    expected = (4.0 / 3.0) * np.pi * radius**3
    assert props.Mass() == pytest.approx(expected, rel=0.01)


def test_round_trip_polydata_through_topods() -> None:
    src = pyvista_cad.from_build123d(b3d.Box(2, 4, 6))
    ts = pyvista_cad.to_topods(src)
    rt = pyvista_cad.from_topods(ts)
    assert rt.n_cells == src.n_cells
    assert rt.bounds.x_min == pytest.approx(src.bounds.x_min, abs=0.01)
    assert rt.bounds.x_max == pytest.approx(src.bounds.x_max, abs=0.01)


def test_tessellate_round_trip_via_accessor() -> None:
    src = pyvista_cad.from_build123d(b3d.Box(1, 1, 1))
    out = src.cad.tessellate(linear_deflection=0.1, angular_deflection=0.5)
    assert isinstance(out, pv.PolyData)
    assert out.n_cells > 0
