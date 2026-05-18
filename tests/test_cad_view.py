"""Tests for the CAD-friendly visualization layer.

Behavioural assertions on the data pipeline (edge extraction, per-face
identity, analytic normals, curve classification, the flatten round
trip, and the no-B-rep degrade path), plus a smoke test of the plotter
component and the ``.cad`` accessor's ``plot()``.

OCP-dependent tests skip cleanly when the ``[step]`` extra is absent.
"""

import numpy as np
import pytest
import pyvista as pv

from pyvista_cad import (
    flatten_to_cad_polydata,
    get_cad_edges,
    topods_to_edges,
    topods_to_multiblock,
)

OCP = pytest.importorskip('OCP', exc_type=ImportError)


@pytest.fixture
def block_with_hole():
    """A box with a through cylindrical hole.

    Asymmetric, and mixes planar faces (lines) with a cylindrical wall
    (circles) so edge-kind classification and analytic normals are both
    exercised.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    box = BRepPrimAPI_MakeBox(20.0, 12.0, 8.0).Shape()
    axis = gp_Ax2(gp_Pnt(7.0, 6.0, -1.0), gp_Dir(0.0, 0.0, 1.0))
    drill = BRepPrimAPI_MakeCylinder(axis, 3.0, 10.0).Shape()
    return BRepAlgoAPI_Cut(box, drill).Shape()


def test_topods_to_edges_returns_polylines(block_with_hole):
    edges = topods_to_edges(block_with_hole, linear_deflection=0.3)
    assert edges.n_cells > 0
    assert edges.n_points >= 2 * edges.n_cells - edges.n_cells  # >=2 pts/edge
    # Every cell is a polyline (VTK line / polyline), never a vertex.
    assert edges.lines.size > 0
    assert 'cad.edge_id' in edges.cell_data
    assert len(np.unique(edges.cell_data['cad.edge_id'])) == edges.n_cells


def test_edge_kind_classifies_lines_and_circles(block_with_hole):
    edges = topods_to_edges(block_with_hole, linear_deflection=0.3)
    kinds = set(np.unique(edges.cell_data['cad.edge_kind']).tolist())
    # 0 == line (box edges), 1 == circle (drilled wall rims).
    assert 0 in kinds
    assert 1 in kinds
    legend = edges.field_data['cad.edge_kind_legend'][0]
    assert '0=line' in legend
    assert '1=circle' in legend


def test_multiblock_preserves_face_identity(block_with_hole):
    mb = topods_to_multiblock(block_with_hole, linear_deflection=0.3)
    assert set(mb.keys()) == {'faces', 'edges'}
    faces = mb['faces']
    # Box (6) minus a face split by the hole + cylindrical wall: > 6.
    assert len(faces) >= 7
    for i, block in enumerate(faces):
        ids = block.cell_data['cad.face_id']
        assert np.all(ids == i), 'each block must carry a single face id'


def test_analytic_normals_are_unit_and_radial():
    """A coarse cylinder must still shade round: normals from the surface,
    not the facets."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

    cyl = BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape()
    mb = topods_to_multiblock(cyl, linear_deflection=1.0)  # very coarse
    wall = max(mb['faces'], key=lambda b: b.n_cells)
    assert 'Normals' in wall.point_data
    n = wall.point_data['Normals']
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    # Cylinder axis is +Z: every wall normal must be purely radial.
    assert np.allclose(n[:, 2], 0.0, atol=1e-6)


def test_flatten_round_trip(block_with_hole):
    mb = topods_to_multiblock(block_with_hole, linear_deflection=0.3)
    poly = flatten_to_cad_polydata(mb)
    assert isinstance(poly, pv.PolyData)
    assert 'cad.face_id' in poly.cell_data
    n_faces = len(mb['faces'])
    assert len(np.unique(poly.cell_data['cad.face_id'])) == n_faces
    stashed = get_cad_edges(poly)
    assert stashed is not None
    assert stashed.n_cells == mb['edges'].n_cells


def test_degrade_path_uses_feature_edges():
    """A plain mesh with no B-rep origin still plots: edges fall back to
    crease feature extraction."""
    from pyvista_cad._cad_view import _resolve_cad

    sphere = pv.Sphere()
    faces, edges = _resolve_cad(
        sphere, linear_deflection=0.1, angular_deflection=0.5, feature_angle=30.0
    )
    assert faces is sphere
    assert edges is not None  # feature-edge PolyData (may be empty for a sphere)


def test_plotter_component_returns_actors(block_with_hole):
    pl = pv.Plotter(off_screen=True)
    result = pl.cad.add(block_with_hole, linear_deflection=0.3)
    assert set(result) == {'faces', 'edges'}
    assert result['faces'] is not None
    assert result['edges'] is not None
    pl.close()


def test_accessor_plot_from_cached_topods(block_with_hole, tmp_path):
    """``mesh.cad.plot()`` resolves through the cached originating shape."""
    from pyvista_cad._bridges.topods import from_topods

    mesh = from_topods(block_with_hole, linear_deflection=0.3)
    out = tmp_path / 'shot.png'
    mesh.cad.plot(off_screen=True, screenshot=str(out))
    assert out.is_file()


def test_plotter_component_registered():
    names = {getattr(c, 'name', None) for c in pv.registered_plotter_components()}
    assert 'cad' in names
