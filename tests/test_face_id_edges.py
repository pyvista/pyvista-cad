"""Topological edges recovered from a per-triangle ``cad.face_id``."""

import numpy as np
import pyvista as pv

from pyvista_cad import face_id_edges
from pyvista_cad._cad_view import _resolve_cad


def _hemispheres() -> pv.PolyData:
    """A sphere split at the equator into two faces that each own their vertices."""
    sphere = pv.Sphere(phi_resolution=31)
    upper = sphere.cell_centers().points[:, 2] > 0
    points, faces, normals, face_id = [], [], [], []
    offset = 0
    for face, mask in enumerate([~upper, upper]):
        ids, local = np.unique(sphere.regular_faces[mask], return_inverse=True)
        points.append(sphere.points[ids])
        normals.append(sphere.point_data['Normals'][ids])
        faces.append(local.reshape(-1, 3) + offset)
        face_id.append(np.full(mask.sum(), face))
        offset += len(ids)
    mesh = pv.PolyData.from_regular_faces(np.vstack(points), np.vstack(faces))
    mesh.point_data['Normals'] = np.vstack(normals)
    mesh.point_data.active_normals_name = 'Normals'
    mesh.cell_data['cad.face_id'] = np.concatenate(face_id)
    return mesh


def test_edges_follow_face_boundaries_not_triangles() -> None:
    cube = pv.Cube().triangulate()
    cube.cell_data['cad.face_id'] = np.repeat(np.arange(6), 2)
    edges = face_id_edges(cube)
    assert edges.n_cells == 12
    assert np.unique(edges.cell_data['cad.edge_id']).size == 12


def test_tangent_boundary_is_dropped() -> None:
    mesh = _hemispheres()
    assert face_id_edges(mesh).n_cells == 0
    equator = face_id_edges(mesh, tangent_tol_deg=0)
    assert equator.n_cells > 0
    np.testing.assert_allclose(equator.points[:, 2], 0, atol=1e-6)


def test_shared_vertex_normals_do_not_hide_sharp_edges() -> None:
    cube = pv.Cube().triangulate()
    assert cube.point_data.active_normals is not None
    cube.cell_data['cad.face_id'] = np.repeat(np.arange(6), 2)
    assert face_id_edges(cube).n_cells == 12


def test_single_closed_face_has_no_edges() -> None:
    sphere = pv.Sphere()
    sphere.cell_data['cad.face_id'] = np.zeros(sphere.n_cells, dtype=int)
    assert face_id_edges(sphere).n_cells == 0


def test_cad_view_uses_face_ids_over_feature_edges() -> None:
    cube = pv.Cube().triangulate()
    _, fallback = _resolve_cad(
        cube, linear_deflection=0.1, angular_deflection=0.5, feature_angle=30.0
    )
    assert 'cad.edge_id' not in fallback.cell_data
    cube.cell_data['cad.face_id'] = np.repeat(np.arange(6), 2)
    _, edges = _resolve_cad(
        cube, linear_deflection=0.1, angular_deflection=0.5, feature_angle=30.0
    )
    assert np.unique(edges.cell_data['cad.edge_id']).size == 12
