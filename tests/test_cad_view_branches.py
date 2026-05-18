"""Branch and fallback coverage for the CAD-friendly view pipeline.

Exercises the free-edge sampler, the analytic-normal fallbacks, the
no-B-rep feature-edge degrade path (and its ``CadWarning``), assembly
``MultiBlock`` resolution, the stashed-edges branch, ``as_cad_multiblock``,
the scalar/RGB/edge-colour rendering branches of ``add_cad``, and
``plot_cad`` keyword splitting. Real OCCT geometry throughout; no mocks.
"""

import warnings

import numpy as np
import pytest
import pyvista as pv

from pyvista_cad import (
    CadWarning,
    TessellationError,
    as_cad_multiblock,
    flatten_to_cad_polydata,
    get_cad_edges,
    topods_to_edges,
    topods_to_multiblock,
)
from pyvista_cad._cad_view import (
    _ANALYTIC_NORMALS_FLAG,
    _CURVE_KINDS,
    _curve_kind,
    _ensure_mesh,
    _face_unit_normal_at_edge,
    _feature_edge_fallback,
    _resolve_cad,
    _shape_has_faces,
    _tangent_internal_edges,
    add_cad,
    plot_cad,
    register_cad_plotter_component,
)

pytest.importorskip('OCP', exc_type=ImportError)

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
)
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Builder, TopoDS_Compound, TopoDS_Edge

from pyvista_cad._bridges.topods import from_topods, to_topods

pv.OFF_SCREEN = True


@pytest.fixture
def block_with_hole():
    """Box with a through hole: planar faces (lines) plus a cylindrical
    wall (circles), asymmetric, curved."""
    box = BRepPrimAPI_MakeBox(20.0, 12.0, 8.0).Shape()
    axis = gp_Ax2(gp_Pnt(7.0, 6.0, -1.0), gp_Dir(0.0, 0.0, 1.0))
    drill = BRepPrimAPI_MakeCylinder(axis, 3.0, 10.0).Shape()
    return BRepAlgoAPI_Cut(box, drill).Shape()


@pytest.fixture
def filleted_box():
    """A box with every edge filleted: curved fillet surfaces meeting
    planar faces and each other."""
    box = BRepPrimAPI_MakeBox(20.0, 12.0, 8.0).Shape()
    mk = BRepFilletAPI_MakeFillet(box)
    ex = TopExp_Explorer(box, TopAbs_EDGE)
    while ex.More():
        mk.Add(2.0, TopoDS.Edge_s(ex.Current()))
        ex.Next()
    return mk.Shape()


def _free_edge_compound():
    """A compound holding one face-less straight edge (wireframe)."""
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(5, 2, 1)).Edge()
    builder = TopoDS_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    builder.Add(comp, edge)
    return comp


# --- free-edge / wireframe sampler ---------------------------------------


def test_free_edge_sampled_as_polyline():
    edges = topods_to_edges(_free_edge_compound())
    assert edges.n_cells == 1
    assert edges.n_points >= 2
    # Straight line: classified as kind 0.
    assert edges.cell_data['cad.edge_kind'][0] == 0


def test_free_edge_compound_with_faces_and_wire():
    """A box plus a detached wireframe edge: both the face-shared edges
    and the free edge are emitted."""
    box = BRepPrimAPI_MakeBox(4.0, 4.0, 4.0).Shape()
    wire = BRepBuilderAPI_MakeEdge(gp_Pnt(10, 0, 0), gp_Pnt(10, 5, 5)).Edge()
    builder = TopoDS_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    builder.Add(comp, box)
    builder.Add(comp, wire)
    edges = topods_to_edges(comp, tangent_tol_deg=0.0)
    # 12 box edges + 1 free wire edge.
    assert edges.n_cells == 13


def test_curveless_free_edge_degrades_with_cad_warning():
    """A face-less edge with no 3D curve (the IGES/wireframe construction
    case) cannot be sampled. OCCT raises ``Standard_NullObject`` (an
    ``Exception`` subclass, not ``RuntimeError``); the helper catches it,
    drops the edge, and the aggregated ``CadWarning`` fires instead of the
    whole extraction crashing."""
    edge = TopoDS_Edge()
    BRep_Builder().MakeEdge(edge)  # edge with no 3D curve
    builder = TopoDS_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    builder.Add(comp, edge)
    with pytest.warns(CadWarning, match=r'face-less edge\(s\) could not be sampled'):
        edges = topods_to_edges(comp)
    assert isinstance(edges, pv.PolyData)
    assert edges.n_cells == 0


def test_topods_to_edges_empty_shape_returns_empty():
    """A compound with no faces and no edges produces an empty edge
    PolyData (the no-points early return)."""
    builder = TopoDS_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    edges = topods_to_edges(comp)
    assert isinstance(edges, pv.PolyData)
    assert edges.n_cells == 0


# --- tangent-edge suppression --------------------------------------------


def test_tangent_tol_zero_keeps_every_edge(block_with_hole):
    kept_all = topods_to_edges(block_with_hole, tangent_tol_deg=0.0)
    suppressed = topods_to_edges(block_with_hole, tangent_tol_deg=8.0)
    assert kept_all.n_cells >= suppressed.n_cells


def test_topods_to_edges_default_tol_dedups_shared_edges(block_with_hole):
    """With the default positive tangent tolerance the per-face loop runs
    the tangent-internal-edge map and the shared-edge dedup path: every
    edge is emitted exactly once with a unique id."""
    edges = topods_to_edges(block_with_hole)  # default tangent_tol_deg=8.0
    assert edges.n_cells > 0
    ids = edges.cell_data['cad.edge_id']
    assert len(np.unique(ids)) == edges.n_cells


def test_multiblock_and_edges_consistent_default_tol(block_with_hole):
    mb = topods_to_multiblock(block_with_hole)
    assert mb['edges'].n_cells == topods_to_edges(block_with_hole).n_cells


def test_filleted_box_tangent_suppression_runs(filleted_box):
    """Tangent-edge suppression on a filleted solid removes the G1-tangent
    fillet/face junctions while keeping the real feature edges.

    Every fillet meets its neighbouring planar face tangentially, so with
    a positive tolerance those junction edges are suppressed and only the
    handful of true sharp edges survive. This exercises the analytic
    surface-normal path on curved faces (planar via the adaptor plane,
    curved via the du x dv cross product at the pcurve midpoint)."""
    kept_all = topods_to_edges(filleted_box, linear_deflection=0.3, tangent_tol_deg=0.0)
    suppressed = topods_to_edges(filleted_box, linear_deflection=0.3, tangent_tol_deg=15.0)
    assert suppressed.n_cells > 0
    # The tangent junctions are genuinely removed, not a no-op: a fully
    # filleted box keeps only its few non-tangent feature edges.
    assert suppressed.n_cells < kept_all.n_cells
    assert suppressed.n_cells <= kept_all.n_cells // 2
    assert suppressed.lines.size > 0


def _first_face_and_edge(shape):
    fx = TopExp_Explorer(shape, TopAbs_FACE)
    face = TopoDS.Face_s(fx.Current())
    ex = TopExp_Explorer(face, TopAbs_EDGE)
    return face, TopoDS.Edge_s(ex.Current())


def test_face_unit_normal_planar_face_is_real_unit_vector():
    """A box face is planar: the analytic normal comes from the adaptor
    plane axis (no Geom downcast) and is a real unit vector, not None."""
    box = BRepPrimAPI_MakeBox(10.0, 6.0, 4.0).Shape()
    _ensure_mesh(box, 0.3, 0.5)
    face, edge = _first_face_and_edge(box)
    n = _face_unit_normal_at_edge(face, edge)
    assert n is not None
    assert np.isclose(np.linalg.norm(n), 1.0)


def test_face_unit_normal_curved_face_is_real_unit_vector():
    """A cylinder wall is a curved face: the analytic normal comes from
    the du x dv cross product at the pcurve midpoint and is unit length,
    proving the curved-surface path is live (not the dead None path)."""
    cyl = BRepPrimAPI_MakeCylinder(4.0, 12.0).Shape()
    _ensure_mesh(cyl, 0.3, 0.5)
    seen = []
    fx = TopExp_Explorer(cyl, TopAbs_FACE)
    while fx.More():
        face = TopoDS.Face_s(fx.Current())
        ex = TopExp_Explorer(face, TopAbs_EDGE)
        edge = TopoDS.Edge_s(ex.Current())
        n = _face_unit_normal_at_edge(face, edge)
        if n is not None:
            seen.append(n)
            assert np.isclose(np.linalg.norm(n), 1.0)
        fx.Next()
    assert seen, 'no analytic normal recovered for any cylinder face'


def test_face_unit_normal_cone_apex_is_none_walls_are_real():
    """A cone exercises both branches of the curved-face path: the
    conical wall yields a real unit normal (du x dv at the pcurve
    midpoint), while the apex seam is a surface singularity where the
    cross product vanishes and the helper returns ``None``."""
    cone = BRepPrimAPI_MakeCone(5.0, 0.0, 10.0).Shape()
    _ensure_mesh(cone, 0.5, 0.5)
    results = []
    fx = TopExp_Explorer(cone, TopAbs_FACE)
    while fx.More():
        face = TopoDS.Face_s(fx.Current())
        ex = TopExp_Explorer(face, TopAbs_EDGE)
        while ex.More():
            n = _face_unit_normal_at_edge(face, TopoDS.Edge_s(ex.Current()))
            results.append(n)
            if n is not None:
                assert np.isclose(np.linalg.norm(n), 1.0)
            ex.Next()
        fx.Next()
    assert any(n is None for n in results), 'apex singularity never hit'
    assert any(n is not None for n in results), 'no real cone normal'


def test_tangent_suppression_cone_keeps_base_edge():
    """The cone's wall-to-base rim is a sharp (non-tangent) edge: the
    curved-normal path resolves both adjacent faces and the edge survives
    suppression, so a positive tolerance does not empty the overlay."""
    cone = BRepPrimAPI_MakeCone(5.0, 0.0, 10.0).Shape()
    kept = topods_to_edges(cone, linear_deflection=0.3, tangent_tol_deg=0.0)
    suppressed = topods_to_edges(cone, linear_deflection=0.3, tangent_tol_deg=15.0)
    assert suppressed.n_cells > 0
    assert suppressed.n_cells <= kept.n_cells


def test_tangent_internal_edges_filleted_box_suppresses_junctions(filleted_box):
    """The tangent-edge map is non-empty for a fully filleted box: every
    fillet/face junction is G1-tangent and gets suppressed. A regression
    guard for the dead-feature bug (it was empty for all curved parts)."""
    skip = _tangent_internal_edges(filleted_box, 15.0)
    assert skip.Extent() > 0


# --- analytic normals on curved surfaces ---------------------------------


def test_sphere_normals_point_radially_outward():
    sphere = BRepPrimAPI_MakeSphere(5.0).Shape()
    mb = topods_to_multiblock(sphere, linear_deflection=0.5)
    face = mb['faces'][0]
    assert 'Normals' in face.point_data
    n = face.point_data['Normals']
    pts = face.points
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert np.median(np.einsum('ij,ij->i', n, radial)) > 0.9


def test_torus_normals_are_unit():
    """A torus has a doubly-curved surface; analytic normals must still
    be unit length on the coarse tessellation."""
    torus = BRepPrimAPI_MakeTorus(10.0, 3.0).Shape()
    mb = topods_to_multiblock(torus, linear_deflection=0.5)
    face = max(mb['faces'], key=lambda b: b.n_points)
    n = face.point_data['Normals']
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)


def test_filleted_box_multiblock_normals_unit(filleted_box):
    mb = topods_to_multiblock(filleted_box, linear_deflection=0.3)
    for block in mb['faces']:
        if 'Normals' in block.point_data:
            n = block.point_data['Normals']
            assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)


def test_cone_faceted_face_emits_cad_warning():
    """The cone apex is a surface singularity where the analytic normal
    is undefined. Two OCCT behaviors are both correct and the library
    contract spans both:

    * Older OCCT triangulations carry no cached normals, so the analytic
      path runs, the apex face abandons it, a single aggregated
      ``CadWarning`` fires, and the face shades faceted (no
      ``_ANALYTIC_NORMALS_FLAG``).
    * Newer ``cadquery-ocp`` builds populate per-node triangulation
      normals, so the analytic path is never needed: no warning, smooth
      shading, and ``_ANALYTIC_NORMALS_FLAG`` is stamped.

    Assert exactly one of the two holds (``uv.lock`` is uncommitted, so
    CI resolves whichever OCCT wheel is current). Either way geometry is
    intact."""
    cone = BRepPrimAPI_MakeCone(5.0, 0.0, 10.0).Shape()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mb = topods_to_multiblock(cone, linear_deflection=0.5)

    abandoned = [
        w
        for w in caught
        if issubclass(w.category, CadWarning)
        and 'abandoned the analytic-normal path' in str(w.message)
    ]
    faces = mb['faces']
    cached_normals = _ANALYTIC_NORMALS_FLAG in faces.field_data
    # Exactly one path: faceted-fallback (warned) XOR cached-normals.
    assert bool(abandoned) != cached_normals

    # Geometry is intact under both paths.
    assert faces.n_blocks >= 1
    assert mb['edges'].n_cells > 0


def test_edges_reused_from_triangulation_lie_on_facets(block_with_hole):
    """Topological edges are recovered from the existing triangulation,
    so every edge point coincides with a face tessellation node."""
    mb = topods_to_multiblock(block_with_hole, linear_deflection=0.4)
    edges = mb['edges']
    merged = mb['faces'].combine(merge_points=False)
    face_pts = merged.points
    sample = edges.points[:: max(1, edges.n_points // 50)]
    for p in sample:
        assert np.min(np.linalg.norm(face_pts - p, axis=1)) < 1e-6


# --- no-B-rep degrade path -----------------------------------------------


def test_feature_edge_fallback_empty_mesh_returns_empty():
    """An empty mesh yields no extractable feature edges; VTK accepts it
    without raising, so the fallback returns an empty PolyData and emits
    no warning."""
    out = _feature_edge_fallback(pv.PolyData(), feature_angle=30.0)
    assert isinstance(out, pv.PolyData)
    assert out.n_cells == 0


def test_resolve_empty_polydata_degrades_without_overlay():
    """A B-rep-less empty mesh resolves to itself with an empty edge
    overlay (the degraded CAD view: shaded faces, no edges)."""
    empty = pv.PolyData()
    faces, edges = _resolve_cad(
        empty, linear_deflection=0.1, angular_deflection=0.5, feature_angle=30.0
    )
    assert faces is empty
    assert edges.n_cells == 0


def test_feature_edge_fallback_drops_noisy_overlay():
    """When extracted edges rival the cell count they are tessellation
    noise and the overlay is dropped (empty PolyData)."""
    noisy = pv.Sphere(theta_resolution=60, phi_resolution=60)
    out = _feature_edge_fallback(noisy, feature_angle=1.0)
    assert out.n_cells == 0


def test_feature_edge_fallback_keeps_real_creases():
    """A box subdivided into many coplanar facets keeps only its 12 sharp
    edges; that count is far below the 25 percent noise threshold, so the
    overlay is kept rather than dropped."""
    dense = pv.Box(quads=True).triangulate()
    for _ in range(3):
        dense = dense.subdivide(1, subfilter='linear')
    out = _feature_edge_fallback(dense, feature_angle=30.0)
    assert 0 < out.n_cells <= 0.25 * dense.n_cells


def test_resolve_plain_polydata_uses_feature_edges():
    sphere = pv.Sphere()
    faces, edges = _resolve_cad(
        sphere, linear_deflection=0.1, angular_deflection=0.5, feature_angle=30.0
    )
    assert faces is sphere
    assert isinstance(edges, pv.PolyData)


def test_resolve_faceted_brep_falls_back_to_creases():
    """A mesh round-tripped through a faceted B-rep has a triangle-soup
    'topology'; when its edge count rivals the cell count the resolver
    falls back to crease extraction."""
    src = pv.Sphere(theta_resolution=12, phi_resolution=12)
    mesh = from_topods(to_topods(src))
    faces, edges = _resolve_cad(
        mesh, linear_deflection=0.1, angular_deflection=0.5, feature_angle=30.0
    )
    assert faces is mesh
    assert edges.n_cells < mesh.n_cells


# --- resolution paths ----------------------------------------------------


def test_resolve_assembly_multiblock_aggregates(block_with_hole):
    """A generic MultiBlock (no faces/edges keys) is resolved per-block:
    faces aggregated, edges merged. A None block is skipped."""
    part_a = from_topods(block_with_hole, linear_deflection=0.3)
    part_b = from_topods(BRepPrimAPI_MakeBox(3.0, 3.0, 3.0).Shape())
    assembly = pv.MultiBlock()
    assembly['a'] = part_a
    assembly['empty'] = None  # exercises the None-block skip
    assembly['b'] = part_b
    faces, edges = _resolve_cad(
        assembly, linear_deflection=0.3, angular_deflection=0.5, feature_angle=30.0
    )
    assert isinstance(faces, pv.MultiBlock)
    assert faces.n_blocks == 2
    assert edges is not None
    assert edges.n_cells > 0


def test_resolve_assembly_nested_multiblock(block_with_hole):
    """A MultiBlock whose block is itself a face/edge MultiBlock is
    resolved recursively."""
    inner = topods_to_multiblock(block_with_hole, linear_deflection=0.4)
    outer = pv.MultiBlock()
    outer['part'] = inner
    faces, edges = _resolve_cad(
        outer, linear_deflection=0.4, angular_deflection=0.5, feature_angle=30.0
    )
    assert isinstance(faces, pv.MultiBlock)
    assert edges is not None
    assert edges.n_cells > 0


def test_resolve_faces_edges_multiblock_used_directly(block_with_hole):
    mb = topods_to_multiblock(block_with_hole, linear_deflection=0.3)
    faces, edges = _resolve_cad(
        mb, linear_deflection=0.3, angular_deflection=0.5, feature_angle=30.0
    )
    assert faces is mb['faces']
    assert edges is mb['edges']


def test_resolve_stashed_edges_polydata(block_with_hole):
    poly = flatten_to_cad_polydata(topods_to_multiblock(block_with_hole, linear_deflection=0.3))
    faces, edges = _resolve_cad(
        poly, linear_deflection=0.3, angular_deflection=0.5, feature_angle=30.0
    )
    assert faces is poly
    assert edges is get_cad_edges(poly)


def test_resolve_cached_topods_recovers_edges(block_with_hole):
    """A PolyData carrying a cached B-rep origin recovers topological
    edges from the cache without re-tessellating the faces."""
    mesh = from_topods(block_with_hole, linear_deflection=0.3)
    faces, edges = _resolve_cad(
        mesh, linear_deflection=0.3, angular_deflection=0.5, feature_angle=30.0
    )
    assert faces is mesh
    assert edges.n_cells > 0


def test_resolve_raw_topods(block_with_hole):
    faces, edges = _resolve_cad(
        block_with_hole, linear_deflection=0.3, angular_deflection=0.5, feature_angle=30.0
    )
    assert isinstance(faces, pv.MultiBlock)
    assert edges.n_cells > 0


# --- as_cad_multiblock ---------------------------------------------------


def test_as_cad_multiblock_keys(block_with_hole):
    cad = as_cad_multiblock(block_with_hole, linear_deflection=0.3)
    assert set(cad.keys()) == {'faces', 'edges'}
    assert cad['edges'].n_cells > 0


def test_as_cad_multiblock_plain_mesh_empty_edges_block():
    """A sphere has no creases at 30 deg; the edges block is an empty
    PolyData, never ``None``."""
    cad = as_cad_multiblock(pv.Sphere())
    assert isinstance(cad['edges'], pv.PolyData)


# --- flatten -------------------------------------------------------------


def test_flatten_empty_multiblock_is_empty_polydata():
    mb = pv.MultiBlock()
    mb['faces'] = pv.MultiBlock()
    mb['edges'] = pv.PolyData()
    poly = flatten_to_cad_polydata(mb)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == 0
    assert get_cad_edges(poly) is not None


def test_flatten_without_faces_or_edges_keys():
    """A bare per-face MultiBlock (no 'faces'/'edges' keys) is treated as
    the faces collection itself."""
    mb = topods_to_multiblock(BRepPrimAPI_MakeBox(3.0, 4.0, 5.0).Shape(), linear_deflection=0.5)
    bare = mb['faces']
    assert 'faces' not in bare.keys()  # noqa: SIM118 (block-name list, not data)
    poly = flatten_to_cad_polydata(bare)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0


# --- add_cad rendering branches ------------------------------------------


def test_add_cad_default_color_and_edges(block_with_hole):
    pl = pv.Plotter()
    actors = add_cad(pl, block_with_hole, linear_deflection=0.3)
    assert actors['faces'] is not None
    assert actors['edges'] is not None
    pl.close()


def test_add_cad_edges_disabled(block_with_hole):
    pl = pv.Plotter()
    actors = add_cad(pl, block_with_hole, edges=False, linear_deflection=0.3)
    assert actors['edges'] is None
    pl.close()


def test_add_cad_custom_edge_color_and_tubes(block_with_hole):
    pl = pv.Plotter()
    actors = add_cad(
        pl,
        block_with_hole,
        edge_color='red',
        line_width=4.0,
        render_lines_as_tubes=True,
        linear_deflection=0.3,
    )
    assert actors['edges'] is not None
    pl.close()


def test_add_cad_scalar_driven_skips_default_color(block_with_hole):
    """When the caller maps a scalar, ``add_cad`` must not inject a solid
    colour that would override the mapping."""
    mesh = from_topods(block_with_hole, linear_deflection=0.3)
    mesh['height'] = mesh.points[:, 2]
    pl = pv.Plotter()
    actors = add_cad(pl, mesh, scalars='height', cmap='viridis', linear_deflection=0.3)
    assert actors['faces'] is not None
    pl.close()


def test_add_cad_rgba_scalars(block_with_hole):
    mesh = from_topods(block_with_hole, linear_deflection=0.3)
    rng = np.random.default_rng(0)
    mesh['rgba'] = (rng.random((mesh.n_points, 4)) * 255).astype(np.uint8)
    pl = pv.Plotter()
    actors = add_cad(pl, mesh, scalars='rgba', rgba=True, linear_deflection=0.3)
    assert actors['faces'] is not None
    pl.close()


def test_add_cad_silhouette(block_with_hole):
    pl = pv.Plotter()
    actors = add_cad(pl, block_with_hole, silhouette=True, smooth=False, linear_deflection=0.3)
    assert actors['faces'] is not None
    pl.close()


def test_add_cad_render_lines_as_tubes_offset(filleted_box):
    """The edge mapper coincident-topology offset path runs for a real
    curved-edge overlay."""
    pl = pv.Plotter()
    actors = add_cad(pl, filleted_box, linear_deflection=0.3, render_lines_as_tubes=True)
    assert actors['edges'] is not None
    pl.close()


def test_add_cad_unstructured_grid_input(block_with_hole):
    """An UnstructuredGrid hits the ``pv.DataSet`` branch: it must be
    surfaced with the explicit ``algorithm='dataset_surface'`` so it does
    not emit a ``PyVistaFutureWarning`` (fatal under filterwarnings=error)
    on the way into ``add_cad``."""
    mesh = from_topods(block_with_hole, linear_deflection=0.3)
    ugrid = mesh.cast_to_unstructured_grid()
    pl = pv.Plotter()
    actors = add_cad(pl, ugrid, linear_deflection=0.3)
    assert actors['faces'] is not None
    pl.close()


def test_as_cad_multiblock_unstructured_grid_input(block_with_hole):
    """The same DataSet -> surface path through ``as_cad_multiblock``."""
    mesh = from_topods(block_with_hole, linear_deflection=0.3)
    cad = as_cad_multiblock(mesh.cast_to_unstructured_grid(), linear_deflection=0.3)
    assert set(cad.keys()) == {'faces', 'edges'}


def test_plot_cad_splits_show_kwargs(block_with_hole, tmp_path):
    out = tmp_path / 'cad.png'
    plot_cad(
        block_with_hole,
        off_screen=True,
        screenshot=str(out),
        linear_deflection=0.3,
        edge_color='black',
    )
    assert out.is_file()


def test_plot_cad_return_cpos(block_with_hole):
    cpos = plot_cad(block_with_hole, off_screen=True, return_cpos=True, linear_deflection=0.4)
    assert cpos is not None


def test_plotter_component_add_via_accessor(block_with_hole):
    pl = pv.Plotter()
    result = pl.cad.add(block_with_hole, linear_deflection=0.3)
    assert set(result) == {'faces', 'edges'}
    pl.close()


def test_register_component_when_already_registered():
    """A second registration is a no-op (idempotency guard)."""
    before = {getattr(c, 'name', None) for c in pv.registered_plotter_components()}
    assert 'cad' in before
    register_cad_plotter_component()
    after = {getattr(c, 'name', None) for c in pv.registered_plotter_components()}
    assert after == before


# --- internal helper branches --------------------------------------------


def test_ensure_mesh_rejects_non_shape():
    with pytest.raises(TessellationError, match=r'OCCT tessellation failed'):
        _ensure_mesh('not-a-shape', 0.1, 0.5)


def test_curve_kind_non_edge_is_other():
    assert _curve_kind('not-an-edge') == len(_CURVE_KINDS) - 1


def test_shape_has_faces_non_shape_is_false():
    assert _shape_has_faces('not-a-shape') is False


def test_shape_has_faces_true_for_solid(block_with_hole):
    assert _shape_has_faces(block_with_hole) is True


# --- image regression: the CAD view itself -------------------------------


def test_cad_view_image_regression(verify_image_cache, block_with_hole):
    """The one place image-regression is the thing under test.

    Real curved BREP geometry (a drilled block: planar faces + a
    cylindrical wall + circular rims), off-axis ``iso`` camera, smooth
    analytic shading with the topological edge overlay. A behavioural
    assertion on the resolved structure guards the data before the
    render is verified.
    """
    verify_image_cache.high_variance_test = True
    cad = as_cad_multiblock(block_with_hole, linear_deflection=0.25)
    # Behavioural guard before show(): a drilled box keeps its sharp
    # edges (>= the 12 box edges) and resolves into multiple faces.
    assert cad['edges'].n_cells >= 12
    assert cad['faces'].n_blocks >= 7

    pl = pv.Plotter()
    add_cad(pl, block_with_hole, linear_deflection=0.25, edge_color='black')
    pl.camera_position = 'iso'
    pl.show()


def test_filleted_box_image_regression(verify_image_cache, filleted_box):
    """Second image baseline: a rounded part is the canonical case where
    analytic smooth shading and curved feature edges visibly differ from
    a generic triangle-mesh render. Off-axis camera, edge overlay."""
    verify_image_cache.high_variance_test = True
    cad = as_cad_multiblock(filleted_box, linear_deflection=0.3)
    assert cad['faces'].n_blocks >= 6
    assert cad['edges'].n_cells > 0

    pl = pv.Plotter()
    add_cad(pl, filleted_box, linear_deflection=0.3, edge_color='black', smooth=True)
    pl.camera_position = [(60, -50, 45), (10, 6, 4), (0, 0, 1)]
    pl.show()
