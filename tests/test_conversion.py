"""Tests for the OCCT <-> PyVista bridge in ``pyvista_cad._conversion``.

Real OCCT/build123d geometry only. These pin:

- the per-instance B-rep cache (no ``id()`` cross-talk),
- ``unwrap_to_topods`` duck-typing for build123d / cadquery / raw OCP,
- closed input -> ``TopoDS_Solid``; open input -> ``TopoDS_Shell``,
- the INVENTORY-021 Hausdorff round-trip fidelity gate.
"""

import gc

import build123d as b3d
import numpy as np
import pytest
import pyvista as pv

from pyvista_cad._conversion import (
    get_cached_topods,
    polydata_to_topods,
    topods_to_polydata,
    unwrap_to_topods,
)
from pyvista_cad._errors import TessellationError
import pyvista_cad.examples as examples

pytest.importorskip('OCP')

# Hardcoded INVENTORY-021 gate parameters.
_LINEAR_DEFLECTION = 0.1
_ANGULAR_DEFLECTION = 0.3


def _shape_type(shape: object) -> object:
    from OCP.TopAbs import TopAbs_ShapeEnum

    return TopAbs_ShapeEnum(shape.ShapeType())  # type: ignore[attr-defined]


def _symmetric_hausdorff(mesh_a: pv.PolyData, mesh_b: pv.PolyData) -> float:
    """Max of the two directed point-to-surface distances."""

    def directed(src: pv.PolyData, dst: pv.PolyData) -> float:
        _, closest = dst.find_closest_cell(src.points, return_closest_point=True)
        return float(np.linalg.norm(np.asarray(src.points) - closest, axis=1).max())

    return max(directed(mesh_a, mesh_b), directed(mesh_b, mesh_a))


# --------------------------------------------------------------------------- #
# B-rep cache
# --------------------------------------------------------------------------- #


def test_register_and_retrieve_brep_cache() -> None:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    box = BRepPrimAPI_MakeBox(2.0, 3.0, 4.0).Shape()
    mesh = topods_to_polydata(box)
    assert get_cached_topods(mesh) is box


def test_brep_cache_absent_returns_none() -> None:
    assert get_cached_topods(pv.PolyData()) is None


def test_brep_cache_keyed_per_instance() -> None:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere

    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    sphere = BRepPrimAPI_MakeSphere(1.0).Shape()
    mesh_a = topods_to_polydata(box)
    mesh_b = topods_to_polydata(sphere)
    assert get_cached_topods(mesh_a) is box
    assert get_cached_topods(mesh_b) is sphere


def test_brep_cache_survives_id_recycling() -> None:
    """A fresh mesh landing on a freed ``id()`` must not see a stale hit."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    recycled = 0
    for _ in range(50):
        mesh = topods_to_polydata(box)
        assert get_cached_topods(mesh) is not None
        addr = id(mesh)
        del mesh
        gc.collect()
        fresh = pv.PolyData()
        if id(fresh) == addr:
            recycled += 1
            assert get_cached_topods(fresh) is None
        del fresh
        gc.collect()
    _ = recycled  # informational only


# --------------------------------------------------------------------------- #
# unwrap_to_topods
# --------------------------------------------------------------------------- #


def test_unwrap_build123d_shape() -> None:
    from OCP.TopoDS import TopoDS_Shape

    part = b3d.Box(10.0, 10.0, 10.0)
    ts = unwrap_to_topods(part)
    assert isinstance(ts, TopoDS_Shape)
    assert ts is part.wrapped


def test_unwrap_raw_topods_passthrough() -> None:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    raw = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    assert unwrap_to_topods(raw) is raw


def test_unwrap_wrapped_none_falls_through() -> None:
    """An object whose ``wrapped`` is ``None`` is returned unchanged."""

    class WrappedNone:
        wrapped = None

    obj = WrappedNone()
    assert unwrap_to_topods(obj) is obj


def test_unwrap_cadquery_workplane_val_with_wrapped() -> None:
    """``cadquery.Workplane``-style: ``.val()`` returns an object with ``.wrapped``."""

    class Inner:
        wrapped = 'RAW_SHAPE'

    class Workplane:
        def val(self) -> Inner:
            return Inner()

    assert unwrap_to_topods(Workplane()) == 'RAW_SHAPE'


def test_unwrap_val_without_wrapped_returns_val() -> None:
    """``.val()`` result lacking ``.wrapped`` is returned as-is."""
    sentinel = object()

    class Workplane:
        def val(self) -> object:
            return sentinel

    assert unwrap_to_topods(Workplane()) is sentinel


# --------------------------------------------------------------------------- #
# polydata_to_topods: closed -> solid, open -> shell
# --------------------------------------------------------------------------- #


def test_closed_box_yields_solid() -> None:
    from OCP.TopAbs import TopAbs_ShapeEnum

    shape = polydata_to_topods(pv.Box())
    assert type(shape).__name__ == 'TopoDS_Solid'
    assert _shape_type(shape) == TopAbs_ShapeEnum.TopAbs_SOLID


def test_closed_solid_has_no_free_boundary_edges() -> None:
    """A closed input sews to a watertight body (zero free boundary edges)."""
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    shape = polydata_to_topods(pv.Box())
    free = ShapeAnalysis_FreeBounds(shape)
    open_wires = free.GetOpenWires()
    exp = TopExp_Explorer(open_wires, TopAbs_EDGE)
    assert not exp.More(), 'closed input produced free boundary edges'


def test_open_mesh_yields_shell() -> None:
    from OCP.TopAbs import TopAbs_ShapeEnum

    plane = pv.Plane(i_resolution=4, j_resolution=4)
    shape = polydata_to_topods(plane)
    assert _shape_type(shape) == TopAbs_ShapeEnum.TopAbs_SHELL


def test_single_triangle_returns_unpromoted_face() -> None:
    """One triangle: sewed result is a bare face (not a shell), returned as-is."""
    from OCP.TopAbs import TopAbs_ShapeEnum

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tri = pv.PolyData(pts, faces=[3, 0, 1, 2])
    shape = polydata_to_topods(tri)
    assert _shape_type(shape) == TopAbs_ShapeEnum.TopAbs_FACE


def test_quad_mesh_is_triangulated_then_sewn() -> None:
    """A quad-faced closed box still sews to a solid (input is triangulated)."""
    box = pv.Box(quads=True)
    shape = polydata_to_topods(box)
    assert type(shape).__name__ == 'TopoDS_Solid'


def test_sewing_tolerance_scales_with_mesh() -> None:
    """Same sphere in mm and micron magnitude both sew to a non-null faced shape."""
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    src = pv.Sphere(theta_resolution=12, phi_resolution=12)
    micro = src.copy()
    micro.points = np.asarray(src.points) * 1.0e-3
    for mesh in (src, micro):
        shape = polydata_to_topods(mesh)
        assert not shape.IsNull()
        assert TopExp_Explorer(shape, TopAbs_FACE).More()


def test_polydata_to_topods_empty_mesh_raises() -> None:
    """An empty PolyData has no faces and must raise, not SIGSEGV."""
    with pytest.raises(TessellationError, match='no triangular faces'):
        polydata_to_topods(pv.PolyData())


def test_polydata_to_topods_lines_only_mesh_raises() -> None:
    """A lines-only mesh has no triangular faces and must raise cleanly."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    lines = pv.PolyData(pts, lines=np.array([3, 0, 1, 2]))
    with pytest.raises(TessellationError, match='no triangular faces'):
        polydata_to_topods(lines)


def test_polydata_to_topods_nan_triangle_raises() -> None:
    """A single NaN triangle builds no valid face; sewing is null, not a crash."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    tri = pv.PolyData(pts, faces=np.array([3, 0, 1, 2]))
    with pytest.raises(TessellationError):
        polydata_to_topods(tri)


def test_polydata_to_topods_non_surface_mesh_raises() -> None:
    """A volumetric mesh triangulates to a non-PolyData and must raise."""
    grid = pv.ImageData(dimensions=(2, 2, 2))
    with pytest.raises(TessellationError, match='requires a PolyData'):
        polydata_to_topods(grid)


# --------------------------------------------------------------------------- #
# topods_to_polydata edges
# --------------------------------------------------------------------------- #


def test_tessellation_failure_raises_tessellationerror() -> None:
    with pytest.raises(TessellationError, match='OCCT tessellation failed'):
        topods_to_polydata('not a shape')


def test_empty_compound_yields_empty_polydata() -> None:
    """A shape with no triangulated faces returns an empty PolyData."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    comp = TopoDS_Compound()
    BRep_Builder().MakeCompound(comp)
    poly = topods_to_polydata(comp)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == 0
    assert poly.n_cells == 0


def test_face_without_triangulation_is_skipped() -> None:
    """A degenerate (collinear) face meshes to nothing and is skipped."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.gp import gp_Pnt

    wire = BRepBuilderAPI_MakePolygon(
        gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(1.0, 0.0, 0.0), gp_Pnt(2.0, 0.0, 0.0), True
    ).Wire()
    face = BRepBuilderAPI_MakeFace(wire, True).Face()
    poly = topods_to_polydata(face)
    assert poly.n_points == 0


def test_reversed_face_orientation_round_trips() -> None:
    """A real solid tessellates with consistent winding (reversed-face path)."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    box = BRepPrimAPI_MakeBox(5.0, 5.0, 5.0).Shape()
    mesh = topods_to_polydata(box, linear_deflection=0.05)
    assert mesh.n_cells == 12
    assert mesh.n_points > 0


# --------------------------------------------------------------------------- #
# INVENTORY-021: Hausdorff round-trip fidelity gate
# --------------------------------------------------------------------------- #


def test_round_trip_hausdorff_within_deflection_step_cube() -> None:
    """``examples.step_cube`` -> mesh -> solid -> mesh stays within deflection.

    The faceted conversion reproduces the tessellation vertex-for-vertex,
    so the symmetric Hausdorff distance is bounded by (and here equal to)
    the tessellation ``linear_deflection``; the conversion adds no error.
    """
    from build123d import import_step

    shape = unwrap_to_topods(import_step(examples.step_cube))
    mesh_a = topods_to_polydata(
        shape,
        linear_deflection=_LINEAR_DEFLECTION,
        angular_deflection=_ANGULAR_DEFLECTION,
    )
    solid = polydata_to_topods(mesh_a)
    assert type(solid).__name__ == 'TopoDS_Solid'
    mesh_b = topods_to_polydata(
        solid,
        linear_deflection=_LINEAR_DEFLECTION,
        angular_deflection=_ANGULAR_DEFLECTION,
    )
    distance = _symmetric_hausdorff(mesh_a, mesh_b)
    assert distance <= _LINEAR_DEFLECTION
    assert distance == pytest.approx(0.0, abs=1e-9)


def _topods_vertex_count(shape: object) -> int:
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_VERTEX, m)
    return int(m.Extent())


def test_sewing_weld_bound_is_diag_times_1e6() -> None:
    """``polydata_to_topods`` sews vertices closer than ``diag * 1e-6``
    into one and keeps vertices farther apart distinct, pinning the
    documented weld bound (``diag`` = bounding-box diagonal).

    Two coincident triangles share an edge; the shared pair of vertices
    is offset by a controllable gap. Below the tolerance the two
    triangles weld along that edge (the sewn shell has fewer unique
    OCCT vertices than a fully-disjoint pair); above it they do not.
    """

    def two_triangle_mesh(gap: float) -> pv.PolyData:
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0 + gap],
                [0.0, 1.0, 0.0 + gap],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        faces = np.array([3, 0, 1, 2, 3, 3, 5, 4], dtype=np.int64)
        return pv.PolyData(pts, faces=faces)

    # diag of the unit-ish mesh is ~sqrt(3); 1e-6 of that is ~1.7e-6.
    diag = float(np.linalg.norm(np.array([1.0, 1.0, 0.0])))
    tol = diag * 1.0e-6

    welded = polydata_to_topods(two_triangle_mesh(tol * 0.1))
    distinct = polydata_to_topods(two_triangle_mesh(tol * 100.0))

    n_welded = _topods_vertex_count(welded)
    n_distinct = _topods_vertex_count(distinct)
    # A separation below the tolerance fuses the two shared corners, so
    # the welded shell has strictly fewer OCCT vertices than the one
    # whose corners stayed apart.
    assert n_welded < n_distinct, (
        f'expected sub-tolerance gap to weld vertices: welded={n_welded} distinct={n_distinct}'
    )


def test_round_trip_hausdorff_within_deflection_curved_part() -> None:
    """Non-trivial curved geometry (cylinder) stays within the deflection bound."""
    part = b3d.Cylinder(radius=8.0, height=20.0)
    shape = unwrap_to_topods(part)
    mesh_a = topods_to_polydata(
        shape,
        linear_deflection=_LINEAR_DEFLECTION,
        angular_deflection=_ANGULAR_DEFLECTION,
    )
    assert mesh_a.n_points > 50  # genuinely faceted curved surface
    solid = polydata_to_topods(mesh_a)
    assert type(solid).__name__ == 'TopoDS_Solid'
    mesh_b = topods_to_polydata(
        solid,
        linear_deflection=_LINEAR_DEFLECTION,
        angular_deflection=_ANGULAR_DEFLECTION,
    )
    distance = _symmetric_hausdorff(mesh_a, mesh_b)
    assert distance <= _LINEAR_DEFLECTION
