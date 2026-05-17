"""CAD-friendly visualization: topological edges + per-face tessellation.

CAD kernels store *exact* B-rep geometry (analytic / NURBS surfaces,
trimmed faces, parametric edge curves). A GPU only draws triangles, so a
CAD viewer tessellates each topological face to a tolerance and draws
the topological *edges* as separate polylines on top. The triangle-mesh
edges a generic mesh viewer shows (``show_edges=True``) are arbitrary
tessellation artifacts and meaningless to a CAD user; the edges they
care about are the feature curves of the model.

This module reproduces that pipeline:

* :func:`topods_to_edges` — extract topological edges, *reusing* the
  face triangulation (no extra meshing pass), so edge polylines lie
  exactly on the rendered facets.
* :func:`topods_to_multiblock` — per-face :class:`pyvista.PolyData`
  blocks (preserving face identity) carrying analytic point normals
  for smooth shading, plus a sibling edges block.
* :func:`flatten_to_cad_polydata` — collapse the per-face MultiBlock to
  a single :class:`pyvista.PolyData` with a ``cad.face_id`` cell array
  and the edges stashed as a companion attribute.

All ``OCP`` imports are lazy: ``import pyvista_cad`` must not load OCP.
"""

from typing import Any, cast
import warnings

import numpy as np
import pyvista as pv

from pyvista_cad._conversion import _register_brep_cache, get_cached_topods, unwrap_to_topods
from pyvista_cad._errors import CadWarning, TessellationError

# Companion-attribute name: the edges PolyData stashed on a flattened
# face PolyData. Mirrors the ``_cad_topods_shape`` stash convention in
# :mod:`pyvista_cad._conversion` (avoids an ``id()``-recycle cache).
_EDGES_ATTR = '_cad_edges'

# field_data marker set on a faces MultiBlock whose blocks carry
# trustworthy per-point *analytic* surface normals (the B-rep
# tessellation path). When present, smooth shading is correct as-is and
# vertex-splitting at creases would discard those normals. When absent
# (a plain mesh / faceted fallback), smooth shading must split sharp
# edges so hard corners do not melt into rounded blobs.
_ANALYTIC_NORMALS_FLAG = 'cad.analytic_normals'

# GeomAbs_CurveType ordinal -> human label. OCCT enum order is stable
# across the supported OCP / OCCT 7.x line.
_CURVE_KINDS: tuple[str, ...] = (
    'line',
    'circle',
    'ellipse',
    'hyperbola',
    'parabola',
    'bezier',
    'bspline',
    'offset',
    'other',
)


def _geom_errors() -> tuple[type[BaseException], type[BaseException], type[BaseException]]:
    """Exception types a failing OCCT geometry query may raise.

    In this OCP build the OCCT ``Standard_*`` exceptions do not share a
    common ``Standard_Failure`` base: ``Standard_NullObject`` (raised for
    an edge with no 3D curve, the IGES/wireframe case) derives directly
    from :class:`Exception`, not from ``Standard_Failure``. Catching only
    ``Standard_Failure`` would let it escape, so both are returned
    alongside ``TypeError`` (a non-shape / wrong-argument call). Imported
    lazily: ``import pyvista_cad`` must not load OCP.
    """
    from OCP.Standard import Standard_Failure, Standard_NullObject

    return (Standard_Failure, Standard_NullObject, TypeError)


def _ensure_mesh(shape: Any, linear_deflection: float, angular_deflection: float) -> None:
    """Run OCCT incremental meshing in place (idempotent if already meshed)."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh

    try:
        BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    except _geom_errors() as exc:
        # OCCT ``Standard_*`` (``Exception`` subclasses, not
        # ``RuntimeError``, in this OCP build) on un-meshable geometry;
        # ``TypeError`` if ``shape`` is not a TopoDS_Shape. Re-raised as a
        # package error, never swallowed.
        msg = f'OCCT tessellation failed: {exc}'
        raise TessellationError(msg) from exc


def _curve_kind(edge: Any) -> int:
    """Return the :data:`_CURVE_KINDS` ordinal for ``edge``'s geometry."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    try:
        code = int(BRepAdaptor_Curve(edge).GetType())
    except _geom_errors():
        # OCCT raises ``Standard_NullObject`` (an ``Exception`` subclass,
        # not ``RuntimeError``) for an edge with no well-defined 3D curve;
        # classify it as 'other'. This is a label fallback only, not
        # geometry loss, so no warning is warranted.
        return len(_CURVE_KINDS) - 1  # 'other'
    return code if 0 <= code < len(_CURVE_KINDS) else len(_CURVE_KINDS) - 1


def _kind_legend() -> str:
    """JSON-ish legend mapping ``cad.edge_kind`` codes to labels."""
    return ';'.join(f'{i}={name}' for i, name in enumerate(_CURVE_KINDS))


def _sample_free_edge(edge: Any, deflection: float) -> np.ndarray | None:
    """Sample a face-less edge along its 3D curve (fallback path).

    Used only for wireframe edges that belong to no face and therefore
    have no ``PolygonOnTriangulation`` (e.g. IGES construction curves).

    Returns an ``(n, 3)`` point array, an empty array when the edge is
    too short to draw, or ``None`` when OCCT could not sample the curve
    at all (the edge is then lost from the rendered wireframe; the
    caller aggregates these into a single diagnostic).
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_TangentialDeflection

    try:
        curve = BRepAdaptor_Curve(edge)
        sampler = GCPnts_TangentialDeflection(curve, deflection, deflection)
        n = sampler.NbPoints()
        if n < 2:
            return np.empty((0, 3), dtype=float)
        pts = np.empty((n, 3), dtype=float)
        for i in range(1, n + 1):
            p = sampler.Value(i)
            pts[i - 1] = (p.X(), p.Y(), p.Z())
    except _geom_errors():
        # OCCT ``Standard_NullObject`` (an ``Exception`` subclass, not
        # ``RuntimeError``, in this OCP build): the edge has no continuous
        # 3D curve. This is the IGES/wireframe construction-curve case;
        # the caller aggregates it into one ``CadWarning``.
        return None
    return pts


def _face_unit_normal_at_edge(face: Any, edge: Any) -> np.ndarray | None:
    """Analytic unit surface normal of ``face`` at ``edge``'s midpoint.

    Orientation-corrected (a ``REVERSED`` face flips the normal).
    Returns ``None`` if the normal is undefined (e.g. degenerate pcurve).
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d, BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.gp import gp_Dir, gp_Pnt, gp_Vec

    try:
        adaptor = BRepAdaptor_Surface(face)
        reversed_face = face.Orientation() == 1  # TopAbs_REVERSED

        # Fast path: a planar face (every triangle of a sewn faceted
        # solid is one) has a constant normal. Read the analytic plane
        # straight from the adaptor: no Geom downcast (absent in this OCP
        # build) and no pcurve needed.
        if adaptor.GetType() == GeomAbs_Plane:
            ax = adaptor.Plane().Axis().Direction()
            n = np.array([ax.X(), ax.Y(), ax.Z()], dtype=float)
            return -n if reversed_face else n

        # Curved face: find the UV at the edge midpoint, then take the
        # normal as the normalized cross product of the surface partials
        # there (du x dv), orientation-corrected. Use the 2D pcurve
        # adaptor (it carries its own parameter range, avoiding the
        # by-reference First/Last out-args of ``CurveOnSurface_s`` that
        # this OCP build does not return). The sole caller
        # (_tangent_internal_edges) only ever passes an edge that bounds
        # ``face``, so its pcurve always has a non-empty range; a
        # collapsed range means OCCT could not build the pcurve, which
        # surfaces as ``Standard_NullObject`` from ``Value`` below and is
        # handled by the ``except`` (no usable normal -> keep the edge).
        pc = BRepAdaptor_Curve2d(edge, face)
        uv = pc.Value(0.5 * (pc.FirstParameter() + pc.LastParameter()))

        p = gp_Pnt()
        du = gp_Vec()
        dv = gp_Vec()
        adaptor.D1(uv.X(), uv.Y(), p, du, dv)
        cross = du.Crossed(dv)
        if cross.Magnitude() < 1e-12:
            # Degenerate point (e.g. a cone apex): no defined normal.
            return None
        d = gp_Dir(cross)
        if reversed_face:
            d.Reverse()
        return np.array([d.X(), d.Y(), d.Z()], dtype=float)
    except _geom_errors():
        # OCCT ``Standard_*`` (``Exception`` subclasses, not
        # ``RuntimeError``, in this OCP build): the normal is undefined at
        # this parameter (degenerate pcurve / singular point). ``None`` is
        # the documented "no usable normal" signal; the caller
        # (_tangent_internal_edges) simply keeps the edge, so this is
        # control flow, not data loss.
        return None


def _tangent_internal_edges(shape: Any, tol_deg: float) -> Any:
    """Map of edges to *suppress*: shared by two G1-tangent faces.

    An edge between two faces whose surface normals are within
    ``tol_deg`` is either a smooth/tangent transition (hidden by
    convention in CAD viewers) or — for a faceted solid where every
    triangle is its own planar face — pure tessellation noise. Real
    feature edges (sharp corners, silhouettes against a boundary) sit
    between faces at a meaningful angle and are kept. Edges that are
    not shared by exactly two faces (true boundaries, non-manifold)
    are never suppressed.
    """
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_MapOfShape,
    )

    skip = TopTools_MapOfShape()
    amap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, amap)
    cos_tol = np.cos(np.deg2rad(tol_deg))

    for i in range(1, amap.Extent() + 1):
        faces = amap.FindFromIndex(i)
        if faces.Extent() != 2:
            continue
        edge = TopoDS.Edge_s(amap.FindKey(i))
        f1 = TopoDS.Face_s(faces.First())
        f2 = TopoDS.Face_s(faces.Last())
        n1 = _face_unit_normal_at_edge(f1, edge)
        n2 = _face_unit_normal_at_edge(f2, edge)
        if n1 is None or n2 is None:
            continue
        if abs(float(np.dot(n1, n2))) >= cos_tol:
            skip.Add(edge)
    return skip


def topods_to_edges(
    shape: Any,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    tangent_tol_deg: float = 8.0,
) -> pv.PolyData:
    """Extract topological edges of ``shape`` as a polyline mesh.

    Edges are recovered from the *existing* face triangulation via
    OCCT's ``PolygonOnTriangulation`` — no extra meshing pass — so the
    polylines lie exactly on the tessellated facets. Each topological
    edge becomes a single polyline cell, deduplicated across the faces
    that share it. Face-less edges (wireframe geometry) fall back to
    direct 3D curve sampling.

    Edges shared by two faces whose surface normals agree to within
    ``tangent_tol_deg`` are dropped: these are smooth/tangent
    transitions (hidden by convention in CAD viewers) or, for a
    faceted solid where every triangle is its own planar face, pure
    tessellation noise. Real sharp edges and open boundaries are kept.

    Parameters
    ----------
    shape : OCP.TopoDS.TopoDS_Shape
        Shape to extract edges from. ``build123d`` / ``cadquery``
        wrappers are unwrapped automatically.
    linear_deflection : float, default: 0.1
        Chordal deviation used if ``shape`` is not already meshed.
    angular_deflection : float, default: 0.5
        Angular deviation (radians) used if not already meshed.
    tangent_tol_deg : float, default: 8.0
        Adjacent-face angle below which a shared edge is treated as a
        tangent/coplanar internal edge and suppressed. Set to ``0`` to
        keep every topological edge.

    Returns
    -------
    pyvista.PolyData
        Line-cell mesh. Cell arrays: ``cad.edge_id`` (int, per edge)
        and ``cad.edge_kind`` (int code; see ``field_data
        ['cad.edge_kind_legend']``).

    Raises
    ------
    pyvista_cad.TessellationError
        If OCCT meshing fails.

    Examples
    --------
    Extract feature edges from a build123d box (requires the
    ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad import topods_to_edges
    >>> edges = topods_to_edges(b3d.Box(10, 10, 10).wrapped)
    >>> type(edges).__name__
    'PolyData'
    >>> edges.n_cells
    12

    """
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_MapOfShape

    shape = unwrap_to_topods(shape)
    _ensure_mesh(shape, linear_deflection, angular_deflection)

    skip_edges = _tangent_internal_edges(shape, tangent_tol_deg) if tangent_tol_deg > 0 else None

    points: list[tuple[float, float, float]] = []
    lines: list[int] = []
    edge_ids: list[int] = []
    edge_kinds: list[int] = []
    seen = TopTools_MapOfShape()
    next_id = 0

    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    while fexp.More():
        face = TopoDS.Face_s(fexp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            fexp.Next()
            continue
        trsf = loc.Transformation()

        eexp = TopExp_Explorer(face, TopAbs_EDGE)
        while eexp.More():
            edge = TopoDS.Edge_s(eexp.Current())
            if not seen.Add(edge):  # already emitted via another face
                eexp.Next()
                continue
            if skip_edges is not None and skip_edges.Contains(edge):
                eexp.Next()
                continue
            poly = BRep_Tool.PolygonOnTriangulation_s(edge, tri, loc)
            if poly is None:
                eexp.Next()
                continue
            node_idx = poly.Nodes()
            n = node_idx.Length()
            if n < 2:
                eexp.Next()
                continue
            offset = len(points)
            for k in range(node_idx.Lower(), node_idx.Upper() + 1):
                node = tri.Node(node_idx.Value(k))
                node.Transform(trsf)
                points.append((node.X(), node.Y(), node.Z()))
            lines.append(n)
            lines.extend(range(offset, offset + n))
            edge_ids.append(next_id)
            edge_kinds.append(_curve_kind(edge))
            next_id += 1
            eexp.Next()
        fexp.Next()

    # Face-less wireframe edges (belong to no face triangulation).
    abandoned_edges = 0
    eexp = TopExp_Explorer(shape, TopAbs_EDGE)
    while eexp.More():
        edge = TopoDS.Edge_s(eexp.Current())
        if seen.Add(edge):
            pts = _sample_free_edge(edge, linear_deflection)
            if pts is None:
                abandoned_edges += 1
            elif len(pts) >= 2:
                offset = len(points)
                points.extend(map(tuple, pts))
                lines.append(len(pts))
                lines.extend(range(offset, offset + len(pts)))
                edge_ids.append(next_id)
                edge_kinds.append(_curve_kind(edge))
                next_id += 1
        eexp.Next()

    if abandoned_edges:
        warnings.warn(
            f'{abandoned_edges} face-less edge(s) could not be sampled '
            'from their analytic curve and are missing from the CAD edge '
            'overlay; the rendered wireframe is incomplete.',
            CadWarning,
            stacklevel=2,
        )

    edges = pv.PolyData()
    if points:
        edges.points = np.asarray(points, dtype=float)
        edges.lines = np.asarray(lines, dtype=np.int64)
        edges.cell_data['cad.edge_id'] = np.asarray(edge_ids, dtype=np.int64)
        edges.cell_data['cad.edge_kind'] = np.asarray(edge_kinds, dtype=np.int64)
        edges.field_data['cad.edge_kind_legend'] = np.array([_kind_legend()])
    return edges


def _face_normals(face: Any, tri: Any, trsf: Any, n_nodes: int) -> np.ndarray | None:
    """Return transformed per-node *analytic* surface normals, or ``None``.

    VTK's facet normals make a coarse CAD mesh look polygonal. Recovering
    the true surface normal at each triangulation node (evaluated from
    the analytic surface at the node's UV) is the single biggest
    CAD-look improvement: a 12-facet cylinder shades perfectly round.

    Uses cached triangulation normals when OCCT stored them; otherwise
    evaluates the underlying ``Geom_Surface`` at the per-node UV.
    """
    # Fast path: triangulation already carries normals.
    try:
        if tri.HasNormals():
            normals = np.empty((n_nodes, 3), dtype=float)
            for i in range(1, n_nodes + 1):
                d = tri.Normal(i).Transformed(trsf)
                normals[i - 1] = (d.X(), d.Y(), d.Z())
            return normals
    except _geom_errors():
        # Cached-normal read failed; fall through to the analytic path.
        pass

    # Analytic path: evaluate surface normal at each node's UV.
    try:
        if not tri.HasUVNodes():
            return None
        from OCP.BRep import BRep_Tool
        from OCP.GeomLProp import GeomLProp_SLProps

        surface = BRep_Tool.Surface_s(face)
        if surface is None:
            return None
        reversed_face = face.Orientation() == 1  # TopAbs_REVERSED
        normals = np.empty((n_nodes, 3), dtype=float)
        for i in range(1, n_nodes + 1):
            uv = tri.UVNode(i)
            props = GeomLProp_SLProps(surface, uv.X(), uv.Y(), 1, 1e-6)
            if not props.IsNormalDefined():
                return None
            d = props.Normal()
            if reversed_face:
                d.Reverse()
            d = d.Transformed(trsf)
            normals[i - 1] = (d.X(), d.Y(), d.Z())
    except _geom_errors():
        # OCCT ``Standard_*`` (``Exception`` subclasses, not
        # ``RuntimeError``) while evaluating the analytic surface: the
        # face falls back to VTK facet normals (faceted shading).
        # The caller aggregates this into one per-call diagnostic.
        return None
    return normals


def topods_to_multiblock(
    shape: Any,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> pv.MultiBlock:
    """Tessellate ``shape`` preserving per-face identity and edges.

    Returns a two-block :class:`pyvista.MultiBlock`::

        MultiBlock
        |-- "faces"  -> MultiBlock(one PolyData per topological face)
        |-- "edges"  -> PolyData (see :func:`topods_to_edges`)

    Each face block carries a ``cad.face_id`` cell array and analytic
    point ``Normals`` (smooth shading on a coarse mesh). A single
    ``BRepMesh`` pass feeds both blocks.

    Parameters
    ----------
    shape : OCP.TopoDS.TopoDS_Shape
        Shape to tessellate. ``build123d`` / ``cadquery`` wrappers are
        unwrapped automatically.
    linear_deflection : float, default: 0.1
        Maximum chordal deviation from the analytic surface.
    angular_deflection : float, default: 0.5
        Maximum angular deviation (radians) between adjacent triangles.

    Returns
    -------
    pyvista.MultiBlock
        ``{'faces': MultiBlock, 'edges': PolyData}``.

    Raises
    ------
    pyvista_cad.TessellationError
        If OCCT meshing fails.

    Warns
    -----
    pyvista_cad.CadWarning
        When the analytic surface normal cannot be evaluated for one or
        more faces, those faces fall back to VTK facet normals and shade
        faceted rather than smooth. A single aggregated warning is
        emitted per call (never one per face).

    Notes
    -----
    Smooth shading relies on per-node analytic surface normals. For a
    face whose analytic surface evaluation fails (degenerate pcurve,
    singular surface point, or a triangulation that carries no UV
    nodes), the block is emitted without a ``Normals`` array and the
    renderer falls back to faceted facet normals for that face. The
    geometry itself is unaffected; only the shading degrades.

    Examples
    --------
    Split a build123d box into per-face and edge blocks (requires the
    ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad import topods_to_multiblock
    >>> mb = topods_to_multiblock(b3d.Box(10, 10, 10).wrapped)
    >>> sorted(mb.keys())
    ['edges', 'faces']
    >>> mb['faces'].n_blocks
    6

    """
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    shape = unwrap_to_topods(shape)
    _ensure_mesh(shape, linear_deflection, angular_deflection)

    faces = pv.MultiBlock()
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    faceted_faces = 0
    while fexp.More():
        face = TopoDS.Face_s(fexp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            fexp.Next()
            continue
        trsf = loc.Transformation()
        n_nodes = tri.NbNodes()

        pts = np.empty((n_nodes, 3), dtype=float)
        for i in range(1, n_nodes + 1):
            node = tri.Node(i)
            node.Transform(trsf)
            pts[i - 1] = (node.X(), node.Y(), node.Z())

        reversed_face = face.Orientation() == 1  # TopAbs_REVERSED
        n_tri = tri.NbTriangles()
        conn = np.empty((n_tri, 4), dtype=np.int64)
        for i in range(1, n_tri + 1):
            i1, i2, i3 = tri.Triangle(i).Get()
            if reversed_face:
                i2, i3 = i3, i2
            conn[i - 1] = (3, i1 - 1, i2 - 1, i3 - 1)

        block = pv.PolyData(pts, conn.ravel() if n_tri else None)
        block.cell_data['cad.face_id'] = np.full(max(n_tri, 0), face_id, dtype=np.int64)
        normals = _face_normals(face, tri, trsf, n_nodes)
        if normals is not None:
            block.point_data['Normals'] = normals
            block.point_data.active_normals_name = 'Normals'
        elif tri.HasUVNodes():
            # The face had UV nodes (so the analytic path was attempted)
            # yet produced no normals: it shades faceted, not smooth.
            faceted_faces += 1
        faces.append(block, name=f'face_{face_id:05d}')
        face_id += 1
        fexp.Next()

    if faceted_faces:
        warnings.warn(
            f'{faceted_faces} face(s) abandoned the analytic-normal path; '
            'they fall back to faceted facet normals and the rendered CAD '
            'view shades coarser than the B-rep supports.',
            CadWarning,
            stacklevel=2,
        )

    edges = topods_to_edges(
        shape,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )

    if faceted_faces == 0 and face_id > 0:
        # Every face carries analytic per-point normals: smooth shading
        # is exact, no crease-splitting needed (and splitting would
        # throw the analytic normals away).
        faces.field_data[_ANALYTIC_NORMALS_FLAG] = np.array([1], dtype=np.int64)

    mb = pv.MultiBlock()
    mb['faces'] = faces
    mb['edges'] = edges
    _register_brep_cache(mb, shape)  # type: ignore[arg-type]
    return mb


def flatten_to_cad_polydata(mb: pv.MultiBlock) -> pv.PolyData:
    """Collapse a :func:`topods_to_multiblock` result to one PolyData.

    Per-face blocks are merged into a single :class:`pyvista.PolyData`
    carrying a ``cad.face_id`` cell array; the edges block is stashed as
    a companion attribute and is recoverable via :func:`get_cad_edges`.

    Parameters
    ----------
    mb : pyvista.MultiBlock
        A ``{'faces', 'edges'}`` block from :func:`topods_to_multiblock`.

    Returns
    -------
    pyvista.PolyData
        Merged faces with ``cad.face_id``; edges available through
        :func:`get_cad_edges`.

    Examples
    --------
    Flatten a per-face block tree to one PolyData (requires the
    ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad import topods_to_multiblock, flatten_to_cad_polydata
    >>> mb = topods_to_multiblock(b3d.Box(10, 10, 10).wrapped)
    >>> poly = flatten_to_cad_polydata(mb)
    >>> type(poly).__name__
    'PolyData'
    >>> 'cad.face_id' in poly.cell_data
    True

    """
    # NOTE: ``name in MultiBlock`` tests block *data*, not names; the
    # block-name list is what we want here.
    names = mb.keys()
    faces = cast('pv.MultiBlock', mb['faces']) if 'faces' in names else mb
    merged = faces.combine(merge_points=False) if len(faces) else pv.PolyData()
    if isinstance(merged, pv.PolyData):
        poly = merged
    else:
        poly = merged.extract_surface(algorithm='dataset_surface')
    edges = mb['edges'] if 'edges' in names else None
    if edges is not None:
        _stash_edges(poly, cast('pv.PolyData', edges))
    cached = get_cached_topods(mb)  # type: ignore[arg-type]
    if cached is not None:
        _register_brep_cache(poly, cached)
    return poly


def _stash_edges(poly: pv.PolyData, edges: pv.PolyData) -> None:
    """Attach an edges PolyData to ``poly`` as a recoverable companion."""
    import contextlib

    with contextlib.suppress(AttributeError, TypeError):
        setattr(poly, _EDGES_ATTR, edges)


def get_cad_edges(poly: pv.PolyData) -> pv.PolyData | None:
    """Return the edges PolyData stashed on ``poly``, if any.

    Parameters
    ----------
    poly : pyvista.PolyData
        A mesh produced by :func:`flatten_to_cad_polydata`.

    Returns
    -------
    pyvista.PolyData or None
        The companion edges mesh, or ``None``.

    Examples
    --------
    Recover the edges companion after flattening (requires the
    ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad import (
    ...     topods_to_multiblock,
    ...     flatten_to_cad_polydata,
    ...     get_cad_edges,
    ... )
    >>> mb = topods_to_multiblock(b3d.Box(10, 10, 10).wrapped)
    >>> poly = flatten_to_cad_polydata(mb)
    >>> edges = get_cad_edges(poly)
    >>> type(edges).__name__
    'PolyData'

    """
    return getattr(poly, _EDGES_ATTR, None)


def _shape_has_faces(shape: Any) -> bool:
    """Return ``True`` if ``shape`` contains at least one ``TopoDS_Face``."""
    try:
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer

        return bool(TopExp_Explorer(shape, TopAbs_FACE).More())
    except (ImportError, RuntimeError, TypeError):
        # No OCP, or ``shape`` is not a TopoDS_Shape: treat as "no
        # trustworthy B-rep faces" so the caller uses the mesh path.
        return False


def _count_faces(shape: Any) -> int:
    """Return the number of ``TopoDS_Face`` in ``shape`` (0 if unknown)."""
    try:
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        n = 0
        while exp.More():
            n += 1
            exp.Next()
    except (ImportError, RuntimeError, TypeError):
        # No OCP, or ``shape`` is not a TopoDS_Shape.
        return 0
    return n


def _feature_edge_fallback(mesh: pv.PolyData, feature_angle: float) -> pv.PolyData:
    """Crease + boundary edges for a mesh with no usable B-rep origin.

    A heavily tessellated surface (e.g. a dense IGES B-spline import)
    produces a feature-edge set so dense it paints the whole model
    black. When the extracted edges rival the mesh's own cell count
    they are tessellation noise, not features — drop the overlay and
    let the shaded faces speak.
    """
    try:
        feat = mesh.extract_feature_edges(
            feature_angle=feature_angle,
            boundary_edges=True,
            feature_edges=True,
            non_manifold_edges=False,
            manifold_edges=False,
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        # VTK rejected the mesh (empty / non-surface input). With no
        # B-rep origin and no extractable feature edges there is no edge
        # overlay at all; the CAD view degrades to a plain shaded mesh.
        warnings.warn(
            f'feature-edge extraction failed ({exc}) and the mesh has no '
            'B-rep origin; the CAD view is rendered without any edge '
            'overlay.',
            CadWarning,
            stacklevel=3,
        )
        return pv.PolyData()
    if mesh.n_cells and feat.n_cells > 0.25 * mesh.n_cells:
        return pv.PolyData()
    return feat


def _faces_have_analytic_normals(face_obj: Any) -> bool:
    """Return ``True`` if ``face_obj`` carries trustworthy analytic normals.

    Only the B-rep tessellation path (:func:`topods_to_multiblock`)
    stamps the :data:`_ANALYTIC_NORMALS_FLAG` field-data marker. A plain
    mesh — even one whose PLY/STL ships baked vertex normals — does not,
    because those normals are smooth-averaged across hard creases and
    melt corners under smooth shading.
    """
    return isinstance(face_obj, pv.MultiBlock) and _ANALYTIC_NORMALS_FLAG in face_obj.field_data


def _resolve_cad(
    obj: Any,
    *,
    linear_deflection: float,
    angular_deflection: float,
    feature_angle: float,
) -> tuple[Any, Any]:
    """Resolve ``obj`` to ``(faces, edges)`` for rendering.

    ``edges`` is a :class:`pyvista.PolyData` (or ``None``); typed
    ``Any`` because PyVista's ``MultiBlock.__getitem__`` stub widens to
    ``MultiBlock | DataSet | None``.

    Resolution order:

    1. ``MultiBlock`` with ``faces`` / ``edges`` keys -> used directly.
    2. generic ``MultiBlock`` (assembly) -> each block resolved
       recursively; faces aggregated, edges merged.
    3. ``PolyData`` with stashed companion edges -> used directly.
    4. ``PolyData`` / wrapper with a cached originating ``TopoDS`` ->
       :func:`topods_to_multiblock`.
    5. raw ``TopoDS`` / ``build123d`` / ``cadquery`` ->
       :func:`topods_to_multiblock`.
    6. plain ``PolyData`` (no B-rep) -> faces as-is, edges via
       :meth:`~pyvista.PolyData.extract_feature_edges` (a documented
       approximation: no topology is available).
    """
    if isinstance(obj, pv.MultiBlock):
        # ``name in MultiBlock`` tests block data; use the name list.
        names = obj.keys()
        if 'faces' in names:
            return obj['faces'], (obj['edges'] if 'edges' in names else None)

        agg_faces = pv.MultiBlock()
        edge_parts: list[pv.PolyData] = []
        all_analytic = True
        n_blocks = 0
        for i, block in enumerate(obj):
            if block is None:
                continue
            bf, be = _resolve_cad(
                block,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                feature_angle=feature_angle,
            )
            agg_faces.append(bf, name=f'block_{i:05d}')
            all_analytic &= _faces_have_analytic_normals(bf)
            n_blocks += 1
            if be is not None and be.n_cells:
                edge_parts.append(be)
        if all_analytic and n_blocks:
            agg_faces.field_data[_ANALYTIC_NORMALS_FLAG] = np.array([1], dtype=np.int64)
        merged_edges = pv.merge(edge_parts) if edge_parts else None
        return agg_faces, merged_edges

    if isinstance(obj, pv.PolyData):
        stashed = get_cad_edges(obj)
        if stashed is not None:
            return obj, stashed
        cached = get_cached_topods(obj)
        # Only trust the cached B-rep when it actually has faces. Some
        # readers (e.g. FCStd) stash a bare sketch/wire — or attach one
        # to an empty placeholder block — as the origin; its edges have
        # nothing to do with the displayed solid and would render as a
        # stray ring far outside the mesh.
        if cached is not None and _shape_has_faces(cached):
            # Keep the *existing* mesh as the face layer: it already
            # carries the caller's scalars / colours and is tessellated.
            # Only the topological edges are recovered from the cached
            # B-rep (no re-tessellation, edges sit on the same facets).
            # For a pristine analytic per-face mesh with smooth normals,
            # call :func:`topods_to_multiblock` directly.
            edges = topods_to_edges(
                cached,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
            )
            # A *faceted* B-rep (e.g. a mesh round-tripped through
            # write_brep) is a triangle soup: every facet is its own
            # planar face, so its "topological" edges are just the
            # triangle wireframe. Detect it by face count, not edge
            # count: a soup has ~one B-rep face per triangle, whereas a
            # genuine B-rep has far fewer faces than triangles even when
            # coarsely tessellated (a Box is 6 faces / 12 triangles, so
            # an edge:triangle ratio test would wrongly flag it). When
            # the face count rivals the triangle count the "topological"
            # edges are tessellation noise — fall back to crease
            # extraction, which yields the same clean outline.
            if obj.n_cells and _count_faces(cached) > 0.5 * obj.n_cells:
                return obj, _feature_edge_fallback(obj, feature_angle)
            return obj, edges
        return obj, _feature_edge_fallback(obj, feature_angle)

    if isinstance(obj, pv.DataSet):
        surf = obj.extract_surface(algorithm='dataset_surface')
        return _resolve_cad(
            surf,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            feature_angle=feature_angle,
        )

    # raw TopoDS / build123d / cadquery
    mb = topods_to_multiblock(
        obj,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    return mb['faces'], mb['edges']


def as_cad_multiblock(
    obj: Any,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    feature_angle: float = 30.0,
) -> pv.MultiBlock:
    """Resolve ``obj`` to a ``{'faces', 'edges'}`` :class:`pyvista.MultiBlock`.

    The same resolution :func:`add_cad` performs, but returned as data
    instead of rendered: ``faces`` is a per-face / per-block MultiBlock
    carrying analytic normals, ``edges`` the topological B-rep curves
    (or feature edges for a mesh with no B-rep origin).

    Parameters
    ----------
    obj : Any
        Anything :func:`add_cad` accepts.
    linear_deflection, angular_deflection : float
        Tessellation tolerances if ``obj`` is not yet meshed.
    feature_angle : float, default: 30.0
        Crease angle for the no-B-rep feature-edge fallback.

    Returns
    -------
    pyvista.MultiBlock
        ``{'faces': MultiBlock | PolyData, 'edges': PolyData}``.

    Examples
    --------
    Resolve a build123d box to faces/edges blocks (requires the
    ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad import as_cad_multiblock
    >>> cad = as_cad_multiblock(b3d.Box(10, 10, 10).wrapped)
    >>> type(cad).__name__
    'MultiBlock'
    >>> sorted(cad.keys())
    ['edges', 'faces']

    """
    faces, edges = _resolve_cad(
        obj,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        feature_angle=feature_angle,
    )
    out = pv.MultiBlock()
    out['faces'] = faces
    out['edges'] = edges if edges is not None else pv.PolyData()
    return out


def add_cad(
    plotter: pv.Plotter,
    obj: Any,
    *,
    color: Any = None,
    edges: bool = True,
    edge_color: Any = None,
    line_width: float = 2.0,
    render_lines_as_tubes: bool = False,
    silhouette: bool = False,
    smooth: bool = True,
    feature_angle: float = 30.0,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    **mesh_kwargs: Any,
) -> dict[str, Any]:
    """Add ``obj`` to ``plotter`` rendered CAD-style.

    Faces are drawn smooth-shaded with their triangle edges hidden;
    topological edges are overlaid as theme-coloured lines with a depth
    bias so they never z-fight the surface. This is the rendering
    primitive behind :meth:`pyvista.Plotter.cad.add` and the ``.cad``
    accessor's ``plot()``.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Target plotter.
    obj : Any
        ``MultiBlock`` / ``PolyData`` / ``DataSet`` / raw ``TopoDS`` /
        ``build123d`` / ``cadquery`` object. A plain mesh with no B-rep
        origin falls back to feature-edge extraction (approximate).
    color : pyvista.ColorLike, optional
        Face colour. Defaults to the active theme colour.
    edges : bool, default: True
        Draw the topological edge overlay.
    edge_color : pyvista.ColorLike, optional
        Edge colour. Defaults to ``pyvista.global_theme.edge_color``.
    line_width : float, default: 2.0
        Edge line width.
    render_lines_as_tubes : bool, default: False
        Render edges as tubes (nicer at thick widths).
    silhouette : bool, default: False
        Add a camera-tracking silhouette outline of the faces.
    smooth : bool, default: True
        Use analytic normals for smooth (non-faceted) shading.
    feature_angle : float, default: 30.0
        Crease angle for the no-B-rep feature-edge fallback.
    linear_deflection, angular_deflection : float
        Tessellation tolerances if ``obj`` is not yet meshed.
    **mesh_kwargs
        Forwarded to :meth:`~pyvista.Plotter.add_mesh` for the faces.

    Returns
    -------
    dict
        ``{'faces': <Actor>, 'edges': <Actor or None>}``.

    Examples
    --------
    Add a build123d box to a plotter CAD-style (requires the
    ``[step]`` extra):

    >>> import pyvista as pv
    >>> pv.OFF_SCREEN = True
    >>> import build123d as b3d
    >>> from pyvista_cad import add_cad
    >>> pl = pv.Plotter()
    >>> actors = add_cad(pl, b3d.Box(10, 10, 10).wrapped)
    >>> sorted(actors.keys())
    ['edges', 'faces']
    >>> pl.close()

    """
    face_obj, edge_obj = _resolve_cad(
        obj,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        feature_angle=feature_angle,
    )

    # If the caller drives face colour by data (scalars / rgb /
    # multi_colors), do not inject a default solid colour — it would
    # override the mapping in ``add_mesh``.
    scalar_driven = any(k in mesh_kwargs for k in ('scalars', 'rgb', 'rgba', 'multi_colors'))
    if color is None and not scalar_driven:
        color = pv.global_theme.color
    if edge_color is None:
        edge_color = pv.global_theme.edge_color

    # Smooth shading is only correct when the faces carry analytic
    # per-point normals (the B-rep path). For a plain mesh the stored /
    # VTK-computed normals are averaged across hard creases, so smooth
    # shading melts every sharp corner into a rounded blob. Splitting
    # vertices at sharp edges keeps each facet region's own normal and
    # restores the crisp CAD silhouette without adding any geometry.
    analytic_normals = _faces_have_analytic_normals(face_obj)
    split_sharp_edges = smooth and not analytic_normals

    face_kwargs: dict[str, Any] = {
        'show_edges': False,
        'smooth_shading': smooth,
        'split_sharp_edges': split_sharp_edges,
        'silhouette': silhouette,
        **mesh_kwargs,
    }
    if color is not None:
        face_kwargs['color'] = color
    face_actor = plotter.add_mesh(face_obj, **face_kwargs)

    edge_actor = None
    if edges and edge_obj is not None and edge_obj.n_cells:
        edge_actor = plotter.add_mesh(
            edge_obj,
            color=edge_color,
            line_width=line_width,
            render_lines_as_tubes=render_lines_as_tubes,
            pickable=False,
            lighting=False,
        )
        # Pull edges slightly toward the camera so they never z-fight
        # the coincident facets they were sampled from.
        mapper = getattr(edge_actor, 'mapper', None)
        if mapper is not None:
            try:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1.0, -1.0)
            except (AttributeError, TypeError):
                # Older VTK mapper without the relative-offset API: edges
                # may z-fight slightly. Cosmetic only, no data loss, so
                # no warning is warranted.
                pass

    return {'faces': face_actor, 'edges': edge_actor}


def plot_cad(obj: Any, **kwargs: Any) -> Any:
    """One-shot CAD-friendly plot of ``obj`` (mirrors ``mesh.plot()``).

    Builds a :class:`pyvista.Plotter`, calls :func:`add_cad`, and shows
    it. ``show``-level keywords (``screenshot``, ``cpos``,
    ``off_screen``, ``window_size``, ``jupyter_backend``, ``return_cpos``)
    are split out and forwarded to :meth:`pyvista.Plotter.show`;
    everything else goes to :func:`add_cad`.

    Parameters
    ----------
    obj : Any
        Anything :func:`add_cad` accepts.
    **kwargs
        Split between :func:`add_cad` and
        :meth:`pyvista.Plotter.show`.

    Returns
    -------
    Any
        Whatever :meth:`pyvista.Plotter.show` returns.

    Examples
    --------
    One-shot CAD plot of a build123d box (requires the ``[step]``
    extra):

    >>> import pyvista as pv
    >>> pv.OFF_SCREEN = True
    >>> import build123d as b3d
    >>> from pyvista_cad import plot_cad
    >>> cpos = plot_cad(b3d.Box(10, 10, 10).wrapped, return_cpos=True)
    >>> type(cpos).__name__
    'CameraPosition'

    """
    show_keys = {
        'screenshot',
        'cpos',
        'off_screen',
        'window_size',
        'jupyter_backend',
        'return_cpos',
        'full_screen',
        'interactive',
        'auto_close',
        'title',
    }
    show_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in show_keys}
    off_screen = show_kwargs.pop('off_screen', None)
    plotter = pv.Plotter(off_screen=off_screen)
    add_cad(plotter, obj, **kwargs)
    return plotter.show(**show_kwargs)


def register_cad_plotter_component() -> None:
    """Register the ``plotter.cad`` component (idempotent).

    Called from :mod:`pyvista_cad` import and resolvable lazily via the
    ``pyvista.plotter_components`` entry point.
    """
    # ``register_plotter_component`` / ``registered_plotter_components``
    # are public since pyvista 0.48 (the dependency floor), so no
    # missing-attribute fallback is needed.
    register = pv.register_plotter_component
    registered = pv.registered_plotter_components
    try:
        already = any(getattr(c, 'name', None) == 'cad' for c in registered())
    except (TypeError, AttributeError):
        # ``registered_plotter_components`` shape varies across pyvista
        # versions; if it cannot be probed, fall through and let the
        # decorator's own idempotency guard handle a double-register.
        already = False
    if already:
        return

    @register('cad')
    class CadPlotterComponent:
        """The ``plotter.cad`` component: CAD-friendly rendering."""

        def __init__(self, plotter: pv.Plotter) -> None:
            self._plotter = plotter

        def add(self, obj: Any, **kwargs: Any) -> dict[str, Any]:
            """Add ``obj`` CAD-style. See :func:`add_cad`.

            Parameters
            ----------
            obj : Any
                Anything :func:`add_cad` accepts.
            **kwargs
                Forwarded to :func:`add_cad`.

            Returns
            -------
            dict
                ``{'faces': <Actor>, 'edges': <Actor or None>}``.

            """
            return add_cad(self._plotter, obj, **kwargs)


# Some pyvista builds expose the decorator directly; register eagerly so
# ``plotter.cad`` works after ``import pyvista_cad`` without relying on
# the entry-point group being discovered.
register_cad_plotter_component()
