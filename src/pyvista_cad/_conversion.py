"""OCCT TopoDS_Shape <-> PyVista conversion workhorse.

All OCP imports happen lazily inside the call. ``import pyvista_cad``
must not load OCP. The build123d and cascadio backends both delegate
to the helpers in this module.
"""

import contextlib
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_cad._errors import TessellationError

# Attribute name used to stash the originating ``TopoDS_Shape`` on a
# ``PolyData`` instance. Stashing the shape on the object itself avoids
# the ``id()``-recycle hazard of a module-level ``dict[int, Any]`` cache:
# when a short-lived mesh is GC'd and a new mesh happens to land at the
# same address, the new mesh starts fresh (no stale hit) because the
# attribute does not exist on it.
_BREP_ATTR = '_cad_topods_shape'


def _register_brep_cache(mesh: pv.PolyData, shape: Any) -> None:
    """Stash ``shape`` on ``mesh`` so :func:`get_cached_topods` can recover it."""
    # PolyData currently accepts arbitrary attributes; the suppress guards
    # against a future pyvista release locking down the namespace.
    with contextlib.suppress(AttributeError, TypeError):
        setattr(mesh, _BREP_ATTR, shape)


def get_cached_topods(mesh: pv.PolyData) -> Any | None:
    """Return the cached originating ``TopoDS_Shape`` for ``mesh``, if any.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh previously produced by :func:`topods_to_polydata`.

    Returns
    -------
    OCP.TopoDS.TopoDS_Shape or None
        The original shape used to build ``mesh``, or ``None`` if no
        cache entry exists.

    """
    return getattr(mesh, _BREP_ATTR, None)


def topods_to_polydata(
    shape: Any,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> pv.PolyData:
    """Tessellate an OCCT ``TopoDS_Shape`` to a :class:`pyvista.PolyData`.

    The originating ``shape`` is stashed in a module-level cache keyed
    on ``id(result)`` so that downstream consumers (notably
    :meth:`CadShape.tessellate`) can skip re-tessellation on round-trip
    reads.

    Parameters
    ----------
    shape : OCP.TopoDS.TopoDS_Shape
        Shape to tessellate.
    linear_deflection : float, default: 0.1
        Maximum chordal deviation of the mesh from the analytical
        surface.
    angular_deflection : float, default: 0.5
        Maximum angular deviation in radians between adjacent
        triangles.

    Returns
    -------
    pyvista.PolyData
        Triangulated representation of ``shape``.

    Raises
    ------
    pyvista_cad.TessellationError
        If OCCT cannot tessellate the shape.

    Examples
    --------
    Tessellate a build123d box shape (requires the ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad._conversion import topods_to_polydata
    >>> mesh = topods_to_polydata(b3d.Box(10, 10, 10).wrapped)
    >>> type(mesh).__name__
    'PolyData'
    >>> [round(b) for b in mesh.bounds]
    [-5, 5, -5, 5, -5, 5]

    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    try:
        BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    except Exception as exc:
        msg = f'OCCT tessellation failed: {exc}'
        raise TessellationError(msg) from exc

    points: list[tuple[float, float, float]] = []
    faces_idx: list[int] = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            explorer.Next()
            continue

        trsf = location.Transformation()
        n_nodes = triangulation.NbNodes()
        offset = len(points)
        for i in range(1, n_nodes + 1):
            node = triangulation.Node(i)
            node.Transform(trsf)
            points.append((node.X(), node.Y(), node.Z()))

        reversed_face = face.Orientation() == 1  # TopAbs_REVERSED == 1
        n_tri = triangulation.NbTriangles()
        for i in range(1, n_tri + 1):
            tri = triangulation.Triangle(i)
            i1, i2, i3 = tri.Get()
            if reversed_face:
                i2, i3 = i3, i2
            faces_idx.extend([3, offset + i1 - 1, offset + i2 - 1, offset + i3 - 1])

        explorer.Next()

    poly = pv.PolyData()
    if points:
        poly.points = np.asarray(points, dtype=float)
        poly.faces = np.asarray(faces_idx, dtype=np.int64)

    _register_brep_cache(poly, shape)
    return poly


def unwrap_to_topods(shape: Any) -> Any:
    """Coerce a build123d / cadquery / raw OCP object into a ``TopoDS_Shape``.

    Parameters
    ----------
    shape : Any
        A ``build123d.Shape`` / ``Compound``, ``cadquery.Workplane`` /
        ``Compound``, or raw ``OCP.TopoDS.TopoDS_Shape``.

    Returns
    -------
    OCP.TopoDS.TopoDS_Shape
        The unwrapped OCCT shape.

    Examples
    --------
    Unwrap a build123d box to a raw OCCT shape (requires the
    ``[step]`` extra):

    >>> import build123d as b3d
    >>> from pyvista_cad._conversion import unwrap_to_topods
    >>> shape = unwrap_to_topods(b3d.Box(10, 10, 10))
    >>> type(shape).__name__
    'TopoDS_Compound'

    """
    if hasattr(shape, 'wrapped') and shape.wrapped is not None:
        return shape.wrapped
    if hasattr(shape, 'val'):
        val = shape.val()
        if hasattr(val, 'wrapped'):
            return val.wrapped
        return val
    return shape


def polydata_to_topods(mesh: pv.PolyData) -> Any:
    """Build a faceted OCCT shape from a :class:`pyvista.PolyData`.

    Each input triangle becomes a planar ``TopoDS_Face``. The faces are
    fed through :class:`BRepBuilderAPI_Sewing` so that shared edges are
    merged into a single :class:`TopoDS_Shell`. When that sewn shell is
    closed (the input mesh is watertight) it is promoted to a
    :class:`TopoDS_Solid`, so downstream STEP/BREP writers serialize one
    solid body rather than thousands of disjoint single-triangle faces.
    Open or non-manifold input yields the sewn shell (or the raw sewn
    compound when sewing cannot merge every face).

    Chordal-deviation bound: the returned shape reproduces the input
    triangulation vertex-for-vertex with one exception. Faces are sewn
    with a tolerance of ``max(diag * 1e-6, 1e-9)`` where ``diag`` is the
    bounding-box diagonal of ``mesh``, so two distinct input vertices
    closer together than ``diag * 1e-6`` may be welded into one by
    :class:`BRepBuilderAPI_Sewing`. Vertex pairs separated by more than
    ``diag * 1e-6`` are preserved exactly. Apart from this welding, the
    conversion is vertex-exact: its deviation from the original analytic
    surface stays bounded by the ``linear_deflection`` tessellation
    tolerance used to produce ``mesh``.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Triangle / quad surface mesh. Non-triangular faces are
        triangulated first.

    Returns
    -------
    OCP.TopoDS.TopoDS_Shape
        A sewn :class:`TopoDS_Shell`, promoted to a
        :class:`TopoDS_Solid` when the input is closed/watertight;
        otherwise the raw sewn compound of planar faces. The result is
        *faceted*: analytical surface information is not present.

    Raises
    ------
    pyvista_cad.TessellationError
        If ``mesh`` has no triangular faces (empty, lines-only, or
        otherwise face-less), if OCCT sewing produces a null shape from
        degenerate geometry, or if OCCT face construction or sewing
        fails.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista_cad._conversion import polydata_to_topods
    >>> shape = polydata_to_topods(pv.Box())
    >>> type(shape).__name__
    'TopoDS_Solid'

    """
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS, TopoDS_Shell

    tri = mesh.triangulate()
    if not isinstance(tri, pv.PolyData):
        msg = 'polydata_to_topods requires a PolyData (got non-surface mesh)'
        raise TessellationError(msg)

    faces = tri.faces.reshape(-1, 4) if tri.n_cells else np.zeros((0, 4), dtype=np.int64)
    points = np.asarray(tri.points, dtype=float)

    # Up-front guard: a face-less / lines-only / empty mesh has no
    # triangle connectivity. Building no faces leaves the sewing empty,
    # ``sewing.SewedShape()`` returns a null shape, and the downstream
    # ``TopoDS.Shell_s`` cast then dereferences null in C++ (process
    # SIGSEGV, exit 139) which no Python ``except`` can catch. Reject it
    # here with a clear, catchable error.
    if faces.size == 0 or not np.any(faces[:, 0] == 3):
        msg = 'cannot convert a mesh with no triangular faces to a TopoDS shape'
        raise TessellationError(msg)

    # Scale the sewing tolerance to the mesh diagonal so that meshes
    # expressed in microns and meshes expressed in metres both sew
    # cleanly. A fixed ``1e-6`` is fine in metres but is larger than
    # the entire model in microns (degenerate sew) and far smaller than
    # numerical noise in kilometres (no sewing). ``1e-6`` of the diagonal
    # is a good middle ground; floor at ``1e-9`` to avoid division-by-
    # noise on degenerate inputs.
    bounds = mesh.bounds
    diag = float(
        np.linalg.norm(
            np.asarray([bounds.x_max, bounds.y_max, bounds.z_max], dtype=float)
            - np.asarray([bounds.x_min, bounds.y_min, bounds.z_min], dtype=float)
        )
    )
    tol = max(diag * 1e-6, 1e-9)
    sewing = BRepBuilderAPI_Sewing(tol)

    try:
        # ``tri.triangulate()`` guarantees every row is a triangle, and
        # the up-front guard already rejected a face-less mesh, so the
        # connectivity column is uniformly 3 here.
        for row in faces:
            ia, ib, ic = int(row[1]), int(row[2]), int(row[3])
            a = gp_Pnt(float(points[ia, 0]), float(points[ia, 1]), float(points[ia, 2]))
            b = gp_Pnt(float(points[ib, 0]), float(points[ib, 1]), float(points[ib, 2]))
            c = gp_Pnt(float(points[ic, 0]), float(points[ic, 1]), float(points[ic, 2]))
            poly = BRepBuilderAPI_MakePolygon(a, b, c, True)
            wire = poly.Wire()
            face = BRepBuilderAPI_MakeFace(wire, True).Face()
            sewing.Add(face)
        sewing.Perform()
        sewed = sewing.SewedShape()
    except Exception as exc:
        msg = f'OCCT face construction failed: {exc}'
        raise TessellationError(msg) from exc

    # A degenerate input (e.g. a single NaN or collinear triangle) can
    # build zero valid faces and leave the sewing empty, so
    # ``SewedShape()`` returns a null shape. Guard before any
    # ``TopoDS.Shell_s`` / ``Solid_s`` cast: casting a null shape
    # dereferences null in C++ (uncatchable SIGSEGV).
    if sewed.IsNull():
        msg = 'OCCT sewing produced a null shape (degenerate mesh geometry)'
        raise TessellationError(msg)

    # If the sewed result is a closed shell, promote it to a solid so
    # writers serialize a single body rather than a free-form shell.
    try:
        shell = TopoDS.Shell_s(sewed)
    except Exception:
        shell = None
    if isinstance(shell, TopoDS_Shell) and shell.Closed():
        # pragma reason: OCCT BRepBuilderAPI_MakeSolid does not raise on
        # a shell that shell.Closed() already validated True; unreachable
        # from a valid pv.PolyData without mocking OCCT.
        try:
            return BRepBuilderAPI_MakeSolid(shell).Solid()
        except Exception:  # pragma: no cover
            return sewed
    return sewed
