"""Bridges to and from gmsh.

gmsh is a global-state mesh generator, so these helpers consult the
currently-initialized gmsh model. The caller is responsible for
``gmsh.initialize()`` / ``gmsh.finalize()`` bracketing.
"""

from typing import Any

import numpy as np
import pyvista as pv

from pyvista_cad._errors import OptionalDependencyError

# gmsh element-type ID -> (vtk cell type, n nodes)
_GMSH_TO_VTK: dict[int, tuple[int, int]] = {
    1: (3, 2),  # 2-node line
    2: (5, 3),  # 3-node triangle
    3: (9, 4),  # 4-node quad
    4: (10, 4),  # 4-node tet
    5: (12, 8),  # 8-node hex
    6: (13, 6),  # 6-node wedge
    7: (14, 5),  # 5-node pyramid
    15: (1, 1),  # 1-node point
}


def from_gmsh(model_name: str | None = None) -> pv.UnstructuredGrid:
    """Pull the current gmsh model into a :class:`pyvista.UnstructuredGrid`.

    Parameters
    ----------
    model_name : str, optional
        Name of the gmsh model to read. When omitted, the current
        model is used.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh from gmsh, with one cell per gmsh element. Element types
        outside :data:`_GMSH_TO_VTK` are skipped with a warning.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If :mod:`gmsh` is not installed.

    Examples
    --------
    Mesh a unit square in gmsh and pull it into PyVista (requires the
    ``[gmsh]`` extra):

    >>> import gmsh
    >>> from pyvista_cad import from_gmsh
    >>> gmsh.initialize()
    >>> gmsh.option.setNumber('General.Terminal', 0)
    >>> _ = gmsh.model.add('demo')
    >>> _ = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    >>> gmsh.model.occ.synchronize()
    >>> gmsh.model.mesh.generate(2)
    >>> grid = from_gmsh()
    >>> type(grid).__name__
    'UnstructuredGrid'
    >>> grid.n_cells > 0
    True
    >>> gmsh.finalize()

    """
    try:
        import gmsh
    except ImportError as exc:
        msg = 'gmsh not installed; install pyvista-cad[gmsh]'
        raise OptionalDependencyError(msg) from exc

    if model_name is not None:
        gmsh.model.setCurrent(model_name)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    cells: list[int] = []
    cell_types: list[int] = []
    elem_types, _elem_tags, elem_nodes = gmsh.model.mesh.getElements()
    for etype, nodes in zip(elem_types, elem_nodes, strict=True):
        if int(etype) not in _GMSH_TO_VTK:
            continue
        vtk_type, n_nodes = _GMSH_TO_VTK[int(etype)]
        flat = np.asarray(nodes, dtype=np.int64).reshape(-1, n_nodes)
        for row in flat:
            cells.append(n_nodes)
            cells.extend(tag_to_idx[int(t)] for t in row)
            cell_types.append(vtk_type)

    if not cells:
        grid = pv.UnstructuredGrid()
        if coords.size:
            grid.points = coords
        return grid

    cells_arr = np.asarray(cells, dtype=np.int64)
    types_arr = np.asarray(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells_arr, types_arr, coords)
    grid.field_data['cad.source_format'] = np.array(['gmsh'])
    return grid


def to_gmsh(mesh: pv.UnstructuredGrid | pv.PolyData, *, model_name: str = 'pyvista_cad') -> None:
    """Install a PyVista mesh as the current gmsh model (in place).

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid or pyvista.PolyData
        Mesh to register.
    model_name : str, default: ``'pyvista_cad'``
        Name of the gmsh model to create.

    Raises
    ------
    pyvista_cad.OptionalDependencyError
        If :mod:`gmsh` is not installed.

    Examples
    --------
    Register a triangulated PyVista mesh as a gmsh model (requires the
    ``[gmsh]`` extra):

    >>> import gmsh
    >>> import pyvista as pv
    >>> from pyvista_cad import to_gmsh
    >>> gmsh.initialize()
    >>> gmsh.option.setNumber('General.Terminal', 0)
    >>> to_gmsh(pv.Sphere())
    >>> gmsh.model.getCurrent()
    'pyvista_cad'
    >>> gmsh.finalize()

    """
    try:
        import gmsh
    except ImportError as exc:
        msg = 'gmsh not installed; install pyvista-cad[gmsh]'
        raise OptionalDependencyError(msg) from exc

    gmsh.model.add(model_name)
    gmsh.model.setCurrent(model_name)
    surface_tag = gmsh.model.addDiscreteEntity(2)

    points = np.asarray(mesh.points, dtype=float).reshape(-1)
    node_tags = list(range(1, len(mesh.points) + 1))
    gmsh.model.mesh.addNodes(2, surface_tag, node_tags, points.tolist())

    triangles: list[int] = []
    cells = _iter_triangles(mesh)
    for tri in cells:
        triangles.extend(int(i) + 1 for i in tri)
    if triangles:
        elem_tags = list(range(1, 1 + len(triangles) // 3))
        gmsh.model.mesh.addElementsByType(surface_tag, 2, elem_tags, triangles)


def _iter_triangles(mesh: pv.UnstructuredGrid | pv.PolyData) -> Any:
    """Yield triangle index triplets from a surface mesh."""
    if isinstance(mesh, pv.PolyData):
        tri = mesh.triangulate()
        faces = tri.faces.reshape(-1, 4)
        for row in faces:
            if int(row[0]) == 3:
                yield row[1:4]
        return
    surf = mesh.extract_surface(algorithm='dataset_surface').triangulate()
    faces = surf.faces.reshape(-1, 4)
    for row in faces:
        if int(row[0]) == 3:
            yield row[1:4]
