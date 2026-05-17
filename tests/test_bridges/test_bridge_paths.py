"""Edge paths in the build123d and gmsh bridges with real objects.

Covers the colour/label propagation, the single-child and childless
``Compound`` branches, the ``_color_rgb`` coercion fallbacks, and the
gmsh empty-model / unknown-element / ``UnstructuredGrid`` paths.
"""

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._bridges.build123d import _color_rgb

b3d = pytest.importorskip('build123d')
gmsh = pytest.importorskip('gmsh')


def test_color_rgb_none_object():
    assert _color_rgb(object()) is None


def test_color_rgb_from_rgb_attributes():
    class C:
        r, g, b = 0.1, 0.2, 0.3

    class Obj:
        color = C()

    assert _color_rgb(Obj()) == pytest.approx((0.1, 0.2, 0.3))


def test_color_rgb_from_sequence():
    class Obj:
        color = (0.4, 0.5, 0.6, 1.0)

    assert _color_rgb(Obj()) == pytest.approx((0.4, 0.5, 0.6, 1.0))


def test_color_rgb_from_rgb_sequence_no_alpha():
    class Obj:
        color = (0.4, 0.5, 0.6)

    assert _color_rgb(Obj()) == pytest.approx((0.4, 0.5, 0.6))


def test_color_rgb_short_sequence_returns_none():
    class Obj:
        color = (0.4, 0.5)

    assert _color_rgb(Obj()) is None


def test_color_rgb_real_build123d_color():
    """A real ``build123d.Color`` is iterable-only and round-trips as
    RGBA in ``[0, 1]`` through the iterable branch."""

    class Obj:
        color = b3d.Color(0.2, 0.4, 0.6, 0.8)

    assert _color_rgb(Obj()) == pytest.approx((0.2, 0.4, 0.6, 0.8), abs=1e-6)


def test_color_rgb_uncoercible_returns_none():
    class Obj:
        color = 'not-a-color'

    assert _color_rgb(Obj()) is None


def test_from_build123d_propagates_label():
    box = b3d.Box(5, 5, 5)
    box.label = 'widget'
    poly = pyvista_cad.from_build123d(box)
    assert isinstance(poly, pv.PolyData)
    assert str(poly.field_data['cad.label'][0]) == 'widget'


def test_from_build123d_real_color_propagates_rgba():
    """A real ``build123d.Color`` (iterable-only, no ``.r/.g/.b`` and
    not subscriptable) propagates as RGBA onto ``cad.color``."""
    box = b3d.Box(5, 5, 5)
    box.color = b3d.Color(0.2, 0.4, 0.6, 0.8)
    poly = pyvista_cad.from_build123d(box)
    assert isinstance(poly, pv.PolyData)
    assert np.allclose(poly.field_data['cad.color'], [0.2, 0.4, 0.6, 0.8], atol=1e-6)


def test_from_build123d_compound_colored_children_propagate_rgba():
    """The multi-child ``Compound`` branch writes per-child RGBA from
    real ``build123d.Color`` objects."""
    box = b3d.Box(2, 2, 2)
    box.label = 'red'
    box.color = b3d.Color(0.9, 0.1, 0.1, 1.0)
    cyl = b3d.Cylinder(0.5, 1)
    cyl.label = 'blue'
    cyl.color = b3d.Color(0.1, 0.1, 0.9, 0.5)
    compound = b3d.Compound(children=[box, cyl])
    out = pyvista_cad.from_build123d(compound)
    assert isinstance(out, pv.MultiBlock)
    assert np.allclose(out['red'].field_data['cad.color'], [0.9, 0.1, 0.1, 1.0], atol=1e-6)
    assert np.allclose(out['blue'].field_data['cad.color'], [0.1, 0.1, 0.9, 0.5], atol=1e-6)


def test_from_build123d_color_rgb_attr_propagates():
    """A shape whose ``.color`` exposes ``.r/.g/.b`` (the contract
    ``_color_rgb`` is written against) propagates onto ``cad.color``.
    A real ``build123d.Color`` no longer exposes these, so the bridge is
    driven through an OCCT-backed wrapper carrying such a colour."""

    class RGBColor:
        r, g, b = 0.1, 0.2, 0.3

    class Wrapper:
        wrapped = b3d.Box(3, 3, 3).wrapped
        color = RGBColor()
        label = 'painted'

    poly = pyvista_cad.from_build123d(Wrapper())
    assert isinstance(poly, pv.PolyData)
    assert np.allclose(poly.field_data['cad.color'], [0.1, 0.2, 0.3], atol=1e-6)
    assert str(poly.field_data['cad.label'][0]) == 'painted'


def test_from_build123d_pass_data_false_omits_metadata():
    box = b3d.Box(5, 5, 5)
    box.label = 'widget'
    poly = pyvista_cad.from_build123d(box, pass_data=False)
    assert 'cad.label' not in poly.field_data
    assert 'cad.color' not in poly.field_data


def test_from_build123d_compound_single_labeled_child():
    box = b3d.Box(2, 2, 2)
    box.label = 'solo'
    compound = b3d.Compound(children=[box])
    out = pyvista_cad.from_build123d(compound)
    assert isinstance(out, pv.MultiBlock)
    assert out.n_blocks == 1
    block = out['solo']
    assert block.n_cells == 12
    assert str(block.field_data['cad.source_format'][0]) == 'build123d'


def test_from_build123d_compound_no_children_is_polydata():
    """A Compound built straight from a solid (no .children) takes the
    single-shape branch and returns PolyData, not a MultiBlock."""
    compound = b3d.Compound(b3d.Box(3, 3, 3).wrapped)
    out = pyvista_cad.from_build123d(compound)
    assert isinstance(out, pv.PolyData)
    assert out.n_cells == 12


def test_from_build123d_compound_unlabeled_child_gets_index_name():
    box = b3d.Box(1, 1, 1)
    cyl = b3d.Cylinder(0.5, 1)
    compound = b3d.Compound(children=[box, cyl])
    out = pyvista_cad.from_build123d(compound)
    assert isinstance(out, pv.MultiBlock)
    assert out.n_blocks == 2
    assert any(name.startswith('part_') for name in list(out.keys()))


def test_color_of_preserves_rgba_through_step_read(tmp_path):
    """A colored + transparent build123d part keeps all four RGBA
    components when read back through ``read_step`` (the STEP reader's
    ``_color_of``, distinct from the bridge's ``_color_rgb``)."""
    rgba = (0.2, 0.4, 0.6, 0.35)
    box = b3d.Box(8, 8, 8)
    box.label = 'TintedBox'
    box.color = b3d.Color(*rgba)
    asm = b3d.Compound(children=[box])

    src = tmp_path / 'tinted.step'
    b3d.export_step(asm, str(src))

    out = pyvista_cad.read_step(src)
    assert isinstance(out, pv.MultiBlock)

    leaves: list[pv.PolyData] = []

    def _collect(node: object) -> None:
        if isinstance(node, pv.MultiBlock):
            for child in node:
                _collect(child)
        elif isinstance(node, pv.PolyData):
            leaves.append(node)

    _collect(out)
    colored = [
        np.asarray(leaf.field_data['cad.color'], dtype=float)
        for leaf in leaves
        if 'cad.color' in leaf.field_data
    ]
    assert len(colored) == 1
    color = colored[0]
    assert color.size == 4
    np.testing.assert_allclose(color, rgba, atol=1e-6)


def test_color_of_rgb_only_stays_three_components():
    """A legacy ``.r/.g/.b``-only color (no alpha) yields a 3-tuple, not
    a spuriously padded RGBA."""
    from pyvista_cad._backends._build123d import _color_of  # noqa: PLC0415

    class RGBColor:
        r, g, b = 0.1, 0.2, 0.3

    class Obj:
        color = RGBColor()

    result = _color_of(Obj())
    assert result == pytest.approx((0.1, 0.2, 0.3))
    assert len(result) == 3


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    yield
    gmsh.finalize()


def test_from_gmsh_empty_model_returns_empty_grid(gmsh_session):
    gmsh.model.add('empty')
    gmsh.model.setCurrent('empty')
    grid = pyvista_cad.from_gmsh('empty')
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells == 0


def test_from_gmsh_nodes_only_returns_point_cloud(gmsh_session):
    """A model with nodes but no recognised elements yields an
    UnstructuredGrid carrying the coordinates and zero cells."""
    gmsh.model.add('nodes_only')
    gmsh.model.setCurrent('nodes_only')
    tag = gmsh.model.addDiscreteEntity(0)
    gmsh.model.mesh.addNodes(0, tag, [1, 2, 3], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    grid = pyvista_cad.from_gmsh('nodes_only')
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells == 0
    assert grid.n_points == 3


def test_to_gmsh_then_from_gmsh_named_model(gmsh_session):
    src = pv.Sphere(theta_resolution=8, phi_resolution=8)
    pyvista_cad.to_gmsh(src, model_name='named')
    grid = pyvista_cad.from_gmsh('named')
    assert grid.n_cells > 0
    assert str(grid.field_data['cad.source_format'][0]) == 'gmsh'


def test_to_gmsh_from_unstructured_grid(gmsh_session):
    """``to_gmsh`` with an UnstructuredGrid exercises the
    ``_iter_triangles`` extract-surface branch."""
    grid = pv.Sphere(theta_resolution=8, phi_resolution=8).cast_to_unstructured_grid()
    pyvista_cad.to_gmsh(grid, model_name='ug')
    out = pyvista_cad.from_gmsh('ug')
    assert out.n_cells > 0


def test_from_gmsh_skips_higher_order_elements(gmsh_session):
    """A 2nd-order mesh emits only element types absent from the
    gmsh->VTK table (8/9/11), so every element is skipped and the grid
    falls through the no-cells branch carrying just the node cloud."""
    gmsh.model.add('o2')
    gmsh.model.setCurrent('o2')
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)
    grid = pyvista_cad.from_gmsh('o2')
    assert isinstance(grid, pv.UnstructuredGrid)
    # 2nd-order surface/volume elements (gmsh types 9 / 11) are absent
    # from the gmsh->VTK table and skipped; the node cloud is far larger
    # than the handful of low-order edge cells that survive.
    assert grid.n_points > 100 * max(grid.n_cells, 1)


def test_to_gmsh_polydata_with_quads_triangulates(gmsh_session):
    """A quad-faced PolyData drives the ``_iter_triangles`` PolyData
    branch: quads are triangulated before export."""
    plane = pv.Plane(i_resolution=3, j_resolution=3)
    assert plane.faces.reshape(-1, 5)[0, 0] == 4  # quads
    pyvista_cad.to_gmsh(plane, model_name='quads')
    out = pyvista_cad.from_gmsh('quads')
    assert out.n_cells > 0


def test_from_gmsh_skips_unknown_element_type(gmsh_session):
    """A model with only line elements (which map to a known type) plus a
    discrete surface confirms the type lookup path, and a tet mesh round
    trips through the volume-element branch."""
    gmsh.model.add('tet')
    gmsh.model.setCurrent('tet')
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    grid = pyvista_cad.from_gmsh('tet')
    assert grid.n_cells > 0
    # Tetrahedra are VTK cell type 10.
    assert 10 in set(grid.celltypes.tolist())
