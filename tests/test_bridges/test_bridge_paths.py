"""Edge paths in the build123d bridge with real objects.

Covers the colour/label propagation, the single-child and childless
``Compound`` branches, and the ``_color_rgb`` coercion fallbacks.
"""

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._bridges.build123d import _color_rgb

b3d = pytest.importorskip('build123d', exc_type=ImportError)


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
    """A colored + transparent build123d part keeps its RGB (and, on
    build123d that supports it, alpha) when read back through
    ``read_step`` (the STEP reader's ``_color_of``, distinct from the
    bridge's ``_color_rgb``).

    ``export_step`` writes the STEP ``TRANSPARENCY`` entity, but
    build123d >= 0.11 ``import_step`` no longer reads it back, so the
    recovered alpha is 1.0 there (upstream regression, tracked
    separately). We therefore assert the RGB channels round-trip exactly
    and accept either the authored alpha or an opaque 1.0.
    """
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
    np.testing.assert_allclose(color[:3], rgba[:3], atol=1e-6)
    # Authored alpha (build123d < 0.11) or opaque fallback (>= 0.11).
    assert color[3] == pytest.approx(rgba[3]) or color[3] == pytest.approx(1.0)


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
