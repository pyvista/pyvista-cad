"""cascadio STEP backend: end-to-end read plus the flattening helper.

``_flatten_to_multiblock`` is a pure transform and is exercised directly
(no cascadio needed). The end-to-end ``read_step(..., backend='cascadio')``
path requires the optional ``cascadio`` dependency and skips without it.
"""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._backends._cascadio import _flatten_to_multiblock


def test_flatten_to_multiblock_flattens_nested_paths() -> None:
    """A nested MultiBlock collapses to one flat level with ``parent/child``
    path names; named and unnamed blocks both resolve."""
    inner = pv.MultiBlock()
    inner.append(pv.Sphere(), name='leaf_a')
    inner.append(pv.Cube(), name='leaf_b')

    outer = pv.MultiBlock()
    outer.append(inner, name='group')
    outer.append(pv.Cone(), name='solo')

    flat = _flatten_to_multiblock(outer)
    assert isinstance(flat, pv.MultiBlock)
    names = list(flat.keys())
    assert names == ['group/leaf_a', 'group/leaf_b', 'solo']
    assert all(isinstance(b, pv.PolyData) for b in flat)


def test_flatten_to_multiblock_wraps_bare_polydata() -> None:
    """A bare PolyData becomes a one-block MultiBlock named ``part_0``."""
    flat = _flatten_to_multiblock(pv.Sphere())
    assert isinstance(flat, pv.MultiBlock)
    assert list(flat.keys()) == ['part_0']


def test_read_step_cascadio_backend_e2e() -> None:
    """``read_step(backend='cascadio')`` reads a real STEP fixture to a
    bounded, non-empty mesh with the cascadio backend recorded."""
    pytest.importorskip('cascadio')
    path = Path(pyvista_cad.examples.step_cube)
    out = pyvista_cad.read_step(path, backend='cascadio')
    combined = out.combine() if isinstance(out, pv.MultiBlock) else out
    assert combined.n_cells > 0
    assert combined.n_points > 0
    assert str(out.field_data['cad.backend'][0]) == 'cascadio'
    assert str(out.field_data['cad.source_format'][0]) == 'step'
    b = combined.bounds
    # The fixture is a cube centred on the origin. Assert that shape
    # as an invariant rather than a literal extent: cascadio's output unit
    # is version-dependent (newer cascadio emits metres, older
    # millimetres), so a fixed ±5.0 is brittle. Centred + cubic +
    # non-degenerate holds in any unit.
    ext_x = b.x_max - b.x_min
    ext_z = b.z_max - b.z_min
    assert ext_x > 0
    assert b.x_min == pytest.approx(-b.x_max, abs=ext_x * 1e-3)
    assert b.z_min == pytest.approx(-b.z_max, abs=ext_z * 1e-3)
    assert ext_x == pytest.approx(ext_z, rel=1e-3)
