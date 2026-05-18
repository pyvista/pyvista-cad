"""BREP read -> write -> re-read invariants over the cached ``TopoDS_Shape``.

The BREP reader caches the originating ``TopoDS_Shape`` against the returned
``PolyData`` so ``cad.exact_volume`` / ``cad.center_of_mass`` evaluate against
exact geometry. Re-reading the written BREP must restore that cache.
"""

from pathlib import Path

import numpy as np
import pytest

import pyvista_cad

pytest.importorskip('OCP', exc_type=ImportError)
b3d = pytest.importorskip('build123d', exc_type=ImportError)


def _box_polydata():
    """A faceted box originating from build123d so it carries a BREP cache."""
    return pyvista_cad.from_build123d(b3d.Box(2.0, 4.0, 6.0))


def test_brep_round_trip_preserves_exact_volume(tmp_path: Path) -> None:
    """``exact_volume`` (computed from the cached TopoDS) is invariant
    across BREP write/read because both sides carry an exact shape."""
    src = _box_polydata()
    src_volume = src.cad.exact_volume()
    assert src_volume == pytest.approx(2.0 * 4.0 * 6.0, rel=1e-9)

    path = tmp_path / 'box.brep'
    pyvista_cad.write_brep(src, path)
    rt = pyvista_cad.read_brep(path)
    rt_volume = rt.cad.exact_volume()
    assert rt_volume == pytest.approx(src_volume, rel=1e-6)


def test_brep_round_trip_preserves_center_of_mass(tmp_path: Path) -> None:
    """``center_of_mass`` is invariant across BREP write/read."""
    src = _box_polydata()
    src_com = src.cad.center_of_mass()

    path = tmp_path / 'box.brep'
    pyvista_cad.write_brep(src, path)
    rt = pyvista_cad.read_brep(path)
    rt_com = rt.cad.center_of_mass()
    np.testing.assert_allclose(rt_com, src_com, atol=1e-6)
