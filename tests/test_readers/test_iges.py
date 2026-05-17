"""Tests for the IGES reader, asserted against the committed impeller fixture."""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('pyiges')


def test_read_iges_impeller_fidelity(impeller_iges_path: Path) -> None:
    """The B-spline impeller tessellates to an exact, known PolyData.

    Counts and bounds are pinned from ``tests/data/README.md``; a drift in
    the pyiges tessellation path moves these immediately.
    """
    out = pyvista_cad.read_iges(impeller_iges_path)
    assert isinstance(out, pv.PolyData)
    assert out.n_cells == 12168
    assert out.n_points == 6400
    assert out.n_cells > 0

    b = out.bounds
    assert b.x_min == pytest.approx(-25.246551, abs=1e-4)
    assert b.x_max == pytest.approx(28.0, abs=1e-4)
    assert b.y_min == pytest.approx(-28.000005, abs=1e-4)
    assert b.y_max == pytest.approx(24.248711, abs=1e-4)
    assert b.z_min == pytest.approx(0.0, abs=1e-4)
    assert b.z_max == pytest.approx(20.0, abs=1e-4)

    assert str(out.field_data['cad.source_format'][0]) == 'iges'
    assert str(out.field_data['cad.backend'][0]) == 'pyiges'


def test_read_iges_via_pv_read_round_trips_metadata(impeller_iges_path: Path) -> None:
    """``pv.read`` dispatch yields the same geometry and ``cad.*`` payload as
    the explicit reader (not merely a non-None object)."""
    direct = pyvista_cad.read_iges(impeller_iges_path)
    out = pv.read(impeller_iges_path)
    assert isinstance(out, pv.PolyData)
    assert out.n_cells == direct.n_cells
    assert out.n_points == direct.n_points
    assert str(out.field_data['cad.source_format'][0]) == 'iges'
    assert str(out.field_data['cad.backend'][0]) == 'pyiges'


def test_read_iges_missing_start_section_raises(tmp_path: Path) -> None:
    """An IGES file with no Start / Global section raises ``CadReadError``."""
    from pyvista_cad import CadReadError

    bad = tmp_path / 'no_start.iges'
    # IGES expects a column-73 section-ID character on every line. Random
    # ASCII with no IGES structure should bounce.
    bad.write_text('this is not an iges file\n' * 10)
    with pytest.raises(CadReadError):
        pyvista_cad.read_iges(bad)
