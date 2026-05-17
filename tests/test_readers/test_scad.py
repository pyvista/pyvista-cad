"""Tests for the OpenSCAD (.scad) reader.

The absent-binary path is exercised for real by pointing ``PATH`` at an
empty directory so the genuine ``shutil.which('openscad')`` returns
``None`` (no ``shutil.which`` monkeypatch). The Linux CI ``test`` job
installs the OpenSCAD binary (``apt-get install -y openscad``), so the
end-to-end cases run there; on a runner or environment without the
binary they skip cleanly.
"""

from pathlib import Path
import shutil

import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._errors import OptionalDependencyError


def test_read_scad_missing_binary_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``PATH`` pointed at an empty directory the real
    ``shutil.which('openscad')`` returns ``None`` and the reader raises
    ``OptionalDependencyError`` naming the missing binary."""
    empty_bin = tmp_path / 'empty_path'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', str(empty_bin))
    # Sanity: the real resolver genuinely fails here (no mock).
    assert shutil.which('openscad') is None

    scad = tmp_path / 'thing.scad'
    scad.write_text('cube([1,1,1]);\n')
    with pytest.raises(OptionalDependencyError, match='openscad'):
        pyvista_cad.read_scad(scad)


@pytest.mark.skipif(
    shutil.which('openscad') is None,
    reason='openscad binary not installed (CI: apt-get install -y openscad)',
)
def test_read_scad_compiles_cube(tmp_path: Path) -> None:
    """End-to-end: a real ``openscad`` compiles a cube to a bounded mesh."""
    scad = tmp_path / 'cube.scad'
    scad.write_text('cube([10, 10, 10]);\n')
    poly = pyvista_cad.read_scad(scad)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0
    assert poly.n_points > 0
    b = poly.bounds
    assert (b.x_min, b.x_max) == pytest.approx((0.0, 10.0), abs=1e-6)
    assert (b.y_min, b.y_max) == pytest.approx((0.0, 10.0), abs=1e-6)
    assert (b.z_min, b.z_max) == pytest.approx((0.0, 10.0), abs=1e-6)
    assert str(poly.field_data['cad.source_format'][0]) == 'scad'
    assert str(poly.field_data['cad.label'][0]) == 'cube'


@pytest.mark.skipif(
    shutil.which('openscad') is None,
    reason='openscad binary not installed (CI: apt-get install -y openscad)',
)
def test_read_scad_via_pv_read(tmp_path: Path) -> None:
    """``pv.read`` dispatch compiles a sphere to a non-empty, bounded mesh."""
    scad = tmp_path / 'sphere.scad'
    scad.write_text('sphere(r=5);\n')
    poly = pv.read(str(scad))
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0
    b = poly.bounds
    assert b.x_min == pytest.approx(-5.0, abs=0.5)
    assert b.x_max == pytest.approx(5.0, abs=0.5)
