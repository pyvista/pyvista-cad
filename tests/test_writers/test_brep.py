"""BREP writer round-trips (INV-025: moved out of tests/test_roundtrip/).

These assert the *writer* contract: ``write_brep`` followed by
``read_brep`` restores geometry and the ``cad.*`` metadata. Read-side
exact-geometry invariants (exact_volume / center_of_mass over the cached
TopoDS_Shape) stay in tests/test_roundtrip/test_brep_roundtrip.py.
"""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('OCP', exc_type=ImportError)
b3d = pytest.importorskip('build123d', exc_type=ImportError)


def test_write_brep_round_trip_geometry(tmp_path: Path) -> None:
    """Cell count and bounds survive ``write_brep`` -> ``read_brep``."""
    src = pyvista_cad.from_build123d(b3d.Box(2.0, 4.0, 6.0))
    path = tmp_path / 'box.brep'
    pyvista_cad.write_brep(src, path)
    assert path.exists()
    rt = pyvista_cad.read_brep(path)
    assert isinstance(rt, pv.PolyData)
    assert rt.n_cells == src.n_cells
    assert rt.bounds == pytest.approx(tuple(src.bounds), abs=1e-6)


def test_write_brep_round_trip_metadata(tmp_path: Path) -> None:
    """``cad.source_format`` / ``cad.backend`` / brep-origin survive the
    writer round-trip."""
    src = pyvista_cad.from_build123d(b3d.Box(3.0, 1.0, 2.0))
    path = tmp_path / 'box.brep'
    pyvista_cad.write_brep(src, path)
    rt = pyvista_cad.read_brep(path)
    assert str(rt.field_data['cad.source_format'][0]) == 'brep'
    assert str(rt.field_data['cad.backend'][0]) == 'ocp'
    assert bool(rt.field_data['cad.has_brep_origin'][0])


def test_write_brep_pv_read_dispatch(tmp_path: Path) -> None:
    """``pv.read`` reads the written BREP back to identical geometry."""
    src = pyvista_cad.from_build123d(b3d.Box(1.0, 1.0, 1.0))
    path = tmp_path / 'cube.brep'
    pyvista_cad.write_brep(src, path)
    rt = pv.read(path)
    assert isinstance(rt, pv.PolyData)
    assert rt.n_cells == src.n_cells


def test_write_brep_unsupported_type_raises(tmp_path: Path) -> None:
    """``write_brep`` rejects a non-PolyData input with ``CadWriteError``
    naming the offending type, before touching OCP."""
    from pyvista_cad import CadWriteError  # noqa: PLC0415

    grid = pv.ImageData(dimensions=(2, 2, 2))
    out = tmp_path / 'x.brep'
    with pytest.raises(CadWriteError, match='ImageData'):
        pyvista_cad.write_brep(grid, out)
    assert not out.exists()


def test_write_brep_unwritable_path_raises(tmp_path: Path) -> None:
    """An unwritable destination surfaces as ``CadWriteError`` instead of a
    silent no-op.

    ``BRepTools.Write_s`` returns a bool status and never raises on a bad
    path, so the writer must check the status and the materialised file.
    """
    from pyvista_cad import CadWriteError  # noqa: PLC0415

    src = pyvista_cad.from_build123d(b3d.Box(1.0, 1.0, 1.0))
    # Parent directory does not exist: OCCT cannot open the stream.
    bad = tmp_path / 'no_such_dir' / 'out.brep'
    with pytest.raises(CadWriteError, match='failed to write BREP'):
        pyvista_cad.write_brep(src, bad)
    assert not bad.exists()
