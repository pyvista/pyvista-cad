"""Tests for the STEP reader."""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('build123d', exc_type=ImportError)


def test_read_step_returns_multiblock() -> None:
    """The bundled STEP cube reads as a MultiBlock carrying the ``cad.*``
    provenance payload and the exact 10 mm cube bounds."""
    mb = pyvista_cad.read_step(pyvista_cad.examples.step_cube)
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks >= 1
    first = mb[0]
    assert first is not None
    assert first.n_points > 0
    assert first.n_cells > 0

    assert str(mb.field_data['cad.source_format'][0]) == 'step'
    assert str(mb.field_data['cad.reader'][0]) == 'pyvista_cad.read_step'
    assert str(mb.field_data['cad.backend'][0]) == 'build123d'

    b = mb.combine().bounds
    assert (b.x_min, b.x_max) == pytest.approx((-5.0, 5.0), abs=1e-4)
    assert (b.y_min, b.y_max) == pytest.approx((-5.0, 5.0), abs=1e-4)
    assert (b.z_min, b.z_max) == pytest.approx((-5.0, 5.0), abs=1e-4)


def test_read_step_bounds_match_cube() -> None:
    mb = pyvista_cad.read_step(pyvista_cad.examples.step_cube)
    combined = mb.combine() if isinstance(mb, pv.MultiBlock) else mb
    b = combined.bounds
    assert b.x_min == pytest.approx(-5.0, abs=0.5)
    assert b.x_max == pytest.approx(5.0, abs=0.5)
    assert b.y_min == pytest.approx(-5.0, abs=0.5)
    assert b.y_max == pytest.approx(5.0, abs=0.5)
    assert b.z_min == pytest.approx(-5.0, abs=0.5)
    assert b.z_max == pytest.approx(5.0, abs=0.5)


def test_read_step_merge_returns_polydata() -> None:
    poly = pyvista_cad.read_step(pyvista_cad.examples.step_cube, merge=True)
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0


def test_read_step_corrupted_header_raises(tmp_path: Path) -> None:
    """STEP file with a corrupted ISO-10303 header raises ``CadReadError``."""
    from pyvista_cad import CadReadError  # noqa: PLC0415

    bad = tmp_path / 'corrupt.step'
    bad.write_bytes(
        b'NOT-A-STEP-HEADER-AT-ALL;\n'
        b'HEADER;\n'
        b'FILE_DESCRIPTION garbled lacking parentheses\n'
        b'ENDSEC;\n'
        b'DATA;\n#1=GARBAGE_ENTITY();\nENDSEC;\nEND-ISO-10303-21;\n'
    )
    with pytest.raises(CadReadError):
        pyvista_cad.read_step(bad)


def test_pv_read_routes_step() -> None:
    """``pv.read`` dispatch must preserve the ``cad.*`` payload, not just
    return nonzero counts."""
    obj = pv.read(pyvista_cad.examples.step_cube)
    assert isinstance(obj, (pv.MultiBlock, pv.PolyData))
    combined = obj.combine() if isinstance(obj, pv.MultiBlock) else obj
    assert combined.n_cells > 0
    assert combined.n_points > 0
    assert str(obj.field_data['cad.source_format'][0]) == 'step'
    assert str(obj.field_data['cad.reader'][0]) == 'pyvista_cad.read_step'
    assert str(obj.field_data['cad.backend'][0]) == 'build123d'


def test_read_step_assembly_two_parts(assembly_step_path: Path) -> None:
    """The OCAF assembly fixture reads as two named, correctly-bounded products.

    build123d resolves the OCAF component placement into the shape itself,
    so the Pin's ``(15, 10, 10)`` component transform is applied exactly
    once (the reader must not re-apply it). A radius-4 mm, height-25 mm
    cylinder placed at ``(15, 10, 10)`` spans X ``[11, 19]``, Y
    ``[6.03, 13.97]``, Z ``[10, 35]``.
    """
    mb = pyvista_cad.read_step(assembly_step_path)
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 2
    assert list(mb.keys()) == ['Base', 'Pin']

    assert str(mb.field_data['cad.units'][0]) == 'mm'
    assert str(mb.field_data['cad.source_format'][0]) == 'step'
    assert str(mb.field_data['cad.reader'][0]) == 'pyvista_cad.read_step'
    assert str(mb.field_data['cad.backend'][0]) == 'build123d'

    combined = mb.combine()
    cb = combined.bounds
    assert cb.x_min == pytest.approx(0.0, abs=1e-4)
    assert cb.x_max == pytest.approx(30.0, abs=1e-4)
    assert cb.y_min == pytest.approx(0.0, abs=1e-4)
    assert cb.y_max == pytest.approx(20.0, abs=1e-4)
    assert cb.z_min == pytest.approx(0.0, abs=1e-4)
    assert cb.z_max == pytest.approx(35.0, abs=1e-4)

    base = mb['Base']
    assert base.n_cells == 12
    assert str(base.field_data['cad.label'][0]) == 'Base'
    bb = base.bounds
    assert (bb.x_min, bb.x_max) == pytest.approx((0.0, 30.0), abs=1e-4)
    assert (bb.y_min, bb.y_max) == pytest.approx((0.0, 20.0), abs=1e-4)
    assert (bb.z_min, bb.z_max) == pytest.approx((0.0, 10.0), abs=1e-4)

    pin = mb['Pin']
    assert pin.n_cells == 100
    assert str(pin.field_data['cad.label'][0]) == 'Pin'
    pb = pin.bounds
    assert (pb.x_min, pb.x_max) == pytest.approx((11.0, 19.0), abs=1e-4)
    assert (pb.y_min, pb.y_max) == pytest.approx((6.029165, 13.970835), abs=1e-4)
    assert (pb.z_min, pb.z_max) == pytest.approx((10.0, 35.0), abs=1e-4)


def test_resolve_backend_explicit_passthrough() -> None:
    """An explicit ``backend`` is returned verbatim by ``_resolve_backend``,
    skipping installed-extra autodetection."""
    from pyvista_cad._readers.step import _resolve_backend  # noqa: PLC0415

    assert _resolve_backend('build123d') == 'build123d'
    assert _resolve_backend('cascadio') == 'cascadio'


def test_resolve_backend_autodetect_build123d() -> None:
    """With build123d installed and no explicit backend, autodetect picks it."""
    from pyvista_cad._readers.step import _resolve_backend  # noqa: PLC0415

    assert _resolve_backend(None) == 'build123d'


def test_read_step_unknown_backend_raises() -> None:
    """An unknown explicit backend reaches the dispatch ``else`` and raises
    ``OptionalDependencyError`` naming the bad backend."""
    from pyvista_cad import OptionalDependencyError  # noqa: PLC0415

    with pytest.raises(OptionalDependencyError, match='bogus'):
        pyvista_cad.read_step(pyvista_cad.examples.step_cube, backend='bogus')  # type: ignore[arg-type]


def test_read_step_cascadio_backend_dispatch() -> None:
    """Explicitly selecting the cascadio backend dispatches to the cascadio
    implementation and returns real geometry.

    The Linux ``dev,full,step-light`` CI combination installs cascadio via
    the ``step-light`` extra, so this path is exercised in CI; it skips
    cleanly where cascadio is absent.
    """
    pytest.importorskip('cascadio')
    out = pyvista_cad.read_step(pyvista_cad.examples.step_cube, backend='cascadio')
    combined = out.combine() if isinstance(out, pv.MultiBlock) else out
    assert combined.n_cells > 0
    assert combined.n_points > 0
