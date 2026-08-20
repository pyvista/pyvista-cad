"""Tests for the OCCT-backed IGES reader (``backend='ocp'``).

Kept separate from ``test_iges.py`` so these run in the ``[step]``-only
CI jobs, where pyiges is absent.
"""

from pathlib import Path
import sys

import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad import CadReadError, OptionalDependencyError
from pyvista_cad._conversion import get_cached_topods

pytest.importorskip('OCP', exc_type=ImportError)

# Bounds of the impeller fixture, pinned in tests/data/README.md from the
# pyiges path. OCCT tessellates the same type-128 surfaces, so the hull
# agrees to well within a chordal-deflection's worth of slack.
_EXPECTED_BOUNDS = (-25.246551, 28.0, -28.0, 24.248711, 0.0, 20.0)


# A structurally valid IGES with an empty Directory Entry section: Start,
# Global, then terminate. Every line is padded to the 72-column body plus
# the section letter and sequence number the format requires.
_EMPTY_IGES = (
    'Empty model.'.ljust(72)
    + 'S      1\n'
    + '1H,,1H;,4Hnone,4Hnone,4Hnone,4Hnone,32,38,6,308,15,4Hnone,1.,1,2HMM,1,0.'
    + 'G      1\n'
    + '08,13H000000.000000,1.E-07,0.,4Hnone,4Hnone,11,0,13H000000.000000;'.ljust(72)
    + 'G      2\n'
    + 'S     1G     2D     0P     0'.ljust(72)
    + 'T      1\n'
)


def test_read_iges_ocp_geometry_and_metadata(impeller_iges_path: Path) -> None:
    """The OCCT backend returns a stamped ``PolyData`` on the pinned hull."""
    out = pyvista_cad.read_iges(impeller_iges_path, backend='ocp')
    assert isinstance(out, pv.PolyData)
    assert out.n_cells > 0

    # Counts are not pinned: OCCT's tessellation density for curved
    # surfaces shifts between OCCT releases. The hull does not.
    for actual, expected in zip(out.bounds, _EXPECTED_BOUNDS, strict=True):
        assert actual == pytest.approx(expected, abs=0.05)

    assert str(out.field_data['cad.source_format'][0]) == 'iges'
    assert str(out.field_data['cad.backend'][0]) == 'ocp'


def test_read_iges_ocp_deflection_controls_density(impeller_iges_path: Path) -> None:
    """A tighter ``linear_deflection`` yields a denser tessellation."""
    coarse = pyvista_cad.read_iges(impeller_iges_path, backend='ocp', linear_deflection=0.5)
    fine = pyvista_cad.read_iges(impeller_iges_path, backend='ocp', linear_deflection=0.01)
    assert fine.n_cells > coarse.n_cells


def test_read_iges_ocp_caches_brep(impeller_iges_path: Path) -> None:
    """The originating ``TopoDS_Shape`` survives on the result.

    Without it ``.cad`` would fall back to crease feature edges instead
    of the model's topological B-rep edges.
    """
    out = pyvista_cad.read_iges(impeller_iges_path, backend='ocp')
    assert get_cached_topods(out) is not None


def test_read_iges_unknown_backend_raises(impeller_iges_path: Path) -> None:
    """An unrecognized backend name is rejected, not silently defaulted."""
    with pytest.raises(OptionalDependencyError, match='unknown IGES backend'):
        # Deliberately off-Literal: the runtime guard, not mypy, is
        # what protects a caller who built the name dynamically.
        pyvista_cad.read_iges(impeller_iges_path, backend='nope')  # type: ignore[arg-type]


def test_read_iges_ocp_missing_file_raises(tmp_path: Path) -> None:
    """A path that does not exist raises before OCCT is invoked."""
    with pytest.raises(FileNotFoundError):
        pyvista_cad.read_iges(tmp_path / 'absent.igs', backend='ocp')


def test_read_iges_ocp_garbage_raises(tmp_path: Path) -> None:
    """A file with no IGES structure raises ``CadReadError``.

    This is the reader-status guard: OCCT reports ``IFSelect_RetError``
    and never reaches the transfer step.
    """
    bad = tmp_path / 'no_start.iges'
    bad.write_text('this is not an iges file\n' * 10)
    with pytest.raises(CadReadError, match='reader status'):
        pyvista_cad.read_iges(bad, backend='ocp')


def test_read_iges_ocp_empty_model_raises(tmp_path: Path) -> None:
    """A well-formed IGES carrying no geometry raises ``CadReadError``.

    Distinct from the garbage case: OCCT parses this happily and returns
    ``IFSelect_RetDone``, then transfers zero roots. Returning an empty
    ``PolyData`` here would be indistinguishable from a successful read
    of a part with no faces.
    """
    empty = tmp_path / 'empty_model.iges'
    empty.write_text(_EMPTY_IGES)
    with pytest.raises(CadReadError, match='transferred no geometry'):
        pyvista_cad.read_iges(empty, backend='ocp')


@pytest.mark.parametrize('missing', ['OCP.IGESControl', 'OCP.IFSelect'])
def test_read_iges_ocp_without_ocp_raises_install_hint(
    impeller_iges_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """A missing OCP surfaces the install hint, not a bare ``ImportError``.

    Both OCP symbols must sit under the same guard: probing one of them
    outside it leaks the raw ``ImportError`` and loses the hint. Binding
    the name to ``None`` in ``sys.modules`` is the standard way to make
    ``import`` fail for an installed module.
    """
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(OptionalDependencyError, match=r'install pyvista-cad\[step\]'):
        pyvista_cad.read_iges(impeller_iges_path, backend='ocp')
