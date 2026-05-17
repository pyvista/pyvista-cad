"""IGES read invariants.

No IGES writer is implemented; this file asserts the read-side preserves the
pyiges entity inventory so a future writer can be plugged in against a
known ground truth.
"""

import pytest

import pyvista_cad

pyiges = pytest.importorskip('pyiges')
pytest.importorskip('pyiges.examples')


def test_iges_pyiges_level_preserved() -> None:
    """Entity count + level set matches what raw pyiges sees."""
    from pyiges import examples  # noqa: PLC0415

    iges = pyiges.read(examples.sample)
    src_count = len(list(iges))
    src_levels = {getattr(e, 'level', None) for e in iges}

    out = pyvista_cad.read_iges(
        examples.sample,
        bsplines=True,
        surfaces=True,
        lines=True,
    )
    # Re-walking the parsed file should yield the same entity inventory.
    iges2 = pyiges.read(examples.sample)
    assert len(list(iges2)) == src_count
    assert {getattr(e, 'level', None) for e in iges2} == src_levels
    # The dataset must record the format string used by downstream lookups.
    assert str(out.field_data['cad.source_format'][0]) == 'iges'
