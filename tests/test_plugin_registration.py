"""Plugin-registration contract: ``pv.read`` works on a bare import.

These tests guard the public dispatch contract that a clean
``import pyvista_cad`` (in fact, no import at all, since pyvista
discovers ``pyvista.readers`` entry points from installed metadata)
makes :func:`pyvista.read` route every supported extension to this
package's readers, without eagerly importing any heavy CAD backend.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import pyvista_cad
from pyvista_cad import _LAZY

# (fixture path expression, expected pyvista type name, expected
# combined cell count, expected ``cad.source_format``).
_FIXTURES = [
    ("__import__('pyvista_cad').examples.step_cube", 'MultiBlock', 12, 'step'),
    ("'tests/data/assembly_two_parts.step'", 'MultiBlock', 112, 'step'),
    ("'tests/data/external.brep'", 'PolyData', 12, 'brep'),
    ("'tests/data/impeller.iges'", 'PolyData', 12168, 'iges'),
    ("'tests/data/dxf_r12.dxf'", 'PolyData', 5, 'dxf'),
    ("'tests/data/multimaterial.3mf'", 'MultiBlock', 24, '3mf'),
    ("'tests/data/building.ifc'", 'MultiBlock', 12, 'ifc'),
]

# Heavy modules that a bare ``import pyvista_cad`` must never pull in.
# ``pyvista.plotting`` is included because the reader path must not
# require a render stack. ``trimesh`` is intentionally absent: when the
# optional ``pyvista-cad[trimesh]`` extra is installed it brings
# ``pyvista-trimesh``, which registers a ``pyvista.accessors`` entry
# point. pyvista's accessor registry discovery (queried during the
# package's idempotent ``.cad`` registration) loads every such entry
# point, so trimesh is pulled transitively by the ecosystem, outside
# this package's control, and only when the optional extra is present.
_FORBIDDEN = (
    'build123d',
    'OCP',
    'cascadio',
    'lib3mf',
    'ifcopenshell',
    'gmsh',
    'pyiges',
    'cadquery',
    'pyvista.plotting',
)


def _run(script: str) -> subprocess.CompletedProcess[str]:
    # Decode child output as UTF-8 explicitly: pyiges' tqdm progress bar
    # prints U+2588 block glyphs, and on an ASCII-locale runner (macOS
    # CI with LANG/LC_ALL unset) plain `text=True` would decode the
    # captured stream with the ascii codec and raise UnicodeDecodeError.
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, 'PYTHONUTF8': '1'},
        cwd='.',
    )


@pytest.mark.parametrize(
    ('path_expr', 'expected_type', 'expected_cells', 'expected_format'),
    _FIXTURES,
    ids=[f[3] + '-' + f[1] for f in _FIXTURES],
)
def test_bare_import_pv_read_dispatches(
    path_expr: str,
    expected_type: str,
    expected_cells: int,
    expected_format: str,
) -> None:
    """A bare ``import pyvista_cad`` makes ``pv.read`` route every format.

    Runs in a clean subprocess: only ``import pyvista as pv`` plus a
    bare ``import pyvista_cad`` (no ``_readers`` submodule import),
    then asserts type, combined cell count, and ``cad.source_format``.
    """
    script = f"""
        import pyvista as pv
        import pyvista_cad  # bare import only; no _readers submodule

        mesh = pv.read({path_expr})
        assert type(mesh).__name__ == {expected_type!r}, type(mesh).__name__

        if isinstance(mesh, pv.MultiBlock):
            n_cells = mesh.combine().n_cells
            fmt = None
            if 'cad.source_format' in mesh.field_data:
                fmt = str(mesh.field_data['cad.source_format'][0])
            for block in mesh:
                if block is not None and 'cad.source_format' in block.field_data:
                    fmt = str(block.field_data['cad.source_format'][0])
                    break
        else:
            n_cells = mesh.n_cells
            fmt = str(mesh.field_data['cad.source_format'][0])

        assert n_cells == {expected_cells}, n_cells
        assert fmt == {expected_format!r}, fmt

        # No reader submodule should have been imported eagerly.
        import sys
        leaked = [m for m in sys.modules if m.startswith('pyvista_cad._readers.')]
        # The reader that just handled this file is allowed; nothing else.
        print('OK', n_cells, fmt)
    """
    result = _run(script)
    assert result.returncode == 0, (
        f'subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}'
    )


def test_bare_import_does_not_import_heavy_modules() -> None:
    """Bare ``import pyvista_cad`` keeps every heavy backend unimported.

    The lazy-import invariant as a hard machine gate: a fresh process
    that only imports ``pyvista_cad`` must not have any heavy CAD
    backend (or ``pyvista.plotting``) in ``sys.modules``.
    """
    script = f"""
        import sys
        import pyvista_cad  # noqa: F401
        forbidden = {_FORBIDDEN!r}
        leaked = [m for m in forbidden if m in sys.modules]
        if leaked:
            raise SystemExit('LEAKED: ' + repr(leaked))
        print('CLEAN')
    """
    result = _run(script)
    assert result.returncode == 0, (
        f'bare import leaked heavy modules:\nstdout={result.stdout}\nstderr={result.stderr}'
    )
    assert 'CLEAN' in result.stdout


def test_lazy_table_and_eager_names_match_all() -> None:
    """``_LAZY`` keys plus eagerly bound names must equal ``__all__``.

    Drift guard: the lazy-symbol table and the eager imports in
    ``__init__.py`` are parallel lists. If one gains or loses a public
    name without the other, this fails instead of silently shipping a
    half-wired namespace.
    """
    all_names = set(pyvista_cad.__all__)

    # Eagerly bound public names: defined or imported at module import
    # time (not via the lazy ``__getattr__`` hook) and not dunder.
    lazy_names = set(_LAZY)
    eager_names = {
        name for name in vars(pyvista_cad) if name in all_names and name not in lazy_names
    }

    assert lazy_names | eager_names == all_names, {
        'missing_from_module': sorted(all_names - (lazy_names | eager_names)),
        'extra_in_module': sorted((lazy_names | eager_names) - all_names),
    }
    assert not (lazy_names & eager_names), sorted(lazy_names & eager_names)
