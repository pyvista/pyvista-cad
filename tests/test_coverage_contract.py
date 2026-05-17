"""Executable enforcement of the per-file coverage mandate.

``pyproject.toml`` sets ``fail_under = 0`` (intentionally: each CI extras
combination exercises a different code subset, so a single global gate
cannot apply). That makes the "100% on the core conversion path, >=90%
everywhere else" rule prose-only: a regression dropping
``_conversion.py`` to 95% would still pass CI green.

This test makes the mandate executable. It runs the whole suite once in
a dedicated subprocess under ``coverage`` (branch coverage on, the
project ``[tool.coverage.run]`` config), emits ``coverage json``, and
asserts per-file thresholds:

* ``_conversion.py``, ``_metadata.py``, ``_accessor.py``, ``_errors.py``
  must be at 100.0% with zero missing lines AND zero missing branches.
* every other shipped module outside ``_backends/`` (``_cad_view.py``,
  ``_readers/*``, ``_bridges/*``, ``examples/*`` except ``_generate.py``,
  ``__init__.py``) must be at >= 90.0%.

``_backends/*`` and ``examples/_generate.py`` are exempt (the former is
optional-dependency glue whose coverage depends on which extras are
installed; the latter is a build-time fixture generator already
``omit``-ed in ``[tool.coverage.run]``).

The mandate is only meaningful when the full optional toolchain is
installed (build123d, OCP, lib3mf, cascadio, ...). On a thin matrix cell
(``dev`` only) most readers/bridges are import-skipped and would sit far
below 90% through no regression, so the contract self-skips unless the
full toolchain is importable. The CI ``dev,full,step-light`` Linux cell
has everything and is where this gate bites.

The nested ``pytest`` run is deselected from re-running this test
(``--deselect`` plus a ``PYVISTA_CAD_COVERAGE_CONTRACT_RUNNING`` env
guard), so there is no recursion and no flakiness from re-entrancy.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

_REENTRY_GUARD = 'PYVISTA_CAD_COVERAGE_CONTRACT_RUNNING'

# Files that must be exactly 100% (line + branch).
_STRICT_100 = (
    '_conversion.py',
    '_metadata.py',
    '_accessor.py',
    '_errors.py',
)

# Optional-backend toolchain the >=90% modules need to be exercised at
# all. If any is missing, the broad floor is meaningless on this cell.
_REQUIRED_FOR_FULL = (
    'build123d',
    'OCP',
    'lib3mf',
    'cascadio',
    'ezdxf',
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / 'src' / 'pyvista_cad'


def _toolchain_complete() -> tuple[bool, str]:
    import importlib.util  # noqa: PLC0415

    missing = [m for m in _REQUIRED_FOR_FULL if importlib.util.find_spec(m) is None]
    return (not missing, ', '.join(missing))


def _rel_module(file_path: str) -> str | None:
    """Return the ``pyvista_cad``-relative POSIX path, or ``None`` if the
    measured file is not part of the shipped package."""
    p = Path(file_path).resolve()
    try:
        rel = p.relative_to(_SRC)
    except ValueError:
        return None
    return rel.as_posix()


def _is_exempt(rel: str) -> bool:
    """``_backends/*`` and the build-time fixture generator are out of
    scope for the mandate."""
    if rel.startswith('_backends/'):
        return True
    return rel == 'examples/_generate.py'


def test_per_file_coverage_contract(tmp_path: Path) -> None:
    """Enforce the 100% / >=90% per-file coverage mandate for real.

    Fails if any strict file is below 100% (line or branch) or any other
    in-scope shipped module is below 90%.
    """
    if os.environ.get(_REENTRY_GUARD):
        pytest.skip('inside the nested coverage subprocess; skip to avoid recursion')

    complete, missing = _toolchain_complete()
    if not complete:
        pytest.skip(
            f'full optional toolchain absent ({missing}); the >=90% floor is '
            'only enforceable with all backends installed (CI '
            'dev,full,step-light cell)'
        )

    data_file = tmp_path / '.coverage'
    json_file = tmp_path / 'coverage.json'
    env = dict(os.environ)
    env[_REENTRY_GUARD] = '1'
    env['COVERAGE_FILE'] = str(data_file)

    run = subprocess.run(
        [
            sys.executable,
            '-m',
            'coverage',
            'run',
            '--rcfile',
            str(_REPO_ROOT / 'pyproject.toml'),
            '-m',
            'pytest',
            '-q',
            '--deselect',
            'tests/test_coverage_contract.py::test_per_file_coverage_contract',
            str(_REPO_ROOT / 'tests'),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, (
        'nested suite run under coverage failed; cannot evaluate the '
        f'coverage contract.\n--- stdout tail ---\n{run.stdout[-3000:]}\n'
        f'--- stderr tail ---\n{run.stderr[-3000:]}'
    )

    json_proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'coverage',
            'json',
            '--rcfile',
            str(_REPO_ROOT / 'pyproject.toml'),
            '-o',
            str(json_file),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert json_proc.returncode == 0, (
        f'`coverage json` failed.\n{json_proc.stdout}\n{json_proc.stderr}'
    )

    report = json.loads(json_file.read_text())
    files = report['files']

    strict_failures: list[str] = []
    floor_failures: list[str] = []
    seen_strict: set[str] = set()

    for file_path, info in files.items():
        rel = _rel_module(file_path)
        if rel is None or _is_exempt(rel):
            continue

        summary = info['summary']
        pct = summary['percent_covered']
        missing_lines = summary['missing_lines']
        # `missing_branches` is present whenever branch=true (it is, via
        # [tool.coverage.run]); default 0 keeps this robust regardless.
        missing_branches = summary.get('missing_branches', 0)

        base = Path(rel).name
        if rel in _STRICT_100 or base in _STRICT_100:
            seen_strict.add(base)
            if missing_lines or missing_branches or pct < 100.0:
                strict_failures.append(
                    f'{rel}: {pct:.2f}% (missing_lines={missing_lines}, '
                    f'missing_branches={missing_branches}) -- must be 100.0%'
                )
        elif pct < 90.0:
            floor_failures.append(f'{rel}: {pct:.2f}% -- must be >= 90.0%')

    not_measured = sorted(set(_STRICT_100) - seen_strict)
    assert not not_measured, (
        f'strict-100 files were never measured by the suite: {not_measured}. '
        'A renamed/relocated module or a collection failure would hide a '
        'regression here.'
    )

    problems = strict_failures + floor_failures
    assert not problems, 'per-file coverage contract violated:\n  ' + '\n  '.join(sorted(problems))
