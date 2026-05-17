"""Packaging contract: install hints and declared extras cannot drift.

Every ``pyvista-cad[<extra>]`` install hint embedded in a source
``OptionalDependencyError`` (or any other src string) must name an
extra that actually exists in ``pyproject.toml``. Parsing both sides
here makes the two lists fail loudly the moment they desynchronise,
instead of shipping an install hint that ``pip`` cannot satisfy.
"""

from pathlib import Path
import re
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - dev/CI run on >= 3.11
    tomllib = pytest.importorskip('tomli')

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / 'pyproject.toml'
_SRC = _REPO_ROOT / 'src' / 'pyvista_cad'

# ``pyvista-cad[a,b]`` and ``pyvista-cad[a]``; capture the bracket body.
_HINT_RE = re.compile(r'pyvista-cad\[([a-z0-9,_-]+)\]')


def _declared_extras() -> set[str]:
    with _PYPROJECT.open('rb') as fh:
        data = tomllib.load(fh)
    return set(data['project'].get('optional-dependencies', {}))


def _referenced_extras() -> dict[str, list[str]]:
    """Map each extra token referenced in src to the files referencing it."""
    refs: dict[str, list[str]] = {}
    for py in _SRC.rglob('*.py'):
        text = py.read_text(encoding='utf-8')
        for match in _HINT_RE.finditer(text):
            for token in match.group(1).split(','):
                refs.setdefault(token, []).append(str(py.relative_to(_REPO_ROOT)))
    return refs


def test_every_referenced_extra_is_declared() -> None:
    """Each ``pyvista-cad[<extra>]`` install hint names a declared extra.

    Both sides are parsed (pyproject extras and src hint strings) so an
    install hint can never reference an extra that ``pip`` cannot
    resolve.
    """
    declared = _declared_extras()
    referenced = _referenced_extras()
    missing = {token: files for token, files in referenced.items() if token not in declared}
    assert not missing, (
        f'install hints reference undeclared extras: {missing}. '
        f'Declared extras: {sorted(declared)}.'
    )


def test_core_extras_exist() -> None:
    """The user-facing per-format extras must exist."""
    declared = _declared_extras()
    for required in ('cadquery', 'trimesh', '3mf'):
        assert required in declared, (
            f'extra {required!r} missing from pyproject; declared: {sorted(declared)}'
        )


def test_full_aggregates_the_format_extras() -> None:
    """``full`` must aggregate the optional-format extras.

    ``full`` is the single-shot install target; if a per-format extra
    is added without being folded into ``full``, users who installed
    ``pyvista-cad[full]`` silently miss it.
    """
    with _PYPROJECT.open('rb') as fh:
        data = tomllib.load(fh)
    extras = data['project']['optional-dependencies']
    assert 'full' in extras, 'no `full` extra declared'

    full_spec = ' '.join(extras['full'])
    match = _HINT_RE.search(full_spec)
    assert match is not None, f'`full` does not reference pyvista-cad[...]: {full_spec}'
    aggregated = set(match.group(1).split(','))

    for required in ('step', '3mf', 'iges', 'ifc', 'gmsh', 'cadquery', 'trimesh'):
        assert required in aggregated, (
            f'`full` does not aggregate {required!r}; aggregates {sorted(aggregated)}'
        )

    # `openscad` has no installable extra (it needs the `openscad`
    # command-line binary, not a pip package), so it must not appear in
    # the `full` aggregate.
    assert 'openscad' not in aggregated, (
        '`full` must not aggregate `openscad`: it has no pip-installable extra'
    )
