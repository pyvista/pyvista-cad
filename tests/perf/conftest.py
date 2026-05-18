"""Shared fixtures and collection hooks for the perf-regression suite.

These tests are opt-in. They run only when ``--benchmark-only`` is
passed (or ``--benchmark-enable``), otherwise they are skipped at
collection time. Each individual test additionally skips itself when
its optional dependency is missing so a partial install can still
exercise the available subset.
"""

import importlib.util
import os
import sys

import pytest


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.fixture
def perf_budget():
    """Return a ``check(ok, message)`` for wall-clock / memory budgets.

    Absolute timing budgets are reliable locally but flake on shared
    GitHub runners (1.5-2x variance, cold caches, noisy neighbors). To
    keep the budgets meaningful as a local regression tripwire without a
    red CI job, breaches are a hard ``AssertionError`` by default but
    only reported (test still passes) when ``PYVISTA_CAD_PERF_SOFT`` is
    set (CI does). A plain stderr note is used rather than
    ``warnings.warn`` because the project runs with ``filterwarnings =
    ['error', ...]``, which would otherwise re-escalate it to a failure.
    The benchmark JSON is still emitted either way, so the planned
    cross-run regression gate keeps its data.
    """
    soft = bool(os.environ.get('PYVISTA_CAD_PERF_SOFT'))

    def _check(ok: bool, message: str) -> None:
        if ok:
            return
        if soft:
            print(f'PERF BUDGET EXCEEDED (soft/CI, non-fatal): {message}', file=sys.stderr)
        else:
            raise AssertionError(message)

    return _check


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the whole perf tree unless --benchmark-only / --benchmark-enable is set."""
    benchmark_only = config.getoption('--benchmark-only', default=False)
    benchmark_enable = config.getoption('--benchmark-enable', default=False)
    if benchmark_only or benchmark_enable:
        return
    skip = pytest.mark.skip(reason='perf suite: pass --benchmark-only to run')
    for item in items:
        if 'tests/perf' in str(item.fspath).replace('\\', '/'):
            item.add_marker(skip)


@pytest.fixture(scope='session')
def has_build123d() -> bool:
    """Whether build123d is importable."""
    return _has('build123d') and _has('OCP')


@pytest.fixture(scope='session')
def has_ezdxf() -> bool:
    """Whether ezdxf is importable."""
    return _has('ezdxf')


@pytest.fixture(scope='session')
def has_lib3mf() -> bool:
    """Whether lib3mf is importable."""
    return _has('lib3mf')
