"""Shared fixtures and collection hooks for the perf-regression suite.

These tests are opt-in. They run only when ``--benchmark-only`` is
passed (or ``--benchmark-enable``), otherwise they are skipped at
collection time. Each individual test additionally skips itself when
its optional dependency is missing so a partial install can still
exercise the available subset.
"""

import importlib.util

import pytest


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
