"""OptionalDependencyError import-error branch for every bridge.

The only acceptable simulation of a missing optional dependency is
forcing the module absent via ``monkeypatch.setitem(sys.modules, ...,
None)``. This mirrors the documented lazy-import contract exercised in
``tests/test_lazy_imports.py``: a ``None`` entry in ``sys.modules`` makes
the next ``import`` raise ``ImportError``, which each bridge translates
into :class:`OptionalDependencyError`.
"""

import sys

import pytest
import pyvista as pv

from pyvista_cad import (
    OptionalDependencyError,
    from_build123d,
    from_cadquery,
    from_topods,
    to_build123d,
    to_cadquery,
    to_topods,
)


def test_from_cadquery_missing_cadquery(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cadquery', None)
    with pytest.raises(OptionalDependencyError, match=r'from_cadquery requires cadquery'):
        from_cadquery(object())


def test_to_cadquery_missing_cadquery(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cadquery', None)
    with pytest.raises(OptionalDependencyError, match=r'to_cadquery requires cadquery'):
        to_cadquery(pv.Sphere())


def test_from_topods_missing_ocp(monkeypatch):
    monkeypatch.setitem(sys.modules, 'OCP', None)
    with pytest.raises(OptionalDependencyError, match=r'OCP not installed'):
        from_topods(object())


def test_to_topods_missing_ocp(monkeypatch):
    monkeypatch.setitem(sys.modules, 'OCP', None)
    with pytest.raises(OptionalDependencyError, match=r'OCP not installed'):
        to_topods(pv.Sphere())


def test_from_build123d_missing_build123d(monkeypatch):
    monkeypatch.setitem(sys.modules, 'build123d', None)
    with pytest.raises(OptionalDependencyError, match=r'build123d not installed'):
        from_build123d(object())


def test_to_build123d_missing_build123d(monkeypatch):
    monkeypatch.setitem(sys.modules, 'build123d', None)
    with pytest.raises(OptionalDependencyError, match=r'build123d not installed'):
        to_build123d(pv.Sphere())
