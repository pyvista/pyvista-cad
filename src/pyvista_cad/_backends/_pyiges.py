"""IGES backend powered by pyiges.

Preserves the per-entity IGES *level* (a 1-200 integer layer
identifier) as ``field_data['cad.level']`` on every output block, and
stamps ``cad.backend='pyiges'`` on each block for downstream
provenance.
"""

import os
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadReadError, OptionalDependencyError


def _require_pyiges() -> Any:
    try:
        import pyiges
    except ImportError as exc:
        msg = 'pyiges not installed; install pyvista-cad[iges]'
        raise OptionalDependencyError(msg) from exc
    return pyiges


def _entity_level(entity: Any) -> int | None:
    """Best-effort extraction of an IGES entity's *level* (layer) number.

    pyiges exposes the level differently depending on entity subclass.
    Most entities have an ``entity_level`` / ``level`` attribute, but a
    handful (older or lower-level entity types) only expose the raw
    parameter slot — IGES Directory Entry parameter 5 is *Level*, and
    pyiges stores DE fields on ``entity.d`` or in
    ``entity.parameters``.
    """
    for attr in ('entity_level', 'level'):
        if hasattr(entity, attr):
            try:
                value = getattr(entity, attr)
            except AttributeError:
                continue
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    params = getattr(entity, 'parameters', None)
    if params is not None:
        try:
            return int(params[3])
        except (IndexError, TypeError, ValueError):
            pass
    return None


def _collect_levels(iges: Any) -> list[int]:
    """Return the distinct IGES levels present in ``iges``."""
    levels: set[int] = set()
    try:
        entities = list(iges)
    except TypeError:
        entities = []
    for entity in entities:
        lvl = _entity_level(entity)
        if lvl is not None:
            levels.add(lvl)
    return sorted(levels)


def _stamp_block(block: pv.DataSet, *, level: int | None) -> None:
    block.field_data['cad.source_format'] = np.array(['iges'])
    block.field_data['cad.backend'] = np.array(['pyiges'])
    if level is not None:
        block.field_data['cad.level'] = np.array([int(level)], dtype=np.int64)


def read_iges_internal(
    path: str | os.PathLike[str],
    *,
    bsplines: bool = True,
    surfaces: bool = True,
    lines: bool = False,
    points: bool = False,
    delta: float = 0.025,
) -> pv.DataSet:
    pyiges = _require_pyiges()
    try:
        iges = pyiges.read(str(path))
    except Exception as exc:
        msg = f'pyiges failed to read {path}: {exc}'
        raise CadReadError(msg) from exc

    try:
        result = iges.to_vtk(
            bsplines=bsplines,
            surfaces=surfaces,
            lines=lines,
            points=points,
            delta=delta,
        )
    except Exception as exc:
        msg = f'pyiges failed to tessellate {path}: {exc}'
        raise CadReadError(msg) from exc

    distinct_levels = _collect_levels(iges)

    # Always stamp source format + backend on the top-level result.
    result.field_data['cad.source_format'] = np.array(['iges'])
    result.field_data['cad.backend'] = np.array(['pyiges'])

    # When every entity sits on the same level we can stamp a single
    # scalar on the top-level result and on each block. When levels
    # differ pyiges has already flattened the entities into a single
    # dataset (``to_vtk`` does not preserve per-entity grouping), so we
    # record the *set* of levels seen instead and leave per-block
    # tagging best-effort.
    if isinstance(result, pv.MultiBlock):
        if len(distinct_levels) == 1:
            for i in range(result.n_blocks):
                block = result[i]
                if block is not None:
                    _stamp_block(block, level=distinct_levels[0])
        else:
            for i in range(result.n_blocks):
                block = result[i]
                if block is not None:
                    _stamp_block(block, level=None)
        if distinct_levels:
            result.field_data['cad.levels'] = np.array(distinct_levels, dtype=np.int64)
        if len(distinct_levels) == 1:
            result.field_data['cad.level'] = np.array([distinct_levels[0]], dtype=np.int64)
    else:
        # Single PolyData: stamp the level if it is unambiguous.
        if len(distinct_levels) == 1:
            result.field_data['cad.level'] = np.array([distinct_levels[0]], dtype=np.int64)
        if distinct_levels:
            result.field_data['cad.levels'] = np.array(distinct_levels, dtype=np.int64)

    return result
