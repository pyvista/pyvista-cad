"""Property-based tests for invariants across the public API.

Uses Hypothesis to drive randomized inputs through the unit table,
DXF INSUNITS round-trip, OCCT volume round-trip, layer split partition,
and MultiBlock walk traversal. Catches drift that example-based tests
miss.
"""

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._metadata import (
    _INSUNITS_TO_STR,
    _TO_MM,
    _VALID_UNITS,
    insunits_code,
    insunits_to_str,
)

# Units we can actually convert through the _TO_MM table (i.e. not 'unitless').
_CONVERTIBLE_UNITS = sorted(_VALID_UNITS - {'unitless'})


# Fixed expected conversion factors, computed from the SI definitions of each
# unit. Each entry is ``(src, dst, factor)`` such that ``1.0 src == factor dst``.
_EXPECTED_CONVERSIONS: list[tuple[str, str, float]] = [
    ('inch', 'mm', 25.4),
    ('mm', 'inch', 1.0 / 25.4),
    ('foot', 'mm', 304.8),
    ('foot', 'inch', 12.0),
    ('yard', 'foot', 3.0),
    ('yard', 'mm', 914.4),
    ('mile', 'foot', 5280.0),
    ('mile', 'mm', 1_609_344.0),
    ('cm', 'mm', 10.0),
    ('m', 'mm', 1000.0),
    ('m', 'cm', 100.0),
    ('km', 'm', 1000.0),
    ('km', 'mm', 1_000_000.0),
    ('micron', 'mm', 1e-3),
    ('nm', 'mm', 1e-6),
    ('nm', 'micron', 1e-3),
    # identity for each unit
    *[(u, u, 1.0) for u in _CONVERTIBLE_UNITS],
]


@pytest.mark.parametrize(('src', 'dst', 'factor'), _EXPECTED_CONVERSIONS)
def test_unit_conversion_fixed_pairs(src: str, dst: str, factor: float) -> None:
    """Each (src, dst) conversion factor matches the SI ground-truth value.

    Sweeps every supported unit at least once (identity entries) and pins
    representative cross-pairs against externally-verifiable constants.
    """
    computed = _TO_MM[src] / _TO_MM[dst]
    assert computed == pytest.approx(factor, rel=1e-12)
    # And the same factor applied to a value of 1.0.
    assert (1.0 * _TO_MM[src] / _TO_MM[dst]) == pytest.approx(factor, rel=1e-12)


def test_unit_conversion_covers_every_unit() -> None:
    """The fixed-pair sweep exercises every convertible unit (anti-drift guard)."""
    seen = {u for pair in _EXPECTED_CONVERSIONS for u in pair[:2]}
    assert seen == set(_CONVERTIBLE_UNITS)


@given(code=st.sampled_from(sorted(_INSUNITS_TO_STR.keys())))
def test_dxf_insunits_round_trip(code: int) -> None:
    """For codes that map to a real (non-unitless) unit, ``insunits_to_str``
    composed with ``insunits_code`` returns the same code.

    Codes 8, 9, 11, and 14-20 collapse to ``'unitless'`` by design and so
    only round-trip back to 0 (one-way).
    """
    unit = insunits_to_str(code)
    if unit == 'unitless' and code != 0:
        # One-way collapse — assert we land at 0, not at the original code.
        assert insunits_code(unit) == 0
        return
    assert insunits_code(unit) == code


@pytest.mark.filterwarnings('ignore::hypothesis.errors.HypothesisWarning')
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=10)
@given(
    w=st.floats(min_value=0.1, max_value=100.0),
    h=st.floats(min_value=0.1, max_value=100.0),
    d=st.floats(min_value=0.1, max_value=100.0),
)
def test_topods_polydata_round_trip_volume(w: float, h: float, d: float) -> None:
    """``from_build123d(Box) -> to_topods -> VolumeProperties`` matches ``w*h*d``.

    Sewing tolerance allows up to 5% slack on either side.
    """
    b3d = pytest.importorskip('build123d')
    pytest.importorskip('OCP')
    from OCP.BRepGProp import BRepGProp  # noqa: PLC0415
    from OCP.GProp import GProp_GProps  # noqa: PLC0415

    mesh = pyvista_cad.from_build123d(b3d.Box(w, h, d))
    shape = pyvista_cad.to_topods(mesh)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    expected = w * h * d
    assert props.Mass() == pytest.approx(expected, rel=0.05)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_cells=st.integers(min_value=1, max_value=50),
    n_layers=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_split_by_layer_partition(n_cells: int, n_layers: int, seed: int) -> None:
    """``split_by_layer`` partitions cells: union covers every cell exactly once."""
    rng = np.random.default_rng(seed)
    # Build a PolyData with n_cells triangles.
    pts = rng.random((3 * n_cells, 3)).astype(np.float32)
    faces = np.empty(4 * n_cells, dtype=np.int64)
    faces[0::4] = 3
    faces[1::4] = np.arange(0, 3 * n_cells, 3)
    faces[2::4] = np.arange(1, 3 * n_cells, 3)
    faces[3::4] = np.arange(2, 3 * n_cells, 3)
    poly = pv.PolyData(pts, faces)
    layer_labels = rng.integers(0, n_layers, size=n_cells)
    poly.cell_data['cad.layer'] = np.array([f'L{i}' for i in layer_labels])

    mb = poly.cad.split_by_layer()
    total = sum(block.n_cells for block in mb if block is not None)
    assert total == n_cells

    # extract_cells writes vtkOriginalCellIds when more than one cell is
    # extracted; for single-cell blocks pyvista skips the array. Tolerate
    # the missing-array case but still demand cell-count consistency.
    seen: list[int] = []
    for block in mb:
        if block is None:
            continue
        if 'vtkOriginalCellIds' in block.cell_data:
            ids = np.asarray(block.cell_data['vtkOriginalCellIds'])
            seen.extend(int(i) for i in ids)
    if seen:
        # When IDs are present we get a true partition: every cell exactly once.
        assert sorted(seen) == sorted(set(seen))
        for sid in seen:
            assert 0 <= sid < n_cells

    # Each block must be homogeneous in cad.layer and equal to its block name.
    for name, block in zip(mb.keys(), mb, strict=True):
        if block is None or block.n_cells == 0:
            continue
        block_layers = {str(x) for x in block.cell_data['cad.layer']}
        assert block_layers == {name}, f'block {name!r} contained mixed layers {block_layers}'


def _build_random_multiblock(
    rng: np.random.Generator,
    depth: int,
    max_breadth: int,
    prefix: str = '',
) -> tuple[pv.MultiBlock, set[str]]:
    """Build a random nested MultiBlock; return ``(mb, label_paths)``.

    ``label_paths`` is the set of every DFS path that ``cad.walk()`` is expected
    to yield — one entry per container and per leaf. Names are assigned
    explicitly so we don't depend on PyVista's auto-naming convention
    (which varies across versions: ``Block-00`` vs ``block_0`` vs
    ``Block_0``).
    """
    mb = pv.MultiBlock()
    labels: set[str] = set()
    breadth = int(rng.integers(1, max_breadth + 1))
    for i in range(breadth):
        name = f'n{i}'
        path = f'{prefix}/{name}' if prefix else name
        labels.add(path)
        if depth > 0 and rng.random() < 0.5:
            child, child_labels = _build_random_multiblock(
                rng, depth - 1, max_breadth, prefix=path
            )
            mb.append(child, name)
            labels |= child_labels
        else:
            mb.append(pv.Sphere(), name)
    return mb, labels


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=30)
@given(
    depth=st.integers(min_value=0, max_value=3),
    max_breadth=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_walk_visits_every_leaf(depth: int, max_breadth: int, seed: int) -> None:
    """``mb.cad.walk()`` yields exactly the constructed (container + leaf) paths."""
    rng = np.random.default_rng(seed)
    mb, expected_labels = _build_random_multiblock(rng, depth, max_breadth)
    visited = list(mb.cad.walk())
    visited_labels = {path for path, _ in visited}
    assert visited_labels == expected_labels
    # And no duplicates.
    assert len(visited) == len(visited_labels)
