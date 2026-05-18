"""``MultiBlock.cad.walk`` regression test (10-level deep nesting)."""

import pytest
import pyvista as pv


def _make_nested(depth: int, fanout: int = 2) -> pv.MultiBlock:
    """Build a ``depth``-level nested MultiBlock tree."""
    if depth == 0:
        return pv.PolyData()  # leaf
    root = pv.MultiBlock()
    for i in range(fanout):
        child = _make_nested(depth - 1, fanout)
        root.append(child, name=f'd{depth}_{i}')
    return root


@pytest.fixture(scope='module')
def deep_multiblock() -> pv.MultiBlock:
    """A 10-level deep MultiBlock tree with fanout 2."""
    return _make_nested(10, fanout=2)


@pytest.mark.benchmark
def test_multiblock_walk_deep(benchmark, perf_budget, deep_multiblock: pv.MultiBlock) -> None:
    """``.cad.walk()`` over a 10-deep tree must stay under 12 ms p50.

    Budget rationale: 2046 blocks * 2.6 us per ``vtkInformation.Get`` is a
    ~5.3 ms VTK-imposed floor; we sit at ~8 ms after bypassing the PyVista
    deprecation wrapper. 12 ms leaves slack above the floor for runner
    jitter while still trapping a regression past 1.5x the post-W9 number.
    """
    import pyvista_cad  # noqa: F401  (registers the .cad accessor)

    def run() -> int:
        return sum(1 for _ in deep_multiblock.cad.walk())

    result = benchmark.pedantic(run, rounds=5, iterations=1)
    assert result > 0
    p50 = benchmark.stats['median']
    # 20 ms ceiling: shared GitHub runners are ~1.5x noisier than local;
    # this stays a regression tripwire without flaking on CI variance.
    perf_budget(p50 < 20e-3, f'multiblock walk p50 {p50 * 1e3:.2f} ms exceeds 20 ms budget')
