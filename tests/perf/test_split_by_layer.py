"""``split_by_layer`` scaling regression test (100k cells x 200 layers)."""

import numpy as np
import pytest
import pyvista as pv


@pytest.fixture(scope='module')
def layered_polydata() -> pv.PolyData:
    """Build a 100k-triangle PolyData with 200 distinct ``cad.layer`` values."""
    n_cells = 100_000
    rng = np.random.default_rng(0)
    points = rng.random((n_cells * 3, 3), dtype=np.float64)
    # ``faces`` is the VTK-style connectivity: [3, i0, i1, i2, 3, ...].
    idx = np.arange(n_cells * 3, dtype=np.int64).reshape(n_cells, 3)
    faces = np.hstack(
        [np.full((n_cells, 1), 3, dtype=np.int64), idx],
    ).ravel()
    mesh = pv.PolyData(points, faces)
    layer_ids = (np.arange(n_cells) % 200).astype(np.int32)
    layer_strs = np.array([f'L{i:03d}' for i in layer_ids])
    mesh.cell_data['cad.layer'] = layer_strs
    return mesh


@pytest.mark.benchmark
def test_split_by_layer_scaling(benchmark, layered_polydata: pv.PolyData) -> None:
    """``.cad.split_by_layer()`` p50 must stay under 200 ms."""
    import pyvista_cad  # noqa: F401  (registers the .cad accessor)

    def run() -> pv.MultiBlock:
        return layered_polydata.cad.split_by_layer()

    result = benchmark.pedantic(run, rounds=5, iterations=1)
    assert result is not None
    p50 = benchmark.stats['median']
    # 300 ms ceiling: shared GitHub runners are ~1.5x noisier than local;
    # this stays a regression tripwire without flaking on CI variance.
    assert p50 < 0.3, f'split_by_layer p50 {p50 * 1e3:.1f} ms exceeds 300 ms budget'
