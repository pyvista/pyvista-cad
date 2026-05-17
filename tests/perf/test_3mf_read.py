"""3MF read regression test (per-vertex throughput)."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def three_mf_50k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a ~50k-vertex 3MF fixture via lib3mf direct calls."""
    lib3mf = pytest.importorskip('lib3mf')

    wrapper = lib3mf.Wrapper()
    model = wrapper.CreateModel()
    mesh = model.AddMeshObject()
    mesh.SetName('perf')

    n_verts = 50_000
    rng = np.random.default_rng(0)
    pts = rng.random((n_verts, 3), dtype=np.float64)

    Position = lib3mf.Position  # noqa: N806
    Triangle = lib3mf.Triangle  # noqa: N806
    for x, y, z in pts:
        v = Position()
        v.Coordinates[0] = float(x)
        v.Coordinates[1] = float(y)
        v.Coordinates[2] = float(z)
        mesh.AddVertex(v)
    # One degenerate-but-valid triangle per ~3 vertices keeps the file
    # representative without inflating the mesh beyond the vertex budget.
    n_tris = n_verts // 3
    for i in range(n_tris):
        t = Triangle()
        t.Indices[0] = 3 * i
        t.Indices[1] = 3 * i + 1
        t.Indices[2] = 3 * i + 2
        mesh.AddTriangle(t)

    model.AddBuildItem(mesh, wrapper.GetIdentityTransform())

    out = tmp_path_factory.mktemp('three_mf_50k') / 'perf.3mf'
    writer = model.QueryWriter('3mf')
    writer.WriteToFile(str(out))
    return out


@pytest.mark.benchmark
def test_3mf_read_per_vertex(benchmark, three_mf_50k: Path) -> None:
    """``read_three_mf`` must stay under 5 us/vertex p50."""
    pytest.importorskip('lib3mf')
    from pyvista_cad import read_three_mf

    result = benchmark.pedantic(read_three_mf, args=(three_mf_50k,), rounds=5, iterations=1)
    assert result is not None
    p50 = benchmark.stats['median']
    per_vert = p50 / 50_000.0
    assert per_vert < 5e-6, f'3MF read p50 {per_vert * 1e6:.2f} us/vertex exceeds 5 us budget'
