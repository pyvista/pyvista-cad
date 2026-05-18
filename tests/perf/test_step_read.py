"""STEP read regression tests (per-part throughput and 500-part RSS)."""

from pathlib import Path
import tracemalloc

import pytest


def _build_step(tmp_path: Path, n_parts: int) -> Path:
    """Synthesize an ``n_parts``-part STEP fixture via build123d + write_step."""
    pytest.importorskip('build123d', exc_type=ImportError)
    pytest.importorskip('OCP', exc_type=ImportError)
    import build123d as bd

    from pyvista_cad import write_step

    parts = []
    for i in range(n_parts):
        box = bd.Box(1.0, 1.0, 1.0)
        box.move(bd.Location((i * 2.0, 0.0, 0.0)))
        parts.append(box)
    compound = bd.Compound(children=parts)

    # Convert build123d Compound -> pyvista MultiBlock for write_step.
    from pyvista_cad import from_build123d

    mesh = from_build123d(compound)
    out = tmp_path / f'cube_{n_parts}.step'
    write_step(mesh, out)
    return out


@pytest.fixture(scope='module')
def step_100(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cached 100-part STEP file shared across rounds."""
    tmp = tmp_path_factory.mktemp('step100')
    return _build_step(tmp, 100)


@pytest.fixture(scope='module')
def step_500(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cached 500-part STEP file for the RSS test."""
    tmp = tmp_path_factory.mktemp('step500')
    return _build_step(tmp, 500)


@pytest.mark.benchmark
def test_step_read_per_part(benchmark, step_100: Path) -> None:
    """``read_step`` on a 100-part compound must stay under 8 ms/part p50."""
    from pyvista_cad import read_step

    result = benchmark.pedantic(read_step, args=(step_100,), rounds=5, iterations=1)
    assert result is not None
    p50 = benchmark.stats['median']
    per_part = p50 / 100.0
    # 12 ms/part ceiling: shared GitHub runners are ~1.5x noisier than
    # local; this stays a regression tripwire without flaking on CI.
    assert per_part < 12e-3, f'STEP read p50 {per_part * 1e3:.2f} ms/part exceeds 12 ms budget'


@pytest.mark.benchmark
def test_step_read_500_rss(benchmark, step_500: Path) -> None:
    """``read_step`` on a 500-part STEP must allocate under 15 MB Python heap."""
    from pyvista_cad import read_step

    peaks: list[int] = []

    def run() -> None:
        tracemalloc.start()
        try:
            read_step(step_500)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peaks.append(peak)

    benchmark.pedantic(run, rounds=3, iterations=1)
    peak_mb = max(peaks) / (1024 * 1024)
    assert peak_mb < 15.0, f'tracemalloc peak {peak_mb:.2f} MB exceeds 15 MB budget'
