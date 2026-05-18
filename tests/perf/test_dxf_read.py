"""DXF read regression test (per-entity throughput)."""

from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def dxf_10k(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a 10k-LINE DXF fixture via ezdxf."""
    ezdxf = pytest.importorskip('ezdxf')
    doc = ezdxf.new()
    msp = doc.modelspace()
    for i in range(10_000):
        x = float(i % 100)
        y = float(i // 100)
        msp.add_line((x, y, 0.0), (x + 1.0, y + 1.0, 0.0))
    out = tmp_path_factory.mktemp('dxf10k') / 'lines.dxf'
    doc.saveas(out)
    return out


@pytest.mark.benchmark
def test_dxf_read_per_entity(benchmark, perf_budget, dxf_10k: Path) -> None:
    """``read_dxf`` must stay under 100 us/entity p50."""
    pytest.importorskip('ezdxf')
    from pyvista_cad import read_dxf

    result = benchmark.pedantic(read_dxf, args=(dxf_10k,), rounds=5, iterations=1)
    assert result is not None
    p50 = benchmark.stats['median']
    per_entity = p50 / 10_000.0
    perf_budget(
        per_entity < 100e-6,
        f'DXF read p50 {per_entity * 1e6:.1f} us/entity exceeds 100 us budget',
    )
