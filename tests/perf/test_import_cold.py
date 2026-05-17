"""Marginal cold-import regression test.

Absolute ``import pyvista_cad`` wall time is dominated by interpreter
startup plus ``import pyvista`` (200+ ms on the runners we have measured)
and the pytest-benchmark harness, none of which we control. An absolute
budget therefore cannot catch a ``pyvista_cad`` eager-import regression.

Instead we measure the *marginal* cost of ``import pyvista_cad`` over
``import pyvista``: two subprocesses, one importing only ``pyvista`` and
one importing ``pyvista`` then ``pyvista_cad``, both timed with
``time.perf_counter`` *inside* the child around the imports only. The
shared interpreter-startup and pyvista cost cancels in the difference,
isolating our package's top-level work (accessor registration,
``_errors``, lazy-import shims). A measured ~18 ms marginal on the
reference runner; the gate is 60 ms to absorb CI noise while still
catching a real eager-import regression (e.g. a backend leaking onto the
top-level path).
"""

import statistics
import subprocess
import sys

import pytest

# Times only the import statements in the child, printing the elapsed
# seconds to stdout so parent-side subprocess-spawn cost is excluded.
_BASE_STMT = 'import time;_t0=time.perf_counter();import pyvista;print(time.perf_counter()-_t0)'
_PKG_STMT = (
    'import time;'
    'import pyvista;'  # warm pyvista before timing so only the delta is measured
    '_t0=time.perf_counter();'
    'import pyvista_cad;'
    'print(time.perf_counter()-_t0)'
)


def _timed_import(stmt: str) -> float:
    out = subprocess.run(
        [sys.executable, '-c', stmt],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


@pytest.mark.benchmark
def test_import_cold_marginal(benchmark) -> None:
    """``import pyvista_cad`` must add no more than 60 ms over ``import pyvista``.

    ``_BASE_STMT`` times ``import pyvista`` from a cold interpreter.
    ``_PKG_STMT`` warms ``pyvista`` first, then times ``import
    pyvista_cad`` alone, so the result is exactly our package's marginal
    top-level cost. The gate fails if that marginal exceeds 60 ms.
    """
    base_samples: list[float] = []
    pkg_marginal_samples: list[float] = []

    def run() -> None:
        base_samples.append(_timed_import(_BASE_STMT))
        pkg_marginal_samples.append(_timed_import(_PKG_STMT))

    benchmark.pedantic(run, rounds=5, iterations=1)

    base_p50 = statistics.median(base_samples)
    marginal_p50 = statistics.median(pkg_marginal_samples)
    assert marginal_p50 < 0.06, (
        f'pyvista_cad marginal cold-import {marginal_p50 * 1e3:.1f} ms '
        f'exceeds 60 ms (pyvista floor={base_p50 * 1e3:.1f} ms, measured '
        f'over {len(pkg_marginal_samples)} rounds)'
    )
