"""Guard: ``import pyvista_cad`` must not import any heavy CAD backend."""

import importlib
import os
import subprocess
import sys
import textwrap

import pytest

from pyvista_cad import OptionalDependencyError

FORBIDDEN = (
    'cascadio',
    'lib3mf',
    'ifcopenshell',
    'OCP',
    'build123d',
    'FreeCAD',
    'pyiges',
    'geomdl',
)


def test_import_pyvista_cad_does_not_import_heavy_deps():
    """In-process check: after `import pyvista_cad`, no forbidden module is loaded.

    Some forbidden modules may have been imported by other test modules
    earlier in the session, so this test runs in a fresh subprocess
    below for the authoritative check.
    """
    for forbidden in FORBIDDEN:
        sys.modules.pop(forbidden, None)
    sys.modules.pop('pyvista_cad', None)
    importlib.import_module('pyvista_cad')
    leaked = [m for m in FORBIDDEN if m in sys.modules]
    assert not leaked, (
        f'pyvista_cad eagerly imported {leaked}; every heavy backend must be lazy-imported.'
    )


def test_import_pyvista_cad_in_subprocess_does_not_import_heavy_deps():
    """Subprocess check is the authoritative guarantee.

    Spawns a clean Python, imports pyvista_cad, and inspects sys.modules.
    """
    script = textwrap.dedent(
        f"""
        import sys
        import pyvista_cad  # noqa: F401
        forbidden = {FORBIDDEN!r}
        leaked = [m for m in forbidden if m in sys.modules]
        if leaked:
            raise SystemExit(f'LEAKED: {{leaked}}')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}'
    )


def test_import_pyvista_cad_is_fast():
    """`import pyvista_cad` should complete under 500 ms on a cold cache.

    The original target of < 100 ms was renegotiated because
    ``import pyvista`` itself costs ~197 ms on a modern Linux/CPython
    3.13 host. Measured cold-import of ``pyvista_cad`` runs ~180-210 ms
    in this repo; 500 ms is a CI-noise-tolerant ceiling that still
    catches the kind of regression (e.g. eager-importing OCP) the
    no-heavy-deps invariant exists to prevent.

    Under pytest-xdist the budget is relaxed to 1.0 s because parallel
    workers contend for CPU on 2-core CI runners (Windows especially).
    The dedicated ``lazy-imports`` CI job runs serial and still enforces
    the 0.5 s budget, so regressions are still caught.
    """
    budget = 1.0 if os.environ.get('PYTEST_XDIST_WORKER') else 0.5
    script = textwrap.dedent(
        """
        import time
        t0 = time.perf_counter()
        import pyvista_cad  # noqa: F401
        t1 = time.perf_counter()
        print(f'{t1 - t0:.4f}')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = float(result.stdout.strip())
    assert elapsed < budget, f'import pyvista_cad took {elapsed:.3f}s (budget: {budget}s)'


# Each entry: (id, trigger, modules-to-mask, extra-token, lib-token).
def _trigger_freecad_ocp() -> None:
    from pyvista_cad._backends._freecad import _require_ocp

    _require_ocp()


def _trigger_ifc() -> None:
    from pyvista_cad._backends._ifcopenshell import _require_ifc

    _require_ifc()


def _trigger_lib3mf() -> None:
    from pyvista_cad._backends._lib3mf import _require_lib3mf

    _require_lib3mf()


def _trigger_pyiges() -> None:
    from pyvista_cad._backends._pyiges import _require_pyiges

    _require_pyiges()


def _trigger_cascadio() -> None:
    from pyvista_cad._backends._cascadio import read_step

    read_step(
        'does-not-exist.step',
        linear_deflection=0.1,
        angular_deflection=0.5,
        merge=False,
    )


def _trigger_build123d_backend() -> None:
    from pyvista_cad._backends._build123d import read_step

    read_step(
        'does-not-exist.step',
        linear_deflection=0.1,
        angular_deflection=0.5,
        merge=False,
    )


def _trigger_brep_ocp() -> None:
    from pyvista_cad._readers.brep import _require_ocp

    _require_ocp()


def _trigger_step_write() -> None:
    import pyvista as pv

    from pyvista_cad import write_step

    write_step(pv.Sphere(), '/tmp/_lazy_import_test.step')


def _trigger_step_read_no_backends() -> None:
    from pyvista_cad._readers.step import _resolve_backend

    _resolve_backend(None)


def _trigger_bridge_build123d_from() -> None:
    import pyvista as pv

    from pyvista_cad import from_build123d

    from_build123d(pv.Sphere())


def _trigger_bridge_build123d_to() -> None:
    import pyvista as pv

    from pyvista_cad import to_build123d

    to_build123d(pv.Sphere())


def _trigger_bridge_topods_from() -> None:
    from pyvista_cad import from_topods

    from_topods(object())


def _trigger_bridge_topods_to() -> None:
    import pyvista as pv

    from pyvista_cad import to_topods

    to_topods(pv.Sphere())


LAZY_IMPORT_CASES = [
    ('freecad_ocp', _trigger_freecad_ocp, ('OCP',), '[step]', 'OCP'),
    ('ifc', _trigger_ifc, ('ifcopenshell',), '[ifc]', 'ifcopenshell'),
    ('lib3mf', _trigger_lib3mf, ('lib3mf',), '[3mf]', 'lib3mf'),
    ('pyiges', _trigger_pyiges, ('pyiges',), '[iges]', 'pyiges'),
    ('cascadio', _trigger_cascadio, ('cascadio',), '[step-light]', 'cascadio'),
    ('build123d_backend', _trigger_build123d_backend, ('build123d',), '[step]', 'build123d'),
    ('brep_ocp', _trigger_brep_ocp, ('OCP',), '[step]', 'OCP'),
    ('step_write_ocp', _trigger_step_write, ('OCP',), '[step]', 'OCP'),
    (
        'step_read_no_backends',
        _trigger_step_read_no_backends,
        ('build123d', 'cascadio'),
        '[step',  # matches either [step] or [step-light]
        'build123d',
    ),
    ('bridge_b3d_from', _trigger_bridge_build123d_from, ('build123d',), '[step]', 'build123d'),
    ('bridge_b3d_to', _trigger_bridge_build123d_to, ('build123d',), '[step]', 'build123d'),
    ('bridge_topods_from', _trigger_bridge_topods_from, ('OCP',), '[step]', 'OCP'),
    ('bridge_topods_to', _trigger_bridge_topods_to, ('OCP',), '[step]', 'OCP'),
]


@pytest.mark.parametrize(
    ('case_id', 'trigger', 'mask_modules', 'extra_token', 'lib_token'),
    [(c[0], c[1], c[2], c[3], c[4]) for c in LAZY_IMPORT_CASES],
    ids=[c[0] for c in LAZY_IMPORT_CASES],
)
def test_lazy_import_raises_optional_dependency_error(
    case_id: str,
    trigger: object,
    mask_modules: tuple[str, ...],
    extra_token: str,
    lib_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each lazy-import site must raise OptionalDependencyError citing the extra."""
    for mod in mask_modules:
        # Drop any already-loaded version of the module, then mask it so
        # subsequent `import mod` raises ImportError without uninstalling.
        for loaded in list(sys.modules):
            if loaded == mod or loaded.startswith(f'{mod}.'):
                monkeypatch.setitem(sys.modules, loaded, None)
        monkeypatch.setitem(sys.modules, mod, None)

    with pytest.raises(OptionalDependencyError) as excinfo:
        trigger()  # type: ignore[operator]

    msg = str(excinfo.value)
    assert f'pyvista-cad{extra_token}' in msg, (
        f'{case_id}: expected {extra_token!r} install hint, got: {msg!r}'
    )
    assert lib_token in msg, (
        f'{case_id}: expected {lib_token!r} mentioned in message, got: {msg!r}'
    )


def test_openscad_missing_binary_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenSCAD has no Python dep; missing-binary path must name the CLI binary."""
    import shutil

    from pyvista_cad._backends._openscad import _resolve_openscad

    monkeypatch.setattr(shutil, 'which', lambda _name: None)
    with pytest.raises(OptionalDependencyError) as excinfo:
        _resolve_openscad(None)
    msg = str(excinfo.value)
    assert 'openscad' in msg.lower(), msg
    assert 'command-line binary on PATH' in msg, msg
