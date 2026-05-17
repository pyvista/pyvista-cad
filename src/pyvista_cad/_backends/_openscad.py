"""OpenSCAD backend.

Compiles a ``.scad`` source file to STL using the ``openscad``
binary, then reads the STL with pyvista. OpenSCAD must be installed
and on ``PATH`` (or the binary path must be supplied explicitly).
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadReadError, OptionalDependencyError


def _resolve_openscad(binary: str | None) -> str:
    candidate = binary or shutil.which('openscad') or shutil.which('OpenSCAD')
    if not candidate:
        msg = (
            'reading .scad requires the openscad command-line binary on PATH '
            '(install OpenSCAD from your OS package manager, or pass '
            "binary='/path/to/openscad')"
        )
        raise OptionalDependencyError(msg)
    return candidate


def read_scad_internal(
    path: str | os.PathLike[str],
    *,
    binary: str | None = None,
    args: tuple[str, ...] = (),
    timeout: float | None = None,
) -> pv.PolyData:
    openscad = _resolve_openscad(binary)
    src = Path(path)
    if not src.exists():
        msg = f'.scad file not found: {path}'
        raise CadReadError(msg)

    # The manifold backend (OpenSCAD >= 2024) is dramatically faster
    # than the default CGAL kernel on boolean-heavy models. Inject it
    # unless the caller already chose a backend, and transparently fall
    # back to the default kernel if this openscad build rejects the flag.
    user_picked_backend = any(a.startswith('--backend') for a in args)
    attempts: list[tuple[str, ...]] = []
    if not user_picked_backend:
        attempts.append(('--backend=manifold', *args))
    attempts.append(args)

    with tempfile.TemporaryDirectory() as tmpdir:
        stl = Path(tmpdir) / 'out.stl'
        result = None
        for attempt_args in attempts:
            cmd = [openscad, '-o', str(stl), *attempt_args, str(src)]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                msg = f'openscad timed out after {timeout}s on {path}'
                raise CadReadError(msg) from exc
            if result.returncode == 0 and stl.exists():
                break
        assert result is not None
        if result.returncode != 0 or not stl.exists():
            msg = f'openscad failed on {path} (exit {result.returncode}): {result.stderr.strip()}'
            raise CadReadError(msg)

        poly = pv.read(stl)
    if not isinstance(poly, pv.PolyData):
        poly = poly.extract_surface(algorithm='dataset_surface')

    poly.field_data['cad.source_format'] = np.array(['scad'])
    poly.field_data['cad.backend'] = np.array(['openscad'])
    poly.field_data['cad.label'] = np.array([src.stem])
    return poly
