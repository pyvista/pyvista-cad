"""STEP reader backed by cascadio (lightweight alternative path).

cascadio loads STEP and emits a glTF binary that PyVista can parse via
its native glTF reader. We strip the data back out into a MultiBlock
keyed on cascadio's mesh names.
"""

import os
import pathlib
import tempfile
from typing import Any

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadReadError, OptionalDependencyError


def read_step(
    path: str | os.PathLike[str],
    *,
    linear_deflection: float,
    angular_deflection: float,
    merge: bool,
) -> pv.MultiBlock | pv.PolyData:
    """Read a STEP file via :mod:`cascadio` (read-only, no OCP).

    Returns
    -------
    pyvista.MultiBlock or pyvista.PolyData
        ``MultiBlock`` of parts, or a single ``PolyData`` when
        ``merge`` is true.

    """
    try:
        import cascadio
    except ImportError as exc:
        msg = 'cascadio not installed; install pyvista-cad[step-light]'
        raise OptionalDependencyError(msg) from exc

    fname = os.fspath(path)
    if not os.path.exists(fname):
        msg = f'No such file: {fname}'
        raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory() as tmp:
        glb_path = pathlib.Path(tmp) / 'out.glb'
        try:
            cascadio.step_to_glb(
                fname,
                str(glb_path),
                tol_linear=linear_deflection,
                tol_angular=angular_deflection,
            )
        except Exception as exc:
            msg = f'cascadio failed to read STEP {fname}: {exc}'
            raise CadReadError(msg) from exc
        loaded: Any = pv.read(str(glb_path))

    mb = _flatten_to_multiblock(loaded)
    mb.field_data['cad.source_format'] = np.array(['step'])
    mb.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
    mb.field_data['cad.backend'] = np.array(['cascadio'])
    for i in range(len(mb)):
        block = mb[i]
        if block is None:
            continue
        block.field_data['cad.source_format'] = np.array(['step'])
        block.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
        block.field_data['cad.backend'] = np.array(['cascadio'])
        name = mb.get_block_name(i)
        if name:
            block.field_data['cad.label'] = np.array([name])

    if merge:
        combined = mb.combine(merge_points=True).extract_surface(algorithm='dataset_surface')
        combined.field_data['cad.source_format'] = np.array(['step'])
        combined.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
        combined.field_data['cad.backend'] = np.array(['cascadio'])
        return combined
    return mb


def _flatten_to_multiblock(obj: Any) -> pv.MultiBlock:
    """Coerce a PyVista glTF read result into a flat MultiBlock."""
    if isinstance(obj, pv.MultiBlock):
        out = pv.MultiBlock()
        for i in range(len(obj)):
            block = obj[i]
            name = obj.get_block_name(i) or f'part_{i}'
            if isinstance(block, pv.MultiBlock):
                sub = _flatten_to_multiblock(block)
                for j in range(len(sub)):
                    sub_name = sub.get_block_name(j) or f'part_{i}_{j}'
                    out.append(sub[j], name=f'{name}/{sub_name}')
            elif block is not None:
                out.append(block, name=name)
        return out
    mb = pv.MultiBlock()
    mb.append(obj, name='part_0')
    return mb
