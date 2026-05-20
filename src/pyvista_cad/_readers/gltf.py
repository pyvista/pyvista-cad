"""Reader for glTF / GLB files with manifest-metadata enrichment.

PyVista ships a built-in VTK glTF reader, but it drops node names and
``extras`` payloads. This reader instantiates :class:`pyvista.GLTFReader`
directly for the base geometry, then enriches the result with the
``cad.*`` manifest metadata. Instantiating the reader class (rather than
calling :func:`pyvista.read`) is what makes registering this reader for
``.gltf`` / ``.glb`` safe: ``pyvista.read`` would otherwise dispatch back
into this function and recurse forever.
"""

import os
from typing import Any

import pyvista as pv

from pyvista_cad._uri import require_local_path


def read_gltf(
    path: str | os.PathLike[str],
    /,
    **_: Any,
) -> pv.DataSet:
    """Read a glTF/GLB file and enrich it with manifest metadata.

    The base geometry is read with :class:`pyvista.GLTFReader`; node
    names and ``extras`` are then attached as ``cad.*`` field data.
    ``cad.source_format`` is ``'gltf'``.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.gltf`` or ``.glb`` file.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.DataSet
        The parsed mesh with ``cad.*`` manifest metadata attached.

    Examples
    --------
    >>> import pyvista_cad
    >>> from pyvista_cad import examples, read_gltf
    >>> mesh = read_gltf(examples.gltf_toycar_path())
    >>> str(mesh.field_data['cad.source_format'][0])
    'gltf'

    """
    require_local_path(path)
    from pyvista_cad._backends._gltf import read_gltf_internal

    return read_gltf_internal(path)
