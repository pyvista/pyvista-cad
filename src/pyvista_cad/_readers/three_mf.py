"""3MF (`.3mf`) reader and writer."""

import os
from typing import Any

import pyvista as pv


@pv.register_reader('.3mf')
def read_three_mf(
    path: str | os.PathLike[str],
    /,
    **_: Any,
) -> pv.MultiBlock:
    """Read a 3MF file as a :class:`pyvista.MultiBlock`.

    Each ``<object>`` in the 3MF package becomes one block. Block
    names match the object name when set; otherwise ``object_<n>``.
    ``field_data['cad.units']`` records the document unit
    (``'micron'``, ``'mm'``, ``'cm'``, ``'inch'``, ``'foot'``, ``'m'``).

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.3mf`` file.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Returns
    -------
    pyvista.MultiBlock
        One block per mesh object.

    Examples
    --------
    Read the bundled two-part 3MF assembly fixture:

    >>> import pyvista_cad
    >>> from pyvista_cad import read_three_mf
    >>> mb = read_three_mf(pyvista_cad.examples.assembly_3mf_path())
    >>> type(mb).__name__
    'MultiBlock'
    >>> str(mb.field_data['cad.source_format'][0])
    '3mf'

    """
    from pyvista_cad._backends._lib3mf import read_three_mf_internal

    return read_three_mf_internal(path)


def write_three_mf(
    dataset: pv.PolyData | pv.MultiBlock,
    path: str | os.PathLike[str],
    /,
    *,
    units: str | None = None,
    **_: Any,
) -> None:
    """Write a :class:`pyvista.PolyData` or :class:`pyvista.MultiBlock` as 3MF.

    Each block becomes one 3MF mesh object. Surface meshes are
    triangulated automatically.

    Parameters
    ----------
    dataset : pyvista.PolyData or pyvista.MultiBlock
        Source dataset.
    path : str or os.PathLike
        Destination ``.3mf`` path.
    units : str or None, optional
        One of ``'micron'``, ``'mm'``, ``'cm'``, ``'inch'``, ``'foot'``,
        ``'m'``. Falls back to ``field_data['cad.units']`` if set,
        otherwise lib3mf's default.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    """
    from pyvista_cad._backends._lib3mf import write_three_mf_internal

    write_three_mf_internal(dataset, path, units=units)
