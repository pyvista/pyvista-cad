"""DXF (AutoCAD Drawing Exchange Format) reader and writer.

The DXF reader handles every drawable entity ``ezdxf`` can decompose:
``POINT``, ``LINE``, ``LWPOLYLINE``, ``POLYLINE``, ``CIRCLE``, ``ARC``,
``ELLIPSE``, ``SPLINE``, ``HELIX``, ``3DFACE``, ``SOLID``, ``TRACE``,
``MESH``, and closed ``POLYFACE`` / ``POLYMESH``. Curves are flattened
to polylines using ``ezdxf``'s tessellation. ACIS-encoded solids
(``REGION`` / ``BODY`` / ``3DSOLID``) cannot be decoded and emit a
``UserWarning``.

The originating layer is stored on every cell in the ``Layer`` and
``cad.layer`` cell-data arrays.
"""

import os
from typing import Any

import pyvista as pv


@pv.register_reader('.dxf')
def read_dxf(
    path: str | os.PathLike[str],
    /,
    *,
    clean: bool = True,
    keep_text: bool = False,
    expand_blocks: bool = True,
    layout: str = 'modelspace',
    **_: Any,
) -> pv.PolyData:
    """Read a DXF file as a :class:`pyvista.PolyData`.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.dxf`` file (ASCII or binary, any AutoCAD version
        supported by :mod:`ezdxf`).
    clean : bool, default: True
        Run :meth:`pyvista.PolyData.clean` on the result to merge
        coincident vertices and drop degenerate cells. DXF files
        almost always store coincident vertices per face, so cleaning
        is on by default.
    keep_text : bool, default: False
        Include ``TEXT`` / ``MTEXT`` / ``ATTRIB`` entities by
        tessellating their bounding rectangles into faces; the text
        content is stored in a ``Text`` cell-data string array.
    expand_blocks : bool, default: True
        Recursively expand ``INSERT`` block references into real
        geometry.
    layout : str, default: ``'modelspace'``
        ``'modelspace'``, ``'paperspace'``, or a specific layout name.
    **_ : Any
        Additional keyword arguments are accepted for forward
        compatibility with :func:`pyvista.read` and ignored here.

    Returns
    -------
    pyvista.PolyData
        Aggregated mesh containing every drawable entity. A ``Layer``
        cell-data string array records the originating DXF layer; the
        same data is mirrored to ``cad.layer`` for forward compat.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    pyvista_cad.CadReadError
        If the file cannot be parsed as a DXF.

    Examples
    --------
    Read the bundled DXF plate fixture:

    >>> import pyvista_cad
    >>> from pyvista_cad import read_dxf
    >>> mesh = read_dxf(pyvista_cad.examples.dxf_plate)
    >>> type(mesh).__name__
    'PolyData'
    >>> str(mesh.field_data['cad.source_format'][0])
    'dxf'

    """
    from pyvista_cad._backends._ezdxf import read_dxf_internal

    return read_dxf_internal(
        path,
        clean=clean,
        keep_text=keep_text,
        expand_blocks=expand_blocks,
        layout=layout,
    )


def write_dxf(
    mesh: pv.PolyData,
    path: str | os.PathLike[str],
    /,
    *,
    layer: str = '0',
    layer_array: str | None = 'Layer',
    dxf_version: str = 'R2018',
    units: str | None = None,
    color_index: int | None = None,
    **_: Any,
) -> None:
    """Write a :class:`pyvista.PolyData` as a DXF file using ezdxf.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Source mesh. Vertex cells become ``POINT``, line cells become
        ``LINE`` (2 points) or ``POLYLINE`` (more), triangular and
        quadrilateral faces become ``3DFACE``, and higher-degree
        polygons become DXF ``MESH`` entities.
    path : str or os.PathLike
        Destination ``.dxf`` path.
    layer : str, default: ``'0'``
        Default DXF layer when ``layer_array`` is unset or missing.
    layer_array : str or None, default: ``'Layer'``
        Name of a cell-data array to use for per-cell layer assignment.
        When ``None``, every cell goes on ``layer``.
    dxf_version : str, default: ``'R2018'``
        ezdxf-supported version string.
    units : str or None, optional
        Canonical unit string (``'mm'``, ``'inch'``, ...). When
        omitted, ``field_data['cad.units']`` is consulted and falls
        back to unitless.
    color_index : int or None, optional
        DXF color index (0-256) applied to every entity. When ``None``,
        DXF's default ``BYLAYER`` color resolution is used.
    **_ : Any
        Additional keyword arguments are accepted for forward
        compatibility and ignored here.

    Returns
    -------
    None
        The file is written to ``path``.

    Raises
    ------
    pyvista_cad.CadWriteError
        If ``mesh`` is not a :class:`pyvista.PolyData` or if writing
        fails on disk.

    Examples
    --------
    Write a sphere to a DXF file in a temporary directory:

    >>> import os
    >>> import tempfile
    >>> import pyvista as pv
    >>> from pyvista_cad import write_dxf
    >>> tmp = tempfile.mkdtemp()
    >>> out = os.path.join(tmp, 'sphere.dxf')
    >>> write_dxf(pv.Sphere(), out)
    >>> os.path.exists(out)
    True

    """
    from pyvista_cad._backends._ezdxf import write_dxf_internal

    write_dxf_internal(
        mesh,
        path,
        layer=layer,
        layer_array=layer_array,
        dxf_version=dxf_version,
        units=units,
        color_index=color_index,
    )
