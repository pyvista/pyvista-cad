"""STEP (`.step`, `.stp`) reader and writer.

Two read backends are supported:

- ``'build123d'`` (default when installed): full OCCT CAF metadata --
  per-part colors, labels, materials, layers, hierarchy. Requires
  ``pyvista-cad[step]``.
- ``'cascadio'``: lighter install, no OCP. Requires
  ``pyvista-cad[step-light]``.

The default backend is auto-selected based on which extra is
installed; ``build123d`` wins if both are present.

The writer emits a faceted STEP. When given a
:class:`pyvista.MultiBlock` it builds a labelled OCAF assembly so the
source hierarchy and block names survive the round trip.
"""

import os
from typing import Any, Literal

import numpy as np
import pyvista as pv

from pyvista_cad._errors import CadWriteError, OptionalDependencyError
from pyvista_cad._uri import require_local_path


def _resolve_backend(
    backend: Literal['build123d', 'cascadio'] | None,
) -> str:
    """Pick a STEP backend based on what is installed."""
    if backend is not None:
        return backend
    try:
        import build123d  # noqa: F401
    except ImportError:
        pass
    else:
        return 'build123d'
    try:
        import cascadio  # noqa: F401
    except ImportError as exc:
        msg = (
            'build123d/cascadio not installed; install pyvista-cad[step] '
            'or pyvista-cad[step-light]'
        )
        raise OptionalDependencyError(msg) from exc
    return 'cascadio'


@pv.register_reader('.step')
@pv.register_reader('.stp')
def read_step(
    path: str | os.PathLike[str],
    /,
    *,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
    merge: bool = False,
    backend: Literal['build123d', 'cascadio'] | None = None,
    **_: Any,
) -> pv.MultiBlock | pv.PolyData:
    """Read a STEP file as a :class:`pyvista.MultiBlock` (or merged PolyData).

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.step`` or ``.stp`` file.
    linear_deflection : float, default: 0.1
        Maximum chordal deviation of the tessellation from the
        underlying analytical surface (model units). The faceted mesh
        deviates from the analytic B-rep surface by at most this value.
    angular_deflection : float, default: 0.5
        Maximum angular deviation in radians between adjacent
        triangles.
    merge : bool, default: False
        When ``True``, combine every part into a single
        :class:`pyvista.PolyData`. When ``False``, return a
        :class:`pyvista.MultiBlock` with one block per STEP part.
    backend : {'build123d', 'cascadio'}, optional
        Explicit backend selection. By default, ``build123d`` is used
        when available, otherwise ``cascadio``.
    **_ : Any
        Additional keyword arguments are accepted for forward
        compatibility with :func:`pyvista.read` and ignored here.

    Returns
    -------
    pyvista.MultiBlock or pyvista.PolyData
        Tessellated STEP geometry. With ``merge=False`` a
        :class:`pyvista.MultiBlock` is returned: root field data
        ``cad.source_format`` / ``cad.reader`` / ``cad.backend`` /
        ``cad.units`` is set, per-part ``cad.color`` / ``cad.label`` /
        ``cad.transform`` is populated where the STEP carries it, and
        nested STEP sub-assemblies become nested
        :class:`pyvista.MultiBlock` blocks (recursion is unbounded over
        the CAF assembly tree). With ``merge=True`` the parts are
        flattened by :meth:`pyvista.MultiBlock.combine` into a single
        :class:`pyvista.PolyData`; only the root ``cad.source_format`` /
        ``cad.reader`` / ``cad.backend`` / ``cad.units`` survive, and
        per-part ``cad.color`` / ``cad.label`` / ``cad.transform`` is
        dropped by the merge.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    pyvista_cad.CadReadError
        If parsing fails.
    pyvista_cad.OptionalDependencyError
        If no STEP backend is installed.

    Examples
    --------
    Read the bundled STEP cube fixture:

    >>> import pyvista as pv
    >>> import pyvista_cad
    >>> from pyvista_cad import read_step
    >>> mesh = read_step(pyvista_cad.examples.step_cube)
    >>> type(mesh).__name__
    'MultiBlock'
    >>> [round(b) for b in mesh.bounds]
    [-5, 5, -5, 5, -5, 5]
    >>> str(mesh.field_data['cad.source_format'][0])
    'step'

    """
    require_local_path(path)
    backend_name = _resolve_backend(backend)
    if backend_name == 'build123d':
        from pyvista_cad._backends._build123d import read_step as _impl
    elif backend_name == 'cascadio':
        from pyvista_cad._backends._cascadio import read_step as _impl
    else:
        msg = f'unknown STEP backend {backend!r}'
        raise OptionalDependencyError(msg)

    return _impl(
        path,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        merge=merge,
    )


def _block_transform(obj: Any) -> Any:
    """Return a ``gp_Trsf`` from an ``obj``'s ``cad.transform`` field, if any.

    The transform is stored as a raveled 16-element row-major 4x4 matrix
    in ``field_data['cad.transform']``. Only the top three rows
    (``m00..m23``) are loaded into the ``gp_Trsf``: the placement is
    assumed affine and the projective bottom row (``m30..m33``) is
    dropped, so a non-affine 4x4 does not round-trip. Returns an
    identity ``gp_Trsf`` when no usable transform is present (missing
    field, or not a 16-element array).
    """
    from OCP.gp import gp_Trsf

    trsf = gp_Trsf()
    field_data = getattr(obj, 'field_data', None)
    if field_data is None or 'cad.transform' not in field_data:
        return trsf
    arr = np.asarray(field_data['cad.transform'], dtype=float).ravel()
    if arr.size != 16:
        return trsf
    m = arr.reshape(4, 4)
    trsf.SetValues(
        float(m[0, 0]),
        float(m[0, 1]),
        float(m[0, 2]),
        float(m[0, 3]),
        float(m[1, 0]),
        float(m[1, 1]),
        float(m[1, 2]),
        float(m[1, 3]),
        float(m[2, 0]),
        float(m[2, 1]),
        float(m[2, 2]),
        float(m[2, 3]),
    )
    return trsf


def _as_surface(obj: Any) -> pv.PolyData:
    """Coerce a dataset to a triangle surface :class:`pyvista.PolyData`."""
    if isinstance(obj, pv.PolyData) and obj.is_all_triangles:
        return obj
    surface = obj.extract_surface(algorithm='dataset_surface')
    if not isinstance(surface, pv.PolyData):
        msg = f'could not extract a surface from {type(obj).__name__}'
        raise CadWriteError(msg)
    return surface


def _build_assembly(
    obj: Any,
    name: str,
    shape_tool: Any,
    failed: list[str],
) -> Any:
    """Recursively build OCAF labels for ``obj``; return its ``TDF_Label``.

    A :class:`pyvista.MultiBlock` becomes a sub-assembly label whose
    components are the recursively built child labels. Any other dataset
    becomes a single leaf product. Recursion is unbounded over nested
    MultiBlocks. Blocks whose faceting fails are appended to ``failed``
    and skipped (the caller raises naming every dropped block).
    """
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TopLoc import TopLoc_Location

    from pyvista_cad._conversion import polydata_to_topods

    if isinstance(obj, pv.MultiBlock):
        asm = shape_tool.NewShape()
        TDataStd_Name.Set_s(asm, TCollection_ExtendedString(name))
        n_components = 0
        for i in range(len(obj)):
            child = obj[i]
            if child is None:
                continue
            key = obj.get_block_name(i) or f'block_{i}'
            child_lbl = _build_assembly(child, key, shape_tool, failed)
            if child_lbl is None:
                continue
            loc = TopLoc_Location(_block_transform(child))
            shape_tool.AddComponent(asm, child_lbl, loc)
            n_components += 1
        # An empty MultiBlock (or a tree of only empty/None MultiBlocks)
        # produced no products. Returning None here lets the caller's
        # ``root is None`` guard reject it, mirroring ``_write_compound``'s
        # ``n == 0`` guard so the writer never emits a product-less STEP.
        if n_components == 0:
            return None
        return asm

    try:
        surface = _as_surface(obj)
        shape = polydata_to_topods(surface)
    except Exception:
        failed.append(name)
        return None

    lbl = shape_tool.AddShape(shape, False, False)
    TDataStd_Name.Set_s(lbl, TCollection_ExtendedString(name))
    return lbl


def _write_assembly(dataset: pv.MultiBlock, path: str | os.PathLike[str]) -> None:
    """Write ``dataset`` as a labelled OCAF STEP assembly."""
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Controller, STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.XSControl import XSControl_WorkSession

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString('pyvista-cad'))
    app.NewDocument(TCollection_ExtendedString('MDTV-XCAF'), doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    failed: list[str] = []
    root_name = dataset.field_data.get('cad.label')
    name = str(root_name[0]) if root_name is not None and len(root_name) else 'assembly'
    root = _build_assembly(dataset, name, shape_tool, failed)

    if failed:
        msg = f'write_step could not facet {len(failed)} block(s): {", ".join(failed)}'
        raise CadWriteError(msg)
    if root is None:
        msg = f'write_step found no geometry to write in {type(dataset).__name__}'
        raise CadWriteError(msg)

    shape_tool.UpdateAssemblies()

    STEPCAFControl_Controller.Init_s()
    work_session = XSControl_WorkSession()
    writer = STEPCAFControl_Writer(work_session, False)
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        msg = f'OCP STEPCAFControl_Writer failed for {path} (status={status})'
        raise CadWriteError(msg)


def _write_compound(
    dataset: pv.PolyData | pv.MultiBlock,
    path: str | os.PathLike[str],
) -> None:
    """Write ``dataset`` as a single flat faceted ``TopoDS_Compound``."""
    from OCP.BRep import BRep_Builder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound

    from pyvista_cad._conversion import polydata_to_topods

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    failed: list[str] = []

    def _walk(obj: Any, name: str) -> int:
        if isinstance(obj, pv.MultiBlock):
            added = 0
            for i in range(len(obj)):
                child = obj[i]
                if child is None:
                    continue
                key = obj.get_block_name(i) or f'block_{i}'
                added += _walk(child, key)
            return added
        try:
            surface = _as_surface(obj)
            shape = polydata_to_topods(surface)
        except Exception:
            failed.append(name)
            return 0
        builder.Add(compound, shape)
        return 1

    n = _walk(dataset, 'root')
    if failed:
        msg = f'write_step could not facet {len(failed)} block(s): {", ".join(failed)}'
        raise CadWriteError(msg)
    if n == 0:
        msg = f'write_step found no geometry to write in {type(dataset).__name__}'
        raise CadWriteError(msg)

    writer = STEPControl_Writer()
    writer.Transfer(compound, STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        msg = f'OCP STEPControl_Writer failed for {path} (status={status})'
        raise CadWriteError(msg)


def write_step(
    dataset: pv.PolyData | pv.MultiBlock,
    path: str | os.PathLike[str],
    /,
    *,
    write_assembly: bool = True,
    **_: Any,
) -> None:
    """Write a :class:`pyvista.PolyData` or :class:`pyvista.MultiBlock` as STEP.

    Each input triangle is converted to a planar face; the faces are
    sewn with ``BRepBuilderAPI_Sewing`` so a watertight mesh becomes a
    single closed shell, which is promoted to a ``TopoDS_Solid`` when
    closed. The output is a faceted STEP, not a B-rep reconstruction:
    analytic surface information is not recovered.

    When ``dataset`` is a :class:`pyvista.MultiBlock` and
    ``write_assembly`` is ``True`` (the default), a labelled OCAF
    assembly is emitted -- one STEP product per block, the block key
    used as the component name, and nested MultiBlocks written as
    nested sub-assemblies (recursion is unbounded). A per-block
    ``field_data['cad.transform']`` (raveled row-major 4x4) is applied
    as the component placement. The placement must be affine: only the
    top three rows are read into the OCCT ``gp_Trsf``. The projective
    bottom row is assumed to be ``[0, 0, 0, 1]`` and is silently
    dropped, so a non-affine 4x4 does not round-trip. When
    ``write_assembly`` is ``False`` or ``dataset`` is a single
    :class:`pyvista.PolyData`, the geometry is collapsed into one flat
    ``TopoDS_Compound``.

    Round-trip metadata: a :func:`read_step` of the written file with
    ``merge=False`` recovers per-part ``cad.color``, ``cad.label`` and
    ``cad.transform``. With ``merge=True`` the parts are flattened by
    :meth:`pyvista.MultiBlock.combine`, which keeps only the root
    ``cad.source_format`` / ``cad.units`` / ``cad.reader`` /
    ``cad.backend``; per-part ``cad.color``, ``cad.label`` and
    ``cad.transform`` do not survive the merge.

    Chordal-deviation bound: the faceted STEP is the input
    triangulation, vertex-for-vertex, except that
    :func:`~pyvista_cad._conversion.polydata_to_topods` sews faces with
    a tolerance of ``max(diag * 1e-6, 1e-9)`` (``diag`` = bounding-box
    diagonal), so vertices closer together than ``diag * 1e-6`` may be
    welded. Apart from that welding, deviation from the original
    analytic surface stays bounded by the ``linear_deflection`` used
    when that surface was tessellated into ``dataset`` (see
    :func:`read_step`); this writer introduces no further geometric
    error.

    Parameters
    ----------
    dataset : pyvista.PolyData or pyvista.MultiBlock
        Source dataset.
    path : str or os.PathLike
        Destination ``.step`` or ``.stp`` path.
    write_assembly : bool, default: True
        Emit a labelled OCAF assembly for MultiBlock inputs. When
        ``False``, MultiBlock inputs are flattened into one compound.
    **_ : Any
        Forward-compat keyword arguments are accepted and ignored.

    Raises
    ------
    pyvista_cad.CadWriteError
        If ``dataset`` is an unsupported type, if no geometry is found,
        or if one or more blocks cannot be faceted (every dropped block
        is named in the message; no block is silently dropped).
    pyvista_cad.OptionalDependencyError
        If OCP is not installed.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> import pyvista as pv
    >>> import pyvista_cad  # noqa: F401
    >>> asm = pv.MultiBlock()
    >>> asm['Base'] = pv.Box()
    >>> asm['Pin'] = pv.Sphere()
    >>> with tempfile.TemporaryDirectory() as d:
    ...     out = pathlib.Path(d) / 'asm.step'
    ...     pyvista_cad.write_step(asm, out)
    ...     out.exists()
    True

    """
    try:
        import OCP  # noqa: F401
    except ImportError as exc:
        msg = 'OCP not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc

    if not isinstance(dataset, (pv.PolyData, pv.MultiBlock)):
        msg = f'write_step requires a pv.PolyData or pv.MultiBlock; got {type(dataset).__name__}'
        raise CadWriteError(msg)

    if write_assembly and isinstance(dataset, pv.MultiBlock):
        _write_assembly(dataset, path)
    else:
        _write_compound(dataset, path)
