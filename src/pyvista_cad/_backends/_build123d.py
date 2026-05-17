"""STEP reader backed by build123d (primary STEP path).

build123d wraps OCP's STEPCAFControl reader and surfaces the full
OCCT CAF metadata: per-part labels, colors, materials, layers, and
nested assemblies.
"""

import os
import re
from typing import Any
import warnings

import numpy as np
import pyvista as pv

from pyvista_cad._conversion import topods_to_polydata
from pyvista_cad._errors import (
    CadReadError,
    CadWarning,
    OptionalDependencyError,
    TessellationError,
)

_BACKEND_NAME = 'build123d'


def read_step(
    path: str | os.PathLike[str],
    *,
    linear_deflection: float,
    angular_deflection: float,
    merge: bool,
) -> pv.MultiBlock | pv.PolyData:
    """Read a STEP file via :mod:`build123d`.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.step`` or ``.stp`` file.
    linear_deflection : float
        OCCT chordal tessellation tolerance.
    angular_deflection : float
        OCCT angular tessellation tolerance (radians).
    merge : bool
        If ``True``, combine every part into a single ``PolyData``;
        otherwise return a ``MultiBlock`` (one block per part) with a
        nested structure that mirrors the STEP assembly tree.

    Returns
    -------
    pyvista.MultiBlock or pyvista.PolyData
        Tessellated STEP geometry.

    """
    try:
        import build123d as b3d
    except ImportError as exc:
        msg = 'build123d not installed; install pyvista-cad[step]'
        raise OptionalDependencyError(msg) from exc

    fname = os.fspath(path)
    if not os.path.exists(fname):
        msg = f'No such file: {fname}'
        raise FileNotFoundError(msg)

    try:
        result = b3d.import_step(fname)
    except (RuntimeError, OSError, ValueError) as exc:
        # build123d wraps OCCT's STEPControl reader; a malformed file
        # surfaces as RuntimeError (OCCT Standard_Failure), OSError
        # (unreadable path) or ValueError (header garbage). Re-raised,
        # never swallowed.
        msg = f'failed to read STEP {fname}: {exc}'
        raise CadReadError(msg) from exc

    # build123d's STEP importer swallows OCCT parse errors and returns an
    # empty Compound rather than raising. Detect "no shapes, no children"
    # and surface a CadReadError so malformed inputs fail loudly.
    def _is_empty(r: Any) -> bool:
        children = getattr(r, 'children', None)
        if children is not None and len(list(children)) > 0:
            return False
        wrapped = getattr(r, 'wrapped', None)
        if wrapped is None:
            return True
        # A Compound with no sub-shapes has IsNull() False but no TopoDS_Iterator
        # children. Use the wrapped shape's IsNull / direct iteration.
        try:
            from OCP.TopoDS import TopoDS_Iterator

            it = TopoDS_Iterator(wrapped)
            return not it.More()
        except (ImportError, RuntimeError, TypeError):
            # No OCP, or ``wrapped`` is not iterable: cannot prove the
            # result is empty, so assume it is not and let the caller
            # proceed (a truly empty result still produces an empty
            # MultiBlock, not a crash).
            return False

    if _is_empty(result):
        msg = (
            f'failed to read STEP {fname}: file parsed to an empty assembly '
            '(likely malformed header or no DATA section)'
        )
        raise CadReadError(msg)

    units = _detect_step_units(fname)

    mb = pv.MultiBlock()
    _populate_multiblock(
        mb,
        result,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        units=units,
    )

    mb.field_data['cad.source_format'] = np.array(['step'])
    mb.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
    mb.field_data['cad.backend'] = np.array([_BACKEND_NAME])
    mb.field_data['cad.units'] = np.array([units])

    if merge:
        combined = mb.combine(merge_points=True).extract_surface(algorithm='dataset_surface')
        combined.field_data['cad.source_format'] = np.array(['step'])
        combined.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
        combined.field_data['cad.backend'] = np.array([_BACKEND_NAME])
        combined.field_data['cad.units'] = np.array([units])
        return combined
    return mb


def _color_of(obj: Any) -> tuple[float, ...] | None:
    """Return ``obj``'s build123d color as RGB or RGBA floats, or ``None``.

    A modern :class:`build123d.Color` is iterable, yielding four floats
    ``(r, g, b, a)`` in ``[0, 1]``, and exposes neither ``.r/.g/.b`` nor
    item access. Legacy color objects carrying ``.r/.g/.b`` (no alpha)
    are still handled. The returned tuple has four components when the
    source carries alpha and three otherwise, matching the ``cad.color``
    contract in :mod:`pyvista_cad._metadata` and the build123d bridge.

    Absence of a ``color`` attribute is the common, non-error case and
    returns ``None`` silently. A ``color`` that is present but whose
    components cannot be read is real color loss: a :class:`~pyvista_cad.CadWarning`
    naming the node is emitted.
    """
    color = getattr(obj, 'color', None)
    if color is None:
        return None
    # build123d Color is iterable-only (r, g, b, a). Legacy color
    # objects expose .r/.g/.b with no alpha. Try the attribute form
    # first, then the iterable form, preserving a 4th alpha component.
    if all(hasattr(color, c) for c in ('r', 'g', 'b')):
        try:
            a = getattr(color, 'a', None)
            if a is not None:
                return (float(color.r), float(color.g), float(color.b), float(a))
            return (float(color.r), float(color.g), float(color.b))
        except (TypeError, ValueError):
            pass
    else:
        try:
            seq = [float(c) for c in color]
        except (TypeError, ValueError):
            seq = []
        if len(seq) >= 4:
            return (seq[0], seq[1], seq[2], seq[3])
        if len(seq) == 3:
            return (seq[0], seq[1], seq[2])
    label = getattr(obj, 'label', '') or repr(obj)
    warnings.warn(
        f'STEP node {label!r}: a color is set but its components could '
        f'not be read ({color!r}); the part renders without its '
        'authored color.',
        CadWarning,
        stacklevel=2,
    )
    return None


def _location_matrix(obj: Any) -> np.ndarray | None:
    """Extract a 4x4 row-major transform from a build123d child's ``.location``.

    Returns ``None`` if no location is present or it is identity.
    """
    loc = getattr(obj, 'location', None)
    if loc is None:
        return None
    wrapped = getattr(loc, 'wrapped', loc)
    try:
        trsf = wrapped.Transformation()
    except AttributeError:
        return None
    mat = np.eye(4, dtype=float)
    try:
        for i in range(1, 4):
            for j in range(1, 5):
                mat[i - 1, j - 1] = float(trsf.Value(i, j))
    except (RuntimeError, TypeError, ValueError) as exc:
        # Indices are fixed-valid (1..3, 1..4); a failure means the
        # OCCT transform itself is broken. The part keeps its untouched
        # local placement, so its world position is wrong: warn.
        label = getattr(obj, 'label', '') or repr(obj)
        warnings.warn(
            f'STEP node {label!r}: its placement transform could not be '
            f'read ({exc}); the part is left at its local origin.',
            CadWarning,
            stacklevel=2,
        )
        return None
    if np.allclose(mat, np.eye(4), atol=1e-12):
        return None
    return mat


def _has_geometry(wrapped: Any) -> bool:
    """Heuristic: a TopoDS_Shape has geometry when it has at least one face."""
    if wrapped is None:
        return False
    try:
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
    except ImportError:
        return True
    try:
        exp = TopExp_Explorer(wrapped, TopAbs_FACE)
        return bool(exp.More())
    except (RuntimeError, TypeError):
        # OCCT raises RuntimeError on a malformed shape; TypeError when
        # ``wrapped`` is not a TopoDS_Shape. Assume geometry is present
        # so the downstream tessellation gets a chance and surfaces a
        # precise TessellationError instead of a silent skip here.
        return True


def _populate_multiblock(
    mb: pv.MultiBlock,
    node: Any,
    *,
    linear_deflection: float,
    angular_deflection: float,
    units: str,
) -> None:
    """Recursively populate ``mb`` from a build123d node tree."""
    try:
        import build123d as b3d
    except ImportError:
        b3d = None  # type: ignore[assignment]

    children: list[Any] = []
    if b3d is not None and isinstance(node, b3d.Compound):
        try:
            children = list(node.children) if node.children else []
        except (AttributeError, RuntimeError) as exc:
            node_label = getattr(node, 'label', '') or repr(node)
            warnings.warn(
                f'STEP assembly node {node_label!r}: its children could '
                f'not be enumerated ({exc}); the whole sub-assembly is '
                'dropped from the result.',
                CadWarning,
                stacklevel=2,
            )
            children = []

    if not children:
        _append_leaf(
            mb,
            node,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            units=units,
        )
        return

    for child in children:
        label = getattr(child, 'label', '') or f'part_{len(mb)}'
        # Recurse for nested compounds; otherwise append a leaf.
        sub_children: list[Any] = []
        if b3d is not None and isinstance(child, b3d.Compound):
            try:
                sub_children = list(child.children) if child.children else []
            except (AttributeError, RuntimeError) as exc:
                warnings.warn(
                    f'STEP assembly node {label!r}: its children could '
                    f'not be enumerated ({exc}); the sub-assembly is '
                    'treated as a single leaf part.',
                    CadWarning,
                    stacklevel=2,
                )
                sub_children = []
        if sub_children:
            sub_mb = pv.MultiBlock()
            _populate_multiblock(
                sub_mb,
                child,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                units=units,
            )
            matrix = _location_matrix(child)
            if matrix is not None:
                sub_mb.field_data['cad.transform'] = matrix.astype(np.float64).ravel()
            sub_mb.field_data['cad.label'] = np.array([label])
            mb.append(sub_mb, name=label)
        else:
            _append_leaf(
                mb,
                child,
                linear_deflection=linear_deflection,
                angular_deflection=angular_deflection,
                units=units,
                fallback_label=label,
            )


def _append_leaf(
    mb: pv.MultiBlock,
    node: Any,
    *,
    linear_deflection: float,
    angular_deflection: float,
    units: str,
    fallback_label: str | None = None,
) -> None:
    """Tessellate one leaf node and append it to ``mb`` as a PolyData block."""
    wrapped = getattr(node, 'wrapped', node)
    label = fallback_label or getattr(node, 'label', '') or f'part_{len(mb)}'
    if not _has_geometry(wrapped):
        return
    try:
        poly = topods_to_polydata(
            wrapped,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
    except (TessellationError, ValueError) as exc:
        warnings.warn(
            f'STEP part {label!r}: tessellation failed ({exc}); this part '
            'is dropped from the result.',
            CadWarning,
            stacklevel=2,
        )
        return
    # build123d's STEP import resolves component/instance placements into
    # ``node.wrapped`` itself: the tessellated coordinates are already in
    # world space. Re-applying ``_location_matrix`` here would translate
    # the part a second time (a +T component round-tripped at +2T). Record
    # the placement in field data so a subsequent ``write_step`` can re-emit
    # it as the OCAF component location, but do NOT transform the geometry.
    matrix = _location_matrix(node)
    if matrix is not None:
        poly.field_data['cad.transform'] = matrix.astype(np.float64).ravel()
    color = _color_of(node)
    if color is not None:
        poly.field_data['cad.color'] = np.asarray(color, dtype=float)
    poly.field_data['cad.label'] = np.array([label])
    poly.field_data['cad.source_format'] = np.array(['step'])
    poly.field_data['cad.reader'] = np.array(['pyvista_cad.read_step'])
    poly.field_data['cad.backend'] = np.array([_BACKEND_NAME])
    poly.field_data['cad.units'] = np.array([units])
    mb.append(poly, name=label)


_UNIT_MAP = {
    'MILLIMETRE': 'mm',
    'MILLIMETER': 'mm',
    'CENTIMETRE': 'cm',
    'CENTIMETER': 'cm',
    'METRE': 'm',
    'METER': 'm',
    'INCH': 'inch',
    'FOOT': 'foot',
}

_LENGTH_UNIT_RE = re.compile(
    r'LENGTH_UNIT.*?SI_UNIT\s*\(\s*\.?([A-Z]*)\.?\s*,\s*\.([A-Z]+)\.\s*\)',
    re.IGNORECASE | re.DOTALL,
)
_CONV_UNIT_RE = re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']+)'", re.IGNORECASE)


def _detect_step_units(fname: str) -> str:
    """Scan a STEP file header for the declared length unit.

    Returns one of ``'mm'``, ``'cm'``, ``'m'``, ``'inch'``, ``'foot'`` or
    ``'unitless'`` if no unit can be parsed.
    """
    try:
        with open(fname, encoding='ascii', errors='ignore') as f:
            # STEP unit declarations live in DATA section; read enough
            # to cover small-to-medium files entirely, large files'
            # leading entities.
            head = f.read(131072)
    except OSError:
        return 'unitless'

    conv = _CONV_UNIT_RE.search(head)
    if conv:
        token = conv.group(1).upper().strip()
        if token in _UNIT_MAP:
            return _UNIT_MAP[token]

    m = _LENGTH_UNIT_RE.search(head)
    if m:
        prefix = (m.group(1) or '').upper()
        base = (m.group(2) or '').upper()
        if base == 'METRE':
            if prefix == 'MILLI':
                return 'mm'
            if prefix == 'CENTI':
                return 'cm'
            if prefix == '':
                return 'm'
        if base in _UNIT_MAP:
            return _UNIT_MAP[base]

    return 'unitless'
