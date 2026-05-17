"""CAD metadata schema, units, and conversion factors.

Defines the ``cad.*`` field-data, cell-data, and point-data namespace
carried on PyVista datasets read from CAD formats.
"""

from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

# Canonical length-unit string set used by ``cad.units``.
Units = Literal['mm', 'cm', 'm', 'km', 'inch', 'foot', 'yard', 'mile', 'micron', 'nm', 'unitless']

_VALID_UNITS: frozenset[str] = frozenset(
    {'mm', 'cm', 'm', 'km', 'inch', 'foot', 'yard', 'mile', 'micron', 'nm', 'unitless'}
)

_TO_MM: dict[str, float] = {
    'mm': 1.0,
    'cm': 10.0,
    'm': 1000.0,
    'km': 1_000_000.0,
    'inch': 25.4,
    'foot': 304.8,
    'yard': 914.4,
    'mile': 1_609_344.0,
    'micron': 0.001,
    'nm': 1e-6,
}

_INSUNITS_TO_STR: dict[int, str] = {
    0: 'unitless',
    1: 'inch',
    2: 'foot',
    3: 'mile',
    4: 'mm',
    5: 'cm',
    6: 'm',
    7: 'km',
    # 8 = micro-inch, 9 = mil; no canonical small-imperial unit yet
    8: 'unitless',
    9: 'unitless',
    10: 'yard',
    # 11 = angstrom; no canonical sub-nanometre unit yet
    11: 'unitless',
    12: 'nm',
    13: 'micron',
    # 14-20 = decimetre, decametre, hectometre, gigametre, AU, light-year, parsec;
    # no canonical mapping for these yet
    14: 'unitless',
    15: 'unitless',
    16: 'unitless',
    17: 'unitless',
    18: 'unitless',
    19: 'unitless',
    20: 'unitless',
}

_STR_TO_INSUNITS: dict[str, int] = {
    'unitless': 0,
    'inch': 1,
    'foot': 2,
    'mile': 3,
    'mm': 4,
    'cm': 5,
    'm': 6,
    'km': 7,
    'yard': 10,
    'nm': 12,
    'micron': 13,
}


def insunits_to_str(code: int) -> str:
    """Translate an AutoCAD ``$INSUNITS`` enum into a canonical unit string.

    Parameters
    ----------
    code : int
        The DXF ``$INSUNITS`` header value.

    Returns
    -------
    str
        One of the canonical :data:`Units` strings.

    Examples
    --------
    >>> from pyvista_cad._metadata import insunits_to_str
    >>> insunits_to_str(4)
    'mm'
    >>> insunits_to_str(0)
    'unitless'

    """
    return _INSUNITS_TO_STR.get(int(code), 'unitless')


def insunits_code(units: str | None) -> int:
    """Translate a canonical unit string back into a DXF ``$INSUNITS`` code.

    Parameters
    ----------
    units : str or None
        A :data:`Units` string. ``None`` is treated as ``'unitless'``.

    Returns
    -------
    int
        The DXF ``$INSUNITS`` header value.

    Examples
    --------
    >>> from pyvista_cad._metadata import insunits_code
    >>> insunits_code('mm')
    4
    >>> insunits_code(None)
    0

    """
    if units is None:
        return 0
    return _STR_TO_INSUNITS.get(units, 0)


def validate_units(units: str) -> str:
    """Validate that a string is a known unit.

    Parameters
    ----------
    units : str
        The unit string to validate.

    Returns
    -------
    str
        The validated unit string (returned unchanged on success).

    Raises
    ------
    ValueError
        If ``units`` is not one of the canonical :data:`Units` strings.

    Examples
    --------
    >>> from pyvista_cad._metadata import validate_units
    >>> validate_units('mm')
    'mm'

    """
    if units not in _VALID_UNITS:
        msg = f'unknown units {units!r}; must be one of {sorted(_VALID_UNITS)}'
        raise ValueError(msg)
    return units


def units_conversion_factor(src: str, dst: str) -> float:
    """Return the multiplier that converts coordinates from ``src`` to ``dst``.

    Parameters
    ----------
    src : str
        Source unit string.
    dst : str
        Destination unit string.

    Returns
    -------
    float
        Multiplier such that ``coord_in_dst = coord_in_src * factor``.

    Raises
    ------
    ValueError
        If either side is ``'unitless'`` (no conversion defined) or an
        unknown unit string.

    Examples
    --------
    >>> from pyvista_cad._metadata import units_conversion_factor
    >>> units_conversion_factor('m', 'mm')
    1000.0
    >>> round(units_conversion_factor('inch', 'mm'), 1)
    25.4

    """
    validate_units(src)
    validate_units(dst)
    if src == 'unitless' or dst == 'unitless':
        msg = f'cannot convert between unitless and {src!r}/{dst!r}'
        raise ValueError(msg)
    return _TO_MM[src] / _TO_MM[dst]


@dataclass
class CadMetadata:
    """Bag of CAD-flavored metadata derived from a PyVista dataset.

    Mirrors the ``cad.*`` field-data schema. The complete set of
    persisted field-data keys is:

    * ``cad.source_format`` -> :attr:`source_format`
    * ``cad.source_format_version`` -> :attr:`source_format_version`
    * ``cad.units`` -> :attr:`units`
    * ``cad.label`` -> :attr:`label`
    * ``cad.guid`` -> :attr:`guid`
    * ``cad.color`` -> :attr:`color` (RGB or RGBA, alpha optional)
    * ``cad.transform`` -> :attr:`transform`
    * ``cad.metadata`` -> :attr:`extra` (JSON-encoded free-form bag)

    There is no separate opacity key: alpha rides on ``cad.color`` as
    an optional fourth component. ``cad.surface_id``, ``cad.normal``,
    and ``cad.uv`` are render-time analytic quantities recomputed by
    the CAD view, not persisted here. ``assembly_tree`` is a
    :class:`~pyvista_cad.CadMultiBlockAccessor` method, not a field.

    Parameters
    ----------
    source_format : str or None
        Originating CAD format (``'STEP'``, ``'DXF'``, ...).
    source_format_version : str or None
        Format-specific version stamp written by the originating
        backend (for example the DXF release such as ``'R2018'``).
    units : str or None
        Length units (``'mm'``, ``'inch'``, ...).
    color : tuple of float or None
        Whole-dataset color in ``[0, 1]``. A 3-tuple is opaque RGB; a
        4-tuple is RGBA where the fourth component is alpha (opacity).
    label : str or None
        Whole-dataset label (e.g. STEP ``PRODUCT.name``).
    guid : str or None
        Unique identifier (IFC ``GlobalId``, etc.).
    transform : numpy.ndarray or None
        Assembly-local-to-world 4x4 transform.
    extra : dict
        Free-form bag of source-format-specific metadata.

    Examples
    --------
    >>> from pyvista_cad import CadMetadata
    >>> md = CadMetadata(source_format='STEP', units='mm')
    >>> md.units
    'mm'

    """

    source_format: str | None = None
    source_format_version: str | None = None
    units: str | None = None
    color: tuple[float, ...] | None = None
    label: str | None = None
    guid: str | None = None
    transform: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataset(cls, dataset: 'pv.DataSet | pv.MultiBlock') -> 'CadMetadata':
        """Pull every ``cad.*`` field-data entry into a :class:`CadMetadata`.

        Parameters
        ----------
        dataset : pyvista.DataSet or pyvista.MultiBlock
            Dataset to read metadata from.

        Returns
        -------
        CadMetadata
            Reconstructed metadata bag.

        Examples
        --------
        >>> import numpy as np
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> from pyvista_cad import CadMetadata
        >>> mesh = pv.Sphere()
        >>> mesh.field_data['cad.source_format'] = np.array(['STEP'])
        >>> CadMetadata.from_dataset(mesh).source_format
        'STEP'

        """
        fd = dataset.field_data
        color: tuple[float, ...] | None = None
        if 'cad.color' in fd:
            arr = np.asarray(fd['cad.color']).ravel()
            if arr.size >= 4:
                color = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
            elif arr.size == 3:
                color = (float(arr[0]), float(arr[1]), float(arr[2]))
        transform = None
        if 'cad.transform' in fd:
            arr = np.asarray(fd['cad.transform']).ravel()
            if arr.size >= 16:
                transform = arr[:16].reshape(4, 4).astype(float)
        extra: dict[str, Any] = {}
        if 'cad.metadata' in fd:
            blob = fd['cad.metadata']
            text = blob[0] if hasattr(blob, '__len__') and len(blob) > 0 else blob
            try:
                extra = json.loads(str(text))
            except (TypeError, ValueError):
                extra = {}
        return cls(
            source_format=_read_field_str(fd, 'cad.source_format'),
            source_format_version=_read_field_str(fd, 'cad.source_format_version'),
            units=_read_field_str(fd, 'cad.units'),
            color=color,
            label=_read_field_str(fd, 'cad.label'),
            guid=_read_field_str(fd, 'cad.guid'),
            transform=transform,
            extra=extra,
        )

    def apply_to(self, dataset: 'pv.DataSet | pv.MultiBlock') -> None:
        """Write every populated attribute back to the dataset's ``field_data``.

        Parameters
        ----------
        dataset : pyvista.DataSet or pyvista.MultiBlock
            Dataset to populate.

        Examples
        --------
        >>> import pyvista as pv
        >>> import pyvista_cad
        >>> from pyvista_cad import CadMetadata
        >>> mesh = pv.Sphere()
        >>> CadMetadata(units='mm').apply_to(mesh)
        >>> str(mesh.field_data['cad.units'][0])
        'mm'

        """
        fd = dataset.field_data
        if self.source_format is not None:
            fd['cad.source_format'] = np.array([self.source_format])
        if self.source_format_version is not None:
            fd['cad.source_format_version'] = np.array([self.source_format_version])
        if self.units is not None:
            fd['cad.units'] = np.array([self.units])
        if self.label is not None:
            fd['cad.label'] = np.array([self.label])
        if self.guid is not None:
            fd['cad.guid'] = np.array([self.guid])
        if self.color is not None:
            fd['cad.color'] = np.asarray(self.color, dtype=float)
        if self.transform is not None:
            fd['cad.transform'] = np.asarray(self.transform, dtype=float).ravel()
        if self.extra:
            fd['cad.metadata'] = np.array([json.dumps(self.extra)])


def _read_field_str(field_data: Any, key: str) -> str | None:
    """Read a string-valued field-data entry, returning ``None`` if absent.

    Parameters
    ----------
    field_data : pyvista FieldData
        Source field-data container.
    key : str
        Field-data key to read.

    Returns
    -------
    str or None
        Decoded string, or ``None`` if ``key`` is absent.

    """
    if key not in field_data:
        return None
    blob = field_data[key]
    if hasattr(blob, '__len__') and len(blob) > 0:
        return str(blob[0])
    return str(blob)
