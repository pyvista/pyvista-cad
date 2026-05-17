"""Tests for the ``cad.*`` metadata schema, units, and round-trips.

Every persisted schema key is round-tripped through
``CadMetadata.apply_to`` / ``from_dataset`` with exact-value assertions,
plus the units helpers, default handling, and the lenient malformed-value
behaviour of ``from_dataset`` (it never raises; callers validate).
"""

import json

import numpy as np
import pytest
import pyvista as pv

from pyvista_cad import CadMetadata
from pyvista_cad._metadata import (
    _read_field_str,
    insunits_code,
    insunits_to_str,
    units_conversion_factor,
    validate_units,
)

_ALL_UNITS = (
    'mm',
    'cm',
    'm',
    'km',
    'inch',
    'foot',
    'yard',
    'mile',
    'micron',
    'nm',
    'unitless',
)


# --------------------------------------------------------------------------- #
# Units enum + helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('unit', _ALL_UNITS)
def test_validate_units_accepts_every_canonical_unit(unit: str) -> None:
    assert validate_units(unit) == unit


@pytest.mark.parametrize('bad', ['parsec', 'PARSEC', '', 'meter'])
def test_validate_units_rejects_unknown(bad: str) -> None:
    with pytest.raises(ValueError, match='unknown units'):
        validate_units(bad)


def test_units_conversion_factor_known_pairs() -> None:
    assert units_conversion_factor('m', 'mm') == 1000.0
    assert units_conversion_factor('mm', 'm') == 0.001
    assert round(units_conversion_factor('inch', 'mm'), 6) == 25.4
    assert units_conversion_factor('mm', 'mm') == 1.0


@pytest.mark.parametrize(('src', 'dst'), [('unitless', 'mm'), ('mm', 'unitless')])
def test_units_conversion_factor_rejects_unitless(src: str, dst: str) -> None:
    with pytest.raises(ValueError, match='unitless'):
        units_conversion_factor(src, dst)


def test_units_conversion_factor_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match='unknown units'):
        units_conversion_factor('parsec', 'mm')


@pytest.mark.parametrize(
    ('code', 'expected'),
    [
        (0, 'unitless'),
        (1, 'inch'),
        (2, 'foot'),
        (3, 'mile'),
        (4, 'mm'),
        (5, 'cm'),
        (6, 'm'),
        (7, 'km'),
        (8, 'unitless'),
        (10, 'yard'),
        (12, 'nm'),
        (13, 'micron'),
        (20, 'unitless'),
        (999, 'unitless'),
    ],
)
def test_insunits_to_str(code: int, expected: str) -> None:
    assert insunits_to_str(code) == expected


@pytest.mark.parametrize(
    ('units', 'expected'),
    [
        ('mm', 4),
        ('m', 6),
        ('inch', 1),
        ('yard', 10),
        ('nm', 12),
        ('micron', 13),
        ('unitless', 0),
        (None, 0),
        ('parsec', 0),
    ],
)
def test_insunits_code(units: str | None, expected: int) -> None:
    assert insunits_code(units) == expected


# --------------------------------------------------------------------------- #
# Full round-trip: every schema key
# --------------------------------------------------------------------------- #


def test_round_trip_every_schema_key_rgb() -> None:
    mesh = pv.Sphere()
    transform = np.arange(16, dtype=float).reshape(4, 4)
    md = CadMetadata(
        source_format='step',
        source_format_version='AP242',
        units='mm',
        color=(0.5, 0.25, 0.75),
        label='Bracket',
        guid='0eCpv9L5zgTO2Bbg6ijyBD',
        transform=transform,
        extra={'foo': 'bar', 'n': 3, 'nested': {'a': [1, 2]}},
    )
    md.apply_to(mesh)

    # Raw field-data wire format.
    assert str(mesh.field_data['cad.source_format'][0]) == 'step'
    assert str(mesh.field_data['cad.source_format_version'][0]) == 'AP242'
    assert str(mesh.field_data['cad.units'][0]) == 'mm'
    assert str(mesh.field_data['cad.label'][0]) == 'Bracket'
    assert str(mesh.field_data['cad.guid'][0]) == '0eCpv9L5zgTO2Bbg6ijyBD'
    assert json.loads(str(mesh.field_data['cad.metadata'][0])) == md.extra

    back = CadMetadata.from_dataset(mesh)
    assert back.source_format == 'step'
    assert back.source_format_version == 'AP242'
    assert back.units == 'mm'
    assert back.label == 'Bracket'
    assert back.guid == '0eCpv9L5zgTO2Bbg6ijyBD'
    assert back.color is not None
    assert len(back.color) == 3
    assert back.color == pytest.approx((0.5, 0.25, 0.75))
    assert back.transform is not None
    np.testing.assert_array_equal(back.transform, transform)
    assert back.transform.shape == (4, 4)
    assert back.extra == {'foo': 'bar', 'n': 3, 'nested': {'a': [1, 2]}}


def test_round_trip_color_rgba() -> None:
    mesh = pv.Sphere()
    CadMetadata(color=(0.1, 0.2, 0.3, 0.4)).apply_to(mesh)
    back = CadMetadata.from_dataset(mesh)
    assert back.color is not None
    assert len(back.color) == 4
    assert back.color == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_round_trip_unusual_ascii_label() -> None:
    """Punctuation-heavy ASCII label survives the field-data round-trip."""
    mesh = pv.Sphere()
    label = r'PART <#42> / rev "B" {final} [v1.0]'
    CadMetadata(label=label).apply_to(mesh)
    assert CadMetadata.from_dataset(mesh).label == label


def test_non_ascii_label_is_rejected_by_vtk_string_array() -> None:
    """Non-ASCII labels cannot be persisted: VTK string arrays are ASCII-only.

    This is a VTK storage constraint surfacing through ``apply_to``, not a
    ``CadMetadata`` behaviour; pin it so the limitation is documented.
    """
    mesh = pv.Sphere()
    with pytest.raises(ValueError, match='non-ASCII'):
        CadMetadata(label='Bruecke-Ω-部品').apply_to(mesh)


@pytest.mark.parametrize('unit', _ALL_UNITS)
def test_round_trip_each_unit_value(unit: str) -> None:
    mesh = pv.Sphere()
    CadMetadata(units=unit).apply_to(mesh)
    assert CadMetadata.from_dataset(mesh).units == unit


def test_round_trip_oversized_transform_is_truncated_to_4x4() -> None:
    """A flattened transform with >16 entries keeps only the first 16 (4x4)."""
    mesh = pv.Sphere()
    mesh.field_data['cad.transform'] = np.arange(25, dtype=float)
    back = CadMetadata.from_dataset(mesh)
    assert back.transform is not None
    assert back.transform.shape == (4, 4)
    np.testing.assert_array_equal(back.transform, np.arange(16, dtype=float).reshape(4, 4))


# --------------------------------------------------------------------------- #
# Defaults and missing keys
# --------------------------------------------------------------------------- #


def test_empty_dataset_yields_all_defaults() -> None:
    md = CadMetadata.from_dataset(pv.Sphere())
    assert md.source_format is None
    assert md.source_format_version is None
    assert md.units is None
    assert md.color is None
    assert md.label is None
    assert md.guid is None
    assert md.transform is None
    assert md.extra == {}


def test_apply_to_skips_none_attributes() -> None:
    """Only populated attributes are written; ``None`` leaves no field-data."""
    mesh = pv.Sphere()
    CadMetadata(units='mm').apply_to(mesh)
    assert 'cad.units' in mesh.field_data
    assert 'cad.source_format' not in mesh.field_data
    assert 'cad.color' not in mesh.field_data
    assert 'cad.transform' not in mesh.field_data
    assert 'cad.metadata' not in mesh.field_data


def test_apply_to_skips_empty_extra() -> None:
    """An empty ``extra`` dict is falsy and is not persisted."""
    mesh = pv.Sphere()
    CadMetadata(extra={}).apply_to(mesh)
    assert 'cad.metadata' not in mesh.field_data


# --------------------------------------------------------------------------- #
# Lenient malformed-value handling (from_dataset never raises)
# --------------------------------------------------------------------------- #


def test_malformed_json_metadata_falls_back_to_empty() -> None:
    mesh = pv.Sphere()
    mesh.field_data['cad.metadata'] = np.array(['{ this is not json'])
    assert CadMetadata.from_dataset(mesh).extra == {}


def test_undersized_transform_is_ignored() -> None:
    mesh = pv.Sphere()
    mesh.field_data['cad.transform'] = np.array([1.0, 2.0, 3.0])
    assert CadMetadata.from_dataset(mesh).transform is None


def test_undersized_color_is_ignored() -> None:
    """A 2-component ``cad.color`` is neither RGB nor RGBA, so it is dropped."""
    mesh = pv.Sphere()
    mesh.field_data['cad.color'] = np.array([0.1, 0.2])
    assert CadMetadata.from_dataset(mesh).color is None


# --------------------------------------------------------------------------- #
# _read_field_str helper
# --------------------------------------------------------------------------- #


def test_read_field_str_missing_key_returns_none() -> None:
    assert _read_field_str(pv.Sphere().field_data, 'cad.nope') is None


def test_read_field_str_reads_first_element() -> None:
    fd = pv.Sphere().field_data
    fd['cad.label'] = np.array(['first', 'second'])
    assert _read_field_str(fd, 'cad.label') == 'first'


def test_read_field_str_empty_array_stringifies_container() -> None:
    """A zero-length array has no element 0, so the container is stringified."""
    fd = pv.Sphere().field_data
    fd['cad.label'] = np.array([], dtype='<U1')
    assert _read_field_str(fd, 'cad.label') == '[]'
