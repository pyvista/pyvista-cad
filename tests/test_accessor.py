"""Tests for the ``.cad`` accessor."""

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad import CadMetadata


def test_cad_accessor_classes_distinct() -> None:
    """The DataSet and MultiBlock accessor classes are distinct public types."""
    assert pyvista_cad.CadDataSetAccessor is not pyvista_cad.CadMultiBlockAccessor
    assert not hasattr(pyvista_cad, 'CadAccessor')


def test_cad_accessor_dispatches_by_dataset_type() -> None:
    """The ``.cad`` accessor dispatches to the dataset-specific class."""
    sphere = pv.Sphere()
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    mb = pv.MultiBlock([pv.Sphere(), pv.Cube()])
    assert isinstance(sphere.cad, pyvista_cad.CadDataSetAccessor)
    assert isinstance(grid.cad, pyvista_cad.CadDataSetAccessor)
    assert isinstance(mb.cad, pyvista_cad.CadMultiBlockAccessor)


def test_error_classes_hierarchy() -> None:
    """``Cad*Error`` classes form the documented hierarchy under ``CadError``."""
    assert issubclass(pyvista_cad.CadReadError, pyvista_cad.CadError)
    assert issubclass(pyvista_cad.CadWriteError, pyvista_cad.CadError)
    assert issubclass(pyvista_cad.UnsupportedFormatError, pyvista_cad.CadError)
    assert issubclass(pyvista_cad.OptionalDependencyError, pyvista_cad.CadError)
    assert issubclass(pyvista_cad.OptionalDependencyError, ImportError)
    assert issubclass(pyvista_cad.TessellationError, pyvista_cad.CadError)


def _line_mesh(layer_values: list[str]) -> pv.PolyData:
    n = len(layer_values)
    points = np.zeros((2 * n, 3), dtype=float)
    points[:, 0] = np.arange(2 * n)
    lines: list[int] = []
    for i in range(n):
        lines.extend([2, 2 * i, 2 * i + 1])
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = np.asarray(lines, dtype=np.int64)
    mesh.cell_data['cad.layer'] = np.asarray(layer_values, dtype=object)
    return mesh


def test_units_property_round_trip() -> None:
    mesh = pv.Sphere()
    mesh.cad.units = 'mm'
    assert mesh.cad.units == 'mm'
    mesh.cad.units = None
    assert mesh.cad.units is None


def test_units_invalid_rejected() -> None:
    mesh = pv.Sphere()
    with pytest.raises(ValueError, match='unknown units'):
        mesh.cad.units = 'parsec'


def test_color_property_round_trip() -> None:
    mesh = pv.Sphere()
    mesh.cad.color = (0.1, 0.2, 0.3)
    got = mesh.cad.color
    assert got is not None
    assert pytest.approx(got) == (0.1, 0.2, 0.3)


def test_color_invalid_shape_rejected() -> None:
    mesh = pv.Sphere()
    with pytest.raises(ValueError, match='3-tuple'):
        mesh.cad.color = (0.5, 0.5)  # type: ignore[arg-type]


def test_label_property_round_trip() -> None:
    mesh = pv.Sphere()
    mesh.cad.label = 'part-A'
    assert mesh.cad.label == 'part-A'


def test_set_units_with_convert_scales_points() -> None:
    mesh = pv.Sphere(radius=10)
    mesh.cad.units = 'm'
    p0 = mesh.points.copy()
    mesh.cad.set_units('mm', convert=True)
    assert mesh.cad.units == 'mm'
    np.testing.assert_allclose(mesh.points, p0 * 1000.0)


def test_set_units_without_convert_keeps_points() -> None:
    mesh = pv.Sphere(radius=1)
    mesh.cad.units = 'mm'
    p0 = mesh.points.copy()
    mesh.cad.set_units('m', convert=False)
    assert mesh.cad.units == 'm'
    np.testing.assert_array_equal(mesh.points, p0)


def test_set_units_convert_requires_current() -> None:
    mesh = pv.Sphere()
    with pytest.raises(ValueError, match='current units'):
        mesh.cad.set_units('mm', convert=True)


def test_layers_property_lists_unique_sorted() -> None:
    mesh = _line_mesh(['OUTLINE', 'HOLES', 'OUTLINE'])
    assert mesh.cad.layers == ['HOLES', 'OUTLINE']


def test_split_by_layer_produces_one_block_per_layer() -> None:
    mesh = _line_mesh(['A', 'B', 'A'])
    mb = mesh.cad.split_by_layer()
    assert isinstance(mb, pv.MultiBlock)
    assert sorted(mb.keys()) == ['A', 'B']


def test_split_by_color_groups_cells() -> None:
    mesh = _line_mesh(['x', 'x', 'x'])
    colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    mesh.cell_data['cad.color'] = colors
    mb = mesh.cad.split_by_color()
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 2


def test_metadata_setter_clears_and_rewrites() -> None:
    mesh = pv.Sphere()
    mesh.cad.units = 'mm'
    mesh.cad.label = 'old'
    mesh.cad.metadata = CadMetadata(units='inch', label='new')
    assert mesh.cad.units == 'inch'
    assert mesh.cad.label == 'new'
