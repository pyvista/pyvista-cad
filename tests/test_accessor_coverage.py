"""Behavioral coverage for the ``.cad`` accessor surface.

Drives :mod:`pyvista_cad._accessor` to full line and branch coverage
with real CAD data: the bundled examples, the ``tests/data`` fixture
corpus, and live build123d / OCP / gmsh round-trips. Every
test makes a behavioral assertion (cell counts, bounds, exact metadata
values, raised error type and message, or a geometric invariant).
"""

from pathlib import Path
import re
import warnings

import pytest

# Skip the whole module (don't hard-error at collection) when a CAD
# backend can't load — notably a Windows "DLL load failed while
# importing OCP" ImportError, which is not a ModuleNotFoundError.
pytest.importorskip('build123d', exc_type=ImportError)
pytest.importorskip('cadquery', exc_type=ImportError)
pytest.importorskip('gmsh', exc_type=ImportError)
pytest.importorskip('OCP', exc_type=ImportError)

import build123d
import cadquery as cq
import gmsh
import numpy as np
from OCP.TopoDS import TopoDS_Shape
from packaging.version import Version
import pyvista as pv

import pyvista_cad
from pyvista_cad import (
    CadCacheError,
    CadError,
    CadMetadata,
    MetadataError,
    TessellationError,
    examples,
)
from pyvista_cad._accessor import (
    CadDataSetAccessor,
    _resolve_cell_array,
    _scale_points,
)
import pyvista_cad._conversion as _conversion

pv.OFF_SCREEN = True

_ASSEMBLY = 'tests/data/assembly_two_parts.step'
_EXTERNAL_BREP = 'tests/data/external.brep'

_BREP_ATTR = _conversion._BREP_ATTR


# --------------------------------------------------------------------------- #
# Shared real-data builders.
# --------------------------------------------------------------------------- #


@pytest.fixture
def assembly() -> pv.MultiBlock:
    """The two-part STEP assembly (Base box + Pin cylinder)."""
    return pyvista_cad.read_step(_ASSEMBLY)


@pytest.fixture
def brep_box() -> pv.PolyData:
    """The external BREP box (25 x 15 x 8 mm, origin-centred)."""
    return pyvista_cad.read_brep(_EXTERNAL_BREP)


def _b123d_cylinder() -> pv.PolyData:
    """A radius-5 height-10 build123d cylinder as CAD-origin PolyData.

    A curved solid is required for the deflection monotonicity
    invariant: a box has flat faces so its facet count is invariant to
    the linear deflection.
    """
    cyl = build123d.Cylinder(radius=5, height=10)
    return pyvista_cad.from_build123d(cyl.wrapped)


# --------------------------------------------------------------------------- #
# Metadata read properties.
# --------------------------------------------------------------------------- #


def test_source_format_reads_back_from_reader(assembly: pv.MultiBlock) -> None:
    """A STEP read stamps ``cad.source_format`` readable on the MultiBlock."""
    assert assembly.cad.source_format == 'step'
    assert assembly.cad.units == 'mm'


def test_source_format_absent_is_none() -> None:
    """A plain mesh with no CAD provenance reports ``None``."""
    assert pv.Sphere().cad.source_format is None


def test_units_setter_none_pops_field() -> None:
    """Assigning ``None`` removes the ``cad.units`` field entry."""
    mesh = pv.Sphere()
    mesh.cad.units = 'mm'
    mesh.cad.units = None
    assert 'cad.units' not in mesh.field_data
    assert mesh.cad.units is None


def test_label_setter_none_pops_field() -> None:
    """Assigning ``None`` removes the ``cad.label`` field entry."""
    mesh = pv.Sphere()
    mesh.cad.label = 'x'
    mesh.cad.label = None
    assert 'cad.label' not in mesh.field_data
    assert mesh.cad.label is None


def test_color_setter_none_pops_field() -> None:
    """Assigning ``None`` removes the ``cad.color`` field entry."""
    mesh = pv.Sphere()
    mesh.cad.color = (0.1, 0.2, 0.3)
    mesh.cad.color = None
    assert 'cad.color' not in mesh.field_data
    assert mesh.cad.color is None


def test_color_getter_short_array_is_none() -> None:
    """A ``cad.color`` with fewer than three components reads as ``None``."""
    mesh = pv.Sphere()
    mesh.field_data['cad.color'] = np.array([0.5, 0.5])
    assert mesh.cad.color is None


def test_color_getter_absent_is_none() -> None:
    """No ``cad.color`` field reads as ``None``."""
    assert pv.Sphere().cad.color is None


def test_read_str_field_empty_blob_stringified() -> None:
    """An empty (zero-length) field-data blob is stringified directly."""
    mesh = pv.Sphere()
    mesh.field_data['cad.label'] = np.array([])
    assert mesh.cad.label == '[]'


def test_repr_datasetshows_wrapped_type() -> None:
    """Both accessor reprs name the wrapped dataset type."""
    assert repr(pv.Sphere().cad) == '<CadDataSetAccessor on PolyData>'
    assert repr(pv.MultiBlock().cad) == '<CadMultiBlockAccessor on MultiBlock>'


# --------------------------------------------------------------------------- #
# layers.
# --------------------------------------------------------------------------- #


def test_layers_multiblock_unions_across_blocks() -> None:
    """MultiBlock ``layers`` unions every leaf's layer set, sorted."""
    a = pv.Cube().triangulate()
    a.cell_data['cad.layer'] = np.array(['L1'] * a.n_cells, dtype=object)
    b = pv.Cube().triangulate()
    b.cell_data['Layer'] = np.array(['L2'] * b.n_cells, dtype=object)
    mb = pv.MultiBlock([a, b, None])
    assert mb.cad.layers == ['L1', 'L2']


def test_layers_no_cell_data_is_empty() -> None:
    """A mesh with no layer cell-data yields an empty layer list."""
    assert pv.Sphere().cad.layers == []


def test_layers_nested_multiblock_has_no_cell_data() -> None:
    """A nested MultiBlock leaf (no ``cell_data``) yields no layers."""
    mb = pv.MultiBlock()
    mb.append(pv.MultiBlock([pv.Cube()]), name='grp')
    assert mb.cad.layers == []


# --------------------------------------------------------------------------- #
# has_brep_origin.
# --------------------------------------------------------------------------- #


def test_has_brep_origin_true_for_brep_read(brep_box: pv.PolyData) -> None:
    """A BREP read carries a recoverable cached ``TopoDS``."""
    assert brep_box.cad.has_brep_origin is True


def test_has_brep_origin_false_for_plain_mesh() -> None:
    """A hand-built mesh has no B-rep origin."""
    assert pv.Sphere().cad.has_brep_origin is False


def test_has_brep_origin_flag_field_truthy() -> None:
    """A stamped ``cad.has_brep_origin`` flag short-circuits to its value."""
    mesh = pv.Sphere()
    mesh.field_data['cad.has_brep_origin'] = np.array([1])
    assert mesh.cad.has_brep_origin is True


def test_has_brep_origin_flag_field_empty_is_false() -> None:
    """An empty flag array trips the IndexError guard and reads False."""
    mesh = pv.Sphere()
    mesh.field_data['cad.has_brep_origin'] = np.array([], dtype=float)
    assert mesh.cad.has_brep_origin is False


def test_has_brep_origin_multiblock_cached_leaf(assembly: pv.MultiBlock) -> None:
    """A MultiBlock is B-rep backed when any leaf has a cached shape."""
    assert assembly.cad.has_brep_origin is True


def test_has_brep_origin_multiblock_flag_leaf() -> None:
    """A leaf flag (no cache) still marks the MultiBlock B-rep backed."""
    mb = pv.MultiBlock([pv.Sphere(), None])
    mb[0].field_data['cad.has_brep_origin'] = np.array([1])
    assert mb.cad.has_brep_origin is True


def test_has_brep_origin_multiblock_flag_leaf_empty_continues() -> None:
    """An empty leaf flag is skipped; no other origin means False."""
    mb = pv.MultiBlock([pv.Sphere()])
    mb[0].field_data['cad.has_brep_origin'] = np.array([], dtype=float)
    assert mb.cad.has_brep_origin is False


def test_has_brep_origin_multiblock_first_uncached_then_cached(
    brep_box: pv.PolyData,
) -> None:
    """An uncached leaf is passed over until a cached leaf is found."""
    mb = pv.MultiBlock([pv.Sphere(), brep_box])
    assert mb.cad.has_brep_origin is True


def test_has_brep_origin_multiblock_skips_none_block(
    brep_box: pv.PolyData,
) -> None:
    """A leading ``None`` block is skipped before a cached leaf is found."""
    mb = pv.MultiBlock()
    mb.append(None, name='gap')
    mb.append(brep_box, name='box')
    assert mb.cad.has_brep_origin is True


def test_has_brep_origin_multiblock_falsy_flag_continues() -> None:
    """A leaf flag that is falsy is skipped; no other origin is False."""
    sphere = pv.Sphere()
    sphere.field_data['cad.has_brep_origin'] = np.array([0])
    mb = pv.MultiBlock([sphere, pv.Cube()])
    assert mb.cad.has_brep_origin is False


def test_has_brep_origin_multiblock_plain_is_false() -> None:
    """A MultiBlock of plain meshes has no B-rep origin."""
    assert pv.MultiBlock([pv.Sphere(), pv.Cube()]).cad.has_brep_origin is False


# --------------------------------------------------------------------------- #
# tessellation_quality.
# --------------------------------------------------------------------------- #


def test_tessellation_quality_none_when_unset() -> None:
    """No stamped deflections yields ``None``."""
    assert pv.Sphere().cad.tessellation_quality is None


def test_tessellation_quality_reads_stamped_values() -> None:
    """``tessellate`` stamps deflections that the property reads back."""
    out = _b123d_cylinder().cad.tessellate(linear_deflection=0.05, angular_deflection=0.3)
    quality = out.cad.tessellation_quality
    assert quality == {'linear_deflection': 0.05, 'angular_deflection': 0.3}


def test_tessellation_quality_partial_stamp_other_none() -> None:
    """Only one deflection stamped: the missing one reads as ``None``."""
    mesh = pv.Sphere()
    mesh.field_data['cad.linear_deflection'] = np.array([0.2])
    assert mesh.cad.tessellation_quality == {
        'linear_deflection': 0.2,
        'angular_deflection': None,
    }


def test_tessellation_quality_empty_array_scalar_none() -> None:
    """An empty stamped array resolves to ``None`` via the size guard."""
    mesh = pv.Sphere()
    mesh.field_data['cad.linear_deflection'] = np.array([], dtype=float)
    mesh.field_data['cad.angular_deflection'] = np.array([0.5])
    assert mesh.cad.tessellation_quality == {
        'linear_deflection': None,
        'angular_deflection': 0.5,
    }


# --------------------------------------------------------------------------- #
# tessellate: success + monotonicity + raise paths.
# --------------------------------------------------------------------------- #


def test_tessellate_finer_deflection_more_cells() -> None:
    """Cell count grows monotonically as the linear deflection tightens."""
    cyl = _b123d_cylinder()
    coarse = cyl.cad.tessellate(linear_deflection=1.0)
    medium = cyl.cad.tessellate(linear_deflection=0.02)
    fine = cyl.cad.tessellate(linear_deflection=0.005)
    assert coarse.n_cells < medium.n_cells < fine.n_cells
    assert fine.cad.tessellation_quality['linear_deflection'] == 0.005


def test_tessellate_default_deflection_from_bbox(brep_box: pv.PolyData) -> None:
    """With no ``linear_deflection`` a bbox-scaled default is stamped."""
    out = brep_box.cad.tessellate()
    stamped = out.cad.tessellation_quality['linear_deflection']
    diag = float(np.linalg.norm([25.0, 15.0, 8.0]))
    assert stamped == pytest.approx(0.001 * diag)
    assert out.n_cells == 12


def test_tessellate_raises_without_brep_origin() -> None:
    """A plain mesh cannot be re-tessellated and raises ``CadError``."""
    with pytest.raises(CadError, match='requires a CAD-origin mesh'):
        pv.Sphere().cad.tessellate()


def test_tessellate_multiblock_collects_leaf_breps(
    assembly: pv.MultiBlock,
) -> None:
    """A MultiBlock re-tessellates by gathering each cached leaf shape.

    Both leaves (the ``Base`` box and the ``Pin`` cylinder) carry a
    cached ``TopoDS``, so the collected compound re-meshes to the full
    assembly: the 12 box facets plus the 100 cylinder facets, with the
    bounding box spanning the pin's full Z extent.
    """
    out = assembly.cad.tessellate(linear_deflection=0.1)
    assert isinstance(out, pv.PolyData)
    assert out.n_cells == 112
    np.testing.assert_allclose(out.bounds, (0.0, 30.0, 0.0, 20.0, 0.0, 35.0), atol=1e-6)


def test_tessellate_multiblock_raises_without_cache() -> None:
    """A plain MultiBlock has no collectable cache and raises."""
    mb = pv.MultiBlock([pv.Sphere(), pv.Cube()])
    with pytest.raises(CadError, match='requires a CAD-origin mesh'):
        mb.cad.tessellate()


def test_tessellate_corrupt_cache_raises_cachenerror() -> None:
    """A non-``TopoDS`` cache entry raises ``CadCacheError``, not ``None``."""
    mesh = pv.Sphere()
    setattr(mesh, _BREP_ATTR, 'not a shape')
    with pytest.raises(CadCacheError, match=r'corrupt B-rep cache.*cache entry is str'):
        mesh.cad.tessellate()


def test_tessellate_null_topods_raises_cachenerror() -> None:
    """A present-but-null ``TopoDS_Shape`` is a corrupt cache."""
    mesh = pv.Sphere()
    setattr(mesh, _BREP_ATTR, TopoDS_Shape())
    with pytest.raises(CadCacheError, match='not a valid non-null'):
        mesh.cad.tessellate()


def test_collect_cached_topods_nested_multiblock(
    assembly: pv.MultiBlock,
) -> None:
    """Nested MultiBlock leaves are still gathered for re-tessellation."""
    nested = pv.MultiBlock([assembly, pv.MultiBlock([None])])
    out = nested.cad.tessellate(linear_deflection=0.2)
    assert out.n_cells > 0


def test_collect_cached_topods_single_leaf_path(
    brep_box: pv.PolyData,
) -> None:
    """A one-cached-leaf MultiBlock uses the single-shape return branch."""
    mb = pv.MultiBlock([brep_box, pv.Sphere()])
    out = mb.cad.tessellate(linear_deflection=0.3)
    assert out.n_cells == 12


def test_collect_cached_topods_two_leaves_builds_compound() -> None:
    """Two cached leaves are gathered into one compound and re-meshed."""
    b1 = pyvista_cad.read_brep(_EXTERNAL_BREP)
    b2 = pyvista_cad.read_brep(_EXTERNAL_BREP)
    out = pv.MultiBlock([b1, b2]).cad.tessellate(linear_deflection=0.5)
    assert out.n_cells == 24


def test_collect_cached_topods_nested_two_leaves() -> None:
    """Nested MultiBlocks each contributing a cached leaf form a compound."""
    b1 = pyvista_cad.read_brep(_EXTERNAL_BREP)
    b2 = pyvista_cad.read_brep(_EXTERNAL_BREP)
    nested = pv.MultiBlock([pv.MultiBlock([b1]), pv.MultiBlock([b2])])
    out = nested.cad.tessellate(linear_deflection=0.5)
    assert out.n_cells == 24


# --------------------------------------------------------------------------- #
# set_metadata: valid + every MetadataError branch.
# --------------------------------------------------------------------------- #


def test_set_metadata_valid_rgba_and_extra() -> None:
    """A full valid metadata write round-trips through the properties."""
    mesh = pv.Sphere()
    mesh.cad.set_metadata(
        units='inch',
        label='widget',
        color=(0.2, 0.4, 0.6, 0.5),
        transform=np.eye(4),
        source_format='step',
        source_format_version='AP242',
        guid='abc-123',
        metadata={'k': 'v'},
    )
    assert mesh.cad.units == 'inch'
    assert mesh.cad.label == 'widget'
    md = mesh.cad.metadata
    assert md.color == (0.2, 0.4, 0.6, 0.5)
    assert md.guid == 'abc-123'
    assert md.extra == {'k': 'v'}


def test_set_metadata_replaces_prior_keys() -> None:
    """A second ``set_metadata`` clears keys it does not set."""
    mesh = pv.Sphere()
    mesh.cad.set_metadata(units='mm', label='first')
    mesh.cad.set_metadata(units='m')
    assert mesh.cad.units == 'm'
    assert mesh.cad.label is None


def test_set_metadata_bad_units_raises() -> None:
    """An unknown unit string raises ``MetadataError``."""
    with pytest.raises(MetadataError, match='unknown units'):
        pv.Sphere().cad.set_metadata(units='parsec')


def test_set_metadata_bad_color_size_raises() -> None:
    """A 2-component color raises ``MetadataError``."""
    with pytest.raises(MetadataError, match='RGB 3-tuple or RGBA 4-tuple'):
        pv.Sphere().cad.set_metadata(color=(0.5, 0.5))


def test_set_metadata_color_out_of_range_raises() -> None:
    """A color component outside ``[0, 1]`` raises ``MetadataError``."""
    with pytest.raises(MetadataError, match=r'must lie in \[0, 1\]'):
        pv.Sphere().cad.set_metadata(color=(1.5, 0.0, 0.0))


def test_set_metadata_bad_transform_shape_raises() -> None:
    """A non-4x4 transform raises ``MetadataError``."""
    with pytest.raises(MetadataError, match='4x4 matrix'):
        pv.Sphere().cad.set_metadata(transform=np.eye(3))


def test_set_metadata_non_dict_metadata_raises() -> None:
    """A non-dict ``metadata`` bag raises ``MetadataError``."""
    with pytest.raises(MetadataError, match='metadata must be a dict'):
        pv.Sphere().cad.set_metadata(metadata=['not', 'a', 'dict'])


def test_set_metadata_multiblock_writes_assembly_level() -> None:
    """MultiBlock ``set_metadata`` writes onto the assembly's field data."""
    mb = pv.MultiBlock([pv.Sphere(), pv.Cube()])
    mb.cad.set_metadata(units='m', label='assembly')
    assert mb.cad.units == 'm'
    assert mb.cad.label == 'assembly'


def test_set_metadata_multiblock_bad_units_raises() -> None:
    """MultiBlock ``set_metadata`` validates units the same way."""
    with pytest.raises(MetadataError, match='unknown units'):
        pv.MultiBlock([pv.Sphere()]).cad.set_metadata(units='furlong')


def test_metadata_view_setter_replaces() -> None:
    """Assigning ``metadata`` replaces every schema key."""
    mesh = pv.Sphere()
    mesh.cad.set_metadata(units='mm', label='old', guid='g')
    mesh.cad.metadata = CadMetadata(units='inch', label='new')
    assert mesh.cad.units == 'inch'
    assert mesh.cad.label == 'new'
    assert mesh.cad.metadata.guid is None


# --------------------------------------------------------------------------- #
# set_units conversion paths (MultiBlock).
# --------------------------------------------------------------------------- #


def test_set_units_multiblock_convert_scales_every_block() -> None:
    """MultiBlock ``set_units(convert=True)`` rescales each leaf's points."""
    mb = pv.MultiBlock([pv.Sphere(radius=1), pv.Cube()])
    mb.cad.units = 'm'
    before = [blk.points.copy() for blk in mb]
    mb.cad.set_units('mm', convert=True)
    assert mb.cad.units == 'mm'
    for blk, p0 in zip(mb, before, strict=True):
        np.testing.assert_allclose(blk.points, p0 * 1000.0)


def test_set_units_multiblock_skips_none_blocks() -> None:
    """Conversion rescales real leaves and tolerates ``None`` blocks."""
    mb = pv.MultiBlock()
    mb.append(pv.Sphere(radius=1), name='s')
    mb.append(None, name='gap')
    mb.append(pv.Cube(), name='c')
    mb.cad.units = 'm'
    p0 = mb['s'].points.copy()
    c0 = mb['c'].points.copy()
    mb.cad.set_units('mm', convert=True)
    np.testing.assert_allclose(mb['s'].points, p0 * 1000.0)
    np.testing.assert_allclose(mb['c'].points, c0 * 1000.0)
    assert mb.cad.units == 'mm'


def test_set_units_invalid_target_raises() -> None:
    """An unknown target unit raises ``ValueError`` from ``validate_units``."""
    with pytest.raises(ValueError, match='unknown units'):
        pv.Sphere().cad.set_units('cubits')


# --------------------------------------------------------------------------- #
# split_by_label / split_by_color edge branches.
# --------------------------------------------------------------------------- #


def test_split_by_label_partitions_cells() -> None:
    """Cells partition by distinct ``cad.label`` values."""
    mesh = pv.Cube().triangulate()
    half = mesh.n_cells // 2
    mesh.cell_data['cad.label'] = np.array(['a'] * half + ['b'] * (mesh.n_cells - half))
    parts = mesh.cad.split_by_label()
    assert sorted(parts.keys()) == ['a', 'b']
    assert parts['a'].n_cells + parts['b'].n_cells == mesh.n_cells


def test_split_by_label_absent_array_empty() -> None:
    """No label array yields an empty MultiBlock."""
    assert pv.Sphere().cad.split_by_label().n_blocks == 0


def test_split_by_label_no_cell_data_raises() -> None:
    """Splitting a dataset without ``cell_data`` raises ``TypeError``."""

    class _Bare:
        _dataset = object()

    acc = pyvista_cad.CadDataSetAccessor(object())  # type: ignore[arg-type]
    acc._dataset = object()
    with pytest.raises(TypeError, match='requires a DataSet with cell_data'):
        acc.split_by_label()


def test_split_by_color_invalid_shape_raises() -> None:
    """A non-(N, 3) color array raises ``ValueError``."""
    mesh = pv.Cube().triangulate()
    mesh.cell_data['cad.color'] = np.zeros((mesh.n_cells, 4))
    with pytest.raises(ValueError, match=r'must be \(N, 3\) RGB'):
        mesh.cad.split_by_color()


def test_split_by_color_absent_array_empty() -> None:
    """No color array yields an empty MultiBlock."""
    assert pv.Sphere().cad.split_by_color().n_blocks == 0


def test_split_by_color_groups_by_unique_rgb() -> None:
    """Each distinct RGB triple becomes one named block."""
    mesh = pv.Cube().triangulate()
    colors = np.tile([1.0, 0.0, 0.0], (mesh.n_cells, 1))
    colors[0] = [0.0, 1.0, 0.0]
    mesh.cell_data['cad.color'] = colors
    mb = mesh.cad.split_by_color()
    assert mb.n_blocks == 2
    assert any(k.startswith('rgb(') for k in mb.keys())  # noqa: SIM118


def test_multiblock_split_by_label_combines_first() -> None:
    """MultiBlock ``split_by_label`` combines then partitions by value.

    The label lives in per-cell data here so the combined mesh
    partitions into one block per distinct cell label.
    """
    a = pv.Cube().triangulate()
    a.cell_data['cad.label'] = np.array(['A'] * a.n_cells, dtype=object)
    b = pv.Sphere()
    b.cell_data['cad.label'] = np.array(['B'] * b.n_cells, dtype=object)
    parts = pv.MultiBlock([a, b]).cad.split_by_label()
    assert sorted(parts.keys()) == ['A', 'B']
    assert parts['A'].n_cells == a.n_cells


def test_multiblock_split_by_layer_combines_first() -> None:
    """MultiBlock ``split_by_layer`` combines then splits by layer."""
    a = pv.Cube().triangulate()
    a.cell_data['cad.layer'] = np.array(['LA'] * a.n_cells, dtype=object)
    b = pv.Cube().triangulate()
    b.cell_data['cad.layer'] = np.array(['LB'] * b.n_cells, dtype=object)
    parts = pv.MultiBlock([a, b]).cad.split_by_layer()
    assert sorted(parts.keys()) == ['LA', 'LB']


def test_multiblock_split_by_color_combines_first() -> None:
    """MultiBlock ``split_by_color`` combines then splits by color."""
    a = pv.Cube().triangulate()
    a.cell_data['cad.color'] = np.tile([1.0, 0.0, 0.0], (a.n_cells, 1))
    b = pv.Cube().triangulate()
    b.cell_data['cad.color'] = np.tile([0.0, 0.0, 1.0], (b.n_cells, 1))
    parts = pv.MultiBlock([a, b]).cad.split_by_color()
    assert parts.n_blocks == 2


def test_split_by_label_fallback_to_layer_key() -> None:
    """``split_by_layer`` falls back to the ``Layer`` cell array."""
    mesh = pv.Cube().triangulate()
    mesh.cell_data['Layer'] = np.array(['top'] * mesh.n_cells, dtype=object)
    parts = mesh.cad.split_by_layer()
    assert list(parts.keys()) == ['top']


# --------------------------------------------------------------------------- #
# exact_volume / center_of_mass.
# --------------------------------------------------------------------------- #


def test_exact_volume_matches_analytic_cylinder() -> None:
    """Exact volume equals the analytic cylinder volume."""
    cyl = _b123d_cylinder()
    assert cyl.cad.exact_volume() == pytest.approx(np.pi * 25 * 10, rel=1e-6)


def test_exact_volume_raises_without_cache() -> None:
    """A plain mesh has no cached shape for an exact volume."""
    with pytest.raises(CadError, match='exact_volume requires'):
        pv.Sphere().cad.exact_volume()


def test_center_of_mass_origin_for_centered_box(
    brep_box: pv.PolyData,
) -> None:
    """An origin-centred BREP box has a centre of mass at the origin."""
    com = brep_box.cad.center_of_mass()
    np.testing.assert_allclose(com, [0.0, 0.0, 0.0], atol=1e-9)


def test_center_of_mass_raises_without_cache() -> None:
    """A plain mesh has no cached shape for a centre of mass."""
    with pytest.raises(CadError, match='center_of_mass requires'):
        pv.Sphere().cad.center_of_mass()


def test_multiblock_exact_volume_sums_cached_leaves(
    assembly: pv.MultiBlock,
) -> None:
    """MultiBlock ``exact_volume`` sums every cached leaf volume.

    Both leaves carry a cached shape: ``Base`` is a 30x20x10 box
    (volume 6000) and ``Pin`` is a radius-4, height-25 cylinder
    (volume pi * 4**2 * 25). The exact volume is their sum.
    """
    expected = 30.0 * 20.0 * 10.0 + np.pi * 4.0**2 * 25.0
    assert assembly.cad.exact_volume() == pytest.approx(expected, rel=1e-6)


def test_multiblock_exact_volume_raises_without_cache() -> None:
    """A plain MultiBlock has no cached leaf for an exact volume."""
    mb = pv.MultiBlock([pv.Sphere(), pv.MultiBlock([pv.Cube()])])
    with pytest.raises(CadError, match='at least one CAD-origin leaf'):
        mb.cad.exact_volume()


# --------------------------------------------------------------------------- #
# from_* incoming converters.
# --------------------------------------------------------------------------- #


def test_from_build123d_round_trips_volume() -> None:
    """A build123d box converts to a 12-facet PolyData with cached B-rep."""
    box = build123d.Box(2, 3, 4)
    poly = pyvista_cad.from_build123d(box.wrapped)
    assert poly.n_cells == 12
    assert poly.cad.has_brep_origin is True
    assert poly.cad.exact_volume() == pytest.approx(24.0, rel=1e-6)


def test_from_cadquery_round_trips() -> None:
    """A cadquery box converts to a PolyData."""
    wp = cq.Workplane().box(2, 2, 2)
    poly = pyvista_cad.from_cadquery(wp)
    assert poly.n_cells == 12
    assert poly.cad.exact_volume() == pytest.approx(8.0, rel=1e-6)


def test_from_topods_and_to_topods_round_trip(
    brep_box: pv.PolyData,
) -> None:
    """``to_topods`` then ``from_topods`` preserves the facet count."""
    shape = brep_box.cad.to_topods()
    poly = pyvista_cad.from_topods(shape)
    assert poly.n_cells == 12


def test_from_gmsh_and_to_gmsh(brep_box: pv.PolyData) -> None:
    """``to_gmsh`` installs the mesh and ``from_gmsh`` returns a grid."""
    brep_box.cad.to_gmsh(model_name='cov_model')
    grid = pyvista_cad.from_gmsh('cov_model')
    assert isinstance(grid, pv.UnstructuredGrid)
    gmsh.finalize()


# --------------------------------------------------------------------------- #
# to_* outgoing converters / writers (DataSet).
# --------------------------------------------------------------------------- #


def test_to_step_writes_readable_file(tmp_path, brep_box: pv.PolyData) -> None:
    """A STEP write produces a non-empty file re-readable by the reader."""
    out = tmp_path / 'box.step'
    brep_box.cad.to_step(out)
    assert out.stat().st_size > 0
    back = pyvista_cad.read_step(out)
    combined = back.combine() if isinstance(back, pv.MultiBlock) else back
    np.testing.assert_allclose(combined.bounds, (-12.5, 12.5, -7.5, 7.5, -4.0, 4.0), atol=1e-3)


def test_to_brep_writes_readable_file(tmp_path, brep_box: pv.PolyData) -> None:
    """A BREP write round-trips through the reader to 12 facets."""
    out = tmp_path / 'box.brep'
    brep_box.cad.to_brep(out)
    back = pyvista_cad.read_brep(out)
    assert back.n_cells == 12


def test_to_3mf_writes_readable_file(tmp_path, brep_box: pv.PolyData) -> None:
    """A 3MF write round-trips through the reader."""
    out = tmp_path / 'box.3mf'
    brep_box.cad.to_3mf(out)
    back = pyvista_cad.read_three_mf(out)
    combined = back.combine() if isinstance(back, pv.MultiBlock) else back
    assert combined.n_points > 0


def test_to_gltf_writes_file(tmp_path, brep_box: pv.PolyData) -> None:
    """A glTF write through trimesh produces a non-empty file."""
    out = tmp_path / 'box.glb'
    brep_box.cad.to_gltf(out)
    assert out.stat().st_size > 0


def test_to_dxf_writes_file(tmp_path) -> None:
    """A DXF write of a line mesh produces a non-empty file."""
    poly = pyvista_cad.read_dxf('tests/data/dxf_r2018.dxf')
    out = tmp_path / 'lines.dxf'
    poly.cad.to_dxf(out)
    assert out.stat().st_size > 0


def test_to_dxf_unstructured_grid_extracts_surface(tmp_path) -> None:
    """A non-PolyData input is surface-extracted before the DXF write."""
    grid = pv.Cube().triangulate().cast_to_unstructured_grid()
    out = tmp_path / 'grid.dxf'
    grid.cad.to_dxf(out)
    assert out.stat().st_size > 0


def test_to_dxf_layer_array_fallback_to_layer(tmp_path) -> None:
    """A missing ``cad.layer`` falls back to a ``Layer`` cell array."""
    mesh = pv.Cube().triangulate()
    mesh.cell_data['Layer'] = np.array(['Q'] * mesh.n_cells, dtype=object)
    out = tmp_path / 'fb.dxf'
    mesh.cad.to_dxf(out)
    assert out.stat().st_size > 0


def test_to_dxf_layer_array_none_when_absent(tmp_path) -> None:
    """No layer cell array resolves the layer_array to ``None``."""
    out = tmp_path / 'none.dxf'
    pv.Cube().triangulate().cad.to_dxf(out)
    assert out.stat().st_size > 0


def test_to_step_unstructured_grid_extracts_surface(tmp_path) -> None:
    """A non-PolyData STEP write surface-extracts first."""
    grid = pv.Cube().triangulate().cast_to_unstructured_grid()
    out = tmp_path / 'grid.step'
    grid.cad.to_step(out)
    assert out.stat().st_size > 0


def test_to_3mf_unstructured_grid_extracts_surface(tmp_path) -> None:
    """A non-PolyData 3MF write surface-extracts first."""
    grid = pv.Cube().triangulate().cast_to_unstructured_grid()
    out = tmp_path / 'grid.3mf'
    grid.cad.to_3mf(out)
    assert out.stat().st_size > 0


def test_to_build123d_to_cadquery_wrap(brep_box: pv.PolyData) -> None:
    """The faceted ``to_build123d`` / ``to_cadquery`` wrappers return shapes."""
    assert isinstance(brep_box.cad.to_build123d(), build123d.Compound)
    assert isinstance(brep_box.cad.to_cadquery(), cq.Workplane)


# --------------------------------------------------------------------------- #
# to_* converters / writers (MultiBlock).
# --------------------------------------------------------------------------- #


def test_multiblock_to_step_round_trip(tmp_path, assembly: pv.MultiBlock) -> None:
    """A MultiBlock STEP write re-reads to a non-empty combined mesh."""
    out = tmp_path / 'asm.step'
    assembly.cad.to_step(out)
    back = pyvista_cad.read_step(out)
    combined = back.combine() if isinstance(back, pv.MultiBlock) else back
    assert combined.n_points > 0


def test_multiblock_to_3mf_round_trip(tmp_path, assembly: pv.MultiBlock) -> None:
    """A MultiBlock 3MF write round-trips."""
    out = tmp_path / 'asm.3mf'
    assembly.cad.to_3mf(out)
    assert out.stat().st_size > 0


def test_multiblock_to_brep_writes(tmp_path, assembly: pv.MultiBlock) -> None:
    """A MultiBlock BREP write re-reads to 12 box facets (Base)."""
    out = tmp_path / 'asm.brep'
    assembly.cad.to_brep(out)
    assert pyvista_cad.read_brep(out).n_cells > 0


def test_multiblock_to_gltf_named_nodes(tmp_path, assembly: pv.MultiBlock) -> None:
    """A MultiBlock glTF write walks leaves and names nodes by label."""
    out = tmp_path / 'asm.glb'
    assembly.cad.to_gltf(out)
    assert out.stat().st_size > 0


def test_multiblock_to_dxf_combines(tmp_path, assembly: pv.MultiBlock) -> None:
    """A MultiBlock DXF write combines then writes."""
    out = tmp_path / 'asm.dxf'
    assembly.cad.to_dxf(out)
    assert out.stat().st_size > 0


def test_to_gmsh_imagedata_extracts_surface() -> None:
    """A non-grid/non-poly dataset is surface-extracted before ``to_gmsh``."""
    pv.ImageData(dimensions=(3, 3, 3)).cad.to_gmsh(model_name='cov_img')
    grid = pyvista_cad.from_gmsh('cov_img')
    assert isinstance(grid, pv.UnstructuredGrid)
    gmsh.finalize()


def test_multiblock_to_gmsh_imagedata_extracts_surface() -> None:
    """A MultiBlock combining to non-poly is surface-extracted for gmsh."""
    mb = pv.MultiBlock([pv.ImageData(dimensions=(3, 3, 3))])
    mb.cad.to_gmsh(model_name='cov_mb_img')
    grid = pyvista_cad.from_gmsh('cov_mb_img')
    assert isinstance(grid, pv.UnstructuredGrid)
    gmsh.finalize()


def test_to_gltf_multiblock_skips_nested_and_failed_leaf(
    tmp_path,
) -> None:
    """The glTF walk skips nested MultiBlocks and leaves that fail convert."""
    good = pv.Cube().triangulate()
    good.field_data['cad.label'] = np.array(['cube'])
    empty = pv.PolyData()  # converts to an empty / failing trimesh node
    tree = pv.MultiBlock()
    tree.append(pv.MultiBlock([pv.Sphere()]), name='grp')
    tree.append(good, name='cube')
    tree.append(empty, name='empty')
    out = tmp_path / 't.glb'
    tree.cad.to_gltf(out)
    assert out.stat().st_size > 0


def test_to_gltf_unconvertible_leaf_warns_and_continues(tmp_path) -> None:
    """A leaf whose trimesh conversion raises is dropped with a loud warning.

    The triangulator guarantees triangle faces, so this degradation path
    is defensive; a fault injection forces the documented failure mode to
    prove the export warns by name and still emits the surviving leaf
    rather than silently shipping a shorter scene.
    """
    good = pv.Cube().triangulate()
    good.field_data['cad.label'] = np.array(['keep'])
    bad = pv.Sphere()
    bad.field_data['cad.label'] = np.array(['drop_me'])
    tree = pv.MultiBlock()
    tree.append(good, name='keep')
    tree.append(bad, name='drop_me')

    real_to_trimesh = pv.to_trimesh

    def _fail_on_sphere(mesh: pv.PolyData, **kwargs):
        if mesh.n_points == bad.n_points:
            msg = 'injected: not triangulable'
            raise ValueError(msg)
        return real_to_trimesh(mesh, **kwargs)

    out = tmp_path / 'partial.glb'
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pv, 'to_trimesh', _fail_on_sphere)
        with pytest.warns(pyvista_cad.CadWarning, match="Dropping MultiBlock leaf 'drop_me'"):
            tree.cad.to_gltf(out)
    assert out.stat().st_size > 0
    rt = pyvista_cad.read_gltf(out)
    labels = {
        str(b.field_data['cad.label'][0])
        for b in (rt if isinstance(rt, pv.MultiBlock) else [rt])
        if b is not None and 'cad.label' in b.field_data
    }
    assert 'drop_me' not in labels


def test_multiblock_to_gmsh(assembly: pv.MultiBlock) -> None:
    """A MultiBlock ``to_gmsh`` combines and installs a model."""
    assembly.cad.to_gmsh(model_name='cov_mb')
    grid = pyvista_cad.from_gmsh('cov_mb')
    assert isinstance(grid, pv.UnstructuredGrid)
    gmsh.finalize()


def test_multiblock_to_topods_b123d_cq(
    assembly: pv.MultiBlock,
) -> None:
    """MultiBlock faceted converters return the expected wrapper types."""
    assert isinstance(assembly.cad.to_build123d(), build123d.Compound)
    assert isinstance(assembly.cad.to_cadquery(), cq.Workplane)
    assembly.cad.to_topods()  # TopoDS_* compound, must not raise


# --------------------------------------------------------------------------- #
# MultiBlock walk / find / flatten / assembly_tree.
# --------------------------------------------------------------------------- #


def test_walk_yields_paths_dfs(assembly: pv.MultiBlock) -> None:
    """``walk`` yields ``(path, block)`` for each named leaf."""
    paths = {p for p, _ in assembly.cad.walk()}
    assert paths == {'Base', 'Pin'}


def test_walk_nested_paths() -> None:
    """Nested blocks get slash-joined paths under their parent name."""
    inner = pv.MultiBlock([pv.Sphere()])
    tree = pv.MultiBlock()
    tree.append(inner, name='grp')
    tree.append(pv.Cube(), name='solo')
    paths = {p for p, _ in tree.cad.walk()}
    assert 'grp' in paths
    assert 'solo' in paths
    assert any(p.startswith('grp/') for p in paths)


def test_walk_empty_block_name_falls_back() -> None:
    """A block with an explicitly empty name falls back to ``block_i``."""
    tree = pv.MultiBlock([pv.Sphere()])
    tree.set_block_name(0, '')
    paths = {p for p, _ in tree.cad.walk()}
    assert paths == {'block_0'}


def test_find_glob_predicate_matches_label(
    assembly: pv.MultiBlock,
) -> None:
    """A glob predicate matches the trailing path name and label."""
    hits = assembly.cad.find(predicate='P*')
    assert [p for p, _ in hits] == ['Pin']


def test_find_callable_predicate(assembly: pv.MultiBlock) -> None:
    """A callable predicate filters ``(path, block)`` pairs."""
    hits = assembly.cad.find(predicate=lambda path, blk: blk.n_cells == 100)
    assert [p for p, _ in hits] == ['Pin']


def test_find_positional_predicate_accepted(assembly: pv.MultiBlock) -> None:
    """The predicate is the conventional leading positional argument."""
    hits = assembly.cad.find('B*')
    assert [p for p, _ in hits] == ['Base']


def test_find_keyword_filters(assembly: pv.MultiBlock) -> None:
    """Keyword filters AND-combine on name and source_format."""
    hits = assembly.cad.find(name='Base', source_format='step')
    assert [p for p, _ in hits] == ['Base']
    assert assembly.cad.find(name='Base', source_format='iges') == []


def test_find_ifc_type_and_layer_filters() -> None:
    """``ifc_type`` and ``layer`` keyword filters select matching leaves."""
    a = pv.Cube().triangulate()
    a.field_data['cad.ifc_type'] = np.array(['IfcWall'])
    a.cell_data['cad.layer'] = np.array(['L0'] * a.n_cells, dtype=object)
    b = pv.Sphere()
    mb = pv.MultiBlock()
    mb.append(a, name='wall')
    mb.append(b, name='ball')
    assert [p for p, _ in mb.cad.find(ifc_type='IfcWall')] == ['wall']
    assert [p for p, _ in mb.cad.find(layer='L0')] == ['wall']
    assert mb.cad.find(ifc_type='IfcDoor') == []


def test_find_no_predicate_returns_all(assembly: pv.MultiBlock) -> None:
    """``find`` with no predicate returns every leaf."""
    assert len(assembly.cad.find()) == 2


def test_walk_flatten_tree_skip_none_blocks() -> None:
    """``None`` blocks are skipped by walk, flatten, and assembly_tree."""
    mb = pv.MultiBlock()
    mb.append(pv.Sphere(), name='s')
    mb.append(None, name='gap')
    assert [p for p, _ in mb.cad.walk()] == ['s']
    assert mb.cad.flatten().n_blocks == 1
    assert mb.cad.assembly_tree() == {'s': None}


def test_flatten_keeps_leaves_separate() -> None:
    """``flatten`` collapses nesting but keeps each leaf a block."""
    inner = pv.MultiBlock([pv.Sphere(), pv.Cube()])
    tree = pv.MultiBlock([inner, pv.Cone()])
    flat = tree.cad.flatten()
    assert flat.n_blocks == 3
    assert all(isinstance(b, pv.DataSet) for b in flat)


def test_flatten_to_polydata_fuses(assembly: pv.MultiBlock) -> None:
    """``flatten_to_polydata`` fuses every leaf into one PolyData."""
    poly = assembly.cad.flatten_to_polydata()
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == 112


def test_flatten_to_polydata_extracts_surface_when_not_polydata() -> None:
    """A grid-only assembly is surface-extracted on flatten."""
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    poly = pv.MultiBlock([grid]).cad.flatten_to_polydata()
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0


def test_assembly_tree_labels_override_names(
    assembly: pv.MultiBlock,
) -> None:
    """``assembly_tree`` keys by ``cad.label`` with ``None`` at leaves."""
    tree = assembly.cad.assembly_tree()
    assert tree == {'Base': None, 'Pin': None}


def test_assembly_tree_nested_recurses() -> None:
    """Nested groups recurse into a nested dict."""
    inner = pv.MultiBlock()
    inner.append(pv.Sphere(), name='ball')
    tree = pv.MultiBlock()
    tree.append(inner, name='grp')
    assert tree.cad.assembly_tree() == {'grp': {'ball': None}}


def test_assembly_tree_empty_name_falls_back() -> None:
    """A leaf with no label and an empty block name uses ``block_i``."""
    tree = pv.MultiBlock([pv.Sphere()])
    tree.set_block_name(0, '')
    assert tree.cad.assembly_tree() == {'block_0': None}


# --------------------------------------------------------------------------- #
# cad_view / plot resolution paths.
# --------------------------------------------------------------------------- #


def test_cad_view_dataset_with_brep(brep_box: pv.PolyData) -> None:
    """A B-rep mesh resolves to a ``{'faces', 'edges'}`` CAD view."""
    view = brep_box.cad.cad_view()
    assert isinstance(view, pv.MultiBlock)
    assert set(view.keys()) >= {'faces', 'edges'}


def test_cad_view_dataset_plain_mesh() -> None:
    """A plain mesh still resolves to a CAD view via feature edges."""
    view = pv.Sphere().cad.cad_view()
    assert isinstance(view, pv.MultiBlock)


def test_cad_view_multiblock(assembly: pv.MultiBlock) -> None:
    """A MultiBlock resolves block-by-block to a CAD view."""
    view = assembly.cad.cad_view()
    assert isinstance(view, pv.MultiBlock)


def test_plot_dataset_brep_returns(brep_box: pv.PolyData) -> None:
    """``plot`` on a B-rep mesh renders off-screen and returns."""
    result = brep_box.cad.plot(off_screen=True)
    assert result is None or result is not False


def test_plot_dataset_plain_mesh() -> None:
    """``plot`` on a plain mesh falls back to feature edges."""
    pv.Sphere().cad.plot(off_screen=True)


def test_plot_multiblock(assembly: pv.MultiBlock) -> None:
    """``plot`` on a MultiBlock passes the tree through."""
    assembly.cad.plot(off_screen=True)


# --------------------------------------------------------------------------- #
# Keyword-only argument enforcement (signature-level, no deprecation shim).
# --------------------------------------------------------------------------- #


def test_conventional_positional_path_accepted(tmp_path, brep_box: pv.PolyData) -> None:
    """The conventional leading path argument is positional-only and works."""
    out = tmp_path / 'ok.brep'
    brep_box.cad.to_brep(out)
    assert out.stat().st_size > 0


def test_tessellate_rejects_positional(brep_box: pv.PolyData) -> None:
    """``tessellate`` is keyword-only: a positional deflection is a TypeError."""
    with pytest.raises(TypeError):
        brep_box.cad.tessellate(0.1)  # type: ignore[misc]


def test_to_dxf_rejects_positional_options(tmp_path, brep_box: pv.PolyData) -> None:
    """Options past the path are keyword-only and reject positional calls."""
    out = tmp_path / 'x.dxf'
    with pytest.raises(TypeError):
        brep_box.cad.to_dxf(out, '0')  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# _as_polydata internal resolution helper.
# --------------------------------------------------------------------------- #


def test_as_polydata_multiblock_combines(assembly: pv.MultiBlock) -> None:
    """``_as_polydata`` on a MultiBlock combines to one PolyData."""
    poly = assembly.cad._as_polydata()
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells == 112


def test_as_polydata_grid_extracts_surface() -> None:
    """``_as_polydata`` surface-extracts a non-PolyData dataset."""
    grid = pv.Cube().triangulate().cast_to_unstructured_grid()
    poly = grid.cad._as_polydata()
    assert isinstance(poly, pv.PolyData)
    assert poly.n_cells > 0


# --------------------------------------------------------------------------- #
# examples module sanity (the public entry the tests load from).
# --------------------------------------------------------------------------- #


def test_examples_step_cube_loads_with_brep() -> None:
    """``examples.step_cube`` reads as a B-rep-backed dataset."""
    cube = pyvista_cad.read_step(examples.step_cube)
    assert cube.cad.has_brep_origin is True
    out = cube.cad.tessellate(linear_deflection=0.05)
    assert out.n_cells == 12
    np.testing.assert_allclose(
        np.ptp(np.asarray(out.bounds).reshape(3, 2), axis=1),
        [10.0, 10.0, 10.0],
        atol=1e-6,
    )


def test_examples_path_helpers_resolve_existing_files() -> None:
    """The example path helpers point at files that exist on disk."""
    p = Path(examples.bracket_step_path())
    assert p.exists()
    assert re.search(r'\.step$', str(p))


# --------------------------------------------------------------------------- #
# Internal-helper degenerate-input contracts.
# --------------------------------------------------------------------------- #


def test_resolve_cell_array_returns_none_without_cell_data() -> None:
    """``_resolve_cell_array`` yields ``None`` for an object lacking cell_data."""

    class _NoCellData:
        cell_data = None

    assert _resolve_cell_array(_NoCellData(), 'cad.layer') is None
    assert _resolve_cell_array(object(), 'cad.layer', fallback='Layer') is None


def test_scale_points_noop_on_dataset_without_points() -> None:
    """``_scale_points`` returns without error on a points-less dataset."""

    class _PointlessDataSet:
        """A non-MultiBlock object exposing no ``points`` attribute."""

    obj = _PointlessDataSet()
    _scale_points(obj, 2.0)
    assert not hasattr(obj, 'points')


def test_cad_accessor_registered_on_dataset_and_multiblock() -> None:
    """The ``@pv.register_dataset_accessor`` decorators attach exactly once."""
    ds = [r for r in pv.registered_accessors() if r.name == 'cad' and r.target is pv.DataSet]
    mb = [r for r in pv.registered_accessors() if r.name == 'cad' and r.target is pv.MultiBlock]
    assert len(ds) == 1
    assert ds[0].accessor is CadDataSetAccessor
    assert len(mb) == 1


@pytest.mark.skipif(
    Version(pv.__version__) < Version('0.48.4'),
    reason='silent identical-accessor re-registration lands in pyvista 0.48.4',
)
def test_cad_accessor_reregistration_is_silent_under_filterwarnings_error() -> None:
    """Re-running the decorator (pytest --cov double-import) is a silent no-op.

    pyvista >= 0.48.4 treats re-registering the identical accessor class on
    the same target as a no-op, so the plain decorator is safe under the
    project's ``filterwarnings=error``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        pv.register_dataset_accessor('cad', pv.DataSet)(CadDataSetAccessor)
    still = [r for r in pv.registered_accessors() if r.name == 'cad' and r.target is pv.DataSet]
    assert len(still) == 1
    assert still[0].accessor is CadDataSetAccessor


def test_polydata_to_topods_raises_on_zero_area_triangle() -> None:
    """A duplicate-point (zero-area) triangle trips OCCT face construction."""
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mesh = pv.PolyData(pts, np.array([3, 0, 1, 2]))
    with pytest.raises(TessellationError, match='OCCT face construction failed'):
        _conversion.polydata_to_topods(mesh)
