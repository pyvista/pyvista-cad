"""Tests for the IFC reader."""

import json
from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad

ifcopenshell = pytest.importorskip('ifcopenshell')
pytest.importorskip('ifcopenshell.util.representation')
pytest.importorskip('ifcopenshell.api')


@pytest.fixture
def simple_ifc_path(tmp_path: Path) -> Path:
    import importlib
    import sys

    # The lazy-imports test pops ``ifcopenshell`` from ``sys.modules``;
    # later in-process re-imports may leave the parent package without
    # its ``util`` / ``api`` submodule attributes. Force a fresh reload.
    for mod_name in list(sys.modules):
        if mod_name == 'ifcopenshell' or mod_name.startswith('ifcopenshell.'):
            sys.modules.pop(mod_name, None)
    importlib.import_module('ifcopenshell')
    importlib.import_module('ifcopenshell.util.representation')
    api = importlib.import_module('ifcopenshell.api')

    m = api.run('project.create_file', version='IFC4')
    prj = api.run('root.create_entity', m, ifc_class='IfcProject', name='Demo')
    api.run('unit.assign_unit', m, length={'is_metric': True, 'raw': 'METERS'})
    ctx = api.run('context.add_context', m, context_type='Model')
    body = api.run(
        'context.add_context',
        m,
        context_type='Model',
        context_identifier='Body',
        target_view='MODEL_VIEW',
        parent=ctx,
    )
    site = api.run('root.create_entity', m, ifc_class='IfcSite', name='Site')
    api.run('aggregate.assign_object', m, products=[site], relating_object=prj)
    bld = api.run('root.create_entity', m, ifc_class='IfcBuilding', name='Bldg')
    api.run('aggregate.assign_object', m, products=[bld], relating_object=site)
    storey = api.run('root.create_entity', m, ifc_class='IfcBuildingStorey', name='Storey')
    api.run('aggregate.assign_object', m, products=[storey], relating_object=bld)
    wall = api.run('root.create_entity', m, ifc_class='IfcWall', name='Wall')
    rep = api.run(
        'geometry.add_wall_representation',
        m,
        context=body,
        length=4.0,
        height=3.0,
        thickness=0.2,
    )
    api.run('geometry.assign_representation', m, product=wall, representation=rep)
    api.run('spatial.assign_container', m, products=[wall], relating_structure=storey)

    path = tmp_path / 'demo.ifc'
    m.write(str(path))
    return path


def _find_wall(mb: pv.MultiBlock) -> pv.PolyData:
    """Recursively locate the first PolyData leaf in a nested MultiBlock."""
    for block in mb:
        if isinstance(block, pv.MultiBlock):
            leaf = _find_wall(block)
            if leaf is not None:
                return leaf
        elif isinstance(block, pv.PolyData) and block.n_cells > 0:
            return block
    return None  # type: ignore[return-value]


def test_read_ifc_wall(simple_ifc_path: Path) -> None:
    mb = pyvista_cad.read_ifc(simple_ifc_path)
    assert isinstance(mb, pv.MultiBlock)
    # Root is IfcProject; nested structure mirrors spatial decomposition.
    assert str(mb.field_data['cad.ifc_type'][0]) == 'IfcProject'
    assert str(mb.field_data['cad.backend'][0]) == 'ifcopenshell'
    assert str(mb.field_data['cad.units'][0]) == 'm'

    wall = _find_wall(mb)
    assert wall is not None
    assert wall.n_cells > 0
    assert str(wall.field_data['cad.ifc_type'][0]) == 'IfcWall'
    assert str(wall.field_data['cad.source_format'][0]) == 'ifc'
    assert str(wall.field_data['cad.backend'][0]) == 'ifcopenshell'
    assert str(wall.field_data['cad.label'][0]) == 'Wall'
    assert str(wall.field_data['cad.units'][0]) == 'm'
    assert len(str(wall.field_data['cad.guid'][0])) == 22


def test_read_ifc_nested_hierarchy(simple_ifc_path: Path) -> None:
    """Project -> Site -> Building -> Storey -> Wall."""
    mb = pyvista_cad.read_ifc(simple_ifc_path)
    assert mb.n_blocks == 1  # IfcSite
    site = mb[0]
    assert isinstance(site, pv.MultiBlock)
    assert str(site.field_data['cad.ifc_type'][0]) == 'IfcSite'
    building = site[0]
    assert isinstance(building, pv.MultiBlock)
    assert str(building.field_data['cad.ifc_type'][0]) == 'IfcBuilding'
    storey = building[0]
    assert isinstance(storey, pv.MultiBlock)
    assert str(storey.field_data['cad.ifc_type'][0]) == 'IfcBuildingStorey'
    wall = storey[0]
    assert isinstance(wall, pv.PolyData)
    assert str(wall.field_data['cad.ifc_type'][0]) == 'IfcWall'


def test_read_ifc_psets(simple_ifc_path: Path) -> None:
    """Pset payload, if any, is JSON-encoded on cad.psets."""
    mb = pyvista_cad.read_ifc(simple_ifc_path)
    wall = _find_wall(mb)
    assert wall is not None
    if 'cad.psets' in wall.field_data:
        decoded = json.loads(str(wall.field_data['cad.psets'][0]))
        assert isinstance(decoded, dict)


def test_read_ifc_pset_numeric_precision(tmp_path: Path) -> None:
    """A high-precision ``IfcReal`` pset value round-trips to within 1e-12.

    Pinning the old ``json.dumps(..., default=str)`` path: any IFC
    measure wrapper that escaped to ``default`` was stringified, which
    collapses precision and breaks ``json.loads`` round-trips.
    """
    import importlib
    import sys

    for mod_name in list(sys.modules):
        if mod_name == 'ifcopenshell' or mod_name.startswith('ifcopenshell.'):
            sys.modules.pop(mod_name, None)
    importlib.import_module('ifcopenshell')
    api = importlib.import_module('ifcopenshell.api')

    m = api.run('project.create_file', version='IFC4')
    prj = api.run('root.create_entity', m, ifc_class='IfcProject', name='Demo')
    api.run('unit.assign_unit', m, length={'is_metric': True, 'raw': 'METERS'})
    ctx = api.run('context.add_context', m, context_type='Model')
    body = api.run(
        'context.add_context',
        m,
        context_type='Model',
        context_identifier='Body',
        target_view='MODEL_VIEW',
        parent=ctx,
    )
    site = api.run('root.create_entity', m, ifc_class='IfcSite', name='Site')
    api.run('aggregate.assign_object', m, products=[site], relating_object=prj)
    bld = api.run('root.create_entity', m, ifc_class='IfcBuilding', name='B')
    api.run('aggregate.assign_object', m, products=[bld], relating_object=site)
    storey = api.run('root.create_entity', m, ifc_class='IfcBuildingStorey', name='S')
    api.run('aggregate.assign_object', m, products=[storey], relating_object=bld)
    wall = api.run('root.create_entity', m, ifc_class='IfcWall', name='Wall')
    rep = api.run(
        'geometry.add_wall_representation',
        m,
        context=body,
        length=4.0,
        height=3.0,
        thickness=0.2,
    )
    api.run('geometry.assign_representation', m, product=wall, representation=rep)
    api.run('spatial.assign_container', m, products=[wall], relating_structure=storey)

    pset = api.run('pset.add_pset', m, product=wall, name='Pset_X')
    api.run('pset.edit_pset', m, pset=pset, properties={'Y': 1.234567890123})

    path = tmp_path / 'precision.ifc'
    m.write(str(path))

    mb = pyvista_cad.read_ifc(path)
    wall_block = _find_wall(mb)
    assert wall_block is not None
    assert 'cad.psets' in wall_block.field_data
    decoded = json.loads(str(wall_block.field_data['cad.psets'][0]))
    assert decoded['Pset_X']['Y'] == pytest.approx(1.234567890123, abs=1e-12)


def test_pset_json_encoder_preserves_precision() -> None:
    """The custom encoder keeps ``float`` / ``Decimal`` / wrapped numerics numeric."""
    from decimal import Decimal

    from pyvista_cad._backends._ifcopenshell import _PsetJSONEncoder

    class _Measure:
        """Mimic an ifcopenshell measure wrapper."""

        def __init__(self, val: float) -> None:
            self.wrappedValue = val  # IfcMeasure attribute name

    payload = {
        'a_float': 1.234567890123,
        'a_decimal': Decimal('1.234567890123'),
        'a_measure': _Measure(1.234567890123),
    }
    encoded = json.dumps(payload, cls=_PsetJSONEncoder)
    decoded = json.loads(encoded)
    assert decoded['a_float'] == pytest.approx(1.234567890123, abs=1e-12)
    assert decoded['a_decimal'] == pytest.approx(1.234567890123, abs=1e-12)
    assert decoded['a_measure'] == pytest.approx(1.234567890123, abs=1e-12)


def test_read_ifc_pv_read_dispatch(simple_ifc_path: Path) -> None:
    mb = pv.read(simple_ifc_path)
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 1


def test_read_ifc_include_filter(simple_ifc_path: Path) -> None:
    walls = pyvista_cad.read_ifc(simple_ifc_path, include=('IfcWall',))
    assert _find_wall(walls) is not None
    nothing = pyvista_cad.read_ifc(simple_ifc_path, include=('IfcSlab',))
    assert nothing.n_blocks == 0


def test_read_building_ifc_fixture_color_fidelity(building_ifc_path: Path) -> None:
    """The committed ``building.ifc`` fixture: the styled wall reads back its
    authored diffuse ``IfcColourRgb`` (0.8, 0.3, 0.2) and known GlobalId."""
    import numpy as np

    mb = pyvista_cad.read_ifc(building_ifc_path)
    assert isinstance(mb, pv.MultiBlock)
    assert str(mb.field_data['cad.units'][0]) == 'm'
    cb = mb.combine().bounds
    assert (cb.x_min, cb.x_max) == pytest.approx((0.0, 4.0), abs=1e-3)
    assert (cb.y_min, cb.y_max) == pytest.approx((0.0, 0.2), abs=1e-3)
    assert (cb.z_min, cb.z_max) == pytest.approx((0.0, 3.0), abs=1e-3)

    wall = _find_wall(mb)
    assert wall is not None
    assert str(wall.field_data['cad.ifc_type'][0]) == 'IfcWall'
    assert str(wall.field_data['cad.guid'][0]) == '0eCpv9L5zgTO2Bbg6ijyBD'
    np.testing.assert_allclose(
        np.asarray(wall.field_data['cad.color'], dtype=float)[:3],
        [0.8, 0.3, 0.2],
        atol=1e-3,
    )


@pytest.mark.filterwarnings('ignore::pytest.PytestUnraisableExceptionWarning')
def test_read_ifc_not_ifc_text_raises(tmp_path: Path) -> None:
    """A plain text file with no IFC header raises ``CadReadError``.

    ``ifcopenshell.file.__del__`` emits an unraisable ``KeyError`` when the
    failed-parse object is finalised; that's an upstream wart unrelated to
    our wrapper, so the warning is filtered.
    """
    from pyvista_cad import CadReadError

    bad = tmp_path / 'not_ifc.ifc'
    bad.write_text('this is not an IFC file, just plain text\n' * 10)
    with pytest.raises(CadReadError):
        pyvista_cad.read_ifc(bad)
