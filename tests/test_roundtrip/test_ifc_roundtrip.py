"""IFC read invariants.

An IFC writer is not implemented in pyvista-cad; the audited round-trip
invariants (hierarchy depth, GlobalIds, units) are pinned against the
read-side here so a future writer can plug in without breaking the
assertions.
"""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('ifcopenshell')
pytest.importorskip('ifcopenshell.util.representation')
pytest.importorskip('ifcopenshell.api')


@pytest.fixture
def simple_ifc_path(tmp_path: Path) -> Path:
    """Replicate the IFC fixture used in tests/test_readers/test_ifc.py."""
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


def _collect_guids(mb: pv.MultiBlock) -> set[str]:
    """Walk every nested leaf and gather its ``cad.guid``."""
    guids: set[str] = set()
    for block in mb:
        if isinstance(block, pv.MultiBlock):
            guids |= _collect_guids(block)
        elif block is not None and 'cad.guid' in block.field_data:
            guids.add(str(block.field_data['cad.guid'][0]))
    return guids


def _max_depth(mb: pv.MultiBlock, current: int = 1) -> int:
    deepest = current
    for block in mb:
        if isinstance(block, pv.MultiBlock):
            deepest = max(deepest, _max_depth(block, current + 1))
    return deepest


def test_ifc_hierarchy_depth_preserved(simple_ifc_path: Path) -> None:
    """Site -> Building -> Storey -> Wall yields four nested MultiBlock levels."""
    mb = pyvista_cad.read_ifc(simple_ifc_path)
    # Top-level mb is IfcProject; under it: Site, Building, Storey, Wall (leaf).
    assert _max_depth(mb) >= 4


def test_ifc_global_ids_preserved(simple_ifc_path: Path) -> None:
    """Every IFC product carries an IFC GlobalId on ``cad.guid``."""
    import ifcopenshell  # noqa: PLC0415

    f = ifcopenshell.open(str(simple_ifc_path))
    source_guids = {p.GlobalId for p in f.by_type('IfcProduct')}

    mb = pyvista_cad.read_ifc(simple_ifc_path)
    read_guids = _collect_guids(mb)

    # Every product GlobalId encountered by the reader must come from the source set.
    assert read_guids.issubset(source_guids)
    assert read_guids, 'reader produced no cad.guid values'


def test_ifc_units_preserved(simple_ifc_path: Path) -> None:
    """The IFC project's METERS unit is surfaced on ``cad.units``."""
    mb = pyvista_cad.read_ifc(simple_ifc_path)
    assert str(mb.field_data['cad.units'][0]) == 'm'


def _find_wall(mb: pv.MultiBlock) -> pv.PolyData | None:
    for block in mb:
        if isinstance(block, pv.MultiBlock):
            leaf = _find_wall(block)
            if leaf is not None:
                return leaf
        elif isinstance(block, pv.PolyData) and block.n_cells > 0:
            return block
    return None


def test_ifc_building_fixture_color_and_guid_read_fidelity(
    building_ifc_path: Path,
) -> None:
    """The styled wall in ``building.ifc`` reads back its authored diffuse
    ``IfcColourRgb`` (0.8, 0.3, 0.2) on ``cad.color`` and its deterministic
    ``GlobalId`` on ``cad.guid``.

    This is the substantive INV-017 coverage: real material/color *read*
    fidelity against a fixture that carries a diffuse style. (The README
    flagged a possible reader color defect; it does not reproduce here,
    the colour surfaces correctly.)
    """
    mb = pyvista_cad.read_ifc(building_ifc_path)
    assert isinstance(mb, pv.MultiBlock)
    assert str(mb.field_data['cad.units'][0]) == 'm'
    assert str(mb.field_data['cad.backend'][0]) == 'ifcopenshell'

    cb = mb.combine().bounds
    assert (cb.x_min, cb.x_max) == pytest.approx((0.0, 4.0), abs=1e-3)
    assert (cb.y_min, cb.y_max) == pytest.approx((0.0, 0.2), abs=1e-3)
    assert (cb.z_min, cb.z_max) == pytest.approx((0.0, 3.0), abs=1e-3)

    wall = _find_wall(mb)
    assert wall is not None
    assert str(wall.field_data['cad.guid'][0]) == '0eCpv9L5zgTO2Bbg6ijyBD'
    assert 'cad.color' in wall.field_data
    color = np.asarray(wall.field_data['cad.color'], dtype=float)
    np.testing.assert_allclose(color[:3], [0.8, 0.3, 0.2], atol=1e-3)


def test_ifc_writer_intentionally_absent_for_v1() -> None:
    """``write_ifc`` is deliberately not shipped for v1.0: the package
    scope is CAD ingestion plus visualization fidelity, not BIM
    authoring (IFC material/color read fidelity stays in scope).

    This is a one-line public-surface guard, not a skip: the real IFC
    material/color coverage is the read-fidelity test above.
    """
    assert not hasattr(pyvista_cad, 'write_ifc')
