"""Tests for ``.cad.walk``, ``.cad.find``, and ``.cad.flatten``."""

import numpy as np
import pyvista as pv

import pyvista_cad  # noqa: F401


def _tagged(name: str, ifc_type: str | None = None) -> pv.PolyData:
    mesh = pv.Cube()
    mesh.field_data['cad.label'] = np.array([name])
    if ifc_type:
        mesh.field_data['cad.ifc_type'] = np.array([ifc_type])
    return mesh


def test_walk_nested() -> None:
    inner = pv.MultiBlock({'a': _tagged('A'), 'b': _tagged('B')})
    root = pv.MultiBlock({'inner': inner, 'c': _tagged('C')})
    paths = [p for p, _ in root.cad.walk()]
    assert 'inner' in paths
    assert 'inner/a' in paths
    assert 'inner/b' in paths
    assert 'c' in paths


def test_find_by_ifc_type() -> None:
    root = pv.MultiBlock(
        {
            'w1': _tagged('w1', ifc_type='IfcWall'),
            'w2': _tagged('w2', ifc_type='IfcWall'),
            's1': _tagged('s1', ifc_type='IfcSlab'),
        }
    )
    walls = root.cad.find(ifc_type='IfcWall')
    assert len(walls) == 2
    assert {p for p, _ in walls} == {'w1', 'w2'}


def test_find_by_name() -> None:
    inner = pv.MultiBlock({'target': _tagged('inner-target')})
    root = pv.MultiBlock({'group': inner, 'target': _tagged('root-target')})
    out = root.cad.find(name='target')
    paths = {p for p, _ in out}
    assert paths == {'group/target', 'target'}


def test_flatten() -> None:
    inner = pv.MultiBlock({'a': _tagged('A'), 'b': _tagged('B')})
    root = pv.MultiBlock({'inner': inner, 'c': _tagged('C')})
    flat = root.cad.flatten()
    assert isinstance(flat, pv.MultiBlock)
    keys = sorted(flat.keys())
    assert keys == ['c', 'inner/a', 'inner/b']


def _cube(center: tuple[float, float, float]) -> pv.PolyData:
    return pv.Cube(center=center)


def test_drop_spatial_outliers_removes_far_marker() -> None:
    root = pv.MultiBlock(
        {
            'a': _cube((0, 0, 0)),
            'b': _cube((1, 0, 0)),
            'c': _cube((0, 1, 0)),
            'd': _cube((1, 1, 0)),
            'geo-reference': _cube((500, 500, 0)),
        }
    )
    pruned = root.cad.drop_spatial_outliers()
    assert sorted(pruned.keys()) == ['a', 'b', 'c', 'd']
    # Original is untouched (new tree, shared geometry).
    assert root.n_blocks == 5


def test_drop_spatial_outliers_preserves_hierarchy_and_field_data() -> None:
    inner = pv.MultiBlock({'a': _cube((0, 0, 0)), 'b': _cube((1, 0, 0))})
    inner.field_data['cad.label'] = np.array(['storey'])
    root = pv.MultiBlock(
        {
            'inner': inner,
            'c': _cube((0, 1, 0)),
            'd': _cube((1, 1, 0)),
            'marker': _cube((1e4, 1e4, 1e4)),
        }
    )
    root.field_data['cad.source_format'] = np.array(['ifc'])
    pruned = root.cad.drop_spatial_outliers()
    assert 'marker' not in pruned.keys()  # noqa: SIM118  # MultiBlock `in` tests data
    assert isinstance(pruned['inner'], pv.MultiBlock)
    assert sorted(pruned['inner'].keys()) == ['a', 'b']
    assert str(pruned.field_data['cad.source_format'][0]) == 'ifc'
    assert str(pruned['inner'].field_data['cad.label'][0]) == 'storey'


def test_drop_spatial_outliers_drops_emptied_container() -> None:
    stray = pv.MultiBlock({'far': _cube((1e6, 0, 0))})
    root = pv.MultiBlock(
        {
            'a': _cube((0, 0, 0)),
            'b': _cube((1, 0, 0)),
            'c': _cube((0, 1, 0)),
            'd': _cube((1, 1, 0)),
            'stray_group': stray,
        }
    )
    pruned = root.cad.drop_spatial_outliers()
    assert 'stray_group' not in pruned.keys()  # noqa: SIM118  # MultiBlock `in` tests data


def test_drop_spatial_outliers_too_few_blocks_is_noop() -> None:
    root = pv.MultiBlock({'a': _cube((0, 0, 0)), 'far': _cube((1e6, 0, 0))})
    pruned = root.cad.drop_spatial_outliers()
    assert sorted(pruned.keys()) == ['a', 'far']


def test_drop_spatial_outliers_degenerate_spread_is_noop() -> None:
    # All centroids coincide: MAD == 0, no robust cut-off exists.
    root = pv.MultiBlock({k: _cube((0, 0, 0)) for k in 'abcde'})
    pruned = root.cad.drop_spatial_outliers()
    assert sorted(pruned.keys()) == list('abcde')
