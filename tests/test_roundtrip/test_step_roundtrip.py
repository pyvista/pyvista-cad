"""STEP read -> write -> re-read behavioral invariants."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad

pytest.importorskip('OCP')
b3d = pytest.importorskip('build123d')


def _assembly() -> pv.MultiBlock:
    """A two-part build123d compound, exported as a MultiBlock fixture."""
    a = b3d.Box(1.0, 2.0, 3.0)
    b = b3d.Box(2.0, 2.0, 2.0).translate((10.0, 0.0, 0.0))
    return pyvista_cad.from_build123d(b3d.Compound(children=[a, b]))


def test_step_round_trip_preserves_n_blocks(tmp_path: Path) -> None:
    """A round-tripped STEP MultiBlock keeps its exact block count.

    The build123d Compound carrying two children is emitted As-Is and the
    OCP reader re-splits it on the way back in, yielding the same two
    blocks. If the writer regresses to a flatten-to-one-compound behaviour
    this test lights up immediately.
    """
    src = _assembly()
    out = tmp_path / 'asm.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out)
    assert isinstance(rt, pv.MultiBlock)
    assert rt.n_blocks == 2


def test_step_round_trip_preserves_units(tmp_path: Path) -> None:
    """STEP units survive round-trip via the writer/reader."""
    src = _assembly()
    src.field_data['cad.units'] = np.array(['mm'])
    out = tmp_path / 'asm.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out)
    assert str(rt.field_data['cad.units'][0]) == 'mm'


def test_step_round_trip_preserves_labels(tmp_path: Path) -> None:
    """Round-tripped labels' stems must intersect the source MultiBlock keys.

    Compares on the stem (trailing digits/underscores stripped) because OCAF
    suffixes part copies as ``_1``, ``_2``. Pure non-emptiness was the prior
    contract and is a tautology — the reader synthesises a label even on
    fully anonymous compounds.
    """
    src = _assembly()
    src_keys = list(src.keys())
    assert src_keys, 'fixture must have at least one block'
    out = tmp_path / 'asm.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out)
    rt_labels = []
    for block in rt:
        if block is None:
            continue
        assert 'cad.label' in block.field_data
        label = str(block.field_data['cad.label'][0])
        assert label != ''
        rt_labels.append(label)

    # Strip trailing digits/underscores OCAF appends to duplicates.
    def _stem(s: str) -> str:
        return s.rstrip(' _0123456789').lower()

    src_stems = {_stem(k) for k in src_keys}
    rt_stems = {_stem(lab) for lab in rt_labels}
    assert src_stems & rt_stems, (
        f'no round-tripped label matched source: src={src_keys} rt={rt_labels}'
    )


def test_step_round_trip_recovers_bounds(tmp_path: Path) -> None:
    """Combined bounds round-trip within a tessellation-tolerance band."""
    src = _assembly()
    src_combined = src.combine()
    out = tmp_path / 'asm.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out)
    rt_combined = rt.combine() if isinstance(rt, pv.MultiBlock) else rt
    np.testing.assert_allclose(
        np.asarray(rt_combined.bounds),
        np.asarray(src_combined.bounds),
        atol=0.01,
    )


def test_step_cross_format_3mf_vertex_count_within_1pct(tmp_path: Path) -> None:
    """STEP read -> 3MF write -> 3MF read keeps the tessellated vertex count
    within 1% (allowing for tessellator drift)."""
    pytest.importorskip('lib3mf')

    src = pyvista_cad.read_step(pyvista_cad.examples.step_cube)
    src_combined = src.combine().extract_surface(algorithm='dataset_surface')
    src_pts = src_combined.n_points

    path = tmp_path / 'cube.3mf'
    pyvista_cad.write_three_mf(src, path)
    rt = pyvista_cad.read_three_mf(path)
    rt_combined = rt.combine().extract_surface(algorithm='dataset_surface')

    drift = abs(rt_combined.n_points - src_pts) / max(src_pts, 1)
    assert drift < 0.01, f'point-count drift {drift:.4f} exceeds 1%'

    # Vertex-count conservation alone does not prove geometric fidelity: a
    # tessellator-resolution regression can keep the count while shifting
    # vertices. Pin the geometry with a combined-bounds check plus a
    # symmetric Hausdorff bound so a silent re-tessellation is caught.
    np.testing.assert_allclose(
        np.asarray(rt_combined.bounds),
        np.asarray(src_combined.bounds),
        atol=0.01,
    )
    fwd = src_combined.compute_implicit_distance(rt_combined)
    bwd = rt_combined.compute_implicit_distance(src_combined)
    hausdorff = max(
        float(np.abs(fwd.point_data['implicit_distance']).max()),
        float(np.abs(bwd.point_data['implicit_distance']).max()),
    )
    assert hausdorff < 0.01, f'symmetric Hausdorff {hausdorff:.5f} exceeds 0.01'
