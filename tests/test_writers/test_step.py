"""Tests for the STEP writer."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial import cKDTree

import pyvista_cad

pytest.importorskip('OCP', exc_type=ImportError)
b3d = pytest.importorskip('build123d', exc_type=ImportError)

# The faceted STEP writer (polydata_to_topods) sews the source triangulation
# into a shell; the chordal deviation between that shell and the source
# analytic surface is bounded by the linear_deflection used to tessellate
# (default 0.1 mm). The end-to-end file round-trip must stay within it.
_LINEAR_DEFLECTION = 0.1


def _symmetric_hausdorff(a: pv.PolyData, b: pv.PolyData) -> float:
    fwd = cKDTree(b.points).query(a.points)[0].max()
    rev = cKDTree(a.points).query(b.points)[0].max()
    return float(max(fwd, rev))


def test_write_step_round_trip(tmp_path: Path) -> None:
    src = pyvista_cad.from_build123d(b3d.Box(2, 4, 6))
    path = tmp_path / 'box.step'
    pyvista_cad.write_step(src, path)
    assert path.exists()
    assert path.stat().st_size > 0

    out = pv.read(path)
    assert isinstance(out, (pv.PolyData, pv.MultiBlock))
    # Each faceted face becomes its own block via STEP As-Is transfer.
    if isinstance(out, pv.MultiBlock):
        combined = out.combine().extract_surface(algorithm='dataset_surface')
        assert combined.n_cells > 0
    else:
        assert out.n_cells > 0


def test_accessor_to_step(tmp_path: Path) -> None:
    src = pyvista_cad.from_build123d(b3d.Box(1, 1, 1))
    path = tmp_path / 'cube.step'
    src.cad.to_step(path)
    assert path.exists()


def test_write_step_round_trip_hausdorff_within_deflection(tmp_path: Path) -> None:
    """A non-trivial curved part survives read -> write_step -> read with a
    symmetric (point-to-surface, both directions) Hausdorff distance no
    larger than the tessellation linear deflection, and combined bounds
    within the same tolerance."""
    src = pyvista_cad.from_build123d(b3d.Cylinder(radius=5.0, height=12.0))
    src_surf = src.extract_surface(algorithm='dataset_surface').triangulate()

    out = tmp_path / 'cyl.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out)
    rt_combined = rt.combine() if isinstance(rt, pv.MultiBlock) else rt
    rt_surf = rt_combined.extract_surface(algorithm='dataset_surface').triangulate()

    np.testing.assert_allclose(
        np.asarray(rt_surf.bounds),
        np.asarray(src_surf.bounds),
        atol=_LINEAR_DEFLECTION,
    )

    hausdorff = _symmetric_hausdorff(src_surf, rt_surf)
    assert hausdorff <= _LINEAR_DEFLECTION, (
        f'symmetric Hausdorff {hausdorff:.6g} exceeds tessellation deflection {_LINEAR_DEFLECTION}'
    )


def _symmetric_point_to_surface_hausdorff(a: pv.PolyData, b: pv.PolyData) -> float:
    """Symmetric point-to-surface Hausdorff (both directions)."""

    def directed(src: pv.PolyData, dst: pv.PolyData) -> float:
        _, closest = dst.find_closest_cell(src.points, return_closest_point=True)
        return float(np.linalg.norm(np.asarray(src.points) - closest, axis=1).max())

    return max(directed(a, b), directed(b, a))


def test_write_step_round_trip_tight_deflection_point_to_surface(tmp_path: Path) -> None:
    """At a tight tessellation deflection the read -> write_step -> read
    round trip stays within that deflection by symmetric point-to-surface
    Hausdorff (both directions), pinning the documented chordal bound
    where precision actually matters."""
    tight = 1.0e-3
    cyl = tmp_path / 'cyl_src.step'
    b3d.export_step(b3d.Cylinder(radius=5.0, height=12.0), str(cyl))
    src = pyvista_cad.read_step(cyl, linear_deflection=tight, merge=True)
    src_surf = src.extract_surface(algorithm='dataset_surface').triangulate()
    assert src_surf.n_points > 200  # genuinely faceted curved surface

    out = tmp_path / 'tight.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out, linear_deflection=tight, merge=True)
    rt_surf = rt.extract_surface(algorithm='dataset_surface').triangulate()

    distance = _symmetric_point_to_surface_hausdorff(src_surf, rt_surf)
    assert distance <= tight, (
        f'symmetric point-to-surface Hausdorff {distance:.6g} exceeds tight deflection {tight}'
    )


def test_write_step_closed_input_yields_closed_shell(tmp_path: Path) -> None:
    """A closed (watertight) input part re-reads as a closed surface: zero
    boundary edges after merging coincident tessellation vertices."""
    src = pyvista_cad.from_build123d(b3d.Cylinder(radius=4.0, height=10.0))
    src_clean = src.extract_surface(algorithm='dataset_surface').triangulate().clean()
    src_edges = src_clean.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    assert src_edges.n_cells == 0, 'fixture must be a closed input'

    out = tmp_path / 'closed.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out)
    rt_combined = rt.combine() if isinstance(rt, pv.MultiBlock) else rt
    rt_clean = rt_combined.extract_surface(algorithm='dataset_surface').triangulate().clean()
    rt_edges = rt_clean.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    assert rt_edges.n_cells == 0, (
        f'closed input did not round-trip to a closed shell ({rt_edges.n_cells} boundary edges)'
    )


def _named_products(path: Path) -> list[str]:
    """Re-read a STEP file via OCP/build123d and return component product
    names (the OCAF ``TDataStd_Name`` strings)."""
    import build123d as _b3d  # noqa: PLC0415

    imported = _b3d.import_step(str(path))
    names: list[str] = []

    def _walk(node: object) -> None:
        lbl = getattr(node, 'label', '') or ''
        if lbl:
            names.append(lbl)
        children = getattr(node, 'children', None)
        if children:
            for c in children:
                _walk(c)

    _walk(imported)
    return names


def test_write_step_single_polydata_compound(tmp_path: Path) -> None:
    """A bare PolyData input takes the flat-compound path and re-reads to a
    surface with the same bounds (within tessellation tolerance)."""
    src = pyvista_cad.from_build123d(b3d.Box(3.0, 5.0, 7.0))
    out = tmp_path / 'solo.step'
    pyvista_cad.write_step(src, out)
    rt = pyvista_cad.read_step(out)
    rt_c = rt.combine() if isinstance(rt, pv.MultiBlock) else rt
    np.testing.assert_allclose(np.asarray(rt_c.bounds), np.asarray(src.bounds), atol=0.05)


def test_write_step_assembly_named_products_and_nesting(tmp_path: Path) -> None:
    """``write_assembly=True`` emits one OCAF product per block and writes a
    nested MultiBlock as a nested sub-assembly; re-reading via build123d
    recovers both the leaf names and the nested-block structure."""
    inner = pv.MultiBlock()
    inner['Gear'] = pyvista_cad.from_build123d(b3d.Box(2.0, 2.0, 2.0))
    asm = pv.MultiBlock()
    asm['Base'] = pyvista_cad.from_build123d(b3d.Box(4.0, 4.0, 4.0))
    asm['Sub'] = inner

    out = tmp_path / 'asm.step'
    pyvista_cad.write_step(asm, out, write_assembly=True)

    names = _named_products(out)
    assert 'Base' in names
    assert 'Gear' in names

    rt = pyvista_cad.read_step(out)
    assert isinstance(rt, pv.MultiBlock)
    assert set(rt.keys()) == {'Base', 'Sub'}
    sub = rt['Sub']
    assert isinstance(sub, pv.MultiBlock)
    assert list(sub.keys()) == ['Gear']
    assert str(rt['Base'].field_data['cad.label'][0]) == 'Base'


def test_write_step_assembly_uses_root_label(tmp_path: Path) -> None:
    """A MultiBlock-level ``cad.label`` names the root assembly product."""
    asm = pv.MultiBlock()
    asm['Part'] = pyvista_cad.from_build123d(b3d.Box(2.0, 2.0, 2.0))
    asm.field_data['cad.label'] = np.array(['MyAssembly'])
    out = tmp_path / 'rootname.step'
    pyvista_cad.write_step(asm, out, write_assembly=True)
    assert 'MyAssembly' in _named_products(out)


def test_write_step_flat_fast_path(tmp_path: Path) -> None:
    """``write_assembly=False`` flattens a MultiBlock into one compound; the
    re-read geometry still covers every input block's bounds."""
    asm = pv.MultiBlock()
    asm['A'] = pyvista_cad.from_build123d(b3d.Box(2.0, 2.0, 2.0))
    asm['B'] = pyvista_cad.from_build123d(b3d.Box(2.0, 2.0, 2.0).translate((10.0, 0.0, 0.0)))
    src_bounds = asm.combine().bounds

    out = tmp_path / 'flat.step'
    pyvista_cad.write_step(asm, out, write_assembly=False)
    rt = pyvista_cad.read_step(out)
    rt_c = rt.combine() if isinstance(rt, pv.MultiBlock) else rt
    np.testing.assert_allclose(np.asarray(rt_c.bounds), np.asarray(src_bounds), atol=0.05)


def test_write_step_per_block_transform_placement(tmp_path: Path) -> None:
    """A per-block ``cad.transform`` is emitted as the OCAF component
    placement and survives the round trip exactly once.

    build123d's STEP import bakes the resolved component placement into the
    located ``wrapped`` shape, so the reader must NOT also re-apply
    ``_location_matrix`` to the tessellated geometry. The authored
    translation must round-trip at factor 1.0 (within tessellation
    tolerance), never doubled.
    """
    box = pyvista_cad.from_build123d(b3d.Box(4.0, 4.0, 4.0))
    base = np.asarray(box.bounds, dtype=float)
    tx, ty, tz = 20.0, 5.0, -3.0
    transform = np.eye(4)
    transform[0, 3], transform[1, 3], transform[2, 3] = tx, ty, tz
    box.field_data['cad.transform'] = transform.ravel()

    asm = pv.MultiBlock()
    asm['Shifted'] = box
    out = tmp_path / 'placed.step'
    pyvista_cad.write_step(asm, out, write_assembly=True)
    rt = pyvista_cad.read_step(out)
    shifted = np.asarray(rt['Shifted'].bounds, dtype=float)

    np.testing.assert_allclose(
        shifted[0::2] - base[0::2],
        np.array([tx, ty, tz]),
        atol=1e-3,
    )
    np.testing.assert_allclose(
        shifted[1::2] - base[1::2],
        np.array([tx, ty, tz]),
        atol=1e-3,
    )


def test_write_step_block_transform_bad_size_is_ignored(tmp_path: Path) -> None:
    """A malformed (non-16-element) ``cad.transform`` is treated as identity
    rather than raising; the block round-trips unshifted."""
    box = pyvista_cad.from_build123d(b3d.Box(4.0, 4.0, 4.0))
    base = np.asarray(box.bounds, dtype=float)
    box.field_data['cad.transform'] = np.arange(9, dtype=float)  # wrong size
    asm = pv.MultiBlock()
    asm['Plain'] = box
    out = tmp_path / 'badxf.step'
    pyvista_cad.write_step(asm, out, write_assembly=True)
    rt = pyvista_cad.read_step(out)
    np.testing.assert_allclose(np.asarray(rt['Plain'].bounds, dtype=float), base, atol=0.05)


def test_write_step_unsupported_type_raises() -> None:
    """Passing a non-PolyData/MultiBlock dataset raises ``CadWriteError``
    naming the offending type."""
    from pyvista_cad import CadWriteError  # noqa: PLC0415

    with pytest.raises(CadWriteError, match='ImageData'):
        pyvista_cad.write_step(pv.ImageData(dimensions=(2, 2, 2)), 'x.step')


def test_write_step_assembly_failed_block_raises_named(tmp_path: Path) -> None:
    """A block whose geometry cannot become a TopoDS is named in a raised
    ``CadWriteError`` -- never silently dropped (assembly path)."""
    from pyvista_cad import CadWriteError  # noqa: PLC0415

    asm = pv.MultiBlock()
    asm['Good'] = pyvista_cad.from_build123d(b3d.Box(1.0, 1.0, 1.0))
    asm['BadCloud'] = pv.PolyData(np.random.default_rng(0).random((12, 3)))
    out = tmp_path / 'fail.step'
    with pytest.raises(CadWriteError, match='BadCloud'):
        pyvista_cad.write_step(asm, out, write_assembly=True)


def test_write_step_compound_failed_block_raises_named(tmp_path: Path) -> None:
    """Same no-silent-drop contract on the flat-compound path."""
    from pyvista_cad import CadWriteError  # noqa: PLC0415

    asm = pv.MultiBlock()
    asm['Solid'] = pyvista_cad.from_build123d(b3d.Box(1.0, 1.0, 1.0))
    asm['Cloud'] = pv.PolyData(np.random.default_rng(1).random((8, 3)))
    out = tmp_path / 'failc.step'
    with pytest.raises(CadWriteError, match='Cloud'):
        pyvista_cad.write_step(asm, out, write_assembly=False)


def test_write_step_empty_multiblock_assembly_path_raises(
    tmp_path: Path,
) -> None:
    """The OCAF assembly path rejects an empty input.

    ``_write_assembly`` must mirror ``_write_compound``'s guard: an empty
    MultiBlock (and a tree of only empty/None nested MultiBlocks) has no
    writable leaf product, so the writer raises ``CadWriteError`` instead
    of silently emitting a product-less STEP.
    """
    from pyvista_cad import CadWriteError  # noqa: PLC0415

    empty = pv.MultiBlock()
    out = tmp_path / 'empty.step'
    with pytest.raises(CadWriteError, match='no geometry'):
        pyvista_cad.write_step(empty, out, write_assembly=True)
    assert not out.exists()

    nested_empty = pv.MultiBlock()
    nested_empty['sub'] = pv.MultiBlock()
    out2 = tmp_path / 'empty_nested.step'
    with pytest.raises(CadWriteError, match='no geometry'):
        pyvista_cad.write_step(nested_empty, out2, write_assembly=True)
    assert not out2.exists()


def test_write_step_empty_compound_raises(tmp_path: Path) -> None:
    """The flat-compound path also rejects an empty MultiBlock."""
    from pyvista_cad import CadWriteError  # noqa: PLC0415

    empty = pv.MultiBlock()
    out = tmp_path / 'emptyc.step'
    with pytest.raises(CadWriteError, match='no geometry'):
        pyvista_cad.write_step(empty, out, write_assembly=False)


def test_write_step_non_triangle_block_is_surfaced(tmp_path: Path) -> None:
    """A non-PolyData block (ImageData) is coerced through
    ``extract_surface`` and written; it re-reads with matching bounds."""
    grid = pv.ImageData(dimensions=(5, 5, 5), spacing=(1.0, 1.0, 1.0))
    asm = pv.MultiBlock()
    asm['Grid'] = grid
    out = tmp_path / 'grid.step'
    pyvista_cad.write_step(asm, out, write_assembly=True)
    rt = pyvista_cad.read_step(out)
    rt_c = rt.combine() if isinstance(rt, pv.MultiBlock) else rt
    np.testing.assert_allclose(np.asarray(rt_c.bounds)[:4], np.asarray(grid.bounds)[:4], atol=0.05)
