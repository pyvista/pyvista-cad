"""3MF read -> write -> re-read behavioral invariants.

Color and build-item transform preservation are tested by synthesising the
source file directly through the lib3mf API so the reader is exercised
against a known-good ground truth (the writer presently drops both).
"""

from pathlib import Path

import numpy as np
import pytest

import pyvista_cad

lib3mf = pytest.importorskip('lib3mf')


def _build_3mf_with_color_and_transform(
    path: Path,
    *,
    rgba: tuple[int, int, int, int],
    translation: tuple[float, float, float],
    linear: np.ndarray | None = None,
) -> None:
    """Write a 3MF file with one object that has an object-level color and a
    non-identity build-item transform. Pure lib3mf - no pyvista_cad writer."""
    w = lib3mf.get_wrapper()
    model = w.CreateModel()
    model.SetUnit(1)  # mm

    obj = model.AddMeshObject()
    obj.SetName('part_with_color')

    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    tris = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    positions = []
    for x, y, z in pts:
        p = lib3mf.Position()
        p.Coordinates = (float(x), float(y), float(z))
        positions.append(p)
    triangles = []
    for a, b, c in tris:
        t = lib3mf.Triangle()
        t.Indices = (a, b, c)
        triangles.append(t)
    obj.SetGeometry(positions, triangles)

    # Object-level color via a ColorGroup property.
    cg = model.AddColorGroup()
    rgb_color = lib3mf.Color()
    rgb_color.Red = rgba[0]
    rgb_color.Green = rgba[1]
    rgb_color.Blue = rgba[2]
    rgb_color.Alpha = rgba[3]
    prop_id = cg.AddColor(rgb_color)
    obj.SetObjectLevelProperty(cg.GetUniqueResourceID(), prop_id)

    if linear is None:
        rows = np.eye(3, dtype=float)
    else:
        rows = np.asarray(linear, dtype=float)
        assert rows.shape == (3, 3)

    transform = lib3mf.Transform()
    # lib3mf's Transform is row-major 4x3 where Fields[i] is the i-th row of an
    # affine 4x3; the first three rows are the linear basis (columns of the 3x3
    # linear part, stored as rows here) and the fourth row is the translation.
    transform.Fields[0][:] = list(rows[:, 0])
    transform.Fields[1][:] = list(rows[:, 1])
    transform.Fields[2][:] = list(rows[:, 2])
    transform.Fields[3][:] = list(translation)
    model.AddBuildItem(obj, transform)

    model.QueryWriter('3mf').WriteToFile(str(path))


@pytest.mark.parametrize(
    'rgba',
    [
        (0, 0, 0, 0),
        (127, 127, 127, 128),
        (255, 255, 255, 255),
        (255, 32, 0, 255),
    ],
)
def test_three_mf_round_trip_preserves_object_color(
    tmp_path: Path, rgba: tuple[int, int, int, int]
) -> None:
    """Object-level RGBA via lib3mf is recovered as ``cad.color`` across the
    full 0/mid/saturated/alpha matrix."""
    path = tmp_path / f'colored_{"_".join(str(c) for c in rgba)}.3mf'
    _build_3mf_with_color_and_transform(path, rgba=rgba, translation=(0.0, 0.0, 0.0))
    rt = pyvista_cad.read_three_mf(path)
    block = rt[0]
    assert 'cad.color' in block.field_data
    color = np.asarray(block.field_data['cad.color'])
    np.testing.assert_allclose(
        color,
        [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0],
        atol=1.0 / 255.0,
    )


def test_three_mf_round_trip_preserves_build_item_transform(tmp_path: Path) -> None:
    """Build-item translations are recovered on ``block.field_data['cad.transform']``."""
    path = tmp_path / 'translated.3mf'
    _build_3mf_with_color_and_transform(path, rgba=(0, 255, 0, 255), translation=(5.0, -3.0, 2.0))
    rt = pyvista_cad.read_three_mf(path)
    block = rt[0]
    assert 'cad.transform' in block.field_data
    matrix = np.asarray(block.field_data['cad.transform'])
    np.testing.assert_allclose(matrix[:3, 3], [5.0, -3.0, 2.0], atol=1e-9)
    # The translation should also have moved the points.
    expected = [5.0 + 0.25, -3.0 + 0.25, 2.0 + 0.25]
    np.testing.assert_allclose(block.points.mean(axis=0), expected, atol=1e-6)


def test_three_mf_round_trip_preserves_rotation_transform(tmp_path: Path) -> None:
    """A 90 deg rotation about Z survives the build-item transform path."""
    path = tmp_path / 'rotated.3mf'
    theta = np.pi / 2
    c, s = np.cos(theta), np.sin(theta)
    linear = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    _build_3mf_with_color_and_transform(
        path, rgba=(0, 0, 255, 255), translation=(0.0, 0.0, 0.0), linear=linear
    )
    rt = pyvista_cad.read_three_mf(path)
    block = rt[0]
    matrix = np.asarray(block.field_data['cad.transform'])
    np.testing.assert_allclose(matrix[:3, :3], linear, atol=1e-9)
    np.testing.assert_allclose(matrix[:3, 3], [0.0, 0.0, 0.0], atol=1e-9)


def test_three_mf_round_trip_preserves_nonuniform_scale_transform(
    tmp_path: Path,
) -> None:
    """A non-uniform scale survives the build-item transform path."""
    path = tmp_path / 'scaled.3mf'
    linear = np.diag([2.0, 3.0, 4.0])
    _build_3mf_with_color_and_transform(
        path,
        rgba=(255, 255, 0, 255),
        translation=(1.0, 2.0, 3.0),
        linear=linear,
    )
    rt = pyvista_cad.read_three_mf(path)
    block = rt[0]
    matrix = np.asarray(block.field_data['cad.transform'])
    np.testing.assert_allclose(matrix[:3, :3], linear, atol=1e-9)
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)
