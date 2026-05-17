"""Image-regression tests.

Uses ``pytest-pyvista``'s ``verify_image_cache`` fixture. Reference
images live under ``tests/image_cache/`` and are regenerated locally
via ``pytest --reset_image_cache``.

Each test follows the pyvista-expert checklist:

* asymmetric geometry (so rotations/flips are detectable),
* explicit, off-axis camera (so default-camera regressions surface),
* distinctive colormap (so scalar-bar / colormap drift is visible),
* edges-on (so any topology change shows up at the silhouette),
* a behavioural assertion BEFORE ``show()`` (so pure-rendering bugs
  cannot mask a data-pipeline failure).
"""

from pathlib import Path

import pytest
import pyvista as pv

import pyvista_cad

_IMAGE_CACHE_DIR = Path(__file__).parent / 'image_cache'


def test_image_cache_has_baselines() -> None:
    """At least one baseline PNG must exist under ``tests/image_cache/``.

    Without this guard, a freshly-cleared cache would let every
    ``verify_image_cache`` test pass-by-default (pytest-pyvista treats a
    missing baseline as a regeneration target on first run), silently
    masking regressions.
    """
    assert _IMAGE_CACHE_DIR.is_dir(), f'image cache dir missing: {_IMAGE_CACHE_DIR}'
    baselines = sorted(_IMAGE_CACHE_DIR.glob('*.png'))
    assert baselines, (
        f'no baseline PNGs found in {_IMAGE_CACHE_DIR}; image-regression tests '
        'would pass vacuously. Regenerate with `pytest --reset_image_cache`.'
    )


# Off-axis hero camera shared across tests.
_HERO_CAMERA = [(5.0, -8.0, 12.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]


def _asymmetric_part() -> pv.PolyData:
    """Slab with an off-centre through-hole — rotation-sensitive."""
    b3d = pytest.importorskip('build123d')
    shape = b3d.Box(2, 4, 6) - b3d.Cylinder(0.5, 7)
    mesh = pyvista_cad.from_build123d(shape)
    assert isinstance(mesh, pv.PolyData)
    return mesh


def _add_distinctive(plotter: pv.Plotter, mesh: pv.DataSet, *, cmap: str) -> None:
    """Attach ``mesh`` with the standard distinctive-style options."""
    # Synthesise a scalar field so the colormap is exercised, not the default
    # solid-colour path.
    if mesh.n_points > 0:
        mesh = mesh.copy()
        mesh.point_data['height'] = mesh.points[:, 2]
        plotter.add_mesh(
            mesh,
            scalars='height',
            cmap=cmap,
            show_edges=True,
            edge_color='black',
            line_width=0.7,
        )
    else:
        plotter.add_mesh(
            mesh,
            cmap=cmap,
            show_edges=True,
            edge_color='black',
            line_width=0.7,
        )


def test_read_dxf_image(verify_image_cache: object) -> None:
    """DXF plate rendered with a distinctive colormap and off-axis camera."""
    poly = pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)
    # Behavioural assertion BEFORE the render so a data-pipeline regression
    # cannot hide behind a passing image diff.
    assert poly.n_cells > 0
    assert poly.n_points > 0

    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    _add_distinctive(plotter, poly, cmap='viridis')
    plotter.camera_position = _HERO_CAMERA
    plotter.show()


def test_read_step_image(verify_image_cache: object) -> None:
    """STEP cube rendered with off-axis camera and an inferno colormap."""
    pytest.importorskip('build123d')
    mb = pyvista_cad.read_step(pyvista_cad.examples.step_cube)
    combined = mb.combine()
    assert combined.n_cells > 0
    assert combined.n_points > 0

    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    _add_distinctive(plotter, combined, cmap='inferno')
    plotter.camera_position = _HERO_CAMERA
    plotter.show()


def test_asymmetric_part_image(verify_image_cache: object) -> None:
    """Slab-with-hole — distinctly asymmetric so flips/rotations are visible."""
    mesh = _asymmetric_part()
    assert mesh.n_cells > 0
    assert mesh.bounds.x_min == pytest.approx(-1.0)
    assert mesh.bounds.x_max == pytest.approx(1.0)
    assert mesh.bounds.z_min == pytest.approx(-3.0)
    assert mesh.bounds.z_max == pytest.approx(3.0)

    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    _add_distinctive(plotter, mesh, cmap='viridis')
    plotter.camera_position = _HERO_CAMERA
    plotter.show()


def test_cad_view_faces_and_topological_edges(verify_image_cache: object) -> None:
    """CAD-style render: smooth faces + B-rep edges, no triangle-mesh edges.

    A box with a drilled hole — asymmetric, mixes planar faces (straight
    edges) with a cylindrical wall (circular edges) so a normals or
    edge-extraction regression is visible at the silhouette.
    """
    pytest.importorskip('OCP')
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    from pyvista_cad import add_cad, topods_to_multiblock

    box = BRepPrimAPI_MakeBox(20.0, 12.0, 8.0).Shape()
    axis = gp_Ax2(gp_Pnt(7.0, 6.0, -1.0), gp_Dir(0.0, 0.0, 1.0))
    drill = BRepPrimAPI_MakeCylinder(axis, 3.0, 10.0).Shape()
    shape = BRepAlgoAPI_Cut(box, drill).Shape()

    mb = topods_to_multiblock(shape, linear_deflection=0.3)
    # Behavioural assertions BEFORE the render.
    assert len(mb['faces']) >= 7
    assert mb['edges'].n_cells > 0
    assert 1 in mb['edges'].cell_data['cad.edge_kind']  # circular rim present

    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    add_cad(plotter, mb, line_width=3.0)
    plotter.show()
