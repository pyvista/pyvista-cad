"""Regenerate the README per-snippet screenshots (``doc/_static/readme_*.png``).

Every runnable Python snippet in ``README.md`` gets a screenshot embedded
beneath it so the images resolve on PyPI (raw GitHub URLs). Each example
uses a *different* part for visual variety:

- Quick start: bundled offline fixtures (L-bracket STEP + layered DXF).
- CAD-friendly plotting: the NIST AM Bench LPBF specimen (gallery download).
- Real-world workflow: the 3-part NIST build assembly (gallery download).

All 3D parts are drawn through the CAD component (``pl.cad.add`` /
``mb.cad.plot``) so the figures show smoothly shaded faces with the
model's topological B-rep edges -- not generic triangle-mesh edges.

Run via ``just render-readme`` (downloads are cached after first fetch).
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh
import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import bracket_step_path, downloads, drawing_dxf_path

STATIC = Path(__file__).resolve().parent.parent / 'doc' / '_static'

FACE_COLOR = '#4a7fb5'
HIGHLIGHT = '#e07a3f'
MUTED = '#b8c4d0'
GHOST = '#cfd6dd'
LAYER_CMAP = ['#d1495b', '#00798c', '#edae49', '#66a182', '#2e4057', '#8d96a3']
WINDOW = (1400, 760)
# CAD topological-edge line width (bumped slightly for readability).
EDGE_LW = 2.5


def _plotter(shape: tuple[int, int] = (1, 1), width: int = WINDOW[0]) -> pv.Plotter:
    pl = pv.Plotter(off_screen=True, shape=shape, window_size=(width, WINDOW[1]), border=False)
    pl.background_color = 'white'
    return pl


def _caption(pl: pv.Plotter, text: str) -> None:
    pl.add_text(text, font_size=11, color='#555555', position='lower_edge')


def _tilted_xy(pl: pv.Plotter, elevation_deg: float = 28.0) -> None:
    """Camera tilted down from the XY view, kept perpendicular to X.

    The view direction lies in the Y-Z plane (no X component), so it
    reads like an isometric but square-on to the X axis: look along -Y,
    raised ``elevation_deg`` above the horizon so the top faces show.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = pl.renderer.bounds
    focal = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin) or 1.0
    e = math.radians(elevation_deg)
    direction = (0.0, -math.cos(e), math.sin(e))
    dist = 2.6 * radius
    position = tuple(f + d * dist for f, d in zip(focal, direction, strict=True))
    pl.camera_position = [position, focal, (0.0, 0.0, 1.0)]


def quick_start() -> None:
    """README "Quick start": read a STEP part; split a DXF drawing by layer.

    Bundled offline fixtures: the 3D STEP solid is drawn CAD-style; the 2D
    layered drawing is split into per-layer blocks and colour-keyed.
    """
    mesh = pv.read(bracket_step_path())

    floorplan = pv.read(drawing_dxf_path())
    layers = floorplan.cad.split_by_layer()

    pl = _plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.cad.add(mesh, color=FACE_COLOR, edge_color='black', line_width=3.5)
    _caption(pl, "pv.read('part.step')")
    pl.view_isometric()
    pl.camera.zoom(1.1)

    pl.subplot(0, 1)
    for i, name in enumerate(layers.keys()):
        block = layers[name]
        if block is not None and block.n_points:
            pl.add_mesh(block, color=LAYER_CMAP[i % len(LAYER_CMAP)], line_width=3, label=name)
    _caption(pl, "pv.read('floor.dxf').cad.split_by_layer()")
    pl.add_legend(bcolor='white', size=(0.22, 0.22), loc='upper right')
    pl.view_xy()

    out = STATIC / 'readme_quickstart.png'
    pl.screenshot(out)
    print(f'Wrote {out}')


def cad_plotting() -> None:
    """README "CAD-friendly plotting": ``cad.plot`` and a scalar-coloured scene.

    The NIST AM Bench specimen. Both panels keep the topological B-rep
    edges; the right panel colours the same cached block by a height scalar.
    """
    mb = pyvista_cad.read_step(downloads.step_part_path())

    part = mb[0]
    part['height'] = part.points[:, 2]

    pl = _plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.cad.add(mb, color=FACE_COLOR, edge_color='black', line_width=EDGE_LW)
    _caption(pl, 'mb.cad.plot()')
    pl.view_isometric()

    pl.subplot(0, 1)
    pl.cad.add(part, scalars='height', cmap='viridis', edge_color='black', line_width=EDGE_LW)
    _caption(pl, "pl.cad.add(part, scalars='height')")
    pl.view_isometric()

    pl.link_views()
    out = STATIC / 'readme_cad_plotting.png'
    pl.screenshot(out)
    print(f'Wrote {out}')


def real_world_workflow() -> None:
    """README "Real-world workflow": locate a part, FEA-tet it, clip it.

    The 3-part NIST build assembly: the assembly as read, the located
    PartCAD component highlighted, then that part driven through gmsh to
    a tetrahedral FEA mesh and clipped to expose the tet interior.
    """
    assembly = pv.read(downloads.step_assembly_path())
    print(assembly.cad.assembly_tree())
    matches = assembly.cad.find('*PartCAD')
    _path, _part = matches[0]

    # Mesh the located part as tets with gmsh (the canonical CAD->FEA path).
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.importShapes(downloads.step_part_path())
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMax', 2.0)
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.5)
        gmsh.model.mesh.generate(3)
        grid = pyvista_cad.from_gmsh()
    finally:
        gmsh.finalize()

    grid = grid.extract_cells(grid.celltypes == 10)  # keep VTK_TETRA
    clip = grid.clip(normal='x', crinkle=True)
    clip['Z'] = clip.points[:, 2]
    clip.save('part_tets.vtu')
    Path('part_tets.vtu').unlink(missing_ok=True)

    found_paths = {p for p, _ in matches}

    pl = _plotter(shape=(1, 3), width=1860)

    pl.subplot(0, 0)
    pl.cad.add(assembly, color=FACE_COLOR, edge_color='black', line_width=EDGE_LW)
    _caption(pl, 'pv.read(step_assembly)')
    _tilted_xy(pl)
    pl.camera.zoom(0.9)

    pl.subplot(0, 1)
    # Recolor: the located PartCAD blocks pop, the rest of the assembly mutes.
    for path, block in assembly.cad.walk():
        if block is None or not getattr(block, 'n_points', 0):
            continue
        color = HIGHLIGHT if path in found_paths else MUTED
        pl.cad.add(block, color=color, edge_color='black', line_width=EDGE_LW)
    _caption(pl, "cad.find('*PartCAD')")
    _tilted_xy(pl)
    pl.camera.zoom(0.9)

    pl.subplot(0, 2)
    pl.add_mesh(grid, color=GHOST, opacity=0.15)
    pl.add_mesh(clip, scalars='Z', cmap='inferno', show_edges=True, edge_color='black')
    _caption(pl, 'gmsh tet mesh + clip')
    pl.view_isometric()

    out = STATIC / 'readme_workflow.png'
    pl.screenshot(out)
    print(f'Wrote {out}')


if __name__ == '__main__':
    quick_start()
    cad_plotting()
    real_world_workflow()
