<p align="center">
  <img src="https://github.com/pyvista/pyvista/raw/main/doc/source/_static/pyvista_logo.svg" alt="PyVista" width="400" />
</p>

# pyvista-cad

[![CI](https://github.com/pyvista/pyvista-cad/actions/workflows/ci.yml/badge.svg)](https://github.com/pyvista/pyvista-cad/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyvista-cad.svg)](https://pypi.org/project/pyvista-cad/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CAD format reading, writing, and CAD-style plotting for [PyVista](https://github.com/pyvista/pyvista).**

<p align="center">
  <img src="https://raw.githubusercontent.com/pyvista/pyvista-cad/main/doc/_static/hero.png" alt="The same flange shown two ways: a generic mesh viewer drawing the raw triangulation on the left, and pyvista-cad drawing smoothly shaded faces with the model's topological edges on the right." width="90%">
</p>

`pyvista-cad` adds CAD-format support to PyVista: STEP, IGES, BREP, DXF, 3MF, IFC, FreeCAD `.fcstd`, OpenSCAD `.scad`, and glTF. It registers a `.cad` accessor on every `pv.DataSet` / `pv.MultiBlock` and wires reader entries into `pv.read(...)`. It also adds CAD-style rendering: smoothly shaded faces with the model's topological B-rep edges instead of triangle-mesh noise, via `.cad.plot()` and a `plotter.cad` component. No monkey-patching, no forks, no direct VTK calls.

## Install matrix

| Extra          | Adds                    | Formats unlocked                    |
| -------------- | ----------------------- | ----------------------------------- |
| (base)         | ezdxf                   | DXF read + write, glTF read + write |
| `[step]`       | build123d, cadquery-ocp | STEP, BREP, FCStd, build123d bridge |
| `[step-light]` | cascadio                | STEP (read-only, faster, no colors) |
| `[3mf]`        | lib3mf                  | 3MF read + write                    |
| `[ifc]`        | ifcopenshell            | IFC read (with property sets)       |
| `[iges]`       | pyiges[full]            | IGES read                           |
| `[openscad]`   | (uses `openscad` CLI)   | SCAD read                           |
| `[full]`       | all of the above        | every supported format              |

**Python support:** 3.10 – 3.13. Python 3.14 is not yet supported: the
`[step]`/`[full]` extras depend on `cadquery-ocp`, which publishes no
cp314 wheels for any platform. 3.14 will be added once those wheels ship.

trimesh interop is provided by pyvista core (`pyvista.from_trimesh`, `pyvista.to_trimesh`) and the `pyvista-trimesh` package's `.trimesh` accessor (install `pyvista-cad[trimesh]`); pyvista-cad does not duplicate it.

FEA meshing via gmsh is intentionally out of scope. `gmsh` is GPLv2+ and would virally license any closed-source product that linked it, so `pyvista-cad`'s dependency set stays fully permissive (MIT / BSD / Apache, plus LGPL-with-exception for the OpenCascade and IFC backends). For CAD-to-tet workflows, drive `gmsh` directly and read the resulting `.msh` file back with `pv.read` (PyVista routes `.msh` through `meshio`); the `scikit-gmsh` package wraps the live-model API if you prefer that.

```bash
pip install pyvista-cad           # DXF and glTF only
pip install pyvista-cad[step]     # add STEP, BREP, FCStd
pip install pyvista-cad[full]     # everything
```

## Quick start

```python
import pyvista as pv
import pyvista_cad  # registers the .cad accessor and reader entries

mesh = pv.read('part.step')              # MultiBlock of parts with cad.color, cad.label
mesh.plot(show_edges=True)

floorplan = pv.read('floor.dxf')         # PolyData with Layer cell data
layers = floorplan.cad.split_by_layer()  # MultiBlock keyed on layer
```

<p align="center">
  <img src="https://raw.githubusercontent.com/pyvista/pyvista-cad/main/doc/_static/readme_quickstart.png" alt="Two independent example fixtures. Left: a STEP part read with pv.read, drawn CAD-style with shaded faces and topological edges. Right: a DXF drawing split into colored per-layer blocks (body, centerlines, dimensions, holes)." width="90%">
</p>

## CAD-friendly plotting

A generic mesh viewer draws the _triangulation_. With `show_edges=True` you see every facet edge, an artifact of the tessellation tolerance. A CAD application instead shows smoothly shaded faces with the model's _topological_ edges (the B-rep feature curves) on top. `pyvista-cad` reproduces that: analytic surface normals so a coarse mesh still shades round, topological edges recovered from the cached B-rep, triangle edges hidden.

```python
import pyvista as pv
import pyvista_cad
from pyvista_cad.examples import downloads

mb = pyvista_cad.read_step(downloads.step_part_path())  # NIST AM Bench specimen

mb.cad.plot()                       # shaded faces + topological edges

# Or compose it into a scene, color faces by a scalar, keep the edges:
part = mb[0]                        # a cached block keeps its B-rep
part['height'] = part.points[:, 2]
pl = pv.Plotter()
pl.cad.add(part, scalars='height', cmap='viridis')
pl.show()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/pyvista/pyvista-cad/main/doc/_static/readme_cad_plotting.png" alt="The NIST AM Bench specimen. Left: cad.plot draws shaded faces with topological B-rep edges. Right: the same cached block colored by a height scalar with the viridis colormap, the topological edges still drawn." width="90%">
</p>

`.cad.plot()` and the `plotter.cad` component accept a `MultiBlock`, `PolyData`, raw `TopoDS`, or a build123d / cadquery object; a plain mesh with no B-rep origin degrades to crease feature edges.

## Real-world workflow

Load a STEP assembly, locate a part, drive it through gmsh to a
tetrahedral FEA mesh, then clip to expose the interior. `pv.read`
handles the resulting `.msh` file natively via `meshio`, so no extra
PyVista dependency is needed.

```python
import gmsh
import pyvista as pv
import pyvista_cad
from pyvista_cad.examples import downloads

assembly = pv.read(downloads.step_assembly_path())  # 3-part NIST build assembly
print(assembly.cad.assembly_tree())                 # nested dict of block names
matches = assembly.cad.find('*PartCAD')             # glob -> list of (path, block)
path, part = matches[0]

gmsh.initialize()
try:
    gmsh.model.occ.importShapes(downloads.step_part_path())
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMax', 2.0)
    gmsh.model.mesh.generate(3)
    gmsh.write('part.msh')
finally:
    gmsh.finalize()

grid = pv.read('part.msh')                          # via meshio
grid = grid.extract_cells(grid.celltypes == 10)     # keep VTK_TETRA
clip = grid.clip(normal='x', crinkle=True)
clip.save('part_tets.vtu')                          # full tet mesh round-trips
```

The Quick start uses bundled offline fixtures (`bracket_step_path()`, a parametric L-bracket committed as STEP; `drawing_dxf_path()`, a layered 2D drawing). The other examples pull real, openly licensed parts from `pyvista_cad.examples.downloads` (cached on first fetch) — the NIST AM Bench LPBF specimen and its 3-part build assembly.

## Common tasks

| Task                                      | How                                                                                       |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| Read a STEP assembly with per-part colors | `pv.read('a.step')` returns a `MultiBlock`; each block has `cad.color` and `cad.label`    |
| Split a DXF by layer                      | `pv.read('a.dxf').cad.split_by_layer()`                                                   |
| Round-trip a 3MF print                    | `pyvista_cad.write_three_mf(mb, 'b.3mf')` (object color + units kept)                     |
| Load an IFC building and filter walls     | `mb = pv.read('b.ifc'); walls = mb.cad.find(ifc_type='IfcWall')`                          |
| Read IFC property sets                    | `json.loads(block.field_data['cad.psets'][0])` returns the source `Pset_*` / `Qto_*` dict |
| Convert build123d to PyVista              | `pyvista_cad.from_build123d(part)` (preserves color, label, transform)                    |
| Mesh a STEP for FEA in gmsh               | Drive `gmsh` directly, `gmsh.write('out.msh')`, then `pv.read('out.msh')` (uses `meshio`) |
| Generate a signed distance field from CAD | see `examples/05_workflows/cad_to_signed_distance.py`                                     |

## Licensing

`pyvista-cad` is MIT. Every runtime dependency is either permissive
(MIT / BSD / Apache) or LGPL with the OpenCascade exception. **No
GPL code is pulled in by any extra**, which makes the package safe to
use in closed-source / proprietary products subject to the standard
LGPL dynamic-link obligations. See [LICENSES.md](LICENSES.md) for the
full dependency-by-dependency breakdown, the LGPL compliance notes, and the rationale for not depending on
`gmsh`.

## Fidelity and limitations

Round-trip fidelity varies by format. Tessellated formats (STEP, IGES, BREP, FCStd) discretize analytic surfaces on read; the originating B-rep is cached so `tessellate()` can refine it. DXF and 3MF round-trip geometry and metadata within documented tolerances. IGES and SCAD are read-only.
