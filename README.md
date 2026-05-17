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
| `[gmsh]`       | gmsh                    | gmsh bridge for FEA meshing         |
| `[openscad]`   | (uses `openscad` CLI)   | SCAD read                           |
| `[full]`       | all of the above        | every supported format              |

trimesh interop is provided by pyvista core (`pyvista.from_trimesh`, `pyvista.to_trimesh`) and the `pyvista-trimesh` package's `.trimesh` accessor (install `pyvista-cad[trimesh]`); pyvista-cad does not duplicate it.

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

## CAD-friendly plotting

A generic mesh viewer draws the _triangulation_. With `show_edges=True` you see every facet edge, an artifact of the tessellation tolerance. A CAD application instead shows smoothly shaded faces with the model's _topological_ edges (the B-rep feature curves) on top. `pyvista-cad` reproduces that: analytic surface normals so a coarse mesh still shades round, topological edges recovered from the cached B-rep, triangle edges hidden.

```python
import pyvista as pv
import pyvista_cad
from pyvista_cad.examples import bracket_step_path

mb = pyvista_cad.read_step(bracket_step_path())

mb.cad.plot()                       # the hero render above

# Or compose it into a scene, color faces by a scalar, keep the edges:
surf = mb.combine().extract_surface()
surf['height'] = surf.points[:, 2]
pl = pv.Plotter()
pl.cad.add(surf, scalars='height', cmap='viridis')
pl.show()
```

`.cad.plot()` and the `plotter.cad` component accept a `MultiBlock`, `PolyData`, raw `TopoDS`, or a build123d / cadquery object; a plain mesh with no B-rep origin degrades to crease feature edges.

## Real-world workflow

Load a STEP assembly, split it into parts, run a filter on one, write the result back out.

```python
import pyvista as pv
import pyvista_cad
from pyvista_cad.examples import bracket_step_path

assembly = pv.read(bracket_step_path())
print(assembly.cad.assembly_tree())                 # nested dict of block names
matches = assembly.cad.find(name='bracket')         # list of (path, block)
path, part = matches[0]
clipped = part.clip('z')
clipped.save('bracket_clipped.vtu')                 # cad.* field data round-trips
```

The bundled `bracket_step_path()` fixture is a parametric L-bracket (50×30×10 mm, two M6 through-holes, 3 mm fillet) generated with build123d and committed as STEP, so no build123d install is needed to read it.

## Common tasks

| Task                                      | How                                                                                       |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| Read a STEP assembly with per-part colors | `pv.read('a.step')` returns a `MultiBlock`; each block has `cad.color` and `cad.label`    |
| Split a DXF by layer                      | `pv.read('a.dxf').cad.split_by_layer()`                                                   |
| Round-trip a 3MF print                    | `pyvista_cad.write_three_mf(mb, 'b.3mf')` (object color + units kept)                     |
| Load an IFC building and filter walls     | `mb = pv.read('b.ifc'); walls = mb.cad.find(ifc_type='IfcWall')`                          |
| Read IFC property sets                    | `json.loads(block.field_data['cad.psets'][0])` returns the source `Pset_*` / `Qto_*` dict |
| Convert build123d to PyVista              | `pyvista_cad.from_build123d(part)` (preserves color, label, transform)                    |
| Mesh a STEP for FEA in gmsh               | `pyvista_cad.to_gmsh(...)` then gmsh; see `examples/05_workflows/cad_to_fea_tet_mesh.py`  |
| Generate a signed distance field from CAD | see `examples/05_workflows/cad_to_signed_distance.py`                                     |

## Fidelity and limitations

Round-trip fidelity varies by format. Tessellated formats (STEP, IGES, BREP, FCStd) discretize analytic surfaces on read; the originating B-rep is cached so `tessellate()` can refine it. DXF and 3MF round-trip geometry and metadata within documented tolerances. IGES and SCAD are read-only.
