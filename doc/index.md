# pyvista-cad

CAD format reading, writing, and CAD-style plotting for [PyVista](https://github.com/pyvista/pyvista).

`pyvista-cad` adds STEP, IGES, BREP, DXF, 3MF, IFC, FreeCAD `.fcstd`,
OpenSCAD `.scad`, and glTF support to PyVista. It registers a `.cad`
accessor plus reader entries on `pv.read(...)`. No monkey-patching. No
forks. No direct VTK calls.

The `.cad` accessor is dual: on a `pv.DataSet` it resolves to
{class}`~pyvista_cad.CadDataSetAccessor`, and on a `pv.MultiBlock` to
{class}`~pyvista_cad.CadMultiBlockAccessor`. PyVista picks the right one
by object type, so `mesh.cad` and `assembly.cad` both work and expose
the surface appropriate to that object (the MultiBlock accessor adds
assembly-tree traversal). See the {ref}`api_reference`.

```{eval-rst}
.. pyvista-plot::

   import pyvista_cad
   from pyvista_cad import examples

   mb = pyvista_cad.read_step(examples.bracket_step_path())
   mb.cad.plot()   # CAD-friendly: shaded faces + topological edges
```

```{toctree}
:hidden:
:caption: Gallery

examples/index
```

```{toctree}
:hidden:
:caption: User guide

user_guide/cad_visualization
user_guide/interop_with_build123d
```

```{toctree}
:hidden:
:caption: Reference

api/index
```
