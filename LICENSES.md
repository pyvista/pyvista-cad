# Dependency licenses

`pyvista-cad` is **MIT**. This file inventories the licenses of its
runtime dependencies so downstream users can audit them for
commercial / proprietary use.

## Base install (`pip install pyvista-cad`)

Fully permissive. No copyleft.

| Package           | License            |
| ----------------- | ------------------ |
| pyvista           | MIT                |
| numpy             | BSD-3-Clause       |
| pooch             | BSD-3-Clause       |
| ezdxf             | MIT                |
| vtk (transitive)  | BSD-3-Clause       |

## Optional extras

| Extra          | Package          | License                              | Commercial-use note                              |
| -------------- | ---------------- | ------------------------------------ | ------------------------------------------------ |
| `[step]`       | build123d        | Apache-2.0                           | Safe.                                            |
| `[step]`       | cadquery-ocp     | LGPL-2.1 + OCCT exception            | Safe when dynamically linked (pip wheel).        |
| `[step-light]` | cascadio         | LGPL-2.1 + OCCT exception            | Same as cadquery-ocp.                            |
| `[3mf]`        | lib3mf           | BSD-2-Clause                         | Safe.                                            |
| `[ifc]`        | ifcopenshell     | LGPL-3.0+                            | Safe when dynamically linked; ship license text. |
| `[iges]`       | pyiges           | MIT                                  | Safe.                                            |
| `[cadquery]`   | cadquery         | Apache-2.0                           | Safe.                                            |
| `[trimesh]`    | pyvista-trimesh  | MIT                                  | Safe.                                            |
| (no extra)     | openscad (CLI)   | GPL-2.0                              | Subprocess only. No linking, GPL does not propagate. |
| (no extra)     | FreeCAD (CLI)    | LGPL-2.0+                            | Subprocess only.                                 |

### LGPL obligations (cadquery-ocp / cascadio / ifcopenshell)

LGPL allows commercial / closed-source use **when the LGPL library is
dynamically linked and the end user can replace it with a modified
copy**. Standard pip wheels satisfy this. Obligations:

1. Distribute the LGPL license text alongside your product.
2. Do not modify the LGPL library and statically bundle it without
   releasing your modifications.
3. If you repackage into a single binary (PyInstaller, Nuitka, etc.),
   audit your build for whether the library is still replaceable. If
   not, you must release the object files or use dynamic linking.

## Why no gmsh?

`gmsh` is **GPLv2+**. Linking it into a Python process forces the GPL
on the entire program, which is unsafe for closed-source /
proprietary consumers. `pyvista-cad` therefore depends on no
GPL-licensed code. Users who need CAD-to-FEA tet meshing should drive
`gmsh` directly, write a `.msh` file with `gmsh.write()`, and read it
back with `pv.read()` (PyVista routes `.msh` through `meshio`, which
is MIT). The
[`scikit-gmsh`](https://github.com/pyvista/scikit-gmsh) package
provides a thin live-model wrapper if preferred; it is GPL-3.0 itself
and is not a `pyvista-cad` dependency.

## Verdict

The default `pyvista-cad[full]` install is suitable for closed-source
commercial use, subject to standard LGPL notice and dynamic-link
obligations for the OpenCascade and IFC backends. No GPL code is
pulled in by any `pyvista-cad` extra.
