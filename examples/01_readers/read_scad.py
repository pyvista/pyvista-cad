"""
Compile an OpenSCAD source file
===============================

``pyvista_cad.read_scad`` shells out to the ``openscad`` CLI, captures
the STL it emits, and returns it as a :class:`pyvista.PolyData`. The
OpenSCAD binary must be installed and on ``PATH``.

This example compiles a real OpenSCAD library, ``threads.scad`` by
Ryan A. Colyer (CC0, https://github.com/rcolyer/threads-scad), a
widely used parametric ISO metric fastener generator. Its top-level
``Demo()`` renders real bolts, nuts, and washers.
"""

# %%

import pyvista as pv

import pyvista_cad
from pyvista_cad.examples import downloads

# %%
# Compile the real ``.scad`` source. The ``openscad`` binary must be
# installed and on ``PATH``. ``threads.scad`` exposes a top-level
# ``screw_resolution`` (mm); the default ``0.2`` tessellates the
# helices very finely and is slow. ``-D screw_resolution=0.6`` is
# ~3x faster and the thread features still read clearly: ample for
# a gallery render. The fast ``manifold`` backend is applied
# automatically (OpenSCAD >= 2024) with a fallback to CGAL.

mesh = pyvista_cad.read_scad(
    downloads.scad_threads_path(),
    args=('-D', 'screw_resolution=0.6'),
)
mesh

# %%
# Render colored by Z so the fastener features read.

mesh['elevation'] = mesh.points[:, 2]

pl = pv.Plotter()
pl.cad.add(mesh, scalars='elevation', cmap='viridis')
pl.show()
