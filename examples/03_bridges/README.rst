.. _bridges:

Bridges to CAD kernels
======================

Round-trip data between PyVista and the major Python CAD kernels:
build123d, cadquery, raw OCP / TopoDS, gmsh, and trimesh. Each bridge
attaches ``cad.source_format`` to the resulting dataset so the origin is
traceable through downstream pipelines.
