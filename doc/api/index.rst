.. _api_reference:

API reference
=============

The public API of ``pyvista-cad`` is everything exported from the
top-level :mod:`pyvista_cad` namespace, plus the two ``.cad`` accessor
classes. Importing :mod:`pyvista_cad` registers the ``.cad`` accessor
on :class:`pyvista.DataSet` and :class:`pyvista.MultiBlock` and wires
the CAD readers into :func:`pyvista.read`.

.. currentmodule:: pyvista_cad

Readers and writers
-------------------

Format readers return a :class:`pyvista.PolyData` or
:class:`pyvista.MultiBlock`; writers take one and a path. IGES and
SCAD are read-only.

.. autosummary::
   :toctree: _autosummary

   read_step
   write_step
   read_iges
   read_brep
   write_brep
   read_dxf
   write_dxf
   read_three_mf
   write_three_mf
   read_ifc
   read_fcstd
   read_scad
   read_gltf
   enrich_gltf

Bridges to other CAD libraries
------------------------------

Convert between PyVista meshes and build123d, CadQuery, and raw OCCT
``TopoDS`` shapes.

trimesh interop is provided by pyvista core
(:func:`pyvista.from_trimesh`, :func:`pyvista.to_trimesh`) and by the
``pyvista-trimesh`` package's ``.trimesh`` accessor (install
``pyvista-cad[trimesh]``); pyvista-cad does not duplicate it.

gmsh interop is intentionally out of scope so the package stays
license-clean: ``gmsh`` is GPLv2+ and would virally license any
closed-source consumer. For CAD-to-tet workflows drive ``gmsh``
directly, write a ``.msh`` file, and read it back with
``pv.read`` (PyVista routes ``.msh`` through ``meshio``).

.. autosummary::
   :toctree: _autosummary

   from_build123d
   to_build123d
   from_cadquery
   to_cadquery
   from_topods
   to_topods

CAD-view conversion helpers
---------------------------

The free functions behind the CAD-friendly render: resolve a mesh or
raw shape into a ``{'faces', 'edges'}`` :class:`pyvista.MultiBlock`,
or plot it directly.

.. autosummary::
   :toctree: _autosummary

   add_cad
   plot_cad
   as_cad_multiblock
   flatten_to_cad_polydata
   get_cad_edges
   topods_to_edges
   topods_to_multiblock

Accessors
---------

``pyvista-cad`` registers a dual ``.cad`` accessor: PyVista resolves
``.cad`` to :class:`CadDataSetAccessor` on a :class:`pyvista.DataSet`
and to :class:`CadMultiBlockAccessor` on a :class:`pyvista.MultiBlock`,
by object type. They share the metadata, split, conversion, and
plotting surface; the MultiBlock accessor adds assembly-tree
traversal (:meth:`~CadMultiBlockAccessor.walk`,
:meth:`~CadMultiBlockAccessor.find`,
:meth:`~CadMultiBlockAccessor.assembly_tree`,
:meth:`~CadMultiBlockAccessor.flatten`).

Construct PyVista meshes from external CAD objects with the
module-level :func:`from_build123d`, :func:`from_cadquery`, and
:func:`from_topods` functions (see *Bridges to other CAD libraries*
above). The accessor
exposes only ``to_*`` (outgoing) conversions; it is bound to an
existing dataset, so it has no ``from_*`` constructors.

.. autosummary::
   :toctree: _autosummary

   CadDataSetAccessor
   CadMultiBlockAccessor

CadDataSetAccessor members
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   CadDataSetAccessor.source_format
   CadDataSetAccessor.units
   CadDataSetAccessor.label
   CadDataSetAccessor.color
   CadDataSetAccessor.metadata
   CadDataSetAccessor.layers
   CadDataSetAccessor.has_brep_origin
   CadDataSetAccessor.tessellation_quality
   CadDataSetAccessor.set_units
   CadDataSetAccessor.set_metadata
   CadDataSetAccessor.split_by_layer
   CadDataSetAccessor.split_by_label
   CadDataSetAccessor.split_by_color
   CadDataSetAccessor.exact_volume
   CadDataSetAccessor.center_of_mass
   CadDataSetAccessor.to_build123d
   CadDataSetAccessor.to_cadquery
   CadDataSetAccessor.to_topods
   CadDataSetAccessor.to_dxf
   CadDataSetAccessor.to_3mf
   CadDataSetAccessor.to_brep
   CadDataSetAccessor.to_step
   CadDataSetAccessor.to_gltf
   CadDataSetAccessor.tessellate
   CadDataSetAccessor.plot
   CadDataSetAccessor.cad_view

CadMultiBlockAccessor members
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   CadMultiBlockAccessor.set_units
   CadMultiBlockAccessor.set_metadata
   CadMultiBlockAccessor.walk
   CadMultiBlockAccessor.find
   CadMultiBlockAccessor.flatten
   CadMultiBlockAccessor.flatten_to_polydata
   CadMultiBlockAccessor.assembly_tree
   CadMultiBlockAccessor.split_by_layer
   CadMultiBlockAccessor.split_by_label
   CadMultiBlockAccessor.split_by_color
   CadMultiBlockAccessor.exact_volume
   CadMultiBlockAccessor.to_build123d
   CadMultiBlockAccessor.to_cadquery
   CadMultiBlockAccessor.to_topods
   CadMultiBlockAccessor.to_dxf
   CadMultiBlockAccessor.to_3mf
   CadMultiBlockAccessor.to_brep
   CadMultiBlockAccessor.to_step
   CadMultiBlockAccessor.to_gltf
   CadMultiBlockAccessor.tessellate
   CadMultiBlockAccessor.plot
   CadMultiBlockAccessor.cad_view

Metadata
--------

.. autosummary::
   :toctree: _autosummary

   CadMetadata
   Units

Exceptions and warnings
-----------------------

.. autosummary::
   :toctree: _autosummary

   CadError
   CadReadError
   CadWriteError
   CadCacheError
   MetadataError
   OptionalDependencyError
   TessellationError
   UnsupportedFormatError
   CadWarning

Example data
------------

Bundled fixture files and downloadable example datasets.

.. currentmodule:: pyvista_cad.examples

.. autosummary::
   :toctree: _autosummary

   bracket_step_path
   drawing_dxf_path
   assembly_3mf_path
   small_building_ifc_path

.. currentmodule:: pyvista_cad.examples.downloads

.. autosummary::
   :toctree: _autosummary

   step_part_path
   step_recoater_path
   step_assembly_path
   iges_impeller_path
   ifc_building_path
   three_mf_colored_path
   gltf_toycar_path
   dxf_drawing_path
   scad_threads_path
   fcstd_nut_path
   scan_pair_paths
