"""PyVista accessor and reader plugin for CAD formats.

Importing this module registers the ``.cad`` accessor on
:class:`pyvista.DataSet` and :class:`pyvista.MultiBlock` and registers
the CAD reader entry points with :func:`pyvista.read`.

Examples
--------
>>> import pyvista as pv
>>> import pyvista_cad
>>> sphere = pv.Sphere()
>>> sphere.cad  # doctest: +ELLIPSIS
<CadDataSetAccessor on PolyData>

"""

from typing import TYPE_CHECKING, Any

# Eager: the ``.cad`` accessor must be registered at import time so
# ``pv.Sphere().cad`` works without any further imports.
from pyvista_cad import _accessor as _accessor
from pyvista_cad._accessor import (
    CadDataSetAccessor,
    CadMultiBlockAccessor,
)

# NOTE: the ``plotter.cad`` component is registered lazily via the
# ``pyvista.plotter_components`` entry point (see ``pyproject.toml``);
# pyvista imports ``pyvista_cad._cad_view`` only on first ``plotter.cad``
# access. Importing it eagerly here pulls in ``pyvista.plotting`` and
# blows the cold-import budget, so we do not.
from pyvista_cad._errors import (
    CadCacheError,
    CadError,
    CadReadError,
    CadWarning,
    CadWriteError,
    MetadataError,
    OptionalDependencyError,
    TessellationError,
    UnsupportedFormatError,
)

if TYPE_CHECKING:  # pragma: no cover - types only
    from pyvista_cad import examples as examples
    from pyvista_cad._bridges.build123d import from_build123d, to_build123d
    from pyvista_cad._bridges.cadquery import from_cadquery, to_cadquery
    from pyvista_cad._bridges.gmsh import from_gmsh, to_gmsh
    from pyvista_cad._bridges.topods import from_topods, to_topods
    from pyvista_cad._cad_view import (
        add_cad,
        as_cad_multiblock,
        flatten_to_cad_polydata,
        get_cad_edges,
        plot_cad,
        topods_to_edges,
        topods_to_multiblock,
    )
    from pyvista_cad._metadata import CadMetadata, Units
    from pyvista_cad._readers.brep import read_brep, write_brep
    from pyvista_cad._readers.dxf import read_dxf, write_dxf
    from pyvista_cad._readers.fcstd import read_fcstd
    from pyvista_cad._readers.ifc import read_ifc
    from pyvista_cad._readers.iges import read_iges
    from pyvista_cad._readers.scad import read_scad
    from pyvista_cad._readers.step import read_step, write_step
    from pyvista_cad._readers.three_mf import read_three_mf, write_three_mf


# Lazy public-symbol table: ``name -> (module_path, attribute)``.
_LAZY: dict[str, tuple[str, str]] = {
    'CadMetadata': ('pyvista_cad._metadata', 'CadMetadata'),
    'Units': ('pyvista_cad._metadata', 'Units'),
    'examples': ('pyvista_cad.examples', None),
    'from_build123d': ('pyvista_cad._bridges.build123d', 'from_build123d'),
    'to_build123d': ('pyvista_cad._bridges.build123d', 'to_build123d'),
    'from_cadquery': ('pyvista_cad._bridges.cadquery', 'from_cadquery'),
    'to_cadquery': ('pyvista_cad._bridges.cadquery', 'to_cadquery'),
    'from_gmsh': ('pyvista_cad._bridges.gmsh', 'from_gmsh'),
    'to_gmsh': ('pyvista_cad._bridges.gmsh', 'to_gmsh'),
    'from_topods': ('pyvista_cad._bridges.topods', 'from_topods'),
    'to_topods': ('pyvista_cad._bridges.topods', 'to_topods'),
    'add_cad': ('pyvista_cad._cad_view', 'add_cad'),
    'as_cad_multiblock': ('pyvista_cad._cad_view', 'as_cad_multiblock'),
    'plot_cad': ('pyvista_cad._cad_view', 'plot_cad'),
    'topods_to_edges': ('pyvista_cad._cad_view', 'topods_to_edges'),
    'topods_to_multiblock': ('pyvista_cad._cad_view', 'topods_to_multiblock'),
    'flatten_to_cad_polydata': ('pyvista_cad._cad_view', 'flatten_to_cad_polydata'),
    'get_cad_edges': ('pyvista_cad._cad_view', 'get_cad_edges'),
    'read_brep': ('pyvista_cad._readers.brep', 'read_brep'),
    'write_brep': ('pyvista_cad._readers.brep', 'write_brep'),
    'read_dxf': ('pyvista_cad._readers.dxf', 'read_dxf'),
    'write_dxf': ('pyvista_cad._readers.dxf', 'write_dxf'),
    'read_fcstd': ('pyvista_cad._readers.fcstd', 'read_fcstd'),
    'read_ifc': ('pyvista_cad._readers.ifc', 'read_ifc'),
    'read_iges': ('pyvista_cad._readers.iges', 'read_iges'),
    'read_scad': ('pyvista_cad._readers.scad', 'read_scad'),
    'read_step': ('pyvista_cad._readers.step', 'read_step'),
    'write_step': ('pyvista_cad._readers.step', 'write_step'),
    'read_three_mf': ('pyvista_cad._readers.three_mf', 'read_three_mf'),
    'write_three_mf': ('pyvista_cad._readers.three_mf', 'write_three_mf'),
}


def __getattr__(name: str) -> Any:
    """Lazily import public symbols on first access.

    Defers ~1-2 ms of per-submodule import overhead off the cold
    ``import pyvista_cad`` path. Once resolved, the attribute is
    cached on the module so subsequent lookups skip this hook.
    """
    target = _LAZY.get(name)
    if target is None:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)
    mod_path, attr = target
    import importlib

    mod = importlib.import_module(mod_path)
    value = mod if attr is None else getattr(mod, attr)
    globals()[name] = value
    return value


def _read_gltf_trampoline(path, **kwargs):
    """Defer the glTF backend import until a glTF/GLB file is read.

    Registered eagerly (below) with ``override=True`` so this
    metadata-aware reader supersedes pyvista's built-in VTK glTF
    reader. ``read_gltf`` reads the base geometry via
    :class:`pyvista.GLTFReader` directly, so this registration does
    not recurse back through ``pyvista.read``.
    """
    from pyvista_cad._readers.gltf import read_gltf

    return read_gltf(path, **kwargs)


def _read_iges_trampoline(path, **kwargs):
    """Defer the pyiges import until an IGES file is actually read.

    Registered eagerly (below) so ``pyvista_cad``'s IGES reader wins
    deterministically over ``pyiges``'s own ``pyvista.readers`` entry
    point. ``pyvista`` consults eagerly registered readers before
    resolving entry points, so this also suppresses the multi-provider
    warning that two ``.iges`` entry points would otherwise raise. The
    heavy ``pyiges`` import stays off the bare-import path.
    """
    from pyvista_cad._readers.iges import read_iges

    return read_iges(path, **kwargs)


# Eager, lightweight reader registration for IGES only. ``pyiges`` also
# ships a ``pyvista.readers`` entry point for ``.iges``/``.igs``; entry
# point ordering across distributions is not guaranteed, so we register
# our metadata-aware reader eagerly with ``override=True`` to guarantee
# it wins. Every other extension is wired purely through this package's
# ``pyvista.readers`` entry points (see ``pyproject.toml``); ``pyvista``
# discovers and loads those lazily without importing any reader module.
import pyvista as _pv

for _iges_ext in ('.iges', '.igs'):
    _pv.register_reader(_iges_ext, _read_iges_trampoline, override=True)
for _gltf_ext in ('.gltf', '.glb'):
    _pv.register_reader(_gltf_ext, _read_gltf_trampoline, override=True)
del _pv, _iges_ext, _gltf_ext


def enrich_gltf(dataset, path):
    """Attach glTF node names / extras to a dataset's field data.

    Returns
    -------
    pyvista.DataSet
        The same dataset, with ``cad.*`` manifest metadata attached.

    """
    from pyvista_cad._backends._gltf import enrich_gltf as _enrich

    return _enrich(dataset, path)


def read_gltf(path):
    """Read a glTF/GLB file and enrich it with manifest metadata.

    Returns
    -------
    pyvista.DataSet
        The parsed mesh with ``cad.*`` manifest metadata attached.

    """
    from pyvista_cad._readers.gltf import read_gltf as _read_gltf

    return _read_gltf(path)


try:
    from pyvista_cad._version import version as __version__
except ImportError:
    __version__ = '0.0.0+unknown'

__all__ = [
    'CadCacheError',
    'CadDataSetAccessor',
    'CadError',
    'CadMetadata',
    'CadMultiBlockAccessor',
    'CadReadError',
    'CadWarning',
    'CadWriteError',
    'MetadataError',
    'OptionalDependencyError',
    'TessellationError',
    'Units',
    'UnsupportedFormatError',
    '__version__',
    'add_cad',
    'as_cad_multiblock',
    'enrich_gltf',
    'examples',
    'flatten_to_cad_polydata',
    'from_build123d',
    'from_cadquery',
    'from_gmsh',
    'from_topods',
    'get_cad_edges',
    'plot_cad',
    'read_brep',
    'read_dxf',
    'read_fcstd',
    'read_gltf',
    'read_ifc',
    'read_iges',
    'read_scad',
    'read_step',
    'read_three_mf',
    'to_build123d',
    'to_cadquery',
    'to_gmsh',
    'to_topods',
    'topods_to_edges',
    'topods_to_multiblock',
    'write_brep',
    'write_dxf',
    'write_step',
    'write_three_mf',
]
