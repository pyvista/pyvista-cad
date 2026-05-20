from os import PathLike as _PathLike

import pyvista as _pv

from pyvista_cad import examples as examples
from pyvista_cad._accessor import (
    CadDataSetAccessor as CadDataSetAccessor,
    CadMultiBlockAccessor as CadMultiBlockAccessor,
)
from pyvista_cad._bridges.build123d import (
    from_build123d as from_build123d,
    to_build123d as to_build123d,
)
from pyvista_cad._bridges.cadquery import (
    from_cadquery as from_cadquery,
    to_cadquery as to_cadquery,
)
from pyvista_cad._bridges.topods import from_topods as from_topods, to_topods as to_topods
from pyvista_cad._errors import (
    CadError as CadError,
    CadReadError as CadReadError,
    CadWriteError as CadWriteError,
    OptionalDependencyError as OptionalDependencyError,
    TessellationError as TessellationError,
    UnsupportedFormatError as UnsupportedFormatError,
)
from pyvista_cad._metadata import CadMetadata as CadMetadata, Units as Units
from pyvista_cad._readers.brep import read_brep as read_brep, write_brep as write_brep
from pyvista_cad._readers.dxf import read_dxf as read_dxf, write_dxf as write_dxf
from pyvista_cad._readers.fcstd import read_fcstd as read_fcstd
from pyvista_cad._readers.ifc import read_ifc as read_ifc
from pyvista_cad._readers.iges import read_iges as read_iges
from pyvista_cad._readers.scad import read_scad as read_scad
from pyvista_cad._readers.step import read_step as read_step, write_step as write_step
from pyvista_cad._readers.three_mf import (
    read_three_mf as read_three_mf,
    write_three_mf as write_three_mf,
)

def enrich_gltf(dataset: _pv.DataSet, path: str | _PathLike[str]) -> _pv.DataSet: ...
def read_gltf(path: str | _PathLike[str]) -> _pv.DataSet: ...

__version__: str
