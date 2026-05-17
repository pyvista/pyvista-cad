"""Real-world example datasets, downloaded and cached on demand.

Every helper here fetches a *real*, openly licensed file from a public
repository, caches it under the pooch cache directory, verifies its
SHA256, and returns a local filesystem path. These back the gallery
examples so the documentation demonstrates genuine engineering data
(NIST additive-manufacturing parts, a buildingSMART BIM model, a
depth-camera reconstruction, ...) rather than synthetic toys.

Files are cached in :func:`pooch.os_cache` under ``pyvista_cad`` and
downloaded only once. Set the ``PYVISTA_CAD_USERDATA_PATH`` environment
variable to override the cache location.

Dataset provenance and licensing (each loader's docstring repeats the
license of the file it returns):

- NIST AM Bench 2022 STEP geometry (``step_part``, ``step_recoater``,
  ``step_assembly``): U.S. Government work, public domain (17 U.S.C.
  105). https://doi.org/10.18434/mds2-2607
- pyiges impeller (``iges_impeller``): MIT, The PyVista Developers.
  https://github.com/pyvista/pyiges
- buildingSMART Sample-Test-Files (``ifc_building``): CC-BY 4.0,
  buildingSMART International Ltd.
  https://github.com/buildingSMART/Sample-Test-Files
- 3MF Consortium samples (``three_mf_colored``): BSD-2-Clause, 3MF
  Consortium. https://github.com/3MFConsortium/3mf-samples
- Khronos glTF-Sample-Assets ToyCar (``gltf_toycar``): CC0 1.0.
  https://github.com/KhronosGroup/glTF-Sample-Assets
- ezdxf example drawings (``dxf_drawing``): MIT, Manfred Moitzi.
  https://github.com/mozman/ezdxf
- rcolyer/threads-scad (``scad_threads``): CC0 1.0, Ryan A. Colyer.
  https://github.com/rcolyer/threads-scad
- FreeCAD-library (``fcstd_nut``): CC-BY 3.0, FreeCAD-library
  contributors. https://github.com/FreeCAD/FreeCAD-library
- T-LESS object models (``scan_pair``): CC-BY 4.0, Hodaň et al.,
  WACV 2017. https://huggingface.co/datasets/bop-benchmark/tless

"""

from pathlib import Path

import pooch

# name -> (url, sha256)
_FILES: dict[str, tuple[str, str]] = {
    'step_part.step': (
        'https://data.nist.gov/od/ds/ark:/88434/mds2-2607/'
        'CAD_Geometry/AMB2022-01-AMMT-PartCAD.STEP',
        '623ee615d015bb231accdff3594592c3c7c563d98c463bd21680da2c0005d055',
    ),
    'step_recoater.step': (
        'https://data.nist.gov/od/ds/ark:/88434/mds2-2607/'
        'CAD_Geometry/AMB2022-01-AMMT-RecoaterGuideCAD.STEP',
        'eae39537e677cac648df0f0660c65d8071df868f64bbe9b5605254c3fde362ab',
    ),
    'step_assembly.step': (
        'https://data.nist.gov/od/ds/ark:/88434/mds2-2607/'
        'CAD_Geometry/AMB2022-01-AMMT-PlateLayoutAssy.STEP',
        'a434876ff9a19ce27a340011d471f14014ccc12d648a8e99a3bdcc817b11e0be',
    ),
    'iges_impeller.igs': (
        'https://raw.githubusercontent.com/pyvista/pyiges/main/src/pyiges/examples/impeller.igs',
        'c9fa023feddc88dfc5a73da00d3cd2d9987b94b27237f28a7f91b09881203df5',
    ),
    'ifc_building.ifc': (
        'https://raw.githubusercontent.com/buildingSMART/Sample-Test-Files/'
        'main/IFC%204.0.2.1%20(IFC%204)/PCERT-Sample-Scene/'
        'Building-Architecture.ifc',
        '3ff9b10bd00c7b96dded51e7ca5a6b69efbea38b049adcdd05fcd247de7e70d5',
    ),
    'three_mf_colored.3mf': (
        'https://raw.githubusercontent.com/3MFConsortium/3mf-samples/'
        'master/examples/material/dodeca_chain_loop_color.3mf',
        'fe4bdcd667044547faf50ce29d0af7c6f2ea39c11caa8c4b26e841d888863cc0',
    ),
    'gltf_toycar.glb': (
        'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/'
        'main/Models/ToyCar/glTF-Binary/ToyCar.glb',
        '01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c',
    ),
    'dxf_drawing.dxf': (
        'https://raw.githubusercontent.com/mozman/ezdxf/master/examples_dxf/colors.dxf',
        '70c12f1e3fd291c4ab02df7628624956fce315263c0ebcb18734d0b6e5b9974a',
    ),
    'scad_threads.scad': (
        'https://raw.githubusercontent.com/rcolyer/threads-scad/'
        '5f25c7c52b3c59339960185b19f28ef2a1a1e693/threads.scad',
        'f8ee04e57a1c721b7d8d68764e55ea448af935242024e94768e7949e384b2bc7',
    ),
    'fcstd_nut.fcstd': (
        'https://raw.githubusercontent.com/FreeCAD/FreeCAD-library/master/'
        'Mechanical%20Parts/Fasteners/Nuts/Metric/Nyloc-Nut_M3.fcstd',
        'd2fe41ac94515b51ed01f176773140795d4e447cf88b87099c369b32e4c11817',
    ),
    'tless_models.zip': (
        'https://huggingface.co/datasets/bop-benchmark/tless/resolve/main/tless_models.zip',
        '6a29c59766b8d2af05e62e71739f0ca7243ad81bc3c7f9a24925504a8cb37928',
    ),
}

_POOCH = pooch.create(
    path=pooch.os_cache('pyvista_cad'),
    base_url='',
    registry={name: f'sha256:{sha}' for name, (_, sha) in _FILES.items()},
    urls={name: url for name, (url, _) in _FILES.items()},
    env='PYVISTA_CAD_USERDATA_PATH',
    retry_if_failed=3,
)


def _fetch(name: str) -> str:
    return str(_POOCH.fetch(name))


def step_part_path() -> str:
    """Return the NIST AM Bench 2022 LPBF bridge specimen (STEP).

    Returns
    -------
    str
        Path to the cached STEP file. Public domain (NIST, 17 U.S.C.
        §105).

    """
    return _fetch('step_part.step')


def step_recoater_path() -> str:
    """Return the small NIST AM Bench recoater-guide fixture (STEP).

    Returns
    -------
    str
        Path to the cached STEP file. Public domain (NIST).

    """
    return _fetch('step_recoater.step')


def step_assembly_path() -> str:
    """Return the NIST AM Bench build-plate layout assembly (STEP).

    The assembly carries real product names and an instance hierarchy
    (tapped substrate plate, four bridge parts, two recoater guides).

    Returns
    -------
    str
        Path to the cached STEP file. Public domain (NIST).

    """
    return _fetch('step_assembly.step')


def iges_impeller_path() -> str:
    """Return a centrifugal impeller exported from SolidWorks (IGES).

    Returns
    -------
    str
        Path to the cached IGES file. MIT (pyiges).

    """
    return _fetch('iges_impeller.igs')


def ifc_building_path() -> str:
    """Return a single-family-house BIM model from buildingSMART (IFC4).

    Full spatial hierarchy (project / site / building / storey) with
    walls, slabs, and property sets.

    Returns
    -------
    str
        Path to the cached IFC file. CC-BY 4.0 (buildingSMART).

    """
    return _fetch('ifc_building.ifc')


def three_mf_colored_path() -> str:
    """Return the 3MF Consortium colored dodecahedron chain (3MF).

    Returns
    -------
    str
        Path to the cached 3MF file. BSD-2-Clause (3MF Consortium).

    """
    return _fetch('three_mf_colored.3mf')


def gltf_toycar_path() -> str:
    """Return the Khronos ToyCar reference asset (binary glTF).

    Returns
    -------
    str
        Path to the cached GLB file. CC0 1.0 (Khronos).

    """
    return _fetch('gltf_toycar.glb')


def dxf_drawing_path() -> str:
    """Return a multi-layer 2D DXF drawing (layers ``0``/``BLUE``/``RED``).

    Returns
    -------
    str
        Path to the cached DXF file. MIT (ezdxf).

    """
    return _fetch('dxf_drawing.dxf')


def scad_threads_path() -> str:
    """Return a parametric ISO metric fastener library (OpenSCAD).

    Compiling the file renders a real bolt/nut/washer demo. Requires
    the ``openscad`` CLI on ``PATH`` to tessellate.

    Returns
    -------
    str
        Path to the cached ``.scad`` source. CC0 1.0 (R. A. Colyer).

    """
    return _fetch('scad_threads.scad')


def fcstd_nut_path() -> str:
    """Return an M3 nyloc lock nut from the FreeCAD parts library (FCStd).

    Returns
    -------
    str
        Path to the cached FCStd file. CC-BY 3.0 (FreeCAD-library
        contributors).

    """
    return _fetch('fcstd_nut.fcstd')


def scan_pair_paths() -> tuple[str, str]:
    """Return as-designed CAD and as-scanned PLY of one T-LESS part.

    Returns the manually authored CAD model and the model
    reconstructed from real Primesense RGB-D scans of the same physical
    object (T-LESS object 5). Both are in millimetres in a shared
    coordinate frame.

    Returns
    -------
    tuple of str
        ``(cad_ply_path, scan_ply_path)``. CC-BY 4.0 (T-LESS).

    """
    members = [
        'models_cad/obj_000005.ply',
        'models_reconst/obj_000005.ply',
    ]
    extracted = _POOCH.fetch(
        'tless_models.zip',
        processor=pooch.Unzip(members=members),
    )
    by_name = {Path(p).as_posix(): p for p in extracted}
    cad = next(p for k, p in by_name.items() if k.endswith(members[0]))
    scan = next(p for k, p in by_name.items() if k.endswith(members[1]))
    return str(cad), str(scan)


__all__ = [
    'dxf_drawing_path',
    'fcstd_nut_path',
    'gltf_toycar_path',
    'ifc_building_path',
    'iges_impeller_path',
    'scad_threads_path',
    'scan_pair_paths',
    'step_assembly_path',
    'step_part_path',
    'step_recoater_path',
    'three_mf_colored_path',
]
