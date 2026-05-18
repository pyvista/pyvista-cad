"""Tests for :mod:`pyvista_cad.examples` and its ``downloads`` submodule.

Two concerns are covered:

* The bundled loaders in :mod:`pyvista_cad.examples` resolve paths to
  files shipped inside the wheel. Every public asset is loaded with its
  matching reader and the resulting dataset is checked for type, non-empty
  geometry, finite bounds, and the expected ``cad.*`` metadata. No network
  is involved; the only skip is a STEP case on a runner with no STEP
  backend (e.g. Windows, where the OCP DLL won't load and cascadio is
  not in the extra).
* The :mod:`pyvista_cad.examples.downloads` helpers fetch hash-verified
  files through ``pooch``. The registry/URL/hash construction is asserted
  for every entry unconditionally (a real assertion on the live ``pooch``
  registry object, so the wrappers are exercised without a socket). One
  real end-to-end fetch-and-read per representative format is additionally
  performed, guarded by an actual-connectivity check so it stays
  CI-runnable and offline-safe.
"""

from collections.abc import Callable
import importlib
import socket

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad import examples

pooch = pytest.importorskip('pooch')

from pyvista_cad.examples import downloads  # noqa: E402


def _step_backend_available() -> bool:
    """True if any STEP backend imports here.

    ``read_step`` needs cascadio or build123d/OCP. On a Windows runner
    the OCP DLL often fails to load (a bare ``ImportError``, not
    ``ModuleNotFoundError``) and cascadio is absent from the ``dev`` /
    ``dev,step`` extras, so STEP is genuinely unavailable — skip those
    cases rather than fail.
    """

    def _ok(mod: str) -> bool:
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
        return True

    return any(_ok(mod) for mod in ('cascadio', 'build123d', 'OCP'))


_requires_step_backend = pytest.mark.skipif(
    not _step_backend_available(),
    reason='no STEP backend importable (cascadio/build123d/OCP unavailable)',
)


def _has_network(host: str = 'raw.githubusercontent.com') -> bool:
    """Return ``True`` only if a TCP connection to ``host`` succeeds."""
    try:
        with socket.create_connection((host, 443), timeout=5):
            return True
    except OSError:
        return False


_NETWORK = _has_network()
_requires_network = pytest.mark.skipif(
    not _NETWORK,
    reason='no network: live pooch fetch unavailable in this sandbox',
)


def _leaf(obj: pv.DataSet | pv.MultiBlock) -> pv.DataSet:
    """Return the first non-empty leaf dataset of ``obj``."""
    node = obj
    while isinstance(node, pv.MultiBlock):
        node = next(b for b in node if b is not None)
    return node


def _assert_well_formed(
    obj: pv.DataSet | pv.MultiBlock,
    *,
    expected_type: type,
    source_format: str,
    backend: str,
) -> None:
    """Assert geometry and ``cad.*`` provenance on a read dataset."""
    assert isinstance(obj, expected_type)
    leaf = _leaf(obj)
    assert leaf.n_points > 0
    assert leaf.n_cells > 0

    bounds = (obj.combine() if isinstance(obj, pv.MultiBlock) else obj).bounds
    assert len(bounds) == 6
    assert np.all(np.isfinite(bounds))
    assert bounds[0] <= bounds[1]
    assert bounds[2] <= bounds[3]
    assert bounds[4] <= bounds[5]

    assert leaf.cad.source_format == source_format
    assert list(leaf.field_data['cad.backend']) == [backend]


# --------------------------------------------------------------------------- #
# Bundled loaders (pyvista_cad.examples.__init__) -- no network.
# --------------------------------------------------------------------------- #

# (loader, reader, MultiBlock|PolyData, cad.source_format, cad.backend)
_BUNDLED: list[tuple[str, Callable[[], str], Callable[[str], object], type, str, str]] = [
    (
        'step_cube',
        lambda: examples.step_cube,
        pyvista_cad.read_step,
        pv.MultiBlock,
        'step',
        'build123d',
    ),
    (
        'dxf_plate',
        lambda: examples.dxf_plate,
        pyvista_cad.read_dxf,
        pv.PolyData,
        'dxf',
        'ezdxf',
    ),
    (
        'bracket_step_path',
        examples.bracket_step_path,
        pyvista_cad.read_step,
        pv.MultiBlock,
        'step',
        'build123d',
    ),
    (
        'drawing_dxf_path',
        examples.drawing_dxf_path,
        pyvista_cad.read_dxf,
        pv.PolyData,
        'dxf',
        'ezdxf',
    ),
    (
        'assembly_3mf_path',
        examples.assembly_3mf_path,
        pyvista_cad.read_three_mf,
        pv.MultiBlock,
        '3mf',
        'lib3mf',
    ),
    (
        'small_building_ifc_path',
        examples.small_building_ifc_path,
        pyvista_cad.read_ifc,
        pv.MultiBlock,
        'ifc',
        'ifcopenshell',
    ),
]


@pytest.mark.parametrize(
    ('name', 'loader', 'reader', 'kind', 'source_format', 'backend'),
    _BUNDLED,
    ids=[row[0] for row in _BUNDLED],
)
def test_bundled_example_loads(
    name: str,
    loader: Callable[[], str],
    reader: Callable[[str], object],
    kind: type,
    source_format: str,
    backend: str,
) -> None:
    """Every bundled asset resolves, reads, and carries ``cad.*`` metadata."""
    if source_format == 'step' and not _step_backend_available():
        pytest.skip('no STEP backend importable (cascadio/build123d/OCP unavailable)')
    path = loader()
    assert isinstance(path, str)
    assert path.endswith(('.step', '.dxf', '.3mf', '.ifc')), (
        f'{name} resolved to unexpected extension: {path}'
    )

    obj = reader(path)
    _assert_well_formed(
        obj,
        expected_type=kind,
        source_format=source_format,
        backend=backend,
    )


@_requires_step_backend
def test_step_cube_bounds_are_the_10mm_cube() -> None:
    """The bundled cube is the documented 10 mm cube centred on the origin."""
    mb = pyvista_cad.read_step(examples.step_cube)
    b = mb.combine().bounds
    assert b.x_min == pytest.approx(-5.0, abs=0.5)
    assert b.x_max == pytest.approx(5.0, abs=0.5)
    assert b.y_min == pytest.approx(-5.0, abs=0.5)
    assert b.y_max == pytest.approx(5.0, abs=0.5)
    assert b.z_min == pytest.approx(-5.0, abs=0.5)
    assert b.z_max == pytest.approx(5.0, abs=0.5)


def test_bundled_module_all_is_public_surface() -> None:
    """Everything in ``__all__`` is importable and the loaders are callable."""
    for attr in examples.__all__:
        assert hasattr(examples, attr)
    assert examples.downloads is downloads
    for path_fn in (
        examples.bracket_step_path,
        examples.drawing_dxf_path,
        examples.assembly_3mf_path,
        examples.small_building_ifc_path,
    ):
        assert isinstance(path_fn(), str)


# --------------------------------------------------------------------------- #
# downloads.py -- registry construction (always runs, no socket).
# --------------------------------------------------------------------------- #


def test_download_registry_matches_files_table() -> None:
    """``_POOCH`` is built consistently from the ``_FILES`` source table."""
    assert set(downloads._POOCH.registry) == set(downloads._FILES)
    assert len(downloads._FILES) == 11
    # pooch.os_cache puts the app name in the path but appends a
    # platform-specific leaf ('Cache' on Windows), so assert the app
    # directory is present rather than that it is the last component.
    assert 'pyvista_cad' in downloads._POOCH.abspath.parts


@pytest.mark.parametrize('name', sorted(downloads._FILES))
def test_download_registry_entry_well_formed(name: str) -> None:
    """Every registry entry has an ``https`` URL and a 64-hex sha256."""
    url, sha = downloads._FILES[name]
    assert url == downloads._POOCH.get_url(name)
    assert url.startswith('https://')
    assert downloads._POOCH.registry[name] == f'sha256:{sha}'
    assert len(sha) == 64
    assert all(c in '0123456789abcdef' for c in sha)


_DOWNLOADERS: list[tuple[str, Callable[[], object]]] = [
    ('step_part_path', downloads.step_part_path),
    ('step_recoater_path', downloads.step_recoater_path),
    ('step_assembly_path', downloads.step_assembly_path),
    ('iges_impeller_path', downloads.iges_impeller_path),
    ('ifc_building_path', downloads.ifc_building_path),
    ('three_mf_colored_path', downloads.three_mf_colored_path),
    ('gltf_toycar_path', downloads.gltf_toycar_path),
    ('dxf_drawing_path', downloads.dxf_drawing_path),
    ('scad_threads_path', downloads.scad_threads_path),
    ('fcstd_nut_path', downloads.fcstd_nut_path),
]


@pytest.mark.parametrize('name', [n for n, _ in _DOWNLOADERS], ids=[n for n, _ in _DOWNLOADERS])
def test_downloader_is_public_and_callable(name: str) -> None:
    """Public downloader names are exported and reference real callables."""
    assert name in downloads.__all__
    assert callable(getattr(downloads, name))


# --------------------------------------------------------------------------- #
# downloads.py -- one real fetch+read per representative format.
# Network-guarded; pooch caches so this is deterministic and offline-safe
# after the first run. Never skipped for the bundled-loader tests above.
# --------------------------------------------------------------------------- #


@_requires_network
@_requires_step_backend
def test_fetch_step_recoater_reads_correctly() -> None:
    """Smallest NIST STEP part fetches, hash-verifies, and reads as solid."""
    obj = pyvista_cad.read_step(downloads.step_recoater_path())
    _assert_well_formed(
        obj,
        expected_type=pv.MultiBlock,
        source_format='step',
        backend='build123d',
    )


@_requires_network
def test_fetch_dxf_drawing_reads_correctly() -> None:
    """ezdxf sample DXF fetches and reads with DXF provenance."""
    obj = pyvista_cad.read_dxf(downloads.dxf_drawing_path())
    _assert_well_formed(
        obj,
        expected_type=pv.PolyData,
        source_format='dxf',
        backend='ezdxf',
    )


@_requires_network
def test_fetch_three_mf_colored_reads_correctly() -> None:
    """3MF Consortium colored sample fetches and reads as a MultiBlock."""
    obj = pyvista_cad.read_three_mf(downloads.three_mf_colored_path())
    _assert_well_formed(
        obj,
        expected_type=pv.MultiBlock,
        source_format='3mf',
        backend='lib3mf',
    )


@_requires_network
def test_fetch_gltf_toycar_reads_correctly() -> None:
    """Khronos ToyCar binary glTF fetches and reads as a MultiBlock."""
    obj = pyvista_cad.read_gltf(downloads.gltf_toycar_path())
    assert isinstance(obj, pv.MultiBlock)
    assert _leaf(obj).n_points > 0


@_requires_network
def test_fetch_remaining_step_and_iges_downloaders() -> None:
    """Exercise the remaining STEP/IGES wrappers via the shared resolver."""
    for fetch in (
        downloads.step_part_path,
        downloads.step_assembly_path,
        downloads.iges_impeller_path,
    ):
        path = fetch()
        assert path.endswith(('.step', '.STEP', '.igs'))

    obj = pyvista_cad.read_iges(downloads.iges_impeller_path())
    _assert_well_formed(
        obj,
        expected_type=pv.PolyData,
        source_format='iges',
        backend='pyiges',
    )


@_requires_network
def test_fetch_ifc_building_reads_correctly() -> None:
    """buildingSMART IFC4 sample fetches and reads as a spatial MultiBlock."""
    obj = pyvista_cad.read_ifc(downloads.ifc_building_path())
    _assert_well_formed(
        obj,
        expected_type=pv.MultiBlock,
        source_format='ifc',
        backend='ifcopenshell',
    )


@_requires_network
def test_fetch_fcstd_and_scad_source_downloaders() -> None:
    """The FCStd and OpenSCAD source wrappers fetch and hash-verify."""
    fcstd = downloads.fcstd_nut_path()
    assert fcstd.endswith('.fcstd')
    scad = downloads.scad_threads_path()
    assert scad.endswith('.scad')
    assert 'module' in __import__('pathlib').Path(scad).read_text(
        encoding='utf-8', errors='replace'
    )


@_requires_network
def test_fetch_scan_pair_unzips_both_members() -> None:
    """``scan_pair_paths`` unzips and returns the CAD and scan PLY pair."""
    cad_path, scan_path = downloads.scan_pair_paths()
    assert cad_path.endswith('obj_000005.ply')
    assert scan_path.endswith('obj_000005.ply')
    assert cad_path != scan_path

    cad = pv.read(cad_path)
    scan = pv.read(scan_path)
    assert cad.n_points > 0
    assert scan.n_points > 0
