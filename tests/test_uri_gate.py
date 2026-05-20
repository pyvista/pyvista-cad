"""URI gate: every pyvista-cad reader must raise ``LocalFileRequiredError``
when handed a remote URI so that :func:`pyvista.read` falls back to
``_download_uri`` and retries with a local path.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import pyvista as pv
from pyvista.core.utilities import reader_registry as rr
from pyvista.core.utilities.reader_registry import LocalFileRequiredError

import pyvista_cad  # registers readers as a side effect

DATA = Path(__file__).parent / 'data'

# Map extensions registered by pyvista-cad to a fixture file that the
# integration test can resolve through a stubbed ``_download_uri``.
_EXT_FIXTURE: dict[str, Path | None] = {
    '.dxf': DATA / 'dxf_r12.dxf',
    '.step': DATA / 'assembly_two_parts.step',
    '.stp': DATA / 'assembly_two_parts.step',
    '.iges': DATA / 'impeller.iges',
    '.igs': DATA / 'impeller.iges',
    '.brep': DATA / 'external.brep',
    '.brp': DATA / 'external.brep',
    '.3mf': DATA / 'multimaterial.3mf',
    '.ifc': DATA / 'building.ifc',
    '.fcstd': None,  # no bundled fixture
    '.scad': None,  # no bundled fixture
    '.gltf': None,  # no bundled fixture under tests/data
    '.glb': None,
}


def _pyvista_cad_readers() -> list[rr.ReaderRegistration]:
    return [r for r in rr.registered_readers() if r.source.startswith('pyvista_cad')]


def test_pyvista_cad_owns_expected_extensions() -> None:
    exts = {r.extension for r in _pyvista_cad_readers()}
    assert exts == set(_EXT_FIXTURE), exts


@pytest.mark.parametrize('reg', _pyvista_cad_readers(), ids=lambda r: r.extension)
def test_reader_raises_local_file_required_on_remote_uri(
    reg: rr.ReaderRegistration,
) -> None:
    """Every registered reader must short-circuit on a URI input.

    No backend is imported (the gate fires before backend lookup), so
    this runs even when the optional extra is not installed.
    """
    uri = f'https://example.invalid/path/sample{reg.extension}'
    with pytest.raises(LocalFileRequiredError):
        reg.handler(uri)


@pytest.mark.parametrize(
    'ext',
    [e for e, p in _EXT_FIXTURE.items() if p is not None and p.exists()],
)
def test_pv_read_round_trips_remote_uri(
    ext: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pv.read('https://...ext')`` must download and dispatch.

    We stub :func:`pyvista.core.utilities.reader_registry._download_uri`
    so the test stays offline; the real codepath inside ``_read_dispatch``
    catches :class:`LocalFileRequiredError`, calls ``_download_uri``, and
    re-invokes the reader with the returned local path.
    """
    src = _EXT_FIXTURE[ext]
    assert src is not None
    assert src.exists()

    calls = {'n': 0}

    def fake_download(uri: str, uri_ext: str) -> str:
        calls['n'] += 1
        assert uri.startswith('https://')
        assert uri_ext == ext
        dst = tmp_path / f'downloaded{ext}'
        shutil.copyfile(src, dst)
        return str(dst)

    monkeypatch.setattr(
        'pyvista.core.utilities.reader_registry._download_uri',
        fake_download,
    )

    uri = f'https://example.invalid/path/sample{ext}'
    try:
        result = pv.read(uri)
    except pyvista_cad.OptionalDependencyError:
        pytest.skip(f'backend for {ext} not installed')
    assert calls['n'] == 1
    assert result is not None
