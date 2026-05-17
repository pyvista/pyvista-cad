"""Shared fixtures for pyvista-cad tests."""

from collections.abc import Generator
import importlib
from pathlib import Path

import pytest
import pyvista as pv

importlib.import_module('pyvista_cad')

pv.OFF_SCREEN = True

_IMAGE_CACHE_DIR = Path(__file__).parent / 'image_cache'

DATA_DIR = Path(__file__).parent / 'data'


@pytest.fixture(scope='session')
def data_dir() -> Path:
    """Directory holding the committed third-party CAD fixtures."""
    return DATA_DIR


@pytest.fixture
def assembly_step_path() -> Path:
    """Two-part STEP assembly written via STEPCAFControl_Writer + OCAF."""
    return DATA_DIR / 'assembly_two_parts.step'


@pytest.fixture
def dxf_r12_path() -> Path:
    """ezdxf R12 DXF fixture."""
    return DATA_DIR / 'dxf_r12.dxf'


@pytest.fixture
def dxf_r2018_path() -> Path:
    """ezdxf R2018 DXF fixture."""
    return DATA_DIR / 'dxf_r2018.dxf'


@pytest.fixture
def external_brep_path() -> Path:
    """OCCT ASCII BREP written by BRepTools (not by our writer)."""
    return DATA_DIR / 'external.brep'


@pytest.fixture
def impeller_iges_path() -> Path:
    """B-spline impeller IGES fixture written via IGESControl_Writer."""
    return DATA_DIR / 'impeller.iges'


@pytest.fixture
def multimaterial_3mf_path() -> Path:
    """Two-object lib3mf model with per-object BaseMaterial colors."""
    return DATA_DIR / 'multimaterial.3mf'


@pytest.fixture
def building_ifc_path() -> Path:
    """IFC4 building with one diffuse-styled wall."""
    return DATA_DIR / 'building.ifc'


@pytest.fixture(scope='session', autouse=True)
def _ensure_image_cache_dir() -> None:
    """Session-level setup: make sure the image-cache directory exists.

    ``pytest-pyvista`` reads ``image_cache_dir`` from ``pyproject.toml``
    and validates it lazily when ``verify_image_cache`` is requested.
    Materialising the directory once per session keeps
    ``--reset_image_cache`` idempotent on fresh checkouts without
    disturbing non-rendering tests (no plotter or fixture wiring beyond
    ensuring the path exists on disk).
    """
    _IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def set_default_theme() -> Generator[None, None, None]:
    """Reset the testing theme for every test."""
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
    yield
    pv.global_theme.load_theme(pv.plotting.themes._TestingTheme())
