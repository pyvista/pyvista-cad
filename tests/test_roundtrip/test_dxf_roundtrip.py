"""DXF read -> write -> re-read behavioral invariants."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._metadata import _VALID_UNITS

# Convertible (non-unitless) units that map to a DXF $INSUNITS code.
_DXF_INSUNITS_UNITS = sorted(_VALID_UNITS - {'unitless'})


def _src() -> pv.PolyData:
    return pyvista_cad.read_dxf(pyvista_cad.examples.dxf_plate)


def test_dxf_round_trip_preserves_layer_set(tmp_path: Path) -> None:
    """The full set of distinct layer names survives a DXF round-trip."""
    src = _src()
    out = tmp_path / 'rt.dxf'
    pyvista_cad.write_dxf(src, out)
    rt = pyvista_cad.read_dxf(out)
    src_layers = sorted({str(x) for x in src.cell_data['Layer']})
    rt_layers = sorted({str(x) for x in rt.cell_data['Layer']})
    assert rt_layers == src_layers
    # cad.layer mirror must match Layer exactly.
    assert sorted({str(x) for x in rt.cell_data['cad.layer']}) == src_layers


@pytest.mark.parametrize('color_index', [0, 1, 7, 255, 256])
def test_dxf_round_trip_preserves_color(tmp_path: Path, color_index: int) -> None:
    """Whole-mesh ``color_index`` round-trips for every ACI table edge case.

    Parametrised across the AutoCAD Color Index extremes:
    ``0`` (ByBlock), ``1`` (red), ``7`` (white/black-on-white default),
    ``255`` (last numeric ACI), and ``256`` (ByLayer).
    The current ezdxf writer emits a single mesh-wide color; per-cell color is
    a known gap. This test pins the whole-mesh behaviour for the full ACI
    domain.
    """
    src = _src()
    out = tmp_path / f'rt_{color_index}.dxf'
    pyvista_cad.write_dxf(src, out, color_index=color_index)
    # Re-reading does not surface ACI today; pin via raw ezdxf to assert intent.
    ezdxf = pytest.importorskip('ezdxf')
    doc = ezdxf.readfile(str(out))
    entity_colors = {e.dxf.color for e in doc.modelspace()}
    assert entity_colors == {color_index}


@pytest.mark.parametrize('unit', _DXF_INSUNITS_UNITS)
def test_dxf_round_trip_preserves_insunits(tmp_path: Path, unit: str) -> None:
    """``$INSUNITS`` survives R2018 round-trip for every convertible unit."""
    src = _src()
    src.field_data['cad.units'] = np.array([unit])
    out = tmp_path / f'rt_{unit}.dxf'
    pyvista_cad.write_dxf(src, out, dxf_version='R2018')
    rt = pyvista_cad.read_dxf(out)
    assert str(rt.field_data['cad.units'][0]) == unit


def test_dxf_round_trip_preserves_dxf_version(tmp_path: Path) -> None:
    """The ``dxf_version`` argument determines ``cad.source_format_version``."""
    src = _src()
    for ver in ('R12', 'R2018'):
        out = tmp_path / f'{ver}.dxf'
        pyvista_cad.write_dxf(src, out, dxf_version=ver)
        rt = pyvista_cad.read_dxf(out)
        assert str(rt.field_data['cad.source_format_version'][0]) == ver
