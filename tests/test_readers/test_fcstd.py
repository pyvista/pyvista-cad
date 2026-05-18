"""Tests for the FCStd reader."""

from pathlib import Path
import zipfile

import pytest
import pyvista as pv

import pyvista_cad
from pyvista_cad._backends._freecad import _parse_document_xml

_DOC_XML = """<?xml version='1.0' encoding='utf-8'?>
<Document SchemaVersion="4">
  <Objects Count="1">
    <Object type="Part::Feature" name="Box"/>
  </Objects>
  <ObjectData Count="1">
    <Object name="Box">
      <Properties Count="3">
        <Property name="Label" type="App::PropertyString">
          <String value="MyBox"/>
        </Property>
        <Property name="Shape" type="Part::PropertyPartShape">
          <Part file="PartShape.brp"/>
        </Property>
        <Property name="ShapeColor" type="App::PropertyColor">
          <PropertyColor>
            <PropertyColor value="2147483647"/>
          </PropertyColor>
        </Property>
      </Properties>
    </Object>
  </ObjectData>
</Document>
"""


def test_parse_document_xml_extracts_label_and_brep_filename():
    out = _parse_document_xml(_DOC_XML.encode())
    assert 'Box' in out
    assert out['Box']['label'] == 'MyBox'
    assert out['Box']['brep_file'] == 'PartShape.brp'


def test_read_fcstd_round_trip(tmp_path: Path):
    pytest.importorskip('OCP', exc_type=ImportError)
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools

    shape = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
    brep_path = tmp_path / 'PartShape.brp'
    BRepTools.Write_s(shape, str(brep_path))

    fcstd = tmp_path / 'doc.FCStd'
    with zipfile.ZipFile(fcstd, 'w') as zf:
        zf.writestr('Document.xml', _DOC_XML)
        zf.write(brep_path, 'PartShape.brp')

    mb = pyvista_cad.read_fcstd(fcstd)
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 1
    block = mb[0]
    assert block.n_points > 0
    assert str(block.field_data['cad.label'][0]) == 'MyBox'
    assert str(block.field_data['cad.source_format'][0]) == 'fcstd'


def test_read_fcstd_via_pv_read(tmp_path: Path):
    pytest.importorskip('OCP', exc_type=ImportError)
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools

    shape = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    brep_path = tmp_path / 'PartShape.brp'
    BRepTools.Write_s(shape, str(brep_path))

    fcstd = tmp_path / 'doc.FCStd'
    with zipfile.ZipFile(fcstd, 'w') as zf:
        zf.writestr('Document.xml', _DOC_XML)
        zf.write(brep_path, 'PartShape.brp')

    mb = pv.read(str(fcstd))
    assert isinstance(mb, pv.MultiBlock)
    assert mb.n_blocks == 1
