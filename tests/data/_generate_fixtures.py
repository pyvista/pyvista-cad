"""Deterministic, in-process generator for the pyvista-cad test fixture corpus.

Run with::

    uv run python tests/data/_generate_fixtures.py

Every fixture is written by a third-party CAD library (OCP/OCCT, ezdxf,
lib3mf, ifcopenshell) and never by pyvista-cad's own writers, so the
reader tests exercise genuine externally produced files. All dimensions,
seeds, and GUIDs are fixed. Re-running produces byte-identical output for
the BREP, DXF, and 3MF fixtures. STEP, IGES, and IFC embed a creation
timestamp in their header; this generator rewrites that single header
line to a fixed value so re-runs are byte-stable too.

Fixtures
--------

``assembly_two_parts.step``
    A real STEP assembly written with ``STEPCAFControl_Writer`` and an
    OCAF document. One product per component:

    - ``Base``: a box 30 x 20 x 10 mm, built at the origin and added
      with an identity component transform. It reads back with bounds
      ``(0, 30, 0, 20, 0, 10)`` and 12 triangle cells.
    - ``Pin``: a cylinder radius 4 mm, height 25 mm, axis +Z. It is
      added with a component transform translating it by
      ``(15, 10, 10)``; the reader applies that component transform
      exactly once and reports Pin bounds
      ``(11.0, 19.0, 6.029165, 13.970835, 10.0, 35.0)`` and 100 cells
      (these exact numbers are what the tessellated re-read produces;
      tests assert them directly).

    Two named products (``Base``, ``Pin``) tied by a
    NEXT_ASSEMBLY_USAGE_OCCURRENCE hierarchy; the read result is a
    MultiBlock with two blocks named ``Base`` and ``Pin``. Combined
    assembly bounds as read: ``(0.0, 30.0, 0.0, 20.0, 0.0,
    35.0)``. Units: mm.

``dxf_r12.dxf`` / ``dxf_r2018.dxf``
    ezdxf documents in the R12 and R2018 DXF versions. Each carries, on
    named layers:

    - a ``LINE`` from ``(0, 0)`` to ``(100, 0)`` on layer ``GEOM``,
    - a closed square ``(0, 0)`` -> ``(50, 0)`` -> ``(50, 50)`` ->
      ``(0, 50)`` on layer ``GEOM`` (a ``POLYLINE`` in R12, an
      ``LWPOLYLINE`` in R2018, since R12 predates LWPOLYLINE),
    - a ``CIRCLE`` centre ``(75, 25)`` radius 10 on layer ``DETAIL``,
    - one ``INSERT`` of block ``MARKER`` (a 4-unit cross) at
      ``(75, 25)`` on layer ``DETAIL``.

    ``$INSUNITS`` is set to 4 (mm) and is stored in R2018; the R12 DXF
    format does not persist ``$INSUNITS``, so it reads back as unset
    there. ``$TDCREATE``, ``$TDUPDATE``, ``$TDINDWG``, and
    ``$TDUSRTIMER`` are pinned to fixed values so the bytes are stable.

``multimaterial.3mf``
    A lib3mf model with two mesh objects:

    - ``CubeA``: a cube edge 10 mm, min corner ``(0, 0, 0)``, bounds
      ``(0, 10, 0, 10, 0, 10)``, 12 cells. Material ``RedMat`` RGBA
      ``(255, 0, 0, 255)``.
    - ``CubeB``: a cube edge 10 mm, min corner ``(20, 0, 0)``, bounds
      ``(20, 30, 0, 10, 0, 10)``, 12 cells. Material ``BlueMat`` RGBA
      ``(0, 0, 255, 255)``.

    Colours are carried by a BaseMaterialGroup, one material per
    object, assigned as the object-level property and verifiable
    directly through lib3mf. Model unit is millimetre; the read result
    is a MultiBlock with blocks ``CubeA`` and ``CubeB``, combined
    bounds ``(0, 30, 0, 10, 0, 10)``. Note: the current 3MF reader does
    not surface the BaseMaterialGroup colour as ``cad.color``; assert
    the colours against lib3mf, not ``cad.color``.

``building.ifc``
    An IFC4 file: ``IfcProject`` -> ``IfcSite`` -> ``IfcBuilding`` ->
    ``IfcBuildingStorey`` containing one extruded ``IfcWall``. The wall
    carries a diffuse ``IfcSurfaceStyleRendering`` / ``IfcStyledItem``
    with ``IfcColourRgb`` ``(0.8, 0.3, 0.2)``; ifcopenshell's geometry
    iterator reports this as the wall material diffuse colour. The wall
    footprint is 4.0 x 0.2 m extruded 3.0 m high, placed at the origin,
    so the read bounds are ``(0.0, 4.0, 0.0, 0.2, 0.0, 3.0)`` in
    metres. The read result is a nested MultiBlock
    (``Site`` -> wall). GUIDs are deterministic; the wall GlobalId is
    ``0eCpv9L5zgTO2Bbg6ijyBD``. The FILE_NAME header timestamp and the
    IFCUNITASSIGNMENT member order are normalised for byte stability.
    Note: the authored colour is present in the file (assert via
    ifcopenshell ``IfcColourRgb``); the current IFC reader has a bug
    handling the ``colour`` object so it may not appear as
    ``cad.color``.

``impeller.iges``
    A B-spline surface hub (radius 12 mm, height 20 mm, swept a full
    turn) plus three B-spline blade surfaces polar-arrayed at 0, 120,
    240 degrees, written with ``IGESControl_Writer`` as rational
    B-spline surface entities (IGES type 128). pyiges tessellates these
    via the default reader path (``surfaces=True``, ``bsplines=True``).
    The read result is a single PolyData with 12168 cells, 6400 points,
    and bounds
    ``(-25.246551, 28.0, -28.000005, 24.248711, 0.0, 20.0)``.

``external.brep``
    A build123d box 25 x 15 x 8 mm written via ``BRepTools.Write_s``.
    build123d ``Box`` is centred on the origin, so the bounds are
    ``(-12.5, 12.5, -7.5, 7.5, -4.0, 4.0)``. OCCT ASCII BREP carries no
    timestamp, so the file is byte-identical on every run.

Recorded expectations
---------------------

The exact bounds, cell counts, names, and colours that the reader tests
hardcode are printed by the verification block at the bottom of this
module and recorded in ``tests/data/README.md``. Regenerate both with
the command above.
"""

import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# A fixed timestamp used everywhere a header date must be normalised.
FIXED_ISO = '2020-01-01T00:00:00'
# Julian day number for 2020-01-01 00:00:00 UTC, as ezdxf stores dates.
FIXED_JULIAN = 2458849.5


def _step_path() -> str:
    return os.path.join(HERE, 'assembly_two_parts.step')


def _normalize_step_timestamp(path: str) -> None:
    """Pin the FILE_NAME timestamp in a STEP header for byte stability."""
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    # STEP FILE_NAME has the form: FILE_NAME('name','TIMESTAMP',...)
    text = re.sub(
        r"(FILE_NAME\('[^']*',')[^']*(',)",
        r'\g<1>' + FIXED_ISO + r'\g<2>',
        text,
        count=1,
    )
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)


def generate_step_assembly() -> str:
    """Write a two-product STEP assembly via STEPCAFControl_Writer + OCAF."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    path = _step_path()

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString('XmlXCAF'))
    app.InitDocument(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

    base_shape = BRepPrimAPI_MakeBox(30.0, 20.0, 10.0).Shape()
    pin_shape = BRepPrimAPI_MakeCylinder(4.0, 25.0).Shape()

    base_lbl = shape_tool.AddShape(base_shape, False, True)
    pin_lbl = shape_tool.AddShape(pin_shape, False, True)
    TDataStd_Name.Set_s(base_lbl, TCollection_ExtendedString('Base'))
    TDataStd_Name.Set_s(pin_lbl, TCollection_ExtendedString('Pin'))

    # Assembly root that owns the two components.
    root_lbl = shape_tool.NewShape()
    TDataStd_Name.Set_s(root_lbl, TCollection_ExtendedString('Assembly'))

    base_trsf = gp_Trsf()
    shape_tool.AddComponent(root_lbl, base_lbl, TopLoc_Location(base_trsf))

    pin_trsf = gp_Trsf()
    pin_trsf.SetTranslation(gp_Vec(15.0, 10.0, 10.0))
    shape_tool.AddComponent(root_lbl, pin_lbl, TopLoc_Location(pin_trsf))

    shape_tool.UpdateAssemblies()

    Interface_Static.SetCVal_s('write.step.schema', 'AP214')
    Interface_Static.SetCVal_s('write.step.unit', 'MM')

    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    status = writer.Write(path)
    if status != IFSelect_RetDone:
        msg = f'STEPCAFControl_Writer.Write failed with status {status}'
        raise RuntimeError(msg)

    _normalize_step_timestamp(path)

    # Verify two named products survive a re-read.
    _verify_step_products(path)
    return path


def _verify_step_products(path: str) -> None:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString('XmlXCAF'))
    app.InitDocument(doc)
    reader = STEPCAFControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        msg = 'STEPCAFControl_Reader.ReadFile failed'
        raise RuntimeError(msg)
    reader.Transfer(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    shape_tool.GetShapes(labels)
    names: set[str] = set()
    for i in range(1, labels.Length() + 1):
        lbl = labels.Value(i)
        attr = TDataStd_Name()
        if lbl.FindAttribute(TDataStd_Name.GetID_s(), attr):
            names.add(attr.Get().ToExtString())
    if not {'Base', 'Pin'}.issubset(names):
        msg = f'STEP assembly missing named products; found {sorted(names)}'
        raise RuntimeError(msg)


def _set_dxf_dates(doc) -> None:
    doc.header['$TDCREATE'] = FIXED_JULIAN
    doc.header['$TDUPDATE'] = FIXED_JULIAN
    doc.header['$TDINDWG'] = 0.0
    doc.header['$INSUNITS'] = 4  # millimetres


def _build_dxf(version: str, path: str) -> None:
    import ezdxf

    # Pins ezdxf's GUIDs and metadata to constant values on write.
    ezdxf.options.write_fixed_meta_data_for_testing = True
    doc = ezdxf.new(version, setup=False)
    doc.layers.add('GEOM', color=1)
    doc.layers.add('DETAIL', color=3)

    block = doc.blocks.new(name='MARKER')
    block.add_line((-2, 0), (2, 0))
    block.add_line((0, -2), (0, 2))

    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0), dxfattribs={'layer': 'GEOM'})
    square = [(0, 0), (50, 0), (50, 50), (0, 50)]
    if version == 'R12':
        # R12 predates LWPOLYLINE; use the classic POLYLINE entity.
        msp.add_polyline2d(square, close=True, dxfattribs={'layer': 'GEOM'})
    else:
        msp.add_lwpolyline(square, close=True, dxfattribs={'layer': 'GEOM'})
    msp.add_circle((75, 25), 10, dxfattribs={'layer': 'DETAIL'})
    msp.add_blockref('MARKER', (75, 25), dxfattribs={'layer': 'DETAIL'})

    # Pin the document GUIDs so R2018 does not stamp random ones.
    doc.header['$FINGERPRINTGUID'] = '{00000000-0000-4000-8000-000000000001}'
    doc.header['$VERSIONGUID'] = '{00000000-0000-4000-8000-000000000002}'
    _set_dxf_dates(doc)
    doc.saveas(path)
    _pin_dxf_dates(path)


def _pin_dxf_dates(path: str) -> None:
    """Pin every header date/timer value ezdxf rewrites on save.

    ezdxf stamps ``$TDUPDATE``/``$TDUSRTIMER``/``$TDINDWG`` with the
    real save time regardless of what we set, so rewrite those Julian
    values to fixed numbers for byte stability.
    """
    fixed = {
        '$TDCREATE': repr(FIXED_JULIAN),
        '$TDUPDATE': repr(FIXED_JULIAN),
        '$TDINDWG': '0.0',
        '$TDUSRTIMER': '0.0',
    }
    with open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    out = list(lines)
    for i, line in enumerate(lines):
        name = line.strip()
        if name in fixed and i + 2 < len(lines) and lines[i + 1].strip() == '40':
            out[i + 2] = fixed[name]
        # ezdxf stamps "CREATED_BY_EZDXF" / "WRITTEN_BY_EZDXF" AppData
        # with "<version> @ <wall-clock timestamp>"; pin the value line.
        if name in ('CREATED_BY_EZDXF', 'WRITTEN_BY_EZDXF') and i + 2 < len(lines):
            out[i + 2] = 'pyvista-cad-fixture @ ' + FIXED_ISO

    out = _sort_dxf_classes(out)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out))


def _sort_dxf_classes(lines: list[str]) -> list[str]:
    """Sort CLASS records in the CLASSES section by class name.

    The R2018 CLASSES section lists registered classes (for example
    ``LAYOUT`` and ``ACDBPLACEHOLDER``) in an unstable set-iteration
    order. Reorder the records by their DXF class name (group code 1).
    """
    n = len(lines)
    # Find the bounds of the CLASSES section payload.
    start = end = None
    for i in range(n - 1):
        if lines[i].strip() == 'SECTION' and lines[i + 2].strip() == 'CLASSES':
            start = i + 3
            break
    if start is None:
        return lines
    for j in range(start, n - 1):
        if lines[j].strip() == '0' and lines[j + 1].strip() == 'ENDSEC':
            end = j
            break
    if end is None:
        return lines

    # Collect the index of every "0 / CLASS" record start. A DXF value
    # line may itself be "0", so only a 0 group code immediately
    # followed by the CLASS keyword begins a new record.
    starts: list[int] = []
    k = start
    while k < end - 1:
        if lines[k].strip() == '0' and lines[k + 1].strip() == 'CLASS':
            starts.append(k)
        k += 1
    if not starts:
        return lines

    records: list[tuple[str, list[str]]] = []
    for idx, blk_start in enumerate(starts):
        blk_end = starts[idx + 1] if idx + 1 < len(starts) else end
        block = lines[blk_start:blk_end]
        cls_name = ''
        # The class record name is the value of group code 1.
        for m in range(len(block) - 1):
            if block[m].strip() == '1':
                cls_name = block[m + 1].strip()
                break
        records.append((cls_name, block))
    records.sort(key=lambda r: r[0])
    reordered: list[str] = []
    for _, block in records:
        reordered.extend(block)
    return lines[:start] + reordered + lines[end:]


def generate_dxf() -> tuple[str, str]:
    r12 = os.path.join(HERE, 'dxf_r12.dxf')
    r2018 = os.path.join(HERE, 'dxf_r2018.dxf')
    _build_dxf('R12', r12)
    _build_dxf('R2018', r2018)
    return r12, r2018


def generate_multimaterial_3mf() -> str:
    import lib3mf

    path = os.path.join(HERE, 'multimaterial.3mf')
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    model.SetUnit(lib3mf.ModelUnit.MilliMeter)

    def cube(origin):
        ox, oy, oz = origin
        verts = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 10],
                [10, 0, 10],
                [10, 10, 10],
                [0, 10, 10],
            ],
            dtype=float,
        ) + np.array([ox, oy, oz], dtype=float)
        tris = [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ]
        return verts, tris

    base_group = model.AddBaseMaterialGroup()
    red = base_group.AddMaterial('RedMat', wrapper.RGBAToColor(255, 0, 0, 255))
    blue = base_group.AddMaterial('BlueMat', wrapper.RGBAToColor(0, 0, 255, 255))

    for name, origin, group_prop in (
        ('CubeA', (0.0, 0.0, 0.0), red),
        ('CubeB', (20.0, 0.0, 0.0), blue),
    ):
        verts, tris = cube(origin)
        obj = model.AddMeshObject()
        obj.SetName(name)
        pos = []
        for v in verts:
            p = lib3mf.Position()
            p.Coordinates = (float(v[0]), float(v[1]), float(v[2]))
            pos.append(p)
        tri_structs = []
        for t in tris:
            tri = lib3mf.Triangle()
            tri.Indices = (int(t[0]), int(t[1]), int(t[2]))
            tri_structs.append(tri)
        obj.SetGeometry(pos, tri_structs)
        obj.SetObjectLevelProperty(base_group.GetResourceID(), group_prop)
        model.AddBuildItem(obj, wrapper.GetIdentityTransform())

    writer = model.QueryWriter('3mf')
    writer.WriteToFile(path)
    _normalize_3mf(path)
    return path


def _normalize_3mf(path: str) -> None:
    """Repack the 3MF with deterministic UUIDs and zip metadata.

    lib3mf assigns a fresh random ``UUID`` to the model and every build
    item on each write. Rewrite them to a fixed sequence and rebuild the
    archive with constant timestamps and ordering so the bytes are
    stable.
    """
    import io
    import zipfile

    with zipfile.ZipFile(path, 'r') as zf:
        members: list[tuple[str, bytes]] = [
            (name, zf.read(name)) for name in sorted(zf.namelist())
        ]

    counter = {'n': 0}

    def _sub(match: 're.Match[str]') -> str:
        counter['n'] += 1
        n = counter['n']
        det = f'00000000-0000-4000-8000-{n:012d}'
        return f'{match.group(1)}"{det}"'

    fixed: list[tuple[str, bytes]] = []
    for name, data in members:
        if name.endswith(('.model', '.xml', '.rels')):
            text = data.decode('utf-8')
            text = re.sub(r'([Uu][Uu][Ii][Dd]=)"[^"]*"', _sub, text)
            data = text.encode('utf-8')
        fixed.append((name, data))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in fixed:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, data)
    with open(path, 'wb') as fh:
        fh.write(buf.getvalue())


def _normalize_ifc_timestamp(path: str) -> None:
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    # FILE_NAME('name','TIMESTAMP',(...),...)
    text = re.sub(
        r"(FILE_NAME\('[^']*',')[^']*(',)",
        r'\g<1>' + FIXED_ISO + r'\g<2>',
        text,
        count=1,
    )

    # IFCUNITASSIGNMENT lists its members in set-iteration order, which
    # varies between runs. Sort the referenced ids numerically.
    def _sort_units(match: 're.Match[str]') -> str:
        ids = re.findall(r'#(\d+)', match.group(1))
        ordered = ','.join('#' + i for i in sorted(ids, key=int))
        return f'IFCUNITASSIGNMENT(({ordered}))'

    text = re.sub(r'IFCUNITASSIGNMENT\(\(([^)]*)\)\)', _sort_units, text)

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)


def generate_building_ifc() -> str:
    import ifcopenshell
    import ifcopenshell.guid

    path = os.path.join(HERE, 'building.ifc')

    # ifcopenshell.api assigns a fresh random GUID to every entity it
    # creates, including internal relationship objects. Patch the GUID
    # factory to a deterministic sequential generator for the whole
    # build so the file is byte-stable.
    counter = {'n': 0}
    real_new = ifcopenshell.guid.new

    def deterministic_new() -> str:
        counter['n'] += 1
        n = counter['n']
        raw = bytes((n * 37 + i * 11) % 256 for i in range(16))
        return ifcopenshell.guid.compress(raw.hex())

    ifcopenshell.guid.new = deterministic_new
    try:
        return _build_ifc(path)
    finally:
        ifcopenshell.guid.new = real_new


def _build_ifc(path: str) -> str:
    import ifcopenshell
    import ifcopenshell.api.aggregate
    import ifcopenshell.api.context
    import ifcopenshell.api.root
    import ifcopenshell.api.spatial
    import ifcopenshell.api.unit

    model = ifcopenshell.file(schema='IFC4')

    project = ifcopenshell.api.root.create_entity(
        model, ifc_class='IfcProject', name='FixtureProject'
    )
    ifcopenshell.api.unit.assign_unit(
        model,
        length={'is_metric': True, 'raw': 'METERS'},
    )

    ctx = ifcopenshell.api.context.add_context(model, context_type='Model')
    body = ifcopenshell.api.context.add_context(
        model,
        context_type='Model',
        context_identifier='Body',
        target_view='MODEL_VIEW',
        parent=ctx,
    )

    site = ifcopenshell.api.root.create_entity(model, ifc_class='IfcSite', name='Site')
    building = ifcopenshell.api.root.create_entity(model, ifc_class='IfcBuilding', name='Building')
    storey = ifcopenshell.api.root.create_entity(
        model, ifc_class='IfcBuildingStorey', name='Storey'
    )

    ifcopenshell.api.aggregate.assign_object(model, products=[site], relating_object=project)
    ifcopenshell.api.aggregate.assign_object(model, products=[building], relating_object=site)
    ifcopenshell.api.aggregate.assign_object(model, products=[storey], relating_object=building)

    wall = ifcopenshell.api.root.create_entity(model, ifc_class='IfcWall', name='Wall')
    ifcopenshell.api.spatial.assign_container(model, relating_structure=storey, products=[wall])

    # Extruded box: 4.0 (x) by 0.2 (y) footprint, 3.0 (z) high.
    profile = model.create_entity(
        'IfcRectangleProfileDef',
        ProfileType='AREA',
        XDim=4.0,
        YDim=0.2,
    )
    placement_pt = model.create_entity('IfcCartesianPoint', Coordinates=(2.0, 0.1, 0.0))
    axis2d = model.create_entity(
        'IfcAxis2Placement2D',
        Location=model.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0)),
    )
    profile.Position = axis2d
    direction = model.create_entity('IfcDirection', DirectionRatios=(0.0, 0.0, 1.0))
    position = model.create_entity(
        'IfcAxis2Placement3D',
        Location=placement_pt,
    )
    solid = model.create_entity(
        'IfcExtrudedAreaSolid',
        SweptArea=profile,
        Position=position,
        ExtrudedDirection=direction,
        Depth=3.0,
    )
    shape_rep = model.create_entity(
        'IfcShapeRepresentation',
        ContextOfItems=body,
        RepresentationIdentifier='Body',
        RepresentationType='SweptSolid',
        Items=[solid],
    )
    product_def = model.create_entity('IfcProductDefinitionShape', Representations=[shape_rep])
    wall.Representation = product_def
    origin = model.create_entity('IfcCartesianPoint', Coordinates=(0.0, 0.0, 0.0))
    wall.ObjectPlacement = model.create_entity(
        'IfcLocalPlacement',
        RelativePlacement=model.create_entity('IfcAxis2Placement3D', Location=origin),
    )

    # Diffuse surface colour carried by an IfcStyledItem on the solid.
    rgb = model.create_entity('IfcColourRgb', Red=0.8, Green=0.3, Blue=0.2)
    rendering = model.create_entity(
        'IfcSurfaceStyleRendering',
        SurfaceColour=rgb,
        ReflectanceMethod='NOTDEFINED',
    )
    surface_style = model.create_entity(
        'IfcSurfaceStyle',
        Name='WallColour',
        Side='BOTH',
        Styles=[rendering],
    )
    presentation_style = model.create_entity(
        'IfcPresentationStyleAssignment', Styles=[surface_style]
    )
    model.create_entity(
        'IfcStyledItem',
        Item=solid,
        Styles=[presentation_style],
    )

    model.write(path)
    _normalize_ifc_timestamp(path)
    return path


def generate_impeller_iges() -> str:
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt
    from OCP.IGESControl import IGESControl_Writer
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TopoDS import TopoDS_Compound

    path = os.path.join(HERE, 'impeller.iges')

    # pyiges, the reader backend, tessellates rational B-spline surface
    # entities (IGES type 128) but not solid/manifold BREP (type 186).
    # Build the impeller from B-spline surface faces fitted to fixed
    # control nets so the fixture re-reads through the default reader
    # path (surfaces=True, bsplines=True) as real curved geometry.
    nu, nv = 16, 8

    def _bspline_face(point_fn):
        arr = TColgp_Array2OfPnt(1, nu, 1, nv)
        for i in range(1, nu + 1):
            for j in range(1, nv + 1):
                x, y, z = point_fn((i - 1) / (nu - 1), (j - 1) / (nv - 1))
                arr.SetValue(i, j, gp_Pnt(x, y, z))
        surf = GeomAPI_PointsToBSplineSurface(arr).Surface()
        return BRepBuilderAPI_MakeFace(surf, 1.0e-6).Face()

    def _hub(u, v):
        # Revolved hub: radius 12 mm, height 20 mm, swept full turn.
        ang = u * 2.0 * math.pi
        return (12.0 * math.cos(ang), 12.0 * math.sin(ang), v * 20.0)

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, _bspline_face(_hub))

    def _blade_factory(phase):
        def _blade(u, v):
            # A twisted blade sheet anchored to the hub, radius 12 -> 28.
            radius = 12.0 + 16.0 * u
            twist = phase + 0.6 * v
            return (
                radius * math.cos(twist),
                radius * math.sin(twist),
                2.0 + 16.0 * v,
            )

        return _blade

    for angle_deg in (0.0, 120.0, 240.0):
        builder.Add(compound, _bspline_face(_blade_factory(math.radians(angle_deg))))

    writer = IGESControl_Writer()
    writer.AddShape(compound)
    writer.ComputeModel()
    if not writer.Write(path):
        msg = 'IGESControl_Writer.Write failed'
        raise RuntimeError(msg)

    _normalize_iges_timestamp(path)
    return path


def _normalize_iges_timestamp(path: str) -> None:
    """Pin the global-section date fields of an IGES file for stability."""
    with open(path, encoding='ascii') as fh:
        lines = fh.readlines()
    # IGES dates look like 15HYYYYMMDD.HHMMSS. Replace every occurrence.
    fixed = '20200101.000000'
    out = [re.sub(r'\d{8}\.\d{6}', fixed, line) for line in lines]
    with open(path, 'w', encoding='ascii') as fh:
        fh.writelines(out)


def generate_external_brep() -> str:
    import build123d as b3d
    from OCP.BRepTools import BRepTools

    path = os.path.join(HERE, 'external.brep')
    with b3d.BuildPart() as part:
        b3d.Box(25.0, 15.0, 8.0)
    BRepTools.Write_s(part.part.wrapped, path)
    return path


def _read_with_pyvista_cad(path: str):
    """Re-read a fixture through the matching pyvista_cad reader."""
    import pyvista_cad

    ext = os.path.splitext(path)[1].lower()
    if ext == '.step':
        return pyvista_cad.read_step(path)
    if ext == '.dxf':
        return pyvista_cad.read_dxf(path)
    if ext == '.3mf':
        return pyvista_cad.read_three_mf(path)
    if ext == '.ifc':
        return pyvista_cad.read_ifc(path)
    if ext == '.iges':
        return pyvista_cad.read_iges(path)
    if ext == '.brep':
        return pyvista_cad.read_brep(path)
    msg = f'no pyvista_cad reader mapped for {ext}'
    raise ValueError(msg)


def _record(label: str, path: str) -> None:
    import pyvista as pv

    size = os.path.getsize(path)
    ds = _read_with_pyvista_cad(path)
    print(f'\n=== {label}: {os.path.basename(path)} ({size} bytes) ===')

    def _fmt_bounds(b):
        return tuple(round(float(x), 6) for x in b)

    if isinstance(ds, pv.MultiBlock):
        print(f'MultiBlock n_blocks={ds.n_blocks}')
        print(f'combined bounds={_fmt_bounds(ds.bounds)}')
        for i in range(ds.n_blocks):
            blk = ds[i]
            name = ds.get_block_name(i)
            if blk is None:
                print(f'  [{i}] {name}: None')
                continue
            color = None
            if 'cad.color' in blk.field_data:
                color = tuple(
                    round(float(x), 4) for x in np.asarray(blk.field_data['cad.color']).ravel()
                )
            label_fd = None
            if 'cad.label' in blk.field_data:
                label_fd = str(np.asarray(blk.field_data['cad.label']).ravel()[0])
            if isinstance(blk, pv.MultiBlock):
                n_cells = sum(b.n_cells for b in blk if b is not None and hasattr(b, 'n_cells'))
                kind = f'MultiBlock(n_blocks={blk.n_blocks})'
            else:
                n_cells = blk.n_cells
                kind = 'PolyData'
            print(
                f'  [{i}] block_name={name!r} cad.label={label_fd!r} {kind} '
                f'n_cells={n_cells} bounds={_fmt_bounds(blk.bounds)} '
                f'cad.color={color}'
            )
    else:
        color = None
        if 'cad.color' in ds.field_data:
            color = tuple(
                round(float(x), 4) for x in np.asarray(ds.field_data['cad.color']).ravel()
            )
        print(
            f'PolyData n_cells={ds.n_cells} n_points={ds.n_points} '
            f'bounds={_fmt_bounds(ds.bounds)} cad.color={color}'
        )
    fd = {}
    for key in ('cad.source_format', 'cad.backend', 'cad.units', 'cad.reader'):
        if key in ds.field_data:
            fd[key] = str(np.asarray(ds.field_data[key]).ravel()[0])
    if fd:
        print(f'  field_data={fd}')


if __name__ == '__main__':
    step = generate_step_assembly()
    r12, r2018 = generate_dxf()
    mf = generate_multimaterial_3mf()
    ifc = generate_building_ifc()
    iges = generate_impeller_iges()
    brep = generate_external_brep()

    print('Generated 6 fixtures into', HERE)
    _record('STEP assembly', step)
    _record('DXF R12', r12)
    _record('DXF R2018', r2018)
    _record('3MF multimaterial', mf)
    _record('IFC building', ifc)
    _record('IGES impeller', iges)
    _record('BREP external', brep)
