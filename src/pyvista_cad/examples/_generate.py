"""Regenerate bundled example CAD fixtures.

Run once with::

    uv run python -m pyvista_cad.examples._generate

The committed fixtures are produced from the canonical generators
below so that they remain reproducible.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _generate_dxf_plate(path: Path) -> None:
    import ezdxf

    doc = ezdxf.new(dxfversion='R2018')
    doc.header['$INSUNITS'] = 4  # mm
    msp = doc.modelspace()
    doc.layers.add('OUTLINE', color=7)
    doc.layers.add('HOLES', color=1)

    outline = [(0, 0, 0), (60, 0, 0), (60, 40, 0), (0, 40, 0), (0, 0, 0)]
    msp.add_lwpolyline([(p[0], p[1]) for p in outline], dxfattribs={'layer': 'OUTLINE'})
    msp.add_circle((15, 20, 0), 5, dxfattribs={'layer': 'HOLES'})
    msp.add_circle((45, 20, 0), 5, dxfattribs={'layer': 'HOLES'})
    msp.add_3dface(
        [(10, 5, 0), (50, 5, 0), (50, 35, 0), (10, 35, 0)],
        dxfattribs={'layer': 'OUTLINE'},
    )

    doc.saveas(str(path))


def _generate_step_cube(path: Path) -> None:
    try:
        import build123d as b3d
    except ImportError as exc:  # pragma: no cover - optional build123d absent in CI/minimal envs
        msg = 'generating the STEP fixture requires build123d; install pyvista-cad[step].'
        raise RuntimeError(msg) from exc

    cube = b3d.Box(10, 10, 10)
    b3d.export_step(cube, str(path))


def _generate_bracket_step(path: Path) -> None:
    """Generate a parametric L-bracket STEP file.

    Geometry: 50x30x10 mm L-bracket with two M6 through-holes and a 3 mm
    fillet at the inside corner.
    """
    try:
        import build123d as b3d
    except ImportError as exc:  # pragma: no cover - optional build123d absent in CI/minimal envs
        msg = 'generating the bracket STEP fixture requires build123d; install pyvista-cad[step].'
        raise RuntimeError(msg) from exc

    # Build the L profile as a 2D sketch then extrude.
    length = 50.0
    height = 30.0
    thickness = 10.0
    fillet_r = 3.0
    hole_r = 6.0 / 2.0  # M6 through-hole, nominal 6 mm

    with b3d.BuildPart() as bracket:
        with b3d.BuildSketch(b3d.Plane.XZ):
            with b3d.BuildLine():
                b3d.Polyline(
                    (0, 0),
                    (length, 0),
                    (length, thickness),
                    (thickness, thickness),
                    (thickness, height),
                    (0, height),
                    close=True,
                )
            b3d.make_face()
        b3d.extrude(amount=thickness)
        # Fillet the inside concave corner edge running along Y.
        concave = [
            e
            for e in bracket.edges()
            if abs(e.length - thickness) < 1e-6
            and abs(e.center().X - thickness) < 1e-6
            and abs(e.center().Z - thickness) < 1e-6
        ]
        if concave:
            b3d.fillet(concave, radius=fillet_r)
        # Two M6 through-holes through the horizontal flange. Drill from
        # the top face down through the +Z thickness band.
        top_face = bracket.faces().sort_by(b3d.Axis.Z).last
        with b3d.BuildSketch(top_face) as hole_sketch:
            with b3d.Locations(
                (length - 10.0, thickness / 2.0),
                (length - 30.0, thickness / 2.0),
            ):
                b3d.Circle(radius=hole_r)
            _ = hole_sketch
        b3d.extrude(amount=-thickness, mode=b3d.Mode.SUBTRACT)

    b3d.export_step(bracket.part, str(path))


def _generate_drawing_dxf(path: Path) -> None:
    """Generate a 2D mechanical drawing DXF with layers and annotations."""
    import ezdxf

    doc = ezdxf.new(dxfversion='R2018', setup=True)
    doc.header['$INSUNITS'] = 4  # mm
    msp = doc.modelspace()

    # Layers: BODY, HOLES, CENTERLINES (dashed), DIMENSIONS
    doc.layers.add('BODY', color=7)
    doc.layers.add('HOLES', color=1)
    if 'DASHED' in doc.linetypes:
        doc.layers.add('CENTERLINES', color=3, linetype='DASHED')
    else:
        doc.layers.add('CENTERLINES', color=3)
    doc.layers.add('DIMENSIONS', color=5)

    # Bracket body outline (rounded rectangle 100x60 with 5mm corner notches).
    body = [
        (0, 0),
        (100, 0),
        (100, 60),
        (0, 60),
        (0, 0),
    ]
    msp.add_lwpolyline(body, close=True, dxfattribs={'layer': 'BODY'})

    # Four mounting holes (diameter 6 mm) at 10 mm inset from each corner.
    hole_centers = [(10, 10), (90, 10), (90, 50), (10, 50)]
    for cx, cy in hole_centers:
        msp.add_circle((cx, cy, 0), 3.0, dxfattribs={'layer': 'HOLES'})

    # Centerlines (dashed) through each hole - horizontal and vertical
    for cx, cy in hole_centers:
        msp.add_line((cx - 6, cy), (cx + 6, cy), dxfattribs={'layer': 'CENTERLINES'})
        msp.add_line((cx, cy - 6), (cx, cy + 6), dxfattribs={'layer': 'CENTERLINES'})

    # Three text annotations.
    msp.add_text(
        'BRACKET v1.0',
        height=4.0,
        dxfattribs={'layer': 'DIMENSIONS'},
    ).set_placement((30, 70))
    msp.add_text(
        '4x DIA 6 THRU',
        height=3.0,
        dxfattribs={'layer': 'DIMENSIONS'},
    ).set_placement((30, -8))
    msp.add_text(
        'MATL: AL 6061-T6',
        height=2.5,
        dxfattribs={'layer': 'DIMENSIONS'},
    ).set_placement((60, -8))

    doc.saveas(str(path))


def _generate_assembly_3mf(path: Path) -> None:
    """Generate a two-part 3MF assembly: red base plate + 4 blue standoffs.

    Uses lib3mf to build the model with per-object colors and per-build-item
    transforms (each standoff has a unique translation).
    """
    import lib3mf

    w = lib3mf.get_wrapper()
    model = w.CreateModel()
    model.SetUnit(1)  # mm

    def _box_geometry(sx: float, sy: float, sz: float) -> tuple[list, list]:
        # 8 vertices of an axis-aligned box from origin to (sx, sy, sz).
        verts = [
            (0, 0, 0),
            (sx, 0, 0),
            (sx, sy, 0),
            (0, sy, 0),
            (0, 0, sz),
            (sx, 0, sz),
            (sx, sy, sz),
            (0, sy, sz),
        ]
        # 12 triangles (CCW outward).
        tri_idx = [
            (0, 2, 1),
            (0, 3, 2),  # bottom (-Z)
            (4, 5, 6),
            (4, 6, 7),  # top (+Z)
            (0, 1, 5),
            (0, 5, 4),  # -Y
            (1, 2, 6),
            (1, 6, 5),  # +X
            (2, 3, 7),
            (2, 7, 6),  # +Y
            (3, 0, 4),
            (3, 4, 7),  # -X
        ]
        positions = []
        for x, y, z in verts:
            p = lib3mf.Position()
            p.Coordinates = (float(x), float(y), float(z))
            positions.append(p)
        triangles = []
        for a, b, c in tri_idx:
            t = lib3mf.Triangle()
            t.Indices = (int(a), int(b), int(c))
            triangles.append(t)
        return positions, triangles

    def _identity() -> object:
        t = lib3mf.Transform()
        t.Fields[0][:] = [1.0, 0.0, 0.0]
        t.Fields[1][:] = [0.0, 1.0, 0.0]
        t.Fields[2][:] = [0.0, 0.0, 1.0]
        t.Fields[3][:] = [0.0, 0.0, 0.0]
        return t

    def _translation(dx: float, dy: float, dz: float) -> object:
        t = lib3mf.Transform()
        t.Fields[0][:] = [1.0, 0.0, 0.0]
        t.Fields[1][:] = [0.0, 1.0, 0.0]
        t.Fields[2][:] = [0.0, 0.0, 1.0]
        t.Fields[3][:] = [float(dx), float(dy), float(dz)]
        return t

    def _make_color(r: int, g: int, b: int, a: int = 255) -> object:
        c = lib3mf.Color()
        c.Red = r
        c.Green = g
        c.Blue = b
        c.Alpha = a
        return c

    # --- Base plate: 50 x 50 x 5, red ---
    plate = model.AddMeshObject()
    plate.SetName('base_plate')
    pos, tri = _box_geometry(50.0, 50.0, 5.0)
    plate.SetGeometry(pos, tri)

    red_group = model.AddColorGroup()
    red_pid = red_group.AddColor(_make_color(220, 40, 40))
    plate.SetObjectLevelProperty(red_group.GetUniqueResourceID(), red_pid)

    # --- Standoff prototype: 5 x 5 x 20, blue ---
    standoff = model.AddMeshObject()
    standoff.SetName('standoff')
    pos, tri = _box_geometry(5.0, 5.0, 20.0)
    standoff.SetGeometry(pos, tri)

    blue_group = model.AddColorGroup()
    blue_pid = blue_group.AddColor(_make_color(40, 80, 220))
    standoff.SetObjectLevelProperty(blue_group.GetUniqueResourceID(), blue_pid)

    # Build items: plate at origin, four standoffs at corners (atop plate).
    model.AddBuildItem(plate, _identity())
    standoff_locations = [
        (0.0, 0.0, 5.0),
        (45.0, 0.0, 5.0),
        (45.0, 45.0, 5.0),
        (0.0, 45.0, 5.0),
    ]
    for dx, dy, dz in standoff_locations:
        model.AddBuildItem(standoff, _translation(dx, dy, dz))

    model.QueryWriter('3mf').WriteToFile(str(path))


def _generate_small_building_ifc(path: Path) -> None:
    """Generate a small IFC4 building: Site > Building > Storey > 4 walls + 1 slab.

    Each wall and the slab carries a property set with basic info.
    """
    import ifcopenshell
    import ifcopenshell.api

    api = ifcopenshell.api

    model = api.run('project.create_file', version='IFC4')
    project = api.run('root.create_entity', model, ifc_class='IfcProject', name='Demo Project')

    # Units (millimetres for length).
    api.run('unit.assign_unit', model)

    # Geometric / model context.
    ctx = api.run('context.add_context', model, context_type='Model')
    body = api.run(
        'context.add_context',
        model,
        context_type='Model',
        context_identifier='Body',
        target_view='MODEL_VIEW',
        parent=ctx,
    )

    site = api.run('root.create_entity', model, ifc_class='IfcSite', name='Demo Site')
    building = api.run('root.create_entity', model, ifc_class='IfcBuilding', name='Demo Building')
    storey = api.run(
        'root.create_entity', model, ifc_class='IfcBuildingStorey', name='Ground Floor'
    )

    api.run('aggregate.assign_object', model, products=[site], relating_object=project)
    api.run('aggregate.assign_object', model, products=[building], relating_object=site)
    api.run('aggregate.assign_object', model, products=[storey], relating_object=building)

    # Slab (floor): 5000 x 5000 x 200 mm at origin.
    slab = api.run('root.create_entity', model, ifc_class='IfcSlab', name='Floor Slab')
    slab_rep = api.run(
        'geometry.add_slab_representation',
        model,
        context=body,
        depth=0.2,
        polyline=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
    )
    api.run('geometry.assign_representation', model, product=slab, representation=slab_rep)
    api.run('spatial.assign_container', model, products=[slab], relating_structure=storey)

    pset = api.run('pset.add_pset', model, product=slab, name='Pset_SlabCommon')
    api.run(
        'pset.edit_pset',
        model,
        pset=pset,
        properties={'IsExternal': False, 'LoadBearing': True},
    )

    # Four walls along the perimeter; each 5 m long, 3 m tall, 0.2 m thick.
    wall_specs = [
        ('North Wall', (0.0, 5.0, 0.0), 0.0),
        ('East Wall', (5.0, 0.0, 0.0), 90.0),
        ('South Wall', (0.0, 0.0, 0.0), 0.0),
        ('West Wall', (0.0, 0.0, 0.0), 90.0),
    ]
    for name, _origin, _rot in wall_specs:
        wall = api.run('root.create_entity', model, ifc_class='IfcWall', name=name)
        rep = api.run(
            'geometry.add_wall_representation',
            model,
            context=body,
            length=5.0,
            height=3.0,
            thickness=0.2,
        )
        api.run('geometry.assign_representation', model, product=wall, representation=rep)
        api.run('spatial.assign_container', model, products=[wall], relating_structure=storey)
        wpset = api.run('pset.add_pset', model, product=wall, name='Pset_WallCommon')
        api.run(
            'pset.edit_pset',
            model,
            pset=wpset,
            properties={'IsExternal': True, 'LoadBearing': True},
        )

    model.write(str(path))


def main() -> None:
    """Regenerate every committed fixture in ``data/``."""
    here = Path(__file__).parent / 'data'
    here.mkdir(parents=True, exist_ok=True)
    _generate_dxf_plate(here / 'plate.dxf')
    _generate_step_cube(here / 'cube.step')
    _generate_bracket_step(here / 'bracket.step')
    _generate_drawing_dxf(here / 'drawing.dxf')
    _generate_assembly_3mf(here / 'assembly.3mf')
    _generate_small_building_ifc(here / 'small_building.ifc')
    logger.info('wrote fixtures to %s', here)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
