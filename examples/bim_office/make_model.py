"""Writes office.ifc, the model of this example, with IfcOpenShell (the ifc extra).

A 6 x 4.5 x 2.8 m office in IFC4, as a BIM program would export it, in millimetres:
walls 20 cm thick
and slabs 25 cm thick as solids, a window in the east wall and a door in the west wall
(each with its opening), every element with its material, and the room as an IfcSpace
whose space boundaries (IfcRelSpaceBoundary) give the faces of the room each element
bounds. Run it from this folder: python make_model.py
"""

from __future__ import annotations

from pathlib import Path

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
import numpy as np

#: The room inside the walls, in metres.
LENGTH, WIDTH, HEIGHT = 6.0, 4.5, 2.8
WALL, SLAB = 0.2, 0.25
#: The window in the east wall (y from, y to, z from, z to) and the door in the west wall.
WINDOW = (1.0, 3.5, 0.9, 2.2)
DOOR = (0.6, 1.5, 0.0, 2.1)
#: The file's length unit is the millimetre (IfcOpenShell's default, and Revit's). Meshes
#: are given to IfcOpenShell in metres, which converts them; the space boundaries below
#: are written directly, in millimetres.
MM = 1000.0


def _run(usecase: str, model, **kwargs):
    return ifcopenshell.api.run(usecase, model, **kwargs)


def _box(lo, hi):
    """Vertices and faces (outward quads) of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return vertices, faces


def build() -> ifcopenshell.file:
    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = _run("root.create_entity", model, ifc_class="IfcProject", name="Office")
    _run("unit.assign_unit", model)
    context = _run("context.add_context", model, context_type="Model")
    body = _run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=context,
    )
    site = _run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = _run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = _run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Ground")
    _run("aggregate.assign_object", model, relating_object=project, products=[site])
    _run("aggregate.assign_object", model, relating_object=site, products=[building])
    _run("aggregate.assign_object", model, relating_object=building, products=[storey])

    materials = {}

    elements = {}

    def solid(ifc_class: str, name: str, boxes):
        product = _run("root.create_entity", model, ifc_class=ifc_class, name=name)
        vertices, faces = [], []
        for lo, hi in boxes:
            v, f = _box(lo, hi)
            vertices.append(v)
            faces.append(f)
        shape = _run(
            "geometry.add_mesh_representation",
            model,
            context=body,
            vertices=vertices,
            faces=faces,
        )
        _run("geometry.assign_representation", model, product=product, representation=shape)
        _run("geometry.edit_object_placement", model, product=product, matrix=np.eye(4))
        return product

    def element(ifc_class: str, name: str, material: str, boxes):
        product = solid(ifc_class, name, boxes)
        _run("spatial.assign_container", model, relating_structure=storey, products=[product])
        if material not in materials:
            materials[material] = _run("material.add_material", model, name=material)
        _run("material.assign_material", model, products=[product], material=materials[material])
        elements[name] = product
        return product

    def opening(wall, filling, box) -> None:
        hole = solid("IfcOpeningElement", f"{wall.Name} opening", [box])
        _run("feature.add_feature", model, feature=hole, element=wall)
        _run("feature.add_filling", model, opening=hole, element=filling)

    outer = (-WALL, -WALL), (LENGTH + WALL, WIDTH + WALL)
    (x0, y0), (x1, y1) = outer
    element("IfcSlab", "Floor slab", "Carpet on concrete", [((x0, y0, -SLAB), (x1, y1, 0.0))])
    element(
        "IfcSlab",
        "Ceiling slab",
        "Suspended ceiling",
        [((x0, y0, HEIGHT), (x1, y1, HEIGHT + SLAB))],
    )
    element("IfcWall", "South wall", "Plaster on brick", [((x0, y0, 0.0), (x1, 0.0, HEIGHT))])
    element("IfcWall", "North wall", "Plaster on brick", [((x0, WIDTH, 0.0), (x1, y1, HEIGHT))])
    east = element(
        "IfcWall", "East wall", "Plaster on brick", [((LENGTH, 0.0, 0.0), (x1, WIDTH, HEIGHT))]
    )
    west = element(
        "IfcWall", "West wall", "Plaster on brick", [((x0, 0.0, 0.0), (0.0, WIDTH, HEIGHT))]
    )
    # The glass sits 8 cm into the wall, the door 5 cm: each opening goes through.
    h0, h1, z0, z1 = WINDOW
    window = element(
        "IfcWindow", "Window", "Glass", [((LENGTH + 0.08, h0, z0), (LENGTH + 0.12, h1, z1))]
    )
    opening(east, window, ((LENGTH - 0.01, h0, z0), (x1 + 0.01, h1, z1)))
    h0, h1, z0, z1 = DOOR
    door = element("IfcDoor", "Door", "Wood", [((-0.15, h0, z0), (-0.05, h1, z1))])
    opening(west, door, ((x0 - 0.01, h0, z0 - 0.01), (0.01, h1, z1)))
    space = _run("root.create_entity", model, ifc_class="IfcSpace", name="Office")
    _run("aggregate.assign_object", model, relating_object=storey, products=[space])
    v, f = _box((0.0, 0.0, 0.0), (LENGTH, WIDTH, HEIGHT))
    shape = _run("geometry.add_mesh_representation", model, context=body, vertices=[v], faces=[f])
    _run("geometry.assign_representation", model, product=space, representation=shape)
    _run("geometry.edit_object_placement", model, product=space, matrix=np.eye(4))
    _space_boundaries(model, space, elements)
    return model


def _plane(model, origin, normal, x_axis):
    """An IfcPlane through ``origin`` facing along ``normal``, its 2D x along
    ``x_axis``."""
    point = model.create_entity("IfcCartesianPoint", Coordinates=[MM * c for c in origin])
    axis = model.create_entity("IfcDirection", DirectionRatios=[float(c) for c in normal])
    ref = model.create_entity("IfcDirection", DirectionRatios=[float(c) for c in x_axis])
    position = model.create_entity(
        "IfcAxis2Placement3D", Location=point, Axis=axis, RefDirection=ref
    )
    return model.create_entity("IfcPlane", Position=position)


def _polyline(model, corners):
    points = [
        model.create_entity("IfcCartesianPoint", Coordinates=[MM * x for x in c]) for c in corners
    ]
    return model.create_entity("IfcPolyline", Points=[*points, points[0]])


def _rectangle(u0, u1, v0, v1):
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def _space_boundaries(model, space, elements) -> None:
    """The faces of the room each element bounds, each in a plane facing out of the
    room as IFC has it: what BIM programs export as space boundaries."""
    window = _rectangle(*WINDOW)
    door = _rectangle(DOOR[0], DOOR[1], -DOOR[2], -DOOR[3])
    faces = [
        # element, plane origin, normal (out of the room), plane x axis, outline, holes;
        # the plane's y axis is the normal times its x axis.
        ("Floor slab", (0, 0, 0), (0, 0, -1), (1, 0, 0), _rectangle(0, LENGTH, 0, -WIDTH), []),
        ("Ceiling slab", (0, 0, HEIGHT), (0, 0, 1), (1, 0, 0), _rectangle(0, LENGTH, 0, WIDTH), []),
        ("South wall", (0, 0, 0), (0, -1, 0), (1, 0, 0), _rectangle(0, LENGTH, 0, HEIGHT), []),
        ("North wall", (0, WIDTH, 0), (0, 1, 0), (1, 0, 0), _rectangle(0, LENGTH, 0, -HEIGHT), []),
        (
            "East wall",
            (LENGTH, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            _rectangle(0, WIDTH, 0, HEIGHT),
            [window],
        ),
        ("Window", (LENGTH, 0, 0), (1, 0, 0), (0, 1, 0), window, []),
        ("West wall", (0, 0, 0), (-1, 0, 0), (0, 1, 0), _rectangle(0, WIDTH, 0, -HEIGHT), [door]),
        ("Door", (0, 0, 0), (-1, 0, 0), (0, 1, 0), door, []),
    ]
    for name, origin, normal, x_axis, outline, holes in faces:
        surface = model.create_entity(
            "IfcCurveBoundedPlane",
            BasisSurface=_plane(model, origin, normal, x_axis),
            OuterBoundary=_polyline(model, outline),
            InnerBoundaries=[_polyline(model, hole) for hole in holes],
        )
        geometry = model.create_entity(
            "IfcConnectionSurfaceGeometry", SurfaceOnRelatingElement=surface
        )
        model.create_entity(
            "IfcRelSpaceBoundary",
            GlobalId=ifcopenshell.guid.new(),
            RelatingSpace=space,
            RelatedBuildingElement=elements[name],
            ConnectionGeometry=geometry,
            PhysicalOrVirtualBoundary="PHYSICAL",
            InternalOrExternalBoundary="EXTERNAL" if "slab" not in name else "INTERNAL",
        )


if __name__ == "__main__":
    path = Path(__file__).with_name("office.ifc")
    build().write(str(path))
    print(f"wrote {path}")
