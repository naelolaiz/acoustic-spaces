"""IFC files, read through IfcOpenShell (the ifc extra)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene
from acoustic_spaces.examples import example_path, list_examples
from acoustic_spaces.geometry import ifc_spaces, load_mesh
from acoustic_spaces.geometry.mesh import RoomMesh

OFFICE = example_path("bim_office").parent / "office.ifc"


def _areas(mesh: RoomMesh) -> dict[tuple[str, ...], float]:
    areas = mesh.face_areas()
    return {
        label: round(float(areas[mesh.face_labels == k].sum()), 4)
        for k, label in enumerate(mesh.labels)
    }


def _office_scene(**geometry) -> Scene:
    data = Scene.from_yaml(example_path("bim_office")).model_dump(mode="json", exclude_none=True)
    data["geometry"] = {"file": str(OFFICE), **geometry}
    return Scene.model_validate(data)


def test_a_room_from_its_space_boundaries() -> None:
    pytest.importorskip("ifcopenshell")
    areas = _areas(load_mesh(OFFICE, space="Office"))
    # Each face of the room is named by the element it lies on; the window and the
    # door fill the holes left in their walls.
    assert areas == {
        ("Floor slab", "Carpet on concrete", "IfcSlab"): 27.0,
        ("Ceiling slab", "Suspended ceiling", "IfcSlab"): 27.0,
        ("South wall", "Plaster on brick", "IfcWall"): 16.8,
        ("North wall", "Plaster on brick", "IfcWall"): 16.8,
        ("East wall", "Plaster on brick", "IfcWall"): pytest.approx(12.6 - 2.5 * 1.3),
        ("Window", "Glass", "IfcWindow"): pytest.approx(2.5 * 1.3),
        ("West wall", "Plaster on brick", "IfcWall"): pytest.approx(12.6 - 0.9 * 2.1),
        ("Door", "Wood", "IfcDoor"): pytest.approx(0.9 * 2.1),
    }
    room = Room.from_yaml(example_path("bim_office"))
    assert room.report.closed and room.report.normals_inward
    assert room.report.volume_m3 == pytest.approx(6.0 * 4.5 * 2.8)
    assert room.problems() == [] and room.warnings() == []
    # Assignments by IFC material, by class and by element name.
    assert room.area_by_material() == pytest.approx(
        {"carpet": 27.0, "ceiling_tiles": 27.0, "plaster": 53.66, "glass": 3.25, "wood": 1.89}
    )


def test_every_element_without_a_space() -> None:
    pytest.importorskip("ifcopenshell")
    mesh = load_mesh(OFFICE)
    assert len(mesh.labels) == 8
    np.testing.assert_allclose(mesh.vertices.min(axis=0), (-0.2, -0.2, -0.25), atol=1e-9)
    np.testing.assert_allclose(mesh.vertices.max(axis=0), (6.2, 4.7, 3.05), atol=1e-9)
    # The window pane sits in its opening, 8 cm into the wall, with the hole cut round it.
    # In the file's order, whichever thread built each element first.
    assert [label[0] for label in mesh.labels] == [
        "Floor slab",
        "Ceiling slab",
        "South wall",
        "North wall",
        "East wall",
        "West wall",
        "Window",
        "Door",
    ]
    window = mesh.vertices[np.unique(mesh.faces[mesh.face_labels == 6])]
    assert window[:, 0].min() == pytest.approx(6.08) and window[:, 0].max() == pytest.approx(6.12)
    east = _areas(mesh)[("East wall", "Plaster on brick", "IfcWall")]
    assert east == pytest.approx(2 * (4.5 * 2.8 - 3.25) + 2 * (2.5 + 1.3) * 0.2 + 0.2 * 2 * 7.3)
    from acoustic_spaces.engine.leaks import find_leaks

    room = Room.from_scene(_office_scene())
    assert not room.report.closed  # the solids only touch: no edges in common
    assert find_leaks(room, rays=300).escaped == 0  # though no ray gets out


def test_the_file_records_its_units() -> None:
    pytest.importorskip("ifcopenshell")
    # The model is in millimetres; units: m reads its numbers as metres.
    mesh = load_mesh(OFFICE, units="m", space="Office")
    np.testing.assert_allclose(mesh.vertices.max(axis=0), (6000.0, 4500.0, 2800.0))


def test_a_space_without_boundaries_takes_the_elements_behind_its_solid(
    tmp_path: Path,
) -> None:
    ifcopenshell = pytest.importorskip("ifcopenshell")
    model = ifcopenshell.open(str(OFFICE))
    for boundary in model.by_type("IfcRelSpaceBoundary"):
        model.remove(boundary)
    path = tmp_path / "no_boundaries.ifc"
    model.write(str(path))
    areas = _areas(load_mesh(path, space="Office"))
    assert len(areas) == 8
    assert areas[("Floor slab", "Carpet on concrete", "IfcSlab")] == pytest.approx(27.0)
    assert areas[("South wall", "Plaster on brick", "IfcWall")] == pytest.approx(16.8)
    # Windows and doors come out ragged, at the triangles' size.
    assert areas[("Window", "Glass", "IfcWindow")] == pytest.approx(3.25, rel=0.15)
    assert areas[("Door", "Wood", "IfcDoor")] == pytest.approx(1.89, rel=0.15)
    room = Room.from_scene(_office_scene(file=str(path), space="Office"))
    assert room.report.closed and room.report.normals_inward
    assert room.report.volume_m3 == pytest.approx(75.6)


def _as_face_surfaces(model) -> None:
    """Each space boundary given as faces with corners in 3D, as some programs export
    them, in place of a bounded plane."""
    import ifcopenshell.util.placement

    for boundary in model.by_type("IfcRelSpaceBoundary"):
        geometry = boundary.ConnectionGeometry
        plane = geometry.SurfaceOnRelatingElement
        position = ifcopenshell.util.placement.get_axis2placement(plane.BasisSurface.Position)

        def loop(curve, kind, position=position):
            corners = [(*p.Coordinates[:2], 0.0, 1.0) for p in curve.Points[:-1]]
            points = [
                model.create_entity("IfcCartesianPoint", Coordinates=list(position @ c)[:3])
                for c in corners
            ]
            polygon = model.create_entity("IfcPolyLoop", Polygon=points)
            return model.create_entity(kind, Bound=polygon, Orientation=True)

        bounds = [loop(plane.OuterBoundary, "IfcFaceOuterBound")]
        bounds += [loop(c, "IfcFaceBound") for c in plane.InnerBoundaries or ()]
        face = model.create_entity("IfcFace", Bounds=bounds)
        faces = model.create_entity("IfcConnectedFaceSet", CfsFaces=[face])
        geometry.SurfaceOnRelatingElement = model.create_entity(
            "IfcFaceBasedSurfaceModel", FbsmFaces=[faces]
        )


def test_boundaries_given_as_faces_in_3d(tmp_path: Path) -> None:
    ifcopenshell = pytest.importorskip("ifcopenshell")
    model = ifcopenshell.open(str(OFFICE))
    _as_face_surfaces(model)
    path = tmp_path / "faces.ifc"
    model.write(str(path))
    assert _areas(load_mesh(path, space="Office")) == _areas(load_mesh(OFFICE, space="Office"))
    room = Room.from_scene(_office_scene(file=str(path), space="Office"))
    assert room.report.closed and room.report.volume_m3 == pytest.approx(75.6)


def test_layer_sets_types_and_standard_cases_name_the_faces(tmp_path: Path) -> None:
    pytest.importorskip("ifcopenshell")
    import ifcopenshell.api

    from acoustic_spaces.geometry.io import _ifc_label

    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    run = ifcopenshell.api.run
    run("root.create_entity", model, ifc_class="IfcProject", name="P")
    wall = run("root.create_entity", model, ifc_class="IfcWallStandardCase", name="Wall 1")
    kind = run("root.create_entity", model, ifc_class="IfcWallType", name="Partition 100")
    run("type.assign_type", model, related_objects=[wall], relating_type=kind)
    layers = run(
        "material.add_material_set", model, name="Stud wall", set_type="IfcMaterialLayerSet"
    )
    for name in ("Gypsum board", "Mineral wool", "Gypsum board"):
        material = run("material.add_material", model, name=name)
        run("material.add_layer", model, layer_set=layers, material=material)
    run("material.assign_material", model, products=[kind], material=layers)
    assert _ifc_label(wall) == (
        "Wall 1",
        "Partition 100",
        "Stud wall",
        "Gypsum board",
        "Mineral wool",
        "IfcWallStandardCase",
        "IfcWall",
    )


def test_spaces_by_name_long_name_or_id_and_errors(tmp_path: Path) -> None:
    pytest.importorskip("ifcopenshell")
    assert ifc_spaces(OFFICE) == [("Office", "Office")]
    assert len(load_mesh(OFFICE, space=" OFFICE ").faces) == len(
        load_mesh(OFFICE, space="Office").faces
    )
    with pytest.raises(ValueError, match="no space 'Kitchen'; it has Office"):
        load_mesh(OFFICE, space="Kitchen")
    bad = tmp_path / "bad.ifc"
    bad.write_text("not IFC\n")
    with pytest.raises(ValueError, match="not a readable IFC file"):
        load_mesh(bad)


def test_space_goes_with_an_ifc_file() -> None:
    with pytest.raises(ValueError, match="space names a room of an IFC file"):
        Scene.model_validate(
            {
                "geometry": {"box": [6, 5, 3], "space": "Office"},
                "materials": {"m": {"absorption": [0.1] * 8}},
                "default_material": "m",
                "sources": [{"name": "s", "position": [1, 1, 1]}],
                "listeners": [{"name": "l", "position": [2, 2, 1]}],
            }
        )
    with pytest.raises(ValueError, match="shoebox.obj is not one"):
        load_mesh(example_path("shoebox").parent / "shoebox.obj", space="Office")


def test_without_ifcopenshell_the_error_and_the_examples_say_how_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.util

    from acoustic_spaces.cli import main

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "ifcopenshell" else find_spec(name, *a),
    )
    monkeypatch.setitem(sys.modules, "ifcopenshell", None)
    with pytest.raises(ImportError, match=r"needs IfcOpenShell: .*uv sync --extra ifc"):
        load_mesh(OFFICE)
    office = next(e for e in list_examples() if e.name == "bim_office")
    assert office.extra == office.missing_extra == "ifc"
    assert main(["examples", "bim_office"]) == 0
    assert "needs the ifc extra: " in capsys.readouterr().out
