import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import skip_without_extra

from acoustic_spaces.room import UNASSIGNED, Room, speed_of_sound
from acoustic_spaces.scene import OCTAVE_BANDS_HZ, Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHOEBOX = EXAMPLES / "shoebox" / "scene.yaml"


def _scene(tmp_path: Path, **changes) -> Path:
    """A copy of the shoebox scene with top-level keys replaced."""
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["geometry"]["file"] = str(EXAMPLES / "shoebox" / "shoebox.obj")
    data.update(changes)
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_shoebox_room() -> None:
    room = Room.from_yaml(SHOEBOX)
    assert room.problems() == []
    assert room.warnings() == []
    assert room.volume_m3 == pytest.approx(144.0)
    assert room.area_by_material() == pytest.approx(
        {"painted_concrete": 84.0, "carpet_on_concrete": 48.0, "acoustic_ceiling": 48.0}
    )
    assert room.sources_inside.tolist() == [True]
    assert room.listeners_inside.tolist() == [True]


def test_sabine_and_eyring_match_the_formulas() -> None:
    room = Room.from_yaml(SHOEBOX)
    materials = room.scene.materials
    k = OCTAVE_BANDS_HZ.index(1000)
    a = (
        84 * materials["painted_concrete"].absorption[k]
        + 48 * materials["carpet_on_concrete"].absorption[k]
        + 48 * materials["acoustic_ceiling"].absorption[k]
    )
    c = speed_of_sound(20.0)
    assert c == pytest.approx(343.2, abs=0.1)
    sabine = 24 * math.log(10) * 144 / (c * a)
    eyring = 24 * math.log(10) * 144 / (-c * 180 * math.log(1 - a / 180))
    assert room.reverberation_time("sabine")[k] == pytest.approx(sabine)
    assert room.reverberation_time("eyring")[k] == pytest.approx(eyring)
    assert np.all(room.reverberation_time("eyring") < room.reverberation_time("sabine"))
    with pytest.raises(ValueError, match="formula"):
        room.reverberation_time("millington")


def test_arau_puchades_allows_for_uneven_absorption(tmp_path: Path) -> None:
    room = Room.from_yaml(SHOEBOX)
    materials = room.scene.materials
    k = OCTAVE_BANDS_HZ.index(1000)
    walls = materials["painted_concrete"].absorption[k]
    floor_ceiling = (
        materials["carpet_on_concrete"].absorption[k] + materials["acoustic_ceiling"].absorption[k]
    ) / 2
    constant = 24 * math.log(10) * 144 / speed_of_sound(20.0)
    # Eyring per axis from the walls facing it, weighted by their share of the area.
    x, y, z = (constant / (-180 * math.log(1 - a)) for a in (walls, walls, floor_ceiling))
    expected = x ** (36 / 180) * y ** (48 / 180) * z ** (96 / 180)
    assert room.reverberation_time("arau_puchades")[k] == pytest.approx(expected)
    # The absorbing ceiling and floor face one way: far longer than Eyring here.
    assert room.reverberation_time("arau_puchades")[k] > 2 * room.reverberation_time("eyring")[k]

    # Spread evenly, the absorption gives Eyring's time.
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    path = _scene(tmp_path, assignments={name: "painted_concrete" for name in data["assignments"]})
    room = Room.from_yaml(path)
    np.testing.assert_allclose(
        room.reverberation_time("arau_puchades", air_absorption=True),
        room.reverberation_time("eyring", air_absorption=True),
    )


def test_names_matching_materials_and_default_material(tmp_path: Path) -> None:
    scene = Scene.from_yaml(_scene(tmp_path, assignments={"floor": "carpet_on_concrete"}))
    # "ceiling" and "walls" match nothing yet.
    room = Room.from_scene(scene)
    assert (room.face_materials == UNASSIGNED).sum() == 10
    assert room.unassigned_labels() == pytest.approx({"ceiling": 48.0, "walls": 84.0})
    assert "surfaces without a material" in room.problems()[0]

    # A geometry name equal to a material name needs no assignment.
    scene.materials["walls"] = scene.materials["painted_concrete"]
    scene.default_material = "acoustic_ceiling"
    room = Room.from_scene(scene)
    assert room.problems() == []
    assert room.area_by_material()["walls"] == pytest.approx(84.0)
    assert room.area_by_material()["acoustic_ceiling"] == pytest.approx(48.0)


def test_names_match_whatever_their_case(tmp_path: Path) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    upper = {name.upper(): material for name, material in data["assignments"].items()}
    room = Room.from_yaml(_scene(tmp_path, assignments=upper))
    assert room.problems() == [] and room.unused_assignments == []
    # The name as written wins over one in another case.
    exact = {**upper, "floor": "acoustic_ceiling"}
    room = Room.from_yaml(_scene(tmp_path, assignments=exact))
    assert room.area_by_material()["acoustic_ceiling"] == pytest.approx(48.0 + 48.0)
    # A material named like a surface, in another case, still takes it.
    scene = Scene.from_yaml(_scene(tmp_path, assignments={"FLOOR": "carpet_on_concrete"}))
    scene.materials["Walls"] = scene.materials["painted_concrete"]
    scene.default_material = "acoustic_ceiling"
    assert Room.from_scene(scene).area_by_material()["Walls"] == pytest.approx(84.0)


def test_points_outside_and_unused_assignments(tmp_path: Path) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    path = _scene(
        tmp_path,
        sources=[{"name": "outside", "position": [10.0, 3.0, 1.5]}],
        assignments={**data["assignments"], "balcony": "painted_concrete"},
    )
    room = Room.from_yaml(path)
    assert room.sources_inside.tolist() == [False]
    assert any("'outside'" in p and "outside the room" in p for p in room.problems())
    assert room.unused_assignments == ["balcony"]
    assert any("balcony" in w for w in room.warnings())


def test_listener_at_a_source_is_a_problem(tmp_path: Path) -> None:
    # At the source the direct sound has no finite level: the response came out NaN.
    path = _scene(
        tmp_path,
        listeners=[
            {"name": "on-talker", "position": [2.0, 3.0, 1.5]},
            {"name": "near-talker", "position": [2.005, 3.0, 1.5]},
            {"name": "close", "position": [2.02, 3.0, 1.5]},
        ],
    )
    room = Room.from_yaml(path)
    problems = room.problems()
    assert len(problems) == 2
    for problem, name in zip(problems, ["on-talker", "near-talker"], strict=True):
        assert problem.startswith(f"listener {name!r} at [")
        assert "within 1 cm of source 'talker'" in problem
    from acoustic_spaces.engine import simulate

    with pytest.raises(ValueError, match="within 1 cm of source 'talker'"):
        simulate(room, listeners=["close"])


def test_flipped_faces_do_not_put_points_outside(tmp_path: Path) -> None:
    # The shoebox with its floor written the other way round.
    obj = tmp_path / "flipped.obj"
    text = (EXAMPLES / "shoebox" / "shoebox.obj").read_text(encoding="utf-8")
    obj.write_text(text.replace("f 1 2 3\nf 1 3 4", "f 3 2 1\nf 4 3 1"), encoding="utf-8")
    path = _scene(
        tmp_path,
        geometry={"file": str(obj)},
        listeners=[{"name": "low", "position": [6.0, 3.0, 0.3]}],
    )
    room = Room.from_yaml(path)
    assert room.problems() == []
    assert room.warnings() == []
    assert room.sources_inside.tolist() == [True]
    assert room.listeners_inside.tolist() == [True]
    assert room.volume_m3 == pytest.approx(144.0)


def test_dxf_in_millimetres_without_scene_units(tmp_path: Path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    from acoustic_spaces.geometry import box_mesh

    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4  # millimetres
    for corners in box_mesh((8000.0, 6000.0, 3000.0)).triangles:
        doc.modelspace().add_3dface([tuple(c) for c in corners])
    dxf = tmp_path / "room.dxf"
    doc.saveas(dxf)
    path = _scene(tmp_path, geometry={"file": str(dxf)}, default_material="painted_concrete")
    room = Room.from_yaml(path)
    assert room.scene.geometry.units is None
    assert room.volume_m3 == pytest.approx(144.0)  # not 8000 x 6000 x 3000 m
    assert room.problems() == []
    # Units in the scene win over the file's.
    path = _scene(
        tmp_path,
        geometry={"file": str(dxf), "units": "cm"},
        default_material="painted_concrete",
    )
    assert Room.from_yaml(path).volume_m3 == pytest.approx(144.0 * 1000.0)


def test_model_of_degenerate_faces_is_refused(tmp_path: Path) -> None:
    obj = tmp_path / "line.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 2 0 0\nv 3 0 0\ng floor\nf 1 2 3\nf 2 3 4\n")
    # It loaded as a closed room of no triangles, its points all "outside the room".
    with pytest.raises(ValueError, match="line.obj has no faces left: all 2 of them have no area"):
        Room.from_yaml(_scene(tmp_path, geometry={"file": str(obj)}))


def test_open_model_skips_volume_and_inside_checks(tmp_path: Path) -> None:
    obj = tmp_path / "open.obj"
    lines = (EXAMPLES / "shoebox" / "shoebox.obj").read_text(encoding="utf-8").splitlines()
    ceiling = lines.index("g ceiling")
    obj.write_text("\n".join(lines[:ceiling] + lines[ceiling + 3 :]) + "\n", encoding="utf-8")
    room = Room.from_yaml(_scene(tmp_path, geometry={"file": str(obj)}))
    assert room.volume_m3 is None
    assert room.sources_inside is None
    assert "not closed" in room.warnings()[0]
    with pytest.raises(ValueError, match="closed"):
        room.reverberation_time()


def test_inconsistent_edges_are_not_said_to_matter_for_the_volume() -> None:
    room = Room.from_yaml(SHOEBOX)
    room.report.inconsistent_edges = 3  # a part that cannot be oriented, as a Moebius panel
    (warning,) = room.warnings()
    assert warning.startswith("3 edges join faces with opposite orientation")
    assert "only the direction of the normals is unknown" in warning
    assert "only matters for the volume" not in warning


@pytest.mark.parametrize(
    "scene", sorted(EXAMPLES.glob("*/scene.yaml")), ids=lambda p: p.parent.name
)
def test_example_scenes_are_ready_to_simulate(scene: Path) -> None:
    skip_without_extra(scene.parent.name)
    room = Room.from_yaml(scene)
    assert room.problems() == []
    if scene.parent.name == "leaky_classroom":  # open on purpose, for the leak finder
        (warning,) = room.warnings()
        assert "not closed" in warning and "validate --leaks" in warning
        return
    warnings = room.warnings()
    if scene.parent.name == "stadium_pa":
        # The players and the front of the pitch are nearer the arrays than their
        # far-field patterns hold, as at any concert; near_field.py measures how far off.
        assert warnings and all("whose far-field pattern holds" in w for w in warnings)
        warnings = []
    assert warnings == []
    assert room.report.closed and room.report.normals_inward


def test_rooms_reuse_an_unchanged_mesh() -> None:
    room = Room.from_yaml(SHOEBOX)
    data = room.scene.model_dump(mode="json", exclude_none=True)
    data["sources"][0]["position"] = [2.0, 2.0, 1.5]
    moved = Room.from_scene(Scene.model_validate(data), reuse=room)
    assert moved.mesh is room.mesh
    assert moved.scene.sources[0].position == (2.0, 2.0, 1.5)
    data["geometry"] = {"box": [5.0, 4.0, 3.0]}
    data["assignments"] = {}
    boxed = Room.from_scene(Scene.model_validate(data), reuse=room)
    assert boxed.mesh is not room.mesh
    assert boxed.volume_m3 == pytest.approx(60.0)
