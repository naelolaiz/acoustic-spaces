import math
from pathlib import Path

import numpy as np
import pytest
import yaml

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


@pytest.mark.parametrize(
    "scene", sorted(EXAMPLES.glob("*/scene.yaml")), ids=lambda p: p.parent.name
)
def test_example_scenes_are_ready_to_simulate(scene: Path) -> None:
    room = Room.from_yaml(scene)
    assert room.problems() == []
    assert room.warnings() == []
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
