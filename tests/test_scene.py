from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from acoustic_spaces.scene import OCTAVE_BANDS_HZ, Material, Scene
from acoustic_spaces.scene.model import load_yaml

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_a_key_written_twice_is_refused(tmp_path: Path) -> None:
    # PyYAML keeps the last: a second materials: dropped the first without a word.
    text = (EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8")
    path = tmp_path / "scene.yaml"
    path.write_text(text + "\nmaterials:\n  other: {absorption: [0.1, 0.1, 0.1, 0.1]}\n")
    with pytest.raises(ValueError, match="found the key 'materials' twice"):
        Scene.from_yaml(path)
    # A key a merge brings in may be given again: the mapping's own wins.
    assert load_yaml("a: &b {x: 1, y: 2}\nc: {<<: *b, y: 3}\n")["c"] == {"x": 1, "y": 3}


def test_example_scene_loads() -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")

    assert scene.name == "Shoebox 8 x 6 x 3 m"
    assert scene.geometry.file.is_absolute()
    assert scene.geometry.file.exists()
    assert set(scene.assignments.values()) <= set(scene.materials)
    assert [s.name for s in scene.sources] == ["talker"]
    assert [listener.name for listener in scene.listeners] == ["seat-a"]


def test_scalar_scattering_is_expanded_per_band() -> None:
    material = Material(absorption=[0.1] * len(OCTAVE_BANDS_HZ), scattering=0.2)
    assert material.scattering == [0.2] * len(OCTAVE_BANDS_HZ)


def test_wrong_band_count_is_rejected() -> None:
    with pytest.raises(ValidationError, match="absorption"):
        Material(absorption=[0.1, 0.2, 0.3])


def test_unknown_material_assignment_is_rejected() -> None:
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["assignments"]["floor"] = "velvet"
    with pytest.raises(ValidationError, match="velvet"):
        Scene.model_validate(data)


def test_round_trip(tmp_path: Path) -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")
    out = tmp_path / "copy.yaml"
    scene.to_yaml(out)
    again = Scene.from_yaml(out)
    assert again.model_dump() == scene.model_dump()


def test_round_trip_keeps_a_null_that_is_not_the_default(tmp_path: Path) -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")
    scene.simulation.early.coherent_max_band_hz = None  # the default is 125
    scene.to_yaml(tmp_path / "copy.yaml")
    assert Scene.from_yaml(tmp_path / "copy.yaml").simulation.early.coherent_max_band_hz is None
    assert Scene.model_validate(scene.to_data()).simulation.early.coherent_max_band_hz is None


def test_orientation_forward() -> None:
    from acoustic_spaces.scene.model import Orientation

    assert Orientation().forward() == pytest.approx((1.0, 0.0, 0.0))
    assert Orientation(yaw=90).forward() == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert Orientation(yaw=180, pitch=90).forward() == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_default_material_must_exist() -> None:
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["default_material"] = "velvet"
    with pytest.raises(ValidationError, match="velvet"):
        Scene.model_validate(data)


def test_example_materials_come_from_presets() -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")
    for material in scene.materials.values():
        assert material.preset is not None
        assert len(material.absorption) == len(OCTAVE_BANDS_HZ)
        assert "pyroomacoustics" in material.source


def test_geometry_from_dimensions() -> None:
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"] = {"box": [8, 6, 3]}
    assert Scene.model_validate(data).geometry.describe() == "box 8 x 6 x 3 m"
    data["geometry"] = {"floor_plan": [[0, 0], [8, 0], [8, 6]], "height": 300, "units": "cm"}
    described = Scene.model_validate(data).geometry.describe()
    assert described == "floor plan with 3 corners, 300 cm high"
    for bad, message in [
        ({"box": [8, 6, 3], "file": "x.obj"}, "exactly one"),
        ({}, "exactly one"),
        ({"floor_plan": [[0, 0], [8, 0], [8, 6]]}, "needs it"),
        ({"box": [8, 6, 3], "height": 3}, "height goes with floor_plan"),
        ({"box": [8, -6, 3]}, "greater than 0"),
    ]:
        data["geometry"] = bad
        with pytest.raises(ValidationError, match=message):
            Scene.model_validate(data)


def test_saved_paths_are_relative_to_the_scene_folder(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(EXAMPLES / "shoebox", tmp_path / "shoebox")
    scene = Scene.from_yaml(tmp_path / "shoebox" / "scene.yaml")
    scene.to_yaml(tmp_path / "shoebox" / "again.yaml")
    text = (tmp_path / "shoebox" / "again.yaml").read_text(encoding="utf-8")
    assert "file: shoebox.obj" in text
    assert "null" not in text
    (tmp_path / "elsewhere").mkdir()
    scene.to_yaml(tmp_path / "elsewhere" / "scene.yaml")
    moved = Scene.from_yaml(tmp_path / "elsewhere" / "scene.yaml")
    assert moved.geometry.file == (tmp_path / "shoebox" / "shoebox.obj").resolve()


def test_a_source_names_what_it_plays(tmp_path: Path) -> None:
    """A source's ``signal`` (the WAV it plays in a demo of the scene) counts from the
    scene's folder, is saved relative to it and is one of the files the scene uses; a
    source without one plays nothing."""
    import shutil

    shutil.copytree(EXAMPLES / "shoebox", tmp_path / "shoebox")
    path = tmp_path / "shoebox" / "scene.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert Scene.model_validate(data).sources[0].signal is None
    data["sources"][0]["signal"] = "sounds/talk.wav"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    scene = Scene.from_yaml(path)
    signal = (tmp_path / "shoebox" / "sounds" / "talk.wav").resolve()
    assert scene.sources[0].signal == signal
    assert signal in scene.files()
    scene.to_yaml(tmp_path / "shoebox" / "again.yaml")
    text = (tmp_path / "shoebox" / "again.yaml").read_text(encoding="utf-8")
    assert "signal: sounds/talk.wav" in text
    (tmp_path / "elsewhere").mkdir()
    scene.to_yaml(tmp_path / "elsewhere" / "scene.yaml")
    assert Scene.from_yaml(tmp_path / "elsewhere" / "scene.yaml").sources[0].signal == signal


def test_seed_is_zero_or_more() -> None:
    # Numpy takes no negative seed, which used to fail only once a run had started.
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["simulation"]["seed"] = 0
    assert Scene.model_validate(data).simulation.seed == 0
    data["simulation"]["seed"] = -1
    with pytest.raises(ValidationError, match="greater than or equal to 0") as info:
        Scene.model_validate(data)
    assert [err["loc"] for err in info.value.errors()] == [("simulation", "seed")]


@pytest.mark.parametrize(
    "where",
    [
        ("sources", 0, "position", 0),
        ("listeners", 0, "position", 1),
        ("sources", 0, "orientation", "yaw"),
        ("sources", 0, "level_db"),
        ("air", "temperature_c"),
        ("simulation", "ir_length_s"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_are_rejected(tmp_path: Path, where: tuple, value: float) -> None:
    # YAML reads .nan and .inf as numbers, and they used to pass every check: a NaN
    # source was only reported "outside the room", a NaN yaw not at all.
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"]["file"] = str(EXAMPLES / "shoebox" / "shoebox.obj")
    target = data
    for key in where[:-1]:
        target = target[key]
    target[where[-1]] = value
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValidationError, match="finite number") as info:
        Scene.from_yaml(path)
    assert [err["loc"] for err in info.value.errors()] == [where]


def test_nan_and_infinity_in_dimensions_are_rejected() -> None:
    data = yaml.safe_load((EXAMPLES / "shoebox" / "scene.yaml").read_text(encoding="utf-8"))
    for geometry in [
        {"box": [float("inf"), 6, 3]},
        {"floor_plan": [[0, 0], [4, 0], [4, float("nan")], [0, 4]], "height": 3},
    ]:
        data["geometry"] = geometry
        with pytest.raises(ValidationError, match="finite number"):
            Scene.model_validate(data)
    data["geometry"] = {"box": [8, 6, 3]}
    data["materials"]["painted_concrete"] = {"absorption": [float("nan")] * 8}
    with pytest.raises(ValidationError, match="finite number"):
        Scene.model_validate(data)


def test_sources_and_listeners_in_groups() -> None:
    """Groups nest with "/"; a source or listener left out says so, and settings left
    unset stay out of the file."""
    data = {
        "name": "groups",
        "geometry": {"box": [8, 6, 3]},
        "materials": {"wall": {"absorption": [0.1] * 8}},
        "default_material": "wall",
        "sources": [
            {"name": "main", "group": " PA / Mains ", "position": [1, 1, 1.5]},
            {"name": "spare", "position": [2, 1, 1.5], "enabled": False},
        ],
        "listeners": [{"name": "seat", "group": "Audience", "position": [5, 3, 1.2]}],
    }
    scene = Scene.model_validate(data)
    main, spare = scene.sources
    assert main.group == "PA/Mains" and main.in_use and main.enabled is None
    assert spare.group is None and not spare.in_use
    assert scene.listeners[0].group == "Audience" and scene.listeners[0].in_use
    written = scene.to_data()
    assert written["sources"][0]["group"] == "PA/Mains" and "enabled" not in written["sources"][0]
    assert written["sources"][1]["enabled"] is False and "group" not in written["sources"][1]
    for wrong in ("PA//Mains", "/PA", "PA/", " "):
        data["sources"][0]["group"] = wrong
        with pytest.raises(ValidationError, match="empty name"):
            Scene.model_validate(data)
