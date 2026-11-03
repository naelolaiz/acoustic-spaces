import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from acoustic_spaces.examples import list_examples
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


def test_readable_text_holds_the_same_data(tmp_path: Path) -> None:
    """The editor's YAML panel writes lists on one line and letters as they are; the
    data are what Save writes."""
    scene = Scene.from_yaml(EXAMPLES / "studio" / "scene.yaml")
    scene = scene.model_copy(update={"name": "Sala de control (6 × 4,5 m)"})
    folder = EXAMPLES / "studio"
    saved, readable = scene.file_text(folder), scene.file_text(folder, readable=True)
    assert load_yaml(readable) == load_yaml(saved)
    assert "  box: [6.0, 4.5, 3.0]\n" in readable and "  box:\n  - 6.0\n" in saved
    assert "name: Sala de control (6 × 4,5 m)\n" in readable and "\\xD7" in saved
    # Save writes PyYAML's safe layout, as before.
    scene.to_yaml(tmp_path / "scene.yaml")
    text = (tmp_path / "scene.yaml").read_text(encoding="utf-8")
    assert text == yaml.safe_dump(load_yaml(text), sort_keys=False)


def test_round_trip_keeps_a_null_that_is_not_the_default(tmp_path: Path) -> None:
    scene = Scene.from_yaml(EXAMPLES / "shoebox" / "scene.yaml")
    scene.simulation.early.coherent_max_band_hz = None  # the default is 125
    scene.to_yaml(tmp_path / "copy.yaml")
    assert Scene.from_yaml(tmp_path / "copy.yaml").simulation.early.coherent_max_band_hz is None
    assert Scene.model_validate(scene.to_data()).simulation.early.coherent_max_band_hz is None


def test_every_example_reads_back_as_it_was() -> None:
    """Written and read again, every example scene is the same scene, with exactly the
    same coefficients: what the file leaves out of a material, its preset, file, entry,
    impedance or transmission gives again."""
    for example in list_examples():
        scene = Scene.from_yaml(example.path)
        folder = example.path.resolve().parent
        again = Scene.model_validate(
            load_yaml(scene.file_text(folder)), context={"base_dir": folder}
        )
        for name, material in scene.materials.items():
            other = again.materials[name]
            assert other.absorption == material.absorption, (example.name, name)
            assert other.scattering == material.scattering, (example.name, name)
            assert other.source == material.source, (example.name, name)
            for bands in ("octave", "third_octave"):
                passing = material.transmission_coefficients(bands)
                assert other.transmission_coefficients(bands) == passing, (example.name, name)
        assert again == scene, example.name


def _box_with(materials: dict, folder: Path) -> Scene:
    return Scene.model_validate(
        {
            "name": "box",
            "geometry": {"box": [8.0, 6.0, 3.0]},
            "materials": materials,
            "default_material": next(iter(materials)),
            "sources": [{"name": "talker", "position": [2.0, 3.0, 1.5]}],
            "listeners": [{"name": "seat", "position": [6.0, 3.0, 1.2]}],
        },
        context={"base_dir": folder},
    )


def _coefficient_file(path: Path, alpha: float, scatter: float) -> Path:
    """A coefficient CSV of one absorption and one scattering value from 125 Hz to 4 kHz."""
    rows = [[125, 250, 500, 1000, 2000, 4000], [alpha] * 6, [scatter] * 6]
    path.write_text("".join(",".join(str(v) for v in row) + "\n" for row in rows), "utf-8")
    return path


def test_a_scene_file_leaves_out_what_a_material_s_entry_gives(tmp_path: Path) -> None:
    """A scene file writes a material's absorption, scattering and source only where its
    preset, file, entry or transmission does not give them, so another preset, file,
    entry or transmission written in their place gives its own (the old values stayed,
    and a lower transmission loss was refused as more than the absorption)."""
    library = EXAMPLES / "material_libraries" / "catt_materials.geo"
    _coefficient_file(tmp_path / "a.csv", 0.1, 0.2)
    _coefficient_file(tmp_path / "b.csv", 0.7, 0.5)
    materials = {
        "walls": {"preset": "plasterboard"},
        "table": {"file": "a.csv"},
        "library": {"file": str(library), "entry": "carpet"},
        "partition": {"transmission_loss": 20.0},
        "own": {"absorption": [0.3] * 8, "scattering": 0.25},
    }
    _box_with(materials, tmp_path).to_yaml(tmp_path / "scene.yaml")
    data = load_yaml((tmp_path / "scene.yaml").read_text(encoding="utf-8"))
    written = data["materials"]
    for name in ("walls", "table", "library", "partition"):
        assert not {"absorption", "scattering", "source"} & set(written[name]), name
    assert written["own"]["absorption"] == [0.3] * 8
    assert written["own"]["scattering"] == [0.25] * 8
    written["walls"]["preset"] = "carpet_thin"
    written["table"]["file"] = "b.csv"
    written["library"]["entry"] = "plasterboard"
    written["partition"]["transmission_loss"] = 10.0
    again = Scene.model_validate(data, context={"base_dir": tmp_path}).materials
    alone = {
        "walls": {"preset": "carpet_thin"},
        "table": {"file": "b.csv"},
        "library": {"file": str(library), "entry": "plasterboard"},
    }
    for name, entry in alone.items():
        assert again[name] == Material.model_validate(entry, context={"base_dir": tmp_path})
    assert again["walls"].absorption[4:] == [0.2, 0.35, 0.4, 0.4]  # carpet_thin, 1-8 kHz
    assert again["table"].absorption == [0.7] * 8 and again["table"].scattering == [0.5] * 8
    assert again["table"].source.startswith("file b.csv")
    assert again["library"].source.startswith("'plasterboard' of catt_materials.geo")
    assert again["partition"].absorption == pytest.approx([0.1] * 8, rel=1e-12)


def test_a_scattering_entry_is_written_by_its_name() -> None:
    """A material whose scattering names a scattering entry is written with that name, and
    without the source it gives, so another preset beside it takes its own source (the
    old one stayed)."""
    scene = Scene.from_yaml(EXAMPLES / "concert_hall" / "scene.yaml")
    audience = scene.materials["audience"]
    written = scene.to_data()["materials"]["audience"]
    assert written["scattering"] == "theatre_audience" and "source" not in written
    assert "theatre_audience" in audience.source
    written["preset"] = "carpet_thin"
    other = Material.model_validate(written)
    assert other == Material(preset="carpet_thin", scattering="theatre_audience")
    assert other.source.startswith("pyroomacoustics materials database: carpet_thin")
    # Other values than the entry's are written as they are.
    changed = audience.model_copy(update={"scattering": [0.5] * 8})
    assert changed.to_data()["scattering"] == [0.5] * 8


def test_scene_data_does_not_read_a_material_s_entry_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scene file leaves out what a material's preset, file and entry, impedance,
    transmission or scattering entry gives without reading them again (a scene of 200
    library entries, 50 presets and 20 grounds took 0.5 s to write), and leaves out the
    same as when it reads them, for these and every example's materials."""
    import acoustic_spaces.scene.model as model

    library = EXAMPLES / "material_libraries" / "catt_materials.geo"
    _coefficient_file(tmp_path / "a.csv", 0.1, 0.2)
    materials = {
        "walls": {"preset": "plasterboard", "scattering": "theatre_audience"},
        "table": {"file": "a.csv"},
        "library": {"file": str(library), "entry": "carpet"},
        "ground": {"impedance": "grass"},
        "partition": {"transmission_loss": 20.0},
    }
    scene = _box_with(materials, tmp_path)
    read = []
    monkeypatch.setattr(model, "_material_of", lambda entry: read.append(entry))
    written = scene.to_data()["materials"]
    assert read == []
    assert written == {
        "walls": {"preset": "plasterboard", "scattering": "theatre_audience", "audience": False},
        "table": {"file": str((tmp_path / "a.csv").resolve()), "audience": False},
        "library": {"file": str(library), "entry": "carpet", "audience": False},
        "ground": {"impedance": {"preset": "grass", "model": "miki"}, "audience": False},
        "partition": {"transmission_loss": [20.0] * 8, "audience": False},
    }
    written["walls"]["preset"] = "carpet_thin"  # the caller's copy
    assert scene.to_data()["materials"]["walls"]["preset"] == "plasterboard"
    monkeypatch.undo()
    for material in [*scene.materials.values()] + [
        m for e in list_examples() for m in Scene.from_yaml(e.path).materials.values()
    ]:
        fast = material.to_data()
        material.__dict__.pop("_given", None)  # read the entry again
        assert list(material.to_data().items()) == list(fast.items())


def test_scene_data_sees_a_material_s_file_changed_since_it_was_read(tmp_path: Path) -> None:
    """A coefficient file rewritten or removed after the scene was read gives other values
    than the scene's, which the scene file then writes."""
    csv = _coefficient_file(tmp_path / "a.csv", 0.1, 0.2)
    scene = _box_with({"table": {"file": "a.csv"}}, tmp_path)
    assert scene.to_data()["materials"]["table"] == {"file": str(csv.resolve()), "audience": False}
    # Of the same size: only its modification time tells the change.
    seen = csv.stat().st_mtime_ns
    _coefficient_file(csv, 0.7, 0.5)
    os.utime(csv, ns=(seen, seen + 2_000_000_000))
    written = scene.to_data()["materials"]["table"]
    assert written["absorption"] == [0.1] * 8 and written["scattering"] == [0.2] * 8
    assert written["source"].startswith("file a.csv")
    csv.unlink()
    assert scene.to_data()["materials"]["table"] == written


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


def test_a_scene_names_its_recording_and_each_source_a_channel(tmp_path: Path) -> None:
    """A scene's ``recording`` is a free recording's name, kept as it is, or a file,
    which counts from the scene's folder, is saved relative to it and is one of the
    files the scene uses; a source's ``channel`` is left, right or mono, or none; a
    name that is neither a free recording nor a file is refused, as is any other
    channel."""
    import shutil

    from acoustic_spaces.demos import pipeline
    from acoustic_spaces.recordings import NAMES

    shutil.copytree(EXAMPLES / "shoebox", tmp_path / "shoebox")
    path = tmp_path / "shoebox" / "scene.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.pop("recording", None)
    data["sources"][0].pop("channel", None)
    scene = Scene.model_validate(data)
    assert scene.recording is None and scene.recording_file is None
    assert scene.sources[0].channel is None
    assert "recording" not in scene.to_data() and "channel" not in scene.to_data()["sources"][0]
    for name in NAMES:
        data["recording"] = name
        scene = Scene.model_validate(data)
        assert scene.recording == name and scene.recording_file is None
        assert all(Path(name).name != f.name for f in scene.files())
    data["recording"], data["sources"][0]["channel"] = "music/band.flac", "left"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    scene = Scene.from_yaml(path)
    band = (tmp_path / "shoebox" / "music" / "band.flac").resolve()
    assert scene.recording == band == scene.recording_file and band in scene.files()
    assert scene.sources[0].channel == "left"
    scene.to_yaml(tmp_path / "shoebox" / "again.yaml")
    text = (tmp_path / "shoebox" / "again.yaml").read_text(encoding="utf-8")
    assert "recording: music/band.flac" in text and "channel: left" in text
    (tmp_path / "elsewhere").mkdir()
    scene.to_yaml(tmp_path / "elsewhere" / "scene.yaml")
    assert Scene.from_yaml(tmp_path / "elsewhere" / "scene.yaml").recording == band
    # A demo's scene written in another folder keeps a free name and finds a file.
    for value, expected in (("potassium", "potassium"), ("music/band.flac", band.as_posix())):
        pipeline.write_scene(tmp_path / "work" / "s.yaml", {"recording": value}, "", path.parent)
        written = yaml.safe_load((tmp_path / "work" / "s.yaml").read_text(encoding="utf-8"))
        assert written["recording"] == expected
    for wrong in ("", "   ", 3):
        data["recording"] = wrong
        with pytest.raises(ValidationError, match="a recording file, or one of"):
            Scene.model_validate(data)
    # A name without a suffix that is no file is a free one mistyped; a file with a
    # suffix may come later, and one without is taken when it is there.
    for wrong, close in (("spicy_funk", "spicy_funk_cake"), ("Speech_Male", "speech_male")):
        data["recording"] = wrong
        with pytest.raises(ValidationError, match=f"no free recording.*speech_male.*{close}"):
            Scene.model_validate(data, context={"base_dir": tmp_path})
    (tmp_path / "take").write_bytes(b"")
    data["recording"] = "take"
    assert (
        Scene.model_validate(data, context={"base_dir": tmp_path}).recording_file
        == (tmp_path / "take").resolve()
    )
    data["recording"], data["sources"][0]["channel"] = None, "centre"
    with pytest.raises(ValidationError, match="channel"):
        Scene.model_validate(data)


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
