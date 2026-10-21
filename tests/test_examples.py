import importlib.util
import re
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pytest
from conftest import skip_without_extra

from acoustic_spaces.cli import main
from acoustic_spaces.engine import simulate
from acoustic_spaces.examples import (
    GROUPS,
    copy_examples,
    example_groups,
    example_path,
    list_examples,
)
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NAMES = [e.name for e in list_examples()]


def test_wheels_carry_every_file_the_examples_use() -> None:
    """CMakeLists.txt installs the examples' files by their extension: every file a
    scene uses (an array's element's too) and every script must have one of them."""
    cmake = EXAMPLES.parent / "CMakeLists.txt"
    if not cmake.is_file():
        pytest.skip("needs the source tree")
    text = cmake.read_text(encoding="utf-8")
    rule = text[text.index("DESTINATION acoustic_spaces/example_scenes") :]
    patterns = re.findall(r'PATTERN "([^"]+)"', rule[: rule.index(")")])
    for scene in sorted(EXAMPLES.glob("*/scene.yaml")):
        folder = scene.parent
        used = [f for f in Scene.from_yaml(scene).files() if f.is_relative_to(folder)]
        used += [*folder.glob("*.py"), *folder.glob("*.sh"), scene]
        left_out = [f.name for f in used if not any(fnmatch(f.name, p) for p in patterns)]
        assert not left_out, folder.name


def test_every_example_folder_is_listed() -> None:
    assert NAMES == sorted(p.parent.name for p in EXAMPLES.glob("*/scene.yaml"))
    assert {"church", "concert_hall", "lecture_hall", "shoebox", "studio"} <= set(NAMES)
    for example in list_examples():
        assert example.title and example.description
        assert example.path.is_file()


def test_every_example_is_in_a_group() -> None:
    """The editor shows the examples by group: a new example needs its place in GROUPS,
    which names only examples there are."""
    listed = [name for _, names in GROUPS for name in names]
    assert sorted(listed) == NAMES
    groups = example_groups()
    assert [group for group, _ in groups] == [group for group, _ in GROUPS]
    assert [e.name for _, members in groups for e in members] == listed
    assert example_groups([]) == []


def test_example_path() -> None:
    assert example_path("studio").parent.name == "studio"
    with pytest.raises(ValueError, match="examples: apartment, archway, bim_office"):
        example_path("cathedral")


@pytest.mark.parametrize("name", NAMES)
def test_examples_simulate(name: str) -> None:
    skip_without_extra(name)
    room = Room.from_yaml(example_path(name))
    sim = room.scene.simulation
    quick = sim.model_copy(
        update={
            "ir_length_s": 0.3,
            "seed": 1,
            "early": sim.early.model_copy(update={"max_order": 2}),
            "late": sim.late.model_copy(update={"rays": 500}),
        }
    )
    result = simulate(room, settings=quick)
    assert len(result) == len(room.scene.sources) * len(room.scene.listeners)
    # Wind and temperature change the speed of sound along the way, and bend the path.
    tolerance = 1e-6 if room.scene.atmosphere is None else 0.02
    for source in room.scene.sources:
        for listener in room.scene.listeners:
            # The nearest of a line's or surface's points, after the source's own delay.
            distance = np.linalg.norm(source.elements() - listener.position, axis=1).min()
            expected = distance / room.speed_of_sound() + source.delay_ms / 1000.0
            delay = result[source.name, listener.name].direct_delay_s
            assert delay == pytest.approx(expected, rel=tolerance)


def test_copy_examples(tmp_path: Path) -> None:
    copied = copy_examples(tmp_path, ["concert_hall", "studio"])
    assert copied == [tmp_path / "concert_hall" / "scene.yaml", tmp_path / "studio" / "scene.yaml"]
    # the copy uses its own model file
    room = Room.from_yaml(copied[0])
    assert room.scene.geometry.file == (tmp_path / "concert_hall" / "concert_hall.obj").resolve()
    assert room.problems() == []
    with pytest.raises(FileExistsError):
        copy_examples(tmp_path, ["studio"])
    with pytest.raises(ValueError, match="cathedral"):
        copy_examples(tmp_path / "other", ["cathedral"])


def test_examples_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["examples"]) == 0
    out = capsys.readouterr().out
    for name in NAMES:
        assert f"{name}: " in out
    assert main(["examples", "church"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("church: ") and "studio" not in out
    assert main(["examples", "--copy", str(tmp_path), "studio"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "studio" / "scene.yaml")
    assert main(["examples", "cathedral"]) == 1
    assert "no example named cathedral" in capsys.readouterr().err


def _script(example: str, name: str):
    """The script ``name``.py of an example, loaded as a module."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / example / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_listening_room_bass(capsys: pytest.CaptureFixture[str]) -> None:
    bass = _script("listening_room", "bass")
    room = Room.from_yaml(example_path("listening_room"))
    # The early part at 63 Hz: the wall's reflection in phase against the wall, half a
    # period late a metre out; in energy the wall makes no difference.
    pressure, energy = (bass.levels(room, coherent, 300) for coherent in (125, None))
    wall, out = (pressure[name][1][0] for name in ("against the wall", "a metre out"))
    assert wall - out > 4.0
    wall, out = (energy[name][1][0] for name in ("against the wall", "a metre out"))
    assert abs(wall - out) < 1.0
    assert bass.main(["--rays", "300"]) == 0
    assert "in pressure up to 125 Hz" in capsys.readouterr().out


def test_stadium_sound_system_is_aligned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the stadium's description says of its sound system."""
    align = _script("stadium_pa", "align")
    seats = align.arrivals(Room.from_yaml(example_path("stadium_pa")), rays=30, seed=1)
    time = {seat: {row.source: row.time_s for row in rows} for seat, rows in seats.items()}
    # The subwoofers arrive with the main arrays at the mix position, and each fill or
    # delay comes 5 to 20 ms after the main array it backs up, so that one hears the stage.
    assert abs(time["mix_position"]["subs"] - time["mix_position"]["main_left"]) < 0.002
    for seat, delay, array in (
        ("pitch_front", "front_fill_left", "main_left"),
        ("pitch_behind_towers", "tower_left", "main_left"),
        ("north_stand_far", "delay_north", "main_left"),
        ("north_stand_top", "delay_north", "main_left"),
        ("east_stand", "delay_east_right", "main_right"),
    ):
        assert 0.005 < time[seat][delay] - time[seat][array] < 0.021, seat
    # Nobody hears an echo.
    assert not [(seat, row.source) for seat, rows in seats.items() for row in rows if row.echo]
    monkeypatch.setattr(align, "arrivals", lambda room, rays=None: seats)
    assert align.main([]) == 0
    out = capsys.readouterr().out
    assert "north_stand_far: " in out and "delay_north" in out and "  echo" not in out


def test_each_player_hears_their_own_monitor_loudest() -> None:
    align = _script("concert_stage", "align")
    seats = align.arrivals(Room.from_yaml(example_path("concert_stage")), rays=30, seed=1)
    assert not [(seat, row.source) for seat, rows in seats.items() for row in rows if row.echo]
    for seat, monitor in (
        ("singer", "vocal_wedges"),
        ("drummer", "drum_fill"),
        ("guitarist", "side_fill_left"),
    ):
        loudest = max(seats[seat], key=lambda row: align.total(row.direct[align.ECHO_BANDS]))
        assert loudest.source == monitor, seat


@pytest.mark.parametrize(
    ("example", "low_db", "high_db"), [("stadium_pa", 3.5, 10.5), ("concert_stage", 3.0, 8.0)]
)
def test_stadium_arrays_near_their_listeners(
    example: str,
    low_db: float,
    high_db: float,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """As the descriptions say: the arrays' far-field patterns would be off by a few dB
    at the listeners up to 2 kHz but by much more at 4 and 8 kHz, which the arrays avoid
    by adding up their boxes one by one; the subwoofers' within 0.6 dB in their bands,
    and the bands their filter leaves out are not counted."""
    near = _script(example, "near_field")
    found = near.differences(Room.from_yaml(example_path(example)))
    assert {"main_left", "subs"} <= set(found)
    arrays = np.array([db for s, rows in found.items() if s != "subs" for db in rows.values()])
    low, high = np.abs(arrays[:, :6]).max(), np.abs(arrays[:, 6:]).max()
    assert 1.0 < low < low_db and high_db - 3.0 < high < high_db
    assert max(np.abs(db[:2]).max() for db in found["subs"].values()) < 0.6
    assert all(np.isnan(db[3:]).all() for db in found["subs"].values())
    monkeypatch.setattr(near, "differences", lambda room: found)
    assert near.main([]) == 0
    assert "main_left, far field above the elements added up (dB)" in capsys.readouterr().out


def test_the_stadium_examples_share_their_parts(tmp_path: Path) -> None:
    """stadium_concert is stadium_pa and concert_stage together, and the three share their
    scripts, their arrays and make_model.py, which writes each one's files as they are."""
    folders = [EXAMPLES / name for name in ("stadium_pa", "concert_stage", "stadium_concert")]
    for name in ("align.py", "near_field.py", "make_model.py", "feedback.py", "box.csv"):
        copies = {(f / name).read_bytes() for f in folders if (f / name).exists()}
        assert len(copies) == 1, name

    def entries(folder: Path, kind: str) -> dict:
        data = Scene.from_yaml(folder / "scene.yaml").to_data()[kind]
        for entry in data:
            if isinstance(entry.get("directivity"), dict):
                entry["directivity"]["file"] = Path(entry["directivity"]["file"]).name
        return {entry["name"]: entry for entry in data}

    for kind in ("sources", "listeners"):
        whole = entries(folders[2], kind)
        parts = {**entries(folders[0], kind), **entries(folders[1], kind)}
        assert whole == parts, kind
    for folder in folders:
        copy = tmp_path / folder.name
        copy.mkdir()
        for name in ("scene.yaml", "make_model.py"):
            (copy / name).write_bytes((folder / name).read_bytes())
        spec = importlib.util.spec_from_file_location("make_model", copy / "make_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()  # writes next to itself, into the copy
        written = sorted(p.name for p in copy.iterdir() if p.suffix in (".obj", ".csv", ".yaml"))
        kept = sorted(p.name for p in folder.iterdir() if p.suffix in (".obj", ".csv", ".yaml"))
        assert written == kept, folder.name
        for name in written:
            assert (copy / name).read_bytes() == (folder / name).read_bytes(), name


def test_stadium_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The singer's microphone: its gain before feedback, where it rings, and a loop
    that dies away below unity gain and grows above it."""
    feedback = _script("concert_stage", "feedback")
    found = feedback.loop(Room.from_yaml(example_path("concert_stage")), rays=30)
    assert "vocal_wedges" in found["speakers"] and "subs" not in found["speakers"]
    for where in ("mix", "ears"):
        assert np.all(np.isfinite(found["pag"][where]))
    # The system can lift the voice far more at the mix position than at the singer's
    # ears, where the wedges are; the loop's peaks only take gain away.
    assert np.all(found["pag"]["mix"] > found["pag"]["ears"] + 20.0)
    assert np.all(found["peaks"] >= 0.0)
    assert len(found["ring_hz"]) == 3
    assert all(44.0 < f < 11_400.0 for f in found["ring_hz"])
    quarter = feedback.RATE // 4
    for gain, grows in ((-6.0, False), (3.0, True)):
        out = feedback.render(found["response"], gain, seconds=1.0)
        first, last = (np.sqrt(np.mean(part**2)) for part in (out[:quarter], out[-quarter:]))
        assert (last > first) == grows, gain
    monkeypatch.setattr(feedback, "loop", lambda room, rays=None: found)
    monkeypatch.setattr(feedback, "RENDER_S", 0.5)
    assert feedback.main(["--out", str(tmp_path / "loop.wav")]) == 0
    assert "It rings first at" in capsys.readouterr().out
    assert (tmp_path / "loop.wav").stat().st_size > 0
