import argparse
import importlib.util
import math
import os
import re
import shutil
import zipfile
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import skip_without_extra

from acoustic_spaces import recordings
from acoustic_spaces.audio import write_wav
from acoustic_spaces.cli import main
from acoustic_spaces.demos import (
    PATH_KINDS,
    audio,
    demo_for,
    list_demos,
    load,
    pipeline,
    scene_demo,
    shell_line,
)
from acoustic_spaces.engine import simulate
from acoustic_spaces.examples import (
    GROUPS,
    copy_examples,
    example_groups,
    example_of,
    example_path,
    list_examples,
)
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NAMES = [e.name for e in list_examples()]
#: The demos that come with the examples: (example, the name before "_demo.py").
DEMO_SCRIPTS = [
    ("church", "church"),
    ("gallery", "walk"),
    ("lakeside_concert", "lakeside"),
    ("orchestra", "orchestra"),
    ("stadium_concert", "stadium"),
]


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
    with pytest.raises(ValueError, match="examples: airport, apartment, archway, bim_office"):
        example_path("cathedral")


def test_a_copy_of_an_example_is_known_by_its_folder_or_its_scene(tmp_path: Path) -> None:
    """A scene is an example's when it sits in a folder of the example's name, or when
    it bears the example's scene name wherever it is saved; any other scene, or one that
    cannot be read, is none."""
    assert example_of(example_path("roadside")) == "roadside"
    assert example_of(tmp_path / "street" / "scene.yaml") == "street"  # the folder decides
    copy = tmp_path / "anywhere" / "mine.yaml"
    copy.parent.mkdir()
    copy.write_text(example_path("roadside").read_text(encoding="utf-8"), encoding="utf-8")
    assert example_of(copy) == "roadside"
    copy.write_text("name: My own road\n", encoding="utf-8")
    assert example_of(copy) is None
    copy.write_text("[not: a scene", encoding="utf-8")
    assert example_of(copy) is None
    assert example_of(tmp_path / "anywhere" / "missing.yaml") is None


def test_the_examples_of_an_installed_copy_are_known(tmp_path: Path, monkeypatch) -> None:
    """A path among an installed package's examples is one of them, writable or not; a
    copy elsewhere is not, nor is a source checkout's examples folder, which a package
    without installed examples uses."""
    from acoustic_spaces import examples

    installed = tmp_path / "site-packages" / "acoustic_spaces" / "example_scenes"
    (installed / "shoebox").mkdir(parents=True)
    monkeypatch.setattr(examples, "_installed_dir", lambda: installed)
    assert os.access(installed / "shoebox", os.W_OK)  # as in a virtual environment
    assert examples.is_installed_example(installed / "shoebox")
    assert examples.is_installed_example(installed / "shoebox" / "scene.yaml")
    assert not examples.is_installed_example(tmp_path / "shoebox")
    monkeypatch.setattr(examples, "_installed_dir", lambda: tmp_path / "no examples here")
    assert not examples.is_installed_example(EXAMPLES / "shoebox")


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
    # A source or listener left out (the roadside's car until it is ticked) is not simulated.
    sources = [s for s in room.scene.sources if s.in_use]
    listeners = [lis for lis in room.scene.listeners if lis.in_use]
    assert len(result) == len(sources) * len(listeners)
    # Wind and temperature change the speed of sound along the way, and bend the path.
    tolerance = 1e-6 if room.scene.atmosphere is None else 0.02
    for source in sources:
        for listener in listeners:
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


def test_the_gallery_stands_or_walks_in_one_scene(tmp_path: Path) -> None:
    """One gallery for the still listeners, the visitor who walks and the one who turns
    their head on the spot 4 m in front of the guide, a quarter at a time and held for
    4 s, round to the front again (the script's turn of before, heard through the head
    tracker): the trajectories turn off and on, and the gallery copies with its demo."""
    from acoustic_spaces.room import Room
    from acoustic_spaces.walkthrough import _walker, head_turners, knots, movers, with_moving

    assert "gallery" in NAMES
    path = example_path("gallery")
    scene = Scene.from_yaml(path)
    listeners = {lis.name: lis for lis in scene.listeners}
    assert set(listeners) == {"in sight", "round the corner", "visitor", "standing"}
    assert all(lis.in_use for lis in listeners.values())
    assert (
        listeners["in sight"].trajectory is None
        and listeners["round the corner"].trajectory is None
    )
    visitor = listeners["visitor"]
    assert visitor.position == (17.0, 4.0, 1.6) and visitor.trajectory.enabled
    assert visitor.trajectory.to == [(4.0, 4.0, 1.6), (4.0, 16.0, 1.6)]
    standing = listeners["standing"]
    assert math.dist(standing.position, scene.sources[0].position) == pytest.approx(4.0)
    turn = knots(standing)
    assert turn.times.tolist() == [0.0, 4.0, 10.0, 14.0, 20.0, 24.0, 30.0, 34.0, 40.0]
    assert turn.angles[:, 0].tolist() == [0.0, 0.0, 90.0, 90.0, 180.0, 180.0, 270.0, 270.0, 360.0]
    assert np.ptp(turn.positions, axis=0).max() == 0.0
    assert [item.name for _kind, item in movers(scene)] == ["visitor", "standing"]
    assert head_turners(scene) == ["standing"]
    # The walkthrough walks the visitor: the listener standing only turns where it is.
    assert _walker(Room.from_scene(scene), None)[1].name == "visitor"
    assert movers(with_moving(scene, [])) == []
    assert scene.outputs[0].format == "binaural"  # what walkthrough plays by default
    assert demo_for("gallery").example == "gallery"
    copied = copy_examples(tmp_path, ["gallery"])
    assert copied == [tmp_path / "gallery" / "scene.yaml"]
    assert copied[0].read_bytes() == path.read_bytes()
    assert (copied[0].parent / "walk_demo.py").is_file()


def _script(example: str, name: str):
    """The script ``name``.py of an example, loaded as a module."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / example / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_listening_room_bass(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
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
    # The scene it is given, as the editor's Run demo window gives it the open one.
    copy = room.scene.model_copy(deep=True)
    copy.sources[0].name = "my own"
    copy.to_yaml(tmp_path / "scene.yaml")
    assert bass.main([str(tmp_path / "scene.yaml"), "--rays", "300"]) == 0
    assert "  my own " in capsys.readouterr().out


def _with_left_out(example: str) -> Room:
    """The example's room with a copy of its main left array and of a listener left out
    of simulations (``enabled: false``), as the stadium's helicopter is: the stadium
    scripts take neither, and everything else simulates as without them."""
    scene = Scene.from_yaml(example_path(example))
    array = next(s for s in scene.sources if s.name == "main_left")
    out = {"name": "left_out", "enabled": False}
    return Room.from_scene(
        scene.model_copy(
            update={
                "sources": [*scene.sources, array.model_copy(update=out)],
                "listeners": [*scene.listeners, scene.listeners[0].model_copy(update=out)],
            }
        )
    )


def test_stadium_sound_system_is_aligned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """What the stadium concert's description says of its sound system and its
    monitors."""
    align = _script("stadium_concert", "align")
    seats = align.arrivals(_with_left_out("stadium_concert"), rays=30, seed=1)
    assert "left_out" not in seats
    assert not [row for rows in seats.values() for row in rows if row.source == "left_out"]
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
    _no_echo_and_own_monitors(align, seats)
    monkeypatch.setattr(align, "arrivals", lambda room, rays=None: seats)
    assert align.main([]) == 0
    out = capsys.readouterr().out
    assert "north_stand_far: " in out and "delay_north" in out and "  echo" not in out


def _no_echo_and_own_monitors(align, seats: dict) -> None:
    """Nobody hears an echo, and each player hears their own monitor loudest."""
    assert not [(seat, row.source) for seat, rows in seats.items() for row in rows if row.echo]
    for seat, monitor in (
        ("singer", "vocal_wedges"),
        ("drummer", "drum_fill"),
        ("guitarist", "side_fill_left"),
    ):
        loudest = max(seats[seat], key=lambda row: align.total(row.direct[align.ECHO_BANDS]))
        assert loudest.source == monitor, seat


def test_each_player_hears_their_own_monitor_loudest() -> None:
    align = _script("concert_stage", "align")
    seats = align.arrivals(Room.from_yaml(example_path("concert_stage")), rays=30, seed=1)
    _no_echo_and_own_monitors(align, seats)


@pytest.mark.parametrize(
    ("example", "low_db", "high_db"),
    [("stadium_concert", 3.5, 10.5), ("concert_stage", 3.0, 9.0)],
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
    found = near.differences(_with_left_out(example))
    assert {"main_left", "subs"} <= set(found)
    assert "left_out" not in found and "left_out" not in found["main_left"]
    arrays = np.array([db for s, rows in found.items() if s != "subs" for db in rows.values()])
    low, high = np.abs(arrays[:, :6]).max(), np.abs(arrays[:, 6:]).max()
    assert 1.0 < low < low_db and high_db - 3.0 < high < high_db
    assert max(np.abs(db[:2]).max() for db in found["subs"].values()) < 0.6
    assert all(np.isnan(db[3:]).all() for db in found["subs"].values())
    monkeypatch.setattr(near, "differences", lambda room: found)
    assert near.main([]) == 0
    assert "main_left, far field above the elements added up (dB)" in capsys.readouterr().out


def test_the_stadium_examples_share_their_parts(tmp_path: Path) -> None:
    """stadium_concert has concert_stage's sources and listeners as they are, and the two
    share their scripts, their arrays and make_model.py, which writes each one's files as
    they are."""
    folders = [EXAMPLES / name for name in ("concert_stage", "stadium_concert")]
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
        stage, whole = (entries(folder, kind) for folder in folders)
        # Each of the stage's is the concert's, entry for entry: none drifts apart.
        assert {name: whole.get(name) for name in stage} == stage, kind
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


def test_the_stadium_crowds_scatter_as_measured() -> None:
    """The crowds of the stadium concert and of the concert stage scatter as a measured
    theatre audience does, little at 125 Hz and most from 1 kHz up, not at one value in
    every band."""
    measured = [0.3, 0.3, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7]
    for name in ("concert_stage", "stadium_concert"):
        scene = Scene.from_yaml(example_path(name))
        crowds = [m for m in scene.materials.values() if m.audience]
        assert crowds, name
        for material in crowds:
            assert material.scattering == pytest.approx(measured), (name, material)


def test_stadium_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The singer's microphone: its gain before feedback, where it rings, and a loop
    that dies away below unity gain and grows above it."""
    feedback = _script("concert_stage", "feedback")
    found = feedback.loop(_with_left_out("concert_stage"), rays=30, seed=1)
    assert "vocal_wedges" in found["speakers"] and "subs" not in found["speakers"]
    assert "left_out" not in found["speakers"]
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


def _demo(example: str, name: str):
    """The demo a script describes, loaded as the framework loads it."""
    return load(EXAMPLES / example / f"{name}.py")


def _demo_args(demo, **values) -> argparse.Namespace:
    """The demo's options at their defaults, with ``values`` over them (paths as paths)."""
    args = demo.defaults()
    for key, value in values.items():
        kind = demo.option(key).kind
        setattr(args, key, Path(value) if kind in PATH_KINDS else value)
    return demo.resolved(args)


@pytest.mark.parametrize("demo", list_demos(), ids=lambda demo: f"{demo.example}-{demo.name}")
def test_a_demo_plays_only_what_its_example_holds(demo, tmp_path: Path) -> None:
    """Every source and listener in the scenes a demo writes is its example's, where the
    example's scene has it: a demo brings none of its own, so the editor shows all it
    plays (what the example's own simulations leave out it ticks, as the stadium's band)."""
    folder = copy_examples(tmp_path, [demo.example])[0].parent
    demo = demo.at(folder)
    args = demo.resolved(demo.defaults())
    demo.run([demo.scene_from] if demo.scene_from else [], args)
    own = Scene.from_yaml(folder / "scene.yaml")
    items = {item.name: item for item in [*own.sources, *own.listeners]}
    # Beside the script, or in the run's folder for a scene that hangs on the run's
    # settings (a vehicle at --speed).
    written = sorted(folder.glob("scene_*.yaml")) + sorted(args.run_folder.glob("scene_*.yaml"))
    assert bool(written) == bool(demo.scene_from), written
    for path in written:
        scene = Scene.from_yaml(path)
        for item in [*scene.sources, *scene.listeners]:
            assert item.name in items, (path.name, item.name)
            assert item.position == items[item.name].position, (path.name, item.name)


def test_church_choir_and_organ_are_built_for_the_catalogue() -> None:
    """The church's choir and organ, a multitrack recording of the catalogue built from
    VocalSet's takes and VCSL's samples: the tune and every voice of the organ fill a
    verse, every key has a sample at most two semitones away, the two sides of the organ
    share the notes, and the recording has a stem for each source of the church's Choir
    and Organ pipes groups, each named after it and played by it, and the verses as its
    parts in time."""
    from acoustic_spaces.demos import dona_nobis_pacem as music

    assert sum(beats for _, beats in music.TUNE) == music.BEATS
    for voice, notes in music.ORGAN.items():
        assert sum(note[1] for note in notes) == music.BEATS, voice
    assert music.sample_name(60) == "C3"  # VCSL's names: C3 is middle C
    for division, voices in (("manual", "SAT"), ("pedal", "B")):
        for midi, start, end in music.organ_events(voices):
            assert 0.0 <= start < end <= music.BEATS
            for key in (midi, midi + 12) if division == "manual" else (midi,):
                assert abs(music.nearest_sample(key, division) - key) <= 2, (division, key)
    files = music.organ_needed()
    assert set(files) == {"Loud", "Quiet", "Loud Pedal", "Quiet Pedal"}
    assert all(name.endswith("_rr1.wav") for name in files["Quiet"])
    # Verse 3 slows down and holds the last chord; the music ends after the church rings.
    assert music.beat_time(3, music.BEATS) - music.VERSES[3] > music.BEATS
    assert music.end_of_music() > music.beat_time(3, 21.0) + music.LAST_CHORD_S
    item = recordings.named("dona_nobis_pacem")
    assert item.kind == "multitrack" and item.build == f"{music.__name__}:build"
    assert item.stem_names == tuple(music.SOURCE_NAMES) and len(item.stem_names) == 23
    assert item.download_mb == music.MUSIC_MB
    assert [start for start, _ in item.sections] == list(music.VERSES.values())
    church = Scene.from_yaml(example_path("church"))
    groups = {s.name: s.group for s in church.sources}
    for name in music.SOURCE_NAMES:
        assert groups[name] == ("Organ pipes" if name in music.ORGAN_SOURCES else "Choir")
    assert {s.name: s.channel for s in church.sources if s.channel} == {
        name: name for name in music.SOURCE_NAMES
    }
    assert church.recording == "dona_nobis_pacem"


def test_church_demo_is_a_preset_of_its_scene(tmp_path: Path) -> None:
    """church_demo.py is the demo of the church's scene with its choir and organ taken
    in (left out of the church's own simulations) and responses 9 s long; the scene's
    tour visits the script's six places at its times, to the end of the music, taking
    in the four the church leaves out; --listeners tours some of them in its order, the
    last to the end; the script's --tour form, steps and --from still read; --sources
    plays a pick of the choir and the organ, named after it, in turn too."""
    demo = _script("church", "church_demo")
    folder = copy_examples(tmp_path, ["church"])[0].parent
    parser = demo.DEMO.parser()
    args = demo.DEMO.resolved(
        parser.parse_args(
            ["all", "--from", str(folder / "scene.yaml"), "--work", str(tmp_path / "demo")]
        )
    )
    assert demo.DEMO.check(args) is None
    assert args.run_folder.name == "church_binaural"  # its own pick, its own defaults
    singers = [name for name in demo_music().SOURCE_NAMES if name.startswith("singer_")]
    assert list(args.playing) == demo_music().SOURCE_NAMES
    plays = scene_demo.plays(args)
    title = recordings.named("dona_nobis_pacem").title
    assert plays["singer_f5"] == f"{title}, its singer_f5 stem"
    assert plays["altar"] == "silent"
    run = Scene.from_yaml(scene_demo.run_scene_file(args))
    assert run.simulation.ir_length_s == 9.0
    assert len([s for s in run.sources if s.in_use]) == 25
    tour = ["nave-back", "side-aisle", "front-pews", "conductor", "in-the-choir", "nave-middle"]
    assert set(tour) <= {lis.name for lis in run.listeners if lis.in_use}
    stops = scene_demo.tour_given(args, run)
    length = demo_music().end_of_music()
    timed = scene_demo.timed_tour(stops, length, None)
    starts = [0.0, 11.5, 25.0, 37.5, 51.0, 63.5]
    assert timed == [
        (name, a, b) for name, a, b in zip(tour, starts, [*starts[1:], length], strict=True)
    ]
    # --listeners: those places in its order, each for its stop, the last to the end.
    two = demo.DEMO.resolved(
        argparse.Namespace(**vars(args) | {"listeners": "conductor,nave-back"})
    )
    assert scene_demo.timed_tour(scene_demo.tour_given(two, run), length, None) == [
        ("conductor", 0.0, 13.5),
        ("nave-back", 13.5, length),
    ]
    # The script's --tour (listener:start_s) and its steps still read.
    old = parser.parse_args(["video", "--tour", "nave-back:0,side-aisle:10,conductor:30"])
    assert old.tour == "nave-back:10s,side-aisle:20s,conductor:end"
    assert {s: demo.DEMO.step(s).name for s in ("music", "scene", "feeds")} == {
        "music": "recording",
        "scene": "recording",
        "feeds": "feeds",
    }
    # A pick of the choir and the organ, by group or by name, and in turn.
    choir = demo.DEMO.resolved(argparse.Namespace(**vars(args) | {"sources": "Choir"}))
    assert choir.run_folder.name == "church_Choir_binaural" and list(choir.playing) == singers
    turns = argparse.Namespace(**vars(args) | {"sources_play": "in-turn"})
    assert [part.run_folder.name for part in demo.DEMO.each(turns)] == [
        "church_Choir_binaural",
        "church_Organ-pipes_binaural",
    ]
    wav = tmp_path / "take.wav"
    wav.write_bytes(b"")
    assert "no such source plays here" in demo.DEMO.check(
        argparse.Namespace(**vars(choir) | {"inputs": {"organ_pedal_c": wav}})
    )
    # The church's own simulations keep the choir and the organ out, as before.
    church = Scene.from_yaml(folder / "scene.yaml")
    assert [item.name for item in [*church.sources, *church.listeners] if item.in_use] == [
        "altar",
        "organ",
        "nave-front",
        "nave-middle",
        "nave-back",
    ]


def demo_music():
    """The church's choir and organ, built (:mod:`acoustic_spaces.demos.dona_nobis_pacem`)."""
    from acoustic_spaces.demos import dona_nobis_pacem

    return dona_nobis_pacem


def test_church_choir_tunes_and_times_a_take() -> None:
    """A take sung slower than the tune, 30 cents sharp and each note a little off: the
    church's choir moves each note onto its beat, the singer's own offset away and 60 %
    of each note's error."""
    demo = demo_music()
    rate, beat_s, offset = 16_000, 0.8, 30.0
    errors = [(-1) ** k * 10.0 for k in range(sum(1 for m, _ in demo.TUNE if m is not None))]
    x, notes, t = np.zeros(int(22.0 * rate)), [], 1.0
    for midi, beats in demo.TUNE:
        if midi is not None:
            cents = offset + errors[len(notes)]
            f0 = 440.0 * 2.0 ** ((midi - 69) / 12 + cents / 1200)
            n = int((beats * beat_s - 0.05) * rate)
            a = int(t * rate)
            x[a : a + n] = np.sin(2.0 * np.pi * f0 * np.arange(n) / rate) * np.hanning(n) ** 0.1
            notes.append({"start": t, "end": t + n / rate, "f0": f0, "after": None})
        t += beats * beat_s
    sung = demo.tuned_and_timed("f1", x, rate, notes)
    first, y = sung[2]
    track = np.zeros(first + len(y))
    track[first:] = y
    sung_notes = [midi for midi, _ in demo.TUNE if midi is not None]
    times = demo.note_times(2)
    checked = 0
    for k, ((start, end), midi) in enumerate(zip(times, sung_notes, strict=True)):
        if end - start < 0.9:
            continue
        middle = int((start + end) / 2 * rate)
        part = track[middle - int(0.15 * rate) : middle + int(0.15 * rate)]
        spectrum = np.abs(np.fft.rfft(part * np.hanning(len(part)), 1 << 20))
        peak = int(np.argmax(spectrum))
        a, b, c = np.log(spectrum[peak - 1 : peak + 2])
        f = (peak + 0.5 * (a - c) / (a - 2.0 * b + c)) * rate / (1 << 20)
        cents = 1200.0 * np.log2(f / (440.0 * 2.0 ** ((midi - 69) / 12)))
        assert abs(cents - 0.4 * errors[k]) < 2.0, (k, cents)
        checked += 1
    assert checked >= 10


def test_stadium_demo_is_a_preset_of_its_scene(tmp_path: Path) -> None:
    """stadium_demo.py is the demo of the stadium's scene: every source the concert
    simulates plays, the sound system the master's channels as the audience hears them
    and the monitors mixes of the stems, its run named as the script named it; the
    script's --song, --music, --from, --tour (listener:phrases) and steps still read;
    --sources picks by name or group, a pick of silent sources alone (a song of yours
    without their stems) is refused, --input takes the sources that play, and
    --sources-play in-turn plays the sound system, the monitors and the band each in a
    part of its own; its stills leave out the view from the side unless --hide none."""
    demo = _demo("stadium_concert", "stadium_demo")
    folder = copy_examples(tmp_path, ["stadium_concert"])[0].parent
    parser = demo.parser()

    def run(*argv: str) -> argparse.Namespace:
        given = ["video", "--scene", str(folder / "scene.yaml"), "--work", str(tmp_path / "demo")]
        return parser.parse_args([*given, *argv])

    used = [s for s in Scene.from_yaml(folder / "scene.yaml").sources if s.in_use]
    assert len(used) == 22 and "helicopter" not in {s.name for s in used}
    every = demo.resolved(run())
    assert demo.check(run()) is None and every.run_folder.name == "stadium_binaural"
    assert every.playing == tuple(s.name for s in used)
    plays = scene_demo.plays(every)
    funk = '"Spicy Funk Cake", Rhythmussportgruppe'
    # The scene's *_right array, on the audience's left as it faces the stage, plays the
    # left channel.
    assert plays["main_left"] == f"{funk}, right channel"
    assert plays["main_right"] == f"{funk}, left channel"
    assert plays["drum_kit"] == f"{funk}, its drums stem"
    assert plays["vocal_wedges"].startswith(f"{funk}, a mix of its stems vocal, backing -6 dB")
    # The script's options and steps.
    assert demo.resolved(run("--song", "francium")).run_folder.name == "stadium_francium_binaural"
    assert Path(run("--from", str(folder / "other.yaml")).scene) == folder / "other.yaml"
    assert run("--tour", "mix_position:1,pitch_side:0,singer:2").tour == "mix_position:1,singer:end"
    assert [demo.step(name).name for name in ("music", "band")] == ["recording", "simulate"]
    assert run().hide == "side" and run("--hide", "none").hide == "none"
    # The sources picked.
    pa = demo.resolved(run("--sources", "PA"))
    assert pa.run_folder.name == "stadium_PA_binaural"
    assert pa.playing == tuple(s.name for s in used if s.group.startswith("PA/"))
    picked = demo.resolved(run("--sources", "Band,PA/Mains"))
    assert picked.run_folder.name == "stadium_Band+PA-Mains_binaural"
    assert sorted(picked.playing) == [
        "drum_kit",
        "guitar_amp",
        "main_left",
        "main_right",
        "singer_voice",
    ]
    wav = tmp_path / "mine.wav"
    write_wav(wav, np.zeros(800), 8000)
    assert demo.check(run("--sources", "PA", "--input", f"drum_kit={wav}")).startswith(
        "--input drum_kit: no such source plays here; it has main_left, main_right, subs"
    )
    assert demo.check(run("--sources", "flute")).startswith(
        "--sources flute: no such source or group here; it has main_left"
    )
    song = tmp_path / "song.wav"
    write_wav(song, np.zeros((800, 2)), 8000)
    own = ("--music", str(song), "--bpm", "120")
    assert demo.check(run(*own, "--sources", "Monitors,drum_kit")).startswith(
        "--sources picks only sources that play nothing (vocal_wedges, drum_fill, "
        "side_fill_left, side_fill_right, drum_kit); those that play: main_left"
    )
    assert demo.check(run(*own, "--sources", "Monitors,PA/Subwoofers")) is None
    vocal = (*own, "--stem", f"vocal={wav}")
    assert demo.check(run(*vocal, "--sources", "singer_voice")) is None
    assert scene_demo.plays(demo.resolved(run(*own)))["vocal_wedges"].startswith("silent: no ")
    turns = demo.each(run("--sources-play", "in-turn"))
    assert [part.run_folder.name for part in turns] == [
        f"stadium_{group}_binaural" for group in ("PA", "Monitors", "Band")
    ]


def test_orchestra_symphony_is_a_recording_of_three_movements() -> None:
    """Beethoven's 8th, a multitrack recording of the catalogue in three pieces (the
    movements I, II and IV the TU Berlin recorded, each read out of its zip), a stem for
    each of the 61 players named after the orchestra's source that plays it, from the
    track auralize.sh gives that source; the trumpets and the timpani rest through II;
    the orchestra's sources have their sections as groups, and each plays its stem.
    Each piece's stems are in a folder of their own, its master beside it."""
    from acoustic_spaces.demos import beethoven_8

    item = recordings.named("beethoven_8")
    assert item.kind == "multitrack" and item.build == f"{beethoven_8.__name__}:build"
    assert item.piece_names == ("I", "II", "IV") == tuple(beethoven_8.ZIPS)
    archives = [piece.archive for piece in item.pieces]
    assert archives == ["movement1.zip", "movement2.zip", "movement4.zip"]
    assert item.download_mb == pytest.approx(sum(p.download_mb for p in item.pieces), abs=1)
    assert item.piece("II").rests == ("tp1", "tp2", "timpani") and item.cut == "quiet"
    assert item.stem_names == tuple(beethoven_8.TRACKS) and len(item.stem_names) == 61
    script = (EXAMPLES / "orchestra" / "auralize.sh").read_text(encoding="utf-8")
    pairs = dict(re.findall(r"^\s+(\w+)=(\d\d_\w+\.wav)$", script, re.MULTILINE))
    assert pairs == beethoven_8.TRACKS
    orchestra = Scene.from_yaml(example_path("orchestra"))
    assert orchestra.recording == "beethoven_8"
    assert {s.name: s.channel for s in orchestra.sources} == {n: n for n in item.stem_names}
    groups = [s.group for s in orchestra.sources]
    assert groups.count("Strings/Violins I") == 14 and groups.count("Winds/Flutes") == 2
    assert {group.split("/")[0] for group in groups} == {"Strings", "Winds", "Brass", "Percussion"}
    lib = Path("lib")
    assert recordings.stem_file(item, "fl1", lib, "IV") == lib / "beethoven_8" / "IV" / "fl1.wav"
    assert recordings.file_of(item, lib, "IV") == lib / "beethoven_8" / "IV.wav"
    assert recordings.file_of(item, lib) == lib / "beethoven_8" / "I.wav"
    with pytest.raises(KeyError, match="it has I, II, IV"):
        recordings.file_of(item, lib, "III")


def test_orchestra_movements_are_built_from_their_zips(tmp_path: Path, monkeypatch) -> None:
    """A movement's 61 stems are its tracks as recorded (not the zip's mix), read out of
    its zip on the TU Berlin's server by range requests, or out of a copy downloaded
    before (the zip, the folder it was unpacked into, or the tracks as the orchestra's
    script kept them); its master is their sum; a build cut short goes on where it
    stopped; an unpacked copy short of a track says so rather than downloading the
    movement."""
    from acoustic_spaces.demos import beethoven_8
    from acoustic_spaces.demos.audio import read, write

    item = recordings.named("beethoven_8")
    rate, served, tracks = 8000, {}, tmp_path / "tracks"
    for k, (piece, url) in enumerate(beethoven_8.ZIPS.items()):
        path = tmp_path / "server" / item.piece(piece).archive
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            for n, track in enumerate(beethoven_8.TRACKS.values()):
                wav = tracks / piece / track
                wav.parent.mkdir(parents=True, exist_ok=True)
                write(wav, np.full(80 + n, (k + 1) * 1e-3 + n * 1e-5), rate)
                archive.write(wav, f"anechoic/{track}")
            archive.writestr(f"mix/movement{k}_mix.wav", "the mix")
        served[url] = path
    opened: list[tuple[str, int]] = []

    def open_zip(url: str, block: int) -> zipfile.ZipFile:
        opened.append((url, block))
        return zipfile.ZipFile(served[url])

    monkeypatch.setattr(beethoven_8.fetch, "open_zip", open_zip)
    lib = tmp_path / "lib"
    stems = recordings.get_stems(item, lib, "II")
    assert opened == [(beethoven_8.ZIPS["II"], beethoven_8.BLOCK)]
    for name, track in beethoven_8.TRACKS.items():
        assert stems[name].read_bytes() == (tracks / "II" / track).read_bytes(), name
    master, got = read(recordings.file_of(item, lib, "II"))
    played = [read(path)[0][:, 0] for path in stems.values()]
    assert got == rate and len(master) == 80 + 60
    assert master[:80, 0] == pytest.approx(sum(x[:80] for x in played), abs=1e-6)
    # Cut short: the stems left come, those there are kept.
    stems["vc_3"].unlink()
    recordings.file_of(item, lib, "II").unlink()
    kept = stems["fl1"].stat().st_mtime_ns
    opened.clear()
    recordings.get(item, lib, "II")
    assert opened == [(beethoven_8.ZIPS["II"], beethoven_8.BLOCK)]
    assert stems["vc_3"].exists() and stems["fl1"].stat().st_mtime_ns == kept
    # Copies downloaded before: the zip, or the folder it was unpacked into.
    copies = tmp_path / "zips"
    copies.mkdir()
    shutil.copyfile(served[beethoven_8.ZIPS["IV"]], copies / "movement4.zip")
    with zipfile.ZipFile(served[beethoven_8.ZIPS["I"]]) as archive:
        archive.extractall(copies / "movement1")
    assert recordings.copy_of(item, "IV", copies) == copies / "movement4.zip"
    assert recordings.copy_of(item, "I", copies) == copies / "movement1"
    assert recordings.copy_of(item, "II", copies) is None
    opened.clear()
    for piece in ("IV", "I"):
        stems = recordings.get_stems(item, lib, piece, copies)
        assert stems["db_6"].read_bytes() == (tracks / piece / "61_Db_6.wav").read_bytes()
        assert recordings.file_of(item, lib, piece).exists()
    assert opened == []
    (copies / "movement1" / "anechoic" / "13_Tp.wav").unlink()
    recordings.stem_file(item, "timpani", lib, "I").unlink()
    with pytest.raises(SystemExit, match="has no 13_Tp.wav"):
        recordings.get_stems(item, lib, "I", copies)
    assert opened == []
    # The tracks as the script kept them before 4.4.0 (demo/tracks/movement2).
    old = tmp_path / "demo" / "tracks"
    shutil.copytree(tracks / "II", old / "movement2")
    shutil.rmtree(recordings.stems_folder(item, lib) / "II")
    recordings.get_stems(item, lib, "II", old)
    assert opened == [] and recordings.stems_there(item, lib, "II")


def test_orchestra_demo_is_a_preset_of_its_scene(tmp_path: Path) -> None:
    """orchestra_demo.py is the demo of the orchestra's scene: each movement the scene's
    tour visits in a part of its own, in the tour's order, each at its seats; the
    script's --movements, --tour (movement:seat), --zips and tracks step still read;
    --listeners plays each movement at the seats of its tour among those, in their
    order, or at all of them where its tour has none; a pick of players leaves out a
    movement they all rest through, unless --input gives one of them a file; a movement
    or seat the scene lacks is refused; the recording step downloads a movement unless
    a copy of its zip is in --zips."""
    demo = _script("orchestra", "orchestra_demo")
    folder = copy_examples(tmp_path, ["orchestra"])[0].parent
    parser = demo.DEMO.parser()

    def run(*argv: str) -> argparse.Namespace:
        given = ["video", "--scene", str(folder / "scene.yaml"), "--work", str(tmp_path / "demo")]
        return parser.parse_args([*given, *argv])

    def toured(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
        parts = demo.DEMO.each(args)
        return [
            (part.piece, [stop.listener for stop in scene_demo.tour_given(part, run_of(part))])
            for part in parts
        ]

    def run_of(part: argparse.Namespace) -> Scene:
        return scene_demo.scene_of(part)

    every = run()
    assert demo.DEMO.check(every) is None
    assert demo.DEMO.resolved(every).run_folder.name == "orchestra_binaural"
    tour = [
        ("I", ["stalls-middle", "stalls-front", "conductor"]),
        ("II", ["stalls-rear", "stalls-middle"]),
        ("IV", ["conductor", "stalls-front", "stalls-rear", "stalls-middle"]),
    ]
    assert toured(every) == tour
    assert [part.run_folder.name for part in demo.DEMO.each(every)] == [
        f"orchestra_piece-{piece}_binaural" for piece in ("I", "II", "IV")
    ]
    (first,) = [part for part in demo.DEMO.each(every) if part.piece == "IV"]
    title = recordings.named("beethoven_8").title
    plays = scene_demo.plays(first)
    assert len(plays) == 61 and plays["fl1"] == f"{title}, IV, its fl1 stem"
    # The script's options and steps.
    old = run("--movements", "4,1", "--tour", "1:conductor,4:stalls-rear,4:conductor")
    assert old.piece == ("IV", "I") and old.tour == "I/conductor,IV/stalls-rear,IV/conductor"
    assert toured(old) == [("IV", ["stalls-rear", "conductor"]), ("I", ["conductor"])]
    assert demo.DEMO.defaults().zips == Path(".")
    assert demo.DEMO.step("tracks").name == "recording"
    # --listeners: the seats of each movement's tour among them, or all of them.
    picked = run("--listeners", "stalls-rear,conductor")
    assert toured(picked) == [
        ("I", ["conductor"]),
        ("II", ["stalls-rear"]),
        ("IV", ["stalls-rear", "conductor"]),
    ]
    assert toured(run("--listeners", "stalls-front"))[1] == ("II", ["stalls-front"])
    # Some players: by section, the movements they all rest through left out.
    winds = run("--sources", "Winds")
    wind = ["bn1", "bn2", "cl1", "cl2", "fl1", "fl2", "ob1", "ob2"]
    assert sorted(demo.DEMO.each(winds)[0].playing) == wind
    assert [piece for piece, _ in toured(winds)] == ["I", "II", "IV"]
    assert [piece for piece, _ in toured(run("--sources", "Percussion/Timpani"))] == ["I", "IV"]
    wav = tmp_path / "take.wav"
    wav.write_bytes(b"")
    fed = run("--sources", "tp1", "--input", f"tp1={wav}")
    assert [piece for piece, _ in toured(fed)] == ["I", "II", "IV"]
    assert demo.DEMO.check(run("--piece", "II", "--sources", "Brass/Trumpets")) == (
        "II: every source that plays rests through it (tp1, tp2); pick others (--sources), "
        "or give one a file of yours (--input)"
    )
    for argv, refused in [
        (("--movements", "3"), "--piece 3: beethoven_8 has the pieces I, II, IV"),
        (("--tour", "3:conductor"), "the tour's stop at conductor is in the piece '3'"),
        (("--tour", "1:balcony"), "the tour stops at 'balcony': no such listener here"),
        (("--zips", str(tmp_path / "none")), "--zips: no folder"),
    ]:
        assert refused in demo.DEMO.check(run(*argv)), argv
    # The download, but for a movement whose zip is in --zips.
    step = demo.DEMO.step("recording")
    (one,) = demo.DEMO.each(run("--piece", "II"))
    assert step.downloads(one) == (
        f"what II of {title} is built from (CC BY 4.0, 605 MB), built into its 61 stems and "
        "kept for every scene"
    )
    zips = tmp_path / "zips"
    zips.mkdir()
    (zips / "movement2.zip").write_bytes(b"")
    (copied,) = demo.DEMO.each(run("--piece", "II", "--zips", str(zips)))
    assert step.downloads(copied) is None


def test_orchestra_pictures_show_the_players_that_play() -> None:
    """The orchestra's stills light up and count the players that play, the others empty
    chairs, and say so; with every one playing they are as before."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import sibling

    pictures = sibling(EXAMPLES / "orchestra" / "orchestra_demo.py", "make_images")
    scene = yaml.safe_load(example_path("orchestra").read_text(encoding="utf-8"))
    every = pictures.Pictures(scene, "stalls-middle", 1)
    assert sum(every.counts.values()) == 61 and every.counts["Violins I"] == 14
    assert every.played == "61 anechoic tracks"
    winds = ["fl1", "fl2", "ob1", "ob2", "cl1", "cl2", "bn1", "bn2"]
    some = pictures.Pictures(scene, "conductor", 4, heard=winds)
    assert some.counts == {"Flutes": 2, "Oboes": 2, "Clarinets": 2, "Bassoons": 2}
    assert some.played == "8 anechoic tracks"
    assert some.cover().size == (1920, 1080) and some.thumbnail().getbbox() is not None
    assert (
        every.cover().tobytes()
        != pictures.Pictures(scene, "stalls-middle", 1, heard=winds).cover().tobytes()
    )


def test_the_stills_draw_the_audience() -> None:
    """The stills draw a scene's audience where its materials say: the stadium's
    standing on the pitch, seated in three stands and the stand behind the stage as
    empty seats, as many people as each material's preset puts on a square metre, with
    how much each takes up at 1 kHz; the concert hall's seated in its chairs."""
    from acoustic_spaces.demos import audiences
    from acoustic_spaces.materials.database import builtin_database

    found = audiences.audiences(Room.from_yaml(example_path("stadium_concert")))
    kinds: dict[str, list] = {}
    for area in found:
        kinds.setdefault(area.kind, []).append(area)
    assert {kind: len(areas) for kind, areas in kinds.items()} == {
        "standing": 1,
        "seated": 3,
        "seats": 1,
    }
    (standing,) = kinds["standing"]
    assert standing.people == 12_308  # 90.5 by 68 m, two on a square metre
    assert sum(area.people for area in kinds["seated"]) == pytest.approx(29_000, rel=0.01)
    assert standing.absorbs == pytest.approx(0.99)
    assert kinds["seats"][0].absorbs == pytest.approx(0.2) and kinds["seats"][0].people == 0
    # Each dot stands for DOT_SPACING_M squared over the people on a square metre.
    dots = standing.dots(seed=0)
    assert len(dots) == pytest.approx(90.5 * 68 * 2 / audiences.DOT_SPACING_M**2, rel=0.05)
    assert audiences._inside(dots, standing.triangles).all()
    assert audiences.audience_lines(found) == [
        "Audience: 12,300 standing, 29,000 seated, empty seats",
        "Absorbing at 1 kHz: standing 99%, seated 95%, empty seats 20%",
    ]
    # In a room: the concert hall's audience in chairs 0.85 by 0.63 m.
    hall = audiences.audiences(Room.from_yaml(example_path("concert_hall")))
    assert [(area.kind, round(area.per_m2, 2)) for area in hall] == [("seated", 1.87)]
    entry = builtin_database().get
    assert audiences.per_m2_of(entry("orchestra_1.5_m2")) == pytest.approx(1 / 1.5)
    assert audiences.per_m2_of(entry("audience_0.72_m2")) == 0.72
    assert audiences.per_m2_of(entry("audience_wooden_chairs_1_m2")) == 1.0
    assert audiences.per_m2_of(entry("chairs_wooden")) == 2.0  # its category's
    assert audiences.audience_lines([]) == []


def test_the_crossings_stills_frame_the_stadium(tmp_path: Path) -> None:
    """The crossing's helicopter flies in from far past the stadium's ground, which its
    simulation widens to take it: the stills frame the ground as the scene has it, the
    helicopter past it an arrowhead on the panel's edge with how far it is from the
    listener, over the stadium a mark of its own; the stadium demo's, with nothing that
    moves, frame all of it."""
    pytest.importorskip("PIL")
    from PIL import ImageDraw

    from acoustic_spaces.demos import scene_pictures
    from acoustic_spaces.room import outline_edges
    from acoustic_spaces.walkthrough import knots, movers

    folder = copy_examples(tmp_path, ["stadium_concert"])[0].parent
    crossing = _demo("stadium_concert", "helicopter_demo")
    stadium = _demo("stadium_concert", "stadium_demo")

    def run(demo, *argv: str) -> argparse.Namespace:
        given = ["video", "--scene", str(folder / "scene.yaml"), "--work", str(tmp_path / "demo")]
        return demo.resolved(demo.parser().parse_args([*given, *argv]))

    args = run(crossing)
    scene = scene_demo.moving_scene(args)
    assert scene.geometry.ground.extent == ((-650.0, -82.0), (650.0, 82.0))
    frame = scene_demo.frame_of(args, scene)
    assert frame == (-100.0, -82.0, 100.0, 82.0)
    still = run(stadium)
    assert scene_demo.frame_of(still, scene_demo.moving_scene(still)) is None
    room = Room.from_scene(scene)
    stops = [pipeline.TourStop("mix_position", 0.0, 0.0, 200.0)]
    drawing = scene_pictures.Pictures(
        scene.to_data(),
        outline_edges(room)[0],
        {"responses": []},
        {},
        stops,
        playing=list(args.playing),
        paths={(kind, item.name): knots(item) for kind, item in movers(scene)},
        room=room,
        hide=["side"],
        frame=frame,
    )
    assert drawing.framed and drawing.box == (-110.0, -92.0, 110.0, 92.0)
    plan = drawing.plan(*scene_pictures.PLAN_BOX)
    area = drawing.clip_of(plan)
    data = {s["name"]: s for s in scene.to_data()["sources"]}
    listener = {lis.name: lis for lis in scene.listeners}["mix_position"]
    far = drawing.placed("sources", data["helicopter"], 80.0)  # still where it sets off
    assert far["position"] == [-600.0, 0.0, 150.0]
    draw = ImageDraw.Draw(scene_pictures.canvas())
    (mark, words, sides) = drawing.draw_beyond(
        draw, plan, area, [(far, scene_pictures.SOURCE)], listener.model_dump()
    )[0]
    away = math.dist(far["position"], listener.position)
    assert words == f"helicopter, {away:,.0f} m away" and sides[0] == "right"
    assert area[0] <= mark[0] < mark[2] <= area[0] + 40  # on the panel's left edge
    over = drawing.placed("sources", data["helicopter"], 97.5)
    assert abs(over["position"][0]) < 1.0  # over the centre spot, inside the frame
    assert drawing.still(0, 80.0).tobytes() != drawing.still(0, 97.5).tobytes()


def test_demo_scripts_take_an_hrtf(tmp_path: Path) -> None:
    """Every demo's simulate command carries ``--hrtf``; its mixes then ask for the
    responses made through that file (``binaural-<name>``) and refuse ones made through
    the spherical head, and without it the other way round (so ``all`` simulates again);
    other formats are left alone; and asked for mixes before the
    scene and the simulation exist, a demo says which step writes the scene."""
    sofa = tmp_path / "me.sofa"
    sofa.write_bytes(b"")  # there, as the check asks
    plain = argparse.Namespace(format="binaural", hrtf=None)
    own = argparse.Namespace(format="binaural", hrtf=sofa)
    assert pipeline.output_label(plain) == "binaural"
    assert pipeline.output_label(own) == "binaural-me"
    assert pipeline.output_label(argparse.Namespace(format="ambix3", hrtf=sofa)) == "ambix3"
    assert pipeline.hrtf_label("binaural", None) == "binaural"

    for demo in (_demo(e, f"{n}_demo") for e, n in DEMO_SCRIPTS):
        args = _demo_args(demo, work=tmp_path / demo.name)
        with_hrtf = argparse.Namespace(**vars(args) | {"hrtf": sofa})
        assert demo.check(with_hrtf) is None
        if "format" in {o.name for o in demo.options}:
            wrong = argparse.Namespace(**vars(with_hrtf) | {"format": "ambix3"})
            assert demo.check(wrong) == "--hrtf is for binaural mixes"
        for simulation in demo.simulations(args):
            assert "--hrtf" not in pipeline.simulate_command(simulation)
            assert pipeline.simulate_command(simulation, sofa).endswith(f"--hrtf {sofa}")
        # No simulation yet: the error names the step that writes the scene.
        for simulation in demo.simulations(with_hrtf):
            with pytest.raises(SystemExit) as caught:
                pipeline.responses_of(simulation, with_hrtf, demo.first_step(args))
            assert "acoustic-spaces simulate" in str(caught.value)
            if demo.scene_from:
                assert f"{demo.script.name} {demo.scene_from}" in str(caught.value)
        # Responses through a spherical head cannot pass for ones through the file.
        results = tmp_path / demo.name / "asked"
        head, ears = (
            [{"listener": "a", "source": "s", "outputs": [{"format": fmt, "wav": "a.wav"}]}]
            for fmt in ("binaural", "binaural-me")
        )
        for entries, asked, fine in (
            (head, args, True),
            (ears, with_hrtf, True),
            (head, with_hrtf, False),
            (ears, args, False),  # nor through the file for ones through the head
        ):
            pipeline.write_json(results / pipeline.PARAMETERS, {"responses": entries})
            output = pipeline.output_label(argparse.Namespace(**vars(asked) | {"hrtf": None}))
            simulation = pipeline.Simulation(tmp_path / "scene.yaml", results, output)
            assert pipeline.simulated(simulation, asked) == fine
            if fine:
                assert pipeline.responses_of(simulation, asked) == entries
            else:
                said = "me.sofa" if asked is with_hrtf else r"spherical head model \(binaural-me\)"
                with pytest.raises(SystemExit, match=said):
                    pipeline.responses_of(simulation, asked)


def _lake(folder: Path, *argv: str):
    """lakeside_demo.py at the copy of its example in ``folder``, and its command line
    ``video`` with ``argv``, as given and resolved."""
    demo = _script("lakeside_concert", "lakeside_demo").DEMO.at(folder)
    args = demo.parser().parse_args(["video", *argv])
    return demo, args, demo.resolved(args)


def test_the_lakeside_demo_is_a_preset_of_the_demo_of_its_scene(tmp_path: Path) -> None:
    """lakeside_demo.py is the demo of the lake's scene: the scene plays the stage's sound
    system and hears the place upwind behind the stage, its responses 3.5 s past the far
    shore's direct sound, with the example's seed; its tour is the scene's, 8 bars a
    stop, its last two phrases long, the bedroom window in the evening's weather, in
    still air (the scene without its atmosphere, simulated beside it) and in the
    evening's again; each command line of the script it was reads as the demo of any
    scene's, and its steps run as those that took them over."""
    from acoustic_spaces.demos import music, run_folder, scene_demo

    folder = copy_examples(tmp_path, ["lakeside_concert"])[0].parent
    demo, args, resolved = _lake(folder)
    assert (demo.name, demo.example, demo.title) == (
        "lakeside",
        "lakeside_concert",
        "A concert across a lake",
    )
    assert demo.check(args) is None
    scene = scene_demo.scene_of(resolved)
    assert [s.name for s in scene.sources if s.in_use] == ["main_left", "main_right", "subs"]
    names = [lis.name for lis in scene.listeners if lis.in_use]
    assert names == ["audience", "boat", "garden", "bedroom_window", "behind_stage"]
    assert scene.simulation.ir_length_s == 3.5 and scene.simulation.seed == 1
    assert scene_demo.plays(resolved) == {
        "main_left": '"Spicy Funk Cake", Rhythmussportgruppe, left channel',
        "main_right": '"Spicy Funk Cake", Rhythmussportgruppe, right channel',
        "subs": '"Spicy Funk Cake", Rhythmussportgruppe, mixed down to mono',
    }
    # Still air: a second simulation, of the scene without its atmosphere.
    assert scene_demo.still_stops(resolved) == ["bedroom_window"]
    first, second = scene_demo.simulations(resolved)
    assert first.scene == folder / "scene.yaml"
    still = Scene.from_yaml(second.scene)
    assert second.scene == folder / "demo" / "scenes" / "scene_still-air.yaml"
    assert still.atmosphere is None and scene.atmosphere is not None
    assert still.simulation.seed == 1  # the same draws in both: the weather's difference
    phrase = music.phrase_s({"bpm": 110})  # Spicy Funk Cake's, from its first downbeat
    downbeat = 0.5

    def tour(*argv: str, length_s: float = 195.0) -> list[tuple[str, float, float | str]]:
        """Each stop's mix and its cuts, in phrases from the first downbeat; the music's
        start 0 and its end "end"."""
        resolved = _lake(folder, *argv)[2]
        scene = scene_demo.scene_of(resolved)
        stops = scene_demo.tour_given(resolved, scene)
        tempo = scene_demo.tempo_of(resolved, scene)
        assert tempo == (110.0, downbeat)
        timed = scene_demo.timed_tour(stops, length_s, tempo)

        def cut(t: float) -> float | str:
            if t == length_s:
                return "end"
            return 0 if t == 0 else round((t - downbeat) / phrase, 6)

        return [(key, cut(a), cut(b)) for key, a, b in timed]

    assert tour() == [
        ("audience", 0, 2),
        ("behind_stage", 2, 3),
        ("boat", 3, 4),
        ("garden", 4, 5),
        ("bedroom_window", 5, 6),
        ("bedroom_window.still", 6, 7),
        ("bedroom_window", 7, 8),
        ("audience", 8, 10),
    ]
    # --listeners: those places alone, in that order, each with its stops of the tour,
    # and back to the first at the end; in the tour's own order, the tour itself.
    assert tour("--listeners", "bedroom_window,audience") == [
        ("bedroom_window", 0, 1),
        ("bedroom_window.still", 1, 2),
        ("bedroom_window", 2, 3),
        ("audience", 3, 5),
        ("bedroom_window", 5, 7),
    ]
    assert [key for key, _, _ in tour("--listeners", "audience,bedroom_window")] == [
        "audience",
        "bedroom_window",
        "bedroom_window.still",
        "bedroom_window",
        "audience",
    ]
    every = "audience,behind_stage,boat,garden,bedroom_window"
    assert tour("--listeners", every) == tour()
    assert tour("--listeners", "audience") == [("audience", 0, 4)]
    # The script's --tour (listener:weather:phrases, one phrase by default, the last stop
    # two) and --last-phrases (0 to the end); --listeners thins out a --tour given.
    assert tour("--tour", "bedroom_window:still:1,audience", "--last-phrases", "0") == [
        ("bedroom_window.still", 0, 1),
        ("audience", 1, "end"),
    ]
    assert tour("--tour", "boat:evening:2,garden::1,audience:evening") == [
        ("boat", 0, 2),
        ("garden", 2, 3),
        ("audience", 3, 5),
    ]
    thinned = ("--tour", "bedroom_window:still:1,audience,boat")
    assert tour(*thinned, "--listeners", "audience,bedroom_window") == [
        ("bedroom_window.still", 0, 1),
        ("audience", 1, 3),  # the last stop, two phrases
    ]
    none = _lake(folder, "--tour", "bedroom_window:still:1", "--listeners", "boat")
    assert "--tour stops at none of the listeners" in none[0].check(none[1])
    parser = demo.parser()
    read = {
        ("all", "--song", "potassium"): {"recording": "potassium"},
        ("all", "--from", "other.yaml"): {"scene": "other.yaml"},
        ("all", "--tour", "boat:evening:2,garden"): {"tour": "boat::2,garden::1"},
        ("all", "--tour", "boat:2,garden:end"): {"tour": "boat:2,garden:end"},
        ("all", "--last-phrases", "0"): {"last_stop": "end"},
        ("all", "--last-phrases", "3"): {"last_stop": "3"},
        ("all", "--sources", "subs"): {"sources": "subs"},
    }
    for argv, given in read.items():
        args = parser.parse_args(list(argv))
        assert {name: getattr(args, name) for name in given} == given, argv
    assert demo.steps_named("music") == ["recording", "feeds"]
    assert demo.steps_named("scenes") == ["simulate"]
    assert [demo.step(name).name for name in ("render", "video")] == ["render", "video"]
    general = scene_demo.DEMO.at(folder)
    same = general.resolved(general.parser().parse_args(["all", "--scene", "scene.yaml"]))
    # Its runs are named after it, the scene's demo's after the scene: their defaults
    # differ (--last-stop), so neither finds the other's mixes as its own.
    assert run_folder(resolved) == folder / "demo" / "lakeside_binaural"
    assert run_folder(same) == folder / "demo" / "scene_binaural"
    script = _script("lakeside_concert", "lakeside_demo").DEMO  # in the example's folder
    assert script.command("all", parser.parse_args(["all"])) == "python lakeside_demo.py all"
    assert script.command("all", parser.parse_args(["all", "--song", "potassium"])).endswith(
        "all --recording potassium"
    )


def test_the_lakeside_demo_plays_a_song_of_your_own(tmp_path: Path, monkeypatch) -> None:
    """``--music`` (``--recording``) plays a song of yours across the lake in place of the
    scene's: read, not downloaded, each loudspeaker its channel of it at one level for
    both channels, the tour moving every 8 bars of ``--bpm``; a song with a tempo of its
    own takes no --bpm, a file that is not there is refused, and one ffmpeg would decode
    needs ffmpeg."""
    import shutil

    from acoustic_spaces.demos import run_folder, scene_demo

    folder = copy_examples(tmp_path, ["lakeside_concert"])[0].parent
    rate = 8000
    t = np.arange(40 * rate) / rate
    song = np.stack([np.sin(2 * np.pi * 220 * t), 0.5 * np.sin(2 * np.pi * 330 * t)], axis=1)
    write_wav(folder / "mine.wav", 0.1 * song, rate)
    demo, args, resolved = _lake(folder, "--music", "mine.wav", "--bpm", "100")
    assert demo.check(args) is None
    assert resolved.recording == folder / "mine.wav"
    assert run_folder(resolved).name == "lakeside_mine_bpm-100_binaural"
    assert scene_demo.plays(resolved)["subs"] == "mine.wav, mixed down to mono"
    scene = scene_demo.scene_of(resolved)
    assert scene_demo.tempo_of(resolved, scene) == (100.0, 0.0)
    stops = scene_demo.tour_given(resolved, scene)
    timed = scene_demo.timed_tour(stops, 40.0, (100.0, 0.0))
    assert timed == [("audience", 0.0, pytest.approx(2 * 19.2)), ("behind_stage", 38.4, 40.0)]
    assert demo.main(["music", "--music", "mine.wav", "--bpm", "100"]) == 0  # nothing to download
    table = pipeline.read_json(scene_demo.feeds_folder(resolved) / scene_demo.FEEDS)
    assert {name: entry["channel"] for name, entry in table["sources"].items()} == {
        "main_left": "left",
        "main_right": "right",
        "subs": "mono",
    }
    assert {Path(entry["signal"]).name for entry in table["sources"].values()} == {"mine.wav"}
    preset = _lake(folder, "--bpm", "100")
    assert preset[0].check(preset[1]) == (
        "--bpm is for a recording of yours: spicy_funk_cake has its own, 110 bpm"
    )
    missing = _lake(folder, "--music", "none.wav")
    assert "--recording none.wav: no such file" in missing[0].check(missing[1])
    (folder / "mine.mp3").write_bytes(b"")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    compressed = _lake(folder, "--music", "mine.mp3", "--bpm", "100")
    assert compressed[0].check(compressed[1]) == (
        "main_left plays mine.mp3: ffmpeg is needed to decode it, and is not on the PATH"
    )


def test_the_lakeside_demo_plays_the_loudspeakers_picked(tmp_path: Path) -> None:
    """``--sources`` picks the loudspeakers that play, by name or as the PA group, all
    three by default; the stage's single source, left out of the scene, plays when it is
    picked, taken in; an ``--input`` for a loudspeaker left out is refused."""
    from acoustic_spaces.demos import run_folder

    folder = copy_examples(tmp_path, ["lakeside_concert"])[0].parent
    demo, args, every = _lake(folder)
    assert [(s.name, s.group, s.on) for s in demo.source_list(every)] == [
        ("main_left", "PA", True),
        ("main_right", "PA", True),
        ("subs", "PA", True),
        ("stage", None, False),
    ]
    assert every.playing == ("main_left", "main_right", "subs")
    assert run_folder(_lake(folder, "--sources", "PA")[2]) == run_folder(every)
    subs = _lake(folder, "--sources", "subs")[2]
    assert subs.playing == ("subs",) and run_folder(subs).name == "lakeside_subs_binaural"
    stage = _lake(folder, "--sources", "stage")
    assert stage[0].check(stage[1]) is None and stage[2].playing == ("stage",)
    assert run_folder(stage[2]).name == "lakeside_stage_binaural"
    taken = Scene.from_yaml(scene_demo.run_scene_file(stage[2]))
    assert [s.name for s in taken.sources if s.in_use] == ["stage", *every.playing]
    tower = _lake(folder, "--sources", "tower")
    assert tower[0].check(tower[1]) == (
        "--sources tower: no such source or group here; it has main_left, main_right, subs, "
        "stage, PA"
    )
    (folder / "bass.wav").write_bytes(b"")
    fed = _lake(folder, "--sources", "subs", "--input", "main_left=bass.wav")
    assert fed[0].check(fed[1]) == "--input main_left: no such source plays here; it has subs"


def test_the_passby_demo_is_a_preset_of_the_demo_of_its_scene(tmp_path: Path) -> None:
    """passby_demo.py is the demo of the roadside's scene with its car taken in, the
    vehicles each in a part of their own: each command line of the script it was reads
    as the demo of any scene's (the vehicle as the sources that play, the example's
    scene as the scene, --speed, --rpm and --input as they were), its steps run as
    those that took them over, and --rpm says it is moot where no sound is made."""
    from acoustic_spaces.demos import run_folder
    from acoustic_spaces.demos.scene_demo import DEMO

    road = _script("roadside", "passby_demo").DEMO
    assert (road.name, road.example, road.title) == (
        "passby",
        "roadside",
        "A car or a truck passing the meadow",
    )
    parser = road.parser()
    read = {
        ("all", "--vehicle", "truck"): {"sources": "truck"},
        ("all", "--vehicle", "car,truck"): {"sources": "car,truck"},
        ("all", "--from", "other.yaml"): {"scene": "other.yaml"},
        ("all", "--speed", "50,100"): {"speed": (50.0, 100.0)},
        ("all", "--rpm", "3000"): {"rpm": 3000.0},
        ("all", "--input", "car=pass.wav"): {"inputs": {"car": Path("pass.wav")}},
    }
    for argv, given in read.items():
        args = parser.parse_args(list(argv))
        assert {name: getattr(args, name) for name in given} == given, argv
    for old, new in {"sound": "feeds", "scene": "simulate", "walk": "render"}.items():
        assert road.step(old).name == new
    args = parser.parse_args(["all"])
    assert (args.sources, args.sources_play) == ("car", "in-turn") and road.check(args) is None
    assert run_folder(road.resolved(args)).name == "passby_binaural"
    general = DEMO.at(EXAMPLES / "roadside").parser()
    same = general.parse_args(["all", "--scene", "scene.yaml", "--sources", "car"])
    assert run_folder(DEMO.at(EXAMPLES / "roadside").resolved(same)).name == ("scene_car_binaural")
    assert road.command("all", args) == "python passby_demo.py all"
    truck = parser.parse_args(["all", "--vehicle", "truck"])
    assert road.command("all", truck) == "python passby_demo.py all --sources truck"
    # A recording of the car's: no sound is made, so no engine turns.
    wav = tmp_path / "pass.wav"
    audio.write(wav, np.zeros((1600, 1)), 16_000)
    taken = parser.parse_args(["all", "--input", f"car={wav}"])
    assert road.check(taken) is None
    assert road.check(parser.parse_args(["all", "--input", f"car={wav}", "--rpm", "3000"])) == (
        "--rpm changes nothing: no road vehicle plays a made sound"
    )


def test_the_streets_passby_demo_is_a_preset_of_the_demo_of_its_scene(capsys) -> None:
    """The street's passby_demo.py is the demo of its scene with its car taken in, each
    vehicle the scene's sources (a group where it has several) in a part of its own:
    the script's --vehicle reads as the sources that play, --car, --police-car and
    --fire-engine as what plays of the vehicle picked, in any order, and a flag for a
    vehicle not picked changes nothing, as in the script; with --sources too, or with a
    vehicle the street has not, it is refused; its steps run as those that took them
    over, and the flags are no options of the demo, which the editor shows as
    --sources."""
    from acoustic_spaces.demos import run_folder

    street = _script("street", "passby_demo").DEMO
    assert (street.name, street.example, street.title) == (
        "passby",
        "street",
        "A vehicle driving down the street",
    )
    parser = street.parser()
    for argv, picked in (
        ((), "car"),
        (("--vehicle", "bus"), "bus"),
        (("--vehicle", "motorcycle"), "motorcycle"),
        (("--vehicle", "tram"), "Tram"),
        (("--car", "car_and_horn"), "Passenger car"),
        (("--car", "horn", "--vehicle", "car"), "car_horn"),
        (("--vehicle", "police_car"), "Police car"),
        (("--police-car", "siren", "--vehicle", "police_car"), "police_car_siren"),
        (("--vehicle", "police_car", "--police-car", "siren"), "police_car_siren"),
        (("--police-car", "siren"), "car"),
        (("--vehicle", "fire_engine"), "Fire engine"),
        (
            ("--vehicle", "fire_engine", "--fire-engine", "truck_and_siren"),
            "fire_engine+fire_engine_siren",
        ),
        (
            ("--fire-engine", "siren_and_horn", "--vehicle", "fire_engine"),
            "fire_engine_siren+fire_engine_horn",
        ),
        (("--vehicle", "fire_engine", "--fire-engine", "siren"), "fire_engine_siren"),
        (("--vehicle", "car,police_car", "--car", "car_and_horn"), "Passenger car,Police car"),
    ):
        args = parser.parse_args(["all", *argv])
        assert args.sources == picked, argv
        assert not [k for k in vars(args) if k.startswith("old_")]
        assert street.check(args) is None or "tram" in argv, (argv, street.check(args))
    for argv, problem in (
        (
            ("--sources", "bus", "--vehicle", "car"),
            "--sources picks what --vehicle, --car, --police-car and --fire-engine pick: "
            "give one or the other",
        ),
        (
            ("--vehicle", "boat"),
            "--vehicle is one of car, bus, motorcycle, police_car, fire_engine, tram, not 'boat'",
        ),
        (("--car", "honk"), "argument --car: invalid choice: 'honk'"),
    ):
        capsys.readouterr()
        with pytest.raises(SystemExit):
            parser.parse_args(["all", *argv])
        assert problem in capsys.readouterr().err, argv
    for old, new in {"sound": "feeds", "scene": "simulate", "walk": "render"}.items():
        assert street.step(old).name == new
    assert not {o.name for o in street.options} & {"vehicle", "car", "police_car", "fire_engine"}
    args = parser.parse_args(["all"])
    assert (args.sources, args.sources_play) == ("car", "in-turn")
    assert run_folder(street.resolved(args)).name == "passby_binaural"
    assert street.command("all", args) == "python passby_demo.py all"
    police = parser.parse_args(["all", "--vehicle", "police_car"])
    assert run_folder(street.resolved(police)).name == "passby_Police-car_binaural"
    assert street.command("all", police) == "python passby_demo.py all " + shell_line(
        ["--sources", "Police car"]
    )
    assert street.moving(street.resolved(police)) == ["police_car", "police_car_siren"]
    several = street.resolved(parser.parse_args(["all", "--vehicle", "car,police_car"]))
    assert [run_folder(p).name for p in several.series_parts] == [
        "passby_binaural",
        "passby_Police-car_binaural",
    ]


def test_the_gallery_walk_is_a_preset_of_the_demo_of_its_scene() -> None:
    """walk_demo.py is the demo of the gallery's scene with the walker and the one turning
    alone: each command line of the script it was reads as the demo of any scene's (the
    talker as its ACE recording, several too, the voice as the vocal effort, the stand
    and the turn as the listener standing's, the example's scene as the scene), its
    steps run as those that took them over, and a run of it is This scene's run on the
    gallery with those listeners, in the same folder."""
    from acoustic_spaces.demos import run_folder
    from acoustic_spaces.demos.scene_demo import DEMO

    walk = _script("gallery", "walk_demo").DEMO
    assert (walk.name, walk.example, walk.title) == (
        "walk",
        "gallery",
        "A walk through the gallery",
    )
    parser = walk.parser()
    read = {
        ("walk", "--talker", "F1"): {"recording": "ace_f1"},
        ("all", "--talker", "M4,F1"): {"recording": ("ace_m4", "ace_f1")},
        ("all", "--voice", "raised"): {"vocal_effort": "raised"},
        ("turn", "--stand", "4,15"): {"stand": "standing=4,15"},
        ("turn", "--turn", "0:0,6:180"): {"turn": "standing=0:0,6:180"},
        ("all", "--turn", "visitor=0:0,6:180"): {"turn": "visitor=0:0,6:180"},
        ("all", "--from", "other.yaml"): {"scene": "other.yaml"},
        ("all", "--listeners", "standing,visitor"): {"listeners": "standing,visitor"},
        ("all", "--input", "guide=talk.wav"): {"inputs": {"guide": Path("talk.wav")}},
    }
    for argv, given in read.items():
        args = parser.parse_args(list(argv))
        assert {name: getattr(args, name) for name in given} == given, argv
    for old, new in {
        "speech": "recording",
        "scene": "simulate",
        "walk": "render",
        "turn": "render",
        "video": "video",
    }.items():
        assert walk.step(old).name == new
    args = parser.parse_args(["all"])
    assert args.listeners == "visitor,standing" and walk.check(args) is None
    general = DEMO.at(EXAMPLES / "gallery").parser()
    same = general.parse_args(["all", "--scene", "scene.yaml", "--listeners", "visitor,standing"])
    assert run_folder(walk.resolved(args)).name == "walk_visitor+standing_binaural"
    assert run_folder(DEMO.at(EXAMPLES / "gallery").resolved(same)).name == (
        "scene_visitor+standing_binaural"
    )
    assert walk.command("all", args) == "python walk_demo.py all"
    assert walk.command("all", parser.parse_args(["all", "--talker", "F1"])).endswith(
        "all --recording ace_f1"
    )


# --------------------------------------------------------------------------- the demo code


@pytest.mark.parametrize(
    "path",
    [*sorted(EXAMPLES.glob("*/*_demo.py")), None],
    ids=lambda p: "scene_demo.py" if p is None else p.name,
)
def test_a_demo_reads_only_options_it_has(path: Path | None) -> None:
    """Every ``args.NAME`` a demo reads is one of its options: one it does not declare
    fails only when the step that reads it runs. None: the demo of any scene, read from
    the package as installed."""
    import ast

    from acoustic_spaces.demos import scene_demo

    if path is None:
        demo, path = scene_demo.DEMO, Path(scene_demo.__file__)
    else:
        demo = load(path)
    have = set(vars(demo.defaults()))
    used = {
        node.attr
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }
    assert sorted(used - have) == []


#: The examples whose demo of the scene plays a free recording through the sources with
#: a channel, and what it is.
RECORDING_EXAMPLES = {
    "concert_stage": "spicy_funk_cake",
    "stadium_concert": "spicy_funk_cake",
    "lakeside_concert": "spicy_funk_cake",
    "open_air_stage": "spicy_funk_cake",
    "studio": "spicy_funk_cake",
    "town_hall_pa": "ace_m4",
    "lecture_hall": "ace_m4",
    "leaky_classroom": "ace_m4",
    "shoebox": "ace_m4",
    "bim_office": "ace_m4",
    "material_libraries": "ace_m4",
    "gallery": "ace_m4",
}


@pytest.mark.parametrize("name", sorted(RECORDING_EXAMPLES))
def test_the_stages_and_talkers_play_out_of_the_box(name: str) -> None:
    """The examples with a sound system, a stage or a talker play a free recording in
    the demo of the scene, each source with a channel its channel of it: the demo has
    nothing to ask for (the recording step downloads it), and the editor's Run demo
    window says what each source plays."""
    skip_without_extra(name)
    from acoustic_spaces import recordings
    from acoustic_spaces.demos.scene_demo import DEMO, plays, recording_download

    scene = Scene.from_yaml(example_path(name))
    assert scene.recording == RECORDING_EXAMPLES[name]
    free = recordings.named(scene.recording)
    playing = [s for s in scene.sources if s.in_use and s.channel is not None]
    assert playing and all(s.signal is None for s in playing)
    demo = DEMO.at(example_path(name).parent)
    args = demo.defaults()
    args.scene = example_path(name)
    args = demo.resolved(args)
    assert demo.check(args) is None
    said = plays(args)
    assert all(said[s.name].startswith(free.title) for s in playing)
    assert recording_download(args).startswith(f"{free.title} ({free.licence}")


def test_left_and_right_are_the_audience_s_in_every_example() -> None:
    """A sound system's left channel plays on the audience's left: the stadium's and the
    stage's loudspeakers as the mix position faces the stage, the lake's as the audience
    does, and the studio's as the listener sits (the monitors play mixes of stems)."""
    for example, name in (
        ("stadium_concert", "mix_position"),
        ("concert_stage", "mix_position"),
        ("lakeside_concert", "audience"),
        ("studio", None),
    ):
        scene = Scene.from_yaml(example_path(example))
        seat = next(lis for lis in scene.listeners if name in (None, lis.name))
        left = np.array(seat.orientation.rotation())[:, 1]  # the listener's left
        sides = [s for s in scene.sources if s.in_use and s.channel in ("left", "right")]
        assert sides, example
        for source in sides:
            side = float(np.dot(np.subtract(source.position, seat.position), left))
            assert source.channel == ("left" if side > 0 else "right"), (example, source.name)


#: The talkers of the examples, by example.
TALKERS = {
    "lecture_hall": "lecturer",
    "leaky_classroom": "teacher",
    "shoebox": "talker",
    "bim_office": "talker",
    "material_libraries": "lecturer",
    "gallery": "guide",
}


@pytest.mark.parametrize("name", sorted(TALKERS))
def test_the_talkers_speak_at_normal_effort(name: str) -> None:
    """Every example's talker speaks with a normal vocal effort, as ISO 9921 has it:
    60 dB(A) 1 m in front, a man's speech spectrum, its sound power from its
    directivity, not the default 94 dB, which is a shout."""
    skip_without_extra(name)
    talker = next(s for s in Scene.from_yaml(example_path(name)).sources if s.name == TALKERS[name])
    assert talker.spectrum == "speech_male" and talker.vocal_effort == "normal"
    assert talker.level_db == talker.talking_db("normal")


def test_the_texts_say_what_the_sources_and_listeners_picks_do() -> None:
    """The tours' --listeners help says a --tour given keeps its own order; every demo
    picks its sources (--sources), as the docs say; the crossing plays the concert
    itself, with no run of the stadium demo first."""
    demos = {demo.name: demo for demo in list_demos()}
    kept = "a --tour given keeps its own order, this option leaving out the others"
    for name in ("stadium", "crossing", "orchestra", "church", "lakeside", "walk"):
        assert kept in " ".join(demos[name].option("listeners").help.split()), name
    without = sorted(
        (demo.example, demo.name)
        for demo in list_demos()
        if "sources" not in {option.name for option in demo.options}
    )
    assert without == []
    root = EXAMPLES.parent
    if not (root / "docs" / "demos.md").is_file():
        pytest.skip("the docs are in the source checkout only")
    docs = " ".join((root / "docs" / "demos.md").read_text(encoding="utf-8").split())
    assert "Every demo lists the sources it can play" in docs
    assert "It needs no run of the stadium demo first: the concert and the helicopter" in docs


def test_the_title_card_is_the_one_before_each_part() -> None:
    """The title card's docstring names the step that puts it before each part of a run
    of several parts."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos.pictures import title_card

    assert "join_parts" in title_card.__doc__ and "join`` step" not in title_card.__doc__
