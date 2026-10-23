import argparse
import importlib.util
import json
import re
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import skip_without_extra

from acoustic_spaces.audio import write_wav
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


def test_the_stadium_crowds_scatter_as_measured() -> None:
    """The crowds of the three stadium scenes scatter as a measured theatre audience
    does, little at 125 Hz and most from 1 kHz up, not at one value in every band."""
    measured = [0.3, 0.3, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7]
    for name in ("stadium_pa", "concert_stage", "stadium_concert"):
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
    found = feedback.loop(Room.from_yaml(example_path("concert_stage")), rays=30, seed=1)
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


def _demo_args(**paths) -> argparse.Namespace:
    return argparse.Namespace(format="binaural", **{k: Path(v) for k, v in paths.items()})


def test_church_demo_scene_and_organ(tmp_path: Path) -> None:
    """church_demo.py's choir and organ: the tune and every voice of the organ fill a
    verse, every key has a sample at most two semitones away, the two sides of the organ
    share the notes, and the scene it writes is the church with 19 singers, the organ's
    four sources and the tour's listeners, without problems."""
    demo = _script("church", "church_demo")
    assert sum(beats for _, beats in demo.TUNE) == demo.BEATS
    for voice, notes in demo.ORGAN.items():
        assert sum(note[1] for note in notes) == demo.BEATS, voice
    assert demo.sample_name(60) == "C3"  # VCSL's names: C3 is middle C
    for division, voices in (("manual", "SAT"), ("pedal", "B")):
        for midi, start, end in demo.organ_events(voices):
            assert 0.0 <= start < end <= demo.BEATS
            for key in (midi, midi + 12) if division == "manual" else (midi,):
                assert abs(demo.nearest_sample(key, division) - key) <= 2, (division, key)
    files = demo.organ_needed()
    assert set(files) == {"Loud", "Quiet", "Loud Pedal", "Quiet Pedal"}
    assert all(name.endswith("_rr1.wav") for name in files["Quiet"])
    # Verse 3 slows down and holds the last chord; the music ends after the church rings.
    assert demo.beat_time(3, demo.BEATS) - demo.VERSES[3] > demo.BEATS
    assert demo.end_of_music() > demo.beat_time(3, 21.0) + demo.LAST_CHORD_S

    args = _demo_args(source=example_path("church"), scene=tmp_path / "scene_choir.yaml")
    args.results = tmp_path / "results_choir"
    demo.scene(args)
    room = Room.from_yaml(args.scene)
    assert room.problems() == []
    names = [source.name for source in room.scene.sources]
    assert len([name for name in names if name.startswith("singer_")]) == 19
    assert [name for name in names if name.startswith("organ_")] == list(demo.ORGAN_SOURCES)
    assert {item.name for item in room.scene.listeners} == {name for name, _ in demo.TOUR}
    assert room.scene.simulation.ir_length_s >= 9.0


def test_church_demo_tunes_and_times_a_take() -> None:
    """A take sung slower than the tune, 30 cents sharp and each note a little off:
    church_demo.py moves each note onto its beat, the singer's own offset away and 60 %
    of each note's error."""
    demo = _script("church", "church_demo")
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


def test_stadium_demo_band_and_feeds(tmp_path: Path) -> None:
    """stadium_demo.py puts the band on the stadium's stage, in a group of its own, and
    gives the sound system the master's channels as the audience sees them: the arrays
    on its left (-y; the scene's *_right, named from the stage) the left channel."""
    demo = _script("stadium_concert", "stadium_demo")
    folder = copy_examples(tmp_path, ["stadium_concert"])[0].parent
    args = _demo_args(source=folder / "scene.yaml", scene=folder / "scene_band.yaml")
    args.results = folder / "results_band"
    demo.band(args)
    demo.band(argparse.Namespace(**vars(args) | {"source": args.scene}))  # once only
    room = Room.from_yaml(args.scene)
    assert room.problems() == []
    sources = {source.name: source for source in room.scene.sources}
    assert [name for name in sources if name in demo.BAND] == list(demo.BAND)
    assert {sources[name].group for name in demo.BAND} == {"Band"}
    scene = demo.scene_of(args)
    roles = {source["name"]: demo.role(source) for source in scene["sources"]}
    assert roles["main_right"] == "left" and roles["main_left"] == "right"
    assert roles["subs"] == "mono" and roles["vocal_wedges"] == "singer"
    assert roles["drum_fill"] == "drummer" and roles["side_fill_left"] == "stage"
    assert all(roles[name] == name for name in demo.BAND)
    # Every song has the stems the band and the monitor mixes lean on; the mixes name
    # no stem that none of the songs has.
    stems = {name: set(song["stems"]) for name, song in demo.SONGS.items()}
    assert all({"drums", "guitar", "keys", "vocal"} <= names for names in stems.values())
    for mix in (demo.SINGER_MIX, demo.DRUMMER_MIX, demo.STAGE_MIX, *demo.BAND_MIX.values()):
        assert set(mix) <= set.union(*stems.values())
    for song in demo.SONGS.values():
        assert set(song) >= {"master", "stems", "bpm", "downbeat_s", "credit", "sources"}
        assert all(len(files) >= 1 for files in song["stems"].values())
        for url, member in [
            *song["master"],
            *(f for files in song["stems"].values() for f in files),
        ]:
            assert url.startswith("https://") and (member is None or not member.startswith("/"))
    # The cuts shift each mix by its loudest arrival: here the delay 30 ms after the
    # main array, which plays 3 dB quieter but reaches the seat 8 dB louder.
    results = tmp_path / "results"
    results.mkdir()
    responses = []
    for name, delay_s, amplitude in (("main_right", 0.35, 0.4), ("delay_east_right", 0.38, 1.0)):
        response = np.zeros((48_000, 2))
        response[int(delay_s * 48_000)] = amplitude
        write_wav(results / f"{name}__east_stand.wav", response, 48_000)
        outputs = [{"format": "binaural", "wav": f"{name}__east_stand.wav"}]
        responses.append({"source": name, "listener": "east_stand", "outputs": outputs})
    (results / "parameters.json").write_text(json.dumps({"responses": responses}), "utf-8")
    table = {"sources": {"main_right": {"gain_db": 0.0}, "delay_east_right": {"gain_db": -3.0}}}
    arrival = demo.arrival_s(argparse.Namespace(results=results), "east_stand", table)
    assert arrival == pytest.approx(0.38, abs=0.002)
    # The tour moves on phrase boundaries and leaves out listeners without a mix.
    work = tmp_path / "demo"
    (work / "feeds").mkdir(parents=True)
    funk = demo.SONGS["spicy_funk_cake"]
    phrase = demo.phrase_s(funk)
    (work / "feeds" / "feeds.json").write_text(
        json.dumps({"length_s": 195.0, "song": funk | {"name": "spicy_funk_cake"}}), "utf-8"
    )
    tour = demo.tour_of(argparse.Namespace(tour=None, work=work), {"mix_position", "singer"})
    assert [name for name, _, _ in tour] == ["mix_position", "singer", "mix_position"]
    assert tour[1][1] == pytest.approx(phrase) and tour[-1][2] == 195.0
    seats = {"seats": {"mix_position": {"rms_db": -40.0}, "singer": {"rms_db": -22.0}}}
    assert demo.gains_of(seats, tour) == {"mix_position": 18.0, "singer": 0.0}
    # The default tour fits in each song, every stop on a 4 or 8 bar boundary half a
    # crossfade before a downbeat, and the last stop longer than a phrase.
    for name, length in (("spicy_funk_cake", 195.0), ("francium", 294.5), ("rubidium", 232.0)):
        song = demo.SONGS[name] | {"name": name}
        (work / "feeds" / "feeds.json").write_text(
            json.dumps({"length_s": length, "song": song}), "utf-8"
        )
        tour = demo.tour_of(argparse.Namespace(tour=None, work=work), set(demo.TITLES))
        left_out = {name for name, n in song.get("tour", {}).items() if n == 0}
        assert len(tour) == len(demo.TOUR) - len(left_out)
        assert not {name for name, _, _ in tour} & left_out
        assert tour[-1][2] - tour[-1][1] > demo.phrase_s(song)
        first = song["downbeat_s"] - demo.CROSSFADE_S / 2
        for _, start, _ in tour[1:]:
            bars = (start - first) / (demo.phrase_s(song) / 2)
            assert bars == pytest.approx(round(bars)), (name, start)
    assert len(tour) == len(demo.TOUR)  # Rubidium fits all twelve listeners


def test_orchestra_demo_tour(tmp_path: Path) -> None:
    """orchestra_demo.py names a track for every source of the orchestra, cuts between
    seats at the quietest moment near each share's end, and plays the movements one
    after the other with a pause, each seat's mix shifted earlier by its lag."""
    demo = _script("orchestra", "orchestra_demo")
    scene = yaml.safe_load(example_path("orchestra").read_text(encoding="utf-8"))
    assert {source["name"] for source in scene["sources"]} == set(demo.TRACKS)
    assert len(set(demo.TRACKS.values())) == len(demo.TRACKS)
    assert {seat for _, seats in demo.TOUR for seat in seats} == set(demo.TITLES)
    assert {seat["name"] for seat in scene["listeners"]} == set(demo.TITLES)
    # A movement of 30 s, quiet from 14 to 16 s: two seats cut there, not at 15.
    rate = 8000
    t = np.arange(30 * rate) / rate
    loud = np.where((t > 14.0) & (t < 16.0), 0.01, 0.5) * np.sin(2 * np.pi * 220 * t)
    env = demo.envelope(loud, rate)
    assert 14.0 <= demo.quiet_moment(env, 0.1, 15.0, demo.QUIET_S) <= 16.0
    assert demo.quiet_moment(env, 0.1, 5.0, 2.0) == pytest.approx(5.0, abs=2.5)
    work = tmp_path / "demo"
    args = argparse.Namespace(
        work=work, format="binaural", tour="1:stalls-middle,1:conductor,2:stalls-rear"
    )
    stereo = np.stack([loud, loud], axis=1)
    for movement, seat in demo.stops_of(args):
        length = (30 if movement == 1 else 10) * rate
        demo.write(demo.mix_path(args, movement, seat), stereo[:length], rate)
    stops = {
        "1:stalls-middle": {"lag_s": 0.05, "length_s": 30.0},
        "1:conductor": {"lag_s": 0.01, "length_s": 30.0},
        "2:stalls-rear": {"lag_s": 0.08, "length_s": 10.0},
    }
    seats = {"rate": rate, "format": "binaural", "gain": 1.0, "stops": stops}
    tour = demo.tour_of(args, seats)
    assert [(s["movement"], s["seat"]) for s in tour] == [
        (1, "stalls-middle"),
        (1, "conductor"),
        (2, "stalls-rear"),
    ]
    assert tour[0]["start"] == 0.0 and 14.0 <= tour[0]["stop"] <= 16.0
    assert tour[1]["start"] == tour[0]["stop"] and tour[1]["stop"] == 30.0
    assert tour[2]["at"] == pytest.approx(30.0 + demo.PAUSE_S) and tour[2]["start"] == 0.0
    audio = demo.tour_audio(args, tour, seats)
    assert len(audio) == round((30.0 + demo.PAUSE_S + 10.0) * rate)
    pause = audio[int(30.5 * rate) : int(32.5 * rate)]
    assert np.abs(pause).max() == 0.0  # between the movements
    assert np.abs(audio[int(5 * rate) : int(6 * rate)]).max() > 0.4  # playing before the cut
    assert np.abs(audio[int(20 * rate) : int(21 * rate)]).max() > 0.4  # and after it


def test_stadium_pictures_show_the_crowd() -> None:
    """The stadium demo's stills draw the crowd where the scene has its audience: standing
    on the pitch, seated in three stands, and the stand behind the stage as empty seats,
    with how many people stand or sit there and how much each takes up at 1 kHz."""
    pytest.importorskip("PIL")
    pictures = _script("stadium_concert", "stadium_pictures")
    path = example_path("stadium_concert")
    scene = Scene.from_yaml(path).model_dump(mode="json", exclude_none=True)
    drawing = pictures.Pictures(scene, path.parent, {}, {}, [], {})
    kinds: dict[str, list] = {}
    for area in drawing.areas:
        kinds.setdefault(area.kind, []).append(area)
    assert sorted(kinds) == ["seated", "seats", "standing"]
    assert len(kinds["standing"]) == 1 and len(kinds["seated"]) == 3 == 4 - len(kinds["seats"])
    (standing,) = kinds["standing"]
    assert standing.people == 12_308  # 90.5 by 68 m, two on a square metre
    assert sum(area.people for area in kinds["seated"]) == pytest.approx(29_000, rel=0.01)
    assert standing.absorbs == pytest.approx(0.99) and kinds["seats"][0].absorbs == 0.2
    # Each dot stands for DOT_SPACING squared over the density in square metres.
    dots = standing.dots(seed=0)
    assert len(dots) == pytest.approx(90.5 * 68 * 2 / pictures.DOT_SPACING**2, rel=0.05)
    assert pictures._inside(dots, standing.triangles).all()
    assert drawing.thumbnail().getbbox() is not None
    # Every still writes the same line under the plan: drawing it changes no area.
    people, lines = [area.people for area in drawing.areas], []
    for _ in range(2):
        img = pictures.Image.new("RGBA", (1920, 40))
        drawing.surfaces(pictures.ImageDraw.Draw(img), 0, 10)
        lines.append(img.tobytes())
    assert lines[0] == lines[1] and [area.people for area in drawing.areas] == people


def test_demo_scripts_take_an_hrtf(tmp_path: Path) -> None:
    """Every demo script's simulate command carries ``--hrtf``; its mixes then ask for the
    responses made through that file (``binaural-<name>``) and refuse ones made through
    the spherical head; other formats are left alone; and asked for mixes before the
    scene and the simulation exist, a script says which step writes the scene."""
    sofa = tmp_path / "me.sofa"
    scripts = {
        ("orchestra", "orchestra_demo"): None,
        ("church", "church_demo"): "scene",
        ("stadium_concert", "stadium_demo"): "band",
        ("lakeside_concert", "lakeside_demo"): "scenes",
    }
    for (example, name), step in scripts.items():
        demo = _script(example, name)
        paths = {"scene": tmp_path / "scene.yaml", "results": tmp_path / "results"}
        if name == "lakeside_demo":
            paths |= {"still": tmp_path / "still.yaml", "still_results": tmp_path / "still"}
        plain = _demo_args(**paths)
        own = argparse.Namespace(**vars(plain), hrtf=sofa)
        if name == "lakeside_demo":

            def command(a, demo=demo):
                return demo.simulate_command(a, a.scene, a.results)

            def check(a, e, demo=demo):
                return demo.check_hrtf(a, "evening", e)

            def responses(a, demo=demo):
                return demo.responses_of(a, "evening")

        else:
            command, check, responses = demo.simulate_command, demo.check_hrtf, demo.responses_of
        assert "--hrtf" not in command(plain), name
        assert command(own).endswith(f" --hrtf {sofa}"), name
        assert demo.output_label(plain) == "binaural"
        assert demo.output_label(own) == "binaural-me"
        assert demo.output_label(argparse.Namespace(format="ambix3", hrtf=sofa)) == "ambix3"
        outputs = {"binaural": "x", "binaural-me": "x"}
        head, ears = (
            [{"listener": "a", "source": "s", "outputs": [{"format": f, "wav": w}]}]
            for f, w in outputs.items()
        )
        for a, e in ((plain, head), (plain, ears), (own, ears)):
            check(a, e)
        with pytest.raises(SystemExit, match="me.sofa"):
            check(own, head)
        with pytest.raises(SystemExit) as caught:
            responses(plain)
        if step:
            assert f"{name}.py {step}" in str(caught.value), name
        assert "acoustic-spaces simulate" in str(caught.value), name


def test_lakeside_demo_scenes_tour_and_gains(tmp_path: Path) -> None:
    """lakeside_demo.py plays the lake scene through its sound system, adds a listener upwind
    and writes the same scene in still air; the tour moves on phrase boundaries, every
    stop is turned up to the audience's loudness, and a place in still air keeps the
    gain of the same place in the evening's weather, so the difference is heard."""
    demo = _script("lakeside_concert", "lakeside_demo")
    folder = copy_examples(tmp_path, ["lakeside_concert"])[0].parent
    args = _demo_args(
        source=folder / "scene.yaml",
        scene=folder / "scene_demo.yaml",
        still=folder / "scene_still.yaml",
        results=folder / "results_demo",
        still_results=folder / "results_still",
    )
    demo.scenes(args)
    demo.scenes(argparse.Namespace(**vars(args) | {"source": args.scene}))  # once only
    evening, still = Room.from_yaml(args.scene), Room.from_yaml(args.still)
    for room in (evening, still):
        assert room.problems() == []
        assert [source.name for source in room.scene.sources] == list(demo.SYSTEM)
        names = [listener.name for listener in room.scene.listeners]
        assert names.count(demo.UPWIND["name"]) == 1 and set(demo.TITLES) == set(names)
        assert room.scene.simulation.ir_length_s == demo.IR_LENGTH_S
    assert evening.scene.atmosphere is not None and still.scene.atmosphere is None
    # The tour: 8 bars a stop, the last one --last-phrases long.
    work = tmp_path / "demo"
    (work / "music").mkdir(parents=True)
    song = demo.SONGS["spicy_funk_cake"] | {"name": "spicy_funk_cake"}
    (work / "music" / "song.json").write_text(json.dumps(song), encoding="utf-8")
    write_wav(work / "music" / "spicy_funk_cake.wav", np.zeros((19_500, 2)), 100)  # 195 s
    tour = demo.tour_of(argparse.Namespace(tour=None, work=work, last_phrases=2.0))
    phrase = demo.phrase_s(song)
    assert [(name, weather) for name, weather, _, _ in tour] == [s[:2] for s in demo.TOUR]
    assert tour[1][2] == pytest.approx(2 * phrase)
    assert tour[-1][3] - tour[-1][2] == pytest.approx(2 * phrase)
    custom = argparse.Namespace(tour="bedroom_window:still:1,audience", work=work, last_phrases=0)
    tour = demo.tour_of(custom)
    assert [stop[:2] for stop in tour] == [("bedroom_window", "still"), ("audience", "evening")]
    assert tour[-1][3] == pytest.approx(195.0)
    # The gains: the audience's loudness (K-weighted, not the dB(A)) everywhere, still
    # air at the evening's gain.
    levels = {"audience": -10.0, "bedroom_window": -30.0}
    seats = {
        "seats": {
            f"{name}.{weather}": {
                "rms_db": level - 8.0 - (5.0 if weather == "still" else 0.0),
                "loudness_db": level - (5.0 if weather == "still" else 0.0),
            }
            for name, level in levels.items()
            for weather in ("evening", "still")
        }
    }
    gains = demo.gains_of(seats, tour)
    assert gains == {"bedroom_window.still": 20.0, "audience.evening": 0.0}
    # Seats rendered before loudness was kept are matched by their dB(A).
    for seat in seats["seats"].values():
        del seat["loudness_db"]
    assert demo.gains_of(seats, tour) == gains
