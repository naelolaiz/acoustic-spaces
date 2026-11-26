"""The names of the demos' runs, so that runs with other settings write over none of each
other's files, and the runs of several parts (several values of an option, sources in
turn, listeners each in a part), played one after another in one video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.demos import (
    IN_TURN,
    RUN_NAME_MOST,
    RUN_RECORD,
    Demo,
    Option,
    Source,
    Step,
    joined_name,
    list_demos,
    load,
    name_part,
    pipeline,
    product,
    record_run,
    run_folder,
)
from acoustic_spaces.demos.join import card_lines, clock, join_parts, shifted_chapter
from acoustic_spaces.examples import copy_examples


def _demo(name: str, example: str):
    return next(d for d in list_demos() if d.name == name and d.example == example)


def test_a_name_part_keeps_what_a_file_name_can_hold() -> None:
    """Letters, digits, ``_``, ``-`` and ``+`` stay; a decimal point is ``p``; anything
    else is ``-``; nothing left is a digest of the text."""
    assert name_part("80.5 km/h") == "80p5-km-h"
    assert name_part("my song") == "my-song"
    assert name_part("binaural-kemar") == "binaural-kemar"
    assert name_part("a+b") == "a+b"
    assert name_part("...") == name_part("...") and len(name_part("...")) == 6
    assert name_part("...") != name_part("///")


def test_a_long_name_ends_in_a_digest() -> None:
    """A name longer than RUN_NAME_MOST keeps its front and ends in a digest of the
    whole, so two long names that differ at their ends differ."""
    short = joined_name(["flight", "departure", "binaural"])
    assert short == "flight_departure_binaural"
    one, two = joined_name(["x" * 120, "a"]), joined_name(["x" * 120, "b"])
    assert len(one) == RUN_NAME_MOST and one.startswith("x" * 80) and one != two


def test_every_demo_names_its_runs_after_their_settings() -> None:
    """At its defaults, a run is named after its demo and its output; what it plays,
    where it is not the demo's own pick, another setting, listeners, HRTFs or format is
    another name, and the name of a run at its defaults again is the same."""
    flight = _demo("flight", "airport")

    def name(*argv: str) -> str:
        return flight.run_name(flight.resolved(flight.parser().parse_args(["all", *argv])))

    assert name() == "flight_binaural"  # the departure, its own pick
    assert name("--sources", "arrival") == name("--vehicle", "arrival")
    assert name("--sources", "arrival") == "flight_arrival_binaural"
    assert name("--listeners", "lateral_450m") == "flight_lateral_450m_binaural"
    assert name("--step", "8") == "flight_step-8_binaural"
    assert name("--format", "ambix3") == "flight_ambix3"
    assert name("--hrtf", "/x/kemar.sofa") == "flight_binaural-kemar"
    assert name("--no-mp4") == "flight_binaural"  # never named
    many = ",".join(f"listener_number_{k}" for k in range(9))
    assert name("--listeners", many).startswith("flight_9-listeners-")
    # Every demo at its defaults: a name of its own in its work folder, a preset of the
    # demo of any scene's too, as its defaults are not that demo's.
    for demo in list_demos():
        if demo.name == "scene":
            continue
        resolved = demo.resolved(demo.defaults())
        assert run_folder(resolved).parent == demo.work_folder(resolved)
        assert run_folder(resolved).name.startswith(demo.name + "_")
        assert product(resolved, ".mp4").parent == demo.work_folder(resolved)


def test_a_song_of_your_own_names_the_run_in_place_of_the_preset(tmp_path: Path) -> None:
    """A recording of yours plays in place of the scene's: the run is named after it."""
    lakeside = _demo("lakeside", "lakeside_concert")
    args = lakeside.defaults()
    assert lakeside.run_name(lakeside.resolved(args)) == "lakeside_binaural"
    song = tmp_path / "my song.wav"
    mine = argparse.Namespace(**vars(args) | {"recording": song, "bpm": 120.0})
    assert lakeside.run_name(lakeside.resolved(mine)) == "lakeside_my-song_bpm-120_binaural"


def test_a_run_plays_the_slow_work_of_the_run_up_to_it() -> None:
    """The run up to a step (Demo.upstream) has the options that matter to none of the
    steps up to it at their defaults: another --at mixes the passes of the same walk."""
    crossing = _demo("crossing", "stadium_concert")
    args = crossing.resolved(argparse.Namespace(**vars(crossing.defaults()) | {"at": 5.0}))
    assert run_folder(args).name == "crossing_at-5_binaural"
    up = crossing.upstream(args, "walk")
    assert run_folder(up).name == "crossing_binaural" and up.at is None
    assert crossing.upstream(args, "mix").at == 5.0


def test_a_written_scene_names_the_scene_it_is_made_from(tmp_path: Path) -> None:
    """A demo's scene is done when it is made from the run's --from, and newer than it;
    another --from writes it again."""
    first, other = tmp_path / "a.yaml", tmp_path / "b.yaml"
    for path in (first, other):
        path.write_text("name: x\n", encoding="utf-8")
    written = tmp_path / "run" / "scene_demo.yaml"
    pipeline.write_scene(written, {"name": "x"}, "# a header\n", tmp_path, made_from=first)
    assert pipeline.scene_made(written, first) and not pipeline.scene_made(written, other)
    later = written.stat().st_mtime + 10
    os.utime(first, (later, later))
    assert not pipeline.scene_made(written, first)  # changed since
    assert not pipeline.scene_made(tmp_path / "none.yaml", first)
    # The files it names are found from where it is written.
    moved = pipeline.files_from({"directivity": {"file": "box.csv"}}, written, tmp_path)
    assert moved == {"directivity": {"file": (tmp_path / "box.csv").resolve().as_posix()}}
    beside = pipeline.files_from({"directivity": {"file": "run/box.csv"}}, written, tmp_path)
    assert beside == {"directivity": {"file": "box.csv"}}


def test_several_values_of_an_option_play_in_turn() -> None:
    """--format binaural,mono: a run of two parts, each an ordinary run of its own
    folder, the run named after both; the command line gives them as they came."""
    flight = _demo("flight", "airport")
    args = flight.parser().parse_args(["all", "--format", "binaural,mono"])
    assert args.format == ("binaural", "mono")
    resolved = flight.resolved(args)
    assert run_folder(resolved).name == "flight_binaural+mono"
    assert [run_folder(p).name for p in resolved.series_parts] == [
        "flight_binaural",
        "flight_mono",
    ]
    command = flight.command("all", args, names_only=True)
    assert command == "python flight_demo.py all --format binaural,mono"
    assert flight.check(args) is None
    # One value is an ordinary run; a value twice plays once.
    one = flight.parser().parse_args(["all", "--format", "mono,mono"])
    assert one.format == "mono" and len(flight.each(one)) == 1
    with pytest.raises(SystemExit):  # not one of its choices
        flight.parser().parse_args(["all", "--levels", "matched,loudest"])


def test_the_series_orders_the_parts_outermost_first() -> None:
    """Two options of several values play every pair, the first of the options
    outermost; --series sets another order, and names the run after it; the listeners
    split into parts only where --series names them."""
    flight = _demo("flight", "airport")

    def parts(*argv: str) -> tuple[str, list[str]]:
        resolved = flight.resolved(flight.parser().parse_args(["all", *argv]))
        return run_folder(resolved).name, [run_folder(p).name for p in resolved.series_parts]

    name, names = parts("--sources", "departure,arrival", "--format", "binaural,mono")
    assert name == "flight_each-departure+arrival_binaural+mono"
    assert names == [
        "flight_binaural",
        "flight_mono",
        "flight_arrival_binaural",
        "flight_arrival_mono",
    ]
    name, names = parts(
        "--sources", "departure,arrival", "--format", "binaural,mono", "--series", "format"
    )
    assert name == "flight_each-departure+arrival_series-format-sources_binaural+mono"
    assert names[:2] == ["flight_binaural", "flight_arrival_binaural"]
    name, names = parts("--listeners", "lateral_450m,approach_2000m", "--series", "listeners")
    assert name == "flight_each-lateral_450m+approach_2000m_series-listeners_binaural"
    assert names == ["flight_lateral_450m_binaural", "flight_approach_2000m_binaural"]
    problem = flight.check(flight.parser().parse_args(["all", "--series", "colour"]))
    assert problem is not None and "--series colour" in problem


def test_each_part_plays_its_own_sources_in_turn() -> None:
    """In turn, each name or group --sources gives plays in a part of its own (the car,
    the police car itself and its siren together, then its siren alone), named apart
    from the run of them together; --sources stands before --step among the options, so
    its turns are outermost; a demo with no choice of sources, or one listener, has no
    parts of them; --hrtf with no binaural part is refused as for one run."""
    street = _demo("passby", "street")

    def run(*argv: str):
        args = street.parser().parse_args(["all", *argv])
        assert street.check(args) is None, street.check(args)
        resolved = street.resolved(args)
        return run_folder(resolved).name, [
            (run_folder(p).name, p.sources) for p in getattr(resolved, "series_parts", [])
        ]

    for vehicles, named in (
        ("car,police_car", "car+Police-car"),
        ("police_car,car", "Police-car+car"),
    ):
        name, parts = run("--vehicle", vehicles)
        assert name == f"passby_each-{named}_binaural"
        assert sorted(parts) == [
            ("passby_Police-car_binaural", "Police car"),
            ("passby_binaural", "car"),
        ]
    name, parts = run("--sources", "car,police_car", "--sources-play", "together")
    assert name == "passby_car+police_car_binaural" and not parts
    name, parts = run("--sources", "Police car,police_car_siren", "--step", "1,2")
    assert [p[0] for p in parts] == [
        "passby_Police-car_step-1_binaural",
        "passby_Police-car_step-2_binaural",
        "passby_police_car_siren_step-1_binaural",
        "passby_police_car_siren_step-2_binaural",
    ]
    flight = _demo("flight", "airport")
    found = flight.check(
        flight.parser().parse_args(["all", "--format", "mono,ambix3", "--hrtf", "me.sofa"])
    )
    assert found == "--hrtf is for binaural mixes"
    one = flight.resolved(
        flight.parser().parse_args(["all", "--listeners", "lateral_450m", "--series", "listeners"])
    )
    assert run_folder(one).name == "flight_lateral_450m_binaural"
    # Each demo's --series names what it has.
    assert "sources (with --sources-play in-turn)" in street.option("series").help
    toy = Demo(
        name="toy",
        example="toy",
        title="Toy",
        description="A toy.",
        script=Path("toy_demo.py"),
        steps=[Step("video", lambda args: None, "the video")],
        options=[Option("work", "where", Path("demo"), "folder")],
    )
    found = toy.check(toy.parser().parse_args(["video", "--series", "sources"]))
    assert found == "--series sources: this demo gives no choice of sources"
    assert "sources" not in toy.option("series").help


def test_values_that_change_nothing_make_one_part() -> None:
    """Several values of an option that changes nothing with the sources picked (the
    railway's horn when it does not sound) are refused, naming it; a part they change
    nothing for (the railway's train without its horn, the street's car, for its
    sirens) plays once, and the run is named after the values that differ, not after
    the part's none."""
    railway = _demo("passby", "railway")
    args = railway.parser().parse_args(["all", "--horn", "blasts,chord"])
    assert railway.check(args) == "--horn changes nothing: no train horn plays a made sound"
    args = railway.parser().parse_args(
        ["all", "--sources", "train,train_horn", "--sources-play", "in-turn"]
        + ["--horn", "blasts,chord"]
    )
    assert railway.check(args) is None
    resolved = railway.resolved(args)
    assert run_folder(resolved).name == "passby_each-train+train_horn_horn-blasts+chord_binaural"
    assert [run_folder(p).name for p in resolved.series_parts] == [
        "passby_train_binaural",
        "passby_train_horn_horn-blasts_binaural",
        "passby_train_horn_horn-chord_binaural",
    ]
    street = _demo("passby", "street")
    args = street.parser().parse_args(
        ["all", "--sources", "car,Police car", "--siren", "wail,yelp"]
    )
    resolved = street.resolved(args)
    assert run_folder(resolved).name == "passby_each-car+Police-car_siren-wail+yelp_binaural"
    assert [run_folder(p).name for p in resolved.series_parts] == [
        "passby_binaural",
        "passby_Police-car_siren-wail_binaural",
        "passby_Police-car_siren-yelp_binaural",
    ]


def _sources_demo(tmp_path: Path) -> Demo:
    """A demo of four sources in two groups, three of them its own pick."""
    sources = [
        Source("main_left", "PA/Mains"),
        Source("subs", "PA/Subwoofers"),
        Source("drum_kit", "Band"),
        Source("helicopter", "Helicopter", on=False),
    ]
    return Demo(
        name="toy",
        example="toy",
        title="Toy",
        description="A toy.",
        script=tmp_path / "toy_demo.py",
        steps=[Step("video", lambda args: None, "the video")],
        options=[Option("work", "where", Path("demo"), "folder")],
        sources=lambda args: sources,
    )


def test_the_sources_a_run_plays_by_name_or_group(tmp_path: Path) -> None:
    """--sources picks by name or by group (PA takes PA/Mains and PA/Subwoofers); the
    demo's own pick plays by default and leaves the run's name as it is; in turn, each
    name given (a group as one) or each top group of the pick is a part."""
    demo = _sources_demo(tmp_path)

    def run(**values):
        return demo.resolved(argparse.Namespace(**vars(demo.defaults()) | values))

    assert run().playing == ("main_left", "subs", "drum_kit") and run_folder(run()).name == "toy"
    assert run(sources="PA").playing == ("main_left", "subs")
    assert run_folder(run(sources="PA")).name == "toy_PA"
    assert run(sources="PA,Band").playing == run().playing
    assert run_folder(run(sources="PA,Band")).name == "toy"  # the demo's own pick
    assert run(sources="helicopter,drum_kit").playing == ("drum_kit", "helicopter")
    turns = run(sources_play=IN_TURN)
    assert run_folder(turns).name == "toy_each-source"
    assert [p.playing for p in turns.series_parts] == [("main_left", "subs"), ("drum_kit",)]
    assert [run_folder(p).name for p in turns.series_parts] == ["toy_PA", "toy_Band"]
    given = run(sources="subs,Band", sources_play=IN_TURN)
    assert [p.playing for p in given.series_parts] == [("subs",), ("drum_kit",)]
    problem = demo.check(argparse.Namespace(**vars(demo.defaults()) | {"sources": "choir"}))
    assert problem is not None and "--sources choir: no such source or group" in problem


def test_a_pick_is_named_by_how_it_differs_from_the_demos_own(tmp_path: Path) -> None:
    """A pick of sources other than the demo's own is named by what it adds (with-) and
    leaves out (without-) where that is shorter than the names given, a group of two
    sources or more by its path when the pick takes or leaves all of it."""
    demo = _sources_demo(tmp_path)

    def name(sources: str) -> str:
        args = argparse.Namespace(**vars(demo.defaults()) | {"sources": sources})
        return run_folder(demo.resolved(args)).name

    assert name("PA") == "toy_PA"  # shorter than without-Band
    assert name("main_left,drum_kit") == "toy_without-subs"
    assert name("PA,Band,helicopter") == "toy_with-helicopter"  # a group of one: its source
    assert name("subs,helicopter") == "toy_subs+helicopter"
    railway = _demo("passby", "railway")
    stadium = _demo("stadium", "stadium_concert")
    song = "stadium_spicy_funk_cake"
    train = "train,train_roof,train_pantograph"  # the railway's own pick
    for demo_, sources, named in (
        (railway, train + ",train_horn", "passby_with-train_horn"),
        (stadium, "PA,Monitors,drum_kit,singer_voice", f"{song}_without-guitar_amp"),
        (stadium, "PA/Mains,PA/Subwoofers,Monitors,Band", f"{song}_without-PA-Fills+PA-Delays"),
    ):
        args = demo_.parser().parse_args(["all", "--sources", sources])
        assert run_folder(demo_.resolved(args)).name == f"{named}_binaural"


def test_the_join_moves_each_runs_chapters_to_its_place() -> None:
    assert clock(65.0) == "1:05" and clock(3725.0) == "1:02:05"
    assert shifted_chapter("1:05 Gate", 10.0) == "1:15 Gate"
    assert shifted_chapter("1:00:00 Late", 4.0) == "1:00:04 Late"
    assert shifted_chapter("no chapter", 4.0) is None


def test_the_title_card_says_what_a_run_plays_and_how(tmp_path: Path) -> None:
    record = {
        "run": "flight_arrival_lateral_450m_step-8_binaural-kemar",
        "title": "An aircraft over the airport",
        "command": "python flight_demo.py all --vehicle arrival",
        "parts": [
            ["demo", "flight"],
            ["plays", "arrival"],
            ["listeners", "lateral_450m"],
            ["changed", "step-8"],
            ["label", "binaural-kemar"],
        ],
    }
    title, name, lines = card_lines(record, 1, 3)
    assert title == "An aircraft over the airport" and name == record["run"]
    assert lines[0] == "Part 2 of 3"
    assert "Plays: arrival" in lines and "Listeners: lateral_450m" in lines
    assert "Settings: step-8" in lines
    assert any(line.startswith("Mixed ") and "kemar" in line for line in lines)
    assert "Run as: python flight_demo.py all --vehicle arrival" in lines
    assert "not comparable in level" in lines[-1]
    plain = card_lines({"run": "church_binaural", "parts": [["demo", "church"]]}, 0, 2)[2]
    assert "Listeners: every one" in plain and "Settings: at their defaults" in plain


def _video(path: Path, seconds: float, tone: int) -> None:
    """A small video of a test picture and a tone, ``seconds`` long."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )  # fmt: skip


def _length_s(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_parts_of_a_run_play_in_one_video(tmp_path: Path, capsys) -> None:
    """The parts' videos (as each part's record names it), each after a title card, into
    one video named after the run, with the chapters of each at their place; the run's
    folder keeps the cards and its record. A part with no video yet is refused."""
    pytest.importorskip("PIL")
    scene = copy_examples(tmp_path, ["airport"])[0]
    demo = load(scene.parent / "flight_demo.py").at(scene.parent)
    args = demo.resolved(demo.parser().parse_args(["video", "--vehicle", "departure,arrival"]))
    assert run_folder(args).name == "flight_each-departure+arrival_binaural"
    first, second = args.series_parts
    for part, seconds, tone in ((first, 1.0, 440), (second, 1.5, 660)):
        folder, video = run_folder(part), product(part, ".mp4")
        folder.mkdir(parents=True, exist_ok=True)
        if part is second:
            with pytest.raises(SystemExit, match="has no video yet"):
                join_parts(demo, args)
        _video(video, seconds, tone)
        record_run(
            folder,
            demo=demo.name,
            title=demo.title,
            run=folder.name,
            parts=[list(p) for p in demo.run_parts(part)],
            command=demo.command("all", part, names_only=True),
            video=video.name,
            length_s=seconds,
            chapters=["0:00 Start"],
        )
    out = join_parts(demo, args)
    work = run_folder(args).parent
    assert out == work / "flight_each-departure+arrival_binaural.mp4"
    assert out.exists() and f"wrote {out}" in capsys.readouterr().out
    assert _length_s(out) == pytest.approx(2 * 4.0 + 1.0 + 1.5, abs=0.15)
    chapters = (work / f"{out.stem}_chapters.txt").read_text().splitlines()
    assert chapters == [
        "0:00 1. flight_binaural",
        "0:04 Start",
        "0:05 2. flight_arrival_binaural",
        "0:09 Start",
    ]
    cards = sorted(p.name for p in (run_folder(args) / "cards").iterdir())
    assert cards == ["00_flight_binaural.png", "01_flight_arrival_binaural.png"]
    record = json.loads((run_folder(args) / RUN_RECORD).read_text())
    assert record["video"] == out.name
    assert record["series"] == ["flight_binaural", "flight_arrival_binaural"]


def _rms_db(path: Path, start_s: float, end_s: float) -> float:
    """The RMS (dB) of the sound of a video between two times."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ss", f"{start_s}", "-to", f"{end_s}",
         "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    x = np.frombuffer(out, dtype=np.float32).astype(float)
    return 10.0 * np.log10(np.mean(x**2))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_movements_of_a_symphony_are_joined_at_one_level(tmp_path: Path) -> None:
    """Parts that differ only in the piece of one recording (``--piece``, heard at one
    level) are joined at the level of the loudest: each turned down by as much as its
    own gain to its peak exceeds the loudest part's, which the cards say; parts of
    another series keep their own level, whatever their records say."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos.join import one_level

    scene = copy_examples(tmp_path, ["orchestra"])[0]
    demo = load(scene.parent / "orchestra_demo.py").at(scene.parent)
    args = demo.resolved(demo.parser().parse_args(["video", "--piece", "I,II"]))
    first, second = args.series_parts
    records = []
    for part, gain in ((first, 6.0), (second, 0.0)):  # the first played 6 dB quieter
        folder, video = run_folder(part), product(part, ".mp4")
        folder.mkdir(parents=True, exist_ok=True)
        _video(video, 2.0, 440)
        record = {"run": folder.name, "parts": [list(p) for p in demo.run_parts(part)]}
        record_run(folder, **record, video=video.name, length_s=2.0, peak_gain_db=gain)
        records.append(record | {"peak_gain_db": gain})
    assert one_level(demo, args, records)
    out = join_parts(demo, args)
    quiet, loud = _rms_db(out, 4.5, 5.5), _rms_db(out, 10.5, 11.5)
    assert loud - quiet == pytest.approx(6.0, abs=0.3)
    assert "comparable in level with each other" in card_lines(records[0], 0, 2, True)[2][-1]
    assert not one_level(demo, args, [{"run": "a"}, {"run": "b"}])  # no gain recorded
    airport = copy_examples(tmp_path, ["airport"])[0]
    flight = load(airport.parent / "flight_demo.py").at(airport.parent)
    flights = flight.resolved(
        flight.parser().parse_args(["video", "--vehicle", "departure,arrival"])
    )
    assert not one_level(flight, flights, records)
