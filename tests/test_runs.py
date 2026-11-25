"""The names of the demos' runs, so that runs with other settings write over none of each
other's files, and the join step, which plays several finished runs in one video."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from acoustic_spaces.demos import (
    JOIN,
    RUN_NAME_MOST,
    RUN_RECORD,
    joined_name,
    list_demos,
    load,
    name_part,
    pipeline,
    product,
    record_run,
    run_folder,
)
from acoustic_spaces.demos.join import (
    DEFAULT_PART,
    card_lines,
    clock,
    finished_runs,
    joined_run_name,
    shifted_chapter,
)
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
    """At its defaults, a run is named after its demo, what it plays where the demo
    gives a choice of it, and its output; another setting, listeners, HRTFs or format
    is another name, and the name of a run at its defaults again is the same."""
    flight = _demo("flight", "airport")
    args = flight.defaults()
    assert flight.run_name(flight.resolved(args)) == "flight_departure_binaural"

    def name(**values) -> str:
        return flight.run_name(flight.resolved(argparse.Namespace(**vars(args) | values)))

    assert name(vehicle="arrival") == "flight_arrival_binaural"
    assert name(listeners="lateral_450m") == "flight_departure_lateral_450m_binaural"
    assert name(step_m=8.0) == "flight_departure_step-8_binaural"
    assert name(format="ambix3") == "flight_departure_ambix3"
    assert name(hrtf=Path("/x/kemar.sofa")) == "flight_departure_binaural-kemar"
    assert name(no_mp4=True, again=False) == "flight_departure_binaural"  # never named
    many = ",".join(f"listener_number_{k}" for k in range(9))
    assert name(listeners=many).startswith("flight_departure_9-listeners-")
    # Every demo at its defaults: a name of its own in its work folder.
    for demo in list_demos():
        if demo.name == "scene":
            continue
        resolved = demo.resolved(demo.defaults())
        assert run_folder(resolved).parent == demo.work_folder(resolved)
        assert run_folder(resolved).name.startswith(demo.name + "_")
        assert product(resolved, ".mp4").parent == demo.work_folder(resolved)


def test_a_song_of_your_own_names_the_run_in_place_of_the_preset(tmp_path: Path) -> None:
    """--music plays in place of --song: the run is named after the song played."""
    lakeside = _demo("lakeside", "lakeside_concert")
    args = lakeside.defaults()
    assert lakeside.run_name(lakeside.resolved(args)) == "lakeside_spicy_funk_cake_binaural"
    mine = argparse.Namespace(**vars(args) | {"music": tmp_path / "my song.wav", "bpm": 120.0})
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


def test_the_join_names_the_video_after_what_its_runs_share() -> None:
    """The joined video's name: the parts all its runs share in front and behind, and
    between them what sets each apart, joined by +; a run with nothing of its own there
    is "default"."""
    departure = {
        "run": "a",
        "parts": [["demo", "flight"], ["plays", "departure"], ["label", "binaural"]],
    }
    arrival = {
        "run": "b",
        "parts": [["demo", "flight"], ["plays", "arrival"], ["label", "binaural"]],
    }
    assert joined_run_name([departure, arrival]) == "flight_departure+arrival_binaural"
    fast = {
        "run": "c",
        "parts": [
            ["demo", "flight"],
            ["plays", "departure"],
            ["changed", "step-8"],
            ["label", "binaural"],
        ],
    }
    assert joined_run_name([departure, fast]) == f"flight_departure_{DEFAULT_PART}+step-8_binaural"
    kemar = {"run": "d", "parts": [["demo", "flight"], ["label", "binaural-kemar"]]}
    plain = {"run": "e", "parts": [["demo", "flight"], ["label", "binaural"]]}
    assert joined_run_name([plain, kemar]) == "flight_binaural+binaural-kemar"
    assert joined_run_name([{"run": "x_y"}, {"run": "x_z"}]) == "x_y+x_z"  # no parts kept


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
def test_the_join_step_plays_finished_runs_in_turn(tmp_path: Path, capsys) -> None:
    """join: the finished runs (a video there, as the run's record names it), each after
    a title card, into one video in the work folder named after them, with the chapters
    of each at their place; fewer than two, or a run that is not there, is refused."""
    pytest.importorskip("PIL")
    scene = copy_examples(tmp_path, ["airport"])[0]
    demo = load(scene.parent / "flight_demo.py").at(scene.parent)
    work = demo.work_folder(demo.defaults())
    runs = {}
    for vehicle, seconds, tone in (("departure", 1.0, 440), ("arrival", 1.5, 660)):
        args = demo.resolved(argparse.Namespace(**vars(demo.defaults()) | {"vehicle": vehicle}))
        folder = run_folder(args)
        video = product(args, ".mp4")
        work.mkdir(parents=True, exist_ok=True)
        _video(video, seconds, tone)
        record_run(
            folder,
            demo=demo.name,
            title=demo.title,
            run=folder.name,
            parts=[list(part) for part in demo.run_parts(args)],
            command=demo.command("all", args, names_only=True),
            video=video.name,
            length_s=seconds,
            chapters=["0:00 Start"],
        )
        runs[vehicle] = folder.name
    record_run(work / "flight_unfinished_binaural", demo=demo.name, video=None)
    assert [run["run"] for run in finished_runs(demo, demo.resolved(demo.defaults()))] == sorted(
        runs.values()
    )
    assert demo.runs(demo.defaults()) == sorted(runs.values())
    with pytest.raises(SystemExit, match="no finished run of the flight demo"):
        demo.main(["join", "--runs", "flight_unfinished_binaural,flight_arrival_binaural"])
    with pytest.raises(SystemExit, match="join needs two finished runs"):
        demo.main(["join", "--runs", runs["arrival"]])
    order = f"{runs['departure']},{runs['arrival']}"
    assert demo.main(["join", "--runs", order]) == 0
    out = work / "flight_departure+arrival_binaural.mp4"
    assert out.exists() and f"wrote {out}" in capsys.readouterr().out
    assert _length_s(out) == pytest.approx(2 * 4.0 + 1.0 + 1.5, abs=0.15)
    chapters = (work / "flight_departure+arrival_binaural_chapters.txt").read_text().splitlines()
    assert chapters == [
        "0:00 1. flight_departure_binaural",
        "0:04 Start",
        "0:05 2. flight_arrival_binaural",
        "0:09 Start",
    ]
    cards = sorted(p.name for p in (work / "flight_departure+arrival_binaural_cards").iterdir())
    assert cards == ["00_flight_departure_binaural.png", "01_flight_arrival_binaural.png"]
    # The join is no run of its own: it records none, and "all" leaves it out.
    assert not (work / "flight_departure+arrival_binaural" / RUN_RECORD).exists()
    assert JOIN.name not in demo.all_steps()
