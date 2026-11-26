"""The demo of any scene's pick of what plays (:mod:`acoustic_spaces.demos.scene_demo`):
``--sources`` by name or by the scene's group, the run that plays them, the walkers that
stand when their source is left out, and the tour ``--listeners`` orders; none of it
simulates or mixes (the runs themselves are in test_demos)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_demos import RATE, _any_scene, _tone

from acoustic_spaces.demos import audio, mix_stem, pipeline, playing, run_folder, scene_demo
from acoustic_spaces.demos.scene_demo import DEMO
from acoustic_spaces.scene import Scene


def _grouped(tmp_path: Path) -> Path:
    """The quick shoebox of test_demos, its talker on the stage, its radio and its
    silent fan in the PA's mains and fills."""
    scene = _any_scene(tmp_path)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    for source, group in zip(data["sources"], ("stage", "PA/Mains", "PA/Fills"), strict=True):
        source["group"] = group
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return scene


def _run(scene: Path, *argv: str) -> tuple[argparse.Namespace, argparse.Namespace]:
    """The arguments of ``acoustic-spaces demo scene.yaml render`` with ``argv``, as
    given and resolved."""
    demo = DEMO.at(scene.parent)
    args = demo.parser().parse_args(["render", "--scene", scene.name, *argv])
    return args, demo.resolved(args)


def test_a_demo_of_any_scene_plays_the_sources_picked(tmp_path: Path) -> None:
    """``--sources`` lists every source in use by its group, each that has something to
    play ticked: by default those play, under the run's name of today; a pick names its
    run, by name or by group (a group of a group too), and a pick of what plays by
    default, however it is named, is the default run; a pick of silent sources alone,
    or of none the scene has, is refused, and so is an ``--input`` for a source left
    out."""
    scene = _grouped(tmp_path)
    demo = DEMO.at(scene.parent)
    args, resolved = _run(scene)
    assert [(s.name, s.group, s.on) for s in demo.source_list(resolved)] == [
        ("talker", "stage", True),
        ("radio", "PA/Mains", True),
        ("fan", "PA/Fills", False),
    ]
    assert demo.check(args) is None
    assert resolved.playing == ("talker", "radio")
    assert run_folder(resolved).name == "scene_binaural"
    assert mix_stem(resolved, "seat-a") == "seat-a"  # the mixes keep their names
    assert demo.option("sources").help == scene_demo.SOURCES_HELP
    assert demo.option("sources_play").default == "together"

    def picked(*argv: str) -> tuple[str, tuple | None, str | None]:
        given, resolved = _run(scene, *argv)
        return run_folder(resolved).name, resolved.playing, demo.check(given)

    assert picked("--sources", "radio") == ("scene_radio_binaural", ("radio",), None)
    assert picked("--sources", "PA") == ("scene_PA_binaural", ("radio", "fan"), None)
    assert picked("--sources", "PA/Mains,stage") == ("scene_binaural", ("talker", "radio"), None)
    assert picked("--sources", "talker,radio") == ("scene_binaural", ("talker", "radio"), None)
    assert picked("--sources", "fan")[2] == (
        "--sources picks only sources that play nothing (fan); those that play: talker, radio"
    )
    assert picked("--sources", "PA/Fills")[2].startswith("--sources picks only sources")
    assert picked("--sources", "nobody")[2] == (
        "--sources nobody: no such source or group here; it has talker, radio, fan, "
        "PA/Fills, PA/Mains, stage"
    )
    # A silent source given a file plays by default, and the run is named after it.
    audio.write(scene.parent / "hum.wav", _tone(1.0, 100.0, RATE, 0.3), RATE)
    name, playing, problem = picked("--input", "fan=hum.wav")
    assert (name, playing, problem) == ("scene_fan-hum_binaural", ("talker", "radio", "fan"), None)
    assert picked("--sources", "talker", "--input", "fan=hum.wav")[2] == (
        "--input fan: no such source plays here; it has talker"
    )


def test_a_demo_of_any_scene_plays_its_sources_in_turn(tmp_path: Path) -> None:
    """``--sources-play in-turn`` plays each top group of those that play by default,
    or each name ``--sources`` gives, in a part of its own, each an ordinary run; with
    ``--series listeners``, each listener a part too, outermost."""
    scene = _grouped(tmp_path)
    demo = DEMO.at(scene.parent)

    def parts(*argv: str) -> tuple[str, list[str], list[tuple | None]]:
        args, resolved = _run(scene, "--sources-play", "in-turn", *argv)
        assert demo.check(args) is None
        each = demo.each(args)
        return (
            run_folder(resolved).name,
            [run_folder(p).name for p in each],
            [p.playing for p in each],
        )

    assert parts() == (
        "scene_each-source_binaural",
        ["scene_stage_binaural", "scene_PA_binaural"],
        [("talker",), ("radio", "fan")],
    )
    assert parts("--sources", "radio,talker") == (
        "scene_each-radio+talker_binaural",
        ["scene_radio_binaural", "scene_talker_binaural"],
        [("radio",), ("talker",)],
    )
    name, names, _ = parts("--sources", "radio,talker", "--series", "listeners")
    assert names == [
        "scene_radio_seat-a_binaural",
        "scene_talker_seat-a_binaural",
        "scene_radio_seat-b_binaural",
        "scene_talker_seat-b_binaural",
    ]
    # A part whose source plays nothing is refused, by its name.
    args, _ = _run(scene, "--sources-play", "in-turn", "--sources", "radio,fan")
    assert demo.check(args) == (
        "scene_fan_binaural: --sources picks only sources that play nothing (fan); "
        "those that play: talker, radio"
    )


def test_a_demo_of_any_scene_mixes_the_sources_picked_alone(tmp_path: Path, monkeypatch) -> None:
    """The mixes of a pick hold the feeds and sound powers of the sources picked alone
    (as :func:`~acoustic_spaces.demos.pipeline.render_mixes` keeps them), each mix
    saying which; by default every feed, as today; the tour lasts as long as the
    longest of those picked plays, and the stills' line credits them alone."""
    scene = _grouped(tmp_path)
    table = {
        "rate": 16_000,
        "length_s": 6.0,
        "sources": {
            "talker": {"signal": str(scene.parent / "sounds" / "talk.wav"), "length_s": 6.0},
            "radio": {"signal": str(scene.parent / "radio.wav"), "length_s": 2.0},
        },
    }
    given: list[tuple[list, argparse.Namespace]] = []
    monkeypatch.setattr(scene_demo, "feeds_table", lambda args, demo=None: table)
    monkeypatch.setattr(
        scene_demo,
        "responses_of",
        lambda args, demo=None: [{"listener": "seat-a"}, {"listener": "seat-b"}],
    )
    monkeypatch.setattr(
        pipeline, "render_mixes", lambda mixes, folder, args, **kw: given.append((mixes, args))
    )
    powers = {s.name: s.sound_power_db for s in scene_demo.scene_of(_run(scene)[1]).sources}
    for argv, names in (((), ["talker", "radio"]), (("--sources", "PA"), ["radio"])):
        resolved = _run(scene, *argv)[1]
        scene_demo.render(resolved)
        mixes, args = given.pop()
        assert [mix.key for mix in mixes] == ["seat-a", "seat-b"]
        for mix in pipeline.only_playing(mixes, args):
            assert list(mix.feeds) == names
            assert mix.weights == {name: powers[name] for name in names}
            assert mix.info.get("sources") == (None if len(names) == 2 else names)
        fed = playing(args, list(table["sources"]))
        assert fed == names
        assert scene_demo.length_of(table, fed) == (6.0 if len(names) == 2 else 2.0)
        credit = scene_demo.plays_line(table, None, fed)
        assert credit == "Playing " + ", ".join(
            f"{name}: {Path(table['sources'][name]['signal']).name}" for name in names
        )
    assert scene_demo.plays_line(table) == "Playing talker: talk.wav, radio: radio.wav"


def test_a_source_left_out_stands_where_it_starts(tmp_path: Path) -> None:
    """Without ``--moving``, a source ``--sources`` leaves out does not move, as it plays
    nothing: it moves no mix, and its trajectory with Doppler, which plays alone, no longer
    keeps the others from walking; ``--moving`` still moves what it names."""
    scene = _grouped(tmp_path)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data["listeners"][1]["trajectory"] = {
        "keyframes": [
            {"time_s": 1.0, "position": [6.0, 2.5, 1.2]},
            {"time_s": 1.5, "position": [6.0, 3.5, 1.2]},
        ]
    }
    data["sources"][1]["trajectory"] = {"to": [[6.0, 2.0, 1.0]], "speed_m_s": 10.0, "doppler": True}
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    demo = DEMO.at(scene.parent)
    args, resolved = _run(scene)
    assert scene_demo.moving_of(resolved) == ["seat-b", "radio"]
    assert demo.check(args).startswith("radio moves with Doppler, which plays alone")
    args, resolved = _run(scene, "--sources", "talker")
    assert demo.check(args) is None
    assert scene_demo.moving_of(resolved) == ["seat-b"]
    radio = next(s for s in scene_demo.moving_scene(resolved).sources if s.name == "radio")
    assert radio.trajectory is not None and not radio.trajectory.enabled
    assert demo.trajectories(resolved) is not None
    args, resolved = _run(scene, "--sources", "talker", "--moving", "radio")
    assert scene_demo.moving_of(resolved) == ["radio"] and demo.check(args) is None
    args, resolved = _run(scene, "--sources", "radio", "--moving", "none")
    assert scene_demo.moving_of(resolved) == []


def test_the_tour_of_any_scene_goes_in_the_order_given(tmp_path: Path, monkeypatch) -> None:
    """The tour visits the listeners in the order ``--listeners`` gives them, the
    scene's without it; each still numbers its stop in the tour."""
    scene = _grouped(tmp_path)
    entries = [{"listener": "seat-a"}, {"listener": "seat-b"}]
    monkeypatch.setattr(scene_demo, "responses_of", lambda args, demo=None: entries)
    seats = {"seats": {name: {"rms_db": -20.0} for name in ("seat-a", "seat-b")}}
    for argv, order in (
        ((), ["seat-a", "seat-b"]),
        (("--tour", "seat-b,seat-a"), ["seat-b", "seat-a"]),
    ):
        resolved = _run(scene, *argv)[1]
        assert scene_demo.listeners_of(resolved, entries) == order
        stops = scene_demo.tour_of(resolved, seats, 60.0)
        assert [stop.key for stop in stops] == order
        assert [stop.at for stop in stops] == [0.0, 20.0]
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import scene_pictures
    from acoustic_spaces.scene import Scene

    report = {
        "bands_hz": [500, 1000],
        "responses": [
            {"source": s, "listener": lis, "spl_a_db": 70.0, "parameters": {"t30": [1.0, 1.0]}}
            for s in ("talker", "radio", "fan")
            for lis in ("seat-a", "seat-b")
        ],
    }
    drawing = scene_pictures.Pictures(
        Scene.from_yaml(scene).to_data(),
        np.zeros((0, 2, 2)),
        report,
        seats["seats"],
        stops,
        playing=["radio"],
    )
    assert drawing.title("seat-b") == ("seat-b", "Listener 1 of 2")
    assert drawing.title("seat-a") == ("seat-a", "Listener 2 of 2")
    assert [s["name"] for s in drawing.playing] == ["radio"]
    assert drawing.level_at("seat-b") == pytest.approx(70.0)  # the radio's alone
    assert drawing.shown(0) == "radio -> seat-b"
    one = scene_pictures.Pictures(
        Scene.from_yaml(scene).to_data(),
        np.zeros((0, 2, 2)),
        report,
        seats["seats"],
        stops[1:],
        playing=["talker", "radio"],
    )
    assert one.title("seat-a") == ("seat-a", "Listener 1 of 1")


def test_the_timeline_is_shared_or_from_each_listener_s_start(tmp_path: Path) -> None:
    """``--timeline``: shared, the tour going from listener to listener on one clock,
    crossfaded; per listener, each from the clock's start for the whole of what moves
    and 3 s more (or ``--seconds``), as long as its mix at most, faded out and 1 s
    apart; auto, the default, per listener when something moves."""
    scene = _grouped(tmp_path)
    resolved = _run(scene)[1]
    assert resolved.timeline == "auto" and resolved.seconds is None
    assert scene_demo.timeline_of(resolved, []) == "shared"
    assert scene_demo.timeline_of(resolved, ["seat-b"]) == "per-listener"
    resolved = _run(scene, "--timeline", "shared")[1]
    assert scene_demo.timeline_of(resolved, ["seat-b"]) == "shared"
    names, gains = ["seat-a", "seat-b"], {"seat-a": 0.0, "seat-b": 3.0}
    shared = scene_demo.shared_stops(names, gains, None, 60.0)
    assert [(s.at, s.start, s.stop) for s in shared] == [(0.0, 0.0, 20.0), (20.0, 20.0, 40.0)]
    assert all(s.joined_before and s.joined_after and s.fade_s == 0.0 for s in shared)
    assert [s.stop for s in scene_demo.shared_stops(names, gains, 30.0, 40.0)] == [20.0, 40.0]
    seats = {
        "moving": ["seat-b"],
        "movement": {"times": [0.0, 4.0, 12.5]},
        "seats": {"seat-a": {"length_s": 30.0}, "seat-b": {"length_s": 10.0}},
    }
    stops = scene_demo.stops_from_start(names, seats, gains, None)
    assert [(s.key, s.at, s.start, s.stop) for s in stops] == [
        ("seat-a", 0.0, 0.0, 15.5),
        ("seat-b", 16.5, 0.0, 10.0),
    ]
    assert [s.fade_s for s in stops] == [3.0, 3.0] and [s.gain_db for s in stops] == [0.0, 3.0]
    assert not any(s.joined_before or s.joined_after for s in stops)
    given = scene_demo.stops_from_start(names, seats, gains, 8.0)
    assert [(s.at, s.stop) for s in given] == [(0.0, 8.0), (9.0, 8.0)]
    still = scene_demo.stops_from_start(names, {"seats": seats["seats"]}, gains, None)
    assert [s.stop for s in still] == [20.0, 10.0]
    rate = 100
    folder = tmp_path / "mixes"
    for name in names:
        audio.write(folder / f"{name}.wav", np.ones((rate * 30, 2)) * 0.1, rate)
    seats = {"rate": rate, "format": "binaural", "seats": {n: {"file": f"{n}.wav"} for n in names}}
    two = [
        pipeline.TourStop(
            "seat-a", 0.0, 0.0, 2.0, joined_before=False, joined_after=False, fade_s=1.0
        ),
        pipeline.TourStop("seat-b", 3.0, 0.0, 2.0, joined_before=False, joined_after=False),
    ]
    heard = pipeline.tour_audio(two, folder, seats, fade_out_s=0.0, peak_db=None)
    assert heard.shape == (5 * rate, 2)
    np.testing.assert_allclose(heard[:rate, 0], 0.1)  # the first second as it is
    assert heard[rate, 0] == pytest.approx(0.1) and heard[2 * rate - 1, 0] < 1e-3
    np.testing.assert_allclose(heard[2 * rate : 3 * rate], 0.0)  # the gap
    np.testing.assert_allclose(heard[3 * rate :, 0], 0.1)  # the last fades with the tour


def test_a_run_s_speed_and_step_go_to_the_trajectories_that_move(tmp_path: Path) -> None:
    """``--speed`` (km/h) and ``--step`` (m) set every trajectory that moves in the run,
    and are kept with the mixes, so that a run at another speed mixes them again."""
    scene = _grouped(tmp_path)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data["listeners"][1]["trajectory"] = {"to": [[6.0, 3.5, 1.2]], "speed_m_s": 1.0}
    data["sources"][2]["trajectory"] = {"to": [[6.0, 2.0, 1.0]], "enabled": False}
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    resolved = _run(scene, "--speed", "7.2", "--step", "0.5")[1]
    moved = scene_demo.moving_scene(resolved)
    seat = next(lis for lis in moved.listeners if lis.name == "seat-b")
    assert seat.trajectory.speed_m_s == pytest.approx(2.0) and seat.trajectory.step_m == 0.5
    fan = next(s for s in moved.sources if s.name == "fan")
    assert fan.trajectory.speed_m_s == 1.2 and fan.trajectory.step_m is None  # it stands
    assert scene_demo.trajectories_of(resolved) == {"speed_kmh": 7.2, "step_m": 0.5}
    assert scene_demo.moving_scene(_run(scene)[1]).listeners == Scene.from_yaml(scene).listeners
    assert DEMO.option("speed").moot(resolved) is None
    assert DEMO.option("speed").moot(_run(scene, "--moving", "none")[1]) == "nothing moves"
    for flag in ("--speed", "--step", "--seconds"):
        args = _run(scene, flag, "0")[0]
        assert DEMO.at(scene.parent).check(args) == f"{flag} is a positive number, not 0.0"


def test_a_silent_walker_walks_unless_sources_leaves_it_out(tmp_path: Path) -> None:
    """A source the scene keeps silent walks in a default run, as before ``--sources``
    was there (the default pick leaves it out, but names no source); a ``--sources``
    given that leaves it out has it stand where it starts."""
    scene = _grouped(tmp_path)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data["sources"][2]["trajectory"] = {"to": [[6.0, 2.0, 1.0]], "speed_m_s": 2.0}
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    args, resolved = _run(scene)
    assert "fan" not in resolved.playing
    assert scene_demo.run_movers(resolved, scene_demo.scene_of(resolved)) is None
    assert scene_demo.moving_of(resolved) == ["fan"]
    fan = next(s for s in scene_demo.moving_scene(resolved).sources if s.name == "fan")
    assert fan.trajectory is not None and fan.trajectory.enabled
    args, resolved = _run(scene, "--sources", "talker")
    assert scene_demo.moving_of(resolved) == []
    fan = next(s for s in scene_demo.moving_scene(resolved).sources if s.name == "fan")
    assert fan.trajectory is not None and not fan.trajectory.enabled
