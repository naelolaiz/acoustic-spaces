"""The demo of any scene's pick of what plays (:mod:`acoustic_spaces.demos.scene_demo`):
``--sources`` by name or by the scene's group, the run that plays them, the walkers that
stand when their source is left out, and the tour ``--listeners`` orders; none of it
simulates or mixes (the runs themselves are in test_demos)."""

from __future__ import annotations

import argparse
import dataclasses
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
    nothing: it moves no mix, and its trajectory with Doppler, beside which no listener
    walks, no longer keeps the others from walking; ``--moving`` still moves what it names."""
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
    assert demo.check(args).startswith(
        "radio moves with Doppler, which no listener that moves hears: leave seat-b"
    )
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
    scene's without it, or stops in the order ``--tour`` gives them (every listener
    mixed), each turned up to the loudness of the loudest or, ``--levels
    as-simulated``, none; each still numbers its stop in the tour."""
    scene = _grouped(tmp_path)
    entries = [{"listener": "seat-a"}, {"listener": "seat-b"}]
    monkeypatch.setattr(scene_demo, "responses_of", lambda args, demo=None: entries)
    seats = {"seats": {name: {"rms_db": -20.0} for name in ("seat-a", "seat-b")}}
    for argv, mixed, order in (
        ((), ["seat-a", "seat-b"], ["seat-a", "seat-b"]),
        (("--listeners", "seat-b,seat-a"), ["seat-b", "seat-a"], ["seat-b", "seat-a"]),
        (("--tour", "seat-b,seat-a"), ["seat-a", "seat-b"], ["seat-b", "seat-a"]),
    ):
        resolved = _run(scene, *argv)[1]
        assert scene_demo.listeners_of(resolved, entries) == mixed
        stops = scene_demo.tour_of(resolved, seats, 60.0)
        assert [stop.key for stop in stops] == order
        assert [stop.at for stop in stops] == [0.0, 20.0]  # --seconds, 20 s, each
    # --levels: each stop turned up to the loudness of the loudest, or none turned up.
    louder = {"seats": {"seat-a": {"rms_db": -20.0}, "seat-b": {"rms_db": -26.0}}}
    for argv, gains in (((), [0.0, 6.0]), (("--levels", "as-simulated"), [0.0, 0.0])):
        resolved = _run(scene, *argv)[1]
        assert [stop.gain_db for stop in scene_demo.tour_of(resolved, louder, 60.0)] == gains
    assert _run(scene, "--levels", "as-simulated")[1].run_folder.name.endswith(
        "_levels-as-simulated_binaural"
    )
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
    assert drawing.title("seat-b") == ("seat-b", "Stop 1 of 2")
    assert drawing.title("seat-a") == ("seat-a", "Stop 2 of 2")
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
    assert one.title("seat-a") == ("seat-a", "Stop 1 of 1")


def test_the_timeline_is_shared_or_from_each_listener_s_start(tmp_path: Path) -> None:
    """``--timeline``: shared, the tour going from listener to listener on one clock,
    crossfaded; per listener, each from the clock's start until what moves is over for
    it and 3 s more (or ``--seconds``), as long as its mix at most, faded out and 1 s
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
    # Each until what moves is over for it: a listener standing while another walks is
    # heard as where nothing moves (mixes made before that hear every one to the end).
    seats["seats"] = {
        "seat-a": {"length_s": 30.0, "until_s": None},
        "seat-b": {"length_s": 30.0, "until_s": 9.0},
    }
    assert [s.stop for s in scene_demo.stops_from_start(names, seats, gains, None)] == [20.0, 12.0]
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


def test_a_run_s_start_sets_off_what_moves_and_the_open_air_holds_it(tmp_path: Path) -> None:
    """``--start`` (s) sets off the first trajectory that moves then and each other as
    much later as the scene has it, and is kept in the run's name; nothing moving, it
    changes nothing, and a time before the clock's start is refused. Outdoors a
    trajectory in use past the ground's edge or above its sky widens them to hold it,
    50 m past and 40 m above, in the scene the run simulates; ``--height`` is moot where
    no helicopter flies."""
    walker = {
        "name": "walker",
        "position": [5, 5, 1.5],
        "trajectory": {"to": [[20, 5, 1.5]], "start_s": 2.0},
    }
    runner = {
        "name": "runner",
        "position": [5, -5, 1.5],
        "trajectory": {"to": [[20, -5, 1.5]], "start_s": 5.0, "speed_m_s": 3.0},
    }
    scene = _outdoors(tmp_path / "field")
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data["listeners"] += [walker, runner]
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    demo = DEMO.at(scene.parent)
    args, resolved = _run(scene, "--start", "10")
    assert demo.check(args) is None and run_folder(resolved).name.endswith("_start-10_binaural")
    moved = {lis.name: lis for lis in scene_demo.moving_scene(resolved).listeners}
    assert moved["walker"].trajectory.start_s == 10.0 and moved["runner"].trajectory.start_s == 13.0
    assert scene_demo.moving_scene(_run(scene)[1]).listeners == Scene.from_yaml(scene).listeners
    assert demo.check(_run(scene, "--start", "-1")[0]) == (
        "--start is a time on the run's clock (s), 0 or more, not -1.0"
    )
    assert demo.check(_run(scene, "--start", "3", "--moving", "none")[0]) == (
        "--start changes nothing: nothing moves"
    )
    assert demo.check(_run(scene, "--height", "100")[0]) == (
        "--height changes nothing: no helicopter flies (a source with a rotorcraft model)"
    )
    # Within the open air: the scene as it is.
    assert scene_demo.run_scene_file(resolved) == scene
    # A walk past the ground's edge, 30 m up at its end (the sky is 20 m): the run's
    # scene holds it, and is simulated as such.
    data["listeners"][2]["trajectory"]["to"] = [[90, 5, 30]]
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    resolved = _run(scene)[1]
    ground = scene_demo.scene_of(resolved).geometry.ground
    assert ground.extent == ((-45.0, -55.0), (140.0, 55.0)) and ground.sky == 70.0
    path = scene_demo.run_scene_file(resolved)
    assert path == scene.parent / "demo" / "scenes" / "scene_open-air.yaml"
    assert Scene.from_yaml(path).geometry.ground.extent == ground.extent


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


def test_a_recording_shorter_than_what_moves_plays_again_from_its_start() -> None:
    """A feed shorter than what moves plays again from its start, AGAIN_S after it
    ends, until it lasts as long, and is cut there; one long enough stays as it is."""
    rate = 10
    x = np.arange(1, 31, dtype=np.float32).reshape(-1, 1)  # 3 s
    gap = int(scene_demo.AGAIN_S * rate)
    out = scene_demo.played_until(x, rate, 12.5)
    assert out.shape == (125, 1) and gap == 20
    assert np.array_equal(out[:30], x) and not out[30:50].any()
    assert np.array_equal(out[50:80], x) and not out[80:100].any()
    assert np.array_equal(out[100:], x[:25])
    assert scene_demo.played_until(x, rate, 3.0) is x
    assert scene_demo.played_until(x[:0], rate, 3.0).shape == (0, 1)


def test_a_run_gives_the_talkers_a_vocal_effort_and_another_recording(tmp_path: Path) -> None:
    """``--vocal-effort`` has every talker in use (a source with a speech spectrum) talk
    so in the run, and ``--recording`` plays another free recording, the sources that
    play it with a speech spectrum taking its talker's: a run that changes a talker's
    level simulates the scene so changed, written once into the work folder's scenes,
    and names its run and feeds after it; a run that changes nothing of it simulates the
    scene itself; and what nothing hears is refused."""
    scene = _any_scene(tmp_path)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data["recording"] = "ace_m4"
    talker = data["sources"][0]
    talker.pop("signal")
    talker["channel"] = "mono"
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    own = Scene.from_yaml(scene)
    resolved = _run(scene)[1]
    assert scene_demo.run_scene_file(resolved) == scene
    assert scene_demo.talkers_of(resolved) == ["talker"]
    args, resolved = _run(scene, "--vocal-effort", "raised")
    assert DEMO.at(scene.parent).check(args) is None
    file = scene_demo.run_scene_file(resolved)
    assert file == tmp_path / "room" / "demo" / "scenes" / "scene_vocal-effort-raised.yaml"
    changed = Scene.from_yaml(file)
    voice = changed.sources[0]
    assert voice.vocal_effort == "raised"
    assert np.allclose(voice.power_db(), own.sources[0].talking("raised").power_db())
    assert [s.power_db().tolist() for s in changed.sources[1:]] == [
        s.power_db().tolist() for s in own.sources[1:]
    ]
    made = file.stat().st_mtime_ns
    assert scene_demo.simulations(resolved)[0].scene == file
    assert scene_demo.run_scene_file(resolved).stat().st_mtime_ns == made  # not again
    assert run_folder(resolved).name == "scene_vocal-effort-raised_binaural"
    assert scene_demo.results_of(resolved).parent.name == "scene_vocal-effort-raised"
    assert _run(scene, "--voice", "raised")[1].vocal_effort == "raised"  # its old flag
    # A woman's talk: the talker takes a woman's speech spectrum, at its vocal effort
    # (the same A-weighted level 1 m in front, another sound power per band) ...
    resolved = _run(scene, "--recording", "ace_f1")[1]
    voice = scene_demo.scene_of(resolved).sources[0]
    assert voice.spectrum == "speech_female" and voice.vocal_effort == "normal"
    assert own.sources[0].vocal_effort == "normal"
    assert voice.level_db == voice.talking_db("normal") != own.sources[0].level_db
    assert scene_demo.run_scene_file(resolved).name == "scene_female-speech.yaml"
    # ... or, given a level in all in place of a vocal effort, at that level.
    data["sources"][0].pop("vocal_effort")
    data["sources"][0]["level_db"] = 70.0
    in_all = tmp_path / "room" / "in_all.yaml"
    in_all.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    voice = scene_demo.scene_of(_run(in_all, "--recording", "ace_f1")[1]).sources[0]
    assert voice.spectrum == "speech_female" and voice.level_db == 70.0
    assert scene_demo.plays(resolved)["talker"].startswith(
        "A woman talking (ACE corpus, talker F1)"
    )
    assert scene_demo.feeds_folder(resolved).name == "scene_ace_f1"
    # Another man's talk changes nothing the simulation hears.
    resolved = _run(scene, "--recording", "ace_m1")[1]
    assert scene_demo.run_scene_file(resolved) == scene
    assert scene_demo.scene_of(resolved).recording == "ace_m1"
    # A talker given a file of its own does not play the recording.
    args = _run(scene, "--recording", "ace_f1", "--input", f"talker={tmp_path / 'x.wav'}")[0]
    assert "no source plays the scene's recording" in DEMO.at(scene.parent).check(args)
    # A scene with no talker in use has nothing to talk louder.
    data["sources"][0]["spectrum"] = "flat"
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    args = _run(scene, "--vocal-effort", "loud")[0]
    assert "no source in use talks" in DEMO.at(scene.parent).check(args)


def test_a_still_says_the_sti_of_the_talker_heard_loudest() -> None:
    """A still says the STI of the speech of the playing talker loudest at its listener,
    with its rating (ISO 9921), from the report, or along the movement at that moment;
    with other sources playing, that it is of that speech alone."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import scene_pictures
    from acoustic_spaces.parameters import sti_rating

    assert [sti_rating(v) for v in (0.2, 0.3, 0.5, 0.62, 0.75, None)] == [
        "bad", "poor", "fair", "good", "excellent", "not defined",
    ]  # fmt: skip
    sources = [
        {"name": "guide", "position": [0, 0, 1.6], "spectrum": "speech_male"},
        {"name": "radio", "position": [4, 0, 1.0], "spectrum": "flat"},
    ]
    scene = {
        "name": "Room",
        "sources": sources,
        "listeners": [{"name": "seat", "position": [2, 2, 1.2]}],
    }
    report = {
        "responses": [
            {"source": "guide", "listener": "seat", "spl_a_db": 60.0, "speech": {"sti": 0.62}},
            {"source": "radio", "listener": "seat", "spl_a_db": 70.0, "speech": {"sti": 0.4}},
        ]
    }
    stops = [pipeline.TourStop("seat", 0.0, 0.0, 1.0)]

    def lines(playing, movement=None):
        drawing = scene_pictures.Pictures(
            scene, np.zeros((0, 2, 2)), report, {}, stops, playing=playing, movement=movement
        )
        return drawing.lines(0, 0.5 if movement else None)

    alone = lines(["guide"])
    assert "Speech of guide: STI 0.62, good" in alone
    assert not any("alone" in str(line) for line in alone)
    both = lines(["guide", "radio"])
    assert "Speech of guide: STI 0.62, good" in both
    assert ("Of that speech alone, the others not counted as noise", scene_pictures.MUTED) in both
    assert not any("Speech of" in str(line) for line in lines(["radio"]))
    moving = {
        "times": [0.0, 0.4],
        "levels_a_db": {"seat": {"guide": [60.0, 55.0]}},
        "sti": {"seat": {"guide": [0.7, 0.44]}},
    }
    assert "Speech of guide: STI 0.44, poor" in lines(["guide"], moving)


def test_a_run_moves_a_listener_or_turns_it_on_the_spot(tmp_path: Path) -> None:
    """``--stand LISTENER=X,Y`` stands a listener elsewhere in the run, its height kept,
    facing the nearest source in use, its trajectory moved and turned with it (the
    scene so changed simulated once); ``--turn LISTENER=T:YAW,...`` turns it on the spot
    in place of its trajectory, the yaws from the way it faces, which a binaural run
    hears through the head tracker; what does not read is refused."""
    from acoustic_spaces.walkthrough import head_turners, knots

    scene = _any_scene(tmp_path)
    own = Scene.from_yaml(scene)
    seat = next(lis for lis in own.listeners if lis.name == "seat-b")
    near = min(own.sources, key=lambda s: np.hypot(s.position[0] - 6.0, s.position[1] - 2.0))
    assert near.name == "radio"
    resolved = _run(scene, "--stand", "seat-b=6,2")[1]
    moved = next(lis for lis in scene_demo.scene_of(resolved).listeners if lis.name == "seat-b")
    assert moved.position == (6.0, 2.0, seat.position[2])
    yaw = np.degrees(np.arctan2(near.position[1] - 2.0, near.position[0] - 6.0))
    assert moved.orientation.yaw == pytest.approx(yaw, abs=0.01)
    file = scene_demo.run_scene_file(resolved)
    assert file.name == "scene_seat-b-at-6-2.yaml"
    assert next(lis for lis in Scene.from_yaml(file).listeners if lis.name == "seat-b") == moved
    turn = "seat-b=0:0,2:90,3:90,5:180"
    resolved = _run(scene, "--turn", turn)[1]
    turned = scene_demo.scene_of(resolved)
    path = knots(next(lis for lis in turned.listeners if lis.name == "seat-b"))
    assert path.times.tolist() == [0.0, 2.0, 3.0, 5.0]
    assert (path.angles[:, 0] - seat.orientation.yaw).tolist() == [0.0, 90.0, 90.0, 180.0]
    assert np.ptp(path.positions, axis=0).max() == 0.0  # on the spot
    assert head_turners(turned) == ["seat-b"]
    assert scene_demo.run_scene_file(resolved) == scene  # the still simulation the same
    assert scene_demo.moving_of(resolved) == ["seat-b"]
    # Both: it stands there and turns from the way it faces there.
    resolved = _run(scene, "--stand", "seat-b=6,2", "--turn", "seat-b=0:0,2:90")[1]
    both = next(lis for lis in scene_demo.scene_of(resolved).listeners if lis.name == "seat-b")
    assert both.trajectory.keyframes[0].orientation.yaw == pytest.approx(yaw + 90.0, abs=0.01)
    demo = DEMO.at(scene.parent)
    for flag, value, said in (
        ("--stand", "seat-b=6", "--stand takes LISTENER=X,Y in metres"),
        ("--stand", "nobody=1,1", "--stand: no listener 'nobody'"),
        ("--turn", "seat-b=2:0,1:90", "--turn takes LISTENER=T:YAW,... with the times rising"),
        ("--turn", "seat-b", "--turn takes LISTENER=T:YAW"),
    ):
        assert said in demo.check(_run(scene, flag, value)[0])


def test_an_example_s_old_flags_and_steps_read_as_their_general_forms() -> None:
    """An option takes a flag of an example's script in another form (``renamed``),
    each of several values turned into its own; a step runs under the name of the
    steps it took over (``aliases``)."""
    from acoustic_spaces.demos import SIMULATE, Demo, Option, Step, simulates

    ran: list[str] = []
    recording = Option(
        "recording",
        "a recording",
        None,
        "choice",
        choices=("ace_m4", "ace_f1"),
        renamed=(("--talker", lambda code: f"ace_{code.lower()}"),),
    )
    demo = Demo(
        name="old",
        example="",
        title="Old",
        description="An old script.",
        script=Path(__file__),
        steps=[
            Step("render", lambda args: ran.append("render"), "mix", aliases=("walk", "turn")),
            dataclasses.replace(SIMULATE, aliases=("scene",)),
        ],
        options=[recording],
    )
    parser = demo.parser()
    assert parser.parse_args(["walk", "--talker", "F1"]).recording == "ace_f1"
    assert parser.parse_args(["all", "--talker", "M4,F1"]).recording == ("ace_m4", "ace_f1")
    assert parser.parse_args(["all", "--recording", "ace_m4"]).recording == "ace_m4"
    assert parser.parse_args(["all"]).recording is None
    assert demo.step("turn").name == "render" and simulates(demo.step("scene"))
    assert demo.main(["walk", "--talker", "M4"]) == 0 and ran == ["render"]


def test_a_tour_reads_its_stops_and_cuts_them_where_the_music_turns() -> None:
    """``--tour`` reads LISTENER[:still][:LENGTH],...: each stop in the scene's weather
    or in still air, for phrases (a number), seconds (12s) or to the end; anything else
    is refused with the form. The stops follow one another from the recording's start:
    one in phrases ends on a phrase's boundary counted from the first downbeat, one in
    seconds after as many, one to the end at the end, one that gives no length after
    ``--seconds`` or 20 s, or, where none gives one and the recording is too short for
    that, each as long; stops of one mix in a row are one, a stop with no time left is
    left out and the last is cut at the end."""
    stops = scene_demo.tour_text("near:2, far:still:12s,far::end, near")
    assert [(s.listener, s.atmosphere, s.phrases, s.seconds, s.to_end) for s in stops] == [
        ("near", "scene", 2.0, None, False),
        ("far", "still", None, 12.0, False),
        ("far", "scene", None, None, True),
        ("near", "scene", None, None, False),
    ]
    assert [scene_demo.stop_key(s) for s in stops] == ["near", "far.still", "far", "near"]
    for bad in ("near:loud", "near:2:3", ":2", "near:0", "near:-1s"):
        with pytest.raises(ValueError, match="--tour"):
            scene_demo.tour_text(bad)
    tempo = (120.0, 0.5)  # a phrase of 8 bars of 4 beats at 120 bpm: 16 s
    tour = scene_demo.tour_text("near:2,far:still:12s,far:1,near")
    assert scene_demo.timed_tour(tour, 80.0, tempo) == [
        ("near", 0.0, 32.5),  # the upbeat and two phrases
        ("far.still", 32.5, 44.5),
        ("far", 44.5, 48.5),  # to the end of the phrase it starts in
        ("near", 48.5, 68.5),  # 20 s
    ]
    assert scene_demo.timed_tour(tour, 80.0, tempo, 10.0)[-1] == ("near", 48.5, 58.5)
    assert scene_demo.timed_tour(tour, 60.0, tempo)[-1] == ("near", 48.5, 60.0)  # cut
    short = scene_demo.tour_text("near:15s,far:0.25")  # ends at the next boundary
    assert scene_demo.timed_tour(short, 80.0, tempo)[-1] == ("far", 15.0, 16.5)
    joined = scene_demo.tour_text("near:5s,near:5s,far:end,near")
    assert scene_demo.timed_tour(joined, 30.0, None) == [("near", 0.0, 10.0), ("far", 10.0, 30.0)]
    plain = scene_demo.tour_text("near,far")  # as --listeners plays them: shared out
    assert scene_demo.timed_tour(plain, 30.0, tempo) == [("near", 0.0, 15.0), ("far", 15.0, 30.0)]
    assert scene_demo.timed_tour(plain, 60.0, tempo)[-1] == ("far", 20.0, 40.0)
    with pytest.raises(ValueError, match="no tempo"):
        scene_demo.timed_tour(scene_demo.tour_text("near:2"), 80.0, None)
    # A listener's stop in still air plays at the gain of the same listener in the
    # scene's own weather, so that the difference the weather makes is heard.
    seats = {"seats": {"near": {"rms_db": -10.0}, "far": {"rms_db": -30.0}}}
    seats["seats"]["far.still"] = {"rms_db": -26.0}
    gains = scene_demo.stops_gains(seats, ["near", "far.still", "far"])
    assert gains == {"near": 0.0, "far.still": 20.0, "far": 20.0}


def test_a_tour_of_a_movement_cuts_at_its_quiet_moments_and_rings_out(tmp_path: Path) -> None:
    """For a recording whose tours share each piece out (a symphony's movements), stops
    that give no length take even shares of the piece, each cut moved to the quietest
    moment within 10 s of where the share puts it (half the shortest stop at most), by
    the loudness of the first stop's mix, so that it falls between phrases; the last
    stop, which runs to the piece's end, runs on while its mix rings, until it is 60 dB
    below its loudest, so that the hall's reverberation dies away as simulated, and
    not at all after a recording that rings out of its own."""
    stops = scene_demo.tour_text("near,far")
    assert scene_demo.timed_tour(stops, 60.0, None, share=True) == [
        ("near", 0.0, 30.0),
        ("far", 30.0, 60.0),
    ]
    assert scene_demo.timed_tour(stops, 60.0, None)[-1] == ("far", 20.0, 40.0)
    assert scene_demo.timed_tour(stops, 60.0, None, 10.0, share=True)[-1] == ("far", 10.0, 20.0)
    # A piece of 30 s, quiet from 13 to 14 s: the cut moves from 15 s to there.
    rate = 8000
    t = np.arange(30 * rate) / rate
    x = np.where((t > 13.0) & (t < 14.0), 0.01, 0.5) * np.sin(2 * np.pi * 220 * t)
    seats = {"rate": rate, "format": "binaural", "seats": {}}
    # far: the hall rings on, 60 dB down 1 s after the music ends, its mix 2.5 s longer.
    t_ring = np.arange(round(2.5 * rate)) / rate
    ring = 0.5 * 10.0 ** (-3.0 * t_ring) * np.sin(2 * np.pi * 220 * (30.0 + t_ring))
    for key, x_of in (("near", x), ("far", np.concatenate([x, ring]))):
        audio.write(tmp_path / f"{key}.binaural.wav", np.stack([x_of, x_of], 1), rate)
        seats["seats"][key] = {"length_s": len(x_of) / rate}
    timed = scene_demo.timed_tour(stops, 30.0, None, share=True)
    cut = scene_demo.quiet_cuts(timed, stops, seats, tmp_path)
    assert [key for key, _, _ in cut] == ["near", "far"]
    assert 13.0 <= cut[0][2] <= 14.0 and cut[1][1] == cut[0][2] and cut[1][2] == 30.0
    # Within half a stop at most: three stops of 10 s, each cut within 5 s of its share,
    # the first at the quiet moment, the second, with none near, where it is quietest.
    three = scene_demo.tour_text("near,far,near")
    shares = scene_demo.timed_tour(three, 30.0, None, share=True)
    first, second = (b for _, _, b in scene_demo.quiet_cuts(shares, three, seats, tmp_path)[:-1])
    assert 13.0 <= first <= 14.0 and 15.0 <= second <= 25.25
    # A stop that gives its length keeps its cut.
    given = scene_demo.tour_text("near:15s,far")
    kept = scene_demo.timed_tour(given, 30.0, None, share=True)
    assert scene_demo.quiet_cuts(kept, given, seats, tmp_path) == kept
    rung = scene_demo.ringing_out(cut, seats, 30.0, tmp_path)[-1]
    assert rung[:2] == ("far", cut[1][1]) and rung[2] == pytest.approx(31.0, abs=0.35)
    early = [("near", 0.0, 12.0), ("far", 12.0, 20.0)]  # ends before the piece does
    assert scene_demo.ringing_out(early, seats, 30.0, tmp_path) == early
    # A mix that has rung out by the recording's end (its music over at 27 s, silence
    # padded to the end, as the church's): no silence after it.
    quiet = np.concatenate([np.where(t < 27.0, x, 0.0), np.zeros(round(2.5 * rate))])
    audio.write(tmp_path / "rung.binaural.wav", np.stack([quiet, quiet], 1), rate)
    seats["seats"]["rung"] = {"length_s": len(quiet) / rate}
    ended = [("far", 0.0, 15.0), ("rung", 15.0, 30.0)]
    assert scene_demo.ringing_out(ended, seats, 30.0, tmp_path) == ended


def _outdoors(folder: Path, seconds: float = 10.0, **more) -> Path:
    """A small outdoor scene in a wind, quick to simulate: a band 2 m up plays a tone of
    ``seconds`` to a listener 10 m away and one 40 m away; ``more`` adds to it."""
    from acoustic_spaces.demos import audio

    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "geometry": {"ground": {"extent": [[-20, -20], [60, 20]], "sky": 20}},
        "materials": {"field": {"impedance": "grass", "scattering": 0.1}},
        "default_material": "field",
        "atmosphere": {"wind": {"speed_m_s": 4.0, "direction_deg": 0}},
        "sources": [{"name": "band", "position": [0, 0, 2], "signal": "tone.wav"}],
        "listeners": [
            {"name": "near", "position": [10, 0, 1.5], "orientation": {"yaw": 180}},
            {"name": "far", "position": [40, 0, 1.5], "orientation": {"yaw": 180}},
        ],
        "simulation": {
            "ir_length_s": 0.3,
            "seed": 1,
            "sample_rate": 16_000,
            "early": {"method": "image_sources", "max_order": 1},
            "late": {"rays": 200},
        },
        "outputs": ["binaural"],
        **more,
    }
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    t = np.arange(int(seconds * 16_000)) / 16_000
    audio.write(folder / "tone.wav", (0.3 * np.sin(2 * np.pi * 300 * t))[:, None], 16_000)
    return folder / "scene.yaml"


def test_a_tour_is_refused_where_it_cannot_play(tmp_path: Path) -> None:
    """A tour that does not read, stops at a listener not in use, is in phrases where
    the recording has no tempo, hears a listener in still air where the scene's air is
    still already or where something moves, or goes on where each listener is heard
    from the clock's start, is refused, each saying why; ``--atmosphere`` is moot in a
    scene whose air is still."""
    walker = {"name": "walker", "position": [5, 5, 1.5], "trajectory": {"to": [[20, 5, 1.5]]}}
    scene = _outdoors(tmp_path / "field")
    demo = DEMO.at(scene.parent)

    def check(*argv: str) -> str | None:
        return demo.check(demo.parser().parse_args(["render", "--scene", scene.name, *argv]))

    assert check("--tour", "near,far:still") is None
    assert "--tour takes [PIECE/]LISTENER[:still][:LENGTH]" in check("--tour", "near:loud")
    assert "'nobody': no such listener here; it has near, far" in check("--tour", "nobody")
    assert "no tempo" in check("--tour", "far:2")
    assert check("--tour", "far:2", "--bpm", "120") is None
    assert "--downbeat is for a recording of yours" in check("--downbeat", "0.5")
    assert "positive" in check("--tour", "far:2", "--bpm", "0")
    assert check("--atmosphere", "still", "--tour", "far:still") is None
    assert "--tour stops at none of the listeners" in check("--tour", "near", "--listeners", "far")
    # --listeners: the scene's tour in the order it gives, a listener it does not visit in
    # a stop of its own; a --tour given in its own order, without the stops left out.
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data["tour"] = [{"listener": "far", "seconds": 5.0}, {"listener": "near", "phrases": 1}]
    data["listeners"].append({"name": "edge", "position": [0, 15, 1.5]})
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def tour(*argv: str) -> list[tuple[str, float | None]]:
        resolved = _run(scene, *argv)[1]
        stops = scene_demo.tour_given(resolved, scene_demo.scene_of(resolved))
        return [(stop.listener, stop.seconds or stop.phrases) for stop in stops]

    assert tour() == [("far", 5.0), ("near", 1.0)]
    assert tour("--listeners", "near,edge,far") == [("near", 1.0), ("edge", None), ("far", 5.0)]
    assert tour("--tour", "edge:3s,far,near", "--listeners", "near,edge") == [
        ("edge", 3.0),
        ("near", None),
    ]
    del data["tour"]
    data["listeners"][-1] = walker
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert "a tour plays on a shared timeline" in check("--tour", "near,far")
    assert "while walker moves" in check("--tour", "far:still", "--timeline", "shared")
    assert check("--tour", "far:still", "--moving", "none") is None
    indoors = _any_scene(tmp_path)
    demo = DEMO.at(indoors.parent)
    scene = indoors
    assert "where the scene's air is still already" in check("--tour", "seat-a:still")
    assert check("--atmosphere", "still") is not None  # moot: still already


def test_a_tour_hears_a_listener_in_the_weather_and_in_still_air(
    tmp_path: Path, monkeypatch
) -> None:
    """A scene's own tour: its stops in turn, a listener heard in the scene's wind and,
    from a second simulation of the scene in still air (``scenes/<name>_still-air.yaml``
    in the work folder), in still air; each still names its stop and the air, the
    heading counts each listener once; the stop in still air plays at the gain of the
    same listener in the wind, so that the tour's audio keeps the difference between
    the two mixes as simulated."""
    pytest.importorskip("PIL")
    from acoustic_spaces.cli import main as cli

    tour = [
        {"listener": "near", "seconds": 2.0},
        {"listener": "far", "atmosphere": "still", "seconds": 3.0},
        {"listener": "far", "to_end": True},
    ]
    scene = _outdoors(tmp_path / "field", tour=tour)
    monkeypatch.chdir(scene.parent)
    assert cli(["demo", "scene.yaml", "--no-mp4"]) == 0
    work = scene.parent / "demo"
    still = Scene.from_yaml(work / "scenes" / "scene_still-air.yaml")
    assert still.still and still.atmosphere is None
    assert [s.atmosphere for s in still.tour] == ["scene"] * 3
    seats = pipeline.read_json(work / "scene_binaural" / "seats" / "seats.json")
    assert list(seats["seats"]) == ["near", "far", "far.still"]
    assert seats["seats"]["far.still"]["atmosphere"] == "still"
    stills = sorted(p.name for p in (work / "scene_binaural" / "stills").glob("*.png"))
    assert stills == ["00_near.png", "01_far.still.png", "02_far.png"]
    heard, rate = audio.read(work / "scene_binaural.wav")
    folder = work / "scene_binaural" / "seats"

    def level(x: np.ndarray, a: float, b: float) -> float:
        return float(audio.db(np.sqrt(np.mean(x[int(a * rate) : int(b * rate)] ** 2))))

    windy, calm = (audio.read(folder / seats["seats"][k]["file"])[0] for k in ("far", "far.still"))
    # Clear of the crossfades (1 s, centred on the cuts) and of the last 2.5 s, faded out.
    simulated = level(calm, 3.0, 4.0) - level(windy, 5.6, 7.0)
    assert abs(simulated) > 1.0  # the wind changes what is heard 40 m away
    assert level(heard, 3.0, 4.0) - level(heard, 5.6, 7.0) == pytest.approx(simulated, abs=0.1)


def test_a_scene_s_tour_names_its_listeners_and_one_length_a_stop(tmp_path: Path) -> None:
    """A scene's ``tour``: each stop at a listener of the scene, in its weather or in
    still air (not where its air is still already), for one length at most; written back
    as short as it reads; its ``tours`` alike, the one for the recording a run plays in
    the tour's place."""
    from pydantic import ValidationError

    from acoustic_spaces.scene import TourStop

    with pytest.raises(ValidationError, match="one length .*not phrases and seconds"):
        TourStop(listener="near", phrases=2, seconds=3)
    with pytest.raises(ValidationError, match="not seconds and to_end"):
        TourStop(listener="near", seconds=3, to_end=True)
    tour = [{"listener": "near", "phrases": 2}, {"listener": "far", "atmosphere": "still"}]
    scene = Scene.from_yaml(_outdoors(tmp_path / "field", tour=tour))
    assert scene.to_data()["tour"] == tour
    data = scene.to_data()
    with pytest.raises(ValidationError, match="tour: no listener 'nobody'"):
        Scene.model_validate({**data, "tour": [{"listener": "nobody"}]})
    with pytest.raises(ValidationError, match="'far' in still air, where the scene's air"):
        Scene.model_validate({**data, "atmosphere": None})
    # Its tours, one for a recording named, each checked as its tour is; a run that plays
    # that recording (a free one, or a file of yours by its name) takes its tour.
    other = [{"listener": "far", "phrases": 1}, {"listener": "near", "to_end": True}]
    toured = Scene.model_validate({**data, "tours": {"potassium": other, "my_song": tour}})
    assert toured.to_data()["tours"] == {"potassium": other, "my_song": tour}
    with pytest.raises(ValidationError, match="tours: potassium: no listener 'nobody'"):
        Scene.model_validate({**data, "tours": {"potassium": [{"listener": "nobody"}]}})
    for played, stops in (
        ("potassium", other),
        (Path("songs/my_song.flac"), tour),
        ("rubidium", tour),  # none of its own: the scene's tour
        (None, tour),
    ):
        heard = scene_demo.toured_for(toured, played)
        assert [s.model_dump(exclude_defaults=True) for s in heard.tour] == stops
