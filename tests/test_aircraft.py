"""The airport example's flight demo (:mod:`acoustic_spaces.demos.aircraft`): Doc 29's
JETW taking off and landing along its walks' ground tracks, its sound power and its
thrust along its flight, and its sound as loud as its levels say."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from acoustic_spaces.demos import aircraft
from acoustic_spaces.examples import example_path
from acoustic_spaces.parameters import A_WEIGHTING_DB

RATE = 16_000


def _args(folder: Path) -> argparse.Namespace:
    return argparse.Namespace(source=folder / "scene.yaml")


def test_the_examples_walks_are_doc_29s_flights() -> None:
    """The example's departure and arrival hold the demo's own flights (the walk's
    places at their times within half a metre of the flight's, as long), their sound
    power at their highest thrust and their engines' installation as directivity."""
    from acoustic_spaces import Scene
    from acoustic_spaces.directivity import read_directivity_csv
    from acoustic_spaces.doc29.source import directivity

    path = Path(example_path("airport"))
    scene = Scene.from_yaml(path)
    for plane in (aircraft.DEPARTURE, aircraft.ARRIVAL):
        source = next(s for s in scene.sources if s.name == plane.name)
        assert source.in_use == plane.departing and source.walk.doppler
        flown = plane.flight_of(scene)
        times = [0.0, *(k.time_s for k in source.walk.keyframes)]
        places = np.array([source.position, *(k.position for k in source.walk.keyframes)])
        assert times[-1] == pytest.approx(flown.duration_s, abs=1e-3)
        on = np.stack(
            [np.interp(times, flown.times, np.asarray(flown.positions)[:, k]) for k in range(3)],
            axis=1,
        )
        assert np.max(np.linalg.norm(on - places, axis=1)) < 0.5, plane.name
        info = plane.info(argparse.Namespace(source=path), scene, 0.0)
        assert source.level_db == info["level_db"]
        assert info["thrust_lb"] == max(flown.powers)
        table = read_directivity_csv(source.directivity.file)
        ways = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.7, -0.7], [1.0, 0.0, 0.0]])
        assert table.energy(ways) == pytest.approx(directivity("Wing").energy(ways), rel=0.01)
    heights = {s.name: s.position[2] for s in scene.sources}
    assert heights["departure"] == 1.0 and heights["arrival"] > 200.0


def test_the_flights_go_as_doc_29_flies_them() -> None:
    """The departure from its start of roll, standing at first, off the runway and
    climbing, its thrust cut back; the arrival on its 3° approach touching down where its
    walk comes down to the runway, its reverse thrust its loudest; each as the summary
    says."""
    from acoustic_spaces import Scene

    path = Path(example_path("airport"))
    scene = Scene.from_yaml(path)
    up = aircraft.DEPARTURE.flight_of(scene)
    assert up.speeds[0] == 0.0 and up.positions[0] == (0.0, 0.0, 1.0)
    assert up.powers[0] == max(up.powers) and up.powers[-1] < 0.7 * up.powers[0]
    heights = np.asarray(up.positions)[:, 2]
    assert np.all(np.diff(heights) >= 0.0) and heights[-1] > 700.0
    assert np.all(np.diff(up.speeds) >= 0.0)
    down = aircraft.ARRIVAL.flight_of(scene)
    places = np.asarray(down.positions)
    landed = int(np.argmax(places[:, 2] <= 1.0 + 1e-9))
    assert places[landed, 0] == pytest.approx(290.0)
    # Down its 3° glide path to 50 ft; below it, the source kept 1 m up (Doc 29's
    # lowest) flattens the last of it.
    high = places[: landed + 1][places[: landed + 1, 2] > 50.0 * 0.3048 + 1e-6]
    climb = np.degrees(np.arctan2(np.diff(high[:, 2]), np.diff(high[:, 0])))
    assert len(climb) >= 4 and climb == pytest.approx(np.full(len(climb), -3.0), abs=0.05)
    assert int(np.argmax(down.powers)) > landed
    assert np.all(np.diff(down.speeds) <= 0.0)
    words = aircraft.ARRIVAL.how_it_goes(_args(path.parent), scene)
    assert words.startswith("The landing jet flies 5583 m along its walk's ground track in ")
    assert "slowing from 252.8 to 50 km/h and coming down from 225 to 1 m up" in words
    assert "as Doc 29 flies JETW's Descend procedure in the scene's air" in words


def _airport(tmp_path: Path) -> Path:
    """The airport example, quick: the arrival from 400 m before its touchdown to 600 m
    after it, heard from a listener beside the runway alone, 16 kHz, seeded."""
    folder = tmp_path / "airport"
    shutil.copytree(Path(example_path("airport")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"]["ground"]["extent"] = [[-200, -200], [1000, 200]]
    data["geometry"]["ground"]["sky"] = 60
    for each in data["sources"]:
        if each["name"] == "arrival":
            each["position"] = [-110.0, 0.0, 21.0]
            each["walk"] = {"to": [[290.0, 0.0, 1.0], [890.0, 0.0, 1.0]], "doppler": True}
    data["listeners"] = [{"name": "beside", "position": [300.0, 120.0, 1.2]}]
    data["simulation"].update({"sample_rate": RATE, "ir_length_s": 0.2, "seed": 1})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 200, "batches": 1}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_the_flight_demo_lands_the_jet_as_loud_as_its_levels_say(tmp_path) -> None:
    """The arrival on the quick airport: its engines' sound, the scene with its flight as
    keyframes, its sound power and directivity, and the pass at the listener. Its sound
    follows its thrust (the reverse thrust after touchdown the loudest) as its levels
    do: in each second of the pass the mix is as loud against its levels as in the
    others, within a decibel. No --speed: Doc 29 times the flight."""
    pytest.importorskip("PIL")
    from acoustic_spaces import Scene
    from acoustic_spaces.demos import audio, load, pipeline
    from acoustic_spaces.demos.passby import FADE_S, blocks_loudness
    from acoustic_spaces.demos.road import SOUND_RMS
    from acoustic_spaces.synthesis import split_bands

    folder = _airport(tmp_path)
    demo = load(folder / "flight_demo.py").at(folder)
    assert all(o.name != "speed" for o in demo.options)
    with pytest.raises(SystemExit):
        demo.parser().parse_args(["all", "--speed", "200"])
    run = ["all", "--vehicle", "arrival", "--step", "8", "--format", "mono", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None
    assert demo.moving(args) == ["arrival"]
    assert demo.summary(args).startswith("The landing jet flies 1000 m along its walk's ground")
    assert demo.main(run) == 0
    work = folder / "demo"
    info = pipeline.read_json(work / "arrival" / "aircraft.json")
    assert info["procedure"] == "Descend" and info["thrust_lb"] == 10000.0
    x, rate = audio.read(work / "arrival" / "engines.wav")
    assert rate == RATE and len(x) == round(info["duration_s"] * RATE)
    # The made sound plays its thrust: around each moment as loud as sounding() says.
    at = np.arange(0.5, info["duration_s"] - 0.5, 1.0)
    plane = aircraft.ARRIVAL
    told = plane.sounding(_args(folder), "arrival", x, RATE, at)
    measured = blocks_loudness(x, RATE, at, SOUND_RMS)
    assert 10.0 * np.log10(measured / told) == pytest.approx(np.zeros(len(at)), abs=0.5)
    assert np.ptp(10.0 * np.log10(told)) > 2.0
    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    (source,) = [s for s in scene.sources if s.in_use]
    assert source.name == "arrival" and source.level_db == info["level_db"]
    assert source.walk.keyframes and source.walk.doppler
    assert source.directivity.file == folder / "scene_demo_arrival.csv"
    facts = pipeline.read_json(work / "walk" / "walk.json")
    fastest, slowest = (round(info[k], 1) for k in ("fastest_kmh", "slowest_kmh"))
    assert facts["speed_text"] == f"From {fastest:g} to {slowest:g} km/h"
    heard = facts["listeners"]["beside"]
    assert math.isfinite(heard["sel_a_db"]) and heard["max_a_db"] > 80.0
    # The mix against the levels, second by second (A-weighted, as heard).
    mix, _ = audio.read(work / "walk" / "beside.mono.wav")
    times = np.asarray(heard["heard_s"], dtype=float)
    levels = np.asarray(heard["levels_a_db"], dtype=float)
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)
    bands = split_bands(mix[:, 0], RATE) ** 2
    gaps = []
    # Not where the sound fades in and out (FADE_S of it, a little longer as heard
    # while the jet goes away).
    for start in np.arange(times[0] + FADE_S, times[-1] - 2.0 * FADE_S - 1.0, 1.0):
        lo, hi = int(start * RATE), int((start + 1.0) * RATE)
        played = float(weights @ bands[:, lo:hi].sum(axis=1))
        inside = (times >= start) & (times < start + 1.0)
        if inside.sum() < 2 or levels[inside].max() < heard["max_a_db"] - 15.0:
            continue
        expected = float(np.mean(10.0 ** (levels[inside] / 10.0)))
        gaps.append(10.0 * math.log10(played / expected))
    assert len(gaps) >= 5
    assert np.ptp(gaps) < 2.0 and np.std(gaps) < 0.6
    assert (work / "stills" / "00_beside.png").exists()


def test_the_flight_demo_says_what_keeps_a_jet_from_flying(tmp_path) -> None:
    """An arrival whose walk never comes down to the runway has nowhere to touch down,
    and a walk standing still gives no ground track."""
    from acoustic_spaces.demos import load

    folder = _airport(tmp_path)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    for each in data["sources"]:
        if each["name"] == "arrival":
            each["walk"]["to"] = [[290.0, 0.0, 10.0]]
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    demo = load(folder / "flight_demo.py").at(folder)
    args = demo.resolved(demo.parser().parse_args(["all", "--vehicle", "arrival"]))
    assert "first comes down to the runway, at most 1 m up" in demo.check(args)
    for each in data["sources"]:
        if each["name"] == "departure":
            each["walk"] = {"to": [[0.0, 0.0, 20.0]], "doppler": True}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    args = demo.resolved(demo.parser().parse_args(["all", "--vehicle", "departure"]))
    assert "two places apart in plan" in demo.check(args)


def test_each_jets_sound_is_as_long_as_its_walk() -> None:
    """The sound a flight is made for lasts as long as the walk the scene gets, whose
    keyframes' times are rounded: as many samples as its pass, which a sound as long as
    the flight itself missed by a sample or two on the departure."""
    from acoustic_spaces import Scene

    scene = Scene.from_yaml(example_path("airport"))
    for jet in (aircraft.DEPARTURE, aircraft.ARRIVAL):
        flown = jet.flight_of(scene)
        info = jet.info(_args(Path(example_path("airport")).parent), scene, 0.0)
        assert info["duration_s"] == flown.keyframes()[-1]["time_s"]
