"""The airport example's jets, of a Doc 29 model (:class:`acoustic_spaces.scene.AircraftModel`):
Doc 29's JETW taking off and landing along their trajectories' ground tracks, its sound
power and its thrust along its flight, and the flight demo, a preset of the demo of its
scene, its sound as loud as its levels say."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from acoustic_spaces.demos import aircraft
from acoustic_spaces.examples import example_path
from acoustic_spaces.parameters import A_WEIGHTING_DB, event_levels

RATE = 16_000


def _places(source) -> list:
    return [source.position, *(k.position for k in source.trajectory.keyframes)]


def test_the_examples_jets_are_doc_29_models() -> None:
    """The example's departure and arrival are of a Doc 29 model, their trajectories
    their ground tracks: the model flies each along its track (its keyframes Doc 29's
    flight, from its start of roll or touching down 290 m along the runway), with its
    sound power at its highest thrust, the old example's to 0.05 dB, and its engines'
    installation as directivity; the scene written again reads the same flights."""
    from acoustic_spaces import Scene
    from acoustic_spaces.doc29.source import directivity
    from acoustic_spaces.room import Room

    scene = Scene.from_yaml(example_path("airport"))
    written = {
        "departure": [142.6, 150.4, 152.1, 148.9, 148.3, 150.7, 148.5, 144.5],
        "arrival": [131.9, 142.1, 139.0, 139.8, 139.0, 137.5, 139.2, 138.5],
    }
    for source in scene.sources:
        model = source.model
        assert (model.method, model.aircraft, model.operation) == ("doc-29", "jetw", source.name)
        assert source.in_use == (source.name == "departure") and source.trajectory.doppler
        flight = source.model_flight()
        air = scene.air
        # Its keyframes, flown again, are the same flight.
        track = aircraft.flown(
            model.operation, _places(source), air.temperature_c, air.pressure_kpa
        )
        assert track.keyframes() == flight.keyframes()
        assert [k.time_s for k in source.trajectory.keyframes] == [
            k["time_s"] for k in flight.keyframes()
        ]
        np.testing.assert_allclose(source.level_db, written[source.name], atol=0.05)
        assert source.level_db == aircraft.flight_power_db(
            model.operation, flight, air.temperature_c, air.humidity_percent, air.pressure_kpa
        )
    heights = {s.name: s.position[2] for s in scene.sources}
    assert heights["departure"] == 1.0 and heights["arrival"] == pytest.approx(224.8294)
    room = Room.from_scene(scene.model_copy(update={"sources": [scene.sources[0]]}))
    ways = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.7, -0.7], [1.0, 0.0, 0.0]])
    (engines,) = room.directivities
    assert engines.energy(ways) == pytest.approx(directivity("Wing").energy(ways), rel=1e-9)
    data = scene.to_data()
    assert all("level_db" not in s and "directivity" not in s for s in data["sources"])
    again = Scene.model_validate(data)
    for a, b in zip(scene.sources, again.sources, strict=True):
        assert a.trajectory == b.trajectory and a.position == b.position
        assert a.level_db == b.level_db


def test_the_flights_go_as_doc_29_flies_them() -> None:
    """The departure from its start of roll, standing at first, off the runway and
    climbing, its thrust cut back; the arrival on its 3° approach touching down where its
    trajectory comes down to the runway, its reverse thrust its loudest."""
    from acoustic_spaces import Scene

    scene = Scene.from_yaml(example_path("airport"))
    up, down = (s.model_flight() for s in scene.sources)
    assert up.speeds[0] == 0.0 and up.positions[0] == (0.0, 0.0, 1.0)
    assert up.powers[0] == max(up.powers) and up.powers[-1] < 0.7 * up.powers[0]
    heights = np.asarray(up.positions)[:, 2]
    assert np.all(np.diff(heights) >= 0.0) and heights[-1] > 700.0
    assert np.all(np.diff(up.speeds) >= 0.0)
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
    # Its roll-out ends where Doc 29's does, short of its ground track's end.
    assert places[-1, 0] == pytest.approx(1582.6873) and 3.6 * down.speeds[-1] == pytest.approx(
        50.0, abs=0.1
    )


def _source(**changes) -> dict:
    data = {
        "name": "arrival",
        "position": [-110.0, 0.0, 21.0],
        "model": {"method": "doc-29", "operation": "arrival"},
        "trajectory": {"to": [[290.0, 0.0, 1.0], [890.0, 0.0, 1.0]], "doppler": True},
    }
    for key, value in changes.items():
        if key == "trajectory":
            data["trajectory"] = data["trajectory"] | value
        else:
            data[key] = value
    return data


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"trajectory": {"to": [[290.0, 0.0, 10.0]]}}, "first comes down to the runway"),
        (
            {"position": [0.0, 0.0, 1.0], "trajectory": {"to": [[6e-7, 0.0, 20.0]]}},
            "two places apart in plan",
        ),
        ({"trajectory": {"speed_m_s": 70.0}}, "give its trajectory no speed_m_s"),
        ({"trajectory": {"doppler": False}}, "doppler: true"),
        ({"level_db": [100.0] * 8}, "give it or level_db, not both"),
        ({"directivity": "cardioid"}, "give it or directivity, not both"),
    ],
)
def test_a_doc_29_model_says_what_keeps_a_jet_from_flying(change, said) -> None:
    """An arrival whose trajectory never comes down to the runway has nowhere to touch
    down; a trajectory standing still in plan (within 1 µm, as `Track.through` takes it)
    gives no ground track; Doc 29 sets the speeds, its pitch falls as it passes, and the
    model gives the sound power and the directivity."""
    from acoustic_spaces.scene import Source

    with pytest.raises(ValueError, match=said):
        Source.model_validate(_source(**change))


def test_a_trajectory_rising_straight_up_first_flies_as_without_that_place() -> None:
    """A trajectory whose first two places stand one above the other (its first leg
    vertical) has one point of the ground track there, the flight path's vertical
    projection onto the ground (Doc 29 Volume 2's glossary): the departure and the
    arrival fly as along it without its second place, not to positions of no number;
    and places on the straight line between two others change nothing of the flight."""
    from acoustic_spaces.scene import Source

    walks = {
        "departure": ([0.0, 0.0, 1.0], [[0.0, 0.0, 30.0], [3000.0, 0.0, 300.0]]),
        "arrival": (
            [-2000.0, 0.0, 300.0],
            [[-2000.0, 0.0, 200.0], [290.0, 0.0, 1.0], [890.0, 0.0, 1.0]],
        ),
    }
    # A place on the straight line between the places either side of it.
    between = {"departure": [1000.0, 0.0, 50.0], "arrival": [-1000.0, 0.0, 70.0]}
    for name, (start, to) in walks.items():
        model = {"method": "doc-29", "operation": name}
        flights = []
        for way in (to, to[1:], [to[0], between[name], *to[1:]]):
            source = Source.model_validate(
                {
                    "name": name,
                    "position": start,
                    "model": model,
                    "trajectory": {"to": way, "doppler": True},
                }
            )
            flights.append(source.model_flight())
        upright, plain, through = flights
        positions = np.asarray(upright.positions)
        assert np.all(np.isfinite(positions)) and math.isfinite(upright.duration_s)
        for other in (plain, through):
            assert positions == pytest.approx(np.asarray(other.positions), abs=1e-9)
            assert upright.times == pytest.approx(other.times, abs=1e-9)
    up = Source.model_validate(
        {
            "name": "departure",
            "position": walks["departure"][0],
            "model": {"method": "doc-29", "operation": "departure"},
            "trajectory": {"to": walks["departure"][1], "doppler": True},
        }
    ).model_flight()
    assert up.positions[0] == (0.0, 0.0, 1.0) and up.speeds[0] == 0.0


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
            each |= {k: v for k, v in _source().items() if k in ("position", "trajectory")}
    data["listeners"] = [{"name": "beside", "position": [300.0, 120.0, 1.2]}]
    data["simulation"].update({"sample_rate": RATE, "ir_length_s": 0.2, "seed": 1})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 200, "batches": 1}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_the_flight_demo_lands_the_jet_as_loud_as_its_levels_say(tmp_path) -> None:
    """The arrival on the quick airport, by the script's own command line: its engines'
    sound, its pass at the listener and its mix. Its sound follows its thrust (the
    reverse thrust after touchdown the loudest) as its levels do: in each second of the
    pass the mix is as loud against its levels as in the others, within a decibel. Doc
    29 times the flight, so --speed changes nothing. The pass as Doc 29 measures it
    (L_ASmax, the SEL 10 dB down) is the pass's, read through the slow time weighting."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import audio, load, pipeline, scene_demo
    from acoustic_spaces.demos.models import FADE_S, SOUND_RMS, gains_along
    from acoustic_spaces.demos.passby import blocks_loudness
    from acoustic_spaces.synthesis import split_bands

    folder = _airport(tmp_path)
    demo = load(folder / "flight_demo.py").at(folder)
    faster = ["all", "--vehicle", "arrival", "--speed", "200"]
    refused = demo.resolved(demo.parser().parse_args(faster))
    assert "--speed changes nothing: Doc 29 flies each aircraft" in demo.check(refused)
    run = ["all", "--vehicle", "arrival", "--step", "8", "--format", "mono", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None
    assert demo.moving(args) == ["arrival"]
    assert demo.main(run) == 0
    assert demo.run_folder(demo.parser().parse_args(run)).name == "flight_arrival_step-8_mono"
    (source,) = [s for s in scene_demo.moving_scene(args).sources if s.name == "arrival"]
    flight = source.model_flight()
    x, rate = audio.read(scene_demo.feeds_folder(args) / "arrival.wav")
    assert rate == RATE and len(x) == round(flight.keyframes()[-1]["time_s"] * RATE)
    # The made sound plays its thrust: around each moment as loud as its model says,
    # but the first and the last, whose blocks reach into its fades in and out.
    at = np.arange(0.5, flight.duration_s - 0.5, 1.0)
    told = 10.0 ** (gains_along(source)(at) / 10.0)
    measured = blocks_loudness(x, RATE, at, SOUND_RMS)
    gaps = 10.0 * np.log10(measured / told)[1:-1]
    assert gaps == pytest.approx(np.zeros(len(at) - 2), abs=0.5)
    assert np.ptp(10.0 * np.log10(told)) > 2.0
    seats = pipeline.read_json(demo.run_folder(args) / "moving" / pipeline.SEATS)
    heard = seats["movement"]["passes"]["beside"]
    assert math.isfinite(heard["sel_a_db"]) and heard["max_a_db"] > 80.0
    assert heard["about"].startswith("Sound power of arrival: ")
    assert "lb of thrust (Doc 29's twin jet JETW landing (a 3° approach), its NPD" in heard["about"]
    first, last = (round(3.6 * v, 1) for v in (flight.speeds[0], flight.speeds[-1]))
    assert heard["about"].endswith(f"from {first:g} to {last:g} km/h")
    # As Doc 29 measures it too (Volume 2: L_Amax slow-weighted, SEL over the 10 dB-down
    # time): under the highest of its places and the whole pass's SEL.
    measured = event_levels(heard["heard_s"], heard["levels_a_db"])
    assert heard["max_as_db"] == pytest.approx(measured.max_db, abs=0.01)
    assert heard["sel_10db_down_a_db"] == pytest.approx(measured.exposure_db, abs=0.01)
    assert heard["max_as_db"] < heard["max_a_db"] - 0.1
    assert heard["sel_10db_down_a_db"] < heard["sel_a_db"]
    # The mix against the levels, second by second (A-weighted, as heard).
    mix, _ = audio.read(demo.run_folder(args) / "moving" / seats["seats"]["beside"]["file"])
    times = np.asarray(heard["heard_s"], dtype=float)
    levels = np.asarray(heard["levels_a_db"], dtype=float)
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)
    bands = split_bands(np.asarray(mix)[:, 0], RATE) ** 2
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
    stills = demo.run_folder(args) / "stills"
    assert any(stills.glob("*.png"))


def test_the_departure_and_the_arrival_play_as_one_series() -> None:
    """--vehicle departure,arrival, as the script took it: a run of two parts, each an
    ordinary run of one jet, the departure's the script's own."""
    from acoustic_spaces.demos import load

    demo = load(Path(example_path("airport")).parent / "flight_demo.py")
    args = demo.parser().parse_args(["all", "--vehicle", "departure,arrival"])
    assert demo.check(args) is None
    assert demo.run_folder(args).name == "flight_each-departure+arrival_binaural"
    parts = demo.each(args)
    assert [demo.run_name(p) for p in parts] == ["flight_binaural", "flight_arrival_binaural"]
    assert [demo.moving(p) for p in parts] == [["departure"], ["arrival"]]
    for part in parts:
        assert demo.check(part) is None
