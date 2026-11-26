"""The helicopter flyover: the example's helicopter as its rotors' model has it, and the
demo run on a quick copy of the example."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from acoustic_spaces import rotorcraft as rc
from acoustic_spaces.demos import audio, pipeline
from acoustic_spaces.demos import helicopter as helicopter_module
from acoustic_spaces.demos.helicopter import POWER_RATE, POWER_S
from acoustic_spaces.examples import example_path
from acoustic_spaces.scene import Scene


def test_the_examples_helicopter_is_a_rotorcraft_model() -> None:
    """The helicopter of the helicopter_flyover example flies level 150 m up at 63 m/s
    against the wind, of a rotorcraft model: its sound power and its directivity are
    those of its rotors' sound at its airspeed (its speed over the ground and the wind's
    at its height), in the scene's air, as the example had them written; each blade
    section's own noise (bpm) is quieter than Pegg's fit to measured rotors."""
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.room import Room

    path = Path(example_path("helicopter_flyover"))
    scene = Scene.from_yaml(path)
    (helicopter,) = scene.sources
    model = helicopter.model
    assert (model.method, model.helicopter, model.broadband) == ("rotorcraft", "as350_b3", "pegg")
    flight = helicopter.trajectory
    assert helicopter.in_use and helicopter.position[2] == 150.0
    assert flight.doppler and flight.face_forward and flight.speed_m_s == 63.0
    assert flight.step_m == 4.0
    wind = float(scene.atmosphere.wind_speed(np.array([150.0]))[0])
    airspeed = helicopter.model_speed_m_s()
    assert airspeed == pytest.approx(63.0 + wind, abs=1e-4)
    air = scene.air
    n = round(POWER_S * POWER_RATE)
    trim, sent = helicopter_module._sent(
        airspeed, air.temperature_c, air.pressure_kpa, POWER_RATE, n, "pegg"
    )
    level = sent.sound_power_db(trim.density, trim.sound_speed)
    np.testing.assert_allclose(helicopter.level_db, level, atol=1e-5)
    # The example's own numbers, written before the model gave them.
    written = [130.7, 130.5, 131.4, 130.4, 127.0, 125.2, 122.0, 120.2]
    np.testing.assert_allclose(helicopter.level_db, written, atol=0.051)
    (directivity,) = Room.from_scene(scene).directivities
    np.testing.assert_array_equal(directivity.table, sent.directivity().table)
    # 123.83 dB(A) of sound power in the IEC octaves by BPM against 132.95 by Pegg's
    # method, the tones in both.
    bpm = helicopter.model_copy(update={"model": model.model_copy(update={"broadband": "bpm"})})
    assert bpm.model_speed_m_s() == airspeed
    assert a_weighted(np.asarray(bpm.level_db)) == pytest.approx(123.83, abs=0.05)
    assert a_weighted(np.asarray(helicopter.level_db)) == pytest.approx(132.95, abs=0.05)


def test_a_rotorcraft_models_scene_reads_again_as_written() -> None:
    """The scene written again (as the editor saves it) leaves out the sound power and
    the directivity the model gives, so that it reads again, the same."""
    scene = Scene.from_yaml(Path(example_path("helicopter_flyover")))
    data = scene.to_data()
    (written,) = data["sources"]
    assert "level_db" not in written and "directivity" not in written
    again = Scene.model_validate(data)
    assert again.sources[0].level_db == scene.sources[0].level_db
    assert again.sources[0].model == scene.sources[0].model


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (
            {
                "trajectory": {
                    "to": [[450, 0, 160]],
                    "speed_m_s": 63,
                    "doppler": True,
                    "face_forward": True,
                }
            },
            "the rotorcraft model trims a helicopter for level flight, but its trajectory goes "
            "from 150.0 to 160.0 m up: give its points one height",
        ),
        (
            {"trajectory": {"to": [[450, 0, 150]], "doppler": True, "face_forward": False}},
            "a helicopter's rotors send their sound in its own frame, which faces the way it "
            "flies: give its trajectory to and speed_m_s, facing forward (not face_forward: "
            "false, nor keyframes, which set how it faces)",
        ),
        (
            {
                "trajectory": {
                    "keyframes": [{"time_s": 14.0, "position": [450, 0, 150]}],
                    "doppler": True,
                }
            },
            "facing forward (not face_forward: false, nor keyframes, which set how it faces)",
        ),
        (
            {"trajectory": {"to": [[450, 0, 150]], "speed_m_s": 63, "face_forward": True}},
            "a helicopter's rotors send their sound each way, down each path with its changing "
            "delay: give its trajectory doppler: true",
        ),
        (
            {"trajectory": None},
            "the rotorcraft model trims a helicopter for level flight at one steady speed: give "
            "the source a trajectory",
        ),
        (
            {"directivity": "omni"},
            "the rotorcraft model gives the source's directivity: give it or directivity, not both",
        ),
        (
            {"level_db": 120.0},
            "model sets the source's sound power: give it or level_db, not both",
        ),
    ],
)
def test_a_rotorcraft_model_flies_level_facing_its_way(change, said) -> None:
    """A helicopter of a rotorcraft model flies level, facing the way it flies, with
    Doppler, and its model gives its sound power and its directivity: each else is
    refused, saying why."""
    import re

    from pydantic import ValidationError

    from acoustic_spaces.scene import Source

    given = {
        "name": "helicopter",
        "position": [-450, 0, 150],
        "model": {"method": "rotorcraft"},
        "trajectory": {
            "to": [[450, 0, 150]],
            "speed_m_s": 63,
            "doppler": True,
            "face_forward": True,
        },
    }
    given |= change
    if given["trajectory"] is None:
        del given["trajectory"]
    with pytest.raises(ValidationError, match=re.escape(said)):
        Source.model_validate(given)


def _flyover(tmp_path: Path) -> Path:
    """The helicopter_flyover example, quick: 40 m of its flight right over the listener
    under its track, heard there and 150 m to its left, 16 kHz. The copy's folder is
    named apart from the example's: demo scripts are imported under their folder's name,
    so a copy named as the example would stand in for it in later tests."""
    folder = tmp_path / "flyover_copy"
    shutil.copytree(Path(example_path("helicopter_flyover")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    (helicopter,) = data["sources"]
    helicopter["position"] = [-20, 0, 150]
    helicopter["trajectory"]["to"] = [[20, 0, 150]]
    data["listeners"] = [
        lis for lis in data["listeners"] if lis["name"] in ("under_track", "left_150m")
    ]
    data["simulation"].update({"sample_rate": 16_000, "ir_length_s": 0.7, "seed": 7})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 500}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_the_flyover_demo_flies_the_helicopter_over(tmp_path: Path) -> None:
    """Every step on the quick flyover, the script's command line: its rotors' sound
    sent each way, its feed what it sends straight down; the scene with the helicopter's
    sound power at its airspeed, found at 48 kHz whatever the simulation's rate; the
    pass (the pitch falling) and the stills; what it sends each way as loud, for its
    sound power, as any made sound is; and in the mixes each listener as far under the
    other as the pass's levels say, the rotors' pattern as their directivity has it.
    On a seed, so that the late part's rays are the same in the mixes and in the
    simulation the test makes again."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load, models, scene_demo
    from acoustic_spaces.demos.helicopter import flight_emission
    from acoustic_spaces.directivity import sphere_mean
    from acoustic_spaces.room import Room
    from acoustic_spaces.walkthrough import simulate_movement

    folder = _flyover(tmp_path)
    demo = load(folder / "flyover_demo.py").at(folder)
    run = ["all", "--step", "8", "--format", "mono", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None and demo.run_name(args) == "flyover_step-8_mono"
    assert demo.plays(args)["helicopter"].startswith(
        "an AS350 B3's two rotors at 67.5 m/s through the air, their tones from the blades' "
        "steady loads (Ffowcs Williams and Hawkings) and their broadband noise by Pegg's method"
    )
    assert demo.main(run) == 0
    feeds = scene_demo.feeds_folder(args)
    fed = pipeline.read_json(feeds / scene_demo.FEEDS)["sources"]["helicopter"]
    assert fed["made"] and fed["each_way"]
    below, rate = audio.read(feeds / "helicopter.wav")
    assert rate == 16_000 and len(below) == round(40.0 / 63.0 * rate)
    scene = scene_demo.moving_scene(args)
    (helicopter,) = scene.sources
    example = Scene.from_yaml(Path(example_path("helicopter_flyover"))).sources[0]
    assert helicopter.model_speed_m_s() == example.model_speed_m_s()
    assert helicopter.level_db == example.level_db
    seats = pipeline.read_json(demo.run_folder(args) / scene_demo.MOVING / pipeline.SEATS)
    passes = seats["movement"]["passes"]
    under = passes["under_track"]
    assert under["closest_m"] == pytest.approx(math.hypot(4.0, 148.8), abs=0.01)  # x = ±4 m
    assert "(its rotors' model, 67.5 m/s through the air)" in under["about"]
    c = 331.3 * math.sqrt(1.0 + scene.air.temperature_c / 273.15)
    towards = 63.0 * 20.0 / math.hypot(20.0, 148.8)
    falls = 12.0 * math.log2((c + towards) / (c - towards))
    assert under["pitch_semitones"] == pytest.approx(falls, abs=0.02)
    assert (demo.run_folder(args) / "stills" / "00_under_track.png").exists()
    # What it sends in the IEC octaves, over the sphere, is a made sound's SOUND_RMS for
    # its sound power: at 48 kHz, where every band is under half the rate.
    air = scene.air
    sent = flight_emission(helicopter, 48_000, air.temperature_c, air.pressure_kpa)
    in_bands = float(np.sum(sphere_mean(sent._grid_powers("octave"))))
    assert in_bands == pytest.approx(models.SOUND_RMS**2, rel=1e-6)
    # The two listeners in the mixes, one gain for both, against the pass's levels.
    # Those add the direct sound's and the grass's reflection's energies band by band,
    # where the mixes add their pressures: the comb of the two, 1.2 m over the grass,
    # and the rotors' tones in it leave the low octaves up to 3 dB apart, 0.26 dB in
    # all (A-weighted); at 2 and 4 kHz the comb's teeth are many to the octave, and there
    # what the rotors send each way is heard as their directivity has it.
    energy, octaves = {}, {}
    moving = demo.run_folder(args) / scene_demo.MOVING
    for name in ("under_track", "left_150m"):
        x, _ = audio.read(pipeline.mix_path(moving, seats, name))
        x = np.asarray(x).reshape(len(x), -1)[:, 0]
        energy[name] = _a_energy(x, rate)
        power = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1.0 / rate)
        octaves[name] = [
            float(np.sum(power[(f >= c / math.sqrt(2.0)) & (f < c * math.sqrt(2.0))]))
            for c in (2000.0, 4000.0)
        ]
    told = passes["under_track"]["sel_a_db"] - passes["left_150m"]["sel_a_db"]
    heard = 10.0 * math.log10(energy["under_track"] / energy["left_150m"])
    assert told > 2.0 and heard == pytest.approx(told, abs=0.5)
    room = Room.from_scene(scene)
    movement = simulate_movement(room)
    power = helicopter.power_db("octave")
    simulated = {}
    for name in ("under_track", "left_150m"):
        bands = np.array(
            [
                movement.result[movement.key("helicopter", name, k)].levels_or_held(power)
                for k in range(len(movement.times))
            ]
        )
        simulated[name] = 10.0 * np.log10(np.sum(10.0 ** (bands[:, 5:7] / 10.0), axis=0))
    for k in range(2):
        told = simulated["under_track"][k] - simulated["left_150m"][k]
        heard = 10.0 * math.log10(octaves["under_track"][k] / octaves["left_150m"][k])
        assert heard == pytest.approx(told, abs=0.2), k


def _a_energy(x: np.ndarray, rate: int) -> float:
    """The A-weighted energy of the mono ``x`` at ``rate``, by its spectrum."""
    from acoustic_spaces.demos.audio import a_gain

    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    return float(np.sum(np.abs(np.fft.rfft(x) * a_gain(f)) ** 2))


def test_the_stadiums_helicopter_is_its_rotors_model() -> None:
    """The stadium_concert example's helicopter, left out of the concert's simulations,
    of a rotorcraft model, crosses the stadium 150 m up at 63 m/s through still air,
    setting off so that it is over the centre spot 97.5 s into the run, halfway through
    "Spicy Funk Cake": its sound power is its rotors' at that airspeed (as the example
    had it written), found along the way it sets off, not where it stands before."""
    from acoustic_spaces.demos import sibling

    path = Path(example_path("stadium_concert"))
    scene = Scene.from_yaml(path)
    helicopter = next(s for s in scene.sources if s.name == "helicopter")
    flight = helicopter.trajectory
    assert not helicopter.in_use and helicopter.model.method == "rotorcraft"
    assert flight.doppler and flight.face_forward and helicopter.position[2] == 150.0
    assert flight.speed_m_s == 63.0 and flight.step_m == 4.0 and flight.start_s == 87.976
    assert helicopter.model_speed_m_s() == 63.0
    written = [129.8, 129.5, 130.8, 130.2, 126.9, 125.1, 122.0, 120.1]
    np.testing.assert_allclose(helicopter.level_db, written, atol=0.051)
    crossing = sibling(path.parent / "stadium_demo.py", "helicopter_demo")
    takes = crossing.over_the_centre_s(argparse.Namespace(scene=path, speed=None))
    assert takes == pytest.approx(600.0 / 63.0) and flight.start_s + takes == pytest.approx(
        97.5, abs=1e-3
    )
    assert helicopter.model_copy(
        update={"trajectory": flight.model_copy(update={"start_s": 0.0})}
    ).level_db == pytest.approx(helicopter.level_db)


def test_a_run_flies_a_helicopter_at_its_height(tmp_path: Path) -> None:
    """``--height`` (m) flies every helicopter of the run at that height, at every place
    of its trajectory: its airspeed the wind's there, so its sound power that of the
    same helicopter written at that height in the scene, and the sky raised over it; the
    scene so changed simulated once, named after it."""
    from acoustic_spaces.demos import scene_demo

    folder = tmp_path / "flyover"
    shutil.copytree(Path(example_path("helicopter_flyover")).parent, folder)
    demo = scene_demo.DEMO.at(folder)
    args = demo.parser().parse_args(["render", "--scene", "scene.yaml", "--height", "250"])
    assert demo.check(args) is None
    resolved = demo.resolved(args)
    run = scene_demo.scene_of(resolved)
    (flown,) = run.sources
    assert flown.position[2] == 250.0 and [p[2] for p in flown.trajectory.to] == [250.0]
    assert run.geometry.ground.sky == 290.0  # 40 m above it; the scene's sky is 200 m
    path = scene_demo.run_scene_file(resolved)
    assert path == folder / "demo" / "scenes" / "scene_height-250.yaml"
    (written,) = Scene.from_yaml(path).sources
    assert flown.model_speed_m_s() == written.model_speed_m_s()
    np.testing.assert_allclose(flown.level_db, written.level_db, atol=1e-9)
    (own,) = Scene.from_yaml(folder / "scene.yaml").sources
    assert flown.model_speed_m_s() > own.model_speed_m_s()  # more wind against it higher up


@pytest.mark.parametrize("model", ["pegg", "bpm"])
def test_the_helicopters_rotors_are_found_once_for_every_length(model: str, monkeypatch) -> None:
    """The demo trims the helicopter and finds its rotors' tones and, by BPM, its noise's
    bands once, then makes each length its steps send from them (its sound power's
    second and its pass): sample for sample what :func:`rotorcraft.emission` makes at
    that length. Coarse tones and blade sums keep it quick; both ways use them."""
    calls = dict.fromkeys(("tones", "bpm_bands"), 0)

    def counted(name: str, **coarse):
        f = getattr(rc, name)

        def call(*args, **kwargs):
            calls[name] += 1
            return f(*args, **coarse | kwargs)

        monkeypatch.setattr(rc, name, call)

    counted("tones", harmonics=6, stations=6)
    counted("bpm_bands", segments=4, azimuths=8)
    # Caches of the test's own, so that nothing coarse stays in the demo's.
    for name in ("_rotors", "_sent"):
        fresh = functools.lru_cache(getattr(helicopter_module, name).__wrapped__)
        monkeypatch.setattr(helicopter_module, name, fresh)
    rate, air = 8000, {"temperature_c": 20.0, "pressure_kpa": 101.325}
    made = {
        n: helicopter_module._sent(40.0, air["temperature_c"], air["pressure_kpa"], rate, n, model)
        for n in (rate, 5077)
    }
    assert calls == {"tones": 2, "bpm_bands": 2 if model == "bpm" else 0}
    trim = rc.trim(dataclasses.replace(rc.AS350_B3, broadband=model), 40.0, **air)
    ways = np.random.default_rng(1).normal(size=(5077, 3))
    ways /= np.linalg.norm(ways, axis=1, keepdims=True)
    for n, (_, sent) in made.items():
        assert sent.length == n
        old = rc.emission(trim, seconds=n / rate, sample_rate=rate, seed=helicopter_module.SEED)
        for part, old_part in zip(sent.parts, old.parts, strict=True):
            assert type(part) is type(old_part)
            for field in dataclasses.fields(part):
                np.testing.assert_array_equal(
                    getattr(part, field.name), getattr(old_part, field.name), err_msg=field.name
                )
        np.testing.assert_array_equal(sent.sent(0, 5077, ways), old.sent(0, 5077, ways))


def _stadium(tmp_path: Path) -> Path:
    """The stadium_concert example, quick: two listeners and no tour, 16 kHz, responses
    0.3 s long after the sound arrives, few rays, in a folder named apart from the
    example's (see :func:`_flyover`)."""
    folder = tmp_path / "stadium_copy"
    shutil.copytree(Path(example_path("stadium_concert")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["listeners"] = [
        lis for lis in data["listeners"] if lis["name"] in ("mix_position", "singer")
    ]
    for tour in ("tour", "tours"):  # they visit the listeners left out
        data.pop(tour, None)
    data["simulation"].update({"sample_rate": 16_000, "ir_length_s": 0.3})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 300}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_the_crossing_is_a_preset_of_the_demo_of_the_stadiums_scene(tmp_path: Path, capsys) -> None:
    """helicopter_demo.py is the demo of the stadium's scene with the helicopter taken
    in, on one clock with the concert, listed after the stadium's own demo: the
    helicopter always plays, beside the concert's sources ``--sources`` picks (in each
    part with ``--sources-play in-turn``); the script's ``--at`` sets it off so that it
    is over the centre spot then, a time it cannot get there by refused, and
    ``--height`` flies it higher; the scene the run simulates holds its flight in the
    open air; each command line of the script it was reads as the demo's, and its steps
    run as those that took them over."""
    from acoustic_spaces.demos import demo_for, demos_of, load, run_folder, scene_demo
    from acoustic_spaces.room import Room

    folder = _stadium(tmp_path)
    demo = load(folder / "helicopter_demo.py").at(folder)
    assert [d.name for d in demos_of("stadium_concert")] == ["stadium", "crossing"]
    assert demo_for("stadium_concert").name == "stadium" and demo.builds_on == "stadium"
    assert (demo.name, demo.example, demo.title) == (
        "crossing",
        "stadium_concert",
        "A helicopter over the stadium concert",
    )

    def run(*argv: str) -> argparse.Namespace:
        args = demo.parser().parse_args(["all", *argv])
        assert demo.check(args) is None, argv
        return demo.resolved(args)

    args = run()
    assert run_folder(args) == folder / "demo" / "crossing_binaural"
    assert args.sources == "PA,Monitors,Band,helicopter" and args.timeline == "shared"
    assert scene_demo.moving_of(args) == ["helicopter"]
    sources = Scene.from_yaml(folder / "scene.yaml").sources
    played = [s.name for s in sources if s.in_use or s.name == "helicopter"]
    assert list(args.playing) == played and "singer_mic" not in played and len(played) == 23
    assert run("--sources", "PA").sources == "PA,helicopter"
    assert run("--sources", "PA,Helicopter").sources == "PA,Helicopter"
    turns = run("--sources", "PA,Monitors", "--sources-play", "in-turn")
    assert turns.sources == "PA+helicopter,Monitors+helicopter"
    assert run("--sources-play", "in-turn").sources == (
        "PA+helicopter,Monitors+helicopter,Band+helicopter"
    )
    # --at: over the centre spot then, 600 m from where it sets off at 63 m/s.
    at = run("--at", "180")
    assert at.start == pytest.approx(180.0 - 600.0 / 63.0, abs=1e-3)
    assert run_folder(at).name == "crossing_start-170p476_binaural"
    flown = {s.name: s for s in scene_demo.moving_scene(at).sources}["helicopter"]
    assert flown.trajectory.start_s == at.start and flown.model_speed_m_s() == 63.0
    faster = run("--at", "180", "--speed", "360")
    assert faster.start == pytest.approx(180.0 - 6.0, abs=1e-3)
    parser = demo.parser()
    for argv, said in (
        (
            ["--at", "5"],
            "--at is 5 s, but the helicopter takes 9.5 s to get over the stadium: "
            "give --at 10 or later",
        ),
        (
            ["--at", "100", "--start", "3"],
            "--at and --start both say when the helicopter flies: give one or the other",
        ),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["all", *argv])
        assert capsys.readouterr().err.strip().endswith(f"error: {said}")
    # The scene the run simulates: the stadium with the helicopter, the ground and the
    # sky widened to take its flight, 50 m past and 40 m above it.
    high = run("--height", "330")
    assert run_folder(high).name == "crossing_height-330_binaural"
    path = scene_demo.run_scene_file(high)
    assert path == folder / "demo" / "scenes" / "scene_with-helicopter_height-330.yaml"
    written = Scene.from_yaml(path)
    ground = written.geometry.ground
    assert ground.extent == ((-650.0, -82.0), (650.0, 82.0)) and ground.sky == 370.0
    helicopter = {s.name: s for s in written.sources}["helicopter"]
    assert helicopter.in_use and helicopter.position == (-600.0, 0.0, 330.0)
    assert helicopter.trajectory.to == [(600.0, 0.0, 330.0)]
    # Never into the stadium's roof, 30 m up, nor under it, as the script refused.
    low = demo.parser().parse_args(["all", "--scene", str(folder / "scene.yaml"), "--height", "30"])
    assert demo.check(low) == (
        "--height 30 flies helicopter into the model, 30 m up under its trajectory: "
        "give --height more than 30"
    )
    assert Room.from_scene(scene_demo.moving_scene(high)).problems() == []
    # The script's command lines and steps.
    read = {
        ("all", "--song", "francium"): {"recording": "francium"},
        ("all", "--from", "other.yaml"): {"scene": "other.yaml"},
        ("all", "--tour", "mix_position:1,pitch_side:0,singer:2"): {
            "tour": "mix_position:1,singer:end"
        },
    }
    for argv, given in read.items():
        args = parser.parse_args(list(argv))
        assert {name: getattr(args, name) for name in given} == given, argv
    assert demo.steps_named("flight") == ["simulate"] and demo.steps_named("scene") == ["simulate"]
    assert demo.steps_named("helicopter") == ["feeds"]
    assert demo.steps_named("walk") == ["render"] and demo.steps_named("mix") == ["render"]
    wav = tmp_path / "rotors.wav"
    audio.write(wav, np.zeros(1600), 16_000)
    assert demo.check(parser.parse_args(["all", "--input", f"helicopter={wav}"])) is None
