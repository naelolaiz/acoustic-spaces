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
from acoustic_spaces.demos.helicopter import HELICOPTER_VEHICLE, POWER_RATE, POWER_S, airspeed
from acoustic_spaces.directivity import read_directivity_csv
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
    crosses the stadium 150 m up at 63 m/s through still air: its sound power and its
    directivity (helicopter.csv) are those of its rotors' sound at that airspeed."""
    path = Path(example_path("stadium_concert"))
    scene = Scene.from_yaml(path)
    helicopter = next(s for s in scene.sources if s.name == "helicopter")
    assert (
        not helicopter.in_use and helicopter.trajectory.doppler and helicopter.position[2] == 150.0
    )
    assert (
        helicopter.trajectory.speed_m_s == 63.0 and HELICOPTER_VEHICLE.problem(helicopter) is None
    )
    assert airspeed(helicopter, scene, 63.0) == pytest.approx(63.0)
    args = argparse.Namespace(speed=None, broadband="pegg")
    trim, sent = HELICOPTER_VEHICLE.emission(
        HELICOPTER_VEHICLE.info(args, scene, 63.0 * 3.6), POWER_S
    )
    level = sent.sound_power_db(trim.density, trim.sound_speed)
    assert helicopter.level_db == pytest.approx(level, abs=0.051)
    table = read_directivity_csv(path.parent / "helicopter.csv")
    np.testing.assert_allclose(table.table, sent.directivity().table, rtol=2e-3)


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
    info = {"rate": rate, "airspeed_m_s": 40.0, "broadband": model} | air
    made = {n: HELICOPTER_VEHICLE.emission(info, n / rate) for n in (rate, 5077)}
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


def _concert_seats(folder: Path, stadium) -> Path:
    """Where the stadium demo's run at its defaults keeps its mixes."""
    return stadium.DEMO.at(folder).run_folder(stadium.DEMO.defaults()) / "seats"


def _concert(folder: Path, stadium, length_s: float, rate: int = 16_000) -> dict:
    """What the stadium demo's render leaves in the work folder, made up: the band's
    scene, a simulation's parameters, the feeds' table (the main arrays the loudest, at
    134 dB) and a mix of noise at each listener, with one gain, in the folder of its run
    at its defaults."""
    stadium.DEMO.at(folder).main(["band"])
    results = folder / "results_band" / "binaural"
    results.mkdir(parents=True)
    responses = [
        {"source": source, "listener": listener, "spl_a_db": level, "outputs": []}
        for listener in ("mix_position", "singer")
        for source, level in (("main_left", 99.0), ("subs", 95.0))
    ]
    pipeline.write_json(results / "parameters.json", {"responses": responses, "sample_rate": rate})
    song = stadium.SONGS["spicy_funk_cake"]
    sources = {
        "main_left": {"plays": "right", "level_db": 134.0, "gain_db": 0.0},
        "subs": {"plays": "mono", "level_db": 132.0, "gain_db": -2.0},
    }
    played = {k: v for k, v in song.items() if k in stadium.SONG_KEPT} | {"name": "spicy_funk_cake"}
    work = folder / "demo"
    pipeline.write_json(
        work / "feeds" / "spicy_funk_cake" / "feeds.json",
        {"rate": rate, "length_s": length_s, "song": played, "sources": sources},
    )
    held = _concert_seats(folder, stadium)
    assert held == work / "stadium_spicy_funk_cake_binaural" / "seats"
    noise = np.random.default_rng(1).normal(0.0, 0.05, (round((length_s + 0.5) * rate), 2))
    seats = {}
    for k, listener in enumerate(("mix_position", "singer")):
        x = noise * (1.0 - 0.5 * k)
        audio.write(held / f"{listener}.binaural.wav", x, rate)
        seats[listener] = {
            "listener": listener,
            "lag_s": 0.1 * (1 - k),
            "rms_db": audio.level_db(x, rate),
            "loudness_db": audio.loudness_db(x, rate),
            "length_s": len(x) / rate,
            "relative_db": -6.02 * k,
        }
    concert = {"rate": rate, "format": "binaural", "label": "binaural", "gain": 0.25}
    concert["seats"] = seats
    pipeline.write_json(held / pipeline.SEATS, concert)
    return concert


def test_the_helicopter_crosses_over_the_concert(tmp_path: Path) -> None:
    """The stadium's second demo: the flight over the stadium (its scene's helicopter,
    150 m up or at --height, the ground and the sky widened, the responses as much
    longer as the helicopter is farther), the
    helicopter's pass at each listener, and each listener's mix of the concert with the
    pass added over the centre spot halfway through the song, in the feeds' units: a
    gated RMS of 0.1 for the loudest loudspeaker's sound power, as the pressure 1 m from
    it; then the stadium demo's video of those mixes."""
    pytest.importorskip("PIL")
    from acoustic_spaces.air import density, speed_of_sound
    from acoustic_spaces.demos import demo_for, demos_of, load, sibling
    from acoustic_spaces.directivity import _grid
    from acoustic_spaces.emission import Mixture

    folder = _stadium(tmp_path)
    demo = load(folder / "helicopter_demo.py").at(folder)
    stadium = sibling(folder / "stadium_demo.py", "stadium_demo")
    assert [d.name for d in demos_of("stadium_concert")] == ["stadium", "crossing"]
    assert demo_for("stadium_concert").name == "stadium" and demo.builds_on == "stadium"
    args = demo.defaults()
    assert args.height is None and args.at is None  # as in the scene, halfway through
    assert demo.check(argparse.Namespace(**vars(args) | {"height": 25.0})).endswith(
        "but the stadium's roof is 30 m up"
    )
    assert demo.check(args) is None  # the flight's scene is written by the first step
    stadium_scene = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    (own,) = [s for s in stadium_scene["sources"] if s["name"] == "helicopter"]
    low = argparse.Namespace(**vars(args) | {"height": None})
    own["position"][2] = 20.0
    (folder / "scene.yaml").write_text(yaml.safe_dump(stadium_scene, sort_keys=False), "utf-8")
    assert demo.check(low) == (
        "the helicopter in scene.yaml flies 20 m up, but the stadium's roof is 30 m up"
    )
    own["position"][2] = 150.0
    (folder / "scene.yaml").write_text(yaml.safe_dump(stadium_scene, sort_keys=False), "utf-8")

    # Every step of the run with a step of 8 m: its scenes and its passes in its folder.
    args = argparse.Namespace(**vars(args) | {"step_m": 8.0})
    run = demo.run_folder(args)
    assert run == folder / "demo" / "crossing_step-8_binaural"
    assert demo.main(["flight", "--step", "8"]) == 0
    flown = yaml.safe_load((run / "scene_flight.yaml").read_text(encoding="utf-8"))
    ground = flown["geometry"]["ground"]
    assert ground["extent"] == [[-650.0, -82], [650.0, 82]] and ground["sky"] == 190.0
    (helicopter,) = [s for s in flown["sources"] if s["name"] == "helicopter"]
    # The scene's, left out of the concert's simulations, its files found from the run's
    # folder.
    assert helicopter == pipeline.files_from(own, run / "scene_flight.yaml", folder)
    assert helicopter["directivity"]["file"] == str(folder / "helicopter.csv")
    assert helicopter["enabled"] is False and helicopter["position"] == [-600, 0, 150]
    assert helicopter["trajectory"] == {"to": [[600, 0, 150]], "speed_m_s": 63, "doppler": True}
    lower = sibling(folder / "helicopter_demo.py", "helicopter_demo").flight_of(own, 40.0)
    assert lower["position"] == [-600.0, 0.0, 40.0] and lower["trajectory"]["to"] == [
        [600.0, 0.0, 40.0]
    ]
    # The stadium's length: outdoors the simulation counts it from the latest direct sound.
    assert flown["simulation"] == stadium_scene["simulation"]
    resolved = demo.resolved(args)
    assert demo.step("flight").done(resolved)
    assert not demo.step("flight").done(argparse.Namespace(**vars(resolved) | {"height": 400.0}))
    # Quick: 40 m of the flight over the centre spot, responses 1.5 s long.
    helicopter["position"] = [-20.0, 0.0, 150.0]
    helicopter["trajectory"]["to"] = [[20.0, 0.0, 150.0]]
    flown["simulation"]["ir_length_s"] = 1.5
    (run / "scene_flight.yaml").write_text(yaml.safe_dump(flown, sort_keys=False), "utf-8")
    for step in ("helicopter", "scene", "walk"):
        assert demo.main([step, "--step", "8"]) == 0
    scene = Scene.from_yaml(run / "scene_overhead.yaml")
    assert [s.name for s in scene.sources if s.in_use] == ["helicopter"]
    assert (run / "scene_overhead_helicopter.csv").exists()

    work = folder / "demo"
    held = _concert_seats(folder, stadium)
    with pytest.raises(SystemExit, match="stadium_demo.py render"):
        demo.main(["mix", "--step", "8"])
    concert = _concert(folder, stadium, 6.0)
    assert demo.main(["mix", "--step", "8"]) == 0
    facts = pipeline.read_json(run / "walk" / "walk.json")
    middle = 20.0 / 63.0  # over the centre spot between its places 4 m either side
    assert [round(p[0]) for p in facts["positions"]] == [-20, -12, -4, 4, 12, 20]
    passes = pipeline.read_json(run / "walk" / pipeline.SEATS)
    crossed = pipeline.read_json(run / "seats" / pipeline.SEATS)
    table = pipeline.read_json(run / "feeds" / "feeds.json")
    crossing = table["helicopter"]
    assert crossing["at_s"] == 3.0 and crossing["height_m"] == 150.0  # halfway through
    assert crossing["start_s"] == pytest.approx(3.0 - middle, abs=1e-4)
    start = round(crossing["start_s"] * 16_000)
    heard = max(seat["length_s"] for seat in passes["seats"].values())
    assert table["length_s"] == pytest.approx(max(6.0, crossing["start_s"] + heard))
    # The scale, from the pressure 1 m from a source of the main arrays' sound power,
    # found here through the emissions' own sound power: of a tone of 1 Pa (RMS) sent
    # every way alike.
    elevation, azimuth = _grid(10.0)
    tone = np.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * np.arange(16_000) / 16_000)
    one_pa = Mixture(tone[None, :], np.ones((len(elevation), len(azimuth), 1)), 16_000)
    level = one_pa.sound_power_db(density(20.0, 101.325), speed_of_sound(20.0))
    pressure = 10.0 ** ((134.0 - 10.0 * math.log10(np.sum(10.0 ** (level / 10.0)))) / 20.0)
    for listener in ("mix_position", "singer"):
        x = audio.read(held / f"{listener}.binaural.wav")[0] / concert["gain"]
        y = audio.read(run / "walk" / f"{listener}.binaural.wav")[0] / passes["gain"]
        expected = np.zeros((max(len(x), start + len(y)), 2))
        expected[: len(x)] += x
        expected[start : start + len(y)] += y * 0.1 / pressure
        assert crossed["seats"][listener]["file"] == f"{listener}.binaural.wav"
        mixed = audio.read(run / "seats" / f"{listener}.binaural.wav")[0]
        np.testing.assert_allclose(mixed, expected * crossed["gain"], atol=2e-6)
        assert crossed["seats"][listener]["lag_s"] == concert["seats"][listener]["lag_s"]
    assert demo.step("mix").done(demo.resolved(args))
    assert not demo.step("mix").done(demo.resolved(argparse.Namespace(**vars(args) | {"at": 5.0})))
    # Through other HRTFs than these mixes (and the concert's): made again, not done.
    kemar = argparse.Namespace(**vars(args) | {"hrtf": "kemar.sofa"})
    assert crossed["label"] == "binaural" and not demo.step("mix").done(demo.resolved(kemar))
    # Another --at plays the same passes: the walk of the run up to the walk, not
    # walked again.
    later = demo.resolved(argparse.Namespace(**vars(args) | {"at": 5.0}))
    assert later.run_folder != run and demo.upstream(later, "walk").run_folder == run
    assert demo.step("walk").done(later)
    with pytest.raises(SystemExit, match="give --at 1 or later"):
        demo.main(["mix", "--step", "8", "--at", "0.2"])

    assert demo.main(["video", "--step", "8", "--no-mp4"]) == 0
    tour, rate = audio.read(work / "crossing_step-8_binaural.wav")
    assert rate == 16_000 and len(tour) / rate == pytest.approx(table["length_s"], abs=0.01)
    assert sorted(p.name for p in (run / "stills").glob("*.png")) == ["00_mix_position.png"]
    # --listeners: the mixes with the helicopter there are made already, and the video
    # tours that listener alone; a listener the concert has no mix at is refused with
    # the stadium demo's command for it.
    singer = argparse.Namespace(**vars(args) | {"listeners": "singer"})
    assert not demo.step("mix").done(demo.resolved(singer))  # a run of its own
    assert demo.main(["mix", "--step", "8", "--listeners", "singer"]) == 0
    assert demo.step("mix").done(demo.resolved(singer))
    assert demo.main(["video", "--step", "8", "--no-mp4", "--listeners", "singer"]) == 0
    assert (demo.run_folder(singer) / "stills" / "00_singer.png").exists()
    concert["seats"].pop("singer")
    pipeline.write_json(held / pipeline.SEATS, concert)
    assert not demo.step("mix").done(demo.resolved(singer))
    with pytest.raises(SystemExit, match="the concert has no mix at singer; run: python stad"):
        demo.main(["mix", "--step", "8", "--listeners", "singer"])


def test_the_crossing_flies_over_the_concerts_sources_picked(tmp_path: Path) -> None:
    """The crossing plays the stadium demo's run of the concert's sources ``--sources``
    picks, every one by default (the runs named as before), and the helicopter always:
    its ``--sources`` names the concert's sources alone, the stadium run it crosses is
    that pick's (and its listeners'), and a pick of the helicopter is refused."""
    from acoustic_spaces.demos import load, run_folder, sibling

    folder = _stadium(tmp_path)
    demo = load(folder / "helicopter_demo.py").at(folder)
    crossing = sibling(folder / "helicopter_demo.py", "helicopter_demo")

    def given(**values) -> argparse.Namespace:
        return argparse.Namespace(**vars(demo.defaults()) | values)

    args = demo.resolved(given())
    concert = [s.name for s in Scene.from_yaml(folder / "scene.yaml").sources]
    concert.remove("helicopter")
    assert [s.name for s in demo.sources(args)] == concert
    assert args.playing == tuple(concert) and run_folder(args).name == "crossing_binaural"
    band = crossing.band_args(args)
    assert band.sources is None and run_folder(band).name == "stadium_spicy_funk_cake_binaural"
    pa = demo.resolved(given(sources="PA", listeners="singer,mix_position"))
    assert run_folder(pa).name == "crossing_PA_singer+mix_position_binaural"
    band = crossing.band_args(pa)
    assert band.sources == "PA" and band.playing == pa.playing
    assert run_folder(band).name == "stadium_spicy_funk_cake_PA_singer+mix_position_binaural"
    assert demo.check(given(sources="helicopter")) == (
        "--sources helicopter: left out of the concert's simulation, so it plays nothing "
        "here; helicopter_demo.py flies it over the concert"
    )
    option = demo.option("sources")
    assert option.steps == ("mix", "video") and "the helicopter always plays" in option.help
    # Its text says the stadium's run is to be rendered with these --sources too.
    assert "the --listeners (or at every listener) and the --sources given here" in (
        " ".join(crossing.__doc__.split())
    )
    # The helicopter's walk is the same whatever of the concert plays: a pick's run
    # plays the walk of the run of every source.
    picked = demo.resolved(given(sources="PA", sources_play="in-turn"))
    assert run_folder(demo.upstream(demo.resolved(given(sources="PA")), "walk")).name == (
        "crossing_binaural"
    )
    assert all(
        run_folder(demo.upstream(demo.resolved(part), "walk")).name == "crossing_binaural"
        for part in demo.parts(picked)
    )


def test_the_crossing_takes_a_file_for_the_helicopter(tmp_path: Path) -> None:
    """``--input helicopter=WAV`` plays a file of yours in place of the helicopter's
    rotors, whatever of the concert plays."""
    from acoustic_spaces.demos import load

    folder = _stadium(tmp_path)
    demo = load(folder / "helicopter_demo.py").at(folder)
    wav = tmp_path / "rotors.wav"
    audio.write(wav, np.zeros(1600), 16_000)
    for picked in (None, "PA"):
        args = argparse.Namespace(
            **vars(demo.defaults()) | {"inputs": {"helicopter": wav}, "sources": picked}
        )
        assert demo.check(args) is None
