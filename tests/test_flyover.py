"""The helicopter flyover: the example's helicopter as its rotors' model has it, and the
demo run on a quick copy of the example."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from acoustic_spaces.demos import audio, pipeline
from acoustic_spaces.demos.helicopter import HELICOPTER_VEHICLE, POWER_S, airspeed
from acoustic_spaces.directivity import read_directivity_csv
from acoustic_spaces.examples import example_path
from acoustic_spaces.scene import Scene


def test_the_examples_helicopter_is_its_rotors_model() -> None:
    """The helicopter of the helicopter_flyover example flies level 150 m up at 63 m/s,
    against the wind: its sound power and its directivity (helicopter.csv) are those of
    its rotors' sound at its airspeed, its speed over the ground and the wind's at its
    height."""
    path = Path(example_path("helicopter_flyover"))
    scene = Scene.from_yaml(path)
    (helicopter,) = scene.sources
    assert helicopter.in_use and helicopter.walk.doppler and helicopter.position[2] == 150.0
    assert helicopter.walk.speed_m_s == 63.0 and HELICOPTER_VEHICLE.problem(helicopter) is None
    wind = float(scene.atmosphere.wind_speed(np.array([150.0]))[0])
    assert airspeed(helicopter, scene, 63.0) == pytest.approx(63.0 + wind)
    info = HELICOPTER_VEHICLE.info(argparse.Namespace(speed=None), scene, 63.0 * 3.6)
    assert info["airspeed_m_s"] == pytest.approx(63.0 + wind, abs=1e-4)
    trim, sent = HELICOPTER_VEHICLE.emission(info, POWER_S)
    level = sent.sound_power_db(trim.density, trim.sound_speed)
    assert helicopter.level_db == pytest.approx(level, abs=0.051)
    table = read_directivity_csv(path.parent / "helicopter.csv")
    np.testing.assert_allclose(table.table, sent.directivity().table, rtol=2e-3)


def _flyover(tmp_path: Path) -> Path:
    """The helicopter_flyover example, quick: 40 m of its flight right over the listener
    under its track, 16 kHz."""
    folder = tmp_path / "helicopter_flyover"
    shutil.copytree(Path(example_path("helicopter_flyover")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    (helicopter,) = data["sources"]
    helicopter["position"] = [-20, 0, 150]
    helicopter["walk"]["to"] = [[20, 0, 150]]
    data["listeners"] = [lis for lis in data["listeners"] if lis["name"] == "under_track"]
    data["simulation"].update({"sample_rate": 16_000, "ir_length_s": 0.7})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 500}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_the_flyover_demo_flies_the_helicopter_over(tmp_path: Path) -> None:
    """A walk that climbs is refused (the rotors are trimmed for level flight). Every
    step on the quick flyover: its rotors' sound straight down, the scene with the
    helicopter's sound power and directivity at its airspeed, facing the way it flies,
    the pass (the pitch falling) and the tour."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load

    folder = _flyover(tmp_path)
    demo = load(folder / "flyover_demo.py").at(folder)
    path = folder / "scene.yaml"
    level = path.read_text(encoding="utf-8")
    data = yaml.safe_load(level)
    data["sources"][0]["walk"]["to"] = [[20, 0, 160]]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(demo.defaults()).endswith(
        "the helicopter flies level, its rotors trimmed for it, but its walk goes from "
        "150.0 to 160.0 m up: give its points one height"
    )
    path.write_text(level, encoding="utf-8")
    assert demo.main(["all", "--step", "8", "--no-mp4"]) == 0
    work = folder / "demo"
    info = pipeline.read_json(work / "helicopter" / "helicopter.json")
    assert info["helicopter"] == "AS350 B3" and info["speed_kmh"] == pytest.approx(226.8)
    assert info["duration_s"] == pytest.approx(40.0 / 63.0, abs=1e-6)
    x, rate = audio.read(work / "helicopter" / "below.wav")
    assert rate == 16_000 and len(x) == round(info["duration_s"] * rate)
    assert np.abs(x).max() == pytest.approx(0.5, rel=1e-3)
    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    (helicopter,) = scene.sources
    assert helicopter.level_db == pytest.approx(info["level_db"])
    assert helicopter.walk.face_forward and helicopter.walk.doppler
    # The example's at 48 kHz: at 16 kHz the 8 kHz band is half above the sound's top.
    table = read_directivity_csv(folder / "scene_demo_helicopter.csv").table[..., :7]
    full = read_directivity_csv(folder / "helicopter.csv").table[..., :7]
    np.testing.assert_allclose(table, full, rtol=2e-3)
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert facts["source"] == "helicopter" and facts["about"].startswith("An AS350 B3 at 226.8")
    under = facts["listeners"]["under_track"]
    assert under["closest_m"] == pytest.approx(math.hypot(4.0, 148.8), abs=0.01)  # x = ±4 m
    c = facts["speed_of_sound"]
    towards = 63.0 * 20.0 / math.hypot(20.0, 148.8)
    falls = 12.0 * math.log2((c + towards) / (c - towards))
    assert under["pitch_semitones"] == pytest.approx(falls, abs=1e-3)
    assert (work / "tour.wav").exists() and (work / "stills" / "00_under_track.png").exists()


def _stadium(tmp_path: Path) -> Path:
    """The stadium_concert example, quick: two listeners, 16 kHz, responses 0.3 s long
    after the sound arrives, few rays."""
    folder = tmp_path / "stadium_concert"
    shutil.copytree(Path(example_path("stadium_concert")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["listeners"] = [
        lis for lis in data["listeners"] if lis["name"] in ("mix_position", "singer")
    ]
    data["simulation"].update({"sample_rate": 16_000, "ir_length_s": 0.3})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 300}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def _concert(folder: Path, stadium, length_s: float, rate: int = 16_000) -> dict:
    """What the stadium demo's render leaves in the work folder, made up: the band's
    scene, a simulation's parameters, the feeds' table (the main arrays the loudest, at
    134 dB) and a mix of noise at each listener, with one gain."""
    stadium.DEMO.at(folder).main(["band"])
    results = folder / "results_band"
    results.mkdir()
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
        work / "feeds" / "feeds.json",
        {"rate": rate, "length_s": length_s, "song": played, "sources": sources},
    )
    noise = np.random.default_rng(1).normal(0.0, 0.05, (round((length_s + 0.5) * rate), 2))
    seats = {}
    for k, listener in enumerate(("mix_position", "singer")):
        x = noise * (1.0 - 0.5 * k)
        audio.write(work / "seats" / f"{listener}.binaural.wav", x, rate)
        seats[listener] = {
            "listener": listener,
            "lag_s": 0.1 * (1 - k),
            "rms_db": audio.level_db(x, rate),
            "loudness_db": audio.loudness_db(x, rate),
            "length_s": len(x) / rate,
            "relative_db": -6.02 * k,
        }
    concert = {"rate": rate, "format": "binaural", "gain": 0.25, "seats": seats}
    pipeline.write_json(work / "seats" / pipeline.SEATS, concert)
    return concert


def test_the_helicopter_crosses_over_the_concert(tmp_path: Path) -> None:
    """The stadium's second demo: the flight over the stadium (300 m above its roof, the
    ground and the sky widened, the responses as much longer as the helicopter is
    farther), the helicopter's pass at each listener, and each listener's mix of the
    concert with the pass added over the centre spot as the song ends, in the feeds'
    units: a gated RMS of 0.1 for the loudest loudspeaker's sound power, as the pressure 1
    m from it; then the stadium demo's video of those mixes."""
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
    assert args.height == 330.0 and args.at is None
    assert demo.check(argparse.Namespace(**vars(args) | {"height": 25.0})).endswith(
        "but the stadium's roof is 30 m up"
    )
    assert demo.check(args) is None  # the flight's scene is written by the first step

    assert demo.main(["flight"]) == 0
    flown = yaml.safe_load((folder / "scene_flight.yaml").read_text(encoding="utf-8"))
    ground = flown["geometry"]["ground"]
    assert ground["extent"] == [[-650.0, -82], [650.0, 82]] and ground["sky"] == 370.0
    helicopter = flown["sources"][-1]
    assert helicopter == {
        "name": "helicopter",
        "group": "Helicopter",
        "enabled": False,
        "position": [-600.0, 0.0, 330.0],
        "walk": {"to": [[600.0, 0.0, 330.0]], "speed_m_s": 63.0, "doppler": True},
    }
    farthest = max(math.dist((600.0, 0.0, 330.0), lis["position"]) for lis in flown["listeners"])
    later = math.ceil((0.3 + farthest / speed_of_sound(20.0)) * 10.0) / 10.0
    assert flown["simulation"]["ir_length_s"] == later
    resolved = demo.resolved(args)
    assert demo.step("flight").done(resolved)
    assert not demo.step("flight").done(argparse.Namespace(**vars(resolved) | {"height": 400.0}))
    # Quick: 40 m of the flight over the centre spot, responses 1.5 s long.
    helicopter["position"] = [-20.0, 0.0, 330.0]
    helicopter["walk"]["to"] = [[20.0, 0.0, 330.0]]
    flown["simulation"]["ir_length_s"] = 1.5
    (folder / "scene_flight.yaml").write_text(yaml.safe_dump(flown, sort_keys=False), "utf-8")
    for step in ("helicopter", "scene", "walk"):
        assert demo.main([step, "--step", "8"]) == 0
    scene = Scene.from_yaml(folder / "scene_overhead.yaml")
    assert [s.name for s in scene.sources if s.in_use] == ["helicopter"]
    assert (folder / "scene_overhead_helicopter.csv").exists()

    work = folder / "demo"
    with pytest.raises(SystemExit, match="stadium_demo.py render"):
        demo.main(["mix", "--step", "8"])
    concert = _concert(folder, stadium, 6.0)
    assert demo.main(["mix", "--step", "8"]) == 0
    facts = pipeline.read_json(work / "walk" / "walk.json")
    middle = 20.0 / 63.0  # over the centre spot between its places 4 m either side
    assert [round(p[0]) for p in facts["positions"]] == [-20, -12, -4, 4, 12, 20]
    passes = pipeline.read_json(work / "walk" / pipeline.SEATS)
    crossed = pipeline.read_json(work / "crossing" / "seats" / pipeline.SEATS)
    table = pipeline.read_json(work / "crossing" / "feeds" / "feeds.json")
    crossing = table["helicopter"]
    assert crossing["at_s"] == 6.0 and crossing["height_m"] == 330.0
    assert crossing["start_s"] == pytest.approx(6.0 - middle, abs=1e-4)
    start = round(crossing["start_s"] * 16_000)
    length = crossing["start_s"] + passes["seats"]["singer"]["length_s"]
    assert table["length_s"] == pytest.approx(length)
    # The scale, from the pressure 1 m from a source of the main arrays' sound power,
    # found here through the emissions' own sound power: of a tone of 1 Pa (RMS) sent
    # every way alike.
    elevation, azimuth = _grid(10.0)
    tone = np.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * np.arange(16_000) / 16_000)
    one_pa = Mixture(tone[None, :], np.ones((len(elevation), len(azimuth), 1)), 16_000)
    level = one_pa.sound_power_db(density(20.0, 101.325), speed_of_sound(20.0))
    pressure = 10.0 ** ((134.0 - 10.0 * math.log10(np.sum(10.0 ** (level / 10.0)))) / 20.0)
    for listener in ("mix_position", "singer"):
        x = audio.read(work / "seats" / f"{listener}.binaural.wav")[0] / concert["gain"]
        y = audio.read(work / "walk" / f"{listener}.binaural.wav")[0] / passes["gain"]
        expected = np.zeros((max(len(x), start + len(y)), 2))
        expected[: len(x)] += x
        expected[start : start + len(y)] += y * 0.1 / pressure
        mixed = audio.read(work / "crossing" / "seats" / f"{listener}.binaural.wav")[0]
        np.testing.assert_allclose(mixed, expected * crossed["gain"], atol=2e-6)
        assert crossed["seats"][listener]["lag_s"] == concert["seats"][listener]["lag_s"]
    assert demo.step("mix").done(demo.resolved(args))
    assert not demo.step("mix").done(demo.resolved(argparse.Namespace(**vars(args) | {"at": 5.0})))
    with pytest.raises(SystemExit, match="give --at 1 or later"):
        demo.main(["mix", "--step", "8", "--at", "0.2"])

    assert demo.main(["video", "--step", "8", "--no-mp4"]) == 0
    tour, rate = audio.read(work / "crossing" / "tour.wav")
    assert rate == 16_000 and len(tour) / rate == pytest.approx(table["length_s"], abs=0.01)
    assert sorted(p.name for p in (work / "crossing" / "stills").glob("*.png")) == [
        "00_mix_position.png"
    ]
