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
