"""The pass-by demos: CNOSSOS-EU's car, the sound made to it, and the demo of the roadside
and the street run on a quick copy of the roadside."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from acoustic_spaces.demos import audio, pipeline
from acoustic_spaces.demos.car import (
    PROPULSION_A,
    ROLLING_A,
    SOUND_RMS,
    car_sound,
    firing_rate_hz,
    light_vehicle_power,
)
from acoustic_spaces.examples import example_path
from acoustic_spaces.scene import Scene


def test_cnossos_gives_a_cars_sound_power() -> None:
    """At 70 km/h and 20 °C the levels are Table F-1's coefficients A; at other speeds
    rolling noise goes with lg(v/70) and propulsion with (v - 70)/70; below 20 km/h as at
    20 km/h; colder air makes the rolling noise louder by 0.08 dB a degree."""
    reference = light_vehicle_power(70.0)
    assert reference.rolling_db == pytest.approx(ROLLING_A)
    assert reference.propulsion_db == pytest.approx(PROPULSION_A)
    town = light_vehicle_power(50.0)
    assert town.rolling_db[4] == pytest.approx(97.3 + 32.5 * math.log10(50 / 70))  # 1 kHz
    assert town.propulsion_db[0] == pytest.approx(94.5 - 1.3 * (50 - 70) / 70)  # 63 Hz
    assert town.sound_power_db == pytest.approx(99.0, abs=0.05)
    assert town.sound_power_a_db == pytest.approx(96.0, abs=0.05)
    slow = light_vehicle_power(10.0)
    assert slow.total_db == pytest.approx(light_vehicle_power(20.0).total_db)
    cold = light_vehicle_power(50.0, temperature_c=10.0)
    assert cold.rolling_db == pytest.approx(town.rolling_db + 0.8)
    assert cold.propulsion_db == pytest.approx(town.propulsion_db)
    with pytest.raises(ValueError, match="positive"):
        light_vehicle_power(0.0)


@pytest.mark.parametrize("example", ["roadside", "street"])
def test_the_examples_car_is_cnossos_at_50_km_h(example: str) -> None:
    """The car of the roadside and the street: a point source 0.05 m up, left out of the
    example's simulations, CNOSSOS-EU's levels at the speed of its walk, with Doppler."""
    scene = Scene.from_yaml(example_path(example))
    (car,) = [s for s in scene.sources if s.name == "car"]
    assert not car.in_use and not car.extended and car.position[2] == 0.05
    assert car.walk.doppler and car.walk.enabled and car.walk.keyframes is None
    power = light_vehicle_power(car.walk.speed_m_s * 3.6, scene.air.temperature_c)
    assert car.level_db == pytest.approx(power.total_db, abs=0.05)


def test_the_cars_sound_has_cnossos_spectrum_and_its_engines_firing() -> None:
    """The sound made to a car's power: its octave bands in the ratio of the sound powers
    (within 0.3 dB), at SOUND_RMS, with the engine's firing as its strongest low tone."""
    from acoustic_spaces.synthesis import split_bands

    power = light_vehicle_power(50.0)
    rate = 16_000
    x = car_sound(power, rate, 4.0, rpm=1800.0)
    assert len(x) == 4 * rate and np.sqrt(np.mean(x**2)) == pytest.approx(SOUND_RMS)
    bands = 10.0 * np.log10(np.mean(split_bands(x, rate) ** 2, axis=1))
    want = power.total_db
    assert bands - bands.max() == pytest.approx(want - want.max(), abs=0.3)
    spectrum = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    low = (f > 40) & (f < 100)
    assert f[low][np.argmax(spectrum[low])] == pytest.approx(firing_rate_hz(1800.0), abs=0.5)
    assert np.array_equal(car_sound(power, rate, 1.0, seed=3), car_sound(power, rate, 1.0, seed=3))
    with pytest.raises(ValueError, match="duration"):
        car_sound(power, rate, 0.0)


def test_the_pitch_falls_by_the_doppler_ratio() -> None:
    """Along a long straight road the direct sound's pitch falls by (c + v) / (c - v);
    seen side on at both ends, it does not change."""
    from types import SimpleNamespace

    from acoustic_spaces.demos.passby import pitch_change

    along = SimpleNamespace(positions=np.array([[-1e4, 0.0, 0.0], [0.0, 0.0, 0.0], [1e4, 0, 0]]))
    falls = pitch_change(along, (0.0, 5.0, 0.0), 20.0, 343.0)
    assert falls == pytest.approx(12 * math.log2(363.0 / 323.0), abs=1e-6)
    across = SimpleNamespace(positions=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert pitch_change(across, (-1.0, 1e6, 0.0), 20.0, 343.0) == pytest.approx(0.0, abs=1e-4)


def _roadside(tmp_path: Path) -> Path:
    """The roadside example, quick: a road 40 m long, two listeners, 16 kHz."""
    folder = tmp_path / "roadside"
    shutil.copytree(Path(example_path("roadside")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    (car,) = [s for s in data["sources"] if s["name"] == "car"]
    car["position"] = [20, 2, 0.05]
    car["walk"]["to"] = [[-20, 2, 0.05]]
    data["listeners"] = [lis for lis in data["listeners"] if lis["name"] in ("verge", "yard_25m")]
    data["simulation"].update({"sample_rate": 16_000, "ir_length_s": 0.2})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 500}
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_the_passby_demo_says_what_keeps_the_car_from_driving(tmp_path: Path) -> None:
    """No car, a car with no walk, one whose walk is off or whose keyframes go at more
    than one speed (unless --speed gives one), and a speed that is not positive are
    refused before anything runs; by default the car goes as fast as its walk does."""
    from acoustic_spaces.demos import load

    folder = _roadside(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    args = demo.defaults()
    assert args.speed is None  # as the scene's walk goes: 13.89 m/s
    assert demo.check(args) is None and demo.moving(demo.resolved(args)) == ["car"]
    assert demo.plays(demo.resolved(args)) == {
        "car": "a car at 50 km/h, made to CNOSSOS-EU's sound power"
    }
    assert demo.summary(demo.resolved(args)).startswith(
        "The car drives 40.0 m along its walk at 50.0 km/h (13.89 m/s, as its walk in the "
        "scene goes): the pass, and the car's sound, last 2.9 s."
    )
    args.speed = -3.0
    assert demo.check(args) == "--speed is a positive number, not -3.0"
    args.speed = None
    path = folder / "scene.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    (car,) = [s for s in data["sources"] if s["name"] == "car"]
    car["walk"]["enabled"] = False
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert "the car's walk is off in scene.yaml" in demo.check(args)
    assert demo.moving(demo.resolved(args)) == []
    # Keyframes at one steady speed drive as they are timed: 40 m in 3 s.
    car["walk"] = {"keyframes": [{"time_s": 3.0, "position": [-20, 2, 0.05]}], "doppler": True}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(args) is None
    assert "at 48.0 km/h (13.33 m/s, as its walk in the scene goes)" in demo.summary(
        demo.resolved(args)
    )
    # Keyframes at two speeds: CNOSSOS-EU has a car's power at one; --speed gives it.
    car["walk"]["keyframes"] = [
        {"time_s": 1.0, "position": [0, 2, 0.05]},
        {"time_s": 3.0, "position": [0, 2, 0.05]},  # it waits
        {"time_s": 5.0, "position": [-20, 2, 0.05]},
    ]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(args).startswith(
        "scene.yaml: the car's keyframes go at 0.0 to 72.0 km/h, but CNOSSOS-EU gives a car's "
        "sound power at one steady speed"
    )
    assert demo.summary(demo.resolved(args)) is None
    args.speed = 36.0
    assert demo.check(args) is None
    assert "40.0 m along its walk at 36.0 km/h (10.00 m/s, --speed)" in demo.summary(
        demo.resolved(args)
    )
    data["sources"] = [s for s in data["sources"] if s["name"] != "car"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(args) == "scene.yaml has no source 'car' to drive"


def test_the_passby_demo_drives_keyframes_as_timed_or_at_speed(tmp_path: Path) -> None:
    """The demo's scene keeps a car's steady keyframes as they are, with Doppler and
    CNOSSOS-EU's power at their speed; with --speed it drives through their places (each
    once) at that speed instead, and a scene written for one speed is written again for
    another."""
    from acoustic_spaces.demos import load
    from acoustic_spaces.demos.passby import driving, walk_speed

    folder = _roadside(tmp_path)
    path = folder / "scene.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    (car,) = [s for s in data["sources"] if s["name"] == "car"]
    car["walk"] = {
        "keyframes": [
            {"time_s": 1.5, "position": [0, 2, 0.05]},
            {"time_s": 3.0, "position": [-20, 2, 0.05], "orientation": {"yaw": 180}},
        ],
        "enabled": True,
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    demo = load(folder / "passby_demo.py").at(folder)
    assert demo.main(["scene"]) == 0
    written = Scene.from_yaml(folder / "scene_demo.yaml")
    (driven,) = [s for s in written.sources if s.name == "car"]
    assert driven.in_use and driven.walk.doppler and len(driven.walk.keyframes) == 2
    assert walk_speed(driven) == pytest.approx(40.0 / 3.0)
    power = light_vehicle_power(48.0, written.air.temperature_c)
    assert driven.level_db == pytest.approx(power.total_db, abs=0.05)
    args = demo.resolved(demo.defaults())
    assert demo.is_done(demo.step("scene"), args)
    args.speed = 72.0
    assert not demo.is_done(demo.step("scene"), args)
    assert demo.main(["scene", "--speed", "72"]) == 0
    (driven,) = [s for s in Scene.from_yaml(folder / "scene_demo.yaml").sources if s.name == "car"]
    assert driven.walk.keyframes is None and driven.walk.speed_m_s == pytest.approx(20.0)
    assert driven.walk.to == [(0.0, 2.0, 0.05), (-20.0, 2.0, 0.05)] and driven.walk.doppler
    assert demo.is_done(demo.step("scene"), args)
    # The same, from Python: a wait drops out of the places.
    source = Scene.from_yaml(path)
    (car,) = [s for s in source.sources if s.name == "car"]
    frames = [*car.walk.keyframes[:1], car.walk.keyframes[0].model_copy(update={"time_s": 2.0})]
    waiting = car.model_copy(update={"walk": car.walk.model_copy(update={"keyframes": frames})})
    assert driving(waiting, 36.0).walk.to == [(0.0, 2.0, 0.05)]


def test_the_passby_demo_drives_the_car_past_every_listener(tmp_path: Path, capsys) -> None:
    """Every step on the quick roadside: the car's sound, the scene with the car alone at
    --speed, the passes (the level along each, its maximum near the middle, the pitch
    falling, one gain for all), the tour and its stills; then each step is done."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load

    folder = _roadside(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    assert demo.main(["all", "--step", "4", "--speed", "72", "--no-mp4"]) == 0
    work = folder / "demo"
    info = pipeline.read_json(work / "car" / "car.json")
    assert info["speed_kmh"] == 72.0 and info["duration_s"] == pytest.approx(2.0)
    x, rate = audio.read(work / "car" / "car.wav")
    assert rate == 16_000 and len(x) == 2 * rate
    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == ["car"]
    (car,) = [s for s in scene.sources if s.name == "car"]
    assert car.walk.speed_m_s == pytest.approx(20.0) and car.level_db == info["level_db"]
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert set(facts["listeners"]) == {"verge", "yard_25m"} and len(facts["times"]) == 11
    verge = facts["listeners"]["verge"]
    assert verge["closest_m"] == pytest.approx(math.hypot(5.5, 1.45), abs=0.01)
    assert 0.8 < verge["max_s"] < 1.2 and verge["max_a_db"] == max(verge["levels_a_db"])
    assert verge["sel_a_db"] > verge["leq_a_db"]  # over 2 s: SEL = Leq + 3 dB
    assert 1.0 < verge["pitch_semitones"] < 12 * math.log2(363.0 / 323.0)
    seats = pipeline.read_json(work / "walk" / "seats.json")
    peaks = [
        np.abs(audio.read(work / "walk" / f"{k}.binaural.wav")[0]).max() for k in seats["seats"]
    ]
    assert max(peaks) == pytest.approx(10.0 ** (-1.0 / 20.0), rel=1e-3)
    assert sorted(p.name for p in (work / "stills").glob("*.png")) == [
        "00_verge.png",
        "01_yard_25m.png",
    ]
    assert (work / "thumbnail.png").exists() and (work / "tour.wav").exists()
    chapters = (work / "chapters.txt").read_text(encoding="utf-8").splitlines()
    assert chapters[0] == "0:00 verge" and chapters[1].endswith(" yard 25m")
    capsys.readouterr()
    assert demo.main(["all", "--step", "4", "--speed", "72", "--no-mp4"]) == 0
    out = capsys.readouterr().out
    for step in ("car", "scene", "walk"):
        assert f"{step}: done already" in out
    # Another speed makes the car, the scene and the passes again.
    args = demo.resolved(demo.parser().parse_args(["all", "--step", "4"]))
    assert not demo.is_done(demo.step("car"), args)
    assert not demo.is_done(demo.step("scene"), args)
