"""The pass-by demos: CNOSSOS-EU's road vehicles, the sound made to them, the police car's
siren, and the demo of the roadside and the street run on a quick copy of the roadside."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from acoustic_spaces.demos import audio, pipeline, siren
from acoustic_spaces.demos.passby import BUS, MOTORCYCLE, THE_CAR, TRUCK
from acoustic_spaces.demos.road import (
    CATEGORIES,
    SOUND_RMS,
    firing_rate_hz,
    vehicle_power,
    vehicle_sound,
)
from acoustic_spaces.examples import example_path
from acoustic_spaces.scene import Scene


def test_cnossos_gives_each_categorys_sound_power() -> None:
    """At 70 km/h and 20 °C the levels are Table F-1's coefficients A; at other speeds
    rolling noise goes with lg(v/70) and propulsion with (v - 70)/70; below 20 km/h as at
    20 km/h; colder air makes the rolling noise louder by K a degree (0.08 dB for cars,
    0.04 for heavy vehicles); two-wheelers have propulsion noise alone."""
    for key, kind in CATEGORIES.items():
        reference = vehicle_power(key, 70.0)
        assert reference.propulsion_db == pytest.approx(kind.propulsion_a)
        if kind.rolling_a is None:
            assert reference.rolling_db is None
            assert reference.total_db == pytest.approx(kind.propulsion_a)
        else:
            assert reference.rolling_db == pytest.approx(kind.rolling_a)
            cold = vehicle_power(key, 70.0, temperature_c=10.0)
            assert cold.rolling_db == pytest.approx(
                reference.rolling_db + 10.0 * kind.temperature_db_per_c
            )
        assert vehicle_power(key, 10.0).total_db == pytest.approx(vehicle_power(key, 20.0).total_db)
    assert [CATEGORIES[k].temperature_db_per_c for k in ("1", "2", "3")] == [0.08, 0.04, 0.04]
    town = vehicle_power("1", 50.0)
    assert town.rolling_db[4] == pytest.approx(100.1 + 32.5 * math.log10(50 / 70))  # 1 kHz
    assert town.propulsion_db[0] == pytest.approx(97.9 - 1.3 * (50 - 70) / 70)  # 63 Hz
    assert town.sound_power_db == pytest.approx(102.0, abs=0.05)
    assert town.sound_power_a_db == pytest.approx(98.4, abs=0.05)
    with pytest.raises(ValueError, match="positive"):
        vehicle_power("1", 0.0)
    with pytest.raises(ValueError, match="no vehicle category '5'"):
        vehicle_power("5", 50.0)


def test_the_table_is_the_2021_one_not_2015s() -> None:
    """Table F-1 as amended in 2021: a car's rolling noise at 1 kHz and propulsion noise at
    63 Hz higher than Directive (EU) 2015/996 had them (97.3 and 94.5 dB, the car's
    before), its B coefficients as they were, and a heavy vehicle's rolling noise at 500
    Hz as the amended table gives it."""
    car, heavy = CATEGORIES["1"], CATEGORIES["3"]
    assert car.rolling_a[4] == 100.1 and car.propulsion_a[0] == 97.9
    assert heavy.rolling_a[3] == 104.9
    assert car.rolling_b == (30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0)


@pytest.mark.parametrize(
    ("example", "vehicle"),
    [
        ("roadside", THE_CAR),
        ("roadside", TRUCK),
        ("street", THE_CAR),
        ("street", BUS),
        ("street", MOTORCYCLE),
        ("street", siren.POLICE_CAR_VEHICLE.car),
    ],
)
def test_the_examples_road_vehicles_are_cnossos_at_their_speed(example: str, vehicle) -> None:
    """The road vehicles of the roadside and the street, the police car's car too: point
    sources 0.05 m up, left out of the example's simulations, CNOSSOS-EU's levels for
    their category at the speed of their walk, with Doppler."""
    scene = Scene.from_yaml(example_path(example))
    (source,) = [s for s in scene.sources if s.name == vehicle.name]
    assert not source.in_use and not source.extended and source.position[2] == 0.05
    assert source.walk.doppler and source.walk.enabled and source.walk.keyframes is None
    power = vehicle_power(vehicle.category, source.walk.speed_m_s * 3.6, scene.air.temperature_c)
    assert source.level_db == pytest.approx(power.total_db, abs=0.05)


def test_the_streets_police_car_has_the_measured_siren(tmp_path: Path) -> None:
    """The street's police car: left out, its siren 1.5 m up along the car's walk, facing
    the way it drives, with the wail's sound power and siren.csv, the siren's directivity
    as the module writes it."""
    from acoustic_spaces.demos.passby import car_problem
    from acoustic_spaces.directivity import write_directivity_csv

    path = Path(example_path("street"))
    scene = Scene.from_yaml(path)
    (car,) = [s for s in scene.sources if s.name == "police_car_siren"]
    assert not car.in_use and car.position[2] == 1.5
    assert car.walk.doppler and car.walk.face_forward
    assert car_problem(scene, path.name, siren.POLICE_CAR_VEHICLE) is None
    # A siren whose walk goes elsewhere does not go along with its car.
    walk = car.walk.model_copy(update={"to": [(-50.0, 2.0, 1.5)]})
    moved = scene.model_copy(
        update={
            "sources": [
                s.model_copy(update={"walk": walk}) if s.name == car.name else s
                for s in scene.sources
            ]
        }
    )
    problem = car_problem(moved, path.name, siren.POLICE_CAR_VEHICLE)
    assert problem is not None and "must go where the police_car's does" in problem
    rate = scene.simulation.sample_rate
    assert car.level_db == pytest.approx(siren.siren_power("wail", rate), abs=0.05)
    assert car.directivity.file == path.parent / "siren.csv"
    table = write_directivity_csv(
        tmp_path / "siren.csv",
        siren.directivity(),
        siren.directivity_comment("acoustic_spaces.demos.siren"),
    )
    assert table.read_text(encoding="utf-8") == (path.parent / "siren.csv").read_text(
        encoding="utf-8"
    )


def test_the_vehicles_sound_has_cnossos_spectrum_and_its_engines_firing() -> None:
    """The sound made to a vehicle's power: its octave bands in the ratio of the sound
    powers (within 0.3 dB), at SOUND_RMS, with the engine's firing as its strongest low
    tone; a car's, a truck's six cylinders, and a motorcycle's with no tyres."""
    from acoustic_spaces.synthesis import split_bands

    rate = 16_000
    for category, rpm, cylinders, low in (
        ("1", 1800.0, 4, (40, 100)),
        ("3", 1200.0, 6, (40, 100)),
        ("4b", 4000.0, 2, (40, 100)),
    ):
        power = vehicle_power(category, 50.0)
        x = vehicle_sound(power, rate, 4.0, rpm=rpm, cylinders=cylinders)
        assert len(x) == 4 * rate and np.sqrt(np.mean(x**2)) == pytest.approx(SOUND_RMS)
        bands = 10.0 * np.log10(np.mean(split_bands(x, rate) ** 2, axis=1))
        want = power.total_db
        assert bands - bands.max() == pytest.approx(want - want.max(), abs=0.3)
        spectrum = np.abs(np.fft.rfft(x))
        f = np.fft.rfftfreq(len(x), 1.0 / rate)
        near = (f > low[0]) & (f < max(low[1], 1.5 * firing_rate_hz(rpm, cylinders)))
        assert f[near][np.argmax(spectrum[near])] == pytest.approx(
            firing_rate_hz(rpm, cylinders), abs=0.5
        )
    power = vehicle_power("1", 50.0)
    assert np.array_equal(
        vehicle_sound(power, rate, 1.0, seed=3), vehicle_sound(power, rate, 1.0, seed=3)
    )
    with pytest.raises(ValueError, match="duration"):
        vehicle_sound(power, rate, 0.0)


def test_the_siren_sweeps_and_alternates_as_the_rules_say() -> None:
    """The wail and the yelp sweep 500 to 1500 Hz up and down within California's rates
    (10 to 30 and 150 to 250 a minute); the two tones are a fourth apart within the
    Swiss rule's 360 to 630 Hz and their pair takes 2.5 to 3.5 s; each is a square wave,
    its odd harmonics 1/k of the fundamental."""
    assert 10 <= siren.WAIL_PER_MINUTE <= 30 and 150 <= siren.YELP_PER_MINUTE <= 250
    period = 60.0 / siren.WAIL_PER_MINUTE
    f = siren.fundamental_hz("wail", np.array([0.0, period / 4, period / 2, period]))
    assert f == pytest.approx([500.0, 1000.0, 1500.0, 500.0])
    low, high = siren.TWO_TONE_HZ
    assert 360 <= low < high <= 630 and high / low == pytest.approx(4.0 / 3.0)
    assert 2.5 <= siren.TWO_TONE_PERIOD_S <= 3.5
    t = np.array([0.1, siren.TWO_TONE_PERIOD_S * 0.6])
    assert siren.fundamental_hz("two_tone", t) == pytest.approx([low, high])
    rate = 16_000
    x = siren.siren_sound("two_tone", rate, 1.0)  # the lower tone alone
    assert np.sqrt(np.mean(x**2)) == pytest.approx(siren.SOUND_RMS)
    spectrum = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / rate)

    def at(hz: float) -> float:
        return float(spectrum[np.argmin(np.abs(f - hz))])

    assert at(3 * low) / at(low) == pytest.approx(1 / 3, rel=0.02)
    assert at(5 * low) / at(low) == pytest.approx(1 / 5, rel=0.02)
    assert at(2 * low) < 1e-3 * at(low)  # no even harmonics
    with pytest.raises(ValueError, match="pattern"):
        siren.fundamental_hz("whoop", t)


def test_the_sirens_level_and_directivity_are_the_measured_ones() -> None:
    """The directivity goes through NBS's 0, -6 and -11 dB at 0, 50 and 90 degrees and
    stays at -11 behind; the yelp's sound power puts 118 dB(A) 3 m away on its axis in
    free field; every pattern has the same sound power in all; the wail's and the yelp's
    loudest octave band is 1 kHz, as California's rule asks (1 or 2 kHz)."""
    from acoustic_spaces.directivity import _from_angles
    from acoustic_spaces.parameters import a_weighted

    assert siren.off_axis_db([0, 25, 50, 90, 135, 180]) == pytest.approx(
        [0.0, -3.0, -6.0, -11.0, -11.0, -11.0]
    )
    balloon = siren.directivity()
    ways = _from_angles(np.array([0.0, 50.0, 90.0, 180.0, 0.0]), np.array([0, 0, 0, 0, 50.0]))
    energy = balloon.energy(ways)[:, 0]
    assert 10 * np.log10(energy / energy[0]) == pytest.approx(
        [0.0, -6.0, -11.0, -11.0, -6.0], abs=0.05
    )
    di = siren.directivity_index_db()
    assert di == pytest.approx(10 * np.log10(energy[0]), abs=0.05)
    rate = 48_000
    yelp = siren.siren_power("yelp", rate)
    on_axis = a_weighted(yelp) + di - 20 * math.log10(3.0) - 10 * math.log10(4 * math.pi)
    assert on_axis == pytest.approx(118.0, abs=1e-9)
    totals = [
        10 * np.log10(np.sum(10 ** (siren.siren_power(p, rate) / 10))) for p in siren.PATTERNS
    ]
    assert totals == pytest.approx([totals[0]] * 3, abs=1e-9)
    for pattern in ("wail", "yelp"):
        assert int(np.argmax(siren.siren_power(pattern, rate))) == 4  # 1 kHz


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


def test_the_demos_give_a_walks_speed_to_a_tenth_of_a_km_h() -> None:
    """The examples' walks go at 13.89 and 19.44 m/s, 50.004 and 69.984 km/h: the
    captions, the stills and the printouts say 50 and 70, and keep a tenth that is
    there."""
    from acoustic_spaces.demos.passby import kmh

    assert kmh(13.89 * 3.6) == "50" and kmh(19.44 * 3.6) == "70"
    assert kmh(72.5) == "72.5" and kmh(54) == "54" and kmh(0.36) == "0.4"


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
    power = vehicle_power("1", 48.0, written.air.temperature_c)
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
    assert info["category"] == "1" and info["rpm"] == 2000.0
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
    for step in ("sound", "scene", "walk"):
        assert f"{step}: done already" in out
    # Another speed makes the car, the scene and the passes again.
    args = demo.resolved(demo.parser().parse_args(["all", "--step", "4"]))
    assert not demo.is_done(demo.step("sound"), args)
    assert not demo.is_done(demo.step("scene"), args)


def _street(tmp_path: Path) -> Path:
    """The street example, quick: no buildings nor barrier, the police car's walk 40 m
    long, the pavement alone, 16 kHz, seeded (two runs trace the same rays)."""
    folder = tmp_path / "street"
    shutil.copytree(Path(example_path("street")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"] = {"ground": {"extent": [[-30, -15], [30, 15]], "sky": 10}}
    data["assignments"] = {"ground": "asphalt"}
    for police in data["sources"]:
        if police["name"].startswith("police_car"):
            z = police["position"][2]
            police["position"] = [20, 2, z]
            police["walk"]["to"] = [[-20, 2, z]]
    data["listeners"] = [lis for lis in data["listeners"] if lis["name"] == "pavement"]
    data["simulation"].update({"sample_rate": 16_000, "ir_length_s": 0.2, "seed": 1})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 500}
    data["simulation"]["early"]["diffraction"] = "none"
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_the_passby_demo_drives_the_police_car_with_its_siren(tmp_path: Path) -> None:
    """--vehicle police_car on the quick street: the car's sound (CNOSSOS-EU's car) and
    the siren's in the pattern of --siren, the scene with the police car's two sources
    alone, at their sound powers, the siren's directivity table beside the scene and
    facing the way it drives, and its pass, the two heard together: their levels'
    energies added, the car's rumble below the siren's pitch and its sound as far under
    the siren's as its level, the siren the same as alone (--police-car siren); the
    motorcycle's sound has no tyres in it."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load
    from acoustic_spaces.parameters import A_WEIGHTING_DB
    from acoustic_spaces.synthesis import split_bands

    folder = _street(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    args = demo.parser().parse_args(["all", "--vehicle", "police_car", "--siren", "two_tone"])
    resolved = demo.resolved(args)
    assert demo.check(args) is None
    assert demo.moving(resolved) == ["police_car", "police_car_siren"]
    assert demo.plays(resolved) == {
        "police_car": "a police car at 50 km/h, made to CNOSSOS-EU's sound power",
        "police_car_siren": "its siren on its roof, two-tone",
    }
    run = ["all", "--vehicle", "police_car", "--siren", "two_tone", "--step", "4", "--no-mp4"]
    assert demo.main(run) == 0
    work = folder / "demo"
    info = pipeline.read_json(work / "police_car" / "police_car.json")
    assert info["siren"] == "two_tone" and info["plays"] == "car_and_siren"
    assert info["siren_sound_power_a_db"] == pytest.approx(128.6, abs=0.1)
    assert info["car"]["category"] == "1" and info["car"]["sound_power_a_db"] == pytest.approx(
        98.4, abs=0.1
    )
    for name in ("police_car.wav", "siren.wav"):
        x, rate = audio.read(work / "police_car" / name)
        assert rate == 16_000 and len(x) == pytest.approx(40.0 / 13.89 * rate, abs=2)
    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    assert scene.name.endswith(": a police car passing")
    assert [s.name for s in scene.sources if s.in_use] == ["police_car", "police_car_siren"]
    car, top = (
        next(s for s in scene.sources if s.name == n) for n in ("police_car", "police_car_siren")
    )
    assert car.level_db == info["car"]["level_db"] and car.directivity == "omni"
    assert top.level_db == info["siren_level_db"] and top.walk.face_forward and top.walk.doppler
    assert top.directivity.file == folder / "scene_demo_police_car_siren.csv"
    table, own = (
        f.read_text(encoding="utf-8").split("azimuth", 1)[1]
        for f in (top.directivity.file, folder / "siren.csv")
    )
    assert table == own  # the example's table, but for the comment over it
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert facts["siren"] == "two_tone" and facts["source"] == "police_car"
    assert facts["parts"] == ["police_car", "police_car_siren"]
    both = facts["listeners"]["pavement"]
    assert both["max_a_db"] == max(both["levels_a_db"]) and both["closest_m"] < 3.0
    parts = both["parts"]
    energy = sum(10.0 ** (p["sel_a_db"] / 10.0) for p in parts.values())
    assert both["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    assert parts["police_car"]["max_a_db"] < parts["police_car_siren"]["max_a_db"] - 20.0
    assert (work / "stills" / "00_pavement.png").exists()
    seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
    mix, _ = audio.read(work / "walk" / "pavement.binaural.wav")
    mixed = mix / seats["gain"]
    # The siren alone: the same siren, without the car's rumble.
    run = [*run, "--police-car", "siren"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.moving(args) == ["police_car_siren"]
    assert not demo.is_done(demo.step("sound"), args)
    assert not demo.is_done(demo.step("scene"), args)
    assert demo.main(run) == 0
    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == ["police_car_siren"]
    facts = pipeline.read_json(work / "walk" / "walk.json")
    alone = facts["listeners"]["pavement"]
    assert alone["sel_a_db"] == pytest.approx(parts["police_car_siren"]["sel_a_db"], abs=0.01)
    assert alone["max_a_db"] == pytest.approx(parts["police_car_siren"]["max_a_db"], abs=0.01)
    seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
    mix, _ = audio.read(work / "walk" / "pavement.binaural.wav")
    siren_only = mix / seats["gain"]
    n = min(len(mixed), len(siren_only))
    low = {}
    for key, x in (("both", mixed), ("siren", siren_only)):
        bands = split_bands(x[:n, 0], 16_000)
        low[key] = [float(np.sum(bands[k] ** 2)) for k in (0, 1)]
    more = [10.0 * math.log10(b / a) for b, a in zip(low["both"], low["siren"], strict=True)]
    # At 63 and 125 Hz the siren sends next to nothing: more than 95 % of the police
    # car's sound there is the car's (the rest is the octave filters' leak).
    assert all(1.0 - 10.0 ** (-m / 10.0) > 0.95 for m in more)
    # Each part plays at its own sound power: the car's sound is as far under the
    # siren's as its level is (some 29 dB(A)), not as loud as the siren.
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)

    def a_energy(x: np.ndarray) -> float:
        return sum(
            float(weights @ np.sum(split_bands(x[:, k], 16_000) ** 2, axis=1))
            for k in range(x.shape[1])
        )

    under = 10.0 * math.log10(a_energy(siren_only[:n]) / a_energy(mixed[:n] - siren_only[:n]))
    told = parts["police_car_siren"]["sel_a_db"] - parts["police_car"]["sel_a_db"]
    assert told > 20.0 and under == pytest.approx(told, abs=1.5)
    # Another pattern makes the sound and the scene again, with the wail's sound power.
    args = demo.resolved(demo.parser().parse_args(["all", "--vehicle", "police_car"]))
    assert not demo.is_done(demo.step("sound"), args)
    assert not demo.is_done(demo.step("scene"), args)
    # The motorcycle: its engine alone, two cylinders at 4000 rpm.
    assert demo.main(["sound", "--vehicle", "motorcycle"]) == 0
    info = pipeline.read_json(work / "motorcycle" / "motorcycle.json")
    assert info["category"] == "4b" and info["rolling_db"] is None
    assert info["rpm"] == 4000.0 and info["cylinders"] == 2
