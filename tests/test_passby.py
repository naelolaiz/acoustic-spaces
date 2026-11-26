"""The pass-by demos: CNOSSOS-EU's road vehicles, the sound made to them, the police car's
siren, and the demo of the roadside and the street run on a quick copy of the roadside."""

from __future__ import annotations

import argparse
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


#: Table F-1 of Directive 2002/49/EC, Annex II, as amended by Commission Delegated
#: Directive (EU) 2021/1226, 63 Hz to 8 kHz: A_R, B_R, A_P, B_P of each category (the
#: consolidated text and UBA's BUB-D 2021, Anlage 4, table A-1, agree).
TABLE_F1 = {
    "1": (
        (83.1, 89.2, 87.7, 93.1, 100.1, 96.7, 86.8, 76.2),
        (30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0),
        (97.9, 92.5, 90.7, 87.2, 84.7, 88.0, 84.4, 77.1),
        (-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0),
    ),
    "2": (
        (88.7, 93.2, 95.7, 100.9, 101.7, 95.1, 87.8, 83.6),
        (30.0, 35.8, 32.6, 23.8, 30.1, 36.2, 38.3, 40.1),
        (105.5, 100.2, 100.5, 98.7, 101.0, 97.8, 91.2, 85.0),
        (-1.9, 4.7, 6.4, 6.5, 6.5, 6.5, 6.5, 6.5),
    ),
    "3": (
        (91.7, 96.2, 98.2, 104.9, 105.1, 98.5, 91.1, 85.6),
        (30.0, 33.5, 31.3, 25.4, 31.8, 37.1, 38.6, 40.6),
        (108.8, 104.2, 103.5, 102.9, 102.6, 98.5, 93.8, 87.5),
        (0.0, 3.0, 4.6, 5.0, 5.0, 5.0, 5.0, 5.0),
    ),
    "4a": (
        None,
        None,
        (93.0, 93.0, 93.5, 95.3, 97.2, 100.4, 95.8, 90.9),
        (4.2, 7.4, 9.8, 11.6, 15.7, 18.9, 20.3, 20.6),
    ),
    "4b": (
        None,
        None,
        (99.9, 101.9, 96.7, 94.4, 95.2, 94.7, 92.1, 88.6),
        (3.2, 5.9, 11.9, 11.6, 11.5, 12.6, 11.1, 12.0),
    ),
}


def test_every_coefficient_is_table_f1s() -> None:
    """Every coefficient of every category, against the amended Table F-1 as published,
    so that a slip in one band of a category no demo plays is caught too."""
    assert set(CATEGORIES) == set(TABLE_F1)
    for key, (a_r, b_r, a_p, b_p) in TABLE_F1.items():
        kind = CATEGORIES[key]
        got = (kind.rolling_a, kind.rolling_b, kind.propulsion_a, kind.propulsion_b)
        assert got == (a_r, b_r, a_p, b_p), key


@pytest.mark.parametrize(
    ("example", "vehicle"),
    [
        ("roadside", THE_CAR),
        ("roadside", TRUCK),
        ("street", siren.HONKING_CAR.body),
        ("street", BUS),
        ("street", MOTORCYCLE),
        ("street", siren.POLICE_CAR_VEHICLE.car),
        ("street", siren.FIRE_ENGINE_VEHICLE.body),
    ],
)
def test_the_examples_road_vehicles_are_cnossos_at_their_speed(example: str, vehicle) -> None:
    """The road vehicles of the roadside and the street, the police car's car and the fire
    engine's truck too: point
    sources 0.05 m up, left out of the example's simulations, CNOSSOS-EU's levels for
    their category at the speed of their walk, with Doppler."""
    scene = Scene.from_yaml(example_path(example))
    (source,) = [s for s in scene.sources if s.name == vehicle.name]
    assert not source.in_use and not source.extended and source.position[2] == 0.05
    assert (
        source.trajectory.doppler
        and source.trajectory.enabled
        and source.trajectory.keyframes is None
    )
    power = vehicle_power(
        vehicle.category, source.trajectory.speed_m_s * 3.6, scene.air.temperature_c
    )
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
    assert car.trajectory.doppler and car.trajectory.face_forward
    assert car_problem(scene, path.name, siren.POLICE_CAR_VEHICLE) is None
    # A siren whose walk goes elsewhere does not go along with its car.
    walk = car.trajectory.model_copy(update={"to": [(-50.0, 2.0, 1.5)]})
    moved = scene.model_copy(
        update={
            "sources": [
                s.model_copy(update={"trajectory": walk}) if s.name == car.name else s
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


def test_the_streets_fire_engine_has_its_siren_and_horn_at_their_rules() -> None:
    """The street's fire engine: left out, its siren on its cab's roof and its air horn
    along its walk, both facing the way it drives with siren.csv, the two tones
    FIRE_SIREN_GAIN_DB louder than the police car's and the horn at its sound power
    averaged over its blasts."""
    from acoustic_spaces.demos import horn
    from acoustic_spaces.demos.passby import car_problem

    path = Path(example_path("street"))
    scene = Scene.from_yaml(path)
    sources = {s.name: s for s in scene.sources}
    rate = scene.simulation.sample_rate
    vehicle = siren.FIRE_ENGINE_VEHICLE
    assert vehicle.every_part() == (siren.FIRE_ENGINE, siren.FIRE_SIREN, siren.FIRE_HORN)
    assert car_problem(scene, path.name, vehicle) is None
    for name, height, power in (
        (siren.FIRE_SIREN, siren.FIRE_SIREN_HEIGHT_M, siren.siren_power("two_tone", rate) + 6.0),
        (siren.FIRE_HORN, horn.HEIGHT_M, horn.horn_power(rate)),
    ):
        source = sources[name]
        assert not source.in_use and source.position[2] == height
        assert source.trajectory.doppler and source.trajectory.face_forward
        assert source.trajectory.to[-1][:2] == sources[siren.FIRE_ENGINE].trajectory.to[-1][:2]
        assert source.level_db == pytest.approx(power, abs=0.05)
        assert source.directivity.file == path.parent / "siren.csv"


def test_the_streets_car_has_its_horn_off_unless_asked() -> None:
    """The street's car: its horn left out, 0.5 m up along the car's walk, facing the
    way it drives with siren.csv, at its sound power averaged over its blasts; the car
    plays alone by default, said as the roadside's car says it."""
    from acoustic_spaces.demos import horn
    from acoustic_spaces.demos.passby import car_problem

    path = Path(example_path("street"))
    scene = Scene.from_yaml(path)
    sources = {s.name: s for s in scene.sources}
    vehicle = siren.HONKING_CAR
    assert vehicle.every_part() == (siren.CAR, siren.CAR_HORN_PART)
    assert car_problem(scene, path.name, vehicle) is None
    source = sources[siren.CAR_HORN_PART]
    assert not source.in_use and source.position[2] == horn.CAR_HEIGHT_M
    assert source.trajectory.doppler and source.trajectory.face_forward
    assert source.trajectory.to[-1][:2] == sources[siren.CAR].trajectory.to[-1][:2]
    rate = scene.simulation.sample_rate
    assert source.level_db == pytest.approx(horn.horn_power(rate, horn.CAR_HORN), abs=0.05)
    assert source.directivity.file == path.parent / "siren.csv"
    args = argparse.Namespace()
    assert vehicle.parts(args) == (siren.CAR,) and vehicle.played(args) == ()
    # Alone, the car says what the roadside's car says.
    facts = {"parts": ["car"], "cylinders": 4, "rpm": 2000.0, "speed_kmh": 50.0}
    assert vehicle.credits(facts, {}) == THE_CAR.credits(facts, {})
    file = {siren.CAR: Path("mine.wav")}
    assert vehicle.credits(facts, file) == THE_CAR.credits(facts, file)


def test_the_car_horn_meets_the_rule_in_the_engine(tmp_path: Path) -> None:
    """The car's disc horns: two blasts a quarter of a second long every 3 s, at their
    level within the rule's 0.2 s; their tone discs' resonance their loudest harmonics;
    110 dB(A) 2 m away on their axis while they sound, as Hella gives its S77, within
    the rule's 105 to 118 on the bench; validation/urban_sources.py's check: above the
    least on the car, the band 1800 to 3550 Hz above 105 dB(A) and above any component
    over it."""
    import sys

    from acoustic_spaces.demos import horn
    from acoustic_spaces.parameters import a_weighted

    kind = horn.CAR_HORN
    assert kind.rise_s <= 0.2 and 105.0 <= kind.on_axis_db <= 118.0
    t = np.array([0.1, 0.3, 0.5, 0.8, 2.9])
    assert horn.sounding(t, kind) == pytest.approx([1.0, 0.0, 1.0, 0.0, 0.0])
    rate = 16_000
    x = horn.horn_sound(rate, kind.every_s, kind)
    assert np.sqrt(np.mean(x**2)) == pytest.approx(horn.SOUND_RMS, rel=1e-3)
    tone = horn.blast(rate, 1.0, kind)
    spectrum = np.abs(np.fft.rfft(tone))
    f = np.fft.rfftfreq(len(tone), 1.0 / rate)
    loudest = f[np.argsort(spectrum)[-2:]]
    assert all(abs(g - horn.DISC_HZ) < 0.25 * horn.DISC_HZ for g in loudest)
    assert all(min(abs(g / h - round(g / h)) for h in kind.hz) < 1e-9 for g in loudest)
    blast = horn.blast_power(48_000, kind)
    di = siren.directivity_index_db()
    on_axis = a_weighted(blast) + di - 20 * math.log10(kind.at_m) - 10 * math.log10(4 * math.pi)
    assert on_axis == pytest.approx(kind.on_axis_db, abs=1e-9)
    assert horn.horn_power(48_000, kind) == pytest.approx(blast + 10 * math.log10(horn.duty(kind)))
    made = horn.horn_sound(rate, kind.power_s, kind)
    assert horn.recorded_gain(made, rate, kind) == pytest.approx(1.0, rel=1e-12)
    honking = siren.HONKING_CAR
    assert honking.input_gain(argparse.Namespace(), siren.CAR_HORN_PART, made, rate) == (
        pytest.approx(1.0)
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    import urban_sources

    out = urban_sources.check_car_horn(tmp_path)
    assert out["bench_db_a"] == pytest.approx(kind.on_axis_db, abs=0.15)
    assert min(out["vehicle_db_a"].values()) >= horn.VEHICLE_DB_A
    assert out["band_db_a"] >= 105.0 and out["band_over_above_db"] > 0.0


@pytest.mark.parametrize(
    "vehicle", [siren.POLICE_CAR_VEHICLE, siren.FIRE_ENGINE_VEHICLE, siren.HONKING_CAR]
)
def test_the_credit_of_a_vehicle_with_a_siren_fits_its_stills(vehicle) -> None:
    """The stills' credit line of the police car, of the fire engine and of the car with
    its horn, every source they play made, fits the still's width whole, not cut
    short."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    from acoustic_spaces.demos.pictures import MARGIN, SIZE, fitted

    facts = {"cylinders": 6, "rpm": 1500.0, "siren": "two_tone", "horn": "blasts"}
    credit = vehicle.credits(facts, {})
    assert credit.endswith("faded in and out over 1 s") == (vehicle is siren.POLICE_CAR_VEHICLE)
    if vehicle is not siren.POLICE_CAR_VEHICLE:
        assert credit.endswith("faded in and out over 1 s, not the horn")
    draw = ImageDraw.Draw(Image.new("RGB", SIZE))
    line, _ = fitted(draw, credit, 19, SIZE[0] - 2 * MARGIN)
    assert line == credit


@pytest.mark.parametrize(
    ("span_s", "step_s"), [(10.8, 2.0), (30.0, 5.0), (104.0, 10.0), (600.0, 50.0)]
)
def test_the_graphs_time_axis_keeps_its_labels_apart(span_s: float, step_s: float) -> None:
    """A car's pass labels every 2 s, a long pass (an aircraft's, 104 s) every 10 s, so
    that the labels never crowd each other, nor the widest of them overlap."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    from acoustic_spaces.demos.passby_pictures import GRAPH_BOX, time_step
    from acoustic_spaces.demos.pictures import SIZE, font

    width = GRAPH_BOX[2]
    step = time_step(span_s, width)
    assert step == step_s
    draw = ImageDraw.Draw(Image.new("RGB", SIZE))
    widest = draw.textlength(f"{span_s:.0f} s", font=font("Regular", 16))
    assert step * width / span_s > widest + 10.0


@pytest.mark.parametrize("example", ["roadside", "street"])
def test_a_vehicles_own_options_matter_with_it_alone(example: str) -> None:
    """The options of some of the vehicles alone (the police car's siren, the road
    vehicles' engine speed) say they matter only with --vehicle at one of them; those
    every vehicle has (the speed) and the sources each picks always matter. What a
    vehicle of several sources plays is --sources, no option of its own."""
    from acoustic_spaces.demos import load

    demo = load(Path(example_path(example)).parent / "passby_demo.py")
    assert demo.option("speed").only is None
    if example == "roadside":
        # The HRTFs matter with binaural mixes alone, in every demo.
        assert all(o.only is None for o in demo.options if o.name != "hrtf")
        assert "sources" not in {o.name for o in demo.options}  # one source a vehicle
        return
    # The tram has no engine of a road vehicle.
    road = ("car", "bus", "motorcycle", "police_car", "fire_engine")
    assert demo.option("rpm").only == ("vehicle", road)
    assert demo.option("siren").only == ("vehicle", ("police_car",))
    assert demo.option("sources").only is None and demo.option("sources").kind == "sources"
    names = {o.name for o in demo.options}
    assert not names & {"car", "police_car", "fire_engine"}
    siren_option = demo.option("siren")
    assert siren_option.applies({"vehicle": "police_car"})
    assert not siren_option.applies({"vehicle": "car"})


def test_the_horn_blasts_as_its_rule_and_its_assumptions_say() -> None:
    """The air horn: BLASTS blasts BLAST_S long, GAP_S apart, every EVERY_S, each at its
    level RISE_S after it starts (the rule: 0.2 s at most); its two trumpets' fundamentals
    its loudest tones; SOUND_RMS over whole periods; 118 dB(A) 2 m away on its axis in a
    free field while it sounds, and its sound power averaged over its blasts and pauses
    as much under that as its sound is silent."""
    from acoustic_spaces.demos import horn
    from acoustic_spaces.parameters import a_weighted

    assert horn.RISE_S <= 0.2 and 105.0 <= horn.ON_AXIS_DB_A <= 118.0
    t = np.array([0.0, horn.RISE_S, 0.3, horn.BLAST_S + 0.1, horn.BLAST_S + horn.GAP_S + 0.3])
    assert horn.sounding(t) == pytest.approx([0.0, 1.0, 1.0, 0.0, 1.0])
    assert horn.sounding(t + horn.EVERY_S) == pytest.approx(horn.sounding(t))
    assert horn.sounding([2 * (horn.BLAST_S + horn.GAP_S) + 0.1]) == pytest.approx([0.0])
    rate = 16_000
    x = horn.horn_sound(rate, horn.EVERY_S)
    assert np.sqrt(np.mean(x**2)) == pytest.approx(horn.SOUND_RMS, rel=1e-3)
    tone = horn.blast(rate, 1.0)
    spectrum = np.abs(np.fft.rfft(tone))
    f = np.fft.rfftfreq(len(tone), 1.0 / rate)
    assert sorted(f[np.argsort(spectrum)[-2:]]) == pytest.approx(list(horn.HORN_HZ))
    blast = horn.blast_power(48_000)
    di = siren.directivity_index_db()
    on_axis = a_weighted(blast) + di - 20 * math.log10(horn.AT_M) - 10 * math.log10(4 * math.pi)
    assert on_axis == pytest.approx(horn.ON_AXIS_DB_A, abs=1e-9)
    assert 0.3 < horn.duty() < 0.4
    assert horn.horn_power(48_000) == pytest.approx(blast + 10 * math.log10(horn.duty()))
    with pytest.raises(ValueError, match="duration"):
        horn.horn_sound(rate, 0.0)
    # A recording of a horn plays its blasts as loud as the made horn's, whatever its
    # pauses: the made horn given back as it is, and one long blast.
    made = horn.horn_sound(rate, horn.POWER_S)
    assert horn.recorded_gain(made, rate) == pytest.approx(1.0, rel=1e-12)
    long = 0.3 * horn.blast(rate, 2.0)
    full = horn.sounding(np.arange(len(made)) / rate) == 1.0
    level = np.sqrt(np.mean((long * horn.recorded_gain(long, rate)) ** 2))
    assert level == pytest.approx(np.sqrt(np.mean(made[full] ** 2)), rel=0.02)
    assert horn.recorded_gain(np.zeros(rate), rate) is None
    vehicle = siren.FIRE_ENGINE_VEHICLE
    run = argparse.Namespace()
    assert vehicle.input_gain(run, siren.FIRE_HORN, long, rate) == horn.recorded_gain(long, rate)
    assert vehicle.input_gain(run, siren.FIRE_SIREN, long, rate) == pytest.approx(
        siren.SOUND_RMS / audio.gated_rms(long, rate)
    )


@pytest.mark.parametrize("kind", ["CAR_HORN", "AIR_HORN"])
def test_a_horns_loudest_place_does_not_follow_the_step(kind: str) -> None:
    """A place is no louder for the beats between a horn's notes (70 Hz between the car's
    360 and 430 Hz, 30 Hz between the air horn's 150 and 180 Hz) that a short step's
    time around it resolves: its sound's power is averaged over 125 ms first, so at 50
    km/h its loudest place is a blast's level to 0.15 dB at every step from 5 cm to 1 m
    (where without it it climbs to 3 to 5 dB over at 5 cm), and over whole periods its
    loudness keeps its sound power."""
    from acoustic_spaces.demos import horn
    from acoustic_spaces.demos.passby import LOUDNESS_WINDOW_S, blocks_loudness

    assert LOUDNESS_WINDOW_S == 0.125  # IEC 61672-1's Fast time constant
    played = getattr(horn, kind)
    rate = 16_000
    x = horn.horn_sound(rate, 3 * played.every_s, played)
    rms = math.sqrt(float(np.mean(x**2)))
    blast = 10.0 * math.log10(1.0 / horn.duty(played))
    speed = 50.0 / 3.6
    for step in (0.05, 0.25, 0.5, 1.0):
        at = np.arange(0.0, 3 * played.every_s, step / speed) + 0.3 * step / speed
        loud = blocks_loudness(x, rate, at, rms)
        assert 10.0 * math.log10(loud.max()) == pytest.approx(blast, abs=0.15), step
        assert np.mean(loud) == pytest.approx(1.0, abs=0.01), step


def test_a_horns_levels_follow_its_blasts() -> None:
    """The captions have a horn as loud as its sound plays around each place: over whole
    periods of its blasts as loud as its sound power, during a blast 1/duty louder,
    silent in its pauses; the fire engine's siren, which sounds all along, as it is."""
    from acoustic_spaces.demos import horn
    from acoustic_spaces.demos.passby import blocks_loudness, sounded
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.walkthrough import PassBy

    rate = 16_000
    x = horn.horn_sound(rate, 3 * horn.EVERY_S)
    at = np.arange(0.0, 3 * horn.EVERY_S, 0.1) + 0.05
    loud = blocks_loudness(x, rate, at, horn.SOUND_RMS)
    # To 0.02 dB: the sound starts with a blast, whose first 62.5 ms are averaged over
    # what of the 125 ms around them lies in the sound.
    assert np.mean(loud) == pytest.approx(1.0, rel=5e-3)
    assert np.max(loud) == pytest.approx(1.0 / horn.duty(), rel=1e-3)
    assert np.min(loud) == 0.0
    vehicle = siren.FIRE_ENGINE_VEHICLE
    assert np.array_equal(vehicle.sounding(None, siren.FIRE_HORN, x, rate, at), loud)
    assert vehicle.sounding(None, siren.FIRE_SIREN, x, rate, at) is None
    assert THE_CAR.sounding(None, "car", x, rate, at) is None
    levels = np.full((len(at), 8), 80.0)
    passing = PassBy("horn", "here", at, levels, np.full(len(at), 85.0))
    heard = sounded(passing, loud)
    blast = 10.0 * math.log10(1.0 / horn.duty())
    assert heard.levels[np.argmax(loud)] == pytest.approx(levels[0] + blast)
    # Over whole periods its energy is its sound power's, as the steady level's (to
    # 0.02 dB, as its loudness above).
    steady = 10.0 ** (a_weighted(levels[0]) / 10.0)
    assert np.mean(10.0 ** (heard.levels_a / 10.0)) == pytest.approx(steady, rel=5e-3)


def test_the_fire_engine_meets_the_rules_in_the_engine(tmp_path: Path) -> None:
    """validation/urban_sources.py's check of the fire engine: its two tones 7 m in front
    as loud as the Swiss rule allows to a tenth of a dB, within it on the bench; its horn
    at the most the rule allows on its bench, above the least on the vehicle, its band
    1800 to 3550 Hz above 105 dB(A) and above any component over it."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    import urban_sources

    out = urban_sources.check_fire_engine(tmp_path)
    loudest = max(out["siren"]["vehicle_db_a"].values())
    assert siren.FIRE_SIREN_DB_A - 0.15 < loudest <= siren.FIRE_SIREN_DB_A
    assert 116.0 <= out["siren"]["bench_db_a"] <= 129.0
    assert out["horn"]["bench_db_a"] == pytest.approx(118.0, abs=0.05)
    assert min(out["horn"]["vehicle_db_a"].values()) >= 93.0
    assert out["horn"]["band_db_a"] >= 105.0 and out["horn"]["band_over_above_db"] > 0.0


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


def run_of(demo, *argv: str) -> Path:
    """The folder of the demo's run with the options ``argv``: named after its settings."""
    return demo.run_folder(demo.parser().parse_args(["all", *argv]))


def mix_at(work: Path, listener: str) -> Path:
    """The mix of a run's pass at ``listener``, as its seats.json names it."""
    seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
    return work / "walk" / seats["seats"][listener]["file"]


def _roadside(tmp_path: Path) -> Path:
    """The roadside example, quick: a road 40 m long, two listeners, 16 kHz."""
    folder = tmp_path / "roadside"
    shutil.copytree(Path(example_path("roadside")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    (car,) = [s for s in data["sources"] if s["name"] == "car"]
    car["position"] = [20, 2, 0.05]
    car["trajectory"]["to"] = [[-20, 2, 0.05]]
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
    car["trajectory"]["enabled"] = False
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert "the car's walk is off in scene.yaml" in demo.check(args)
    assert demo.moving(demo.resolved(args)) == []
    # Keyframes at one steady speed drive as they are timed: 40 m in 3 s.
    car["trajectory"] = {
        "keyframes": [{"time_s": 3.0, "position": [-20, 2, 0.05]}],
        "doppler": True,
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(args) is None
    assert "at 48.0 km/h (13.33 m/s, as its walk in the scene goes)" in demo.summary(
        demo.resolved(args)
    )
    # Keyframes at two speeds: CNOSSOS-EU has a car's power at one; --speed gives it.
    car["trajectory"]["keyframes"] = [
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


def test_a_vehicles_sources_left_out_walk_at_the_demos_speed(tmp_path: Path) -> None:
    """At --speed, the sources of a vehicle that a run leaves out (the police car itself
    with --sources police_car_siren, the fire engine's horn, the train's horn) walk at that
    speed too in the demo's scene, out of use: its sources still go along one walk,
    which the walk step checks."""
    import shutil

    from acoustic_spaces import Scene
    from acoustic_spaces.demos import load
    from acoustic_spaces.demos.passby import car_problem
    from acoustic_spaces.demos.rail import TRAIN_VEHICLE
    from acoustic_spaces.demos.siren import FIRE_ENGINE_VEHICLE, POLICE_CAR_VEHICLE

    for example, vehicle, run in (
        (
            "street",
            POLICE_CAR_VEHICLE,
            ["--vehicle", "police_car", "--sources", "police_car_siren"],
        ),
        (
            "street",
            FIRE_ENGINE_VEHICLE,
            ["--vehicle", "fire_engine", "--sources", "fire_engine_siren"],
        ),
        ("railway", TRAIN_VEHICLE, ["--train", "s_bahn"]),
    ):
        folder = tmp_path / f"{example}_{vehicle.name}"
        shutil.copytree(Path(example_path(example)).parent, folder)
        demo = load(folder / "passby_demo.py").at(folder)
        assert demo.main(["scene", *run, "--speed", "30"]) == 0
        scene = Scene.from_yaml(run_of(demo, *run, "--speed", "30") / "scene_demo.yaml")
        assert car_problem(scene, "scene_demo.yaml", vehicle) is None, vehicle.name
        mine = [s for s in scene.sources if s.name in vehicle.every_part()]
        assert any(not s.in_use for s in mine)
        assert {round(s.trajectory.speed_m_s, 4) for s in mine} == {round(30.0 / 3.6, 4)}


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
    car["trajectory"] = {
        "keyframes": [
            {"time_s": 1.5, "position": [0, 2, 0.05]},
            {"time_s": 3.0, "position": [-20, 2, 0.05], "orientation": {"yaw": 180}},
        ],
        "enabled": True,
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    demo = load(folder / "passby_demo.py").at(folder)
    assert demo.main(["scene"]) == 0
    written = Scene.from_yaml(run_of(demo) / "scene_demo.yaml")
    (driven,) = [s for s in written.sources if s.name == "car"]
    assert driven.in_use and driven.trajectory.doppler and len(driven.trajectory.keyframes) == 2
    assert walk_speed(driven) == pytest.approx(40.0 / 3.0)
    power = vehicle_power("1", 48.0, written.air.temperature_c)
    assert driven.level_db == pytest.approx(power.total_db, abs=0.05)
    args = demo.resolved(demo.defaults())
    assert demo.is_done(demo.step("scene"), args)
    # Another speed is another run, with a scene of its own: the one at the walk's speed
    # stays as it is.
    args = demo.resolved(demo.parser().parse_args(["all", "--speed", "72"]))
    assert not demo.is_done(demo.step("scene"), args)
    assert demo.main(["scene", "--speed", "72"]) == 0
    fast = run_of(demo, "--speed", "72") / "scene_demo.yaml"
    (driven,) = [s for s in Scene.from_yaml(fast).sources if s.name == "car"]
    assert driven.trajectory.keyframes is None and driven.trajectory.speed_m_s == pytest.approx(
        20.0
    )
    assert (
        driven.trajectory.to == [(0.0, 2.0, 0.05), (-20.0, 2.0, 0.05)] and driven.trajectory.doppler
    )
    assert demo.is_done(demo.step("scene"), args)
    assert demo.is_done(demo.step("scene"), demo.resolved(demo.defaults()))
    # The editor follows the car along the walk the demo simulates: this one, at --speed.
    assert demo.trajectories(args).sources == Scene.from_yaml(fast).sources
    # The same, from Python: a wait drops out of the places.
    source = Scene.from_yaml(path)
    (car,) = [s for s in source.sources if s.name == "car"]
    frames = [
        *car.trajectory.keyframes[:1],
        car.trajectory.keyframes[0].model_copy(update={"time_s": 2.0}),
    ]
    waiting = car.model_copy(
        update={"trajectory": car.trajectory.model_copy(update={"keyframes": frames})}
    )
    assert driving(waiting, 36.0).trajectory.to == [(0.0, 2.0, 0.05)]


def test_the_passby_demo_drives_the_car_past_every_listener(tmp_path: Path, capsys) -> None:
    """Every step on the quick roadside: the car's sound, the scene with the car alone at
    --speed, the passes (the level along each, its maximum near the middle, the pitch
    falling, one gain for all), the tour and its stills; then each step is done."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load

    folder = _roadside(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    assert demo.main(["all", "--step", "4", "--speed", "72", "--no-mp4"]) == 0
    run = ["--step", "4", "--speed", "72", "--no-mp4"]
    work = run_of(demo, *run)
    assert work == folder / "demo" / "passby_car_speed-72_step-4_binaural"
    info = pipeline.read_json(work / "car" / "car.json")
    assert info["category"] == "1" and info["rpm"] == 2000.0
    assert info["speed_kmh"] == 72.0 and info["duration_s"] == pytest.approx(2.0)
    x, rate = audio.read(work / "car" / "car.wav")
    assert rate == 16_000 and len(x) == 2 * rate
    scene = Scene.from_yaml(work / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == ["car"]
    (car,) = [s for s in scene.sources if s.name == "car"]
    assert car.trajectory.speed_m_s == pytest.approx(20.0) and car.level_db == info["level_db"]
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert set(facts["listeners"]) == {"verge", "yard_25m"} and len(facts["times"]) == 11
    verge = facts["listeners"]["verge"]
    assert verge["closest_m"] == pytest.approx(math.hypot(5.5, 1.45), abs=0.01)
    assert 0.8 < verge["max_s"] < 1.2 and verge["max_a_db"] == max(verge["levels_a_db"])
    assert verge["sel_a_db"] > verge["leq_a_db"]  # over 2 s: SEL = Leq + 3 dB
    assert 1.0 < verge["pitch_semitones"] < 12 * math.log2(363.0 / 323.0)
    seats = pipeline.read_json(work / "walk" / "seats.json")
    # Each mix is named after what plays and where it is heard.
    assert seats["seats"]["verge"]["file"] == "car__verge.binaural.wav"
    peaks = [
        np.abs(audio.read(work / "walk" / s["file"])[0]).max() for s in seats["seats"].values()
    ]
    assert max(peaks) == pytest.approx(10.0 ** (-1.0 / 20.0), rel=1e-3)
    assert sorted(p.name for p in (work / "stills").glob("*.png")) == [
        "00_verge.png",
        "01_yard_25m.png",
    ]
    named = folder / "demo" / work.name
    for product in ("_thumbnail.png", ".wav"):
        assert named.with_name(work.name + product).exists()
    chapters = named.with_name(work.name + "_chapters.txt").read_text(encoding="utf-8")
    assert chapters.splitlines()[0] == "0:00 verge"
    assert chapters.splitlines()[1].endswith(" yard 25m")
    capsys.readouterr()
    assert demo.main(["all", *run]) == 0
    out = capsys.readouterr().out
    for step in ("sound", "scene", "walk"):
        assert f"{step}: done already" in out
    # Another speed is another run, with its own car, scene and passes.
    args = demo.resolved(demo.parser().parse_args(["all", "--step", "4"]))
    assert not demo.is_done(demo.step("sound"), args)
    assert not demo.is_done(demo.step("scene"), args)
    # --listeners: a run of its own, whose video plays the passes of the run at every
    # listener (made already, so the walk is done), with that listener alone; its
    # files take the listener's name, and the run at every listener keeps its own.
    capsys.readouterr()
    assert demo.main(["all", *run, "--listeners", "yard_25m"]) == 0
    assert "walk: done already" in capsys.readouterr().out
    yard = run_of(demo, *run, "--listeners", "yard_25m")
    assert yard.name == "passby_car_yard_25m_speed-72_step-4_binaural"
    chapters = yard.with_name(yard.name + "_chapters.txt").read_text(encoding="utf-8")
    assert chapters.splitlines() == ["0:00 yard 25m"]
    assert sorted(p.name for p in (work / "stills").glob("*.png")) == [
        "00_verge.png",
        "01_yard_25m.png",
    ]
    assert sorted(p.name for p in (yard / "stills").glob("*.png")) == ["00_yard_25m.png"]
    assert not (yard / "walk").exists()
    # A walk at one listener simulates it alone, into its own run's folder.
    for step in ("sound", "scene", "walk"):
        assert demo.main([step, *run, "--listeners", "verge"]) == 0
    alone = run_of(demo, *run, "--listeners", "verge")
    assert set(pipeline.read_json(alone / "walk" / "walk.json")["listeners"]) == {"verge"}
    assert set(pipeline.read_json(work / "walk" / "walk.json")["listeners"]) == {
        "verge",
        "yard_25m",
    }
    # The video of a run in another format than the passes': there are none of it.
    with pytest.raises(SystemExit, match="walk.json; run first: python passby_demo.py walk"):
        demo.main(["video", *run, "--listeners", "verge", "--format", "mono"])
    assert pipeline.label_text("binaural-kemar") == "binaural through the HRTFs of kemar"
    assert pipeline.label_text(None) == "of an output not kept"


def _street(tmp_path: Path) -> Path:
    """The street example, quick: no buildings nor barrier, the police car's and the fire
    engine's walks 40 m long and the car's 80 m, the pavement alone, 16 kHz, seeded (two
    runs trace the same rays)."""
    folder = tmp_path / "street"
    shutil.copytree(Path(example_path("street")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"] = {"ground": {"extent": [[-45, -15], [45, 15]], "sky": 10}}
    data["assignments"] = {"ground": "asphalt"}
    for each in data["sources"]:
        if each["name"].startswith("police_car"):
            z = each["position"][2]
            each["position"] = [20, 2, z]
            each["trajectory"]["to"] = [[-20, 2, z]]
        if each["name"].startswith("fire_engine"):
            z = each["position"][2]
            each["position"] = [-20, -2, z]
            each["trajectory"]["to"] = [[20, -2, z]]
        if each["name"].startswith("car"):
            # Long enough for the horn's second blasts, after its sound's fade in.
            z = each["position"][2]
            each["position"] = [-40, -2, z]
            each["trajectory"]["to"] = [[40, -2, z]]
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
    the siren's as its level, the siren the same as alone (--sources police_car_siren); the
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
    # The demo says the vehicle in words, not as its source's name.
    assert demo.summary(resolved).startswith("The police car drives 40.0 m along its walk")
    run = ["all", "--vehicle", "police_car", "--siren", "two_tone", "--step", "4", "--no-mp4"]
    assert demo.main(run) == 0
    work = run_of(demo, *run[1:])
    info = pipeline.read_json(work / "police_car" / "police_car.json")
    assert info["siren"] == "two_tone" and info["parts"] == ["police_car", "police_car_siren"]
    assert info["siren_sound_power_a_db"] == pytest.approx(128.6, abs=0.1)
    car = info["vehicle"]
    assert car["category"] == "1" and car["sound_power_a_db"] == pytest.approx(98.4, abs=0.1)
    for name in ("police_car.wav", "siren.wav"):
        x, rate = audio.read(work / "police_car" / name)
        assert rate == 16_000 and len(x) == pytest.approx(40.0 / 13.89 * rate, abs=2)
    scene = Scene.from_yaml(work / "scene_demo.yaml")
    assert scene.name.endswith(": a police car passing")
    assert [s.name for s in scene.sources if s.in_use] == ["police_car", "police_car_siren"]
    car, top = (
        next(s for s in scene.sources if s.name == n) for n in ("police_car", "police_car_siren")
    )
    assert car.level_db == info["vehicle"]["level_db"] and car.directivity == "omni"
    assert (
        top.level_db == info["siren_level_db"]
        and top.trajectory.face_forward
        and top.trajectory.doppler
    )
    assert top.directivity.file == work / "scene_demo_police_car_siren.csv"
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
    mix, _ = audio.read(mix_at(work, "pavement"))
    mixed = mix / seats["gain"]
    # The siren alone: the same siren, without the car's rumble.
    run = [*run, "--sources", "police_car_siren"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.moving(args) == ["police_car_siren"]
    assert run_of(demo, *run[1:]).name == (
        "passby_police_car_police_car_siren_siren-two_tone_step-4_binaural"
    )
    assert not demo.is_done(demo.step("sound"), args)
    assert not demo.is_done(demo.step("scene"), args)
    assert demo.main(run) == 0
    work = run_of(demo, *run[1:])
    scene = Scene.from_yaml(work / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == ["police_car_siren"]
    facts = pipeline.read_json(work / "walk" / "walk.json")
    alone = facts["listeners"]["pavement"]
    assert alone["sel_a_db"] == pytest.approx(parts["police_car_siren"]["sel_a_db"], abs=0.01)
    assert alone["max_a_db"] == pytest.approx(parts["police_car_siren"]["max_a_db"], abs=0.01)
    seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
    mix, _ = audio.read(mix_at(work, "pavement"))
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
    work = run_of(demo, "--vehicle", "motorcycle")
    info = pipeline.read_json(work / "motorcycle" / "motorcycle.json")
    assert info["category"] == "4b" and info["rolling_db"] is None
    assert info["rpm"] == 4000.0 and info["cylinders"] == 2


def test_the_passby_demo_drives_the_fire_engine_with_its_siren_and_horn(
    tmp_path: Path,
) -> None:
    """--vehicle fire_engine on the quick street: the truck's sound (CNOSSOS-EU's medium
    heavy vehicle), the two tones and the horn's blasts, the scene with its three
    sources alone at their sound powers, the siren and the horn facing the way it drives
    with their directivity tables beside the scene, and its pass, the three heard
    together, each at its own sound power: the horn's sound as far under the siren's as
    its level; --sources fire_engine_siren,fire_engine_horn leaves out the truck."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import horn, load
    from acoustic_spaces.parameters import A_WEIGHTING_DB
    from acoustic_spaces.synthesis import split_bands

    folder = _street(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["all", "--vehicle", "fire_engine", "--step", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None
    parts = [siren.FIRE_ENGINE, siren.FIRE_SIREN, siren.FIRE_HORN]
    assert demo.moving(args) == parts
    assert demo.plays(args) == {
        "fire_engine": "a fire engine at 50 km/h, made to CNOSSOS-EU's sound power",
        "fire_engine_siren": "its siren on its cab's roof, two-tone",
        "fire_engine_horn": "its air horn 1.2 m up, blasts",
    }
    assert demo.main(run) == 0
    work = run_of(demo, *run[1:])
    info = pipeline.read_json(work / "fire_engine" / "fire_engine.json")
    assert info["parts"] == parts and info["siren"] == "two_tone"
    assert info["horn"] == "blasts" and info["vehicle"]["category"] == "2"
    assert info["siren_sound_power_a_db"] == pytest.approx(134.6, abs=0.1)
    assert info["horn_sound_power_a_db"] == pytest.approx(122.7, abs=0.1)
    for name in ("fire_engine.wav", "siren.wav", horn.HORN_FILE):
        x, rate = audio.read(work / "fire_engine" / name)
        assert rate == 16_000 and len(x) == pytest.approx(40.0 / 13.89 * rate, abs=2)
    scene = Scene.from_yaml(work / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == parts
    sources = {s.name: s for s in scene.sources}
    assert sources["fire_engine"].level_db == info["vehicle"]["level_db"]
    for name, key in ((siren.FIRE_SIREN, "siren"), (siren.FIRE_HORN, "horn")):
        assert sources[name].level_db == info[f"{key}_level_db"]
        assert sources[name].trajectory.face_forward and sources[name].trajectory.doppler
        assert sources[name].directivity.file == work / f"scene_demo_{name}.csv"
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert facts["parts"] == parts and facts["horn"] == "blasts"
    assert facts["about"].startswith("A fire engine at 50 km/h, its siren on, two-tone; its")
    heard = facts["listeners"]["pavement"]
    energy = sum(10.0 ** (p["sel_a_db"] / 10.0) for p in heard["parts"].values())
    assert heard["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
    mix, _ = audio.read(mix_at(work, "pavement"))
    every = mix / seats["gain"]
    # The siren and the horn alone, then the siren alone: the horn's sound is what the
    # second leaves out of the first.
    mixes = {}
    picks = {"siren_and_horn": f"{siren.FIRE_SIREN},{siren.FIRE_HORN}", "siren": siren.FIRE_SIREN}
    for plays, pick in picks.items():
        again = [*run, "--sources", pick]
        args = demo.resolved(demo.parser().parse_args(again))
        assert not demo.is_done(demo.step("sound"), args)
        assert demo.main(again) == 0
        work = run_of(demo, *again[1:])
        scene = Scene.from_yaml(work / "scene_demo.yaml")
        assert [s.name for s in scene.sources if s.in_use] == list(demo.moving(args))
        seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
        mix, _ = audio.read(mix_at(work, "pavement"))
        mixes[plays] = mix / seats["gain"]
    assert demo.moving(args) == [siren.FIRE_SIREN]
    n = min(len(every), *(len(m) for m in mixes.values()))
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)

    def a_energy(x: np.ndarray) -> float:
        return sum(
            float(weights @ np.sum(split_bands(x[:, k], 16_000) ** 2, axis=1))
            for k in range(x.shape[1])
        )

    alone = mixes["siren"][:n]
    under = 10.0 * math.log10(a_energy(alone) / a_energy(mixes["siren_and_horn"][:n] - alone))
    told = heard["parts"]["fire_engine_siren"]["sel_a_db"]
    told -= heard["parts"]["fire_engine_horn"]["sel_a_db"]
    assert told > 3.0 and under == pytest.approx(told, abs=1.5)
    # The truck adds its rumble under the horn's and the siren's pitches.
    truck = every[:n] - mixes["siren_and_horn"][:n]
    low = [float(np.sum(split_bands(x[:, 0], 16_000)[0] ** 2)) for x in (truck, alone)]
    assert low[0] > 10.0 * low[1]
    # And it plays at its own sound power: as far under the siren's and the horn's
    # sound as its level is under theirs.
    under = 10.0 * math.log10(a_energy(mixes["siren_and_horn"][:n]) / a_energy(truck))
    devices = sum(
        10.0 ** (heard["parts"][name]["sel_a_db"] / 10.0)
        for name in ("fire_engine_siren", "fire_engine_horn")
    )
    told = 10.0 * math.log10(devices) - heard["parts"]["fire_engine"]["sel_a_db"]
    assert told > 20.0 and under == pytest.approx(told, abs=1.5)


def test_the_passby_demo_drives_the_car_with_its_horn(tmp_path: Path) -> None:
    """--vehicle car on the quick street: the car alone by default, said as a car alone
    is; --sources car,car_horn adds its horn, facing the way it drives with its
    directivity table beside the scene, and the two are heard together, each at its own
    sound power: the horn's sound as far over the car's as its level; --sources car_horn
    plays the horn alone."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import horn, load
    from acoustic_spaces.parameters import A_WEIGHTING_DB
    from acoustic_spaces.synthesis import split_bands

    folder = _street(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["all", "--vehicle", "car", "--step", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None and demo.moving(args) == [siren.CAR]
    assert demo.plays(args) == {"car": "a car at 50 km/h, made to CNOSSOS-EU's sound power"}
    assert demo.main(run) == 0
    work = run_of(demo, *run[1:])
    scene = Scene.from_yaml(work / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == [siren.CAR]
    facts = pipeline.read_json(work / "walk" / "walk.json")
    car_db_a = facts["sound_power_a_db"]
    assert facts["about"] == f"A car at 50 km/h, {car_db_a:.1f} dB(A) of sound power (CNOSSOS-EU)"

    with_horn = [*run, "--sources", "car,car_horn"]
    args = demo.resolved(demo.parser().parse_args(with_horn))
    assert not demo.is_done(demo.step("sound"), args)
    assert demo.moving(args) == [siren.CAR, siren.CAR_HORN_PART]
    assert demo.plays(args)["car_horn"] == "its horn behind its front bumper, blasts"
    assert demo.main(with_horn) == 0
    work = run_of(demo, *with_horn[1:])
    info = pipeline.read_json(work / "car" / "car.json")
    assert info["parts"] == ["car", "car_horn"] and info["horn"] == "blasts"
    assert info["horn_level_db"] == [
        round(float(v), 1) for v in horn.horn_power(16_000, horn.CAR_HORN)
    ]
    x, rate = audio.read(work / "car" / horn.CAR_HORN.file)
    assert rate == 16_000 and len(x) == pytest.approx(80.0 / 13.89 * rate, abs=2)
    scene = Scene.from_yaml(work / "scene_demo.yaml")
    sources = {s.name: s for s in scene.sources}
    assert [s.name for s in scene.sources if s.in_use] == [siren.CAR, siren.CAR_HORN_PART]
    assert sources["car_horn"].level_db == info["horn_level_db"]
    assert sources["car_horn"].trajectory.face_forward and sources["car_horn"].trajectory.doppler
    assert sources["car_horn"].directivity.file == work / "scene_demo_car_horn.csv"
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert facts["about"].startswith("A car at 50 km/h, its horn in blasts\n")
    heard = facts["listeners"]["pavement"]
    seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
    mix, _ = audio.read(mix_at(work, "pavement"))
    both = mix / seats["gain"]

    def a_energy(x: np.ndarray) -> float:
        weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)
        return sum(
            float(weights @ np.sum(split_bands(x[:, k], 16_000) ** 2, axis=1))
            for k in range(x.shape[1])
        )

    # The horn alone, the louder of the two in both runs, plays at one gain in both: the
    # car's sound is what the horn alone leaves out of the two.
    alone = [*run, "--sources", "car_horn"]
    args = demo.resolved(demo.parser().parse_args(alone))
    assert demo.moving(args) == [siren.CAR_HORN_PART]
    assert demo.main(alone) == 0
    work = run_of(demo, *alone[1:])
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert facts["about"].startswith("A car's horn alone at 50 km/h, blasts\n")
    seats = pipeline.read_json(work / "walk" / pipeline.SEATS)
    mix, _ = audio.read(mix_at(work, "pavement"))
    honk = mix / seats["gain"]
    n = min(len(honk), len(both))
    over = 10.0 * math.log10(a_energy(honk[:n]) / a_energy(both[:n] - honk[:n]))
    told = heard["parts"]["car_horn"]["sel_a_db"] - heard["parts"]["car"]["sel_a_db"]
    assert told > 3.0 and over == pytest.approx(told, abs=1.5)


def _passby(example: str):
    """The pass-by demo of ``example`` (the airport's flight, the flyover's helicopter)."""
    from acoustic_spaces.demos import load

    script = {"airport": "flight_demo.py", "helicopter_flyover": "flyover_demo.py"}
    return load(Path(example_path(example)).parent / script.get(example, "passby_demo.py"))


@pytest.mark.parametrize(
    ("example", "name", "playing"),
    [
        ("roadside", "passby_car_binaural", None),
        ("street", "passby_car_binaural", ("car",)),
        ("railway", "passby_regional_binaural", ("train", "train_roof", "train_pantograph")),
        ("airport", "flight_departure_binaural", None),
        ("helicopter_flyover", "flyover_binaural", None),
    ],
)
def test_a_pass_by_run_by_default_is_named_and_plays_as_before(example, name, playing) -> None:
    """A run with every option at its default keeps its name and plays what it played
    before --sources: the street's car without its horn, the train at every height
    without its horn; a demo whose vehicles have one source each has no --sources."""
    demo = _passby(example)
    args = demo.resolved(demo.defaults())
    assert demo.run_name(args) == name and getattr(args, "playing", None) == playing
    assert ("sources" in {o.name for o in demo.options}) == (playing is not None)
    vehicles = {"street": siren.HONKING_CAR}
    if example == "street":
        assert vehicles[example].parts(args) == playing
        # The police car with its siren, the fire engine with its siren and its horn.
        for vehicle, parts in (
            ("police_car", ("police_car", "police_car_siren")),
            ("fire_engine", ("fire_engine", "fire_engine_siren", "fire_engine_horn")),
            ("tram", ("tram", "tram_roof")),
        ):
            each = demo.resolved(argparse.Namespace(**vars(demo.defaults()) | {"vehicle": vehicle}))
            assert each.playing == parts and demo.run_name(each) == f"passby_{vehicle}_binaural"


def test_a_pass_by_run_plays_the_sources_it_picks_alone() -> None:
    """--sources: the vehicle's sources named play, the run named after them; what the
    demo feeds, moves and says plays is those alone; its own pick, given in full, is
    the default run."""
    demo = _passby("street")

    def run(*argv: str):
        return demo.resolved(demo.parser().parse_args(["all", *argv]))

    args = run("--sources", "car,car_horn")
    assert demo.run_name(args) == "passby_car_car+car_horn_binaural"
    assert args.playing == ("car", "car_horn") and demo.moving(args) == ["car", "car_horn"]
    assert list(demo.plays(args)) == ["car", "car_horn"]
    args = run("--vehicle", "police_car", "--sources", "police_car_siren")
    assert demo.run_name(args) == "passby_police_car_police_car_siren_binaural"
    assert demo.moving(args) == ["police_car_siren"]
    assert demo.plays(args) == {"police_car_siren": "a police car's siren alone at 50 km/h, wail"}
    assert siren.POLICE_CAR_VEHICLE.played_by(args).endswith(" wail police_car_siren")
    own = run("--vehicle", "police_car", "--sources", "police_car,police_car_siren")
    assert demo.run_name(own) == "passby_police_car_binaural"
    # The run's mixes are named after the vehicle alone, as before.
    assert own.run_plays == args.run_plays == "police_car"
    args = run("--vehicle", "fire_engine", "--sources", "fire_engine_siren,fire_engine_horn")
    assert demo.plays(args) == {
        "fire_engine_siren": "a fire engine's siren at 50 km/h, two-tone",
        "fire_engine_horn": "its air horn 1.2 m up, blasts",
    }
    assert demo.check(args) is None


def test_options_that_change_nothing_for_the_sources_picked_are_left_out() -> None:
    """--siren without the police car's siren, --rpm without the car itself, --horn
    without the train's horn and --track with its horn alone change nothing: the run
    takes them at their defaults, its name and its command without them."""
    street, railway = _passby("street"), _passby("railway")

    def run(demo, *argv: str):
        return demo.resolved(demo.parser().parse_args(["all", *argv]))

    car = run(street, "--vehicle", "police_car", "--sources", "police_car", "--siren", "yelp")
    assert car.siren == "wail" and street.run_name(car) == "passby_police_car_police_car_binaural"
    assert "--siren" not in street.command("all", car)
    siren_alone = run(
        street, "--vehicle", "police_car", "--sources", "police_car_siren", "--rpm", "3000"
    )
    assert siren_alone.rpm is None and "rpm" not in street.run_name(siren_alone)
    both = run(street, "--vehicle", "police_car", "--siren", "yelp", "--rpm", "3000")
    assert street.run_name(both) == "passby_police_car_rpm-3000_siren-yelp_binaural"
    assert run(railway, "--horn", "chord").horn == "blasts"
    assert railway.run_name(run(railway, "--horn", "chord")) == "passby_regional_binaural"
    every = "train,train_roof,train_pantograph,train_horn"
    chord = run(railway, "--sources", every, "--horn", "chord")
    assert railway.run_name(chord) == "passby_regional_with-train_horn_horn-chord_binaural"
    horn = run(railway, "--sources", "train_horn", "--track", "slab")
    assert (
        horn.track == "ballast" and railway.run_name(horn) == "passby_regional_train_horn_binaural"
    )


def test_a_pass_by_run_refuses_sources_its_vehicle_has_not() -> None:
    """--sources names a source of the vehicle, of each vehicle of a series: a name that
    is none, or none of one of the vehicles', is refused before anything runs, as is
    --horn off, gone for --sources."""
    demo = _passby("street")

    def check(*argv: str) -> str | None:
        return demo.check(demo.parser().parse_args(["all", *argv]))

    assert check("--sources", "horn") == (
        "--sources horn: none of them is the car's; it has car, car_horn"
    )
    assert check("--sources", "car,siren").startswith("--sources siren: no such source")
    assert check("--vehicle", "car,police_car", "--sources", "car_horn") == (
        "passby_police_car_car_horn_binaural: --sources car_horn: none of them is the police "
        "car's; it has police_car, police_car_siren"
    )
    assert check("--vehicle", "bus", "--sources", "bus") is None
    assert check("--vehicle", "car,bus", "--sources", "car").startswith(
        "passby_bus_car_binaural: --sources car: no such source"
    )
    assert check("--vehicle", "police_car", "--sources", "police_car_siren") is None
    with pytest.raises(SystemExit):
        _passby("railway").parser().parse_args(["all", "--horn", "off"])


def test_a_vehicles_sources_in_turn_are_parts_of_one_run() -> None:
    """--sources-play in-turn: each source the vehicle plays by default a part of the
    run, alone, named after it; with --vehicle several, each vehicle's own."""
    demo = _passby("street")
    args = demo.parser().parse_args(
        ["all", "--vehicle", "fire_engine", "--sources-play", "in-turn"]
    )
    assert demo.run_folder(args).name == "passby_fire_engine_each-source_binaural"
    parts = demo.each(args)
    assert [p.playing for p in parts] == [
        ("fire_engine",),
        ("fire_engine_siren",),
        ("fire_engine_horn",),
    ]
    assert [demo.run_name(p) for p in parts] == [
        "passby_fire_engine_fire_engine_binaural",
        "passby_fire_engine_fire_engine_siren_binaural",
        "passby_fire_engine_fire_engine_horn_binaural",
    ]
    # As the demo's text says: the police car, then its siren; each one --sources
    # names (the car, then its horn); a vehicle that plays one source, an ordinary run.
    for argv, playing in (
        (["--vehicle", "police_car"], [("police_car",), ("police_car_siren",)]),
        (["--vehicle", "car", "--sources", "car,car_horn"], [("car",), ("car_horn",)]),
        (["--vehicle", "car"], [("car",)]),
    ):
        args = demo.parser().parse_args(["all", *argv, "--sources-play", "in-turn"])
        assert [p.playing for p in demo.each(args)] == playing, argv
    from acoustic_spaces.demos import sibling

    script = Path(example_path("street")).parent / "passby_demo.py"
    text = " ".join(sibling(script, "passby_demo").__doc__.split())
    assert "--sources-play in-turn` each source the vehicle plays by default alone" in text


def test_the_credit_and_the_about_line_say_the_sources_that_play() -> None:
    """The stills of a run of some of a vehicle's sources say those alone: the siren
    alone, the train's roof alone (its sound power theirs), the train's horn alone; the
    tram's credit names its bogies and roof, which it has, not a pantograph."""
    from acoustic_spaces.demos import rail

    police = siren.POLICE_CAR_VEHICLE
    facts = {"parts": ["police_car_siren"], "siren": "wail"}
    assert police.credits(facts, {}).startswith("Police car's sound, siren: wail")
    assert "tyres" not in police.credits(facts, {})
    train, tram = rail.TRAIN_VEHICLE, rail.TRAM_VEHICLE
    facts = {"speed_kmh": 160.0, "train": "regional"}
    assert "noise at each bogie, roof and pantograph" in train.credits(facts, {})
    assert "noise at each roof with" in train.credits(facts | {"parts": ["train_roof"]}, {})
    assert "noise at each bogie and roof with" in tram.credits({"speed_kmh": 50.0}, {})
    horn = train.credits(facts | {"parts": ["train_horn"], "horn": "blasts"}, {})
    assert horn.startswith("Train's sound made up: horn: 660 and 370 Hz in turn")
    assert "noise" not in horn and "faded" not in horn
    info = {
        "train": "regional",
        "track": "ballast",
        "length_m": 151.0,
        "speed_kmh": 160.0,
        "sound_power_a_db": 120.0,
    }
    assert train.facts(None, info)["about"].endswith("120.0 dB(A) of sound power (Schall 03)")
    roof = train.facts(None, info | {"playing": ["train_roof"]})["about"]
    assert roof.endswith("(Schall 03) at its roof alone")
    alone = info | {"playing": ["train_horn"], "horn": "chord", "horn_sound_power_a_db": 125.3}
    assert train.facts(None, alone)["about"].endswith(
        "\nits horn's chord alone in blasts 125.3 dB(A)"
    )


def test_a_trains_info_and_levels_are_of_the_sources_that_play() -> None:
    """The train's info keeps the points at every height (its wheels shake the ground
    whichever play), but its sound power and the levels its scene gives are those of the
    sources that play; another pick is another info, so its sound is made again."""
    from acoustic_spaces.demos import rail

    scene = Scene.from_yaml(Path(example_path("railway")))
    train = rail.TRAIN_VEHICLE
    base = {"train": "s_bahn", "speed": None, "track": None, "horn": "blasts"}
    every = train.info(argparse.Namespace(**base), scene, 120.0)
    roof = train.info(argparse.Namespace(**base, playing=("train_roof",)), scene, 120.0)
    assert every != roof and roof["parts"] == every["parts"]
    assert set(roof["parts"]) == {"train", "train_roof", "train_pantograph"}
    assert roof["playing"] == ["train_roof"] and list(train.levels(roof)) == ["train_roof"]
    points = rail.train_points(rail.TRAINS["s_bahn"], 120.0, "ballast")
    up = [p for p in points if p.height_m == 4.0]
    from acoustic_spaces.parameters import a_weighted

    assert roof["sound_power_a_db"] == round(a_weighted(rail.train_power_db(up)), 2)
    assert every["sound_power_a_db"] == round(a_weighted(rail.train_power_db(points)), 2)
    assert "horn" not in every
    horn = train.info(argparse.Namespace(**base, playing=("train_horn",)), scene, 120.0)
    assert list(train.levels(horn)) == ["train_horn"]
    assert horn["sound_power_a_db"] == horn["horn_sound_power_a_db"]
    police = siren.POLICE_CAR_VEHICLE
    street = Scene.from_yaml(Path(example_path("street")))
    args = argparse.Namespace(speed=None, siren=None, rpm=None, playing=("police_car_siren",))
    info = police.info(args, street, 50.0)
    assert info["vehicle"] is None and list(police.levels(info)) == ["police_car_siren"]
    assert info["sound_power_a_db"] == info["siren_sound_power_a_db"]


def test_the_listeners_order_is_the_tours() -> None:
    """--listeners b,a: the passes play in that order, whatever the scene's."""
    from acoustic_spaces.demos.passby import heard_only, stops_of

    facts = {"listeners": {"verge": {}, "meadow_25m": {}, "yard_25m": {}}}
    seats = {
        "seats": {
            name: {"length_s": 3.0, "loudness_db": -20.0, "rms_db": -20.0}
            for name in facts["listeners"]
        }
    }
    args = argparse.Namespace(listeners="yard_25m,verge")
    stops = stops_of(heard_only(args, facts, "walk"), seats, "walk")
    assert [s.key for s in stops] == ["yard_25m", "verge"]
    assert [s.at for s in stops] == [0.0, 4.0]
    every = stops_of(heard_only(argparse.Namespace(listeners=None), facts, "walk"), seats, "walk")
    assert [s.key for s in every] == ["verge", "meadow_25m", "yard_25m"]
