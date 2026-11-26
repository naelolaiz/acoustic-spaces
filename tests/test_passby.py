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
    ("example", "name", "category", "rpm", "cylinders"),
    [
        ("roadside", "car", "1", 2000.0, 4),
        ("roadside", "truck", "3", 1200.0, 6),
        ("street", "car", "1", 2000.0, 4),
        ("street", "bus", "2", 1500.0, 6),
        ("street", "motorcycle", "4b", 4000.0, 2),
        ("street", "police_car", "1", 2000.0, 4),
        ("street", "fire_engine", "2", 1500.0, 6),
    ],
)
def test_the_examples_road_vehicles_are_cnossos_at_their_speed(
    example: str, name: str, category: str, rpm: float, cylinders: int
) -> None:
    """The road vehicles of the roadside and the street, the police car and the fire
    engine themselves too: point sources 0.05 m up, left out of the example's
    simulations, with a model of CNOSSOS-EU's category, which gives them its levels at
    the speed of their trajectory, with Doppler, and the engine their demo makes."""
    scene = Scene.from_yaml(example_path(example))
    (source,) = [s for s in scene.sources if s.name == name]
    assert not source.in_use and not source.extended and source.position[2] == 0.05
    assert (
        source.trajectory.doppler
        and source.trajectory.enabled
        and source.trajectory.keyframes is None
    )
    model = source.model
    assert (model.method, model.category, model.rpm, model.cylinders) == (
        "cnossos-eu",
        category,
        rpm,
        cylinders,
    )
    power = vehicle_power(category, source.trajectory.speed_m_s * 3.6, scene.air.temperature_c)
    assert source.level_db == pytest.approx(power.total_db, abs=0.05)


def test_the_streets_sirens_and_horns_have_their_models(tmp_path: Path) -> None:
    """The street's sirens and horns: left out, each in its vehicle's group along its
    vehicle's trajectory (the same places on the road at its own height, at its speed),
    facing the way it drives with siren.csv, the measured siren's directivity as the
    module writes it; each with a model whose sound power is what the modules make: the
    police car's siren wailing, the fire engine's two tones 6 dB louder and nothing
    else, its air horn and the car's disc horns averaged over their blasts and pauses."""
    from acoustic_spaces.demos import horn
    from acoustic_spaces.directivity import write_directivity_csv

    path = Path(example_path("street"))
    scene = Scene.from_yaml(path)
    sources = {s.name: s for s in scene.sources}
    rate = 48_000
    for name, vehicle, height, model, power in (
        ("police_car_siren", "police_car", 1.5, ("siren", "wail"), siren.siren_power("wail", rate)),
        (
            "fire_engine_siren",
            "fire_engine",
            siren.FIRE_SIREN_HEIGHT_M,
            ("siren", "two_tone"),
            siren.siren_power("two_tone", rate) + siren.FIRE_SIREN_GAIN_DB,
        ),
        (
            "fire_engine_horn",
            "fire_engine",
            horn.HEIGHT_M,
            ("horn", "air_horn"),
            horn.horn_power(rate),
        ),
        (
            "car_horn",
            "car",
            horn.CAR_HEIGHT_M,
            ("horn", "disc_horns"),
            horn.horn_power(rate, horn.CAR_HORN),
        ),
    ):
        source, body = sources[name], sources[vehicle]
        assert not source.in_use and source.group == body.group and body.group
        assert source.position[2] == height and source.position[:2] == body.position[:2]
        assert source.trajectory.doppler and source.trajectory.face_forward
        assert [p[:2] for p in source.trajectory.to] == [p[:2] for p in body.trajectory.to]
        assert source.trajectory.speed_m_s == body.trajectory.speed_m_s
        kind = source.model.pattern if model[0] == "siren" else source.model.kind
        assert (source.model.method, kind) == model
        assert source.level_db == pytest.approx(power, abs=0.005)
        assert source.directivity.file == path.parent / "siren.csv"
    assert sources["fire_engine_siren"].model.patterns == ["two_tone"]
    assert sources["fire_engine_siren"].model.gain_db == siren.FIRE_SIREN_GAIN_DB
    table = write_directivity_csv(
        tmp_path / "siren.csv",
        siren.directivity(),
        siren.directivity_comment("acoustic_spaces.demos.siren"),
    )
    assert table.read_text(encoding="utf-8") == (path.parent / "siren.csv").read_text(
        encoding="utf-8"
    )


def test_the_sirens_and_horns_sound_powers_are_their_sounds() -> None:
    """The tables of the sirens' and the horns' sound power, which a scene reads without
    making their sound, are what the modules make of their sounds at 48 kHz, to the
    hundredth of a dB; a horn's model sounds the kind the module plays, and only a train
    horn sounds its notes as a chord."""
    from acoustic_spaces.demos import horn
    from acoustic_spaces.scene import HornModel, SirenModel

    for pattern, table in siren.POWER_DB.items():
        np.testing.assert_allclose(siren.siren_power(pattern, 48_000), table, atol=0.005)
        assert SirenModel(method="siren", pattern=pattern).power_db(None, 20.0) == list(table)
    assert SirenModel(method="siren", gain_db=6.0).power_db(None, 20.0) == [
        round(v + 6.0, 6) for v in siren.POWER_DB["wail"]
    ]
    for (kind, signal), table in horn.POWER_DB.items():
        np.testing.assert_allclose(
            horn.horn_power(48_000, horn.KINDS[kind, signal]), table, atol=0.005
        )
        model = HornModel(method="horn", kind=kind, signal=signal)
        assert model.power_db(None, 20.0) == list(table)
    with pytest.raises(ValueError, match="chord"):
        HornModel(method="horn", kind="air_horn", signal="chord")
    with pytest.raises(ValueError, match="two_tone"):
        SirenModel(method="siren", pattern="wail", patterns=["two_tone"])


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
    # A recording of the horn plays its blasts as loud as the made horn's.
    from acoustic_spaces.demos.models import input_gain

    (car_horn,) = [
        s for s in Scene.from_yaml(example_path("street")).sources if s.name == "car_horn"
    ]
    assert input_gain(car_horn, made, rate) == pytest.approx(1.0)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    import urban_sources

    out = urban_sources.check_car_horn(tmp_path)
    assert out["bench_db_a"] == pytest.approx(kind.on_axis_db, abs=0.15)
    assert min(out["vehicle_db_a"].values()) >= horn.VEHICLE_DB_A
    assert out["band_db_a"] >= 105.0 and out["band_over_above_db"] > 0.0


@pytest.mark.parametrize("group", ["Police car", "Fire engine", "Passenger car"])
def test_the_credit_of_a_vehicle_with_a_siren_fits_its_stills(group: str) -> None:
    """The stills' credit of the police car, of the fire engine and of the car with its
    horn, every source of theirs made, says each in a few words and fits the still's
    width in the two lines a long credit takes, not cut short; a sound that sounds all
    along fades in and out, a horn's blasts rise and fall of themselves."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    from acoustic_spaces.demos.models import made_brief, made_credits
    from acoustic_spaces.demos.pictures import FOOTER_TWO, MARGIN, SIZE, credit_lines, font
    from acoustic_spaces.demos.scene_demo import plays_line

    scene = Scene.from_yaml(example_path("street"))
    sources = [s for s in scene.sources if s.group == group]
    table = {
        "sources": {
            s.name: {
                "signal": f"{s.name}.wav",
                "made": True,
                "credit": "the whole credit",
                "brief": made_brief(s),
                **({} if s.model.steady else {"now_and_then": True}),
            }
            for s in sources
        }
    }
    credit = plays_line(table)
    horns = [s.name for s in sources if s.model.method == "horn"]
    assert credit == made_credits([(s.name, made_brief(s), s.name not in horns) for s in sources])
    assert credit.startswith(f"Sounds made up for {sources[0].name}: ")
    tail = "; faded in and out over 1 s" if len(horns) < len(sources) else ""
    if horns and tail:
        tail += ", not " + " and ".join(horns)
    assert credit.endswith(tail)
    draw = ImageDraw.Draw(Image.new("RGB", SIZE))
    width = SIZE[0] - 2 * MARGIN
    lines = credit_lines(draw, credit, width)
    assert 1 <= len(lines) <= 2
    assert all(draw.textlength(line, font=font("Regular", FOOTER_TWO)) <= width for line in lines)
    assert " ".join(lines).split() == credit.split()
    # One made sound alone is credited in full.
    table["sources"] = {sources[0].name: table["sources"][sources[0].name]}
    assert plays_line(table) == "the whole credit"


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


def test_the_model_options_matter_with_the_sources_that_have_them() -> None:
    """The options a model leaves to a run (--rpm a road vehicle's engine, --siren a
    siren's pattern, --horn a train horn's notes) matter for the street's sources that
    have one: the run sets them in those alone, its name says them once, each part of a
    series where they change something; a pick with none of them is refused, said as
    which kind of made sound is missing, and a siren that sounds one pattern keeps it."""
    demo = _passby("street")

    def run(*argv: str):
        return demo.resolved(demo.parser().parse_args(["all", *argv]))

    both = run("--vehicle", "police_car", "--siren", "yelp", "--rpm", "3000")
    assert demo.run_name(both) == "passby_Police-car_rpm-3000_siren-yelp_binaural"
    plays = demo.plays(both)
    assert "3000 rpm" in plays["police_car"] and plays["police_car_siren"].startswith(
        "a siren's yelp"
    )
    assert demo.check(run("--sources", "police_car", "--siren", "yelp")) == (
        "--siren changes nothing: no siren plays a made sound"
    )
    assert demo.check(run("--sources", "police_car_siren", "--rpm", "3000")) == (
        "--rpm changes nothing: no road vehicle plays a made sound"
    )
    assert demo.check(run("--sources", "Fire engine", "--horn", "chord")) == (
        "--horn changes nothing: no train horn plays a made sound"
    )
    # The fire engine's siren sounds its two tones whatever --siren says.
    fire = run("--vehicle", "fire_engine,police_car", "--siren", "wail")
    assert demo.check(fire) is None
    assert [demo.run_name(p) for p in fire.series_parts] == [
        "passby_Fire-engine_binaural",
        "passby_Police-car_siren-wail_binaural",
    ]
    assert demo.plays(fire.series_parts[0])["fire_engine_siren"].startswith("a siren's two-tone")


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
    silent in its pauses."""
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
    """The roadside's demo drives the car its scene gives a CNOSSOS-EU model: a car whose
    trajectory is off, or that --moving leaves standing, keyframes at more than one speed
    (the scene does not read: no --speed rescues it), a speed that is not positive and a
    scene with no car to take in are refused before anything runs; by default the car
    goes as fast as its trajectory does, and its sound is made to the model's power at
    that speed."""
    from acoustic_spaces.demos import load

    folder = _roadside(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    args = demo.defaults()
    assert args.speed is None and args.sources == "car"  # as the scene's trajectory goes
    assert demo.check(args) is None and demo.moving(demo.resolved(args)) == ["car"]
    assert demo.plays(demo.resolved(args))["car"] == (
        "CNOSSOS-EU light motor vehicles at 50 km/h, made to the method's sound power (tyres "
        "and an engine of 4 cylinders at 2000 rpm)"
    )
    args.speed = -3.0
    assert demo.check(args) == "--speed is a positive number, not -3.0"
    args.speed = None
    needs = "CNOSSOS-EU gives a road vehicle's sound power as it drives, at one steady speed"
    standing = demo.parser().parse_args(["all", "--moving", "none"])
    assert demo.check(standing) == f"--moving leaves car standing, but {needs}: name it too"
    path = folder / "scene.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    (car,) = [s for s in data["sources"] if s["name"] == "car"]
    car["trajectory"]["enabled"] = False
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(args) == (
        f"car's trajectory is off in scene.yaml, but {needs}: turn it on (Moving on its trajectory)"
    )
    assert demo.moving(demo.resolved(args)) == []
    # Keyframes at one steady speed drive as they are timed: 40 m in 3 s.
    car["trajectory"] = {
        "keyframes": [{"time_s": 3.0, "position": [-20, 2, 0.05]}],
        "doppler": True,
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(args) is None
    assert "light motor vehicles at 48 km/h" in demo.plays(demo.resolved(args))["car"]
    # Keyframes at two speeds: CNOSSOS-EU has a car's power at one, and the scene says so.
    car["trajectory"]["keyframes"] = [
        {"time_s": 1.0, "position": [0, 2, 0.05]},
        {"time_s": 3.0, "position": [0, 2, 0.05]},  # it waits
        {"time_s": 5.0, "position": [-20, 2, 0.05]},
    ]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    refused = (
        f"scene.yaml: source car: its keyframes go at 0.0 to 72.0 km/h, but {needs}: time them "
        "for one speed, or give the trajectory points at a speed"
    )
    assert demo.check(args) == refused
    assert demo.check(demo.parser().parse_args(["all", "--speed", "36"])) == refused
    data["sources"] = [s for s in data["sources"] if s["name"] != "car"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert demo.check(args) == (
        "no source in use plays anything; take in some of those the scene leaves out "
        "(--sources truck)"
    )


def test_the_passby_demo_drives_keyframes_as_timed_or_at_speed(tmp_path: Path) -> None:
    """The roadside's demo keeps a car's steady keyframes as they are, with Doppler and
    CNOSSOS-EU's power at their speed; with --speed it drives through their places (each
    once) at that speed instead, at the power the model gives there, and the editor
    follows the car along the trajectory the run simulates."""
    from acoustic_spaces.demos import load

    folder = _roadside(tmp_path)
    path = folder / "scene.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    (car,) = [s for s in data["sources"] if s["name"] == "car"]
    car["trajectory"] = {
        "keyframes": [
            {"time_s": 1.5, "position": [0, 2, 0.05]},
            {"time_s": 3.0, "position": [-20, 2, 0.05], "orientation": {"yaw": 180}},
        ],
        "doppler": True,
        "enabled": True,
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    demo = load(folder / "passby_demo.py").at(folder)
    args = demo.resolved(demo.defaults())
    scene = demo.trajectories(args)
    (driven,) = [s for s in scene.sources if s.name == "car"]
    assert driven.in_use and driven.trajectory.doppler and len(driven.trajectory.keyframes) == 2
    power = vehicle_power("1", 48.0, scene.air.temperature_c)
    assert driven.level_db == pytest.approx(power.total_db, abs=0.05)
    fast = demo.resolved(demo.parser().parse_args(["all", "--speed", "72"]))
    scene = demo.trajectories(fast)
    (driven,) = [s for s in scene.sources if s.name == "car"]
    assert driven.trajectory.keyframes is None
    assert driven.trajectory.speed_m_s == pytest.approx(20.0) and driven.trajectory.doppler
    assert driven.trajectory.to == [(0.0, 2.0, 0.05), (-20.0, 2.0, 0.05)]
    power = vehicle_power("1", 72.0, scene.air.temperature_c)
    assert driven.level_db == pytest.approx(power.total_db, abs=0.05)
    # Another speed is another run, with its own sound and mixes.
    assert demo.run_name(fast) == "passby_speed-72_binaural"
    assert demo.run_name(args) == "passby_binaural"


def test_the_passby_demo_drives_the_car_past_every_listener(tmp_path: Path, capsys) -> None:
    """Every step on the quick roadside: the car's sound made to its model's power, the
    scene with the car taken in, the pass at --speed every --step (the level along it at
    each listener, its maximum near the middle, the pitch falling, one gain for all), the
    tour and its stills; then each step is done, and a run at one listener is a run of its
    own."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load
    from acoustic_spaces.demos.models import FADE_S

    folder = _roadside(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["--step", "4", "--speed", "72", "--no-mp4"]
    assert demo.main(["all", *run]) == 0
    work = run_of(demo, *run)
    assert work == folder / "demo" / "passby_speed-72_step-4_binaural"
    # The car's sound: its pass of 2 s, faded in and out, at the feeds' RMS where it sounds.
    fed = work.parent / "feeds" / "scene_with-car_speed-72"  # the car taken in
    made = pipeline.read_json(fed / "feeds.json")["sources"]["car"]
    assert made["made"] and "four-cylinder engine at 2000 rpm" in made["credit"]
    x, rate = audio.read(fed / "car.wav")
    assert rate == 16_000 and len(x) == 2 * rate
    # Faded in over its first FADE_S (all of its first half): quiet at first.
    fade = int(FADE_S * rate)
    assert np.abs(x[: fade // 8]).max() < np.abs(x[fade - fade // 8 : fade]).max() / 4
    seats = pipeline.read_json(work / "moving" / pipeline.SEATS)
    passes = seats["movement"]["passes"]
    assert set(passes) == {"verge", "yard_25m"}
    verge = passes["verge"]
    assert verge["closest_m"] == pytest.approx(math.hypot(5.5, 1.45), abs=0.01)
    assert 0.8 < verge["max_s"] < 1.4
    assert verge["max_a_db"] == pytest.approx(max(verge["levels_a_db"]), abs=0.005)
    assert verge["sel_a_db"] > verge["leq_a_db"]  # over 2 s: SEL = Leq + 3 dB
    assert 1.0 < verge["pitch_semitones"] < 12 * math.log2(363.0 / 323.0)
    assert verge["about"] == "Sound power of car at 72 km/h: " + (
        f"{vehicle_power('1', 72.0, 20.0).sound_power_a_db:.1f} dB(A) (CNOSSOS-EU)"
    )
    peaks = [
        np.abs(audio.read(work / "moving" / s["file"])[0]).max() for s in seats["seats"].values()
    ]
    assert max(peaks) == pytest.approx(10.0 ** (-1.0 / 20.0), rel=1e-3)
    assert sorted(p.name for p in (work / "stills").glob("*.png")) == [
        "00_verge.png",
        "01_yard_25m.png",
    ]
    for product in ("_thumbnail.png", ".wav"):
        assert work.with_name(work.name + product).exists()
    chapters = work.with_name(work.name + "_chapters.txt").read_text(encoding="utf-8")
    assert chapters.splitlines()[0] == "0:00 verge"
    assert chapters.splitlines()[1].endswith(" yard 25m")
    capsys.readouterr()
    assert demo.main(["all", *run]) == 0
    out = capsys.readouterr().out
    for step in ("feeds", "simulate"):
        assert f"{step}: done already" in out
    assert f"{work / 'moving'}: there already" in out  # the pass is not mixed again
    # Another speed is another run, with its own sound and mixes.
    args = demo.resolved(demo.parser().parse_args(["all", "--step", "4"]))
    assert not demo.is_done(demo.step("feeds"), args)
    assert not demo.is_done(demo.step("render"), args)
    # --listeners: a run of its own, at that listener alone, in a folder of its own.
    assert demo.main(["all", *run, "--listeners", "yard_25m"]) == 0
    yard = run_of(demo, *run, "--listeners", "yard_25m")
    assert yard.name == "passby_yard_25m_speed-72_step-4_binaural"
    chapters = yard.with_name(yard.name + "_chapters.txt").read_text(encoding="utf-8")
    assert chapters.splitlines() == ["0:00 yard 25m"]
    assert sorted(p.name for p in (yard / "stills").glob("*.png")) == ["00_yard_25m.png"]
    assert sorted(p.name for p in (work / "stills").glob("*.png")) == [
        "00_verge.png",
        "01_yard_25m.png",
    ]


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


def _parts(demo, *argv: str) -> dict[str, argparse.Namespace]:
    """Each part of a run of ``demo`` with ``argv`` (resolved), by the sources it plays."""
    args = demo.resolved(demo.parser().parse_args(["all", *argv]))
    return {p.sources: p for p in getattr(args, "series_parts", None) or [args]}


def _heard(part: argparse.Namespace, listener: str = "pavement") -> tuple[np.ndarray, dict]:
    """The mix of a run (``part``, resolved) of the demo of a scene at ``listener``, on
    one scale for every run: as the render played it, before the one gain that brings
    its loudest peak to -1 dBFS, with the loudest source that plays at its sound power
    (a run plays each source by its sound power against the loudest one's); and that
    listener's pass."""
    from acoustic_spaces.demos import run_folder
    from acoustic_spaces.demos.scene_demo import run_scene_file

    work = run_folder(part)
    seats = pipeline.read_json(work / "moving" / pipeline.SEATS)
    x, _rate = audio.read(work / "moving" / seats["seats"][listener]["file"])
    powers = {s.name: s.sound_power_db for s in Scene.from_yaml(run_scene_file(part)).sources}
    loudest = max(powers[name] for name in seats["moving"])
    return x / seats["gain"] * 10.0 ** (loudest / 20.0), seats["movement"]["passes"][listener]


def _a_energy(x: np.ndarray, rate: int) -> float:
    """The A-weighted energy of ``x`` (samples, channels), by octave band."""
    from acoustic_spaces.parameters import A_WEIGHTING_DB
    from acoustic_spaces.synthesis import split_bands

    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)
    return sum(
        float(weights @ np.sum(split_bands(x[:, k], rate) ** 2, axis=1)) for k in range(x.shape[1])
    )


def test_the_street_demo_drives_the_police_car_with_its_siren(tmp_path: Path) -> None:
    """--vehicle police_car --siren two_tone on the quick street: the car's sound
    (CNOSSOS-EU's light motor vehicle) and the siren's two tones made along their pass,
    the scene the run plays with the police car's two sources taken in at their models'
    sound powers, the siren facing the way it drives with siren.csv, both with Doppler;
    with the siren alone and the car alone in turn in the same run, the pass of the two
    together is their energies added, and their audio is as far apart as their levels:
    the car's sound some 29 dB(A) under the siren's, and all of the police car's below
    the siren's pitch, in the octaves of 63 and 125 Hz."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load, run_folder
    from acoustic_spaces.demos.scene_demo import MOVING, feeds_folder, moving_done
    from acoustic_spaces.synthesis import split_bands

    folder = _street(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["--siren", "two_tone", "--step", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(["all", "--vehicle", "police_car", *run]))
    assert demo.check(args) is None
    assert demo.moving(args) == ["police_car", "police_car_siren"]
    assert demo.plays(args)["police_car_siren"] == (
        "a siren's two-tone, made to the sound power of the 100 W electronic siren NBS "
        "measured (SP 480-28, 1978)"
    )
    picks = ["--sources", "Police car,police_car_siren,police_car", *run]
    assert demo.main(["all", *picks]) == 0
    works = _parts(demo, *picks)
    assert [run_folder(w).name for w in works.values()] == [
        "passby_Police-car_step-4_siren-two_tone_binaural",
        "passby_police_car_siren_step-4_siren-two_tone_binaural",
        "passby_police_car_step-4_binaural",
    ]
    # Each part keeps its feeds, its simulation and its mixes: a run again finds them
    # there, each part's feeds kept by what it takes in (the police car's, the car's).
    for part in works.values():
        assert all(demo.is_done(demo.step(s), part) for s in ("feeds", "simulate"))
        assert moving_done(part, run_folder(part) / MOVING, demo.moving(part), ["pavement"])
    feeds = folder / "demo" / "feeds" / "scene_with-police_car+police_car_siren_siren-two_tone"
    assert feeds_folder(works["Police car"]) == feeds
    assert feeds_folder(works["police_car"]) == folder / "demo" / "feeds" / "scene_with-police_car"
    fed = pipeline.read_json(feeds / "feeds.json")
    assert set(fed["sources"]) == {"police_car", "police_car_siren"}
    assert all(v["made"] for v in fed["sources"].values())
    assert fed["sources"]["police_car_siren"]["credit"].startswith(
        "Sound made up for police_car_siren: a siren's two-tone, a square wave"
    )
    for name in ("police_car", "police_car_siren"):
        x, rate = audio.read(feeds / f"{name}.wav")
        assert rate == 16_000 and len(x) == pytest.approx(40.0 / 13.89 * rate, abs=2)
    scene = Scene.from_yaml(
        folder / "demo" / "scenes" / "scene_with-police_car+police_car_siren_siren-two_tone.yaml"
    )
    sources = {s.name: s for s in scene.sources}
    # Taken in beside the scene's own traffic, silent here.
    assert [s.name for s in scene.sources if s.in_use] == [
        "traffic",
        "police_car",
        "police_car_siren",
    ]
    top = sources["police_car_siren"]
    assert top.level_db == pytest.approx(list(siren.POWER_DB["two_tone"]))
    assert sources["police_car"].level_db == pytest.approx(
        vehicle_power("1", 13.89 * 3.6, scene.air.temperature_c).total_db, abs=0.05
    )
    assert top.trajectory.face_forward and top.trajectory.doppler
    assert top.directivity.file == folder / "siren.csv"
    heard = {sources: _heard(work) for sources, work in works.items()}
    both, siren_alone, car = (heard[k] for k in ("Police car", "police_car_siren", "police_car"))
    assert both[1]["about"] == (
        f"Sound power of police_car at 50 km/h: {vehicle_power('1', 50.004).sound_power_a_db:.1f}"
        " dB(A) (CNOSSOS-EU)\nSound power of police_car_siren: 128.6 dB(A) (a siren, two-tone)"
    )
    energy = 10.0 ** (siren_alone[1]["sel_a_db"] / 10.0) + 10.0 ** (car[1]["sel_a_db"] / 10.0)
    assert both[1]["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    told = siren_alone[1]["sel_a_db"] - car[1]["sel_a_db"]
    n = min(len(x) for x, _ in heard.values())
    under = 10.0 * math.log10(_a_energy(siren_alone[0][:n], 16_000) / _a_energy(car[0][:n], 16_000))
    assert told > 25.0 and under == pytest.approx(told, abs=1.5)
    # At 63 and 125 Hz the siren sends next to nothing: more than 95 % of the police
    # car's sound there is the car's.
    for band in (0, 1):
        low = {
            key: float(np.sum(split_bands(x[:n, 0], 16_000)[band] ** 2))
            for key, (x, _) in heard.items()
        }
        assert low["police_car"] > 0.95 * low["Police car"] > 20.0 * low["police_car_siren"]
    # Another pattern is another sound and another scene.
    wail = demo.resolved(
        demo.parser().parse_args(["all", "--vehicle", "police_car", "--step", "4"])
    )
    assert not demo.is_done(demo.step("feeds"), wail)
    assert not demo.is_done(demo.step("simulate"), wail)
    # The motorcycle: its engine alone, two cylinders at 4000 rpm.
    assert demo.main(["feeds", "--vehicle", "motorcycle"]) == 0
    fed = pipeline.read_json(folder / "demo" / "feeds" / "scene_with-motorcycle" / "feeds.json")
    assert fed["sources"]["motorcycle"]["credit"].startswith(
        "Sound made up for motorcycle: a two-cylinder engine at 4000 rpm"
    )


def test_the_street_demo_drives_the_fire_engine_with_its_siren_and_horn(
    tmp_path: Path,
) -> None:
    """--vehicle fire_engine on the quick street: the truck's sound (CNOSSOS-EU's medium
    heavy vehicle), the two tones and the air horn's blasts, its three sources taken in
    at their models' sound powers, the siren and the horn facing the way it drives with
    siren.csv; with the siren and the horn, the siren alone and the horn alone in turn in
    the same run: the pass of two is their energies added, the horn heard at each place
    as loud as its blasts play there, and their audio as far apart as their levels; the
    truck adds its rumble under the horn's and the siren's pitches."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import horn, load
    from acoustic_spaces.demos.scene_demo import feeds_folder, run_scene_file
    from acoustic_spaces.synthesis import split_bands

    folder = _street(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["--step", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(["all", "--vehicle", "fire_engine", *run]))
    assert demo.check(args) is None
    parts = ["fire_engine", "fire_engine_siren", "fire_engine_horn"]
    assert demo.moving(args) == parts
    picks = [
        "--sources",
        "Fire engine,fire_engine_siren+fire_engine_horn,fire_engine_siren,fire_engine_horn",
        *run,
    ]
    assert demo.main(["all", *picks]) == 0
    works = _parts(demo, *picks)
    fed = pipeline.read_json(feeds_folder(works["Fire engine"]) / "feeds.json")["sources"]
    assert fed["fire_engine_horn"]["now_and_then"] and "now_and_then" not in fed["fire_engine"]
    # Its three names are too long for the scene's file name, which counts them.
    taken_in = run_scene_file(works["Fire engine"])
    assert taken_in.parent == folder / "demo" / "scenes"
    assert taken_in.name.startswith("scene_with-3-left-out-")
    scene = Scene.from_yaml(taken_in)
    sources = {s.name: s for s in scene.sources}
    assert [s.name for s in scene.sources if s.in_use] == ["traffic", *parts]
    assert sources["fire_engine_siren"].level_db == pytest.approx(
        [v + siren.FIRE_SIREN_GAIN_DB for v in siren.POWER_DB["two_tone"]]
    )
    assert sources["fire_engine_horn"].level_db == pytest.approx(
        list(horn.POWER_DB["air_horn", "blasts"])
    )
    for name in parts[1:]:
        assert sources[name].trajectory.face_forward and sources[name].trajectory.doppler
        assert sources[name].directivity.file == folder / "siren.csv"
    heard = {sources: _heard(work) for sources, work in works.items()}
    every, devices, alone, honk = (
        heard[k]
        for k in (
            "Fire engine",
            "fire_engine_siren+fire_engine_horn",
            "fire_engine_siren",
            "fire_engine_horn",
        )
    )
    energy = 10.0 ** (alone[1]["sel_a_db"] / 10.0) + 10.0 ** (honk[1]["sel_a_db"] / 10.0)
    assert devices[1]["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    n = min(len(x) for x, _ in heard.values())
    told = alone[1]["sel_a_db"] - honk[1]["sel_a_db"]
    under = 10.0 * math.log10(_a_energy(alone[0][:n], 16_000) / _a_energy(honk[0][:n], 16_000))
    assert told > 3.0 and under == pytest.approx(told, abs=1.5)
    # The truck adds its rumble under the horn's and the siren's pitches.
    low = [float(np.sum(split_bands(x[:n, 0], 16_000)[0] ** 2)) for x, _ in (every, devices)]
    assert low[0] > 10.0 * low[1]


def test_the_street_demo_drives_the_car_with_its_horn(tmp_path: Path) -> None:
    """On the quick street the car plays alone by default, said as a car alone is;
    --car car_and_horn (the group Passenger car) adds its pair of disc horns, facing the
    way it drives with siren.csv; with the horn alone and the car alone in turn in the
    same run, the pass of the two is their energies added, the horn heard at each place
    as loud as its blasts play there, its audio as far over the car's as its level."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import horn, load

    folder = _street(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["--step", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(["all", *run]))
    assert demo.check(args) is None and demo.moving(args) == ["car"]
    honking = demo.resolved(demo.parser().parse_args(["all", "--car", "car_and_horn", *run]))
    assert demo.moving(honking) == ["car", "car_horn"]
    assert demo.plays(honking)["car_horn"] == (
        "a pair of disc horns' blasts, made to 110 dB(A) 2 m in front of them (Hella's S77, "
        "approved to UN Regulation 28)"
    )
    picks = ["--sources", "Passenger car,car_horn,car", *run]
    assert demo.main(["all", *picks]) == 0
    works = _parts(demo, *picks)
    scene = Scene.from_yaml(folder / "demo" / "scenes" / "scene_with-car+car_horn.yaml")
    sources = {s.name: s for s in scene.sources}
    assert sources["car_horn"].level_db == pytest.approx(
        list(horn.POWER_DB["disc_horns", "blasts"])
    )
    assert sources["car_horn"].directivity.file == folder / "siren.csv"
    x, rate = audio.read(folder / "demo" / "feeds" / "scene_with-car+car_horn" / "car_horn.wav")
    assert rate == 16_000 and len(x) == pytest.approx(80.0 / 13.89 * rate, abs=2)
    heard = {sources: _heard(work) for sources, work in works.items()}
    both, honk, car = (heard[k] for k in ("Passenger car", "car_horn", "car"))
    assert car[1]["about"] == (
        f"Sound power of car at 50 km/h: {vehicle_power('1', 50.004).sound_power_a_db:.1f} "
        "dB(A) (CNOSSOS-EU)"
    )
    energy = 10.0 ** (honk[1]["sel_a_db"] / 10.0) + 10.0 ** (car[1]["sel_a_db"] / 10.0)
    assert both[1]["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    n = min(len(x) for x, _ in heard.values())
    told = honk[1]["sel_a_db"] - car[1]["sel_a_db"]
    over = 10.0 * math.log10(_a_energy(honk[0][:n], 16_000) / _a_energy(car[0][:n], 16_000))
    assert told > 3.0 and over == pytest.approx(told, abs=1.5)


def _passby(example: str):
    """The pass-by demo of ``example`` (the airport's flight, the flyover's helicopter)."""
    from acoustic_spaces.demos import load

    script = {"airport": "flight_demo.py", "helicopter_flyover": "flyover_demo.py"}
    return load(Path(example_path(example)).parent / script.get(example, "passby_demo.py"))


@pytest.mark.parametrize(
    ("example", "name", "playing"),
    [
        ("airport", "flight_departure_binaural", None),
        ("helicopter_flyover", "flyover_binaural", None),
    ],
)
def test_a_pass_by_run_by_default_is_named_and_plays_as_before(example, name, playing) -> None:
    """A run with every option at its default keeps its name and plays what it played
    before --sources: the train at every height without its horn; a demo whose vehicles
    have one source each has no --sources."""
    demo = _passby(example)
    args = demo.resolved(demo.defaults())
    assert demo.run_name(args) == name and getattr(args, "playing", None) == playing
    assert ("sources" in {o.name for o in demo.options}) == (playing is not None)


def test_a_street_run_plays_the_sources_it_picks() -> None:
    """--sources on the street: the sources named play, the run named after them; what
    the demo moves and says plays is those alone; names joined with + play together in
    one part, as a group does, and the demo's own pick given again is the default run."""
    demo = _passby("street")

    def run(*argv: str):
        return demo.resolved(demo.parser().parse_args(["all", *argv]))

    args = run("--sources", "car+car_horn")
    assert demo.run_name(args) == "passby_car+car_horn_binaural"
    assert demo.moving(args) == ["car", "car_horn"]
    assert [k for k in demo.plays(args) if k != "traffic"] == ["car", "car_horn"]
    group = run("--sources", "Passenger car")
    assert demo.moving(group) == demo.moving(args)
    siren_alone = run("--sources", "police_car_siren")
    assert demo.run_name(siren_alone) == "passby_police_car_siren_binaural"
    assert demo.moving(siren_alone) == ["police_car_siren"]
    assert demo.run_name(run("--sources", "car")) == "passby_binaural"
    two = run("--sources", "fire_engine_siren+fire_engine_horn")
    assert demo.check(two) is None and demo.moving(two) == ["fire_engine_siren", "fire_engine_horn"]


def test_options_that_change_nothing_for_the_sources_picked_are_refused() -> None:
    """--horn without the train's horn and --track with its horn alone change nothing:
    a run of one part is refused, naming the option; a part of several where it changes
    nothing takes it at its default, its name without it."""
    railway = _passby("railway")

    def run(*argv: str):
        return railway.parser().parse_args(["all", *argv])

    assert railway.check(run("--horn", "chord")) == (
        "--horn changes nothing: no train horn plays a made sound"
    )
    assert railway.check(run("--sources", "train_horn", "--track", "slab")) == (
        "--track changes nothing: no train or tram plays (a source with a schall-03 model)"
    )
    chord = railway.resolved(run("--sources", "Train", "--horn", "chord"))
    assert railway.check(chord) is None
    assert railway.run_name(chord) == "passby_Train_horn-chord_binaural"
    turns = run("--sources", "train,train_horn", "--sources-play", "in-turn", "--track", "slab")
    assert railway.check(turns) is None
    parts = [
        (p.sources, p.track, railway.run_name(railway.resolved(p))) for p in railway.parts(turns)
    ]
    assert parts == [
        ("train", "slab", "passby_train_track-slab_binaural"),
        ("train_horn", None, "passby_train_horn_binaural"),
    ]


def test_a_pass_by_run_refuses_sources_its_scene_has_not() -> None:
    """--sources names a source or a group of the street, each name a + joins too: a name
    that is none is refused before anything runs, with what the street has; the
    railway's --horn off is gone for --sources."""
    demo = _passby("street")

    def check(*argv: str) -> str | None:
        return demo.check(demo.parser().parse_args(["all", *argv]))

    have = (
        "traffic, car, car_horn, bus, motorcycle, police_car, police_car_siren, fire_engine, "
        "fire_engine_siren, fire_engine_horn, tram, tram_roof, breaker, generator, dumper, "
        "dumper_alarm, Fire engine, Passenger car, Police car, Road works, Tram"
    )
    assert (
        check("--sources", "horn") == f"--sources horn: no such source or group here; it has {have}"
    )
    assert check("--sources", "car+siren") == (
        f"--sources siren: no such source or group here; it has {have}"
    )
    assert check("--sources", "bus") is None
    assert check("--sources", "Police car,car_horn") is None
    with pytest.raises(SystemExit):
        _passby("railway").parser().parse_args(["all", "--horn", "off"])


def test_a_vehicles_sources_in_turn_are_parts_of_one_run() -> None:
    """On the street each name or group --sources gives is a part of the run, one after
    the other (the default --sources-play in-turn): the fire engine's sources each
    alone, named after it, when it names them; the police car, then its siren alone;
    a vehicle that plays one source, an ordinary run."""
    from acoustic_spaces.demos import run_folder

    demo = _passby("street")
    args = demo.parser().parse_args(
        ["all", "--sources", "fire_engine,fire_engine_siren,fire_engine_horn"]
    )
    assert run_folder(demo.resolved(args)).name == (
        "passby_each-fire_engine+fire_engine_siren+fire_engine_horn_binaural"
    )
    parts = demo.each(args)
    assert [p.sources for p in parts] == ["fire_engine", "fire_engine_siren", "fire_engine_horn"]
    assert [demo.run_name(p) for p in parts] == [
        "passby_fire_engine_binaural",
        "passby_fire_engine_siren_binaural",
        "passby_fire_engine_horn_binaural",
    ]
    for argv, playing in (
        (["--sources", "Police car,police_car_siren"], ["Police car", "police_car_siren"]),
        (["--vehicle", "police_car"], ["Police car"]),
        (["--vehicle", "car"], ["car"]),
    ):
        args = demo.parser().parse_args(["all", *argv])
        assert [p.sources for p in demo.each(args)] == playing, argv


def test_the_credit_and_the_about_line_say_the_sources_that_play() -> None:
    """The stills of a run of some of a vehicle's sources say those alone: the train's
    roof alone (its sound power theirs), the train's horn alone; the tram's credit names
    its bogies and roof, which it has, not a pantograph."""
    from acoustic_spaces.demos import rail

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
