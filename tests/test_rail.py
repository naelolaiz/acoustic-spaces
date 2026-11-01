"""The trains and trams of the pass-by demos (:mod:`acoustic_spaces.demos.rail`):
Schall 03's data sheets, its equation 1, the sound power of a moving unit and its
directivity, a train's points and their sound."""

import argparse
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.demos import rail
from acoustic_spaces.examples import example_path
from acoustic_spaces.parameters import A_WEIGHTING_DB

RATE = 16_000


def test_each_rows_shares_add_up_to_its_level() -> None:
    """The data sheets' octave shares Δa_f are whole dB of their row's a_A: they add up
    to it within half a dB (Schall 03 rounds them), and each source is one Tabelle 5 or
    13 names, its rolling noise for each kind of brakes."""
    for category in rail.CATEGORIES.values():
        sources = rail.TRAM_SOURCES if category.tram else rail.SOURCES
        for brakes in category.rolling:
            ms = [row.m for row in category.rows(brakes)]
            assert len(ms) == len(set(ms)) and set(ms) <= set(sources), category.key
            assert {1, 2} <= set(ms), category.key
            for row in category.rows(brakes):
                assert len(row.shares_db) == 8
                total = 10.0 * math.log10(sum(10.0 ** (s / 10.0) for s in row.shares_db))
                assert abs(total) <= 0.5, (category.key, brakes, row.m, total)


def test_rows_read_from_the_data_sheets() -> None:
    """A few rows as Beiblatt 1 and 2 print them."""
    loco = rail.CATEGORIES[7]
    assert loco.axles == 4 and loco.german == "E-Lok"
    discs = {row.m: row for row in loco.rows("discs")}
    assert discs[1].a_a_db == 66.0 and discs[2].a_a_db == 55.0
    assert discs[8] == rail.Row(8, 61.0, (-28, -19, -6, -4, -6, -10, -14, -23))
    blocks = {row.m: row for row in loco.rows("cast_iron_blocks")}
    assert blocks[2] == rail.Row(2, 71.0, (-40, -30, -22, -9, -3, -5, -15, -26))
    assert rail.CATEGORIES[5].axles == 10
    wagon = {row.m: row for row in rail.CATEGORIES[10].rows("composite_blocks")}
    assert wagon[1].a_a_db == 67.0 and wagon[2].a_a_db == 58.0 and wagon[7].a_a_db == 40.0
    tram = {row.m: row for row in rail.CATEGORIES[21].rows()}
    assert tram[1] == rail.Row(1, 63.0, (-34, -25, -20, -10, -2, -7, -12, -20))
    assert tram[4].a_a_db == 39.0 and rail.SOURCES[5].height_m == 5.0
    assert rail.TRAM_SOURCES[4].height_m == 4.0
    with pytest.raises(ValueError, match="has brakes cast_iron_blocks, discs, not"):
        loco.rows("composite_blocks")


def test_equation_1_at_its_reference_and_off_it() -> None:
    """At 100 km/h on ballast with its own axles a unit's levels are the sheet's; the
    speed factors, the axles (rolling noise alone), the track and the air conditioning
    move them as Schall 03 says."""
    coach = rail.Unit(9, 26.4, "axle_discs")
    rows = {row.m: row for row in coach.kind.rows("axle_discs")}
    at = coach.line_levels_a(100.0)
    assert set(at) == {1, 2, 6, 7, 8}
    for m, level in at.items():
        np.testing.assert_allclose(level, rows[m].a_a_db + np.array(rows[m].shares_db))
    fast = coach.line_levels_a(200.0)
    lg2 = math.log10(2.0)
    np.testing.assert_allclose(fast[1] - at[1], np.array(rail.SPEED_FACTORS["rolling"]) * lg2)
    np.testing.assert_allclose(fast[6] - at[6], 50.0 * lg2)
    np.testing.assert_allclose(fast[8] - at[8], -10.0 * lg2)
    six = rail.Unit(9, 26.4, "axle_discs", axles=6).line_levels_a(100.0)
    np.testing.assert_allclose(six[2] - at[2], 10.0 * math.log10(1.5))
    np.testing.assert_allclose(six[7], at[7])
    slab = coach.line_levels_a(100.0, "slab")
    np.testing.assert_allclose(slab[1] - at[1], [1, 1, 1, 8, 4, 1, 1, 1])
    np.testing.assert_allclose(slab[7] - at[7], 1.0)
    np.testing.assert_allclose(slab[6], at[6])
    tram = rail.Unit(21, 30.0)
    cool = rail.Unit(21, 30.0, air_conditioning=True).line_levels_a(50.0, "street")
    plain = tram.line_levels_a(50.0)
    np.testing.assert_allclose(cool[4] - plain[4], 8.0)
    np.testing.assert_allclose(cool[1] - plain[1], [2, 3, 2, 5, 8, 4, 2, 1])
    # A freight wagon's body rings only if it is a tank wagon's.
    wagon = rail.Unit(10, 17.8, "composite_blocks")
    assert set(wagon.line_levels_a(100.0)) == {1, 2, 7}
    assert set(rail.Unit(10, 17.8, "composite_blocks", tank=True).line_levels_a(100.0)) == {
        1,
        2,
        3,
        4,
        7,
    }
    with pytest.raises(ValueError, match="tracks ballast, slab"):
        coach.line_levels_a(100.0, "street")
    with pytest.raises(ValueError, match="no vehicle category Fz 11"):
        rail.Unit(11, 20.0)


def test_a_moving_unit_sends_its_level_per_metre_and_hour() -> None:
    """A unit going by once an hour at v leaves on each metre the energy it sends while
    it crosses it: its sound power is the level per metre plus 10 lg(1000 v), not
    A-weighted, with the directivity's mean over the sphere."""
    level = np.full(8, 50.0)
    power = rail.unit_power(level, 72.0)
    seconds_per_metre = 1.0 / (72.0 / 3.6)
    expected = level + 10.0 * math.log10(3600.0 / seconds_per_metre)
    np.testing.assert_allclose(power, expected - np.array(A_WEIGHTING_DB) + rail.DIRECTIVITY_DB)
    assert rail.DIRECTIVITY_DB == pytest.approx(0.28, abs=0.005)


def test_the_directivity_is_schall_03s_round_the_track() -> None:
    """Across the track D_I is +1.7 dB, along it -6.6 dB; in the scene the pattern
    averages 0 dB, DIRECTIVITY_DB under Schall 03's."""
    assert rail.directivity_db(90.0) == pytest.approx(1.73, abs=0.01)
    assert rail.directivity_db(0.0) == pytest.approx(-6.58, abs=0.01)
    balloon = rail.directivity()
    ways = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    gains = 10.0 * np.log10(balloon.energy(ways)[:, 0])
    expected = rail.directivity_db([90.0, 0.0, 90.0, 180.0]) - rail.DIRECTIVITY_DB
    np.testing.assert_allclose(gains, expected, atol=0.05)


def test_a_trains_points_share_out_its_units_sound_power() -> None:
    """The regional train's points: two bogies a unit at the rail top, a roof point a
    unit, the locomotive's pantograph; together each unit's sound power."""
    train = rail.TRAINS["regional"]
    points = rail.train_points(train, 160.0)
    by_height = {h: [p for p in points if p.height_m == h] for h in (0.0, 4.0, 5.0)}
    assert [len(by_height[h]) for h in (0.0, 4.0, 5.0)] == [12, 6, 1]
    assert by_height[0.0][0].behind_m == pytest.approx(0.15 * 19.0)
    assert by_height[0.0][1].behind_m == pytest.approx(0.85 * 19.0)
    assert by_height[0.0][2].behind_m == pytest.approx(19.0 + 0.15 * 26.4)
    assert by_height[5.0][0].behind_m == pytest.approx(9.5) and by_height[5.0][0].unit == 0
    for i, unit in enumerate(train.units):
        mine = [p for p in points if p.unit == i]
        energy = sum(10.0 ** (p.power_db / 10.0) for p in mine)
        levels = unit.line_levels_a(160.0)
        expected = rail.unit_power(
            10.0 * np.log10(sum(10.0 ** (v / 10.0) for v in levels.values())), 160.0
        )
        np.testing.assert_allclose(10.0 * np.log10(energy), expected)


def test_a_points_sound_has_its_spectrum_and_its_share_of_the_level() -> None:
    from acoustic_spaces.demos.road import SOUND_RMS
    from acoustic_spaces.synthesis import split_bands

    train = rail.TRAINS["regional"]
    first, other = rail.train_points(train, 160.0)[:3:2]
    x = rail.point_sound(first.power_db, RATE, 4.0, seed=1)
    y = rail.point_sound(other.power_db, RATE, 4.0, reference_db=first.power_db, seed=2)
    assert len(x) == 4 * RATE and np.sqrt(np.mean(x**2)) == pytest.approx(SOUND_RMS)
    ratio = 10.0 * np.log10(np.mean(y**2) / np.mean(x**2))
    expected = 10.0 * np.log10(
        np.sum(10.0 ** (other.power_db / 10.0)) / np.sum(10.0 ** (first.power_db / 10.0))
    )
    assert ratio == pytest.approx(expected, abs=1e-9)
    bands = np.mean(split_bands(x, RATE) ** 2, axis=1)
    shares = 10.0 ** (first.power_db / 10.0)
    loud = shares > 0.01 * shares.max()
    np.testing.assert_allclose(
        10.0 * np.log10(bands[loud] / bands.sum()),
        10.0 * np.log10(shares[loud] / shares.sum()),
        atol=0.5,
    )


def _railway(tmp_path):
    """The railway example, quick: no barrier, the train's walks 40 m long, the listener
    where the EU's rule measures alone, 16 kHz, seeded."""
    import shutil

    import yaml

    folder = tmp_path / "railway"
    shutil.copytree(Path(example_path("railway")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"] = {"ground": {"extent": [[-40, -15], [40, 15]], "sky": 10}}
    data["assignments"] = {"ground": "meadow"}
    for each in data["sources"]:
        if each["name"].startswith("train"):
            z = each["position"][2]
            each["position"] = [-20, 0, z]
            each["walk"]["to"] = [[20, 0, z]]
    data["listeners"] = [lis for lis in data["listeners"] if lis["name"] == "tsi_7m5"]
    data["simulation"].update({"sample_rate": RATE, "ir_length_s": 0.2, "seed": 1})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 500}
    data["simulation"]["early"]["diffraction"] = "none"
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def test_a_trains_sound_is_as_loud_as_its_levels_say(tmp_path) -> None:
    """The S-Bahn's five bogies along one walk: the sound of them all over the sound of
    the first alone, A-weighted, is what the levels say (each bogie's level that much
    later and that much louder or quieter, added), within half a decibel."""
    from acoustic_spaces import Room, Scene
    from acoustic_spaces.demos.passby import Trail, faded, trailed
    from acoustic_spaces.spatial import Mono
    from acoustic_spaces.synthesis import split_bands
    from acoustic_spaces.walkthrough import pass_by, render_walk, simulate_walk

    folder = _railway(tmp_path)
    speed = 80.0
    points = [p for p in rail.train_points(rail.TRAINS["s_bahn"], speed) if p.height_m == 0.0]
    assert len(points) == 5
    scene = Scene.from_yaml(folder / "scene.yaml")
    sources = []
    for source in scene.sources:
        if source.name == "train":
            walk = source.walk.model_copy(update={"speed_m_s": speed / 3.6})
            source = source.model_copy(
                update={"enabled": True, "walk": walk, "level_db": list(points[0].power_db)}
            )
            sources.append(source)
    room = Room.from_scene(scene.model_copy(update={"sources": sources}))
    walked = simulate_walk(room, "train", listener="tsi_7m5", step_m=4.0)
    n = int(round(40.0 / (speed / 3.6) * RATE))
    first = faded(rail.point_sound(points[0].power_db, RATE, n / RATE, seed=0), RATE)
    behind = tuple(
        (
            (p.behind_m - points[0].behind_m) / (speed / 3.6),
            faded(
                rail.point_sound(
                    p.power_db, RATE, n / RATE, reference_db=points[0].power_db, seed=k
                ),
                RATE,
            ),
            p.power_db - points[0].power_db,
        )
        for k, p in enumerate(points[1:], start=1)
    )
    alone = render_walk(walked, {"train": first}, Mono())
    every = render_walk(
        walked, {"train": first}, Mono(), followers=[(lag, y) for lag, y, _ in behind]
    )
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)

    def a_energy(x: np.ndarray) -> float:
        return float(weights @ np.sum(split_bands(x[:, 0], RATE) ** 2, axis=1))

    heard = 10.0 * math.log10(a_energy(every) / a_energy(alone))
    passing = pass_by(walked)
    together = trailed(passing, Trail(0.0, behind), max(lag for lag, _, _ in behind))
    told = together.exposure_a - passing.exposure_a
    assert told > 5.0 and heard == pytest.approx(told, abs=0.5)


def test_a_train_runs_at_its_own_speed_unless_told(tmp_path) -> None:
    """Without --speed each train runs at its own speed, whatever its walk's in the
    scene (the regional train's 160 km/h): the S-Bahn at 120 km/h, the freight train at
    100; --speed still says. The tram, the street's only train, goes as its walk goes."""
    from acoustic_spaces.demos import load

    folder = _railway(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    assert demo.option("speed").unset == "the train's own"
    for train, speed in (("s_bahn", 120.0), ("freight", 100.0)):
        args = demo.resolved(demo.parser().parse_args(["all", "--train", train]))
        assert demo.check(args) is None
        told = f"at {speed:.1f} km/h ({speed / 3.6:.2f} m/s, the train's own)"
        assert told in demo.summary(args)
    run = ["all", "--train", "s_bahn", "--speed", "80"]
    assert "at 80.0 km/h (22.22 m/s, --speed)" in demo.summary(
        demo.resolved(demo.parser().parse_args(run))
    )
    street = load(Path(example_path("street")).parent / "passby_demo.py")
    assert street.option("speed").unset == "as in the scene"
    assert rail.TRAM_VEHICLE.own_speed_kmh(argparse.Namespace()) is None


def test_the_train_horn_sounds_the_rules_notes_at_its_level() -> None:
    """The train's horn: the TSI's high and low notes in turn, each in blasts of its
    own; each note's sound power put back on EN 15153-2's site (its directivity, the
    spreading, the site's ground) is 109 dB(C) 25 m ahead; its made sound plays each
    note at its sound power against the other's; and its sound power is the notes', each
    over its share of the time."""
    import importlib

    from acoustic_spaces.demos import siren
    from acoustic_spaces.parameters import c_weighted

    horn = importlib.import_module("acoustic_spaces.demos.horn")
    kind = horn.TRAIN_HORN
    assert kind.in_turn and kind.notes == ((660.0,), (370.0,))
    for (f,), (low, high) in zip(kind.notes, horn.TRAIN_NOTES_HZ, strict=True):
        assert low <= f <= high
    least, most = horn.TRAIN_LEVELS_DB_C
    assert least <= kind.on_axis_db <= most and kind.weighting == "C" and kind.at_m == 25.0
    t = np.linspace(0.0, 3.0 * kind.every_s, 120_001)
    gates = [horn.sounding(t, kind, note) for note in range(2)]
    assert np.max(gates[0] * gates[1]) == 0.0
    np.testing.assert_array_equal(gates[0] + gates[1], horn.sounding(t, kind))
    site = kind.site
    rise = site.horn_m - site.microphone_m
    spread = 10.0 * math.log10(4.0 * math.pi * (kind.at_m**2 + rise**2))
    off = float(siren.off_axis_db(math.degrees(math.atan2(rise, kind.at_m))))
    for note in range(2):
        power = horn.blast_power(RATE, kind, note)
        ground = horn.ground_db(RATE, kind, note)
        assert np.ptp(ground) > 1.0
        heard = power + siren.directivity_index_db() + off - spread + ground
        assert c_weighted(heard) == pytest.approx(kind.on_axis_db, abs=1e-9)
    x = horn.horn_sound(RATE, kind.power_s, kind)
    times = np.arange(len(x)) / RATE
    steady = [horn.sounding(times, kind, note) > 0.999 for note in range(2)]
    played = [float(np.mean(x[s] ** 2)) for s in steady]
    powers = [float(np.sum(10.0 ** (horn.blast_power(RATE, kind, n) / 10.0))) for n in range(2)]
    assert 10.0 * math.log10(played[0] / played[1]) == pytest.approx(
        10.0 * math.log10(powers[0] / powers[1]), abs=0.1
    )
    total = sum(horn.duty(kind, note) * powers[note] for note in range(2))
    average = float(np.sum(10.0 ** (horn.horn_power(RATE, kind) / 10.0)))
    assert average == pytest.approx(total, rel=1e-9)


def test_the_train_horn_sounds_its_notes_together_as_a_chord() -> None:
    """--horn chord: the TSI's two notes together in the train horn's blasts, the two
    109 dB(C) together 25 m ahead on EN 15153-2's site (a group's level); its made
    sound holds both notes in every blast; a recording plays as loud as the chord's
    made blasts; and the run, its walk and its credit say it is the chord."""
    import importlib

    from acoustic_spaces.demos import siren
    from acoustic_spaces.parameters import c_weighted

    horn = importlib.import_module("acoustic_spaces.demos.horn")
    kind, notes = horn.TRAIN_CHORD, horn.TRAIN_HORN
    assert not kind.in_turn and kind.notes == ((660.0, 370.0),)
    timing = ("blasts", "blast_s", "gap_s", "every_s", "rise_s", "on_axis_db", "site")
    assert all(getattr(kind, k) == getattr(notes, k) for k in timing)
    site = kind.site
    rise = site.horn_m - site.microphone_m
    spread = 10.0 * math.log10(4.0 * math.pi * (kind.at_m**2 + rise**2))
    off = float(siren.off_axis_db(math.degrees(math.atan2(rise, kind.at_m))))
    power = horn.blast_power(RATE, kind)
    heard = power + siren.directivity_index_db() + off - spread + horn.ground_db(RATE, kind)
    assert c_weighted(heard) == pytest.approx(kind.on_axis_db, abs=1e-9)
    x = horn.horn_sound(RATE, kind.every_s, kind)
    for start in (0.0, kind.blast_s + kind.gap_s):
        blast = x[int((start + 0.1) * RATE) : int((start + 0.9) * RATE)]
        spectrum = np.abs(np.fft.rfft(blast))
        f = np.fft.rfftfreq(len(blast), 1.0 / RATE)
        assert sorted(f[np.argsort(spectrum)[-2:]]) == pytest.approx([370.0, 660.0], abs=1.5)
    vehicle = rail.TRAIN_VEHICLE
    chord = argparse.Namespace(horn="chord")
    assert vehicle.horn_on(chord) and vehicle.horn_of(chord) == "chord"
    np.testing.assert_array_equal(vehicle.horn.power("chord", RATE), horn.horn_power(RATE, kind))
    made = horn.horn_sound(RATE, kind.power_s, kind)
    assert vehicle.input_gain(chord, "train_horn", made, RATE) == pytest.approx(1.0, rel=1e-12)
    blasts = argparse.Namespace(horn="blasts")
    assert vehicle.input_gain(blasts, "train_horn", made, RATE) != pytest.approx(1.0, rel=1e-3)
    assert vehicle.played_by(chord) != vehicle.played_by(blasts)
    facts = {"speed_kmh": 160.0, "horn": "chord"}
    assert "horn: 660 and 370 Hz together, in blasts" in vehicle.credits(facts, {})
    assert "horn: 660 and 370 Hz in turn, in blasts" in vehicle.credits(
        facts | {"horn": "blasts"}, {}
    )


def test_a_horns_first_blast_is_not_faded_in(tmp_path) -> None:
    """A pass's sound fades in and out over FADE_S, but a horn's blasts rise and fall
    of themselves: its feed fades over its own rise time only, so the first blast of
    the train's horn plays at its level after rising as every blast does; the train's
    other sources, the fire engine's and the car's horns as the train's horn."""
    import importlib

    from acoustic_spaces.demos import passby, siren
    from acoustic_spaces.demos.audio import write

    horn = importlib.import_module("acoustic_spaces.demos.horn")
    vehicle = rail.TRAIN_VEHICLE
    assert vehicle.fade_s("train_horn") == horn.TRAIN_HORN.rise_s == horn.RISE_S
    assert vehicle.fade_s("train") == passby.FADE_S
    assert siren.FIRE_ENGINE_VEHICLE.fade_s(siren.FIRE_HORN) == horn.AIR_HORN.rise_s
    assert siren.FIRE_ENGINE_VEHICLE.fade_s(siren.FIRE_SIREN) == passby.FADE_S
    assert siren.HONKING_CAR.fade_s(siren.CAR_HORN_PART) == horn.CAR_HORN.rise_s
    duration = 6.0
    args = argparse.Namespace(work=tmp_path, horn="blasts")
    (tmp_path / "train").mkdir()
    write(tmp_path / "train" / "train_horn.wav", vehicle.horn.sound("blasts", RATE, duration), RATE)
    made = vehicle.made(args, RATE, duration, "train_horn")
    x = passby.feed(None, args, RATE, duration, vehicle, "train_horn")
    rise, n = int(horn.RISE_S * RATE), len(made)
    np.testing.assert_array_equal(x[rise : n - rise], made[rise : n - rise])
    first = slice(rise, int(horn.TRAIN_HORN.blast_s * RATE) - rise)
    assert np.sqrt(np.mean(x[first] ** 2)) > 0.9 * np.max(np.abs(made)) / 3.0
    assert np.max(np.abs(x[:rise])) <= np.max(np.abs(made[:rise]))


def test_a_sites_ground_adds_each_harmonics_reflection_in_pressure() -> None:
    """On the train horn's site each harmonic's pressure and its reflection's off the
    ballast add, |1 + D Q (r1/r2) e^(-jk(r2 - r1))|² (e^(jωt), as the engine's), D the
    reflection's directivity against the direct sound's and Q the ground's spherical
    reflection factor, averaged over each band's harmonics by their energy; the band
    of the high note's fundamental, half a wavelength short of cancelling, the lowest."""
    import importlib

    from acoustic_spaces.demos import siren
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
    from acoustic_spaces.materials.ground import PRESETS, spherical_reflection, surface_impedance

    horn = importlib.import_module("acoustic_spaces.demos.horn")
    kind = horn.TRAIN_HORN
    hs, hr, d = kind.site.horn_m, kind.site.microphone_m, kind.at_m
    r1, r2 = math.hypot(d, hs - hr), math.hypot(d, hs + hr)
    sent = 10.0 ** (
        (
            siren.off_axis_db(math.degrees(math.atan2(hs + hr, d)))
            - siren.off_axis_db(math.degrees(math.atan2(hs - hr, d)))
        )
        / 20.0
    )
    for note, (f0,) in enumerate(kind.notes):
        k = np.arange(1, int(RATE / 2 / f0) + 1)
        f = k * f0
        size = np.clip((RATE / 2 - f) / (0.1 * RATE / 2), 0.0, 1.0) / k
        wavenumber = 2.0 * np.pi * f / 343.0
        q = spherical_reflection(
            1.0 / surface_impedance(f, PRESETS["gravel"][0]), (hs + hr) / r2, wavenumber, r2
        )
        gain = np.abs(1.0 + sent * q * (r1 / r2) * np.exp(-1j * wavenumber * (r2 - r1))) ** 2
        band = np.clip(np.round(np.log2(f / OCTAVE_BANDS_HZ[0])), 0, 7)
        want = np.zeros(8)
        for b in range(8):
            if np.any(band == b):
                e = size[band == b] ** 2
                want[b] = 10.0 * np.log10(np.sum(e * gain[band == b]) / np.sum(e))
        got = horn.ground_db(RATE, kind, note)
        np.testing.assert_allclose(got, want, atol=1e-9)
    # The high note: 660 Hz, (r2 - r1) about 0.4 of its wavelength.
    assert int(np.argmin(horn.ground_db(RATE, kind, 0))) == OCTAVE_BANDS_HZ.index(500)


def test_the_railway_demo_runs_a_train_past(tmp_path, monkeypatch) -> None:
    """--train s_bahn on the quick railway: its points' sounds (a column each), the scene
    with the train's three sources alone at their front points' sound power, Schall 03's
    directivity beside it, facing the way it goes, and the pass: as long as the walk and
    the train, its level the parts' energies added, and each height's sound in the mix
    as far under the rail top's as its level is. The video's frames draw from its first
    moment, before the first bogie is heard (its level minus infinity there), to its
    last."""
    pytest.importorskip("PIL")
    from acoustic_spaces import Scene
    from acoustic_spaces.demos import audio, load, passby, pipeline
    from acoustic_spaces.synthesis import split_bands

    # Without ffmpeg (--no-mp4) the video draws only its stills: draw its frames too.
    drawn = []
    real_animation = pipeline.animation

    def animation(folder, duration, draw, *rest, **options):
        for t in np.linspace(0.0, duration, 7):
            drawn.append(draw(float(t)).size)
        return real_animation(folder, duration, draw, *rest, **options)

    monkeypatch.setattr(pipeline, "animation", animation)

    # Each height is rendered alone and the three are added: keep what is added.
    added = []
    real_summed = passby.summed

    def summed(played):
        added.append([np.array(x) for x in played])
        return real_summed(played)

    monkeypatch.setattr(passby, "summed", summed)
    folder = _railway(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["all", "--train", "s_bahn", "--speed", "80", "--step", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None
    parts = ["train", "train_roof", "train_pantograph"]
    assert demo.moving(args) == parts
    assert demo.plays(args)["train"].startswith(
        "an S-Bahn: a four-car electric multiple unit at 80 km/h, made to Schall 03's"
    )
    assert demo.main(run) == 0
    work = folder / "demo"
    info = pipeline.read_json(work / "train" / "train.json")
    assert info["train"] == "s_bahn" and info["track"] == "ballast"
    assert [len(info["parts"][p]) for p in parts] == [5, 2, 1]
    x, rate = audio.read(work / "train" / "train.wav")
    assert rate == RATE and x.shape == (round(40.0 / (80.0 / 3.6) * RATE), 5)
    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == parts
    for name in parts:
        source = next(s for s in scene.sources if s.name == name)
        assert source.level_db == info["parts"][name][0]["level_db"]
        assert source.walk.face_forward and source.walk.doppler
        assert source.directivity.file == folder / "scene_demo_train.csv"
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert facts["parts"] == parts and facts["train"] == "s_bahn"
    heard = facts["listeners"]["tsi_7m5"]
    energy = sum(10.0 ** (p["sel_a_db"] / 10.0) for p in heard["parts"].values())
    assert heard["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    # The pass lasts the walk and the train: the last bogie's sound ends that much later.
    lasts = 40.0 / (80.0 / 3.6) + (info["parts"]["train"][-1]["behind_m"]) / (80.0 / 3.6)
    assert heard["heard_s"][-1] > lasts - 0.5
    mix, _ = audio.read(work / "walk" / "tsi_7m5.binaural.wav")
    assert len(mix) / RATE > lasts
    # Each height plays at its own sound power: its sound in the mix is as far under the
    # rail top's as its SEL is (the roof's some 16 dB(A) under, the pantograph's 28), not as
    # loud as it.
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)

    def a_energy(x: np.ndarray) -> float:
        return sum(
            float(weights @ np.sum(split_bands(x[:, k], RATE) ** 2, axis=1))
            for k in range(x.shape[1])
        )

    (played,) = added
    rail_top = a_energy(played[0])
    for name, x in zip(parts[1:], played[1:], strict=True):
        under = 10.0 * math.log10(rail_top / a_energy(x))
        told = heard["parts"]["train"]["sel_a_db"] - heard["parts"][name]["sel_a_db"]
        assert told > 5.0 and under == pytest.approx(told, abs=1.5), name
    assert (work / "stills" / "00_tsi_7m5.png").exists()
    assert len(drawn) == 7
    assert facts["listeners"]["tsi_7m5"]["levels_a_db"][0] == -math.inf
    # Another track makes the sound again.
    again = demo.resolved(demo.parser().parse_args([*run, "--track", "slab"]))
    assert not demo.is_done(demo.step("sound"), again)


def test_the_railway_demo_sounds_the_trains_horn(tmp_path, monkeypatch) -> None:
    """--horn blasts on the quick railway: the horn's sound beside the train's, its
    source in the scene at its front at its sound power with the measured siren's
    directivity, facing the way it goes, and in the mix as far over the rail top's
    sound as its SEL is, its captions following its blasts. Without --horn it stays
    off."""
    pytest.importorskip("PIL")
    from acoustic_spaces import Scene
    from acoustic_spaces.demos import audio, load, passby, pipeline
    from acoustic_spaces.synthesis import split_bands

    added = []
    real_summed = passby.summed

    def summed(played):
        added.append([np.array(x) for x in played])
        return real_summed(played)

    monkeypatch.setattr(passby, "summed", summed)
    folder = _railway(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["all", "--train", "s_bahn", "--speed", "80", "--step", "8", "--no-mp4"]
    parts = ["train", "train_roof", "train_pantograph"]
    assert demo.moving(demo.resolved(demo.parser().parse_args(run))) == parts
    run += ["--horn", "blasts"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None
    assert demo.moving(args) == [*parts, "train_horn"]
    blasts = "its horn at its front, 2 m above the rail top, in blasts"
    assert demo.plays(args)["train_horn"] == blasts
    assert demo.main(run) == 0
    work = folder / "demo"
    info = pipeline.read_json(work / "train" / "train.json")
    assert info["horn"] == "blasts" and len(info["horn_level_db"]) == 8
    x, rate = audio.read(work / "train" / "train_horn.wav")
    assert rate == RATE and x.shape == (round(40.0 / (80.0 / 3.6) * RATE), 1)
    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    assert [s.name for s in scene.sources if s.in_use] == [*parts, "train_horn"]
    blower = next(s for s in scene.sources if s.name == "train_horn")
    assert blower.level_db == info["horn_level_db"] and blower.position[2] == 2.2
    assert blower.directivity.file == folder / "scene_demo_train_horn.csv"
    lines = blower.directivity.file.read_text(encoding="utf-8").splitlines()
    assert lines[3:] == (folder / "siren.csv").read_text(encoding="utf-8").splitlines()[3:]
    assert blower.walk.face_forward and blower.walk.doppler
    facts = pipeline.read_json(work / "walk" / "walk.json")
    assert facts["parts"] == [*parts, "train_horn"] and facts["horn"] == "blasts"
    assert ", its horn in blasts " in facts["about"]
    heard = facts["listeners"]["tsi_7m5"]
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)

    def a_energy(x: np.ndarray) -> float:
        return sum(
            float(weights @ np.sum(split_bands(x[:, k], RATE) ** 2, axis=1))
            for k in range(x.shape[1])
        )

    (played,) = added
    over = 10.0 * math.log10(a_energy(played[3]) / a_energy(played[0]))
    told = heard["parts"]["train_horn"]["sel_a_db"] - heard["parts"]["train"]["sel_a_db"]
    assert told > 5.0 and over == pytest.approx(told, abs=1.5)


def test_the_street_demo_runs_the_tram_past(tmp_path) -> None:
    """--vehicle tram in the street: its four bogies at the rail top and its roof's two
    halves, on a track set in the street."""
    pytest.importorskip("PIL")
    import shutil

    import yaml

    from acoustic_spaces.demos import load, pipeline

    folder = tmp_path / "street"
    shutil.copytree(Path(example_path("street")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"] = {"ground": {"extent": [[-30, -15], [30, 15]], "sky": 10}}
    data["assignments"] = {"ground": "asphalt"}
    for each in data["sources"]:
        if each["name"].startswith("tram"):
            z = each["position"][2]
            each["position"] = [-20, 0, z]
            each["walk"]["to"] = [[20, 0, z]]
    data["listeners"] = [lis for lis in data["listeners"] if lis["name"] == "pavement"]
    data["simulation"].update({"sample_rate": RATE, "ir_length_s": 0.2, "seed": 1})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 500}
    data["simulation"]["early"]["diffraction"] = "none"
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    demo = load(folder / "passby_demo.py").at(folder)
    run = ["all", "--vehicle", "tram", "--step", "4", "--no-mp4"]
    assert demo.moving(demo.resolved(demo.parser().parse_args(run))) == ["tram", "tram_roof"]
    assert demo.main(run) == 0
    info = pipeline.read_json(folder / "demo" / "tram" / "tram.json")
    assert info["train"] == "tram" and info["track"] == "street"
    assert [len(info["parts"][p]) for p in ("tram", "tram_roof")] == [4, 2]
    facts = pipeline.read_json(folder / "demo" / "walk" / "walk.json")
    assert facts["about"].startswith("A low-floor tram, 30 m long, at 50 km/h on track set")


def test_the_lines_on_a_trains_stills_fit_them() -> None:
    """The line on the stills that says what goes past, for every train on every track,
    the credit line, and what they say of a house the train is heard in, fit the
    still's width whole."""
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    from acoustic_spaces.demos import indoors
    from acoustic_spaces.demos.pictures import MARGIN, SIZE, fitted, font

    draw = ImageDraw.Draw(Image.new("RGB", SIZE))
    house_lines = ["Floor shaking 100 VdB at most, its rumble 100.0 dB(A)"]
    for choice in indoors.choices("train")[1:]:
        house = indoors.House.of(choice, "train")
        assert house is not None
        house_lines += house.words().split("\n")
    for line in house_lines:
        assert draw.textlength(line, font=font("Regular", 23)) <= SIZE[0] - 1010 - MARGIN, line
    for vehicle in (rail.TRAIN_VEHICLE, rail.TRAM_VEHICLE):
        for key in vehicle.trains:
            for track in vehicle.tracks or (vehicle.track,):
                train = rail.TRAINS[key]
                info = {
                    "train": key,
                    "track": track,
                    "length_m": train.length_m,
                    "speed_kmh": train.speed_kmh,
                    "sound_power_a_db": 133.3,
                }
                about = vehicle.facts(None, info)["about"]
                for line in about.split("\n"):
                    width = draw.textlength(line, font=font("Regular", 23))
                    assert width <= SIZE[0] - 1010 - MARGIN, line
                credit = vehicle.credits({"speed_kmh": train.speed_kmh}, {})
                assert fitted(draw, credit, 19, SIZE[0] - 2 * MARGIN)[0] == credit
