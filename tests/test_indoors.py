"""The pass-by demos heard in a house at each listener: the façade's level difference,
the rumble of the ground's vibration, and the two in the railway demo's mix."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import facade, vibration
from acoustic_spaces.demos import indoors

RATE = 16_000


def test_the_window_is_agcs_double_glazing() -> None:
    """The window's sound reduction index is AGC's 4-16-4 from 125 Hz to 4 kHz, 6 dB
    less an octave lower and held an octave higher; ISO 717-1 rates those octave values
    Rw 29 (C -1, Ctr -4) dB, within a decibel of AGC's 30 (-1; -4) from third octaves.
    The room hears R + 10 lg(V / (6 T0 S)) - 3 dB less than the free field outside."""
    r = facade.glazing_r_db()
    assert r[1:7].tolist() == list(facade.GLAZING_R_DB)
    assert r[0] == pytest.approx(20.5 - 5.95, abs=0.01) and r[7] == 36.5
    rw, c, ctr = facade.weighted(facade.GLAZING_R_DB[:5])
    assert abs(rw - 30) <= 1 and (c, ctr) == (-1, -4)
    house = facade.HOUSE
    assert house.window_m2 == pytest.approx(1.23 * 1.48)
    term = 10.0 * math.log10(50.0 / (6.0 * 0.5 * 1.23 * 1.48))
    assert house.room_term_db == pytest.approx(term) and term == pytest.approx(9.62, abs=0.01)
    assert np.allclose(house.indoor_db(), 3.0 - np.round(r, 2) - term)


def test_iso_717_rates_its_reference_curve_2_db_over_its_value_at_500_hz() -> None:
    """The reference curve itself rates 2 dB over its 500 Hz value, 2 dB short in each
    of the five bands being the 10 dB allowed; one 11 dB under it at 250 Hz alone
    rates a decibel under its 500 Hz value."""
    reference = np.asarray(facade.REFERENCE_DB) - 22.0
    assert facade.weighted(reference)[0] == 32
    dipped = reference.copy()
    dipped[1] -= 11.0
    assert facade.weighted(dipped)[0] == 29


def test_the_house_hears_each_band_through_the_facade() -> None:
    """Noise heard through the façade: each octave band's energy changed by the façade's
    level difference, within 0.1 dB, in every channel, though it steps by up to 11 dB
    from one band to the next."""
    from acoustic_spaces.synthesis import split_bands

    x = np.random.default_rng(3).standard_normal((RATE * 4, 2))
    heard = indoors.House(airborne=True, rumble=False).heard(x, RATE)
    for k in range(2):
        before = np.sum(split_bands(x[:, k], RATE) ** 2, axis=1)
        after = np.sum(split_bands(heard[:, k], RATE) ** 2, axis=1)
        assert np.allclose(10 * np.log10(after / before), facade.HOUSE.indoor_db(), atol=0.1)


def test_a_curve_goes_on_past_where_the_manual_draws_it() -> None:
    """Extended, a base curve is the manual's from 10 to 300 ft, holds its 10 ft value
    nearer, and farther falls on as it falls at 300 ft, without a step; unextended it
    still refuses those distances."""
    for curve in vibration.CURVES.values():
        near, far = vibration.NEAREST_M, vibration.FARTHEST_M
        assert curve.level(20.0, 80.0, extend=True) == curve.level(20.0, 80.0)
        assert curve.level(near / 2, extend=True) == pytest.approx(curve.level(near))
        step = 1e-4
        inside = curve.level(far) - curve.level(far * 10 ** (-step))
        beyond = curve.level(far * 10**step, extend=True) - curve.level(far)
        assert beyond == pytest.approx(inside, rel=1e-2) and beyond < 0.0
        assert curve.level(far * 10, extend=True) == pytest.approx(
            curve.level(far) + inside / step, abs=1e-3
        )
        with pytest.raises(ValueError):
            curve.level(far * 2)


class _Path:
    """A walk along x at ``speed`` m/s from -l to l, y = 0."""

    def __init__(self, length: float, speed: float) -> None:
        self.times = np.array([0.0, 2 * length / speed])
        self.positions = np.array([[-length, 0.0, 0.0], [length, 0.0, 0.0]])


def test_the_ground_shakes_most_as_the_train_runs_past() -> None:
    """A train's wheels along a straight track: the ground 20 m from it shakes as the
    curve says 20 m away while its wheels pass, less as they come and go, not at all
    before its front is on the walk or after its last wheels leave it; a freight
    train's wagons 3 dB under its locomotive."""
    house = indoors.House(airborne=False, rumble=True)
    speed = 72.0
    path = _Path(100.0, speed / 3.6)
    wheels = indoors.Wheels("locomotive", speed, (3.0, 16.0, 30.0), (0.0, 0.0, -3.0))
    times = np.linspace(-1.0, 13.0, 1401)
    ground = house.ground_vdb(wheels, path, (0.0, 20.0, 1.2), times)
    alongside = vibration.CURVES["locomotive"].level(20.0, speed)
    assert np.max(ground) == pytest.approx(alongside, abs=1e-6)
    # The front's first wheels are 3 m behind it: nothing until they are on the walk.
    assert np.all(np.isneginf(ground[times < 3.0 / 20.0 - 1e-9]))
    assert np.isfinite(ground[times > 3.0 / 20.0 + 1e-9][0])
    after = times > 10.0 + 30.0 / 20.0 + 1e-9
    assert np.all(np.isneginf(ground[after]))
    # When the wagon alone is level with the house, 3 dB under the locomotive's.
    wagon = int(np.argmin(np.abs(times - (100.0 + 30.0) / 20.0)))
    loco = int(np.argmin(np.abs(times - (100.0 + 3.0) / 20.0)))
    assert ground[loco] == pytest.approx(alongside, abs=1e-6)
    nearest = max(
        vibration.CURVES["locomotive"].level(math.hypot(dx, 20.0), speed, extend=True) + off
        for dx, off in ((27.0, 0.0), (14.0, 0.0), (0.0, -3.0))
    )
    assert ground[wagon] == pytest.approx(nearest, abs=1e-6)
    # Far off, the curve extended: 60 m along the track, 63 m away.
    far = int(np.argmin(np.abs(times - (100.0 - 60.0 + 3.0) / 20.0)))
    assert ground[far] == pytest.approx(
        vibration.CURVES["locomotive"].level(math.hypot(60.0, 20.0), speed), abs=0.05
    )


@pytest.mark.parametrize("peak", sorted(vibration.RUMBLE_HZ))
def test_the_rumble_is_made_to_its_level(peak: str) -> None:
    """The rumble's sound has the A-weighted mean square its level gives in the mix's
    units, and an unweighted level 5 dB under the vibration's, as the manual says,
    within 1.2 dB for each peak of table 6-14."""
    from acoustic_spaces.demos.audio import a_gain

    house = indoors.House(airborne=False, rumble=True, peak=peak)
    n = RATE * 6
    times = np.linspace(0.0, 6.0, 61)
    vdb = np.full(len(times), 75.0)
    level = vdb + vibration.NOISE_DB[peak]
    x = house.rumble_sound(times, level, RATE, n, loudest_db=110.0, seed=2)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    spectrum = np.abs(np.fft.rfft(x)) ** 2 * 2.0 / n**2
    a_ms = float(np.sum(spectrum * a_gain(f) ** 2))
    fade = int(indoors.FADE_S * RATE)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 4
    share = (n - 2 * fade + 2 * float(np.sum(ramp))) / n
    told = float(indoors.mean_square(10.0 ** (level[0] / 10.0), 110.0)) * share
    assert 10 * math.log10(a_ms / told) == pytest.approx(0.0, abs=0.01)
    unweighted = level[0] + 10 * math.log10(float(np.sum(spectrum)) / a_ms)
    assert unweighted - vdb[0] == pytest.approx(vibration.UNWEIGHTED_DB, abs=1.2)
    centre = f[np.argmax(spectrum)]
    assert centre / vibration.RUMBLE_HZ[peak] == pytest.approx(1.0, abs=0.13)


def test_a_mix_in_pascals_is_the_engines_own() -> None:
    """A level's mean square in the mix: a part as loud as the loudest heard at r metres
    in the free field, Lw - 10 lg(4 π r²), has the mean square of its feed over r²."""
    from acoustic_spaces.demos.road import SOUND_RMS

    lw, r = 105.0, 7.5
    level = lw - 10.0 * math.log10(4.0 * math.pi * r**2)
    assert float(indoors.mean_square(10.0 ** (level / 10.0), lw)) == pytest.approx(
        SOUND_RMS**2 / r**2
    )


def test_the_rumble_reaches_every_format_as_a_pressure() -> None:
    """The same in each ear, in mono and in W; a microphone's omnidirectional part;
    1/√n in each main loudspeaker, none in the LFE."""
    from acoustic_spaces.spatial import Ambisonics, Binaural, Mono
    from acoustic_spaces.spatial.loudspeakers import Loudspeakers
    from acoustic_spaces.spatial.microphone import Microphone

    assert indoors.pressure_gains(Mono()).tolist() == [1.0]
    assert indoors.pressure_gains(Binaural()).tolist() == [1.0, 1.0]
    assert indoors.pressure_gains(Ambisonics(2)).tolist() == [1.0] + [0.0] * 8
    layout = Loudspeakers.layout("5.1")
    gains = indoors.pressure_gains(layout)
    lfe = [s.lfe for s in layout.speakers]
    assert gains[lfe].tolist() == [0.0] and np.allclose(gains[~np.array(lfe)], 1 / math.sqrt(5))
    pair = Microphone("xy", ("L", "R"), (0.5, 0.5), ((0.7, 0.7, 0.0), (0.7, -0.7, 0.0)))
    assert indoors.pressure_gains(pair).tolist() == [0.5, 0.5]


def test_indoors_takes_off_the_vehicle_and_the_rumble() -> None:
    """--indoors: off, the vehicle and the rumble, the vehicle alone, the rumble
    alone."""
    assert indoors.choices("tram") == ("off", "tram_and_rumble", "tram", "rumble")
    assert indoors.House.of("off", "tram") is None and indoors.House.of(None, "tram") is None
    both = indoors.House.of("tram_and_rumble", "tram")
    assert both is not None and both.airborne and both.rumble
    alone = indoors.House.of("tram", "tram")
    assert alone is not None and alone.airborne and not alone.rumble
    rumble = indoors.House.of("rumble", "tram")
    assert rumble is not None and rumble.rumble and not rumble.airborne
    with pytest.raises(ValueError):
        indoors.House.of("train", "tram")


def _long_railway(tmp_path: Path) -> Path:
    """The quick railway with a walk long enough that the fades at its ends lose under
    0.1 dB of what the house hears: 160 m."""
    import yaml
    from test_rail import _railway

    folder = _railway(tmp_path)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["geometry"] = {"ground": {"extent": [[-90, -15], [90, 15]], "sky": 10}}
    for each in data["sources"]:
        if each["name"].startswith("train"):
            z = each["position"][2]
            each["position"] = [-80, 0, z]
            each["walk"]["to"] = [[80, 0, z]]
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def _a_energy(x: np.ndarray) -> float:
    """The A-weighted energy of ``x`` (samples, channels), by its spectrum."""
    from acoustic_spaces.demos.audio import a_gain

    f = np.fft.rfftfreq(len(x), 1.0 / RATE)
    spectrum = np.abs(np.fft.rfft(x, axis=0)) ** 2
    spectrum[1:] *= 2.0
    return float(np.sum(spectrum * (a_gain(f) ** 2)[:, None])) / len(x)


def test_the_railway_demo_hears_the_train_and_its_rumble_indoors(tmp_path, monkeypatch) -> None:
    """--indoors train_and_rumble, mono: each part of the train some 32 to 35 dB(A)
    quieter than outdoors, the rumble one more part whose floor shakes as the curve
    says at the track's distance, the house's 3 dB more. In the mix the rumble has the
    energy its SEL gives in the engine's units (to 0.05 dB), the rail top's sound is as
    much under its sound outdoors as their SELs are (to 0.1 dB), and the two are as far
    apart as their SELs (to 1 dB). The rumble alone simulates nothing and plays the
    rumble alone; another --indoors walks again."""
    pytest.importorskip("PIL")
    from acoustic_spaces.demos import load, passby, pipeline, rail

    added = []
    real_summed = passby.summed

    def summed(played):
        added.append([np.array(x) for x in played])
        return real_summed(played)

    monkeypatch.setattr(passby, "summed", summed)
    folder = _long_railway(tmp_path)
    demo = load(folder / "passby_demo.py").at(folder)
    assert demo.option("indoors").choices == indoors.choices("train")
    common = ["--train", "s_bahn", "--speed", "80", "--step", "8", "--no-mp4"]
    common += ["--format", "mono"]
    outdoors = ["walk", *common]
    run = ["all", *common, "--indoors", "train_and_rumble"]
    assert demo.main(["sound", *common]) == 0 and demo.main(["scene", *common]) == 0
    assert demo.main(outdoors) == 0
    out = pipeline.read_json(folder / "demo" / "walk" / "walk.json")["listeners"]["tsi_7m5"]
    assert demo.main(run) == 0
    facts = pipeline.read_json(folder / "demo" / "walk" / "walk.json")
    assert facts["indoors"] == "train_and_rumble" and facts["plays"].endswith(
        "; indoors train_and_rumble"
    )
    assert facts["about"].endswith(
        "In a house here, through its closed window (EN ISO 12354-3)\n"
        "and the rumble of its floors and walls (FTA)"
    )
    heard = facts["listeners"]["tsi_7m5"]
    parts = heard["parts"]
    assert list(parts) == ["train", "train_roof", "train_pantograph", "rumble"]
    energy = sum(10.0 ** (p["sel_a_db"] / 10.0) for p in parts.values())
    assert heard["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    # The train through the window: some 32 to 35 dB(A) under outdoors, near what ISO
    # 717-1 gives traffic's spectrum, Rw + Ctr + 10 lg(V / (6 T0 S)) - 3 = 31.6 dB.
    rw, _, ctr = facade.weighted(facade.GLAZING_R_DB[:5])
    traffic = rw + ctr + facade.HOUSE.room_term_db - facade.FACADE_REFLECTION_DB
    assert traffic == pytest.approx(31.6, abs=0.05)
    for name in ("train", "train_roof", "train_pantograph"):
        assert 0.0 < out["parts"][name]["sel_a_db"] - parts[name]["sel_a_db"] - traffic < 4.0
    # The floor: the curve at the listener's 7.5 m from the track, at 80 km/h, + 3 dB.
    curve = vibration.CURVES[rail.TRAINS["s_bahn"].curve]
    assert heard["vibration_vdb"] == pytest.approx(curve.level(7.5, 80.0) + 3.0, abs=0.05)
    assert parts["rumble"]["max_a_db"] == pytest.approx(heard["vibration_vdb"] - 50.0, abs=0.01)
    # In the mix: the rumble at its SEL in the engine's own units, the rail top's sound
    # through the window as much under its sound outdoors as their SELs are, and the
    # two as far apart as their SELs say. The pass-by's own sound of a train's part is
    # within 0.6 dB of its SEL, outdoors as indoors (the rail top's 0.6 dB under it on
    # this meadow): so the last within 1 dB.
    from acoustic_spaces import Scene

    scene = Scene.from_yaml(folder / "scene_demo.yaml")
    loudest = max(each.sound_power_db for each in scene.sources if each.in_use)
    played = added[-1]
    assert len(played) == 4
    rumble = _a_energy(played[3]) / RATE
    wanted = float(indoors.mean_square(10.0 ** (parts["rumble"]["sel_a_db"] / 10.0), loudest))
    assert 10.0 * math.log10(rumble / wanted) == pytest.approx(0.0, abs=0.05)
    through = 10.0 * math.log10(_a_energy(added[0][0]) / _a_energy(played[0]))
    told = out["parts"]["train"]["sel_a_db"] - parts["train"]["sel_a_db"]
    assert through == pytest.approx(told, abs=0.1)
    under = 10.0 * math.log10(_a_energy(played[0]) / _a_energy(played[3]))
    told = parts["train"]["sel_a_db"] - parts["rumble"]["sel_a_db"]
    assert under == pytest.approx(told, abs=1.0)
    # The rumble alone: nothing simulated, the rumble alone in the mix.
    again = demo.resolved(demo.parser().parse_args(["all", *common, "--indoors", "rumble"]))
    assert not demo.is_done(demo.step("walk"), again)

    def nothing(*args, **kwargs):
        raise AssertionError("the rumble alone simulates nothing")

    monkeypatch.setattr("acoustic_spaces.walkthrough.simulate_walk", nothing)
    assert demo.main(["walk", *common, "--indoors", "rumble"]) == 0
    alone = pipeline.read_json(folder / "demo" / "walk" / "walk.json")["listeners"]["tsi_7m5"]
    assert "parts" not in alone and "pitch_semitones" not in alone
    assert alone["sel_a_db"] == pytest.approx(parts["rumble"]["sel_a_db"], abs=0.01)
    assert len(added[-1]) == 1


def test_the_tram_has_indoors_and_the_street_s_other_vehicles_not() -> None:
    """In the street demo --indoors is the tram's alone."""
    from acoustic_spaces.demos import load
    from acoustic_spaces.examples import example_path

    demo = load(Path(example_path("street")).parent / "passby_demo.py")
    option = demo.option("indoors")
    assert option.only == ("vehicle", ("tram",))
    assert option.choices == indoors.choices("tram")
