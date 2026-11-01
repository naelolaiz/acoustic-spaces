"""The pass-by demos heard in the house a listener stands in front of: the façade's
level difference, the rumble of the ground's vibration, the two in the railway demo's
mix, and what is felt there."""

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
    The room hears R + 10 lg(V / (6 T0 S)) less than 2 m in front of the façade."""
    r = facade.glazing_r_db()
    assert r[1:7].tolist() == list(facade.GLAZING_R_DB)
    assert r[0] == pytest.approx(20.5 - 5.95, abs=0.01) and r[7] == 36.5
    rw, c, ctr = facade.weighted(facade.GLAZING_R_DB[:5])
    assert abs(rw - 30) <= 1 and (c, ctr) == (-1, -4)
    house = facade.HOUSE
    assert house.window_m2 == pytest.approx(1.23 * 1.48)
    term = 10.0 * math.log10(50.0 / (6.0 * 0.5 * 1.23 * 1.48))
    assert house.room_term_db == pytest.approx(term) and term == pytest.approx(9.62, abs=0.01)
    assert np.allclose(house.indoor_db(), -np.round(r, 2) - term)


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


def test_a_listener_0_5_to_2_m_in_front_of_a_wall_is_in_its_house() -> None:
    """The street's pavement stands 2 m in front of the houses, its balcony 1 m in front
    of the offices, below their roof; its garden in front of none. The railway's
    house_18m stands 2 m in front of its house. Nearer than 0.5 m, farther than 2 m,
    inside a footprint or above its roof, a listener is in front of none."""
    from acoustic_spaces import Scene
    from acoustic_spaces.examples import example_path

    street = Scene.from_yaml(example_path("street"))
    found = indoors.facings(street, ["pavement", "balcony", "garden"])
    assert (found["pavement"].name, found["pavement"].distance_m) == ("houses", 2.0)
    assert (found["balcony"].name, found["balcony"].distance_m) == ("offices", 1.0)
    assert found["garden"] is None
    railway = Scene.from_yaml(example_path("railway"))
    found = indoors.facings(railway, [lis.name for lis in railway.listeners])
    assert {k: v and v.name for k, v in found.items()} == {
        "tsi_7m5": None,
        "meadow_25m": None,
        "house_18m": "house",
        "behind_barrier_25m": None,
    }
    # The houses' wall along y = 6, from x = -30 to 10, 9 m high.
    for at, house in (
        ((0.0, 5.5, 1.6), "houses"),
        ((0.0, 5.6, 1.6), None),
        ((0.0, 3.9, 1.6), None),
        ((0.0, 4.0, 8.9), "houses"),
        ((0.0, 4.0, 9.0), None),
        ((0.0, 10.0, 1.6), None),
        # Off the corner (10, 6), square in front of neither wall.
        ((11.0, 5.0, 1.6), None),
        ((11.0, 7.0, 1.6), "houses"),
    ):
        facing = indoors.facing(street, at)
        assert (facing and facing.name) == house, at
    facing = indoors.facing(street, (0.0, 4.0, 1.6))
    assert facing is not None and facing.wall == (0.0, 6.0)
    away = facing.distances([(0.0, 0.0), (-40.0, 6.0), (0.0, 10.0)])
    np.testing.assert_allclose(away, [6.0, 40.0, 4.0])


def test_the_ground_shakes_a_room_from_behind_its_listener() -> None:
    """In a house, the ground shakes as the curve says at the distance from each wheel
    to the room's wall behind the listener, not to the listener in front of it; nor to
    the building's nearest corner: the street's row of houses ends 4 m from the
    breaker, but the pavement's room is 20 m along the street from it."""
    speed = 72.0
    path = _Path(100.0, speed / 3.6)
    wheels = indoors.Wheels("locomotive", speed, (3.0,), (0.0,))
    times = np.linspace(0.0, 10.0, 201)
    house = indoors.Facing("house", (0.0, 12.0), 2.0)
    at_house = indoors.ground_vdb(wheels, path, house, times)
    at_listener = indoors.ground_vdb(wheels, path, (0.0, 10.0, 1.5), times)
    curve = vibration.CURVES["locomotive"]
    assert np.max(at_house) == pytest.approx(curve.level(12.0, speed), abs=1e-6)
    assert np.max(at_listener) == pytest.approx(curve.level(10.0, speed), abs=1e-6)
    from acoustic_spaces import Scene
    from acoustic_spaces.examples import example_path

    street = Scene.from_yaml(example_path("street"))
    breaker = next(s for s in street.sources if s.name == "breaker").position
    room = indoors.facings(street, ["pavement"])["pavement"]
    assert room is not None and room.distances([breaker])[0] == pytest.approx(math.hypot(20, 4))
    houses = next(b for b in street.geometry.all_buildings() if b.name == "houses")
    assert vibration.horizontal_distance(breaker, houses.footprint, closed=True) == 4.0


def test_what_is_felt_is_the_vibrations_velocity() -> None:
    """The shaker's track: noise in the third octave at 25 Hz whose RMS is the velocity
    of the vibration level, 1 micro-inch/s at 0 VdB, faded in and out. People feel it
    from 65 VdB."""
    n = RATE * 6
    times = [0.0, 6.0]
    x = indoors.shaker_sound(times, [80.0, 80.0], RATE, n, seed=1)
    fade = int(indoors.FADE_S * RATE)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 4
    share = (n - 2 * fade + 2 * float(np.sum(ramp))) / n
    velocity = vibration.VELOCITY_REFERENCE_M_S * 10.0 ** (80.0 / 20.0)
    assert math.sqrt(float(np.mean(x**2)) / share) == pytest.approx(velocity, rel=1e-9)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    inside = (f >= 25.0 / indoors.THIRD) & (f < 25.0 * indoors.THIRD)
    assert float(np.sum(spectrum[inside]) / np.sum(spectrum)) > 0.99  # the fades spread it
    assert np.all(indoors.shaker_sound(times, [-math.inf] * 2, RATE, n) == 0.0)
    assert indoors.felt(65.0) == "felt" and indoors.felt(64.9) == "too little to feel"
    assert indoors.shaking_line(70.4, False) == "Ground shaking 70 VdB at most, felt"
    assert indoors.shaking_line(70.4, True, 20.44) == (
        "Floor shaking 70 VdB at most, felt; its rumble 20.4 dB(A)"
    )


def test_the_shakers_tracks_have_one_gain(tmp_path) -> None:
    """The listeners' shaker tracks: one gain for all, the loudest peak at -1 dBFS, its
    full scale a velocity and a level; none when nothing is felt."""
    from acoustic_spaces.demos.audio import read

    a, b = np.full(RATE, 2e-4), np.full(RATE, -1e-4)
    told = indoors.write_shakers(tmp_path, {"a": a, "b": b}, RATE)
    full = 2e-4 / 10.0 ** (-1.0 / 20.0)
    assert told["full_scale_mm_s"] == pytest.approx(full * 1e3, rel=1e-5) and told["hz"] == 25.0
    assert told["full_scale_vdb"] == pytest.approx(
        20.0 * math.log10(full / vibration.VELOCITY_REFERENCE_M_S), abs=0.01
    )
    x, rate = read(tmp_path / "a.shaker.wav")
    y, _ = read(tmp_path / "b.shaker.wav")
    assert rate == RATE and float(np.max(np.abs(x))) == pytest.approx(10 ** (-1 / 20), rel=1e-3)
    assert float(np.mean(y)) == pytest.approx(-0.5 * 10 ** (-1 / 20), rel=1e-3)
    assert indoors.write_shakers(tmp_path, {"c": np.zeros(4)}, RATE) == {}


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


#: The long railway's house: its wall 9.5 m from the track, 2 m behind tsi_7m5.
HOUSE_WALL_M = 9.5


def _long_railway(tmp_path: Path) -> Path:
    """The quick railway with a walk long enough that the fades at its ends lose under
    0.1 dB of what the house hears: 160 m; a house 2 m behind tsi_7m5."""
    import yaml
    from test_rail import _railway

    folder = _railway(tmp_path)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    wall = HOUSE_WALL_M
    data["geometry"] = {
        "ground": {"extent": [[-90, -15], [90, 15]], "sky": 10},
        "buildings": [
            {"name": "house", "footprint": [[-5, wall], [5, wall], [5, 14], [-5, 14]], "height": 6}
        ],
    }
    data["assignments"] = {"ground": "meadow", "house": "brick"}
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
    """--indoors train_and_rumble, mono, at tsi_7m5, 2 m in front of a house: each part
    of the train some 35 to 39 dB(A) quieter inside than in front of it, the rumble
    one more part whose floor shakes as the curve says at the house's wall, the
    house's 3 dB more. In the mix the rumble has the energy its SEL gives in the
    engine's units (to 0.05 dB), the rail top's sound is as much under its sound
    outdoors as their SELs are (to 0.5 dB), and the two are as far apart as their SELs
    (to 1 dB). What is felt there is the floor's vibration, outdoors the ground's. The
    rumble alone simulates nothing and plays the rumble alone; another --indoors walks
    again."""
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
    common += ["--format", "mono", "--listeners", "tsi_7m5"]
    outdoors = ["walk", *common]
    run = ["all", *common, "--indoors", "train_and_rumble"]
    assert demo.main(["sound", *common]) == 0 and demo.main(["scene", *common]) == 0
    assert demo.main(outdoors) == 0
    out = pipeline.read_json(folder / "demo" / "walk" / "walk.json")["listeners"]["tsi_7m5"]
    curve = vibration.CURVES[rail.TRAINS["s_bahn"].curve]
    # Outdoors: the ground under the listener, 7.5 m from the track.
    assert out["shaking"] == "ground" and "inside" not in out
    assert out["vibration_vdb"] == pytest.approx(curve.level(7.5, 80.0), abs=0.05)
    assert demo.main(run) == 0
    facts = pipeline.read_json(folder / "demo" / "walk" / "walk.json")
    assert facts["indoors"] == "train_and_rumble" and facts["plays"].endswith(
        "; indoors train_and_rumble"
    )
    heard = facts["listeners"]["tsi_7m5"]
    assert heard["inside"] == "house" and heard["shaking"] == "floor"
    assert heard["about"] == (
        "Inside the house here, through a closed window (EN ISO 12354-3)\n"
        "and the rumble of the floors and walls (FTA)"
    )
    assert facts["shaker"]["hz"] == 25.0
    assert (folder / "demo" / "walk" / "tsi_7m5.shaker.wav").exists()
    parts = heard["parts"]
    assert list(parts) == ["train", "train_roof", "train_pantograph", "rumble"]
    energy = sum(10.0 ** (p["sel_a_db"] / 10.0) for p in parts.values())
    assert heard["sel_a_db"] == pytest.approx(10.0 * math.log10(energy), abs=0.02)
    # The train through the window: some 35 to 39 dB(A) under 2 m in front of it, near
    # what ISO 717-1 gives traffic's spectrum, Rw + Ctr + 10 lg(V / (6 T0 S)) = 34.6 dB.
    rw, _, ctr = facade.weighted(facade.GLAZING_R_DB[:5])
    traffic = rw + ctr + facade.HOUSE.room_term_db
    assert traffic == pytest.approx(34.6, abs=0.05)
    for name in ("train", "train_roof", "train_pantograph"):
        assert 0.0 < out["parts"][name]["sel_a_db"] - parts[name]["sel_a_db"] - traffic < 4.0
    # The floor: the curve at the house's wall, 9.5 m from the track, at 80 km/h, + 3 dB.
    assert heard["vibration_vdb"] == pytest.approx(curve.level(HOUSE_WALL_M, 80.0) + 3.0, abs=0.05)
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
    # The wall's reflection, some 4 m longer, combs the sound every 86 Hz or so, which
    # moves its energy within each band against the filter's crossovers, where the
    # façade's level difference steps by 9 to 11 dB: the 500 Hz and 1 kHz bands lose
    # 0.3 dB less than it says (measured), so to 0.5 dB.
    assert through == pytest.approx(told, abs=0.5)
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


def test_a_house_beside_still_plant() -> None:
    """Plant nearer than 10 ft shakes as it does 10 ft away, not without bound; a level
    unknown in a band outdoors is unknown in the house too, not the outdoor one."""
    from acoustic_spaces.spatial import Mono

    breaker = indoors.Plant("breaker", "hoe_ram", (0.0, 0.0, 1.0))
    house = indoors.House.of("works_and_rumble", "works")
    still = indoors.StillHouses(
        house,
        "works_and_rumble",
        (breaker,),
        {"out": (0.0, 0.0, 1.5), "in": (0.0, -1.0, 1.5)},
        {"out": None, "in": indoors.Facing("h", (0.0, 0.0), 1.0)},
        100.0,
        Mono(),
    )
    nearest = vibration.plant_level("hoe_ram", vibration.NEAREST_M)
    assert still.ground_vdb("out") == still.ground_vdb("in") == pytest.approx(nearest)
    report = {"responses": [{"listener": "in", "spl_db": [None] + [50.0] * 7, "spl_a_db": 60.0}]}
    (entry,) = still.report(report)["responses"]
    assert entry["spl_a_db"] is None and entry["spl_db"] is None
