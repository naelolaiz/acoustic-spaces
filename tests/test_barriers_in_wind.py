"""Barriers and buildings in wind: the parabolic equation with them inside scales the
paths in the source and listener's vertical plane by the weather's effect."""

import math

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import parabolic, refracted
from acoustic_spaces.engine.simulate import early_paths

WINDY = {"preset": "windy", "wind": {"direction_deg": 0}}


def scene(
    atmosphere=WINDY,
    listeners=((40.0, 0.0, 1.5),),
    barriers=({"name": "screen", "path": [[15, -20], [15, 20]], "height": 4},),
    buildings=(),
    file=None,
    **early,
) -> Scene:
    """A source 1 m up over hard ground, a barrier 15 m away across the x axis, and
    listeners."""
    geometry = {
        "ground": {"extent": [[-60.0, -60.0], [90.0, 60.0]], "sky": 20.0},
        "barriers": list(barriers),
        "buildings": list(buildings),
    }
    if file is not None:
        geometry.update({"file": str(file), "up_axis": "z"})
    data = {
        "geometry": geometry,
        "materials": {"hard": {"absorption": [0.02] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": [0.0, 0.0, 1.0]}],
        "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
        "simulation": {
            "seed": 1,
            "early": {
                "method": "image_sources",
                "max_order": 1,
                "diffraction": "first_order",
                "diffraction_reflections": "both_sides",
                **early,
            },
            "late": {"rays": 200},
            "ir_length_s": 0.3,
        },
    }
    if atmosphere is not None:
        data["atmosphere"] = atmosphere
    return Scene.model_validate(data)


@pytest.fixture
def runs(monkeypatch):
    """The parabolic equation's runs, recorded and not run: each gives the pressure 1 in
    still air and sqrt(2) in the weather, so the weather's effect is 2 everywhere."""
    found = []

    def march(frequencies, speed, impedance, zs, ranges, heights, top, **options):
        weather = bool(np.ptp(speed(np.array([0.0, 5.0, 50.0]))) > 0.0)
        found.append({"weather": weather, "ranges": tuple(ranges), **options})
        amplitude = math.sqrt(2.0) if weather else 1.0
        return np.full((len(np.atleast_1d(frequencies)), len(ranges)), amplitude, dtype=complex)

    monkeypatch.setattr(parabolic, "march", march)
    return found


@pytest.fixture
def asked(monkeypatch):
    """The listeners (by name) that ask for the runs behind barriers."""
    names = []
    place = refracted.Refraction._place_barrier

    def spy(self, request, energies, parts):
        names.append(request.listener)
        return place(self, request, energies, parts)

    monkeypatch.setattr(refracted.Refraction, "_place_barrier", spy)
    return names


def test_who_asks_for_the_runs_behind_barriers(runs, asked, tmp_path) -> None:
    building = {
        "name": "shed",
        "footprint": [[-25, -5], [-35, -5], [-35, 5], [-25, 5]],
        "height": 6,
    }
    listeners = (
        (40.0, 0.0, 1.5),  # behind the barrier: asks
        (30.0, 60.0, 1.5),  # its way passes the barrier's end: does not
        (16.0, 0.0, 12.0),  # nearer than 1.5 times the heights added: does not
        (-30.0, 0.0, 7.0),  # on the shed's roof: does not
        (-50.0, 0.0, 1.5),  # behind the shed (over its roof's two edges): asks
        (35.0, -10.0, 15.0),  # high over the barrier, which is below the line of sight: asks
    )
    simulate(
        Room.from_scene(
            scene(listeners=listeners, buildings=[building], diffraction="second_order")
        )
    )
    assert sorted(asked) == [0, 4, 5]
    cuts = {run["obstacles"] for run in runs if "obstacles" in run and run["obstacles"]}
    assert {((15.0, 15.0, 4.0),), ((25.0, 35.0, 6.0),)} <= cuts
    # No paths over edges behind the barrier: none in the plane but where the barrier is
    # below the line of sight.
    asked.clear()
    simulate(Room.from_scene(scene(listeners=listeners, diffraction="none")))
    assert asked == [5]
    asked.clear()
    simulate(Room.from_scene(scene(listeners=listeners, max_order=0)))
    assert asked == []
    asked.clear()
    simulate(Room.from_scene(scene(atmosphere=None, listeners=listeners)))
    assert asked == []
    # A crate of the model file in the way: the rays.
    crate = tmp_path / "crate.obj"
    corners = [(x, y, z) for z in (0, 2) for y in (-1, 1) for x in (24, 26)]
    sides = [(1, 2, 4, 3), (5, 6, 8, 7), (1, 2, 6, 5), (3, 4, 8, 7), (1, 3, 7, 5), (2, 4, 8, 6)]
    crate.write_text(
        "g crate\n"
        + "".join(f"v {x} {y} {z}\n" for x, y, z in corners)
        + "".join("f " + " ".join(map(str, s)) + "\n" for s in sides)
    )
    asked.clear()
    simulate(Room.from_scene(scene(listeners=((40.0, 0.0, 1.5), (40.0, 8.0, 1.5)), file=crate)))
    assert asked == [1]


def test_one_run_in_the_weather_and_one_in_still_air_for_each_direction(runs) -> None:
    # Two listeners in one degree's direction, two mirrored across the wind's.
    listeners = ((40.0, 0.0, 1.5), (70.0, 0.4, 1.5), (40.0, 20.0, 1.5), (40.0, -20.0, 1.5))
    simulate(Room.from_scene(scene(listeners=listeners)))
    behind = [run for run in runs if run.get("obstacles")]
    weather = [run for run in behind if run["weather"]]
    still = [run for run in behind if not run["weather"]]
    # Per band: one run in the weather along the axis (two listeners), one for the two
    # mirrored listeners (the same wind along their ways and the same cut); still air,
    # one for the cut along the axis and one for the cut of both mirrored.
    assert len(weather) == len(still) == 2 * 8
    assert sorted({run["ranges"] for run in weather}) == sorted(
        {(40.0, math.hypot(70.0, 0.4)), (math.hypot(40.0, 20.0),) * 2}
    )
    assert all(run["refracting"] and run["independent"] for run in behind)
    # From 1 kHz up a quarter wavelength's steps, else a tenth's.
    steps = sorted({run["steps_per_wavelength"] for run in behind})
    assert steps == [4.0, 10.0]


def test_the_paths_in_the_plane_take_the_weather_s_effect(runs) -> None:
    """With the weather's effect 2 in every band, each path in the plane carries twice
    its energy in still air, keeps its spectrum (so their sum in pressure carries twice
    its energy too) and comes as much sooner as the first of them."""
    still = early_paths(scene(atmosphere=None))
    windy = early_paths(scene())
    assert len(windy) == len(still) and np.all(windy.edge >= 0)
    # Over the top (the plane's y = 0), not round the ends.
    plane = [i for i in range(len(still)) if abs(still.points[i][-2][1]) < 1e-6]
    assert len(plane) >= 4  # the four paths over the top, off the ground or not
    plane = np.array(plane)[np.argsort(still.distance_m[plane])]
    again = np.array(
        [int(np.flatnonzero(windy.distance_m == still.distance_m[i])[0]) for i in plane]
    )
    np.testing.assert_allclose(windy.energy[again], 2.0 * still.energy[plane], rtol=1e-12)
    np.testing.assert_array_equal(windy.spectrum[again], still.spectrum[plane])
    shift = windy.delay_s[again] - still.delay_s[plane]
    assert np.ptp(shift) < 1e-12 and shift[0] < -1e-4  # downwind: sooner


def test_the_paths_round_an_end_take_the_open_ground_s_effect_on_both_sides(runs) -> None:
    """Listeners either side of the shadow boundary of the barrier's end: the path
    round the end takes the weather's effect over the open ground (2 here) in the bands
    of the open ground's parabolic equation, on both sides, and its leg's shadow only
    above them."""
    barrier = {"name": "screen", "path": [[15, -20], [15, 0]], "height": 4}
    listeners = ((60.0, -1.0, 1.5), (60.0, 1.0, 1.5))
    still = Room.from_scene(scene(atmosphere=None, listeners=listeners, barriers=[barrier]))
    windy = Room.from_scene(scene(listeners=listeners, barriers=[barrier]))
    # 60 m downwind the open ground's parabolic equation gives 63 to 500 Hz.
    open_bands = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    for name in ("l0", "l1"):
        a = early_paths(still, listener=name)
        b = early_paths(windy, listener=name)
        ends = [
            i
            for i in range(len(a))
            if any(np.allclose(c[:2], [15.0, 0.0]) for c in a.points[i][1:-1])
        ]
        assert ends, name
        for i in ends:
            j = int(np.flatnonzero(b.distance_m == a.distance_m[i])[0])
            np.testing.assert_allclose(b.energy[j, open_bands], 2.0 * a.energy[i, open_bands])
            # Downwind the leg has no shadow above them.
            np.testing.assert_allclose(b.energy[j, ~open_bands], a.energy[i, ~open_bands])


def test_a_breath_of_wind_behind_a_barrier_is_still_air() -> None:
    """In a wind of 0.01 m/s at 10 m the levels behind the barrier, in the middle and
    near its end, are within 0.02 dB of those in still air in every band."""
    listeners = ((40.0, 0.0, 1.5), (40.0, 18.0, 1.5))
    breath = {"wind": {"speed_m_s": 0.01, "direction_deg": 0}}
    levels = []
    for atmosphere in (None, breath):
        result = simulate(Room.from_scene(scene(atmosphere=atmosphere, listeners=listeners)))
        levels.append(
            [10 * np.log10(result["s", f"l{k}"].early.energy.sum(axis=0)) for k in range(2)]
        )
    np.testing.assert_allclose(levels[1], levels[0], atol=0.02)
