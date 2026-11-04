"""Barriers and buildings in wind: the parabolic equation with them inside scales the
paths in the source and listener's vertical plane by the weather's effect."""

import math

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import parabolic, refracted
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.simulate import early_paths

WINDY = {"preset": "windy", "wind": {"direction_deg": 0}}


def scene(
    atmosphere=WINDY,
    listeners=((40.0, 0.0, 1.5),),
    barriers=({"name": "screen", "path": [[15, -20], [15, 20]], "height": 4},),
    buildings=(),
    file=None,
    extent=((-60.0, -60.0), (90.0, 60.0)),
    ir_length_s=0.3,
    **early,
) -> Scene:
    """A source 1 m up over hard ground, a barrier 15 m away across the x axis, and
    listeners."""
    geometry = {
        "ground": {"extent": [list(corner) for corner in extent], "sky": 20.0},
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
            "ir_length_s": ir_length_s,
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
        found.append(
            {"weather": weather, "ranges": tuple(ranges), "frequencies": frequencies, **options}
        )
        amplitude = math.sqrt(2.0) if weather else 1.0
        return np.full((len(np.atleast_1d(frequencies)), len(ranges)), amplitude, dtype=complex)

    monkeypatch.setattr(parabolic, "march", march)
    return found


@pytest.fixture
def field(monkeypatch):
    """The parabolic equation's runs, not run: each gives, at every listener of its, the
    pressure ``field.wind(obstacles, range)`` in the weather (sqrt(2) by default) and 1 in
    still air, both 0 under a roof (a building's stretch over the listener's range)."""

    class Field:
        wind = staticmethod(lambda obstacles, r: math.sqrt(2.0))

    def march(frequencies, speed, impedance, zs, ranges, heights, top, **options):
        weather = bool(np.ptp(speed(np.array([0.0, 5.0, 50.0]))) > 0.0)
        obstacles = options.get("obstacles", ())
        values = []
        for r in ranges:
            if any(near < r < far for near, far, _ in obstacles):
                values.append(0.0)
            else:
                values.append(Field.wind(obstacles, r) if weather else 1.0)
        rows = len(np.atleast_1d(frequencies))
        return np.tile(np.array(values, dtype=complex), (rows, 1))

    monkeypatch.setattr(parabolic, "march", march)
    return Field


@pytest.fixture
def scaled(monkeypatch):
    """By listener (index), the factor (bands,) the runs behind barriers scaled the
    energy of its paths in the plane by."""
    found = {}
    place = refracted.Refraction._place_barrier

    def spy(self, request, energies, parts):
        plane = parts[request.listener][request.part]
        before = plane.energy.sum(axis=0)
        place(self, request, energies, parts)
        found[request.listener] = plane.energy.sum(axis=0) / before

    monkeypatch.setattr(refracted.Refraction, "_place_barrier", spy)
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
    # A listener on a whole degree's direction (0), one between 0 and 1 degree, and two
    # mirrored across the wind's, between 26 and 27 degrees either side.
    listeners = ((40.0, 0.0, 1.5), (70.0, 0.4, 1.5), (40.0, 20.0, 1.5), (40.0, -20.0, 1.5))
    simulate(Room.from_scene(scene(listeners=listeners)))
    behind = [run for run in runs if run.get("obstacles")]
    weather = [run for run in behind if run["weather"]]
    still = [run for run in behind if not run["weather"]]
    # Per band, in the weather and in still air: one run along 0 degrees (the first two
    # listeners), one along 1 degree (the second), and one along each of 26 and 27
    # degrees for the two mirrored listeners (the same wind along their ways and the
    # same cut).
    assert len(weather) == len(still) == 4 * 8
    far, mirrored = math.hypot(70.0, 0.4), math.hypot(40.0, 20.0)
    assert sorted({run["ranges"] for run in weather}) == sorted(
        {(40.0, far), (far,), (mirrored, mirrored)}
    )
    assert all(run["refracting"] and run["independent"] for run in behind)
    # From 1 kHz up a quarter wavelength's steps, else a tenth's; 8 frequencies a band up
    # to 2 kHz, 16 at 4 kHz and 32 at 8 kHz, where the ground's fringes are densest.
    steps = sorted({run["steps_per_wavelength"] for run in behind})
    assert steps == [4.0, 10.0]
    samples = {(float(np.min(run["frequencies"])), len(run["frequencies"])) for run in behind}
    assert [n for _, n in sorted(samples)] == [8, 8, 8, 8, 8, 8, 16, 32]


def test_the_paths_in_the_plane_take_the_weather_s_effect(runs) -> None:
    """With the weather's effect 2 in every band, each path in the plane carries twice
    its energy in still air, keeps its spectrum (so their sum in pressure carries twice
    its energy too) and comes as much sooner as the first of them."""
    still = early_paths(scene(atmosphere=None))
    windy = early_paths(scene())
    assert np.all(windy.edge >= 0)
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
    """Listeners either side of the shadow boundary of the barrier's end: the paths
    round the end take the weather's effect over the open ground (2 here) in the bands
    of the open ground's parabolic equation, on both sides, delayed alike there (their
    phases changing as the open ground's field does, not at all here), and their legs'
    own times and shadows only above them."""
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
        assert len(ends) >= 2, name  # off the ground or not
        shifts, times = [], []
        for i in ends:
            rows = np.flatnonzero(b.distance_m == a.distance_m[i])
            energy = b.energy[rows].sum(axis=0)
            np.testing.assert_allclose(energy[open_bands], 2.0 * a.energy[i, open_bands])
            # Downwind the legs have no shadow above them.
            np.testing.assert_allclose(energy[~open_bands], a.energy[i, ~open_bands])
            low = rows[b.energy[rows][:, open_bands].sum(axis=1) > 0.0]
            high = rows[b.energy[rows][:, ~open_bands].sum(axis=1) > 0.0]
            shifts.append(float(b.delay_s[low[0]] - a.delay_s[i]))
            times.append(float(b.delay_s[high[0]] - a.delay_s[i]))
        assert np.ptp(shifts) < 1e-12 and np.ptp(times) > 1e-5


def test_a_breath_of_wind_behind_a_barrier_is_still_air() -> None:
    """In a wind of 0.01 m/s at 10 m the levels behind the barrier, in the middle and
    near its end, are within 0.02 dB of those in still air in every band."""
    listeners = ((40.0, 0.0, 1.5), (40.0, 18.0, 1.5))
    breath = {"wind": {"speed_m_s": 0.01, "direction_deg": 0}}
    levels = []
    for atmosphere in (None, breath):
        result = simulate(Room.from_scene(scene(atmosphere=atmosphere, listeners=listeners)))
        levels.append([10 * np.log10(total_energy(result["s", f"l{k}"].early)) for k in range(2)])
    np.testing.assert_allclose(levels[1], levels[0], atol=0.02)


def test_a_listener_close_behind_a_slanting_wall_takes_the_effect_behind_it(field, scaled):
    """Behind a barrier and a building's back wall at 45 degrees to the way, 0.7 to 1.3 m
    behind them: along the nearest whole degree the barrier stands beyond the listener,
    or the listener would be under the roof. The weather's effect is the one behind them
    (2 here), not the open ground's (5) nor none (1, the still air under a roof)."""

    def wind(obstacles, r):
        return math.sqrt(2.0 if any(far < r for _, far, _ in obstacles) else 5.0)

    field.wind = wind
    angle = math.radians(-0.49)
    listeners = [(r * math.cos(angle), r * math.sin(angle), 1.5) for r in (129.6, 130.0, 130.2)]
    slanting = {"name": "b", "path": [[120, -10], [140, 10]], "height": 6}
    building = {"name": "h", "footprint": [[100, -10], [120, -10], [140, 10], [100, 10]]}
    for barriers, buildings in (([slanting], []), ([], [{**building, "height": 6}])):
        scaled.clear()
        found = scene(
            listeners=listeners,
            barriers=barriers,
            buildings=buildings,
            extent=((-20.0, -60.0), (160.0, 60.0)),
            ir_length_s=0.6,
            diffraction="second_order",
        )
        simulate(Room.from_scene(found))
        assert sorted(scaled) == [0, 1, 2]
        for factor in scaled.values():
            np.testing.assert_allclose(factor, 2.0, rtol=1e-9)


def test_the_effect_changes_smoothly_from_one_degree_s_run_to_the_next(field, scaled):
    """Behind a barrier at 14 degrees to the way, 120 m out, where the barrier stands
    farther along each whole degree: either side of 4.5 degrees the weather's effect is
    the same, halfway between those along 4 and 5 degrees (in dB); along 4.2 degrees, the
    run along 4 degrees alone."""

    def wind(obstacles, r):
        return math.sqrt(2.0 ** (obstacles[0][0] / 10.0)) if obstacles else 1.0

    field.wind = wind
    barrier = {"name": "b", "path": [[-20, -15], [160, 45]], "height": 3}
    angles = (4.2, 4.4999, 4.5001)
    listeners = [
        (120 * math.cos(math.radians(a)), 120 * math.sin(math.radians(a)), 1.5) for a in angles
    ]
    simulate(
        Room.from_scene(
            scene(
                listeners=listeners,
                barriers=[barrier],
                extent=((-30.0, -40.0), (170.0, 60.0)),
                ir_length_s=0.6,
            )
        )
    )

    def crossing(degrees):
        # Where the way along ``degrees`` meets the barrier (x = 3 y + 25).
        a = math.radians(degrees)
        return 25.0 / (math.cos(a) - 3.0 * math.sin(a))

    halfway = 2.0 ** ((crossing(4.0) + crossing(5.0)) / 20.0)
    np.testing.assert_allclose(scaled[0], 2.0 ** (crossing(4.0) / 10.0), rtol=1e-6)
    np.testing.assert_allclose(scaled[1], halfway, rtol=1e-3)
    np.testing.assert_allclose(scaled[2], halfway, rtol=1e-3)


def test_paths_over_the_edges_of_a_model_file_s_objects_keep_their_legs(runs, tmp_path):
    """A listener in the open beside a crate of the model file, downwind: the paths over
    the crate's edges keep their energy in still air (their straight legs have no shadow
    downwind), not the open ground's effect (2 here)."""
    crate = tmp_path / "crate.obj"
    corners = [(x, y, z) for z in (0, 2) for y in (-1, 1) for x in (24, 26)]
    sides = [(1, 2, 4, 3), (5, 6, 8, 7), (1, 2, 6, 5), (3, 4, 8, 7), (1, 3, 7, 5), (2, 4, 8, 6)]
    crate.write_text(
        "g crate\n"
        + "".join(f"v {x} {y} {z}\n" for x, y, z in corners)
        + "".join("f " + " ".join(map(str, s)) + "\n" for s in sides)
    )
    listeners = ((40.0, 3.0, 1.5),)
    still = early_paths(scene(atmosphere=None, listeners=listeners, barriers=(), file=crate))
    windy = early_paths(scene(listeners=listeners, barriers=(), file=crate))
    over = np.flatnonzero(still.edge >= 0)
    assert len(over) >= 2
    for i in over:
        rows = np.flatnonzero(windy.distance_m == still.distance_m[i])
        np.testing.assert_allclose(windy.energy[rows].sum(axis=0), still.energy[i], rtol=1e-12)


def test_the_paths_round_an_end_take_the_effect_along_their_own_way(runs):
    """Behind the middle of a barrier 40 m long, the open ground's runs for the paths round
    each end go as far as their way round it, seen from above (not the listener's 60 m)."""
    simulate(Room.from_scene(scene(listeners=((60.0, 0.0, 4.0),))))
    open_ground = {r for run in runs if not run.get("obstacles") for r in run["ranges"]}
    way = math.hypot(15.0, 20.0) + math.hypot(45.0, 20.0)
    assert any(abs(r - way) < 1e-9 for r in open_ground)
    assert not any(abs(r - 60.0) < 1e-9 for r in open_ground)


def test_a_way_that_touches_a_building_s_corner_from_outside_meets_nothing() -> None:
    """A line through a corner of a footprint that lies to one side of it, where the two
    crossings there come out a rounding error apart."""
    from acoustic_spaces.geometry.outdoor import Layout

    turn, out = 1.9050292966180866, 96.16459900132378
    width, depth = 2.2063752752244827, 4.628016878024163
    d = np.array([math.cos(turn), math.sin(turn)])
    n = np.array([-d[1], d[0]])
    corner = out * d
    footprint = np.array(
        [
            corner,
            corner + d * width - n * depth,
            corner - n * 2 * depth,
            corner - d * width - n * depth,
        ]
    )
    layout = Layout((-500, -500, 500, 500), 0.0, 50.0, (("b", footprint, 6.0),), ())
    assert layout.obstacles_along(np.zeros(2), d, 400.0) == []


def test_the_sound_starts_with_the_paths_over_the_barrier_where_they_come_sooner() -> None:
    """Behind barriers in wind no path is clear; downwind the paths over them come sooner
    than the straight line's time, and the response (a map's point too, whose bands
    added in energy begin there) starts with the first of them; upwind, at the straight
    line's time."""
    from acoustic_spaces.engine.early import EarlyReflections

    def over(delay):
        return EarlyReflections(
            order=np.zeros(2, dtype=np.int64),
            delay_s=np.array([delay, delay + 0.003]),
            distance_m=np.array([34.0, 35.0]),
            direction=np.array([[-1.0, 0.0, 0.0]] * 2),
            energy=np.ones((2, 8)),
            edge=np.array([3, 3]),
        )

    onset = refracted.direct_times(np.array([0.1, 0.1]), [over(0.098), over(0.102)])
    np.testing.assert_array_equal(onset, [0.098, 0.1])


@pytest.mark.parametrize("towards", [0.0, 180.0])
def test_a_listener_beside_a_building_takes_the_open_ground_s_effect(towards) -> None:
    """A listener in the open 3 m beside a building's wall, 60 m from the source down-
    and upwind: the paths round the building's corners change with the weather as the
    open ground's field along their way does, at each frequency, so that with the direct
    sound they take the open ground's effect (within 0.7 dB from 63 to 500 Hz)."""
    building = {"name": "h", "footprint": [[30, 0], [40, 0], [40, 10], [30, 10]], "height": 6}
    weather = {"preset": "windy", "wind": {"direction_deg": towards}}
    listeners = ((60.0, -3.0, 1.5),)
    effects = []
    for buildings in ((building,), ()):
        levels = []
        for atmosphere in (None, weather):
            room = Room.from_scene(
                scene(
                    atmosphere=atmosphere,
                    listeners=listeners,
                    barriers=(),
                    buildings=buildings,
                    diffraction="second_order",
                )
            )
            levels.append(10 * np.log10(total_energy(early_paths(room, listener="l0"))))
        effects.append(levels[1] - levels[0])
    np.testing.assert_allclose(effects[0][:4], effects[1][:4], atol=0.7)


def test_a_listener_on_a_barrier_s_line_keeps_main_s_rule() -> None:
    """A listener (or a map point) on a barrier's line seen from above, 58 m from the
    source exactly: the parabolic equation would take it past the barrier, where nothing
    below its top gets through, in the weather and in still air alike; it keeps the
    open ground's rule instead of raising."""
    wall = {"name": "wall", "path": [[40, -50], [40, 50]], "height": 4}
    room = Room.from_scene(
        scene(listeners=((40.0, -42.0, 1.5),), barriers=(wall,), extent=((-10, -60), (90, 60)))
    )
    levels = 10 * np.log10(total_energy(early_paths(room, listener="l0")))
    assert np.all(np.isfinite(levels))


@pytest.mark.parametrize(("chosen", "counts"), [(32, [16, 32]), (16, [16, 16]), (8, [8, 8])])
def test_the_frequencies_at_4_and_8_khz_behind_barriers_can_be_fewer(runs, chosen, counts):
    """``simulation.early.barrier_frequencies`` sets how many frequencies an octave the
    runs behind barriers take at 8 kHz, and at 4 kHz no more than 16; below, 8."""
    simulate(Room.from_scene(scene(barrier_frequencies=chosen)))
    behind = [run for run in runs if run.get("obstacles")]
    samples = {(float(np.min(run["frequencies"])), len(run["frequencies"])) for run in behind}
    assert [n for _, n in sorted(samples)] == [8] * 6 + counts
