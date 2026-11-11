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
    lawn=None,
    sources=((0.0, 0.0, 1.0),),
    **early,
) -> Scene:
    """A source 1 m up over hard ground (grass on the polygon ``lawn``), a barrier 15 m
    away across the x axis, and listeners; or ``sources`` (named s, or s0, s1 ...)."""
    geometry = {
        "ground": {"extent": [list(corner) for corner in extent], "sky": 20.0},
        "barriers": list(barriers),
        "buildings": list(buildings),
    }
    materials = {"hard": {"absorption": [0.02] * 8, "scattering": 0.0}}
    if lawn is not None:
        geometry["ground"]["zones"] = [{"name": "lawn", "polygon": [list(c) for c in lawn]}]
        materials["grass"] = {"impedance": "grass"}
    if file is not None:
        geometry.update({"file": str(file), "up_axis": "z"})
    data = {
        "geometry": geometry,
        "materials": materials,
        "assignments": {"lawn": "grass"} if lawn is not None else {},
        "default_material": "hard",
        "sources": [
            {"name": "s" if len(sources) == 1 else f"s{k}", "position": list(p)}
            for k, p in enumerate(sources)
        ],
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
        # Its way passes 10 m from the barrier's end, where the path over the top still
        # takes part as the edge fades it out: asks, along the way just inside the end.
        (30.0, 60.0, 1.5),
        (16.0, 0.0, 12.0),  # nearer than 1.5 times the heights added: does not
        (-30.0, 0.0, 7.0),  # on the shed's roof: asks, the runs ending over it
        (-50.0, 0.0, 1.5),  # behind the shed (over its roof's two edges): asks
        (35.0, -10.0, 15.0),  # high over the barrier, which is below the line of sight: asks
    )
    simulate(
        Room.from_scene(
            scene(listeners=listeners, buildings=[building], diffraction="second_order")
        )
    )
    assert sorted(asked) == [0, 1, 3, 4, 5]
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


def test_the_reflection_off_a_roof_takes_the_weather_s_effect_behind_it(runs) -> None:
    """A source and a listener 10 m up either side of a building 6 m high: the path off
    its roof is in the plane, which the runs with the building inside hold, and takes
    their effect (2 here) as the direct sound does. Against the full-wave solution 40
    to 50 m away (``validation/barrier_wind.py``'s ``mast``) the weather's effect is
    then within 0.03 dB from 63 to 500 Hz downwind, where with the path's own straight
    legs it was 0.56 dB off, and 1.4 to 3.7 dB at 2 to 8 kHz 50 m away."""
    ends = {"sources": ((0.0, 0.0, 10.0),), "listeners": ((50.0, 0.0, 10.0),)}
    still = early_paths(scene(atmosphere=None, barriers=(), buildings=(ROOF,), **ends))
    windy = early_paths(scene(barriers=(), buildings=(ROOF,), **ends))
    off_roof = [
        i
        for i in range(len(still))
        if still.edge[i] < 0 and still.order[i] == 1 and abs(still.points[i][1][2] - 6.0) < 1e-9
    ]
    assert len(off_roof) == 1
    i = off_roof[0]
    again = int(np.flatnonzero(windy.distance_m == still.distance_m[i])[0])
    np.testing.assert_allclose(windy.energy[again], 2.0 * still.energy[i], rtol=1e-12)


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
    # 60 m downwind the open ground's parabolic equation gives 63 Hz to 2 kHz.
    open_bands = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
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


def test_the_paths_round_an_end_take_the_open_ground_s_effect_in_line_with_it(runs) -> None:
    """A listener exactly in line with the source and the barrier's end, where the
    straight way touches the end: the direct sound is cut off there, and the paths round
    the end take the open ground's effect (2 here) from 63 to 500 Hz all the same."""
    barrier = {"name": "screen", "path": [[15, -20], [15, 10]], "height": 4}
    listeners = ((60.0, 40.0, 1.5),)
    a = early_paths(scene(atmosphere=None, listeners=listeners, barriers=[barrier]))
    b = early_paths(scene(listeners=listeners, barriers=[barrier]))
    assert not np.any((a.order == 0) & (a.edge < 0))  # cut off
    ends = [
        i for i in range(len(a)) if any(np.allclose(c[:2], [15.0, 10.0]) for c in a.points[i][1:-1])
    ]
    assert len(ends) >= 2  # off the ground or not
    for i in ends:
        rows = np.flatnonzero(b.distance_m == a.distance_m[i])
        energy = b.energy[rows].sum(axis=0)
        np.testing.assert_allclose(energy[:4], 2.0 * a.energy[i, :4])


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


def test_the_runs_behind_a_barrier_share_the_march_up_to_it(monkeypatch) -> None:
    """The runs behind a barrier along different degrees take the march over the open
    ground before it from one another (:class:`parabolic.Trunk`): the listeners get the
    bytes they get where every run marches from the source by itself."""
    listeners = ((25.0, 0.0, 1.5), (25.0, 8.0, 1.5))
    shared = simulate(Room.from_scene(scene(listeners=listeners, barrier_frequencies=8)))
    march = parabolic.march
    trunks = []

    def alone(*args, trunk=None, **options):
        trunks.append(trunk)
        return march(*args, **options)

    monkeypatch.setattr(parabolic, "march", alone)
    apart = simulate(Room.from_scene(scene(listeners=listeners, barrier_frequencies=8)))
    assert any(trunk is not None for trunk in trunks)
    for k in range(len(listeners)):
        one, other = shared["s", f"l{k}"].early, apart["s", f"l{k}"].early
        assert one.energy.tobytes() == other.energy.tobytes()
        assert one.spectrum.tobytes() == other.spectrum.tobytes()


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


def test_the_paths_round_an_end_take_the_ground_along_their_own_way(monkeypatch) -> None:
    """Grass on the side of the barrier's end, the listener's own line over hard ground
    beyond its first 3 m: the open ground's run for the paths round that end takes the
    grass along its way, unfolded round the end (15 m along it, and 2.3 m past the end),
    and the hard ground after; the run round the other end, hard ground all along."""
    found = []

    def march(frequencies, speed, impedance, zs, ranges, heights, top, **options):
        if not options.get("obstacles"):
            found.append((max(ranges), [impedance(r) for r in (5.0, 14.0, 40.0)]))
        return np.ones((len(np.atleast_1d(frequencies)), len(ranges)), dtype=complex)

    monkeypatch.setattr(parabolic, "march", march)
    lawn = ((-60.0, -0.5), (90.0, -0.5), (90.0, 60.0), (-60.0, 60.0))
    barrier = {"name": "screen", "path": [[15, -20], [15, 0]], "height": 4}
    simulate(Room.from_scene(scene(listeners=((60.0, -10.0, 1.5),), barriers=[barrier], lawn=lawn)))
    beyond = math.hypot(45.0, 10.0)
    near = [z for reach, z in found if abs(reach - (15.0 + beyond)) < 1e-9]
    far = [z for reach, z in found if abs(reach - (25.0 + beyond)) < 1e-9]
    assert near and far
    hard = far[0][0][0]  # the same at each frequency, in runs of 8 or 24 frequencies
    assert np.allclose(far[0][0], hard)
    for z in far:
        for at in z:
            np.testing.assert_allclose(at, hard)
    for grass, also, rest in near:
        np.testing.assert_allclose(grass, also)
        np.testing.assert_allclose(rest, hard)
        assert not np.allclose(grass, hard)


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


#: A building 6 m high, 10 m by 20 m, with a point 1 m above the middle of its roof.
ROOF = {"name": "h", "footprint": [[15, -10], [25, -10], [25, 10], [15, 10]], "height": 6}
ON_ROOF = (20.0, 0.0, 7.0)


def _walked(pairs, towards, **kwargs) -> np.ndarray:
    """The weather's effect (dB, the ``windy`` preset towards ``towards`` degrees over
    still air) from 63 to 500 Hz between each (source, listener) of ``pairs``."""
    levels = []
    for atmosphere in (None, {"preset": "windy", "wind": {"direction_deg": towards}}):
        room = Room.from_scene(scene(atmosphere=atmosphere, barriers=(), **kwargs))
        levels.append(
            [10.0 * np.log10(total_energy(early_paths(room, s, r)))[:4] for s, r in pairs]
        )
    return np.subtract(levels[1], levels[0])


def _steps(effect: np.ndarray) -> np.ndarray:
    """(bands,) the largest change of ``effect`` (points, bands) from one point to the next."""
    return np.abs(np.diff(effect, axis=0)).max(axis=0)


# Where the barrier's equation does not run (an object of the model file), the direct
# ray and the one off the ground are there where the straight paths the edges' terms
# make up for are. The same wall as a scene's barrier, where the equation runs, changes
# the weather's effect by at most 0.1 dB in 10 cm from 63 to 500 Hz across the direct
# sound's shadow boundary and 0.04 dB (-0.7 to 1.3 dB effect) across the ground
# reflection's, source 8 m up. Here the bent rays' level and time, which the edges'
# terms do not follow, step by up to 0.2 dB across the direct sound's boundary and
# 0.6 dB across the reflection's behind the wall: bounds 0.3 and 0.7 dB. Unbent, the
# paths' sets disagreed between the straight and the bent boundary by 4 to 10 dB.

#: The largest steps (dB) a 10 cm walk across the boundaries may take, see above.
DIRECT_STEP, REFLECTION_STEP = 0.3, 0.7


@pytest.fixture
def bent(monkeypatch):
    """By source position, the band levels (dB) of the paths to each listener as the
    weather bends them (:meth:`~acoustic_spaces.engine.refracted.Planned.finish`)."""
    found = {}
    finish = refracted.Planned.finish

    def spy(self, pressures):
        paths = finish(self, pressures)
        found[tuple(self.refraction.source)] = [10.0 * np.log10(total_energy(p)) for p in paths]
        return paths

    monkeypatch.setattr(refracted.Planned, "finish", spy)
    return found


#: The roof walks' effects, by :func:`_roof_walk`'s arguments: the walk from the roof's
#: source downwind in octave bands also serves the points close behind the edge.
_ROOF_WALKS: dict[tuple, np.ndarray] = {}


def _roof_walk(
    bent, on_roof: str, towards: float, bands: str, xs, equation: str = "narrow_angle"
) -> np.ndarray:
    """(points, bands) the weather's effect (dB, the ``windy`` preset towards
    ``towards`` degrees over still air) between the point over ``ROOF`` (``on_roof``:
    the "source" or the "listener") and points 1.5 m up at ``xs`` on the x axis, with
    ``simulation.early.parabolic_equation: equation``. The runs behind the building take
    16 frequencies an octave at 8 kHz, half the default, for speed; from the source in
    third octaves the default, as with 16 the 10 kHz band's fringes there curve by up to
    0.09 dB over three 10 cm steps (0.03 dB with 32; to the listener 0.013 dB with 16)."""
    key = (on_roof, towards, bands, tuple(xs), equation)
    if key in _ROOF_WALKS:
        return _ROOF_WALKS[key]
    ground = [(x, 0.0, 1.5) for x in xs]
    if on_roof == "source":
        ends = {"sources": (ON_ROOF,), "listeners": ground}
        pairs = [("s", f"l{k}") for k in range(len(xs))]
    else:
        ends = {"sources": ground, "listeners": (ON_ROOF,)}
        pairs = [(f"s{k}", "l0") for k in range(len(xs))]
    levels = []
    for atmosphere in (None, {"preset": "windy", "wind": {"direction_deg": towards}}):
        made = scene(
            atmosphere=atmosphere,
            barriers=(),
            buildings=(ROOF,),
            barrier_frequencies=16 if bands == "octave" or on_roof == "listener" else 32,
            parabolic_equation=equation,
            **ends,
        )
        made.simulation.bands = bands
        made.simulation.late.rays = 20  # the early paths alone are looked at
        room = Room.from_scene(made)
        if atmosphere is None:
            levels.append(
                [10.0 * np.log10(total_energy(early_paths(room, s, r))) for s, r in pairs]
            )
            continue
        bent.clear()
        simulate(room)  # every listener of a source in one plan, as a map takes them
        if on_roof == "source":
            levels.append(bent[ON_ROOF])
        else:
            levels.append([bent[point][0] for point in ground])
    _ROOF_WALKS[key] = np.subtract(levels[1], levels[0])
    return _ROOF_WALKS[key]


def _jump(effect: np.ndarray) -> np.ndarray:
    """(bands,) how much the middle of three 10 cm steps of ``effect`` (4 points, bands)
    differs from the mean of the steps either side of it: a gradient gives none."""
    steps = np.diff(effect, axis=0)
    return np.abs(steps[1] - 0.5 * (steps[0] + steps[2]))


#: How much the weather's effect may jump (dB, in any band, octave or third octave) in
#: a 10 cm step across a roof edge's straight shadow boundary, of the direct sound
#: (x = 47.5 to 47.6) or of the ground reflection (62.5 to 62.6), from a point 1 m over
#: the roof (20 m along the x axis), against the mean of the steps either side. With
#: the building inside the parabolic equation there, whose effect scales the paths in
#: the plane on both sides, it jumps by at most 0.04 dB (0.012 dB in octave bands);
#: mixing bent rays with the
#: straight paths over the edge, and the open ground's equation where the reflection
#: came into view, it jumped by up to 1.71 dB at 500 Hz and 3.11 dB at 400 Hz in third
#: octaves.
ROOF_JUMP = 0.1


#: The weather's effect (dB, 63 to 500 Hz) 27.5 and 42.5 m from a source 1 m over a
#: roof 6 m up, 1.5 m up, downwind (0) and upwind (180) in the ``windy`` preset: the
#: full-wave solution of ``validation/barrier_wind.py`` (``full_wave_levels``, a rigid
#: ring block from the axis to 5 m, its roof on the solution's cells, rigid ground).
ROOFTOP_FULL_WAVE = {
    0.0: [[-0.011, -0.483, 1.079, -0.086], [0.092, -0.376, -2.089, 1.345]],
    180.0: [[0.003, 0.46, -1.227, 0.151], [-0.127, 0.266, 2.36, -2.904]],
}


#: Points on the x axis, 1.5 m up: across the straight shadow boundary of the roof
#: edge's direct sound and of its ground reflection, and close behind the edge in its
#: shadow (12.8 and 16.3 m from the source over the roof).
DIRECT_WALK = [47.4, 47.5, 47.6, 47.7]
REFLECTION_WALK = [62.4, 62.5, 62.6, 62.7]
NEAR_EDGE = [32.8, 36.3]


@pytest.mark.parametrize("bands", ["octave", "third_octave"])
@pytest.mark.parametrize("towards", [0.0, 180.0])
def test_a_source_on_a_roof_in_wind_has_no_strips_at_the_edge_s_shadows(
    bent, towards, bands
) -> None:
    """Listeners 1.5 m up walked in 10 cm steps across the straight shadow boundaries
    of the roof's edge, from a source on the roof, down- and upwind; at the boundaries,
    from 63 to 500 Hz, the weather's effect is within 0.3 dB of the full-wave solution's
    (0.12 dB here; 0.91 dB off where the equation took no building). In third octaves
    the ground reflection's boundary alone, where the open ground's equation took over
    (3.11 dB); across the direct sound's they jump by 0.014 dB (a one-off check)."""
    if bands == "octave":
        near = NEAR_EDGE if towards == 0.0 else []  # for the test after this one
        effect = _roof_walk(bent, "source", towards, bands, near + DIRECT_WALK + REFLECTION_WALK)
        effect = effect[len(near) :]
        assert np.all(_jump(effect[:4]) < ROOF_JUMP)
        np.testing.assert_allclose(effect[[1, 5], :4], ROOFTOP_FULL_WAVE[towards], atol=0.3)
        effect = effect[4:]
    else:
        effect = _roof_walk(bent, "source", towards, bands, REFLECTION_WALK)
    assert np.all(_jump(effect) < ROOF_JUMP)


@pytest.mark.parametrize("bands", ["octave", "third_octave"])
@pytest.mark.parametrize("towards", [0.0, 180.0])
def test_a_listener_on_a_roof_in_wind_has_no_strips_at_the_edge_s_shadows(
    bent, towards, bands
) -> None:
    """Sources 1.5 m up walked in 10 cm steps across the straight shadow boundary of
    the ground reflection off the roof's edge, to a listener on the roof (the direct
    sound's boundary, 27 m away where the wide-angle equation runs for each source, is
    the source's walk's)."""
    effect = _roof_walk(bent, "listener", towards, bands, [62.4, 62.5, 62.6, 62.7])
    assert np.all(_jump(effect) < ROOF_JUMP)


#: The weather's effect (dB, 63 to 500 Hz) 12.8 and 16.3 m downwind of the source over
#: the roof, 1.5 m up: the full-wave solution as for ``ROOFTOP_FULL_WAVE``.
NEAR_EDGE_FULL_WAVE = [[0.022, 0.078, 0.067, 0.073], [-0.021, 0.200, 0.061, 0.192]]
#: How much the weather may change the sound (dB) close behind the edge above 500 Hz:
#: the effect of the travel times it changes, along the edge's straight leg to the
#: listener and the one off the ground, at most 0.092 dB in any band 12.5 to 16.3 m
#: away (by hand: each leg's travel time to first order in the profile, their band
#: energy averaged over 4001 frequencies), and the rest the full-wave solution shows
#: from 63 to 500 Hz beyond that, at most 0.112 dB (500 Hz, 16.3 m). At 4 kHz the runs'
#: 16 frequencies take up to 0.21 dB more (``BARRIER_SAMPLES``); 8 kHz, sampled 16
#: times here, is left out.
NEAR_EDGE_BOUND = 0.092 + 0.112


@pytest.mark.parametrize("equation", ["narrow_angle", "wide_angle"])
def test_close_behind_a_roof_s_edge_the_weather_changes_little(bent, equation) -> None:
    """Listeners 1.5 m up 12.8 and 16.3 m downwind of a source 1 m over the roof, in
    its edge's shadow, where the sound goes down steeply behind the edge: the wide-angle
    equation, with the roof inside, gives them in either setting. From 63 to 500 Hz the
    weather's effect is within 0.05 dB of the full-wave solution's (0.032 dB here;
    0.12 dB off, at 500 Hz 16.3 m away, with the open ground's rule there), and above it within
    ``NEAR_EDGE_BOUND`` (0.13 dB at 1 kHz; 0.33 dB at 4 kHz 16.3 m away, 0.25 dB with 64
    frequencies; the narrow-angle march there was 1.2 dB off at 2 kHz)."""
    if equation == "narrow_angle":  # the default: the source's walk's runs
        xs = NEAR_EDGE + DIRECT_WALK + REFLECTION_WALK
    else:
        xs = NEAR_EDGE
    effect = _roof_walk(bent, "source", 0.0, "octave", xs, equation)[: len(NEAR_EDGE)]
    np.testing.assert_allclose(effect[:, :4], NEAR_EDGE_FULL_WAVE, atol=0.05)
    assert np.all(np.abs(effect[:, 4:6]) < NEAR_EDGE_BOUND)
    assert np.all(np.abs(effect[:, 6]) < NEAR_EDGE_BOUND + 0.21)


@pytest.mark.parametrize("towards", [0.0, 180.0])
def test_a_model_file_s_wall_in_wind_has_no_strips_at_its_shadows(towards, tmp_path) -> None:
    """A wall of the model file 4 m high and 0.3 m thick 15 m from a source 8 m up:
    listeners walked in 10 cm steps across the straight shadow boundaries of its top, of
    the direct sound (x = 24.86) and of the ground reflection (x = 36.3)."""
    wall = tmp_path / "wall.obj"
    corners = [(x, y, z) for z in (0, 4) for y in (-40, 40) for x in (15, 15.3)]
    sides = [(1, 2, 4, 3), (5, 6, 8, 7), (1, 2, 6, 5), (3, 4, 8, 7), (1, 3, 7, 5), (2, 4, 8, 6)]
    wall.write_text(
        "g wall\n"
        + "".join(f"v {x} {y} {z}\n" for x, y, z in corners)
        + "".join("f " + " ".join(map(str, s)) + "\n" for s in sides)
    )
    direct, reflection = np.arange(24.4, 25.45, 0.1), np.arange(35.8, 36.85, 0.1)
    xs = np.round(np.concatenate([direct, reflection]), 2)
    effect = _walked(
        [("s", f"l{k}") for k in range(len(xs))],
        towards,
        sources=((0.0, 0.0, 8.0),),
        listeners=[(x, 0.0, 1.5) for x in xs],
        file=wall,
    )
    assert np.all(_steps(effect[: len(direct)]) < DIRECT_STEP)
    assert np.all(_steps(effect[len(direct) :]) < REFLECTION_STEP)


def test_a_listener_on_a_barrier_s_line_keeps_straight_legs() -> None:
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


@pytest.mark.parametrize(
    "atmosphere",
    [None, {"preset": "windy", "wind": {"direction_deg": 180}}],
    ids=["still", "upwind"],
)
def test_the_level_changes_smoothly_past_a_barrier_s_end(atmosphere) -> None:
    """Behind a barrier, 4 cm apart either side of where the direct sound comes round its
    end: the paths over its top fade out past the end instead of stopping, in still air
    and upwind, where they take the weather's effect just inside the end."""
    listeners = [(120.0, -0.02, 1.5), (120.0, 0.02, 1.5)]
    barrier = {"name": "screen", "path": [[15, -20], [15, 0]], "height": 4}
    room = Room.from_scene(
        scene(
            atmosphere=atmosphere,
            listeners=listeners,
            barriers=[barrier],
            extent=((-60.0, -60.0), (150.0, 60.0)),
            ir_length_s=0.6,
        )
    )
    result = simulate(room)
    levels = [10.0 * np.log10(total_energy(result[("s", f"l{k}")].early)) for k in range(2)]
    assert np.max(np.abs(levels[1] - levels[0])) < 1.0


#: A building whose corner (15, 10) is in line with the source and (60, 40).
CORNER = {"name": "h", "footprint": [[15, 0], [25, 0], [25, 10], [15, 10]], "height": 6}


@pytest.mark.parametrize(
    ("source", "line", "obstacles"),
    [
        # The way seen from above misses the barrier's end, rounded: it meets the
        # barrier's line at 1 + 2e-16 of the barrier's length.
        (
            (0.0, 0.0),
            (60.0, 40.0),
            {"barriers": [{"name": "screen", "path": [[15, -20], [15, 10]], "height": 4}]},
        ),
        # Rounded the other way: the way seen from above crosses the barrier at its end,
        # but the edge along its top, as the model holds it, ends 4e-15 m short of it.
        (
            (3.0, -2.0),
            (71.0, 28.0),
            {"barriers": [{"name": "screen", "path": [[24, -24], [20, 5.5]], "height": 4}]},
        ),
        # A corner, with no path over an edge in the plane: the line keeps the straight
        # legs, as the shadow beside it does, and points within the tolerance past it
        # keep the lit side's level.
        ((0.0, 0.0), (60.0, 40.0), {"barriers": (), "buildings": [CORNER]}),
        # Another barrier across the way beyond the end cuts the direct sound off on
        # both sides: the end's own barrier, not the direct sound, tells the side.
        (
            (0.0, 0.0),
            (60.0, 40.0),
            {
                "barriers": [
                    {"name": "screen", "path": [[15, -20], [15, 10]], "height": 4},
                    {"name": "wide", "path": [[57.5, -5], [17.5, 55]], "height": 4},
                ]
            },
        ),
    ],
    ids=["barrier", "barrier-rounded-across", "building", "barrier-and-another"],
)
def test_the_level_in_line_with_a_barrier_s_end_is_that_beside_it(source, line, obstacles) -> None:
    """Downwind, a listener exactly in line with the source and a barrier's end (or a
    building's corner), 1 mm either side and 10 µm past it, within the tolerance the
    paths over edges are found to: from 63 to 500 Hz the level on the line is that just
    behind it, the side whose paths it has (the line meets the barrier), and the level
    10 µm past, where it does not, that 1 mm past, within 0.05 dB; with the barrier
    alone, on the line and past its end within 1.0 dB (the step across the end's
    shadow boundary, see test_the_level_changes_smoothly_past_a_barrier_s_end). Fewer
    frequencies at 4 and 8 kHz, which are not compared, keep it quick."""
    # The barrier or building whose end or corner is in line stands to the right of the
    # way, seen from above: behind is right.
    way = np.subtract(line, source)
    left = np.array([-way[1], way[0]]) / np.hypot(*way)
    listeners = [(*np.add(line, d * left), 1.5) for d in (-0.001, 0.0, 1e-5, 0.001)]
    room = Room.from_scene(
        scene(
            listeners=listeners,
            sources=((*source, 1.0),),
            barrier_frequencies=8,
            **obstacles,
        )
    )
    behind, on, near, past = (
        10.0 * np.log10(total_energy(early_paths(room, listener=f"l{k}")))[:4] for k in range(4)
    )
    np.testing.assert_allclose(on, behind, atol=0.05)
    np.testing.assert_allclose(near, past, atol=0.05)
    if "buildings" not in obstacles and len(obstacles["barriers"]) == 1:
        np.testing.assert_allclose(on, past, atol=1.0)
