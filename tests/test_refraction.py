"""Wind and temperature outdoors: the atmosphere, curved rays, the parabolic equation."""

import importlib
import math

import numpy as np
import pytest
from pydantic import ValidationError
from test_ground import exact_reflection

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine.parabolic import march
from acoustic_spaces.engine.refraction import Profile, Rays, node_heights
from acoustic_spaces.engine.shadow import shadow_start
from acoustic_spaces.examples import example_path
from acoustic_spaces.materials.ground import surface_impedance
from acoustic_spaces.scene import Atmosphere

late = importlib.import_module("acoustic_spaces.engine.late")
C = 343.2


def field(atmosphere=None, listeners=((100.0, 0.0, 1.5),), **simulation) -> Scene:
    """A source 2 m up over a lawn, and listeners."""
    data = {
        "name": "field",
        "geometry": {"ground": {"extent": [[-150.0, -20.0], [150.0, 20.0]], "sky": 20.0}},
        "materials": {"lawn": {"impedance": "grass"}},
        "assignments": {"ground": "lawn"},
        "sources": [{"name": "s", "position": [0.0, 0.0, 2.0]}],
        "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
        "simulation": {
            "seed": 1,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
            "late": {"rays": 300},
            "ir_length_s": 0.1,
            **simulation,
        },
    }
    if atmosphere is not None:
        data["atmosphere"] = atmosphere
    return Scene.model_validate(data)


# -- the atmosphere ------------------------------------------------------------------


def test_presets_and_their_fields() -> None:
    windy = Atmosphere.model_validate({"preset": "windy", "wind": {"direction_deg": 90}})
    assert windy.wind.speed_m_s == 5.0 and windy.wind.direction_deg == 90.0
    assert windy.wind.height_m == 10.0 and not windy.still
    assert Atmosphere.model_validate({"preset": "neutral"}).still
    sunny = Atmosphere.model_validate({"preset": "sunny_day"})
    assert sunny.temperature(10.0, 20.0) < 20.0 and sunny.wind is None
    night = Atmosphere.model_validate({"preset": "clear_night"})
    assert night.temperature(10.0, 20.0) > 20.0
    stronger = Atmosphere.model_validate({"preset": "windy", "wind": {"speed_m_s": 8}})
    assert stronger.wind.speed_m_s == 8.0
    with pytest.raises(ValidationError, match="did you mean windy"):
        Atmosphere.model_validate({"preset": "windyy"})


def test_the_speed_of_sound_with_height() -> None:
    windy = Atmosphere.model_validate({"preset": "windy"})
    still = windy.sound_speed(10.0, 20.0)
    assert windy.wind_speed(10.0) == pytest.approx(5.0)
    assert windy.sound_speed(10.0, 20.0, np.array([1.0, 0.0])) == pytest.approx(still + 5.0)
    assert windy.sound_speed(10.0, 20.0, np.array([-1.0, 0.0])) == pytest.approx(still - 5.0)
    assert windy.sound_speed(10.0, 20.0, np.array([0.0, 1.0])) == pytest.approx(still)
    assert windy.wind_speed(0.0) == 0.0
    linear = Atmosphere.model_validate({"wind": {"speed_m_s": 4, "profile": "linear"}})
    assert linear.wind_speed(5.0) == pytest.approx(2.0)
    constant = Atmosphere.model_validate({"wind": {"speed_m_s": 4, "profile": "constant"}})
    assert constant.wind_speed(50.0) == pytest.approx(4.0)
    table = Atmosphere.model_validate({"table": [[0, 15, 0], [10, 14, 3]]})
    assert table.temperature(5.0, 20.0) == pytest.approx(14.5)
    assert table.wind_speed(20.0) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("atmosphere", "message"),
    [
        ({"table": [[0, 15, 0]]}, "two rows or more"),
        ({"table": [[0, 15, 0], [0, 14, 1]]}, "two rows or more"),
        ({"table": [[0, 15, 0], [10, 14, -1]]}, "not negative"),
        ({"table": [[0, 15, 0], [10, 14, 1]], "temperature_log_k": 0.5}, "not both"),
    ],
)
def test_atmospheres_that_are_refused(atmosphere, message) -> None:
    with pytest.raises(ValidationError, match=message):
        Atmosphere.model_validate(atmosphere)


def test_rooms_have_no_atmosphere() -> None:
    data = Scene.from_yaml(example_path("shoebox")).to_data()
    data["atmosphere"] = {"preset": "windy"}
    with pytest.raises(ValidationError, match="atmosphere is for outdoor scenes"):
        Scene.model_validate(data)


@pytest.mark.parametrize(
    "atmosphere",
    [
        {"table": [[0, 15, 0], [10, 14, 3]]},
        {"preset": "sunny_day", "table": [[0, 15, 0], [10, 14, 3]]},
        {"preset": "windy", "wind": {"direction_deg": 30}},
    ],
)
def test_an_atmosphere_is_written_as_it_reads(atmosphere) -> None:
    scene = field(atmosphere)
    again = Scene.model_validate(scene.to_data())
    assert again.atmosphere == scene.atmosphere


# -- curved rays ---------------------------------------------------------------------


def linear(g: float, c0: float = 340.0) -> Profile:
    """c = c0 + g z, exactly linear between the usual nodes."""
    z = node_heights(None)
    return Profile(z, c0 + g * z)


def test_in_a_uniform_atmosphere_rays_are_straight() -> None:
    found = Rays(Profile.uniform(C), 2.0).eigenrays(1.5, 50.0, 1)
    assert len(found) == 2 and list(found.ground_bounces) == [0, 1]
    for i, image in enumerate((2.0 - 1.5, 2.0 + 1.5)):
        distance = math.hypot(50.0, image)
        assert found.time[i] == pytest.approx(distance / C, rel=1e-9)
        assert found.length[i] == pytest.approx(distance, rel=1e-9)
        assert found.spreading[i] * distance**2 == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("g", [0.1, -0.05])
def test_rays_in_a_linear_profile_are_arcs_with_exact_times(g) -> None:
    profile, zs, zr = linear(g), 2.0, 1.5
    found = Rays(profile, zs).eigenrays(zr, 120.0, 0)
    assert len(found) == 1
    cs, cr = float(profile.speed(zs)), float(profile.speed(zr))
    # Along an arc of a linear profile: t = acosh(1 + g^2 d^2 / (2 c1 c2)) / |g|.
    d2 = 120.0**2 + (zs - zr) ** 2
    assert found.time[0] == pytest.approx(math.acosh(1 + g * g * d2 / (2 * cs * cr)) / abs(g))
    # Snell's law: the horizontal slowness is the same at both ends.
    p = found.p[0]
    assert math.cos(found.launch[0]) / cs == pytest.approx(p, rel=1e-9)
    assert math.cos(found.arrival[0]) / cr == pytest.approx(p, rel=1e-9)


def test_downwind_more_paths_upwind_a_shadow() -> None:
    windy = Atmosphere.model_validate({"preset": "windy"})
    down = Profile.of(windy, 20.0, np.array([1.0, 0.0]))
    up = Profile.of(windy, 20.0, np.array([-1.0, 0.0]))
    still = Profile.uniform(C)
    assert len(Rays(still, 5.0).eigenrays(1.5, 600.0, 3)) == 2
    assert len(Rays(down, 5.0).eigenrays(1.5, 600.0, 3)) >= 4
    start = shadow_start(up, 5.0, 1.5)
    assert start is not None and shadow_start(down, 5.0, 1.5) is None
    rays = Rays(up, 5.0)
    assert len(rays.eigenrays(1.5, 0.9 * start, 3)) >= 1
    # Past where the ray grazing the ground comes up, rays that turn just above it carry
    # on a little: the wind's log profile bends them less the higher they turn.
    assert len(rays.eigenrays(1.5, 2.0 * start, 3)) == 0


def test_the_level_of_rays_that_turn_in_the_air() -> None:
    """Against an independent ray tracer (fourth-order Runge-Kutta in 2 cm steps) on the
    smooth profile, 5 m/s of wind 10 m up, from 5 m up to 1.5 m up downwind: the level
    over a straight path's (dB). Taken from the layers' local slopes alone, the 300 m
    paths came out 2.2 dB too low."""
    windy = Atmosphere.model_validate({"preset": "windy"})
    rays = Rays(Profile.of(windy, 20.0, np.array([1.0, 0.0])), 5.0)
    for distance, bounces, reference in ((100.0, 0, -1.34), (300.0, 0, -3.76), (300.0, 1, -3.07)):
        found = rays.eigenrays(1.5, distance, 1)
        i = list(found.ground_bounces).index(bounces)
        level = 10 * math.log10(found.spreading[i] * found.length[i] ** 2)
        assert level == pytest.approx(reference, abs=0.2)


@pytest.mark.parametrize(("g", "angle"), [(0.1, 5.0), (-0.1, -3.0), (0.02, 30.0)])
def test_the_compiled_tracer_follows_the_arcs(g, angle) -> None:
    native = pytest.importorskip("acoustic_spaces._native")
    z = node_heights(None)
    rows = np.column_stack([z, 340.0 + g * z, np.zeros_like(z)])
    zs, theta0 = 2.0, math.radians(angle)
    p = math.cos(theta0) / (340.0 + g * zs)
    position = np.array([[0.0, 0.0, zs]])
    direction = np.array([[math.cos(theta0), 0.0, math.sin(theta0)]])
    arc = time = 0.0
    while arc < 400.0:
        end, direction, length, seconds, _ = native.bent_chords(
            position, direction, np.array([400.0 - arc]), np.zeros((1, 3)), rows,
            np.array([1.0, 0.0]), 0.0, 0.02,
        )  # fmt: skip
        arc, time, position = arc + length[0], time + seconds[0], end
        theta = theta0 - g * p * arc
        x = (math.sin(theta0) - math.sin(theta)) / (g * p)
        height = zs + (math.cos(theta) - math.cos(theta0)) / (g * p)
        if not 0.0 < height < 1000.0:
            break
        assert end[0, 0] == pytest.approx(x, abs=1e-6)
        assert end[0, 2] == pytest.approx(height, abs=1e-6)
        exact = (math.atanh(math.sin(theta0)) - math.atanh(math.sin(theta))) / g
        assert time == pytest.approx(exact, rel=1e-9)


# -- the parabolic equation ------------------------------------------------------------


@pytest.mark.parametrize("f", [125.0, 500.0])
def test_the_parabolic_equation_against_the_exact_solution(f) -> None:
    """In still air over grass, from 1 m up to 1.5 m up, 30 and 100 m away."""
    zs, zr, ranges = 1.0, 1.5, np.array([30.0, 100.0])
    k = 2 * math.pi * f / C
    z = complex(surface_impedance(np.array(f), 200.0))
    pressure = march(
        [f], lambda h: np.full(np.shape(h), C), lambda r: np.conj(z), zs, ranges,
        np.full(2, zr), 25.0,
    )[0]  # fmt: skip
    for i, r in enumerate(ranges):
        r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
        q = exact_reflection(1.0 / z, k, r, zs, zr)
        exact = abs((np.exp(-1j * k * r1) / r1 + q * np.exp(-1j * k * r2) / r2) * r1)
        level = 20 * math.log10(abs(pressure[i]) * r1) - 20 * math.log10(exact)
        assert level == pytest.approx(0.0, abs=0.3)


@pytest.mark.parametrize("f", [125.0, 250.0, 500.0])
def test_the_parabolic_equation_starts_wide(f) -> None:
    """Greene's starting field: the direct sound 24 degrees up, and its reflection off a
    rigid ground 35 degrees up, within 0.2 dB."""
    r = 40.0
    height = 5.0 + r * math.tan(math.radians(24.0))
    pressure = march(
        [f], lambda h: np.full(np.shape(h), C), lambda r: None, 5.0, np.array([r]),
        np.array([height]), 60.0,
    )[0, 0]  # fmt: skip
    # Over a rigid ground: the source and its image.
    k = 2 * math.pi * f / C
    r1, r2 = math.hypot(r, height - 5.0), math.hypot(r, height + 5.0)
    exact = abs(np.exp(1j * k * r1) / r1 + np.exp(1j * k * r2) / r2)
    assert 20 * math.log10(abs(pressure) / exact) == pytest.approx(0.0, abs=0.2)


# -- simulations -----------------------------------------------------------------------


def test_still_air_changes_nothing() -> None:
    listeners = ((60.0, 0.0, 1.5), (-60.0, 5.0, 1.5))
    plain = simulate(Room.from_scene(field(None, listeners)))
    neutral = simulate(Room.from_scene(field({"preset": "neutral"}, listeners)))
    calm = simulate(Room.from_scene(field({"wind": {"speed_m_s": 0}}, listeners)))
    for other in (neutral, calm):
        for key in plain.responses:
            a, b = plain[key], other[key]
            assert a.direct_delay_s == b.direct_delay_s
            assert np.array_equal(a.early.delay_s, b.early.delay_s)
            assert np.array_equal(a.early.energy, b.early.energy)
            assert np.array_equal(a.late, b.late)


def test_wind_bends_the_sound() -> None:
    listeners = ((120.0, 0.0, 1.5), (-120.0, 0.0, 1.5))
    windy = simulate(Room.from_scene(field({"preset": "windy"}, listeners)))
    still = simulate(Room.from_scene(field(None, listeners)))
    straight = math.hypot(120.0, 0.5) / Room.from_scene(field()).speed_of_sound()
    assert windy["s", "l0"].direct_delay_s < straight - 1e-3  # downwind: sooner
    assert windy["s", "l1"].direct_delay_s >= straight - 1e-9  # upwind: later, or in shadow
    assert still["s", "l1"].direct_delay_s == pytest.approx(straight)
    # Upwind 120 m away the listener is in the shadow: the high bands are far quieter.
    down = 10 * np.log10(windy["s", "l0"].energy().sum(axis=1))
    up = 10 * np.log10(windy["s", "l1"].energy().sum(axis=1))
    assert np.all(up[5:] < down[5:] - 10.0)


def test_the_parabolic_equation_runs_the_same_on_any_number_of_threads() -> None:
    listeners = ((80.0, 0.0, 1.5), (60.0, 15.0, 1.5), (-70.0, 10.0, 1.5))
    results = [
        simulate(Room.from_scene(field({"preset": "windy"}, listeners, performance={"threads": n})))
        for n in (1, 3)
    ]
    for key in results[0].responses:
        assert np.array_equal(results[0][key].early.energy, results[1][key].early.energy)


def test_uniform_rows_trace_the_late_rays_as_straight_ones(monkeypatch) -> None:
    """The compiled tracer's bent rays, in an atmosphere where they go straight, give
    the straight tracer's late energy: the same ray paths, times and losses."""
    pytest.importorskip("acoustic_spaces._native")
    scene = field(None, ((60.0, 0.0, 1.5), (-30.0, 10.0, 3.0)))
    room = Room.from_scene(scene)
    straight = simulate(room)
    c = room.speed_of_sound()

    def uniform(model):
        return {
            "atmosphere": np.array([[0.0, c, 0.0], [1000.0, c, 0.0]]),
            "wind_direction": np.array([1.0, 0.0]),
            "ground": float(model.ground),
            "chord_sagitta": 0.02,
        }

    monkeypatch.setattr(late, "bending", uniform)
    bent = simulate(room)
    for key in straight.responses:
        np.testing.assert_allclose(bent[key].late, straight[key].late, rtol=1e-9, atol=0.0)


def test_bent_late_rays_stay_on_the_cpu() -> None:
    from acoustic_spaces.engine.ray_device import _unsupported

    values = {"atmosphere": np.zeros((2, 3)), "transmission": np.zeros(0), "groups": 0}
    assert _unsupported(values) == "rays bent by the atmosphere are traced on the CPU"


def test_particles_bend_and_stay_put_in_still_air() -> None:
    from acoustic_spaces.engine.particles import trace_particles

    still = trace_particles(Room.from_scene(field()), count=200, duration=0.2, frames=5)
    neutral = trace_particles(
        Room.from_scene(field({"preset": "neutral"})), count=200, duration=0.2, frames=5
    )
    assert np.array_equal(still.positions, neutral.positions)
    windy = trace_particles(
        Room.from_scene(field({"preset": "windy"})), count=200, duration=0.2, frames=5
    )
    assert np.all(np.isfinite(windy.positions)) and np.all(np.isfinite(windy.energy))
    assert not np.allclose(windy.positions, still.positions)
