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


def test_sources_alike_share_the_parabolic_equation_s_runs(monkeypatch) -> None:
    """Sources at one height over one ground (the places of a walk) asked to share the
    parabolic equation's runs take fewer of them and get what runs of their own give."""
    from acoustic_spaces.engine import parabolic

    scene = field({"preset": "windy"}, ((80.0, 0.0, 1.5), (-60.0, 20.0, 1.5)))
    item = scene.sources[0]
    places = ((0.0, 0.0), (-30.0, 5.0), (25.0, -8.0))
    sources = [
        item.model_copy(update={"name": f"s{k}", "position": (x, y, 2.0)})
        for k, (x, y) in enumerate(places)
    ]
    room = Room.from_scene(scene.model_copy(update={"sources": sources}))
    runs = []
    march_ = parabolic.march

    def counted(*args, **kwargs):
        runs.append(kwargs.get("independent", False))
        return march_(*args, **kwargs)

    monkeypatch.setattr(parabolic, "march", counted)
    separate = simulate(room)
    alone = len(runs)
    runs.clear()
    shared = simulate(room, share_waves=[s.name for s in sources])
    assert 0 < len(runs) < alone and all(runs)
    for key in separate.responses:
        a, b = (r[key].early.energy.sum(axis=0) for r in (separate, shared))
        np.testing.assert_allclose(10 * np.log10(b / a), 0.0, atol=0.01)


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


def test_bent_late_rays_are_traced_on_any_device() -> None:
    from acoustic_spaces.engine.ray_device import _unsupported

    values = {"atmosphere": np.zeros((2, 3)), "transmission": np.zeros(0), "groups": 0}
    assert _unsupported(values) == ""


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


def test_the_parabolic_equation_s_bands_read_back_as_computed() -> None:
    """Downwind, the bands the parabolic equation gives sit on one path of the family
    and the others are silent in them: the level relative to free field in those bands
    is the parabolic equation's, the silent paths' higher bands leaking nothing in."""
    windy = {
        "table": [[0.0, 19.6, 0.0], [100.0, 19.6, 10.0], [1000.0, 19.6, 10.0]],
        "wind": {"direction_deg": 0.0},
    }
    data = field(windy, ((100.0, 0.0, 1.0),)).model_dump()
    data["materials"]["lawn"]["scattering"] = 0.0  # no late sound off the ground
    result = simulate(Room.from_scene(Scene.model_validate(data)))
    response = result["s", "l0"]
    early = response.early
    wave_bands = 4  # 63 to 500 Hz
    carrying = np.flatnonzero(early.energy[:, :wave_bands].sum(axis=1) > 0.0)
    assert len(carrying) == 1 and len(early) > 1
    silent = np.flatnonzero(early.energy[:, :wave_bands].sum(axis=1) == 0.0)
    assert np.all(early.energy[silent, wave_bands:] > 0.0)  # the rays' bands are theirs
    computed = 10.0 * np.log10(
        early.energy[carrying[0], :wave_bands] / response.free_field[:wave_bands]
    )
    read = response.relative_to_free_field()[:wave_bands]
    np.testing.assert_allclose(read, computed, atol=0.02)


# -- weather zones -----------------------------------------------------------------------


def zoned(zones, atmosphere=None, listeners=((300.0, 0.0, 1.5), (-300.0, 0.0, 1.5))) -> Scene:
    """A windy field with weather zones, listeners 300 m downwind and upwind."""
    data = field({"preset": "windy", **(atmosphere or {}), "zones": zones}, listeners).model_dump(
        mode="json", exclude_none=True
    )
    data["geometry"]["ground"]["extent"] = [[-400.0, -50.0], [400.0, 50.0]]
    return Scene.model_validate(data)


#: A zone over the downwind half of the field.
DOWNWIND_HALF = [[100.0, -100.0], [500.0, -100.0], [500.0, 100.0], [100.0, 100.0]]


def test_a_weather_zone_takes_the_atmosphere_s_weather_but_for_what_it_gives() -> None:
    atmosphere = Atmosphere(
        wind={"speed_m_s": 5.0, "direction_deg": 90.0},
        temperature_log_k=0.3,
        zones=[
            {"name": "lee", "polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 1.0}},
            {"polygon": DOWNWIND_HALF, "preset": "sunny_day"},
            {"polygon": DOWNWIND_HALF, "preset": "windy", "temperature_c": 12.0},
            {"polygon": DOWNWIND_HALF, "preset": "neutral"},
            {"polygon": DOWNWIND_HALF, "table": [[0.0, 10.0, 0.0], [10.0, 12.0, 3.0]]},
            {"polygon": DOWNWIND_HALF, "temperature_gradient_k_m": -0.01},
        ],
    )
    own, lee, sunny, windy, neutral, table, gradient = atmosphere.weathers()
    assert own.zones is None and own.wind.speed_m_s == 5.0 and own.temperature_log_k == 0.3
    # Fields left out stay the atmosphere's: the wind's direction, the log term.
    assert lee.wind.speed_m_s == 1.0 and lee.wind.direction_deg == 90.0
    assert lee.temperature_log_k == 0.3 and lee.roughness_m == 0.05
    # A preset is a weather whole, its wind blowing the atmosphere's way.
    assert sunny.wind is None and sunny.temperature_log_k == -0.5
    assert windy.wind.speed_m_s == 5.0 and windy.wind.direction_deg == 90.0
    assert windy.temperature_log_k == 0.0 and neutral.still
    # A table replaces the formulas and keeps the atmosphere's wind direction; a
    # gradient drops the table.
    assert table.table is not None and table.temperature_log_k == 0.0
    assert table.wind.direction_deg == 90.0
    assert gradient.table is None and gradient.temperature_gradient_k_m == -0.01
    assert gradient.temperature_log_k == 0.3
    assert atmosphere.ground_temperatures(20.0) == [20.0, 20.0, 20.0, 12.0, 20.0, 20.0, 20.0]
    assert atmosphere.zone_names() == ["lee"] + [f"weather_{k}" for k in range(2, 7)]
    assert not atmosphere.still
    # Still air with a windy zone is not still; a zone that gives nothing changes nothing.
    assert not Atmosphere(zones=[{"polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 2}}]).still
    assert Atmosphere(zones=[{"polygon": DOWNWIND_HALF}]).still
    # Written and read back, a zone keeps what it gives and nothing more.
    data = atmosphere.model_dump(mode="json", exclude_none=True)
    assert data["zones"][0] == {"name": "lee", "polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 1.0}}
    assert Atmosphere.model_validate(data) == atmosphere
    with pytest.raises(ValidationError, match="did you mean windy"):
        Atmosphere(zones=[{"polygon": DOWNWIND_HALF, "preset": "wind"}])
    with pytest.raises(ValidationError, match="not both"):
        Atmosphere(
            zones=[
                {
                    "polygon": DOWNWIND_HALF,
                    "table": [[0.0, 10.0, 0.0], [10.0, 12.0, 3.0]],
                    "temperature_log_k": 0.2,
                }
            ]
        )
    with pytest.raises(ValidationError, match="not modelled yet"):
        data = field(
            {"preset": "neutral", "zones": [{"polygon": DOWNWIND_HALF, "preset": "windy"}]}
        )
        data = data.model_dump(mode="json", exclude_none=True)
        data["geometry"]["ground"]["terrain"] = {"heights": [[0, 1], [1, 2]], "spacing": 300}
        Scene.model_validate(data)


def test_the_weathers_along_a_way_and_their_mean() -> None:
    from acoustic_spaces.engine.weather import Weather

    atmosphere = Atmosphere(
        preset="windy",
        zones=[{"name": "lee", "polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 1.0}}],
    )
    weather = Weather(atmosphere, 20.0, (-400.0, -50.0, 400.0, 50.0))
    assert len(weather) == 2 and weather.zoned
    assert weather.zone_at([[0.0, 0.0], [200.0, 10.0], [450.0, 0.0]]).tolist() == [0, 1, 0]
    # A way 300 m downwind: a third under the atmosphere's wind, two thirds in the lee.
    a, b = weather.shares([[0.0, 0.0], [300.0, 0.0]])
    np.testing.assert_allclose(a, [1 / 3, 2 / 3], atol=0.01)
    np.testing.assert_allclose(b, [1 / 3, 2 / 3], atol=0.01)  # both winds blow along it
    profile = weather.profile(a, b)
    expected = (a[0] * weather.still[0] + a[1] * weather.still[1]) + (
        b[0] * weather.wind[0] + b[1] * weather.wind[1]
    )
    np.testing.assert_allclose(profile.c, expected)
    assert profile.speed(10.0) == pytest.approx(343.2 + 5.0 / 3 + 1.0 * 2 / 3, abs=0.3)
    # Across the wind the wind's share is nothing; a way of no length takes its point.
    a, b = weather.shares([[200.0, -40.0], [200.0, 40.0]])
    np.testing.assert_allclose(a, [0.0, 1.0]) and np.testing.assert_allclose(b, [0.0, 0.0])
    a, _ = weather.shares([[200.0, 0.0]])
    assert a.tolist() == [0.0, 1.0]
    # The weather at each distance along a direction, the same object within a zone.
    speed_at = weather.along(np.zeros(2), np.array([1.0, 0.0]), 300.0)
    assert speed_at(50.0) is speed_at(99.0) and speed_at(150.0) is speed_at(299.0)
    assert speed_at(50.0) is not speed_at(150.0)
    assert speed_at(50.0)(np.array([10.0]))[0] > speed_at(150.0)(np.array([10.0]))[0]
    # The mean weather over the ground: the lee covers three eighths of it.
    rows, towards = weather.mean_rows()
    np.testing.assert_allclose(towards, [1.0, 0.0])
    assert np.interp(10.0, rows[:, 0], rows[:, 2]) == pytest.approx(
        5.0 * 5 / 8 + 1.0 * 3 / 8, abs=0.05
    )
    with pytest.raises(ValueError, match="weather zone 'lee'"):
        Weather(
            Atmosphere(zones=[{"name": "lee", "polygon": [[0, 0], [1, 1], [1, 0], [0, 1]]}]),
            20.0,
            (-400.0, -50.0, 400.0, 50.0),
        )


def test_the_parabolic_equation_marches_through_the_weathers() -> None:
    from acoustic_spaces.engine.weather import Weather

    f, ranges, heights = [125.0], np.array([100.0, 200.0, 300.0]), np.full(3, 1.5)
    z = np.conj(surface_impedance(np.array(125.0), 200.0))
    atmosphere = Atmosphere(preset="windy")
    weather = Weather(atmosphere, 20.0, (-400.0, -50.0, 400.0, 50.0))
    speed_at = weather.along(np.zeros(2), np.array([1.0, 0.0]), 300.0)
    plain = march(f, speed_at(0.0), lambda r: z, 2.0, ranges, heights, 60.0)
    # One weather all the way: the same march, bit for bit.
    same = march(f, speed_at(0.0), lambda r: z, 2.0, ranges, heights, 60.0, speed_at=speed_at)
    assert np.array_equal(plain, same)
    # The wind dropping 150 m out tells from there on, and not before.
    lee = Atmosphere(preset="windy", zones=[{"polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 0.5}}])
    lee = lee.model_copy(
        update={
            "zones": [
                lee.zones[0].model_copy(
                    update={
                        "polygon": [
                            (150.0, -100.0),
                            (500.0, -100.0),
                            (500.0, 100.0),
                            (150.0, 100.0),
                        ]
                    }
                )
            ]
        }
    )
    along = Weather(lee, 20.0, (-400.0, -50.0, 400.0, 50.0)).along(
        np.zeros(2), np.array([1.0, 0.0]), 300.0
    )
    bent = march(f, along(0.0), lambda r: z, 2.0, ranges, heights, 60.0, speed_at=along)
    assert np.array_equal(bent[:, 0], plain[:, 0])
    assert not np.array_equal(bent[:, 1], plain[:, 1])
    assert abs(20 * math.log10(abs(bent[0, 2] / plain[0, 2]))) > 0.5


def test_weather_zones_in_a_scene() -> None:
    from acoustic_spaces.engine.model import TraceModel
    from acoustic_spaces.engine.refracted import bending
    from acoustic_spaces.engine.wave import _Weather

    plain = simulate(Room.from_scene(zoned(None)))
    # A zone that restates the weather changes nothing worth the name.
    same = simulate(
        Room.from_scene(zoned([{"polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 5.0}}]))
    )
    for name in ("l0", "l1"):
        a, b = plain["s", name].energy(), same["s", name].energy()
        np.testing.assert_allclose(10 * np.log10(b.sum(axis=1) / a.sum(axis=1)), 0.0, atol=1e-6)
    # The wind dropping over the downwind half tells downwind; upwind nothing crosses it.
    room = Room.from_scene(zoned([{"polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 0.5}}]))
    lee = simulate(room)
    down = 10 * np.log10(
        lee["s", "l0"].energy().sum(axis=1) / plain["s", "l0"].energy().sum(axis=1)
    )
    assert np.any(np.abs(down) > 0.5)
    assert np.array_equal(lee["s", "l1"].energy(), plain["s", "l1"].energy())
    assert lee["s", "l0"].direct_delay_s > plain["s", "l0"].direct_delay_s  # less wind: later
    # The late rays take the mean weather over the ground, by area (the lee: 3/8 of it).
    model = TraceModel.from_room(room)
    rows = bending(model)["atmosphere"]
    at_10 = np.interp(10.0, rows[:, 0], rows[:, 2])
    assert at_10 == pytest.approx(5.0 * 5 / 8 + 0.5 * 3 / 8, abs=0.05)
    assert np.interp(
        10.0,
        *bending(TraceModel.from_room(Room.from_scene(zoned(None))))["atmosphere"][:, [0, 2]].T,
    ) == pytest.approx(5.0, abs=0.01)
    # The wave solver's cells take the weather they stand over.
    weather = _Weather.of(model, np.array([0.0, 0.0, 2.0]))
    origin, h, shape = np.array([-10.0, -5.0, 0.0]), 10.0, (30, 1, 2)
    ratios = weather.ratios(origin, h, shape, 400.0).reshape(2, 1, 30)
    plain_weather = _Weather.of(
        TraceModel.from_room(Room.from_scene(zoned(None))), np.array([0.0, 0.0, 2.0])
    )
    plain_ratios = plain_weather.ratios(origin, h, shape, 400.0).reshape(2, 1, 30)
    xs = origin[0] + (np.arange(30) + 0.5) * h
    inside = xs > 100.0
    np.testing.assert_allclose(ratios[:, :, ~inside], plain_ratios[:, :, ~inside])
    assert np.all(ratios[1, 0, inside] < plain_ratios[1, 0, inside])  # 15 m up, less wind
    low, high = weather.extremes(origin, h, shape)
    assert (
        low <= plain_weather.extremes(origin, h, shape)[0] + 1e-9
        and high <= plain_weather.extremes(origin, h, shape)[1] + 1e-9
    )
    at = weather.at(np.array([[200.0, 0.0, 15.0], [-200.0, 0.0, 15.0]]))
    assert (
        at[0] < plain_weather.at(np.array([[200.0, 0.0, 15.0]]))[0]
        and at[1] == plain_weather.at(np.array([[-200.0, 0.0, 15.0]]))[0]
    )


def test_inspect_names_the_weather_zones(tmp_path, capsys) -> None:
    from acoustic_spaces.cli import main

    path = tmp_path / "scene.yaml"
    zoned(
        [
            {"name": "lee", "polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 1.0}},
            {"polygon": DOWNWIND_HALF, "preset": "neutral", "temperature_c": 12.0},
        ]
    ).to_yaml(path)
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "weather zone lee (4 corners): wind 1 m/s 10 m up (log); blowing towards 0 deg" in out
    assert "weather zone weather_2 (4 corners): neutral weather; still air; ground at 12 °C" in out
