"""Wind and temperature outdoors: the atmosphere, curved rays, the parabolic equation."""

import importlib
import math
import time
import warnings

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import brentq
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
    over a straight path's (dB)."""
    windy = Atmosphere.model_validate({"preset": "windy"})
    rays = Rays(Profile.of(windy, 20.0, np.array([1.0, 0.0])), 5.0)
    for distance, bounces, reference in ((100.0, 0, -1.34), (300.0, 0, -3.76), (300.0, 1, -3.07)):
        found = rays.eigenrays(1.5, distance, 1)
        i = list(found.ground_bounces).index(bounces)
        level = 10 * math.log10(found.spreading[i] * found.length[i] ** 2)
        assert level == pytest.approx(reference, abs=0.2)


def _level(found, i: int = 0) -> float:
    """A ray's level over free field at its length (dB)."""
    return 10 * math.log10(found.spreading[i] * found.length[i] ** 2)


@pytest.mark.parametrize(
    ("g", "zs", "lowest", "highest", "reference"),
    [
        (0.17, 1.0, 6.3, 6.6, -0.016),
        (-0.17, 10.0, 4.4, 4.8, -0.016),
        (0.17, 1.0, 3.6, 4.0, -0.010),
        (-0.17, 10.0, 6.0, 6.4, -0.012),
    ],
)
def test_the_level_of_a_ray_that_turns_next_to_the_listener(
    g, zs, lowest, highest, reference
) -> None:
    """In c = 340 + g z the rays are arcs of circles centred 340 / |g| = 2000 m below
    the ground (above it where g < 0). 150 m from a source 1 m up in 0.17/s, a listener
    6.3 to 6.6 m up is 0.01 to 4.4 mm below the top of the direct ray's arc, one 3.6 to
    4.0 m up 0.30 to 0.41 m below it; from 10 m up in -0.17/s, one 4.4 to 4.8 m up is
    0.09 to 8.8 mm above its bottom, one 6.0 to 6.4 m up 0.12 to 0.19 m above it. The
    ray's level over free field at its length, from the ray tube of the arc in closed
    form, is -0.016 dB (-0.0154 to -0.0163) next to the turn, -0.010 dB (-0.0097 to
    -0.0106) and -0.012 dB (-0.0120 to -0.0128) further off."""
    rays = Rays(linear(g), zs)
    for zr in np.arange(lowest, highest + 0.005, 0.01):
        found = rays.eigenrays(zr, 150.0, 0)
        assert len(found) == 1
        assert _level(found) == pytest.approx(reference, abs=0.005)


@pytest.mark.parametrize(("g", "zs", "reference"), [(0.17, 1.0, -0.0162), (-0.17, 10.0, -0.0164)])
def test_the_level_of_a_ray_that_grazes_the_listener_or_the_source(g, zs, reference) -> None:
    """150 m from a source zs up in c = 340 + g z, the direct ray turns at the
    listener's height where that is 6.614 m (0.17/s, from 1 m up) or 4.355 m (-0.17/s,
    from 10 m up). For listeners 0.0001 to 1 mm either side of it the ray's level over
    free field at its length is -0.0162 and -0.0164 dB, from the arc in closed form (to
    0.0001 dB). With the source there and the listener at zs it is (c_s / c_r)^2 higher,
    +0.0081 and +0.0082 dB, for sources 0.0001 to 3 mm either side, within 0.24 mm of
    which the ray leaves the source flatter than the first rays the search launches."""
    zc = -340.0 / g
    graze = zc + math.copysign(math.hypot(150.0, zs - zc), g)
    for offset in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3):
        found = Rays(linear(g), zs).eigenrays(graze + offset, 150.0, 0)
        assert len(found) == 1
        assert _level(found) == pytest.approx(reference, abs=0.002)
    for offset in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3, -3e-3):
        source = graze + offset
        found = Rays(linear(g), source).eigenrays(zs, 150.0, 0)
        assert len(found) == 1
        louder = 20 * math.log10((340.0 + g * source) / (340.0 + g * zs))
        assert _level(found) == pytest.approx(reference + louder, abs=0.002)


@pytest.mark.parametrize(("g", "zs"), [(0.01, 1.0), (-0.01, 10.0)])
def test_rays_that_graze_the_listener_or_the_source_in_a_weak_gradient(g, zs) -> None:
    """The same in 0.01/s (from 1 m up, the ray turning 1.331 m up) and -0.01/s (from
    10 m up, 9.669 m up), for listeners 0.01 um to 1 mm and sources 0.01 um to 0.01 mm
    either side: one ray, at p^3 c_s^2 |g| L^2 / (X s_s) over free field at its length L
    (-0.00006 dB; with the source there and the listener at zs, (c_s / c_r)^2 higher),
    from the arc in closed form. Near there the range changes by more than the search's
    tolerance from one float of p to the next (as the square root of how far p is from
    grazing), and a ray leaving the source within 0.1 mm of it is flatter than the first
    one the search launches."""
    zc = -340.0 / g
    graze = zc + math.copysign(math.hypot(150.0, zs - zc), g)

    def level(z_end: float, z_turn: float) -> float:
        p = 1.0 / (340.0 + g * z_turn)
        c = 340.0 + g * z_end
        length = math.acos(p * c) / (p * abs(g))
        return 10 * math.log10(
            p**3 * c * c * abs(g) * length**2 / (150.0 * math.sqrt(1 - (p * c) ** 2))
        )

    offsets = [k * 10.0**e for e in range(-8, -2) for k in (1, -1)]
    for offset in offsets:
        found = Rays(linear(g), zs).eigenrays(graze + offset, 150.0, 0)
        assert len(found) == 1
        assert _level(found) == pytest.approx(level(zs, graze), abs=0.0005)
    for offset in offsets[:-4]:
        found = Rays(linear(g), graze + offset).eigenrays(zs, 150.0, 0)
        assert len(found) == 1
        louder = 20 * math.log10((340.0 + g * (graze + offset)) / (340.0 + g * zs))
        assert _level(found) == pytest.approx(level(zs, graze + offset) + louder, abs=0.0005)


def test_the_level_of_a_ray_that_grazes_the_ground() -> None:
    """In c = 340 - 0.17 z, 300 m from a source 1 m up, the ray off the ground grazes it
    for a listener 14.06 m up. For listeners 14.20 to 14.23 m up it meets the ground
    0.016 to 0.020 degrees from the horizontal, at -22.398, -22.090, -21.803 and
    -21.534 dB over free field at its length, from the arc in closed form. For listeners
    14.07 to 14.18 m up, both that ray (-35.657 to -23.087 dB) and the one that turns
    just above the ground (+0.012 dB) leave the source either side of the ray that runs
    level at the ground, within 1.2e-5 of its p, between the same two rays the search
    launches, where the range turns back."""
    rays = Rays(linear(-0.17), 1.0)
    for zr, reference in ((14.20, -22.398), (14.21, -22.090), (14.22, -21.803), (14.23, -21.534)):
        found = rays.eigenrays(zr, 300.0, 1)
        off_ground = [i for i in range(len(found)) if found.ground_bounces[i] == 1]
        assert len(off_ground) == 1
        assert _level(found, off_ground[0]) == pytest.approx(reference, abs=0.05)
    for zr, reference in ((14.07, -35.657), (14.10, -28.135), (14.13, -25.526), (14.18, -23.087)):
        found = rays.eigenrays(zr, 300.0, 1)
        assert sorted(found.ground_bounces) == [0, 1]
        for i in range(2):
            expected = reference if found.ground_bounces[i] else 0.012
            assert _level(found, i) == pytest.approx(expected, abs=0.002)


def test_the_level_of_a_ray_that_turns_over_a_corner_of_the_layers() -> None:
    """Against the ray tube of the smooth profile (its range integrated numerically)
    of the `windy` preset downwind at 20 C: 150 m from a source 1 m up, listeners 5.70
    to 5.90 m up are 82 to 49 mm under the top of the direct ray, which passes the
    corner between two layers at 5.85 m. The ray's level over free field at its length
    is -2.325 + 0.248 (z - 5.70) dB (to 0.001 dB), and the layers' within 0.1 dB of it."""
    windy = Atmosphere.model_validate({"preset": "windy"})
    rays = Rays(Profile.of(windy, 20.0, np.array([1.0, 0.0])), 1.0)
    for zr in np.arange(5.70, 5.905, 0.01):
        found = rays.eigenrays(zr, 150.0, 0)
        assert len(found) == 1
        assert _level(found) == pytest.approx(-2.325 + 0.248 * (zr - 5.70), abs=0.15)


def test_the_level_of_a_ray_from_a_level_layer() -> None:
    """In c = 340 m/s up to 2 m and 340 + 0.17 (z - 2) above, 150 m from a source 1 m
    up, the ray to a listener 5.0 to 6.15 m up goes straight up to 2 m and turns 11 cm to
    8 mm above the listener. Its range is p c0 (2 - zs) / s0 + (s0 + sr) / (p g), s0 and
    sr its sines at 2 m and at the listener, and its level over free field at its
    length, from that in closed form, +0.65 to +0.48 dB."""
    z = np.union1d(node_heights(None), [2.0])
    rays = Rays(Profile(z, 340.0 + 0.17 * np.maximum(z - 2.0, 0.0)), 1.0)

    def level(zr: float) -> float:
        cr = 340.0 + 0.17 * (zr - 2.0)

        def sine(p, c):
            return math.sqrt(1.0 - (p * c) ** 2)

        def distance(p):
            return p * 340.0 / sine(p, 340.0) + (sine(p, 340.0) + sine(p, cr)) / (p * 0.17)

        p = brentq(lambda q: distance(q) - 150.0, 0.9 / cr, (1.0 - 1e-15) / cr, xtol=1e-20)
        s0, sr = sine(p, 340.0), sine(p, cr)
        slope = 340.0 / s0**3 - (1.0 / s0 + 1.0 / sr) / (p * p * 0.17)
        length = 1.0 / s0 + (math.acos(p * 340.0) + math.acos(p * cr)) / (p * 0.17)
        return 10 * math.log10(p * 340.0**2 / (150.0 * abs(slope) * s0 * sr) * length**2)

    for zr in np.arange(5.0, 6.155, 0.05):
        found = rays.eigenrays(zr, 150.0, 0)
        assert len(found) == 1
        assert _level(found) == pytest.approx(level(zr), abs=0.002)


@pytest.mark.parametrize("above", [0.05, 0.4])
def test_the_level_of_a_ray_from_next_to_a_corner_of_the_profile(above) -> None:
    """In c = 340 + 0.17 z up to the node at 6.291 m and a gradient of 0.05 or 0.4/s
    above it, a row of the profile there (as a table's), sources 0.1 um below and above
    that node, listeners 1 m up 0.5 and 2 m beyond the 145.6 m where the ray that turns
    at the node comes down to them: the ray leaves the source almost level and turns
    just above the node, passing it once or twice besides. Its range is the sum of
    k s / (p g) over the source, the node and the listener, s its sine there and g the
    gradient each side, and its level over free field at its length, from that in closed
    form, -7.71 to -7.57 dB (0.05/s) and +7.68 to +6.56 dB (0.4/s)."""
    z = node_heights(None)
    zn = float(z[np.argmin(np.abs(z - 6.29))])
    c = 340.0 + 0.17 * np.minimum(z, zn) + above * np.maximum(z - zn, 0.0)
    profile = Profile(z, c, rows=np.array([zn]))

    def level(zs: float, distance: float) -> float:
        def speed(h):
            return 340.0 + 0.17 * min(h, zn) + above * max(h - zn, 0.0)

        def sine(p, h):
            return math.sqrt(max(1.0 - (p * speed(h)) ** 2, 0.0))

        # (k, height, gradient): the range is the sum of k s / (p g).
        if zs < zn:
            terms = [(1, zs, 0.17), (-2, zn, 0.17), (2, zn, above), (1, 1.0, 0.17)]
        else:
            terms = [(1, zs, above), (1, zn, above), (-1, zn, 0.17), (1, 1.0, 0.17)]
        p = brentq(
            lambda q: sum(k * sine(q, h) / (q * g) for k, h, g in terms) - distance,
            0.99 / speed(zn),
            1.0 / speed(zn),
            xtol=1e-22,
        )
        slope = -sum(k / (sine(p, h) * g) for k, h, g in terms) / (p * p)
        angle = {h: math.acos(min(p * speed(h), 1.0)) for h in (zs, zn, 1.0)}
        if zs < zn:
            arcs = abs(angle[zs] - angle[zn]) + abs(angle[zn] - angle[1.0])
            length = arcs / (p * 0.17) + 2 * angle[zn] / (p * above)
        else:
            arcs = abs(angle[zn] - angle[1.0])
            length = (angle[zs] + angle[zn]) / (p * above) + arcs / (p * 0.17)
        spreading = p * speed(zs) ** 2 / (distance * abs(slope) * sine(p, zs) * sine(p, 1.0))
        return 10 * math.log10(spreading * length**2)

    graze = math.sqrt((zn + 2000.0) ** 2 - 2001.0**2)
    for beyond in (0.5, 2.0):
        for zs in (zn - 1e-7, zn + 1e-7):
            found = Rays(profile, zs).eigenrays(1.0, graze + beyond, 0)
            turned = [i for i in range(len(found)) if found.launch[i] > 0.0]
            assert len(turned) == 1
            assert _level(found, turned[0]) == pytest.approx(level(zs, graze + beyond), abs=0.002)


def test_the_level_of_rays_that_turn_next_to_a_table_s_rows() -> None:
    """A table of no wind up to 2 m, then 3, 6 and 8 m/s at 10, 50 and 200 m (15 C), whose
    gradient changes sharply at its rows. 150 m downwind from a source 1 m up, the rays
    to listeners 0.75 m up turn 1.80 and 0.18 m above the 2 m row, those to listeners
    5.75 and 7.75 m up 3.1 and 1.6 m below the 10 m row, and the one to a listener
    9.75 m up 5 um above it. Against the exact range of the profile (linear between its
    nodes: its layers' arcs summed in closed form, differentiated at high precision),
    their levels over free field at their lengths are +2.802, +2.810, +0.401, +0.290 and
    -35.217 dB."""
    table = [[0.0, 15.0, 0.0], [2.0, 15.0, 0.0], [10.0, 15.0, 3.0], [50.0, 15.0, 6.0]]
    atmosphere = Atmosphere.model_validate({"table": [*table, [200.0, 15.0, 8.0]]})
    rays = Rays(Profile.of(atmosphere, 15.0, np.array([1.0, 0.0])), 1.0)
    for zr, references in (
        (0.75, (2.802, 2.810)),
        (5.75, (0.401,)),
        (7.75, (0.290,)),
        (9.75, (-35.217,)),
    ):
        found = rays.eigenrays(zr, 150.0, 0)
        turned = [i for i in np.argsort(found.p) if found.launch[i] > 0.0]
        assert len(turned) == len(references)
        for i, reference in zip(turned, references, strict=True):
            assert _level(found, i) == pytest.approx(reference, abs=0.02)


def _on(found, up: bool, arrives_up: bool, cycle: int) -> list[int]:
    """The rays of ``found`` on one branch."""
    return [
        i
        for i, b in enumerate(found.branch)
        if (b.up, b.arrives_up, b.cycle) == (up, arrives_up, cycle)
    ]


#: No wind up to 2 m, then 3, 6 and 8 m/s at 10, 50 and 200 m (15 C).
_CALM_BELOW = [
    [0.0, 15.0, 0.0],
    [2.0, 15.0, 0.0],
    [10.0, 15.0, 3.0],
    [50.0, 15.0, 6.0],
    [200.0, 15.0, 8.0],
]


@pytest.mark.parametrize("rise", [0.0, 2.2e-16, 1e-9, 1e-6, 1e-4])
def test_the_direct_ray_through_level_air_is_straight(rise) -> None:
    """In uniform air, and in a table's level layer (no wind up to 2 m), the direct ray
    150 m from a source 1.5 m up to a listener 0 to 0.1 mm higher is the straight one, at
    length / c and at the level of free field, and the only one on its branch. Its
    slowness lies within 2e-13 of a level ray's, where one float's step moves its range
    by centimetres, or the nearest float is a level ray's."""
    atmosphere = Atmosphere.model_validate({"table": _CALM_BELOW})
    for profile in (Profile.uniform(C), Profile.of(atmosphere, 15.0, np.array([1.0, 0.0]))):
        zs = 1.5
        zr = zs + rise
        found = Rays(profile, zs).eigenrays(zr, 150.0, 0)
        direct = _on(found, True, True, 0) + _on(found, False, False, 0)
        assert len(direct) == 1
        i = direct[0]
        length = math.hypot(150.0, zr - zs)
        assert found.length[i] == pytest.approx(length, rel=1e-12)
        assert found.time[i] == pytest.approx(length / float(profile.speed(zs)), rel=1e-12)
        assert found.launch[i] == pytest.approx(math.atan2(zr - zs, 150.0), abs=1e-15)
        assert _level(found, i) == pytest.approx(0.0, abs=1e-9)
        assert not found.capped[i]


@pytest.mark.parametrize("zs", [2.0, 2.0 - 1e-9])
@pytest.mark.parametrize("distance", [150.0, 300.0])
def test_a_source_over_level_air_has_no_fold_below(zs, distance) -> None:
    """Upwind of a table with no wind up to 2 m and 3 m/s at 10 m (15 C), from a source
    on the 2 m row to a listener 6.25 m up: the air slows above the row, so no ray
    leaving the source reaches the listener directly beyond the 87.7 m of the one leaving
    level, and below the row, in level air, none turns. The one ray is the ground's,
    straight down and back up to the row, then an arc: its range, time and level over
    free field at its length are in closed form. So too from 1 nm under the row, where a
    direct ray would run level through that nanometre to within 1e-22 of grazing, finer
    than a float resolves (at -94 and -102 dB)."""
    profile = Profile.of(
        Atmosphere.model_validate({"table": _CALM_BELOW}), 15.0, np.array([-1.0, 0.0])
    )
    c0 = float(profile.speed(2.0))
    g = (float(profile.speed(10.0)) - c0) / 8.0
    zr = 6.25
    cr = c0 + g * (zr - 2.0)

    def sine(p, c):
        return math.sqrt(1.0 - (p * c) ** 2)

    def reach(p):
        return (zs + 2.0) * p * c0 / sine(p, c0) + (sine(p, c0) - sine(p, cr)) / (p * g)

    p = brentq(lambda q: reach(q) - distance, 0.9 / c0, (1.0 - 1e-16) / c0, xtol=1e-22)
    s0, sr = sine(p, c0), sine(p, cr)
    slope = (zs + 2.0) * c0 / s0**3 + (1.0 / sr - 1.0 / s0) / (p * p * g)
    length = (zs + 2.0) / s0 + abs(math.acos(p * c0) - math.acos(p * cr)) / (p * abs(g))
    time = (zs + 2.0) / (c0 * s0) + (math.acosh(1.0 / (p * c0)) - math.acosh(1.0 / (p * cr))) / g
    level = 10 * math.log10(p * c0**2 / (distance * abs(slope) * s0 * sr) * length**2)
    found = Rays(profile, zs).eigenrays(zr, distance, 1)
    assert list(found.ground_bounces) == [1]
    assert found.p[0] == pytest.approx(p, rel=1e-12)
    assert found.time[0] == pytest.approx(time, rel=1e-9)
    assert _level(found) == pytest.approx(level, abs=0.002)


@pytest.mark.parametrize(
    ("zs", "zr", "distance", "references"),
    [(5.0, 1.25, 150.0, (-7.475, -3.309)), (10.0, 0.75, 300.0, (-11.065, -8.073))],
)
def test_rays_down_into_level_air(zs, zr, distance, references) -> None:
    """Upwind of a table with no wind up to 2 m and 3 m/s at 10 m (15 C), from a source 5
    or 10 m up to a listener 1.25 or 0.75 m up in the calm air under 2 m: against the
    exact range of the profile (linear between its nodes: its layers' arcs summed in
    closed form, differentiated at high precision), the levels over free field at their
    lengths of the direct ray and the one off the ground are those given (dB). Among the
    rays launched is the one that runs level in the calm air, some 1e300 m long, which
    the search takes in without an overflow."""
    profile = Profile.of(
        Atmosphere.model_validate({"table": _CALM_BELOW}), 15.0, np.array([-1.0, 0.0])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        found = Rays(profile, zs).eigenrays(zr, distance, 1)
    assert len(found) == 2
    for branch, reference in zip(((False, False, 0), (False, True, 0)), references, strict=True):
        (i,) = _on(found, *branch)
        assert _level(found, i) == pytest.approx(reference, abs=0.02)


_WIND_ROW = [[0.0, 15.0, 0.0], [10.0, 15.0, 4.0], [100.0, 15.0, 8.0]]
_WARM_ROWS = [[0, 10, 0], [1, 12, 0], [2, 13, 0], [4, 13.5, 0], [10, 13.6, 0], [50, 12, 0]]


@pytest.mark.parametrize(
    ("table", "zs", "distance", "references"),
    [
        (_WIND_ROW, 12.0, 340.0, (-6.118, -5.837)),
        (_WIND_ROW, 12.0, 360.0, (-6.098, -5.862)),
        (_WARM_ROWS, 4.0, 150.0, (-14.181, -14.171)),
        (_WARM_ROWS, 4.0, 300.0, (-11.184, -11.176)),
    ],
)
def test_the_level_of_rays_that_turn_just_above_the_source(table, zs, distance, references) -> None:
    """Downwind of a table of wind (4 m/s at its 10 m row, 8 at 100 m) from a source 12 m
    up, and in one of temperatures alone (13.5 C at its 4 m row, 13.6 C at 10 m and 12 C at
    50 m, 10 to 13 C below) from a source on that row, the rays to a listener 1.5 m up
    turn 0.09 to 0.31 m and 0.2 mm to 0.15 m above the source, with rows of the table
    between the ends. Against the exact range of the profile (linear between its nodes:
    its layers' arcs summed in closed form, differentiated at high precision), the levels
    over free field at their lengths of the direct ray and the one off the ground are
    those given (dB)."""
    rays = Rays(
        Profile.of(Atmosphere.model_validate({"table": table}), 15.0, np.array([1.0, 0.0])), zs
    )
    found = rays.eigenrays(1.5, distance, 1)
    assert len(found) == 2
    for branch, reference in zip(((True, False, 0), (True, True, 1)), references, strict=True):
        (i,) = _on(found, *branch)
        assert _level(found, i) == pytest.approx(reference, abs=0.05)


@pytest.mark.parametrize(
    ("zs", "zr", "distance", "references"),
    [
        (10.4, 5.0, 150.0, {(True, False, 0): -13.412, (True, True, 1): -13.380}),
        (10.4 - 1e-5, 5.0, 150.0, {(True, False, 0): -13.475, (True, True, 1): -19.239}),
        (
            1.0,
            10.4,
            300.0,
            {
                (True, False, 0): -10.457,
                (False, False, 1): -10.452,
                (True, True, 1): -10.402,
                (True, False, 1): -13.428,
            },
        ),
        (
            1.0,
            10.4 - 1e-5,
            300.0,
            {
                (True, False, 0): -10.470,
                (False, False, 1): -10.468,
                (True, True, 1): -13.023,
                (True, False, 1): -18.529,
            },
        ),
    ],
)
def test_the_level_of_rays_with_an_end_on_one_of_two_close_rows(
    zs, zr, distance, references
) -> None:
    """A table of no wind up to its 10 m row, then 2 and 4 m/s at 10.4 and 100 m (15 C),
    downwind: from a source on the 10.4 m row or 10 um under it to a listener 5 m up
    150 m away, and from a source 1 m up to a listener there 300 m away, the rays turn
    0.16 mm to 0.36 m above that row. Against the exact range of the profile (as above),
    their levels over free field at their lengths are those given (dB), by branch."""
    table = [[0, 15, 0], [10, 15, 0], [10.4, 15, 2], [100, 15, 4]]
    profile = Profile.of(Atmosphere.model_validate({"table": table}), 15.0, np.array([1.0, 0.0]))
    found = Rays(profile, zs).eigenrays(zr, distance, 1)
    assert len(found) == len(references)
    for branch, reference in references.items():
        (i,) = _on(found, *branch)
        assert _level(found, i) == pytest.approx(reference, abs=0.02)


def _straight_rays(found, zs: float, zr: float, distance: float) -> None:
    """``found`` holds the ray off the ground and, where ``zs`` and ``zr`` differ, the
    direct ray, at their straight lengths and the level of free field."""
    bounces = sorted(found.ground_bounces)
    assert bounces == [1] or bounces == [0, 1]
    assert bounces == [0, 1] or zs == zr
    for i, n in enumerate(found.ground_bounces):
        assert found.length[i] == pytest.approx(
            math.hypot(distance, zr - zs * (1 - 2 * n)), rel=1e-9
        )
        assert _level(found, i) == pytest.approx(0.0, abs=0.005)


@pytest.mark.parametrize("direction", [1.0, -1.0])
@pytest.mark.parametrize(
    "rows",
    [
        *(
            [[0.0, 15.0, 2.0], [10.0, 15.0, 2.0 + dw], [12.0, 15.0, 2.0 + dw], [60.0, 15.0, 5.0]]
            for dw in (1e-10, 1e-8, 1e-6)
        ),
        [[0.0, 15.0, float(np.float32(2.3))], [10.0, 15.0, 2.3], [60.0, 15.0, 5.0]],
    ],
)
def test_rays_through_air_level_but_for_a_hair(rows, direction) -> None:
    """Tables of 2 m/s of wind at the ground and 1e-10 to 1e-6 m/s more at 10 m, or of
    2.3 m/s held at the ground as a 32-bit float (5e-8 m/s less) and at 10 m as a 64-bit
    one, then 5 m/s at 60 m (15 C), downwind and upwind: below 10 m the gradient is
    1e-7/s or less, too weak for neighbouring floats of a ray's slowness to tell apart.
    100 m from a source 1 or 5 m up to a listener 1, 3 or 5 m up, the rays are the one off
    the ground and, between ends at different heights, the direct one, at their straight
    lengths and the level of free field (to 0.005 dB, as the exact range of the profile
    gives); between ends at one height the direct ray, level to within 1e-8 rad, is not
    resolved."""
    atmosphere = Atmosphere.model_validate({"table": rows})
    profile = Profile.of(atmosphere, 15.0, np.array([direction, 0.0]))
    for zs, zr in ((1.0, 1.0), (5.0, 5.0), (1.0, 3.0)):
        _straight_rays(Rays(profile, zs).eigenrays(zr, 100.0, 2), zs, zr, 100.0)


@pytest.mark.parametrize(("zs", "zr", "distance"), [(5.0, 5.0, 100.0), (4.0, 6.0, 200.0)])
def test_rays_through_a_step_of_one_float_in_level_air(zs, zr, distance) -> None:
    """In c = 340 m/s up to 5 m, the next float above it (5.7e-14 m/s more) from 5 to
    10 m and 0.05/s more above 10 m, from a source 5 or 4 m up to a listener 5 or 6 m up,
    100 or 200 m away: the rays are those of uniform air below 10 m, as above."""
    z = np.union1d(node_heights(None), [10.0])
    step = np.nextafter(340.0, np.inf)
    profile = Profile(z, np.where(z < 5.0, 340.0, step + 0.05 * np.maximum(z - 10.0, 0.0)))
    _straight_rays(Rays(profile, zs).eigenrays(zr, distance, 2), zs, zr, distance)


@pytest.mark.parametrize(
    ("table", "height", "references"),
    [
        (
            _CALM_BELOW,
            1.999,
            {
                (True, False, 0): (-12.108, 0.000),
                (True, True, 1): (-12.907, 0.435),
                (False, True, 1): (-13.966, 0.979),
                (True, False, 1): (0.954, 0.980),
                (True, True, 2): (2.795, 2.810),
            },
        ),
        (
            [[0.0, 15.0, 0.0], [3.0, 15.0, 2.0], [6.0, 15.0, 2.0], [20.0, 15.0, 5.0]],
            3.001,
            {
                (True, False, 0): (1.387, 1.416),
                (False, True, 0): (1.414,),
                (True, True, 1): (3.557, 3.821),
            },
        ),
    ],
)
def test_the_level_of_rays_that_run_almost_level_through_level_air(
    table, height, references
) -> None:
    """Downwind of a table with no wind up to 2 m (as above), and of one with 2 m/s from 3
    to 6 m (none at the ground, 5 m/s at 20 m), 15 C, from a source to a listener 1 mm
    under the top of the calm air or 1 mm over the bottom of the steady wind, 400 m away:
    some rays turn just beyond that edge, crossing the millimetre of level air within
    1e-5 rad of the horizontal, most of their range p c dz / s. Against the exact range
    of the profile (as above), the levels over free field at their lengths of the rays
    on these branches are those given (dB)."""
    profile = Profile.of(Atmosphere.model_validate({"table": table}), 15.0, np.array([1.0, 0.0]))
    found = Rays(profile, height).eigenrays(height, 400.0, 2)
    for branch, levels in references.items():
        on = _on(found, *branch)
        assert sorted(_level(found, i) for i in on) == pytest.approx(levels, abs=0.002)


@pytest.mark.parametrize(
    ("preset", "height", "distance"),
    [
        ("windy", 1.5, 10.0),
        ("windy", 12.0, 30.0),
        ("clear_night", 4.0, 10.0),
        ("clear_night", 4.0, 30.0),
        ("clear_night", 4.0, 50.0),
    ],
)
def test_the_level_of_a_ray_between_ends_at_one_height(preset, height, distance) -> None:
    """Downwind in the windy and clear-night presets (15 C), between a source and a
    listener 1.5, 4 or 12 m up and 10 to 50 m apart, the direct ray turns 0.3 to 9 cm
    above them, in the layer they are in: against the exact range of the profile (as
    above), its level over free field at its length is 0.000 dB (to 0.001)."""
    profile = Profile.of(Atmosphere.model_validate({"preset": preset}), 15.0, np.array([1.0, 0.0]))
    found = Rays(profile, height).eigenrays(height, distance, 0)
    assert len(found) == 1
    assert _level(found) == pytest.approx(0.0, abs=0.002)


@pytest.mark.parametrize(
    ("preset", "near", "under", "distance", "reference"),
    [
        ("windy", 6.3, 0.0, 26.3, 0.000),
        ("windy", 4.0, 1e-3, 10.0, -0.043),
        ("clear_night", 12.0, 0.0, 20.0, 0.000),
    ],
)
def test_the_level_of_a_ray_between_ends_just_under_a_node(
    preset, near, under, distance, reference
) -> None:
    """As above, between ends a float or 1 mm under the profile's node nearest 6.3, 4 or
    12 m: the direct ray turns 0.3 to 3.6 cm above them, in the layer above the node.
    Against the exact range of the profile (as above), its level over free field at its
    length is that given (dB)."""
    profile = Profile.of(Atmosphere.model_validate({"preset": preset}), 15.0, np.array([1.0, 0.0]))
    node = float(profile.z[np.argmin(np.abs(profile.z - near))])
    height = float(np.nextafter(node, 0.0)) - under
    found = Rays(profile, height).eigenrays(height, distance, 0)
    assert len(found) == 1
    assert _level(found) == pytest.approx(reference, abs=0.002)


@pytest.mark.parametrize("weak", ["line", "table"])
def test_the_level_of_rays_in_a_gradient_of_1e_8_per_second(weak) -> None:
    """In c = 340 m/s - 1e-8 z, and downwind of the table above with 1e-7 m/s more wind at
    10 m than at the ground (15 C), between a source and a listener 1 m up and 1 km
    apart: the ray off the ground leaves within 0.002 rad of the horizontal, where the
    parts of its range singular at its ends and at the ground (7e7 m and more) carry
    the rounding of their sines, 4e-3 m all told, and the nearest ray to run level in a
    level layer (above 1 km, or 10 to 12 m up in the table, neither of which it reaches)
    is 6e-9 s/m away in slowness. Against the exact range of the profile (as above), its
    level over free field at its length is 0.000 dB, and in the table those of the
    direct rays that turn 15.7 and 20.1 m up are 7.204 and 7.208 dB."""
    if weak == "line":
        z = node_heights(None)
        profile, direct = Profile(z, 340.0 - 1e-8 * z), []
    else:
        rows = [
            [0.0, 15.0, 2.0],
            [10.0, 15.0, 2.0000001],
            [12.0, 15.0, 2.0000001],
            [60.0, 15.0, 5.0],
        ]
        profile = Profile.of(Atmosphere.model_validate({"table": rows}), 15.0, np.array([1.0, 0.0]))
        direct = [7.204, 7.208]
    found = Rays(profile, 1.0).eigenrays(1.0, 1000.0, 2)
    off = [i for i in range(len(found)) if found.ground_bounces[i] == 1]
    assert len(off) == 1
    assert _level(found, off[0]) == pytest.approx(0.0, abs=0.002)
    levels = sorted(_level(found, i) for i in _on(found, True, False, 0))
    assert levels[len(levels) - len(direct) :] == pytest.approx(direct, abs=0.002)


def test_the_levels_of_rays_over_rows_1e_8_m_s_apart() -> None:
    """Downwind of the table above with 1e-8 m/s more wind at 10 m than at the ground
    instead (15 C), from a source 10 m up to a listener 1 m up and 1 km away: a direct
    ray and one off the ground turn 26 to 28 m up, another two 12.7 to 13 m up, and two
    go down within 0.011 rad of the horizontal, one of them off the ground. Against the
    exact range of the profile (as above), their levels over free field at their lengths
    are 1.817, 2.313, 1.830, 2.324, 0.000 and 0.000 dB (to 0.02 dB)."""
    rows = [[0.0, 15.0, 2.0], [10.0, 15.0, 2.00000001], [12.0, 15.0, 2.00000001], [60.0, 15.0, 5.0]]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 15.0, np.array([1.0, 0.0]))
    found = Rays(profile, 10.0).eigenrays(1.0, 1000.0, 2)
    levels = sorted(_level(found, i) for i in range(len(found)))
    assert levels == pytest.approx([0.0, 0.0, 1.817, 1.830, 2.313, 2.324], abs=0.02)


def test_the_level_of_a_ray_that_crosses_a_row_almost_level() -> None:
    """Downwind of a table of no wind at the ground, 3 m/s at 10 m, none at 10.3 m and
    4 m/s at 100 m (15 C), from a source 1 m up to a listener 10.75 m up and 150 m away:
    the direct ray crosses the row at 10 m, the fastest air below the listener, within
    0.001 rad of the horizontal. Against the exact range of the profile (as above), its
    level over free field at its length is -21.226 dB, and that of the ray off the ground
    -9.023 dB."""
    rows = [[0.0, 15.0, 0.0], [10.0, 15.0, 3.0], [10.3, 15.0, 0.0], [100.0, 15.0, 4.0]]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 15.0, np.array([1.0, 0.0]))
    found = Rays(profile, 1.0).eigenrays(10.75, 150.0, 1)
    assert list(found.ground_bounces) == [0, 1]
    assert [_level(found, 0), _level(found, 1)] == pytest.approx([-21.226, -9.023], abs=0.002)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_the_levels_of_rays_below_rows_1e_10_m_s_apart(direction) -> None:
    """Downwind and upwind of a table of 2 m/s of wind at the ground, 1e-10 m/s more at
    10 and 12 m and 5 m/s at 60 m (15 C), from a source 1 m up to a listener 3 m up and
    1 km away: the direct ray and the one off the ground, which stay below 10 m, are at
    the level of free field (to 0.005 dB), as the exact range of the profile gives."""
    rows = [
        [0.0, 15.0, 2.0],
        [10.0, 15.0, 2.0000000001],
        [12.0, 15.0, 2.0000000001],
        [60.0, 15.0, 5.0],
    ]
    profile = Profile.of(
        Atmosphere.model_validate({"table": rows}), 15.0, np.array([direction, 0.0])
    )
    found = Rays(profile, 1.0).eigenrays(3.0, 1000.0, 2)
    low = [i for i, b in enumerate(found.branch) if b.cycle == 0 and b.arrives_up]
    assert sorted(found.ground_bounces[low]) == [0, 1]
    assert [_level(found, i) for i in low] == pytest.approx([0.0, 0.0], abs=0.005)


def test_the_levels_of_rays_between_ends_on_a_node_near_a_caustic() -> None:
    """In the `clear_night` preset (15 C), from a source on the profile's node nearest
    4.7 m to a listener 1e-12 m above it and 437.3 m away, the rays off the ground that
    leave and arrive going the same way turn 5.08 m up. Against the smooth profile's ray
    tube (its ray equations and their variations integrated numerically), their level
    over free field at their lengths is 8.201 dB; theirs are within 0.1 dB of it."""
    profile = Profile.of(Atmosphere.model_validate({"preset": "clear_night"}), 15.0, None)
    node = float(profile.z[np.argmin(np.abs(profile.z - 4.7))])
    found = Rays(profile, node).eigenrays(node + 1e-12, 437.3, 1)
    turning = [i for i, b in enumerate(found.branch) if b.up == b.arrives_up]
    assert len(turning) == 2
    assert [_level(found, i) for i in turning] == pytest.approx([8.201, 8.201], abs=0.1)


@pytest.mark.parametrize(("height", "distance"), [(1.0, 50.0), (5.0, 200.0)])
def test_the_level_of_a_direct_ray_in_a_gradient_of_1e_6_per_second(height, distance) -> None:
    """In c = 340 m/s - 1e-6 z, between a source and a listener 1 m up and 50 m apart, or
    5 m up and 200 m apart, the direct ray leaves 7e-8 or 3e-7 rad below the horizontal
    and turns 9e-7 or 1.5e-5 m below them; rays a millionth of its slowness steeper reach
    the thin layers near the ground, whose gradients rounding moves by more than a
    millionth, so its neighbours either side lie within 30 floats of it in slowness.
    Against the exact range of the profile (as above), its level over free field at its
    length is 0.000 dB; it is within 0.05 dB."""
    z = node_heights(None)
    found = Rays(Profile(z, 340.0 - 1e-6 * z), height).eigenrays(height, distance, 2)
    direct = [i for i in _on(found, False, True, 0) if found.ground_bounces[i] == 0]
    assert len(direct) == 1
    assert _level(found, direct[0]) == pytest.approx(0.0, abs=0.05)


def test_the_level_of_a_ray_between_ends_on_the_bottom_of_level_air() -> None:
    """Upwind of a table of 2 m/s of wind at the ground, 2.001 m/s at 10 and 12 m and
    5 m/s at 60 m (15 C), between a source and a listener 10 m up, at the bottom of the
    level air, and 50 m apart: the direct ray below it leaves 7e-6 rad below the
    horizontal and turns 9e-5 m below them, 8e-14 s/m in slowness from the ray that runs
    level in the level air, which it never enters. Against the exact range of the profile
    (as above), its level over free field at its length is 0.000 dB (to 0.002 dB)."""
    rows = [[0.0, 15.0, 2.0], [10.0, 15.0, 2.001], [12.0, 15.0, 2.001], [60.0, 15.0, 5.0]]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 15.0, np.array([-1.0, 0.0]))
    found = Rays(profile, 10.0).eigenrays(10.0, 50.0, 2)
    below = [i for i in _on(found, False, True, 0) if found.ground_bounces[i] == 0]
    assert len(below) == 1
    assert _level(found, below[0]) == pytest.approx(0.0, abs=0.002)


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


def test_the_gpu_tracer_s_shadow_follows_the_arc_that_grazes_the_ground() -> None:
    """The GPU tracer's table of how far from where it touches the ground the ray that
    grazes it reaches each height, in c = 331.3 - 0.1 z: sqrt(2 R z - z^2), R = 3313 m,
    to 1e-9. With the ray's sine at the ground taken from p (1 / 331.3), which rounds
    short of 1, it fell 6e-6 short."""
    from acoustic_spaces.engine.gpu_tracer import _atmosphere_tables

    z = node_heights(None)
    reach, _ = _atmosphere_tables(np.column_stack([z, 331.3 - 0.1 * z, np.zeros_like(z)]))
    radius = 331.3 / 0.1
    np.testing.assert_allclose(reach[0, 1:], np.sqrt(2 * radius * z[1:] - z[1:] ** 2), rtol=1e-9)


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
    started = time.perf_counter()
    shared = simulate(room, share_waves=[s.name for s in sources])
    seconds = time.perf_counter() - started
    assert 0 < len(runs) < alone and all(runs)
    # Each source's early time is its own: their sum is within the run's.
    assert sum(shared.details[s.name]["early_s"] for s in sources) < seconds
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


def test_the_parabolic_equation_over_a_ground_that_absorbs_nothing() -> None:
    """A ground absorbing nothing has an admittance of 0 and an infinite impedance: the
    parabolic equation's bands (63 to 500 Hz) over it, 100 m downwind and upwind, come
    within 0.05 dB of those over a ground absorbing 0.001."""
    levels = []
    for absorption in (0.0, 0.001):
        data = field({"preset": "windy"}, ((100.0, 0.0, 1.5), (-100.0, 0.0, 1.5))).model_dump()
        data["materials"]["lawn"] = {"absorption": [absorption] * 8, "scattering": 0.0}
        result = simulate(Room.from_scene(Scene.model_validate(data)))
        energy = [result["s", key].early.energy[:, :4].sum(axis=0) for key in ("l0", "l1")]
        levels.append(10.0 * np.log10(energy))
    np.testing.assert_allclose(levels[0], levels[1], atol=0.05)


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
