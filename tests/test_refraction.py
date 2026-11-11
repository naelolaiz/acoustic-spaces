"""Wind and temperature outdoors: the atmosphere, curved rays, the parabolic equation."""

import importlib
import itertools
import math
import sys
import time
import warnings
import weakref
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import brentq
from test_ground import exact_reflection

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.parabolic import march
from acoustic_spaces.engine.refracted import _FIRST_MODE, refraction_for
from acoustic_spaces.engine.refraction import Profile, Rays, node_heights
from acoustic_spaces.engine.shadow import shadow_start
from acoustic_spaces.examples import example_path
from acoustic_spaces.materials import THIRD_OCTAVE_BANDS_HZ
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


def scattering_lawn(scene: Scene) -> Scene:
    """``scene`` (:func:`field`) with a lawn that scatters 0.1: its rays lose that share
    at the ground and the late rays scatter it, which is the scene's late part."""
    data = scene.to_data()
    data["materials"]["lawn"]["scattering"] = 0.1
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


def test_the_shadow_starts_continuously_as_the_heights_go_to_the_ground() -> None:
    """Upwind in the ``windy`` preset the profile is linear below its first node, its
    grazing ray a circle of radius R = c0 / |dc/dz|: the shadow starts at
    sqrt(2 R z - z^2) a height, which goes to 0 with the heights, also below the first
    height whose speed differs from c0 in floating point (1e-16 m) and on the ground,
    where it was no shadow. At 1e-12 m the rays' own arithmetic rounds: 1.2e-3 off."""
    up = Profile.of(Atmosphere.model_validate({"preset": "windy"}), 20.0, np.array([-1.0, 0.0]))
    radius = float(up.c[0]) / abs(float(up.gradient[0]))
    for z, rel in ((1e-6, 1e-8), (1e-12, 2e-3), (1e-16, 1e-12), (0.0, 0.0)):
        limit = 2.0 * math.sqrt(2.0 * radius * z - z * z)
        assert shadow_start(up, z, z) == pytest.approx(limit, rel=rel, abs=0.0)
    assert shadow_start(up, 0.0, 1e-16) == pytest.approx(math.sqrt(2.0 * radius * 1e-16))


def test_the_shadow_of_a_profile_level_at_the_ground_comes_from_its_rays() -> None:
    """A measured profile level up to 0.5 m and falling above it: the ray that grazes the
    ground stays level and never reaches 50 m, so the shadow starts beyond any distance
    (above 1e299 m), from the ground and from inside the level layer alike; the limit of
    a falling ground layer (R = c0 / |dc/dz|) does not apply where dc/dz = 0. A guard
    (the rays alone pass it too): it catches that limit taken where dc/dz = 0, which
    divides by zero."""
    level = Profile(
        np.array([0.0, 0.01, 0.5, 100.0]),
        np.array([340.0, 340.0, 340.0, 330.0]),
        np.array([0.0, 0.5, 100.0]),
    )
    for zs in (0.0, 0.3, 1.0):
        start = shadow_start(level, zs, 50.0)
        assert start is not None and start > 1e299


def test_a_leg_along_the_ground_upwind_is_in_the_shadow() -> None:
    """Legs 10 m upwind whose ends sink to the ground (a ground reflection to the foot of
    an edge), 1e-3 m to 0: each falls into the shadow, 8 kHz attenuated as the first mode
    of the ground's circle limit has it, exp(-2 Im kappa (10 m - start)), start =
    2 sqrt(2 R z - z^2), within 0.1 dB (0.07 dB at 1e-12 m, where the rays' own
    arithmetic rounds the start), down to the ground itself, which was left unattenuated.
    One Refraction keeps one start a millimetre of heights and the heights it was found
    for: a leg on the ground after one 0.4 mm up takes that leg's start and radius
    (a radius from the start and the ground's heights divided by zero)."""
    source = np.array([0.0, 0.0, 1.0])
    scene = Scene.model_validate(
        {
            "geometry": {"ground": {"extent": [[-50.0, -20.0], [50.0, 20.0]], "sky": 20.0}},
            "materials": {"hard": {"absorption": [0.02] * 8, "scattering": 0.0}},
            "assignments": {"ground": "hard"},
            "atmosphere": {"preset": "windy", "wind": {"direction_deg": 180}},
            "sources": [{"name": "s", "position": list(source)}],
            "listeners": [{"name": "l", "position": [20.0, 0.0, 1.5]}],
        }
    )
    room = Room.from_scene(scene)

    def refraction():
        return refraction_for(
            TraceModel.from_room(room),
            source,
            speed_of_sound=room.speed_of_sound(),
            air=np.zeros(8),
        )

    def level(refraction, z: float) -> float:
        return 10 * np.log10(
            refraction._leg(np.array([0.0, 0.0, z]), np.array([10.0, 0.0, z]))[0][-1]
        )

    heights = (1e-3, 4e-4, 1e-6, 1e-12, 1e-16, 0.0)
    levels = np.array([level(refraction(), z) for z in heights])
    probe = refraction()
    profile = probe.profile(probe._towards(np.array([1.0, 0.0])))
    radius = float(profile.c[0]) / abs(float(profile.gradient[0]))
    k = 2.0 * np.pi * 8000.0 / float(profile.c[0])
    ell = (radius / (2.0 * k**2)) ** (1.0 / 3.0)
    kappa = np.sqrt(k**2 - _FIRST_MODE * np.exp(1j * np.pi / 3.0) / ell**2 + 0j)
    start = 2.0 * np.sqrt(2.0 * radius * np.array(heights) - np.square(heights))
    expected = 10 * np.log10(np.exp(-2.0 * abs(kappa.imag) * (10.0 - start)))
    np.testing.assert_allclose(levels, expected, atol=0.1)
    shared = refraction()
    first, ground = level(shared, 4e-4), level(shared, 0.0)
    assert ground == first < -40.0


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


def test_the_levels_of_rays_in_a_preset_against_its_smooth_profile() -> None:
    """Against the ray tube of the smooth profile (its ray equations and their
    variations integrated numerically) of the `windy` preset downwind at 20 C, from a
    source 1 m up to a listener 8.5 m up and 600 m away: the direct ray, the ray off the
    ground and the two that turn back down after it have levels over free field at their
    lengths of -4.927, -4.751, 3.199 and 3.488 dB; the layers' are within 0.2 dB of them."""
    windy = Atmosphere.model_validate({"preset": "windy"})
    found = Rays(Profile.of(windy, 20.0, np.array([1.0, 0.0])), 1.0).eigenrays(8.5, 600.0, 1)
    levels = sorted(_level(found, i) for i in range(len(found)))
    assert levels == pytest.approx([-4.927, -4.751, 3.199, 3.488], abs=0.2)


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
    form, -7.71 to -7.57 dB (0.05/s) and +7.68 to +6.56 dB (0.4/s); the layers' are
    within 0.005 dB of it."""
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
            assert _level(found, turned[0]) == pytest.approx(level(zs, graze + beyond), abs=0.005)


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


def _exact_range(profile, zs: float, zr: float, p: float, branch, ground: bool) -> Decimal:
    """The range of the ray with parameter ``p`` along ``branch`` from a source at ``zs``
    to the height ``zr`` (off the ground where ``ground``, else turning low in the air),
    summed over the layers of ``profile`` (straight between its nodes, level above the
    last) in closed form to 40 digits."""
    with localcontext() as context:
        context.prec = 40
        z = [Decimal(float(h)) for h in profile.z]
        c = [Decimal(float(v)) for v in profile.c]
        p, zs, zr = Decimal(float(p)), Decimal(float(zs)), Decimal(float(zr))

        def pieces(a, b):
            # each layer's part between a and b: its ends and the speeds there
            for j in range(len(z)):
                last = j == len(z) - 1
                lo, hi = max(a, z[j]), (b if last else min(b, z[j + 1]))
                if lo < hi:
                    g = 0 if last else (c[j + 1] - c[j]) / (z[j + 1] - z[j])
                    yield lo, hi, c[j] + g * (lo - z[j]), c[j] + g * (hi - z[j])

        def turn(pieces):
            # where the ray runs level, the first piece of ``pieces`` it does so in
            for lo, hi, a, b in pieces:
                if p * max(a, b) >= 1:
                    return lo + (1 / p - a) * (hi - lo) / (b - a) if a != b else lo
            return None

        def leg(a, b):
            total = Decimal(0)
            for lo, hi, ca, cb in pieces(a, b):
                sa = max(1 - (p * ca) ** 2, Decimal(0)).sqrt()
                sb = max(1 - (p * cb) ** 2, Decimal(0)).sqrt()
                total += (
                    p * ca * (hi - lo) / sa if ca == cb else (sa - sb) * (hi - lo) / (p * (cb - ca))
                )
            return total

        events = branch.lower_events + branch.upper_turns
        if not events:
            return leg(min(zs, zr), max(zs, zr))
        low = z[0] if ground else turn(reversed(list(pieces(z[0], min(zs, zr)))))
        high = turn(pieces(max(zs, zr), max(zs, zr) + 10**6)) if branch.upper_turns else None
        first = leg(zs, high) if branch.up else leg(low, zs)
        last = leg(low, zr) if branch.arrives_up else leg(zr, high)
        return first + (events - 1) * leg(low, high) + last if events > 1 else first + last


def _assert_on_the_listener(
    profile, zs: float, zr: float, distance: float, found, straight: tuple[int, ...] = ()
) -> None:
    """Each ray of ``found`` but those ``straight`` (along air whose speed does not change
    with height) reaches the listener: by the exact range of ``profile``
    (:func:`_exact_range`) at its p, it lands within the tolerance (1e-6 of the distance)
    of the listener, or the range crosses the distance within three floats of its p (a
    steep crossing)."""
    for i, branch in enumerate(found.branch):
        if i in straight:
            continue
        ps = [float(found.p[i])]
        for _ in range(3):
            ps = [float(np.nextafter(ps[0], 0.0)), *ps, float(np.nextafter(ps[-1], 1.0))]
        ground = bool(found.ground[i])
        misses = [
            float(_exact_range(profile, zs, zr, q, branch, ground) - Decimal(distance)) for q in ps
        ]
        crosses = any((a < 0) != (b < 0) for a, b in itertools.pairwise(misses))
        assert abs(misses[3]) <= 1e-6 * (1 + distance) or crosses, (branch, misses[3])


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


@pytest.mark.parametrize("rise", [3e-12, 1e-10, 1e-9])
def test_the_levels_of_rays_through_air_1e_12_to_1e_9_off_level(rise) -> None:
    """Downwind of a table of no wind at the ground, 2 m/s at 2 m, 3e-12, 1e-10 or
    1e-9 m/s more at 2.5 m and 5 m/s at 20 m (15 C), between a source and a listener
    2.2 m up and 300 m apart, in the layer whose gradient is that much off level: two
    direct rays turn in the air above 2.5 m, and two pairs of rays off the ground that
    leave and arrive going the same way turn there before or after it. Against the exact
    range of the profile, the levels over free field at their lengths of the direct
    rays, the ray off the ground and those pairs are 0.230, 0.245, 1.187, 1.710 (twice)
    and 1.888 (twice) dB; theirs are within 0.02 dB, and every ray found reaches the
    listener."""
    rows = [[0.0, 15.0, 0.0], [2.0, 15.0, 2.0], [2.5, 15.0, 2.0 + rise], [20.0, 15.0, 5.0]]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 15.0, np.array([1.0, 0.0]))
    found = Rays(profile, 2.2).eigenrays(2.2, 300.0, 1)
    rays = [i for i, b in enumerate(found.branch) if b.cycle == 0 or b.up == b.arrives_up]
    levels = sorted(_level(found, i) for i in rays)
    assert levels == pytest.approx([0.230, 0.245, 1.187, 1.710, 1.710, 1.888, 1.888], abs=0.02)
    _assert_on_the_listener(profile, 2.2, 2.2, 300.0, found)


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


@pytest.mark.parametrize(
    ("g", "c0", "zs", "zr", "distance"),
    [
        (-5.22718109404673e-12, 345.607, 24.901651974979007, 24.90265197497901, 709.9),
        (-3.688596948245909e-09, 338.079, 0.12607498764233685, 0.13544249376408263, 2952.7),
        (-1.565474885763014e-11, 332.254, 28.6, 28.6001, 41.9),
        (-1.7956537118736916e-10, 336.091, 0.12707498664233685, 0.12697498664233686, 130.1),
    ],
)
def test_the_levels_of_rays_that_leave_almost_level_in_a_weak_gradient(
    g, c0, zs, zr, distance
) -> None:
    """In c = c0 + g z, g from -4e-9 to -5e-12/s, between a source and a listener 1 mm,
    9 mm and 0.1 mm higher or 0.1 mm lower and 710, 2953, 42 and 130 m away: the direct
    ray leaves 1.4e-6, 3.2e-6 and 2.4e-6 rad above or 7.7e-7 rad below the horizontal,
    where neighbouring floats of p give ranges 0.054, 0.043, 0.001 and 0.032 m apart,
    more than the tolerance (1e-6 of the distance), and no float of p takes it within the
    tolerance of the listener. Against the exact range of the profile, the direct ray and
    the one off the ground are at the level of free field at their lengths (to 0.002 dB);
    the one off the ground is found, every ray found reaches the listener, and each is at
    that level to 0.05 dB."""
    z = node_heights(None)
    profile = Profile(z, c0 + g * z)
    found = Rays(profile, zs).eigenrays(zr, distance, 2)
    assert sorted(found.ground_bounces).count(1) == 1
    _assert_on_the_listener(profile, zs, zr, distance, found)
    assert [_level(found, i) for i in range(len(found))] == pytest.approx(
        [0.0] * len(found), abs=0.05
    )


def test_the_levels_of_rays_between_ends_either_side_of_a_row_in_level_air() -> None:
    """Downwind of a table whose wind is 1.8 m/s from 29.276 to 40.4 m and 4.8e-8 m/s
    less from there up to 140.1 m, where the air cools from 19.73 to 18.98 C (20.3 C at
    the ground), from a source 1 um above the row at 40.4 m to a listener 99 um below
    it and 13.6 m away: the gradient is -4e-9/s below the row and -4.4e-3/s above, and
    the direct ray leaves 9e-6 rad below the horizontal, where neighbouring floats of p
    give ranges 4e-5 m apart, more than the tolerance. Against the exact range of the
    profile, the levels over free field at their lengths of the direct ray and of the
    one off the ground are -0.835 and +0.015 dB; the one off the ground is found, every
    ray found reaches the listener, and each is at its level to 0.05 dB."""
    rows = [
        [0.0, 20.3, 0.99],
        [0.27, 19.73, 0.99],
        [28.9, 19.73, 0.9900000095367432],
        [29.276, 19.73, 1.8],
        [40.4, 19.73, 1.7999999523162842],
        [140.1, 18.98, 1.7999999523162842],
    ]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 20.3, np.array([1.0, 0.0]))
    found = Rays(profile, 40.400000999999996).eigenrays(40.39990099999999, 13.6, 2)
    off = [i for i in range(len(found)) if found.ground_bounces[i] == 1]
    assert len(off) == 1
    _assert_on_the_listener(profile, 40.400000999999996, 40.39990099999999, 13.6, found)
    for i in range(len(found)):
        expected = 0.015 if found.ground_bounces[i] else -0.835
        assert _level(found, i) == pytest.approx(expected, abs=0.05)


def test_the_levels_of_rays_between_ends_just_under_the_top_of_level_air() -> None:
    """Downwind of a table whose wind is 1.03 m/s up to 5.3 m and 4.78 m/s at 7 m, where
    the air is 23.2 C at the ground, 7.6e-7 C warmer from 1.544 to 3.97 m and 22.91 C
    from 5.3 m up, between a source and a listener 3.969 m up, 1 mm under the top of the
    layer whose speed does not change with height, and 2259.1 m apart: the direct ray runs
    straight along the layer, at the level of free field. Against the exact range of the
    profile, the levels over free field at their lengths of the other rays, by the way
    they leave and arrive and their passes off the ground, are those listed; every ray
    found but the straight one reaches the listener, and each is at its level to
    0.05 dB."""
    rows = [
        [0.0, 23.2, 1.03],
        [1.544, 23.200000762939453, 1.03],
        [3.97, 23.200000762939453, 1.03],
        [5.3, 22.91, 1.03],
        [7.009, 22.91, 4.78],
    ]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 23.2, np.array([1.0, 0.0]))
    height = 3.9690000000000003
    found = Rays(profile, height).eigenrays(height, 2259.1, 2)
    references = {
        (True, True, 0): 0.0,
        (False, True, 0): 0.0002,
        (True, False, 0): 0.3608,
        (True, True, 1): 0.2839,
        (False, False, 1): 0.2839,
        (False, True, 1): 0.2241,
        (True, False, 1): 0.5755,
        (True, True, 2): 0.4448,
        (False, False, 2): 0.4448,
        (True, False, 2): 0.6594,
    }
    straight = _on(found, True, True, 0)
    assert len(straight) == 1
    assert found.length[straight[0]] == 2259.1
    for i, b in enumerate(found.branch):
        expected = references[(b.up, b.arrives_up, b.cycle)]
        assert _level(found, i) == pytest.approx(expected, abs=0.05)
    _assert_on_the_listener(profile, height, height, 2259.1, found, tuple(straight))


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


def test_the_levels_of_rays_that_turn_back_in_a_warm_layer_2_km_away() -> None:
    """At 53 degrees to a table's wind (0.6 of it along the way), 2.96 m/s up to 141.8 m
    (3.9e-7 m/s more from 63.4 m) and 2.79 m/s from 188.9 m, where the air is 19.1 C up
    to 63.4 m, 20.93 C at 71.1 m and 20.02 C from 141.8 m, between a source 5.06 m and a
    listener 0.58 m up and 2134.3 m apart: a ray straight down to the listener and one
    off the ground turn back in the warm layer, where the width from the ranges of the
    rays either side changes by 2 % from rays half as far apart and the other way gives
    none. Against the exact range of the profile, the levels over free field at their
    lengths of all the rays, by the way they leave and arrive and their passes off the
    ground, are those listed; every ray found reaches the listener, and each is at its
    level to 0.15 dB."""
    rows = [
        [0.0, 19.1, 2.96],
        [63.4, 19.1, 2.960000388607037],
        [71.1, 20.93, 2.960000388607037],
        [141.8, 20.02, 2.960000388607037],
        [188.934, 20.02, 2.79],
        [247.2, 20.02, 2.7899999618530273],
    ]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 19.1, np.array([0.6, 0.8]))
    zs, zr = 5.062836911282339, 0.5767738159927351
    found = Rays(profile, zs).eigenrays(zr, 2134.3, 2)
    references = {
        (False, False, 0): 0.0,
        (False, True, 0): 0.0,
        (True, False, 0): 1.6281,
        (True, True, 1): 1.6506,
        (False, False, 1): 1.8379,
        (False, True, 1): 1.8633,
    }
    assert sorted((b.up, b.arrives_up, b.cycle) for b in found.branch) == sorted(references)
    for i, b in enumerate(found.branch):
        expected = references[(b.up, b.arrives_up, b.cycle)]
        assert _level(found, i) == pytest.approx(expected, abs=0.15)
    _assert_on_the_listener(profile, zs, zr, 2134.3, found)


@pytest.mark.parametrize(
    ("ground_c", "zr", "distance", "branch", "reference"),
    [
        (15.0, 4.21, 800.0, (True, False, 3), 7.641),
        (20.0, 4.61, 300.0, (True, False, 1), 6.106),
        (20.0, 1.5, 152.0, (True, True, 1), -0.740),
        (20.0, 5.81, 800.0, (False, False, 3), 9.122),
    ],
)
def test_the_level_of_a_ray_that_turns_just_above_a_node_over_an_end(
    ground_c, zr, distance, branch, reference
) -> None:
    """Downwind in the `windy` preset, from a source 5 m up (within a quarter of a layer
    under the profile's node at 5.063 m), the ray along ``branch`` turns just above that
    node, or above the node at 5.852 m over a listener 5.81 m up. Against the smooth
    profile's ray tube (quadrature in the elevation angle, and the ray equations shot
    and differenced, which agree to 0.002 dB), its level over free field at its length
    is that given (dB); the sampled profile's own is 10 to 15 dB lower there. The
    engine's is within 0.39 dB of it; the bound is 1 dB, well short of that step."""
    windy = Atmosphere.model_validate({"preset": "windy"})
    rays = Rays(Profile.of(windy, ground_c, np.array([1.0, 0.0])), 5.0)
    found = rays.eigenrays(zr, distance, 3)
    (i,) = _on(found, *branch)
    assert _level(found, i) == pytest.approx(reference, abs=1.0)


def test_the_levels_of_paths_to_seeded_listeners_against_the_smooth_profile() -> None:
    """Downwind in the `windy` preset at 20 °C, from a source 5 m up to 15 listeners at
    random 100 to 1000 m away and 0.5 to 12 m up (seed 2), against the smooth profile's
    paths (validation/curved_rays.py, which has 1300 listeners a family): the 66 paths
    not near a caustic are at most 0.52 dB off its levels, and the sums at the 14
    listeners none of whose paths is at most 0.32 dB off. The bounds are 1 and 0.5 dB; a
    ray turning just above the node over the source took one path 970 m away 2.17 dB
    off."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    import curved_rays

    weather = "windy_downwind"
    r = curved_rays.family(weather, 5.0, curved_rays.places(weather, 15, 2))
    assert r["unmatched"] == 0 and r["listeners_summed"] >= 12
    assert r["max_db"] <= 1.0
    assert r["sum_max_db"] <= 0.5


def test_the_level_over_a_node_just_above_the_source_does_not_step() -> None:
    """The same 300 m away, listeners 4.60 and 4.61 m up: the smooth profile's rays add
    up to 9.52 dB over free field at both; theirs do to within 0.1 dB of each other (the
    one ray turning over the node took 2.5 dB off the sum in that centimetre)."""
    windy = Atmosphere.model_validate({"preset": "windy"})
    rays = Rays(Profile.of(windy, 20.0, np.array([1.0, 0.0])), 5.0)
    sums = [
        10 * math.log10(np.sum(rays.eigenrays(zr, 300.0, 3).spreading) * 300.0**2)
        for zr in (4.60, 4.61)
    ]
    assert sums[1] == pytest.approx(sums[0], abs=0.1)


def test_the_levels_of_rays_in_a_duct_through_air_3_8e_12_off_level() -> None:
    """Upwind of a table whose wind is 1.377 m/s up to 1.56 cm, 1.391 m/s at 1.124 m and
    3.8e-12 m/s less at 1.438 m, 1.390 m/s at 1.836 m and 3.68 m/s at 20.56 m (15 C),
    between a source 1.2 m and a listener 1.3 m up, in the layer that far off level, and
    1000 m apart. Against the exact range of the profile, the levels over free field at
    their lengths of the direct rays that leave and arrive going up, leave going down and
    arrive going up, and leave going up and arrive going down are 0.000, 0.120 and
    0.784 dB, and that of the ray off the ground that arrives going up -40.03 dB, whose
    width from the ranges of the rays either side changes by more than itself from rays
    half as far apart, where the other way gives none. Those direct rays are found and
    each is at its level to 0.05 dB, as is the ray off the ground where it is found, and
    every ray found reaches the listener."""
    rows = [
        [0.0, 15.0, 1.377],
        [0.0156, 15.0, 1.377],
        [1.1244, 15.0, 1.3910851851131807],
        [1.4384, 15.0, 1.3910851851131807 - 3.839e-12],
        [1.836, 15.0, 1.3897448975807714],
        [20.56, 15.0, 3.6814268499198146],
    ]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 15.0, np.array([-1.0, 0.0]))
    found = Rays(profile, 1.2).eigenrays(1.3, 1000.0, 2)
    for way, expected in (
        ((True, True, 0), 0.0),
        ((False, True, 0), 0.120),
        ((True, False, 0), 0.784),
    ):
        direct = [i for i in _on(found, *way) if found.ground_bounces[i] == 0]
        assert len(direct) == 1
        assert _level(found, direct[0]) == pytest.approx(expected, abs=0.05)
    for i in _on(found, False, True, 0):
        if found.ground_bounces[i] == 1:
            assert _level(found, i) == pytest.approx(-40.03, abs=0.05)
    _assert_on_the_listener(profile, 1.2, 1.3, 1000.0, found)


def test_the_level_of_a_ray_that_turns_just_under_air_level_but_for_a_hair() -> None:
    """Upwind of a table whose wind rises from 1.609 m/s at the ground to 1.873 m/s at
    0.23 m, holds there to 0.35 m (1.3e-11 m/s more), falls to 1.827 m/s at 1.17 m and
    stays between 1.824 and 1.830 m/s up to 4.3 m, then rises to 1.870 m/s at 4.87 m and
    2.46 m/s at 51.7 m (15 C), between a source 26.68 m and a listener 1 um higher and
    2666.3 m apart: the one ray leaves going down, crosses the air whose speed changes by
    1.1e-10 m/s per metre from 0.23 to 0.35 m up almost level and turns back up 0.19 m
    above the ground; the ranges of rays that turn a little higher grow without bound.
    Against the exact range of the profile, its level over free field at its length is
    -13.251 dB; it is found, at that level to 0.05 dB, and reaches the listener."""
    rows = [
        [0.0, 15.0, 1.609],
        [0.1079, 15.0, 1.7190481183040909],
        [0.2297, 15.0, 1.8730965701257702],
        [0.3476, 15.0, 1.873096570138857],
        [1.1727, 15.0, 1.8272922145800337],
        [1.2228, 15.0, 1.8263528372808842],
        [1.7875, 15.0, 1.8265131353829955],
        [1.8526, 15.0, 1.8247728427730359],
        [2.6495, 15.0, 1.8249833614040154],
        [2.6941, 15.0, 1.827546875399252],
        [2.8196, 15.0, 1.827546875399252],
        [3.0504, 15.0, 1.827894024124376],
        [3.4506, 15.0, 1.8292313089550494],
        [4.2952, 15.0, 1.829231308942329],
        [4.8699, 15.0, 1.8696592625129773],
        [51.7, 15.0, 2.4600799228198786],
    ]
    profile = Profile.of(Atmosphere.model_validate({"table": rows}), 15.0, np.array([-1.0, 0.0]))
    found = Rays(profile, 26.68).eigenrays(26.680001, 2666.3, 2)
    assert len(found) == 1
    assert _level(found, 0) == pytest.approx(-13.251, abs=0.05)
    _assert_on_the_listener(profile, 26.68, 26.680001, 2666.3, found)


def test_each_caustic_advances_the_phase_by_a_quarter_period() -> None:
    """A ray past a fold caustic leads one that has touched none by a quarter period:
    stationary phase on the Airy integral gives it exp(-i pi/4) against exp(i pi/4) with
    time as exp(-i omega t), so a factor j with time as exp(j omega t), as the early
    paths have it (a delay tau being exp(-j omega tau); a quarter-period delay would be
    -j). Downwind of a source 2 m up, 200 m away 1.5 m up over a ground that reflects
    everything in phase, some of the paths off the ground touch a caustic."""
    source, listener = np.array([0.0, 0.0, 2.0]), np.array([200.0, 0.0, 1.5])
    scene = Scene.model_validate(
        {
            "name": "downwind",
            "geometry": {"ground": {"extent": [[-50.0, -20.0], [250.0, 20.0]], "sky": 20.0}},
            "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
            "assignments": {"ground": "hard"},
            "atmosphere": {"preset": "windy"},
            "sources": [{"name": "s", "position": list(source)}],
            "listeners": [{"name": "l", "position": list(listener)}],
        }
    )
    room = Room.from_scene(scene)
    refraction = refraction_for(
        TraceModel.from_room(room),
        source,
        speed_of_sound=room.speed_of_sound(),
        air=np.zeros(8),
        coherent=2,
    )
    rays = refraction.rays(refraction._towards(np.array([1.0, 0.0])))  # the wind's way
    found = rays.eigenrays(1.5, 200.0, 2)
    assert set(found.caustics.tolist()) == {0, 1}
    ones = np.ones(8)
    corners = np.array([source[:2], listener[:2]])
    paths, drawn, _ = refraction._curved(rays, found, 1.5, corners, (), listener, ones, ones)
    assert drawn.all()
    turn = np.broadcast_to((1j**found.caustics)[:, None, None], paths.spectrum.shape)
    np.testing.assert_allclose(paths.spectrum, turn, rtol=0, atol=1e-12)


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
    to 1e-9: the ray's sine is 0 where it grazes the ground, not taken from p (1 / 331.3),
    whose product with the speed there rounds short of 1."""
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


def _windy(h) -> np.ndarray:
    """The ``windy`` preset's effective speed of sound downwind (m/s) at heights h (m)."""
    windy = Atmosphere.model_validate({"preset": "windy", "wind": {"direction_deg": 0}})
    heights = np.atleast_1d(np.asarray(h, dtype=float))
    return np.asarray(windy.sound_speed(heights, 20.0, np.array([1.0, 0.0])), dtype=float)


def _band_db(pressure: np.ndarray, ranges: np.ndarray, zs: float, zr: float) -> np.ndarray:
    """The level over the free field's (dB) of the band's slices ``pressure`` (F, n)."""
    return 10 * np.log10(np.mean(np.abs(pressure) ** 2, axis=0) * np.hypot(ranges, zs - zr) ** 2)


@pytest.mark.parametrize("independent", [True, False])
def test_steep_sound_in_wind_against_the_fast_field_program(independent) -> None:
    """The phase shift the narrow-angle equation takes at each step (the split step of
    Salomons 2001, app. H) strays for sound that comes in steeply, at any step: from a
    source 30 m up in the ``windy`` preset downwind over grass, the 100 Hz third-octave
    band 47 to 117 m away (the reflection 34 to 15 degrees up) is up to 1.01 dB from the
    fast field program (validation/ffp.py) at the full step (0.91 dB in one march) and
    0.45 dB in steps half as long (0.35 dB). The wide-angle equation, which takes these
    points, is within 0.1 dB of it at every point."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    import ffp

    from acoustic_spaces.engine.diffraction import band_frequencies

    f = band_frequencies(bands="third_octave")[0][3]
    z = np.conj(surface_impedance(f, 200.0))
    zs, zr = 30.0, 1.5
    ranges = np.array([47.0, 53.0, 59.0, 66.0, 74.0, 83.0, 93.0, 104.0, 117.0])
    heights = np.full(len(ranges), zr)
    pressure = march(f, _windy, lambda r: z, zs, ranges, heights, 50.0, independent=independent)
    reference = [ffp.pressure(fi, _windy, zs, ranges, zr, zi) for fi, zi in zip(f, z, strict=True)]
    difference = _band_db(pressure, ranges, zs, zr) - _band_db(np.array(reference), ranges, zs, zr)
    assert np.max(np.abs(difference)) < 0.1


def test_flat_sound_and_still_air_keep_the_narrow_angle_equation(monkeypatch) -> None:
    """Points the reflection reaches at 15 degrees or flatter get what the narrow-angle
    march to all the points gives, bit for bit, whatever steep points share the run; in
    wind the steep points get what the wide-angle equation gives them on its own, and in
    still air, where a step is exact but at the ground, they keep the narrow-angle march,
    as they do through weather zones all of still air, but not one windy zone among them.
    Shorter steps are no closer: from 1 m up to 1.5 m up 200 m away, downwind in the
    ``windy`` preset over grass, the 63 Hz band is within 0.3 dB of the wide-angle
    equation at the full step (0.21 dB under the fast field program), and more than 0.6 dB
    over it at a 32nd of the step on the same heights (0.93 dB over the program)."""
    from acoustic_spaces.engine import parabolic, wide_angle
    from acoustic_spaces.engine.diffraction import band_frequencies

    f = band_frequencies()[0][0]
    z = np.conj(surface_impedance(f, 200.0))
    still = lambda h: np.full(np.shape(h), C)  # noqa: E731
    # 6 m up 20 m away the reflection comes in 19 degrees up.
    ranges, heights = np.array([20.0, 50.0, 200.0]), np.array([6.0, 1.5, 1.5])
    flat = np.array([False, True, True])
    for speed in (_windy, still):
        for independent in (True, False):
            args = (f, speed, lambda r: z, 1.0, ranges, heights, 30.0)
            new = march(*args, independent=independent)
            with monkeypatch.context() as m:
                m.setattr(parabolic, "STEEP_DEG", 90.0)  # no point steep
                narrow = march(*args, independent=independent)
            assert np.array_equal(new[:, flat], narrow[:, flat])
            assert np.array_equal(new[:, ~flat], narrow[:, ~flat]) == (speed is still)
            if speed is _windy:
                steep = (f, speed, lambda r: z, 1.0, ranges[~flat], heights[~flat], 30.0)
                alone = wide_angle.march(*steep, independent=independent)
                assert np.array_equal(new[:, ~flat], alone)
    calm = lambda h: np.full(np.shape(h), C)  # noqa: E731
    for zones, kept in (((still, calm), True), ((still, _windy), False)):
        args = (f, still, lambda r: z, 1.0, ranges, heights, 30.0)
        zoned = {"speed_at": lambda r, zones=zones: zones[int(r // 25.0) % 2]}
        new = march(*args, **zoned)
        with monkeypatch.context() as m:
            m.setattr(parabolic, "STEEP_DEG", 90.0)
            narrow = march(*args, **zoned)
        assert np.array_equal(new[:, flat], narrow[:, flat])
        assert np.array_equal(new[:, ~flat], narrow[:, ~flat]) == kept
    args = (f, _windy, lambda r: z, 1.0, ranges[2:], heights[2:], 30.0)
    wide = _band_db(wide_angle.march(*args), ranges[2:], 1.0, 1.5)[0]
    assert abs(_band_db(march(*args), ranges[2:], 1.0, 1.5)[0] - wide) < 0.3
    monkeypatch.setattr(parabolic, "RANGE_STEP_WAVELENGTHS", parabolic.RANGE_STEP_WAVELENGTHS / 32)
    monkeypatch.setattr(parabolic, "RANGE_STEP_MIN_M", parabolic.RANGE_STEP_MIN_M / 32)
    assert _band_db(march(*args), ranges[2:], 1.0, 1.5)[0] - wide > 0.6


# -- simulations -----------------------------------------------------------------------


def test_still_air_changes_nothing() -> None:
    # The lawn scatters 0.1: over a ground that scatters nothing there is no late part.
    listeners = ((60.0, 0.0, 1.5), (-60.0, 5.0, 1.5))
    plain = simulate(Room.from_scene(scattering_lawn(field(None, listeners))))
    neutral = simulate(Room.from_scene(scattering_lawn(field({"preset": "neutral"}, listeners))))
    calm = simulate(Room.from_scene(scattering_lawn(field({"wind": {"speed_m_s": 0}}, listeners))))
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


def test_sources_share_the_air_s_profiles_and_keep_no_plan_once_finished(monkeypatch) -> None:
    """The sources of a run find each profile of the air once, as a source finds it on
    its own; and of the plans waiting for the parabolic equation, each is gone once its
    source has taken the pressures, before the next one's turn."""
    from acoustic_spaces.engine import refracted

    scene = field({"preset": "windy"}, ((80.0, 0.0, 1.5), (-60.0, 20.0, 1.5)))
    item = scene.sources[0]
    places = ((0.0, 0.0), (-30.0, 5.0), (25.0, -8.0))
    sources = [
        item.model_copy(update={"name": f"s{k}", "position": (x, y, 2.0)})
        for k, (x, y) in enumerate(places)
    ]
    room = Room.from_scene(scene.model_copy(update={"sources": sources}))
    bent, plans, waiting = [], [], []
    plan_, finish_ = refracted.Refraction.plan, refracted.Planned.finish

    def plan(self, *args):
        planned = plan_(self, *args)
        bent.append(self)
        plans.append(weakref.ref(planned))
        return planned

    def finish(self, pressures):
        waiting.append(sum(ref() is not None for ref in plans))
        return finish_(self, pressures)

    monkeypatch.setattr(refracted.Refraction, "plan", plan)
    monkeypatch.setattr(refracted.Planned, "finish", finish)
    simulate(room, share_waves=[s.name for s in sources])
    assert waiting == [3, 2, 1]
    assert all(ref() is None for ref in plans)
    profiles = bent[0]._profiles
    assert len(bent) == 3 and all(refraction._profiles is profiles for refraction in bent)
    assert len(profiles) > 1
    own = refraction_for(
        TraceModel.from_room(room),
        np.array(sources[2].position),
        speed_of_sound=room.speed_of_sound(),
        air=np.zeros(8),
    )
    for key, profile in profiles.items():
        for name in ("z", "c", "rows", "gradient", "trend", "corners"):
            a, b = getattr(profile, name), getattr(own.profile(key), name)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert own._profiles is not profiles and own._profiles.keys() == profiles.keys()


def test_uniform_rows_trace_the_late_rays_as_straight_ones(monkeypatch) -> None:
    """The compiled tracer's bent rays, in an atmosphere where they go straight, give
    the straight tracer's late energy: the same ray paths, times and losses (without the
    listeners' gather of the first reflections, which bent rays leave to the rain). The
    lawn scatters 0.1: over a ground that scatters nothing there is no late part."""
    pytest.importorskip("acoustic_spaces._native")
    scene = scattering_lawn(field(None, ((60.0, 0.0, 1.5), (-30.0, 10.0, 3.0))))
    room = Room.from_scene(scene)
    monkeypatch.setattr(late, "FIRST_GATHER", 0)
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


def downwind(
    rays: int,
    max_order: int = 1,
    scattering: float = 0.0,
    method: str = "vector",
    seed: int = 1,
    barriers: tuple = (),
) -> Scene:
    """The fast field program's scene (``validation/benchmark_1995.py``): a source 2 m
    up over grass, listeners 1.5 m up 200 and 300 m downwind in the windy preset."""
    material = {
        "impedance": {"model": "delany_bazley", "flow_resistivity": 200.0},
        "scattering": scattering,
    }
    return Scene.model_validate(
        {
            "name": "downwind",
            "geometry": {
                "ground": {"extent": [[-50.0, -50.0], [350.0, 50.0]], "sky": 50.0},
                "barriers": list(barriers),
            },
            "materials": {"field": material},
            "default_material": "field",
            "sources": [{"name": "s", "position": [0.0, 0.0, 2.0]}],
            "listeners": [{"name": f"{r:g}", "position": [r, 0.0, 1.5]} for r in (200, 300)],
            "atmosphere": {"preset": "windy", "wind": {"direction_deg": 0.0}},
            "simulation": {
                "seed": seed,
                "ir_length_s": 1.2,
                "early": {"method": "image_sources", "max_order": max_order, "diffraction": "none"},
                "late": {"rays": rays},
                "scattering": {"method": method},
            },
        }
    )


@pytest.mark.parametrize("scattering, method", [(0.0, "vector"), (0.1, "random")])
def test_the_late_rays_leave_out_the_ground_bounces_the_bent_early_part_holds(
    scattering, method
) -> None:
    """Downwind the early stage holds the curved paths off the ground with up to
    MAX_GROUND_BOUNCES bounces whatever its order: the late rays leave all of them out,
    so the early order changes the late part not at all, and over a ground that scatters
    nothing more late rays add nothing (they added 11 dB at 500 Hz with 20,000 rays, the
    paths of 2 and 3 bounces counted twice). Over a ground that scatters, 200 and 20,000
    rays give the 500 Hz level within the noise of its scattered sound (the paths counted
    twice added 9 and 6 dB; at 250 Hz the scattered sound near a listener in bent air
    comes in bursts, a known issue, and is not compared)."""
    run = {
        (rays, order): simulate(
            Room.from_scene(downwind(rays, order, scattering, method)), threads=1
        )
        for rays, order in ((200, 1), (20_000, 1), (20_000, 2))
    }
    for key in run[200, 1].responses:
        one, two = run[20_000, 1][key], run[20_000, 2][key]
        assert np.array_equal(one.early.energy, two.early.energy)
        assert np.array_equal(one.late, two.late)
        if scattering == 0.0:
            # Nothing scatters and every specular path is the early stage's: the late
            # part is empty, however many rays.
            assert not one.late.any() and not run[200, 1][key].late.any()
        else:
            few, many = (run[rays, 1][key].relative_to_free_field()[3] for rays in (200, 20_000))
            assert abs(few - many) < 0.5


def test_a_wall_s_reflection_above_the_early_order_stays_with_the_late_rays() -> None:
    """In wind the early stage holds a wall's reflection with bounces off the ground only
    up to its order (the wall's counting as one and the ground's as one): with order 1
    the paths off the wall and the ground are the late rays', with order 2 the early
    stage's. The late rays' secondary sources radiate the specular sound of each
    reflection to the listeners, so the late part with order 1 has more, from when the
    first such path arrives on."""
    wall = {"name": "wall", "path": [[-40.0, 15.0], [340.0, 15.0]], "height": 10.0}
    runs = []
    for order in (1, 2):
        scene = downwind(2_000, order, barriers=(wall,))
        scene.simulation.late.reception = "secondary_sources"
        runs.append(simulate(Room.from_scene(scene), threads=1))
    for key in runs[0].responses:
        one, two = (run[key] for run in runs)
        step = one.time_step
        more = one.late.sum(axis=0) - two.late.sum(axis=0)
        assert more.min() >= 0.0 and more.sum() > 0.1 * two.late.sum()
        added = two.early.delay_s[~np.isin(two.early.delay_s, one.early.delay_s)]
        assert np.flatnonzero(more)[0] * step >= added.min() - step


def test_a_coverage_map_point_in_wind_reads_as_a_listener_there() -> None:
    """A map point 100 m downwind of a source 1 m up over grass that scatters nothing, and
    a listener there, within 0.1 dB in every band (the map's late part added the bounces
    the early stage holds: 5 dB at 250 Hz with this seed)."""
    from acoustic_spaces.engine.coverage import coverage_map

    data = {
        "name": "map",
        "geometry": {"ground": {"extent": [[-10.0, -10.0], [130.0, 10.0]], "sky": 20.0}},
        "materials": {"field": {"impedance": "grass", "scattering": 0.0}},
        "default_material": "field",
        "sources": [{"name": "s", "position": [0.0, 0.0, 1.0]}],
        "listeners": [{"name": "l", "position": [100.0, 0.0, 1.5]}],
        "atmosphere": {"preset": "windy", "wind": {"direction_deg": 0.0}},
        "coverage": {"spacing": 20.0, "height": 1.5, "rays": 2000},
        "simulation": {
            "seed": 2,
            "ir_length_s": 0.5,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
            "late": {"rays": 2000},
        },
    }
    room = Room.from_scene(Scene.model_validate(data))
    mapped = coverage_map(room, threads=1)
    at = int(np.argmin(np.linalg.norm(mapped.grid.points - [100.0, 0.0, 1.5], axis=1)))
    assert np.allclose(mapped.grid.points[at], [100.0, 0.0, 1.5])
    listener = simulate(room, threads=1)["s", "l"].relative_to_free_field()
    point = np.asarray(mapped.maps["s"].values["re_free_field"][at])
    np.testing.assert_allclose(point, listener, atol=0.1)


def test_still_air_traces_the_late_part_as_before() -> None:
    """Without an atmosphere that bends the sound every reflection counts towards the
    early order as before: the late part of a closed room and of a field in neutral air
    as computed before the late rays learnt the bent early stage's paths (the field's
    lawn scatters 0.1: over a ground that scatters nothing there is no late part)."""
    box = Scene.model_validate(
        {
            "name": "box",
            "geometry": {"box": [6.0, 5.0, 3.0]},
            "materials": {"m": {"absorption": [0.1] * 8, "scattering": 0.2}},
            "default_material": "m",
            "sources": [{"name": "s", "position": [1.5, 1.2, 1.4]}],
            "listeners": [{"name": "l0", "position": [4.2, 3.1, 1.2]}],
            "simulation": {"seed": 3, "ir_length_s": 0.2, "late": {"rays": 300}},
        }
    )
    neutral = scattering_lawn(field({"preset": "neutral"}, ((60.0, 0.0, 1.5), (-30.0, 10.0, 3.0))))
    before = (1.6025686813266748, 8.349338904503138e-07, 8.012851349037927e-06)
    found = []
    for scene in (box, neutral):
        result = simulate(Room.from_scene(scene), threads=1)
        found += [float(result[key].late.sum()) for key in sorted(result.responses)]
    np.testing.assert_allclose(found, before, rtol=1e-6)


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
    wave_bands = 6  # 63 Hz to 2 kHz
    carrying = np.flatnonzero(early.energy[:, :wave_bands].sum(axis=1) > 0.0)
    assert len(carrying) == 1 and len(early) > 1
    silent = np.flatnonzero(early.energy[:, :wave_bands].sum(axis=1) == 0.0)
    assert np.all(early.energy[silent, wave_bands:] > 0.0)  # the rays' bands are theirs
    computed = 10.0 * np.log10(
        early.energy[carrying[0], :wave_bands] / response.free_field[:wave_bands]
    )
    read = response.relative_to_free_field()[:wave_bands]
    np.testing.assert_allclose(read, computed, atol=0.02)


def test_octave_and_third_octave_runs_end_the_equation_at_the_same_frequency(
    monkeypatch,
) -> None:
    """The open ground's bands from the parabolic equation end at a frequency, the upper
    edge of the 2 kHz octave (2828 Hz), in octave and third-octave runs alike: 100 m
    downwind the equation runs in the octave bands from 63 Hz to 2 kHz and in the third
    octaves from 50 Hz to 2.5 kHz (its upper edge 2806 Hz, nominal centres), whose
    frequencies all lie in those octaves, the highest within 5 % of the edge. One more
    run takes the top band on through the crossover above it, past the edge, and the
    lowest band's run goes on down to an octave below it (44 Hz), for the response's
    shape there."""
    import acoustic_spaces.engine.parabolic as parabolic

    marched = []
    run = parabolic.march

    def spy(frequencies, *args, **kwargs):
        marched.append(np.atleast_1d(frequencies).copy())
        return run(frequencies, *args, **kwargs)

    monkeypatch.setattr(parabolic, "march", spy)
    join = 2000.0 * math.sqrt(2.0)
    for bands, runs in (("octave", 6), ("third_octave", 18)):
        marched.clear()
        simulate(Room.from_scene(field({"preset": "windy"}, bands=bands)), threads=1)
        skirt = [m for m in marched if m.min() > join / 1.05]
        assert len(marched) == runs + 1 and len(skirt) == 1, bands
        frequencies = np.concatenate([m for m in marched if m.min() <= join / 1.05])
        assert join / 1.05 < frequencies.max() < join, bands
        under = frequencies[frequencies < 44.0]
        assert len(under) and under.min() > 22.0 and len(marched[0]) > len(under), bands


def test_the_shadow_deepens_alike_in_octave_and_third_octave_runs() -> None:
    """100 m upwind in the wind, deep in the shadow, the bands above the parabolic
    equation's are its top octave band's level deepened octave by octave as over the
    octave below: read in octave bands, a third-octave run's 4 and 8 kHz agree with an
    octave run's no worse than the computed 1 and 2 kHz do (whose three third octaves
    take other frequencies of the octave)."""
    levels = []
    for bands in ("octave", "third_octave"):
        scene = field({"preset": "windy"}, ((-100.0, 0.0, 1.5),), bands=bands)
        response = simulate(Room.from_scene(scene), threads=1)["s", "l0"]
        energy = response.early.energy.sum(axis=0).reshape(8, -1).sum(axis=1)
        free = np.asarray(response.free_field).reshape(8, -1).sum(axis=1)
        levels.append(10.0 * np.log10(energy / free))
    difference = np.abs(levels[1] - levels[0])
    assert levels[0][6] < levels[0][5] - 10.0  # deep in the shadow
    assert difference[6:].max() <= difference[4:6].max()


def test_a_silent_top_octave_leaves_the_bands_above_it_silent(monkeypatch) -> None:
    """Where the parabolic equation's top octave band comes out silent (as in a shadow
    deeper than its numbers reach), the third octaves above it are silent too, not the
    0 ** -x of a slope of zero."""
    from acoustic_spaces.engine import refracted

    place = refracted.Refraction._place

    def silent_top(self, wave, pressure, energy, parts, listeners, runs=None, blur=None):
        energy = np.array(energy, dtype=float)
        octave = refracted.octave_of(self.model.bands)
        top = octave[np.flatnonzero(wave.bands)[-1]]
        energy[octave == top] = 0.0
        return place(self, wave, pressure, energy, parts, listeners, runs, blur)

    monkeypatch.setattr(refracted.Refraction, "_place", silent_top)
    scene = field({"preset": "windy"}, ((-100.0, 0.0, 1.5),), bands="third_octave")
    response = simulate(Room.from_scene(scene), threads=1)["s", "l0"]
    energy = response.early.energy.sum(axis=0)
    assert np.all(np.isfinite(energy))
    assert np.all(energy[THIRD_OCTAVE_BANDS_HZ.index(3150) :] == 0.0)


#: The level re free field (dB) at 1 and 2 kHz, the band's energy averaged over 24
#: frequencies, of ``validation/ffp.py`` (its top 200 m up): ``windy`` downwind, a
#: source 2 m and a listener 1.5 m up, 119.5, 120 and 120.5 m apart; ``clear_night``,
#: a source 0.5 m up, a listener 250 m away 2.00, 2.05 and 2.10 m up.
FFP_WINDY_120 = [[3.710, 2.507], [3.713, 2.517], [3.716, 2.528]]
FFP_NIGHT_250 = [[-4.087, 0.887], [-4.063, 0.691], [-4.045, 0.525]]


def caustic_scene(preset: str, zs: float, listeners, bands: str = "octave", **early) -> Scene:
    """A source ``zs`` up over grass (Delany and Bazley, 200 kPa s/m2) in ``preset``,
    downwind, and listeners; nothing from the late part; ``early`` added to its
    ``simulation.early``."""
    far = max(p[0] for p in listeners)
    return Scene.model_validate(
        {
            "name": "caustic",
            "geometry": {"ground": {"extent": [[-50.0, -50.0], [far + 50.0, 50.0]], "sky": 50.0}},
            "materials": {
                "field": {
                    "impedance": {"model": "delany_bazley", "flow_resistivity": 200.0},
                    "scattering": 0.0,
                }
            },
            "default_material": "field",
            "sources": [{"name": "s", "position": [0.0, 0.0, zs]}],
            "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
            "air": {"temperature_c": 20.0, "humidity_percent": 50.0},
            "atmosphere": {"preset": preset, "wind": {"direction_deg": 0.0}},
            "simulation": {
                "seed": 1,
                "bands": bands,
                "early": {
                    "method": "image_sources",
                    "max_order": 1,
                    "diffraction": "none",
                    **early,
                },
                "late": {"rays": 200},
                "ir_length_s": 1.0,
            },
        }
    )


#: The level re free field (dB) at 1 and 2 kHz of ``validation/ffp.py`` (its top 200 m
#: up), the band's energy averaged over 48 frequencies: ``windy`` downwind, a source 8 m
#: and a listener 7.5 m up, 30 m apart (the ground's fringes some 85 Hz apart). Over 8
#: frequencies an octave it is -1.50 and -1.46 dB, over 24 1.00 and 0.68.
FFP_HIGH_30 = [1.001, 0.635]


def test_the_equation_s_bands_average_the_ground_s_fringes_from_1_khz() -> None:
    """Over the open ground the parabolic equation's 1 and 2 kHz bands take 24
    frequencies an octave, enough for the ground's fringes high above it: within 0.1 dB
    of a fast field program's band (8 an octave, as below 1 kHz, missed it by 2.5 and
    2.1 dB)."""
    scene = caustic_scene("windy", 8.0, [(30.0, 0.0, 7.5)])
    level = simulate(Room.from_scene(scene), threads=1)["s", "l0"].relative_to_free_field()
    np.testing.assert_allclose(level[4:6], FFP_HIGH_30, atol=0.1)


@pytest.mark.parametrize(
    ("preset", "zs", "walk", "ends", "ffp"),
    [
        # 119.8 to 119.95 m downwind in the wind, 1.5 m up: two rays off the ground appear
        # (between 119.86 and 119.87 m); the program's level at 119.5, 120 and 120.5 m.
        (
            "windy",
            2.0,
            [(119.8 + 0.01 * k, 0.0, 1.5) for k in range(16)],
            [(119.5, 0.0, 1.5), (120.0, 0.0, 1.5), (120.5, 0.0, 1.5)],
            FFP_WINDY_120,
        ),
        # 250 m away on a clear night, 2.00 to 2.10 m up: two rays off the ground merge.
        (
            "clear_night",
            0.5,
            [(250.0, 0.0, 2.0 + 0.01 * k) for k in range(11)],
            [(250.0, 0.0, 2.0), (250.0, 0.0, 2.05), (250.0, 0.0, 2.1)],
            FFP_NIGHT_250,
        ),
    ],
)
def test_the_level_runs_smoothly_through_a_caustic_up_to_2_khz(preset, zs, walk, ends, ffp):
    """Where the rays off the ground merge and vanish (a caustic) the bands up to 2 kHz
    change by at most 0.1 dB a centimetre along ``walk``, as a fast field program's do
    (``validation/ffp.py``, 24 frequencies a band), and are within 0.2 dB of it at
    ``ends`` (the rays alone stepped by 12 and 14 dB a centimetre)."""
    listeners = walk + ends
    result = simulate(Room.from_scene(caustic_scene(preset, zs, listeners)), threads=1)
    level = np.array(
        [result["s", f"l{k}"].relative_to_free_field()[:6] for k in range(len(listeners))]
    )
    assert np.abs(np.diff(level[: len(walk)], axis=0)).max() < 0.1
    np.testing.assert_allclose(level[len(walk) :, 4:6], ffp, atol=0.2)  # 1 and 2 kHz


#: ``windy`` downwind, a source 2 m and a listener 1.5 m up, 100 m apart: the level re
#: free field (dB) at 1 and 2 kHz of ``validation/ffp.py`` (its top 200 m up, the band's
#: energy averaged over 24 frequencies), and that of the rays, which reach it, as
#: release 4.2.1 gave it (measured on the code before the parabolic equation took those
#: bands over).
FFP_WINDY_100 = [3.575, 2.338]
RAYS_WINDY_100 = [0.4847, -0.8237]


def test_the_open_ground_s_frequencies_choose_how_1_and_2_khz_are_found() -> None:
    """``simulation.early.open_ground_frequencies``: 24 an octave at 1 and 2 kHz are
    within 0.1 dB of a fast field program where the rays are 3.1 dB off, as the rays
    alone (``rays``) give it exactly as before; 8 an octave fall within 0.1 dB of the
    program there too, but not on 24's levels. The other bands do not change there, where
    rays reach the listener."""
    levels = {}
    for choice in (24, 8, "rays"):
        scene = caustic_scene("windy", 2.0, [(100.0, 0.0, 1.5)], open_ground_frequencies=choice)
        result = simulate(Room.from_scene(scene), threads=1)
        levels[choice] = result["s", "l0"].relative_to_free_field()
    np.testing.assert_allclose(levels[24][4:6], FFP_WINDY_100, atol=0.1)
    np.testing.assert_allclose(levels["rays"][4:6], RAYS_WINDY_100, atol=0.01)
    assert np.all(np.asarray(FFP_WINDY_100) - levels["rays"][4:6] > 3.0)
    np.testing.assert_allclose(levels[8][4:6], FFP_WINDY_100, atol=0.1)
    assert np.abs(levels[8][4:6] - levels[24][4:6]).min() > 0.005
    for choice in (8, "rays"):
        others = np.r_[0:4, 6:8]
        np.testing.assert_allclose(levels[choice][others], levels[24][others], atol=1e-9)


#: ``windy`` upwind of a source 2 m up over a lawn that scatters 0.1
#: (:func:`scattering_lawn`), listeners 1.5 m up 10, 20 and 70 m away, the shadow
#: starting 31.4 m away: the level re free field (dB) at 1 and 2 kHz as release 4.2.1
#: gave it (measured on the code before the parabolic equation took those bands over):
#: from the rays short of half the way to the shadow, from the equation beyond.
RAYS_WINDY_UPWIND = [[0.2317, 0.5340], [1.9875, 1.0129], [-9.7644, -11.8441]]


def test_the_open_ground_s_rays_give_way_to_the_equation_near_a_shadow() -> None:
    """With ``rays`` the open ground's 1 and 2 kHz bands come from the rays short of
    half the way to the shadow and from the parabolic equation at the early paths' own
    frequencies (as with 8) from there on and in the shadow, as before."""
    early = {"method": "image_sources", "max_order": 1, "diffraction": "none"}
    listeners = ((-10.0, 0.0, 1.5), (-20.0, 0.0, 1.5), (-70.0, 0.0, 1.5))
    levels = {}
    for choice in (8, "rays"):
        scene = scattering_lawn(
            field(
                {"preset": "windy"},
                listeners,
                early={**early, "open_ground_frequencies": choice},
            )
        )
        result = simulate(Room.from_scene(scene), threads=1)
        levels[choice] = np.array(
            [result["s", f"l{k}"].relative_to_free_field()[4:6] for k in range(3)]
        )
    np.testing.assert_allclose(levels["rays"], RAYS_WINDY_UPWIND, atol=0.005)
    np.testing.assert_allclose(levels[8][1:], levels["rays"][1:], atol=0.002)
    assert np.abs(levels[8][0] - levels["rays"][0]).min() > 0.015


def test_the_open_ground_s_frequencies_move_4_and_8_khz_in_the_shadow_alone() -> None:
    """The bands above 2 kHz do not change with ``open_ground_frequencies`` where rays
    reach the listener; in the shadow they deepen from the 1 and 2 kHz octave bands
    (the 2 kHz level and its fall from 1 kHz per octave), so there they move with them:
    by the 2 kHz change plus once (4 kHz) and twice (8 kHz) the change of that fall. The
    lawn scatters 0.1, so that the levels hold a late part."""
    early = {"method": "image_sources", "max_order": 1, "diffraction": "none"}
    listeners = ((-10.0, 0.0, 1.5), (-100.0, 0.0, 1.5))  # lit, and deep in the shadow
    levels = {}
    for choice in (24, 8):
        scene = scattering_lawn(
            field(
                {"preset": "windy"}, listeners, early={**early, "open_ground_frequencies": choice}
            )
        )
        result = simulate(Room.from_scene(scene), threads=1)
        levels[choice] = np.array([result["s", f"l{k}"].relative_to_free_field() for k in range(2)])
    change = levels[8] - levels[24]
    np.testing.assert_allclose(change[0][6:], 0.0, atol=1e-9)
    fall = change[1][5] - change[1][4]
    np.testing.assert_allclose(change[1][6:], change[1][5] + fall * np.array([1, 2]), atol=1e-3)
    assert abs(change[1][7]) > 0.3


def _open_runs(monkeypatch) -> list:
    """The frequencies (Hz) of each run of the parabolic equation, which gives the
    pressure 1 everywhere instead of marching."""
    from acoustic_spaces.engine import parabolic

    runs = []

    def flat(frequencies, speed, impedance, zs, ranges, heights, top, **options):
        runs.append(np.atleast_1d(frequencies).copy())
        return np.ones((len(runs[-1]), len(ranges)), dtype=complex)

    monkeypatch.setattr(parabolic, "march", flat)
    return runs


def test_the_open_ground_s_frequencies_reach_listeners_maps_and_walks(monkeypatch) -> None:
    """The setting reaches listeners, coverage maps and the places of a walking source
    alike: 24 or 8 frequencies in each run at 1 and 2 kHz over the open ground, and
    with ``rays`` none there, the source at the ground's end, everything downwind of it
    or across the wind, short of any shadow. A listener and a walk, whose responses are
    rendered from the spectra, take one more run past the top band at its spacing (10
    frequencies with 24, 4 with 8, 4 above the 500 Hz band with ``rays``); a map only
    where its band filters read that crossover, with ``rays``. A map's point and a
    listener there so keep the same samples of the equation's field in the bands read
    through the filters (up to 500 Hz), from which the time parameters there are read."""
    from acoustic_spaces.engine import refracted
    from acoustic_spaces.engine.coherent import FILTERED_MAX_BAND_HZ, band_count
    from acoustic_spaces.engine.coverage import coverage_map
    from acoustic_spaces.scene import Coverage
    from acoustic_spaces.walkthrough import simulate_walk

    runs = _open_runs(monkeypatch)
    kept, stage = {}, [""]
    place = refracted.Refraction._place

    def keep(self, wave, pressure, energy, parts, listeners, runs=None, blur=None):
        place(self, wave, pressure, energy, parts, listeners, runs, blur)
        if tuple(listeners[wave.listener]) == (-20.0, 0.0, 1.5):
            kept[stage[0]] = parts[wave.listener][0].fine[wave.carrier]

    monkeypatch.setattr(refracted.Refraction, "_place", keep)
    filtered = band_count(FILTERED_MAX_BAND_HZ, 8)
    for choice, count in ((24, 24), (8, 8), ("rays", 0)):
        kept.clear()
        early = {"method": "image_sources", "max_order": 1, "diffraction": "none"}
        data = field({"preset": "windy"}, ((-20.0, 0.0, 1.5),), late={"rays": 50}).model_dump()
        data["simulation"]["early"] = {**early, "open_ground_frequencies": choice}
        data["sources"][0]["position"] = [-145.0, 0.0, 2.0]
        still = Scene.model_validate(data)
        data["sources"][0]["walk"] = {"to": [[-144.0, 0.0, 2.0]], "speed_m_s": 1.0}
        walking = Scene.model_validate(data)
        for name, run in (
            ("listener", lambda still=still: simulate(Room.from_scene(still), threads=1)),
            (
                "map",
                lambda still=still: coverage_map(
                    Room.from_scene(still), settings=Coverage(spacing=40.0, height=1.5), threads=1
                ),
            ),
            (
                "walk",
                lambda walking=walking: simulate_walk(
                    Room.from_scene(walking), step_m=0.5, threads=1
                ),
            ),
        ):
            runs.clear()
            stage[0] = name
            run()
            # The skirt: past 2 kHz's upper edge, or (with rays) 500 Hz's below 1 kHz.
            above = [f for f in runs if f.min() > 707.0]
            past = [f.min() > 2000.0 * math.sqrt(2.0) or f.max() < 1000.0 for f in above]
            high = [len(f) for f, skirt in zip(above, past, strict=True) if not skirt]
            skirts = [len(f) for f, skirt in zip(above, past, strict=True) if skirt]
            assert len(runs) > len(high) > 0 if count else not high, (choice, name, high)
            assert all(n == count for n in high), (choice, name, high)
            expected = {24: [10], 8: [4], "rays": [4]}[choice]
            assert skirts == (expected if name != "map" or choice == "rays" else []), (
                choice,
                name,
                skirts,
            )
        # The listener's and the map point's (at the same place) in the filtered bands.
        listener, point = kept["listener"][:filtered], kept["map"][:filtered]
        np.testing.assert_array_equal(listener, point, str(choice))


def test_one_coherent_band_keeps_its_run_below_it() -> None:
    """With ``coherent_max_band_hz: 63`` in octave bands, 100 m downwind in the windy
    preset, the 63 Hz band is the only coherent one: it takes no skirt above, and its run
    goes on down to half its lower edge as with more bands (four frequencies, 25 to 42
    Hz), whose samples the carrier keeps. The band's run is the same as with 125, and so
    is its level (+6.2750 dB re free field both, ``coh63.py``; 4e-8 dB apart, the 63 Hz
    shape drawn with and without the 125 Hz band's samples beside it)."""
    from acoustic_spaces.engine.fine import fine_frequencies

    levels = []
    for cap in (125, 63):
        early = {"method": "image_sources", "max_order": 1, "diffraction": "none"}
        scene = field({"preset": "windy"}, early={**early, "coherent_max_band_hz": cap})
        response = simulate(Room.from_scene(scene), threads=1)["s", "l0"]
        levels.append(response.relative_to_free_field()[0])
    fine = response.early.fine
    carrier = int(np.flatnonzero(np.isfinite(fine.real).any(axis=(1, 2)))[0])
    under = fine_frequencies(8, 1)[0] < 44.0
    assert under.sum() == 4 and np.isfinite(fine[carrier, 0, under].real).all()
    assert levels[1] == pytest.approx(levels[0], abs=1e-6)


def scattering_levels(scattering: float, zs: float, xs, atmosphere: dict) -> np.ndarray:
    """(len(xs), bands) the early level (dB) of a source ``zs`` up over grass whose
    scattering coefficient is ``scattering``, set here so that the scene's default does
    not count, at listeners 1.5 m up ``xs`` m along x in ``atmosphere``."""
    from acoustic_spaces.engine.coherent import total_energy

    data = {
        "name": "scattering",
        "geometry": {"ground": {"extent": [[-50.0, -50.0], [250.0, 50.0]], "sky": 50.0}},
        "materials": {"field": {"impedance": "grass", "scattering": scattering}},
        "default_material": "field",
        "sources": [{"name": "s", "position": [0.0, 0.0, zs]}],
        "listeners": [{"name": f"l{k}", "position": [x, 0.0, 1.5]} for k, x in enumerate(xs)],
        "atmosphere": atmosphere,
        "simulation": {
            "seed": 1,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
            "late": {"rays": 50},
            "ir_length_s": 0.1,
        },
    }
    result = simulate(Room.from_scene(Scene.model_validate(data)), threads=1)
    return np.array(
        [10.0 * np.log10(total_energy(result["s", f"l{k}"].early)) for k in range(len(xs))]
    )


@pytest.mark.parametrize(
    "xs",
    [(119.865, 119.8675, 119.870, 119.8725), (162.0, 162.25, 162.5, 162.75)],
    ids=["caustic", "ray event"],
)
def test_the_equation_s_bands_leave_the_ground_s_scattering_and_go_on_across_the_rays(
    xs,
) -> None:
    """Downwind in the ``windy`` preset, the source 2 m up: the parabolic equation's
    bands (63 Hz to 2 kHz) are the same bit for bit over grass that scatters nothing and
    over grass whose scattering coefficient is 0.1, and with 0.1 they go on across the
    rays' caustic between 119.8675 and 119.870 m and where a ray comes in between 162.25
    and 162.5 m as on either side: each step within a quarter more than the larger of its
    neighbours' (at most 1.08 times measured: 0.0431 dB at 250 Hz at the ray event against
    0.0400 dB beside it; at the caustic all steps within 0.0005 dB)."""
    windy = {"preset": "windy", "wind": {"direction_deg": 0.0}}
    smooth = scattering_levels(0.0, 2.0, xs, windy)[:, :6]
    rough = scattering_levels(0.1, 2.0, xs, windy)[:, :6]
    np.testing.assert_array_equal(rough, smooth)
    steps = np.abs(np.diff(rough, axis=0))
    assert np.all(steps[1] <= 1.25 * np.maximum(steps[0], steps[2]) + 0.001), steps


def test_the_equation_s_bands_go_on_at_the_upwind_shadow_s_start() -> None:
    """In 1 m/s of wind upwind, the source 10 m up over grass whose scattering
    coefficient is 0.1, the rays stop arriving between 181.794 and 181.795 m: every
    millimetre from 181.788 to 181.800 m the parabolic equation's bands (63 Hz to 2 kHz)
    move by at most 0.002 dB (0.0016 measured, 250 Hz, the field's own slope), and are
    those over grass that scatters nothing bit for bit."""
    xs = np.round(np.arange(181.788, 181.8005, 0.001), 4)
    upwind = {"wind": {"speed_m_s": 1.0, "direction_deg": 180.0}}
    rough = scattering_levels(0.1, 10.0, xs, upwind)[:, :6]
    np.testing.assert_array_equal(rough, scattering_levels(0.0, 10.0, xs, upwind)[:, :6])
    assert np.abs(np.diff(rough, axis=0)).max() <= 0.002


def test_the_open_ground_s_frequencies_change_nothing_in_still_air() -> None:
    """Without wind or a temperature gradient the setting changes nothing."""
    energies = []
    for choice in (24, 8, "rays"):
        scene = caustic_scene("neutral", 2.0, [(100.0, 0.0, 1.5)], open_ground_frequencies=choice)
        energies.append(simulate(Room.from_scene(scene), threads=1)["s", "l0"].early.energy)
    for energy in energies[1:]:
        np.testing.assert_array_equal(energy, energies[0])


def test_the_parabolic_equation_over_a_ground_that_absorbs_nothing() -> None:
    """A ground absorbing nothing has an admittance of 0 and an infinite impedance: the
    parabolic equation's bands (63 Hz to 2 kHz) over it, 100 m downwind and upwind, come
    within 0.05 dB of those over a ground absorbing 0.001."""
    levels = []
    for absorption in (0.0, 0.001):
        data = field({"preset": "windy"}, ((100.0, 0.0, 1.5), (-100.0, 0.0, 1.5))).model_dump()
        data["materials"]["lawn"] = {"absorption": [absorption] * 8, "scattering": 0.0}
        result = simulate(Room.from_scene(Scene.model_validate(data)))
        energy = [result["s", key].early.energy[:, :6].sum(axis=0) for key in ("l0", "l1")]
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

    # The lawn scatters 0.1: over a ground that scatters nothing there is no late part.
    plain = simulate(Room.from_scene(scattering_lawn(zoned(None))))
    # A zone that restates the weather changes nothing worth the name.
    same = simulate(
        Room.from_scene(
            scattering_lawn(zoned([{"polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 5.0}}]))
        )
    )
    for name in ("l0", "l1"):
        a, b = plain["s", name].energy(), same["s", name].energy()
        np.testing.assert_allclose(10 * np.log10(b.sum(axis=1) / a.sum(axis=1)), 0.0, atol=1e-6)
    # The wind dropping over the downwind half tells downwind; upwind nothing crosses it.
    room = Room.from_scene(
        scattering_lawn(zoned([{"polygon": DOWNWIND_HALF, "wind": {"speed_m_s": 0.5}}]))
    )
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


def test_the_shares_of_three_weathers_add_up_to_one() -> None:
    """Three weathers each under a third of a 300 m way: rounded one by one, the shares
    come to 0.335 each, 1.005 in all, which took the still air's speed 0.5 % too fast
    and brought the direct sound 4.3 ms early. Divided by their sum, they give the
    exact mean profile, each weather weighted by its share of the way."""
    from acoustic_spaces.engine.refracted import MIX_STEP
    from acoustic_spaces.engine.weather import Weather

    def band(x0: float, x1: float) -> list[list[float]]:
        return [[x0, -100.0], [x1, -100.0], [x1, 100.0], [x0, 100.0]]

    scene = zoned(
        [
            {"name": "lee", "polygon": band(100.0, 200.0), "wind": {"speed_m_s": 2.0}},
            {"name": "lake", "polygon": band(200.0, 300.0), "temperature_c": 15.0},
        ],
        listeners=((300.0, 0.0, 1.5), (299.0, 0.0, 1.5)),
    )
    weather = Weather(scene.atmosphere, scene.air.temperature_c, (-400.0, -50.0, 400.0, 50.0))
    a, _ = weather.shares([[0.0, 0.0], [300.0, 0.0]])
    assert np.round(a / MIX_STEP).sum() * MIX_STEP == pytest.approx(1.005)
    # Each weather a third, the winds all blowing along the way.
    exact = Profile(weather.nodes, weather.still.mean(axis=0) + weather.wind.mean(axis=0))
    expected = Rays(exact, 2.0).eigenrays(1.5, 300.0, 0).time[0]
    result = simulate(Room.from_scene(scene))
    assert result["s", "l0"].direct_delay_s == pytest.approx(expected, abs=1e-5)
    # A metre farther, the sound comes about 1 m / c later, not earlier.
    later = result["s", "l0"].direct_delay_s - result["s", "l1"].direct_delay_s
    assert later == pytest.approx(1.0 / exact.c[0], rel=0.05)


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
