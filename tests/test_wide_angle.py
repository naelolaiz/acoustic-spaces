"""The wide-angle parabolic equation outdoors (acoustic_spaces.engine.wide_angle)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from test_ground import exact_reflection

from acoustic_spaces import Room, Scene, _native, simulate
from acoustic_spaces.engine import parabolic, wide_angle
from acoustic_spaces.engine.wide_angle import (
    RANGE_STEP_WAVELENGTHS,
    STABILITY_X,
    TERMS,
    march,
    rational,
)
from acoustic_spaces.materials.ground import surface_impedance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import benchmark_1995  # noqa: E402
import ffp  # noqa: E402

C = 343.2


def still(h):
    return np.full(np.shape(h), C)


def product(a, b, x):
    out = np.ones_like(x, dtype=complex)
    for aj, bj in zip(a, b, strict=True):
        out *= (1.0 + aj * x) / (1.0 + bj * x)
    return out


# -- the rational approximation ---------------------------------------------------------


def test_a_step_follows_the_exponential_to_wide_angles() -> None:
    """A step of three wavelengths: within 1e-7 of exp(i sigma (sqrt(1 + X) - 1)) for
    sound up to 45 degrees from the horizontal (X = -0.5) and for air up to 30 % slower
    than at the ground (X up to 0.5), within 2e-5 for air 40 % slower (X up to 1), and
    within 1e-13 near the horizontal."""
    assert RANGE_STEP_WAVELENGTHS == 3.0
    sigma = 2.0 * math.pi * RANGE_STEP_WAVELENGTHS
    a, b = rational(sigma, TERMS)
    assert a.shape == b.shape == (TERMS,)
    x = np.linspace(-0.5, 1.0, 3001)
    exact = np.exp(1j * sigma * (np.sqrt(1.0 + x) - 1.0))
    error = np.abs(product(a, b, x) - exact)
    assert error.max() < 2e-5
    assert error[np.abs(x) < 0.1].max() < 1e-13
    assert error[(x >= -0.5) & (x <= 0.5)].max() < 1e-7


def test_a_step_damps_what_does_not_propagate() -> None:
    """Below X = -1 the waves decay with height and do not propagate: the step never
    makes them, nor anything the absorbing layer (Im X > 0) holds, grow, and it takes
    out the waves at the stability point altogether."""
    for wavelengths in (0.5, RANGE_STEP_WAVELENGTHS):
        a, b = rational(2.0 * math.pi * wavelengths, TERMS)
        x = np.linspace(-60.0, 1.5, 2001)[None, :] + 1j * np.linspace(0.0, 3.0, 61)[:, None]
        assert np.abs(product(a, b, x)).max() <= 1.0 + 1e-9
        assert abs(product(a, b, np.array([STABILITY_X]))[0]) < 1e-9


def test_the_starter_s_function() -> None:
    """The self-starter's (1 - X)^2 (1 + X)^(-1/4) exp(i sigma (sqrt(1 + X) - 1))."""
    sigma = 2.0 * math.pi * 1.5
    a, b = rational(sigma, TERMS, alpha=-0.25, nu=1)
    x = np.linspace(-0.5, 0.5, 1001)
    exact = (1.0 - x) ** 2 * (1.0 + x) ** -0.25 * np.exp(1j * sigma * (np.sqrt(1.0 + x) - 1.0))
    assert np.abs(product(a, b, x) - exact).max() < 1e-8


# -- the compiled steps -------------------------------------------------------------------


def _system(rows: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    def complex_normal(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    weights = complex_normal(rows, 3, n) * 0.1
    weights[:, 1] += 2.0
    operator = complex_normal(rows, 3, n)
    weights[:, 0, 0] = weights[:, 2, -1] = operator[:, 0, 0] = operator[:, 2, -1] = 0.0
    return weights, operator, complex_normal(rows, 3) * 0.2, complex_normal(rows, 3) * 0.2


def _dense(diagonals: np.ndarray) -> np.ndarray:
    return np.diag(diagonals[1]) + np.diag(diagonals[0][1:], -1) + np.diag(diagonals[2][:-1], 1)


def test_the_steps_solve_their_tridiagonal_systems() -> None:
    m, op, a, b = _system(2, 9)
    psi = np.random.default_rng(1).normal(size=(2, 9)) + 0j
    expected = psi.copy()
    for row in range(2):
        for j in range(3):
            left = _dense(m[row]) + b[row, j] * _dense(op[row])
            right = _dense(m[row]) + a[row, j] * _dense(op[row])
            expected[row] = np.linalg.solve(left, right @ expected[row])
    steps = _native.RationalSteps(m, op, a, b)
    assert (steps.rows, steps.points, steps.terms) == (2, 9, 3)
    one = psi.copy()
    steps.apply(one)
    np.testing.assert_allclose(one, expected, rtol=1e-12, atol=1e-12)
    # The rows are independent: threads change nothing, bit for bit.
    many = psi.copy()
    _native.RationalSteps(m, op, a, b, threads=2).apply(many, threads=2)
    assert np.array_equal(one, many)


def test_the_steps_check_what_they_are_given() -> None:
    m, op, a, b = _system(2, 5)
    with pytest.raises(ValueError, match="rows, 3, points"):
        _native.RationalSteps(m, op[:, :, :4].copy(), a, b)
    with pytest.raises(ValueError, match="rows, terms"):
        _native.RationalSteps(m, op, a, b[:, :2].copy())
    steps = _native.RationalSteps(m, op, a, b)
    with pytest.raises(ValueError, match="rows, points"):
        steps.apply(np.zeros((2, 4), dtype=complex))
    with pytest.raises(TypeError):
        steps.apply(np.zeros((2, 5)))  # real: not stepped in place
    singular = np.zeros_like(m)
    with pytest.raises(RuntimeError, match="singular"):
        _native.RationalSteps(singular, singular, a, b)


# -- against exact solutions --------------------------------------------------------------


@pytest.mark.parametrize("f", [125.0, 500.0])
def test_still_air_against_the_exact_solution(f) -> None:
    """Over grass, from 1 m up to 1.5 m up, 30 and 100 m away: the complex pressure
    within 0.5 % (0.04 dB, 0.3 degrees)."""
    zs, zr, ranges = 1.0, 1.5, np.array([30.0, 100.0])
    k = 2 * math.pi * f / C
    z = complex(surface_impedance(np.array(f), 200.0))
    pressure = march([f], still, lambda r: np.conj(z), zs, ranges, np.full(2, zr), 25.0)[0]
    for i, r in enumerate(ranges):
        r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
        q = exact_reflection(1.0 / z, k, r, zs, zr)
        exact = (np.exp(-1j * k * r1) / r1 + q * np.exp(-1j * k * r2) / r2) * r1
        # Time as exp(-i omega t) here, as exp(j omega t) in the exact solution.
        assert abs(pressure[i] * r1 / np.conj(exact) - 1.0) < 5e-3


@pytest.mark.parametrize("f", [125.0, 500.0])
def test_it_starts_wide(f) -> None:
    """The self-starter: the direct sound 45 degrees up and its reflection off a rigid
    ground 50 degrees up, within 0.15 dB (Greene's starter, the narrow-angle equation's,
    is 1 dB short there)."""
    r = 40.0
    height = 5.0 + r * math.tan(math.radians(45.0))
    k = 2 * math.pi * f / C
    r1, r2 = math.hypot(r, height - 5.0), math.hypot(r, height + 5.0)
    exact = abs(np.exp(1j * k * r1) / r1 + np.exp(1j * k * r2) / r2)
    levels = [
        20 * math.log10(abs(solver([f], still, None, 5.0, [r], [height], 80.0)[0, 0]) / exact)
        for solver in (march, parabolic.march)
    ]
    assert levels[0] == pytest.approx(0.0, abs=0.15)
    assert abs(levels[1]) > 0.6


def test_a_deep_gradient_against_the_fast_field_program() -> None:
    """The 1995 benchmark's case 2 (the speed of sound rising 0.1 m/s per metre, here to
    1000 m up) at 100 Hz: within 0.05 dB of the fast field program to 5 km, where the
    narrow-angle equation, which does not carry the sound coming back down from hundreds
    of metres up, is 6 dB off."""
    f, ranges = 100.0, np.array([1000.0, 2000.0, 5000.0])
    speed = benchmark_1995.speed_of(0.1, 1000.0)
    z = complex(benchmark_1995.impedance_of(f))
    heights = np.full(len(ranges), benchmark_1995.ZR)
    zs = benchmark_1995.ZS
    reference = ffp.pressure(f, speed, zs, ranges, benchmark_1995.ZR, z)
    wide = march([f], speed, lambda r: z, zs, ranges, heights, 1200.0)[0]
    narrow = parabolic.march([f], speed, lambda r: z, zs, ranges, heights, 1200.0)[0]
    np.testing.assert_allclose(20 * np.log10(np.abs(wide / reference)), 0.0, atol=0.05)
    assert abs(20 * math.log10(abs(narrow[-1] / reference[-1]))) > 3.0


# -- along the way ------------------------------------------------------------------------


def test_threads_and_a_still_screen_change_nothing() -> None:
    f = [100.0, 125.0, 160.0]
    z = np.conj(surface_impedance(np.array(f), 200.0))
    args = (f, still, lambda r: z, 2.0, np.array([60.0, 150.0]), np.full(2, 1.5), 30.0)
    one = march(*args)
    assert np.array_equal(one, march(*args, threads=3))
    calm = march(*args, screen=lambda r, dr, heights: np.zeros_like(heights))
    assert np.array_equal(one, calm)


def test_the_weather_and_the_ground_along_the_way() -> None:
    """The same profile handed on again changes nothing; a ground that turns rigid 100 m
    out changes the sound only beyond, as much as the narrow-angle equation has it."""
    f, ranges, heights = [125.0], np.array([80.0, 200.0, 300.0]), np.full(3, 1.5)
    grass = np.conj(surface_impedance(np.array(f), 200.0))

    def windy(h):
        return C + 0.05 * np.minimum(np.asarray(h, dtype=float), 100.0)

    def again(r):
        return lambda h: windy(h)  # a new object every time: rebuilt, and the same

    plain = march(f, windy, lambda r: grass, 2.0, ranges, heights, 60.0)
    rebuilt = march(f, windy, lambda r: grass, 2.0, ranges, heights, 60.0, speed_at=again)
    np.testing.assert_allclose(rebuilt, plain, rtol=1e-12)

    def mixed(r):
        return grass if r < 100.0 else None

    wide = march(f, windy, mixed, 2.0, ranges, heights, 60.0)[0]
    narrow = parabolic.march(f, windy, mixed, 2.0, ranges, heights, 60.0)[0]
    assert wide[0] == plain[0, 0]
    assert abs(20 * math.log10(abs(wide[2] / plain[0, 2]))) > 1.0
    np.testing.assert_allclose(20 * np.log10(np.abs(wide / narrow)), 0.0, atol=0.3)


# -- in a scene ---------------------------------------------------------------------------


def field(early: dict) -> Scene:
    """A source 2 m up over a lawn, a listener 150 m downwind in a strong wind."""
    return Scene.model_validate(
        {
            "name": "field",
            "geometry": {"ground": {"extent": [[-20.0, -20.0], [170.0, 20.0]], "sky": 20.0}},
            "materials": {"lawn": {"impedance": "grass"}},
            "assignments": {"ground": "lawn"},
            "sources": [{"name": "s", "position": [0.0, 0.0, 2.0]}],
            "listeners": [{"name": "l", "position": [150.0, 0.0, 1.5]}],
            "atmosphere": {"preset": "windy", "wind": {"direction_deg": 0.0}},
            "simulation": {
                "seed": 1,
                "early": {"method": "image_sources", "max_order": 1, **early},
                "late": {"rays": 200},
                "ir_length_s": 0.1,
            },
        }
    )


def test_the_scene_chooses_the_parabolic_equation(monkeypatch) -> None:
    """``simulation.early.parabolic_equation``: narrow_angle by default, wide_angle on
    demand; in the gradients of the weather near the ground the two agree."""
    assert field({}).simulation.early.parabolic_equation == "narrow_angle"
    calls = {"narrow": 0, "wide": 0}
    for name, module in (("narrow", parabolic), ("wide", wide_angle)):

        def counted(*args, _march=module.march, _name=name, **kwargs):
            calls[_name] += 1
            return _march(*args, **kwargs)

        monkeypatch.setattr(module, "march", counted)
    narrow = simulate(Room.from_scene(field({})))
    assert calls == {"narrow": 4, "wide": 0}  # the bands up to 500 Hz
    wide = simulate(Room.from_scene(field({"parabolic_equation": "wide_angle"})))
    assert calls == {"narrow": 4, "wide": 4}
    a = narrow["s", "l"].relative_to_free_field()[:4]
    b = wide["s", "l"].relative_to_free_field()[:4]
    np.testing.assert_allclose(b, a, atol=0.3)
    assert not np.array_equal(a, b)
