"""The fast field program (validation/ffp.py) and the 1995 benchmark script
(validation/benchmark_1995.py): the reference the engine's outdoor sound in wind and
temperature gradients is compared with."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import benchmark_1995  # noqa: E402
import ffp  # noqa: E402
from ground_reflection import exact_reflection, field  # noqa: E402

from acoustic_spaces.engine.parabolic import march  # noqa: E402
from acoustic_spaces.materials.ground import PRESETS, surface_impedance  # noqa: E402


def test_bessel_functions_are_exact() -> None:
    """J0 and Y0 at tabulated points (Abramowitz and Stegun), the series and the
    expansion where they meet, and J0 of a complex argument from both."""
    h = ffp.hankel0(np.array([1.0, 10.0, 20.0]))
    np.testing.assert_allclose(h.real, [0.7651976866, -0.2459357645, 0.1670246643], atol=1e-9)
    np.testing.assert_allclose(h.imag, [0.0882569642, 0.0556711673, 0.0626405968], atol=1e-9)
    z = np.array([17.0 - 2.0j])
    series = ffp._series(z)[0]
    expansion = 0.5 * (ffp._expansion(z) + np.conj(ffp._expansion(np.conj(z))))
    np.testing.assert_allclose(series, expansion, rtol=1e-9)  # where they meet
    assert ffp.check_bessel()["integrals"] < 1e-7


def test_the_fast_field_program_matches_the_exact_ground_solution() -> None:
    """Still air over grass: the field 100 and 300 m from a source 5 m up, 1 m up,
    against the exact solution for a locally reacting ground, at 100 Hz and 1 kHz."""
    c = 343.0
    still = lambda z: np.full(np.shape(z), c)  # noqa: E731
    zs, zr = 5.0, 1.0
    ranges = np.array([100.0, 300.0])
    for f in (100.0, 1000.0):
        k = 2.0 * math.pi * f / c
        z = complex(surface_impedance(np.array(f), PRESETS["grass"][0], speed_of_sound=c))
        got = ffp.pressure(f, still, zs, ranges, zr, np.conj(z))
        level = 20.0 * np.log10(np.abs(got) * np.hypot(ranges, zs - zr))
        exact = [
            20.0 * math.log10(abs(field(exact_reflection(1.0 / z, k, r, zs, zr), k, a, b)))
            for r, a, b in zip(
                ranges, np.hypot(ranges, zs - zr), np.hypot(ranges, zs + zr), strict=True
            )
        ]
        np.testing.assert_allclose(level, exact, atol=0.01)
    # A rigid ground: the image source, and reciprocity with the listener above.
    r1, r2 = np.hypot(ranges, zs - zr), np.hypot(ranges, zs + zr)
    k = 2.0 * math.pi * 100.0 / c
    exact = np.exp(1j * k * r1) / r1 + np.exp(1j * k * r2) / r2
    got = ffp.pressure(100.0, still, zs, ranges, zr, None)
    np.testing.assert_allclose(got, exact, rtol=1e-3)
    swapped = ffp.pressure(100.0, still, zr, ranges, zs, None)
    np.testing.assert_allclose(swapped, got, rtol=1e-9)


def test_the_fast_field_program_and_the_parabolic_equation_agree_in_wind() -> None:
    """Downward refraction of 0.1/s up to 100 m, 100 Hz, over the benchmark's ground:
    the engine's parabolic equation within 0.1 dB of the fast field program at 300 m
    and 1 km, and the layers converged (halving them moves the level under 0.02 dB)."""
    speed = benchmark_1995.speed_of(0.1, 100.0)
    z = complex(benchmark_1995.impedance_of(100.0))
    ranges = np.array([300.0, 1000.0])
    reference = ffp.pressure(100.0, speed, 5.0, ranges, 1.0, z)
    finer = ffp.pressure(100.0, speed, 5.0, ranges, 1.0, z, layer=0.4)
    wave = march([100.0], speed, lambda r: z, 5.0, ranges, np.array([1.0, 1.0]), 150.0)[0]
    level = benchmark_1995.free_field_db
    np.testing.assert_allclose(level(wave, ranges), level(reference, ranges), atol=0.1)
    np.testing.assert_allclose(level(finer, ranges), level(reference, ranges), atol=0.02)


def test_a_listener_at_the_source_height_is_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="height"):
        ffp.pressure(100.0, lambda z: np.full(np.shape(z), 343.0), 2.0, [50.0], 2.0)


def test_the_benchmark_scene_carries_the_case_s_atmosphere() -> None:
    """A case's scene: the ground's speed of sound is 343 m/s, the gradient is the
    wind along the listeners' line, and it stops where the case says."""
    from acoustic_spaces import Room

    scene = benchmark_1995.scene_of(0.1, 100.0, [100.0, 500.0])
    room = Room.from_scene(scene)
    assert abs(room.speed_of_sound() - 343.0) < 1e-9
    towards = np.array([1.0, 0.0])
    speed = scene.atmosphere.sound_speed(
        np.array([0.0, 50.0, 100.0, 400.0]), scene.air.temperature_c, towards
    )
    np.testing.assert_allclose(speed, [343.0, 348.0, 353.0, 353.0], atol=1e-9)
    upwind = benchmark_1995.scene_of(-0.1, 1000.0, [100.0])
    speed = upwind.atmosphere.sound_speed(
        np.array([0.0, 1000.0, 2000.0]), upwind.air.temperature_c, towards
    )
    np.testing.assert_allclose(speed, [343.0, 243.0, 243.0], atol=1e-9)
    assert benchmark_1995.scene_of(0.0, 1000.0, [100.0]).atmosphere is None


def test_the_benchmark_reads_the_program_through_the_engine_s_filters() -> None:
    """The band filters the benchmark reads the program through are the engine's own,
    and each band's slices lie within that band."""
    from acoustic_spaces.materials import OCTAVE_BANDS_HZ
    from acoustic_spaces.parameters import octave_filter_gains

    n, rate = 1 << 12, 8000.0
    f = np.fft.rfftfreq(n, 1.0 / rate)[1:]
    np.testing.assert_allclose(
        benchmark_1995._filter_gains(f), octave_filter_gains(n, rate)[:, 1:] ** 2, atol=1e-12
    )
    frequencies, widths, members = benchmark_1995._frequency_plan(benchmark_1995.BANDS)
    assert len(members) == list(OCTAVE_BANDS_HZ).index(1000) + 1
    for band, own in enumerate(members):
        centre = OCTAVE_BANDS_HZ[band]  # the slices take the nominal centres
        assert np.all(frequencies[own] > centre / math.sqrt(2.0))
        assert np.all(frequencies[own] < centre * math.sqrt(2.0))
        assert np.allclose(widths[own].sum(), centre / math.sqrt(2.0))  # the band's width
    assert len(frequencies[members[-1]]) == benchmark_1995.RAY_BAND_SLICES
    assert frequencies[0] < OCTAVE_BANDS_HZ[0] / math.sqrt(2.0)  # the band below 63 Hz


def test_the_response_has_no_notch_where_the_parabolic_equation_s_bands_meet_the_rays() -> None:
    """Case 2 of the benchmark, 100 m downwind: the early part of the impulse response,
    whose bands up to 500 Hz come from the parabolic equation and above it from the
    rays, follows the fast field program across the join at 707 Hz."""
    from acoustic_spaces import Room
    from acoustic_spaces.engine import coherent, insertion

    gradient, cap = benchmark_1995.CASES["case 2, downward refraction 0.1/s"]
    scene = benchmark_1995.scene_of(gradient, cap, [100.0])
    response = insertion.simulate(Room.from_scene(scene))[("s", "100 m")]
    rate, n = 8000.0, 1 << 13
    signal = coherent.signal(response.early, n, rate)[:, 0]
    spectrum = np.fft.rfft(signal, n)
    f = np.fft.rfftfreq(n, 1.0 / rate)
    # The free field's level through the same synthesis: the direct sound alone.
    direct = response.early[[int(np.argmin(response.early.delay_s))]]
    alone = direct.__class__(
        order=direct.order,
        delay_s=direct.delay_s,
        distance_m=direct.distance_m,
        direction=direct.direction,
        energy=np.asarray(response.free_field)[None],
        spectrum=np.ones_like(direct.spectrum),
    )
    reference = np.fft.rfft(coherent.signal(alone, n, rate)[:, 0], n)
    speed = benchmark_1995.speed_of(gradient, cap)
    for frequency in (600.0, 650.0, 700.0, 750.0, 800.0):
        i = int(np.argmin(np.abs(f - frequency)))
        engine = 20.0 * np.log10(np.abs(spectrum[i]) / np.abs(reference[i]))
        p = ffp.pressure(
            f[i],
            speed,
            benchmark_1995.ZS,
            np.array([100.0]),
            benchmark_1995.ZR,
            complex(benchmark_1995.impedance_of(f[i])),
        )
        exact = 20.0 * np.log10(np.abs(p[0]) * math.hypot(100.0, 4.0))
        assert abs(engine - exact) < 0.5, (frequency, engine, exact)
