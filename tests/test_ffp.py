"""The fast field program (validation/ffp.py) and the 1995 benchmark script
(validation/benchmark_1995.py): the reference the engine's outdoor sound in wind and
temperature gradients is compared with, the shadow zone's residue series among it."""

from __future__ import annotations

import dataclasses
import functools
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ai_zeros, airy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import benchmark_1995  # noqa: E402
import ffp  # noqa: E402
from ground_reflection import exact_reflection, field  # noqa: E402

from acoustic_spaces.engine import refracted  # noqa: E402
from acoustic_spaces.engine.parabolic import march  # noqa: E402
from acoustic_spaces.engine.shadow import mode_roots, residue_pressure, rigid_roots  # noqa: E402
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


def test_the_residue_series_matches_the_fast_field_program() -> None:
    """The creeping waves of upward refraction (engine/shadow.py; Pierce 1981, sec. 9-3):
    the zeros of Ai' against SciPy's, the engine's own first one among them; the roots of
    Ai'(s) = q Ai(s) over grounds of admittance; and the residue series against the fast
    field program in a profile of k**2 linear in height, of radius 10 km, from 5 m up to
    1 m up 400 and 900 m away at 200 Hz, over a rigid ground and the benchmark's
    (Delany and Bazley, 200 kPa s/m²): within 0.1 dB and 1 degree, down to 24 dB under
    the free field."""
    np.testing.assert_allclose(rigid_roots(60), ai_zeros(60)[1], rtol=0, atol=1e-10)
    assert refracted._FIRST_MODE == pytest.approx(rigid_roots(1)[0], rel=0, abs=1e-14)
    q = np.array([0.0, 0.5 - 0.3j, 2.0 + 1.0j, 5.0 - 5.0j])
    s = mode_roots(q, 20)
    ai, aip, _, _ = airy(s)
    assert np.abs(aip / ai - q[:, None]).max() < 1e-10
    assert all(len(np.unique(np.round(roots, 6))) == 20 for roots in s)  # none twice
    radius, f, c0 = 10_000.0, 200.0, benchmark_1995.C0
    speed = lambda z: c0 / np.sqrt(1.0 + 2.0 * np.maximum(z, 0.0) / radius)  # noqa: E731
    zs, zr, ranges = benchmark_1995.ZS, benchmark_1995.ZR, np.array([400.0, 900.0])
    k = np.array([2.0 * math.pi * f / c0])
    level = benchmark_1995.free_field_db
    for z in (None, complex(benchmark_1995.impedance_of(f))):
        reference = ffp.pressure(f, speed, zs, ranges, zr, z, top=600.0)
        beta = np.array([0.0 if z is None else 1.0 / z])
        series = np.array([residue_pressure(k, beta, radius, zs, zr, r)[0] for r in ranges])
        # The series' unit source is exp(ikR) / (4 pi R), the program's exp(ikR) / R.
        series *= 4.0 * math.pi
        np.testing.assert_allclose(level(series, ranges), level(reference, ranges), atol=0.1)
        np.testing.assert_allclose(np.angle(series / reference), 0.0, atol=math.radians(1.0))
    assert level(reference, ranges)[-1] < -20.0  # deep in the shadow over the grass


def test_a_listener_at_the_source_height_is_refused() -> None:
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
    assert scene.simulation.early.parabolic_equation == "narrow_angle"
    wide = benchmark_1995.scene_of(0.1, 1000.0, [100.0], "wide_angle")
    assert wide.simulation.early.parabolic_equation == "wide_angle"


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


#: The level re free field (dB) of :func:`ffp.pressure` in case 2 of the benchmark,
#: 100 m downwind, at the spectrum's bins nearest 2.5, 2.6, 2.7, 3.3 and 3.5 kHz (8 kHz
#: over 8192 samples), kept here as the program takes seconds for each.
CROSSOVER_FFP_100M = {
    2500.0: -2.4647,
    2599.609375: -4.8172,
    2700.1953125: -7.9892,
    3299.8046875: -0.8814,
    3500.0: 1.8103,
}


def _level(response, frequencies, rate: float = 8000.0, n: int = 8000) -> np.ndarray:
    """The early part's coherent level re free field (dB) of ``response`` at the bins of
    ``n`` samples at ``rate`` nearest ``frequencies`` (Hz): its spectrum over the free
    field's through the same synthesis, the direct sound alone."""
    from acoustic_spaces.engine import coherent

    early = response.early
    spectrum = np.fft.rfft(coherent.signal(early, n, rate)[:, 0], n)
    direct = early[[int(np.argmin(early.delay_s))]]
    alone = direct.__class__(
        order=direct.order,
        delay_s=direct.delay_s,
        distance_m=direct.distance_m,
        direction=direct.direction,
        energy=np.asarray(response.free_field)[None],
        spectrum=np.ones_like(direct.spectrum),
    )
    reference = np.fft.rfft(coherent.signal(alone, n, rate)[:, 0], n)
    f = np.fft.rfftfreq(n, 1.0 / rate)
    at = [int(np.argmin(np.abs(f - frequency))) for frequency in frequencies]
    return 20.0 * np.log10(np.abs(spectrum[at]) / np.abs(reference[at]))


@functools.lru_cache(maxsize=8)
def _benchmark(
    case: str,
    ranges: tuple,
    bands: str = "octave",
    equation: str = "narrow_angle",
    open_ground_frequencies: int | str = 24,
):
    """The responses of the benchmark's ``case`` at ``ranges`` (m) by listener name, in
    ``bands`` with the parabolic ``equation`` and ``open_ground_frequencies`` (kept for
    the tests that read them)."""
    from acoustic_spaces import Room, Scene
    from acoustic_spaces.engine import insertion

    gradient, cap = benchmark_1995.CASES[case]
    data = benchmark_1995.scene_of(gradient, cap, list(ranges), equation).model_dump(
        mode="json", exclude_none=True
    )
    data["simulation"]["bands"] = bands
    data["simulation"]["early"]["open_ground_frequencies"] = open_ground_frequencies
    responses = insertion.simulate(Room.from_scene(Scene.model_validate(data)), threads=1)
    return {f"{r:g} m": responses[("s", f"{r:g} m")] for r in ranges}


def test_the_response_follows_the_program_either_side_of_the_crossover_to_the_rays() -> None:
    """Case 2 of the benchmark, 100 m downwind: the early part of the impulse response,
    whose bands up to 2 kHz come from the parabolic equation and above it from the
    rays, follows the fast field program on either side of the crossover between them
    (2378 to 3364 Hz), where the equation's field goes on through the crossover."""
    response = _benchmark("case 2, downward refraction 0.1/s", (100.0,))["100 m"]
    engine = _level(response, list(CROSSOVER_FFP_100M), n=1 << 13)
    for (frequency, exact), level in zip(CROSSOVER_FFP_100M.items(), engine, strict=True):
        assert abs(level - exact) < 0.5, (frequency, level, exact)


#: C50 and C80 (dB) and EDT (s) at 63 Hz of case 2 of the benchmark 100 m downwind, read
#: through the engine's band filters and echogram from a dense run of the parabolic
#: equation: 27 times the early paths' frequencies in every band, carried on below the
#: 63 Hz band down to 11 Hz, where its filter is 53 dB down (``ref_low.py`` in
#: /mnt/project-files/wind-fixes-2026-11-10/crossover/build-scripts; within 0.02 dB and
#: 0.1 ms of 81 times the frequencies and of going on down to 5.5 Hz).
DENSE_63 = {"c50": 36.89, "c80": 52.36, "edt": 0.0643}


def test_the_time_parameters_at_63_hz_follow_a_dense_run_of_the_equation() -> None:
    """Case 2, 100 m away: C50 and C80 at 63 Hz, read through the band filters from the
    equation's samples and on down to an octave below the band, are within 0.1 and
    0.15 dB of a dense run of the equation read the same way (0.004 and 0.07 dB
    measured; the band's own samples held below it were 0.30 and 0.28 dB off), and EDT
    within a step of the echogram (1 ms; 0.6 ms measured)."""
    p = _benchmark("case 2, downward refraction 0.1/s", (100.0,))["100 m"].parameters()
    assert abs(p.c50[0] - DENSE_63["c50"]) < 0.1, p.c50[0]
    assert abs(p.c80[0] - DENSE_63["c80"]) < 0.15, p.c80[0]
    assert abs(p.edt[0] - DENSE_63["edt"]) < 0.001, p.edt[0]


#: The level re free field (dB) of :func:`ffp.pressure` in case 2 of the benchmark, 100
#: and 300 m downwind: in the crossover above the parabolic equation's top band, with
#: the ground's dip in it (2850 Hz 100 m away, 3050 Hz 300 m away), and at the join of
#: its 500 Hz and 1 kHz bands (707 Hz). From two runs of the program, by ``ffp_ref.py``
#: and by ``ffp_pts.py``, which agree within 0.001 dB
#: (/mnt/project-files/wind-fixes-2026-11-10/crossover/ffp_case2.json and
#: critique-results/ffp2*.json).
DIP_FFP = {
    100.0: {
        **{2775.0: -10.8288, 2800.0: -11.7135, 2825.0: -12.4066, 2850.0: -12.7778},
        **{2900.0: -12.2595, 3000.0: -8.6527, 3100.0: -5.3105},
        **{650.0: -3.2980, 680.0: -2.6364, 710.0: -2.0176},
    },
    300.0: {
        **{2800.0: -5.7791, 2850.0: -7.4226, 2900.0: -9.3928, 3000.0: -14.3514},
        **{3100.0: -14.7522, 650.0: -1.9012, 680.0: -1.2866, 710.0: -0.7266},
    },
}


def test_the_ground_s_dip_in_the_top_crossover_follows_the_program() -> None:
    """Case 2 of the benchmark: in the crossover above the parabolic equation's top band,
    where the ground's dip falls, the response follows the fast field program within
    0.3 dB 100 m away and 0.5 dB 300 m away. The 2 kHz band holding its shape past its
    last frequency missed the dip by 1.3 dB at 2850 Hz and 2.0 dB at 3 kHz 100 m away and
    by 5.3 dB at 3 kHz 300 m away; through the equation's field the worst from 2 to
    3.45 kHz is 0.19 dB (100 m) and 0.45 dB at 3100 Hz (300 m, where the rays' band
    takes over; ``repair_cmp.py`` on ``eng.py``, see the bounds' source)."""
    responses = _benchmark("case 2, downward refraction 0.1/s", (100.0, 300.0))
    for distance, bound in ((100.0, 0.3), (300.0, 0.5)):
        exact = {f: level for f, level in DIP_FFP[distance].items() if f >= 2800.0}
        np.testing.assert_allclose(
            _level(responses[f"{distance:g} m"], list(exact)),
            list(exact.values()),
            atol=bound,
            err_msg=str(distance),
        )


def test_the_join_of_the_equation_s_bands_follows_the_program() -> None:
    """Case 2, 100 and 300 m away: at the join of the parabolic equation's 500 Hz and
    1 kHz bands (650 to 710 Hz) the response follows the fast field program within
    0.1 dB (0.07 dB measured), where each band holding its shape into the other's half
    of the crossover was up to 0.43 dB off."""
    responses = _benchmark("case 2, downward refraction 0.1/s", (100.0, 300.0))
    for distance in (100.0, 300.0):
        exact = {f: level for f, level in DIP_FFP[distance].items() if f < 1000.0}
        np.testing.assert_allclose(
            _level(responses[f"{distance:g} m"], list(exact)),
            list(exact.values()),
            atol=0.1,
            err_msg=str(distance),
        )


def test_the_ground_s_dip_in_a_third_octave_run_follows_the_program() -> None:
    """The same in third-octave bands, 100 m away, in the crossover above the equation's
    2.5 kHz band: within 0.3 dB of the fast field program at 2775 to 2850 Hz (2.2 dB off
    while the band held its shape)."""
    third = _benchmark("case 2, downward refraction 0.1/s", (100.0,), "third_octave")["100 m"]
    frequencies = [2775.0, 2800.0, 2825.0, 2850.0]
    np.testing.assert_allclose(
        _level(third, frequencies), [DIP_FFP[100.0][f] for f in frequencies], atol=0.3
    )


def _preset(preset: str, upwind: bool, distance: float):
    """The response ``distance`` (m) down or up the wind of the weather ``preset``, over
    the benchmark's ground, between its presets' heights (2 and 1.5 m)."""
    from acoustic_spaces import Room, Scene
    from acoustic_spaces.engine import insertion

    data = benchmark_1995.scene_of(0.1, 100.0, [distance]).model_dump(
        mode="json", exclude_none=True
    )
    data["atmosphere"] = {"preset": preset, "wind": {"direction_deg": 180.0 if upwind else 0.0}}
    data["air"]["temperature_c"] = benchmark_1995.PRESET_GROUND_C
    data["sources"][0]["position"][2] = benchmark_1995.PRESET_ZS
    for listener in data["listeners"]:
        listener["position"][2] = benchmark_1995.PRESET_ZR
    room = Room.from_scene(Scene.model_validate(data))
    return insertion.simulate(room, threads=1)[("s", f"{distance:g} m")]


def test_in_the_shadow_the_bands_keep_their_own_samples() -> None:
    """In the windy preset 100 m upwind, in the shadow, the path that carries the
    parabolic equation's field keeps no fine samples, and its bands are drawn through
    their own: the bands above the equation's deepen there from the octave below and are
    not continuous with its field (the 4 kHz band 15.5 dB above the equation's own over
    2.8 to 4 kHz, ``level_exp.py`` against ``pe_fine.py``). 100 m downwind, out of the
    shadow, the carrier keeps them."""
    assert _preset("windy", True, 100.0).early.fine is None
    lit = _preset("windy", False, 100.0).early
    assert lit.fine is not None and np.isfinite(lit.fine.real).any()


def test_the_skirt_follows_the_scene_s_frequencies_and_only_a_rendered_response_runs_it(
    monkeypatch,
) -> None:
    """Case 2, 100 m away: one more run of the parabolic equation, through the upper
    crossover of the listener's top band of it, at the spacing the scene's
    ``open_ground_frequencies`` gives (59 Hz with 24 an octave, 177 Hz with 8; with
    ``rays`` above the 500 Hz band, the top one short of the shadow), and with 24 none
    where no response is rendered from the spectra (a coverage map, ``rendered=False``),
    whose band filters do not read the crossover above 2 kHz."""
    from acoustic_spaces import synthesis
    from acoustic_spaces.engine import parabolic, refracted
    from acoustic_spaces.engine.fine import fine_frequencies

    runs: list[int] = []
    march = parabolic.march
    monkeypatch.setattr(
        parabolic, "march", lambda f, *a, **k: runs.append(len(f)) or march(f, *a, **k)
    )
    case = "case 2, downward refraction 0.1/s"
    top = synthesis.crossover_frequencies(8) * 2.0 ** (synthesis.crossover_width(8) / 2)
    # The runs are counted on a response of its own, not one kept by _benchmark.
    counted = _benchmark.__wrapped__(case, (100.0,))
    assert len(runs) == 7  # a run in each of the equation's six bands, and the skirt
    for setting, band, step in ((24, 5, 1414.2 / 24), (8, 5, 1414.2 / 8), ("rays", 3, 353.6 / 8)):
        responses = (
            counted
            if setting == 24
            else _benchmark(case, _RANGES[setting], "octave", "narrow_angle", setting)
        )
        early = responses["100 m"].early
        carrier = int(np.flatnonzero(np.isfinite(early.fine.real).any(axis=(1, 2)))[0])
        layout = fine_frequencies(8, 8)[band]
        at = layout[np.isfinite(early.fine[carrier, band].real)]
        assert np.allclose(np.diff(at), step, rtol=1e-3) and top[band] <= at[-1] < top[band] + step
        assert not np.isfinite(early.fine[carrier, band + 1 :].real).any()
    runs.clear()
    plain = functools.partialmethod(refracted.Refraction.__init__, rendered=False)
    monkeypatch.setattr(refracted.Refraction, "__init__", plain)
    _benchmark.__wrapped__(case, (100.0,))
    assert len(runs) == 6


def test_the_wide_angle_equation_s_field_goes_through_the_crossover_too() -> None:
    """With the wide-angle parabolic equation, case 2 100 m away: the ground's dip in the
    crossover above its top band within 0.3 dB of the fast field program (the band held
    its shape there 2.0 dB off at 3 kHz)."""
    response = _benchmark("case 2, downward refraction 0.1/s", (100.0,), "octave", "wide_angle")
    exact = {f: DIP_FFP[100.0][f] for f in (2850.0, 2900.0, 3000.0)}
    np.testing.assert_allclose(
        _level(response["100 m"], list(exact)), list(exact.values()), atol=0.3
    )


#: The frequencies (Hz) of :data:`DIP_FFP` 100 m away in the top crossover.
_TOP = (2800.0, 2850.0, 2900.0, 3000.0, 3100.0)
#: The listeners' distances (m) taken with ``open_ground_frequencies`` 8 and ``rays``,
#: so that the tests share one response of each (:func:`_benchmark`).
_RANGES = {8: (100.0,), "rays": (100.0, 300.0)}


def test_with_8_an_octave_and_with_rays_the_equation_s_field_follows_the_program_too() -> None:
    """Case 2. With ``open_ground_frequencies: 8``, 100 m away, the join of the 500 Hz
    and 1 kHz bands and the ground's dip in the top crossover follow the fast field
    program within 0.1 and 0.3 dB (0.06 and 0.10 dB measured; the bands' own samples
    give 0.43 at 680 Hz and 2.03 dB at 3 kHz). With ``rays`` the equation's top band is
    the 500 Hz one and its field runs on through the join to the rays' 1 kHz band: 300 m
    away at 710 Hz, where the rays' band weighs most in the join, the response is closer
    to the program than the bands' own samples draw it (0.53 against 0.77 dB low, the
    rest the rays' band's own; ``rays_join.py``)."""
    case = "case 2, downward refraction 0.1/s"
    bound = {650.0: 0.1, 680.0: 0.1, 710.0: 0.1, **dict.fromkeys(_TOP, 0.3)}
    response = _benchmark(case, _RANGES[8], "octave", "narrow_angle", 8)["100 m"]
    error = _level(response, list(bound)) - [DIP_FFP[100.0][f] for f in bound]
    assert np.all(np.abs(error) < list(bound.values())), error
    rays = _benchmark(case, _RANGES["rays"], "octave", "narrow_angle", "rays")["300 m"]
    own = types.SimpleNamespace(
        early=dataclasses.replace(rays.early, fine=None, fine_delay=None),
        free_field=rays.free_field,
    )
    exact = DIP_FFP[300.0][710.0]
    assert abs(_level(rays, [710.0])[0] - exact) < abs(_level(own, [710.0])[0] - exact)
