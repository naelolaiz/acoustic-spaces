"""A thin rigid barrier on a rigid plane: the strip it makes with its image (engine/strip.py)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import hankel1, jv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import barrier_strip  # noqa: E402

from acoustic_spaces.engine import strip  # noqa: E402


def slope(kappa, end) -> np.ndarray:
    """A unit line source's rate of change across the barrier at its foot, towards
    the source ``end`` (m from the barrier, m above the plane)."""
    r = math.hypot(*end)
    return 0.25j * kappa * hankel1(1, kappa * r) * end[0] / r


def test_a_barrier_far_lower_than_a_wavelength_is_a_plate_in_a_uniform_flow() -> None:
    """Far lower than a wavelength and than the ends' distances, the barrier with its
    image is a plate across a uniform flow U, the rate of change of the source's and
    its image's field at its foot: the field's jump across it is 2 U sqrt(H^2 - z^2),
    and what it sheds is that rate times the listener's times pi H^2, to a
    thousandth; four times as high it sheds sixteen times as much, to a hundredth
    (the next order is the square of k H)."""
    kappa = np.array([0.7 + 0.0j, 2.0 + 0.0j, 1.5 + 0.8j])
    far, near, height = (5.0, 1.0), (7.0, 2.0), 0.01
    shed = strip._rigid(kappa, height, far, near, 12)
    # The source and its image: twice the source's rate, away from it.
    flow = -2.0 * slope(kappa, far)
    plate = flow * slope(kappa, near) * math.pi * height**2
    np.testing.assert_allclose(shed, plate, rtol=1e-3)
    higher = strip._rigid(kappa, 4.0 * height, far, near, 12)
    np.testing.assert_allclose(higher, 16.0 * shed, rtol=1e-2)
    # The reference written apart (validation/barrier_strip.py), by the same plate.
    reference = barrier_strip.Strip(height, 12, 144)
    apart = reference.scattered(kappa, reference.density(kappa, (-far[0], far[1])), near)
    np.testing.assert_allclose(apart, plate, rtol=1e-3)


def test_the_kernel_s_remainder_from_its_series_and_as_it_stands() -> None:
    """The kernel less its part 1 / (2 pi zeta^2), as a smooth part and the factor of
    ln(zeta): from the series of H1 below |kappa zeta| = 0.5, where the two large
    terms would cancel, as it stands above. The two agree to ten digits where both
    hold, and for the smallest distances the series gives
    -kappa^2 / (4 pi) (ln(kappa zeta / 2) + gamma - 1/2) + i kappa^2 / 8, with
    -kappa^2 / (4 pi) the factor of the logarithm."""
    kappa = np.array([2.0 + 0.0j, 1.2 + 0.9j, 0.05 + 3.0j])

    def whole(wavenumbers, zeta) -> np.ndarray:
        smooth, logarithm = strip._remainder(wavenumbers, zeta)
        return smooth + logarithm * np.log(zeta)[None]

    for size in (0.2, 0.35, 0.49, 0.51, 3.0):
        for q in range(len(kappa)):
            zeta = np.array([size / abs(kappa[q])])
            found = whole(kappa[q : q + 1], zeta)[0, 0]
            x = kappa[q] * zeta[0]
            stands = 0.25j * kappa[q] * hankel1(1, x) / zeta[0]
            stands -= 1.0 / (2.0 * math.pi * zeta[0] ** 2)
            assert abs(found - stands) < 1e-10 * abs(stands)
    zeta = np.array([1e-6])
    smooth, logarithm = strip._remainder(kappa, zeta)
    np.testing.assert_allclose(logarithm[:, 0], -(kappa**2) / (4.0 * math.pi), rtol=1e-9)
    limit = -(kappa**2) / (4.0 * math.pi) * (np.log(0.5 * kappa) + 0.5772156649015329 - 0.5)
    np.testing.assert_allclose(smooth[:, 0], limit + 0.125j * kappa**2, rtol=1e-9)
    # Across the switch each part is one function.
    for part in strip._remainder(np.array([1.0 + 0.0j]), np.array([0.5 - 1e-9, 0.5 + 1e-9])):
        assert abs(part[0, 0] - part[0, 1]) < 1e-8 * abs(part[0, 0])


def test_the_logarithm_s_factor_is_faded_where_it_would_grow(monkeypatch) -> None:
    """The factor of ln(zeta) in the kernel's remainder is -kappa J1(kappa zeta) /
    (2 pi zeta) times exp(-(Im(kappa) zeta / 12)^8): the whole of it for a real
    wavenumber, nothing where Im(kappa) zeta is 28 (the fade is below the smallest
    number), and never exp(12) times |kappa| / (2 pi zeta). Whatever the fade, the
    two parts add up to the remainder, from the series and as it stands; where J1 is
    exp(100), and where it is beyond the largest number, the smooth part is the
    remainder, 1 / (2 pi zeta^2) below nothing."""
    zeta = np.array([0.1, 1.0, 3.0, 6.0, 10.0, 13.0, 14.0, 20.0])

    def factor(kappa: complex) -> np.ndarray:
        return -kappa * jv(1, kappa * zeta) / (2.0 * math.pi * zeta)

    real = strip._remainder(np.array([3.0 + 0.0j]), zeta)[1][0]
    np.testing.assert_allclose(real, factor(3.0 + 0.0j), rtol=1e-14)
    kappa = 0.3 + 2.0j
    faded = strip._remainder(np.array([kappa]), zeta)[1][0]
    kept = np.exp(-((2.0 * zeta / 12.0) ** 8))
    np.testing.assert_allclose(faded[:6], (factor(kappa) * kept)[:6], rtol=1e-12)
    assert np.all(faded[6:] == 0.0) and abs(faded[5]) > 0.0
    assert np.all(np.abs(faded) < math.exp(12.0) * abs(kappa) / (2.0 * math.pi * zeta))
    assert abs(factor(kappa)[7]) > math.exp(30.0)
    # From the series (|kappa zeta| = 0.25) and as it stands (2.4): with the fade far
    # off, as it is, and where it takes 0.07 % and 15 % of the factor.
    for kappa, distance in ((0.3 + 0.4j, 0.5), (1.2 + 0.2j, 2.0)):
        stands = 0.25j * kappa * hankel1(1, kappa * distance) / distance
        stands -= 1.0 / (2.0 * math.pi * distance**2)
        factors = []
        for fade_at in (1e9, 12.0, 0.5):
            monkeypatch.setattr(strip, "_FADE_AT", fade_at)
            smooth, logarithm = strip._remainder(np.array([kappa]), np.array([distance]))
            assert abs(smooth[0, 0] + logarithm[0, 0] * math.log(distance) - stands) < 1e-12
            factors.append(logarithm[0, 0])
        monkeypatch.undo()
        assert abs(factors[1] / factors[0] - 1.0) < 1e-9
        assert 5e-4 < abs(factors[2] / factors[0] - 1.0) < 0.2
    # J1 is exp(100), and beyond the largest number.
    smooth, logarithm = strip._remainder(np.array([1.0 + 10.0j, 1.0 + 100.0j]), np.array([10.0]))
    assert np.all(logarithm == 0.0)
    assert np.all(np.abs(smooth + 1.0 / (200.0 * math.pi)) < 1e-40)


def test_far_out_on_the_ray_the_equation_keeps_its_digits(monkeypatch) -> None:
    """Both ends 0.85 m from a barrier 6 m high, 3 and 1 m above the plane, at k = 2:
    the ray reaches Im(kappa) = 7.95, where J1 is 1e40 along the strip and what the
    strip sheds has fallen from 0.028 to 3e-11. Each of the ray's nodes is within a
    thousandth of the reference's (the kernel as it stands, which does not grow; the
    two are 5.2e-4 apart at most, 6.3e-5 on the ray's first half), and the last
    eighteen within a ten-thousandth of Kirchhoff's screen, to which the strip tends
    there. What the barrier sheds beyond the screen keeps twelve digits with 28
    terms more, with five nodes a term, and with the fade at 9 or at 16 (with the
    whole of the logarithm's factor the far nodes are lost to rounding, by amounts
    that differ from one machine to another)."""
    height, near, far, k = 6.0, (0.85, 3.0), (0.85, 1.0), 2.0
    kappa, weights = strip._ray(k, math.hypot(1.7, 2.0))
    assert abs(kappa.imag.max() - 7.95) < 0.01
    assert abs(jv(1, kappa[-1] * 2.0 * height)) > 9e39
    terms = math.ceil(
        np.abs(kappa).max() * height + strip.TERMS_OVER + strip.TERMS_NEAR * height / 0.85
    )
    found = strip._rigid(kappa, height, far, near, terms)
    assert abs(found[0]) > 0.027 and abs(found[-1]) < 4e-11
    reference = barrier_strip.Strip(height, 90, 1080)
    theirs = np.concatenate(
        [
            reference.scattered(part, reference.density(part, (-far[0], far[1])), near, 4000)
            for part in np.split(kappa, 6)
        ]
    )
    np.testing.assert_allclose(found, theirs, rtol=1e-3)
    np.testing.assert_allclose(found[:24], theirs[:24], rtol=1.2e-4)
    screen = strip._screen(kappa, height, far, near)
    np.testing.assert_allclose(found[30:], screen[30:], rtol=1e-4)
    beyond = strip.beyond_screen(np.array([k]), height, near, far)
    np.testing.assert_allclose(beyond[0], np.sum(weights * (found - screen)), rtol=1e-12)
    for name, value in (
        ("TERMS_OVER", 40),
        ("NODES_A_TERM", 5),
        ("_FADE_AT", 9.0),
        ("_FADE_AT", 16.0),
    ):
        monkeypatch.setattr(strip, name, value)
        strip._frame.cache_clear()
        other = strip.beyond_screen(np.array([k]), height, near, far)
        monkeypatch.undo()
        strip._frame.cache_clear()
        np.testing.assert_allclose(other, beyond, rtol=1e-11)


def test_the_rule_for_the_logarithm_is_exact_for_polynomials() -> None:
    """The rule of the density's frame for the root's weight times ln|t - s| times a
    function, from the function's values at the nodes: for 1 it gives
    pi (t^2 / 2 - 1/4 - ln(2) / 2), for s (U_1 / 2) pi/4 (T_3(t) / 3 - T_1(t)), and for
    cos(3 s) what a rule of 20001 nodes gives with the singularity taken off; none of
    the nodes is a collocation point."""
    t, s, w, _, _, apart, log_rule = strip._frame(6)
    assert np.min(apart) > 1e-6

    def of_one(at):  # the logarithm's own integral
        return math.pi * (0.5 * at**2 - 0.25 - 0.5 * math.log(2.0))

    np.testing.assert_allclose(log_rule.sum(axis=1), of_one(t), rtol=1e-12)
    of_s = 0.25 * math.pi * ((4.0 * t**3 - 3.0 * t) / 3.0 - t)
    np.testing.assert_allclose(log_rule @ s, of_s, rtol=1e-12, atol=1e-14)
    fine_s, fine_w = strip._rule(20001)
    for j, at in enumerate(t):
        # The function less its value at t, which leaves no singularity, and that value
        # times the logarithm's own integral.
        rest = (np.cos(3.0 * fine_s) - math.cos(3.0 * at)) * np.log(np.abs(at - fine_s))
        whole = rest @ fine_w + math.cos(3.0 * at) * of_one(at)
        assert abs(log_rule[j] @ np.cos(3.0 * s) - whole) < 1e-8


def test_the_panels_of_a_sum_are_graded_about_an_end_s_foot() -> None:
    """The panels of a sum against an end's field are as wide at the end's foot as the
    end is far from the strip, as wide as they are far from the foot further off, and
    no wider than asked; a foot beyond the range is taken to its end. With them a
    peak a millionth of the range wide, d / (x^2 + d^2), is summed to ten digits (to
    its closed form; the nodes' places are rounded to sixteen), and a polynomial of
    degree 23 exactly."""
    x, w = strip._graded(-2.0, 3.0, 0.5, 1e-3, 0.4)
    assert len(x) % strip.PANEL_NODES == 0 and np.all(np.diff(x) > 0.0)
    widths = w.reshape(-1, strip.PANEL_NODES).sum(axis=1)
    middles = x.reshape(-1, strip.PANEL_NODES).mean(axis=1)
    assert abs(widths.sum() - 5.0) < 1e-12
    assert np.all(widths < 0.4 + 1e-12)
    assert np.all(widths < np.abs(middles - 0.5) + 0.5 * widths + 1e-3 + 1e-12)
    np.testing.assert_allclose(sorted(widths)[:2], 1e-3, rtol=1e-9)
    assert abs(w @ x**23 / ((3.0**24 - 2.0**24) / 24.0) - 1.0) < 1e-12
    for at, d in ((0.5, 5e-6), (3.0, 5e-6), (-2.0, 5e-6), (7.0, 4.0)):
        foot = min(max(at, -2.0), 3.0)
        x, w = strip._graded(-2.0, 3.0, at, math.hypot(d, at - foot), 0.4)
        assert -2.0 < x.min() and x.max() < 3.0 and np.all(w > 0.0)
        exact = math.atan((3.0 - at) / d) + math.atan((at + 2.0) / d)
        assert abs(w @ (d / ((x - at) ** 2 + d**2)) - exact) < 1e-10 * exact
    # Without the grading the peak is lost.
    x, w = strip._graded(-2.0, 3.0, 0.5, 0.4, 0.4)
    assert abs(w @ (5e-6 / ((x - 0.5) ** 2 + 5e-6**2)) - math.pi) > 3.0


def kirchhoff(kappa: complex, height: float, far, near) -> complex:
    """Kirchhoff's integral over the strip for a unit line source at ``far`` and its
    image and one at ``near`` (what ``strip._screen`` sums), by scipy's adaptive
    rule."""

    def line(end, z: float) -> tuple[complex, complex]:
        r = math.hypot(end[0], end[1] - z)
        return 0.25j * hankel1(0, kappa * r), 0.25j * kappa * hankel1(1, kappa * r) * end[0] / r

    def summed(z: float) -> complex:
        u_near, du_near = line(near, z)
        total = 0.0j
        for image in (far[1], -far[1]):
            u_far, du_far = line((far[0], image), z)
            total += -u_near * du_far - u_far * du_near
        return total

    marks = [near[1]] if near[1] < height else None
    limits = {"points": marks, "limit": 2000, "epsabs": 1e-13, "epsrel": 1e-11}
    real = quad(lambda z: summed(z).real, -height, height, **limits)[0]
    imaginary = quad(lambda z: summed(z).imag, -height, height, **limits)[0]
    return complex(real, imaginary)


@pytest.mark.parametrize(
    "height, far, near, within",
    [
        (6.0, (5.0, 1.0), (0.002, 3.0), 3e-4),
        (2.0, (5.0, 1.0), (0.001, 0.0), 4e-6),
        (1.0, (5.0, 0.5), (0.0006, 1.0008), 4e-7),
    ],
)
def test_an_end_within_millimetres_of_the_barrier(height, far, near, within) -> None:
    """An end 2 mm beside a barrier 6 m high, 1 mm from the foot of one 2 m high, and
    1 mm from the top of one 1 m high: its field on the strip changes over that
    distance, and the sums against it are graded about its foot. What the rigid
    strip sheds is the density's sum by the root's rule with as many nodes as
    resolve the distance (40 H over it), to a billionth, and the reference's (every
    term, its own brute-force rule) as closely as with the end decimetres away: the
    two are 2.2e-4, 2.1e-6 and 2.2e-7 apart at most here, and 7.0e-5, 1.4e-6 and
    7.4e-7 with the end half a metre from the first two barriers and 0.5 m from the
    third's top. What Kirchhoff's screen sheds is scipy's adaptive sum, to a
    hundred-millionth."""
    kappa = np.array([0.37 + 0.0j, 3.0 + 0.0j, 2.0 + 1.5j])
    terms = math.ceil(3.0 * height + 12 + 5.0 * height / strip._away(far, height))
    found = strip._rigid(kappa, height, far, near, terms)
    fine = math.ceil(40.0 * height / strip._away(near, height))
    coefficients = strip._density(kappa, height, far, terms)
    s, w = strip._rule(fine)
    jump = coefficients @ strip._even(coefficients.shape[1], s)
    to_near = np.hypot(near[0], near[1] - height * s)
    brute = height * np.sum(jump * strip._slope(kappa, to_near, near[0]) * w[None, :], axis=1)
    np.testing.assert_allclose(found, brute, rtol=1e-9)
    count = int(1.3 * 3.6 * height + 24 + 5.0 * height / far[0])
    reference = barrier_strip.Strip(height, count, 12 * count)
    density = reference.density(kappa, (-far[0], far[1]))
    np.testing.assert_allclose(found, reference.scattered(kappa, density, near, fine), rtol=within)
    # The reference's own rule for the equation does not resolve the end beside the barrier.
    if near[1] < height:
        assert np.all(np.abs(reference.scattered(kappa, density, near) / found - 1.0) > 0.5)
    screen = strip._screen(kappa, height, far, near)
    adaptive = [kirchhoff(complex(one), height, far, near) for one in kappa]
    np.testing.assert_allclose(screen, adaptive, rtol=1e-8)


@pytest.mark.parametrize("far, near", [((0.15, 1.0), (0.05, 0.5)), ((0.3, 0.0), (0.1, 2.3))])
def test_kirchhoff_s_screen_with_both_ends_close_to_the_barrier(far, near) -> None:
    """With both ends within 5 to 30 cm of a barrier 2 m high, one of them beside it
    or 30 cm above its top, at wavenumbers far lower than the ends' distances would
    ask for: the panels are no wider than the farther end's distance from the
    barrier either, and what Kirchhoff's screen sheds is scipy's adaptive sum to a
    hundred-millionth."""
    kappa = np.array([0.37 + 0.0j, 3.0 + 0.0j, 2.0 + 1.5j])
    screen = strip._screen(kappa, 2.0, far, near)
    adaptive = [kirchhoff(complex(one), 2.0, far, near) for one in kappa]
    np.testing.assert_allclose(screen, adaptive, rtol=1e-8)


def test_the_equation_s_rule_has_nodes_to_spare(monkeypatch) -> None:
    """With ``NODES_A_TERM`` nodes of the root's rule for each term of the density,
    what a barrier with k H = 32 sheds from one end 30 m away to the other 20 m away
    (48 terms) is that with twice as many nodes to twelve digits; with a third of
    them it is 5e-5 off."""
    kappa, ends = np.array([8.0 + 0.0j]), ((30.0, 1.5), (20.0, 2.0))
    shed = {}
    for nodes in (strip.NODES_A_TERM, 2 * strip.NODES_A_TERM, strip.NODES_A_TERM // 3):
        monkeypatch.setattr(strip, "NODES_A_TERM", nodes)
        strip._frame.cache_clear()
        shed[nodes] = strip._rigid(kappa, 4.0, *ends, 48)[0]
    monkeypatch.undo()
    strip._frame.cache_clear()
    usual, twice, third = shed.values()
    assert abs(usual / twice - 1.0) < 1e-12
    assert 2e-5 < abs(third / twice - 1.0) < 1e-4


def test_how_far_the_density_s_terms_hold_what_the_barrier_adds(monkeypatch) -> None:
    """With ``TERMS_OVER`` terms more than |kappa| H the density's series is cut where
    its terms still count for an end at the barrier's foot, the more the larger k H.
    Against 44 more: what a barrier 2.4 m high adds at k H = 31 between an end at
    its foot and one 60 m away is 6e-5 of itself off (4e-5 dB in the level), with the
    ends 4 cm and 20 m away 3e-5, and with 20 more 1e-8; what a barrier 1 m high
    adds between such ends is 2e-7 off at k H = 13 and 1e-9 at k H = 8."""

    def off(k: float, height: float, near, far, over: int = strip.TERMS_OVER) -> float:
        adds = []
        for more in (over, 44):
            monkeypatch.setattr(strip, "TERMS_OVER", more)
            adds.append(strip.beyond_screen(np.array([k]), height, near, far)[0])
        return abs(adds[0] / adds[1] - 1.0)

    assert 3e-5 < off(12.951, 2.4, (1e-4, 0.0), (60.0, 30.0)) < 1.2e-4
    assert 1e-5 < off(12.951, 2.4, (0.04, 0.0), (20.0, 1.5)) < 6e-5
    assert off(12.951, 2.4, (1e-4, 0.0), (60.0, 30.0), 20) < 1e-7
    assert off(12.951, 1.0, (0.04, 0.0), (20.0, 1.5)) < 1e-6
    assert off(8.0, 1.0, (0.04, 0.0), (20.0, 1.5)) < 1e-8


def test_what_the_barrier_sheds_has_a_limit_as_an_end_nears_its_face() -> None:
    """As an end nears the face of a barrier 6 m high, halfway up, what the barrier
    sheds beyond the march's screen tends to a limit: it changes by a tenth as much
    from one tenth of the distance to the next, from a centimetre to a micrometre,
    and from there to a nanometre by less than a millionth of the free field."""
    k = np.array([2.0 * math.pi * 20.0 / 343.0, 2.0])
    far = (5.0, 1.0)
    shed = [strip.beyond_screen(k, 6.0, (0.1**power, 3.0), far) * far[0] for power in range(2, 10)]
    steps = np.abs(np.diff(shed, axis=0))  # (7, 2)
    np.testing.assert_allclose(steps[1:4] / steps[:3], 0.1, rtol=0.02)
    assert np.all(steps[3:] < 1e-6)
    assert np.all(steps[0] > 1e-5)


@pytest.mark.parametrize("wavenumber", [0.8, 3.0, 13.0])
def test_a_point_source_s_field_is_the_sum_over_the_wavenumbers_along_the_barrier(
    wavenumber: float,
) -> None:
    """The unit line source's field, summed over the wavenumbers along the barrier on
    the ray exp(-i pi / 4) s, is a point source's, exp(ikr) / r, to a millionth: for
    the shortest path, by which the ray's length is set, and for longer ones."""
    reach = 5.5
    kappa, weights = strip._ray(wavenumber, reach)
    for r in (reach, 1.4 * reach, 4.0 * reach):
        summed = np.sum(weights * 0.25j * hankel1(0, kappa * r))
        exact = np.exp(1j * wavenumber * r) / r
        assert abs(summed - exact) < 1e-6 * abs(exact)
    # A ray set for a longer path does not hold a shorter one.
    kappa, weights = strip._ray(wavenumber, 4.0 * reach)
    short = np.sum(weights * 0.25j * hankel1(0, kappa * 0.5 * reach))
    exact = np.exp(1j * wavenumber * 0.5 * reach) / (0.5 * reach)
    assert abs(short - exact) > 1e-4 * abs(exact)


def test_what_the_barrier_sheds_takes_the_sum_over_the_wavenumbers() -> None:
    """The field in two dimensions times the root that makes a point source's field of
    it far from the ends, sqrt(8 pi k / (i l)) with the length l of the path over the
    barrier's top, is not what the barrier sheds beyond the march's screen for a
    point source: at 44.5 Hz behind a barrier 1 m high it is 13 % off with the
    source 2 m before the barrier and the listener 3.5 m past it, and 3 % with the
    two 10 and 15 m from it."""
    k = 2.0 * math.pi * 44.5 / 343.0
    kappa = np.array([k + 0.0j])
    off = []
    for near, far in (((2.0, 0.5), (3.5, 1.5)), ((10.0, 1.5), (15.0, 1.5))):
        summed = strip.beyond_screen(np.array([k]), 1.0, near, far)[0]
        flat = strip._rigid(kappa, 1.0, far, near, 40) - strip._screen(kappa, 1.0, far, near)
        length = math.hypot(near[0], near[1] - 1.0) + math.hypot(far[0], far[1] - 1.0)
        by_root = flat[0] * np.sqrt(8.0 * math.pi * k / (1j * length))
        off.append(abs(by_root - summed) / abs(summed))
    assert 0.125 < off[0] < 0.14
    assert 0.03 < off[1] < 0.04


@pytest.mark.parametrize("height", [0.1, 0.6, 2.0])
def test_the_sound_is_the_same_with_the_ends_swapped(height: float) -> None:
    """What the rigid strip sheds from one end to the other, solved for the one and
    summed against the other's field, is the same solved for the other; and so is
    what the march's screen sheds."""
    kappa = np.array([0.9 + 0.0j, 4.0 + 0.0j, 3.0 + 1.0j, 11.0 + 0.2j])
    one, other = (2.0, 0.3), (6.0, 1.7)
    terms = 40 + math.ceil(2.0 * 11.0 * height)
    np.testing.assert_allclose(
        strip._rigid(kappa, height, one, other, terms),
        strip._rigid(kappa, height, other, one, terms),
        rtol=2e-6,
    )
    there, back = strip._screen(kappa, height, one, other), strip._screen(kappa, height, other, one)
    np.testing.assert_allclose(there, back, rtol=1e-9)


@pytest.mark.parametrize(
    "height, source, listener",
    [
        (0.1, (-0.5, 0.01), (5.0, 1.51)),
        (0.3, (-3.0, 1.51), (5.0, 1.51)),
        (1.1, (-8.0, 0.51), (5.0, 1.51)),
        (0.6, (-0.2, 0.4), (4.0, 0.0)),
    ],
)
def test_what_the_rigid_barrier_sheds_against_the_reference_written_apart(
    height, source, listener
) -> None:
    """A point source's sound shed by the rigid barrier, from the density's even
    terms, solved for the end farther from the barrier and summed on the ray: within
    a thousandth of the field without the barrier of the reference
    (validation/barrier_strip.py: every term, solved for the source, the kernel as it
    stands, the sum over the real wavenumbers and the decaying ones), at 55, 220 and
    620 Hz."""
    k = np.array([1.0, 4.0, 11.4])
    reach = math.hypot(listener[0] - source[0], listener[1] - source[1])
    ends = sorted([(-source[0], source[1]), listener], key=lambda end: strip._away(end, height))
    found = np.empty(len(k), dtype=complex)
    for index, wavenumber in enumerate(k):
        kappa, weights = strip._ray(float(wavenumber), reach)
        terms = math.ceil(float(np.max(np.abs(kappa))) * height + 12 + 5.0 * height / ends[1][0])
        found[index] = np.sum(weights * strip._rigid(kappa, height, ends[1], ends[0], terms))
    # The reference solves for the source as it is given: no swap of the ends.
    reference = barrier_strip.Strip(height, 48, 600)
    theta, w_theta = np.polynomial.legendre.leggauss(240)
    theta, w_theta = 0.25 * np.pi * (theta + 1.0), 0.25 * np.pi * w_theta
    g, w_g = np.polynomial.legendre.leggauss(80)
    gamma, w_g = 6.0 / reach * (g + 1.0), 6.0 / reach * w_g
    for index, wavenumber in enumerate(k):
        both = np.concatenate([(wavenumber * np.sin(theta)).astype(complex), 1j * gamma])
        weights = np.concatenate(
            [w_theta * wavenumber * np.sin(theta), w_g * gamma / np.sqrt(wavenumber**2 + gamma**2)]
        )
        shed = 4.0 * np.sum(
            weights * reference.scattered(both, reference.density(both, source), listener)
        )
        open_field = 4.0 * np.sum(weights * barrier_strip.free(both, source, listener))
        assert abs(found[index] - shed) < 1e-3 * abs(open_field)


def test_the_reference_s_field_without_the_barrier_is_the_direct_sound_and_the_reflection() -> None:
    """The reference's sum over the wavenumbers gives, without the barrier, the direct
    sound and the plane's reflection, each exp(ikr) / r; and with the ends swapped it
    gives the same sound shed (its own check of the ends' swap, which the levels of
    close sources are computed with)."""
    k = np.array([1.0, 6.0])
    source, listener = (-3.0, 0.51), (5.0, 1.51)
    open_field, shed = barrier_strip.point(0.4, k, source, listener, terms=32)
    direct, image = math.hypot(8.0, 1.0), math.hypot(8.0, 2.02)
    exact = np.exp(1j * k * direct) / direct + np.exp(1j * k * image) / image
    np.testing.assert_allclose(open_field, exact, rtol=1e-5)
    reference = barrier_strip.Strip(0.4, 32, 384)
    kappa = np.array([1.0 + 0.0j, 6.0 + 0.0j, 2.0j])
    there = reference.scattered(kappa, reference.density(kappa, source), listener)
    back = reference.scattered(
        kappa, reference.density(kappa, (-listener[0], listener[1])), (-source[0], source[1])
    )
    np.testing.assert_allclose(there, back, rtol=1e-5)


def test_where_the_strip_s_equation_is_not_solved() -> None:
    """Above ``MOST_KH`` and where the density would need more than ``TERMS_MOST``
    terms there is no result, and the uniform theory's terms stand: the count goes
    with the barrier's height over the distance between the ends and over the
    farther end's distance from the barrier, and is too high with both ends halfway
    up and 0.13 of the height from the barrier, or one at its face and the other
    0.22 of the height away (0.145, and 0.25, are solved), and with one end at a
    tenth of the height and the other at nine tenths, both 0.04 of the height from
    the barrier (0.05 is solved). A barrier no higher than the plane, an end in the
    barrier's plane or below the plane's raise a ValueError."""
    k = np.array([1.0, 2.0])
    assert strip.beyond_screen(k, 0.5 * strip.MOST_KH, (5.0, 1.0), (5.0, 1.0)) is not None
    assert strip.beyond_screen(k, 0.5 * strip.MOST_KH + 0.01, (5.0, 1.0), (5.0, 1.0)) is None
    assert strip.beyond_screen(k, 6.0, (0.5, 0.5), (0.5, 0.5)) is None
    assert strip.beyond_screen(k, 6.0, (0.5, 0.5), (15.0, 0.5)) is not None
    low = np.array([0.05])
    for height in (3.0, 10.0):
        half = 0.5 * height
        for one, other, solved in (
            (0.13, 0.13, False),
            (0.145, 0.145, True),
            (0.001, 0.22, False),
            (0.001, 0.25, True),
        ):
            ends = (one * height, half), (other * height, half)
            assert (strip.beyond_screen(low, height, *ends) is not None) == solved
        for off, solved in ((0.04, False), (0.05, True)):
            ends = (off * height, 0.1 * height), (off * height, 0.9 * height)
            assert (strip.beyond_screen(low, height, *ends) is not None) == solved
    for height, source, listener in (
        (0.0, (5.0, 1.0), (5.0, 1.0)),
        (1.0, (0.0, 1.0), (5.0, 1.0)),
        (1.0, (5.0, 1.0), (-1.0, 1.0)),
        (1.0, (5.0, -0.1), (5.0, 1.0)),
    ):
        with pytest.raises(ValueError, match="between two ends on or above the plane"):
            strip.beyond_screen(k, height, source, listener)


def test_the_density_takes_more_terms_for_an_end_close_to_the_barrier(monkeypatch) -> None:
    """The density is solved for the end farther from the barrier, with
    ``TERMS_OVER`` terms more than |kappa| H at the far end of the ray and
    ``TERMS_NEAR`` times the height over that end's distance: with a source 5 cm
    before a barrier 1.1 m high it is solved for the listener, 5 m past it, and with
    the listener half a metre from the barrier, 0.64 m from its top, with
    5 * 1.1 / 0.64 terms for that."""
    asked: list[tuple] = []
    density = strip._density

    def counted(kappa, height, end, terms):
        asked.append((end, terms))
        return density(kappa, height, end, terms)

    monkeypatch.setattr(strip, "_density", counted)
    k = np.array([2.0])
    strip.beyond_screen(k, 1.1, (5.0, 0.5), (5.0, 1.5))
    strip.beyond_screen(k, 1.1, (0.05, 0.5), (5.0, 1.5))
    strip.beyond_screen(k, 1.1, (0.05, 0.5), (0.5, 1.5))
    places = [((5.0, 0.5), (5.0, 1.5)), ((0.05, 0.5), (5.0, 1.5)), ((0.05, 0.5), (0.5, 1.5))]
    for (end, terms), (source, listener) in zip(asked, places, strict=True):
        assert end == listener
        reach = math.hypot(source[0] + listener[0], listener[1] - source[1])
        furthest = float(np.max(np.abs(strip._ray(2.0, reach)[0])))
        away = math.hypot(listener[0], 1.5 - 1.1)  # from the barrier's top
        assert terms == math.ceil(furthest * 1.1 + 12 + 5.0 * 1.1 / away)
    assert asked[2][1] - asked[1][1] > 7
