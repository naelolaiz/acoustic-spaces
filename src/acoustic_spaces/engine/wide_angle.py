"""The ground wave in strong refraction: a wide-angle parabolic equation (split-step Padé).

The Green's function parabolic equation (:mod:`~acoustic_spaces.engine.parabolic`)
marches the field exactly in still air but takes the refraction as a phase shift at each
step, which holds for sound within some 15 degrees of the horizontal and for a speed of
sound that stays close to the ground's. Where a gradient goes on for hundreds of metres
up, sound returns from high up at 20 to 40 degrees and the speed of sound up there is far
from the ground's, and over a few kilometres that phase strays by several wavelengths:
3 to 20 dB at a pure tone where several bent paths interfere. This equation carries both
(Collins 1993, J. Acoust. Soc. Am. 93, 1736; his RAM program).

The field p = psi(r, z) exp(i ka r) / sqrt(r) (time as exp(-i omega t)) goes from range r
to r + dr by

    psi(r + dr) = exp(i ka dr (sqrt(1 + X) - 1)) psi,
    X = (d^2/dz^2 + k(z)^2 - ka^2) / ka^2,

the one-way operator of the Helmholtz equation, exact for every angle in a medium that
does not change along the range. The exponential is a rational function of X (a Padé
approximant of TERMS terms, with its numerator vanishing at X = STABILITY_X so that the
waves that do not propagate decay, as RAM has it), applied as a product of TERMS factors
(1 + a_j X) / (1 + b_j X), each a tridiagonal solve. The heights are finite differences
of fourth order (Numerov's: the second derivative as M^-1 D, M = (1, 10, 1) / 12), so
(1 + b X) psi_new = (1 + a X) psi becomes (M + b L) psi_new = (M + a L) psi with
L = (D + M Q) / ka^2 and Q = k(z)^2 - ka^2, both tridiagonal. The ground's impedance Z
(normalised, time as exp(-i omega t)) holds at the bottom, d psi / dz = -i (k / Z) psi,
through a point below the ground, to fourth order too: the derivative is the centred
difference less a sixth of the second difference, as the reflection factor of a plane
wave on the grid then is the exact one's to (kz dz)^4 / 180. Above ``top`` an absorbing
layer (as the Green's function equation's) takes the sound that goes up.

The field starts at the first step's range from the source exactly as the operator has
it (Collins' self-starter, J. Acoust. Soc. Am. 92, 2069, 1992):

    psi(r0) = sqrt(2 pi / ka) exp(i pi / 4) (1 + X)^(-1/4) exp(i ka r0 (sqrt(1 + X) - 1))
              delta(z - zs),

the delta first smoothed by (1 - X)^-2 and then the rest applied as one rational function,
so the source's field over the ground, to the widest angles, is the operator's own.

The frequencies of one band are marched together, on the grid and in the steps of the
highest, under the absorbing layer of the lowest; the steps run compiled
(:class:`acoustic_spaces._native.RationalSteps`), the frequencies on ``threads`` threads.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from acoustic_spaces.engine.parabolic import LAYER_MIN_M, LAYER_STRENGTH, LAYER_WAVELENGTHS

#: Height steps per wavelength.
STEPS_PER_WAVELENGTH = 20
#: Range step in wavelengths.
RANGE_STEP_WAVELENGTHS = 3.0
#: Factors of the rational approximation of a step.
TERMS = 10
#: Where the approximation's numerator vanishes (X below -1 is sound that does not
#: propagate).
STABILITY_X = -3.0


def march(
    frequencies,
    speed,
    impedance,
    zs: float,
    ranges: np.ndarray,
    heights: np.ndarray,
    top: float,
    screen=None,
    speed_at=None,
    threads: int = 1,
) -> np.ndarray:
    """(F, n) the complex pressure at ``frequencies`` (F,) Hz (time as exp(-i omega t),
    free field exp(ikR) / R) at the points ``ranges`` (n,, m along the ground from the
    source) and ``heights`` (n,, m above it), from a source at height ``zs``, under the
    effective speed of sound ``speed(z)``: as :func:`acoustic_spaces.engine.parabolic.march`,
    with the same ``impedance``, ``top``, ``screen`` and ``speed_at``, on ``threads``
    threads."""
    from acoustic_spaces import _native

    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    heights = np.asarray(heights, dtype=float)
    c0 = float(np.asarray(speed(np.array([0.0])))[0])
    ka = 2.0 * math.pi * f / c0  # (F,)
    n, dz, thickness = _grid(f, c0, top)
    z = np.arange(n) * dz
    absorbing = 1j * LAYER_STRENGTH * np.clip((z - top) / thickness, 0.0, 1.0) ** 2
    step = RANGE_STEP_WAVELENGTHS * c0 / f.max()
    threads = max(int(threads), 1)

    def impedances(at: float) -> np.ndarray:
        value = impedance(at) if impedance is not None else None
        if value is None:
            return np.full(len(f), np.inf, dtype=complex)
        return np.broadcast_to(np.asarray(value, dtype=complex), (len(f),)).copy()

    def steps(weights, operator, scales, alpha: float = 0.0, nu: int = 0):
        # The rational function of X for each frequency, compiled with its factors.
        a = np.empty((len(f), TERMS), dtype=complex)
        b = np.empty((len(f), TERMS), dtype=complex)
        for row in range(len(f)):
            a[row], b[row] = rational(ka[row] * scales, TERMS, alpha, nu)
        return _native.RationalSteps(weights, operator, a, b, threads)

    current = speed
    key = impedances(0.0)
    weights, operator = _operators(f, ka, current(z), absorbing, key, dz)
    order = np.argsort(ranges, kind="stable")
    out = np.zeros((len(f), len(ranges)), dtype=complex)

    # The self-starter: the delta at the source's height, smoothed by (1 - X)^-2, then
    # the rest of the field at r0 as one rational function.
    r = max(min(step, float(ranges[order[0]])), 1e-9) if len(ranges) else step
    psi = np.zeros((len(f), n), dtype=complex)
    at = min(zs / dz, n - 2.0)
    below = int(at)
    weight = np.sqrt(2.0 * math.pi / ka) * np.exp(0.25j * math.pi) / dz
    psi[:, below] = (1.0 - (at - below)) * weight
    psi[:, below + 1] = (at - below) * weight
    smooth = np.zeros((len(f), 2), dtype=complex)
    _native.RationalSteps(weights, operator, smooth, smooth - 1.0, threads).apply(psi, threads)
    steps(weights, operator, r, alpha=-0.25, nu=1).apply(psi, threads)

    made: dict[float, object] = {}
    for index in order:
        goal = ranges[index]
        while goal - r > 1e-9:
            dr = min(step, goal - r)
            value = impedances(r + 0.5 * dr)
            changed = not np.array_equal(value, key)
            if speed_at is not None:
                there = speed_at(r + 0.5 * dr)
                if there is not current:
                    current = there
                    changed = True
            if changed:
                key = value
                weights, operator = _operators(f, ka, current(z), absorbing, key, dz)
                made.clear()
            if dr not in made:
                if len(made) > 1:
                    made.clear()
                made[dr] = steps(weights, operator, dr)
            if screen is not None:
                half = np.exp(0.5j * ka[:, None] * screen(r, dr, z)[None, :])
                psi *= half
            made[dr].apply(psi, threads)
            if screen is not None:
                psi *= half
            r += dr
        p = psi * (np.exp(1j * ka * r) / math.sqrt(r))[:, None]
        h = heights[index]
        out[:, index] = [np.interp(h, z, row.real) + 1j * np.interp(h, z, row.imag) for row in p]
    return out


@functools.lru_cache(maxsize=4096)
def rational(
    sigma: float, terms: int, alpha: float = 0.0, nu: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(a, b), each (terms,), read-only: the product of (1 + a_j X) / (1 + b_j X) that
    approximates (1 - X)^(2 nu) (1 + X)^alpha exp(i sigma (sqrt(1 + X) - 1)) about X = 0
    (a Padé approximant of order ``terms`` over ``terms``), its numerator vanishing at
    :data:`STABILITY_X`. Kept for the session: the same arguments give the same factors
    to the bit, which the linear algebra library alone does not promise (on an Intel
    Mac, two identical marches differed in their last bits), so a march repeated, or on
    other threads, gives the same field."""
    count = 2 * terms
    c = _series(sigma, count, alpha, nu)
    # f(X) times the denominator less the numerator vanishes to the order 2 terms; the
    # unknowns are the numerator's coefficients from X^1 on, then the denominator's.
    system = np.zeros((count, count), dtype=complex)
    right = np.zeros(count, dtype=complex)
    for i in range(1, count + 1):
        right[i - 1] = -c[i]
        if i <= terms:
            system[i - 1, i - 1] = -1.0
        for j in range(1, min(i, terms) + 1):
            system[i - 1, terms + j - 1] = c[i - j]
    system[-1] = 0.0
    system[-1, :terms] = STABILITY_X ** np.arange(1, terms + 1)
    right[-1] = -1.0
    solved = np.linalg.solve(system, right)
    numerator = np.concatenate([[1.0], solved[:terms]])
    denominator = np.concatenate([[1.0], solved[terms:]])
    a, b = -1.0 / np.roots(numerator[::-1]), -1.0 / np.roots(denominator[::-1])
    a.setflags(write=False)
    b.setflags(write=False)
    return a, b


def _series(sigma: float, order: int, alpha: float, nu: int) -> np.ndarray:
    """The Taylor coefficients, X^0 to X^order, of (1 - X)^(2 nu) (1 + X)^alpha
    exp(i sigma (sqrt(1 + X) - 1))."""
    exponent = np.zeros(order + 1, dtype=complex)  # i sigma (sqrt(1 + X) - 1)
    binomial = 1.0
    for m in range(1, order + 1):
        binomial *= (0.5 - (m - 1)) / m
        exponent[m] = 1j * sigma * binomial
    out = np.zeros(order + 1, dtype=complex)
    out[0] = 1.0
    for m in range(1, order + 1):  # g' = h' g, term by term
        out[m] = sum(k * exponent[k] * out[m - k] for k in range(1, m + 1)) / m
    if alpha:
        power = np.ones(order + 1)
        for m in range(1, order + 1):
            power[m] = power[m - 1] * (alpha - (m - 1)) / m
        out = np.convolve(out, power)[: order + 1]
    for _ in range(2 * nu):
        out[1:] = out[1:] - out[:-1]
    return out


def _operators(f, ka, speed, absorbing, zg, dz) -> tuple[np.ndarray, np.ndarray]:
    """(F, 3, n) each, the diagonals (lower, centre, upper) of M and L at the heights,
    for the effective speed of sound ``speed`` (n,) and the ground's impedances ``zg``
    (F,; infinite: rigid)."""
    k = 2.0 * math.pi * f[:, None] / np.asarray(speed, dtype=float)[None, :]
    q = (k + absorbing[None, :]) ** 2 - ka[:, None] ** 2  # (F, n)
    rows, n = q.shape
    weights = np.zeros((rows, 3, n), dtype=complex)
    operator = np.zeros((rows, 3, n), dtype=complex)
    weights[:, 0, 1:] = 1.0 / 12.0
    weights[:, 1] = 10.0 / 12.0
    weights[:, 2, :-1] = 1.0 / 12.0
    operator[:, 0, 1:] = 1.0 / dz**2 + q[:, :-1] / 12.0
    operator[:, 1] = -2.0 / dz**2 + 10.0 * q / 12.0
    operator[:, 2, :-1] = 1.0 / dz**2 + q[:, 1:] / 12.0
    # The ground: the point below it is psi(-dz) = up psi(dz) + here psi(0), from
    # psi(-dz) - psi(dz) = 2 i g (psi(0) + (psi(-dz) - 2 psi(0) + psi(dz)) / 6), g = k dz / Z,
    # and Q there continues Q's slope.
    rigid = ~np.isfinite(zg)
    g = np.where(rigid, 0.0, 1j * k[:, 0] * dz / np.where(rigid, 1.0, zg))
    up = (1.0 + g / 3.0) / (1.0 - g / 3.0)
    here = (4.0 * g / 3.0) / (1.0 - g / 3.0)
    q_below = 2.0 * q[:, 0] - q[:, 1]
    weights[:, 1, 0] = (10.0 + here) / 12.0
    weights[:, 2, 0] = (1.0 + up) / 12.0
    operator[:, 1, 0] = (here - 2.0) / dz**2 + (10.0 * q[:, 0] + q_below * here) / 12.0
    operator[:, 2, 0] = (1.0 + up) / dz**2 + (q[:, 1] + q_below * up) / 12.0
    operator /= (ka**2)[:, None, None]
    return weights, operator


def _grid(f: np.ndarray, c0: float, top: float) -> tuple[int, float, float]:
    """The heights' count and spacing, and the absorbing layer's thickness, for the
    frequencies ``f`` under an absorbing layer from ``top``."""
    dz = c0 / f.max() / STEPS_PER_WAVELENGTH
    thickness = max(LAYER_WAVELENGTHS * c0 / f.min(), LAYER_MIN_M)
    return int(math.ceil((top + thickness) / dz)) + 1, dz, thickness


def footprint(frequencies, c0: float, top: float) -> int:
    """About the memory (bytes) :func:`march` takes for ``frequencies`` under an
    absorbing layer from ``top`` in air of speed ``c0`` on the ground."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n = _grid(f, c0, top)[0]
    # Two steps' factors and diagonals, the operators and the field.
    return 16 * len(f) * n * (2 * (TERMS + 6) + 8)


__all__ = ["footprint", "march", "rational"]
