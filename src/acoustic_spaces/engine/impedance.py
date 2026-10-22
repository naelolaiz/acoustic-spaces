"""Locally reacting surfaces for the wave solver: frequency-dependent admittances fitted
to a material's absorption coefficients.

A material table gives each band's random-incidence absorption. The wave solver needs
the surface's specific admittance (the normal velocity times rho c over the pressure),
as a function of frequency it can step in time. :class:`Admittance` is a constant
conductance beta plus parallel mass-resistance-stiffness branches,

    Y(s) = beta + sum_j 1 / (M_j s + R_j + S_j / s),

each resonant (at f_j, with quality factor Q_j) and passive, so the scheme stays stable
whatever the values. :func:`fit_admittance` chooses beta and the branches so that the
random-incidence absorption of Y (Paris's formula for a locally reacting surface),
counted through the analysis's band filters, matches the material's in every band the
wave solver covers: one branch at each band's centre and one beyond each end. It fits
the admittance the scheme actually has at its time step, which bends a little from Y
towards the grid's Nyquist frequency.

A locally reacting surface absorbs at most :data:`MAX_ABSORPTION` (0.951) at random
incidence; higher coefficients get that. The band filters overlap, so bands whose
coefficients change steeply from one to the next (0.1 to 0.6 in an octave) are matched
only roughly; :attr:`Admittance.achieved` holds what the fit reached.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Highest random-incidence absorption a locally reacting surface reaches (Paris's
#: formula), and the normalised impedance that reaches it.
MAX_ABSORPTION = 0.951
_BEST_IMPEDANCE = 1.5672
#: Frequencies per octave the fit evaluates the absorption at.
_PER_OCTAVE = 32
#: The fit frees each branch's quality factor when fixed ones miss by more than this.
_TOLERANCE = 0.005
#: Bounds of the quality factors, and how much it costs (in absorption) to move one by
#: a factor of e from the bands' own.
_QUALITY = (0.5, 10.0)
_QUALITY_COST = 0.02


def absorption_of_impedance(impedance: np.ndarray) -> np.ndarray:
    """Random-incidence absorption of a locally reacting surface with a real normalised
    impedance (Paris's formula, for a diffuse field)."""
    z = np.asarray(impedance, dtype=float)
    return 8.0 / z * (1.0 + 1.0 / (1.0 + z) - 2.0 / z * np.log1p(z))


def random_incidence_absorption(admittance: np.ndarray) -> np.ndarray:
    """Random-incidence absorption of a locally reacting surface with the complex
    specific admittance ``admittance``: Paris's formula, the integral over incidence of
    ``1 - |R|^2`` weighted by ``sin(2 theta)``, in closed form."""
    y = np.asarray(admittance, dtype=complex)
    g, b = y.real, y.imag
    d = g + g * g + b * b
    safe = np.where(d > 0.0, d, 1.0)
    x = b / safe
    # atan(b / d) / b without dividing by zero for a real admittance.
    small = np.abs(x) < 1e-4
    ratio = np.where(small, 1.0 - x * x / 3.0, np.arctan(x) / np.where(small, 1.0, x)) / safe
    magnitude = g * g + b * b
    log = np.log(((1.0 + g) ** 2 + b * b) / np.where(magnitude > 0.0, magnitude, 1.0))
    alpha = 8.0 * g * (1.0 - g * log + (g * g - b * b) * ratio)
    return np.where(g > 0.0, alpha, 0.0)


def admittance(absorption: np.ndarray) -> np.ndarray:
    """Real specific (normalised) admittance of a locally reacting surface whose
    random-incidence absorption is ``absorption``: the inverse of Paris's formula, on
    the side of hard surfaces. Coefficients from :data:`MAX_ABSORPTION` up get the most
    absorbing admittance."""
    alpha = np.clip(np.asarray(absorption, dtype=float), 0.0, 1.0)
    low = np.full(alpha.shape, math.log(_BEST_IMPEDANCE))
    high = np.full(alpha.shape, math.log(1e9))
    for _ in range(60):  # bisection in log(impedance); absorption falls as it grows
        mid = 0.5 * (low + high)
        above = absorption_of_impedance(np.exp(mid)) > alpha
        low = np.where(above, mid, low)
        high = np.where(above, high, mid)
    beta = np.exp(-0.5 * (low + high))
    return np.where(alpha <= 0.0, 0.0, beta)


def _branch_values(
    f: np.ndarray, conductance, quality, frequency, rate: float | None
) -> np.ndarray:
    """The branches' admittances (branches, len(f)): exact, or with ``rate`` the
    scheme's at that time step."""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)[None, :]
    gamma = np.asarray(conductance, float)[:, None]
    q = np.asarray(quality, float)[:, None]
    w0 = 2.0 * np.pi * np.asarray(frequency, float)[:, None]
    if rate is None:
        return gamma / (1.0 + 1j * q * (w / w0 - w0 / w))
    # m (z - 2 + 1/z) W + r (z - 1/z) W / 2 + q (z + 1/z) W / 2 = (z - 1/z) P / 2, with
    # M = Q R / w0, S = Q R w0 and R = 1 / conductance; divided through by R.
    t = 1.0 / rate
    s, c = np.sin(w * t), np.cos(w * t)
    half = np.sin(0.5 * w * t) ** 2
    return 1j * s * gamma / (-4.0 * q / (w0 * t) * half + 1j * s + q * w0 * t * c)


def _relaxation_values(f: np.ndarray, conductance, corner, rate: float | None) -> np.ndarray:
    """The admittances (branches, len(f)) of branches without mass, 1 / (R + S / s),
    of conductance 1 / R and corner frequency S / (2 pi R): exact, or with ``rate`` the
    scheme's at that time step."""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)[None, :]
    gamma = np.asarray(conductance, float)[:, None]
    wc = 2.0 * np.pi * np.asarray(corner, float)[:, None]
    if rate is None:
        return gamma * 1j * w / (1j * w + wc)
    # r (z - 1/z) W / 2 + q (z + 1/z) W / 2 = (z - 1/z) P / 2, with q = S T.
    t = 1.0 / rate
    s, c = np.sin(w * t), np.cos(w * t)
    return gamma * 1j * s / (1j * s + wc * t * c)


@dataclass(frozen=True)
class Admittance:
    """A locally reacting surface's specific admittance: ``beta`` plus one resonant
    branch per entry of ``conductance`` (its admittance at resonance, 1 / R), ``quality``
    and ``frequency_hz``, and one branch without mass per entry of ``relaxation`` (its
    admittance at high frequencies, 1 / R) and ``relaxation_hz`` (its corner frequency,
    S / (2 pi R)), as a porous ground's admittance needs."""

    beta: float
    conductance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    quality: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frequency_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: The band absorption the fit reached, or None when not fitted.
    achieved: np.ndarray | None = None
    relaxation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    relaxation_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __call__(self, f: np.ndarray, rate: float | None = None) -> np.ndarray:
        """The admittance at frequencies ``f`` (Hz): exact, or with ``rate`` the one the
        scheme has at that many steps a second."""
        f = np.asarray(f, dtype=float)
        branches = _branch_values(
            f.ravel(), self.conductance, self.quality, self.frequency_hz, rate
        )
        out = self.beta + branches.sum(axis=0)
        if len(self.relaxation):
            out = out + _relaxation_values(
                f.ravel(), self.relaxation, self.relaxation_hz, rate
            ).sum(axis=0)
        return out.reshape(f.shape)

    def branches(self, rate: float) -> np.ndarray:
        """The branches as the wave solver takes them (branches, 3): mass over the time
        step, resistance, and stiffness times the time step."""
        t = 1.0 / rate
        r = 1.0 / np.asarray(self.conductance, float)
        w0 = 2.0 * np.pi * np.asarray(self.frequency_hz, float)
        q = np.asarray(self.quality, float)
        out = np.column_stack([q * r / (w0 * t), r, q * r * w0 * t]).reshape(-1, 3)
        if len(self.relaxation):
            r = 1.0 / np.asarray(self.relaxation, float)
            wc = 2.0 * np.pi * np.asarray(self.relaxation_hz, float)
            out = np.vstack([out, np.column_stack([np.zeros(len(r)), r, r * wc * t])])
        return out


def band_weights(f: np.ndarray, bands: int, total: int) -> np.ndarray:
    """How much each of the frequencies ``f`` (on a log grid) counts in each of the
    lowest ``bands`` of the ``total`` (8 octave or 24 third-octave) bands (bands,
    len(f)): the energy of white noise through the analysis's band filters."""
    from acoustic_spaces.materials.bands import BANDS_PER_OCTAVE, exact_centres, resolution_of
    from acoustic_spaces.parameters import FILTER_ORDER, OCTAVE_RATIO

    resolution = resolution_of(total)
    fm = exact_centres(resolution)[:bands, None]
    half = OCTAVE_RATIO ** (0.5 / BANDS_PER_OCTAVE[resolution])
    width = half - 1.0 / half
    gains = f / (1.0 + ((f / fm - fm / f) / width) ** (2 * FILTER_ORDER))  # |H|^2 df
    return gains / gains.sum(axis=1, keepdims=True)


def _frequencies(bands: int, total: int) -> np.ndarray:
    from acoustic_spaces.materials.bands import exact_centres, resolution_of

    fm = exact_centres(resolution_of(total))[:bands]
    lo, hi = fm[0] / 4.0, fm[-1] * 4.0
    return np.geomspace(lo, hi, int(round(_PER_OCTAVE * math.log2(hi / lo))) + 1)


def band_absorption(
    admittance: Admittance, bands: int, total: int, rate: float | None = None
) -> np.ndarray:
    """The random-incidence absorption of ``admittance`` in the lowest ``bands`` of the
    ``total`` bands, counted as :func:`fit_admittance` counts it."""
    f = _frequencies(bands, total)
    return band_weights(f, bands, total) @ random_incidence_absorption(admittance(f, rate))


def fit_admittance(
    absorption: Sequence[float], total: int, rate: float | None = None
) -> Admittance:
    """The admittance whose random-incidence absorption, at ``rate`` steps a second
    (None: exactly), matches ``absorption`` in each of the lowest len(absorption) of the
    ``total`` (8 octave or 24 third-octave) bands."""
    target = tuple(float(a) for a in np.clip(np.asarray(absorption, float), 0.0, MAX_ABSORPTION))
    return _fit(target, int(total), None if rate is None else float(rate))


@functools.lru_cache(maxsize=256)
def _fit(target: tuple[float, ...], total: int, rate: float | None) -> Admittance:
    from scipy.optimize import least_squares

    from acoustic_spaces.materials.bands import BANDS_PER_OCTAVE, exact_centres, resolution_of

    goal = np.array(target)
    bands = len(goal)
    if np.all(goal <= 0.0):
        return Admittance(0.0, achieved=np.zeros(bands))
    if np.ptp(goal) < 1e-9:  # the same in every band: a constant
        beta = float(admittance(goal[:1])[0])
        out = Admittance(beta)
        return Admittance(beta, achieved=band_absorption(out, bands, total, rate))

    resolution = resolution_of(total)
    step = 2.0 ** (1.0 / BANDS_PER_OCTAVE[resolution])
    fm = exact_centres(resolution)[:bands]
    resonance = np.concatenate([[fm[0] / step], fm, [fm[-1] * step]])
    own_quality = 1.0 / (math.sqrt(step) - 1.0 / math.sqrt(step))
    f = _frequencies(bands, total)
    weights = band_weights(f, bands, total)
    n = len(resonance)
    ones = np.ones(n)

    def admittance_of(beta, conductance, quality):
        return beta + _branch_values(f, conductance, quality, resonance, rate).sum(axis=0)

    def misses(x, free):
        quality = np.exp(x[1 + n :]) if free else own_quality * ones
        found = weights @ random_incidence_absorption(admittance_of(x[0], x[1 : 1 + n], quality))
        if not free:
            return found - goal
        return np.concatenate([found - goal, _QUALITY_COST * (x[1 + n :] - math.log(own_quality))])

    # Start from the lowest band's constant, each branch adding the rest of its band's.
    base = float(admittance(np.array([goal.min()]))[0])
    ends = np.concatenate([[goal[0]], goal, [goal[-1]]])
    start = np.concatenate([[base], np.maximum(admittance(ends) - base, 1e-4)])
    fixed = least_squares(misses, start, bounds=(0.0, np.inf), args=(False,))
    x, quality = fixed.x, own_quality * ones
    if np.abs(fixed.fun).max() > _TOLERANCE:
        log_q = np.full(n, math.log(own_quality))
        lower = np.concatenate([np.zeros(1 + n), np.full(n, math.log(_QUALITY[0]))])
        upper = np.concatenate([np.full(1 + n, np.inf), np.full(n, math.log(_QUALITY[1]))])
        free = least_squares(
            misses, np.concatenate([x, log_q]), bounds=(lower, upper), args=(True,)
        )
        if np.abs(free.fun[:bands]).max() < np.abs(fixed.fun).max():
            x, quality = free.x[: 1 + n], np.exp(free.x[1 + n :])
    used = x[1:] > 1e-9
    out = Admittance(float(x[0]), x[1:][used], quality[used], resonance[used])
    return Admittance(
        out.beta,
        out.conductance,
        out.quality,
        out.frequency_hz,
        achieved=band_absorption(out, bands, total, rate),
    )


#: Corner frequencies of :func:`fit_impedance`'s branches: this many per octave, from
#: :data:`_RELAXATION_REACH` octaves below the lowest band to as many above the highest.
_RELAXATION_PER_OCTAVE = 1
_RELAXATION_REACH = 3


def fit_impedance(
    admittance: Callable[[np.ndarray], np.ndarray],
    bands: int,
    total: int,
    rate: float | None = None,
) -> Admittance:
    """The admittance whose value, at ``rate`` steps a second (None: exactly), matches
    that of a surface with an impedance, ``admittance(f)`` (normalised, complex, in the
    engine's convention, at frequencies f in Hz), over the lowest ``bands`` of the
    ``total`` bands: its reflection at every angle and its phase, not only its
    random-incidence absorption, as a ground's dip needs.

    A porous ground's admittance grows with frequency at a nearly constant phase, as no
    resonant branch does: the fit is ``beta`` and branches without mass whose corner
    frequencies are spread evenly in log frequency, whose sum makes that shape (as a
    ladder of resistors and capacitors makes a fractional-order element). With the
    corners fixed the admittance is linear in the branches' conductances, which the fit
    finds by least squares, at least 0 each, with the misses relative to the
    admittance's size."""
    from acoustic_spaces.materials.bands import BANDS_PER_OCTAVE, exact_centres, resolution_of

    resolution = resolution_of(total)
    half = 2.0 ** (0.5 / BANDS_PER_OCTAVE[resolution])
    fm = exact_centres(resolution)[:bands]
    f = _frequencies(bands, total)
    f = f[(f >= fm[0] / half) & (f <= fm[-1] * half)]
    target = np.asarray(admittance(f), dtype=complex)
    return _fit_values(
        tuple(target.real.tolist()),
        tuple(target.imag.tolist()),
        bands,
        int(total),
        None if rate is None else float(rate),
    )


@functools.lru_cache(maxsize=64)
def _fit_values(
    real: tuple[float, ...], imag: tuple[float, ...], bands: int, total: int, rate: float | None
) -> Admittance:
    from scipy.optimize import least_squares

    from acoustic_spaces.materials.bands import BANDS_PER_OCTAVE, exact_centres, resolution_of

    resolution = resolution_of(total)
    half = 2.0 ** (0.5 / BANDS_PER_OCTAVE[resolution])
    fm = exact_centres(resolution)[:bands]
    f = _frequencies(bands, total)
    f = f[(f >= fm[0] / half) & (f <= fm[-1] * half)]
    target = np.array(real) + 1j * np.array(imag)
    scale = np.abs(target) + 1e-3
    low = fm[0] / 2.0**_RELAXATION_REACH
    octaves = math.log2(fm[-1] / fm[0]) + 2 * _RELAXATION_REACH
    corner = low * 2.0 ** (
        np.arange(int(round(octaves * _RELAXATION_PER_OCTAVE)) + 1) / _RELAXATION_PER_OCTAVE
    )
    columns = np.vstack(
        [np.ones((1, len(f))), _relaxation_values(f, np.ones(len(corner)), corner, rate)]
    )
    a = columns.T / scale[:, None]  # (frequencies, 1 + branches)
    matrix = np.vstack([a.real, a.imag])
    goal = np.concatenate([(target / scale).real, (target / scale).imag])
    start = np.full(matrix.shape[1], 1e-3)
    x = least_squares(lambda x: matrix @ x - goal, start, bounds=(0.0, np.inf)).x
    used = x[1:] > 1e-9
    out = Admittance(float(x[0]), relaxation=x[1:][used], relaxation_hz=corner[used])
    return Admittance(
        out.beta,
        achieved=band_absorption(out, bands, total, rate),
        relaxation=out.relaxation,
        relaxation_hz=out.relaxation_hz,
    )


__all__ = [
    "MAX_ABSORPTION",
    "Admittance",
    "absorption_of_impedance",
    "admittance",
    "band_absorption",
    "band_weights",
    "fit_admittance",
    "fit_impedance",
    "random_incidence_absorption",
]
