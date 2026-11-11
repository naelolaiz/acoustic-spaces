"""The shadow zone of upward refraction: sound that creeps along the ground.

Where the speed of sound falls with height (upwind, or the ground warmed by the sun),
rays bend up and none reaches a listener far enough away: past the limiting ray, which
grazes the ground, lies a geometric shadow. Sound still gets there, diffracted round the
curved rays as round a curved surface: creeping waves along the ground, which fade with
distance (and with frequency) into the shadow.

For a linear profile of the squared wave number, k(z)**2 = k0**2 (1 + 2 z / R) (R the
radius of curvature of a ray launched horizontally), the field of a point source over a
ground of normalised admittance beta is a sum of normal modes, the residue series
(Pierce 1981, sec. 9-3; Berry and Daigle 1988; Salomons 2001, appendix):

    p = i exp(2 i pi / 3) / (4 l) sum_n H0(kappa_n r) Ai(s_n + exp(-i pi/3) zs / l)
        Ai(s_n + exp(-i pi/3) zr / l) / ((s_n - q**2) Ai(s_n)**2)

with l = (R / (2 k0**2))**(1/3), q = k0 l beta exp(-i pi/6), the roots s_n of
Ai'(s) = q Ai(s) and kappa_n**2 = k0**2 - s_n exp(i pi/3) / l**2; a unit source gives
exp(i k R) / (4 pi R) in free field, with time going as exp(-i omega t). Each mode decays
along the ground, the first slowest.

A real profile is fitted by the R that puts the shadow where its own rays put it (see
:func:`equivalent_radius`).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import airye

_ROTATE = np.exp(-1j * np.pi / 3.0)
#: The most modes summed.
MAX_MODES = 60
#: Modes are summed until the next ones have faded this much (nepers) more than the first
#: by the listener's range.
_FADED = 30.0
#: Steps from a rigid ground (q = 0) to the ground's q in the search for the roots.
_STEPS = 24


def _ratio(s: np.ndarray) -> np.ndarray:
    """Ai'(s) / Ai(s)."""
    ai, aip, _, _ = airye(s)
    return aip / ai


def rigid_roots(count: int) -> np.ndarray:
    """(count,) the zeros a'_n of Ai' (all negative): the modes over a rigid ground."""
    n = np.arange(1, count + 1)
    t = 3.0 * np.pi * (4.0 * n - 3.0) / 8.0
    s = -(t ** (2.0 / 3.0)) * (1.0 - 7.0 / (48.0 * t**2) + 35.0 / (288.0 * t**4))
    s = s.astype(complex)
    for _ in range(8):
        # Newton on Ai'(s) = 0, Ai'' = s Ai.
        s = s - _ratio(s) / s
    return s.real


def mode_roots(q: np.ndarray, count: int) -> np.ndarray:
    """(F, count) the roots s of Ai'(s) = q Ai(s) for each ``q`` (F,), by Newton's
    method from those of a rigid ground (q = 0), q growing to its value in steps."""
    q = np.asarray(q, dtype=complex)
    s = np.broadcast_to(rigid_roots(count).astype(complex), (len(q), count)).copy()
    for step in range(1, _STEPS + 1):
        target = q[:, None] * (step / _STEPS)
        for _ in range(3 if step < _STEPS else 12):
            rho = _ratio(s)
            s = s - (rho - target) / (s - rho**2)
    return s


def residue_pressure(
    k: np.ndarray,
    beta: np.ndarray,
    radius: float,
    zs: float,
    zr: float,
    r: float,
    modes: int | None = None,
) -> np.ndarray:
    """(F,) the pressure (time as exp(-i omega t), a unit source: exp(i k R) / (4 pi R)
    in free field) at heights ``zs`` and ``zr`` (m) above a ground of normalised
    admittance ``beta`` (F,), ``r`` m apart along it, for wave numbers ``k`` (F,) on the
    ground and a profile of radius ``radius`` (m, > 0: upward refraction)."""
    k = np.asarray(k, dtype=float)
    beta = np.asarray(beta, dtype=complex)
    ell = (radius / (2.0 * k**2)) ** (1.0 / 3.0)
    q = k * ell * beta * np.exp(-1j * np.pi / 6.0)
    count = MAX_MODES if modes is None else int(modes)
    s = mode_roots(q, count)
    kappa = np.sqrt(k[:, None] ** 2 - s * np.exp(1j * np.pi / 3.0) / ell[:, None] ** 2)
    if modes is None:
        # Leave out the modes that have faded far beyond the first by the listener.
        faded = (kappa.imag - kappa[:, :1].imag) * r > _FADED
        keep = ~np.all(faded, axis=0)
        last = int(np.flatnonzero(keep).max()) + 1 if keep.any() else 1
        s, kappa = s[:, :last], kappa[:, :last]
    zeta = lambda w: (2.0 / 3.0) * w**1.5  # noqa: E731
    ws = s + _ROTATE * zs / ell[:, None]
    wr = s + _ROTATE * zr / ell[:, None]
    a_s, _, _, _ = airye(ws)
    a_r, _, _, _ = airye(wr)
    a_0, _, _, _ = airye(s)
    x = kappa * r
    # Each term with its exponentials gathered, which may be far out of range apart.
    exponent = 1j * x - zeta(ws) - zeta(wr) + 2.0 * zeta(s)
    term = (
        np.sqrt(2.0 / (np.pi * x))
        * np.exp(exponent - 0.25j * np.pi)
        * a_s
        * a_r
        / ((s - q[:, None] ** 2) * a_0**2)
    )
    return 1j * np.exp(2j * np.pi / 3.0) / (4.0 * ell) * term.sum(axis=1)


def shadow_start(profile, zs: float, zr: float) -> float | None:
    """The distance (m) along the ground at which the geometric shadow of ``profile``
    (a :class:`~acoustic_spaces.engine.refraction.Profile`) starts for a source and a
    listener at heights ``zs`` and ``zr``: where the ray that grazes the ground reaches
    the listener's height; None where no shadow forms (the speed does not fall from the
    ground to either height). In a profile whose gradient eases with height (the wind's
    log profile), rays that turn just above the ground reach farther (three quarters as
    far again 5 m up in the ``windy`` preset): the shadow proper begins there. Below the
    first height whose speed differs from the ground's in floating point (and on the
    ground), the profile's own law there, linear up to its first node, decides: the
    speed falls if its gradient is negative, and the grazing ray, a circle of radius
    R = c0 / |dc/dz|, reaches height z at sqrt(2 R z - z**2), so that the start goes to
    0 continuously with the heights. Where the first layer is level (dc/dz = 0) the
    ray fan alone decides, as above it."""
    from acoustic_spaces.engine.refraction import _Fan

    c0 = float(profile.c[0])
    gradient = float(profile.gradient[0])
    heights = np.linspace(0.0, max(zs, zr, 0.0), 64)[1:]
    speeds = profile.speed(heights)
    level = (speeds == c0) & (heights < profile.z[1]) & (gradient != 0.0)
    if not np.all(np.where(level, gradient < 0.0, speeds < c0)):
        return None
    fan = _Fan(profile, np.array([1.0 / c0]), 0.0)
    if not np.isinf(fan.hi[0]) and fan.hi[0] < max(zs, zr):
        return None
    out = 0.0
    for z in (max(zs, 0.0), max(zr, 0.0)):
        if gradient < 0.0 and profile.speed(z) == c0:
            radius = c0 / abs(gradient)
            out += math.sqrt(max(2.0 * radius * z - z * z, 0.0))
        else:
            out += float(fan.at(np.array([[z]]))[0, 0, 0] - fan.at(np.array([[0.0]]))[0, 0, 0])
    return out


def equivalent_radius(start: float, zs: float, zr: float, profile) -> float:
    """The radius R of the linear profile whose shadow starts at ``start`` m for heights
    ``zs`` and ``zr``: its grazing ray reaches height z at sqrt(2 R z). With both heights
    on the ground, the limit as they go to 0, whatever ``start``: that of ``profile``'s
    own law there (see :func:`shadow_start`), c0 / |dc/dz| (infinite where its first
    layer is level, where no creeping wave forms)."""
    if zs + zr <= 0.0:
        gradient = abs(float(profile.gradient[0]))
        return float(profile.c[0]) / gradient if gradient > 0.0 else math.inf
    return start**2 / (2.0 * (math.sqrt(zs) + math.sqrt(zr)) ** 2)


__all__ = [
    "MAX_MODES",
    "equivalent_radius",
    "mode_roots",
    "residue_pressure",
    "rigid_roots",
    "shadow_start",
]
