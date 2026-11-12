"""A thin rigid barrier on a rigid plane, exactly, for a point source: the reference
that ``engine/strip.py`` and the solver for rows are held to (tests/test_strip.py,
tests/test_rows.py), written apart from the engine's.

By the plane's image the barrier, H m high, is a rigid strip |z| <= H in free space,
lit by the source and by its image. In two dimensions (time as exp(-i omega t),
G = i/4 H0(kr)) its scattered field is a double layer over the strip,
int phi(z') dG/dx' dz', whose density makes the strip rigid:

    f.p. int phi(z') (i k / 4) H1(k |z - z'|) / |z - z'| dz' = - du_inc/dx (0, z).

Unlike the engine's, this one keeps every Chebyshev term (not the even ones alone),
takes the kernel's remainder as it stands (no series), sums over the strip with the
root's Gauss-Chebyshev rule alone (with more nodes for an end close to the barrier,
not with panels graded about the end's foot), and sums a point source's field over
the wavenumbers along the barrier on the real axis, as ``validation/rows.py`` does
for the wave calculation (``point_source``):

    p = 4 [ int_0^{pi/2} g(k sin th) k sin th dth
            + int_0^inf g(i gamma) gamma / sqrt(k^2 + gamma^2) dgamma ].

:class:`Strip` solves for the source it is given; :func:`point` and :func:`levels`
give it the end that is farther along the plane from the barrier (the sound is the
same with the ends swapped).

tests/test_strip.py checks it against the plate in a uniform flow (k -> 0), its field
without the barrier against the direct sound and the plane's reflection, and what the
barrier sheds with the ends swapped; tests/test_rows.py compares its levels with the
wave calculation's for a barrier 10 cm high on a roof.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import hankel1

#: The most nodes of the rule for the sum against the nearer end's field.
FINEST_RULE = 400000


def _polynomials(terms: int, t: np.ndarray) -> np.ndarray:
    """(terms, len(t)) the Chebyshev polynomials of the second kind U_n(t)."""
    theta = np.arccos(np.clip(t, -1.0, 1.0))
    n = np.arange(terms)[:, None]
    return np.sin((n + 1) * theta[None, :]) / np.sin(theta)[None, :]


class Strip:
    """The strip |z| <= ``height`` at x = 0: ``terms`` Chebyshev terms, collocation at
    the zeros of U_terms, and a Gauss-Chebyshev rule of ``nodes`` nodes for the weight
    sqrt(1 - s^2) (none of them a collocation point)."""

    def __init__(self, height: float, terms: int = 24, nodes: int = 301):
        while math.gcd(nodes + 1, terms + 1) != 1:
            nodes += 1
        self.h, self.n, self.m = float(height), terms, nodes
        self.t = np.cos(np.pi * np.arange(1, terms + 1) / (terms + 1))
        angle = np.pi * np.arange(1, nodes + 1) / (nodes + 1)
        self.s, self.w = np.cos(angle), np.pi / (nodes + 1) * np.sin(angle) ** 2
        self.at_t, self.at_s = _polynomials(terms, self.t), _polynomials(terms, self.s)
        apart = self.t[:, None] - self.s[None, :]
        self.zeta = self.h * np.abs(apart)
        # int sqrt(1 - s^2) ln|t - s| ds, exactly, less the rule's.
        exact = np.pi * (0.5 * self.t**2 - 0.25 - 0.5 * math.log(2.0))
        self.log_fix = exact - (np.log(np.abs(apart)) * self.w[None, :]).sum(axis=1)

    def density(self, kappa: np.ndarray, source) -> np.ndarray:
        """(Q, n) the coefficients of sqrt(1 - t^2) U_n(t) at the wavenumbers ``kappa``
        (Q,, complex) for a unit line source at ``source`` (x < 0, z) and its image
        in the plane z = 0."""
        kappa = np.asarray(kappa, dtype=complex)
        zeta = self.zeta[None]
        rest = 0.25j * kappa[:, None, None] * hankel1(1, kappa[:, None, None] * zeta) / zeta
        rest = rest - 1.0 / (2.0 * np.pi * zeta**2)
        a = self.h * np.einsum("qjm,nm,m->qjn", rest, self.at_s, self.w)
        log = -(kappa**2) / (4.0 * np.pi)
        a += self.h * log[:, None, None] * (self.at_t.T * self.log_fix[:, None])[None]
        a += (-(np.arange(self.n) + 1.0)[None, :] * self.at_t.T / (2.0 * self.h))[None]
        xs, zs = source
        z = self.h * self.t
        b = np.zeros((len(kappa), self.n), dtype=complex)
        for image in (zs, -zs):
            r = np.hypot(xs, z - image)
            slope = 0.25j * kappa[:, None] * hankel1(1, kappa[:, None] * r[None, :])
            b += slope * (-xs / r)[None, :]
        return np.linalg.solve(a, b[:, :, None])[:, :, 0]

    def scattered(
        self, kappa: np.ndarray, coefficients: np.ndarray, listener, nodes: int | None = None
    ) -> np.ndarray:
        """(Q,) the strip's scattered field at ``listener`` (x > 0, z), summed with the
        equation's rule, or with one of ``nodes`` nodes: the listener's field on the
        strip changes over its distance from it, which the rule has to resolve."""
        kappa = np.asarray(kappa, dtype=complex)
        xl, zl = listener
        s, w, at_s = self.s, self.w, self.at_s
        if nodes is not None:
            angle = np.pi * np.arange(1, nodes + 1) / (nodes + 1)
            s, w = np.cos(angle), np.pi / (nodes + 1) * np.sin(angle) ** 2
            at_s = _polynomials(self.n, s)
        r = np.hypot(xl, zl - self.h * s)
        slope = 0.25j * kappa[:, None] * hankel1(1, kappa[:, None] * r[None, :]) * (xl / r)[None, :]
        return self.h * np.sum((coefficients @ at_s) * slope * w[None, :], axis=1)


def free(kappa: np.ndarray, source, listener) -> np.ndarray:
    """(Q,) the unit line fields of the source and of its image at the listener."""
    kappa = np.asarray(kappa, dtype=complex)
    xs, zs = source
    out = np.zeros(len(kappa), dtype=complex)
    for image in (zs, -zs):
        out += 0.25j * hankel1(0, kappa * math.hypot(listener[0] - xs, listener[1] - image))
    return out


def point(
    height: float, k, source, listener, terms: int | None = None, angles=None, decays=80
) -> tuple[np.ndarray, np.ndarray]:
    """(K,) a point source's pressure at the wavenumbers ``k`` (free field
    exp(ikr) / r) without the barrier (the direct sound and the plane's reflection),
    and (K,) what the rigid barrier ``height`` m high sheds; ``source`` (x < 0, z) and
    ``listener`` (x > 0, z) from the barrier's foot. ``terms`` Chebyshev terms, by
    default 24 more than 1.3 k H and 5 H over how far along the plane the end that is
    farther along it is from the barrier, and twelve nodes a term (for the sum
    against the other end's field 40 H over that end's distance from the barrier
    where that is more; a ValueError beyond ``FINEST_RULE`` nodes); ``angles`` of the
    real wavenumbers, by default 240, and 0.6 k times the ends' distance where that
    is more (the field turns by k times that distance over them)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if abs(listener[0]) > abs(source[0]):
        # The sound is the same with the ends swapped: the equation is solved for the
        # end farther along the plane from the barrier, whose field on it takes fewer
        # terms.
        source, listener = (-listener[0], listener[1]), (-source[0], source[1])
    fine = math.ceil(40.0 * height / math.hypot(listener[0], max(listener[1] - height, 0.0)))
    if fine > FINEST_RULE:
        raise ValueError("an end closer to the barrier than this reference's rule resolves")
    g, w_g = np.polynomial.legendre.leggauss(decays)
    reach = math.hypot(listener[0] - source[0], listener[1] - source[1])
    gamma, w_g = 6.0 / reach * (g + 1.0), 6.0 / reach * w_g
    open_field = np.empty(len(k), dtype=complex)
    shed = np.empty(len(k), dtype=complex)
    for index, wavenumber in enumerate(k):
        count = terms
        if count is None:
            count = int(1.3 * wavenumber * height + 24 + 5.0 * height / abs(source[0]))
        strip = Strip(height, count, 12 * count)
        turns = angles if angles is not None else max(240, math.ceil(0.6 * wavenumber * reach))
        theta, w_theta = np.polynomial.legendre.leggauss(turns)
        theta, w_theta = 0.25 * np.pi * (theta + 1.0), 0.25 * np.pi * w_theta
        kappa = wavenumber * np.sin(theta)
        weights = np.concatenate([w_theta * kappa, w_g * gamma / np.sqrt(wavenumber**2 + gamma**2)])
        both = np.concatenate([kappa.astype(complex), 1j * gamma])
        scattered = np.empty(len(both), dtype=complex)
        closer = fine if fine > strip.m else None
        for a in range(0, len(both), 32):
            part = both[a : a + 32]
            density = strip.density(part, source)
            scattered[a : a + 32] = strip.scattered(part, density, listener, closer)
        open_field[index] = 4.0 * np.sum(weights * free(both, source, listener))
        shed[index] = 4.0 * np.sum(weights * scattered)
    return open_field, shed


def levels(height: float, source, listener, wavenumbers) -> np.ndarray:
    """(B,) the band levels (dB re the free field at the listener's distance from the
    source) behind the barrier, ``wavenumbers`` (B, n) those of each band."""
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    open_field, shed = point(height, wavenumbers.ravel(), source, listener)
    energy = (np.abs(open_field + shed) ** 2).reshape(wavenumbers.shape).mean(axis=1)
    distance = math.hypot(listener[0] - source[0], listener[1] - source[1])
    return 10.0 * np.log10(energy * distance**2)
