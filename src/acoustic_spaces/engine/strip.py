"""A thin rigid barrier on a rigid plane in still air: what it sheds beyond a screen with
no field behind it.

The solver for rows (:mod:`~acoustic_spaces.engine.rows`) marches the field over a
barrier as over a screen with no field below its top, and adds what the rigid top
sheds beyond that. The uniform theory of diffraction gives that for a half-plane, lit
by the source and by its image in the surface the barrier stands on: the first order
of the waves between the top and the surface. The sound that goes on between the two
(from the top down the barrier to the surface and back, and so on) is not in it,
which is up to 3.2 dB for a barrier 10 cm high and 1.8 dB for one 1.1 m high in the
63 to 500 Hz octave bands (docs/validation.md, "Rows of roofs and barriers against a
wave calculation").

On a rigid plane the barrier and its image in the plane are a rigid strip in free
space, lit by the source and its image, and its field is known without such a limit.
In the vertical plane across the barrier (two dimensions, time as exp(-i omega t),
G = i/4 H0(kr) a unit line source's field) the strip sheds a double layer,

    T_rigid(L) = int phi(z') dG(L; 0, z')/dx' dz'    over the strip, |z'| < H,

whose density, the jump of the field across the strip, makes the strip rigid:

    f.p. int phi(z') (i k / 4) H1(k |z - z'|) / |z - z'| dz' = - du_S/dx (0, z)

with u_S the field of the source and of its image without the barrier. phi is the
root of 1 - (z / H)^2 times a sum of Chebyshev polynomials of the second kind, on
which the kernel's part 1 / (2 pi (z - z')^2) acts in closed form; the rest of the
kernel, a smooth function and another times a logarithm, is summed with the root's
Gauss-Chebyshev rule and with the rule that takes the logarithm in closed form
(:func:`_density`, :func:`_frame`, :func:`_remainder`). The density is solved for the
end farther from the barrier and summed against the field of the nearer one, which
changes on the strip over that end's distance from it: in panels graded about the
end's foot (:func:`_graded`), however close the end is.

The march's screen, in the mean of the march from the source and that from the
listener, is Kirchhoff's: it sheds

    T_screen(L) = int (u_L du_S/dx - u_S du_L/dx) dz'    over the strip,

with u_L the field of a unit line source at the listener. What the barrier adds to
the marched field is T_rigid - T_screen (:func:`beyond_screen`), the first order of
the difference between the rigid barrier and the march's screen in the field the
marches bring; the uniform theory's terms are what is left of it for a high barrier.

A point source's field is the sum of the fields in two dimensions over the
wavenumbers along the barrier, p = 4 int g(sqrt(k^2 - q^2)) dq from 0 on (free field
exp(ikr) / r), summed on the ray q = exp(-i pi / 4) s, where a path's field falls off
with its length l as exp(-Im(sqrt(k^2 + i s^2)) l): as exp(-s^2 l / (2 k)) while s is
far below k, and as exp(-s l / sqrt(2)) far above it (:func:`_ray`). The root of a
path's length, with which the solver for rows makes a point source's field of the
march's, does not hold for this field: at 44.5 Hz behind a barrier 1 m high it is
13 % off with the source 2 m before the barrier and the listener 3.5 m past it, and
3 % with the two 10 and 15 m from it (tests/test_strip.py).

Where the way between the ends crosses the barrier at a slant, the ends are apart along
it, by y, and p = 2 int g(sqrt(k^2 - q^2)) exp(i q y) dq over every q. A path's field
then turns slowest at q = k cos(beta), tan(beta) = l / y, and would grow on the ray. The
sum goes in from the second quadrant to q = 0, along the real axis to the q of the
straight way between the ends, and out into the fourth quadrant on a line on which every
path's field falls off, with the wavenumber across the barrier in the quarter plane in
which the strip's equation is solved all the way (:func:`_offset`). That is the strip's
part; the screen's is the marches', which run in the vertical plane along the way over a
screen across it, and is summed on the ray as before. Against the reference's sum over
the real wavenumbers for ends apart along the barrier (validation/barrier_strip.py;
eight places, 44.5 to 500 Hz) the rigid strip's shed sound is within 1.5e-5 of itself at
15 degrees from across the barrier, 4.3e-6 at 45, 7.6e-6 at 60 and 3.0e-5 at 75, with
113 to 179 wavenumbers up to 45 degrees, 147 to 261 at 60 and 242 to 411 at 75, where
the ray has 48.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy.special import hankel1, jv

#: The strip's equation is solved where k H is this at most: its cost goes with the
#: square of k H. Above it the uniform theory's terms stand (:func:`beyond_screen`
#: returns None).
MOST_KH = 32.0
#: The density's terms: ``TERMS_OVER`` more than |kappa| H, and ``TERMS_NEAR`` times the
#: barrier's height over the distance of the end that is farther from it, whose field
#: on the barrier they have to hold. They hold what the barrier adds to 6e-5 of itself
#: at k H = 31 with the farther end far off and the nearer one at the barrier's foot,
#: the least found (4e-5 dB in the level; 4e-5 of itself with that end 4 cm from the
#: foot, 2e-7 at k H = 13 and 1e-9 at k H = 8; with 20 over |kappa| H, 1e-8).
#: ``TERMS_MOST`` at most, for the cost, which up to there goes with their square, the
#: kernel's at each collocation point and node (None beyond). Where the ends are
#: close to each other the ray reaches 30 over the distance between them, and the
#: count is about 30 H over that distance and 5 H over the farther end's distance
#: from the barrier, and 12: beyond ``TERMS_MOST`` where 6 H over the first and H
#: over the second come to more than 30 together (for a way across the barrier; at a
#: slant more: :func:`beyond_screen`). That is where the ends are closer
#: to each other than about a quarter of the barrier's height (at one height: both
#: within about an eighth of it from the barrier, or one at its face and the other
#: within a quarter), where both are within about a twentieth of its height from the
#: barrier, at whatever heights, and between the two (a third of the height apart
#: and both 0.08 of it from the barrier).
TERMS_OVER = 12
TERMS_NEAR = 5.0
TERMS_MOST = 160
#: Nodes of the rule for the root's weight, for each term (with two, what a barrier
#: with k H = 31 adds is the same to ten digits).
NODES_A_TERM = 3
#: Nodes of the sum over the wavenumbers along the barrier.
RAY_NODES = 48
#: The sums over the strip against an end's field (:func:`_graded`): the nodes of a
#: panel, and the most by which the summed function's phase turns over half a panel
#: (rad).
PANEL_NODES = 12
PANEL_TURN = 4.0
#: The sum for ends offset along the barrier (:func:`_offset`). The line on which it
#: leaves the real axis goes down at ``TILT`` of the angle between the straight way
#: and the barrier (45 degrees at most), with ``RAY_NODES`` nodes times the cotangent
#: of twice the line's angle where that is more than 1 (along it the summed function's
#: phase turns that many times as often). On the real axis the strip's part is taken
#: at ``REAL_POINTS`` Chebyshev points and one more for each radian that the phase of
#: its longest wave that counts turns, where it turns fastest, over a radian of the
#: points' angle: the wave that crosses the strip ``CROSSINGS`` times from one of its
#: edges, the barrier's top and the top's image, to the other (with 6 crossings the
#: sound shed is within 2e-7 of that with 16 at k H = 8 to 31 and slants of 15 to 75
#: degrees, with 4 up to 2.6e-5 off and with 2 up to 1.4e-2). Between the points it
#: is summed with a rule of that many nodes, ``SUM_NODES`` and one more for each
#: radian of the phase of the straight way's wave.
TILT = 0.5
REAL_POINTS = 16
CROSSINGS = 6
SUM_NODES = 32
#: The sum for ends offset along the barrier is measured up to this slant (degrees).
SLANT_MOST_DEG = 75.0
#: Below this |kappa zeta| the kernel's remainder is summed from its series.
_SERIES_BELOW = 0.5
#: The factor of the remainder's logarithm is faded out where Im(kappa) zeta passes
#: this (:func:`_remainder`).
_FADE_AT = 12.0
_EULER = 0.5772156649015329


def beyond_screen(k, height: float, source, listener, slant: float = 0.0) -> np.ndarray | None:
    """(F,) what a thin rigid barrier ``height`` m high on a rigid plane sheds at the
    listener beyond a screen with no field behind it (the mean of the two marches'),
    at the wavenumbers ``k`` (F,, rad/m), for a point source (free field
    exp(ikr) / r). ``source`` and ``listener`` are (m from the barrier, m above the
    plane), each on its side of the barrier. None where ``k`` times the height is
    above ``MOST_KH`` at any wavenumber or the density would need more than
    ``TERMS_MOST`` terms (about where six times the height over the distance between
    the ends and the height over the farther end's distance from the barrier come to
    more than 30 together: the ends closer to each other than about a quarter of the
    height, both within about a twentieth of it from the barrier, or between the
    two).

    With a ``slant`` (radians, ``SLANT_MOST_DEG`` degrees at most: a ValueError
    beyond) the barrier is turned by that angle from across the way between the ends,
    seen from above, and the ends' distances from it are along that way: across the
    barrier they are those times the slant's cosine, and along it the ends are the two
    distances' sum times its sine apart. The strip's part is summed for ends that far
    apart along it (:func:`_offset`); the screen's part is the marches', which run
    along the way: that of a screen across the way. At a slant the ends are closer to
    the barrier across it and the sum reaches further out, so the density needs more
    terms, and None comes with the ends further from the barrier than the rule above
    gives for a way across it: with both ends halfway up the barrier and as far from it
    as each other along the way, the equation is solved from 0.14 of its height across
    the barrier, 0.16 at 15 degrees, 0.20 at 30, 0.26 at 45, 0.40 at 60 and 0.80 at 75
    (k H up to 24)."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    height = float(height)
    (ds, zs), (dl, zl) = source, listener
    if height <= 0.0 or ds <= 0.0 or dl <= 0.0 or zs < 0.0 or zl < 0.0:
        raise ValueError("a barrier above the plane, between two ends on or above the plane")
    if not abs(slant) <= math.radians(SLANT_MOST_DEG):
        raise ValueError(f"a slant of {SLANT_MOST_DEG:.0f} degrees at most")
    if float(np.max(k)) * height > MOST_KH:
        return None
    # The density is solved for the end farther from the barrier, whose field on it is
    # the smoother one, and summed against the nearer one's: the sound is the same with
    # the two swapped.
    ends = sorted(((ds, zs), (dl, zl)), key=lambda end: _away(end, height))
    near, far = ends[0], ends[1]
    reach = math.hypot(ds + dl, zl - zs)
    out = np.empty(len(k), dtype=complex)
    if slant:
        cosine, along = math.cos(slant), (ds + dl) * abs(math.sin(slant))
        across = sorted(((ds * cosine, zs), (dl * cosine, zl)), key=lambda end: _away(end, height))
        straight = math.hypot((ds + dl) * cosine, zl - zs)
        # The longest of the strip's waves that counts for the points on the real axis:
        # from the image of one end over the top to the image of the other, and
        # ``CROSSINGS`` times across the strip, twice the barrier's height.
        longest = sum(math.hypot(end[0], height + end[1]) for end in across)
        longest += 2.0 * CROSSINGS * height
    for index, wavenumber in enumerate(k):
        kappa, weights = _ray(float(wavenumber), reach)
        if slant:
            marched = np.sum(weights * _screen(kappa, height, far, near))
            kappa, weights = _offset(float(wavenumber), straight, along, longest)
        most = float(np.max(np.abs(kappa)))
        closest = _away(across[1] if slant else far, height)
        terms = math.ceil(most * height + TERMS_OVER + TERMS_NEAR * height / closest)
        if terms > TERMS_MOST:
            return None
        if slant:
            rigid = _rigid(kappa, height, across[1], across[0], terms)
            out[index] = np.sum(weights * rigid) - marched
            continue
        shed = _rigid(kappa, height, far, near, terms) - _screen(kappa, height, far, near)
        out[index] = np.sum(weights * shed)
    return out


def _away(end, height: float) -> float:
    """How far (m) an end (m from the barrier, m above the plane) is from the barrier."""
    return math.hypot(end[0], max(end[1] - height, 0.0))


def _ray(wavenumber: float, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """The wavenumbers in the plane across the barrier (complex) and the weights with
    which a point source's field is the sum of the fields in two dimensions: the
    wavenumbers along the barrier on the ray exp(-i pi / 4) s, as far as a path
    ``reach`` m long, the shortest, still counts."""
    furthest = max(6.5 * math.sqrt(wavenumber / reach), 30.0 / reach)
    g, w = np.polynomial.legendre.leggauss(RAY_NODES)
    s, w = 0.5 * furthest * (g + 1.0), 0.5 * furthest * w
    return np.sqrt(wavenumber**2 + 1j * s**2), 4.0 * np.exp(-0.25j * math.pi) * w


def _offset(
    wavenumber: float, reach: float, along: float, longest: float
) -> tuple[np.ndarray, np.ndarray]:
    """The wavenumbers in the plane across the barrier and the weights with which a
    point source's field is the sum of the fields in two dimensions, for ends
    ``along`` m apart along the barrier and ``reach`` m apart across it by the
    straight way: p = 2 int g(sqrt(k^2 - q^2)) exp(i q along) dq over all q.

    A wave of g with a path l m long across the barrier turns as
    exp(i (sqrt(k^2 - q^2) l + q along)), slowest at q = k cos(beta), where
    tan(beta) = l / along: that of the straight way furthest out, at q0, and every
    longer one before it. The sum goes where nothing grows and the wavenumber across
    stays in the quarter plane in which the strip's equation is solved (its real and
    imaginary parts not negative): in from the ray q = -exp(-i pi / 4) s to 0, as
    head-on; along the real axis from 0 to q0; and out on the line
    q = q0 + exp(-i phi) s, with phi less than beta, on which a wave of the path l
    falls off as exp(-s (l cos(phi) - along sin(phi))) far out. On the real axis the
    waves turn fast but g times exp(-i kappa reach) does not: it is taken at
    Chebyshev points of the angle theta, q = k cos(theta), as many as hold its wave
    of the path ``longest`` (m), the longest that counts, and summed against the
    straight way's wave with a rule fine enough for that wave."""
    whole = math.hypot(reach, along)
    beta = math.atan2(reach, along)
    g, w = np.polynomial.legendre.leggauss(RAY_NODES)
    # In from the second quadrant: exp(i q along) falls off as exp(-s along / sqrt(2)).
    furthest = max(6.5 * math.sqrt(wavenumber / reach), 30.0 / reach)
    furthest = min(furthest, 30.0 * math.sqrt(2.0) / along)
    s, ws = 0.5 * furthest * (g + 1.0), 0.5 * furthest * w
    kappa_in = np.sqrt(wavenumber**2 + 1j * s**2)
    turn = np.exp(1j * np.exp(0.75j * math.pi) * s * along)
    weights_in = 2.0 * np.exp(-0.25j * math.pi) * ws * turn
    # The real axis, by the angle theta from beta to a right angle.
    span = 0.5 * math.pi - beta
    # The points are evenly spaced in the angle t, x = cos(t), and hold a wave whose
    # phase turns by less than half a turn from one to the next where it turns fastest.
    spaced = np.linspace(0.0, math.pi, 181)
    swept = beta + 0.5 * span * (1.0 + np.cos(spaced))
    fastest = float(np.max(np.cos(swept) * 0.5 * span * np.sin(spaced)))
    count = REAL_POINTS + math.ceil(wavenumber * (longest - reach) * fastest)
    at = math.pi * (2.0 * np.arange(count) + 1.0) / (2.0 * count)
    x = np.cos(at)
    kappa_real = wavenumber * np.sin(beta + 0.5 * span * (x + 1.0)) + 0.0j
    nodes = count + SUM_NODES + math.ceil(wavenumber * (whole - reach))
    t, wt = np.polynomial.legendre.leggauss(nodes)
    theta = beta + 0.5 * span * (t + 1.0)
    carrier = np.exp(1j * wavenumber * (reach * np.sin(theta) + along * np.cos(theta)))
    carrier = carrier * (wavenumber * np.sin(theta) * 0.5 * span * wt)
    # The polynomial through the points, at the rule's nodes (the barycentric form).
    gap = t[:, None] - x[None, :]
    on = np.abs(gap) < 1e-14
    basis = ((-1.0) ** np.arange(count) * np.sin(at))[None, :] / np.where(on, 1.0, gap)
    basis = np.where(on.any(axis=1)[:, None], on.astype(float), basis)
    basis = basis / basis.sum(axis=1)[:, None]
    weights_real = 2.0 * (carrier @ basis) * np.exp(-1j * kappa_real * reach)
    # Out into the fourth quadrant.
    phi = min(0.25 * math.pi, TILT * beta)
    width = math.sin(beta) * math.sqrt(wavenumber / (whole * math.sin(2.0 * phi)))
    furthest = max(6.5 * width, 30.0 / (math.sqrt(2.0) * whole * math.sin(beta - phi)))
    g, w = np.polynomial.legendre.leggauss(
        math.ceil(RAY_NODES * max(1.0, 1.0 / math.tan(2.0 * phi)))
    )
    s, ws = 0.5 * furthest * (g + 1.0), 0.5 * furthest * w
    q = wavenumber * math.cos(beta) + np.exp(-1j * phi) * s
    kappa_out = np.sqrt(wavenumber**2 - q**2)
    weights_out = 2.0 * np.exp(-1j * phi) * ws * np.exp(1j * q * along)
    return (
        np.concatenate([kappa_in, kappa_real, kappa_out]),
        np.concatenate([weights_in, weights_real, weights_out]),
    )


def _line(kappa: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(Q, n) a unit line source's field ``distance`` (n,) m away."""
    return 0.25j * hankel1(0, kappa[:, None] * distance[None, :])


def _slope(kappa: np.ndarray, distance: np.ndarray, across: float) -> np.ndarray:
    """(Q, n) that field's rate of change with the place across the barrier, towards
    the source, ``across`` m off: on the barrier, d/dx' of G(end; 0, z')."""
    return (
        0.25j
        * kappa[:, None]
        * hankel1(1, kappa[:, None] * distance[None, :])
        * (across / distance)[None, :]
    )


def _remainder(kappa: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The strip's kernel (i kappa / 4) H1(kappa zeta) / zeta less its part
    1 / (2 pi zeta^2), for distances ``zeta`` > 0 along the strip, as (Q, ...) a
    smooth part and the factor of ln(zeta): the remainder is the first plus the second
    times ln(zeta). The factor is -kappa J1(kappa zeta) / (2 pi zeta) times
    exp(-(Im(kappa) zeta / ``_FADE_AT``)^8). On the ray J1 grows as
    exp(Im(kappa) zeta) while the remainder falls off as fast, and with the whole of
    it the two parts would cancel to rounding times that growth; faded, the factor
    grows by less than exp(12), and what the fade leaves of the logarithm in the
    smooth part is as smooth as zeta^8 ln(zeta). For small |kappa zeta| from the
    series of H1, in which each part is apart."""
    shape = (-1,) + (1,) * zeta.ndim
    x = kappa.reshape(shape) * zeta[None]
    lead = np.broadcast_to(0.25j * kappa.reshape(shape) / zeta[None], x.shape)
    fade = (kappa.imag.reshape(shape) * zeta[None] / _FADE_AT) ** 8
    kept = np.exp(-fade)
    # Where nothing is kept of it, J1 may be beyond the largest number.
    bessel = jv(1, np.where(kept > 0.0, x, 0.0))
    logarithm = 2.0j / math.pi * lead * bessel * kept
    log_zeta = np.broadcast_to(np.log(zeta[None]), x.shape)
    with np.errstate(all="ignore"):
        smooth = lead * hankel1(1, x) - 1.0 / (2.0 * math.pi * zeta[None] ** 2)
        smooth = smooth - logarithm * log_zeta
    small = np.abs(x) < _SERIES_BELOW
    if np.any(small):
        half = 0.5 * x[small]
        # H1(x) + 2i / (pi x) = J1(x) (1 + 2i/pi ln(x/2)) - i/pi sum (-1)^m
        # (psi(m+1) + psi(m+2)) / (m! (m+1)!) (x/2)^(2m+1)
        total = np.zeros_like(half)
        harmonic, power, factor = 0.0, half.copy(), 1.0
        for m in range(8):
            digammas = -2.0 * _EULER + 2.0 * harmonic + 1.0 / (m + 1)
            total += (-1.0) ** m * digammas * factor * power
            harmonic += 1.0 / (m + 1)
            factor /= (m + 1.0) * (m + 2.0)
            power = power * half * half
        own = np.broadcast_to(np.log(0.5 * kappa).reshape(shape), x.shape)[small]
        series = bessel[small] * (1.0 + 2.0j / math.pi * own) - 1.0j / math.pi * total
        # What the fade leaves of the logarithm: J1 (1 - exp(-fade)) 2i/pi ln(zeta).
        series -= 2.0j / math.pi * bessel[small] * np.expm1(-fade[small]) * log_zeta[small]
        smooth[small] = lead[small] * series
    return smooth, logarithm


def _rule(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Chebyshev nodes and weights for int sqrt(1 - s^2) f(s) ds over -1 to 1."""
    angle = math.pi * np.arange(1, nodes + 1) / (nodes + 1)
    return np.cos(angle), math.pi / (nodes + 1) * np.sin(angle) ** 2


def _even(terms: int, t: np.ndarray) -> np.ndarray:
    """(terms, len(t)) the Chebyshev polynomials of the second kind of even degree,
    U_0, U_2, ..., at ``t``."""
    return _even_at(terms, np.arccos(np.clip(t, -1.0, 1.0)))


def _even_at(terms: int, angle: np.ndarray) -> np.ndarray:
    """(terms, len(angle)) those polynomials at the cosines of ``angle`` (rad, between
    0 and pi)."""
    degree = 2 * np.arange(terms)[:, None]
    return np.sin((degree + 1) * angle[None, :]) / np.sin(angle)[None, :]


def _graded(
    lower: float, upper: float, at: float, finest: float, coarsest: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of a sum from ``lower`` to ``upper`` for a function that
    changes over ``finest`` at ``at`` and elsewhere over its distance from there, as
    an end's field does on the strip at the end's foot: Gauss-Legendre panels of
    ``PANEL_NODES`` nodes, ``finest`` wide at ``at`` (taken into the range) and as
    wide as they are far from it further off, ``coarsest`` at most."""
    at = min(max(at, lower), upper)
    edges = [at]
    for end in (lower, upper):
        span = abs(end - at)
        offset, width, side = 0.0, min(finest, coarsest), []
        while offset < span:
            offset = min(offset + width, span)
            side.append(at + math.copysign(offset, end - at))
            width = min(max(width, offset), coarsest)
        edges = side[::-1] + edges if end == lower else edges + side
    left, right = np.array(edges[:-1]), np.array(edges[1:])
    g, w = np.polynomial.legendre.leggauss(PANEL_NODES)
    half = 0.5 * (right - left)
    nodes = (0.5 * (left + right))[:, None] + half[:, None] * g[None, :]
    return nodes.ravel(), (half[:, None] * w[None, :]).ravel()


@functools.lru_cache(maxsize=32)
def _frame(count: int):
    """What the density's equation with ``count`` even terms is set up on: the
    collocation points t (the positive zeros of U_2n), the rule's nodes and weights
    for the root's weight (none of the nodes a collocation point), the polynomials
    at both, the distances |t - s|, and the rule (j, m) for the root's weight times
    ln|t_j - s| times a smooth function, from that function's values at the nodes:
    its expansion in the polynomials U_p, each of which the logarithm takes in closed
    form (int sqrt(1 - s^2) U_p(s) ln|t - s| ds = pi/2 (T_{p+2}(t) / (p + 2) -
    T_p(t) / p), and pi (t^2 / 2 - 1/4 - ln(2) / 2) for p = 0)."""
    t = np.cos(math.pi * np.arange(1, count + 1) / (2 * count + 1))
    nodes = NODES_A_TERM * 2 * count
    while math.gcd(nodes + 1, 2 * count + 1) != 1:
        nodes += 1
    s, w = _rule(nodes)
    degree = np.arange(nodes)[:, None]
    angle_s, angle_t = np.arccos(s)[None, :], np.arccos(t)[None, :]
    all_at_s = np.sin((degree + 1) * angle_s) / np.sin(angle_s)  # (p, m)
    with np.errstate(divide="ignore"):
        above = np.cos((degree + 2) * angle_t) / (degree + 2)  # (p, j), T_{p+2}(t) / (p + 2)
        taken = 0.5 * math.pi * (above - np.cos(degree * angle_t) / degree)
    taken[0] = math.pi * (0.5 * t**2 - 0.25 - 0.5 * math.log(2.0))
    log_rule = 2.0 / math.pi * (taken.T @ all_at_s) * w[None, :]  # (j, m)
    apart = np.abs(t[:, None] - s[None, :])
    return t, s, w, _even(count, t), _even(count, s), apart, log_rule


def _density(kappa: np.ndarray, height: float, end, terms: int) -> np.ndarray:
    """The jump of the field across the strip for a unit line source at ``end`` (m from
    the strip, m above the plane) and its image in the plane, at the wavenumbers
    ``kappa`` (Q,): the coefficients (Q, n) of sqrt(1 - t^2) U_2m(t), t = z / H (the
    two sources together are even about the plane), by collocation at the positive
    zeros of U_2n."""
    count = math.ceil(0.5 * terms)
    t, s, w, at_t, at_s, apart, log_rule = _frame(count)
    smooth, logarithm = _remainder(kappa, height * apart)  # (Q, j, m)
    # ln(zeta) = ln(H) + ln|t - s|: the first with the rule's weights, the second
    # with the rule for the logarithm.
    weighted = (smooth + math.log(height) * logarithm) * w[None, None, :]
    weighted += logarithm * log_rule[None]
    matrix = height * np.einsum("qjm,nm->qjn", weighted, at_s)
    # f.p. int sqrt(1 - s^2) U_n(s) / (t - s)^2 ds = -pi (n + 1) U_n(t).
    matrix += (-(2.0 * np.arange(count) + 1.0)[None, :] * at_t.T / (2.0 * height))[None]
    distance, above = end
    z = height * t
    right = np.zeros((len(kappa), count), dtype=complex)
    for image in (above, -above):
        right += _slope(kappa, np.hypot(distance, z - image), distance)
    return np.linalg.solve(matrix, right[:, :, None])[:, :, 0]


def _rigid(kappa: np.ndarray, height: float, far, near, terms: int) -> np.ndarray:
    """(Q,) what the rigid strip sheds at ``near`` of a unit line source at ``far`` and
    its image in the plane (each: m from the strip, m above the plane; the two on
    either side of it)."""
    coefficients = _density(kappa, height, far, terms)
    count = coefficients.shape[1]
    distance, above = near
    # Summed over the angle whose cosine is z' / H, in which the root's weight is the
    # sine's square. The nearer end's field on the strip changes fastest where it
    # would be without end, at the cosine (above + i distance) / H: over that
    # angle's imaginary part, where its real part is.
    foot = np.arccos(complex(above, distance) / height)
    turns = 2 * count + 1 + float(np.max(np.abs(kappa))) * height
    angle, w = _graded(0.0, math.pi, foot.real, abs(foot.imag), 2.0 * PANEL_TURN / turns)
    jump = coefficients @ _even_at(count, angle)  # (Q, m), without the root
    out_ward = _slope(kappa, np.hypot(distance, above - height * np.cos(angle)), distance)
    return height * np.sum(jump * out_ward * (w * np.sin(angle) ** 2)[None, :], axis=1)


def _screen(kappa: np.ndarray, height: float, far, near) -> np.ndarray:
    """(Q,) what a screen with no field behind it sheds at ``near`` of a unit line
    source at ``far`` and its image in the plane, in the mean of the march one way
    and that the other way: Kirchhoff's integral over the strip."""
    # Each end's field on the strip changes over its distance from the end: the
    # nearer one's sets the panels at its foot, the farther one's the widest panel,
    # as does the phase, which turns by twice |kappa| a metre at most.
    most = float(np.max(np.abs(kappa)))
    widest = min(PANEL_TURN / most, _away(far, height))
    z, w = _graded(-height, height, min(near[1], height), _away(near, height), widest)
    to_near = np.hypot(near[0], near[1] - z)
    u_near, du_near = _line(kappa, to_near), _slope(kappa, to_near, near[0])
    out = np.zeros(len(kappa), dtype=complex)
    for image in (far[1], -far[1]):
        to_far = np.hypot(far[0], z - image)
        u_far, du_far = _line(kappa, to_far), -_slope(kappa, to_far, far[0])
        out += np.sum((u_near * du_far - u_far * du_near) * w[None, :], axis=1)
    return out


__all__ = ["MOST_KH", "beyond_screen"]
