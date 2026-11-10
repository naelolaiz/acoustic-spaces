"""Curved rays outdoors: sound bent by wind and temperature that change with height.

The atmosphere (:class:`~acoustic_spaces.scene.Atmosphere`) is stratified: the speed of
sound and the wind depend on the height above the ground only. Along each horizontal
direction sound travels at the effective speed c(z) = c_T(z) + u(z) cos(angle to the
wind), which this module takes piecewise linear in height, between nodes close together
near the ground (where the profiles change fastest) and constant above the last. In each
layer a ray is then an arc of a circle, and the distance it covers, its travel time and
its length are exact (Pierce 1981, ch. 8; Salomons 2001, ch. 3).

A ray keeps its ray parameter p = cos(elevation) / c (Snell's law) and turns back where
p c(z) = 1. The rays from a source that reach a listener, the eigenrays, are found by
shooting: every way a ray can get there (straight up or down, after turning in the air or
bouncing off the ground so many times) is a branch whose range is a function of p, and
each crossing of the listener's range is refined by bisection. There may be several
(downward refraction) or none (the shadow of upward refraction, see
:mod:`~acoustic_spaces.engine.shadow`).

Their level comes from the spreading of neighbouring rays (the ray tube), which is 1/R**2
in still air and infinite on caustics, where it is capped; each caustic a ray touches
advances its phase by a quarter period (a factor j, time going as exp(j omega t)).
"""

from __future__ import annotations

import itertools
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

#: Heights (m) of the profile's nodes: the ground, then spaced geometrically from
#: ``_FIRST_NODE_M`` to ``_TOP_M`` (the atmosphere is constant above).
_FIRST_NODE_M = 0.01
_TOP_M = 1000.0
_NODES = 160
#: Launch angles tried per branch before bisection: spaced as u**2 between horizontal
#: and vertical, closest near the horizontal, where the rays that go far are.
_LAUNCH_ANGLES = 1000
#: The most steps refining a ray parameter between two of the launch angles tried.
_BISECTIONS = 80
#: How close (m, plus a share of the range) an eigenray's range must come to the
#: listener's to count: a bracket over a jump of the range is no eigenray.
_RANGE_TOLERANCE = 1e-6
#: Where a branch of rays starts between two of the fan's, its range is searched up to
#: that start unless it runs away from the listener's there, or its first ray misses
#: by more than this many times its last step (the range of rays that only just reach
#: a height goes as the square root of how far they are from that: at most 2.4 times
#: the last step to the end).
_EDGE_REACH = 4.0
#: Turning heights (m) within this of each other count as the same: a ray at the end of
#: a branch's reach that turns at the listener's height ends two branches
#: (:meth:`Rays._folds`), and rays a float of p apart that turn at the same heights hold
#: no jump of the range between them (:meth:`Rays._bisect`).
_SAME_TURN_M = 1e-6
#: The most :meth:`Rays._graze` allows a ray taken at a fold or at a steep crossing to
#: miss the listener by, as a share of the distance: the most in a gradient weak enough
#: that neighbouring floats of p could hold any range between them.
_GRAZE_SHARE = 1e-4
#: A ray taken at a fold or at a steep crossing lies within this many floats of p of a
#: ray that runs level at the source's or the listener's height: only there do
#: neighbouring floats of p hold ranges further apart than the tolerance, the range going
#: as the square root of how far p is from there, and not by the rounding of the range.
_GRAZE_FLOATS = 4.0
#: The widths of a ray's tube from its range less its singular parts and from its range
#: itself agree where they differ by no more than this share of the width
#: (:meth:`Rays._choose`). The latter changes too much to be taken over the former where
#: it changes by more, from rays half as far apart; and, unless they agree, where its
#: rays lie ``_FEW_FLOATS`` floats of p apart or fewer or the ray is taken at a fold or a
#: steep crossing, to be taken at all.
_AGREEMENT = 1e-2
#: Where the width from the range less its singular parts cannot be had, the width from
#: the range itself is taken only where it changes by no more than this share of it from
#: rays half as far apart; for a ray that turns in the air, that from the rays a small
#: step either side is taken instead of the fitted one where the two differ by more. The
#: former is taken over the sampled profile's own slope about a node of it
#: (:meth:`Rays._sampled`) only where it changes by no more than this.
_ALONE = 1e-1
#: The width from the range less its singular parts is taken over one from the range
#: itself that it disagrees with, or alone, only where it changes by no more than this
#: share of it from rays half as far apart.
_STABLE = 1e-3
#: Rays half as far apart as others count as new ones only where those lie at least this
#: many floats of p apart (from one ray to the next): else they round to the same floats.
_HALF_FLOATS = 4.0
#: The most a ray tube focuses: a caustic's level is capped at this many times the
#: spherical spreading over the path's length (10 dB).
CAUSTIC_CAP = 10.0
#: The most times a ray turns back between the ground and the air above, for ducts
#: that hold it up off the ground.
_MAX_CYCLES = 40
#: Samples per segment of a ray where its tube's width is checked for caustics.
_CAUSTIC_SAMPLES = 10
#: The greatest sagitta (m) of a chord drawn for an arc of a ray.
CHORD_SAGITTA_M = 0.02
#: A ray that turns in the air has the slope of its range, less the parts singular at
#: its ends and the ground, fitted over rays whose turning points lie up to this many of
#: the profile's layers above and below its own, and this many rays across them, the
#: span halved up to this many times until all keep to its branch: in a profile linear
#: between nodes, the range of such rays wiggles as their turning points cross the
#: nodes, and its derivative grows without bound as a turning point nears a node where
#: the gradient changes (as one over the square root of the height between them),
#: though the smooth profile's stays finite.
_TURN_LAYERS = 2.0
_TURN_RAYS = 9
_TURN_HALVINGS = 8
#: The step in p, relative, of the slope of a ray's range less its singular parts where
#: no rays turning about its own are fitted (that range is smooth in p).
_SMOOTH_STEP = 1e-6
#: The rays a ray's tube is fitted over keep within this share of the way from it to
#: the nearest ray that runs level in a level layer of the profile: the range of the
#: rays that cross such a layer, p c dz / s through it, grows without bound there.
_LEVEL_SHARE = 1.0 / 16.0
#: The most the rounding of the parts of a ray's range singular at its ends may move
#: the width of its tube, as a share of it, for the tube to be worked out from the range
#: less them (:meth:`Rays._width`); else from the range itself.
_WIDTH_ROUNDING = 1e-4
#: Where the layers a ray's neighbours cross depart from their trend, the most the width
#: of its tube from rays a small step either side may differ from that from rays a
#: quarter of that step either side, as a share of it, for it to be taken (else it comes
#: from the range itself): the parts, from the trend, then leave some of the range's own
#: singularity about the rays that run level at their heights.
_LOOSE_AGREEMENT = 1e-3
#: Where the rays a small step either side of a ray lie fewer than this many floats
#: from its parameter p, its tube comes from them only where rays a quarter as far
#: either side agree (as ``_LOOSE_AGREEMENT``): the slope of the range from rays N
#: floats either side can carry up to about 1/N of it in rounding.
_FEW_FLOATS = 300
#: Speeds at the profile's nodes within this many units in the last place of each
#: other count as the same, and nodes this close to a line as on it: the rounding of
#: the speeds worked out at the nodes (:attr:`Profile.trend`).
_ROUNDING_ULPS = 16.0
#: The eigenray searches a fan of rays keeps (:meth:`Rays.eigenrays`).
_KEPT_SEARCHES = 256


@dataclass(frozen=True)
class Profile:
    """The effective speed of sound (m/s) along one horizontal direction, linear in
    height between nodes ``z`` (m above the ground, from 0, increasing) and constant
    above the last. ``rows`` are the heights (m) of those nodes where the profile itself
    is linear between (a table's rows), not sampled from a smooth one."""

    z: np.ndarray
    c: np.ndarray
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False, compare=False)
    #: (N + 1,) dc/dz in each layer, the constant one above the last node with it.
    gradient: np.ndarray = field(init=False, repr=False, compare=False)
    #: (N + 1,) the gradient of the straight stretch of the profile each layer lies in:
    #: the layers between two of the ``rows`` whose nodes' speeds lie on one line to
    #: rounding (``_ROUNDING_ULPS``), as a table's between its rows; 0 where the speed
    #: changes over the stretch by no more than that rounding (level, to the arithmetic)
    #: and above the last node. The singular parts of the range take it
    #: (:meth:`Rays._singular`): the gradients of the layers within a stretch differ by
    #: their nodes' rounding, which grows without bound against a weak one.
    trend: np.ndarray = field(init=False, repr=False, compare=False)
    #: The indices of the nodes that are corners of the profile, where its gradient may
    #: change by any amount: the ``rows`` and the edges of its level stretches (the top
    #: of the profile among them).
    corners: np.ndarray = field(init=False, repr=False, compare=False)
    #: The indices of the nodes where the trend changes that are not corners, as where a
    #: smooth profile is sampled: they hold no singular part of the range
    #: (:meth:`Rays._singular`), which wiggles about the smooth profile's as a ray's
    #: turning point passes one.
    gentle: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=float)
        c = np.asarray(self.c, dtype=float)
        rows = np.asarray(self.rows, dtype=float).reshape(-1)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "rows", rows)
        gradient = np.append(np.diff(c) / np.diff(z), 0.0)
        trend = _trend(z, c, np.isin(z, rows))
        level = trend == 0.0
        inner = np.arange(1, len(z))
        corner = (level[inner - 1] != level[inner]) | np.isin(z[inner], rows)
        bends = inner[trend[inner] != trend[inner - 1]]
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "trend", trend)
        object.__setattr__(self, "corners", inner[corner])
        object.__setattr__(self, "gentle", np.setdiff1d(bends, inner[corner]))

    @classmethod
    def of(cls, atmosphere, ground_c: float, towards=None) -> Profile:
        """The profile of ``atmosphere`` (:class:`~acoustic_spaces.scene.Atmosphere`)
        with the air at ``ground_c`` °C on the ground, along the horizontal unit vector
        ``towards`` (2,) (None: the speed of sound in still air)."""
        z = node_heights(atmosphere)
        c = np.asarray(atmosphere.sound_speed(z, ground_c, towards), dtype=float)
        return cls(z, c, table_heights(atmosphere))

    @classmethod
    def uniform(cls, speed: float) -> Profile:
        return cls(np.array([0.0, _TOP_M]), np.array([speed, speed], dtype=float))

    def speed(self, z) -> np.ndarray:
        """The speed at heights ``z`` (constant above the last node, and below the
        ground as on it)."""
        return np.interp(np.asarray(z, dtype=float), self.z, self.c)

    def layer(self, z) -> np.ndarray:
        """The layer each height lies in: i for z[i] <= z < z[i + 1], the last for the
        heights above (N for the constant layer above the last node)."""
        index = np.searchsorted(self.z, np.asarray(z, dtype=float), side="right") - 1
        return np.clip(index, 0, len(self.z) - 1)


def node_heights(atmosphere) -> np.ndarray:
    """The heights (m) of the nodes the profile of ``atmosphere`` is linear between: the
    geometric spacing, and a table's own heights."""
    z = np.concatenate([[0.0], np.geomspace(_FIRST_NODE_M, _TOP_M, _NODES)])
    return np.union1d(z, table_heights(atmosphere))


def table_heights(atmosphere) -> np.ndarray:
    """The heights (m) of the rows of the table of ``atmosphere``, if it has one."""
    if atmosphere is None or atmosphere.table is None:
        return np.zeros(0)
    return np.array([row[0] for row in atmosphere.table], dtype=float)


def _trend(z: np.ndarray, c: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """(N + 1,) :attr:`Profile.trend` of the speeds ``c`` at the nodes ``z``: from each
    stretch's first node, as many layers up as leave the nodes between on the line from
    its speed to its last node's, to rounding, ending at the nodes where ``breaks``."""
    tolerance = _ROUNDING_ULPS * float(np.spacing(np.max(np.abs(c))))
    out = np.zeros(len(z))
    a = 0
    while a < len(z) - 1:
        b = a + 1
        while b < len(z) - 1 and not breaks[b]:
            share = (z[a + 1 : b + 1] - z[a]) / (z[b + 1] - z[a])
            line = c[a] + share * (c[b + 1] - c[a])
            if np.any(np.abs(c[a + 1 : b + 1] - line) > tolerance):
                break
            b += 1
        rise = c[b] - c[a]
        out[a:b] = rise / (z[b] - z[a]) if abs(rise) > tolerance else 0.0
        a = b
    return out


def _log_ratio(x: np.ndarray) -> np.ndarray:
    """log(1 + x) / x, 1 at 0."""
    small = np.abs(x) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0 - 0.5 * x, np.log1p(safe) / safe)


def _arc_ratio(x: np.ndarray) -> np.ndarray:
    """arcsin(x) / x, 1 at 0."""
    small = np.abs(x) < 1e-6
    safe = np.where(small, 1.0, np.clip(x, -1.0, 1.0))
    return np.where(small, 1.0 + x * x / 6.0, np.arcsin(safe) / safe)


def _sine(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """|sin(elevation)| of a ray with parameter ``p`` where the speed is ``c``."""
    pc = p * c
    return np.sqrt(np.maximum((1.0 - pc) * (1.0 + pc), 0.0))


def _piece(p, ca, cb, dz, turns_a=False, turns_b=False):
    """The horizontal distance, travel time and length of a ray with parameter ``p``
    through a piece of a layer ``dz`` (>= 0) high, from speed ``ca`` at its bottom to
    ``cb`` at its top (linear between), where the ray never turns (p c <= 1): exact for
    an arc of a circle, written to stay accurate for straight rays and at turning
    points. Where ``turns_a`` (``turns_b``) the ray turns at the bottom (top) and its
    sine there is 0: from p (1 / p), which rounds to 1 - 1.1e-16 for one ray in six,
    it would be 1.5e-8 and move the range (by 6e-5 m 150 m away in 0.17 m/s per
    metre) as much as the ranges of the neighbouring rays that give the spreading
    (:meth:`Rays._ray`) differ by where the ray turns within millimetres of the
    listener's or the source's height."""
    sa = np.where(turns_a, 0.0, _sine(p, ca))
    sb = np.where(turns_b, 0.0, _sine(p, cb))
    gdz = cb - ca
    total = np.maximum(sa + sb, 1e-300)
    width = p * (ca + cb) / total
    dx = width * dz
    e2 = p * gdz * width / (1.0 + sb)
    dt = dz * (_log_ratio(gdz / ca) / ca + _log_ratio(e2) * p * width / (1.0 + sb))
    k = ca * p * width + sa
    ds = dz * k * _arc_ratio(p * gdz * k)
    return dx, dt, ds


def _segment(profile: Profile, p, layer, lo, hi):
    """(..., 3) the distance, time and length of rays with parameter ``p`` over the
    heights between ``lo`` and ``hi`` (lo <= hi) inside ``layer`` that they can reach (p
    c <= 1; nothing where they cannot), arrays broadcast together."""
    z, c, g = profile.z, profile.c, profile.gradient
    layer = np.asarray(layer)
    base, c0, slope = z[layer], c[layer], g[layer]
    p = np.asarray(p, dtype=float)
    lo, hi = np.broadcast_arrays(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    c_lo = c0 + slope * (lo - base)
    limit = 1.0 / np.maximum(p, 1e-300)
    # Where the speed reaches 1 / p the ray turns: only the side of the layer below
    # that speed is reachable (none at all, an infinite height, in a gradient too weak
    # to bring it there).
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        turn = base + (limit - c0) / np.where(slope != 0.0, slope, 1.0)
    rising = slope > 0.0
    falling = slope < 0.0
    # Compared in height, not speed: a piece up to a turning height worked out as
    # ``turn`` is (:class:`_Fan`) turns there, though the speed there may round below
    # 1 / p.
    to_turn = rising & (hi >= turn)
    from_turn = falling & (lo <= turn)
    top = np.where(to_turn, np.minimum(hi, turn), hi)
    bottom = np.where(from_turn, np.maximum(lo, turn), lo)
    open_ = (top > bottom) & ~((slope == 0.0) & (c_lo > limit))
    cb = np.where(to_turn, limit, c0 + slope * (top - base))
    ca = np.where(from_turn, limit, c0 + slope * (bottom - base))
    # A piece running level at both ends, a turn and where p c rounds to 1, is no wider
    # than that rounding in a layer with a gradient: it holds nothing.
    sa = np.where(from_turn, 0.0, _sine(p, ca))
    sb = np.where(to_turn, 0.0, _sine(p, cb))
    open_ &= (slope == 0.0) | (sa + sb > 0.0)
    # Closed pieces: any speeds the ray could have, for the formulas to stay finite.
    ca, cb = np.where(open_, ca, 0.0), np.where(open_, cb, 0.0)
    dz = np.where(open_, top - bottom, 0.0)
    dx, dt, ds = _piece(
        np.where(open_, p, 0.0),
        np.where(open_, ca, 1.0),
        np.where(open_, cb, 1.0),
        dz,
        open_ & from_turn,
        open_ & to_turn,
    )
    return np.where(open_[..., None], np.stack([dx, dt, ds], axis=-1), 0.0)


@dataclass
class _Fan:
    """Rays with parameters ``p`` (P,) from a source at height ``zs``: their cumulative
    distance, time and length from the ground up (blocked heights adding nothing), and
    the heights they stay between."""

    profile: Profile
    p: np.ndarray
    zs: float
    table: np.ndarray = field(init=False)  # (P, N + 1, 3) at the nodes
    lo: np.ndarray = field(init=False)  # (P,) the ground (0) or a lower turning point
    hi: np.ndarray = field(init=False)  # (P,) an upper turning point or inf
    ground: np.ndarray = field(init=False)  # (P,) whether lo is the ground

    def __post_init__(self) -> None:
        profile, p = self.profile, self.p
        z, c, g = profile.z, profile.c, profile.gradient
        layers = np.arange(len(z) - 1)
        pieces = _segment(profile, p[:, None], layers[None], z[None, :-1], z[None, 1:])
        self.table = np.concatenate([np.zeros((len(p), 1, 3)), np.cumsum(pieces, axis=1)], axis=1)
        limit = 1.0 / np.maximum(p, 1e-300)
        blocked = c[None, :] > limit[:, None]  # (P, N + 1) at the nodes
        source = int(profile.layer(self.zs))
        above = blocked[:, source + 1 :]
        has_upper = above.any(axis=1)
        first = source + 1 + np.argmax(above, axis=1)
        under = first - 1
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            upper = z[under] + (limit - c[under]) / g[under]
        self.hi = np.where(has_upper, np.maximum(upper, self.zs), np.inf)
        below = blocked[:, : source + 1]
        has_lower = below.any(axis=1)
        last = source - np.argmax(below[:, ::-1], axis=1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lower = z[last] + (limit - c[last]) / g[last]
        self.lo = np.where(has_lower, np.minimum(lower, self.zs), 0.0)
        self.ground = ~has_lower

    def at(self, z) -> np.ndarray:
        """(P, ..., 3) the cumulative distance, time and length at heights ``z`` (finite,
        broadcast against the rays: (P,) or (P, K))."""
        z = np.asarray(z, dtype=float)
        layer = self.profile.layer(z)
        p = self.p.reshape(self.p.shape + (1,) * (z.ndim - 1)) if z.ndim else self.p
        rows = np.arange(len(self.p)).reshape(p.shape) if z.ndim else np.arange(len(self.p))
        start = self.table[rows, layer]
        return start + _segment(self.profile, p, layer, self.profile.z[layer], z)


#: The ways a ray reaches the listener: (launched up, arrives going up).
_KINDS = ((True, True), (True, False), (False, False), (False, True))


@dataclass
class Branch:
    """One way of reaching the listener: launched up or down, arriving going up or down,
    in its ``cycle``-th pass (each pass up and down between the air above and the ground
    or a lower turning point)."""

    up: bool
    arrives_up: bool
    cycle: int

    @property
    def lower_events(self) -> int:
        """Bounces off the ground or turns low in the air on the way."""
        if self.up:
            return self.cycle
        return self.cycle + (1 if self.arrives_up else 0)

    @property
    def upper_turns(self) -> int:
        if self.up:
            return self.cycle + (0 if self.arrives_up else 1)
        return self.cycle

    def segments(self) -> list[tuple[bool, int]]:
        """(going up, pass) of each stretch of the ray between turns, from the source."""
        out = []
        going_up, index = (True, 0) if self.up else (False, -1)
        last = (self.arrives_up, self.cycle if (self.up or self.arrives_up) else self.cycle - 1)
        while (going_up, index) != last:
            out.append((going_up, index))
            going_up, index = (False, index) if going_up else (True, index + 1)
            if len(out) > 2 * _MAX_CYCLES + 4:
                raise RuntimeError("ray branch never arrives")
        out.append(last)
        return out


def _times(n: int, h: np.ndarray) -> np.ndarray:
    """n h, 0 for n = 0 even where h is infinite (a ray that escapes upwards)."""
    return n * h if n else np.zeros_like(h)


def _heights(fan: _Fan, zr: float):
    """For the rays of ``fan`` and a listener at height ``zr``: (P, 3) the distance, time
    and length from the lowest height they reach to the source (a), to the listener (b)
    and to the highest height they reach (h, inf where they escape upwards); whether
    they turn back high up; whether the listener lies between those heights."""
    lo, hi = fan.lo, fan.hi
    f_lo = fan.at(lo)
    a = fan.at(np.full(len(fan.p), fan.zs)) - f_lo
    b = fan.at(np.full(len(fan.p), zr)) - f_lo
    finite = np.isfinite(hi)
    h = np.where(finite[:, None], fan.at(np.where(finite, hi, fan.zs)) - f_lo, np.inf)
    reach = (zr >= lo - 1e-12) & (zr <= hi + 1e-12)
    return a, b, h, finite, reach


def _phase(fan: _Fan, zr: float, up, arrives_up, cycle, heights=None) -> np.ndarray:
    """(..., P, 3) the distance, time and length from the source at ``fan.zs`` to the
    height ``zr`` along the branches given by ``up``, ``arrives_up`` and ``cycle`` (see
    :class:`Branch`; arrays broadcast against the rays), nan where a ray cannot get
    there that way. Launched up, a ray reaches the listener going up after ``cycle``
    bounces (or low turns) and as many high turns, or going down after one more high
    turn; launched down, going down after ``cycle`` of each, or going up after one more
    bounce."""
    a, b, h, finite, reach = _heights(fan, zr) if heights is None else heights
    up, arrives_up, cycle = np.broadcast_arrays(
        np.asarray(up, dtype=bool), np.asarray(arrives_up, dtype=bool), np.asarray(cycle)
    )
    turns = 2 * cycle + 2 * (up & ~arrives_up)
    straight = np.where(
        up & arrives_up, zr >= fan.zs, np.where(~up & ~arrives_up, zr <= fan.zs, True)
    )
    ok = reach & np.where(turns > 0, finite, straight)
    high = np.where(
        (turns > 0)[..., None], turns[..., None] * np.where(finite[:, None], h, 0.0), 0.0
    )
    out = np.where(up[..., None], -a, a) + np.where(arrives_up[..., None], b, -b) + high
    ok &= np.all(np.isfinite(out), axis=-1)
    return np.where(ok[..., None], out, np.nan)


@dataclass
class _Singular:
    """The parts of the range of the rays along a branch that are singular where they
    run level at the source, the listener, the ground, a corner of the profile or in a
    level layer (:meth:`Rays._singular`)."""

    #: (k, c, g, at): the range holds k s / (p g), s a ray's sine at the speed c; at
    #: "source", "listener" or the height of the ground or the corner, where the part
    #: is 0 for the rays that do not reach it.
    terms: list[tuple[float, float, float, str | float]]
    #: Whether the listener's part is among them: the rays need not reach it.
    listener: bool
    #: Whether the ray bounces off the ground, or could: rays that reach it must too.
    bounce: bool
    #: (k, c, at): the range holds k p c / s through level layers of speed c, k their
    #: height, as many times as the ray passes it; 0 for the rays that do not reach
    #: the height ``at`` in them.
    legs: list[tuple[float, float, float]]


@dataclass
class Eigenrays:
    """The rays from a source to a listener in a stratified atmosphere, along the
    horizontal direction from the source to the listener (the listener ``range`` m
    away)."""

    range: float
    p: np.ndarray  # (n,) ray parameters (s/m)
    branch: list[Branch]
    launch: np.ndarray  # (n,) elevation leaving the source (rad, up positive)
    arrival: np.ndarray  # (n,) elevation of the ray's travel at the listener
    time: np.ndarray  # (n,) s
    length: np.ndarray  # (n,) m along the ray
    spreading: np.ndarray  # (n,) intensity over the source's at 1 m (1/R**2 straight)
    caustics: np.ndarray  # (n,) caustics passed
    ground: np.ndarray  # (n,) bool: the lower events are ground bounces
    bounces: list[np.ndarray]  # horizontal distance from the source of each bounce
    capped: np.ndarray  # (n,) whether the spreading was capped at a caustic

    def __len__(self) -> int:
        return len(self.p)

    @property
    def ground_bounces(self) -> np.ndarray:
        return np.array([len(b) for b in self.bounces], dtype=np.int64)


class Rays:
    """Rays in one profile from a source at height ``zs`` (m above the ground)."""

    def __init__(self, profile: Profile, zs: float, launch_angles: int = _LAUNCH_ANGLES):
        self.profile = profile
        self.zs = float(zs)
        self.cs = float(profile.speed(self.zs))
        u = np.linspace(1.0, 0.0, launch_angles, endpoint=False)[::-1]
        angle = 0.5 * np.pi * u**2  # grazing first, vertical last
        grid = np.cos(angle) / self.cs
        grid[-1] = 0.0
        # The rays that run level at the source and, where the air is faster there, at
        # the ground too: a ray's range turns back on either side of them, so two
        # eigenrays could lie between two of the rays launched, unseen.
        level = [1.0 / self.cs]
        if profile.c[0] > self.cs:
            level.append(1.0 / float(profile.c[0]))
        self.grid = np.unique(np.concatenate([grid, level]))[::-1]
        self._fan = _Fan(profile, self.grid, self.zs)
        # The rays that run level in the profile's level layers and stretches (see
        # _LEVEL_SHARE).
        self._levels = 1.0 / profile.c[(profile.gradient == 0.0) | (profile.trend == 0.0)]
        self._searches: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def eigenrays(
        self, zr: float, distance: float, max_bounces: int, max_time: float = np.inf
    ) -> Eigenrays:
        """The eigenrays to a listener at height ``zr`` ``distance`` m away horizontally
        (> 0) that bounce off the ground ``max_bounces`` times at most and arrive by
        ``max_time``. The last ``_KEPT_SEARCHES`` searches are kept: the paths off the
        open ground, behind a barrier and round its ends often ask the same again. The
        rays returned are shared, so read-only."""
        key = (float(zr), float(distance), int(max_bounces), float(max_time))
        with self._lock:
            found = self._searches.get(key)
            if found is not None:
                self._searches.move_to_end(key)
                return found
        found = self._search(*key)
        for value in vars(found).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        with self._lock:
            self._searches[key] = found
            while len(self._searches) > _KEPT_SEARCHES:
                self._searches.popitem(last=False)
        return found

    def _search(self, zr: float, distance: float, max_bounces: int, max_time: float) -> Eigenrays:
        """:meth:`eigenrays`, searched."""
        fan = self._fan
        heights = _heights(fan, zr)
        kinds = np.array(_KINDS, dtype=bool)
        lows, highs, branches = [], [], []
        edges = []
        for cycle in range(_MAX_CYCLES + 1):
            f = _phase(fan, zr, kinds[:, :1], kinds[:, 1:], cycle, heights)[..., 0] - distance
            finite = np.isfinite(f)
            # Neighbouring rays on either side of the listener's range (where the range
            # jumps between them instead, the refinement finds no root).
            pair = finite[:, :-1] & finite[:, 1:] & (np.sign(f[:, :-1]) != np.sign(f[:, 1:]))
            for kind, k in zip(*np.nonzero(pair), strict=True):
                lows.append(self.grid[k + 1])
                highs.append(self.grid[k])
                branches.append((kind, cycle))
            # Where a branch starts between two rays (the listener at the top or the
            # bottom of one's reach, or the ray escaping), its root may lie between the
            # first ray on it and that start.
            for kind, k in zip(*np.nonzero(finite[:, :-1] != finite[:, 1:]), strict=True):
                inside = k if finite[kind, k] else k + 1
                outside = k + 1 if inside == k else k
                # Unless the range runs far off its trend in the last step, the listener
                # is out of reach there.
                deeper = 2 * inside - outside
                if 0 <= deeper < f.shape[1] and finite[kind, deeper]:
                    # (The ray that runs level in a level layer reaches some 1e300 m.)
                    with np.errstate(over="ignore"):
                        last, step = f[kind, inside], f[kind, inside] - f[kind, deeper]
                        if last * step > 0.0 or abs(last) > _EDGE_REACH * abs(step):
                            continue
                edges.append((kind, cycle, inside, outside, f[kind, inside]))
            if cycle >= 1 and not np.any(f <= 0.0):
                # The rays only go farther pass after pass: once none of a pass's falls
                # short of the listener, none after it does.
                break
        cycle_count = cycle + 1
        found, folds = [], []
        if edges:
            kind, cycle, inside, outside, f_in = (np.array(c) for c in zip(*edges, strict=True))
            up, arrives = kinds[kind, 0], kinds[kind, 1]
            near = self._edge(zr, up, arrives, cycle, self.grid[inside], self.grid[outside])
            f_edge = self._range(zr, near, up, arrives, cycle) - distance
            crossing = np.isfinite(f_edge) & (np.sign(f_edge) != np.sign(f_in))
            for i in np.flatnonzero(crossing):
                a, b = sorted((float(near[i]), float(self.grid[inside[i]])))
                lows.append(a)
                highs.append(b)
                branches.append((int(kind[i]), int(cycle[i])))
            # Two branches that end on the same ray, where it turns at the listener's
            # height (to rounding): one arriving before that turn and one after it.
            ends = _Fan(self.profile, near, self.zs)
            upper = np.abs(ends.hi - zr) <= np.abs(ends.lo - zr)
            turning = np.minimum(np.abs(ends.hi - zr), np.abs(ends.lo - zr)) <= _SAME_TURN_M
            inner = np.nextafter(near, self.grid[inside])
            for i, j in itertools.combinations(np.flatnonzero(turning), 2):
                if near[i] == near[j] and up[i] == up[j] and upper[i] == upper[j]:
                    a = Branch(bool(up[i]), bool(arrives[i]), int(cycle[i]))
                    b = Branch(bool(up[j]), bool(arrives[j]), int(cycle[j]))
                    folds.append((float(near[i]), float(inner[i]), a, b, bool(upper[i]), zr))
        # The same where the ray turns at the source's height, running level there: one
        # launched up and one down, which turns there once less high up (where the
        # gradient above the source is positive) or once more low (where the one below
        # it is negative). The other runs on from the source on its own side, which a
        # level layer there would carry off level.
        below, above = self._sides(self.zs, self.profile.trend)
        p = 1.0 / self.cs
        for upper, turning, other in ((True, above > 0.0, below), (False, below < 0.0, above)):
            if not turning or other == 0.0:
                continue
            for c in range(cycle_count):
                for arrives in (True, False):
                    a = Branch(True, arrives, c)
                    if upper:
                        lowest, highest = a.lower_events, a.upper_turns - 1
                    else:
                        lowest, highest = a.lower_events + 1, a.upper_turns
                    if highest >= 0 and lowest - highest == int(arrives):
                        b = Branch(False, arrives, highest)
                        folds.append((p, float(np.nextafter(p, 0.0)), a, b, upper, self.zs))
        found += [(q, b, True) for q, b in self._folds(zr, distance, folds)]
        if branches:
            kind, cycle = (np.array(c) for c in zip(*branches, strict=True))
            roots, steady = self._bisect(
                zr, distance, kinds[kind, 0], kinds[kind, 1], cycle, np.array(lows), np.array(highs)
            )
            for i, root in enumerate(roots):
                if np.isfinite(root):
                    branch = Branch(bool(kinds[kind[i], 0]), bool(kinds[kind[i], 1]), int(cycle[i]))
                    found.append((float(root), branch, bool(steady[i])))
        straight = None
        if self._level_between(zr):
            # Through level air the direct ray is straight: in closed form, where the
            # search could not tell its p from the next float's when it runs almost
            # level (the range then changes by more than the tolerance between them).
            found = [(q, b, f) for q, b, f in found if b.cycle or b.up != b.arrives_up]
            straight = self._straight(zr, distance)
        return self._describe(zr, distance, found, max_bounces, max_time, straight)

    def _sides(self, height: float, g: np.ndarray) -> tuple[float, float]:
        """The gradients ``g`` (per layer: the profile's own or its trend) just below and
        just above ``height`` (the same inside a layer; on the ground, the one above for
        both, as off a mirror)."""
        j = int(self.profile.layer(height))
        below = g[j - 1] if j > 0 and height == self.profile.z[j] else g[j]
        return float(below), float(g[j])

    def _level_between(self, zr: float) -> bool:
        """Whether the profile is level (its speed the same) at every height between the
        source's and ``zr``, or, where they are the same, on one side of it."""
        low, high = min(self.zs, zr), max(self.zs, zr)
        if low == high:
            return 0.0 in self._sides(low, self.profile.gradient)
        first = int(self.profile.layer(low))
        last = int(np.searchsorted(self.profile.z, high, side="left")) - 1
        return bool(np.all(self.profile.gradient[first : last + 1] == 0.0))

    def _straight(self, zr: float, distance: float) -> dict:
        """The straight direct ray through level air to a listener at height ``zr``
        ``distance`` m away (:meth:`_ray`'s description)."""
        rise = zr - self.zs
        length = math.hypot(distance, rise)
        angle = math.atan2(rise, distance)
        up = rise >= 0.0
        return {
            "p": distance / (length * self.cs),
            "branch": Branch(up, up, 0),
            "launch": angle,
            "arrival": angle,
            "time": length / self.cs,
            "length": length,
            "spreading": 1.0 / length**2,
            "caustics": 0,
            "ground": False,
            "bounces": np.zeros(0),
            "capped": False,
        }

    def _edge(self, zr, up, arrives_up, cycle, inside, outside) -> np.ndarray:
        """The ray parameters closest to ``outside`` from ``inside`` (arrays) for which
        the branches still reach the listener."""
        for _ in range(_BISECTIONS):
            if np.all(np.abs(outside - inside) <= np.spacing(np.maximum(inside, outside))):
                break
            middle = 0.5 * (inside + outside)
            ok = np.isfinite(self._range(zr, middle, up, arrives_up, cycle))
            inside = np.where(ok, middle, inside)
            outside = np.where(ok, outside, middle)
        return inside

    def _folds(self, zr, distance, folds) -> list[tuple[float, Branch]]:
        """The eigenrays at folds (p, inner, a, b, upper, end): branches ``a`` and ``b``
        that end on the same ray, with parameter p, where it turns at the height ``end``
        of the source or the listener (high up if ``upper``, else low), if one turns
        there once more than the other. Their ranges run on from one into the other
        there: where they cross the listener's between ``inner``, the next float of p
        towards the rays they hold, and p, the eigenray is the one of those rays that
        comes nearest. The range may change there by more than the tolerance from one
        float to the next, as the square root of how far p is from grazing the end, but
        by no more than :meth:`_graze` allows, and never by ``_GRAZE_SHARE`` of the
        distance: a bigger step is no fold (a branch that runs off level in a level layer
        instead, its range without bound). Only where p lies within ``_GRAZE_FLOATS``
        floats of the ray that runs level at that end (:meth:`_grazing`)."""
        pairs, ends = [], []
        for p, inner, a, b, upper, end in folds:
            turns = [(br.upper_turns, br.lower_events) for br in (a, b)]
            if upper:
                once = turns[0][1] == turns[1][1] and abs(turns[0][0] - turns[1][0]) == 1
            else:
                once = turns[0][0] == turns[1][0] and abs(turns[0][1] - turns[1][1]) == 1
            if once:
                pairs += [(inner, a), (p, a), (p, b), (inner, b)]
                ends.append((p, inner, max(a.cycle, b.cycle), end))
        if not pairs:
            return []
        ps = np.array([q for q, _ in pairs])
        up, arrives, cycle = (
            np.array(c)
            for c in zip(*((b.up, b.arrives_up, b.cycle) for _, b in pairs), strict=True)
        )
        f = (self._range(zr, ps, up, arrives, cycle) - distance).reshape(-1, 4)
        found = []
        for n, row in enumerate(f):
            p, inner, most, end = ends[n]
            if not self._grazing(np.array(p), (end,)):
                continue
            bound = _RANGE_TOLERANCE * (1.0 + distance)
            bound += min(float(self._graze(zr, p, inner, most, end)), _GRAZE_SHARE * distance)
            if not np.all(np.isfinite(row)):
                continue
            sign = np.signbit(row) & (row != 0.0)
            with np.errstate(over="ignore"):
                steps = np.abs(np.diff(row))
            if np.any(sign[:-1] != sign[1:]) and np.all(steps <= bound):
                found.append(pairs[4 * n + int(np.argmin(np.abs(row)))])
        return [(float(q), b) for q, b in found]

    def _graze(self, zr, p, q, cycle, fold=None) -> np.ndarray:
        """The most the range of rays along branches of ``cycle`` passes can change from
        the rays with parameter ``p`` to those with ``q`` (neighbouring floats, arrays)
        by its parts singular where they graze the source's or the listener's height: k
        |s(p) - s(q)| / (p g) at either, s their sines there, g the profile's trend there
        (:attr:`Profile.trend`; the gentler side's at a node, none where it is level),
        k the most times a ray can reach that height or turn at it. At the end ``fold``
        (its height), where the rays turn, no less than the change over one float of p
        from grazing it. Without bound as g nears 0: its callers cap it at
        ``_GRAZE_SHARE`` of the distance."""
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        k = 2.0 * (2.0 * np.asarray(cycle, dtype=float) + 3.0)
        out = np.zeros(np.broadcast(p, q, k).shape)
        flat = np.minimum(p, q)
        for height in (self.zs, zr):
            slopes = [abs(g) for g in self._sides(height, self.profile.trend) if g != 0.0]
            if not slopes:
                continue
            c = float(self.profile.speed(height))
            change = np.abs(_sine(p, c) - _sine(q, c))
            if fold is not None and height == fold:
                change = np.maximum(change, np.sqrt(2.0 * np.spacing(flat) / flat))
            out = out + k * change / (flat * min(slopes))
        return out

    def _grazing(self, p, heights) -> np.ndarray:
        """Whether rays with parameters ``p`` (an array) lie within ``_GRAZE_FLOATS``
        floats of p of a ray that runs level at one of ``heights`` where the profile's
        trend is not level on both sides of it (:meth:`_graze`)."""
        out = np.zeros(np.shape(p), dtype=bool)
        for height in heights:
            if any(g != 0.0 for g in self._sides(height, self.profile.trend)):
                level = 1.0 / float(self.profile.speed(height))
                out |= np.abs(p - level) <= _GRAZE_FLOATS * np.spacing(p)
        return out

    def _bisect(self, zr, distance, up, arrives_up, cycle, lo, hi):
        """The ray parameters where the branches (``up``, ``arrives_up``, ``cycle``)
        reach ``distance``, each between ``lo`` and ``hi`` (nan where the range jumps
        there instead), and whether each was taken at a steep crossing, missing by more
        than the tolerance."""
        f_lo = self._range(zr, lo, up, arrives_up, cycle) - distance
        f_hi = self._range(zr, hi, up, arrives_up, cycle) - distance
        # The Illinois method: false position, halving the weight of an end that stays.
        side = np.zeros(len(lo), dtype=np.int64)
        for step in range(_BISECTIONS):
            with np.errstate(invalid="ignore", divide="ignore"):
                guess = (lo * f_hi - hi * f_lo) / (f_hi - f_lo)
            # Halve where false position stalls or leaves the bracket.
            bad = ~np.isfinite(guess) | (guess <= lo) | (guess >= hi) | (step % 8 == 7)
            mid = np.where(bad, 0.5 * (lo + hi), guess)
            f = self._range(zr, mid, up, arrives_up, cycle) - distance
            low = np.sign(f) == np.sign(f_lo)
            f_hi = np.where(low & (side == 1), 0.5 * f_hi, f_hi)
            f_lo = np.where(~low & (side == -1), 0.5 * f_lo, f_lo)
            lo, f_lo = np.where(low, mid, lo), np.where(low, f, f_lo)
            hi, f_hi = np.where(low, hi, mid), np.where(low, f_hi, f)
            side = np.where(low, 1, -1)
            if np.all((np.abs(f) <= 1e-12 * (1.0 + distance)) | (hi - lo <= np.spacing(hi))):
                break
        fans = [_Fan(self.profile, q, self.zs) for q in (lo, hi)]
        f_lo, f_hi = (_phase(fan, zr, up, arrives_up, cycle)[:, 0] - distance for fan in fans)
        nearer = np.abs(f_lo) <= np.abs(f_hi)
        root = np.where(nearer, lo, hi)
        miss = np.where(nearer, np.abs(f_lo), np.abs(f_hi))
        # A jump of the range is no crossing. But near where a ray grazes an end in a
        # weak gradient the range may change by more than the tolerance from one float
        # of p to the next (as the square root of how far p is from there): a crossing
        # between neighbouring floats whose rays turn at the same heights is one where
        # it misses by no more than its parts singular there change (:meth:`_graze`),
        # by no more than _GRAZE_SHARE of the distance, and within _GRAZE_FLOATS floats
        # of p of the ray that runs level at an end (:meth:`_grazing`).
        tolerance = _RANGE_TOLERANCE * (1.0 + distance)
        steady = (hi - lo <= np.spacing(hi)) & (fans[0].ground == fans[1].ground)
        for a, b in ((fans[0].lo, fans[1].lo), (fans[0].hi, fans[1].hi)):
            with np.errstate(invalid="ignore"):
                steady &= (a == b) | (np.abs(a - b) <= _SAME_TURN_M)
        steady &= miss <= tolerance + np.minimum(
            self._graze(zr, lo, hi, cycle), _GRAZE_SHARE * distance
        )
        steady &= self._grazing(root, (self.zs, zr))
        found = miss <= tolerance
        return np.where(found | steady, root, np.nan), steady & ~found

    def _range(self, zr, p, up, arrives_up, cycle) -> np.ndarray:
        fan = _Fan(self.profile, np.asarray(p, dtype=float), self.zs)
        return _phase(fan, zr, up, arrives_up, cycle)[:, 0]

    def _describe(self, zr, distance, found, max_bounces, max_time, straight=None) -> Eigenrays:
        rows = []
        for p, branch, steady in found:
            rows.append(self._ray(zr, distance, p, branch, steady))
        rows = [r for r in rows if r is not None]
        if straight is not None:
            rows.append(straight)
        rows = [r for r in rows if r["time"] <= max_time]
        rows = [r for r in rows if not r["ground"] or len(r["bounces"]) <= max_bounces]
        # The same ray found from two branches that meet (where its lower event turns
        # from a bounce to a turn in the air): keep one.
        rows.sort(key=lambda r: r["time"])
        kept = []
        for r in rows:
            same = [
                k
                for k in kept
                if abs(k["time"] - r["time"]) <= 1e-9 * (1.0 + r["time"])
                and abs(k["launch"] - r["launch"]) <= 1e-7
            ]
            if not same:
                kept.append(r)
        return Eigenrays(
            range=distance,
            p=np.array([r["p"] for r in kept]),
            branch=[r["branch"] for r in kept],
            launch=np.array([r["launch"] for r in kept]),
            arrival=np.array([r["arrival"] for r in kept]),
            time=np.array([r["time"] for r in kept]),
            length=np.array([r["length"] for r in kept]),
            spreading=np.array([r["spreading"] for r in kept]),
            caustics=np.array([r["caustics"] for r in kept], dtype=np.int64),
            ground=np.array([r["ground"] for r in kept], dtype=bool),
            bounces=[r["bounces"] for r in kept],
            capped=np.array([r["capped"] for r in kept], dtype=bool),
        )

    def _ray(self, zr, distance, p, branch, steady=False) -> dict | None:
        # Rays a step either side, small against how far the ray is from grazing at
        # either end, where its range changes fastest.
        top = 1.0 / max(self.cs, float(self.profile.speed(zr)))
        step = 1e-4 * max(min(p, top - p), 1e-12 * top)
        ps = np.clip(np.array([p, p - step, p + step]), 0.0, 1.0 / self.cs)
        fan = _Fan(self.profile, ps, self.zs)
        out = _phase(fan, zr, branch.up, branch.arrives_up, branch.cycle)
        if not np.isfinite(out[0, 0]):
            return None
        x, t, s = out[0]
        ground = bool(fan.ground[0])
        turns = []
        if branch.upper_turns:
            turns.append(float(fan.hi[0]))
        if branch.lower_events and not ground:
            turns.append(float(fan.lo[0]))
        sin_s = float(_sine(p, self.cs))
        sin_r = float(_sine(p, float(self.profile.speed(zr))))
        # The width of the tube at which its level is capped.
        capped = p * self.cs**2 * float(s) ** 2 / (CAUSTIC_CAP * distance)
        # The tube's width, |dr/dp| sin_s sin_r, from the range itself and from the range
        # less the parts of it that are singular where the ray grazes an end or the
        # ground (a square root of how far it is from grazing there, which no
        # difference of neighbouring rays follows), those parts' slopes added back.
        raw = self._raw_slope(zr, p, branch, turns, out, ps)
        parts = self._singular(zr, p, branch, ground, float(fan.lo[0]), float(fan.hi[0]))
        new = self._width(zr, distance, p, branch, turns, parts, fan, float(s), capped)
        # A ray that turns in the air whose range itself could not be fitted over the
        # rays that turn about it (flatter ones would not leave the source or reach the
        # listener) has the slope of the sampled profile's range: off the smooth one's
        # where its turning point is near one of the nodes it is sampled at.
        sampled = bool(turns) and raw is not None and raw[3] is None and self._sampled(turns)
        chosen = self._choose(new, raw, sin_s * sin_r, capped, steady, sampled)
        if chosen is None:
            return None
        if chosen is new:
            tube = distance * abs(chosen[0])
        else:
            tube = distance * abs(chosen[0]) * sin_s * sin_r
        launch = math.copysign(math.acos(min(p * self.cs, 1.0)), 1.0 if branch.up else -1.0)
        arrival = math.copysign(
            math.acos(min(p * float(self.profile.speed(zr)), 1.0)),
            1.0 if branch.arrives_up else -1.0,
        )
        free = 1.0 / max(s, 1e-12) ** 2
        spreading = p * self.cs**2 / tube if tube > 0.0 else np.inf
        capped = not spreading <= CAUSTIC_CAP * free
        if capped:
            spreading = CAUSTIC_CAP * free
        bounces = np.zeros(0)
        if ground and branch.lower_events:
            bounces = self._events(fan, branch)
        return {
            "p": float(p),
            "branch": branch,
            "launch": launch,
            "arrival": arrival,
            "time": float(t),
            "length": float(s),
            "spreading": float(spreading),
            "caustics": self._caustics(fan, zr, branch),
            "ground": ground,
            "bounces": bounces,
            "capped": capped,
        }

    @staticmethod
    def _choose(new, raw, sines, capped, steady, sampled):
        """The one of two the tube of a ray takes: ``new``, (its width from its range less
        its singular parts, the same from rays half as far apart: a callable), and
        ``raw``, (the slope of its range itself, the same from rays half as far apart: a
        callable, how many floats of p apart its rays are, and where it is fitted over
        rays that turn about the ray, the same three from the rays a small step either
        side, else None), a width once times ``sines``,
        the ray's sines at both ends multiplied; None where either cannot be had. How
        much each changes from rays half as far apart stands for its error, as a share
        of the width (or of ``capped``, the width at which the level is capped, where
        that is more). Where they agree to ``_AGREEMENT``, the one that changes less;
        else ``new`` only where it changes by no more than ``_STABLE`` and a tenth of how
        far they differ and ``raw`` by more than ``_AGREEMENT``; else ``raw``. But,
        unless they agree, ``raw`` only where it changes by no more than ``_AGREEMENT``
        where the ray was taken at a fold or a steep crossing (``steady``) or its rays
        lie ``_FEW_FLOATS`` floats of p apart or fewer, else by no more than ``_ALONE``
        where there is no ``new``, there from the rays a small step either side where
        those lie more than ``_FEW_FLOATS`` floats apart and the fitted slope differs
        from theirs by more than ``_ALONE``; and ``new`` alone only where it changes by
        no more than ``_STABLE``. Where ``raw`` is the sampled profile's own slope about
        a node of it (``sampled``, :meth:`_sampled`) and they do not agree, ``new``
        where it changes by no more than ``_ALONE``, as one taken alone. None where
        neither can be taken."""

        def change(estimate, scale=1.0):
            again = estimate[1]()
            if again is None or not np.isfinite(again):
                return np.inf
            return abs(again - estimate[0]) * scale

        if new is not None and raw is None:
            return new if change(new) <= _STABLE * max(abs(new[0]), capped) else None
        if new is not None:
            width = raw[0] * sines
            gap = abs(new[0] - width)
            moved, shaky = change(new), change(raw, sines)
            if gap <= _AGREEMENT * max(abs(width), abs(new[0]), capped):
                return new if moved < shaky else raw
            if moved <= min(0.1 * gap, _STABLE * max(abs(new[0]), capped)) and (
                shaky > _AGREEMENT * max(abs(width), capped)
            ):
                return new
            if sampled and moved <= _ALONE * max(abs(new[0]), capped):
                return new
        if raw is None:
            return None
        if steady or raw[2] <= _FEW_FLOATS:
            bound = _AGREEMENT
        elif new is None:
            bound = _ALONE
            local = raw[3]
            if local is not None and local[2] > _FEW_FLOATS:
                if not abs(raw[0] - local[0]) * sines <= bound * max(abs(raw[0] * sines), capped):
                    raw = local
        else:
            return raw
        if not change(raw, sines) <= bound * max(abs(raw[0] * sines), capped):
            return None
        return raw

    def _raw_slope(self, zr, p, branch, turns, out, ps):
        """The slope in p of the range itself of the ray with parameter ``p`` along
        ``branch`` (``out`` its distance, time and length and those of the rays ``ps`` a
        step either side): where it turns in the air at the heights ``turns``, fitted
        over the rays whose turning points lie ``_TURN_LAYERS`` layers about its own, or
        over half, a quarter or an eighth of that span, the first over which they all
        reach the listener; else from the rays ``ps`` either side. That slope, the same
        from rays half as far apart (a callable, None from it where they would lie fewer
        than ``_HALF_FLOATS`` floats of p apart), how many floats of p apart the rays
        are, and for a fitted slope the same three from the rays ``ps`` (else None); None
        where the rays either side do not reach the listener."""
        if not np.all(np.isfinite(out[:, 0])):
            return None
        top = 1.0 / max(self.cs, float(self.profile.speed(zr)))
        span = self._turn_span(p, turns) if turns else 0.0
        for _ in range(4):
            if span <= 0.0:
                break
            slope = self._raw_fit(zr, p, branch, span, top)
            if slope is not None and span < _HALF_FLOATS * _TURN_RAYS * float(np.spacing(p)):
                return (
                    slope,
                    lambda: None,
                    span / float(np.spacing(p)),
                    self._local(zr, p, branch, out, ps),
                )
            if slope is not None:
                return (
                    slope,
                    lambda span=span: self._raw_fit(zr, p, branch, 0.5 * span, top),
                    (span / float(np.spacing(p))),
                    self._local(zr, p, branch, out, ps),
                )
            span *= 0.5
        return (*self._local(zr, p, branch, out, ps), None)

    def _local(self, zr, p, branch, out, ps):
        """The slope in p of the range of the ray with parameter ``p`` along ``branch``
        from the rays ``ps`` a step either side (``out`` their distances), the same from
        rays half as far apart (a callable, None from it where they would lie fewer than
        ``_HALF_FLOATS`` floats of p apart) and how many floats of p apart the rays are."""
        h = 0.5 * float(ps[2] - ps[1])

        def half_step():
            if h < _HALF_FLOATS * float(np.spacing(p)):
                return None
            qs = np.clip(np.array([p - 0.5 * h, p + 0.5 * h]), 0.0, 1.0 / self.cs)
            r = self._range(zr, qs, branch.up, branch.arrives_up, branch.cycle)
            with np.errstate(divide="ignore", invalid="ignore"):
                return float((r[1] - r[0]) / (qs[1] - qs[0]))

        local = (out[2, 0] - out[1, 0]) / (ps[2] - ps[1])
        return float(local), half_step, h / float(np.spacing(p))

    def _singular(self, zr, p, branch, ground, lo, hi) -> _Singular:
        """The parts of the range of rays along ``branch`` that are singular where they
        graze the source's height, the listener's, the ground (if the ray with parameter
        ``p`` bounces off it, ``ground``, or could: too flat to reach it) and the
        profile's corners. In a layer of gradient g the distance up from a speed ca to
        cb is (sa - sb) / (p g), sa and sb the sines there, so the range holds k s / (p
        g) of each such height, s a ray's sine there: g the profile's trend there
        (:attr:`Profile.trend`), that of the stretch the ray goes through there, the one
        above or below it at a node, as the ray leaves or reaches it; at a corner the ray
        passes, 1 / g that of the stretch above less that of the one below, each time it
        passes. A level stretch adds none of these. Through each layer whose speed does
        not change at all with height, between the heights ``lo`` and ``hi`` the ray
        stays between, it goes p c dz / s (dz how high, as many times as it passes it),
        which grows without bound as p nears 1 / c (see ``_LEVEL_SHARE``)."""
        z, c, trend = self.profile.z, self.profile.c, self.profile.trend
        terms = []
        listener = False
        for end, height, upward, k in (
            ("source", self.zs, branch.up, 1.0 if branch.up else -1.0),
            ("listener", zr, not branch.arrives_up, 1.0 - 2.0 * branch.arrives_up),
        ):
            j = int(self.profile.layer(height))
            j -= 1 if j > 0 and height == z[j] and not upward else 0
            speed = float(self.profile.speed(height))
            if trend[j] != 0.0:
                terms.append((k, speed, float(trend[j]), end))
                listener = listener or end == "listener"
        flat = _sine(p, c[0]) == 0.0 and trend[0] != 0.0
        bounce = bool(branch.lower_events) and (ground or flat)
        if bounce and trend[0] != 0.0:
            # Each way down to the ground and back up covers s0 / (p g) of the lowest
            # layer: 0 for a ray that turns before it, as for one that turns in it.
            terms.append((2.0 * branch.lower_events, float(c[0]), float(trend[0]), 0.0))
        for node in self.profile.corners:
            height = float(z[node])
            below, above = trend[node - 1], trend[node]
            inverse = (1.0 / above if above else 0.0) - (1.0 / below if below else 0.0)
            passes = self._passes(zr, branch, height)
            if passes and inverse:
                terms.append((float(passes), float(c[node]), 1.0 / inverse, height))
        legs = []
        for j in np.flatnonzero(self.profile.gradient == 0.0):
            bottom = max(float(z[j]), lo)
            top = min(float(z[j + 1]) if j + 1 < len(z) else np.inf, hi)
            if top <= bottom:
                continue
            cuts = sorted({bottom, top} | {h for h in (self.zs, zr) if bottom < h < top})
            for a, b in itertools.pairwise(cuts):
                middle = 0.5 * (a + b)
                passes = self._passes(zr, branch, middle) if np.isfinite(b) else 0
                if passes:
                    legs.append((passes * (b - a), float(c[j]), middle))
        return _Singular(terms, listener, bounce, legs)

    def _rounding(self) -> float:
        """How far apart two of the profile's speeds may lie and be the same to rounding
        (``_ROUNDING_ULPS``)."""
        return _ROUNDING_ULPS * float(np.spacing(np.max(np.abs(self.profile.c))))

    def _passes(self, zr, branch, height) -> int:
        """How many times a ray along ``branch`` that reaches ``height`` passes it, its
        turns lying beyond it (leaving or reaching it at the source or the listener does
        not count)."""
        segments = branch.segments()
        if len(segments) == 1:
            return int(min(self.zs, zr) < height < max(self.zs, zr))
        first = height > self.zs if segments[0][0] else height < self.zs
        last = height < zr if segments[-1][0] else height > zr
        return len(segments) - 2 + int(first) + int(last)

    def _keeps(self, fan: _Fan, branch: Branch, parts: _Singular) -> bool:
        """Whether the rays of ``fan`` along ``branch`` hold the singular ``parts`` of the
        ray they were spread about: those that reach the ground (their sine there not 0)
        bounce off it if it does, and not if it does not."""
        if not branch.lower_events:
            return True
        reaching = _sine(fan.p, self.profile.c[0]) > 0.0
        return bool(np.all(fan.ground[reaching] == parts.bounce))

    def _smooth_range(self, zr, ps, branch, parts: _Singular):
        """The ranges of rays with parameters ``ps`` along ``branch`` less their singular
        ``parts`` (:meth:`_singular`), and their fan. The rays need not leave the
        source or reach the listener where it holds a part: the fan stops them where
        they turn, where that part is 0, which carries on the rest smoothly."""
        fan = _Fan(self.profile, ps, self.zs)
        a, b, h, finite, reach = _heights(fan, zr)
        if parts.listener:
            reach = np.ones_like(reach)
        r = _phase(fan, zr, branch.up, branch.arrives_up, branch.cycle, (a, b, h, finite, reach))
        r = r[:, 0]
        for k, c, g, at in parts.terms:
            part = k * _sine(fan.p, c) / (fan.p * g)
            if not isinstance(at, str):
                part = np.where((fan.lo <= at) & (at <= fan.hi), part, 0.0)
            r = r - part
        for k, c, at in parts.legs:
            sine = _sine(fan.p, c)
            reached = (fan.lo <= at) & (at <= fan.hi) & (sine > 0.0)
            r = r - np.where(reached, k * fan.p * c / np.where(reached, sine, 1.0), 0.0)
        return r, fan

    def _width(self, zr, distance, p, branch, turns, parts, fan, length, capped):
        """The width of the tube of the ray with parameter ``p`` along ``branch`` (the
        first of ``fan``, ``length`` m long), which turns in the air at the heights
        ``turns``, and the same from rays half as far apart (a callable): the slope of
        its range less its singular ``parts``, times its sines at both ends, those parts'
        slopes added back. The slope is fitted (:meth:`_turning_slope`) where the ray
        turns in the air, else, or where no fit can be had, from rays a small step either
        side (:meth:`_local_width`). Where the rays it would be fitted over would pass
        the ray that leaves the source level, it comes from rays a small step either side
        first, and from the fit (over those rays slid down in p) only where they give
        none, if at each gentle corner they cross (:meth:`_step`) the speed is at least
        that at either end and the ray's sine at least 0.9 of the lesser of its own at
        its ends. None where no rays about it keep to its branch, or where the rounding
        of the parts could move the width by more than ``_WIDTH_ROUNDING`` of it (or of
        the width at which its level is capped): parts that dwarf the range, in a weak
        gradient, or where rays a step and a quarter step either side disagree
        (:meth:`_local_width`); the tube then comes from the range itself
        (:meth:`_choose`)."""
        gap = float(np.min(np.abs(self._levels - p), initial=np.inf))
        sin_s = float(_sine(p, self.cs))
        sin_r = float(_sine(p, float(self.profile.speed(zr))))
        near = bool(turns) and p + self._turn_span(p, turns) > 1.0 / self.cs
        if near:
            # Rays a small step either side follow a gentle corner as it is, where the
            # fit spreads it as over the smooth profile the corners are sampled from;
            # but the fit, slid to one side of the ray, does not follow one just past
            # both its ends on the way to its turn, where it runs nearly as level as
            # there.
            _, gentle, _ = self._step(zr, p, branch, parts)
            c = self.profile.c[gentle]
            past = c >= max(self.cs, float(self.profile.speed(zr))) - self._rounding()
            near = bool(np.all(past & (_sine(p, c) >= 0.9 * min(sin_s, sin_r))))

        def local():
            return self._local_width(zr, distance, p, branch, parts, fan, length, capped)

        def back(fit):
            return (
                None if fit is None else self._added_back(zr, distance, p, parts, fan, length, *fit)
            )

        def fitted(fit):
            width = back(fit)
            if width is None:
                return None
            if fit[1] < _HALF_FLOATS * _TURN_RAYS * float(np.spacing(p)):
                return width, lambda: None
            # The same over rays half as far apart.
            return width, lambda: back(
                self._turning_slope(zr, p, branch, turns, parts, gap, 0.5 * fit[1], 1)
            )

        if near:
            ways = [local, lambda: fitted(self._turning_slope(zr, p, branch, turns, parts, gap))]
        else:
            fit = self._turning_slope(zr, p, branch, turns, parts, gap) if turns else None
            ways = [local if fit is None else lambda: fitted(fit)]
        for way in ways:
            width = way()
            if width is not None:
                return width
        return None

    def _local_width(self, zr, distance, p, branch, parts, fan, length, capped):
        """The width of the tube of the ray with parameter ``p`` (:meth:`_width`) from
        rays a small step either side (:meth:`_local_slope`), and the same from rays half
        as far (a callable, None from it where they would lie fewer than
        ``_HALF_FLOATS`` floats of p apart); where the layers they cross depart from
        their trend (:meth:`_crossed`), or they lie fewer than ``_FEW_FLOATS`` floats
        from it, only where rays a quarter of that step either side give it to
        ``_LOOSE_AGREEMENT`` of it (or of ``capped``, the width at which its level is
        capped). None where it cannot be had."""
        step, _, loose = self._step(zr, p, branch, parts)
        few = step < _FEW_FLOATS * float(np.spacing(p))

        def at(h):
            fit = self._local_slope(zr, p, branch, parts, h)
            return (
                None if fit is None else self._added_back(zr, distance, p, parts, fan, length, *fit)
            )

        width = at(step)
        if width is None:
            return None
        if loose or few:
            quarter = at(0.25 * step)
            if quarter is None:
                return None
            if abs(quarter - width) > _LOOSE_AGREEMENT * max(abs(width), capped):
                return None
        if step < _HALF_FLOATS * float(np.spacing(p)):
            return width, lambda: None
        return width, lambda: at(0.5 * step)

    def _local_slope(self, zr, p, branch, parts: _Singular, h):
        """The slope in p of the range of a ray with parameter ``p`` along ``branch``
        less its singular ``parts``, from rays a step ``h`` either side, and how far in p
        they lie; None where they do not keep to its branch."""
        if not h > 0.0:
            return None
        # Central, or from the steeper side where the flatter ray could not leave the
        # source (p c_s > 1).
        central = p + h <= 1.0 / self.cs
        steps = np.array([-1.0, 1.0]) if central else np.array([-2.0, -1.0, 0.0])
        # (As the floats they round to: a step may be only a few of them.)
        ps = p + h * steps
        smooth, pair = self._smooth_range(zr, ps, branch, parts)
        if not (
            ps[-1] > ps[0] and np.all(np.isfinite(smooth)) and self._keeps(pair, branch, parts)
        ):
            return None
        if central:
            return (smooth[1] - smooth[0]) / (ps[1] - ps[0]), h
        # The derivative at p of the parabola through them.
        return float(np.polyfit((ps - p) / h, smooth, 2)[1] / h), 0.25 * h

    def _added_back(self, zr, distance, p, parts: _Singular, fan, length, slope, span):
        """The width of the tube of the ray with parameter ``p`` (:meth:`_width`) from
        the ``slope`` of its range less its singular ``parts``, found from rays within
        ``span`` in p of it; None where the rounding of the parts could move it by more
        than ``_WIDTH_ROUNDING`` of it."""
        sin_s = float(_sine(p, self.cs))
        sin_r = float(_sine(p, float(self.profile.speed(zr))))
        width = slope * sin_s * sin_r
        lo, hi = float(fan.lo[0]), float(fan.hi[0])
        sizes, backs = distance, 0.0
        for k, c, g, at in parts.terms:
            # A part k s / (p g) has the slope -k / (s p^2 g), s the sine at its height:
            # times sin_s sin_r, the source's or the listener's own sine cancels; 0 where
            # the ray turns just short of the ground or the corner, and so is its slope.
            sine = sin_s if at == "source" else sin_r if at == "listener" else float(_sine(p, c))
            sizes += abs(k * sine / (p * g))
            if at == "source":
                back = k * sin_r / (g * p * p)
            elif at == "listener":
                back = k * sin_s / (g * p * p)
            elif lo <= at <= hi and sine > 0.0:
                back = k * sin_s * sin_r / (sine * p * p * g)
            else:
                back = 0.0
            width -= back
            backs += abs(back)
        for k, c, _ in parts.legs:
            # A leg k p c / s through a level layer has the slope k c / s^3.
            sine = float(_sine(p, c))
            if sine == 0.0:
                return None
            sizes += k * p * c / sine
            back = k * c * sin_s * sin_r / sine**3
            width += back
            backs += abs(back)
        eps = float(np.finfo(float).eps)
        rounding = 64.0 * eps * sizes / span * sin_s * sin_r + 8.0 * eps * backs
        capped = p * self.cs**2 * length**2 / (CAUSTIC_CAP * distance)
        return width if rounding <= _WIDTH_ROUNDING * max(abs(width), capped) else None

    def _crossed(self, zr, p, branch, step):
        """The level layers (a mask over them) and the gentle corners (nodes where the
        trend changes that are not corners, so hold no part) that rays along ``branch``
        up to 2 steps (``_SMOOTH_STEP``) steeper or 1 flatter than the one with
        parameter ``p`` cross on their way to the listener (not those level layers they
        only touch, at an end), and whether the gradient of any other layer they cross
        departs from its trend by more than 1e-6 of it (by the rounding of the speeds at
        its nodes, in a stretch too weak for it)."""
        ps = np.array([max(p - 2.0 * step, 0.5 * p), min(p + step, 1.0 / self.cs)])
        fan = _Fan(self.profile, ps, self.zs)
        lower = float(np.min(fan.lo)) if branch.lower_events else min(self.zs, zr)
        upper = float(np.max(fan.hi)) if branch.upper_turns else max(self.zs, zr)
        z, gradient, trend = self.profile.z, self.profile.gradient, self.profile.trend
        above = np.append(z[1:], np.inf)
        crossed = (z <= upper) & (above >= lower)
        flat = crossed & ((gradient == 0.0) | (trend == 0.0))
        sloped = crossed & ~flat
        loose = np.abs(gradient[sloped] - trend[sloped]) > 1e-6 * np.abs(trend[sloped])
        # A level layer they only touch, at an end, holds none of their range.
        level = flat & (z < upper) & (above > lower)
        gentle = self.profile.gentle
        crossed_gentle = gentle[(z[gentle] >= lower) & (z[gentle] <= upper)]
        return level, crossed_gentle, bool(np.any(loose))

    def _step(self, zr, p, branch, parts: _Singular):
        """The step in p to rays either side of the one with parameter ``p`` along
        ``branch`` (``_SMOOTH_STEP`` of it, or less: 1e-4 of the way to the nearest ray
        that runs level in a level layer they cross, and a tenth of that to the nearest
        that runs level at a gentle corner they cross, at a corner of its singular
        ``parts`` between the heights of its ends, or at the height of any of them where
        the layers they cross depart from their trend), those gentle corners and whether
        they do (:meth:`_crossed`)."""
        step = _SMOOTH_STEP * p
        c = self.profile.c
        # Rays flatter than the one that runs level at a corner between the ends turn
        # short of it, so never reach the far end.
        lower, upper = min(self.zs, zr), max(self.zs, zr)
        between = np.array(
            [
                1.0 / speed - p
                for _, speed, _, at in parts.terms
                if not isinstance(at, str) and lower < at < upper
            ]
        )
        # (Rays a smaller step away cross no more of them.)
        for _ in range(2):
            level, gentle, loose = self._crossed(zr, p, branch, step)
            singular = np.concatenate([1.0 / c[gentle] - p, between[between != 0.0]])
            if loose:
                # The parts, from the trend, then leave some of the range's own about
                # where the ray runs level there.
                ends = np.array([1.0 / speed - p for _, speed, _, _ in parts.terms])
                singular = np.concatenate([singular, ends[ends != 0.0]])
            room = min(
                1e-4 * float(np.min(np.abs(1.0 / c[level] - p), initial=np.inf)),
                0.1 * float(np.min(np.abs(singular), initial=np.inf)),
            )
            if room >= step:
                break
            step = room
        return step, gentle, loose

    def _turn_span(self, p, turns) -> float:
        """The span of p over which the turning points of rays about the one with
        parameter ``p`` that turns at the heights ``turns`` move ``_TURN_LAYERS`` of the
        profile's layers either way."""
        z, g = self.profile.z, self.profile.gradient
        width = 0.0
        for height in turns:
            j = int(self.profile.layer(height))
            if j < len(z) - 1:
                # The turning height moves by c / (p g) per unit of p, c being 1 / p.
                width = max(width, _TURN_LAYERS * (z[j + 1] - z[j]) * p * p * abs(g[j]))
        return width

    def _sampled(self, turns) -> bool:
        """Whether a gentle corner of the profile (:attr:`Profile.gentle`) lies within
        ``_TURN_LAYERS`` layers of any of the heights ``turns``, counted in nodes: one
        the turning points of the rays a turning ray's tube is fitted over pass
        (:meth:`_turn_span`), where the sampled profile's range wiggles about the smooth
        one's. A table's straight stretch has none: the profile is linear between its
        rows."""
        reach = math.ceil(_TURN_LAYERS)
        gentle = self.profile.gentle
        for height in turns:
            j = int(self.profile.layer(height))
            if np.any((gentle > j - reach) & (gentle <= j + reach)):
                return True
        return False

    def _turning_slope(self, zr, p, branch, turns, parts: _Singular, gap, span=None, tries=None):
        """The slope in p of the range of a ray with parameter ``p`` along ``branch``
        that turns in the air at the heights ``turns``, less its singular ``parts``, and
        the half-width in p of the rays it is fitted over: those whose turning points lie
        ``_TURN_LAYERS`` layers about its own, or fewer, as many as keep to its branch
        (:meth:`_keeps`) and to ``_LEVEL_SHARE`` of the ``gap`` in p to the nearest ray
        that runs level in a level layer, and slid down in p where they would pass the
        ray that leaves the source level; None where none do."""
        width = self._turn_span(p, turns) if span is None else span
        top = 1.0 / self.cs
        for _ in range(_TURN_HALVINGS if tries is None else tries):
            if width <= 0.0:
                break
            ps = p + width * np.linspace(-1.0, 1.0, _TURN_RAYS)
            # None flatter than the ray that leaves the source level.
            ps = np.minimum(ps - max(ps[-1] - top, 0.0), top)
            if ps[0] > 0.0 and width <= _LEVEL_SHARE * gap:
                r, fan = self._smooth_range(zr, ps, branch, parts)
                if np.all(np.isfinite(r)) and self._keeps(fan, branch, parts):
                    # The derivative at p of a parabola fitted through them.
                    return float(np.polyfit((ps - p) / width, r, 2)[1] / width), width
            width *= 0.5
        return None

    def _raw_fit(self, zr, p, branch, span, top) -> float | None:
        """The slope in p of the range itself of the ray with parameter ``p`` along
        ``branch``, fitted over rays up to ``span`` either side of it, none flatter than
        ``top``; None where they do not all reach the listener."""
        ps = p + span * np.linspace(-1.0, 1.0, _TURN_RAYS)
        if ps[0] > 0.0 and ps[-1] <= top:
            r = self._range(zr, ps, branch.up, branch.arrives_up, branch.cycle)
            if np.all(np.isfinite(r)):
                # The derivative at the middle of a parabola fitted through them.
                return float(np.polyfit((ps - p) / span, r, 2)[1] / span)
        return None

    def _events(self, fan: _Fan, branch: Branch) -> np.ndarray:
        """Horizontal distances from the source of the lower events of the first ray of
        ``fan`` along ``branch``."""
        f_lo = fan.at(fan.lo)[0, 0]
        a = fan.at(np.full(len(fan.p), self.zs))[0, 0] - f_lo
        h = fan.at(np.where(np.isfinite(fan.hi), fan.hi, self.zs))[0, 0] - f_lo
        start = a if branch.up else -a
        out = []
        for going_up, index in branch.segments()[:-1]:
            if not going_up:
                out.append(_times(2 * index + 2, h) - start)
        return np.array(out, dtype=float)

    def _caustics(self, fan: _Fan, zr: float, branch: Branch) -> int:
        """How many caustics the first ray of ``fan`` (the others a step of p either
        side) touches along ``branch``: the sign changes of its tube's width, the
        spacing of its neighbours at the same height times its sine (signed with its
        direction), which a bounce off the ground mirrors."""
        p = fan.p
        lo, hi = fan.lo, fan.hi
        f_lo = fan.at(lo)
        a = fan.at(np.full(len(p), self.zs)) - f_lo
        finite = np.isfinite(hi)
        h = np.where(finite[:, None], fan.at(np.where(finite, hi, self.zs)) - f_lo, np.inf)
        start = a[:, 0] if branch.up else -a[:, 0]
        segments = branch.segments()
        fractions = (np.arange(_CAUSTIC_SAMPLES) + 0.5) / _CAUSTIC_SAMPLES * 0.9 + 0.05
        widths = []
        bounces = 0
        for n, (going_up, index) in enumerate(segments):
            z_from = self.zs if n == 0 else (lo[0] if going_up else hi[0])
            z_to = zr if n == len(segments) - 1 else (hi[0] if going_up else lo[0])
            z = z_from + (z_to - z_from) * fractions
            above = fan.at(np.broadcast_to(z, (len(p), len(z))))[..., 0] - f_lo[:, 0, None]
            if going_up:
                psi = _times(2 * index, h[:, 0, None]) + above
            else:
                psi = _times(2 * index + 2, h[:, 0, None]) - above
            with np.errstate(invalid="ignore"):
                x = psi - start[:, None]
            spacing = (x[2] - x[1]) / (p[2] - p[1])
            sine = _sine(p[0], self.profile.speed(z)) * (1.0 if going_up else -1.0)
            widths.append(spacing * sine * (-1.0) ** bounces)
            if not going_up and fan.ground[0]:
                bounces += 1
        width = np.concatenate(widths)
        width = width[np.isfinite(width) & (width != 0.0)]
        return int(np.count_nonzero(np.diff(np.sign(width))))

    def curve(self, zr: float, rays: Eigenrays, index: int) -> np.ndarray:
        """(k, 2) points (horizontal distance from the source, height) along eigenray
        ``index`` of ``rays`` from the source to the listener at height ``zr``, close
        enough for the chords between them to stay within ``CHORD_SAGITTA_M`` of the arc;
        ground bounces and turns among them."""
        p = rays.p[index]
        branch = rays.branch[index]
        if not branch.cycle and branch.up == branch.arrives_up and self._level_between(zr):
            return np.array([[0.0, self.zs], [rays.range, zr]])
        fan = _Fan(self.profile, np.array([p]), self.zs)
        lo, hi = float(fan.lo[0]), float(fan.hi[0])
        segments = branch.segments()
        points = [np.array([[0.0, self.zs]])]
        x = 0.0
        for n, (going_up, _) in enumerate(segments):
            z_from = self.zs if n == 0 else (lo if going_up else hi)
            z_to = zr if n == len(segments) - 1 else (hi if going_up else lo)
            heights = self._arc_heights(p, min(z_from, z_to), max(z_from, z_to))
            if not going_up:
                heights = heights[::-1]
            across = fan.at(heights[None])[0, :, 0]
            along = np.abs(across - across[0])
            points.append(np.column_stack([x + along[1:], heights[1:]]))
            x += along[-1]
        return np.vstack(points)

    def _arc_heights(self, p: float, lo: float, hi: float) -> np.ndarray:
        """Heights from ``lo`` to ``hi`` (increasing) at which chords of the ray with
        parameter ``p`` stay within ``CHORD_SAGITTA_M`` of its arcs."""
        profile = self.profile
        z, c, g = profile.z, profile.c, profile.gradient
        inside = z[(z > lo) & (z < hi)]
        cuts = np.concatenate([[lo], inside, [hi]])
        a, b = cuts[:-1], cuts[1:]
        layer = profile.layer(0.5 * (a + b))
        slope = g[layer]
        curvature = p * np.abs(slope)
        length = _segment(profile, p, layer, a, b)[..., 2]
        with np.errstate(divide="ignore"):
            chord = np.sqrt(8.0 * CHORD_SAGITTA_M / curvature)
        count = np.where(
            (curvature > 0.0) & (b > a), np.clip(np.ceil(length / chord), 1, 4096), 1
        ).astype(np.int64)
        # The elevation changes evenly along an arc.
        ea = np.arccos(np.minimum(p * profile.speed(a), 1.0))
        eb = np.arccos(np.minimum(p * profile.speed(b), 1.0))
        piece = np.repeat(np.arange(len(a)), count)
        step = np.arange(len(piece)) - np.repeat(np.cumsum(count) - count, count) + 1
        fraction = step / count[piece]
        angles = ea[piece] + (eb[piece] - ea[piece]) * fraction
        bent = count[piece] > 1
        safe = np.where(bent, slope[piece], 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            heights = z[layer[piece]] + (np.cos(angles) / max(p, 1e-300) - c[layer[piece]]) / safe
        heights = np.where(bent, heights, b[piece])
        heights = np.where(step == count[piece], b[piece], np.clip(heights, a[piece], b[piece]))
        return np.concatenate([[lo], heights])


__all__ = [
    "CAUSTIC_CAP",
    "CHORD_SAGITTA_M",
    "Branch",
    "Eigenrays",
    "Profile",
    "Rays",
    "node_heights",
    "table_heights",
]
