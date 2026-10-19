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
delays its phase by a quarter period.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class Profile:
    """The effective speed of sound (m/s) along one horizontal direction, linear in
    height between nodes ``z`` (m above the ground, from 0, increasing) and constant
    above the last."""

    z: np.ndarray
    c: np.ndarray
    #: (N + 1,) dc/dz in each layer, the constant one above the last node with it.
    gradient: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=float)
        c = np.asarray(self.c, dtype=float)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "gradient", np.append(np.diff(c) / np.diff(z), 0.0))

    @classmethod
    def of(cls, atmosphere, ground_c: float, towards=None) -> Profile:
        """The profile of ``atmosphere`` (:class:`~acoustic_spaces.scene.Atmosphere`)
        with the air at ``ground_c`` °C on the ground, along the horizontal unit vector
        ``towards`` (2,) (None: the speed of sound in still air)."""
        z = node_heights(atmosphere)
        return cls(z, np.asarray(atmosphere.sound_speed(z, ground_c, towards), dtype=float))

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
    if atmosphere is not None and atmosphere.table is not None:
        z = np.union1d(z, [row[0] for row in atmosphere.table])
    return z


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


def _piece(p, ca, cb, dz):
    """The horizontal distance, travel time and length of a ray with parameter ``p``
    through a piece of a layer ``dz`` (>= 0) high, from speed ``ca`` at its bottom to
    ``cb`` at its top (linear between), where the ray never turns (p c <= 1): exact for
    an arc of a circle, written to stay accurate for straight rays and at turning
    points."""
    sa, sb = _sine(p, ca), _sine(p, cb)
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
    c_hi = c0 + slope * (hi - base)
    limit = 1.0 / np.maximum(p, 1e-300)
    # Where the speed reaches 1 / p the ray turns: only the side of the layer below
    # that speed is reachable.
    with np.errstate(divide="ignore", invalid="ignore"):
        turn = base + (limit - c0) / np.where(slope != 0.0, slope, 1.0)
    rising = slope > 0.0
    falling = slope < 0.0
    top = np.where(rising & (c_hi > limit), np.minimum(hi, turn), hi)
    bottom = np.where(falling & (c_lo > limit), np.maximum(lo, turn), lo)
    open_ = (top > bottom) & ~((slope == 0.0) & (c_lo > limit))
    cb = np.where(rising & (c_hi > limit), limit, c0 + slope * (top - base))
    ca = np.where(falling & (c_lo > limit), limit, c0 + slope * (bottom - base))
    # Closed pieces: any speeds the ray could have, for the formulas to stay finite.
    ca, cb = np.where(open_, ca, 0.0), np.where(open_, cb, 0.0)
    dz = np.where(open_, top - bottom, 0.0)
    dx, dt, ds = _piece(np.where(open_, p, 0.0), np.where(open_, ca, 1.0), np.where(open_, cb, 1.0), dz)
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
        self.table = np.concatenate(
            [np.zeros((len(p), 1, 3)), np.cumsum(pieces, axis=1)], axis=1
        )
        limit = 1.0 / np.maximum(p, 1e-300)
        blocked = c[None, :] > limit[:, None]  # (P, N + 1) at the nodes
        source = int(profile.layer(self.zs))
        above = blocked[:, source + 1 :]
        has_upper = above.any(axis=1)
        first = source + 1 + np.argmax(above, axis=1)
        under = first - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            upper = z[under] + (limit - c[under]) / g[under]
        self.hi = np.where(has_upper, np.maximum(upper, self.zs), np.inf)
        below = blocked[:, : source + 1]
        has_lower = below.any(axis=1)
        last = source - np.argmax(below[:, ::-1], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
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
    high = np.where((turns > 0)[..., None], turns[..., None] * np.where(finite[:, None], h, 0.0), 0.0)
    out = np.where(up[..., None], -a, a) + np.where(arrives_up[..., None], b, -b) + high
    ok &= np.all(np.isfinite(out), axis=-1)
    return np.where(ok[..., None], out, np.nan)


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
        self.grid = np.cos(angle) / self.cs
        self.grid[-1] = 0.0
        self._fan = _Fan(profile, self.grid, self.zs)

    def eigenrays(
        self, zr: float, distance: float, max_bounces: int, max_time: float = np.inf
    ) -> Eigenrays:
        """The eigenrays to a listener at height ``zr`` ``distance`` m away horizontally
        (> 0) that bounce off the ground ``max_bounces`` times at most and arrive by
        ``max_time``."""
        zr, distance = float(zr), float(distance)
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
                    last, step = f[kind, inside], f[kind, inside] - f[kind, deeper]
                    if last * step > 0.0 or abs(last) > _EDGE_REACH * abs(step):
                        continue
                edges.append((kind, cycle, inside, outside, f[kind, inside]))
            if cycle >= 1 and not np.any(f <= 0.0):
                # The rays only go farther pass after pass: once none of a pass's falls
                # short of the listener, none after it does.
                break
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
        found = []
        if branches:
            kind, cycle = (np.array(c) for c in zip(*branches, strict=True))
            roots = self._bisect(
                zr, distance, kinds[kind, 0], kinds[kind, 1], cycle, np.array(lows), np.array(highs)
            )
            for i, root in enumerate(roots):
                if np.isfinite(root):
                    branch = Branch(bool(kinds[kind[i], 0]), bool(kinds[kind[i], 1]), int(cycle[i]))
                    found.append((float(root), branch))
        return self._describe(zr, distance, found, max_bounces, max_time)

    def _edge(self, zr, up, arrives_up, cycle, inside, outside) -> np.ndarray:
        """The ray parameters closest to ``outside`` from ``inside`` (arrays) for which
        the branches still reach the listener."""
        for _ in range(_BISECTIONS // 2):
            middle = 0.5 * (inside + outside)
            ok = np.isfinite(self._range(zr, middle, up, arrives_up, cycle))
            inside = np.where(ok, middle, inside)
            outside = np.where(ok, outside, middle)
        return inside

    def _bisect(self, zr, distance, up, arrives_up, cycle, lo, hi) -> np.ndarray:
        """The ray parameters where the branches (``up``, ``arrives_up``, ``cycle``)
        reach ``distance``, each between ``lo`` and ``hi``; nan where the range jumps
        there instead."""
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
            if np.all((np.abs(f) <= 1e-12 * (1.0 + distance)) | (hi - lo <= 1e-15 * hi)):
                break
        f_lo = self._range(zr, lo, up, arrives_up, cycle) - distance
        f_hi = self._range(zr, hi, up, arrives_up, cycle) - distance
        nearer = np.abs(f_lo) <= np.abs(f_hi)
        root = np.where(nearer, lo, hi)
        miss = np.where(nearer, np.abs(f_lo), np.abs(f_hi))
        # A jump of the range is no crossing.
        return np.where(miss <= _RANGE_TOLERANCE * (1.0 + distance), root, np.nan)

    def _range(self, zr, p, up, arrives_up, cycle) -> np.ndarray:
        fan = _Fan(self.profile, np.asarray(p, dtype=float), self.zs)
        return _phase(fan, zr, up, arrives_up, cycle)[:, 0]

    def _describe(self, zr, distance, found, max_bounces, max_time) -> Eigenrays:
        rows = []
        for p, branch in found:
            rows.append(self._ray(zr, distance, p, branch))
        rows = [r for r in rows if r is not None]
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

    def _ray(self, zr, distance, p, branch) -> dict | None:
        # A step small against how far the ray is from grazing at either end, where its
        # range changes fastest.
        top = 1.0 / max(self.cs, float(self.profile.speed(zr)))
        step = 1e-4 * max(min(p, top - p), 1e-12 * top)
        ps = np.array([p, p - step, p + step])
        ps = np.clip(ps, 0.0, 1.0 / self.cs)
        fan = _Fan(self.profile, ps, self.zs)
        out = _phase(fan, zr, branch.up, branch.arrives_up, branch.cycle)
        if not np.all(np.isfinite(out[:, 0])):
            return None
        x, t, s = out[0]
        slope = (out[2, 0] - out[1, 0]) / (ps[2] - ps[1])
        sin_s = float(_sine(p, self.cs))
        sin_r = float(_sine(p, float(self.profile.speed(zr))))
        launch = math.copysign(math.acos(min(p * self.cs, 1.0)), 1.0 if branch.up else -1.0)
        arrival = math.copysign(
            math.acos(min(p * float(self.profile.speed(zr)), 1.0)),
            1.0 if branch.arrives_up else -1.0,
        )
        tube = distance * abs(slope) * sin_s * sin_r
        free = 1.0 / max(s, 1e-12) ** 2
        spreading = p * self.cs**2 / tube if tube > 0.0 else np.inf
        capped = not spreading <= CAUSTIC_CAP * free
        if capped:
            spreading = CAUSTIC_CAP * free
        ground = bool(fan.ground[0])
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

__all__ = ["CAUSTIC_CAP", "CHORD_SAGITTA_M", "Branch", "Eigenrays", "Profile", "Rays", "node_heights"]
