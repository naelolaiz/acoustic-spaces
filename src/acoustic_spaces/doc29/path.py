"""A flight path built of straight segments, as Doc 29 builds it (Volume 2 section 3.6):
a profile of heights, speeds and powers along a ground track, the take-off and
landing rolls cut into segments by speed, the initial climb and the final approach by
height, airborne segments that change speed much by speed, and the profile merged with
the track's nodes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .tables import FOOT_M, KNOT_M_S, ProfilePoint

#: The heights (m) that cut the initial climb and the final approach below 4231 ft
#: (1289.6 m), for at most 1.5 dB of change of the lateral attenuation along a segment
#: (Volume 2 section 3.6.4): its heights in feet, which Volume 3's reference cases
#: use (its heights in metres are these rounded).
CLIMB_HEIGHTS_M = tuple(feet * FOOT_M for feet in (62, 136, 224, 335, 484, 705, 1099, 2000, 4231))
#: How much an airborne segment's speed may change before it is cut (m/s), 20 kt, and
#: the step of speed it is cut into, and a roll on the runway is, 10 m/s (section 3.6).
SPEED_CHANGE_M_S = 20.0 * KNOT_M_S
SPEED_STEP_M_S = 10.0
#: How near two points of the same speed and power may be before one goes (m),
#: section 3.6.7, and how low a source may be over the runway (m), section 3.5.
NEAREST_POINTS_M = 10.0
LOWEST_SOURCE_M = 1.0


@dataclass(frozen=True)
class Node:
    """A point of a flight path: where it is (m), its speed (m/s), its power, its bank
    angle (degrees, positive in a left turn), and whether it is on the runway."""

    s: float
    height: float
    speed: float
    power: float
    bank: float = 0.0
    ground: bool = False


@dataclass(frozen=True)
class Segment:
    """A straight segment of a flight path, from ``start`` to ``end`` (x, y, z in m),
    with the speed, power and bank angle at each end, and whether it is part of the
    take-off roll or the landing roll."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    speeds: tuple[float, float]
    powers: tuple[float, float]
    banks: tuple[float, float] = (0.0, 0.0)
    roll: str = ""  # "take-off", "landing" or "" (airborne)


@dataclass(frozen=True)
class Track:
    """A ground track: its points (x, y in m) in the direction of flight, and the
    distance along it of its first point (m). Doc 29 measures a departure's track from
    the start of roll and an arrival's from the landing threshold (section 3.3.2)."""

    points: tuple[tuple[float, float], ...]
    start_m: float = 0.0

    @classmethod
    def through(cls, points: Sequence[tuple[float, float]], origin: tuple[float, float]) -> Track:
        """The track through ``points`` measured from ``origin``, a point on it (the
        start of roll or the landing threshold)."""
        pts = tuple((float(x), float(y)) for x, y in points)
        lengths = np.hypot(*np.diff(np.asarray(pts), axis=0).T)
        best, along = math.inf, 0.0
        walked = 0.0
        for (a, b), length in zip(zip(pts[:-1], pts[1:], strict=True), lengths, strict=True):
            ab = np.subtract(b, a)
            t = float(np.clip(np.dot(np.subtract(origin, a), ab) / (length**2), 0.0, 1.0))
            off = float(np.hypot(*(np.add(a, t * ab) - np.asarray(origin))))
            if off < best:
                best, along = off, walked + t * length
            walked += float(length)
        return cls(points=pts, start_m=-along)

    @property
    def node_distances(self) -> np.ndarray:
        """The distance along the track of each of its points (m)."""
        lengths = np.hypot(*np.diff(np.asarray(self.points), axis=0).T)
        return self.start_m + np.concatenate(([0.0], np.cumsum(lengths)))

    def extended(self, first_m: float, last_m: float) -> Track:
        """The track reaching back to ``first_m`` and on to ``last_m`` at least, its
        first and last legs made longer if they must be (section 3.6.6)."""
        s = self.node_distances
        pts = [np.asarray(p, dtype=float) for p in self.points]
        start = self.start_m
        if first_m < s[0]:
            ahead = pts[1] - pts[0]
            pts[0] = pts[0] - ahead / np.hypot(*ahead) * (s[0] - first_m)
            start = first_m
        if last_m > s[-1]:
            ahead = pts[-1] - pts[-2]
            pts[-1] = pts[-1] + ahead / np.hypot(*ahead) * (last_m - s[-1])
        return Track(points=tuple((float(x), float(y)) for x, y in pts), start_m=start)

    def at(self, s: float) -> tuple[float, float]:
        """The point of the track ``s`` along it (m)."""
        nodes = self.node_distances
        i = int(np.clip(np.searchsorted(nodes, s) - 1, 0, len(nodes) - 2))
        f = (s - nodes[i]) / (nodes[i + 1] - nodes[i])
        a, b = np.asarray(self.points[i]), np.asarray(self.points[i + 1])
        x, y = a + f * (b - a)
        return float(x), float(y)


def speed_between(v1, v2, f):
    """The speed ``f`` of the distance from ``v1`` to ``v2`` (m/s), its square linear
    in the distance, so that the speed is linear in time (eq. 3-2); ``f`` a number or
    an array of them."""
    return np.sqrt(np.maximum(v1**2 + f * (v2**2 - v1**2), 0.0))


def interpolate(a: Node, b: Node, s: float) -> Node:
    """The node ``s`` along the way from ``a`` to ``b``: the height and the bank angle
    linear in the distance, the speed and the power in its square, linear in time
    (eq. 3-2)."""
    f = (s - a.s) / (b.s - a.s) if b.s != a.s else 0.0
    return Node(
        s=s,
        height=a.height + f * (b.height - a.height),
        speed=float(speed_between(a.speed, b.speed, f)),
        power=power_between(a.power, b.power, f),
        bank=a.bank + f * (b.bank - a.bank),
        ground=a.ground and b.ground,
    )


def power_between(p1: float, p2: float, f: float) -> float:
    """The power ``f`` of the way from ``p1`` to ``p2`` (eq. 3-2d to 3-2g)."""
    if p1 >= 0 and p2 >= 0:
        return math.sqrt(p1**2 + f * (p2**2 - p1**2))
    if p1 < 0 and p2 < 0:
        return -math.sqrt(p1**2 + f * (p2**2 - p1**2))
    if p1 < 0:
        return p1 + (p2 - p1) * math.sqrt(f)
    return p2 + (p1 - p2) * math.sqrt(1.0 - f)


def _by_speed(a: Node, b: Node, ground: bool) -> list[Node]:
    """The nodes that cut the way from ``a`` to ``b`` into steps of speed of at most
    :data:`SPEED_STEP_M_S`, its length shared out as at a constant acceleration and its
    power in equal steps (eq. 3-3e to 3-3i)."""
    n = int(1 + abs(b.speed - a.speed) / SPEED_STEP_M_S)
    if n < 2:
        return []
    length = b.s - a.s
    dv = (b.speed - a.speed) / n
    dp = (b.power - a.power) / n
    found = []
    s = a.s
    for k in range(1, n):
        s += (a.speed + dv * (k - 0.5)) * 2.0 * length / ((a.speed + b.speed) * n)
        f = (s - a.s) / length
        found.append(
            Node(
                s=s,
                height=a.height + f * (b.height - a.height),
                speed=a.speed + k * dv,
                power=a.power + k * dp,
                bank=a.bank + f * (b.bank - a.bank),
                ground=ground,
            )
        )
    return found


def _by_height(a: Node, b: Node, climbing: bool) -> list[Node]:
    """The nodes that cut the way from ``a`` to ``b`` at the heights of
    :data:`CLIMB_HEIGHTS_M`, scaled to the end's height (the start's on approach) if it
    is below 1289.6 m (eq. 3-4)."""
    top = b.height if climbing else a.height
    low = a.height if climbing else b.height
    if low >= CLIMB_HEIGHTS_M[-1] or top <= low:
        return []
    end = min(top, CLIMB_HEIGHTS_M[-1])
    n = int(np.argmin([abs(h - end) for h in CLIMB_HEIGHTS_M]))
    heights = [end * h / CLIMB_HEIGHTS_M[n] for h in CLIMB_HEIGHTS_M[: n + 1]]
    heights = [h for h in heights if low < h < top and not math.isclose(h, top)]
    if not climbing:
        heights.reverse()
    found = []
    for h in heights:
        s = a.s + (h - a.height) / (b.height - a.height) * (b.s - a.s)
        found.append(interpolate(a, b, s))
    return found


def lowest_approach_node(profile: Sequence[ProfilePoint]) -> float:
    """The distance (m, to touchdown) of the lowest node of an approach above the
    runway: its last profile point in the air, or the lowest height that cuts its last
    segment to touchdown (section 3.6.4) if one does. Volume 3's reference cases put
    the landing threshold there (the jets' point at 15.2 m, 290.2 m before touchdown,
    and the propeller aircraft's cut at 17.2 m, 328.1 m before it)."""
    nodes = [
        Node(s=p.distance_m, height=p.height_m, speed=p.speed_m_s, power=p.power) for p in profile
    ]
    landing = next(i for i, n in enumerate(nodes) if n.height <= 0.0)
    a, b = nodes[landing - 1], nodes[landing]
    cuts = _by_height(a, b, climbing=False)
    return cuts[-1].s if cuts else a.s


def flight_path(
    profile: Sequence[ProfilePoint],
    track: Track,
    departure: bool,
    threshold_m: float = 0.0,
) -> list[Segment]:
    """The segments of the flight of ``profile`` along ``track``, a departure or an
    arrival. An arrival's profile measures its distances to touchdown and its track
    from the landing threshold: ``threshold_m`` is the profile's distance at the
    threshold (negative, before touchdown). The profile is carried on, at its last
    speed and power and its last slope, to the end of the track, or back to its start
    on arrival, and the track on to the end of the profile (section 3.6.6)."""
    shift = 0.0 if departure else -threshold_m
    nodes = [
        Node(s=p.distance_m + shift, height=p.height_m, speed=p.speed_m_s, power=p.power)
        for p in profile
    ]
    ends = track.node_distances
    if departure and nodes[-1].s < ends[-1]:
        a, b = nodes[-2], nodes[-1]
        slope = (b.height - a.height) / (b.s - a.s)
        nodes.append(replace(b, s=float(ends[-1]), height=b.height + slope * (ends[-1] - b.s)))
    if not departure and nodes[0].s > ends[0]:
        a, b = nodes[0], nodes[1]
        slope = (b.height - a.height) / (b.s - a.s)
        nodes.insert(0, replace(a, s=float(ends[0]), height=a.height + slope * (ends[0] - a.s)))
    track = track.extended(nodes[0].s, nodes[-1].s)
    ground = [n.height <= 0.0 for n in nodes]
    nodes = [replace(n, ground=g) for n, g in zip(nodes, ground, strict=True)]

    cut: list[Node] = [nodes[0]]
    for a, b in zip(nodes[:-1], nodes[1:], strict=True):
        if a.ground and b.ground:
            cut.extend(_by_speed(a, b, ground=True))
        else:
            pieces = [a, *_by_height(a, b, climbing=departure), b]
            for c, d in zip(pieces[:-1], pieces[1:], strict=True):
                if abs(d.speed - c.speed) > SPEED_CHANGE_M_S:
                    cut.extend(_by_speed(c, d, ground=False))
                if d is not b:
                    cut.append(d)
        cut.append(b)

    merged: list[Node] = []
    track_nodes = [float(s) for s in track.node_distances[1:-1]]
    for a, b in zip(cut[:-1], cut[1:], strict=True):
        merged.append(a)
        merged.extend(interpolate(a, b, s) for s in track_nodes if a.s < s < b.s)
    merged.append(cut[-1])

    kept = [merged[0]]
    for n in merged[1:]:
        last = kept[-1]
        if (
            abs(n.s - last.s) < NEAREST_POINTS_M
            and math.isclose(n.speed, last.speed)
            and math.isclose(n.power, last.power)
        ):
            continue
        kept.append(n)

    segments = []
    roll = "take-off" if departure else "landing"
    for a, b in zip(kept[:-1], kept[1:], strict=True):
        if b.s <= a.s:
            continue
        xa, ya = track.at(a.s)
        xb, yb = track.at(b.s)
        segments.append(
            Segment(
                start=(xa, ya, max(a.height, LOWEST_SOURCE_M)),
                end=(xb, yb, max(b.height, LOWEST_SOURCE_M)),
                speeds=(a.speed, b.speed),
                powers=(a.power, b.power),
                banks=(a.bank, b.bank),
                roll=roll if a.ground and b.ground else "",
            )
        )
    return segments
