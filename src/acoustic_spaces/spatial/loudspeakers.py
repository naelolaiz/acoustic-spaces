"""Loudspeaker layouts and vector base amplitude panning (VBAP, Pulkki 1997).

Each arrival is panned between the pair (horizontal layouts) or triangle (layouts with
height) of loudspeakers around its direction, with gains of unit total power. Layouts
with height but nothing below the listener get an imaginary loudspeaker straight down
(and one straight up when nothing is high). Its share is dropped, so arrivals from below
play from the speakers at their azimuth; straight down, all its neighbours play alike.
A gap wider than 90 degrees between the speakers on the horizon (behind the listener in
5.x.4) gets imaginary speakers on the horizon too, which hand their share to the two
speakers either side, so arrivals there stay on the horizon instead of going up to the
height layer. Four or more speakers on one plane (the height ring of the .4 layouts, the
rear corners of 7.x.4) get an imaginary speaker at their middle, which they play alike,
so a symmetric layout pans symmetrically and straight up plays from all four.
Low-frequency effects channels (LFE) take no part in panning and stay silent: a room's
bass reaches every speaker already, and bass management belongs to the playback system.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces.spatial.directions import from_angles


@dataclass(frozen=True)
class Speaker:
    name: str
    azimuth: float  # degrees, positive to the left
    elevation: float = 0.0  # degrees, positive up
    lfe: bool = False  # a low-frequency effects channel: silent


def _layout(*speakers: tuple[str, float, float]) -> tuple[Speaker, ...]:
    return tuple(Speaker(*s) for s in speakers)


def _with_lfe(speakers: tuple[Speaker, ...]) -> tuple[Speaker, ...]:
    """The layout with an LFE channel after the centre, as in WAV channel order."""
    return (*speakers[:3], Speaker("LFE", 0.0, 0.0, lfe=True), *speakers[3:])


#: Named layouts after ITU-R BS.2051, in the channel order of multichannel WAV files.
LAYOUTS: dict[str, tuple[Speaker, ...]] = {
    "stereo": _layout(("L", 30, 0), ("R", -30, 0)),
    "5.0": _layout(("L", 30, 0), ("R", -30, 0), ("C", 0, 0), ("Ls", 110, 0), ("Rs", -110, 0)),
    "7.0": _layout(
        ("L", 30, 0),
        ("R", -30, 0),
        ("C", 0, 0),
        ("Lrs", 135, 0),
        ("Rrs", -135, 0),
        ("Lss", 90, 0),
        ("Rss", -90, 0),
    ),
    "5.0.4": _layout(
        ("L", 30, 0),
        ("R", -30, 0),
        ("C", 0, 0),
        ("Ls", 110, 0),
        ("Rs", -110, 0),
        ("Ltf", 45, 30),
        ("Rtf", -45, 30),
        ("Ltb", 135, 30),
        ("Rtb", -135, 30),
    ),
    "7.0.4": _layout(
        ("L", 30, 0),
        ("R", -30, 0),
        ("C", 0, 0),
        ("Lrs", 135, 0),
        ("Rrs", -135, 0),
        ("Lss", 90, 0),
        ("Rss", -90, 0),
        ("Ltf", 45, 30),
        ("Rtf", -45, 30),
        ("Ltb", 135, 30),
        ("Rtb", -135, 30),
    ),
}
LAYOUTS.update({f"{name[0]}.1{name[3:]}": _with_lfe(LAYOUTS[name]) for name in list(LAYOUTS)[1:]})

#: WAVE_FORMAT_EXTENSIBLE speaker positions, for the channel mask of named layouts.
_WAV_POSITIONS = {
    "L": 0x1,
    "R": 0x2,
    "C": 0x4,
    "LFE": 0x8,
    "Lrs": 0x10,
    "Rrs": 0x20,
    "Ls": 0x200,
    "Rs": 0x400,
    "Lss": 0x200,
    "Rss": 0x400,
    "Ltf": 0x1000,
    "Rtf": 0x4000,
    "Ltb": 0x8000,
    "Rtb": 0x20000,
}
# 5.0 files put the surrounds in the back positions.
_WAV_POSITIONS_5 = {**_WAV_POSITIONS, "Ls": 0x10, "Rs": 0x20}

_EPS = 1e-9
#: Speakers this close to the horizon (degrees) count as on it, and a gap between them
#: wider than ``_MAX_GAP_DEG`` gets imaginary speakers on the horizon.
_HORIZON_DEG = 10.0
_MAX_GAP_DEG = 90.0


#: Twice the area of a triangle of points on the unit sphere below which rounding
#: leaves the direction of its plane loose. Such a triangle has two corners within
#: ``_CLOSE`` of each other: twice the area is at least the product of its sides over 2,
#: as its circle on the sphere has a radius of 1 at most.
_THIN = 1e-5
_CLOSE = 1.1 * (2.0 * _THIN) ** (1.0 / 3.0)


def _triples_to_try(points: np.ndarray, reach: float) -> list[tuple[int, int, int]] | None:
    """The triples of ``points`` (on or near the unit sphere) in increasing order among
    which a search over every triple for faces of their convex hull finds all it does,
    when it holds a triple's points to less than ``reach`` from its plane (well within
    it) for any triple whose cross product reaches ``_THIN``.

    A plane with every point on one side, as near as that, touches the hull at a corner
    and holds only the points within ``reach`` of the plane of one of the hull's
    triangles at that corner (Qhull's), as the others lie further below it: so the
    triples of the corner's star, it and those points. Thinner triples have two points
    within ``_CLOSE``, and every triple with two such is tried. None, for every triple
    to be tried, where the points are not 64-bit floats (whose rounding that takes),
    where Qhull cannot take them (fewer than four, all on one plane, or not finite), or
    where it leaves one off the hull or makes a triangle as thin.
    """
    from scipy.spatial import ConvexHull, QhullError

    if not isinstance(points, np.ndarray) or points.dtype != np.float64:
        return None
    if points.shape[1:] != (3,) or len(points) < 4 or not np.isfinite(points).all():
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    if len(hull.vertices) < len(points):
        return None
    faces = hull.simplices
    a, b, c = (points[faces[:, m]] for m in range(3))
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal, axis=1)
    if length.min() < _THIN:
        return None
    normal /= length[:, None]
    near = np.abs(points @ normal.T - np.einsum("fk,fk->f", normal, a)) < reach
    triples = set()
    for corner in range(len(points)):
        star = np.flatnonzero(near[:, np.any(faces == corner, axis=1)].any(axis=1))
        triples.update(itertools.combinations(star.tolist(), 3))
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for i, j in zip(*np.nonzero(np.triu(gaps < _CLOSE, 1)), strict=True):
        for k in range(len(points)):
            if k != i and k != j:
                triples.add(tuple(sorted((int(i), int(j), k))))
    return sorted(triples)


def _flat_faces(points: np.ndarray, exhaustive: bool = False) -> list[tuple[int, ...]]:
    """The points on each face of the convex hull of ``points`` that holds four or more.

    A face's points are those within 1e-9 of the plane of a triple of them with every
    point on one side of it. The triples tried are those of :func:`_triples_to_try`
    (rounding moves the direction of a plane by up to about 1e-14 over its cross
    product, so up to 1e-9 at ``_THIN``), unless ``exhaustive``: then every triple is.
    """
    triples = None if exhaustive else _triples_to_try(points, 2e-8)
    if triples is None:
        triples = itertools.combinations(range(len(points)), 3)
    faces = set()
    for i, j, k in triples:
        normal = np.cross(points[j] - points[i], points[k] - points[i])
        if np.linalg.norm(normal) < 1e-12:
            continue
        side = (points - points[i]) @ (normal / np.linalg.norm(normal))
        on = np.abs(side) < 1e-9
        if on.sum() > 3 and (np.all(side < 1e-9) or np.all(side > -1e-9)):
            faces.add(tuple(np.flatnonzero(on)))
    return sorted(faces)


def _hull_triangles(points: np.ndarray, exhaustive: bool = False) -> list[tuple[int, int, int]]:
    """Faces of the convex hull of points on the unit sphere around the origin.

    Points are nudged a tiny, fixed amount first, so four on one plane give two
    triangles rather than four overlapping ones. A face is a triple of points with
    every other point on one side of its plane, to 1e-12 times its cross product (so
    to 1e-7 at ``_THIN``). The triples tried are those of :func:`_triples_to_try`,
    unless ``exhaustive``: then every triple is.
    """
    nudged = points + 1e-7 * np.random.default_rng(0).standard_normal(points.shape)
    triples = None if exhaustive else _triples_to_try(nudged, 3e-7)
    if triples is None:
        triples = itertools.combinations(range(len(points)), 3)
    faces = []
    for i, j, k in triples:
        normal = np.cross(nudged[j] - nudged[i], nudged[k] - nudged[i])
        if np.linalg.norm(normal) < 1e-12:
            continue
        side = (nudged - nudged[i]) @ normal
        side[[i, j, k]] = 0.0
        if np.all(side <= 1e-12) or np.all(side >= -1e-12):
            faces.append((i, j, k))
    return faces


@dataclass(frozen=True)
class Loudspeakers:
    """A loudspeaker layout: a name from :data:`LAYOUTS` or a list of speakers."""

    speakers: tuple[Speaker, ...]
    name: str = "speakers"
    _bases: list = field(init=False, repr=False, compare=False)
    _downmix: np.ndarray = field(init=False, repr=False, compare=False)

    @classmethod
    def layout(cls, name: str) -> Loudspeakers:
        if name not in LAYOUTS:
            raise ValueError(f"unknown layout {name!r}; use one of {', '.join(LAYOUTS)}")
        return cls(LAYOUTS[name], name)

    def __post_init__(self) -> None:
        if len(self.speakers) < 2:
            raise ValueError("a loudspeaker layout needs at least two speakers")
        names = [s.name for s in self.speakers]
        if len(set(names)) != len(names):
            raise ValueError("loudspeaker names must be unique")
        if sum(not s.lfe for s in self.speakers) < 2:
            raise ValueError("a loudspeaker layout needs at least two speakers besides LFE")
        object.__setattr__(self, "speakers", tuple(self.speakers))
        bases, downmix = self._make_bases()
        object.__setattr__(self, "_bases", bases)
        object.__setattr__(self, "_downmix", downmix)

    @property
    def channels(self) -> int:
        return len(self.speakers)

    @property
    def label(self) -> str:
        return self.name

    def pressure_gains(self) -> np.ndarray:
        """(channels,) what each channel picks up of a sound pressure that is the same
        all over the room: 1/√n in each of the n main speakers, at unit power as the
        panning plays one direction (on a layout without height speakers, as it plays
        sound from straight above or below), and none in the LFE."""
        mains = [not s.lfe for s in self.speakers]
        return np.where(mains, 1.0 / math.sqrt(sum(mains)), 0.0)

    @property
    def channel_names(self) -> list[str]:
        return [s.name for s in self.speakers]

    @property
    def horizontal(self) -> bool:
        return all(abs(s.elevation) < 1e-6 for s in self._mains())

    def _mains(self) -> list[Speaker]:
        return [s for s in self.speakers if not s.lfe]

    def channel_mask(self) -> int:
        """WAV channel mask of a named layout; 0 (no assignment) for others."""
        if self.name not in LAYOUTS:
            return 0
        positions = _WAV_POSITIONS_5 if self.name.startswith("5.") else _WAV_POSITIONS
        mask = 0
        for s in self.speakers:
            mask |= positions[s.name]
        return mask

    def directions(self) -> np.ndarray:
        return from_angles([s.azimuth for s in self.speakers], [s.elevation for s in self.speakers])

    def _make_bases(self):
        """Pairs or triangles of the speakers other than LFE, by their index among them
        (imaginary speakers after the real ones), and the gains (points, mains) with
        which each of them plays through the real speakers."""
        mains = self._mains()
        n_real = len(mains)
        real = from_angles([s.azimuth for s in mains], [s.elevation for s in mains])
        if self.horizontal:
            order = np.argsort([s.azimuth % 360.0 for s in mains])
            pairs = []
            for a, b in zip(order, np.roll(order, -1), strict=True):
                span = (mains[b].azimuth - mains[a].azimuth) % 360.0
                if 0.0 < span < 180.0 - 1e-6:
                    pairs.append((np.linalg.inv(real[[a, b], :2].T), (a, b)))
            return pairs, np.eye(n_real)
        elevations = np.array([s.elevation for s in mains])
        extra, downmix = [], []
        # Wide gaps on the horizon: imaginary speakers spread evenly across each, played
        # by the speakers either side with the pair's own VBAP gains, so that panning on
        # the horizon is the pair's; across 180 degrees or more, with constant power.
        horizon = sorted(
            (i for i, s in enumerate(mains) if abs(s.elevation) < _HORIZON_DEG),
            key=lambda i: mains[i].azimuth % 360.0,
        )
        if len(horizon) >= 2:
            for a, b in zip(horizon, horizon[1:] + horizon[:1], strict=True):
                gap = (mains[b].azimuth - mains[a].azimuth) % 360.0
                count = math.ceil(gap / _MAX_GAP_DEG - 1e-9) - 1
                for k in range(1, count + 1):
                    t = k / (count + 1)
                    point = from_angles(mains[a].azimuth + t * gap, 0.0)
                    row = np.zeros(n_real)
                    if gap < 180.0 - 1e-6:
                        pair = np.column_stack([real[a, :2], real[b, :2]])
                        row[a], row[b] = np.linalg.solve(pair, point[:2])
                    else:
                        row[a], row[b] = math.cos(t * math.pi / 2), math.sin(t * math.pi / 2)
                    extra.append(point)
                    downmix.append(row)
        # Nothing below (or above): an imaginary speaker at the pole, whose share is dropped.
        if elevations.min() > -_HORIZON_DEG:
            extra.append([0.0, 0.0, -1.0])
            downmix.append(np.zeros(n_real))
        if elevations.max() < _HORIZON_DEG:
            extra.append([0.0, 0.0, 1.0])
            downmix.append(np.zeros(n_real))
        points = np.vstack([real, np.array(extra).reshape(-1, 3)])
        downmix = np.vstack([np.eye(n_real), np.array(downmix).reshape(-1, n_real)])
        # Four or more on one face of the hull: an imaginary speaker at their middle that
        # they play alike (adding up to its direction), so that the face is split evenly
        # around it rather than along one diagonal.
        for face in _flat_faces(points):
            middle = points[list(face)].sum(axis=0)
            length = np.linalg.norm(middle)
            points = np.vstack([points, middle / length])
            downmix = np.vstack([downmix, downmix[list(face)].sum(axis=0) / length])
        bases = []
        for face in _hull_triangles(points):
            matrix = points[list(face)].T
            if abs(np.linalg.det(matrix)) > 1e-9:
                bases.append((np.linalg.inv(matrix), face))
        return bases, downmix

    def gains(self, directions: np.ndarray) -> np.ndarray:
        """VBAP gains (dirs, channels) of unit power for head directions (dirs, 3)."""
        v = np.atleast_2d(np.asarray(directions, dtype=float))
        mains = self._mains()
        n_real = len(mains)
        main_directions = from_angles([s.azimuth for s in mains], [s.elevation for s in mains])
        out = np.zeros((len(v), n_real))
        horizontal = self.horizontal
        for row, p in enumerate(v):
            if horizontal:
                p = p[:2]
                if np.linalg.norm(p) < _EPS:  # straight up or down: all speakers alike
                    out[row] = 1.0 / math.sqrt(n_real)
                    continue
            best, best_g, best_idx = -math.inf, None, None
            for inverse, idx in self._bases:
                g = inverse @ p
                if g.min() > best:
                    best, best_g, best_idx = g.min(), g, idx
            if best_g is None or best < -1e-6:
                # Outside every pair (behind a stereo pair): the nearest speaker.
                out[row, int(np.argmax(main_directions[:, : len(p)] @ p))] = 1.0
            else:
                # On an edge or a corner of the triangle, rounding leaves the speakers away
                # from it gains of about 1e-17, positive with some processors' arithmetic and
                # not with others'; they play nothing, so all give the same gains.
                g = np.where(best_g > 1e-12, best_g, 0.0)
                out[row] = g @ self._downmix[list(best_idx)]
                if np.linalg.norm(out[row]) < 1e-6:  # at an imaginary speaker at a pole
                    pole = best_idx[int(np.argmax(g))]
                    for _, face in self._bases:
                        if pole in face:
                            others = [j for j in face if j != pole]
                            out[row] = np.maximum(out[row], (self._downmix[others] > 0).any(axis=0))
            out[row] /= np.linalg.norm(out[row])
        full = np.zeros((len(v), self.channels))
        full[:, [i for i, s in enumerate(self.speakers) if not s.lfe]] = out
        return full


__all__ = ["LAYOUTS", "Loudspeakers", "Speaker"]
