"""Loudspeaker layouts and vector base amplitude panning (VBAP, Pulkki 1997).

Each arrival is panned between the pair (horizontal layouts) or triangle (layouts with
height) of loudspeakers around its direction, with gains of unit total power. Layouts
with height but nothing below the listener get an imaginary loudspeaker straight down
(and one straight up when nothing is high). Its share is dropped, so arrivals from below
play from the speakers at their azimuth; straight down, all its neighbours play alike.
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


def _hull_triangles(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Faces of the convex hull of points on the unit sphere around the origin.

    Points are nudged a tiny, fixed amount first, so four on one plane (a ring of height
    speakers) give two triangles rather than four overlapping ones.
    """
    nudged = points + 1e-7 * np.random.default_rng(0).standard_normal(points.shape)
    faces = []
    for i, j, k in itertools.combinations(range(len(points)), 3):
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
        object.__setattr__(self, "_bases", self._make_bases())

    @property
    def channels(self) -> int:
        return len(self.speakers)

    @property
    def label(self) -> str:
        return self.name

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
        """Pairs or triangles of the speakers other than LFE, by their index among them."""
        mains = self._mains()
        real = from_angles([s.azimuth for s in mains], [s.elevation for s in mains])
        if self.horizontal:
            order = np.argsort([s.azimuth % 360.0 for s in mains])
            pairs = []
            for a, b in zip(order, np.roll(order, -1), strict=True):
                span = (mains[b].azimuth - mains[a].azimuth) % 360.0
                if 0.0 < span < 180.0 - 1e-6:
                    pairs.append((np.linalg.inv(real[[a, b], :2].T), (a, b)))
            return pairs
        elevations = np.array([s.elevation for s in mains])
        extra = []
        if elevations.min() > -10.0:
            extra.append([0.0, 0.0, -1.0])
        if elevations.max() < 10.0:
            extra.append([0.0, 0.0, 1.0])
        points = np.vstack([real, np.array(extra).reshape(-1, 3)])
        bases = []
        for face in _hull_triangles(points):
            matrix = points[list(face)].T
            if abs(np.linalg.det(matrix)) > 1e-9:
                bases.append((np.linalg.inv(matrix), face))
        return bases

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
                g = np.maximum(best_g, 0.0)
                for k, i in enumerate(best_idx):
                    if i < n_real:
                        out[row, i] = g[k]
                if np.linalg.norm(out[row]) < 1e-6:  # at an imaginary speaker
                    imaginary = [i for i in best_idx if i >= n_real][0]
                    for _, face in self._bases:
                        if imaginary in face:
                            out[row, [j for j in face if j < n_real]] = 1.0
            out[row] /= np.linalg.norm(out[row])
        full = np.zeros((len(v), self.channels))
        full[:, [i for i, s in enumerate(self.speakers) if not s.lfe]] = out
        return full


__all__ = ["LAYOUTS", "Loudspeakers", "Speaker"]
