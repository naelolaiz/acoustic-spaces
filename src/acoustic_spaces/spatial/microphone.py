"""A listener's microphone: the pattern its mono output picks up sound through.

A first-order capsule picks up ``w + d . v`` of sound arriving from the direction ``v``
(head coordinates: x forward, y left, z up, as the listener's ``orientation`` turns
them). Aimed forward that is ``a + (1 - a) cos(theta)``, with the ``a`` the pattern's
name gives in :data:`acoustic_spaces.directivity.PATTERNS`: 1 straight ahead, the same
at every frequency. A coincident pair puts two capsules in one place, their channels
left and right:

- ``xy``: two capsules of one pattern aimed ``angle_deg`` apart in the horizontal
  plane, the left one to the left of the front.
- ``blumlein``: two figure-eights aimed 45 degrees to each side.
- ``ms``: a mid capsule aimed forward and a figure-eight aimed left, decoded to
  ``M + g S`` on the left and ``M - g S`` on the right, ``g`` the side's gain.

Each early path is picked up from the direction it arrives from, and each impulse of
the late part from the direction it is drawn from (:mod:`acoustic_spaces.spatial.render`).
A pair's channels hold the same late impulses, as two capsules in one place hear the
same reverberation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from acoustic_spaces.directivity import PATTERNS, FirstOrder
from acoustic_spaces.materials import OCTAVE_BANDS_HZ


@dataclass(frozen=True)
class Microphone:
    """Capsules in one place: channel ``k`` picks up ``weights[k] + axes[k] . v`` of
    sound arriving from ``v`` in head coordinates."""

    label: str
    names: tuple[str, ...]
    weights: tuple[float, ...]
    axes: tuple[tuple[float, float, float], ...]

    @property
    def channels(self) -> int:
        return len(self.names)

    @property
    def channel_names(self) -> list[str]:
        return list(self.names)

    def gains(self, directions: np.ndarray) -> np.ndarray:
        """What each channel picks up (n, channels) of sound arriving from
        ``directions`` (n, 3), unit vectors in head coordinates."""
        v = np.atleast_2d(np.asarray(directions, dtype=float))
        return np.asarray(self.weights) + v @ np.asarray(self.axes).T

    def capsules(self) -> list[tuple[FirstOrder, np.ndarray]]:
        """Each channel as a first-order pattern and the rotation that aims it: columns
        forward, left and up in head coordinates (an MS pair's left channel is a
        first-order pattern aimed between the front and the left)."""
        out = []
        for w, axis in zip(self.weights, self.axes, strict=True):
            d = np.asarray(axis, dtype=float)
            length = float(np.linalg.norm(d))
            forward = d / length if length > 0 else np.array([1.0, 0.0, 0.0])
            side = np.cross([0.0, 0.0, 1.0], forward)
            if np.linalg.norm(side) < 1e-9:  # aimed straight up or down
                side = np.array([0.0, 1.0, 0.0])
            side /= np.linalg.norm(side)
            turn = np.column_stack([forward, side, np.cross(forward, side)])
            a = w / (w + length) if w + length > 0 else 1.0
            out.append((FirstOrder((a,) * len(OCTAVE_BANDS_HZ)), turn))
        return out


def first_order(pattern: str) -> Microphone:
    """One capsule of a first-order ``pattern`` (a name of ``PATTERNS``) aimed forward."""
    a = PATTERNS[pattern]
    return Microphone(pattern, ("M",), (a,), ((1.0 - a, 0.0, 0.0),))


def xy(angle_deg: float = 90.0, capsule: str = "cardioid") -> Microphone:
    """Two ``capsule`` patterns aimed ``angle_deg`` apart in the horizontal plane."""
    a = PATTERNS[capsule]
    half = math.radians(angle_deg) / 2.0
    c, s = (1.0 - a) * math.cos(half), (1.0 - a) * math.sin(half)
    return Microphone("xy", ("L", "R"), (a, a), ((c, s, 0.0), (c, -s, 0.0)))


def blumlein() -> Microphone:
    """Two figure-eights at 90 degrees, aimed 45 degrees to each side."""
    return replace(xy(90.0, "figure8"), label="blumlein")


def ms(mid: str = "cardioid", side_db: float = 0.0) -> Microphone:
    """A ``mid`` capsule aimed forward and a figure-eight aimed left ``side_db`` above
    it, decoded to left and right."""
    a = PATTERNS[mid]
    g = 10.0 ** (side_db / 20.0)
    return Microphone("ms", ("L", "R"), (a, a), ((1.0 - a, g, 0.0), (1.0 - a, -g, 0.0)))


def of_listener(listener) -> Microphone | None:
    """The microphone a listener's ``pattern`` describes (``scene.Listener``); None
    when it has none or it is omni."""
    pattern = getattr(listener, "pattern", None)
    if pattern is None or pattern == "omni":
        return None
    if isinstance(pattern, str):
        return first_order(pattern)
    if pattern.pair == "xy":
        return xy(pattern.angle_deg, pattern.capsule)
    if pattern.pair == "blumlein":
        return blumlein()
    return ms(pattern.mid, pattern.side_db)


def for_listener(fmt, listener):
    """The format ``listener``'s output in ``fmt`` is rendered in: its microphone when
    ``fmt`` is mono and it has a pattern, else ``fmt``."""
    from acoustic_spaces.spatial.render import Mono

    if isinstance(fmt, Mono):
        microphone = of_listener(listener)
        if microphone is not None:
            return microphone
    return fmt


__all__ = ["Microphone", "blumlein", "first_order", "for_listener", "ms", "of_listener", "xy"]
