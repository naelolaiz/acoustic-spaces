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

A pattern may change with frequency, as a real capsule's does (wider in the bass, say):
an ``a`` per octave band (8) or third-octave band (24), given as pattern names or
values, for a single capsule, an XY pair's capsules or an MS pair's mid one. Its ``a``
goes from band to band across the crossovers of the synthesis's band filters
(:func:`acoustic_spaces.synthesis.band_gains`): a capsule picks up
``(d . v) + a(f) (1 - d . v)``, ``d`` its aim, so the response is rendered twice, once
through the part that does not change (``d . v``) and once through the part ``a``
scales, which then goes through a filter of gain ``a`` in each band
(:meth:`Microphone.parts`). The two take the same paths and impulses, so this is
exact; a pattern that is the same in every band is rendered once, as before.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from acoustic_spaces.directivity import PATTERNS, FirstOrder
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.materials.bands import resolution_of

#: The band whose pattern a microphone that changes with frequency is drawn in (its
#: balloon in the editor): 1 kHz, as a source's balloon is.
SHOWN_BAND = OCTAVE_BANDS_HZ.index(1000)
#: A direction or an axis, in head coordinates.
Vector = tuple[float, float, float]


def _vector(values) -> Vector:
    x, y, z = (float(v) for v in values)
    return (x, y, z)


@dataclass(frozen=True)
class Microphone:
    """Capsules in one place: channel ``k`` picks up ``weights[k] + axes[k] . v`` of
    sound arriving from ``v`` in head coordinates."""

    label: str
    names: tuple[str, ...]
    weights: tuple[float, ...]
    axes: tuple[Vector, ...]
    #: For a pattern that changes with frequency, the ``a`` of its capsules in each band
    #: (8 or 24; the channels share it), each channel's capsule aim (a unit vector) and
    #: what it picks up besides that capsule (an MS pair's figure-eight); ``weights``
    #: and ``axes`` then hold the pattern at 1 kHz (:data:`SHOWN_BAND`). None for one
    #: pattern at every frequency.
    a: tuple[float, ...] | None = None
    aims: tuple[Vector, ...] = ()
    sides: tuple[Vector, ...] = ()

    @property
    def channels(self) -> int:
        return len(self.names)

    @property
    def channel_names(self) -> list[str]:
        return list(self.names)

    def gains(self, directions: np.ndarray) -> np.ndarray:
        """What each channel picks up (n, channels) of sound arriving from
        ``directions`` (n, 3), unit vectors in head coordinates."""
        if self.a is not None:
            raise ValueError(
                f"the {self.label} microphone picks up differently in each band: "
                "render it through its parts()"
            )
        v = np.atleast_2d(np.asarray(directions, dtype=float))
        return np.asarray(self.weights) + v @ np.asarray(self.axes).T

    def parts(self) -> tuple[Microphone, Microphone]:
        """A microphone whose pattern changes with frequency as two that do not: what
        each channel picks up whatever ``a`` (``d . v`` and its side's) and what ``a``
        scales (``1 - d . v``), ``d`` its capsule's aim. It picks up the first plus the
        second through a filter of gain ``a`` in each band (:meth:`a_in`)."""
        if self.a is None:
            raise ValueError(f"the {self.label} microphone is the same in every band")
        steady = tuple(
            _vector(np.add(aim, side)) for aim, side in zip(self.aims, self.sides, strict=True)
        )
        scaled = tuple(_vector(np.negative(aim)) for aim in self.aims)
        zeros, ones = (0.0,) * self.channels, (1.0,) * self.channels
        return (
            Microphone(self.label, self.names, zeros, steady),
            Microphone(self.label, self.names, ones, scaled),
        )

    def a_in(self, bands: int) -> np.ndarray:
        """``a`` in ``bands`` bands (8 octaves or 24 third octaves), as a source's
        pattern goes from one to the other."""
        if self.a is None:
            raise ValueError(f"the {self.label} microphone is the same in every band")
        return np.asarray(FirstOrder(self.a).in_bands(resolution_of(bands)).a, dtype=float)

    def band_gains(self, directions: np.ndarray, bands: int) -> np.ndarray:
        """What each channel picks up (n, channels, bands) in each of ``bands`` bands (8
        octaves or 24 third octaves) of sound arriving from ``directions`` (n, 3)."""
        if self.a is None:
            return np.repeat(self.gains(directions)[:, :, None], bands, axis=2)
        steady, scaled = self.parts()
        a = self.a_in(bands)
        return steady.gains(directions)[:, :, None] + scaled.gains(directions)[:, :, None] * a

    def spectral_gains(self, directions: np.ndarray, shapes: np.ndarray) -> np.ndarray:
        """What each channel picks up (n, channels, bins) of sound arriving from
        ``directions`` (n, 3) at the frequencies of the synthesis's band filters
        ``shapes`` (bands, bins) (:func:`acoustic_spaces.synthesis.band_gains`), which
        sum to one: each band's gain across its share of the frequencies."""
        if self.a is None:
            return self.gains(directions)[:, :, None]
        return self.band_gains(directions, len(shapes)) @ np.asarray(shapes)

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


Pattern = str | float | Sequence[str | float]


def pattern_a(pattern: Pattern) -> float | tuple[float, ...]:
    """The ``a`` of a pattern: a name of ``PATTERNS`` or a value, or one of them per
    band, which gives one ``a`` when the bands agree."""
    if isinstance(pattern, str | int | float):
        return PATTERNS[pattern] if isinstance(pattern, str) else float(pattern)
    values = tuple(PATTERNS[v] if isinstance(v, str) else float(v) for v in pattern)
    FirstOrder(values)  # 8 or 24 of them, each from 0 to 1
    return values[0] if len(set(values)) == 1 else values


def pattern_label(pattern: Pattern) -> str:
    """How a pattern is named: its name, or its value or name in each band."""
    if isinstance(pattern, str | int | float):
        return str(pattern)
    if len({str(v) for v in pattern}) == 1:
        return str(pattern[0])
    return "per band: " + ", ".join(str(v) for v in pattern)


def _capsules(
    label: str,
    names: tuple[str, ...],
    a: float | tuple[float, ...],
    aims: list[Vector],
    sides: list[Vector],
) -> Microphone:
    """Capsules of one pattern (``a``, or one per band) aimed along ``aims``, each
    channel also picking up its row of ``sides``."""

    def axes(value: float) -> tuple[Vector, ...]:
        return tuple(
            _vector((1.0 - value) * np.asarray(aim) + np.asarray(side))
            for aim, side in zip(aims, sides, strict=True)
        )

    if not isinstance(a, tuple):
        return Microphone(label, names, (a,) * len(names), axes(a))
    shown = FirstOrder(a).in_bands("octave").a[SHOWN_BAND]
    return Microphone(
        label,
        names,
        (shown,) * len(names),
        axes(shown),
        a=a,
        aims=tuple(_vector(aim) for aim in aims),
        sides=tuple(_vector(side) for side in sides),
    )


def first_order(pattern: Pattern) -> Microphone:
    """One capsule of a first-order ``pattern`` (a name of ``PATTERNS`` or a value of
    ``a``, or one per band) aimed forward."""
    return _capsules(
        pattern_label(pattern), ("M",), pattern_a(pattern), [(1.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]
    )


def xy(angle_deg: float = 90.0, capsule: Pattern = "cardioid") -> Microphone:
    """Two ``capsule`` patterns aimed ``angle_deg`` apart in the horizontal plane."""
    half = math.radians(angle_deg) / 2.0
    c, s = math.cos(half), math.sin(half)
    zero = (0.0, 0.0, 0.0)
    return _capsules(
        "xy", ("L", "R"), pattern_a(capsule), [(c, s, 0.0), (c, -s, 0.0)], [zero, zero]
    )


def blumlein() -> Microphone:
    """Two figure-eights at 90 degrees, aimed 45 degrees to each side."""
    return replace(xy(90.0, "figure8"), label="blumlein")


def ms(mid: Pattern = "cardioid", side_db: float = 0.0) -> Microphone:
    """A ``mid`` capsule aimed forward and a figure-eight aimed left ``side_db`` above
    it, decoded to left and right."""
    g = 10.0 ** (side_db / 20.0)
    forward = (1.0, 0.0, 0.0)
    return _capsules(
        "ms", ("L", "R"), pattern_a(mid), [forward, forward], [(0.0, g, 0.0), (0.0, -g, 0.0)]
    )


def of_listener(listener) -> Microphone | None:
    """The microphone a listener's ``pattern`` describes (``scene.Listener``); None
    when it has none or it is omni."""
    pattern = getattr(listener, "pattern", None)
    if pattern is None or pattern == "omni":
        return None
    if isinstance(pattern, str | list | tuple):
        return None if pattern_a(pattern) == 1.0 else first_order(pattern)
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


__all__ = [
    "SHOWN_BAND",
    "Microphone",
    "blumlein",
    "first_order",
    "for_listener",
    "ms",
    "of_listener",
    "pattern_a",
    "pattern_label",
    "xy",
]
