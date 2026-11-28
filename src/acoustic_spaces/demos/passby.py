"""What the demo of any scene (:mod:`acoustic_spaces.demos.scene_demo`) does with a
source going past a listener: its sound faded in and out (:func:`faded`), what goes
along its trajectory behind it (:class:`Trail`, a train's points behind its front, and
:func:`trailed`), how far the pitch of its direct sound falls as it passes
(:func:`pitch_change`), how loud a sound that sounds now and then is around each place
of its trajectory (:func:`blocks_loudness`, :func:`sounded`), and the level of sources
going along one trajectory together (:func:`heard_together`), from each one's
:func:`~acoustic_spaces.walkthrough.pass_by`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

#: A source's sound fades in over its first FADE_S and out over its last, where it is
#: farthest, so that it does not start or stop at once (s).
FADE_S = 1.0


@dataclasses.dataclass(frozen=True)
class Trail:
    """What goes along a source's trajectory with it: the source itself ``lead_s`` behind
    where its trajectory has it (its sound and its levels that much later), and the sounds
    ``behind`` it, each (lag_s, signal, offset_db) ``lag_s`` behind the source, its
    signal scaled as the source's is, its sound power per octave band ``offset_db`` over
    the source's (a train's bogies and cars, behind its front)."""

    lead_s: float = 0.0
    behind: tuple[tuple[float, np.ndarray, np.ndarray], ...] = ()

    @property
    def length_s(self) -> float:
        """How far behind the trajectory the last of it goes (s)."""
        return self.lead_s + max((lag for lag, _, _ in self.behind), default=0.0)


def trailed(passing, trail: Trail, extra_s: float):
    """``passing`` (a :class:`~acoustic_spaces.walkthrough.PassBy` of a source alone)
    with what goes along with it (``trail``): each copy's level that much later, its
    energy band by band ``offset_db`` over the source's, all added, the times going on
    ``extra_s`` past the trajectory's last place, in its steps."""
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.walkthrough import PassBy

    if not trail.behind and trail.lead_s == 0.0 and extra_s <= 0.0:
        return passing
    times = np.asarray(passing.times, dtype=float)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    more = int(math.ceil(max(extra_s, 0.0) / step - 1e-9))
    grid = np.concatenate([times, times[-1] + step * np.arange(1, more + 1)])
    energy = 10.0 ** (np.asarray(passing.levels, dtype=float) / 10.0)
    copies = [(0.0, np.zeros(energy.shape[1])), *((lag, off) for lag, _, off in trail.behind)]
    total = np.zeros((len(grid), energy.shape[1]))
    for lag, offset in copies:
        at = grid - trail.lead_s - lag
        scale = 10.0 ** (np.asarray(offset, dtype=float) / 10.0)
        for band in range(energy.shape[1]):
            total[:, band] += scale[band] * np.interp(
                at, times, energy[:, band], left=0.0, right=0.0
            )
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(total)
    levels_a = np.array([a_weighted(row) for row in levels])
    return PassBy(passing.source, passing.listener, grid, levels, levels_a)


#: Engines with their cylinders in words, for the stills.
CYLINDER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight"}


def faded(x: np.ndarray, rate: int, fade_s: float = FADE_S) -> np.ndarray:
    """``x`` (samples, or samples and channels) faded in over its first ``fade_s`` and
    out over its last (half its length at most each), as a pass's sound plays."""
    x = np.array(x, dtype=float)
    n = len(x)
    fade = min(int(fade_s * rate), n // 2)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    ramp = ramp.reshape(-1, *(1,) * (x.ndim - 1))
    x[:fade] *= ramp
    x[n - fade :] *= ramp[::-1]
    return x


def pitch_change(points, listener, speed_m_s: float | None, c: float) -> float:
    """How far the pitch of a source's direct sound falls at ``listener`` (a position)
    from where its trajectory starts to where it ends (semitones): heard at c / (c - v_r), v_r
    the source's speed towards the listener, along its first leg and its last, at
    ``speed_m_s`` or, None, at each leg's own speed (its length over its time)."""
    positions = np.asarray(points.positions, dtype=float)
    at = np.asarray(listener, dtype=float)

    def towards(k: int, leg: np.ndarray, between: tuple[int, int]) -> float:
        to = at - positions[k]
        distance = float(np.linalg.norm(to))
        length = max(float(np.linalg.norm(leg)), 1e-12)
        if speed_m_s is None:
            times = points.times
            speed = length / float(times[between[1]] - times[between[0]])
        else:
            speed = speed_m_s
        return speed * float(np.dot(leg / length, to)) / max(distance, 1e-12)

    first = towards(0, positions[1] - positions[0], (0, 1))
    last = towards(-1, positions[-1] - positions[-2], (-2, -1))
    return 12.0 * math.log2((c - last) / (c - first))


#: How long a sound's power is averaged over around each moment before each place
#: takes its share (s): 125 ms, the time constant of IEC 61672-1's Fast time weighting,
#: so that a place is no louder or quieter for the beats between a horn's notes, many
#: a second, its step happens to fall on.
LOUDNESS_WINDOW_S = 0.125


def blocks_loudness(x: np.ndarray, rate: int, at: np.ndarray, rms: float) -> np.ndarray:
    """How loud ``x`` (at ``rate``) is around each of the times ``at`` (s), in energy
    against a sound of RMS ``rms``: its mean square, averaged over
    :data:`LOUDNESS_WINDOW_S` around each moment (over the part of that inside ``x`` at
    its ends), from halfway to the time before to halfway to the next (the first from
    the start, the last to the end), over ``rms`` squared; 0 where none of ``x`` is."""
    at = np.asarray(at, dtype=float)
    if len(at) == 0:
        return np.zeros(0)
    power = np.asarray(x, dtype=float) ** 2
    if power.ndim > 1:
        power = power.mean(axis=tuple(range(1, power.ndim)))
    n = len(power)
    half = int(round(LOUDNESS_WINDOW_S * rate / 2.0))
    total = np.concatenate([[0.0], np.cumsum(power)])
    k = np.arange(n)
    lo, hi = np.maximum(k - half, 0), np.minimum(k + half + 1, n)
    smooth = (total[hi] - total[lo]) / np.maximum(hi - lo, 1)
    middle = (at[1:] + at[:-1]) / 2.0
    edges = np.concatenate([[0.0], middle, [n / rate]])
    starts = np.round(np.clip(edges[:-1], 0.0, n / rate) * rate).astype(int)
    ends = np.round(np.clip(edges[1:], 0.0, n / rate) * rate).astype(int)
    run = np.concatenate([[0.0], np.cumsum(smooth)])
    inside = ends > starts
    out = np.zeros(len(at))
    out[inside] = (run[ends[inside]] - run[starts[inside]]) / (ends[inside] - starts[inside])
    return out / rms**2


def sounded(passing, loud: np.ndarray):
    """``passing`` (a :class:`~acoustic_spaces.walkthrough.PassBy`) with each place's
    energy times ``loud`` (one a place): a source that sounds now and then, heard as it
    sounds around each place."""
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.walkthrough import PassBy

    energy = 10.0 ** (np.asarray(passing.levels, dtype=float) / 10.0)
    energy = energy * np.asarray(loud, dtype=float)[:, None]
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(energy)
    levels_a = np.array([a_weighted(row) for row in levels])
    return PassBy(passing.source, passing.listener, passing.times, levels, levels_a)


def heard_together(passes: Sequence):
    """The level along a trajectory of sources going along it together, from each one's
    :func:`~acoustic_spaces.walkthrough.pass_by`: their energies added band by band at
    the times the first's places are heard, each other's read there between its own
    places (linearly in energy; a few milliseconds apart, as their direct sounds
    travel)."""
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.walkthrough import PassBy

    first = passes[0]
    if len(passes) == 1:
        return first
    energy = 10.0 ** (first.levels / 10.0)
    for other in passes[1:]:
        theirs = 10.0 ** (other.levels / 10.0)
        energy = energy + np.stack(
            [np.interp(first.times, other.times, band) for band in theirs.T], axis=1
        )
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(energy)
    levels_a = np.array([a_weighted(row) for row in levels])
    return PassBy(first.source, first.listener, first.times, levels, levels_a)


__all__ = [
    "CYLINDER_WORDS",
    "FADE_S",
    "LOUDNESS_WINDOW_S",
    "Trail",
    "blocks_loudness",
    "faded",
    "heard_together",
    "pitch_change",
    "sounded",
    "trailed",
]
