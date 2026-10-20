"""Walkthroughs: a listener or a source walking through the room, heard as it goes.

A listener or a point source with a ``walk`` in the scene goes from its position
through the points of ``walk.to``, at ``walk.speed_m_s``, facing the way it goes unless
``walk.face_forward`` is false (it then keeps its orientation)::

    listeners:
      - name: visitor
        position: [2.0, 3.0, 1.6]
        walk: {to: [[12.0, 3.0, 1.6], [12.0, 8.0, 1.6]], speed_m_s: 1.2}

:func:`walk_points` places the walker every ``step_m`` along its path, and
:func:`simulate_walk` simulates a response at each place. The early sound changes
quickly from place to place, the late field slowly, so the late part, the costliest to
trace, is traced only every ``late_step_m``; the places in between mix the late parts of
those on either side, in proportion to how near they are. :func:`render_walk` then
plays signals through them: the response of the place the walker is nearest to, with a
crossfade of ``crossfade_s`` half-way between two places, so that the sound changes
smoothly as it walks. Each place's late part is drawn from the same random sequence,
so the crossfades join tails that only differ in their decay. The walker stands at the
end of its path once it is there, for as long as the signals last.

There is no Doppler shift: at walking speed it is under half a percent (8 cents).
Outdoors in turbulence the early sound flickers along the walk as in
:func:`~acoustic_spaces.render.auralize`, through the same eddies from place to place.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.engine.simulate import SimulationResult, simulate
from acoustic_spaces.render import (
    _as_2d,
    convolve,
    flickers,
    gusts_for,
    power_gains,
    resample,
    through_eddies,
)
from acoustic_spaces.room import Room
from acoustic_spaces.scene.model import Simulation as SimulationSettings

#: Distance (m) between the places a walk is simulated at.
STEP_M = 0.25
#: Distance (m) between the places a walk's late part is traced at.
LATE_STEP_M = 2.0
#: Length (s) of the crossfade from one place's response to the next.
CROSSFADE_S = 0.05


@dataclass(frozen=True)
class WalkPoints:
    """The places a walker is simulated at."""

    #: When it is at each place (s from the start).
    times: np.ndarray
    #: Where each place is (N, 3).
    positions: np.ndarray
    #: The way it faces at each place (yaw in degrees), or None when it keeps its own.
    yaws: np.ndarray | None


@dataclass(frozen=True)
class WalkResponses:
    """Responses along a walk, from :func:`simulate_walk`."""

    #: The walker's name.
    name: str
    #: ``"listeners"`` or ``"sources"``: what walks.
    kind: str
    #: The places it was simulated at.
    points: WalkPoints
    #: The names the walker has at its places in :attr:`result`, in order
    #: (``"<name> #<place> @ <time> s"``).
    names: tuple[str, ...]
    #: The simulation, with the walker at every place.
    result: SimulationResult
    #: The places that have their own late part (indices); the others mix theirs.
    late_places: tuple[int, ...] = ()

    @property
    def duration_s(self) -> float:
        return float(self.points.times[-1])


def _walker(room: Room, name: str | None):
    """The listener or source that walks: ``name``, or the only one with a walk."""
    scene = room.scene
    found = [
        (kind, item)
        for kind in ("listeners", "sources")
        for item in getattr(scene, kind)
        if (item.name == name if name is not None else item.walk is not None)
    ]
    if name is not None and not found:
        raise ValueError(f"no listener or source named {name!r}")
    if not found:
        raise ValueError("no listener or source walks: give one a walk")
    if len(found) > 1:
        names = ", ".join(item.name for _, item in found)
        raise ValueError(f"more than one walks ({names}): pick one")
    kind, item = found[0]
    if item.walk is None:
        raise ValueError(f"{item.name!r} has no walk")
    return kind, item


def walk_points(item, step_m: float = STEP_M) -> WalkPoints:
    """The places along ``item``'s walk (a scene listener or source), ``step_m`` or a
    little less apart, from its position to the walk's last point."""
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    corners = np.array([item.position, *item.walk.to], dtype=np.float64)
    legs = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(legs)])
    if along[-1] == 0.0:
        raise ValueError(f"the walk of {item.name!r} goes nowhere: it never leaves its position")
    count = max(int(math.ceil(along[-1] / step_m)) + 1, 2)
    s = np.linspace(0.0, along[-1], count)
    positions = np.column_stack([np.interp(s, along, corners[:, k]) for k in range(3)])
    yaws = None
    if item.walk.face_forward:
        heading = np.diff(corners, axis=0)[:, :2]
        flat = np.hypot(heading[:, 0], heading[:, 1]) < 1e-9
        angles = np.degrees(np.arctan2(heading[:, 1], heading[:, 0]))
        # Straight up or down (a lift): the way it faced before.
        last = item.orientation.yaw
        for k in range(len(angles)):
            if flat[k]:
                angles[k] = last
            last = angles[k]
        # From each leg's heading to the next's the shorter way round, ...
        turns = (np.diff(angles) + 180.0) % 360.0 - 180.0
        angles = angles[0] + np.concatenate([[0.0], np.cumsum(turns)])
        # ... at turn_deg_s through each corner: half of the turn before it, half after
        # it, and no further than half-way along either leg.
        along_turn, yaw_turn = [0.0], [angles[0]]
        for j, turn in enumerate(turns, start=1):
            if turn == 0.0:
                continue
            half = 0.5 * item.walk.speed_m_s * abs(turn) / item.walk.turn_deg_s
            half = min(half, 0.5 * legs[j - 1], 0.5 * legs[j])
            along_turn += [along[j] - half, along[j] + half]
            yaw_turn += [angles[j - 1], angles[j]]
        along_turn.append(along[-1])
        yaw_turn.append(angles[-1])
        yaws = 180.0 - (180.0 - np.interp(s, along_turn, yaw_turn)) % 360.0  # (-180, 180]
    return WalkPoints(s / item.walk.speed_m_s, positions, yaws)


def late_mixing(distances: np.ndarray, late_step_m: float) -> dict[int, dict[int, float]]:
    """The places of a walk (``distances`` along it, m, rising) that take their late part
    from others, each with the places it mixes and their weights: places are traced every
    ``late_step_m`` or a little less, from the first to the last."""
    distances = np.asarray(distances, dtype=np.float64)
    if late_step_m <= 0:
        raise ValueError("late_step_m must be positive")
    count = max(int(math.ceil(distances[-1] / late_step_m)) + 1, 2)
    targets = np.linspace(0.0, distances[-1], count)
    traced = np.unique(np.abs(distances[None, :] - targets[:, None]).argmin(axis=1))
    mixing = {}
    for k, d in enumerate(distances):
        if k in traced:
            continue
        j = int(np.searchsorted(distances[traced], d))
        a, b = int(traced[j - 1]), int(traced[j])
        w = float((d - distances[a]) / (distances[b] - distances[a]))
        mixing[k] = {a: 1.0 - w, b: w}
    return mixing


def simulate_walk(
    room: Room,
    name: str | None = None,
    *,
    listener: str | None = None,
    step_m: float = STEP_M,
    late_step_m: float | None = LATE_STEP_M,
    settings: SimulationSettings | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> WalkResponses:
    """Simulate the responses along a walk: ``name`` walks (by default the only
    listener or source with a walk), placed every ``step_m``.

    A walking listener hears every source; a walking source is heard at ``listener``
    (by default the scene's only listener), with the other sources standing still.

    The late part is traced every ``late_step_m`` (see the module's description), or at
    every place with ``late_step_m=None`` or the statistical late method. ``settings``
    (default: the scene's) and ``threads`` (default: their ``performance.threads``) go
    to :func:`~acoustic_spaces.engine.simulate`.
    """
    kind, item = _walker(room, name)
    points = walk_points(item, step_m)
    # Numbered, as places less than 10 ms apart would share a time.
    names = tuple(f"{item.name} #{k} @ {t:.2f} s" for k, t in enumerate(points.times))
    copies = []
    for k, (place, label) in enumerate(zip(points.positions, names, strict=True)):
        update = {"name": label, "position": tuple(place), "walk": None}
        if points.yaws is not None:
            update["orientation"] = item.orientation.model_copy(update={"yaw": points.yaws[k]})
        copies.append(item.model_copy(update=update))
    scene = room.scene
    if kind == "listeners":
        update = {"listeners": copies}
    else:
        listeners = scene.listeners
        if listener is not None:
            listeners = [lis for lis in listeners if lis.name == listener]
            if not listeners:
                raise ValueError(f"no listener named {listener!r}")
        elif len(listeners) != 1:
            raise ValueError(f"pick a listener: {', '.join(lis.name for lis in listeners)}")
        others = [s for s in scene.sources if s.name != item.name]
        update = {"sources": [*others, *copies], "listeners": listeners}
    walked = Room.from_scene(scene.model_copy(update=update), reuse=room)
    inside = walked.listeners_inside if kind == "listeners" else walked.sources_inside
    if inside is not None:
        offset = 0 if kind == "listeners" else len(walked.scene.sources) - len(copies)
        out = [k for k in range(len(copies)) if not inside[offset + k]]
        if out:
            k = out[0]
            where = ", ".join(f"{v:.2f}" for v in points.positions[k])
            raise ValueError(
                f"the walk of {item.name!r} leaves the room at ({where}) m, "
                f"{points.times[k]:.2f} s in"
            )
    distances = points.times * item.walk.speed_m_s
    # The statistical late method gives every place its own, as cheaply as a mix.
    statistical = (settings or scene.simulation).late.method == "statistical"
    mixing = {} if late_step_m is None or statistical else late_mixing(distances, late_step_m)
    late_from = {
        names[k]: {names[other]: weight for other, weight in parts.items()}
        for k, parts in mixing.items()
    }
    result = simulate(
        walked, settings=settings, threads=threads, progress=progress, late_from=late_from
    )
    late_places = tuple(k for k in range(len(names)) if k not in mixing)
    return WalkResponses(item.name, kind, points, names, result, late_places)


def _boundaries(times: np.ndarray, crossfade_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Where each crossfade from one place's response to the next starts, and how long
    it is: half-way between the places, no longer than the time between them."""
    middles = 0.5 * (times[1:] + times[:-1])
    lengths = np.minimum(crossfade_s, np.diff(times))
    return middles - 0.5 * lengths, lengths


def _past(starts, lengths, k: int, t: np.ndarray) -> np.ndarray:
    """How far the crossfade into place ``k`` has gone at times ``t``, 0 to 1 (1 for the
    first place, which has none): a raised cosine, which starts and ends without a kink,
    so no crossfade is heard as a click."""
    if k == 0:
        return np.ones(len(t))
    if k > len(starts):
        return np.zeros(len(t))
    start, length = starts[k - 1], lengths[k - 1]
    if length <= 0:
        return (t >= start).astype(np.float64)
    return 0.5 - 0.5 * np.cos(np.pi * np.clip((t - start) / length, 0.0, 1.0))


def crossfade_weights(times, t, crossfade_s: float = CROSSFADE_S) -> np.ndarray:
    """Each place's weight (N, len(t)) at times ``t``: 1 while the walker is nearest
    to it, falling to 0 over the crossfade to the next (a raised cosine centred half-way
    between them). They add up to 1."""
    starts, lengths = _boundaries(np.asarray(times, dtype=np.float64), crossfade_s)
    t = np.asarray(t, dtype=np.float64)
    return np.array(
        [_past(starts, lengths, k, t) - _past(starts, lengths, k + 1, t) for k in range(len(times))]
    )


def _sound_powers(walk: WalkResponses) -> dict[str, float]:
    """Sound power (dB re 1 pW) of each source of the walk's simulation, a walking source
    under its own name rather than its places'."""
    places = set(walk.names) if walk.kind == "sources" else set()
    sources = walk.result.room.scene.sources
    return {walk.name if s.name in places else s.name: s.sound_power_db for s in sources}


def render_walk(
    walk: WalkResponses,
    signals: Mapping[str, np.ndarray | tuple[np.ndarray, float]],
    fmt=None,
    *,
    crossfade_s: float = CROSSFADE_S,
    sample_rate: int | None = None,
    seed: int | None = None,
    equal_power: bool = False,
) -> np.ndarray:
    """``signals`` (source names to a signal at ``sample_rate``, or a (signal, rate)
    pair) played along the walk, as (samples, channels) in format ``fmt`` (default
    mono) at ``sample_rate`` (default: the simulation's). As in
    :func:`~acoustic_spaces.render.auralize`, each signal is scaled by its source's
    sound power against the loudest one's, unless ``equal_power``."""
    from acoustic_spaces.spatial import Mono

    if not signals:
        raise ValueError("no signals to render")
    fmt = fmt or Mono()
    result = walk.result
    rate = int(sample_rate or result.settings.sample_rate)
    if seed is None:
        # One random sequence for every place's late part, so that crossfades join
        # tails that differ only in their decay.
        seed = result.settings.seed
        if seed is None:
            seed = next(iter(result)).seed
    if walk.kind == "listeners":
        sources = {s for s, _ in result.responses}
        listener = None
    else:
        (listener,) = {lis for _, lis in result.responses}
        sources = {s for s, _ in result.responses if s not in walk.names} | {walk.name}
    unknown = sorted(set(signals) - sources)
    if unknown:
        raise ValueError(f"no source {unknown[0]!r}; the walk has {', '.join(sorted(sources))}")
    gains = {} if equal_power else power_gains(_sound_powers(walk), signals)
    dry = {}
    for source, value in signals.items():
        signal, signal_rate = value if isinstance(value, tuple) else (value, rate)
        mono = _as_2d(signal, f"signal of {source!r}").mean(axis=1) * gains.get(source, 1.0)
        dry[source] = resample(mono, signal_rate, rate)

    cache: dict[tuple[str, str], np.ndarray] = {}

    def key_of(source: str, k: int) -> tuple[str, str]:
        if walk.kind == "listeners":
            return (source, walk.names[k])
        return (walk.names[k] if source == walk.name else source, listener)

    def response(source: str, k: int) -> np.ndarray:
        key = key_of(source, k)
        if key not in cache:
            cache.clear()  # a place's responses are used one after the other
            cache[key] = result[key].spatial_impulse_response(fmt, rate, seed)
        return cache[key]

    moving = list(dry) if walk.kind == "listeners" else [s for s in dry if s == walk.name]
    still = [s for s in dry if s not in moving]
    taps = max(len(response(s, 0)) for s in dry)
    length = max(len(x) for x in dry.values()) + taps - 1
    out = np.zeros((length, fmt.channels))
    # Outdoors in turbulence the early paths flicker as the eddies drift past, the same
    # eddies all along the walk.
    turbulent = [r for r in result if flickers(r)]
    gusts = gusts_for(turbulent[0], length / rate, seed) if turbulent else None
    for source in still:
        if flickers(result[key_of(source, 0)]):
            end = len(dry[source]) + len(response(source, 0)) - 1
            out[:end] += through_eddies(
                dry[source], result[key_of(source, 0)], fmt, rate, seed, gusts, 0, end
            )
            continue
        wet = convolve(dry[source], response(source, 0))
        out[: len(wet)] += wet
    if not moving:
        return out
    times = walk.points.times
    starts, lengths = _boundaries(times, crossfade_s)
    for k in range(len(times)):
        begin = 0 if k == 0 else int(math.floor(starts[k - 1] * rate))
        end = length if k == len(times) - 1 else int(math.ceil((starts[k] + lengths[k]) * rate)) + 1
        begin, end = max(begin, 0), min(end, length)
        if begin >= end:
            continue  # the signals are over before the walker gets here
        t = np.arange(begin, end) / rate
        weight = (_past(starts, lengths, k, t) - _past(starts, lengths, k + 1, t))[:, None]
        for source in moving:
            if flickers(result[key_of(source, k)]):
                wet = through_eddies(
                    dry[source], result[key_of(source, k)], fmt, rate, seed, gusts, begin, end
                )
                out[begin:end] += weight[: len(wet)] * wet
                continue
            ir = response(source, k)
            first = max(begin - len(ir) + 1, 0)
            piece = dry[source][first:end]
            if len(piece) == 0:
                continue
            wet = convolve(piece, ir)[begin - first : end - first]
            out[begin : begin + len(wet)] += weight[: len(wet)] * wet
    return out


__all__ = [
    "CROSSFADE_S",
    "LATE_STEP_M",
    "STEP_M",
    "WalkPoints",
    "WalkResponses",
    "crossfade_weights",
    "late_mixing",
    "render_walk",
    "simulate_walk",
    "walk_points",
]
