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

At walking speed there is no Doppler shift (it is under half a percent, 8 cents). A
walk with ``doppler: true`` (a car, a train, an aircraft) plays each early path through
a delay that changes as the walker goes (:mod:`acoustic_spaces.doppler`), so the pitch
of each shifts as it does on its way; the late part is crossfaded as without it, and
:func:`pass_by` gives the level over time, its maximum and the sound exposure level.
Outdoors in turbulence the early sound flickers along the walk as in
:func:`~acoustic_spaces.render.auralize`, through the same eddies from place to place.

A walk may instead give ``keyframes``: where the walker is and how it faces at given
times after its start, between which it moves and turns evenly, so that it can wait,
turn on the spot or go at any pace (:func:`knots` gives either kind of walk as the knots
it moves between). Several listeners and sources may move at once, each on its own
walk, all on one clock: :func:`simulate_movement` simulates them together, every mover
as a copy at each place it takes, in one simulation; :func:`render_movement` plays the
signals through it as heard at every listener in use, still or moving; and
:func:`save_movement` writes one WAV per listener. A walk with Doppler moves alone, there
as in a walk of its own: every listener hears each early path through its changing
delay. ``walk.enabled: false`` keeps a walk
but leaves its walker standing still, and :func:`with_moving` picks which walks are on
for one run, so that one scene serves both still and moving.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

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
from acoustic_spaces.scene.model import Orientation
from acoustic_spaces.scene.model import Simulation as SimulationSettings

#: Distance (m) between the places a walk is simulated at.
STEP_M = 0.25
#: Angle (degrees) a walker turns between the places of a walk along keyframes where it
#: turns without moving on (on the spot, or nearly).
STEP_DEG = 20.0
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
    #: Along keyframes, how it faces at each place (N, 3: yaw, pitch and roll in degrees).
    angles: np.ndarray | None = None


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
    #: Whether the early paths go through delays that change as it goes (``walk.doppler``).
    doppler: bool = False

    @property
    def duration_s(self) -> float:
        return float(self.points.times[-1])


def _walker(room: Room, name: str | None):
    """The listener or source that walks: ``name``, or the only one with a walk that is
    enabled."""
    scene = room.scene
    found = [
        (kind, item)
        for kind in ("listeners", "sources")
        for item in getattr(scene, kind)
        if (item.name == name if name is not None else item.walk is not None and item.walk.enabled)
    ]
    if name is not None and not found:
        raise ValueError(f"no listener or source named {name!r}")
    if not found:
        raise ValueError("no listener or source walks: give one a walk")
    if len(found) > 1:
        names = ", ".join(item.name for _, item in found)
        raise ValueError(
            f"more than one walks ({names}): pick one, or play them together (--all, "
            "simulate_movement)"
        )
    kind, item = found[0]
    if item.walk is None:
        raise ValueError(f"{item.name!r} has no walk")
    if not item.walk.enabled:
        raise ValueError(f"the walk of {item.name!r} is off (enabled: false): it stands still")
    return kind, item


def _wrapped(angles: np.ndarray) -> np.ndarray:
    """Angles (degrees) in (-180, 180]."""
    return 180.0 - (180.0 - np.asarray(angles, dtype=np.float64)) % 360.0


def _angles(orientation: Orientation) -> np.ndarray:
    return np.array([orientation.yaw, orientation.pitch, orientation.roll], dtype=np.float64)


@dataclass(frozen=True)
class Knots:
    """The knots of a walk: where the walker is and how it faces at each, moving and
    turning evenly from one to the next (:func:`knots`)."""

    #: When it is at each knot (s from the start; the first is 0).
    times: np.ndarray
    #: Where each knot is (K, 3).
    positions: np.ndarray
    #: How it faces at each knot (K, 3: yaw, pitch and roll in degrees), unwrapped so that
    #: it turns the shorter way round from one to the next.
    angles: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.times[-1])

    @property
    def length_m(self) -> float:
        """How far the walk goes."""
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    @property
    def top_speed(self) -> float:
        """The fastest it goes (m/s), over a leg."""
        legs = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        taken = np.diff(self.times)
        return float(np.max(legs[taken > 0] / taken[taken > 0], initial=0.0))

    def at(self, times) -> tuple[np.ndarray, np.ndarray]:
        """Where it is and how it faces (yaw, pitch and roll in (-180, 180]) at ``times``
        (s, a number or an array): at its start before, at its end after."""
        t = np.clip(np.asarray(times, dtype=np.float64), 0.0, self.times[-1])
        positions = np.stack([np.interp(t, self.times, self.positions[:, k]) for k in range(3)], -1)
        angles = np.stack([np.interp(t, self.times, self.angles[:, k]) for k in range(3)], -1)
        return positions, _wrapped(angles)


def knots(item) -> Knots:
    """The knots of ``item``'s walk (a scene listener or source): along ``to``, its
    corners at ``speed_m_s`` and, facing forward, where each turn through a corner starts
    and ends; along ``keyframes``, its position at 0 s and each keyframe, facing as it
    says or as before."""
    walk = item.walk
    if walk is None:
        raise ValueError(f"{item.name!r} has no walk")
    own = _angles(item.orientation)
    if walk.keyframes is not None:
        times = np.array([0.0, *(frame.time_s for frame in walk.keyframes)])
        positions = np.array(
            [item.position, *(frame.position for frame in walk.keyframes)], dtype=np.float64
        )
        angles, facing = [own], own
        for frame in walk.keyframes:
            if frame.orientation is not None:
                facing = _angles(frame.orientation)
            angles.append(facing)
        unwrapped = np.degrees(np.unwrap(np.radians(np.array(angles)), axis=0))
        return Knots(times, positions, unwrapped)
    corners = np.array([item.position, *walk.to], dtype=np.float64)
    legs = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(legs)])
    if along[-1] == 0.0:
        raise ValueError(f"the walk of {item.name!r} goes nowhere: it never leaves its position")
    if not walk.face_forward:
        return Knots(along / walk.speed_m_s, corners, np.tile(own, (len(corners), 1)))
    heading = np.diff(corners, axis=0)[:, :2]
    flat = np.hypot(heading[:, 0], heading[:, 1]) < 1e-9
    yaws = np.degrees(np.arctan2(heading[:, 1], heading[:, 0]))
    # Straight up or down (a lift): the way it faced before.
    last = item.orientation.yaw
    for k in range(len(yaws)):
        if flat[k]:
            yaws[k] = last
        last = yaws[k]
    # From each leg's heading to the next's the shorter way round, ...
    turns = (np.diff(yaws) + 180.0) % 360.0 - 180.0
    yaws = yaws[0] + np.concatenate([[0.0], np.cumsum(turns)])
    # ... at turn_deg_s through each corner: half of the turn before it, half after
    # it, and no further than half-way along either leg.
    along_turn, yaw_turn = [0.0], [yaws[0]]
    for j, turn in enumerate(turns, start=1):
        if turn == 0.0:
            continue
        half = 0.5 * walk.speed_m_s * abs(turn) / walk.turn_deg_s
        half = min(half, 0.5 * legs[j - 1], 0.5 * legs[j])
        along_turn += [along[j] - half, along[j] + half]
        yaw_turn += [yaws[j - 1], yaws[j]]
    along_turn.append(along[-1])
    yaw_turn.append(yaws[-1])
    # The knots: the corners and the turns' ends, the yaw at each from the turns.
    distances = np.unique(np.concatenate([along, along_turn]))
    positions = np.column_stack([np.interp(distances, along, corners[:, k]) for k in range(3)])
    angles = np.tile(own, (len(distances), 1))
    angles[:, 0] = np.interp(distances, along_turn, yaw_turn)
    return Knots(distances / walk.speed_m_s, positions, angles)


def _sample_times(path: Knots, step_m: float, step_deg: float) -> np.ndarray:
    """When to place a walker along ``path``: at its knots, and along each leg every
    ``step_m`` or a little less and every ``step_deg`` of its turn or a little less,
    whichever gives more places (a wait on the spot gives none)."""
    times = [path.times]
    legs = np.linalg.norm(np.diff(path.positions, axis=0), axis=1)
    turns = np.abs(np.diff(path.angles, axis=0)).max(axis=1)
    for j, (leg, turn) in enumerate(zip(legs, turns, strict=True)):
        count = max(int(math.ceil(leg / step_m)), int(math.ceil(turn / step_deg)))
        if count > 1:
            times.append(np.linspace(path.times[j], path.times[j + 1], count + 1)[1:-1])
    return np.unique(np.concatenate(times))


def walk_points(item, step_m: float = STEP_M, step_deg: float = STEP_DEG) -> WalkPoints:
    """The places along ``item``'s walk (a scene listener or source), ``step_m`` or a
    little less apart, from its position to the walk's last point; along keyframes also
    every ``step_deg`` of a turn on the spot."""
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    if step_deg <= 0:
        raise ValueError("step_deg must be positive")
    path = knots(item)
    if item.walk.keyframes is not None:
        times = _sample_times(path, step_m, step_deg)
        positions, angles = path.at(times)
        return WalkPoints(times, positions, angles[:, 0], angles)
    # Along ``to``, evenly spaced over the whole path, as its legs join without a stop.
    along = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path.positions, axis=0), axis=1))]
    )
    count = max(int(math.ceil(along[-1] / step_m)) + 1, 2)
    s = np.linspace(0.0, along[-1], count)
    positions, angles = path.at(np.interp(s, along, path.times))
    yaws = angles[:, 0] if item.walk.face_forward else None
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
        # A place where a traced one is (a turn or a wait on the spot) takes its late part.
        j = int(np.searchsorted(distances[traced], d, side="right"))
        if j == 0:
            mixing[k] = {int(traced[0]): 1.0}
        elif j == len(traced) or distances[traced[j - 1]] == d:
            mixing[k] = {int(traced[j - 1]): 1.0}
        else:
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
    doppler = item.walk.doppler
    if doppler:
        _check_doppler(room, item, knots(item).top_speed, settings)
    # Numbered, as places less than 10 ms apart would share a time.
    names = tuple(f"{item.name} #{k} @ {t:.2f} s" for k, t in enumerate(points.times))
    copies = [
        _copy_at(item, label, place, None if points.angles is None else points.angles[k], yaw)
        for k, (place, label, yaw) in enumerate(
            zip(
                points.positions,
                names,
                points.yaws if points.yaws is not None else [None] * len(names),
                strict=True,
            )
        )
    ]
    scene = room.scene
    if kind == "listeners":
        update = {"listeners": copies}
    else:
        listeners = [lis for lis in scene.listeners if lis.in_use]
        if listener is not None:
            listeners = [lis for lis in scene.listeners if lis.name == listener]
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
    mixing = _late_mixing_of(points.positions, late_step_m, settings or scene.simulation)
    late_from = {
        names[k]: {names[other]: weight for other, weight in parts.items()}
        for k, parts in mixing.items()
    }
    # A walking source's places share the parabolic equation's runs where they can.
    result = simulate(
        walked,
        settings=settings,
        threads=threads,
        progress=progress,
        late_from=late_from,
        share_waves=names if kind == "sources" else None,
    )
    late_places = tuple(k for k in range(len(names)) if k not in mixing)
    return WalkResponses(item.name, kind, points, names, result, late_places, doppler)


def _check_doppler(room: Room, item, speed: float, settings: SimulationSettings | None) -> None:
    """What a walk with Doppler cannot do: go as fast as sound, or take the wave solver."""
    c = room.speed_of_sound()
    if speed >= c:
        raise ValueError(
            f"{item.name!r} walks at {speed:g} m/s, as fast as sound or "
            f"faster ({c:.0f} m/s): Doppler needs it slower"
        )
    if (settings or room.scene.simulation).wave.enabled:
        raise ValueError(
            "a walk with Doppler does not take the wave solver's bands: turn "
            "simulation.wave off, or the walk's doppler"
        )


def _copy_at(item, label: str, place, angles=None, yaw: float | None = None):
    """``item`` standing at ``place`` as ``label``, facing ``angles`` (yaw, pitch, roll)
    or turned to ``yaw``, or as it does itself. The walk is asked for by name: its
    places are in use."""
    update = {"name": label, "position": tuple(place), "walk": None, "enabled": None}
    if angles is not None:
        yaw, pitch, roll = (float(a) for a in angles)
        update["orientation"] = Orientation(yaw=yaw, pitch=pitch, roll=roll)
    elif yaw is not None:
        update["orientation"] = item.orientation.model_copy(update={"yaw": float(yaw)})
    return item.model_copy(update=update)


def _late_mixing_of(
    positions: np.ndarray, late_step_m: float | None, settings: SimulationSettings
) -> dict[int, dict[int, float]]:
    """:func:`late_mixing` over the places ``positions``, by the distance along them;
    nothing with ``late_step_m`` None or the statistical late method, which gives every
    place its own as cheaply as a mix."""
    if late_step_m is None or settings.late.method == "statistical":
        return {}
    distances = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    )
    if distances[-1] == 0.0:
        return {k: {0: 1.0} for k in range(1, len(distances))}
    return late_mixing(distances, late_step_m)


# -- several movers on one clock -----------------------------------------------------


def movers(scene) -> list[tuple[str, object]]:
    """The listeners and sources of ``scene`` that move: in use, with a walk that is
    enabled, as ("listeners" or "sources", item), listeners first."""
    return [
        (kind, item)
        for kind in ("listeners", "sources")
        for item in getattr(scene, kind)
        if item.in_use and item.walk is not None and item.walk.enabled
    ]


def with_moving(scene, names: Iterable[str] | None):
    """``scene`` with only ``names`` moving: their walks turned on and every other walk
    off, so that its walker stands where it starts. None keeps the scene's own. Each
    name must be a source or listener in use with a walk."""
    if names is None:
        return scene
    names = set(names)
    walkers = [
        item.name
        for kind in ("listeners", "sources")
        for item in getattr(scene, kind)
        if item.in_use and item.walk is not None
    ]
    unknown = sorted(names - set(walkers))
    if unknown:
        have = ", ".join(walkers) if walkers else "nothing"
        raise ValueError(f"{unknown[0]!r} has no walk to move along; the scene's walks: {have}")
    update = {}
    for kind in ("listeners", "sources"):
        items = []
        for item in getattr(scene, kind):
            on = item.name in names
            if item.walk is not None and item.walk.enabled != on:
                item = item.model_copy(
                    update={"walk": item.walk.model_copy(update={"enabled": on})}
                )
            items.append(item)
        update[kind] = items
    return scene.model_copy(update=update)


@dataclass(frozen=True)
class Movement:
    """Responses with every mover at each moment of its walk, from
    :func:`simulate_movement`."""

    #: The scene's room, as it was given (the movers with their walks).
    room: Room
    #: The moments the responses are at (s from the start, the first 0).
    times: np.ndarray
    #: The simulation: the still sources and listeners under their names, each mover as
    #: a copy at each place it takes (:attr:`copies`).
    result: SimulationResult
    #: Each mover (("listeners" or "sources", name)) and the name of its copy at each
    #: moment of :attr:`times`.
    copies: dict[tuple[str, str], tuple[str, ...]]
    #: How many of the movers' places have a late part of their own; the others mix theirs.
    late_places: int = 0
    #: Whether the mover, then alone, walks with Doppler (``walk.doppler``): each pair it
    #: is part of plays its early paths through delays that change as it goes.
    doppler: bool = False

    @property
    def duration_s(self) -> float:
        return float(self.times[-1])

    @property
    def places(self) -> int:
        """How many places the movers were simulated at in all."""
        return sum(len(set(names)) for names in self.copies.values())

    def key(self, source: str, listener: str, k: int) -> tuple[str, str]:
        """The response of ``source`` at ``listener`` at moment ``k``, in :attr:`result`."""
        source = self.copies.get(("sources", source), (source,) * len(self.times))[k]
        listener = self.copies.get(("listeners", listener), (listener,) * len(self.times))[k]
        return source, listener

    def moves(self, source: str, listener: str) -> bool:
        """Whether either of the pair moves."""
        return ("sources", source) in self.copies or ("listeners", listener) in self.copies


def simulate_movement(
    room: Room,
    *,
    step_m: float = STEP_M,
    step_deg: float = STEP_DEG,
    late_step_m: float | None = LATE_STEP_M,
    settings: SimulationSettings | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> Movement:
    """Simulate the scene with every mover (:func:`movers`) on one clock, in one
    simulation: the moments are every mover's places (every ``step_m`` along its walk,
    every ``step_deg`` of a turn on the spot, and its knots), and at each moment each
    mover stands as a copy where its walk has it then, a copy serving every moment it
    stands still at (once its walk is over, say). Every listener in use hears every
    source in use, still or moving. The late part is traced every ``late_step_m`` along
    each mover's walk, as in :func:`simulate_walk`.

    A walk with Doppler moves alone: with another mover it is refused. It must go slower
    than sound, without the wave solver, as in :func:`simulate_walk`.
    """
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    if step_deg <= 0:
        raise ValueError("step_deg must be positive")
    scene = room.scene
    moving = movers(scene)
    if not moving:
        raise ValueError("nothing moves: give a listener or a source a walk that is enabled")
    paths = {item.name: knots(item) for _, item in moving}
    doppler = next((item for _, item in moving if item.walk.doppler), None)
    if doppler is not None:
        if len(moving) > 1:
            others = ", ".join(item.name for _, item in moving if item is not doppler)
            raise ValueError(
                f"{doppler.name!r} walks with Doppler, which moves alone: leave {others} "
                f"standing (turn their walks off), or turn its doppler off"
            )
        _check_doppler(room, doppler, paths[doppler.name].top_speed, settings)
    times = np.unique(np.concatenate([_sample_times(k, step_m, step_deg) for k in paths.values()]))
    copies: dict[tuple[str, str], tuple[str, ...]] = {}
    added: dict[str, list] = {"listeners": [], "sources": []}
    mixing: dict[str, dict[str, float]] = {}
    late_places = 0
    for kind, item in moving:
        positions, angles = paths[item.name].at(times)
        # One copy per distinct place: the same while it stands still.
        places: list[int] = []  # the moment each place was first seen at
        at: list[int] = []  # each moment's place
        for k in range(len(times)):
            for j, first in enumerate(places):
                if np.array_equal(positions[k], positions[first]) and np.array_equal(
                    angles[k], angles[first]
                ):
                    at.append(j)
                    break
            else:
                at.append(len(places))
                places.append(k)
        names = tuple(f"{item.name} #{j} @ {times[first]:.2f} s" for j, first in enumerate(places))
        copies[kind, item.name] = tuple(names[j] for j in at)
        added[kind] += [
            _copy_at(item, names[j], positions[first], angles[first])
            for j, first in enumerate(places)
        ]
        parts = _late_mixing_of(positions[places], late_step_m, settings or scene.simulation)
        mixing |= {
            names[j]: {names[other]: weight for other, weight in mix.items()}
            for j, mix in parts.items()
        }
        late_places += len(places) - len(parts)
    moving_names = {item.name for _, item in moving}
    still = {
        kind: [item for item in getattr(scene, kind) if item.name not in moving_names]
        for kind in ("listeners", "sources")
    }
    update = {kind: [*still[kind], *added[kind]] for kind in ("listeners", "sources")}
    walked = Room.from_scene(scene.model_copy(update=update), reuse=room)
    for kind, inside in (
        ("listeners", walked.listeners_inside),
        ("sources", walked.sources_inside),
    ):
        if inside is None:
            continue
        offset = len(still[kind])
        for j, copy in enumerate(added[kind]):
            if not inside[offset + j]:
                where = ", ".join(f"{v:.2f}" for v in copy.position)
                name, _, when = copy.name.rpartition(" @ ")
                raise ValueError(
                    f"the walk of {name.rpartition(' #')[0]!r} leaves the room at ({where}) m, "
                    f"{when} in"
                )
    # The moving sources' places share the parabolic equation's runs where they can.
    result = simulate(
        walked,
        settings=settings,
        threads=threads,
        progress=progress,
        late_from=mixing,
        share_waves=[copy.name for copy in added["sources"]],
    )
    return Movement(room, times, result, copies, late_places, doppler is not None)


def render_movement(
    movement: Movement,
    signals: Mapping[str, np.ndarray | tuple[np.ndarray, float]],
    fmt=None,
    *,
    crossfade_s: float = CROSSFADE_S,
    sample_rate: int | None = None,
    seed: int | None = None,
    equal_power: bool = False,
) -> dict[str, np.ndarray]:
    """``signals`` (source names to a signal at ``sample_rate``, or a (signal, rate)
    pair) played through the movement, as heard at every listener in use: each as
    (samples, channels) in format ``fmt`` (default mono) at ``sample_rate`` (default: the
    simulation's), all of one length and, as in :func:`render_walk`, each signal scaled
    by its source's sound power against the loudest one's unless ``equal_power``. A pair
    that moves crossfades from each moment's response to the next's, as a walk does, or
    with Doppler (:attr:`Movement.doppler`) plays each early path through its changing
    delay and crossfades the late part, as :func:`render_walk` does; a still pair plays
    through its one response."""
    from acoustic_spaces.spatial import Mono

    if not signals:
        raise ValueError("no signals to render")
    fmt = fmt or Mono()
    result = movement.result
    scene = movement.room.scene
    rate = int(sample_rate or result.settings.sample_rate)
    if seed is None:
        # One random sequence for every place's late part, so that crossfades join
        # tails that differ only in their decay.
        seed = result.settings.seed
        if seed is None:
            seed = next(iter(result)).seed
    sources = {s.name for s in scene.sources if s.in_use}
    unknown = sorted(set(signals) - sources)
    if unknown:
        raise ValueError(f"no source {unknown[0]!r}; the scene has {', '.join(sorted(sources))}")
    listeners = [lis.name for lis in scene.listeners if lis.in_use]
    if not listeners:
        raise ValueError("no listener in use to hear the movement")
    gains = {} if equal_power else power_gains(scene.sources, signals)
    dry = {}
    for source, value in signals.items():
        signal, signal_rate = value if isinstance(value, tuple) else (value, rate)
        mono = _as_2d(signal, f"signal of {source!r}").mean(axis=1) * gains.get(source, 1.0)
        dry[source] = resample(mono, signal_rate, rate)
    taps = int(math.ceil(max(r.length_s for r in result) * rate))
    length = max(len(x) for x in dry.values()) + taps - 1
    # Outdoors in turbulence the early paths flicker as the eddies drift past, the same
    # eddies all along.
    turbulent = [r for r in result if flickers(r)]
    gusts = gusts_for(turbulent[0], length / rate, seed) if turbulent else None
    times = movement.times
    starts, lengths = _boundaries(times, crossfade_s)
    cache: dict[tuple[str, str], np.ndarray] = {}
    # Where each mover's copies stand, for the steps of a walk with Doppler.
    places = {
        item.name: item.position
        for kind in ("listeners", "sources")
        for item in getattr(result.room.scene, kind)
    }

    def response(key: tuple[str, str]) -> np.ndarray:
        if key not in cache:
            if len(cache) > 2 * len(dry):
                cache.clear()  # a moment's responses are used one after the other
            cache[key] = result[key].spatial_impulse_response(fmt, rate, seed)
        return cache[key]

    outputs = {}
    for listener in listeners:
        out = np.zeros((length, fmt.channels))
        for source, signal in dry.items():
            if not movement.moves(source, listener):
                key = movement.key(source, listener, 0)
                if flickers(result[key]):
                    end = len(signal) + len(response(key)) - 1
                    out[:end] += through_eddies(signal, result[key], fmt, rate, seed, gusts, 0, end)
                else:
                    wet = convolve(signal, response(key))
                    out[: len(wet)] += wet
                continue
            if movement.doppler:
                keys = [movement.key(source, listener, k) for k in range(len(times))]
                walks = ("sources", source) in movement.copies
                positions = np.array([places[key[0] if walks else key[1]] for key in keys])
                _doppler(
                    result,
                    keys,
                    times,
                    positions,
                    signal,
                    out,
                    fmt,
                    rate,
                    seed,
                    gusts,
                    source_walks=walks,
                    crossfade_s=crossfade_s,
                )
                continue
            for k in range(len(times)):
                begin = 0 if k == 0 else int(math.floor(starts[k - 1] * rate))
                end = (
                    length
                    if k == len(times) - 1
                    else int(math.ceil((starts[k] + lengths[k]) * rate)) + 1
                )
                begin, end = max(begin, 0), min(end, length)
                if begin >= end:
                    continue  # the signals are over before this moment
                t = np.arange(begin, end) / rate
                weight = (_past(starts, lengths, k, t) - _past(starts, lengths, k + 1, t))[:, None]
                key = movement.key(source, listener, k)
                if flickers(result[key]):
                    wet = through_eddies(signal, result[key], fmt, rate, seed, gusts, begin, end)
                    out[begin:end] += weight[: len(wet)] * wet
                    continue
                ir = response(key)
                first = max(begin - len(ir) + 1, 0)
                piece = signal[first:end]
                if len(piece) == 0:
                    continue
                wet = convolve(piece, ir)[begin - first : end - first]
                out[begin : begin + len(wet)] += weight[: len(wet)] * wet
        outputs[listener] = out
    return outputs


def save_movement(
    rendered: Mapping[str, np.ndarray],
    out: str | Path,
    rate: int,
    fmt=None,
    *,
    normalize: bool = False,
) -> tuple[dict[str, Path], float]:
    """Write what :func:`render_movement` gave: ``out`` itself for one listener, else
    ``<out stem>_<listener>.wav`` beside it for each (with ``-2``, ``-3``... where two
    listeners' names would give the same file); ``normalize`` brings the loudest
    peak of them all to -1 dBFS, one gain for all, so that the balance between the
    listeners holds. Gives each listener's file and the gain applied."""
    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.results import unique_stems

    if not rendered:
        raise ValueError("nothing rendered")
    out = Path(out)
    peak = max(float(np.abs(x).max(initial=0.0)) for x in rendered.values())
    gain = 10.0 ** (-1.0 / 20.0) / peak if normalize and peak > 0 else 1.0
    mask = fmt.channel_mask() if fmt is not None and hasattr(fmt, "channel_mask") else None
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix or ".wav"
    # "A/B" and "A_B" would write one file: the second gets -2.
    stems = unique_stems(
        [
            f"{out.stem}_" + "".join(c if c.isalnum() or c in "-_." else "_" for c in listener)
            for listener in rendered
        ],
        (suffix,),
    )
    paths = {}
    for (listener, data), stem in zip(rendered.items(), stems, strict=True):
        path = out if len(rendered) == 1 else out.with_name(stem + suffix)
        write_wav(path, data * gain, rate, mask)
        paths[listener] = path
    return paths, gain


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
    if walk.doppler:
        # The paths' delays change along the walk: as long as the longest response.
        longest = max(result[key_of(s, k)].length_s for s in moving for k in range(len(walk.names)))
        taps = max(taps, int(math.ceil(longest * rate)))
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
        if walk.doppler:
            break  # the late parts below, the early paths through their delays
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
    if walk.doppler:
        for source in moving:
            _doppler(
                result,
                [key_of(source, k) for k in range(len(times))],
                times,
                walk.points.positions,
                dry[source],
                out,
                fmt,
                rate,
                seed,
                gusts,
                source_walks=walk.kind == "sources",
                crossfade_s=crossfade_s,
            )
    return out


def _doppler(
    result,
    keys,
    times,
    positions,
    signal,
    out,
    fmt,
    rate,
    seed,
    gusts,
    *,
    source_walks: bool,
    crossfade_s: float,
) -> None:
    """Adds to ``out`` the mono ``signal`` heard along a walk with Doppler, through
    ``keys`` (the pair's response in ``result`` at each place, the walker there at
    ``times`` and ``positions``; ``source_walks`` when the walker is the source): the late
    parts crossfaded from place to place, the early paths each through its own changing
    delay (:func:`acoustic_spaces.doppler.render`)."""
    from acoustic_spaces import doppler
    from acoustic_spaces.engine.early import EarlyReflections

    length = len(out)
    starts, lengths = _boundaries(times, crossfade_s)
    late: dict[tuple[str, str], np.ndarray] = {}
    for k in range(len(times)):
        begin = 0 if k == 0 else int(math.floor(starts[k - 1] * rate))
        end = length if k == len(times) - 1 else int(math.ceil((starts[k] + lengths[k]) * rate)) + 1
        begin, end = max(begin, 0), min(end, length)
        if begin >= end:
            continue
        t = np.arange(begin, end) / rate
        weight = (_past(starts, lengths, k, t) - _past(starts, lengths, k + 1, t))[:, None]
        if keys[k] not in late:
            late.clear()  # a place's response is used one after the other
            chosen = result[keys[k]]
            chosen = dataclasses.replace(chosen, early=EarlyReflections.empty(chosen.bands))
            late[keys[k]] = chosen.spatial_impulse_response(fmt, rate, seed)
        ir = late[keys[k]]
        first = max(begin - len(ir) + 1, 0)
        piece = signal[first:end]
        if len(piece) == 0:
            continue
        wet = convolve(piece, ir)[begin - first : end - first]
        out[begin : begin + len(wet)] += weight[: len(wet)] * wet
    responses = [result[key] for key in keys]
    steps = np.linalg.norm(np.diff(np.asarray(positions, dtype=float), axis=0), axis=1)
    # The most a path's delay moves from one place to the next and stays the same path's:
    # the step over the speed of sound, with room to spare.
    step_s = 1.5 * float(steps.max(initial=0.0)) / responses[0].speed_of_sound + 1e-4
    out += doppler.render(
        signal,
        responses,
        times,
        fmt,
        rate,
        length,
        source_walks=source_walks,
        planes=doppler.plane_ids(result.room.mesh),
        step_s=step_s,
        gusts=gusts,
        seed=seed,
    )


@dataclass(frozen=True)
class PassBy:
    """The level at a listener along a walk, from :func:`pass_by`: as a source passes
    it, or as it passes a source."""

    source: str
    listener: str
    #: (N,) when the sound of each place is heard (s from the start): for a walking
    #: source, when it is there plus its direct sound's delay; for a walking listener,
    #: when it is there.
    times: np.ndarray
    #: (N, 8) the sound pressure level per octave band (dB) at each place, of the
    #: source's sound power, as it would be with the source standing there.
    levels: np.ndarray
    #: (N,) the same, A-weighted (dB(A)).
    levels_a: np.ndarray

    @property
    def duration_s(self) -> float:
        """How long the walk is heard for (s), from the first place's sound to the
        last's."""
        return float(self.times[-1] - self.times[0])

    @property
    def max_a(self) -> float:
        """The highest A-weighted level along the walk (dB(A)), L_Amax."""
        return float(np.max(self.levels_a))

    @property
    def max_time(self) -> float:
        """When the highest A-weighted level is heard (s from the start)."""
        return float(self.times[int(np.argmax(self.levels_a))])

    @property
    def exposure(self) -> np.ndarray:
        """(8,) the sound exposure level per octave band (dB): the level that, held for
        one second, carries the energy of the whole walk."""
        return _exposure(self.times, self.levels)

    @property
    def exposure_a(self) -> float:
        """The A-weighted sound exposure level (dB(A)), SEL or L_AE."""
        return float(_exposure(self.times, self.levels_a[:, None])[0])

    @property
    def equivalent_a(self) -> float:
        """The A-weighted equivalent level over the walk (dB(A)), L_Aeq,T with T its
        :attr:`duration_s`."""
        return self.exposure_a - 10.0 * math.log10(max(self.duration_s, 1e-12))


def _exposure(times: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """10 lg of the integral over ``times`` (trapezoids) of 10^(L/10), per column."""
    power = 10.0 ** (np.asarray(levels, dtype=float) / 10.0)
    if len(times) < 2:
        return np.full(power.shape[1], -np.inf)
    energy = np.sum(0.5 * (power[1:] + power[:-1]) * np.diff(times)[:, None], axis=0)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(energy)


def pass_by(walk: WalkResponses, source: str | None = None) -> PassBy:
    """The level along ``walk`` at its listener (a walking source) or of ``source`` (a
    walking listener; by default the only source): each place's level, of the source's
    sound power, at the time it is heard (quasi-static: as with the source standing
    there), with its maximum, the sound exposure level and the equivalent level."""
    from acoustic_spaces.parameters import a_weighted

    result = walk.result
    scene = result.room.scene
    if walk.kind == "sources":
        (listener,) = {lis for _, lis in result.responses}
        if source is not None and source != walk.name:
            raise ValueError(f"{source!r} does not walk; {walk.name!r} does")
        source = walk.name
        keys = [(name, listener) for name in walk.names]
        powers = {s.name: s.power_db("octave") for s in scene.sources}
        power = [powers[name] for name in walk.names]
    else:
        sources = sorted({s for s, _ in result.responses})
        if source is None:
            if len(sources) != 1:
                raise ValueError(f"pick a source: {', '.join(sources)}")
            source = sources[0]
        if source not in sources:
            raise ValueError(f"no source {source!r}; the walk has {', '.join(sources)}")
        listener = walk.name
        keys = [(source, name) for name in walk.names]
        level = next(s for s in scene.sources if s.name == source).power_db("octave")
        power = [level] * len(keys)
    responses = [result[key] for key in keys]
    times = np.asarray(walk.points.times, dtype=float)
    if walk.kind == "sources":
        times = times + np.array([r.direct_delay_s for r in responses])
    levels = np.array([r.levels(p) for r, p in zip(responses, power, strict=True)])
    levels_a = np.array([a_weighted(row) for row in levels])
    return PassBy(source, listener, times, levels, levels_a)


__all__ = [
    "CROSSFADE_S",
    "LATE_STEP_M",
    "STEP_DEG",
    "STEP_M",
    "Knots",
    "Movement",
    "PassBy",
    "WalkPoints",
    "WalkResponses",
    "crossfade_weights",
    "knots",
    "late_mixing",
    "movers",
    "pass_by",
    "render_movement",
    "render_walk",
    "save_movement",
    "simulate_movement",
    "simulate_walk",
    "walk_points",
    "with_moving",
]
