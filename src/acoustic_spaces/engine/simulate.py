"""Run the early and late stages for every source and listener of a room."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import late as late_stage
from acoustic_spaces.engine.model import DIRECTIVITY_A, TraceModel
from acoustic_spaces.engine.response import Response
from acoustic_spaces.geometry import distance_to_surface
from acoustic_spaces.parameters import RoomParameters
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Scene
from acoustic_spaces.scene.model import Simulation as SimulationSettings
from acoustic_spaces.spatial.directions import fibonacci_sphere

#: Slot width of the late energy (s).
TIME_STEP = 1e-3


def mixing_time(volume_m3: float | None) -> float:
    """Time after the direct sound when the late field is diffuse (s): sqrt(V) ms."""
    return 1e-3 * math.sqrt(volume_m3) if volume_m3 else 0.02


@dataclass
class SimulationResult:
    """Responses for every (source, listener) pair, plus run details."""

    room: Room
    settings: SimulationSettings
    responses: dict[tuple[str, str], Response]
    #: Per source: rays, escaped rays, receiver radius, seed and timings.
    details: dict[str, dict[str, float]] = field(default_factory=dict)

    def parameters(self) -> dict[tuple[str, str], RoomParameters]:
        """ISO 3382-1 parameters of every response."""
        return {key: r.parameters() for key, r in self.responses.items()}

    def __getitem__(self, key: tuple[str, str]) -> Response:
        return self.responses[key]

    def __iter__(self):
        return iter(self.responses.values())

    def __len__(self) -> int:
        return len(self.responses)


def _seed(base: int, index: int) -> int:
    return int(np.random.SeedSequence([base, index]).generate_state(1, np.uint64)[0])


def _as_room(room: Room | Scene | str | Path) -> Room:
    if isinstance(room, Room):
        return room
    if isinstance(room, Scene):
        return Room.from_scene(room)
    return Room.from_yaml(room)


def _pick(items, names, kind):
    if names is None:
        return list(items)
    by_name = {item.name: item for item in items}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"unknown {kind}: {', '.join(missing)}")
    return [by_name[n] for n in names]


def _candidates(
    model: TraceModel, source: np.ndarray, method: str, order: int, rays: int, threads: int
):
    if method == "image_sources":
        return early_stage.exhaustive_candidates(model, source, order)
    return early_stage.ray_traced_candidates(model, source, order, rays, threads)


def early_paths(
    room: Room | Scene | str | Path,
    source: str | None = None,
    listener: str | None = None,
    *,
    max_order: int | None = None,
    method: str | None = None,
    threads: int = 0,
) -> early_stage.EarlyReflections:
    """Specular paths from one source to one listener, without the late part.

    Names default to the first source and listener; ``max_order`` and ``method``
    default to ``scene.simulation.early``. Useful for drawing paths.
    """
    room = _as_room(room)
    scene = room.scene
    cfg = scene.simulation.early
    src = _pick(scene.sources, None if source is None else [source], "source")[0]
    lis = _pick(scene.listeners, None if listener is None else [listener], "listener")[0]
    model = TraceModel.from_room(room)
    position = np.array(src.position, dtype=float)
    order = cfg.max_order if max_order is None else max_order
    candidates = _candidates(model, position, method or cfg.method, order, cfg.rays, threads)
    return early_stage.find_paths(
        model,
        position,
        np.array(lis.position, dtype=float),
        candidates,
        speed_of_sound=room.speed_of_sound(),
        air=room.air_attenuation(),
        max_delay_s=scene.simulation.ir_length_s,
        directivity_a=DIRECTIVITY_A[src.directivity],
        forward=np.array(src.orientation.forward()),
    )


def simulate(
    room: Room | Scene | str | Path,
    *,
    settings: SimulationSettings | None = None,
    sources: Sequence[str] | None = None,
    listeners: Sequence[str] | None = None,
    threads: int = 0,
    progress: Callable[[str], None] | None = None,
) -> SimulationResult:
    """Simulate the room's impulse responses as set in ``scene.simulation``.

    ``room`` may be a :class:`Room`, a :class:`Scene` or a scene file. ``sources`` and
    ``listeners`` pick names from the scene (all by default). ``threads=0`` uses every
    core. Raises ``ValueError`` if :meth:`Room.problems` reports anything.
    """
    room = _as_room(room)
    problems = room.problems()
    if problems:
        raise ValueError("; ".join(problems))
    scene = room.scene
    settings = settings or scene.simulation
    say = progress or (lambda _: None)

    chosen_sources = _pick(scene.sources, sources, "source")
    chosen_listeners = _pick(scene.listeners, listeners, "listener")

    model = TraceModel.from_room(room)
    c = room.speed_of_sound()
    air = room.air_attenuation()
    volume = room.volume_m3
    length = settings.ir_length_s
    seed = settings.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, np.uint32)[0])
    early_cfg, late_cfg = settings.early, settings.late
    statistical = late_cfg.method == "statistical"
    if statistical and volume is None:
        raise ValueError("the statistical late method needs a closed model")

    # Early paths end `transition` after the direct sound. Without a set transition the
    # stochastic tracer hands over by order alone, and the statistical tail starts at
    # the mixing time.
    if settings.transition_ms is not None:
        transition = settings.transition_ms * 1e-3
    elif statistical:
        transition = mixing_time(volume)
    else:
        transition = math.inf

    listener_pos = np.array([lis.position for lis in chosen_listeners], dtype=float)
    responses: dict[tuple[str, str], Response] = {}
    details: dict[str, dict[str, float]] = {}
    for s_index, source in enumerate(chosen_sources):
        src = np.array(source.position, dtype=float)
        a = DIRECTIVITY_A[source.directivity]
        forward = np.array(source.orientation.forward())
        direct = np.linalg.norm(listener_pos - src, axis=1) / c
        cutoff = np.minimum(direct + transition, length)
        info: dict[str, float] = {"seed": _seed(seed, s_index)}

        started = time.perf_counter()
        say(f"{source.name}: early reflections ({early_cfg.method}, order {early_cfg.max_order})")
        candidates = _candidates(
            model, src, early_cfg.method, early_cfg.max_order, early_cfg.rays, threads
        )
        paths = [
            early_stage.find_paths(
                model,
                src,
                listener_pos[k],
                candidates,
                speed_of_sound=c,
                air=air,
                max_delay_s=float(cutoff[k]),
                directivity_a=a,
                forward=forward,
            )
            for k in range(len(chosen_listeners))
        ]
        info["candidates"] = float(sum(len(x) for x in candidates))
        info["early_s"] = time.perf_counter() - started

        started = time.perf_counter()
        if statistical:
            say(f"{source.name}: statistical late reverberation")
            t60 = room.reverberation_time("eyring", air_absorption=True)
            late = late_stage.statistical_energy(
                t60,
                volume,
                direct + transition,
                max_time=length,
                time_step=TIME_STEP,
                speed_of_sound=c,
            )
        else:
            say(f"{source.name}: stochastic ray tracing ({late_cfg.rays} rays)")
            radius = late_cfg.receiver_radius
            if radius is None:
                clearance = distance_to_surface(room.mesh, listener_pos)
                radius = late_stage.receiver_radius(
                    volume, late_cfg.rays, length, c, clearance=clearance
                )
            late = late_stage.stochastic_energy(
                model,
                src,
                listener_pos,
                rays=late_cfg.rays,
                max_time=length,
                time_step=TIME_STEP,
                speed_of_sound=c,
                air=air,
                reception=late_cfg.reception,
                covered_order=early_cfg.max_order,
                covered_time=cutoff,
                radius=radius,
                directions=fibonacci_sphere(late_cfg.directions),
                splitting=late_cfg.splitting,
                seed=info["seed"],
                directivity_a=a,
                forward=forward,
                threads=threads,
            )
            info["rays"] = float(late.rays)
            info["escaped_rays"] = float(late.escaped_rays)
            info["traced_rays"] = float(late.traced_rays)
            info["receiver_radius_m"] = float(np.min(radius))
        info["late_s"] = time.perf_counter() - started
        details[source.name] = info

        for k, listener in enumerate(chosen_listeners):
            responses[(source.name, listener.name)] = Response(
                source=source.name,
                listener=listener.name,
                early=paths[k],
                late=late.energy[k],
                time_step=late.time_step,
                length_s=length,
                direct_delay_s=float(direct[k]),
                transition_s=float(cutoff[k]),
                volume_m3=volume,
                speed_of_sound=c,
                seed=_seed(info["seed"], k + 1),
                orientation=np.array(listener.orientation.rotation()),
                late_directions=late.directions,
                late_directional=late.directional[k]
                if late.directional.size
                else np.zeros((0, 0, 0)),
                direction_step=late.direction_step,
            )
    return SimulationResult(room=room, settings=settings, responses=responses, details=details)


__all__ = ["TIME_STEP", "SimulationResult", "early_paths", "mixing_time", "simulate"]
