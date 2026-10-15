"""Run the early and late stages for every source and listener of a room."""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import late as late_stage
from acoustic_spaces.engine import wave as wave_stage
from acoustic_spaces.engine.diffraction import EdgeDiffraction, Edges, find_edges
from acoustic_spaces.engine.grazing import apply_grazing
from acoustic_spaces.engine.model import TraceModel
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


def _towards(source: np.ndarray, listener: np.ndarray) -> np.ndarray | None:
    """Unit vector from the listener towards the source, or None where they meet."""
    d = np.asarray(source, dtype=float) - np.asarray(listener, dtype=float)
    length = float(np.linalg.norm(d))
    return d / length if length > 0 else None


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


class _Edges:
    """The model's diffracting edges, found when first needed."""

    def __init__(self, model: TraceModel) -> None:
        self.model = model
        self._edges: Edges | None = None

    def diffraction(
        self, mode: str, source: np.ndarray, room: Room, name: str, orientation: np.ndarray
    ) -> EdgeDiffraction | None:
        """Diffraction of the sound of source ``name`` at ``source``; None when off."""
        if mode == "none":
            return None
        if self._edges is None:
            self._edges = find_edges(self.model)
        if len(self._edges) == 0:
            return None
        return EdgeDiffraction(
            self.model,
            source,
            speed_of_sound=room.speed_of_sound(),
            air=room.air_attenuation(self.model.bands),
            directivity=room.directivity(name, self.model.bands),
            orientation=orientation,
            mode=mode,
            edges=self._edges,
        )


def early_paths(
    room: Room | Scene | str | Path,
    source: str | None = None,
    listener: str | None = None,
    *,
    max_order: int | None = None,
    method: str | None = None,
    diffraction: str | None = None,
    threads: int = 0,
) -> early_stage.EarlyReflections:
    """Specular and diffracted paths from one source to one listener, without the late
    part.

    Names default to the first source and listener; ``max_order``, ``method`` and
    ``diffraction`` default to ``scene.simulation.early``. Useful for drawing paths.
    """
    room = _as_room(room)
    scene = room.scene
    cfg = scene.simulation.early
    src = _pick(scene.sources, None if source is None else [source], "source")[0]
    lis = _pick(scene.listeners, None if listener is None else [listener], "listener")[0]
    model = TraceModel.from_room(room)
    target = np.array(lis.position, dtype=float)
    order = cfg.max_order if max_order is None else max_order
    orientation = np.array(src.orientation.rotation())
    edges = _Edges(model)
    elements = src.elements()
    parts = []
    for position in elements:
        candidates = _candidates(model, position, method or cfg.method, order, cfg.rays, threads)
        paths = early_stage.find_paths(
            model,
            position,
            target,
            candidates,
            speed_of_sound=room.speed_of_sound(),
            air=room.air_attenuation(model.bands),
            max_delay_s=scene.simulation.ir_length_s,
            directivity=room.directivity(src.name, model.bands),
            orientation=orientation,
        )
        found = edges.diffraction(
            diffraction or cfg.diffraction, position, room, src.name, orientation
        )
        if found is not None:
            paths = found.apply(target, paths, scene.simulation.ir_length_s)
        parts.append(apply_grazing(model, target, paths))
    return _shared(parts, model.absorption.shape[1])


def _shared(parts: list[early_stage.EarlyReflections], bands: int) -> early_stage.EarlyReflections:
    """The paths of a line or surface source's points, each with its share of the power."""
    if len(parts) == 1:
        return parts[0]
    share = 1.0 / len(parts)
    return early_stage.EarlyReflections.concatenate(
        [dataclasses.replace(p, energy=share * p.energy) for p in parts], bands
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
    core. Raises ``ValueError`` if :meth:`Room.problems` reports anything or there is no
    listener to simulate.
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
    if not chosen_listeners:
        raise ValueError("no listener to simulate")

    model = TraceModel.from_room(room, settings)
    c = room.speed_of_sound()
    air = room.air_attenuation(model.bands)
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
    # stochastic tracer hands over by order alone, and the statistical method keeps
    # early paths until the mixing time.
    if settings.transition_ms is not None:
        transition = settings.transition_ms * 1e-3
    elif statistical:
        transition = mixing_time(volume)
    else:
        transition = math.inf

    listener_pos = np.array([lis.position for lis in chosen_listeners], dtype=float)
    edges = _Edges(model)
    responses: dict[tuple[str, str], Response] = {}
    details: dict[str, dict[str, float]] = {}
    for source in chosen_sources:
        if settings.wave.enabled and source.extended:
            raise ValueError(
                f"source {source.name!r} is a line or surface; the wave solver takes point "
                "sources only"
            )

    def add(info: dict[str, float], key: str, value: float) -> None:
        info[key] = info.get(key, 0.0) + value

    def trace_point(label, position, directivity, orientation, point_seed, rays, info):
        """Direct sound times, early paths and late energy at every listener of one point
        source, which radiates all the power; timings and counts go into ``info``."""
        direct = np.linalg.norm(listener_pos - position, axis=1) / c
        cutoff = np.minimum(direct + transition, length)
        started = time.perf_counter()
        say(f"{label}: early reflections ({early_cfg.method}, order {early_cfg.max_order})")
        candidates = _candidates(
            model, position, early_cfg.method, early_cfg.max_order, early_cfg.rays, threads
        )
        paths = [
            early_stage.find_paths(
                model,
                position,
                listener_pos[k],
                candidates,
                speed_of_sound=c,
                air=air,
                max_delay_s=float(cutoff[k]),
                directivity=directivity,
                orientation=orientation,
            )
            for k in range(len(chosen_listeners))
        ]
        add(info, "candidates", float(sum(len(x) for x in candidates)))
        diffraction = edges.diffraction(
            early_cfg.diffraction, position, room, source.name, orientation
        )
        if diffraction is not None:
            say(f"{label}: edge diffraction ({early_cfg.diffraction})")
            paths = [
                diffraction.apply(listener_pos[k], paths[k], float(cutoff[k]))
                for k in range(len(chosen_listeners))
            ]
            info["edges"] = float(len(diffraction.edges))
            add(info, "diffracted_paths", float(sum(int(p.diffracted.sum()) for p in paths)))
        # The diffuse field is reckoned less the paths as they would come without the
        # audience, so what the seats take from them is lost, not made up by the tail.
        plain = paths
        paths = [apply_grazing(model, listener_pos[k], p) for k, p in enumerate(paths)]
        add(info, "early_s", time.perf_counter() - started)

        started = time.perf_counter()
        if statistical:
            say(f"{label}: statistical late reverberation")
            t60 = room.reverberation_time("eyring", air_absorption=True, bands=model.bands)
            late = late_stage.statistical_energy(
                t60,
                volume,
                direct,
                plain,
                max_time=length,
                time_step=TIME_STEP,
                speed_of_sound=c,
                decay=late_stage.diffuse_decay(model, position, listener_pos, air, c),
            )
        else:
            say(f"{label}: stochastic ray tracing ({rays} rays per band)")
            radius = late_cfg.receiver_radius
            if radius is None:
                # As for a point source with all the rays: the parts' rays add up.
                clearance = distance_to_surface(room.mesh, listener_pos)
                radius = late_stage.receiver_radius(
                    volume, late_cfg.rays, length, c, clearance=clearance
                )
            late = late_stage.stochastic_energy(
                model,
                position,
                listener_pos,
                rays=rays,
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
                seed=point_seed,
                directivity=directivity,
                orientation=orientation,
                threads=threads,
            )
            add(info, "rays", float(late.rays))
            add(info, "escaped_rays", float(late.escaped_rays))
            add(info, "traced_rays", float(late.traced_rays))
            info["receiver_radius_m"] = float(np.min(radius))
        add(info, "late_s", time.perf_counter() - started)
        return direct, paths, late

    for s_index, source in enumerate(chosen_sources):
        src = np.array(source.position, dtype=float)
        elements = source.elements()
        directivity = room.directivity(source.name, model.bands)
        orientation = np.array(source.orientation.rotation())
        info: dict[str, float] = {"seed": _seed(seed, s_index)}
        if len(elements) > 1:
            info["elements"] = float(len(elements))
        traced = []
        for e_index, position in enumerate(elements):
            # One point source of a line or surface: its share of the rays, a seed of its own.
            single = len(elements) == 1
            traced.append(
                trace_point(
                    source.name if single else f"{source.name} ({e_index + 1})",
                    position,
                    directivity,
                    orientation,
                    info["seed"] if single else _seed(info["seed"], e_index + 1),
                    math.ceil(late_cfg.rays / len(elements)),
                    info,
                )
            )
        direct = np.min([t[0] for t in traced], axis=0)
        cutoff = np.minimum(direct + transition, length)
        if len(traced) == 1:
            _, paths, late = traced[0]
        else:
            share = 1.0 / len(traced)
            bands = model.absorption.shape[1]
            paths = [
                _shared([t[1][k] for t in traced], bands) for k in range(len(chosen_listeners))
            ]
            late = dataclasses.replace(
                traced[0][2],
                energy=share * np.sum([t[2].energy for t in traced], axis=0),
                directional=share * np.sum([t[2].directional for t in traced], axis=0),
            )

        waves: list = [None] * len(chosen_listeners)
        if settings.wave.enabled:
            started = time.perf_counter()
            waves, wave_info = wave_stage.wave_responses(
                model,
                src,
                listener_pos,
                [listener.name for listener in chosen_listeners],
                max_band_hz=settings.wave.max_band_hz,
                length_s=length,
                sample_rate=settings.sample_rate,
                speed_of_sound=c,
                points_per_wavelength=settings.wave.points_per_wavelength,
                threads=threads,
                say=say,
                label=source.name,
            )
            info.update(wave_info)
            info["wave_s"] = time.perf_counter() - started
        details[source.name] = info

        fed = source.gain_db != 0 or source.eq_db is not None or source.delay_ms > 0
        for k, listener in enumerate(chosen_listeners):
            response = Response(
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
                wave=waves[k],
                source_direction=_towards(src, listener_pos[k]),
            )
            if fed:
                response = response.fed(source.feed_db(model.bands), source.delay_ms * 1e-3)
            responses[(source.name, listener.name)] = response
    return SimulationResult(room=room, settings=settings, responses=responses, details=details)


__all__ = ["TIME_STEP", "SimulationResult", "early_paths", "mixing_time", "simulate"]
