"""Run the early and late stages for every source and listener of a room."""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import late as late_stage
from acoustic_spaces.engine import tuning
from acoustic_spaces.engine import wave as wave_stage
from acoustic_spaces.engine.coherent import band_count
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


def response_length(room: Room, length_s: float, starts, ends) -> float:
    """How long responses last (s): ``length_s`` (``simulation.ir_length_s``). Outdoors,
    where sources and listeners can lie hundreds of metres apart, ``length_s`` counts
    from the latest direct sound, from any of ``starts`` (n, 3) to any of ``ends``
    (m, 3), so no response is cut off before its sound arrives."""
    if room.outdoor is None:
        return length_s
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    ends = np.atleast_2d(np.asarray(ends, dtype=float))
    farthest = float(np.max(np.linalg.norm(starts[:, None] - ends[None], axis=2)))
    return farthest / room.speed_of_sound() + length_s


def check_late_method(room: Room, settings: SimulationSettings) -> None:
    """Raise ``ValueError`` if the statistical late method cannot run in ``room``: it
    needs a closed room, and outdoors there is no diffuse field to reckon it from."""
    if settings.late.method != "statistical":
        return
    if room.outdoor is not None:
        raise ValueError(
            "the statistical late method needs a closed room; outdoors, use the stochastic "
            "ray tracer (simulation.late.method: stochastic_ray_tracing)"
        )
    if room.volume_m3 is None:
        raise ValueError("the statistical late method needs a closed model")


def _transition(settings: SimulationSettings, volume_m3: float | None) -> float:
    """How long after the direct sound the early paths end (s). Without a set
    transition the stochastic tracer hands over by order alone, and the statistical
    method keeps early paths until the mixing time."""
    if settings.transition_ms is not None:
        return settings.transition_ms * 1e-3
    if settings.late.method == "statistical":
        return mixing_time(volume_m3)
    return math.inf


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
        self,
        mode: str,
        source: np.ndarray,
        room: Room,
        name: str,
        orientation: np.ndarray,
        coherent: int = 0,
    ) -> EdgeDiffraction | None:
        """Diffraction of the sound of source ``name`` at ``source``, coherent in the
        lowest ``coherent`` bands; None when off."""
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
            coherent=coherent,
        )


def early_paths(
    room: Room | Scene | str | Path,
    source: str | None = None,
    listener: str | None = None,
    *,
    max_order: int | None = None,
    method: str | None = None,
    diffraction: str | None = None,
    threads: int | None = None,
) -> early_stage.EarlyReflections:
    """Specular and diffracted paths from one source to one listener, without the late
    part: those a simulation keeps as early paths, up to ``scene.simulation``'s
    ``transition_ms`` after the direct sound (the mixing time with the statistical late
    method) and within ``ir_length_s``.

    Names default to the first source and listener; ``max_order``, ``method`` and
    ``diffraction`` default to ``scene.simulation.early``, ``threads`` to
    ``scene.simulation.performance.threads``. Useful for drawing paths.
    """
    room = _as_room(room)
    scene = room.scene
    cfg = scene.simulation.early
    if threads is None:
        threads = scene.simulation.performance.threads
    src = _pick(scene.sources, None if source is None else [source], "source")[0]
    lis = _pick(scene.listeners, None if listener is None else [listener], "listener")[0]
    model = TraceModel.from_room(room)
    target = np.array(lis.position, dtype=float)
    order = cfg.max_order if max_order is None else max_order
    orientation = np.array(src.orientation.rotation())
    edges = _Edges(model)
    elements = src.elements()
    # A line's or a surface's points are uncorrelated: their paths add in energy. Indoors
    # each point's own do too; outdoors they add in pressure among themselves.
    coherent = coherent_bands(scene.simulation, model.absorption.shape[1])
    coherent = coherent if len(elements) == 1 or room.outdoor is not None else 0
    transition = _transition(scene.simulation, room.volume_m3)
    length = response_length(room, scene.simulation.ir_length_s, elements, target)
    parts = []
    for position in elements:
        cutoff = min(
            float(np.linalg.norm(target - position)) / room.speed_of_sound() + transition,
            length,
        )
        candidates = _candidates(model, position, method or cfg.method, order, cfg.rays, threads)
        paths = early_stage.find_paths(
            model,
            position,
            target,
            candidates,
            speed_of_sound=room.speed_of_sound(),
            air=room.air_attenuation(model.bands),
            max_delay_s=cutoff,
            directivity=room.directivity(src.name, model.bands),
            orientation=orientation,
            coherent=coherent,
        )
        found = edges.diffraction(
            diffraction or cfg.diffraction, position, room, src.name, orientation, coherent
        )
        if found is not None:
            paths = found.apply(target, paths, cutoff)
        parts.append(apply_grazing(model, target, paths))
    return _shared(parts, model.absorption.shape[1])


def _mixing(
    late_from: Mapping[str, Mapping[str, float]], names: list[str], kind: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Which of ``names`` are traced for their late part (indices), and the weights of
    each name's late part on theirs (names, traced); None when all are traced."""
    if not any(name in late_from for name in names):
        return None
    index = {name: k for k, name in enumerate(names)}
    traced = [k for k, name in enumerate(names) if name not in late_from]
    column = {k: j for j, k in enumerate(traced)}
    weights = np.zeros((len(names), len(traced)))
    for k, name in enumerate(names):
        if name not in late_from:
            weights[k, column[k]] = 1.0
            continue
        if not late_from[name]:
            raise ValueError(f"the late part of {name!r} comes from no {kind}")
        for other, weight in late_from[name].items():
            if weight <= 0:
                raise ValueError(f"the late part of {name!r} takes positive weights")
            if other not in index or other in late_from:
                raise ValueError(
                    f"the late part of {name!r} comes from {other!r}, which is no {kind} "
                    "traced for its own"
                )
            weights[k, column[index[other]]] += weight
    return np.array(traced, dtype=np.int64), weights


def _mix_listeners(late: late_stage.LateEnergy, weights: np.ndarray) -> late_stage.LateEnergy:
    """``late``, traced at some listeners, for every listener: a row of ``weights`` each."""

    def mix(a: np.ndarray) -> np.ndarray:
        return np.einsum("lt,t...->l...", weights, a) if a.size else a

    return dataclasses.replace(
        late, energy=mix(late.energy), directional=mix(late.directional), moments=mix(late.moments)
    )


def _mix_sources(
    lates: Sequence[late_stage.LateEnergy], weights: Sequence[float]
) -> late_stage.LateEnergy:
    """The late energy of several sources, summed with ``weights``."""

    def mix(name: str) -> np.ndarray:
        parts = [getattr(late, name) for late in lates]
        if not parts[0].size:
            return parts[0]
        return np.einsum("t,t...->...", np.asarray(weights, dtype=float), np.array(parts))

    return dataclasses.replace(
        lates[0],
        energy=mix("energy"),
        directional=mix("directional"),
        moments=mix("moments"),
        rays=0,
        escaped_rays=0,
        traced_rays=0,
    )


def coherent_bands(settings, bands: int) -> int:
    """How many of the lowest ``bands`` the early paths add in pressure in, with the
    simulation ``settings`` (``scene.simulation``)."""
    wave = settings.wave.max_band_hz if settings.wave.enabled else None
    return band_count(settings.early.coherent_max_band_hz, bands, wave)


def _shared(parts: list[early_stage.EarlyReflections], bands: int) -> early_stage.EarlyReflections:
    """The paths of a line or surface source's points, each with its share of the power
    and the point it comes from."""
    if len(parts) == 1:
        return parts[0]
    share = 1.0 / len(parts)
    return early_stage.EarlyReflections.concatenate(
        [
            dataclasses.replace(p, energy=share * p.energy, point=np.full(len(p), k))
            for k, p in enumerate(parts)
        ],
        bands,
    )


def _where(plan) -> str:
    """Where a late stage's plan traced the rays, and why, for progress messages."""
    where = {"cpu": "the CPU", "gpu": "the GPU", "both": "the CPU and the GPU"}[plan.device]
    return f"{where} ({plan.reason})" if plan.reason else where


def simulate(
    room: Room | Scene | str | Path,
    *,
    settings: SimulationSettings | None = None,
    sources: Sequence[str] | None = None,
    listeners: Sequence[str] | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
    groups: np.ndarray | None = None,
    late_from: Mapping[str, Mapping[str, float]] | None = None,
) -> SimulationResult:
    """Simulate the room's impulse responses as set in ``scene.simulation``.

    ``room`` may be a :class:`Room`, a :class:`Scene` or a scene file. ``sources`` and
    ``listeners`` pick names from the scene (all by default). ``threads`` overrides
    ``simulation.performance.threads`` (0 uses every core). Raises ``ValueError`` if
    :meth:`Room.problems` reports anything or there is no listener to simulate.

    ``groups`` (the group of each face, or -1;
    :func:`~acoustic_spaces.engine.reweight.face_groups`) keeps in each response the
    moments of the late part's reflections off each group, so that
    :meth:`Response.reweighted` gives it for other absorption (stochastic ray tracing
    only).

    ``late_from`` gives listeners or sources the late part of others, mixed:
    ``{name: {other: weight}}``, the others being listeners (for a listener) or sources
    (for a source) that keep their own. Only those are traced by the stochastic ray
    tracer, by far the costliest stage; the statistical method gives everyone their own.
    Walks use it (:func:`acoustic_spaces.walkthrough.simulate_walk`): the late field
    changes slowly from place to place, the early sound does not.
    """
    room = _as_room(room)
    problems = room.problems()
    if problems:
        raise ValueError("; ".join(problems))
    scene = room.scene
    settings = settings or scene.simulation
    if threads is None:
        threads = settings.performance.threads
    say = progress or (lambda _: None)

    chosen_sources = _pick(scene.sources, sources, "source")
    chosen_listeners = _pick(scene.listeners, listeners, "listener")
    if not chosen_listeners:
        raise ValueError("no listener to simulate")
    # Seeds follow the scene's order, so a run of some sources or listeners gives them
    # what a run of all does, as coverage maps do.
    source_index = {source.name: k for k, source in enumerate(scene.sources)}
    listener_index = {listener.name: k for k, listener in enumerate(scene.listeners)}

    model = TraceModel.from_room(room, settings)
    c = room.speed_of_sound()
    air = room.air_attenuation(model.bands)
    volume = room.volume_m3
    listener_pos = np.array([lis.position for lis in chosen_listeners], dtype=float)
    length = response_length(
        room,
        settings.ir_length_s,
        [point for source in chosen_sources for point in source.elements()],
        listener_pos,
    )
    seed = settings.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, np.uint32)[0])
    early_cfg, late_cfg = settings.early, settings.late
    # Not set in the scene: this computer's default.
    late_device = tuning.resolve_device(late_cfg.device)
    statistical = late_cfg.method == "statistical"
    check_late_method(room, settings)
    if groups is not None and statistical:
        raise ValueError("face groups need the stochastic ray tracer, not the statistical method")

    # Early paths end `transition` after the direct sound.
    transition = _transition(settings, volume)

    late_from = {} if late_from is None or statistical else late_from
    unknown = sorted(set(late_from) - {item.name for item in [*chosen_listeners, *chosen_sources]})
    if unknown:
        raise ValueError(f"late_from names {unknown[0]!r}, which is no listener or source here")
    listener_mix = _mixing(late_from, [lis.name for lis in chosen_listeners], "listener")
    source_mix = _mixing(late_from, [source.name for source in chosen_sources], "source")
    # The listeners the late part is traced at.
    traced_at = slice(None) if listener_mix is None else listener_mix[0]
    edges = _Edges(model)
    low_bands = coherent_bands(settings, model.absorption.shape[1])
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

    def trace_point(
        label, position, directivity, orientation, point_seed, rays, info, single, own_late
    ):
        """Direct sound times, early paths and late energy at every listener of one point
        source, which radiates all the power; timings and counts go into ``info``.
        Unless ``single``, one of the uncorrelated points of a line or surface, whose
        paths add in energy (indoors; outdoors in pressure among themselves). Unless
        ``own_late``, no late energy (None): it comes from other sources."""
        coherent = low_bands if single or room.outdoor is not None else 0
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
                coherent=coherent,
            )
            for k in range(len(chosen_listeners))
        ]
        add(info, "candidates", float(sum(len(x) for x in candidates)))
        diffraction = edges.diffraction(
            early_cfg.diffraction, position, room, source.name, orientation, coherent
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
        elif not own_late:
            late = None
        else:
            say(f"{label}: stochastic ray tracing ({rays} rays per band)")
            radius = late_cfg.receiver_radius
            if radius is None:
                # As for a point source with all the rays: the parts' rays add up.
                clearance = distance_to_surface(room.mesh, listener_pos[traced_at])
                radius = late_stage.receiver_radius(
                    volume, late_cfg.rays, length, c, clearance=clearance
                )
            late = late_stage.stochastic_energy(
                model,
                position,
                listener_pos[traced_at],
                rays=rays,
                max_time=length,
                time_step=TIME_STEP,
                speed_of_sound=c,
                air=air,
                reception=late_cfg.reception,
                covered_order=early_cfg.max_order,
                covered_time=cutoff[traced_at],
                radius=radius,
                directions=fibonacci_sphere(late_cfg.directions),
                splitting=late_cfg.splitting,
                seed=point_seed,
                directivity=directivity,
                orientation=orientation,
                threads=threads,
                groups=groups,
                device=late_device,
            )
            if late.plan is not None and late_device != "cpu":
                say(f"{label}: late rays traced on {_where(late.plan)}")
            add(info, "rays", float(late.rays))
            add(info, "escaped_rays", float(late.escaped_rays))
            add(info, "traced_rays", float(late.traced_rays))
            info["receiver_radius_m"] = float(np.min(radius))
            if listener_mix is not None:
                late = _mix_listeners(late, listener_mix[1])
        add(info, "late_s", time.perf_counter() - started)
        return direct, paths, late

    done = {}
    for source in chosen_sources:
        src = np.array(source.position, dtype=float)
        elements = source.elements()
        directivity = room.directivity(source.name, model.bands)
        orientation = np.array(source.orientation.rotation())
        info: dict[str, float] = {"seed": _seed(seed, source_index[source.name])}
        if len(elements) > 1:
            info["elements"] = float(len(elements))
        if length != settings.ir_length_s:
            info["ir_length_s"] = length
        own_late = source.name not in late_from
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
                    single,
                    own_late,
                )
            )
        direct = np.min([t[0] for t in traced], axis=0)
        cutoff = np.minimum(direct + transition, length)
        if len(traced) == 1:
            _, paths, late = traced[0]
        else:
            bands = model.absorption.shape[1]
            paths = [
                _shared([t[1][k] for t in traced], bands) for k in range(len(chosen_listeners))
            ]
            late = None
            if own_late:
                share = 1.0 / len(traced)
                late = dataclasses.replace(
                    traced[0][2],
                    energy=share * np.sum([t[2].energy for t in traced], axis=0),
                    directional=share * np.sum([t[2].directional for t in traced], axis=0),
                    moments=share * np.sum([t[2].moments for t in traced], axis=0),
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
                device=tuning.resolve_device(settings.wave.device),
                scheme=settings.wave.scheme,
                say=say,
                label=source.name,
            )
            info.update(wave_info)
            info["wave_s"] = time.perf_counter() - started
        details[source.name] = info
        done[source.name] = (src, direct, cutoff, paths, late, waves)

    for s_index, source in enumerate(chosen_sources):
        src, direct, cutoff, paths, late, waves = done[source.name]
        if late is None:
            row = source_mix[1][s_index]
            used = np.flatnonzero(row)
            late = _mix_sources(
                [done[chosen_sources[source_mix[0][j]].name][4] for j in used], row[used]
            )
        info = details[source.name]
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
                seed=_seed(info["seed"], listener_index[listener.name] + 1),
                orientation=np.array(listener.orientation.rotation()),
                late_directions=late.directions,
                late_directional=late.directional[k]
                if late.directional.size
                else np.zeros((0, 0, 0)),
                direction_step=late.direction_step,
                wave=waves[k],
                source_direction=_towards(src, listener_pos[k]),
                late_moments=late.moments[k] if late.moments.size else np.zeros((0, 0, 0)),
                moment_step=late.moment_step,
            )
            if fed:
                response = response.fed(source.feed_db(model.bands), source.delay_ms * 1e-3)
            responses[(source.name, listener.name)] = response
    return SimulationResult(room=room, settings=settings, responses=responses, details=details)


__all__ = [
    "TIME_STEP",
    "SimulationResult",
    "check_late_method",
    "early_paths",
    "mixing_time",
    "response_length",
    "simulate",
]
