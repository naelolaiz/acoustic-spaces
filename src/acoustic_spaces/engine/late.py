"""Late reverberation: energy per octave band and time slot at each listener.

- ``stochastic_energy``: stochastic ray tracing in the compiled core. Rays reflect as
  the scene's ``simulation.scattering`` says (vector-based scattering by default, with
  the scattering of each surface's size added); how their energy reaches the listeners
  is set by ``reception`` (see :data:`RECEPTION`). Specular paths the early
  stage computes exactly are left out, so the two parts add up without double counting
  or a crossfade. Outdoors in wind or a temperature gradient the rays bend
  (:func:`~acoustic_spaces.engine.refracted.bending`).
- ``stochastic_map_energy``: the same tracer counting energy over a coverage map
  instead of at listeners.
- ``statistical_energy``: the reflected energy of Barron's revised theory, an
  exponential decay from the direct sound on, less the energy of the early paths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine import ray_device
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.refracted import bending
from acoustic_spaces.engine.vegetation import plant_arguments

_DECAY = 6.0 * math.log(10.0)  # 60 dB in nepers of energy

#: Time resolution of the late energy's arrival directions (s).
DIRECTION_STEP = 0.01

#: Reception modes: (diffuse rain, specular rain) in the compiled tracer.
#:
#: - ``diffuse_rain`` (default): the scattered part of each reflection radiates to every
#:   listener it can see (Lambert); specular rays count when they cross a detector
#:   sphere. Exact on average; noisier late in rooms with smooth, reflective walls.
#: - ``secondary_sources``: all reflected energy radiates that way, as in ODEON's late
#:   method. Much smoother, but it spreads specular energy as if the field were
#:   diffuse: in a box with smooth walls EDT came out 20 % shorter than the other modes.
#: - ``detector``: classic ray tracing, every ray counted by the sphere.
RECEPTION = {
    "diffuse_rain": (True, False),
    "secondary_sources": (True, True),
    "detector": (False, False),
}


@dataclass
class LateEnergy:
    """Energy per listener, band and time slot (time-integrated intensity)."""

    energy: np.ndarray  # (listeners, bands, bins)
    time_step: float  # s
    escaped_rays: int = 0
    rays: int = 0
    #: Rays traced, counting those split off (stochastic ray tracing only).
    traced_rays: int = 0
    #: Arrival directions (room coordinates, towards where the energy comes from) and the
    #: same energy split between them (listeners, directions, bands, slots of
    #: ``direction_step``); empty without directions.
    directions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    directional: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 0)))
    direction_step: float = DIRECTION_STEP
    #: With face groups (``stochastic_energy(groups=...)``): per listener, the energy in
    #: slots of ``moment_step`` times 1, the reflections the rays had off each group n_i,
    #: and n_i n_j (i <= j) (listeners, terms, bands, slots), for re-weighting the
    #: response for new absorption (:mod:`acoustic_spaces.engine.reweight`); else empty.
    moments: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 0)))
    moment_step: float = 0.0
    #: Where the rays were traced (stochastic ray tracing only).
    plan: ray_device.Plan | None = None


#: Weight window of the stochastic tracer (see :func:`stochastic_energy`): resolution of
#: the expected ray energy (s), how far (as a factor) a ray may stray from it before it
#: is split or thinned out, the most copies per split and per original ray, and the
#: fewest rays of the first batch.
SPLIT_WINDOW_S = 0.05
SPLIT_RATIO = 4.0
MAX_SPLIT = 8
MAX_FAMILY = 256
MIN_FIRST_RAYS = 1000
#: Most groups of faces whose reflections the tracer counts, and the time resolution of
#: the moments of those counts (s).
MAX_GROUPS = 16
MOMENT_STEP = 0.01

#: Smallest and largest automatic detector radius (m). Below about 0.5 m the specular
#: part of the tail in rooms with reflective, smooth walls gets noisy at usual ray counts.
RADIUS_RANGE = (0.5, 1.0)


def receiver_radius(
    volume_m3: float | np.ndarray | None,
    rays: int,
    max_time: float,
    c: float,
    clearance: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Detector sphere radius: Lehnert's log10(c * t_max) * sqrt(V / (pi * N)), kept
    within ``RADIUS_RANGE`` and inside the room.

    ``volume_m3`` may hold one volume per listener: that of the room it is in, where
    partitions or openings divide the model. ``clearance`` is each listener's distance
    to the nearest surface; a sphere reaching through a wall would read too low, so the
    radius stays below 90 % of it.
    """
    if volume_m3 is None:
        r = np.asarray(RADIUS_RANGE[0])
    else:
        volume = np.asarray(volume_m3, dtype=float)
        r = math.log10(max(c * max_time, 10.0)) * np.sqrt(volume / (math.pi * rays))
        r = np.clip(r, *RADIUS_RANGE)
    if clearance is None:
        return float(r) if r.ndim == 0 else r
    return np.maximum(np.minimum(r, 0.9 * np.asarray(clearance, dtype=float)), 1e-3)


def _exchange(model: TraceModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each room's surface area (rooms,), absorption area (bands, rooms) and the area
    that passes sound from one room to another (bands, rooms, rooms), what passes
    counted as absorbed on its side. A face counts for the room on each of its sides,
    and one standing in a room twice."""
    rooms = len(model.room_volumes)
    bands = model.absorption.shape[1]
    area = model.room.mesh.face_areas()
    face_room = model.face_room
    surface = np.zeros(rooms)
    absorbed = np.zeros((bands, rooms))
    for side in (0, 1):
        ok = face_room[:, side] >= 0
        np.add.at(surface, face_room[ok, side], area[ok])
        np.add.at(absorbed.T, face_room[ok, side], area[ok, None] * model.absorption[ok])
    passing = np.zeros((bands, rooms, rooms))
    if model.transmission is not None:
        between = (face_room >= 0).all(axis=1)
        share = (area[between, None] * model.transmission[between]).T  # (bands, faces)
        one, two = face_room[between, 0], face_room[between, 1]
        for b in range(bands):
            np.add.at(passing[b], (one, two), share[b])
            np.add.at(passing[b], (two, one), share[b])
    return surface, absorbed, passing


def room_levels(model: TraceModel, source_room: int, air: np.ndarray) -> np.ndarray:
    """The weight window's level of each room relative to the source's room (see
    :func:`stochastic_energy`): the energy each would hold in the steady state of a
    diffuse field, the rooms exchanging energy through the faces between them in
    proportion to their area and transmission. Empty with fewer than two rooms or no
    transmission."""
    rooms = len(model.room_volumes)
    if model.transmission is None or rooms < 2 or source_room < 0:
        return np.zeros(0)
    area = model.room.mesh.face_areas()
    volume = np.asarray(model.room_volumes, dtype=float)
    _, absorbed, passing = _exchange(model)
    energy = np.zeros((model.absorption.shape[1], rooms))
    for b in range(len(energy)):
        # Power balance per room: (c / 4) [A_r w_r - sum over faces of S tau w_q] = P.
        balance = np.diag(4.0 * np.asarray(air, float)[b] * volume + 1e-12 * area.sum())
        balance += np.diag(absorbed[b]) - passing[b]
        power = np.zeros(rooms)
        power[source_room] = 1.0
        energy[b] = np.maximum(np.linalg.solve(balance, power), 0.0) * volume
    total = energy.mean(axis=0)
    # In powers of the window's ratio: rooms that an opening joins hold much the same,
    # and rays that cross between them should not be split and thinned at every pass.
    with np.errstate(divide="ignore"):
        steps = np.round(np.log(total / total[source_room]) / math.log(SPLIT_RATIO))
    return np.where(total > 0.0, SPLIT_RATIO**steps, 0.0)


def diffuse_decay(
    model: TraceModel,
    source: np.ndarray,
    points: np.ndarray,
    air: np.ndarray,
    speed_of_sound: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """How the diffuse field decays at ``points`` where the model holds several rooms,
    for :func:`statistical_energy`: amplitudes (points, bands, rooms) and rates (bands,
    rooms) such that the energy a unit source sends reaches a point at the sum of
    amplitude exp(-rate t) per second, t from emission. None with one room.

    Coupled-room theory: the energy density of each room i follows
    V_i dw_i/dt = -(c / 4) (A_i + 4 m V_i) w_i + (c / 4) sum_j S_ij tau_ij w_j, with A_i
    the room's absorption area after Eyring, -S_i ln(1 - a_i) with what passes counted
    as absorbed, m the air's attenuation and S_ij tau_ij the area that passes sound from
    room j. The source's sound starts in its own room. With one room it is Barron's
    (c / V) exp(-13.8 t / T) with Eyring's T."""
    volume = np.asarray(model.room_volumes, dtype=float)
    if len(volume) < 2:
        return None
    source_room = int(model.room.room_of(np.asarray(source, float)[None])[0])
    where = model.room.room_of(np.atleast_2d(np.asarray(points, float)))
    surface, absorbed, passing = _exchange(model)
    mean = np.minimum(absorbed / np.maximum(surface, 1e-300), 1.0 - 1e-9)
    eyring = -surface * np.log1p(-mean) + 4.0 * np.asarray(air, float)[:, None] * volume
    bands, rooms = eyring.shape
    amplitude = np.zeros((len(where), bands, rooms))
    rate = np.zeros((bands, rooms))
    scale = 1.0 / np.sqrt(volume)
    inside = where >= 0
    for b in range(bands):
        # V^-1 (S tau - A) is similar to a symmetric matrix, so its modes are real.
        coupling = (speed_of_sound / 4.0) * scale[:, None] * (passing[b] - np.diag(eyring[b]))
        values, modes = np.linalg.eigh(coupling * scale[None, :])
        rate[b] = np.maximum(-values, 1e-9)
        if source_room < 0:
            continue
        amplitude[inside, b] = (
            speed_of_sound
            * modes[where[inside]]
            * modes[source_room]
            * (scale[where[inside], None] * scale[source_room])
        )
    return amplitude, rate


def _split_target(ray_energy: np.ndarray, reflections: np.ndarray) -> np.ndarray:
    """The expected energy of a ray per window, from a first batch of rays.

    Never rising, and rounded to quarter octaves: splitting only needs its order of
    magnitude, and rounding keeps results independent of how threads sum the batch.
    """
    mean = np.divide(ray_energy, reflections, out=np.zeros_like(ray_energy), where=reflections > 0)
    target = np.zeros_like(mean)
    level = math.inf
    for j, value in enumerate(mean):
        if value > 0:
            level = min(level, value)
        target[j] = level if math.isfinite(level) else 0.0
    with np.errstate(divide="ignore"):
        quarter_octaves = np.round(4.0 * np.log2(target))
    return np.where(target > 0, np.exp2(quarter_octaves / 4.0), 0.0)


def _in_batches(trace, rays: int, splitting: bool, own: np.ndarray | None = None):
    """Runs ``trace(count, first, target)``, which returns (arrays, escaped rays,
    ray_energy, reflections, traced rays), for all ``rays``: at once, or with
    ``splitting`` in two batches where the second uses the weight window measured by
    the first. Returns (arrays, escaped rays, traced rays), each array the mean of the
    batches weighted by their share of rays.

    ``own`` says for each row of the arrays (listener or map point) whether it lies in
    the source's room. The first batch sends few rays to the other rooms (no copies
    pass into them), so rows elsewhere take the second batch alone."""
    first = max(min(rays // 2, MIN_FIRST_RAYS), rays // 10, 1) if splitting else rays
    arrays, escaped, ray_energy, reflections, traced = trace(first, 0, np.zeros(0))
    if first < rays:
        # Each batch carries the whole source energy; weight them by their share of rays.
        target = _split_target(ray_energy, reflections) * first / (rays - first)
        more, more_escaped, _, _, more_traced = trace(rays - first, first, target)
        share = np.full(len(arrays[0]), first / rays)
        if own is not None:
            share = np.where(own, share, 0.0)
        arrays = [
            share.reshape(-1, *[1] * (a.ndim - 1)) * a
            + (1.0 - share).reshape(-1, *[1] * (a.ndim - 1)) * b
            for a, b in zip(arrays, more, strict=True)
        ]
        escaped += more_escaped
        traced += more_traced
    return arrays, escaped, traced


def _rooms(model: TraceModel, source: np.ndarray, air: np.ndarray, points) -> tuple:
    """The tracer's transmission arguments for a source: face_room, source_room and
    room_level, and whether each of ``points`` (listeners or map points) lies in the
    source's room (None when there is nothing to tell apart)."""
    if model.transmission is None or len(model.room_volumes) == 0:
        return np.zeros((0, 2), dtype=np.int64), -1, np.zeros(0), None
    source_room = int(model.room.room_of(np.asarray(source, float)[None])[0])
    level = room_levels(model, source_room, air)
    own = None
    if len(level):
        own = model.room.room_of(points) == source_room
    return np.ascontiguousarray(model.face_room, dtype=np.int64), source_room, level, own


def band_groups(model: TraceModel) -> list[slice]:
    """The bands each ray carries. Vector-based scattering turns a ray by its surface's
    scattering coefficient, so where the bands' coefficients differ (the scattering of a
    surface's size always makes them differ), every band is traced with rays of its own:
    one direction for all would scatter the treble as much as the bass. With random
    scattering, which weights each band by its own coefficient, or the same coefficient
    in every band, one ray carries them all; so it does when the model asks for one
    ray for all bands (``by_band`` off). Either way each band has the rays asked for,
    so tracing band by band traces that many rays per band."""
    bands = model.absorption.shape[1]
    # Faces that reflect nothing (openings) scatter nothing either.
    scattering = np.asarray(model.scattering)[np.any(model.absorption < 1.0, axis=1)]
    uniform = bool(np.all(scattering == scattering[:, :1]))
    if model.surface_size is not None:
        uniform = uniform and bool(np.all(scattering == 1.0))  # fully diffuse at any size
    if not model.vector_scattering or not model.by_band or uniform:
        return [slice(0, bands)]
    return [slice(b, b + 1) for b in range(bands)]


def _by_band(groups: list[slice], rays: int, run) -> tuple[list[np.ndarray], int, int]:
    """Runs ``run(group, rays)`` (returning arrays with bands on their second last axis,
    escaped rays and traced rays) for each group of bands, so every band has ``rays``
    rays, and joins the arrays along the band axis."""
    parts = [run(group, rays) for group in groups]
    arrays = [np.concatenate(found, axis=-2) for found in zip(*(p[0] for p in parts), strict=True)]
    return arrays, sum(p[1] for p in parts), sum(p[2] for p in parts)


def _directivity_arrays(directivity: Directivity | None, orientation, bands: int):
    if directivity is None:
        directivity_a, table = np.zeros(0), np.zeros((0, 0, bands))
    else:
        directivity_a, table = directivity.native()
    orientation = np.eye(3) if orientation is None else np.asarray(orientation, float)
    return (
        np.ascontiguousarray(directivity_a, dtype=np.float64),
        np.ascontiguousarray(orientation, dtype=np.float64),
        np.ascontiguousarray(table, dtype=np.float64),
    )


#: The tracer's arguments that _band_arguments gives last, in order.
_SIZE_ARGUMENTS = (
    "surface_size",
    "band_hz",
    "vector_scattering",
    "size_from_order",
    "transmission",
)


def _band_arguments(model: TraceModel, air: np.ndarray, directivity: tuple, band: slice) -> tuple:
    """For the bands ``band``: the tracer's directivity_a and directivity_table (from
    ``directivity``, both arrays for all bands), absorption, scattering and air, and the
    arguments surface_size, band_hz, vector_scattering, size_from_order and
    transmission."""
    first_order, table = directivity
    if model.surface_size is None:
        size, hz = np.zeros((0, 2)), np.zeros(0)
    else:
        size = np.ascontiguousarray(model.surface_size, dtype=np.float64)
        hz = np.ascontiguousarray(np.asarray(model.band_hz, dtype=np.float64)[band])
    if model.transmission is None:
        passing = np.zeros((0, 0))
    else:
        passing = np.ascontiguousarray(model.transmission[:, band], dtype=np.float64)
    return (
        np.ascontiguousarray(first_order[band] if len(first_order) else first_order),
        np.ascontiguousarray(table[..., band]),
        np.ascontiguousarray(model.absorption[:, band], dtype=np.float64),
        np.ascontiguousarray(model.scattering[:, band], dtype=np.float64),
        np.ascontiguousarray(np.asarray(air, dtype=np.float64)[band]),
        (size, hz, bool(model.vector_scattering), int(model.size_from_order), passing),
    )


def stochastic_energy(
    model: TraceModel,
    source: np.ndarray,
    listeners: np.ndarray,
    *,
    rays: int,
    max_time: float,
    time_step: float,
    speed_of_sound: float,
    air: np.ndarray,
    reception: str = "diffuse_rain",
    covered_order: int = -1,
    covered_time: np.ndarray | None = None,
    radius: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    splitting: bool = True,
    seed: int = 0,
    directivity: Directivity | None = None,
    orientation: np.ndarray | None = None,
    threads: int = 0,
    groups: np.ndarray | None = None,
    moment_step: float = MOMENT_STEP,
    device: str | ray_device.Plan = "cpu",
) -> LateEnergy:
    """Energy per listener, band and time slot by stochastic ray tracing.

    Specular-only paths with up to ``covered_order`` reflections reaching listener k
    no later than ``covered_time[k]`` (s, from emission) are left out; ``None`` puts no
    time limit on that. With ``directions`` (unit vectors in room coordinates), each
    arrival is also counted at the nearest of them, every ``DIRECTION_STEP``.

    With ``splitting``, a first tenth of the rays measures how much energy a ray
    usually carries over time; the rest are split when they carry much more and thinned
    out by Russian roulette when they carry much less (a weight window). The result is
    unbiased either way, but late tails in rooms where a few rays escape the absorption
    for a long time (smooth walls, absorption on the ceiling) no longer break up when
    those rays run out.

    ``groups`` (one per face: a group index, or -1) has the tracer count the reflections
    off each group and sum the moments of those counts in the result's ``moments``, at
    most ``MAX_GROUPS`` groups.

    ``device`` says where the rays are traced: cpu, gpu, both or auto (see
    :mod:`acoustic_spaces.engine.ray_device`), or a ``ray_device.Plan`` to follow; the
    result's ``plan`` says where they went.
    """
    if reception not in RECEPTION:
        raise ValueError(f"unknown reception {reception!r}; use one of {', '.join(RECEPTION)}")
    if reception == "secondary_sources" and covered_order < 0:
        raise ValueError("secondary_sources reception needs the early stage (covered_order >= 0)")
    diffuse_rain, specular_rain = RECEPTION[reception]
    listeners = np.ascontiguousarray(np.atleast_2d(listeners), dtype=np.float64)
    covered = np.zeros(0) if covered_time is None else np.asarray(covered_time, float)
    if radius is None:
        radius = receiver_radius(model.room.volume_m3, rays, max_time, speed_of_sound)
    radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(listeners),))
    directions = np.zeros((0, 3)) if directions is None else np.asarray(directions, float)
    directivity_a, orientation, table = _directivity_arrays(
        directivity, orientation, model.absorption.shape[1]
    )
    face_room, source_room, level, own = _rooms(model, source, air, listeners)
    bent = bending(model)
    if groups is None:
        face_group, n_groups = np.zeros(0, dtype=np.int64), 0
    else:
        face_group = np.ascontiguousarray(groups, dtype=np.int64)
        n_groups = int(face_group.max(initial=-1)) + 1
        if n_groups > MAX_GROUPS:
            raise ValueError(f"at most {MAX_GROUPS} groups of faces, not {n_groups}")

    # Where the rays go, chosen at the first batch.
    plans: list[ray_device.Plan] = []

    def run(band: slice, count: int):
        first_order, band_table, absorption, scattering, band_air, sizes = _band_arguments(
            model, air, (directivity_a, table), band
        )

        def trace(count: int, first: int, target: np.ndarray):
            values = {
                "bvh": model.bvh,
                "source": np.asarray(source, dtype=np.float64),
                "directivity_a": first_order,
                "orientation": orientation,
                "directivity_table": band_table,
                "absorption": absorption,
                "scattering": scattering,
                "air": band_air,
                **dict(zip(_SIZE_ARGUMENTS, sizes, strict=True)),
                "face_room": face_room,
                "source_room": source_room,
                "receivers": listeners,
                "directions": np.ascontiguousarray(directions.reshape(-1, 3)),
                "rays": int(count),
                "max_time": float(max_time),
                "time_step": float(time_step),
                "dir_time_step": DIRECTION_STEP,
                "speed_of_sound": float(speed_of_sound),
                "receiver_radius": np.ascontiguousarray(radius),
                "diffuse_rain": bool(diffuse_rain),
                "specular_rain": bool(specular_rain),
                "covered_order": int(covered_order),
                "covered_time": np.ascontiguousarray(covered, dtype=np.float64),
                "seed": int(seed) & 0xFFFFFFFFFFFFFFFF,
                "threads": int(threads),
                "window_time_step": SPLIT_WINDOW_S,
                "first_ray": int(first),
                "split_target": np.ascontiguousarray(target, dtype=np.float64),
                "split_ratio": SPLIT_RATIO,
                "max_split": MAX_SPLIT,
                "max_family": MAX_FAMILY,
                "room_level": level,
                "face_group": face_group,
                "groups": n_groups,
                "moment_time_step": float(moment_step),
                **bent,
                **plant_arguments(model, band),
            }
            if not plans:
                plans.append(ray_device.choose(device, values))
            found, plans[0] = ray_device.trace(plans[0], values)
            energy, directional, moments, *rest = found
            return [energy, directional, moments], *rest

        return _in_batches(trace, count, splitting, own)

    (energy, directional, moments), escaped, traced = _by_band(band_groups(model), rays, run)
    return LateEnergy(
        energy=energy,
        time_step=time_step,
        escaped_rays=int(escaped),
        rays=rays,
        traced_rays=int(traced),
        directions=directions,
        directional=directional,
        moments=moments,
        moment_step=float(moment_step) if n_groups else 0.0,
        plan=plans[0] if plans else None,
    )


#: Time bins of coverage maps, counted from each point's direct sound: 1 ms for the
#: first 100 ms, then 10 ms.
MAP_FINE_STEP = 1e-3
MAP_FINE_BINS = 100
MAP_COARSE_STEP = 1e-2


@dataclass
class MapEnergy:
    """Energy per map point, band and time bin (time-integrated intensity), with bins
    counted from each point's ``onset``: ``fine_bins`` of ``fine_step``, then
    ``coarse_step``."""

    energy: np.ndarray  # (points, bands, bins)
    onset: np.ndarray  # (points,) s from emission
    fine_step: float = MAP_FINE_STEP
    fine_bins: int = MAP_FINE_BINS
    coarse_step: float = MAP_COARSE_STEP
    rays: int = 0
    escaped_rays: int = 0
    traced_rays: int = 0


def stochastic_map_energy(
    model: TraceModel,
    source: np.ndarray,
    grid,
    onset: np.ndarray,
    *,
    rays: int,
    max_time: float,
    speed_of_sound: float,
    air: np.ndarray,
    covered_order: int = -1,
    covered_time: np.ndarray | None = None,
    splitting: bool = True,
    seed: int = 0,
    directivity: Directivity | None = None,
    orientation: np.ndarray | None = None,
    threads: int = 0,
) -> MapEnergy:
    """Energy over a coverage map (a :class:`~acoustic_spaces.engine.coverage.MapGrid`)
    by stochastic ray tracing: every leg of every ray counts its track length through
    each point's slab of air. ``onset`` is the arrival of the direct sound at each point
    (s); ``covered_order``, ``covered_time`` (per point), ``splitting`` and the rest are
    as for :func:`stochastic_energy`."""
    from acoustic_spaces import _native

    n = len(grid.points)
    onset = np.ascontiguousarray(np.broadcast_to(np.asarray(onset, float), (n,)))
    if covered_time is None:
        covered = np.full(n, math.inf)
    else:
        covered = np.ascontiguousarray(np.broadcast_to(np.asarray(covered_time, float), (n,)))
    directivity_a, orientation, table = _directivity_arrays(
        directivity, orientation, model.absorption.shape[1]
    )
    face_room, source_room, level, own = _rooms(model, source, air, grid.points)
    bent = bending(model)

    def run(band: slice, count: int):
        first_order, band_table, absorption, scattering, band_air, sizes = _band_arguments(
            model, air, (directivity_a, table), band
        )

        def trace(count: int, first: int, target: np.ndarray):
            energy, *rest = _native.trace_map(
                model.bvh,
                np.asarray(source, dtype=np.float64),
                first_order,
                orientation,
                band_table,
                absorption,
                scattering,
                band_air,
                *sizes,
                face_room,
                source_room,
                np.asarray(grid.origin, dtype=np.float64),
                float(grid.cell),
                np.ascontiguousarray(grid.index, dtype=np.int64),
                np.ascontiguousarray(grid.low, dtype=np.float64),
                np.ascontiguousarray(grid.high, dtype=np.float64),
                np.ascontiguousarray(grid.volume, dtype=np.float64),
                onset,
                covered,
                MAP_FINE_STEP,
                MAP_FINE_BINS,
                MAP_COARSE_STEP,
                int(count),
                float(max_time),
                float(speed_of_sound),
                int(covered_order),
                int(seed) & 0xFFFFFFFFFFFFFFFF,
                int(threads),
                SPLIT_WINDOW_S,
                int(first),
                np.ascontiguousarray(target, dtype=np.float64),
                SPLIT_RATIO,
                MAX_SPLIT,
                MAX_FAMILY,
                level,
                **bent,
                **plant_arguments(model, band),
            )
            return [energy], *rest

        return _in_batches(trace, count, splitting, own)

    (energy,), escaped, traced = _by_band(band_groups(model), rays, run)
    return MapEnergy(
        energy=energy,
        onset=onset,
        rays=rays,
        escaped_rays=int(escaped),
        traced_rays=int(traced),
    )


def statistical_energy(
    reverberation_time: np.ndarray,
    volume_m3: float,
    direct_times: np.ndarray,
    early: Sequence[EarlyReflections] = (),
    *,
    max_time: float,
    time_step: float,
    speed_of_sound: float,
    decay: tuple[np.ndarray, np.ndarray] | None = None,
) -> LateEnergy:
    """Energy per listener, band and time slot of a diffuse field, less the early paths.

    In Barron's revised theory the reflected energy of a unit source arrives from the
    direct sound (``direct_times``, s from emission) on at (c / V) exp(-13.8 t / T)
    per second, t from emission and T the ``reverberation_time`` (``simulate`` passes
    Eyring's, with air absorption). Its total, c T / (13.8 V) exp(-13.8 t_d / T), is
    4 / A exp(-13.8 t_d / T) with A = -S ln(1 - a) the absorption area behind Eyring's
    T: the Sabine form, as Barron has it. It leaves out that the first reflection is
    already absorbed, as the classical 4 (1 - a) / (a S) does not, so it reads 0.2 dB
    higher at a mean absorption of 0.1 and 0.8 dB higher at 0.3.

    ``early`` holds each listener's early paths, part of that energy: the specular share
    of reflections up to the early stage's order, until its transition. The late part is
    everything else: the scattered share of each early reflection, the higher orders
    before the transition and the whole field after it. It is Barron's energy less that
    of the early paths (the direct sound aside), slot by slot. Where an early path brings
    more than Barron's decay has delivered by its arrival, the late energy before it
    makes up the difference: its running sum is the largest that never falls and never
    rises above Barron's less the early paths'. No slot goes negative and the total
    stays Barron's, unless the early paths alone carry more.

    Where partitions or openings divide the model into rooms, ``decay`` (from
    :func:`diffuse_decay`) gives each listener's field as coupled-room theory has it, in
    place of ``reverberation_time`` and ``volume_m3``: a room joined to a livelier one
    decays in two slopes, and one behind a partition holds less.
    """
    t60 = np.asarray(reverberation_time, dtype=float)
    starts = np.asarray(direct_times, dtype=float)
    if decay is None:
        rate = _DECAY / t60[:, None]  # (bands, 1)
        amplitude = np.broadcast_to(speed_of_sound / volume_m3, (len(starts), len(t60), 1))
    else:
        amplitude, rate = decay
    bins = int(math.ceil(max_time / time_step))
    edges = np.arange(bins + 1) * time_step
    out = np.zeros((len(starts), len(t60), bins))
    for i, start in enumerate(starts):
        # Integral of (c / V) exp(-rate t) (in each room's modes, with several) from the
        # direct sound to each slot edge, less the early reflections arrived by then.
        r = rate[:, :, None]
        arrived = (
            amplitude[i][:, :, None]
            / r
            * (np.exp(-r * start) - np.exp(-r * np.maximum(edges, start)))
        ).sum(axis=1)
        if i < len(early):
            paths = early[i]
            slot = np.maximum(paths.delay_s / time_step, 0.0).astype(np.int64)
            keep = (paths.order > 0) & (slot < bins)
            per_slot = np.zeros((bins, len(t60)))
            np.add.at(per_slot, slot[keep], paths.energy[keep])
            arrived[:, 1:] -= np.cumsum(per_slot, axis=0).T
        # The largest non-decreasing curve from zero that stays below it.
        late = np.maximum(np.minimum.accumulate(arrived[:, ::-1], axis=1)[:, ::-1], 0.0)
        out[i] = np.diff(late, axis=1)
    return LateEnergy(energy=out, time_step=time_step)
