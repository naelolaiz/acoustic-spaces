"""Late reverberation: energy per octave band and time slot at each listener.

- ``stochastic_energy``: stochastic ray tracing in the compiled core. Rays reflect as
  the scene's ``simulation.scattering`` says (vector-based scattering by default, with
  the scattering of each surface's size added); how their energy reaches the listeners
  is set by ``reception`` (see :data:`RECEPTION`). Specular paths the early
  stage computes exactly are left out, so the two parts add up without double counting
  or a crossfade.
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
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

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


#: Weight window of the stochastic tracer (see :func:`stochastic_energy`): resolution of
#: the expected ray energy (s), how far (as a factor) a ray may stray from it before it
#: is split or thinned out, the most copies per split and per original ray, and the
#: fewest rays of the first batch.
SPLIT_WINDOW_S = 0.05
SPLIT_RATIO = 4.0
MAX_SPLIT = 8
MAX_FAMILY = 256
MIN_FIRST_RAYS = 1000

#: Smallest and largest automatic detector radius (m). Below about 0.5 m the specular
#: part of the tail in rooms with reflective, smooth walls gets noisy at usual ray counts.
RADIUS_RANGE = (0.5, 1.0)


def receiver_radius(
    volume_m3: float | None,
    rays: int,
    max_time: float,
    c: float,
    clearance: np.ndarray | float | None = None,
) -> np.ndarray | float:
    """Detector sphere radius: Lehnert's log10(c * t_max) * sqrt(V / (pi * N)), kept
    within ``RADIUS_RANGE`` and inside the room.

    ``clearance`` is each listener's distance to the nearest surface; a sphere reaching
    through a wall would read too low, so the radius stays below 90 % of it.
    """
    if volume_m3 is None:
        r = RADIUS_RANGE[0]
    else:
        r = math.log10(max(c * max_time, 10.0)) * math.sqrt(volume_m3 / (math.pi * rays))
        r = min(max(r, RADIUS_RANGE[0]), RADIUS_RANGE[1])
    if clearance is None:
        return float(r)
    return np.maximum(np.minimum(r, 0.9 * np.asarray(clearance, dtype=float)), 1e-3)


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


def _in_batches(trace, rays: int, splitting: bool):
    """Runs ``trace(count, first, target)``, which returns (arrays, escaped rays,
    ray_energy, reflections, traced rays), for all ``rays``: at once, or with
    ``splitting`` in two batches where the second uses the weight window measured by
    the first. Returns (arrays, escaped rays, traced rays), each array the mean of the
    batches weighted by their share of rays."""
    first = max(min(rays // 2, MIN_FIRST_RAYS), rays // 10, 1) if splitting else rays
    arrays, escaped, ray_energy, reflections, traced = trace(first, 0, np.zeros(0))
    if first < rays:
        # Each batch carries the whole source energy; weight them by their share of rays.
        target = _split_target(ray_energy, reflections) * first / (rays - first)
        more, more_escaped, _, _, more_traced = trace(rays - first, first, target)
        arrays = [
            (first * a + (rays - first) * b) / rays for a, b in zip(arrays, more, strict=True)
        ]
        escaped += more_escaped
        traced += more_traced
    return arrays, escaped, traced


def band_groups(model: TraceModel) -> list[slice]:
    """The bands each ray carries. Vector-based scattering turns a ray by its surface's
    scattering coefficient, so where the bands' coefficients differ (the scattering of a
    surface's size always makes them differ), every band is traced with rays of its own:
    one direction for all would scatter the treble as much as the bass. With random
    scattering, which weights each band by its own coefficient, or the same coefficient
    in every band, one ray carries them all. Either way each band has the rays asked
    for, so tracing band by band traces that many rays per band."""
    bands = model.absorption.shape[1]
    scattering = np.asarray(model.scattering)
    uniform = bool(np.all(scattering == scattering[:, :1]))
    if model.surface_size is not None:
        uniform = uniform and bool(np.all(scattering == 1.0))  # fully diffuse at any size
    if not model.vector_scattering or uniform:
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


def _band_arguments(model: TraceModel, air: np.ndarray, directivity: tuple, band: slice) -> tuple:
    """For the bands ``band``: the tracer's directivity_a and directivity_table (from
    ``directivity``, both arrays for all bands), absorption, scattering and air, and the
    arguments surface_size, band_hz, vector_scattering and size_from_order."""
    first_order, table = directivity
    if model.surface_size is None:
        size, hz = np.zeros((0, 2)), np.zeros(0)
    else:
        size = np.ascontiguousarray(model.surface_size, dtype=np.float64)
        hz = np.ascontiguousarray(np.asarray(OCTAVE_BANDS_HZ, dtype=np.float64)[band])
    return (
        np.ascontiguousarray(first_order[band] if len(first_order) else first_order),
        np.ascontiguousarray(table[..., band]),
        np.ascontiguousarray(model.absorption[:, band], dtype=np.float64),
        np.ascontiguousarray(model.scattering[:, band], dtype=np.float64),
        np.ascontiguousarray(np.asarray(air, dtype=np.float64)[band]),
        (size, hz, bool(model.vector_scattering), int(model.size_from_order)),
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
    """
    from acoustic_spaces import _native

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

    def run(band: slice, count: int):
        first_order, band_table, absorption, scattering, band_air, sizes = _band_arguments(
            model, air, (directivity_a, table), band
        )

        def trace(count: int, first: int, target: np.ndarray):
            energy, directional, *rest = _native.trace_stochastic(
                model.bvh,
                np.asarray(source, dtype=np.float64),
                first_order,
                orientation,
                band_table,
                absorption,
                scattering,
                band_air,
                *sizes,
                listeners,
                np.ascontiguousarray(directions.reshape(-1, 3)),
                int(count),
                float(max_time),
                float(time_step),
                DIRECTION_STEP,
                float(speed_of_sound),
                np.ascontiguousarray(radius),
                bool(diffuse_rain),
                bool(specular_rain),
                int(covered_order),
                np.ascontiguousarray(covered, dtype=np.float64),
                int(seed) & 0xFFFFFFFFFFFFFFFF,
                int(threads),
                SPLIT_WINDOW_S,
                int(first),
                np.ascontiguousarray(target, dtype=np.float64),
                SPLIT_RATIO,
                MAX_SPLIT,
                MAX_FAMILY,
            )
            return [energy, directional], *rest

        return _in_batches(trace, count, splitting)

    (energy, directional), escaped, traced = _by_band(band_groups(model), rays, run)
    return LateEnergy(
        energy=energy,
        time_step=time_step,
        escaped_rays=int(escaped),
        rays=rays,
        traced_rays=int(traced),
        directions=directions,
        directional=directional,
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
            )
            return [energy], *rest

        return _in_batches(trace, count, splitting)

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
    """
    t60 = np.asarray(reverberation_time, dtype=float)
    rate = _DECAY / t60[:, None]  # (bands, 1)
    bins = int(math.ceil(max_time / time_step))
    edges = np.arange(bins + 1) * time_step
    out = np.zeros((len(direct_times), len(t60), bins))
    for i, start in enumerate(np.asarray(direct_times, dtype=float)):
        # Integral of (c / V) exp(-rate t) from the direct sound to each slot edge, less
        # the early reflections arrived by then.
        arrived = (
            speed_of_sound
            / volume_m3
            / rate
            * (np.exp(-rate * start) - np.exp(-rate * np.maximum(edges, start)))
        )
        if i < len(early):
            paths = early[i]
            slot = (paths.delay_s / time_step).astype(np.int64)
            keep = (paths.order > 0) & (slot < bins)
            per_slot = np.zeros((bins, len(t60)))
            np.add.at(per_slot, slot[keep], paths.energy[keep])
            arrived[:, 1:] -= np.cumsum(per_slot, axis=0).T
        # The largest non-decreasing curve from zero that stays below it.
        late = np.maximum(np.minimum.accumulate(arrived[:, ::-1], axis=1)[:, ::-1], 0.0)
        out[i] = np.diff(late, axis=1)
    return LateEnergy(energy=out, time_step=time_step)
