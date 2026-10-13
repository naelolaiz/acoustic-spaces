"""Late reverberation: energy per octave band and time slot at each listener.

- ``stochastic_energy``: stochastic ray tracing in the compiled core. Rays reflect
  specularly or diffusely by the scattering coefficient; how their energy reaches the
  listeners is set by ``reception`` (see :data:`RECEPTION`). Specular paths the early
  stage computes exactly are left out, so the two parts add up without double counting
  or a crossfade.
- ``statistical_energy``: an exponential decay from the Eyring reverberation time,
  scaled by Barron's revised theory (reflected energy density c / V for a unit source),
  starting at the transition time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.model import TraceModel

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
    if directivity is None:
        directivity_a, table = np.zeros(0), np.zeros((0, 0, model.absorption.shape[1]))
    else:
        directivity_a, table = directivity.native()
    orientation = np.eye(3) if orientation is None else np.asarray(orientation, float)

    def trace(count: int, first: int, target: np.ndarray):
        return _native.trace_stochastic(
            model.bvh,
            np.asarray(source, dtype=np.float64),
            np.ascontiguousarray(directivity_a, dtype=np.float64),
            np.ascontiguousarray(orientation, dtype=np.float64),
            np.ascontiguousarray(table, dtype=np.float64),
            model.absorption,
            model.scattering,
            np.ascontiguousarray(air, dtype=np.float64),
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

    first = max(min(rays // 2, MIN_FIRST_RAYS), rays // 10) if splitting else rays
    energy, directional, escaped, ray_energy, reflections, traced = trace(first, 0, np.zeros(0))
    if first < rays:
        # Each batch carries the whole source energy; weight them by their share of rays.
        target = _split_target(ray_energy, reflections) * first / (rays - first)
        more, more_directional, more_escaped, _, _, more_traced = trace(rays - first, first, target)
        energy = (first * energy + (rays - first) * more) / rays
        directional = (first * directional + (rays - first) * more_directional) / rays
        escaped += more_escaped
        traced += more_traced
    return LateEnergy(
        energy=energy,
        time_step=time_step,
        escaped_rays=int(escaped),
        rays=rays,
        traced_rays=int(traced),
        directions=directions,
        directional=directional,
    )


def statistical_energy(
    reverberation_time: np.ndarray,
    volume_m3: float,
    start_times: np.ndarray,
    *,
    max_time: float,
    time_step: float,
    speed_of_sound: float,
) -> LateEnergy:
    """Exponential decay per band from each listener's start time onwards."""
    t60 = np.asarray(reverberation_time, dtype=float)
    bins = int(math.ceil(max_time / time_step))
    edges = np.arange(bins + 1) * time_step
    out = np.zeros((len(start_times), len(t60), bins))
    for i, start in enumerate(np.asarray(start_times, dtype=float)):
        lo = np.maximum(edges[:-1], start)
        hi = np.maximum(edges[1:], start)
        rate = _DECAY / t60[:, None]  # (bands, 1)
        # Integral of (c / V) exp(-rate t) over each slot.
        out[i] = speed_of_sound / volume_m3 / rate * (np.exp(-rate * lo) - np.exp(-rate * hi))
    return LateEnergy(energy=out, time_step=time_step)
