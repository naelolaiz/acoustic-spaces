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
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.engine.model import TraceModel

_DECAY = 6.0 * math.log(10.0)  # 60 dB in nepers of energy

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
    seed: int = 0,
    directivity_a: float = 1.0,
    forward: np.ndarray | None = None,
    threads: int = 0,
) -> LateEnergy:
    """Energy per listener, band and time slot by stochastic ray tracing.

    Specular-only paths with up to ``covered_order`` reflections reaching listener k
    no later than ``covered_time[k]`` (s, from emission) are left out; ``None`` puts no
    time limit on that.
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
    forward = np.array([1.0, 0.0, 0.0]) if forward is None else np.asarray(forward, float)
    energy, _, escaped = _native.trace_stochastic(
        model.bvh,
        np.asarray(source, dtype=np.float64),
        float(directivity_a),
        forward,
        model.absorption,
        model.scattering,
        np.ascontiguousarray(air, dtype=np.float64),
        listeners,
        np.zeros((0, 3)),
        int(rays),
        float(max_time),
        float(time_step),
        float(max(time_step, 0.01)),
        float(speed_of_sound),
        np.ascontiguousarray(radius),
        bool(diffuse_rain),
        bool(specular_rain),
        int(covered_order),
        np.ascontiguousarray(covered, dtype=np.float64),
        int(seed) & 0xFFFFFFFFFFFFFFFF,
        int(threads),
    )
    return LateEnergy(energy=energy, time_step=time_step, escaped_rays=int(escaped), rays=rays)


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
