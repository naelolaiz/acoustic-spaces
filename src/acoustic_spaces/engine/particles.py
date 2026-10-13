"""Sound particles: where the energy leaving a source is at each moment, for animations.

Particles leave the source evenly in all directions, weighted by its directivity, and
travel at the speed of sound. At each wall they lose the absorbed share of their
energy in the chosen octave band (and some to the air on the way) and reflect
specularly or, with the wall's scattering coefficient as probability, in a Lambert
direction, as in the stochastic tracer. Particles that find a gap in an open model fly
on out of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_spaces.engine.model import DIRECTIVITY_A, TraceModel, directivity_weight
from acoustic_spaces.room import Room
from acoustic_spaces.spatial.directions import fibonacci_sphere

#: Particles stop after this many reflections even if time remains.
MAX_REFLECTIONS = 200


@dataclass
class Particles:
    """Positions of sound particles at a series of times."""

    times: np.ndarray  # (frames,) s after emission
    positions: np.ndarray  # (frames, particles, 3) m
    #: Energy relative to a particle's share at emission (frames, particles).
    energy: np.ndarray
    #: Reflections so far (frames, particles).
    order: np.ndarray
    band_hz: int
    #: Name of the source the particles left.
    source: str = ""


def _lambert(normals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cosine-weighted directions around each unit normal."""
    u, v = rng.random(len(normals)), rng.random(len(normals))
    r, phi = np.sqrt(u), 2.0 * np.pi * v
    local = np.column_stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1.0 - u)])
    helper = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t1 = np.cross(normals, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    return local[:, :1] * t1 + local[:, 1:2] * t2 + local[:, 2:] * normals


def trace_particles(
    room: Room,
    source: str | None = None,
    *,
    count: int = 4000,
    duration: float = 0.1,
    frames: int = 100,
    band: int = 4,
    seed: int = 0,
) -> Particles:
    """Trace ``count`` particles from ``source`` (default: the first) for ``duration``
    seconds and sample them at ``frames`` evenly spaced times from 0 to ``duration``.

    ``band`` picks the octave band whose absorption and scattering apply (4 = 1 kHz).
    """
    from acoustic_spaces.materials import OCTAVE_BANDS_HZ

    scene = room.scene
    sources = {s.name: s for s in scene.sources}
    if source is None:
        source = scene.sources[0].name
    if source not in sources:
        raise ValueError(f"no source {source!r}; the scene has {', '.join(sources)}")
    if count < 1 or frames < 1 or duration <= 0:
        raise ValueError("count and frames must be positive, and so must duration")
    if not 0 <= band < len(OCTAVE_BANDS_HZ):
        raise ValueError(f"band must be 0 to {len(OCTAVE_BANDS_HZ) - 1}")
    spec = sources[source]
    model = TraceModel.from_room(room)
    rng = np.random.default_rng(seed)
    c = room.speed_of_sound()
    air = float(room.air_attenuation()[band])  # energy, per metre

    origin = np.asarray(spec.position, dtype=float)
    directions = fibonacci_sphere(count)
    forward = np.asarray(spec.orientation.forward(), dtype=float)
    energy = directivity_weight(DIRECTIVITY_A[spec.directivity], forward, directions)

    # Segments: particle k leaves start[s, k] at time t0[s, k] along direction[s, k].
    t0 = [np.zeros(count)]
    start = [np.repeat(origin[None], count, axis=0)]
    direction = [directions]
    energies = [energy]
    alive = np.ones(count, dtype=bool)
    face = np.full(count, -1, dtype=np.int64)
    for _ in range(MAX_REFLECTIONS):
        p, d, t = start[-1], direction[-1], t0[-1]
        reach = np.where(alive, np.inf, 0.0)
        dist, hit = model.intersect(p, d, reach, face)
        alive &= hit >= 0
        dist = np.where(alive, dist, 0.0)
        face = np.where(alive, hit, -1)
        normal = model.plane_normal[model.face_plane[np.maximum(face, 0)]]
        # Point the normal back at the arriving particle, whichever way the face faces.
        normal = np.where((np.einsum("ij,ij->i", normal, d) > 0)[:, None], -normal, normal)
        specular = d - 2.0 * np.einsum("ij,ij->i", d, normal)[:, None] * normal
        scattered = rng.random(count) < model.scattering[np.maximum(face, 0), band]
        new_d = np.where(scattered[:, None], _lambert(normal, rng), specular)
        kept = (1.0 - model.absorption[np.maximum(face, 0), band]) * np.exp(-air * dist)
        # A particle that escaped keeps flying along its last segment (no next one).
        t_next = np.where(alive, t + dist / c, np.inf)
        t0.append(t_next)
        start.append(np.where(alive[:, None], p + dist[:, None] * d, p))
        direction.append(np.where(alive[:, None], new_d, d))
        energies.append(np.where(alive, energies[-1] * kept, energies[-1]))
        if not np.any(alive & (t_next <= duration)):
            break
        alive &= t_next <= duration
    t0_all, start_all = np.array(t0), np.array(start)
    direction_all, energy_all = np.array(direction), np.array(energies)

    times = np.linspace(0.0, duration, frames)
    positions = np.zeros((frames, count, 3))
    energy_out = np.zeros((frames, count))
    order_out = np.zeros((frames, count), dtype=np.int64)
    columns = np.arange(count)
    for f, time in enumerate(times):
        s = np.sum(t0_all <= time, axis=0) - 1  # the latest segment begun
        elapsed = time - t0_all[s, columns]
        positions[f] = start_all[s, columns] + c * elapsed[:, None] * direction_all[s, columns]
        energy_out[f] = energy_all[s, columns] * np.exp(-air * c * elapsed)
        order_out[f] = s
    return Particles(times, positions, energy_out, order_out, OCTAVE_BANDS_HZ[band], source)


__all__ = ["MAX_REFLECTIONS", "Particles", "trace_particles"]
