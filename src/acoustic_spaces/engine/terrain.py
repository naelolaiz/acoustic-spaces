"""The early paths over the ground of a terrain.

A terrain (``geometry.ground.terrain``) is a grid of heights cut into triangles. The
early stage reflects sound off every face as off a plane, and off a gently rolling
ground that goes wrong in two ways: a facet 5 m across sends a 5 m wave back as a
plain would, and as a listener moves, the reflections off neighbouring facets come
and go in jumps, two at once where the ground bulges and none where it hollows. So
the early paths of a scene with a terrain do not reflect off its ground faces at all
(the stochastic late tail still bounces its rays off every face). In their place the
ground reflects once, off the **mean plane** of the terrain between the source and the
listener, as the outdoor standards (CNOSSOS-EU, Nord2000) have it: the plane through
the least-squares line of the ground's profile in the vertical plane between the two
(:func:`mean_plane`), level across it. The sound reflects off it where the image of
the source in the plane sees the listener; a source or listener below the plane
counts as on it, so that the reflection grazes the ground, as the standards' equivalent
heights have it. The reflection takes the ground's reflection factor under the
reflection point (its material, or the grounds of the zones its Fresnel zone covers,
as on a flat ground) and nothing may stand in its way but the ground itself. Over a
plain, a flat terrain or a plane slope the mean plane is the ground, and the
reflection is the one the flat ground gives. The sound bent over the terrain's crests
reflects off the mean planes on each side of them in the same way
(:mod:`acoustic_spaces.engine.crests`). Nothing of this touches a scene without a
terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.coherent import ground_factors, reflection_phases
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import TraceModel, _ground_faces, source_weights

_TINY = 1e-9


def mean_plane(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean plane of the ground along ``profile`` (k, 3), the points of the
    terrain's profile between two points: a point on it and its unit normal, upwards.
    The plane holds the least-squares line of the heights against the distance along
    the profile, and the horizontal across the profile."""
    profile = np.asarray(profile, dtype=float)
    across = profile[-1, :2] - profile[0, :2]
    run = float(np.linalg.norm(across))
    axis = across / run if run > _TINY else np.array([1.0, 0.0])
    s = (profile[:, :2] - profile[0, :2]) @ axis
    z = profile[:, 2]
    sm, zm = s.mean(), z.mean()
    spread = float(((s - sm) ** 2).sum())
    slope = float(((s - sm) * (z - zm)).sum() / spread) if spread > _TINY else 0.0
    origin = np.array([profile[0, 0], profile[0, 1], zm - slope * sm])
    normal = np.array([-slope * axis[0], -slope * axis[1], 1.0])
    return origin, normal / np.linalg.norm(normal)


@dataclass(frozen=True)
class GroundReflection:
    """A reflection off the mean plane between two points: where (``at``), off which
    ground face, the two points as the plane has them (on it when below it) and their
    heights over it."""

    at: np.ndarray
    face: int
    first: np.ndarray
    second: np.ndarray
    height_first: float
    height_second: float
    normal: np.ndarray

    @property
    def image(self) -> np.ndarray:
        """The image of the first point in the plane."""
        return self.first - 2.0 * self.height_first * self.normal

    @property
    def cosine(self) -> float:
        """The cosine of the angle of incidence."""
        leg = float(np.linalg.norm(self.at - self.first))
        return self.height_first / leg if leg > _TINY else 0.0


def ground_reflection(
    model: TraceModel, a: np.ndarray, b: np.ndarray, profile: np.ndarray, ground: np.ndarray
) -> GroundReflection | None:
    """The reflection off the mean plane of ``profile`` (the terrain between ``a`` and
    ``b``) on the way from ``a`` to ``b``, or None when the reflection point lies off
    the ground (past its edge, or under a building) or something stands in the way
    (``ground`` masks the terrain's faces, which do not)."""
    origin, normal = mean_plane(profile)
    heights = [float((p - origin) @ normal) for p in (a, b)]
    feet = [p - h * normal for p, h in zip((a, b), heights, strict=True)]
    heights = [max(h, 0.0) for h in heights]
    total = heights[0] + heights[1]
    t = heights[0] / total if total > _TINY else 0.5
    at = feet[0] + t * (feet[1] - feet[0])
    face = ground_face_under(model, at, ground)
    if face < 0:
        return None
    first, second = (feet[k] + heights[k] * normal for k in range(2))
    from acoustic_spaces.engine.diffraction import _clear

    legs = np.array([a, at]), np.array([at, b])
    if not _clear(model, legs[0], legs[1], through=ground).all():
        return None
    return GroundReflection(at, int(face), first, second, heights[0], heights[1], normal)


def ground_face_under(model: TraceModel, point: np.ndarray, ground: np.ndarray) -> int:
    """The ground face straight under (or over) ``point``, or -1 when a building's
    roof or nothing lies there."""
    layout = model.room.outdoor
    start = np.array([[point[0], point[1], layout.sky - 1e-6]])
    length = np.array([layout.sky - layout.ground + 1.0])
    _, hit = model.intersect(start, np.array([[0.0, 0.0, -1.0]]), length, np.array([-1]))
    if hit[0] < 0 or not ground[hit[0]]:
        return -1
    return int(hit[0])


def reflection_factor(
    model: TraceModel,
    found: GroundReflection,
    distance: float,
    before: np.ndarray,
    after: np.ndarray,
    coherent: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(bands,) the energy share of a reflection off the mean plane along a path
    ``distance`` m long, from ``before`` to ``after``, and (C, M) its pressure's shape
    in the lowest ``coherent`` bands (ones when none)."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    face = np.array([found.face])
    incoming = (found.at - found.first)[None]
    share = model.reflection(face, incoming, 1, impedance=True)[0]
    spectrum = np.ones((coherent, band_frequencies(bands=model.bands)[0].shape[1]), complex)
    cosine = np.array([found.cosine])
    factor = None
    if coherent:
        spectrum = reflection_phases(model, face, cosine, coherent)[0]
    if model.factor_faces(face).any():
        ends = (before[None], found.at[None], after[None])
        factor = ground_factors(model, face, cosine, np.array([distance]), ends=ends)[0]
        gain = np.mean(np.abs(factor) ** 2, axis=1)  # (bands,)
        share = share * gain
        if coherent:
            level = np.sqrt(gain[:coherent, None])
            spectrum = spectrum * np.where(
                level > 0.0, factor[:coherent] / np.where(level > 0.0, level, 1.0), 1.0
            )
    return share, spectrum


def terrain_paths(
    model: TraceModel,
    paths: EarlyReflections,
    source: np.ndarray,
    listener: np.ndarray,
    *,
    speed_of_sound: float,
    air: np.ndarray,
    max_delay_s: float,
    directivity: Directivity | None = None,
    orientation: np.ndarray | None = None,
    coherent: int = 0,
) -> EarlyReflections:
    """``paths`` (from :func:`~acoustic_spaces.engine.early.find_paths`) over a
    terrain: without those that reflect off its ground faces, and with the reflection
    off the mean plane of the ground between the source and the listener, when the
    direct sound reaches it (the sound bent over the crests in the way takes its
    reflections in :mod:`acoustic_spaces.engine.crests`)."""
    layout = model.room.outdoor
    if layout is None or layout.terrain is None:
        return paths
    ground = _ground_faces(model.room)
    if len(paths) and paths.faces:
        keep = np.array([not ground[np.asarray(f, dtype=np.int64)].any() for f in paths.faces])
        paths = paths[keep]
    source = np.asarray(source, dtype=float)
    listener = np.asarray(listener, dtype=float)
    direct = (paths.order == 0) & (paths.edge == -1)
    if not direct.any() or float(np.linalg.norm(listener[:2] - source[:2])) <= _TINY:
        return paths
    profile = layout.terrain.profile(source[:2], listener[:2])
    found = ground_reflection(model, source, listener, profile, ground)
    if found is None:
        return paths
    legs = np.array([found.at - source, listener - found.at])
    lengths = np.linalg.norm(legs, axis=1)
    distance = float(lengths.sum())
    if distance / speed_of_sound > max_delay_s:
        return paths
    bands = model.absorption.shape[1]
    air = np.asarray(air, dtype=float)
    share, spectrum = reflection_factor(model, found, distance, source, listener, coherent)
    energy = share * np.exp(-air * distance) / (4.0 * np.pi * distance**2)
    departure = legs[0] / max(lengths[0], 1e-300)
    energy = (
        energy
        * source_weights(
            directivity, orientation, departure, bands, distances=np.array([distance])
        )[0]
    )
    arrival = -legs[1] / max(lengths[1], 1e-300)
    reflected = EarlyReflections(
        order=np.ones(1, dtype=np.int64),
        delay_s=np.array([distance / speed_of_sound]),
        distance_m=np.array([distance]),
        direction=arrival[None],
        energy=energy[None],
        points=[np.vstack([source, found.at, listener])],
        faces=[np.array([found.face], dtype=np.int64)],
        spectrum=spectrum[None] if coherent else None,
    )
    return EarlyReflections.concatenate([paths, reflected], bands)


__all__ = [
    "GroundReflection",
    "ground_face_under",
    "ground_reflection",
    "mean_plane",
    "reflection_factor",
    "terrain_paths",
]
