"""Grazing incidence over the audience: sound that reaches a listener low over rows of
seats loses much more than the seats' absorption says, as it passes heads and seat
backs on the way.

The model is the fit of Jurkiewicz, Rummler, Green, Wulfrank and Kahle ("Impact of
grazing incidence attenuation on early reflection design", Proc. IOA 47 Pt. 1, 2025) to
measurements over a seated audience in a 1:20 scale model. Sound arriving at an
elevation φ above the audience plane, having travelled d metres over the audience,
keeps

    min(1, (3.40 sin²φ + 2.17 sin φ + 0.02) ^ ((d - 1) / 10))

of its energy: 17 dB less at grazing incidence after 11 m, 0.6 dB per metre at 5°, and
nothing lost from 17.8° up. It was fitted in the 500 Hz to 2 kHz octave bands, where the
attenuation hardly changed with frequency, and applies there only (in third octaves,
400 Hz to 2.5 kHz). Lower down the seat dip depends on the seats themselves, and higher
up the fit has no data.

Here it applies to the last leg of each early path (the direct sound, a reflection or a
diffracted path) to a listener above a surface whose material is ``audience``: φ is the
leg's elevation above that surface's plane, and d how far the leg runs over audience
surfaces no higher than ``CLEARANCE_M`` above the listener's own height over the
audience (so the stalls do not shade a balcony seat). The leg from the source to the
first reflection is not attenuated, as in the fit, which therefore errs on the side of
too little attenuation. The late part is left as it is; the statistical method reckons
its diffuse field less the early paths as they would come without the audience, so what
the seats take from them is lost and not made up by the tail.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import TraceModel

#: The bands the fit covers (Hz): the 500 Hz to 2 kHz octaves and their third octaves.
FIT_RANGE_HZ = (400, 2500)
#: Coefficients of the fit's polynomial in sin φ: a sin²φ + b sin φ + c.
FIT = (3.40, 2.17, 0.02)
#: The elevation from which nothing is lost (the polynomial's value is 1): 17.8°.
_A, _B, _C = FIT
MAX_ELEVATION_RAD = math.asin((-_B + math.sqrt(_B**2 + 4 * _A * (1 - _C))) / (2 * _A))
#: Highest a listener's ears may be above an audience surface for it to count (m).
MAX_EAR_HEIGHT_M = 2.0
#: How much higher than the listener's ears above the audience a leg still counts as
#: running over it (m).
CLEARANCE_M = 2.0
#: Step along a leg when measuring how far it runs over the audience (m).
STEP_M = 0.25
#: Longest stretch of a leg measured (m).
MAX_DISTANCE_M = 60.0


def attenuation(elevation_rad, distance_m) -> np.ndarray:
    """The share of energy left after ``distance_m`` over the audience at
    ``elevation_rad`` above its plane (the fit above; the first metre is free)."""
    s = np.sin(np.minimum(np.abs(np.asarray(elevation_rad, dtype=float)), math.pi / 2))
    base = FIT[0] * s**2 + FIT[1] * s + FIT[2]
    exponent = np.maximum(np.asarray(distance_m, dtype=float) - 1.0, 0.0) / 10.0
    return np.minimum(1.0, base**exponent)


def audience_faces(model: TraceModel) -> np.ndarray:
    """(F,) faces whose material is audience seating."""
    room = model.room
    flags = [room.scene.materials[n].audience for n in room.material_names]
    return np.array(flags + [False], dtype=bool)[room.face_materials]


def fit_bands(model: TraceModel) -> np.ndarray:
    """(bands,) the model's bands the fit covers."""
    centres = np.asarray(model.band_hz, dtype=float)
    return (centres >= FIT_RANGE_HZ[0]) & (centres <= FIT_RANGE_HZ[1])


def _below(model: TraceModel, points: np.ndarray, reach: np.ndarray):
    """The face straight below each point within ``reach`` (or -1), and how far down."""
    down = np.broadcast_to([0.0, 0.0, -1.0], points.shape)
    t, face = model.intersect(points, down, reach, np.full(len(points), -1))
    return face, t


def grazing_factors(
    model: TraceModel, listener: np.ndarray, paths: EarlyReflections, audience: np.ndarray
) -> np.ndarray | None:
    """(paths, bands) share of each path's energy left by grazing incidence over the
    audience (1 outside the fit's bands), or None when the listener is not above an
    audience surface."""
    listener = np.asarray(listener, dtype=float)
    face, height = _below(model, listener[None], np.array([MAX_EAR_HEIGHT_M]))
    if face[0] < 0 or not audience[face[0]]:
        return None
    normal = model.plane_normal[model.face_plane[face[0]]]
    normal = normal if normal[2] >= 0 else -normal
    bands = fit_bands(model)
    out = np.ones((len(paths), model.absorption.shape[1]))
    if len(paths) == 0 or not bands.any():
        return out
    previous = np.array([p[-2] for p in paths.points]) if paths.points else None
    if previous is None or len(previous) != len(paths):
        return out
    leg = listener - previous
    length = np.linalg.norm(leg, axis=1)
    direction = leg / np.maximum(length, 1e-300)[:, None]
    elevation = np.arcsin(np.clip(np.abs(direction @ normal), 0.0, 1.0))
    low = np.flatnonzero((elevation < MAX_ELEVATION_RAD) & (length > STEP_M))
    if len(low) == 0:
        return out
    # How far each low leg runs over the audience: points every STEP_M back from the
    # listener, counted where an audience surface lies not too far below.
    steps = int(math.ceil(min(float(length[low].max()), MAX_DISTANCE_M) / STEP_M))
    along = (np.arange(steps) + 0.5) * STEP_M
    points = listener - direction[low, None, :] * along[None, :, None]  # (L, S, 3)
    inside = along[None, :] < length[low, None]
    reach = np.full(int(inside.sum()), float(height[0]) + CLEARANCE_M)
    below, _ = _below(model, points[inside], reach)
    over = np.zeros(inside.shape, dtype=bool)
    over[inside] = (below >= 0) & audience[np.maximum(below, 0)]
    distance = over.sum(axis=1) * STEP_M
    left = attenuation(elevation[low], distance)
    out[np.ix_(low, np.flatnonzero(bands))] = left[:, None]
    return out


def apply_grazing(model: TraceModel, listener: np.ndarray, paths: EarlyReflections):
    """``paths`` to ``listener`` with the grazing attenuation over the audience applied
    to their energies; unchanged when the model has no audience surfaces."""
    audience = audience_faces(model)
    if not audience.any():
        return paths
    factors = grazing_factors(model, listener, paths, audience)
    if factors is None:
        return paths
    return EarlyReflections(
        order=paths.order,
        delay_s=paths.delay_s,
        distance_m=paths.distance_m,
        direction=paths.direction,
        energy=paths.energy * factors,
        points=paths.points,
        faces=paths.faces,
        edge=paths.edge,
        crossings=paths.crossings,
    )


__all__ = [
    "CLEARANCE_M",
    "FIT",
    "FIT_RANGE_HZ",
    "MAX_EAR_HEIGHT_M",
    "MAX_ELEVATION_RAD",
    "apply_grazing",
    "attenuation",
    "audience_faces",
    "fit_bands",
    "grazing_factors",
]
