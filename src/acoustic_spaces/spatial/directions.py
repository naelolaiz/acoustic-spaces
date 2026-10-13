"""Directions: unit vectors, angles and grids.

Head coordinates follow SOFA and AmbiX: x forward, y left, z up. Azimuth turns from
the front towards the left (counterclockwise seen from above) and elevation is positive
upwards, both in degrees unless a name says otherwise.
"""

from __future__ import annotations

import math

import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform unit vectors (n, 3) on a Fibonacci lattice, each standing for
    an equal share of the sphere."""
    i = np.arange(n, dtype=float)
    z = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def from_angles(azimuth_deg, elevation_deg) -> np.ndarray:
    """Unit vectors (..., 3) from azimuth and elevation in degrees."""
    az, el = np.broadcast_arrays(
        np.radians(np.asarray(azimuth_deg, dtype=float)),
        np.radians(np.asarray(elevation_deg, dtype=float)),
    )
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def to_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth in (-180, 180] and elevation in [-90, 90] degrees of vectors (..., 3)."""
    v = np.asarray(directions, dtype=float)
    azimuth = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    elevation = np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))
    return azimuth, elevation


def to_head(directions: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Room directions (..., 3) in the head coordinates of a listener whose ``rotation``
    has the forward, left and up axes as columns (see ``Orientation.rotation``)."""
    return np.asarray(directions, dtype=float) @ np.asarray(rotation, dtype=float)


__all__ = ["fibonacci_sphere", "from_angles", "to_angles", "to_head"]
