"""Rotations: of directions (3 x 3 matrices from angles or quaternions) and of
Ambisonics sound fields (one matrix per order of the AmbiX channels).

Angles follow the scene's ``Orientation``: yaw turns left (from +x towards +y), pitch
looks up and roll tilts the head to the right, applied in the order roll, pitch, yaw. A
rotation matrix has the turned forward, left and up axes as its columns.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic_spaces.spatial.ambisonics import MAX_ORDER, real_spherical_harmonics
from acoustic_spaces.spatial.directions import fibonacci_sphere


def angles_matrix(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """The rotation (3, 3) of yaw, pitch and roll in degrees."""
    from acoustic_spaces.scene.model import Orientation

    return np.array(Orientation(yaw=yaw, pitch=pitch, roll=roll).rotation())


def quaternion_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """The rotation (3, 3) of a quaternion, normalised first (identity if it is zero)."""
    q = np.array([w, x, y, z], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm == 0.0 or not math.isfinite(norm):
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch and roll in degrees of a rotation (3, 3); the inverse of
    :func:`angles_matrix` (yaw and roll are 0 to 180 either way, pitch up to 90)."""
    r = np.asarray(rotation, dtype=float)
    forward, left, up = r[:, 0], r[:, 1], r[:, 2]
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    pitch = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    roll = math.degrees(math.atan2(left[2], up[2]))
    return yaw + 0.0, pitch + 0.0, roll + 0.0


class FieldRotation:
    """Rotations of an Ambisonics (ACN / SN3D) sound field of ``order``.

    :meth:`matrix` gives the (channels, channels) matrix that turns the channels of any
    sound field into those of the same field turned by a rotation: a plane wave from
    ``d`` comes from ``R @ d`` afterwards. Orders do not mix, so the matrix is block
    diagonal; each block is fitted to the spherical harmonics on a grid of directions
    (exact, as the harmonics of one order turn into each other).
    """

    def __init__(self, order: int) -> None:
        if not 0 <= order <= MAX_ORDER:
            raise ValueError(f"Ambisonics order must be 0 to {MAX_ORDER}, got {order}")
        self.order = order
        self.channels = (order + 1) ** 2
        self._grid = fibonacci_sphere(max(4 * self.channels, 16))
        harmonics = real_spherical_harmonics(order, self._grid)
        self._blocks = [
            (slice(n * n, (n + 1) ** 2), np.linalg.pinv(harmonics[:, n * n : (n + 1) ** 2]))
            for n in range(order + 1)
        ]

    def matrix(self, rotation: np.ndarray) -> np.ndarray:
        """(channels, channels): the field turned by ``rotation`` (3, 3)."""
        turned = real_spherical_harmonics(self.order, self._grid @ np.asarray(rotation).T)
        out = np.zeros((self.channels, self.channels))
        for block, inverse in self._blocks:
            out[block, block] = (inverse @ turned[:, block]).T
        return out


__all__ = ["FieldRotation", "angles_matrix", "matrix_angles", "quaternion_matrix"]
