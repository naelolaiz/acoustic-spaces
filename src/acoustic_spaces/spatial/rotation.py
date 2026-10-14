"""Rotations: of directions (3 x 3 matrices from angles or quaternions) and of
Ambisonics sound fields (one matrix per order of the AmbiX channels).

Angles follow the scene's ``Orientation``: yaw turns left (from +x towards +y), pitch
looks up and roll tilts the head to the right, applied in the order roll, pitch, yaw. A
rotation matrix has the turned forward, left and up axes as its columns.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic_spaces.spatial.ambisonics import MAX_ORDER


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
    diagonal. The first-order block is the rotation itself, its axes in ACN order (y, z,
    x), and each higher one follows from the one below by Ivanic and Ruedenberg's
    recursion (J. Phys. Chem. 1996, with the 1998 correction), which is exact and takes
    well under a millisecond at seventh order.
    """

    def __init__(self, order: int) -> None:
        if not 0 <= order <= MAX_ORDER:
            raise ValueError(f"Ambisonics order must be 0 to {MAX_ORDER}, got {order}")
        self.order = order
        self.channels = (order + 1) ** 2
        self._steps = [_recursion(n) for n in range(2, order + 1)]

    def matrix(self, rotation: np.ndarray) -> np.ndarray:
        """(channels, channels): the field turned by ``rotation`` (3, 3)."""
        out = np.zeros((self.channels, self.channels))
        out[0, 0] = 1.0
        if self.order == 0:
            return out
        acn = [1, 2, 0]  # y, z, x
        below = np.asarray(rotation, dtype=float)[np.ix_(acn, acn)]  # m = -1, 0, 1
        out[1:4, 1:4] = below
        # Rows and columns of the first-order block for m = 0, 1, -1.
        first = below[np.ix_([1, 2, 0], [1, 2, 0])]
        for n, (u, v, w, rows) in enumerate(self._steps, start=2):
            # p[i, a, b]: Ivanic and Ruedenberg's P(i, a, b) for i = 0, 1, -1, with rows
            # a = -n - 1 to n + 1 (zero beyond n - 1) and columns b = -n to n.
            p = np.zeros((3, 2 * n + 3, 2 * n + 1))
            p[:, 2:-2, 1:-1] = first[:, 0, None, None] * below
            p[:, 2:-2, 0] = first[:, 1, None] * below[:, 0] + first[:, 2, None] * below[:, -1]
            p[:, 2:-2, -1] = first[:, 1, None] * below[:, -1] - first[:, 2, None] * below[:, 0]
            (iu,), (iv1, fv1, iv2, fv2), (iw1, fw1, iw2, fw2) = rows
            below = (
                u * p[0][iu]
                + v * (p[1][iv1] * fv1 + p[2][iv2] * fv2)
                + w * (p[1][iw1] * fw1 + p[2][iw2] * fw2)
            )
            out[n * n : (n + 1) ** 2, n * n : (n + 1) ** 2] = below
        return out


def _recursion(n: int):
    """For order ``n``: Ivanic and Ruedenberg's coefficients u, v and w (rows m, columns
    m' from -n to n), and for each row m the rows of P (offset by n + 1) and the factors
    that make their functions U, V and W."""
    m = np.arange(-n, n + 1)
    row, column = m[:, None], m[None, :]
    first = (row == 0).astype(float)
    size = np.abs(row)
    scale = np.where(np.abs(column) < n, (n + column) * (n - column), 2 * n * (2 * n - 1))
    u = np.sqrt((n + row) * (n - row) / scale)
    v = 0.5 * np.sqrt((1 + first) * (size + n - 1) * (size + n) / scale) * (1 - 2 * first)
    w = -0.5 * np.sqrt(np.maximum((n - size - 1) * (n - size), 0) / scale) * (1 - first)
    one = (np.abs(m) == 1).astype(float)
    up, down = m > 0, m < 0
    # V: P(1, m - 1) and P(-1, -m + 1) above m = 0, P(1, m + 1) and P(-1, -m - 1) below.
    iv1 = np.where(up, m - 1, np.where(down, m + 1, 1))
    fv1 = np.where(up, np.sqrt(1 + one), np.where(down, 1 - one, 1.0))
    iv2 = np.where(up, -m + 1, np.where(down, -m - 1, -1))
    fv2 = np.where(up, one - 1, np.where(down, np.sqrt(1 + one), 1.0))
    # W (none at m = 0, where w is 0): P(1, m + 1) + P(-1, -m - 1) above, P(1, m - 1) -
    # P(-1, -m + 1) below.
    iw1 = np.where(up, m + 1, np.where(down, m - 1, 0))
    iw2 = np.where(up, -m - 1, np.where(down, -m + 1, 0))
    fw1 = np.where(m == 0, 0.0, 1.0)
    fw2 = np.where(up, 1.0, np.where(down, -1.0, 0.0))
    offset = n + 1
    rows = (
        (m + offset,),
        (iv1 + offset, fv1[:, None], iv2 + offset, fv2[:, None]),
        (iw1 + offset, fw1[:, None], iw2 + offset, fw2[:, None]),
    )
    return u, v, w, rows


__all__ = ["FieldRotation", "angles_matrix", "matrix_angles", "quaternion_matrix"]
