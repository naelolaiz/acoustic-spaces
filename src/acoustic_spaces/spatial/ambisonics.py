"""Ambisonics in the AmbiX convention: ACN channel order, SN3D normalisation.

Channel ``n * n + n + m`` carries the real spherical harmonic of order n and degree m,
without the Condon-Shortley phase and normalised so every order peaks at 1. First order
is then W = 1, Y = y, Z = z, X = x for a plane wave from the unit vector (x, y, z).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Highest order the encoder accepts.
MAX_ORDER = 7


def real_spherical_harmonics(order: int, directions: np.ndarray) -> np.ndarray:
    """ACN / SN3D spherical harmonics (dirs, (order + 1) ** 2) of unit vectors (dirs, 3)."""
    if not 0 <= order <= MAX_ORDER:
        raise ValueError(f"Ambisonics order must be 0 to {MAX_ORDER}, got {order}")
    v = np.atleast_2d(np.asarray(directions, dtype=float))
    length = np.linalg.norm(v, axis=1, keepdims=True)
    if not np.all(length > 0):
        raise ValueError("every direction needs a length above zero (and no NaN)")
    v = v / length
    sin_el = np.clip(v[:, 2], -1.0, 1.0)
    cos_el = np.sqrt(np.maximum(0.0, 1.0 - sin_el**2))
    azimuth = np.arctan2(v[:, 1], v[:, 0])
    out = np.empty((len(v), (order + 1) ** 2))
    for m in range(order + 1):
        # Associated Legendre functions P_l^m(sin el) for l = m .. order, by recurrence.
        p_mm = math.prod(range(1, 2 * m, 2)) * cos_el**m  # (2m - 1)!! cos^m
        legendre = {m: p_mm}
        if m < order:
            legendre[m + 1] = sin_el * (2 * m + 1) * p_mm
        for n in range(m + 2, order + 1):
            legendre[n] = (
                (2 * n - 1) * sin_el * legendre[n - 1] - (n + m - 1) * legendre[n - 2]
            ) / (n - m)
        for n in range(m, order + 1):
            norm = math.sqrt(
                (1.0 if m == 0 else 2.0) * math.factorial(n - m) / math.factorial(n + m)
            )
            out[:, n * n + n + m] = norm * legendre[n] * np.cos(m * azimuth)
            if m > 0:
                out[:, n * n + n - m] = norm * legendre[n] * np.sin(m * azimuth)
    return out


_FIRST_ORDER = ("W", "Y", "Z", "X")


@dataclass(frozen=True)
class Ambisonics:
    """AmbiX (ACN / SN3D) encoding of the given order: (order + 1) ** 2 channels."""

    order: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.order <= MAX_ORDER:
            raise ValueError(f"Ambisonics order must be 0 to {MAX_ORDER}, got {self.order}")

    @property
    def channels(self) -> int:
        return (self.order + 1) ** 2

    @property
    def label(self) -> str:
        return f"ambix{self.order}"

    @property
    def channel_names(self) -> list[str]:
        if self.order <= 1:
            return list(_FIRST_ORDER[: self.channels])
        return [f"ACN{k}" for k in range(self.channels)]

    def gains(self, directions: np.ndarray) -> np.ndarray:
        """Encoding gains (dirs, channels) of plane waves from head directions (dirs, 3)."""
        return real_spherical_harmonics(self.order, directions)


__all__ = ["MAX_ORDER", "Ambisonics", "real_spherical_harmonics"]
