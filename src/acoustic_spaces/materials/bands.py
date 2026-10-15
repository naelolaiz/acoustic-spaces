"""Frequency bands and conversion of coefficient data onto them.

A simulation runs in octave bands (``OCTAVE_BANDS_HZ``, the default) or in third-octave
bands (``THIRD_OCTAVE_BANDS_HZ``), three to each octave band with its centre in the
middle. Coefficients come in either; each list converts to the other: an octave value
holds for its three third-octave bands, and an octave takes the mean of its three.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

#: Octave-band centre frequencies used for all frequency-dependent data (Hz).
OCTAVE_BANDS_HZ: tuple[int, ...] = (63, 125, 250, 500, 1000, 2000, 4000, 8000)
#: Third-octave-band centre frequencies (Hz, nominal, IEC 61260-1): three to each octave
#: band of ``OCTAVE_BANDS_HZ``, from 50 Hz to 10 kHz.
THIRD_OCTAVE_BANDS_HZ: tuple[int, ...] = (
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
)  # fmt: skip

#: How finely a simulation splits the spectrum.
Resolution = Literal["octave", "third_octave"]
#: Bands per octave at each resolution.
BANDS_PER_OCTAVE: dict[str, int] = {"octave": 1, "third_octave": 3}


def band_centres(resolution: Resolution = "octave") -> tuple[int, ...]:
    """Nominal centre frequencies (Hz) of the bands at ``resolution``."""
    if resolution == "octave":
        return OCTAVE_BANDS_HZ
    if resolution == "third_octave":
        return THIRD_OCTAVE_BANDS_HZ
    raise ValueError(f"unknown band resolution {resolution!r}; use 'octave' or 'third_octave'")


def resolution_of(bands: int) -> Resolution:
    """The resolution with ``bands`` bands: 8 octave bands or 24 third-octave bands."""
    if bands == len(OCTAVE_BANDS_HZ):
        return "octave"
    if bands == len(THIRD_OCTAVE_BANDS_HZ):
        return "third_octave"
    raise ValueError(
        f"expected {len(OCTAVE_BANDS_HZ)} octave or {len(THIRD_OCTAVE_BANDS_HZ)} "
        f"third-octave bands, got {bands}"
    )


def exact_centres(resolution: Resolution = "octave") -> np.ndarray:
    """Exact mid-band frequencies (Hz, base 10, IEC 61260-1) of the bands at ``resolution``."""
    step = 0.3 / BANDS_PER_OCTAVE[resolution]
    k = np.round(np.log10(np.asarray(band_centres(resolution), float) / 1000.0) / step)
    return 1000.0 * 10.0 ** (step * k)


def band_limits(resolution: Resolution = "octave") -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edges (Hz) of each band: its nominal centre over and times
    2^(1/2) for octaves, 2^(1/6) for third octaves."""
    centre = np.asarray(band_centres(resolution), float)
    half = 2.0 ** (0.5 / BANDS_PER_OCTAVE[resolution])
    return centre / half, centre * half


def crossover_frequencies(resolution: Resolution = "octave") -> np.ndarray:
    """Edges between adjacent bands (Hz): the octave bands' nominal centres times
    2^(1/2), and between third octaves 1000 Hz times 2^((k + 1/2) / 3)."""
    if resolution == "octave":
        return np.array(OCTAVE_BANDS_HZ[:-1], dtype=float) * math.sqrt(2.0)
    k = np.arange(len(THIRD_OCTAVE_BANDS_HZ) - 1) - THIRD_OCTAVE_BANDS_HZ.index(1000)
    return 1000.0 * 2.0 ** ((k + 0.5) / 3.0)


def octave_of(resolution: Resolution = "octave") -> np.ndarray:
    """The index in ``OCTAVE_BANDS_HZ`` of the octave band each band lies in."""
    per = BANDS_PER_OCTAVE[resolution]
    return np.arange(len(band_centres(resolution))) // per


def convert(values: Sequence[float], resolution: Resolution) -> list[float]:
    """Per-band ``values`` (octave or third-octave) at ``resolution``: an octave value
    for each of its three third octaves, or the mean of three third octaves for their
    octave."""
    values = np.asarray(values, dtype=float)
    have = resolution_of(len(values))
    if have == resolution:
        return [float(v) for v in values]
    if resolution == "third_octave":
        return [float(v) for v in np.repeat(values, 3)]
    return [float(v) for v in values.reshape(-1, 3).mean(axis=1)]


def to_octaves(energy: np.ndarray, axis: int = 0) -> np.ndarray:
    """Energies with octave or third-octave bands along ``axis``, in octave bands: the
    mean of each octave's three third octaves.

    The simulation's band energies are spectral levels (a unit source's direct sound
    carries 1 / (4 pi r^2) in every band, however wide), so an octave's is the mean of
    its thirds', each covering a third of it.
    """
    energy = np.asarray(energy, dtype=float)
    axis = axis % energy.ndim
    if resolution_of(energy.shape[axis]) == "octave":
        return energy
    shape = energy.shape[:axis] + (len(OCTAVE_BANDS_HZ), 3) + energy.shape[axis + 1 :]
    return energy.reshape(shape).mean(axis=axis + 1)


def to_bands(
    frequencies: Sequence[float], values: Sequence[float], resolution: Resolution = "octave"
) -> tuple[list[float], list[int]]:
    """Map coefficients given at arbitrary centre frequencies onto the bands at
    ``resolution``.

    Values whose centre falls inside a band (third-octave data in an octave band, for
    example) are averaged. A band with no data takes the value of the nearest band that
    has some; those bands are returned as the second element so callers can say so.
    """
    freqs = np.asarray(frequencies, dtype=float)
    vals = np.asarray(values, dtype=float)
    if freqs.shape != vals.shape or freqs.ndim != 1 or len(freqs) == 0:
        raise ValueError("frequencies and values must be non-empty and of equal length")

    centres = band_centres(resolution)
    if resolution == "octave":
        low, high = band_limits(resolution)
    else:
        edges = crossover_frequencies(resolution)
        low = np.concatenate([[edges[0] / 2.0 ** (1 / 3)], edges])
        high = np.concatenate([edges, [edges[-1] * 2.0 ** (1 / 3)]])
    out = np.full(len(centres), np.nan)
    for k in range(len(centres)):
        inside = (freqs >= low[k]) & (freqs < high[k])
        if inside.any():
            out[k] = float(vals[inside].mean())

    known = np.flatnonzero(~np.isnan(out))
    if len(known) == 0:
        raise ValueError(f"no data between {low[0]:.0f} Hz and {high[-1] / 1000:.1f} kHz")
    extrapolated = [centres[k] for k in np.flatnonzero(np.isnan(out))]
    for k in np.flatnonzero(np.isnan(out)):
        out[k] = out[known[np.argmin(np.abs(known - k))]]
    return [float(v) for v in out], extrapolated


def to_octave_bands(
    frequencies: Sequence[float], values: Sequence[float]
) -> tuple[list[float], list[int]]:
    """Map coefficients given at arbitrary centre frequencies onto ``OCTAVE_BANDS_HZ``
    (:func:`to_bands` at octave resolution)."""
    return to_bands(frequencies, values, "octave")


def finer_than_octaves(frequencies: Sequence[float]) -> bool:
    """Whether data at ``frequencies`` has more than one value in some octave band, so
    third-octave bands keep detail that octave bands would average away."""
    freqs = np.asarray(frequencies, dtype=float)
    low, high = band_limits("octave")
    return any(
        int(((freqs >= lo) & (freqs < hi)).sum()) > 1 for lo, hi in zip(low, high, strict=True)
    )


__all__ = [
    "BANDS_PER_OCTAVE",
    "OCTAVE_BANDS_HZ",
    "THIRD_OCTAVE_BANDS_HZ",
    "Resolution",
    "band_centres",
    "band_limits",
    "convert",
    "crossover_frequencies",
    "exact_centres",
    "finer_than_octaves",
    "octave_of",
    "resolution_of",
    "to_bands",
    "to_octave_bands",
    "to_octaves",
]
