"""Frequency bands and conversion of coefficient data onto them."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Octave-band centre frequencies used for all frequency-dependent data (Hz).
OCTAVE_BANDS_HZ: tuple[int, ...] = (63, 125, 250, 500, 1000, 2000, 4000, 8000)


def to_octave_bands(
    frequencies: Sequence[float], values: Sequence[float]
) -> tuple[list[float], list[int]]:
    """Map coefficients given at arbitrary centre frequencies onto ``OCTAVE_BANDS_HZ``.

    Values whose centre falls inside an octave (third-octave data, for example) are
    averaged. An octave with no data takes the value of the nearest octave that has
    some; those octaves are returned as the second element so callers can say so.
    """
    freqs = np.asarray(frequencies, dtype=float)
    vals = np.asarray(values, dtype=float)
    if freqs.shape != vals.shape or freqs.ndim != 1 or len(freqs) == 0:
        raise ValueError("frequencies and values must be non-empty and of equal length")

    out = np.full(len(OCTAVE_BANDS_HZ), np.nan)
    for k, fc in enumerate(OCTAVE_BANDS_HZ):
        lo, hi = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
        inside = (freqs >= lo) & (freqs < hi)
        if inside.any():
            out[k] = float(vals[inside].mean())

    known = np.flatnonzero(~np.isnan(out))
    if len(known) == 0:
        raise ValueError("no data between 44 Hz and 11 kHz")
    extrapolated = [OCTAVE_BANDS_HZ[k] for k in np.flatnonzero(np.isnan(out))]
    for k in np.flatnonzero(np.isnan(out)):
        out[k] = out[known[np.argmin(np.abs(known - k))]]
    return [float(v) for v in out], extrapolated
