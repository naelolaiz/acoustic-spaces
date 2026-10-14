"""Binaural decoding of Ambisonics: filters that turn the AmbiX channels of a sound
field into the signals at the two ears.

The filters fit the HRTF of every measured direction with the spherical harmonics up to
the order (least squares, ``"ls"``). Low orders cannot follow the fast changes of the
HRTF's phase with direction at high frequencies, and least squares then loses level
and spaciousness; above a cutoff (the order times 600 Hz, after Schörkhuber,
Zaunschirm and Höldrich 2018) the magnitude least squares decoder (``"magls"``, the
default) fits only the magnitudes and keeps the phase that varies smoothly from the bin
below.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic_spaces.spatial.ambisonics import MAX_ORDER, real_spherical_harmonics
from acoustic_spaces.spatial.hrtf import HRTF

#: Decoding methods.
DECODERS = ("magls", "ls")
#: Above this many hertz per order, ``magls`` fits magnitudes only.
MAGLS_HZ_PER_ORDER = 600.0


def binaural_decoder(order: int, hrtf: HRTF, method: str = "magls") -> np.ndarray:
    """Filters (taps, (order + 1) ** 2, 2) from the AmbiX channels of ``order`` to the
    left and right ear, at the HRTF's sample rate. A plane wave encoded from a direction
    decodes to about that direction's HRTF."""
    if not 0 <= order <= MAX_ORDER:
        raise ValueError(f"Ambisonics order must be 0 to {MAX_ORDER}, got {order}")
    if method not in DECODERS:
        raise ValueError(f"unknown decoder {method!r}; choose from {', '.join(DECODERS)}")
    harmonics = real_spherical_harmonics(order, hrtf.directions)  # (directions, channels)
    inverse = np.linalg.pinv(harmonics)
    # Fit on a fine frequency grid, then keep a filter twice the HRIRs' length.
    fine = _power_of_two(max(8 * hrtf.taps, 2048))
    taps = _power_of_two(max(2 * hrtf.taps, 64))
    spectra = np.fft.rfft(hrtf.hrir, fine, axis=-1).transpose(2, 0, 1)  # (bins, dirs, 2)
    filters = inverse @ spectra  # (bins, channels, 2)
    if method == "magls":
        frequencies = np.fft.rfftfreq(fine, 1.0 / hrtf.sample_rate)
        start = int(np.searchsorted(frequencies, MAGLS_HZ_PER_ORDER * max(order, 1)))
        # Each bin takes the phase of the fit one bin below, advanced by the delay
        # common to all the HRIRs, so the filters keep the HRIRs' timing.
        delay = float(np.median(np.argmax(np.abs(hrtf.hrir), axis=-1)))
        advance = np.exp(-2j * math.pi * delay / fine)
        magnitudes = np.abs(spectra)
        for k in range(max(start, 1), len(frequencies)):
            phase = np.exp(1j * np.angle(harmonics @ filters[k - 1])) * advance
            filters[k] = inverse @ (magnitudes[k] * phase)
    out = np.fft.irfft(filters, fine, axis=0)[:taps]  # (taps, channels, 2)
    fade = taps // 8
    out[taps - fade :] *= (0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, fade)))[:, None, None]
    return np.ascontiguousarray(out)


def _power_of_two(n: int) -> int:
    return 1 << int(math.ceil(math.log2(n)))


__all__ = ["DECODERS", "MAGLS_HZ_PER_ORDER", "binaural_decoder"]
