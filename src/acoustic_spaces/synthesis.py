"""Impulse response synthesis from octave-band (or third-octave-band) energies.

The filter bank splits the spectrum into the eight octave bands with zero-phase,
half-octave raised-cosine crossovers, or into the 24 third-octave bands with crossovers
a sixth of an octave wide. The band gains sum to exactly one, so a signal split into
bands and summed again comes back unchanged (perfect reconstruction). Each function
takes the bands from the number of band signals or energies it is given (8 or 24).

Energy convention: the simulation gives time-integrated intensity for a source of unit
energy, so the direct sound at r metres carries 1 / (4 pi r^2). Impulse responses are
scaled by ``ENERGY_TO_IR`` so that this direct sound is an impulse of height 1 / r:
1 at 1 m, and 0.01 of energy at 10 m (the reference for strength G).

A path or time slot with energy ``E`` in every band becomes a broadband impulse or
noise burst of energy ``E``. Band energies are spectral levels, so the part of the
output inside band b carries ``E_b`` times that band's share of the spectrum.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from acoustic_spaces.materials import bands as _bands
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

#: Impulse response energy per unit of simulated energy (time-integrated intensity).
ENERGY_TO_IR = 4.0 * math.pi
#: Energy of the impulse response of the direct sound at 10 m, the reference for G.
REFERENCE_ENERGY_10M = 0.01
#: Width of each crossover between octave bands, in octaves: half a band.
CROSSOVER_WIDTH = 0.5
#: Zero padding around FFT filtering, seconds; longer than the crossovers ring.
_PAD_S = 0.2


def crossover_frequencies(bands: int = len(OCTAVE_BANDS_HZ)) -> np.ndarray:
    """Edges between adjacent bands (Hz) of the 8 octave bands or 24 third-octave bands."""
    return _bands.crossover_frequencies(_bands.resolution_of(bands))


def crossover_width(bands: int = len(OCTAVE_BANDS_HZ)) -> float:
    """Width of each crossover (octaves): half a band."""
    return CROSSOVER_WIDTH / _bands.BANDS_PER_OCTAVE[_bands.resolution_of(bands)]


def band_gains(n_fft: int, sample_rate: float, bands: int = len(OCTAVE_BANDS_HZ)) -> np.ndarray:
    """Zero-phase gains (bands, n_fft // 2 + 1) on the rFFT grid of the 8 octave bands
    or 24 third-octave bands; they sum to one."""
    return _band_gains(int(n_fft), float(sample_rate), int(bands)).copy()


@functools.lru_cache(maxsize=8)
def _band_gains(n_fft: int, sample_rate: float, bands: int = len(OCTAVE_BANDS_HZ)) -> np.ndarray:
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    log_f = np.log2(np.maximum(f, 1e-6))
    width = crossover_width(bands)
    steps = []
    for edge in np.log2(crossover_frequencies(bands)):
        x = np.clip((log_f - (edge - width / 2)) / width, 0.0, 1.0)
        steps.append(np.sin(0.5 * np.pi * x) ** 2)  # 0 below the edge, 1 above
    steps = np.array(steps)
    gains = np.empty((bands, len(f)))
    gains[0] = 1.0 - steps[0]
    gains[1:-1] = steps[:-1] - steps[1:]
    gains[-1] = steps[-1]
    gains.flags.writeable = False
    return gains


def fft_size(n: int, sample_rate: float) -> int:
    """FFT length for filtering n samples with zero padding on both sides."""
    return 1 << int(math.ceil(math.log2(n + 2 * int(_PAD_S * sample_rate) + 1)))


def combine_bands(
    band_signals: np.ndarray, sample_rate: float, quadrature: np.ndarray | None = None
) -> np.ndarray:
    """Filter each band signal with its band and sum: (bands, n) -> (n,). With
    ``quadrature`` (bands, n), signals to add a quarter turn ahead in phase (their
    spectra times j)."""
    band_signals = np.asarray(band_signals, dtype=float)
    n = band_signals.shape[1]
    n_fft = fft_size(n, sample_rate)
    spectrum = np.fft.rfft(band_signals, n_fft, axis=1)
    if quadrature is not None:
        spectrum = spectrum + 1j * np.fft.rfft(np.asarray(quadrature, dtype=float), n_fft, axis=1)
    gains = _band_gains(n_fft, float(sample_rate), len(band_signals))
    out = np.fft.irfft((spectrum * gains).sum(axis=0), n_fft)
    return out[:n]


def split_bands(
    signal: np.ndarray, sample_rate: float, bands: int = len(OCTAVE_BANDS_HZ)
) -> np.ndarray:
    """Split a signal into the 8 octave bands or 24 third-octave bands: (n,) ->
    (bands, n); the rows sum to the input. As the bands are zero-phase, what each
    spreads before the first sample or after the last is not in it (see
    :func:`acoustic_spaces.parameters.octave_bands`)."""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    n_fft = fft_size(n, sample_rate)
    spectrum = np.fft.rfft(signal, n_fft)
    gains = _band_gains(n_fft, float(sample_rate), int(bands))
    return np.fft.irfft(spectrum[None, :] * gains, n_fft, axis=1)[:, :n]


def band_energy_share(
    sample_rate: float, n_fft: int = 1 << 16, bands: int = len(OCTAVE_BANDS_HZ)
) -> np.ndarray:
    """Energy of a unit impulse inside each band: its share of the spectrum."""
    g = band_gains(n_fft, sample_rate, bands) ** 2
    weights = np.full(g.shape[1], 2.0)
    weights[0] = 1.0
    if n_fft % 2 == 0:
        weights[-1] = 1.0
    return (g * weights).sum(axis=1) / n_fft


def place_impulses(
    delays_s: np.ndarray,
    amplitudes: np.ndarray,
    length: int,
    sample_rate: float,
    taps: int = 32,
) -> np.ndarray:
    """Band signals (bands, length) with a fractional-delay impulse per path.

    ``amplitudes`` is (paths, bands). Delays between samples use a Hann-windowed sinc.
    """
    amplitudes = np.atleast_2d(np.asarray(amplitudes, dtype=float))
    out = np.zeros((amplitudes.shape[1], length))
    if len(amplitudes) == 0:
        return out
    pos = np.asarray(delays_s, dtype=float) * sample_rate
    base = np.floor(pos).astype(np.int64)
    frac = pos - base
    k = np.arange(-taps // 2 + 1, taps // 2 + 1)
    x = k[None, :] - frac[:, None]
    kernel = np.sinc(x) * np.where(np.abs(x) < taps / 2, np.cos(np.pi * x / taps) ** 2, 0.0)
    idx = base[:, None] + k[None, :]
    inside = (idx >= 0) & (idx < length)
    for b in range(out.shape[0]):
        values = amplitudes[:, b, None] * kernel
        np.add.at(out[b], idx[inside], values[inside])
    return out


def _slots(n: int, sample_rate: float, time_step: float) -> np.ndarray:
    """The time slot of ``time_step`` each of ``n`` samples falls in."""
    return np.floor(np.arange(n) / (sample_rate * time_step) + 1e-9).astype(np.int64)


def dirac_sequence(
    length: int,
    sample_rate: float,
    volume_m3: float | None,
    speed_of_sound: float,
    rng: np.random.Generator,
    time_step: float,
) -> np.ndarray:
    """Random-sign impulses whose density grows like reflections in a room: 4 pi c^3 t^2 / V.

    Every time slot of :func:`shape_noise` (``time_step`` long, a whole number of samples
    or not) gets at least one impulse, so no slot's energy is lost.
    """
    t = np.arange(length) / sample_rate
    if volume_m3 is None:
        density = np.full(length, float(sample_rate))
    else:
        density = 4.0 * np.pi * speed_of_sound**3 * t**2 / volume_m3
        density = np.clip(density, 1000.0, sample_rate)
    pulses = rng.random(length) < density / sample_rate
    signs = np.where(rng.random(length) < 0.5, -1.0, 1.0)
    seq = np.where(pulses, signs, 0.0)
    slot = _slots(length, sample_rate, time_step)
    for s in np.flatnonzero(np.bincount(slot, weights=np.abs(seq)) == 0):
        start, stop = np.searchsorted(slot, [s, s + 1])
        seq[rng.integers(start, stop)] = signs[start]
    return seq


def smooth_energy(
    energy: np.ndarray, time_step: float, start_s: float, window_s: float = 0.01
) -> np.ndarray:
    """Spread each slot's energy after ``start_s`` over a ``window_s`` window centred
    on it.

    The late energy of a ray tracer is a sum of discrete hits; a single strong ray late
    in the tail would become a click. Smoothing moves energy by at most half a window
    and keeps the total. A slot within half a window of ``start_s`` or of the end
    spreads over the part of its window from ``start_s`` to the end, so no energy moves
    ahead of ``start_s`` (into the early window of C50, when that is 50 ms after the
    direct sound) or past the end.
    """
    energy = np.asarray(energy, dtype=float)
    n = energy.shape[-1]
    half = int(round(window_s / time_step / 2.0))
    start = int(math.ceil(start_s / time_step))
    if half < 1 or start >= n:
        return energy.copy()
    out = energy.copy()
    out[..., start:] = 0.0
    # The slots whose whole window lies after start_s: a moving average.
    first, last = start + half, n - half
    if last > first:
        width = 2 * half + 1
        inner = np.zeros_like(energy)
        inner[..., first:last] = energy[..., first:last]
        padded = np.pad(inner, [(0, 0)] * (inner.ndim - 1) + [(half, half)])
        cumulative = np.cumsum(padded, axis=-1)
        cumulative = np.concatenate([np.zeros_like(cumulative[..., :1]), cumulative], axis=-1)
        out += (cumulative[..., width:] - cumulative[..., :-width]) / width
    # The others, near either end: over the part of their window inside.
    for k in (*range(start, min(first, n)), *range(max(last, first), n)):
        low, high = max(k - half, start), min(k + half, n - 1)
        out[..., low : high + 1] += energy[..., k : k + 1] / (high - low + 1)
    return out


def shape_noise(
    energy: np.ndarray,
    time_step: float,
    noise: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """Band signals (bands, n): ``noise`` scaled so each slot carries ``energy[band, slot]``."""
    slot = np.minimum(_slots(len(noise), sample_rate, time_step), energy.shape[1] - 1)
    noise_energy = np.bincount(slot, weights=noise**2, minlength=energy.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(noise_energy > 0, np.sqrt(energy / noise_energy), 0.0)
    return gain[:, slot] * noise[None, :]
