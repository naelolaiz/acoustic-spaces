"""Early paths that add in pressure at low frequencies.

The echogram adds the early paths in energy, band by band, as if their phases were
random. At low frequencies they are not: a band's filter rings longer than the early
paths lie apart, and their pressures add. The impulse response has always added them
that way (its paths are impulses that add sample by sample), so the parameters of the
63 and 125 Hz bands read from it differed from the echogram's by up to 2 dB.

In the lowest bands, up to ``simulation.early.coherent_max_band_hz``, every early path
therefore carries its pressure (``EarlyReflections.spectrum``) at the frequencies the
edge diffraction is evaluated at: the phase of each reflection factor on the way, and
for paths over an edge the whole complex UTD coefficient, whose terms then add to the
specular paths in pressure, as the theory has them, rather than through the cross
terms of the higher bands. Its level in each band stays the energy that band gives it.
A surface reflects like a locally reacting one whose admittance has the material's
absorption, fitted as for the wave solver
(:func:`~acoustic_spaces.engine.impedance.fit_admittance`): R = (cos theta - Y) /
(cos theta + Y) at the angle of incidence theta. A hard surface reflects in phase at
every angle; an absorbing one turns the phase of sound that grazes it, by half a turn at
grazing incidence. Above those bands each path keeps the phase it has at their top
(:func:`held_phase`).

The impulse response builds those bands' early part from the pressures, and the
echogram reads each band's early energy from the same signal through the band filters
of the analysis of measured responses
(:func:`~acoustic_spaces.parameters.band_filter_gains`), as for the wave solver's bands:
an impulse response and the echogram it came from now give the same parameters in those
bands. The late part still adds in energy, and so do the uncorrelated points of a line
or surface source.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.materials.bands import (
    BANDS_PER_OCTAVE,
    OCTAVE_BANDS_HZ,
    band_centres,
    resolution_of,
)

if TYPE_CHECKING:
    from acoustic_spaces.engine.early import EarlyReflections
    from acoustic_spaces.engine.model import TraceModel

#: Paths whose spectra are built at once.
_CHUNK = 256
#: How much longer than the last early path the echogram's early part is taken (s),
#: for the band filters' ringing.
_TAIL_S = 0.3


def band_count(max_band_hz: int | None, bands: int, wave_max_band_hz: int | None = None) -> int:
    """How many of the lowest of ``bands`` (8 octave or 24 third-octave bands) add in
    pressure: those of the octave bands up to ``max_band_hz`` (None: none), and none
    when the wave solver runs up to that band or higher (``wave_max_band_hz``), as its
    bands take their place."""
    if max_band_hz is None:
        return 0
    if wave_max_band_hz is not None and wave_max_band_hz >= max_band_hz:
        return 0
    octaves = OCTAVE_BANDS_HZ.index(int(max_band_hz)) + 1
    return octaves * BANDS_PER_OCTAVE[resolution_of(bands)]


def sample_frequencies(bands: int, count: int) -> np.ndarray:
    """(count, M) the frequencies (Hz) a spectrum holds in each of the lowest ``count``
    of ``bands`` bands: those the edge diffraction is evaluated at."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    return band_frequencies(bands=resolution_of(bands))[0][:count]


def reflection_phases(
    model: TraceModel, faces: np.ndarray, cosines: np.ndarray, count: int
) -> np.ndarray:
    """(n, count, M) e^(j arg R) of the reflections off ``faces`` (n,) of sound that
    meets them at incidence cosines ``cosines`` (n,), at the frequencies of
    :func:`sample_frequencies`: R is that of a locally reacting surface whose
    admittance gives the face's absorption in the lowest ``count`` bands."""
    from acoustic_spaces.engine.impedance import fit_admittance

    faces = np.asarray(faces, dtype=np.int64)
    total = model.absorption.shape[1]
    f = sample_frequencies(total, count)
    out = np.ones((len(faces), *f.shape), dtype=complex)
    if len(faces) == 0 or count == 0:
        return out
    rows, inverse = np.unique(model.absorption[faces, :count], axis=0, return_inverse=True)
    admittance = np.array([fit_admittance(row, total, None)(f) for row in rows])
    y = admittance[inverse.reshape(-1)]
    cos = np.clip(np.asarray(cosines, dtype=float), 0.0, 1.0)[:, None, None]
    total_ = cos + y
    hard = np.abs(total_) < 1e-12  # a hard surface met at grazing incidence
    r = (cos - y) / np.where(hard, 1.0, total_)
    magnitude = np.abs(r)
    out[...] = np.where(hard | (magnitude < 1e-12), 1.0, r / np.maximum(magnitude, 1e-300))
    return out


def shapes(spectrum: np.ndarray, bands: int, f: np.ndarray) -> np.ndarray:
    """(n, len(f)) the spectra (n, count, M) of :attr:`EarlyReflections.spectrum` at
    frequencies ``f`` (Hz): magnitude and unwrapped phase interpolated linearly in log
    frequency between the samples, and held beyond the first and the last."""
    n, count, per_band = spectrum.shape
    samples = np.log2(sample_frequencies(bands, count).reshape(-1))
    flat = spectrum.reshape(n, -1)
    if len(samples) == 1:
        return np.repeat(flat, len(f), axis=1)
    x = np.log2(np.clip(np.asarray(f, dtype=float), 2.0 ** samples[0], 2.0 ** samples[-1]))
    j = np.clip(np.searchsorted(samples, x, side="right") - 1, 0, len(samples) - 2)
    w = (x - samples[j]) / (samples[j + 1] - samples[j])
    magnitude = np.abs(flat)
    phase = np.unwrap(np.angle(flat), axis=1)
    level = magnitude[:, j] * (1.0 - w) + magnitude[:, j + 1] * w
    angle = phase[:, j] * (1.0 - w) + phase[:, j + 1] * w
    return level * np.exp(1j * angle)


def held_phase(paths: EarlyReflections) -> np.ndarray:
    """(n,) the phase factor of each path at the top of the coherent bands, which it
    keeps in the bands above them: there the paths add in energy, and a path whose
    phase stepped back to zero at the crossover would cancel itself in it, as two bands
    half a turn apart do (ones without coherent bands)."""
    if paths.coherent_bands == 0 or len(paths) == 0:
        return np.ones(len(paths), dtype=complex)
    top = paths.spectrum[:, -1, -1]
    magnitude = np.abs(top)
    return np.where(magnitude > 1e-12, top / np.maximum(magnitude, 1e-300), 1.0)


def amplitude_spectra(
    paths: EarlyReflections, f: np.ndarray, gains: np.ndarray, only_coherent: bool = False
) -> np.ndarray:
    """(n, len(f)) each path's pressure at frequencies ``f`` relative to its delay, as
    the impulse response has it: its amplitude in each band through the synthesis's
    band ``gains`` (bands, len(f)) at those frequencies, in the coherent bands times its
    spectrum's shape and above them times its :func:`held_phase`; with
    ``only_coherent``, the coherent bands alone."""
    bands = paths.energy.shape[1]
    count = paths.coherent_bands
    amplitude = np.sqrt(synthesis.ENERGY_TO_IR * paths.energy)
    low = amplitude[:, :count] @ gains[:count]
    if count:
        low = low * shapes(paths.spectrum, bands, f)
    if only_coherent:
        return low
    return low + (amplitude[:, count:] @ gains[count:]) * held_phase(paths)[:, None]


def band_impulses(
    paths: EarlyReflections, n: int, sample_rate: float, gains: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """The band signals (bands, n) of the early paths above the coherent bands, each
    path times ``gains`` (paths,) (default ones), as :func:`synthesis.place_impulses`
    has them, and those of the quarter turn of their :func:`held_phase` for
    :func:`synthesis.combine_bands` (None when no path has one)."""
    amplitudes = np.sqrt(synthesis.ENERGY_TO_IR * paths.energy)
    amplitudes[:, : paths.coherent_bands] = 0.0  # those come from their pressure
    if gains is not None:
        amplitudes = amplitudes * np.asarray(gains, dtype=float)[:, None]
    held = held_phase(paths)
    if np.all(np.abs(held.imag) < 1e-12):
        held = held.real
        return synthesis.place_impulses(
            paths.delay_s, amplitudes * held[:, None], n, sample_rate
        ), None
    signals = synthesis.place_impulses(
        paths.delay_s, amplitudes * held.real[:, None], n, sample_rate
    )
    quadrature = synthesis.place_impulses(
        paths.delay_s, amplitudes * held.imag[:, None], n, sample_rate
    )
    return signals, quadrature


def signal(
    paths: EarlyReflections, n: int, sample_rate: float, weights: np.ndarray | None = None
) -> np.ndarray:
    """(n, K) the coherent bands' part of the early paths' impulse response, at
    ``sample_rate``, each path weighted by ``weights`` (paths, K) (default: one column
    of ones). Zero without coherent bands."""
    k = 1 if weights is None else np.asarray(weights).shape[1]
    count = paths.coherent_bands
    if count == 0 or len(paths) == 0:
        return np.zeros((n, k))
    bands = paths.energy.shape[1]
    n_fft = synthesis.fft_size(n, sample_rate)
    gains = synthesis.band_gains(n_fft, sample_rate, bands)
    support = np.flatnonzero(gains[:count].sum(axis=0) > 0.0)
    top = int(support.max()) + 1
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)[:top]
    weights = np.ones((len(paths), 1)) if weights is None else np.asarray(weights, dtype=float)
    out = np.zeros((n_fft // 2 + 1, k), dtype=complex)
    for start in range(0, len(paths), _CHUNK):
        part = slice(start, start + _CHUNK)
        spectra = amplitude_spectra(paths[part], f, gains[:, :top], only_coherent=True)
        spectra *= np.exp(-2j * np.pi * np.outer(paths.delay_s[part], f))
        for column in range(k):  # one at a time: a mono response and W add alike
            out[:top, column] += spectra.T @ weights[part, column]
    return np.fft.irfft(out, n_fft, axis=0)[:n]


def total_energy(paths: EarlyReflections) -> np.ndarray:
    """(bands,) the paths' energy together in each band: their sum, and in the coherent
    bands that of their pressures added at each of the band's frequencies, averaged over
    them; each frequency stands for its slice of the band, over which the interference
    of two paths that arrive far apart averages out."""
    from acoustic_spaces.engine.diffraction import band_frequencies

    out = paths.energy.sum(axis=0)
    count = paths.coherent_bands
    if count == 0 or len(paths) == 0:
        return out
    bands = paths.energy.shape[1]
    f, width = band_frequencies(bands=resolution_of(bands))
    f, width = f[:count], width[:count]
    pressure = np.sqrt(paths.energy[:, :count, None]) * paths.spectrum
    pressure = pressure * np.exp(-2j * np.pi * paths.delay_s[:, None, None] * f[None])
    apart = paths.delay_s[:, None] - paths.delay_s[None, :]
    for b in range(count):
        products = np.real(pressure[:, None, b] * np.conj(pressure[None, :, b])).mean(axis=2)
        out[b] = float(np.sum(products * np.sinc(apart * width[b])))
    return out


class EarlyPressure:
    """The early part of an impulse response at a low sample rate, as the echogram
    reads the coherent bands from it: as spectra from 0.2 s before emission, the
    pressure and the three components of the paths' arrival directions times it, or
    with ``weights`` (paths, K) the sums of the paths weighted by each column.

    Each band is scaled so that every path alone carries its energy in it: the analysis
    filters also let in a little of the neighbouring bands, whose levels differ, and the
    filters of the impulse response's bands overlap with them. What is left is the
    paths' interference."""

    def __init__(self, paths: EarlyReflections, weights: np.ndarray | None = None) -> None:
        bands = paths.energy.shape[1]
        self.bands = bands
        self.count = paths.coherent_bands
        top = band_centres(resolution_of(bands))[max(self.count, 1) - 1]
        # Rate high enough for the band filters of the coherent bands to have died out,
        # and a whole number of samples to a millisecond; each sample stands for the
        # interval it is the middle of.
        self.rate = 1000.0 * 2.0 ** math.ceil(math.log2(max(32.0 * top / 1000.0, 1.0)))
        self.lead = synthesis._PAD_S
        last = float(paths.delay_s.max()) if len(paths) else 0.0
        length = int(math.ceil((self.lead + last + _TAIL_S) * self.rate))
        self.n_fft = 1 << max(int(math.ceil(math.log2(length))), 4)
        f = np.fft.rfftfreq(self.n_fft, 1.0 / self.rate)
        gains = synthesis.band_gains(self.n_fft, self.rate, bands)
        if weights is None:
            weights = np.column_stack([np.ones(len(paths)), paths.direction])
        self.spectra = np.zeros((np.shape(weights)[1], len(f)), dtype=complex)
        self._alone = np.zeros(len(f))  # the paths' squared spectra, summed
        self._energy = paths.energy[:, : self.count].sum(axis=0)
        for start in range(0, len(paths), _CHUNK):
            part = slice(start, start + _CHUNK)
            spectra = amplitude_spectra(paths[part], f, gains)
            self._alone += np.sum(np.abs(spectra) ** 2, axis=0)
            shift = paths.delay_s[part] + self.lead - 0.5 / self.rate
            spectra *= np.exp(-2j * np.pi * np.outer(shift, f))
            self.spectra += weights[part].T @ spectra

    def band(
        self,
        b: int,
        time_step: float,
        slots: int,
        origin: float = 0.0,
        start: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """The spectra through band b's analysis filter (K, T) at the samples that fall
        into the ``slots`` of ``time_step``, slot 0 at ``origin`` (s from emission); the
        slot of each sample (T,); and the factor that makes a squared sample energy.

        The filters ring before an arrival as much as after it: what they spread to
        before ``start`` (s from emission; default ``origin``), the arrival of the
        direct sound, counts as arriving with it, as in the analysis of a measured
        response."""
        from acoustic_spaces.parameters import band_filter_gains

        gain = band_filter_gains(self.n_fft, self.rate, self.bands)[b]
        time = np.arange(self.n_fft) / self.rate - self.lead  # where each interval begins
        time = np.maximum(time, origin if start is None else max(start, origin)) - origin
        slot = (time / time_step + 1e-9).astype(np.int64)
        keep = slot < slots
        pressure = np.fft.irfft(gain * self.spectra, self.n_fft, axis=1)[:, keep]
        weights = np.full(len(gain), 2.0)
        weights[0] = weights[-1] = 1.0
        alone = float(np.sum(weights * gain**2 * self._alone)) / self.n_fft
        scale = float(self._energy[b]) / alone if alone > 0.0 else 0.0
        return pressure, slot[keep], scale

    def energy(
        self,
        time_step: float,
        slots: int,
        axis: np.ndarray | None = None,
        exponent: int = 2,
        origin: float = 0.0,
        start: float | None = None,
    ) -> np.ndarray:
        """(count, slots) energy per coherent band and time slot, as
        :meth:`~acoustic_spaces.engine.wave.WaveResponse.energy` reads the wave
        solver's bands, through each band's analysis filter (``origin`` and ``start``
        as for :meth:`band`). With ``axis``, that of a figure of eight facing it
        (``exponent`` as there)."""
        out = np.zeros((self.count, slots))
        for b in range(self.count):
            pressure, slot, scale = self.band(b, time_step, slots, origin, start)
            if axis is None:
                power = pressure[0] ** 2
            else:
                side = np.asarray(axis) @ pressure[1:]
                power = side**2 if exponent == 2 else np.abs(side * pressure[0])
            out[b] = np.bincount(slot, power * scale, minlength=slots)[:slots]
        return out


__all__ = [
    "EarlyPressure",
    "amplitude_spectra",
    "band_count",
    "band_impulses",
    "held_phase",
    "reflection_phases",
    "sample_frequencies",
    "shapes",
    "signal",
    "total_energy",
]
