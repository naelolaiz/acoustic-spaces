"""Rendering a simulated response in a spatial format.

Early paths are placed like in the mono response and given the gains (or, binaurally,
the HRTF) of their arrival direction. The late part is the mono response's noise, the
same sample for sample: in each octave band, each of its impulses comes from one of the
recorded arrival directions, drawn in proportion to the energy arriving from there at
that time (with the same random number in every band, so an impulse keeps one direction
where the bands agree). The omnidirectional channel (W, or the power of a panned layout)
keeps the mono energy, and each band's directions match the simulation on average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.spatial.directions import fibonacci_sphere, to_head
from acoustic_spaces.spatial.hrtf import HRTF, spherical_head

if TYPE_CHECKING:
    from acoustic_spaces.engine.response import Response

#: Directions the late part comes from when a response recorded none.
ISOTROPIC_DIRECTIONS = 64
#: Length of the per-path filters of binaural early reflections (s).
_KERNEL_S = 0.3
#: Paths filtered at once.
_CHUNK = 64
#: Length of the windowed sinc for fractional delays, as in ``synthesis.place_impulses``.
_SINC_TAPS = 32


class Format(Protocol):
    channels: int
    label: str
    channel_names: list[str]


@dataclass(frozen=True)
class Mono:
    """The omnidirectional response, as ``Response.impulse_response`` gives it."""

    channels: int = 1
    label: str = "mono"

    @property
    def channel_names(self) -> list[str]:
        return ["M"]

    def gains(self, directions: np.ndarray) -> np.ndarray:
        return np.ones((len(np.atleast_2d(directions)), 1))


@dataclass
class Binaural:
    """Left and right ear, through ``hrtf`` (default: :func:`spherical_head`)."""

    hrtf: HRTF | None = None
    channels: int = 2
    label: str = "binaural"
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def channel_names(self) -> list[str]:
        return ["L", "R"]

    def at(self, sample_rate: float) -> HRTF:
        """The HRTF at ``sample_rate``."""
        if sample_rate not in self._cache:
            if self.hrtf is None:
                self._cache[sample_rate] = spherical_head(sample_rate)
            else:
                self._cache[sample_rate] = self.hrtf.resampled(sample_rate)
        return self._cache[sample_rate]


def _late_directions(response: Response) -> tuple[np.ndarray, np.ndarray, float]:
    """Directions (room coordinates) and each one's share of the late energy per band
    and slot (directions, bands, slots), smoothed over three slots."""
    energy = response.late_directional
    if len(response.late_directions) == 0 or energy.size == 0 or not np.any(energy > 0):
        directions = fibonacci_sphere(ISOTROPIC_DIRECTIONS)
        return directions, np.full((len(directions), response.bands, 1), 1 / len(directions)), 1.0
    padded = np.pad(energy, ((0, 0), (0, 0), (1, 1)))
    smooth = padded[..., :-2] + padded[..., 1:-1] + padded[..., 2:]
    total = smooth.sum(axis=0, keepdims=True)
    share = np.where(total > 0, smooth / np.where(total > 0, total, 1.0), 1.0 / len(energy))
    return response.late_directions, share, response.direction_step


def _late_impulses(response: Response, n: int, sample_rate: float, seed: int | None):
    """The mono late part as band signals (bands, n), the samples holding its impulses,
    and the direction each impulse comes from in each band (impulses, bands)."""
    rng = np.random.default_rng(response.seed if seed is None else seed)
    noise = synthesis.dirac_sequence(
        n, sample_rate, response.volume_m3, response.speed_of_sound, rng, response.time_step
    )
    from acoustic_spaces.engine.response import SMOOTH_AFTER_S

    late = synthesis.smooth_energy(
        response.late, response.time_step, response.direct_delay_s + SMOOTH_AFTER_S
    )
    signals = synthesis.shape_noise(
        synthesis.ENERGY_TO_IR * late, response.time_step, noise, sample_rate
    )
    directions, share, step = _late_directions(response)
    at = np.flatnonzero(noise)
    slot = np.minimum((at / sample_rate / step).astype(np.int64), share.shape[2] - 1)
    # Inverse transform sampling for all slots at once: shifting each slot's cumulative
    # shares by the slot number keeps the whole table sorted.
    count, bands, slots = share.shape
    cdf = np.cumsum(share, axis=0) / np.maximum(share.sum(axis=0), 1e-300)
    shifted = (cdf + np.arange(slots)).transpose(1, 2, 0).reshape(bands, -1)
    # One random number per impulse for all bands: where the bands' shares agree, the
    # impulse comes from one direction at all frequencies, as a reflection would.
    u = rng.random(len(at))
    chosen = np.empty((len(at), bands), dtype=np.int64)
    for b in range(bands):
        found = np.searchsorted(shifted[b], u + slot, side="right")
        chosen[:, b] = found - slot * count
    return signals, at, np.clip(chosen, 0, count - 1), directions


def _render_gains(response: Response, fmt, n: int, sample_rate: float, seed):
    """``response`` in ``fmt`` (n, channels), and its early paths alone when the wave
    solver's bands are to take their place (else None)."""
    early = response.early
    amplitudes = np.sqrt(synthesis.ENERGY_TO_IR * early.energy)
    early_gains = fmt.gains(to_head(early.direction, response.orientation))
    has_late = bool(np.any(response.late > 0))
    if has_late:
        signals, at, chosen, directions = _late_impulses(response, n, sample_rate, seed)
        late_gains = fmt.gains(to_head(directions, response.orientation))
    out = np.empty((n, fmt.channels))
    alone = np.empty((n, fmt.channels)) if response.wave is not None else None
    for c in range(fmt.channels):
        bands = synthesis.place_impulses(
            early.delay_s, amplitudes * early_gains[:, c : c + 1], n, sample_rate
        )
        if alone is not None:
            alone[:, c] = synthesis.combine_bands(bands, sample_rate)
        if has_late:
            bands[:, at] += signals[:, at] * late_gains[chosen, c].T
        out[:, c] = synthesis.combine_bands(bands, sample_rate)
    return out, alone


def _render_binaural(response: Response, fmt: Binaural, n: int, sample_rate, seed):
    """As :func:`_render_gains`, at the two ears."""
    hrtf = fmt.at(sample_rate)
    out = np.zeros((2, n + hrtf.taps))

    # Early paths: band shaping, fractional delay and HRTF in one short filter each.
    early = response.early
    if len(early):
        n_k = 1 << int(math.ceil(math.log2(_KERNEL_S * sample_rate)))
        half = n_k // 2
        gains = synthesis.band_gains(n_k, sample_rate)
        amplitudes = np.sqrt(synthesis.ENERGY_TO_IR * early.energy)
        heads = to_head(early.direction, response.orientation)
        position = early.delay_s * sample_rate
        base = np.floor(position).astype(np.int64)
        # The fractional delay of the mono response: a Hann-windowed sinc.
        taps = np.arange(-_SINC_TAPS // 2 + 1, _SINC_TAPS // 2 + 1)
        for start in range(0, len(early), _CHUNK):
            part = slice(start, start + _CHUNK)
            x = taps[None, :] - (position[part] - base[part])[:, None]
            sinc = np.zeros((len(x), n_k))
            sinc[:, taps % n_k] = np.sinc(x) * np.cos(np.pi * x / _SINC_TAPS) ** 2
            spectrum = (amplitudes[part] @ gains) * np.fft.rfft(sinc, axis=1)
            ears = np.fft.rfft(hrtf.interpolate(heads[part]), n_k, axis=-1)
            kernels = np.roll(np.fft.irfft(spectrum[:, None, :] * ears, n_k, axis=-1), half, -1)
            for k, offset in enumerate(base[part] - half):
                lo, hi = max(offset, 0), min(offset + n_k, out.shape[1])
                if lo < hi:
                    out[:, lo:hi] += kernels[k, :, lo - offset : hi - offset]
    alone = out[:, :n].T.copy() if response.wave is not None else None

    # Late part: each impulse through the HRIR of its direction, then the band filters.
    if np.any(response.late > 0):
        from acoustic_spaces import _native

        signals, at, chosen, directions = _late_impulses(response, n, sample_rate, seed)
        # One interpolated response per direction of the late energy.
        kernels = np.ascontiguousarray(hrtf.interpolate(to_head(directions, response.orientation)))
        ears = np.empty((signals.shape[0], 2, out.shape[1]))
        for b in range(signals.shape[0]):
            ears[b] = _native.scatter_kernels(
                out.shape[1], at, np.ascontiguousarray(chosen[:, b]), signals[b, at], kernels
            )
        for ear in range(2):
            out[ear] += synthesis.combine_bands(ears[:, ear], sample_rate)
    return out[:, :n].T, alone


def _icosahedron() -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    points = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            points += [(0.0, a, b), (a, b, 0.0), (b, 0.0, a)]
    points = np.array(points)
    return points / np.linalg.norm(points, axis=1, keepdims=True)


#: Directions of the plane waves the wave solver's first-order field is split into.
WAVE_DIRECTIONS = _icosahedron()


def _with_wave(
    response: Response, fmt, out: np.ndarray, early: np.ndarray, sample_rate: float
) -> np.ndarray:
    """``out`` with its lowest bands replaced by the wave solver's, in ``fmt``: those of
    its ``early`` paths in pressure, those of the rest, the noise, in energy (as in the
    mono response)."""
    from acoustic_spaces.spatial.ambisonics import Ambisonics, real_spherical_harmonics

    wave = response.wave
    n = len(out)
    out = wave.high_pass(early, sample_rate) + wave.high_pass(out - early, sample_rate, late=True)
    first = wave.field(sample_rate, n)  # W, X, Y, Z in room coordinates
    if isinstance(fmt, Ambisonics):
        # Exact to first order; the wave solver gives nothing higher.
        low = np.zeros((n, fmt.channels))
        low[:, 0] = first[:, 0]
        if fmt.order >= 1:
            velocity = first[:, 1:] @ response.orientation  # in head coordinates
            low[:, 1:4] = velocity @ real_spherical_harmonics(1, np.eye(3))[:, 1:4]
        return out + low
    # Otherwise as plane waves from the vertices of an icosahedron (first order exact).
    planes = (first[:, :1] + 3.0 * first[:, 1:] @ WAVE_DIRECTIONS.T) / len(WAVE_DIRECTIONS)
    head = to_head(WAVE_DIRECTIONS, response.orientation)
    if isinstance(fmt, Binaural):
        hrtf = fmt.at(sample_rate)
        size = 1 << int(math.ceil(math.log2(n + hrtf.taps)))
        ears = np.fft.rfft(hrtf.interpolate(head), size, axis=-1)  # (planes, 2, bins)
        spectrum = np.einsum("bk,keb->be", np.fft.rfft(planes, size, axis=0), ears)
        return out + np.fft.irfft(spectrum, size, axis=0)[:n]
    return out + planes @ fmt.gains(head)


def from_output(output) -> Format:
    """The format an ``outputs`` entry of a scene file (``scene.Output``) describes."""
    from acoustic_spaces.spatial.ambisonics import Ambisonics
    from acoustic_spaces.spatial.hrtf import read_sofa
    from acoustic_spaces.spatial.loudspeakers import Loudspeakers, Speaker

    if output.format == "mono":
        return Mono()
    if output.format == "ambisonics":
        return Ambisonics(output.order)
    if output.format == "binaural":
        if output.hrtf is None:
            return Binaural()
        return Binaural(read_sofa(output.hrtf), label=f"binaural-{Path(output.hrtf).stem}")
    if output.layout is not None:
        return Loudspeakers.layout(output.layout)
    speakers = tuple(Speaker(s.name, s.azimuth, s.elevation, s.lfe) for s in output.speakers)
    return Loudspeakers(speakers)


def spatial_impulse_response(
    response: Response,
    fmt: Format,
    sample_rate: int = 48_000,
    seed: int | None = None,
) -> np.ndarray:
    """Impulse responses (samples, channels) of ``response`` in ``fmt``.

    Levels match the mono response: an arrival from straight ahead has the same height
    in W, in a speaker it is panned to, and (below the head's influence) at both ears.
    """
    n = int(math.ceil(response.length_s * sample_rate))
    if isinstance(fmt, Mono):
        return response.impulse_response(sample_rate, seed)[:, None]
    if isinstance(fmt, Binaural):
        out, early = _render_binaural(response, fmt, n, sample_rate, seed)
    else:
        out, early = _render_gains(response, fmt, n, sample_rate, seed)
    if response.wave is not None:
        out = _with_wave(response, fmt, out, early, sample_rate)
    return out


__all__ = [
    "ISOTROPIC_DIRECTIONS",
    "WAVE_DIRECTIONS",
    "Binaural",
    "Format",
    "Mono",
    "from_output",
    "spatial_impulse_response",
]
