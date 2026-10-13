"""Room acoustic parameters of ISO 3382-1, from energy decays or impulse responses.

All parameters are per octave band and measured from the arrival of the direct sound:

- ``edt``, ``t20``, ``t30``: decay times (s) from a line fitted to the backward-integrated
  (Schroeder) curve between 0 and -10 dB, -5 and -25 dB, and -5 and -35 dB.
- ``c50``, ``c80``: clarity (dB), energy before 50 or 80 ms over the energy after.
- ``d50``: definition, energy before 50 ms over all of it (0 to 1).
- ``ts``: centre time (s), the energy-weighted mean arrival time.
- ``g``: strength (dB), total energy over that of the same source at 10 m in free field.

A response that stops before it has decayed loses the energy of its missing tail and
reads too short; the tail is estimated from the last part of the decay and added back
(ISO 3382-1, 5.3.4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

#: Fit ranges in dB for EDT, T20 and T30.
DECAY_RANGES = {"edt": (0.0, -10.0), "t20": (-5.0, -25.0), "t30": (-5.0, -35.0)}
#: Parameter names, units and short descriptions, in display order.
PARAMETERS = {
    "edt": ("s", "early decay time"),
    "t20": ("s", "reverberation time from the -5 to -25 dB decay"),
    "t30": ("s", "reverberation time from the -5 to -35 dB decay"),
    "c50": ("dB", "clarity, 50 ms"),
    "c80": ("dB", "clarity, 80 ms"),
    "d50": ("", "definition"),
    "ts": ("s", "centre time"),
    "g": ("dB", "strength"),
}
#: The fit for the missing tail stays this far (dB) above the level the decay reaches
#: at the end of the response, and spans at most ``_TAIL_SPAN_DB``.
_TAIL_HEADROOM_DB = 10.0
_TAIL_SPAN_DB = 20.0

#: Ratio between adjacent octave mid-band frequencies, base 10 (IEC 61260-1).
OCTAVE_RATIO = 10.0**0.3
#: Order of the Butterworth octave filters for impulse responses.
FILTER_ORDER = 3
#: Room before an impulse response for the ringing of its band filters (s).
_FILTER_LEAD_S = 0.2

#: Noise floor search (Lundeby et al., 1995): first averaging interval (s), intervals
#: per 10 dB of decay afterwards, margin of the late-decay fit above the noise and its
#: span, how far past the crosspoint the noise is measured (dB), and iterations.
_LUNDEBY_INTERVAL_S = 0.01
_LUNDEBY_INTERVALS_PER_10DB = 5
_LUNDEBY_MARGIN_DB = 5.0
_LUNDEBY_SPAN_DB = 20.0
_LUNDEBY_NOISE_AFTER_DB = 10.0
_LUNDEBY_ITERATIONS = 5


@dataclass
class RoomParameters:
    """ISO 3382-1 parameters, one value per octave band (NaN where undefined)."""

    edt: np.ndarray
    t20: np.ndarray
    t30: np.ndarray
    c50: np.ndarray
    c80: np.ndarray
    d50: np.ndarray
    ts: np.ndarray
    g: np.ndarray

    def as_dict(self) -> dict[str, list[float | None]]:
        """Plain lists, with None for undefined values (JSON friendly)."""
        return {
            f.name: [None if not np.isfinite(v) else float(v) for v in getattr(self, f.name)]
            for f in fields(self)
        }

    def table(self) -> str:
        """A text table: one row per parameter, one column per band."""
        bands = "".join(f"{_band_label(f):>8}" for f in OCTAVE_BANDS_HZ)
        lines = [f"{'':<10}{bands}"]
        for name, (unit, _) in PARAMETERS.items():
            label = f"{name.upper() if name != 'ts' else 'Ts'}" + (f" ({unit})" if unit else "")
            fmt = "8.3f" if unit == "s" or not unit else "8.1f"
            cells = "".join(
                f"{v:{fmt}}" if np.isfinite(v) else f"{'-':>8}" for v in getattr(self, name)
            )
            lines.append(f"{label:<10}{cells}")
        return "\n".join(lines)


def _band_label(f: int) -> str:
    return f"{f} Hz" if f < 1000 else f"{f // 1000} kHz"


def _line(level: np.ndarray, t: np.ndarray, top: float, bottom: float):
    """Least-squares line (slope dB/s, intercept dB) through ``level`` between two
    levels, or None when the curve does not fall that far or does not fall at all."""
    if not np.any(level <= bottom):
        return None
    start = int(np.argmax(level <= top))
    stop = int(np.argmax(level <= bottom))
    if stop - start < 2:
        return None
    slope, intercept = np.polyfit(t[start : stop + 1], level[start : stop + 1], 1)
    return (float(slope), float(intercept)) if slope < 0 else None


def _tail(e: np.ndarray, dt: float) -> tuple[float, float]:
    """Energy after the end of ``e`` (per-slot energies) and its decay rate (1/s).

    A line is fitted to the backward-integrated curve of the data alone, low on the
    decay but ``_TAIL_HEADROOM_DB`` above where it would have reached by the end, where
    the missing tail barely bends it; extended to the end, that line gives the energy
    still to come. Integrated curves are smooth even when the energy per slot is
    sparse, so the estimate stays stable with few rays.
    """
    remaining = np.cumsum(e[::-1])[::-1]
    if len(remaining) == 0 or remaining[0] <= 0:
        return 0.0, math.inf
    total = float(remaining[0])
    t = np.arange(len(e)) * dt
    end = len(e) * dt
    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(np.maximum(remaining / total, 1e-300))
    line = _line(level, t, -5.0, -15.0)
    for _ in range(2):
        if line is None:
            return 0.0, math.inf
        bottom = min(max(line[0] * end + line[1] + _TAIL_HEADROOM_DB, -60.0), -15.0)
        line = _line(level, t, min(bottom + _TAIL_SPAN_DB, -5.0), bottom)
    if line is None:
        return 0.0, math.inf
    slope, intercept = line
    tail = total * 10.0 ** ((slope * end + intercept) / 10.0)
    return float(tail), float(-slope * math.log(10.0) / 10.0)


def _decay_time(level: np.ndarray, t: np.ndarray, top: float, bottom: float) -> float:
    """-60 dB over the slope of a line fitted to ``level`` between two levels."""
    line = _line(level, t, top, bottom)
    return math.nan if line is None else -60.0 / line[0]


def decay_curve(energy: np.ndarray, time_step: float) -> np.ndarray:
    """Backward-integrated (Schroeder) decay in dB at the start of each slot.

    ``energy`` is one band's energy per slot, starting at the direct sound; the curve
    starts at 0 dB and includes the estimated energy after the end of the response.
    """
    e = np.asarray(energy, dtype=float)
    tail, _ = _tail(e, time_step)
    remaining = np.cumsum(e[::-1])[::-1] + tail
    if len(remaining) == 0 or remaining[0] <= 0:
        return np.full(len(e), -np.inf)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(remaining, 0.0) / remaining[0])


def _clarity(early: float, late: float) -> float:
    if early <= 0:
        return -math.inf
    return 10.0 * math.log10(early / late) if late > 0 else math.inf


def _band_parameters(e: np.ndarray, dt: float, offset: float, reference: float | None):
    """Parameters of one band; ``e`` starts at the slot holding the onset, which lies
    ``offset`` seconds into it."""
    edges = np.arange(len(e) + 1) * dt - offset  # slot edges, time after the onset
    tail, rate = _tail(e, dt)
    total = float(e.sum()) + tail
    nan = math.nan
    if total <= 0:
        return dict(edt=nan, t20=nan, t30=nan, c50=nan, c80=nan, d50=nan, ts=nan, g=nan)

    remaining = np.concatenate([np.cumsum(e[::-1])[::-1], [0.0]]) + tail
    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(np.maximum(remaining / total, 1e-300))
    out = {
        name: _decay_time(level, edges, top, bottom) for name, (top, bottom) in DECAY_RANGES.items()
    }
    cumulative = total - remaining
    early50 = float(np.interp(0.05, edges, cumulative))
    early80 = float(np.interp(0.08, edges, cumulative))
    centre = np.clip(0.5 * (edges[:-1] + edges[1:]), 0.0, None)
    out["c50"] = _clarity(early50, total - early50)
    out["c80"] = _clarity(early80, total - early80)
    out["d50"] = early50 / total
    # An exponential tail from the end T at rate k has its centre of gravity at T + 1 / k.
    tail_moment = tail * (edges[-1] + 1.0 / rate) if tail > 0 else 0.0
    out["ts"] = float((centre * e).sum() + tail_moment) / total
    out["g"] = 10.0 * math.log10(total / reference) if reference else nan
    return out


def from_energy(
    energy: np.ndarray,
    time_step: float,
    onset_s: float,
    reference_energy: float | None = None,
) -> RoomParameters:
    """Parameters from energy per band and time slot (bands, slots), slot 0 at emission.

    ``onset_s`` is the arrival time of the direct sound. ``reference_energy`` is the
    energy of the same source at 10 m in free field; without it ``g`` is NaN.
    """
    energy = np.atleast_2d(np.asarray(energy, dtype=float))
    first = min(int(onset_s / time_step), energy.shape[1] - 1)
    offset = onset_s - first * time_step
    per_band = [
        _band_parameters(row[first:], time_step, offset, reference_energy) for row in energy
    ]
    return RoomParameters(
        **{f.name: np.array([p[f.name] for p in per_band]) for f in fields(RoomParameters)}
    )


def onset(ir: np.ndarray, threshold_db: float = -20.0) -> int:
    """Sample where the response first rises to ``threshold_db`` below its peak (ISO 3382-1)."""
    power = np.asarray(ir, dtype=float) ** 2
    peak = power.max()
    if peak <= 0:
        return 0
    return int(np.argmax(power >= peak * 10.0 ** (threshold_db / 10.0)))


def octave_midbands() -> np.ndarray:
    """Exact mid-band frequencies (Hz) of ``OCTAVE_BANDS_HZ``, base 10 (IEC 61260-1)."""
    nominal = np.array(OCTAVE_BANDS_HZ, dtype=float)
    k = np.round(np.log(nominal / 1000.0) / math.log(OCTAVE_RATIO))
    return 1000.0 * OCTAVE_RATIO**k


def octave_filter_gains(n_fft: int, sample_rate: float, order: int = FILTER_ORDER) -> np.ndarray:
    """Magnitudes (bands, n_fft // 2 + 1) of Butterworth octave band-pass filters.

    A band-pass of ``order`` (that of its low-pass prototype; 3 meets IEC 61260-1
    class 1) has 1 at mid-band, -3 dB at the band edges fm G^(+-1/2) and about -20 dB
    at the neighbouring mid-bands.
    """
    f = np.maximum(np.fft.rfftfreq(n_fft, 1.0 / sample_rate), 1e-9)
    fm = octave_midbands()[:, None]
    width = math.sqrt(OCTAVE_RATIO) - 1.0 / math.sqrt(OCTAVE_RATIO)
    return 1.0 / np.sqrt(1.0 + ((f / fm - fm / f) / width) ** (2 * order))


def octave_bands(signal: np.ndarray, sample_rate: float, filters: str = "iec") -> np.ndarray:
    """Split a signal into octave bands: (n,) -> (bands, n).

    ``filters="iec"`` applies the magnitudes of :func:`octave_filter_gains` with zero
    phase, so no band is delayed against another; ``"crossover"`` uses the crossovers of
    :mod:`acoustic_spaces.synthesis`, whose bands sum back to the input.
    """
    from acoustic_spaces import synthesis

    if filters == "crossover":
        return synthesis.split_bands(signal, sample_rate)
    if filters != "iec":
        raise ValueError(f"unknown filters {filters!r}; use 'iec' or 'crossover'")
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    n_fft = synthesis.fft_size(n, sample_rate)
    spectrum = np.fft.rfft(signal, n_fft)
    gains = octave_filter_gains(n_fft, sample_rate)
    return np.fft.irfft(spectrum[None, :] * gains, n_fft, axis=1)[:, :n]


def band_energy_share(sample_rate: float, filters: str = "iec") -> np.ndarray:
    """Energy of a unit impulse inside each band, for converting broadband references."""
    from acoustic_spaces import synthesis

    if filters == "crossover":
        return synthesis.band_energy_share(sample_rate)
    n_fft = 1 << 16
    g = octave_filter_gains(n_fft, sample_rate) ** 2
    weights = np.full(g.shape[1], 2.0)
    weights[0] = weights[-1] = 1.0
    return (g * weights).sum(axis=1) / n_fft


def _levels(e: np.ndarray, width: int, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Mid times (s) and levels (dB) of ``e`` averaged over blocks of ``width`` samples."""
    blocks = len(e) // width
    mean = e[: blocks * width].reshape(blocks, width).mean(axis=1)
    t = (np.arange(blocks) + 0.5) * width / sample_rate
    return t, 10.0 * np.log10(np.maximum(mean, 1e-300))


def noise_crosspoint(energy: np.ndarray, sample_rate: float) -> tuple[int, float, float]:
    """Where a squared response sinks into its background noise (Lundeby et al., 1995).

    ``energy`` is one band's squared response from the direct sound on. Returns the
    crosspoint (samples), the mean noise power per sample and the decay range (dB, from
    the loudest 10 ms down to the noise). A response that ends in silence, or is still
    decaying in its last tenth, has no crosspoint (``len(energy)``), no noise and an
    infinite range; one with no decay above its noise has a range of 0.
    """
    e = np.asarray(energy, dtype=float)
    n = len(e)
    last = e[int(0.9 * n) :]
    if len(last) == 0 or last.mean() <= 0:
        return n, 0.0, math.inf
    noise = float(last.mean())
    noise_db = 10.0 * math.log10(noise)
    width = max(1, int(round(_LUNDEBY_INTERVAL_S * sample_rate)))
    t, level = _levels(e, width, sample_rate)
    peak_db = float(level.max(initial=-np.inf))
    line = None
    if len(level) >= 3:
        line = _line(level, t, float(level.max()), noise_db + 2.0 * _LUNDEBY_MARGIN_DB)
    if line is None:
        return n, 0.0, 0.0
    slope, intercept = line
    cross = (noise_db - intercept) / slope
    for _ in range(_LUNDEBY_ITERATIONS):
        width = max(1, int(round(-10.0 / slope / _LUNDEBY_INTERVALS_PER_10DB * sample_rate)))
        t, level = _levels(e, width, sample_rate)
        start = min(cross - _LUNDEBY_NOISE_AFTER_DB / slope, 0.9 * n / sample_rate)
        tail = e[int(start * sample_rate) :]
        if len(level) < 3 or tail.mean() <= 0:
            break
        noise_db = 10.0 * math.log10(float(tail.mean()))
        bottom = noise_db + _LUNDEBY_MARGIN_DB
        line = _line(level, t, min(bottom + _LUNDEBY_SPAN_DB, float(level.max())), bottom)
        if line is None:
            break
        slope, intercept = line
        previous, cross = cross, (noise_db - intercept) / slope
        if abs(cross - previous) < width / sample_rate:
            break
    stop = int(round(cross * sample_rate))
    if stop >= 0.9 * n:
        return n, 0.0, math.inf  # the "noise" is the end of the decay itself
    return max(1, stop), 10.0 ** (noise_db / 10.0), peak_db - noise_db


@dataclass
class BandDecay:
    """Squared octave-band responses of one impulse response, from its onset on."""

    energy: np.ndarray  # (bands, samples)
    sample_rate: float
    #: Arrival of the direct sound in the impulse response (s).
    onset_s: float
    #: Per band: sample where the response sinks into the background noise (the length
    #: when no noise floor shows), the noise power per sample, and the decay range (dB).
    crosspoint: np.ndarray
    noise: np.ndarray
    decay_range_db: np.ndarray

    def cleaned(self, band: int) -> np.ndarray:
        """One band's energy up to its crosspoint, with the noise taken off."""
        return self.energy[band, : self.crosspoint[band]] - self.noise[band]


def band_decay(
    ir: np.ndarray, sample_rate: float, *, filters: str = "iec", noise: bool = True
) -> BandDecay:
    """Filter ``ir`` into octave bands and, with ``noise``, find each band's noise floor."""
    ir = np.asarray(ir, dtype=float)
    start = onset(ir)
    # Zero-phase filters ring before an impulse as much as after it. Room is made for
    # that ringing, and everything before the onset counts as arriving with it.
    lead = int(round(_FILTER_LEAD_S * sample_rate))
    padded = np.concatenate([np.zeros(lead), ir])
    squared = octave_bands(padded, sample_rate, filters) ** 2
    energy = squared[:, lead + start :]
    energy[:, 0] += squared[:, : lead + start].sum(axis=1)
    found = [
        noise_crosspoint(row, sample_rate) if noise else (len(row), 0.0, math.inf) for row in energy
    ]
    return BandDecay(
        energy=energy,
        sample_rate=sample_rate,
        onset_s=start / sample_rate,
        crosspoint=np.array([c for c, _, _ in found], dtype=np.int64),
        noise=np.array([x for _, x, _ in found]),
        decay_range_db=np.array([r for _, _, r in found]),
    )


def from_impulse_response(
    ir: np.ndarray,
    sample_rate: float,
    reference_energy: float | Sequence[float] | None = None,
    *,
    filters: str = "iec",
    noise: bool = True,
) -> RoomParameters:
    """Parameters from a mono impulse response, split into octave bands.

    ``reference_energy`` is the energy of the same source at 10 m in free field: per
    band (as from :func:`free_field_reference`), or that of a broadband impulse, which
    for responses made by this package is 0.01. ``filters`` picks the band filters
    (see :func:`octave_bands`). With ``noise``, each band is cut where it sinks into the
    background noise, the noise is subtracted and the decay beyond the cut estimated
    (ISO 3382-1, 5.3.4); decay times whose range does not clear the noise by 10 dB
    (EDT 20 dB, T20 35 dB, T30 45 dB) are NaN.
    """
    decay = band_decay(ir, sample_rate, filters=filters, noise=noise)
    reference = None
    if reference_energy is not None:
        reference = np.asarray(reference_energy, dtype=float)
        if reference.ndim == 0:
            reference = reference * band_energy_share(sample_rate, filters)
    dt = 1.0 / sample_rate
    per_band = []
    for b in range(decay.energy.shape[0]):
        p = _band_parameters(decay.cleaned(b), dt, 0.0, None if reference is None else reference[b])
        for name, (_, bottom) in DECAY_RANGES.items():
            if decay.decay_range_db[b] < -bottom + 10.0:
                p[name] = math.nan
        per_band.append(p)
    return RoomParameters(
        **{f.name: np.array([p[f.name] for p in per_band]) for f in fields(RoomParameters)}
    )


def free_field_reference(
    ir: np.ndarray, sample_rate: float, distance_m: float, *, filters: str = "iec"
) -> np.ndarray:
    """Per-band reference energies for G from a free-field impulse response.

    ``ir`` is the source measured at ``distance_m`` with no reflections (or with them
    windowed out); the result is its band energies moved to 10 m (inverse square law).
    """
    energy = octave_bands(np.asarray(ir, dtype=float), sample_rate, filters) ** 2
    return energy.sum(axis=1) * (distance_m / 10.0) ** 2


__all__ = [
    "DECAY_RANGES",
    "PARAMETERS",
    "RoomParameters",
    "decay_curve",
    "from_energy",
    "BandDecay",
    "FILTER_ORDER",
    "OCTAVE_RATIO",
    "band_decay",
    "band_energy_share",
    "free_field_reference",
    "from_impulse_response",
    "noise_crosspoint",
    "octave_bands",
    "octave_filter_gains",
    "octave_midbands",
    "onset",
]
