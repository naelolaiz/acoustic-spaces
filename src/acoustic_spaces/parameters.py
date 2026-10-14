"""Room acoustic parameters of ISO 3382-1, from energy decays or impulse responses.

All parameters are per octave band and measured from the arrival of the direct sound:

- ``edt``, ``t20``, ``t30``: decay times (s) from a line fitted to the backward-integrated
  (Schroeder) curve between 0 and -10 dB, -5 and -25 dB, and -5 and -35 dB.
- ``c50``, ``c80``: clarity (dB), energy before 50 or 80 ms over the energy after.
- ``d50``: definition, energy before 50 ms over all of it (0 to 1).
- ``ts``: centre time (s), the energy-weighted mean arrival time.
- ``g``: strength (dB), total energy over that of the same source at 10 m in free field.

:func:`speech_transmission_index` gives the STI of IEC 60268-16 from the same energies.

A response that stops before it has decayed loses the energy of its missing tail and
reads too short; the tail is estimated from the last part of the decay and added back
(ISO 3382-1, 5.3.4). One that stops before it has decayed by 20 dB leaves too much to
that estimate, and all its parameters are NaN.
"""

from __future__ import annotations

import functools
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
#: at the end of the response, and spans at most ``_TAIL_SPAN_DB``. A response must
#: decay to ``_MIN_DECAY_DB`` for any parameter.
_TAIL_HEADROOM_DB = 10.0
_TAIL_SPAN_DB = 20.0
_MIN_DECAY_DB = -20.0

#: Ratio between adjacent octave mid-band frequencies, base 10 (IEC 61260-1).
OCTAVE_RATIO = 10.0**0.3
#: Order of the Butterworth octave filters for impulse responses.
FILTER_ORDER = 3
#: Room before an impulse response for the ringing of its band filters (s).
_FILTER_LEAD_S = 0.2
#: Samples at the end of an impulse response below this fraction of its peak are
#: digital silence (zero padding, a latency shift), not background noise.
_SILENCE = 1e-12

#: Noise floor search (Lundeby et al., 1995): first averaging interval (s), intervals
#: per 10 dB of decay afterwards, margin of the late-decay fit above the noise and its
#: span, how far past the crosspoint the noise is measured (dB), and iterations. A
#: "noise" more than ``_NO_NOISE_DB`` below the loudest interval is the band filters'
#: ringing in silence.
_LUNDEBY_INTERVAL_S = 0.01
_LUNDEBY_INTERVALS_PER_10DB = 5
_LUNDEBY_MARGIN_DB = 5.0
_LUNDEBY_SPAN_DB = 20.0
_LUNDEBY_NOISE_AFTER_DB = 10.0
_LUNDEBY_ITERATIONS = 5
_NO_NOISE_DB = 120.0


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
    # The closed form of a straight-line fit: maps call this for thousands of points.
    x, y = t[start : stop + 1], level[start : stop + 1]
    n = len(x)
    sx, sy = float(x.sum()), float(y.sum())
    slope = (n * float(x @ y) - sx * sy) / (n * float(x @ x) - sx * sx)
    intercept = (sy - slope * sx) / n
    return (slope, intercept) if slope < 0 else None


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
    undefined = dict(edt=nan, t20=nan, t30=nan, c50=nan, c80=nan, d50=nan, ts=nan, g=nan)
    if total <= 0:
        return undefined

    remaining = np.concatenate([np.cumsum(e[::-1])[::-1], [0.0]]) + tail
    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(np.maximum(remaining / total, 1e-300))
    out = {
        name: _decay_time(level, edges, top, bottom) for name, (top, bottom) in DECAY_RANGES.items()
    }
    # A response that stops before its decay has fallen to _MIN_DECAY_DB leaves too much
    # to the estimated tail: nothing can be read from it, as T20 and T30 already cannot.
    # The fall counts at the end of the curve, tail included, or along the early decay's
    # slope, as one strong ray late in a sparse tail can hold the end of the curve up.
    if level[-1] > _MIN_DECAY_DB and not 60.0 * edges[-1] / out["edt"] >= -_MIN_DECAY_DB:
        return undefined
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


#: Speech transmission index (IEC 60268-16:2020, annex A): its octave bands, the
#: modulation frequencies (Hz), the male weighting factors alpha and beta, and the
#: absolute speech reception threshold of each band (dB).
STI_BANDS_HZ: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000, 8000)
STI_MODULATION_HZ: tuple[float, ...] = (
    0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5
)  # fmt: skip
_STI_ALPHA = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
_STI_BETA = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
_STI_THRESHOLD_DB = np.array([46.0, 27.0, 12.0, 6.5, 7.5, 8.0, 12.0])


def modulation_transfer(energy: np.ndarray, time_step: float) -> np.ndarray:
    """Modulation transfer function (bands, modulation frequencies) of an energy
    response e(t) (bands, slots): the magnitude of the sum of e(t) exp(-2 pi j F t)
    over the sum of e(t), at each F of ``STI_MODULATION_HZ`` (Schroeder's method)."""
    energy = np.atleast_2d(np.asarray(energy, dtype=float))
    slots = energy.shape[1]
    # One kernel serves every response up to a power of two long.
    kernel = _modulation_kernel(1 << max(10, (slots - 1).bit_length()), float(time_step))
    kernel = kernel[:slots]
    total = energy.sum(axis=1, keepdims=True)
    return np.abs(energy @ kernel) / np.where(total > 0, total, np.inf)


@functools.lru_cache(maxsize=8)
def _modulation_kernel(slots: int, time_step: float) -> np.ndarray:
    t = (np.arange(slots) + 0.5) * time_step
    kernel = np.exp(-2j * np.pi * np.outer(t, STI_MODULATION_HZ))
    kernel.flags.writeable = False
    return kernel


def _masking_slope_db(level: np.ndarray) -> np.ndarray:
    """Level-dependent auditory masking of the next band up (IEC 60268-16, A.2.3)."""
    return np.select(
        [level < 63.0, level < 67.0, level < 100.0],
        [0.5 * level - 65.0, 1.8 * level - 146.9, 0.5 * level - 59.8],
        -10.0,
    )


def speech_transmission_index(
    energy: np.ndarray,
    time_step: float,
    *,
    signal_db: Sequence[float] | np.ndarray | None = None,
    noise_db: Sequence[float] | np.ndarray | float | None = None,
) -> float:
    """STI (0 to 1) of an energy response per octave band (bands, slots), from the
    modulation transfer function (IEC 60268-16:2020, indirect method, male weighting).

    ``energy`` has the bands of ``OCTAVE_BANDS_HZ`` (the 63 Hz band is not used) or of
    ``STI_BANDS_HZ``. ``signal_db`` is the speech level per band at the listener (dB
    SPL); with it, auditory masking and the reception threshold apply, and
    ``noise_db`` (dB SPL per band, or one value for all) lowers the modulation by the
    signal-to-noise ratio. Without either, the STI is that of the room alone.
    """
    energy = np.atleast_2d(np.asarray(energy, dtype=float))
    bands = len(STI_BANDS_HZ)

    def per_band(values, what: str) -> np.ndarray:
        v = np.broadcast_to(np.asarray(values, dtype=float), (np.size(values),)).ravel()
        if v.size == 1:
            return np.full(bands, v[0])
        if v.size == len(OCTAVE_BANDS_HZ):
            return v[1:]
        if v.size == bands:
            return v
        raise ValueError(f"{what}: give one value or one per octave band")

    if energy.shape[0] == len(OCTAVE_BANDS_HZ):
        energy = energy[1:]
    elif energy.shape[0] != bands:
        raise ValueError("energy must have the octave bands 63 Hz (or 125 Hz) to 8 kHz")
    m = modulation_transfer(energy, time_step)
    if noise_db is not None and signal_db is None:
        raise ValueError("noise_db needs signal_db, the speech level at the listener")
    if signal_db is not None:
        signal = 10.0 ** (per_band(signal_db, "signal_db") / 10.0)
        noise = 0.0 if noise_db is None else 10.0 ** (per_band(noise_db, "noise_db") / 10.0)
        m = m / (1.0 + noise / signal)[:, None]
        intensity = signal + noise
        level = 10.0 * np.log10(intensity)
        masking = np.zeros(bands)
        masking[1:] = intensity[:-1] * 10.0 ** (_masking_slope_db(level[:-1]) / 10.0)
        threshold = 10.0 ** (_STI_THRESHOLD_DB / 10.0)
        m = m * (intensity / (intensity + masking + threshold))[:, None]
    with np.errstate(divide="ignore"):
        snr = 10.0 * np.log10(m / np.maximum(1.0 - m, 0.0))
    transmission = (np.clip(np.nan_to_num(snr, nan=-15.0), -15.0, 15.0) + 15.0) / 30.0
    mti = transmission.mean(axis=1)
    sti = float(_STI_ALPHA @ mti - _STI_BETA @ np.sqrt(mti[:-1] * mti[1:]))
    return min(max(sti, 0.0), 1.0)


def onset(ir: np.ndarray, threshold_db: float = -20.0) -> int:
    """Sample where the response first rises to ``threshold_db`` below its peak (ISO 3382-1)."""
    power = np.asarray(ir, dtype=float) ** 2
    peak = power.max(initial=0.0)
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
    the loudest 10 ms down to the noise). A response that ends in silence (its last
    tenth more than 120 dB below the loudest 10 ms), or is still decaying in its last
    tenth, has no crosspoint (``len(energy)``), no noise and an infinite range; one with
    no decay above its noise has a range of 0.
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
    if noise_db < peak_db - _NO_NOISE_DB:
        return n, 0.0, math.inf
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
    """Filter ``ir`` into octave bands and, with ``noise``, find each band's noise floor.

    ``ir`` is (samples,) or (samples, 1), as :func:`~acoustic_spaces.audio.read_wav`
    gives a mono file.
    """
    ir = np.asarray(ir, dtype=float)
    if ir.ndim == 2 and ir.shape[1] == 1:
        ir = ir[:, 0]
    if ir.ndim != 1:
        raise ValueError("the impulse response must have one channel: (samples,) or (samples, 1)")
    start = onset(ir)
    # Zero-phase filters ring before an impulse as much as after it. Room is made for
    # that ringing, and everything before the onset counts as arriving with it.
    lead = int(round(_FILTER_LEAD_S * sample_rate))
    padded = np.concatenate([np.zeros(lead), ir])
    squared = octave_bands(padded, sample_rate, filters) ** 2
    energy = squared[:, lead + start :]
    if energy.shape[1]:
        energy[:, 0] += squared[:, : lead + start].sum(axis=1)
    # Digital silence at the end (zero padding, the shift that removes latency) is no
    # background noise: the noise floor is looked for before it, unless too little is
    # left for that, as after a lone impulse, whose filtered ringing then shows no noise.
    loud = np.flatnonzero(np.abs(ir) > _SILENCE * np.abs(ir).max(initial=0.0))
    end = loud[-1] + 1 - start if len(loud) else 0
    if end < 3 * _LUNDEBY_INTERVAL_S * sample_rate:
        end = energy.shape[1]
    found = []
    for row in energy:
        cross, level, range_db = (
            noise_crosspoint(row[:end], sample_rate) if noise else (end, 0.0, math.inf)
        )
        found.append((len(row) if cross >= end else cross, level, range_db))
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

    ``ir`` is (samples,) or (samples, 1); an empty or silent one has every parameter
    NaN. ``reference_energy`` is the energy of the same source at 10 m in free field: per
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
    "STI_BANDS_HZ",
    "STI_MODULATION_HZ",
    "RoomParameters",
    "decay_curve",
    "from_energy",
    "modulation_transfer",
    "speech_transmission_index",
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
