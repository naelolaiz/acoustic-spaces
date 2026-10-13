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
        return 10.0 * np.log10(remaining / remaining[0])


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


def from_impulse_response(
    ir: np.ndarray,
    sample_rate: float,
    reference_energy: float | None = None,
) -> RoomParameters:
    """Parameters from a mono impulse response, split into octave bands.

    ``reference_energy`` is the energy of the broadband impulse response of the same
    source at 10 m in free field; for responses made by this package that is 0.01.
    """
    from acoustic_spaces.synthesis import band_energy_share, split_bands

    ir = np.asarray(ir, dtype=float)
    start = onset(ir)
    bands = split_bands(ir, sample_rate) ** 2
    reference = None
    if reference_energy is not None:
        reference = reference_energy * band_energy_share(sample_rate)
    dt = 1.0 / sample_rate
    per_band = [
        _band_parameters(row[start:], dt, 0.0, None if reference is None else reference[b])
        for b, row in enumerate(bands)
    ]
    return RoomParameters(
        **{f.name: np.array([p[f.name] for p in per_band]) for f in fields(RoomParameters)}
    )


__all__ = [
    "DECAY_RANGES",
    "PARAMETERS",
    "RoomParameters",
    "decay_curve",
    "from_energy",
    "from_impulse_response",
    "onset",
]
