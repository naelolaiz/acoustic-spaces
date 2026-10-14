"""Comparing a simulated response with a measured one, by the parameters of ISO 3382-1.

A model is checked against the room it stands for by comparing its parameters with
those measured there, octave band by octave band, in just noticeable differences (JNDs,
ISO 3382-1:2009, annex A, table A.1): a difference of less than one JND is too small to
be heard.

- :func:`compare_impulse_responses` analyses two impulse responses, each at its own
  sample rate, as :func:`~acoustic_spaces.parameters.from_impulse_response` does.
- :func:`compare_parameters` compares two sets of parameters already computed.

Both give a :class:`Comparison`: per parameter and band the simulated and the measured
value, their difference and that difference in JNDs; and per parameter, how many bands
are within one JND and the mean absolute difference in JNDs from 125 Hz to 4 kHz. A
band where either side has no value is missing, with the reason, never zero: a
response that decays by less than 20 dB, a decay time whose range does not clear the
background noise, or G without a reference on both sides.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import (
    _MIN_DECAY_DB,
    _NOISE_CLEARANCE_DB,
    ALL_PARAMETERS,
    DECAY_RANGES,
    PARAMETERS,
    RoomParameters,
    _band_label,
    _label,
    _symbol,
    band_decay,
    from_impulse_response,
)

#: Just noticeable differences (ISO 3382-1:2009, annex A, table A.1), per parameter: the
#: JND, and whether it is a fraction of the measured value (the decay times) rather than
#: a step in the parameter's own unit (dB, s, or none for D50). The directional decay
#: times take that of T30. The standard gives none for LJ, IACC and stage support, which
#: are not compared.
JND: dict[str, tuple[float, bool]] = {
    "edt": (0.05, True),
    "t20": (0.05, True),
    "t30": (0.05, True),
    "c50": (1.0, False),
    "c80": (1.0, False),
    "d50": (0.05, False),
    "ts": (0.010, False),
    "g": (1.0, False),
    "lf": (0.05, False),
    "lfc": (0.05, False),
    "t30_x": (0.05, True),
    "t30_y": (0.05, True),
    "t30_z": (0.05, True),
}
#: The octave bands the mean difference of each parameter is taken over (Hz).
SUMMARY_BANDS_HZ: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000)


@dataclass
class ParameterComparison:
    """One parameter of a simulated and a measured response, one value per octave band.

    ``difference`` is the simulated value minus the measured one, and ``difference_jnd``
    the same in JNDs, so both are positive where the simulation reads higher. They are
    NaN in the bands that cannot be compared; ``missing`` holds the reason for each of
    those, and None for the bands that are compared.
    """

    name: str
    simulated: np.ndarray
    measured: np.ndarray
    #: The JND in each band, in the parameter's unit (NaN for a relative JND where
    #: there is no measured value to take it of).
    jnd: np.ndarray
    difference: np.ndarray
    difference_jnd: np.ndarray
    missing: list[str | None]

    @property
    def compared(self) -> int:
        """How many bands have a value on both sides."""
        return sum(reason is None for reason in self.missing)

    @property
    def within_jnd(self) -> int:
        """How many of the compared bands are at most one JND apart."""
        compared = np.isfinite(self.difference_jnd)
        return int(np.count_nonzero(np.abs(self.difference_jnd[compared]) <= 1.0))

    @property
    def mean_abs_jnd(self) -> float:
        """Mean absolute difference in JNDs over the compared bands from 125 Hz to 4 kHz
        (``SUMMARY_BANDS_HZ``); NaN when none of them is compared."""
        middle = np.isin(OCTAVE_BANDS_HZ, SUMMARY_BANDS_HZ) & np.isfinite(self.difference_jnd)
        if not middle.any():
            return math.nan
        return float(np.abs(self.difference_jnd[middle]).mean())


def _plain(values: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in values]


def _cell(text: str, mark: str = " ") -> str:
    """One band of a table: 7 characters, right-aligned, and room for a mark."""
    return f"{text:>7}{mark}"


@dataclass
class Comparison:
    """A simulated and a measured response compared parameter by parameter.

    ``simulated`` and ``measured`` are the parameters of each side, and ``parameters``
    a :class:`ParameterComparison` for each name of
    :data:`~acoustic_spaces.parameters.PARAMETERS`, and of those further ones with a JND
    that both sides give, which indexing gives as well:
    ``comparison["t30"].difference_jnd``.
    """

    simulated: RoomParameters
    measured: RoomParameters
    parameters: dict[str, ParameterComparison]

    def __getitem__(self, name: str) -> ParameterComparison:
        return self.parameters[name]

    def as_dict(self) -> dict:
        """Plain lists and numbers, with None for missing values (JSON friendly)."""
        out = {}
        for name, p in self.parameters.items():
            jnd, relative = JND[name]
            mean = p.mean_abs_jnd
            out[name] = {
                "jnd": jnd,
                "jnd_relative": relative,
                "simulated": _plain(p.simulated),
                "measured": _plain(p.measured),
                "difference": _plain(p.difference),
                "difference_jnd": _plain(p.difference_jnd),
                "missing": list(p.missing),
                "compared": p.compared,
                "within_jnd": p.within_jnd,
                "mean_abs_jnd": mean if math.isfinite(mean) else None,
            }
        return {
            "bands_hz": list(OCTAVE_BANDS_HZ),
            "summary_bands_hz": list(SUMMARY_BANDS_HZ),
            "parameters": out,
        }

    def table(self) -> str:
        """A text table: per parameter, the simulated and the measured value in each
        band and their difference in JNDs, marked * where it is more than one; then
        why the bands shown as - cannot be compared."""
        lines = [f"{'':<16}" + "".join(_cell(_band_label(f)) for f in OCTAVE_BANDS_HZ)]
        # Missing bands, grouped: (reason, bands) -> the parameters it applies to.
        groups: dict[tuple[str, tuple[int, ...]], list[str]] = {}
        for name, p in self.parameters.items():
            fmt = ".1f" if ALL_PARAMETERS[name][0] == "dB" else ".3f"
            for row, (what, values) in enumerate((("sim", p.simulated), ("meas", p.measured))):
                cells = "".join(_cell(f"{v:{fmt}}" if np.isfinite(v) else "-") for v in values)
                lines.append(f"{_label(name) if row == 0 else '':<10}{what:<6}{cells}")
            cells = "".join(
                _cell(f"{j:+.1f}", "*" if abs(j) > 1.0 else " ") if np.isfinite(j) else _cell("-")
                for j in p.difference_jnd
            )
            lines.append(f"{'':<10}{'JNDs':<6}{cells}")
            bands: dict[str, list[int]] = {}
            for f, reason in zip(OCTAVE_BANDS_HZ, p.missing, strict=True):
                if reason is not None:
                    bands.setdefault(reason, []).append(f)
            for reason, where in bands.items():
                groups.setdefault((reason, tuple(where)), []).append(_symbol(name))
        lines.append("JNDs: simulated minus measured, in just noticeable differences; * beyond one")
        if groups:
            lines.append("missing:")
        for (reason, where), names in groups.items():
            at = (
                "in every band"
                if len(where) == len(OCTAVE_BANDS_HZ)
                else "at " + ", ".join(_band_label(f) for f in where)
            )
            lines.append(f"  {', '.join(names)} {at}: {reason}")
        return "\n".join(line.rstrip() for line in lines)

    def summary(self) -> str:
        """A text table: per parameter, how many of the compared bands are within one
        JND, and the mean absolute difference in JNDs from 125 Hz to 4 kHz."""
        lines = [f"{'':<10}{'within 1 JND':>14}{'mean |JNDs|':>14}"]
        for name, p in self.parameters.items():
            mean = p.mean_abs_jnd
            within = f"{p.within_jnd} of {p.compared}"
            lines.append(
                f"{_label(name):<10}{within:>14}{f'{mean:.1f}' if math.isfinite(mean) else '-':>14}"
            )
        span = f"{_band_label(SUMMARY_BANDS_HZ[0])} to {_band_label(SUMMARY_BANDS_HZ[-1])}"
        lines.append(f"mean |JNDs|: the mean absolute difference in JNDs from {span}")
        return "\n".join(lines)


def _missing(
    name: str, side: str, params: RoomParameters, band: int, decay_range_db: np.ndarray | None
) -> str | None:
    """Why the value of ``name`` in ``band`` of one side cannot be compared, or None."""
    value = float(getattr(params, name)[band])
    if math.isfinite(value):
        return None
    symbol = _symbol(name)
    if not math.isnan(value):
        return f"the {side} {symbol} is infinite"  # clarity with no energy on one side
    if all(math.isnan(float(getattr(params, n)[band])) for n in PARAMETERS):
        # The parameters rule: a band whose decay stops short has none at all.
        return f"the {side} response decays by less than {-_MIN_DECAY_DB:.0f} dB"
    if name == "g":
        return f"no reference for the {side} G"
    if name in DECAY_RANGES:
        bottom = DECAY_RANGES[name][1]
        needed = _NOISE_CLEARANCE_DB - bottom
        if decay_range_db is not None and decay_range_db[band] < needed:
            return (
                f"the {side} decay clears the noise by {decay_range_db[band]:.0f} dB, "
                f"{symbol} needs {needed:.0f} dB"
            )
        return f"the {side} decay does not reach {bottom:.0f} dB for {symbol}"
    return f"no {side} {symbol}"


def _compare(
    simulated: RoomParameters,
    measured: RoomParameters,
    simulated_range_db: np.ndarray | None = None,
    measured_range_db: np.ndarray | None = None,
) -> Comparison:
    bands = len(OCTAVE_BANDS_HZ)
    for side, params in (("simulated", simulated), ("measured", measured)):
        if any(np.shape(getattr(params, name)) != (bands,) for name in PARAMETERS):
            raise ValueError(
                f"the {side} parameters need one value per octave band, "
                f"{_band_label(OCTAVE_BANDS_HZ[0])} to {_band_label(OCTAVE_BANDS_HZ[-1])}"
            )
    names = list(PARAMETERS) + [
        name
        for name in ALL_PARAMETERS
        if name not in PARAMETERS
        and name in JND
        and all(np.isfinite(getattr(p, name)).any() for p in (simulated, measured))
    ]
    compared = {}
    for name in names:
        s = np.asarray(getattr(simulated, name), dtype=float)
        m = np.asarray(getattr(measured, name), dtype=float)
        step, relative = JND[name]
        # A relative JND is taken of the measured value, the one the model is checked
        # against.
        jnd = step * np.abs(m) if relative else np.full(bands, step)
        missing = []
        for b in range(bands):
            reasons = [
                _missing(name, "simulated", simulated, b, simulated_range_db),
                _missing(name, "measured", measured, b, measured_range_db),
            ]
            missing.append("; ".join(r for r in reasons if r) or None)
        ok = np.array([reason is None for reason in missing])
        # Only the compared bands are subtracted, as infinities would warn; the others
        # stay NaN.
        difference = np.full(bands, math.nan)
        difference[ok] = s[ok] - m[ok]
        difference_jnd = np.full(bands, math.nan)
        difference_jnd[ok] = difference[ok] / jnd[ok]
        compared[name] = ParameterComparison(
            name=name,
            simulated=s,
            measured=m,
            jnd=jnd,
            difference=difference,
            difference_jnd=difference_jnd,
            missing=missing,
        )
    return Comparison(simulated=simulated, measured=measured, parameters=compared)


def compare_parameters(simulated: RoomParameters, measured: RoomParameters) -> Comparison:
    """Compare two sets of parameters, one value per octave band of ``OCTAVE_BANDS_HZ``.

    NaN values are missing bands, whose reason is read from the parameters themselves:
    a band with no parameter at all decays by less than 20 dB, G alone is NaN without a
    reference, and a decay time alone when the decay does not reach the bottom of its fit
    range (:func:`compare_impulse_responses` also tells when the noise is in the way).
    LF, LFC and the directional decay times are compared where both sides give them.
    """
    return _compare(simulated, measured)


def compare_impulse_responses(
    simulated: np.ndarray,
    simulated_rate: float,
    measured: np.ndarray,
    measured_rate: float,
    *,
    simulated_reference: float | Sequence[float] | None = None,
    measured_reference: float | Sequence[float] | None = None,
    filters: str = "iec",
    noise: bool = True,
) -> Comparison:
    """Compare a simulated and a measured impulse response, each at its own sample rate.

    Each is mono, (samples,) or (samples, 1), and analysed as by
    :func:`~acoustic_spaces.parameters.from_impulse_response` with the same ``filters``
    and ``noise`` settings. The references are those of its ``reference_energy``: the
    energy of the same source at 10 m in free field, per band or of a broadband impulse
    (0.01 for responses made by this package). G is compared only where both sides have
    one; without, it is missing in every band.
    """
    sides = {}
    for side, ir, rate, reference in (
        ("simulated", simulated, simulated_rate, simulated_reference),
        ("measured", measured, measured_rate, measured_reference),
    ):
        params = from_impulse_response(ir, rate, reference, filters=filters, noise=noise)
        # The decay ranges tell which decay times the background noise leaves out.
        decay = band_decay(ir, rate, filters=filters, noise=noise)
        sides[side] = (params, decay.decay_range_db)
    (sim, sim_range), (meas, meas_range) = sides["simulated"], sides["measured"]
    return _compare(sim, meas, sim_range, meas_range)


__all__ = [
    "JND",
    "SUMMARY_BANDS_HZ",
    "Comparison",
    "ParameterComparison",
    "compare_impulse_responses",
    "compare_parameters",
]
