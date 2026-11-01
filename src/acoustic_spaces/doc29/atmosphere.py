"""An aircraft's NPD curves adjusted to the air at the aerodrome (Doc 29 Volume 2
Appendix D).

The NPD curves and the spectral classes of the ANP database are normalised to the
"AIR-1845 atmosphere": the average attenuation rates of SAE AIR-1845, per third octave
(:data:`AIR_1845_DB_PER_M`, Volume 2 table D-1), a notional atmosphere of no one
temperature and humidity. Appendix D adjusts the curves to another one in three steps:

1. the aircraft's spectral class (its spectrum 1000 ft under it at the time of the
   maximum level) is taken back to the source, the AIR-1845 attenuation over 1000 ft
   added back (eq. D-1);
2. it is taken out to each of the ten NPD distances by spherical spreading and the
   attenuation of each atmosphere, AIR-1845's and the aerodrome's (eq. D-2, D-3);
3. at each distance, both spectra are A-weighted and summed, and the difference
   between them, :func:`increments_db` (eq. D-4), is added to the NPD curves' levels
   (:func:`adjusted`), the maximum levels' and the sound exposure levels' alike.

The aerodrome's attenuation is ISO 9613-1's (:func:`acoustic_spaces.air.absorption_db_per_m`),
the absorption of SAE ARP-5534, the method Appendix D recommends, averaged over each
third octave as a sound of flat spectrum through it loses it (:func:`band_attenuation_db`):
over a long path a band's upper frequencies are lost first, so its attenuation per metre
falls as the path grows longer, as ARP-5534's does. It gives Volume 2's worked example
(JETW's spectral classes at 10 °C, 80 % relative humidity and 101.325 kPa, its tables
D-3c and D-4) within 0.1 dB, the tables' rounding.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic_spaces.air import absorption_db_per_m

from .tables import FOOT_M, NPD_FEET, NpdCurves, SpectralClass

#: The AIR-1845 atmosphere's attenuation rates (dB/m) per third octave of the spectral
#: classes, 50 Hz to 10 kHz (Volume 2 table D-1, there in dB/100 m).
AIR_1845_DB_PER_M: tuple[float, ...] = tuple(
    rate / 100.0
    for rate in (
        0.033, 0.033, 0.033, 0.066, 0.066, 0.098, 0.131, 0.131, 0.197, 0.230, 0.295, 0.361,
        0.459, 0.590, 0.754, 0.983, 1.311, 1.705, 2.295, 3.115, 3.607, 5.246, 7.213, 9.836,
    )
)  # fmt: skip
#: The spectral classes' distance (m), 1000 ft.
REFERENCE_M = 1000.0 * FOOT_M
#: The A-weighting (dB) of the third octaves 50 Hz to 10 kHz (IEC 61672-1).
A_WEIGHTING_DB: tuple[float, ...] = (
    -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9,
    -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5,
)  # fmt: skip
#: Frequencies each third octave's attenuation is averaged over.
SAMPLES = 401


def _exact_centres() -> np.ndarray:
    """The third octaves' exact mid-band frequencies (Hz, base 10)."""
    return 1000.0 * 10.0 ** ((np.arange(24) - 13) / 10.0)


def band_attenuation_db(
    distance_m: float,
    temperature_c: float = 15.0,
    humidity_percent: float = 70.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """The attenuation (dB) of each third octave, 50 Hz to 10 kHz, over ``distance_m`` of
    air at ``temperature_c``, ``humidity_percent`` and ``pressure_kpa``: ISO 9613-1's
    absorption at each frequency through the band (its edges a tenth of a decade apart),
    the band's energy of a flat spectrum averaged."""
    out = np.empty(24)
    for k, centre in enumerate(_exact_centres()):
        f = np.geomspace(centre * 10.0**-0.05, centre * 10.0**0.05, SAMPLES)
        lost = absorption_db_per_m(f, temperature_c, humidity_percent, pressure_kpa)
        kept = np.trapezoid(10.0 ** (-lost * float(distance_m) / 10.0), f) / (f[-1] - f[0])
        out[k] = -10.0 * math.log10(float(kept))
    return out


def _a_level(levels: np.ndarray) -> float:
    return float(10.0 * np.log10(np.sum(10.0 ** ((levels + np.array(A_WEIGHTING_DB)) / 10.0))))


def increments_db(
    spectral_class: SpectralClass,
    temperature_c: float = 15.0,
    humidity_percent: float = 70.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """What the air at ``temperature_c``, ``humidity_percent`` and ``pressure_kpa`` adds
    to the NPD levels at each of :data:`~.tables.NPD_FEET` against the AIR-1845
    atmosphere's (dB, eq. D-1 to D-4), for an aircraft of ``spectral_class``."""
    rates = np.array(AIR_1845_DB_PER_M)
    source = np.asarray(spectral_class.levels, dtype=float) + rates * REFERENCE_M
    out = np.empty(len(NPD_FEET))
    for k, feet in enumerate(NPD_FEET):
        d = feet * FOOT_M
        spread = source - 20.0 * math.log10(d / REFERENCE_M)
        reference = spread - rates * d
        here = spread - band_attenuation_db(d, temperature_c, humidity_percent, pressure_kpa)
        out[k] = _a_level(here) - _a_level(reference)
    return out


def adjusted(
    curves: NpdCurves,
    spectral_class: SpectralClass,
    temperature_c: float = 15.0,
    humidity_percent: float = 70.0,
    pressure_kpa: float = 101.325,
) -> NpdCurves:
    """``curves`` (an aircraft's NPD curves of one descriptor and operation) in the air
    at ``temperature_c``, ``humidity_percent`` and ``pressure_kpa``: each level plus
    :func:`increments_db` of its distance, for ``spectral_class``, the aircraft's of
    that operation."""
    delta = increments_db(spectral_class, temperature_c, humidity_percent, pressure_kpa)
    return NpdCurves(
        powers=curves.powers,
        levels=tuple(
            tuple(float(v) for v in np.asarray(row, dtype=float) + delta) for row in curves.levels
        ),
    )


__all__ = [
    "AIR_1845_DB_PER_M",
    "A_WEIGHTING_DB",
    "REFERENCE_M",
    "adjusted",
    "band_attenuation_db",
    "increments_db",
]
