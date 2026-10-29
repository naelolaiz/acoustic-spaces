"""Sound in air: speed of sound and atmospheric absorption (ISO 9613-1:1993)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, Resolution, exact_centres

#: Exact octave midband frequencies (base-10 series) for the nominal ``OCTAVE_BANDS_HZ``.
EXACT_OCTAVE_HZ = tuple(1000.0 * 10.0 ** (0.3 * k) for k in range(-4, 4))


#: Speed of sound in air at 0 °C (m/s).
SOUND_SPEED_0C = 331.3


def speed_of_sound(temperature_c: float) -> float:
    """Speed of sound in air (m/s) at the given temperature."""
    return SOUND_SPEED_0C * math.sqrt(1.0 + temperature_c / 273.15)


#: Specific gas constant of dry air (J/(kg K)).
DRY_AIR_J_PER_KG_K = 287.05


def density(temperature_c: float = 15.0, pressure_kpa: float = 101.325) -> float:
    """Density of dry air (kg/m³) at the given temperature and pressure, as an ideal gas."""
    return pressure_kpa * 1000.0 / (DRY_AIR_J_PER_KG_K * (temperature_c + 273.15))


def absorption_db_per_m(
    frequencies_hz: Sequence[float] | np.ndarray,
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """Pure-tone atmospheric attenuation coefficient (dB/m), ISO 9613-1 equations 3 to 5."""
    f = np.asarray(frequencies_hz, dtype=float)
    t = temperature_c + 273.15
    t0, t01 = 293.15, 273.16
    pa_pr = pressure_kpa / 101.325
    c_sat = -6.8346 * (t01 / t) ** 1.261 + 4.6151
    h = humidity_percent * 10.0**c_sat / pa_pr  # molar concentration of water vapour, %
    fr_o = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = (
        pa_pr * (t / t0) ** -0.5 * (9.0 + 280.0 * h * math.exp(-4.170 * ((t / t0) ** (-1 / 3) - 1)))
    )
    return (
        8.686
        * f**2
        * (
            1.84e-11 / pa_pr * (t / t0) ** 0.5
            + (t / t0) ** -2.5
            * (
                0.01275 * math.exp(-2239.1 / t) / (fr_o + f**2 / fr_o)
                + 0.1068 * math.exp(-3352.0 / t) / (fr_n + f**2 / fr_n)
            )
        )
    )


def energy_attenuation_per_m(
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
    bands: Resolution = "octave",
) -> np.ndarray:
    """Energy attenuation ``m`` (1/m) per octave band, or per third-octave band with
    ``bands="third_octave"``, at the exact mid-band frequencies: intensity falls as
    exp(-m * distance).

    This is the ``m`` of the ``4 m V`` air term in Sabine's and Eyring's formulas.
    """
    frequencies = EXACT_OCTAVE_HZ if bands == "octave" else exact_centres(bands)
    alpha = absorption_db_per_m(frequencies, temperature_c, humidity_percent, pressure_kpa)
    return alpha / (10.0 / math.log(10.0))


__all__ = [
    "EXACT_OCTAVE_HZ",
    "OCTAVE_BANDS_HZ",
    "absorption_db_per_m",
    "density",
    "energy_attenuation_per_m",
    "speed_of_sound",
]
