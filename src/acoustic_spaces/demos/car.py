"""The car of the pass-by demos: its sound power as CNOSSOS-EU gives it, and a sound
made to it. Nothing of the simulation uses it; a demo gives the car its levels and plays
its sound.

CNOSSOS-EU, the European Union's common method for noise maps (Commission Directive (EU)
2015/996, Annex, section 2.2 and Appendix F, Table F-1), gives the sound power of one
vehicle of category 1 (passenger cars, delivery vans of 3.5 t or less, SUVs, MPVs, with
trailers and caravans) at a steady speed v (km/h) on its reference road surface (an
average of dense asphalt concrete 0/11 and stone mastic asphalt 0/11, 2 to 7 years old),
per octave band from 63 Hz to 8 kHz, as two parts at one point 0.05 m above the road:

    rolling noise     L_WR = A_R + B_R lg(v / 70) + 0.08 (20 - t)
    propulsion noise  L_WP = A_P + B_P (v - 70) / 70

with t the air temperature (°C); below 20 km/h the levels are those at 20 km/h. The two
add up (L_W = 10 lg(10^(L_WR/10) + 10^(L_WP/10))). CNOSSOS-EU spreads that power from
the point as a source in free field does (20 lg d + 11 dB) and adds the ground's
reflection apart, as a simulation does: L_W is the car's ``level_db``, per band, as it
stands, radiated alike in every direction. Corrections the method has for other road
surfaces, studded tyres and the acceleration near junctions are left out: the car goes
at one speed on the reference surface.

:func:`car_sound` makes a car's sound, dry, to those levels: tyre noise, and the engine's
firing at its revolutions, each with the octave spectrum of its part. Its time structure
(steady noise from the tyres, the pulses of four cylinders firing in turn) is made up;
its spectrum per octave band is CNOSSOS-EU's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

#: CNOSSOS-EU's coefficients for category 1 (Directive (EU) 2015/996, Annex, Appendix F,
#: Table F-1), per octave band 63 Hz to 8 kHz: rolling noise A_R and B_R, propulsion
#: noise A_P and B_P (dB).
ROLLING_A = (79.7, 85.7, 84.5, 90.2, 97.3, 93.9, 84.1, 74.3)
ROLLING_B = (30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0)
PROPULSION_A = (94.5, 89.2, 88.0, 85.9, 84.2, 86.9, 83.3, 76.1)
PROPULSION_B = (-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0)
#: The speed the coefficients are given at, and the lowest speed the levels change below
#: (km/h).
REFERENCE_SPEED_KMH = 70.0
MIN_SPEED_KMH = 20.0
#: The rolling noise's change with the air temperature (dB per °C, category 1) and the
#: temperature it is given at (°C).
TEMPERATURE_DB_PER_C = 0.08
REFERENCE_TEMPERATURE_C = 20.0
#: How high above the road the car's sound comes from (m).
SOURCE_HEIGHT_M = 0.05

#: The engine of :func:`car_sound`: revolutions per minute and cylinders (four-stroke:
#: each fires once every two turns).
RPM = 2000.0
CYLINDERS = 4
#: How much the time between two firings varies at random (a share of it), and how much
#: the cylinders differ in strength (each its own share, in turn), which gives an engine
#: its orders at half the firing rate and below.
FIRING_JITTER = 0.01
CYLINDER_STRENGTHS = (1.0, 0.86, 1.08, 0.93)
#: How long each firing's pulse of sound takes to die away (s, to 1/e).
PULSE_DECAY_S = 0.002
#: The RMS of :func:`car_sound`'s output (full scale): the level is the source's
#: ``level_db``, not the file's.
SOUND_RMS = 0.1


@dataclass(frozen=True)
class VehiclePower:
    """A vehicle's sound power per octave band (dB re 1 pW, ``OCTAVE_BANDS_HZ``), from
    :func:`light_vehicle_power`."""

    speed_kmh: float
    temperature_c: float
    #: (8,) rolling noise, L_WR.
    rolling_db: np.ndarray
    #: (8,) propulsion noise, L_WP.
    propulsion_db: np.ndarray

    @property
    def total_db(self) -> np.ndarray:
        """(8,) the two parts together, L_W."""
        return 10.0 * np.log10(
            10.0 ** (self.rolling_db / 10.0) + 10.0 ** (self.propulsion_db / 10.0)
        )

    @property
    def sound_power_db(self) -> float:
        """The sound power level in all (dB re 1 pW)."""
        return float(10.0 * np.log10(np.sum(10.0 ** (self.total_db / 10.0))))

    @property
    def sound_power_a_db(self) -> float:
        """The A-weighted sound power level (dB(A) re 1 pW)."""
        from acoustic_spaces.parameters import a_weighted

        return a_weighted(self.total_db)


def light_vehicle_power(
    speed_kmh: float, temperature_c: float = REFERENCE_TEMPERATURE_C
) -> VehiclePower:
    """The sound power of a car (CNOSSOS-EU category 1) at a steady ``speed_kmh`` on the
    reference road surface, in air at ``temperature_c``, per octave band; below 20 km/h
    as at 20 km/h."""
    if not math.isfinite(speed_kmh) or speed_kmh <= 0.0:
        raise ValueError(f"a car's speed is a positive number of km/h, not {speed_kmh!r}")
    v = max(float(speed_kmh), MIN_SPEED_KMH)
    temperature = TEMPERATURE_DB_PER_C * (REFERENCE_TEMPERATURE_C - float(temperature_c))
    rolling = (
        np.array(ROLLING_A)
        + np.array(ROLLING_B) * math.log10(v / REFERENCE_SPEED_KMH)
        + temperature
    )
    propulsion = np.array(PROPULSION_A) + np.array(PROPULSION_B) * (
        (v - REFERENCE_SPEED_KMH) / REFERENCE_SPEED_KMH
    )
    return VehiclePower(float(speed_kmh), float(temperature_c), rolling, propulsion)


def firing_rate_hz(rpm: float = RPM, cylinders: int = CYLINDERS) -> float:
    """How often a four-stroke engine fires (Hz): each cylinder once every two turns."""
    return rpm / 60.0 * cylinders / 2.0


def _to_bands(signal: np.ndarray, rate: int, levels_db: np.ndarray, rounds: int = 3) -> np.ndarray:
    """``signal`` split into the octave bands, each brought to an energy in proportion to
    10^(level/10), and summed again; ``rounds`` times, as the bands overlap at their
    edges and each band's energy moves a little when its neighbours change."""
    from acoustic_spaces.synthesis import split_bands

    wanted = 10.0 ** (np.asarray(levels_db, dtype=float) / 10.0)
    for _ in range(rounds):
        bands = split_bands(signal, rate, len(OCTAVE_BANDS_HZ))
        energy = np.mean(bands**2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gains = np.where(energy > 0.0, np.sqrt(wanted / energy), 0.0)
        signal = (bands * gains[:, None]).sum(axis=0)
    return signal


def _firing(n: int, rate: int, rpm: float, cylinders: int, rng: np.random.Generator) -> np.ndarray:
    """The engine's pulses: one each firing, the time between them varying at random by
    FIRING_JITTER, each cylinder at its strength, each pulse a burst of noise that dies
    away in PULSE_DECAY_S."""
    period = 1.0 / firing_rate_hz(rpm, cylinders)
    count = int(math.ceil(n / rate / period)) + 2
    steps = period * (1.0 + FIRING_JITTER * rng.standard_normal(count))
    times = np.cumsum(steps) - steps[0] * rng.random()
    strengths = np.resize(np.array(CYLINDER_STRENGTHS[:cylinders]), count)
    if len(CYLINDER_STRENGTHS) < cylinders:
        strengths = np.ones(count)
    pulses = np.zeros(n)
    at = np.round(times * rate).astype(np.int64)
    inside = (at >= 0) & (at < n)
    np.add.at(pulses, at[inside], strengths[inside])
    length = int(round(8.0 * PULSE_DECAY_S * rate))
    t = np.arange(length) / rate
    burst = rng.standard_normal(length) * np.exp(-t / PULSE_DECAY_S)
    from acoustic_spaces.render import convolve

    return convolve(pulses, burst)[:n]


def car_sound(
    power: VehiclePower,
    sample_rate: int,
    duration_s: float,
    *,
    rpm: float = RPM,
    cylinders: int = CYLINDERS,
    seed: int = 0,
) -> np.ndarray:
    """A car's sound, dry and mono, ``duration_s`` long at ``sample_rate``: tyre noise
    (noise with the octave spectrum of ``power``'s rolling noise) and the engine at
    ``rpm`` (pulses at its firing rate with the spectrum of the propulsion noise), the
    two at the ratio of their sound powers in each band; at an RMS of SOUND_RMS, as the
    level is the source's ``level_db``."""
    if duration_s <= 0.0:
        raise ValueError("the car's sound needs a positive duration")
    if rpm <= 0.0 or cylinders < 1:
        raise ValueError("an engine turns (rpm > 0) and has a cylinder at least")
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    tyres = _to_bands(rng.standard_normal(n), sample_rate, power.rolling_db)
    engine = _to_bands(
        _firing(n, sample_rate, rpm, cylinders, rng), sample_rate, power.propulsion_db
    )
    x = tyres + engine
    return x * (SOUND_RMS / float(np.sqrt(np.mean(x**2))))


__all__ = [
    "CYLINDERS",
    "MIN_SPEED_KMH",
    "PROPULSION_A",
    "PROPULSION_B",
    "REFERENCE_SPEED_KMH",
    "REFERENCE_TEMPERATURE_C",
    "ROLLING_A",
    "ROLLING_B",
    "RPM",
    "SOUND_RMS",
    "SOURCE_HEIGHT_M",
    "TEMPERATURE_DB_PER_C",
    "VehiclePower",
    "car_sound",
    "firing_rate_hz",
    "light_vehicle_power",
]
