"""The road vehicles of the pass-by demos (cars, buses, lorries, motorcycles): their sound
power as CNOSSOS-EU gives it, and a sound made to it. Nothing of the simulation uses it;
a demo gives the vehicle its levels and plays its sound.

CNOSSOS-EU, the European Union's common method for noise maps (Directive 2002/49/EC,
Annex II, section 2.2 and Appendix F, Table F-1, as amended by Commission Delegated
Directive (EU) 2021/1226), gives the sound power of one vehicle of each category at a
steady speed v (km/h) on its reference road surface (an average of dense asphalt
concrete 0/11 and stone mastic asphalt 0/11, 2 to 7 years old), per octave band from 63
Hz to 8 kHz, as two parts at one point 0.05 m above the road:

    rolling noise     L_WR = A_R + B_R lg(v / 70) + K (20 - t)
    propulsion noise  L_WP = A_P + B_P (v - 70) / 70

with t the air temperature (°C) and K 0.08 dB a degree for category 1, 0.04 for
categories 2 and 3; below 20 km/h the levels are those at 20 km/h. The two add up (L_W
= 10 lg(10^(L_WR/10) + 10^(L_WP/10))); two-wheelers (category 4) have propulsion noise
alone. The point radiates alike in every direction, and the method spreads its power as
a source in free field does (20 lg d + 11 dB), adding the ground's reflection apart, as a
simulation does: L_W is the vehicle's ``level_db``, per band, as it stands. Corrections
the method has for other road surfaces, studded tyres and the acceleration near
junctions are left out: the vehicle goes at one speed on the reference surface.

The table is the one in force since the 2021 amendment, which also says that the point
radiates uniformly (Directive (EU) 2015/996 had it radiate into the half space above the
road): its coefficients A are higher than 2015/996's, by 1.9 to 3.5 dB for a car's
rolling noise (:data:`CATEGORIES`).

:func:`vehicle_sound` makes a vehicle's sound, dry, to those levels: tyre noise, and the
engine's firing at its revolutions, each with the octave spectrum of its part. Its time
structure (steady noise from the tyres, the pulses of the cylinders firing in turn) is
made up; its spectrum per octave band is CNOSSOS-EU's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ


@dataclass(frozen=True)
class Category:
    """A vehicle category of CNOSSOS-EU (Annex II, table 2.2.a) and its coefficients of
    Table F-1, per octave band 63 Hz to 8 kHz (dB): rolling noise A_R and B_R (None for
    two-wheelers, which have propulsion noise alone), propulsion noise A_P and B_P, and
    the rolling noise's change with the air temperature, K (dB per °C)."""

    key: str
    name: str
    covers: str
    rolling_a: tuple[float, ...] | None
    rolling_b: tuple[float, ...] | None
    propulsion_a: tuple[float, ...]
    propulsion_b: tuple[float, ...]
    temperature_db_per_c: float


#: CNOSSOS-EU's vehicle categories with their coefficients (Directive 2002/49/EC, Annex
#: II, Appendix F, Table F-1, as amended by Delegated Directive (EU) 2021/1226; the same
#: in Germany's BUB-D 2021, Annex 4, Table A-1). Category 5 is open, for future needs.
CATEGORIES: dict[str, Category] = {
    "1": Category(
        "1",
        "light motor vehicles",
        "passenger cars, delivery vans of 3.5 t or less, SUVs, MPVs, with trailers and caravans",
        (83.1, 89.2, 87.7, 93.1, 100.1, 96.7, 86.8, 76.2),
        (30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0),
        (97.9, 92.5, 90.7, 87.2, 84.7, 88.0, 84.4, 77.1),
        (-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0),
        0.08,
    ),
    "2": Category(
        "2",
        "medium heavy vehicles",
        "medium heavy vehicles, delivery vans over 3.5 t, buses, motorhomes, with two axles "
        "and twin tyres on the rear axle",
        (88.7, 93.2, 95.7, 100.9, 101.7, 95.1, 87.8, 83.6),
        (30.0, 35.8, 32.6, 23.8, 30.1, 36.2, 38.3, 40.1),
        (105.5, 100.2, 100.5, 98.7, 101.0, 97.8, 91.2, 85.0),
        (-1.9, 4.7, 6.4, 6.5, 6.5, 6.5, 6.5, 6.5),
        0.04,
    ),
    "3": Category(
        "3",
        "heavy vehicles",
        "heavy duty vehicles, touring cars, buses, with three or more axles",
        (91.7, 96.2, 98.2, 104.9, 105.1, 98.5, 91.1, 85.6),
        (30.0, 33.5, 31.3, 25.4, 31.8, 37.1, 38.6, 40.6),
        (108.8, 104.2, 103.5, 102.9, 102.6, 98.5, 93.8, 87.5),
        (0.0, 3.0, 4.6, 5.0, 5.0, 5.0, 5.0, 5.0),
        0.04,
    ),
    "4a": Category(
        "4a",
        "mopeds",
        "two-, three- and four-wheel mopeds",
        None,
        None,
        (93.0, 93.0, 93.5, 95.3, 97.2, 100.4, 95.8, 90.9),
        (4.2, 7.4, 9.8, 11.6, 15.7, 18.9, 20.3, 20.6),
        0.0,
    ),
    "4b": Category(
        "4b",
        "motorcycles",
        "motorcycles with and without sidecars, tricycles and quadricycles",
        None,
        None,
        (99.9, 101.9, 96.7, 94.4, 95.2, 94.7, 92.1, 88.6),
        (3.2, 5.9, 11.9, 11.6, 11.5, 12.6, 11.1, 12.0),
        0.0,
    ),
}
#: The speed the coefficients are given at, and the lowest speed the levels change below
#: (km/h).
REFERENCE_SPEED_KMH = 70.0
MIN_SPEED_KMH = 20.0
#: The temperature the rolling noise is given at (°C).
REFERENCE_TEMPERATURE_C = 20.0
#: How high above the road a vehicle's sound comes from (m).
SOURCE_HEIGHT_M = 0.05

#: The engine of a made sound, unless one is given: revolutions per minute and cylinders
#: of a car's (four-stroke: each fires once every two turns).
RPM = 2000.0
CYLINDERS = 4
#: How much the time between two firings varies at random (a share of it), and how much
#: the cylinders differ in strength (each its own share, in turn, up to six), which gives
#: an engine its orders at half the firing rate and below.
FIRING_JITTER = 0.01
CYLINDER_STRENGTHS = (1.0, 0.86, 1.08, 0.93, 1.03, 0.9)
#: How long each firing's pulse of sound takes to die away (s, to 1/e).
PULSE_DECAY_S = 0.002
#: The RMS of :func:`vehicle_sound`'s output (full scale): the level is the source's
#: ``level_db``, not the file's.
SOUND_RMS = 0.1


@dataclass(frozen=True)
class VehiclePower:
    """A vehicle's sound power per octave band (dB re 1 pW, ``OCTAVE_BANDS_HZ``), from
    :func:`vehicle_power`."""

    category: str
    speed_kmh: float
    temperature_c: float
    #: (8,) rolling noise, L_WR; None for two-wheelers.
    rolling_db: np.ndarray | None
    #: (8,) propulsion noise, L_WP.
    propulsion_db: np.ndarray

    @property
    def total_db(self) -> np.ndarray:
        """(8,) the two parts together, L_W."""
        energy = 10.0 ** (self.propulsion_db / 10.0)
        if self.rolling_db is not None:
            energy = energy + 10.0 ** (self.rolling_db / 10.0)
        return 10.0 * np.log10(energy)

    @property
    def sound_power_db(self) -> float:
        """The sound power level in all (dB re 1 pW)."""
        return float(10.0 * np.log10(np.sum(10.0 ** (self.total_db / 10.0))))

    @property
    def sound_power_a_db(self) -> float:
        """The A-weighted sound power level (dB(A) re 1 pW)."""
        from acoustic_spaces.parameters import a_weighted

        return a_weighted(self.total_db)


def vehicle_power(
    category: str, speed_kmh: float, temperature_c: float = REFERENCE_TEMPERATURE_C
) -> VehiclePower:
    """The sound power of a vehicle of CNOSSOS-EU's ``category`` ("1", "2", "3", "4a" or
    "4b") at a steady ``speed_kmh`` on the reference road surface, in air at
    ``temperature_c``, per octave band; below 20 km/h as at 20 km/h."""
    if category not in CATEGORIES:
        raise ValueError(
            f"CNOSSOS-EU has no vehicle category {category!r}; it has {', '.join(CATEGORIES)}"
        )
    if not math.isfinite(speed_kmh) or speed_kmh <= 0.0:
        raise ValueError(f"a vehicle's speed is a positive number of km/h, not {speed_kmh!r}")
    kind = CATEGORIES[category]
    v = max(float(speed_kmh), MIN_SPEED_KMH)
    rolling = None
    if kind.rolling_a is not None and kind.rolling_b is not None:
        rolling = (
            np.array(kind.rolling_a)
            + np.array(kind.rolling_b) * math.log10(v / REFERENCE_SPEED_KMH)
            + kind.temperature_db_per_c * (REFERENCE_TEMPERATURE_C - float(temperature_c))
        )
    propulsion = np.array(kind.propulsion_a) + np.array(kind.propulsion_b) * (
        (v - REFERENCE_SPEED_KMH) / REFERENCE_SPEED_KMH
    )
    return VehiclePower(category, float(speed_kmh), float(temperature_c), rolling, propulsion)


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


def vehicle_sound(
    power: VehiclePower,
    sample_rate: int,
    duration_s: float,
    *,
    rpm: float = RPM,
    cylinders: int = CYLINDERS,
    seed: int = 0,
) -> np.ndarray:
    """A vehicle's sound, dry and mono, ``duration_s`` long at ``sample_rate``: tyre noise
    (noise with the octave spectrum of ``power``'s rolling noise; none for a two-wheeler)
    and the engine at ``rpm`` with ``cylinders`` (pulses at its firing rate with the
    spectrum of the propulsion noise), the two at the ratio of their sound powers in each
    band; at an RMS of SOUND_RMS, as the level is the source's ``level_db``."""
    if duration_s <= 0.0:
        raise ValueError("the vehicle's sound needs a positive duration")
    if rpm <= 0.0 or cylinders < 1:
        raise ValueError("an engine turns (rpm > 0) and has a cylinder at least")
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    engine = _to_bands(
        _firing(n, sample_rate, rpm, cylinders, rng), sample_rate, power.propulsion_db
    )
    x = engine
    if power.rolling_db is not None:
        x = _to_bands(noise, sample_rate, power.rolling_db) + engine
    return x * (SOUND_RMS / float(np.sqrt(np.mean(x**2))))


__all__ = [
    "CATEGORIES",
    "CYLINDERS",
    "MIN_SPEED_KMH",
    "REFERENCE_SPEED_KMH",
    "REFERENCE_TEMPERATURE_C",
    "RPM",
    "SOUND_RMS",
    "SOURCE_HEIGHT_M",
    "Category",
    "VehiclePower",
    "firing_rate_hz",
    "vehicle_power",
    "vehicle_sound",
]
