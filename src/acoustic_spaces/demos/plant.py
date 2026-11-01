"""The road works of the street example: construction plant as the Defra noise database
measured it, a lorry's reversing alarm, and the sounds made to them; the street's works
demo (:mod:`acoustic_spaces.demos.works`) plays them. Nothing of the simulation uses it.

Each machine is a row of Defra's *Update of noise database for prediction of noise on
construction and open sites* (Hepworth Acoustics for the Department for Environment,
Food and Rural Affairs, 2005; the data BS 5228-1 Annex C gives): its sound pressure
level in each octave band from 63 Hz to 8 kHz, "standardised to 10 metres from the
plant", and its A-weighted level there, the L_eq of an activity or, for plant on the
move, the L_max of a drive-by. Its sound power in each band is that level plus 28 dB
(:data:`TO_POWER_DB`): a point radiating over hard ground into a half sphere, L_p = L_W -
20 lg r - 8 dB at r metres. How high each machine's sound comes
from is assumed (Defra gives no height), and so is how its sound goes on in time:

- the breaker's blows: a burst of noise each, BLOWS_PER_MINUTE of them, the middle of the
  320 to 640 a minute Epiroc gives its MB 1650 (a 1650 kg breaker, as Defra's), each
  dying away over BLOW_DECAY_S, the time between them varying at random by BLOW_JITTER;
- the engines' firing: the road vehicles' pulses
  (:func:`acoustic_spaces.demos.road.vehicle_sound`), at the speed Defra gives (the
  generator's 3000 rpm) or one assumed (the dumper's);

each brought to its row's octave spectrum, so that the blows or the firing carry all of
the machine's sound Defra measured (its engine, its hydraulics, what it strikes). A
machine's made sound is steady: Defra's L_eq of an activity holds the activity's pauses,
which the made sound spreads over all of its time.

The reversing alarm is a tonal one, its sound a sine at ALARM_HZ, on and off for equal
times ALARM_PER_S times a second, each beep rising and falling over ALARM_RISE_S. SAE
J994 (its alarms' rule) asks for 700 to 2800 Hz, 1 to 2 beeps a second, on and off for
about as long, and its Type B for 107 dB(A) 1.2 m in front of the alarm on its axis, 1.2
m above a reflecting plane. Hella's 009148021 is such an alarm at 1200 Hz
(ALARM_ON_AXIS_DB_A at ALARM_AT_M). Its sound power while it sounds is that level as if
in a free field (the plane's reflection, 7 dB under the direct sound at the microphone,
left out) less its directivity index, as directional as the siren NBS measured
(:func:`acoustic_spaces.demos.siren.directivity`, assumed, as for the fire engine's air
horn); its sound power in the scene is its average over its beeps and pauses, half of it.
The rate of its beeps, ALARM_PER_S, is assumed within J994's; its height, ALARM_HEIGHT_M,
is what makers of alarms advise (about 4 feet up, at the vehicle's back, facing behind
it). The alarms C. Laroche and others measured behind four heavy vehicles sounded at
1140 to 3025 Hz (Canadian Acoustics 21(3), 1993).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

#: A machine's sound power per octave band over its level 10 m away (dB): a point over
#: hard ground radiating into a half sphere, 20 lg 10 + 10 lg 2π.
TO_POWER_DB = 28.0
#: How far Defra's levels are from the plant (m).
DEFRA_AT_M = 10.0
#: The RMS of every made sound (full scale): the level is the source's ``level_db``.
SOUND_RMS = 0.1


@dataclass(frozen=True)
class Machine:
    """A machine of the road works: a row of Defra's database and how its sound is made."""

    #: Its source in the street example.
    name: str
    #: In words, for the demo's lines.
    noun: str
    #: Defra's table and row, and the row's own words.
    table: int
    row: int
    item: str
    #: Its unweighted octave-band levels 10 m away, 63 Hz to 8 kHz (dB), and its
    #: A-weighted level there (dB(A)), as the row gives them.
    levels_10m_db: tuple[float, ...]
    level_10m_db_a: float
    #: Whether the row is the L_max of a drive-by, not the L_eq of an activity.
    drive_by: bool
    #: How high its sound comes from (m; assumed).
    height_m: float
    #: How its sound is made: "blows" or "engine".
    made: str
    #: An engine's speed and cylinders (four-stroke).
    rpm: float = 0.0
    cylinders: int = 0
    #: Its net installed power (kW), for the EU's limits.
    power_kw: float | None = None
    #: The row of the FTA's table 7-4 for its ground vibration
    #: (:data:`acoustic_spaces.vibration.PLANT`), or None for one the manual counts
    #: among those that make little (it names air compressors, light trucks and loaders).
    vibration: str | None = None

    def sound_power(self) -> np.ndarray:
        """(8,) its sound power per octave band (dB re 1 pW)."""
        return np.asarray(self.levels_10m_db, dtype=float) + TO_POWER_DB

    def sound_power_a(self) -> float:
        """Its A-weighted sound power (dB(A) re 1 pW)."""
        from acoustic_spaces.parameters import a_weighted

        return float(a_weighted(self.sound_power()))


#: The machines, by their sources in the street example.
BREAKER = Machine(
    "breaker",
    "a 1650 kg breaker on a 15 t excavator",
    1,
    9,
    "Breaker Mounted on Excavator 121 (15 t) 1650 kg breaker (Breaking Up Brick Foundations)",
    (88.0, 88.0, 86.0, 89.0, 83.0, 83.0, 80.0, 76.0),
    90.0,
    False,
    1.0,
    "blows",
    power_kw=121.0,
    vibration="hoe_ram",
)
GENERATOR = Machine(
    "generator",
    "a 6 kVA diesel generator",
    4,
    87,
    "Diesel Generator 7.5 6 kVA / 3000 rpm (Power for Lighting)",
    (77.0, 72.0, 64.0, 60.0, 59.0, 57.0, 54.0, 42.0),
    65.0,
    False,
    0.5,
    "engine",
    rpm=3000.0,
    cylinders=1,
    power_kw=7.5,
)
DUMPER = Machine(
    "dumper",
    "a 9 t dumper",
    4,
    4,
    "Dumper 75 9 t (Distribution of Materials, a drive-by)",
    (82.0, 76.0, 75.0, 74.0, 68.0, 68.0, 64.0, 55.0),
    76.0,
    True,
    1.5,
    "engine",
    rpm=2200.0,
    cylinders=4,
    power_kw=75.0,
    vibration="loaded_trucks",
)
MACHINES = {m.name: m for m in (BREAKER, GENERATOR, DUMPER)}

#: The breaker's blows: how many a minute, each a burst of noise dying away over
#: BLOW_DECAY_S, the time between them varying at random by BLOW_JITTER (assumed but
#: the rate, the middle of the MB 1650's 320 to 640).
BLOWS_PER_MINUTE = 480.0
BLOWS_RANGE = (320.0, 640.0)
BLOW_DECAY_S = 0.012
BLOW_JITTER = 0.03

#: The reversing alarm: its source in the street example, its tone (Hz), its level on
#: its axis (dB(A)) that far in front of it (m), how many beeps a second, how long each
#: beep rises and falls (s) and how high it is (m).
ALARM = "dumper_alarm"
ALARM_HZ = 1200.0
ALARM_ON_AXIS_DB_A = 107.0
ALARM_AT_M = 1.2
ALARM_PER_S = 1.5
ALARM_RISE_S = 0.005
ALARM_HEIGHT_M = 1.2
#: SAE J994's bounds: its tone (Hz) and its beeps a second.
J994_HZ = (700.0, 2800.0)
J994_PER_S = (1.0, 2.0)


def defra_a_sum(machine: Machine) -> float:
    """The A-weighted sum of a row's octave levels (dB(A)), to hold against its own."""
    from acoustic_spaces.parameters import a_weighted

    return float(a_weighted(np.asarray(machine.levels_10m_db, dtype=float)))


def eu_limit_db_a(kind: str, value: float) -> int:
    """The EU's most sound power (dB(A) re 1 pW, rounded to the nearest whole decibel) a
    machine placed on the market from 2006 may have, Stage II of Directive 2000/14/EC,
    Article 12: ``kind`` "excavator" (``value`` its net installed power, kW), "dumper"
    (a wheeled one, kW) or "generator" (its electric power, kW)."""
    if kind == "excavator":
        level = 93.0 if value <= 15.0 else 80.0 + 11.0 * math.log10(value)
    elif kind == "dumper":
        level = 101.0 if value <= 55.0 else 82.0 + 11.0 * math.log10(value)
    elif kind == "generator":
        level = (95.0 if value <= 2.0 or value > 10.0 else 96.0) + math.log10(value)
    else:
        raise ValueError(f"no limit for {kind!r}")
    return int(math.floor(level + 0.5))


# --------------------------------------------------------------------------- the sounds


def _bursts(n: int, rate: int, per_s: float, decay_s: float, jitter: float, rng) -> np.ndarray:
    """A burst of noise ``per_s`` times a second, each dying away over ``decay_s``, the
    time between them varying at random by ``jitter``."""
    from acoustic_spaces.render import convolve

    period = 1.0 / per_s
    count = int(math.ceil(n / rate / period)) + 2
    steps = period * (1.0 + jitter * rng.standard_normal(count))
    times = np.cumsum(steps) - steps[0] * rng.random()
    pulses = np.zeros(n)
    at = np.round(times * rate).astype(np.int64)
    inside = (at >= 0) & (at < n)
    np.add.at(pulses, at[inside], 1.0)
    length = int(round(8.0 * decay_s * rate))
    t = np.arange(length) / rate
    burst = rng.standard_normal(length) * np.exp(-t / decay_s)
    return convolve(pulses, burst)[:n]


def machine_sound(
    machine: Machine,
    sample_rate: int,
    duration_s: float,
    *,
    blows_per_minute: float = BLOWS_PER_MINUTE,
    seed: int = 0,
) -> np.ndarray:
    """A machine's sound, dry and mono, ``duration_s`` long at ``sample_rate``: its blows
    or its engine's firing, with its row's octave spectrum, at an RMS of SOUND_RMS."""
    from acoustic_spaces.demos.road import _firing, _to_bands

    if duration_s <= 0.0:
        raise ValueError("a machine's sound needs a positive duration")
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    if machine.made == "blows":
        if blows_per_minute <= 0.0:
            raise ValueError("a breaker strikes a blow at least each minute")
        x = _bursts(n, sample_rate, blows_per_minute / 60.0, BLOW_DECAY_S, BLOW_JITTER, rng)
    else:
        x = _firing(n, sample_rate, machine.rpm, machine.cylinders, rng)
    x = _to_bands(x, sample_rate, machine.sound_power())
    return x * (SOUND_RMS / float(np.sqrt(np.mean(x**2))))


def beeping(times) -> np.ndarray:
    """How loud the alarm is at ``times`` (s), from 0 to 1: on for the first half of each
    1/ALARM_PER_S, rising and falling over ALARM_RISE_S by a squared sine."""
    period = 1.0 / ALARM_PER_S
    into = np.mod(np.asarray(times, dtype=float), period)
    on = period / 2.0
    edge = np.minimum(into, on - into)
    ramp = np.sin(0.5 * np.pi * np.clip(edge / ALARM_RISE_S, 0.0, 1.0)) ** 2
    return np.where(into <= on, ramp, 0.0)


def alarm_sound(sample_rate: int, duration_s: float) -> np.ndarray:
    """The reversing alarm's sound, dry and mono, ``duration_s`` long at ``sample_rate``
    from the start of a beep: a sine at ALARM_HZ times :func:`beeping`, at an RMS of
    SOUND_RMS over whole periods of its beeps."""
    if duration_s <= 0.0:
        raise ValueError("the alarm's sound needs a positive duration")
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    x = np.sin(2.0 * np.pi * ALARM_HZ * t) * beeping(t)
    return x * (SOUND_RMS / _alarm_rms(sample_rate))


#: How many seconds of the alarm its level is taken from: whole periods of its beeps.
ALARM_POWER_S = 6.0


@functools.lru_cache(maxsize=8)
def _alarm_rms(sample_rate: int) -> float:
    t = np.arange(int(round(ALARM_POWER_S * sample_rate))) / sample_rate
    x = np.sin(2.0 * np.pi * ALARM_HZ * t) * beeping(t)
    return float(np.sqrt(np.mean(x**2)))


def alarm_duty() -> float:
    """The share of the time the alarm sounds, in energy: the mean of :func:`beeping`
    squared over a period of its beeps."""
    t = np.linspace(0.0, 1.0 / ALARM_PER_S, 300_001)[:-1]
    return float(np.mean(beeping(t) ** 2))


@functools.lru_cache(maxsize=8)
def _alarm_bands(sample_rate: int) -> tuple[float, ...]:
    """The share of the alarm's tone in each octave band, as the octave filters split it."""
    from acoustic_spaces.synthesis import split_bands

    t = np.arange(int(round(ALARM_POWER_S * sample_rate))) / sample_rate
    energy = np.mean(split_bands(np.sin(2.0 * np.pi * ALARM_HZ * t), sample_rate) ** 2, axis=1)
    return tuple(float(e) for e in energy / energy.sum())


def alarm_beep_power(sample_rate: int) -> np.ndarray:
    """(8,) the alarm's sound power per octave band while it sounds (dB re 1 pW):
    ALARM_ON_AXIS_DB_A at ALARM_AT_M on its axis as in a free field, less its directivity
    index; each band A-weighted at its centre, as the engine weighs it."""
    from acoustic_spaces.demos.siren import directivity_index_db
    from acoustic_spaces.parameters import a_weighted

    bands = 10.0 * np.log10(np.maximum(np.array(_alarm_bands(sample_rate)), 1e-30))
    total = ALARM_ON_AXIS_DB_A + 20.0 * math.log10(ALARM_AT_M) + 10.0 * math.log10(4.0 * math.pi)
    total -= directivity_index_db()
    return total - a_weighted(bands) + bands


def alarm_power(sample_rate: int) -> np.ndarray:
    """(8,) the alarm's sound power per octave band (dB re 1 pW) averaged over its beeps
    and its pauses, as its sound is made: :func:`alarm_beep_power` times
    :func:`alarm_duty`."""
    return alarm_beep_power(sample_rate) + 10.0 * math.log10(alarm_duty())


__all__ = [
    "ALARM",
    "ALARM_AT_M",
    "ALARM_HEIGHT_M",
    "ALARM_HZ",
    "ALARM_ON_AXIS_DB_A",
    "ALARM_PER_S",
    "ALARM_RISE_S",
    "BLOWS_PER_MINUTE",
    "BLOWS_RANGE",
    "BREAKER",
    "DEFRA_AT_M",
    "DUMPER",
    "GENERATOR",
    "J994_HZ",
    "J994_PER_S",
    "MACHINES",
    "SOUND_RMS",
    "TO_POWER_DB",
    "Machine",
    "alarm_beep_power",
    "alarm_duty",
    "alarm_power",
    "alarm_sound",
    "beeping",
    "defra_a_sum",
    "eu_limit_db_a",
    "machine_sound",
]
