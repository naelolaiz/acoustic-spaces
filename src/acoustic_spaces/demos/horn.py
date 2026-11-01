"""The air horn of the pass-by demos' fire engine: its sound, its blasts and its level
as the rule for vehicles' horns sets them (Council Directive 70/388/EEC, Annex I), and
the vehicle that carries it (:class:`~acoustic_spaces.demos.siren.FireEngine`). Nothing
of the simulation uses it.

The rule tests a horn on a bench, in a free field or an anechoic room, 2 m in front of
it in the direction it is loudest, the horn and the microphone 1.2 m above the ground
(points 1.2.1.1 and 1.2.3.5), and asks for 105 to 118 dB(A) there (point 1.2.1.6); its
sound between 1800 and 3550 Hz louder than any component above 3550 Hz, and at least
105 dB(A) (point 1.2.3.1); its least level within 0.2 s of its start (point 1.2.1.7);
and a sound that keeps its tone (point 1.1). On the vehicle the loudest level 7 m in
front of it, 0.5 to 1.5 m above the ground, is to be 93 dB(A) at least (points 2.1.1 to
2.1.4).

The horn here is as loud as the rule lets a horn be on the bench, ON_AXIS_DB_A 2 m in
front of it on its axis in a free field. The rest is assumed, as no measured air horn
of a fire engine was found: two trumpets sounding together at HORN_HZ, a minor third
apart, as a pair of a car's horns is (about 405 to 420 Hz and 500 Hz: J. Lansey, "The
Psychoacoustics of the Two-Tone Horn", 2025), an octave and a half lower, within the 125
to 180 Hz that Wikipedia's article on vehicle horns gives trucks' and buses' air horns
(without a source); each a sawtooth (all its harmonics, each 1/k of the fundamental), as
a diaphragm that lets the air through in a puff each cycle would send; as directional
in every band as the siren NBS measured is
(:func:`~acoustic_spaces.demos.siren.directivity`); and sounded in BLASTS short blasts
BLAST_S long, GAP_S apart, every EVERY_S, each rising to its level over RISE_S (the rule:
within 0.2 s) and falling as fast. Its sound power in the scene is its average over its
blasts and pauses, as its made sound has them; a recording of a horn plays its blasts as
loud as the made horn's, whatever its own pauses (:func:`recorded_gain`).
"""

from __future__ import annotations

import functools
import math

import numpy as np

#: The two trumpets' fundamentals (Hz), a minor third (5:6) apart.
HORN_HZ = (150.0, 180.0)
#: How it is sounded: BLASTS blasts BLAST_S long, GAP_S apart, from the start of every
#: EVERY_S; each rises to its level over RISE_S and falls over as long (s).
BLASTS = 2
BLAST_S = 0.6
GAP_S = 0.3
EVERY_S = 3.0
RISE_S = 0.05
#: The rule's bench: the horn's level on its axis in a free field (dB(A); the most the
#: rule allows, 105 the least) and how far away (m).
ON_AXIS_DB_A = 118.0
AT_M = 2.0
#: How high the horn is (m): as high as on the rule's bench, on the vehicle too (where a
#: fire engine carries its horns varies; assumed).
HEIGHT_M = 1.2
#: On the vehicle the rule asks this much at least (dB(A)), 7 m in front of it, 0.5 to
#: 1.5 m up.
VEHICLE_DB_A = 93.0
#: The rule's band, which is to be louder than any component above it (Hz).
RULE_BAND_HZ = (1800.0, 3550.0)
#: The RMS of :func:`horn_sound`'s output over whole periods of its blasts (full
#: scale): the level is the source's ``level_db``, not the file's.
SOUND_RMS = 0.1
#: How many seconds of it its sound power is taken from: whole periods of its blasts.
POWER_S = 3 * EVERY_S
HORN_FILE = "horn.wav"


def sounding(times) -> np.ndarray:
    """How loud the horn is at ``times`` (s), from 0 (silent) to 1 (a blast at its
    level): BLASTS blasts from the start of every EVERY_S, each rising and falling over
    RISE_S by a squared sine."""
    t = np.mod(np.asarray(times, dtype=float), EVERY_S)
    out = np.zeros_like(t)
    for k in range(BLASTS):
        start = k * (BLAST_S + GAP_S)
        into = t - start
        edge = np.minimum(into, BLAST_S - into)
        inside = (into >= 0.0) & (into <= BLAST_S)
        ramp = np.sin(0.5 * np.pi * np.clip(edge / RISE_S, 0.0, 1.0)) ** 2
        out = np.where(inside, ramp, out)
    return out


def blast(sample_rate: int, duration_s: float) -> np.ndarray:
    """The two trumpets sounding without a pause, ``duration_s`` long at
    ``sample_rate``: a sawtooth at each of HORN_HZ (its harmonics, each 1/k of the
    fundamental, faded out as they near the Nyquist frequency), as loud as each other,
    unscaled."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    nyquist = sample_rate / 2.0
    x = np.zeros(n)
    for f in HORN_HZ:
        k = 1
        while k * f < nyquist:
            fade = min(max((nyquist - k * f) / (0.1 * nyquist), 0.0), 1.0)
            x += fade * np.sin(2.0 * np.pi * k * f * t) / k
            k += 1
    return x


def horn_sound(sample_rate: int, duration_s: float) -> np.ndarray:
    """The horn's sound, dry and mono, ``duration_s`` long at ``sample_rate``, from the
    start of its blasts: :func:`blast` times :func:`sounding`, at an RMS of SOUND_RMS
    over whole periods of its blasts."""
    if duration_s <= 0.0:
        raise ValueError("the horn's sound needs a positive duration")
    tone = blast(sample_rate, duration_s)
    gate = sounding(np.arange(len(tone)) / sample_rate)
    return tone * gate * (SOUND_RMS / _rms(sample_rate))


@functools.lru_cache(maxsize=8)
def _rms(sample_rate: int) -> float:
    """The RMS of :func:`blast` times :func:`sounding` over POWER_S."""
    tone = blast(sample_rate, POWER_S)
    x = tone * sounding(np.arange(len(tone)) / sample_rate)
    return float(np.sqrt(np.mean(x**2)))


#: A horn sounds in the blocks this long (s) within GATE_DB of its loudest block.
BLOCK_S = 0.05
GATE_DB = 20.0


def sounding_rms(x: np.ndarray, sample_rate: int) -> float:
    """The RMS of ``x`` while it sounds: the median of its BLOCK_S blocks' within GATE_DB
    of the loudest, so that neither its pauses nor how its blasts rise and fall count
    (0 when it is silent)."""
    from acoustic_spaces.demos.audio import mono

    m = mono(x)
    block = max(1, int(BLOCK_S * sample_rate))
    n = len(m) // block
    if n == 0:
        return float(np.sqrt(np.mean(m**2))) if len(m) else 0.0
    power = np.mean(m[: n * block].reshape(n, block) ** 2, axis=1)
    loudest = float(np.max(power))
    if loudest <= 0.0:
        return 0.0
    return float(np.sqrt(np.median(power[power >= loudest * 10.0 ** (-GATE_DB / 10.0)])))


def recorded_gain(x: np.ndarray, sample_rate: int) -> float | None:
    """The gain of ``x``, a recording of a horn, that makes it as loud while it sounds as
    :func:`horn_sound` is while it sounds (:func:`sounding_rms` of both), whatever its
    blasts and pauses; None when it is silent."""
    rms = sounding_rms(x, sample_rate)
    if rms <= 0.0:
        return None
    return _sounding(sample_rate) / rms


@functools.lru_cache(maxsize=8)
def _sounding(sample_rate: int) -> float:
    """:func:`sounding_rms` of :func:`horn_sound` over POWER_S."""
    return sounding_rms(horn_sound(sample_rate, POWER_S), sample_rate)


def duty() -> float:
    """The share of the time the horn sounds, in energy: the mean of :func:`sounding`
    squared over a period of its blasts."""
    t = np.linspace(0.0, EVERY_S, 300_001)[:-1]
    return float(np.mean(sounding(t) ** 2))


@functools.lru_cache(maxsize=8)
def _bands(sample_rate: int) -> tuple[float, ...]:
    """The share of a blast's sound in each octave band."""
    from acoustic_spaces.synthesis import split_bands

    x = blast(sample_rate, POWER_S)
    energy = np.mean(split_bands(x, sample_rate) ** 2, axis=1)
    return tuple(float(e) for e in energy / energy.sum())


def blast_power(sample_rate: int) -> np.ndarray:
    """(8,) the horn's sound power per octave band while it sounds (dB re 1 pW):
    ON_AXIS_DB_A at AT_M on its axis, less its directivity index and the spreading of
    free field."""
    from acoustic_spaces.demos.siren import directivity_index_db
    from acoustic_spaces.parameters import a_weighted

    bands = 10.0 * np.log10(np.array(_bands(sample_rate)))
    total = ON_AXIS_DB_A + 20.0 * math.log10(AT_M) + 10.0 * math.log10(4.0 * math.pi)
    total -= directivity_index_db()
    return total - a_weighted(bands) + bands


def horn_power(sample_rate: int) -> np.ndarray:
    """(8,) the horn's sound power per octave band (dB re 1 pW) averaged over its blasts
    and the pauses between them, as its sound is made (:func:`horn_sound`): its
    :func:`blast_power` times :func:`duty`."""
    return blast_power(sample_rate) + 10.0 * math.log10(duty())


__all__ = [
    "AT_M",
    "BLOCK_S",
    "BLASTS",
    "BLAST_S",
    "EVERY_S",
    "GATE_DB",
    "GAP_S",
    "HORN_FILE",
    "HEIGHT_M",
    "HORN_HZ",
    "ON_AXIS_DB_A",
    "RISE_S",
    "RULE_BAND_HZ",
    "SOUND_RMS",
    "VEHICLE_DB_A",
    "blast",
    "blast_power",
    "duty",
    "horn_power",
    "horn_sound",
    "recorded_gain",
    "sounding",
    "sounding_rms",
]
