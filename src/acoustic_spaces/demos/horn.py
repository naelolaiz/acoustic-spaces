"""The horns of the pass-by demos: the fire engine's air horn
(:class:`~acoustic_spaces.demos.siren.FireEngine`) and the street car's pair of disc
horns (:class:`~acoustic_spaces.demos.siren.HonkingCar`), their sound, their blasts and
their level as the rule for vehicles' horns sets them (Council Directive 70/388/EEC,
Annex I); each a :class:`Kind`, the air horn's (:data:`AIR_HORN`) the default of every
function here. Nothing of the simulation uses it.

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

The car's horns (:data:`CAR_HORN`) are a pair of disc horns as a maker sells them for
cars: Hella's S77, approved to UN Regulation 28, at 360 and 430 Hz, 110 +/- 3 dB(A) 2 m
away (Hella's sheet for its S70, S77 and S98), within the rule's 105 to 118. A disc
horn's diaphragm strikes its magnet's core every cycle and its tone disc rings: its main
components near 3 kHz, harmonics of the diaphragm's frequency (US patent 8,689,723,
"Disk for horn of vehicle"). Each is made so: its harmonics struck alike and shaped by a
resonance at DISC_HZ, its Q DISC_Q (assumed: a broad peak, which leaves the two
fundamentals 24 and 23 dB under it). The rest is assumed as for the air
horn: as directional as the siren NBS measured, behind the car's front bumper,
CAR_HEIGHT_M up, and sounded in two short blasts every few seconds, a driver's warning.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Kind:
    """A horn: its fundamentals (Hz), its level on its axis (dB(A)) that far in front of
    it (m) on the rule's bench, and how it is sounded: ``blasts`` blasts ``blast_s``
    long, ``gap_s`` apart, from the start of every ``every_s``, each rising to its level
    over ``rise_s`` and falling as fast (s); its name and its sound's file; and for a
    disc horn its tone disc's resonance."""

    noun: str
    hz: tuple[float, ...]
    on_axis_db_a: float
    at_m: float
    blasts: int
    blast_s: float
    gap_s: float
    every_s: float
    rise_s: float
    file: str
    #: A disc horn's tone disc: its resonance (Hz) and how sharp it is (its Q), which
    #: shape the harmonics of each fundamental, all struck alike; 0 for a sawtooth's.
    disc_hz: float = 0.0
    disc_q: float = 0.0

    @property
    def power_s(self) -> float:
        """How many seconds of it its sound power is taken from: three periods of its
        blasts."""
        return 3.0 * self.every_s


#: The fire engine's air horn.
AIR_HORN = Kind(
    "air horn", HORN_HZ, ON_AXIS_DB_A, AT_M, BLASTS, BLAST_S, GAP_S, EVERY_S, RISE_S, HORN_FILE
)
#: The car's pair of disc horns: Hella's S77 (360 and 430 Hz, 110 dB(A) at 2 m), their
#: tone discs ringing at DISC_HZ with a Q of DISC_Q (assumed), in two blasts of a quarter
#: of a second every 3 s (assumed), CAR_HEIGHT_M up (assumed).
DISC_HZ = 3000.0
DISC_Q = 2.0
CAR_HORN = Kind(
    "horn", (360.0, 430.0), 110.0, AT_M, 2, 0.25, 0.15, 3.0, 0.02, "horn.wav", DISC_HZ, DISC_Q
)
CAR_HEIGHT_M = 0.5


def sounding(times, kind: Kind = AIR_HORN) -> np.ndarray:
    """How loud the horn is at ``times`` (s), from 0 (silent) to 1 (a blast at its
    level): its blasts from the start of every period, each rising and falling over its
    rise time by a squared sine."""
    t = np.mod(np.asarray(times, dtype=float), kind.every_s)
    out = np.zeros_like(t)
    for k in range(kind.blasts):
        start = k * (kind.blast_s + kind.gap_s)
        into = t - start
        edge = np.minimum(into, kind.blast_s - into)
        inside = (into >= 0.0) & (into <= kind.blast_s)
        ramp = np.sin(0.5 * np.pi * np.clip(edge / kind.rise_s, 0.0, 1.0)) ** 2
        out = np.where(inside, ramp, out)
    return out


def blast(sample_rate: int, duration_s: float, kind: Kind = AIR_HORN) -> np.ndarray:
    """The horns sounding without a pause, ``duration_s`` long at ``sample_rate``: at
    each of its fundamentals a sawtooth (its harmonics, each 1/k of the fundamental), or
    for a disc horn its harmonics struck alike and shaped by its tone disc's resonance
    (:func:`disc`), faded out as they near the Nyquist frequency; the horns as loud as
    each other, unscaled."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    nyquist = sample_rate / 2.0
    x = np.zeros(n)
    for f in kind.hz:
        k = 1
        while k * f < nyquist:
            fade = min(max((nyquist - k * f) / (0.1 * nyquist), 0.0), 1.0)
            size = disc(k * f, kind) if kind.disc_hz > 0.0 else 1.0 / k
            x += fade * size * np.sin(2.0 * np.pi * k * f * t)
            k += 1
    return x


def disc(f, kind: Kind) -> np.ndarray:
    """How a disc horn's tone disc shapes its harmonic at ``f`` (Hz), from 1 at its
    resonance: a resonance of the kind's disc_hz and disc_q."""
    f = np.asarray(f, dtype=float)
    off = f / kind.disc_hz - kind.disc_hz / f
    return 1.0 / np.sqrt(1.0 + (kind.disc_q * off) ** 2)


def horn_sound(sample_rate: int, duration_s: float, kind: Kind = AIR_HORN) -> np.ndarray:
    """The horn's sound, dry and mono, ``duration_s`` long at ``sample_rate``, from the
    start of its blasts: :func:`blast` times :func:`sounding`, at an RMS of SOUND_RMS
    over whole periods of its blasts."""
    if duration_s <= 0.0:
        raise ValueError("the horn's sound needs a positive duration")
    tone = blast(sample_rate, duration_s, kind)
    gate = sounding(np.arange(len(tone)) / sample_rate, kind)
    return tone * gate * (SOUND_RMS / _rms(sample_rate, kind))


@functools.lru_cache(maxsize=8)
def _rms(sample_rate: int, kind: Kind = AIR_HORN) -> float:
    """The RMS of :func:`blast` times :func:`sounding` over the kind's power_s."""
    tone = blast(sample_rate, kind.power_s, kind)
    x = tone * sounding(np.arange(len(tone)) / sample_rate, kind)
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


def recorded_gain(x: np.ndarray, sample_rate: int, kind: Kind = AIR_HORN) -> float | None:
    """The gain of ``x``, a recording of a horn, that makes it as loud while it sounds as
    :func:`horn_sound` is while it sounds (:func:`sounding_rms` of both), whatever its
    blasts and pauses; None when it is silent."""
    rms = sounding_rms(x, sample_rate)
    if rms <= 0.0:
        return None
    return _sounding(sample_rate, kind) / rms


@functools.lru_cache(maxsize=8)
def _sounding(sample_rate: int, kind: Kind = AIR_HORN) -> float:
    """:func:`sounding_rms` of :func:`horn_sound` over the kind's power_s."""
    return sounding_rms(horn_sound(sample_rate, kind.power_s, kind), sample_rate)


def duty(kind: Kind = AIR_HORN) -> float:
    """The share of the time the horn sounds, in energy: the mean of :func:`sounding`
    squared over a period of its blasts."""
    t = np.linspace(0.0, kind.every_s, 300_001)[:-1]
    return float(np.mean(sounding(t, kind) ** 2))


@functools.lru_cache(maxsize=8)
def _bands(sample_rate: int, kind: Kind = AIR_HORN) -> tuple[float, ...]:
    """The share of a blast's sound in each octave band."""
    from acoustic_spaces.synthesis import split_bands

    x = blast(sample_rate, kind.power_s, kind)
    energy = np.mean(split_bands(x, sample_rate) ** 2, axis=1)
    return tuple(float(e) for e in energy / energy.sum())


def blast_power(sample_rate: int, kind: Kind = AIR_HORN) -> np.ndarray:
    """(8,) the horn's sound power per octave band while it sounds (dB re 1 pW): its
    level on its axis on the bench as in a free field, less its directivity index and the
    spreading of free field."""
    from acoustic_spaces.demos.siren import directivity_index_db
    from acoustic_spaces.parameters import a_weighted

    bands = 10.0 * np.log10(np.array(_bands(sample_rate, kind)))
    total = kind.on_axis_db_a + 20.0 * math.log10(kind.at_m) + 10.0 * math.log10(4.0 * math.pi)
    total -= directivity_index_db()
    return total - a_weighted(bands) + bands


def horn_power(sample_rate: int, kind: Kind = AIR_HORN) -> np.ndarray:
    """(8,) the horn's sound power per octave band (dB re 1 pW) averaged over its blasts
    and the pauses between them, as its sound is made (:func:`horn_sound`): its
    :func:`blast_power` times :func:`duty`."""
    return blast_power(sample_rate, kind) + 10.0 * math.log10(duty(kind))


__all__ = [
    "AIR_HORN",
    "AT_M",
    "CAR_HEIGHT_M",
    "CAR_HORN",
    "DISC_HZ",
    "DISC_Q",
    "Kind",
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
    "disc",
    "duty",
    "horn_power",
    "horn_sound",
    "recorded_gain",
    "sounding",
    "sounding_rms",
]
