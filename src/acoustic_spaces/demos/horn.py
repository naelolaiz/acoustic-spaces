"""The horns of the pass-by demos: the fire engine's air horn
(:class:`~acoustic_spaces.demos.siren.FireEngine`) and the street car's pair of disc
horns (:class:`~acoustic_spaces.demos.siren.HonkingCar`), their sound, their blasts and
their level as the rule for vehicles' horns sets them (Council Directive 70/388/EEC,
Annex I), and the train's warning horn
(:class:`~acoustic_spaces.demos.rail.TrainVehicle`) as the rule for trains' horns sets
it; each a :class:`Kind`, the air horn's (:data:`AIR_HORN`) the default of every
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

The train's horn (:data:`TRAIN_HORN`) sounds the two notes of the EU's rule for
locomotives and passenger trains (TSI LOC&PAS, Regulation (EU) No 1302/2014, point
4.2.7.2.1), one after the other: the high note at 660 Hz and the low at 370 Hz (660 +/-
30 and 370 +/- 20 Hz). Its level is EN 15153-2:2020's (point 4.2.7.2.2; its clause
5.2.2), which asks for 101 to 109 dB(C) of each note sounded alone (L_pCeq), measured
on the vehicle over an open, flat site 25 m ahead of the train's front on its centre
line (its Figure 1); each note here is at 109, the most. That level holds the site's
ground reflection, which its sound power here is taken out of (:class:`Site`,
:func:`ground_db`): the horn's harmonics, each its own frequency, add in pressure with
their reflections off the ground. The site's ground, the heights of the horn and the
microphone over it are assumptions (the sample of EN 15153-2 seen leaves out where the
microphone stands): the ballast of the railway example, the horn 2.0 m above the rail
top, 0.2 m above the ballast, and the microphone 1.5 m up, as M. Toward et al.
measured train horns ("Sound propagation and audibility of train horns", ISVR and
RSSB). Each note is an air horn's, a sawtooth, as the fire engine's is, rich in
harmonics as Toward et al. found a train horn; its blasts are a driver's warning,
assumed; it is as directional as the siren NBS measured (Toward et al. measured a train
horn 1 dB down 30 degrees off its axis, where the siren is 3.6 dB down).
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
class Site:
    """Where a horn's level is measured over the ground (not in a free field): the
    ground (a preset of :data:`acoustic_spaces.materials.ground.PRESETS`, reacting
    locally), the horn's and the microphone's heights above it (m)."""

    ground: str
    horn_m: float
    microphone_m: float


@dataclass(frozen=True)
class Kind:
    """A horn: its fundamentals (Hz), its level on its axis (dB, weighted as
    ``weighting`` says) that far in front of it (m) where its rule measures it, and how
    it is sounded: ``blasts`` blasts ``blast_s`` long, ``gap_s`` apart, from the start of
    every ``every_s``, each rising to its level over ``rise_s`` and falling as fast (s);
    its name and its sound's file; for a disc horn its tone disc's resonance; whether
    its fundamentals sound together or one a blast in turn, each then at that level;
    and the ground it is measured over (none: a free field)."""

    noun: str
    hz: tuple[float, ...]
    on_axis_db: float
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
    #: The weighting of its level: "A" or "C".
    weighting: str = "A"
    #: Whether each blast sounds the next of its fundamentals alone (a train's horn's
    #: notes), each as loud as its level; else they sound together.
    in_turn: bool = False
    site: Site | None = None

    @property
    def power_s(self) -> float:
        """How many seconds of it its sound power is taken from: three periods of its
        blasts."""
        return 3.0 * self.every_s

    @property
    def notes(self) -> tuple[tuple[float, ...], ...]:
        """What it sounds, each the fundamentals of one blast: all of them, or one at a
        time."""
        return tuple((f,) for f in self.hz) if self.in_turn else (self.hz,)


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
#: The train's horn: its high note and its low note in turn (TSI LOC&PAS's 660 and 370
#: Hz), each 109 dB(C) 25 m ahead of the train's front (EN 15153-2's most), measured
#: over the site TRAIN_SITE; sounded a second each, 0.2 s apart, every 4 s (assumed).
TRAIN_HORN_HEIGHT_M = 2.0
TRAIN_SITE = Site("gravel", 0.2 + TRAIN_HORN_HEIGHT_M, 1.5)
TRAIN_HORN = Kind(
    "horn",
    (660.0, 370.0),
    109.0,
    25.0,
    2,
    1.0,
    0.2,
    4.0,
    RISE_S,
    "train_horn.wav",
    weighting="C",
    in_turn=True,
    site=TRAIN_SITE,
)
#: The rule's notes (Hz) and the levels it lets each be 25 m ahead (dB(C)).
TRAIN_NOTES_HZ = ((630.0, 690.0), (350.0, 390.0))
TRAIN_LEVELS_DB_C = (101.0, 109.0)


def sounding(times, kind: Kind = AIR_HORN, note: int | None = None) -> np.ndarray:
    """How loud the horn is at ``times`` (s), from 0 (silent) to 1 (a blast at its
    level): its blasts from the start of every period, each rising and falling over its
    rise time by a squared sine; with ``note`` (an index of its :attr:`~Kind.notes`),
    the blasts of that note alone."""
    t = np.mod(np.asarray(times, dtype=float), kind.every_s)
    out = np.zeros_like(t)
    count = len(kind.notes)
    for k in range(kind.blasts):
        if note is not None and k % count != note:
            continue
        start = k * (kind.blast_s + kind.gap_s)
        into = t - start
        edge = np.minimum(into, kind.blast_s - into)
        inside = (into >= 0.0) & (into <= kind.blast_s)
        ramp = np.sin(0.5 * np.pi * np.clip(edge / kind.rise_s, 0.0, 1.0)) ** 2
        out = np.where(inside, ramp, out)
    return out


def _partials(f: float, nyquist: float, kind: Kind):
    """The harmonics of the fundamental ``f`` (Hz) a blast sounds, each (k, its size)
    with k times ``f`` under ``nyquist``: a sawtooth's (1/k), or for a disc horn its
    tone disc's (:func:`disc`), faded out over the last tenth below ``nyquist``."""
    k = 1
    while k * f < nyquist:
        fade = min(max((nyquist - k * f) / (0.1 * nyquist), 0.0), 1.0)
        size = disc(k * f, kind) if kind.disc_hz > 0.0 else 1.0 / k
        yield k, fade * size
        k += 1


def blast(
    sample_rate: int, duration_s: float, kind: Kind = AIR_HORN, note: int | None = None
) -> np.ndarray:
    """The horns sounding without a pause, ``duration_s`` long at ``sample_rate``: at
    each of its fundamentals (those of ``note``, an index of its :attr:`~Kind.notes`,
    when given) a sawtooth (its harmonics, each 1/k of the fundamental), or for a disc
    horn its harmonics struck alike and shaped by its tone disc's resonance
    (:func:`disc`), faded out as they near the Nyquist frequency; the horns as loud as
    each other, unscaled."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    nyquist = sample_rate / 2.0
    x = np.zeros(n)
    for f in kind.hz if note is None else kind.notes[note]:
        for k, size in _partials(f, nyquist, kind):
            x += size * np.sin(2.0 * np.pi * k * f * t)
    return x


def disc(f, kind: Kind) -> np.ndarray:
    """How a disc horn's tone disc shapes its harmonic at ``f`` (Hz), from 1 at its
    resonance: a resonance of the kind's disc_hz and disc_q."""
    f = np.asarray(f, dtype=float)
    off = f / kind.disc_hz - kind.disc_hz / f
    return 1.0 / np.sqrt(1.0 + (kind.disc_q * off) ** 2)


def horn_sound(sample_rate: int, duration_s: float, kind: Kind = AIR_HORN) -> np.ndarray:
    """The horn's sound, dry and mono, ``duration_s`` long at ``sample_rate``, from the
    start of its blasts: :func:`blast` times :func:`sounding` (for notes in turn, each
    note's times its own blasts, as loud against the others as its sound power is), at
    an RMS of SOUND_RMS over whole periods of its blasts."""
    if duration_s <= 0.0:
        raise ValueError("the horn's sound needs a positive duration")
    return _made(sample_rate, duration_s, kind) * (SOUND_RMS / _rms(sample_rate, kind))


def _made(sample_rate: int, duration_s: float, kind: Kind) -> np.ndarray:
    """The horn's sound, unscaled."""
    if not kind.in_turn:
        tone = blast(sample_rate, duration_s, kind)
        return tone * sounding(np.arange(len(tone)) / sample_rate, kind)
    out = np.zeros(int(round(duration_s * sample_rate)))
    for note in range(len(kind.notes)):
        tone = blast(sample_rate, duration_s, kind, note)
        gate = sounding(np.arange(len(tone)) / sample_rate, kind, note)
        out += _note_size(sample_rate, kind, note) * tone * gate
    return out


@functools.lru_cache(maxsize=8)
def _note_size(sample_rate: int, kind: Kind, note: int) -> float:
    """How much a note's :func:`blast` is scaled by, against the others, for its mean
    square to be as its :func:`blast_power`."""
    tone = blast(sample_rate, kind.power_s, kind, note)
    power = float(np.sum(10.0 ** (blast_power(sample_rate, kind, note) / 10.0)))
    return math.sqrt(power / float(np.mean(tone**2)))


@functools.lru_cache(maxsize=8)
def _rms(sample_rate: int, kind: Kind = AIR_HORN) -> float:
    """The RMS of the unscaled sound over the kind's power_s."""
    x = _made(sample_rate, kind.power_s, kind)
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


def duty(kind: Kind = AIR_HORN, note: int | None = None) -> float:
    """The share of the time the horn sounds (``note`` alone, when given), in energy:
    the mean of :func:`sounding` squared over a period of its blasts."""
    t = np.linspace(0.0, kind.every_s, 300_001)[:-1]
    return float(np.mean(sounding(t, kind, note) ** 2))


@functools.lru_cache(maxsize=16)
def _bands(sample_rate: int, kind: Kind = AIR_HORN, note: int = 0) -> tuple[float, ...]:
    """The share of a blast's sound (of ``note``) in each octave band."""
    from acoustic_spaces.synthesis import split_bands

    x = blast(sample_rate, kind.power_s, kind, note)
    energy = np.mean(split_bands(x, sample_rate) ** 2, axis=1)
    return tuple(float(e) for e in energy / energy.sum())


def weighted(levels_db, weighting: str) -> float:
    """The level (dB) of levels per octave band, A- or C-weighted (``weighting``)."""
    from acoustic_spaces.parameters import a_weighted, c_weighted

    return {"A": a_weighted, "C": c_weighted}[weighting](levels_db)


def _site_paths(kind: Kind) -> tuple[float, float, float, float]:
    """On the kind's site: how far its microphone is from it (m), how far its image in
    the ground is (m), and how far off its axis the direct sound and the reflection
    leave it (degrees, the horn facing the microphone level)."""
    site = kind.site
    assert site is not None
    rise, depth = site.horn_m - site.microphone_m, site.horn_m + site.microphone_m
    return (
        math.hypot(kind.at_m, rise),
        math.hypot(kind.at_m, depth),
        math.degrees(math.atan2(abs(rise), kind.at_m)),
        math.degrees(math.atan2(depth, kind.at_m)),
    )


def ground_db(
    sample_rate: int, kind: Kind = AIR_HORN, note: int = 0, speed_of_sound: float = 343.0
) -> np.ndarray:
    """(8,) how much the ground of the kind's site adds to its level at its microphone
    per octave band (dB), ``note`` sounding: each harmonic's pressure added to its
    reflection's (the spherical wave's reflection factor off the site's ground, reacting
    locally, :func:`~acoustic_spaces.materials.ground.spherical_reflection`, sent as far
    off the horn's axis as it leaves it), in energy against the direct sound alone,
    averaged over the band's harmonics by their energy; 0 in a free field, and in a band
    without harmonics."""
    from acoustic_spaces.demos.siren import off_axis_db
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
    from acoustic_spaces.materials.ground import (
        PRESETS,
        spherical_reflection,
        surface_impedance,
    )

    out = np.zeros(len(OCTAVE_BANDS_HZ))
    if kind.site is None:
        return out
    pairs = [
        (k * f, float(size) ** 2)
        for f in kind.notes[note]
        for k, size in _partials(f, sample_rate / 2.0, kind)
    ]
    f = np.array([p[0] for p in pairs])
    energy = np.array([p[1] for p in pairs])
    direct, image, across, down = _site_paths(kind)
    sent = 10.0 ** ((float(off_axis_db(down)) - float(off_axis_db(across))) / 20.0)
    admittance = 1.0 / surface_impedance(
        f, PRESETS[kind.site.ground][0], speed_of_sound=speed_of_sound
    )
    wavenumber = 2.0 * np.pi * f / speed_of_sound
    depth = kind.site.horn_m + kind.site.microphone_m
    q = spherical_reflection(admittance, np.asarray(depth / image), wavenumber, np.asarray(image))
    # e^(jωt), as the engine's reflection factors.
    reflected = sent * q * (direct / image) * np.exp(-1j * wavenumber * (image - direct))
    gain = np.abs(1.0 + reflected) ** 2
    band = np.clip(np.round(np.log2(f / OCTAVE_BANDS_HZ[0])).astype(int), 0, len(out) - 1)
    for b in range(len(out)):
        inside = band == b
        if inside.any():
            share = np.sum(energy[inside] * gain[inside]) / np.sum(energy[inside])
            out[b] = 10.0 * math.log10(float(share))
    return out


def blast_power(sample_rate: int, kind: Kind = AIR_HORN, note: int = 0) -> np.ndarray:
    """(8,) the horn's sound power per octave band while it sounds ``note`` (an index
    of its :attr:`~Kind.notes`: its only one, unless they sound in turn) (dB re 1 pW):
    its level on its axis where its rule measures it, weighted as that is, less its
    directivity index and the spreading of free field, and on a site less what the
    site's ground adds there (:func:`ground_db`)."""
    from acoustic_spaces.demos.siren import directivity_index_db, off_axis_db

    bands = 10.0 * np.log10(np.array(_bands(sample_rate, kind, note)))
    distance, off, heard = kind.at_m, 0.0, bands
    if kind.site is not None:
        distance, _, across, _ = _site_paths(kind)
        off = float(off_axis_db(across))
        heard = bands + ground_db(sample_rate, kind, note)
    total = kind.on_axis_db + 20.0 * math.log10(distance) + 10.0 * math.log10(4.0 * math.pi)
    total -= directivity_index_db() + off
    return total - weighted(heard, kind.weighting) + bands


def horn_power(sample_rate: int, kind: Kind = AIR_HORN) -> np.ndarray:
    """(8,) the horn's sound power per octave band (dB re 1 pW) averaged over its blasts
    and the pauses between them, as its sound is made (:func:`horn_sound`): its
    :func:`blast_power` times :func:`duty`, note by note."""
    if not kind.in_turn:
        return blast_power(sample_rate, kind) + 10.0 * math.log10(duty(kind))
    energy = sum(
        duty(kind, note) * 10.0 ** (blast_power(sample_rate, kind, note) / 10.0)
        for note in range(len(kind.notes))
    )
    return 10.0 * np.log10(energy)


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
    "Site",
    "TRAIN_HORN",
    "TRAIN_HORN_HEIGHT_M",
    "TRAIN_LEVELS_DB_C",
    "TRAIN_NOTES_HZ",
    "TRAIN_SITE",
    "VEHICLE_DB_A",
    "blast",
    "blast_power",
    "disc",
    "duty",
    "ground_db",
    "horn_power",
    "horn_sound",
    "recorded_gain",
    "sounding",
    "sounding_rms",
    "weighted",
]
