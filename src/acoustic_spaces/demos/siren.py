"""The sirens of a scene's sources of a siren model (``model: {method: siren}``, the
street example's police car and fire engine): their sound in each pattern, and the level
and the directivity of a measured siren. Nothing of the simulation uses it.

The patterns are the rules':

- **wail** and **yelp**: the pitch swept up and down WAIL_PER_MINUTE and YELP_PER_MINUTE
  times a minute, within the 10 to 30 and the 150 to 250 that California's rule for
  emergency vehicles' sirens asks (California Code of Regulations, title 13, section
  1028), between the fundamentals LOW_HZ and HIGH_HZ of the electronic sirens the
  National Bureau of Standards measured (NBS Special Publication 480-28, Fisher, Toth,
  Blomquist and Forrer, 1978, appendix B: 500 to 1500 Hz);
- **two_tone**: the European two tones, a fourth apart (3:4), TWO_TONE_PERIOD_S for the
  pair, each half of it, within the Swiss rule for priority vehicles' two-tone horns
  (Verordnung über die technischen Anforderungen an Strassenfahrzeuge, SR 741.41, Annex
  11: base frequencies 360 to 630 Hz, ratio 3:4, a cycle of 2.5 to 3.5 s, the values of
  DIN 14610).

Each is a square wave at its fundamental, as the Class D amplifier of an electronic
siren drives its loudspeaker (NBS SP 480-28, appendix B), its odd harmonics up to the
Nyquist frequency. The loudspeaker is taken to radiate them alike (its response is not
in the report), and as loud in every pattern: one drive, the same sound power.

The level and the directivity are those NBS measured on a commercial 100 W electronic
siren in an anechoic room (section 3.1): 3 m away, its yelp averaged over 20 s on an
energy basis, 118 dB(A) on its axis, 112 dB(A) 50 degrees off it and 107 dB(A) at 90
degrees, falling smoothly. The directivity goes through those three, in dB linearly
with the angle off the axis, the same round the axis (NBS measured the horizontal
plane only) and behind as at 90 degrees (NBS measured no further): both assumptions.
The siren's sound power follows from the yelp's level on its axis and that
directivity.

The fire engine's siren plays the two tones FIRE_SIREN_GAIN_DB louder than the police
car's, as loud as the Swiss rule lets a two-tone horn be on a vehicle (FIRE_SIREN_DB_A 7
m in front of it, 0.5 to 1.5 m up), with the same directivity; its air horn is
:mod:`acoustic_spaces.demos.horn`'s.
"""

from __future__ import annotations

import functools
import math

import numpy as np

#: The siren's patterns.
PATTERNS = ("wail", "yelp", "two_tone")
#: How many times a minute the wail and the yelp sweep up and down (California: 10 to 30
#: and 150 to 250).
WAIL_PER_MINUTE = 12.0
YELP_PER_MINUTE = 180.0
#: The fundamental's range of a wail and a yelp (Hz; NBS SP 480-28, appendix B).
LOW_HZ, HIGH_HZ = 500.0, 1500.0
#: The two tones (Hz): the lower, and the upper a fourth (4:3) over it; the time the pair
#: takes (s), each tone half of it.
TWO_TONE_HZ = (440.0, 440.0 * 4.0 / 3.0)
TWO_TONE_PERIOD_S = 3.0
#: The measured siren's directivity: angles off its axis (degrees) and its level there
#: (dB, re the axis); 180 degrees as 90 is assumed.
POLAR_DEG = (0.0, 50.0, 90.0, 180.0)
POLAR_DB = (0.0, -6.0, -11.0, -11.0)
#: Its yelp's level on its axis (dB(A), energy average) and how far away (m).
ON_AXIS_DB_A = 118.0
AT_M = 3.0
#: How many seconds of each pattern its sound power is taken from: whole sweeps or
#: pairs of tones of all three.
POWER_S = 15.0
#: The RMS of :func:`siren_sound`'s output (full scale): the level is the source's
#: ``level_db``, not the file's.
SOUND_RMS = 0.1
#: The most the Swiss rule lets two tones be on a vehicle (dB(A)), 7 m in front of it at
#: the loudest of 0.5 to 1.5 m up; the fire engine's siren FIRE_SIREN_HEIGHT_M up (m,
#: assumed), as in the street example; and how much louder than the police car's its
#: two tones are (dB): as loud as the rule allows, to the tenth of a dB under it (114.9
#: dB(A)), as the engine gives the level 7 m in front of the siren, taken at the
#: vehicle's front, over the street's asphalt, every 0.1 m up
#: (validation/urban_sources.py).
FIRE_SIREN_DB_A = 115.0
FIRE_SIREN_HEIGHT_M = 3.0
FIRE_SIREN_GAIN_DB = 6.0


def fundamental_hz(pattern: str, times: np.ndarray) -> np.ndarray:
    """The siren's fundamental (Hz) at ``times`` (s) in ``pattern``: a wail or a yelp
    from LOW_HZ up to HIGH_HZ and down again, in straight lines; the two tones in turn,
    the lower first."""
    t = np.asarray(times, dtype=float)
    if pattern in ("wail", "yelp"):
        period = 60.0 / (WAIL_PER_MINUTE if pattern == "wail" else YELP_PER_MINUTE)
        phase = np.mod(t / period, 1.0)
        return LOW_HZ + (HIGH_HZ - LOW_HZ) * (1.0 - np.abs(2.0 * phase - 1.0))
    if pattern == "two_tone":
        upper = np.mod(t / TWO_TONE_PERIOD_S, 1.0) >= 0.5
        return np.where(upper, TWO_TONE_HZ[1], TWO_TONE_HZ[0])
    raise ValueError(f"a siren's pattern is one of {', '.join(PATTERNS)}, not {pattern!r}")


def siren_sound(pattern: str, sample_rate: int, duration_s: float) -> np.ndarray:
    """The siren's sound, dry and mono, ``duration_s`` long at ``sample_rate``, from the
    start of its pattern: a square wave at :func:`fundamental_hz` (its odd harmonics,
    each 1/k of the fundamental, faded out as it nears the Nyquist frequency), at an RMS
    of SOUND_RMS."""
    if duration_s <= 0.0:
        raise ValueError("the siren's sound needs a positive duration")
    n = int(round(duration_s * sample_rate))
    f = fundamental_hz(pattern, np.arange(n) / sample_rate)
    phase = 2.0 * np.pi * (np.cumsum(f) - f[0]) / sample_rate
    nyquist = sample_rate / 2.0
    x = np.zeros(n)
    k = 1
    while k * float(f.min()) < nyquist:
        # Each harmonic fades out over the last tenth below the Nyquist frequency.
        fade = np.clip((nyquist - k * f) / (0.1 * nyquist), 0.0, 1.0)
        x += fade * np.sin(k * phase) / k
        k += 2
    return x * (SOUND_RMS / float(np.sqrt(np.mean(x**2))))


def off_axis_db(angles_deg) -> np.ndarray:
    """The siren's level off its axis (dB re the axis) at ``angles_deg`` from it."""
    return np.interp(np.asarray(angles_deg, dtype=float), POLAR_DEG, POLAR_DB)


def directivity():
    """The siren's directivity (:class:`~acoustic_spaces.directivity.Balloon`, octave
    bands, alike in all): :func:`off_axis_db` round its axis, forward."""
    from acoustic_spaces.directivity import GRID_STEP_DEG, Balloon, _from_angles, _grid
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

    elevation, azimuth = _grid(GRID_STEP_DEG)
    ways = _from_angles(azimuth[None, :], elevation[:, None])
    off = np.degrees(np.arccos(np.clip(ways[..., 0], -1.0, 1.0)))
    energy = 10.0 ** (off_axis_db(off) / 10.0)
    table = np.repeat(energy[..., None], len(OCTAVE_BANDS_HZ), axis=2)
    return Balloon.from_table(table, "siren")


def directivity_index_db() -> float:
    """How much louder the siren is on its axis than a source of its sound power that
    sends alike every way (dB)."""
    return float(10.0 * np.log10(directivity().table[:, :, 0].max()))


@functools.lru_cache(maxsize=8)
def _bands(pattern: str, rate: int) -> tuple[float, ...]:
    """The share of the siren's sound in each octave band, over POWER_S of ``pattern``."""
    from acoustic_spaces.synthesis import split_bands

    x = siren_sound(pattern, rate, POWER_S)
    energy = np.mean(split_bands(x, rate) ** 2, axis=1)
    return tuple(float(e) for e in energy / energy.sum())


def siren_power(pattern: str, sample_rate: int) -> np.ndarray:
    """(8,) the siren's sound power per octave band (dB re 1 pW) in ``pattern``: the
    yelp's, ON_AXIS_DB_A at AT_M on its axis less its directivity index and the spreading
    of free field; every pattern with the yelp's sound power in all, spread over its
    own bands."""
    from acoustic_spaces.parameters import a_weighted

    yelp = 10.0 * np.log10(np.array(_bands("yelp", sample_rate)))
    yelp_a = ON_AXIS_DB_A + 20.0 * math.log10(AT_M) + 10.0 * math.log10(4.0 * math.pi)
    yelp_a -= directivity_index_db()
    total = yelp_a - a_weighted(yelp)
    return total + 10.0 * np.log10(np.array(_bands(pattern, sample_rate)))


#: :func:`siren_power` of each pattern at 48 kHz (dB re 1 pW, 63 Hz to 8 kHz, to 0.01
#: dB), which a scene's siren model takes (:class:`~acoustic_spaces.scene.SirenModel`):
#: its sound made and split into octave bands takes seconds, a scene is read at once
#: (tests/test_sources.py checks the table against the function).
POWER_DB: dict[str, tuple[float, ...]] = {
    "wail": (76.13, 76.12, 81.38, 123.25, 128.63, 120.46, 119.97, 118.77),
    "yelp": (76.14, 76.14, 81.67, 123.25, 128.63, 120.46, 119.97, 118.77),
    "two_tone": (77.97, 78.42, 86.97, 130.13, 115.56, 119.19, 116.27, 115.21),
}


def directivity_comment(script: str) -> str:
    """The comment over the siren's directivity table."""
    return (
        "The directivity of a 100 W electronic siren measured by the National Bureau of "
        "Standards\n(SP 480-28, 1978): 0, -6 and -11 dB at 0, 50 and 90 degrees off its "
        f"axis, as at 90 behind;\nwritten by {script}"
    )


__all__ = [
    "AT_M",
    "FIRE_SIREN_DB_A",
    "FIRE_SIREN_GAIN_DB",
    "FIRE_SIREN_HEIGHT_M",
    "HIGH_HZ",
    "LOW_HZ",
    "ON_AXIS_DB_A",
    "PATTERNS",
    "POWER_DB",
    "POLAR_DB",
    "POLAR_DEG",
    "SOUND_RMS",
    "TWO_TONE_HZ",
    "TWO_TONE_PERIOD_S",
    "WAIL_PER_MINUTE",
    "YELP_PER_MINUTE",
    "directivity",
    "directivity_index_db",
    "fundamental_hz",
    "off_axis_db",
    "siren_power",
    "siren_sound",
]
