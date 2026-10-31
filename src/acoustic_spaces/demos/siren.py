"""The siren of the pass-by demos' police car: its sound in each pattern, the level and
the directivity of a measured siren, and the vehicle that carries it
(:mod:`acoustic_spaces.demos.passby`). Nothing of the simulation uses it.

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
"""

from __future__ import annotations

import argparse
import functools
import math
import os
from pathlib import Path

import numpy as np

from acoustic_spaces.demos import Option
from acoustic_spaces.demos.audio import write as write_wav
from acoustic_spaces.demos.passby import (
    CYLINDER_WORDS,
    FADE_S,
    RoadVehicle,
    Vehicle,
    car_of,
    kmh,
    pass_s,
    speed_of,
)

#: The sources of the street example's police car: the car itself, tyres and engine,
#: 0.05 m above the road; and its siren, on its roof, along the same walk.
POLICE_CAR = "police_car"
SIREN = "police_car_siren"
#: What the police car plays (``--police-car``): the car with its siren on, or the siren
#: alone.
PLAYS = ("car_and_siren", "siren")
#: The siren's patterns.
PATTERNS = ("wail", "yelp", "two_tone")
#: The patterns in words.
PATTERN_WORDS = {
    "wail": "wail (the pitch swept up and down slowly)",
    "yelp": "yelp (the pitch swept up and down fast)",
    "two_tone": "two-tone (two pitches a fourth apart, in turn)",
}
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
#: The pattern the police car plays unless ``--siren`` says.
PATTERN = "wail"
#: How many seconds of each pattern its sound power is taken from: whole sweeps or
#: pairs of tones of all three.
POWER_S = 15.0
#: The RMS of :func:`siren_sound`'s output (full scale): the level is the source's
#: ``level_db``, not the file's.
SOUND_RMS = 0.1
SIREN_FILE = "siren.wav"
POLICE_CAR_INFO = "police_car.json"


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


class PoliceCar(Vehicle):
    """A police car driving past with its siren on: the street example's sources
    ``police_car``, the car itself (tyres and engine, CNOSSOS-EU's passenger car, as
    :class:`~acoustic_spaces.demos.passby.RoadVehicle` makes it) 0.05 m above the road,
    and ``police_car_siren`` along the same walk at the height of a siren on its roof,
    facing the way it drives, playing :func:`siren_sound` in ``--siren``'s pattern with
    :func:`siren_power` and the measured siren's directivity; ``--police-car siren``
    plays the siren alone. At 50 km/h the car's sound power is some 32 dB(A) under the
    wail's, but the siren sends next to nothing under its fundamental (a square wave has
    its odd harmonics alone): in the octaves of 63 to 250 Hz the police car's sound is
    the car's, the rumble of it going by (:func:`siren_power` gives those octaves 44 to
    55 dB under the siren's whole power)."""

    name, demo_name = POLICE_CAR, "passby"
    title = "a police car with its siren on"
    sound_file, info_file = SIREN_FILE, POLICE_CAR_INFO
    #: The vehicle in a word, for the demo's words.
    noun = "police car"

    def __init__(self, script: str = "passby_demo.py") -> None:
        self.script = script
        #: The car itself: a passenger car, its engine four cylinders at 2000 rpm.
        self.car = RoadVehicle(
            POLICE_CAR, "1", "a passenger car", rpm=2000.0, cylinders=4, noun=self.noun
        )
        self.steady = self.car.steady

    def options(self) -> list[Option]:
        return [
            Option(
                "siren",
                "the police car's siren: "
                + "; ".join(PATTERN_WORDS[p] for p in PATTERNS)
                + f". Wail and yelp sweep {LOW_HZ:g} to {HIGH_HZ:g} Hz "
                f"{WAIL_PER_MINUTE:g} and {YELP_PER_MINUTE:g} times a minute (California's "
                f"rule), the two tones are {TWO_TONE_HZ[0]:g} and {TWO_TONE_HZ[1]:.1f} Hz, "
                f"{TWO_TONE_PERIOD_S:g} s the pair (the Swiss rule, DIN 14610's values)",
                PATTERN,
                "choice",
                PATTERNS,
                steps=(self.sound_step, "scene", "walk"),
            ),
            Option(
                "police_car",
                "what the police car plays: car_and_siren, the car itself (tyres and its "
                "engine, CNOSSOS-EU's passenger car at --speed, from 0.05 m above the road) "
                "and its siren on its roof, as a police car going by sounds; siren, the "
                "siren alone",
                PLAYS[0],
                "choice",
                PLAYS,
                steps=(self.sound_step, "scene", "walk"),
            ),
            *self.car.options(),
        ]

    def pattern_of(self, args: argparse.Namespace) -> str:
        """The siren's pattern: ``--siren``, or PATTERN."""
        return getattr(args, "siren", None) or PATTERN

    def plays_of(self, args: argparse.Namespace) -> str:
        """What it plays: ``--police-car``, or the car with its siren."""
        return getattr(args, "police_car", None) or PLAYS[0]

    def with_car(self, args: argparse.Namespace) -> bool:
        """Whether the car itself plays along with its siren."""
        return self.plays_of(args) == "car_and_siren"

    def check(self, args: argparse.Namespace) -> None:
        if self.plays_of(args) not in PLAYS:
            raise SystemExit(
                f"error: --police-car is one of {', '.join(PLAYS)}, not {self.plays_of(args)!r}"
            )
        self.car.check(args)

    def parts(self, args: argparse.Namespace) -> tuple[str, ...]:
        return (POLICE_CAR, SIREN) if self.with_car(args) else (SIREN,)

    def every_part(self) -> tuple[str, ...]:
        return (POLICE_CAR, SIREN)

    def engines(self) -> tuple[RoadVehicle, ...]:
        return (self.car,)

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sounds are made of, at ``speed_kmh`` in ``scene``: the pattern, the
        speed, the length and the siren's sound power, and the car's (CNOSSOS-EU's, with
        its engine), unless the siren plays alone; ``sound_power_db`` and
        ``sound_power_a_db`` of what it plays, in all."""
        from acoustic_spaces.parameters import a_weighted

        pattern = self.pattern_of(args)
        rate = int(scene.simulation.sample_rate)
        level = siren_power(pattern, rate)
        car = self.car.info(args, scene, speed_kmh) if self.with_car(args) else None
        bands = 10.0 ** (level / 10.0)
        if car is not None:
            bands = bands + 10.0 ** (np.asarray(car["level_db"]) / 10.0)
        return {
            "plays": self.plays_of(args),
            "siren": pattern,
            "speed_kmh": round(speed_kmh, 6),
            "rate": rate,
            "duration_s": round(pass_s(car_of(scene, self), args.speed), 6),
            "siren_level_db": [round(float(v), 1) for v in level],
            "siren_sound_power_a_db": round(a_weighted(level), 2),
            "directivity_index_db": round(directivity_index_db(), 2),
            "car": car,
            "sound_power_db": round(10.0 * math.log10(float(np.sum(bands))), 2),
            "sound_power_a_db": round(a_weighted(10.0 * np.log10(bands)), 2),
        }

    def levels(self, info: dict) -> dict[str, list[float]]:
        siren = {SIREN: info["siren_level_db"]}
        return siren if info["car"] is None else {POLICE_CAR: info["car"]["level_db"], **siren}

    def sound_files(self, args: argparse.Namespace) -> tuple[str, ...]:
        car = (self.car.sound_file,) if self.with_car(args) else ()
        return (*car, SIREN_FILE)

    def sound_path(self, args: argparse.Namespace, part: str) -> Path:
        file = self.car.sound_file if part == POLICE_CAR else SIREN_FILE
        return args.work / self.name / file

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        said = [] if info["car"] is None else [self.car.make(args, info["car"], folder)]
        x = siren_sound(info["siren"], info["rate"], info["duration_s"])
        write_wav(folder / SIREN_FILE, x, info["rate"])
        said.append(
            f"{info['duration_s']:.1f} s of a siren's {info['siren'].replace('_', '-')}, "
            f"{info['siren_sound_power_a_db']:.1f} dB(A) of sound power"
        )
        return "; ".join(said)

    def edit(self, source: dict, info: dict, args: argparse.Namespace) -> None:
        """The sound power of each of its sources, and the siren's directivity in a CSV
        table beside the demo's scene, facing the way it drives."""
        from acoustic_spaces.directivity import write_directivity_csv

        super().edit(source, info, args)
        if source["name"] != SIREN:
            return
        table = args.scene.with_name(f"{args.scene.stem}_{SIREN}.csv")
        write_directivity_csv(table, directivity(), directivity_comment(self.script))
        named = Path(os.path.relpath(table, args.source.parent)).as_posix()
        source["directivity"] = {"file": named}
        source.setdefault("walk", {})["face_forward"] = True

    def feeds(self, args: argparse.Namespace, speed: str) -> dict[str, str]:
        pattern = self.pattern_of(args).replace("_", "-")
        if not self.with_car(args):
            return {SIREN: f"a police car's siren alone {speed}{pattern}"}
        return {POLICE_CAR: self.car.plays(args, speed), SIREN: f"its siren on its roof, {pattern}"}

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        return "; ".join(self.feeds(args, speed).values())

    def played_by(self, args: argparse.Namespace) -> str:
        speed = f"{round(speed_of(args, self), 4):g} km/h"
        car = f" {self.car.rpm_of(args):g} rpm" if self.with_car(args) else ""
        return f"{self.name} {speed} {self.pattern_of(args)} {self.plays_of(args)}{car}"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        pattern = info["siren"].replace("_", "-")
        siren = f"siren {info['siren_sound_power_a_db']:.1f} dB(A)"
        facts = {
            "siren": info["siren"],
            "plays": info["plays"],
            "sound_power_a_db": info["sound_power_a_db"],
            "siren_sound_power_a_db": info["siren_sound_power_a_db"],
        }
        if info["car"] is None:
            about = f"A police car's siren alone at {kmh(info['speed_kmh'])} km/h, {pattern}"
            return facts | {"about": f"{about}\n{siren} of sound power"}
        car = info["car"]
        return facts | {
            "car_sound_power_a_db": car["sound_power_a_db"],
            "rpm": car["rpm"],
            "cylinders": car["cylinders"],
            "about": f"A police car at {kmh(info['speed_kmh'])} km/h, its siren on {pattern}\n"
            f"{siren}, car {car['sound_power_a_db']:.1f} dB(A) of sound power",
        }

    def credit(self, facts: dict) -> str:
        return self.credits(facts, {})

    def credits(self, facts: dict, files: dict[str, Path | None]) -> str:
        said = []
        if facts.get("plays", PLAYS[0]) == "car_and_siren":
            file = files.get(POLICE_CAR)
            cylinders = CYLINDER_WORDS.get(facts["cylinders"], facts["cylinders"])
            said.append(
                f"car: {file.name}"
                if file is not None
                else f"car: tyres and a {cylinders}-cylinder engine at {facts['rpm']:g} rpm "
                "(CNOSSOS-EU's)"
            )
        file = files.get(SIREN)
        said.append(
            f"siren: {file.name}"
            if file is not None
            else f"siren: {facts['siren'].replace('_', '-')}, a square wave, as loud and "
            "directional as a 100 W siren NBS measured (SP 480-28, 1978)"
        )
        return f"Police car's sound, {'; '.join(said)}; faded in and out over {FADE_S:g} s"

    def texts(self) -> dict[str, str]:
        return {
            "speed": "how fast the police car drives along its walk (km/h; m/s times 3.6), "
            "which sets how long its pass and its sound last, and the car's sound power; by "
            "default as its walk in the scene goes",
            "input": "a WAV file a source of the police car plays instead of its made sound, "
            f"as {POLICE_CAR}=WAV for the car itself (a dry recording, on board or close to "
            f"it) or {SIREN}=WAV for its siren (a dry recording of a siren, sent with the "
            "measured siren's directivity), mixed down to mono, at least as long as the pass",
            "sound": f"demo/{self.name}/{self.car.sound_file} and {SIREN_FILE}: the car's "
            "sound, made to CNOSSOS-EU, and the siren's, dry",
            "header": "# The example with its police car driving past alone with its siren "
            "on, at the speed\n# of the pass-by demo, written by passby_demo.py.\n",
        }


def directivity_comment(script: str) -> str:
    """The comment over the siren's directivity table."""
    return (
        "The directivity of a 100 W electronic siren measured by the National Bureau of "
        "Standards\n(SP 480-28, 1978): 0, -6 and -11 dB at 0, 50 and 90 degrees off its "
        f"axis, as at 90 behind;\nwritten by {script}"
    )


#: The police car of the street example.
POLICE_CAR_VEHICLE = PoliceCar()

__all__ = [
    "AT_M",
    "HIGH_HZ",
    "LOW_HZ",
    "ON_AXIS_DB_A",
    "PATTERNS",
    "PLAYS",
    "POLICE_CAR",
    "POLICE_CAR_VEHICLE",
    "POLAR_DB",
    "POLAR_DEG",
    "SIREN",
    "SOUND_RMS",
    "TWO_TONE_HZ",
    "TWO_TONE_PERIOD_S",
    "WAIL_PER_MINUTE",
    "YELP_PER_MINUTE",
    "PoliceCar",
    "directivity",
    "directivity_index_db",
    "fundamental_hz",
    "off_axis_db",
    "siren_power",
    "siren_sound",
]
