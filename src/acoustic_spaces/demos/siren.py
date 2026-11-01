"""The sirens of the pass-by demos' police car and fire engine: their sound in each
pattern, the level and the directivity of a measured siren, and the vehicles that carry
them (:class:`EmergencyVehicle`, :mod:`acoustic_spaces.demos.passby`). Nothing of the
simulation uses it.

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
#: The sources of the street example's fire engine: the vehicle itself, tyres and
#: engine, 0.05 m above the road; its siren, on its cab's roof; and its air horn, at its
#: front, all along the same walk.
FIRE_ENGINE = "fire_engine"
FIRE_SIREN = "fire_engine_siren"
FIRE_HORN = "fire_engine_horn"
#: What the fire engine plays (``--fire-engine``), and the sources of each choice.
FIRE_PLAYS = ("truck_siren_and_horn", "truck_and_siren", "siren_and_horn", "siren")
FIRE_PARTS = (
    (FIRE_ENGINE, FIRE_SIREN, FIRE_HORN),
    (FIRE_ENGINE, FIRE_SIREN),
    (FIRE_SIREN, FIRE_HORN),
    (FIRE_SIREN,),
)
#: The most the Swiss rule lets two tones be on a vehicle (dB(A)), 7 m in front of it at
#: the loudest of 0.5 to 1.5 m up; the fire engine's siren FIRE_SIREN_HEIGHT_M up (m,
#: assumed), as in the street example; and how much louder than the police car's its
#: two tones are (dB): as loud as the rule allows, to the tenth of a dB under it (114.9
#: dB(A)), as the engine gives the level there over the street's asphalt, every 0.1 m
#: up (validation/urban_sources.py).
FIRE_SIREN_DB_A = 115.0
FIRE_SIREN_HEIGHT_M = 3.0
FIRE_SIREN_GAIN_DB = 6.0
#: The sources of the street example's car: the car itself and its horn, along one
#: walk; what it plays (``--car``), and the sources of each choice.
CAR = "car"
CAR_HORN_PART = "car_horn"
CAR_PLAYS = ("car", "car_and_horn", "horn")
CAR_PARTS = ((CAR,), (CAR, CAR_HORN_PART), (CAR_HORN_PART,))


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


class Device:
    """A warning device a vehicle carries along its walk: the example's source ``part``,
    what it is in a word (``noun``; ``key`` starts its keys in the vehicle's info),
    where it sits on the vehicle (``place``), its made sound's ``file`` and the words of
    the stills' credit for it (``loudness``)."""

    key = noun = "siren"
    #: How a recording of it is levelled, in words, when not as the made sound's RMS.
    recorded = ""
    #: Whether it sounds all along (a siren), or now and then (a horn's blasts), its
    #: sound power the average over its sounding and its pauses: then the RMS its made
    #: sound has over whole periods of them.
    steady = True
    rms = SOUND_RMS

    def __init__(self, part: str, file: str, place: str, loudness: str) -> None:
        self.part, self.file, self.place, self.loudness = part, file, place, loudness

    def pattern(self, args: argparse.Namespace) -> str:
        """How it sounds, for the arguments."""
        raise NotImplementedError

    def power(self, pattern: str, rate: int) -> np.ndarray:
        """(8,) its sound power per octave band (dB re 1 pW) sounding ``pattern``."""
        raise NotImplementedError

    def sound(self, pattern: str, rate: int, duration_s: float) -> np.ndarray:
        """Its sound, dry and mono, ``duration_s`` long at ``rate``."""
        raise NotImplementedError

    def input_gain(self, x: np.ndarray, rate: int) -> float | None:
        """The gain of ``x``, a recording of it, so that it plays as loud as its made
        sound, which has no pauses: its gated RMS brought to SOUND_RMS; None when it is
        silent."""
        from acoustic_spaces.demos.audio import gated_rms

        rms = gated_rms(x, rate)
        return SOUND_RMS / rms if rms > 0.0 else None

    def doing(self, pattern: str) -> str:
        """It sounding ``pattern``, in words: "its siren on, wail"."""
        return f"its {self.noun} on, {pattern.replace('_', '-')}"

    def made(self, pattern: str) -> str:
        """Its made sound in ``pattern``, in words: "a siren's wail"."""
        return f"a {self.noun}'s {pattern.replace('_', '-')}"

    def credit(self, pattern: str) -> str:
        """The stills' credit for its made sound."""
        return f"{self.noun}: {pattern.replace('_', '-')}, {self.loudness}"


class Siren(Device):
    """A siren: :func:`siren_sound` in the pattern of ``--siren`` (or always ``fixed``),
    with :func:`siren_power` ``gain_db`` louder."""

    def __init__(
        self,
        part: str,
        file: str,
        place: str,
        loudness: str,
        fixed: str | None = None,
        gain_db: float = 0.0,
    ) -> None:
        super().__init__(part, file, place, loudness)
        self.fixed, self.gain_db = fixed, float(gain_db)

    def pattern(self, args: argparse.Namespace) -> str:
        return self.fixed or getattr(args, "siren", None) or PATTERN

    def power(self, pattern: str, rate: int) -> np.ndarray:
        return siren_power(pattern, rate) + self.gain_db

    def sound(self, pattern: str, rate: int, duration_s: float) -> np.ndarray:
        return siren_sound(pattern, rate, duration_s)

    def credit(self, pattern: str) -> str:
        return f"siren: {pattern.replace('_', '-')}, a square wave, {self.loudness}"


class Horn(Device):
    """A horn (:mod:`acoustic_spaces.demos.horn`) in blasts: the air horn's two
    trumpets, or the ``kind`` given (:class:`~acoustic_spaces.demos.horn.Kind`)."""

    key = "horn"
    recorded = ", its blasts as loud as the made horn's, whatever its pauses"
    steady = False

    def __init__(self, part: str, file: str, place: str, loudness: str, kind=None) -> None:
        from acoustic_spaces.demos.horn import AIR_HORN
        from acoustic_spaces.demos.horn import SOUND_RMS as HORN_RMS

        super().__init__(part, file, place, loudness)
        self.kind = kind or AIR_HORN
        self.noun = self.kind.noun
        self.rms = HORN_RMS

    def pattern(self, args: argparse.Namespace) -> str:
        return "blasts"

    def power(self, pattern: str, rate: int) -> np.ndarray:
        from acoustic_spaces.demos.horn import horn_power

        return horn_power(rate, self.kind)

    def sound(self, pattern: str, rate: int, duration_s: float) -> np.ndarray:
        from acoustic_spaces.demos.horn import horn_sound

        return horn_sound(rate, duration_s, self.kind)

    def input_gain(self, x: np.ndarray, rate: int) -> float | None:
        """A recording of a horn plays its blasts as loud as the made horn's, whatever
        its pauses (:func:`~acoustic_spaces.demos.horn.recorded_gain`)."""
        from acoustic_spaces.demos.horn import recorded_gain

        return recorded_gain(x, rate, self.kind)

    def doing(self, pattern: str) -> str:
        return f"its {self.noun} in blasts"

    def made(self, pattern: str) -> str:
        article = "an" if self.noun[0] in "aeiou" else "a"
        return f"{article} {self.noun}'s blasts"

    def credit(self, pattern: str) -> str:
        hz = self.kind.hz
        tones = " and ".join(f"{f:g}" for f in hz)
        return f"{self.noun}: {tones} Hz in blasts, {self.loudness}"


class EmergencyVehicle(Vehicle):
    """A road vehicle with its warning devices on, driving past: the example's source of
    its name, the vehicle itself (``body``, tyres and engine, CNOSSOS-EU's, as
    :class:`~acoustic_spaces.demos.passby.RoadVehicle` makes it) 0.05 m above the road,
    and its ``devices`` (:class:`Device`), each a source of its own along the same walk,
    facing the way it drives, with the directivity of the siren NBS measured
    (:func:`directivity`). What it plays is the option of its name (``plays``: a choice
    to the sources it plays, the first by default); each source is simulated alone and
    they are heard together (:mod:`acoustic_spaces.demos.passby`)."""

    demo_name = "passby"
    sound_file = SIREN_FILE

    def __init__(
        self,
        body: RoadVehicle,
        devices: tuple[Device, ...],
        plays: dict[str, tuple[str, ...]],
        title: str,
        itself: str,
        plays_help: str,
        header: str,
        script: str,
    ) -> None:
        self.body, self.devices, self.plays_to = body, devices, plays
        self.name, self.noun, self.title, self.itself = body.name, body.noun, title, itself
        self.plays_help, self.header, self.script = plays_help, header, script
        self.info_file = f"{self.name}.json"
        self.steady = body.steady

    def options(self) -> list[Option]:
        return [
            Option(
                self.name,
                self.plays_help,
                next(iter(self.plays_to)),
                "choice",
                tuple(self.plays_to),
                steps=(self.sound_step, "scene", "walk"),
            ),
            *self.body.options(),
        ]

    def plays_of(self, args: argparse.Namespace) -> str:
        """What it plays: the option of its name, or its first choice."""
        return getattr(args, self.name, None) or next(iter(self.plays_to))

    def with_body(self, args: argparse.Namespace) -> bool:
        """Whether the vehicle itself plays along with its devices."""
        return self.name in self.parts(args)

    def played(self, args: argparse.Namespace) -> tuple[Device, ...]:
        """Its devices that play."""
        parts = self.parts(args)
        return tuple(d for d in self.devices if d.part in parts)

    def check(self, args: argparse.Namespace) -> None:
        if self.plays_of(args) not in self.plays_to:
            flag = "--" + self.name.replace("_", "-")
            raise SystemExit(
                f"error: {flag} is one of {', '.join(self.plays_to)}, not {self.plays_of(args)!r}"
            )
        self.body.check(args)

    def parts(self, args: argparse.Namespace) -> tuple[str, ...]:
        return self.plays_to.get(self.plays_of(args), ())

    def every_part(self) -> tuple[str, ...]:
        return (self.name, *(d.part for d in self.devices))

    def engines(self) -> tuple[RoadVehicle, ...]:
        return (self.body,)

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sounds are made of, at ``speed_kmh`` in ``scene``: what it plays, the
        speed, the length, each device's pattern and sound power, and the vehicle's
        (CNOSSOS-EU's, with its engine) unless it does not play; ``sound_power_db`` and
        ``sound_power_a_db`` of what it plays, in all."""
        from acoustic_spaces.parameters import a_weighted

        rate = int(scene.simulation.sample_rate)
        out: dict = {
            "plays": self.plays_of(args),
            "speed_kmh": round(speed_kmh, 6),
            "rate": rate,
            "duration_s": round(pass_s(car_of(scene, self), args.speed), 6),
        }
        vehicle = self.body.info(args, scene, speed_kmh) if self.with_body(args) else None
        bands = 0.0 if vehicle is None else 10.0 ** (np.asarray(vehicle["level_db"]) / 10.0)
        for device in self.played(args):
            pattern = device.pattern(args)
            level = device.power(pattern, rate)
            out[device.key] = pattern
            out[f"{device.key}_level_db"] = [round(float(v), 1) for v in level]
            out[f"{device.key}_sound_power_a_db"] = round(a_weighted(level), 2)
            bands = bands + 10.0 ** (level / 10.0)
        total = np.asarray(bands, dtype=float)
        return out | {
            "directivity_index_db": round(directivity_index_db(), 2),
            "vehicle": vehicle,
            "sound_power_db": round(10.0 * math.log10(float(np.sum(total))), 2),
            "sound_power_a_db": round(a_weighted(10.0 * np.log10(total)), 2),
        }

    def levels(self, info: dict) -> dict[str, list[float]]:
        own = {} if info["vehicle"] is None else {self.name: info["vehicle"]["level_db"]}
        return own | {
            d.part: info[f"{d.key}_level_db"] for d in self.devices if f"{d.key}_level_db" in info
        }

    def sound_files(self, args: argparse.Namespace) -> tuple[str, ...]:
        own = (self.body.sound_file,) if self.with_body(args) else ()
        return (*own, *(d.file for d in self.played(args)))

    def sounding(
        self, args: argparse.Namespace, part: str, played, rate: int, at: np.ndarray
    ) -> np.ndarray | None:
        """A device that sounds now and then (a horn) is heard around each place as
        loud as its sound plays there, faded in and out with it: its mean square there
        against its made sound's over whole periods of its blasts and pauses, which its
        sound power is the average of."""
        from acoustic_spaces.demos.passby import blocks_loudness

        device = next((d for d in self.devices if d.part == part), None)
        if device is None or device.steady:
            return None
        return blocks_loudness(np.asarray(played, dtype=float), rate, at, device.rms)

    def input_gain(self, part: str, x: np.ndarray, rate: int) -> float | None:
        device = next((d for d in self.devices if d.part == part), None)
        return super().input_gain(part, x, rate) if device is None else device.input_gain(x, rate)

    def sound_path(self, args: argparse.Namespace, part: str) -> Path:
        files = {d.part: d.file for d in self.devices}
        return args.work / self.name / files.get(part, self.body.sound_file)

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        said = [] if info["vehicle"] is None else [self.body.make(args, info["vehicle"], folder)]
        for device in self.played(args):
            pattern = info[device.key]
            x = device.sound(pattern, info["rate"], info["duration_s"])
            write_wav(folder / device.file, x, info["rate"])
            said.append(
                f"{info['duration_s']:.1f} s of {device.made(pattern)}, "
                f"{info[f'{device.key}_sound_power_a_db']:.1f} dB(A) of sound power"
            )
        return "; ".join(said)

    def edit(self, source: dict, info: dict, args: argparse.Namespace) -> None:
        """The sound power of each of its sources, and each device's directivity in a
        CSV table beside the demo's scene, facing the way it drives."""
        from acoustic_spaces.directivity import write_directivity_csv

        super().edit(source, info, args)
        if source["name"] not in {d.part for d in self.devices}:
            return
        table = args.scene.with_name(f"{args.scene.stem}_{source['name']}.csv")
        write_directivity_csv(table, directivity(), directivity_comment(self.script))
        named = Path(os.path.relpath(table, args.source.parent)).as_posix()
        source["directivity"] = {"file": named}
        source.setdefault("walk", {})["face_forward"] = True

    def feeds(self, args: argparse.Namespace, speed: str) -> dict[str, str]:
        devices = self.played(args)
        out = {}
        if self.with_body(args):
            out[self.name] = self.body.plays(args, speed)
        else:
            first = devices[0]
            alone = " alone" if len(devices) == 1 else ""
            pattern = first.pattern(args).replace("_", "-")
            out[first.part] = f"a {self.noun}'s {first.noun}{alone} {speed}{pattern}"
            devices = devices[1:]
        for device in devices:
            pattern = device.pattern(args).replace("_", "-")
            out[device.part] = f"its {device.noun} {device.place}, {pattern}"
        return out

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        return "; ".join(self.feeds(args, speed).values())

    def played_by(self, args: argparse.Namespace) -> str:
        speed = f"{round(speed_of(args, self), 4):g} km/h"
        patterns = " ".join(d.pattern(args) for d in self.devices)
        rpm = f" {self.body.rpm_of(args):g} rpm" if self.with_body(args) else ""
        return f"{self.name} {speed} {patterns} {self.plays_of(args)}{rpm}"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        devices = [d for d in self.devices if d.key in info]
        speed = kmh(info["speed_kmh"])
        facts = {"plays": info["plays"], "sound_power_a_db": info["sound_power_a_db"]}
        for device in devices:
            facts[device.key] = info[device.key]
            facts[f"{device.key}_sound_power_a_db"] = info[f"{device.key}_sound_power_a_db"]
        powers = [f"{d.noun} {info[f'{d.key}_sound_power_a_db']:.1f} dB(A)" for d in devices]
        car = info["vehicle"]
        if car is None:
            nouns = " and ".join(d.noun for d in devices) + (" alone" if len(devices) == 1 else "")
            patterns = "; ".join(info[d.key].replace("_", "-") for d in devices)
            about = f"A {self.noun}'s {nouns} at {speed} km/h, {patterns}"
            return facts | {"about": f"{about}\n{', '.join(powers)} of sound power"}
        if not devices:
            # The vehicle alone, its devices off: as the road vehicle alone says it.
            alone = self.body.facts(args, car)
            return facts | alone | {"vehicle_sound_power_a_db": car["sound_power_a_db"]}
        powers.append(f"{self.itself} {car['sound_power_a_db']:.1f} dB(A)")
        doing = "; ".join(d.doing(info[d.key]) for d in devices)
        return facts | {
            "vehicle_sound_power_a_db": car["sound_power_a_db"],
            "rpm": car["rpm"],
            "cylinders": car["cylinders"],
            "about": f"A {self.noun} at {speed} km/h, {doing}\n{', '.join(powers)} of sound power",
        }

    def credit(self, facts: dict) -> str:
        return self.credits(facts, {})

    def credits(self, facts: dict, files: dict[str, Path | None]) -> str:
        parts = self.plays_to.get(facts.get("plays", ""), self.every_part())
        if parts == (self.name,):
            # The vehicle alone, its devices off: as the road vehicle alone says it.
            return self.body.credits(facts, files)
        said = []
        if self.name in parts:
            file = files.get(self.name)
            cylinders = CYLINDER_WORDS.get(facts["cylinders"], facts["cylinders"])
            said.append(
                f"{self.itself}: {file.name}"
                if file is not None
                else f"{self.itself}: tyres and a {cylinders}-cylinder engine at "
                f"{facts['rpm']:g} rpm (CNOSSOS-EU's)"
            )
        for device in self.devices:
            if device.part in parts:
                file = files.get(device.part)
                pattern = facts.get(device.key, "")
                said.append(
                    f"{device.noun}: {file.name}" if file is not None else device.credit(pattern)
                )
        noun = self.noun[0].upper() + self.noun[1:]
        return f"{noun}'s sound, {'; '.join(said)}; faded in and out over {FADE_S:g} s"

    def texts(self) -> dict[str, str]:
        devices = self.devices
        recordings = " or ".join(
            f"{d.part}=WAV for its {d.noun} (a dry recording of one, sent with the measured "
            f"siren's directivity{d.recorded})"
            for d in devices
        )
        files = " and ".join(d.file for d in devices)
        nouns = " and ".join(f"the {d.noun}'s" for d in devices)
        return {
            "speed": f"how fast the {self.noun} drives along its walk (km/h; m/s times 3.6), "
            f"which sets how long its pass and its sound last, and the {self.itself}'s sound "
            "power; by default as its walk in the scene goes",
            "input": f"a WAV file a source of the {self.noun} plays instead of its made "
            f"sound, as {self.name}=WAV for the {self.itself} itself (a dry recording, on "
            f"board or close to it) or {recordings}, mixed down to mono, at least as long as "
            "the pass",
            "sound": f"demo/{self.name}/{self.body.sound_file} and {files}: the "
            f"{self.itself}'s sound, made to CNOSSOS-EU, and {nouns}, dry",
            "header": self.header,
        }


class PoliceCar(EmergencyVehicle):
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

    def __init__(self, script: str = "passby_demo.py") -> None:
        #: The car itself: a passenger car, its engine four cylinders at 2000 rpm.
        car = RoadVehicle(
            POLICE_CAR, "1", "a passenger car", rpm=2000.0, cylinders=4, noun="police car"
        )
        loudness = "as loud and directional as a 100 W siren NBS measured (SP 480-28, 1978)"
        super().__init__(
            car,
            (Siren(SIREN, SIREN_FILE, "on its roof", loudness),),
            dict(zip(PLAYS, ((POLICE_CAR, SIREN), (SIREN,)), strict=True)),
            "a police car with its siren on",
            "car",
            "what the police car plays: car_and_siren, the car itself (tyres and its "
            "engine, CNOSSOS-EU's passenger car at --speed, from 0.05 m above the road) "
            "and its siren on its roof, as a police car going by sounds; siren, the "
            "siren alone",
            "# The example with its police car driving past alone with its siren on, at "
            "the speed\n# of the pass-by demo, written by passby_demo.py.\n",
            script,
        )
        #: The car itself, as :class:`~acoustic_spaces.demos.passby.RoadVehicle`.
        self.car = car

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
            *super().options(),
        ]

    def pattern_of(self, args: argparse.Namespace) -> str:
        """The siren's pattern: ``--siren``, or PATTERN."""
        return self.devices[0].pattern(args)


class FireEngine(EmergencyVehicle):
    """A fire engine driving past with its siren on and its air horn in blasts: the
    street example's sources ``fire_engine``, the vehicle itself (tyres and engine,
    CNOSSOS-EU's medium heavy vehicle, two axles) 0.05 m above the road;
    ``fire_engine_siren``, the two tones of DIN 14610 on its cab's roof,
    FIRE_SIREN_GAIN_DB louder than the police car's, which makes them FIRE_SIREN_DB_A 7 m
    in front of it, the most the Swiss rule allows on a vehicle, with the siren
    FIRE_SIREN_HEIGHT_M up; and ``fire_engine_horn``, an air horn
    (:mod:`acoustic_spaces.demos.horn`) as loud as the rule for horns lets one be on its
    bench; all three along one walk. ``--fire-engine`` plays them all, or leaves out the
    horn, the vehicle itself, or both."""

    def __init__(self, script: str = "passby_demo.py") -> None:
        from acoustic_spaces.demos.horn import AT_M, HEIGHT_M, HORN_FILE, ON_AXIS_DB_A

        #: The vehicle itself: two axles, its engine six cylinders at 1500 rpm, as the
        #: street's bus's.
        truck = RoadVehicle(
            FIRE_ENGINE, "2", "a fire engine with two axles", rpm=1500.0, cylinders=6
        )
        siren = Siren(
            FIRE_SIREN,
            SIREN_FILE,
            "on its cab's roof",
            f"{FIRE_SIREN_DB_A:g} dB(A) at 7 m (the Swiss rule's most), NBS's directivity",
            fixed="two_tone",
            gain_db=FIRE_SIREN_GAIN_DB,
        )
        horn = Horn(
            FIRE_HORN,
            HORN_FILE,
            f"{HEIGHT_M:g} m up",
            f"{ON_AXIS_DB_A:g} dB(A) at {AT_M:g} m (the horn rule's most)",
        )
        super().__init__(
            truck,
            (siren, horn),
            dict(zip(FIRE_PLAYS, FIRE_PARTS, strict=True)),
            "a fire engine with its siren on and its air horn in blasts",
            "truck",
            "what the fire engine plays: truck_siren_and_horn, the truck itself (tyres and "
            "its engine, CNOSSOS-EU's medium heavy vehicle at --speed, from 0.05 m above "
            "the road), its siren's two tones on its cab's roof and its air horn's blasts "
            "1.2 m up, as a fire engine going by on a call sounds; truck_and_siren, "
            "without the horn; siren_and_horn, the siren and the horn alone; siren, the "
            "siren alone",
            "# The example with its fire engine driving past alone with its siren on and "
            "its air\n# horn in blasts, at the speed of the pass-by demo, written by "
            "passby_demo.py.\n",
            script,
        )


class HonkingCar(EmergencyVehicle):
    """The street example's car, its horn off unless ``--car`` says: the source ``car``,
    the car itself (tyres and engine, CNOSSOS-EU's passenger car, as
    :class:`~acoustic_spaces.demos.passby.RoadVehicle` makes it) 0.05 m above the road,
    as :data:`~acoustic_spaces.demos.passby.THE_CAR` is; and ``car_horn``, its pair of
    disc horns (:data:`~acoustic_spaces.demos.horn.CAR_HORN`) behind its front bumper,
    CAR_HEIGHT_M up, along the same walk, facing the way it drives, with the measured
    siren's directivity, in two short blasts every few seconds. ``--car car`` plays the
    car alone, as it plays without a horn; ``car_and_horn`` both; ``horn`` the horn
    alone."""

    def __init__(self, script: str = "passby_demo.py") -> None:
        from acoustic_spaces.demos.horn import CAR_HEIGHT_M, CAR_HORN

        #: The car itself: a passenger car, its engine four cylinders at 2000 rpm, as
        #: the car's of the roadside.
        car = RoadVehicle(CAR, "1", "a passenger car", rpm=2000.0, cylinders=4)
        hz = " and ".join(f"{f:g}" for f in CAR_HORN.hz)
        horn = Horn(
            CAR_HORN_PART,
            CAR_HORN.file,
            "behind its front bumper",
            f"{CAR_HORN.on_axis_db_a:g} dB(A) at {CAR_HORN.at_m:g} m (Hella's S77), its "
            f"tone discs at {CAR_HORN.disc_hz:g} Hz",
            kind=CAR_HORN,
        )
        super().__init__(
            car,
            (horn,),
            dict(zip(CAR_PLAYS, CAR_PARTS, strict=True)),
            "a passenger car",
            "car",
            "what the car plays: car, the car itself (tyres and its engine, CNOSSOS-EU's "
            "passenger car at --speed, from 0.05 m above the road), as a car going by "
            "sounds; car_and_horn, the car with its horn in two short blasts every "
            f"{CAR_HORN.every_s:g} s ({hz} Hz, {CAR_HORN.on_axis_db_a:g} dB(A) 2 m in "
            f"front of it, {CAR_HEIGHT_M:g} m up), as a driver warning of it sounds; "
            "horn, the horn alone",
            "# The example with its car driving past alone, at the speed and with the "
            "sound\n# power of the pass-by demo (CNOSSOS-EU; its horn as --car says), "
            "written by\n# passby_demo.py.\n",
            script,
        )


def directivity_comment(script: str) -> str:
    """The comment over the siren's directivity table."""
    return (
        "The directivity of a 100 W electronic siren measured by the National Bureau of "
        "Standards\n(SP 480-28, 1978): 0, -6 and -11 dB at 0, 50 and 90 degrees off its "
        f"axis, as at 90 behind;\nwritten by {script}"
    )


#: The police car, the fire engine and the car with its horn of the street example.
POLICE_CAR_VEHICLE = PoliceCar()
FIRE_ENGINE_VEHICLE = FireEngine()
HONKING_CAR = HonkingCar()

__all__ = [
    "AT_M",
    "CAR",
    "CAR_HORN_PART",
    "CAR_PARTS",
    "CAR_PLAYS",
    "FIRE_ENGINE",
    "FIRE_ENGINE_VEHICLE",
    "FIRE_HORN",
    "FIRE_PARTS",
    "FIRE_PLAYS",
    "FIRE_SIREN",
    "FIRE_SIREN_DB_A",
    "FIRE_SIREN_GAIN_DB",
    "FIRE_SIREN_HEIGHT_M",
    "HONKING_CAR",
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
    "Device",
    "EmergencyVehicle",
    "FireEngine",
    "HonkingCar",
    "Horn",
    "PoliceCar",
    "Siren",
    "directivity",
    "directivity_index_db",
    "fundamental_hz",
    "off_axis_db",
    "siren_power",
    "siren_sound",
]
