"""The helicopter of the flyover demo (:mod:`acoustic_spaces.demos.passby`): the sound its
rotors send each way, and the sound power and directivity that go with it in the scene.

The helicopter is the example's source ``helicopter``, a point at the main rotor's hub
with a trajectory at one height and one speed, facing the way it flies (``face_forward``). Its
rotors are trimmed for level flight (:func:`acoustic_spaces.rotorcraft.trim`) at its
airspeed: the trajectory's speed over the ground less the wind at its height, from the scene's
atmosphere (its own weather, not its zones'), in the scene's air. The trim faces the
way it moves through the air; facing along the trajectory instead, the helicopter does not
turn into a wind from the side (it would fly crabwise), so a scene's wind is best along
the trajectory or none. What the rotors send (:func:`acoustic_spaces.rotorcraft.emission`)
plays down each early path; the scene's source gets the sound power and the directivity
of that emission (:meth:`~acoustic_spaces.emission.Emission.sound_power_db`,
:meth:`~acoustic_spaces.emission.Emission.directivity`), for the levels the simulation
reports and the late part. Nothing of the simulation uses this module.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
from pathlib import Path

import numpy as np

from acoustic_spaces.demos import Option, pipeline
from acoustic_spaces.demos.audio import write as write_wav
from acoustic_spaces.demos.passby import FADE_S, Vehicle, car_of, kmh, pass_s

#: The source of the example that flies over.
HELICOPTER = "helicopter"
#: How far apart the helicopter's places are simulated by default (m): 150 m up, places
#: this far apart follow the exact field of a flyover more closely than a car's places 1
#: m apart do the car's (``validation/doppler.py``).
STEP_M = 4.0
#: How far the trajectory's points may differ in height and still be level flight (m).
LEVEL_M = 0.5
#: The noise's seed, so that every step makes the same sound.
SEED = 0
#: Seconds of the rotors' sound its sound power and directivity are taken from: the
#: tones repeat, and the noise has its bands' power exactly in any length.
POWER_S = 1.0
#: The helicopter's sound sent straight down, dry: what the sound step writes to hear.
BELOW_FILE = "below.wav"
INFO_FILE = "helicopter.json"


def airspeed(source, scene, speed_m_s: float) -> float:
    """The airspeed of ``source`` (the helicopter, a scene source with a trajectory) in
    ``scene`` flying at ``speed_m_s`` over the ground along its first leg: that, less
    the wind of the scene's own atmosphere at its height, both as vectors."""
    from acoustic_spaces.walkthrough import heading

    ground = speed_m_s * heading(source)
    atmosphere = scene.atmosphere
    wind = np.zeros(3)
    if atmosphere is not None:
        speed = float(atmosphere.wind_speed(np.array([source.position[2]]))[0])
        way = math.radians(atmosphere.wind.direction_deg) if atmosphere.wind else 0.0
        wind = speed * np.array([math.cos(way), math.sin(way), 0.0])
    return float(np.linalg.norm(ground - wind))


#: Its broadband noise's models, in words.
BROADBAND = {
    "bpm": "Brooks, Pope and Marcolini's airfoil self-noise of each blade section",
    "pegg": "Pegg's method (NASA TM-80200)",
}


#: The rate the sound power and the directivity of a flight are found at (Hz): every
#: octave band lies under its half.
POWER_RATE = 48_000


def helicopters() -> dict:
    """The helicopters a scene's rotorcraft model flies, by its name of them
    (:class:`acoustic_spaces.rotorcraft.Helicopter`)."""
    from acoustic_spaces import rotorcraft

    return {"as350_b3": rotorcraft.AS350_B3}


@functools.lru_cache(maxsize=4)
def _rotors(
    airspeed_m_s: float,
    temperature_c: float,
    pressure_kpa: float,
    rate: int,
    broadband: str,
    helicopter: str = "as350_b3",
):
    """What of ``helicopter``'s sound does not depend on how long it lasts, found once for
    all the lengths :func:`_sent` makes: its trim, its two rotors' tones (one period in
    each direction) and, by ``broadband`` "bpm", its broadband noise's bands
    (:func:`acoustic_spaces.rotorcraft.bpm_powers`)."""
    from acoustic_spaces import rotorcraft

    flier = dataclasses.replace(helicopters()[helicopter], broadband=broadband)
    trim = rotorcraft.trim(flier, airspeed_m_s, temperature_c, pressure_kpa)
    tones = tuple(
        rotorcraft.tones(trim, which, samples=0, sample_rate=rate) for which in ("main", "tail")
    )
    bands = rotorcraft.bpm_powers(trim) if broadband == "bpm" else None
    return trim, tones, bands


@functools.lru_cache(maxsize=4)
def _sent(
    airspeed_m_s: float,
    temperature_c: float,
    pressure_kpa: float,
    rate: int,
    n: int,
    broadband: str,
    helicopter: str = "as350_b3",
):
    """``helicopter``'s trim and the ``n`` samples it sends at ``rate``, its broadband
    noise by ``broadband``: :func:`acoustic_spaces.rotorcraft.emission`, sample for
    sample, from :func:`_rotors` and ``n`` samples of noise."""
    from acoustic_spaces import rotorcraft
    from acoustic_spaces.emission import Sum

    trim, tones, bands = _rotors(
        airspeed_m_s, temperature_c, pressure_kpa, rate, broadband, helicopter
    )
    noise = rotorcraft.broadband(trim, samples=n, sample_rate=rate, seed=SEED, bands=bands)
    return trim, Sum((*(dataclasses.replace(tone, samples=n) for tone in tones), noise))


@functools.lru_cache(maxsize=16)
def flight_sound(
    helicopter: str,
    broadband: str,
    airspeed_m_s: float,
    temperature_c: float,
    pressure_kpa: float,
):
    """The sound power per octave band (dB re 1 pW, a tuple) and the directivity (a
    :class:`~acoustic_spaces.directivity.Balloon`, octave bands) of what ``helicopter``'s
    rotors send in level flight at ``airspeed_m_s`` in air at ``temperature_c`` and
    ``pressure_kpa``, its broadband noise by ``broadband``: of POWER_S of it at
    POWER_RATE (:meth:`~acoustic_spaces.emission.Emission.sound_power_db`,
    :meth:`~acoustic_spaces.emission.Emission.directivity`), what a scene's rotorcraft
    model gives its source."""
    rate = POWER_RATE
    trim, sent = _sent(
        airspeed_m_s,
        temperature_c,
        pressure_kpa,
        rate,
        round(POWER_S * rate),
        broadband,
        helicopter,
    )
    level = sent.sound_power_db(trim.density, trim.sound_speed)
    name = f"{helicopters()[helicopter].name}'s rotors at {airspeed_m_s:g} m/s through the air"
    return tuple(round(float(v), 6) for v in level), sent.directivity(name=name)


def flight_emission(source, rate: int, temperature_c: float, pressure_kpa: float):
    """What ``source`` (a scene source with a rotorcraft model, its trajectory as the
    run plays it) sends each way on the run's clock at ``rate``: silent until its
    trajectory sets off, then its rotors' sound along its pass, faded in and out over
    FADE_S, as loud as a made sound is for its sound power (:func:`emission_gain`)."""
    from acoustic_spaces.demos.models import FADE_S, pass_times
    from acoustic_spaces.emission import Delayed, Faded

    model = source.model
    start, end = pass_times(source)
    n = int(round((end - start) * rate))
    trim, sent = _sent(
        source.model_speed_m_s(),
        temperature_c,
        pressure_kpa,
        int(rate),
        n,
        model.broadband,
        model.helicopter,
    )
    faded = Faded(sent, min(int(FADE_S * rate), n // 2), n)
    gain = emission_gain(source.level_db, trim.density, trim.sound_speed)
    return Delayed(faded.scaled(gain), int(round(start * rate)))


def emission_gain(power_db, density: float, sound_speed: float) -> float:
    """The gain that brings an emission in pascals 1 m away whose sound power per octave
    band is ``power_db`` (dB re 1 pW) to a made sound's units: the mean square over the
    sphere of what it sends in those bands to SOUND_RMS squared, as a made sound's RMS
    is its sound power (:mod:`acoustic_spaces.demos.models`)."""
    from acoustic_spaces.demos.models import SOUND_RMS

    watts = float(np.sum(10.0 ** (np.asarray(power_db, dtype=float) / 10.0))) * 1e-12
    mean_square = watts * density * sound_speed / (4.0 * math.pi)
    return SOUND_RMS / math.sqrt(mean_square)


def below_sound(source, rate: int, temperature_c: float, pressure_kpa: float) -> np.ndarray:
    """What ``source`` (as :func:`flight_emission`) sends straight down, dry, on the
    run's clock: its feed, to hear."""
    sent = flight_emission(source, rate, temperature_c, pressure_kpa)
    n = sent.length
    return sent.sent(0, n, np.tile([0.0, 0.0, -1.0], (n, 1)))


class Helicopter(Vehicle):
    """The AS350 B3 (:data:`acoustic_spaces.rotorcraft.AS350_B3`) flying over, in the demo
    that ``script`` runs."""

    name, demo_name = HELICOPTER, "flyover"

    def __init__(self, script: str = "flyover_demo.py") -> None:
        self.script = script

    go, goes, going, way = "fly", "flies", "flying over", "its flight path"
    noun = HELICOPTER
    steady = "its rotors are trimmed for one steady speed"
    step_m = STEP_M
    sound_file, info_file = BELOW_FILE, INFO_FILE
    sound_step = HELICOPTER
    title = "an Airbus Helicopters AS350 B3"

    def options(self) -> list[Option]:
        from acoustic_spaces import rotorcraft

        return [
            Option(
                "broadband",
                "how the rotors' broadband noise is found: pegg, Pegg's fit to measured "
                "rotors' noise (NASA TM-80200), or bpm, each blade section's own noise in "
                "the flow past it (Brooks, Pope and Marcolini's airfoil self-noise) summed "
                "round the turn; in the certification's flyover pegg comes out over the "
                "helicopter's certified level and bpm under it",
                rotorcraft.AS350_B3.broadband,
                "choice",
                rotorcraft.BROADBAND_MODELS,
                steps=(self.name, "scene", "walk"),
            )
        ]

    def check(self, args: argparse.Namespace) -> None:
        return None

    def problem(self, source) -> str | None:
        """A helicopter flies level: its trajectory keeps one height."""
        from acoustic_spaces.walkthrough import knots

        heights = knots(source).positions[:, 2]
        if float(np.ptp(heights)) > LEVEL_M:
            return (
                f"the {self.name} flies level, its rotors trimmed for it, but its trajectory goes "
                f"from {heights.min():.1f} to {heights.max():.1f} m up: give its points one "
                "height"
            )
        return None

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sound is made of: its speed over the ground and through the air,
        the air, the length, and its rotors' sound power."""
        from acoustic_spaces import rotorcraft
        from acoustic_spaces.parameters import a_weighted

        air = scene.air
        flying = round(airspeed(car_of(scene, self), scene, speed_kmh / 3.6), 4)
        rate = int(scene.simulation.sample_rate)
        broadband = str(args.broadband)
        trim, sent = _sent(
            flying, air.temperature_c, air.pressure_kpa, rate, round(POWER_S * rate), broadband
        )
        level = sent.sound_power_db(trim.density, trim.sound_speed)
        return {
            "helicopter": rotorcraft.AS350_B3.name,
            "mass_kg": rotorcraft.AS350_B3.mass_kg,
            "broadband": broadband,
            "speed_kmh": round(speed_kmh, 6),
            "airspeed_m_s": flying,
            "temperature_c": air.temperature_c,
            "pressure_kpa": air.pressure_kpa,
            "rate": rate,
            "duration_s": round(pass_s(car_of(scene, self), args.speed), 6),
            "seed": SEED,
            "level_db": [round(float(v), 1) for v in level],
            "sound_power_db": round(10.0 * math.log10(float(np.sum(10.0 ** (level / 10.0)))), 2),
            "sound_power_a_db": round(a_weighted(level), 2),
            "power_kw": round(trim.power_w / 1000.0, 1),
        }

    def emission(self, info: dict, seconds: float):
        """What its rotors send for ``seconds``, as ``info`` has them."""
        rate = info["rate"]
        return _sent(
            info["airspeed_m_s"],
            info["temperature_c"],
            info["pressure_kpa"],
            rate,
            round(seconds * rate),
            info["broadband"],
        )

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        """What its rotors send straight down, dry, as long as its pass: to hear."""
        trim, sent = self.emission(info, info["duration_s"])
        n = sent.length
        below = sent.sent(0, n, np.tile([0.0, 0.0, -1.0], (n, 1)))
        peak = float(np.abs(below).max(initial=0.0))
        write_wav(folder / self.sound_file, below * (0.5 / peak if peak > 0 else 1.0), info["rate"])
        return (
            f"{info['duration_s']:.1f} s of an {info['helicopter']} at {kmh(info['speed_kmh'])} "
            f"km/h ({info['airspeed_m_s']:.1f} m/s through the air), sent straight down; "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power"
        )

    def edit(self, source: dict, info: dict, args: argparse.Namespace) -> None:
        """Its sound power, and its directivity in a CSV table beside the demo's scene."""
        from acoustic_spaces.directivity import write_directivity_csv

        source["level_db"] = info["level_db"]
        source.pop("spectrum", None)
        trim, sent = self.emission(info, POWER_S)
        table = args.scene.with_name(f"{args.scene.stem}_{self.name}.csv")
        write_directivity_csv(
            table,
            sent.directivity(),
            f"The directivity of an {info['helicopter']}'s rotors at "
            f"{info['airspeed_m_s']:.1f} m/s through the air, written by {self.script}",
        )
        named = pipeline.named_from(table, args.source.parent)
        source["directivity"] = {"file": named}
        source.setdefault("trajectory", {})["face_forward"] = True

    def made(self, args: argparse.Namespace, rate: int, duration_s: float, part: str = ""):
        """What its rotors send along its pass, at ``rate``."""
        from acoustic_spaces.demos.passby import car_info

        info = car_info(args, self)
        if info["rate"] != rate:
            raise SystemExit(
                f"error: the {self.name}'s sound is made at {info['rate']} Hz, the scene "
                f"simulated at {rate} Hz"
            )
        return self.emission(info, duration_s)[1]

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        return f"a helicopter {speed}with its rotors' sound sent each way"

    def played_by(self, args: argparse.Namespace) -> str:
        from acoustic_spaces.demos.passby import speed_of

        return f"helicopter {round(speed_of(args, self), 4):g} km/h {args.broadband}"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        return {
            "helicopter": info["helicopter"],
            "airspeed_m_s": info["airspeed_m_s"],
            "sound_power_a_db": info["sound_power_a_db"],
            "broadband": info["broadband"],
            "about": f"An {info['helicopter']} at {kmh(info['speed_kmh'])} km/h, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power (its rotors' model)",
        }

    def credit(self, facts: dict) -> str:
        return (
            f"Helicopter's sound: an {facts['helicopter']}'s two rotors at "
            f"{facts['airspeed_m_s']:.1f} m/s through the air, their tones from the blades' "
            "steady loads (Ffowcs Williams and Hawkings) and their broadband noise by "
            f"{BROADBAND[facts['broadband']]}; faded in and out over {FADE_S:g} s"
        )

    def texts(self) -> dict[str, str]:
        return {
            "speed": "how fast the helicopter flies along its trajectory (km/h; m/s times 3.6), "
            "which sets how long its pass and its sound last (the trajectory's length over the "
            "speed) and its rotors' trim; by default as its trajectory in the scene goes: its "
            "speed along points, or the one steady speed of its keyframes (whose places it "
            "goes through at --speed when given)",
            "input": "a WAV file the helicopter plays instead of its rotors' sound, as "
            "helicopter=WAV: a dry recording, mixed down to mono and sent every way with the "
            "rotors' directivity, at least as long as the pass; the captions' levels are the "
            "rotors' model's",
            "sound": "demo/RUN/helicopter/below.wav: what the helicopter's rotors send straight "
            "down, dry",
            "header": "# The example with its helicopter flying over alone, at the speed and "
            "with the sound\n# power and directivity of its rotors, written by "
            f"{self.script}.\n",
        }


#: The helicopter of the helicopter_flyover example.
HELICOPTER_VEHICLE = Helicopter()

__all__ = [
    "BROADBAND",
    "HELICOPTER",
    "HELICOPTER_VEHICLE",
    "LEVEL_M",
    "STEP_M",
    "Helicopter",
    "airspeed",
]
