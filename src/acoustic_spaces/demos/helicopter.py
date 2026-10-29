"""The helicopter of the flyover demo (:mod:`acoustic_spaces.demos.passby`): the sound its
rotors send each way, and the sound power and directivity that go with it in the scene.

The helicopter is the example's source ``helicopter``, a point at the main rotor's hub
with a walk at one height and one speed, facing the way it flies (``face_forward``). Its
rotors are trimmed for level flight (:func:`acoustic_spaces.rotorcraft.trim`) at its
airspeed: the walk's speed over the ground less the wind at its height, from the scene's
atmosphere (its own weather, not its zones'), in the scene's air. The trim faces the
way it moves through the air; facing along the walk instead, the helicopter does not
turn into a wind from the side (it would fly crabwise), so a scene's wind is best along
the walk or none. What the rotors send (:func:`acoustic_spaces.rotorcraft.emission`)
plays down each early path; the scene's source gets the sound power and the directivity
of that emission (:meth:`~acoustic_spaces.emission.Emission.sound_power_db`,
:meth:`~acoustic_spaces.emission.Emission.directivity`), for the levels the simulation
reports and the late part. Nothing of the simulation uses this module.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
from pathlib import Path

import numpy as np

from acoustic_spaces.demos.audio import write as write_wav
from acoustic_spaces.demos.passby import FADE_S, Vehicle, car_of, pass_s

#: The source of the example that flies over.
HELICOPTER = "helicopter"
#: How far apart the helicopter's places are simulated by default (m): 150 m up, places
#: this far apart follow the exact field of a flyover more closely than a car's places 1
#: m apart do the car's (``validation/doppler.py``).
STEP_M = 4.0
#: How far the walk's points may differ in height and still be level flight (m).
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
    """The airspeed of ``source`` (the helicopter, a scene source with a walk) in
    ``scene`` flying at ``speed_m_s`` over the ground along its first leg: that, less
    the wind of the scene's own atmosphere at its height, both as vectors."""
    from acoustic_spaces.walkthrough import knots

    path = knots(source)
    leg = np.diff(path.positions[:2], axis=0)[0]
    heading = leg / max(float(np.linalg.norm(leg)), 1e-12)
    ground = speed_m_s * heading
    atmosphere = scene.atmosphere
    wind = np.zeros(3)
    if atmosphere is not None:
        speed = float(atmosphere.wind_speed(np.array([source.position[2]]))[0])
        way = math.radians(atmosphere.wind.direction_deg) if atmosphere.wind else 0.0
        wind = speed * np.array([math.cos(way), math.sin(way), 0.0])
    return float(np.linalg.norm(ground - wind))


@functools.lru_cache(maxsize=4)
def _sent(airspeed_m_s: float, temperature_c: float, pressure_kpa: float, rate: int, n: int):
    """The AS350 B3's trim and the ``n`` samples it sends at ``rate``."""
    from acoustic_spaces import rotorcraft

    trim = rotorcraft.trim(rotorcraft.AS350_B3, airspeed_m_s, temperature_c, pressure_kpa)
    return trim, rotorcraft.emission(trim, seconds=n / rate, sample_rate=rate, seed=SEED)


class Helicopter(Vehicle):
    """The AS350 B3 (:data:`acoustic_spaces.rotorcraft.AS350_B3`) flying over."""

    name, demo_name = HELICOPTER, "flyover"
    go, goes, going, way = "fly", "flies", "flying over", "its flight path"
    steady = "its rotors are trimmed for one steady speed"
    step_m = STEP_M
    sound_file, info_file = BELOW_FILE, INFO_FILE

    def options(self) -> list:
        return []

    def check(self, args: argparse.Namespace) -> None:
        return None

    def problem(self, source) -> str | None:
        """A helicopter flies level: its walk keeps one height."""
        from acoustic_spaces.walkthrough import knots

        heights = knots(source).positions[:, 2]
        if float(np.ptp(heights)) > LEVEL_M:
            return (
                f"the {self.name} flies level, its rotors trimmed for it, but its walk goes "
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
        trim, sent = _sent(flying, air.temperature_c, air.pressure_kpa, rate, round(POWER_S * rate))
        level = sent.sound_power_db(trim.density, trim.sound_speed)
        return {
            "helicopter": rotorcraft.AS350_B3.name,
            "mass_kg": rotorcraft.AS350_B3.mass_kg,
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
        )

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        """What its rotors send straight down, dry, as long as its pass: to hear."""
        trim, sent = self.emission(info, info["duration_s"])
        n = sent.length
        below = sent.sent(0, n, np.tile([0.0, 0.0, -1.0], (n, 1)))
        peak = float(np.abs(below).max(initial=0.0))
        write_wav(folder / self.sound_file, below * (0.5 / peak if peak > 0 else 1.0), info["rate"])
        return (
            f"{info['duration_s']:.1f} s of an {info['helicopter']} at {info['speed_kmh']:g} "
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
            f"{info['airspeed_m_s']:.1f} m/s through the air, written by flyover_demo.py",
        )
        named = Path(os.path.relpath(table, args.source.parent)).as_posix()
        source["directivity"] = {"file": named}
        source.setdefault("walk", {})["face_forward"] = True

    def made(self, args: argparse.Namespace, rate: int, duration_s: float):
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
        return f"a helicopter {speed}its rotors' sound sent each way"

    def played_by(self, args: argparse.Namespace) -> str:
        from acoustic_spaces.demos.passby import speed_of

        return f"helicopter {round(speed_of(args, self), 4):g} km/h"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        return {
            "helicopter": info["helicopter"],
            "airspeed_m_s": info["airspeed_m_s"],
            "sound_power_a_db": info["sound_power_a_db"],
            "about": f"An {info['helicopter']} at {info['speed_kmh']:g} km/h, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power (its rotors' model)",
        }

    def credit(self, facts: dict) -> str:
        return (
            f"Helicopter's sound: an {facts['helicopter']}'s two rotors at "
            f"{facts['airspeed_m_s']:.1f} m/s through the air, their tones from the blades' "
            "steady loads (Ffowcs Williams and Hawkings) and their broadband noise by Pegg's "
            f"method (NASA TM-80200); faded in and out over {FADE_S:g} s"
        )

    def texts(self) -> dict[str, str]:
        return {
            "speed": "how fast the helicopter flies along its walk (km/h; m/s times 3.6), "
            "which sets how long its pass and its sound last (the walk's length over the "
            "speed) and its rotors' trim; by default as its walk in the scene goes: its "
            "speed along points, or the one steady speed of its keyframes (whose places it "
            "goes through at --speed when given)",
            "input": "a WAV file the helicopter plays instead of its rotors' sound, as "
            "helicopter=WAV: a dry recording, mixed down to mono and sent every way with the "
            "rotors' directivity, at least as long as the pass; the captions' levels are the "
            "rotors' model's",
            "sound": "demo/helicopter/below.wav: what the helicopter's rotors send straight "
            "down, dry",
            "header": "# The example with its helicopter flying over alone, at the speed and "
            "with the sound\n# power and directivity of its rotors, written by "
            "flyover_demo.py.\n",
        }


#: The helicopter of the helicopter_flyover example.
HELICOPTER_VEHICLE = Helicopter()

__all__ = ["HELICOPTER", "HELICOPTER_VEHICLE", "LEVEL_M", "STEP_M", "Helicopter", "airspeed"]
