"""A vehicle going past, heard at each listener of an outdoor example, its pitch falling
as it goes by (Doppler): the pass-by demos of the roadside and the street examples
(``passby_demo.py`` beside each scene), where a car, a truck, a bus, a motorcycle, a
police car with its siren on or a fire engine with its siren and its air horn drives past
(``--vehicle``), and of the helicopter_flyover
example (``flyover_demo.py``), where a helicopter flies over; one module for all of them,
each with its :class:`Vehicle`.

A road vehicle is the example's source of its name (``car``, ``truck``, ``bus``,
``motorcycle``): a point source 0.05 m above the road, where CNOSSOS-EU places a
vehicle's sound, with a walk along the road with Doppler, left out of the example's own
simulations (``enabled: false``), which keep the traffic as a still line source. Its walk
goes through points at a speed (``to``, ``speed_m_s``) or through keyframes at one
steady speed, as CNOSSOS-EU gives a vehicle's sound at a steady speed. The demo writes
``scene_demo.yaml`` beside its script: the vehicle in use, along its walk at the walk's
own speed or at ``--speed`` (through the keyframes' places, for keyframes), with
CNOSSOS-EU's sound power of its category at that speed and the scene's air temperature
(:mod:`acoustic_spaces.demos.road`), and every other source left out, so that the
vehicle is heard alone. It plays a sound made to that power (``demo/car/car.wav`` for
the car: tyre noise and an engine at ``--rpm``), or the dry recording ``--input
car=WAV`` gives. The police car is two sources along one walk, the car's sound and its
siren on its roof, with the level and the directivity of a measured one
(:mod:`acoustic_spaces.demos.siren`); the fire engine three, the vehicle, its siren and
its air horn (:mod:`acoustic_spaces.demos.horn`); a helicopter plays the sound its rotors send each
way (:mod:`acoustic_spaces.demos.helicopter`).

Each listener in use (or each ``--listeners`` names) is simulated with the vehicle every
``--step`` along its walk
(:func:`acoustic_spaces.walkthrough.simulate_walk`) and hears it through those places,
each early path through its own delay as it changes and the late sound crossfaded from
place to place (:func:`~acoustic_spaces.walkthrough.render_walk`); the level along the
way, its maximum and the sound exposure level come from the same simulation
(:func:`~acoustic_spaces.walkthrough.pass_by`). The video plays one pass at each
listener in turn, the vehicle moving on the plan and the level heard over time drawn
under it.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from acoustic_spaces.demos import (
    Demo,
    Option,
    Step,
    chosen_order,
    common_options,
    input_option,
    listener_names,
    listeners_in,
    listeners_option,
    pipeline,
)
from acoustic_spaces.demos.audio import gated_rms, level_db, loudness_db, mono, read, resampled
from acoustic_spaces.demos.audio import write as write_wav

#: The source of the examples that drives past.
CAR = "car"
#: The speed of the examples' car (km/h), as their scenes give it.
SPEED_KMH = 50.0
#: How far the legs of a vehicle's keyframes may differ in speed and still be one steady
#: speed (a share of the slowest): 1 % is 0.2 dB of CNOSSOS-EU's rolling noise at most.
STEADY = 0.01
#: How far apart the places of the walk are simulated (m): through places 1 m apart a
#: car's sound at 1 kHz follows the exact field's waveform to within -54 dB, -57 dB a
#: quarter of a metre apart (validation/doppler.py).
STEP_M = 1.0
#: The vehicle's sound fades in over its first FADE_S and out over its last, where it
#: is farthest, so that it does not start or stop at once; and the silence between two
#: listeners' passes (s).
FADE_S = 1.0
GAP_S = 1.0
CAR_FILE = "car.wav"
CAR_INFO = "car.json"
WALK = "walk.json"


# --------------------------------------------------------------------------- the vehicle


@dataclasses.dataclass(frozen=True)
class Trail:
    """What goes along a source's walk with it: the source itself ``lead_s`` behind
    where its walk has it (its sound and its levels that much later), and the sounds
    ``behind`` it, each (lag_s, signal, offset_db) ``lag_s`` behind the source, its
    signal scaled as the source's is, its sound power per octave band ``offset_db`` over
    the source's (a train's bogies and cars, behind its front)."""

    lead_s: float = 0.0
    behind: tuple[tuple[float, np.ndarray, np.ndarray], ...] = ()

    @property
    def length_s(self) -> float:
        """How far behind the walk the last of it goes (s)."""
        return self.lead_s + max((lag for lag, _, _ in self.behind), default=0.0)


def trailed(passing, trail: Trail, extra_s: float):
    """``passing`` (a :class:`~acoustic_spaces.walkthrough.PassBy` of a source alone)
    with what goes along with it (``trail``): each copy's level that much later, its
    energy band by band ``offset_db`` over the source's, all added, the times going on
    ``extra_s`` past the walk's last place, in its steps."""
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.walkthrough import PassBy

    if not trail.behind and trail.lead_s == 0.0 and extra_s <= 0.0:
        return passing
    times = np.asarray(passing.times, dtype=float)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    more = int(math.ceil(max(extra_s, 0.0) / step - 1e-9))
    grid = np.concatenate([times, times[-1] + step * np.arange(1, more + 1)])
    energy = 10.0 ** (np.asarray(passing.levels, dtype=float) / 10.0)
    copies = [(0.0, np.zeros(energy.shape[1])), *((lag, off) for lag, _, off in trail.behind)]
    total = np.zeros((len(grid), energy.shape[1]))
    for lag, offset in copies:
        at = grid - trail.lead_s - lag
        scale = 10.0 ** (np.asarray(offset, dtype=float) / 10.0)
        for band in range(energy.shape[1]):
            total[:, band] += scale[band] * np.interp(
                at, times, energy[:, band], left=0.0, right=0.0
            )
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(total)
    levels_a = np.array([a_weighted(row) for row in levels])
    return PassBy(passing.source, passing.listener, grid, levels, levels_a)


class Vehicle:
    """What a pass-by demo plays past the listeners: the example's source of that name,
    the words the demo says it with, and how its sound and its sound power are made
    (:class:`RoadVehicle`, :class:`~acoustic_spaces.demos.siren.PoliceCar`,
    :class:`~acoustic_spaces.demos.helicopter.Helicopter`)."""

    #: The example's source, which goes past, and the demo's short name.
    name, demo_name = CAR, "passby"
    #: How it goes: "drive", "drives" and "driving past", along "the road".
    go, goes, going, way = "drive", "drives", "driving past", "the road"
    #: The vehicle in a word or two, for the demo's words ("car", "police car"; its
    #: source's name may join them with an underscore), and the word for the vehicles
    #: of a demo of several, this one the first.
    noun = "car"
    fleet_noun = "vehicle"
    #: Whether it goes at a speed of its own without --speed (:meth:`own_speed_kmh`).
    own_speed = False
    #: Whether its own numbers time its walk, at no one speed (an aircraft's flight
    #: profile, :mod:`acoustic_spaces.demos.aircraft`): it goes as they say, its
    #: keyframes written into the demo's scene (:meth:`edit`), and --speed does not
    #: apply.
    timed = False
    #: What it is, in a few words, for the demo's choice of vehicle.
    title = "a car"
    #: Why its keyframes must keep one speed.
    steady = "CNOSSOS-EU gives a car's sound power at one steady speed"
    #: How far apart its places are simulated by default (m).
    step_m = STEP_M
    #: Its sound's file in the work folder's folder of its name, and what that is made of.
    sound_file, info_file = CAR_FILE, CAR_INFO
    #: The demo's step that makes its sound.
    sound_step = "sound"

    def options(self) -> list[Option]:
        """The options of its own, for the step that makes its sound."""
        return []

    def check(self, args: argparse.Namespace) -> None:
        """Stop with an error when its own options cannot make its sound."""
        return None

    def own_speed_kmh(self, args: argparse.Namespace) -> float | None:
        """The speed it goes at without ``--speed`` (km/h), or None to go as its walk in
        the scene goes (a train goes at its own)."""
        return None

    def problem(self, source) -> str | None:
        """What else keeps ``source`` (its scene source, with a walk that is on) from
        going past: nothing, unless the vehicle needs more."""
        return None

    def mean_speed_m_s(self, args: argparse.Namespace, scene) -> float:
        """The speed it goes at along its walk in ``scene`` (m/s), the example's: its
        walk's (:func:`walk_speed`), unless it is :attr:`timed`, whose own numbers give
        the walk of the demo's scene, its length over its time."""
        return walk_speed(car_of(scene, self), self)

    def how_it_goes(self, args: argparse.Namespace, scene) -> str | None:
        """How a :attr:`timed` vehicle goes, in a line for the demo's summary (None: as
        a vehicle at one speed is said)."""
        return None

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sound is made of, at ``speed_kmh`` in ``scene``, with the sound power
        it gives the scene (``level_db``, ``sound_power_db``, ``sound_power_a_db``) and
        the pass's ``duration_s`` and ``rate``."""
        raise NotImplementedError

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        """Write its sound, dry, into ``folder``, made to ``info``; what to print."""
        raise NotImplementedError

    def parts(self, args: argparse.Namespace) -> tuple[str, ...]:
        """The example's sources it plays, all along one walk: its own alone, unless it
        carries more (the police car's siren)."""
        return (self.name,)

    def every_part(self) -> tuple[str, ...]:
        """Every source of the example it can play, its own first: each goes along its
        walk."""
        return (self.name,)

    def engines(self) -> tuple[RoadVehicle, ...]:
        """The road vehicles of it whose engines ``--rpm`` turns."""
        return ()

    def levels(self, info: dict) -> dict[str, list[float]]:
        """The sound power per octave band (dB) each source it plays gets, of ``info``."""
        return {self.name: info["level_db"]}

    def edit(self, source: dict, info: dict, args: argparse.Namespace) -> None:
        """Give ``source`` (the entry of one of its :meth:`parts` in the demo's scene, as
        data) its sound power."""
        source["level_db"] = self.levels(info)[source["name"]]
        source.pop("spectrum", None)

    def sound_files(self, args: argparse.Namespace) -> tuple[str, ...]:
        """The files of its made sounds in the work folder's folder of its name."""
        return (self.sound_file,)

    def sound_path(self, args: argparse.Namespace, part: str) -> Path:
        """The file of the made sound the source ``part`` plays."""
        return args.work / self.name / self.sound_file

    def made(self, args: argparse.Namespace, rate: int, duration_s: float, part: str = ""):
        """What the source ``part`` (by default its own) plays along its pass at
        ``rate``: its made sound, read back (None before it is made)."""
        path = self.sound_path(args, part or self.name)
        if not path.exists():
            return None
        x, r = read(path)
        return resampled(mono(x), r, rate)

    def trail(self, args: argparse.Namespace, part: str, played: np.ndarray, rate: int) -> Trail:
        """What goes along the walk of its source ``part`` with it, which plays
        ``played`` at ``rate`` (:class:`Trail`): nothing, unless the vehicle is a train
        (:mod:`acoustic_spaces.demos.rail`)."""
        return Trail()

    def sounding(
        self, args: argparse.Namespace, part: str, played, rate: int, at: np.ndarray
    ) -> np.ndarray | None:
        """How loud the source ``part`` sounds around each of the times ``at`` (s into
        its pass, one a place), against the sound power its scene source has, in energy:
        None for a sound as loud all along (a vehicle's; ``played``, its feed at
        ``rate``, is not read), unless it sounds now and then (a horn's blasts,
        :class:`~acoustic_spaces.demos.siren.EmergencyVehicle`)."""
        return None

    def input_gain(self, part: str, x: np.ndarray, rate: int) -> float | None:
        """The gain of ``x``, a recording ``--input`` gives the source ``part``, so that it
        plays as loud as a made sound: its gated RMS
        (:func:`~acoustic_spaces.demos.audio.gated_rms`) brought to SOUND_RMS, the RMS of
        a made sound that has no pauses; None when it is silent."""
        from acoustic_spaces.demos.road import SOUND_RMS

        rms = gated_rms(x, rate)
        return SOUND_RMS / rms if rms > 0.0 else None

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        """What it plays unless ``--input`` gives it a file, ``speed`` its speed in words."""
        raise NotImplementedError

    def feeds(self, args: argparse.Namespace, speed: str) -> dict[str, str]:
        """What each of its :meth:`parts` plays unless ``--input`` gives it a file."""
        return {self.name: self.plays(args, speed)}

    def played_by(self, args: argparse.Namespace) -> str:
        """Its made sound, as walk.json keeps it."""
        raise NotImplementedError

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        """What walk.json says of it, with the stills' line ``about``."""
        raise NotImplementedError

    def credit(self, facts: dict) -> str:
        """The stills' credit line for its made sound."""
        raise NotImplementedError

    def credits(self, facts: dict, files: dict[str, Path | None]) -> str:
        """The stills' credit line: for its made sound, or the file ``--input`` gives it
        (``files``, by source)."""
        file = files.get(self.name)
        if file is not None:
            return f"Playing {self.name}: {file.name}, faded in and out over {FADE_S:g} s"
        return self.credit(facts)

    def texts(self) -> dict[str, str]:
        """The help of the demo's options and steps that speak of it: ``speed``,
        ``input``, ``sound`` and the demo scene's ``header``."""
        raise NotImplementedError

    def fleet_texts(self) -> dict[str, str] | None:
        """The help of a demo of several vehicles, this one the first, as :meth:`texts`
        (None: a road vehicle's, :func:`road_texts`)."""
        return None


#: Engines with their cylinders in words, for the stills.
CYLINDER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight"}


class RoadVehicle(Vehicle):
    """A road vehicle of a CNOSSOS-EU category (:mod:`acoustic_spaces.demos.road`) going
    past at one steady speed: its sound power that category's at that speed, and its sound
    made to it by tyres and an engine of ``cylinders`` turning at ``rpm`` (unless
    ``--rpm`` says)."""

    def __init__(
        self, name: str, category: str, title: str, rpm: float, cylinders: int, noun: str = ""
    ) -> None:
        from acoustic_spaces.demos.road import CATEGORIES

        if category not in CATEGORIES:
            raise ValueError(f"CNOSSOS-EU has no vehicle category {category!r}")
        self.name, self.category, self.title = name, category, title
        self.rpm, self.cylinders = float(rpm), int(cylinders)
        #: The vehicle in a word, for the demo's words ("car", "bus").
        self.noun = noun or name.replace("_", " ")
        self.sound_file, self.info_file = f"{name}.wav", f"{name}.json"
        self.steady = f"CNOSSOS-EU gives a {self.noun}'s sound power at one steady speed"

    def engines(self) -> tuple[RoadVehicle, ...]:
        return (self,)

    @property
    def kind(self):
        """Its CNOSSOS-EU category (:class:`~acoustic_spaces.demos.road.Category`)."""
        from acoustic_spaces.demos.road import CATEGORIES

        return CATEGORIES[self.category]

    def options(self) -> list[Option]:
        return [
            Option(
                "rpm",
                rpm_help((self,)),
                None,
                "float",
                steps=(self.sound_step,),
                unit="rpm",
                unset="the vehicle's own",
            )
        ]

    def rpm_of(self, args: argparse.Namespace) -> float:
        """How fast its engine turns: ``--rpm``, or its own."""
        rpm = getattr(args, "rpm", None)
        return self.rpm if rpm is None else float(rpm)

    def check(self, args: argparse.Namespace) -> None:
        rpm = getattr(args, "rpm", None)
        if rpm is not None and not (math.isfinite(float(rpm)) and float(rpm) > 0.0):
            raise SystemExit(f"error: --rpm is a positive number, not {rpm!r}")

    def power(self, speed_kmh: float, temperature_c: float):
        """Its sound power at ``speed_kmh`` in air at ``temperature_c``."""
        from acoustic_spaces.demos.road import vehicle_power

        return vehicle_power(self.category, speed_kmh, temperature_c)

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sound is made of, at ``speed_kmh`` in ``scene``: the speed, the
        engine, the air's temperature, the length and CNOSSOS-EU's sound power."""
        power = self.power(speed_kmh, scene.air.temperature_c)
        rolling = power.rolling_db
        return {
            "category": self.category,
            "speed_kmh": round(speed_kmh, 6),
            "rpm": self.rpm_of(args),
            "cylinders": self.cylinders,
            "temperature_c": scene.air.temperature_c,
            "rate": int(scene.simulation.sample_rate),
            "duration_s": round(pass_s(car_of(scene, self), args.speed), 6),
            "level_db": [round(float(v), 1) for v in power.total_db],
            "rolling_db": None if rolling is None else [round(float(v), 2) for v in rolling],
            "propulsion_db": [round(float(v), 2) for v in power.propulsion_db],
            "sound_power_db": round(power.sound_power_db, 2),
            "sound_power_a_db": round(power.sound_power_a_db, 2),
        }

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        from acoustic_spaces.demos.road import vehicle_sound

        power = self.power(info["speed_kmh"], info["temperature_c"])
        x = vehicle_sound(
            power, info["rate"], info["duration_s"], rpm=info["rpm"], cylinders=self.cylinders
        )
        write_wav(folder / self.sound_file, x, info["rate"])
        return (
            f"{info['duration_s']:.1f} s of a {self.noun} at {kmh(info['speed_kmh'])} km/h, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power"
        )

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        return f"a {self.noun} {speed}made to CNOSSOS-EU's sound power"

    def played_by(self, args: argparse.Namespace) -> str:
        return f"{self.name} {round(speed_of(args, self), 4):g} km/h {self.rpm_of(args):g} rpm"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        """What walk.json says of it: its engine, its sound power and the stills' line."""
        return {
            "rpm": self.rpm_of(args),
            "cylinders": self.cylinders,
            "category": self.category,
            "sound_power_a_db": info["sound_power_a_db"],
            "about": f"A {self.noun} at {kmh(info['speed_kmh'])} km/h, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power (CNOSSOS-EU)",
        }

    def credit(self, facts: dict) -> str:
        cylinders = CYLINDER_WORDS.get(facts["cylinders"], str(facts["cylinders"]))
        tyres = "tyre noise and a" if self.kind.rolling_a is not None else "a"
        parts = "each" if self.kind.rolling_a is not None else "its sound"
        return (
            f"{self.noun.capitalize()}'s sound made up: {tyres} {cylinders}-cylinder engine at "
            f"{facts['rpm']:g} rpm, {parts} with the octave spectrum of CNOSSOS-EU's "
            f"{self.kind.name} (category {self.category}) at {kmh(facts['speed_kmh'])} km/h "
            f"(Directive 2002/49/EC as amended in 2021); faded in and out over {FADE_S:g} s"
        )

    def texts(self) -> dict[str, str]:
        return road_texts(self.noun, self.name, f"demo/{self.name}/{self.sound_file}")


def rpm_help(fleet: Sequence[RoadVehicle]) -> str:
    """The help of ``--rpm`` for a demo of the road vehicles ``fleet``."""
    return (
        "how fast the engine of the vehicle's made sound turns (revolutions per minute); "
        "by default the vehicle's own: "
        + ", ".join(f"{v.noun} {v.rpm:g} ({v.cylinders} cylinders)" for v in fleet)
    )


def road_texts(noun: str, name: str, sound: str) -> dict[str, str]:
    """The help of a road vehicle's pass-by demo, for ``noun`` (the vehicle in a word)
    whose source is ``name`` and whose made sound is ``sound``."""
    return {
        "speed": f"how fast the {noun} drives along its walk (km/h; m/s times 3.6), which "
        f"sets how long its pass and its sound last (the walk's length over the speed) and "
        "its sound power; by default as its walk in the scene goes: its speed along "
        "points, or the one steady speed of its keyframes (whose places it goes "
        "through at --speed when given)",
        "input": f"a WAV file the {noun} plays instead of its made sound, as {name}=WAV: a "
        "dry recording (on board, or close to the vehicle; one of it passing has its "
        "Doppler, the road and the distance in it already), mixed down to mono, at least "
        f"as long as the pass; the captions' levels are CNOSSOS-EU's {noun} at --speed",
        "sound": f"{sound}: the {noun}'s sound, made to CNOSSOS-EU",
        "header": f"# The example with its {noun} driving past alone, at the speed and with "
        "the sound\n# power of the pass-by demo (CNOSSOS-EU), written by passby_demo.py.\n",
    }


#: The car of the roadside and the street examples: a passenger car (CNOSSOS-EU category
#: 1), its made sound's engine four cylinders at 2000 rpm.
THE_CAR = RoadVehicle(CAR, "1", "a passenger car", rpm=2000.0, cylinders=4)
#: The roadside's truck: an articulated truck with three axles or more (category 3), a
#: six-cylinder diesel at 1200 rpm.
TRUCK = RoadVehicle(
    "truck", "3", "an articulated truck, three axles or more", rpm=1200.0, cylinders=6
)
#: The street's bus: a city bus with two axles (category 2), a six-cylinder diesel at
#: 1500 rpm.
BUS = RoadVehicle("bus", "2", "a city bus with two axles", rpm=1500.0, cylinders=6)
#: The street's motorcycle (category 4b): a two-cylinder engine at 4000 rpm.
MOTORCYCLE = RoadVehicle("motorcycle", "4b", "a motorcycle", rpm=4000.0, cylinders=2)
#: Every road vehicle of the examples.
ROAD_VEHICLES = (THE_CAR, TRUCK, BUS, MOTORCYCLE)


def walk_speed(car, vehicle: Vehicle = THE_CAR) -> float:
    """The speed the walk of ``car`` (a scene source) goes at (m/s): its ``speed_m_s``
    along points, or along keyframes the one steady speed they keep (for a
    :attr:`~Vehicle.timed` vehicle, their length over their time); a ValueError says
    when keyframes go at different speeds or wait."""
    from acoustic_spaces.walkthrough import knots

    if car.walk.keyframes is None:
        return float(car.walk.speed_m_s)
    path = knots(car)
    if vehicle.timed:
        return path.length_m / path.duration_s
    speeds = np.linalg.norm(np.diff(path.positions, axis=0), axis=1) / np.diff(path.times)
    slow, fast = float(speeds.min()), float(speeds.max())
    if slow <= 0.0 or fast > slow * (1.0 + STEADY):
        raise ValueError(
            f"the {vehicle.noun}'s keyframes go at {3.6 * slow:.1f} to {3.6 * fast:.1f} km/h, "
            f"but {vehicle.steady}: time them for one speed (or give its walk points at a "
            f"speed), or {vehicle.go} it through their places at one speed with --speed"
        )
    return path.length_m / path.duration_s


def kmh(speed: float) -> str:
    """A speed (km/h) as the demos' words and stills give it: to a tenth, without a
    trailing zero, so a walk of 13.89 m/s (50.004 km/h) reads 50."""
    return f"{round(float(speed), 1):g}"


def forced_speed(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> float | None:
    """The speed a run gives the vehicle instead of its walk's (km/h): ``--speed``, or
    its own (:meth:`Vehicle.own_speed_kmh`, a train's); None to go as its walk goes."""
    if getattr(args, "speed", None) is not None:
        return float(args.speed)
    return vehicle.own_speed_kmh(args)


def speed_of(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> float:
    """The speed the vehicle goes at (km/h): ``--speed``, checked positive and finite,
    or without it its own (a train's) or the speed of its walk in the example's scene
    (:func:`walk_speed`; a :attr:`~Vehicle.timed` vehicle's own mean speed)."""
    if getattr(args, "speed", None) is not None:
        speed = float(args.speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise SystemExit(f"error: --speed is a positive number of km/h, not {args.speed!r}")
        return speed
    own = vehicle.own_speed_kmh(args)
    if own is not None:
        return own
    scene = source_scene(args)
    problem = car_problem(scene, args.source.name, vehicle)
    if problem:
        raise SystemExit(f"error: {problem}")
    try:
        return 3.6 * vehicle.mean_speed_m_s(args, scene)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


def car_of(scene, vehicle: Vehicle = THE_CAR):
    """The vehicle's source in ``scene`` (a :class:`~acoustic_spaces.scene.Scene`)."""
    car = next((s for s in scene.sources if s.name == vehicle.name), None)
    if car is None:
        raise SystemExit(f"error: the scene has no source {vehicle.name!r} to {vehicle.go}")
    return car


def car_problem(scene, name: str, vehicle: Vehicle = THE_CAR) -> str | None:
    """What keeps the vehicle of ``scene`` (read from the file ``name``) from going
    past: none, a line or a surface, no walk, a walk that is off, or what the vehicle
    itself needs (:meth:`Vehicle.problem`)."""
    noun = vehicle.name
    car = next((s for s in scene.sources if s.name == noun), None)
    if car is None:
        return f"{name} has no source {noun!r} to {vehicle.go}"
    if car.extended:
        return f"the {noun} of {name} is a line or a surface: only a point source {vehicle.goes}"
    if car.walk is None:
        return (
            f"the {noun} has no walk in {name}: give it one along {vehicle.way} (walk: to:, "
            "or the editor's Movement panel)"
        )
    if not car.walk.enabled:
        return (
            f"the {noun}'s walk is off in {name}: turn it on (Moves in the editor, or walk: "
            "enabled: true)"
        )
    problem = vehicle.problem(car)
    if problem:
        return f"{name}: {problem}"
    for part in vehicle.every_part()[1:]:
        problem = part_problem(scene, car, part, vehicle)
        if problem:
            return f"{name}: {problem}"
    return None


def part_problem(scene, car, part: str, vehicle: Vehicle) -> str | None:
    """What keeps the source ``part`` of ``scene`` from going along with ``car`` (the
    vehicle's own source): none, unless it is missing, a line or a surface, without a
    walk that is on, or its walk goes elsewhere (in plan) or at other times."""
    from acoustic_spaces.walkthrough import knots

    source = next((s for s in scene.sources if s.name == part), None)
    noun = vehicle.name
    if source is None:
        return f"no source {part!r}, which goes along with the {noun}"
    if source.extended:
        return f"the {part} is a line or a surface: only a point source goes along"
    if source.walk is None or not source.walk.enabled:
        return f"the {part} has no walk that is on: give it the {noun}'s"
    own, along = knots(car), knots(source)
    if len(own.times) != len(along.times) or not (
        np.allclose(own.times, along.times, atol=1e-6)
        and np.allclose(own.positions[:, :2], along.positions[:, :2], atol=1e-6)
    ):
        return (
            f"the {part}'s walk must go where the {noun}'s does (in plan) and when: "
            f"give it the {noun}'s points or keyframes, at its own height"
        )
    return None


def places_of(car) -> list[tuple[float, ...]]:
    """The places the keyframes of ``car`` go through after its position, each once:
    the points of a walk through them."""
    places: list[tuple[float, ...]] = []
    last = tuple(car.position)
    for frame in car.walk.keyframes:
        if tuple(frame.position) != last:
            places.append(tuple(frame.position))
            last = tuple(frame.position)
    return places


def driving(car, speed_kmh: float | None):
    """``car`` (a scene source) in use, with Doppler, along its walk as it is (None) or
    at ``speed_kmh``: a walk through points at that speed, through the keyframes' places
    for keyframes."""
    update: dict = {"doppler": True}
    if speed_kmh is not None:
        update["speed_m_s"] = speed_kmh / 3.6
        if car.walk.keyframes is not None:
            update |= {"to": places_of(car), "keyframes": None}
    walk = car.walk.model_copy(update=update)
    return car.model_copy(update={"walk": walk, "enabled": True})


def pass_s(car, speed_kmh: float | None = None) -> float:
    """How long the vehicle takes along its walk, as it is or at ``speed_kmh`` (s)."""
    from acoustic_spaces.walkthrough import knots

    return float(knots(driving(car, speed_kmh)).duration_s)


def source_scene(args: argparse.Namespace):
    """The example's scene (``--from``)."""
    from acoustic_spaces.scene import Scene

    if not args.source.exists():
        raise SystemExit(f"error: no {args.source}")
    return Scene.from_yaml(args.source)


def car_input(args: argparse.Namespace, vehicle: Vehicle = THE_CAR, part: str = "") -> Path | None:
    """The file ``--input NAME=WAV`` gives the vehicle (or its source ``part``) to play,
    or None."""
    return (getattr(args, "inputs", None) or {}).get(part or vehicle.name)


def car_info(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> dict:
    """What the vehicle's made sound is made of, for the arguments and the example's
    scene (:meth:`Vehicle.info`)."""
    scene = source_scene(args)
    problem = car_problem(scene, args.source.name, vehicle)
    if problem:
        raise SystemExit(f"error: {problem}")
    return vehicle.info(args, scene, speed_of(args, vehicle))


def car(demo: Demo, args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> None:
    """The vehicle's sound, dry, as long as its pass (for the car ``car/car.wav``: tyres
    and engine made to CNOSSOS-EU's sound power at ``--speed``,
    :func:`acoustic_spaces.demos.road.vehicle_sound`), and what it is made of."""
    vehicle.check(args)
    info = car_info(args, vehicle)
    folder = args.work / vehicle.name
    said = vehicle.make(args, info, folder)
    pipeline.write_json(folder / vehicle.info_file, info)
    files = ", ".join(str(folder / f) for f in vehicle.sound_files(args))
    print(f"wrote {files}: {said}")


def car_done(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> bool:
    """Whether the vehicle's sound is there and made for the arguments and the scene as
    it is now, or not needed (``--input NAME=...``)."""
    if all(car_input(args, vehicle, part) is not None for part in vehicle.parts(args)):
        return True
    folder = args.work / vehicle.name
    path = folder / vehicle.info_file
    if not path.exists() or not all((folder / f).exists() for f in vehicle.sound_files(args)):
        return False
    try:
        return pipeline.read_json(path) == car_info(args, vehicle)
    except SystemExit:
        return False


def plays(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> dict[str, str]:
    """What the vehicle plays unless ``--input`` gives it a file."""
    try:
        speed = f"at {round(speed_of(args, vehicle), 2):g} km/h, "
    except SystemExit:  # the check under the form says what keeps it from going
        speed = ""
    return vehicle.feeds(args, speed)


def played_by(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> str:
    """What the vehicle plays, as walk.json keeps it: its made sound, and the files
    ``--input`` gives its sources instead."""
    files = {part: car_input(args, vehicle, part) for part in vehicle.parts(args)}
    given = {part: file for part, file in files.items() if file is not None}
    if len(files) == 1 and given:
        return str(*given.values())
    return vehicle.played_by(args) + "".join(f"; {p}={f}" for p, f in given.items())


def feed(
    demo: Demo,
    args: argparse.Namespace,
    rate: int,
    duration_s: float,
    vehicle: Vehicle = THE_CAR,
    part: str = "",
):
    """What the vehicle (or its source ``part``) plays along its pass, at ``rate``: its
    made sound (:meth:`Vehicle.made`), or the file ``--input NAME=WAV`` gives it (mixed
    down to mono, cut to the pass, as loud as its made sound,
    :meth:`Vehicle.input_gain`), faded in and out over
    FADE_S."""
    from acoustic_spaces.emission import Emission, Faded

    n = int(round(duration_s * rate))
    fade = min(int(FADE_S * rate), n // 2)
    part = part or vehicle.name
    file = car_input(args, vehicle, part)
    if file is None:
        x = vehicle.made(args, rate, duration_s, part)
        if x is None:
            path = vehicle.sound_path(args, part)
            raise SystemExit(
                f"error: no {path}; run first: {demo.command(vehicle.sound_step, args)}"
            )
        if isinstance(x, Emission):
            if x.length < n:
                raise SystemExit(
                    f"error: the {vehicle.noun}'s sound is shorter than its pass; run: "
                    f"{demo.command(vehicle.sound_step, args)}"
                )
            return Faded(x, fade, n)
    else:
        x, r = read(file)
        x = resampled(mono(x), r, rate)
        if len(x) < n:
            raise SystemExit(
                f"error: {file.name} is {len(x) / rate:.1f} s long; the {vehicle.noun} takes "
                f"{duration_s:.1f} s along its walk"
            )
        gain = vehicle.input_gain(part, x[:n], rate)
        if gain is None:
            raise SystemExit(f"error: {file} is silent")
        x = x * gain
    x = np.array(x[:n], dtype=float)
    if len(x) < n:
        raise SystemExit(
            f"error: the {vehicle.noun}'s sound is shorter than its pass; run: "
            f"{demo.command(vehicle.sound_step, args)}"
        )
    return faded(x, rate)


def faded(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` faded in over its first FADE_S and out over its last (half its length at
    most each), as a pass's sound plays."""
    x = np.array(x, dtype=float)
    n = len(x)
    fade = min(int(FADE_S * rate), n // 2)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    x[:fade] *= ramp
    x[n - fade :] *= ramp[::-1]
    return x


# --------------------------------------------------------------------------- the scene


def scene(demo: Demo, args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> None:
    """``--scene`` (scene_demo.yaml) from ``--from`` (scene.yaml): the vehicle's sources
    it plays (:meth:`Vehicle.parts`) in use, along their walk with Doppler, at its own
    speed or at ``--speed`` (:func:`driving`), and their sound power at that speed
    (:meth:`Vehicle.edit`), every other source left out; those of the vehicle's it
    leaves out (:meth:`Vehicle.every_part`) still walk at that speed, so that its
    sources go along one walk."""
    info = car_info(args, vehicle)
    data = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    data["name"] = f"{data.get('name', 'scene')}: a {vehicle.noun} passing"
    own = {s.name: s for s in source_scene(args).sources}
    parts, every = vehicle.parts(args), vehicle.every_part()
    for source in data["sources"]:
        name = source["name"]
        source["enabled"] = name in parts
        if name in parts:
            vehicle.edit(source, info, args)
        if name in every and source.get("walk"):
            walk = source["walk"]
            if forced_speed(args, vehicle) is not None:
                walk["speed_m_s"] = round(info["speed_kmh"] / 3.6, 6)
                if walk.get("keyframes"):
                    walk["to"] = [list(place) for place in places_of(own[name])]
                    walk.pop("keyframes")
            if name in parts:
                walk["doppler"] = True
    pipeline.write_scene(args.scene, data, vehicle.texts()["header"], args.source.parent)
    speed = f"{kmh(info['speed_kmh'])} km/h" + (" on average" if vehicle.timed else "")
    print(
        f"wrote {args.scene}: the {vehicle.noun} at {speed}, "
        f"{info['sound_power_db']:.1f} dB of sound power"
    )


def scene_done(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> bool:
    """Whether scene_demo.yaml is there, newer than the example's scene, and for this
    vehicle at this speed, with the sources it plays in use and the sound power it
    gives them (the police car's siren in the pattern of ``--siren``)."""
    from acoustic_spaces.scene import Scene

    if not args.scene.exists():
        return False
    if args.source.exists() and args.scene.stat().st_mtime < args.source.stat().st_mtime:
        return False
    try:
        sources = {s.name: s for s in Scene.from_yaml(args.scene).sources}
        powers = vehicle.levels(car_info(args, vehicle))
        speed = speed_of(args, vehicle)
        made = {name: 3.6 * walk_speed(sources[name], vehicle) for name in powers}
    except (Exception, SystemExit):  # a scene of another kind: written again
        return False
    others = set(vehicle.every_part()) - set(powers)
    return all(
        sources[name].in_use
        and sources[name].walk.doppler
        and abs(made[name] - speed) < 1e-4 * made[name]
        and np.allclose(np.asarray(sources[name].level_db, dtype=float), power, atol=0.05)
        for name, power in powers.items()
    ) and not any(sources[name].in_use for name in others if name in sources)


def room_of(demo: Demo, args: argparse.Namespace):
    from acoustic_spaces import Room

    if not args.scene.exists():
        raise SystemExit(f"error: no {args.scene}; run first: {demo.command('scene', args)}")
    return Room.from_yaml(args.scene)


# --------------------------------------------------------------------------- the pass


def pitch_change(points, listener, speed_m_s: float | None, c: float) -> float:
    """How far the pitch of the vehicle's direct sound falls at ``listener`` (a position)
    from where its walk starts to where it ends (semitones): heard at c / (c - v_r), v_r
    the vehicle's speed towards the listener, along its first leg and its last, at
    ``speed_m_s`` or, None, at each leg's own speed (its length over its time)."""
    positions = np.asarray(points.positions, dtype=float)
    at = np.asarray(listener, dtype=float)

    def towards(k: int, leg: np.ndarray, between: tuple[int, int]) -> float:
        to = at - positions[k]
        distance = float(np.linalg.norm(to))
        length = max(float(np.linalg.norm(leg)), 1e-12)
        if speed_m_s is None:
            times = points.times
            speed = length / float(times[between[1]] - times[between[0]])
        else:
            speed = speed_m_s
        return speed * float(np.dot(leg / length, to)) / max(distance, 1e-12)

    first = towards(0, positions[1] - positions[0], (0, 1))
    last = towards(-1, positions[-1] - positions[-2], (-2, -1))
    return 12.0 * math.log2((c - last) / (c - first))


def listeners_of(room) -> list[str]:
    return [lis.name for lis in room.scene.listeners if lis.in_use]


def alone(room, part: str):
    """``room`` with its source ``part`` alone in use, the others left out."""
    from acoustic_spaces import Room

    scene = room.scene
    sources = [
        s if s.name == part else s.model_copy(update={"enabled": False}) for s in scene.sources
    ]
    return Room.from_scene(scene.model_copy(update={"sources": sources}), reuse=room)


def summed(mixes: Sequence[np.ndarray]) -> np.ndarray:
    """The sum of ``mixes`` (samples by channels), the shorter ones padded with silence."""
    out = np.zeros((max(len(m) for m in mixes), mixes[0].shape[1]))
    for mix in mixes:
        out[: len(mix)] += mix
    return out


def blocks_loudness(x: np.ndarray, rate: int, at: np.ndarray, rms: float) -> np.ndarray:
    """How loud ``x`` (at ``rate``) is around each of the times ``at`` (s), in energy
    against a sound of RMS ``rms``: its mean square from halfway to the time before to
    halfway to the next (the first from the start, the last to the end), over
    ``rms`` squared; 0 where none of ``x`` is."""
    at = np.asarray(at, dtype=float)
    if len(at) == 0:
        return np.zeros(0)
    middle = (at[1:] + at[:-1]) / 2.0
    edges = np.concatenate([[0.0], middle, [len(x) / rate]])
    out = np.zeros(len(at))
    for k in range(len(at)):
        lo = int(round(max(edges[k], 0.0) * rate))
        hi = int(round(min(edges[k + 1], len(x) / rate) * rate))
        if hi > lo:
            out[k] = float(np.mean(np.asarray(x[lo:hi], dtype=float) ** 2)) / rms**2
    return out


def sounded(passing, loud: np.ndarray):
    """``passing`` (a :class:`~acoustic_spaces.walkthrough.PassBy`) with each place's
    energy times ``loud`` (one a place): a source that sounds now and then, heard as it
    sounds around each place."""
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.walkthrough import PassBy

    energy = 10.0 ** (np.asarray(passing.levels, dtype=float) / 10.0)
    energy = energy * np.asarray(loud, dtype=float)[:, None]
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(energy)
    levels_a = np.array([a_weighted(row) for row in levels])
    return PassBy(passing.source, passing.listener, passing.times, levels, levels_a)


def heard_together(passes: Sequence):
    """The level along a walk of sources going along it together, from each one's
    :func:`~acoustic_spaces.walkthrough.pass_by`: their energies added band by band at
    the times the first's places are heard, each other's read there between its own
    places (linearly in energy; a few milliseconds apart, as their direct sounds
    travel)."""
    from acoustic_spaces.parameters import a_weighted
    from acoustic_spaces.walkthrough import PassBy

    first = passes[0]
    if len(passes) == 1:
        return first
    energy = 10.0 ** (first.levels / 10.0)
    for other in passes[1:]:
        theirs = 10.0 ** (other.levels / 10.0)
        energy = energy + np.stack(
            [np.interp(first.times, other.times, band) for band in theirs.T], axis=1
        )
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(energy)
    levels_a = np.array([a_weighted(row) for row in levels])
    return PassBy(first.source, first.listener, first.times, levels, levels_a)


def walk(demo: Demo, args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> None:
    """Each listener in use (those ``--listeners`` names, in its order): the vehicle
    simulated every ``--step`` along its walk, its
    sound played through those places with Doppler, and the level along the way; one
    gain for every listener (the loudest peak at -1 dBFS), into ``walk/``: a WAV per
    listener, seats.json and walk.json. A vehicle of several sources (the police car and
    its siren, :meth:`Vehicle.parts`) has each simulated and played alone, and heard
    together: their sounds added, each at its own sound power, and their levels'
    energies added."""
    from acoustic_spaces.demos.scene_demo import format_of
    from acoustic_spaces.render import power_gains
    from acoustic_spaces.walkthrough import pass_by, render_walk, simulate_walk

    noun = vehicle.name
    room = room_of(demo, args)
    problem = car_problem(room.scene, args.scene.name, vehicle)
    if problem:
        raise SystemExit(f"error: {problem}")
    parts = vehicle.parts(args)
    sources = {s.name: s for s in room.scene.sources}
    for part in parts:
        if part not in sources or not sources[part].in_use:
            raise SystemExit(
                f"error: the {part} is left out of {args.scene.name}; run: "
                f"{demo.command('scene', args)}"
            )
        if not sources[part].walk.doppler:
            raise SystemExit(
                f"error: the {part} in {args.scene.name} has no Doppler; run: "
                f"{demo.command('scene', args)}"
            )
    car = sources[parts[0]]
    try:
        speed = walk_speed(car, vehicle)
    except ValueError as exc:
        raise SystemExit(f"error: {args.scene.name}: {exc}") from None
    rate = int(room.scene.simulation.sample_rate)
    duration = pass_s(car)
    # Each part is played alone below, where the rendering scales it against itself
    # alone: here each gets its sound power against the loudest part's, as one
    # rendering of them all would give it.
    gains = power_gains({part: sources[part].sound_power_db for part in parts}, parts)
    sent, trails, fed = {}, {}, {}
    for part in parts:
        x = feed(demo, args, rate, duration, vehicle, part)
        fed[part] = x
        gain = gains[part]
        sent[part] = x.scaled(gain) if not isinstance(x, np.ndarray) else (x * gain, rate)
        # What follows the source along its walk (a train's bogies and cars) plays as it
        # does, its signal scaled as the source's.
        trail = vehicle.trail(args, part, x, rate)
        behind = tuple((lag, y * gain, off) for lag, y, off in trail.behind)
        trails[part] = Trail(trail.lead_s, behind)
    longest = max(t.length_s for t in trails.values())
    rooms = {part: alone(room, part) for part in parts} if len(parts) > 1 else {parts[0]: room}
    fmt = format_of(args, room.scene)
    names = chosen_order(args, listeners_of(room))
    if not names:
        raise SystemExit(f"error: every listener of {args.scene.name} is left out")
    c = room.speed_of_sound()
    mixes, heard, points = {}, {}, None
    for name in names:
        played, passes = [], []
        for part in parts:
            print(f"simulating the {part} every {args.step_m:g} m along its walk, heard at {name}")
            walked = simulate_walk(
                rooms[part], part, listener=name, step_m=float(args.step_m), progress=print
            )
            trail = trails[part]
            more = f" and {len(trail.behind)} more behind it" if trail.behind else ""
            print(f"playing the {part}{more} through {len(walked.names)} places with Doppler")
            wet = render_walk(
                walked,
                {part: sent[part]},
                fmt,
                sample_rate=rate,
                followers=[(lag, y) for lag, y, _ in trail.behind],
            )
            lead = int(round(trail.lead_s * rate))
            played.append(np.concatenate([np.zeros((lead, wet.shape[1])), wet]))
            passing = pass_by(walked)
            loud = vehicle.sounding(args, part, fed[part], rate, walked.points.times)
            if loud is not None:
                passing = sounded(passing, loud)
            passes.append(trailed(passing, trail, longest))
            points = walked.points if part == parts[0] else points
        mixes[name] = summed(played)
        level = heard_together(passes)
        position = next(lis for lis in room.scene.listeners if lis.name == name).position
        distances = np.linalg.norm(points.positions - np.asarray(position), axis=1)
        heard[name] = {
            "heard_s": [round(float(t), 4) for t in level.times],
            "levels_a_db": [round(float(v), 2) for v in level.levels_a],
            "max_a_db": round(level.max_a, 2),
            "max_s": round(level.max_time, 3),
            "sel_a_db": round(level.exposure_a, 2),
            "leq_a_db": round(level.equivalent_a, 2),
            "closest_m": round(float(distances.min()), 2),
            "pitch_semitones": round(
                pitch_change(points, position, None if vehicle.timed else speed, c), 3
            ),
        }
        if len(parts) > 1:
            heard[name]["parts"] = {
                part: {"max_a_db": round(p.max_a, 2), "sel_a_db": round(p.exposure_a, 2)}
                for part, p in zip(parts, passes, strict=True)
            }
        print(
            f"{name}: L_Amax {level.max_a:.1f} dB(A), SEL {level.exposure_a:.1f} dB(A), "
            f"closest {distances.min():.1f} m"
        )
    peak = max(float(np.abs(m).max(initial=0.0)) for m in mixes.values())
    gain = 10.0 ** (-1.0 / 20.0) / peak if peak > 0 else 1.0
    out = args.work / "walk"
    seats = {}
    for name, mix in mixes.items():
        mix = mix * gain
        write_wav(out / f"{name}.{args.format}.wav", mix, rate)
        seats[name] = {
            "listener": name,
            "lag_s": 0.0,
            "rms_db": level_db(mix, rate),
            "loudness_db": loudness_db(mix, rate),
            "length_s": len(mix) / rate,
        }
    label = pipeline.output_label(args)
    pipeline.write_json(
        out / pipeline.SEATS,
        {"rate": rate, "format": args.format, "label": label, "gain": gain, "seats": seats},
    )
    made = vehicle.info(args, room.scene, speed * 3.6)
    pipeline.write_json(
        out / WALK,
        {
            "plays": played_by(args, vehicle),
            "label": label,
            "step_m": float(args.step_m),
            "source": noun,
            "parts": list(parts),
            "speed_kmh": round(speed * 3.6, 4),
            **vehicle.facts(args, made),
            "speed_of_sound": round(c, 3),
            "times": [round(float(t), 4) for t in points.times],
            "positions": np.round(points.positions, 4).tolist(),
            "listeners": heard,
        },
    )
    print(
        f"wrote {out}: {len(names)} listeners, the {vehicle.noun} {duration:.1f} s along its walk"
    )


def walk_done(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> bool:
    """Whether the passes are there, made from the scene, the vehicle's sound, the step
    and the format of this run, newer than what they are made from, and at every
    listener the run plays (:func:`heard_by`)."""
    path = args.work / "walk" / WALK
    if not path.exists() or not args.scene.exists():
        return False
    made = path.stat().st_mtime
    feeds = [
        args.scene,
        *(
            car_input(args, vehicle, part) or vehicle.sound_path(args, part)
            for part in vehicle.parts(args)
        ),
    ]
    if any(f.exists() and f.stat().st_mtime > made for f in feeds):
        return False
    facts = pipeline.read_json(path)
    try:
        plays = played_by(args, vehicle)
    except SystemExit:
        return False
    return (
        facts.get("plays") == plays
        and facts.get("label") == pipeline.output_label(args)
        and facts.get("step_m") == float(args.step_m)
        and set(heard_by(args)) <= set(facts.get("listeners", {}))
    )


def heard_by(args: argparse.Namespace) -> list[str]:
    """The listeners a run plays: those ``--listeners`` names, in its order, or every
    listener in use of the demo's scene, in its order."""
    names = listener_names(getattr(args, "listeners", None))
    return listeners_in(args.scene) if names is None else names


# --------------------------------------------------------------------------- the video


def stops_of(facts: dict, seats: dict, hint: str) -> list[pipeline.TourStop]:
    """One stop per listener of ``facts``: its whole pass, turned up to the loudness of
    the loudest (K-weighted), GAP_S of silence between two."""
    names = list(facts["listeners"])
    pipeline.missing_mixes(seats, names, hint)
    gains = pipeline.gains_of(seats, names)
    stops, at = [], 0.0
    for name in names:
        length = float(seats["seats"][name]["length_s"])
        stops.append(
            pipeline.TourStop(
                name, at, 0.0, length, gains[name], joined_before=False, joined_after=False
            )
        )
        at += length + GAP_S
    return stops


def heard_only(args: argparse.Namespace, facts: dict, hint: str) -> dict:
    """``facts`` (walk.json) with the listeners ``--listeners`` names alone, in its
    order; as they are when it names none. Stop with ``hint`` when one has no pass."""
    names = listener_names(getattr(args, "listeners", None))
    if names is None:
        return facts
    heard = facts["listeners"]
    missing = [name for name in names if name not in heard]
    if missing:
        raise SystemExit(f"error: no pass at {missing[0]}; run: {hint}")
    return facts | {"listeners": {name: heard[name] for name in names}}


def credit_of(args: argparse.Namespace, facts: dict, vehicle: Vehicle = THE_CAR) -> str:
    """The stills' credit line: how the vehicle's sound was made, or the file it plays."""
    return vehicle.credits(
        facts, {part: car_input(args, vehicle, part) for part in vehicle.parts(args)}
    )


def video(demo: Demo, args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> None:
    """tour.wav (each listener's pass in turn, turned up to one loudness, the loudest peak
    at -1 dBFS) and tour.mp4: the scene from above with the vehicle going past, the level
    heard over time under it, and what is heard, with a scrolling spectrogram (ffmpeg,
    unless ``--no-mp4``); a still of each listener at its loudest, and the thumbnail."""
    from acoustic_spaces.demos import passby_pictures, pictures
    from acoustic_spaces.room import outline_edges
    from acoustic_spaces.walkthrough import knots

    folder = args.work / "walk"
    path = folder / WALK
    if not path.exists():
        raise SystemExit(f"error: no {path}; run first: {demo.command('walk', args)}")
    facts = pipeline.read_json(path)
    seats = pipeline.seats_of(folder, demo.command("walk", args))
    wanted = pipeline.output_label(args)
    if seats.get("label") != wanted:
        raise SystemExit(
            f"error: the passes in {folder} are mixed {pipeline.label_text(seats.get('label'))}"
            f", and this run asks for them {pipeline.label_text(wanted)}: give the video the "
            f"--format and --hrtf the walk had, or mix the passes again: "
            f"{demo.command('walk', args)}"
        )
    facts = heard_only(args, facts, demo.command("walk", args))
    stops = stops_of(facts, seats, demo.command("walk", args))
    audio = pipeline.tour_audio(stops, folder, seats, crossfade_s=0.0, fade_out_s=0.0)
    write_wav(args.work / "tour.wav", audio, seats["rate"])
    print(f"wrote {args.work / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    room = room_of(demo, args)
    data = room.scene.to_data()
    tool = pictures.settings_lines(demo, args, data)
    tool[0] += f"; the {vehicle.noun} simulated every {facts['step_m']:g} m, with Doppler"
    drawing = passby_pictures.Pictures(
        data,
        outline_edges(room)[0],
        facts,
        seats["seats"],
        stops,
        path=knots(car_of(room.scene, vehicle)),
        credit=credit_of(args, facts, vehicle),
        tool=tool,
        fmt=args.format,
    )
    pictures.save(drawing.thumbnail(), args.work / "thumbnail.png")
    moments = [(stop.key, stop.at + facts["listeners"][stop.key]["max_s"]) for stop in stops]
    pipeline.animation(
        args.work,
        len(audio) / seats["rate"],
        drawing.frame,
        args.work / "tour.wav",
        stills=moments,
        no_mp4=args.no_mp4,
        chapters=[f"{pictures.mmss(stop.at)} {stop.key.replace('_', ' ')}" for stop in stops],
    )


# --------------------------------------------------------------------------- the demo


def moving(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> list[str]:
    """Who the demo plays moving: the vehicle's sources it plays, when the example gives
    them a walk that is on."""
    try:
        problem = car_problem(source_scene(args), args.source.name, vehicle)
        return [] if problem else list(vehicle.parts(args))
    except (Exception, SystemExit):
        return []


def problems(demo: Demo, args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> str | None:
    """The vehicle must go: the example's scene gives it a walk that is on, at one
    steady speed unless ``--speed`` gives one, and the speed, the step and the vehicle's
    own numbers are positive."""
    from acoustic_spaces.scene import Scene

    numbers = ["speed", *(o.name for o in vehicle.options() if o.kind == "float"), "step_m"]
    for option in numbers:
        value = getattr(args, option, None)
        if value is not None and not (math.isfinite(float(value)) and float(value) > 0.0):
            return f"{demo.option(option).option} is a positive number, not {value!r}"
    source = Path(demo.resolved(args).source)
    if not source.exists():
        return f"no {source}"
    try:
        scene = Scene.from_yaml(source)
    except Exception as exc:  # the scene's own message
        return f"{source.name}: {exc}"
    problem = car_problem(scene, source.name, vehicle)
    if problem is None and problems_of_speed(args, scene, vehicle):
        return f"{source.name}: {problems_of_speed(args, scene, vehicle)}"
    return problem


def summary(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> str | None:
    """How the vehicle goes in a run with ``args`` (paths resolved), in a line: how
    far, how fast (km/h and m/s, and whose speed) and so how long the pass and its sound
    last; None when it cannot go (the check says why)."""
    from acoustic_spaces.walkthrough import knots

    try:
        scene = source_scene(args)
        if car_problem(scene, args.source.name, vehicle) or problems_of_speed(args, scene, vehicle):
            return None
        if vehicle.timed:
            return vehicle.how_it_goes(args, scene)
        own = vehicle.own_speed_kmh(args)
        path = knots(driving(car_of(scene, vehicle), forced_speed(args, vehicle)))
    except (Exception, SystemExit):
        return None
    speed = path.length_m / path.duration_s
    noun, way = vehicle.noun, vehicle.way.removeprefix("the ").removeprefix("its ")
    if getattr(args, "speed", None) is not None:
        whose = "--speed"
    elif own is not None:
        whose = f"the {noun}'s own"
    else:
        whose = "as its walk in the scene goes"
    edit = f"the {way}" if own is not None else f"the {way}, and its speed"
    return (
        f"The {noun} {vehicle.goes} {path.length_m:.1f} m along its walk at "
        f"{3.6 * speed:.1f} km/h ({speed:.2f} m/s, {whose}): the pass, and the {noun}'s "
        f"sound, last {path.duration_s:.1f} s. --speed {vehicle.goes} it faster or slower "
        f"along the same {way}; Edit movement… changes {edit}."
    )


def problems_of_speed(args: argparse.Namespace, scene, vehicle: Vehicle = THE_CAR) -> str | None:
    """What keeps the vehicle of ``scene`` from one speed: a ``--speed`` that is not
    positive, or without one keyframes that are not steady; nothing for a
    :attr:`~Vehicle.timed` vehicle, which goes as its own numbers say."""
    if getattr(args, "speed", None) is not None:
        speed = float(args.speed)
        return None if math.isfinite(speed) and speed > 0.0 else "--speed is not positive"
    if vehicle.own_speed_kmh(args) is not None or vehicle.timed:
        return None
    try:
        walk_speed(car_of(scene, vehicle), vehicle)
    except ValueError as exc:
        return str(exc)
    return None


def demo(
    script: Path,
    example: str,
    title: str,
    description: str,
    vehicle: Vehicle | Sequence[Vehicle] = THE_CAR,
) -> Demo:
    """The pass-by demo of ``example``, run by ``script`` (its ``passby_demo.py``), with
    ``vehicle`` going past (the car by default), or one of several (``--vehicle``, the
    first by default), each the example's source of its name."""
    vehicles = (vehicle,) if isinstance(vehicle, Vehicle) else tuple(vehicle)
    if not vehicles or len({v.sound_step for v in vehicles}) > 1:
        raise ValueError("a pass-by demo has a vehicle at least, all with one sound step")
    by_name = {v.name: v for v in vehicles}

    def pick(args: argparse.Namespace) -> Vehicle:
        name = getattr(args, "vehicle", None) or vehicles[0].name
        if name not in by_name:
            raise SystemExit(f"error: --vehicle is one of {', '.join(by_name)}, not {name!r}")
        return by_name[name]

    first = vehicles[0]
    several = len(vehicles) > 1
    noun = first.fleet_noun if several else first.name
    texts = first.texts()
    if several:
        texts = first.fleet_texts() or road_texts(
            "vehicle", "NAME", f"demo/NAME/NAME.wav (the {first.name}'s, …)"
        )
        texts["input"] = texts["input"].replace(
            "as NAME=WAV", f"as NAME=WAV for the vehicle of --vehicle ({first.name}=WAV, …)"
        )
        carried = [f"{part}=WAV" for v in vehicles for part in v.every_part()[1:]]
        if carried:
            texts["input"] += f"; what a vehicle carries plays one too ({', '.join(carried)})"
    own: list[Option] = []
    for each in vehicles:
        for option in each.options():
            if all(o.name != option.name for o in own):
                own.append(option)
    fleet = [engine for v in vehicles for engine in v.engines()]
    own = [dataclasses.replace(o, help=rpm_help(fleet)) if o.name == "rpm" else o for o in own]
    if several:
        # An option of some of the vehicles alone (the police car's --siren) matters
        # only with --vehicle at one of them; the editor shows it then alone.
        having = {
            o.name: tuple(v.name for v in vehicles if any(m.name == o.name for m in v.options()))
            for o in own
        }
        own = [
            o
            if len(having[o.name]) == len(vehicles)
            else dataclasses.replace(o, only=("vehicle", having[o.name]))
            for o in own
        ]
    choice = [
        Option(
            "vehicle",
            "what goes past: "
            + "; ".join(f"{v.name}, {v.title}" for v in vehicles)
            + " (each the example's source of its name, with its walk)",
            first.name,
            "choice",
            tuple(by_name),
            steps=(first.sound_step, "scene", "walk"),
        )
    ]
    # A vehicle whose own numbers time its walk (an aircraft) goes at no --speed.
    speed = (
        []
        if all(v.timed for v in vehicles)
        else [
            Option(
                "speed",
                texts["speed"],
                None,
                "float",
                steps=(first.sound_step, "scene", "walk"),
                unit="km/h",
                unset=(
                    f"the {first.noun}'s own"
                    if all(v.own_speed for v in vehicles)
                    else "as in the scene"
                ),
            )
        ]
    )
    made = Demo(
        name=first.demo_name,
        example=example,
        title=title,
        description=description,
        script=Path(script),
        steps=[],
        options=[
            *(choice if several else []),
            *speed,
            *own,
            Option(
                "step_m",
                f"how far apart the {noun} is simulated along its walk (m)",
                first.step_m,
                "float",
                steps=("walk",),
                flag="--step",
                unit="m",
            ),
            Option(
                "source",
                "the example's scene",
                Path("scene.yaml"),
                "file",
                steps=(first.sound_step, "scene"),
                flag="--from",
                filter="Scene files (*.yaml *.yml)",
            ),
            Option(
                "scene",
                f"the scene with the {noun} {first.going} alone",
                Path("scene_demo.yaml"),
                "path",
            ),
            *[o for o in common_options() if o.name != "again"],
            dataclasses.replace(input_option(("walk", "video")), help=texts["input"]),
            listeners_option(
                ("walk", "video"),
                help=f"only these listeners, in this order, comma separated (default: every "
                f"listener in use, in the scene's order): the {noun} is simulated along its "
                "walk and heard at those alone, and the video plays their passes in turn",
            ),
        ],
        scene_from="scene",
        problems=lambda shown, args: problems(shown, args, pick(args)),
        plays=lambda args: plays(args, pick(args)),
        moving=lambda args: moving(args, pick(args)),
        summary=lambda args: summary(args, pick(args)),
        # The scene step keeps the example's listeners as they are.
        listeners=lambda args: listeners_in(args.source),
    )
    made.steps = [
        Step(
            first.sound_step,
            lambda args: car(made, args, pick(args)),
            texts["sound"],
            lambda args: car_done(args, pick(args)),
        ),
        Step(
            "scene",
            lambda args: scene(made, args, pick(args)),
            f"scene_demo.yaml: the {noun} alone, "
            + ("along its flight" if first.timed else "at --speed")
            + ", with Doppler",
            lambda args: scene_done(args, pick(args)),
        ),
        Step(
            "walk",
            lambda args: walk(made, args, pick(args)),
            f"simulate the {noun} along its walk at every listener and play it with Doppler",
            lambda args: walk_done(args, pick(args)),
        ),
        Step(
            "video",
            lambda args: video(made, args, pick(args)),
            "tour.wav, tour.mp4 and the stills in the work folder",
        ),
    ]
    return made


__all__ = [
    "BUS",
    "CAR",
    "FADE_S",
    "GAP_S",
    "MOTORCYCLE",
    "ROAD_VEHICLES",
    "SPEED_KMH",
    "STEADY",
    "STEP_M",
    "THE_CAR",
    "TRUCK",
    "RoadVehicle",
    "Trail",
    "Vehicle",
    "car_problem",
    "demo",
    "driving",
    "faded",
    "forced_speed",
    "pitch_change",
    "road_texts",
    "rpm_help",
    "trailed",
    "walk_speed",
]
