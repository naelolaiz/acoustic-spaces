"""A vehicle going past, heard at each listener of an outdoor example, its pitch falling
as it goes by (Doppler): the pass-by demos of the roadside and the street examples
(``passby_demo.py`` beside each scene), where a car drives past, and of the
helicopter_flyover example (``flyover_demo.py``), where a helicopter flies over; one
module for all of them, each with its :class:`Vehicle`.

The car is the example's source ``car``: a point source 0.05 m above the road, where
CNOSSOS-EU places a vehicle's sound, with a walk along the road with Doppler, left out of
the example's own simulations (``enabled: false``), which keep the traffic as a still
line source. Its walk goes through points at a speed (``to``, ``speed_m_s``) or through
keyframes at one steady speed, as CNOSSOS-EU gives a car's sound at a steady speed. The
demo writes ``scene_demo.yaml`` beside its script: the car in use, along its walk at the
walk's own speed or at ``--speed`` (through the keyframes' places, for keyframes), with
CNOSSOS-EU's sound power of a passenger car at that speed and the scene's air
temperature (:mod:`acoustic_spaces.demos.car`), and every other source left out, so that
the car is heard alone. It plays a sound made to that power
(``demo/car/car.wav``: tyre noise and a four-cylinder engine at ``--rpm``), or the dry
recording ``--input car=WAV`` gives. A helicopter goes the same way, with the sound its
rotors send each way (:mod:`acoustic_spaces.demos.helicopter`).

Each listener in use is simulated with the vehicle every ``--step`` along its walk
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
from pathlib import Path

import numpy as np
import yaml

from acoustic_spaces.demos import Demo, Option, Step, common_options, input_option, pipeline
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


class Vehicle:
    """What a pass-by demo plays past the listeners: the example's source of that name,
    the words the demo says it with, and how its sound and its sound power are made.
    This one is the car of the roadside and the street examples."""

    #: The example's source, which goes past, and the demo's short name.
    name, demo_name = CAR, "passby"
    #: How it goes: "drive", "drives" and "driving past", along "the road".
    go, goes, going, way = "drive", "drives", "driving past", "the road"
    #: Why its keyframes must keep one speed.
    steady = "CNOSSOS-EU gives a car's sound power at one steady speed"
    #: How far apart its places are simulated by default (m).
    step_m = STEP_M
    #: Its sound's file in the work folder's folder of its name, and what that is made of.
    sound_file, info_file = CAR_FILE, CAR_INFO

    def options(self) -> list[Option]:
        """The options of its own, for the step that makes its sound."""
        return [
            Option(
                "rpm",
                "how fast the car's engine turns (revolutions per minute), for its made sound",
                2000.0,
                "float",
                steps=(self.name,),
                unit="rpm",
            )
        ]

    def check(self, args: argparse.Namespace) -> None:
        """Stop with an error when its own options cannot make its sound."""
        if float(args.rpm) <= 0.0:
            raise SystemExit(f"error: --rpm is a positive number, not {args.rpm!r}")

    def problem(self, source) -> str | None:
        """What else keeps ``source`` (its scene source, with a walk that is on) from
        going past: nothing, for a car."""
        return None

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sound is made of, at ``speed_kmh`` in ``scene``: the speed, the
        engine, the air's temperature, the length and CNOSSOS-EU's sound power."""
        from acoustic_spaces.demos.car import light_vehicle_power

        power = light_vehicle_power(speed_kmh, scene.air.temperature_c)
        return {
            "speed_kmh": round(speed_kmh, 6),
            "rpm": float(args.rpm),
            "temperature_c": scene.air.temperature_c,
            "rate": int(scene.simulation.sample_rate),
            "duration_s": round(pass_s(car_of(scene, self), args.speed), 6),
            "level_db": [round(float(v), 1) for v in power.total_db],
            "rolling_db": [round(float(v), 2) for v in power.rolling_db],
            "propulsion_db": [round(float(v), 2) for v in power.propulsion_db],
            "sound_power_db": round(power.sound_power_db, 2),
            "sound_power_a_db": round(power.sound_power_a_db, 2),
        }

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        """Write its sound, dry, into ``folder``, made to ``info``; what to print."""
        from acoustic_spaces.demos.car import car_sound, light_vehicle_power

        power = light_vehicle_power(info["speed_kmh"], info["temperature_c"])
        x = car_sound(power, info["rate"], info["duration_s"], rpm=info["rpm"])
        write_wav(folder / self.sound_file, x, info["rate"])
        return (
            f"{info['duration_s']:.1f} s of a car at {info['speed_kmh']:g} km/h, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power"
        )

    def edit(self, source: dict, info: dict, args: argparse.Namespace) -> None:
        """Give ``source`` (its entry of the demo's scene, as data) its sound power."""
        source["level_db"] = info["level_db"]
        source.pop("spectrum", None)

    def made(self, args: argparse.Namespace, rate: int, duration_s: float):
        """What it plays along its pass at ``rate``: its made sound, read back, and
        faded in and out over FADE_S."""
        path = args.work / self.name / self.sound_file
        if not path.exists():
            return None
        x, r = read(path)
        return resampled(mono(x), r, rate)

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        """What it plays unless ``--input`` gives it a file, ``speed`` its speed in words."""
        return f"a car {speed}made to CNOSSOS-EU's sound power"

    def played_by(self, args: argparse.Namespace) -> str:
        """Its made sound, as walk.json keeps it."""
        return f"car {round(speed_of(args, self), 4):g} km/h {args.rpm:g} rpm"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        """What walk.json says of it: its engine, its sound power and the stills' line."""
        return {
            "rpm": float(args.rpm),
            "sound_power_a_db": info["sound_power_a_db"],
            "about": f"A car at {info['speed_kmh']:g} km/h, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power (CNOSSOS-EU)",
        }

    def credit(self, facts: dict) -> str:
        """The stills' credit line for its made sound."""
        return (
            f"Car's sound made up: tyre noise and a four-cylinder engine at "
            f"{facts['rpm']:g} rpm, each with the octave spectrum of CNOSSOS-EU's passenger "
            f"car at {facts['speed_kmh']:g} km/h (Directive (EU) 2015/996); faded in and out "
            f"over {FADE_S:g} s"
        )

    def texts(self) -> dict[str, str]:
        """The help of the demo's options and steps that speak of it."""
        return {
            "speed": "how fast the car drives along its walk (km/h; m/s times 3.6), which sets "
            "how long its pass and its sound last (the walk's length over the speed) and "
            "its sound power; by default as its walk in the scene goes: its speed along "
            "points, or the one steady speed of its keyframes (whose places it goes "
            "through at --speed when given)",
            "input": "a WAV file the car plays instead of its made sound, as car=WAV: a dry "
            "recording (on board or at the wheel; one of a car passing has its Doppler, "
            "the road and the distance in it already), mixed down to mono, at least as "
            "long as the pass; the captions' levels are CNOSSOS-EU's car at --speed",
            "sound": "demo/car/car.wav: the car's sound, made to CNOSSOS-EU",
            "header": "# The example with its car driving past alone, at the speed and with "
            "the sound\n# power of the pass-by demo (CNOSSOS-EU), written by passby_demo.py.\n",
        }


#: The car of the roadside and the street examples.
THE_CAR = Vehicle()


def walk_speed(car, vehicle: Vehicle = THE_CAR) -> float:
    """The speed the walk of ``car`` (a scene source) goes at (m/s): its ``speed_m_s``
    along points, or along keyframes the one steady speed they keep; a ValueError says
    when keyframes go at different speeds or wait."""
    from acoustic_spaces.walkthrough import knots

    if car.walk.keyframes is None:
        return float(car.walk.speed_m_s)
    path = knots(car)
    speeds = np.linalg.norm(np.diff(path.positions, axis=0), axis=1) / np.diff(path.times)
    slow, fast = float(speeds.min()), float(speeds.max())
    if slow <= 0.0 or fast > slow * (1.0 + STEADY):
        raise ValueError(
            f"the {vehicle.name}'s keyframes go at {3.6 * slow:.1f} to {3.6 * fast:.1f} km/h, "
            f"but {vehicle.steady}: time them for one speed (or give its walk points at a "
            f"speed), or {vehicle.go} it through their places at one speed with --speed"
        )
    return path.length_m / path.duration_s


def speed_of(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> float:
    """The speed the vehicle goes at (km/h): ``--speed``, checked positive and finite,
    or without it the speed of its walk in the example's scene (:func:`walk_speed`)."""
    if args.speed is not None:
        speed = float(args.speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise SystemExit(f"error: --speed is a positive number of km/h, not {args.speed!r}")
        return speed
    scene = source_scene(args)
    problem = car_problem(scene, args.source.name, vehicle)
    if problem:
        raise SystemExit(f"error: {problem}")
    try:
        return 3.6 * walk_speed(car_of(scene, vehicle), vehicle)
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
    return f"{name}: {problem}" if problem else None


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


def car_input(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> Path | None:
    """The file ``--input NAME=WAV`` gives the vehicle to play, or None."""
    return (getattr(args, "inputs", None) or {}).get(vehicle.name)


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
    :func:`acoustic_spaces.demos.car.car_sound`), and what it is made of."""
    vehicle.check(args)
    info = car_info(args, vehicle)
    folder = args.work / vehicle.name
    said = vehicle.make(args, info, folder)
    pipeline.write_json(folder / vehicle.info_file, info)
    print(f"wrote {folder / vehicle.sound_file}: {said}")


def car_done(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> bool:
    """Whether the vehicle's sound is there and made for the arguments and the scene as
    it is now, or not needed (``--input NAME=...``)."""
    if car_input(args, vehicle) is not None:
        return True
    folder = args.work / vehicle.name
    path = folder / vehicle.info_file
    if not path.exists() or not (folder / vehicle.sound_file).exists():
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
    return {vehicle.name: vehicle.plays(args, speed)}


def played_by(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> str:
    """What the vehicle plays, as walk.json keeps it."""
    file = car_input(args, vehicle)
    if file is not None:
        return str(file)
    return vehicle.played_by(args)


def feed(demo: Demo, args: argparse.Namespace, rate: int, duration_s: float, vehicle=THE_CAR):
    """What the vehicle plays along its pass, at ``rate``: its made sound
    (:meth:`Vehicle.made`), or the file ``--input NAME=WAV`` gives it (mixed down to
    mono, cut to the pass, at the made car's RMS), faded in and out over FADE_S."""
    from acoustic_spaces.demos.car import SOUND_RMS
    from acoustic_spaces.emission import Emission, Faded

    n = int(round(duration_s * rate))
    fade = min(int(FADE_S * rate), n // 2)
    file = car_input(args, vehicle)
    if file is None:
        x = vehicle.made(args, rate, duration_s)
        if x is None:
            path = args.work / vehicle.name / vehicle.sound_file
            raise SystemExit(f"error: no {path}; run first: {demo.command(vehicle.name, args)}")
        if isinstance(x, Emission):
            if x.length < n:
                raise SystemExit(
                    f"error: the {vehicle.name}'s sound is shorter than its pass; run: "
                    f"{demo.command(vehicle.name, args)}"
                )
            return Faded(x, fade, n)
    else:
        x, r = read(file)
        x = resampled(mono(x), r, rate)
        if len(x) < n:
            raise SystemExit(
                f"error: {file.name} is {len(x) / rate:.1f} s long; the {vehicle.name} takes "
                f"{duration_s:.1f} s along its walk"
            )
        rms = gated_rms(x[:n], rate)
        if rms <= 0.0:
            raise SystemExit(f"error: {file} is silent")
        x = x * (SOUND_RMS / rms)
    x = np.array(x[:n], dtype=float)
    if len(x) < n:
        raise SystemExit(
            f"error: the {vehicle.name}'s sound is shorter than its pass; run: "
            f"{demo.command(vehicle.name, args)}"
        )
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    x[:fade] *= ramp
    x[n - fade :] *= ramp[::-1]
    return x


# --------------------------------------------------------------------------- the scene


def scene(demo: Demo, args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> None:
    """``--scene`` (scene_demo.yaml) from ``--from`` (scene.yaml): the vehicle in use,
    along its walk with Doppler, at its own speed or at ``--speed`` (:func:`driving`),
    and its sound power at that speed (:meth:`Vehicle.edit`), every other source left
    out."""
    info = car_info(args, vehicle)
    data = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    data["name"] = f"{data.get('name', 'scene')}: a {vehicle.name} passing"
    car = car_of(source_scene(args), vehicle)
    for source in data["sources"]:
        if source["name"] == vehicle.name:
            source["enabled"] = True
            vehicle.edit(source, info, args)
            walk = source["walk"]
            if args.speed is not None:
                walk["speed_m_s"] = round(info["speed_kmh"] / 3.6, 6)
                if walk.get("keyframes"):
                    walk["to"] = [list(place) for place in places_of(car)]
                    walk.pop("keyframes")
            walk["doppler"] = True
        else:
            source["enabled"] = False
    pipeline.write_scene(args.scene, data, vehicle.texts()["header"], args.source.parent)
    print(
        f"wrote {args.scene}: the {vehicle.name} at {info['speed_kmh']:g} km/h, "
        f"{info['sound_power_db']:.1f} dB of sound power"
    )


def scene_done(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> bool:
    """Whether scene_demo.yaml is there, newer than the example's scene, and for this
    speed."""
    from acoustic_spaces.scene import Scene

    if not args.scene.exists():
        return False
    if args.source.exists() and args.scene.stat().st_mtime < args.source.stat().st_mtime:
        return False
    try:
        car = car_of(Scene.from_yaml(args.scene), vehicle)
        made = 3.6 * walk_speed(car, vehicle)
    except (Exception, SystemExit):  # a scene of another kind: written again
        return False
    return car.walk.doppler and abs(made - speed_of(args, vehicle)) < 1e-4 * made


def room_of(demo: Demo, args: argparse.Namespace):
    from acoustic_spaces import Room

    if not args.scene.exists():
        raise SystemExit(f"error: no {args.scene}; run first: {demo.command('scene', args)}")
    return Room.from_yaml(args.scene)


# --------------------------------------------------------------------------- the pass


def pitch_change(points, listener, speed_m_s: float, c: float) -> float:
    """How far the pitch of the vehicle's direct sound falls at ``listener`` (a position)
    from where its walk starts to where it ends (semitones): heard at c / (c - v_r), v_r
    the vehicle's speed towards the listener, along its first leg and its last."""
    positions = np.asarray(points.positions, dtype=float)
    at = np.asarray(listener, dtype=float)

    def towards(k: int, leg: np.ndarray) -> float:
        to = at - positions[k]
        distance = float(np.linalg.norm(to))
        heading = leg / max(float(np.linalg.norm(leg)), 1e-12)
        return speed_m_s * float(np.dot(heading, to)) / max(distance, 1e-12)

    first = towards(0, positions[1] - positions[0])
    last = towards(-1, positions[-1] - positions[-2])
    return 12.0 * math.log2((c - last) / (c - first))


def listeners_of(room) -> list[str]:
    return [lis.name for lis in room.scene.listeners if lis.in_use]


def walk(demo: Demo, args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> None:
    """Each listener in use: the vehicle simulated every ``--step`` along its walk, its
    sound played through those places with Doppler, and the level along the way; one
    gain for every listener (the loudest peak at -1 dBFS), into ``walk/``: a WAV per
    listener, seats.json and walk.json."""
    from acoustic_spaces.demos.scene_demo import format_of
    from acoustic_spaces.walkthrough import pass_by, render_walk, simulate_walk

    noun = vehicle.name
    room = room_of(demo, args)
    car = car_of(room.scene, vehicle)
    problem = car_problem(room.scene, args.scene.name, vehicle)
    if problem:
        raise SystemExit(f"error: {problem}")
    if not car.walk.doppler:
        raise SystemExit(
            f"error: the {noun} in {args.scene.name} has no Doppler; run: "
            f"{demo.command('scene', args)}"
        )
    try:
        speed = walk_speed(car, vehicle)
    except ValueError as exc:
        raise SystemExit(f"error: {args.scene.name}: {exc}") from None
    rate = int(room.scene.simulation.sample_rate)
    duration = pass_s(car)
    x = feed(demo, args, rate, duration, vehicle)
    sent = x if not isinstance(x, np.ndarray) else (x, rate)
    fmt = format_of(args, room.scene)
    names = listeners_of(room)
    if not names:
        raise SystemExit(f"error: every listener of {args.scene.name} is left out")
    c = room.speed_of_sound()
    mixes, heard, points = {}, {}, None
    for name in names:
        print(f"simulating the {noun} every {args.step_m:g} m along its walk, heard at {name}")
        walked = simulate_walk(room, noun, listener=name, step_m=float(args.step_m), progress=print)
        print(f"playing the {noun} through {len(walked.names)} places with Doppler")
        mixes[name] = render_walk(walked, {noun: sent}, fmt, sample_rate=rate)
        level = pass_by(walked)
        points = walked.points
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
            "pitch_semitones": round(pitch_change(points, position, speed, c), 3),
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
            "speed_kmh": round(speed * 3.6, 4),
            **vehicle.facts(args, made),
            "speed_of_sound": round(c, 3),
            "times": [round(float(t), 4) for t in points.times],
            "positions": np.round(points.positions, 4).tolist(),
            "listeners": heard,
        },
    )
    print(f"wrote {out}: {len(names)} listeners, the {noun} {duration:.1f} s along its walk")


def walk_done(args: argparse.Namespace, vehicle: Vehicle = THE_CAR) -> bool:
    """Whether the passes are there, made from the scene, the vehicle's sound, the step
    and the format of this run, and newer than what they are made from."""
    path = args.work / "walk" / WALK
    if not path.exists() or not args.scene.exists():
        return False
    made = path.stat().st_mtime
    sound = args.work / vehicle.name / vehicle.sound_file
    feeds = [args.scene, car_input(args, vehicle) or sound]
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
    )


# --------------------------------------------------------------------------- the video


def stops_of(facts: dict, seats: dict, hint: str) -> list[pipeline.TourStop]:
    """One stop per listener: its whole pass, turned up to the loudness of the loudest
    (K-weighted), GAP_S of silence between two."""
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


def credit_of(args: argparse.Namespace, facts: dict, vehicle: Vehicle = THE_CAR) -> str:
    """The stills' credit line: how the vehicle's sound was made, or the file it plays."""
    file = car_input(args, vehicle)
    if file is not None:
        return f"Playing {vehicle.name}: {file.name}, faded in and out over {FADE_S:g} s"
    return vehicle.credit(facts)


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
    if seats.get("label") != pipeline.output_label(args):
        raise SystemExit(
            f"error: the passes in {folder} are of another format; run: "
            f"{demo.command('walk', args)}"
        )
    stops = stops_of(facts, seats, demo.command("walk", args))
    audio = pipeline.tour_audio(stops, folder, seats, crossfade_s=0.0, fade_out_s=0.0)
    write_wav(args.work / "tour.wav", audio, seats["rate"])
    print(f"wrote {args.work / 'tour.wav'} ({len(audio) / seats['rate']:.1f} s)")
    room = room_of(demo, args)
    data = room.scene.to_data()
    tool = pictures.settings_lines(demo, args, data)
    tool[0] += f"; the {vehicle.name} simulated every {facts['step_m']:g} m, with Doppler"
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
    """Who the demo plays moving: the vehicle, when the example gives it a walk that is
    on."""
    try:
        problem = car_problem(source_scene(args), args.source.name, vehicle)
        return [] if problem else [vehicle.name]
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
        path = knots(driving(car_of(scene, vehicle), args.speed))
    except (Exception, SystemExit):
        return None
    speed = path.length_m / path.duration_s
    whose = "--speed" if args.speed is not None else "as its walk in the scene goes"
    noun, way = vehicle.name, vehicle.way.removeprefix("the ").removeprefix("its ")
    return (
        f"The {noun} {vehicle.goes} {path.length_m:.1f} m along its walk at "
        f"{3.6 * speed:.1f} km/h ({speed:.2f} m/s, {whose}): the pass, and the {noun}'s "
        f"sound, last {path.duration_s:.1f} s. --speed {vehicle.goes} it faster or slower "
        f"along the same {way}; Edit movement… changes the {way}, and its speed."
    )


def problems_of_speed(args: argparse.Namespace, scene, vehicle: Vehicle = THE_CAR) -> str | None:
    """What keeps the vehicle of ``scene`` from one speed: a ``--speed`` that is not
    positive, or without one keyframes that are not steady."""
    if args.speed is not None:
        speed = float(args.speed)
        return None if math.isfinite(speed) and speed > 0.0 else "--speed is not positive"
    try:
        walk_speed(car_of(scene, vehicle), vehicle)
    except ValueError as exc:
        return str(exc)
    return None


def demo(
    script: Path, example: str, title: str, description: str, vehicle: Vehicle = THE_CAR
) -> Demo:
    """The pass-by demo of ``example``, run by ``script`` (its ``passby_demo.py``), with
    ``vehicle`` going past (the car by default)."""
    noun, texts = vehicle.name, vehicle.texts()
    made = Demo(
        name=vehicle.demo_name,
        example=example,
        title=title,
        description=description,
        script=Path(script),
        steps=[],
        options=[
            Option(
                "speed",
                texts["speed"],
                None,
                "float",
                steps=(noun, "scene", "walk"),
                unit="km/h",
                unset="as in the scene",
            ),
            *vehicle.options(),
            Option(
                "step_m",
                f"how far apart the {noun} is simulated along its walk (m)",
                vehicle.step_m,
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
                steps=(noun, "scene"),
                flag="--from",
                filter="Scene files (*.yaml *.yml)",
            ),
            Option(
                "scene",
                f"the scene with the {noun} {vehicle.going} alone",
                Path("scene_demo.yaml"),
                "path",
            ),
            *[o for o in common_options() if o.name != "again"],
            dataclasses.replace(input_option(("walk", "video")), help=texts["input"]),
        ],
        scene_from="scene",
        problems=lambda shown, args: problems(shown, args, vehicle),
        plays=lambda args: plays(args, vehicle),
        moving=lambda args: moving(args, vehicle),
        summary=lambda args: summary(args, vehicle),
    )
    made.steps = [
        Step(
            noun,
            lambda args: car(made, args, vehicle),
            texts["sound"],
            lambda args: car_done(args, vehicle),
        ),
        Step(
            "scene",
            lambda args: scene(made, args, vehicle),
            f"scene_demo.yaml: the {noun} alone, at --speed, with Doppler",
            lambda args: scene_done(args, vehicle),
        ),
        Step(
            "walk",
            lambda args: walk(made, args, vehicle),
            f"simulate the {noun} along its walk at every listener and play it with Doppler",
            lambda args: walk_done(args, vehicle),
        ),
        Step(
            "video",
            lambda args: video(made, args, vehicle),
            "tour.wav, tour.mp4 and the stills in the work folder",
        ),
    ]
    return made


__all__ = [
    "CAR",
    "FADE_S",
    "GAP_S",
    "SPEED_KMH",
    "STEADY",
    "STEP_M",
    "THE_CAR",
    "Vehicle",
    "car_problem",
    "demo",
    "driving",
    "pitch_change",
    "walk_speed",
]
