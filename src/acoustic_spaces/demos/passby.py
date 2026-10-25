"""A car driving past, heard at each listener of an outdoor example, its pitch falling as
it goes by (Doppler): the pass-by demos of the roadside and the street examples
(``passby_demo.py`` beside each scene), one module for both.

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
recording ``--input car=WAV`` gives.

Each listener in use is simulated with the car every ``--step`` along its walk
(:func:`acoustic_spaces.walkthrough.simulate_walk`) and hears the car through those
places, each early path through its own delay as it changes and the late sound
crossfaded from place to place (:func:`~acoustic_spaces.walkthrough.render_walk`); the
level along the way, its maximum and the sound exposure level come from the same
simulation (:func:`~acoustic_spaces.walkthrough.pass_by`). The video plays one pass at
each listener in turn, the car moving on the plan and the level heard over time drawn
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
#: How far the legs of a car's keyframes may differ in speed and still be one steady
#: speed (a share of the slowest): 1 % is 0.2 dB of CNOSSOS-EU's rolling noise at most.
STEADY = 0.01
#: How far apart the places of the walk are simulated (m): through places 1 m apart a
#: car's sound at 1 kHz follows the exact field's waveform to within -54 dB, -57 dB a
#: quarter of a metre apart (validation/doppler.py).
STEP_M = 1.0
#: The car's sound fades in over its first FADE_S and out over its last, where the car
#: is farthest, so that it does not start or stop at once; and the silence between two
#: listeners' passes (s).
FADE_S = 1.0
GAP_S = 1.0
CAR_FILE = "car.wav"
CAR_INFO = "car.json"
WALK = "walk.json"


# --------------------------------------------------------------------------- the car


def walk_speed(car) -> float:
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
            f"the {CAR}'s keyframes go at {3.6 * slow:.1f} to {3.6 * fast:.1f} km/h, but "
            "CNOSSOS-EU gives a car's sound power at one steady speed: time them for one "
            "speed (or give its walk points at a speed), or drive it through their places "
            "at one speed with --speed"
        )
    return path.length_m / path.duration_s


def speed_of(args: argparse.Namespace) -> float:
    """The speed the car drives at (km/h): ``--speed``, checked positive and finite, or
    without it the speed of its walk in the example's scene (:func:`walk_speed`)."""
    if args.speed is not None:
        speed = float(args.speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise SystemExit(f"error: --speed is a positive number of km/h, not {args.speed!r}")
        return speed
    scene = source_scene(args)
    problem = car_problem(scene, args.source.name)
    if problem:
        raise SystemExit(f"error: {problem}")
    try:
        return 3.6 * walk_speed(car_of(scene))
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


def car_of(scene):
    """The car of ``scene`` (a :class:`~acoustic_spaces.scene.Scene`)."""
    car = next((s for s in scene.sources if s.name == CAR), None)
    if car is None:
        raise SystemExit(f"error: the scene has no source {CAR!r} to drive")
    return car


def car_problem(scene, name: str) -> str | None:
    """What keeps the car of ``scene`` (read from the file ``name``) from driving past:
    none, a line or a surface, no walk, or a walk that is off."""
    car = next((s for s in scene.sources if s.name == CAR), None)
    if car is None:
        return f"{name} has no source {CAR!r} to drive"
    if car.extended:
        return f"the {CAR} of {name} is a line or a surface: only a point source drives"
    if car.walk is None:
        return (
            f"the {CAR} has no walk in {name}: give it one along the road (walk: to:, or the "
            "editor's Movement panel)"
        )
    if not car.walk.enabled:
        return (
            f"the {CAR}'s walk is off in {name}: turn it on (Moves in the editor, or walk: "
            "enabled: true)"
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
    """How long the car takes along its walk, as it is or at ``speed_kmh`` (s)."""
    from acoustic_spaces.walkthrough import knots

    return float(knots(driving(car, speed_kmh)).duration_s)


def source_scene(args: argparse.Namespace):
    """The example's scene (``--from``)."""
    from acoustic_spaces.scene import Scene

    if not args.source.exists():
        raise SystemExit(f"error: no {args.source}")
    return Scene.from_yaml(args.source)


def car_input(args: argparse.Namespace) -> Path | None:
    """The file ``--input car=WAV`` gives the car to play, or None."""
    return (getattr(args, "inputs", None) or {}).get(CAR)


def car_info(args: argparse.Namespace) -> dict:
    """What ``car/car.wav`` is made of, for the arguments and the example's scene: the
    speed, the engine, the air's temperature, the length and CNOSSOS-EU's sound power."""
    from acoustic_spaces.demos.car import light_vehicle_power

    scene = source_scene(args)
    problem = car_problem(scene, args.source.name)
    if problem:
        raise SystemExit(f"error: {problem}")
    speed = speed_of(args)
    power = light_vehicle_power(speed, scene.air.temperature_c)
    return {
        "speed_kmh": round(speed, 6),
        "rpm": float(args.rpm),
        "temperature_c": scene.air.temperature_c,
        "rate": int(scene.simulation.sample_rate),
        "duration_s": round(pass_s(car_of(scene), args.speed), 6),
        "level_db": [round(float(v), 1) for v in power.total_db],
        "rolling_db": [round(float(v), 2) for v in power.rolling_db],
        "propulsion_db": [round(float(v), 2) for v in power.propulsion_db],
        "sound_power_db": round(power.sound_power_db, 2),
        "sound_power_a_db": round(power.sound_power_a_db, 2),
    }


def car(demo: Demo, args: argparse.Namespace) -> None:
    """``car/car.wav``: the car's sound, dry, as long as its pass (tyres and engine made
    to CNOSSOS-EU's sound power at ``--speed``, :func:`acoustic_spaces.demos.car.car_sound`),
    and car.json, what it is made of."""
    from acoustic_spaces.demos.car import car_sound, light_vehicle_power

    if float(args.rpm) <= 0.0:
        raise SystemExit(f"error: --rpm is a positive number, not {args.rpm!r}")
    info = car_info(args)
    power = light_vehicle_power(info["speed_kmh"], info["temperature_c"])
    x = car_sound(power, info["rate"], info["duration_s"], rpm=info["rpm"])
    folder = args.work / "car"
    write_wav(folder / CAR_FILE, x, info["rate"])
    pipeline.write_json(folder / CAR_INFO, info)
    print(
        f"wrote {folder / CAR_FILE}: {info['duration_s']:.1f} s of a car at "
        f"{info['speed_kmh']:g} km/h, {info['sound_power_a_db']:.1f} dB(A) of sound power"
    )


def car_done(args: argparse.Namespace) -> bool:
    """Whether the car's sound is there and made for the arguments and the scene as it
    is now, or not needed (``--input car=...``)."""
    if car_input(args) is not None:
        return True
    path = args.work / "car" / CAR_INFO
    if not path.exists() or not (args.work / "car" / CAR_FILE).exists():
        return False
    try:
        return pipeline.read_json(path) == car_info(args)
    except SystemExit:
        return False


def plays(args: argparse.Namespace) -> dict[str, str]:
    """What the car plays unless ``--input`` gives it a file."""
    try:
        speed = f"at {round(speed_of(args), 2):g} km/h, "
    except SystemExit:  # the check under the form says what keeps it from driving
        speed = ""
    return {CAR: f"a car {speed}made to CNOSSOS-EU's sound power"}


def played_by(args: argparse.Namespace) -> str:
    """What the car plays, as walk.json keeps it."""
    file = car_input(args)
    if file is not None:
        return str(file)
    return f"car {round(speed_of(args), 4):g} km/h {args.rpm:g} rpm"


def feed(demo: Demo, args: argparse.Namespace, rate: int, duration_s: float) -> np.ndarray:
    """What the car plays along its pass, at ``rate``: its made sound, or the file
    ``--input car=WAV`` gives it (mixed down to mono, cut to the pass, at the made
    sound's RMS), faded in and out over FADE_S."""
    from acoustic_spaces.demos.car import SOUND_RMS

    n = int(round(duration_s * rate))
    file = car_input(args)
    if file is None:
        path = args.work / "car" / CAR_FILE
        if not path.exists():
            raise SystemExit(f"error: no {path}; run first: {demo.command('car', args)}")
        x, r = read(path)
        x = resampled(mono(x), r, rate)
    else:
        x, r = read(file)
        x = resampled(mono(x), r, rate)
        if len(x) < n:
            raise SystemExit(
                f"error: {file.name} is {len(x) / rate:.1f} s long; the car takes "
                f"{duration_s:.1f} s along its walk"
            )
        rms = gated_rms(x[:n], rate)
        if rms <= 0.0:
            raise SystemExit(f"error: {file} is silent")
        x = x * (SOUND_RMS / rms)
    x = np.array(x[:n], dtype=float)
    if len(x) < n:
        raise SystemExit(
            f"error: the car's sound is shorter than its pass; run: {demo.command('car', args)}"
        )
    fade = min(int(FADE_S * rate), n // 2)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    x[:fade] *= ramp
    x[n - fade :] *= ramp[::-1]
    return x


# --------------------------------------------------------------------------- the scene


def scene(demo: Demo, args: argparse.Namespace) -> None:
    """``--scene`` (scene_demo.yaml) from ``--from`` (scene.yaml): the car in use, along
    its walk with Doppler, at its own speed or at ``--speed`` (:func:`driving`), and
    CNOSSOS-EU's sound power at that speed, every other source left out."""
    info = car_info(args)
    data = yaml.safe_load(args.source.read_text(encoding="utf-8"))
    data["name"] = f"{data.get('name', 'scene')}: a car passing"
    car = car_of(source_scene(args))
    for source in data["sources"]:
        if source["name"] == CAR:
            source["enabled"] = True
            source["level_db"] = info["level_db"]
            source.pop("spectrum", None)
            walk = source["walk"]
            if args.speed is not None:
                walk["speed_m_s"] = round(info["speed_kmh"] / 3.6, 6)
                if walk.get("keyframes"):
                    walk["to"] = [list(place) for place in places_of(car)]
                    walk.pop("keyframes")
            walk["doppler"] = True
        else:
            source["enabled"] = False
    pipeline.write_scene(
        args.scene,
        data,
        "# The example with its car driving past alone, at the speed and with the sound\n"
        "# power of the pass-by demo (CNOSSOS-EU), written by passby_demo.py.\n",
        args.source.parent,
    )
    print(
        f"wrote {args.scene}: the car at {info['speed_kmh']:g} km/h, "
        f"{info['sound_power_db']:.1f} dB of sound power"
    )


def scene_done(args: argparse.Namespace) -> bool:
    """Whether scene_demo.yaml is there, newer than the example's scene, and for this
    speed."""
    from acoustic_spaces.scene import Scene

    if not args.scene.exists():
        return False
    if args.source.exists() and args.scene.stat().st_mtime < args.source.stat().st_mtime:
        return False
    try:
        car = car_of(Scene.from_yaml(args.scene))
        made = 3.6 * walk_speed(car)
    except (Exception, SystemExit):  # a scene of another kind: written again
        return False
    return car.walk.doppler and abs(made - speed_of(args)) < 1e-4 * made


def room_of(demo: Demo, args: argparse.Namespace):
    from acoustic_spaces import Room

    if not args.scene.exists():
        raise SystemExit(f"error: no {args.scene}; run first: {demo.command('scene', args)}")
    return Room.from_yaml(args.scene)


# --------------------------------------------------------------------------- the pass


def pitch_change(points, listener, speed_m_s: float, c: float) -> float:
    """How far the pitch of the car's direct sound falls at ``listener`` (a position)
    from where its walk starts to where it ends (semitones): heard at c / (c - v_r), v_r
    the car's speed towards the listener, along its first leg and its last."""
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


def walk(demo: Demo, args: argparse.Namespace) -> None:
    """Each listener in use: the car simulated every ``--step`` along its walk, its sound
    played through those places with Doppler, and the level along the way; one gain
    for every listener (the loudest peak at -1 dBFS), into ``walk/``: a WAV per
    listener, seats.json and walk.json."""
    from acoustic_spaces.demos.scene_demo import format_of
    from acoustic_spaces.walkthrough import pass_by, render_walk, simulate_walk

    room = room_of(demo, args)
    car = car_of(room.scene)
    problem = car_problem(room.scene, args.scene.name)
    if problem:
        raise SystemExit(f"error: {problem}")
    if not car.walk.doppler:
        raise SystemExit(
            f"error: the car in {args.scene.name} has no Doppler; run: "
            f"{demo.command('scene', args)}"
        )
    try:
        speed = walk_speed(car)
    except ValueError as exc:
        raise SystemExit(f"error: {args.scene.name}: {exc}") from None
    rate = int(room.scene.simulation.sample_rate)
    duration = pass_s(car)
    x = feed(demo, args, rate, duration)
    fmt = format_of(args, room.scene)
    names = listeners_of(room)
    if not names:
        raise SystemExit(f"error: every listener of {args.scene.name} is left out")
    c = room.speed_of_sound()
    mixes, heard, points = {}, {}, None
    for name in names:
        print(f"simulating the car every {args.step_m:g} m along its walk, heard at {name}")
        walked = simulate_walk(room, CAR, listener=name, step_m=float(args.step_m), progress=print)
        print(f"playing the car through {len(walked.names)} places with Doppler")
        mixes[name] = render_walk(walked, {CAR: (x, rate)}, fmt, sample_rate=rate)
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
    from acoustic_spaces.demos.car import light_vehicle_power

    power = light_vehicle_power(speed * 3.6, room.scene.air.temperature_c)
    pipeline.write_json(
        out / WALK,
        {
            "plays": played_by(args),
            "label": label,
            "step_m": float(args.step_m),
            "speed_kmh": round(speed * 3.6, 4),
            "rpm": float(args.rpm),
            "sound_power_a_db": round(power.sound_power_a_db, 2),
            "speed_of_sound": round(c, 3),
            "times": [round(float(t), 4) for t in points.times],
            "positions": np.round(points.positions, 4).tolist(),
            "listeners": heard,
        },
    )
    print(f"wrote {out}: {len(names)} listeners, the car {duration:.1f} s along its walk")


def walk_done(args: argparse.Namespace) -> bool:
    """Whether the passes are there, made from the scene, the car's sound, the step and
    the format of this run, and newer than what they are made from."""
    path = args.work / "walk" / WALK
    if not path.exists() or not args.scene.exists():
        return False
    made = path.stat().st_mtime
    feeds = [args.scene, car_input(args) or args.work / "car" / CAR_FILE]
    if any(f.exists() and f.stat().st_mtime > made for f in feeds):
        return False
    facts = pipeline.read_json(path)
    try:
        plays = played_by(args)
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


def credit_of(args: argparse.Namespace, facts: dict) -> str:
    """The stills' credit line: how the car's sound was made, or the file it plays."""
    file = car_input(args)
    if file is not None:
        return f"Playing car: {file.name}, faded in and out over {FADE_S:g} s"
    return (
        f"Car's sound made up: tyre noise and a four-cylinder engine at {facts['rpm']:g} rpm, "
        f"each with the octave spectrum of CNOSSOS-EU's passenger car at "
        f"{facts['speed_kmh']:g} km/h (Directive (EU) 2015/996); faded in and out over "
        f"{FADE_S:g} s"
    )


def video(demo: Demo, args: argparse.Namespace) -> None:
    """tour.wav (each listener's pass in turn, turned up to one loudness, the loudest peak
    at -1 dBFS) and tour.mp4: the scene from above with the car driving past, the level
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
    tool[0] += f"; the car simulated every {facts['step_m']:g} m, with Doppler"
    drawing = passby_pictures.Pictures(
        data,
        outline_edges(room)[0],
        facts,
        seats["seats"],
        stops,
        path=knots(car_of(room.scene)),
        credit=credit_of(args, facts),
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


def moving(args: argparse.Namespace) -> list[str]:
    """Who the demo plays moving: the car, when the example gives it a walk that is on."""
    try:
        return [] if car_problem(source_scene(args), args.source.name) else [CAR]
    except (Exception, SystemExit):
        return []


def problems(demo: Demo, args: argparse.Namespace) -> str | None:
    """The car must drive: the example's scene gives it a walk that is on, at one steady
    speed unless ``--speed`` gives one, and the speed, the engine and the step are
    positive."""
    from acoustic_spaces.scene import Scene

    for option in ("speed", "rpm", "step_m"):
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
    problem = car_problem(scene, source.name)
    if problem is None and problems_of_speed(args, scene):
        return f"{source.name}: {problems_of_speed(args, scene)}"
    return problem


def summary(args: argparse.Namespace) -> str | None:
    """How the car drives in a run with ``args`` (paths resolved), in a line: how far,
    how fast (km/h and m/s, and whose speed) and so how long the pass and its sound last;
    None when it cannot drive (the check says why)."""
    from acoustic_spaces.walkthrough import knots

    try:
        scene = source_scene(args)
        if car_problem(scene, args.source.name) or problems_of_speed(args, scene):
            return None
        path = knots(driving(car_of(scene), args.speed))
    except (Exception, SystemExit):
        return None
    speed = path.length_m / path.duration_s
    whose = "--speed" if args.speed is not None else "as its walk in the scene goes"
    return (
        f"The car drives {path.length_m:.1f} m along its walk at {3.6 * speed:.1f} km/h "
        f"({speed:.2f} m/s, {whose}): the pass, and the car's sound, last "
        f"{path.duration_s:.1f} s. --speed drives it faster or slower along the same road; "
        "Edit movement… changes the road, and its speed."
    )


def problems_of_speed(args: argparse.Namespace, scene) -> str | None:
    """What keeps the car of ``scene`` from one speed: a ``--speed`` that is not positive,
    or without one keyframes that are not steady."""
    if args.speed is not None:
        speed = float(args.speed)
        return None if math.isfinite(speed) and speed > 0.0 else "--speed is not positive"
    try:
        walk_speed(car_of(scene))
    except ValueError as exc:
        return str(exc)
    return None


def demo(script: Path, example: str, title: str, description: str) -> Demo:
    """The pass-by demo of ``example``, run by ``script`` (its ``passby_demo.py``)."""
    made = Demo(
        name="passby",
        example=example,
        title=title,
        description=description,
        script=Path(script),
        steps=[],
        options=[
            Option(
                "speed",
                "how fast the car drives along its walk (km/h; m/s times 3.6), which sets "
                "how long its pass and its sound last (the walk's length over the speed) and "
                "its sound power; by default as its walk in the scene goes: its speed along "
                "points, or the one steady speed of its keyframes (whose places it goes "
                "through at --speed when given)",
                None,
                "float",
                steps=("car", "scene", "walk"),
                unit="km/h",
                unset="as in the scene",
            ),
            Option(
                "rpm",
                "how fast the car's engine turns (revolutions per minute), for its made sound",
                2000.0,
                "float",
                steps=("car",),
                unit="rpm",
            ),
            Option(
                "step_m",
                "how far apart the car is simulated along its walk (m)",
                STEP_M,
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
                steps=("car", "scene"),
                flag="--from",
                filter="Scene files (*.yaml *.yml)",
            ),
            Option(
                "scene",
                "the scene with the car driving past alone",
                Path("scene_demo.yaml"),
                "path",
            ),
            *[o for o in common_options() if o.name != "again"],
            dataclasses.replace(
                input_option(("walk", "video")),
                help="a WAV file the car plays instead of its made sound, as car=WAV: a dry "
                "recording (on board or at the wheel; one of a car passing has its Doppler, "
                "the road and the distance in it already), mixed down to mono, at least as "
                "long as the pass; the captions' levels are CNOSSOS-EU's car at --speed",
            ),
        ],
        scene_from="scene",
        problems=problems,
        plays=plays,
        moving=moving,
        summary=summary,
    )
    made.steps = [
        Step(
            "car",
            lambda args: car(made, args),
            "demo/car/car.wav: the car's sound, made to CNOSSOS-EU",
            car_done,
        ),
        Step(
            "scene",
            lambda args: scene(made, args),
            "scene_demo.yaml: the car alone, at --speed, with Doppler",
            scene_done,
        ),
        Step(
            "walk",
            lambda args: walk(made, args),
            "simulate the car along its walk at every listener and play it with Doppler",
            walk_done,
        ),
        Step(
            "video",
            lambda args: video(made, args),
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
    "car_problem",
    "demo",
    "driving",
    "pitch_change",
    "walk_speed",
]
