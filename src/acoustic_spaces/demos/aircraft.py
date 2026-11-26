"""The aircraft of the airport example's flight demo (:mod:`acoustic_spaces.demos.passby`):
Doc 29's hypothetical wing-engined twin jet JETW (Volume 3) taking off and landing,
flown as Doc 29 flies it and sounding as its NPD curves say.

Each is the example's source of its name, ``departure`` or ``arrival``, a point with a
trajectory that gives its ground track: its points' places in plan, the departure's from its
start of roll, the arrival's touching down where its trajectory first comes down to the
runway (at most 1 m up). The demo's scene step writes the flight along that track into
the demo's scene as the trajectory's keyframes (:func:`acoustic_spaces.doc29.source.flight`):
the heights, speeds and thrust Doc 29's Appendix B gives JETW for Volume 3's procedures
(:data:`PROCEDURES`) at its default weights, in the scene's air (its temperature and
pressure; ANP's reference headwind of 8 kt), as its flight path's segments
(:func:`acoustic_spaces.doc29.path.flight_path`); the example's own trajectory is that flight
already. The aircraft faces the way it flies.

Its sound power and directivity are its NPD curves' and its spectral class's
(:mod:`acoustic_spaces.doc29.source`): the scene's source has its sound power at the
highest thrust of its flight, and its sound plays louder and softer as its thrust
changes along the flight, as the NPD curves' maximum level 1000 ft away does; the level
along the pass follows it place by place (:meth:`Aircraft.sounding`). Its sound is noise
made to that power, one spectrum all along, with none of an aircraft's tones (its
fan's, its buzz-saw noise). Nothing of the simulation uses this module.
"""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path

import numpy as np

from acoustic_spaces.demos import pipeline
from acoustic_spaces.demos.audio import write as write_wav
from acoustic_spaces.demos.passby import FADE_S, Vehicle, car_input, car_of, kmh, source_scene

#: Volume 3's aircraft the demo flies, and the procedures of each operation (Part 2,
#: table C-6): the ICAO-A noise abatement departure and the 3° approach.
JET = "JETW"
PROCEDURES = {"D": "ICAO_A", "A": "Descend"}
#: How far apart the aircraft's places are simulated by default (m), as the
#: helicopter's (:data:`acoustic_spaces.demos.helicopter.STEP_M`): from 100 m away and
#: more, places 4 m apart follow the exact field closely (``validation/doppler.py``).
STEP_M = 4.0
#: The noise's seed, so that every step makes the same sound.
SEED = 0
#: How often the made sound's gain follows the thrust (Hz).
GAIN_RATE_HZ = 100.0
SOUND_FILE = "engines.wav"
INFO_FILE = "aircraft.json"


@functools.cache
def _data():
    """Volume 3's JETW: its aircraft entry, NPD curves, spectral classes, performance and
    procedures."""
    from acoustic_spaces.doc29 import volume3

    return (
        volume3.aircraft()[JET],
        volume3.npd(),
        volume3.spectral_classes(),
        volume3.performance()[JET],
        volume3.departure_procedures(),
        volume3.approach_procedures(),
    )


def _curves(letter: str):
    """JETW's maximum levels' NPD curves and its spectral class for the operation of
    Doc 29's ``letter`` ("D" or "A")."""
    plane, npd, classes, *_ = _data()
    spectrum = plane.departure_spectrum if letter == "D" else plane.approach_spectrum
    return npd[(plane.npd, "LAmax", letter)], classes[spectrum]


def track_points(source) -> list[tuple[float, float, float]]:
    """The places (x, y, z) of the trajectory of ``source`` (a scene source): its position,
    then its points or its keyframes' places, each once."""
    trajectory = source.trajectory

    def point(at) -> tuple[float, float, float]:
        x, y, z = at
        return float(x), float(y), float(z)

    places = [point(source.position)]
    later = (
        [k.position for k in trajectory.keyframes] if trajectory.keyframes else list(trajectory.to)
    )
    for place in map(point, later):
        if place != places[-1]:
            places.append(place)
    return places


class Aircraft(Vehicle):
    """JETW on one operation, ``"D"`` (taking off, the source ``departure``) or ``"A"``
    (landing, ``arrival``), flown by its procedure of :data:`PROCEDURES`."""

    timed = True
    go, goes, going, way = "fly", "flies", "flying over", "its flight path"
    fleet_noun = "aircraft"
    step_m = STEP_M
    sound_file, info_file = SOUND_FILE, INFO_FILE
    sound_step = "aircraft"
    steady = "Doc 29 flies it as its procedure says"
    slow_weighted = True

    def __init__(self, operation: str, script: str = "flight_demo.py") -> None:
        if operation not in PROCEDURES:
            raise ValueError(f"an operation is D or A, not {operation!r}")
        self.operation = operation
        self.departing = operation == "D"
        self.name = "departure" if self.departing else "arrival"
        self.demo_name = "flight"
        self.noun = "departing jet" if self.departing else "landing jet"
        self.procedure = PROCEDURES[operation]
        self.title = (
            f"Doc 29's twin jet {JET} taking off ({self.procedure.replace('_', '-')} procedure)"
            if self.departing
            else f"Doc 29's twin jet {JET} landing (a 3° approach)"
        )
        self.script = script

    # -- the flight --------------------------------------------------------------------

    def problem(self, source) -> str | None:
        """Its trajectory gives a ground track, two places apart at least; an arrival's comes
        down to the runway, where it touches down."""
        from acoustic_spaces.doc29.path import LOWEST_SOURCE_M, SAME_PLAN_POINT_M

        places = track_points(source)
        # Track.through draws a ground track if a place lies more than this from the first.
        x0, y0, _ = places[0]
        if not any(math.hypot(x - x0, y - y0) > SAME_PLAN_POINT_M for x, y, _ in places):
            return f"the {self.name}'s trajectory needs two places apart in plan, its ground track"
        if not self.departing and not any(z <= LOWEST_SOURCE_M + 1e-6 for *_, z in places):
            return (
                f"the {self.name} touches down where its trajectory first comes down to the "
                f"runway, at most {LOWEST_SOURCE_M:g} m up, but none of its places is that low"
            )
        return None

    def segments_of(self, scene) -> tuple:
        """The segments of its flight path (:class:`~acoustic_spaces.doc29.path.Segment`)
        along the ground track of its trajectory in ``scene``, in the scene's air: what Doc 29
        sums its levels over."""
        places = tuple(track_points(car_of(scene, self)))
        air = scene.air
        return _segments(self.operation, places, air.temperature_c, air.pressure_kpa)

    def flight_of(self, scene):
        """Its flight (:class:`~acoustic_spaces.doc29.source.Flight`) along the ground
        track of its trajectory in ``scene``, in the scene's air: its segments' keyframes."""
        return _flight(self.segments_of(scene))

    def mean_speed_m_s(self, args: argparse.Namespace, scene) -> float:
        flown = self.flight_of(scene)
        lengths = np.linalg.norm(np.diff(np.asarray(flown.positions), axis=0), axis=1)
        return float(lengths.sum()) / flown.duration_s

    def how_it_goes(self, args: argparse.Namespace, scene) -> str | None:
        flown = self.flight_of(scene)
        first, last = flown.positions[0][2], flown.positions[-1][2]
        going = "speeding up" if self.departing else "slowing"
        height = "climbing" if self.departing else "coming down"
        return (
            f"The {self.noun} flies {self.track_m(scene):.0f} m along its trajectory's ground "
            f"track in {flown.duration_s:.1f} s, {going} from {kmh(3.6 * flown.speeds[0])} "
            f"to {kmh(3.6 * flown.speeds[-1])} km/h and {height} from {first:.0f} to "
            f"{last:.0f} m up, as Doc 29 flies {JET}'s {self.procedure} procedure in the "
            "scene's air: the pass, and its sound, last as long. Edit movement… changes "
            "its ground track."
        )

    def track_m(self, scene) -> float:
        """How long its ground track is (m)."""
        places = np.asarray(track_points(car_of(scene, self)))[:, :2]
        return float(np.linalg.norm(np.diff(places, axis=0), axis=1).sum())

    # -- its sound ---------------------------------------------------------------------

    def curves(self):
        """Its maximum levels' NPD curves and its spectral class, for its operation."""
        return _curves(self.operation)

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sound is made of: its flight in the example's scene (its air), its
        highest thrust and its sound power and directivity's offset there."""
        from acoustic_spaces.doc29 import source as doc29_source
        from acoustic_spaces.materials.bands import power_levels
        from acoustic_spaces.parameters import a_weighted

        example = source_scene(args)
        flown = self.flight_of(example)
        maximum, spectrum = self.curves()
        top = max(flown.powers)
        air = example.air
        thirds = doc29_source.sound_power_db(
            maximum, spectrum, top, air.temperature_c, air.humidity_percent, air.pressure_kpa
        )
        plane = _data()[0]
        level = power_levels(thirds.tolist(), "octave") + doc29_source.below_offset_db(
            plane.installation
        )
        return {
            "aircraft": JET,
            "operation": self.operation,
            "procedure": self.procedure,
            "spectral_class": spectrum.name,
            "installation": plane.installation,
            "temperature_c": air.temperature_c,
            "pressure_kpa": air.pressure_kpa,
            "rate": int(example.simulation.sample_rate),
            # As long as the trajectory the scene gets, whose keyframes' times are rounded: a
            # sound as long as the flight itself can end a sample or two before its pass.
            "duration_s": flown.keyframes()[-1]["time_s"],
            "speed_kmh": round(float(speed_kmh), 6),
            "slowest_kmh": round(3.6 * min(flown.speeds), 4),
            "fastest_kmh": round(3.6 * max(flown.speeds), 4),
            "seed": SEED,
            "thrust_lb": round(top, 1),
            "level_db": [round(float(v), 1) for v in level],
            "sound_power_db": round(10.0 * math.log10(float(np.sum(10.0 ** (level / 10.0)))), 2),
            "sound_power_a_db": round(a_weighted(level), 2),
        }

    def gains_db(self, args: argparse.Namespace, times) -> np.ndarray:
        """How far under its sound power the aircraft sounds at ``times`` (s into its
        flight, dB): its NPD curves' maximum level 1000 ft away at its thrust then,
        against its highest thrust's."""
        from acoustic_spaces.doc29.source import power_change_db

        flown = self.flight_of(source_scene(args))
        maximum, _ = self.curves()
        return power_change_db(maximum, flown.power_at(times), max(flown.powers))

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        """Its engines' sound, dry, as long as its flight: noise with the octave spectrum
        of its sound power, at an RMS of SOUND_RMS at its highest thrust, its gain
        following its thrust along the flight (:meth:`gains_db`)."""
        from acoustic_spaces.demos.road import SOUND_RMS, _to_bands

        rate = info["rate"]
        n = int(round(info["duration_s"] * rate))
        noise = np.random.default_rng(SEED).standard_normal(n)
        x = _to_bands(noise, rate, np.asarray(info["level_db"], dtype=float))
        x *= SOUND_RMS / float(np.sqrt(np.mean(x**2)))
        coarse = np.arange(0.0, info["duration_s"] + 1.0 / GAIN_RATE_HZ, 1.0 / GAIN_RATE_HZ)
        gains = np.interp(np.arange(n) / rate, coarse, self.gains_db(args, coarse))
        x *= 10.0 ** (gains / 20.0)
        write_wav(folder / self.sound_file, x, rate)
        low = float(np.min(gains))
        return (
            f"{info['duration_s']:.1f} s of {JET}'s engines along its {self.procedure} "
            f"flight, {info['sound_power_a_db']:.1f} dB(A) of sound power at "
            f"{info['thrust_lb']:.0f} lb of thrust, down to {-low:.1f} dB under it"
        )

    def sounding(
        self, args: argparse.Namespace, part: str, played, rate: int, at: np.ndarray
    ) -> np.ndarray | None:
        """Its sound at each place, against its sound power, in energy: as its thrust
        then (:meth:`gains_db`), which its made sound's gain follows; a recording
        ``--input`` gives it, as loud as it plays around each place."""
        from acoustic_spaces.demos.passby import blocks_loudness
        from acoustic_spaces.demos.road import SOUND_RMS

        if car_input(args, self) is not None:
            return blocks_loudness(np.asarray(played, dtype=float), rate, at, SOUND_RMS)
        return 10.0 ** (self.gains_db(args, at) / 10.0)

    def edit(self, source: dict, info: dict, args: argparse.Namespace) -> None:
        """Its flight as its trajectory's keyframes, its sound power at its highest thrust, and
        its engines' installation as its directivity, in a CSV table beside the demo's
        scene."""
        from acoustic_spaces.directivity import write_directivity_csv
        from acoustic_spaces.doc29.source import directivity

        flown = self.flight_of(source_scene(args))
        source["position"] = [round(v, 4) for v in flown.positions[0]]
        yaw, pitch, roll = (round(v, 4) for v in flown.orientations[0])
        source["orientation"] = {"yaw": yaw, "pitch": pitch, "roll": roll}
        trajectory = source.setdefault("trajectory", {})
        trajectory.pop("to", None)
        trajectory["keyframes"] = flown.keyframes()
        source["level_db"] = info["level_db"]
        source.pop("spectrum", None)
        table = args.scene.with_name(f"{args.scene.stem}_{self.name}.csv")
        write_directivity_csv(
            table,
            directivity(info["installation"]),
            f"The directivity of {JET}'s engines ({info['installation']}-mounted), Doc 29's "
            f"installation correction, written by {self.script}",
        )
        named = pipeline.named_from(table, args.source.parent)
        source["directivity"] = {"file": named}

    # -- its words ---------------------------------------------------------------------

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        return f"{JET}'s engines along its flight, made to Doc 29's NPD curves"

    def played_by(self, args: argparse.Namespace) -> str:
        return f"{JET} {self.procedure} {self.operation}"

    def speed_words(self, info: dict) -> str:
        """How fast it goes, for the stills: from its first speed to its last (a
        departure only speeds up, an arrival only slows)."""
        ends = (info["slowest_kmh"], info["fastest_kmh"])
        first, last = ends if self.departing else ends[::-1]
        return f"From {kmh(first)} to {kmh(last)} km/h"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        doing = "taking off" if self.departing else "landing"
        return {
            "aircraft": info["aircraft"],
            "procedure": info["procedure"],
            "sound_power_a_db": info["sound_power_a_db"],
            "speed_text": self.speed_words(info),
            "about": f"Doc 29's twin jet {info['aircraft']} {doing}, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power\nat "
            f"{info['thrust_lb']:.0f} lb of thrust (its NPD curves)",
        }

    def credit(self, facts: dict) -> str:
        return (
            f"Aircraft's sound: noise made to the NPD curves and spectral class of {JET}, "
            f"ECAC Doc 29's hypothetical twin jet, its thrust along its flight by Doc 29's "
            f"Appendix B; faded in and out over {FADE_S:g} s"
        )

    def texts(self) -> dict[str, str]:
        return {
            "input": f"a WAV file the {self.name} plays instead of its engines' made sound, "
            f"as {self.name}=WAV: a dry recording, mixed down to mono and sent every way "
            "with the engines' directivity, at least as long as the flight, its level "
            "following the recording's",
            "sound": f"demo/RUN/{self.name}/{SOUND_FILE}: its engines' sound along its flight, dry",
            "header": f"# The example with its {self.noun} alone, flown as Doc 29 flies "
            f"{JET}'s\n# {self.procedure} procedure, with the sound power and directivity of "
            f"its NPD curves,\n# written by {self.script}.\n",
        }

    def fleet_texts(self) -> dict[str, str] | None:
        return {
            "input": "a WAV file the aircraft plays instead of its engines' made sound, as "
            "NAME=WAV: a dry recording, mixed down to mono and sent every way with the "
            "engines' directivity, at least as long as the flight, its level following "
            "the recording's",
            "sound": f"demo/RUN/NAME/{SOUND_FILE}: the aircraft's engines' sound along its "
            "flight, dry (the departure's, …)",
            "header": self.texts()["header"],
        }


#: How far from the line through its neighbours a place of a ground track may lie and
#: still be on it, not a corner (m): a flight's keyframes, rounded to 0.1 mm, lie on
#: their track within that.
ON_THE_LINE_M = 1e-3


def _corners(plan: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The places of ``plan`` (x, y, in order) that make a ground track: its ends and
    its corners, leaving out each place on the straight line between the places kept on
    either side of it (within :data:`ON_THE_LINE_M`), which changes nothing of the track
    but where its legs break, so that a flight's own keyframes fly as the track they
    were flown along."""
    kept = [plan[0]]
    for k in range(1, len(plan) - 1):
        (ax, ay), (bx, by), (cx, cy) = kept[-1], plan[k], plan[k + 1]
        length = math.hypot(cx - ax, cy - ay)
        if length == 0.0:
            kept.append(plan[k])
            continue
        off = abs((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / length
        ahead = (bx - ax) * (cx - ax) + (by - ay) * (cy - ay)
        if off > ON_THE_LINE_M or not 0.0 <= ahead <= length**2:
            kept.append(plan[k])
    kept.append(plan[-1])
    return kept


@functools.lru_cache(maxsize=8)
def _flight(segments: tuple):
    from acoustic_spaces.doc29.source import flight

    return flight(segments)


@functools.lru_cache(maxsize=8)
def _segments(
    operation: str,
    places: tuple[tuple[float, float, float], ...],
    temperature_c: float,
    pressure_kpa: float,
) -> tuple:
    """JETW's flight path of ``operation`` by its procedure along the ground track
    through ``places`` (from the start of roll, or touching down where they first come
    down to the runway), in air at ``temperature_c`` and ``pressure_kpa``: its segments
    along the part of the profile the track spans."""
    from acoustic_spaces.doc29 import performance
    from acoustic_spaces.doc29.path import LOWEST_SOURCE_M, Track, flight_path
    from acoustic_spaces.doc29.source import profile_between

    _, _, _, plane, departures, approaches = _data()
    air = performance.Aerodrome(
        temperature_c=temperature_c,
        pressure_inhg=performance.SEA_LEVEL_INHG * pressure_kpa / 101.325,
    )
    plan = [(x, y) for x, y, _ in places]
    if operation == "D":
        steps = departures[(JET, PROCEDURES["D"])]
        points = performance.departure(plane, steps, plane.weights[("D", "1")], air)
        origin = plan[0]
    else:
        points = performance.arrival(plane, approaches[(JET, PROCEDURES["A"])], None, air)
        origin = next((x, y) for x, y, z in places if z <= LOWEST_SOURCE_M + 1e-6)
    track = Track.through(_corners(plan), origin)
    ends = track.node_distances
    profile = profile_between(performance.as_profile(points), float(ends[0]), float(ends[-1]))
    return tuple(flight_path(profile, track, departure=operation == "D"))


#: The operations of a scene's Doc 29 model (``model: {method: doc-29}``), by the
#: letter Doc 29 gives them.
OPERATIONS = {"departure": "D", "arrival": "A"}


def flight_problem(operation: str, places) -> str | None:
    """Why the ground track of ``places`` (x, y, z: a trajectory's places, in order)
    cannot be flown by ``operation`` ("departure" or "arrival"), or None: it needs two
    places apart in plan, and an arrival a place down on the runway, where it touches
    down."""
    from acoustic_spaces.doc29.path import LOWEST_SOURCE_M, SAME_PLAN_POINT_M

    places = [tuple(float(v) for v in p) for p in places]
    x0, y0, _ = places[0]
    if not any(math.hypot(x - x0, y - y0) > SAME_PLAN_POINT_M for x, y, _ in places):
        return f"the {operation}'s trajectory needs two places apart in plan, its ground track"
    if operation == "arrival" and not any(z <= LOWEST_SOURCE_M + 1e-6 for *_, z in places):
        return (
            f"the arrival touches down where its trajectory first comes down to the runway, "
            f"at most {LOWEST_SOURCE_M:g} m up, but none of its places is that low"
        )
    return None


def flight_segments(operation: str, places, temperature_c: float, pressure_kpa: float) -> tuple:
    """The segments (:class:`~acoustic_spaces.doc29.path.Segment`) of JETW's flight path
    of ``operation`` ("departure" or "arrival") by its procedure of :data:`PROCEDURES`
    along the ground track of ``places`` (a trajectory's places, x, y, z), in air at
    ``temperature_c`` and ``pressure_kpa``: from the start of roll at the first place,
    or touching down where the places first come down to the runway. What Doc 29 sums
    its levels over (:func:`acoustic_spaces.doc29.noise.event`)."""
    points = tuple(tuple(float(v) for v in p) for p in places)
    return _segments(OPERATIONS[operation], points, temperature_c, pressure_kpa)


def flown(operation: str, places, temperature_c: float, pressure_kpa: float):
    """JETW's flight (:class:`~acoustic_spaces.doc29.source.Flight`) of ``operation``
    along the ground track of ``places`` in air at ``temperature_c`` and
    ``pressure_kpa``: its :func:`flight_segments` as keyframes."""
    return _flight(flight_segments(operation, places, temperature_c, pressure_kpa))


def flight_power_db(
    operation: str, flight, temperature_c: float, humidity_percent: float, pressure_kpa: float
) -> list[float]:
    """The sound power per octave band (dB re 1 pW) of JETW on ``operation`` at the
    highest thrust of ``flight``: its NPD curves' maximum level 1000 ft under it with its
    spectral class's spectrum, in air at ``temperature_c``, ``humidity_percent`` and
    ``pressure_kpa`` (:func:`acoustic_spaces.doc29.source.sound_power_db`), raised by its
    engines' installation's offset (:func:`~acoustic_spaces.doc29.source.below_offset_db`),
    as it sends that much straight down."""
    from acoustic_spaces.doc29 import source as doc29_source
    from acoustic_spaces.materials.bands import power_levels

    maximum, spectrum = _curves(OPERATIONS[operation])
    thirds = doc29_source.sound_power_db(
        maximum, spectrum, max(flight.powers), temperature_c, humidity_percent, pressure_kpa
    )
    offset = doc29_source.below_offset_db(_data()[0].installation)
    return [round(float(v) + offset, 6) for v in power_levels(thirds.tolist(), "octave")]


def flight_gains_db(operation: str, flight, times) -> np.ndarray:
    """How far under its sound power JETW on ``operation`` sounds at ``times`` (s into
    ``flight``, dB): its NPD curves' maximum level 1000 ft away at its thrust then,
    against its highest thrust's."""
    from acoustic_spaces.doc29.source import power_change_db

    maximum, _ = _curves(OPERATIONS[operation])
    return power_change_db(maximum, flight.power_at(times), max(flight.powers))


def engines_directivity():
    """The directivity of JETW's engines (:func:`acoustic_spaces.doc29.source.directivity`
    of its installation: Doc 29's installation correction, eq. 4-15)."""
    from acoustic_spaces.doc29.source import directivity

    return directivity(_data()[0].installation, f"{JET}'s engines")


#: The airport example's departure and arrival.
DEPARTURE = Aircraft("D")
ARRIVAL = Aircraft("A")

__all__ = [
    "ARRIVAL",
    "DEPARTURE",
    "JET",
    "OPERATIONS",
    "PROCEDURES",
    "STEP_M",
    "Aircraft",
    "engines_directivity",
    "flight_gains_db",
    "flight_power_db",
    "flight_problem",
    "flight_segments",
    "flown",
    "track_points",
]
