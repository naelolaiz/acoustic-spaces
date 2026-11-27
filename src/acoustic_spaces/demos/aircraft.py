"""The aircraft of a scene's source of a Doc 29 model (``model: {method: doc-29}``, the
airport example's jets): Doc 29's hypothetical wing-engined twin jet JETW (Volume 3)
taking off and landing, flown as Doc 29 flies it and sounding as its NPD curves say.

Each is a source with a trajectory that gives its ground track: its points' places in
plan, the departure's from its start of roll, the arrival's touching down where its
trajectory first comes down to the runway (at most 1 m up). Its model flies it along
that track, its trajectory taking the flight's keyframes
(:func:`acoustic_spaces.doc29.source.flight`):
the heights, speeds and thrust Doc 29's Appendix B gives JETW for Volume 3's procedures
(:data:`PROCEDURES`) at its default weights, in the scene's air (its temperature and
pressure; ANP's reference headwind of 8 kt), as its flight path's segments
(:func:`acoustic_spaces.doc29.path.flight_path`). The aircraft faces the way it flies.

Its sound power and directivity are its NPD curves' and its spectral class's
(:mod:`acoustic_spaces.doc29.source`): the scene's source has its sound power at the
highest thrust of its flight, and its sound plays louder and softer as its thrust
changes along the flight, as the NPD curves' maximum level 1000 ft away does; the level
along the pass follows it place by place
(:meth:`~acoustic_spaces.scene.model.AircraftModel.gains_db`). Its sound is noise
made to that power, one spectrum all along, with none of an aircraft's tones (its
fan's, its buzz-saw noise). Nothing of the simulation uses this module.
"""

from __future__ import annotations

import functools
import math

import numpy as np

#: Volume 3's aircraft the demo flies, and the procedures of each operation (Part 2,
#: table C-6): the ICAO-A noise abatement departure and the 3° approach.
JET = "JETW"
PROCEDURES = {"D": "ICAO_A", "A": "Descend"}


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


__all__ = [
    "JET",
    "OPERATIONS",
    "PROCEDURES",
    "engines_directivity",
    "flight_gains_db",
    "flight_power_db",
    "flight_problem",
    "flight_segments",
    "flown",
]
