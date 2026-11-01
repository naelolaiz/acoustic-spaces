"""An aircraft of Doc 29 as a source of a scene: the sound power and the directivity that
give its NPD curves' levels in the engine, and its flight as a walk's keyframes.

Doc 29 gives an aircraft's sound as levels at receivers: its NPD curves (the maximum
level and the sound exposure level at distances from 200 to 25,000 ft, at each power,
in the AIR-1845 atmosphere, at 160 kt) and its spectral class (the spectrum 1000 ft
under it at the time of its maximum level). The NPD curves are measured 1.2 m above
flat soft ground (Volume 2 section 4.2), so they hold that ground's reflection. A point
source of the scene that sends the same sound gets:

- **its spectrum** from the spectral class, scaled to the NPD curves' maximum level
  1000 ft under the aircraft at its power (:func:`sound_power_db`);
- **its level** back at the source: the AIR-1845 atmosphere's attenuation over the
  1000 ft added back (Appendix D eq. D-1), and the soft ground's reflection taken out
  (:func:`ground_gain_db`): grass (Nord2000's class D, 200 kPa s/m², Miki's
  impedance) at normal incidence, its reflection's energy added to the direct sound's,
  as no coherent dip shows in the spectral classes (the first would be near 70 Hz);
- **its directivity** from the engines' installation (eq. 4-15,
  :func:`~acoustic_spaces.doc29.noise.installation_db`), the depression angle under
  the wing plane measured across the flight path, the same in every band
  (:func:`directivity`); nought straight down, where the NPD curves are measured, so the
  scene's sound power is the power below that much over (:func:`below_offset_db`).

Doc 29 gives the NPD curves no directivity along the flight path but behind a take-off
roll (eq. 4-22), which the scene's source does not hold, nor the impedance adjustment
(eq. 4-6), a tenth of a decibel. The NPD curves at the powers of a flight differ by
nearly the same at every distance: the aircraft's sound power changes with its power
as the NPD curves' maximum level 1000 ft away does (:func:`power_change_db`), its
spectrum kept.

:func:`flight` turns a flight path's segments (:func:`~acoustic_spaces.doc29.path.flight_path`)
into keyframes, when the aircraft is where: along a segment its speed's square changes
evenly with the distance (a constant acceleration, as Doc 29 interpolates it), cut into
legs whose speeds differ by at most :data:`LEG_SPEED_STEP_M_S`, each leg timed for that
acceleration. The aircraft faces the way it flies, along each segment's heading and
climb, and turns to the next one's over its first :data:`TURN_S`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .atmosphere import A_WEIGHTING_DB, AIR_1845_DB_PER_M, REFERENCE_M, SAMPLES
from .noise import installation_db
from .path import Segment, power_between
from .tables import NpdCurves, ProfilePoint, SpectralClass

#: How high the NPD curves' microphone is over the ground (m), Volume 2 section 4.2,
#: and the flow resistivity of its soft ground (kPa s/m²), taken to be grass
#: (Nord2000's class D, :data:`acoustic_spaces.materials.ground.PRESETS`).
MICROPHONE_M = 1.2
SOFT_GROUND_KPA_S_M2 = 200.0
#: How much the speed may change along one leg of a flight's keyframes (m/s).
LEG_SPEED_STEP_M_S = 5.0
#: How long the aircraft takes to turn to a segment's heading and climb (s).
TURN_S = 1.0
#: A segment shorter than this (m) is passed over: where the track's nodes and the
#: profile's points nearly meet (the walk's own keyframes as a track), the flight path
#: has segments a few micrometres long, flown in microseconds.
SHORTEST_M = 1e-3


def _centres() -> np.ndarray:
    """The spectral classes' third octaves' exact mid-band frequencies (Hz, base 10)."""
    return 1000.0 * 10.0 ** ((np.arange(len(AIR_1845_DB_PER_M)) - 13) / 10.0)


def _a_level(levels: np.ndarray) -> float:
    return float(10.0 * np.log10(np.sum(10.0 ** ((levels + np.array(A_WEIGHTING_DB)) / 10.0))))


def ground_gain_db(temperature_c: float = 15.0) -> np.ndarray:
    """(24,) what the soft ground under the NPD curves' microphone adds to the sound of
    an aircraft 1000 ft from it straight above, per third octave of the spectral
    classes, 50 Hz to 10 kHz (dB): the reflection's energy added to the direct sound's,
    ``1 + |Q|² (r1 / r2)²``, averaged over the band, Q the reflection factor of a point
    source's sound off grass at normal incidence
    (:func:`~acoustic_spaces.materials.ground.spherical_reflection`), r1 the 1000 ft and
    r2 the image's path, 2.4 m longer, in air at ``temperature_c``."""
    from acoustic_spaces.air import speed_of_sound
    from acoustic_spaces.materials.ground import spherical_reflection, surface_impedance

    c = speed_of_sound(temperature_c)
    direct, image = REFERENCE_M, REFERENCE_M + 2.0 * MICROPHONE_M
    out = np.empty(len(AIR_1845_DB_PER_M))
    for k, centre in enumerate(_centres()):
        f = np.geomspace(centre * 10.0**-0.05, centre * 10.0**0.05, SAMPLES)
        admittance = 1.0 / surface_impedance(f, SOFT_GROUND_KPA_S_M2)
        q = spherical_reflection(
            admittance, np.ones_like(f), 2.0 * np.pi * f / c, np.full_like(f, image)
        )
        out[k] = 10.0 * math.log10(float(np.mean(1.0 + np.abs(q) ** 2 * (direct / image) ** 2)))
    return out


def sound_power_db(
    maximum: NpdCurves,
    spectral_class: SpectralClass,
    power: float,
    temperature_c: float = 15.0,
) -> np.ndarray:
    """(24,) the sound power (dB re 1 pW) per third octave, 50 Hz to 10 kHz, of a point
    source that sends each way what the aircraft sends straight down at ``power``: in a
    free field of the AIR-1845 atmosphere, 1000 ft from it, the spectrum of
    ``spectral_class`` at the maximum level of its NPD curves ``maximum`` there, less
    the NPD's soft ground (:func:`ground_gain_db`)."""
    levels = np.asarray(spectral_class.levels, dtype=float)
    levels = levels + float(maximum.level(power, REFERENCE_M)) - _a_level(levels)
    free = levels - ground_gain_db(temperature_c)
    rates = np.asarray(AIR_1845_DB_PER_M)
    return free + rates * REFERENCE_M + 10.0 * math.log10(4.0 * math.pi * REFERENCE_M**2)


def power_change_db(maximum: NpdCurves, power, reference: float) -> np.ndarray:
    """How much louder the aircraft is at ``power`` (a number or an array) than at
    ``reference`` (dB): the maximum levels of its NPD curves ``maximum`` 1000 ft away."""
    here = np.asarray(maximum.level(np.asarray(power, dtype=float), REFERENCE_M))
    return here - float(maximum.level(reference, REFERENCE_M))


def depression_deg(directions) -> np.ndarray:
    """(n,) the depression angle (degrees) under the wing plane of ``directions`` (n, 3)
    in the aircraft's frame (x forward, y left, z up), measured across the flight path:
    ``atan2(-z, |y|)``, negative above the wing plane; nought straight ahead and behind."""
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    return np.degrees(np.arctan2(-d[:, 2], np.abs(d[:, 1])))


def _table(installation: str) -> np.ndarray:
    """(elevations, azimuths, 8) the energy the installation gives each way of the
    balloons' grid, 10^(ΔI / 10), in every octave band."""
    from acoustic_spaces.directivity import GRID_STEP_DEG, _from_angles, _grid
    from acoustic_spaces.materials import OCTAVE_BANDS_HZ

    elevation, azimuth = _grid(GRID_STEP_DEG)
    ways = _from_angles(azimuth[None, :], elevation[:, None]).reshape(-1, 3)
    energy = 10.0 ** (installation_db(installation, depression_deg(ways)) / 10.0)
    table = energy.reshape(len(elevation), len(azimuth), 1)
    return np.repeat(table, len(OCTAVE_BANDS_HZ), axis=-1)


def directivity(installation: str, name: str = ""):
    """The directivity (a :class:`~acoustic_spaces.directivity.Balloon` of octave bands)
    of the engines' ``installation`` ("Wing", "Fuselage" or a propeller's, which has
    none): the installation correction each way (eq. 4-15) at its depression angle
    (:func:`depression_deg`), the same in every band, over its mean on the sphere."""
    from acoustic_spaces.directivity import Balloon

    return Balloon.from_table(_table(installation), name)


def below_offset_db(installation: str) -> float:
    """How far the sound power of a source with the installation's :func:`directivity`
    is over the power of one that sends each way what it sends straight down (dB),
    where the installation correction is nought: 10 lg of its energy's mean over the
    sphere, as the balloon is scaled to average 1."""
    from acoustic_spaces.directivity import sphere_mean

    return float(10.0 * np.log10(sphere_mean(_table(installation))[0]))


@dataclass(frozen=True)
class Flight:
    """A flight as a walk's keyframes: when (s from its start) the aircraft is where
    (x, y, z in m), how it faces there (yaw, pitch and roll in degrees, as a scene's
    orientation), and its speed (m/s) and power there. It flies at one speed along each
    leg, from one keyframe to the next, and its power changes along the leg as Doc 29's
    does along a segment (eq. 3-2)."""

    times: tuple[float, ...]
    positions: tuple[tuple[float, float, float], ...]
    orientations: tuple[tuple[float, float, float], ...]
    speeds: tuple[float, ...]
    powers: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return self.times[-1]

    def power_at(self, times) -> np.ndarray:
        """The power at ``times`` (s), each in its leg (eq. 3-2, the share of the leg's
        length the share of its time, at one speed); the first's before the start, the
        last's after the end."""
        at = np.atleast_1d(np.asarray(times, dtype=float))
        known = np.asarray(self.times)
        legs = np.clip(np.searchsorted(known, at, side="right") - 1, 0, len(known) - 2)
        out = np.empty(len(at))
        for k, (t, leg) in enumerate(zip(at, legs, strict=True)):
            share = (t - known[leg]) / (known[leg + 1] - known[leg])
            share = min(max(share, 0.0), 1.0)
            out[k] = power_between(self.powers[leg], self.powers[leg + 1], share)
        return out

    def keyframes(self, digits: int = 4) -> list[dict]:
        """The keyframes after the first, as a scene's walk takes them (the first is the
        source's own position and orientation), rounded to ``digits``."""
        return [
            {
                "time_s": round(t, digits),
                "position": [round(v, digits) for v in p],
                "orientation": dict(
                    zip(("yaw", "pitch", "roll"), (round(v, digits) for v in o), strict=True)
                ),
            }
            for t, p, o in zip(
                self.times[1:], self.positions[1:], self.orientations[1:], strict=True
            )
        ]


def profile_between(
    profile: Sequence[ProfilePoint], first_m: float, last_m: float
) -> tuple[ProfilePoint, ...]:
    """The part of ``profile`` from ``first_m`` to ``last_m`` along its ground track (m,
    as its distances go), its ends interpolated as Doc 29 does between two points (the
    height linear in the distance, the speed and the power in its square, eq. 3-2); a
    profile shorter than that is kept to its own ends."""
    from .path import Node, interpolate

    nodes = [Node(p.distance_m, p.height_m, p.speed_m_s, p.power) for p in profile]
    first, last = max(first_m, nodes[0].s), min(last_m, nodes[-1].s)
    if not first < last:
        raise ValueError("the profile and the track share no part")

    def at(s: float) -> Node:
        k = int(np.clip(np.searchsorted([n.s for n in nodes], s) - 1, 0, len(nodes) - 2))
        return interpolate(nodes[k], nodes[k + 1], s)

    kept = [at(first), *(n for n in nodes if first < n.s < last), at(last)]
    return tuple(ProfilePoint(n.s, n.height, n.speed, n.power) for n in kept)


def _facing(segment: Segment) -> tuple[float, float, float]:
    """The yaw, pitch and roll (degrees) along ``segment``: its heading, its climb, and
    its bank angle (positive in a left turn, a roll to the left)."""
    d = np.subtract(segment.end, segment.start)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return yaw, pitch, 0.0 - segment.banks[0]


def flight(segments: Sequence[Segment], step_m_s: float = LEG_SPEED_STEP_M_S) -> Flight:
    """The keyframes of a flight along ``segments``, in order, from the first's start at
    0 s (see the module): each segment at a constant acceleration, its speed's square
    changing evenly with its length, cut into legs whose speeds differ by at most
    ``step_m_s``; each leg's time is its length over the mean of its ends' speeds."""
    if not segments:
        raise ValueError("a flight needs a segment at least")
    times, positions, orientations = [0.0], [tuple(segments[0].start)], [_facing(segments[0])]
    speeds, powers = [segments[0].speeds[0]], [segments[0].powers[0]]
    for segment in segments:
        start, end = np.asarray(segment.start, dtype=float), np.asarray(segment.end, dtype=float)
        length = float(np.linalg.norm(end - start))
        if length < SHORTEST_M:
            continue
        v1, v2 = segment.speeds
        if v1 <= 0.0 and v2 <= 0.0:
            raise ValueError("a segment of a flight has a speed at one end at least")
        facing = _facing(segment)
        # Where along it (m) and when (s from its start) each leg ends, and its speed.
        if math.isclose(v1, v2, rel_tol=1e-12, abs_tol=1e-12):
            ends = [(length, length / v1, v2)]
        else:
            n = max(1, math.ceil(abs(v2 - v1) / step_m_s - 1e-9))
            accel = (v2**2 - v1**2) / (2.0 * length)
            ends = []
            for k in range(1, n + 1):
                v = v1 + k * (v2 - v1) / n
                ends.append(((v**2 - v1**2) / (2.0 * accel), (v - v1) / accel, v))
        turned = not np.allclose(facing, orientations[-1], atol=1e-3)
        if turned:
            # Turn over the first TURN_S of the segment (or half of its first leg).
            s, t, _ = ends[0]
            first = min(TURN_S, t / 2.0)
            v_turn = v1 + (ends[0][2] - v1) * first / t
            s_turn = first * (v1 + v_turn) / 2.0
            ends.insert(0, (s_turn, first, v_turn))
        t0 = times[-1]
        for s, t, v in ends:
            share = s / length
            times.append(t0 + t)
            positions.append(tuple(float(x) for x in start + share * (end - start)))
            orientations.append(facing)
            speeds.append(float(v))
            powers.append(power_between(segment.powers[0], segment.powers[1], share))
    return Flight(
        times=tuple(float(t) for t in times),
        positions=tuple(positions),  # type: ignore[arg-type]
        orientations=tuple(orientations),
        speeds=tuple(speeds),
        powers=tuple(float(p) for p in powers),
    )


__all__ = [
    "LEG_SPEED_STEP_M_S",
    "MICROPHONE_M",
    "SHORTEST_M",
    "SOFT_GROUND_KPA_S_M2",
    "TURN_S",
    "Flight",
    "below_offset_db",
    "depression_deg",
    "directivity",
    "flight",
    "ground_gain_db",
    "power_change_db",
    "profile_between",
    "sound_power_db",
]
