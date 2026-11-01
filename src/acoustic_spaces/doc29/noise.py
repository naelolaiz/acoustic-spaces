"""The noise of one aircraft movement at receivers by Doc 29 (Volume 2 chapter 4):
each segment's level from the NPD curves at its power and distance, corrected for the
speed, the engines' installation, the lateral attenuation, the segment's finite
length and, behind a take-off roll, the start-of-roll directivity; summed over the
segments to the event's sound exposure level, or its maximum level taken."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .path import Segment
from .tables import FOOT_M, KNOT_M_S, Aircraft, NpdCurves

#: The NPD curves' reference speed (m/s), 160 kt, and the time t0 of the scaled
#: distance of the finite segment correction (s) (section 4.5.6, Appendix E).
REFERENCE_SPEED_M_S = 160.0 * KNOT_M_S
T0_S = 1.0
#: The nearest distance read off the NPD curves (m), section 4.2.
NEAREST_NPD_M = 30.0
#: The lowest finite segment correction (dB), section 4.5.6.
LOWEST_FRACTION_DB = -150.0
#: The distance beyond which the start-of-roll directivity weakens (m), 2500 ft
#: (eq. 4-25).
SOR_NORMALISING_M = 762.0
#: The acoustic impedance of the NPD curves' reference atmosphere (N s/m3), eq. 4-6.
REFERENCE_IMPEDANCE = 409.81
#: The engine installation's coefficients a, b, c (eq. 4-15).
INSTALLATION = {
    "wing": (0.0039, 0.062, 0.8786),
    "fuselage": (0.1225, 0.329, 1.0),
}


def impedance_db(temperature_c: float = 15.0, pressure_kpa: float = 101.325) -> float:
    """The impedance adjustment of the NPD levels to the air at the aerodrome (dB),
    eq. 4-6 and 4-7."""
    delta = pressure_kpa / 101.325
    theta = (temperature_c + 273.15) / (15.0 + 273.15)
    return 10.0 * math.log10(416.86 * delta / math.sqrt(theta) / REFERENCE_IMPEDANCE)


def installation_db(installation: str, depression_deg: np.ndarray) -> np.ndarray:
    """The engine installation correction (dB) at the depression angle (degrees) under
    the wing plane, eq. 4-15; nought for propellers, and at negative angles that at
    nought."""
    phi = np.radians(np.maximum(np.asarray(depression_deg, dtype=float), 0.0))
    coefficients = INSTALLATION.get(installation.lower())
    if coefficients is None:
        return np.zeros_like(phi)
    a, b, c = coefficients
    top = (a * np.cos(phi) ** 2 + np.sin(phi) ** 2) ** b
    bottom = c * np.sin(2 * phi) ** 2 + np.cos(2 * phi) ** 2
    return np.asarray(10.0 * np.log10(top / bottom))


def lateral_attenuation_db(elevation_deg: np.ndarray, lateral_m: np.ndarray) -> np.ndarray:
    """The lateral attenuation (dB) at an elevation angle (degrees) and a lateral
    displacement from the ground track (m), eq. 4-18 and 4-19."""
    beta = np.asarray(elevation_deg, dtype=float)
    ell = np.asarray(lateral_m, dtype=float)
    gamma = np.where(ell <= 914.0, 1.089 * (1.0 - np.exp(-0.00274 * ell)), 1.0)
    far = np.where(
        beta < 0.0,
        10.857,
        np.where(beta <= 50.0, 1.137 - 0.0229 * beta + 9.72 * np.exp(-0.142 * beta), 0.0),
    )
    return np.asarray(gamma * far)


def fraction_db(alpha1: np.ndarray, alpha2: np.ndarray) -> np.ndarray:
    """The finite segment correction (dB) of eq. 4-20 for the ends ``alpha1`` and
    ``alpha2`` of the segment in scaled distances."""

    def part(a: np.ndarray) -> np.ndarray:
        return a / (1.0 + a**2) + np.arctan(a)

    energy = (part(np.asarray(alpha2)) - part(np.asarray(alpha1))) / math.pi
    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(np.maximum(energy, 0.0))
    return np.asarray(np.maximum(level, LOWEST_FRACTION_DB))


def start_of_roll_db(jet: bool, azimuth_deg: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    """The start-of-roll directivity (dB) behind a take-off roll at the azimuth from the
    aircraft's heading (90 to 180 degrees) and the distance (m), eq. 4-22 to 4-25."""
    psi = np.clip(np.asarray(azimuth_deg, dtype=float), 90.0, 180.0)
    r = np.pi * psi / 180.0
    if jet:
        level = (
            2329.44
            - 8.0573 * psi
            + 11.51 * np.exp(r)
            - 3.4601 * psi / np.log(r)
            - 17403338.3 * np.log(r) / psi**2
        )
    else:
        level = (
            -34643.898
            + 30722162 / psi
            - 11491573931 / psi**2
            + 2.34928567e12 / psi**3
            - 2.83584442e14 / psi**4
            + 2.02271504e16 / psi**5
            - 7.90084471e17 / psi**6
            + 1.30506872e19 / psi**7
        )
    d = np.asarray(distance_m, dtype=float)
    return np.asarray(np.where(d > SOR_NORMALISING_M, level * SOR_NORMALISING_M / d, level))


@dataclass(frozen=True)
class SegmentLevels:
    """One segment's contribution at each receiver, with the terms that make it up
    (arrays, one value a receiver): the distances (m), angles (degrees) and levels
    (dB) Volume 3 lists for its reference cases."""

    q: np.ndarray
    slant: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    lateral: np.ndarray
    npd_distance: np.ndarray
    power: np.ndarray
    elevation: np.ndarray
    climb: float
    depression: np.ndarray
    installation: np.ndarray
    lateral_attenuation: np.ndarray
    baseline: np.ndarray
    duration: np.ndarray
    maximum: np.ndarray
    fraction: np.ndarray
    start_of_roll: np.ndarray
    impedance: float
    exposure: np.ndarray
    #: The segment's maximum level at each receiver (dB(A)), eq. 4-8a.
    peak: np.ndarray


def _power(segment: Segment, q: np.ndarray, length: float) -> np.ndarray:
    p1, p2 = segment.powers
    f = np.clip(q / length, 0.0, 1.0)
    if p1 >= 0 and p2 >= 0:
        p = np.sqrt(p1**2 + f * (p2**2 - p1**2))
    elif p1 < 0 and p2 < 0:
        p = -np.sqrt(p1**2 + f * (p2**2 - p1**2))
    elif p1 < 0:
        p = p1 + (p2 - p1) * np.sqrt(f)
    else:
        p = p2 + (p1 - p2) * np.sqrt(1.0 - f)
    return np.maximum(p, 1.0)


def segment_levels(
    aircraft: Aircraft,
    exposure: NpdCurves,
    maximum: NpdCurves,
    segment: Segment,
    receivers: np.ndarray,
    impedance: float = 0.0,
) -> SegmentLevels:
    """The contribution of ``segment`` of a flight of ``aircraft`` at ``receivers``
    (an array of x, y, z in m, one a row), its exposure from the SEL curves
    ``exposure`` and its maximum level from the LAmax curves ``maximum``."""
    o = np.atleast_2d(np.asarray(receivers, dtype=float))
    s1 = np.asarray(segment.start, dtype=float)
    s2 = np.asarray(segment.end, dtype=float)
    ab = s2 - s1
    length = float(np.linalg.norm(ab))
    u = ab / length
    horizontal = float(np.hypot(u[0], u[1]))
    climb = math.degrees(math.atan2(u[2], horizontal))
    rel = o - s1
    q = rel @ u
    foot = s1 + np.outer(q, u)
    slant = np.linalg.norm(o - foot, axis=1)
    d1 = np.linalg.norm(o - s1, axis=1)
    d2 = np.linalg.norm(o - s2, axis=1)
    behind, ahead = q < 0.0, q > length
    alongside = ~(behind | ahead)
    nearest = np.where(behind, d1, np.where(ahead, d2, slant))

    t = np.array([u[0], u[1]]) / horizontal if horizontal > 0 else np.array([1.0, 0.0])
    cross = t[0] * rel[:, 1] - t[1] * rel[:, 0]
    lateral = np.abs(cross)
    # The extended path's closest point: its elevation from the receiver, the sign of
    # its height over the receiver.
    above = foot[:, 2] - o[:, 2]
    rise = np.sign(above) * np.sqrt(np.maximum(slant**2 - lateral**2, 0.0))
    beta_p = np.degrees(np.arctan2(rise, lateral))
    # The equivalent level path's elevation: through the closest point alongside, at
    # the height of the nearest end's perpendicular to the ground track otherwise.
    end_height = np.where(behind, s1[2], s2[2]) - o[:, 2]
    h = np.maximum(end_height, 0.0) / math.cos(math.radians(climb))
    beta = np.where(alongside, beta_p, np.degrees(np.arctan2(h, lateral)))
    ell = lateral.copy()
    npd_distance = slant.copy()

    take_off_behind = (segment.roll == "take-off") & behind
    landing_ahead = (segment.roll == "landing") & ahead
    z1, z2 = s1[2] - o[:, 2], s2[2] - o[:, 2]
    beta1 = np.degrees(np.arcsin(np.clip(z1 / d1, -1.0, 1.0)))
    beta2 = np.degrees(np.arcsin(np.clip(z2 / d2, -1.0, 1.0)))
    ell1 = np.sqrt(np.maximum(d1**2 - z1**2, 0.0))
    ell2 = np.sqrt(np.maximum(d2**2 - z2**2, 0.0))
    beta = np.where(take_off_behind, beta1, np.where(landing_ahead, beta2, beta))
    ell = np.where(take_off_behind, ell1, np.where(landing_ahead, ell2, ell))
    npd_distance = np.where(take_off_behind | landing_ahead, nearest, npd_distance)

    bank1, bank2 = segment.banks
    along = bank1 + np.clip(q / length, 0.0, 1.0) * (bank2 - bank1)
    bank = np.where(behind, bank1, np.where(ahead, bank2, along))
    starboard = cross < 0.0
    depression = np.maximum(
        np.where(take_off_behind, beta1, beta_p) + np.where(starboard, bank, -bank), 0.0
    )

    power = _power(segment, q, length)
    v1, v2 = segment.speeds
    if segment.roll:
        speed = np.full_like(q, (v1 + v2) / 2.0)
    else:
        f = np.clip(q / length, 0.0, 1.0)
        speed = np.sqrt(v1**2 + f * (v2**2 - v1**2))
    duration = 10.0 * np.log10(REFERENCE_SPEED_M_S / speed)

    distance = np.maximum(npd_distance, NEAREST_NPD_M)
    baseline = exposure.level(power, distance)
    peak_npd = maximum.level(power, distance)
    installation = installation_db(aircraft.installation, depression)
    attenuation = lateral_attenuation_db(beta, ell)

    d0 = 2.0 / math.pi * REFERENCE_SPEED_M_S * T0_S
    scaled = d0 * 10.0 ** ((baseline - peak_npd) / 10.0)
    alpha1 = -q / scaled
    alpha2 = -(q - length) / scaled
    fraction = np.where(
        take_off_behind,
        fraction_db(np.zeros_like(q), length / scaled),
        np.where(
            landing_ahead,
            fraction_db(-length / scaled, np.zeros_like(q)),
            fraction_db(alpha1, alpha2),
        ),
    )
    sor = np.zeros_like(q)
    if segment.roll == "take-off":
        # The azimuth in the plane of the runway, from the heading to the receiver.
        psi = np.degrees(np.arctan2(lateral, q))
        sor = np.where(behind, start_of_roll_db(aircraft.jet, psi, d1), 0.0)
    level = baseline + duration + installation - attenuation + fraction + sor + impedance

    # The maximum level: the NPD curve at the nearest point, the attenuation by the
    # elevation of and the ground distance to that point.
    near_z = np.where(behind, z1, np.where(ahead, z2, foot[:, 2] - o[:, 2]))
    beta_max = np.degrees(np.arcsin(np.clip(near_z / np.maximum(nearest, 1e-9), -1.0, 1.0)))
    ell_max = np.sqrt(np.maximum(nearest**2 - near_z**2, 0.0))
    peak = (
        maximum.level(power, np.maximum(nearest, NEAREST_NPD_M))
        + installation_db(aircraft.installation, beta_max + np.where(starboard, bank, -bank))
        - lateral_attenuation_db(beta_max, ell_max)
        + sor
        + impedance
    )
    return SegmentLevels(
        q=q,
        slant=slant,
        d1=d1,
        d2=d2,
        lateral=ell,
        npd_distance=npd_distance,
        power=power,
        elevation=beta,
        climb=climb,
        depression=depression,
        installation=installation,
        lateral_attenuation=attenuation,
        baseline=baseline,
        duration=duration,
        maximum=peak_npd,
        fraction=fraction,
        start_of_roll=sor,
        impedance=impedance,
        exposure=level,
        peak=peak,
    )


@dataclass(frozen=True)
class Event:
    """One movement's levels at the receivers (dB(A)), and each segment's part."""

    sel: np.ndarray
    lamax: np.ndarray
    segments: tuple[SegmentLevels, ...]


def event(
    aircraft: Aircraft,
    exposure: NpdCurves,
    maximum: NpdCurves,
    segments: Sequence[Segment],
    receivers: np.ndarray,
    impedance: float = 0.0,
) -> Event:
    """The sound exposure level (eq. 4-11) and the maximum level (eq. 4-10) of the
    flight ``segments`` of ``aircraft`` at ``receivers``."""
    parts = tuple(
        segment_levels(aircraft, exposure, maximum, s, receivers, impedance) for s in segments
    )
    exposures = np.array([p.exposure for p in parts])
    peaks = np.array([p.peak for p in parts])
    sel = 10.0 * np.log10(np.sum(10.0 ** (exposures / 10.0), axis=0))
    return Event(sel=sel, lamax=peaks.max(axis=0), segments=parts)


__all__ = [
    "FOOT_M",
    "Event",
    "SegmentLevels",
    "event",
    "fraction_db",
    "impedance_db",
    "installation_db",
    "lateral_attenuation_db",
    "segment_levels",
    "start_of_roll_db",
]
