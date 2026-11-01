"""An aircraft of Doc 29 in a scene: its NPD curves in another air (Volume 2 Appendix D,
acoustic_spaces.doc29.atmosphere) against Volume 2's worked example, and its sound power,
directivity and flight as a scene's source (acoustic_spaces.doc29.source)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from acoustic_spaces.doc29 import atmosphere, noise, path, performance, source, volume3
from acoustic_spaces.doc29.tables import FOOT_M

#: Volume 2 table D-3c: the attenuation (dB) of each third octave, 50 Hz to 10 kHz, at
#: 10 °C, 80 % and 101.325 kPa by SAE ARP-5534, 200 and 1000 ft away.
D3C = {
    200.0: (
        0.004, 0.007, 0.010, 0.016, 0.024, 0.034, 0.048, 0.064, 0.082, 0.101, 0.122, 0.147,
        0.178, 0.222, 0.287, 0.388, 0.545, 0.791, 1.178, 1.784, 2.726, 4.176, 6.373, 9.625,
    ),
    1000.0: (
        0.022, 0.034, 0.052, 0.079, 0.118, 0.170, 0.238, 0.318, 0.409, 0.506, 0.612, 0.734,
        0.890, 1.109, 1.434, 1.935, 2.717, 3.940, 5.855, 8.837, 13.435, 20.425, 30.805,
        45.700,
    ),
}  # fmt: skip
#: Volume 2 table D-4: the increments of JETW's departure (class 103) and approach
#: (class 205) at the NPD distances, there.
D4 = {
    "103": (0.1, 0.3, 0.4, 0.7, 1.2, 1.8, 2.1, 2.4, 2.9, 3.6),
    "205": (0.0, 0.2, 0.3, 0.6, 1.1, 1.7, 2.2, 2.7, 3.2, 3.7),
}
#: Volume 2 table D-6b: JETW's SEL NPD curves there, by operation and power.
D6B = {
    ("A", 2000.0): (100.7, 96.9, 94.1, 91.1, 86.1, 80.7, 76.9, 72.7, 68.1, 63.4),
    ("A", 2500.0): (100.9, 97.1, 94.3, 91.3, 86.3, 80.9, 77.1, 72.9, 68.3, 63.6),
    ("A", 7500.0): (102.5, 98.7, 95.9, 92.9, 87.9, 82.5, 78.7, 74.5, 69.9, 65.2),
    ("D", 10000.0): (100.6, 96.8, 94.0, 91.0, 86.0, 80.6, 76.6, 72.2, 67.6, 63.1),
    ("D", 15000.0): (103.9, 100.1, 97.3, 94.3, 89.3, 83.9, 79.9, 75.5, 70.9, 66.4),
    ("D", 20000.0): (108.1, 104.3, 101.5, 98.5, 93.5, 88.1, 84.1, 79.7, 75.1, 70.6),
    ("D", 22500.0): (109.8, 106.0, 103.2, 100.2, 95.2, 89.8, 85.8, 81.4, 76.8, 72.3),
}
WORKED = (10.0, 80.0, 101.325)


def test_the_air_takes_what_volume_2s_worked_example_says() -> None:
    """Each third octave's attenuation through ISO 9613-1, a flat spectrum averaged over
    the band, against table D-3c's (ARP-5534's): within 2.5 % 1000 ft away, and 200 ft
    away within 0.1 dB, where the table's three decimals are most of it."""
    near = atmosphere.band_attenuation_db(200.0 * FOOT_M, *WORKED)
    far = atmosphere.band_attenuation_db(1000.0 * FOOT_M, *WORKED)
    assert near == pytest.approx(D3C[200.0], abs=0.1)
    assert far == pytest.approx(D3C[1000.0], rel=0.025)
    # A longer path loses a band's top first: less per metre further away.
    assert np.all(far / 1000.0 <= near / 200.0 + 1e-12)


def test_the_increments_and_the_revised_npd_curves_are_volume_2s() -> None:
    """The increments of table D-4 and the revised SEL curves of table D-6b, within 0.1
    dB, the tables' rounding and the attenuation's few per cent; the increments grow
    with the distance."""
    classes = volume3.spectral_classes()
    for name, expected in D4.items():
        found = atmosphere.increments_db(classes[name], *WORKED)
        assert found == pytest.approx(expected, abs=0.1), name
        assert np.all(np.diff(found) > 0.0)
    npd = volume3.npd()
    jet = volume3.aircraft()["JETW"]
    for operation, name in (("A", jet.approach_spectrum), ("D", jet.departure_spectrum)):
        revised = atmosphere.adjusted(npd[("JETW", "SEL", operation)], classes[name], *WORKED)
        for power, row in zip(revised.powers, revised.levels, strict=True):
            assert row == pytest.approx(D6B[(operation, power)], abs=0.1), (operation, power)
    # The reference atmosphere's own increments are nought.
    same = npd[("JETW", "LAmax", "D")]
    assert np.all(np.isfinite(atmosphere.adjusted(same, classes["103"]).levels))


def test_the_source_gives_its_npd_level_1000_ft_under_it() -> None:
    """The sound power taken back out to 1000 ft (spherical spreading, the AIR-1845
    atmosphere, the soft ground's reflection) is the NPD curves' maximum level there at
    every power of the curves, and has the spectral class's shape; the ground adds from
    nearly 3 dB at the bottom (|Q| near 1) to under 0.5 dB at the top."""
    jet = volume3.aircraft()["JETW"]
    classes, npd = volume3.spectral_classes(), volume3.npd()
    gain = source.ground_gain_db()
    assert np.all(np.diff(gain) < 0.0)
    assert 2.7 < gain[0] < 10.0 * math.log10(2.0) and 0.3 < gain[-1] < 0.6
    rates = np.asarray(atmosphere.AIR_1845_DB_PER_M)
    weights = np.asarray(atmosphere.A_WEIGHTING_DB)
    for operation, name in (("D", jet.departure_spectrum), ("A", jet.approach_spectrum)):
        curves = npd[("JETW", "LAmax", operation)]
        spectrum = np.asarray(classes[name].levels)
        for power in curves.powers:
            power_db = source.sound_power_db(curves, classes[name], power)
            at = (
                power_db
                - 10.0 * math.log10(4.0 * math.pi * atmosphere.REFERENCE_M**2)
                - rates * atmosphere.REFERENCE_M
                + gain
            )
            level = 10.0 * math.log10(np.sum(10.0 ** ((at + weights) / 10.0)))
            assert level == pytest.approx(float(curves.level(power, atmosphere.REFERENCE_M)))
            assert np.ptp(at - spectrum) == pytest.approx(0.0, abs=1e-9)
        change = source.power_change_db(curves, curves.powers, curves.powers[-1])
        assert change[-1] == 0.0 and np.all(np.diff(change) > 0.0)


@pytest.mark.parametrize("installation", ["Wing", "Fuselage", "Prop"])
def test_the_directivity_is_the_installation_correction(installation) -> None:
    """The balloon sends each way the installation correction of its depression angle
    under the wing plane, across the flight path, over its mean; straight down (where
    the NPD curves are measured) it sends the power below, the scene's sound power less
    :func:`below_offset_db`. A propeller's is none."""
    balloon = source.directivity(installation)
    offset = source.below_offset_db(installation)
    down = float(balloon.energy([[0.0, 0.0, -1.0]])[0, 0])
    assert 10.0 * math.log10(down) == pytest.approx(-offset, abs=1e-6)
    ways = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.3, 0.5, -0.6]])
    ways /= np.linalg.norm(ways, axis=1, keepdims=True)
    phi = source.depression_deg(ways)
    assert phi[:3] == pytest.approx([0.0, 45.0, 0.0])
    assert phi[3] == pytest.approx(math.degrees(math.atan2(0.6, 0.5)))
    expected = noise.installation_db(installation, phi) - offset
    found = 10.0 * np.log10(balloon.energy(ways)[:, 0])
    assert found == pytest.approx(expected, abs=0.02)
    assert np.ptp(balloon.energy(ways), axis=1) == pytest.approx(np.zeros(len(ways)))
    if installation == "Prop":
        assert offset == pytest.approx(0.0, abs=1e-9)
    elif installation == "Wing":  # louder 45° below the wing than straight down or alongside
        assert found[1] > found[0] and found[1] > -offset
    else:  # engines on the fuselage: loudest straight down, the wing between
        assert -offset > found[1] > found[0]


def _departure_segments():
    plane = volume3.performance()["JETW"]
    steps = volume3.departure_procedures()[("JETW", "ICAO_A")]
    points = performance.departure(plane, steps, plane.weights[("D", "1")])
    profile = source.profile_between(performance.as_profile(points), 0.0, 6000.0)
    track = path.Track(points=((0.0, 0.0), (6000.0, 0.0)))
    return profile, path.flight_path(profile, track, departure=True)


def test_a_flight_keeps_doc_29s_speeds_powers_and_heights() -> None:
    """JETW's departure as keyframes: each segment at its constant acceleration (its
    speed's square even in its length), in legs of at most 5 m/s, each as long in time
    as its length over its ends' mean speed; the powers along it Doc 29's, the heights
    the segments', and the aircraft facing along each segment, turned within a second
    of its start."""
    profile, segments = _departure_segments()
    assert profile[0].distance_m == 0.0 and profile[-1].distance_m == pytest.approx(6000.0)
    flown = source.flight(segments)
    times, places = np.asarray(flown.times), np.asarray(flown.positions)
    speeds = np.asarray(flown.speeds)
    legs = np.linalg.norm(np.diff(places, axis=0), axis=1)
    assert np.all(np.diff(times) > 0.0) and np.all(np.abs(np.diff(speeds)) <= 5.0 + 1e-9)
    assert legs / np.diff(times) == pytest.approx((speeds[1:] + speeds[:-1]) / 2.0)
    total = sum(np.linalg.norm(np.subtract(s.end, s.start)) * 2.0 / sum(s.speeds) for s in segments)
    assert flown.duration_s == pytest.approx(total)
    assert places[0] == pytest.approx(segments[0].start) and places[-1] == pytest.approx(
        segments[-1].end
    )
    assert flown.power_at(times) == pytest.approx(flown.powers)
    assert flown.power_at([-1.0, 1e6]) == pytest.approx([flown.powers[0], flown.powers[-1]])
    # Each keyframe faces along its segment once a second into it.
    for segment in segments:
        d = np.subtract(segment.end, segment.start)
        climb = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        inside = [
            k for k, p in enumerate(places) if segment.start[0] < p[0] <= segment.end[0] and k > 0
        ]
        for k in inside[1:] if len(inside) > 1 else inside:
            assert flown.orientations[k][1] == pytest.approx(climb, abs=1e-9)
    frames = flown.keyframes()
    assert len(frames) == len(times) - 1 and frames[0]["time_s"] == round(times[1], 4)
    assert set(frames[-1]["orientation"]) == {"yaw", "pitch", "roll"}


def test_a_profiles_part_is_interpolated_as_doc_29_does() -> None:
    """The part of a profile along a stretch of track ends where it is cut, its height
    linear, its speed and power in their squares (eq. 3-2); a stretch past the profile
    keeps to the profile."""
    plane = volume3.performance()["JETW"]
    steps = volume3.approach_procedures()[("JETW", "Descend")]
    profile = performance.as_profile(performance.arrival(plane, steps))
    part = source.profile_between(profile, -3000.0, 1e6)
    assert part[0].distance_m == -3000.0
    assert dataclasses.astuple(part[-1]) == pytest.approx(dataclasses.astuple(profile[-1]))
    a, b = next(
        (p, q)
        for p, q in zip(profile[:-1], profile[1:], strict=True)
        if p.distance_m < -3000.0 <= q.distance_m
    )
    f = (-3000.0 - a.distance_m) / (b.distance_m - a.distance_m)
    assert part[0].height_m == pytest.approx(a.height_m + f * (b.height_m - a.height_m))
    assert part[0].speed_m_s ** 2 == pytest.approx(
        a.speed_m_s**2 + f * (b.speed_m_s**2 - a.speed_m_s**2)
    )
    with pytest.raises(ValueError, match="share no part"):
        source.profile_between(profile, 1e6, 2e6)
