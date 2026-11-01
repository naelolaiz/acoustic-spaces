"""Aircraft noise by ECAC Doc 29 (acoustic_spaces.doc29) against Volume 3's reference
cases and Volume 2's equations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acoustic_spaces import doc29
from acoustic_spaces.doc29 import noise as ev
from acoustic_spaces.doc29 import path, volume3
from acoustic_spaces.doc29.tables import FOOT_M, KNOT_M_S, NPD_FEET


def test_reference_levels() -> None:
    """All 99 sound exposure levels of Volume 3's table B-1, to its 0.01 dB."""
    reference = volume3.sel_results()
    receptors = volume3.receptors()
    assert len(reference) == 99
    for case in volume3.CASES:
        names = [r for (c, r) in reference if c == case.name]
        levels = case.event(np.array([receptors[r] for r in names])).sel
        expected = [reference[(case.name, r)] for r in names]
        assert levels == pytest.approx(expected, abs=0.006), case.name


def test_segment_counts() -> None:
    """Each case's flight path has as many segments as Volume 3's segment results."""
    counts = {case.name: len(case.segments()) for case in volume3.CASES}
    assert counts == {
        "JETFAC": 43, "JETFAS": 33, "JETFDC": 39, "JETFDS": 29,
        "JETWAC": 43, "JETWAS": 33, "JETWDC": 39, "JETWDS": 29,
        "PROPAC": 32, "PROPAS": 22, "PROPDC": 37, "PROPDS": 27,
    }  # fmt: skip


#: Rows of Volume 3's segment results (worksheet B-2): (case, receptor, segment) and
#: its lateral attenuation, finite segment correction, start-of-roll correction and
#: segment SEL (dB): an observer ahead of an airborne segment, behind the take-off roll
#: (its start-of-roll directivity), beside the initial climb, ahead of the landing roll
#: (its own finite segment correction) and alongside an approach segment.
SEGMENTS = [
    (("JETFAC", "R02", 1), 6.376859, -5.787474, 0.0, 28.372045),
    (("JETFDS", "R03", 1), 8.689253, -14.997404, -13.479123, 69.679887),
    (("PROPDS", "R01", 9), 0.0, -49.642808, 0.0, 42.958995),
    (("JETFAS", "R05", 33), None, -16.349672, 0.0, 54.098788),
    (("JETFAS", "R04", 24), 4.870563, None, 0.0, 76.870729),
]


@pytest.mark.parametrize(("where", "attenuation", "fraction", "sor", "level"), SEGMENTS)
def test_reference_segments(where, attenuation, fraction, sor, level) -> None:
    name, receptor, number = where
    case = volume3.Case.named(name)
    part = case.event(np.array([volume3.receptors()[receptor]])).segments[number - 1]
    if attenuation is not None:
        assert float(part.lateral_attenuation[0]) == pytest.approx(attenuation, abs=0.002)
    if fraction is not None:
        assert float(part.fraction[0]) == pytest.approx(fraction, abs=0.002)
    assert float(part.start_of_roll[0]) == pytest.approx(sor, abs=0.002)
    assert float(part.exposure[0]) == pytest.approx(level, abs=0.003)


def test_npd_curves() -> None:
    """The NPD curves give their table at its powers and distances, interpolate
    linearly in power and in log distance, and extrapolate the last two beyond."""
    curves = volume3.npd()[("JETF", "SEL", "D")]
    for power, row in zip(curves.powers, curves.levels, strict=True):
        got = curves.level(np.full(10, power), np.array(NPD_FEET) * FOOT_M)
        assert got == pytest.approx(row, abs=1e-9)
    mid = curves.level(12500.0, math.sqrt(200 * 400) * FOOT_M)
    a, b = curves.levels[0][:2], curves.levels[1][:2]
    assert float(mid) == pytest.approx((sum(a) / 2 + sum(b) / 2) / 2)
    row = curves.levels[-1]
    slope = (row[-1] - row[-2]) / math.log10(25000 / 16000)
    far = curves.level(22500.0, 50000 * FOOT_M)
    assert float(far) == pytest.approx(row[-1] + slope * math.log10(2.0))


def test_corrections() -> None:
    """Volume 2's corrections at the values it states."""
    assert ev.impedance_db() == pytest.approx(0.074, abs=0.0005)
    assert ev.installation_db("Fuselage", np.array([0.0, 90.0])) == pytest.approx(
        [10 * 0.329 * math.log10(0.1225), 0.0]
    )
    assert float(ev.installation_db("Wing", np.array(-5.0))) == pytest.approx(
        10 * 0.062 * math.log10(0.0039)
    )
    assert float(ev.installation_db("Prop", np.array(10.0))) == 0.0
    assert float(ev.lateral_attenuation_db(np.array(0.0), np.array(2000.0))) == pytest.approx(
        10.857
    )
    assert float(ev.lateral_attenuation_db(np.array(60.0), np.array(100.0))) == 0.0
    assert float(ev.lateral_attenuation_db(np.array(-1.0), np.array(2000.0))) == 10.857
    # The start-of-roll directivity of a jet straight behind it, and a turboprop's.
    assert float(ev.start_of_roll_db(True, np.array(180.0), np.array(500.0))) == pytest.approx(
        -13.4791, abs=1e-4
    )
    far = ev.start_of_roll_db(True, np.array(180.0), np.array(1524.0))
    assert float(far) == pytest.approx(-13.4791 / 2, abs=1e-4)
    # An infinite segment has all the energy, half of one ends beside the receiver.
    assert float(ev.fraction_db(np.array(-1e9), np.array(1e9))) == pytest.approx(0.0)
    assert float(ev.fraction_db(np.array(0.0), np.array(1e9))) == pytest.approx(
        10 * math.log10(0.5)
    )


def test_take_off_roll() -> None:
    """Volume 2's example of a take-off roll: 1600 m to 75 m/s in 8 segments, 25 to
    375 m long (figure 3-5)."""
    profile = [
        doc29.ProfilePoint(0.0, 0.0, 0.0, 20000.0),
        doc29.ProfilePoint(1600.0, 0.0, 75.0, 18000.0),
        doc29.ProfilePoint(5000.0, 1000.0, 90.0, 15000.0),
    ]
    track = doc29.Track(points=((0.0, 0.0), (20000.0, 0.0)))
    segments = doc29.flight_path(profile, track, departure=True)
    roll = [s for s in segments if s.roll == "take-off"]
    assert [round(s.end[0], 6) for s in roll] == [25, 100, 225, 400, 625, 900, 1225, 1600]
    assert [s.powers[1] for s in roll] == pytest.approx(np.linspace(19750, 18000, 8))
    assert all(s.start[2] == path.LOWEST_SOURCE_M for s in roll)


def test_climb_heights() -> None:
    """The initial climb is cut at Volume 2's heights scaled to its end, 304.8 m, as in
    its example: in the heights in feet that Volume 3's reference cases use (Volume 2's
    example rounds its heights in metres: 17.2 and 37.8 m)."""
    a = path.Node(s=0.0, height=0.0, speed=80.0, power=20000.0, ground=True)
    b = path.Node(s=3000.0, height=304.8, speed=81.0, power=20000.0)
    cuts = path._by_height(a, b, climbing=True)
    heights = [n.height for n in cuts]
    assert len(heights) == 6
    assert heights[:2] == pytest.approx([17.195, 37.719], abs=0.001)


def test_threshold() -> None:
    """Volume 3 lands its aircraft with the threshold at their approach's lowest node:
    the jets' point at 15.2 m, 290.2 m before touchdown, the propeller aircraft's cut
    at 17.2 m, 328.1 m before it."""
    profiles = volume3.profiles()
    assert path.lowest_approach_node(profiles[("JETF", "A")]) == pytest.approx(-290.2)
    assert path.lowest_approach_node(profiles[("PROP", "A")]) == pytest.approx(-328.1, abs=0.05)


def test_tables_read_anp_units(tmp_path) -> None:
    """A table with semicolons and ANP's feet and knots reads as Volume 3's metres."""
    table = tmp_path / "profiles.csv"
    table.write_text(
        "ACFT_ID;Op Mode;Profile ID;Stage Length;Point Number;Distance (ft);"
        "Height (ft);Speed (kt);Corrected Net Thrust (lb)\n"
        "X;D;STANDARD;1;2;1000;500;160;20000\n"
        "X;D;STANDARD;1;1;0;0;0;25000\n"
    )
    points = doc29.read_fixed_point_profiles(table)[("X", "D", "STANDARD", "1")]
    assert [p.power for p in points] == [25000.0, 20000.0]
    assert points[1].distance_m == pytest.approx(304.8)
    assert points[1].height_m == pytest.approx(152.4)
    assert points[1].speed_m_s == pytest.approx(160 * KNOT_M_S)


def test_maximum_level_under_the_path() -> None:
    """Under a long level flight at the NPD curves' speed, the maximum level is the
    curve's at the height and the exposure the SEL curve's (the corrections nought)."""
    plane = volume3.aircraft()["PROP"]
    sel = volume3.npd()[("PROP", "SEL", "D")]
    lamax = volume3.npd()[("PROP", "LAmax", "D")]
    height = 1000 * FOOT_M
    speed = ev.REFERENCE_SPEED_M_S
    segment = doc29.Segment(
        start=(-1e6, 0.0, height),
        end=(1e6, 0.0, height),
        speeds=(speed, speed),
        powers=(100.0, 100.0),
    )
    result = doc29.event(plane, sel, lamax, [segment], np.array([[0.0, 0.0, 0.0]]))
    assert float(result.sel[0]) == pytest.approx(92.9, abs=1e-6)
    assert float(result.lamax[0]) == pytest.approx(86.1, abs=1e-6)
