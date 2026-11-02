"""Aircraft noise by ECAC Doc 29 (acoustic_spaces.doc29) against Volume 3's reference
cases and Volume 2's equations."""

from __future__ import annotations

import math
import warnings

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


def _level_flight(power: float, height: float = 1000 * FOOT_M) -> doc29.Segment:
    """A long level flight at the NPD curves' speed, ``height`` (m) over the origin."""
    speed = ev.REFERENCE_SPEED_M_S
    return doc29.Segment(
        start=(-1e6, 0.0, height),
        end=(1e6, 0.0, height),
        speeds=(speed, speed),
        powers=(power,) * 2,
    )


def test_power_far_outside_the_npd_curves_warns() -> None:
    """Volume 2 section 4.2 extrapolates the NPD curves in power by eq. 4-3 and cautions
    against extrapolation more than 5 dB from the NPD envelope: an event whose NPD
    level at a segment's power departs more than 5 dB from the level at the nearest of
    the curves' powers warns, once however many segments do; 4.9 dB does not."""
    plane = volume3.aircraft()["JETW"]
    sel, lamax = (volume3.npd()[("JETW", d, "D")] for d in ("SEL", "LAmax"))
    receiver = np.array([[0.0, 0.0, 0.0]])
    d = 1000 * FOOT_M
    # The LAmax curves' slope above 22500 lb at 1000 ft (dB/lb), steeper than the SEL's.
    slope = float(lamax.level(22500.0, d) - lamax.level(20000.0, d)) / 2500.0
    assert slope > float(sel.level(22500.0, d) - sel.level(20000.0, d)) / 2500.0 > 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", doc29.NpdExtrapolationWarning)
        for power in (22500.0, 22500.0 + 4.9 / slope, 10000.0):
            doc29.event(plane, sel, lamax, [_level_flight(power)], receiver)
    with pytest.warns(doc29.NpdExtrapolationWarning) as caught:
        doc29.event(plane, sel, lamax, [_level_flight(22500.0 + 5.1 / slope)], receiver)
    assert "LAmax NPD curves' powers, 10000 to 22500" in str(caught[0].message)
    # Three times the top power, 1000 ft and 2000 ft up: one warning, of its SEL curves,
    # whose levels there extrapolate from the top two curves (eq. 4-3).
    beyond = 3 * 22500.0
    with pytest.warns(doc29.NpdExtrapolationWarning) as caught:
        result = doc29.event(
            plane, sel, lamax, [_level_flight(beyond), _level_flight(beyond, 2 * d)], receiver
        )
    assert len(caught) == 1
    tops = [sel.levels[-1][NPD_FEET.index(feet)] for feet in (1000.0, 2000.0)]
    belows = [sel.levels[-2][NPD_FEET.index(feet)] for feet in (1000.0, 2000.0)]
    over = [
        (top - below) / 2500.0 * (beyond - 22500.0) for top, below in zip(tops, belows, strict=True)
    ]
    assert [part.baseline[0] for part in result.segments] == pytest.approx(
        [top + up for top, up in zip(tops, over, strict=True)], abs=1e-9
    )
    words = str(caught[0].message)
    assert "its flight's power, 67500, lies so far outside its SEL NPD curves' powers" in words
    assert f"depart up to {max(over):.1f} dB" in words and "Corrected_Net_Thrust_(lb)" in words
    # Below the curves as well: 1000 lb against the lowest, 10000 lb.
    with pytest.warns(doc29.NpdExtrapolationWarning, match="power, 1000, lies"):
        doc29.event(plane, sel, lamax, [_level_flight(1000.0)], receiver)


def test_tables_read_units_in_brackets(tmp_path) -> None:
    """A fixed-point profile's column gives its unit in round or square brackets, in
    any case: feet, knots, kts, m and m/s read as Appendix G's ft and kt (G3.8) and
    Volume 3's metres; a column naming another unit, or none, is an error."""
    table = tmp_path / "profiles.csv"
    header = "ACFT_ID;Op Mode;Profile ID;Stage Length;Point Number;{};{};{};Power\n"
    rows = "X;D;P;1;1;0;0;0;25000\nX;D;P;1;2;1000;500;160;20000\n"
    for names, length, speed in [
        (("Distance [feet]", "Height (FT)", "Speed (knots)"), FOOT_M, KNOT_M_S),
        (("Distance (m)", "Height [ m ]", "Speed (kts)"), 1.0, KNOT_M_S),
        (("Distance (ft)", "Height AFE (ft)", "TAS (M/S)"), FOOT_M, 1.0),
    ]:
        table.write_text(header.format(*names) + rows)
        (_, point) = doc29.read_fixed_point_profiles(table)[("X", "D", "P", "1")]
        assert point.distance_m == pytest.approx(1000.0 * length, rel=1e-12)
        assert point.height_m == pytest.approx(500.0 * length, rel=1e-12)
        assert point.speed_m_s == pytest.approx(160.0 * speed, rel=1e-12)
    for names, wrong in [
        (("Distance (nmi)", "Height (ft)", "Speed (kt)"), r"'Distance \(nmi\)' names none"),
        (("Distance (ft)", "Height", "Speed (kt)"), "'Height' names none"),
        (
            ("Distance (ft)", "Height (ft)", "Speed (km/h)"),
            r"units of a speed in brackets, \(m/s\)",
        ),
        (("Distance (ft) (m)", "Height (ft)", "Speed (kt)"), "names more than one"),
    ]:
        table.write_text(header.format(*names) + rows)
        with pytest.raises(ValueError, match=wrong):
            doc29.read_fixed_point_profiles(table)


def test_lateral_directivity_by_its_start(tmp_path) -> None:
    """An aircraft's lateral directivity names its engine installation by its start
    (Volume 2 Appendix G2: "Fuselage-mounted", "Wing-mounted" or "Prop"), whose
    correction is eq. 4-15's; any other is an error naming the aircraft."""
    table = tmp_path / "aircraft.csv"
    header = (
        "ACFT_ID,Description,Engine Type,Number of Engines,NPD_ID,Power Parameter,"
        "Approach Spectral Class ID,Departure Spectral Class ID,Lateral Directivity Identifier\n"
    )
    rows = [
        "A,a,Jet,2,A,CNT (lb),1,2,Wing-mounted",
        "B,b,Jet,3,B,CNT (lb),1,2,FUSELAGE mounted",
        "C,c,Turboprop,2,C,CNT (lb),1,2,Prop",
        "D,d,Jet,2,D,CNT (lb),1,2,Wing",
    ]
    table.write_text(header + "\n".join(rows) + "\n")
    planes = doc29.read_aircraft(table)
    assert [p.installation for p in planes.values()] == ["Wing", "Fuselage", "Prop", "Wing"]
    phi = np.array([0.0, 20.0, 45.0])
    assert ev.installation_db("Wing-mounted", phi) == pytest.approx(ev.installation_db("Wing", phi))
    assert float(ev.installation_db("Wing-mounted", np.array(0.0))) == pytest.approx(
        10 * 0.062 * math.log10(0.0039)
    )
    table.write_text(header + rows[0] + "\nE,e,Jet,2,E,CNT (lb),1,2,Pod\n")
    with pytest.raises(
        ValueError, match="aircraft.csv, aircraft 'E': the lateral directivity 'Pod'"
    ):
        doc29.read_aircraft(table)
    with pytest.raises(ValueError, match="none of Fuselage, Wing, Prop"):
        ev.installation_db("Tail", phi)


def test_a_track_whose_first_points_coincide() -> None:
    """A walk's places one above the other give one point of the ground track: a track
    through points of which two in a row coincide leaves the later out, and a track
    with a leg of no length still finds its points along it and extends itself
    the way its first and last legs with a length go (section 3.6.6)."""
    track = doc29.Track.through([(0.0, 0.0), (0.0, 0.0), (0.0, 1e-7), (300.0, 400.0)], (0.0, 0.0))
    assert track.points == ((0.0, 0.0), (300.0, 400.0)) and track.start_m == 0.0
    with pytest.raises(ValueError, match="two points more than"):
        doc29.Track.through([(5.0, 5.0), (5.0, 5.0)], (5.0, 5.0))
    flat = doc29.Track(points=((0.0, 0.0), (0.0, 0.0), (300.0, 400.0), (300.0, 400.0)))
    assert flat.at(0.0) == (0.0, 0.0)
    assert flat.at(-50.0) == pytest.approx((-30.0, -40.0))
    assert flat.at(250.0) == pytest.approx((150.0, 200.0))
    assert flat.at(600.0) == pytest.approx((360.0, 480.0))
    longer = flat.extended(-50.0, 600.0)
    assert longer.points[0] == pytest.approx((-30.0, -40.0))
    assert longer.points[-1] == pytest.approx((360.0, 480.0))
    assert longer.at(0.0) == pytest.approx((0.0, 0.0))
