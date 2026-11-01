"""Flight performance by ECAC Doc 29 Volume 2 Appendix B (acoustic_spaces.doc29.performance)
against Volume 3 Part 2's reference profiles and the standard atmosphere."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from acoustic_spaces import doc29
from acoustic_spaces.doc29 import performance as pf
from acoustic_spaces.doc29 import volume3

CASES = volume3.profile_cases()


def test_cases() -> None:
    """Volume 3 Part 2 has nine arrivals and seventeen departures, 314 profile points."""
    assert sum(c.operation == "A" for c in CASES) == 9
    assert sum(c.operation == "D" for c in CASES) == 17
    assert sum(len(points) for points in volume3.reference_profiles().values()) == 314


@pytest.mark.parametrize("case", CASES, ids=[c.number for c in CASES])
def test_reference_profiles(case: volume3.ProfileCase) -> None:
    """Each case's fixed-point profile: as many points as Volume 3's, each height, true
    airspeed and corrected net thrust to its rounding (0.1 ft, kt and lb), each distance
    along the track within 0.15 ft (Volume 3's own distances for the same steps differ
    by 0.2 ft between cases: ICAO_A's acceleration from 3787.5 ft is 4678.3 ft long in
    case 6, 4678.5 ft in case 74)."""
    mine = case.points()
    reference = volume3.reference_profiles()[case.number]
    assert len(mine) == len(reference)
    for point, ref in zip(mine, reference, strict=True):
        assert point.distance_ft == pytest.approx(ref.distance_ft, abs=0.15)
        assert point.height_ft == pytest.approx(ref.height_ft, abs=0.051)
        assert point.speed_kt == pytest.approx(ref.speed_kt, abs=0.051)
        assert point.thrust_lb == pytest.approx(ref.thrust_lb, abs=0.051)


def test_standard_atmosphere() -> None:
    """Appendix B3's atmosphere against the standard atmosphere's tables (ISO 2533): at
    10,000 ft the pressure ratio 0.6877, the density ratio 0.7385 and -4.8 °C; a 1 inHg
    higher sea-level pressure lowers the pressure altitude about 1000 ft (900 ft)."""
    air = pf.STANDARD
    assert air.delta(0.0) == pytest.approx(1.0)
    assert air.sigma(0.0) == pytest.approx(1.0)
    assert air.delta(10000.0) == pytest.approx(0.6877, abs=2e-4)
    assert air.sigma(10000.0) == pytest.approx(0.7385, abs=2e-4)
    assert air.temperature(10000.0) == pytest.approx(-4.81, abs=0.01)
    assert air.pressure_altitude(10000.0) == pytest.approx(10000.0)
    high = pf.Aerodrome(pressure_inhg=30.92)
    assert high.pressure_altitude(0.0) == pytest.approx(-900.0, abs=20.0)
    hot = pf.Aerodrome(elevation_ft=5000.0, temperature_c=40.0)
    assert hot.temperature(5000.0) == pytest.approx(40.0)
    assert hot.sigma(5000.0) < pf.STANDARD.sigma(5000.0)


def _ground(tas_kt: float, headwind_kt: float, gamma: float) -> float:
    """The speed along a descent's path over the ground (ft/s), Appendix B7.1.2."""
    v, w = pf.KNOT_FT_S * tas_kt, pf.KNOT_FT_S * headwind_kt
    return math.sqrt(v**2 - (w * math.sin(gamma)) ** 2) - w * math.cos(gamma)


def test_idle_steps_keep_their_deceleration() -> None:
    """Away from the standard atmosphere at sea level, Appendix B7.1.2 and B7.1.4 hold
    the idle steps' deceleration: a Level-Idle step's length and a Descend-Idle step's
    start (its height before another descent, its speed before a level step) change so
    that the aircraft slows as it does at sea level in the standard atmosphere. Volume
    3's arrivals are all at sea level in it."""
    plane = volume3.performance()["JETW"]
    steps = volume3.approach_procedures()[("JETW", "Level_Idle")]
    air = pf.Aerodrome(elevation_ft=5000.0, temperature_c=35.0, headwind_kt=8.0)
    points = pf.arrival(plane, steps, aerodrome=air)
    standard = pf.arrival(plane, steps)
    k, w = pf.KNOT_FT_S, air.headwind_kt

    def level(p: list[pf.Point], i: int, wind: float) -> float:
        a, b = p[i], p[i + 1]
        return (
            k**2
            * ((b.speed_kt - wind) ** 2 - (a.speed_kt - wind) ** 2)
            / (2.0 * (b.distance_ft - a.distance_ft))
        )

    def descent(p: list[pf.Point], i: int, j: int, wind: float) -> float:
        gamma = math.radians(3.0)
        a, b = p[i], p[j]
        path = (a.height_ft - b.height_ft) / math.sin(gamma)
        return (_ground(b.speed_kt, wind, gamma) ** 2 - _ground(a.speed_kt, wind, gamma) ** 2) / (
            2.0 * path
        )

    # The points: the descent (0, 1), the two Level-Idle steps (2, 3), the three
    # Descend-Idle steps (4, 5, 6 and its end 7), the final descent (8, 9, 10).
    for i in (2, 3):
        assert level(points, i, w) == pytest.approx(level(standard, i, w), rel=1e-9)
    for i, j in ((4, 5), (5, 6), (6, 8)):
        assert descent(points, i, j, w) == pytest.approx(descent(standard, i, j, w), rel=1e-9)
    assert points[5].height_ft != pytest.approx(standard[5].height_ft, abs=1.0)
    assert points[3].distance_ft - points[2].distance_ft != pytest.approx(16500.0, abs=10.0)

    # A Descend-Idle step before a level step keeps its height and starts faster or
    # slower instead.
    custom = list(steps[:4]) + [
        pf.ApproachStep("Level", "25", altitude_ft=2460.6, speed_kt=167.4, distance_ft=5000.0),
        pf.ApproachStep("Descend", "30", altitude_ft=2460.6, speed_kt=135.0, angle_deg=3.0),
        *steps[7:],
    ]
    points = pf.arrival(plane, custom, aerodrome=air)
    standard = pf.arrival(plane, custom)
    # Its points: the step's start (4) and end (5), the level step's start (6).
    assert points[4].height_ft == pytest.approx(3000.0)
    assert points[4].speed_kt != pytest.approx(178.2 / math.sqrt(air.sigma(8000.0)), abs=1.0)
    assert descent(points, 4, 6, w) == pytest.approx(descent(standard, 4, 6, w), rel=1e-9)


def test_no_change_of_rating_no_transition() -> None:
    """A climb at the take-off's thrust rating follows the take-off without a
    transition; a change of rating adds a point 1000 ft on (Appendix B6.1.6), with half
    the step when it is shorter than 2000 ft, and a level step always has one."""
    plane = volume3.performance()["JETW"]
    steps = volume3.departure_procedures()[("JETW", "ICAO_A")]
    points = pf.departure(plane, steps[:3], plane.weights[("D", "1")])
    assert len(points) == 5
    assert points[3].distance_ft - points[2].distance_ft == pytest.approx(1000.0)
    short = [*steps[:2], dataclasses.replace(steps[2], altitude_ft=1550.0)]
    points = pf.departure(plane, short, plane.weights[("D", "1")])
    assert points[3].distance_ft - points[2].distance_ft == pytest.approx(
        (points[4].distance_ft - points[2].distance_ft) / 2.0
    )


def test_tables_read_anp_layouts(tmp_path) -> None:
    """Tables with semicolons and other names for the columns read as Volume 3's."""
    (tmp_path / "aircraft.csv").write_text(
        "ACFT_ID;Description;Engine Type;Number of Engines;Maximum Gross Landing Weight (lb);"
        "Maximum Sea Level Static Thrust (lb)\nX;test;Jet;2;100000;20000\n"
    )
    (tmp_path / "jet.csv").write_text(
        "ACFT_ID;Thrust_Rating;E;F;Ga;Gb;H\nX;MaxTake-off;20000;-20;0.3;0;0\n"
        "X;MaxTake-offHiTemp;19000;-20;0.3;0;-10\n"
    )
    (tmp_path / "aero.csv").write_text(
        "ACFT_ID;Op_Type;Flap_ID;B;C;R\nX;D;5;0.008;0.4;0.07\nX;D;0;;;0.05\n"
    )
    plane = pf.read_performance(
        tmp_path / "aircraft.csv", jet=tmp_path / "jet.csv", aerodynamic=tmp_path / "aero.csv"
    )["X"]
    assert plane.engines == 2
    assert set(plane.jet) == {"maxtakeoff", "maxtakeoffhitemp"}
    assert plane.flap("D", "5").b == pytest.approx(0.008)
    assert plane.flap("D", "0").speed is None
    engines = pf._Engines(plane, pf.Aerodrome(temperature_c=40.0), 80000.0, departure=True)
    # Above the breakpoint the high-temperature coefficients rule.
    t = pf.Aerodrome(temperature_c=40.0).temperature(0.0)
    assert engines.rated("MaxTakeoff", 0.0, 1.0, 0.0) == pytest.approx(19000.0 - 10.0 * t)


def test_profile_flies() -> None:
    """A profile from procedural steps, in metres, makes a flight path and its noise:
    the level falls away from the track."""
    plane = volume3.performance()["JETW"]
    steps = volume3.departure_procedures()[("JETW", "ICAO_A")]
    profile = pf.as_profile(pf.departure(plane, steps, plane.weights[("D", "1")]))
    assert profile[1].distance_m == pytest.approx(4897.5 * doc29.tables.FOOT_M, abs=0.05)
    track = doc29.Track(points=((0.0, 0.0), (60000.0, 0.0)))
    segments = doc29.flight_path(profile, track, departure=True)
    noise = volume3.aircraft()["JETW"]
    curves = volume3.npd()
    receivers = np.array([[6000.0, 0.0, 1.2], [6000.0, 1000.0, 1.2], [6000.0, 3000.0, 1.2]])
    result = doc29.event(
        noise, curves[("JETW", "SEL", "D")], curves[("JETW", "LAmax", "D")], segments, receivers
    )
    assert np.all(np.isfinite(result.sel))
    assert result.sel[0] > result.sel[1] > result.sel[2]


ANP_NAMES = {
    "performance_aircraft.csv": "Aircraft.csv",
    "npd.csv": "NPD_data.csv",
    "spectral_classes.csv": "Spectral_Classes.csv",
    "fixed_point_profiles.csv": "Default_fixed_point_profiles.csv",
    "jet_coefficients.csv": "Jet_engine_coefficients.csv",
    "propeller_coefficients.csv": "Propeller_engine_coefficients.csv",
    "aerodynamic_coefficients.csv": "Aerodynamic_coefficients.csv",
    "default_weights.csv": "Default_weights.csv",
    "departure_steps.csv": "Default_departure_procedural_steps.csv",
    "approach_steps.csv": "Default_approach_procedural_steps.csv",
}


def test_anp_database(tmp_path) -> None:
    """A folder of tables named as the ANP database's, separated by semicolons (here
    Volume 3's, which rely on no ANP data), reads as one database: its procedures fly
    Volume 3's profiles, and an aircraft without procedures flies its fixed-point
    profile."""
    folder = tmp_path / "download" / "ANP"
    folder.mkdir(parents=True)
    for name, anp in ANP_NAMES.items():
        text = (volume3.DATA / name).read_text(encoding="utf-8")
        (folder / anp).write_text(text.replace(",", ";"), encoding="utf-8")
    (folder / "Substitution_table.csv").write_text("ACFT_ID;Substitute\nA;B\n")
    database = doc29.anp.read_database(tmp_path)
    assert set(database.aircraft) == {"JETW", "PROP"}
    sel, lamax = database.curves("JETW", "D")
    assert sel == volume3.npd()[("JETW", "SEL", "D")]
    reference = volume3.reference_profiles()
    for number, operation, procedure, air in [
        ("6", "D", "ICAO_A", pf.STANDARD),
        ("2C", "A", "Level_Idle", pf.Aerodrome(headwind_kt=0.0)),
    ]:
        flown = database.profile("JETW", operation, procedure, aerodrome=air)
        expected = pf.as_profile(reference[number])
        assert [p.distance_m for p in flown] == pytest.approx(
            [p.distance_m for p in expected], abs=0.05
        )
    fixed = database.profile("PROP", "A", "FPP")
    assert fixed == volume3.profiles()[("PROP", "A")]
    with pytest.raises(ValueError, match="two files"):
        (folder / "aircraft_copy.csv").write_text("x")
        doc29.anp.find_tables(tmp_path)
