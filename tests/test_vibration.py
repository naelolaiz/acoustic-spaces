"""Ground-borne vibration by the FTA's general method (:mod:`acoustic_spaces.vibration`),
held against the manual's own tables and worked example, and the railway's and the
street's scripts that use it."""

import argparse
import math
from pathlib import Path

import pytest

from acoustic_spaces import vibration as v
from acoustic_spaces.demos import listeners_in, plant, rail, scripts_of, sibling
from acoustic_spaces.examples import example_path

MPH = v.MILE_KM


def test_the_base_curves_give_the_manuals_example() -> None:
    """Example 6-1 of the manual: light rail at 40 mph, houses of wood frame 125 ft away
    on soil that propagates efficiently. The rapid transit curve gives 65 VdB there, as
    the example reads it; from that reading its sums come out as the example's: 65 -
    1.9 = 63.1 for the speed, + 10 - 5 = 68.1 for the soil and the coupling, + 6 - 2 =
    72.1 on the first floor."""
    d = 125.0 * v.FOOT_M
    base = v.CURVES["rapid_transit"].level(d)
    assert round(base) == 65 and base == pytest.approx(65.43, abs=0.01)
    speed = v.speed_db(40 * MPH, v.CURVES["rapid_transit"].reference_kmh)
    assert round(65 + speed, 1) == 63.1
    efficient = v.Line(ground="efficient")
    assert efficient.path_db(d) == 10.0 and v.COUPLINGS["wood_frame"] == -5.0
    assert round(65 + speed + 10 - 5, 1) == 68.1
    indoors = v.indoor_db("wood_frame", 1)
    assert indoors == -1.0  # -5 coupling, -2 to the first floor, +6 its resonances
    assert round(65 + speed + 10 + indoors, 1) == 72.1
    # From the curve itself, unrounded, 0.4 dB more.
    whole = v.ground_level("rapid_transit", d, 40 * MPH, efficient) + indoors
    assert whole == pytest.approx(72.5, abs=0.01)


def test_the_curves_are_the_manuals_and_fall_with_distance() -> None:
    """The three curves at the manual's speeds, 10 to 300 ft out, each falling all the
    way, as figure 6-4 draws them; rapid transit 73 VdB 50 ft out, as the manual's text
    has light rail "70 VdB or more near their tracks", and a bus under 70 VdB from 25
    ft, as it has buses and trucks rarely over 70. Its text puts heavy locomotives 5 to
    10 dB over rail transit; its curves put them 10.9 to 12.3 dB over between 25 and
    100 ft."""
    feet = [10, 25, 50, 100, 200, 300]
    found = {k: [c.level(ft * v.FOOT_M) for ft in feet] for k, c in v.CURVES.items()}
    assert [round(x, 1) for x in found["locomotive"]] == [94.6, 89.8, 84.6, 78.4, 71.4, 67.1]
    assert [round(x, 1) for x in found["rapid_transit"]] == [81.6, 77.5, 73.1, 67.5, 60.6, 55.8]
    assert [round(x, 1) for x in found["rubber_tyred"]] == [75.5, 69.6, 63.5, 56.8, 50.6, 47.5]
    for levels in found.values():
        assert all(a > b for a, b in zip(levels, levels[1:], strict=False))
    # Only as near and as far as figure 6-4 draws them: 10 to 300 ft.
    assert (v.NEAREST_M, v.FARTHEST_M) == pytest.approx((3.048, 91.44))
    for outside in (3.0, 92.0):
        with pytest.raises(ValueError, match="10 to 300 ft"):
            v.CURVES["locomotive"].level(outside)
    gaps = [
        a - b for a, b in zip(found["locomotive"][1:4], found["rapid_transit"][1:4], strict=True)
    ]
    assert all(10.9 < gap < 12.4 for gap in gaps)
    assert v.CURVES["rubber_tyred"].reference_kmh == pytest.approx(30 * MPH)


def test_speed_follows_twenty_log_speed_as_table_6_11_prints_it() -> None:
    """Table 6-11's rows of speed against 50 mph and 30 mph, to its 0.1 dB."""
    rows = {60: (1.6, 6.0), 50: (0.0, 4.4), 40: (-1.9, 2.5), 30: (-4.4, 0.0), 20: (-8.0, -3.5)}
    for mph, (rail_db, bus_db) in rows.items():
        assert round(v.speed_db(mph * MPH, 50 * MPH), 1) == rail_db
        assert round(v.speed_db(mph * MPH, 30 * MPH), 1) == bus_db


def test_the_adjustments_add_up_as_the_manual_says() -> None:
    """Vehicles and track add, but only the greater counts when the track is worn or
    corrugated; special trackwork 10 dB within 100 ft and 5 dB to 200 ft, the greater of
    it and the track's condition; a treatment and the structure add to them; a rock layer
    gains with distance, between table 6-12's rows by interpolation."""
    assert v.Line(vehicles="stiff_suspension", track="jointed").source_db() == 13.0
    assert v.Line(vehicles="stiff_suspension", track="worn").source_db() == 10.0
    assert v.Line(vehicles="worn_wheels", track="worn").source_db() == 10.0
    assert v.Line(special_trackwork_m=20.0).source_db() == 10.0
    assert v.Line(special_trackwork_m=45.0).source_db() == 5.0
    assert v.Line(special_trackwork_m=70.0).source_db() == 0.0
    assert v.Line(track="worn", special_trackwork_m=45.0).source_db() == 10.0
    assert v.Line(track="jointed", treatment="floating_slab").source_db() == -10.0
    assert v.Line(structure="elevated").path_db(30.0) == -10.0
    assert v.Line(structure="subway_rock", ground="rock").path_db(100 * v.FOOT_M) == -11.0
    rock = v.Line(ground="rock")
    assert rock.path_db(125 * v.FOOT_M) == pytest.approx(5.0)
    assert rock.path_db(5.0) == 2.0 and rock.path_db(500.0) == 9.0
    with pytest.raises(ValueError, match="unknown track"):
        v.Line(track="smooth")


def test_buildings_and_floors() -> None:
    """Table 6-13: 2 dB a floor to the fifth, 1 dB a floor to the tenth, the floors'
    resonances 6 dB; where the structure is not known, the manual's conservative
    outdoor-to-indoor changes."""
    assert [v.floors_db(n) for n in (1, 2, 5, 6, 10)] == [-2.0, -4.0, -10.0, -11.0, -15.0]
    for floor in (0, 11):
        with pytest.raises(ValueError, match="floors 1 to 10"):
            v.floors_db(floor)
    assert v.indoor_db("large_masonry_footings", 3) == -13.0 - 6.0 + 6.0
    assert v.indoor_db("on_rock", 1) == 4.0
    assert [v.indoor_db("light_wood_frame", n) for n in (1, 2, 3)] == [3.0, 6.0, 6.0]
    with pytest.raises(ValueError, match="floors 1 to 3"):
        v.indoor_db("light_wood_frame", 4)
    assert v.indoor_db("small_masonry", 4) == 3.0 and v.indoor_db("large", 7) == 0.0
    assert set(v.BUILDINGS) == set(v.COUPLINGS) | set(v.GENERIC)


def test_the_criteria_and_the_events() -> None:
    """Table 6-2's events a day and table 6-3's and 6-4's limits."""
    assert [v.events(n) for n in (71, 70, 30, 29)] == [
        "frequent",
        "occasional",
        "occasional",
        "infrequent",
    ]
    assert v.criterion(2, "frequent") == v.Criterion(72.0, 35.0)
    assert v.criterion(2, "infrequent") == v.Criterion(80.0, 43.0)
    assert v.criterion(3, "occasional") == v.Criterion(78.0, 43.0)
    assert v.criterion(1, "infrequent") == v.Criterion(65.0, None)
    assert v.criterion("theatre", "occasional") == v.Criterion(80.0, 43.0)
    assert v.criterion("concert_hall", "frequent") == v.Criterion(65.0, 25.0)
    assert v.NOISE_DB == {"low": -50.0, "mid": -35.0, "high": -20.0}
    assert v.noise_level(80.0, "mid") == 45.0


def test_construction_plant_by_table_7_4() -> None:
    """Each row's level is its peak particle velocity 12 dB down (a crest factor of 4)
    within the table's rounding, but for a hydromill in rock, which the table prints at
    75 VdB where its 0.017 in/s gives 72.6; the damage criteria likewise. Eq. 7-2 and
    7-3 fall alike: 30 dB a decade."""
    for key, row in v.PLANT.items():
        gap = row.level_vdb - v.level_of(row.ppv_in_s)
        if key == "hydromill_rock":
            assert gap == pytest.approx(2.4, abs=0.05)
        else:
            assert abs(gap) <= 0.5, key
    for row in v.DAMAGE.values():
        assert abs(row.level_vdb - v.level_of(row.ppv_in_s)) <= 0.5
    d = 10.0
    assert v.plant_ppv("hoe_ram", v.PLANT_REFERENCE_M) == pytest.approx(0.089)
    assert v.plant_level("hoe_ram", 10 * v.PLANT_REFERENCE_M) == pytest.approx(57.0)
    for key in v.PLANT:
        drop = v.level_of(v.plant_ppv(key, d)) - v.level_of(v.PLANT[key].ppv_in_s)
        assert drop == pytest.approx(v.plant_level(key, d) - v.PLANT[key].level_vdb)
    assert v.mm_s(0.2) == pytest.approx(5.08)
    assert v.VELOCITY_REFERENCE_M_S == pytest.approx(25.4e-9)


def test_distances_in_plan() -> None:
    """The distance in plan to a line and to a footprint's outline, and where a falling
    level crosses a limit."""
    track = [(-150.0, 0.0, 0.2), (150.0, 0.0, 0.2)]
    assert v.horizontal_distance((10.0, 7.5, 1.6), track) == pytest.approx(7.5)
    assert v.horizontal_distance((160.0, 0.0, 0.0), track) == pytest.approx(10.0)
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert v.horizontal_distance((5.0, -4.0), square, closed=True) == pytest.approx(4.0)
    assert v.horizontal_distance((-3.0, 5.0), square, closed=True) == pytest.approx(3.0)
    curve = v.CURVES["rapid_transit"]
    near, far = v.NEAREST_M, v.FARTHEST_M
    d = v.distance_for(curve.level, 65.0, near, far)
    assert d is not None and curve.level(d) == pytest.approx(65.0, abs=1e-6)
    assert v.distance_for(curve.level, 100.0, near, far) == near
    assert v.distance_for(curve.level, 40.0, near, far) is None


def test_the_trains_and_the_works_have_their_curves_and_rows() -> None:
    """Multiple units and trams on the rapid transit curve, the locomotive-hauled trains
    on the locomotive one (the manual's table 6-9); the breaker a hoe ram and the dumper
    loaded trucks, the generator none."""
    curves = {name: train.curve for name, train in rail.TRAINS.items()}
    assert curves == {
        "s_bahn": "rapid_transit",
        "regional": "locomotive",
        "intercity": "locomotive",
        "freight": "locomotive",
        "tram": "rapid_transit",
    }
    rows = {name: machine.vibration for name, machine in plant.MACHINES.items()}
    assert rows == {"breaker": "hoe_ram", "generator": None, "dumper": "loaded_trucks"}


def _script(example: str):
    return sibling(Path(example_path(example)).parent / "vibration.py", "vibration")


def test_the_railways_script(capsys: pytest.CaptureFixture[str]) -> None:
    """The railway's script: each train at its own speed at the scene's listeners, the
    house small masonry (3 dB more indoors), 24 trains a day (infrequent events, 80 VdB
    and 43 dB(A)), the freight train's wagons against frequent events 3 dB under."""
    (demo,) = scripts_of("railway")
    assert demo.name == "vibration"
    # The editor ticks the houses among the scene's listeners in use.
    assert demo.option("listeners").kind == "listeners"
    scene = example_path("railway")
    args = demo.resolved(argparse.Namespace(scene=scene, listeners=None))
    assert demo.listeners(args) == listeners_in(scene) != []
    assert _script("railway").main([]) == 0
    out = capsys.readouterr().out
    assert "residences, 24 trains a day: infrequent events" in out
    regional = v.ground_level("locomotive", 25.0, 160.0)
    assert regional == pytest.approx(86.3, abs=0.05)
    assert f"meadow_25m                25.0 m{regional:9.1f}{regional + 3:9.1f}" in out
    assert "its wagons: limits 72 VdB and 35 dB(A) indoors" in out
    wagons = v.ground_level("locomotive", 7.5, 100.0) - 3.0
    assert f"tsi_7m5                    7.5 m{wagons:9.1f}" in out
    tram = v.ground_level("rapid_transit", 25.0, 50.0)
    assert f"{tram:9.1f}{tram + 3:9.1f}{tram + 3 - 50:9.1f}\n" in out
    # --listeners: those houses alone, in that order.
    assert _script("railway").main(["--listeners", "meadow_25m,tsi_7m5"]) == 0
    out = capsys.readouterr().out
    table = out.split("Indoors")[0]
    assert table.index("meadow_25m") < table.index("tsi_7m5")
    assert "behind_barrier_25m" not in out
    with pytest.raises(SystemExit, match="--listeners: garden; the scene's listeners"):
        _script("railway").main(["--listeners", "garden"])


def test_the_streets_script(capsys: pytest.CaptureFixture[str]) -> None:
    """The street's script: the tram 6 m from the houses over the limit for frequent
    events; the breaker 4 m from them at 5.94 mm/s, over the damage criteria of the
    two most fragile kinds of building, and at the offices 8 m away under all."""
    assert [d.name for d in scripts_of("street")] == ["vibration"]
    assert _script("street").main([]) == 0
    out = capsys.readouterr().out
    tram = v.ground_level("rapid_transit", 6.0, 50.0)
    assert f"the ground outside {tram:.1f} VdB; on floor 1 (+3 dB) {tram + 3:.1f} VdB" in out
    assert f"the vibration is {tram + 3 - 72:.1f} dB over the limit" in out
    ppv = v.plant_ppv("hoe_ram", 4.0)
    assert v.mm_s(ppv) == pytest.approx(5.94, abs=0.005)
    assert (
        f"at the houses, 4.0 m: PPV {v.mm_s(ppv):.2f} mm/s ({ppv:.3f} in/s), "
        f"{v.plant_level('hoe_ram', 4.0):.1f} VdB; over the damage criteria of categories "
        "III and IV" in out
    )
    assert "generator: left out" in out
    assert math.isclose(v.plant_ppv("loaded_trucks", 4.0), 0.1998, abs_tol=5e-5)
