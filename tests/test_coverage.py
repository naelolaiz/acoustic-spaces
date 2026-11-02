import json
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import CoverageResult, coverage_map, map_grid
from acoustic_spaces.engine.coverage import (
    ROOM_MAP_PARAMETERS,
    TIME_STEP,
    _in_room,
    _point_values,
    audience_surfaces,
)
from acoustic_spaces.engine.insertion import bare_room
from acoustic_spaces.engine.late import stochastic_map_energy
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.parameters import speech_transmission_index
from acoustic_spaces.scene import Coverage
from acoustic_spaces.viz import _level_image

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHOEBOX = EXAMPLES / "shoebox" / "scene.yaml"
APARTMENT = EXAMPLES / "apartment" / "scene.yaml"


def _shoebox(**changes) -> Room:
    data = Scene.from_yaml(SHOEBOX).model_dump(mode="json", exclude_none=True)
    for key, value in changes.items():
        data[key] = value
    return Room.from_scene(Scene.model_validate(data, context={"base_dir": SHOEBOX.parent}))


def test_grid_covers_the_floor() -> None:
    room = _shoebox()
    assert audience_surfaces(room) == ["floor"]
    grid = map_grid(room, height=1.2, spacing=1.0)
    assert len(grid) == 48
    assert grid.index.shape == (6, 8, 1)
    assert sorted(grid.index.ravel().tolist()) == list(range(48))
    np.testing.assert_allclose(grid.points[:, 2], 1.2)
    np.testing.assert_allclose(grid.volume, 1.0)
    assert grid.low.min() >= 0.7 - 1e-9 and np.all(grid.high - grid.low == pytest.approx(1.0))
    corners = grid.squares()
    assert corners.shape == (48, 4, 3)
    np.testing.assert_allclose(corners[:, :, :2].min(axis=(0, 1)), [0.0, 0.0])
    np.testing.assert_allclose(corners[:, :, :2].max(axis=(0, 1)), [8.0, 6.0])


def test_grid_columns_off_the_edge_count_part_of_their_volume() -> None:
    grid = map_grid(_shoebox(), spacing=1.5)
    # 8 m by 6 m in 1.5 m columns: 6 by 4, centred, so the end columns hang over.
    assert grid.index.shape[:2] == (4, 6)
    assert grid.origin == pytest.approx((-0.5, 0.0))
    edge = np.isin(grid.columns[:, 0], [0, 5])
    # 1 m of the end columns' 1.5 m lies in the room.
    np.testing.assert_allclose(grid.volume[edge], 2 / 3 * 1.5**2 * 1.0, rtol=0.01)
    np.testing.assert_allclose(grid.volume[~edge], 1.5**2 * 1.0)
    # The 3D view cuts the end columns into fifths: 3 of 5 across, all 5 along.
    corners, owner = grid.tiles()
    counts = np.bincount(owner, minlength=len(grid))
    assert set(counts[edge]) == {15} and set(counts[~edge]) == {1}
    sides = np.ptp(corners[:, :, 0], axis=1) * np.ptp(corners[:, :, 1], axis=1)
    assert sides.sum() == pytest.approx(grid.cover.sum() * (1.5 / 5) ** 2)


def _share(grid, rectangles) -> np.ndarray:
    """The part of each point's column in the union of ``rectangles`` (x0, y0, x1, y1)
    that do not overlap, less those with a negative sign: (sign, x0, y0, x1, y1)."""
    x0 = grid.origin[0] + grid.columns[:, 0] * grid.cell
    y0 = grid.origin[1] + grid.columns[:, 1] * grid.cell
    area = np.zeros(len(grid))
    for sign, a0, b0, a1, b1 in rectangles:
        across = np.clip(np.minimum(x0 + grid.cell, a1) - np.maximum(x0, a0), 0.0, None)
        along = np.clip(np.minimum(y0 + grid.cell, b1) - np.maximum(y0, b0), 0.0, None)
        area += sign * across * along
    return area / grid.cell**2


@pytest.mark.parametrize("spacing", [1.25, 1.5])
def test_grid_slabs_count_the_part_of_their_column_in_the_room(spacing: float) -> None:
    # The 8 m by 6 m floor in columns that overhang its walls: at 1.25 m the end
    # columns hold 0.7 of theirs, those along the sides 0.9 and the corners 0.63. The
    # volume each slab's track lengths are spread over is that part (each cut edge
    # within half of one of 24 samples across the column).
    grid = map_grid(_shoebox(), spacing=spacing)
    thickness = grid.high - grid.low
    used = grid.volume / (spacing**2 * thickness)
    exact = _share(grid, [(1, 0.0, 0.0, 8.0, 6.0)])
    assert np.any(exact < 0.95)
    np.testing.assert_allclose(used, exact, atol=1 / 24)


def test_grid_slabs_leave_out_buildings() -> None:
    # The street example in 3 m columns, 22 of which reach into one of its two
    # buildings (rectangles from above, 9 m and 15 m high), whose ground under them goes
    # on: no ray goes in, so their slabs count the open air alone.
    room = Room.from_yaml(EXAMPLES / "street" / "scene.yaml")
    grid = map_grid(room, spacing=3.0, height=1.5)
    blocks = [
        (-1, *footprint.min(axis=0), *footprint.max(axis=0))
        for _name, footprint, _roof in room.outdoor.buildings
    ]
    ground = (1, *room.report.bounds_min[:2], *room.report.bounds_max[:2])
    exact = _share(grid, [ground, *blocks])
    straddling = _share(grid, [(-sign, *rest) for sign, *rest in blocks]) > 0.0
    assert straddling.sum() == 22 and exact[straddling].min() == pytest.approx(0.5)
    used = grid.volume / (grid.cell**2 * (grid.high - grid.low))
    np.testing.assert_allclose(used, exact, atol=1 / 24)
    # The run without the houses for their insertion loss shares the points, and its
    # rays fill the columns where the houses stood.
    houses = _share(grid, [(1, *rest) for _sign, *rest in blocks[:1]]) > 0.0
    bare = _in_room(bare_room(room, [room.outdoor.buildings[0][0]]), grid)
    np.testing.assert_allclose(bare.volume[houses], grid.cell**2 * (grid.high - grid.low)[houses])
    np.testing.assert_array_equal(
        bare.volume[~houses & straddling], grid.volume[~houses & straddling]
    )


def _apartment(transmission: bool = True) -> Room:
    """The apartment example, or without its materials that let sound through."""
    data = Scene.from_yaml(APARTMENT).model_dump(mode="json", exclude_none=True)
    if not transmission:
        for name in ("wall", "closed", "open"):
            del data["materials"][name]
        data["assignments"].update(partition="plaster", door="plaster")
    room = Room.from_scene(Scene.model_validate(data, context={"base_dir": APARTMENT.parent}))
    assert (room.face_transmission() is None) != transmission
    return room


def _late_energy(room: Room, surfaces: list[str], spacing: float):
    """The map grid and the late energy (points, bands) the rays bring each point."""
    model = TraceModel.from_room(room)
    c = room.speed_of_sound()
    source = np.array(room.scene.sources[0].position)
    grid = map_grid(room, surfaces, spacing=spacing)
    found = stochastic_map_energy(
        model,
        source,
        grid,
        np.linalg.norm(grid.points - source, axis=1) / c,
        rays=3000,
        max_time=0.5,
        speed_of_sound=c,
        air=room.air_attenuation(model.bands),
        seed=1,
        threads=2,
    )
    return grid, found.energy.sum(axis=2)


def _level(energy: np.ndarray, these: np.ndarray, those: np.ndarray) -> float:
    """The mean energy of points ``these`` over that of ``those``, in dB."""
    return 10 * math.log10(energy[these].mean() / energy[those].mean())


def test_a_column_across_a_wall_between_rooms_gets_a_point_in_each() -> None:
    # The apartment in 0.7 m columns: those from x = 4.85 m cross the partition at 5 m,
    # with the bedroom's late energy 31 dB below the living room's at 1 kHz. The
    # bedroom's points there get the bedroom's, as its next column does, the living
    # room's 0.15 m its own point with the living room's. Counting both rooms' rays over
    # the whole column read 24 dB too high at 1 kHz in the bedroom.
    room = _apartment()
    grid, energy = _late_energy(room, ["living_floor", "bedroom_floor"], 0.7)
    x = grid.points[:, 0].round(2)
    bands = slice(1, 7)  # 125 Hz to 4 kHz, the bedroom's sound mostly at the bottom
    late = energy[:, bands].sum(axis=1)
    assert abs(_level(late, x == 5.2, x == 5.9)) < 1.0
    assert abs(_level(energy[:, 4], x == 4.92, x == 4.5)) < 1.0
    living, bedroom = room.room_of([[2.5, 3.2, 1.1], [7.5, 1.5, 0.7]])
    assert set(grid.room[x == 4.92]) == {living} and set(grid.room[x == 5.2]) == {bedroom}
    assert set(grid.room[x < 4.9]) == {living} and set(grid.room[x > 5.0]) == {bedroom}
    np.testing.assert_array_equal(grid.level, 0)  # one floor
    # Away from the outer walls the bedroom's points hold 0.55 m of 0.7 m.
    inner = (x == 5.2) & (grid.points[:, 1] > 1.0) & (grid.points[:, 1] < 3.0)
    used = grid.volume[inner] / (0.7**2 * (grid.high - grid.low)[inner])
    np.testing.assert_allclose(used, 0.55 / 0.7, atol=1 / 24)


def test_a_map_of_one_room_leaves_out_the_next() -> None:
    # The bedroom alone in 0.7 m columns: the first reaches 0.1 m into the much louder
    # living room, whose rays its slabs counted (22 dB too high at 1 kHz).
    grid, energy = _late_energy(_apartment(), ["bedroom_floor"], 0.7)
    late = energy[:, 1:7].sum(axis=1)
    assert abs(_level(late, grid.columns[:, 0] == 0, grid.columns[:, 0] == 1)) < 1.0


def test_a_wall_that_lets_nothing_through_keeps_each_rooms_rays() -> None:
    # Without the partition's and the door's transmission the bedroom hears nothing, not
    # even at the points of the columns that cross into the living room.
    room = _apartment(transmission=False)
    grid, energy = _late_energy(room, ["living_floor", "bedroom_floor"], 0.7)
    x = grid.points[:, 0].round(2)
    assert np.all(energy[x > 5.0] == 0.0)
    assert abs(_level(energy[:, 4], x == 4.92, x == 4.5)) < 1.0


def test_a_map_image_shows_each_rooms_part_of_a_column_with_its_own_point() -> None:
    # The apartment's 4 m deep floor in 0.7 m columns, drawn in fifths where the wall
    # between the rooms at x = 5 m crosses them (x 4.85 to 5.55 m): each fifth shows the
    # point of its room. In the first and last rows the living room's fifths over the
    # floor are 4 of 25, too few for a point of their own: they stay blank rather than
    # show the bedroom's point.
    room = Room.from_yaml(APARTMENT)
    living, bedroom = room.room_of([[2.5, 3.2, 1.1], [7.5, 1.5, 0.7]])
    grid = map_grid(room, ["living_floor", "bedroom_floor"], spacing=0.7)
    image = _level_image(grid, grid.room.astype(float), np.arange(len(grid)))
    x = grid.origin[0] + (np.arange(image.shape[1]) + 0.5) * grid.cell / 5
    y = grid.origin[1] + (np.arange(image.shape[0]) + 0.5) * grid.cell / 5
    over = (y > 0.0) & (y < 4.0)
    floor = image[over]
    assert set(floor[:, (x > 5.0) & (x < 5.55)].ravel()) == {bedroom}
    near = floor[:, np.isclose(x, 4.92)].ravel()  # the living room's fifths
    rows = (y[over] > 0.6) & (y[over] < 3.4)  # the second to fifth rows of columns
    assert rows.sum() == 20
    np.testing.assert_array_equal(near[rows], living)
    assert np.isnan(near[~rows]).all()


def test_grid_puts_a_balcony_over_the_stalls(tmp_path: Path) -> None:
    obj = tmp_path / "two_floors.obj"
    obj.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 10 0 0",
                "v 10 10 0",
                "v 0 10 0",
                "v 6 0 3",
                "v 10 0 3.5",
                "v 10 10 3.5",
                "v 6 10 3",
                "v 0 10 5",
                "v 0 0 5",
                "g audience_stalls",
                "f 1 2 3 4",
                "g audience_balcony",
                "f 5 6 7 8",
                "g wall",
                "f 1 4 9 10",
            ]
        )
    )
    scene = Scene.model_validate(
        {
            "geometry": {"file": str(obj)},
            "materials": {"m": {"absorption": [0.1] * 8, "scattering": 0.1}},
            "default_material": "m",
            "sources": [{"name": "s", "position": [1.0, 5.0, 1.5]}],
            "listeners": [{"name": "l", "position": [3.0, 5.0, 1.2]}],
        }
    )
    room = Room.from_scene(scene)
    assert audience_surfaces(room) == ["audience_stalls", "audience_balcony"]
    grid = map_grid(room, height=1.0, spacing=2.0)
    assert grid.index.shape == (5, 5, 2)
    under = grid.columns[:, 0] >= 3  # columns from x = 6 m
    assert np.sum(under) == 2 * 2 * 5
    upper = grid.points[:, 2] > 3.0
    np.testing.assert_allclose(grid.points[~upper, 2], 1.0)
    # The balcony rises from 3 m to 3.5 m: each point is 1 m above the slope under it.
    np.testing.assert_allclose(grid.points[upper, 2] - grid.surface[upper], 1.0)
    assert np.all(grid.points[upper, 0] > 6.0)
    # Only the floors that face up: the wall is ignored.
    with pytest.raises(ValueError, match="faces up"):
        map_grid(room, ["wall"])
    with pytest.raises(ValueError, match="no surface named stage"):
        map_grid(room, ["stage"])


def test_grid_stands_on_the_top_of_a_slab(tmp_path: Path) -> None:
    # A slab 0.3 m thick with its underside and its top in one group, the underside
    # first, as an IFC slab read without `space:` has them.
    obj = tmp_path / "slab.obj"
    obj.write_text(
        "\n".join(
            [
                "v 0 0 -0.3",
                "v 8 0 -0.3",
                "v 0 6 -0.3",
                "v 0 0 0",
                "v 8 0 0",
                "v 0 6 0",
                "g audience_slab",
                "f 1 3 2",
                "f 4 5 6",
            ]
        )
    )
    scene = Scene.model_validate(
        {
            "geometry": {"file": str(obj)},
            "materials": {"m": {"absorption": [0.1] * 8, "scattering": 0.1}},
            "default_material": "m",
            "sources": [{"name": "s", "position": [1.0, 1.0, 1.5]}],
            "listeners": [{"name": "l", "position": [2.0, 2.0, 1.2]}],
        }
    )
    grid = map_grid(Room.from_scene(scene), height=1.2, spacing=1.0)
    # Columns over the slab and along its slanted edge, whose centre is off it, alike.
    assert len(grid) == 26 and not grid.cover[:, 12].all()
    np.testing.assert_allclose(grid.surface, 0.0)
    np.testing.assert_allclose(grid.points[:, 2], 1.2)


def test_map_matches_listeners_at_its_points() -> None:
    room = _shoebox()
    grid = map_grid(room)
    picks = [3, 21, 44]
    listeners = [{"name": f"p{k}", "position": grid.points[k].tolist()} for k in picks]
    data = room.scene.model_dump(mode="json", exclude_none=True)
    data["listeners"] = listeners
    data["simulation"]["seed"] = 7
    data["outputs"] = [{"format": "mono"}]
    room = Room.from_scene(Scene.model_validate(data, context={"base_dir": SHOEBOX.parent}))

    responses = simulate(room)
    coverage = coverage_map(room)
    assert isinstance(coverage, CoverageResult)
    assert coverage.combined is None
    values = coverage.get("talker").values
    assert set(values) == set(ROOM_MAP_PARAMETERS)  # a room's: no outdoor ones
    for k in picks:
        response = responses["talker", f"p{k}"]
        p = response.parameters()
        bands = slice(1, 7)  # 125 Hz to 4 kHz, where both have enough rays
        np.testing.assert_allclose(values["g"][k, bands], p.g[bands], atol=0.3)
        np.testing.assert_allclose(values["c80"][k, bands], p.c80[bands], atol=0.6)
        np.testing.assert_allclose(values["edt"][k, bands], p.edt[bands], rtol=0.08)
        np.testing.assert_allclose(values["t30"][k, bands], p.t30[bands], rtol=0.05)
        # The talker's sound power per band (a man's speech spectrum), plus G less the
        # free-field level 10 m from a source of 1 pW: 10 log10(1 / (4 pi 10^2)).
        power = room.scene.sources[0].power_db("octave")
        spl = power + p.g + 10 * math.log10(1 / (400 * math.pi))
        np.testing.assert_allclose(values["spl"][k, bands], spl[bands], atol=0.3)
        sti = speech_transmission_index(response.energy(), response.time_step, signal_db=spl)
        assert values["sti"][k] == pytest.approx(sti, abs=0.02)


def test_sources_together_add_their_energy() -> None:
    sources = [
        {"name": "a", "position": [2.0, 3.0, 1.5], "level_db": 85.0},
        {"name": "b", "position": [6.5, 1.5, 1.5], "level_db": 79.0},
    ]
    simulation = {
        "early": {"method": "image_sources", "max_order": 2},
        "late": {"method": "stochastic_ray_tracing", "rays": 5000},
        "ir_length_s": 1.5,
        "seed": 3,
    }
    room = _shoebox(sources=sources, simulation=simulation)
    coverage = coverage_map(room, settings=Coverage(spacing=2.0))
    a, b, both = coverage.get("a"), coverage.get("b"), coverage.get()
    assert both is coverage.combined and both.source == "all sources"
    total = 10 * np.log10(10 ** (a.values["spl"] / 10) + 10 ** (b.values["spl"] / 10))
    np.testing.assert_allclose(both.values["spl"], total, atol=0.05)
    assert np.all(both.value("spl_a") >= np.maximum(a.value("spl_a"), b.value("spl_a")))
    # Strength is relative to the loudest source alone.
    g = total - 85.0 + 10 * math.log10(8 * 400 * math.pi)
    np.testing.assert_allclose(both.values["g"], g, atol=1e-3)
    assert coverage.details["map"]["points"] == len(coverage.grid)
    assert set(coverage.details) == {"map", "a", "b"}
    with pytest.raises(ValueError, match="unknown source"):
        coverage.get("c")
    only = coverage_map(room, settings=Coverage(spacing=2.0), sources=["b"])
    assert list(only.maps) == ["b"] and only.combined is None


def test_statistical_tail_and_noise() -> None:
    simulation = {
        "early": {"method": "image_sources", "max_order": 1},
        "late": {"method": "statistical"},
        "ir_length_s": 2.0,
        "seed": 1,
    }
    room = _shoebox(simulation=simulation)
    quiet = coverage_map(room, settings=Coverage(spacing=2.0)).get()
    noisy = coverage_map(room, settings=Coverage(spacing=2.0, noise_db=70.0)).get()
    eyring = room.reverberation_time("eyring", air_absorption=True)
    np.testing.assert_allclose(np.median(quiet.values["t30"], axis=0)[2:7], eyring[2:7], rtol=0.1)
    assert np.all(noisy.value("sti") < quiet.value("sti") - 0.1)
    assert np.all((0 <= quiet.value("sti")) & (quiet.value("sti") <= 1))


def test_statistical_map_matches_listeners_at_its_points() -> None:
    # Barron's reflected energy less the early paths, as at a listener: the scattered
    # share of the early reflections and the higher orders before the transition count.
    simulation = {
        "early": {"method": "image_sources", "max_order": 1},
        "late": {"method": "statistical"},
        "ir_length_s": 1.5,
        "seed": 1,
    }
    grid = map_grid(_shoebox(), spacing=2.0)
    picks = [0, len(grid) // 2]
    listeners = [{"name": f"p{k}", "position": grid.points[k].tolist()} for k in picks]
    room = _shoebox(simulation=simulation, listeners=listeners)
    values = coverage_map(room, settings=Coverage(spacing=2.0)).get().values
    responses = simulate(room)
    for k in picks:
        g = responses["talker", f"p{k}"].parameters().g
        np.testing.assert_allclose(values["g"][k], g, atol=0.01)


def test_short_responses_end_within_the_fine_late_slots() -> None:
    # 105 ms: the late energy has coarse slots, but the far points end before them.
    simulation = {
        "early": {"max_order": 1},
        "late": {"method": "stochastic_ray_tracing", "rays": 1000},
        "ir_length_s": 0.105,
        "seed": 1,
    }
    m = coverage_map(_shoebox(simulation=simulation), settings=Coverage(spacing=2.0)).get()
    assert m.values["g"].shape == (len(m.grid), 8)
    # Where the decay of a band from 125 Hz up has not reached -20 dB, G and the STI
    # are undefined.
    defined = np.isfinite(m.values["g"][:, 1:]).all(axis=1)
    np.testing.assert_array_equal(np.isfinite(m.value("sti")), defined)


def test_no_sti_without_the_level_of_every_band() -> None:
    def sti(t60, seconds: float) -> float:
        t = np.arange(round(seconds / TIME_STEP)) * TIME_STEP
        energy = np.array([np.exp(-6.0 * math.log(10.0) * t / v) for v in t60])
        return float(_point_values(energy, np.full(8, 85.0), 1e-2, None, "male")["sti"])

    full = sti([0.5] * 8, 2.0)
    assert 0.5 < full < 0.8
    # 50 ms of a decay of 0.5 s: 6 dB, too little for G in any band (it was 0).
    assert math.isnan(sti([0.5] * 8, 0.05))
    # One band too slow to decay by 20 dB in 2 s: 125 Hz counts, 63 Hz does not.
    assert math.isnan(sti([0.5, 10.0] + [0.5] * 6, 2.0))
    assert sti([10.0] + [0.5] * 7, 2.0) == pytest.approx(full)


def test_map_values_and_json() -> None:
    room = _shoebox(simulation={"late": {"method": "statistical"}, "seed": 2})
    coverage = coverage_map(room, settings=Coverage(spacing=2.0, surfaces=["floor"]))
    m = coverage.get()
    assert m.value("c80", 4).shape == (len(coverage.grid),)
    assert m.value("sti").shape == (len(coverage.grid),)
    with pytest.raises(ValueError, match="give a band"):
        m.value("c80")
    with pytest.raises(ValueError, match="unknown parameter"):
        m.value("rasti")
    data = json.loads(json.dumps(m.as_dict()))
    assert data["source"] == "talker"
    assert len(data["points"]) == len(coverage.grid)
    assert len(data["values"]["t30"][0]) == 8


def test_map_ignores_listeners_outside_the_room() -> None:
    room = _shoebox(
        listeners=[{"name": "outside", "position": [20.0, 3.0, 1.2]}],
        simulation={"late": {"method": "statistical"}},
    )
    assert any(p.startswith("listener ") for p in room.problems())
    assert len(coverage_map(room, settings=Coverage(spacing=3.0)).grid) == 6
    outside = _shoebox(sources=[{"name": "s", "position": [20.0, 3.0, 1.2]}])
    with pytest.raises(ValueError, match="outside the room"):
        coverage_map(outside)


def test_coverage_settings_are_checked() -> None:
    assert Coverage().height == 1.2 and Coverage().spacing == 1.0
    assert Coverage(noise_db=[40.0] * 8).noise_db == [40.0] * 8
    for bad in ({"noise_db": [40.0, 30.0]}, {"spacing": 0}, {"height": -1}, {"rays": 0}):
        with pytest.raises(ValueError):
            Coverage.model_validate(bad)
    scene = Scene.from_yaml(SHOEBOX)
    assert scene.coverage is None
    data = scene.model_dump(mode="json", exclude_none=True)
    data["coverage"] = {"surfaces": ["floor"], "spacing": 0.5}
    again = Scene.model_validate(data, context={"base_dir": SHOEBOX.parent})
    assert again.coverage.spacing == 0.5 and again.coverage.surfaces == ["floor"]
