import json
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import CoverageResult, coverage_map, map_grid
from acoustic_spaces.engine.coverage import MAP_PARAMETERS, audience_surfaces
from acoustic_spaces.parameters import speech_transmission_index
from acoustic_spaces.scene import Coverage

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


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
    np.testing.assert_allclose(grid.volume[edge], 0.6 * 1.5**2 * 1.0)
    np.testing.assert_allclose(grid.volume[~edge], 1.5**2 * 1.0)
    # The 3D view cuts the end columns into fifths: 3 of 5 across, all 5 along.
    corners, owner = grid.tiles()
    counts = np.bincount(owner, minlength=len(grid))
    assert set(counts[edge]) == {15} and set(counts[~edge]) == {1}
    sides = np.ptp(corners[:, :, 0], axis=1) * np.ptp(corners[:, :, 1], axis=1)
    assert sides.sum() == pytest.approx(grid.volume.sum() / 1.0)  # slabs 1 m thick


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
    assert set(values) == set(MAP_PARAMETERS)
    for k in picks:
        response = responses["talker", f"p{k}"]
        p = response.parameters()
        bands = slice(1, 7)  # 125 Hz to 4 kHz, where both have enough rays
        np.testing.assert_allclose(values["g"][k, bands], p.g[bands], atol=0.3)
        np.testing.assert_allclose(values["c80"][k, bands], p.c80[bands], atol=0.6)
        np.testing.assert_allclose(values["edt"][k, bands], p.edt[bands], rtol=0.08)
        np.testing.assert_allclose(values["t30"][k, bands], p.t30[bands], rtol=0.05)
        # The talker's 94 dB of sound power, spread evenly over the eight bands.
        spl = 94.0 - 10 * math.log10(8) + p.g + 10 * math.log10(1 / (400 * math.pi))
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
