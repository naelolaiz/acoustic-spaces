"""The uncertainty of a coverage map from its materials' uncertain absorption."""

from __future__ import annotations

import dataclasses
import json
import math
from statistics import NormalDist

import numpy as np
import pytest
from test_long_term import WALL, field

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import coverage as coverage_stage
from acoustic_spaces.engine import long_term, sensitivity
from acoustic_spaces.engine.coverage import (
    LONG_TERM_INSERTION_LOSS_MAP_PARAMETERS,
    LONG_TERM_MAP_PARAMETERS,
    uncertainty_text,
)
from acoustic_spaces.engine.reweight import groups_of, with_absorption
from acoustic_spaces.engine.uncertainty import MAX_ABSORPTION, absorption_nodes, material_spread

FLOOR = [0.1, 0.15, 0.25, 0.35, 0.4, 0.45, 0.45, 0.45]
WALLS = [0.14, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.22]
#: A tolerance's standard deviation, in tolerances (GUM, JCGM 100:2008, 4.3.7).
STEP = 1.0 / math.sqrt(3.0)


def _box(*, listener=(5.8, 3.9, 1.2), coverage=None, simulation=None, **materials) -> Scene:
    """A box whose floor has a tolerance of 0.1 and whose walls a standard uncertainty
    of 0.03, its ceiling none, with a map of six points over its floor; ``materials``
    replace those of the same names."""
    given = {
        "floor": {
            "absorption": FLOOR,
            "uncertainty": {"absorption": 0.1, "distribution": "uniform"},
        },
        "ceiling": {"absorption": [0.15] * 8},
        "walls": {"absorption": WALLS, "scattering": 0.3, "uncertainty": {"absorption": 0.03}},
        **materials,
    }
    return Scene.model_validate(
        {
            "geometry": {"box": [8, 6, 3.5]},
            "materials": given,
            "assignments": {"floor": "floor", "ceiling": "ceiling", "walls": "walls"},
            "sources": [
                {
                    "name": "s",
                    "position": [2.0, 2.6, 1.5],
                    "level_db": 70,
                    "spectrum": "speech_male",
                }
            ],
            "listeners": [{"name": "a", "position": list(listener)}],
            "coverage": {"surfaces": ["floor"], "spacing": 3.0, **(coverage or {})},
            "simulation": {
                "seed": 2,
                "ir_length_s": 1.2,
                "late": {"rays": 500},
                **(simulation or {}),
            },
        }
    )


def test_the_nodes_are_a_draws_absorption_at_one_standard_deviation() -> None:
    scene = _box()
    plus, minus = absorption_nodes(scene, "walls")
    np.testing.assert_allclose(plus - WALLS, 0.03)
    np.testing.assert_allclose(WALLS - minus, 0.03)
    # A tolerance of 0.1 is a rectangular distribution of that half-width.
    plus, minus = absorption_nodes(scene, "floor")
    np.testing.assert_allclose(plus - FLOOR, 0.1 * STEP)
    np.testing.assert_allclose(FLOOR - minus, 0.1 * STEP)
    # Held where a draw is: between no absorption and the most a draw takes, and not
    # moved where the material absorbs everything.
    edge = [0.02, 0.02, 0.5, 0.5, 0.97, 0.97, 1.0, 1.0]
    ceiling = {"absorption": edge, "uncertainty": {"absorption": 0.1, "distribution": "uniform"}}
    plus, minus = absorption_nodes(_box(ceiling=ceiling), "ceiling")
    most = MAX_ABSORPTION
    np.testing.assert_allclose(
        plus, [0.02 + 0.1 * STEP] * 2 + [0.5 + 0.1 * STEP] * 2 + [most, most, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        minus, [0.0, 0.0] + [0.5 - 0.1 * STEP] * 2 + [0.97 - 0.1 * STEP] * 2 + [1.0, 1.0]
    )
    with pytest.raises(ValueError, match="'ceiling' has no absorption uncertainty"):
        absorption_nodes(scene, "ceiling")


def test_moved_absorption_is_kept_as_given() -> None:
    scene = _box()
    values = np.array(WALLS) + 0.1 * STEP
    moved = with_absorption(scene, ["walls"], [values], "moved", digits=None)
    assert moved.materials["walls"].absorption == [float(v) for v in values]
    assert moved.materials["walls"].source == "moved"
    assert moved.materials["walls"].scattering == scene.materials["walls"].scattering
    assert moved.materials["floor"] == scene.materials["floor"]
    rounded = with_absorption(scene, ["walls"], [values], "moved")
    assert rounded.materials["walls"].absorption == [round(float(v), 4) for v in values]


def test_a_maps_spread_is_the_root_sum_of_squares_of_its_materials_parts() -> None:
    scene = _box()
    room = Room.from_scene(scene)
    assert sensitivity.spread_materials(room) == ["floor", "walls"]  # the ceiling has none
    said: list[str] = []
    result = sensitivity.coverage_map(room, spread=True, progress=said.append)
    assert result.details["spread"]["materials"] == 2.0
    assert result.details["spread"]["runs"] == 4.0
    assert (
        sum(message.startswith("Spread: the map with the absorption of") for message in said) == 4
    )
    chosen = result.get()
    spread = chosen.spread
    assert list(spread.materials) == ["floor", "walls"]
    assert spread.coverage == 0.95 and spread.one_sided == 0.95
    assert spread.assumed == {} and spread.held == {}

    # The map itself is the one a run without the spread gives.
    plain = long_term.coverage_map(room)
    assert plain.get().spread is None and "spread" not in plain.details
    for name, values in plain.get().values.items():
        np.testing.assert_array_equal(chosen.values[name], values)

    # A material's part is half the difference of the maps traced on the same points
    # with the same rays, its absorption one standard uncertainty up and down.
    up, down = (
        coverage_stage.coverage_map(
            Room.from_scene(with_absorption(scene, ["walls"], [absorption], "moved", digits=None)),
            grid=result.grid,
        )
        .get()
        .values
        for absorption in absorption_nodes(scene, "walls")
    )
    for name in ("g", "spl", "spl_a", "sti", "c50"):
        part = spread.materials["walls"][name]
        assert part.shape == chosen.values[name].shape
        np.testing.assert_allclose(part, 0.5 * (up[name] - down[name]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            spread.standard[name], np.hypot(part, spread.materials["floor"][name]), rtol=1e-12
        )
        band = 4 if part.ndim == 2 else None
        whole = spread.standard[name]
        np.testing.assert_array_equal(
            chosen.spread_of(name, band=band), whole if band is None else whole[:, band]
        )
    # More absorption on the same rays: less level at every point, in every band.
    assert (spread.materials["walls"]["g"] < 0).all()
    assert (spread.materials["floor"]["spl_a"] < 0).all()


def test_the_interval_and_the_bounds_take_the_result_as_normal() -> None:
    # The late part from the room's statistics, which needs no rays; a built-in material
    # of unknown source, which takes the default tolerance; a ceiling that absorbs
    # everything at 8 kHz.
    scene = _box(
        ceiling={
            "absorption": [0.3] * 7 + [1.0],
            "uncertainty": {"absorption": 0.05},
        },
        walls={"preset": "smooth_brickwork_flush_pointing"},
        simulation={"late": {"method": "statistical"}, "uncertainty": {"one_sided": 0.9}},
    )
    result = sensitivity.coverage_map(Room.from_scene(scene), spread=True)
    chosen = result.get()
    assert result.details["spread"]["runs"] == 6.0
    assert chosen.spread.assumed == {"walls": "tolerance"}
    assert chosen.spread.held == {"ceiling": [8000]}
    assert chosen.spread.one_sided == 0.9

    value, u = chosen.value("spl_a"), chosen.spread_of("spl_a")
    assert u.shape == (len(result.grid),) and (u > 0).all()
    two_sided = NormalDist().inv_cdf(0.975)  # 1.96
    one_sided = NormalDist().inv_cdf(0.9)  # 1.28
    np.testing.assert_allclose(chosen.spread_of("spl_a", "low"), value - two_sided * u)
    np.testing.assert_allclose(chosen.spread_of("spl_a", "high"), value + two_sided * u)
    np.testing.assert_allclose(chosen.spread_of("spl_a", "at_least"), value - one_sided * u)
    np.testing.assert_allclose(chosen.spread_of("spl_a", "at_most"), value + one_sided * u)
    np.testing.assert_allclose(
        chosen.spread_of("t30", "high", band=4),
        chosen.value("t30", 4) + two_sided * chosen.spread_of("t30", band=4),
    )
    # An end is held to what the result can be: an STI between 0 and 1.
    standard = {**chosen.spread.standard, "sti": np.full(len(result.grid), 0.9)}
    wide = dataclasses.replace(chosen, spread=dataclasses.replace(chosen.spread, standard=standard))
    assert (wide.spread_of("sti", "low") == 0.0).all()
    assert (wide.spread_of("sti", "at_most") == 1.0).all()

    with pytest.raises(ValueError, match="give a band"):
        chosen.spread_of("t30")
    with pytest.raises(ValueError, match="unknown part 'mean'"):
        chosen.spread_of("sti", "mean")
    with pytest.raises(ValueError, match="unknown parameter"):
        chosen.spread_of("rasti")
    with pytest.raises(ValueError, match="without its spread"):
        dataclasses.replace(chosen, spread=None).spread_of("sti")

    data = json.loads(json.dumps(result.as_dict()))
    assert data["details"]["spread"]["runs"] == 6.0
    written = data["maps"]["s"]["uncertainty"]
    assert written["coverage"] == 0.95 and written["one_sided"] == 0.9
    assert written["assumed"] == {"walls": "tolerance"}
    assert written["held"] == {"ceiling": [8000]}
    assert list(written["materials"]) == ["floor", "ceiling", "walls"]
    assert len(written["standard"]["t30"]) == len(result.grid)
    assert len(written["standard"]["t30"][0]) == 8
    np.testing.assert_allclose(written["standard"]["spl_a"], u, atol=5e-5)
    np.testing.assert_allclose(
        written["materials"]["floor"]["sti"], chosen.spread.materials["floor"]["sti"], atol=5e-5
    )


def test_the_runs_of_a_spread_trace_the_same_rays() -> None:
    """Without a seed in the scene, two runs would differ by their rays' random error,
    some tenths of a decibel with these few rays, and that would count as the
    material's part: with one seed for all the runs, a material that hardly moves has
    hardly a part."""
    walls = {"absorption": WALLS, "uncertainty": {"absorption": 1e-6}}
    scene = _box(floor={"absorption": FLOOR}, walls=walls, simulation={"seed": None})
    result = sensitivity.coverage_map(Room.from_scene(scene), spread=True)
    assert result.details["spread"]["runs"] == 2.0
    part = result.get().spread.materials["walls"]["g"]
    assert 0.0 < np.abs(part).max() < 1e-4


def test_a_map_is_run_once_without_a_spread_to_find() -> None:
    late = {"late": {"method": "statistical"}}
    # Asked for by the scene, with no uncertain material on the surfaces.
    certain = _box(
        floor={"absorption": FLOOR},
        walls={"absorption": WALLS},
        coverage={"spread": True},
        simulation=late,
    )
    said: list[str] = []
    result = sensitivity.coverage_map(Room.from_scene(certain), progress=said.append)
    assert result.get().spread is None and "spread" not in result.details
    assert not any(message.startswith("Spread") for message in said)
    # The scene's switch is the default, which the argument overrides.
    room = Room.from_scene(_box(coverage={"spread": True}, simulation=late))
    assert sensitivity.coverage_map(room).get().spread is not None
    assert sensitivity.coverage_map(room, spread=False).get().spread is None
    room = Room.from_scene(_box(simulation=late))
    assert room.scene.coverage.spread is False
    assert sensitivity.coverage_map(room).get().spread is None


def test_a_maps_spread_agrees_with_a_listeners_monte_carlo_at_a_point() -> None:
    """The law of propagation on the map against the Monte Carlo of a listener put at
    one of its points (other rays, re-weighted for each draw and not traced again).
    With 20 000 rays and 2 000 draws they agree within 4 % for the A-weighted level and
    the STI and within 8 % for the bands' levels; these few rays and draws leave the
    draws' own error, some 6 % of a standard deviation, and the rays' besides."""
    few = {"late": {"rays": 1500}, "ir_length_s": 0.8}
    result = sensitivity.coverage_map(Room.from_scene(_box(simulation=few)), spread=True)
    k = int(np.argmin(np.linalg.norm(result.grid.points - np.array([5.8, 3.9, 1.2]), axis=1)))
    law = result.get().spread.standard

    scene = _box(listener=result.grid.points[k].tolist(), simulation=few)
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    traced = simulate(room, groups=face_group)
    drawn = material_spread(traced, scene, face_group, materials, draws=150, seed=5, rays=False)
    standard, levels = drawn["s", "a"].standard, drawn["s", "a"].levels.standard
    assert law["spl_a"][k] == pytest.approx(levels.spl_a, rel=0.15)
    assert law["sti"][k] == pytest.approx(levels.sti, rel=0.2)
    bands = slice(1, 7)  # 125 Hz to 4 kHz
    np.testing.assert_allclose(law["g"][k, bands], standard.g[bands], rtol=0.15)
    np.testing.assert_allclose(law["spl"][k, bands], levels.spl[bands], rtol=0.15)
    bands = slice(2, 7)  # from 250 Hz: the response is too short for a T30 below
    np.testing.assert_allclose(law["t30"][k, bands], standard.t30[bands], rtol=0.25)


def test_a_spread_covers_the_insertion_loss_and_the_long_term_levels() -> None:
    scene = field(
        barriers=[WALL],
        classes=[{"share": 1.0}],
        insertion_loss_of=["wall"],
        coverage={"spacing": 20.0, "height": 1.5, "rays": 100},
        atmosphere={"preset": "neutral"},
    )
    data = scene.model_dump(mode="json", exclude_none=True)
    data["materials"]["hard"]["uncertainty"] = {"absorption": 0.03}
    # Left out: a ground's impedance gives its reflection, not its absorption.
    data["materials"]["field"]["uncertainty"] = {"absorption": 0.03}
    room = Room.from_scene(Scene.model_validate(data))
    assert sensitivity.spread_materials(room) == ["hard"]
    result = sensitivity.coverage_map(room, spread=True, threads=1)
    assert result.details["spread"]["runs"] == 2.0
    found = result.get()
    assert set(found.spread.standard) == set(found.values)
    outdoors = ("re_free_field", "insertion_loss", "insertion_loss_a")
    for name in (*outdoors, *LONG_TERM_MAP_PARAMETERS, *LONG_TERM_INSERTION_LOSS_MAP_PARAMETERS):
        u = found.spread.standard[name]
        assert u.shape == found.values[name].shape
        assert np.isfinite(u).all() and (u > 0).any()
    part = found.spread.materials["hard"]
    # The level without the wall is that of every run, so the insertion loss moves as
    # the level relative to free field does, the other way.
    np.testing.assert_allclose(part["insertion_loss"], -part["re_free_field"], atol=1e-9)
    # One weather the whole time: the long-term level is the level.
    np.testing.assert_allclose(part["long_term"], part["spl"], atol=1e-9)
    np.testing.assert_allclose(part["long_term_insertion_loss"], part["insertion_loss"], atol=1e-9)


def test_a_standard_uncertainty_shows_two_significant_digits() -> None:
    assert uncertainty_text(1.0103) == "1.0"
    assert uncertainty_text(0.0392) == "0.039"
    assert uncertainty_text(0.96) == "0.96"
    assert uncertainty_text(11.3) == "11"
    assert uncertainty_text(123.4) == "123"
    assert uncertainty_text(0.0) == "0"
    assert uncertainty_text(2e-9) == "0.0000"
    assert uncertainty_text(math.nan) == "–"
