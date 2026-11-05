"""The spread of the results that the materials' uncertain absorption leads to."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine.reweight import groups_of, log_ratio
from acoustic_spaces.engine.uncertainty import (
    EDGE_ERRORS_DB,
    MAX_ABSORPTION,
    MIN_BATCHES,
    absorption_draws,
    edge_errors,
    material_spread,
)

WALLS = [0.04, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12]


def _box(uncertainty: dict | None) -> Scene:
    walls = {"absorption": WALLS, "scattering": 0.3}
    if uncertainty is not None:
        walls["uncertainty"] = uncertainty
    return Scene.model_validate(
        {
            "geometry": {"box": [8, 6, 3.5]},
            "materials": {
                "floor": {"absorption": [0.1, 0.15, 0.25, 0.35, 0.4, 0.45, 0.45, 0.45]},
                "ceiling": {"absorption": [0.15] * 8},
                "walls": walls,
            },
            "assignments": {"floor": "floor", "ceiling": "ceiling", "walls": "walls"},
            "sources": [{"name": "s", "position": [2.0, 2.6, 1.5]}],
            "listeners": [{"name": "a", "position": [5.8, 3.9, 1.2]}],
            "simulation": {"seed": 2, "ir_length_s": 1.6, "late": {"rays": 500}},
        }
    )


def test_an_uncertainty_is_given_per_band_or_once_and_a_tolerance_counts_a_third():
    scene = _box({"absorption": 0.03, "distribution": "uniform", "source": "ISO 354 lab"})
    spec = scene.materials["walls"].uncertainty
    assert spec.absorption == [0.03] * 8
    np.testing.assert_allclose(spec.standard(), 0.03 / np.sqrt(3.0))
    assert scene.materials["walls"].to_data()["uncertainty"]["distribution"] == "uniform"
    assert Scene.model_validate(scene.to_data()).materials["walls"].uncertainty == spec
    for wrong in ({"absorption": [0.01] * 3}, {"absorption": -0.01}, {"absorption": 0.1, "x": 1}):
        with pytest.raises(ValueError):
            _box(wrong)


def test_draws_share_one_deviate_per_material_and_stay_in_range():
    scene = _box({"absorption": 0.05})
    taken = absorption_draws(scene, ["floor", "walls"], 4000, seed=1)
    assert taken.shape == (4000, 2, 8)
    # A material without an uncertainty keeps its absorption.
    np.testing.assert_array_equal(taken[:, 0], np.broadcast_to(taken[0, 0], (4000, 8)))
    walls = taken[:, 1]
    # The bands move together, by the same deviate (where none is held at 0).
    free = np.all(walls > 0.0, axis=1)
    deviation = walls[free] - np.asarray(WALLS)
    np.testing.assert_allclose(deviation, deviation[:, :1] * np.ones(8), atol=1e-12)
    assert walls.min() >= 0.0 and walls.max() <= MAX_ABSORPTION
    # The 8 kHz band (0.12, 2.4 u from 0) is barely held: its spread is the uncertainty.
    assert abs(walls[:, 7].std() - 0.05) < 0.003
    uniform = absorption_draws(
        _box({"absorption": 0.03, "distribution": "uniform"}), ["walls"], 4000, 1
    )
    deviation = uniform[:, 0, 7] - WALLS[7]
    assert np.abs(deviation).max() <= 0.03 and abs(deviation.std() - 0.03 / np.sqrt(3.0)) < 0.002


def test_the_spread_of_the_parameters_follows_the_absorption_s():
    scene = _box({"absorption": 0.02})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    spread = material_spread(result, scene, face_group, materials, draws=60, seed=3, rays=False)
    found = spread["s", "a"]
    assert found.draws == 60 and found.coverage == 0.95 and found.batches == 0
    assert np.all(np.isnan(found.rays.t30))
    np.testing.assert_array_equal(found.standard.t30, found.absorption.t30)
    traced = result["s", "a"].parameters()
    bands = slice(1, 7)
    # The draws centre on the traced values and spread out from them.
    np.testing.assert_allclose(found.mean.t30[bands], traced.t30[bands], rtol=0.03)
    assert np.all(found.standard.t30[bands] > 0.0)
    assert np.all(found.low.t30[bands] < traced.t30[bands])
    assert np.all(found.high.t30[bands] > traced.t30[bands])
    # The decay time falls as the walls absorb more, faster the less they absorb, so
    # its spread lies between the changes that u up and u down bring, each re-weighted
    # alone, near their mean.
    before = np.array([scene.materials[m].coefficients("octave")[0] for m in materials])
    changes = []
    for sign in (1.0, -1.0):
        after = before.copy()
        after[materials.index("walls")] += sign * 0.02
        moved = result["s", "a"].reweighted(
            face_group, log_ratio(before, after), absorption=(before, after)
        )
        changes.append(np.abs(moved.parameters().t30[bands] - traced.t30[bands]))
    up, down = changes
    assert np.all(up < down)
    assert np.all(found.standard.t30[bands] > 0.9 * up)
    assert np.all(found.standard.t30[bands] < 1.1 * down)
    np.testing.assert_allclose(found.standard.t30[bands], 0.5 * (up + down), rtol=0.25)
    # No material with an uncertainty: nothing to spread.
    plain = _box(None)
    with pytest.raises(ValueError, match="no material"):
        material_spread(result, plain, face_group, materials, draws=10)


def test_the_spread_lists_the_errors_over_edges_it_does_not_count():
    """Where a response has paths over edges, the spread names the method's errors over
    them, each with the validation's largest error in each band as the band adds its
    paths: in pressure in the lowest bands, in energy above. A box has no edges."""
    from types import SimpleNamespace

    scene = _box({"absorption": 0.02})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    spread = material_spread(result, scene, face_group, materials, draws=10, rays=False)
    assert spread["s", "a"].not_counted == {}
    assert spread["s", "a"].as_dict()["not_counted"] == {}

    def response(edges, coherent):
        early = SimpleNamespace(
            edge=np.array(edges),
            energy=np.zeros((len(edges), 8)),
            spectrum=None if coherent == 0 else np.zeros((len(edges), coherent, 8)),
        )
        return SimpleNamespace(early=early)

    assert edge_errors(response([-1, -1], 2), scene) == {}
    assert edge_errors(response([-1, -2], 2), scene) == {}  # over the terrain's crests
    found = edge_errors(response([-1, 4], 2), scene)
    # first_order with one_side by default: a path reflected on both sides is left out.
    assert list(found) == ["one edge", "one side", "corners of edges"]
    for name, bound in found.items():
        table = EDGE_ERRORS_DB[name]
        np.testing.assert_array_equal(bound[:2], table["pressure"][:2])
        np.testing.assert_array_equal(bound[2:], table["energy"][2:])
    assert np.all(np.isnan(found["corners of edges"][2:]))  # not measured in energy
    wide = scene.model_copy(deep=True)
    wide.simulation.early.diffraction = "second_order"
    wide.simulation.early.diffraction_reflections = "both_sides"
    found = edge_errors(response([7], 8), wide)
    assert list(found) == ["one edge", "two edges", "corners of edges"]
    np.testing.assert_array_equal(found["two edges"], EDGE_ERRORS_DB["two edges"]["pressure"])


def test_the_rays_error_adds_to_the_absorption_s_as_student_s_t():
    """The rays' standard error, from 10 batches, enters as Student's t of 9 degrees of
    freedom: its standard uncertainty is the error times sqrt(9 / 7), and the combined
    one the root of the sum of the squares of the two, as JCGM 100 combines independent
    inputs; the absorption's own draws stay as they were."""
    scene = _box({"absorption": 0.02})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    response = result["s", "a"]
    assert len(response.late_batches) == 10
    both = material_spread(result, scene, face_group, materials, draws=400, seed=3)["s", "a"]
    alone = material_spread(result, scene, face_group, materials, draws=400, seed=3, rays=False)
    alone = alone["s", "a"]
    assert both.batches == 10
    errors = response.standard_errors()
    bands = slice(1, 7)
    for name in ("t30", "edt", "c80", "d50", "g"):
        np.testing.assert_allclose(
            getattr(both.rays, name)[bands],
            getattr(errors, name)[bands] * np.sqrt(9.0 / 7.0),
            rtol=1e-12,
        )
        np.testing.assert_array_equal(
            getattr(both.absorption, name), getattr(alone.absorption, name)
        )
        combined = np.hypot(getattr(both.absorption, name), getattr(both.rays, name))
        # 400 draws give a standard deviation within about 7 % (two of its own).
        np.testing.assert_allclose(getattr(both.standard, name)[bands], combined[bands], rtol=0.1)
        assert np.all(getattr(both.high, name)[bands] > getattr(both.low, name)[bands])
    # The lateral fraction has no standard error: the absorption alone counts.
    assert np.all(np.isnan(both.rays.lf))
    np.testing.assert_array_equal(both.standard.lf, alone.standard.lf)
    # With fewer than MIN_BATCHES batches Student's t has no variance: the rays do not
    # count.
    few = scene.model_copy(deep=True)
    few.simulation.late.batches = MIN_BATCHES - 1
    room = Room.from_scene(few)
    result = simulate(room, sources=["s"], groups=face_group)
    spread = material_spread(result, few, face_group, materials, draws=10)["s", "a"]
    assert spread.batches == 0 and np.all(np.isnan(spread.rays.t30))


def test_simulate_writes_the_spread_when_a_material_has_an_uncertainty(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    import json

    import yaml

    from acoustic_spaces.cli import main

    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(_box({"absorption": 0.02, "source": "lab"}).to_data()))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300", "--spread-draws", "12"]
    assert main(args) == 0
    stdout = capsys.readouterr().out
    assert "uncertainty from the absorption and the rays (10 batches), 12 draws: at most" in stdout
    assert "(absorption " in stdout and ", rays " in stdout
    entry = json.loads((out / "parameters.json").read_text())["responses"][0]
    spread = entry["uncertainty"]
    assert spread["draws"] == 12 and spread["coverage"] == 0.95 and spread["batches"] == 10
    assert spread["standard"]["t30"][3] > 0.0
    assert spread["absorption"]["t30"][3] > 0.0 and spread["rays"]["t30"][3] > 0.0
    assert spread["low"]["t30"][3] < entry["parameters"]["t30"][3] < spread["high"]["t30"][3]
    # Third-octave bands do not re-weight: the run goes on without the spread.
    assert main([*args, "--bands", "third_octave"]) == 0
    captured = capsys.readouterr()
    assert "warning: no spread" in captured.err
    entry = json.loads((out / "parameters.json").read_text())["responses"][0]
    assert "uncertainty" not in entry
