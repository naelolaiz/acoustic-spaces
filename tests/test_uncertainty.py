"""The spread of the results that the materials' uncertain absorption leads to."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine.reweight import groups_of, log_ratio
from acoustic_spaces.engine.uncertainty import MAX_ABSORPTION, absorption_draws, material_spread

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
    spread = material_spread(result, scene, face_group, materials, draws=60, seed=3)
    found = spread["s", "a"]
    assert found.draws == 60 and found.coverage == 0.95
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
