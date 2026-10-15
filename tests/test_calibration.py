"""Re-weighting traced rays for new absorption, and fitting absorption to measurements."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.calibration import calibrate, with_absorption
from acoustic_spaces.cli import main
from acoustic_spaces.engine.reweight import face_groups, groups_of, log_ratio, reweighted_result

MATERIALS = ["floor", "ceiling", "walls"]
#: The absorption the recovery test measures with, floor and walls rising with frequency.
TRUTH = {
    "floor": [0.1, 0.15, 0.25, 0.35, 0.4, 0.45, 0.45, 0.45],
    "ceiling": [0.15] * 8,
    "walls": [0.04, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12],
}


def _box(absorption: dict, rays: int = 1500, length: float = 1.0, **simulation) -> Scene:
    return Scene.model_validate(
        {
            "geometry": {"box": [8, 6, 3.5]},
            "materials": {
                name: {"absorption": list(values), "scattering": 0.3}
                for name, values in absorption.items()
            },
            "assignments": {name: name for name in MATERIALS},
            "sources": [
                {"name": "s", "position": [2.0, 2.6, 1.5]},
                {"name": "t", "position": [6.4, 4.5, 1.7]},
            ],
            "listeners": [
                {"name": "a", "position": [5.8, 3.9, 1.2]},
                {"name": "b", "position": [3.3, 1.2, 1.6]},
                {"name": "c", "position": [1.3, 4.8, 1.2]},
            ],
            "simulation": {
                "seed": 2,
                "ir_length_s": length,
                "late": {"rays": rays},
                **simulation,
            },
        }
    )


def _flat(floor: float, ceiling: float, walls: float) -> dict:
    return {"floor": [floor] * 8, "ceiling": [ceiling] * 8, "walls": [walls] * 8}


def test_reweighted_responses_match_tracing_again():
    old = np.array([0.1, 0.3, 0.05])
    room = Room.from_scene(_box(_flat(*old), rays=4000, length=1.2))
    group = face_groups(room, MATERIALS)
    traced = simulate(room, sources=["s"], groups=group)
    base = traced["s", "a"]
    assert base.late_moments.shape[:2] == (1 + 3 + 6, 8)

    # The same absorption gives the same response.
    same = base.reweighted(group, np.zeros((3, 8)))
    np.testing.assert_allclose(same.energy(), base.energy(), rtol=1e-12)

    new = np.array([0.15, 0.25, 0.08])
    ratio = log_ratio(np.repeat(old[:, None], 8, 1), np.repeat(new[:, None], 8, 1))
    fresh = simulate(Room.from_scene(_box(_flat(*new), rays=4000, length=1.2)), sources=["s"])
    for listener in ("a", "b", "c"):
        guess = traced["s", listener].reweighted(group, ratio)
        again = fresh["s", listener]
        # The early paths are the same paths, so their energies re-weight exactly.
        np.testing.assert_array_equal(guess.early.delay_s, again.early.delay_s)
        np.testing.assert_allclose(guess.early.energy, again.early.energy, rtol=1e-9)
        found, expected = guess.parameters(), again.parameters()
        bands = slice(1, 7)
        np.testing.assert_allclose(found.t30[bands], expected.t30[bands], rtol=0.04)
        np.testing.assert_allclose(found.edt[bands], expected.edt[bands], rtol=0.05)
        np.testing.assert_allclose(found.c80[bands], expected.c80[bands], atol=0.3)
        np.testing.assert_allclose(found.g[bands], expected.g[bands], atol=0.1)


def test_a_run_re_weights_when_only_absorption_changes():
    scene = _box(TRUTH, rays=500, length=0.8)
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    assert materials == MATERIALS
    result = simulate(room, sources=["s"], groups=face_group)
    harder = np.multiply(TRUTH["walls"], 0.5)
    changed = with_absorption(scene, ["walls"], [harder])
    found = reweighted_result(result, scene, face_group, materials, changed)
    ratio = log_ratio(
        np.array([TRUTH[m] for m in MATERIALS]), [TRUTH["floor"], TRUTH["ceiling"], harder]
    )
    expected = result["s", "a"].reweighted(face_group, ratio)
    np.testing.assert_allclose(found["s", "a"].energy(), expected.energy())
    assert np.all(found["s", "a"].parameters().t30[2:] > result["s", "a"].parameters().t30[2:])
    assert reweighted_result(result, scene, face_group, materials, scene) is result

    # Anything else changed asks for a new run.
    moved = scene.model_copy(deep=True)
    moved.listeners[0] = moved.listeners[0].model_copy(update={"position": (5.0, 3.9, 1.2)})
    rougher = scene.model_copy(deep=True)
    rougher.materials["walls"] = rougher.materials["walls"].model_copy(
        update={"scattering": [0.5] * 8}
    )
    opened = with_absorption(scene, ["walls"], [np.ones(8)])
    for other in (moved, rougher, opened):
        assert reweighted_result(result, scene, face_group, materials, other) is None
    # So does a run whose settings keep no counts.
    for change in ({"late": {"method": "statistical"}}, {"bands": "third_octave"}):
        assert groups_of(Room.from_scene(_box(TRUTH, **change))) is None


def test_calibration_recovers_the_absorption_it_measured():
    target = simulate(Room.from_scene(_box(TRUTH)))
    measured = {pair: response.parameters() for pair, response in target.responses.items()}
    # Floor and ceiling start in the right proportion, a third too high; the walls
    # nearly twice as high.
    factor = {"floor": 1.3, "ceiling": 1.3, "walls": 1.9}
    start = {name: np.multiply(values, factor[name]) for name, values in TRUTH.items()}
    found = calibrate(_box(start), measured, rounds=3)

    assert found.materials == MATERIALS
    np.testing.assert_allclose(found.initial, [start[m] for m in MATERIALS])
    for name in found.parameters:
        assert np.all(found.errors("before")[name] > 1.5)
        assert np.all(found.errors("after")[name] < 0.3)
    error = found.absorption - np.array([TRUTH[m] for m in MATERIALS])
    assert np.all(np.abs(error) < 0.05)
    assert np.all(np.abs(error) <= 2.0 * found.uncertainty + 0.02)
    # The measurements tell the walls, three fifths of the surface, from the rest well
    # from 500 Hz up. Floor and ceiling are much alike to the sound, the more where they
    # absorb alike: at 63 Hz the measurements hardly tell them apart, and they keep their
    # proportion.
    assert np.all(np.abs(error[2]) < 0.03)
    assert np.all(np.abs(error[2, 3:]) < 0.015)
    assert np.all(found.uncertainty[2, 3:] < 0.05)
    assert np.all(found.uncertainty[:2, 0] > 0.1)

    fitted = found.fitted_scene()
    for k, name in enumerate(MATERIALS):
        np.testing.assert_allclose(fitted.materials[name].absorption, found.absorption[k])
        assert fitted.materials[name].source.startswith("absorption calibrated")
    report = found.as_dict()
    assert report["materials"]["walls"]["absorption"] == pytest.approx(found.absorption[2])
    assert set(report["pairs"]["s a"]) == {"measured", "before", "after"}
    assert "T30 after" in found.table()


def test_a_third_octave_material_keeps_its_shape():
    thirds = np.linspace(0.1, 0.56, 24)
    scene = _box(TRUTH)
    scene.materials["floor"] = scene.materials["floor"].model_copy(
        update={"absorption": list(thirds)}
    )
    octaves = thirds.reshape(-1, 3).mean(axis=1)
    out = with_absorption(scene, ["floor"], [octaves + 0.05])
    np.testing.assert_allclose(out.materials["floor"].absorption, thirds + 0.05, atol=1e-4)
    np.testing.assert_allclose(scene.materials["floor"].absorption, thirds)


@pytest.mark.parametrize(
    ("change", "materials", "message"),
    [
        ({"late": {"method": "statistical"}}, None, "stochastic ray tracing"),
        ({"bands": "third_octave"}, None, "octave bands"),
        ({"wave": {"enabled": True}}, None, "wave solver"),
        ({}, ["carpet"], "no surface has the material carpet"),
        ({}, ["floor", "floor"], "named once"),
    ],
)
def test_calibration_says_what_it_cannot_do(change, materials, message):
    measured = {("s", "a"): simulate(Room.from_scene(_box(TRUTH, rays=200)))["s", "a"]}
    scene = _box(TRUTH, rays=200, **change)
    with pytest.raises(ValueError, match=message):
        calibrate(scene, {p: r.parameters() for p, r in measured.items()}, materials)


def test_calibration_checks_its_measurements():
    parameters = simulate(Room.from_scene(_box(TRUTH, rays=200)))["s", "a"].parameters()
    scene = _box(TRUTH, rays=200)
    with pytest.raises(ValueError, match="no source and listener for s/z"):
        calibrate(scene, {("s", "z"): parameters})
    with pytest.raises(ValueError, match="cannot fit spl"):
        calibrate(scene, {("s", "a"): parameters}, parameters=["spl"])
    with pytest.raises(ValueError, match="no measurements"):
        calibrate(scene, {})
    with pytest.raises(ValueError, match="at least one round"):
        calibrate(scene, {("s", "a"): parameters}, rounds=0)


def test_at_most_sixteen_materials_at_once():
    names = [f"m{k}" for k in range(17)]
    scene = Scene.model_validate(
        {
            "geometry": {"box": [8, 6, 3.5]},
            "materials": {name: {"absorption": [0.1] * 8} for name in names},
            "assignments": {"floor": "m0", "ceiling": "m1", "walls": "m2"},
            "sources": [{"name": "s", "position": [2.0, 2.6, 1.5]}],
            "listeners": [{"name": "a", "position": [5.8, 3.9, 1.2]}],
            "simulation": {"late": {"rays": 200}},
        }
    )
    parameters = simulate(Room.from_scene(scene))["s", "a"].parameters()
    with pytest.raises(ValueError, match="at most 16 materials"):
        calibrate(scene, {("s", "a"): parameters}, names)


def test_the_calibrate_command(tmp_path, capsys):
    target = simulate(Room.from_scene(_box(TRUTH, rays=500)))
    pairs = {f"{s} {lis}": r.parameters().as_dict() for (s, lis), r in target.responses.items()}
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    scene = tmp_path / "scene.yaml"
    _box({name: np.add(values, 0.05) for name, values in TRUTH.items()}).to_yaml(scene)

    fitted, report = tmp_path / "fitted.yaml", tmp_path / "calibration.json"
    argv = ["calibrate", str(scene), "--measured-json", str(measured), "--rays", "500"]
    argv += ["--materials", "floor", "walls", "--rounds", "1", "-o", str(fitted)]
    assert main([*argv, "--json", str(report)]) == 0
    out = capsys.readouterr().out
    assert "6 measured pairs, 1 rounds" in out
    assert "T30 before" in out

    data = json.loads(report.read_text(encoding="utf-8"))
    assert list(data["materials"]) == ["floor", "walls"]
    after = yaml.safe_load(fitted.read_text(encoding="utf-8"))["materials"]
    assert after["floor"]["absorption"] == pytest.approx(data["materials"]["floor"]["absorption"])
    assert after["ceiling"]["absorption"] == pytest.approx(np.add(TRUTH["ceiling"], 0.05))
    assert Scene.from_yaml(fitted).simulation.late.rays == 1500  # the scene's own

    assert main(["calibrate", str(scene)]) == 1
    assert "give --measured or --measured-json" in capsys.readouterr().err
    assert main(["calibrate", str(scene), "--measured", "s", "a", str(tmp_path / "x.wav")]) == 1
