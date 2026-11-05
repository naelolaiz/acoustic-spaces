"""The spread of the results that the materials' uncertain absorption leads to."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine.reweight import groups_of, log_ratio
from acoustic_spaces.engine.uncertainty import (
    EDGE_ERRORS_DB,
    ISO_12999_2_THIRDS,
    MAX_ABSORPTION,
    MIN_BATCHES,
    PRESET_TOLERANCE,
    _Admittances,
    _draws,
    absorption_draws,
    assumed,
    edge_errors,
    material_spread,
    octave_reproducibility,
    uncertainty_of,
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


def _with_preset(presets: str | None = None) -> Scene:
    """The box with a built-in material on its walls, a preset with values of its own
    on its floor, and the ceiling's own uncertainty."""
    data = _box(None).to_data()
    data["materials"]["walls"] = {"preset": "plasterboard", "scattering": 0.3}
    data["materials"]["floor"] |= {"preset": "carpet_tufted_9.5mm"}
    data["materials"]["ceiling"]["uncertainty"] = {"absorption": 0.02}
    if presets is not None:
        data["simulation"]["uncertainty"] = {"presets": presets}
    return Scene.model_validate(data)


def test_a_built_in_material_of_unknown_source_takes_the_scene_s_default():
    """A material whose absorption is a built-in database entry's, which names no
    source, takes plus or minus 0.1 by default; a preset given values of its own, or a
    material with an uncertainty of its own, keeps what it gives."""
    scene = _with_preset()
    walls, floor, ceiling = (scene.materials[n] for n in ("walls", "floor", "ceiling"))
    assert walls.builtin and not floor.builtin and not ceiling.builtin
    spec = uncertainty_of(scene, "walls")
    assert spec.distribution == "uniform" and spec.absorption == [PRESET_TOLERANCE] * 8
    assert "unknown source" in spec.source
    assert uncertainty_of(scene, "floor") is None
    assert uncertainty_of(scene, "ceiling") is ceiling.uncertainty
    assert assumed(scene, ["walls", "floor", "ceiling"]) == {"walls": "tolerance"}
    # The preset's own values written out with it (a scene dumped whole) are still its.
    dumped = Scene.model_validate(scene.model_dump(exclude_none=True))
    assert dumped.materials["walls"].builtin and not dumped.materials["floor"].builtin
    assert scene.model_copy(deep=True).materials["walls"].builtin
    # Twice ISO 12999-2's reproducibility, as a standard uncertainty.
    iso = _with_preset("iso_12999_2")
    spec = uncertainty_of(iso, "walls")
    alpha = np.asarray(walls.coefficients("octave")[0])
    assert spec.distribution == "normal"
    np.testing.assert_allclose(spec.absorption, 2.0 * octave_reproducibility(alpha))
    assert assumed(iso, ["walls"]) == {"walls": "iso_12999_2"}
    none = _with_preset("none")
    assert uncertainty_of(none, "walls") is None and assumed(none, ["walls"]) == {}
    with pytest.raises(ValueError):
        _with_preset("half")


def test_octave_bands_take_the_mean_of_iso_12999_2_s_third_octaves():
    """sigma_R = m alpha + n (ISO 12999-2:2020, Table 1) in an octave band is the mean of
    its third octaves' in the table, which runs from 100 Hz to 5 kHz as ISO 354 measures:
    the 63 Hz and 8 kHz octaves, outside it, take its 100 Hz and 5 kHz rows."""
    table = ISO_12999_2_THIRDS
    assert len(table) == 18 and min(table) == 100 and max(table) == 5000
    # At alpha = 1, sigma_R from 500 Hz to 5 kHz as Vercammen (ICA 2019, Table 2) gives
    # the standard's: 0.065 0.06 0.055 0.055 0.056 0.055 0.056 (2.5 kHz: 0.056 here,
    # 0.057 there) 0.062 0.07 0.086.
    found = [round(m + n, 3) for f, (m, n) in sorted(table.items()) if f >= 500]
    assert found == [0.065, 0.06, 0.055, 0.055, 0.056, 0.055, 0.056, 0.056, 0.062, 0.07, 0.086]
    alpha = np.linspace(0.1, 0.8, 8)
    sigma = octave_reproducibility(alpha)
    rows = {
        0: (100,),
        1: (100, 125, 160),
        4: (800, 1000, 1250),
        6: (3150, 4000, 5000),
        7: (5000,),
    }
    for band, thirds in rows.items():
        expected = np.mean([table[f][0] * alpha[band] + table[f][1] for f in thirds])
        assert sigma[band] == pytest.approx(expected, rel=1e-12)


def test_the_phases_between_fits_turn_as_a_fit_for_each_draw_does():
    """The paths that add in pressure turn their phases with each draw's absorption: the
    admittance taken between fits at nine absorptions gives the levels that a fit for
    each draw gives, within the fits' own scatter, at a few hundredths of the time."""
    scene = _box({"absorption": 0.1, "distribution": "uniform"})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    response = result["s", "a"]
    assert response.early.spectrum is not None  # the lowest bands add in pressure
    coherent = response.early.spectrum.shape[1]
    traced = np.array([scene.materials[n].coefficients("octave")[0] for n in materials])
    taken, deviates, widths = _draws(scene, materials, 40, 5)
    between = _Admittances(traced, taken, deviates, widths)
    for drawn in taken[::8]:
        ratio = log_ratio(traced, drawn)
        fitted = response.reweighted(face_group, ratio, absorption=(traced, drawn))
        taken_between = response.reweighted(
            face_group, ratio, absorption=(traced, drawn), admittance_of=between
        )
        for name in ("g", "c80"):
            a = np.asarray(getattr(fitted.parameters(), name))[:coherent]
            b = np.asarray(getattr(taken_between.parameters(), name))[:coherent]
            np.testing.assert_allclose(b, a, atol=0.2)
        # The bands above add in energy: no phase, the same levels.
        np.testing.assert_array_equal(fitted.early.energy, taken_between.early.energy)


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


def test_simulate_spreads_a_scene_of_built_in_materials_by_default(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    """A scene of built-in materials gets the spread of their default tolerance, which
    the output names; with presets: none it has none, and a run that cannot re-weight
    goes on without it and without a warning."""
    import json

    import yaml

    from acoustic_spaces.cli import main

    data = _with_preset().to_data()
    del data["materials"]["ceiling"]["uncertainty"]
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(data))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300", "--spread-draws", "12"]
    assert main(args) == 0
    stdout = capsys.readouterr().out
    assert (
        "built-in materials of unknown source taken at plus or minus 0.1 "
        "(simulation.uncertainty.presets): walls"
    ) in stdout
    spread = json.loads((out / "parameters.json").read_text())["responses"][0]["uncertainty"]
    assert spread["assumed"] == {"walls": "tolerance"} and spread["draws"] == 12
    assert main([*args, "--bands", "third_octave"]) == 0
    captured = capsys.readouterr()
    assert "warning" not in captured.err
    assert "uncertainty" not in json.loads((out / "parameters.json").read_text())["responses"][0]
    data["simulation"]["uncertainty"] = {"presets": "none"}
    scene.write_text(yaml.safe_dump(data))
    assert main(args) == 0
    capsys.readouterr()
    assert "uncertainty" not in json.loads((out / "parameters.json").read_text())["responses"][0]


def test_the_spread_reports_its_draws_and_stops_when_its_progress_raises():
    """``progress`` hears of each response's draws as they go (the editor's status bar),
    and what it raises (the editor's Cancel) stops the draws."""
    scene = _box({"absorption": 0.02})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    heard: list[str] = []
    material_spread(result, scene, face_group, materials, draws=60, progress=heard.append)
    assert heard == [f"the spread: s -> a (1 of 1), draw {k} of 60" for k in (1, 26, 51)]

    class Stop(Exception):
        pass

    def stop(message: str) -> None:
        if "draw 26" in message:
            raise Stop

    with pytest.raises(Stop):
        material_spread(result, scene, face_group, materials, draws=60, progress=stop)


def test_simulate_spread_draws_0_leaves_the_spread_out(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    """``--spread-draws 0`` simulates without the spread (and so without face groups,
    which keep the late part off the GPU); 1 is refused before the run."""
    import json

    import yaml

    from acoustic_spaces.cli import main

    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(_box({"absorption": 0.02, "source": "lab"}).to_data()))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300"]
    assert main([*args, "--spread-draws", "0"]) == 0
    captured = capsys.readouterr()
    assert "spread" not in captured.err and "uncertainty from" not in captured.out
    assert "uncertainty" not in json.loads((out / "parameters.json").read_text())["responses"][0]
    other = tmp_path / "other"
    assert main([*args[:3], str(other), *args[4:], "--spread-draws", "1"]) == 1
    assert "--spread-draws: 0 (no spread) or 2 or more" in capsys.readouterr().err
    assert not other.exists()
