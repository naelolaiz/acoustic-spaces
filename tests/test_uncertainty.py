"""The spread of the results that the materials' uncertain absorption leads to."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine.reweight import groups_of, log_ratio, reweighted_result
from acoustic_spaces.engine.uncertainty import (
    ADAPTIVE,
    ADAPTIVE_BATCH,
    ADMITTANCE_STEPS,
    EDGE_ERRORS_DB,
    ISO_12999_2_OCTAVES,
    ISO_12999_2_THIRDS,
    MAX_ABSORPTION,
    MIN_BATCHES,
    PRESET_TOLERANCE,
    _Admittances,
    _draws,
    _limits,
    _Tally,
    absorption_draws,
    assumed,
    edge_errors,
    grounds_left_out,
    held,
    material_spread,
    numerical_tolerance,
    octave_reproducibility,
    spread_groups,
    uncertainty_of,
)
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

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
    # By value: a copy with other absorption (a calibration's fitted scene) is not, as
    # it is not once saved and read again.
    from acoustic_spaces.calibration import with_absorption

    fitted = with_absorption(scene, ["walls"], [[0.3] * 8])
    assert not fitted.materials["walls"].builtin and uncertainty_of(fitted, "walls") is None
    assert not Scene.model_validate(fitted.to_data()).materials["walls"].builtin
    assert not walls.model_copy(update={"preset": None}).builtin
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


def test_octave_bands_take_iso_12999_2_s_octaves_else_the_mean_of_its_thirds():
    """sigma_R = m alpha + n (ISO 12999-2:2020) in an octave band: from 250 Hz to 4 kHz
    the standard's octave band's (Table 2), elsewhere the mean of its third octaves' in
    Table 1: the 63 Hz octave has its 63 and 80 Hz rows, the 8 kHz octave, past the
    table, takes the 5 kHz row."""
    table = ISO_12999_2_THIRDS
    assert len(table) == 20 and min(table) == 63 and max(table) == 5000
    assert sorted(ISO_12999_2_OCTAVES) == [250, 500, 1000, 2000, 4000]
    # At alpha = 1, sigma_R from 500 Hz to 5 kHz in Table 1 (Vercammen, ICA 2019, Table 2,
    # has 0.057 at 2.5 kHz).
    found = [round(m + n, 3) for f, (m, n) in sorted(table.items()) if f >= 500]
    assert found == [0.065, 0.06, 0.055, 0.055, 0.056, 0.055, 0.056, 0.056, 0.062, 0.07, 0.086]
    alpha = np.linspace(0.1, 0.8, 8)
    sigma = octave_reproducibility(alpha)
    thirds = {0: (63, 80), 1: (100, 125, 160), 7: (5000,)}
    for band, rows in thirds.items():
        expected = np.mean([table[f][0] * alpha[band] + table[f][1] for f in rows])
        assert sigma[band] == pytest.approx(expected, rel=1e-12)
    for band in range(2, 7):
        m, n = ISO_12999_2_OCTAVES[OCTAVE_BANDS_HZ[band]]
        assert sigma[band] == pytest.approx(m * alpha[band] + n, rel=1e-12)


def test_iso_12999_2_tables_give_the_standard_s_examples():
    """The expanded uncertainty U = 2 sigma_R of the standard's examples, to 0.01: a
    third-octave absorption (ISO 12999-2:2020, Table 4) and a practical one (Table 5)."""
    example = {  # band (Hz): (alpha_s, U)
        63: (0.33, 0.33), 80: (0.35, 0.26), 100: (0.39, 0.22), 125: (0.38, 0.17),
        160: (0.37, 0.13), 200: (0.36, 0.11), 250: (0.36, 0.09), 315: (0.36, 0.08),
        400: (0.43, 0.08), 500: (0.49, 0.08), 630: (0.58, 0.08), 800: (0.63, 0.08),
        1000: (0.68, 0.08), 1250: (0.71, 0.09), 1600: (0.73, 0.09), 2000: (0.75, 0.09),
        2500: (0.77, 0.10), 3150: (0.79, 0.11), 4000: (0.81, 0.13), 5000: (0.81, 0.16),
    }  # fmt: skip
    for band, (alpha, expanded) in example.items():
        m, n = ISO_12999_2_THIRDS[band]
        assert round(2.0 * (m * alpha + n), 2) == pytest.approx(expanded), band
    practical = {250: (0.50, 0.09), 500: (0.65, 0.08), 1000: (0.70, 0.08),
                 2000: (0.85, 0.08), 4000: (0.80, 0.10)}  # fmt: skip
    for band, (alpha, expanded) in practical.items():
        m, n = ISO_12999_2_OCTAVES[band]
        assert round(2.0 * (m * alpha + n), 2) == pytest.approx(expanded), band


def test_the_phases_between_fits_turn_as_a_fit_for_each_draw_does():
    """The paths that add in pressure turn their phases with each draw's absorption: the
    admittance taken between fits at nine absorptions gives the levels that a fit for
    each draw gives, within the fits' own scatter, at a few hundredths of the time; and
    so does the one between fits every quarter of the tolerance from 0, which the draws
    that come batch by batch take."""
    scene = _box({"absorption": 0.1, "distribution": "uniform"})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    response = result["s", "a"]
    assert response.early.spectrum is not None  # the lowest bands add in pressure
    coherent = response.early.spectrum.shape[1]
    traced = np.array([scene.materials[n].coefficients("octave")[0] for n in materials])
    taken, deviates, widths = _draws(scene, materials, 40, 5)
    limits = [_limits(scene.materials[n], own) for n, own in zip(materials, traced, strict=True)]
    between = _Admittances(traced, widths, limits)
    between.over(taken, deviates)
    stepped = _Admittances(traced, widths, limits)
    stepped.along(taken, deviates, [ADMITTANCE_STEPS["uniform"]] * len(materials))
    for drawn in taken[::8]:
        ratio = log_ratio(traced, drawn)
        fitted = response.reweighted(face_group, ratio, absorption=(traced, drawn))
        for admittance_of in (between, stepped):
            taken_between = response.reweighted(
                face_group, ratio, absorption=(traced, drawn), admittance_of=admittance_of
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


def test_materials_are_drawn_apart_and_not_below_what_they_let_through():
    data = _box({"absorption": 0.05}).to_data()
    data["materials"]["ceiling"]["uncertainty"] = {"absorption": 0.03}
    # A door of 6 dB: a quarter of the sound passes, and counts as absorbed.
    data["materials"]["floor"] = {
        "absorption": [0.3] * 8,
        "transmission_loss": 6.0,
        "uncertainty": {"absorption": 0.1},
    }
    scene = Scene.model_validate(data)
    taken = absorption_draws(scene, ["floor", "ceiling", "walls"], 4000, seed=1)
    passing = scene.materials["floor"].transmission_coefficients("octave")
    assert 0.25 < passing[0] < 0.26 and taken[:, 0].min() == passing[0]
    assert np.mean(taken[:, 0, 0] == passing[0]) > 0.2  # 0.3 is half a u above it
    # Each material from a deviate of its own.
    ceiling, walls = taken[:, 1, 7] - 0.15, taken[:, 2, 7] - WALLS[7]
    assert abs(np.corrcoef(ceiling, walls)[0, 1]) < 0.05
    assert abs(ceiling.std() - 0.03) < 0.002 and abs(walls.std() - 0.05) < 0.003


def test_a_ground_s_absorption_uncertainty_is_left_out(tmp_path, capsys):
    """The impedance gives a ground's reflection, which re-weighting by 1 - a does not
    follow: an absorption uncertainty beside it is left out, and said to be."""
    import yaml

    from acoustic_spaces.cli import main

    data = _box({"absorption": 0.02}).to_data()
    data["materials"]["floor"] = {"impedance": "grass", "uncertainty": {"absorption": 0.05}}
    scene = Scene.model_validate(data)
    room = Room.from_scene(scene)
    assert scene.materials["floor"].uncertainty is not None
    assert uncertainty_of(scene, "floor") is None and grounds_left_out(room) == ["floor"]
    taken = absorption_draws(scene, ["floor", "walls"], 50, seed=1)
    np.testing.assert_array_equal(taken[:, 0], np.broadcast_to(taken[0, 0], (50, 8)))
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data))
    args = ["simulate", str(path), "--out", str(tmp_path / "out"), "--rays", "300"]
    assert main([*args, "--spread-draws", "4"]) == 0
    assert (
        "warning: the absorption uncertainty of floor is left out of the spread: a ground's "
        "impedance gives its reflection"
    ) in capsys.readouterr().err


def test_the_spread_of_the_parameters_follows_the_absorption_s():
    scene = _box({"absorption": 0.02})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    spread = material_spread(result, scene, face_group, materials, draws=60, seed=3, rays=False)
    found = spread["s", "a"]
    assert found.draws == 60 and found.coverage == 0.95 and found.batches == 0
    assert found.one_sided == 0.95
    assert np.all(np.isnan(found.rays.t30))
    np.testing.assert_array_equal(found.standard.t30, found.absorption.t30)
    traced = result["s", "a"].parameters()
    bands = slice(1, 7)
    # The draws centre on the traced values and spread out from them.
    np.testing.assert_allclose(found.mean.t30[bands], traced.t30[bands], rtol=0.03)
    assert np.all(found.standard.t30[bands] > 0.0)
    assert np.all(found.low.t30[bands] < traced.t30[bands])
    assert np.all(found.high.t30[bands] > traced.t30[bands])
    # The one-sided bounds of 95 % lie within the 95 % interval, which leaves 2.5 % out
    # at each end, and those of 90 % within them; nothing else changes with them.
    assert np.all(found.low.t30[bands] < found.at_least.t30[bands])
    assert np.all(found.at_least.t30[bands] < traced.t30[bands])
    assert np.all(traced.t30[bands] < found.at_most.t30[bands])
    assert np.all(found.at_most.t30[bands] < found.high.t30[bands])
    ninety = material_spread(
        result, scene, face_group, materials, draws=60, seed=3, rays=False, one_sided=0.9
    )["s", "a"]
    assert ninety.one_sided == 0.9
    assert np.all(ninety.at_least.t30[bands] > found.at_least.t30[bands])
    assert np.all(ninety.at_most.t30[bands] < found.at_most.t30[bands])
    for part in ("mean", "standard", "low", "high", "absorption"):
        np.testing.assert_array_equal(
            getattr(ninety, part).t30[bands], getattr(found, part).t30[bands]
        )
    for bad in (0.5, 1.0, 0.3):
        with pytest.raises(ValueError, match="one-sided"):
            material_spread(result, scene, face_group, materials, draws=10, one_sided=bad)
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

    settings = scene.simulation
    assert edge_errors(response([-1, -1], 2), settings) == {}
    assert edge_errors(response([-1, -2], 2), settings) == {}  # over the terrain's crests
    found = edge_errors(response([-1, 4], 2), settings)
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
    found = edge_errors(response([7], 8), wide.simulation)
    assert list(found) == ["one edge", "two edges", "corners of edges"]
    np.testing.assert_array_equal(found["two edges"], EDGE_ERRORS_DB["two edges"]["pressure"])


def test_the_errors_over_edges_are_those_of_the_run_s_settings():
    """A run made with other settings than the scene's (``simulate --diffraction``)
    lists the errors over edges of the settings it was made with."""
    data = _box({"absorption": 0.02}).to_data()
    # An L-shaped room, the listener round its corner from the source.
    data["geometry"] = {
        "floor_plan": [[0, 0], [8, 0], [8, 4], [4, 4], [4, 8], [0, 8]],
        "height": 3,
    }
    data["assignments"] = {"floor": "floor", "ceiling": "ceiling", "default": "walls"}
    data["sources"] = [{"name": "s", "position": [7.0, 2.0, 1.5]}]
    data["listeners"] = [{"name": "a", "position": [2.0, 7.0, 1.2]}]
    scene = Scene.model_validate(data)
    assert scene.simulation.early.diffraction == "first_order"
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, groups=face_group)
    assert np.any(result["s", "a"].early.edge >= 0)
    spread = material_spread(result, scene, face_group, materials, draws=4, rays=False)
    assert list(spread["s", "a"].not_counted) == ["one edge", "one side", "corners of edges"]
    settings = scene.simulation.model_copy(deep=True)
    settings.early.diffraction = "second_order"
    settings.early.diffraction_reflections = "both_sides"
    result = simulate(room, settings=settings, groups=face_group)
    spread = material_spread(result, scene, face_group, materials, draws=4, rays=False)
    assert list(spread["s", "a"].not_counted) == ["one edge", "two edges", "corners of edges"]


#: A ceiling that absorbs everything at 500 Hz and 1 kHz, as five of the built-in
#: materials do in some band.
FULL = [0.45, 0.6, 0.9, 1.0, 1.0, 0.95, 0.9, 0.9]


def _full_ceiling(ceiling=FULL, rays: int = 500) -> Scene:
    data = _box(None).to_data()
    data["materials"]["ceiling"] = {
        "absorption": list(ceiling),
        "uncertainty": {"absorption": 0.05, "source": "lab"},
    }
    data["simulation"]["late"]["rays"] = rays
    return Scene.model_validate(data)


def test_a_band_that_absorbs_everything_is_held_and_the_others_are_drawn():
    scene = _full_ceiling()
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    assert "ceiling" in materials and held(scene, materials) == {"ceiling": [500, 1000]}
    groups, why, own = spread_groups(room)
    assert groups[1] == materials and why is None and own
    taken = absorption_draws(scene, materials, 2000, seed=1)[:, materials.index("ceiling")]
    np.testing.assert_array_equal(taken[:, 3:5], 1.0)
    assert np.all(taken[:, :3].std(axis=0) > 0.04)
    assert taken[:, 5].max() == MAX_ABSORPTION  # 0.95: one u from the top
    before = np.array([scene.materials[m].coefficients("octave")[0] for m in materials])
    np.testing.assert_array_equal(log_ratio(before, before), 0.0)
    for wrong in (0.9, 1.0):  # from everything, and to it
        after = before.copy()
        after[materials.index("ceiling"), 3 if wrong < 1.0 else 0] = wrong
        with pytest.raises(ValueError, match="no energy to re-weight"):
            log_ratio(before, after)

    result = simulate(room, sources=["s"], groups=face_group)
    spread = material_spread(result, scene, face_group, materials, draws=40, seed=3, rays=False)
    found = spread["s", "a"]
    assert found.held == {"ceiling": [500, 1000]} and found.as_dict()["held"] == found.held
    # Only the ceiling is uncertain: where it is held nothing moves.
    np.testing.assert_allclose(found.absorption.g[3:5], 0.0, atol=1e-9)
    assert np.all(found.absorption.g[[1, 2, 5, 6]] > 0.01)

    # A change in its other bands re-weights; one of a band that absorbs everything
    # does not, nor one to that.
    def changed(band: int, value: float) -> Scene:
        data = scene.to_data()
        data["materials"]["ceiling"]["absorption"][band] = value
        return Scene.model_validate(data)

    assert reweighted_result(result, scene, face_group, materials, changed(1, 0.7)) is not None
    assert reweighted_result(result, scene, face_group, materials, changed(3, 0.9)) is None
    assert reweighted_result(result, scene, face_group, materials, changed(1, 1.0)) is None


def test_the_bands_beside_one_that_absorbs_everything_re_weight_as_tracing_again():
    old, new = np.array(FULL), np.array([0.5, 0.65, 0.95, 1.0, 1.0, 0.99, 0.95, 0.95])
    scene = _full_ceiling(rays=3000)
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    before = np.array([scene.materials[m].coefficients("octave")[0] for m in materials])
    after = before.copy()
    after[materials.index("ceiling")] = new
    guess = result["s", "a"].reweighted(
        face_group, log_ratio(before, after), absorption=(before, after)
    )
    again = simulate(Room.from_scene(_full_ceiling(new, rays=3000)), sources=["s"])["s", "a"]
    found, expected, traced = guess.parameters(), again.parameters(), result["s", "a"].parameters()
    bands = [1, 2, 5, 6]
    assert np.all(np.abs(old - new)[bands] >= 0.04)
    np.testing.assert_allclose(found.g[bands], expected.g[bands], atol=0.15)
    np.testing.assert_allclose(found.t30[bands], expected.t30[bands], rtol=0.1)
    assert np.all(found.g[bands] < traced.g[bands])
    # The bands that absorb everything stay as traced.
    np.testing.assert_array_equal(found.g[3:5], traced.g[3:5])


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
        np.testing.assert_allclose(getattr(both.standard, name)[bands], combined[bands], rtol=1e-12)
        # The rays' term has no mean: the mean is that of the absorption's draws.
        np.testing.assert_array_equal(getattr(both.mean, name), getattr(alone.mean, name))
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


def test_the_rays_deviates_are_student_s_t_and_the_interval_the_draws_quantiles():
    from acoustic_spaces.parameters import RoomParameters

    names = [f.name for f in dataclasses.fields(RoomParameters)]
    same = RoomParameters(**{name: np.full(8, 5.0) for name in names})
    errors = RoomParameters(**{name: np.ones(8) for name in names})
    tally = _Tally(errors, 10, 0.95, 0.95)
    tally.add([same] * 20000, np.random.default_rng(4))
    spread = tally.spread(None)
    # Student's t of 9 degrees of freedom: wider than a normal deviate, whose standard
    # deviation is 1 and whose 97.5 % point 1.96.
    np.testing.assert_allclose(spread.standard.g, np.sqrt(9.0 / 7.0), rtol=1e-12)
    np.testing.assert_allclose(spread.high.g - 5.0, 2.262, rtol=0.04)
    np.testing.assert_allclose(5.0 - spread.at_least.g, 1.833, rtol=0.04)
    np.testing.assert_array_equal(spread.absorption.g, 0.0)
    np.testing.assert_array_equal(spread.mean.g, 5.0)

    # Without the rays' error the interval's ends are the quantiles of the draws'
    # parameters, of the coverage asked for.
    scene = _box({"absorption": 0.02})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    traced = np.array([scene.materials[m].coefficients("octave")[0] for m in materials])
    drawn = absorption_draws(scene, materials, 40, seed=3)
    g = np.array(
        [
            result["s", "a"]
            .reweighted(face_group, log_ratio(traced, row), absorption=(traced, row))
            .parameters()
            .g
            for row in drawn
        ]
    )
    above = slice(2, 7)  # where the paths add in energy: no phases between fits
    for coverage, tails in ((0.95, (0.025, 0.975)), (0.5, (0.25, 0.75))):
        found = material_spread(
            result, scene, face_group, materials, draws=40, seed=3, rays=False, coverage=coverage
        )["s", "a"]
        assert found.coverage == coverage
        np.testing.assert_allclose(
            found.low.g[above], np.quantile(g, tails[0], axis=0)[above], rtol=1e-9
        )
        np.testing.assert_allclose(
            found.high.g[above], np.quantile(g, tails[1], axis=0)[above], rtol=1e-9
        )
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="coverage"):
            material_spread(result, scene, face_group, materials, draws=10, coverage=bad)

    # Four batches are the fewest that count: Student's t of 3 degrees of freedom.
    assert MIN_BATCHES == 4
    four = scene.model_copy(deep=True)
    four.simulation.late.batches = 4
    result = simulate(Room.from_scene(four), sources=["s"], groups=face_group)
    spread = material_spread(result, four, face_group, materials, draws=10)["s", "a"]
    errors = result["s", "a"].standard_errors()
    assert spread.batches == 4
    np.testing.assert_allclose(spread.rays.g[1:7], errors.g[1:7] * np.sqrt(3.0), rtol=1e-12)


def test_the_combined_uncertainty_is_steady_with_the_fewest_batches():
    """With 4 batches the rays' deviates are Student's t of 3 degrees of freedom, whose
    sample standard deviation does not settle (no fourth moment): 400 draws of them are
    off by more than a tenth in four trials of ten. The combined standard uncertainty
    takes the rays' part in closed form, so it is the same whatever the deviates."""
    from acoustic_spaces.parameters import RoomParameters

    names = [f.name for f in dataclasses.fields(RoomParameters)]
    same = RoomParameters(**{name: np.full(8, 5.0) for name in names})
    errors = RoomParameters(**{name: np.ones(8) for name in names})
    off = 0
    for seed in range(40):
        tally = _Tally(errors, 4, 0.95, 0.95)
        tally.add([same] * 400, np.random.default_rng(seed))
        spread = tally.spread(None)
        np.testing.assert_allclose(spread.standard.g, np.sqrt(3.0), rtol=1e-12)
        np.testing.assert_array_equal(spread.rays.g, spread.standard.g)
        sampled = np.random.default_rng(seed).standard_t(3, size=400).std(ddof=1)
        off += abs(sampled / np.sqrt(3.0) - 1.0) > 0.1
    assert off >= 8


def test_an_interval_stays_within_what_the_parameter_can_be():
    """Where the rays' error is large against a value near its limit, the draws with
    the rays' deviates pass the limit: a decay time below 0, a D50 above 1. The
    interval's ends and the one-sided bounds are held at the limit; a level has none."""
    from acoustic_spaces.parameters import ALL_PARAMETERS, LIMITS, Levels, RoomParameters

    speech = {"sti", "stipa"}
    assert speech < {f.name for f in dataclasses.fields(Levels)}
    assert set(LIMITS) - speech < set(ALL_PARAMETERS)
    assert all(ALL_PARAMETERS[name][0] != "dB" for name in set(LIMITS) - speech)
    names = [f.name for f in dataclasses.fields(RoomParameters)]
    values = {name: np.full(8, 0.05) for name in names} | {"d50": np.full(8, 0.97)}
    errors = RoomParameters(**{name: np.full(8, 0.1) for name in names})
    tally = _Tally(errors, 10, 0.95, 0.95)
    tally.add([RoomParameters(**values)] * 2000, np.random.default_rng(1))
    spread = tally.spread(None)
    for name in ("t30", "edt", "ts"):
        np.testing.assert_array_equal(getattr(spread.low, name), 0.0)
        np.testing.assert_array_equal(getattr(spread.at_least, name), 0.0)
        assert np.all(getattr(spread.high, name) > 0.2)
    np.testing.assert_array_equal(spread.high.d50, 1.0)
    np.testing.assert_array_equal(spread.at_most.d50, 1.0)
    assert np.all(spread.low.d50 < 0.8) and np.all(spread.low.d50 > 0.0)
    # A level has no limit, and the mean and the standard uncertainty are not held.
    assert np.all(spread.low.g < -0.1) and np.all(spread.at_least.g < -0.1)
    np.testing.assert_allclose(spread.mean.t30, 0.05, rtol=1e-12)
    np.testing.assert_allclose(spread.standard.t30, 0.1 * np.sqrt(9.0 / 7.0), rtol=1e-12)


def _speech_box() -> Scene:
    """The box with a talker of 70 dB, and an uncertainty on its walls and its ceiling."""
    data = _box({"absorption": 0.02}).to_data()
    data["materials"]["ceiling"]["uncertainty"] = {"absorption": 0.03}
    data["sources"][0].update(level_db=70.0, spectrum="speech_male")
    return Scene.model_validate(data)


def test_a_response_gives_its_levels_and_speech_results_with_their_errors():
    """levels_and_speech holds what a report gives one by one, and level_errors their
    standard errors from the batches of rays: a band's level has its G's, and U50
    without noise C50's."""
    from acoustic_spaces.parameters import a_weighted

    scene = _speech_box()
    response = simulate(Room.from_scene(scene), sources=["s"])["s", "a"]
    power, noise, talker = scene.level_inputs("s")
    assert noise is None and talker == "male"
    np.testing.assert_array_equal(power, scene.sources[0].power_db("octave"))
    assert scene.level_inputs("another") == (94.0, None, "male")
    heard = response.levels_and_speech(power, noise, talker)
    spl = response.levels_or_held(power)
    speech = response.speech(power, noise, talker=talker)
    np.testing.assert_array_equal(heard.spl, spl)
    assert heard.spl_a == a_weighted(spl)
    assert (heard.sti, heard.stipa) == (speech.sti, speech.stipa)
    np.testing.assert_array_equal(heard.u50, speech.u50)
    # A room has no level relative to free field.
    assert np.all(np.isnan(heard.re_free_field)) and np.isnan(heard.re_free_field_a)
    plain = heard.as_dict()
    assert plain["spl_a_db"] == heard.spl_a and plain["sti"] == heard.sti
    assert plain["re_free_field_db"] == [None] * 8 and plain["re_free_field_a_db"] is None
    errors = response.level_errors(power, noise, talker)
    parameters = response.standard_errors()
    np.testing.assert_allclose(errors.spl, parameters.g, rtol=1e-9)
    np.testing.assert_allclose(errors.u50, parameters.c50, rtol=1e-6)
    # The bands' errors do not all add in the A-weighted level.
    assert 0.0 < errors.spl_a < errors.spl.max()
    assert 0.0 < errors.sti < 0.05 and 0.0 < errors.stipa < 0.05
    # Background noise lowers the STI and U50.
    noisy = response.levels_and_speech(power, 40.0, talker)
    assert noisy.sti < heard.sti and np.all(noisy.u50 < heard.u50)
    one = dataclasses.replace(response, late_batches=response.late_batches[:1])
    assert one.level_errors(power) is None and one.standard_errors() is None


def test_the_spread_covers_the_levels_and_speech_results():
    """The draws also give the spread of the levels and speech results: a band's level
    is its G moved by the source's power, U50 without noise is C50, and the A-weighted
    level and the STI are those of each draw's response, with the rays' error of their
    own from the batches."""
    scene = _speech_box()
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    response = result["s", "a"]
    found = material_spread(result, scene, face_group, materials, draws=60, seed=3)["s", "a"]
    levels = found.levels
    power, noise, talker = scene.level_inputs("s")
    moved = power + 10.0 * np.log10(1.0 / (400.0 * np.pi))
    for part in ("mean", "low", "high", "at_least", "at_most"):
        np.testing.assert_allclose(
            getattr(levels, part).spl, getattr(found, part).g + moved, rtol=1e-12
        )
        np.testing.assert_allclose(getattr(levels, part).u50, getattr(found, part).c50, atol=1e-9)
    for part in ("standard", "absorption", "rays"):
        np.testing.assert_allclose(getattr(levels, part).spl, getattr(found, part).g, rtol=1e-9)
    # Each draw's response again, with a fit of the admittance of its own where the
    # spread takes it between fits: the lowest bands, which weigh little in these.
    before = np.array([scene.materials[m].coefficients("octave")[0] for m in materials])
    drawn = []
    for after in absorption_draws(scene, materials, 60, 3):
        again = response.reweighted(
            face_group, log_ratio(before, after), absorption=(before, after)
        )
        drawn.append(again.levels_and_speech(power, noise, talker))
    errors = response.level_errors(power, noise, talker)
    heard = response.levels_and_speech(power, noise, talker)
    for name, close in (("spl_a", 0.01), ("sti", 0.001), ("stipa", 0.001)):
        values = np.array([getattr(v, name) for v in drawn])
        assert getattr(levels.mean, name) == pytest.approx(values.mean(), abs=close)
        assert getattr(levels.absorption, name) == pytest.approx(values.std(ddof=1), rel=0.02)
        assert getattr(levels.absorption, name) > 0.0
        # Ten batches: Student's t of 9 degrees of freedom.
        rays = getattr(errors, name) * np.sqrt(9.0 / 7.0)
        assert getattr(levels.rays, name) == pytest.approx(rays, rel=1e-12)
        combined = np.hypot(getattr(levels.absorption, name), rays)
        assert getattr(levels.standard, name) == pytest.approx(combined, rel=1e-12)
        assert getattr(levels.low, name) < getattr(heard, name) < getattr(levels.high, name)
        assert getattr(levels.low, name) < getattr(levels.at_least, name)
        assert getattr(levels.at_most, name) < getattr(levels.high, name)
    assert 0.0 < levels.low.sti and levels.high.sti < 1.0
    # A room has no level relative to free field, and the report holds None for it.
    assert np.all(np.isnan(levels.standard.re_free_field))
    plain = found.as_dict()["levels"]
    assert set(plain) == {"mean", "standard", "low", "high", "at_least", "at_most"} | {
        "absorption",
        "rays",
    }
    assert plain["standard"]["sti"] == levels.standard.sti
    assert plain["mean"]["spl_db"] == list(levels.mean.spl)
    assert plain["mean"]["re_free_field_a_db"] is None
    # Without the rays' error the absorption's is all of it.
    alone = material_spread(result, scene, face_group, materials, draws=60, seed=3, rays=False)[
        "s", "a"
    ].levels
    assert np.isnan(alone.rays.sti) and alone.standard.sti == alone.absorption.sti
    assert alone.absorption.sti == levels.absorption.sti and alone.mean.sti == levels.mean.sti


def test_the_levels_intervals_agree_with_the_parameters_and_an_sti_s_stays_in_range():
    """A band's level and its level relative to free field take the deviates of the
    rays' error that its G takes, U50 those of C50, and the A-weighted level relative
    to free field those of the A-weighted level, so their intervals are those moved;
    the room parameters' deviates are the same with the levels and without. An STI's
    interval is held between 0 and 1."""
    from acoustic_spaces.parameters import Levels, RoomParameters

    names = [f.name for f in dataclasses.fields(RoomParameters)]
    same = RoomParameters(**{name: np.full(8, 1.0) for name in names})
    errors = RoomParameters(**{name: np.full(8, 0.1) for name in names})
    bands = np.full(8, 60.0)
    heard = Levels(
        spl=bands,
        spl_a=60.0,
        sti=0.97,
        stipa=0.03,
        u50=bands,
        re_free_field=bands,
        re_free_field_a=1.0,
    )
    tenth = np.full(8, 0.1)
    level_errors = Levels(
        spl=tenth,
        spl_a=0.05,
        sti=0.05,
        stipa=0.05,
        u50=tenth,
        re_free_field=tenth,
        re_free_field_a=0.05,
    )
    tally = _Tally(errors, 10, 0.95, 0.95, level_errors)
    tally.add([same] * 2000, np.random.default_rng(1), [heard] * 2000)
    spread = tally.spread(None)
    levels = spread.levels
    for part in ("low", "high", "at_least", "at_most"):
        g, c50 = getattr(spread, part).g, getattr(spread, part).c50
        np.testing.assert_allclose(getattr(levels, part).spl - 60.0, g - 1.0, atol=1e-12)
        np.testing.assert_allclose(getattr(levels, part).re_free_field - 60.0, g - 1.0, atol=1e-12)
        np.testing.assert_allclose(getattr(levels, part).u50 - 60.0, c50 - 1.0, atol=1e-12)
        assert getattr(levels, part).re_free_field_a - 1.0 == pytest.approx(
            getattr(levels, part).spl_a - 60.0, abs=1e-12
        )
    without = _Tally(errors, 10, 0.95, 0.95)
    without.add([same] * 2000, np.random.default_rng(1))
    without = without.spread(None)
    assert without.levels is None and "levels" not in without.as_dict()
    for name in names:
        np.testing.assert_array_equal(getattr(without.low, name), getattr(spread.low, name))
    assert levels.high.sti == 1.0 and levels.at_most.sti == 1.0 and 0.8 < levels.low.sti < 0.97
    assert levels.low.stipa == 0.0 and levels.at_least.stipa == 0.0
    assert 0.03 < levels.high.stipa < 0.2
    # A level has no limit, and the mean and the standard uncertainty are not held.
    assert levels.low.spl_a < 59.95 and levels.high.spl_a > 60.05
    assert levels.mean.sti == pytest.approx(0.97, rel=1e-12)
    assert levels.standard.sti == pytest.approx(0.05 * np.sqrt(9.0 / 7.0), rel=1e-12)


def test_a_response_s_spread_does_not_depend_on_the_other_responses():
    data = _box({"absorption": 0.02}).to_data()
    data["listeners"].append({"name": "b", "position": [3.3, 1.2, 1.6]})
    scene = Scene.model_validate(data)
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)
    both = material_spread(result, scene, face_group, materials, draws=30, seed=3)
    alone = dataclasses.replace(result, responses={("s", "b"): result["s", "b"]})
    alone = material_spread(alone, scene, face_group, materials, draws=30, seed=3)
    for part in ("mean", "standard", "low", "high", "at_least", "at_most"):
        np.testing.assert_array_equal(
            getattr(both["s", "b"], part).t30, getattr(alone["s", "b"], part).t30
        )


def test_simulate_writes_the_spread_when_a_material_has_an_uncertainty(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    import json

    import yaml

    from acoustic_spaces.cli import _uncertainty_text, main

    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(_box({"absorption": 0.02, "source": "lab"}).to_data()))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300", "--spread-draws", "12"]
    assert main(args) == 0
    stdout = capsys.readouterr().out
    assert "uncertainty from the absorption and the rays (10 batches), 12 draws: at most" in stdout
    entry = json.loads((out / "parameters.json").read_text())["responses"][0]
    spread = entry["uncertainty"]
    assert spread["draws"] == 12 and spread["coverage"] == 0.95 and spread["batches"] == 10
    assert spread["absorption"]["t30"][3] > 0.0 and spread["rays"]["t30"][3] > 0.0
    # The combined standard uncertainty is the root sum of squares of its budget.
    assert spread["standard"]["t30"][3] == pytest.approx(
        np.hypot(spread["absorption"]["t30"][3], spread["rays"]["t30"][3]), rel=1e-12
    )
    # The line gives the budget of the result it names, where the rays' error counts.
    rays = {n: v for n, v in spread["standard"].items() if spread["rays"][n][3] is not None}
    standard = {n: rays.get(n, [None] * 8) for n in spread["standard"]}
    line = _uncertainty_text({**entry, "uncertainty": {**spread, "standard": standard}})
    assert "(absorption " in line and ", rays " in line
    assert spread["low"]["t30"][3] < entry["parameters"]["t30"][3] < spread["high"]["t30"][3]
    assert spread["one_sided"] == 0.95
    assert spread["low"]["t30"][3] <= spread["at_least"]["t30"][3] < spread["mean"]["t30"][3]
    assert spread["mean"]["t30"][3] < spread["at_most"]["t30"][3] <= spread["high"]["t30"][3]
    assert ", 95 % at least " in stdout and " and 95 % at most " in stdout
    # The levels and speech results have the same parts, and the report's lines give
    # the A-weighted level's and the STI's.
    levels = spread["levels"]
    assert set(levels) == {"mean", "standard", "low", "high", "at_least", "at_most"} | {
        "absorption",
        "rays",
    }
    for name in ("spl_a_db", "sti"):
        assert levels["standard"][name] == pytest.approx(
            np.hypot(levels["absorption"][name], levels["rays"][name]), rel=1e-12
        )
        assert levels["low"][name] < levels["mean"][name] < levels["high"][name]
    assert levels["standard"]["spl_db"] == pytest.approx(spread["standard"]["g"], rel=1e-9)
    assert " dB(A) (u = " in stdout and " % at most " in stdout.split(" dB(A) (u = ")[1]
    assert "STI " in stdout and ", STIPA " in stdout
    assert ", 95 % at least " in stdout.split("STI ")[1].split(", STIPA ")[0]
    # Third-octave bands do not re-weight: the run goes on without the spread.
    assert main([*args, "--bands", "third_octave"]) == 0
    captured = capsys.readouterr()
    assert "warning: no spread" in captured.err
    assert " dB(A); per octave band" in captured.out and " (STIPA " in captured.out
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
    assert "no spread from the materials' absorption uncertainty: it needs octave" in captured.err
    assert "uncertainty" not in json.loads((out / "parameters.json").read_text())["responses"][0]
    data["simulation"]["uncertainty"] = {"presets": "none"}
    scene.write_text(yaml.safe_dump(data))
    assert main(args) == 0
    capsys.readouterr()
    assert "uncertainty" not in json.loads((out / "parameters.json").read_text())["responses"][0]


def test_simulate_says_why_there_is_no_spread_and_announces_none_it_does_not_draw(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    import json

    import yaml

    from acoustic_spaces.cli import main
    from acoustic_spaces.engine.late import MAX_GROUPS

    # The only material with an uncertainty is on no surface: no spread, none announced.
    data = _box(None).to_data()
    data["materials"]["spare"] = {"absorption": [0.2] * 8, "uncertainty": {"absorption": 0.05}}
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(data))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300", "--spread-draws", "12"]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "spread" not in captured.err and "uncertainty from" not in captured.out
    assert main([*args, "--bands", "third_octave"]) == 0
    assert "spread" not in capsys.readouterr().err

    # More materials on the surfaces than the tracer counts reflections off: said, with
    # the numbers, as a warning when a material gives an uncertainty of its own.
    data = _box({"absorption": 0.02}).to_data()
    walls = data["materials"].pop("walls")
    plain = {k: v for k, v in walls.items() if k != "uncertainty"}
    corners = [[5 + 4 * np.cos(a), 5 + 4 * np.sin(a)] for a in np.arange(16) * np.pi / 8]
    data["geometry"] = {
        "floor_plan": [[round(float(v), 3) for v in c] for c in corners],
        "height": 3,
    }
    sides = [f"wall_{k}" for k in range(1, 17)]
    data["assignments"] = {"floor": "floor", "ceiling": "ceiling"} | {w: w for w in sides}
    data["materials"] |= {side: plain for side in sides[1:]} | {sides[0]: walls}
    data["sources"][0]["position"] = [4.0, 4.5, 1.5]
    data["listeners"][0]["position"] = [6.5, 5.5, 1.2]
    room = Room.from_scene(Scene.model_validate(data))
    assert len(set(room.material_names)) > MAX_GROUPS == 16
    why = "at most 16 materials on the surfaces, and there are 18"
    assert spread_groups(room) == (None, why, True)
    scene.write_text(yaml.safe_dump(data))
    assert main(args) == 0
    captured = capsys.readouterr()
    assert f"warning: no spread from the materials' absorption uncertainty: it needs {why}" in (
        captured.err
    )
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
    assert "--spread-draws: 0 (no spread), 2 or more, or adaptive" in capsys.readouterr().err
    assert not other.exists()


def test_the_scene_gives_the_draws_and_the_command_line_overrides_them(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    """``simulation.uncertainty.draws`` is a number (0 for no spread, not 1) or adaptive,
    and ``--spread-draws`` takes its place."""
    import json

    import yaml

    from acoustic_spaces.cli import main
    from acoustic_spaces.scene.model import Uncertainty

    assert Uncertainty().draws == 400
    assert Uncertainty(draws="adaptive").draws == ADAPTIVE
    assert Uncertainty(draws=0).draws == 0
    for bad in (1, -3, "many"):
        with pytest.raises(ValueError):
            Uncertainty(draws=bad)
    data = _box({"absorption": 0.02}).to_data()
    data["simulation"]["uncertainty"] = {"draws": 14}
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(data))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300"]
    assert main(args) == 0
    assert ", 14 draws: at most" in capsys.readouterr().out
    spread = json.loads((out / "parameters.json").read_text())["responses"][0]["uncertainty"]
    # A number of draws given beforehand is not tested for being stable.
    assert spread["draws"] == 14 and spread["stable"] is None
    assert main([*args, "--spread-draws", "12"]) == 0
    assert ", 12 draws: at most" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*args, "--spread-draws", "many"])
    assert "give a number of draws, or adaptive" in capsys.readouterr().err


def test_the_one_sided_bounds_are_the_draws_quantiles():
    """The value a result is at least with probability p is the draws' quantile of
    1 - p, the one it is at most that of p: for a normal spread the mean less and plus
    1.645 standard uncertainties at 95 %, 1.282 at 90 %."""
    rng = np.random.default_rng(5)
    standard = np.array([0.02, 0.3, 1.5])
    found = _normal_parameters(rng, 20_000, standard, same=True)
    values = np.stack([p.t30 for p in found])
    for probability, factor in ((0.95, 1.645), (0.9, 1.282)):
        tally = _Tally(None, 0, 0.95, probability)
        tally.add(found, rng)
        spread = tally.spread(None)
        assert spread.one_sided == probability
        np.testing.assert_array_equal(
            spread.at_least.c80, np.quantile(values, 1.0 - probability, axis=0)
        )
        np.testing.assert_array_equal(spread.at_most.c80, np.quantile(values, probability, axis=0))
        np.testing.assert_allclose((spread.at_least.t30 - 10.0) / standard, -factor, atol=0.04)
        np.testing.assert_allclose((spread.at_most.t30 - 10.0) / standard, factor, atol=0.04)
        assert spread.as_dict()["one_sided"] == probability
        assert spread.as_dict()["at_most"]["t30"] == list(spread.at_most.t30)


def test_the_scene_gives_the_one_sided_probability_and_the_command_line_overrides_it(
    tmp_path, capsys: pytest.CaptureFixture[str]
):
    """``simulation.uncertainty.one_sided`` is the probability of the one-sided bounds,
    0.95 unless the scene says otherwise, and ``--one-sided`` takes its place."""
    import json

    import yaml

    from acoustic_spaces.cli import main
    from acoustic_spaces.scene.model import Uncertainty

    assert Uncertainty().one_sided == 0.95
    for bad in (0.5, 1.0, 95):
        with pytest.raises(ValueError):
            Uncertainty(one_sided=bad)
    data = _box({"absorption": 0.02}).to_data()
    data["simulation"]["uncertainty"] = {"draws": 14, "one_sided": 0.9}
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(data))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300"]
    assert main(args) == 0
    stdout = capsys.readouterr().out
    assert ", 90 % at least " in stdout and " and 90 % at most " in stdout
    ninety = json.loads((out / "parameters.json").read_text())["responses"][0]["uncertainty"]
    assert ninety["one_sided"] == 0.9
    assert main([*args, "--one-sided", "0.95"]) == 0
    assert " and 95 % at most " in capsys.readouterr().out
    spread = json.loads((out / "parameters.json").read_text())["responses"][0]["uncertainty"]
    # The same draws: bounds further out, the rest as it was.
    assert spread["one_sided"] == 0.95 and spread["standard"] == ninety["standard"]
    assert spread["at_most"]["t30"][3] > ninety["at_most"]["t30"][3]
    assert spread["at_least"]["t30"][3] < ninety["at_least"]["t30"][3]
    other = tmp_path / "other"
    assert main([*args[:3], str(other), *args[4:], "--one-sided", "0.4"]) == 1
    assert "error: --one-sided: Input should be greater than 0.5" in capsys.readouterr().err
    assert not other.exists()


def test_the_numerical_tolerance_is_half_a_unit_of_the_last_meaningful_digit():
    """JCGM 101 7.9.2: a standard uncertainty written as c 10^l, c a whole number of
    n_dig digits, has the numerical tolerance 10^l / 2. Its example: 0.000 35 g with two
    meaningful digits is 35 10^-5 g, so 0.000 005 g."""
    np.testing.assert_allclose(numerical_tolerance(0.00035, 2), 0.000005, rtol=1e-12)
    # One digit: 0.062 is 6 10^-2; 0.094 is 9 10^-2; 0.096 is 10 10^-2, that is 1 10^-1.
    one = numerical_tolerance(np.array([0.062, 0.094, 0.096, 0.25, 1.0, 12.0]))
    np.testing.assert_allclose(one, [0.005, 0.005, 0.05, 0.05, 0.5, 5.0], rtol=1e-12)
    # No spread: nothing to settle.
    assert np.all(np.isinf(numerical_tolerance(np.array([0.0, 1e-15]))))


def _normal_parameters(
    rng: np.random.Generator, draws: int, standard: np.ndarray, same: bool = False
):
    """``draws`` room parameters whose every value is normal about 10 with ``standard``
    (bands,): each parameter with draws of its own, or all with the ``same``."""
    import dataclasses

    from acoustic_spaces.parameters import RoomParameters

    names = [f.name for f in dataclasses.fields(RoomParameters)]
    shared = 10.0 + standard * rng.standard_normal((draws, standard.size))
    values = {
        name: shared if same else 10.0 + standard * rng.standard_normal((draws, standard.size))
        for name in names
    }
    return [RoomParameters(**{name: values[name][k] for name in names}) for k in range(draws)]


def test_the_adaptive_test_is_that_of_jcgm_101_7_9_4():
    """Over h batches of M draws: with s the standard deviation of the batches' means
    over the root of h, and likewise for their standard deviations and their interval's
    ends, the results are stable when 2 s is at most the numerical tolerance of the
    standard deviation of all h M values; and likewise here for the one-sided bounds.
    Worked out here again from the draws."""
    rng = np.random.default_rng(11)
    standard = np.array([0.012, 0.25, 0.7, 1.4, 0.0036, 0.09, 3.0, 0.5])
    tally = _Tally(None, 0, 0.95, 0.95)
    batches = []
    for _ in range(6):
        found = _normal_parameters(rng, ADAPTIVE_BATCH, standard, same=True)
        batches.append(np.stack([p.t30 for p in found]))
        tally.add(found, rng)
    assert tally.draws == 6 * ADAPTIVE_BATCH
    estimates = np.array(
        [
            [
                b.mean(axis=0),
                b.std(axis=0, ddof=1),
                *np.quantile(b, [0.025, 0.975, 0.05, 0.95], axis=0),
            ]
            for b in batches
        ]
    )
    twice = 2.0 * estimates.std(axis=0, ddof=1) / np.sqrt(6.0)
    everything = np.concatenate(batches)
    tolerance = numerical_tolerance(everything.std(axis=0, ddof=1))
    ratios = (twice / tolerance).max(axis=0)
    # 600 draws settle the bands whose standard deviation starts with a 1 (a wide
    # tolerance), not those that start with a 7 or a 9.
    assert ratios[0] < 1.0 and ratios[3] < 1.0 and ratios[2] > 1.0 and ratios[5] > 1.0
    # Every parameter has the same draws here, so the furthest of all is t30's furthest.
    found = tally.unstable()
    assert found is not None
    times, _name, band = found
    assert band == int(np.argmax(ratios))
    np.testing.assert_allclose(times, ratios[band], rtol=1e-9)
    # Only the bands that pass: stable.
    passing = _Tally(None, 0, 0.95, 0.95)
    for b in batches:
        columns = b[:, [0, 3]]
        passing.add(
            [_all_parameters(row) for row in columns],
            rng,
        )
    assert passing.unstable() is None


def _all_parameters(values: np.ndarray):
    """Room parameters that are all ``values``."""
    import dataclasses

    from acoustic_spaces.parameters import RoomParameters

    return RoomParameters(**{f.name: values for f in dataclasses.fields(RoomParameters)})


def test_adaptive_batches_stop_once_stable_near_what_a_normal_spread_needs():
    """For a normal spread the interval's ends settle last: an end's standard deviation
    is 2.67 u / sqrt(N), so 2 s is within a tolerance of u / 2 (a standard uncertainty
    that starts with a 1) from 114 draws and within u / 4 (one that starts with a 2)
    from 456. Batches of 100 stop near there, and what they then give is within the
    tolerance of the truth."""
    stops = []
    for seed in range(20):
        rng = np.random.default_rng([7, seed])
        standard = np.full(8, 0.2)
        tally = _Tally(None, 0, 0.95, 0.95)
        while True:
            tally.add(_normal_parameters(rng, ADAPTIVE_BATCH, standard), rng)
            if tally.draws >= 2 * ADAPTIVE_BATCH and tally.unstable() is None:
                break
            assert tally.draws < 5000
        stops.append(tally.draws)
        spread = tally.spread(True)
        assert spread.stable is True and spread.draws == tally.draws
        # Twice the tolerance (0.05): each is within one about 19 times in 20.
        np.testing.assert_allclose(spread.standard.t30, 0.2, atol=0.1)
        np.testing.assert_allclose(spread.low.t30, 10.0 - 1.96 * 0.2, atol=0.1)
        np.testing.assert_allclose(spread.high.t30, 10.0 + 1.96 * 0.2, atol=0.1)
    # 456 draws for one result alone; all the parameters and bands together need more.
    assert 400 <= np.median(stops) <= 1500, stops


def test_what_400_draws_leave_open():
    """What the docs and the editor's tip say of 400 draws, for a normal spread: twice
    the standard deviation of the standard uncertainty is about 7 % of it, and that of
    an end of the 95 % interval about a quarter (2 * 2.67 / sqrt(400))."""
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((4000, 400))
    deviations = draws.std(axis=1, ddof=1)
    ends = np.quantile(draws, 0.025, axis=1)
    assert abs(2.0 * deviations.std() - 0.0708) < 0.004
    assert abs(2.0 * ends.std() - 0.267) < 0.015


def test_two_small_batches_pass_by_chance_too_easily():
    """Why the first test waits for ``ADAPTIVE_MIN_BATCHES`` batches: two batches of 100
    pass the test about one time in six for one result whose standard uncertainty starts
    with a 2, which needs some 460 draws (114 * 4), and about one time in twenty for one
    that starts with a 3, which needs about a thousand (114 * 9)."""
    rng = np.random.default_rng(17)
    for lead, least, most in ((2.06, 0.10, 0.23), (3.06, 0.015, 0.095)):
        passed = 0
        for _ in range(300):
            tally = _Tally(None, 0, 0.95, 0.95)
            for _batch in range(2):
                values = lead * rng.standard_normal((ADAPTIVE_BATCH, 1))
                tally.add([_all_parameters(row) for row in values], rng)
            passed += tally.unstable() is None
        assert least < passed / 300 < most, (lead, passed)


def test_adaptive_draws_go_on_until_the_results_are_stable(monkeypatch):
    """``draws="adaptive"`` takes batches of 100, ten at least, until every result of
    every response passes the test, or the draws reach their limit, and says which."""
    from acoustic_spaces.engine import uncertainty

    scene = _box({"absorption": 0.02})
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = simulate(room, sources=["s"], groups=face_group)

    def spread(**options):
        return material_spread(result, scene, face_group, materials, **options)["s", "a"]

    assert uncertainty.ADAPTIVE_MIN_BATCHES == 10
    with pytest.raises(ValueError, match="1000 draws or more"):
        spread(draws=ADAPTIVE, max_draws=900)
    # For the test's speed, batches of 20 and the first test after 2 of them.
    monkeypatch.setattr(uncertainty, "ADAPTIVE_BATCH", 20)
    monkeypatch.setattr(uncertainty, "ADAPTIVE_MIN_BATCHES", 2)
    heard: list[str] = []
    short = spread(draws=ADAPTIVE, seed=3, max_draws=50, progress=heard.append)
    # Two batches fit under 50 draws; over a hundred results are not all stable by then.
    assert short.draws == 40 and short.stable is False and short.as_dict()["stable"] is False
    assert heard == [f"the spread: s -> a (1 of 1), draw {k}" for k in (1, 21)]
    again = spread(draws=ADAPTIVE, seed=3, max_draws=40)
    np.testing.assert_array_equal(again.standard.t30, short.standard.t30)
    # With a tolerance ten times as wide the results settle a hundred times sooner, no
    # sooner than the first test, which comes after ADAPTIVE_MIN_BATCHES batches.
    tolerance = uncertainty.numerical_tolerance
    monkeypatch.setattr(
        uncertainty, "numerical_tolerance", lambda u, digits=1: 10.0 * tolerance(u, digits)
    )
    monkeypatch.setattr(uncertainty, "ADAPTIVE_MIN_BATCHES", 3)
    wide = spread(draws=ADAPTIVE, seed=3, max_draws=2000)
    assert wide.stable is True and 60 <= wide.draws < 1000 and wide.draws % 20 == 0
    # A number of draws given beforehand is not tested.
    fixed = spread(draws=20, seed=3)
    assert fixed.stable is None and fixed.draws == 20


def test_simulate_draws_adaptively_and_warns_at_the_limit(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch
):
    """``--spread-draws adaptive``: the output says the draws it took and whether the
    results were stable, with a warning when they reached the limit first."""
    import json

    import yaml

    from acoustic_spaces.cli import main
    from acoustic_spaces.engine import uncertainty

    # For the test's speed, two batches of 20; a tolerance nothing meets.
    monkeypatch.setattr(uncertainty, "ADAPTIVE_BATCH", 20)
    monkeypatch.setattr(uncertainty, "ADAPTIVE_MAX_DRAWS", 40)
    monkeypatch.setattr(uncertainty, "ADAPTIVE_MIN_BATCHES", 2)
    tolerance = uncertainty.numerical_tolerance
    monkeypatch.setattr(
        uncertainty, "numerical_tolerance", lambda u, digits=1: 1e-9 * tolerance(u, digits)
    )
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(_box({"absorption": 0.02}).to_data()))
    out = tmp_path / "out"
    args = ["simulate", str(scene), "--out", str(out), "--rays", "300"]
    assert main([*args, "--spread-draws", "adaptive"]) == 0
    captured = capsys.readouterr()
    assert "draws until the results are stable, 40 at most" in captured.err
    assert "warning: the spread's results were not stable after 40 draws" in captured.err
    assert ", 40 draws (adaptive, not stable): at most" in captured.out
    spread = json.loads((out / "parameters.json").read_text())["responses"][0]["uncertainty"]
    assert spread["draws"] == 40 and spread["stable"] is False
