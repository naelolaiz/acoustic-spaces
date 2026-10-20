"""Outdoor results: the level relative to free field, and the insertion loss of
buildings, barriers and vegetation."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.cli import main
from acoustic_spaces.engine import insertion
from acoustic_spaces.engine.coverage import (
    INSERTION_LOSS_MAP_PARAMETERS,
    ROOM_MAP_PARAMETERS,
    coverage_map,
)
from acoustic_spaces.engine.model import free_field_energy
from acoustic_spaces.engine.reweight import groups_of, reweighted_result
from acoustic_spaces.materials.bands import band_centres
from acoustic_spaces.materials.ground import spherical_reflection, surface_impedance
from acoustic_spaces.parameters import a_weighted
from acoustic_spaces.results import write_results

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
C = 343.0
WALL = {"name": "wall", "path": [[10, -30], [10, 30]], "height": 3.0}
SHED = {"name": "shed", "footprint": [[20, 20], [26, 20], [26, 26], [20, 26]], "height": 3.0}


def field(
    ground=None,
    *,
    barriers=(),
    buildings=(),
    listeners=((20, 0, 1.5),),
    coverage=None,
    **simulation,
):
    """A source 1 m over a ground, listeners and, if given, barriers and buildings."""
    geometry = {"ground": {"extent": [[-40, -40], [40, 40]], "sky": 20}}
    if barriers:
        geometry["barriers"] = list(barriers)
    if buildings:
        geometry["buildings"] = list(buildings)
    data = {
        "geometry": geometry,
        "materials": {
            "field": ground or {"impedance": "grass", "scattering": 0.0},
            "hard": {"absorption": [0.05] * 8, "scattering": 0.0},
        },
        "default_material": "field",
        "assignments": {"wall": "hard", "shed": "hard"},
        "sources": [{"name": "s", "position": [0, 0, 1.0]}],
        "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
        "simulation": {
            "ir_length_s": 0.3,
            "seed": 2,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "direct"},
            "late": {"rays": 1000},
            **simulation,
        },
    }
    if coverage is not None:
        data["coverage"] = coverage
    return Scene.model_validate(data)


def shoebox() -> Scene:
    """The shoebox example, quick to run."""
    scene = Scene.from_yaml(SHOEBOX)
    sim = scene.simulation.model_copy(
        update={
            "ir_length_s": 0.4,
            "seed": 1,
            "late": scene.simulation.late.model_copy(update={"rays": 500}),
            "early": scene.simulation.early.model_copy(update={"max_order": 1}),
        }
    )
    return scene.model_copy(update={"simulation": sim, "outputs": scene.outputs[:1]})


ABSORBING = {"absorption": [1.0] * 8, "scattering": 0.0}


def wall_reflection_db(points, air, reflection: float = 0.95) -> np.ndarray:
    """The level over free field (dB) that an endless hard wall at x = 10 adds at
    ``points`` (..., 3) for the source of :func:`field`: its image source, 5 % absorbed,
    and the air's attenuation ``air`` (1/m, per band or one) over the longer way."""
    points = np.asarray(points, dtype=float)
    direct = np.linalg.norm(points - [0.0, 0.0, 1.0], axis=-1)
    image = np.linalg.norm(points - [20.0, 0.0, 1.0], axis=-1)
    extra = np.exp(-np.multiply.outer(image - direct, np.atleast_1d(air)))
    out = 10.0 * np.log10(1.0 + reflection * ((direct / image) ** 2)[..., None] * extra)
    return out[..., 0] if np.ndim(air) == 0 else out


# -- the level relative to free field -----------------------------------------------


def test_free_field_energy_is_the_direct_sound_in_free_field() -> None:
    room = Room.from_scene(field())
    air = room.air_attenuation()
    points = np.array([[20.0, 0.0, 1.5], [0.0, 30.0, 4.0]])
    energy = free_field_energy([[0.0, 0.0, 1.0]], points, air)
    r = np.linalg.norm(points - [0, 0, 1.0], axis=1)
    expected = np.exp(-np.outer(r, air)) / (4 * math.pi * r[:, None] ** 2)
    np.testing.assert_allclose(energy, expected, rtol=1e-12)
    # A line's points share the energy; a point on the source has none to give.
    pair = free_field_energy([[0, 0, 1.0], [0, 2, 1.0]], points[:1], air)
    one = free_field_energy([[0, 2, 1.0]], points[:1], air)
    np.testing.assert_allclose(pair, 0.5 * (energy[:1] + one), rtol=1e-12)
    assert np.isnan(free_field_energy([[0, 0, 1.0]], [[0, 0, 1.0]], air)).all()


def test_over_an_absorbing_ground_the_level_is_that_of_free_field() -> None:
    response = simulate(Room.from_scene(field(ABSORBING)))[("s", "l0")]
    assert response.relative_to_free_field() == pytest.approx(np.zeros(8), abs=0.01)


def test_the_level_relative_to_free_field_over_grass_against_the_exact_reflection() -> None:
    """The ground dip: the direct sound and the spherical wave's reflection, averaged
    over each band, as the response's level relative to free field."""
    r, zs, zr = 20.0, 1.0, 1.5
    response = simulate(Room.from_scene(field()))[("s", "l0")]
    level = response.relative_to_free_field()
    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
    theory = []
    for centre in band_centres("octave"):
        f = np.linspace(centre / math.sqrt(2), centre * math.sqrt(2), 401)
        k = 2 * math.pi * f / C
        q = spherical_reflection(1.0 / surface_impedance(f, 200.0), (zs + zr) / r2, k, r2)
        pressure = (np.exp(-1j * k * r1) / r1 + q * np.exp(-1j * k * r2) / r2) * r1
        theory.append(10 * math.log10(np.mean(np.abs(pressure) ** 2)))
    # From 1 kHz up the bands average the interference over their frequencies; up to
    # 500 Hz they are read through octave-band filters, as a sound level meter reads
    # them, whose skirts let in a little of the bands round a dip.
    assert level[4:] == pytest.approx(theory[4:], abs=0.05)
    assert level[:4] == pytest.approx(theory[:4], abs=0.75)
    assert level[3] < -4.0 and level[3] == level.min()  # the dip
    # Indoors there is no free field to compare with.
    indoor = simulate(shoebox(), threads=1)
    some = next(iter(indoor.responses.values()))
    assert some.free_field is None and np.isnan(some.relative_to_free_field()).all()


def test_the_level_relative_to_free_field_keeps_out_the_source_s_feed_and_directivity() -> None:
    plain = field(ABSORBING).to_data()
    plain["sources"][0].update(gain_db=6.0, directivity="cardioid", eq_db=[3.0] * 8)
    response = simulate(Room.from_scene(Scene.model_validate(plain)))[("s", "l0")]
    assert response.relative_to_free_field() == pytest.approx(np.zeros(8), abs=0.01)


# -- the insertion loss -------------------------------------------------------------


def test_what_an_insertion_loss_can_name_is_checked() -> None:
    scene = field(barriers=[WALL], buildings=[SHED], insertion_loss_of=["wall", "shed"])
    assert scene.geometry.part_names() == ["shed", "wall"]
    for names, message in (
        (["gate"], "names 'gate', which is no building, barrier or vegetation"),
        (["wall", "wall"], "names 'wall' twice"),
        ([], "at least 1 item"),
    ):
        with pytest.raises(ValidationError, match=message):
            field(barriers=[WALL], insertion_loss_of=names)
    data = Scene.from_yaml(SHOEBOX).to_data()
    data["simulation"]["insertion_loss_of"] = ["wall"]
    with pytest.raises(ValidationError, match="for outdoor scenes"):
        Scene.model_validate(data)


def test_a_scene_without_its_parts() -> None:
    scene = field(barriers=[WALL, {**WALL, "name": None}], insertion_loss_of=["wall"])
    bare = scene.without(["barrier_2"])
    assert bare.geometry.barrier_names() == ["wall"]
    assert bare.simulation.insertion_loss_of is None
    assert "barriers" not in scene.without(["wall", "barrier_2"]).to_data()["geometry"]
    with pytest.raises(ValueError, match="no building, barrier or vegetation is named 'gate'"):
        scene.without(["gate"])
    # Indoors there is nothing to take away.
    with pytest.raises(ValueError, match="for outdoor scenes"):
        insertion.simulate(Scene.from_yaml(SHOEBOX), without=["wall"])


def test_a_barrier_s_insertion_loss_over_an_absorbing_ground() -> None:
    """Without the barrier the level is that of free field, so the barrier's insertion
    loss is its level relative to free field with the sign turned; a barrier that is not
    in the way takes nothing off."""
    scene = field(
        ABSORBING,
        barriers=[WALL],
        listeners=((20, 0, 1.5), (-20, 0, 1.5)),
        insertion_loss_of=["wall"],
    )
    room = Room.from_scene(scene)
    result = insertion.simulate(room)
    assert result.removed == ("wall",) and set(result.insertion_loss) == {("s", "l0"), ("s", "l1")}
    behind = result.insertion_loss[("s", "l0")]
    relative = result[("s", "l0")].relative_to_free_field()
    assert behind == pytest.approx(-relative, abs=0.05)
    assert (behind > 5.0).all() and behind[-1] > behind[0]  # more off the higher bands
    # On the source's side the wall sends a reflection back, a negative insertion loss:
    # that of its image source, except in the low bands, where the wall is small against
    # the wavelength and the diffraction from its top edge interferes with the reflection.
    in_front = -result.insertion_loss[("s", "l1")]
    image = wall_reflection_db((-20, 0, 1.5), room.air_attenuation("octave"))
    assert in_front[4:] == pytest.approx(image[4:], abs=0.1)
    assert (in_front[:4] > 0.3).all() and np.abs(in_front[:4] - image[:4]).max() < 0.6
    assert result.details["insertion_loss"]["parts"] == 1.0
    # The same as two runs made by hand with the same seed.
    by_hand = simulate(Room.from_scene(scene))[("s", "l0")].relative_to_free_field()
    bare = simulate(Room.from_scene(scene.without(["wall"])))[("s", "l0")].relative_to_free_field()
    assert behind == pytest.approx(bare - by_hand, abs=1e-9)


def test_without_an_insertion_loss_the_run_is_the_plain_one() -> None:
    scene = field(barriers=[WALL])
    plain = simulate(Room.from_scene(scene))
    same = insertion.simulate(Room.from_scene(scene))
    assert same.removed == () and same.insertion_loss == {}
    a, b = plain[("s", "l0")], same[("s", "l0")]
    np.testing.assert_array_equal(a.energy(), b.energy())


def test_a_random_seed_is_shared_by_both_runs() -> None:
    scene = field(ABSORBING, barriers=[WALL], seed=None, listeners=((-20, 0, 1.5),))
    result = insertion.simulate(Room.from_scene(scene), without=["wall"])
    assert result.settings.seed is not None  # drawn once, for both runs
    # The wall behind the source sends a reflection back, whatever the seed.
    loss = result.insertion_loss[("s", "l0")]
    assert (loss < -0.3).all() and (loss > -1.5).all()


def test_re_weighting_drops_the_insertion_loss() -> None:
    scene = field(barriers=[WALL], insertion_loss_of=["wall"])
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = insertion.simulate(room, groups=face_group)
    data = scene.to_data()
    data["materials"]["hard"]["absorption"] = [0.3] * 8
    changed = Scene.model_validate(data)
    found = reweighted_result(result, scene, face_group, materials, changed)
    assert found is not None and found.removed == () and found.insertion_loss == {}


def test_the_report_gives_the_level_relative_to_free_field_and_the_insertion_loss(
    tmp_path,
) -> None:
    scene = field(barriers=[WALL], insertion_loss_of=["wall"])
    result = insertion.simulate(Room.from_scene(scene))
    report = write_results(result, tmp_path / "out")
    assert report["removed"] == ["wall"]
    entry = report["responses"][0]
    response = result[("s", "l0")]
    relative = response.relative_to_free_field()
    assert entry["re_free_field_db"] == pytest.approx(relative, abs=0.006)
    spl = np.array(entry["spl_db"])
    assert entry["re_free_field_a_db"] == pytest.approx(
        a_weighted(spl) - a_weighted(spl - relative), abs=0.02
    )
    loss = result.insertion_loss[("s", "l0")]
    assert entry["insertion_loss_db"] == pytest.approx(loss, abs=0.006)
    assert entry["insertion_loss_a_db"] == pytest.approx(
        insertion.a_weighted_loss(spl, loss), abs=0.02
    )
    saved = json.loads((tmp_path / "out" / "parameters.json").read_text())
    assert saved["responses"][0]["insertion_loss_db"] == entry["insertion_loss_db"]
    # Indoors the report is as it was.
    indoor = write_results(simulate(shoebox(), threads=1), tmp_path / "room")
    assert "removed" not in indoor
    assert not {"re_free_field_db", "insertion_loss_db"} & set(indoor["responses"][0])


# -- maps -------------------------------------------------------------------------


def test_outdoor_maps_give_the_level_relative_to_free_field() -> None:
    scene = field(ABSORBING, barriers=[WALL], coverage={"spacing": 8.0, "height": 1.5, "rays": 300})
    result = coverage_map(Room.from_scene(scene))
    assert result.parameters() == [*ROOM_MAP_PARAMETERS, "re_free_field"]
    # At 4 kHz the wall is large against the wavelength: in front of it, free field and
    # the reflection of its image source; behind it, the shadow.
    relative = result.get().value("re_free_field", 6)
    in_front = result.grid.points[:, 0] < 8.0
    air = Room.from_scene(scene).air_attenuation("octave")[6]
    image = wall_reflection_db(result.grid.points[in_front], air)
    assert (image > 0.1).all() and relative[in_front] == pytest.approx(image, abs=0.15)
    assert (relative[result.grid.points[:, 0] > 12.0] < -3.0).all()  # behind the wall
    assert "re_free_field" in result.as_dict()["parameters"]


def test_indoor_maps_keep_their_parameters() -> None:
    from acoustic_spaces.scene import Coverage

    result = coverage_map(shoebox(), settings=Coverage(spacing=2.0, rays=300), threads=1)
    names = list(result.as_dict()["parameters"])
    assert names == ["spl", "spl_a", "g", "c50", "c80", "d50", "edt", "t20", "t30", "ts", "sti"]


def test_insertion_loss_maps_share_their_points() -> None:
    scene = field(
        ABSORBING,
        barriers=[WALL],
        buildings=[SHED],
        insertion_loss_of=["shed"],
        coverage={"spacing": 8.0, "height": 1.5, "rays": 300},
    )
    room = Room.from_scene(scene)
    result = insertion.coverage_map(room, without=["wall"])
    assert result.removed == ("wall",)
    assert set(INSERTION_LOSS_MAP_PARAMETERS) <= set(result.parameters())
    loss = result.get().value("insertion_loss", 6)
    behind = result.grid.points[:, 0] > 10.0
    # Behind the wall it takes sound away; in front, it sends some back.
    assert (loss[behind] > 3.0).all() and (loss[~behind] < 0.0).all() and loss.min() > -3.5
    plain = coverage_map(Room.from_scene(scene.without(["wall"])), grid=result.grid)
    kept, bare = result.get().values, plain.get().values
    np.testing.assert_allclose(
        kept["insertion_loss"], bare["re_free_field"] - kept["re_free_field"]
    )
    np.testing.assert_allclose(
        kept["insertion_loss_a"],
        [
            a_weighted(s + d) - a_weighted(s)
            for s, d in zip(kept["spl"], kept["insertion_loss"], strict=True)
        ],
    )
    # The shed stands where the plain map has points of its own; the grid is the scene's.
    assert len(result.grid) < len(coverage_map(Room.from_scene(scene.without(["shed"]))).grid)


# -- the command line ---------------------------------------------------------------


def test_the_command_line(tmp_path, capsys) -> None:
    scene = field(barriers=[WALL], coverage={"spacing": 10.0, "height": 1.5, "rays": 200})
    path = tmp_path / "field.yaml"
    scene.to_yaml(path)
    out = tmp_path / "out"
    assert main(["simulate", str(path), "--out", str(out), "--without", "wall"]) == 0
    text = capsys.readouterr().out
    assert "re free field" in text and "insertion loss of wall" in text
    report = json.loads((out / "parameters.json").read_text())
    assert report["removed"] == ["wall"] and "insertion_loss_a_db" in report["responses"][0]
    assert main(["simulate", str(path), "--out", str(out), "--without", "gate"]) == 1
    assert "no building, barrier or vegetation is named 'gate'" in capsys.readouterr().err
    plot = tmp_path / "map.png"
    args = ["map", str(path), "--parameter", "insertion_loss_a", "--plot", str(plot)]
    assert main(args) == 1
    assert "this map has none" in capsys.readouterr().err
    pytest.importorskip("matplotlib")
    assert main([*args, "--without", "wall"]) == 0
    text = capsys.readouterr().out
    assert "Insertion loss (dB(A))" in text and "Re free field (dB)" in text
    assert plot.is_file()
