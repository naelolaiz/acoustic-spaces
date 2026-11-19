import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import skip_without_extra

from acoustic_spaces.room import UNASSIGNED, Room, speed_of_sound
from acoustic_spaces.scene import OCTAVE_BANDS_HZ, Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHOEBOX = EXAMPLES / "shoebox" / "scene.yaml"


def _scene(tmp_path: Path, **changes) -> Path:
    """A copy of the shoebox scene with top-level keys replaced."""
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["geometry"]["file"] = str(EXAMPLES / "shoebox" / "shoebox.obj")
    data.update(changes)
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_shoebox_room() -> None:
    room = Room.from_yaml(SHOEBOX)
    assert room.problems() == []
    assert room.warnings() == []
    assert room.volume_m3 == pytest.approx(144.0)
    assert room.area_by_material() == pytest.approx(
        {"painted_concrete": 84.0, "carpet_on_concrete": 48.0, "acoustic_ceiling": 48.0}
    )
    assert room.sources_inside.tolist() == [True]
    assert room.listeners_inside.tolist() == [True]


def test_a_microphone_no_output_hears_is_warned_about(tmp_path: Path) -> None:
    seat = {"name": "seat-a", "position": [6.0, 3.0, 1.2], "pattern": {"pair": "ms"}}
    heard = Room.from_yaml(_scene(tmp_path, listeners=[seat]))
    assert not [w for w in heard.warnings() if "microphone" in w]
    unheard = Room.from_yaml(_scene(tmp_path, listeners=[seat], outputs=["binaural"]))
    assert [w for w in unheard.warnings() if "microphone" in w] == [
        "the microphone of listener 'seat-a' (ms) is heard only in the mono output, which "
        "the scene's outputs leave out"
    ]


def test_sabine_and_eyring_match_the_formulas() -> None:
    room = Room.from_yaml(SHOEBOX)
    materials = room.scene.materials
    k = OCTAVE_BANDS_HZ.index(1000)
    a = (
        84 * materials["painted_concrete"].absorption[k]
        + 48 * materials["carpet_on_concrete"].absorption[k]
        + 48 * materials["acoustic_ceiling"].absorption[k]
    )
    c = speed_of_sound(20.0)
    assert c == pytest.approx(343.2, abs=0.1)
    sabine = 24 * math.log(10) * 144 / (c * a)
    eyring = 24 * math.log(10) * 144 / (-c * 180 * math.log(1 - a / 180))
    assert room.reverberation_time("sabine")[k] == pytest.approx(sabine)
    assert room.reverberation_time("eyring")[k] == pytest.approx(eyring)
    assert np.all(room.reverberation_time("eyring") < room.reverberation_time("sabine"))
    with pytest.raises(ValueError, match="formula"):
        room.reverberation_time("millington")


def test_arau_puchades_allows_for_uneven_absorption(tmp_path: Path) -> None:
    room = Room.from_yaml(SHOEBOX)
    materials = room.scene.materials
    k = OCTAVE_BANDS_HZ.index(1000)
    walls = materials["painted_concrete"].absorption[k]
    floor_ceiling = (
        materials["carpet_on_concrete"].absorption[k] + materials["acoustic_ceiling"].absorption[k]
    ) / 2
    constant = 24 * math.log(10) * 144 / speed_of_sound(20.0)
    # Eyring per axis from the walls facing it, weighted by their share of the area.
    x, y, z = (constant / (-180 * math.log(1 - a)) for a in (walls, walls, floor_ceiling))
    expected = x ** (36 / 180) * y ** (48 / 180) * z ** (96 / 180)
    assert room.reverberation_time("arau_puchades")[k] == pytest.approx(expected)
    # The absorbing ceiling and floor face one way: far longer than Eyring here.
    assert room.reverberation_time("arau_puchades")[k] > 2 * room.reverberation_time("eyring")[k]

    # Spread evenly, the absorption gives Eyring's time.
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    path = _scene(tmp_path, assignments={name: "painted_concrete" for name in data["assignments"]})
    room = Room.from_yaml(path)
    np.testing.assert_allclose(
        room.reverberation_time("arau_puchades", air_absorption=True),
        room.reverberation_time("eyring", air_absorption=True),
    )


def test_names_matching_materials_and_default_material(tmp_path: Path) -> None:
    scene = Scene.from_yaml(_scene(tmp_path, assignments={"floor": "carpet_on_concrete"}))
    # "ceiling" and "walls" match nothing yet.
    room = Room.from_scene(scene)
    assert (room.face_materials == UNASSIGNED).sum() == 10
    assert room.unassigned_labels() == pytest.approx({"ceiling": 48.0, "walls": 84.0})
    assert "surfaces without a material" in room.problems()[0]

    # A geometry name equal to a material name needs no assignment.
    scene.materials["walls"] = scene.materials["painted_concrete"]
    scene.default_material = "acoustic_ceiling"
    room = Room.from_scene(scene)
    assert room.problems() == []
    assert room.area_by_material()["walls"] == pytest.approx(84.0)
    assert room.area_by_material()["acoustic_ceiling"] == pytest.approx(48.0)


def test_names_match_whatever_their_case(tmp_path: Path) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    upper = {name.upper(): material for name, material in data["assignments"].items()}
    room = Room.from_yaml(_scene(tmp_path, assignments=upper))
    assert room.problems() == [] and room.unused_assignments == []
    # The name as written wins over one in another case.
    exact = {**upper, "floor": "acoustic_ceiling"}
    room = Room.from_yaml(_scene(tmp_path, assignments=exact))
    assert room.area_by_material()["acoustic_ceiling"] == pytest.approx(48.0 + 48.0)
    # A material named like a surface, in another case, still takes it.
    scene = Scene.from_yaml(_scene(tmp_path, assignments={"FLOOR": "carpet_on_concrete"}))
    scene.materials["Walls"] = scene.materials["painted_concrete"]
    scene.default_material = "acoustic_ceiling"
    assert Room.from_scene(scene).area_by_material()["Walls"] == pytest.approx(84.0)


def test_points_outside_and_unused_assignments(tmp_path: Path) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    path = _scene(
        tmp_path,
        sources=[{"name": "outside", "position": [10.0, 3.0, 1.5]}],
        assignments={**data["assignments"], "balcony": "painted_concrete"},
    )
    room = Room.from_yaml(path)
    assert room.sources_inside.tolist() == [False]
    assert any("'outside'" in p and "outside the room" in p for p in room.problems())
    assert not any("close off no room" in p for p in room.problems())
    assert room.unused_assignments == ["balcony"]
    assert any("balcony" in w for w in room.warnings())
    # With none of them in a room, the problems say what the faces of solids do.
    path = _scene(
        tmp_path,
        sources=[{"name": "outside", "position": [10.0, 3.0, 1.5]}],
        listeners=[{"name": "far", "position": [12.0, 3.0, 1.5]}],
    )
    problems = Room.from_yaml(path).problems()
    assert len(problems) == 2 and all("outside the room; no source or" in p for p in problems)
    assert all("walls drawn with their thickness, close off no room" in p for p in problems)


def test_listener_at_a_source_is_a_problem(tmp_path: Path) -> None:
    # At the source the direct sound has no finite level: the response came out NaN.
    path = _scene(
        tmp_path,
        listeners=[
            {"name": "on-talker", "position": [2.0, 3.0, 1.5]},
            {"name": "near-talker", "position": [2.005, 3.0, 1.5]},
            {"name": "close", "position": [2.02, 3.0, 1.5]},
        ],
    )
    room = Room.from_yaml(path)
    problems = room.problems()
    assert len(problems) == 2
    for problem, name in zip(problems, ["on-talker", "near-talker"], strict=True):
        assert problem.startswith(f"listener {name!r} at [")
        assert "within 1 cm of source 'talker'" in problem
    from acoustic_spaces.engine import simulate

    with pytest.raises(ValueError, match="within 1 cm of source 'talker'"):
        simulate(room, listeners=["close"])


def test_points_on_or_next_to_a_surface(tmp_path: Path) -> None:
    # On a wall the inside checks can say either: this listener counted as inside, and
    # its detector sphere shrank to 1 mm. A centimetre from one, the sphere is 9 mm and
    # catches next to no rays: G came out 1.5 dB apart between two seeds.
    path = _scene(
        tmp_path,
        sources=[{"name": "floor", "position": [2.0, 3.0, 0.0]}],
        listeners=[
            {"name": "on", "position": [8.0, 3.0, 1.2]},
            {"name": "next to", "position": [7.99, 3.0, 1.2]},
            {"name": "clear", "position": [7.9, 3.0, 1.2]},
        ],
    )
    room = Room.from_yaml(path)
    problems = room.problems()
    assert len(problems) == 2
    assert problems[0].startswith(
        "source 'floor' at [2.0, 3.0, 0.0] is on a surface (within 1 µm of it), neither"
    )
    assert problems[1].startswith("listener 'on' at [8.0, 3.0, 1.2] is on a surface")
    (near,) = [w for w in room.warnings() if "from a surface" in w]
    assert near.startswith("listener 'next to' is 1 cm from a surface")
    assert "a radius of 0.9 cm instead of 50 cm or more" in near
    assert "move it 5.6 cm or more" in near


def test_a_point_on_an_opening_is_in_the_air() -> None:
    # The sky lets all sound through: a listener at its edge stands in the air. The late
    # part's sphere, kept clear of every face, shrinks to 1 mm there all the same.
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"ground": {"extent": [[-50.0, -20.0], [50.0, 20.0]], "sky": 20.0}},
                "materials": {"lawn": {"impedance": "grass"}},
                "assignments": {"ground": "lawn"},
                "sources": [{"name": "s", "position": [0.0, 0.0, 2.0]}],
                "listeners": [{"name": "edge", "position": [50.0, 0.0, 1.5]}],
            }
        )
    )
    assert room.problems() == []
    (near,) = [w for w in room.warnings() if "detector sphere" in w]
    assert near.startswith("listener 'edge' is on an opening")
    assert "a radius of 0.1 cm" in near


def _outdoors(width: float, source, listener, line_m: float | None = None) -> Room:
    """A hard ground ``width`` m wide under an open sky, with a wall 4 m high along
    x = -3 from y = -20 to 20; the source a line ``line_m`` long along y, if given."""
    half = width / 2.0
    source = {"name": "s", "position": [float(x) for x in source]}
    if line_m is not None:
        source |= {"line_m": line_m}
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {
                    "ground": {"extent": [[-half, -half], [half, half]], "sky": 30.0},
                    "barriers": [
                        {"name": "wall", "path": [[-3.0, -20.0], [-3.0, 20.0]], "height": 4.0}
                    ],
                },
                "materials": {"hard": {"absorption": [0.02] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [source],
                "listeners": [{"name": "l", "position": [float(x) for x in listener]}],
            }
        )
    )


def _early_level(room: Room) -> tuple[int, np.ndarray]:
    """How many early paths ``room``'s source has to its listener, and their level per
    band (dB)."""
    from acoustic_spaces.engine import early_paths
    from acoustic_spaces.engine.coherent import total_energy

    paths = early_paths(room)
    return len(paths), 10.0 * np.log10(total_energy(paths))


def test_the_distance_of_a_point_on_a_surface_grows_with_the_model() -> None:
    """A point is on a surface within a micrometre of it, or within twice the least
    length of a ray traced on the CPU where that is more."""
    from acoustic_spaces.engine.early import ray_start
    from acoustic_spaces.engine.model import TraceModel

    assert Room.from_yaml(SHOEBOX).on_surface_m() == 1e-6  # 8 by 6 by 3 m
    for width in (60.0, 2000.0, 10000.0):
        room = _outdoors(width, [5.0, 0.0, 1.5], [20.0, 0.0, 1.5])
        least = float(TraceModel.from_room(room).bvh.min_distance)
        assert ray_start(room.mesh) == pytest.approx(least, rel=1e-12)
        assert room.on_surface_m() == pytest.approx(max(1e-6, 2.0 * least), rel=1e-12)
    # 2 km wide, 2.8 km from corner to corner: 5.7 µm.
    room = _outdoors(2000.0, [5.0, 0.0, 1.5], [20.0, 0.0, 1.5])
    assert room.on_surface_m() == pytest.approx(5.657e-6, rel=1e-3)


@pytest.mark.parametrize("width", [2000.0, 10000.0])
@pytest.mark.parametrize(
    ("place", "other"),
    [
        (lambda e: [0.5, 0.0, e], [0.0, 0.0, 3.0]),  # over the ground, the source above
        (lambda e: [-3.0 + e, 0.5, 3.0], [1.0, 0.0, 3.5]),  # at the wall's face
    ],
    ids=["ground", "wall"],
)
def test_a_listener_nearer_a_surface_than_a_ray_can_meet_it_is_on_it(width, place, other) -> None:
    """A ray traced on the CPU meets no face nearer its start than 1e-9 of the model's
    diagonal, so the path traced back from a listener misses a reflection whose last leg
    is shorter, as that of a source almost over the listener is: 6 dB of the level over
    a hard ground alone. Where that is more than the micrometre of a small model, the
    room takes the listener as on the surface within twice that distance; from there on
    the paths and their level are those of the same place in a model 60 m wide, where a
    ray meets a face from 0.09 µm."""
    from acoustic_spaces.engine import simulate

    within = _outdoors(width, other, other).on_surface_m()
    assert within > 5e-6
    on = _outdoors(width, other, place(0.4 * within))
    (problem,) = on.problems()
    assert problem.startswith("listener 'l' at ")
    assert f"is on a surface (within {within * 1e6:.3g} µm of it, twice the least" in problem
    with pytest.raises(ValueError, match="is on a surface"):
        simulate(on)
    # Being a problem, it is not also warned of as near a surface, as it is from there on.
    assert not [w for w in on.warnings() if "detector sphere" in w]
    for share in (1.05, 3.0):
        off = _outdoors(width, other, place(share * within))
        assert off.problems() == []
        (near,) = [w for w in off.warnings() if "detector sphere" in w]
        assert near.startswith("listener 'l' is ") and "from a surface" in near
        count, level = _early_level(off)
        same, expected = _early_level(_outdoors(60.0, other, place(share * within)))
        assert count == same
        assert level == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("width", [2000.0, 10000.0])
@pytest.mark.parametrize(
    "place",
    [
        lambda e: [5.0, 0.0, e],  # over the ground
        lambda e: [-3.0 + e, 0.5, 3.0],  # at the wall's face
    ],
    ids=["ground", "wall"],
)
def test_a_source_nearer_a_surface_than_a_ray_can_meet_it_is_on_it(width, place) -> None:
    """The rays of a source nearer a surface than a ray can meet it go through the
    surface. The room takes the source as on the surface within twice that distance, as
    it does a listener; from there on the paths and their level are those of the same
    place in a model 60 m wide."""
    from acoustic_spaces.engine import simulate

    listener = [20.0, 0.0, 1.5]
    within = _outdoors(width, listener, listener).on_surface_m()
    on = _outdoors(width, place(0.4 * within), listener)
    (problem,) = on.problems()
    assert problem.startswith("source 's' at ")
    assert f"is on a surface (within {within * 1e6:.3g} µm of it, twice the least" in problem
    with pytest.raises(ValueError, match="is on a surface"):
        simulate(on)
    for share in (1.05, 3.0):
        off = _outdoors(width, place(share * within), listener)
        assert off.problems() == []
        count, level = _early_level(off)
        same, expected = _early_level(_outdoors(60.0, place(share * within), listener))
        assert count == same
        assert level == pytest.approx(expected, abs=1e-9)


def test_a_line_source_reaches_a_surface_by_the_distance_of_the_model() -> None:
    """Each of the points a line source is made of counts: a line beside the wall and
    past its end, in a model 2 km wide, reaches the surface with one of them 2 µm from
    the wall's face, its middle half a metre past the wall's end; 6 µm from the face it
    does not."""
    listener = [20.0, 0.0, 1.5]
    on = _outdoors(2000.0, [-3.0 + 2e-6, 20.5, 2.0], listener, line_m=2.0)
    points = on.scene.sources[0].elements()
    assert points[:, 0] == pytest.approx(-3.0 + 2e-6, abs=1e-9)
    assert sorted(points[:, 1]) == pytest.approx([19.75, 20.25, 20.75, 21.25])
    (problem,) = on.problems()
    assert problem.startswith("source 's' at ")
    assert " reaches on a surface (within 5.66 µm of it, twice the least" in problem
    assert _outdoors(2000.0, [-3.0 + 6e-6, 20.5, 2.0], listener, line_m=2.0).problems() == []


def test_a_listener_on_an_opening_by_the_distance_of_the_model() -> None:
    """The sky lets all sound through: a listener 3 µm from its side in a model 2 km
    wide, nearer than the 5.7 µm there, is in the air, and is warned of as on an
    opening."""
    room = _outdoors(2000.0, [990.0, 0.0, 2.0], [1000.0 - 3e-6, 0.0, 1.5])
    assert room.problems() == []
    (near,) = [w for w in room.warnings() if "detector sphere" in w]
    assert near.startswith("listener 'l' is on an opening")


def test_flipped_faces_do_not_put_points_outside(tmp_path: Path) -> None:
    # The shoebox with its floor written the other way round.
    obj = tmp_path / "flipped.obj"
    text = (EXAMPLES / "shoebox" / "shoebox.obj").read_text(encoding="utf-8")
    obj.write_text(text.replace("f 1 2 3\nf 1 3 4", "f 3 2 1\nf 4 3 1"), encoding="utf-8")
    path = _scene(
        tmp_path,
        geometry={"file": str(obj)},
        listeners=[{"name": "low", "position": [6.0, 3.0, 0.3]}],
    )
    room = Room.from_yaml(path)
    assert room.problems() == []
    assert room.warnings() == []
    assert room.sources_inside.tolist() == [True]
    assert room.listeners_inside.tolist() == [True]
    assert room.volume_m3 == pytest.approx(144.0)


def test_dxf_in_millimetres_without_scene_units(tmp_path: Path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    from acoustic_spaces.geometry import box_mesh

    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 4  # millimetres
    for corners in box_mesh((8000.0, 6000.0, 3000.0)).triangles:
        doc.modelspace().add_3dface([tuple(c) for c in corners])
    dxf = tmp_path / "room.dxf"
    doc.saveas(dxf)
    path = _scene(tmp_path, geometry={"file": str(dxf)}, default_material="painted_concrete")
    room = Room.from_yaml(path)
    assert room.scene.geometry.units is None
    assert room.volume_m3 == pytest.approx(144.0)  # not 8000 x 6000 x 3000 m
    assert room.problems() == []
    # Units in the scene win over the file's.
    path = _scene(
        tmp_path,
        geometry={"file": str(dxf), "units": "cm"},
        default_material="painted_concrete",
    )
    assert Room.from_yaml(path).volume_m3 == pytest.approx(144.0 * 1000.0)


def test_model_of_degenerate_faces_is_refused(tmp_path: Path) -> None:
    obj = tmp_path / "line.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 2 0 0\nv 3 0 0\ng floor\nf 1 2 3\nf 2 3 4\n")
    # It loaded as a closed room of no triangles, its points all "outside the room".
    with pytest.raises(ValueError, match="line.obj has no faces left: all 2 of them have no area"):
        Room.from_yaml(_scene(tmp_path, geometry={"file": str(obj)}))


def test_open_model_skips_volume_and_inside_checks(tmp_path: Path) -> None:
    obj = tmp_path / "open.obj"
    lines = (EXAMPLES / "shoebox" / "shoebox.obj").read_text(encoding="utf-8").splitlines()
    ceiling = lines.index("g ceiling")
    obj.write_text("\n".join(lines[:ceiling] + lines[ceiling + 3 :]) + "\n", encoding="utf-8")
    room = Room.from_yaml(_scene(tmp_path, geometry={"file": str(obj)}))
    assert room.volume_m3 is None
    assert room.sources_inside is None
    assert "not closed" in room.warnings()[0]
    with pytest.raises(ValueError, match="closed"):
        room.reverberation_time()


def test_inconsistent_edges_are_not_said_to_matter_for_the_volume() -> None:
    room = Room.from_yaml(SHOEBOX)
    room.report.inconsistent_edges = 3  # a part that cannot be oriented, as a Moebius panel
    (warning,) = room.warnings()
    assert warning.startswith("3 edges join faces with opposite orientation")
    assert "only the direction of the normals is unknown" in warning
    assert "only matters for the volume" not in warning


@pytest.mark.parametrize(
    "scene", sorted(EXAMPLES.glob("*/scene.yaml")), ids=lambda p: p.parent.name
)
def test_example_scenes_are_ready_to_simulate(scene: Path) -> None:
    skip_without_extra(scene.parent.name)
    room = Room.from_yaml(scene)
    assert room.problems() == []
    if scene.parent.name == "leaky_classroom":  # open on purpose, for the leak finder
        (warning,) = room.warnings()
        assert "not closed" in warning and "validate --leaks" in warning
        return
    warnings = room.warnings()
    assert warnings == []
    assert room.report.closed and room.report.normals_inward


def test_rooms_reuse_an_unchanged_mesh() -> None:
    room = Room.from_yaml(SHOEBOX)
    data = room.scene.model_dump(mode="json", exclude_none=True)
    data["sources"][0]["position"] = [2.0, 2.0, 1.5]
    moved = Room.from_scene(Scene.model_validate(data), reuse=room)
    assert moved.mesh is room.mesh
    assert moved.scene.sources[0].position == (2.0, 2.0, 1.5)
    data["geometry"] = {"box": [5.0, 4.0, 3.0]}
    data["assignments"] = {}
    boxed = Room.from_scene(Scene.model_validate(data), reuse=room)
    assert boxed.mesh is not room.mesh
    assert boxed.volume_m3 == pytest.approx(60.0)
