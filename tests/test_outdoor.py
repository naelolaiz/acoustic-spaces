"""Outdoor scenes: the ground, the open sky over it, buildings and barriers."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import early, early_paths
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.coverage import coverage_map, map_grid
from acoustic_spaces.engine.diffraction import find_edges
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.simulate import response_length
from acoustic_spaces.geometry.outdoor import building_mesh, in_polygon
from acoustic_spaces.scene import Coverage
from acoustic_spaces.scene.model import sky_material

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
HALL = [[0, 0], [10, 0], [10, 8], [0, 8]]


def outdoor(geometry, sources=((0, 0, 2),), listeners=((20, 0, 1.5),), **simulation) -> Scene:
    return Scene.model_validate(
        {
            "geometry": geometry,
            "materials": {"hard": {"absorption": [0.2] * 8, "scattering": 0.0}},
            "default_material": "hard",
            "sources": [{"name": f"s{k}", "position": list(p)} for k, p in enumerate(sources)],
            "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
            "simulation": {
                "ir_length_s": 0.3,
                "seed": 1,
                "early": {"method": "image_sources", "max_order": 2},
                "late": {"rays": 2000},
                **simulation,
            },
        }
    )


OPEN = {"ground": {"extent": [[-50, -50], [50, 50]], "sky": 30}}


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"ground": {}}, "the ground needs an extent"),
        ({"buildings": [{"footprint": HALL, "height": 6}]}, "add geometry.ground"),
        ({"ground": {}, "box": [5, 4, 3]}, "not a box or floor_plan"),
        ({"ground": {}, "buildings": [{"name": "sky", "footprint": HALL, "height": 6}]}, "'sky'"),
        (
            {
                "ground": {},
                "buildings": [{"name": "a", "footprint": HALL, "height": 6}],
                "barriers": [{"name": "a", "path": [[20, 0], [20, 5]], "height": 2}],
            },
            "must be unique",
        ),
        ({"ground": {"extent": [[5, 0], [0, 5]]}}, "min < max"),
    ],
)
def test_outdoor_geometry_is_checked(geometry, message) -> None:
    with pytest.raises(ValidationError, match=message):
        outdoor(geometry)


def test_the_sky_is_an_opening_left_out_of_the_scene_file() -> None:
    scene = outdoor(OPEN)
    assert scene.materials["sky"] == sky_material()
    data = scene.to_data()
    assert "sky" not in data["materials"]
    assert Scene.model_validate(data) == scene
    # A sky of the scene's own stays.
    data["materials"]["sky"] = {"absorption": [0.5] * 8}
    data["assignments"] = {"sky": "sky"}
    assert Scene.model_validate(data).to_data()["materials"]["sky"]["absorption"] == [0.5] * 8
    # Indoors nothing changes.
    indoor = Scene.from_yaml(SHOEBOX)
    assert "sky" not in indoor.materials and indoor.geometry.ground is None


def test_the_open_air_is_a_closed_box_with_the_sky_round_it() -> None:
    room = Room.from_scene(outdoor(OPEN))
    assert room.report.closed and not room.problems()
    assert room.outdoor.extent == (-50.0, -50.0, 50.0, 50.0)
    assert room.outdoor.sky == 30.0
    assert room.volume_m3 == pytest.approx(100 * 100 * 30)
    assert room.sky_faces().sum() == 10  # four walls and a lid, two triangles each
    assert set(room.mesh.label_names()) == {"ground", "sky"}


def test_the_ground_reaches_past_the_buildings_by_the_margin() -> None:
    scene = outdoor(
        {
            "ground": {"margin": 10, "height": 1.0},
            "buildings": [{"name": "hall", "footprint": HALL, "height": 6}],
        },
        sources=[(-5, 4, 2.5)],
        listeners=[(15, 4, 2.5)],
    )
    room = Room.from_scene(scene)
    assert room.outdoor.extent == (-10.0, -10.0, 20.0, 18.0)
    assert room.outdoor.ground == 1.0
    assert room.outdoor.sky == pytest.approx(17.0)  # the roof at 7 m, and 10 m more
    names = set(room.mesh.label_names())
    assert {"hall_roof", "hall_wall_1", "hall_walls", "hall", "walls", "roofs"} <= names
    # Open underneath: the ground is the building's floor.
    built = room.building_faces()
    assert not np.any(np.all(np.abs(room.mesh.triangles[built][:, :, 2] - 1.0) < 1e-9, axis=1))


def test_a_point_inside_a_building_is_a_problem() -> None:
    scene = outdoor(
        {"ground": {}, "buildings": [{"name": "hall", "footprint": HALL, "height": 6}]},
        sources=[(5, 4, 2)],
        listeners=[(15, 4, 1.5)],
    )
    problems = Room.from_scene(scene).problems()
    assert any("inside building 'hall'" in p for p in problems)
    with pytest.raises(ValueError, match="inside building"):
        simulate(Room.from_scene(scene))


def test_a_point_past_the_edge_of_the_ground_is_a_problem() -> None:
    problems = Room.from_scene(outdoor(OPEN, listeners=[(60, 0, 1.5)])).problems()
    assert any("outside the open air" in p for p in problems)


def test_footprints_and_points() -> None:
    square = np.array(HALL, dtype=float)
    points = np.array([[5, 4], [11, 4], [-1, 4], [5, 9]], dtype=float)
    assert in_polygon(points, square).tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="building 'x'"):
        building_mesh("x", [[0, 0], [1, 0], [2, 0]], 3.0, 0.0)


def test_direct_sound_and_one_ground_reflection_in_the_open() -> None:
    """Over a flat ground under an open sky, the only paths are the direct sound and its
    reflection off the ground: nothing comes back from the sky."""
    room = Room.from_scene(outdoor(OPEN))
    paths = early_paths(room)
    assert paths.order.tolist() == [0, 1]
    r1 = math.dist([0, 0, 2], [20, 0, 1.5])
    r2 = math.dist([0, 0, -2], [20, 0, 1.5])
    assert paths.distance_m == pytest.approx([r1, r2])
    ratio = paths.energy[1] / paths.energy[0]
    air = room.air_attenuation(TraceModel.from_room(room).bands)
    expected = 0.8 * (r1 / r2) ** 2 * np.exp(-air * (r2 - r1))
    assert ratio == pytest.approx(expected, rel=1e-6)
    # Specular rays find the same two paths.
    traced = early_paths(room, method="ray_tracing")
    assert traced.distance_m == pytest.approx([r1, r2])


def test_late_rays_leave_through_the_sky() -> None:
    """Without anything round it, the late field over the ground is weak: almost every
    ray leaves through the sky after one bounce."""
    result = simulate(Room.from_scene(outdoor(OPEN)))
    response = result[("s0", "l0")]
    assert response.late.sum() < 0.05 * response.early.energy.sum()


def test_responses_count_from_the_latest_direct_sound() -> None:
    far = outdoor(
        {"ground": {"extent": [[-10, -10], [800, 10]], "sky": 20}}, listeners=[(700, 0, 2)]
    )
    room = Room.from_scene(far)
    result = simulate(room)
    expected = 700 / room.speed_of_sound() + 0.3
    assert result[("s0", "l0")].length_s == pytest.approx(expected)
    assert result.details["s0"]["ir_length_s"] == pytest.approx(expected)
    # Indoors the length is as set.
    shoebox = Room.from_scene(Scene.from_yaml(SHOEBOX))
    assert response_length(shoebox, 1.5, [[0, 0, 0]], [[100, 0, 0]]) == 1.5


def test_the_statistical_late_method_is_refused_outdoors() -> None:
    scene = outdoor(OPEN, late={"method": "statistical"})
    with pytest.raises(ValueError, match="outdoors, use the stochastic"):
        simulate(Room.from_scene(scene))
    with pytest.raises(ValueError, match="outdoors, use the stochastic"):
        coverage_map(Room.from_scene(scene))


def test_a_barrier_diffracts_over_its_top_and_round_its_ends() -> None:
    scene = outdoor(
        {
            "ground": {"margin": 20},
            "barriers": [{"name": "screen", "path": [[5, -10], [5, 10]], "height": 3}],
        },
        sources=[(0, 0, 1)],
        listeners=[(15, 0, 1.5)],
        early={"method": "image_sources", "max_order": 1, "diffraction": "first_order"},
    )
    room = Room.from_scene(scene)
    edges = find_edges(TraceModel.from_room(room))
    # The top and the two ends; not its foot, nor the edges of the ground or the sky.
    assert len(edges) == 3
    paths = early_paths(room)
    assert paths.diffracted.all()  # the screen hides the direct sound and the ground
    over = math.dist([0, 0, 1], [5, 0, 3]) + math.dist([5, 0, 3], [15, 0, 1.5])
    assert paths.distance_m.min() == pytest.approx(over)


@pytest.mark.parametrize("method", ["image_sources", "ray_tracing"])
def test_a_barrier_stops_the_reflection_off_the_ground_at_its_foot(method: str) -> None:
    """A source 5 m in front of a barrier 3 m high and listeners 5 m behind it, as high
    as the source: for the middle one the reflection off the ground meets the barrier
    right at its foot, where the barrier stops it as it does 1 cm either side. It had
    passed there whole."""
    geometry = {
        "ground": {"margin": 20},
        "barriers": [{"name": "screen", "path": [[5, -10], [5, 10]], "height": 3}],
    }
    for x in (9.99, 10.0, 10.01):
        scene = outdoor(
            geometry,
            sources=[(0, 0, 1)],
            listeners=[(x, 0, 1)],
            early={"method": method, "max_order": 3, "diffraction": "none"},
        )
        assert len(early_paths(Room.from_scene(scene))) == 0


def test_the_reflection_at_a_barrier_s_foot_bends_over_it_as_beside_it() -> None:
    """As above, with the edge diffraction: the reflection off the ground that the
    barrier stops right at its foot bends over its top there as it does 1 cm either
    side, so the early level of the middle listener, and of one 3 nm farther (the
    barrier then lies nearer to the end of the leg from the source than that leg is
    tested), lies within 0.1 dB of the mean of theirs in every band. The reflection had
    passed the barrier's foot whole: 6 to 27 dB more."""
    geometry = {
        "ground": {"margin": 20},
        "barriers": [{"name": "screen", "path": [[5, -10], [5, 10]], "height": 3}],
    }
    levels = []
    for x in (9.99, 10.0, 10.0 + 3e-9, 10.01):
        scene = outdoor(
            geometry,
            sources=[(0, 0, 1)],
            listeners=[(x, 0, 1)],
            early={"max_order": 2, "diffraction": "first_order"},
        )
        levels.append(10 * np.log10(total_energy(early_paths(Room.from_scene(scene)))))
    beside = (levels[0] + levels[-1]) / 2
    assert np.abs(np.array(levels[1:-1]) - beside).max() < 0.1


@pytest.mark.parametrize(
    "geometry",
    [
        {"barriers": [{"name": "screen", "path": [[5, -10], [5, 0], [15, 0]], "height": 3}]},
        {
            "buildings": [
                {"name": "hall", "footprint": [[-5, -10], [5, -10], [5, 0], [-5, 0]], "height": 6}
            ],
            "barriers": [{"name": "screen", "path": [[5, 0], [5, 10]], "height": 3}],
        },
    ],
    ids=["turning", "abutting"],
)
def test_a_barrier_stops_the_reflection_off_the_ground_where_it_turns(geometry) -> None:
    """A source at (2, 3, 1) and listeners at (8, -3, 1) on either side of a barrier 3 m
    high that turns a corner at (5, 0), or of one that abuts a building's corner there,
    where the reflection off the ground meets the corner at its foot, or, for listeners 1
    to 3 um aside, 0.5 to 1.5 um aside: the barrier, or the building, stops it there
    as it does for listeners 1 mm either side, and no path of up to the third order is
    found. The reflection had passed the turning barrier whole there and for listeners 1
    to 3 um aside, and the other at the corner, as had the reflection off the roof at its
    corner; paths that reflect at the corner off a wall, the ground or the roof and the
    wall again had passed for listeners 1 to 3 um aside as well."""
    for offset in (-1e-3, -3e-6, -1e-6, 0.0, 1e-6, 3e-6, 1e-3):
        aside = offset / 2**0.5  # level, across the reflection
        scene = outdoor(
            {"ground": {"margin": 20}, **geometry},
            sources=[(2, 3, 1)],
            listeners=[(8 + aside, -3 + aside, 1)],
            early={"method": "image_sources", "max_order": 3, "diffraction": "none"},
        )
        assert len(early_paths(Room.from_scene(scene))) == 0


BLOCK = {"name": "block", "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]], "height": 6}
TOWER = {"name": "tower", "footprint": [[10, 0], [14, 0], [14, 10], [10, 10]], "height": 12}
ANGLE = [
    {"name": "a", "path": [[0, 0], [8, 0]], "height": 4},
    {"name": "c", "path": [[0, 0], [0, 8]], "height": 4},
]


def _in_line_and_beside(
    geometry, source, listener, shift, diffraction, method="image_sources", max_order=2
):
    """The early paths of a listener and of one ``shift`` from it, and their levels."""
    out = []
    for moved in (np.zeros(3), np.asarray(shift)):
        scene = outdoor(
            {"ground": {"margin": 60}, **geometry},
            sources=[source],
            listeners=[tuple(np.asarray(listener) + moved)],
            early={"method": method, "max_order": max_order, "diffraction": diffraction},
        )
        paths = early_paths(Room.from_scene(scene))
        out.append((paths, 10 * np.log10(total_energy(paths))))
    return out


def _at_one_point(paths) -> int:
    """How many paths without an edge reflect twice or more at one point."""
    return sum(
        1
        for i in range(len(paths))
        if paths.edge[i] < 0
        and any(
            math.dist(paths.points[i][k], paths.points[i][k + 1]) < 1e-6
            for k in range(1, paths.order[i])
        )
    )


LONG_HALL = {"name": "hall", "footprint": [[0, -15], [10, -15], [10, 15], [0, 15]], "height": 6}
#: A taller building with a wall in the plane of the hall's end wall y = -15.
TALLER_BESIDE = {
    "name": "tall",
    "footprint": [[20, -25], [40, -25], [40, -15], [20, -15]],
    "height": 20,
}


@pytest.mark.parametrize(
    ("geometry", "source", "listener", "shift", "method"),
    [
        (
            {"buildings": [LONG_HALL]},
            (-9.99, 0, 7.5),
            (15.01, 0, 7.5),
            (0, 0, 1e-4),
            "image_sources",
        ),
        ({"buildings": [BLOCK]}, (-5, -5, 1.5), (-10, -10, 1.5), (1e-4, 0, 0), "image_sources"),
        ({"barriers": ANGLE}, (-2, -2, 1.5), (-4, -4, 2), (1e-4, 0, 0), "image_sources"),
        ({"barriers": ANGLE}, (2, -2, 1.5), (4, -4, 2), (1e-4, 0, 0), "image_sources"),
        (
            {"buildings": [LONG_HALL, TALLER_BESIDE]},
            (-9.99, 0, 7.5),
            (15.01, 0, 7.5),
            (0, 0, 1e-4),
            "ray_tracing",
        ),
    ],
    ids=[
        "roof and end wall",
        "upright corner",
        "outside two barriers",
        "beside two barriers",
        "roof and end wall, the rays' search",
    ],
)
def test_a_way_through_an_outer_edge_is_no_reflection(
    geometry, source, listener, shift, method
) -> None:
    """A source and a listener that an outer edge sees in the same direction across it,
    so that the way from one to the edge and on to the other is the way a reflection off
    both faces that meet there would take: over a hall's roof, the same height above it
    and the same distance from its end wall; in line with a building's upright corner;
    in line with the upright edge where two barriers meet at a right angle, outside them
    and beside one. At least one of the two planes has no face where that sound would
    meet it (the end wall above the roof, a wall or a barrier past the corner), so no
    path reflects off both there, and the early level is the one 0.1 mm beside. Without
    the edge diffraction within 0.005 dB: with the path, which had stood in line and not
    beside it, the level in line had been up to 1.5, 0.7, 1.1 and 0.5 dB more. With the
    edge diffraction within 0.1 dB (0.07 dB at 63 Hz outside the two barriers and
    0.02 dB at the upright corner, where paths that bend at an edge and reflect at the
    corner stand beside the line and not in it; less than 0.001 dB for the other two):
    over the roof the level in line had been 4.3 dB more at 63 Hz. The rays' own search
    (the default) tests those two planes one after the other only where a ray meets
    faces of both in a row, as once a taller building beside the hall has a wall in the
    plane of the hall's end wall (the last case): there the level in line had been up to
    0.44 dB more without the edge diffraction and 1.8 dB more at 63 Hz with it, and is
    within 0.005 dB of the one beside with either."""
    (exact, level), (beside, level_beside) = _in_line_and_beside(
        geometry, source, listener, shift, "none", method
    )
    assert _at_one_point(exact) == 0
    assert len(exact) == len(beside)
    assert np.abs(level - level_beside).max() < 0.005
    (exact, level), (beside, level_beside) = _in_line_and_beside(
        geometry, source, listener, shift, "first_order", method
    )
    assert _at_one_point(exact) == 0
    assert np.abs(level - level_beside).max() < 0.1


@pytest.mark.parametrize(
    ("geometry", "source", "listener", "shift"),
    [
        ({"buildings": [BLOCK]}, (-5, 5, 1.5), (-10, 5, 3), (0, 0, 1e-4)),
        ({"buildings": [BLOCK]}, (-5, 3, 1.5), (-10, 8, 3), (0, 0, 1e-4)),
        ({"buildings": [BLOCK, TOWER]}, (5, 5, 7.5), (7.5, 5, 6.75), (0, 0, 1e-4)),
        ({"barriers": ANGLE}, (2, 2, 1.5), (4, 4, 2), (1e-4, 0, 0)),
    ],
    ids=["foot of a wall", "foot of a wall, at a slant", "wall on a roof", "inside two barriers"],
)
@pytest.mark.parametrize("diffraction", ["none", "first_order"])
def test_a_way_through_an_inner_corner_reflects_there(
    geometry, source, listener, shift, diffraction
) -> None:
    """The same where the two faces make a corner on the sound's side: a wall and the
    ground at its foot, a taller building's wall and the roof it stands on the edge of,
    two barriers seen from inside their angle. The path that reflects off both right at
    the corner stands, as the one that reflects off one and then the other does 0.1 mm
    beside, and the early levels agree within 0.005 dB."""
    (exact, level), (beside, level_beside) = _in_line_and_beside(
        geometry, source, listener, shift, diffraction
    )
    assert _at_one_point(exact) == 1 and _at_one_point(beside) == 0
    assert len(exact) == len(beside)
    assert np.abs(level - level_beside).max() < 0.005


@pytest.mark.parametrize(
    ("bend", "gap", "height", "order"), [(30.0, 2e-4, 1.5, 2), (60.0, 2e-3, 2e-3, 3)]
)
def test_no_way_through_a_barrier_from_a_source_in_its_bend(bend, gap, height, order) -> None:
    """A source ``gap`` (m) from both arms of a barrier bent by ``bend`` degrees,
    ``height`` over the ground, and a listener behind one arm, in a model 10 km wide and
    in one 200 m wide. Without diffraction no path reaches the listener; with
    ``first_order`` the two models have the same paths and early levels to 0.01 dB. The
    paths' tolerance is 1 mm in the wide model, and the source (0.2 mm from both arms),
    or the reflection off the ground behind an arm (1 mm), lay within it of an arm's
    plane: a path that reflected off both arms in the bend and went on through one of
    them counted as a reflection there, at the level of the direct sound."""
    turn = math.radians(bend)
    reach = gap / math.sin(turn / 2)
    source = (math.cos(turn / 2) * reach, math.sin(turn / 2) * reach, height)
    path = [[12.0, 0.0], [0.0, 0.0], [12.0 * math.cos(turn), 12.0 * math.sin(turn)]]
    for diffraction in ("none", "first_order"):
        found = []
        for half in (5000.0, 100.0):
            geometry = {
                "ground": {"extent": [[-half, -half], [half, half]], "sky": 40},
                "barriers": [{"name": "wall", "path": path, "height": 4.0}],
            }
            early_options = {"method": "image_sources", "max_order": order}
            scene = outdoor(
                geometry,
                sources=[source],
                listeners=[(6.0, -3.0, 2.0)],
                early={**early_options, "diffraction": diffraction},
            )
            found.append(early_paths(Room.from_scene(scene)))
        wide, narrow = found
        if diffraction == "none":
            assert len(wide.order) == len(narrow.order) == 0
        else:
            assert len(wide.order) == len(narrow.order) > 0
            level = 10 * np.log10(total_energy(wide) / total_energy(narrow))
            assert np.abs(level).max() < 0.01


@pytest.mark.parametrize("bend", [30.0, 60.0])
def test_a_source_in_a_barrier_s_bend_has_a_wave_each_way_round_it(bend) -> None:
    """A source 1.2 mm from both arms of a barrier bent by ``bend`` degrees, 1.5 m over
    the ground in a model 10 km wide, and a listener out along the bend's bisector, with
    the bands from 1 kHz up adding in energy. The paths off one arm and then the other
    and the other way round mirror the source otherwise, so the early stage keeps both,
    and each is a wave of its own though their images lie 4 to 8 mm apart (less than a
    millionth of the model's size): the bands from 1 kHz up are those of the same
    places in a model 200 m wide (within 0.01 dB). With images within a
    hundred-thousandth of the model's size one wave, they were 2.3 dB higher."""
    turn = math.radians(bend)
    reach = 1.2e-3 / math.sin(turn / 2)
    source = (math.cos(turn / 2) * reach, math.sin(turn / 2) * reach, 1.5)
    listener = (8.0 * math.cos(turn / 2), 8.0 * math.sin(turn / 2), 1.0)
    path = [[12.0, 0.0], [0.0, 0.0], [12.0 * math.cos(turn), 12.0 * math.sin(turn)]]
    level = []
    for half in (5000.0, 100.0):
        geometry = {
            "ground": {"extent": [[-half, -half], [half, half]], "sky": 40},
            "barriers": [{"name": "wall", "path": path, "height": 4.0}],
        }
        early = {"method": "image_sources", "max_order": 2, "coherent_max_band_hz": 500}
        scene = outdoor(geometry, sources=[source], listeners=[listener], early=early)
        level.append(10 * np.log10(total_energy(early_paths(Room.from_scene(scene)))))
    np.testing.assert_allclose(level[0][4:], level[1][4:], atol=0.01)


@pytest.mark.parametrize(
    ("solids", "source", "listener"),
    [
        (
            {
                "buildings": [
                    {"name": "b", "footprint": [[0, 0], [10, 0], [10, 10], [0, 10]], "height": 6}
                ]
            },
            (-3e-4, -3e-4, 3.0),
            (12.0, -5e-4, 1.0),
        ),
        (
            {
                "barriers": [
                    {"name": "w", "path": [[-10, 0], [10, 0]], "height": 3},
                    {"name": "v", "path": [[0, 0], [0, 10]], "height": 3},
                ]
            },
            (3e-4, -3e-4, 3e-4),
            (5.0, -5.0, 1.5),
        ),
    ],
)
def test_no_reflection_at_an_outer_corner_from_points_by_its_planes(
    solids, source, listener
) -> None:
    """A source 0.3 mm from the planes of two walls of a building and a listener 0.5 mm
    from one of them past its end, or a source 0.3 mm from the planes of both barriers
    of a T, on the side away from its stem, and 0.3 mm over the ground, in a model 10 km
    wide and in one 200 m wide: the same paths, and early levels to 0.01 dB with
    ``first_order`` and to 0.05 dB without diffraction (where a reflection off the wall
    at grazing incidence and then off the ground, 37 to 58 dB under the direct sound,
    which the wide model joins into one corner at the wall's foot, is 3 dB weaker). The
    paths' tolerance is 1 mm in the wide model, and a corner's planes took a side only
    from a point before or after it farther than that from them: a path reflected off
    the roof from under the eave, or off the stem of the T at the junction from the side
    where the stem has no faces."""
    for diffraction, within in (("none", 0.05), ("first_order", 0.01)):
        found = []
        for half in (5000.0, 100.0):
            geometry = {"ground": {"extent": [[-half, -half], [half, half]], "sky": 40}, **solids}
            scene = outdoor(
                geometry,
                sources=[source],
                listeners=[listener],
                early={"method": "image_sources", "max_order": 3, "diffraction": diffraction},
            )
            found.append(early_paths(Room.from_scene(scene)))
        wide, narrow = found
        assert len(wide.order) == len(narrow.order)
        level = 10 * np.log10(total_energy(wide) / total_energy(narrow))
        assert np.abs(level).max() < within


def test_a_way_through_a_sharp_corner_of_three_planes_reflects_there() -> None:
    """A roof, the wall of a taller building along its edge and a barrier that stands on
    the roof at 80 degrees to that wall make a corner on the sound's side, though the
    roof ends at the wall: straight across the line where the roof meets the barrier,
    on the sound's side of it, lies the taller building, and the roof goes on behind
    the barrier. For a listener in line with the image of the source in the three and
    that corner, the path that reflects off all three right there stands, as the one
    that reflects off them one after the other does 0.1 mm beside. (The levels are not
    compared: with the same paths of the same energies they are up to 1.1 dB apart
    below 500 Hz in line and beside, with this rule as without it.)"""
    corner = np.array([10.0, 5.0, 6.0])
    along = np.radians(170.0)
    geometry = {
        "buildings": [
            {"name": "low", "footprint": [[0, 0], [10, 0], [10, 20], [0, 20]], "height": 6},
            {"name": "tall", "footprint": [[10, 0], [14, 0], [14, 20], [10, 20]], "height": 12},
        ],
        "barriers": [
            {
                "name": "b",
                "path": [[10, 5], [10 + 8 * np.cos(along), 5 + 8 * np.sin(along)]],
                "height": 9,
            }
        ],
    }
    source = corner + (2.5 * np.cos(np.radians(130.0)), 2.5 * np.sin(np.radians(130.0)), 1.0)
    image = source
    for normal in ((-np.sin(along), np.cos(along), 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        image = image - 2.0 * ((image - corner) @ np.array(normal)) * np.array(normal)
    listener = corner + 1.5 * (corner - image)
    (exact, _), (beside, _) = _in_line_and_beside(
        geometry, tuple(source), listener, (0, 0, 1e-4), "none", max_order=3
    )

    def near(paths, within):
        return sum(
            1
            for i in range(len(paths))
            if paths.edge[i] < 0
            and paths.order[i] == 3
            and all(math.dist(point, corner) < within for point in paths.points[i][1:4])
        )

    assert near(exact, 1e-6) == 1
    assert near(beside, 1e-3) == 1 and near(beside, 1e-6) == 0
    assert len(exact) == len(beside)


def test_a_way_through_an_outer_corner_of_three_planes_is_no_reflection() -> None:
    """A taller building that stands on a roof, a source and a listener in line with the
    foot of one of its upright corners, all three on the roof's side of both walls: the
    roof lies there, but each wall goes on from the corner only behind the other, so no
    path reflects off the three or off two of them at that point. Without the edge
    diffraction the level is within 0.005 dB of the one 0.1 mm beside; with the paths
    that had stood in line it had been up to 1.6 dB from it."""
    geometry = {
        "buildings": [
            {"name": "low", "footprint": [[0, 0], [30, 0], [30, 20], [0, 20]], "height": 6},
            {"name": "tall", "footprint": [[10, 5], [14, 5], [14, 15], [10, 15]], "height": 12},
        ]
    }
    (exact, level), (beside, level_beside) = _in_line_and_beside(
        geometry, (9, 4, 7), (8, 3, 8), (1e-4, 0, 0), "none", max_order=3
    )
    assert _at_one_point(exact) == 0
    assert len(exact) == len(beside)
    assert np.abs(level - level_beside).max() < 0.005


def _foot_of_a_wall():
    """The trace model of ``BLOCK`` on its ground, the planes of its wall x = 0, of the
    ground and of its roof, and the early paths' tolerance."""
    room = Room.from_scene(outdoor({"ground": {"margin": 60}, "buildings": [BLOCK]}))
    model = TraceModel.from_room(room)
    tolerance = early._EDGE_TOLERANCE * float(np.ptp(room.mesh.vertices, axis=0).max())
    upright = np.abs(model.plane_normal[:, 0]) > 0.99
    flat = np.abs(model.plane_normal[:, 2]) > 0.99
    (wall,) = np.flatnonzero(upright & (np.abs(model.plane_offset) < 1e-9))
    (ground,) = np.flatnonzero(flat & (np.abs(model.plane_offset) < 1e-9))
    (roof,) = np.flatnonzero(flat & (np.abs(np.abs(model.plane_offset) - 6.0) < 1e-9))
    return model, wall, ground, roof, tolerance


@pytest.mark.parametrize("rounding", [0.0, 1e-13, -1e-13])
@pytest.mark.parametrize("aside", [0.0, 1e-6, -1e-6])
def test_a_plane_a_path_runs_in_judges_no_corner(rounding, aside) -> None:
    """A path that runs in the ground's plane and reflects at the foot of a building's
    wall, as for a source and a listener nearer the ground than the paths' tolerance:
    the ground's plane has no side the path keeps to, so the wall's faces are not judged
    by it, whichever side of that plane the rounding puts the point on, and the corner
    stands as one. The same for a point a micrometre before or behind the wall's plane,
    nearer than the paths' tolerance and farther than the intersection test's least
    distance."""
    model, wall, ground, _, tolerance = _foot_of_a_wall()
    # Two rows, the second with room for a third plane that it does not have.
    planes = np.array([[wall, ground, -1], [wall, ground, -1]])
    fronts = np.zeros((2, 3, 3))
    fronts[:, 0] = (-1.0, 0.0, 0.0)
    point = np.array([[aside, 5.0, rounding]] * 2)
    assert early._inner_corner(model, planes[:1, :2], fronts[:1, :2], point[:1], tolerance)[0]
    assert early._inner_corner(model, planes, fronts, point, tolerance).all()
    # With a side to keep to, the wall's faces count: above the ground the corner is
    # one, and below it, where the wall has no face, it is none.
    fronts[:, 1] = ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
    assert early._inner_corner(model, planes, fronts, point, tolerance).tolist() == [True, False]


def test_a_corner_a_hair_past_a_walls_end_stands() -> None:
    """A point 3 um past the end of a wall along its foot, nearer than the paths'
    tolerance, so that a path may reflect off the wall and the ground there: the wall's
    face, within reach of the point, lies above the ground as the sound does, and the
    corner stands. With the sound below the ground, where the wall has no face, it
    does not: 3 um before the wall's end, or past it as far as the wall's face is
    within reach, twice the paths' tolerance and the least distance its edges are
    taken with. Farther out no face of the wall tells of the corner, and it stands."""
    model, wall, ground, _, tolerance = _foot_of_a_wall()
    least = model.bvh.min_distance
    past = [3e-6, -3e-6, 3e-6, 2.0 * tolerance + 0.5 * least, 2.0 * tolerance + 2.0 * least]
    planes = np.array([[wall, ground]] * 5)
    fronts = np.array([[[-1.0, 0.0, 0.0], [0.0, 0.0, up]] for up in (1.0, -1.0, -1.0, -1.0, -1.0)])
    point = np.array([[0.0, 10.0 + by, 0.0] for by in past])
    stands = early._inner_corner(model, planes, fronts, point, tolerance)
    assert stands.tolist() == [True, False, False, False, True]


def test_a_corner_is_judged_from_the_planes_not_from_the_point() -> None:
    """A point that reflects off a roof's edge over a wall may lie up to the paths'
    tolerance off either plane. For sound over the roof the wall has no face above the
    roof's edge, with the point 3 um or 0.98 of the tolerance below the roof's plane,
    on the wall's face, as with it in the plane; and for sound below the ground the
    wall has none there with the point 3 um above the ground. An opening is no face:
    where the ground ends at the open side of the scene, that side tells nothing of a
    corner with the ground, whichever side of the ground the sound is on."""
    model, wall, ground, roof, tolerance = _foot_of_a_wall()
    over = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for below in (0.0, 3e-6, 0.98 * tolerance):
        point = np.array([[0.0, 5.0, 6.0 - below]])
        assert not early._inner_corner(
            model, np.array([[wall, roof]]), over[None], point, tolerance
        )[0]
    under = np.array([[[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]])
    point = np.array([[0.0, 5.0, 3e-6]])
    assert not early._inner_corner(model, np.array([[wall, ground]]), under, point, tolerance)[0]
    upright = np.abs(model.plane_normal[:, 0]) > 0.99
    (side,) = np.flatnonzero(upright & (np.abs(np.abs(model.plane_offset) - 70.0) < 1e-9))
    assert model.transparent[model.face_plane == side].all()
    point = np.array([[70.0, 5.0, 0.0]])
    for up in (1.0, -1.0):
        inside = np.array([[[-1.0, 0.0, 0.0], [0.0, 0.0, up]]])
        assert early._inner_corner(model, np.array([[side, ground]]), inside, point, tolerance)[0]


@pytest.mark.parametrize(
    ("geometry", "source", "listener"),
    [
        (
            {
                "barriers": [
                    {"name": "a", "path": [[0, 0], [600, 0]], "height": 4},
                    {"name": "c", "path": [[0, 0], [0, 600]], "height": 4},
                ]
            },
            (-2, -2, 1.5),
            (-4, -4, 2),
        ),
        (
            {
                "buildings": [
                    {
                        "name": "long",
                        "footprint": [[0, 0], [600, 0], [600, 600], [0, 600]],
                        "height": 4,
                    }
                ]
            },
            (-5, -5, 1.5),
            (-10, -10, 1.5),
        ),
    ],
    ids=["outside two barriers", "upright corner"],
)
def test_a_way_through_an_outer_edge_of_long_low_faces_is_no_reflection(
    geometry, source, listener
) -> None:
    """The same in line with the upright edge where two barriers 600 m long and 4 m
    high meet, outside them, and with the upright corner of a building as long and as
    low: a face that long ends where its edge is, not farther out for its length (a
    point's test against a triangle leaves the more room past its short edge the longer
    the triangle is). With the path, the level in line had been up to 1.1 and 0.7 dB
    more than 0.1 mm beside; it is within 0.005 dB of it."""
    (exact, level), (beside, level_beside) = _in_line_and_beside(
        geometry, source, listener, (1e-4, 0, 0), "none"
    )
    assert _at_one_point(exact) == 0
    assert len(exact) == len(beside)
    assert np.abs(level - level_beside).max() < 0.005


def _room_with_a_pyramid(path: Path) -> dict:
    """The geometry of a room 20 by 20 by 10 m with a square pyramid on its floor, 4 m
    wide and 3 m high with its tip at (10, 10, 3), written as a mesh to ``path``."""
    corners = [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)]
    corners += [(x, y, 10) for x, y, _ in corners]
    corners += [(8, 8, 0), (12, 8, 0), (12, 12, 0), (8, 12, 0), (10, 10, 3)]
    corners = np.array(corners, dtype=float)
    faces = []
    for a, b, c, d in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)):
        faces += [(a, b, c), (a, c, d)]
    for i in range(4):
        j = (i + 1) % 4
        faces += [(i, j, 8 + j), (i, 8 + j, 8 + i), (8 + i, 8 + j, 12)]
    lines = [f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in corners]
    for a, b, c in faces:  # each with its normal into the room
        normal = np.cross(corners[b] - corners[a], corners[c] - corners[a])
        if normal @ (np.array([10.0, 10.0, 9.0]) - corners[[a, b, c]].mean(axis=0)) < 0:
            b, c = c, b
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    path.write_text("\n".join(lines) + "\n")
    return {"file": str(path)}


def test_a_way_through_the_tip_of_a_pyramid_is_no_reflection(tmp_path: Path) -> None:
    """A pyramid on a room's floor: each slope's face lies below the planes of the
    slopes beside it, and at the tip it is narrower than a right angle between its two
    ridges. A listener in line with the tip and the image of the source in two slopes
    side by side hears no reflection off the two at the tip, as 0.1 mm beside (without
    the edge diffraction within 0.005 dB; with that path the level in line had been up
    to 0.36 dB more). From inside the pyramid the same two slopes make a corner at the
    tip."""
    geometry = _room_with_a_pyramid(tmp_path / "pyramid.obj")
    tip = np.array([10.0, 10.0, 3.0])
    out = np.array([[0.0, -3.0, 2.0], [3.0, 0.0, 2.0]]) / math.sqrt(13.0)
    source = np.array([12.5, 7.0, 5.0])
    image = source
    for normal in out:
        image = image - 2.0 * ((image - tip) @ normal) * normal
    listener = tip + 3.0 * (tip - image) / math.dist(tip, image)
    paths, levels = [], []
    for moved in (0.0, 1e-4):
        scene = outdoor(
            geometry,
            sources=[source],
            listeners=[listener + (moved, 0.0, 0.0)],
            early={"method": "image_sources", "max_order": 2, "diffraction": "none"},
        )
        paths.append(early_paths(Room.from_scene(scene)))
        levels.append(10 * np.log10(total_energy(paths[-1])))
    assert _at_one_point(paths[0]) == 0
    assert len(paths[0]) == len(paths[1])
    assert np.abs(levels[0] - levels[1]).max() < 0.005

    room = Room.from_scene(outdoor(geometry, sources=[source], listeners=[listener]))
    model = TraceModel.from_room(room)
    tolerance = early._EDGE_TOLERANCE * float(np.ptp(room.mesh.vertices, axis=0).max())
    slopes = [
        int(np.flatnonzero(np.abs(np.abs(model.plane_normal @ normal) - 1.0) < 1e-9)[0])
        for normal in out
    ]
    for side, stands in ((1.0, False), (-1.0, True)):
        corner = early._inner_corner(
            model, np.array([slopes]), side * out[None], tip[None], tolerance
        )
        assert corner.tolist() == [stands]


def test_planes_nearly_parallel_do_not_judge_each_other() -> None:
    """A barrier with a bend of half a degree: twice the paths' tolerance from a point
    on the bend, neither part's plane lies farther from the other's than the room left
    round a plane, so neither tells where the other's faces are, and a point there
    stands as a corner of the two as before, also a micrometre behind them."""
    bent = {"name": "bent", "path": [[0, 0], [5, 0], [10, 0.04]], "height": 4}
    room = Room.from_scene(outdoor({"ground": {"margin": 60}, "barriers": [bent]}))
    model = TraceModel.from_room(room)
    tolerance = early._EDGE_TOLERANCE * float(np.ptp(room.mesh.vertices, axis=0).max())
    along = np.abs(model.plane_normal[:, 1]) > 0.99
    (first,) = np.flatnonzero(along & (np.abs(model.plane_offset) < 1e-9))
    (second,) = np.flatnonzero(along & (np.abs(model.plane_normal[:, 0]) > 1e-3))
    fronts = np.array([[[0.0, -1.0, 0.0], [0.008, -1.0, 0.0]]])
    for behind in (0.0, 1e-6):
        point = np.array([[5.0, behind, 2.0]])
        assert early._inner_corner(model, np.array([[first, second]]), fronts, point, tolerance)[0]


def test_arcs_meet_where_they_have_a_direction_in_common() -> None:
    """``early._arcs_meet``: the arcs round +x and round +y that reach 60 degrees to
    either side overlap, and those that reach 26 degrees do not; arcs that are each the
    whole circle meet; none meets an empty one, be the others cut or whole; the arcs
    round +x and -x of 60 degrees do not meet, and do once one of them is not counted;
    the half circles round +x, +y and -x share the one direction +y; the one direction
    +x lies in the half circle round +y; the arcs round +y and -x of 60 degrees overlap.
    Arcs round the two diagonals above the x axis overlap where they reach 53 degrees
    and not where they reach 37."""
    across = np.array([[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]] * 10)
    least = np.array(
        [
            [0.5, 0.5, -2.0],
            [0.9, 0.9, -2.0],
            [-2.0, -2.0, -2.0],
            [0.5, 1.5, -2.0],
            [-2.0, 1.5, -2.0],
            [0.5, -2.0, 0.5],
            [0.5, -2.0, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, -2.0],
            [-2.0, 0.5, 0.5],
        ]
    )
    use = np.ones((10, 3), dtype=bool)
    use[6, 2] = False
    meet = early._arcs_meet(across, least, use)
    assert meet.tolist() == [True, False, True, False, False, False, True, True, True, True]
    slant = math.sqrt(0.5)
    across = np.array([[(slant, slant), (-slant, slant)]] * 2)
    meet = early._arcs_meet(across, np.array([[0.6, 0.6], [0.8, 0.8]]), np.ones((2, 2), dtype=bool))
    assert meet.tolist() == [True, False]


def test_a_corners_side_is_the_farther_neighbours() -> None:
    """``early._turn`` takes the side of each of a corner's planes that a path keeps to
    from the farther of the points before and after the corner, and none where that lies
    within the intersection test's least distance of the plane. A path in the ground's
    plane, which the rounding puts a hair on either side of it before and after the foot
    of a wall, turns there. One from a point in a roof's plane to the roof's edge over a
    wall and on to a point above the roof does not: the wall has no face above the
    roof."""
    model, wall, ground, roof, tolerance = _foot_of_a_wall()
    counted = np.column_stack([~model.transparent, model.partition])

    def turns(planes, before, point, after):
        point = np.array([point], dtype=float)
        faces = [[early._face_at(model, plane, point[0], tolerance) for plane in planes]]
        assert min(faces[0]) >= 0
        return early._turn(
            model,
            np.array([planes]),
            np.array(faces),
            np.array([before], dtype=float),
            point,
            np.array([after], dtype=float),
            tolerance,
            counted,
        )[0][0]

    for before, after in ((1e-13, -2e-13), (-2e-13, 1e-13)):
        assert turns([wall, ground], (-3, 4, before), (0, 5, 0), (-4, 7, after))
    for before, after in ((6 + 1e-13, 9), (9, 6 + 1e-13), (6 - 1e-13, 9)):
        assert not turns([wall, roof], (3, 4, before), (0, 5, 6), (4, 7, after))


def test_inside_corners_of_buildings_do_not_diffract() -> None:
    """An L-shaped building: the corner where its wings meet points into it."""
    plan = [[0, 0], [20, 0], [20, 10], [10, 10], [10, 20], [0, 20]]
    for footprint in (plan, plan[::-1]):
        scene = outdoor(
            {"ground": {}, "buildings": [{"name": "l", "footprint": footprint, "height": 5}]},
            sources=[(30, 30, 1.5)],
            listeners=[(-10, -10, 1.5)],
        )
        edges = find_edges(TraceModel.from_room(Room.from_scene(scene)))
        ends = np.stack([edges.start, edges.end], axis=1)
        inner = np.all(np.isclose(ends[:, :, :2], [10, 10]), axis=(1, 2))
        assert not inner.any()
        # Five outer corners and six roof edges.
        assert len(edges) == 11


def test_buildings_that_touch() -> None:
    scene = outdoor(
        {
            "ground": {},
            "buildings": [
                {"name": "a", "footprint": HALL, "height": 6},
                {"name": "b", "footprint": [[10, 0], [20, 0], [20, 8], [10, 8]], "height": 9},
            ],
        },
        sources=[(-5, 4, 2)],
        listeners=[(25, 4, 1.5)],
    )
    room = Room.from_scene(scene)
    assert room.report.closed and not room.problems()
    assert room.outdoor.building_at([[5, 4, 2], [15, 4, 8], [15, 4, 10]]) == ["a", "b", None]


def test_coverage_maps_cover_the_ground_round_the_buildings() -> None:
    scene = outdoor(
        {"ground": {"margin": 10}, "buildings": [{"name": "hall", "footprint": HALL, "height": 6}]},
        sources=[(-5, 4, 1.5)],
        listeners=[(15, 4, 1.5)],
        early={"method": "image_sources", "max_order": 1},
    )
    room = Room.from_scene(scene)
    grid = map_grid(room, spacing=2.0)
    # 30 m by 28 m of ground in 2 m cells, less the 20 under the building.
    assert len(grid) == 15 * 14 - 20
    assert all(b is None for b in room.outdoor.building_at(grid.points))
    result = coverage_map(room, settings=Coverage(spacing=2.0, rays=500))
    assert len(result.grid) == len(grid)
    # Responses run past the latest direct sound, as at listeners.
    assert result.details["map"]["ir_length_s"] > 0.3


def test_indoor_rooms_are_untouched() -> None:
    room = Room.from_scene(Scene.from_yaml(SHOEBOX))
    model = TraceModel.from_room(room)
    assert room.outdoor is None and model.sky is None and model.ground is None
    assert not room.sky_faces().any() and room.shown_faces().all()
