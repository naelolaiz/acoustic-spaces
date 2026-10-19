"""Outdoor scenes: the ground, the open sky over it, buildings and barriers."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import early_paths
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
    assert np.all(paths.edge >= 0)  # the screen hides the direct sound and the ground
    over = math.dist([0, 0, 1], [5, 0, 3]) + math.dist([5, 0, 3], [15, 0, 1.5])
    assert paths.distance_m.min() == pytest.approx(over)


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
