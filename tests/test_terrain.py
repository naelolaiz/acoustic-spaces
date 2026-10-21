"""Terrain: uneven ground from a height grid, and buildings from a map."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.cli import main
from acoustic_spaces.engine import early_paths, insertion
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.coverage import coverage_map, map_grid
from acoustic_spaces.engine.crests import (
    CREST,
    knife_edge_loss,
    taut_string,
)
from acoustic_spaces.engine.diffraction import find_edges
from acoustic_spaces.engine.model import TraceModel, _ground_faces
from acoustic_spaces.geometry.outdoor import TerrainSurface, outdoor_mesh
from acoustic_spaces.geometry.terrain import (
    HeightGrid,
    read_footprints,
    read_height_grid,
    write_asc,
)
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.scene import Coverage

HARD = {"hard": {"absorption": [0.2] * 8, "scattering": 0.0}}
SIMULATION = {
    "ir_length_s": 0.3,
    "seed": 1,
    "early": {"method": "image_sources", "max_order": 1},
    "late": {"rays": 1000},
}


#: A ground that reflects nothing worth hearing: the sound over the crests alone.
SOFT = {"absorption": [0.999] * 8, "scattering": 0.0}
#: A rigid ground, the exact solutions' wedge.
RIGID = {"absorption": [0.0] * 8, "scattering": 0.0}


def scene(
    geometry, sources=((5, 50, 2),), listeners=((90, 50, 1.5),), material=None, **simulation
) -> Scene:
    return Scene.model_validate(
        {
            "geometry": geometry,
            "materials": HARD if material is None else {"hard": material},
            "default_material": "hard",
            "sources": [{"name": f"s{k}", "position": list(p)} for k, p in enumerate(sources)],
            "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
            "simulation": {**SIMULATION, **simulation},
        }
    )


def grid_rows(fn, size: int = 11, spacing: float = 10.0) -> list[list[float]]:
    """Heights ``fn(x, y)`` on a square grid from (0, 0), ``size`` points a side."""
    xs = [k * spacing for k in range(size)]
    return [[float(fn(x, y)) for x in xs] for y in xs]


def ridge(x: float, y: float) -> float:
    """A hill with a ridge along y at x = 50, slopes of 0.6, 20 m high."""
    return max(0.0, 20.0 - 0.6 * abs(x - 50.0))


HILL = {"terrain": {"heights": grid_rows(ridge), "spacing": 10}, "sky": 40}


# -- the grid --------------------------------------------------------------------------


def test_heights_between_the_nodes_are_bilinear_and_past_the_edge_the_edge() -> None:
    grid = HeightGrid((10, 20), 5.0, [[0, 1, 2], [2, 3, 4]])
    assert grid.extent == (10, 20, 20, 25)
    assert grid.height_at([[10, 20], [15, 20], [20, 25]]).tolist() == [0, 1, 4]
    assert grid.height_at([[12.5, 22.5]]) == pytest.approx([1.5])
    assert grid.height_at([[-100, -100], [100, 100], [15, 100]]).tolist() == [0, 4, 3]
    assert grid.height_at([[12.5, 22.5, 99.0]]) == pytest.approx([1.5])  # z is ignored
    assert grid.lowest_over((11, 21, 14, 24)) == 0 and grid.highest_over((11, 21, 14, 24)) == 3
    scaled = grid.scaled(2.0, 1.0)
    assert scaled.extent == (20, 40, 40, 50) and scaled.heights.tolist()[1] == [5, 7, 9]
    assert grid.shifted(-10, -20).extent == (0, 0, 10, 5)
    with pytest.raises(ValueError, match="at least two rows"):
        HeightGrid((0, 0), 1.0, [[0, 1]])
    with pytest.raises(ValueError, match="must be a number"):
        HeightGrid((0, 0), 1.0, [[0, 1], [math.nan, 1]])


def test_esri_ascii_grids_are_read_from_the_north_row_down(tmp_path: Path) -> None:
    text = "ncols 3\nnrows 2\nxllcorner 100\nyllcorner 200\ncellsize 10\nNODATA_value -9999\n"
    text += "4 5 6\n1 2 3\n"
    path = tmp_path / "heights.asc"
    path.write_text(text)
    grid = read_height_grid(path)
    assert grid.origin == (105.0, 205.0) and grid.spacing == 10.0
    assert grid.heights.tolist() == [[1, 2, 3], [4, 5, 6]]
    centred = tmp_path / "centred.asc"
    centred.write_text(text.replace("xllcorner", "xllcenter").replace("yllcorner", "yllcenter"))
    assert read_height_grid(centred).origin == (100.0, 200.0)
    for bad, message in (
        (text.replace("4 5 6", "4 -9999 6"), "NODATA"),
        (text.replace("1 2 3\n", "1 2\n"), "need 6 heights, the file holds 5"),
        (text.replace("cellsize 10\n", ""), "lacks cellsize"),
        (text.replace("4 5 6", "4 five 6"), "no number"),
    ):
        path.write_text(bad)
        with pytest.raises(ValueError, match=message):
            read_height_grid(path)
    out = tmp_path / "out.asc"
    write_asc(out, grid)
    again = read_height_grid(out)
    assert again.origin == grid.origin and again.heights.tolist() == grid.heights.tolist()


def test_xyz_points_make_a_grid_in_any_order(tmp_path: Path) -> None:
    path = tmp_path / "heights.xyz"
    lines = ["x, y, z", "# a comment"]
    for j in range(3):
        for i in range(4):
            lines.append(f"{10 + 2 * i}, {5 + 2 * j}, {i + 10 * j}")
    path.write_text("\n".join(lines[:2] + list(reversed(lines[2:]))) + "\n")
    grid = read_height_grid(path)
    assert grid.origin == (10.0, 5.0) and grid.spacing == 2.0 and grid.shape == (4, 3)
    assert grid.heights.tolist()[2] == [20, 21, 22, 23]
    path.write_text("\n".join(lines[:-1]) + "\n")  # one point missing
    with pytest.raises(ValueError, match="lacks 1 of them"):
        read_height_grid(path)
    path.write_text("\n".join([*lines, "10.3 5 0"]) + "\n")  # off the grid
    with pytest.raises(ValueError, match="not on a regular grid"):
        read_height_grid(path)
    with pytest.raises(ValueError, match="Esri ASCII grid"):
        read_height_grid(tmp_path / "heights.tif")


# -- buildings from a map ----------------------------------------------------------


def geojson(features: list) -> dict:
    return {"type": "FeatureCollection", "features": features}


def feature(coordinates, properties, kind="Polygon") -> dict:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": kind, "coordinates": coordinates},
    }


SQUARE = [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]


def test_footprints_come_from_polygons_and_multipolygons(tmp_path: Path) -> None:
    path = tmp_path / "buildings.geojson"
    path.write_text(
        json.dumps(
            geojson(
                [
                    feature(SQUARE, {"name": "hall", "height": 9}),
                    feature(
                        [[[20, 0], [30, 0], [30, 10], [20, 10]], [[22, 2], [28, 2], [28, 8]]],
                        {"height": "6.5"},
                    ),
                    feature(
                        [[[[40, 0], [50, 0], [50, 10]]], [[[60, 0], [70, 0], [70, 10]]]],
                        {"name": "twins", "height": 4},
                        "MultiPolygon",
                    ),
                    feature([[0, 0], [1, 1]], {"height": 4}, "LineString"),
                    feature(SQUARE, {"name": "unknown"}),
                ]
            )
        )
    )
    with pytest.raises(ValueError, match="has no property 'height'"):
        read_footprints(path)
    found = read_footprints(path, default_height=3.0)
    assert [(f.name, f.height, len(f.footprint), f.holes) for f in found] == [
        ("hall", 9.0, 4, 0),
        ("", 6.5, 4, 1),
        ("twins_1", 4.0, 3, 0),
        ("twins_2", 4.0, 3, 0),
        ("unknown", 3.0, 4, 0),
    ]
    assert [f.height for f in read_footprints(path, height=7.0)] == [7.0] * 5
    assert [f.name for f in read_footprints(path, name="", default_height=1)] == [""] * 5
    path.write_text(json.dumps(feature(SQUARE, {"height": 5})))
    assert len(read_footprints(path)) == 1
    for data, message in (
        ({"type": "Point"}, "not GeoJSON buildings"),
        (geojson([feature(SQUARE, {"height": "tall"})]), "is no number"),
        (geojson([feature(SQUARE, {"height": -1})]), "must be positive"),
        (geojson([feature([[[0, 0], [1, 1]]], {"height": 3})]), "at least three corners"),
        (
            geojson(
                [feature([[[-3.70, 40.41], [-3.70, 40.4101], [-3.7001, 40.4101]]], {"height": 3})]
            ),
            "longitude and latitude",
        ),
    ):
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match=message):
            read_footprints(path)
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_footprints(path)


def buildings_file(tmp_path: Path, name: str = "map.geojson") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            geojson(
                [
                    feature(
                        [[[30, 30], [40, 30], [40, 40], [30, 40]]], {"name": "hall", "height": 9}
                    ),
                    feature([[[60, 60], [66, 60], [66, 66], [60, 66]]], {"height": 4}),
                ]
            )
        )
    )
    return path


def test_a_scene_takes_its_buildings_from_a_file(tmp_path: Path) -> None:
    path = buildings_file(tmp_path)
    geometry = {
        "ground": {"extent": [[0, 0], [100, 100]], "sky": 30},
        "buildings": [{"footprint": [[10, 10], [14, 10], [14, 14], [10, 14]], "height": 3}],
        "buildings_file": {"file": str(path)},
    }
    s = scene(geometry, sources=[(5, 50, 2)], listeners=[(90, 50, 1.5)])
    assert s.geometry.building_names() == ["building_1", "hall", "building_2"]
    assert s.geometry.describe() == "outdoors, 3 buildings"
    room = Room.from_scene(s)
    assert {"hall_roof", "building_2_wall_1", "buildings"} <= set(room.mesh.label_names())
    assert room.report.closed and not room.problems()
    assert room.outdoor.building_at([[35, 35, 5], [63, 63, 3], [63, 63, 5]]) == [
        "hall",
        "building_2",
        None,
    ]
    # Leaving one out, by name: for an insertion loss, or in the file's settings.
    without = s.without(["building_2"])
    assert without.geometry.buildings_file.without == ["building_2"]
    assert without.geometry.building_names() == ["building_1", "hall"]
    data = s.to_data()
    data["geometry"]["buildings_file"]["without"] = ["nobody"]
    with pytest.raises(ValidationError, match="no building of"):
        Scene.model_validate(data)
    data["geometry"]["buildings_file"] = {"file": str(path), "height": 5, "default_height": 3}
    with pytest.raises(ValidationError, match="default_height goes with a height property"):
        Scene.model_validate(data)
    with pytest.raises(ValidationError, match="add geometry.ground"):
        scene({"box": [5, 4, 3], "buildings_file": {"file": str(path)}})
    with pytest.raises(ValidationError, match="cannot be read"):
        scene({"ground": {}, "buildings_file": {"file": str(tmp_path / "none.geojson")}})
    # The map's origin moves the buildings; the scene file keeps the file's path.
    data = s.to_data()
    data["geometry"]["ground"]["map_origin"] = [30, 30]
    moved = Scene.model_validate(data)
    assert moved.geometry.file_buildings()[0].footprint[0] == (0.0, 0.0)
    assert Path(data["geometry"]["buildings_file"]["file"]) == path


def test_footprints_with_holes_are_built_over_and_said_so(tmp_path: Path) -> None:
    path = tmp_path / "court.geojson"
    path.write_text(
        json.dumps(
            feature(
                [[[20, 20], [40, 20], [40, 40], [20, 40]], [[25, 25], [35, 25], [35, 35]]],
                {"name": "court", "height": 6},
            )
        )
    )
    room = Room.from_scene(
        scene(
            {"ground": {"margin": 20}, "buildings_file": {"file": str(path)}},
            sources=[(10, 30, 2)],
            listeners=[(50, 30, 1.5)],
        )
    )
    assert any("holes (courtyards), which are built over: court" in w for w in room.warnings())
    assert room.outdoor.building_at([[30, 30, 3]]) == ["court"]


# -- terrain in a scene ------------------------------------------------------------


def test_terrain_settings_are_checked() -> None:
    for terrain, message in (
        ({}, "a file or heights, one of them"),
        ({"heights": [[0, 1], [1, 2]]}, "need their spacing"),
        ({"heights": [[0, 1], [1]], "spacing": 1}, "all the rows as long as the first"),
        ({"heights": [[0, 1]], "spacing": 1}, "at least two rows"),
        ({"file": "x.asc", "spacing": 2}, "has its own spacing and origin"),
        ({"file": "x.asc", "origin": [1, 2]}, "has its own spacing and origin"),
    ):
        with pytest.raises(ValidationError, match=message):
            scene({"ground": {"terrain": terrain}})
    with pytest.raises(ValueError, match="cannot be read"):
        Room.from_scene(scene({"ground": {"terrain": {"file": "/nowhere/x.asc"}}}))
    # With a terrain the ground needs no extent: the terrain's is the default.
    s = scene({"ground": HILL})
    assert s.geometry.describe() == "outdoors, terrain"
    assert Room.from_scene(s).outdoor.extent == (0.0, 0.0, 100.0, 100.0)
    with pytest.raises(ValidationError, match="over a terrain are not modelled yet"):
        Scene.model_validate({**s.to_data(), "atmosphere": {"preset": "windy"}})
    still = Scene.model_validate({**s.to_data(), "atmosphere": {"turbulence": "moderate"}})
    assert still.atmosphere.still


def test_a_flat_terrain_is_the_flat_ground() -> None:
    """Heights all alike give the paths the flat ground gives: the direct sound and
    one reflection off the ground, to rounding, from the same triangulated box."""
    where = {"sources": [(5, 50, 4)], "listeners": [(90, 50, 3.5)]}
    flat = scene({"ground": {"extent": [[0, 0], [100, 100]], "sky": 40, "height": 2}}, **where)
    terrain = scene(
        {
            "ground": {
                "terrain": {"heights": grid_rows(lambda x, y: 0.0), "spacing": 10},
                "sky": 40,
                "height": 2,
            }
        },
        **where,
    )
    rooms = [Room.from_scene(s) for s in (flat, terrain)]
    assert all(r.report.closed and not r.problems() for r in rooms)
    assert rooms[1].outdoor.ground == 2.0 and rooms[1].outdoor.sky == 42.0
    assert rooms[1].volume_m3 == pytest.approx(rooms[0].volume_m3)
    paths = [early_paths(r) for r in rooms]
    assert paths[1].order.tolist() == [0, 1]
    assert paths[1].distance_m == pytest.approx(paths[0].distance_m, abs=1e-9)
    assert paths[1].energy == pytest.approx(paths[0].energy, rel=1e-9)
    assert rooms[1].outdoor.ground_at([[0, 0], [33, 71], [100, 100]]).tolist() == [2, 2, 2]
    assert _ground_faces(rooms[1]).sum() == 200
    # The ground's size scatters nothing, as flat ground's does not.
    model = TraceModel.from_room(rooms[1])
    assert np.all(model.surface_size[_ground_faces(rooms[1])] == model.surface_size[0])


def test_a_slope_reflects_as_a_tilted_plane() -> None:
    """Over a plane sloping up 20 % the reflection is off a tilted mirror: its path is
    the image source's in that plane."""
    s = scene(
        {
            "ground": {
                "terrain": {"heights": grid_rows(lambda x, y: 0.2 * x), "spacing": 10},
                "sky": 50,
            }
        },
        sources=[(20, 50, 6)],
        listeners=[(70, 50, 16)],
    )
    room = Room.from_scene(s)
    assert room.report.closed and not room.problems()
    paths = early_paths(room)
    assert paths.order.tolist() == [0, 1]
    normal = np.array([-0.2, 0.0, 1.0]) / math.hypot(0.2, 1.0)
    source = np.array([20.0, 50.0, 6.0])
    image = source - 2.0 * (source @ normal) * normal  # the plane goes through the origin
    assert paths.distance_m[1] == pytest.approx(np.linalg.norm(image - [70, 50, 16]), abs=1e-9)
    assert room.outdoor.ground_at([[20, 50], [70, 50]]).tolist() == pytest.approx([4, 14])
    assert room.outdoor.building_at([[20, 50, 6]]) == [None]


def test_a_ridge_diffracts_and_a_listener_behind_it_hears_over_it() -> None:
    room = Room.from_scene(scene({"ground": HILL}, early={"diffraction": "direct"}))
    assert room.report.closed and not room.problems()
    assert room.volume_m3 == pytest.approx(
        100 * 100 * 60 - 100 * 10 * (2 + 8 + 14 + 20 + 14 + 8 + 2)
    )
    # The terrain's own ridges are no edges of the model: the sound over its crests
    # follows the string pulled taut over its profile (acoustic_spaces.engine.crests).
    assert len(find_edges(TraceModel.from_room(room))) == 0
    paths = early_paths(room)
    # One path over the ridge, bent at the crest: the wedge's wave, with the
    # reflections off the slopes before and after the crest in its terms.
    assert len(paths) == 1 and paths.edge[0] == CREST and paths.diffracted.all()
    assert paths.order[0] == 0 and len(paths.faces[0]) == 0
    over = math.dist([5, 50, 2], [50, 50, 20]) + math.dist([50, 50, 20], [90, 50, 1.5])
    assert paths.distance_m[0] == pytest.approx(over, rel=1e-6)
    np.testing.assert_allclose(paths.points[0], [[5, 50, 2], [50, 50, 20], [90, 50, 1.5]])
    assert paths.direction[0] @ [-1, 0, 0] > 0.9  # from the crest
    air = room.air_attenuation("octave")
    direct = math.dist([5, 50, 2], [90, 50, 1.5])
    level = 10 * np.log10(paths.energy[0] * 4 * math.pi * direct**2 / np.exp(-air * over))
    assert np.all(level < -5.0) and np.all(level > -40.0) and np.all(np.diff(level) < 0)
    # Below the terrain is outside the open air, though above the ground's lowest point.
    buried = Room.from_scene(scene({"ground": HILL}, listeners=[(50, 50, 10)]))
    assert any("below the terrain" in p for p in buried.problems())
    result = simulate(room)
    assert result["s0", "l0"].early.energy.sum() > 0


def test_the_sound_over_a_ridge_is_the_exact_wedges() -> None:
    """With the source and the listener on the slopes, the ridge is the rigid wedge
    the Biot-Tolstoy-Medwin solution is exact for: the engine's level re free field
    against it (validation/edge_diffraction.py's btm_first, over an edge 80 m long),
    within a tenth of a decibel up to 2 kHz and within 1.5 dB at 8 kHz, where the
    engine's uniform theory of diffraction is farthest from it near the faces."""
    exact = {
        ((35, 50, 14), (65, 50, 13)): [
            -7.64,
            -10.44,
            -13.4,
            -16.39,
            -19.41,
            -22.42,
            -25.42,
            -28.42,
        ],  # noqa: E501
        ((30, 40, 9), (70, 55, 9.5)): [
            -9.21,
            -12.1,
            -15.08,
            -18.08,
            -21.09,
            -24.09,
            -27.11,
            -30.12,
        ],  # noqa: E501
    }
    for (source, listener), btm in exact.items():
        room = Room.from_scene(
            scene(
                {"ground": HILL},
                sources=[source],
                listeners=[listener],
                material=RIGID,
                early={"diffraction": "direct"},
            )
        )
        paths = early_paths(room)
        assert len(paths) == 1 and paths.edge[0] == CREST
        air = room.air_attenuation("octave")
        direct = math.dist(source, listener)
        level = 10 * np.log10(
            paths.energy[0] * 4 * math.pi * direct**2 / np.exp(-air * paths.distance_m[0])
        )
        np.testing.assert_allclose(level[:6], btm[:6], atol=0.12)
        np.testing.assert_allclose(level[6:], btm[6:], atol=1.5)


def test_buildings_and_barriers_stand_on_the_terrain() -> None:
    footprint = [[70, 40], [76, 40], [76, 46], [70, 46]]
    s = scene(
        {
            "ground": HILL,
            "buildings": [{"name": "hut", "footprint": footprint, "height": 4}],
            "barriers": [{"name": "fence", "path": [[30, 30], [60, 70]], "height": 2}],
            "vegetation": [
                {"name": "oak", "tree": [85, 85], "crown_diameter": 6, "base": 2, "height": 8}
            ],
        },
        listeners=[(90, 50, 1.5)],
    )
    room = Room.from_scene(s)
    assert room.report.closed and not room.problems()
    layout = room.outdoor
    mesh = room.mesh
    # The hut's roof is its height over the terrain's mean height at its corners, and
    # its walls' feet follow the terrain.
    corners = layout.ground_at(np.array(footprint, dtype=float))
    assert corners.tolist() == pytest.approx([8, 4.4, 4.4, 8])
    roof = float(corners.mean()) + 4.0
    assert layout.buildings[0][2] == pytest.approx(roof)
    hut = room.building_faces()
    assert mesh.triangles[hut][:, :, 2].max() == pytest.approx(roof)
    feet = mesh.triangles[hut].reshape(-1, 3)
    feet = feet[feet[:, 2] < roof - 1e-9]
    assert feet[:, 2] == pytest.approx(layout.ground_at(feet), abs=1e-9)
    assert layout.building_at([[73, 43, 7], [73, 43, roof + 0.1], [73, 43, 5]]) == [
        "hut",
        None,
        None,
    ]
    # The fence is 2 m over the terrain all along, and its foot lies on it.
    fence = np.array([label == ("fence", "barriers") for label in mesh.labels])[mesh.face_labels]
    points = mesh.triangles[fence].reshape(-1, 3)
    ground = layout.ground_at(points)
    above = points[:, 2] - ground
    assert np.all((np.abs(above) < 1e-9) | (np.abs(above - 2.0) < 1e-9))
    assert np.any(np.abs(above - 2.0) < 1e-9) and np.any(np.abs(above) < 1e-9)
    assert ground.max() > 15  # it climbs the hill
    # The oak stands on the terrain under it.
    assert layout.vegetation[0].low == pytest.approx(layout.ground_at([[85, 85]])[0] + 2.0)
    # Edges: the hut's eight outer corners, the fence's two ends and its top, which
    # bends where it crosses the ridge; no foot on the ground, nothing buried, and not
    # the ridge itself (the sound over the terrain's crests is the string's).
    edges = find_edges(TraceModel.from_room(room))
    assert len(edges) == 8 + 4
    with pytest.raises(ValueError, match="rises to z = 20 m, above its roof"):
        Room.from_scene(
            scene(
                {
                    "ground": HILL,
                    "buildings": [
                        {
                            "name": "low",
                            "footprint": [[30, 40], [70, 40], [70, 46], [30, 46]],
                            "height": 3,
                        }
                    ],
                }
            )
        )


def test_the_profile_of_a_line_lies_on_the_surface() -> None:
    rng = np.random.default_rng(3)
    grid = HeightGrid((0, 0), 10.0, rng.uniform(0, 20, (8, 8)))
    surface = TerrainSurface.of(grid, (3.0, 5.0, 65.0, 68.0))
    assert surface.nx == 7 and surface.ny == 7  # 62 m and 63 m across in 10 m cells
    assert surface.xs[-1] == 65.0 and surface.ys[-1] == 68.0
    for a, b in (
        ((4.0, 6.0), (60.0, 50.0)),
        ((10.0, 60.0), (30.0, 10.0)),
        ((20.0, 20.0), (20.0, 61.0)),
    ):
        profile = surface.profile(a, b)
        assert profile[0, :2].tolist() == list(a) and profile[-1, :2].tolist() == list(b)
        assert profile[:, 2] == pytest.approx(surface.height_at(profile), abs=1e-9)
        for t in rng.uniform(0, 1, 50):
            point = np.array(a) + t * (np.array(b) - np.array(a))
            along = np.linalg.norm(profile[:, :2] - a, axis=1)
            expected = np.interp(np.linalg.norm(point - a), along, profile[:, 2])
            assert surface.height_at([point])[0] == pytest.approx(expected, abs=1e-9)
    # Over a whole number of cells the nodes are the grid's, and the surface is the grid
    # there.
    aligned = TerrainSurface.of(grid, (0.0, 0.0, 70.0, 70.0))
    assert aligned.height_at([[10, 10], [60, 60]]) == pytest.approx(
        grid.height_at([[10, 10], [60, 60]])
    )


def test_zones_take_the_terrain_triangles_under_them() -> None:
    s = scene(
        {
            "ground": {
                **HILL,
                "zones": [
                    {"name": "road", "polygon": [[-10, 45], [110, 45], [110, 55], [-10, 55]]}
                ],
            }
        }
    )
    s = Scene.model_validate({**s.to_data(), "assignments": {"road": "hard"}})
    room = Room.from_scene(s)
    assert room.report.closed and not room.problems()
    road = np.array([label == ("road", "zones", "ground") for label in room.mesh.labels])
    road = road[room.mesh.face_labels]
    assert road.sum() == 20  # one strip of cells, two triangles each
    assert np.all(_ground_faces(room)[road])
    model = TraceModel.from_room(room)
    assert model.ground_part is not None and (model.ground_part[road] == 1).all()


def test_the_extent_keeps_to_the_terrain() -> None:
    with pytest.raises(ValueError, match="reaches past the terrain"):
        Room.from_scene(scene({"ground": {**HILL, "extent": [[-10, 0], [100, 100]]}}))
    inside = Room.from_scene(
        scene({"ground": {**HILL, "extent": [[20, 20], [90, 90]]}}, sources=[(25, 50, 2)])
    )
    assert inside.outdoor.extent == (20.0, 20.0, 90.0, 90.0) and inside.report.closed
    # Without an extent the ground is the whole terrain, whatever the margin.
    hut = [{"name": "hut", "footprint": [[70, 40], [76, 40], [76, 46], [70, 46]], "height": 4}]
    room = Room.from_scene(scene({"ground": {**HILL, "margin": 50}, "buildings": hut}))
    assert room.outdoor.extent == (0.0, 0.0, 100.0, 100.0)
    with pytest.raises(ValueError, match="reaches past the edge"):
        Room.from_scene(
            scene(
                {
                    "ground": HILL,
                    "buildings": [
                        {**hut[0], "footprint": [[95, 40], [105, 40], [105, 46], [95, 46]]}
                    ],
                }
            )
        )
    far = {"terrain": {"heights": [[0, 0], [0, 0]], "spacing": 10, "origin": [500000, 4000000]}}
    with pytest.raises(ValueError, match="set ground.map_origin"):
        Room.from_scene(scene({"ground": far}))


def test_terrain_files_and_map_origins(tmp_path: Path) -> None:
    grid = HeightGrid((500000, 4000000), 10.0, grid_rows(ridge))
    path = tmp_path / "hill.asc"
    write_asc(path, grid)
    s = scene(
        {"ground": {"terrain": {"file": str(path)}, "map_origin": [500000, 4000000], "sky": 40}},
    )
    room = Room.from_scene(s)
    assert room.outdoor.extent == (0.0, 0.0, 100.0, 100.0)
    assert room.outdoor.ground_at([[50, 50]]).tolist() == [20.0]
    assert room.outdoor.terrain.describe() == "11 x 11 points, from z = 0 to 20 m (grid 10 m)"
    # Units scale the grid's coordinates and heights, and the ground's height is added.
    data = s.to_data()
    data["geometry"]["units"] = "cm"
    data["geometry"]["ground"]["height"] = 100
    data["sources"][0]["position"] = [50, 500, 100]
    data["listeners"][0]["position"] = [800, 500, 150]
    small = Room.from_scene(Scene.model_validate(data))
    assert small.outdoor.extent == (0.0, 0.0, 1.0, 1.0)
    assert small.outdoor.ground_at([[0.5, 0.5]]).tolist() == pytest.approx([1.2])
    # A scene file next to its terrain refers to it by a relative path.
    yaml = tmp_path / "scene.yaml"
    yaml.write_text(
        "geometry:\n  ground: {terrain: {file: hill.asc}, map_origin: [500000, 4000000], sky: 40}\n"
        "materials: {hard: {absorption: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]}}\n"
        "default_material: hard\nsources: [{name: s, position: [5, 50, 2]}]\n"
        "listeners: [{name: l, position: [80, 50, 1.5]}]\n"
    )
    loaded = Scene.from_yaml(yaml)
    assert loaded.geometry.ground.terrain.file == path
    assert Scene.model_validate(loaded.to_data()).geometry.ground.terrain.file == path


def test_inspect_describes_the_terrain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = buildings_file(tmp_path)
    s = scene(
        {"ground": {**HILL, "extent": [[0, 0], [100, 100]]}, "buildings_file": {"file": str(path)}}
    )
    yaml = tmp_path / "scene.yaml"
    yaml.write_text(json.dumps(s.to_data()))  # JSON is YAML
    assert main(["inspect", str(yaml)]) == 0
    out = capsys.readouterr().out
    assert "terrain: 11 x 11 points, from z = 0 to 20 m (grid 10 m)" in out
    assert "buildings from map.geojson: 2" in out
    assert "geometry: outdoors, terrain, 2 buildings" in out
    assert "2.00 m over the ground" in out and "1.50 m over the ground" in out


def test_coverage_maps_and_insertion_losses_follow_the_terrain(tmp_path: Path) -> None:
    path = buildings_file(tmp_path)
    s = scene(
        {"ground": {**HILL, "extent": [[0, 0], [100, 100]]}, "buildings_file": {"file": str(path)}},
        insertion_loss_of=["hall"],
    )
    room = Room.from_scene(s)
    grid = map_grid(room, spacing=10.0, height=1.5)
    # Points stand 1.5 m over the terrain, not over its lowest point.
    assert grid.points[:, 2] == pytest.approx(room.outdoor.ground_at(grid.points) + 1.5, abs=1e-6)
    assert all(b is None for b in room.outdoor.building_at(grid.points))
    result = insertion.simulate(room)
    assert np.all(np.isfinite(result.insertion_loss[("s0", "l0")]))


# -- sound over the crests ------------------------------------------------------------


def hill(x: float, y: float) -> float:
    """A smooth hill 12 m high across the middle, 15 m wide at half height."""
    return 12.0 * math.exp(-(((x - 50.0) / 15.0) ** 2))


SMOOTH = {"terrain": {"heights": grid_rows(hill, size=21, spacing=5), "spacing": 5}, "sky": 40}


def straight_level(response, air) -> np.ndarray:
    """(bands,) the level of the early sound at the listener, added in pressure in
    every band, in dB re the direct sound in free field over the straight distance,
    the air's attenuation (``air``, per metre per band) taken off each path over its
    own length: over the crests, the one path of order 0 with the ground's reflections
    in its wave; in sight, the direct sound, the nearest crest's wave and the ground's
    reflections."""
    early = response.early
    rows = np.flatnonzero(early.order == 0)
    assert len(rows) >= 1, early.order
    corners = early.points[rows[0]]
    d = float(np.linalg.norm(corners[-1] - corners[0]))
    dry = replace(early, energy=early.energy * np.exp(air[None] * early.distance_m[:, None]))
    return 10 * np.log10(total_energy(dry) * 4 * math.pi * d**2)


def test_the_knife_edge_loss_is_kurze_and_andersons_fit() -> None:
    n = np.array([-1.0, -0.2, -0.1, 0.0, 1.0, 10.0])
    loss = knife_edge_loss(n)
    assert loss[:2].tolist() == [0.0, 0.0]
    assert loss[2] == pytest.approx(5 + 20 * math.log10(0.7927 / math.tan(0.7927)), abs=0.01)
    assert loss[3] == pytest.approx(5.0)
    assert loss[4] == pytest.approx(13.1, abs=0.05)
    assert loss[5] == pytest.approx(5 + 20 * math.log10(math.sqrt(20 * math.pi)), abs=0.01)
    fine = knife_edge_loss(np.linspace(-0.3, 5.0, 2000))
    assert np.all(np.diff(fine) >= -1e-9)  # rises with the Fresnel number, without a jump
    assert np.abs(np.diff(fine)).max() < 0.2


def test_the_taut_string_bends_at_the_crests_only() -> None:
    s = np.arange(5.0)
    assert taut_string(s, np.array([0.0, 1.0, 3.0, 1.0, 0.0])) == [0, 2, 4]
    assert taut_string(s, np.array([2.0, 1.0, 0.0, 1.0, 2.0])) == [0, 4]  # a valley
    assert taut_string(s, np.array([0.0, 1.0, 2.0, 3.0, 4.0])) == [0, 4]  # a plane
    assert taut_string(s, np.array([0.0, 3.0, 0.0, 3.0, 0.0])) == [0, 1, 3, 4]  # two hills
    assert taut_string(s, np.array([0.0, 2.0, 2.0, 2.0, 0.0])) == [0, 1, 3, 4]  # a flat top


def test_a_listener_behind_a_smooth_hill_hears_over_its_crest() -> None:
    s = scene(
        {"ground": SMOOTH},
        sources=[(10, 50, 1.5)],
        listeners=[(90, 50, 1.5), (40, 50, hill(40, 50) + 1.5)],
        early={"diffraction": "direct"},
    )
    room = Room.from_scene(s)
    assert not room.problems()
    assert len(find_edges(TraceModel.from_room(room))) == 0  # too gentle for any edge
    result = simulate(room)
    behind = result["s0", "l0"].early
    assert len(behind) == 1 and behind.edge[0] == CREST and behind.order[0] == 0
    corners = behind.points[0]  # the string lies on the rounded crest from 45 to 55
    assert len(corners) >= 3 and np.all(np.abs(corners[1:-1, 0] - 50) <= 5)
    np.testing.assert_allclose(corners[1:-1, 2], [hill(x, 50) for x in corners[1:-1, 0]])
    assert behind.direction[0] @ [-1, 0, 0] > 0.9  # from the crest
    level = straight_level(result["s0", "l0"], room.air_attenuation("octave"))
    assert np.all(level < -5.0) and np.all(level > -40.0) and level[-1] < level[0] - 15.0
    assert np.all(result["s0", "l0"].relative_to_free_field() < 0.0)
    # Over a soft ground the crest's wave alone: it falls with frequency, near the
    # thin screen's attenuation (Kurze and Anderson's fit, N = 2 delta f / c) with the
    # rounded crest's extra screening at the highest bands. The hard ground's
    # reflections off the flanks add to it, in and out of phase from band to band.
    soft = scene(
        {"ground": SMOOTH},
        sources=[(10, 50, 1.5)],
        listeners=[(90, 50, 1.5)],
        material=SOFT,
        early={"diffraction": "direct"},
    )
    soft_room = Room.from_scene(soft)
    quiet = straight_level(simulate(soft_room)["s0", "l0"], soft_room.air_attenuation("octave"))
    delta = np.linalg.norm(np.diff(corners, axis=0), axis=1).sum() - 80.0
    fresnel = 2.0 * delta * np.array(OCTAVE_BANDS_HZ) / soft_room.speed_of_sound()
    assert np.all(np.diff(quiet) < 0) and np.all(np.abs(quiet + knife_edge_loss(fresnel)) < 4.0)
    assert np.all(level > quiet - 6.0) and np.all(level < quiet + 12.0)
    assert level.mean() > quiet.mean() + 3.0
    # In sight on the slope, the direct sound and the ground's reflection, as usual.
    facing = result["s0", "l1"].early
    assert facing.order[0] == 0 and not facing.diffracted.any()


def test_the_level_is_continuous_across_the_edge_of_a_hills_shadow() -> None:
    listeners = [(70.0, 50.0, hill(70, 50) + h) for h in np.arange(1.5, 20.0, 0.25)]
    s = scene(
        {"ground": SMOOTH},
        sources=[(10, 50, 6)],
        listeners=listeners,
        material=SOFT,  # over a ground that reflects, its reflections would interfere
        early={"diffraction": "direct"},
    )
    room = Room.from_scene(s)
    result = simulate(room)
    # Climbing out of the shadow the level rises to free field without a jump, by about
    # a decibel a step at most up to 250 Hz, and steeply but steadily at the higher
    # bands, whose Fresnel zones are centimetres across and which ripple a little in
    # the lit zone, as the knife edge's exact field does.
    air = room.air_attenuation("octave")
    levels = np.array([straight_level(result["s0", f"l{k}"], air) for k in range(len(listeners))])
    steps = np.diff(levels, axis=0)
    assert np.all(steps[:, :3] > -0.2) and np.all(steps > -1.5)
    assert np.abs(steps[:, :3]).max() < 1.5 and np.abs(steps).max() < 4.0
    assert levels[0, 4] < -8.0
    # Clear of the Fresnel zone at the top from 500 Hz up; in the lowest bands the
    # exact knife edge still gives -0.7 dB at 63 Hz and up to +1.3 dB (its first
    # ripple) at 125 and 250 Hz.
    np.testing.assert_allclose(levels[-1, 3:], 0.0, atol=0.4)
    assert levels[-1, 0] == pytest.approx(-0.67, abs=0.3)
    assert np.all(levels[-1, 1:3] > -0.5) and np.all(levels[-1, 1:3] < 1.6)


def test_over_two_hills_each_crest_bends_the_sound_in_turn() -> None:
    source, listener = (10, 50, 1.5), (130, 50, 1.5)
    levels = {}
    for name, fn in (
        ("both", lambda x, y: hill(x, y) + hill(x - 40, y)),
        ("first", hill),
        ("second", lambda x, y: hill(x - 40, y)),
    ):
        heights = grid_rows(fn, size=29, spacing=5)
        terrain = {"terrain": {"heights": heights, "spacing": 5}, "sky": 40}
        room = Room.from_scene(
            scene(
                {"ground": terrain},
                sources=[source],
                listeners=[listener],
                early={"diffraction": "direct"},
            )
        )
        paths = early_paths(room)
        assert len(paths) == 1 and paths.edge[0] == CREST and paths.order[0] == 0
        x = paths.points[0][1:-1, 0]
        assert np.any(np.abs(x - 50) <= 2.5) == (name != "second")
        assert np.any(np.abs(x - 90) <= 2.5) == (name != "first")
        levels[name] = straight_level(simulate(room)["s0", "l0"], room.air_attenuation("octave"))
    # The second crest bends the sound the first sends over it, wedge after wedge:
    # less gets through than over either hill alone.
    assert np.all(levels["both"] < np.minimum(levels["first"], levels["second"]) - 1.0)
    assert np.all(levels["both"] > -90.0)


def test_a_hill_the_string_is_about_to_touch_screens_the_sound_a_little() -> None:
    """Rising over the second hill, the level falls towards it without a jump."""
    heights = grid_rows(lambda x, y: 0.5 * (hill(x, y) + hill(x - 40, y)), size=29, spacing=5)
    two = {"terrain": {"heights": heights, "spacing": 5}, "sky": 40}
    listeners = [(130.0, 50.0, 1.5 + h) for h in np.arange(0.0, 8.0, 0.25)]
    room = Room.from_scene(
        scene(
            {"ground": two},
            sources=[(10, 50, 14)],
            listeners=listeners,
            material=SOFT,
            early={"diffraction": "direct"},
        )
    )
    result = simulate(room)
    air = room.air_attenuation("octave")
    levels = np.array([straight_level(result["s0", f"l{k}"], air) for k in range(len(listeners))])
    steps = np.abs(np.diff(levels, axis=0))
    assert steps[:, :5].max() < 1.5 and steps.max() < 4.0
    assert levels[0].max() < -1.0  # screened at first
    # At the top the highest bands are back at free field, the lowest still within
    # the hill's Fresnel zone: the exact knife edge gives -1.4 dB there at 63 Hz.
    assert levels[-1, -1] == pytest.approx(0.0, abs=0.05) and levels[-1, 2:].min() > -0.3
    assert levels[-1, 0] == pytest.approx(-1.37, abs=0.4)


def test_a_building_on_the_string_stops_the_sound_over_the_crest() -> None:
    hall = [{"name": "hall", "footprint": [[48, 45], [52, 45], [52, 55], [48, 55]], "height": 3}]
    s = scene(
        {"ground": SMOOTH, "buildings": hall},
        sources=[(10, 50, 1.5)],
        listeners=[(90, 50, 1.5)],
        early={"diffraction": "direct"},
    )
    paths = early_paths(Room.from_scene(s))
    assert len(paths) == 0


def test_a_plane_slope_screens_nothing() -> None:
    slope = {"terrain": {"heights": grid_rows(lambda x, y: 0.2 * x), "spacing": 10}, "sky": 40}
    room = Room.from_scene(
        scene({"ground": slope}, sources=[(5, 50, 3)], listeners=[(90, 50, 19.5)])
    )
    paths = early_paths(room)
    d = math.dist([5, 50, 3], [90, 50, 19.5])
    air = room.air_attenuation("octave")
    assert paths.order[0] == 0
    np.testing.assert_allclose(paths.energy[0], np.exp(-air * d) / (4 * math.pi * d**2), rtol=1e-12)


def test_coverage_maps_see_over_the_hills() -> None:
    s = scene({"ground": SMOOTH}, sources=[(10, 50, 1.5)], early={"diffraction": "direct"})
    result = coverage_map(Room.from_scene(s), settings=Coverage(spacing=10.0, height=1.5))
    level = result.get("s0").values["re_free_field"]
    assert np.all(np.isfinite(level))
    x = result.grid.points[:, 0]
    assert level[x > 60].max() < -1.0 and level[x > 60].mean() < -8.0
    assert level[x < 40].max() > -3.0


def test_terrain_meshes_are_built_from_the_geometry_alone() -> None:
    geometry = scene({"ground": HILL}).geometry
    mesh, layout = outdoor_mesh(geometry)
    assert layout.terrain is not None and layout.ground == 0.0 and layout.sky == 60.0
    assert len(mesh.faces) == 200 + 2 * 40 + 2  # the ground, the sky's walls and its lid
    assert set(mesh.label_names()) == {"ground", "sky"}
