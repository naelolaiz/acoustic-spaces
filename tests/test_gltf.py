"""The glTF export: the file's layout, as the spec and Blender's importer need it."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import early_paths, trace_particles
from acoustic_spaces.gltf import export_gltf
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


def gltf(point) -> np.ndarray:
    """Scene coordinates (Z up) in glTF ones (Y up)."""
    x, y, z = np.asarray(point, dtype=float).T
    return np.stack([x, z, -y], axis=-1)


def read_glb(path: Path) -> tuple[dict, bytes]:
    """The JSON and binary chunks of a .glb file, checking the container on the way."""
    data = path.read_bytes()
    assert struct.unpack_from("<4sII", data) == (b"glTF", 2, len(data))
    length, kind = struct.unpack_from("<II", data, 12)
    assert kind == 0x4E4F534A and length % 4 == 0
    text = data[20 : 20 + length]
    assert text.rstrip(b" ").endswith(b"}")  # padded with spaces
    doc = json.loads(text)
    start = 20 + length
    length, kind = struct.unpack_from("<II", data, start)
    assert kind == 0x004E4942 and length % 4 == 0 and start + 8 + length == len(data)
    binary = data[start + 8 :]
    assert doc["buffers"] == [{"byteLength": len(binary)}]
    return doc, binary


def array(doc: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    start = view["byteOffset"] + accessor.get("byteOffset", 0)
    assert start % 4 == 0
    values = np.frombuffer(binary, dtype, accessor["count"] * width, start)
    assert start + values.nbytes <= view["byteOffset"] + view["byteLength"]
    if "min" in accessor:
        assert accessor["min"] == np.atleast_1d(values.reshape(-1, width).min(axis=0)).tolist()
        assert accessor["max"] == np.atleast_1d(values.reshape(-1, width).max(axis=0)).tolist()
    return values.reshape(-1, width) if width > 1 else values


def check_layout(doc: dict, binary: bytes) -> None:
    """What every file must get right: views, accessors, primitives and the node tree."""
    assert doc["asset"]["version"] == "2.0"
    for view in doc["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(binary)
    for index in range(len(doc["accessors"])):
        array(doc, binary, index)
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            assert "min" in doc["accessors"][primitive["attributes"]["POSITION"]]
            count = doc["accessors"][primitive["attributes"]["POSITION"]]["count"]
            indices = array(doc, binary, primitive["indices"])
            assert indices.max() < count
            assert len(indices) % {1: 2, 4: 3}[primitive["mode"]] == 0
    children = [c for node in doc["nodes"] for c in node.get("children", [])]
    roots = doc["scenes"][doc["scene"]]["nodes"]
    assert sorted(children + roots) == list(range(len(doc["nodes"])))
    for key, items in doc.items():
        assert items != [], f"{key} is an empty array"


def nodes_named(doc: dict) -> dict[str, dict]:
    return {node["name"]: node for node in doc["nodes"]}


def children_of(doc: dict, name: str) -> list[dict]:
    return [doc["nodes"][k] for k in nodes_named(doc)[name].get("children", [])]


def rotate(q, v) -> np.ndarray:
    """``v`` turned by the unit quaternion ``q`` (x, y, z, w)."""
    u, w, v = np.array(q[:3]), q[3], np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def test_room_surfaces_and_materials(tmp_path: Path) -> None:
    room = Room.from_yaml(SHOEBOX)
    out = tmp_path / "room.glb"
    export_gltf(room, out)
    doc, binary = read_glb(out)
    check_layout(doc, binary)
    assert doc["scenes"][0]["name"] == "Shoebox 8 x 6 x 3 m"
    assert "animations" not in doc

    surfaces = children_of(doc, "room")
    assert [node["name"] for node in surfaces] == ["floor", "ceiling", "walls"]
    assert {node["extras"]["material"] for node in surfaces} == set(room.material_names)
    corners = []
    for node in surfaces:
        primitive = doc["meshes"][node["mesh"]]["primitives"][0]
        assert primitive["mode"] == 4
        material = doc["materials"][primitive["material"]]
        assert material["name"] == node["extras"]["material"]
        assert material["alphaMode"] == "BLEND" and material["doubleSided"]
        assert material["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(0.4)
        scene_material = room.scene.materials[material["name"]]
        assert material["extras"]["absorption"] == scene_material.absorption
        assert material["extras"]["scattering"] == scene_material.scattering
        assert material["extras"]["bands_hz"] == list(OCTAVE_BANDS_HZ)
        corners.append(array(doc, binary, primitive["attributes"]["POSITION"]))
    # Z up becomes Y up: the 8 x 6 x 3 m room spans x 0..8, y 0..3 and z -6..0.
    corners = np.vstack(corners)
    (x0, y0, z0), (x1, y1, z1) = room.report.bounds_min, room.report.bounds_max
    assert (x1 - x0, y1 - y0, z1 - z0) == (8, 6, 3)
    np.testing.assert_allclose(corners.min(axis=0), [x0, z0, -y1])
    np.testing.assert_allclose(corners.max(axis=0), [x1, z1, -y0])
    assert sum(node["extras"]["area_m2"] for node in surfaces) == pytest.approx(180.0)

    talker = nodes_named(doc)["talker"]
    np.testing.assert_allclose(talker["translation"], gltf([2.0, 3.0, 1.5]))
    assert talker["extras"] == {
        "kind": "source",
        "directivity": "omni",
        "directivity_index_db": [0.0] * 8,
        "level_db": 94.0,
    }
    assert "children" not in talker  # no aim for an omnidirectional source
    seat = nodes_named(doc)["seat-a"]
    np.testing.assert_allclose(seat["translation"], [6.0, 1.2, -3.0])
    np.testing.assert_allclose(rotate(seat["rotation"], [1.0, 0.0, 0.0]), [-1, 0, 0], atol=1e-12)
    assert [node["name"] for node in children_of(doc, "seat-a")] == ["seat-a aim"]


def test_markers_face_their_way_and_unassigned_surfaces_are_grey(tmp_path: Path) -> None:
    scene = Scene.model_validate(
        {
            "geometry": {"box": [8, 6, 3]},
            "materials": {"wood": {"absorption": [0.1] * 8, "scattering": 0.2}},
            "assignments": {"floor": "wood"},
            "sources": [
                {
                    "name": "horn",
                    "position": [2, 3, 1.5],
                    "orientation": {"yaw": 120, "pitch": -20, "roll": 10},
                    "directivity": "cardioid",
                }
            ],
            "listeners": [
                {"name": "head", "position": [6, 3, 1.2], "orientation": {"yaw": -70, "roll": 30}}
            ],
        }
    )
    room = Room.from_scene(scene)
    out = tmp_path / "box.glb"
    export_gltf(room, out)
    doc, binary = read_glb(out)
    check_layout(doc, binary)

    surfaces = children_of(doc, "room")
    assert [node["name"] for node in surfaces][:3] == ["floor", "ceiling", "wall_1 / walls"]
    assert "material" not in surfaces[1]["extras"]
    grey = doc["materials"][doc["meshes"][surfaces[1]["mesh"]]["primitives"][0]["material"]]
    assert grey["name"] == "no material" and "extras" not in grey
    red, green, blue, _ = grey["pbrMetallicRoughness"]["baseColorFactor"]
    assert red == green == blue
    assert [m["name"] for m in doc["materials"]].count("no material") == 1

    for item in (*scene.sources, *scene.listeners):
        node = nodes_named(doc)[item.name]
        axes = np.array(item.orientation.rotation())  # forward, left and up as columns
        for k in range(3):
            # The node's local axes, in glTF coordinates, are the head's.
            local = gltf(np.eye(3)[k])
            np.testing.assert_allclose(
                rotate(node["rotation"], local), gltf(axes[:, k]), atol=1e-12
            )
        aim = children_of(doc, item.name)[0]
        primitive = doc["meshes"][aim["mesh"]]["primitives"][0]
        arrow = array(doc, binary, primitive["attributes"]["POSITION"])
        assert arrow[:, 0].max() > 5 * np.abs(arrow[:, 1:]).max()  # it points along +x
    assert nodes_named(doc)["horn"]["extras"]["directivity"] == "cardioid"


def test_reflection_paths_are_lines_by_order(tmp_path: Path) -> None:
    room = Room.from_yaml(SHOEBOX)
    paths = early_paths(room, max_order=2)
    out = tmp_path / "paths.glb"
    export_gltf(room, out, paths=paths)
    doc, binary = read_glb(out)
    check_layout(doc, binary)

    orders = children_of(doc, "reflection paths")
    assert [node["name"] for node in orders] == ["paths order 0", "paths order 1", "paths order 2"]
    assert [node["extras"]["paths"] for node in orders] == [1, 6, 18]
    for order, node in enumerate(orders):
        primitive = doc["meshes"][node["mesh"]]["primitives"][0]
        assert primitive["mode"] == 1  # LINES
        points = array(doc, binary, primitive["attributes"]["POSITION"])
        lines = array(doc, binary, primitive["indices"]).reshape(-1, 2)
        assert len(lines) == node["extras"]["paths"] * (order + 1)  # segments per path
        chosen = [p for p, o in zip(paths.points, paths.order, strict=True) if o == order]
        np.testing.assert_allclose(points, gltf(np.vstack(chosen)), rtol=1e-6)
        np.testing.assert_allclose(
            points[lines[0, 0]], gltf(room.scene.sources[0].position), rtol=1e-6
        )
        material = doc["materials"][primitive["material"]]
        assert material["name"] == node["name"]
    white = doc["materials"][doc["meshes"][orders[0]["mesh"]]["primitives"][0]["material"]]
    assert white["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]


def test_particle_animation(tmp_path: Path) -> None:
    room = Room.from_yaml(SHOEBOX)
    particles = trace_particles(room, count=40, duration=0.05, frames=11, seed=2)
    assert particles.source == "talker"
    out = tmp_path / "particles.glb"
    export_gltf(room, out, particles=particles, slow_motion=40.0, db_range=6.0)
    doc, binary = read_glb(out)
    check_layout(doc, binary)

    group = nodes_named(doc)["particles from talker"]
    assert group["extras"]["band_hz"] == 1000 and group["extras"]["slow_motion"] == 40.0
    nodes = children_of(doc, "particles from talker")
    assert len(nodes) == 40 and len({node["mesh"] for node in nodes}) == 1
    (animation,) = doc["animations"]
    assert animation["name"] == "particles from talker"
    assert len(animation["channels"]) == 80

    inputs = {sampler["input"] for sampler in animation["samplers"]}
    assert len(inputs) == 1
    times = array(doc, binary, inputs.pop())
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(times, np.linspace(0.0, 2.0, 11), rtol=1e-6)  # 50 ms -> 2 s

    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(particles.energy)
    expected_scale = np.clip(1.0 + level / 6.0, 0.0, 1.0)
    assert expected_scale[0].min() == 1.0 and expected_scale[-1].min() == 0.0
    index = {doc["nodes"].index(node): k for k, node in enumerate(nodes)}
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        assert sampler["interpolation"] == "LINEAR"
        k = index[channel["target"]["node"]]
        values = array(doc, binary, sampler["output"])
        if channel["target"]["path"] == "translation":
            np.testing.assert_allclose(values, gltf(particles.positions[:, k]), rtol=1e-6)
        else:
            assert channel["target"]["path"] == "scale"
            np.testing.assert_allclose(
                values, np.repeat(expected_scale[:, k, None], 3, 1), rtol=1e-6
            )
    np.testing.assert_allclose(nodes[0]["translation"], gltf(room.scene.sources[0].position))

    with pytest.raises(ValueError, match="positive"):
        export_gltf(room, out, particles=particles, slow_motion=0.0)


def test_trimesh_reads_the_file(tmp_path: Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    room = Room.from_yaml(SHOEBOX)
    out = tmp_path / "room.glb"
    export_gltf(room, out, paths=early_paths(room, max_order=1))
    scene = trimesh.load(out)
    surfaces = ["floor", "ceiling", "walls"]
    area = sum(scene.geometry[scene.graph[name][1]].area for name in surfaces)
    assert area == pytest.approx(room.report.surface_area_m2)
    bounds = np.vstack([scene.geometry[scene.graph[name][1]].bounds for name in surfaces])
    np.testing.assert_allclose(bounds.min(axis=0), [0.0, 0.0, -6.0], atol=1e-6)
    np.testing.assert_allclose(bounds.max(axis=0), [8.0, 3.0, 0.0], atol=1e-6)
