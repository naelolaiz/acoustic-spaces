from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.geometry import (
    RoomMesh,
    check_mesh,
    clean_mesh,
    load_mesh,
    points_inside,
)
from acoustic_spaces.geometry.triangulate import triangulate_polygon

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHOEBOX = EXAMPLES / "shoebox" / "shoebox.obj"


def _area_by_name(mesh: RoomMesh) -> dict[str, float]:
    areas = mesh.face_areas()
    out: dict[str, float] = {}
    for k, label in enumerate(mesh.labels):
        out[label[0] if label else ""] = float(areas[mesh.face_labels == k].sum())
    return out


def _box(lx: float, ly: float, lz: float) -> RoomMesh:
    v = np.array(
        [[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
         [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]],
        dtype=float,
    )  # fmt: skip
    f = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5], [0, 4, 5], [0, 5, 1],
         [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    )  # fmt: skip
    return RoomMesh(v, f, np.zeros(len(f)), [("box",)])


# -- triangulation ------------------------------------------------------------------


def test_triangulate_concave_l_shape_keeps_area() -> None:
    # L-shaped floor of 3 x 3 m with a 1 x 1 m notch: 8 m2.
    pts = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0], [2, 1, 0], [0, 1, 0]], float)
    tris = triangulate_polygon(pts)
    assert len(tris) == len(pts) - 2
    area = sum(0.5 * np.linalg.norm(np.cross(pts[b] - pts[a], pts[c] - pts[a])) for a, b, c in tris)
    assert area == pytest.approx(5.0)


def test_triangulate_keeps_winding_on_vertical_polygon() -> None:
    pts = np.array([[0, 0, 0], [0, 0, 3], [0, 4, 3], [0, 4, 0]], float)
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    for a, b, c in triangulate_polygon(pts):
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        assert np.dot(n, normal) > 0


# -- OBJ ----------------------------------------------------------------------------


def test_obj_shoebox_groups_and_areas() -> None:
    mesh = load_mesh(SHOEBOX)
    assert len(mesh.faces) == 12
    assert _area_by_name(mesh) == pytest.approx({"floor": 48.0, "ceiling": 48.0, "walls": 84.0})


def test_obj_quads_negative_indices_texture_normals_and_labels(tmp_path: Path) -> None:
    obj = tmp_path / "room.obj"
    obj.write_text(
        "# quad floor, materials and groups\n"
        "o hall\n"
        "v 0 0 0\nv 4 0 0\nv 4 5 0\nv 0 5 0\n"
        "vt 0 0\nvn 0 0 1\n"
        "g floor\nusemtl oak\n"
        "f 1/1/1 2/1/1 3/1/1 4/1/1\n"
        "v 0 0 3\nv 4 0 3\n"
        "g front\nusemtl\n"
        "f -6//1 -1//1 -5//1\n",
        encoding="utf-8",
    )
    mesh = load_mesh(obj)
    assert len(mesh.faces) == 3
    assert mesh.labels == [("oak", "floor", "hall"), ("front", "hall")]
    assert _area_by_name(mesh)["oak"] == pytest.approx(20.0)
    assert mesh.label_names() == ["oak", "floor", "hall", "front"]


def test_units_and_up_axis(tmp_path: Path) -> None:
    obj = tmp_path / "tri.obj"
    obj.write_text("v 0 0 0\nv 100 0 0\nv 0 250 0\nf 1 2 3\n", encoding="utf-8")
    mesh = load_mesh(obj, units="cm", up_axis="y")
    # y up -> z up: (x, y, z) becomes (x, -z, y).
    np.testing.assert_allclose(mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 0, 2.5]])


@pytest.mark.parametrize(
    ("content", "message"),
    [("v 0 0 0\nf 1 2 3\n", "missing vertex"), ("# nothing\n", "no faces")],
)
def test_obj_errors(tmp_path: Path, content: str, message: str) -> None:
    obj = tmp_path / "bad.obj"
    obj.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_mesh(obj)


def test_unsupported_suffix_and_units(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load_mesh(tmp_path / "room.skp")
    with pytest.raises(ValueError, match="units"):
        load_mesh(SHOEBOX, units="furlong")
    with pytest.raises(FileNotFoundError):
        load_mesh(tmp_path / "missing.obj")


# -- trimesh formats ----------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".stl", ".ply", ".off", ".glb"])
def test_trimesh_formats(tmp_path: Path, suffix: str) -> None:
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(8.0, 6.0, 3.0))
    path = tmp_path / f"box{suffix}"
    box.export(path)
    mesh = load_mesh(path, up_axis="z")
    cleaned, _, _ = clean_mesh(mesh)
    report = check_mesh(cleaned)
    assert report.closed
    assert report.volume_m3 == pytest.approx(144.0)
    assert report.surface_area_m2 == pytest.approx(180.0)
    assert mesh.label_names()  # every face carries at least the file or node name


def test_glb_is_y_up_by_default(tmp_path: Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(1.0, 3.0, 1.0))  # tall along y
    path = tmp_path / "tall.glb"
    box.export(path)
    mesh = load_mesh(path)
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    np.testing.assert_allclose(extent, [1.0, 1.0, 3.0])


# -- DXF ----------------------------------------------------------------------------


def test_dxf_3dfaces_mesh_and_blocks_by_layer(tmp_path: Path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_3dface([(0, 0, 0), (4, 0, 0), (4, 5, 0), (0, 5, 0)], dxfattribs={"layer": "FLOOR"})
    msp.add_3dface([(0, 0, 3), (4, 0, 3), (4, 5, 3)], dxfattribs={"layer": "CEILING"})
    block = doc.blocks.new("PANEL")
    block.add_3dface([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])  # layer 0
    msp.add_blockref("PANEL", (1, 5, 1), dxfattribs={"layer": "ABSORBER"})
    mesh_entity = msp.add_mesh(dxfattribs={"layer": "STAGE"})
    with mesh_entity.edit_data() as data:
        data.vertices = [(0, 0, 0.5), (2, 0, 0.5), (2, 1, 0.5), (0, 1, 0.5)]
        data.faces = [(0, 1, 2, 3)]
    path = tmp_path / "room.dxf"
    doc.saveas(path)

    mesh = load_mesh(path)
    areas = _area_by_name(mesh)
    assert areas == pytest.approx({"FLOOR": 20.0, "CEILING": 10.0, "ABSORBER": 1.0, "STAGE": 2.0})
    absorber = mesh.triangles[mesh.face_labels == mesh.labels.index(("ABSORBER",))]
    np.testing.assert_allclose(absorber[..., 1], 5.0)  # moved by the block insert


# -- cleanup and checks -------------------------------------------------------------


def test_box_report() -> None:
    mesh, merged, removed = clean_mesh(_box(8, 6, 3))
    report = check_mesh(mesh, merged, removed)
    assert report.closed
    assert report.inconsistent_edges == 0
    assert report.volume_m3 == pytest.approx(144.0)
    assert report.surface_area_m2 == pytest.approx(180.0)
    assert "into the room" in report.summary()


def test_open_mesh_has_boundary_edges_and_no_volume() -> None:
    box = _box(8, 6, 3)
    open_box = RoomMesh(box.vertices, box.faces[2:], box.face_labels[2:], box.labels)
    report = check_mesh(open_box)
    assert not report.closed
    assert report.boundary_edges == 4
    assert report.volume_m3 is None


def test_flipped_face_is_reported() -> None:
    box = _box(8, 6, 3)
    faces = box.faces.copy()
    faces[0] = faces[0, ::-1]
    report = check_mesh(RoomMesh(box.vertices, faces, box.face_labels, box.labels))
    assert report.inconsistent_edges > 0
    assert report.volume_m3 is None


def test_clean_merges_duplicates_and_drops_degenerate_faces() -> None:
    box = _box(8, 6, 3)
    # Unshared corners, as many exporters write them, plus a zero-area sliver.
    vertices = box.vertices[box.faces.reshape(-1)]
    faces = np.arange(len(vertices)).reshape(-1, 3)
    vertices = np.vstack([vertices, [[1, 1, 0], [2, 2, 0], [3, 3, 0]]])
    faces = np.vstack([faces, [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]])
    mesh = RoomMesh(vertices + 1e-9, faces, np.zeros(len(faces)), [("box",)])

    cleaned, merged, removed = clean_mesh(mesh)
    assert len(cleaned.vertices) == 8 + 3
    assert merged == 36 - 8
    assert removed == 1
    assert check_mesh(cleaned).closed


def test_points_inside_either_orientation() -> None:
    box = _box(8, 6, 3)
    points = [[4, 3, 1.5], [9, 3, 1.5], [4, 3, -0.1], [0.01, 0.01, 0.01]]
    expected = [True, False, False, True]
    assert points_inside(box, points).tolist() == expected
    flipped = RoomMesh(box.vertices, box.faces[:, ::-1], box.face_labels, box.labels)
    assert points_inside(flipped, points).tolist() == expected


@pytest.mark.parametrize("clockwise", [False, True])
def test_extruded_floor_plan_is_a_closed_room(clockwise: bool) -> None:
    from acoustic_spaces.geometry import box_mesh, extruded_mesh

    plan = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 9], [0, 9]], dtype=float)
    plan = plan[::-1] if clockwise else plan
    mesh = extruded_mesh(plan, 3.0)
    report = check_mesh(mesh)
    assert report.closed and report.normals_inward
    assert report.volume_m3 == pytest.approx(180.0)
    areas = mesh.face_areas()
    by_label = {label[0]: areas[mesh.face_labels == k].sum() for k, label in enumerate(mesh.labels)}
    assert by_label["floor"] == pytest.approx(60.0) and by_label["ceiling"] == pytest.approx(60.0)
    # Wall k is the plan's edge from corner k to corner k + 1, whichever way it runs.
    edges = np.linalg.norm(np.roll(plan, -1, axis=0) - plan, axis=1)
    for k, length in enumerate(edges, start=1):
        assert by_label[f"wall_{k}"] == pytest.approx(3.0 * length)
    assert all(label[1:] == ("walls",) for label in mesh.labels[2:])
    box = box_mesh((8.0, 6.0, 3.0))
    assert check_mesh(box).volume_m3 == pytest.approx(144.0)
    with pytest.raises(ValueError, match="positive"):
        box_mesh((8.0, 0.0, 3.0))
    with pytest.raises(ValueError, match="turns straight back"):
        extruded_mesh(np.array([[0, 0], [1, 1], [2, 2]]), 3.0)
    with pytest.raises(ValueError, match="walls 1 and 3 of the floor plan cross"):
        extruded_mesh(np.array([[0, 0], [1, 1], [1, 0], [0, 1]]), 3.0)


def test_plan_problems() -> None:
    from acoustic_spaces.geometry import plan_problem

    assert plan_problem(np.array([[0, 0], [24, 0], [24, 8], [8, 8], [8, 20], [0, 20]])) is None
    assert plan_problem(np.array([[0, 0], [2, 0], [4, 0], [4, 4], [0, 4]])) is None  # straight
    assert "three" in plan_problem(np.array([[0, 0], [1, 0]]))
    assert "corners 2 and 3" in plan_problem(np.array([[0, 0], [1, 0], [1, 0], [0, 1]]))
    # A corner that touches another wall, and a spike that runs back on itself.
    assert "walls 1 and 3" in plan_problem(np.array([[0, 0], [4, 0], [4, 4], [2, 0], [0, 4]]))
    spike = np.array([[0, 0], [4, 0], [4, 4], [2, 4], [2, 2], [2, 4], [0, 4]])
    assert "back at corner 5" in plan_problem(spike)
