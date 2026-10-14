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


def _signed_areas(pts: np.ndarray, tris: list[tuple[int, int, int]]) -> list[float]:
    return [0.5 * float(np.cross(pts[b] - pts[a], pts[c] - pts[a])[2]) for a, b, c in tris]


@pytest.mark.parametrize(("repeated", "at"), [(1, 2), (4, 5), (0, 6)])
def test_triangulate_skips_a_repeated_corner(repeated: int, at: int) -> None:
    # The L-shaped floor with one corner written twice: a convex one, the reflex one or
    # the first corner again at the end. A repeat at a convex corner kept every ear next
    # to it from being clipped, and the fan it fell back to covered 8 m2.
    pts = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0], [2, 1, 0], [0, 1, 0]], float)
    pts = np.insert(pts, at, pts[repeated], axis=0)
    areas = _signed_areas(pts, triangulate_polygon(pts))
    assert len(areas) == 4 and min(areas) > 0  # the winding kept, nothing degenerate
    assert sum(areas) == pytest.approx(5.0)


def test_triangulate_polygon_round_a_hole() -> None:
    # A 4 x 4 m outline joined to a 2 x 2 m hole along an edge that runs both ways, so
    # the corners at either end of it appear twice.
    corners = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0), (1, 1), (1, 3), (3, 3), (3, 1), (1, 1)]
    pts = np.array([[x, y, 0.0] for x, y in corners])
    areas = _signed_areas(pts, triangulate_polygon(pts))
    assert min(areas) > 0
    assert sum(areas) == pytest.approx(12.0)


def test_obj_room_with_a_repeated_floor_corner(tmp_path: Path) -> None:
    obj = tmp_path / "l_room.obj"
    plan = [(0, 0), (3, 0), (3, 0), (3, 3), (2, 3), (2, 1), (0, 1)]
    lines = [f"v {x} {y} 0" for x, y in plan] + [f"v {x} {y} 2" for x, y in plan]
    lines += ["g floor", "f 1 2 3 4 5 6 7", "g ceiling", "f 14 13 12 11 10 9 8", "g walls"]
    lines += [f"f {k} {k + 7} {k % 7 + 8} {k % 7 + 1}" for k in range(1, 8) if k != 2]
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mesh, _, _ = clean_mesh(load_mesh(obj))
    report = check_mesh(mesh)
    assert report.closed and report.normals_inward
    assert report.volume_m3 == pytest.approx(10.0)
    assert _area_by_name(mesh)["floor"] == pytest.approx(5.0)
    assert report.surface_area_m2 == pytest.approx(5.0 + 5.0 + 2.0 * 12.0)


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


def test_obj_with_several_groups_on_one_line(tmp_path: Path) -> None:
    obj = tmp_path / "room.obj"
    obj.write_text(
        "v 0 0 0\nv 8 0 0\nv 8 6 0\nv 0 6 0\nusemtl oak\ng floor stage\nf 1 2 3 4\n",
        encoding="utf-8",
    )
    # Each name can take an assignment (it was one name, "floor stage").
    assert load_mesh(obj).labels == [("oak", "floor", "stage")]


def test_obj_line_continuation(tmp_path: Path) -> None:
    obj = tmp_path / "room.obj"
    obj.write_text(
        "v 0 0 0\nv 4 0 \\\n 0\nv 4 5 0\nv 0 5 0  # a corner\ng floor\nf 1 2 \\\n3 4\n",
        encoding="utf-8",
    )
    mesh = load_mesh(obj)
    np.testing.assert_allclose(mesh.vertices[1], [4, 0, 0])
    assert len(mesh.faces) == 2 and mesh.face_areas().sum() == pytest.approx(20.0)
    assert mesh.labels == [("floor",)]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("v 1e400 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", r":1: vertex coordinates must be finite"),
        ("v 0 0 0\nv nan 0 0\nv 0 1 0\nf 1 2 3\n", r":2: vertex coordinates must be finite"),
        ("v 0 0 0\nv 1 0 x\nv 0 1 0\nf 1 2 3\n", r":2: vertex coordinates must be numbers"),
        ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 \\ 3\n", r":4: face corner '\\\\' is no vertex"),
    ],
    ids=["infinite", "nan", "text", "stray backslash"],
)
def test_obj_bad_numbers_name_their_line(tmp_path: Path, content: str, message: str) -> None:
    obj = tmp_path / "bad.obj"
    obj.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_mesh(obj)


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


def _box_dxf(path: Path, size: tuple[float, float, float], insunits: int) -> None:
    """A closed box of 3DFACEs whose header records the drawing units ($INSUNITS)."""
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = insunits
    box = _box(*size)
    for corners in box.triangles:
        doc.modelspace().add_3dface([tuple(c) for c in corners], dxfattribs={"layer": "WALLS"})
    doc.saveas(path)


@pytest.mark.parametrize(
    ("insunits", "extent"), [(4, 8.0), (5, 80.0), (1, 203.2), (6, 8000.0), (0, 8000.0)]
)
def test_dxf_units_from_the_header(tmp_path: Path, insunits: int, extent: float) -> None:
    # Drawn as 8000 x 6000 x 3000: millimetres, centimetres, inches, metres, no units.
    path = tmp_path / "box.dxf"
    _box_dxf(path, (8000.0, 6000.0, 3000.0), insunits)
    mesh = load_mesh(path)
    size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    np.testing.assert_allclose(size, np.array([8.0, 6.0, 3.0]) * extent / 8.0)
    # Units given by the caller win over the file's.
    mesh = load_mesh(path, units="cm")
    assert np.ptp(mesh.vertices[:, 0]) == pytest.approx(80.0)


def test_dxf_triangle_without_a_fourth_corner(tmp_path: Path) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    doc.modelspace().add_3dface([(10, 10, 0), (14, 10, 0), (14, 15, 0)])
    path = tmp_path / "triangle.dxf"
    doc.saveas(path)
    # Leave out the fourth corner (group codes 13, 23 and 33), as the format allows.
    lines = path.read_text(encoding="utf-8").splitlines()
    pairs = [(code.strip(), value) for code, value in zip(lines[::2], lines[1::2], strict=True)]
    start = pairs.index(("2", "ENTITIES"))
    end = pairs.index(("0", "ENDSEC"), start)
    pairs[start:end] = [p for p in pairs[start:end] if p[0] not in ("13", "23", "33")]
    path.write_text("".join(f"{code}\n{value}\n" for code, value in pairs), encoding="utf-8")
    assert not ezdxf.readfile(path).modelspace()[0].dxf.hasattr("vtx3")

    mesh = load_mesh(path)
    # Not a quad with a phantom corner at the origin (15 m2).
    assert len(mesh.faces) == 1
    assert mesh.face_areas().sum() == pytest.approx(10.0)
    assert mesh.triangles.min(axis=(0, 1)).tolist() == [10.0, 10.0, 0.0]


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


def test_clean_turns_flipped_faces_to_agree_with_their_neighbours() -> None:
    # A closed box whose floor was written the other way round: the floor's solid angle
    # counted against the rest and put points near it outside the room.
    box = _box(8, 6, 3)
    faces = box.faces.copy()
    faces[:2] = faces[:2, ::-1]
    flipped = RoomMesh(box.vertices, faces, box.face_labels, box.labels)
    points = [[4, 3, 0.05], [4, 3, 0.3], [4, 3, 1.5]]
    assert not points_inside(flipped, points).any()

    cleaned, _, _ = clean_mesh(flipped)
    report = check_mesh(cleaned)
    assert report.closed and report.inconsistent_edges == 0
    assert report.volume_m3 == pytest.approx(144.0)
    # The floor turns back, not the larger rest: the normals still point inward.
    assert report.normals_inward
    np.testing.assert_allclose(cleaned.triangles, box.triangles)
    assert points_inside(cleaned, points).all()


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


def test_mesh_without_faces_is_not_closed() -> None:
    # Two triangles whose corners lie on one line: both are dropped.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    mesh = RoomMesh(vertices, [[0, 1, 2], [1, 2, 3]], [0, 0], [("floor",)])
    cleaned, _, removed = clean_mesh(mesh)
    report = check_mesh(cleaned, removed=removed)
    assert report.faces == 0 and removed == 2
    assert not report.closed  # it was closed, with nothing to close
    assert report.volume_m3 is None
    assert "no faces" in report.summary()


@pytest.mark.parametrize("sliver", [False, True])
def test_faces_are_split_where_an_edge_runs_past_a_corner(sliver: bool) -> None:
    # The floor is cut in two across its middle, the walls along it are not: their bottom
    # edges run past the floor's middle corners (T-junctions). SketchUp's exports close
    # such a gap with zero-area slivers, which cleaning drops.
    box = _box(8, 6, 3)
    vertices = np.vstack([box.vertices, [[4, 0, 0], [4, 6, 0]]])
    floor = [[0, 8, 9], [0, 9, 3], [8, 1, 2], [8, 2, 9]]
    faces = np.vstack([floor, box.faces[2:]] + ([[[0, 1, 8], [2, 3, 9]]] if sliver else []))
    mesh = RoomMesh(vertices, faces, np.zeros(len(faces)), [("box",)])
    assert check_mesh(mesh).closed == sliver  # the sliver closes it, without area

    cleaned, _, removed = clean_mesh(mesh)
    report = check_mesh(cleaned)
    assert removed == (2 if sliver else 0)
    assert report.closed and report.inconsistent_edges == 0 and report.normals_inward
    assert report.volume_m3 == pytest.approx(144.0)
    assert report.surface_area_m2 == pytest.approx(180.0)
    assert len(cleaned.faces) == 4 + 10 + 2  # each wall's bottom face split in two


def test_rooms_touching_along_an_edge_are_closed() -> None:
    # Two boxes that share one vertical edge: four faces meet along it, in pairs.
    a = _box(1, 1, 1)
    b = RoomMesh(a.vertices + [1.0, 1.0, 0.0], a.faces, a.face_labels, a.labels)
    cleaned, merged, _ = clean_mesh(RoomMesh.concatenate([a, b]))
    report = check_mesh(cleaned)
    assert merged == 2
    assert report.non_manifold_edges == 1 and report.odd_edges == 0
    assert report.closed and report.inconsistent_edges == 0
    assert report.volume_m3 == pytest.approx(2.0)


def test_a_face_given_twice_leaves_the_model_open() -> None:
    box = _box(8, 6, 3)
    faces = np.vstack([box.faces, box.faces[:1]])
    report = check_mesh(RoomMesh(box.vertices, faces, np.zeros(len(faces)), box.labels))
    assert report.boundary_edges == 0 and report.odd_edges == 3
    assert not report.closed and report.volume_m3 is None
    assert "3 edges shared by an odd number of faces" in report.summary()


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
