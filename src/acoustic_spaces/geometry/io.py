"""CAD importers.

- OBJ: read directly, keeping ``usemtl`` material names, ``g`` groups and ``o``
  objects. This is the main path: SketchUp, Rhino, Revit and Blender all export it.
- STL, PLY, OFF, glTF/GLB: read through trimesh; names come from glTF materials
  and nodes, or from the file name for formats without groups.
- DXF: read through ezdxf; 3DFACE, MESH and polyface entities, also inside block
  references, labelled by their layer.

All importers return metres with +z up.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from acoustic_spaces.geometry.mesh import RoomMesh
from acoustic_spaces.geometry.triangulate import triangulate_polygon

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
SUPPORTED_SUFFIXES = (".obj", ".stl", ".ply", ".off", ".gltf", ".glb", ".dxf")
#: Formats whose convention is +y up; everything else is taken as +z up.
_Y_UP_SUFFIXES = (".gltf", ".glb")


def load_mesh(path: str | Path, units: str = "m", up_axis: str | None = None) -> RoomMesh:
    """Load a room model and convert it to metres with +z up.

    ``up_axis`` overrides the format's convention ("y" for glTF, "z" otherwise);
    Blender's OBJ exporter, for example, writes +y up.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported geometry format {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    if units not in UNIT_SCALE:
        raise ValueError(f"unknown units {units!r}; use one of {', '.join(UNIT_SCALE)}")
    if not path.exists():
        raise FileNotFoundError(f"geometry file not found: {path}")

    if suffix == ".obj":
        mesh = read_obj(path)
    elif suffix == ".dxf":
        mesh = read_dxf(path)
    else:
        mesh = read_with_trimesh(path)

    if up_axis is None:
        up_axis = "y" if suffix in _Y_UP_SUFFIXES else "z"
    vertices = mesh.vertices * UNIT_SCALE[units]
    if up_axis == "y":
        # (x, y, z) with y up -> (x, -z, y) with z up; a rotation, so winding is kept.
        vertices = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    elif up_axis != "z":
        raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")
    return RoomMesh(vertices, mesh.faces, mesh.face_labels, mesh.labels)


class _MeshBuilder:
    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []
        self.face_labels: list[int] = []
        self.labels: dict[tuple[str, ...], int] = {}

    def add_polygon(self, corners: list[int], label: tuple[str, ...]) -> None:
        if len(corners) < 3:
            return
        label_id = self.labels.setdefault(label, len(self.labels))
        if len(corners) == 3:
            triangles = [(0, 1, 2)]
        else:
            triangles = triangulate_polygon(np.array([self.vertices[i] for i in corners]))
        for a, b, c in triangles:
            self.faces.append((corners[a], corners[b], corners[c]))
            self.face_labels.append(label_id)

    def build(self, source: Path) -> RoomMesh:
        if not self.faces:
            raise ValueError(f"no faces found in {source}")
        return RoomMesh(
            np.array(self.vertices, dtype=np.float64),
            np.array(self.faces, dtype=np.int64),
            np.array(self.face_labels, dtype=np.int64),
            list(self.labels),
        )


def _label(*names: str | None) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        if name and name not in out:
            out.append(name)
    return tuple(out)


def read_obj(path: Path) -> RoomMesh:
    """Wavefront OBJ: polygons with their usemtl / g / o names, in file units."""
    builder = _MeshBuilder()
    material = group = obj = None
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            tag, rest = parts[0], (parts[1].strip() if len(parts) > 1 else "")
            if tag == "v":
                values = rest.split()
                if len(values) < 3:
                    raise ValueError(f"{path}:{lineno}: vertex needs three coordinates")
                builder.vertices.append((float(values[0]), float(values[1]), float(values[2])))
            elif tag == "f":
                corners = []
                for token in rest.split():
                    index = int(token.split("/", 1)[0])
                    index = index - 1 if index > 0 else len(builder.vertices) + index
                    if not 0 <= index < len(builder.vertices):
                        raise ValueError(f"{path}:{lineno}: face refers to a missing vertex")
                    corners.append(index)
                builder.add_polygon(corners, _label(material, group, obj))
            elif tag == "usemtl":
                material = rest or None
            elif tag == "g":
                group = rest or None
            elif tag == "o":
                obj = rest or None
    return builder.build(path)


def read_with_trimesh(path: Path) -> RoomMesh:
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    parts: list[RoomMesh] = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
            continue
        vertices = trimesh.transformations.transform_points(geometry.vertices, transform)
        material = getattr(getattr(geometry.visual, "material", None), "name", None)
        names = [material, node, geometry_name]
        if len(scene.geometry) == 1:
            names.append(path.stem)
        label = _label(*names)
        parts.append(RoomMesh(vertices, geometry.faces, np.zeros(len(geometry.faces)), [label]))
    if not parts:
        raise ValueError(f"no faces found in {path}")
    return RoomMesh.concatenate(parts)


def read_dxf(path: Path) -> RoomMesh:
    import ezdxf
    from ezdxf.render import MeshBuilder

    doc = ezdxf.readfile(path)
    builder = _MeshBuilder()

    def add_vertices(points: Iterable) -> list[int]:
        start = len(builder.vertices)
        builder.vertices.extend((float(p[0]), float(p[1]), float(p[2])) for p in points)
        return list(range(start, len(builder.vertices)))

    def visit(entity, layer_override: str | None = None) -> None:
        kind = entity.dxftype()
        layer = layer_override if entity.dxf.layer == "0" and layer_override else entity.dxf.layer
        label = _label(layer)
        if kind == "3DFACE":
            corners = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2, entity.dxf.vtx3]
            if np.allclose(corners[2], corners[3]):
                corners = corners[:3]
            builder.add_polygon(add_vertices(corners), label)
        elif kind == "MESH" or (kind == "POLYLINE" and entity.is_poly_face_mesh):
            mesh = (
                MeshBuilder.from_mesh(entity)
                if kind == "MESH"
                else MeshBuilder.from_polyface(entity)
            )
            ids = add_vertices(mesh.vertices)
            for face in mesh.faces:
                builder.add_polygon([ids[i] for i in face], label)
        elif kind == "INSERT":
            for child in entity.virtual_entities():
                visit(child, layer)

    for entity in doc.modelspace():
        visit(entity)
    return builder.build(path)
