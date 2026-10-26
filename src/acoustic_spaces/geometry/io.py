"""CAD importers.

- OBJ: read directly, keeping ``usemtl`` material names, ``g`` groups (several on
  one line, too) and ``o`` objects. This is the main path: SketchUp, Rhino, Revit and
  Blender all export it.
- STL, PLY, OFF, glTF/GLB: read through trimesh; names come from glTF materials
  and nodes, or from the file name for formats without groups.
- DXF: read through ezdxf; 3DFACE, MESH, polyface and polygon mesh entities and closed
  3D polylines (each a face), also inside block references, labelled by their layer, in
  the units the file records ($INSUNITS).
- SketchUp (experimental): read through openskp, the ``sketchup`` extra; faces labelled
  by their material.
- IFC: read through IfcOpenShell, the ``ifc`` extra; the building elements (walls, slabs,
  windows, furniture) with their openings cut, or one room's faces from its space
  boundaries, labelled by element name, type, materials and IFC class.
- Collada (DAE): read directly, labelled by material and geometry name, in the unit and
  with the up axis the file records.

All importers return metres with +z up.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from acoustic_spaces._extras import install_hint
from acoustic_spaces.geometry.mesh import RoomMesh
from acoustic_spaces.geometry.triangulate import triangulate_polygon

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
SUPPORTED_SUFFIXES = (
    ".obj",
    ".stl",
    ".ply",
    ".off",
    ".gltf",
    ".glb",
    ".dxf",
    ".skp",
    ".ifc",
    ".dae",
)
#: The extra each format that needs one is read with, and the module that extra brings.
READER_EXTRAS = {
    ".skp": ("sketchup", "openskp"),
    ".ifc": ("ifc", "ifcopenshell"),
}
#: Formats whose convention is +y up (openskp hands SketchUp models over as glTF does);
#: everything else is taken as +z up.
_Y_UP_SUFFIXES = (".gltf", ".glb", ".skp")
#: Metres per drawing unit for the DXF $INSUNITS codes a room may be drawn in: inches,
#: feet, millimetres, centimetres, metres, yards and decimetres. Files without units
#: (code 0) or in others are read as metres.
_DXF_UNIT_SCALE = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 10: 0.9144, 14: 0.1}


def missing_extra(path: str | Path) -> str | None:
    """The extra that reading ``path`` needs (see ``READER_EXTRAS``) when it is not
    installed; None when none is needed or it is installed."""
    from importlib.util import find_spec

    need = READER_EXTRAS.get(Path(path).suffix.lower())
    return None if need is None or find_spec(need[1]) is not None else need[0]


def extra_for(path: str | Path) -> str | None:
    """The extra that reading ``path`` needs, installed or not; None for none."""
    return READER_EXTRAS.get(Path(path).suffix.lower(), (None, None))[0]


def load_mesh(
    path: str | Path,
    units: str | None = None,
    up_axis: str | None = None,
    *,
    space: str | None = None,
) -> RoomMesh:
    """Load a room model and convert it to metres with +z up.

    ``units`` are those of the file's coordinates; None takes the units a DXF, IFC or
    Collada file records, and metres for other files.
    ``up_axis`` overrides the format's convention ("y" for glTF, the axis a Collada file
    records, "z" otherwise); Blender's OBJ exporter, for example, writes +y up.
    ``space`` names a room of an IFC file (an IfcSpace) to take on its own, from its
    space boundaries; None reads all building elements.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported geometry format {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    if units is not None and units not in UNIT_SCALE:
        raise ValueError(f"unknown units {units!r}; use one of {', '.join(UNIT_SCALE)}")
    if not path.exists():
        raise FileNotFoundError(f"geometry file not found: {path}")
    if space is not None and suffix != ".ifc":
        raise ValueError(f"space names a room of an IFC file; {path.name} is not one")
    if up_axis not in (None, "y", "z"):
        raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")

    file_scale = file_up = None
    if suffix == ".obj":
        mesh = read_obj(path)
    elif suffix == ".dxf":
        mesh, file_scale = _read_dxf(path)
    elif suffix == ".skp":
        mesh = read_skp(path)
    elif suffix == ".ifc":
        mesh, file_scale = _read_ifc(path, space)
    elif suffix == ".dae":
        mesh, file_scale, file_up = _read_collada(path)
    else:
        mesh = read_with_trimesh(path)
    # Each reader's own checks aside, for every format: a coordinate that is no number
    # would make the model's extent and volume NaN or infinite.
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{path}: vertex coordinates must be finite")
    if len(mesh.faces) and not 0 <= int(mesh.faces.min()) <= int(mesh.faces.max()) < len(
        mesh.vertices
    ):
        raise ValueError(f"{path}: a face refers to a missing vertex")

    if up_axis is None:
        up_axis = file_up or ("y" if suffix in _Y_UP_SUFFIXES else "z")
    scale = UNIT_SCALE[units] if units is not None else (file_scale or 1.0)
    vertices = mesh.vertices * scale
    if up_axis == "y":
        # (x, y, z) with y up -> (x, -z, y) with z up; a rotation, so winding is kept.
        vertices = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
    elif up_axis == "x":
        # Only a Collada file records +x up: (x, y, z) -> (-y, -z, x), right being -y.
        vertices = np.column_stack([-vertices[:, 1], -vertices[:, 2], vertices[:, 0]])
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

    def add_triangles(self, vertices: np.ndarray, triangles: np.ndarray, label) -> None:
        start = len(self.vertices)
        label_id = self.labels.setdefault(label, len(self.labels))
        self.vertices.extend(map(tuple, np.asarray(vertices, dtype=np.float64)))
        self.faces.extend(map(tuple, np.asarray(triangles, dtype=np.int64) + start))
        self.face_labels.extend([label_id] * len(triangles))

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


def _obj_statements(lines: Iterable[str]) -> Iterator[tuple[int, str]]:
    """The statements of an OBJ file and the line each starts on: comments removed, and
    a line that ends in a backslash joined to the next."""
    pending, start = "", 0
    for lineno, raw in enumerate(lines, start=1):
        text = raw.split("#", 1)[0].strip()
        if not pending:
            start = lineno
        if text.endswith("\\"):
            pending += text[:-1] + " "
            continue
        statement = (pending + text).strip()
        pending = ""
        if statement:
            yield start, statement
    if pending.strip():
        yield start, pending.strip()


def read_obj(path: Path) -> RoomMesh:
    """Wavefront OBJ: polygons with their usemtl / g / o names, in file units.

    ``g`` may name several groups at once; the faces after it carry all of them.
    """
    builder = _MeshBuilder()
    material = obj = None
    groups: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, line in _obj_statements(fh):
            parts = line.split(maxsplit=1)
            tag, rest = parts[0], (parts[1].strip() if len(parts) > 1 else "")
            if tag == "v":
                values = rest.split()
                if len(values) < 3:
                    raise ValueError(f"{path}:{lineno}: vertex needs three coordinates")
                try:
                    vertex = (float(values[0]), float(values[1]), float(values[2]))
                except ValueError:
                    raise ValueError(
                        f"{path}:{lineno}: vertex coordinates must be numbers"
                    ) from None
                if not all(math.isfinite(v) for v in vertex):
                    raise ValueError(f"{path}:{lineno}: vertex coordinates must be finite")
                builder.vertices.append(vertex)
            elif tag == "f":
                corners = []
                for token in rest.split():
                    try:
                        index = int(token.split("/", 1)[0])
                    except ValueError:
                        raise ValueError(
                            f"{path}:{lineno}: face corner {token!r} is no vertex number"
                        ) from None
                    index = index - 1 if index > 0 else len(builder.vertices) + index
                    if not 0 <= index < len(builder.vertices):
                        raise ValueError(f"{path}:{lineno}: face refers to a missing vertex")
                    corners.append(index)
                builder.add_polygon(corners, _label(material, *groups, obj))
            elif tag == "usemtl":
                material = rest or None
            elif tag == "g":
                groups = rest.split()
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


def read_skp(path: Path) -> RoomMesh:
    """A SketchUp file through openskp (the ``sketchup`` extra), in metres with +y up.

    openskp hands the faces over grouped by colour, so each face is labelled with the
    model's material of that colour (``default`` when none has it, as for unpainted
    faces). A face painted differently on its two sides comes twice, once per side; the
    first side with a material is kept.
    """
    try:
        from openskp import SkpFile
    except ImportError as exc:
        hint = install_hint("sketchup")
        raise ImportError(f"reading SketchUp files needs openskp: {hint}") from exc

    skp = SkpFile.open(path)
    by_colour: dict[tuple[int, ...], str] = {}
    for material in skp.parse().materials:
        by_colour.setdefault(tuple(int(c) for c in material.color[:3]), material.name)
    scene = skp.build_scene(include_curve_sets=False)
    kept: dict[tuple, tuple[np.ndarray, str | None]] = {}
    for primitive in scene.glb_primitives:
        pbr = scene.gltf_materials[primitive.material_index].get("pbrMetallicRoughness", {})
        rgb = pbr.get("baseColorFactor", [1.0, 1.0, 1.0])[:3]
        name = by_colour.get(tuple(round(c * 255) for c in rgb))
        corners = np.asarray(primitive.positions, dtype=np.float64).reshape(-1, 3)
        for triangle in np.asarray(primitive.indices, dtype=np.int64).reshape(-1, 3):
            points = corners[triangle]
            key = tuple(sorted(map(tuple, np.round(points, 6))))
            if key not in kept or (kept[key][1] is None and name is not None):
                kept[key] = (points, name)
    if not kept:
        raise ValueError(f"no faces found in {path}")
    labels: dict[tuple[str, ...], int] = {}
    face_labels = [
        labels.setdefault((name or "default",), len(labels)) for _, name in kept.values()
    ]
    vertices = np.concatenate([points for points, _ in kept.values()])
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return RoomMesh(vertices, faces, np.array(face_labels, dtype=np.int64), list(labels))


def read_dxf(path: Path) -> RoomMesh:
    """DXF: faces labelled by their layer, in drawing units (see :func:`load_mesh`)."""
    return _read_dxf(path)[0]


def _read_dxf(path: Path) -> tuple[RoomMesh, float | None]:
    """The faces of a DXF file in drawing units, and the metres per drawing unit that
    its header records (None when it records none of ``_DXF_UNIT_SCALE``)."""
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
            corners = [entity.dxf.vtx0, entity.dxf.vtx1, entity.dxf.vtx2]
            # A triangle repeats its third corner as the fourth, or leaves the fourth
            # out, which ezdxf then reads as the origin.
            if entity.dxf.hasattr("vtx3") and not np.allclose(entity.dxf.vtx3, corners[2]):
                corners.append(entity.dxf.vtx3)
            builder.add_polygon(add_vertices(corners), label)
        elif kind == "POLYLINE" and entity.is_3d_polyline:
            # A closed 3D polyline outlines one face; an open one bounds nothing.
            if entity.is_closed:
                corners = [v.dxf.location for v in entity.vertices]
                builder.add_polygon(add_vertices(corners), label)
        elif kind == "MESH" or (
            kind == "POLYLINE" and (entity.is_poly_face_mesh or entity.is_polygon_mesh)
        ):
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
    return builder.build(path), _DXF_UNIT_SCALE.get(doc.header.get("$INSUNITS", 0))


#: Elements an IFC import leaves out: openings and other features (cut from or added to
#: their elements already), virtual elements, and building services (ducts, pipes,
#: lights), which bound no room.
_IFC_SKIPPED = ("IfcFeatureElement", "IfcVirtualElement", "IfcDistributionElement")
#: Without space boundaries, a room's faces are cut into triangles no longer than this
#: (metres), each labelled by the element behind its middle.
IFC_SPACE_FACE_M = 0.5
#: How far (metres) behind a room's face the element it lies against is looked for.
_IFC_ELEMENT_DEPTH_M = 2.0


def read_ifc(path: Path, space: str | None = None) -> RoomMesh:
    """IFC: building elements, or one room (``space``), in file units (see
    :func:`load_mesh`)."""
    return _read_ifc(path, space)[0]


def _open_ifc(path: Path):
    try:
        import ifcopenshell
    except ImportError as exc:
        raise ImportError(f"reading IFC files needs IfcOpenShell: {install_hint('ifc')}") from exc
    try:
        return ifcopenshell.open(str(path))
    except Exception as exc:  # IfcOpenShell raises several kinds for a file it cannot parse
        raise ValueError(f"{path}: not a readable IFC file ({exc})") from exc


def _read_ifc(path: Path, space: str | None) -> tuple[RoomMesh, float]:
    """The faces of an IFC file in its length unit, and the metres per that unit.

    Without ``space``, every building element is read as IfcOpenShell builds it (walls
    with their openings cut, windows, doors, slabs, furniture), so walls come with both
    sides and their thickness. With it, only the room of that name (an IfcSpace, by
    name, long name or GlobalId): its space boundaries, the faces of the room each
    element bounds, as BIM programs export them; or, for a file without them, the
    space's own solid, each part labelled by the element behind it.
    """
    model = _open_ifc(path)
    import ifcopenshell.util.unit

    scale = float(ifcopenshell.util.unit.calculate_unit_scale(model))
    if space is None:
        builder = _ifc_elements(model)
    else:
        room = _ifc_space(model, path, space)
        builder = _ifc_boundaries(model, room, scale)
        if not builder.faces:
            builder = _ifc_space_solid(model, room, path)
    mesh = builder.build(path)
    if space is not None:
        mesh = _facing_in(mesh)
    # IfcOpenShell builds shapes in metres; the file's unit is kept for ``units``.
    return RoomMesh(mesh.vertices / scale, mesh.faces, mesh.face_labels, mesh.labels), scale


def _facing_in(mesh: RoomMesh) -> RoomMesh:
    """A room's faces turned to face into it, as this package's rooms do (IFC has a
    space's boundaries, and its solid, face out of it)."""
    a, b, c = (mesh.vertices[mesh.faces[:, k]] for k in range(3))
    if np.einsum("ij,ij->", a, np.cross(b, c)) <= 0.0:  # six times the signed volume
        return mesh
    return RoomMesh(mesh.vertices, mesh.faces[:, ::-1].copy(), mesh.face_labels, mesh.labels)


def _ifc_label(element) -> tuple[str, ...]:
    """What the faces of an element are named: its name, its type's name, its materials
    (a layer set's name, then each layer's material) and its IFC class, with the class
    it specialises for IfcWallStandardCase and the like."""
    import ifcopenshell.util.element as util

    names = [element.Name, getattr(util.get_type(element), "Name", None)]
    material = util.get_material(element, should_skip_usage=True)
    if material is not None:
        names.append(getattr(material, "Name", None) or getattr(material, "LayerSetName", None))
        names.extend(m.Name for m in util.get_materials(element))
    kind = element.is_a()
    names += [kind, kind.removesuffix("StandardCase").removesuffix("ElementedCase")]
    return _label(*(n.strip() if isinstance(n, str) else None for n in names))


def _ifc_shapes(model, products) -> Iterator[tuple[object, np.ndarray, np.ndarray]]:
    """Each product's triangles as IfcOpenShell builds them, in metres in the world."""
    import os

    import ifcopenshell.geom

    if not products:
        return
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    iterator = ifcopenshell.geom.iterator(settings, model, os.cpu_count() or 1, include=products)
    if not iterator.initialize():
        return
    while True:
        shape = iterator.get()
        vertices = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        triangles = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
        if len(triangles):
            yield model.by_id(shape.id), vertices, triangles
        if not iterator.next():
            break


def _ifc_elements(model) -> _MeshBuilder:
    elements = [e for e in model.by_type("IfcElement") if not any(e.is_a(k) for k in _IFC_SKIPPED)]
    builder = _MeshBuilder()
    # In file order: shapes come from several threads in whichever order they finish.
    shapes = sorted(_ifc_shapes(model, elements), key=lambda shape: shape[0].id())
    for element, vertices, triangles in shapes:
        builder.add_triangles(vertices, triangles, _ifc_label(element))
    return builder


def ifc_spaces(path: str | Path) -> list[tuple[str, str]]:
    """The rooms (IfcSpace) of an IFC file: for each, what ``space`` names it by (its
    name, or its GlobalId when it has none) and how it reads (name and long name)."""
    out = []
    for space in _open_ifc(Path(path)).by_type("IfcSpace"):
        shown = " ".join(n.strip() for n in (space.Name, space.LongName) if n and n.strip())
        out.append(((space.Name or "").strip() or space.GlobalId, shown or space.GlobalId))
    return out


def _ifc_space(model, path: Path, name: str):
    key = name.strip().casefold()
    spaces = model.by_type("IfcSpace")
    for space in spaces:
        if key in {
            (n or "").strip().casefold() for n in (space.Name, space.LongName, space.GlobalId)
        }:
            return space
    known = ", ".join(" ".join(filter(None, (s.Name, s.LongName))) or s.GlobalId for s in spaces)
    raise ValueError(
        f"{path}: no space {name!r}; "
        + (f"it has {known}" if spaces else "it has no spaces (IfcSpace): leave space out")
    )


def _ifc_curve(curve) -> list | None:
    """The corners of a boundary curve in its plane, or None for curves not read."""
    if curve.is_a("IfcPolyline"):
        return [p.Coordinates for p in curve.Points]
    if curve.is_a("IfcIndexedPolyCurve"):
        return list(curve.Points.CoordList)  # arcs are taken as their corners
    if curve.is_a("IfcCompositeCurve"):
        out = []
        for segment in curve.Segments:
            part = _ifc_curve(segment.ParentCurve)
            if part is None:
                return None
            out += part if segment.SameSense else part[::-1]
        return out
    return None


def _ring(corners) -> np.ndarray | None:
    """Corners without repeats or the closing corner; None for fewer than three."""
    points = np.asarray(corners, dtype=np.float64)
    if len(points) == 0:
        return None
    keep = np.r_[True, np.abs(np.diff(points, axis=0)).max(axis=1) > 1e-9]
    points = points[keep]
    if len(points) > 1 and np.abs(points[0] - points[-1]).max() <= 1e-9:
        points = points[:-1]
    return points if len(points) >= 3 else None


def _triangulate_rings(rings: list[np.ndarray]) -> np.ndarray:
    """Triangles over an outline (the first ring) with holes, all in 2D, turning
    counter-clockwise."""
    import mapbox_earcut

    points = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    triangles = np.asarray(mapbox_earcut.triangulate_float64(points, ends), dtype=np.int64).reshape(
        -1, 3
    )
    a, b, c = (points[triangles[:, k]] for k in range(3))
    turn = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    triangles[turn < 0] = triangles[turn < 0][:, ::-1]
    return triangles


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    """Two unit vectors across a plane, the first times the second giving ``normal``."""
    n = normal / np.linalg.norm(normal)
    u = np.cross(n, (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    return np.array([u, np.cross(n, u)])


def _ifc_surface(surface) -> tuple[np.ndarray, np.ndarray] | None:
    """The corners (in the space's coordinates and file units) and triangles of a
    boundary's surface, facing as the file has it; None for surfaces not read."""
    import ifcopenshell.util.placement

    if surface.is_a("IfcCurveBoundedPlane"):
        curves = [surface.OuterBoundary, *(surface.InnerBoundaries or ())]
        rings = [_ring([c[:2] for c in _ifc_curve(curve) or ()]) for curve in curves]
        if rings[0] is None:
            return None
        rings = [rings[0], *(r for r in rings[1:] if r is not None)]
        triangles = _triangulate_rings(rings)
        flat = np.concatenate(rings)
        plane = ifcopenshell.util.placement.get_axis2placement(surface.BasisSurface.Position)
        corners = flat @ plane[:3, :2].T + plane[:3, 3]
        return corners, triangles
    if surface.is_a("IfcFaceBasedSurfaceModel") or surface.is_a("IfcFaceSurface"):
        faces = (
            [f for s in surface.FbsmFaces for f in s.CfsFaces]
            if surface.is_a("IfcFaceBasedSurfaceModel")
            else [surface]
        )
        corners, triangles = [], []
        for face in faces:
            loops = sorted(face.Bounds, key=lambda b: not b.is_a("IfcFaceOuterBound"))
            rings = [_ring([p.Coordinates for p in b.Bound.Polygon]) for b in loops]
            if rings[0] is None:
                continue
            rings = [rings[0], *(r for r in rings[1:] if r is not None)]
            outline = rings[0]
            normal = np.cross(outline - outline.mean(axis=0), np.roll(outline, -1, axis=0)).sum(0)
            if not np.any(normal):
                continue
            basis = _plane_basis(normal)
            start = sum(len(c) for c in corners)
            triangles.append(_triangulate_rings([r @ basis.T for r in rings]) + start)
            corners += rings
        return (np.concatenate(corners), np.concatenate(triangles)) if triangles else None
    return None


def _ifc_boundaries(model, space, scale: float) -> _MeshBuilder:
    """A room's faces from its space boundaries, in metres, each labelled by the
    element it lies on (``virtual boundary`` where none does, as at an opening to
    another room)."""
    import ifcopenshell.util.placement

    placement = ifcopenshell.util.placement.get_local_placement(space.ObjectPlacement)
    builder = _MeshBuilder()
    for boundary in space.BoundedBy or ():
        geometry = boundary.ConnectionGeometry
        if geometry is None or not geometry.is_a("IfcConnectionSurfaceGeometry"):
            continue
        found = _ifc_surface(geometry.SurfaceOnRelatingElement)
        if found is None:
            continue
        corners, triangles = found
        world = corners @ placement[:3, :3].T + placement[:3, 3]
        element = boundary.RelatedBuildingElement
        if element is None or element.is_a("IfcVirtualElement"):
            label = _label("virtual boundary", getattr(element, "Name", None))
        else:
            label = _ifc_label(element)
        builder.add_triangles(world * scale, triangles, label)
    return builder


def _ifc_space_solid(model, space, path: Path) -> _MeshBuilder:
    """A room without space boundaries: its solid, cut into small triangles each
    labelled by the element behind its middle (the space's own name where none is)."""
    import trimesh

    from acoustic_spaces import _native

    shapes = list(_ifc_shapes(model, [space]))
    if not shapes:
        raise ValueError(f"{path}: space {space.Name!r} has neither space boundaries nor a shape")
    _, vertices, triangles = shapes[0]
    vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, IFC_SPACE_FACE_M)
    elements = _ifc_elements(model)
    own = _label(space.Name, space.LongName, "IfcSpace")
    builder = _MeshBuilder()
    if not elements.faces:
        builder.add_triangles(vertices, triangles, own)
        return builder
    behind = elements.build(path)
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    middles = (a + b + c) / 3.0
    # From just inside the room, outwards: the space's solid faces out of the room.
    dist, face = _native.Bvh(behind.vertices, behind.faces).intersect(
        np.ascontiguousarray(middles - 0.01 * normals),
        np.ascontiguousarray(normals),
        np.full(len(middles), _IFC_ELEMENT_DEPTH_M),
        np.full(len(middles), -1, dtype=np.int64),
    )
    hit = (face >= 0) & np.isfinite(dist)
    labels = [
        behind.labels[behind.face_labels[f]] if ok else own for f, ok in zip(face, hit, strict=True)
    ]
    for label in dict.fromkeys(labels):
        keep = [k for k, other in enumerate(labels) if other == label]
        builder.add_triangles(vertices, triangles[keep], label)
    return builder


def read_collada(path: Path) -> RoomMesh:
    """Collada (DAE), in file units (see :func:`load_mesh`)."""
    return _read_collada(path)[0]


def _read_collada(path: Path) -> tuple[RoomMesh, float, str]:
    """The faces of a Collada file's scene, labelled by material and geometry name, the
    metres per unit and the up axis the file records ("x", "y" or "z").

    Triangles, polylists and polygons are read (lines are left out, and holes in
    polygons); nodes keep their matrices, translations, rotations and scales, and
    instanced nodes (SketchUp's components) are placed wherever they are used.
    """
    import xml.etree.ElementTree as ElementTree

    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"{path}: not a readable Collada file ({exc})") from exc
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]  # COLLADA 1.4 and 1.5 namespaces
    if root.tag != "COLLADA":
        raise ValueError(f"{path}: not a Collada file (it holds <{root.tag}>)")
    ids = {element.get("id"): element for element in root.iter() if element.get("id")}

    def target(element) -> object | None:
        return ids.get((element.get("url") or element.get("target") or "").removeprefix("#"))

    builder = _MeshBuilder()

    def visit(node, matrix: np.ndarray, depth: int) -> None:
        if depth > 100:
            raise ValueError(f"{path}: nodes nested too deep (a node that holds itself?)")
        matrix = matrix @ _collada_transform(node)
        for child in node:
            if child.tag == "node":
                visit(child, matrix, depth + 1)
            elif child.tag == "instance_node" and target(child) is not None:
                visit(target(child), matrix, depth + 1)
            elif child.tag == "instance_geometry" and target(child) is not None:
                materials = {
                    bound.get("symbol"): target(bound) for bound in child.iter("instance_material")
                }
                _collada_geometry(builder, target(child), matrix, materials, ids, path)

    reference = root.find("scene/instance_visual_scene")
    scene = target(reference) if reference is not None else None
    if scene is None:
        scene = root.find("library_visual_scenes/visual_scene")
    for node in scene.findall("node") if scene is not None else ():
        visit(node, np.eye(4), 0)
    unit = root.find("asset/unit")
    scale = float(unit.get("meter", 1.0)) if unit is not None else 1.0
    up = (root.findtext("asset/up_axis") or "Y_UP").strip()
    if up not in ("X_UP", "Y_UP", "Z_UP"):
        raise ValueError(f"{path}: unknown up axis {up!r}; use X_UP, Y_UP or Z_UP")
    return builder.build(path), scale, up[0].lower()


def _numbers(element, kind=float) -> np.ndarray:
    return np.array([kind(v) for v in (element.text or "").split()])


def _collada_transform(node) -> np.ndarray:
    """A node's transformations, in the order it lists them."""
    out = np.eye(4)
    for child in node:
        step = np.eye(4)
        if child.tag == "matrix":
            step = _numbers(child).reshape(4, 4)
        elif child.tag == "translate":
            step[:3, 3] = _numbers(child)
        elif child.tag == "scale":
            step[:3, :3] = np.diag(_numbers(child))
        elif child.tag == "rotate":
            x, y, z, degrees = _numbers(child)
            axis = np.array([x, y, z]) / np.linalg.norm([x, y, z])
            angle = math.radians(degrees)
            cross = np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
            )
            step[:3, :3] = (
                np.eye(3) * math.cos(angle)
                + math.sin(angle) * cross
                + (1 - math.cos(angle)) * np.outer(axis, axis)
            )
        else:
            continue
        out = out @ step
    return out


def _collada_geometry(builder: _MeshBuilder, geometry, matrix, materials, ids, path) -> None:
    mesh = geometry.find("mesh")
    if mesh is None:
        return  # splines and other kinds of geometry
    vertices = mesh.find("vertices")
    position = None if vertices is None else vertices.find("input[@semantic='POSITION']")
    source = None if position is None else ids.get(position.get("source", "").removeprefix("#"))
    if source is None or source.find("float_array") is None:
        return
    accessor = source.find("technique_common/accessor")
    stride = int(accessor.get("stride", 3)) if accessor is not None else 3
    points = _numbers(source.find("float_array")).reshape(-1, stride)[:, :3]
    if not (np.isfinite(points).all() and np.isfinite(matrix).all()):
        raise ValueError(f"{path}: vertex coordinates must be finite")
    points = points @ matrix[:3, :3].T + matrix[:3, 3]
    mirrored = np.linalg.det(matrix[:3, :3]) < 0  # a mirrored copy turns its faces over
    for primitive in mesh:
        if primitive.tag not in ("triangles", "polylist", "polygons"):
            continue  # lines, and strips and fans, which exporters seldom write
        inputs = primitive.findall("input")
        offsets = [int(i.get("offset", 0)) for i in inputs]
        vertex = next((i for i in inputs if i.get("semantic") == "VERTEX"), None)
        if vertex is None:
            continue
        step, at = max(offsets) + 1, int(vertex.get("offset", 0))
        if primitive.tag == "polygons":
            outlines = [*primitive.findall("p"), *primitive.findall("ph/p")]
            polygons = [_numbers(p, int).reshape(-1, step)[:, at] for p in outlines]
        else:
            corners = np.concatenate(
                [_numbers(p, int) for p in primitive.findall("p")] or [np.zeros(0, int)]
            )
            corners = corners.reshape(-1, step)[:, at]
            if primitive.tag == "triangles":
                counts = np.full(len(corners) // 3, 3)
            else:
                counts = _numbers(primitive.find("vcount"), int)
            polygons = np.split(corners, np.cumsum(counts)[:-1]) if len(counts) else []
        symbol = primitive.get("material")
        material = materials.get(symbol)
        if material is None:
            material = ids.get(symbol)  # some exporters name the material itself
        label = _label(
            None if material is None else material.get("name") or material.get("id"),
            None if material is not None else primitive.get("material"),
            geometry.get("name"),
            geometry.get("id"),
        )
        start = len(builder.vertices)
        builder.vertices.extend(map(tuple, points))
        for polygon in polygons:
            builder.add_polygon([start + int(i) for i in polygon[:: -1 if mirrored else 1]], label)
