"""Scenes as glTF 2.0 binary files (``.glb``) for Blender and other 3D tools.

The file holds the room, one object per surface name with a see-through material per
scene material (its absorption and scattering per octave band as extras, which Blender
shows as custom properties), and the sources and listeners turned the way they face,
with an arrow along the aim. It can also hold early reflection paths as lines, one
object per order, and sound particles as an animation in slow motion: by default
100 ms of sound play over 4 s. Blender opens it with File > Import > glTF 2.0.

Scenes are Z-up and glTF is Y-up: a point (x, y, z) is written as (x, z, -y), which
the importer turns back.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from acoustic_spaces._version import __version__
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.room import UNASSIGNED, Room, _label_text
from acoustic_spaces.viz import LISTENER_COLOUR, MATERIAL_COLOURS, SOURCE_COLOUR, path_groups

if TYPE_CHECKING:
    from acoustic_spaces.engine import EarlyReflections
    from acoustic_spaces.engine.particles import Particles

#: Particles the command line and the editor export: about 3 MB at 100 frames.
PARTICLE_COUNT = 1000
#: Seconds of animation per second of sound.
SLOW_MOTION = 40.0
#: Opacity of the room's surfaces.
SURFACE_ALPHA = 0.4
UNASSIGNED_GREY = "#9e9e9e"
PARTICLE_COLOUR = "#ffd166"

#: Scene to glTF coordinates: (x, y, z) -> (x, z, -y), a rotation about x.
_TO_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_FLOAT, _UINT = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_LINES, _TRIANGLES = 1, 4
_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942


def _linear(colour: str, alpha: float = 1.0) -> list[float]:
    """An sRGB hex colour as the linear RGBA factors glTF materials take."""
    srgb = np.array([int(colour[k : k + 2], 16) for k in (1, 3, 5)]) / 255.0
    rgb = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return [round(float(v), 6) for v in rgb] + [alpha]


def _point(position) -> list[float]:
    return [float(v) for v in _TO_GLTF @ np.asarray(position, dtype=float)]


def _quaternion(m: np.ndarray) -> list[float]:
    """The unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(1.0 + trace)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    return [float(v) for v in q]


def _lathe(profile: list[tuple[float, float]], segments: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed surface of revolution about the x axis, facing outwards.

    ``profile`` lists (x, radius) from one end on the axis to the other, x increasing
    along the sides. Returns the vertices and the triangles.
    """
    angles = 2.0 * np.pi * np.arange(segments) / segments
    vertices, triangles = [], []
    count, previous, was_ring = 0, None, False
    for x, r in profile:
        n = segments if r > 0 else 1
        cos, sin = np.cos(angles[:n]), np.sin(angles[:n])
        vertices.append(np.column_stack([np.full(n, x), r * cos, r * sin]))
        ring = count + np.arange(segments) % n  # a point on the axis repeats its index
        if was_ring:
            triangles.append(np.column_stack([previous, np.roll(previous, -1), np.roll(ring, -1)]))
        if previous is not None and n > 1:
            triangles.append(np.column_stack([previous, np.roll(ring, -1), ring]))
        previous, was_ring = ring, n > 1
        count += n
    return np.vstack(vertices), np.vstack(triangles)


def _sphere(radius: float, segments: int, rings: int) -> tuple[np.ndarray, np.ndarray]:
    """A sphere around the origin with its poles on the x axis."""
    theta = np.pi * np.arange(rings + 1) / rings
    profile = list(zip(-radius * np.cos(theta), radius * np.sin(theta), strict=True))
    profile[0], profile[-1] = (-radius, 0.0), (radius, 0.0)
    return _lathe(profile, segments)


def _arrow(length: float) -> tuple[np.ndarray, np.ndarray]:
    """An arrow along +x from the origin, shaped as the 3D view draws them."""
    shaft, tip, head = 0.05 * length, 0.1 * length, 0.75 * length
    return _lathe([(0.0, 0.0), (0.0, shaft), (head, shaft), (head, tip), (length, 0.0)], 16)


def _marker_radius(room: Room) -> float:
    """Radius of the source and listener spheres, as the editor's 3D view draws them."""
    lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    return max(0.012 * size, 0.05)


class _Document:
    """A glTF document and its binary chunk, built up piece by piece."""

    def __init__(self, name: str) -> None:
        self.json: dict = {
            "asset": {"version": "2.0", "generator": f"acoustic_spaces {__version__}"},
            "scene": 0,
            "scenes": [{"name": name, "nodes": []}],
        }
        self.binary = bytearray()

    def add(self, kind: str, item: dict) -> int:
        """Append ``item`` to the top-level array ``kind`` and return its index."""
        items = self.json.setdefault(kind, [])
        items.append(item)
        return len(items) - 1

    def accessors(
        self, data: np.ndarray, target: int | None = None, bounds: bool = False
    ) -> list[int]:
        """One accessor per entry of ``data`` (n, count) or (n, count, 3), all in one
        buffer view, as 32-bit floats or unsigned integers; returns their indices."""
        integer = np.issubdtype(data.dtype, np.integer)
        data = np.ascontiguousarray(data, dtype=np.uint32 if integer else np.float32)
        self.binary.extend(bytes(-len(self.binary) % 4))
        view = {"buffer": 0, "byteOffset": len(self.binary), "byteLength": data.nbytes}
        if target is not None:
            view["target"] = target
        self.binary.extend(data.tobytes())
        view_index = self.add("bufferViews", view)
        out = []
        for n, block in enumerate(data):
            accessor = {
                "bufferView": view_index,
                "componentType": _UINT if integer else _FLOAT,
                "count": len(block),
                "type": "SCALAR" if block.ndim == 1 else "VEC3",
            }
            if n:
                accessor["byteOffset"] = n * block.nbytes
            if bounds:
                accessor["min"] = np.atleast_1d(block.min(axis=0)).tolist()
                accessor["max"] = np.atleast_1d(block.max(axis=0)).tolist()
            out.append(self.add("accessors", accessor))
        return out

    def accessor(self, data: np.ndarray, target: int | None = None, bounds: bool = False) -> int:
        return self.accessors(data[None], target, bounds)[0]

    def material(
        self, name: str, colour: str, alpha: float = 1.0, glow: bool = False, extras=None
    ) -> int:
        material = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": _linear(colour, alpha),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            "doubleSided": True,
        }
        if alpha < 1.0:
            material["alphaMode"] = "BLEND"
        if glow:
            material["emissiveFactor"] = _linear(colour)[:3]
        if extras:
            material["extras"] = extras
        return self.add("materials", material)

    def mesh(
        self,
        name: str,
        vertices: np.ndarray,
        indices: np.ndarray,
        material: int,
        mode: int = _TRIANGLES,
        normals: np.ndarray | None = None,
    ) -> int:
        """A mesh of one primitive; ``vertices`` and ``normals`` in scene coordinates."""
        position = self.accessor(vertices @ _TO_GLTF.T, _ARRAY_BUFFER, bounds=True)
        attributes = {"POSITION": position}
        if normals is not None:
            attributes["NORMAL"] = self.accessor(normals @ _TO_GLTF.T, _ARRAY_BUFFER)
        primitive = {
            "attributes": attributes,
            "indices": self.accessor(indices.ravel(), _ELEMENT_ARRAY_BUFFER),
            "material": material,
            "mode": mode,
        }
        return self.add("meshes", {"name": name, "primitives": [primitive]})

    def node(self, name: str, parent: int | None = None, **fields) -> int:
        """A node under ``parent``, or at the root of the scene."""
        index = self.add("nodes", {"name": name, **fields})
        if parent is None:
            self.json["scenes"][0]["nodes"].append(index)
        else:
            self.json["nodes"][parent].setdefault("children", []).append(index)
        return index

    def write(self, path: str | Path) -> None:
        self.binary.extend(bytes(-len(self.binary) % 4))
        self.json["buffers"] = [{"byteLength": len(self.binary)}]
        text = json.dumps(self.json, ensure_ascii=False, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        chunks = (
            struct.pack("<II", len(text), _JSON_CHUNK)
            + text
            + struct.pack("<II", len(self.binary), _BIN_CHUNK)
            + self.binary
        )
        Path(path).write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


def _room_material(doc: _Document, room: Room, k: int) -> int:
    """Material ``k`` of the room (or grey for none), with its coefficients as extras."""
    if k == UNASSIGNED:
        return doc.material("no material", UNASSIGNED_GREY, SURFACE_ALPHA)
    name = room.material_names[k]
    material = room.scene.materials[name]
    extras = {
        "absorption": list(material.absorption),
        "scattering": list(material.scattering),
        "bands_hz": list(OCTAVE_BANDS_HZ),
    }
    if material.source:
        extras["source"] = material.source
    colour = MATERIAL_COLOURS[k % len(MATERIAL_COLOURS)]
    return doc.material(name, colour, SURFACE_ALPHA, extras=extras)


def _add_room(doc: _Document, room: Room) -> None:
    """One node per surface name under a ``room`` node; materials as they are used."""
    mesh = room.mesh
    areas = mesh.face_areas()
    materials: dict[int, int] = {}
    group = doc.node("room")
    for label in np.unique(mesh.face_labels):
        faces = np.flatnonzero(mesh.face_labels == label)
        k = int(room.face_materials[faces[0]])
        if k not in materials:
            materials[k] = _room_material(doc, room, k)
        extras = {"area_m2": round(float(areas[faces].sum()), 3)}
        if k != UNASSIGNED:
            extras["material"] = room.material_names[k]
        used, local = np.unique(mesh.faces[faces], return_inverse=True)
        name = _label_text(mesh.labels[label])
        index = doc.mesh(name, mesh.vertices[used], local.reshape(-1, 3), materials[k])
        doc.node(name, group, mesh=index, extras=extras)


def _add_points(doc: _Document, room: Room, radius: float) -> None:
    """Sources and listeners as spheres turned the way they face, with an arrow along
    the aim (for listeners, and for sources that are not omnidirectional)."""
    sphere, triangles = _sphere(radius, 24, 16)
    arrow = _arrow(6.0 * radius)
    scene = room.scene
    for kind, items, colour in (
        ("source", scene.sources, SOURCE_COLOUR),
        ("listener", scene.listeners, LISTENER_COLOUR),
    ):
        material = doc.material(kind, colour)
        marker = doc.mesh(kind, sphere, triangles, material, normals=sphere / radius)
        aim = None
        for k, item in enumerate(items):
            extras = {"kind": kind}
            directional = kind == "listener"
            if kind == "source":
                directivity = room.directivities[k]
                directional = not directivity.omni
                extras.update(
                    directivity=directivity.name or "omni",
                    directivity_index_db=[
                        round(float(v), 2) if np.isfinite(v) else None for v in directivity.index()
                    ],
                    level_db=item.level_db,
                )
            rotation = _TO_GLTF @ np.array(item.orientation.rotation()) @ _TO_GLTF.T
            node = doc.node(
                item.name,
                translation=_point(item.position),
                rotation=_quaternion(rotation),
                mesh=marker,
                extras=extras,
            )
            if directional:
                if aim is None:
                    aim = doc.mesh(f"{kind} aim", *arrow, material)
                doc.node(f"{item.name} aim", node, mesh=aim)


def _add_paths(doc: _Document, paths: EarlyReflections) -> None:
    """Early reflection paths as lines under ``reflection paths``: one node per order of
    specular paths, and one for the paths diffracted at an edge."""
    group = doc.node("reflection paths")
    for kind in path_groups(paths):
        chosen = [p for p, keep in zip(paths.points, kind.mask, strict=True) if keep]
        points = np.vstack(chosen)
        starts = np.cumsum([len(p) for p in chosen])  # where the next path begins
        first = np.arange(len(points) - 1)
        first = first[~np.isin(first + 1, starts)]
        if kind.order is None:
            name, extras = "paths diffracted", {"diffracted": True, "paths": len(chosen)}
        else:
            name, extras = f"paths order {kind.order}", {"order": kind.order, "paths": len(chosen)}
        material = doc.material(name, kind.colour, glow=True)
        lines = np.column_stack([first, first + 1])
        mesh = doc.mesh(name, points, lines, material, mode=_LINES)
        doc.node(name, group, mesh=mesh, extras=extras)


def _add_particles(
    doc: _Document, particles: Particles, radius: float, slow_motion: float, db_range: float
) -> None:
    """A node per particle under one group node, and one animation moving and scaling them."""
    times = np.asarray(particles.times * slow_motion, dtype=np.float32)
    if np.any(np.diff(times) <= 0):
        raise ValueError("particle frames must follow each other in time")
    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(particles.energy)
    scale = np.clip(1.0 + level / db_range, 0.0, 1.0)  # (frames, particles)
    positions = particles.positions @ _TO_GLTF.T

    name = f"particles from {particles.source}" if particles.source else "particles"
    material = doc.material("particle", PARTICLE_COLOUR, glow=True)
    sphere, triangles = _sphere(radius, 8, 6)
    mesh = doc.mesh("particle", sphere, triangles, material, normals=sphere / radius)
    group = doc.node(
        name,
        extras={
            "band_hz": particles.band_hz,
            "duration_ms": float(1e3 * particles.times[-1]),
            "slow_motion": slow_motion,
            "db_range": db_range,
        },
    )
    nodes = [
        doc.node(
            f"particle {k}",
            group,
            mesh=mesh,
            translation=positions[0, k].tolist(),
            scale=[float(scale[0, k])] * 3,
        )
        for k in range(positions.shape[1])
    ]
    time = doc.accessor(times, bounds=True)
    moves = doc.accessors(positions.transpose(1, 0, 2))
    sizes = doc.accessors(np.repeat(scale.T[:, :, None], 3, axis=2))
    samplers, channels = [], []
    for node, move, size in zip(nodes, moves, sizes, strict=True):
        for output, path in ((move, "translation"), (size, "scale")):
            channels.append({"sampler": len(samplers), "target": {"node": node, "path": path}})
            samplers.append({"input": time, "output": output, "interpolation": "LINEAR"})
    doc.add("animations", {"name": name, "samplers": samplers, "channels": channels})


def export_gltf(
    room: Room,
    path: str | Path,
    *,
    paths: EarlyReflections | None = None,
    particles: Particles | None = None,
    slow_motion: float = SLOW_MOTION,
    db_range: float = 60.0,
) -> None:
    """Write ``room`` with its sources and listeners to ``path`` as glTF 2.0 (``.glb``).

    ``paths`` (from :func:`acoustic_spaces.engine.early_paths` or a response's
    ``early``) adds the early reflection paths, coloured by order as in the 3D view.
    ``particles`` (from :func:`acoustic_spaces.engine.trace_particles`) adds an
    animation with a small sphere per particle that moves with it and shrinks with its
    level: full size at its share of the emitted energy, gone ``db_range`` dB below.
    The animation runs ``slow_motion`` times slower than the sound, with a keyframe at
    each of the particles' frames.
    """
    if slow_motion <= 0 or db_range <= 0:
        raise ValueError("slow_motion and db_range must be positive")
    doc = _Document(room.scene.name)
    radius = _marker_radius(room)
    _add_room(doc, room)
    _add_points(doc, room, radius)
    if paths is not None and len(paths):
        _add_paths(doc, paths)
    if particles is not None:
        _add_particles(doc, particles, 0.4 * radius, slow_motion, db_range)
    doc.write(path)


__all__ = ["PARTICLE_COUNT", "SLOW_MOTION", "export_gltf"]
