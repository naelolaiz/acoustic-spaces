"""Outdoor scenes: flat ground under an open sky, with buildings and barriers on it.

The ground, four walls of sky round its edge and the sky overhead close a box, so the
open air is a room like any other: its volume is known, sources and listeners are
checked to lie in it, and every face knows which side the air is on. The sky lets all
sound through (the material ``sky``, an opening), so sound that reaches it leaves the
scene as it leaves into the open; nothing reflects off it, and the edges it meets
diffract nothing, so the ground reads as endless.

- The ground is flat, at ``height``, over ``extent``, or over the buildings, barriers and
  the model file's objects with ``margin`` round them. The sky lies ``sky`` above it, or
  ``margin`` above the highest of them.
- A building is its footprint raised to its height: walls and a flat roof, open
  underneath, where it stands on the ground. Its surfaces are ``<name>_wall_1``... (the
  wall from the footprint's first corner to its second, and so on), ``<name>_walls``,
  ``<name>_roof`` and ``<name>``, and ``walls``, ``roofs`` and ``buildings`` for those of
  every building.
- A barrier is a thin screen along a path, from the ground to its height: surfaces
  ``<name>`` and ``barriers``.
- The model file's objects stand on the ground. Faces that lie on it are left out (a
  building's floor, the file's own ground), so they are open underneath too.
- Edges that lie on the ground diffract nothing: where a wall or a barrier meets it, the
  ground closes the corner (see :func:`acoustic_spaces.engine.diffraction.find_edges`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_spaces.geometry.builders import extruded_mesh, plan_problem
from acoustic_spaces.geometry.mesh import RoomMesh

#: The surface names of the ground and of the sky round and over it.
GROUND = "ground"
SKY = "sky"
#: Points this close (m) to the ground's height lie on it.
ON_GROUND_M = 0.01


@dataclass(frozen=True, eq=False)
class Layout:
    """Where the ground and the sky of an outdoor scene lie (m)."""

    #: (x_min, y_min, x_max, y_max) of the ground.
    extent: tuple[float, float, float, float]
    #: Height (z) of the ground.
    ground: float
    #: Height (z) of the sky.
    sky: float
    #: Each building's name, footprint (n, 2) and roof height (z), for the inside checks.
    buildings: tuple[tuple[str, np.ndarray, float], ...] = ()

    def building_at(self, points: np.ndarray) -> list[str | None]:
        """The building each point (n, 3) lies in, or None."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out: list[str | None] = [None] * len(points)
        for name, footprint, roof in self.buildings:
            inside = in_polygon(points[:, :2], footprint) & (points[:, 2] < roof)
            inside &= points[:, 2] > self.ground
            for k in np.flatnonzero(inside).tolist():
                out[k] = out[k] or name
        return out


def in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Whether each point (n, 2) lies inside the simple polygon (m, 2) (even-odd rule)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    a, b = polygon, np.roll(polygon, -1, axis=0)
    crosses = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return np.count_nonzero(crosses & (x < at), axis=1) % 2 == 1


def _footprint_problem(name: str, footprint: np.ndarray) -> str | None:
    problem = plan_problem(footprint)
    if problem is None:
        return None
    words = problem.replace("a floor plan", "a footprint").replace(
        "the floor plan", "the footprint"
    )
    return f"building {name!r}: {words}"


def building_mesh(name: str, footprint, height: float, ground: float) -> RoomMesh:
    """A building on the ground at height ``ground``: its ``footprint`` (n, 2) raised by
    ``height``, walls and a flat roof, open underneath; faces point out of it."""
    footprint = np.asarray(footprint, dtype=float)
    problem = _footprint_problem(name, footprint)
    if problem is not None:
        raise ValueError(problem)
    room = extruded_mesh(footprint, height)
    keep = room.face_labels != 0  # the floor: the ground is under it
    labels = [(f"{name}_roof", name, "roofs", "buildings")]
    for label in room.labels[2:]:
        labels.append((f"{name}_{label[0]}", f"{name}_walls", name, "walls", "buildings"))
    vertices = room.vertices + [0.0, 0.0, ground]
    faces = room.faces[keep][:, [0, 2, 1]]  # out of the building, not into it
    return RoomMesh(vertices, faces, room.face_labels[keep] - 1, labels)


def barrier_mesh(name: str, path, height: float, ground: float) -> RoomMesh:
    """A thin screen from the ground at height ``ground`` up by ``height``, along
    ``path`` (n, 2)."""
    path = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 1e-9 * max(float(np.ptp(path, axis=0).max()), 1.0)):
        k = int(np.argmax(lengths <= 1e-9 * max(float(np.ptp(path, axis=0).max()), 1.0)))
        raise ValueError(f"barrier {name!r}: points {k + 1} and {k + 2} are in one place")
    n = len(path)
    bottom = np.column_stack([path, np.full(n, ground)])
    top = np.column_stack([path, np.full(n, ground + height)])
    faces = []
    for i in range(n - 1):
        faces += [(i, i + 1, n + i + 1), (i, n + i + 1, n + i)]
    return RoomMesh(
        np.vstack([bottom, top]), np.array(faces), np.zeros(len(faces)), [(name, "barriers")]
    )


def standing(objects: RoomMesh, ground: float) -> RoomMesh:
    """``objects`` without the faces that lie on the ground at height ``ground``."""
    flat = np.all(np.abs(objects.triangles[:, :, 2] - ground) <= ON_GROUND_M, axis=1)
    return RoomMesh(
        objects.vertices, objects.faces[~flat], objects.face_labels[~flat], objects.labels
    )


def ground_box(extent: tuple[float, float, float, float], ground: float, sky: float) -> RoomMesh:
    """The ground over ``extent`` at height ``ground``, closed by the sky up to ``sky``;
    faces point into the air."""
    x0, y0, x1, y1 = extent
    box = extruded_mesh(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), sky - ground)
    face_labels = np.minimum(box.face_labels, 1)  # the floor: ground; the rest: sky
    return RoomMesh(box.vertices + [0.0, 0.0, ground], box.faces, face_labels, [(GROUND,), (SKY,)])


def outdoor_mesh(geometry, objects: RoomMesh | None = None, scale: float = 1.0):
    """The mesh of an outdoor scene (``geometry``, a scene's :class:`Geometry` with a
    ``ground``) and its :class:`Layout`. ``objects`` is the model file's mesh, in
    metres; ``scale`` the geometry's units in metres."""
    spec = geometry.ground
    ground = scale * spec.height
    parts: list[tuple[str, RoomMesh]] = []
    buildings = []
    for name, building in zip(geometry.building_names(), geometry.buildings or [], strict=True):
        footprint = scale * np.asarray(building.footprint, dtype=float)
        mesh = building_mesh(name, footprint, scale * building.height, ground)
        parts.append((f"building {name!r}", mesh))
        buildings.append((name, footprint, ground + scale * building.height))
    for name, barrier in zip(geometry.barrier_names(), geometry.barriers or [], strict=True):
        path = scale * np.asarray(barrier.path, dtype=float)
        parts.append(
            (f"barrier {name!r}", barrier_mesh(name, path, scale * barrier.height, ground))
        )
    if objects is not None:
        objects = standing(objects, ground)
        if len(objects.faces) == 0:
            raise ValueError("the model file has nothing on the ground: all its faces lie on it")
        parts.append(("the model file", objects))

    points = [mesh.vertices[np.unique(mesh.faces)] for _, mesh in parts]
    for what, corners in zip([w for w, _ in parts], points, strict=True):
        if corners[:, 2].min() < ground - ON_GROUND_M:
            raise ValueError(f"{what} reaches below the ground (z = {ground:g} m)")
    top = max((float(p[:, 2].max()) for p in points), default=ground)
    if spec.extent is not None:
        (x0, y0), (x1, y1) = (scale * np.asarray(corner, dtype=float) for corner in spec.extent)
        for what, corners in zip([w for w, _ in parts], points, strict=True):
            lo, hi = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
            if lo[0] <= x0 or lo[1] <= y0 or hi[0] >= x1 or hi[1] >= y1:
                raise ValueError(f"{what} reaches past the edge of the ground (its extent)")
    else:
        everything = np.vstack(points)
        margin = scale * spec.margin
        (x0, y0), (x1, y1) = (
            everything[:, :2].min(axis=0) - margin,
            everything[:, :2].max(axis=0) + margin,
        )
    sky = ground + scale * spec.sky if spec.sky is not None else top + scale * spec.margin
    if sky <= top:
        raise ValueError(
            f"the sky ({sky - ground:g} m above the ground) is below the top of the scene "
            f"({top - ground:g} m above it): raise geometry.ground.sky"
        )
    extent = (float(x0), float(y0), float(x1), float(y1))
    mesh = RoomMesh.concatenate([ground_box(extent, ground, sky), *[m for _, m in parts]])
    return mesh, Layout(extent, float(ground), float(sky), tuple(buildings))


def on_ground(points: np.ndarray, ground: float) -> np.ndarray:
    """Whether each point (..., 3) lies on the ground at height ``ground``."""
    return np.abs(np.asarray(points)[..., 2] - ground) <= ON_GROUND_M


__all__ = [
    "GROUND",
    "ON_GROUND_M",
    "SKY",
    "Layout",
    "barrier_mesh",
    "building_mesh",
    "ground_box",
    "in_polygon",
    "on_ground",
    "outdoor_mesh",
    "standing",
]
