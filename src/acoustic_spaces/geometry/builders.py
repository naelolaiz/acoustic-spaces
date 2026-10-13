"""Simple rooms built from dimensions instead of a CAD file.

- :func:`box_mesh`: a rectangular room from the origin to (length, width, height);
- :func:`extruded_mesh`: a floor plan (any simple polygon, convex or not) raised to a
  height, as for most halls, classrooms and offices.

Faces point into the room. Surfaces are named ``floor``, ``ceiling`` and ``walls``; each
wall is also ``wall_1``, ``wall_2``, ... in the order of the plan's edges (for a box:
y = 0, x = length, y = width, x = 0), so materials can go to all walls or to one.
"""

from __future__ import annotations

import numpy as np

from acoustic_spaces.geometry.mesh import RoomMesh
from acoustic_spaces.geometry.triangulate import triangulate_polygon


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def extruded_mesh(plan: np.ndarray, height: float) -> RoomMesh:
    """A room whose floor is the polygon ``plan`` (N, 2) at z = 0 and whose ceiling is
    the same polygon at z = ``height``."""
    plan = np.asarray(plan, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] != 2 or len(plan) < 3:
        raise ValueError("a floor plan needs at least three (x, y) corners")
    if height <= 0:
        raise ValueError("the room height must be positive")
    area = _signed_area(plan)
    if abs(area) < 1e-9:
        raise ValueError("the floor plan has no area")
    counterclockwise = plan if area > 0 else plan[::-1]
    n = len(plan)
    floor = np.column_stack([counterclockwise, np.zeros(n)])
    ceiling = np.column_stack([counterclockwise, np.full(n, float(height))])
    vertices = np.vstack([floor, ceiling])
    faces, face_labels = [], []
    labels = [("floor",), ("ceiling",)]
    for a, b, c in triangulate_polygon(floor):
        # Seen from above the plan runs counterclockwise: the floor faces up, the
        # ceiling (reversed) down, both into the room.
        up = np.cross(floor[b] - floor[a], floor[c] - floor[a])[2] > 0
        faces.append((a, b, c) if up else (a, c, b))
        face_labels.append(0)
        faces.append((n + a, n + c, n + b) if up else (n + a, n + b, n + c))
        face_labels.append(1)
    # Walls: the room lies to the left of each counterclockwise edge.
    edge_of = range(n) if area > 0 else [(n - 2 - k) % n for k in range(n)]
    for k, edge in enumerate(edge_of):
        i, j = k, (k + 1) % n
        labels.append((f"wall_{edge + 1}", "walls"))
        faces.append((i, n + j, j))
        faces.append((i, n + i, n + j))
        face_labels += [len(labels) - 1] * 2
    return RoomMesh(vertices, np.array(faces), np.array(face_labels), labels)


def box_mesh(size: tuple[float, float, float]) -> RoomMesh:
    """A rectangular room from (0, 0, 0) to ``size`` = (length, width, height)."""
    length, width, height = (float(v) for v in size)
    if min(length, width, height) <= 0:
        raise ValueError("box dimensions must be positive")
    plan = np.array([[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]])
    return extruded_mesh(plan, height)


__all__ = ["box_mesh", "extruded_mesh"]
