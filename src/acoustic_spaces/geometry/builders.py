"""Simple rooms built from dimensions instead of a CAD file.

- :func:`box_mesh`: a rectangular room from the origin to (length, width, height);
- :func:`extruded_mesh`: a floor plan (any simple polygon, convex or not) raised to a
  height, as for most halls, classrooms and offices; :func:`plan_problem` says why a
  polygon cannot be one.

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


def plan_problem(plan: np.ndarray) -> str | None:
    """Why the polygon ``plan`` (N, 2) cannot be a floor plan, or None: too few corners,
    two corners in one place, no area, or walls that cross or touch."""
    plan = np.asarray(plan, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] != 2 or len(plan) < 3:
        return "a floor plan needs at least three (x, y) corners"
    n = len(plan)
    ends = np.roll(plan, -1, axis=0)
    size = float(np.ptp(plan, axis=0).max())
    tolerance = 1e-9 * max(size, 1.0)
    lengths = np.linalg.norm(ends - plan, axis=1)
    if np.any(lengths <= tolerance):
        k = int(np.argmax(lengths <= tolerance))
        return f"corners {k + 1} and {(k + 1) % n + 1} of the floor plan are in one place"
    # A wall that runs straight back along the one before it.
    into, out = plan - np.roll(plan, 1, axis=0), ends - plan
    turn = into[:, 0] * out[:, 1] - into[:, 1] * out[:, 0]
    back = (np.abs(turn) <= tolerance * lengths * np.roll(lengths, 1)) & (
        np.sum(into * out, axis=1) < 0
    )
    if back.any():
        return f"the floor plan turns straight back at corner {int(np.argmax(back)) + 1}"
    # Walls i and j (not neighbours) must not meet: a segment test for every pair.
    i, j = np.triu_indices(n, k=2)
    keep = (j - i) % n != n - 1  # the last wall and the first are neighbours
    i, j = i[keep], j[keep]
    a, b, c, d = plan[i], ends[i], plan[j], ends[j]

    def cross(o, p, q):
        return (p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1]) - (p[:, 1] - o[:, 1]) * (q[:, 0] - o[:, 0])

    d1, d2, d3, d4 = cross(c, d, a), cross(c, d, b), cross(a, b, c), cross(a, b, d)
    scale = tolerance * np.maximum(lengths[i], lengths[j])
    proper = (d1 * d2 < -(scale**2)) & (d3 * d4 < -(scale**2))

    def on(o, p, q, side):  # q on the segment o-p (collinear and inside its box)
        lo, hi = np.minimum(o, p) - tolerance, np.maximum(o, p) + tolerance
        return (np.abs(side) <= scale) & np.all((q >= lo) & (q <= hi), axis=1)

    touch = on(c, d, a, d1) | on(c, d, b, d2) | on(a, b, c, d3) | on(a, b, d, d4)
    bad = np.flatnonzero(proper | touch)
    if len(bad):
        k = bad[0]
        return f"walls {i[k] + 1} and {j[k] + 1} of the floor plan cross or touch"
    if abs(_signed_area(plan)) < 1e-9:
        return "the floor plan has no area"
    return None


def extruded_mesh(plan: np.ndarray, height: float) -> RoomMesh:
    """A room whose floor is the polygon ``plan`` (N, 2) at z = 0 and whose ceiling is
    the same polygon at z = ``height``. The plan must be a simple polygon (see
    :func:`plan_problem`)."""
    plan = np.asarray(plan, dtype=np.float64)
    problem = plan_problem(plan)
    if problem is not None:
        raise ValueError(problem)
    if height <= 0:
        raise ValueError("the room height must be positive")
    area = _signed_area(plan)
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


__all__ = ["box_mesh", "extruded_mesh", "plan_problem"]
