"""Triangulation of planar polygons, including concave ones (ear clipping)."""

from __future__ import annotations

import numpy as np


def triangulate_polygon(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Split a simple planar polygon into triangles.

    ``points`` is (N, 3). Returns index triples into ``points`` that keep the
    polygon's winding, so face normals stay as the source file defined them.
    Triangles and convex polygons are handled exactly; concave polygons such as
    L-shaped floors use ear clipping on the polygon's dominant plane.
    """
    n = len(points)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    normal = _newell_normal(points)
    if not np.any(normal):
        return [(0, i, i + 1) for i in range(1, n - 1)]
    drop = int(np.argmax(np.abs(normal)))
    keep = [axis for axis in range(3) if axis != drop]
    uv = points[:, keep]
    # Make the 2-D polygon counter-clockwise so "convex corner" has one meaning.
    sign = 1.0 if normal[drop] > 0 else -1.0
    if drop == 1:
        sign = -sign  # dropping y flips the handedness of the (x, z) plane

    remaining = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3 and guard < 4 * n * n:
        guard += 1
        clipped = False
        for k in range(len(remaining)):
            a, b, c = remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]
            if sign * _cross2(uv[a], uv[b], uv[c]) <= 1e-12:
                continue  # reflex or collinear corner
            if any(
                _inside_triangle(uv[p], uv[a], uv[b], uv[c], sign)
                for p in remaining
                if p not in (a, b, c)
            ):
                continue
            triangles.append((a, b, c))
            del remaining[k]
            clipped = True
            break
        if not clipped:
            break  # self-intersecting or degenerate: fall back to a fan below
    if len(remaining) == 3:
        triangles.append(tuple(remaining))  # type: ignore[arg-type]
    elif len(remaining) > 3:
        triangles.extend(
            (remaining[0], remaining[i], remaining[i + 1]) for i in range(1, len(remaining) - 1)
        )
    return triangles


def _newell_normal(points: np.ndarray) -> np.ndarray:
    nxt = np.roll(points, -1, axis=0)
    return np.array(
        [
            np.sum((points[:, 1] - nxt[:, 1]) * (points[:, 2] + nxt[:, 2])),
            np.sum((points[:, 2] - nxt[:, 2]) * (points[:, 0] + nxt[:, 0])),
            np.sum((points[:, 0] - nxt[:, 0]) * (points[:, 1] + nxt[:, 1])),
        ]
    )


def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _inside_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, sign: float
) -> bool:
    d1 = sign * _cross2(a, b, p)
    d2 = sign * _cross2(b, c, p)
    d3 = sign * _cross2(c, a, p)
    return d1 >= -1e-12 and d2 >= -1e-12 and d3 >= -1e-12
