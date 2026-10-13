"""Early reflections: exact specular paths with their delay, direction and band energies.

Two ways to find the candidate image sources, then one exact check:

- ``ray_traced_candidates``: deterministic specular rays from the source; every sequence
  of planes a ray hits becomes a candidate. The cost grows with the number of rays, not
  with the number of surfaces, so this suits large models (the default).
- ``exhaustive_candidates``: every sequence of planes up to the maximum order, pruned to
  images in front of the next plane. Complete, but grows as planes ** order.

``find_paths`` then traces each candidate back from the listener: the path must hit the
expected planes in order and reach the source unobstructed. Paths found are exact
whichever way they were proposed. A path through the edge where two planes meet (a
corner of a box, with the listener and source placed symmetrically) is kept once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.model import TraceModel, source_weights

#: Distance (metres) below which a point counts as lying on a plane.
_ON_PLANE = 1e-9
#: Tolerance, relative to the model size, for a path reflecting exactly on an edge.
_EDGE_TOLERANCE = 1e-7
#: Upper bound on candidates for the exhaustive search.
MAX_CANDIDATES = 2_000_000


@dataclass
class EarlyReflections:
    """Specular paths from one source to one listener (order 0 is the direct sound)."""

    order: np.ndarray  # (n,) reflections along the path
    delay_s: np.ndarray  # (n,)
    distance_m: np.ndarray  # (n,)
    direction: np.ndarray  # (n, 3) unit vectors from the listener towards the arrival
    energy: np.ndarray  # (n, bands) time-integrated intensity, unit-energy source
    points: list[np.ndarray] = field(default_factory=list)  # (order + 2, 3) per path
    faces: list[np.ndarray] = field(default_factory=list)  # faces hit, source side first

    def __len__(self) -> int:
        return len(self.order)

    @classmethod
    def empty(cls, bands: int) -> EarlyReflections:
        return cls(
            order=np.zeros(0, dtype=np.int64),
            delay_s=np.zeros(0),
            distance_m=np.zeros(0),
            direction=np.zeros((0, 3)),
            energy=np.zeros((0, bands)),
        )

    @classmethod
    def concatenate(cls, parts: list[EarlyReflections], bands: int) -> EarlyReflections:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty(bands)
        out = cls(
            order=np.concatenate([p.order for p in parts]),
            delay_s=np.concatenate([p.delay_s for p in parts]),
            distance_m=np.concatenate([p.distance_m for p in parts]),
            direction=np.concatenate([p.direction for p in parts]),
            energy=np.concatenate([p.energy for p in parts]),
            points=[x for p in parts for x in p.points],
            faces=[x for p in parts for x in p.faces],
        )
        return out.sorted()

    def sorted(self) -> EarlyReflections:
        """The same paths in order of arrival."""
        idx = np.argsort(self.delay_s, kind="stable")
        return EarlyReflections(
            order=self.order[idx],
            delay_s=self.delay_s[idx],
            distance_m=self.distance_m[idx],
            direction=self.direction[idx],
            energy=self.energy[idx],
            points=[self.points[i] for i in idx],
            faces=[self.faces[i] for i in idx],
        )


def ray_traced_candidates(
    model: TraceModel, source: np.ndarray, max_order: int, rays: int, threads: int = 0
) -> list[np.ndarray]:
    """Plane sequences hit by ``rays`` specular rays; entry k-1 has shape (n, k)."""
    from acoustic_spaces import _native

    if max_order <= 0:
        return []
    faces = _native.specular_sequences(
        model.bvh, np.asarray(source, dtype=np.float64), int(rays), int(max_order), int(threads)
    )
    planes = np.where(faces >= 0, model.face_plane[np.maximum(faces, 0)], -1)
    out = []
    for k in range(1, max_order + 1):
        prefix = planes[:, :k]
        prefix = prefix[np.all(prefix >= 0, axis=1)]
        out.append(np.unique(prefix, axis=0) if len(prefix) else np.zeros((0, k), np.int64))
    return out


def exhaustive_candidates(
    model: TraceModel, source: np.ndarray, max_order: int, limit: int = MAX_CANDIDATES
) -> list[np.ndarray]:
    """Every plane sequence whose images stay in front of the next plane."""
    normals, offsets = model.plane_normal, model.plane_offset
    seqs = np.zeros((1, 0), dtype=np.int64)
    images = np.asarray(source, dtype=np.float64)[None, :]
    out = []
    total = 0
    for _ in range(max_order):
        new_seqs, new_images = [], []
        chunk = max(1, 4_000_000 // max(1, model.plane_count))
        for start in range(0, len(seqs), chunk):
            s, img = seqs[start : start + chunk], images[start : start + chunk]
            dist = img @ normals.T - offsets  # (n, planes)
            ok = dist > _ON_PLANE if model.oriented else np.abs(dist) > _ON_PLANE
            if s.shape[1]:
                ok[np.arange(len(s)), s[:, -1]] = False
            rows, cols = np.nonzero(ok)
            new_seqs.append(np.column_stack([s[rows], cols]))
            new_images.append(model.mirror(img[rows], cols))
        seqs = np.concatenate(new_seqs).astype(np.int64)
        images = np.concatenate(new_images)
        total += len(seqs)
        if total > limit:
            raise ValueError(
                f"more than {limit} image sources up to order {len(out) + 1}; lower "
                "max_order or use the ray_tracing early method"
            )
        out.append(seqs)
    return out


def _distance_to_plane(model: TraceModel, points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    normals = model.plane_normal[planes]
    return np.abs(np.einsum("ij,ij->i", points, normals) - model.plane_offset[planes])


def _face_at(model: TraceModel, plane: int, point: np.ndarray, tolerance: float) -> int:
    """A face of ``plane`` that ``point`` lies on (edges included), or -1."""
    if abs(model.plane_normal[plane] @ point - model.plane_offset[plane]) > tolerance:
        return -1
    candidates = np.flatnonzero(model.face_plane == plane)
    tri = model.room.mesh.triangles[candidates]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    area2 = np.einsum("ij,ij->i", n, n)
    # Barycentric coordinates of the point projected onto each triangle.
    w_a = np.einsum("ij,ij->i", np.cross(c - b, point - b), n) / area2
    w_b = np.einsum("ij,ij->i", np.cross(a - c, point - c), n) / area2
    w_c = 1.0 - w_a - w_b
    edge = np.sqrt(area2) / np.maximum(
        np.linalg.norm(np.stack([b - a, c - b, a - c]), axis=2).max(axis=0), 1e-300
    )  # twice the area over the longest edge: a height of the triangle
    slack = tolerance / np.maximum(edge, 1e-300)
    inside = np.flatnonzero(np.min(np.stack([w_a, w_b, w_c]), axis=0) >= -slack)
    return int(candidates[inside[0]]) if len(inside) else -1


def find_paths(
    model: TraceModel,
    source: np.ndarray,
    listener: np.ndarray,
    candidates: list[np.ndarray],
    *,
    speed_of_sound: float,
    air: np.ndarray,
    max_delay_s: float,
    directivity: Directivity | None = None,
    orientation: np.ndarray | None = None,
) -> EarlyReflections:
    """Validate candidate plane sequences for one listener; add the direct sound.

    Each path's energy is weighted by the source's ``directivity`` (default: omni) in
    the direction it leaves the source; ``orientation`` has the source's forward, left
    and up axes as columns.
    """
    source = np.asarray(source, dtype=np.float64)
    listener = np.asarray(listener, dtype=np.float64)
    bands = model.absorption.shape[1]
    reflection = (1.0 - model.absorption) * (1.0 - model.scattering)  # specular energy

    tolerance = _EDGE_TOLERANCE * float(np.ptp(model.room.mesh.vertices, axis=0).max())
    parts = []
    direct = np.zeros((1, 0), dtype=np.int64)
    for seqs in [direct, *candidates]:
        if len(seqs) == 0:
            continue
        k = seqs.shape[1]
        # Image chain: level j is the source mirrored across the first j planes.
        chain = [np.broadcast_to(source, (len(seqs), 3)).copy()]
        for j in range(k):
            chain.append(model.mirror(chain[-1], seqs[:, j]))
        distance = np.linalg.norm(chain[-1] - listener, axis=1)
        keep = distance / speed_of_sound <= max_delay_s
        seqs, chain, distance = seqs[keep], [c[keep] for c in chain], distance[keep]
        n = len(seqs)
        if n == 0:
            continue

        # Trace back from the listener towards each image in turn.
        current = np.broadcast_to(listener, (n, 3)).copy()
        valid = np.ones(n, dtype=bool)
        faces = np.full((n, k), -1, dtype=np.int64)
        points = [None] * (k + 1)
        points[k] = current.copy()
        ignore = np.full(n, -1, dtype=np.int64)
        for j in range(k - 1, -1, -1):
            plane = seqs[:, j]
            # A reflection point on an edge already lies on the next plane too: the path
            # reflects off both at the same point (a corner), so there is nothing to trace.
            corner = valid & (_distance_to_plane(model, current, plane) <= tolerance)
            if j == k - 1:
                corner[:] = False
            trace = valid & ~corner
            delta = chain[j + 1] - current
            length = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(length, 1e-300)[:, None]
            t, face = model.intersect(current, direction, length, ignore)
            face = np.where(trace, face, -1)
            hit = face >= 0
            current = current + direction * np.where(hit, t, 0.0)[:, None]
            valid &= hit | corner
            # A hit on another face exactly where it meets the expected plane also counts.
            wrong = hit & (model.face_plane[np.maximum(face, 0)] != plane)
            for i in np.flatnonzero(wrong | corner):
                face[i] = _face_at(model, plane[i], current[i], tolerance)
            valid &= face >= 0
            faces[:, j] = face
            points[j] = current.copy()
            ignore = face
        delta = source - current
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1e-300)[:, None]
        _, face = model.intersect(current, direction, length * (1.0 - 1e-9), ignore)
        valid &= face < 0
        if not valid.any():
            continue

        idx = np.flatnonzero(valid)
        # Plane sequences that commute (perpendicular planes) give the same image.
        _, unique = np.unique(np.round(chain[-1][idx] / tolerance), axis=0, return_index=True)
        idx = idx[np.sort(unique)]
        dist = distance[idx]
        energy = np.exp(-np.outer(dist, air)) / (4.0 * np.pi * dist[:, None] ** 2)
        for j in range(k):
            energy *= reflection[faces[idx, j]]
        first = points[0][idx] if k else np.broadcast_to(listener, (len(idx), 3))
        departure = first - source
        departure /= np.maximum(np.linalg.norm(departure, axis=1), 1e-300)[:, None]
        energy *= source_weights(directivity, orientation, departure)
        last = points[k - 1][idx] if k else np.broadcast_to(source, (len(idx), 3))
        arrival = last - listener
        arrival /= np.maximum(np.linalg.norm(arrival, axis=1), 1e-300)[:, None]
        path_points = [
            np.vstack([source, *[points[j][i] for j in range(k)], listener]) for i in idx
        ]
        parts.append(
            EarlyReflections(
                order=np.full(len(idx), k, dtype=np.int64),
                delay_s=dist / speed_of_sound,
                distance_m=dist,
                direction=arrival,
                energy=energy,
                points=path_points,
                faces=[faces[i].copy() for i in idx],
            )
        )
    return EarlyReflections.concatenate(parts, bands)
