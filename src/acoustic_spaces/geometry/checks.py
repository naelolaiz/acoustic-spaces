"""Mesh cleanup and the checks that tell a user whether a model is fit to simulate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces.geometry.mesh import RoomMesh

#: Vertices closer than this (metres) are merged.
MERGE_TOLERANCE_M = 1e-6
#: Faces smaller than this (square metres) are dropped as degenerate.
MIN_FACE_AREA_M2 = 1e-10


@dataclass
class MeshReport:
    vertices: int
    faces: int
    merged_vertices: int
    removed_degenerate_faces: int
    boundary_edges: int
    non_manifold_edges: int
    inconsistent_edges: int
    surface_area_m2: float
    volume_m3: float | None
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    notes: list[str] = field(default_factory=list)
    #: True when face normals point into the room, False outward, None when unknown
    #: (open model or inconsistent orientation).
    normals_inward: bool | None = None

    @property
    def closed(self) -> bool:
        """True when every edge is shared by exactly two faces."""
        return self.boundary_edges == 0 and self.non_manifold_edges == 0

    def summary(self) -> str:
        lo, hi = np.array(self.bounds_min), np.array(self.bounds_max)
        size = " x ".join(f"{d:.2f}" for d in hi - lo)
        lines = [
            f"{self.faces} triangles, {self.vertices} vertices, extent {size} m",
            f"surface area {self.surface_area_m2:.2f} m2"
            + (f", volume {self.volume_m3:.2f} m3" if self.volume_m3 is not None else ""),
            "closed" if self.closed else f"open: {self.boundary_edges} boundary edges",
        ]
        if self.non_manifold_edges:
            lines.append(f"{self.non_manifold_edges} edges shared by more than two faces")
        if self.inconsistent_edges:
            lines.append(f"{self.inconsistent_edges} edges with inconsistent face orientation")
        if self.merged_vertices:
            lines.append(f"merged {self.merged_vertices} duplicate vertices")
        if self.removed_degenerate_faces:
            lines.append(f"removed {self.removed_degenerate_faces} degenerate faces")
        lines.extend(self.notes)
        return "\n".join(lines)


def clean_mesh(mesh: RoomMesh) -> tuple[RoomMesh, int, int]:
    """Merge duplicate vertices, drop degenerate faces and orient the faces alike.

    Faces that meet along an edge are turned to agree with each other, so a closed
    model with a few faces flipped by its exporter still has a volume and an inside
    test that works; each connected part keeps the orientation most of its area had.
    Returns the cleaned mesh, the number of merged vertices and of removed faces.
    """
    keys = np.round(mesh.vertices / MERGE_TOLERANCE_M).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    vertices = mesh.vertices[first]
    faces = inverse[mesh.faces]
    merged = len(mesh.vertices) - len(vertices)

    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    distinct = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    keep = distinct & (area > MIN_FACE_AREA_M2)
    faces = _orient_faces(faces[keep], area[keep])
    cleaned = RoomMesh(vertices, faces, mesh.face_labels[keep], list(mesh.labels))
    return cleaned, merged, int(np.count_nonzero(~keep))


def _orient_faces(faces: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """The faces with their corner order reversed where needed to agree with their
    neighbours.

    Two faces sharing an edge agree when they run along it in opposite directions. The
    orientation spreads from one face of each connected part to the rest, breadth
    first, across every edge that exactly two faces share; each part then keeps the
    orientation that most of its area (``areas``, one per face) had. Edges shared by
    more than two faces pass nothing on, and a part that cannot be oriented (a Moebius
    strip) keeps the edges where its orientation clashes.
    """
    count = len(faces)
    if count == 0:
        return faces.copy()

    # Every edge of every face, and the faces on both sides of each two-face edge.
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(count), 3)
    # One number per undirected edge (faster to sort than pairs).
    key = directed.min(axis=1) * (int(faces.max()) + 1) + directed.max(axis=1)
    _, edge, counts = np.unique(key, return_inverse=True, return_counts=True)
    edge = edge.reshape(-1)
    shared = np.flatnonzero(counts[edge] == 2)
    pairs = shared[np.argsort(edge[shared], kind="stable")].reshape(-1, 2)
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    # Running the same way along their edge, one of the two faces has to turn.
    clash = np.all(directed[pairs[:, 0]] == directed[pairs[:, 1]], axis=1)

    # Neighbours of each face, as ranges of one sorted list.
    ends = np.concatenate([a, b])
    order = np.argsort(ends, kind="stable")
    start = np.searchsorted(ends[order], np.arange(count + 1)).tolist()
    neighbours = np.concatenate([b, a])[order].tolist()
    clashes = np.concatenate([clash, clash])[order].tolist()

    turn = [False] * count
    seen = [False] * count
    for seed in range(count):
        if seen[seed]:
            continue
        seen[seed] = True
        part, queue = [seed], deque([seed])
        while queue:
            face = queue.popleft()
            for k in range(start[face], start[face + 1]):
                other = neighbours[k]
                if not seen[other]:
                    seen[other] = True
                    turn[other] = turn[face] != clashes[k]
                    part.append(other)
                    queue.append(other)
        turned = np.array([turn[f] for f in part])
        if areas[part][turned].sum() > areas[part][~turned].sum():
            for f in part:
                turn[f] = not turn[f]
    out = faces.copy()
    flip = np.array(turn)
    out[flip] = out[flip][:, ::-1]
    return out


def check_mesh(mesh: RoomMesh, merged: int = 0, removed: int = 0) -> MeshReport:
    """Topology, size and orientation checks on a cleaned mesh."""
    faces = mesh.faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, counts = np.unique(undirected, axis=0, return_counts=True)
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)

    boundary = int(np.count_nonzero(counts == 1))
    non_manifold = int(np.count_nonzero(counts > 2))
    inconsistent = int(np.count_nonzero(directed_counts > 1))

    volume = None
    inward = None
    notes: list[str] = []
    if boundary == 0 and non_manifold == 0 and inconsistent == 0 and len(faces):
        tri = mesh.triangles
        signed = float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6)
        volume = abs(signed)
        inward = signed < 0
        notes.append(
            "face normals point into the room" if signed < 0 else "face normals point outward"
        )

    lo = mesh.vertices.min(axis=0) if len(mesh.vertices) else np.zeros(3)
    hi = mesh.vertices.max(axis=0) if len(mesh.vertices) else np.zeros(3)
    return MeshReport(
        vertices=len(mesh.vertices),
        faces=len(faces),
        merged_vertices=merged,
        removed_degenerate_faces=removed,
        boundary_edges=boundary,
        non_manifold_edges=non_manifold,
        inconsistent_edges=inconsistent,
        surface_area_m2=float(mesh.face_areas().sum()),
        volume_m3=volume,
        bounds_min=tuple(float(v) for v in lo),
        bounds_max=tuple(float(v) for v in hi),
        notes=notes,
        normals_inward=inward,
    )


def points_inside(mesh: RoomMesh, points: np.ndarray) -> np.ndarray:
    """Whether each point lies inside a closed mesh (generalised winding number).

    Sums the signed solid angle every triangle subtends at the point; the total is
    4*pi inside and 0 outside, whatever the orientation of the faces.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    tri = mesh.triangles
    inside = np.empty(len(points), dtype=bool)
    for k, p in enumerate(points):
        a, b, c = tri[:, 0] - p, tri[:, 1] - p, tri[:, 2] - p
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", b, c) * la
            + np.einsum("ij,ij->i", c, a) * lb
        )
        winding = np.sum(2.0 * np.arctan2(numerator, denominator)) / (4.0 * np.pi)
        inside[k] = abs(winding) > 0.5
    return inside


def distance_to_surface(mesh: RoomMesh, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest triangle of the mesh (metres)."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    tri = mesh.triangles
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    out = np.empty(len(points))
    for k, p in enumerate(points):
        out[k] = np.sqrt(_squared_distance_to_triangles(p, a, ab, ac).min(initial=np.inf))
    return out


def _squared_distance_to_triangles(
    p: np.ndarray, a: np.ndarray, ab: np.ndarray, ac: np.ndarray
) -> np.ndarray:
    """Squared distance from p to each triangle (a, a + ab, a + ac)."""
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    ap = p - a
    # Inside the prism over the triangle: the distance to the plane.
    with np.errstate(divide="ignore", invalid="ignore"):
        d_plane = np.einsum("ij,ij->i", ap, n) ** 2 / nn
        u = np.einsum("ij,ij->i", np.cross(ap, ac), n) / nn
        v = np.einsum("ij,ij->i", np.cross(ab, ap), n) / nn
    inside = (u >= 0) & (v >= 0) & (u + v <= 1) & (nn > 0)
    # Otherwise the nearest point is on an edge.
    best = np.full(len(a), np.inf)
    for start, edge in ((a, ab), (a, ac), (a + ab, ac - ab)):
        ee = np.einsum("ij,ij->i", edge, edge)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.clip(np.einsum("ij,ij->i", p - start, edge) / ee, 0.0, 1.0)
        t = np.where(ee > 0, t, 0.0)
        closest = start + edge * t[:, None]
        best = np.minimum(best, np.einsum("ij,ij->i", p - closest, p - closest))
    return np.where(inside, d_plane, best)
