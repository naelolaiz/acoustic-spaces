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
#: A vertex closer than this (metres) to an open edge of another face, between its ends,
#: splits that edge (a T-junction).
T_JUNCTION_TOLERANCE_M = 1e-5


@dataclass
class MeshReport:
    vertices: int
    faces: int
    merged_vertices: int
    #: Faces cleaning removed: degenerate ones, and copies of another face.
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

    #: Edges shared by three or another odd number of faces: where a partition or an
    #: opening between two rooms meets the walls, or a crack.
    odd_edges: int = 0
    #: Volume of each room (m3): the separate volumes of air the faces enclose, two or
    #: more where partitions or openings divide the model (see
    #: :mod:`acoustic_spaces.geometry.cells`).
    room_volumes_m3: tuple[float, ...] = ()
    #: Faces with no room or solid on either side: a hole joins the room behind them to
    #: the outside, or they stand outside the model.
    open_faces: int = 0
    #: Whether the faces are those of solids (``RoomMesh.solids``), which close off no
    #: room.
    solids: bool = False

    @property
    def closed(self) -> bool:
        """True when the faces enclose at least one room and every face has a room or a
        solid (the inside of a closed pillar) on one side at least. Free-standing panels
        inside a room leave it closed."""
        return self.faces > 0 and len(self.room_volumes_m3) > 0 and self.open_faces == 0

    @property
    def rooms(self) -> int:
        return len(self.room_volumes_m3)

    def summary(self) -> str:
        lo, hi = np.array(self.bounds_min), np.array(self.bounds_max)
        size = " x ".join(f"{d:.2f}" for d in hi - lo)
        if self.closed:
            state = "closed"
            if self.rooms > 1:
                volumes = ", ".join(f"{v:.2f}" for v in self.room_volumes_m3)
                state += f", {self.rooms} rooms of {volumes} m3"
        elif self.faces:
            state = f"open: {self.boundary_edges} boundary edges"
            if self.odd_edges:
                state += f", {self.odd_edges} edges shared by an odd number of faces"
            if self.open_faces:
                state += f", {self.open_faces} faces with no room on either side"
            if self.solids:
                state += ", the faces of solids (building elements), no room"
        else:
            state = "no faces"
        lines = [
            f"{self.faces} triangles, {self.vertices} vertices, extent {size} m",
            f"surface area {self.surface_area_m2:.2f} m2"
            + (f", volume {self.volume_m3:.2f} m3" if self.volume_m3 is not None else ""),
            state,
        ]
        if self.non_manifold_edges:
            lines.append(f"{self.non_manifold_edges} edges shared by more than two faces")
        if self.inconsistent_edges:
            lines.append(f"{self.inconsistent_edges} edges with inconsistent face orientation")
        if self.merged_vertices:
            lines.append(f"merged {self.merged_vertices} duplicate vertices")
        if self.removed_degenerate_faces:
            lines.append(f"removed {self.removed_degenerate_faces} degenerate or repeated faces")
        lines.extend(self.notes)
        return "\n".join(lines)


def clean_mesh(mesh: RoomMesh) -> tuple[RoomMesh, int, int]:
    """Merge duplicate vertices, drop degenerate faces, copies of a face and the
    vertices no face uses, split faces at T-junctions and orient the faces alike.

    A copy is a face with the same corners as an earlier one, turning the same way: the
    same face given twice, which would leave every edge of it shared by three faces and
    the model with no inside. Two faces with the same corners turning opposite ways
    both stay: that is where two closed models touch, each with its own face.

    A face whose edge runs past a corner of the faces on its other side (a T-junction,
    which CAD exporters leave, or close with a zero-area sliver that is dropped here) is
    split at that corner, so the faces share the edge and the model can close. Faces
    that meet along an edge are turned to agree with each other, so a closed
    model with a few faces flipped by its exporter still has a volume and an inside
    test that works; each connected part keeps the orientation most of its area had.
    Where two rooms modelled closed touch over part of a face (see
    :func:`_imprint_overlaps`), each of their faces there is cut along the other's
    outline, so that the two have the same outline where they lie together.
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
    # Copies: the corners from the lowest on, which keeps the way they turn.
    kept = np.flatnonzero(keep)
    lowest = np.argmin(faces[kept], axis=1)
    turned = faces[kept[:, None], (lowest[:, None] + np.arange(3)) % 3]
    _, once = np.unique(turned, axis=0, return_index=True)
    keep[kept] = False
    keep[kept[once]] = True
    faces, labels = _split_t_junctions(vertices, faces[keep], mesh.face_labels[keep])
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    faces = _orient_faces(faces, area)
    vertices, faces, labels, corners = _imprint_overlaps(vertices, faces, labels, mesh.solids)
    if len(corners):
        faces, labels = _split_t_junctions(vertices, faces, labels, corners)
    # A stray point of the file, or a corner only degenerate faces had, would count
    # wherever the model's extent is taken from its vertices.
    used, faces = np.unique(faces, return_inverse=True)
    faces = faces.reshape(-1, 3).astype(np.int64, copy=False)
    cleaned = RoomMesh(vertices[used], faces, labels, list(mesh.labels), mesh.solids)
    return cleaned, merged, int(np.count_nonzero(~keep))


def _edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every face edge (the three of face i are rows i, n + i and 2n + i) as corner
    pairs, the index of its undirected edge, and how many faces share each of those."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = directed.min(axis=1) * (int(faces.max(initial=0)) + 1) + directed.max(axis=1)
    _, edge, counts = np.unique(key, return_inverse=True, return_counts=True)
    return directed, edge.reshape(-1), counts


def open_edges_of(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The edges that only one of the faces has: (E, 2) their corners (vertex indices, in
    the order the face runs along them) and (E,) that face."""
    directed, edge, counts = _edge_counts(faces)
    rows = np.flatnonzero(counts[edge] == 1)
    return directed[rows], rows % len(faces)


def _split_t_junctions(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    corners: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The faces split at the vertices of odd edges, and at ``corners``, that lie on
    their edges, with their labels.

    An odd edge is one shared by an odd number of faces: an open edge, which no other
    face shares, or one of three faces along a line that the edge of a fourth runs past.
    Each end of one within ``T_JUNCTION_TOLERANCE_M`` of another face's edge, between
    its ends, becomes a corner of that face: it turns into a fan of faces from its
    opposite corner, in the same orientation. The edge may be open too, or shared: a
    partition drawn between two rooms ends on the edge where their floors meet. A face
    whose own opposite corner lies on the edge has no area at that tolerance: an
    exporter's sliver closing the T-junction, which the split closes instead, so it is
    dropped. A face with T-junctions on more than one edge is split one edge per pass.
    """
    for _ in range(8):
        if len(faces) == 0:
            break
        directed, edge, counts = _edge_counts(faces)
        odd_rows = np.flatnonzero(counts[edge] % 2 == 1)
        if len(odd_rows) == 0 and corners is None:
            break
        # Only the ends of odd edges (and ``corners``) can lie on another edge: sorted
        # along x, the ones within an edge's reach are one slice.
        ends = np.unique(directed[odd_rows])
        if corners is not None:
            ends = np.union1d(ends, corners)
        order = np.argsort(vertices[ends, 0], kind="stable")
        ends, xs = ends[order], vertices[ends[order], 0]
        n = len(faces)
        # The edges with an end of an odd edge within reach of their x range, besides
        # their own two.
        x = vertices[directed, 0]
        first = np.searchsorted(xs, x.min(axis=1) - T_JUNCTION_TOLERANCE_M)
        reach = np.searchsorted(xs, x.max(axis=1) + T_JUNCTION_TOLERANCE_M, side="right")
        reach -= first
        own = np.isin(directed, ends).sum(axis=1)
        rows = np.flatnonzero(reach > own)
        hit_row, hit_end, hit_t = _ends_on_edges(vertices, directed, ends, rows, first, reach)
        # Each face is split along the first of its edges (in the order of ``directed``)
        # that has an end on it.
        hit_face = hit_row % n
        _, once = np.unique(hit_face, return_index=True)
        replaced: dict[int, list[list[int]]] = {}
        for row in np.sort(hit_row[once]).tolist():
            on = slice(*np.searchsorted(hit_row, [row, row + 1]).tolist())
            near, t = hit_end[on], hit_t[on]
            face, k = row % n, row // n  # the face runs a -> b from its corner k
            a, b = directed[row]
            c = int(faces[face, (k + 2) % 3])
            if c in near:
                replaced[face] = []
                continue
            corners = [int(a), *near[np.argsort(t)].tolist(), int(b)]
            replaced[face] = [[p, q, c] for p, q in zip(corners[:-1], corners[1:], strict=True)]
        if not replaced:
            break
        kept = np.setdiff1d(np.arange(n), list(replaced))
        new = [f for split in replaced.values() for f in split]
        new_labels = [labels[face] for face, split in replaced.items() for _ in split]
        faces = np.concatenate([faces[kept], np.array(new, dtype=faces.dtype).reshape(-1, 3)])
        labels = np.concatenate([labels[kept], np.array(new_labels, dtype=labels.dtype)])
    return faces, labels


#: Faces whose unit normals and distances from the origin (metres) agree to this lie in
#: one plane for :func:`_imprint_overlaps`, as the engine groups them.
_PLANE_TOLERANCE = 1e-5


def _imprint_overlaps(
    vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, solids: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The faces cut where two rooms modelled closed touch over part of a face: the
    vertices with the new corners after them, the faces, their labels and the corners of
    the faces cut, for :func:`_split_t_junctions` to take into the faces along them.

    Two closed models that touch over part of a face (a narrow room's wall against a
    wider room's, a room upstairs on part of the ceiling of one below) each have a face
    there, and neither's edges run along the other's: every edge is shared by two
    faces, so no T-junction shows, the two faces share no outline and the model cannot
    tell where they lie together. The faces of a plane make sheets, joined across the
    edges they share lying on either side of them. Where two sheets face opposite ways,
    each with a room behind it and the space round the model in front (as the cells of
    :func:`~acoustic_spaces.geometry.cells.mesh_cells` have them before the cut), each of
    their faces is cut along the lines of the edges of the other's outline that run
    through it, into convex pieces that each lie wholly inside or outside the other,
    and each piece is split into triangles again, turning as the face did. Sheets with
    the same outline (a wall drawn once for each room, the two rooms' walls the same)
    are left as they are, and so are the faces of a shell standing in a room, which face
    the room's own the same way, and the faces of solids (``solids``). A new corner within
    ``T_JUNCTION_TOLERANCE_M`` of a vertex or of another new corner is that one."""
    none = np.zeros(0, dtype=np.int64)
    if solids or len(faces) < 2:
        return vertices, faces, labels, none
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal = cross / np.maximum(np.linalg.norm(cross, axis=1), 1e-300)[:, None]
    # One direction for each plane: that of its largest component as rounded, the first
    # of equal ones, as the engine's planes have it.
    rounded = np.round(normal / _PLANE_TOLERANCE)
    flip = np.take_along_axis(rounded, np.abs(rounded).argmax(axis=1)[:, None], 1)[:, 0] < 0
    normal = np.where(flip[:, None], -normal, normal)
    offset = np.einsum("ij,ij->i", normal, tri[:, 0])
    keys = np.round(np.column_stack([normal, offset]) / _PLANE_TOLERANCE).astype(np.int64)
    plane = np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)
    # The sheets: faces of a plane joined across the edges two of them share lying on
    # either side of it.
    n = len(faces)
    owner = np.tile(np.arange(n), 3)
    first = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    second = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    third = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    lo, hi = np.minimum(first, second), np.maximum(first, second)
    _, edge, uses = np.unique(
        np.column_stack([plane[owner], lo, hi]), axis=0, return_inverse=True, return_counts=True
    )
    edge = edge.reshape(-1)
    twice = np.flatnonzero(uses[edge] == 2)
    twice = twice[np.argsort(edge[twice], kind="stable")].reshape(-1, 2)
    along = vertices[hi[twice[:, 0]]] - vertices[lo[twice[:, 0]]]
    base = vertices[lo[twice[:, 0]]]
    out0 = np.cross(along, vertices[third[twice[:, 0]]] - base)
    out1 = np.cross(along, vertices[third[twice[:, 1]]] - base)
    joined = twice[np.einsum("ij,ij->i", out0, out1) < 0.0]
    sheet = _components(n, owner[joined])
    # Which way each sheet faces along its plane's direction: 1, -1, or 0 if both.
    count = int(sheet.max()) + 1
    way = np.where(flip, -1, 1)
    low = np.full(count, 2)
    high = np.full(count, -2)
    np.minimum.at(low, sheet, way)
    np.maximum.at(high, sheet, way)
    facing = np.where(low == high, low, 0)
    sheet_plane = np.zeros(count, dtype=np.int64)
    sheet_plane[sheet] = plane
    both = np.zeros((int(plane.max()) + 1, 2), dtype=bool)
    both[sheet_plane[facing != 0], (facing[facing != 0] + 1) // 2] = True
    several = np.flatnonzero(both.all(axis=1))
    if len(several) == 0:
        return vertices, faces, labels, none
    outline = np.ones(len(owner), dtype=bool)
    outline[joined.reshape(-1)] = False

    behind: list[np.ndarray] = []

    def outward() -> np.ndarray:
        """(sheets,) which way the space round the model lies from each sheet along its
        plane's direction, the sheet's faces having a room behind them: 1, -1, or 0."""
        if not behind:
            from acoustic_spaces.geometry.cells import mesh_cells

            cells = mesh_cells(vertices, faces)
            room = np.zeros(len(cells.volume), dtype=bool)
            room[cells.rooms] = True
            front, back = cells.side[:, 0], cells.side[:, 1]
            out = np.where(
                cells.outside[front] & room[back],
                1,
                np.where(cells.outside[back] & room[front], -1, 0),
            )
            lo_, hi_ = np.full(count, 2), np.full(count, -2)
            np.minimum.at(lo_, sheet, way * out)
            np.maximum.at(hi_, sheet, way * out)
            behind.append(np.where(lo_ == hi_, lo_, 0))
        return behind[0]

    def touching(mine: int, others: np.ndarray) -> np.ndarray:
        """Whether the sheets ``others`` and ``mine`` face opposite ways, each with a room
        behind it and the space round the model in front."""
        ok = facing[others] == -facing[mine]
        if facing[mine] == 0 or not ok.any():
            return ok & False
        ways = outward()
        return ok & (ways[mine] != 0) & (ways[others] == -ways[mine])

    pieces: dict[int, list[list[int]]] = {}
    points: list[np.ndarray] = []
    for p in several.tolist():
        members = np.flatnonzero(plane == p)
        rows = np.flatnonzero(outline & (plane[owner] == p))
        segments = (lo[rows], hi[rows], sheet[owner[rows]])
        pieces.update(
            _cut_plane(
                vertices,
                faces,
                members,
                sheet[members],
                segments,
                touching,
                normal[members[0]],
                points,
            )
        )
    if not pieces:
        return vertices, faces, labels, none

    # Each new corner is the vertex within the tolerance of it or of a new corner joined
    # to it that way, or else the first of those new corners.
    from scipy.spatial import cKDTree

    new = np.array(points, dtype=float).reshape(-1, 3)
    distance, nearest = cKDTree(vertices).query(new, distance_upper_bound=T_JUNCTION_TOLERANCE_M)
    near = cKDTree(new).query_pairs(T_JUNCTION_TOLERANCE_M) if len(new) else set()
    pairs = np.array(sorted(near), dtype=np.int64)
    group = _components(len(new), pairs.reshape(-1, 2))
    on_vertex = np.full(int(group.max(initial=-1)) + 1, -1, dtype=np.int64)
    at = np.flatnonzero(np.isfinite(distance))
    np.maximum.at(on_vertex, group[at], nearest[at])
    leader = np.full(len(on_vertex), len(new), dtype=np.int64)
    np.minimum.at(leader, group, np.arange(len(new)))
    fresh = (on_vertex[group] < 0) & (leader[group] == np.arange(len(new)))
    number = np.full(len(new), -1, dtype=np.int64)
    number[fresh] = len(vertices) + np.arange(int(fresh.sum()))
    target = np.where(on_vertex[group] >= 0, on_vertex[group], number[leader[group]])
    vertices = np.vstack([vertices, new[fresh]])

    cut = sorted(pieces)
    made = np.array(
        [[k if k >= 0 else int(target[-1 - k]) for k in t] for f in cut for t in pieces[f]],
        dtype=faces.dtype,
    ).reshape(-1, 3)
    made_labels = np.array([labels[f] for f in cut for _ in pieces[f]], dtype=labels.dtype)
    kept = np.setdiff1d(np.arange(len(faces)), cut)
    faces = np.concatenate([faces[kept], made])
    labels = np.concatenate([labels[kept], made_labels])
    return vertices, faces, labels, np.unique(made)


def _components(count: int, pairs: np.ndarray) -> np.ndarray:
    """(count,) the connected component of each of ``count`` nodes joined by ``pairs``
    (K, 2)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ones = np.ones(len(pairs), dtype=np.int8)
    graph = coo_matrix((ones, (pairs[:, 0], pairs[:, 1])), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def _cut_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    members: np.ndarray,
    sheets: np.ndarray,
    segments: tuple[np.ndarray, np.ndarray, np.ndarray],
    touching,
    normal: np.ndarray,
    points: list[np.ndarray],
) -> dict[int, list[list[int]]]:
    """The faces ``members`` of the plane with unit ``normal``, in ``sheets``, that the
    outline of a sheet touching theirs (``segments``: the ends of the outlines' edges and
    their sheets; ``touching(sheet, sheets)``, see :func:`_imprint_overlaps`) runs
    through, each as its pieces' triangles: corners as vertex indices, or as -1 - k for
    the new corner ``points[k]``, which this appends."""
    tol = T_JUNCTION_TOLERANCE_M
    # A frame in the plane, (u, w, normal) right-handed: a face that turns
    # counter-clockwise in (u, w) has the plane's normal.
    k = (int(np.abs(normal).argmax()) + 1) % 3
    u = -normal[k] * normal
    u[k] += 1.0
    u /= np.linalg.norm(u, axis=0)
    w = np.cross(normal, u)
    start, end, seg_sheet = segments
    used = np.unique(np.concatenate([faces[members].reshape(-1), start, end]))
    flat = np.column_stack(
        [np.einsum("ij,j->i", vertices[used], u), np.einsum("ij,j->i", vertices[used], w)]
    )

    def at(index: np.ndarray) -> np.ndarray:
        return flat[np.searchsorted(used, index)]

    seg_a, seg_b = at(start), at(end)
    out: dict[int, list[list[int]]] = {}
    begin = len(points)
    for face, sheet in zip(members.tolist(), sheets.tolist(), strict=True):
        p = at(faces[face])
        turn = float(_turn(p[:1], p[1:2], p[2:])[0])
        ids = faces[face] if turn > 0 else faces[face][::-1]
        through = (seg_sheet != sheet) & _runs_through(at(ids), seg_a, seg_b, tol)
        if through.any():
            through[through] = touching(sheet, seg_sheet[through])
        if not through.any():
            continue
        polygons = [list(zip(ids.tolist(), at(ids), strict=True))]
        for line in _lines(seg_a[through], seg_b[through], tol):
            polygons = [
                part for polygon in polygons for part in _split_convex(polygon, line, tol, points)
            ]
        triangles = []
        for polygon in polygons:
            keep = _corners(polygon, tol)
            for j in range(1, len(keep) - 1):
                t = [keep[0], keep[j], keep[j + 1]]
                triangles.append(t if turn > 0 else t[::-1])
        out[face] = triangles
    # The new corners in space, in the plane (at its faces' mean distance from the origin).
    level = float(np.mean(np.einsum("ij,j->i", vertices[faces[members, 0]], normal)))
    for j in range(begin, len(points)):
        points[j] = level * normal + points[j][0] * u + points[j][1] * w
    return out


def _turn(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Twice the signed area of the triangles (a, b, c) in the plane, (N, 2) each:
    positive where they turn counter-clockwise."""
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _runs_through(triangle: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """(N,) whether the segments from ``a`` to ``b`` (N, 2) run through the triangle
    (3, 2), counter-clockwise, farther than ``tol`` inside each of its edges."""
    low = np.zeros(len(a))
    high = np.ones(len(a))
    for k in range(3):
        p, q = triangle[k], triangle[(k + 1) % 3]
        edge = q - p
        length = float(np.hypot(edge[0], edge[1]))
        # Distance inside the edge, less the tolerance, at a and at b.
        da = (edge[0] * (a[:, 1] - p[1]) - edge[1] * (a[:, 0] - p[0])) / length - tol
        db = (edge[0] * (b[:, 1] - p[1]) - edge[1] * (b[:, 0] - p[0])) / length - tol
        with np.errstate(divide="ignore", invalid="ignore"):
            t = da / (da - db)
        entering = (da <= 0.0) & (db > 0.0)
        leaving = (da > 0.0) & (db <= 0.0)
        low = np.where(entering, np.maximum(low, t), low)
        high = np.where(leaving, np.minimum(high, t), high)
        high = np.where((da <= 0.0) & (db <= 0.0), -1.0, high)
    return high > low


def _lines(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """(L, 3) the lines of the segments from ``a`` to ``b`` (N, 2), once each: a unit
    normal (its larger component positive) and its distance from the origin."""
    d = b - a
    across = np.column_stack([-d[:, 1], d[:, 0]]) / np.hypot(d[:, 0], d[:, 1])[:, None]
    larger = np.where(np.abs(across[:, 0]) >= np.abs(across[:, 1]), across[:, 0], across[:, 1])
    across *= np.where(larger < 0.0, -1.0, 1.0)[:, None]
    lines = np.column_stack([across, np.einsum("ij,ij->i", across, a)])
    return lines[np.sort(np.unique(np.round(lines / tol), axis=0, return_index=True)[1])]


def _split_convex(
    polygon: list[tuple[int, np.ndarray]],
    line: np.ndarray,
    tol: float,
    points: list[np.ndarray],
) -> list[list[tuple[int, np.ndarray]]]:
    """The convex ``polygon`` (corners as an index, see :func:`_cut_plane`, and a point
    in the plane) split by ``line`` (see :func:`_lines`) where it has corners farther
    than ``tol`` on both sides of it; a corner within ``tol`` of it goes to both parts.
    A new corner where it crosses an edge is appended to ``points``."""
    distance = np.array([line[0] * q[0] + line[1] * q[1] - line[2] for _, q in polygon])
    side = np.where(distance > tol, 1, np.where(distance < -tol, -1, 0))
    if not ((side > 0).any() and (side < 0).any()):
        return [polygon]
    parts: tuple[list, list] = ([], [])
    for j, (corner, q) in enumerate(polygon):
        if side[j] >= 0:
            parts[0].append((corner, q))
        if side[j] <= 0:
            parts[1].append((corner, q))
        k = (j + 1) % len(polygon)
        if side[j] * side[k] < 0:
            t = distance[j] / (distance[j] - distance[k])
            point = q + t * (polygon[k][1] - q)
            points.append(point)
            parts[0].append((-len(points), point))
            parts[1].append((-len(points), point))
    return list(parts)


def _corners(polygon: list[tuple[int, np.ndarray]], tol: float) -> list[int]:
    """The corners of the convex ``polygon`` farther than ``tol`` from the line through
    the corners on either side of them: the others lie on its edges."""
    keep = list(polygon)
    changed = True
    while changed and len(keep) > 3:
        changed = False
        for j in range(len(keep)):
            a, b, c = keep[j - 1][1], keep[j][1], keep[(j + 1) % len(keep)][1]
            chord = c - a
            length = float(np.hypot(chord[0], chord[1]))
            off = abs(chord[0] * (b[1] - a[1]) - chord[1] * (b[0] - a[0])) / max(length, 1e-300)
            if off <= tol:
                del keep[j]
                changed = True
                break
    return [corner for corner, _ in keep]


#: (edge, end) pairs tested at once in :func:`_split_t_junctions`, to bound its memory.
_PAIRS_PER_BLOCK = 1 << 18


def _ends_on_edges(
    vertices: np.ndarray,
    directed: np.ndarray,
    ends: np.ndarray,
    rows: np.ndarray,
    first: np.ndarray,
    reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ends (``ends``, sorted along x) that lie on the edges ``rows`` of
    ``directed``, within ``T_JUNCTION_TOLERANCE_M`` of the edge and farther than that
    from both its ends: the edge's row, the end and where along the edge it lies (0 at
    its first corner, 1 at its second), in the order of the rows and, for each, of the
    ends. Edge ``row`` is tested against the ``reach[row]`` ends from ``first[row]`` on,
    those within its x range."""
    sizes = reach[rows]
    blocks = np.searchsorted(
        np.cumsum(sizes), np.arange(1, sizes.sum() // _PAIRS_PER_BLOCK + 1) * _PAIRS_PER_BLOCK
    )
    out_row, out_end, out_t = [], [], []
    for part in np.split(np.arange(len(rows)), blocks):
        row, size = rows[part], sizes[part]
        pair_row = np.repeat(row, size)
        step = np.arange(int(size.sum())) - np.repeat(np.cumsum(size) - size, size)
        end = ends[np.repeat(first[row], size) + step]
        pa, pb = vertices[directed[pair_row, 0]], vertices[directed[pair_row, 1]]
        d = pb - pa
        length = np.linalg.norm(d, axis=1)
        point = vertices[end]
        t = np.einsum("ij,ij->i", point - pa, d) / length**2
        off = np.linalg.norm(point - (pa + t[:, None] * d), axis=1)
        inside = (
            (off < T_JUNCTION_TOLERANCE_M)
            & (t * length > T_JUNCTION_TOLERANCE_M)
            & ((1.0 - t) * length > T_JUNCTION_TOLERANCE_M)
        )
        out_row.append(pair_row[inside])
        out_end.append(end[inside])
        out_t.append(t[inside])
    return np.concatenate(out_row), np.concatenate(out_end), np.concatenate(out_t)


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
    directed, edge, counts = _edge_counts(faces)
    owner = np.tile(np.arange(count), 3)
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
    directed, edge, counts = _edge_counts(faces)
    # Faces agree along an edge when as many run along it one way as the other.
    way = np.where(directed[:, 0] < directed[:, 1], 1.0, -1.0)
    balance = np.bincount(edge, weights=way, minlength=len(counts))

    boundary = int(np.count_nonzero(counts == 1))
    non_manifold = int(np.count_nonzero(counts > 2))
    odd = int(np.count_nonzero((counts > 2) & (counts % 2 == 1)))
    # Where three faces meet (a partition and the wall it divides), they cannot all agree.
    inconsistent = int(np.count_nonzero((counts > 1) & (counts % 2 == 0) & (balance != 0)))

    cells = mesh.cells()
    rooms = tuple(float(v) for v in cells.room_volumes)
    open_faces = int(np.count_nonzero(cells.open_faces))
    volume = None
    inward = None
    notes: list[str] = []
    if rooms and open_faces == 0:
        volume = float(sum(rooms))
        # Faces with a room on one side only: do their normals point into it?
        face_room = cells.face_room
        one_sided = (face_room >= 0).sum(axis=1) == 1
        front = face_room[one_sided, 0] >= 0
        if len(front) and front.all():
            inward = True
            notes.append("face normals point into the room")
        elif len(front) and not front.any():
            inward = False
            notes.append("face normals point outward")

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
        odd_edges=odd,
        room_volumes_m3=rooms,
        open_faces=open_faces,
        solids=mesh.solids,
    )


def points_inside(mesh: RoomMesh, points: np.ndarray) -> np.ndarray:
    """Whether each point lies inside a room of a closed mesh (any of its rooms, when
    partitions or openings divide it; see :mod:`acoustic_spaces.geometry.cells`)."""
    return room_of(mesh, points) >= 0


def room_of(mesh: RoomMesh, points: np.ndarray) -> np.ndarray:
    """The room each point lies in (an index into the rooms of ``mesh.cells()``), or
    -1 outside every room. Generalised winding numbers: the signed solid angles that a
    room's faces subtend at the point add up to 4 pi inside it and 0 outside."""
    from acoustic_spaces.geometry.cells import room_of as cell_room_of

    return cell_room_of(mesh.vertices, mesh.faces, mesh.cells(), points)


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


def triangle_gaps(one: np.ndarray, other: np.ndarray) -> np.ndarray:
    """The least distance (metres) between each of the triangles ``one`` (n, 3, 3: the
    three corners of each) and the one of ``other`` at its place: nothing where the two
    touch or cross."""
    one = np.asarray(one, dtype=np.float64).reshape(-1, 3, 3)
    other = np.asarray(other, dtype=np.float64).reshape(-1, 3, 3)
    best = np.full(len(one), np.inf)
    for these, those in ((one, other), (other, one)):
        a = those[:, 0]
        ab, ac = those[:, 1] - a, those[:, 2] - a
        for k in range(3):
            # A corner to the face; and nothing where the edge from it goes through the face.
            best = np.minimum(best, _squared_distance_to_triangles(these[:, k], a, ab, ac))
            edge = these[:, (k + 1) % 3] - these[:, k]
            best = np.where(_crosses(these[:, k], edge, a, ab, ac), 0.0, best)
    for j in range(3):
        for k in range(3):
            best = np.minimum(
                best,
                _squared_segment_gaps(
                    one[:, j],
                    one[:, (j + 1) % 3] - one[:, j],
                    other[:, k],
                    other[:, (k + 1) % 3] - other[:, k],
                ),
            )
    return np.sqrt(best)


def _crosses(
    start: np.ndarray, line: np.ndarray, a: np.ndarray, ab: np.ndarray, ac: np.ndarray
) -> np.ndarray:
    """Whether each segment from ``start`` along ``line`` goes through its triangle
    (a, a + ab, a + ac): its ends lie on the two sides of the triangle's plane, or one
    of them in it, and the point of the segment in the plane lies in the triangle. A
    segment in the plane goes through nothing; where rounding leaves its ends just off
    the plane, on its two sides, the point looked for is still one of the segment, so
    that a segment outside the triangle does not go through it."""
    normal = np.cross(ab, ac)
    before = ((start - a) * normal).sum(axis=1)
    after = before + (line * normal).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        point = start - a + line * (before / (before - after))[:, None]
        area = (normal * normal).sum(axis=1)
        u = (np.cross(point, ac) * normal).sum(axis=1) / area
        v = (np.cross(ab, point) * normal).sum(axis=1) / area
        return (before * after <= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)


def _squared_segment_gaps(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Squared least distance between each segment from ``p`` along ``d`` and its
    segment from ``q`` along ``e`` (the nearest points of the two, each kept on its
    segment)."""
    r = p - q
    dd, ee, de = (d * d).sum(axis=1), (e * e).sum(axis=1), (d * e).sum(axis=1)
    dr, er = (d * r).sum(axis=1), (e * r).sum(axis=1)
    across = dd * ee - de * de
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(across > 0, np.clip((de * er - dr * ee) / across, 0.0, 1.0), 0.0)
        t = (de * s + er) / ee
        s = np.where(
            t < 0,
            np.clip(-dr / dd, 0.0, 1.0),
            np.where(t > 1, np.clip((de - dr) / dd, 0.0, 1.0), s),
        )
    # A segment of no length is its start.
    s, t = np.nan_to_num(s), np.clip(np.nan_to_num(t), 0.0, 1.0)
    apart = r + d * s[:, None] - e * t[:, None]
    return (apart * apart).sum(axis=1)


#: Most pairs of a point and a triangle that :func:`triangles_near` measures at once.
_PAIRS = 1_000_000


def triangles_near(
    triangles: np.ndarray, points: np.ndarray, reach: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pairs of one of ``points`` and one of ``triangles`` (n, 3, 3: the three
    corners of each) no farther apart than ``reach`` metres: the points' indices, the
    triangles' and the distances (metres), in no order.

    Only the points in a triangle's box, grown by the reach, are measured against it:
    along the axis that leaves the fewest, each box takes the run of the sorted points
    that its span holds."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    found: tuple[list, list, list] = ([], [], [])
    if len(points) and len(tri):
        low, high = tri.min(axis=1) - reach, tri.max(axis=1) + reach
        runs = []
        for axis in range(3):
            order = np.argsort(points[:, axis], kind="stable")
            along = points[order, axis]
            first = np.searchsorted(along, low[:, axis], side="left")
            count = np.searchsorted(along, high[:, axis], side="right") - first
            runs.append((int(count.sum()), order, first, count))
        _, order, first, count = min(runs, key=lambda run: run[0])
        ends = np.cumsum(count)
        start = 0
        while start < len(tri):
            done = int(ends[start - 1]) if start else 0
            stop = max(start + 1, int(np.searchsorted(ends, done + _PAIRS, side="right")))
            held = count[start:stop]
            triangle = np.repeat(np.arange(start, stop), held)
            place = np.arange(len(triangle)) - np.repeat(np.cumsum(held) - held, held)
            point = order[np.repeat(first[start:stop], held) + place]
            inside = ((points[point] >= low[triangle]) & (points[point] <= high[triangle])).all(
                axis=1
            )
            triangle, point = triangle[inside], point[inside]
            a = tri[triangle, 0]
            distance = np.sqrt(
                _squared_distance_to_triangles(
                    points[point], a, tri[triangle, 1] - a, tri[triangle, 2] - a
                )
            )
            near = distance <= reach
            for kept, values in zip(found, (point, triangle, distance), strict=True):
                kept.append(values[near])
            start = stop
    if not found[0]:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    return tuple(np.concatenate(values) for values in found)


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
