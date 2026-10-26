"""The rooms of a model: the separate volumes of air its faces enclose.

Two rooms can share a face: a partition between them, or an opening (a face that lets
all sound through). Such a model has edges where three faces meet (the partition and the
wall it divides), which no single inside and outside describes, so the rooms are found
from the sides of the faces instead.

Around each edge, the faces that meet there are sorted by angle. Between two neighbours
lies a wedge of space onto which one side of each opens, so those two sides face the
same volume. Joining sides edge by edge splits the sides of all the faces into cells. A
cell enclosed by faces has a positive volume; the space around the model, a negative
one. A shell inside another one (a pillar standing in a room) touches it at no edge, so
its outer sides join the cell they lie in, which points just off its faces tell.

Rooms are the enclosed cells that the outside of the model borders. A cell enclosed by
rooms alone is solid: the inside of a pillar modelled as a closed box. Both sides of a
free-standing panel face the same cell, and the two coincident faces where two closed
models touch enclose a cell with no volume, which is neither.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

#: Cells whose volume is below this share of the model's bounding box hold nothing: the
#: sliver between coincident faces, or the inside of a model that is open (joined to the
#: outside through a hole, its faces have the same cell on both sides).
EMPTY_SHARE = 1e-9
#: Angles (radians) around an edge closer than this count as equal: coincident faces.
_SAME_ANGLE = 1e-9
#: The most planes of a shell's faces whose points vote on the cell it lies in (see
#: ``_nest``): a box's six.
_NEST_PLANES = 6


@dataclass(frozen=True, eq=False)
class MeshCells:
    """The cells of a mesh (see the module notes)."""

    #: (F, 2) cell of the front (the side the face's normal points to) and of the back
    #: of each face.
    side: np.ndarray
    #: (C,) volume of each cell (m3): positive when enclosed, negative for the space
    #: around the model, about 0 for slivers and the inside of an open model.
    volume: np.ndarray
    #: (R,) the cells that are rooms, in the order of their first face.
    rooms: np.ndarray

    @property
    def outside(self) -> np.ndarray:
        """(C,) True for the cells of the space around the model."""
        return self.volume < -self._tolerance

    @property
    def enclosed(self) -> np.ndarray:
        """(C,) True for cells with a volume: rooms and solids."""
        return self.volume > self._tolerance

    @property
    def _tolerance(self) -> float:
        return float(np.abs(self.volume).max(initial=0.0)) * EMPTY_SHARE

    @property
    def face_room(self) -> np.ndarray:
        """(F, 2) room (an index into :attr:`rooms`) of the front and back of each face,
        or -1 where a side faces no room."""
        index = np.full(len(self.volume), -1, dtype=np.int64)
        index[self.rooms] = np.arange(len(self.rooms))
        return index[self.side]

    @property
    def room_volumes(self) -> np.ndarray:
        """(R,) volume of each room (m3)."""
        return self.volume[self.rooms]

    @property
    def open_faces(self) -> np.ndarray:
        """(F,) True for faces with neither side in an enclosed cell: a hole joins the
        room behind them to the outside, or they stand outside the model."""
        return ~self.enclosed[self.side].any(axis=1)

    @property
    def closed(self) -> bool:
        """At least one room, and every face has a side in an enclosed cell."""
        return len(self.rooms) > 0 and not self.open_faces.any()


def mesh_cells(vertices: np.ndarray, faces: np.ndarray) -> MeshCells:
    """The cells of the mesh with ``vertices`` (V, 3) and ``faces`` (F, 3)."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = len(faces)
    if n == 0:
        return MeshCells(np.zeros((0, 2), np.int64), np.zeros(0), np.zeros(0, np.int64))
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal = cross / np.maximum(np.linalg.norm(cross, axis=1), 1e-300)[:, None]

    # Every corner pair of every face (rows f, n + f, 2n + f), as an edge from its lower
    # to its higher vertex, with the face's third corner.
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    third = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    owner = np.tile(np.arange(n), 3)
    lo, hi = directed.min(axis=1), directed.max(axis=1)
    _, edge = np.unique(lo * (int(faces.max()) + 1) + hi, return_inverse=True)
    edge = edge.reshape(-1)
    along = vertices[hi] - vertices[lo]
    along /= np.maximum(np.linalg.norm(along, axis=1), 1e-300)[:, None]
    # The direction from the edge into each face, square to the edge.
    into = vertices[third] - vertices[lo]
    into -= np.einsum("ij,ij->i", into, along)[:, None] * along
    into /= np.maximum(np.linalg.norm(into, axis=1), 1e-300)[:, None]

    # The angle of each face around its edge, from the edge's first face.
    by_edge = np.argsort(edge, kind="stable")
    start = np.searchsorted(edge[by_edge], np.arange(edge.max() + 1))
    reference = into[by_edge[start[edge]]]
    turn = np.cross(along, reference)
    angle = np.mod(
        np.arctan2(np.einsum("ij,ij->i", into, turn), np.einsum("ij,ij->i", into, reference)),
        2.0 * np.pi,
    )
    angle = np.where(angle > 2.0 * np.pi - _SAME_ANGLE, 0.0, angle)
    # Whether the front of each face looks towards growing angles.
    ahead = np.einsum("ij,ij->i", normal[owner], np.cross(along, into)) > 0.0
    # Coincident faces: the ones whose front looks back first, so that the sliver between
    # two faces that look away from each other has both their backs.
    level = np.round(angle / _SAME_ANGLE).astype(np.int64)
    order = np.lexsort((owner, ahead, level, edge))
    group = edge[order]
    first = np.searchsorted(group, group, side="left")
    last = np.searchsorted(group, group, side="right") - 1
    following = np.where(np.arange(len(order)) == last, first, np.arange(len(order)) + 1)
    a, b = order, order[following]
    # Side 0 is the front, 1 the back: face a's side looking ahead, face b's looking back.
    side_a = 2 * owner[a] + np.where(ahead[a], 0, 1)
    side_b = 2 * owner[b] + np.where(ahead[b], 1, 0)
    label = _components(2 * n, side_a, side_b)

    # Volume of each cell: its sides with the normal pointing out of it.
    cone = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    side = label.reshape(n, 2)
    count = int(label.max()) + 1
    volume = np.bincount(side[:, 0], weights=-cone, minlength=count) + np.bincount(
        side[:, 1], weights=cone, minlength=count
    )
    side, volume = _nest(tri, normal, side, volume)
    tolerance = float(np.abs(volume).max(initial=0.0)) * EMPTY_SHARE
    outside = volume < -tolerance
    enclosed = volume > tolerance
    # Rooms: enclosed cells that a face shares with the outside.
    front, back = side[:, 0], side[:, 1]
    bordering = np.concatenate([front[outside[back]], back[outside[front]]])
    is_room = np.zeros(len(volume), dtype=bool)
    is_room[bordering] = True
    is_room &= enclosed
    first_face = np.full(len(volume), 2 * n, dtype=np.int64)
    np.minimum.at(first_face, side.reshape(-1), np.arange(2 * n))
    rooms = np.flatnonzero(is_room)
    rooms = rooms[np.argsort(first_face[rooms], kind="stable")]
    return MeshCells(side=side, volume=volume, rooms=rooms)


def _components(nodes: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected component (numbered from 0 in order of first node) of each of ``nodes``
    nodes joined pairwise by ``a`` and ``b``."""
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(nodes, nodes))
    _, label = connected_components(graph, directed=False)
    _, first, inverse = np.unique(label, return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(first))
    return rank[inverse.reshape(-1)].astype(np.int64)


def _nest(tri, normal, side, volume):
    """Joins the outer sides of shells, and panels, that lie inside a cell to that cell.

    A shell joined to the rest at no edge is a cell of negative volume by itself; a
    free-standing panel that touches nothing, a cell of no volume with both sides of its
    faces. A point just off a face, on the cell's side, tells which enclosed cell it lies
    in (the smallest that holds it), if any. A face may lie on another surface, as a
    platform's underside on the floor, and its point then on that surface's far side:
    points off the largest faces of up to ``_NEST_PLANES`` planes vote by those faces'
    area, until the rest could no longer change the outcome.
    """
    tolerance = float(np.abs(volume).max(initial=0.0)) * EMPTY_SHARE
    both = side[side[:, 0] == side[:, 1], 0]
    loose = np.zeros(len(volume), dtype=bool)
    loose[both] = np.abs(volume[both]) <= tolerance
    shells = np.flatnonzero((volume < -tolerance) | loose)
    if len(shells) < 2 and not loose.any():
        return side, volume
    size = float(np.ptp(tri.reshape(-1, 3), axis=0).max()) or 1.0  # the faces' extent
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    enclosed = np.flatnonzero(volume > tolerance)
    parent = np.arange(len(volume))
    # The sides of each cell, in the order of their faces.
    order = np.argsort(side.reshape(-1), kind="stable")
    bounds = np.searchsorted(side.reshape(-1)[order], np.stack([shells, shells + 1]))
    for shell, start, stop in zip(shells.tolist(), *bounds.tolist(), strict=True):
        # Only a cell larger than the shell can hold it (none holds the model's outside).
        among = [c for c in enclosed.tolist() if volume[c] > -volume[shell]]
        if not among:
            continue
        rows, column = np.divmod(order[start:stop], 2)
        # The largest face of each plane the shell's sides lie in, the largest first.
        probes: list[tuple[int, np.ndarray]] = []
        planes: list[tuple[np.ndarray, float]] = []
        for k in np.argsort(-area[rows], kind="stable").tolist():
            face = int(rows[k])
            out = (-1.0 if column[k] else 1.0) * normal[face]  # off the shell's side
            offset = float(out @ tri[face, 0])
            if any(
                out @ seen > 1.0 - 1e-9 and abs(offset - at) <= 1e-6 * size for seen, at in planes
            ):
                continue
            planes.append((out, offset))
            probes.append((face, out))
            if len(probes) == _NEST_PLANES:
                break
        votes: dict[int, float] = {}
        left = float(sum(area[face] for face, _ in probes))
        for face, out in probes:
            point = tri[face].mean(axis=0) + 1e-6 * size * out
            holders = [c for c in among if _winding(tri, normal, side, c, point) > 0.5]
            holder = min(holders, key=lambda c: volume[c]) if holders else -1
            votes[holder] = votes.get(holder, 0.0) + area[face]
            left -= area[face]
            leading = sorted(votes.values(), reverse=True) + [0.0]
            if leading[0] > leading[1] + left:
                break
        # A tie, as between the two sides of a rug lying on the floor, goes to a cell.
        holder = max(votes, key=lambda c: (votes[c], c >= 0))
        if holder >= 0:
            parent[shell] = holder
    _, compact = np.unique(parent, return_inverse=True)
    compact = compact.reshape(-1)
    return compact[side], np.bincount(compact, weights=volume)


def _winding(tri, normal, side, cell, point) -> float:
    """Winding number of the boundary of ``cell`` (its sides, normals out of it) around
    ``point``: 1 inside, 0 outside."""
    total = 0.0
    for column, sign in ((0, -1.0), (1, 1.0)):
        rows = np.flatnonzero(side[:, column] == cell)
        if len(rows):
            total += sign * float(solid_angles(tri[rows], point).sum())
    return total / (4.0 * np.pi)


def solid_angles(tri: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Signed solid angle (sr) each triangle (n, 3, 3) subtends at ``point``, positive
    when the point lies behind it (against its normal) (Van Oosterom and Strackee)."""
    a, b, c = tri[:, 0] - point, tri[:, 1] - point, tri[:, 2] - point
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    return 2.0 * np.arctan2(numerator, denominator)


def room_of(vertices: np.ndarray, faces: np.ndarray, cells: MeshCells, points) -> np.ndarray:
    """The room (an index into ``cells.rooms``) each point lies in, or -1."""
    return cell_of(vertices, faces, cells, points, cells.rooms)


def cell_of(vertices: np.ndarray, faces: np.ndarray, cells: MeshCells, points, among) -> np.ndarray:
    """Which of the cells ``among`` each point lies in (an index into ``among``), or
    -1 for none of them."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64)).reshape(-1, 3)
    out = np.full(len(points), -1, dtype=np.int64)
    among = np.asarray(among, dtype=np.int64).reshape(-1)
    if len(points) == 0 or len(among) == 0:
        return out
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
    for k, cell in enumerate(among.tolist()):
        front = np.flatnonzero(cells.side[:, 0] == cell)
        back = np.flatnonzero(cells.side[:, 1] == cell)
        # Sides facing the cell, turned so their normals point out of it.
        boundary = np.concatenate([tri[front][:, ::-1], tri[back]])
        for i in np.flatnonzero(out < 0).tolist():
            if solid_angles(boundary, points[i]).sum() / (4.0 * np.pi) > 0.5:
                out[i] = k
    return out
