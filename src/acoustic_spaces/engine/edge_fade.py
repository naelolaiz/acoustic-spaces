"""The terms at the edges of every specular path, so that each path fades in and out at
the edges that cut it off instead of stepping.

The edge diffraction (:mod:`~acoustic_spaces.engine.diffraction`) bends the direct sound
round each edge and, with ``first_order``, the first-order reflections, one event at a
time: each term it adds stands alone. A path with more reflections, such as the copy of
a first-order reflection off the ground, or a second-order reflection, would come or go
at full strength where an edge cuts it off: where a leg of it passes an edge, or where a
point it reflects at passes the edge of the face. Here every specular path, found or
just missed, takes the terms of the uniform theory of diffraction (Kouyoumjian and
Pathak, 1974) at each of its events.

An event is a leg of the path, where it passes edges, or a reflection, near the edges
of the face it reflects off. Its term at an edge is the UTD's term of the boundary that
belongs to it: the incident wave's at a leg, the reflection's off the face at a
reflection. It runs from the path's image on the source side (the source mirrored in
the reflections before) to its image on the listener side (the listener mirrored in
the reflections after), so that the path over the edge has the path's own length near
the boundary, relative to the path.

The terms of one event add: every edge counts on its own side of its boundary, as the
edges round an opening do in the boundary diffraction wave (Rubinowicz, 1917), so that
both edges of a gap count where the path goes through it, and both where one of them
cuts it off. The events of a path multiply, as the path must get past each in turn:
with b = 1 where the path gets past an event and 0 where it does not, and G the sum of
the event's terms, the path carries the product of b + G over its events. Each term is
added with its share of that product, less what the edge diffraction already adds for
it (its term on the lit side where the path is found, on the dark side where it is
not), so that both together make the product.

A path that is not found is followed where one event stops it and an edge of what stops
it explains that, on the dark side of its boundary: a leg blocked by a face of the edge,
or a point of reflection past the edge of its face. Where a point of reflection lies
past the end of its plane's faces at an edge that stands on one of them (a corner of the
room on the floor; not a wall as high as a roof elsewhere in the plane), the face of the
edge that blocks a leg next to it stops the path, and the reflection takes the face
nearest to that point. Its reflection points are those of the image method; of the
orders of reflections off perpendicular planes, which have one image, the one that can
be. Paths that pass through a surface that lets sound through take no terms, as they are
not the geometric wave the terms continue.

Reflections off perpendicular planes commute: in either order they give one image, and
the path takes the order its points come in along the way. The path over an edge may
take another, such as off the floor after a corner that stands on it rather than before,
so each path also takes the terms of the events of every other order of its reflections
that gives its image, those that differ from its own once each (all orders up to four
reflections, beyond that those that swap two next to each other).

At a leg, the terms of the edge's reflections whose paths cannot be, with the source or
the listener side behind the face's plane, stand in for those reflections, which the
early stage follows in front of the plane only for paths of a lower order than the
highest; they take what the face gives the reflection where it meets the plane, the
scattering of its surface's size included where the early stage weakens the
reflection for it (:func:`~acoustic_spaces.engine.model.size_scattering`).

A leg that gets past an edge on the dark side of its boundary, as beside the end of a
strip, takes the edge's term where the edge's face crosses the leg's line on the leg:
where the line comes nearest to the edge, it must lie on the leg's side of the planes
the leg reflects off at its ends. Beyond one of them the edge belongs to the next leg
(a barrier's top past the reflection off the ground that ends a leg, which blocks the
leg after it), save for an edge along the plane's normal, its own image there.

Each term makes a path over its edge, which must be clear from the edge to where it
reflects first before and after it (where the line from the edge to the image on either
side first meets a plane it may reflect off there). In the bands where paths add in
pressure it carries its phase; in the others the path it borders, where that is found,
takes the cross term. A term stops where the point it diffracts at leaves the end of its
edge (the UTD has no terms for the corners of edges), so the level can still step
slightly there; where that end lies in a plane the path reflects off next to the event,
the edge goes on as its image in the plane, and the leg on one side of the reflection
or the other takes the term.
"""

from __future__ import annotations

import itertools

import numpy as np

from acoustic_spaces.engine.diffraction import (
    _OFFSET,
    _ON_BOUNDARY,
    _TOLERANCE,
    MIN_RELATIVE_ENERGY,
    EdgeDiffraction,
    _clear,
    _first_solid,
    _match,
    _reach,
    _replace_energy,
    _take,
    _waves,
    boundary_angles,
    wedge_prefactor,
)
from acoustic_spaces.engine.early import EarlyReflections, _face_at, assemble_paths
from acoustic_spaces.engine.model import size_scattering

#: Kinds of event: a leg of a path passes an edge, or the path reflects near the edge of
#: the face it reflects off. Along a path of k reflections, leg j is event 2j and
#: reflection j (from 1) event 2j - 1.
_LEG, _FACE = 0, 1
#: Pairs of an event and an edge tested at once for whether the path over the edge
#: arrives in time, and of those that do, pairs evaluated at once.
_BLOCK = 1 << 18
_CHUNK = 4096
#: Up to this order every order of a path's reflections that gives its image is looked
#: at; above it, those that swap two reflections next to each other.
_ALL_ORDERS = 4
#: The pressure of a term, relative to its path, below which it is left out (-60 dB).
_NEGLIGIBLE = 1e-3


class EdgeFade:
    """The terms at the edges of ``diffraction``'s model of one source's specular paths
    (see the module notes), for any number of listeners. ``candidates`` are the plane
    sequences the early stage tests at each listener (as
    :func:`~acoustic_spaces.engine.early.find_paths` takes them); those not found there
    may still be found past an edge."""

    def __init__(self, diffraction: EdgeDiffraction, candidates: list[np.ndarray]) -> None:
        self.d = diffraction
        self.model = diffraction.model
        self.edges = diffraction.edges
        self.candidates = {0: np.zeros((1, 0), dtype=np.int64)}  # the direct sound
        for seqs in candidates:
            if len(seqs) and seqs.shape[1]:
                self.candidates[seqs.shape[1]] = np.asarray(seqs, dtype=np.int64)
        self.tolerance = _TOLERANCE * diffraction.size
        self._bisector = self.edges.bisector()
        # The edges of each plane (of either face), by plane.
        pairs = np.unique(
            np.concatenate(
                [
                    np.column_stack([self.edges.plane[:, side], np.arange(len(self.edges))])
                    for side in (0, 1)
                ]
            ),
            axis=0,
        )
        self._plane_edges = pairs[:, 1]
        self._plane_start = np.searchsorted(pairs[:, 0], np.arange(self.model.plane_count + 1))
        self._standing: dict[tuple[int, int], bool] = {}  # see _stands

    def covered(self, before: int, after: int) -> bool:
        """Whether the edge diffraction bends the waves from a source side mirrored in
        ``before`` planes to a listener side mirrored in ``after`` planes."""
        if self.d.mode == "direct":
            return before == 0 and after == 0
        if self.d.both_sides:
            return before <= 1 and after <= 1
        return before + after <= 1

    # -- the paths and their events -----------------------------------------------------

    def _found(self, paths: EarlyReflections, k: int) -> dict | None:
        """The specular paths of order ``k`` in ``paths`` that take terms."""
        rows = np.flatnonzero((paths.order == k) & (paths.edge < 0) & (paths.crossings == 0))
        rows = np.array(
            [i for i in rows if len(paths.points[i]) == k + 2 and len(paths.faces[i]) == k],
            dtype=np.int64,
        )
        if len(rows) == 0:
            return None
        points = np.stack([paths.points[i] for i in rows])
        faces = np.stack([np.asarray(paths.faces[i], dtype=np.int64) for i in rows])
        faces = faces.reshape(len(rows), k)
        # Not those where a leg has no length, but in the corner of two perpendicular
        # planes, where the two orders of its reflections meet.
        keep = self._lengths_ok(points, self.model.face_plane[faces])
        if not keep.any():
            return None
        rows, points, faces = rows[keep], points[keep], faces[keep]
        return {
            "points": points,
            "planes": self.model.face_plane[faces],
            "faces": faces,
            "row": rows,
            "distance": paths.distance_m[rows],
            "energy": paths.energy[rows],
            "spectrum": None if paths.spectrum is None else paths.spectrum[rows],
            "blocked": np.full((len(rows), k + 1), -1, dtype=np.int64),
            "stop_at": np.zeros((len(rows), k + 1, 3)),
            "fail": np.full(len(rows), -1, dtype=np.int64),
        }

    def _lengths_ok(self, points: np.ndarray, planes: np.ndarray) -> np.ndarray:
        """Whether every leg of the paths through ``points`` (n, k + 2, 3), reflecting off
        ``planes``, has a length, but those between reflections off perpendicular planes,
        which may meet in their corner."""
        length = np.linalg.norm(np.diff(points, axis=1), axis=2)  # (n, k + 1)
        ok = length > self.tolerance
        for j in range(1, planes.shape[1]):  # leg j, between reflections j and j + 1
            ok[:, j] |= _perpendicular(self.model, planes[:, j - 1], planes[:, j])
        return np.all(ok, axis=1)

    def _missed(self, seqs: np.ndarray, listener, paths, max_delay_s) -> dict | None:
        """The candidate plane sequences ``seqs`` (n, k) not found at ``listener`` that
        one event stops: every reflection on a face of its plane and every leg clear but
        one leg (blocked) or one reflection (off its faces, the legs next to it clear or
        blocked); with their reflection points as the image method places them, the
        plane of the first solid face on each blocked leg, and the event that stops
        them."""
        model, source = self.model, self.d.source
        n, k = seqs.shape
        c = self.d.speed_of_sound
        chain = [np.broadcast_to(source, (n, 3)).copy()]
        for j in range(k):
            chain.append(model.mirror(chain[-1], seqs[:, j]))
        distance = np.linalg.norm(chain[-1] - listener, axis=1)
        keep = distance / c <= max_delay_s
        # A sequence found already is no miss.
        found = np.flatnonzero((paths.edge < 0) & (paths.order == k))
        if len(found):
            images = listener + paths.direction[found] * paths.distance_m[found, None]
            hit = _match(
                chain[-1],
                np.full(n, k),
                images,
                paths.order[found],
                found,
                1e-6 * self.d.size,
            )
            keep &= hit < 0
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            return None
        seqs, chain, distance = seqs[idx], [p[idx] for p in chain], distance[idx]
        n = len(idx)

        # Reflection points, back from the listener towards each image in turn.
        points = np.empty((n, k + 2, 3))
        points[:, 0], points[:, k + 1] = source, listener
        possible = np.ones(n, dtype=bool)
        current = np.broadcast_to(listener, (n, 3)).copy()
        for j in range(k, 0, -1):
            plane = seqs[:, j - 1]
            normal = model.plane_normal[plane]
            near = np.einsum("ij,ij->i", normal, current) - model.plane_offset[plane]
            far = np.einsum("ij,ij->i", normal, chain[j]) - model.plane_offset[plane]
            two_sided = model.plane_two_sided[plane]
            front = np.where(two_sided, np.abs(near) > self.tolerance, near > self.tolerance)
            # In the corner where it meets the plane of the next reflection, perpendicular
            # to it, the path reflects off both at one point.
            corner = np.zeros(n, dtype=bool)
            if j < k:
                behind = np.where(two_sided, np.abs(far) > self.tolerance, far < -self.tolerance)
                corner = (
                    (np.abs(near) <= self.tolerance)
                    & behind
                    & _perpendicular(model, plane, seqs[:, j])
                )
            possible &= (front & (near * far < 0.0)) | corner
            t = near / np.where(possible, near - far, 1.0)
            current = current + t[:, None] * (chain[j] - current)
            points[:, j] = current
        legs = np.diff(points, axis=1)  # (n, k + 1, 3)
        length = np.linalg.norm(legs, axis=2)
        # Not at a corner, but where two perpendicular planes meet.
        possible &= self._lengths_ok(points, seqs)
        direction = legs / np.maximum(length, 1e-300)[..., None]
        for j in range(1, k + 1):  # a leg of no length goes the way of the one before
            none = length[:, j] <= self.tolerance
            direction[none, j] = direction[none, j - 1]
        # Plane sequences that commute (perpendicular planes) give the same image: the
        # path is the one whose reflections come in its order along the way.
        rows = np.flatnonzero(possible)
        _, first = np.unique(np.round(chain[-1][rows] / self.tolerance), axis=0, return_index=True)
        possible[np.setdiff1d(rows, rows[first])] = False

        # The face each reflection point lies on, or -1.
        faces = np.full((n, k), -1, dtype=np.int64)
        step = _OFFSET * self.d.size
        for j in range(1, k + 1):
            rows = np.flatnonzero(possible)
            arriving = direction[rows, j - 1]
            start = points[rows, j] - step * arriving
            _, hit = model.intersect(
                start, arriving, np.full(len(rows), 2.0 * step), np.full(len(rows), -1)
            )
            plane = seqs[rows, j - 1]
            on = (hit >= 0) & (model.face_plane[np.maximum(hit, 0)] == plane)
            for i in np.flatnonzero((hit >= 0) & ~on):  # another face right there
                hit[i] = _face_at(model, plane[i], points[rows[i], j], self.tolerance)
                on[i] = hit[i] >= 0
            faces[rows, j - 1] = np.where(on, hit, -1)

        # Each leg: clear, or the plane of the first solid face on it and where it meets
        # that face.
        blocked = np.full((n, k + 1), -1, dtype=np.int64)
        stop_at = np.zeros((n, k + 1, 3))
        crossed = np.zeros(n, dtype=np.int64)
        for j in range(k + 1):
            rows = np.flatnonzero(possible)
            ignore = faces[rows, j - 1] if j else np.full(len(rows), -1)
            share, crossings = model.passes(
                points[rows, j], direction[rows, j], length[rows, j] * (1.0 - 1e-9), ignore
            )
            passing = np.any(share > 0.0, axis=1)
            crossed[rows] += np.where(passing, crossings, 0)
            stop = np.flatnonzero(~passing)
            if len(stop):
                at = rows[stop]
                where, face = _first_solid(
                    model,
                    points[at, j],
                    direction[at, j],
                    length[at, j] * (1.0 - 1e-9),
                    ignore[stop],
                )
                blocked[at, j] = np.where(face >= 0, model.face_plane[np.maximum(face, 0)], -2)
                stop_at[at, j] = points[at, j] + where[:, None] * direction[at, j]
        off = faces < 0
        stopped = blocked != -1
        one_leg = (off.sum(axis=1) == 0) & (stopped.sum(axis=1) == 1)
        fail = np.where(one_leg, 2 * np.argmax(stopped, axis=1), -1)
        if k:
            # One reflection off its faces, and no leg blocked but those next to it.
            one_face = off.sum(axis=1) == 1
            at = np.argmax(off, axis=1) + 1  # the reflection off its faces, from 1
            legs_next = np.zeros((n, k + 1), dtype=bool)
            legs_next[np.arange(n), at - 1] = True
            legs_next[np.arange(n), at] = True
            one_face &= ~np.any(stopped & ~legs_next, axis=1)
            fail = np.where(one_face, 2 * at - 1, fail)
        keep = possible & (crossed == 0) & (fail >= 0)
        keep &= ~np.any(blocked == -2, axis=1)
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            return None
        return {
            "points": points[idx],
            "planes": seqs[idx],
            "faces": faces[idx],
            "row": np.full(len(idx), -1, dtype=np.int64),
            "distance": distance[idx],
            "blocked": blocked[idx],
            "stop_at": stop_at[idx],
            "fail": fail[idx],
        }

    def _events(self, go: dict, k: int) -> list[dict]:
        """The events of the paths ``go`` of order ``k``: each leg (passing edges), and
        each reflection (near the edges of its face), with the rows of ``go`` they belong
        to, the source and listener sides of their waves, and whether the edge
        diffraction bends them; and those of the other orders of the path's reflections
        that give the same image (reflections off perpendicular planes commute), which
        the path over an edge may take, once each."""
        points, planes = go["points"], go["planes"]
        n = len(points)
        commute = np.ones((n, k, k), dtype=bool)
        for a in range(k):
            for b in range(k):
                if a != b:
                    commute[:, a, b] = _perpendicular(self.model, planes[:, a], planes[:, b])
        out = self._ordered(go, k, np.arange(n), tuple(range(k)), commute)
        if k > 1:
            orders = itertools.permutations(range(k)) if k <= _ALL_ORDERS else _swaps(k)
            for order in orders:
                if order == tuple(range(k)):
                    continue
                # Only pairs of reflections that commute may change places.
                rows = np.ones(n, dtype=bool)
                where = np.argsort(order)
                for a in range(k):
                    for b in range(a + 1, k):
                        if where[a] > where[b]:
                            rows &= commute[:, a, b]
                rows = np.flatnonzero(rows)
                if len(rows):
                    out += self._ordered(go, k, rows, order, commute[rows])
        return _once(out, self.tolerance)

    def _ordered(self, go: dict, k: int, rows: np.ndarray, order: tuple, commute) -> list:
        """The events of the paths ``rows`` of ``go`` with their reflections in ``order``
        (the reflections of the path, in the order taken); each reflection's event keeps
        its place in the path, and each leg its place along the way."""
        model = self.model
        order = list(order)
        points, planes = go["points"][rows], go["planes"][rows][:, order]
        commute = commute[:, order][:, :, order]
        own = order == list(range(k))
        n = len(rows)
        none = np.full(n, -1, dtype=np.int64)
        sources = [points[:, 0]]
        for j in range(k):
            sources.append(model.mirror(sources[-1], planes[:, j]))
        listeners = [points[:, k + 1]]
        for j in range(k, 0, -1):
            listeners.insert(0, model.mirror(listeners[0], planes[:, j - 1]))

        def run(before: int, after: int) -> np.ndarray:
            """(n, 2, k) the planes the path over an edge may reflect off first on either
            side, between the reflections ``before`` (up to) and ``after`` (from): each
            that commutes with those between it and the edge."""
            ends = np.full((n, 2, max(k, 1)), -1, dtype=np.int64)
            for i in range(before, -1, -1):
                ok = np.all(commute[:, i, i + 1 : before + 1], axis=1)
                ends[:, 0, i] = np.where(ok, planes[:, i], -1)
            for i in range(after, k):
                ok = np.all(commute[:, i, after:i], axis=1)
                ends[:, 1, i] = np.where(ok, planes[:, i], -1)
            return ends

        # For the share the size of a face leaves the reflections of a path with one more
        # (see _sizes): the planes before and after each leg in this order, the path's
        # last reflection and listener, and the reflection that a reflection put before
        # it all moves to the order from which the early stage weakens it.
        width = max(k, 1)
        sides = []
        for j in range(k + 1):
            before = np.full((n, width), -1, dtype=np.int64)
            after = np.full((n, width), -1, dtype=np.int64)
            before[:, :j] = planes[:, :j]
            after[:, : k - j] = planes[:, j:]
            sides.append(np.stack([before, after], axis=1))
        i = model.size_from_order - 1
        moved = go["faces"][rows, i - 1] if 1 <= i <= k else none
        size = {
            "last": points[:, k],
            "hearer": points[:, k + 1],
            "moved": moved,
            "moved_from": points[:, i - 1] if 1 <= i <= k else points[:, 0],
            "moved_at": points[:, i] if 1 <= i <= k else points[:, 0],
        }

        out = []
        for j in range(k + 1):  # leg j, from point j to point j + 1
            blocked = go["blocked"][rows, j] if own else none
            out.append(
                {
                    "row": rows,
                    "kind": np.full(n, _LEG),
                    "at": np.full(n, 2 * j),
                    "covered": np.full(n, self.covered(j, k - j)),
                    "source": sources[j],
                    "listener": listeners[j],
                    "start": points[:, j] if own else points[:, 0],
                    "end": points[:, j + 1] if own else points[:, k + 1],
                    # The planes it reflects off at either end: their edges' terms are
                    # those of the reflections.
                    "next_to": np.column_stack(
                        [planes[:, j - 1] if j else none, planes[:, j] if j < k else none]
                    ),
                    "ends": run(j - 1, j),
                    "plane": none,
                    # What blocks the leg and where, as for the legs next to a reflection.
                    "blocked": np.column_stack([blocked, none]),
                    "stop_at": np.stack(
                        [go["stop_at"][rows, j] if own else np.zeros((n, 3)), np.zeros((n, 3))],
                        axis=1,
                    ),
                    "sides": sides[j],
                    **size,
                }
            )
        for j in range(1, k + 1):  # reflection j, at point j
            out.append(
                {
                    "row": rows,
                    "kind": np.full(n, _FACE),
                    "at": np.full(n, 2 * order[j - 1] + 1),
                    "covered": np.full(n, self.covered(j - 1, k - j)),
                    "source": sources[j - 1],
                    "listener": listeners[j],
                    "start": points[:, j - 1] if own else points[:, 0],
                    "end": points[:, j + 1] if own else points[:, k + 1],
                    "next_to": np.column_stack([none, none]),
                    "ends": run(j - 2, j),
                    "plane": planes[:, j - 1],
                    "blocked": go["blocked"][rows][:, [j - 1, j]]
                    if own
                    else np.full((n, 2), -1, dtype=np.int64),
                    "stop_at": go["stop_at"][rows][:, [j - 1, j]] if own else np.zeros((n, 2, 3)),
                    "sides": np.full((n, 2, width), -1, dtype=np.int64),
                    **size,
                }
            )
        return out

    def _table(self, groups: list[tuple[int, dict]]) -> tuple[dict, int]:
        """The events of the paths of all ``groups`` (their order and paths), one row
        for each path and event, with the path (counted over the groups), its group and
        its row there; and the number of paths."""
        columns = []
        first = 0
        for g, (k, go) in enumerate(groups):
            for event in self._events(go, k):
                row = event["row"]
                column = dict(event)
                column.update(
                    order=np.full(len(row), k),
                    group=np.full(len(row), g),
                    path=first + row,
                    fail=go["fail"][row],
                    distance=go["distance"][row],
                )
                columns.append(column)
            first += len(go["fail"])
        for key in ("ends", "sides"):
            width = max(c[key].shape[2] for c in columns)
            for c in columns:
                pad = width - c[key].shape[2]
                if pad:
                    c[key] = np.pad(c[key], ((0, 0), (0, 0), (0, pad)), constant_values=-1)
        return {key: np.concatenate([c[key] for c in columns]) for key in columns[0]}, first

    def _of_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pairs (index, edge) of each of ``planes`` with the edges with a face in it."""
        planes = np.asarray(planes, dtype=np.int64)
        valid = planes >= 0
        lo = np.where(valid, self._plane_start[np.maximum(planes, 0)], 0)
        hi = np.where(valid, self._plane_start[np.maximum(planes, 0) + 1], 0)
        count = hi - lo
        index = np.repeat(np.arange(len(planes)), count)
        offset = np.arange(len(index)) - np.repeat(np.cumsum(count) - count, count)
        return index, self._plane_edges[np.repeat(lo, count) + offset]

    def _pairs(self, table: dict, slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(event, edge) pairs to test for the events ``slots`` of ``table``: for a leg,
        every edge but those of the planes the path over the edge may reflect off at its
        ends (their terms are those of the reflections); for a reflection, the edges of
        its plane."""
        edges = self.edges
        legs = slots[table["kind"][slots] == _LEG]
        slot = np.repeat(legs, len(edges))
        edge = np.tile(np.arange(len(edges)), len(legs))
        ends = table["ends"].reshape(len(table["ends"]), -1)[slot]
        near = np.zeros(len(slot), dtype=bool)
        for side in (0, 1):
            for end in ends.T:
                near |= edges.plane[edge, side] == end
        faces = slots[table["kind"][slots] == _FACE]
        index, face_edge = self._of_planes(table["plane"][faces])
        return (
            np.concatenate([slot[~near], faces[index]]),
            np.concatenate([edge[~near], face_edge]),
        )

    def _explains(self, table: dict, slot: np.ndarray, edge: np.ndarray) -> np.ndarray:
        """Whether ``edge`` belongs to what stops the paths at the events ``slot`` of
        ``table``: a leg is blocked by a face of it; a reflection lies off its face (one
        of its plane's edges), and a face of it blocks each leg next to it that is
        blocked."""
        blocked, stop_at = table["blocked"][slot], table["stop_at"][slot]
        on = [self._on_face(edge, blocked[:, side], stop_at[:, side]) for side in (0, 1)]
        face = ((blocked[:, 0] < 0) | on[0]) & ((blocked[:, 1] < 0) | on[1])
        return np.where(table["kind"][slot] == _LEG, on[0], face)

    def _on_face(self, edge: np.ndarray, plane: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Whether ``point``, in ``plane``, lies on a face of ``edge``: in the plane of
        that face, on the side of the edge where the face goes, and beside the edge
        (another face in the same plane, such as a wall in line with it, does not
        count)."""
        edges = self.edges
        offset = point - edges.start[edge]
        along = np.einsum("ij,ij->i", offset, edges.direction[edge])
        out = (along >= -self.tolerance) & (along <= edges.length[edge] + self.tolerance)
        angle = edges.wedge[edge] * np.pi
        face_n = np.cos(angle)[:, None] * edges.along[edge]
        face_n += np.sin(angle)[:, None] * edges.normal[edge]
        on_0 = (edges.plane[edge, 0] == plane) & (
            np.einsum("ij,ij->i", offset, edges.along[edge]) > -self.tolerance
        )
        on_n = (edges.plane[edge, 1] == plane) & (
            np.einsum("ij,ij->i", offset, face_n) > -self.tolerance
        )
        return out & (plane >= 0) & (on_0 | on_n)

    def _stands(self, edge: np.ndarray, plane: np.ndarray) -> np.ndarray:
        """Whether ``edge`` stands on ``plane``: an end of it lies on a face of the plane
        (its sides included), not just in the plane (a wall as high as a roof elsewhere),
        and the edge leaves it."""
        edges, model = self.edges, self.model
        safe = np.maximum(plane, 0)
        normal, offset = model.plane_normal[safe], model.plane_offset[safe]
        ends = [
            np.abs(np.einsum("ij,ij->i", normal, point[edge]) - offset) <= self.tolerance
            for point in (edges.start, edges.end)
        ]
        leaves = np.abs(np.einsum("ij,ij->i", normal, edges.direction[edge])) > 1e-6
        out = (plane >= 0) & (ends[0] | ends[1]) & leaves
        for i in np.flatnonzero(out):
            key = (int(edge[i]), int(plane[i]))
            if key not in self._standing:
                end = edges.start[edge[i]] if ends[0][i] else edges.end[edge[i]]
                self._standing[key] = _face_at(model, key[1], end, self.tolerance) >= 0
            out[i] = self._standing[key]
        return out

    # -- the terms --------------------------------------------------------------------

    def terms(self, listener: np.ndarray, paths: EarlyReflections, max_delay_s: float) -> dict:
        """The terms at the edges of ``paths`` (the specular paths at ``listener``) and
        of the paths just missed there, up to ``max_delay_s``: for :meth:`join`."""
        listener = np.asarray(listener, dtype=float)
        top = max([int(paths.order.max()) if len(paths) else 0, *self.candidates])
        self._top = top
        groups = []
        for k in range(top + 1):
            groups.append((k, self._found(paths, k)))
            if k in self.candidates:
                groups.append((k, self._missed(self.candidates[k], listener, paths, max_delay_s)))
        groups = [(k, go) for k, go in groups if go is not None]
        parts = [
            self._combine(listener, go, k, t)
            for (k, go), t in zip(groups, self._group_terms(groups, max_delay_s), strict=True)
            if t is not None
        ]
        squared = max(
            float(np.sum((listener - self.d.source) ** 2)), (_TOLERANCE * self.d.size) ** 2
        )
        floor = MIN_RELATIVE_ENERGY / (4.0 * np.pi * squared)
        out = {"listener": listener, "floor": floor, "count": 0, "specular": paths}
        if parts:
            for key in ("target", "edge", "order", "distance", "energy", "cross"):
                out[key] = np.concatenate([p[key] for p in parts])
            if self.d.coherent:
                out["spectrum"] = np.concatenate([p["spectrum"] for p in parts])
            out["points"] = [x for p in parts for x in p["points"]]
            out["faces"] = [x for p in parts for x in p["faces"]]
            out["count"] = len(out["target"])
        return out

    def join(self, paths: EarlyReflections, terms: dict) -> EarlyReflections:
        """``paths`` (the specular paths of :meth:`terms` with the edge diffraction's) with
        the ``terms`` of :meth:`terms`: the cross terms in the energy of the paths they
        border, and the paths over the edges that carry at least the edge diffraction's
        least energy (:data:`MIN_RELATIVE_ENERGY` of the direct sound in free field) in
        some band."""
        if not terms["count"]:
            return paths
        energy, target = terms["energy"], terms["target"]
        lit = np.flatnonzero(target >= 0)
        if len(lit):
            bordered = _rows(paths, terms["specular"], target[lit])
            total, energy = _take(paths.energy, bordered, terms["cross"][lit], energy, lit)
            paths = _replace_energy(paths, total)
        idx = np.flatnonzero(energy.max(axis=1) >= terms["floor"])
        if len(idx) == 0:
            return paths
        listener = terms["listener"]
        points = [terms["points"][i] for i in idx]
        arrival = np.array([p[-2] for p in points]) - listener
        arrival /= np.maximum(np.linalg.norm(arrival, axis=1), 1e-300)[:, None]
        distance = terms["distance"][idx]
        added = EarlyReflections(
            order=terms["order"][idx],
            delay_s=distance / self.d.speed_of_sound,
            distance_m=distance,
            direction=arrival,
            energy=energy[idx],
            points=points,
            faces=[terms["faces"][i] for i in idx],
            edge=terms["edge"][idx],
            spectrum=terms["spectrum"][idx] if self.d.coherent else None,
        )
        return EarlyReflections.concatenate([paths, added], paths.energy.shape[1])

    def _group_terms(self, groups: list[tuple[int, dict]], max_delay_s: float) -> list:
        """The terms of the paths of each of ``groups`` (their order and paths, all found
        or all missed), all evaluated together, as :meth:`terms` gathers them; None for a
        group without any."""
        if not groups:
            return []
        table, count = self._table(groups)
        # The event that stops each missed path first: the path is kept only where an
        # edge of what stops it explains that. A reflection off its face may also be
        # stopped by a leg next to it, blocked by a face of an edge that stands on the
        # reflection's plane (a corner on the floor): the edge and its image in the
        # plane are one edge, and the path over it reflects off the plane elsewhere.
        fail = table["fail"]
        stops = table["at"] == fail
        beside = (fail % 2 == 1) & (table["kind"] == _LEG) & (np.abs(table["at"] - fail) == 1)
        parts = self._terms_at(table, np.flatnonzero(stops | beside), False, max_delay_s)
        stop = np.full(count, -1, dtype=np.int64)
        for part in parts:
            path, at = part["path"], part["at"]
            own = part["explains"] & (at == fail[part["slot"]])
            stop[path[own]] = at[own]
        for part in parts:
            path, at, slot = part["path"], part["at"], part["slot"]
            plane = table["next_to"][slot, np.where(at < fail[slot], 1, 0)]
            leg = part["explains"] & (at != fail[slot]) & self._stands(part["edge"], plane)
            leg &= stop[path] < 0
            stop[path[leg]] = at[leg]
        parts = [_subset(part, part["at"] == stop[part["path"]]) for part in parts]
        explained = stop >= 0
        table["fail"] = np.where(explained[table["path"]], stop[table["path"]], fail)
        first = 0
        for _, go in groups:
            n = len(go["fail"])
            go["fail"] = np.where(explained[first : first + n], stop[first : first + n], go["fail"])
            first += n
        stops = table["at"] == table["fail"]
        going = ~stops & ((table["fail"] < 0) | explained[table["path"]])
        parts += self._terms_at(table, np.flatnonzero(going), True, max_delay_s)
        parts = [part for part in parts if len(part["go"])]
        if not parts:
            return [None] * len(groups)
        t = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        t = _subset(t, np.argsort(t["group"], kind="stable"))
        bounds = np.searchsorted(t["group"], np.arange(len(groups) + 1))
        return [
            _subset(t, slice(lo, hi)) if hi > lo else None
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        ]

    def _terms_at(self, table: dict, slots: np.ndarray, passes: bool, max_delay_s) -> list:
        """The terms of the events ``slots`` of ``table`` at every edge they may meet, in
        blocks of about :data:`_BLOCK` pairs; ``passes`` says whether the paths get past
        their events."""
        if len(slots) == 0:
            return []
        plane = np.maximum(table["plane"][slots], 0)
        count = np.where(
            table["kind"][slots] == _LEG,
            len(self.edges),
            self._plane_start[plane + 1] - self._plane_start[plane],
        )
        block = (np.cumsum(count) - count) // _BLOCK
        out = []
        for b in np.unique(block):
            slot, edge = self._pairs(table, slots[block == b])
            out += self._evaluate(table, slot, edge, passes, max_delay_s)
        return out

    def _evaluate(self, table, slot, edge, passes, max_delay_s) -> list[dict]:
        """The terms of the pairs of events ``slot`` of ``table`` and ``edge``, in chunks
        of :data:`_CHUNK` pairs over edges the path reaches in time."""
        d, edges = self.d, self.edges
        src, rcv = table["source"][slot], table["listener"][slot]
        t_s, rho_s = edges.distances(edge, src)
        t_r, rho_r = edges.distances(edge, rcv)
        reach, t_k, r1, r2 = _reach(
            np.zeros(len(edge), dtype=np.int64),
            edges.length[edge],
            t_s,
            rho_s,
            t_r,
            rho_r,
            d.speed_of_sound,
            max_delay_s,
        )
        # Where the path over the edge reaches the end of it that lies in a plane the
        # path may reflect off next to the event, that is where the edge meets its image
        # in the plane: the path over either counts, and _pressure gives it to one leg.
        end = ~reach & (np.abs(t_k - edges.length[edge]) <= self.tolerance)
        end &= (r1 + r2) / d.speed_of_sound <= max_delay_s
        rows = np.flatnonzero(end)
        if len(rows):
            ends = table["ends"][slot[rows]].reshape(len(rows), -1)
            point = (
                edges.start[edge[rows]]
                + edges.length[edge[rows], None] * edges.direction[edge[rows]]
            )
            normal = self.model.plane_normal[np.maximum(ends, 0)]  # (n, E, 3)
            offset = self.model.plane_offset[np.maximum(ends, 0)]
            gap = np.abs(np.einsum("nej,nj->ne", normal, point) - offset)
            reach[rows] = np.any((ends >= 0) & (gap <= self.tolerance), axis=1)
        keep = np.flatnonzero(reach & (rho_s > self.tolerance) & (rho_r > self.tolerance))
        out = []
        for begin in range(0, len(keep), _CHUNK):
            sel = keep[begin : begin + _CHUNK]
            part = self._pressure(
                table, slot[sel], edge[sel], t_k[sel], r1[sel], r2[sel], rho_s[sel], passes
            )
            if part is not None:
                out.append(part)
        return out

    def _pressure(self, table, slot, edge, t_k, r1, r2, rho_s, passes) -> dict | None:
        """The terms of the pairs of events ``slot`` of ``table`` and ``edge``, the point
        ``t_k`` along the edge where the path over it reaches it, ``r1`` and ``r2`` from
        the event's source and listener sides: their pressure (P, B, M) relative to
        each path, with the phase of the longer way over the edge; None where none is
        left."""
        d, edges, model = self.d, self.edges, self.model
        _, _, phi_s, in_s = edges.frame(edge, table["source"][slot])
        _, _, phi_r, in_r = edges.frame(edge, table["listener"][slot])
        ok = in_s & in_r
        point = edges.start[edge] + t_k[:, None] * edges.direction[edge]
        n = edges.wedge[edge]
        start, end = table["start"][slot], table["end"][slot]
        leg = table["kind"][slot] == _LEG
        # At a leg, the point on the edge on the leg's side of the planes the path over
        # it may reflect off at either end (the side away from the image there), as the
        # path over the edge must be: of the twins either side of a plane the edge
        # crosses (a wall standing on the ground), the one that is there; on the plane
        # itself, the leg after it.
        ends = table["ends"][slot]
        images = (table["source"][slot], table["listener"][slot])
        ok &= _on_leg(model, point, images, ends, leg)
        # At a reflection, the face of the edge in the plane, with the source side in
        # front of it.
        plane = table["plane"][slot]
        front_0 = ~leg & (edges.plane[edge, 0] == plane) & (phi_s < np.pi)
        front_n = ~leg & (edges.plane[edge, 1] == plane) & (phi_s > (n - 1.0) * np.pi)
        ok &= leg | front_0 | front_n
        column = np.where(leg, 0, np.where(front_0, 1, 2))
        # Each edge counts on its own side of its boundary; exactly on it, on the side
        # the path is on.
        angle = boundary_angles(n, phi_r, phi_s)  # (P, 3)
        lit = (angle > _ON_BOUNDARY) | ((np.abs(angle) <= _ON_BOUNDARY) & passes)
        mine = lit[np.arange(len(edge)), column]
        # A leg that gets past the edge on its dark side has the edge's face across its
        # line there: on the leg itself, as where a strip lies beside it, and not past a
        # plane it reflects off, in the image of the room beyond (a barrier on the ground
        # across the line from the source to the listener's image, where the leg ends on
        # the ground before it): that edge is the next leg's, where it stops the path.
        if passes:
            dark = np.flatnonzero(ok & leg & ~mine)
            if len(dark):
                closest = _closest(images[0][dark], images[1][dark], edges, edge[dark])
                at = (images[0][dark], images[1][dark])
                # An edge along a plane's normal is its own image in it (a corner standing
                # on the ground): there the side its point lies on says it all.
                normal = model.plane_normal[np.maximum(ends[dark], 0)]  # (n, 2, m, 3)
                along = np.abs(np.einsum("nsmj,nj->nsm", normal, edges.direction[edge[dark]]))
                tilted = np.where(along > 1.0 - 1e-9, -1, ends[dark])
                ok[dark] &= _on_leg(model, closest, at, tilted, leg[dark])
        explains = np.zeros(len(edge), dtype=bool)
        if not passes:
            explains = ~mine & self._explains(table, slot, edge)
        # The path over the edge must be clear of everything, from just off the edge in
        # the middle of its open side and just inside its ends, to where it reflects
        # before and after the edge (not the path's own points, which may lie off their
        # faces where it is missed).
        sel = np.flatnonzero(ok)
        if len(sel):
            offset = _OFFSET * d.size
            e = edge[sel]
            along = np.clip(t_k[sel], offset, edges.length[e] - offset) - t_k[sel]
            off_edge = point[sel] + along[:, None] * edges.direction[e]
            off_edge += offset * self._bisector[e]
            images, planes = (table["source"], table["listener"]), ends[sel]
            reflect = [
                _towards(model, point[sel], images[side][slot[sel]], planes[:, side], own)
                for side, own in enumerate((start[sel], end[sel]))
            ]
            clear = _clear(model, np.vstack([off_edge, off_edge]), np.vstack(reflect))
            sel = sel[clear[: len(sel)] & clear[len(sel) :]]
        if len(sel) == 0:
            return None
        edge, slot, column, n, point = edge[sel], slot[sel], column[sel], n[sel], point[sel]
        r1, r2, rho_s, lit, t_k = r1[sel], r2[sel], rho_s[sel], lit[sel], t_k[sel]
        phi_s, phi_r, explains, leg = phi_s[sel], phi_r[sel], explains[sel], leg[sel]
        index = np.arange(len(edge))

        # The term of the event's own boundary, relative to the path; and at a leg, those
        # of the edge's reflections whose paths cannot be, with the source or the
        # listener side behind the face's plane or on it, on their dark side (where a
        # path is found or just missed, its own event has the reflection's term). Only
        # the leg carries them: a reflection near the edge with the same two sides
        # (another path of the same order) would carry them once more. Only a path of a
        # lower order than the highest takes them, as in front of the face's plane the
        # early stage follows its reflection off the face only then. Where the edge
        # diffraction bends the event it has them all, in front of the plane and behind
        # it, and those it has are left as they are.
        wavenumber = d.wavenumber[None]  # (1, B, M)
        sin_beta = np.clip(rho_s / r1, 1e-12, 1.0)
        length_param = r1 * r2 / (r1 + r2) * sin_beta**2
        rho_r = r2 * sin_beta
        tol = self.tolerance
        never = np.zeros(len(edge), dtype=bool)
        listener_behind = np.column_stack(
            [never, rho_r * np.sin(phi_r) <= tol, rho_r * np.sin(n * np.pi - phi_r) <= tol]
        )
        behind = listener_behind | np.column_stack(
            [never, rho_s * np.sin(phi_s) <= tol, rho_s * np.sin(n * np.pi - phi_s) <= tol]
        )
        behind[index, column] = False
        behind &= leg[:, None]
        followed = behind & (table["order"][slot] < self._top)[:, None]
        # A reflection's own term without its face's factor, as the path has that
        # already; the others with theirs, and with the share the size of the face leaves
        # where the early stage weakens the reflection for it (the third and later of a
        # path): the reflection it takes over from has that share where it meets the
        # plane.
        factors = np.ones((len(edge), 2, *d.wavenumber.shape), dtype=complex)
        faced = np.flatnonzero(followed.any(axis=1))
        if len(faced):
            factors[faced] = d._face_factors(edge[faced], sin_beta[faced], phi_s[faced])
        if model.surface_size is not None:
            for side in (0, 1):
                rows = np.flatnonzero(followed[:, side + 1])
                if len(rows):
                    share = self._sizes(
                        table,
                        slot[rows],
                        edges.face[edge[rows], side],
                        listener_behind[rows, side + 1],
                    )
                    factors[rows, side] *= share[:, :, None]
        waves = _waves(n, phi_r, phi_s, wavenumber * length_param[:, None, None], lit, factors)
        own = waves[index, column]
        others = np.sum(waves * followed[:, :, None, None], axis=1)
        distance = table["distance"][slot]
        detour = r1 + r2 - distance
        common = wedge_prefactor(n[:, None, None], wavenumber, sin_beta[:, None, None])
        common = common * (distance / np.sqrt(r1 * r2 * (r1 + r2)))[:, None, None]
        common = common * np.exp(-1j * wavenumber * detour[:, None, None])
        own, others = common * own, common * others

        # Terms too weak to matter are left out.
        sel = np.flatnonzero(np.abs(own + others).max(axis=(1, 2)) >= _NEGLIGIBLE)
        if len(sel) == 0:
            return None
        index = np.arange(len(sel))
        edge, slot, column, point, lit = edge[sel], slot[sel], column[sel], point[sel], lit[sel]
        r1, r2, detour, explains = r1[sel], r2[sel], detour[sel], explains[sel]
        return {
            "slot": slot,
            "path": table["path"][slot],
            "group": table["group"][slot],
            "go": table["row"][slot],
            "at": table["at"][slot],
            "covered": table["covered"][slot],
            "edge": edge.astype(np.int64),
            "lit": lit[index, column],
            "explains": explains,
            "column": column,
            "pressure": own[sel],
            "others": others[sel],
            "detour": detour,
            "length": r1 + r2,
            "point": point,
        }

    def _sizes(self, table, slot, face, listener_side) -> np.ndarray:
        """(n, B) the share of the pressure that the sizes of the faces leave the path
        with one more reflection, off ``face`` (n,), that the wave of ``face`` at the
        leg events ``slot`` of ``table`` takes over from behind the face's plane (on the
        listener's side of the edge where ``listener_side``, else on the source's),
        relative to the path's own: what the early stage weakens that path by (the
        reflections from :attr:`TraceModel.size_from_order` on) where it meets the plane.

        Where the planes of the path between the leg and the listener all stand
        perpendicular to the face's, the path meets the plane last, at the listener, from
        its last reflection; where those between the source and the leg do, first, at
        the source, and its reflection that moves to the first weakened takes that
        weakening. Otherwise it reflects off the face on the leg."""
        model = self.model
        first = model.size_from_order
        n = len(slot)
        k = table["order"][slot]
        j = table["at"][slot] // 2
        plane = model.face_plane[face]
        sides = table["sides"][slot]  # (n, 2, K)
        normal = model.plane_normal[plane]
        upright = []
        for s in (0, 1):
            dot = np.einsum("nkj,nj->nk", model.plane_normal[np.maximum(sides[:, s], 0)], normal)
            upright.append(np.all((sides[:, s] < 0) | (np.abs(dot) < 1e-9), axis=1))
        out = np.ones((n, len(model.band_hz)))
        last = listener_side & upright[1]
        rows = np.flatnonzero(last & (k + 1 >= first))
        if len(rows):
            out[rows] = self._size_share(
                face[rows], table["last"][slot[rows]], table["hearer"][slot[rows]]
            )
        moved = table["moved"][slot]
        rows = np.flatnonzero(~listener_side & upright[0] & (moved >= 0))
        if len(rows):
            out[rows] = self._size_share(
                moved[rows], table["moved_from"][slot[rows]], table["moved_at"][slot[rows]]
            )
        rows = np.flatnonzero(~np.where(listener_side, upright[1], upright[0]) & (j + 1 >= first))
        if len(rows):
            far = np.where(
                listener_side[rows, None],
                table["listener"][slot[rows]],
                table["source"][slot[rows]],
            )
            out[rows] = self._size_share(face[rows], table["start"][slot[rows]], far)
        return out

    def _size_share(self, face: np.ndarray, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """(n, B) the share of the pressure of a reflection off ``face`` (n,) that its
        surface's size leaves (:func:`~acoustic_spaces.engine.model.size_scattering`),
        for sound from ``near`` (n, 3) meeting the face's plane where ``far`` (n, 3) lies
        on it, or across from it."""
        model = self.model
        plane = model.face_plane[face]
        normal, offset = model.plane_normal[plane], model.plane_offset[plane]
        at = far - (np.einsum("ij,ij->i", normal, far) - offset)[:, None] * normal
        incoming = at - near
        distance = np.linalg.norm(incoming, axis=1)
        cosine = np.abs(np.einsum("ij,ij->i", normal, incoming)) / np.maximum(distance, 1e-300)
        width, length = model.surface_size[face].T
        bands = np.zeros((len(face), len(model.band_hz)))
        scattering = size_scattering(
            width, length, distance, cosine, self.d.speed_of_sound, bands, model.band_hz
        )
        return np.sqrt(1.0 - scattering)

    def _combine(self, listener, go: dict, k: int, t: dict) -> dict:
        """The paths over the edges for the terms ``t`` of the paths ``go``: each term
        with its share of its path's product over the events (see the module notes),
        less what the edge diffraction adds for it."""
        d, edges = self.d, self.edges
        wavenumber = d.wavenumber  # (B, M)
        ids, path = np.unique(t["go"], return_inverse=True)
        path = path.reshape(-1)
        fail = go["fail"][ids]
        missed = fail >= 0
        events = 2 * k + 1
        total = np.zeros((len(ids), events, *wavenumber.shape), dtype=complex)
        np.add.at(total, (path, t["at"]), t["pressure"] + t["others"])
        # Each event's terms take the product of the events before it, the one that
        # stops a missed path first.
        share = np.zeros_like(total)
        running = np.ones((len(ids), *wavenumber.shape), dtype=complex)
        rows = np.flatnonzero(missed)
        share[rows, fail[rows]] = 1.0
        running[rows] = total[rows, fail[rows]]
        for at in range(events):
            rows = np.flatnonzero(fail != at)
            share[rows, at] = running[rows]
            running[rows] *= 1.0 + total[rows, at]
        # The edge diffraction adds a term where it bends the event: on the lit side where
        # the path is found; where it is not, on the dark side at the event that stops it
        # (at the others, its paths over the edge are stopped there too); and the terms of
        # reflections that cannot be that the fade takes, wherever it has the path over
        # the edge.
        stops = t["at"] == fail[path]
        added = t["covered"] & np.where(missed[path], stops & ~t["lit"], t["lit"])
        others = (t["covered"] & (~missed[path] | stops)).astype(float)[:, None, None]
        pressure = (t["pressure"] + t["others"]) * share[path, t["at"]]
        pressure -= added.astype(float)[:, None, None] * t["pressure"] + others * t["others"]

        # The paths' own energy, and the faces they reflect off: a path that reflects off
        # its faces takes the face of the edge whose term explains that most.
        faces = go["faces"][ids].copy()
        if "energy" in go:
            energy = go["energy"][ids]
            spectrum = None if go["spectrum"] is None else go["spectrum"][ids]
        else:
            stopping = t["explains"] & (t["at"] == fail[path]) & (t["at"] % 2 == 1)
            strength = np.mean(np.abs(t["pressure"]) ** 2, axis=(1, 2))
            order = np.lexsort((strength, path))
            order = order[stopping[order]]
            if len(order):
                last = np.r_[path[order][1:] != path[order][:-1], True]
                best = order[last]
                face = edges.face[t["edge"][best], np.where(t["column"][best] == 1, 0, 1)]
                faces[path[best], (t["at"][best] - 1) // 2] = face
            # A reflection off its faces next to the leg that stops the path: the face of
            # its plane nearest to it.
            for i, j in zip(*np.nonzero(faces < 0), strict=True):
                faces[i, j] = _nearest_face(
                    self.model, go["planes"][ids[i], j], go["points"][ids[i], j + 1]
                )
            energy, spectrum = self._energy(listener, go, k, ids, faces)
        energy, faces = energy[path], faces[path]
        spectrum = None if spectrum is None else spectrum[path]

        coherent = d.coherent
        own = np.mean(np.abs(pressure) ** 2, axis=2)  # (P, B)
        out_spectrum = None
        if coherent:
            p = pressure[:, :coherent]
            if spectrum is not None:
                p = spectrum * p
            own[:, :coherent] = np.mean(np.abs(p) ** 2, axis=2)
            # Relative to the path over the edge, which arrives detour metres later.
            p = p * np.exp(1j * wavenumber[None, :coherent] * t["detour"][:, None, None])
            level = np.sqrt(np.mean(np.abs(p) ** 2, axis=2, keepdims=True))
            out_spectrum = np.where(level > 0.0, p / np.where(level > 0.0, level, 1.0), 1.0)
        out_energy = energy * own
        # The cross term with the path where it is found, in the bands where the two add
        # in energy.
        cross = 2.0 * d._fade(t["detour"]) * energy * np.mean(np.real(pressure), axis=2)
        cross[:, :coherent] = 0.0
        cross[missed[path]] = 0.0

        points = go["points"][ids][path]
        path_points, path_faces = [], []
        for i, at in enumerate(t["at"]):
            j = (at + 1) // 2
            if at % 2 == 0:  # leg j: over the edge on the way
                path_points.append(
                    np.vstack([points[i, : j + 1], t["point"][i], points[i, j + 1 :]])
                )
                path_faces.append(faces[i].copy())
            else:  # reflection j: at the edge instead
                path_points.append(np.vstack([points[i, :j], t["point"][i], points[i, j + 1 :]]))
                path_faces.append(np.delete(faces[i], j - 1))
        return {
            "target": np.where(missed[path], -1, go["row"][ids][path]),
            "edge": t["edge"],
            "order": np.where(t["at"] % 2 == 0, k, k - 1).astype(np.int64),
            "distance": t["length"],
            "energy": out_energy,
            "cross": cross,
            "spectrum": out_spectrum,
            "points": path_points,
            "faces": path_faces,
        }

    def _energy(self, listener, go, k, ids, faces):
        """The energy (n, bands) and, in the coherent bands, the spectrum of the missed
        paths ``ids`` of ``go`` as if found, reflecting off ``faces``."""
        points = go["points"][ids]
        n = len(ids)
        bands = self.model.absorption.shape[1]
        found = assemble_paths(
            self.model,
            self.d.source,
            listener,
            [points[:, j] for j in range(1, k + 2)],
            faces,
            go["distance"][ids],
            np.ones((n, bands)),
            np.zeros(n, dtype=np.int64),
            speed_of_sound=self.d.speed_of_sound,
            air=self.d.air,
            directivity=self.d.directivity,
            orientation=self.d.orientation,
            coherent=self.d.coherent,
            tolerance=self.tolerance,
        )
        return found.energy, found.spectrum


def _rows(paths: EarlyReflections, specular: EarlyReflections, rows: np.ndarray) -> np.ndarray:
    """Where the paths ``rows`` of ``specular`` are in ``paths``, which has them with the
    edge diffraction's energies, among its paths and sorted by arrival: by their length
    and faces."""
    where = {
        (float(paths.distance_m[i]), tuple(paths.faces[i].tolist())): i
        for i in np.flatnonzero(paths.edge < 0)
    }
    return np.array(
        [where[(float(specular.distance_m[r]), tuple(specular.faces[r].tolist()))] for r in rows],
        dtype=np.int64,
    )


def _perpendicular(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether the planes ``a`` and ``b`` are perpendicular: reflections off them commute."""
    dot = np.einsum("ij,ij->i", model.plane_normal[a], model.plane_normal[b])
    return np.abs(dot) < 1e-9


def _swaps(k: int):
    """The orders of k reflections that swap two next to each other."""
    for j in range(k - 1):
        order = list(range(k))
        order[j], order[j + 1] = order[j + 1], order[j]
        yield tuple(order)


def _once(events: list[dict], tolerance: float) -> list[dict]:
    """``events`` without those that repeat an earlier one: of the same path and kind,
    off the same plane, with the same source and listener sides."""
    keys = np.concatenate(
        [
            np.column_stack(
                [
                    e["row"],
                    e["kind"],
                    e["plane"],
                    np.round(e["source"] / tolerance),
                    np.round(e["listener"] / tolerance),
                ]
            )
            for e in events
        ]
    )
    _, first = np.unique(keys, axis=0, return_index=True)
    keep = np.zeros(len(keys), dtype=bool)
    keep[first] = True
    out, at = [], 0
    for e in events:
        mine = keep[at : at + len(e["row"])]
        at += len(e["row"])
        if mine.all():
            out.append(e)
        elif mine.any():
            out.append({key: value[mine] for key, value in e.items()})
    return out


def _on_leg(model, point, images, ends, leg) -> np.ndarray:
    """(n,) whether each ``point`` lies on the leg's side of the planes ``ends`` (n, 2,
    m) the leg may reflect off before and after (the side away from ``images``, its
    images on the source and the listener side); on the plane itself, the leg after it.
    True where ``leg`` is False."""
    ok = np.ones(len(point), dtype=bool)
    for side, image in enumerate(images):
        for plane in ends[:, side].T:
            there = leg & (plane >= 0)
            if not there.any():
                continue
            normal = model.plane_normal[np.maximum(plane, 0)]
            offset = model.plane_offset[np.maximum(plane, 0)]
            away = offset - np.einsum("ij,ij->i", normal, image)
            at = (np.einsum("ij,ij->i", normal, point) - offset) * np.sign(away)
            ok &= ~there | ((at >= 0.0) if side == 0 else (at > 0.0))
    return ok


def _closest(start, end, edges, edge) -> np.ndarray:
    """(n, 3) the points of the lines from ``start`` to ``end`` (n, 3) nearest to the
    lines of ``edges`` ``edge`` (n,); ``start`` where they are parallel."""
    line = end - start
    u = edges.direction[edge]
    w = start - edges.start[edge]
    along = np.einsum("ij,ij->i", line, u)
    square = np.einsum("ij,ij->i", line, line)
    across = square - along**2  # |line|^2 sin^2 of the angle between them
    usable = across > 1e-12 * np.maximum(square, 1e-300)
    s = (along * np.einsum("ij,ij->i", u, w) - np.einsum("ij,ij->i", line, w)) / np.where(
        usable, across, 1.0
    )
    return start + np.where(usable, s, 0.0)[:, None] * line


def _towards(model, point, image, planes, fallback):
    """Where the lines from ``point`` to ``image`` (its side of a path over an edge, the
    mirror image of what lies beyond the planes) first meet one of ``planes`` (n, m), -1
    for none; ``fallback`` where they meet none."""
    toward = image - point
    first = np.full(len(point), np.inf)
    for plane in planes.T:
        safe = np.maximum(plane, 0)
        normal, offset = model.plane_normal[safe], model.plane_offset[safe]
        across = np.einsum("ij,ij->i", normal, toward)
        usable = (plane >= 0) & (np.abs(across) > 1e-12)
        t = (offset - np.einsum("ij,ij->i", normal, point)) / np.where(usable, across, 1.0)
        first = np.where(usable & (t > 0.0) & (t <= 1.0) & (t < first), t, first)
    met = np.isfinite(first)
    return np.where(met[:, None], point + np.where(met, first, 0.0)[:, None] * toward, fallback)


def _nearest_face(model, plane: int, point: np.ndarray) -> int:
    """The face of ``plane`` nearest to ``point`` in it: the one ``point`` lies least far
    outside of (by its distance beyond the triangle's sides)."""
    candidates = np.flatnonzero(model.face_plane == plane)
    tri = model.room.mesh.triangles[candidates]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    area2 = np.einsum("ij,ij->i", n, n)
    # Each barycentric coordinate times its height is the distance inside the side
    # opposite (negative outside).
    inside = []
    for q, r in ((b, c), (c, a), (a, b)):
        w = np.einsum("ij,ij->i", np.cross(r - q, point - q), n) / area2
        inside.append(w * np.sqrt(area2) / np.maximum(np.linalg.norm(r - q, axis=1), 1e-300))
    return int(candidates[np.argmax(np.min(inside, axis=0))])


def _subset(part: dict, keep: np.ndarray) -> dict:
    """The terms ``keep`` of ``part``."""
    return {key: value[keep] for key, value in part.items()}
