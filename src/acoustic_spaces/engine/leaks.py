"""The leak finder: where rays escape from a model that is not closed.

The mesh checks count the open edges of a model, but do not show where its holes are.
:func:`find_leaks` traces rays from the sources, reflecting them diffusely off every
surface (a Lambert direction each time, whatever the materials, so that they reach
every corner) and passing them through surfaces that let sound through with the share
they let through. A ray that finds no surface ahead has escaped. Each escape is put
down to the gap whose open edges it passes closest to; gaps whose edges come within
a short distance of each other are one gap (a ceiling that stops short of a wall
leaves open edges on both). An escape right after a ray passed through a surface that
lets sound through (an open window to the outside) is put down to that surface. The
surfaces next to a gap say where to look, and its area is estimated from how often
rays find it: in a field of diffuse reflections, each reflection meets a share of the
boundary in proportion to its area, so a gap takes the share of reflections that
escape through it, times the area of the surfaces.

``acoustic-spaces validate --leaks`` prints the report and ``acoustic-spaces view
--leaks`` draws the escaping rays in 3D, as does the editor's View menu.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from acoustic_spaces.engine.particles import _lambert
from acoustic_spaces.room import Room, _label_text
from acoustic_spaces.spatial.directions import fibonacci_sphere

#: Rays that the finder traces by default, shared by the sources.
RAYS = 5000
#: Reflections each ray makes at most.
MAX_REFLECTIONS = 100
#: Open edges closer together than this share of the model's diagonal (and than
#: ``MIN_JOIN_M``) bound one gap.
JOIN_SHARE = 0.02
MIN_JOIN_M = 0.05
#: A gap is flat when its open edges spread out across their plane less than this share
#: of how far they spread out along it.
FLAT = 0.01
#: Surfaces named for each gap, the ones with the longest open edges first.
NAMED_SURFACES = 3


@dataclass
class Gap:
    """A place where rays left the model."""

    #: Rays that escaped through it.
    rays: int
    #: (3,) the mean of the points where they left (m).
    centre: np.ndarray
    #: Names of the surfaces next to it, the longest open edges first; for a surface
    #: that lets sound through, its own name.
    surfaces: tuple[str, ...]
    #: Length of the open edges round it (m); 0 for a surface that lets sound through,
    #: or where rays slipped between faces that the mesh joins.
    open_edges_m: float
    #: Its area (m2): the area its open edges run round, when they close up into loops;
    #: else as the rays found it, from the share of reflections that met it. 0 for a
    #: surface that lets sound through and a source outside the room.
    area_m2: float
    #: Its length and width (m): how far its open edges reach along the two directions
    #: they spread out most in; zeros where there are no open edges.
    size_m: tuple[float, float] = (0.0, 0.0)
    #: ``"gap"``: open edges round a hole; ``"surface"``: a surface that lets sound
    #: through (an opening the scene means to have, not a leak); ``"between faces"``:
    #: an edge of a closed mesh that rays slipped through; ``"outside"``: rays that left
    #: a source outside the room without meeting a surface (``surfaces`` names the
    #: source).
    kind: str = "gap"


@dataclass
class _Escapes:
    """The rays that escaped: where from, which way, and how they got there."""

    starts: np.ndarray  # (k, 3)
    directions: np.ndarray  # (k, 3)
    #: (k,) the face each passed through just before escaping, or -1.
    passed: np.ndarray
    #: (k,) whether it met a surface before escaping.
    bounced: np.ndarray
    #: (k,) its source (an index into the scene's sources; 0 without sources).
    source: np.ndarray


@dataclass
class Leaks:
    """Where the rays of :func:`find_leaks` escaped."""

    #: Rays traced.
    rays: int
    #: Reflections they made in all (passes through surfaces included).
    reflections: int
    #: Reflections each ray made at most.
    max_reflections: int
    #: (k, 3) where each escaping ray last reflected (or left its source).
    starts: np.ndarray
    #: (k, 3) its direction from there.
    directions: np.ndarray
    #: (k, 3) where it left the model: where it crosses the plane of its gap's open
    #: edges (for a gap that is not flat, its point nearest them; for an edge between
    #: joined faces, its point nearest that), or ``starts`` when it left through a
    #: surface or straight from its source.
    points: np.ndarray
    #: (k,) the gap each escaping ray left through (an index into ``gaps``).
    gap: np.ndarray
    #: The gaps, the one the most rays escaped through first.
    gaps: list[Gap]
    #: Total area of the surfaces (m2).
    surface_area_m2: float

    @property
    def escaped(self) -> int:
        return len(self.starts)

    @property
    def leaked(self) -> int:
        """Rays that escaped through a gap, not through a surface that lets sound
        through."""
        return sum(g.rays for g in self.gaps if g.kind != "surface")

    def summary(self) -> str:
        """A few lines for people: how many rays escaped, and where."""
        share = 100.0 * self.escaped / self.rays if self.rays else 0.0
        escaped = f"{self.escaped} escaped ({share:.1f} %)" if self.escaped else "none escaped"
        lines = [f"{self.rays} rays, up to {self.max_reflections} reflections each: {escaped}"]
        for k, gap in enumerate(self.gaps, 1):
            names = ", ".join(gap.surfaces)
            rays = f"{k}. {gap.rays} rays ({100.0 * gap.rays / self.rays:.1f} %)"
            area = f"about {gap.area_m2:.2g} m2"
            if gap.kind == "surface":
                text = f"{rays} left through {names}, which lets sound through"
            elif gap.kind == "outside":
                text = f"{rays} left source {names} without meeting a surface: it is outside"
            elif gap.kind == "between faces":
                text = f"{rays} slipped through an edge of {names} ({area})"
            else:
                length, width = gap.size_m
                text = f"{rays} through a gap of {area} ({length:.2f} x {width:.2f} m) by {names}"
            lines.append(textwrap.fill(text, 88, initial_indent="  ", subsequent_indent="     "))
            if gap.kind in ("gap", "between faces"):
                x, y, z = gap.centre
                lines.append(f"     at ({x:.2f}, {y:.2f}, {z:.2f}) m")
        return "\n".join(lines)


def find_leaks(
    room: Room,
    *,
    rays: int = RAYS,
    max_reflections: int = MAX_REFLECTIONS,
    seed: int = 0,
) -> Leaks:
    """Trace ``rays`` rays from the room's sources (shared by them; from the middle of
    the model when it has none) for up to ``max_reflections`` diffuse reflections each,
    and find where they escape."""
    if rays < 1 or max_reflections < 1:
        raise ValueError("rays and max_reflections must be positive")
    from acoustic_spaces import _native

    mesh = room.mesh
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    normals = mesh.face_normals()
    through = room.face_transmission()
    passing = np.zeros(len(mesh.faces)) if through is None else through.mean(axis=1)
    rng = np.random.default_rng(seed)

    points, owner = [], []
    for k, source in enumerate(room.scene.sources):
        elements = source.elements()
        points.extend(elements)
        owner.extend([k] * len(elements))
    if not points:
        lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
        points, owner = [0.5 * (lo + hi)], [0]
    which = np.arange(rays) % len(points)
    source = np.asarray(owner, dtype=np.int64)[which]
    p, d = np.asarray(points, dtype=float)[which], fibonacci_sphere(rays)
    face = np.full(rays, -1, dtype=np.int64)
    passed = np.full(rays, -1, dtype=np.int64)  # the surface a ray last passed through
    alive = np.ones(rays, dtype=bool)
    gone_rays = []
    reflections = 0
    for _ in range(max_reflections):
        index = np.flatnonzero(alive)
        if len(index) == 0:
            break
        dist, hit = bvh.intersect(
            np.ascontiguousarray(p[index]),
            np.ascontiguousarray(d[index]),
            np.full(len(index), np.inf),
            np.ascontiguousarray(face[index]),
        )
        out = hit < 0
        gone = index[out]
        gone_rays.append(gone)
        alive[gone] = False  # p and d keep where and which way it left

        index, dist, hit = index[~out], dist[~out], hit[~out]
        reflections += len(index)
        incoming = d[index]
        p[index] += dist[:, None] * incoming
        face[index] = hit
        crossing = rng.random(len(index)) < passing[hit]
        normal = normals[hit]
        normal = np.where((np.einsum("ij,ij->i", normal, incoming) > 0)[:, None], -normal, normal)
        d[index] = np.where(crossing[:, None], incoming, _lambert(normal, rng))
        passed[index] = np.where(crossing, hit, -1)

    gone = np.concatenate(gone_rays) if gone_rays else np.zeros(0, dtype=np.int64)
    escaped = _Escapes(
        starts=p[gone],
        directions=d[gone],
        passed=passed[gone],
        bounced=face[gone] >= 0,
        source=source[gone],
    )
    area = float(mesh.face_areas().sum())
    leak_points, gap, gaps = _gaps(room, escaped, reflections, area)
    return Leaks(
        rays=rays,
        reflections=reflections,
        max_reflections=max_reflections,
        starts=escaped.starts,
        directions=escaped.directions,
        points=leak_points,
        gap=gap,
        gaps=gaps,
        surface_area_m2=area,
    )


def open_edges(room: Room) -> tuple[np.ndarray, np.ndarray]:
    """The edges of the room's mesh that only one face has: (E, 2) their corners (indices
    into the mesh's vertices, in the order the face runs along them) and (E,) that
    face."""
    faces = room.mesh.faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = directed.min(axis=1) * (int(faces.max(initial=0)) + 1) + directed.max(axis=1)
    _, edge, counts = np.unique(key, return_inverse=True, return_counts=True)
    rows = np.flatnonzero(counts[edge.reshape(-1)] == 1)
    return directed[rows], rows % len(faces)


def _gaps(
    room: Room,
    escaped: _Escapes,
    reflections: int,
    area: float,
) -> tuple[np.ndarray, np.ndarray, list[Gap]]:
    """Where each escaping ray left, which gap that was, and the gaps."""
    mesh = room.mesh
    starts, directions = escaped.starts, escaped.directions
    count = len(starts)
    leak_points = starts.copy()
    corners, owners = open_edges(room)
    closed = len(corners) == 0
    if closed:
        corners, owners = _all_edges(room)
    edges = mesh.vertices[corners]
    # What each ray left by: 0 a gap's open edges, 1 a surface that lets sound through,
    # 2 the edge between two faces of a closed mesh, 3 a source outside the room; and
    # the gap's edge group, the surface, the face's surface or the source.
    kind = np.zeros(count, dtype=np.int64)
    key = np.zeros(count, dtype=np.int64)
    via = escaped.passed >= 0
    kind[via] = 1
    key[via] = mesh.face_labels[escaped.passed[via]]
    outside = closed & ~via & ~escaped.bounced
    kind[outside] = 3
    key[outside] = escaped.source[outside]
    rest = np.flatnonzero(~via & ~outside)
    parts = np.zeros(len(edges), dtype=np.int64) if closed else _edge_groups(room, edges)
    if len(rest) and not closed:
        # Through the flat gaps they cross first, where they cross them.
        crossed, where = _through_flat_gaps(starts[rest], directions[rest], corners, parts, mesh)
        key[rest] = crossed
        leak_points[rest] = where
        rest = rest[crossed < 0]
    if len(rest):
        nearest, t = _nearest_edges(starts[rest], directions[rest], edges)
        leak_points[rest] = starts[rest] + t[:, None] * directions[rest]
        if closed:
            kind[rest] = 2
            key[rest] = mesh.face_labels[owners[nearest]]
        else:
            key[rest] = parts[nearest]
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)

    found = []
    for k, name in sorted({(int(a), int(b)) for a, b in zip(kind, key, strict=True)}):
        members = np.flatnonzero((kind == k) & (key == name))
        edge_length = 0.0
        loop, size = None, (0.0, 0.0)
        if k == 0:
            mine = parts == name
            per_label: dict[int, float] = {}
            for label, length in zip(mesh.face_labels[owners[mine]], lengths[mine], strict=True):
                per_label[int(label)] = per_label.get(int(label), 0.0) + float(length)
            order = sorted(per_label, key=lambda lab: -per_label[lab])[:NAMED_SURFACES]
            surfaces = tuple(_label_text(mesh.labels[lab]) for lab in order)
            edge_length = float(lengths[mine].sum())
            loop = _loop_area(corners[mine], mesh.vertices)
            size = _spread(edges[mine].reshape(-1, 3))
        elif k == 3:
            surfaces = (room.scene.sources[name].name,)
        else:
            surfaces = (_label_text(mesh.labels[name]),)
        share = len(members) / max(reflections + count, 1)
        found_area = area * share / max(1.0 - share, 1e-12) if k in (0, 2) else 0.0
        gap = Gap(
            rays=len(members),
            centre=leak_points[members].mean(axis=0),
            surfaces=surfaces,
            open_edges_m=edge_length,
            area_m2=found_area if k != 0 or loop is None else loop,
            size_m=size,
            kind=("gap", "surface", "between faces", "outside")[k],
        )
        found.append((gap, members))
    found.sort(key=lambda item: -item[0].rays)
    group = np.full(count, -1, dtype=np.int64)
    for k, (_, members) in enumerate(found):
        group[members] = k
    return leak_points, group, [g for g, _ in found]


def _all_edges(room: Room) -> tuple[np.ndarray, np.ndarray]:
    """Every edge of the room's mesh once, as :func:`open_edges` gives the open ones."""
    faces = room.mesh.faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = directed.min(axis=1) * (int(faces.max(initial=0)) + 1) + directed.max(axis=1)
    _, rows = np.unique(key, return_index=True)
    return directed[rows], rows % len(faces)


def _loop_area(corners: np.ndarray, vertices: np.ndarray) -> float | None:
    """The area a gap's open edges (E, 2) run round, or None when they do not close
    up into loops: half the length of the sum of their ends' cross products, exact
    for a flat hole."""
    out = np.bincount(corners[:, 0], minlength=len(vertices))
    into = np.bincount(corners[:, 1], minlength=len(vertices))
    if np.any(out != into):
        return None
    a, b = vertices[corners[:, 0]], vertices[corners[:, 1]]
    centre = a.mean(axis=0)  # nearer the edges than the origin, for rounding
    return 0.5 * float(np.linalg.norm(np.cross(a - centre, b - centre).sum(axis=0)))


def _through_flat_gaps(
    starts: np.ndarray,
    directions: np.ndarray,
    corners: np.ndarray,
    parts: np.ndarray,
    mesh,
) -> tuple[np.ndarray, np.ndarray]:
    """For each ray, the first flat gap it passes through ahead of its start (its group
    of open edges in ``parts``, or -1 for none), and where.

    A gap is flat when its open edges (``corners``) lie in one plane and close up into
    loops; a ray passes through it when it crosses that plane inside the loops (an odd
    number of their edges round the crossing).
    """
    group = np.full(len(starts), -1, dtype=np.int64)
    where = starts.copy()
    first = np.full(len(starts), np.inf)
    for name in np.unique(parts):
        mine = corners[parts == name]
        if _loop_area(mine, mesh.vertices) is None:
            continue
        points = mesh.vertices[mine.reshape(-1)]
        centre = points.mean(axis=0)
        _, spread, axes = np.linalg.svd(points - centre, full_matrices=False)
        if len(spread) < 3 or spread[2] > FLAT * spread[0]:
            continue
        across = directions @ axes[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (centre - starts) @ axes[2] / across
        ahead = (np.abs(across) > 1e-9) & (t > 1e-9) & (t < first)
        t = np.where(ahead, t, 0.0)
        crossing = starts + t[:, None] * directions
        # In the plane's own coordinates: the crossing and the loops' edges.
        q = (crossing - centre) @ axes[:2].T
        a = (mesh.vertices[mine[:, 0]] - centre) @ axes[:2].T
        b = (mesh.vertices[mine[:, 1]] - centre) @ axes[:2].T
        inside = _inside_loops(q, a, b) & ahead
        group[inside] = name
        where[inside] = crossing[inside]
        first[inside] = t[inside]
    return group, where


def _inside_loops(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether each point (n, 2) lies inside the loops of edges from ``a`` to ``b``
    (E, 2): an odd number of them cross the line from it towards +x."""
    inside = np.zeros(len(q), dtype=bool)
    rise = b[:, 1] - a[:, 1]
    slope = np.divide(b[:, 0] - a[:, 0], rise, out=np.zeros(len(a)), where=rise != 0)
    block = max(1, 2_000_000 // max(len(a), 1))
    for begin in range(0, len(q), block):
        x, y = q[begin : begin + block, :1], q[begin : begin + block, 1:]
        spans = (a[None, :, 1] > y) != (b[None, :, 1] > y)
        crossing = a[None, :, 0] + (y - a[None, :, 1]) * slope[None, :]
        inside[begin : begin + block] = np.count_nonzero(spans & (x < crossing), axis=1) % 2 == 1
    return inside


def _spread(points: np.ndarray) -> tuple[float, float]:
    """How far ``points`` (n, 3), two or more, reach along the two directions they
    spread out most in."""
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    along = centred @ axes[:2].T
    reach = along.max(axis=0) - along.min(axis=0)
    return float(reach[0]), float(reach[1])


def _edge_groups(room: Room, edges: np.ndarray) -> np.ndarray:
    """(E,) a group per open edge: edges that meet, or come within the join distance
    of each other, bound one gap."""
    if len(edges) == 0:
        return np.zeros(0, dtype=np.int64)
    lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
    reach = max(JOIN_SHARE * float(np.linalg.norm(hi - lo)), MIN_JOIN_M)
    # Points along every edge, no further apart than half the join distance.
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    steps = np.maximum(np.ceil(2.0 * lengths / reach).astype(np.int64), 1)
    owner = np.repeat(np.arange(len(edges)), steps + 1)
    fraction = np.concatenate([np.linspace(0.0, 1.0, s + 1) for s in steps])
    samples = edges[owner, 0] + fraction[:, None] * (edges[owner, 1] - edges[owner, 0])
    pairs = cKDTree(samples).query_pairs(reach, output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(edges), len(edges)))
    _, label = connected_components(graph, directed=False)
    return label.astype(np.int64)


def _nearest_edges(
    origins: np.ndarray, directions: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each ray, the edge it passes closest to ahead of its origin, and the
    distance along the ray to where it passes closest (0 when no edge lies ahead)."""
    a = edges[:, 0]
    e = edges[:, 1] - a
    ee = np.maximum(np.einsum("ij,ij->i", e, e), 1e-300)
    nearest = np.zeros(len(origins), dtype=np.int64)
    along = np.zeros(len(origins))
    block = max(1, 2_000_000 // len(edges))
    for begin in range(0, len(origins), block):
        o, d = origins[begin : begin + block], directions[begin : begin + block]
        w = o[:, None, :] - a[None, :, :]  # (k, E, 3)
        de = d @ e.T
        dw = np.einsum("kj,kej->ke", d, w)
        ew = np.einsum("ej,kej->ke", e, w)
        # Closest points of the line along each ray and of each edge, then the ray's
        # point kept ahead and the edge's within its ends.
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(ee - de**2 > 1e-12 * ee, (ew - de * dw) / (ee - de**2), 0.0)
        s = np.clip(s, 0.0, 1.0)
        t = np.maximum(s * de - dw, 0.0)
        s = np.clip((ew + t * de) / ee, 0.0, 1.0)
        gap = w + t[..., None] * d[:, None, :] - s[..., None] * e[None, :, :]
        distance = np.einsum("kej,kej->ke", gap, gap)
        # An edge it leaves from, or one behind it, is not where it went.
        distance[t <= 1e-9] = np.inf
        best = distance.argmin(axis=1)
        rows = np.arange(len(o))
        nearest[begin : begin + block] = best
        along[begin : begin + block] = np.where(np.isfinite(distance[rows, best]), t[rows, best], 0)
    return nearest, along


__all__ = ["MAX_REFLECTIONS", "RAYS", "Gap", "Leaks", "find_leaks", "open_edges"]
