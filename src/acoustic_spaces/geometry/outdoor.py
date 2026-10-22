"""Outdoor scenes: ground under an open sky, with buildings and barriers on it.

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
- Vegetation (trees, hedges, belts of trees) is no surface: each is a volume
  (:class:`PlantShape`) that sound passes through, losing some of itself (see
  :mod:`acoustic_spaces.engine.vegetation`). The ground reaches round it, and the sky
  over it, as round and over a building.
- Zones are parts of the ground with surfaces of their own (a road, a lawn): polygons,
  cut to the ground's extent, that may touch but not overlap. The ground is then split
  into strips at the zones' corners and each strip into pieces between the zones' edges,
  so that its faces meet edge to edge and the box stays closed.
- A terrain (``ground.terrain``, a :class:`~acoustic_spaces.geometry.terrain.HeightGrid`)
  replaces the flat ground by the grid sampled over the extent (given, or the whole
  grid: the terrain sets the ground's size, not the margin) and triangulated, the sky
  walls rising from its edge (:func:`terrain_box`); the zones take the triangles whose
  centres lie in them. Buildings and barriers are sunk a little below the lowest ground
  under them, so that they stand on the terrain however it runs: a building's roof lies
  its height over the terrain's mean height at its footprint's corners, a barrier's top
  follows the terrain along its path, and vegetation stands on the terrain under its
  middle. The model file's objects are taken as they are. Edges buried in the terrain
  do not diffract, nor do its own ridges, which bend too little at each grid line: the
  sound over its crests is :mod:`acoustic_spaces.engine.crests`'s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.geometry.builders import extruded_mesh, plan_problem
from acoustic_spaces.geometry.mesh import RoomMesh
from acoustic_spaces.geometry.terrain import HeightGrid
from acoustic_spaces.geometry.triangulate import triangulate_polygon

#: The surface names of the ground and of the sky round and over it.
GROUND = "ground"
SKY = "sky"
#: Points this close (m) to the ground's height lie on it.
ON_GROUND_M = 0.01
#: Coordinates farther than this (m) from the origin lose precision in the mesh:
#: a map's coordinates need ``ground.map_origin`` near the scene.
FAR_M = 1e5


@dataclass(frozen=True, eq=False)
class Layout:
    """Where the ground and the sky of an outdoor scene lie (m)."""

    #: (x_min, y_min, x_max, y_max) of the ground.
    extent: tuple[float, float, float, float]
    #: Height (z) of the ground; with a terrain, of its lowest point over the extent.
    ground: float
    #: Height (z) of the sky.
    sky: float
    #: Each building's name, footprint (n, 2) and roof height (z), for the inside checks.
    buildings: tuple[tuple[str, np.ndarray, float], ...] = ()
    #: Each ground zone's name and polygon (n, 2), cut to the extent.
    zones: tuple[tuple[str, np.ndarray], ...] = ()
    #: The volumes of the scene's vegetation, in ``geometry.vegetation`` order.
    vegetation: tuple[PlantShape, ...] = ()
    #: The terrain's surface as the mesh has it, in metres; None for flat ground.
    terrain: TerrainSurface | None = None

    def ground_at(self, points: np.ndarray) -> np.ndarray:
        """(n,) the ground's height under each point (n, 2 or 3)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if self.terrain is None:
            return np.full(len(points), self.ground)
        return self.terrain.height_at(points)

    def zone_at(self, points: np.ndarray) -> np.ndarray:
        """(n,) the index of the zone each point (n, 2 or 3) lies in, or -1."""
        points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
        out = np.full(len(points), -1, dtype=np.int64)
        for k, (_, polygon) in enumerate(self.zones):
            out[(out < 0) & in_polygon(points, polygon)] = k
        return out

    def fresnel_shares(
        self, before: np.ndarray, at: np.ndarray, after: np.ndarray, wavelength: np.ndarray
    ) -> np.ndarray:
        """(n, bands, 1 + zones) the share of the Fresnel zone of each reflection off the
        ground, at ``at`` (n, 3) on the way from ``before`` to ``after`` (n, 3), that
        lies outside the zones (first) and in each zone, for each of ``wavelength``
        (bands,) m. The Fresnel zone is where a path through the ground is at most
        :data:`FRESNEL_PATH_DIFFERENCE` wavelengths longer than the reflected one: an
        ellipse round the reflection point, small at high frequencies and near normal
        incidence, long near grazing incidence. Past the extent, the ground goes on as
        it is at its edge."""
        before, at, after = (np.atleast_2d(np.asarray(p, dtype=float)) for p in (before, at, after))
        wavelength = np.asarray(wavelength, dtype=float)
        n, bands = len(at), len(wavelength)
        out = np.zeros((n, bands, 1 + len(self.zones)))
        here = self.zone_at(at) + 1
        out[np.arange(n)[:, None], np.arange(bands)[None, :], here[:, None]] = 1.0
        if not self.zones or n == 0:
            return out
        centre, axes = _fresnel_ellipses(before, at, after, self.ground_at(at), wavelength)
        # Only ellipses that reach the edge of a zone are shared between grounds.
        edges = [(p, np.roll(p, -1, axis=0)) for _, p in self.zones]
        start = np.vstack([a for a, _ in edges])
        end = np.vstack([b for _, b in edges])
        reach = np.linalg.norm(axes, axis=-2).max(axis=-1)  # (n, bands) the longest semi-axis
        near = _segment_distance(centre.reshape(-1, 2), start, end).reshape(n, bands) < reach
        rows, columns = np.nonzero(near)
        if len(rows) == 0:
            return out
        x0, y0, x1, y1 = self.extent
        samples = centre[rows, columns, None, :] + np.einsum(
            "kij,sj->ksi", axes[rows, columns], _DISK
        )  # (k, S, 2)
        samples = np.clip(samples, [x0, y0], [x1, y1]).reshape(-1, 2)
        part = (self.zone_at(samples) + 1).reshape(len(rows), len(_DISK))
        shares = np.zeros((len(rows), out.shape[2]))
        for p in range(out.shape[2]):
            shares[:, p] = np.mean(part == p, axis=1)
        out[rows, columns] = shares
        return out

    def building_at(self, points: np.ndarray) -> list[str | None]:
        """The building each point (n, 3) lies in, or None."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out: list[str | None] = [None] * len(points)
        ground = self.ground_at(points)
        for name, footprint, roof in self.buildings:
            inside = in_polygon(points[:, :2], footprint) & (points[:, 2] < roof)
            inside &= points[:, 2] > ground
            for k in np.flatnonzero(inside).tolist():
                out[k] = out[k] or name
        return out


#: The Fresnel zone of a reflection off the ground: where a path through the ground is at
#: most this many wavelengths longer than the reflected one (the first Fresnel zone).
FRESNEL_PATH_DIFFERENCE = 0.5


@dataclass(frozen=True, eq=False)
class TerrainSurface:
    """The ground of a terrain as the scene's mesh has it: the grid sampled at its own
    spacing over the extent (a little finer where the extent is not a whole number of
    cells across), each cell cut into two triangles along the diagonal that bends
    less. Heights between the nodes are those of the triangles, so what stands on the
    ground meets the mesh exactly."""

    #: The grid it was sampled from, in metres with the ground's height added.
    grid: HeightGrid
    extent: tuple[float, float, float, float]
    #: (nx + 1,) and (ny + 1,) the nodes' x and y.
    xs: np.ndarray
    ys: np.ndarray
    #: (ny + 1, nx + 1) the nodes' heights.
    z: np.ndarray
    #: (ny, nx) whether a cell is cut from its (i, j) corner to its (i + 1, j + 1) one;
    #: otherwise from (i + 1, j) to (i, j + 1).
    first: np.ndarray

    @classmethod
    def of(cls, grid: HeightGrid, extent) -> TerrainSurface:
        x0, y0, x1, y1 = extent
        d = grid.spacing
        nx = max(1, int(math.ceil((x1 - x0) / d - 1e-9)))
        ny = max(1, int(math.ceil((y1 - y0) / d - 1e-9)))
        xs = x0 + (x1 - x0) * np.arange(nx + 1) / nx
        ys = y0 + (y1 - y0) * np.arange(ny + 1) / ny
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        z = grid.height_at(np.column_stack([gx.ravel(), gy.ravel()])).reshape(ny + 1, nx + 1)
        first = np.abs(z[:-1, :-1] - z[1:, 1:]) <= np.abs(z[:-1, 1:] - z[1:, :-1])
        return cls(grid, tuple(float(v) for v in extent), xs, ys, z, first)

    @property
    def nx(self) -> int:
        return len(self.xs) - 1

    @property
    def ny(self) -> int:
        return len(self.ys) - 1

    @property
    def lowest(self) -> float:
        return float(self.z.min())

    @property
    def highest(self) -> float:
        return float(self.z.max())

    def _cell(self, points: np.ndarray):
        """The cell (i, j) under each point (n, 2) and where in it (u, v in [0, 1]);
        past the edge, the nearest cell's edge."""
        x0, y0, x1, y1 = self.extent
        nx, ny = self.nx, self.ny
        u = np.clip((points[:, 0] - x0) / (x1 - x0) * nx, 0.0, nx)
        v = np.clip((points[:, 1] - y0) / (y1 - y0) * ny, 0.0, ny)
        i = np.minimum(np.floor(u).astype(np.int64), nx - 1)
        j = np.minimum(np.floor(v).astype(np.int64), ny - 1)
        return i, j, u - i, v - j

    def height_at(self, points) -> np.ndarray:
        """(n,) the surface's height at each point (n, 2 or 3)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
        i, j, u, v = self._cell(points)
        z = self.z
        za, zb, zc, ze = z[j, i], z[j, i + 1], z[j + 1, i + 1], z[j + 1, i]
        first = self.first[j, i]
        lower = np.where(first, u >= v, u + v <= 1.0)  # the triangle at the (i, j) corner
        along_first = np.where(
            lower, za + (zb - za) * u + (zc - zb) * v, za + (zc - ze) * u + (ze - za) * v
        )
        along_second = np.where(
            lower,
            za + (zb - za) * u + (ze - za) * v,
            zc + (zb - zc) * (1.0 - v) + (ze - zc) * (1.0 - u),
        )
        return np.where(first, along_first, along_second)

    def profile(self, a, b) -> np.ndarray:
        """(k, 3) the surface along the line from ``a`` to ``b`` (2,): its corners at
        every edge of a triangle it crosses, so that the line between them lies on the
        surface."""
        a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
        x0, y0, x1, y1 = self.extent
        nx, ny = self.nx, self.ny
        d = b - a
        ts = [np.array([0.0, 1.0])]
        for lines, start, end in ((self.xs, a[0], d[0]), (self.ys, a[1], d[1])):
            if abs(end) > 1e-12:
                ts.append((lines - start) / end)
        ts = np.unique(np.clip(np.concatenate(ts), 0.0, 1.0))
        # The cells between the crossings, and where the line crosses their diagonals.
        size, corner = np.array([x1 - x0, y1 - y0]), np.array([x0, y0])
        mid = (a + (0.5 * (ts[:-1] + ts[1:]))[:, None] * d - corner) / size * [nx, ny]
        i = np.clip(np.floor(mid[:, 0]).astype(int), 0, nx - 1)
        j = np.clip(np.floor(mid[:, 1]).astype(int), 0, ny - 1)
        ua, du = (a - corner) / size * [nx, ny], d / size * [nx, ny]
        # Along the line u = ua + t du: the diagonal u - i = v - j, or u - i + v - j = 1.
        first = self.first[j, i]
        rate = np.where(first, du[0] - du[1], du[0] + du[1])
        start = np.where(first, (ua[0] - i) - (ua[1] - j), (ua[0] - i) + (ua[1] - j) - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = -start / rate
        inside = (np.abs(rate) > 1e-12) & (crossing > ts[:-1]) & (crossing < ts[1:])
        ts = np.unique(np.concatenate([ts, crossing[inside]]))
        points = a + ts[:, None] * d
        return np.column_stack([points, self.height_at(points)])

    def describe(self) -> str:
        return (
            f"{self.nx + 1} x {self.ny + 1} points, from z = {self.lowest:g} to "
            f"{self.highest:g} m (grid {self.grid.spacing:g} m)"
        )


@dataclass(frozen=True, eq=False)
class PlantShape:
    """The volume of a tree, a hedge or a belt of trees (m): with an ``outline`` (n, 2),
    the polygon raised from ``low`` to ``high`` (z); with a ``centre`` (3,), a lone
    tree's crown, a spheroid ``radius`` across horizontally from ``low`` to ``high``."""

    name: str
    kind: str  # belt, hedge or tree
    low: float
    high: float
    outline: np.ndarray | None = None
    centre: np.ndarray | None = None
    radius: float = 0.0

    @property
    def half_height(self) -> float:
        return 0.5 * (self.high - self.low)

    def corners(self) -> np.ndarray:
        """(k, 3) corners of a box round it."""
        if self.outline is not None:
            plan = self.outline
        else:
            x, y = self.centre[:2]
            r = self.radius
            plan = np.array([[x - r, y - r], [x + r, y + r]])
        n = len(plan)
        return np.vstack([np.column_stack([plan, np.full(n, z)]) for z in (self.low, self.high)])

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Whether each point (..., 3) lies inside."""
        points = np.asarray(points, dtype=float)
        if self.outline is None:
            q = (points - self.centre) / [self.radius, self.radius, self.half_height]
            return np.einsum("...i,...i->...", q, q) < 1.0
        flat = points.reshape(-1, 3)
        within = (flat[:, 2] > self.low) & (flat[:, 2] < self.high)
        within &= in_polygon(flat[:, :2], self.outline)
        return within.reshape(points.shape[:-1])


def strip(path, width: float) -> np.ndarray:
    """The outline (n, 2) of a strip ``width`` wide along ``path`` (m, 2): its sides half
    the width either side, mitred at the bends (at most twice that far out)."""
    path = np.asarray(path, dtype=float)
    along = np.diff(path, axis=0)
    lengths = np.linalg.norm(along, axis=1)
    scale = max(float(np.ptp(path, axis=0).max()), 1.0)
    if np.any(lengths <= 1e-9 * scale):
        k = int(np.argmax(lengths <= 1e-9 * scale))
        raise ValueError(f"points {k + 1} and {k + 2} are in one place")
    along /= lengths[:, None]
    normals = np.column_stack([-along[:, 1], along[:, 0]])
    offsets = np.vstack([normals[:1], normals[:-1] + normals[1:], normals[-1:]])
    offsets /= np.maximum(np.linalg.norm(offsets, axis=1), 1e-12)[:, None]
    # The mitre: as far out as keeps the sides half the width off the path.
    reach = np.ones(len(path))
    reach[1:-1] = 1.0 / np.maximum(np.einsum("ij,ij->i", offsets[1:-1], normals[1:]), 0.5)
    side = 0.5 * width * offsets * reach[:, None]
    return np.vstack([path + side, (path - side)[::-1]])


def vegetation_shapes(
    geometry, scale: float = 1.0, terrain: TerrainSurface | None = None
) -> tuple[PlantShape, ...]:
    """The volumes of ``geometry.vegetation`` (a scene's :class:`Geometry`), in metres
    (``scale``: the geometry's units in metres) over its ground, or over ``terrain``
    (in metres) under each one's middle."""
    flat = scale * geometry.ground.height
    shapes = []
    for name, plant in zip(geometry.vegetation_names(), geometry.vegetation or [], strict=True):
        ground = flat
        if terrain is not None:
            if plant.kind == "tree":
                middle = np.array([[scale * v for v in plant.tree]])
            elif plant.kind == "hedge":
                middle = (scale * np.asarray(plant.path, dtype=float)).mean(axis=0)[None]
            else:
                middle = (scale * np.asarray(plant.footprint, dtype=float)).mean(axis=0)[None]
            ground = float(terrain.height_at(middle)[0])
        low, high = ground + scale * plant.base, ground + scale * plant.height
        if plant.kind == "tree":
            x, y = (scale * v for v in plant.tree)
            centre = np.array([x, y, 0.5 * (low + high)])
            radius = 0.5 * scale * plant.crown_diameter
            shapes.append(PlantShape(name, "tree", low, high, centre=centre, radius=radius))
            continue
        if plant.kind == "hedge":
            try:
                outline = strip(scale * np.asarray(plant.path, dtype=float), scale * plant.width)
            except ValueError as exc:
                raise ValueError(f"hedge {name!r}: {exc}") from None
        else:
            outline = scale * np.asarray(plant.footprint, dtype=float)
        problem = plan_problem(outline)
        if problem is not None:
            words = problem.replace("a floor plan", "an outline").replace(
                "the floor plan", "the outline"
            )
            raise ValueError(f"{plant.kind} {name!r}: {words}")
        shapes.append(PlantShape(name, plant.kind, low, high, outline=outline))
    return tuple(shapes)


def _sunflower(count: int) -> np.ndarray:
    """(count, 2) points spread evenly over the unit disk, each standing for an equal
    area of it."""
    k = np.arange(count)
    angle = k * np.pi * (3.0 - np.sqrt(5.0))
    return np.sqrt((k + 0.5) / count)[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])


#: The points of a Fresnel zone looked up in the zones to share it out, on the unit disk.
_DISK = _sunflower(96)


def _fresnel_ellipses(before, at, after, ground, wavelength):
    """The Fresnel zones of reflections off the ground at height ``ground`` ((n,), under
    each reflection; see :meth:`Layout.fresnel_shares`): (n, bands, 2) their centres and
    (n, bands, 2, 2) the maps from the unit disk onto them (their semi-axes as columns).

    The zone is where the ground cuts the ellipsoid of points whose distances to the
    image of ``before`` and to ``after`` add up to the reflected path's length L plus
    the path difference d: semi-axes a = (L + d) / 2 along the line between those foci
    and b = sqrt(d (2 L + d)) / 2 across it."""
    ground = np.asarray(ground, dtype=float)
    image = before.copy()
    image[:, 2] = 2.0 * ground - image[:, 2]
    axis = after - image
    length = np.maximum(np.linalg.norm(axis, axis=1), 1e-12)  # (n,)
    e = axis / length[:, None]
    delta = FRESNEL_PATH_DIFFERENCE * np.asarray(wavelength, dtype=float)  # (bands,)
    a2 = 0.25 * (length[:, None] + delta) ** 2  # (n, bands)
    b2 = 0.25 * delta * (2.0 * length[:, None] + delta)
    kappa = 1.0 / a2 - 1.0 / b2  # the ellipsoid: |v|² / b² + kappa (e . v)² <= 1
    horizontal = e[:, :2]
    h2 = np.sum(horizontal**2, axis=1)[:, None]  # (n, 1)
    ez = e[:, 2][:, None]
    w = (ground - 0.5 * (image[:, 2] + after[:, 2]))[:, None]  # the plane from the centre
    along = 1.0 / b2 + kappa * h2  # the cut's curvature along the horizontal of the axis
    shift = -kappa * ez * w / along  # (n, bands): the cut's centre, along that horizontal
    r2 = 1.0 - w**2 * (kappa * ez**2 + 1.0 / b2 - (kappa * ez) ** 2 * h2 / along)
    r = np.sqrt(np.clip(r2, 0.0, None))
    middle = 0.5 * (image[:, :2] + after[:, :2])
    centre = middle[:, None, :] + shift[..., None] * horizontal[:, None, :]
    # The semi-axes: r / sqrt(along) along the horizontal of the axis, r b across it.
    flat = np.sqrt(h2) < 1e-12
    u = np.where(flat, [[1.0, 0.0]], horizontal / np.maximum(np.sqrt(h2), 1e-300))  # (n, 2)
    across = np.column_stack([-u[:, 1], u[:, 0]])
    first = (r / np.sqrt(along))[..., None] * u[:, None, :]  # (n, bands, 2)
    second = (r * np.sqrt(b2))[..., None] * across[:, None, :]
    return centre, np.stack([first, second], axis=-1)


def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """(n,) the distance from each of ``points`` (n, 2) to the nearest of the segments
    from ``start`` to ``end`` (E, 2)."""
    d = end - start  # (E, 2)
    t = np.einsum("nej,ej->ne", points[:, None, :] - start[None], d) / np.maximum(
        np.sum(d**2, axis=1), 1e-300
    )
    nearest = start[None] + np.clip(t, 0.0, 1.0)[..., None] * d[None]
    return np.min(np.linalg.norm(points[:, None, :] - nearest, axis=2), axis=1)


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
    """A building on flat ground at height ``ground``: its ``footprint`` (n, 2) raised
    by ``height``, walls and a flat roof, open underneath; faces point out of it (on a
    terrain, see :func:`building_on_terrain`)."""
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
    """A thin screen from flat ground at height ``ground`` up by ``height``, along
    ``path`` (n, 2) (on a terrain, see :func:`barrier_on_terrain`)."""
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


def ground_box(
    extent: tuple[float, float, float, float],
    ground: float,
    sky: float,
    zones: tuple[tuple[str, np.ndarray], ...] = (),
) -> RoomMesh:
    """The ground over ``extent`` at height ``ground``, closed by the sky up to ``sky``;
    faces point into the air. ``zones`` are named polygons (n, 2) inside the extent that
    do not overlap: parts of the ground with surfaces ``<name>``, ``zones`` and
    ``ground``."""
    x0, y0, x1, y1 = extent
    if not zones:
        box = extruded_mesh(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), sky - ground)
        face_labels = np.minimum(box.face_labels, 1)  # the floor: ground; the rest: sky
        return RoomMesh(
            box.vertices + [0.0, 0.0, ground], box.faces, face_labels, [(GROUND,), (SKY,)]
        )
    tolerance = 1e-9 * max(x1 - x0, y1 - y0)
    polygons = [np.asarray(polygon, dtype=float) for _, polygon in zones]
    xs = _merged(np.concatenate([[x0, x1], *[p[:, 0] for p in polygons]]), tolerance)
    segments = []  # the zones' edges that are not along x, left end first
    for polygon in polygons:
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            if abs(b[0] - a[0]) > tolerance:
                segments.append((a, b) if a[0] < b[0] else (b, a))

    # Strips between the xs, each cut into pieces between the edges that cross it.
    pieces = []  # (strip, (bottom at left, at right), (top at left, at right), label)
    on_line: list[list[float]] = [[] for _ in xs]
    for strip, (xa, xb) in enumerate(zip(xs[:-1], xs[1:], strict=True)):
        lines = [(y0, y0), (y1, y1)]
        for a, b in segments:
            if a[0] <= xa + tolerance and b[0] >= xb - tolerance:
                slope = (b[1] - a[1]) / (b[0] - a[0])
                lines.append((a[1] + slope * (xa - a[0]), a[1] + slope * (xb - a[0])))
        lines.sort(key=lambda line: line[0] + line[1])
        for low, high in zip(lines[:-1], lines[1:], strict=True):
            if high[0] - low[0] <= tolerance and high[1] - low[1] <= tolerance:
                continue  # one edge twice: shared by two zones, or on the extent's edge
            middle = np.array([[0.5 * (xa + xb), 0.25 * (low[0] + low[1] + high[0] + high[1])]])
            inside = [k for k, p in enumerate(polygons) if in_polygon(middle, p)[0]]
            pieces.append((strip, low, high, 2 + inside[0] if inside else 0))
            on_line[strip] += [low[0], high[0]]
            on_line[strip + 1] += [low[1], high[1]]
    heights = [_merged(np.array(ys), tolerance) for ys in on_line]

    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[int, int], int] = {}

    def corner(line: int, y: float) -> int:
        """The vertex on line ``line`` (x = xs[line]) at y, among that line's heights."""
        row = int(np.argmin(np.abs(heights[line] - y)))
        if (line, row) not in index:
            index[(line, row)] = len(vertices)
            vertices.append((float(xs[line]), float(heights[line][row]), ground))
        return index[(line, row)]

    def chain(line: int, low: float, high: float) -> list[int]:
        ys = heights[line]
        return [corner(line, y) for y in ys[(ys >= low - tolerance) & (ys <= high + tolerance)]]

    faces, labels = [], []
    for strip, low, high, label in pieces:
        left = chain(strip, low[0], high[0])
        right = chain(strip + 1, low[1], high[1])
        for a, b, c in _zip_chains(
            left, right, [vertices[k][1] for k in left], [vertices[k][1] for k in right]
        ):
            faces.append(_facing(vertices, (a, b, c), (0.0, 0.0, 1.0)))
            labels.append(label)

    # The sky: four walls whose foot meets every corner of the ground on its edge, and
    # a lid.
    top = [len(vertices) + k for k in range(4)]
    vertices += [(x0, y0, sky), (x1, y0, sky), (x1, y1, sky), (x0, y1, sky)]
    last = len(xs) - 1
    walls = (
        ([corner(i, y0) for i in range(len(xs))], xs, top[:2], [x0, x1], (0.0, 1.0, 0.0)),
        ([corner(i, y1) for i in range(len(xs))], xs, top[3:1:-1], [x0, x1], (0.0, -1.0, 0.0)),
        (
            [corner(0, y) for y in heights[0]],
            heights[0],
            [top[0], top[3]],
            [y0, y1],
            (1.0, 0.0, 0.0),
        ),
        (
            [corner(last, y) for y in heights[last]],
            heights[last],
            top[1:3],
            [y0, y1],
            (-1.0, 0.0, 0.0),
        ),
    )
    for foot, along, rim, rim_along, inward in walls:
        for a, b, c in _zip_chains(foot, rim, list(along), rim_along):
            faces.append(_facing(vertices, (a, b, c), inward))
            labels.append(1)
    for triangle in ((top[0], top[1], top[2]), (top[0], top[2], top[3])):
        faces.append(_facing(vertices, triangle, (0.0, 0.0, -1.0)))
        labels.append(1)
    names = [(GROUND,), (SKY,), *[(name, "zones", GROUND) for name, _ in zones]]
    return RoomMesh(np.array(vertices), np.array(faces), np.array(labels), names)


def terrain_box(
    surface: TerrainSurface, sky: float, zones: tuple[tuple[str, np.ndarray], ...] = ()
) -> RoomMesh:
    """The terrain ``surface`` closed by the sky up to ``sky``; faces point into the
    air. The sky's walls rise from the ground's edge. ``zones`` (see
    :func:`ground_box`) take the triangles whose centres lie in them."""
    nx, ny = surface.nx, surface.ny
    gx, gy = np.meshgrid(surface.xs, surface.ys, indexing="xy")
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    z = surface.z.ravel()
    vertices = [np.column_stack([nodes, z])]

    def node(i, j):
        return j * (nx + 1) + i

    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    i, j = i.ravel(), j.ravel()
    a, b, c, e = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
    first = surface.first[j, i]
    cells = np.where(
        first[:, None, None],
        np.stack([np.stack([a, b, c], axis=1), np.stack([a, c, e], axis=1)], axis=1),
        np.stack([np.stack([a, b, e], axis=1), np.stack([b, c, e], axis=1)], axis=1),
    )  # (cells, 2, 3), counter-clockwise seen from above: normals point up
    faces = cells.reshape(-1, 3)
    labels = np.zeros(len(faces), dtype=np.int64)
    if zones:
        centres = vertices[0][faces][:, :, :2].mean(axis=1)
        part = Layout(surface.extent, 0.0, sky, zones=zones).zone_at(centres)
        labels = np.where(part >= 0, 2 + part, 0)
    faces = [faces]
    labels = [labels]

    # The sky: a wall over each edge of the ground's rim, and a lid.
    rim = np.array(
        [node(k, 0) for k in range(nx + 1)]
        + [node(nx, k) for k in range(1, ny + 1)]
        + [node(k, ny) for k in range(nx - 1, -1, -1)]
        + [node(0, k) for k in range(ny - 1, 0, -1)]
    )
    top = len(vertices[0]) + np.arange(len(rim))
    vertices.append(np.column_stack([nodes[rim], np.full(len(rim), sky)]))
    after = np.roll(np.arange(len(rim)), -1)
    # The rim runs counter-clockwise seen from above, so a wall wound from one rim
    # point up and on to the next faces into the box.
    walls = np.concatenate(
        [
            np.stack([rim, top[after], rim[after]], axis=1),
            np.stack([rim, top, top[after]], axis=1),
        ]
    )
    faces.append(walls)
    labels.append(np.ones(len(walls), dtype=np.int64))
    corners = top[[0, nx, nx + ny, 2 * nx + ny]]
    lid = np.array([[corners[0], corners[2], corners[1]], [corners[0], corners[3], corners[2]]])
    faces.append(lid)
    labels.append(np.ones(2, dtype=np.int64))
    names = [(GROUND,), (SKY,), *[(name, "zones", GROUND) for name, _ in zones]]
    return RoomMesh(np.vstack(vertices), np.vstack(faces), np.concatenate(labels), names)


def building_on_terrain(name: str, footprint, height: float, surface: TerrainSurface) -> RoomMesh:
    """A building standing on a terrain: its ``footprint`` (n, 2) raised to a flat roof
    ``height`` over the terrain's mean height at its corners, each wall's foot following
    the surface. Surfaces as :func:`building_mesh` names them; faces point out."""
    footprint = np.asarray(footprint, dtype=float)
    problem = _footprint_problem(name, footprint)
    if problem is not None:
        raise ValueError(problem)
    roof = float(surface.height_at(footprint).mean()) + height
    n = len(footprint)
    outward = 1.0 if _signed_area(footprint) > 0.0 else -1.0
    vertices: list[np.ndarray] = []
    faces, face_labels = [], []
    labels = [(f"{name}_roof", name, "roofs", "buildings")]
    base = 0
    top = np.column_stack([footprint, np.full(n, roof)])
    for a, b, c in triangulate_polygon(top):
        faces.append(_facing(top, (a, b, c), (0.0, 0.0, 1.0)))
        face_labels.append(0)
    vertices.append(top)
    base += n
    for k in range(n):
        p, q = footprint[k], footprint[(k + 1) % n]
        foot = surface.profile(p, q)
        if foot[:, 2].max() > roof - ON_GROUND_M:
            raise ValueError(
                f"building {name!r}: the terrain under it rises to z = {foot[:, 2].max():g} m, "
                f"above its roof at z = {roof:g} m (its height over the terrain's mean "
                "height at its corners); raise it"
            )
        polygon = np.vstack([foot, [[*q, roof], [*p, roof]]])
        along = q - p
        normal = (outward * along[1], -outward * along[0], 0.0)
        labels.append((f"{name}_wall_{k + 1}", f"{name}_walls", name, "walls", "buildings"))
        for a, b, c in triangulate_polygon(polygon):
            faces.append(_facing(polygon, (base + a, base + b, base + c), normal, offset=base))
            face_labels.append(len(labels) - 1)
        vertices.append(polygon)
        base += len(polygon)
    return RoomMesh(np.vstack(vertices), np.array(faces), np.array(face_labels), labels)


def barrier_on_terrain(name: str, path, height: float, surface: TerrainSurface) -> RoomMesh:
    """A thin screen standing on a terrain along ``path`` (n, 2), ``height`` over it
    all along, its foot following the surface."""
    path = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 1e-9 * max(float(np.ptp(path, axis=0).max()), 1.0)):
        k = int(np.argmax(lengths <= 1e-9 * max(float(np.ptp(path, axis=0).max()), 1.0)))
        raise ValueError(f"barrier {name!r}: points {k + 1} and {k + 2} are in one place")
    foot = [surface.profile(path[0], path[1])]
    for a, b in zip(path[1:-1], path[2:], strict=True):
        foot.append(surface.profile(a, b)[1:])
    bottom = np.vstack(foot)
    n = len(bottom)
    top = bottom + [0.0, 0.0, height]
    faces = []
    for i in range(n - 1):
        faces += [(i, i + 1, n + i + 1), (i, n + i + 1, n + i)]
    return RoomMesh(
        np.vstack([bottom, top]), np.array(faces), np.zeros(len(faces)), [(name, "barriers")]
    )


def _signed_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _merged(values: np.ndarray, tolerance: float) -> np.ndarray:
    """The sorted values, those within ``tolerance`` of the one before them left out."""
    values = np.sort(np.asarray(values, dtype=float))
    keep = np.concatenate([[True], np.diff(values) > tolerance])
    return values[keep]


def _zip_chains(first, second, first_at, second_at) -> list[tuple[int, int, int]]:
    """Triangles between two chains of vertices on two parallel lines, each in order of
    ``first_at`` and ``second_at`` (their positions along the lines)."""
    out = []
    i = j = 0
    while i < len(first) - 1 or j < len(second) - 1:
        if j == len(second) - 1 or (i < len(first) - 1 and first_at[i + 1] <= second_at[j + 1]):
            out.append((first[i], second[j], first[i + 1]))
            i += 1
        else:
            out.append((first[i], second[j], second[j + 1]))
            j += 1
    return out


def _facing(vertices, triangle, towards, offset: int = 0) -> tuple[int, int, int]:
    """``triangle`` wound so that its normal points ``towards``; its indices are into
    ``vertices`` from ``offset`` on."""
    a, b, c = (np.asarray(vertices[k - offset]) for k in triangle)
    if np.cross(b - a, c - a) @ np.asarray(towards) < 0.0:
        return (triangle[0], triangle[2], triangle[1])
    return tuple(triangle)


def zone_polygons(geometry, extent, scale: float = 1.0) -> tuple[tuple[str, np.ndarray], ...]:
    """The ground zones of ``geometry`` (a scene's :class:`Geometry`), in metres, cut to
    ``extent``; zones left with no area are left out. Raises for zones that overlap or
    whose edges cross."""
    names = geometry.ground.zone_names()
    out = []
    for name, zone in zip(names, geometry.ground.zones or [], strict=True):
        polygon = scale * np.asarray(zone.polygon, dtype=float)
        problem = plan_problem(polygon)
        if problem is not None:
            words = problem.replace("a floor plan", "a zone").replace("the floor plan", "the zone")
            raise ValueError(f"ground zone {name!r}: {words}")
        clipped = clip_polygon(polygon, extent)
        if clipped is not None:
            out.append((name, clipped))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if _overlap(out[i][1], out[j][1]):
                raise ValueError(f"ground zones {out[i][0]!r} and {out[j][0]!r} overlap")
    return tuple(out)


def clip_polygon(polygon: np.ndarray, extent) -> np.ndarray | None:
    """The part of ``polygon`` (n, 2) inside the rectangle ``extent`` (x_min, y_min,
    x_max, y_max), or None when nothing with an area is left (Sutherland-Hodgman)."""
    x0, y0, x1, y1 = extent
    points = [tuple(p) for p in np.asarray(polygon, dtype=float)]
    for axis, bound, keep_above in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        inside = [(p[axis] >= bound) if keep_above else (p[axis] <= bound) for p in points]
        out = []
        for k, p in enumerate(points):
            q, q_in = points[k - 1], inside[k - 1]
            if inside[k] != q_in:
                t = (bound - q[axis]) / (p[axis] - q[axis])
                cut = (q[0] + t * (p[0] - q[0]), q[1] + t * (p[1] - q[1]))
                out.append(tuple(bound if a == axis else cut[a] for a in (0, 1)))
            if inside[k]:
                out.append(p)
        points = out
        if not points:
            return None
    clipped = np.array(points)
    tolerance = 1e-9 * max(x1 - x0, y1 - y0)
    keep = np.linalg.norm(clipped - np.roll(clipped, 1, axis=0), axis=1) > tolerance
    clipped = clipped[keep]
    if len(clipped) < 3 or abs(_signed_area(clipped)) <= tolerance * max(x1 - x0, y1 - y0):
        return None
    return clipped


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two simple polygons overlap: their edges cross, or a point inside one of
    them lies inside the other. Polygons that only touch do not."""
    for p, q in zip(a, np.roll(a, -1, axis=0), strict=True):
        for r, t in zip(b, np.roll(b, -1, axis=0), strict=True):
            if _cross(p, q, r, t):
                return True
    # One inside the other, or the same: a point just inside each tested in the other.
    return bool(in_polygon(_inner_point(a), b)[0] or in_polygon(_inner_point(b), a)[0])


def _cross(p, q, r, t) -> bool:
    """Whether segments pq and rt cross at a point inside both (not at an end)."""

    def side(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    d1, d2, d3, d4 = side(r, t, p), side(r, t, q), side(p, q, r), side(p, q, t)
    scale = 1e-12 * max(np.ptp(np.array([p, q, r, t]), axis=0).max(), 1.0) ** 2
    return d1 * d2 < -scale and d3 * d4 < -scale


def _inner_point(polygon: np.ndarray) -> np.ndarray:
    """(1, 2) a point inside the simple ``polygon``: the middle of one of its triangles."""
    flat = np.column_stack([polygon, np.zeros(len(polygon))])
    a, b, c = triangulate_polygon(flat)[0]
    return ((polygon[a] + polygon[b] + polygon[c]) / 3.0)[None]


def outdoor_mesh(geometry, objects: RoomMesh | None = None, scale: float = 1.0):
    """The mesh of an outdoor scene (``geometry``, a scene's :class:`Geometry` with a
    ``ground``) and its :class:`Layout`. ``objects`` is the model file's mesh, in
    metres; ``scale`` the geometry's units in metres."""
    spec = geometry.ground
    ground = scale * spec.height
    grid = spec.height_grid()
    if grid is not None:
        return _terrain_mesh(geometry, objects, scale, grid.scaled(scale, ground))
    parts: list[tuple[str, RoomMesh | None]] = []
    buildings = []
    for building in geometry.all_buildings():
        name = building.name
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
    plants = vegetation_shapes(geometry, scale) if geometry.vegetation else ()
    parts += [(f"{p.kind} {p.name!r}", None) for p in plants]
    points += [p.corners() for p in plants]
    top = max((float(p[:, 2].max()) for p in points), default=ground)
    if spec.extent is not None:
        (x0, y0), (x1, y1) = (scale * np.asarray(corner, dtype=float) for corner in spec.extent)
        for (what, mesh), corners in zip(parts, points, strict=True):
            lo, hi = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
            # Vegetation may run on past the edge (a wood the ground cuts off).
            if mesh is None:
                continue
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
    zones = zone_polygons(geometry, extent, scale)
    mesh = RoomMesh.concatenate(
        [ground_box(extent, ground, sky, zones), *[m for _, m in parts if m is not None]]
    )
    return mesh, Layout(extent, float(ground), float(sky), tuple(buildings), zones, plants)


def _terrain_mesh(geometry, objects: RoomMesh | None, scale: float, grid: HeightGrid):
    """:func:`outdoor_mesh` over a terrain (``grid``, in metres with the ground's
    height added): the ground is the grid over the extent, which lies inside the grid
    (given, or the whole grid)."""
    spec = geometry.ground
    tx0, ty0, tx1, ty1 = grid.extent
    if max(abs(v) for v in grid.extent) > FAR_M:
        raise ValueError(
            f"the terrain's coordinates reach {max(abs(v) for v in grid.extent) / 1000:.0f} "
            "km from the origin, too far for the geometry's precision: set ground.map_origin "
            "to a point of the map near the scene"
        )
    # What stands on the ground, seen from above, before anything is built.
    footprints = [
        (f"building {b.name!r}", scale * np.asarray(b.footprint, dtype=float))
        for b in geometry.all_buildings()
    ]
    footprints += [
        (f"barrier {name!r}", scale * np.asarray(barrier.path, dtype=float))
        for name, barrier in zip(geometry.barrier_names(), geometry.barriers or [], strict=True)
    ]
    if objects is not None:
        footprints.append(("the model file", objects.vertices[np.unique(objects.faces)][:, :2]))
    if spec.extent is not None:
        (x0, y0), (x1, y1) = (scale * np.asarray(corner, dtype=float) for corner in spec.extent)
        if x0 < tx0 or y0 < ty0 or x1 > tx1 or y1 > ty1:
            raise ValueError(
                f"the ground's extent reaches past the terrain, which runs from ({tx0:g}, "
                f"{ty0:g}) to ({tx1:g}, {ty1:g}) m"
            )
    else:
        x0, y0, x1, y1 = tx0, ty0, tx1, ty1
    for what, corners in footprints:
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        if lo[0] <= x0 or lo[1] <= y0 or hi[0] >= x1 or hi[1] >= y1:
            raise ValueError(
                f"{what} reaches past the edge of the ground (its extent, or the terrain's)"
            )
    extent = (float(x0), float(y0), float(x1), float(y1))
    surface = TerrainSurface.of(grid, extent)

    parts: list[tuple[str, RoomMesh | None]] = []
    buildings = []
    for building in geometry.all_buildings():
        name = building.name
        footprint = scale * np.asarray(building.footprint, dtype=float)
        mesh = building_on_terrain(name, footprint, scale * building.height, surface)
        parts.append((f"building {name!r}", mesh))
        buildings.append((name, footprint, float(mesh.vertices[:, 2].max())))
    for name, barrier in zip(geometry.barrier_names(), geometry.barriers or [], strict=True):
        path = scale * np.asarray(barrier.path, dtype=float)
        mesh = barrier_on_terrain(name, path, scale * barrier.height, surface)
        parts.append((f"barrier {name!r}", mesh))
    if objects is not None:
        if objects.vertices[np.unique(objects.faces)][:, 2].min() < surface.lowest - ON_GROUND_M:
            raise ValueError(
                f"the model file reaches below the terrain's lowest point (z = "
                f"{surface.lowest:g} m)"
            )
        parts.append(("the model file", objects))
    plants = vegetation_shapes(geometry, scale, surface) if geometry.vegetation else ()
    parts += [(f"{p.kind} {p.name!r}", None) for p in plants]
    points = [mesh.vertices[np.unique(mesh.faces)] for _, mesh in parts if mesh is not None]
    points += [p.corners() for p in plants]
    top = max([float(p[:, 2].max()) for p in points] + [surface.highest])
    ground = surface.lowest
    sky = surface.highest + scale * spec.sky if spec.sky is not None else top + scale * spec.margin
    if sky <= top:
        raise ValueError(
            f"the sky ({sky - surface.highest:g} m above the terrain's top) is below the top "
            f"of the scene ({top - surface.highest:g} m above it): raise geometry.ground.sky"
        )
    zones = zone_polygons(geometry, extent, scale)
    mesh = RoomMesh.concatenate(
        [terrain_box(surface, sky, zones), *[m for _, m in parts if m is not None]]
    )
    layout = Layout(
        extent, float(ground), float(sky), tuple(buildings), zones, plants, terrain=surface
    )
    return mesh, layout


def on_ground(points: np.ndarray, ground: float) -> np.ndarray:
    """Whether each point (..., 3) lies on the ground at height ``ground``."""
    return np.abs(np.asarray(points)[..., 2] - ground) <= ON_GROUND_M


__all__ = [
    "GROUND",
    "ON_GROUND_M",
    "SKY",
    "Layout",
    "TerrainSurface",
    "barrier_mesh",
    "barrier_on_terrain",
    "building_mesh",
    "building_on_terrain",
    "clip_polygon",
    "ground_box",
    "in_polygon",
    "on_ground",
    "outdoor_mesh",
    "standing",
    "terrain_box",
    "zone_polygons",
]
