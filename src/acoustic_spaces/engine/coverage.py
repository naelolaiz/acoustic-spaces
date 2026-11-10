"""Coverage maps: room acoustic parameters on a grid over the audience area.

The audience surfaces, picked by name, are covered with square columns ``spacing``
metres wide. Each column holds a map point at listening ``height`` above the surface
(two or more where surfaces lie above each other, such as a balcony over the stalls, or
where a wall between rooms crosses it), with a slab of air around it. For every source:

- the direct sound and the specular reflections up to ``simulation.early.max_order``
  are found exactly at each point, as at a listener;
- everything else comes from one stochastic ray tracing run in which every leg of
  every ray adds its track length through each slab it crosses
  (:func:`~acoustic_spaces.engine.late.stochastic_map_energy`), so a map costs about
  as much as one listener; with the statistical late method, the exponential decay of
  :func:`~acoustic_spaces.engine.late.statistical_energy`, as at a listener.

Each point's parameters follow from its energy per band and time slot as at a
listener (ISO 3382-1), together with the speech transmission index (IEC 60268-16) and
sound pressure levels from the sources' sound power.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import NamedTuple

import numpy as np

from acoustic_spaces import parameters as iso3382
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import late as late_stage
from acoustic_spaces.engine import run_size
from acoustic_spaces.engine.coherent import band_count, coherent_energy
from acoustic_spaces.engine.grazing import apply_grazing
from acoustic_spaces.engine.model import TraceModel, free_field_energy
from acoustic_spaces.engine.refracted import direct_times, refraction_for
from acoustic_spaces.engine.simulate import (
    _as_room,
    _candidates,
    _Edges,
    _max_partitions,
    _pick,
    _seed,
    _transition,
    check_late_method,
    response_length,
)
from acoustic_spaces.engine.turbulence import Eddies
from acoustic_spaces.geometry import points_inside
from acoustic_spaces.geometry.outdoor import GROUND, Layout, in_polygon
from acoustic_spaces.i18n import N_
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.materials.bands import to_octaves
from acoustic_spaces.parameters import A_WEIGHTING_DB, Talker, a_weighted
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Coverage, Scene
from acoustic_spaces.scene.model import Simulation

#: Time resolution of each point's energy response (s).
TIME_STEP = 1e-3
#: Sub-samples per column side for finding the surfaces under it.
_SAMPLES = 5
#: Least part of a column over the surfaces that gets a map point.
_MIN_COVER = 0.2
#: Surfaces under one column closer than this (m) belong to the same layer.
_LAYER_GAP = 1.5
#: Thickest slab of air around a map point (m).
_MAX_THICKNESS = 1.0
#: Lowest slab bottom above the surface (m).
_ABOVE_SURFACE = 0.05
#: Samples per column side for the part of a slab in the air of its point's room.
_AIR_SAMPLES = 24
#: Most of those samples looked at in one go.
_AIR_CHUNK = 2**18
#: The way the first face under a point is looked for: down, tilted a little so that it
#: runs along no wall.
_DOWN = np.array([0.0123, 0.0071, -1.0]) / np.linalg.norm([0.0123, 0.0071, -1.0])


class MapParameter(NamedTuple):
    """How a map parameter is shown: ``label`` with its unit, what it is, whether it
    has a value per octave band, and the ``scale`` and number ``format`` of the shown
    value (the stored value is in SI units: Ts in seconds shows in milliseconds). The
    label and description are marked for translation; show them through ``_()``."""

    label: str
    description: str
    per_band: bool
    scale: float
    format: str

    def text(self, value: float) -> str:
        return self.format.format(value * self.scale) if math.isfinite(value) else "–"


#: The parameters of every map point.
MAP_PARAMETERS: dict[str, MapParameter] = {
    "spl": MapParameter(N_("SPL (dB)"), N_("sound pressure level"), True, 1.0, "{:.1f}"),
    "spl_a": MapParameter(
        N_("SPL (dB(A))"), N_("A-weighted sound pressure level"), False, 1.0, "{:.1f}"
    ),
    "g": MapParameter(N_("G (dB)"), N_("strength"), True, 1.0, "{:.1f}"),
    "c50": MapParameter(N_("C50 (dB)"), N_("clarity, 50 ms"), True, 1.0, "{:.1f}"),
    "c80": MapParameter(N_("C80 (dB)"), N_("clarity, 80 ms"), True, 1.0, "{:.1f}"),
    "d50": MapParameter(N_("D50"), N_("definition"), True, 1.0, "{:.2f}"),
    "edt": MapParameter(N_("EDT (s)"), N_("early decay time"), True, 1.0, "{:.2f}"),
    "t20": MapParameter(
        N_("T20 (s)"), N_("reverberation time from a 20 dB decay"), True, 1.0, "{:.2f}"
    ),
    "t30": MapParameter(
        N_("T30 (s)"), N_("reverberation time from a 30 dB decay"), True, 1.0, "{:.2f}"
    ),
    "ts": MapParameter(N_("Ts (ms)"), N_("centre time"), True, 1e3, "{:.0f}"),
    "sti": MapParameter(N_("STI"), N_("speech transmission index"), False, 1.0, "{:.2f}"),
    "re_free_field": MapParameter(
        N_("Re free field (dB)"),
        N_("level relative to free field, outdoors: the excess attenuation with its sign turned"),
        True,
        1.0,
        "{:.1f}",
    ),
    "insertion_loss": MapParameter(
        N_("Insertion loss (dB)"),
        N_("insertion loss of the buildings, barriers or vegetation the map was also run without"),
        True,
        1.0,
        "{:.1f}",
    ),
    "insertion_loss_a": MapParameter(
        N_("Insertion loss (dB(A))"),
        N_("A-weighted insertion loss of the buildings, barriers or vegetation"),
        False,
        1.0,
        "{:.1f}",
    ),
    "long_term": MapParameter(
        N_("Long-term level (dB)"),
        N_("level over a long term, outdoors: the mean over the weather classes and the day"),
        True,
        1.0,
        "{:.1f}",
    ),
    "rating_level": MapParameter(
        N_("Rating level (dB(A))"),
        N_("level over a long term with each period's penalty, A-weighted (L_den)"),
        False,
        1.0,
        "{:.1f}",
    ),
    "long_term_insertion_loss": MapParameter(
        N_("Long-term insertion loss (dB)"),
        N_(
            "insertion loss over a long term: the long-term level without the buildings, "
            "barriers or vegetation less that with them"
        ),
        True,
        1.0,
        "{:.1f}",
    ),
    "long_term_insertion_loss_a": MapParameter(
        N_("Long-term insertion loss (dB(A))"),
        N_("A-weighted insertion loss over a long term of the buildings, barriers or vegetation"),
        False,
        1.0,
        "{:.1f}",
    ),
}
#: The map parameters of an outdoor scene alone, those of a map also run without some of
#: its buildings, barriers or vegetation (``simulation.insertion_loss_of``), those of a
#: map run in weather classes (``long_term``), and those of a map run both ways.
OUTDOOR_MAP_PARAMETERS = ("re_free_field",)
INSERTION_LOSS_MAP_PARAMETERS = ("insertion_loss", "insertion_loss_a")
LONG_TERM_MAP_PARAMETERS = ("long_term", "rating_level")
LONG_TERM_INSERTION_LOSS_MAP_PARAMETERS = ("long_term_insertion_loss", "long_term_insertion_loss_a")
#: The parameters of every map; an outdoor map, one also run without some of its
#: buildings, barriers or vegetation, and one run in weather classes have the others as
#: well.
ROOM_MAP_PARAMETERS = tuple(
    name
    for name in MAP_PARAMETERS
    if name
    not in OUTDOOR_MAP_PARAMETERS
    + INSERTION_LOSS_MAP_PARAMETERS
    + LONG_TERM_MAP_PARAMETERS
    + LONG_TERM_INSERTION_LOSS_MAP_PARAMETERS
)


@dataclass
class MapGrid:
    """Map points over the audience area: columns ``cell`` metres wide from the
    corner ``origin`` (x, y), each holding up to ``index.shape[2]`` points."""

    origin: tuple[float, float]
    cell: float
    #: (ny, nx, slots) the points of each column, by level and then by room, -1 for none.
    index: np.ndarray
    #: (n, 3) the points, at listening height above the column's centre, or above the
    #: middle of their part of the column when the centre is not in it.
    points: np.ndarray
    #: (n, 2) column (x, y) of each point.
    columns: np.ndarray
    #: (n,) height of the surface under each point (m).
    surface: np.ndarray
    #: (n,) bottom and top of each point's slab of air (m), and the volume of the slab
    #: in the air of the point's room, outside buildings (m^3).
    low: np.ndarray
    high: np.ndarray
    volume: np.ndarray
    #: (m, 3, 2) the surfaces under the map seen from above: x, y of their triangles.
    footprint: np.ndarray
    #: (n, 25) which fifths of the column, by x then y, lie over the surfaces (in the
    #: point's room, where the model holds several).
    cover: np.ndarray
    #: (n,) the room each point lies in (an index into ``Room.room_volumes``) where the
    #: model holds several, else -1. A point's slab counts the rays in its room alone.
    room: np.ndarray
    #: (n,) the level of each point in its column, from 0 at the lowest surface under it
    #: (the stalls; a balcony above them is 1). Where a wall between rooms crosses a
    #: column, it holds a point on each side that holds at least a fifth of it, on the
    #: same level.
    level: np.ndarray

    def __len__(self) -> int:
        return len(self.points)

    def squares(self) -> np.ndarray:
        """(n, 4, 3) corners of each point's column, at the point's height."""
        x0, y0 = self.origin
        x = x0 + self.columns[:, 0] * self.cell
        y = y0 + self.columns[:, 1] * self.cell
        d = self.cell
        z = self.points[:, 2]
        corners = [(x, y), (x + d, y), (x + d, y + d), (x, y + d)]
        return np.stack([np.column_stack([cx, cy, z]) for cx, cy in corners], axis=1)

    def tiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Squares at the points' height over the part of each column above the
        surfaces (in the point's room): (m, 4, 3) corners and the point each belongs to.
        A column only partly over them, or split between rooms, is cut into fifths along
        each side."""
        full = self.cover.all(axis=1)
        whole = self.squares()[full]
        owner = [np.flatnonzero(full)]
        parts = [whole]
        partial = np.flatnonzero(~full)
        if len(partial):
            k, sub = np.nonzero(self.cover[partial])
            k = partial[k]
            d = self.cell / _SAMPLES
            x = self.origin[0] + self.columns[k, 0] * self.cell + (sub // _SAMPLES) * d
            y = self.origin[1] + self.columns[k, 1] * self.cell + (sub % _SAMPLES) * d
            z = self.points[k, 2]
            corners = [(x, y), (x + d, y), (x + d, y + d), (x, y + d)]
            parts.append(np.stack([np.column_stack([cx, cy, z]) for cx, cy in corners], axis=1))
            owner.append(k)
        return np.concatenate(parts), np.concatenate(owner)


def audience_surfaces(room: Room) -> list[str]:
    """The surfaces a map covers when none are named: those whose names contain
    "audience", else "floor" (outdoors, "ground")."""
    names = room.mesh.label_names()
    audience = [n for n in names if "audience" in n.lower()]
    if audience:
        return audience
    if room.outdoor is not None:
        return [GROUND]
    if "floor" in names:
        return ["floor"]
    raise ValueError(
        "name the surfaces the coverage map lies over (coverage: surfaces); this model has: "
        + ", ".join(names)
    )


def _map_faces(room: Room, surfaces: Sequence[str] | None) -> np.ndarray:
    """(n, 3, 3) the triangles a map lies over: those of the named surfaces (default:
    :func:`audience_surfaces`) that can be walked on, not walls."""
    mesh = room.mesh
    names = list(surfaces) if surfaces else audience_surfaces(room)
    known = set(mesh.label_names())
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(
            f"no surface named {', '.join(missing)}; this model has: {', '.join(sorted(known))}"
        )
    wanted = np.array([any(n in label for n in names) for label in mesh.labels], dtype=bool)
    faces = np.flatnonzero(wanted[mesh.face_labels] & (np.abs(mesh.face_normals()[:, 2]) > 0.25))
    if len(faces) == 0:
        raise ValueError(f"{', '.join(names)}: no surface there faces up")
    return mesh.triangles[faces]


def _columns(tri: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    """The plan extent (low and high corner) of ``tri`` and the columns of a grid
    ``spacing`` apart over it, along x and y."""
    lo, hi = tri[:, :, :2].reshape(-1, 2).min(axis=0), tri[:, :, :2].reshape(-1, 2).max(axis=0)
    nx, ny = (np.maximum(1, np.ceil((hi - lo) / spacing - 1e-9)).astype(int)).tolist()
    return lo, hi, nx, ny


def map_size(
    room: Room,
    settings: Coverage,
    simulation: Simulation,
    sources: Sequence,
    threads: int,
) -> run_size.Need:
    """The memory a coverage map with ``settings`` needs (see
    :func:`~acoustic_spaces.engine.run_size.map_need`), before its grid is laid out:
    about a point for each ``coverage.spacing`` squared of the surfaces' plan area."""
    tri = _map_faces(room, settings.surfaces)
    lo, hi, _, _ = _columns(tri, settings.spacing)
    u, v = tri[:, 1, :2] - tri[:, 0, :2], tri[:, 2, :2] - tri[:, 0, :2]
    area = 0.5 * float(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum())
    points = max(1, math.ceil(area / settings.spacing**2))
    top = float(tri[:, :, 2].max()) + settings.height
    corners = np.array(
        [[lo[0], lo[1], top], [hi[0], lo[1], top], [lo[0], hi[1], top], [hi[0], hi[1], top]]
    )
    # Each point's sub-samples of the surfaces, three values each, while it is laid out.
    grid = float(points * _SAMPLES**2 * 3 * run_size.VALUE_BYTES)
    return run_size.map_need(room, simulation, sources, points, corners, grid, threads)


def map_grid(
    room: Room,
    surfaces: Sequence[str] | None = None,
    *,
    height: float = 1.2,
    spacing: float = 1.0,
) -> MapGrid:
    """Map points ``spacing`` metres apart, ``height`` metres above the named surfaces
    (default: :func:`audience_surfaces`). Only faces that can be walked on count, not
    walls, and only points inside the room are kept (outdoors, in the open air: not
    inside a building). Where the model holds several rooms, a column that a wall
    between them crosses gets a point in each that holds at least a fifth of it."""
    if not (height > 0 and spacing > 0):
        raise ValueError("height and spacing must be positive")
    mesh = room.mesh
    tri = _map_faces(room, surfaces)
    lo, hi, nx, ny = _columns(tri, spacing)
    # Centre the grid on the surfaces.
    origin = lo - 0.5 * (np.array([nx, ny]) * spacing - (hi - lo))
    step = spacing / _SAMPLES
    thickness = min(spacing, _MAX_THICKNESS)
    closed = room.report.closed
    several = len(room.room_volumes) > 1
    air = _Air(room) if closed or several else None

    # Every sub-sample of every column with the height of each surface above or below it.
    hits_x, hits_y, hits_z = [], [], []
    for a, b, c in tri:
        t_lo = np.min([a[:2], b[:2], c[:2]], axis=0)
        t_hi = np.max([a[:2], b[:2], c[:2]], axis=0)
        g_lo = np.maximum(np.ceil((t_lo - origin) / step - 0.5), 0).astype(int)
        g_hi = np.minimum(
            np.floor((t_hi - origin) / step - 0.5), [nx * _SAMPLES - 1, ny * _SAMPLES - 1]
        ).astype(int)
        if np.any(g_hi < g_lo):
            continue
        gx, gy = np.meshgrid(
            np.arange(g_lo[0], g_hi[0] + 1), np.arange(g_lo[1], g_hi[1] + 1), indexing="ij"
        )
        gx, gy = gx.ravel(), gy.ravel()
        px = origin[0] + (gx + 0.5) * step
        py = origin[1] + (gy + 0.5) * step
        # Barycentric coordinates in the xy plane.
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        w_a = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / det
        w_b = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / det
        w_c = 1.0 - w_a - w_b
        inside = (w_a >= -1e-9) & (w_b >= -1e-9) & (w_c >= -1e-9)
        hits_x.append(gx[inside])
        hits_y.append(gy[inside])
        hits_z.append((w_a * a[2] + w_b * b[2] + w_c * c[2])[inside])
    if not hits_x:
        raise ValueError("the map surfaces are too small for the spacing; use a smaller one")
    sx, sy, sz = (np.concatenate(v) for v in (hits_x, hits_y, hits_z))

    # With several rooms, the room each sub-sample lies in, at the middle of a slab over it.
    where = np.full(len(sz), -1, dtype=np.int64)
    if several and air is not None:
        middle = sz + max(height, _ABOVE_SURFACE + thickness / 2)
        across = origin + (np.column_stack([sx, sy]) + 0.5) * step
        where = air.rooms(np.column_stack([across, middle]))

    centre = _SAMPLES // 2
    column = (sy // _SAMPLES) * nx + sx // _SAMPLES
    order = np.argsort(column, kind="stable")
    sx, sy, sz, where, column = sx[order], sy[order], sz[order], where[order], column[order]
    starts = np.flatnonzero(np.r_[True, column[1:] != column[:-1]])
    ends = np.r_[starts[1:], len(column)]

    # Per column, each level of surface under it (stalls, a balcony above them, ...),
    # and with several rooms each room's part of it: its height, the part of the column
    # over it, and where the point goes across: the column's centre, or the middle of
    # the part over the surface when the centre is not over it (columns along a slanted
    # wall or a wall between rooms). A slab whose top and underside are in one group is
    # walked on at its top.
    rows: list[tuple[int, int, int, int, float, float, float, float]] = []
    covers: list[np.ndarray] = []
    for s, e in zip(starts, ends, strict=True):
        ix, iy = int(sx[s] // _SAMPLES), int(sy[s] // _SAMPLES)
        sub_x, sub_y, z, w = sx[s:e] % _SAMPLES, sy[s:e] % _SAMPLES, sz[s:e], where[s:e]
        z_sorted = np.sort(z)
        breaks = np.flatnonzero(np.diff(z_sorted) > _LAYER_GAP)
        bottoms = z_sorted[np.r_[0, breaks + 1]]
        tops = z_sorted[np.r_[breaks, len(z_sorted) - 1]]
        for k, (bottom, top) in enumerate(zip(bottoms, tops, strict=True)):
            level_on = (z >= bottom) & (z <= top)
            # The parts in no room (in a wall) get no point.
            for r in np.unique(w[level_on]) if several else [-1]:
                if several and r < 0:
                    continue
                on = level_on & (w == r) if several else level_on
                key, z_on = sub_x[on] * _SAMPLES + sub_y[on], z[on]
                sub = np.unique(key)
                at_centre = on & (sub_x == centre) & (sub_y == centre)
                if at_centre.any():
                    px, py, height_at = centre + 0.5, centre + 0.5, float(z[at_centre].max())
                else:
                    px = float(np.mean(sub // _SAMPLES)) + 0.5
                    py = float(np.mean(sub % _SAMPLES)) + 0.5
                    height_at = float(np.mean([z_on[key == q].max() for q in sub]))
                if len(sub) < _MIN_COVER * _SAMPLES**2:
                    continue  # a sliver: too few rays would cross its slab
                x = origin[0] + (ix + px / _SAMPLES) * spacing
                y = origin[1] + (iy + py / _SAMPLES) * spacing
                rows.append((ix, iy, k, int(r), height_at, len(sub) / _SAMPLES**2, x, y))
                cover = np.zeros(_SAMPLES**2, dtype=bool)
                cover[sub] = True
                covers.append(cover)

    if not rows:
        raise ValueError("the map surfaces are too small for the spacing; use a smaller one")
    ix, iy, layer, in_room = (np.array([r[i] for r in rows], dtype=np.int64) for i in range(4))
    surface, fraction, px, py = (np.array([r[i] for r in rows]) for i in range(4, 8))
    points = np.column_stack([px, py, surface + height])
    low = np.maximum(points[:, 2] - thickness / 2, surface + _ABOVE_SURFACE)
    high = low + thickness

    keep = np.ones(len(points), dtype=bool)
    if closed:
        keep = np.asarray(points_inside(mesh, points), dtype=bool)
    if several and air is not None:
        keep &= air.rooms(points) == in_room  # not in the corner of the next room
    if room.outdoor is not None:
        keep &= np.array([b is None for b in room.outdoor.building_at(points)], dtype=bool)
    if not keep.any():
        raise ValueError("no map point lies inside the room; check height and surfaces")
    cover = np.array(covers)
    ix, iy, layer, in_room, surface, fraction, points, low, high, cover = (
        v[keep] for v in (ix, iy, layer, in_room, surface, fraction, points, low, high, cover)
    )
    if not several:
        in_room = np.full(len(points), -1, dtype=np.int64)
    columns = np.column_stack([ix, iy])
    volume = _slab_volumes(room, air, origin, spacing, columns, low, high, in_room, fraction)
    # Number the levels of each column from 0 after dropping points; each point takes
    # the first free slot of its column, by level and then by room.
    cell = iy * nx + ix
    pairs, which = np.unique(cell * (int(layer.max()) + 1) + layer, return_inverse=True)
    owner = pairs // (int(layer.max()) + 1)
    level = (np.arange(len(pairs)) - np.searchsorted(owner, owner))[which.reshape(-1)]
    slots = int(np.bincount(cell).max())
    index = np.full((ny, nx, slots), -1, dtype=np.int64)
    for k in range(len(points)):
        slot = int(np.flatnonzero(index[iy[k], ix[k]] < 0)[0])
        index[iy[k], ix[k], slot] = k
    return MapGrid(
        origin=(float(origin[0]), float(origin[1])),
        cell=float(spacing),
        index=index,
        points=points,
        columns=columns,
        surface=surface,
        low=low,
        high=high,
        volume=volume,
        footprint=np.ascontiguousarray(tri[:, :, :2]),
        cover=cover,
        room=in_room,
        level=level.astype(np.int64),
    )


def _in_room(room: Room, grid: MapGrid) -> MapGrid:
    """``grid`` with the rooms of its points and the air in their slabs those of
    ``room``: another run of the scene the grid was laid out in, such as one without
    some of its buildings, shares its points."""
    several = len(room.room_volumes) > 1
    air = _Air(room) if room.report.closed or several else None
    in_room = np.full(len(grid), -1, dtype=np.int64)
    if several and air is not None:
        in_room = air.rooms(grid.points)
    volume = _slab_volumes(
        room,
        air,
        np.asarray(grid.origin),
        grid.cell,
        grid.columns,
        grid.low,
        grid.high,
        in_room,
        grid.cover.mean(axis=1),
    )
    return dataclasses.replace(grid, room=in_room, volume=volume)


class _Air:
    """Which room points lie in: that of the side facing them of the first face under
    them, since whatever lies between a point and that face is in one room.
    :meth:`~acoustic_spaces.room.Room.room_of` finds the same from the solid angles of
    the faces, at a far greater cost per point."""

    def __init__(self, room: Room) -> None:
        from acoustic_spaces import _native

        mesh = room.mesh
        self.bvh = _native.Bvh(mesh.vertices, mesh.faces)
        # The side of each face that a ray going down meets: the back where its normal
        # points down.
        self.side = (mesh.face_normals() @ _DOWN > 0.0).astype(np.int64)
        self.face_room = room.face_room

    def rooms(self, points: np.ndarray) -> np.ndarray:
        """(n,) the room (an index into ``Room.room_volumes``) each point lies in, or -1
        for none: outside the model or in a solid."""
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        n = len(points)
        _, face = self.bvh.intersect(
            points, np.tile(_DOWN, (n, 1)), np.full(n, np.inf), np.full(n, -1, dtype=np.int64)
        )
        out = np.full(n, -1, dtype=np.int64)
        hit = face >= 0
        out[hit] = self.face_room[face[hit], self.side[face[hit]]]
        return out


def _slab_volumes(
    room: Room,
    air: _Air | None,
    origin: np.ndarray,
    spacing: float,
    columns: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    in_room: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    """(n,) the volume of each slab (between ``low`` and ``high`` over its column of
    ``columns``) that the rays reach (m^3), from samples across the column at the slab's
    middle: in the model's air (with ``in_room`` >= 0, in that room), outside every
    building. Past the edge of the surfaces a slab may go on over a stage or an aisle,
    or end at a wall; each edge of its part may be off by half a sample. In a model that
    is not closed, where the inside is unknown, the part of the column over the surfaces
    (``fraction``) stands for it; so it does where no sample is in the air of the
    point's room (a sliver of it at the slab's middle)."""
    g = (np.arange(_AIR_SAMPLES) + 0.5) / _AIR_SAMPLES
    gx, gy = (v.ravel() for v in np.meshgrid(g, g, indexing="ij"))
    corner = origin + columns * spacing
    middle = 0.5 * (low + high)
    share = np.ones(len(columns))
    count = max(1, _AIR_CHUNK // len(gx))
    for first in range(0, len(columns), count):
        part = slice(first, first + count)
        n = len(corner[part])
        samples = np.stack(
            [
                corner[part, 0, None] + gx * spacing,
                corner[part, 1, None] + gy * spacing,
                np.broadcast_to(middle[part, None], (n, len(gx))),
            ],
            axis=-1,
        ).reshape(-1, 3)
        ok = np.ones((n, len(gx)), dtype=bool)
        if air is not None:
            found = air.rooms(samples).reshape(n, -1)
            wanted = in_room[part, None]
            ok = np.where(wanted >= 0, found == wanted, found >= 0)
        if room.outdoor is not None:
            ok &= ~_in_buildings(room.outdoor, samples.reshape(n, -1, 3), corner[part], spacing)
        share[part] = ok.mean(axis=1)
    if not room.report.closed:
        share = np.minimum(share, fraction)
    share = np.where(share > 0.0, share, fraction)
    return spacing**2 * share * (high - low)


def _in_buildings(
    outdoor: Layout, samples: np.ndarray, corner: np.ndarray, spacing: float
) -> np.ndarray:
    """(n, s) whether each of ``samples`` (n, s, 3), s points across each of n columns
    ``spacing`` wide whose corners of least x and y are ``corner`` (n, 2), all at one
    height in a column, lies in a building, as :meth:`~Layout.building_at` finds. Each
    building is tested only on the columns that reach into the box round its footprint
    seen from above, and only below its roof."""
    out = np.zeros(samples.shape[:2], dtype=bool)
    for _name, footprint, roof in outdoor.buildings:
        reach = (corner < footprint.max(axis=0)) & (corner + spacing > footprint.min(axis=0))
        over = np.flatnonzero(reach.all(axis=1) & (samples[:, 0, 2] < roof))
        if len(over) == 0:
            continue
        xyz = samples[over].reshape(-1, 3)
        inside = in_polygon(xyz[:, :2], footprint) & (xyz[:, 2] > outdoor.ground_at(xyz))
        out[over] |= inside.reshape(len(over), -1)
    return out


#: What a map's spread gives of a parameter: its standard uncertainty, the ends of its
#: interval, and the value it is at least and at most (:meth:`CoverageMap.spread_of`).
SPREAD_PARTS = ("standard", "low", "high", "at_least", "at_most")


def uncertainty_text(value: float) -> str:
    """A standard uncertainty as a map shows it: to two significant digits, which are
    as many as it has (GUM, JCGM 100:2008, 7.2.6), and to four decimals at most."""
    if not math.isfinite(value):
        return "–"
    if value == 0.0:
        return "0"
    decimals = min(4, max(0, 1 - math.floor(math.log10(abs(value)))))
    return f"{value:.{decimals}f}"


@dataclass
class MapSpread:
    """The uncertainty of a map's parameters that the uncertainty of the materials'
    absorption leads to, by the law of propagation of uncertainty
    (:func:`~acoustic_spaces.engine.sensitivity.coverage_map`)."""

    #: Parameter -> its combined standard uncertainty, shaped as the map's values: the
    #: root of the sum of the squares of the materials' parts. NaN where the value is
    #: undefined, in the map or in a run of the spread.
    standard: dict[str, np.ndarray]
    #: Material -> parameter -> its part: half of the map with its absorption one
    #: standard uncertainty up less the map with it one down (negative where more
    #: absorption lowers the value).
    materials: dict[str, dict[str, np.ndarray]]
    #: The probability of the interval, and that of the one-sided bounds.
    coverage: float = 0.95
    one_sided: float = 0.95
    #: As a listener's spread has them (:class:`~acoustic_spaces.engine.uncertainty.
    #: Spread`): the built-in materials that took the scene's default uncertainty, and
    #: the bands in which a material absorbs everything, which no run moved.
    assumed: dict[str, str] = field(default_factory=dict)
    held: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class CoverageMap:
    """Parameters at every map point for one source, or for all of them together."""

    source: str
    grid: MapGrid
    #: Parameter name (see :data:`MAP_PARAMETERS`) -> (points, bands) or (points,).
    values: dict[str, np.ndarray]
    #: Their uncertainty from the materials' absorption, where the map was run with its
    #: spread (:func:`~acoustic_spaces.engine.sensitivity.coverage_map`).
    spread: MapSpread | None = None

    def parameters(self) -> list[str]:
        """The parameters the map has, in the order of :data:`MAP_PARAMETERS`: those of
        :data:`OUTDOOR_MAP_PARAMETERS` outdoors only, those of
        :data:`INSERTION_LOSS_MAP_PARAMETERS` when the map was also run without some of
        the scene's buildings, barriers or vegetation, those of
        :data:`LONG_TERM_MAP_PARAMETERS` when it was run in weather classes, and those of
        :data:`LONG_TERM_INSERTION_LOSS_MAP_PARAMETERS` when it was run both ways."""
        return [name for name in MAP_PARAMETERS if name in self.values]

    def value(self, name: str, band: int | None = None) -> np.ndarray:
        """One value per point: a parameter in one band (index into
        ``OCTAVE_BANDS_HZ``), or a parameter that has no bands."""
        if name not in self.values:
            raise ValueError(f"unknown parameter {name!r}; use one of {', '.join(self.values)}")
        return self._one_per_point(name, self.values[name], band)

    @staticmethod
    def _one_per_point(name: str, v: np.ndarray, band: int | None) -> np.ndarray:
        if v.ndim == 1:
            return v
        if band is None:
            raise ValueError(f"{name} is per octave band; give a band")
        return v[:, band]

    def spread_of(self, name: str, part: str = "standard", band: int | None = None) -> np.ndarray:
        """One value per point of the spread of parameter ``name`` (as :meth:`value`
        takes it): its ``standard`` uncertainty; the ``low`` or the ``high`` end of the
        interval that holds it with the spread's coverage probability; or the value it
        is ``at_least`` or ``at_most`` with the spread's one-sided probability. The
        interval and the bounds take the result as normal about the map's value (GUM,
        JCGM 100:2008, 6.3 and G.2): the value less and plus the normal quantile times
        the standard uncertainty (1.96 for 95 %; 1.645 for a one-sided 95 %), held to
        what the parameter can be (:data:`~acoustic_spaces.parameters.LIMITS`)."""
        if self.spread is None:
            raise ValueError("the map was run without its spread")
        if part not in SPREAD_PARTS:
            raise ValueError(f"unknown part {part!r}; use one of {', '.join(SPREAD_PARTS)}")
        if name not in self.spread.standard:
            raise ValueError(f"unknown parameter {name!r}; use one of {', '.join(self.values)}")
        u = self.spread.standard[name]
        if part == "standard":
            return self._one_per_point(name, u, band)
        if part in ("low", "high"):
            factor = NormalDist().inv_cdf(0.5 + 0.5 * self.spread.coverage)
        else:
            factor = NormalDist().inv_cdf(self.spread.one_sided)
        sign = -1.0 if part in ("low", "at_least") else 1.0
        least, most = iso3382.LIMITS.get(name, (-math.inf, math.inf))
        with np.errstate(invalid="ignore"):
            out = np.clip(self.values[name] + sign * factor * u, least, most)
        return self._one_per_point(name, out, band)

    def as_dict(self) -> dict:
        """Points and values as plain lists, None where undefined (JSON friendly); with
        its spread, ``uncertainty`` holds the probabilities, each parameter's
        ``standard`` uncertainty and each material's part of it (``materials``)."""

        def plain(a: np.ndarray):
            rounded = np.round(a.astype(float), 4)
            return np.where(np.isfinite(rounded), rounded, None).tolist()

        out = {
            "source": self.source,
            "bands_hz": list(OCTAVE_BANDS_HZ),
            "points": self.grid.points.round(3).tolist(),
            "values": {name: plain(v) for name, v in self.values.items()},
        }
        if self.spread is not None:
            out["uncertainty"] = {
                "coverage": self.spread.coverage,
                "one_sided": self.spread.one_sided,
                "standard": {name: plain(v) for name, v in self.spread.standard.items()},
                "materials": {
                    material: {name: plain(v) for name, v in parts.items()}
                    for material, parts in self.spread.materials.items()
                },
                "assumed": dict(self.spread.assumed),
                "held": {name: list(bands) for name, bands in self.spread.held.items()},
            }
        return out


@dataclass
class CoverageResult:
    """Coverage maps of every source, and of all of them together."""

    room: Room
    settings: Coverage
    grid: MapGrid
    maps: dict[str, CoverageMap]
    #: All sources playing together at their levels; None with one source.
    combined: CoverageMap | None = None
    details: dict[str, dict[str, float]] = field(default_factory=dict)
    #: The buildings, barriers and vegetation the map was also run without, for the
    #: insertion loss (:func:`~acoustic_spaces.engine.insertion.coverage_map`).
    removed: tuple[str, ...] = ()

    def parameters(self) -> list[str]:
        """The parameters of its maps (see :meth:`CoverageMap.parameters`)."""
        return self.get().parameters()

    def get(self, source: str | None = None) -> CoverageMap:
        """The map of one source; None gives all sources together (or the only one)."""
        if source is None:
            return self.combined if self.combined is not None else next(iter(self.maps.values()))
        if source not in self.maps:
            raise ValueError(f"unknown source {source!r}; use one of {', '.join(self.maps)}")
        return self.maps[source]

    def as_dict(self) -> dict:
        """The settings, the points and every map as plain lists (JSON friendly)."""
        maps = list(self.maps.values())
        if self.combined is not None:
            maps.append(self.combined)
        return {
            "scene": self.room.scene.name,
            "settings": self.settings.model_dump(mode="json"),
            "bands_hz": list(OCTAVE_BANDS_HZ),
            "points": self.grid.points.round(3).tolist(),
            "parameters": {name: MAP_PARAMETERS[name].label for name in self.parameters()},
            "maps": {
                m.source: {k: v for k, v in m.as_dict().items() if k not in ("points", "bands_hz")}
                for m in maps
            },
            "details": self.details,
            **({"removed": list(self.removed)} if self.removed else {}),
        }


def _unturned(paths: early_stage.EarlyReflections) -> early_stage.EarlyReflections:
    """``paths`` without the cosines of their reflections
    (:attr:`~acoustic_spaces.engine.early.EarlyReflections.cosines`): an array per path
    that only re-weighting for other absorption uses, and a map is never re-weighted."""
    return dataclasses.replace(paths, cosines=[]) if paths.cosines else paths


@dataclass
class _Part:
    """What one point source (all a source has, or one of a line's or surface's) brings
    to each point, as if it radiated all the source's power."""

    onset: np.ndarray  # (n,) arrival of its direct sound
    early: list[early_stage.EarlyReflections]  # per point
    late: late_stage.MapEnergy | None
    #: The statistical tail's decay at each point where the model holds several rooms.
    decay: tuple[np.ndarray, np.ndarray] | None = None
    #: For the statistical tail: the early paths without the audience's grazing
    #: attenuation, where it changed them (else None).
    plain: list[early_stage.EarlyReflections] | None = None


@dataclass
class _SourceRun:
    """Everything one source contributes at each point."""

    #: Sound power per octave band (dB re 1 pW), the feed's gain included.
    power_db: np.ndarray
    #: How long after emission the source plays (s), and the speech it carries.
    delay_s: float
    talker: Talker
    #: Its point sources, which share its power equally.
    parts: list[_Part]
    #: Statistical tail: (T60 per band, volume, speed of sound).
    statistical: tuple[np.ndarray, float, float] | None
    #: Outdoors: the energy per octave band at each point (points, bands) of a source of
    #: unit power there in free field; else None.
    free_field: np.ndarray | None = None

    @property
    def onset(self) -> np.ndarray:
        """(n,) arrival of the first direct sound at each point."""
        if len(self.parts) == 1:
            return self.parts[0].onset
        return np.min([part.onset for part in self.parts], axis=0)


def _curve(run: _SourceRun, k: int, length: float, bands: int) -> np.ndarray:
    """Energy (bands, slots of ``TIME_STEP``) at point k, slot 0 at its direct sound."""
    start = run.onset[k]
    slots = max(1, int(math.ceil((length - start) / TIME_STEP - 1e-9)))
    out = np.zeros((bands, slots))
    share = 1.0 / len(run.parts)
    for part in run.parts:
        curve = _part_curve(part, run.statistical, k, start, slots, length, bands)
        out += curve if len(run.parts) == 1 else share * curve
    return out


def _part_curve(
    part: _Part,
    statistical: tuple[np.ndarray, float, float] | None,
    k: int,
    start: float,
    slots: int,
    length: float,
    bands: int,
) -> np.ndarray:
    """One point source's energy at point k on slots from ``start`` (s, from emission)."""
    onset = part.onset[k]
    out = np.zeros((bands, slots))
    paths = part.early[k]
    delays, energy = paths.delay_s, paths.energy
    idx = np.floor((delays - start) / TIME_STEP + 1e-9).astype(np.int64)
    keep = (idx >= 0) & (idx < slots)
    coherent = paths.coherent_bands
    for b in range(coherent, bands):
        np.add.at(out[b], idx[keep], energy[keep, b])
    if coherent and len(paths):
        # The lowest bands in pressure, as at a listener; what the band filters spread
        # to before the direct sound counts with it.
        out[:coherent] = coherent_energy(paths, TIME_STEP, slots, origin=start)
    late = part.late
    if late is not None:
        # The traced energy's slots start at this source's own direct sound.
        offset = int(round((onset - start) / TIME_STEP))
        target = out[:, offset:]
        fine = min(late.fine_bins, target.shape[1])
        target[:, :fine] += late.energy[k, :, :fine]
        if target.shape[1] > late.fine_bins:  # a short response may end before the coarse slots
            ratio = int(round(late.coarse_step / TIME_STEP))
            coarse = late.energy[k, :, late.fine_bins :]
            spread = np.repeat(coarse / ratio, ratio, axis=1)
            end = min(target.shape[1], late.fine_bins + spread.shape[1])
            target[:, late.fine_bins : end] += spread[:, : end - late.fine_bins]
    if statistical is not None:
        t60, volume, c = statistical
        # As at a listener, on its slots from emission; their running sum carried over
        # to this point's slots.
        tail = late_stage.statistical_energy(
            t60,
            volume,
            [onset],
            [part.early[k] if part.plain is None else part.plain[k]],
            max_time=length,
            time_step=TIME_STEP,
            speed_of_sound=c,
            decay=None if part.decay is None else (part.decay[0][k : k + 1], part.decay[1]),
        ).energy[0]
        running = np.concatenate([np.zeros((bands, 1)), np.cumsum(tail, axis=1)], axis=1)
        times = np.arange(running.shape[1]) * TIME_STEP
        # Nothing arrives before the direct sound, in the slot that holds it either.
        times[min(int(onset / TIME_STEP), len(times) - 1)] = onset
        at = start + np.arange(slots + 1) * TIME_STEP
        out += np.diff([np.interp(at, times, row) for row in running], axis=1)
    return out


def _point_values(
    energy: np.ndarray,
    power_db: np.ndarray,
    reference: float,
    noise_db,
    talker: Talker,
    free_field: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """All map parameters of one point's energy response (slot 0 at its direct sound);
    ``power_db`` is the source's sound power per band, ``reference`` the free-field
    energy at 10 m of a unit source. Outdoors, ``free_field`` is the energy per octave
    band of the same source here in free field, for the level relative to it."""
    p = iso3382.from_energy(
        energy, TIME_STEP, 0.0, reference_energy=reference, open_air=free_field is not None
    )
    out = {name: getattr(p, name) for name in ("g", "c50", "c80", "d50", "edt", "t20", "t30", "ts")}
    spl = power_db + p.g + 10.0 * math.log10(reference)
    out["spl"] = spl
    out["spl_a"] = np.array(a_weighted(spl))
    # No STI without the level of each of its bands, 125 Hz to 8 kHz (silent, or a decay
    # too short for G).
    out["sti"] = np.array(
        iso3382.speech_transmission_index(
            energy, TIME_STEP, signal_db=spl, noise_db=noise_db, talker=talker
        )
        if np.isfinite(spl[1:]).all()
        else math.nan
    )
    if free_field is not None:
        # The energy the point holds over the free field's: nothing extrapolated past the
        # response's end, as G does for a room's decay.
        with np.errstate(divide="ignore", invalid="ignore"):
            out["re_free_field"] = 10.0 * np.log10(energy.sum(axis=1) / free_field)
    return out


def _trace_part(
    label: str,
    position: np.ndarray,
    *,
    room: Room,
    model: TraceModel,
    edges: _Edges,
    grid: MapGrid,
    sim: Simulation,
    name: str,
    directivity,
    orientation: np.ndarray,
    transition: float | np.ndarray,
    length: float,
    rays: int,
    seed: int,
    statistical: bool,
    threads: int,
    say: Callable[[str], None],
    info: dict[str, float],
    single: bool = True,
) -> _Part:
    """Early paths and traced late energy at every point of one point source, which
    radiates all the power; timings and counts add up in ``info``. Unless ``single``,
    one of the uncorrelated points of a line or surface, whose paths add in energy
    (indoors; outdoors in pressure among themselves)."""

    def add(key: str, value: float) -> None:
        info[key] = info.get(key, 0.0) + value

    c = room.speed_of_sound()
    air = room.air_attenuation(model.bands)
    onset = np.linalg.norm(grid.points - position, axis=1) / c
    cutoff = np.minimum(onset + transition, length)

    t = time.perf_counter()
    say(f"{label}: early reflections at {len(grid)} points")
    candidates = _candidates(
        model, position, sim.early.method, sim.early.max_order, sim.early.rays, threads
    )
    # Maps do not run the wave solver, so the lowest bands add in pressure up to
    # coherent_max_band_hz whatever simulation.wave says. Indoors a line's or a
    # surface's points add in energy; outdoors each point's own paths add in pressure,
    # for the ground dip.
    coherent = band_count(sim.early.coherent_max_band_hz, model.absorption.shape[1])
    coherent = coherent if single or room.outdoor is not None else 0
    diffraction = edges.diffraction(
        sim.early.diffraction,
        position,
        room,
        name,
        orientation,
        coherent,
        both_sides=sim.early.diffraction_reflections == "both_sides",
        threads=threads,
        candidates=candidates,
    )
    found, straight = [], []
    diffracted = 0
    for k in range(len(grid)):
        paths = early_stage.find_paths(
            model,
            position,
            grid.points[k],
            candidates,
            speed_of_sound=c,
            air=air,
            max_delay_s=float(cutoff[k]),
            directivity=directivity,
            orientation=orientation,
            coherent=coherent,
            max_partitions=_max_partitions(sim),
        )
        paths = _unturned(paths)
        straight.append(paths)
        if diffraction is not None:
            paths = diffraction.apply(grid.points[k], paths, float(cutoff[k]))
            diffracted += int(paths.diffracted.sum())
        found.append(_unturned(paths))
    refraction = refraction_for(
        model,
        position,
        speed_of_sound=c,
        air=air,
        directivity=directivity,
        orientation=orientation,
        coherent=coherent,
        max_order=sim.early.max_order,
        threads=threads,
        wide_angle=sim.early.parabolic_equation == "wide_angle",
        barrier_frequencies=sim.early.barrier_frequencies,
        edges=None if diffraction is None else diffraction.edges,
    )
    if refraction is not None:
        say(f"{label}: paths bent by the wind and the temperature")
        found = [_unturned(p) for p in refraction.apply(grid.points, straight, found, cutoff)]
        # The response starts when the bent direct sound arrives.
        onset = direct_times(onset, found)
    if model.plants is not None:
        found = [model.plants.attenuate(paths) for paths in found]
    eddies = Eddies.of(model, c)
    if eddies is not None:
        found = [eddies.attach(paths) for paths in found]
    plain = found
    early = [apply_grazing(model, grid.points[k], paths) for k, paths in enumerate(found)]
    if diffraction is not None:
        info["edges"] = float(len(diffraction.edges))
        add("diffracted_paths", float(diffracted))
    add("early_s", time.perf_counter() - t)

    t = time.perf_counter()
    late = None
    if not statistical:
        say(f"{label}: stochastic ray tracing over the map ({rays} rays per band)")
        late = late_stage.stochastic_map_energy(
            model,
            position,
            grid,
            onset,
            rays=rays,
            max_time=length,
            speed_of_sound=c,
            air=air,
            covered_order=sim.early.max_order,
            covered_time=cutoff,
            splitting=sim.late.splitting,
            seed=seed,
            directivity=directivity,
            orientation=orientation,
            threads=threads,
        )
        add("rays", float(late.rays))
        add("traced_rays", float(late.traced_rays))
    decay = None
    if statistical:
        decay = late_stage.diffuse_decay(model, position, grid.points, air, c)
    add("late_s", time.perf_counter() - t)
    # As at a listener, the statistical tail is reckoned less the paths without the
    # audience's attenuation, so what the seats take is lost.
    changed = statistical and any(a is not b for a, b in zip(early, plain, strict=True))
    return _Part(onset, early, late, decay, plain if changed else None)


def coverage_map(
    room: Room | Scene | str | Path,
    *,
    settings: Coverage | None = None,
    simulation: Simulation | None = None,
    sources: Sequence[str] | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
    grid: MapGrid | None = None,
) -> CoverageResult:
    """Coverage maps of the scene's sources (all by default) over its audience area.

    ``settings`` default to ``scene.coverage`` (surfaces, height, spacing, rays and
    background noise); ``simulation`` (default: ``scene.simulation``) gives the early
    order, late method, response length, seed and threads (``threads`` overrides
    them) as for :func:`~acoustic_spaces.engine.simulate`. Each source radiates its sound power per
    band (:meth:`~acoustic_spaces.scene.Source.power_db`: its level, spectrum, gain and
    equalizer), which the speech transmission index takes as the speech level, weighted
    for female speech when the source's spectrum is ``speech_female``. In the map of
    all sources together, each source's sound arrives ``delay_ms`` later. Outdoors,
    every point also has its level relative to free field (``re_free_field``).

    Maps do not run the wave solver: whatever ``simulation.wave`` says, the lowest bands
    come from the early paths, in pressure up to ``early.coherent_max_band_hz``, and
    the rays.

    ``grid`` takes the place of the one the settings give, so that two runs of a scene
    share their points (:func:`~acoustic_spaces.engine.insertion.coverage_map`); the
    rooms its points lie in and the air in their slabs are those of ``room``.
    """
    room = _as_room(room)
    # Listeners play no part in a map; sources named are mapped even when left out.
    problems = room.problems(
        in_use=("sources",) if sources is None else (), named={"sources": sources or ()}
    )
    problems = [p for p in problems if not p.startswith("listener ")]
    if problems:
        raise ValueError("; ".join(problems))
    scene = room.scene
    settings = settings or scene.coverage or Coverage()
    sim = simulation or scene.simulation
    if threads is None:
        threads = sim.performance.threads
    say = progress or (lambda _: None)
    chosen = _pick(scene.sources, sources, "source")

    started = time.perf_counter()
    if grid is None:
        run_size.check(map_size(room, settings, sim, chosen, threads))
        grid = map_grid(room, settings.surfaces, height=settings.height, spacing=settings.spacing)
    else:
        grid = _in_room(room, grid)
    say(f"Map: {len(grid)} points, {grid.cell:g} m apart")
    model = TraceModel.from_room(room, sim)
    edges = _Edges(model)
    c = room.speed_of_sound()
    volume = room.volume_m3
    length = response_length(
        room,
        sim.ir_length_s,
        [point for source in chosen for point in source.elements()],
        grid.points,
    )
    bands = model.absorption.shape[1]
    seed = sim.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, np.uint32)[0])
    statistical = sim.late.method == "statistical"
    check_late_method(room, sim)
    transition = _transition(sim, room, grid.points)
    rays = settings.rays or sim.late.rays
    reference = 1.0 / (4.0 * math.pi * 100.0)
    details: dict[str, dict[str, float]] = {"map": {"points": float(len(grid))}}
    if length != sim.ir_length_s:
        details["map"]["ir_length_s"] = length

    runs: dict[str, _SourceRun] = {}
    names = {source.name for source in chosen}
    for s_index, source in enumerate(scene.sources):
        if source.name not in names:
            continue
        info: dict[str, float] = {"seed": _seed(seed, s_index)}
        directivity = room.directivity(source.name, model.bands)
        orientation = np.array(source.orientation.rotation())
        elements = source.elements()
        if len(elements) > 1:
            info["elements"] = float(len(elements))
        parts = []
        for e_index, position in enumerate(elements):
            # One point source of a line or surface: its share of the rays, a seed of its own.
            single = len(elements) == 1
            parts.append(
                _trace_part(
                    source.name if single else f"{source.name} ({e_index + 1})",
                    position,
                    room=room,
                    model=model,
                    edges=edges,
                    grid=grid,
                    sim=sim,
                    name=source.name,
                    directivity=directivity,
                    orientation=orientation,
                    transition=transition,
                    length=length,
                    rays=math.ceil(rays / len(elements)),
                    seed=info["seed"] if single else _seed(info["seed"], e_index + 1),
                    statistical=statistical,
                    threads=threads,
                    say=say,
                    info=info,
                    single=single,
                )
            )
            if not runs and e_index == 0:
                # The early paths at every point grow with the points too: as many for
                # each point source as the first's took.
                need = run_size.map_need(room, sim, chosen, len(grid), grid.points, 0.0, threads)
                early = run_size.held_bytes((parts[0].early, parts[0].plain))
                total = sum(len(s.elements()) for s in chosen)
                run_size.check(dataclasses.replace(need, bytes=need.bytes + total * early))
        stat = None
        if statistical:
            t60 = room.reverberation_time("eyring", air_absorption=True, bands=model.bands)
            stat = (np.asarray(t60, dtype=float), float(volume), c)
        details[source.name] = info
        free_field = None
        if room.outdoor is not None:
            free_field = to_octaves(
                free_field_energy(
                    elements,
                    grid.points,
                    room.air_attenuation(model.bands),
                    directivity,
                    orientation,
                ),
                axis=1,
            )
        runs[source.name] = _SourceRun(
            source.power_db("octave"),
            source.delay_ms * 1e-3,
            source.talker,
            parts,
            stat,
            free_field,
        )

    say("Map: parameters at every point")
    noise = settings.noise_db
    octaves = len(OCTAVE_BANDS_HZ)  # parameters in octave bands, whatever was traced
    per_source: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in runs}
    combined: list[dict[str, np.ndarray]] = []
    # In each band, the sources together at their levels relative to the loudest.
    loudest = np.max([run.power_db for run in runs.values()], axis=0)
    talkers = {run.talker for run in runs.values()}
    talker = talkers.pop() if len(talkers) == 1 else "male"
    for k in range(len(grid)):
        curves = {}
        for name, run in runs.items():
            curve = to_octaves(_curve(run, k, length, bands))
            curves[name] = curve
            free_field = None if run.free_field is None else run.free_field[k]
            values = _point_values(curve, run.power_db, reference, noise, run.talker, free_field)
            per_source[name].append(values)
        if len(runs) > 1:
            # All sources together, on the time axis of the earliest direct sound; each
            # source's comes its delay later.
            first = min(run.onset[k] + run.delay_s for run in runs.values())
            last = max(run.delay_s for run in runs.values())
            slots = int(math.ceil((length + last - first) / TIME_STEP - 1e-9))
            total = np.zeros((octaves, slots))
            free_field = None if room.outdoor is None else np.zeros(octaves)
            for name, run in runs.items():
                shift = int(round((run.onset[k] + run.delay_s - first) / TIME_STEP))
                curve = curves[name][:, : max(0, slots - shift)]
                weight = 10.0 ** ((run.power_db - loudest) / 10.0)
                total[:, shift : shift + curve.shape[1]] += weight[:, None] * curve
                if free_field is not None:
                    free_field += weight * run.free_field[k]
            combined.append(_point_values(total, loudest, reference, noise, talker, free_field))

    def stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            name: np.array([r[name] for r in rows]) for name in MAP_PARAMETERS if name in rows[0]
        }

    maps = {name: CoverageMap(name, grid, stack(rows)) for name, rows in per_source.items()}
    together = CoverageMap("all sources", grid, stack(combined)) if combined else None
    details["map"]["seconds"] = time.perf_counter() - started
    return CoverageResult(room, settings, grid, maps, together, details)


__all__ = [
    "A_WEIGHTING_DB",
    "INSERTION_LOSS_MAP_PARAMETERS",
    "LONG_TERM_INSERTION_LOSS_MAP_PARAMETERS",
    "LONG_TERM_MAP_PARAMETERS",
    "MAP_PARAMETERS",
    "OUTDOOR_MAP_PARAMETERS",
    "ROOM_MAP_PARAMETERS",
    "SPREAD_PARTS",
    "TIME_STEP",
    "CoverageMap",
    "CoverageResult",
    "MapGrid",
    "MapParameter",
    "MapSpread",
    "audience_surfaces",
    "coverage_map",
    "map_grid",
    "uncertainty_text",
]
