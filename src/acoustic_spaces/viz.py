"""Pictures of rooms and responses.

- 3D view (PyVista): surfaces coloured by material, sources with their directivity
  balloons, listeners, and early reflection paths coloured by order (paths diffracted
  at an edge in a colour of their own); outdoors the vegetation, and the wind: arrows
  over the ground, the weather zones, and masts with the wind's profile up them.
- Leaks (PyVista): the open edges of a model that is not closed, and the rays of the
  leak finder that escape through its gaps.
- Animation (PyVista): sound particles spreading from a source and reflecting around
  the room, in a window with a time slider or saved as a GIF or MP4 video; a walker
  going along its walk, as a video to go with its sound.
- Response plots (matplotlib): echogram with the early paths, decay curves per band, and
  arrival directions as the listener faces; the decays of a simulated and a measured
  response band by band; polar patterns of a source directivity; the wind, temperature
  and speed of sound over the ground outdoors.
- Coverage maps: a parameter over the audience area, from above (matplotlib) or as
  coloured tiles in the 3D view.

Needs the viz extra (a release wheel installed with ``[viz]`` after its file name, or
``uv sync --extra viz`` in a source checkout). The 3D view works in a desktop window, in
Jupyter, and off screen for screenshots.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from acoustic_spaces._extras import install_hint
from acoustic_spaces.geometry import UNIT_SCALE, plan_problem
from acoustic_spaces.geometry.outdoor import _inner_point, clip_polygon, in_polygon
from acoustic_spaces.i18n import N_, _
from acoustic_spaces.room import UNASSIGNED, Room

if TYPE_CHECKING:
    import pyvista

    from acoustic_spaces.directivity import Directivity
    from acoustic_spaces.engine import CoverageMap, EarlyReflections, Response
    from acoustic_spaces.engine.leaks import Leaks
    from acoustic_spaces.engine.particles import Particles
    from acoustic_spaces.parameters import BandDecay
    from acoustic_spaces.scene.model import Atmosphere
    from acoustic_spaces.walkthrough import WalkResponses

#: ColorBrewer Set3: light colours, so the saturated source and listener markers stand out.
MATERIAL_COLOURS = (
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
)
UNASSIGNED_COLOUR = "#ff00ff"
SOURCE_COLOUR = "#ff5a5f"
LISTENER_COLOUR = "#4da3ff"
FACING_COLOUR = "#ffd23f"
BACKGROUND = ("#16171d", "#3a3d4a")  # bottom, top
PANEL = "#26272f"
EDGE_COLOUR = "#d0d0d0"
#: Reflection paths by order, direct sound first; orders past the end reuse the last.
ORDER_COLOURS = ("#ffffff", "#ffd166", "#f78c6b", "#ef476f", "#b5179e", "#7209b7", "#4361ee")
#: Paths that bend over an edge.
DIFFRACTED_COLOUR = "#06d6a0"
#: Octave band colours for decay curves, low to high.
BAND_COLOURS = (
    "#5e4fa2",
    "#3288bd",
    "#66c2a5",
    "#abdda4",
    "#e6f598",
    "#fdae61",
    "#f46d43",
    "#d53e4f",
)
#: Open edges and the rays that escape through them.
LEAK_COLOUR = "#ff4d4d"
ESCAPE_COLOUR = "#ffd166"
#: Escaping rays drawn at most, so a big hole does not hide the room.
MAX_ESCAPES_DRAWN = 400
#: A simulated and a measured response drawn together.
COMPARISON_COLOURS = ("#4da3ff", "#ffd166")
#: Trees, hedges and belts of trees outdoors.
VEGETATION_COLOUR = "#5cb85c"
#: The wind and the weather zones outdoors.
WIND_COLOUR = "#4cc9f0"
#: Arrows of the wind over the ground in the 3D view, about this many.
WIND_ARROWS = 40
#: The longest arrow of the wind, the strongest's, as a share of the space between them.
WIND_REACH = 0.8
#: Arrows up each mast of the wind's profile.
PROFILE_ARROWS = 5
# Translators: the wind over an outdoor scene, as the 3D view's legend shows it.
WIND_KEY = N_("wind {speed:.1f} m/s at {height:g} m")
# Translators: a weather zone's name and its wind's speed, drawn on it.
ZONE_WIND = N_("{name}: {speed:.1f} m/s")
# Translators: a weather zone's name, its wind's speed and its ground temperature.
ZONE_WIND_TEMPERATURE = N_("{name}: {speed:.1f} m/s, {temperature:.1f} °C")
# Translators: a weather zone's name and its ground temperature, in still air.
ZONE_TEMPERATURE = N_("{name}: {temperature:.1f} °C")
# Translators: the wind's speed at a height, beside an arrow up the wind's profile and
# beside the wind and each weather zone in the scene tree.
PROFILE_SPEED = N_("{speed:.1f} m/s at {height:g} m")


def _outline(part: pyvista.PolyData) -> pyvista.PolyData:
    """Edges of a surface patch without the triangulation's internal diagonals."""
    return part.extract_feature_edges(
        feature_angle=20.0,
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=True,
    )


def _pyvista():
    try:
        import pyvista
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(f"the 3D view needs PyVista: {install_hint('viz')}") from exc
    return pyvista


def room_polydata(room: Room, faces: np.ndarray | None = None) -> pyvista.PolyData:
    """Room surfaces (all, or the given face indices) as PyVista PolyData.

    The ``material`` cell array holds the index into ``room.material_names``.
    """
    pv = _pyvista()
    mesh = room.mesh
    faces = np.arange(len(mesh.faces)) if faces is None else np.asarray(faces)
    cells = np.column_stack([np.full(len(faces), 3), mesh.faces[faces]]).ravel()
    poly = pv.PolyData(mesh.vertices, cells)
    poly.cell_data["material"] = room.face_materials[faces]
    return poly


#: A reflection order's name by the suffix English gives its number.
_ORDER_NAMES = {
    # Translators: reflection orders: 1st order, 2nd order... {order} is the number.
    "st": N_("{order}st order"),
    "nd": N_("{order}nd order"),
    "rd": N_("{order}rd order"),
    "th": N_("{order}th order"),
}


def _order_name(order: int) -> str:
    if order == 0:
        # Translators: the direct sound's paths, in a plot's legend.
        return _("direct")
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(order if order < 20 else order % 10, "th")
    return _(_ORDER_NAMES[suffix]).format(order=order)


def paths_polydata(
    paths: EarlyReflections, order: int | None = None, *, mask: np.ndarray | None = None
) -> pyvista.PolyData:
    """The specular paths of one reflection order, or the paths ``mask`` picks, as
    polylines."""
    pv = _pyvista()
    if mask is None:
        mask = (paths.order == order) & ~paths.diffracted
    chosen = [p for p, keep in zip(paths.points, mask, strict=True) if keep]
    points = np.vstack(chosen) if chosen else np.zeros((0, 3))
    lines, start = [], 0
    for p in chosen:
        lines.extend([len(p), *range(start, start + len(p))])
        start += len(p)
    return pv.PolyData(points, lines=np.array(lines, dtype=np.int64))


class PathGroup(NamedTuple):
    """Paths drawn together: specular paths of one ``order``, or (order None) those
    diffracted at an edge."""

    label: str
    colour: str
    mask: np.ndarray
    order: int | None


def path_groups(paths: EarlyReflections, max_order: int | None = None) -> list[PathGroup]:
    """The specular paths by order, then those diffracted at an edge (with up to
    ``max_order`` reflections)."""
    groups = []
    specular = ~paths.diffracted
    top = int(paths.order.max()) if len(paths) else -1
    if max_order is not None:
        top = min(top, max_order)
    for order in range(top + 1):
        mask = specular & (paths.order == order)
        if mask.any():
            colour = ORDER_COLOURS[min(order, len(ORDER_COLOURS) - 1)]
            groups.append(PathGroup(_order_name(order), colour, mask, order))
    mask = paths.diffracted & (paths.order <= top)
    if mask.any():
        groups.append(PathGroup(_("diffracted"), DIFFRACTED_COLOUR, mask, None))
    return groups


def plot_paths(
    paths: EarlyReflections,
    plotter: pyvista.Plotter,
    *,
    max_order: int | None = None,
    line_width: float = 2.0,
) -> list[list[str]]:
    """Draw specular paths coloured by order and diffracted paths in their own colour;
    returns their legend entries."""
    legend = []
    for group in path_groups(paths, max_order):
        plotter.add_mesh(
            paths_polydata(paths, mask=group.mask),
            color=group.colour,
            line_width=line_width * 1.5 if group.order == 0 else line_width,
            opacity=0.8 if (group.order or 0) >= 2 else 1.0,
        )
        legend.append([f"{group.label} ({int(group.mask.sum())})", group.colour])
    return legend


def plot_leaks(
    leaks: Leaks, room: Room, plotter: pyvista.Plotter, *, render: bool = True
) -> list[list[str]]:
    """Draw the open edges of ``room``, the rays of ``leaks`` (from
    :func:`acoustic_spaces.engine.leaks.find_leaks`) that escape, from their last
    reflection to a little past where they leave, and a numbered marker on each gap;
    returns their legend entries."""
    from acoustic_spaces.engine.leaks import open_edges

    pv = _pyvista()
    legend = []
    corners, _faces = open_edges(room)
    if len(corners):
        lines = np.column_stack([np.full(len(corners), 2), corners]).reshape(-1)
        edges = pv.PolyData(room.mesh.vertices, lines=lines)
        plotter.add_mesh(edges, color=LEAK_COLOUR, line_width=5.0, render=False)
        legend.append([_("open edges ({count})").format(count=len(corners)), LEAK_COLOUR])
    if leaks.escaped:
        lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
        size = float(np.linalg.norm(hi - lo)) or 1.0
        step = max(1, -(-leaks.escaped // MAX_ESCAPES_DRAWN))
        start = leaks.starts[::step]
        end = leaks.points[::step] + 0.1 * size * leaks.directions[::step]
        points = np.stack([start, end], axis=1).reshape(-1, 3)
        lines = np.column_stack([np.full(len(start), 2), np.arange(2 * len(start)).reshape(-1, 2)])
        rays = pv.PolyData(points, lines=lines.reshape(-1))
        plotter.add_mesh(rays, color=ESCAPE_COLOUR, line_width=1.5, opacity=0.6, render=False)
        legend.append([_("escaping rays ({count})").format(count=leaks.escaped), ESCAPE_COLOUR])
        radius = max(0.012 * size, 0.05)
        centres = np.array([gap.centre for gap in leaks.gaps])
        for centre in centres:
            plotter.add_mesh(
                pv.Sphere(radius=radius, center=centre), color=LEAK_COLOUR, render=False
            )
        plotter.add_point_labels(
            centres + [0.0, 0.0, 1.5 * radius],
            [
                _("gap {number}: {share:.1f} %").format(
                    number=k, share=100.0 * gap.rays / leaks.rays
                )
                for k, gap in enumerate(leaks.gaps, 1)
            ],
            font_size=12,
            text_color="white",
            shape_color=PANEL,
            shape_opacity=0.8,
            show_points=False,
            always_visible=True,
            render=False,
        )
    if render:
        plotter.render()
    return legend


def balloon_polydata(
    directivity: Directivity,
    center,
    rotation=None,
    *,
    band: int = 4,
    size: float = 1.0,
    db_range: float = 30.0,
) -> pyvista.PolyData:
    """A directivity balloon around ``center`` in one octave band (``band`` indexes
    ``OCTAVE_BANDS_HZ``; 4 is 1 kHz): its radius grows from 0 at ``db_range`` dB below
    the loudest direction to ``size`` there. ``rotation`` has the source's forward, left
    and up axes as columns (``Orientation.rotation()``). The point data ``level`` holds
    the level in dB re the loudest direction.
    """
    pv = _pyvista()
    sphere = pv.Sphere(radius=1.0, theta_resolution=72, phi_resolution=36)
    local = np.asarray(sphere.points, dtype=float)
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        energy = directivity.in_bands("octave").energy(local)  # arrays: third octaves
        level = 10.0 * np.log10(np.maximum(energy[:, band], 1e-30))
    level -= level.max()
    radius = size * np.clip(1.0 + level / db_range, 0.0, 1.0)
    turn = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    sphere.points = np.asarray(center, dtype=float) + (local * radius[:, None]) @ turn.T
    sphere.point_data["level"] = level
    return sphere


def extent_polydata(source, radius: float) -> pyvista.PolyData | None:
    """A line source as a tube of ``radius`` (m), a surface source as its rectangle, or
    None for a point source."""
    if not source.extended:
        return None
    pv = _pyvista()
    corners = source.outline()
    if len(corners) == 2:
        return pv.Line(corners[0], corners[1]).tube(radius=radius)
    return pv.PolyData(corners, faces=[4, 0, 1, 2, 3])


def nearest_spacing(points) -> np.ndarray:
    """The distance from each of ``points`` (rows of coordinates) to the nearest other
    one: infinite for a point alone."""
    points = np.asarray(points, dtype=float)
    nearest = np.full(len(points), np.inf)
    if len(points) < 2:
        return nearest
    for start in range(0, len(points), 256):  # a block of rows at a time
        part = np.linalg.norm(points[start : start + 256, None] - points[None], axis=-1)
        part[np.arange(len(part)), np.arange(start, start + len(part))] = np.inf
        nearest[start : start + len(part)] = part.min(axis=1)
    return nearest


def clear_lengths(points, directions, radius, length) -> np.ndarray:
    """The length of the arrow from each of ``points`` along its unit row of
    ``directions`` (zeros for none): ``length``, cut short where the arrow would run into
    the marker (of ``radius``) of another point ahead of it, but no shorter than twice
    its own marker's radius."""
    points = np.asarray(points, dtype=float)
    directions = np.asarray(directions, dtype=float)
    radius = np.broadcast_to(np.asarray(radius, dtype=float), len(points))
    length = np.broadcast_to(np.asarray(length, dtype=float), len(points)).copy()
    for start in range(0, len(points), 256):  # a block of arrows at a time
        rows = slice(start, start + 256)
        ahead = points[None] - points[rows, None]  # from each arrow's start to every point
        along = np.einsum("bnd,bd->bn", ahead, directions[rows])
        aside = np.linalg.norm(ahead - along[..., None] * directions[rows, None], axis=-1)
        reach = length[rows, None]
        # The tip is a tenth of the arrow's length in radius.
        hit = (along > 0) & (aside < radius + 0.1 * reach) & (along - radius < reach)
        stop = np.where(hit, along - radius, np.inf).min(axis=1)
        length[rows] = np.minimum(length[rows], np.maximum(0.9 * stop, 2 * radius[rows]))
    return length


def marker_radius(positions, size: float) -> float:
    """The radius of the markers of sources and listeners at ``positions`` where they do
    not crowd, in a model ``size`` metres across: a fifth of their usual spacing (the
    median distance to the nearest one), from 1.2 % to 2.5 % of the model's size."""
    nearest = nearest_spacing(np.asarray(positions, dtype=float).reshape(-1, 3))
    apart = nearest[np.isfinite(nearest)]
    usual = float(np.median(apart)) if len(apart) else np.inf
    return max(float(np.clip(0.2 * usual, 0.012 * size, 0.025 * size)), 0.05)


def marker_sizes(positions, size: float, directions=None) -> tuple[np.ndarray, np.ndarray]:
    """The radius of the marker of each source or listener at ``positions`` and the
    length of its arrow along its unit row of ``directions``, in a model ``size`` metres
    across, from how far apart the points are: a fifth of their usual spacing (the
    median distance to the nearest one), from 1.2 % to 2.5 % of the model's size, and an
    arrow six times that. Where points crowd, their markers shrink so as not to overlap,
    down to a fifth, and an arrow stops short of a marker ahead of it."""
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    nearest = nearest_spacing(points)
    full = marker_radius(points, size)
    radius = np.clip(0.45 * nearest, 0.2 * full, full)
    if directions is None:
        return radius, 6 * radius
    return radius, clear_lengths(points, np.reshape(directions, (-1, 3)), radius, 6 * radius)


def vegetation_polydata(shape) -> pyvista.PolyData:
    """The volume of a tree, a hedge or a belt of trees
    (:class:`~acoustic_spaces.geometry.outdoor.PlantShape`) as a closed surface: a lone
    tree's crown, or an outline raised from its bottom to its top."""
    pv = _pyvista()
    if shape.outline is None:
        crown = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=16)
        axes = np.array([shape.radius, shape.radius, shape.half_height])
        crown.points = np.asarray(crown.points) * axes + shape.centre
        return crown
    n = len(shape.outline)
    bottom = np.column_stack([shape.outline, np.full(n, shape.low)])
    plan = pv.PolyData(bottom, faces=[n, *range(n)])
    return plan.extrude((0.0, 0.0, shape.high - shape.low), capping=True).triangulate()


def plot_vegetation(room: Room, plotter: pyvista.Plotter, *, render: bool = True) -> list:
    """Add an outdoor ``room``'s trees, hedges and belts of trees to ``plotter``, as
    see-through green volumes; returns their legend entries."""
    shapes = room.outdoor.vegetation if room.outdoor is not None else ()
    for shape in shapes:
        part = vegetation_polydata(shape)
        plotter.add_mesh(part, color=VEGETATION_COLOUR, opacity=0.3, pickable=False, render=render)
        edges = _outline(part)
        if edges.n_points:  # a lone tree's round crown has none
            plotter.add_mesh(
                edges, color=VEGETATION_COLOUR, line_width=1.0, pickable=False, render=render
            )
    return [[_("vegetation"), VEGETATION_COLOUR]] if shapes else []


class WindMap:
    """The weather over an outdoor room's ground as the views draw it: the scene's own
    (index 0) and each weather zone's (1 on), where each holds, and each one's wind at
    one ``height`` above the ground, the height the scene gives its wind's speed at, so
    that the zones' winds compare with the scene's. Zones are polygons in metres, cut
    to the ground; one that cannot be (its sides cross) or that lies off the ground is
    None and holds nowhere."""

    def __init__(self, room: Room) -> None:
        from acoustic_spaces.scene.model import Wind

        scene = room.scene
        atmosphere = scene.atmosphere
        self.layout = room.outdoor
        self.weathers = atmosphere.weathers()
        self.names = [scene.name, *atmosphere.zone_names()]
        self.zones = [None, *(atmosphere.zones or [])]
        self.height = float((atmosphere.wind or Wind()).height_m)
        self.speeds = np.array([float(w.wind_speed(self.height)) for w in self.weathers])
        self.directions = np.array([[*w.to_wind(), 0.0] for w in self.weathers])
        scale = UNIT_SCALE[scene.geometry.units or "m"]
        self.polygons: list[np.ndarray | None] = [None]
        for zone in atmosphere.zones or []:
            polygon = scale * np.asarray(zone.polygon, dtype=float)
            simple = len(polygon) >= 3 and plan_problem(polygon) is None
            self.polygons.append(clip_polygon(polygon, self.layout.extent) if simple else None)

    @classmethod
    def of(cls, room: Room) -> WindMap | None:
        """The map of ``room``, or None indoors and outdoors without an atmosphere."""
        if room.outdoor is None or room.scene.atmosphere is None:
            return None
        return cls(room)

    @property
    def windy(self) -> bool:
        """Whether the wind blows anywhere at the map's height."""
        return bool(np.any(self.speeds > 0.0))

    def weather_at(self, points) -> np.ndarray:
        """(n,) which weather holds at each point (n, 2 or 3): 0 for the scene's own, k
        for its zone k (the first where zones overlap)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
        out = np.zeros(len(points), dtype=np.int64)
        for k, polygon in enumerate(self.polygons):
            if polygon is not None:
                out[(out == 0) & in_polygon(points, polygon)] = k
        return out

    def centre(self, k: int) -> np.ndarray | None:
        """(2,) a point inside weather zone ``k``: its centroid, or another inside if
        the centroid is not (an L-shaped zone); None for a zone that holds nowhere."""
        polygon = self.polygons[k]
        if polygon is None:
            return None
        a, b = polygon, np.roll(polygon, -1, axis=0)
        cross = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
        centroid = ((a + b) * cross[:, None]).sum(axis=0) / (3.0 * cross.sum())
        if in_polygon(centroid[None], polygon)[0]:
            return centroid
        return _inner_point(polygon)[0]

    def label(self, k: int) -> str:
        """What weather zone ``k`` holds, in words: its name, its wind's speed (if the
        wind blows anywhere) and its ground temperature (if it has one of its own)."""
        name, zone = self.names[k], self.zones[k]
        temperature = None if zone is None else zone.temperature_c
        if not self.windy:
            if temperature is None:
                return name
            return _(ZONE_TEMPERATURE).format(name=name, temperature=temperature)
        if temperature is None:
            return _(ZONE_WIND).format(name=name, speed=self.speeds[k])
        return _(ZONE_WIND_TEMPERATURE).format(
            name=name, speed=self.speeds[k], temperature=temperature
        )

    def arrows(self, spacing: float | None = None, clear=None, within=None) -> WindArrows:
        """The arrows of the wind over the ground, one in the middle of each cell of a
        grid no more than ``spacing`` metres apart (by default about
        :data:`WIND_ARROWS` cells), each the wind of the weather there: the strongest
        :data:`WIND_REACH` of a cell long, the others in proportion, none where the air
        is still. Arrows that would pass near one of ``clear`` (m, 2), where sources and
        listeners stand, are left out, and a weather zone the grid misses gets one inside
        it. ``within`` (x_min, y_min, x_max, y_max) keeps only the cells it overlaps."""
        x0, y0, x1, y1 = self.layout.extent
        width, depth = x1 - x0, y1 - y0
        if spacing is None:
            spacing = math.sqrt(width * depth / WIND_ARROWS)
        # Two rows at least where there is room: sources and listeners often stand in a
        # line down the middle.
        nx, ny = (
            max(1 if side < 0.5 * spacing else 2, math.ceil(side / spacing))
            for side in (width, depth)
        )
        columns, rows = np.arange(nx), np.arange(ny)
        if within is not None:
            u0, v0, u1, v1 = within
            columns = columns[
                (x0 + (columns + 1) * width / nx > u0) & (x0 + columns * width / nx < u1)
            ]
            rows = rows[(y0 + (rows + 1) * depth / ny > v0) & (y0 + rows * depth / ny < v1)]
        xs = x0 + (columns + 0.5) * width / nx
        ys = y0 + (rows + 0.5) * depth / ny
        points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        cell = min(width / nx, depth / ny)
        reach = WIND_REACH * cell / float(self.speeds.max()) if self.windy else 0.0
        weather = self.weather_at(points)
        keep = self.speeds[weather] > 0.0
        clear = np.asarray(clear if clear is not None else [], dtype=float).reshape(-1, 2)
        if len(clear):
            half = 0.5 * reach * (self.speeds[weather] * self.directions[weather][:, :2].T).T
            start, along = points - half, 2.0 * half
            ahead = clear[None] - start[:, None]  # (n, m)
            t = (
                np.einsum("nmd,nd->nm", ahead, along)
                / np.maximum(np.einsum("nd,nd->n", along, along), 1e-300)[:, None]
            )
            nearest = start[:, None] + np.clip(t, 0.0, 1.0)[..., None] * along[:, None]
            gap = np.linalg.norm(nearest - clear[None], axis=-1).min(axis=1)
            keep &= gap >= 0.3 * cell
        points, weather = points[keep], weather[keep]
        held = set(weather.tolist())
        for k in range(1, len(self.polygons)):
            centre = self.centre(k)
            if k not in held and centre is not None and self.speeds[k] > 0.0:
                points, weather = np.vstack([points, centre]), np.append(weather, k)
        vectors = reach * (self.speeds[weather] * self.directions[weather].T).T
        return WindArrows(points.reshape(-1, 2), vectors.reshape(-1, 3), weather, reach, cell)


class WindArrows(NamedTuple):
    """The arrows of the wind over the ground (see :meth:`WindMap.arrows`)."""

    #: (n, 2) the middle of each arrow (m).
    middles: np.ndarray
    #: (n, 3) each arrow, horizontal, from its tail to its tip (m).
    vectors: np.ndarray
    #: (n,) the weather each arrow shows: 0 the scene's own, k its zone k.
    weather: np.ndarray
    #: How long an arrow is per m/s of wind (m).
    reach: float
    #: The size of the grid's cells (m).
    cell: float


def _arrows(starts, vectors, radius: float) -> pyvista.PolyData:
    """Arrows from each of ``starts`` (n, 3) along and as long as its row of ``vectors``,
    as one surface, all as thick: tips ``radius`` metres in radius (less on an arrow too
    short for them)."""
    pv = _pyvista()
    starts, vectors = np.asarray(starts, dtype=float), np.asarray(vectors, dtype=float)
    parts = []
    for start, vector in zip(starts, vectors, strict=True):
        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            continue
        tip = min(radius, 0.2 * length)
        parts.append(
            pv.Arrow(
                start=start,
                direction=vector,
                tip_length=min(0.35, 2.5 * tip / length),
                tip_radius=tip / length,
                shaft_radius=0.4 * tip / length,
                scale=length,
            )
        )
    return pv.merge(parts) if parts else pv.PolyData()


def _profile_points(wind: WindMap, arrows: WindArrows, clear) -> dict[int, np.ndarray]:
    """Where each windy weather's mast stands, in the middle of one of its ``arrows``:
    the one nearest the middle of the ground (of a zone, for a zone's) among those three
    cells or more from the sources and listeners (``clear``, (m, 2)), or failing that
    the one farthest from them."""
    x0, y0, x1, y1 = wind.layout.extent
    out = {}
    for k in np.flatnonzero(wind.speeds > 0.0):
        mine = arrows.middles[arrows.weather == k]
        if len(mine) == 0:
            continue
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) if k == 0 else wind.centre(int(k))
        apart = np.full(len(mine), 3.0 * arrows.cell)
        if len(clear):
            nearest = np.linalg.norm(mine[:, None] - clear[None], axis=-1).min(axis=1)
            apart = np.minimum(apart, nearest)
        # Clear first, then nearest the middle.
        best = np.lexsort((np.linalg.norm(mine - middle, axis=1), -apart))[0]
        out[int(k)] = mine[best]
    return out


def _plan_points(room: Room) -> np.ndarray:
    """(n, 2) what the sources and listeners take up seen from above, for the wind's
    arrows to keep clear of: where they stand, and points along the arrows the 3D view
    draws from them the way they face."""
    items = [*room.scene.sources, *room.scene.listeners]
    if not items:
        return np.zeros((0, 2))
    lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
    forward = np.array([item.orientation.forward() for item in items], dtype=float)
    positions = np.array([item.position for item in items], dtype=float)
    _radii, arrows = marker_sizes(positions, float(np.linalg.norm(hi - lo)) or 1.0, forward)
    drawn = [d is None or not d.omni for d in room.directivities]
    drawn += [True] * len(room.scene.listeners)
    along = np.linspace(0.0, 1.0, 9)
    points = [positions[:, :2]]
    for position, way, length, shown in zip(positions, forward, arrows, drawn, strict=True):
        if shown:
            points.append(position[:2] + np.outer(along * length, way[:2]))
    return np.vstack(points)


def plot_wind(
    room: Room,
    plotter: pyvista.Plotter,
    *,
    field: bool = True,
    profile: bool = False,
    render: bool = True,
) -> list:
    """Add the wind over an outdoor ``room``'s ground to ``plotter``. ``field``: sky blue
    arrows the way it blows, as long as it is strong, at the height the scene gives the
    wind's speed at above the ground, and each weather zone as see-through walls up to
    that height with its name and wind. ``profile``: a mast in each weather where the
    wind blows, with arrows at heights up it and a curve through the wind at every
    height, how the wind grows with height over the ground. Nothing indoors or in still
    air. Returns the legend entry of the wind."""
    wind = WindMap.of(room)
    if wind is None:
        return []
    pv = _pyvista()
    layout = wind.layout
    zones, labels = [], []
    for k in range(1, len(wind.polygons)):
        polygon = wind.polygons[k]
        if polygon is None or not field:
            continue
        floor = layout.ground_at(polygon)
        # Under the sky, a little, where the sky is lower than the wind's height.
        top = floor + np.minimum(wind.height, 0.95 * (layout.sky - floor))
        n = len(polygon)
        rings = np.vstack([np.column_stack([polygon, floor]), np.column_stack([polygon, top])])
        walls = [[4, j, (j + 1) % n, n + (j + 1) % n, n + j] for j in range(n)]
        zones.append(pv.PolyData(rings, faces=np.ravel(walls)))
        edges = [[n + 1, *range(n), 0], [n + 1, *range(n, 2 * n), n]]
        zones.append(pv.PolyData(rings, lines=np.ravel(edges)))
        centre = wind.centre(k)
        high = float(layout.ground_at(centre[None])[0]) + 1.15 * float(np.max(top - floor))
        labels.append(([*centre, high], wind.label(k)))
    for part in zones:
        line = part.n_lines > 0
        plotter.add_mesh(
            part,
            color=WIND_COLOUR,
            opacity=0.7 if line else 0.12,
            line_width=1.5,
            pickable=False,
            render=False,
        )
    legend = []
    if wind.windy:
        clear = _plan_points(room)
        arrows = wind.arrows(clear=clear)
        middles, vectors = arrows.middles, arrows.vectors
        ground = layout.ground_at(middles)
        lift = np.minimum(wind.height, 0.95 * (layout.sky - ground))
        # As thick as the markers of the sources and listeners make readable.
        lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
        everything = [item.position for item in (*room.scene.sources, *room.scene.listeners)]
        radius = min(
            0.3 * marker_radius(everything, float(np.linalg.norm(hi - lo)) or 1.0),
            0.15 * WIND_REACH * arrows.cell,
        )
        masts = _profile_points(wind, arrows, clear) if profile else {}
        # Buildings taller than the arrows hide them: none inside one.
        roofs = np.full(len(middles), -np.inf)
        for _name, footprint, roof in layout.buildings:
            inside = in_polygon(middles, footprint)
            roofs[inside] = np.maximum(roofs[inside], roof)
        keep = np.full(len(middles), field) & (roofs < ground + lift)
        for at in masts.values():
            keep &= np.linalg.norm(middles - at, axis=1) > 1e-9  # the mast's top arrow
        if keep.any():
            centres = np.column_stack([middles, ground + lift])
            plotter.add_mesh(
                _arrows(centres[keep] - 0.5 * vectors[keep], vectors[keep], radius),
                color=WIND_COLOUR,
                opacity=0.85,
                pickable=False,
                render=False,
            )
        for k, at in masts.items():
            labels.extend(_plot_mast(wind, k, at, arrows.reach, radius, plotter))
        legend.append([_(WIND_KEY).format(speed=wind.speeds[0], height=wind.height), WIND_COLOUR])
    if labels:
        plotter.add_point_labels(
            np.array([at for at, _text in labels], dtype=float),
            [text for _at, text in labels],
            font_size=11,
            text_color=WIND_COLOUR,
            shape_color=PANEL,
            shape_opacity=0.8,
            show_points=False,
            always_visible=True,
            render=False,
        )
    if render:
        plotter.render()
    return legend


def _plot_mast(wind: WindMap, k: int, at, reach: float, radius: float, plotter) -> list:
    """Add weather ``k``'s mast at ``at`` (2,): a pole from the ground up to the map's
    height, arrows from it at heights evenly up it, ``reach`` metres long per m/s and as
    thick as the wind's (``radius``) or less to keep apart, and a curve through the wind
    at every height between; returns the label of its lowest arrow (the highest's wind
    is the one the arrows over the ground show)."""
    pv = _pyvista()
    weather, direction = wind.weathers[k], wind.directions[k]
    floor = float(wind.layout.ground_at(at[None])[0])
    top = min(wind.height, 0.95 * (wind.layout.sky - floor))
    base = np.array([at[0], at[1], floor])
    heights = np.linspace(0.0, top, 49)
    curve = (
        base
        + np.outer(heights, [0, 0, 1])
        + np.outer(reach * weather.wind_speed(heights), direction)
    )
    plotter.add_mesh(
        pv.Line(base, base + [0.0, 0.0, top]),
        color="white",
        line_width=2.0,
        pickable=False,
        render=False,
    )
    plotter.add_mesh(
        pv.lines_from_points(curve), color=WIND_COLOUR, line_width=2.0, pickable=False, render=False
    )
    levels = top * np.arange(1, PROFILE_ARROWS + 1) / PROFILE_ARROWS
    starts = base + np.outer(levels, [0, 0, 1])
    vectors = np.outer(reach * weather.wind_speed(levels), direction)
    arrows = _arrows(starts, vectors, min(radius, 0.25 * top / PROFILE_ARROWS))
    plotter.add_mesh(arrows, color=WIND_COLOUR, pickable=False, render=False)
    text = _(PROFILE_SPEED).format(speed=float(weather.wind_speed(levels[0])), height=levels[0])
    return [(starts[0] + vectors[0], text)]


def plot_room(
    room: Room,
    plotter: pyvista.Plotter | None = None,
    *,
    opacity: float = 0.45,
    show_labels: bool = True,
    paths: EarlyReflections | None = None,
    max_order: int | None = None,
    coverage: CoverageMap | None = None,
    parameter: str = "g",
    band: int = 4,
    leaks: Leaks | None = None,
    wind: bool = True,
) -> pyvista.Plotter:
    """Add the room to ``plotter`` (a new one if None) and return it.

    ``paths`` adds early reflection paths (from :func:`acoustic_spaces.engine.early_paths`
    or a response's ``early``), coloured by order up to ``max_order``. ``coverage`` adds
    a coverage map of one ``parameter`` and ``band`` (see :func:`plot_coverage`).
    ``leaks`` adds the open edges and the rays that escape through them (see
    :func:`plot_leaks`). Outdoors, ``wind`` adds the wind and the weather zones (see
    :func:`plot_wind`).
    """
    pv = _pyvista()
    if plotter is None:
        plotter = pv.Plotter()
    legend = []
    # Outdoors, the open sky would hide the rest: only its edges, where the ground ends.
    shown = room.shown_faces()
    if not shown.all():
        plotter.add_mesh(
            _outline(room_polydata(room, np.flatnonzero(~shown))), color=EDGE_COLOUR, line_width=1.0
        )
    for k, name in enumerate(room.material_names):
        faces = np.flatnonzero((room.face_materials == k) & shown)
        if len(faces) == 0:
            continue
        part = room_polydata(room, faces)
        colour = MATERIAL_COLOURS[k % len(MATERIAL_COLOURS)]
        plotter.add_mesh(part, color=colour, opacity=opacity)
        plotter.add_mesh(_outline(part), color=EDGE_COLOUR, line_width=1.5)
        legend.append([name, colour])
    missing = np.flatnonzero((room.face_materials == UNASSIGNED) & shown)
    if len(missing):
        part = room_polydata(room, missing)
        plotter.add_mesh(part, color=UNASSIGNED_COLOUR, opacity=max(opacity, 0.8))
        plotter.add_mesh(_outline(part), color=EDGE_COLOUR, line_width=1.5)
        legend.append([_("no material"), UNASSIGNED_COLOUR])
    legend.extend(plot_vegetation(room, plotter))
    if wind:
        legend.extend(plot_wind(room, plotter, render=False))

    lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    sources, listeners = room.scene.sources, room.scene.listeners
    everything = [*sources, *listeners]
    radii, arrows = marker_sizes(
        [item.position for item in everything],
        size,
        [item.orientation.forward() for item in everything],
    )

    points, names = [], []
    for k, (source, directivity) in enumerate(zip(sources, room.directivities, strict=True)):
        radius, arrow = radii[k], arrows[k]
        opacity = 1.0 if source.in_use else 0.3  # left out of simulations: faded
        sphere = pv.Sphere(radius=radius, center=source.position)
        plotter.add_mesh(sphere, color=SOURCE_COLOUR, opacity=opacity)
        extent = extent_polydata(source, radius)
        if extent is not None:
            plotter.add_mesh(extent, color=SOURCE_COLOUR, opacity=0.6 * opacity)
        if not directivity.omni:
            direction = source.orientation.forward()
            shape = pv.Arrow(start=source.position, direction=direction, scale=arrow)
            plotter.add_mesh(shape, color=SOURCE_COLOUR, opacity=opacity)
            balloon = balloon_polydata(
                directivity, source.position, source.orientation.rotation(), size=arrow
            )
            plotter.add_mesh(
                balloon,
                scalars="level",
                cmap="magma",
                clim=(-30, 0),
                opacity=0.55 * opacity,
                show_scalar_bar=False,
            )
        points.append(np.add(source.position, [0.0, 0.0, 1.5 * radius]))
        names.append(source.name)
    for k, listener in enumerate(listeners, len(sources)):
        radius, arrow = radii[k], arrows[k]
        opacity = 1.0 if listener.in_use else 0.3
        sphere = pv.Sphere(radius=radius, center=listener.position)
        plotter.add_mesh(sphere, color=LISTENER_COLOUR, opacity=opacity)
        direction = listener.orientation.forward()
        shape = pv.Arrow(start=listener.position, direction=direction, scale=arrow)
        plotter.add_mesh(shape, color=LISTENER_COLOUR, opacity=opacity)
        points.append(np.add(listener.position, [0.0, 0.0, 1.5 * radius]))
        names.append(listener.name)
    legend.append(["source", SOURCE_COLOUR])
    legend.append(["listener", LISTENER_COLOUR])
    if paths is not None:
        legend.extend(plot_paths(paths, plotter, max_order=max_order))
    if coverage is not None:
        plot_coverage(coverage, plotter, parameter=parameter, band=band)
    if leaks is not None:
        legend.extend(plot_leaks(leaks, room, plotter, render=False))

    if show_labels and points:
        plotter.add_point_labels(
            np.asarray(points, dtype=float),
            names,
            font_size=14,
            text_color="white",
            shape_color=PANEL,
            shape_opacity=0.8,
            show_points=False,
            always_visible=True,
        )
    plotter.set_background(BACKGROUND[0], top=BACKGROUND[1])
    # Small and in a corner, where it hides little of the room and no colour bar.
    plotter.add_legend(
        legend,
        bcolor=PANEL,
        face="rectangle",
        loc="upper right",
        size=(0.15, 0.026 * len(legend)),
    )
    plotter.add_text(room.scene.name, font_size=11, color="white")
    plotter.add_axes()
    plotter.view_isometric()
    return plotter


def show_room(
    room: Room,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (1280, 800),
    **kwargs,
) -> None:
    """Open the room in a window, or save a PNG to ``screenshot`` without one.

    Keyword arguments go to :func:`plot_room` (for example ``paths``).
    """
    pv = _pyvista()
    plotter = pv.Plotter(off_screen=screenshot is not None, window_size=list(window_size))
    plot_room(room, plotter, **kwargs)
    if screenshot is not None:
        plotter.screenshot(str(screenshot))
        plotter.close()
    else:
        plotter.show()


def _particle_scalars(particles: Particles, frame: int, color: str, db_range: float):
    if color == "order":
        return particles.order[frame].astype(float)
    level = 10.0 * np.log10(np.maximum(particles.energy[frame], 1e-30))
    return np.maximum(level, -db_range)


def animate_particles(
    room: Room,
    particles: Particles,
    path: str | Path | None = None,
    *,
    fps: int = 25,
    color: str = "energy",
    db_range: float = 40.0,
    window_size: tuple[int, int] = (1024, 640),
) -> None:
    """Show sound particles (from :func:`acoustic_spaces.engine.trace_particles`) moving
    through the room: in a window with a time slider, or saved to ``path`` as an animated
    GIF or an MP4 video.

    ``color`` is ``"energy"`` (level in dB relative to a particle's share at emission,
    down to ``-db_range``) or ``"order"`` (reflections so far).
    """
    if color not in ("energy", "order"):
        raise ValueError("color must be 'energy' or 'order'")
    suffix = None if path is None else Path(path).suffix.lower()
    if suffix not in (None, ".gif", ".mp4"):
        raise ValueError("animations are saved as .gif or .mp4")
    pv = _pyvista()
    plotter = pv.Plotter(off_screen=path is not None, window_size=list(window_size))
    plot_room(room, plotter, opacity=0.12)
    cloud = pv.PolyData(particles.positions[0].copy())
    cloud["value"] = _particle_scalars(particles, 0, color, db_range)
    if color == "order":
        top = max(int(particles.order.max()), 1)
        title, clim, cmap = _("reflections"), [0, top], "turbo"
    else:
        title = _("level at {band} Hz (dB)").format(band=particles.band_hz)
        clim, cmap = [-db_range, 0.0], "inferno"
    plotter.add_mesh(
        cloud,
        scalars="value",
        clim=clim,
        cmap=cmap,
        render_points_as_spheres=True,
        point_size=max(3.0, window_size[0] / 200.0),
        scalar_bar_args={"title": title, "color": "white", "fmt": "%.0f"},
    )
    clock = plotter.add_text("t = 0.0 ms", position="lower_left", font_size=12, color="white")

    def show(frame: int) -> None:
        cloud.points = particles.positions[frame]
        cloud["value"] = _particle_scalars(particles, frame, color, db_range)
        clock.SetText(0, f"t = {1e3 * particles.times[frame]:.1f} ms")

    if path is None:
        last = len(particles.times) - 1

        def on_slider(value: float) -> None:
            show(int(round(value / (1e3 * particles.times[-1] or 1.0) * last)))

        plotter.add_slider_widget(
            on_slider,
            [0.0, 1e3 * float(particles.times[-1])],
            value=0.0,
            title=_("time (ms)"),
            color="white",
            pointa=(0.35, 0.92),
            pointb=(0.95, 0.92),
        )
        plotter.show()
        return
    _record(plotter, path, fps, len(particles.times), show)


# The viz extra installs both; named here for an install from before it did.
_MP4_NEEDS = (
    'MP4 videos need imageio and imageio-ffmpeg: pip install "imageio>=2.31" '
    '"imageio-ffmpeg>=0.5", or uv sync --extra viz in a source checkout'
)


def _record(plotter, path: str | Path, fps: int, frames: int, show, sound=None) -> None:
    """Save ``frames`` frames of ``plotter``, each set up by ``show(frame)``, to ``path``:
    an animated GIF or an MP4 video, with the WAV file ``sound`` as an MP4's sound track.
    Closes the plotter."""
    mp4 = Path(path).suffix.lower() == ".mp4"
    if mp4:
        audio = {} if sound is None else {"audio_path": str(sound), "audio_codec": "aac"}
        try:
            plotter.open_movie(str(path), framerate=fps, **audio)
        except ImportError as exc:
            plotter.close()
            raise ImportError(_MP4_NEEDS) from exc
    images = []
    for frame in range(frames):
        show(frame)
        if mp4:
            plotter.write_frame()
        else:
            plotter.render()
            images.append(plotter.screenshot(return_img=True))
    plotter.close()
    if not mp4:
        from PIL import Image

        # One palette for every frame, so colours that stay put do not flicker.
        sample = np.concatenate([images[0], images[len(images) // 2], images[-1]])
        palette = Image.fromarray(sample).quantize(colors=255)
        frames = [Image.fromarray(image).quantize(palette=palette, dither=0) for image in images]
        frames[0].save(
            str(path), save_all=True, append_images=frames[1:], duration=1000 / fps, loop=0
        )


def check_video(path: str | Path) -> None:
    """Raise what saving an animation to ``path`` would, before the work of making it:
    ValueError unless it is a .gif or .mp4, ImportError without PyVista or, for an MP4,
    without imageio and imageio-ffmpeg."""
    suffix = Path(path).suffix.lower()
    if suffix not in (".gif", ".mp4"):
        raise ValueError("animations are saved as .gif or .mp4")
    _pyvista()
    if suffix == ".mp4":
        import importlib.util

        if not all(importlib.util.find_spec(name) for name in ("imageio", "imageio_ffmpeg")):
            raise ImportError(_MP4_NEEDS)


def animate_walk(
    room: Room,
    walk: WalkResponses,
    path: str | Path,
    *,
    fps: int = 25,
    window_size: tuple[int, int] = (1024, 640),
    sound: str | Path | None = None,
) -> None:
    """Save a walk (from :func:`acoustic_spaces.walkthrough.simulate_walk`) as an
    animated GIF or an MP4 video: the room, the path and the walker going along it, in
    time with the sound of :func:`~acoustic_spaces.walkthrough.render_walk`. ``room`` is
    the scene's room. ``sound``, a WAV file of that sound in one or two channels, becomes
    an MP4's sound track, and the video then lasts as long as the sound (a GIF has none)."""
    check_video(path)
    duration = walk.duration_s
    mp4 = Path(path).suffix.lower() == ".mp4"
    if sound is not None and mp4:
        from acoustic_spaces.audio import read_wav

        data, rate = read_wav(sound)
        if data.shape[1] > 2:
            raise ValueError(f"a video's sound has one or two channels, not {data.shape[1]}")
        duration = max(duration, len(data) / rate)
    pv = _pyvista()
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plot_room(room, plotter, opacity=0.12)
    points = walk.points
    plotter.add_mesh(pv.lines_from_points(points.positions), color="white", line_width=3)
    size = max(0.2, 0.02 * float(np.ptp(room.mesh.vertices, axis=0).max()))
    marker = pv.Sphere(radius=size, center=(0.0, 0.0, 0.0))
    shape = marker.points.copy()
    marker.points = shape + points.positions[0]
    colour = SOURCE_COLOUR if walk.kind == "sources" else LISTENER_COLOUR
    plotter.add_mesh(marker, color=colour)
    yaws = points.yaws
    if yaws is None and walk.result is not None:
        # It keeps the orientation it has in the scene.
        scene = walk.result.room.scene
        walkers = scene.listeners if walk.kind == "listeners" else scene.sources
        first = next(w for w in walkers if w.name == walk.names[0])
        yaws = np.full(len(points.times), first.orientation.yaw)
    arrow = None
    if yaws is not None:
        # An arrow the way the walker faces, just above it so that the path does not
        # hide it, and its yaw in the text.
        arrow = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            tip_length=0.3,
            tip_radius=0.1,
            shaft_radius=0.04,
            scale=8.0 * size,
        )
        arrow.points += (0.0, 0.0, 1.5 * size)
        arrow_shape = arrow.points.copy()
        yaws = np.unwrap(np.radians(yaws))
        plotter.add_mesh(arrow, color=FACING_COLOUR, lighting=False)
    clock = plotter.add_text("", position="lower_left", font_size=12, color="white")
    plotter.view_isometric()
    frames = int(math.floor(duration * fps)) + 1

    def show(frame: int) -> None:
        t = frame / fps
        where = [np.interp(t, points.times, points.positions[:, k]) for k in range(3)]
        marker.points = shape + where
        text = f"{walk.name}: {t:.1f} s"
        if arrow is not None:
            yaw = float(np.interp(t, points.times, yaws))
            c, s = math.cos(yaw), math.sin(yaw)
            turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            arrow.points = arrow_shape @ turn.T + where
            # Translators: added to the name and time shown on a walkthrough video.
            text += _(", facing yaw {yaw:.0f}°").format(
                yaw=180.0 - (180.0 - math.degrees(yaw)) % 360.0
            )
        clock.SetText(0, text)

    _record(plotter, path, fps, frames, show, sound if mp4 else None)


# -- response plots ----------------------------------------------------------------


def _figure(**kwargs):
    """A matplotlib figure that needs no display or pyplot backend."""
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(f"response plots need matplotlib: {install_hint('viz')}") from exc
    return Figure(**kwargs)


def _dark(ax, grid: bool = True) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#6b6e7a")
    if grid:
        ax.grid(True, color="#3a3d4a", linewidth=0.6)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _two_panels():
    """A dark figure with two stacked panels."""
    fig = _figure(figsize=(10, 7), layout="constrained")
    axes = fig.subplots(2, 1)
    fig.patch.set_facecolor(BACKGROUND[0])
    for ax in axes:
        _dark(ax)
    return fig, axes


def _legend(ax, **kwargs) -> None:
    legend = ax.legend(facecolor=PANEL, edgecolor="#6b6e7a", fontsize=8, **kwargs)
    for text in legend.get_texts():
        text.set_color("white")


def plot_response(response: Response, path: str | Path | None = None, *, band: int = 4):
    """Echogram and decay curves of one response; saved to ``path`` if given.

    The echogram shows the energy per millisecond in one octave band (``band`` indexes
    ``OCTAVE_BANDS_HZ``; 4 is 1 kHz) with the early specular paths as stems coloured
    by order. The decay panel shows the backward-integrated curve of every band.
    A response in third-octave bands shows them averaged into octave bands.
    Returns the matplotlib figure.
    """
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, to_octaves
    from acoustic_spaces.parameters import decay_curve

    energy = response.octave_energy()
    t_ms = np.arange(energy.shape[1]) * response.time_step * 1e3
    reference = 1.0 / (4.0 * np.pi * 100.0)  # free field at 10 m, as for G

    fig, (ax_e, ax_d) = _two_panels()

    with np.errstate(divide="ignore"):
        late = to_octaves(response.late)[band]
        early = to_octaves(response.early.energy, axis=1)[:, band]
        late_db = 10.0 * np.log10(np.maximum(late, 1e-30) / reference)
        early_db = 10.0 * np.log10(np.maximum(early, 1e-30) / reference)
    late_t = np.arange(len(late_db)) * response.time_step * 1e3
    ax_e.fill_between(late_t, late_db, -100, step="post", color="#4da3ff", alpha=0.35, lw=0)
    ax_e.step(late_t, late_db, where="post", color="#4da3ff", lw=0.8, label=_("late (per ms)"))
    floor = float(np.max(early_db, initial=np.max(late_db, initial=0.0))) - 70.0
    for group in path_groups(response.early):
        delays = response.early.delay_s[group.mask] * 1e3
        ax_e.vlines(delays, floor, early_db[group.mask], color=group.colour, lw=1.2)
        ax_e.plot([], [], color=group.colour, label=group.label)
    ax_e.set_ylim(floor, floor + 80.0)
    ax_e.set_xlim(0, t_ms[-1] if len(t_ms) else 1)
    ax_e.set_xlabel(_("time (ms)"))
    ax_e.set_ylabel(_("level re 10 m free field (dB)"))
    ax_e.set_title(
        _("{source} to {listener}: echogram, {band} Hz band").format(
            source=response.source, listener=response.listener, band=OCTAVE_BANDS_HZ[band]
        )
    )
    _legend(ax_e, loc="upper right")

    start = int(response.direct_delay_s / response.time_step)
    for b, f in enumerate(OCTAVE_BANDS_HZ):
        decay = decay_curve(energy[b, start:], response.time_step)
        if not np.isfinite(decay[0]):
            continue
        ax_d.plot(
            t_ms[start:] - t_ms[start], decay, color=BAND_COLOURS[b], lw=1.2, label=_band_name(f)
        )
    ax_d.set_ylim(-70, 2)
    ax_d.set_xlim(0, t_ms[-1] if len(t_ms) else 1)
    ax_d.set_xlabel(_("time after the direct sound (ms)"))
    ax_d.set_ylabel(_("decay (dB)"))
    ax_d.set_title(_("backward-integrated decay per octave band"))
    _legend(ax_d, ncol=4)

    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


def _direction_axes(ax, title: str) -> None:
    ax.set_xlim(180, -180)  # left of the listener on the left
    ax.set_ylim(-90, 90)
    ax.set_xticks([180, 135, 90, 45, 0, -45, -90, -135, -180])
    back, left, front, right = _("back"), _("left"), _("front"), _("right")
    ax.set_xticklabels([back, "135", left, "45", front, "-45", right, "-135", back])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel(_("azimuth (degrees)"))
    ax.set_ylabel(_("elevation (degrees)"))
    ax.set_title(title)


def plot_directions(response: Response, path: str | Path | None = None, *, band: int = 4):
    """Where the sound comes from, as the listener faces: early reflections (dots sized
    by level and coloured by delay) and the late energy per arrival direction, in one
    octave band (``band`` indexes ``OCTAVE_BANDS_HZ``; 4 is 1 kHz). Azimuth runs from
    the left to the right of the listener. A response in third-octave bands shows them
    averaged into octave bands. Returns the matplotlib figure.
    """
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, to_octaves
    from acoustic_spaces.spatial.directions import from_angles, to_angles, to_head

    fig, (ax_e, ax_l) = _two_panels()
    reference = 1.0 / (4.0 * np.pi * 100.0)
    early = response.early
    if len(early):
        az, el = to_angles(to_head(early.direction, response.orientation))
        with np.errstate(divide="ignore"):
            in_band = to_octaves(early.energy, axis=1)[:, band]
            level = 10.0 * np.log10(np.maximum(in_band, 1e-30) / reference)
        floor = level.max() - 40.0
        keep = level > floor
        delay_ms = (early.delay_s - response.direct_delay_s) * 1e3
        order = np.argsort(-delay_ms[keep])  # earliest on top
        dots = ax_e.scatter(
            az[keep][order],
            el[keep][order],
            s=4 + 200 * ((level[keep][order] - floor) / 40.0) ** 2,
            c=delay_ms[keep][order],
            cmap="plasma",
            edgecolors="white",
            linewidths=0.4,
        )
        bar = fig.colorbar(dots, ax=ax_e, pad=0.01)
        bar.set_label(_("delay after the direct sound (ms)"), color="white")
        bar.ax.tick_params(colors="white")
    _direction_axes(
        ax_e,
        _(
            "{source} to {listener}: early reflections, {band} Hz band (dot size: level, top 40 dB)"
        ).format(source=response.source, listener=response.listener, band=OCTAVE_BANDS_HZ[band]),
    )

    if len(response.late_directions) and response.late_directional.size:
        energy = to_octaves(response.late_directional, axis=1)[:, band].sum(axis=1)
        heads = to_head(response.late_directions, response.orientation)
        grid_az, grid_el = np.meshgrid(np.arange(179, -180, -2.0), np.arange(-89, 90, 2.0))
        cells = from_angles(grid_az, grid_el)
        nearest = np.argmax(cells.reshape(-1, 3) @ heads.T, axis=1).reshape(grid_az.shape)
        # Energy per steradian: each direction stands for an equal share of the sphere.
        with np.errstate(divide="ignore"):
            level = 10.0 * np.log10(np.maximum(energy / energy.max(), 1e-6))
        image = ax_l.imshow(
            level[nearest],
            extent=(180, -180, -90, 90),
            origin="lower",
            cmap="magma",
            vmin=-20,
            vmax=0,
            aspect="auto",
            interpolation="nearest",
        )
        bar = fig.colorbar(image, ax=ax_l, pad=0.01)
        bar.set_label(_("level re loudest direction (dB)"), color="white")
        bar.ax.tick_params(colors="white")
        title = _("late energy by arrival direction, {band} Hz band").format(
            band=OCTAVE_BANDS_HZ[band]
        )
    else:
        ax_l.text(
            0,
            0,
            _("late energy arrives evenly from all directions"),
            color="white",
            ha="center",
            va="center",
        )
        title = _("late energy by arrival direction")
    _direction_axes(ax_l, title)

    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


def _polar_panels():
    """A dark figure with two polar panels side by side."""
    fig = _figure(figsize=(11, 5.6), layout="constrained")
    axes = fig.subplots(1, 2, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(BACKGROUND[0])
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white")
        ax.spines["polar"].set_color("#6b6e7a")
        ax.grid(True, color="#4a4d5a", linewidth=0.6)
        ax.title.set_color("white")
    return fig, axes


def plot_directivity(
    directivity: Directivity, path: str | Path | None = None, *, db_range: float = 30.0
):
    """Polar patterns of a source directivity in every octave band, in dB re the loudest
    direction of each band: the horizontal plane (seen from above, front at the top and
    the source's left on the left) and the vertical plane through the front (seen from
    the side, front on the right and up at the top). Returns the matplotlib figure.
    """
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

    directivity = directivity.in_bands("octave")  # an array's is in third octaves
    angle = np.radians(np.arange(0.0, 360.5, 1.0))
    planes = {
        "horizontal": np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)]),
        "vertical": np.column_stack([np.cos(angle), np.zeros_like(angle), np.sin(angle)]),
    }
    loudest = directivity.energy(planes["horizontal"]).max(axis=0)
    loudest = np.maximum(loudest, directivity.energy(planes["vertical"]).max(axis=0))
    fig, axes = _polar_panels()
    for ax, (plane, directions), title, zero in zip(
        axes,
        planes.items(),
        (_("horizontal plane (from above)"), _("vertical plane (from the side)")),
        ("N", "E"),
        strict=True,
    ):
        ax.set_theta_zero_location(zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            level = 10.0 * np.log10(directivity.energy(directions) / loudest)
        same = np.allclose(level, level[:, :1], atol=0.01, equal_nan=True)
        for b, f in enumerate(OCTAVE_BANDS_HZ):
            if same and b != 4:
                continue  # the same pattern in every band: one line
            label = _("every band") if same else _band_name(f)
            ax.plot(
                angle,
                np.clip(level[:, b], -db_range, 0.0),
                color=BAND_COLOURS[b],
                lw=1.3,
                label=label,
            )
        ax.set_rlim(-db_range, 0.0)
        ticks = np.arange(-db_range, 0.1, 10.0)
        ax.set_rticks(ticks, [f"{t:.0f} dB" for t in ticks])
        ax.set_rlabel_position(200 if plane == "horizontal" else 290)
        if plane == "horizontal":
            ax.set_xticks(
                np.radians([0, 90, 180, 270]), [_("front"), _("left"), _("back"), _("right")]
            )
        else:
            ax.set_xticks(
                np.radians([0, 90, 180, 270]), [_("front"), _("up"), _("back"), _("down")]
            )
        ax.set_title(title, pad=14)
    _legend(axes[1], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    index = directivity.index()[4]
    name = directivity.name or _("directivity")
    fig.suptitle(
        # The name on a line of its own: a per-band one is long.
        _("{name}\ndB re the loudest direction; directivity index {index:.1f} dB at 1 kHz").format(
            name=name, index=index
        ),
        color="white",
    )
    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


def plot_atmosphere(
    atmosphere: Atmosphere,
    ground_c: float,
    path: str | Path | None = None,
    *,
    top: float = 100.0,
):
    """An outdoor atmosphere from the ground to ``top`` metres up: the temperature and
    the wind's speed (left), and the speed of sound (right) in still air and, with a
    wind, as sound going with it and against it, the speed that bends sound: towards
    where it is lower. ``ground_c`` is the temperature at the ground (the scene's air
    temperature). Returns the matplotlib figure."""
    z = np.concatenate([[0.0], np.geomspace(min(0.01, top / 100.0), top, 400)])
    fig = _figure(figsize=(10, 5), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND[0])
    left, right = fig.subplots(1, 2, sharey=True)
    for ax in (left, right):
        _dark(ax)
    left.plot(atmosphere.temperature(z, ground_c), z, color="#fdae61", lw=1.5)
    left.set_xlabel(_("temperature (°C)"), color="#fdae61")
    left.set_ylabel(_("height above the ground (m)"))
    wind = atmosphere.wind_speed(z)
    if np.any(wind > 0.0):
        speed = left.twiny()
        speed.plot(wind, z, color="#80b1d3", lw=1.5)
        speed.set_xlabel(_("wind speed (m/s)"), color="#80b1d3")
        speed.tick_params(colors="white")
        speed.set_xlim(left=0.0)
        right.plot(
            atmosphere.sound_speed(z, ground_c),
            z,
            color="#d9d9d9",
            lw=1.2,
            ls="--",
            label=_("across the wind"),
        )
        along = atmosphere.to_wind()
        for sign, colour, label in (
            (1.0, "#80b1d3", _("downwind")),
            (-1.0, "#fb8072", _("upwind")),
        ):
            right.plot(
                atmosphere.sound_speed(z, ground_c, sign * along),
                z,
                color=colour,
                lw=1.5,
                label=label,
            )
        _legend(right, loc="best")
    else:
        right.plot(atmosphere.sound_speed(z, ground_c), z, color="#d9d9d9", lw=1.5)
    right.set_xlabel(_("speed of sound (m/s)"))
    left.set_ylim(0.0, top)
    if atmosphere.still:
        title = _("Still air: sound goes in straight lines")
    else:
        title = _("Sound bends towards where the speed of sound is lower")
    fig.suptitle(title, color="white")
    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


def _decay_curves(decay: BandDecay) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Each band's backward-integrated decay (dB) over 1 ms slots from the direct sound
    on, up to where it sinks into the noise, with the noise taken off and the missing
    tail added, as the parameters are computed: (times in ms, levels) per band, or None
    for a band with too little to draw."""
    from acoustic_spaces.parameters import decay_curve

    window = max(1, int(round(1e-3 * decay.sample_rate)))
    curves = []
    for b in range(decay.energy.shape[0]):
        e = decay.cleaned(b)
        slots = len(e) // window
        curve = None
        if slots >= 2:
            per_ms = e[: slots * window].reshape(slots, window).sum(axis=1)
            level = decay_curve(per_ms, 1e-3)
            if np.isfinite(level[0]):
                curve = (np.arange(slots) * 1.0, level)
        curves.append(curve)
    return curves


def plot_impulse_response(
    ir: np.ndarray,
    sample_rate: float,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    filters: str = "iec",
    noise: bool = True,
):
    """Level and decay curves of a measured (or any) impulse response.

    The top panel shows the squared response in dB over 1 ms windows and, per octave
    band with ``noise``, the background noise level found for it. The bottom panel shows
    each band's backward-integrated decay up to where it sinks into the noise, with the
    noise taken off and the missing tail added, as the parameters are computed.
    Returns the matplotlib figure; it is saved to ``path`` if given.
    """
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
    from acoustic_spaces.parameters import band_decay

    ir = np.asarray(ir, dtype=float)
    decay = band_decay(ir, sample_rate, filters=filters, noise=noise)
    window = max(1, int(round(1e-3 * sample_rate)))
    blocks = len(ir) // window
    power = (ir[: blocks * window] ** 2).reshape(blocks, window).mean(axis=1)
    peak = float(power.max(initial=0.0)) or 1.0
    t_ms = (np.arange(blocks) + 0.5) * window / sample_rate * 1e3

    fig, (ax_e, ax_d) = _two_panels()
    level = 10.0 * np.log10(np.maximum(power / peak, 1e-30))
    ax_e.plot(t_ms, level, color="#4da3ff", lw=0.8, label=_("broadband, 1 ms windows"))
    onset_ms = decay.onset_s * 1e3
    for b in range(len(OCTAVE_BANDS_HZ)):
        if decay.noise[b] > 0:
            noise_db = 10.0 * np.log10(decay.noise[b] / peak)
            end_ms = onset_ms + decay.crosspoint[b] / sample_rate * 1e3
            ax_e.hlines(noise_db, end_ms, t_ms[-1], color=BAND_COLOURS[b], lw=1.0)
            ax_e.plot([end_ms], [noise_db], "o", color=BAND_COLOURS[b], ms=3)
    if np.any(decay.noise > 0):
        ax_e.plot([], [], "o-", color="#d0d0d0", ms=3, label=_("band noise from its crosspoint"))
    ax_e.set_ylim(max(float(level.min()), -120.0) - 5.0, 5.0)
    ax_e.set_xlim(0, t_ms[-1] if len(t_ms) else 1)
    ax_e.set_xlabel(_("time (ms)"))
    ax_e.set_ylabel(_("level re peak (dB)"))
    ax_e.set_title(_("{title}: level").format(title=title) if title else _("level"))
    _legend(ax_e, loc="upper right")

    for b, (f, curve) in enumerate(zip(OCTAVE_BANDS_HZ, _decay_curves(decay), strict=True)):
        if curve is not None:
            ax_d.plot(*curve, color=BAND_COLOURS[b], lw=1.2, label=_band_name(f))
    ax_d.set_ylim(-70, 2)
    ax_d.set_xlim(0, max(t_ms[-1] - onset_ms, 1.0) if len(t_ms) else 1)
    ax_d.set_xlabel(_("time after the direct sound (ms)"))
    ax_d.set_ylabel(_("decay (dB)"))
    ax_d.set_title(_("backward-integrated decay per octave band, noise removed"))
    _legend(ax_d, ncol=4)

    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


def plot_comparison(
    simulated: np.ndarray,
    simulated_rate: float,
    measured: np.ndarray,
    measured_rate: float,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    labels: tuple[str, str] = ("simulated", "measured"),
    filters: str = "iec",
    noise: bool = True,
):
    """Decay curves of a simulated and a measured impulse response, one panel per
    octave band.

    Each panel overlays the backward-integrated (Schroeder) decays of the two, each from
    its own direct sound on, as :func:`plot_impulse_response` draws them: up to where the
    band sinks into its noise, with the noise taken off and the missing tail added.
    ``labels`` name the two responses in the legend. Returns the matplotlib figure; it
    is saved to ``path`` if given.
    """
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
    from acoustic_spaces.parameters import band_decay

    fig = _figure(figsize=(12, 6.5), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND[0])
    axes = fig.subplots(2, 4, sharex=True, sharey=True).ravel()
    for ax, f in zip(axes, OCTAVE_BANDS_HZ, strict=True):
        _dark(ax)
        ax.set_title(_band_name(f))
    floor_db, end_ms = -70.0, 1.0
    for ir, rate, label, colour in zip(
        (simulated, measured),
        (simulated_rate, measured_rate),
        labels,
        COMPARISON_COLOURS,
        strict=True,
    ):
        decay = band_decay(np.asarray(ir, dtype=float), rate, filters=filters, noise=noise)
        for ax, curve in zip(axes, _decay_curves(decay), strict=True):
            if curve is not None:
                ax.plot(*curve, color=colour, lw=1.2)
                # Time runs to where the slowest decay leaves the panel at the bottom.
                below = np.flatnonzero(curve[1] < floor_db)
                end_ms = max(end_ms, float(curve[0][below[0] if len(below) else -1]))
        # The legend's entry, drawn empty: a band may have no curve to label.
        axes[0].plot([], [], color=colour, lw=1.2, label=label)
    axes[0].set_xlim(0, end_ms)
    axes[0].set_ylim(floor_db, 2)
    for ax in axes[4:]:
        ax.set_xlabel(_("time after the direct sound (ms)"))
    for ax in axes[::4]:
        ax.set_ylabel(_("decay (dB)"))
    _legend(axes[0], loc="upper right")
    fig.suptitle(
        title or _("backward-integrated decay per octave band, noise removed"), color="white"
    )
    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


# -- coverage maps -----------------------------------------------------------------

#: Colour map of coverage maps.
MAP_COLOURS = "turbo"
#: Opacity of a coverage map in 3D: see-through, so the sources and listeners at
#: listening height, half under its tiles, stay in sight.
MAP_OPACITY = 0.7


def _band_name(f: int) -> str:
    return f"{f} Hz" if f < 1000 else f"{f // 1000} kHz"


def map_values(coverage: CoverageMap, parameter: str = "g", band: int = 4):
    """The values a map shows of ``parameter`` (in ``band`` when it has bands, as an
    index into ``OCTAVE_BANDS_HZ``), in the units of its label, and its title."""
    from acoustic_spaces.engine.coverage import MAP_PARAMETERS
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ

    if parameter not in MAP_PARAMETERS:
        raise ValueError(f"unknown parameter {parameter!r}; use one of {', '.join(MAP_PARAMETERS)}")
    info = MAP_PARAMETERS[parameter]
    values = coverage.value(parameter, band if info.per_band else None) * info.scale
    label = _(info.label)
    title = f"{label}, {_band_name(OCTAVE_BANDS_HZ[band])}" if info.per_band else label
    return values, title


def map_polydata(coverage: CoverageMap, parameter: str = "g", band: int = 4) -> pyvista.PolyData:
    """The map's columns as tiles at listening height, over the audience surfaces
    (:meth:`~acoustic_spaces.engine.coverage.MapGrid.tiles`); the cell data ``value``
    holds the parameter as :func:`map_values` gives it (NaN where undefined) and
    ``point`` the map point of each tile."""
    pv = _pyvista()
    corners, owner = coverage.grid.tiles()
    n = len(corners)
    cells = np.column_stack([np.full(n, 4), np.arange(4 * n).reshape(n, 4)]).ravel()
    poly = pv.PolyData(corners.reshape(-1, 3), cells)
    poly.cell_data["value"] = map_values(coverage, parameter, band)[0][owner]
    poly.cell_data["point"] = owner
    return poly


def _limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def plot_coverage(
    coverage: CoverageMap,
    plotter: pyvista.Plotter,
    *,
    parameter: str = "g",
    band: int = 4,
    render: bool = True,
):
    """Add a coverage map to a 3D view: coloured tiles at listening height over the
    audience area, with a colour bar. Returns the actor of the tiles."""
    pv = _pyvista()
    poly = map_polydata(coverage, parameter, band)
    _values, title = map_values(coverage, parameter, band)
    actor = plotter.add_mesh(
        poly,
        scalars="value",
        cmap=MAP_COLOURS,
        clim=_limits(poly.cell_data["value"]),
        nan_color="#6b6e7a",
        opacity=MAP_OPACITY,
        pickable=False,
        render=render,
        scalar_bar_args={
            "title": f"{coverage.source}: {title}",
            "color": "white",
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.08,
            "width": 0.06,
            "height": 0.55,
            "fmt": "%.3g",
            "title_font_size": 14,
            "label_font_size": 13,
        },
    )
    # Lines between whole columns; cut columns at the edges get none.
    grid = coverage.grid
    whole = grid.squares()[grid.cover.all(axis=1)] + [0.0, 0.0, 0.01]  # over the tiles
    if len(whole):
        n = len(whole)
        lines = np.column_stack([np.full(n, 5), np.arange(4 * n).reshape(n, 4), 4 * np.arange(n)])
        outline = pv.PolyData(whole.reshape(-1, 3), lines=lines.ravel())
        plotter.add_mesh(
            outline,
            color=BACKGROUND[0],
            line_width=0.6,
            opacity=MAP_OPACITY,
            pickable=False,
            render=render,
        )
    return actor


def _floor_plan(ax, room: Room) -> None:
    """The walls seen from above, as thin lines, and any vegetation's outlines."""
    from matplotlib.collections import LineCollection

    _vegetation_plan(ax, room)
    mesh = room.mesh
    walls = np.abs(mesh.face_normals()[:, 2]) < 0.25
    tri = mesh.triangles[walls][:, :, :2]
    if len(tri) == 0:
        return
    # Each wall seen from above is a line: the two corners farthest apart.
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    length = np.linalg.norm(tri[:, pairs[:, 0]] - tri[:, pairs[:, 1]], axis=2)
    best = pairs[np.argmax(length, axis=1)]
    rows = np.arange(len(tri))
    segments = np.stack([tri[rows, best[:, 0]], tri[rows, best[:, 1]]], axis=1)
    ax.add_collection(LineCollection(segments, colors=EDGE_COLOUR, linewidths=0.8, alpha=0.6))


def _vegetation_plan(ax, room: Room) -> None:
    """The outlines of an outdoor room's trees, hedges and belts of trees, seen from
    above, as dashed green lines."""
    from matplotlib.patches import Ellipse, Polygon

    for shape in room.outdoor.vegetation if room.outdoor is not None else ():
        style = dict(fill=False, edgecolor=VEGETATION_COLOUR, linestyle="--", linewidth=1.0)
        if shape.outline is None:
            size = 2.0 * shape.radius
            ax.add_patch(Ellipse(shape.centre[:2], size, size, **style))
        else:
            ax.add_patch(Polygon(shape.outline, closed=True, **style))


def plot_map(
    coverage: CoverageMap,
    path: str | Path | None = None,
    *,
    parameter: str = "g",
    band: int = 4,
    room: Room | None = None,
):
    """A coverage map seen from above: one panel per level (stalls, balcony, ...),
    columns coloured by ``parameter`` in ``band`` (an index into ``OCTAVE_BANDS_HZ``;
    4 is 1 kHz) and labelled with their values when there is room. With ``room`` the
    walls and sources are drawn too. Returns the matplotlib figure; it is saved to
    ``path`` if given."""
    from acoustic_spaces.engine.coverage import MAP_PARAMETERS

    grid = coverage.grid
    values, title = map_values(coverage, parameter, band)
    ny, nx, levels = grid.index.shape
    lo, hi = _limits(values)
    x0, y0 = grid.origin
    extent = (x0, x0 + nx * grid.cell, y0, y0 + ny * grid.cell)
    # Each panel shows the map, and with a room all of its walls and sources.
    xs, ys = [extent[0], extent[1]], [extent[2], extent[3]]
    if room is not None:
        lo_b, hi_b = room.report.bounds_min, room.report.bounds_max
        xs += [lo_b[0], hi_b[0]]
        ys += [lo_b[1], hi_b[1]]
    width, depth = max(xs) - min(xs), max(ys) - min(ys)
    margin = 0.03 * max(width, depth)
    fig_width = min(16.0, 2.0 + 7.0 * levels)
    panel = (fig_width - 2.0) / levels
    fig_height = min(10.0, max(3.5, panel * depth / max(width, 1e-9) + 1.4))
    fig = _figure(figsize=(fig_width, fig_height), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND[0])
    axes = fig.subplots(1, levels, squeeze=False)[0]
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as Outline

    # All triangles counter-clockwise, so the nonzero rule fills their union.
    tri = grid.footprint.copy()
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    turn = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0
    tri[turn] = tri[turn][:, ::-1]
    codes = np.tile([Outline.MOVETO, Outline.LINETO, Outline.LINETO, Outline.CLOSEPOLY], len(tri))
    corners = np.concatenate([tri, tri[:, :1]], axis=1).reshape(-1, 2)
    outline = Outline(corners, codes)
    info = MAP_PARAMETERS[parameter]
    label_cells = nx * ny <= 400
    image = None
    for level, ax in enumerate(axes):
        _dark(ax, grid=False)
        index = grid.index[:, :, level]
        shown = np.full(index.shape, np.nan)
        has = index >= 0
        shown[has] = values[index[has]]
        image = ax.imshow(
            np.ma.masked_invalid(shown),
            origin="lower",
            extent=extent,
            cmap=MAP_COLOURS,
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        # Columns at the edge of the surfaces show only the part over them.
        image.set_clip_path(PathPatch(outline, transform=ax.transData))
        if label_cells:
            size = max(5.0, min(9.0, 60.0 / max(nx, ny)))
            for iy, ix in zip(*np.nonzero(has), strict=True):
                k = index[iy, ix]
                value = values[k]
                text = info.format.format(value) if np.isfinite(value) else "–"
                cx, cy = grid.points[k, :2]
                shade = (value - lo) / (hi - lo) if np.isfinite(value) else 0.5
                colour = "black" if 0.25 < shade < 0.85 else "white"
                ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=colour)
        if room is not None:
            _floor_plan(ax, room)
            for source in room.scene.sources:
                x, y, _z = source.position
                if source.extended:
                    corners = source.outline()
                    if len(corners) > 2:
                        corners = np.vstack([corners, corners[:1]])
                    ax.plot(corners[:, 0], corners[:, 1], "-", color=SOURCE_COLOUR, lw=2.5)
                ax.plot([x], [y], "o", color=SOURCE_COLOUR, ms=7, mec="white", mew=0.8)
                ax.annotate(
                    source.name,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="white",
                    fontsize=8,
                )
        ax.set_aspect("equal")
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_xlabel(_("x (m)"))
        ax.set_ylabel(_("y (m)"))
        if levels > 1:
            heights = grid.surface[index[has]]
            ax.set_title(
                _("level {number}, floor at about {height:.1f} m").format(
                    number=level + 1, height=np.median(heights)
                )
            )
    bar = fig.colorbar(image, ax=list(axes), shrink=0.9)
    bar.set_label(title, color="white")
    bar.ax.tick_params(colors="white")
    bar.outline.set_edgecolor("#6b6e7a")
    fig.suptitle(
        _("{source}: {title}, {count} points {spacing:g} m apart").format(
            source=coverage.source, title=title, count=len(grid), spacing=grid.cell
        ),
        color="white",
    )
    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig
