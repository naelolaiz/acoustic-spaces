"""Pictures of rooms and responses.

- 3D view (PyVista): surfaces coloured by material, sources with their directivity
  balloons, listeners, and early reflection paths coloured by order (paths diffracted
  at an edge in a colour of their own).
- Leaks (PyVista): the open edges of a model that is not closed, and the rays of the
  leak finder that escape through its gaps.
- Animation (PyVista): sound particles spreading from a source and reflecting around
  the room, in a window with a time slider or saved as a GIF or MP4 video; a walker
  going along its walk, as a video to go with its sound.
- Response plots (matplotlib): echogram with the early paths, decay curves per band, and
  arrival directions as the listener faces; the decays of a simulated and a measured
  response band by band; polar patterns of a source directivity.
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
from acoustic_spaces.room import UNASSIGNED, Room

if TYPE_CHECKING:
    import pyvista

    from acoustic_spaces.directivity import Directivity
    from acoustic_spaces.engine import CoverageMap, EarlyReflections, Response
    from acoustic_spaces.engine.leaks import Leaks
    from acoustic_spaces.engine.particles import Particles
    from acoustic_spaces.parameters import BandDecay
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


def _order_name(order: int) -> str:
    if order == 0:
        return "direct"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(order if order < 20 else order % 10, "th")
    return f"{order}{suffix} order"


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
        groups.append(PathGroup("diffracted", DIFFRACTED_COLOUR, mask, None))
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
    corners, _ = open_edges(room)
    if len(corners):
        lines = np.column_stack([np.full(len(corners), 2), corners]).reshape(-1)
        edges = pv.PolyData(room.mesh.vertices, lines=lines)
        plotter.add_mesh(edges, color=LEAK_COLOUR, line_width=5.0, render=False)
        legend.append([f"open edges ({len(corners)})", LEAK_COLOUR])
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
        legend.append([f"escaping rays ({leaks.escaped})", ESCAPE_COLOUR])
        radius = max(0.012 * size, 0.05)
        centres = np.array([gap.centre for gap in leaks.gaps])
        for centre in centres:
            plotter.add_mesh(
                pv.Sphere(radius=radius, center=centre), color=LEAK_COLOUR, render=False
            )
        plotter.add_point_labels(
            centres + [0.0, 0.0, 1.5 * radius],
            [
                f"gap {k}: {100.0 * gap.rays / leaks.rays:.1f} %"
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
) -> pyvista.Plotter:
    """Add the room to ``plotter`` (a new one if None) and return it.

    ``paths`` adds early reflection paths (from :func:`acoustic_spaces.engine.early_paths`
    or a response's ``early``), coloured by order up to ``max_order``. ``coverage`` adds
    a coverage map of one ``parameter`` and ``band`` (see :func:`plot_coverage`).
    ``leaks`` adds the open edges and the rays that escape through them (see
    :func:`plot_leaks`).
    """
    pv = _pyvista()
    if plotter is None:
        plotter = pv.Plotter()
    legend = []
    for k, name in enumerate(room.material_names):
        faces = np.flatnonzero(room.face_materials == k)
        if len(faces) == 0:
            continue
        part = room_polydata(room, faces)
        colour = MATERIAL_COLOURS[k % len(MATERIAL_COLOURS)]
        plotter.add_mesh(part, color=colour, opacity=opacity)
        plotter.add_mesh(_outline(part), color=EDGE_COLOUR, line_width=1.5)
        legend.append([name, colour])
    missing = np.flatnonzero(room.face_materials == UNASSIGNED)
    if len(missing):
        part = room_polydata(room, missing)
        plotter.add_mesh(part, color=UNASSIGNED_COLOUR, opacity=max(opacity, 0.8))
        plotter.add_mesh(_outline(part), color=EDGE_COLOUR, line_width=1.5)
        legend.append(["no material", UNASSIGNED_COLOUR])

    lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
    size = float(np.linalg.norm(hi - lo)) or 1.0
    radius = max(0.012 * size, 0.05)
    arrow = 6 * radius

    points, names = [], []
    for source, directivity in zip(room.scene.sources, room.directivities, strict=True):
        plotter.add_mesh(pv.Sphere(radius=radius, center=source.position), color=SOURCE_COLOUR)
        extent = extent_polydata(source, radius)
        if extent is not None:
            plotter.add_mesh(extent, color=SOURCE_COLOUR, opacity=0.6)
        if not directivity.omni:
            direction = source.orientation.forward()
            shape = pv.Arrow(start=source.position, direction=direction, scale=arrow)
            plotter.add_mesh(shape, color=SOURCE_COLOUR)
            balloon = balloon_polydata(
                directivity, source.position, source.orientation.rotation(), size=arrow
            )
            plotter.add_mesh(
                balloon,
                scalars="level",
                cmap="magma",
                clim=(-30, 0),
                opacity=0.55,
                show_scalar_bar=False,
            )
        points.append(source.position)
        names.append(source.name)
    for listener in room.scene.listeners:
        plotter.add_mesh(pv.Sphere(radius=radius, center=listener.position), color=LISTENER_COLOUR)
        direction = listener.orientation.forward()
        shape = pv.Arrow(start=listener.position, direction=direction, scale=arrow)
        plotter.add_mesh(shape, color=LISTENER_COLOUR)
        points.append(listener.position)
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
            np.asarray(points, dtype=float) + [0.0, 0.0, 1.5 * radius],
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
        title, clim, cmap = "reflections", [0, top], "turbo"
    else:
        title, clim, cmap = f"level at {particles.band_hz} Hz (dB)", [-db_range, 0.0], "inferno"
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
            title="time (ms)",
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
            text += f", facing yaw {180.0 - (180.0 - math.degrees(yaw)) % 360.0:.0f}°"
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
    ax_e.step(late_t, late_db, where="post", color="#4da3ff", lw=0.8, label="late (per ms)")
    floor = float(np.max(early_db, initial=np.max(late_db, initial=0.0))) - 70.0
    for group in path_groups(response.early):
        delays = response.early.delay_s[group.mask] * 1e3
        ax_e.vlines(delays, floor, early_db[group.mask], color=group.colour, lw=1.2)
        ax_e.plot([], [], color=group.colour, label=group.label)
    ax_e.set_ylim(floor, floor + 80.0)
    ax_e.set_xlim(0, t_ms[-1] if len(t_ms) else 1)
    ax_e.set_xlabel("time (ms)")
    ax_e.set_ylabel("level re 10 m free field (dB)")
    ax_e.set_title(
        f"{response.source} to {response.listener}: echogram, {OCTAVE_BANDS_HZ[band]} Hz band"
    )
    _legend(ax_e, loc="upper right")

    start = int(response.direct_delay_s / response.time_step)
    for b, f in enumerate(OCTAVE_BANDS_HZ):
        decay = decay_curve(energy[b, start:], response.time_step)
        if not np.isfinite(decay[0]):
            continue
        label = f"{f} Hz" if f < 1000 else f"{f // 1000} kHz"
        ax_d.plot(t_ms[start:] - t_ms[start], decay, color=BAND_COLOURS[b], lw=1.2, label=label)
    ax_d.set_ylim(-70, 2)
    ax_d.set_xlim(0, t_ms[-1] if len(t_ms) else 1)
    ax_d.set_xlabel("time after the direct sound (ms)")
    ax_d.set_ylabel("decay (dB)")
    ax_d.set_title("backward-integrated decay per octave band")
    _legend(ax_d, ncol=4)

    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


def _direction_axes(ax, title: str) -> None:
    ax.set_xlim(180, -180)  # left of the listener on the left
    ax.set_ylim(-90, 90)
    ax.set_xticks([180, 135, 90, 45, 0, -45, -90, -135, -180])
    ax.set_xticklabels(["back", "135", "left", "45", "front", "-45", "right", "-135", "back"])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("azimuth (degrees)")
    ax.set_ylabel("elevation (degrees)")
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
        bar.set_label("delay after the direct sound (ms)", color="white")
        bar.ax.tick_params(colors="white")
    _direction_axes(
        ax_e,
        f"{response.source} to {response.listener}: early reflections, "
        f"{OCTAVE_BANDS_HZ[band]} Hz band (dot size: level, top 40 dB)",
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
        bar.set_label("level re loudest direction (dB)", color="white")
        bar.ax.tick_params(colors="white")
        title = f"late energy by arrival direction, {OCTAVE_BANDS_HZ[band]} Hz band"
    else:
        ax_l.text(
            0,
            0,
            "late energy arrives evenly from all directions",
            color="white",
            ha="center",
            va="center",
        )
        title = "late energy by arrival direction"
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
        ("horizontal plane (from above)", "vertical plane (from the side)"),
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
            label = "every band" if same else f"{f} Hz" if f < 1000 else f"{f // 1000} kHz"
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
            ax.set_xticks(np.radians([0, 90, 180, 270]), ["front", "left", "back", "right"])
        else:
            ax.set_xticks(np.radians([0, 90, 180, 270]), ["front", "up", "back", "down"])
        ax.set_title(title, pad=14)
    _legend(axes[1], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    index = directivity.index()[4]
    name = directivity.name or "directivity"
    fig.suptitle(
        # The name on a line of its own: a per-band one is long.
        f"{name}\ndB re the loudest direction; directivity index {index:.1f} dB at 1 kHz",
        color="white",
    )
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
    ax_e.plot(t_ms, level, color="#4da3ff", lw=0.8, label="broadband, 1 ms windows")
    onset_ms = decay.onset_s * 1e3
    for b in range(len(OCTAVE_BANDS_HZ)):
        if decay.noise[b] > 0:
            noise_db = 10.0 * np.log10(decay.noise[b] / peak)
            end_ms = onset_ms + decay.crosspoint[b] / sample_rate * 1e3
            ax_e.hlines(noise_db, end_ms, t_ms[-1], color=BAND_COLOURS[b], lw=1.0)
            ax_e.plot([end_ms], [noise_db], "o", color=BAND_COLOURS[b], ms=3)
    if np.any(decay.noise > 0):
        ax_e.plot([], [], "o-", color="#d0d0d0", ms=3, label="band noise from its crosspoint")
    ax_e.set_ylim(max(float(level.min()), -120.0) - 5.0, 5.0)
    ax_e.set_xlim(0, t_ms[-1] if len(t_ms) else 1)
    ax_e.set_xlabel("time (ms)")
    ax_e.set_ylabel("level re peak (dB)")
    ax_e.set_title(f"{title}: level" if title else "level")
    _legend(ax_e, loc="upper right")

    for b, (f, curve) in enumerate(zip(OCTAVE_BANDS_HZ, _decay_curves(decay), strict=True)):
        if curve is not None:
            ax_d.plot(*curve, color=BAND_COLOURS[b], lw=1.2, label=_band_name(f))
    ax_d.set_ylim(-70, 2)
    ax_d.set_xlim(0, max(t_ms[-1] - onset_ms, 1.0) if len(t_ms) else 1)
    ax_d.set_xlabel("time after the direct sound (ms)")
    ax_d.set_ylabel("decay (dB)")
    ax_d.set_title("backward-integrated decay per octave band, noise removed")
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
        ax.set_xlabel("time after the direct sound (ms)")
    for ax in axes[::4]:
        ax.set_ylabel("decay (dB)")
    _legend(axes[0], loc="upper right")
    fig.suptitle(title or "backward-integrated decay per octave band, noise removed", color="white")
    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig


# -- coverage maps -----------------------------------------------------------------

#: Colour map of coverage maps.
MAP_COLOURS = "turbo"


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
    title = f"{info.label}, {_band_name(OCTAVE_BANDS_HZ[band])}" if info.per_band else info.label
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
    _, title = map_values(coverage, parameter, band)
    actor = plotter.add_mesh(
        poly,
        scalars="value",
        cmap=MAP_COLOURS,
        clim=_limits(poly.cell_data["value"]),
        nan_color="#6b6e7a",
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
            outline, color=BACKGROUND[0], line_width=0.6, pickable=False, render=render
        )
    return actor


def _floor_plan(ax, room: Room) -> None:
    """The walls seen from above, as thin lines."""
    from matplotlib.collections import LineCollection

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
                x, y, _ = source.position
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
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if levels > 1:
            heights = grid.surface[index[has]]
            ax.set_title(f"level {level + 1}, floor at about {np.median(heights):.1f} m")
    bar = fig.colorbar(image, ax=list(axes), shrink=0.9)
    bar.set_label(title, color="white")
    bar.ax.tick_params(colors="white")
    bar.outline.set_edgecolor("#6b6e7a")
    fig.suptitle(
        f"{coverage.source}: {title}, {len(grid)} points {grid.cell:g} m apart", color="white"
    )
    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig
