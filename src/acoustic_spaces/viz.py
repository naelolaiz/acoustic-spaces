"""Pictures of rooms and responses.

- 3D view (PyVista): surfaces coloured by material, sources, listeners, and early
  reflection paths coloured by order.
- Animation (PyVista): sound particles spreading from a source and reflecting around
  the room, in a window with a time slider or saved as a GIF or MP4 video.
- Response plots (matplotlib): echogram with the early paths, decay curves per band, and
  arrival directions as the listener faces.

Needs the viz extra (``pip install "acoustic-spaces[viz]"``). The 3D view works in a
desktop window, in Jupyter, and off screen for screenshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from acoustic_spaces.room import UNASSIGNED, Room

if TYPE_CHECKING:
    import pyvista

    from acoustic_spaces.engine import EarlyReflections, Response
    from acoustic_spaces.engine.particles import Particles

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
BACKGROUND = ("#16171d", "#3a3d4a")  # bottom, top
PANEL = "#26272f"
EDGE_COLOUR = "#d0d0d0"
#: Reflection paths by order, direct sound first; orders past the end reuse the last.
ORDER_COLOURS = ("#ffffff", "#ffd166", "#f78c6b", "#ef476f", "#b5179e", "#7209b7", "#4361ee")
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
        raise ImportError('the 3D view needs PyVista: pip install "acoustic-spaces[viz]"') from exc
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


def paths_polydata(paths: EarlyReflections, order: int) -> pyvista.PolyData:
    """The paths of one reflection order as polylines."""
    pv = _pyvista()
    chosen = [p for p, o in zip(paths.points, paths.order, strict=True) if o == order]
    points = np.vstack(chosen) if chosen else np.zeros((0, 3))
    lines, start = [], 0
    for p in chosen:
        lines.extend([len(p), *range(start, start + len(p))])
        start += len(p)
    return pv.PolyData(points, lines=np.array(lines, dtype=np.int64))


def plot_paths(
    paths: EarlyReflections,
    plotter: pyvista.Plotter,
    *,
    max_order: int | None = None,
    line_width: float = 2.0,
) -> list[list[str]]:
    """Draw specular paths coloured by order; returns their legend entries."""
    legend = []
    top = int(paths.order.max()) if len(paths) else -1
    if max_order is not None:
        top = min(top, max_order)
    for order in range(top + 1):
        if not np.any(paths.order == order):
            continue
        colour = ORDER_COLOURS[min(order, len(ORDER_COLOURS) - 1)]
        plotter.add_mesh(
            paths_polydata(paths, order),
            color=colour,
            line_width=line_width if order else line_width * 1.5,
            opacity=1.0 if order < 2 else 0.8,
        )
        legend.append([f"{_order_name(order)} ({np.sum(paths.order == order)})", colour])
    return legend


def plot_room(
    room: Room,
    plotter: pyvista.Plotter | None = None,
    *,
    opacity: float = 0.45,
    show_labels: bool = True,
    paths: EarlyReflections | None = None,
    max_order: int | None = None,
) -> pyvista.Plotter:
    """Add the room to ``plotter`` (a new one if None) and return it.

    ``paths`` adds early reflection paths (from :func:`acoustic_spaces.engine.early_paths`
    or a response's ``early``), coloured by order up to ``max_order``.
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
    for source in room.scene.sources:
        plotter.add_mesh(pv.Sphere(radius=radius, center=source.position), color=SOURCE_COLOUR)
        if source.directivity != "omni":
            direction = source.orientation.forward()
            shape = pv.Arrow(start=source.position, direction=direction, scale=arrow)
            plotter.add_mesh(shape, color=SOURCE_COLOUR)
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
    plotter.add_legend(legend, bcolor=PANEL, face="rectangle", size=(0.22, 0.04 * len(legend)))
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
    GIF or, with imageio and ffmpeg installed, an MP4 video.

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
    if suffix == ".mp4":
        try:
            plotter.open_movie(str(path), framerate=fps)
        except ImportError as exc:
            plotter.close()
            raise ImportError(
                'MP4 needs imageio and ffmpeg: pip install "imageio[ffmpeg]"'
            ) from exc
    images = []
    for frame in range(len(particles.times)):
        show(frame)
        if suffix == ".mp4":
            plotter.write_frame()
        else:
            plotter.render()
            images.append(plotter.screenshot(return_img=True))
    plotter.close()
    if suffix == ".gif":
        from PIL import Image

        # One palette for every frame, so colours that stay put do not flicker.
        sample = np.concatenate([images[0], images[len(images) // 2], images[-1]])
        palette = Image.fromarray(sample).quantize(colors=255)
        frames = [Image.fromarray(image).quantize(palette=palette, dither=0) for image in images]
        frames[0].save(
            str(path), save_all=True, append_images=frames[1:], duration=1000 / fps, loop=0
        )


# -- response plots ----------------------------------------------------------------


def _figure(**kwargs):
    """A matplotlib figure that needs no display or pyplot backend."""
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            'response plots need matplotlib: pip install "acoustic-spaces[viz]"'
        ) from exc
    return Figure(**kwargs)


def _two_panels():
    """A dark figure with two stacked panels."""
    fig = _figure(figsize=(10, 7), layout="constrained")
    axes = fig.subplots(2, 1)
    fig.patch.set_facecolor(BACKGROUND[0])
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#6b6e7a")
        ax.grid(True, color="#3a3d4a", linewidth=0.6)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
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
    Returns the matplotlib figure.
    """
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
    from acoustic_spaces.parameters import decay_curve

    energy = response.energy()
    t_ms = np.arange(energy.shape[1]) * response.time_step * 1e3
    reference = 1.0 / (4.0 * np.pi * 100.0)  # free field at 10 m, as for G

    fig, (ax_e, ax_d) = _two_panels()

    with np.errstate(divide="ignore"):
        late_db = 10.0 * np.log10(np.maximum(response.late[band], 1e-30) / reference)
        early_db = 10.0 * np.log10(np.maximum(response.early.energy[:, band], 1e-30) / reference)
    late_t = np.arange(len(late_db)) * response.time_step * 1e3
    ax_e.fill_between(late_t, late_db, -100, step="post", color="#4da3ff", alpha=0.35, lw=0)
    ax_e.step(late_t, late_db, where="post", color="#4da3ff", lw=0.8, label="late (per ms)")
    floor = float(np.max(early_db, initial=np.max(late_db, initial=0.0))) - 70.0
    for order in np.unique(response.early.order):
        mask = response.early.order == order
        colour = ORDER_COLOURS[min(int(order), len(ORDER_COLOURS) - 1)]
        ax_e.vlines(response.early.delay_s[mask] * 1e3, floor, early_db[mask], color=colour, lw=1.2)
        ax_e.plot([], [], color=colour, label=_order_name(int(order)))
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
    the left to the right of the listener. Returns the matplotlib figure.
    """
    from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
    from acoustic_spaces.spatial.directions import from_angles, to_angles, to_head

    fig, (ax_e, ax_l) = _two_panels()
    reference = 1.0 / (4.0 * np.pi * 100.0)
    early = response.early
    if len(early):
        az, el = to_angles(to_head(early.direction, response.orientation))
        with np.errstate(divide="ignore"):
            level = 10.0 * np.log10(np.maximum(early.energy[:, band], 1e-30) / reference)
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
        energy = response.late_directional[:, band].sum(axis=1)
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
    from acoustic_spaces.parameters import band_decay, decay_curve

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

    step = 1e-3
    for b, f in enumerate(OCTAVE_BANDS_HZ):
        e = decay.cleaned(b)
        slots = len(e) // window
        if slots < 2:
            continue
        per_ms = e[: slots * window].reshape(slots, window).sum(axis=1)
        curve = decay_curve(per_ms, step)
        if not np.isfinite(curve[0]):
            continue
        label = f"{f} Hz" if f < 1000 else f"{f // 1000} kHz"
        ax_d.plot(np.arange(slots) * step * 1e3, curve, color=BAND_COLOURS[b], lw=1.2, label=label)
    ax_d.set_ylim(-70, 2)
    ax_d.set_xlim(0, max(t_ms[-1] - onset_ms, 1.0) if len(t_ms) else 1)
    ax_d.set_xlabel("time after the direct sound (ms)")
    ax_d.set_ylabel("decay (dB)")
    ax_d.set_title("backward-integrated decay per octave band, noise removed")
    _legend(ax_d, ncol=4)

    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig
