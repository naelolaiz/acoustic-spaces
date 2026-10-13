"""Pictures of rooms and responses.

- 3D view (PyVista): surfaces coloured by material, sources, listeners, and early
  reflection paths coloured by order.
- Response plots (matplotlib): echogram with the early paths, and decay curves per band.

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

    fig = _figure(figsize=(10, 7), layout="constrained")
    ax_e, ax_d = fig.subplots(2, 1)
    fig.patch.set_facecolor(BACKGROUND[0])
    for ax in (ax_e, ax_d):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#6b6e7a")
        ax.grid(True, color="#3a3d4a", linewidth=0.6)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

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
    legend = ax_e.legend(loc="upper right", facecolor=PANEL, edgecolor="#6b6e7a", fontsize=8)
    for text in legend.get_texts():
        text.set_color("white")

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
    legend = ax_d.legend(ncol=4, facecolor=PANEL, edgecolor="#6b6e7a", fontsize=8)
    for text in legend.get_texts():
        text.set_color("white")

    if path is not None:
        fig.savefig(str(path), dpi=120, facecolor=fig.get_facecolor())
    return fig
