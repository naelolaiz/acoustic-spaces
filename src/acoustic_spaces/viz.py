"""3D view of a room: surfaces coloured by material, sources and listeners.

Needs PyVista (``pip install "acoustic-spaces[viz]"``). Works in a desktop window,
in Jupyter, and off screen for screenshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from acoustic_spaces.room import UNASSIGNED, Room

if TYPE_CHECKING:
    import pyvista

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


def plot_room(
    room: Room,
    plotter: pyvista.Plotter | None = None,
    *,
    opacity: float = 0.45,
    show_labels: bool = True,
) -> pyvista.Plotter:
    """Add the room to ``plotter`` (a new one if None) and return it."""
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
    """Open the room in a window, or save a PNG to ``screenshot`` without one."""
    pv = _pyvista()
    plotter = pv.Plotter(off_screen=screenshot is not None, window_size=list(window_size))
    plot_room(room, plotter, **kwargs)
    if screenshot is not None:
        plotter.screenshot(str(screenshot))
        plotter.close()
    else:
        plotter.show()
