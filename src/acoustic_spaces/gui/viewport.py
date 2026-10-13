"""The 3D view of the editor: the room's surfaces coloured by material, sources and
listeners, the selection, and early reflection paths.

A click selects what is under the mouse (sources and listeners before the surfaces in
front of them). In placing mode a click on a surface asks for a new source or listener
there instead. Without OpenGL (a headless session) the view shows a note and the rest
of the editor works as usual.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from acoustic_spaces.geometry import points_inside
from acoustic_spaces.gui.document import LISTENER_HEIGHT, SOURCE_HEIGHT
from acoustic_spaces.room import UNASSIGNED, Room
from acoustic_spaces.viz import (
    BACKGROUND,
    EDGE_COLOUR,
    LISTENER_COLOUR,
    MATERIAL_COLOURS,
    PANEL,
    SOURCE_COLOUR,
    UNASSIGNED_COLOUR,
    _outline,
    balloon_polydata,
    plot_paths,
    room_polydata,
)

#: Colour of the selected item.
SELECTED_COLOUR = "#ffd166"
#: A click is a press and release less than this many pixels apart (else a drag).
CLICK_PIXELS = 4
#: Platforms without OpenGL windows.
HEADLESS_PLATFORMS = ("offscreen", "minimal")


def material_colour(room: Room, material: int) -> str:
    """The colour a material index of ``room`` is drawn in."""
    if material == UNASSIGNED:
        return UNASSIGNED_COLOUR
    return MATERIAL_COLOURS[material % len(MATERIAL_COLOURS)]


def surface_materials(room: Room) -> list[int]:
    """Material index per entry of ``room.mesh.labels`` (UNASSIGNED when none)."""
    out = [UNASSIGNED] * len(room.mesh.labels)
    for label, material in zip(room.mesh.face_labels, room.face_materials, strict=True):
        out[label] = int(material)
    return out


def placement_position(
    room: Room, point: np.ndarray, normal: np.ndarray, kind: str
) -> tuple[float, float, float]:
    """Where a source or listener (``kind`` "sources" or "listeners") goes when the
    user clicks ``point`` on a surface with ``normal``.

    On a floor (a surface facing up) it stands at the usual height above the point: a
    talker's mouth for sources, a seated ear for listeners. On a wall or ceiling it goes
    half a metre into the room from the point.
    """
    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    length = float(np.linalg.norm(normal))
    normal = normal / length if length > 0 else np.array([0.0, 0.0, 1.0])
    if abs(normal[2]) > 0.7 and point[2] < float(np.mean(room.mesh.vertices[:, 2])):
        height = SOURCE_HEIGHT if kind == "sources" else LISTENER_HEIGHT
        return tuple(float(v) for v in point + [0.0, 0.0, height])
    candidates = np.array([point + 0.5 * normal, point - 0.5 * normal])
    if room.report.closed:
        inside = points_inside(room.mesh, candidates)
        if inside.any():
            return tuple(float(v) for v in candidates[int(np.argmax(inside))])
    centre = room.mesh.vertices.mean(axis=0)
    best = int(np.argmin(np.linalg.norm(candidates - centre, axis=1)))
    return tuple(float(v) for v in candidates[best])


def opengl_available() -> bool:
    return QGuiApplication.platformName() not in HEADLESS_PLATFORMS


def _key(actor) -> str:
    """The same key for every Python wrapper of one VTK actor."""
    return actor.GetAddressAsString("vtkActor")


class Viewport(QWidget):
    """The room in 3D. Emits ``picked(kind, key)`` when something is clicked (kind
    "surface" with a label index, "sources" or "listeners" with an index, or "" for
    empty space) and ``place(kind, position)`` in placing mode."""

    picked = Signal(str, object)
    place = Signal(str, tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        self.note = ""
        self.placing: str | None = None
        self._room: Room | None = None
        self._mesh = None
        self._actors: dict[str, tuple[str, object]] = {}
        self._markers: list = []
        self._surfaces: list = []
        self._press = (0, 0)
        if opengl_available():
            try:
                from pyvistaqt import QtInteractor

                self.plotter = QtInteractor(self)
                layout.addWidget(self.plotter.interactor)
                self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
                self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)
                self.plotter.set_background(BACKGROUND[0], top=BACKGROUND[1])
                self.plotter.add_axes()
            except Exception as exc:  # no usable OpenGL context
                self.plotter = None
                self.note = f"The 3D view is not available here: {exc}"
        else:
            self.note = "The 3D view needs a display with OpenGL."
        if self.plotter is None:
            label = QLabel(self.note)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            layout.addWidget(label)

    @property
    def available(self) -> bool:
        return self.plotter is not None

    def close_view(self) -> None:
        if self.plotter is not None:
            self.plotter.close()

    # -- drawing --------------------------------------------------------------------

    def show_room(self, room: Room, selection: tuple[str, object] | None = None, paths=None):
        """Draw ``room`` with ``selection`` highlighted and, if given, early ``paths``."""
        self._room = room
        if self.plotter is None:
            return
        import pyvista as pv

        plotter = self.plotter
        new_geometry = room.mesh is not self._mesh
        camera = None if new_geometry else plotter.camera_position
        self._mesh = room.mesh
        plotter.clear()
        self._actors.clear()
        self._markers, self._surfaces = [], []
        kind, key = selection or ("", None)

        materials = surface_materials(room)
        for label, material in enumerate(materials):
            faces = np.flatnonzero(room.mesh.face_labels == label)
            if len(faces) == 0:
                continue
            part = room_polydata(room, faces)
            chosen = kind == "surface" and key == label
            colour = SELECTED_COLOUR if chosen else material_colour(room, material)
            opacity = 0.75 if chosen else 0.4 if material != UNASSIGNED else 0.8
            actor = plotter.add_mesh(part, color=colour, opacity=opacity, render=False)
            self._actors[_key(actor)] = ("surface", label)
            self._surfaces.append(actor)
            plotter.add_mesh(
                _outline(part),
                color=SELECTED_COLOUR if chosen else EDGE_COLOUR,
                line_width=3.0 if chosen else 1.5,
                render=False,
            )

        lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
        size = float(np.linalg.norm(hi - lo)) or 1.0
        radius = max(0.012 * size, 0.05)
        points, names = [], []
        for group, colour in (("sources", SOURCE_COLOUR), ("listeners", LISTENER_COLOUR)):
            for index, item in enumerate(getattr(room.scene, group)):
                chosen = kind == group and key == index
                scale = 1.6 if chosen else 1.0
                sphere = pv.Sphere(radius=radius * scale, center=item.position)
                actor = plotter.add_mesh(
                    sphere, color=SELECTED_COLOUR if chosen else colour, render=False
                )
                self._actors[_key(actor)] = (group, index)
                self._markers.append(actor)
                directivity = room.directivities[index] if group == "sources" else None
                if directivity is None or not directivity.omni:
                    arrow = pv.Arrow(
                        start=item.position, direction=item.orientation.forward(), scale=6 * radius
                    )
                    plotter.add_mesh(arrow, color=colour, render=False)
                if chosen and directivity is not None and not directivity.omni:
                    balloon = balloon_polydata(
                        directivity, item.position, item.orientation.rotation(), size=10 * radius
                    )
                    plotter.add_mesh(
                        balloon,
                        scalars="level",
                        cmap="magma",
                        clim=(-30, 0),
                        opacity=0.55,
                        show_scalar_bar=False,
                        pickable=False,
                        render=False,
                    )
                points.append(item.position)
                names.append(item.name)
        if points:
            plotter.add_point_labels(
                np.asarray(points, dtype=float) + [0.0, 0.0, 1.8 * radius],
                names,
                font_size=12,
                text_color="white",
                shape_color=PANEL,
                shape_opacity=0.8,
                show_points=False,
                always_visible=True,
                render=False,
            )
        if paths is not None:
            plot_paths(paths, plotter)
        if camera is None:
            plotter.view_isometric()
            plotter.reset_camera()
        else:
            plotter.camera_position = camera
        plotter.render()

    def reset_camera(self) -> None:
        if self.plotter is not None:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self.plotter.render()

    def screenshot(self, path) -> None:
        if self.plotter is not None:
            self.plotter.screenshot(str(path))

    # -- picking --------------------------------------------------------------------

    def start_placing(self, kind: str | None) -> None:
        """Make the next click on a surface ask for a new ``kind`` ("sources" or
        "listeners") there; None goes back to selecting."""
        self.placing = kind
        cursor = Qt.CursorShape.CrossCursor if kind else Qt.CursorShape.ArrowCursor
        if self.plotter is not None:
            self.plotter.interactor.setCursor(cursor)

    def _on_press(self, *_) -> None:
        self._press = self.plotter.iren.get_event_position()

    def _on_release(self, *_) -> None:
        x, y = self.plotter.iren.get_event_position()
        if max(abs(x - self._press[0]), abs(y - self._press[1])) > CLICK_PIXELS:
            return  # the camera was dragged
        self.pick(x, y)

    def pick(self, x: int, y: int) -> None:
        """Act on a click at display position (x, y), from the bottom left corner."""
        if self.plotter is None or self._room is None:
            return
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        renderer = self.plotter.renderer
        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.PickFromListOn()
        if self.placing is None:
            for actor in self._markers:
                picker.AddPickList(actor)
            if picker.Pick(x, y, 0, renderer) and picker.GetActor() is not None:
                self.picked.emit(*self._actors[_key(picker.GetActor())])
                return
            picker.InitializePickList()
        for actor in self._surfaces:
            picker.AddPickList(actor)
        if not picker.Pick(x, y, 0, renderer) or picker.GetActor() is None:
            if self.placing is None:
                self.picked.emit("", None)
            return
        if self.placing is not None:
            point = np.array(picker.GetPickPosition())
            normal = np.array(picker.GetPickNormal())
            kind, self.placing = self.placing, None
            self.start_placing(None)
            self.place.emit(kind, placement_position(self._room, point, normal, kind))
            return
        self.picked.emit(*self._actors[_key(picker.GetActor())])


__all__ = [
    "SELECTED_COLOUR",
    "Viewport",
    "material_colour",
    "opengl_available",
    "placement_position",
    "surface_materials",
]
