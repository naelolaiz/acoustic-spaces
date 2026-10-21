"""The 3D view of the editor: the room's surfaces coloured by material, sources and
listeners, the selection, early reflection paths and a coverage map.

A click selects what is under the mouse (sources and listeners before the surfaces in
front of them). Dragging a source or listener moves it across the room, keeping its
height above the floor; with Shift it moves up and down instead. In placing mode a click
on a surface asks for a new source or listener there. Without OpenGL (a headless
session) the view shows a note and the rest of the editor works as usual.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from acoustic_spaces.geometry import points_inside
from acoustic_spaces.gui.document import LISTENER_HEIGHT, SOURCE_HEIGHT, follow_floor
from acoustic_spaces.i18n import _
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
    extent_polydata,
    plot_coverage,
    plot_leaks,
    plot_paths,
    plot_vegetation,
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
    empty space), ``moved(kind, index, position)`` when a source or listener was
    dragged, and ``place(kind, position)`` in placing mode."""

    picked = Signal(str, object)
    moved = Signal(str, int, tuple)
    place = Signal(str, tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        self.note = ""
        self.placing: str | None = None
        self._room: Room | None = None
        self._shown = False  # a room was drawn: the camera stays where it is
        self._actors: dict[str, tuple[str, object]] = {}
        self._markers: list = []
        self._surfaces: list = []
        self._point_actors: dict[tuple[str, int], list] = {}
        self._press = (0, 0)
        self._drag: dict | None = None
        if opengl_available():
            try:
                from pyvistaqt import QtInteractor

                self.plotter = QtInteractor(self)
                layout.addWidget(self.plotter.interactor)
                self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_press)
                self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_release)
                # Drags of sources and listeners never reach VTK, so the camera stays.
                self.plotter.interactor.installEventFilter(self)
                self.plotter.set_background(BACKGROUND[0], top=BACKGROUND[1])
                self.plotter.add_axes()
            except Exception as exc:  # no usable OpenGL context
                self.plotter = None
                self.note = _("The 3D view is not available here: {reason}").format(reason=exc)
        else:
            self.note = _("The 3D view needs a display with OpenGL.")
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

    def show_room(
        self,
        room: Room,
        selection: tuple[str, object] | None = None,
        paths=None,
        coverage=None,
        leaks=None,
        hidden: set[tuple[str, str]] = frozenset(),
    ):
        """Draw ``room`` with ``selection`` highlighted and, if given, early ``paths``,
        a coverage map (``coverage`` is (CoverageMap, parameter, band)) and the
        ``leaks`` from :func:`acoustic_spaces.engine.leaks.find_leaks`. The sources and
        listeners in ``hidden`` ((kind, name) pairs) are left out; those left out of
        simulations are drawn faded.

        The camera stays where it is, also when the room changes shape; only the first
        room gets the usual view (:meth:`reset_camera` gives it again)."""
        self._room = room
        if self.plotter is None:
            return
        import pyvista as pv

        plotter = self.plotter
        camera = plotter.camera_position if self._shown else None
        self._shown = True
        plotter.clear()
        plotter.enable_lightkit()  # clear() takes the lights too, leaving the room unlit
        self._actors.clear()
        self._markers, self._surfaces = [], []
        self._point_actors = {}
        self._drag = None
        kind, key = selection or ("", None)

        materials = surface_materials(room)
        sky = room.sky_faces()
        for label, material in enumerate(materials):
            faces = np.flatnonzero(room.mesh.face_labels == label)
            if len(faces) == 0:
                continue
            part = room_polydata(room, faces)
            if sky[faces[0]]:
                # Outdoors, the open sky would hide the rest: only its edges, where the
                # ground ends.
                plotter.add_mesh(_outline(part), color=EDGE_COLOUR, line_width=1.0, render=False)
                continue
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
        plot_vegetation(room, plotter, render=False)

        lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
        size = float(np.linalg.norm(hi - lo)) or 1.0
        radius = max(0.012 * size, 0.05)
        points, names = [], []
        for group, colour in (("sources", SOURCE_COLOUR), ("listeners", LISTENER_COLOUR)):
            for index, item in enumerate(getattr(room.scene, group)):
                if (group, item.name) in hidden:
                    continue
                chosen = kind == group and key == index
                scale = 1.6 if chosen else 1.0
                # Left out of simulations: faded.
                opacity = 1.0 if item.in_use or chosen else 0.3
                sphere = pv.Sphere(radius=radius * scale, center=item.position)
                actor = plotter.add_mesh(
                    sphere,
                    color=SELECTED_COLOUR if chosen else colour,
                    opacity=opacity,
                    render=False,
                )
                self._actors[_key(actor)] = (group, index)
                self._markers.append(actor)
                parts = self._point_actors[group, index] = [actor]
                extent = extent_polydata(item, 0.5 * radius) if group == "sources" else None
                if extent is not None:  # a line or surface source
                    shape = plotter.add_mesh(
                        extent,
                        color=SELECTED_COLOUR if chosen else colour,
                        opacity=0.6 * opacity,
                        render=False,
                    )
                    self._actors[_key(shape)] = (group, index)
                    self._markers.append(shape)
                    parts.append(shape)
                directivity = room.directivities[index] if group == "sources" else None
                if directivity is None or not directivity.omni:
                    arrow = pv.Arrow(
                        start=item.position,
                        direction=item.orientation.forward(),
                        scale=6 * radius * scale,
                    )
                    parts.append(
                        plotter.add_mesh(
                            arrow,
                            color=SELECTED_COLOUR if chosen else colour,
                            opacity=opacity,
                            render=False,
                        )
                    )
                    if chosen:  # a short arrow to the top of the head shows the roll
                        up = np.array(item.orientation.rotation())[:, 2]
                        top = pv.Arrow(start=item.position, direction=up, scale=3 * radius * scale)
                        parts.append(plotter.add_mesh(top, color=SELECTED_COLOUR, render=False))
                if chosen and directivity is not None and not directivity.omni:
                    balloon = balloon_polydata(
                        directivity, item.position, item.orientation.rotation(), size=10 * radius
                    )
                    parts.append(
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
                    )
                # The label of the selected point goes above its arrow to the top.
                lift = (3 * scale + 1.8 if chosen else 1.8) * radius
                points.append(np.add(item.position, [0.0, 0.0, lift]))
                names.append(item.name)
        if points:
            plotter.add_point_labels(
                np.asarray(points, dtype=float),
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
        if coverage is not None:
            shown, parameter, band = coverage
            plot_coverage(shown, plotter, parameter=parameter, band=band, render=False)
        if leaks is not None:
            plot_leaks(leaks, room, plotter, render=False)
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

    # -- dragging sources and listeners ------------------------------------------------

    def _display(self, widget: QWidget, event: QMouseEvent) -> tuple[int, int]:
        """VTK display coordinates (pixels from the bottom left) of a mouse event."""
        ratio = widget.devicePixelRatioF()
        position = event.position()
        return round(position.x() * ratio), round((widget.height() - position.y() - 1) * ratio)

    def marker_at(self, x: int, y: int) -> tuple[str, int] | None:
        """The source or listener at display position (x, y), if any."""
        if self.plotter is None or not self._markers:
            return None
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.PickFromListOn()
        for actor in self._markers:
            picker.AddPickList(actor)
        if picker.Pick(x, y, 0, self.plotter.renderer) and picker.GetActor() is not None:
            return self._actors[_key(picker.GetActor())]
        return None

    def _ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """The line of sight through display position (x, y): a point and a direction."""
        renderer = self.plotter.renderer
        ends = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(x, y, depth)
            renderer.DisplayToWorld()
            world = np.array(renderer.GetWorldPoint(), dtype=float)
            ends.append(world[:3] / world[3])
        direction = ends[1] - ends[0]
        return ends[0], direction / max(float(np.linalg.norm(direction)), 1e-300)

    def drag_position(self, start, x: float, y: float, vertical: bool = False):
        """Where a source or listener at ``start`` goes when dragged to display position
        (x, y): across the room, following the floor (see
        :func:`~acoustic_spaces.gui.document.follow_floor`), or straight up and down."""
        start = np.asarray(start, dtype=float)
        origin, direction = self._ray(x, y)
        if vertical:
            # The vertical plane through the point that faces the camera.
            normal = np.array([direction[0], direction[1], 0.0])
            if np.linalg.norm(normal) < 1e-6:
                return tuple(float(v) for v in start)
            normal /= np.linalg.norm(normal)
        else:
            normal = np.array([0.0, 0.0, 1.0])
        along = float(np.dot(direction, normal))
        if abs(along) < 1e-9:
            return tuple(float(v) for v in start)
        t = float(np.dot(start - origin, normal)) / along
        if t <= 0:
            return tuple(float(v) for v in start)
        hit = origin + t * direction
        if vertical:
            return (float(start[0]), float(start[1]), float(hit[2]))
        return follow_floor(self._room.mesh, start, hit[:2])

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if self.plotter is None or watched is not self.plotter.interactor:
            return False
        if kind == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.MouseButton.LeftButton or self.placing is not None:
                return False
            x, y = self._display(watched, event)
            target = self.marker_at(x, y)
            if target is None:
                return False
            group, index = target
            start = getattr(self._room.scene, group)[index].position
            self._drag = {"target": target, "press": (x, y), "start": start, "position": None}
            return True
        if self._drag is None:
            return False
        if kind == QEvent.Type.MouseMove:
            x, y = self._display(watched, event)
            px, py = self._drag["press"]
            if self._drag["position"] is None and max(abs(x - px), abs(y - py)) <= CLICK_PIXELS:
                return True
            vertical = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            position = self.drag_position(self._drag["start"], x, y, vertical)
            self._drag["position"] = position
            offset = np.subtract(position, self._drag["start"])
            for actor in self._point_actors.get(self._drag["target"], []):
                actor.SetPosition(*offset)
            self.plotter.render()
            return True
        if kind == QEvent.Type.MouseButtonRelease:
            drag, self._drag = self._drag, None
            group, index = drag["target"]
            if drag["position"] is None:
                self.picked.emit(group, index)
            else:
                self.moved.emit(group, index, tuple(drag["position"]))
            return True
        return False

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
