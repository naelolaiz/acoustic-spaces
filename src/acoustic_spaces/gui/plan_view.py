"""The plan: the room seen from above, to draw floor plans, move their corners and move
sources and listeners by dragging.

Floor plan and box rooms show their corners as handles. Dragging one moves it (a box
becomes a floor plan with the same walls), double-clicking a wall adds a corner there,
and the right-click menu removes one. *Draw room* starts a new floor plan: click its
corners in turn and close it on the first one, with a double-click or with Enter.
Rooms from model files show their outline, and a drawn room replaces the model.

Sources and listeners move by dragging and keep their height above the floor. Corners
and points snap to the grid; Shift moves them freely. A click selects what is under the
mouse, as in the 3D view; the wheel zooms and dragging empty space pans.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.geometry import plan_problem
from acoustic_spaces.gui.document import (
    LISTENER_HEIGHT,
    SOURCE_HEIGHT,
    follow_floor,
    plan_corners,
    surface_heights,
)
from acoustic_spaces.gui.viewport import SELECTED_COLOUR, material_colour, surface_materials
from acoustic_spaces.room import Room
from acoustic_spaces.viz import BACKGROUND, EDGE_COLOUR, LISTENER_COLOUR, SOURCE_COLOUR

#: Grid steps the corners and points snap to (label, metres; 0 is off).
SNAP_STEPS = (
    ("Off", 0.0),
    ("1 cm", 0.01),
    ("5 cm", 0.05),
    ("10 cm", 0.1),
    ("25 cm", 0.25),
    ("50 cm", 0.5),
    ("1 m", 1.0),
)
DEFAULT_SNAP = 0.1
#: How near (pixels) the mouse must be to a corner or point to take it, and to a wall.
HANDLE_PIXELS = 8
WALL_PIXELS = 5
#: A click is a press and release less than this many pixels apart (else a drag).
CLICK_PIXELS = 4
#: The grid's labelled lines are at least this many pixels apart.
GRID_PIXELS = 60
#: Space (pixels) kept round the room when it is fitted into the view.
MARGIN_PIXELS = 40
#: Colour of a floor plan that cannot be (walls that cross).
INVALID_COLOUR = "#ef476f"
GRID_COLOURS = ("#1d1e25", "#2b2d37", "#4a4d5a")  # minor, major, axes

HINTS = {
    "plan": "Drag corners, sources and listeners; double-click a wall to add a corner; "
    "right-click for more. Shift: no snapping.",
    "box": "Drag a corner to reshape the box (it becomes a floor plan); drag sources and "
    "listeners. Shift: no snapping.",
    "file": "The model's outline from above: drag sources and listeners. Draw room "
    "replaces the model with a floor plan.",
    "drawing": "Click the corners of the new room; close it on the first corner, with a "
    "double-click or Enter. Backspace takes a corner back, Esc stops.",
    "placing": "Click where the new {what} goes; Esc cancels.",
}


def snap(point, step: float) -> np.ndarray:
    """``point`` on the nearest grid crossing ``step`` metres apart (unchanged for 0)."""
    point = np.asarray(point, dtype=float)
    if step <= 0:
        return point
    return np.round(point / step) * step + 0.0


def grid_step(pixels_per_metre: float, pixels: float = GRID_PIXELS) -> float:
    """The smallest of 1, 2 or 5 times a power of ten that is at least ``pixels`` long."""
    target = pixels / max(pixels_per_metre, 1e-12)
    power = 10.0 ** math.floor(math.log10(target))
    for factor in (1.0, 2.0, 5.0, 10.0):
        if factor * power >= target * (1 - 1e-9):
            return factor * power
    return 10.0 * power  # pragma: no cover - the loop always returns


def wall_labels(room: Room) -> list[int]:
    """For each wall of a floor plan or box room (``wall_1``, ``wall_2``...), the index
    of its entry in ``room.mesh.labels`` (-1 when the mesh has none)."""
    corners = plan_corners(room.scene)
    if corners is None:
        return []
    index = {label[0]: k for k, label in enumerate(room.mesh.labels)}
    return [index.get(f"wall_{k + 1}", -1) for k in range(len(corners))]


def outline_segments(room: Room) -> tuple[np.ndarray, np.ndarray]:
    """The model's edges seen from above: (E, 2, 2) end points of the edges where the
    surface folds or ends, and (E,) the label of a face beside each."""
    mesh = room.mesh
    faces = mesh.faces
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    unique, start, count = np.unique(edges, axis=0, return_index=True, return_counts=True)
    normals = mesh.face_normals()
    keep = count != 2
    pairs = count == 2
    first, second = owner[start[pairs]], owner[start[pairs] + 1]
    fold = np.abs(np.sum(normals[first] * normals[second], axis=1)) < math.cos(math.radians(10))
    keep[np.flatnonzero(pairs)[fold]] = True
    points = mesh.vertices[unique[keep]][:, :, :2]
    # Edges that are points from above (vertical edges) add nothing.
    long = np.linalg.norm(points[:, 1] - points[:, 0], axis=1) > 1e-9
    labels = mesh.face_labels[owner[start[keep]]]
    return points[long], labels[long]


def _qcolor(colour: str, alpha: int = 255) -> QColor:
    out = QColor(colour)
    out.setAlpha(alpha)
    return out


def _distance_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Distance from ``p`` to the segment a-b, and where along it (0 to 1) the nearest
    point is."""
    d = b - a
    t = float(np.clip(np.dot(p - a, d) / max(float(np.dot(d, d)), 1e-300), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d))), t


class PlanCanvas(QWidget):
    """The drawing itself; :class:`PlanView` adds the bar above it."""

    #: A click selected something: like :class:`~acoustic_spaces.gui.viewport.Viewport`.
    picked = Signal(str, object)
    #: New corners (m) for the floor plan, and what the change was.
    corners_changed = Signal(list, str)
    #: A source or listener ("sources" or "listeners", its index) was dragged here (m).
    moved = Signal(str, int, tuple)
    #: A new floor plan was drawn: its corners (m).
    drawn = Signal(list)
    #: Put a new source or listener here (m).
    place = Signal(str, tuple)
    #: Something to tell the user (a refused drag, a drawing that crosses itself).
    message = Signal(str)
    #: Drawing or placing started or stopped, or the cursor moved: the bar's text.
    state_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.room: Room | None = None
        self.selection: tuple[str, object] = ("", None)
        self.snap_step = DEFAULT_SNAP
        self.drawing: list[np.ndarray] | None = None
        self.placing: str | None = None
        self.cursor_position: np.ndarray | None = None
        self.corners: np.ndarray | None = None
        self._mesh = None
        self._walls: list[int] = []
        self._outline = (np.zeros((0, 2, 2)), np.zeros(0, dtype=np.int64))
        self._scale = 40.0  # pixels per metre
        self._centre = np.zeros(2)  # the world point in the middle of the widget
        self._needs_fit = True
        self._user_view = False  # zoomed or panned: resizing keeps the view
        self._materials: list[int] = []
        self._press: tuple[QPointF, object] | None = None
        self._drag: tuple | None = None  # ("corner", k) / (kind, index) / ("pan",)
        self._preview: np.ndarray | None = None
        self._hover: tuple | None = None

    # -- contents -------------------------------------------------------------------

    def show_room(self, room: Room, selection: tuple[str, object] | None = None) -> None:
        """Draw ``room`` with ``selection`` highlighted."""
        new_geometry = room.mesh is not self._mesh
        self.room = room
        self.selection = selection or ("", None)
        self.corners = plan_corners(room.scene)
        self._walls = wall_labels(room)
        self._materials = surface_materials(room)
        if new_geometry:
            self._mesh = room.mesh
            self._outline = outline_segments(room)
        if self._needs_fit:
            self.fit()
        self.update()

    @property
    def mode(self) -> str:
        """What the plan is doing: "drawing", "placing", or the kind of room ("plan",
        "box" or "file")."""
        if self.drawing is not None:
            return "drawing"
        if self.placing is not None:
            return "placing"
        if self.room is None:
            return "file"
        geometry = self.room.scene.geometry
        return "plan" if geometry.floor_plan is not None else "box" if geometry.box else "file"

    def fit(self) -> None:
        """Show the whole room and every source and listener; resizing the view keeps
        them in it until the view is zoomed or panned."""
        self._user_view = False
        if self.room is None:
            return
        points = [self.room.mesh.vertices[:, :2]]
        for kind in ("sources", "listeners"):
            points.append(np.array([p.position[:2] for p in getattr(self.room.scene, kind)]))
        if self.drawing:
            points.append(np.array(self.drawing))
        points = np.vstack(points)
        lo, hi = points.min(axis=0), points.max(axis=0)
        size = np.maximum(hi - lo, 1e-3)
        width = max(self.width() - 2 * MARGIN_PIXELS, 50)
        height = max(self.height() - 2 * MARGIN_PIXELS, 50)
        self._scale = float(min(width / size[0], height / size[1]))
        self._centre = (lo + hi) / 2
        self._needs_fit = False
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_view:
            self.fit()

    # -- coordinates ----------------------------------------------------------------

    def to_screen(self, point) -> QPointF:
        x, y = (float(v) for v in point[:2])
        return QPointF(
            self.width() / 2 + (x - self._centre[0]) * self._scale,
            self.height() / 2 - (y - self._centre[1]) * self._scale,
        )

    def to_world(self, position: QPointF) -> np.ndarray:
        return np.array(
            [
                self._centre[0] + (position.x() - self.width() / 2) / self._scale,
                self._centre[1] - (position.y() - self.height() / 2) / self._scale,
            ]
        )

    def _snapped(self, position: QPointF, free: bool = False) -> np.ndarray:
        return snap(self.to_world(position), 0.0 if free else self.snap_step)

    def _pixels(self, a, b: QPointF) -> float:
        p = self.to_screen(a)
        return math.hypot(p.x() - b.x(), p.y() - b.y())

    # -- what is under the mouse ------------------------------------------------------

    def _points(self):
        """(kind, index, item) of every source and listener."""
        if self.room is None:
            return []
        return [
            (kind, k, item)
            for kind in ("sources", "listeners")
            for k, item in enumerate(getattr(self.room.scene, kind))
        ]

    def hit(self, position: QPointF) -> tuple | None:
        """What is at ``position`` (widget pixels): ("sources" or "listeners", index),
        ("corner", k), ("wall", k, t along it), or None."""
        best, nearest = None, HANDLE_PIXELS
        for kind, k, item in self._points():
            d = self._pixels(item.position, position)
            if d <= nearest:
                best, nearest = (kind, k), d
        if best is not None:
            return best
        if self.corners is None:
            return None
        for k, corner in enumerate(self.corners):
            d = self._pixels(corner, position)
            if d <= nearest:
                best, nearest = ("corner", k), d
        if best is not None:
            return best
        p = np.array([position.x(), position.y()])
        nearest = WALL_PIXELS
        n = len(self.corners)
        for k in range(n):
            a, b = self.to_screen(self.corners[k]), self.to_screen(self.corners[(k + 1) % n])
            d, t = _distance_to_segment(p, np.array([a.x(), a.y()]), np.array([b.x(), b.y()]))
            if d <= nearest:
                best, nearest = ("wall", k, t), d
        return best

    def _surface_at(self, position: QPointF) -> int | None:
        """The label of the model's surface whose outline is nearest ``position``."""
        segments, labels = self._outline
        if len(segments) == 0:
            return None
        p = self.to_world(position)
        a, b = segments[:, 0], segments[:, 1]
        d = b - a
        t = np.clip(np.sum((p - a) * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-300), 0, 1)
        distance = np.linalg.norm(p - (a + t[:, None] * d), axis=1) * self._scale
        k = int(np.argmin(distance))
        return int(labels[k]) if distance[k] <= WALL_PIXELS else None

    # -- drawing a room and placing points --------------------------------------------

    def start_drawing(self) -> None:
        """Start drawing a new floor plan, corner by corner."""
        self.placing = None
        self.drawing = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.state_changed.emit()
        self.update()

    def cancel(self) -> bool:
        """Stop drawing, placing or dragging; True if there was something to stop."""
        busy = self.drawing is not None or self.placing is not None or self._drag is not None
        self.drawing = self.placing = None
        self._drag = self._preview = self._press = None
        self.unsetCursor()
        self.state_changed.emit()
        self.update()
        return busy

    def add_drawn_corner(self, point) -> None:
        """Add a corner to the room being drawn; a click on the first corner closes it."""
        if self.drawing is None:
            return
        point = np.asarray(point, dtype=float)
        if len(self.drawing) >= 3 and self._pixels(self.drawing[0], self.to_screen(point)) <= (
            HANDLE_PIXELS
        ):
            self.finish_drawing()
            return
        if self.drawing and np.allclose(point, self.drawing[-1]):
            return
        self.drawing.append(point)
        self.update()

    def undo_drawn_corner(self) -> None:
        if self.drawing:
            self.drawing.pop()
            self.update()

    def finish_drawing(self) -> bool:
        """Close the room being drawn and emit it; False (and a message) if it cannot be
        a floor plan, and drawing goes on."""
        if self.drawing is None:
            return False
        corners = np.array(self.drawing, dtype=float).reshape(-1, 2)
        problem = plan_problem(corners)
        if problem is not None:
            self.message.emit(f"Not a room yet: {problem}")
            return False
        self.drawing = None
        self.unsetCursor()
        self.state_changed.emit()
        self.drawn.emit([[float(x), float(y)] for x, y in corners])
        return True

    def start_placing(self, kind: str | None) -> None:
        """Make the next click put a new ``kind`` ("sources" or "listeners") there."""
        self.drawing = None
        self.placing = kind
        if kind is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.state_changed.emit()

    def placement(self, kind: str, point) -> tuple[float, float, float]:
        """Where a ``kind`` placed at plan position ``point`` goes: at the usual height
        above the floor seen from above there (a balcony over the stalls), or above the
        lowest surface when the model's faces do not say which way is in."""
        x, y = (float(v) for v in point)
        usual = SOURCE_HEIGHT if kind == "sources" else LISTENER_HEIGHT
        if self.room is None:
            return (x, y, usual)
        floors = surface_heights(self.room.mesh, x, y, up=True)
        if self.room.report.normals_inward and len(floors):
            floor = float(floors.max())
        else:
            heights = surface_heights(self.room.mesh, x, y)
            floor = float(heights.min()) if len(heights) else float(self.room.report.bounds_min[2])
        return (x, y, floor + usual)

    # -- editing the floor plan -------------------------------------------------------

    def add_corner(self, wall: int, point) -> None:
        """Split ``wall`` with a new corner at ``point``."""
        if self.corners is None:
            return
        corners = [list(map(float, c)) for c in self.corners]
        corners.insert(wall + 1, [float(v) for v in point])
        self.corners_changed.emit(corners, "add corner")

    def remove_corner(self, k: int) -> None:
        if self.corners is None:
            return
        if len(self.corners) <= 3:
            self.message.emit("A floor plan needs at least three corners.")
            return
        corners = [list(map(float, c)) for i, c in enumerate(self.corners) if i != k]
        self.corners_changed.emit(corners, "remove corner")

    # -- mouse and keys -----------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._press, self._drag = (position, None), ("pan",)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.drawing is not None:
            free = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.add_drawn_corner(self._snapped(position, free))
            return
        if self.placing is not None:
            kind = self.placing
            self.start_placing(None)
            free = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.place.emit(kind, self.placement(kind, self._snapped(position, free)))
            return
        self._press = (position, self.hit(position))
        self._drag = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        self.cursor_position = self.to_world(position)
        free = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self._press is not None:
            start, target = self._press
            moved = math.hypot(position.x() - start.x(), position.y() - start.y())
            if self._drag is None and moved > CLICK_PIXELS:
                self._drag = ("pan",) if target is None or target[0] == "wall" else target
            if self._drag is not None:
                self._dragging(position, free)
        else:
            hover = self.hit(position) if self.drawing is None else None
            if hover != self._hover:
                self._hover = hover
                handle = hover is not None and hover[0] != "wall"
                if self.drawing is None and self.placing is None:
                    if handle:
                        self.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        self.unsetCursor()
        self.state_changed.emit()
        self.update()

    def _dragging(self, position: QPointF, free: bool) -> None:
        start, _ = self._press
        kind = self._drag[0]
        if kind == "pan":
            self._user_view = True
            dx, dy = position.x() - start.x(), position.y() - start.y()
            self._centre = self._centre - np.array([dx, -dy]) / self._scale
            self._press = (position, self._press[1])
            return
        point = self._snapped(position, free)
        if kind == "corner":
            self._preview = self.corners.copy()
            self._preview[self._drag[1]] = point
        else:
            self._preview = point
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press is None:
            return
        _, target = self._press
        drag, preview = self._drag, self._preview
        self._press = self._drag = self._preview = None
        if self._hover is None:
            self.unsetCursor()
        if drag is None:
            self._click(event.position(), target)
        elif drag[0] == "corner" and preview is not None:
            problem = plan_problem(preview)
            if problem is not None:
                self.message.emit(f"Not moved: {problem}")
            else:
                self.corners_changed.emit(preview.tolist(), "move corner")
        elif drag[0] in ("sources", "listeners") and preview is not None:
            kind, index = drag
            item = getattr(self.room.scene, kind)[index]
            self.moved.emit(kind, index, follow_floor(self.room.mesh, item.position, preview))
        self.update()

    def _click(self, position: QPointF, target) -> None:
        if target is None:
            label = self._surface_at(position) if self.corners is None else None
            self.picked.emit(*(("surface", label) if label is not None else ("", None)))
        elif target[0] == "wall":
            label = self._walls[target[1]] if target[1] < len(self._walls) else -1
            self.picked.emit(*(("surface", label) if label >= 0 else ("", None)))
        elif target[0] in ("sources", "listeners"):
            self.picked.emit(target[0], target[1])

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.drawing is not None:
            self.finish_drawing()
            return
        target = self.hit(event.position())
        if target is not None and target[0] == "wall":
            free = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.add_corner(target[1], self._snapped(event.position(), free))

    def context_menu(self, position: QPointF) -> QMenu | None:
        """The right-click menu at ``position``: remove a corner, add one on a wall,
        place a source or listener."""
        if self.drawing is not None or self.room is None:
            return None
        target = self.hit(position)
        point = self._snapped(position)
        menu = QMenu(self)
        if target is not None and target[0] == "corner":
            remove = menu.addAction("Remove corner")
            remove.setEnabled(len(self.corners) > 3)
            remove.triggered.connect(lambda *_: self.remove_corner(target[1]))
        elif target is not None and target[0] == "wall":
            menu.addAction("Add corner here").triggered.connect(
                lambda *_: self.add_corner(target[1], point)
            )
        if not menu.isEmpty():
            menu.addSeparator()
        for kind, text in (("sources", "Place source here"), ("listeners", "Place listener here")):
            menu.addAction(text).triggered.connect(
                lambda *_, kind=kind: self.place.emit(kind, self.placement(kind, point))
            )
        return menu

    def contextMenuEvent(self, event) -> None:
        menu = self.context_menu(QPointF(event.pos()))
        if menu is not None:
            menu.exec(event.globalPos())
            menu.deleteLater()

    def wheelEvent(self, event: QWheelEvent) -> None:
        position = event.position()
        before = self.to_world(position)
        factor = 1.0015 ** event.angleDelta().y()
        self._user_view = True
        self._scale = float(np.clip(self._scale * factor, 1e-3, 1e5))
        self._centre = self._centre + before - self.to_world(position)
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if self.drawing is not None and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finish_drawing()
        elif self.drawing is not None and key == Qt.Key.Key_Backspace:
            self.undo_drawn_corner()
        elif key == Qt.Key.Key_Escape and self.cancel():
            pass
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:
        self.cursor_position = None
        self.state_changed.emit()
        super().leaveEvent(event)

    # -- painting -----------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BACKGROUND[0]))
        self._paint_grid(painter)
        if self.room is not None:
            if self.corners is not None:
                self._paint_plan(painter)
            else:
                self._paint_outline(painter)
            self._paint_points(painter)
        if self.drawing is not None:
            self._paint_drawing(painter)
        painter.end()

    def _paint_grid(self, painter: QPainter) -> None:
        major = grid_step(self._scale)
        minor = major / 5
        lo = self.to_world(QPointF(0, self.height()))
        hi = self.to_world(QPointF(self.width(), 0))
        font = QFont(self.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(font.pointSizeF() * 0.85, 6.0))
        painter.setFont(font)
        for step, colour in ((minor, GRID_COLOURS[0]), (major, GRID_COLOURS[1])):
            if step * self._scale < 6:
                continue
            painter.setPen(QPen(QColor(colour), 1))
            for x in np.arange(math.floor(lo[0] / step) * step, hi[0] + step, step):
                sx = self.to_screen((x, 0)).x()
                painter.drawLine(QPointF(sx, 0), QPointF(sx, self.height()))
            for y in np.arange(math.floor(lo[1] / step) * step, hi[1] + step, step):
                sy = self.to_screen((0, y)).y()
                painter.drawLine(QPointF(0, sy), QPointF(self.width(), sy))
        origin = self.to_screen((0, 0))
        painter.setPen(QPen(QColor(GRID_COLOURS[2]), 1))
        painter.drawLine(QPointF(origin.x(), 0), QPointF(origin.x(), self.height()))
        painter.drawLine(QPointF(0, origin.y()), QPointF(self.width(), origin.y()))
        painter.setPen(QColor("#8a8d99"))
        for x in np.arange(math.ceil(lo[0] / major) * major, hi[0], major):
            sx = self.to_screen((x, 0)).x()
            if sx < self.width() - 40:  # clear of the unit
                painter.drawText(QPointF(sx + 3, self.height() - 4), f"{round(x, 9) + 0.0:g}")
        for y in np.arange(math.ceil(lo[1] / major) * major, hi[1], major):
            sy = self.to_screen((0, y)).y()
            if sy > 16:
                painter.drawText(QPointF(3, sy - 3), f"{round(y, 9) + 0.0:g}")
        painter.drawText(QPointF(self.width() - 18, self.height() - 4), "m")

    def _material_of(self, label: int) -> str:
        if 0 <= label < len(self._materials):
            return material_colour(self.room, self._materials[label])
        return EDGE_COLOUR

    def _paint_plan(self, painter: QPainter) -> None:
        corners = self.corners
        dragging = (
            self._drag is not None and self._drag[0] == "corner" and self._preview is not None
        )
        if dragging:
            corners = self._preview
        valid = not dragging or plan_problem(corners) is None
        polygon = QPolygonF([self.to_screen(c) for c in corners])
        floor = next(
            (k for k, label in enumerate(self.room.mesh.labels) if label[0] == "floor"), -1
        )
        painter.setPen(Qt.PenStyle.NoPen)
        fill = self._material_of(floor) if valid else INVALID_COLOUR
        painter.setBrush(_qcolor(fill, 45))
        painter.drawPolygon(polygon)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        kind, key = self.selection
        n = len(corners)
        font = painter.font()
        for k in range(n):
            label = self._walls[k] if k < len(self._walls) else -1
            chosen = kind == "surface" and key == label and label >= 0
            colour = SELECTED_COLOUR if chosen else self._material_of(label)
            if not valid:
                colour = INVALID_COLOUR
            a, b = self.to_screen(corners[k]), self.to_screen(corners[(k + 1) % n])
            painter.setPen(QPen(QColor(colour), 4 if chosen else 3))
            painter.drawLine(a, b)
            hovered = self._hover is not None and self._hover[0] == "wall" and self._hover[1] == k
            if dragging and self._drag[1] in (k, (k + 1) % n) or hovered:
                length = float(np.linalg.norm(corners[(k + 1) % n] - corners[k]))
                self._label(painter, (a + b) / 2, f"{length:.2f} m", "#ffffff")
            elif math.hypot(b.x() - a.x(), b.y() - a.y()) > 70:
                painter.setFont(font)
                self._label(painter, (a + b) / 2, f"wall_{k + 1}", "#8a8d99")
        for k, corner in enumerate(corners):
            p = self.to_screen(corner)
            active = dragging and self._drag[1] == k
            hovered = self._hover == ("corner", k)
            size = 5 if active or hovered else 3.5
            painter.setPen(QPen(QColor(BACKGROUND[0]), 1))
            painter.setBrush(QColor(SELECTED_COLOUR if active else "#ffffff"))
            painter.drawRect(QRectF(p.x() - size, p.y() - size, 2 * size, 2 * size))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_outline(self, painter: QPainter) -> None:
        segments, labels = self._outline
        kind, key = self.selection
        path, chosen = QPainterPath(), QPainterPath()
        for (a, b), label in zip(segments, labels, strict=True):
            target = chosen if kind == "surface" and key == int(label) else path
            target.moveTo(self.to_screen(a))
            target.lineTo(self.to_screen(b))
        painter.setPen(QPen(_qcolor(EDGE_COLOUR, 120), 1))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(SELECTED_COLOUR), 2))
        painter.drawPath(chosen)

    def _paint_points(self, painter: QPainter) -> None:
        kind, key = self.selection
        drag = (
            self._drag
            if self._drag is not None and self._drag[0] in ("sources", "listeners")
            else None
        )
        for group, k, item in self._points():
            chosen = kind == group and key == k
            position = np.asarray(item.position[:2], dtype=float)
            if drag == (group, k) and self._preview is not None:
                position = self._preview
            colour = (
                SELECTED_COLOUR
                if chosen or drag == (group, k)
                else (SOURCE_COLOUR if group == "sources" else LISTENER_COLOUR)
            )
            p = self.to_screen(position)
            forward = np.asarray(item.orientation.forward()[:2], dtype=float)
            if np.linalg.norm(forward) > 0.2:
                forward = forward / np.linalg.norm(forward) * 18
                painter.setPen(QPen(QColor(colour), 2))
                painter.drawLine(p, QPointF(p.x() + forward[0], p.y() - forward[1]))
            radius = 7.0 if chosen or self._hover == (group, k) else 5.5
            painter.setPen(QPen(QColor(BACKGROUND[0]), 1.5))
            painter.setBrush(QColor(colour))
            painter.drawEllipse(p, radius, radius)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            text = item.name
            if drag == (group, k) and self._preview is not None:
                text += f"  ({position[0]:.2f}, {position[1]:.2f})"
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QPointF(p.x() + radius + 4, p.y() - radius), text)

    def _paint_drawing(self, painter: QPainter) -> None:
        points = [self.to_screen(c) for c in self.drawing]
        hover = None
        if self.cursor_position is not None:
            hover = self.to_screen(snap(self.cursor_position, self.snap_step))
        pen = QPen(QColor(SELECTED_COLOUR), 2)
        painter.setPen(pen)
        for a, b in zip(points, points[1:], strict=False):
            painter.drawLine(a, b)
        if points and hover is not None:
            painter.drawLine(points[-1], hover)
            length = float(
                np.linalg.norm(snap(self.cursor_position, self.snap_step) - self.drawing[-1])
            )
            self._label(painter, (points[-1] + hover) / 2, f"{length:.2f} m", "#ffffff")
            if len(points) >= 2:
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(hover, points[0])
        for k, p in enumerate(points):
            closing = k == 0 and len(points) >= 3
            size = 6 if closing else 3.5
            painter.setPen(QPen(QColor(BACKGROUND[0]), 1))
            painter.setBrush(QColor(SELECTED_COLOUR if closing else "#ffffff"))
            painter.drawRect(QRectF(p.x() - size, p.y() - size, 2 * size, 2 * size))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _label(self, painter: QPainter, at: QPointF, text: str, colour: str) -> None:
        metrics = painter.fontMetrics()
        width, height = metrics.horizontalAdvance(text) + 6, metrics.height() + 2
        box = QRectF(at.x() - width / 2, at.y() - height / 2, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_qcolor(BACKGROUND[0], 200))
        painter.drawRoundedRect(box, 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor(colour))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)


class PlanView(QWidget):
    """The plan with a bar above it: Draw room, the snapping grid, Fit, and what the
    mouse does here."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 0)
        self.draw = QPushButton("Draw room")
        self.draw.setCheckable(True)
        self.draw.setToolTip("Draw a new floor plan, corner by corner; it replaces the room")
        self.snap = QComboBox()
        for label, step in SNAP_STEPS:
            self.snap.addItem(label, step)
        self.snap.setCurrentIndex([s for _, s in SNAP_STEPS].index(DEFAULT_SNAP))
        self.snap.setToolTip("Corners and points snap to this grid; hold Shift to move freely")
        fit = QPushButton("Fit")
        fit.setToolTip("Show the whole room")
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.where = QLabel()
        self.where.setMinimumWidth(130)
        self.where.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bar.addWidget(self.draw)
        bar.addWidget(QLabel("Snap"))
        bar.addWidget(self.snap)
        bar.addWidget(fit)
        bar.addWidget(self.hint, 1)
        bar.addWidget(self.where)
        layout.addLayout(bar)
        self.canvas = PlanCanvas(self)
        layout.addWidget(self.canvas, 1)

        self.draw.toggled.connect(self._draw_toggled)
        self.snap.currentIndexChanged.connect(
            lambda *_: setattr(self.canvas, "snap_step", float(self.snap.currentData()))
        )
        fit.clicked.connect(self.canvas.fit)
        self.canvas.state_changed.connect(self._state)
        # The canvas's signals, for the window.
        self.picked = self.canvas.picked
        self.corners_changed = self.canvas.corners_changed
        self.moved = self.canvas.moved
        self.drawn = self.canvas.drawn
        self.place = self.canvas.place
        self.message = self.canvas.message
        self._state()

    def show_room(self, room: Room, selection=None) -> None:
        self.canvas.show_room(room, selection)
        self._state()

    def fit(self) -> None:
        self.canvas.fit()

    def start_drawing(self) -> None:
        self.canvas.start_drawing()

    def start_placing(self, kind: str | None) -> None:
        self.canvas.start_placing(kind)

    def cancel(self) -> bool:
        return self.canvas.cancel()

    def _draw_toggled(self, on: bool) -> None:
        if on and self.canvas.drawing is None:
            self.canvas.start_drawing()
        elif not on and self.canvas.drawing is not None:
            self.canvas.cancel()

    def _state(self) -> None:
        canvas = self.canvas
        mode = canvas.mode
        text = HINTS[mode]
        if mode == "placing":
            text = text.format(what="source" if canvas.placing == "sources" else "listener")
        self.hint.setText(text)
        self.draw.blockSignals(True)
        self.draw.setChecked(canvas.drawing is not None)
        self.draw.blockSignals(False)
        where = canvas.cursor_position
        self.where.setText("" if where is None else f"x {where[0]:.2f}, y {where[1]:.2f} m")


__all__ = [
    "DEFAULT_SNAP",
    "SNAP_STEPS",
    "PlanCanvas",
    "PlanView",
    "grid_step",
    "outline_segments",
    "snap",
    "wall_labels",
]
