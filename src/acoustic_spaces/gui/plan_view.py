"""The plan: the room seen from above, to draw floor plans, move their corners and move
sources and listeners by dragging; outdoors, to draw buildings, barriers, vegetation
and ground zones on the ground.

Floor plan and box rooms show their corners as handles. Dragging one moves it (a box
becomes a floor plan with the same walls), double-clicking a wall adds a corner there,
and the right-click menu removes one. *Draw room* starts a new floor plan: click its
corners in turn and close it on the first one, with a double-click or with Enter.
Rooms from model files show their outline, and a drawn room replaces the model.

Outdoors the plan shows the buildings' and barriers' outlines, the vegetation in dashed
green, the ground zones in their materials' colours, the weather zones dotted in sky
blue and the wind in sky blue arrows, the way it blows and as long as it is strong, with
each weather zone's wind and the scene's in the corner; the Geometry form's buttons
draw new ones: a building's, a belt of trees' or a zone's outline closes like a floor
plan, a barrier's or a hedge's path ends with a double-click or Enter, and a tree
stands where the next click lands.

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

from acoustic_spaces.geometry import UNIT_SCALE, plan_problem
from acoustic_spaces.gui.document import (
    LISTENER_HEIGHT,
    NEW_PART_SIZES,
    SOURCE_HEIGHT,
    follow_floor,
    plan_corners,
    surface_heights,
)
from acoustic_spaces.gui.movement import handles_of, moves_at, path_of, pose_at
from acoustic_spaces.gui.numbers import fmt, number, point_text
from acoustic_spaces.gui.tips import label_for, share_tips
from acoustic_spaces.gui.viewport import SELECTED_COLOUR, material_colour, surface_materials
from acoustic_spaces.i18n import N_, _
from acoustic_spaces.room import Room, outline_edges
from acoustic_spaces.viz import (
    BACKGROUND,
    EDGE_COLOUR,
    LISTENER_COLOUR,
    SOURCE_COLOUR,
    VEGETATION_COLOUR,
    WIND_COLOUR,
    WIND_KEY,
    WindMap,
    clear_lengths,
    common_radius,
    nearest_spacing,
)

#: Grid steps the corners and points snap to (label, metres; 0 is off).
SNAP_STEPS = (
    (N_("Off"), 0.0),
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
WEATHER_COLOUR = WIND_COLOUR
#: The arrows of the wind are no more than this many pixels apart.
WIND_PIXELS = 70
#: What a drawing is for: a room's floor plan, or outdoors a building's footprint, a
#: barrier's path, a belt of trees' footprint, a hedge's path, where a lone tree stands
#: or a ground zone's outline; those that close into a polygon, and those that are a
#: path from end to end.
PURPOSES = ("room", "building", "barrier", "belt", "hedge", "tree", "zone", "weather")
CLOSED = frozenset({"room", "building", "belt", "zone", "weather"})
PATHS = frozenset({"barrier", "hedge"})
GRID_COLOURS = ("#1d1e25", "#2b2d37", "#4a4d5a")  # minor, major, axes

#: Why a drawing cannot be finished yet, by purpose ({problem} for an outline's).
NOT_YET = {
    "room": N_("Not a room yet: {problem}"),
    "building": N_("Not a building yet: {problem}"),
    "belt": N_("Not a belt of trees yet: {problem}"),
    "zone": N_("Not a zone yet: {problem}"),
    "weather": N_("Not a weather zone yet: {problem}"),
    "barrier": N_("Not a barrier yet: it needs two points at least"),
    "hedge": N_("Not a hedge yet: it needs two points at least"),
}

HINTS = {
    "plan": N_(
        "Drag corners, sources and listeners; double-click a wall to add a corner; "
        "right-click for more. Shift: no snapping."
    ),
    "box": N_(
        "Drag a corner to reshape the box (it becomes a floor plan); drag sources and "
        "listeners. Shift: no snapping."
    ),
    "file": N_(
        "The model's outline from above: drag sources and listeners. Draw room "
        "replaces the model with a floor plan."
    ),
    "outdoor": N_(
        "The scene from above: drag sources and listeners. The Geometry form's buttons "
        "draw buildings, barriers, vegetation and zones. Shift: no snapping."
    ),
    "room": N_(
        "Click the corners of the new room; close it on the first corner, with a "
        "double-click or Enter. Backspace takes a corner back, Esc stops."
    ),
    "building": N_(
        "Click the corners of the new building's footprint; close it on the first "
        "corner, with a double-click or Enter. Backspace takes a corner back, Esc stops."
    ),
    "barrier": N_(
        "Click the points along the new barrier, end to end; finish with a double-click "
        "or Enter. Backspace takes a point back, Esc stops."
    ),
    "belt": N_(
        "Click the corners of the belt of trees; close it on the first corner, with a "
        "double-click or Enter. Backspace takes a corner back, Esc stops."
    ),
    "hedge": N_(
        "Click the points along the hedge, end to end; finish with a double-click or "
        "Enter. Backspace takes a point back, Esc stops."
    ),
    "tree": N_("Click where the tree stands; Esc cancels."),
    "zone": N_(
        "Click the corners of the ground zone; close it on the first corner, with a "
        "double-click or Enter. Backspace takes a corner back, Esc stops."
    ),
    "weather": N_(
        "Click the corners of the new weather zone; close it on the first corner, with a "
        "double-click or Enter. Backspace takes a corner back, Esc stops."
    ),
    "sources": N_("Click where the new source goes; Esc cancels."),
    "listeners": N_("Click where the new listener goes; Esc cancels."),
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


#: The model's edges seen from above, for the plan: :func:`acoustic_spaces.room.outline_edges`.
outline_segments = outline_edges


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
    #: A keyframe handle of a mover (its kind, index and the row, 0 for the mover
    #: itself) was dragged here (m).
    keyframe_moved = Signal(str, int, int, tuple)
    #: A new floor plan was drawn: its corners (m).
    drawn = Signal(list)
    #: An outdoor part was drawn: what for (one of :data:`PURPOSES` but "room") and its
    #: corners or points (m; one point for a tree).
    part_drawn = Signal(str, list)
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
        #: (kind, name) of the sources and listeners hidden in the views.
        self.hidden: set[tuple[str, str]] = frozenset()
        self.selection: tuple[str, object] = ("", None)
        #: Whether to draw the wind outdoors.
        self.show_wind = True
        self._wind: WindMap | None = None
        self.snap_step = DEFAULT_SNAP
        self.drawing: list[np.ndarray] | None = None
        self.purpose: str = "room"
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
        # ("corner", k) / (kind, index) / ("keyframe", row) / ("pan",)
        self._drag: tuple | None = None
        self._preview: np.ndarray | None = None
        self._hover: tuple | None = None
        #: The mover whose keyframe handles are drawn ((kind, index)), while the
        #: Movement panel is open.
        self.mover: tuple[str, int] | None = None
        #: The moment of the movement the movers are drawn at (s); None: at rest.
        self.time: float | None = None

    # -- contents -------------------------------------------------------------------

    def show_room(
        self,
        room: Room,
        selection: tuple[str, object] | None = None,
        hidden: set[tuple[str, str]] = frozenset(),
        wind: bool = True,
        mover: tuple[str, int] | None = None,
        time: float | None = None,
    ) -> None:
        """Draw ``room`` with ``selection`` highlighted, without the sources and
        listeners in ``hidden`` ((kind, name) pairs); outdoors, with the wind if
        ``wind``. With ``mover`` (a source or listener as (kind, index)), its keyframes
        get handles to drag; with ``time``, every mover is drawn where it is then."""
        new_geometry = room.mesh is not self._mesh
        self.room = room
        self.selection = selection or ("", None)
        self.hidden = hidden
        self.show_wind = wind
        self.mover = mover
        self.time = time
        self._wind = WindMap.of(room)
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
        """What the plan is doing: "drawing", "placing", or the kind of scene ("plan",
        "box", "file" or "outdoor")."""
        if self.drawing is not None:
            return "drawing"
        if self.placing is not None:
            return "placing"
        if self.room is None:
            return "file"
        geometry = self.room.scene.geometry
        if geometry.ground is not None:
            return "outdoor"
        return "plan" if geometry.floor_plan is not None else "box" if geometry.box else "file"

    def fit(self) -> None:
        """Show the whole room and every source and listener, with the places their walks
        go through; resizing the view keeps them in it until the view is zoomed or
        panned."""
        self._user_view = False
        if self.room is None:
            return
        points = [self.room.mesh.vertices[:, :2]]
        for kind in ("sources", "listeners"):
            points += [handles_of(item)[:, :2] for item in getattr(self.room.scene, kind)]
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
        """(kind, index, item) of every source and listener shown, each where it is at
        :attr:`time`."""
        if self.room is None:
            return []
        return [
            (kind, k, pose_at(item, self.time))
            for kind in ("sources", "listeners")
            for k, item in enumerate(getattr(self.room.scene, kind))
            if (kind, item.name) not in self.hidden
        ]

    def set_time(self, time: float | None) -> None:
        """Draw every mover where it is at ``time`` (s) of the movement; None: at rest."""
        self.time = time
        self.update()

    def _handles(self) -> np.ndarray:
        """Where the keyframes of :attr:`mover` are, (K, 3); none without a mover shown."""
        if self.room is None or self.mover is None:
            return np.zeros((0, 3))
        kind, index = self.mover
        items = getattr(self.room.scene, kind)
        if not 0 <= index < len(items) or (kind, items[index].name) in self.hidden:
            return np.zeros((0, 3))
        return handles_of(items[index])

    def hit(self, position: QPointF) -> tuple | None:
        """What is at ``position`` (widget pixels): ("sources" or "listeners", index),
        ("keyframe", row) of the mover's handles, ("corner", k), ("wall", k, t along it),
        or None."""
        best, nearest = None, HANDLE_PIXELS
        for row, handle in enumerate(self._handles()):
            d = self._pixels(handle, position)
            if d <= nearest:
                best, nearest = ("keyframe", row), d
        if best is not None:
            return best
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

    def start_drawing(self, purpose: str = "room") -> None:
        """Start drawing a new floor plan, corner by corner, or outdoors one of the
        other :data:`PURPOSES`."""
        if purpose not in PURPOSES:
            raise ValueError(f"nothing to draw for {purpose!r}")
        self.placing = None
        self.drawing = []
        self.purpose = purpose
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.state_changed.emit()
        self.update()

    def cancel(self) -> bool:
        """Stop drawing, placing or dragging; True if there was something to stop."""
        busy = self.drawing is not None or self.placing is not None or self._drag is not None
        self.drawing = self.placing = None
        self.purpose = "room"
        self._drag = self._preview = self._press = None
        self.unsetCursor()
        self.state_changed.emit()
        self.update()
        return busy

    def add_drawn_corner(self, point) -> None:
        """Add a corner to the outline being drawn (a click on the first corner closes
        it), a point to the path, or the tree's place, which finishes it."""
        if self.drawing is None:
            return
        point = np.asarray(point, dtype=float)
        closing = self.purpose in CLOSED and len(self.drawing) >= 3
        if closing and self._pixels(self.drawing[0], self.to_screen(point)) <= HANDLE_PIXELS:
            self.finish_drawing()
            return
        if self.drawing and np.allclose(point, self.drawing[-1]):
            return
        self.drawing.append(point)
        if self.purpose == "tree":
            self.finish_drawing()
            return
        self.update()

    def undo_drawn_corner(self) -> None:
        if self.drawing:
            self.drawing.pop()
            self.update()

    def finish_drawing(self) -> bool:
        """Close the outline being drawn, or end the path, and emit it; False (and a
        message) if it cannot be one yet, and drawing goes on."""
        if self.drawing is None:
            return False
        purpose = self.purpose
        corners = np.array(self.drawing, dtype=float).reshape(-1, 2)
        if purpose in CLOSED:
            problem = plan_problem(corners)
            if problem is not None:
                self.message.emit(_(NOT_YET[purpose]).format(problem=problem))
                return False
        elif purpose in PATHS and len(corners) < 2:
            self.message.emit(_(NOT_YET[purpose]))
            return False
        elif purpose == "tree" and len(corners) < 1:
            return False
        self.drawing = None
        self.purpose = "room"
        self.unsetCursor()
        self.state_changed.emit()
        points = [[float(x), float(y)] for x, y in corners]
        if purpose == "room":
            self.drawn.emit(points)
        else:
            self.part_drawn.emit(purpose, points)
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
        self.corners_changed.emit(corners, _("add corner"))

    def remove_corner(self, k: int) -> None:
        if self.corners is None:
            return
        if len(self.corners) <= 3:
            self.message.emit(_("A floor plan needs at least three corners."))
            return
        corners = [list(map(float, c)) for i, c in enumerate(self.corners) if i != k]
        self.corners_changed.emit(corners, _("remove corner"))

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
        _start, target = self._press
        drag, preview = self._drag, self._preview
        self._press = self._drag = self._preview = None
        if self._hover is None:
            self.unsetCursor()
        if drag is None:
            self._click(event.position(), target)
        elif drag[0] == "corner" and preview is not None:
            problem = plan_problem(preview)
            if problem is not None:
                self.message.emit(_("Not moved: {problem}").format(problem=problem))
            else:
                self.corners_changed.emit(preview.tolist(), _("move corner"))
        elif drag[0] in ("sources", "listeners") and preview is not None:
            kind, index = drag
            item = getattr(self.room.scene, kind)[index]
            if moves_at(item, self.time):
                self.message.emit(
                    fmt(
                        _(
                            "Not moved: {name} is drawn where it is at {time:.2f} s; drag its "
                            "keyframes, or its start at 0 s"
                        ),
                        name=item.name,
                        time=self.time,
                    )
                )
            else:
                self.moved.emit(kind, index, follow_floor(self.room.mesh, item.position, preview))
        elif drag[0] == "keyframe" and preview is not None:
            handles = self._handles()
            if drag[1] < len(handles):
                kind, index = self.mover
                self.keyframe_moved.emit(
                    kind, index, drag[1], follow_floor(self.room.mesh, handles[drag[1]], preview)
                )
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
        elif target[0] == "keyframe" and self.mover is not None:
            self.picked.emit(*self.mover)

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
            remove = menu.addAction(_("Remove corner"))
            remove.setEnabled(len(self.corners) > 3)
            remove.triggered.connect(lambda *_: self.remove_corner(target[1]))
        elif target is not None and target[0] == "wall":
            menu.addAction(_("Add corner here")).triggered.connect(
                lambda *_: self.add_corner(target[1], point)
            )
        if not menu.isEmpty():
            menu.addSeparator()
        for kind, text in (
            ("sources", _("Place source here")),
            ("listeners", _("Place listener here")),
        ):
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
            if self.room.outdoor is not None:
                self._paint_outdoors(painter)
            self._paint_walks(painter)
            self._paint_points(painter)
            self._paint_wind_key(painter)
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
                painter.drawText(QPointF(sx + 3, self.height() - 4), number(round(x, 9) + 0.0))
        for y in np.arange(math.ceil(lo[1] / major) * major, hi[1], major):
            sy = self.to_screen((0, y)).y()
            if sy > 16:
                painter.drawText(QPointF(3, sy - 3), number(round(y, 9) + 0.0))
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
                self._label(
                    painter, (a + b) / 2, fmt(_("{length:.2f} m"), length=length), "#ffffff"
                )
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

    def _paint_outdoors(self, painter: QPainter) -> None:
        """The ground zones in their materials' colours, and the vegetation in dashed
        green: a belt's or a hedge's outline, a tree's crown."""
        outdoor = self.room.outdoor
        labels = {label[0]: k for k, label in enumerate(self.room.mesh.labels)}
        font = painter.font()
        painter.setFont(font)
        for name, polygon in outdoor.zones:
            if len(polygon) < 3:
                continue
            colour = self._material_of(labels.get(name, -1))
            outline = QPolygonF([self.to_screen(c) for c in polygon])
            painter.setPen(QPen(_qcolor(colour, 160), 1.5))
            painter.setBrush(_qcolor(colour, 40))
            painter.drawPolygon(outline)
            self._label(painter, outline.boundingRect().center(), name, colour)
        atmosphere = self.room.scene.atmosphere
        if atmosphere is not None and atmosphere.zones:
            # The weather zones, dotted in sky blue.
            scale = UNIT_SCALE[self.room.scene.geometry.units or "m"]
            pen = QPen(_qcolor(WEATHER_COLOUR, 220), 1.5)
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.setBrush(_qcolor(WEATHER_COLOUR, 20))
            names = atmosphere.zone_names()
            outlines = [
                QPolygonF([self.to_screen(scale * np.asarray(c)) for c in zone.polygon])
                for zone in atmosphere.zones
            ]
            for outline in outlines:
                painter.drawPolygon(outline)
            self._paint_wind(painter)
            wind = self._wind if self.show_wind else None
            for k, (name, outline) in enumerate(zip(names, outlines, strict=True), start=1):
                # Inside its top edge, clear of the arrow in its middle.
                box = outline.boundingRect()
                at = QPointF(
                    box.center().x(),
                    min(box.top() + painter.fontMetrics().height(), box.center().y()),
                )
                text = name if wind is None else wind.label(k)
                self._label(painter, at, text, WEATHER_COLOUR)
        else:
            self._paint_wind(painter)
        pen = QPen(_qcolor(VEGETATION_COLOUR, 220), 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(_qcolor(VEGETATION_COLOUR, 30))
        for shape in outdoor.vegetation:
            if shape.outline is not None:
                outline = QPolygonF([self.to_screen(c) for c in shape.outline])
                painter.drawPolygon(outline)
                at = outline.boundingRect().center()
            else:
                at = self.to_screen(shape.centre[:2])
                radius = shape.radius * self._scale
                painter.drawEllipse(at, radius, radius)
            self._label(painter, at, shape.name, VEGETATION_COLOUR)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_wind(self, painter: QPainter) -> None:
        """The wind over the ground: arrows the way it blows, as long as it is strong, on
        a grid at most :data:`WIND_PIXELS` apart, clear of the sources and listeners (see
        :meth:`~acoustic_spaces.viz.WindMap.arrows`)."""
        wind = self._wind
        if wind is None or not self.show_wind or not wind.windy:
            return
        markers = [item.position[:2] for _kind, _k, item in self._points()]
        lo = self.to_world(QPointF(0, self.height()))
        hi = self.to_world(QPointF(self.width(), 0))
        arrows = wind.arrows(WIND_PIXELS / self._scale, markers, (*lo, *hi))
        painter.setPen(QPen(_qcolor(WIND_COLOUR, 200), 1.6))
        for middle, vector in zip(arrows.middles, arrows.vectors, strict=True):
            length = float(np.linalg.norm(vector)) * self._scale
            self._arrow(painter, self.to_screen(middle), vector / np.linalg.norm(vector), length)

    def _arrow(self, painter: QPainter, centre: QPointF, direction, length: float) -> None:
        """An arrow ``length`` pixels long through ``centre`` along ``direction`` (x, y,
        in the world), with the pen set."""
        if length <= 0.0:
            return
        d = QPointF(float(direction[0]), -float(direction[1])) * (0.5 * length)
        tail, tip = centre - d, centre + d
        painter.drawLine(tail, tip)
        head = min(0.35 * length, 9.0)
        angle = math.atan2(-d.y(), -d.x())  # back along the shaft, as on the screen
        for turn in (-0.5, 0.5):
            painter.drawLine(
                tip,
                tip + QPointF(math.cos(angle + turn), math.sin(angle + turn)) * head,
            )

    def _paint_wind_key(self, painter: QPainter) -> None:
        """In the top right corner, the scene's own wind: which way, how strong and at
        what height, which the arrows' lengths are for."""
        wind = self._wind
        if wind is None or not self.show_wind or not wind.windy:
            return
        text = fmt(_(WIND_KEY), speed=wind.speeds[0], height=wind.height)
        metrics = painter.fontMetrics()
        icon = metrics.height() + 6
        width, height = metrics.horizontalAdvance(text) + icon + 12, metrics.height() + 8
        box = QRectF(self.width() - width - 8, 8, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_qcolor(BACKGROUND[0], 220))
        painter.drawRoundedRect(box, 3, 3)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(WIND_COLOUR), 1.6))
        if wind.speeds[0] > 0.0:
            centre = QPointF(box.left() + 4 + icon / 2, box.center().y())
            self._arrow(painter, centre, wind.directions[0], icon - 4)
        painter.setPen(QColor(WIND_COLOUR))
        painter.drawText(
            box.adjusted(icon + 6, 0, -4, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )

    def _paint_walks(self, painter: QPainter) -> None:
        """Every walk as a dashed line through its knots, faded when it is left standing,
        and the handles of the mover's keyframes, numbered as the table's rows."""
        kind, key = self.selection
        for group in ("sources", "listeners"):
            for k, item in enumerate(getattr(self.room.scene, group)):
                path = path_of(item)
                if path is None or (group, item.name) in self.hidden:
                    continue
                chosen = (kind, key) == (group, k)
                colour = (
                    SELECTED_COLOUR
                    if chosen
                    else (SOURCE_COLOUR if group == "sources" else LISTENER_COLOUR)
                )
                alpha = 200 if item.in_use and item.walk.enabled else 70
                painter.setPen(QPen(_qcolor(colour, alpha), 1.5, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolyline(QPolygonF([self.to_screen(p) for p in path.positions]))
        handles = self._handles()
        if len(handles) == 0:
            return
        drag = self._drag if self._drag is not None and self._drag[0] == "keyframe" else None
        for row, handle in enumerate(handles):
            at = handle[:2]
            if drag is not None and drag[1] == row and self._preview is not None:
                at = self._preview
            p = self.to_screen(at)
            size = 5.0 if self._hover == ("keyframe", row) or drag is not None else 4.0
            painter.setPen(QPen(QColor(BACKGROUND[0]), 1))
            painter.setBrush(QColor(SELECTED_COLOUR if row > 0 else "#ffffff"))
            painter.drawRect(QRectF(p.x() - size, p.y() - size, 2 * size, 2 * size))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QPointF(p.x() + size + 3, p.y() + size + 4), str(row))

    def _paint_points(self, painter: QPainter) -> None:
        kind, key = self.selection
        drag = (
            self._drag
            if self._drag is not None and self._drag[0] in ("sources", "listeners")
            else None
        )
        shown = []
        for group, k, item in self._points():
            position = np.asarray(item.position[:2], dtype=float)
            if drag == (group, k) and self._preview is not None:
                position = self._preview
            shown.append((group, k, item, position, self.to_screen(position)))
        # Markers all alike, smaller where two would overlap on the screen, and arrows
        # that stop short of a marker ahead.
        screen = np.array([(p.x(), p.y()) for *_, p in shown]).reshape(-1, 2)
        radii = np.full(len(screen), common_radius(nearest_spacing(screen), 5.5, 3.5 / 5.5))
        aims = np.zeros_like(screen)
        for row, entry in enumerate(shown):
            forward = np.asarray(entry[2].orientation.forward()[:2], dtype=float) * [1.0, -1.0]
            if np.linalg.norm(forward) > 0.2:  # drawn: not pointing up or down
                aims[row] = forward / np.linalg.norm(forward)
        arrows = clear_lengths(screen, aims, radii, 18.0)
        for (group, k, item, position, p), size, arrow in zip(shown, radii, arrows, strict=True):
            chosen = kind == group and key == k
            colour = (
                SELECTED_COLOUR
                if chosen or drag == (group, k)
                else (SOURCE_COLOUR if group == "sources" else LISTENER_COLOUR)
            )
            # Left out of simulations: faded.
            alpha = 255 if item.in_use or chosen else 90
            if group == "sources" and item.extended:  # a line or surface, seen from above
                corners = item.outline()[:, :2] + (position - np.asarray(item.position[:2]))
                outline = QPolygonF([self.to_screen(c) for c in corners])
                painter.setPen(QPen(_qcolor(colour, min(alpha, 200)), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                if len(corners) > 2:
                    painter.drawPolygon(outline)
                else:
                    painter.drawPolyline(outline)
            forward = np.asarray(item.orientation.forward()[:2], dtype=float)
            if np.linalg.norm(forward) > 0.2:
                forward = forward / np.linalg.norm(forward) * arrow
                painter.setPen(QPen(_qcolor(colour, alpha), 2))
                painter.drawLine(p, QPointF(p.x() + forward[0], p.y() - forward[1]))
            radius = 7.0 if chosen or self._hover == (group, k) else size
            painter.setPen(QPen(_qcolor(BACKGROUND[0], alpha), 1.5))
            painter.setBrush(_qcolor(colour, alpha))
            painter.drawEllipse(p, radius, radius)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            text = item.name
            if drag == (group, k) and self._preview is not None:
                text += f"  ({point_text(position[:2])})"
            painter.setPen(_qcolor("#ffffff", alpha))
            painter.drawText(QPointF(p.x() + radius + 4, p.y() - radius), text)

    def _paint_drawing(self, painter: QPainter) -> None:
        points = [self.to_screen(c) for c in self.drawing]
        hover = None
        if self.cursor_position is not None:
            hover = self.to_screen(snap(self.cursor_position, self.snap_step))
        pen = QPen(QColor(SELECTED_COLOUR), 2)
        painter.setPen(pen)
        if self.purpose == "tree":
            if hover is not None:  # the crown the tree would have there
                radius = 0.5 * NEW_PART_SIZES["tree"]["crown_diameter"] * self._scale
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(hover, radius, radius)
            return
        closed = self.purpose in CLOSED
        for a, b in zip(points, points[1:], strict=False):
            painter.drawLine(a, b)
        if points and hover is not None:
            painter.drawLine(points[-1], hover)
            length = float(
                np.linalg.norm(snap(self.cursor_position, self.snap_step) - self.drawing[-1])
            )
            self._label(
                painter,
                (points[-1] + hover) / 2,
                fmt(_("{length:.2f} m"), length=length),
                "#ffffff",
            )
            if closed and len(points) >= 2:
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(hover, points[0])
        for k, p in enumerate(points):
            closing = closed and k == 0 and len(points) >= 3
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
        self.draw = QPushButton(_("Draw room"))
        self.draw.setCheckable(True)
        self.draw.setToolTip(
            _(
                "Draw a new floor plan, corner by corner; it replaces the room. Buildings, "
                "barriers, vegetation and ground zones are drawn from the Geometry form"
            )
        )
        self.snap = QComboBox()
        for label, step in SNAP_STEPS:
            self.snap.addItem(_(label) if step == 0 else label, step)
        self.snap.setCurrentIndex([step for _label, step in SNAP_STEPS].index(DEFAULT_SNAP))
        self.snap.setToolTip(
            _("Corners, sources and listeners snap to this grid; hold Shift to place them freely")
        )
        fit = QPushButton(_("Fit"))
        fit.setToolTip(_("Zoom to show the whole room"))
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.where = QLabel()
        self.where.setToolTip(_("Where the mouse is in the plan, in metres"))
        self.where.setMinimumWidth(130)
        self.where.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bar.addWidget(self.draw)
        bar.addWidget(label_for(_("Snap"), self.snap))
        bar.addWidget(self.snap)
        bar.addWidget(fit)
        bar.addWidget(self.hint, 1)
        bar.addWidget(self.where)
        layout.addLayout(bar)
        self.canvas = PlanCanvas(self)
        layout.addWidget(self.canvas, 1)
        share_tips(self)

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
        self.keyframe_moved = self.canvas.keyframe_moved
        self.drawn = self.canvas.drawn
        self.part_drawn = self.canvas.part_drawn
        self.place = self.canvas.place
        self.message = self.canvas.message
        self._state()

    def show_room(
        self, room: Room, selection=None, hidden=frozenset(), wind=True, mover=None, time=None
    ) -> None:
        self.canvas.show_room(room, selection, hidden, wind, mover, time)
        self._state()

    def set_time(self, time: float | None) -> None:
        self.canvas.set_time(time)

    def fit(self) -> None:
        self.canvas.fit()

    def start_drawing(self, purpose: str = "room") -> None:
        self.canvas.start_drawing(purpose)

    def start_placing(self, kind: str | None) -> None:
        self.canvas.start_placing(kind)

    def cancel(self) -> bool:
        return self.canvas.cancel()

    def _draw_toggled(self, on: bool) -> None:
        if on and (self.canvas.drawing is None or self.canvas.purpose != "room"):
            self.canvas.start_drawing()
        elif not on and self.canvas.drawing is not None:
            self.canvas.cancel()

    def _state(self) -> None:
        canvas = self.canvas
        mode = canvas.mode
        if mode == "drawing":
            key = canvas.purpose
        else:
            key = canvas.placing if mode == "placing" else mode
        text = _(HINTS[key])
        self.hint.setText(text)
        self.hint.setToolTip(text)  # in full, should the bar cut it short
        self.draw.blockSignals(True)
        self.draw.setChecked(canvas.drawing is not None and canvas.purpose == "room")
        self.draw.blockSignals(False)
        where = canvas.cursor_position
        self.where.setText(
            "" if where is None else fmt(_("x {x:.2f}, y {y:.2f} m"), x=where[0], y=where[1])
        )


__all__ = [
    "CLOSED",
    "DEFAULT_SNAP",
    "PATHS",
    "PURPOSES",
    "SNAP_STEPS",
    "PlanCanvas",
    "PlanView",
    "grid_step",
    "outline_segments",
    "snap",
    "wall_labels",
]
