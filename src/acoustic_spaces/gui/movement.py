"""The Movement panel: the walk of one source or listener as a table of keyframes, where
it is and how it faces at each time, edited in place, and a timeline every mover is
previewed along in the 3D view and the plan.

A keyframe needs no speed: it says where the mover is at a time, and the speed to the
next one follows, the distance over the time between them, shown beside it. The
timeline runs to the longest walk, or as far as **End** says, so that a keyframe can go
past the end: **Add keyframe** puts one at the playhead, where the mover would be then,
to drag into place in the plan or the 3D view. Every change is one step of the
document's history, as the forms' are, and shows at once. **Render movement…** plays
what the sources play through the movement
(:func:`~acoustic_spaces.walkthrough.simulate_movement`) in the background.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QTimer, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gui.document import DECIMALS, Document, EditError
from acoustic_spaces.gui.numbers import fmt, number, parse_number
from acoustic_spaces.gui.tips import label_for, share_tips
from acoustic_spaces.i18n import N_, _
from acoustic_spaces.scene.model import Orientation
from acoustic_spaces.walkthrough import Knots, knots

#: What can move: point sources and listeners.
KINDS = ("sources", "listeners")
ANGLES = ("yaw", "pitch", "roll")
#: The table's columns.
TIME, X, Y, Z, YAW, PITCH, ROLL, SPEED = range(8)
HEADERS = (N_("Time"), "x", "y", "z", N_("Yaw"), N_("Pitch"), N_("Roll"), N_("Speed"))
#: The playhead's steps along the slider.
SLIDER_STEPS = 1000
#: How often the playhead moves while playing (ms).
TICK_MS = 30
#: Past the longest walk, where a keyframe goes when the playhead is on one already (s).
STEP_S = 1.0


def _place(position) -> list[float]:
    return [round(float(v), DECIMALS) + 0.0 for v in position]


def _angles(orientation: Orientation) -> list[float]:
    return [float(orientation.yaw), float(orientation.pitch), float(orientation.roll)]


def path_of(item) -> Knots | None:
    """The knots of ``item``'s walk, None when it has no walk or its walk goes nowhere
    (which the scene's checks report)."""
    if item.walk is None:
        return None
    try:
        return knots(item)
    except ValueError:
        return None


def handles_of(item) -> np.ndarray:
    """Where ``item``'s keyframes can be dragged: its own position (row 0 of the table)
    and each keyframe's, (K, 3); only its position for a walk along ``to`` or none."""
    walk = item.walk
    frames = walk.keyframes if walk is not None and walk.keyframes is not None else []
    return np.array([item.position, *(frame.position for frame in frames)], dtype=float)


def moves_at(item, time_s: float | None) -> bool:
    """Whether the views show ``item`` away from its position at ``time_s``: it moves (in
    use, its walk enabled) and the time is past the start."""
    return (
        time_s is not None
        and time_s > 0.0
        and item.in_use
        and item.walk is not None
        and item.walk.enabled
        and path_of(item) is not None
    )


def pose_at(item, time_s: float | None):
    """``item`` where it is and facing as it does at ``time_s`` of the movement: itself
    unless it moves then (:func:`moves_at`)."""
    if not moves_at(item, time_s):
        return item
    position, angles = path_of(item).at(time_s)
    orientation = Orientation(yaw=angles[0], pitch=angles[1], roll=angles[2])
    return item.model_copy(update={"position": tuple(position), "orientation": orientation})


def movement_duration(scene) -> float:
    """How long the longest walk of ``scene`` lasts (s), enabled or not; 0 with none."""
    longest = 0.0
    for kind in KINDS:
        for item in getattr(scene, kind):
            path = path_of(item)
            if path is not None:
                longest = max(longest, path.duration_s)
    return longest


class MovementPanel(QWidget):
    """The keyframes of one mover and the timeline of them all (see the module)."""

    #: The playhead moved: the time (s) the views show every mover at.
    preview_changed = Signal(float)
    #: Render movement… was clicked.
    render_requested = Signal()
    #: Another mover was picked here: ("sources" or "listeners", index).
    selected = Signal(str, object)
    #: An edit was refused; the message says why.
    failed = Signal(str)
    #: Something to tell the user.
    message = Signal(str)

    def __init__(self, document: Callable[[], Document]) -> None:
        super().__init__()
        self._document = document
        self._loading = False
        #: The mover shown: ("sources" or "listeners", index), None with none to show.
        self.key: tuple[str, int] | None = None
        self._movers: list[tuple[str, int]] = []
        self._times = np.zeros(1)
        self._angles = np.zeros((1, 3))
        self._end = 0.0
        self._time = 0.0
        self._origin = 0.0
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        head = QHBoxLayout()
        self.mover = QComboBox()
        self.mover.setToolTip(
            _(
                "The source or listener whose keyframes the table shows (a line or a surface "
                "does not move); picking one here selects it in the scene"
            )
        )
        self.mover.currentIndexChanged.connect(self._mover_picked)
        head.addWidget(label_for(_("Mover"), self.mover))
        head.addWidget(self.mover, 1)
        self.moves = QCheckBox(_("Moves"))
        self.moves.setToolTip(
            _(
                "Tick to have it go along its keyframes in a movement; untick to keep them "
                "but leave it standing where it starts (enabled in the scene file)"
            )
        )
        self.moves.toggled.connect(self._moves)
        head.addWidget(self.moves)
        head.addStretch(1)
        self.render = QPushButton(_("Render movement…"))
        self.render.setToolTip(
            _(
                "Simulate every mover along its keyframes, on one clock, and write what each "
                "listener in use hears of what the sources play (their Plays files) as a WAV "
                "file, in the background (walkthrough --all)"
            )
        )
        self.render.clicked.connect(self.render_requested)
        head.addWidget(self.render)
        layout.addLayout(head)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels([_(text) for text in HEADERS])
        tips = (
            _(
                "When it is at this keyframe, s from the start of the movement; type another "
                "time to move the keyframe along the timeline. The first row is where it "
                "starts, at 0 s"
            ),
            _(
                "Where it is at this keyframe along x (m); dragging the keyframe's handle in "
                "the plan or the 3D view moves it too"
            ),
            _(
                "Where it is at this keyframe along y (m); dragging the keyframe's handle in "
                "the plan or the 3D view moves it too"
            ),
            _(
                "Where it is at this keyframe along z (m); z is up. In the 3D view the handle "
                "moves up and down with Shift"
            ),
            _(
                "Which way it faces at this keyframe: turned left from +x towards +y "
                "(degrees). Dimmed where it keeps facing as at the keyframe before; type an "
                "angle to turn it here, evenly from the keyframe before"
            ),
            _(
                "How far it looks up at this keyframe (degrees). Dimmed where it keeps facing "
                "as at the keyframe before; type an angle to turn it here"
            ),
            _(
                "How far it tilts to the right at this keyframe (degrees). Dimmed where it "
                "keeps facing as at the keyframe before; type an angle to turn it here"
            ),
            _(
                "How fast it goes from this keyframe to the next: the distance between them "
                "over the time between them (m/s). Nothing to set: change a time or a "
                "position and the speed follows"
            ),
        )
        for col, tip in enumerate(tips):
            self.table.horizontalHeaderItem(col).setToolTip(tip)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemChanged.connect(self._cell_changed)
        self.table.currentCellChanged.connect(lambda *_: self._row_picked())
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.add = QPushButton(_("Add keyframe"))
        self.add.setToolTip(
            _(
                "Add a keyframe at the playhead, where the mover is then (on a keyframe "
                "already, or at 0 s: a second past the end of its walk, where it ends), then "
                "drag it into place or type where it is"
            )
        )
        self.add.clicked.connect(self.add_keyframe)
        self.remove = QPushButton(_("Remove keyframe"))
        self.remove.setToolTip(
            _(
                "Remove the selected keyframe; the mover goes straight from the one before to "
                "the one after. With the last one, the walk goes too"
            )
        )
        self.remove.clicked.connect(self.remove_keyframe)
        self.convert = QPushButton(_("Convert to keyframes"))
        self.convert.setToolTip(
            _(
                "Turn a walk along points at a speed (to, speed_m_s) into keyframes that move "
                "the same way, timed, so that they can be edited here"
            )
        )
        self.convert.clicked.connect(self.convert_walk)
        for button in (self.add, self.remove, self.convert):
            buttons.addWidget(button)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        buttons.addWidget(self.hint, 1)
        layout.addLayout(buttons)

        timeline = QHBoxLayout()
        self.play = QPushButton(_("Play"))
        self.play.setToolTip(
            _(
                "Move every mover along its keyframes in the 3D view and the plan, from the "
                "playhead to the end; nothing is simulated or changed"
            )
        )
        self.play.clicked.connect(self.toggle_play)
        timeline.addWidget(self.play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setToolTip(
            _(
                "The playhead: the 3D view and the plan show every mover where it is at this "
                "time, and Add keyframe adds one here"
            )
        )
        self.slider.valueChanged.connect(self._slid)
        timeline.addWidget(self.slider, 1)
        self.time_box = QDoubleSpinBox()
        self.time_box.setDecimals(2)
        self.time_box.setSingleStep(0.1)
        self.time_box.setSuffix(" s")
        self.time_box.setKeyboardTracking(False)
        self.time_box.setToolTip(self.slider.toolTip())
        self.time_box.valueChanged.connect(self.set_time)
        timeline.addWidget(self.time_box)
        self.end_box = QDoubleSpinBox()
        self.end_box.setDecimals(1)
        self.end_box.setSingleStep(1.0)
        self.end_box.setRange(0.0, 36000.0)
        self.end_box.setSuffix(" s")
        self.end_box.setKeyboardTracking(False)
        self.end_box.setToolTip(
            _(
                "Where the timeline ends: at least the longest walk; make it longer to add "
                "keyframes past the end, and the movers that have ended stand where their "
                "walks end"
            )
        )
        self.end_box.valueChanged.connect(self._end_changed)
        timeline.addWidget(label_for(_("End"), self.end_box))
        timeline.addWidget(self.end_box)
        layout.addLayout(timeline)
        share_tips(self)
        self.reload()

    # -- what is shown ------------------------------------------------------------------

    @property
    def document(self) -> Document:
        return self._document()

    @property
    def item(self):
        """The mover shown, a scene source or listener, or None."""
        if self.key is None:
            return None
        kind, index = self.key
        items = getattr(self.document.scene, kind)
        return items[index] if 0 <= index < len(items) else None

    @property
    def time(self) -> float:
        """Where the playhead is (s)."""
        return self._time

    @property
    def end(self) -> float:
        """Where the timeline ends (s)."""
        return self._end

    def reset(self) -> None:
        """Another scene was opened: the playhead at 0, the timeline as long as its walks."""
        self.pause()
        self.key = None
        self._end = self._time = 0.0
        self.reload()

    def reload(self) -> None:
        """Follow the document: the movers it has, the keyframes of the one shown, the
        timeline's length."""
        scene = self.document.scene
        shown = None
        if self.item is not None:
            shown = (self.key[0], self.item.name)
        self._movers = [
            (kind, index)
            for kind in KINDS
            for index, item in enumerate(getattr(scene, kind))
            if kind != "sources" or not item.extended
        ]
        self._loading = True
        try:
            self.mover.clear()
            for kind, index in self._movers:
                item = getattr(scene, kind)[index]
                text = _("Source: {name}") if kind == "sources" else _("Listener: {name}")
                self.mover.addItem(text.format(name=item.name))
            self.key = next(
                (
                    (kind, index)
                    for kind, index in self._movers
                    if (kind, getattr(scene, kind)[index].name) == shown
                ),
                self._movers[0] if self._movers else None,
            )
            if self.key is not None:
                self.mover.setCurrentIndex(self._movers.index(self.key))
        finally:
            self._loading = False
        self._fill()
        self._set_range()

    def show_item(self, kind: str, key) -> None:
        """Show the keyframes of the selected source or listener; anything else keeps
        the mover shown."""
        if (kind, key) in self._movers and (kind, key) != self.key:
            self.key = (kind, key)
            self._loading = True
            try:
                self.mover.setCurrentIndex(self._movers.index(self.key))
            finally:
                self._loading = False
            self._fill()

    def _mover_picked(self, index: int) -> None:
        if self._loading or not 0 <= index < len(self._movers):
            return
        self.key = self._movers[index]
        self._fill()
        self.selected.emit(*self.key)

    def _fill(self) -> None:
        """The table from the mover's walk: row 0 where it starts, a row per keyframe."""
        item = self.item
        self._loading = True
        try:
            self.table.setRowCount(0)
            walk = item.walk if item is not None else None
            path = path_of(item) if item is not None else None
            timed = walk is not None and walk.keyframes is not None
            along = walk is not None and not timed
            self.moves.setEnabled(walk is not None)
            self.moves.setChecked(walk is not None and walk.enabled)
            self.add.setEnabled(item is not None and not along)
            self.remove.setEnabled(timed)
            self.convert.setVisible(along)
            self.convert.setEnabled(path is not None)
            if item is None:
                self.hint.setText(_("Place a source or a listener to move it."))
                self._times, self._angles = np.zeros(1), np.zeros((1, 3))
                return
            if path is None:
                times = np.zeros(1)
                positions = np.array([item.position], dtype=float)
                angles = np.array([_angles(item.orientation)])
                given = [True]
            else:
                times, positions = path.times, path.positions
                # The angles as given where they are, else as it faces then, wrapped.
                angles = (path.angles + 180.0) % 360.0 - 180.0
                angles[0] = _angles(item.orientation)
                given = [True] * len(times)
                if timed:
                    for row, frame in enumerate(walk.keyframes, start=1):
                        if frame.orientation is None:
                            given[row] = False
                        else:
                            angles[row] = _angles(frame.orientation)
            self._times, self._angles = times, angles
            legs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            taken = np.diff(times)
            dim = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            self.table.setRowCount(len(times))
            for row in range(len(times)):
                cells = [
                    number(times[row], ".2f"),
                    *(number(v, ".2f") for v in positions[row]),
                    *(number(a, ".1f") for a in angles[row]),
                    number(legs[row] / taken[row], ".2f")
                    if row < len(legs) and taken[row] > 0
                    else "",
                ]
                for col, text in enumerate(cells):
                    cell = QTableWidgetItem(text)
                    editable = not along and col != SPEED and (row > 0 or col != TIME)
                    if not editable:
                        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if YAW <= col <= ROLL and not given[row]:
                        cell.setForeground(dim)
                        cell.setToolTip(
                            _("Keeps facing as at the keyframe before; type an angle to turn here")
                        )
                    self.table.setItem(row, col, cell)
                header = QTableWidgetItem(str(row))
                header.setToolTip(
                    _("Where it starts, at 0 s: its own position and orientation")
                    if row == 0
                    else _("Keyframe {n}").format(n=row)
                )
                self.table.setVerticalHeaderItem(row, header)
            if along:
                self.hint.setText(
                    fmt(
                        _(
                            "{name} walks through its points at {speed:.2f} m/s, turning at "
                            "{turn:.0f}°/s through the corners: Convert to keyframes to edit it "
                            "here."
                        ),
                        name=item.name,
                        speed=walk.speed_m_s,
                        turn=walk.turn_deg_s,
                    )
                )
            elif timed:
                self.hint.setText(
                    _(
                        "Between two keyframes it moves and turns evenly: a keyframe at the "
                        "same place waits there, one with other angles turns on the spot."
                    )
                )
            else:
                self.hint.setText(
                    _(
                        "Move the playhead to when it should be somewhere else and add a "
                        "keyframe there, then drag the keyframe into place."
                    )
                )
        finally:
            self._loading = False

    def _row_picked(self) -> None:
        """A row was selected: the playhead goes to its keyframe."""
        row = self.table.currentRow()
        if not self._loading and 0 <= row < len(self._times) and self.item is not None:
            self.set_time(float(self._times[row]))

    # -- edits ----------------------------------------------------------------------

    def _run(self, action: Callable[[], object]) -> bool:
        """Run a document edit, reporting one that is refused."""
        try:
            action()
            return True
        except EditError as exc:
            self.failed.emit(str(exc))
            self._fill()
            return False

    def _edit_keyframes(self, description: str, change: Callable[[list[dict]], None]) -> bool:
        """Change the keyframes of the mover shown with ``change`` (on the list of their
        YAML entries, in order of time afterwards); none left takes the walk away."""
        kind, index = self.key

        def change_data(data: dict) -> None:
            entry = data[kind][index]
            walk = entry.get("walk") or {}
            frames = list(walk.get("keyframes") or [])
            change(frames)
            if not frames:
                entry.pop("walk", None)
                return
            frames.sort(key=lambda frame: frame["time_s"])
            walk["keyframes"], walk["to"] = frames, []
            entry["walk"] = walk

        return self._run(lambda: self.document.edit(description, change_data))

    def _cell_changed(self, cell: QTableWidgetItem) -> None:
        if self._loading or self.item is None:
            return
        row, col = cell.row(), cell.column()
        try:
            value = parse_number(cell.text())
            if not math.isfinite(value):
                raise ValueError
        except ValueError:
            self.failed.emit(_("{text} is not a number").format(text=cell.text().strip()))
            self._fill()
            return
        kind, index = self.key
        item = self.item
        if row == 0:
            if X <= col <= Z:
                position = list(item.position)
                position[col - X] = value
                self._run(lambda: self.document.update_point(kind, index, position=position))
            elif YAW <= col <= ROLL:
                orientation = item.orientation.model_dump()
                orientation[ANGLES[col - YAW]] = value
                self._run(lambda: self.document.update_point(kind, index, orientation=orientation))
            return
        k = row - 1
        if col == TIME:
            if value <= 0.0:
                self.failed.emit(_("a keyframe comes after the start: give it a time above 0 s"))
                self._fill()
                return

            def retime(frames: list[dict]) -> None:
                frames[k]["time_s"] = value

            self._edit_keyframes(_("retime keyframe"), retime)
        elif X <= col <= Z:

            def displace(frames: list[dict]) -> None:
                position = list(frames[k]["position"])
                position[col - X] = value
                frames[k]["position"] = _place(position)

            self._edit_keyframes(_("move keyframe"), displace)
        elif YAW <= col <= ROLL:
            angles = [float(a) for a in self._angles[row]]
            angles[col - YAW] = value

            def turn(frames: list[dict]) -> None:
                frames[k]["orientation"] = dict(zip(ANGLES, angles, strict=True))

            self._edit_keyframes(_("turn keyframe"), turn)

    def move_keyframe(self, kind: str, index: int, row: int, position) -> bool:
        """Put keyframe ``row`` of a mover (row 0: the mover itself) at ``position`` (m),
        from a drag in the plan or the 3D view."""
        if (kind, index) != self.key:
            return False
        if row == 0:
            return self._run(lambda: self.document.move_point(kind, index, position))

        def displace(frames: list[dict]) -> None:
            frames[row - 1]["position"] = _place(position)

        return self._edit_keyframes(_("move keyframe"), displace)

    def add_keyframe(self) -> None:
        """A keyframe at the playhead, where the mover is then; on a keyframe already (or
        at 0 s), :data:`STEP_S` past the end of its walk, where it ends."""
        item = self.item
        if item is None:
            return
        if item.walk is not None and item.walk.keyframes is None:
            self.failed.emit(
                _("{name} walks through points at a speed: Convert to keyframes first").format(
                    name=item.name
                )
            )
            return
        path = path_of(item)
        times = path.times if path is not None else np.zeros(1)
        time = self._time
        if time <= 0.0 or np.any(np.abs(times - time) < 1e-6):
            time = float(times[-1]) + STEP_S
        time = round(time, DECIMALS)
        if path is None:
            position, angles, turning = item.position, _angles(item.orientation), False
        else:
            position, angles = path.at(time)
            # Inside a turn, the facing it has there, so that the turn goes on as before;
            # elsewhere it keeps facing as at the keyframe before, as a new one does.
            k = int(np.searchsorted(times, time, side="right")) - 1
            turning = k + 1 < len(times) and not np.allclose(path.angles[k], path.angles[k + 1])
        frame: dict = {"time_s": time, "position": _place(position)}
        if turning:
            frame["orientation"] = dict(zip(ANGLES, _place(angles), strict=True))

        def add(frames: list[dict]) -> None:
            frames.append(frame)

        if self._edit_keyframes(_("add keyframe"), add):
            row = int(np.searchsorted(self._times, time))
            self.table.setCurrentCell(row, X)
            self.set_time(time)

    def remove_keyframe(self) -> None:
        """Take the selected keyframe away; the last one takes the walk with it."""
        row = self.table.currentRow()
        item = self.item
        if item is None or item.walk is None or item.walk.keyframes is None:
            return
        if row < 1:
            self.message.emit(
                _("Select a keyframe to remove; the first row is where {name} starts.").format(
                    name=item.name
                )
            )
            return

        def drop(frames: list[dict]) -> None:
            del frames[row - 1]

        if self._edit_keyframes(_("remove keyframe"), drop):
            self.table.setCurrentCell(min(row, self.table.rowCount() - 1), X)

    def convert_walk(self) -> None:
        """Keyframes for a walk along ``to``: each knot of the walk as it goes, its turns
        through the corners included, so that it moves the same way; facing forward, it
        faces its first leg from the start, so its own orientation turns that way too."""
        item = self.item
        path = path_of(item) if item is not None else None
        if path is None or item.walk.keyframes is not None:
            return
        kind, index = self.key
        angles = (path.angles + 180.0) % 360.0 - 180.0
        frames = [
            {
                "time_s": round(float(path.times[k]), DECIMALS),
                "position": _place(path.positions[k]),
                "orientation": dict(zip(ANGLES, _place(angles[k]), strict=True)),
            }
            for k in range(1, len(path.times))
        ]

        def convert(data: dict) -> None:
            entry = data[kind][index]
            entry["orientation"] = dict(zip(ANGLES, _place(angles[0]), strict=True))
            entry["walk"]["keyframes"], entry["walk"]["to"] = frames, []

        self._run(lambda: self.document.edit(_("convert walk to keyframes"), convert))

    def _moves(self, on: bool) -> None:
        if not self._loading and self.key is not None:
            kind, index = self.key
            self._run(lambda: self.document.set_walk_enabled(kind, index, on))

    # -- the timeline -------------------------------------------------------------------

    def _set_range(self) -> None:
        """The timeline runs to the longest walk at least, and to End."""
        longest = movement_duration(self.document.scene)
        self._end = max(self._end, longest)
        self._loading = True
        try:
            self.end_box.setMinimum(longest)
            self.end_box.setValue(self._end)
            self.time_box.setMaximum(self._end)
        finally:
            self._loading = False
        self.set_time(self._time)

    def _end_changed(self, end: float) -> None:
        if not self._loading:
            self._end = end
            self._set_range()

    def set_time(self, time_s: float) -> None:
        """Put the playhead at ``time_s`` (s, kept within the timeline) and show every
        mover there."""
        time_s = min(max(float(time_s), 0.0), self._end)
        self._time = time_s
        loading, self._loading = self._loading, True
        try:
            self.time_box.setValue(time_s)
            self.slider.setValue(round(SLIDER_STEPS * time_s / self._end) if self._end else 0)
        finally:
            self._loading = loading
        self.preview_changed.emit(time_s)

    def _slid(self, value: int) -> None:
        if not self._loading:
            self.pause()
            self.set_time(self._end * value / SLIDER_STEPS)

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def toggle_play(self) -> None:
        """Play from the playhead to the end (from the start, at the end), or pause."""
        if self.playing:
            self.pause()
            return
        if self._end <= 0.0:
            return
        if self._time >= self._end:
            self.set_time(0.0)
        self._origin = self._time
        self._clock.start()
        self._timer.start()
        self.play.setText(_("Pause"))

    def _tick(self) -> None:
        time_s = self._origin + self._clock.elapsed() / 1000.0
        if time_s >= self._end:
            self.set_time(self._end)
            self.pause()
        else:
            self.set_time(time_s)

    def pause(self) -> None:
        self._timer.stop()
        self.play.setText(_("Play"))

    def hideEvent(self, event) -> None:
        self.pause()
        super().hideEvent(event)


__all__ = [
    "ANGLES",
    "HEADERS",
    "KINDS",
    "MovementPanel",
    "handles_of",
    "movement_duration",
    "moves_at",
    "path_of",
    "pose_at",
]
