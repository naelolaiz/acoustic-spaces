"""The editor's main window: the 3D view and the plan in the middle, the scene tree,
the form of the selected item, the results of the last run, the coverage map and a
message log around them."""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QGuiApplication,
    QKeySequence,
    QShortcut,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QStyle,
    QTabWidget,
)

from acoustic_spaces import __version__
from acoustic_spaces.engine import reweight
from acoustic_spaces.examples import list_examples
from acoustic_spaces.geometry import points_inside
from acoustic_spaces.gui.document import (
    LISTENER_HEIGHT,
    SOURCE_HEIGHT,
    Document,
    EditError,
    box_scene,
    model_scene,
)
from acoustic_spaces.gui.inspector import BAND_LABELS, Inspector, label_text
from acoustic_spaces.gui.map_view import MapPanel
from acoustic_spaces.gui.plan_view import PlanView
from acoustic_spaces.gui.results_view import ResultsPanel
from acoustic_spaces.gui.tree import SceneTree
from acoustic_spaces.gui.viewport import Viewport
from acoustic_spaces.gui.worker import MapJob, SimulationJob
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, builtin_database

TITLE = "Acoustic Spaces editor"
SCENE_FILTER = "Scenes (*.yaml *.yml)"
MODEL_FILTER = "Models (*.obj *.stl *.ply *.off *.gltf *.glb *.dxf *.skp)"
GLB_FILTER = "glTF binary (*.glb)"
MAP_FILTERS = "PNG picture (*.png);;JSON, every map (*.json)"
#: Kinds of item Edit > Remove works on.
REMOVABLE = ("sources", "listeners", "material")


@contextmanager
def busy():
    """A wait cursor while something runs in the window's own thread."""
    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QGuiApplication.restoreOverrideCursor()


class MainWindow(QMainWindow):
    """One scene being edited, simulated and listened to."""

    def __init__(self, document: Document | None = None) -> None:
        super().__init__()
        self.document = document or Document(box_scene())
        self.selection: tuple[str, object] = ("scene", None)
        self.result = None
        self._result_scene = None
        #: The run as traced, for re-weighting it when only absorption changes: its
        #: result, scene, face groups and their materials; None when it cannot be.
        self._traced = None
        self._job_groups = None
        self.coverage = None
        self._coverage_scene = None
        self.job: SimulationJob | None = None
        self._job_scene = None
        self._job_document: Document | None = None
        self._started = 0.0
        #: The document is a bundled example: Save asks for a new place.
        self.from_example = False

        self.viewport = Viewport()
        self.plan = PlanView()
        self.views = QTabWidget()
        self.views.setDocumentMode(True)
        self.views.addTab(self.viewport, "3D view")
        self.views.addTab(self.plan, "Plan")
        self.views.setTabToolTip(0, "The room in 3D: click to select, drag sources and listeners")
        self.views.setTabToolTip(
            1, "The room from above: draw floor plans, drag corners and points"
        )
        self.setCentralWidget(self.views)
        self.tree = SceneTree()
        self.inspector = Inspector(lambda: self.document)
        self.results = ResultsPanel()
        self.map_panel = MapPanel()
        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setMaximumBlockCount(2000)
        # Forms scroll rather than make the window as tall as the longest one.
        forms = QScrollArea()
        forms.setWidgetResizable(True)
        forms.setFrameShape(QFrame.Shape.NoFrame)
        forms.setWidget(self.inspector)
        self.docks = {}
        for name, title, widget, area in (
            ("tree", "Scene", self.tree, Qt.DockWidgetArea.LeftDockWidgetArea),
            ("inspector", "Properties", forms, Qt.DockWidgetArea.RightDockWidgetArea),
            ("results", "Results", self.results, Qt.DockWidgetArea.BottomDockWidgetArea),
            ("map", "Coverage map", self.map_panel, Qt.DockWidgetArea.BottomDockWidgetArea),
            ("messages", "Messages", self.messages, Qt.DockWidgetArea.BottomDockWidgetArea),
        ):
            dock = QDockWidget(title, self)
            dock.setObjectName(name)
            dock.setWidget(widget)
            self.addDockWidget(area, dock)
            self.docks[name] = dock
        self.tabifyDockWidget(self.docks["results"], self.docks["map"])
        self.tabifyDockWidget(self.docks["map"], self.docks["messages"])
        self.docks["results"].raise_()

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        self.summary = QLabel()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.summary)
        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(lambda: self.results.set_playing(False))
        # Head-tracked playback runs in a thread: (thread, stop, outcome).
        self._player: tuple[threading.Thread, threading.Event, dict] | None = None
        self._player_timer = QTimer(self)
        self._player_timer.setInterval(100)
        self._player_timer.timeout.connect(self._player_poll)

        self._actions()
        self._menus()
        self.tree.selected.connect(lambda kind, key: self.select(kind, key, from_tree=True))
        self.viewport.picked.connect(self._picked)
        self.viewport.place.connect(self._place)
        self.viewport.moved.connect(self._move_point)
        self.plan.picked.connect(self._picked)
        self.plan.place.connect(self._place)
        self.plan.moved.connect(self._move_point)
        self.plan.corners_changed.connect(self._set_corners)
        self.plan.drawn.connect(self._drawn)
        self.plan.message.connect(self.log)
        self.inspector.failed.connect(self._refused)
        self.inspector.select.connect(self.select)
        self.results.pair_changed.connect(lambda *_: self._draw())
        self.results.paths_toggled.connect(self._paths_toggled)
        self.results.export_requested.connect(self.export_results)
        self.results.listen_requested.connect(self.listen)
        self.results.stop_requested.connect(self.stop_playing)
        self.results.head.message.connect(self.log)
        self.map_panel.changed.connect(self._draw)
        self.map_panel.run_requested.connect(self.run_map)
        self.map_panel.save_requested.connect(self.save_map)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._escape)
        # Delete removes the selected item only from the tree or the 3D view, so the key
        # keeps its meaning in the forms.
        for widget in (self.tree, self.viewport, self.plan):
            shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), widget)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self.remove_selected)

        self.resize(1400, 900)
        self._sized = False
        self._attach(self.document)
        if not self.viewport.available:
            self.log(self.viewport.note)

    # -- menus ----------------------------------------------------------------------

    def _action(self, text: str, slot, shortcut=None, tip: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
        action.triggered.connect(lambda *_: slot())
        return action

    def _actions(self) -> None:
        keys = QKeySequence.StandardKey
        a = self._action
        self.new_action = a("&New box room…", self.new_box, keys.New, "Start from a box")
        self.new_model_action = a(
            "New from &model file…", self.new_from_model, None, "Start from an OBJ or DXF model"
        )
        self.new_plan_action = a(
            "New room from a &floor plan…",
            self.new_floor_plan,
            None,
            "Draw the floor plan of a new room, corner by corner",
        )
        self.open_action = a("&Open…", self.open_dialog, keys.Open)
        self.save_action = a("&Save", self.save, keys.Save)
        self.save_as_action = a("Save &as…", self.save_as, keys.SaveAs)
        self.export_action = a("&Export results…", self.export_results, None)
        self.blender_action = a(
            "Export to &Blender…",
            self.export_blender,
            tip="The room, and the reflection paths on show, as glTF",
        )
        self.particles_action = a(
            "Export &particle animation to Blender…",
            lambda: self.export_blender(particles=True),
            tip="Sound particles from the selected source (or the first) as a glTF animation",
        )
        self.quit_action = a("&Quit", self.close, keys.Quit)
        self.undo_action = a("&Undo", self.undo, keys.Undo)
        self.redo_action = a("&Redo", self.redo, keys.Redo)
        self.place_source_action = a(
            "Place &source",
            lambda: self.start_placing("sources"),
            tip="Click a floor or wall in the 3D view to put a source there",
        )
        self.place_listener_action = a(
            "Place &listener",
            lambda: self.start_placing("listeners"),
            tip="Click a floor or wall in the 3D view to put a listener there",
        )
        self.draw_action = a(
            "&Draw floor plan",
            self.draw_room,
            None,
            "Draw a new floor plan in the plan view; it replaces the room",
        )
        self.add_material_action = a("Add &material from the database…", self.add_material)
        self.remove_action = a("Re&move selected", self.remove_selected)
        self.reset_camera_action = a("&Reset camera", self.viewport.reset_camera, "Ctrl+0")
        self.show_3d_action = a(
            "&3D view", lambda: self.views.setCurrentWidget(self.viewport), "Ctrl+1"
        )
        self.show_plan_action = a("P&lan", lambda: self.views.setCurrentWidget(self.plan), "Ctrl+2")
        self.paths_action = a("Show reflection &paths", lambda: None)
        self.paths_action.setCheckable(True)
        self.paths_action.toggled.connect(self.results.paths.setChecked)
        self.screenshot_action = a("Save &screenshot…", self.save_screenshot)
        self.run_action = a("&Run", self.run_simulation, "F5", "Simulate every source and listener")
        self.map_action = a(
            "Run coverage &map",
            self.run_map,
            "F6",
            "Room parameters on a grid over the audience area, for every source",
        )
        self.map_action.setIconText("Map")
        self.cancel_action = a("&Cancel", self.cancel_simulation, "Shift+F5")
        self.cancel_action.setEnabled(False)
        self.about_action = a("&About", self.about)
        self.undo_action.setIconText("Undo")
        self.redo_action.setIconText("Redo")
        icon = self.style().standardIcon
        pixmaps = QStyle.StandardPixmap
        for action, pixmap in (
            (self.open_action, pixmaps.SP_DialogOpenButton),
            (self.save_action, pixmaps.SP_DialogSaveButton),
            (self.undo_action, pixmaps.SP_ArrowBack),
            (self.redo_action, pixmaps.SP_ArrowForward),
            (self.run_action, pixmaps.SP_MediaPlay),
            (self.map_action, pixmaps.SP_FileDialogDetailedView),
            (self.cancel_action, pixmaps.SP_MediaStop),
        ):
            action.setIcon(icon(pixmap))

    def _menus(self) -> None:
        bar = self.menuBar()
        file = bar.addMenu("&File")
        file.addActions(
            [self.new_action, self.new_plan_action, self.new_model_action, self.open_action]
        )
        self.examples_menu = file.addMenu("Open e&xample")
        try:
            examples = list_examples()
        except FileNotFoundError:
            examples = []
        for example in examples:
            action = self.examples_menu.addAction(example.title)
            action.setStatusTip(example.description.split(". ")[0])
            action.triggered.connect(lambda *_, path=example.path: self.open_example(path))
        self.examples_menu.setEnabled(bool(examples))
        file.addSeparator()
        file.addActions([self.save_action, self.save_as_action, self.export_action])
        file.addActions([self.blender_action, self.particles_action])
        file.addSeparator()
        file.addAction(self.quit_action)

        edit = bar.addMenu("&Edit")
        edit.addActions([self.undo_action, self.redo_action])
        edit.addSeparator()
        edit.addActions([self.place_source_action, self.place_listener_action])
        edit.addAction(self.draw_action)
        edit.addAction(self.add_material_action)
        edit.addSeparator()
        edit.addAction(self.remove_action)

        view = bar.addMenu("&View")
        view.addActions([self.show_3d_action, self.show_plan_action])
        view.addSeparator()
        view.addActions([self.reset_camera_action, self.paths_action, self.screenshot_action])
        view.addSeparator()
        for dock in self.docks.values():
            view.addAction(dock.toggleViewAction())

        simulation = bar.addMenu("&Simulation")
        simulation.addActions([self.run_action, self.map_action, self.cancel_action])
        bar.addMenu("&Help").addAction(self.about_action)

        tools = self.addToolBar("Main")
        tools.setObjectName("main-toolbar")
        tools.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tools.addActions([self.open_action, self.save_action])
        tools.addSeparator()
        tools.addActions([self.undo_action, self.redo_action])
        tools.addSeparator()
        tools.addActions([self.place_source_action, self.place_listener_action])
        tools.addSeparator()
        tools.addActions([self.run_action, self.map_action, self.cancel_action])

    # -- the document ---------------------------------------------------------------

    def _attach(self, document: Document) -> None:
        """Make ``document`` the one being edited."""
        old = self.document
        if old is not document:
            old.changed.disconnect(self._changed)
            old.state_changed.disconnect(self._state)
            if self.job is not None:  # its results would be of a scene no longer open
                self.job.cancel()
        self.document = document
        document.changed.connect(self._changed)
        document.state_changed.connect(self._state)
        self.result = self._result_scene = self._traced = None
        self.results.set_result(None)
        self.coverage = self._coverage_scene = None
        self.map_panel.set_result(None)
        self.selection = ("scene", None)
        self._changed()
        self.plan.cancel()
        self.viewport.start_placing(None)
        self.plan.fit()
        self.viewport.reset_camera()  # edits keep the camera; another room starts afresh
        self._state()

    def _changed(self) -> None:
        if not self._valid(self.selection):
            self.selection = ("scene", None)
        self.tree.rebuild(self.document.room, self.selection)
        self.inspector.show_item(*self.selection)
        self._reweight()
        self.results.mark_stale(self._stale(), reweighted=self._reweighted())
        self.map_panel.mark_stale(self._map_stale())
        self._draw()
        self._summary()

    def _state(self) -> None:
        document = self.document
        name = document.path.name if document.path is not None else "untitled"
        self.setWindowTitle(f"{name}[*] - {TITLE}")
        self.setWindowModified(document.modified)
        undo, redo = document.undo_text(), document.redo_text()
        self.undo_action.setText(f"&Undo {undo}" if undo else "&Undo")
        self.redo_action.setText(f"&Redo {redo}" if redo else "&Redo")
        self.undo_action.setToolTip(f"Undo {undo}" if undo else "Nothing to undo")
        self.redo_action.setToolTip(f"Redo {redo}" if redo else "Nothing to redo")
        self.undo_action.setEnabled(undo is not None)
        self.redo_action.setEnabled(redo is not None)

    def _valid(self, selection: tuple[str, object]) -> bool:
        kind, key = selection
        scene, room = self.document.scene, self.document.room
        if kind == "surface":
            return isinstance(key, int) and 0 <= key < len(room.mesh.labels)
        if kind == "material":
            return key in scene.materials
        if kind in ("sources", "listeners"):
            return isinstance(key, int) and 0 <= key < len(getattr(scene, kind))
        return True

    def _summary(self) -> None:
        room = self.document.room
        parts = []
        if room.volume_m3:
            k = OCTAVE_BANDS_HZ.index(1000)
            rt = room.reverberation_time("eyring", air_absorption=True)[k]
            parts += [f"{room.volume_m3:,.0f} m³", f"RT {rt:.2f} s at {BAND_LABELS[k]} (Eyring)"]
        else:
            parts.append("model not closed")
        problems = room.problems()
        if problems:
            parts.append(f"{len(problems)} problem{'s' if len(problems) > 1 else ''}")
        self.summary.setText(", ".join(parts))
        self.summary.setToolTip("\n".join(problems))

    def log(self, text: str) -> None:
        self.messages.appendPlainText(text)
        self.statusBar().showMessage(text, 8000)

    def _refused(self, message: str) -> None:
        self.log(f"Not changed: {message}")

    # -- selection and the 3D view ------------------------------------------------

    def select(self, kind: str, key=None, *, from_tree: bool = False) -> None:
        """Select an item (from the tree, the 3D view or a form) and show its form."""
        selection = (kind, key)
        if not self._valid(selection):
            selection = ("scene", None)
        self.selection = selection
        if not from_tree:
            self.tree.select(selection)
        self.inspector.show_item(*selection)
        self._follow(selection)
        self._draw()

    def _follow(self, selection: tuple[str, object]) -> None:
        """Show the results of a selected source or listener."""
        kind, key = selection
        if kind in ("sources", "listeners"):
            name = getattr(self.document.scene, kind)[key].name
            if kind == "sources":
                self.results.show_pair(source=name)
            else:
                self.results.show_pair(listener=name)

    def _picked(self, kind: str, key) -> None:
        """A click in the 3D view; empty space selects the scene."""
        self.select(kind or "scene", key)
        if kind == "surface":
            self.log(f"Surface {label_text(self.document.room.mesh.labels[key])}")

    def _shown_response(self):
        """The response whose reflection paths the 3D view shows, if any."""
        if self.results.paths.isChecked() and not self._stale():
            return self.results.response()
        return None

    def _draw(self) -> None:
        self.remove_action.setEnabled(self.selection[0] in REMOVABLE)
        response = self._shown_response()
        paths = None if response is None else response.early
        coverage = None if self._map_stale() else self.map_panel.shown_in_3d()
        self.viewport.show_room(self.document.room, self.selection, paths, coverage)
        self.plan.show_room(self.document.room, self.selection)

    def _paths_toggled(self, on: bool) -> None:
        self.paths_action.blockSignals(True)
        self.paths_action.setChecked(on)
        self.paths_action.blockSignals(False)
        if on and self.result is None:
            self.log("Reflection paths come from a run: run the simulation (F5) first.")
        self._draw()

    def start_placing(self, kind: str) -> None:
        """Put a new source or listener where the next click on a surface (in the plan,
        on the floor) lands."""
        what = "source" if kind == "sources" else "listener"
        if self.views.currentWidget() is self.plan:
            self.plan.start_placing(kind)
            self.statusBar().showMessage(f"Click in the plan to place the {what}; Esc cancels.", 0)
        elif self.viewport.available:
            self.viewport.start_placing(kind)
            self.statusBar().showMessage(
                f"Click a floor or wall to place the {what}; Esc cancels.", 0
            )
        else:
            self._place(kind, self._default_position(kind))

    def _escape(self) -> None:
        if self.viewport.placing is not None:
            self.viewport.start_placing(None)
        self.plan.cancel()
        self.statusBar().clearMessage()

    def _default_position(self, kind: str) -> tuple[float, float, float]:
        """Where a new source or listener goes without a click: above the middle of the
        floor, or the middle of the model if that is not inside."""
        room = self.document.room
        lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
        height = SOURCE_HEIGHT if kind == "sources" else LISTENER_HEIGHT
        point = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + height])
        if room.report.closed and not points_inside(room.mesh, point[None])[0]:
            point = (lo + hi) / 2
        return tuple(float(v) for v in point)

    def _place(self, kind: str, position: tuple) -> None:
        self.statusBar().clearMessage()
        try:
            index = self.document.add_point(kind, position)
        except EditError as exc:
            self._refused(str(exc))
            return
        item = getattr(self.document.scene, kind)[index]
        where = ", ".join(f"{v:.2f}" for v in position)
        self.log(f"Added {item.name} at ({where})")
        self.select(kind, index)

    def _move_point(self, kind: str, index: int, position: tuple) -> None:
        """A source or listener dragged in the 3D view or the plan."""
        try:
            self.document.move_point(kind, index, position)
        except EditError as exc:
            self._refused(str(exc))
            self._draw()
            return
        item = getattr(self.document.scene, kind)[index]
        where = ", ".join(f"{v:.2f}" for v in item.position)
        self.log(f"Moved {item.name} to ({where})")
        self.select(kind, index)

    def _set_corners(self, corners: list, description: str) -> None:
        """New corners for the floor plan, from the plan view."""
        was_box = self.document.scene.geometry.box is not None
        try:
            self.document.set_floor_plan(corners, description)
        except EditError as exc:
            self._refused(str(exc))
            self._draw()
            return
        if was_box:
            self.log("The box is now a floor plan with the same walls.")

    def draw_room(self) -> None:
        """Draw a new floor plan in the plan view."""
        self.views.setCurrentWidget(self.plan)
        self.plan.start_drawing()

    def _drawn(self, corners: list) -> None:
        if self.document.scene.geometry.file is not None and not self.confirm_replace_model():
            return
        default = self.document.scene.default_material
        try:
            moved = self.document.draw_floor_plan(corners)
        except EditError as exc:
            self._refused(str(exc))
            return
        self.plan.fit()
        self.viewport.reset_camera()
        height = self.document.scene.geometry.height
        self.log(
            f"New floor plan with {len(corners)} corners, {height:g} m high; change the "
            "height under Geometry."
        )
        if moved:
            self.log(f"Moved {', '.join(moved)} into the new room.")
        if self.document.scene.default_material != default:
            self.log(
                f"Surfaces without an assignment are now {self.document.scene.default_material} "
                "(the scene's default material)."
            )
        self.select("geometry", None)

    def confirm_replace_model(self) -> bool:
        answer = QMessageBox.question(
            self,
            TITLE,
            "Replace the model file with the drawn floor plan? Material assignments to the "
            "model's surfaces will no longer match anything.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # -- editing --------------------------------------------------------------------

    def undo(self) -> None:
        self.document.undo()

    def redo(self) -> None:
        self.document.redo()

    def add_material(self) -> None:
        names = sorted(e.id for e in builtin_database().entries("absorption"))
        preset, ok = QInputDialog.getItem(
            self, "Add material", "Material from the database:", names, 0, True
        )
        if not ok or not preset:
            return
        try:
            name = self.document.add_material(preset)
        except EditError as exc:
            self._refused(str(exc))
            return
        self.select("material", name)

    def remove_selected(self) -> None:
        kind, key = self.selection
        try:
            if kind in ("sources", "listeners"):
                self.document.remove_point(kind, key)
            elif kind == "material":
                self.document.remove_material(key)
            else:
                return
        except EditError as exc:
            self._refused(str(exc))
            return
        self.select("scene", None)

    # -- files ----------------------------------------------------------------------

    def _discard_ok(self) -> bool:
        """True when unsaved changes may go: none, saved now, or the user says so."""
        if not self.document.modified:
            return True
        answer = self.ask_save()
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def ask_save(self) -> QMessageBox.StandardButton:
        buttons = (
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        return QMessageBox.question(
            self, TITLE, "The scene has unsaved changes. Save them?", buttons
        )

    def new_box(self) -> None:
        if not self._discard_ok():
            return
        self.from_example = False
        self._attach(Document(box_scene()))
        self.log("New box room; change its size under Geometry.")

    def new_floor_plan(self) -> None:
        """A new room whose floor plan the user draws in the plan view."""
        if not self._discard_ok():
            return
        self.from_example = False
        self._attach(Document(box_scene()))
        self.draw_room()
        self.log("New room: click its corners in the plan, and close it on the first one.")

    def new_from_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Model file", "", MODEL_FILTER)
        if not path or not self._discard_ok():
            return
        try:
            with busy():
                document = Document(model_scene(path))
        except EditError as exc:
            self._error(f"Could not load {Path(path).name}", str(exc))
            return
        self.from_example = False
        self._attach(document)
        self.log(
            f"Loaded {Path(path).name}; set its units and up axis under Geometry if it "
            "looks wrong, then assign materials to its surfaces."
        )

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open scene", "", SCENE_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path, *, example: bool = False) -> bool:
        if not self._discard_ok():
            return False
        try:
            with busy():
                document = Document.open(path)
        except EditError as exc:
            self._error(f"Could not open {Path(path).name}", str(exc))
            return False
        self.from_example = example
        self._attach(document)
        self.log(f"Opened {path}")
        return True

    def open_example(self, path: str | Path) -> bool:
        """Open a bundled example; saving it asks where to keep a copy."""
        return self.open_file(path, example=True)

    def save(self) -> bool:
        if self.document.path is None or self.from_example:
            return self.save_as()
        return self._save_to(self.document.path)

    def save_as(self) -> bool:
        start = self.document.path
        if start is None or self.from_example:
            start = Path.home() / (start.parent.name if start else "scene") / "scene.yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Save scene", str(start), SCENE_FILTER)
        if path and not Path(path).suffix:
            path += ".yaml"
            # The dialog asked before replacing the name as typed, not this one.
            if Path(path).exists() and not self._confirm_overwrite(Path(path)):
                return False
        return bool(path) and self._save_to(Path(path))

    def _confirm_overwrite(self, path: Path) -> bool:
        answer = QMessageBox.question(
            self,
            TITLE,
            f"{path.name} already exists. Replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _save_to(self, path: Path) -> bool:
        scene = self.document.scene
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.document.save(path, copy_files=self.from_example)
        except (OSError, ValueError) as exc:
            self._error("Could not save", str(exc))
            return False
        if self.document.scene is not scene:
            self._files_moved(scene, self.document.scene)
        self.from_example = False
        self.log(f"Saved {path}")
        return True

    def _files_moved(self, old, new) -> None:
        """The scene ``old`` became ``new``, the same scene with copies of its files (an
        example saved elsewhere): runs and maps of ``old`` hold for ``new``."""
        if self._result_scene is old:
            self._result_scene = new
        if self._coverage_scene is old:
            self._coverage_scene = new
        if self._job_scene is old:
            self._job_scene = new
        self._changed()

    def save_screenshot(self) -> None:
        if not self.viewport.available:
            self.log(self.viewport.note)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "room.png", "PNG (*.png)")
        if path:
            self.viewport.screenshot(path)
            self.log(f"Saved {path}")

    def export_blender(self, particles: bool = False) -> bool:
        """Write the room as glTF for Blender, with the reflection paths on show, or with
        sound particles from the selected source (or the first) as an animation."""
        from acoustic_spaces.engine import trace_particles
        from acoustic_spaces.gltf import PARTICLE_COUNT, SLOW_MOTION, export_gltf

        title = "Export particle animation to Blender" if particles else "Export to Blender"
        start = "particles.glb" if particles else "room.glb"
        path, _ = QFileDialog.getSaveFileName(self, title, start, GLB_FILTER)
        if not path:
            return False
        room = self.document.room
        response = None if particles else self._shown_response()
        traced = None
        try:
            with busy():
                if particles:
                    kind, key = self.selection
                    source = room.scene.sources[key if kind == "sources" else 0].name
                    traced = trace_particles(room, source, count=PARTICLE_COUNT)
                paths = None if response is None else response.early
                export_gltf(room, path, paths=paths, particles=traced)
        except (OSError, ValueError) as exc:
            self._error("Could not export", str(exc))
            return False
        if traced is not None:
            sound = float(traced.times[-1])
            what = (
                f"{PARTICLE_COUNT} particles from {traced.source}, {1e3 * sound:g} ms of "
                f"sound over {SLOW_MOTION * sound:g} s"
            )
        elif response is not None:
            what = (
                f"the room with {len(response.early)} reflection paths from "
                f"{response.source} to {response.listener}"
            )
        else:
            what = "the room"
        self.log(f"Wrote {path}: {what}. In Blender: File > Import > glTF 2.0.")
        return True

    def _error(self, title: str, text: str) -> None:
        self.log(f"{title}: {text}")
        QMessageBox.warning(self, title, text)

    # -- simulation -----------------------------------------------------------------

    def _stale(self) -> bool:
        return self.result is not None and self._result_scene is not self.document.scene

    def _reweighted(self) -> bool:
        """True when the results on show re-weight the run for other absorption."""
        return self._traced is not None and self.result is not self._traced[0]

    def _reweight(self) -> None:
        """When the scene differs from the run's in absorption alone, show the run
        re-weighted for it, at once (:func:`~acoustic_spaces.engine.reweight.
        reweighted_result`); otherwise leave the results to be marked out of date."""
        if self._traced is None or self._result_scene is self.document.scene:
            return
        result, scene, face_group, materials = self._traced
        try:
            found = reweight.reweighted_result(
                result, scene, face_group, materials, self.document.scene
            )
        except ValueError:
            found = None
        if found is not None:
            self.result, self._result_scene = found, self.document.scene
            self.results.set_result(found)

    def _map_stale(self) -> bool:
        return self.coverage is not None and self._coverage_scene is not self.document.scene

    def run_simulation(self) -> None:
        if self.job is not None:
            return
        problems = self.document.room.problems()
        if problems:
            self.log("Cannot run: " + "; ".join(problems))
            self.docks["messages"].raise_()
            return
        # Count reflections off each material, so that changing absorption afterwards
        # re-weights the run instead of making it out of date.
        self._job_groups = reweight.groups_of(self.document.room)
        groups = self._job_groups[0] if self._job_groups is not None else None
        job = SimulationJob(self.document.room, groups=groups)
        job.done.connect(self._done)
        self._start(job, f"Simulating {self.document.scene.name}…")

    def run_map(self) -> None:
        """Compute the coverage map of every source in the background."""
        if self.job is not None:
            return
        # Listeners play no part in a map.
        problems = [p for p in self.document.room.problems() if not p.startswith("listener ")]
        if problems:
            self.log("Cannot map: " + "; ".join(problems))
            self.docks["messages"].raise_()
            return
        job = MapJob(self.document.room)
        job.done.connect(self._map_done)
        self._start(job, f"Mapping {self.document.scene.name}…")

    def _start(self, job: SimulationJob, message: str) -> None:
        self.job = job
        self._job_scene = self.document.scene
        self._job_document = self.document
        job.progress.connect(self._progress)
        # Bound methods, so Qt queues the worker's signals to this window's thread.
        job.failed.connect(self._failed)
        job.cancelled.connect(self._cancelled)
        job.finished.connect(self._job_ended)
        self._started = time.perf_counter()
        self.run_action.setEnabled(False)
        self.map_action.setEnabled(False)
        self.map_panel.set_running(True)
        self.cancel_action.setEnabled(True)
        self.progress.show()
        self.log(message)
        job.start()

    def cancel_simulation(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.log("Cancelling at the next stage…")

    def _progress(self, message: str) -> None:
        self.log(f"{time.perf_counter() - self._started:6.1f} s  {message}")

    def _closed_meanwhile(self) -> bool:
        """True (and a message) when the scene of the job that just ended is no longer
        open: another was opened or started while the job ran."""
        if self._job_document is self.document:
            return False
        self.log("Dropped the results of the run: its scene is no longer open.")
        return True

    def _done(self, result) -> None:
        if self._closed_meanwhile():
            return
        first = self.result is None
        self.result, self._result_scene = result, self._job_scene
        groups = self._job_groups
        self._traced = None if groups is None else (result, self._job_scene, *groups)
        self.results.set_result(result)  # keeps the source and listener on show
        if first:
            self._follow(self.selection)
        self._reweight()  # absorption changed while it ran
        self.results.mark_stale(self._stale(), reweighted=self._reweighted())
        self.docks["results"].raise_()
        elapsed = time.perf_counter() - self._started
        count = len(result.responses)
        self.log(f"Done in {elapsed:.1f} s: {count} response{'s' if count != 1 else ''}.")
        self._draw()

    def _map_done(self, result) -> None:
        if self._closed_meanwhile():
            return
        self.coverage, self._coverage_scene = result, self._job_scene
        self.map_panel.set_result(result, result.room)
        self.map_panel.mark_stale(self._map_stale())
        self.docks["map"].raise_()
        elapsed = time.perf_counter() - self._started
        self.log(f"Map done in {elapsed:.1f} s: {len(result.grid)} points.")
        self._draw()

    def _failed(self, message: str) -> None:
        self.log(f"The simulation failed: {message}")
        self.docks["messages"].raise_()

    def _cancelled(self) -> None:
        self.log("Simulation cancelled.")

    def _job_ended(self) -> None:
        job, self.job = self.job, None
        if job is not None:
            job.wait()
        self.run_action.setEnabled(True)
        self.map_action.setEnabled(True)
        self.map_panel.set_running(False)
        self.cancel_action.setEnabled(False)
        self.progress.hide()

    # -- results --------------------------------------------------------------------

    def export_results(self) -> None:
        if self.result is None:
            self.log("Nothing to export yet: run the simulation (F5) first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the impulse responses")
        if folder:
            self.write_results(folder)

    def write_results(self, folder: str | Path) -> dict | None:
        from acoustic_spaces.results import REPORT_FILE, write_results

        try:
            with busy():
                report = write_results(self.result, folder, progress=self.log)
        except (OSError, ValueError, ImportError) as exc:
            self._error("Could not export", str(exc))
            return None
        files = sum(len(r["outputs"]) for r in report["responses"])
        self.log(f"Wrote {files} impulse response files and {REPORT_FILE} to {folder}")
        if report["peak"] > 1.0:
            self.log(
                "Note: the float WAV files exceed full scale (a listener is closer than 1 m "
                "to a source); scale them before playing them back."
            )
        if self._stale():
            self.log("These results are from before the latest changes to the scene.")
        return report

    def save_map(self) -> None:
        if self.coverage is None:
            self.log("No map yet: run the coverage map (F6) first.")
            return
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save the coverage map", str(self.document.base_dir), MAP_FILTERS
        )
        if path:
            if not Path(path).suffix:
                path += ".json" if "json" in chosen.lower() else ".png"
            self.write_map(path)

    def write_map(self, path: str | Path) -> bool:
        """Save the map on show as a PNG picture, or every map as JSON (by the suffix)."""
        import json

        path = Path(path)
        try:
            with busy():
                if path.suffix.lower() == ".json":
                    path.write_text(json.dumps(self.coverage.as_dict(), indent=1), "utf-8")
                else:
                    from acoustic_spaces.viz import plot_map

                    panel = self.map_panel
                    plot_map(
                        panel.current(),
                        path,
                        parameter=panel.name(),
                        band=panel.band_index(),
                        room=panel.room,
                    )
        except (OSError, ValueError) as exc:
            self._error("Could not save the map", str(exc))
            return False
        self.log(f"Wrote {path}")
        if self._map_stale():
            self.log("This map is from before the latest changes to the scene.")
        return True

    def _listening_format(self):
        """Binaural (with the scene's HRTF, if it sets one) when the scene has a binaural
        output, otherwise mono."""
        from acoustic_spaces.spatial import Mono, from_output

        for output in self.result.room.scene.outputs:
            if output.format == "binaural":
                return from_output(output)
        return Mono()

    def listen(self) -> None:
        if self.results.pair() is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Dry recording to play in the room", "", "WAV files (*.wav)"
        )
        if path:
            self.play_file(path)

    def play_file(self, path: str | Path) -> bool:
        """Play a WAV file from the shown source to the shown listener."""
        from acoustic_spaces.audio import read_wav
        from acoustic_spaces.measurement import _sounddevice
        from acoustic_spaces.render import auralize

        pair = self.results.pair()
        if pair is None:
            return False
        if self.results.stop.isEnabled():  # one playback at a time
            self.stop_playing()
        if self.results.tracking.isChecked():
            return self.play_tracked(path)
        try:
            sd = _sounddevice()
            signal, rate = read_wav(path)
            fmt = self._listening_format()
            with busy():
                mix = auralize(self.result, {pair[0]: (signal, rate)}, pair[1], fmt)
        except (ImportError, OSError, ValueError) as exc:
            self._error("Cannot play", str(exc))
            return False
        peak = float(np.abs(mix).max(initial=0.0))
        if peak > 0:
            mix = mix * (10.0 ** (-1.0 / 20.0) / peak)
        sample_rate = int(self.result.settings.sample_rate)
        try:
            sd.play(mix.astype(np.float32), sample_rate)
        except Exception as exc:  # PortAudio errors are not one type
            self._error("Cannot play", str(exc))
            return False
        self.results.set_playing(True)
        self._play_timer.start(int(1000 * len(mix) / sample_rate) + 250)
        self.log(f"Playing {Path(path).name} from {pair[0]} at {pair[1]} ({fmt.label})")
        return True

    def _tracking_order(self) -> int:
        """The highest Ambisonics order among the scene's outputs, or 3."""
        orders = [o.order for o in self.result.room.scene.outputs if o.format == "ambisonics"]
        return max([*orders, 1]) if orders else 3

    def play_tracked(self, path: str | Path) -> bool:
        """Play a WAV file from the shown source to the shown listener binaurally, turned
        by the head of the head controls, in real time."""
        from acoustic_spaces.audio import read_wav
        from acoustic_spaces.headtracking import HeadTrackedBinaural
        from acoustic_spaces.measurement import _sounddevice
        from acoustic_spaces.render import resample
        from acoustic_spaces.spatial import Ambisonics, Binaural

        pair = self.results.pair()
        if pair is None:
            return False
        if self.results.stop.isEnabled():  # one playback at a time
            self.stop_playing()
        sample_rate = int(self.result.settings.sample_rate)
        order = self._tracking_order()
        fmt = self._listening_format()
        try:
            _sounddevice()
            signal, rate = read_wav(path)
            with busy():
                dry = resample(signal.mean(axis=1), rate, sample_rate)
                response = self.result[pair]
                ir = response.spatial_impulse_response(Ambisonics(order), sample_rate)
                renderer = HeadTrackedBinaural(
                    ir,
                    sample_rate,
                    hrtf=fmt.at(sample_rate) if isinstance(fmt, Binaural) else None,
                    orientation=self.results.head.head,
                    block=512,
                )
        except (ImportError, OSError, ValueError) as exc:
            self._error("Cannot play", str(exc))
            return False
        # Half the dry signal's level, so the room's reverberation does not clip.
        energy = float((ir[:, 0] ** 2).sum())
        gain = 0.5 / np.sqrt(energy) if energy > 0 else 1.0
        stop, outcome = threading.Event(), {}

        def run() -> None:
            from acoustic_spaces.render import play

            try:
                outcome["report"] = play(renderer, sample_rate, dry, gain=gain, stop=stop)
            except Exception as exc:  # PortAudio errors are not one type
                outcome["error"] = exc

        thread = threading.Thread(target=run, name="head-tracked-playback", daemon=True)
        self._player = (thread, stop, outcome)
        thread.start()
        self._player_timer.start()
        self.results.set_playing(True)
        self.log(
            f"Playing {Path(path).name} from {pair[0]} at {pair[1]}, head-tracked "
            f"(order {order} Ambisonics to binaural)"
        )
        return True

    def _player_poll(self) -> None:
        if self._player is None or self._player[0].is_alive():
            return
        _, _, outcome = self._player
        self._player = None
        self._player_timer.stop()
        self.results.set_playing(False)
        if "error" in outcome:
            self._error("Cannot play", str(outcome["error"]))
        elif outcome["report"].xruns:
            self.log(f"{outcome['report'].xruns} block(s) dropped out while playing.")

    def stop_playing(self) -> None:
        from acoustic_spaces.measurement import _sounddevice

        if self._player is not None:
            thread, stop, _ = self._player
            stop.set()
            thread.join(2.0)
            self._player_poll()
        self._play_timer.stop()
        self.results.set_playing(False)
        try:
            _sounddevice().stop()
        except ImportError:
            pass

    # -- the window -----------------------------------------------------------------

    def about(self) -> None:
        QMessageBox.about(
            self,
            TITLE,
            f"<p><b>Acoustic Spaces</b> {__version__}</p>"
            "<p>Room acoustics simulation: early reflections by ray tracing or image "
            "sources, the late tail by stochastic ray tracing, and binaural, Ambisonics "
            "and loudspeaker impulse responses.</p>"
            "<p>GPL-3.0-or-later. "
            '<a href="https://github.com/naelolaiz/acoustic_spaces_simulator">Source</a></p>',
        )

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._sized:  # dock sizes apply once the window is laid out
            self._sized = True
            QTimer.singleShot(0, self._size_docks)

    def _size_docks(self) -> None:
        horizontal, vertical = Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        self.resizeDocks([self.docks["tree"], self.docks["inspector"]], [330, 360], horizontal)
        height = max(340, int(self.height() / 2.5))
        self.resizeDocks([self.docks["results"], self.docks["messages"]], [height] * 2, vertical)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._discard_ok():
            event.ignore()
            return
        if self.job is not None:
            self.job.cancel()
            self.job.wait()
        if self.results.stop.isEnabled():
            self.stop_playing()
        self.results.head.stop()
        self.viewport.close_view()
        event.accept()


def run(argv: list[str] | None = None) -> int:
    """Open the editor (with the scene file in ``argv``, if any) and run until it closes."""
    argv = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication([sys.argv[0] if sys.argv else TITLE])
    app.setApplicationName(TITLE)
    window = MainWindow()
    if argv:
        window.open_file(argv[0])
    window.show()
    return app.exec()


__all__ = ["MainWindow", "run"]
