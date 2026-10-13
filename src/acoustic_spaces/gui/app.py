"""The editor's main window: the 3D view in the middle, the scene tree, the form of the
selected item, the results of the last run and a message log around it."""

from __future__ import annotations

import sys
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
)

from acoustic_spaces import __version__
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
from acoustic_spaces.gui.results_view import ResultsPanel
from acoustic_spaces.gui.tree import SceneTree
from acoustic_spaces.gui.viewport import Viewport
from acoustic_spaces.gui.worker import SimulationJob
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, builtin_database

TITLE = "Acoustic Spaces editor"
SCENE_FILTER = "Scenes (*.yaml *.yml)"
MODEL_FILTER = "Models (*.obj *.stl *.ply *.off *.gltf *.glb *.dxf)"
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
        self.job: SimulationJob | None = None
        self._job_scene = None
        self._started = 0.0
        #: The document is a bundled example: Save asks for a new place.
        self.from_example = False

        self.viewport = Viewport()
        self.setCentralWidget(self.viewport)
        self.tree = SceneTree()
        self.inspector = Inspector(lambda: self.document)
        self.results = ResultsPanel()
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
            ("messages", "Messages", self.messages, Qt.DockWidgetArea.BottomDockWidgetArea),
        ):
            dock = QDockWidget(title, self)
            dock.setObjectName(name)
            dock.setWidget(widget)
            self.addDockWidget(area, dock)
            self.docks[name] = dock
        self.tabifyDockWidget(self.docks["results"], self.docks["messages"])
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

        self._actions()
        self._menus()
        self.tree.selected.connect(lambda kind, key: self.select(kind, key, from_tree=True))
        self.viewport.picked.connect(self._picked)
        self.viewport.place.connect(self._place)
        self.inspector.failed.connect(self._refused)
        self.inspector.select.connect(self.select)
        self.results.pair_changed.connect(lambda *_: self._draw())
        self.results.paths_toggled.connect(self._paths_toggled)
        self.results.export_requested.connect(self.export_results)
        self.results.listen_requested.connect(self.listen)
        self.results.stop_requested.connect(self.stop_playing)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._escape)
        # Delete removes the selected item only from the tree or the 3D view, so the key
        # keeps its meaning in the forms.
        for widget in (self.tree, self.viewport):
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
        self.open_action = a("&Open…", self.open_dialog, keys.Open)
        self.save_action = a("&Save", self.save, keys.Save)
        self.save_as_action = a("Save &as…", self.save_as, keys.SaveAs)
        self.export_action = a("&Export results…", self.export_results, None)
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
        self.add_material_action = a("Add &material from the database…", self.add_material)
        self.remove_action = a("Re&move selected", self.remove_selected)
        self.reset_camera_action = a("&Reset camera", self.viewport.reset_camera, "Ctrl+0")
        self.paths_action = a("Show reflection &paths", lambda: None)
        self.paths_action.setCheckable(True)
        self.paths_action.toggled.connect(self.results.paths.setChecked)
        self.screenshot_action = a("Save &screenshot…", self.save_screenshot)
        self.run_action = a("&Run", self.run_simulation, "F5", "Simulate every source and listener")
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
            (self.cancel_action, pixmaps.SP_MediaStop),
        ):
            action.setIcon(icon(pixmap))

    def _menus(self) -> None:
        bar = self.menuBar()
        file = bar.addMenu("&File")
        file.addActions([self.new_action, self.new_model_action, self.open_action])
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
        file.addSeparator()
        file.addAction(self.quit_action)

        edit = bar.addMenu("&Edit")
        edit.addActions([self.undo_action, self.redo_action])
        edit.addSeparator()
        edit.addActions([self.place_source_action, self.place_listener_action])
        edit.addAction(self.add_material_action)
        edit.addSeparator()
        edit.addAction(self.remove_action)

        view = bar.addMenu("&View")
        view.addActions([self.reset_camera_action, self.paths_action, self.screenshot_action])
        view.addSeparator()
        for dock in self.docks.values():
            view.addAction(dock.toggleViewAction())

        simulation = bar.addMenu("&Simulation")
        simulation.addActions([self.run_action, self.cancel_action])
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
        tools.addActions([self.run_action, self.cancel_action])

    # -- the document ---------------------------------------------------------------

    def _attach(self, document: Document) -> None:
        """Make ``document`` the one being edited."""
        old = self.document
        if old is not document:
            old.changed.disconnect(self._changed)
            old.state_changed.disconnect(self._state)
        self.document = document
        document.changed.connect(self._changed)
        document.state_changed.connect(self._state)
        self.result = self._result_scene = None
        self.results.set_result(None)
        self.selection = ("scene", None)
        self._changed()
        self._state()

    def _changed(self) -> None:
        if not self._valid(self.selection):
            self.selection = ("scene", None)
        self.tree.rebuild(self.document.room, self.selection)
        self.inspector.show_item(*self.selection)
        self.results.mark_stale(self._stale())
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
        self._draw()

    def _picked(self, kind: str, key) -> None:
        """A click in the 3D view; empty space selects the scene."""
        self.select(kind or "scene", key)
        if kind == "surface":
            self.log(f"Surface {label_text(self.document.room.mesh.labels[key])}")

    def _draw(self) -> None:
        self.remove_action.setEnabled(self.selection[0] in REMOVABLE)
        paths = None
        if self.results.paths.isChecked() and not self._stale():
            response = self.results.response()
            paths = None if response is None else response.early
        self.viewport.show_room(self.document.room, self.selection, paths)

    def _paths_toggled(self, on: bool) -> None:
        self.paths_action.blockSignals(True)
        self.paths_action.setChecked(on)
        self.paths_action.blockSignals(False)
        if on and self.result is None:
            self.log("Reflection paths come from a run: run the simulation (F5) first.")
        self._draw()

    def start_placing(self, kind: str) -> None:
        """Put a new source or listener where the next click on a surface lands."""
        if self.viewport.available:
            self.viewport.start_placing(kind)
            what = "source" if kind == "sources" else "listener"
            self.statusBar().showMessage(
                f"Click a floor or wall to place the {what}; Esc cancels.", 0
            )
        else:
            self._place(kind, self._default_position(kind))

    def _escape(self) -> None:
        if self.viewport.placing is not None:
            self.viewport.start_placing(None)
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
        return bool(path) and self._save_to(Path(path))

    def _save_to(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.document.save(path, copy_files=self.from_example)
        except (OSError, ValueError) as exc:
            self._error("Could not save", str(exc))
            return False
        self.from_example = False
        self.log(f"Saved {path}")
        return True

    def save_screenshot(self) -> None:
        if not self.viewport.available:
            self.log(self.viewport.note)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "room.png", "PNG (*.png)")
        if path:
            self.viewport.screenshot(path)
            self.log(f"Saved {path}")

    def _error(self, title: str, text: str) -> None:
        self.log(f"{title}: {text}")
        QMessageBox.warning(self, title, text)

    # -- simulation -----------------------------------------------------------------

    def _stale(self) -> bool:
        return self.result is not None and self._result_scene is not self.document.scene

    def run_simulation(self) -> None:
        if self.job is not None:
            return
        problems = self.document.room.problems()
        if problems:
            self.log("Cannot run: " + "; ".join(problems))
            self.docks["messages"].raise_()
            return
        self.job = SimulationJob(self.document.room)
        self._job_scene = self.document.scene
        self.job.progress.connect(self._progress)
        self.job.done.connect(self._done)
        # Bound methods, so Qt queues the worker's signals to this window's thread.
        self.job.failed.connect(self._failed)
        self.job.cancelled.connect(self._cancelled)
        self.job.finished.connect(self._job_ended)
        self._started = time.perf_counter()
        self.run_action.setEnabled(False)
        self.cancel_action.setEnabled(True)
        self.progress.show()
        self.log(f"Simulating {self.document.scene.name}…")
        self.job.start()

    def cancel_simulation(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.log("Cancelling at the next stage…")

    def _progress(self, message: str) -> None:
        self.log(f"{time.perf_counter() - self._started:6.1f} s  {message}")

    def _done(self, result) -> None:
        self.result, self._result_scene = result, self._job_scene
        self.results.set_result(result)
        self.results.mark_stale(self._stale())
        self.docks["results"].raise_()
        elapsed = time.perf_counter() - self._started
        count = len(result.responses)
        self.log(f"Done in {elapsed:.1f} s: {count} response{'s' if count != 1 else ''}.")
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

    def stop_playing(self) -> None:
        from acoustic_spaces.measurement import _sounddevice

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
