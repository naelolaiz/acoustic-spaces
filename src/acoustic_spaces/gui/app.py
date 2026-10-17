"""The editor's main window: the 3D view and the plan in the middle, the scene tree,
the form of the selected item, the results of the last run, the coverage map and a
message log around them."""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLibraryInfo, Qt, QTimer, QTranslator
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QGuiApplication,
    QKeySequence,
    QShortcut,
    QShowEvent,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
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
    QToolButton,
)

from acoustic_spaces import __version__, i18n
from acoustic_spaces._extras import install_hint
from acoustic_spaces.engine import ray_device, reweight, tuning
from acoustic_spaces.examples import example_path, list_examples
from acoustic_spaces.geometry import SUPPORTED_SUFFIXES, points_inside
from acoustic_spaces.gui.document import (
    LISTENER_HEIGHT,
    SOURCE_HEIGHT,
    Document,
    EditError,
    box_scene,
    model_scene,
)
from acoustic_spaces.gui.inspector import BAND_LABELS, Inspector, label_text
from acoustic_spaces.gui.levels import level_colour, rich_lines
from acoustic_spaces.gui.map_view import MapPanel
from acoustic_spaces.gui.numbers import fmt, point_text
from acoustic_spaces.gui.plan_view import PlanView
from acoustic_spaces.gui.results_view import ResultsPanel
from acoustic_spaces.gui.tree import SceneTree
from acoustic_spaces.gui.tune_dialog import TuneDialog
from acoustic_spaces.gui.viewport import Viewport
from acoustic_spaces.gui.worker import MapJob, SimulationJob, TuneJob
from acoustic_spaces.i18n import N_, _, ngettext, pgettext
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, builtin_database
from acoustic_spaces.room import Room

TITLE = N_("Acoustic Spaces editor")
SCENE_FILTER = N_("Scenes (*.yaml *.yml)")
# Translators: {patterns} is the list of model file patterns, such as *.obj *.dxf.
MODEL_FILTER = N_("Models ({patterns})")
MODEL_PATTERNS = " ".join("*" + s for s in SUPPORTED_SUFFIXES)
GLB_FILTER = N_("glTF binary (*.glb)")
# Translators: two file types of a save dialog, kept apart by ;; (keep it).
MAP_FILTERS = N_("PNG picture (*.png);;JSON, every map (*.json)")
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
        #: The room the leaks on show were found in, and the leaks; None before any.
        self._found_leaks = None
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
        self.views.addTab(self.viewport, _("3D view"))
        self.views.addTab(self.plan, _("Plan"))
        self.views.setTabToolTip(
            0, _("The room in 3D: click to select, drag sources and listeners")
        )
        self.views.setTabToolTip(
            1, _("The room from above: draw floor plans, drag corners and points")
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
            ("tree", _("Scene"), self.tree, Qt.DockWidgetArea.LeftDockWidgetArea),
            ("inspector", _("Properties"), forms, Qt.DockWidgetArea.RightDockWidgetArea),
            ("results", _("Results"), self.results, Qt.DockWidgetArea.BottomDockWidgetArea),
            ("map", _("Coverage map"), self.map_panel, Qt.DockWidgetArea.BottomDockWidgetArea),
            ("messages", _("Messages"), self.messages, Qt.DockWidgetArea.BottomDockWidgetArea),
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
        self.progress.setToolTip(
            _("Something runs in the background; Simulation > Cancel (Shift+F5) stops it")
        )
        self.progress.hide()
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
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
        #: What plays: the file, source and listener, whether head-tracked, where in the
        #: file it started and when, the file's length (s), the gain, the scene it was
        #: simulated for, the status bar's text, and whether the log said the scene has
        #: changed since. None while nothing plays.
        self._played: dict | None = None

        self._actions()
        self._menus()
        # While a recording plays, the status bar says so, whichever panel is on show.
        self.now_playing = QLabel()
        self.now_playing.setToolTip(
            _(
                "What plays through the room: the recording, and the source and listener of the "
                "response it plays through"
            )
        )
        self.stop_button = QToolButton()
        self.stop_button.setDefaultAction(self.stop_action)
        self.stop_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for index, widget in enumerate((self.now_playing, self.stop_button)):
            widget.hide()
            self.statusBar().insertPermanentWidget(index, widget)
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
        self.results.pair_changed.connect(lambda *_: self._follow_playback())
        self.results.tracking.toggled.connect(lambda *_: self._follow_playback())
        self.results.playing_changed.connect(self._playing)
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
            self.log(self.viewport.note, "warning")

    # -- menus ----------------------------------------------------------------------

    def _action(self, text: str, slot, shortcut=None, tip: str = "") -> QAction:
        """An action whose tip shows in the status bar, and on the toolbar and in the
        menus with its shortcut."""
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
            keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            action.setToolTip(f"{tip} ({keys})" if keys else tip)
        action.triggered.connect(lambda *_: slot())
        return action

    def _actions(self) -> None:
        keys = QKeySequence.StandardKey
        a = self._action
        self.new_action = a(
            _("&New box room…"), self.new_box, keys.New, _("Start a new scene in a box-shaped room")
        )
        self.new_model_action = a(
            _("New from &model file…"),
            self.new_from_model,
            None,
            _("Start from a model file (OBJ, DXF, glTF, STL, IFC, Collada or SketchUp)"),
        )
        self.new_plan_action = a(
            _("New room from a &floor plan…"),
            self.new_floor_plan,
            None,
            _("Draw the floor plan of a new room, corner by corner"),
        )
        self.open_action = a(
            _("&Open…"), self.open_dialog, keys.Open, _("Open a scene file (YAML)")
        )
        self.save_action = a(
            _("&Save"),
            self.save,
            keys.Save,
            _("Save the scene; an example asks where to keep your copy"),
        )
        self.save_as_action = a(
            _("Save &as…"), self.save_as, keys.SaveAs, _("Save the scene in another file")
        )
        self.export_action = a(
            _("&Export results…"),
            self.export_results,
            None,
            _(
                "Write every response of the last run to a folder: a WAV file per output format "
                "and parameters.json, as acoustic-spaces simulate does"
            ),
        )
        self.blender_action = a(
            _("Export to &Blender…"),
            self.export_blender,
            tip=_("The room, and the reflection paths on show, as glTF"),
        )
        self.particles_action = a(
            _("Export &particle animation to Blender…"),
            lambda: self.export_blender(particles=True),
            tip=_("Sound particles from the selected source (or the first) as a glTF animation"),
        )
        self.quit_action = a(
            _("&Quit"), self.close, keys.Quit, _("Close the editor; it asks about unsaved changes")
        )
        self.undo_action = a(_("&Undo"), self.undo, keys.Undo, _("Take back the last change"))
        self.redo_action = a(_("&Redo"), self.redo, keys.Redo, _("Make the change Undo took back"))
        self.place_source_action = a(
            _("Place &source"),
            lambda: self.start_placing("sources"),
            tip=_("Click a floor or wall in the 3D view to put a source there"),
        )
        self.place_listener_action = a(
            _("Place &listener"),
            lambda: self.start_placing("listeners"),
            tip=_("Click a floor or wall in the 3D view to put a listener there"),
        )
        self.draw_action = a(
            _("&Draw floor plan"),
            self.draw_room,
            None,
            _("Draw a new floor plan in the plan view; it replaces the room"),
        )
        self.add_material_action = a(
            _("Add &material from the database…"),
            self.add_material,
            tip=_("A material of the built-in database of absorption coefficients"),
        )
        self.add_library_material_action = a(
            _("Add material from a &library…"),
            self.add_library_material,
            tip=_("A material of a CATT-Acoustic (ABS lines) or ODEON (.Li8) material library"),
        )
        self.remove_action = a(
            _("Re&move selected"),
            self.remove_selected,
            tip=_(
                "Remove the selected source, listener or material (Delete, in the scene tree "
                "or a view)"
            ),
        )
        self.reset_camera_action = a(
            _("&Reset camera"),
            self.viewport.reset_camera,
            "Ctrl+0",
            _("Show the whole room in the 3D view from the usual angle"),
        )
        self.show_3d_action = a(
            _("&3D view"),
            lambda: self.views.setCurrentWidget(self.viewport),
            "Ctrl+1",
            _("Show the room in 3D"),
        )
        self.show_plan_action = a(
            _("P&lan"),
            lambda: self.views.setCurrentWidget(self.plan),
            "Ctrl+2",
            _("Show the room from above, to draw and change floor plans"),
        )
        self.paths_action = a(
            _("Show reflection &paths"),
            lambda: None,
            tip=_(
                "Draw the early reflection paths of the source and listener the Results "
                "panel shows, coloured by order"
            ),
        )
        self.paths_action.setCheckable(True)
        self.paths_action.toggled.connect(self.results.paths.setChecked)
        self.leaks_action = a(
            _("Show &leaks"),
            lambda: None,
            tip=_(
                "Trace rays from the sources and show where they escape from a model that is "
                "not closed"
            ),
        )
        self.leaks_action.setCheckable(True)
        self.leaks_action.toggled.connect(lambda *_: self._draw())
        self.screenshot_action = a(
            _("Save &screenshot…"), self.save_screenshot, tip=_("Save the 3D view as a PNG picture")
        )
        self.run_action = a(
            _("&Run"), self.run_simulation, "F5", _("Simulate every source and listener")
        )
        self.map_action = a(
            _("Run coverage &map"),
            self.run_map,
            "F6",
            _("Room parameters on a grid over the audience area, for every source"),
        )
        self.map_action.setIconText(_("Map"))
        self.tune_action = a(
            _("&Tune CPU and GPU…"),
            self.run_tune,
            tip=_(
                "Time the CPU and the GPU on this scene and keep the fastest on this computer "
                "for the device setting auto (acoustic-spaces tune; needs the gpu extra)"
            ),
        )
        self.cancel_action = a(
            _("&Cancel"),
            self.cancel_simulation,
            "Shift+F5",
            _("Stop the run, coverage map or tuning going on, at its next stage"),
        )
        self.cancel_action.setEnabled(False)
        self.stop_action = a(
            _("S&top playback"),
            self.stop_playing,
            tip=_("Stop the recording playing through the room (Esc)"),
        )
        self.stop_action.setToolTip(_("Stop playback (Esc)"))
        self.stop_action.setEnabled(False)
        self.about_action = a(
            _("&About"), self.about, tip=_("The version, licence and source of Acoustic Spaces")
        )
        self.undo_action.setIconText(_("Undo"))
        self.redo_action.setIconText(_("Redo"))
        icon = self.style().standardIcon
        pixmaps = QStyle.StandardPixmap
        for action, pixmap in (
            (self.open_action, pixmaps.SP_DialogOpenButton),
            (self.save_action, pixmaps.SP_DialogSaveButton),
            (self.undo_action, pixmaps.SP_ArrowBack),
            (self.redo_action, pixmaps.SP_ArrowForward),
            (self.run_action, pixmaps.SP_MediaPlay),
            (self.map_action, pixmaps.SP_FileDialogDetailedView),
            # Cancel's icon differs from Stop playback's, so neither passes for the other.
            (self.cancel_action, pixmaps.SP_BrowserStop),
            (self.stop_action, pixmaps.SP_MediaStop),
        ):
            action.setIcon(icon(pixmap))

    def _menus(self) -> None:
        bar = self.menuBar()
        file = bar.addMenu(_("&File"))
        file.addActions(
            [self.new_action, self.new_plan_action, self.new_model_action, self.open_action]
        )
        self.examples_menu = file.addMenu(_("Open e&xample"))
        try:
            examples = list_examples()
        except FileNotFoundError:
            examples = []
        for example in examples:
            # The examples' titles and descriptions are in the catalogs (tools/catalog.py).
            action = self.examples_menu.addAction(pgettext("example", example.title))
            tip = pgettext("example", example.description).split(". ")[0]
            if example.missing_extra:
                # Opening it would only say how to install what reading its model needs.
                action.setEnabled(False)
                # Translators: {extra} is the name of an optional part of the
                # installation (gui, ifc...); {hint} says how to install it.
                tip = _("Needs the {extra} extra: {hint}").format(
                    extra=example.missing_extra, hint=install_hint(example.missing_extra)
                )
            action.setStatusTip(tip)
            action.setToolTip(tip)
            action.triggered.connect(lambda *_, path=example.path: self.open_example(path))
        self.examples_menu.setEnabled(bool(examples))
        file.addSeparator()
        file.addActions([self.save_action, self.save_as_action, self.export_action])
        file.addActions([self.blender_action, self.particles_action])
        file.addSeparator()
        file.addAction(self.quit_action)

        edit = bar.addMenu(_("&Edit"))
        edit.addActions([self.undo_action, self.redo_action])
        edit.addSeparator()
        edit.addActions([self.place_source_action, self.place_listener_action])
        edit.addAction(self.draw_action)
        edit.addActions([self.add_material_action, self.add_library_material_action])
        edit.addSeparator()
        edit.addAction(self.remove_action)

        view = bar.addMenu(_("&View"))
        view.addActions([self.show_3d_action, self.show_plan_action])
        view.addSeparator()
        view.addActions([self.reset_camera_action, self.paths_action, self.leaks_action])
        view.addAction(self.screenshot_action)
        view.addSeparator()
        for dock in self.docks.values():
            toggle = dock.toggleViewAction()
            toggle.setStatusTip(
                # Translators: {panel} is a panel's title, such as Results.
                _("Show or hide the {panel} panel").format(panel=dock.windowTitle())
            )
            toggle.setToolTip(toggle.statusTip())
            view.addAction(toggle)
        view.addSeparator()
        self.language_menu = view.addMenu(_("&Language"))
        self._language_choices()

        simulation = bar.addMenu(_("&Simulation"))
        simulation.addActions([self.run_action, self.map_action, self.cancel_action])
        simulation.addSeparator()
        simulation.addAction(self.tune_action)
        simulation.addSeparator()
        simulation.addAction(self.stop_action)
        help_menu = bar.addMenu(_("&Help"))
        help_menu.addAction(self.about_action)
        menus = (file, self.examples_menu, edit, view, self.language_menu, simulation, help_menu)
        for menu in menus:
            menu.setToolTipsVisible(True)

        tools = self.addToolBar(_("Main"))
        tools.setObjectName("main-toolbar")
        tools.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tools.addActions([self.open_action, self.save_action])
        tools.addSeparator()
        tools.addActions([self.undo_action, self.redo_action])
        tools.addSeparator()
        tools.addActions([self.place_source_action, self.place_listener_action])
        tools.addSeparator()
        tools.addActions([self.run_action, self.map_action, self.cancel_action])
        tools.addSeparator()
        tools.addAction(self.stop_action)

    def _language_choices(self) -> None:
        """View > Language: the computer's language, or one of the catalogs'."""
        group = QActionGroup(self)
        system = i18n.LANGUAGES[i18n.system_language()]
        choices = [(None, _("System ({language})").format(language=system))]
        choices += list(i18n.LANGUAGES.items())
        saved = i18n.saved_language()
        self.language_actions = {}
        for code, text in choices:
            action = self.language_menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(code == saved)
            if code is None:
                tip = _(
                    "Show the editor and the command line in this computer's language, from "
                    "their next start"
                )
            else:
                tip = _(
                    "Show the editor and the command line in this language, from their next start"
                )
            action.setStatusTip(tip)
            action.setToolTip(tip)
            action.triggered.connect(lambda *args, code=code: self.choose_language(code))
            group.addAction(action)
            self.language_actions[code] = action

    def choose_language(self, code: str | None) -> None:
        """Keep ``code`` (None: the computer's) as the language from the next start."""
        try:
            i18n.save_language(code)
        except OSError as exc:
            self.log(_("Cannot keep the language: {reason}").format(reason=exc), "error")
            self.language_actions[i18n.saved_language()].setChecked(True)
            return
        self.language_actions[code].setChecked(True)
        name = i18n.LANGUAGES[code or i18n.system_language()]
        variable = os.environ.get(i18n.ENV, "").strip()
        if variable:
            self.log(
                _(
                    "{variable} is set to {value}, so it decides the language, not this choice"
                ).format(variable=i18n.ENV, value=variable),
                "warning",
            )
        else:
            self.log(
                _(
                    "Language from the next start of the editor and the command line: {language}"
                ).format(language=name)
            )

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
        self._show_playing()

    def _state(self) -> None:
        document = self.document
        name = document.path.name if document.path is not None else _("untitled")
        # Translators: the window's title; [*] shows whether the scene has unsaved
        # changes (keep it as it is).
        self.setWindowTitle(_("{name}[*] - {title}").format(name=name, title=_(TITLE)))
        self.setWindowModified(document.modified)
        undo, redo = document.undo_text(), document.redo_text()
        # Translators: {edit} says what the last edit did, such as "move the source".
        self.undo_action.setText(_("&Undo {edit}").format(edit=undo) if undo else _("&Undo"))
        self.redo_action.setText(_("&Redo {edit}").format(edit=redo) if redo else _("&Redo"))
        self.undo_action.setToolTip(
            _("Undo {edit}").format(edit=undo) if undo else _("Nothing to undo")
        )
        self.redo_action.setToolTip(
            _("Redo {edit}").format(edit=redo) if redo else _("Nothing to redo")
        )
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
        if room.volume_m3:
            k = OCTAVE_BANDS_HZ.index(1000)
            rt = room.reverberation_time("eyring", air_absorption=True)[k]
            text = fmt(
                # Translators: the status bar's summary of the room: its volume and its
                # reverberation time at one band, such as "1 kHz".
                _("{volume:,.0f} m³, RT {rt:.2f} s at {band} (Eyring)"),
                volume=room.volume_m3,
                rt=rt,
                band=BAND_LABELS[k],
            )
        else:
            text = _("model not closed")
        lines = [(text, "info")]
        problems = room.problems()
        if problems:
            count = len(problems)
            lines.append(
                (
                    ngettext(", {count} problem", ", {count} problems", count).format(count=count),
                    "error",
                )
            )
        self.summary.setText(rich_lines(self.summary, lines).replace("<br>", ""))
        self.summary.setToolTip(
            "\n".join(
                [
                    _(
                        "The room's volume and its reverberation time at 1 kHz estimated by "
                        "the Eyring formula, kept up to date as you edit"
                    ),
                    *problems,
                ]
            )
        )

    def log(self, text: str, level: str = "info") -> None:
        """Add ``text`` to the messages, a warning in amber and an error in red, and show
        it in the status bar."""
        style = QTextCharFormat()
        colour = level_colour(self.messages, level)
        if colour is not None:
            style.setForeground(colour)
        bar = self.messages.verticalScrollBar()
        following = bar.value() == bar.maximum()
        cursor = QTextCursor(self.messages.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.messages.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(text, style)
        if following:  # as appending does: stay at the end unless scrolled back
            bar.setValue(bar.maximum())
        self.statusBar().showMessage(text, 8000)

    def _refused(self, message: str) -> None:
        self.log(_("Not changed: {reason}").format(reason=message), "error")

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
            self._follow_playback()

    def _picked(self, kind: str, key) -> None:
        """A click in the 3D view; empty space selects the scene."""
        self.select(kind or "scene", key)
        if kind == "surface":
            self.log(
                _("Surface {name}").format(name=label_text(self.document.room.mesh.labels[key]))
            )

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
        leaks = self._leaks() if self.leaks_action.isChecked() else None
        self.viewport.show_room(self.document.room, self.selection, paths, coverage, leaks)
        self.plan.show_room(self.document.room, self.selection)

    def _leaks(self):
        """Where rays escape from the room, found again when its geometry, materials or
        sources change."""
        room = self.document.room
        if self._found_leaks is not None:
            before, leaks = self._found_leaks
            if before.mesh is room.mesh and all(
                getattr(before.scene, part) == getattr(room.scene, part)
                for part in ("sources", "materials", "assignments", "default_material")
            ):
                return leaks
        from acoustic_spaces.engine.leaks import find_leaks

        leaks = find_leaks(room)
        self._found_leaks = (room, leaks)
        self.log(
            _("Leaks: {summary}").format(summary=leaks.summary()),
            "warning" if leaks.leaked else "info",
        )
        return leaks

    def _paths_toggled(self, on: bool) -> None:
        self.paths_action.blockSignals(True)
        self.paths_action.setChecked(on)
        self.paths_action.blockSignals(False)
        if on and self.result is None:
            self.log(_("Reflection paths come from a run: run the simulation (F5) first."))
        self._draw()

    def start_placing(self, kind: str) -> None:
        """Put a new source or listener where the next click on a surface (in the plan,
        on the floor) lands."""
        source = kind == "sources"
        if self.views.currentWidget() is self.plan:
            self.plan.start_placing(kind)
            self.statusBar().showMessage(
                _("Click in the plan to place the source; Esc cancels.")
                if source
                else _("Click in the plan to place the listener; Esc cancels."),
                0,
            )
        elif self.viewport.available:
            self.viewport.start_placing(kind)
            self.statusBar().showMessage(
                _("Click a floor or wall to place the source; Esc cancels.")
                if source
                else _("Click a floor or wall to place the listener; Esc cancels."),
                0,
            )
        else:
            self._place(kind, self._default_position(kind))

    def _escape(self) -> None:
        """Esc stops placing or drawing, or else the playback."""
        placing = self.viewport.placing is not None
        if placing:
            self.viewport.start_placing(None)
        if not self.plan.cancel() and not placing and self.results.stop.isEnabled():
            self.stop_playing()
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
        self.log(_("Added {name} at ({point})").format(name=item.name, point=point_text(position)))
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
        self.log(
            _("Moved {name} to ({point})").format(name=item.name, point=point_text(item.position))
        )
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
            self.log(_("The box is now a floor plan with the same walls."))

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
            fmt(
                ngettext(
                    "New floor plan with {count} corner, {height:g} m high; change the "
                    "height under Geometry.",
                    "New floor plan with {count} corners, {height:g} m high; change the "
                    "height under Geometry.",
                    len(corners),
                ),
                count=len(corners),
                height=height,
            )
        )
        if moved:
            # Translators: {names} lists the sources and listeners moved.
            self.log(_("Moved {names} into the new room.").format(names=", ".join(moved)))
        if self.document.scene.default_material != default:
            self.log(
                _(
                    "Surfaces without an assignment are now {material} (the scene's default "
                    "material)."
                ).format(material=self.document.scene.default_material)
            )
        self.select("geometry", None)

    def confirm_replace_model(self) -> bool:
        answer = QMessageBox.question(
            self,
            _(TITLE),
            _(
                "Replace the model file with the drawn floor plan? Material assignments to the "
                "model's surfaces will no longer match anything."
            ),
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
            self, _("Add material"), _("Material from the database:"), names, 0, True
        )
        if not ok or not preset:
            return
        try:
            name = self.document.add_material(preset)
        except EditError as exc:
            self._refused(str(exc))
            return
        self.select("material", name)

    def add_library_material(self, path: str | None = None) -> None:
        """Pick a CATT-Acoustic or ODEON library file, then one of its materials."""
        from acoustic_spaces.materials import read_library

        if path is None:
            path, _filter = QFileDialog.getOpenFileName(
                self,
                _("Material library"),
                str(self.document.base_dir),
                _("Material libraries (*.geo *.txt *.li8 *.Li8 *.lib);;All files (*)"),
            )
            if not path:
                return
        try:
            materials = read_library(path)
        except (OSError, ValueError) as exc:
            self._refused(str(exc))
            return
        labels = [m.label for m in materials]
        label, ok = QInputDialog.getItem(self, _("Add material"), _("Material:"), labels, 0, False)
        if not ok:
            return
        found = materials[labels.index(label)]
        try:
            name = self.document.add_library_material(
                path, found.name if found.number is None else str(found.number)
            )
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
            self, _(TITLE), _("The scene has unsaved changes. Save them?"), buttons
        )

    def new_box(self) -> None:
        if not self._discard_ok():
            return
        self.from_example = False
        self._attach(Document(box_scene()))
        self.log(_("New box room; change its size under Geometry."))

    def new_floor_plan(self) -> None:
        """A new room whose floor plan the user draws in the plan view."""
        if not self._discard_ok():
            return
        self.from_example = False
        self._attach(Document(box_scene()))
        self.draw_room()
        self.log(_("New room: click its corners in the plan, and close it on the first one."))

    def new_from_model(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, _("Model file"), "", _(MODEL_FILTER).format(patterns=MODEL_PATTERNS)
        )
        if not path or not self._discard_ok():
            return
        try:
            with busy():
                document = Document(model_scene(path))
        except EditError as exc:
            self._error(_("Could not load {name}").format(name=Path(path).name), str(exc))
            return
        self.from_example = False
        self._attach(document)
        self.log(
            _(
                "Loaded {name}; set its units and up axis under Geometry if it looks wrong, "
                "then assign materials to its surfaces."
            ).format(name=Path(path).name)
        )

    def open_dialog(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, _("Open scene"), "", _(SCENE_FILTER))
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path, *, example: bool = False) -> bool:
        if not self._discard_ok():
            return False
        try:
            with busy():
                document = Document.open(path)
        except EditError as exc:
            self._error(_("Could not open {name}").format(name=Path(path).name), str(exc))
            return False
        self.from_example = example
        self._attach(document)
        self.log(_("Opened {path}").format(path=path))
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
        path, _filter = QFileDialog.getSaveFileName(
            self, _("Save scene"), str(start), _(SCENE_FILTER)
        )
        if path and not Path(path).suffix:
            path += ".yaml"
            # The dialog asked before replacing the name as typed, not this one.
            if Path(path).exists() and not self._confirm_overwrite(Path(path)):
                return False
        return bool(path) and self._save_to(Path(path))

    def _confirm_overwrite(self, path: Path) -> bool:
        answer = QMessageBox.question(
            self,
            _(TITLE),
            _("{name} already exists. Replace it?").format(name=path.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _save_to(self, path: Path) -> bool:
        scene = self.document.scene
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.document.save(path, copy_files=self.from_example)
        except (OSError, ValueError) as exc:
            self._error(_("Could not save"), str(exc))
            return False
        if self.document.scene is not scene:
            self._files_moved(scene, self.document.scene)
        self.from_example = False
        self.log(_("Saved {path}").format(path=path))
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
            self.log(self.viewport.note, "warning")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, _("Save screenshot"), "room.png", _("PNG (*.png)")
        )
        if path:
            self.viewport.screenshot(path)
            self.log(_("Saved {path}").format(path=path))

    def export_blender(self, particles: bool = False) -> bool:
        """Write the room as glTF for Blender, with the reflection paths on show, or with
        sound particles from the selected source (or the first) as an animation."""
        from acoustic_spaces.engine import trace_particles
        from acoustic_spaces.gltf import PARTICLE_COUNT, SLOW_MOTION, export_gltf

        title = _("Export particle animation to Blender") if particles else _("Export to Blender")
        start = "particles.glb" if particles else "room.glb"
        path, _filter = QFileDialog.getSaveFileName(self, title, start, _(GLB_FILTER))
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
            self._error(_("Could not export"), str(exc))
            return False
        if traced is not None:
            sound = float(traced.times[-1])
            text = fmt(
                _(
                    "Wrote {path}: {count} particles from {source}, {sound:g} ms of sound over "
                    "{slow:g} s. In Blender: File > Import > glTF 2.0."
                ),
                path=path,
                count=PARTICLE_COUNT,
                source=traced.source,
                sound=1e3 * sound,
                slow=SLOW_MOTION * sound,
            )
        elif response is not None:
            count = len(response.early)
            text = ngettext(
                "Wrote {path}: the room with {count} reflection path from {source} to "
                "{listener}. In Blender: File > Import > glTF 2.0.",
                "Wrote {path}: the room with {count} reflection paths from {source} to "
                "{listener}. In Blender: File > Import > glTF 2.0.",
                count,
            ).format(path=path, count=count, source=response.source, listener=response.listener)
        else:
            text = _("Wrote {path}: the room. In Blender: File > Import > glTF 2.0.").format(
                path=path
            )
        self.log(text)
        return True

    def _error(self, title: str, text: str) -> None:
        # Translators: an error's title, then what went wrong.
        self.log(_("{title}: {reason}").format(title=title, reason=text), "error")
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
            self.log(_("Cannot run: {problems}").format(problems="; ".join(problems)), "error")
            self.docks["messages"].raise_()
            return
        # Count reflections off each material, so that changing absorption afterwards
        # re-weights the run instead of making it out of date.
        self._job_groups = reweight.groups_of(self.document.room)
        groups = self._job_groups[0] if self._job_groups is not None else None
        job = SimulationJob(self.document.room, groups=groups)
        job.done.connect(self._done)
        self._start(job, _("Simulating {scene}…").format(scene=self.document.scene.name))

    def run_map(self) -> None:
        """Compute the coverage map of every source in the background."""
        if self.job is not None:
            return
        # Listeners play no part in a map.
        problems = [p for p in self.document.room.problems() if not p.startswith("listener ")]
        if problems:
            self.log(_("Cannot map: {problems}").format(problems="; ".join(problems)), "error")
            self.docks["messages"].raise_()
            return
        job = MapJob(self.document.room)
        job.done.connect(self._map_done)
        self._start(job, _("Mapping {scene}…").format(scene=self.document.scene.name))

    def run_tune(self) -> None:
        """Time the CPU and the GPU in the background (``acoustic-spaces tune``) on this
        scene, or on the concert hall example when this one cannot run as it is."""
        if self.job is not None:
            return
        room = self.document.room
        if room.problems():
            room = Room.from_yaml(example_path(tuning.SCENE))
        job = TuneJob(room)
        job.done.connect(self._tune_done)
        self._start(job, _("Timing the CPU and the GPU on {scene}…").format(scene=room.scene.name))

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
        self.tune_action.setEnabled(False)
        self.map_panel.set_running(True)
        self.cancel_action.setEnabled(True)
        self.progress.show()
        self.log(message)
        job.start()

    def cancel_simulation(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.log(_("Cancelling at the next stage…"))

    def _progress(self, message: str) -> None:
        # Translators: the time since a run started, then what it does now.
        self.log(
            fmt(_("{time:6.1f} s  {step}"), time=time.perf_counter() - self._started, step=message)
        )

    def _closed_meanwhile(self) -> bool:
        """True (and a message) when the scene of the job that just ended is no longer
        open: another was opened or started while the job ran."""
        if self._job_document is self.document:
            return False
        self.log(_("Dropped the results of the run: its scene is no longer open."), "warning")
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
        self.log(
            fmt(
                ngettext(
                    "Done in {time:.1f} s: {count} response.",
                    "Done in {time:.1f} s: {count} responses.",
                    count,
                ),
                time=elapsed,
                count=count,
            )
        )
        self._draw()

    def _map_done(self, result) -> None:
        if self._closed_meanwhile():
            return
        self.coverage, self._coverage_scene = result, self._job_scene
        self.map_panel.set_result(result, result.room)
        self.map_panel.mark_stale(self._map_stale())
        self.docks["map"].raise_()
        elapsed = time.perf_counter() - self._started
        count = len(result.grid)
        self.log(
            fmt(
                ngettext(
                    "Map done in {time:.1f} s: {count:,} point.",
                    "Map done in {time:.1f} s: {count:,} points.",
                    count,
                ),
                time=elapsed,
                count=count,
            )
        )
        self._draw()

    def _tune_done(self, result) -> None:
        elapsed = time.perf_counter() - self._started
        late = ", ".join(
            fmt(_("{length:g} s {device}"), length=t.length, device=t.device) for t in result.late
        )
        if result.wave:
            text = _("Tuned in {time:.1f} s: late stage {late}; wave solver {wave}.")
        else:
            text = _("Tuned in {time:.1f} s: late stage {late}.")
        wave = tuning.wave_pick(result.wave.costs) if result.wave else ""
        self.log(fmt(text, time=elapsed, late=late, wave=wave))
        settings = self.document.scene.simulation
        dialog = TuneDialog(result, settings.late.device, settings.wave.device, self)
        dialog.finished.connect(lambda code: self._tune_answered(dialog, code))
        dialog.open()

    def _tune_answered(self, dialog: TuneDialog, code: int) -> None:
        """Save what the tune window asked for on OK; nothing on Cancel."""
        dialog.deleteLater()
        if code != QDialog.DialogCode.Accepted.value:
            self.log(_("Tuning not saved: device auto and the default are as they were."))
            return
        dialog.timings.keep()
        tuning.set_default(dialog.default())
        self.log(
            _("Timings saved in {path}. This computer's default device: {device}.").format(
                path=dialog.timings.file, device=dialog.default()
            )
        )
        if dialog.follows():
            self.follow_default_device()
        self.inspector.refresh()

    def follow_default_device(self) -> None:
        """Let the scene's late reverberation and wave solver run on this computer's
        default device (Run on: default) instead of devices of their own."""

        def change(data: dict) -> None:
            simulation = data.get("simulation", {})
            simulation.get("late", {}).pop("device", None)
            simulation.get("wave", {}).pop("device", None)

        try:
            self.document.edit(_("run on the default device"), change)
        except EditError as exc:
            self._refused(str(exc))
            return
        self.log(_("Late reverberation and the wave solver now run on this computer's default."))

    def _job_kind(self) -> str:
        return self.job.kind if self.job is not None else "simulation"

    def _failed(self, message: str) -> None:
        text = {
            "coverage map": _("The coverage map failed: {reason}"),
            "tuning": _("The tuning failed: {reason}"),
        }.get(self._job_kind(), _("The simulation failed: {reason}"))
        self.log(text.format(reason=message), "error")
        self.docks["messages"].raise_()

    def _cancelled(self) -> None:
        self.log(
            {
                "coverage map": _("Coverage map cancelled."),
                "tuning": _("Tuning cancelled."),
            }.get(self._job_kind(), _("Simulation cancelled."))
        )

    def _job_ended(self) -> None:
        job, self.job = self.job, None
        if job is not None:
            job.wait()
        # Done (or failed) while another window was in front: flash the editor's entry in
        # the taskbar (the Dock icon bounces on macOS), as desktop programs do. Nothing
        # pops up: the results and the messages already show what happened.
        QApplication.alert(self)
        self.run_action.setEnabled(True)
        self.map_action.setEnabled(True)
        self.tune_action.setEnabled(True)
        self.map_panel.set_running(False)
        self.cancel_action.setEnabled(False)
        self.progress.hide()

    # -- results --------------------------------------------------------------------

    def export_results(self) -> None:
        if self.result is None:
            self.log(_("Nothing to export yet: run the simulation (F5) first."))
            return
        folder = QFileDialog.getExistingDirectory(self, _("Folder for the impulse responses"))
        if folder:
            self.write_results(folder)

    def write_results(self, folder: str | Path) -> dict | None:
        from acoustic_spaces.results import REPORT_FILE, write_results

        try:
            with busy():
                report = write_results(self.result, folder, progress=self.log)
        except (OSError, ValueError, ImportError) as exc:
            self._error(_("Could not export"), str(exc))
            return None
        files = sum(len(r["outputs"]) for r in report["responses"])
        self.log(
            ngettext(
                "Wrote {count} impulse response file and {report} to {folder}",
                "Wrote {count} impulse response files and {report} to {folder}",
                files,
            ).format(count=files, report=REPORT_FILE, folder=folder)
        )
        if report["peak"] > 1.0:
            self.log(
                _(
                    "Note: the float WAV files exceed full scale (a listener is closer than 1 m "
                    "to a source); scale them before playing them back."
                )
            )
        if self._stale():
            self.log(_("These results are from before the latest changes to the scene."), "warning")
        return report

    def save_map(self) -> None:
        if self.coverage is None:
            self.log(_("No map yet: run the coverage map (F6) first."))
            return
        path, chosen = QFileDialog.getSaveFileName(
            self, _("Save the coverage map"), str(self.document.base_dir), _(MAP_FILTERS)
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
            self._error(_("Could not save the map"), str(exc))
            return False
        self.log(_("Wrote {path}").format(path=path))
        if self._map_stale():
            self.log(_("This map is from before the latest changes to the scene."), "warning")
        return True

    def _listening_format(self):
        """Binaural, with the HRTF of the scene's binaural output if it has one, else the
        spherical head: as head-tracked listening hears it, whatever the scene writes."""
        from acoustic_spaces.spatial import Binaural, from_output

        for output in self.result.room.scene.outputs:
            if output.format == "binaural":
                return from_output(output)
        return Binaural()

    @staticmethod
    def _level(energy: float, gain: float | None) -> float:
        """The gain to play at, the same with head tracking or without: the dry signal at
        half its level through a response of omnidirectional ``energy``, so the
        reverberation seldom clips. ``gain``, of a playback that moves to another source,
        listener or way of playing, stays as it is up to 6 dB over that, so the
        difference between listeners stays audible."""
        safe = 0.5 / np.sqrt(energy) if energy > 0 else 1.0
        return safe if gain is None else min(gain, 2.0 * safe)

    def listen(self) -> None:
        if self.results.pair() is None:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self, _("Dry recording to play in the room"), "", _("WAV files (*.wav)")
        )
        if path:
            self.play_file(path)

    def play_file(self, path: str | Path, *, start: float = 0.0, gain: float | None = None) -> bool:
        """Play a WAV file from the shown source to the shown listener, from ``start``
        seconds into it, at the level of :meth:`_level`, lowered where it would clip."""
        from acoustic_spaces.audio import read_wav
        from acoustic_spaces.measurement import _sounddevice
        from acoustic_spaces.render import auralize

        pair = self.results.pair()
        if pair is None:
            return False
        if self.results.stop.isEnabled():  # one playback at a time
            self.stop_playing()
        if self.results.tracking.isChecked():
            return self.play_tracked(path, start=start, gain=gain)
        try:
            sd = _sounddevice()
            signal, rate = read_wav(path)
            listening = self._listening_format()
            sample_rate = int(self.result.settings.sample_rate)
            with busy():
                rest = signal[int(round(start * rate)) :]
                mix = auralize(self.result, {pair[0]: (rest, rate)}, pair[1], listening)
                omni = self.result[pair].impulse_response(sample_rate)
        except (ImportError, OSError, ValueError) as exc:
            self._error(_("Cannot play"), str(exc))
            return False
        gain = self._level(float((omni**2).sum()), gain)
        peak = float(np.abs(mix).max(initial=0.0)) * gain
        if peak > 10.0 ** (-1.0 / 20.0):  # no louder than -1 dB full scale
            gain *= 10.0 ** (-1.0 / 20.0) / peak
        mix = mix * gain
        try:
            sd.play(mix.astype(np.float32), sample_rate)
        except Exception as exc:  # PortAudio errors are not one type
            self._error(_("Cannot play"), str(exc))
            return False
        names = {"name": Path(path).name, "source": pair[0], "listener": pair[1]}
        text = _("Playing {name} from {source} at {listener}").format(**names)
        self._played = self._playback(path, pair, False, start, len(signal) / rate, gain, text)
        self.results.set_playing(True)
        self._play_timer.start(int(1000 * len(mix) / sample_rate) + 250)
        # Translators: {way} says how it is heard, such as "binaural (spherical head)".
        self.log(
            _("Playing {name} from {source} at {listener} ({way})").format(
                **names, way=listening.label
            )
        )
        self._show_playing()
        return True

    def _tracking_order(self) -> int:
        """The highest Ambisonics order among the scene's outputs, or 3."""
        orders = [o.order for o in self.result.room.scene.outputs if o.format == "ambisonics"]
        return max([*orders, 1]) if orders else 3

    def play_tracked(
        self, path: str | Path, *, start: float = 0.0, gain: float | None = None
    ) -> bool:
        """Play a WAV file from the shown source to the shown listener binaurally, turned
        by the head of the head controls, in real time, from ``start`` seconds into it,
        at the level of :meth:`_level`."""
        from acoustic_spaces.audio import read_wav
        from acoustic_spaces.headtracking import HeadTrackedBinaural
        from acoustic_spaces.measurement import _sounddevice
        from acoustic_spaces.render import resample
        from acoustic_spaces.spatial import Ambisonics

        pair = self.results.pair()
        if pair is None:
            return False
        if self.results.stop.isEnabled():  # one playback at a time
            self.stop_playing()
        sample_rate = int(self.result.settings.sample_rate)
        order = self._tracking_order()
        listening = self._listening_format()
        try:
            _sounddevice()
            signal, rate = read_wav(path)
            with busy():
                dry = resample(signal.mean(axis=1), rate, sample_rate)
                dry = dry[int(round(start * sample_rate)) :]
                response = self.result[pair]
                ir = response.spatial_impulse_response(Ambisonics(order), sample_rate)
                renderer = HeadTrackedBinaural(
                    ir,
                    sample_rate,
                    hrtf=listening.at(sample_rate),
                    orientation=self.results.head.head,
                    block=512,
                )
        except (ImportError, OSError, ValueError) as exc:
            self._error(_("Cannot play"), str(exc))
            return False
        gain = self._level(float((ir[:, 0] ** 2).sum()), gain)  # W: omnidirectional
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
        names = {"name": Path(path).name, "source": pair[0], "listener": pair[1]}
        text = _("Playing {name} from {source} at {listener}, head-tracked").format(**names)
        self._played = self._playback(path, pair, True, start, len(signal) / rate, gain, text)
        self.results.set_playing(True)
        self.log(
            _(
                "Playing {name} from {source} at {listener}, head-tracked (order {order} "
                "Ambisonics to binaural)"
            ).format(**names, order=order)
        )
        self._show_playing()
        return True

    def _playback(self, path, pair, tracked, start, length, gain, text) -> dict:
        """What plays, for ``_played``."""
        return {
            "path": path,
            "pair": pair,
            "tracked": tracked,
            "start": start,
            "since": time.perf_counter(),
            "length": length,
            "gain": gain,
            "scene": self._result_scene,
            "text": text,
            "told": False,
        }

    def _follow_playback(self) -> None:
        """Play on, from where it is, for the source and listener on show and with head
        tracking as ticked: the run has the responses of every source and listener."""
        played, pair = self._played, self.results.pair()
        tracked = self.results.tracking.isChecked()
        if played is None or pair is None or (pair, tracked) == (played["pair"], played["tracked"]):
            return
        position = played["start"] + time.perf_counter() - played["since"]
        if position >= played["length"]:
            return  # only the reverberation is left
        self.play_file(played["path"], start=position, gain=played["gain"])

    def _player_poll(self) -> None:
        if self._player is None or self._player[0].is_alive():
            return
        _thread, _stop, outcome = self._player
        self._player = None
        self._player_timer.stop()
        self.results.set_playing(False)
        if "error" in outcome:
            self._error(_("Cannot play"), str(outcome["error"]))
        elif outcome["report"].xruns:
            count = outcome["report"].xruns
            self.log(
                ngettext(
                    "{count} block dropped out while playing.",
                    "{count} blocks dropped out while playing.",
                    count,
                ).format(count=count),
                "warning",
            )

    def _playing(self, playing: bool) -> None:
        self.stop_action.setEnabled(playing)
        self.now_playing.setVisible(playing)
        self.stop_button.setVisible(playing)
        if not playing:
            self._played = None

    def _show_playing(self) -> None:
        """Say in the status bar what plays, and whether it is from before the latest
        changes to the scene: what plays does not follow them."""
        played = self._played
        if played is None:
            return
        changed = played["scene"] is not self.document.scene
        self.now_playing.setText(
            # Translators: {playing} is what plays, as "Playing {name} from..." says.
            _("{playing} (before the latest changes)").format(playing=played["text"])
            if changed
            else played["text"]
        )
        if changed and not played["told"]:
            played["told"] = True
            self.log(
                _(
                    "The sound playing is from before the latest changes to the scene and does "
                    "not follow them: to hear them, Listen again (after running again, F5, if "
                    "the results are out of date)."
                )
            )

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
            _(TITLE),
            f"<p><b>Acoustic Spaces</b> {__version__}</p><p>"
            + _(
                "Room acoustics simulation: early reflections by ray tracing or image "
                "sources, the late tail by stochastic ray tracing, and binaural, Ambisonics "
                "and loudspeaker impulse responses."
            )
            + "</p><p>GPL-3.0-or-later. "
            + '<a href="https://github.com/naelolaiz/acoustic-spaces">'
            + _("Source")
            + "</a></p>",
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


def install_qt_translation(app: QApplication) -> QTranslator | None:
    """Qt's own words (file dialogs, standard buttons, the menu of a text field) in the
    language in use, from the translations Qt ships, when it has them."""
    code = i18n.language()
    if code not in i18n.LANGUAGES or code == "en":
        return None
    translator = QTranslator(app)
    folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if not translator.load(f"qtbase_{code}", folder):
        return None
    app.installTranslator(translator)
    return translator


def run(argv: list[str] | None = None) -> int:
    """Open the editor (with the scene file in ``argv``, if any) and run until it closes.
    ``--no-device-trial`` in ``argv`` switches off the late stage's device trial (see
    :mod:`acoustic_spaces.engine.ray_device`)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if ray_device.TRIAL_FLAG in argv:
        argv.remove(ray_device.TRIAL_FLAG)
        ray_device.set_trial(False)
    app = QApplication.instance() or QApplication([sys.argv[0] if sys.argv else TITLE])
    app.setApplicationName(TITLE)
    install_qt_translation(app)
    window = MainWindow()
    if argv:
        window.open_file(argv[0])
    window.show()
    return app.exec()


__all__ = ["MainWindow", "install_qt_translation", "run"]
