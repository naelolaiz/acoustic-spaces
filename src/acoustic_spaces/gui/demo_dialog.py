"""The window behind Simulation > Run demo…: which demo of the open scene to run, which
of its steps, and its options at the defaults the command line uses.

Every scene has *This scene* (:mod:`acoustic_spaces.demos.scene_demo`): its sources
play the WAV files their ``signal`` names, and every listener is heard in turn in a
video. A scene of an example that comes with a demo (:mod:`acoustic_spaces.demos`), or
a copy of one, also has that demo, with its own recordings, reading the open scene. The
form is built from the demo's own options, so the editor and the command line run
exactly the same thing. It shows the options that change what the demo plays, each
labelled in words with its flag in the tip; Advanced shows the step to run, what
fine-tunes the run, the files and folders the demo reads and writes, and that command,
to run again or to put in a script. A demo whose sources can play files of the user's
(``--input``) lists them in a table, each with what it plays and the WAV file it plays
instead. A number
shows its unit, and one whose default the scene gives (the pass-by demos' ``--speed``)
says so at zero; the line under the form says what the options make of the run, where
the demo says (how far, how fast and how long the vehicle drives), under the Movement box.
Above Run, a line says what the run downloads and how much: the recordings of a demo
whose steps fetch them (:attr:`~acoustic_spaces.demos.Step.downloads`), until they are
there.

The Movement box lists the sources and listeners of the scene that have a walk, ticked
when they move in the run. This scene's ticks start as Moves sets them in the scene and
can be changed for one run (``--moving``); an example's demo shows what it moves. Edit
movement… closes the window and opens the Movement panel on the walker.

However the window closes, it keeps the options changed from where a first run starts
them, in the user's settings (:func:`~acoustic_spaces.i18n.setting`), and opens with
them again, each demo with its own (:func:`keeps` says which options): a pick from a
list, a number, a tick and a file of the user's (the HRTFs) whatever the scene, and what
names something in the scene (its listeners, the files its sources play, what moves,
the format, a tour) for that scene file, which also keeps the demo picked. A line above
Run lists them, and Defaults puts the demo's options back.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces import i18n
from acoustic_spaces.demos import (
    MAP_KINDS,
    NO_MOVERS,
    PATH_KINDS,
    Demo,
    Option,
    listener_names,
    moving_names,
)
from acoustic_spaces.gui.inspector import int_spin, spin
from acoustic_spaces.gui.sizing import body_size, fit, reading_width, scroll_bar_width, scrolled
from acoustic_spaces.i18n import _, ngettext

#: The folder under the user's home the editor runs a demo in when the scene's own folder
#: is not to be written: the examples of an installed copy, or a folder that cannot be.
HOME_FOLDER = "acoustic-spaces"
#: The step the window starts on: every step, leaving out what is done already.
ALL = "all"
#: The user's setting (:func:`~acoustic_spaces.i18n.setting`) that keeps the options
#: changed in the window for the next time it opens: ``{"demos": {demo: {option:
#: value}}, "scenes": {scene file: {"demo": the demo picked, "demos": {demo: {option:
#: value}}}}}``, demos by :func:`demo_key`, values as JSON (:func:`saved_value`).
DEMO_SETTING = "editor_run_demo"
#: How many scene files the setting keeps their own options for, the latest closed.
KEPT_SCENES = 20
#: The kinds of option whose value names something in the scene (its listeners, the
#: sources a file plays for, what moves, an output's label, a tour's stops): kept per
#: scene file (:func:`per_scene`). The others (a pick from a list, a number, a tick, a
#: file of the user's) are kept per demo, whatever the scene.
SCENE_KINDS = ("text", "listeners", "inputs", "movers")
#: What :func:`restored` gives for a kept value the option does not take.
NOT_TAKEN: Any = object()
#: The largest number a number field holds, either way (:func:`field_for`).
NUMBER_RANGE = 1_000_000


def writable(folder: Path, name: str) -> Path:
    """``folder`` when it can be written and is not among the examples of an installed
    copy (:func:`~acoustic_spaces.examples.is_installed_example`), else one called
    ``name`` under the user's home (``~/acoustic-spaces/<name>``). An installed copy's
    examples are left alone even where they can be written (a virtual environment), as
    an upgrade or an uninstall would take what a demo wrote there with them."""
    from acoustic_spaces.examples import is_installed_example

    if not is_installed_example(folder) and os.access(folder, os.W_OK):
        return folder
    return Path.home() / HOME_FOLDER / name


def work_base(demo: Demo) -> Path:
    """The folder the editor runs ``demo`` in: the example's own folder, where the
    command line runs it, or one of the same name under the user's home for the
    examples of an installed copy or a folder that cannot be written."""
    return writable(demo.script.parent, demo.example)


def reads_scene(demo: Demo) -> Option | None:
    """The option through which ``demo`` reads the scene it plays in: the demo of any
    scene's leading option, or the one whose default is the example's ``scene.yaml``."""
    for option in demo.options:
        if option.kind != "file":
            continue
        if option.name == demo.leading:
            return option
        if option.default is not None and Path(option.default).name == "scene.yaml":
            return option
    return None


def demos_for(scene: str | Path | None) -> list[tuple[Demo, argparse.Namespace]]:
    """The demos of the open scene, each in the folder it runs in and with the arguments
    it starts with: *This scene* first, then the demos of the example the scene is or
    is a copy of (:func:`~acoustic_spaces.examples.example_of`: its folder's name, or
    the example's scene name wherever the copy is saved), its own first, then the
    example's scripts that are no demos (:func:`~acoustic_spaces.demos.scripts_of`), each
    reading the open scene instead of the example's."""
    from acoustic_spaces.demos import demos_of, scripts_of
    from acoustic_spaces.demos.scene_demo import DEMO
    from acoustic_spaces.examples import example_of

    found: list[tuple[Demo, argparse.Namespace]] = []
    path = Path(scene).resolve() if scene is not None else None
    base = writable(path.parent, path.parent.name) if path else Path.home() / HOME_FOLDER
    demo = DEMO.at(base)
    args = demo.defaults()
    args.scene = path
    args = demo.resolved(args)  # with the scene's own HRTFs, when it names some
    if path is None or base != path.parent:  # not beside the scene, its default
        args.work = base / "demo"
    found.append((demo, args))
    name = example_of(path) if path is not None else None
    for example in [*demos_of(name), *scripts_of(name)] if name is not None else []:
        demo = example.at(writable(path.parent, example.example))
        args = starting_arguments(demo)
        option = reads_scene(demo)
        if option is not None:
            setattr(args, option.name, path)
        found.append((demo, args))
    return found


def starting_arguments(demo: Demo) -> argparse.Namespace:
    """Every option of ``demo`` (already at its :func:`work_base`) at its default, with
    the paths filled in: what the demo writes goes under that folder, what it reads
    comes from the example when the folder has no copy of it
    (:meth:`~acoustic_spaces.demos.Demo.default_path`)."""
    args = demo.resolved(demo.defaults())
    for option in demo.options:
        if option.kind != "file" or option.default is None:
            continue
        if not Path(getattr(args, option.name)).exists():
            setattr(args, option.name, demo.default_path(option))
    return args


def demo_key(demo: Demo) -> str:
    """How the settings name ``demo``: its example and name (``stadium_concert/align``,
    as two examples have an ``align``), or its name alone for the demo of any scene."""
    return f"{demo.example}/{demo.name}" if demo.example else demo.name


def scene_key(scene: str | Path | None) -> str | None:
    """How the settings name the scene file ``scene``: its full path; None for none."""
    return None if scene is None else str(Path(scene).resolve())


def keeps(demo: Demo, option: Option) -> bool:
    """Whether the window keeps ``option`` of ``demo`` for the next time it opens: every
    option but the folders and files the demo writes, the example's files it reads and
    the scene it plays, which follow where the scene is."""
    if option.kind in ("path", "folder"):
        return False
    if option.kind == "file":
        return option.default is None and option is not reads_scene(demo)
    return True


def per_scene(demo: Demo, option: Option) -> bool:
    """Whether the window keeps ``option`` of ``demo`` per scene file: one of a kind
    that names something in a scene (:data:`SCENE_KINDS`), except the credit line of a
    song of the user's (``--credit`` beside ``--music``), which goes with the song."""
    if option.kind not in SCENE_KINDS:
        return False
    return not (option.name == "credit" and any(o.name == "music" for o in demo.options))


def saved_value(demo: Demo, option: Option, value: Any) -> Any:
    """``value`` (as :func:`value_of` gives it) as the settings keep it: JSON data, with
    paths in full, counted from the folder ``demo`` runs in."""
    if value is None:
        return None
    if option.kind in MAP_KINDS:
        return {str(name): str(demo.folder / file) for name, file in value.items()}
    if option.kind in PATH_KINDS:
        return str(demo.folder / value)  # a path in full stays as it is
    return value


def restored(option: Option, data: Any) -> Any:
    """The value of ``option`` that ``data`` (kept as :func:`saved_value` writes it) gives,
    or :data:`NOT_TAKEN` for one the option does not take: a song or a stem no longer
    offered, a number of the wrong kind or more than its field holds, a settings file
    edited by hand."""
    kind = option.kind
    if kind == "flag":
        return data if isinstance(data, bool) else NOT_TAKEN
    if kind in MAP_KINDS:
        if data is None:
            return None
        if not isinstance(data, dict):
            return NOT_TAKEN
        if not all(isinstance(n, str) and isinstance(f, str) and f for n, f in data.items()):
            return NOT_TAKEN
        if kind == "files":  # the names the option lists, the others dropped
            data = {name: file for name, file in data.items() if name in option.choices}
        return {name: Path(file) for name, file in data.items()} or NOT_TAKEN
    if option.choices:  # a list to pick from, which always holds a value
        return data if isinstance(data, str) and data in option.choices else NOT_TAKEN
    if kind in ("int", "float"):
        if data is None:  # given none: its unset text, or an empty field
            return None if option.unset or option.default is None else NOT_TAKEN
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            return NOT_TAKEN
        if kind == "int" and not isinstance(data, int):
            return NOT_TAKEN
        if option.default is not None or option.unset:  # a field of numbers, not text
            lowest = 0 if option.unset else -NUMBER_RANGE
            if not lowest <= data <= NUMBER_RANGE:
                return NOT_TAKEN
        try:
            number = data if kind == "int" else float(data)
        except OverflowError:
            return NOT_TAKEN
        return number if kind == "int" or math.isfinite(number) else NOT_TAKEN
    if data is None:
        return None
    if not isinstance(data, str):
        return NOT_TAKEN
    if kind == "file":
        return Path(data) if data else NOT_TAKEN
    return data


def _demo_setting() -> dict:
    """The setting :data:`DEMO_SETTING`, with its two tables there and tables."""
    data = i18n.setting(DEMO_SETTING)
    data = data if isinstance(data, dict) else {}
    tables = {name: data.get(name) for name in ("demos", "scenes")}
    return {name: table if isinstance(table, dict) else {} for name, table in tables.items()}


def _scene_entry(setting: dict, scene: str | None) -> dict:
    entry = setting["scenes"].get(scene) if scene is not None else None
    return entry if isinstance(entry, dict) else {}


def kept_options(demo: Demo, scene: str | None) -> dict[str, Any]:
    """The options of ``demo`` the window kept (:func:`keeps`), by name, at the values
    it kept them: those of a kind that names something in a scene
    (:func:`per_scene`) as kept for the scene file ``scene`` (:func:`scene_key`;
    none for None), the others as kept for the demo. A value the option does not take
    (:func:`restored`) is left out."""
    setting = _demo_setting()
    key = demo_key(demo)
    own = setting["demos"].get(key)
    named = _scene_entry(setting, scene).get("demos")
    named = named.get(key) if isinstance(named, dict) else None
    values: dict[str, Any] = {}
    for option in demo.options:
        table = named if per_scene(demo, option) else own
        if not keeps(demo, option) or not isinstance(table, dict) or option.name not in table:
            continue
        value = restored(option, table[option.name])
        if value is not NOT_TAKEN:
            values[option.name] = value
    return values


def kept_demo(scene: str | None) -> str | None:
    """The name of the demo the window showed when it last closed on the scene file
    ``scene`` (:func:`scene_key`), or None."""
    picked = _scene_entry(_demo_setting(), scene).get("demo")
    return picked if isinstance(picked, str) else None


def keep_options(
    demo: Demo, scene: str | None, values: dict[str, Any], picked: str | None = None
) -> None:
    """Keep ``values`` (the options of ``demo`` changed from their start, by name, as
    :func:`saved_value` gives them) for the next time the window opens; the others are
    forgotten. Those of a kind that names something in a scene go with the scene file
    ``scene`` (none kept for None), as does ``picked``, the demo picked; the setting
    keeps the :data:`KEPT_SCENES` scene files closed last. Not written where the user's
    settings cannot be: what the window keeps is a convenience, not the run's."""
    setting = _demo_setting()
    key = demo_key(demo)
    by_kind = {
        named: {
            name: value
            for name, value in values.items()
            if per_scene(demo, demo.option(name)) == named
        }
        for named in (False, True)
    }
    if by_kind[False]:
        setting["demos"][key] = by_kind[False]
    else:
        setting["demos"].pop(key, None)
    if scene is not None:
        scenes = setting["scenes"]
        entry = _scene_entry(setting, scene)
        scenes.pop(scene, None)
        named = entry.get("demos")
        named = named if isinstance(named, dict) else {}
        if by_kind[True]:
            named[key] = by_kind[True]
        else:
            named.pop(key, None)
        entry["demos"] = named
        if picked is not None:
            entry["demo"] = picked
        scenes[scene] = entry  # the latest last
        while len(scenes) > KEPT_SCENES:
            del scenes[next(iter(scenes))]
    try:
        i18n.save_setting(DEMO_SETTING, setting)
    except OSError:
        pass


class PathField(QWidget):
    """The field of an option that names a file or a folder: the path, and a button that
    opens the file window on it."""

    #: The path was typed or chosen.
    changed = Signal()

    def __init__(
        self,
        option: Option,
        tip: str,
        parent: QWidget | None = None,
        sets: bool = False,
    ) -> None:
        super().__init__(parent)
        self.option = option
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setToolTip(tip)
        self.edit.textChanged.connect(lambda *_a: self.changed.emit())
        self.browse = QPushButton(_("Browse…"))
        self.browse.setToolTip(
            _("Choose the folder") if option.kind == "folder" else _("Choose the file")
        )
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.edit, 1)
        row.addWidget(self.browse)
        self.sets: QPushButton | None = None
        if sets:
            self.sets = QPushButton(_("Download HRTFs…"))
            self.sets.setToolTip(
                _("Free HRTF sets to download, measured on dummy heads and on people")
            )
            self.sets.clicked.connect(self._download)
            row.addWidget(self.sets)

    def value(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None

    def set_value(self, value) -> None:
        self.edit.setText("" if value is None else str(value))

    def _download(self) -> None:
        """Pick a free HRTF set, download it and put it in the field."""
        from acoustic_spaces.gui.hrtf_dialog import HrtfDialog

        dialog = HrtfDialog(self)
        accepted, path = dialog.exec(), dialog.path
        dialog.deleteLater()
        if accepted and path is not None:
            self.edit.setText(str(path))

    def _browse(self) -> None:
        start = self.edit.text() or str(Path.home())
        name = self.option.option
        filters = self.option.filter or _("All files (*)")
        if self.option.kind == "folder":
            chosen = QFileDialog.getExistingDirectory(
                self, _("Folder for {option}").format(option=name), start
            )
        elif self.option.kind == "file":
            chosen, _filter = QFileDialog.getOpenFileName(
                self, _("File for {option}").format(option=name), start, filters
            )
        else:
            chosen, _filter = QFileDialog.getSaveFileName(
                self, _("File for {option}").format(option=name), start, filters
            )
        if chosen:
            self.edit.setText(chosen)


class InputsField(QWidget):
    """The field of ``--input``: a row for each source the demo feeds (:attr:`Demo.plays`,
    which :meth:`set_sources` gives), with what it plays and the WAV file it plays
    instead, typed or chosen with Choose WAV…; Default empties it again. For a ``files``
    option (``--stem STEM=WAV``), a row for each name the option takes, with its file."""

    #: A file was typed, chosen or taken back.
    changed = Signal()

    def __init__(self, option: Option, tip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.option = option
        self.inputs: dict[str, Path] = {}
        self.plays: dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if option.kind == "inputs":
            self.SOURCE, self.PLAYS, self.FILE = 0, 1, 2
            heads = (
                (_("Source"), _("The source, as the scene names it")),
                (_("Plays"), _("What the source plays in the demo, unless a file is given")),
                (
                    _("Instead"),
                    _(
                        "The WAV file the source plays instead (--input SOURCE=WAV); empty: "
                        "what Plays says. The table's tip says how loud it plays"
                    ),
                ),
            )
        else:
            self.SOURCE, self.PLAYS, self.FILE = 0, None, 1
            heads = (
                (_("Name"), _("The names the option takes ({how})").format(how=option.metavar)),
                (
                    _("File"),
                    _("The WAV file for it ({option} {how}); empty: none").format(
                        option=option.option, how=option.metavar
                    ),
                ),
            )
        self.table = QTableWidget(0, len(heads))
        self.table.setToolTip(tip)
        for column, (text, head_tip) in enumerate(heads):
            item = QTableWidgetItem(text)
            item.setToolTip(head_tip)
            self.table.setHorizontalHeaderItem(column, item)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            self.FILE, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemChanged.connect(self._typed)
        self.table.itemSelectionChanged.connect(self._selected)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.choose = QPushButton(_("Choose WAV…"))
        self.choose.setToolTip(_("Choose the WAV file the selected source plays instead"))
        self.choose.clicked.connect(self._choose)
        self.default = QPushButton(_("Default"))
        self.default.setToolTip(_("The selected source plays what Plays says again"))
        self.default.clicked.connect(self._default)
        row.addWidget(self.choose)
        row.addWidget(self.default)
        row.addStretch(1)
        layout.addLayout(row)
        self._selected()
        if option.kind != "inputs":
            self.set_sources(dict.fromkeys(option.choices, ""))

    def set_sources(self, plays: dict[str, str]) -> None:
        """The rows: each source and what it plays. A file given to a source no longer
        listed is dropped."""
        if plays == self.plays and self.table.rowCount() == len(plays):
            return
        self.plays = dict(plays)
        self.inputs = {name: file for name, file in self.inputs.items() if name in plays}
        self.table.blockSignals(True)
        self.table.setRowCount(len(plays))
        for row, (name, what) in enumerate(plays.items()):
            for column, text in ((self.SOURCE, name), (self.PLAYS, what)):
                if column is None:
                    continue
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            file = self.inputs.get(name)
            self.table.setItem(row, self.FILE, QTableWidgetItem("" if file is None else str(file)))
        self.table.resizeColumnToContents(self.SOURCE)
        if self.PLAYS is not None:
            self.table.resizeColumnToContents(self.PLAYS)
        self.table.blockSignals(False)
        # As tall as its rows, up to about eight; a long list scrolls.
        rows = sum(self.table.rowHeight(r) for r in range(min(len(plays), 8)))
        self.table.setFixedHeight(
            self.table.horizontalHeader().height() + rows + 2 * self.table.frameWidth()
        )
        self._selected()

    def value(self) -> dict[str, Path] | None:
        return dict(self.inputs) or None

    def set_value(self, value) -> None:
        self.inputs = {str(name): Path(file) for name, file in (value or {}).items()}
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            file = self.inputs.get(self.table.item(row, self.SOURCE).text())
            self.table.item(row, self.FILE).setText("" if file is None else str(file))
        self.table.blockSignals(False)
        self._selected()

    def _source(self) -> str | None:
        row = self.table.currentRow()
        if row < 0 or not self.table.item(row, self.SOURCE).isSelected():
            return None
        return self.table.item(row, self.SOURCE).text()

    def _selected(self) -> None:
        name = self._source()
        self.choose.setEnabled(name is not None)
        self.default.setEnabled(name is not None and name in self.inputs)

    def _set(self, name: str, file: Path | None) -> None:
        if file is None:
            self.inputs.pop(name, None)
        else:
            self.inputs[name] = file
        self.set_value(self.inputs)
        self.changed.emit()

    def _typed(self, item: QTableWidgetItem) -> None:
        if item.column() != self.FILE:
            return
        text = item.text().strip()
        self._set(self.table.item(item.row(), self.SOURCE).text(), Path(text) if text else None)

    def _choose(self) -> None:
        name = self._source()
        if name is None:
            return
        start = str(self.inputs.get(name) or Path.home())
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            _("WAV file for {source}").format(source=name),
            start,
            self.option.filter or _("All files (*)"),
        )
        if chosen:
            self._set(name, Path(chosen))

    def _default(self) -> None:
        name = self._source()
        if name is not None:
            self._set(name, None)


class ListenersField(QWidget):
    """The field of ``--listeners``: a tick for each listener the demo can play
    (:attr:`Demo.listeners`, which :meth:`set_listeners` gives), all ticked by default;
    a run plays the ticked ones. All and None tick or untick them all."""

    #: A tick changed.
    changed = Signal()

    def __init__(self, option: Option, tip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.option = option
        self.names: list[str] = []
        #: The listeners unticked, kept by name while the list changes.
        self.unticked: set[str] = set()
        #: What :meth:`set_value` asked for before the list was there.
        self._wanted: list[str] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setToolTip(tip)
        self.list.itemChanged.connect(self._ticked)
        layout.addWidget(self.list)
        row = QHBoxLayout()
        self.all = QPushButton(_("All"))
        self.all.setToolTip(_("Tick every listener: the demo plays them all, its default"))
        self.all.clicked.connect(lambda: self._tick_all(True))
        self.none = QPushButton(_("None"))
        self.none.setToolTip(_("Untick every listener, to tick the few to hear"))
        self.none.clicked.connect(lambda: self._tick_all(False))
        row.addWidget(self.all)
        row.addWidget(self.none)
        row.addStretch(1)
        layout.addLayout(row)

    def set_listeners(self, names: list[str]) -> None:
        """The rows: the listeners the demo can play, in its order, ticked unless they
        were unticked before (or :meth:`set_value` left them out)."""
        names = list(names)
        if names == self.names and self.list.count() == len(names):
            return
        if self._wanted is not None and names:
            self.unticked = set(names) - set(self._wanted)
            self._wanted = None
        self.names = names
        self.list.blockSignals(True)
        self.list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(
                (item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
            )
            ticked = name not in self.unticked
            item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        # As tall as its rows, up to about six; a longer list scrolls.
        rows = sum(self.list.sizeHintForRow(r) for r in range(min(len(names), 6)))
        self.list.setFixedHeight(rows + 2 * self.list.frameWidth())

    def ticked(self) -> list[str]:
        return [name for name in self.names if name not in self.unticked]

    def value(self) -> str | None:
        """``--listeners``: None while every listener is ticked (or the list is not
        there), else the ticked ones, comma separated (empty for none)."""
        if not self.names:
            return None if self._wanted is None else ",".join(self._wanted)
        ticked = self.ticked()
        return None if len(ticked) == len(self.names) else ",".join(ticked)

    def set_value(self, value) -> None:
        names = listener_names(value)
        if not self.names:
            self._wanted = names
            return
        self.unticked = set() if names is None else set(self.names) - set(names)
        self.list.blockSignals(True)
        for row, name in enumerate(self.names):
            state = Qt.CheckState.Unchecked if name in self.unticked else Qt.CheckState.Checked
            self.list.item(row).setCheckState(state)
        self.list.blockSignals(False)

    def _ticked(self, item: QListWidgetItem) -> None:
        name = item.text()
        if item.checkState() == Qt.CheckState.Checked:
            self.unticked.discard(name)
        else:
            self.unticked.add(name)
        self.changed.emit()

    def _tick_all(self, ticked: bool) -> None:
        self.set_value(None if ticked else "")
        self.changed.emit()


def walkers_of(scene: str | Path | None) -> list[tuple[str, object]]:
    """The sources and listeners of the scene file that have a walk, on or off, in use or
    left out of its simulations, as ("listeners" or "sources", item), listeners first;
    none when it cannot be read."""
    if scene is None:
        return []
    from acoustic_spaces.scene import Scene

    try:
        loaded = Scene.from_yaml(scene)
    except Exception:  # the demo's check says what is wrong with it
        return []
    return [
        (kind, item)
        for kind in ("listeners", "sources")
        for item in getattr(loaded, kind)
        if item.walk is not None
    ]


def walk_text(item) -> str:
    """A walk in a few words: how it goes, how far and for how long."""
    from acoustic_spaces.walkthrough import knots

    path = knots(item)
    walk = item.walk
    if walk.keyframes is not None:
        count = len(walk.keyframes)
        text = ngettext(
            "{count} keyframe, {length:.1f} m in {time:.1f} s",
            "{count} keyframes, {length:.1f} m in {time:.1f} s",
            count,
        ).format(count=count, length=path.length_m, time=path.duration_s)
    else:
        text = _("At {speed:g} m/s ({kmh:.1f} km/h), {length:.1f} m in {time:.1f} s").format(
            speed=walk.speed_m_s,
            kmh=3.6 * walk.speed_m_s,
            length=path.length_m,
            time=path.duration_s,
        )
    return text + (_(", with Doppler") if walk.doppler else "")


class MovementBox(QGroupBox):
    """What moves in a run: a row for each source and listener of the scene with a walk,
    ticked when it moves; one left out of the scene's simulations shows only when it is
    ticked (the pass-by demos' vehicle, which they put in the scenes they write). With
    :meth:`set_mode` ``editable`` (a demo with a ``movers`` option, ``--moving``) the
    ticks are the run's, starting as the scene's Moves sets them; otherwise they show
    what the demo moves. Edit movement… asks for the Movement panel on the selected
    walker (:attr:`edit`)."""

    #: A tick changed.
    changed = Signal()
    #: Edit movement…: the walker's kind ("sources" or "listeners") and name, or two
    #: empty strings when the scene has none.
    edit = Signal(str, str)

    MOVES, NAME, KIND, WALK = range(4)

    def __init__(self, walkers: list[tuple[str, object]], parent: QWidget | None = None) -> None:
        super().__init__(_("Movement"), parent)
        self.setToolTip(
            _(
                "The sources and listeners with a walk. Ticked ones move along it while the "
                "piece plays, all on one clock; the others stand where they start"
            )
        )
        self.walkers = [(kind, item.name, item.walk.enabled) for kind, item in walkers]
        #: Whether each walker is in the scene's simulations.
        self.used = [item.in_use for _kind, item in walkers]
        self.editable = False
        layout = QVBoxLayout(self)
        # As tall as its contents, however tall the window is.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.table = QTableWidget(len(self.walkers), 4)
        self.table.setToolTip(self.toolTip())
        heads = (
            (_("Moves"), _("Tick to move it along its walk in this run")),
            (_("Name"), _("The source or listener, as the scene names it")),
            (_("What"), _("A source or a listener")),
            (_("Walk"), _("How it goes, how far and for how long")),
        )
        for column, (text, tip) in enumerate(heads):
            head = QTableWidgetItem(text)
            head.setToolTip(tip)
            self.table.setHorizontalHeaderItem(column, head)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            self.WALK, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for row, (kind, item) in enumerate(walkers):
            moves = QTableWidgetItem()
            name = QTableWidgetItem(item.name)
            what = QTableWidgetItem(_("Listener") if kind == "listeners" else _("Source"))
            walk = QTableWidgetItem(walk_text(item))
            for cell in (moves, name, what, walk):
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if not item.in_use:
                    cell.setToolTip(
                        _(
                            "Left out of the scene's simulations: a demo that moves it puts "
                            "it in the scene it writes"
                        )
                    )
            self.table.setItem(row, self.MOVES, moves)
            self.table.setItem(row, self.NAME, name)
            self.table.setItem(row, self.KIND, what)
            self.table.setItem(row, self.WALK, walk)
        self.table.itemChanged.connect(self._ticked)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.edit_button = QPushButton(_("Edit movement…"))
        self.edit_button.setToolTip(
            _(
                "Close this window and open the Movement panel on the selected walker, to "
                "change its walk or give a source or listener one; save the scene before "
                "running the demo"
            )
        )
        self.edit_button.clicked.connect(self._edit)
        self.reset = QPushButton(_("As in the scene"))
        self.reset.setToolTip(_("Tick what Moves ticks in the scene again"))
        self.reset.clicked.connect(lambda: self.set_value(None))
        row.addWidget(self.edit_button)
        row.addWidget(self.reset)
        row.addStretch(1)
        layout.addLayout(row)

    @property
    def own(self) -> list[str]:
        """The walkers in use whose walk is on in the scene."""
        pairs = zip(self.walkers, self.used, strict=True)
        return [name for (_kind, name, on), used in pairs if on and used]

    @property
    def any_used(self) -> bool:
        """Whether a walker is in the scene's simulations."""
        return any(self.used)

    def ticked(self) -> list[str]:
        return [
            self.walkers[row][1]
            for row in range(self.table.rowCount())
            if self.table.item(row, self.MOVES).checkState() == Qt.CheckState.Checked
        ]

    def set_mode(self, editable: bool, ticked: list[str], note: str) -> None:
        """Whether the ticks are the run's to change, what is ticked, and the line above."""
        self.editable = editable
        self.note.setText(note)
        self.reset.setVisible(editable and self.any_used)
        self._tick(ticked)
        shown = []
        for row, (_kind, name, _on) in enumerate(self.walkers):
            self.table.setRowHidden(row, not self.used[row] and name not in ticked)
            if not self.table.isRowHidden(row):
                shown.append(row)
        rows = sum(self.table.rowHeight(r) for r in shown[:6])
        self.table.setFixedHeight(
            self.table.horizontalHeader().height() + rows + 2 * self.table.frameWidth()
        )
        self.table.setVisible(bool(shown))
        for column in (self.MOVES, self.NAME, self.KIND):  # the tick boxes in too
            self.table.resizeColumnToContents(column)

    def value(self) -> str | None:
        """``--moving``: None while the ticks are the scene's own, else the ticked names
        (``none`` for no tick)."""
        ticked = self.ticked()
        if set(ticked) == set(self.own):
            return None
        return ",".join(ticked) if ticked else NO_MOVERS

    def set_value(self, value) -> None:
        names = moving_names(value)
        self._tick(self.own if names is None else names)
        self.changed.emit()

    def _tick(self, names: list[str]) -> None:
        """Tick ``names``; the ticks can be clicked only when they are the run's."""
        self.table.blockSignals(True)
        for row, (_kind, name, _on) in enumerate(self.walkers):
            item = self.table.item(row, self.MOVES)
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if not self.editable:
                flags &= ~Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if name in names else Qt.CheckState.Unchecked)
        self.table.blockSignals(False)

    def _ticked(self, item: QTableWidgetItem) -> None:
        if item.column() == self.MOVES and self.editable:
            self.changed.emit()

    def _edit(self) -> None:
        shown = [r for r in range(len(self.walkers)) if not self.table.isRowHidden(r)]
        if not shown:
            self.edit.emit("", "")
            return
        row = self.table.currentRow()
        kind, name, _on = self.walkers[row if row in shown else shown[0]]
        self.edit.emit(kind, name)


def field_for(option: Option, tip: str) -> QWidget:
    """The widget that asks for ``option``: a tick for a flag, a list for a choice, a
    spin box for a number (with its unit; at zero, its ``unset`` text, for one the scene
    gives by default), a path with a Browse… button for a file or a folder, a table of
    the sources for ``--input``, a tick for each listener for ``--listeners``, and a
    text field for the rest (empty where the demo takes none)."""
    if option.kind in MAP_KINDS:
        return InputsField(option, tip)
    if option.kind == "listeners":
        return ListenersField(option, tip)
    if option.kind in PATH_KINDS:
        return PathField(option, tip, sets=option.name == "hrtf")
    if option.kind == "flag":
        box = QCheckBox()
        box.setChecked(bool(option.default))
        box.setToolTip(tip)
        return box
    if option.choices:
        combo = QComboBox()
        combo.addItems(list(option.choices))
        if option.default in option.choices:
            combo.setCurrentText(str(option.default))
        combo.setToolTip(tip)
        return combo
    if option.kind in ("int", "float") and (option.default is not None or option.unset):
        number = (
            int_spin(-NUMBER_RANGE, NUMBER_RANGE)
            if option.kind == "int"
            else spin(-NUMBER_RANGE, NUMBER_RANGE, 2, 0.5)
        )
        if option.unit:
            number.setSuffix(f" {option.unit}")
        if option.unset:  # zero, the lowest, stands for not given
            number.setMinimum(0)
            number.setSpecialValueText(option.unset)
        number.setValue(option.default if option.default is not None else 0)
        number.setToolTip(tip)
        return number
    edit = QLineEdit("" if option.default is None else str(option.default))
    edit.setToolTip(tip)
    return edit


def value_of(option: Option, widget: QWidget):
    """What the field holds, in the type the demo's step expects (None where it is
    empty: the demo is given no such option)."""
    if isinstance(widget, (PathField, InputsField, MovementBox, ListenersField)):
        return widget.value()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        text = widget.text().strip()
        if text and option.kind in ("int", "float"):
            try:
                return int(text) if option.kind == "int" else float(text)
            except ValueError:
                return text  # the demo's check says it is no number
        return text or None
    value = widget.value()
    if option.unset and value == widget.minimum():
        return None  # its unset text: the demo is given none
    return int(value) if option.kind == "int" else float(value)


def show_value(option: Option, widget: QWidget, value: Any) -> None:
    """Put ``value`` (as :func:`value_of` gives it) in the field ``widget`` of ``option``;
    a list keeps its pick where the value is none of its choices."""
    if isinstance(widget, (PathField, InputsField, MovementBox, ListenersField)):
        widget.set_value(value)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QComboBox):
        if value in option.choices:
            widget.setCurrentText(str(value))
    elif isinstance(widget, QLineEdit):
        widget.setText("" if value is None else str(value))
    else:  # a number; none is its unset text, at the lowest
        widget.setValue(widget.minimum() if value is None else value)


class DemoDialog(QDialog):
    """Which demo to run and how. :meth:`chosen` gives the demo (in the folder it runs
    in), the step and the arguments OK asked for."""

    def __init__(
        self,
        demos: list[Demo] | None = None,
        parent: QWidget | None = None,
        start: str | None = None,
        *,
        scene: str | Path | None = None,
        unsaved: bool = False,
    ) -> None:
        """``scene`` is the open scene's file, ``unsaved`` whether it has changes the
        file lacks; ``demos`` (the examples' demos at their folders, for tests) replaces
        the scene's; ``start`` names the demo to start on (by default the one on show
        when the window last closed on the scene, else the example's own demo when the
        scene has one, else *This scene*)."""
        super().__init__(parent)
        self.setWindowTitle(_("Run demo"))
        #: The scene file as the settings name it, for what the window keeps per scene.
        self.scene_key = scene_key(scene) if demos is None else None
        if demos is not None:
            pairs = [(d.at(work_base(d)), None) for d in demos]
        else:
            pairs = demos_for(scene)
            names = [demo.name for demo, _args in pairs]
            if start is None:  # as last left, or the example's own demo after This scene
                start = kept_demo(self.scene_key)
                if start not in names:
                    start = names[min(1, len(names) - 1)]
        self.demos = [demo for demo, _args in pairs]
        self.starts = {demo.name: args for demo, args in pairs if args is not None}
        self.unsaved = unsaved or (demos is None and scene is None)
        self.fields: dict[str, QWidget] = {}
        #: The demo whose options the form shows, and the arguments it started at.
        self.shown: Demo | None = None
        self.start_args = self.opened_args = argparse.Namespace()
        self.kept_values: dict[str, Any] = {}
        #: What Edit movement… asked for: the walker's kind and name (empty for none).
        self.edit_request: tuple[str, str] | None = None
        # The form scrolls where the screen is too small for it, rather than squeezing
        # its tables and buttons; the buttons stay below it.
        body = QWidget()
        layout = self._layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            _(
                "A demo plays sound through the scene and makes a video of what each listener "
                "hears, and an example's script prints what it finds; the messages say what "
                "they are doing, and Cancel stops them."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        head = QFormLayout()
        self.picker = QComboBox()
        for demo in self.demos:
            self.picker.addItem(_(demo.title), demo.name)  # This scene is translated
        self.picker.setToolTip(
            _(
                "The demo to run: This scene, with the sounds its sources play, or one of "
                "the example's own demos or scripts when the scene is an example's or a "
                "copy of one"
            )
        )
        if start is not None:
            index = self.picker.findData(start)
            if index >= 0:
                self.picker.setCurrentIndex(index)
        self.step = QComboBox()
        self.step.setToolTip(
            _(
                "The step to run. all runs the demo from nothing to its video, leaving out "
                "the steps whose result is there already"
            )
        )
        label = QLabel(_("Demo"))
        label.setBuddy(self.picker)
        label.setToolTip(self.picker.toolTip())
        head.addRow(label, self.picker)
        # One demo only (a scene of no example): nothing to pick from.
        head.setRowVisible(self.picker, len(self.demos) > 1)
        layout.addLayout(head)
        self.about = QLabel()
        self.about.setWordWrap(True)
        layout.addWidget(self.about)

        self.movement = MovementBox(walkers_of(scene if demos is None else None))
        #: Whether the Movement box shows: for a scene's demos, not for a script.
        self._moves = demos is None and scene is not None
        self.movement.setVisible(self._moves)
        self.movement.changed.connect(self._refresh)
        self.movement.edit.connect(self._edit_movement)
        layout.addWidget(self.movement)
        # What the options make of the run, where the demo says: beside what moves.
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.options = QWidget()
        layout.addWidget(self.options)

        # What fine-tunes a run, the files and folders it reads and writes and the
        # command line: shown on asking.
        self.more = QToolButton()
        self.more.setText(_("Advanced"))
        self.more.setCheckable(True)
        self.more.setArrowType(Qt.ArrowType.RightArrow)
        self.more.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.more.setAutoRaise(True)
        self.more.setToolTip(
            _(
                "Show the step to run, what fine-tunes the run, the files and folders the "
                "demo reads and writes, and the same run as a command"
            )
        )
        self.more.toggled.connect(self._show_advanced)
        layout.addWidget(self.more)
        self.advanced = QWidget()
        inner = self._advanced_layout = QVBoxLayout(self.advanced)
        inner.setContentsMargins(0, 0, 0, 0)
        steps = self._steps = QFormLayout()
        label = QLabel(_("Step"))
        label.setBuddy(self.step)
        label.setToolTip(self.step.toolTip())
        steps.addRow(label, self.step)
        inner.addLayout(steps)
        self.about_step = QLabel()
        self.about_step.setWordWrap(True)
        inner.addWidget(self.about_step)
        self.advanced_options = QWidget()
        inner.addWidget(self.advanced_options)
        self.command = QLineEdit()
        self.command.setReadOnly(True)
        self.command.setToolTip(_("The same run from a terminal, in the folder below"))
        self.folder = QLabel()
        self.folder.setWordWrap(True)
        command_label = QLabel(_("Command"))
        command_label.setBuddy(self.command)
        command_label.setToolTip(self.command.toolTip())
        row = QFormLayout()
        row.addRow(command_label, self.command)
        inner.addLayout(row)
        inner.addWidget(self.folder)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)
        # What the run downloads, where its steps say: above Run, whatever is open.
        self.downloads = QLabel()
        self.downloads.setWordWrap(True)
        layout.addWidget(self.downloads)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        layout.addWidget(self.problem)
        # The options kept for next time, Advanced's among them: whatever is open.
        self.kept = QLabel()
        self.kept.setWordWrap(True)
        layout.addWidget(self.kept)
        layout.addStretch(1)  # a taller window leaves the room above the buttons
        self.scroll = scrolled(body)
        outer = QVBoxLayout(self)
        outer.addWidget(self.scroll)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(_("Run"))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setToolTip(
            _("Run the step in the background; the messages say how it goes")
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setToolTip(
            _("Close without running anything; the options changed are kept for next time")
        )
        self.defaults = self.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        self.defaults.setText(_("Defaults"))
        self.defaults.setToolTip(
            _(
                "Put every option of this demo back where a first run starts it, the "
                "command line's defaults, and forget the ones kept from before"
            )
        )
        self.defaults.clicked.connect(self._defaults)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

        self.picker.currentIndexChanged.connect(self._chose_demo)
        self.step.currentIndexChanged.connect(self._refresh)
        self._chose_demo()
        self._fit(reading_width(self))

    def _fit(self, width: int) -> None:
        """As large as the form asks at ``width`` (or wider when it needs more), the
        buttons below, as far as the screen leaves room."""
        margins = self.layout().contentsMargins()
        across = margins.left() + margins.right()
        width = max(width - across, self.scroll.widget().minimumSizeHint().width())
        width += scroll_bar_width(self.scroll)
        body = body_size(self.scroll, width)
        spacing = self.layout().spacing()
        height = body.height() + spacing + self.buttons.sizeHint().height()
        fit(self, QSize(width + across, height + margins.top() + margins.bottom()))

    # -- the form -------------------------------------------------------------------

    @property
    def demo(self) -> Demo:
        """The demo on show, in the folder it runs in."""
        return self.demos[max(self.picker.currentIndex(), 0)]

    def _chose_demo(self) -> None:
        """Another demo picked: keep the options of the one left, and show this one's as
        the window kept them."""
        if self.shown is not None:
            self.remember(self.shown)
        self._build(self.demo)

    def _defaults(self) -> None:
        """Defaults: the options of the demo on show where a first run starts them; on
        closing, the window forgets those it kept."""
        self._build(self.demo, kept=False)

    def _kept_over(
        self, demo: Demo, start: argparse.Namespace, kept: dict[str, Any]
    ) -> argparse.Namespace:
        """``start`` with the options ``kept`` for ``demo`` (:func:`kept_options`),
        leaving out listeners and walkers the scene no longer has, a tick list kept with
        none of its names left (back to the scene's own), and HRTFs where the format
        kept for the scene is not binaural, which takes none."""
        args = argparse.Namespace(**vars(start))
        for name, value in kept.items():
            setattr(args, name, value)
        if getattr(args, "hrtf", None) and getattr(args, "format", "binaural") != "binaural":
            args.hrtf = None
        walkers = [name for _kind, name, _on in self.movement.walkers]
        for option in demo.options:
            value = getattr(args, option.name, None)
            if value is None or option.kind not in ("listeners", "movers"):
                continue
            if option.kind == "movers":
                names = moving_names(value)
                known = walkers
            else:
                names = listener_names(value)
                known = self._listeners(demo, args) or names
            left = [name for name in names if name in known]
            if option.kind == "movers" and not names:
                left = [NO_MOVERS]  # nothing moves, as kept
            setattr(args, option.name, ",".join(left) if left else getattr(start, option.name))
        return args

    def _build(self, demo: Demo, kept: bool = True) -> None:
        """Build the form of ``demo``: its steps, and every option where a first run
        starts it, or with ``kept`` as the window kept it."""
        start = self.starts.get(demo.name) or starting_arguments(demo)
        #: The options kept for the demo, and the arguments the form opened with.
        self.kept_values = kept_options(demo, self.scene_key) if kept else {}
        args = self._kept_over(demo, start, self.kept_values) if kept else start
        self.shown, self.start_args, self.opened_args = demo, start, args
        self.about.setText(demo.description)
        self.step.blockSignals(True)
        self.step.clear()
        self.step.addItems([step.name for step in demo.steps] + ([] if demo.stepless else [ALL]))
        self.step.setCurrentText(demo.steps[0].name if demo.stepless else ALL)
        self.step.blockSignals(False)
        # A script is one step, and moves nothing.
        self._steps.setRowVisible(self.step, not demo.stepless)
        self.about_step.setVisible(not demo.stepless)
        self.movement.setVisible(self._moves and not demo.stepless)

        form, more = QWidget(), QWidget()
        rows, more_rows = QFormLayout(form), QFormLayout(more)
        for layout in (rows, more_rows):
            layout.setContentsMargins(0, 0, 0, 0)
        self.fields = {}
        self._rows: dict[str, QFormLayout] = {}
        self._show_movement(demo, args)
        reads = reads_scene(demo)
        for option in demo.options:
            if option.kind == "movers":  # the Movement box's ticks
                self.fields[option.name] = self.movement
                continue
            tip = option.description
            widget = field_for(option, tip)
            show_value(option, widget, getattr(args, option.name, option.default))
            if isinstance(widget, (PathField, InputsField, ListenersField)):
                widget.changed.connect(self._refresh)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._refresh)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._refresh)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._refresh)
            else:
                widget.valueChanged.connect(self._refresh)
            label = QLabel(_(option.label))
            label.setBuddy(widget)
            label.setToolTip(f"{tip} ({option.option})")
            form_rows = more_rows if option.advanced or option is reads else rows
            form_rows.addRow(label, widget)
            self.fields[option.name] = widget
            self._rows[option.name] = form_rows
        self._layout.replaceWidget(self.options, form)
        self._advanced_layout.replaceWidget(self.advanced_options, more)
        for old in (self.options, self.advanced_options):
            old.setParent(None)  # off the window at once, not when Qt deletes it
            old.deleteLater()
        self.options, self.advanced_options = form, more
        if self.isVisible():
            # Shown now, not when Qt next gets round to it, so that _fit measures them.
            form.show()
            more.show()
        self._refresh()
        if self.isVisible():  # another demo: as tall as its form, as wide as the window
            self._fit(self.width())

    def _show_advanced(self, shown: bool) -> None:
        """Advanced: the step, what fine-tunes the run, its files and the command."""
        self.more.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)
        self.advanced.setVisible(shown)
        if self.isVisible():
            self._fit(self.width())

    def _show_movement(self, demo: Demo, args: argparse.Namespace) -> None:
        """The Movement box for ``demo``: ticks to change for a demo with a ``movers``
        option, what it moves for one that moves its own walkers, none for one that moves
        nothing."""
        box = self.movement
        option = next((o for o in demo.options if o.kind == "movers"), None)
        names = None
        if option is None and demo.moving is not None:
            try:
                names = demo.moving(demo.resolved(args))
            except Exception:  # the line under the form says what is wrong
                names = []
        if not box.any_used and not names:
            note = _(
                "Nothing in this scene moves. Edit movement… gives a source or a listener a "
                "walk along keyframes."
            )
            box.set_mode(False, [], note)
        elif option is not None:
            names = moving_names(getattr(args, option.name, None))
            note = _(
                "Ticked sources and listeners move along their walks while the piece plays; "
                "the others stand where they start. The ticks start as Moves sets them in "
                "the scene, and change this run only ({option})."
            ).format(option=option.option)
            box.set_mode(True, box.own if names is None else names, note)
        elif names is not None:
            note = _(
                "This demo moves what is ticked; Moves in the scene turns a walk on or off, "
                "and Edit movement… changes it."
            )
            box.set_mode(False, names, note)
        else:
            note = _(
                "This demo moves nothing: every source and listener stands where it starts. "
                "This scene plays the walks."
            )
            box.set_mode(False, [], note)

    def _edit_movement(self, kind: str, name: str) -> None:
        self.edit_request = (kind, name)
        self.reject()

    def _refresh(self) -> None:
        """The step's help, the command line and what is wrong with the options."""
        demo, step = self.demo, self.chosen_step()
        if step == ALL:
            self.about_step.setText(
                _("all: every step in order, leaving out what is done already.")
            )
        else:
            self.about_step.setText(f"{step}: {demo.step(step).help}.")
        args = self.arguments()
        for widget in self.fields.values():
            if isinstance(widget, InputsField) and widget.option.kind == "inputs":
                widget.set_sources(self._plays(demo, args))
                args = self.arguments()
            elif isinstance(widget, ListenersField):
                widget.set_listeners(self._listeners(demo, args))
                args = self.arguments()
        self.command.setText(demo.command(step, args))
        # An option that matters only with another's values (the police car's --siren
        # with --vehicle police_car) shows only then.
        values, moved = vars(args), False
        for name, form_rows in self._rows.items():
            widget, shown = self.fields[name], demo.option(name).applies(values)
            if isinstance(widget, ListenersField):  # none to tick where the scene cannot tell
                shown = shown and bool(widget.names)
            if form_rows.isRowVisible(widget) != shown:
                form_rows.setRowVisible(widget, shown)
                moved = True
        if moved and self.isVisible():
            self._fit(self.width())
        if demo.moving is not None and all(o.kind != "movers" for o in demo.options):
            self._show_movement(demo, args)  # what it moves may follow an option (--vehicle)
        self.folder.setText(_("Runs in {folder}").format(folder=demo.folder))
        self.summary.setText(self._summary(demo, args) or "")
        self.summary.setVisible(bool(self.summary.text()))
        what = self._downloads(demo, step, args)
        self.downloads.setText(_("This run downloads {what}.").format(what=what) if what else "")
        self.downloads.setVisible(bool(what))
        problem = demo.check(args)
        if self.unsaved:
            problem = _("save the scene first (File > Save): the demo reads it from its file")
        self.problem.setText(_("Cannot run: {reason}").format(reason=problem) if problem else "")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(problem is None)
        kept = ", ".join(self._kept_labels(demo, args))
        self.kept.setText(
            _(
                "Changed from the defaults and kept for next time: {options}. Defaults puts "
                "them back."
            ).format(options=kept)
            if kept
            else ""
        )
        self.kept.setVisible(bool(kept))

    def _changed(self, demo: Demo, args: argparse.Namespace) -> dict[str, Any]:
        """The options of ``demo`` the window keeps (:func:`keeps`) that ``args`` gives
        otherwise than where the form started them, by name, as the settings keep them;
        an option kept for the demo, whatever the scene, and left as the form opened it
        stays as kept, even where it is this scene's start or held back for it, as HRTFs
        are beside a format that takes none, emptied or not. A file left empty is not
        kept: a run then takes the scene's own, if any, as at its start."""
        changed = {}
        binaural = getattr(args, "format", "binaural") == "binaural"
        for option in demo.options:
            if not keeps(demo, option):
                continue
            value = saved_value(demo, option, getattr(args, option.name, None))
            opened = saved_value(demo, option, getattr(self.opened_args, option.name, None))
            held = option.name == "hrtf" and value is None and not binaural
            if (
                option.name in self.kept_values
                and not per_scene(demo, option)
                and (value == opened or held)
            ):
                changed[option.name] = saved_value(demo, option, self.kept_values[option.name])
            elif option.kind == "file" and value is None:
                continue
            elif value != saved_value(demo, option, getattr(self.start_args, option.name, None)):
                changed[option.name] = value
        return changed

    def _kept_labels(self, demo: Demo, args: argparse.Namespace) -> list[str]:
        """The labels of the options the window keeps (:func:`keeps`) that ``args``
        gives otherwise than where the form started them and that the next opening on
        the scene shows so, in the form's order: not an empty file, what names
        something in a scene without a scene file, a value the option cannot take back
        (a tempo that is no number), or no listener at all."""
        labels = []
        for option in demo.options:
            if not keeps(demo, option):
                continue
            value = saved_value(demo, option, getattr(args, option.name, None))
            if value == saved_value(demo, option, getattr(self.start_args, option.name, None)):
                continue
            if (option.kind == "file" and value is None) or (
                per_scene(demo, option) and self.scene_key is None
            ):
                continue
            if restored(option, value) is NOT_TAKEN or (option.kind == "listeners" and not value):
                continue
            labels.append(_("Movement") if option.kind == "movers" else _(option.label))
        return labels

    # -- what the window keeps ------------------------------------------------------

    def remember(self, demo: Demo, picked: bool = False) -> None:
        """Keep the options of ``demo``, the demo the form shows, changed from where it
        started them (:func:`keep_options`), and with ``picked`` that it was on show."""
        changed = self._changed(demo, self.arguments())
        keep_options(demo, self.scene_key, changed, demo.name if picked else None)

    def done(self, result: int) -> None:
        """Closed (Run, Cancel or Edit movement…): the options and the demo on show kept
        for the next time the window opens on the scene."""
        if self.shown is not None:
            self.remember(self.shown, picked=True)
        super().done(result)

    @staticmethod
    def _summary(demo: Demo, args: argparse.Namespace) -> str | None:
        """What the run does, as the demo says it (:attr:`Demo.summary`); none when it
        says nothing or cannot tell (the line below says what is wrong)."""
        if demo.summary is None:
            return None
        try:
            return demo.summary(demo.resolved(args))
        except Exception:
            return None

    @staticmethod
    def _downloads(demo: Demo, step: str, args: argparse.Namespace) -> str | None:
        """What the run downloads, as its steps say (:attr:`Step.downloads`), leaving out
        the steps that are done; none when it downloads nothing or the options cannot
        tell (the line below says what is wrong)."""
        try:
            values = demo.resolved(args)
            steps = demo.steps if step == ALL else [demo.step(step)]
            found = [
                what
                for s in steps
                if s.downloads is not None
                and not (s.done is not None and s.done(values))
                and (what := s.downloads(values))
            ]
        except Exception:
            return None
        return "; ".join(found) or None

    @staticmethod
    def _listeners(demo: Demo, args: argparse.Namespace) -> list[str]:
        """The listeners ``demo`` can play with ``args``; none when it cannot tell (a
        scene that cannot be read: the line under the form says why)."""
        if demo.listeners is None:
            return []
        try:
            return list(demo.listeners(demo.resolved(args)))
        except (Exception, SystemExit):
            return []

    @staticmethod
    def _plays(demo: Demo, args: argparse.Namespace) -> dict[str, str]:
        """The sources ``demo`` feeds with ``args`` and what each plays; none when it
        cannot tell (a scene that cannot be read: the line under the form says why)."""
        if demo.plays is None:
            return {}
        try:
            return demo.plays(demo.resolved(args))
        except Exception:
            return {}

    # -- what OK asked for ----------------------------------------------------------

    def chosen_step(self) -> str:
        return self.step.currentText() or ALL

    def arguments(self) -> argparse.Namespace:
        """The options as the fields hold them, those that do not matter with the
        others (hidden, :attr:`Option.only`) at their defaults."""
        demo = self.shown or self.demo  # whose fields they are, while another is picked
        values = {name: value_of(demo.option(name), w) for name, w in self.fields.items()}
        for name in values:
            option = demo.option(name)
            if not option.applies(values):
                values[name] = option.default
        return argparse.Namespace(**values)

    def chosen(self) -> tuple[Demo, str, argparse.Namespace]:
        """The demo (in the folder it runs in), the step and the arguments."""
        return self.demo, self.chosen_step(), self.arguments()


__all__ = [
    "ALL",
    "DEMO_SETTING",
    "KEPT_SCENES",
    "SCENE_KINDS",
    "DemoDialog",
    "InputsField",
    "ListenersField",
    "MovementBox",
    "PathField",
    "demo_key",
    "demos_for",
    "field_for",
    "keep_options",
    "keeps",
    "kept_demo",
    "kept_options",
    "reads_scene",
    "restored",
    "saved_value",
    "scene_key",
    "show_value",
    "starting_arguments",
    "value_of",
    "walk_text",
    "walkers_of",
    "work_base",
    "writable",
]
