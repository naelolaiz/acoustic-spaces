"""The window behind Simulation > Run demo…: which demo of the open scene to run, which
of its steps, and its options at the defaults the command line uses.

Every scene has *This scene* (:mod:`acoustic_spaces.demos.scene_demo`): its sources
play the WAV files their ``signal`` names, and every listener is heard in turn in a
video. A scene of an example that comes with a demo (:mod:`acoustic_spaces.demos`), or
a copy of one, also has that demo, with its own recordings, reading the open scene. The
form is built from the demo's own options, so the editor and the command line run
exactly the same thing; the line at the bottom shows that command, to run again or to
put in a script. A demo whose sources can play files of the user's (``--input``) lists
them in a table, each with what it plays and the WAV file it plays instead.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.demos import Demo, Option
from acoustic_spaces.gui.inspector import int_spin, spin
from acoustic_spaces.i18n import _

#: The folder under the user's home the editor runs a demo in when the example itself
#: cannot be written.
HOME_FOLDER = "acoustic-spaces"
#: The step the window starts on: every step, leaving out what is done already.
ALL = "all"


def writable(folder: Path, name: str) -> Path:
    """``folder`` when it can be written, else one called ``name`` under the user's home
    (``~/acoustic-spaces/<name>``): where a demo writes when the scene lies among the
    examples of an installed copy."""
    if os.access(folder, os.W_OK):
        return folder
    return Path.home() / HOME_FOLDER / name


def work_base(demo: Demo) -> Path:
    """The folder the editor runs ``demo`` in: the example's own folder, where the
    command line runs it, or one of the same name under the user's home when the
    example cannot be written (the examples of an installed copy)."""
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
    it starts with: *This scene* first, then the demo of the example the scene belongs
    to (its folder's name, in the examples or in a copy of one), reading the open scene
    instead of the example's."""
    from acoustic_spaces.demos import demo_for
    from acoustic_spaces.demos.scene_demo import DEMO

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
    example = demo_for(path.parent.name) if path is not None else None
    if example is not None:
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
    comes from the example when the folder has no copy of it."""
    args = demo.resolved(demo.defaults())
    for option in demo.options:
        if option.kind != "file" or option.default is None:
            continue
        if not Path(getattr(args, option.name)).exists():
            beside = demo.script.parent / option.default
            if beside.exists():
                setattr(args, option.name, beside)
    return args


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
        if dialog.exec() and dialog.path is not None:
            self.edit.setText(str(dialog.path))

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
    instead, typed or chosen with Choose WAV…; Default empties it again."""

    #: A file was typed, chosen or taken back.
    changed = Signal()
    SOURCE, PLAYS, FILE = range(3)

    def __init__(self, option: Option, tip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.option = option
        self.inputs: dict[str, Path] = {}
        self.plays: dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 3)
        self.table.setToolTip(tip)
        heads = (
            (_("Source"), _("The source, as the scene names it")),
            (_("Plays"), _("What the source plays in the demo, unless a file is given")),
            (
                _("Instead"),
                _(
                    "The WAV file the source plays instead (--input SOURCE=WAV); empty: what "
                    "Plays says. The table's tip says how loud it plays"
                ),
            ),
        )
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
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            file = self.inputs.get(name)
            self.table.setItem(row, self.FILE, QTableWidgetItem("" if file is None else str(file)))
        self.table.resizeColumnToContents(self.SOURCE)
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


def field_for(option: Option, tip: str) -> QWidget:
    """The widget that asks for ``option``: a tick for a flag, a list for a choice, a
    spin box for a number, a path with a Browse… button for a file or a folder, a table
    of the sources for ``--input``, and a text field for the rest (empty where the demo
    takes none)."""
    if option.kind == "inputs":
        return InputsField(option, tip)
    if option.kind in ("file", "path", "folder"):
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
    if option.kind in ("int", "float") and option.default is not None:
        number = (
            int_spin(-1_000_000, 1_000_000) if option.kind == "int" else spin(-1e6, 1e6, 2, 0.5)
        )
        number.setValue(option.default)
        number.setToolTip(tip)
        return number
    edit = QLineEdit("" if option.default is None else str(option.default))
    edit.setToolTip(tip)
    return edit


def value_of(option: Option, widget: QWidget):
    """What the field holds, in the type the demo's step expects (None where it is
    empty: the demo is given no such option)."""
    if isinstance(widget, (PathField, InputsField)):
        return widget.value()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        text = widget.text().strip()
        return text or None
    value = widget.value()
    return int(value) if option.kind == "int" else float(value)


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
        the scene's; ``start`` names the demo to start on (by default the example's own
        demo when the scene has one, else *This scene*)."""
        super().__init__(parent)
        self.setWindowTitle(_("Run demo"))
        if demos is not None:
            pairs = [(d.at(work_base(d)), None) for d in demos]
        else:
            pairs = demos_for(scene)
            if start is None:
                start = pairs[-1][0].name
        self.demos = [demo for demo, _args in pairs]
        self.starts = {demo.name: args for demo, args in pairs if args is not None}
        self.unsaved = unsaved or (demos is None and scene is None)
        self.fields: dict[str, QWidget] = {}
        layout = self._layout = QVBoxLayout(self)
        intro = QLabel(
            _(
                "A demo plays sound through the open scene, mixes what every listener hears "
                "and puts a video together. This scene plays the WAV file each source's Plays "
                "names; an example's own demo downloads the recordings it plays and can take "
                "a while. The messages say what it is doing, and Cancel stops it."
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
                "The demo to run: This scene, with the sounds its sources play, or the "
                "example's own demo when the scene is an example's or a copy of one"
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
        for text, widget in ((_("Demo"), self.picker), (_("Step"), self.step)):
            label = QLabel(text)
            label.setBuddy(widget)
            label.setToolTip(widget.toolTip())
            head.addRow(label, widget)
        layout.addLayout(head)
        self.about = QLabel()
        self.about.setWordWrap(True)
        layout.addWidget(self.about)
        self.about_step = QLabel()
        self.about_step.setWordWrap(True)
        layout.addWidget(self.about_step)

        self.options = QWidget()
        layout.addWidget(self.options)

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
        layout.addLayout(row)
        layout.addWidget(self.folder)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        layout.addWidget(self.problem)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(_("Run"))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setToolTip(
            _("Run the step in the background; the messages say how it goes")
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setToolTip(
            _("Close without running anything")
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.picker.currentIndexChanged.connect(self._chose_demo)
        self.step.currentIndexChanged.connect(self._refresh)
        self._chose_demo()

    # -- the form -------------------------------------------------------------------

    @property
    def demo(self) -> Demo:
        """The demo on show, in the folder it runs in."""
        return self.demos[max(self.picker.currentIndex(), 0)]

    def _chose_demo(self) -> None:
        """Build the form of the demo on show: its steps and every option at its
        default."""
        demo = self.demo
        self.about.setText(demo.description)
        self.step.blockSignals(True)
        self.step.clear()
        self.step.addItems([step.name for step in demo.steps] + [ALL])
        self.step.setCurrentText(ALL)
        self.step.blockSignals(False)

        form = QWidget()
        rows = QFormLayout(form)
        rows.setContentsMargins(0, 0, 0, 0)
        self.fields = {}
        args = self.starts.get(demo.name) or starting_arguments(demo)
        for option in demo.options:
            tip = option.description
            widget = field_for(option, tip)
            if isinstance(widget, (PathField, InputsField)):
                widget.set_value(getattr(args, option.name))
                widget.changed.connect(self._refresh)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._refresh)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._refresh)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._refresh)
            else:
                widget.valueChanged.connect(self._refresh)
            label = QLabel(option.option)
            label.setBuddy(widget)
            label.setToolTip(tip)
            rows.addRow(label, widget)
            self.fields[option.name] = widget
        self._layout.replaceWidget(self.options, form)
        self.options.setParent(None)  # off the window at once, not when Qt deletes it
        self.options.deleteLater()
        self.options = form
        self._refresh()

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
            if isinstance(widget, InputsField):
                widget.set_sources(self._plays(demo, args))
                args = self.arguments()
        self.command.setText(demo.command(step, args))
        self.folder.setText(_("Runs in {folder}").format(folder=demo.folder))
        problem = demo.check(args)
        if self.unsaved:
            problem = _("save the scene first (File > Save): the demo reads it from its file")
        self.problem.setText(_("Cannot run: {reason}").format(reason=problem) if problem else "")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(problem is None)

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
        """The options as the fields hold them."""
        demo = self.demo
        values = {name: value_of(demo.option(name), w) for name, w in self.fields.items()}
        return argparse.Namespace(**values)

    def chosen(self) -> tuple[Demo, str, argparse.Namespace]:
        """The demo (in the folder it runs in), the step and the arguments."""
        return self.demo, self.chosen_step(), self.arguments()


__all__ = [
    "ALL",
    "DemoDialog",
    "InputsField",
    "PathField",
    "demos_for",
    "field_for",
    "reads_scene",
    "starting_arguments",
    "value_of",
    "work_base",
    "writable",
]
