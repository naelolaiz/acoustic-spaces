"""The scene as YAML text, coloured, to read and to edit: the same data Save writes,
with lists of numbers or names on one line. Edits in the forms show in the text at once;
edits to the text become the scene with Apply, after the checks every other edit goes
through, in one step that Undo takes back. A refused text stays as typed, with the lines
the reason points at marked in red."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFontDatabase,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPaintEvent,
    QPalette,
    QResizeEvent,
    QShortcut,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gui.document import Document, EditError
from acoustic_spaces.gui.levels import level_colour, rich_lines
from acoustic_spaces.i18n import _
from acoustic_spaces.scene import Scene
from acoustic_spaces.scene.model import load_yaml

#: Spaces that Tab puts in: YAML indents with spaces, never tabs.
INDENT = "  "
#: The colours of the parts of the text, on a light theme and on a dark one.
COLOURS = {
    "key": ("#0550ae", "#79c0ff"),
    "string": ("#116329", "#7ee787"),
    "number": ("#8250df", "#d2a8ff"),
    "constant": ("#953800", "#ffa657"),
    "comment": ("#6e7781", "#8b949e"),
    "punctuation": ("#57606a", "#8b949e"),
}
#: The kinds of span that :func:`yaml_spans` finds.
KINDS = tuple(COLOURS)

_NUMBER = re.compile(
    r"[-+]?(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][-+]?\d+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)|0x[0-9a-fA-F_]+|0o[0-7_]+"
)
# What PyYAML reads as true, false or null (YAML 1.1: on and off too).
_CONSTANT = re.compile(
    r"(?:true|True|TRUE|false|False|FALSE|yes|Yes|YES|no|No|NO|on|On|ON|off|Off|OFF"
    r"|null|Null|NULL|~)"
)
# A key in a block mapping: quoted, or plain up to the colon and the space after it.
_KEY = re.compile(
    r"""("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'"""
    r"""|(?:[^\s#'"\[\]{},:!&*|>%@`-]|-(?=\S))(?:[^:#\n]|:(?=\S)|(?<=\S)\#)*?)"""
    r"""\s*:(?=\s|$)"""
)
_FLOW = "[]{},"
# Inside a quoted scalar that runs on to the next line: which quote closes it.
_IN_QUOTE = {1: "'", 2: '"'}


def _closing(line: str, start: int, quote: str) -> int:
    """Where in ``line`` the quoted scalar from ``start`` (just past its opening quote)
    ends, past its closing quote; -1 when it runs on."""
    i = start
    while i < len(line):
        c = line[i]
        if quote == '"' and c == "\\":
            i += 2
            continue
        if c == quote:
            if quote == "'" and line[i + 1 : i + 2] == "'":  # '' is a quote inside
                i += 2
                continue
            return i + 1
        i += 1
    return -1


def _scalar_kind(text: str) -> str:
    if _NUMBER.fullmatch(text):
        return "number"
    if _CONSTANT.fullmatch(text):
        return "constant"
    return "string"


def yaml_spans(line: str, state: int = 0) -> tuple[list[tuple[int, int, str]], int]:
    """The coloured spans of one line of YAML, as (start, length, kind) with a kind of
    :data:`KINDS`, and the state the next line starts in: 0, or 1 and 2 inside a
    single- or double-quoted scalar that runs on to it (``state`` is this line's)."""
    spans: list[tuple[int, int, str]] = []
    pos = 0
    if state in _IN_QUOTE:
        end = _closing(line, 0, _IN_QUOTE[state])
        if end < 0:
            return [(0, len(line), "string")] if line else [], state
        spans.append((0, end, "string"))
        value, state = _value(line, end)
        return spans + value, state
    pos = len(line) - len(line.lstrip(" "))
    if line.startswith(("---", "...")) and line[3:4] in ("", " "):
        spans.append((0, 3, "punctuation"))
        return spans + _rest(line, 3), 0
    while line[pos : pos + 1] == "-" and line[pos + 1 : pos + 2] in ("", " "):
        spans.append((pos, 1, "punctuation"))
        pos += 1
        pos += len(line[pos:]) - len(line[pos:].lstrip(" "))
    key = _KEY.match(line, pos)
    if key is not None:
        spans.append((key.start(1), key.end(1) - key.start(1), "key"))
        colon = key.end() - 1
        spans.append((colon, 1, "punctuation"))
        pos = colon + 1
    value, state = _value(line, pos)
    return spans + value, state


def _rest(line: str, pos: int) -> list[tuple[int, int, str]]:
    """What may follow a value on its line: a comment."""
    tail = line[pos:]
    stripped = tail.lstrip()
    if stripped.startswith("#"):
        start = pos + len(tail) - len(stripped)
        return [(start, len(line) - start, "comment")]
    return []


def _value(line: str, pos: int) -> tuple[list[tuple[int, int, str]], int]:
    """The spans of a value from ``pos`` to the end of the line."""
    spans: list[tuple[int, int, str]] = []
    depth = 0  # inside brackets: commas and colons part the values there
    while pos < len(line):
        c = line[pos]
        if c == " ":
            pos += 1
        elif c == "#" and (pos == 0 or line[pos - 1] == " "):
            spans.append((pos, len(line) - pos, "comment"))
            break
        elif c in "'\"":
            end = _closing(line, pos + 1, c)
            if end < 0:
                spans.append((pos, len(line) - pos, "string"))
                return spans, 1 if c == "'" else 2
            spans.append((pos, end - pos, "string"))
            pos = end
        elif c in _FLOW or (depth and c == ":"):
            depth += 1 if c in "[{" else -1 if c in "]}" else 0
            spans.append((pos, 1, "punctuation"))
            pos += 1
        elif c in "&*!":  # an anchor, an alias or a tag
            end = pos + 1
            while end < len(line) and line[end] not in " ,[]{}":
                end += 1
            spans.append((pos, end - pos, "constant"))
            pos = end
        elif c in "|>" and not depth:  # a block scalar's lines follow
            spans.append((pos, 1, "punctuation"))
            return spans + _rest(line, pos + 1), 0
        else:
            end = pos
            stop = " ,[]{}" if depth else ""
            while end < len(line):
                ch = line[end]
                if ch in stop and (ch != " " or line[end + 1 : end + 2] in ("#", "", ",")):
                    break
                if ch == " " and line[end + 1 : end + 2] == "#":
                    break
                if depth and ch == ":" and line[end + 1 : end + 2] in (" ", ""):
                    break
                end += 1
            text = line[pos:end].rstrip()
            if text:
                spans.append((pos, len(text), _scalar_kind(text)))
            pos = max(end, pos + 1)
    return spans, 0


def next_indent(line: str) -> int:
    """How far in the line after ``line`` starts: as far as ``line``, as far as its
    key after a list item's dash (``- name: …``), and a step further after a key that
    opens a mapping or a list (``geometry:``)."""
    lead = pos = len(line) - len(line.lstrip(" "))
    while line[pos : pos + 1] == "-" and line[pos + 1 : pos + 2] in ("", " "):
        pos += 1
        pos += len(line[pos:]) - len(line[pos:].lstrip(" "))
    keyed = _KEY.match(line, pos) is not None
    indent = pos if pos > lead and keyed else lead
    code = line.split(" #")[0].rstrip()
    if keyed and code.endswith(":"):
        indent = pos + len(INDENT)
    return indent


def line_of(text: str, loc: Sequence[str | int]) -> int | None:
    """The line (from 0) in YAML ``text`` of the deepest entry along ``loc``: keys and
    list positions, as the scene schema's errors give them; parts that name no entry
    (the schema's names for the kinds of an entry) are passed over. None when the text
    is no YAML, or has none of them."""
    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return None
    line = None
    for part in loc:
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if isinstance(key, yaml.ScalarNode) and key.value == str(part):
                    line, node = key.start_mark.line, value
                    break
        elif isinstance(node, yaml.SequenceNode) and isinstance(part, int):
            if 0 <= part < len(node.value):
                node = node.value[part]
                line = node.start_mark.line
    return line


def _yaml_error(exc: yaml.YAMLError) -> tuple[str, list[int]]:
    """What is wrong with YAML that does not read, and the lines (from 0) to mark: where
    PyYAML found the problem and where what it was reading began."""
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or str(exc)
    if mark is None:
        return _("not valid YAML: {problem}").format(problem=problem), []
    reason = _("not valid YAML at line {line}, column {column}: {problem}").format(
        line=mark.line + 1, column=mark.column + 1, problem=problem
    )
    lines = [mark.line]
    context, start = getattr(exc, "context", None), getattr(exc, "context_mark", None)
    if context and start is not None:
        # Translators: {context} is PyYAML's, such as "while parsing a flow sequence".
        reason += " " + _("({context} from line {line})").format(
            context=context, line=start.line + 1
        )
        lines.append(start.line)
    return reason, lines


def item_path(kind: str, key: object) -> tuple[str | int, ...]:
    """Where an item the editor selects (as the scene tree names it) is in the scene's
    YAML: its keys and list positions; empty for the scene itself."""
    if kind in ("sources", "listeners") and isinstance(key, int):
        return (kind, key)
    if kind == "material" and isinstance(key, str):
        return ("materials", key)
    if kind == "scene" and key == "wind":
        return ("atmosphere",)
    if kind in ("geometry", "simulation", "outputs", "coverage"):
        return (kind,)
    return ()


class YamlHighlighter(QSyntaxHighlighter):
    """Colours YAML: keys, strings, numbers, true, false and null, comments and
    punctuation, in shades for the theme of ``widget``."""

    def __init__(self, document: QTextDocument, widget: QWidget) -> None:
        super().__init__(document)
        self._widget = widget
        self.formats: dict[str, QTextCharFormat] = {}
        self.restyle()

    def restyle(self) -> None:
        """Take the colours for the widget's theme (light or dark)."""
        dark = self._widget.palette().color(QPalette.ColorRole.WindowText).lightness() > 128
        self.formats = {}
        for kind, colours in COLOURS.items():
            style = QTextCharFormat()
            style.setForeground(QColor(colours[dark]))
            if kind == "comment":
                style.setFontItalic(True)
            self.formats[kind] = style
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt's name)
        state = max(self.previousBlockState(), 0)
        spans, after = yaml_spans(text, state)
        for start, length, kind in spans:
            self.setFormat(start, length, self.formats[kind])
        self.setCurrentBlockState(after)


class _LineNumbers(QWidget):
    """The numbers of the lines beside the text."""

    def __init__(self, edit: YamlEdit) -> None:
        super().__init__(edit)
        self.edit = edit

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt's name)
        return QSize(self.edit.numbers_width(), 0)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt's name)
        self.edit.paint_numbers(self, event)


class YamlEdit(QPlainTextEdit):
    """A text field for YAML: monospaced, unwrapped, with line numbers; Tab puts in
    spaces and a new line starts where the one before did (further in after a key that
    opens a mapping)."""

    def __init__(self) -> None:
        super().__init__()
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.highlighter = YamlHighlighter(self.document(), self)
        self.numbers = _LineNumbers(self)
        self.blockCountChanged.connect(self._margin)
        self.updateRequest.connect(self._scrolled)
        self._margin()

    @property
    def marked(self) -> list[int]:
        """The lines (from 0) marked in red, where they are now that the text may have
        changed around them."""
        return sorted({s.cursor.blockNumber() for s in self.extraSelections()})

    def numbers_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 8 + self.fontMetrics().horizontalAdvance("9") * max(digits, 3)

    def _margin(self, *args: object) -> None:
        self.setViewportMargins(self.numbers_width(), 0, 0, 0)

    def _scrolled(self, rect: QRect, dy: int) -> None:
        if dy:
            self.numbers.scroll(0, dy)
        else:
            self.numbers.update(0, rect.y(), self.numbers.width(), rect.height())

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt's name)
        super().resizeEvent(event)
        inner = self.contentsRect()
        self.numbers.setGeometry(
            QRect(inner.left(), inner.top(), self.numbers_width(), inner.height())
        )

    def paint_numbers(self, widget: QWidget, event: QPaintEvent) -> None:
        painter = QPainter(widget)
        palette = self.palette()
        painter.fillRect(event.rect(), palette.color(QPalette.ColorRole.AlternateBase))
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        height = self.fontMetrics().height()
        error = level_colour(self, "error") or palette.color(QPalette.ColorRole.Text)
        plain = palette.color(QPalette.ColorRole.PlaceholderText)
        marked = set(self.marked)
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible():
                number = block.blockNumber()
                painter.setPen(error if number in marked else plain)
                painter.drawText(
                    0,
                    top,
                    widget.width() - 4,
                    height,
                    Qt.AlignmentFlag.AlignRight,
                    str(number + 1),
                )
            top += round(self.blockBoundingRect(block).height())
            block = block.next()
        painter.end()

    def mark(self, lines: Sequence[int]) -> None:
        """Mark ``lines`` (from 0) in red, what a refusal points at, until the next
        call; the marks move with the text as it is edited. None clears them."""
        shade = QColor(level_colour(self, "error") or QColor("red"))
        shade.setAlpha(60)
        selections = []
        for line in sorted(set(lines)):
            block = self.document().findBlockByNumber(line)
            if not block.isValid():
                continue
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(shade)
            selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            selection.cursor = QTextCursor(block)
            selections.append(selection)
        self.setExtraSelections(selections)
        self.numbers.update()

    def go_to(self, line: int, *, top: bool = False) -> None:
        """Put the cursor at the start of ``line`` (from 0) and show it (at the top)."""
        block = self.document().findBlockByNumber(line)
        if not block.isValid():
            return
        self.setTextCursor(QTextCursor(block))
        if top:
            self.verticalScrollBar().setValue(line)
        self.ensureCursorVisible()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt's name)
        key, modifiers = event.key(), event.modifiers()
        plain = modifiers in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier)
        if key == Qt.Key.Key_Tab and plain:
            self.insertPlainText(INDENT)
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and plain:
            cursor = self.textCursor()
            self.insertPlainText(
                "\n" + " " * next_indent(cursor.block().text()[: cursor.positionInBlock()])
            )
            self.ensureCursorVisible()
            return
        super().keyPressEvent(event)


class YamlPanel(QWidget):
    """The scene's YAML, with Apply, Revert and Find above it and what the text and the
    scene's checks say below it."""

    #: An edit refused, with the reason, for the messages.
    failed = Signal(str)

    def __init__(self, document: Callable[[], Document]) -> None:
        super().__init__()
        self._document = document
        #: The text of the scene last shown, that scene, and the folder the files it
        #: names are written relative to.
        self._shown = ""
        self._scene: Scene | None = None
        self._folder: Path | None = None
        #: What the last Apply said when it was refused: (text, level) lines.
        self._refusal: list[tuple[str, str]] = []
        #: Whether the panel is on show (its tab in front): only then does the text
        #: follow every change of the scene; otherwise it catches up when it shows.
        self.on_show = False
        #: The room last checked and what its checks said, for the lines below the text.
        self._checked: tuple[object, list[tuple[str, str]]] | None = None
        self.view = YamlEdit()
        self.view.setToolTip(
            _(
                "The scene's settings as YAML, the data Save writes: change them here and "
                "Apply (Ctrl+Return), or in the forms, and the text follows. Comments are "
                "not kept."
            )
        )
        self.apply_button = QPushButton(_("Apply"))
        self.apply_button.setToolTip(
            _(
                "Make the text the scene, checked as every edit is, in one step that Undo "
                "takes back (Ctrl+Return); a refused text stays, with its lines marked in "
                "red"
            )
        )
        self.revert_button = QPushButton(_("Revert"))
        self.revert_button.setToolTip(_("Drop the edits to the text and show the scene as it is"))
        self.search = QLineEdit()
        self.search.setPlaceholderText(_("Find"))
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip(
            _(
                "Find words in the text (Ctrl+F): typing finds the first, Enter the next, "
                "from the top again after the last"
            )
        )
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bar = QHBoxLayout()
        bar.addWidget(self.apply_button)
        bar.addWidget(self.revert_button)
        bar.addStretch(1)
        bar.addWidget(self.search, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status)

        self.apply_button.clicked.connect(self.apply)
        self.revert_button.clicked.connect(self.revert)
        self.view.textChanged.connect(self._edited)
        self.search.textChanged.connect(lambda *_args: self.find_next(from_top=True))
        self.search.returnPressed.connect(self.find_next)
        context = Qt.ShortcutContext.WidgetWithChildrenShortcut
        for keys in (Qt.Modifier.CTRL | Qt.Key.Key_Return, Qt.Modifier.CTRL | Qt.Key.Key_Enter):
            QShortcut(QKeySequence(keys), self, self.apply, context=context)
        QShortcut(QKeySequence.StandardKey.Find, self, self.search.setFocus, context=context)
        self._buttons()

    @property
    def document(self) -> Document:
        return self._document()

    # -- following the scene ------------------------------------------------------------

    def edited(self) -> bool:
        """Whether the text holds edits not applied."""
        return self.view.toPlainText() != self._shown

    def in_step(self) -> bool:
        """Whether the text was last shown for the scene as it is, in its folder."""
        document = self.document
        return document.scene is self._scene and document.base_dir == self._folder

    def reset(self) -> None:
        """Another scene is open: show it, dropping the edits to the text."""
        self._scene = self._folder = None
        self._refusal = []
        self.view.mark([])
        self._shown = ""
        self.view.setPlainText("")
        if self.on_show:
            self.revert()
        else:
            self._buttons()

    def scene_changed(self) -> None:
        """The scene changed (an edit, undo, redo or load), or its folder (a save): the
        text follows, unless it holds edits, or when the panel shows."""
        if self.in_step():
            return
        if self.on_show and not self.edited():
            self.revert()
        else:
            self._status()

    def revert(self) -> None:
        """Show the scene as it is, dropping the edits to the text; the view stays at
        the same lines."""
        document = self.document
        text = document.scene.file_text(document.base_dir, readable=True)
        self._scene, self._folder = document.scene, document.base_dir
        self._refusal = []
        if text != self.view.toPlainText():
            bar = self.view.verticalScrollBar()
            top, line = bar.value(), self.view.textCursor().blockNumber()
            self._shown = text
            self.view.setPlainText(text)
            self.view.go_to(min(line, self.view.blockCount() - 1))
            bar.setValue(top)
        self._shown = text
        self.view.mark([])
        self._buttons()

    def show_item(self, kind: str, key: object = None) -> None:
        """Show the entry of the item the editor selected, at the top of the text."""
        if not self.on_show:
            return
        path = item_path(kind, key)
        if path:
            line = line_of(self.view.toPlainText(), path)
            if line is not None:
                self.view.go_to(line, top=True)

    def set_shown(self, shown: bool) -> None:
        """The panel came on show (its dock shown, or its tab brought to the front), or
        went: on show, the text catches up with the scene, unless it holds edits."""
        self.on_show = shown
        if shown and not self.in_step() and not self.edited():
            self.revert()
        else:
            self._status()

    # -- applying the text --------------------------------------------------------------

    def apply(self) -> bool:
        """Make the text the scene; True when it did (or there was nothing to change).
        A refused text stays as it is, its lines marked in red, the reason below it."""
        text = self.view.toPlainText()
        try:
            data = load_yaml(text)
        except yaml.YAMLError as exc:
            return self._refused(*_yaml_error(exc))
        if not isinstance(data, dict):
            return self._refused(
                _(
                    "the text holds no settings: write them as key: value lines, as a scene "
                    "file does"
                ),
                [],
            )
        if not self.edited() or (self.in_step() and data == load_yaml(self._shown)):
            self.revert()  # only the layout or comments changed
            return True

        def change(current: dict) -> None:
            current.clear()
            current.update(data)

        try:
            self.document.edit(_("edit scene YAML"), change)
        except EditError as exc:
            lines = []
            cause = exc.__cause__
            if isinstance(cause, ValidationError):
                for error in cause.errors():
                    line = line_of(text, error["loc"])
                    if line is not None:
                        lines.append(line)
            return self._refused(str(exc), lines)
        # The scene changed, so the text follows it (the same data, laid out again).
        self.revert()
        return True

    def _refused(self, reason: str, lines: list[int]) -> bool:
        self.view.mark(lines)
        self._refusal = [(_("Not applied: {reason}").format(reason=reason), "error")]
        if lines:
            self.view.go_to(min(lines))
        self._status()
        self.failed.emit(reason)
        return False

    # -- finding ------------------------------------------------------------------------

    def find_next(self, *, from_top: bool = False) -> bool:
        """Select the next place the Find words are (from the top, or past the last
        found); True when there is one."""
        words = self.search.text()
        if not words:
            return False
        if from_top:
            self.view.moveCursor(QTextCursor.MoveOperation.Start)
        if self.view.find(words):
            return True
        self.view.moveCursor(QTextCursor.MoveOperation.Start)
        return self.view.find(words)

    # -- what the panel says ------------------------------------------------------------

    def _edited(self) -> None:
        if not self.edited() and self._refusal:  # back to the scene's own text
            self._refusal = []
            self.view.mark([])
        else:
            self.view.numbers.update()  # marks that moved with the text
        self._buttons()

    def _buttons(self) -> None:
        edited = self.edited()
        self.apply_button.setEnabled(edited)
        self.revert_button.setEnabled(edited or not self.in_step())
        self._status()

    def _status(self) -> None:
        if not self.on_show and not self._refusal:
            return  # brought up to date when it shows
        lines = list(self._refusal)
        changed = not self.in_step()
        if not lines:
            if self.edited() and changed:
                lines.append(
                    (
                        _(
                            "Edited, and the scene has changed since: Apply makes the text "
                            "the scene, dropping those changes; Revert shows them"
                        ),
                        "warning",
                    )
                )
            elif self.edited():
                lines.append((_("Edited: Apply makes the text the scene"), "info"))
            else:
                lines.append((_("In step with the scene"), "info"))
        self.status.setText(rich_lines(self.status, lines + self._checks()))

    def _checks(self) -> list[tuple[str, str]]:
        """The scene's problems (red) and warnings (amber), as its form shows them."""
        room = self.document.room
        if self._checked is None or self._checked[0] is not room:
            checks = [
                (_("Problem: {problem}").format(problem=problem), "error")
                for problem in room.problems()
            ]
            checks += [
                (_("Warning: {warning}").format(warning=warning), "warning")
                for warning in room.warnings()
            ]
            self._checked = (room, checks)
        return self._checked[1]


__all__ = ["YamlEdit", "YamlHighlighter", "YamlPanel", "item_path", "line_of", "yaml_spans"]
