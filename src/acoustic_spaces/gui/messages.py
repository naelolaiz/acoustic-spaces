"""The editor's messages: what it did and what went wrong, a warning in amber and an
error in red, every other message on a shaded row so that one ends where the next
begins. Ticks show or hide each level, a list and a filter keep only the messages
about one source, listener or material or with some words, and Clear removes the
messages on show. A path in a message to a file or folder that is there opens with a
click in this computer's default app, and its right-click menu copies it."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, Qt, QUrl
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QGuiApplication,
    QPalette,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gui.levels import LEVELS, level_colour
from acoustic_spaces.gui.tips import label_for, share_tips
from acoustic_spaces.i18n import _

#: How many messages the panel keeps; past it, the oldest go a few hundred at a time.
LIMIT = 2000
_DROP = 200


#: The files a click on their path opens: what the simulator and its demos write and
#: what plays or shows it (sound, video, pictures, text and tables, scenes, 3D models,
#: PDFs). Nothing that could run (a script, a program) is among them.
OPENS = frozenset(
    {
        *(".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aif", ".aiff"),
        *(".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"),
        *(".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"),
        *(".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".md", ".log", ".srt", ".vtt"),
        *(".obj", ".stl", ".ply", ".glb", ".gltf", ".dxf", ".ifc", ".pdf"),
    }
)
#: How many words, joined by their spaces, a path may span (``C:\Users\Ana María\…``).
_WORDS = 6
#: What may come before a path in a sentence, or after it.
_LEADING = "\"'([<"
_TRAILING = ".,:;!?\"')]>"


def _opens(path: str) -> bool:
    """True when a click may open ``path``: a file of a kind :data:`OPENS` lists, or a
    folder (in the file manager) whose name has no extension, as an app on a Mac (a
    folder named ``NAME.app``) has."""
    try:
        if os.path.isdir(path):
            return not Path(path).suffix
        return Path(path).suffix.lower() in OPENS and os.path.isfile(path)
    except (OSError, ValueError):  # a name too long, or with a null in it
        return False


def paths_in(text: str, folder: str | Path | None = None) -> list[tuple[int, int, str]]:
    """The paths in ``text`` that a click may open (:func:`_opens`), each as its
    ``(start, end)`` in ``text`` and the path in full. A path relative to ``folder`` (the
    folder a demo runs in, for what it prints) counts when it names a file; with no
    ``folder``, only paths in full count. Spaces may be part of a path, up to a few
    words of one, the longest that is there winning; a sentence's punctuation around a
    path is no part of it."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    found: list[tuple[int, int, str]] = []
    index = 0
    while index < len(words):
        start = words[index][0]
        while start < words[index][1] and text[start] in _LEADING:
            start += 1
        best = None
        for last in range(index, min(index + _WORDS, len(words))):
            span = text[start : words[last][1]]
            for candidate in dict.fromkeys((span, span.rstrip(_TRAILING))):
                full = _full(candidate, folder)
                if full is not None and _opens(full):
                    best = (start, start + len(candidate), full, last)
                    break
        if best is None:
            index += 1
            continue
        found.append(best[:3])
        index = best[3] + 1
    return found


def _full(candidate: str, folder: str | Path | None) -> str | None:
    """``candidate`` in full when it may be a path to look for, else None: a path in
    full below its root, or a file's path relative to ``folder``."""
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute():
        return str(path) if len(path.parts) > 1 else None
    if folder is None or path.suffix.lower() not in OPENS:
        return None
    return os.path.abspath(Path(folder) / path)


def _default_app(path: str) -> bool:
    """Open ``path`` with this computer's default app for it (a folder in the file
    manager); False when none did."""
    return QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _pattern(name: str) -> re.Pattern:
    """``name`` as a whole word: not part of a longer name such as main_left for main."""
    return re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")


def row_shade(widget: QWidget) -> QColor:
    """The background of every other message on ``widget``'s theme: the theme's
    alternate rows, or a touch of its text colour over the base where those are too
    alike to tell apart."""
    palette = widget.palette()
    base = palette.color(QPalette.ColorRole.Base)
    shade = palette.color(QPalette.ColorRole.AlternateBase)
    if abs(shade.lightness() - base.lightness()) < 12:
        text = palette.color(QPalette.ColorRole.Text)
        mix = 0.08
        shade = QColor.fromRgbF(
            *(
                b + (t - b) * mix
                for b, t in zip(base.getRgbF()[:3], text.getRgbF()[:3], strict=True)
            )
        )
    return shade


class MessageView(QPlainTextEdit):
    """The messages' text, read only: a path in it that opens (:func:`paths_in`) is
    underlined, a click opens it, and its right-click menu also copies it."""

    def __init__(self, panel: MessagesPanel) -> None:
        super().__init__()
        self.panel = panel
        self.setReadOnly(True)
        self.viewport().setMouseTracking(True)  # a hand over a path, as on a link
        self._text_cursor = self.viewport().cursor()
        self._pressed = ""

    def path_at(self, point: QPoint) -> str:
        """The path under ``point`` of the viewport, or ""."""
        return self.anchorAt(point)

    def menu_for(self, point: QPoint) -> QMenu:
        """The right-click menu at ``point`` of the viewport: Open and Copy path first
        when it is on a path, then the usual Copy and Select All."""
        menu = self.createStandardContextMenu()
        path = self.path_at(point)
        if path:
            first = menu.actions()[0] if menu.actions() else None
            opening = menu.addAction(_("Open"))
            opening.setToolTip(_("Open {path} with this computer's default app").format(path=path))
            opening.triggered.connect(lambda: self.panel.open_path(path))
            copying = menu.addAction(_("Copy path"))
            copying.setToolTip(_("Copy {path} for pasting elsewhere").format(path=path))
            copying.triggered.connect(lambda: QGuiApplication.clipboard().setText(path))
            if first is not None:
                menu.insertActions(first, [opening, copying])
                menu.insertSeparator(first)
            menu.setToolTipsVisible(True)
        return menu

    def contextMenuEvent(self, event) -> None:
        menu = self.menu_for(event.pos())  # the viewport's, as the event comes from it
        menu.exec(event.globalPos())
        menu.deleteLater()

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        on_path = bool(self.path_at(event.position().toPoint()))
        self.viewport().setCursor(
            Qt.CursorShape.PointingHandCursor if on_path else self._text_cursor
        )

    def mousePressEvent(self, event) -> None:
        self._pressed = ""
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = self.path_at(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        pressed, self._pressed = self._pressed, ""
        if (
            pressed
            and event.button() == Qt.MouseButton.LeftButton
            and self.path_at(event.position().toPoint()) == pressed
            and not self.textCursor().hasSelection()  # a click, not a selection
        ):
            self.panel.open_path(pressed)

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            path = self.path_at(event.pos())
            if path:
                QToolTip.showText(
                    event.globalPos(),
                    _(
                        "Click to open {path} with this computer's default app; "
                        "right-click to copy the path"
                    ).format(path=path),
                    self.viewport(),
                )
            else:
                QToolTip.hideText()
            return True
        return super().viewportEvent(event)


class MessagesPanel(QWidget):
    """The messages, newest last, with the choices of which to show above them."""

    def __init__(self) -> None:
        super().__init__()
        #: Every message kept: (text, level, the paths in it that open as
        #: :func:`paths_in` gives them), oldest first.
        self.entries: list[tuple[str, str, list[tuple[int, int, str]]]] = []
        #: Each scene item's name and how many messages name it.
        self._names: dict[str, int] = {}
        self._patterns: dict[str, re.Pattern] = {}
        self._shown = 0  # messages on show, for the shading of the next one
        self.view = MessageView(self)
        self.levels = {
            "info": QCheckBox(),
            "warning": QCheckBox(),
            "error": QCheckBox(),
        }
        tips = {
            "info": _("Show the messages that only say what happened, such as a run's steps"),
            "warning": _("Show the warnings (in amber): results to check, not failures"),
            "error": _("Show the errors (in red): what could not be done, and why"),
        }
        for level, box in self.levels.items():
            box.setChecked(True)
            box.setToolTip(tips[level])
            colour = level_colour(self, level)
            if colour is not None:
                palette = box.palette()
                palette.setColor(QPalette.ColorRole.WindowText, colour)
                palette.setColor(QPalette.ColorRole.ButtonText, colour)
                box.setPalette(palette)
            box.toggled.connect(self._render)
        self.item = QComboBox()
        self.item.setToolTip(
            _(
                "Show only the messages about one source, listener or material; the list "
                "holds those that messages name, with how many do"
            )
        )
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(_("Filter"))
        self.filter.setClearButtonEnabled(True)
        self.filter.setToolTip(
            _("Show only the messages holding these words (capitals or not); empty shows all")
        )
        self.clear_button = QPushButton(_("Clear"))
        self.clear_button.setToolTip(
            _("Remove the messages on show; those that the choices here hide stay")
        )
        bar = QHBoxLayout()
        for box in self.levels.values():
            bar.addWidget(box)
        bar.addWidget(label_for(_("Item"), self.item))
        bar.addWidget(self.item, 1)
        bar.addWidget(self.filter, 1)
        bar.addWidget(self.clear_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)

        self.item.currentIndexChanged.connect(self._render)
        self.filter.textChanged.connect(self._render)
        self.clear_button.clicked.connect(self.clear_shown)
        share_tips(self)
        self._items()
        self._counts()

    # -- adding and removing ----------------------------------------------------------

    def add(self, text: str, level: str = "info", *, folder: str | Path | None = None) -> None:
        """Keep ``text`` (``level`` one of :data:`~acoustic_spaces.gui.levels.LEVELS`)
        and show it when the choices let it through. Its paths to files and folders that
        are there now open with a click, those relative to ``folder`` counted from there
        (:func:`paths_in`)."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}, not {level!r}")
        links = paths_in(text, folder)
        self.entries.append((text, level, links))
        named = [name for name, pattern in self._patterns.items() if pattern.search(text)]
        for name in named:
            self._names[name] += 1
        if len(self.entries) > LIMIT + _DROP:
            del self.entries[:_DROP]
            self._recount()
            self._render()
            return
        if named:
            self._items()
        self._counts()
        if self._passes(text, level):
            bar = self.view.verticalScrollBar()
            following = bar.value() == bar.maximum()
            self._insert(text, level, links)
            if following:  # as appending does: stay at the end unless scrolled back
                bar.setValue(bar.maximum())

    def open_path(self, path: str) -> bool:
        """Open ``path`` with this computer's default app (a folder in the file
        manager), as a click on it does; a message says when it is no longer there or
        when nothing opened it."""
        if not os.path.exists(path):
            self.add(_("Not opened: {path} is no longer there").format(path=path), "warning")
            return False
        if not _opens(path):  # never a script or a program, whatever asks
            self.add(
                _(
                    "Not opened: only sound, video, pictures, text, tables, scenes, 3D "
                    "models, PDFs and folders open from here, not {path}"
                ).format(path=path),
                "warning",
            )
            return False
        if not _default_app(path):
            self.add(_("No app on this computer opened {path}").format(path=path), "error")
            return False
        return True

    def clear_shown(self) -> None:
        """Remove the messages on show, keeping those the choices hide."""
        self.entries = [entry for entry in self.entries if not self._passes(*entry[:2])]
        self._recount()
        self._render()

    def set_names(self, names) -> None:
        """The names of the scene's sources, listeners and materials, for the list of
        items: it holds those that messages name."""
        names = sorted(set(names))
        if names == sorted(self._names):
            return
        self._patterns = {name: _pattern(name) for name in names}
        self._recount()
        self._render()

    # -- what is on show --------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """How many messages are kept at each level."""
        counts = dict.fromkeys(LEVELS, 0)
        for _text, level, _links in self.entries:
            counts[level] += 1
        return counts

    def _passes(self, text: str, level: str) -> bool:
        if not self.levels[level].isChecked():
            return False
        name = self.item.currentData()
        if name and not self._patterns[name].search(text):
            return False
        words = self.filter.text().strip().casefold()
        return not words or words in text.casefold()

    def _recount(self) -> None:
        self._names = dict.fromkeys(self._patterns, 0)
        for text, _level, _links in self.entries:
            for name, pattern in self._patterns.items():
                if pattern.search(text):
                    self._names[name] += 1
        self._items()
        self._counts()

    def _counts(self) -> None:
        counts = self.counts()
        self.levels["info"].setText(_("Info ({count})").format(count=counts["info"]))
        self.levels["warning"].setText(_("Warnings ({count})").format(count=counts["warning"]))
        self.levels["error"].setText(_("Errors ({count})").format(count=counts["error"]))

    def _items(self) -> None:
        """List "All items" and each item a message names, keeping the one on show."""
        shown = self.item.currentData()
        self.item.blockSignals(True)
        self.item.clear()
        self.item.addItem(_("All items"), "")
        for name, count in sorted(self._names.items()):
            if count:
                self.item.addItem(f"{name} ({count})", name)
        # The callers show the messages again when no message names it any more.
        self.item.setCurrentIndex(max(self.item.findData(shown) if shown else 0, 0))
        self.item.blockSignals(False)

    def _render(self) -> None:
        """Show again the messages the choices let through, the newest at the bottom."""
        self.view.clear()
        self._shown = 0
        for text, level, links in self.entries:
            if self._passes(text, level):
                self._insert(text, level, links)
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _insert(self, text: str, level: str, links: Sequence[tuple[int, int, str]] = ()) -> None:
        style = QTextCharFormat()
        colour = level_colour(self.view, level)
        if colour is not None:
            style.setForeground(colour)
        link = QTextCharFormat(style)
        link.setAnchor(True)
        link.setFontUnderline(True)
        if colour is None:  # a warning or an error keeps its colour, underlined
            link.setForeground(self.view.palette().color(QPalette.ColorRole.Link))
        block = QTextBlockFormat()
        if self._shown % 2:
            block.setBackground(row_shade(self.view))
        document = self.view.document()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if document.isEmpty():
            cursor.setBlockFormat(block)
        else:
            cursor.insertBlock(block, style)
        # Its lines take the same shade; its paths are links to themselves.
        at = 0
        for start, end, path in links:
            cursor.insertText(text[at:start], style)
            link.setAnchorHref(path)
            cursor.insertText(text[start:end], link)
            at = end
        cursor.insertText(text[at:], style)
        self._shown += 1


__all__ = ["LIMIT", "OPENS", "MessageView", "MessagesPanel", "paths_in", "row_shade"]
