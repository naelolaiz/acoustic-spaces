"""The editor's messages: what it did and what went wrong, a warning in amber and an
error in red, every other message on a shaded row so that one ends where the next
begins. Ticks show or hide each level, a list and a filter keep only the messages
about one source, listener or material or with some words, and Clear removes the
messages on show."""

from __future__ import annotations

import re

from PySide6.QtGui import QColor, QPalette, QTextBlockFormat, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gui.levels import LEVELS, level_colour
from acoustic_spaces.gui.tips import label_for, share_tips
from acoustic_spaces.i18n import _

#: How many messages the panel keeps; past it, the oldest go a few hundred at a time.
LIMIT = 2000
_DROP = 200


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


class MessagesPanel(QWidget):
    """The messages, newest last, with the choices of which to show above them."""

    def __init__(self) -> None:
        super().__init__()
        #: Every message kept: (text, level), oldest first.
        self.entries: list[tuple[str, str]] = []
        #: Each scene item's name and how many messages name it.
        self._names: dict[str, int] = {}
        self._patterns: dict[str, re.Pattern] = {}
        self._shown = 0  # messages on show, for the shading of the next one
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
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

    def add(self, text: str, level: str = "info") -> None:
        """Keep ``text`` (``level`` one of :data:`~acoustic_spaces.gui.levels.LEVELS`)
        and show it when the choices let it through."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}, not {level!r}")
        self.entries.append((text, level))
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
            self._insert(text, level)
            if following:  # as appending does: stay at the end unless scrolled back
                bar.setValue(bar.maximum())

    def clear_shown(self) -> None:
        """Remove the messages on show, keeping those the choices hide."""
        self.entries = [entry for entry in self.entries if not self._passes(*entry)]
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
        for _text, level in self.entries:
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
        for text, _level in self.entries:
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
        for text, level in self.entries:
            if self._passes(text, level):
                self._insert(text, level)
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _insert(self, text: str, level: str) -> None:
        style = QTextCharFormat()
        colour = level_colour(self.view, level)
        if colour is not None:
            style.setForeground(colour)
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
        cursor.insertText(text, style)  # its lines take the same shade
        self._shown += 1


__all__ = ["LIMIT", "MessagesPanel", "row_shade"]
