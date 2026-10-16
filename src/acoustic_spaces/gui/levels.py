"""Warnings in amber and errors in red, in the editor's messages and forms, in shades
that read on light and dark themes alike."""

from __future__ import annotations

from html import escape

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

#: How much a message matters: info (the theme's text colour), warning or error.
LEVELS = ("info", "warning", "error")
#: Each level's colour on a light theme and on a dark one.
_COLOURS = {"warning": ("#9a6700", "#e3b341"), "error": ("#c62828", "#ff7b72")}


def level_colour(widget: QWidget, level: str) -> QColor | None:
    """The colour of ``level`` on ``widget``'s theme; None for info, which keeps the
    theme's own text colour."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}, not {level!r}")
    if level == "info":
        return None
    dark = widget.palette().color(QPalette.ColorRole.WindowText).lightness() > 128
    return QColor(_COLOURS[level][dark])


def rich_lines(widget: QWidget, lines: list[tuple[str, str]]) -> str:
    """``lines`` of (text, level) as rich text for a label on ``widget``."""
    shown = []
    for text, level in lines:
        colour = level_colour(widget, level)
        text = escape(text)
        shown.append(
            text if colour is None else f'<span style="color:{colour.name()}">{text}</span>'
        )
    return "<br>".join(shown)
