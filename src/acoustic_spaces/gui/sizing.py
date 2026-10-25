"""Window sizes from what a window holds and the screen it is on, so that nothing in it
starts cut off: a window opens as large as its contents ask, as far as its screen leaves
room, and what does not fit then scrolls."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QAbstractScrollArea, QFrame, QScrollArea, QStyle, QWidget

#: How many characters of its font a window of text and paths is wide, at least.
READING_CHARACTERS = 90


def screen_room(widget: QWidget) -> QSize:
    """The room the screen ``widget`` is on (or the main screen) leaves for a window: its
    available area, less its frame and title bar."""
    screen = widget.screen() or QGuiApplication.primaryScreen()
    area = screen.availableGeometry()
    frame = widget.style().pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight, None, widget)
    line = widget.fontMetrics().height()
    return QSize(area.width() - line, area.height() - frame - 2 * line)


def reading_width(widget: QWidget) -> int:
    """A width at which a window's wrapped text and its paths read well:
    :data:`READING_CHARACTERS` characters of its font."""
    return widget.fontMetrics().averageCharWidth() * READING_CHARACTERS


def fit(window: QWidget, wanted: QSize) -> None:
    """Resize ``window`` to ``wanted``, within the room its screen leaves."""
    window.resize(wanted.boundedTo(screen_room(window)))


def fit_width(window: QWidget, width: int) -> None:
    """Resize ``window`` to ``width`` (or its own size hint's when wider) and to the
    height its contents take at that width, within the room its screen leaves."""
    width = max(width, window.sizeHint().width())
    layout = window.layout()
    if layout is not None and layout.hasHeightForWidth():
        height = layout.totalHeightForWidth(width)
    else:
        height = window.sizeHint().height()
    fit(window, QSize(width, max(height, window.minimumSizeHint().height())))


def scrolled(body: QWidget) -> QScrollArea:
    """``body`` in a frameless scroll area: shown whole where there is room, scrolling
    where there is not, rather than squeezing what it holds."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(body)
    return area


def scroll_bar_width(area: QAbstractScrollArea) -> int:
    """The width the vertical scroll bar of ``area`` takes from what it shows: the bar's
    own size, as the area lays it out, or none where the bar lies over it. (The style's
    metric asked without the bar's font is not it: on macOS it scales from 72 dots per
    inch, not the screen's, and comes out 4 pixels short on a 96 one.)"""
    bar = area.verticalScrollBar()
    if bar.style().pixelMetric(QStyle.PixelMetric.PM_ScrollView_ScrollBarOverlap, None, bar):
        return 0
    return bar.sizeHint().width()


def scroll_chrome(area: QScrollArea) -> int:
    """The width the scroll area ``area`` takes round what it shows: its frame on both
    sides and a vertical scroll bar."""
    return 2 * area.frameWidth() + scroll_bar_width(area)


def body_size(area: QScrollArea, width: int) -> QSize:
    """The size the scroll area ``area`` needs to show its widget whole at ``width``
    (wrapped text taking as many lines as it does at that width), with room for a
    vertical scroll bar."""
    body = area.widget()
    inner = width - scroll_bar_width(area)
    layout = body.layout()
    if layout is not None and layout.hasHeightForWidth():
        height = layout.totalHeightForWidth(inner)
    else:
        height = body.sizeHint().height()
    return QSize(width, max(height, body.minimumSizeHint().height()))


__all__ = [
    "READING_CHARACTERS",
    "body_size",
    "fit",
    "fit_width",
    "reading_width",
    "screen_room",
    "scroll_bar_width",
    "scroll_chrome",
    "scrolled",
]
