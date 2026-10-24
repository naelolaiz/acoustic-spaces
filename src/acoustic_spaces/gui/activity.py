"""The status bar's sign that something runs in the background: a moving bar, what
runs, its latest step and the time so far, and a button that cancels it. Resting the
mouse on it says the same at length."""

from __future__ import annotations

import time

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QToolButton, QWidget

from acoustic_spaces.gui.numbers import fmt
from acoustic_spaces.i18n import _

#: How wide the text may grow in the status bar, in pixels; past it, it is cut with "…".
TEXT_WIDTH = 460


class _CutLabel(QLabel):
    """A one-line label that cuts its text with "…" where it does not fit, and asks for
    a steady width so that the status bar does not jump at every step."""

    def __init__(self) -> None:
        super().__init__()
        self._full = ""
        self.setMinimumWidth(120)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self._cut()

    def full_text(self) -> str:
        return self._full

    def sizeHint(self) -> QSize:
        return QSize(TEXT_WIDTH, super().sizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._cut()

    def _cut(self) -> None:
        metrics = self.fontMetrics()
        self.setText(metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, self.width()))


class Activity(QWidget):
    """What runs in the background, for the status bar; hidden while nothing does."""

    def __init__(self, cancel: QAction) -> None:
        super().__init__()
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # moves to and fro: the steps' number is not known
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(110)
        self.label = _CutLabel()
        self.cancel = QToolButton()
        self.cancel.setDefaultAction(cancel)
        self.cancel.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.bar)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.cancel)
        self._task, self._step, self._started = "", "", 0.0
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._update)
        self.hide()

    def start(self, task: str) -> None:
        """Show that ``task`` (such as "Simulating hall…") has started."""
        self._task, self._step, self._started = task, "", time.perf_counter()
        self._update()
        self._clock.start()
        self.show()

    def step(self, step: str) -> None:
        """Say what the task does now."""
        self._step = step
        self._update()

    def stop(self) -> None:
        self._clock.stop()
        self.hide()

    def text(self) -> str:
        """What the status bar says, before it is cut to fit."""
        return self.label.full_text()

    def _update(self) -> None:
        seconds = time.perf_counter() - self._started
        # Translators: a background task such as "Simulating hall…", then how long it has
        # run so far.
        text = fmt(_("{task} {time:.0f} s"), task=self._task, time=seconds)
        if self._step:
            # Translators: the task and its time, then what it does now.
            text = _("{task}: {step}").format(task=text, step=self._step)
        self.label.set_full_text(text)
        lines = [self._task]
        if self._step:
            lines.append(_("Now: {step}").format(step=self._step))
        lines += [
            fmt(_("Running for {time:.0f} s"), time=seconds),
            _(
                "Cancel, or Simulation > Cancel (Shift+F5), stops it: a run at its next "
                "stage, a demo at once"
            ),
        ]
        tip = "\n".join(lines)
        for widget in (self, self.bar, self.label):
            widget.setToolTip(tip)


__all__ = ["Activity", "TEXT_WIDTH"]
