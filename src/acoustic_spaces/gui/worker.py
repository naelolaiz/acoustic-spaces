"""Simulations in the background, so the editor stays responsive while rays fly.

The compiled tracer releases Python's lock while it works, so the window keeps
drawing. A cancelled run stops at the next stage (the early or late part of the next
source); its result is dropped.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from acoustic_spaces.engine import simulate
from acoustic_spaces.room import Room
from acoustic_spaces.scene.model import Simulation


class Cancelled(Exception):
    """Raised inside the simulation to stop it."""


class SimulationJob(QThread):
    """Runs :func:`acoustic_spaces.engine.simulate` on a room in a worker thread."""

    progress = Signal(str)
    #: The :class:`acoustic_spaces.engine.SimulationResult`.
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, room: Room, settings: Simulation | None = None, threads: int = 0):
        super().__init__()
        self.room = room
        self.settings = settings
        self.threads = threads
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self._stop:
            raise Cancelled
        self.progress.emit(message)

    def run(self) -> None:
        try:
            result = simulate(
                self.room, settings=self.settings, threads=self.threads, progress=self._say
            )
        except Cancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # shown to the user rather than lost in the thread
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        if self._stop:
            self.cancelled.emit()
        else:
            self.done.emit(result)


__all__ = ["Cancelled", "SimulationJob"]
