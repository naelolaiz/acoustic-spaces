"""Simulations and coverage maps in the background, so the editor stays responsive
while rays fly.

The compiled tracer releases Python's lock while it works, so the window keeps
drawing. A cancelled run stops at the next stage (the early or late part of the next
source); its result is dropped.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Signal

from acoustic_spaces.engine import coverage_map, simulate
from acoustic_spaces.room import Room
from acoustic_spaces.scene.model import Simulation


class Cancelled(Exception):
    """Raised inside the simulation to stop it."""


class SimulationJob(QThread):
    """Runs :func:`acoustic_spaces.engine.simulate` on a room in a worker thread."""

    #: What it runs, for messages.
    kind = "simulation"
    progress = Signal(str)
    #: The :class:`acoustic_spaces.engine.SimulationResult`.
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        room: Room,
        settings: Simulation | None = None,
        threads: int | None = None,
        groups: np.ndarray | None = None,
    ):
        super().__init__()
        self.room = room
        self.settings = settings
        self.threads = threads
        #: Face groups to count reflections off (``simulate(groups=...)``).
        self.groups = groups
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self._stop:
            raise Cancelled
        self.progress.emit(message)

    def compute(self):
        return simulate(
            self.room,
            settings=self.settings,
            threads=self.threads,
            progress=self._say,
            groups=self.groups,
        )

    def run(self) -> None:
        try:
            result = self.compute()
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


class MapJob(SimulationJob):
    """Runs :func:`acoustic_spaces.engine.coverage_map`; ``done`` carries the
    :class:`~acoustic_spaces.engine.CoverageResult`."""

    kind = "coverage map"

    def compute(self):
        return coverage_map(
            self.room, simulation=self.settings, threads=self.threads, progress=self._say
        )


class TuneJob(SimulationJob):
    """Runs :func:`acoustic_spaces.engine.tuning.tune` (``acoustic-spaces tune``);
    ``done`` carries the :class:`~acoustic_spaces.engine.tuning.Tuning`, not yet
    saved: the tune window saves it on OK."""

    kind = "tuning"

    def compute(self):
        from acoustic_spaces.engine import tuning

        return tuning.tune(self.room, threads=self.threads, say=self._say, keep=False)


__all__ = ["Cancelled", "MapJob", "SimulationJob", "TuneJob"]
