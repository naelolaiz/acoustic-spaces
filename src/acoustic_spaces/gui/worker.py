"""Simulations, coverage maps, exports, demo runs and downloads in the background, so
the editor stays responsive while rays fly.

The compiled tracer releases Python's lock while it works, so the window keeps
drawing. A cancelled run stops at the next stage (the early or late part of the next
source); its result is dropped.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from acoustic_spaces.engine.long_term import coverage_map, simulate
from acoustic_spaces.room import Room
from acoustic_spaces.scene.model import Simulation


class Cancelled(Exception):
    """Raised inside the simulation to stop it."""


class SimulationJob(QThread):
    """Runs :func:`acoustic_spaces.engine.simulate` on a room in a worker thread, and
    again without what ``simulation.insertion_loss_of`` names, for its insertion loss
    (:mod:`acoustic_spaces.engine.insertion`), and in each weather class of the scene's
    ``long_term`` (:mod:`acoustic_spaces.engine.long_term`)."""

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
    """Runs :func:`acoustic_spaces.engine.coverage_map` (also for an insertion loss and
    in weather classes);
    ``done`` carries the :class:`~acoustic_spaces.engine.CoverageResult`."""

    kind = "coverage map"

    def compute(self):
        return coverage_map(
            self.room, simulation=self.settings, threads=self.threads, progress=self._say
        )


class MovementJob(SimulationJob):
    """Runs :func:`acoustic_spaces.walkthrough.simulate_movement` on a room, every mover
    on one clock, plays ``signals`` (source names to the WAV files they play) through it
    in format ``fmt`` and writes what each listener in use hears
    (:func:`~acoustic_spaces.walkthrough.save_movement`: ``out`` for one listener, else
    ``<out stem>_<listener>.wav`` each), one gain bringing the loudest peak to -1 dBFS;
    ``done`` carries the movement, the output's label, the files written by listener, the
    gain, the sample rate, the channels and the length (s)."""

    kind = "movement"

    def __init__(self, room: Room, out, signals, fmt=None, threads: int | None = None):
        super().__init__(room, threads=threads)
        self.out = Path(out)
        self.signals = dict(signals)
        self.fmt = fmt

    def compute(self) -> dict:
        from acoustic_spaces.audio import read_wav
        from acoustic_spaces.walkthrough import (
            render_movement,
            save_movement,
            simulate_movement,
        )

        movement = simulate_movement(self.room, threads=self.threads, progress=self._say)
        self._say("rendering the movement")
        signals = {name: read_wav(path) for name, path in self.signals.items()}
        rendered = render_movement(movement, signals, self.fmt)
        rate = int(movement.result.settings.sample_rate)
        self._say(f"writing {self.out}")
        paths, gain = save_movement(rendered, self.out, rate, self.fmt, normalize=True)
        first = next(iter(rendered.values()))
        return {
            "movement": movement,
            "output": getattr(self.fmt, "label", "mono"),
            "paths": paths,
            "gain": gain,
            "rate": rate,
            "channels": int(first.shape[1]),
            "length_s": len(first) / rate,
        }


class TuneJob(SimulationJob):
    """Runs :func:`acoustic_spaces.engine.tuning.tune` (``acoustic-spaces tune``);
    ``done`` carries the :class:`~acoustic_spaces.engine.tuning.Tuning`, not yet
    saved: the tune window saves it on OK."""

    kind = "tuning"

    def compute(self):
        from acoustic_spaces.engine import tuning

        return tuning.tune(self.room, threads=self.threads, say=self._say, keep=False)


class DemoJob(SimulationJob):
    """Runs a step of a demo (:mod:`acoustic_spaces.demos`) in a process of its own, as
    :meth:`Demo.run_argv` gives it (``python NAME_demo.py STEP``, or ``python -m
    acoustic_spaces.demos.scene_demo STEP`` for the demo of any scene): what the demo
    prints becomes progress messages, and Cancel stops it where it is. ``done`` carries
    the demo, the step and the folder the demo wrote into."""

    kind = "demo"

    def __init__(self, demo, step: str, args) -> None:
        super().__init__(None)
        self.demo = demo
        self.step = step
        self.args = args
        #: Every line the demo printed, in order.
        self.lines: list[str] = []
        self._process: subprocess.Popen | None = None

    @property
    def argv(self) -> list[str]:
        """The command as a list (:meth:`Demo.run_argv`)."""
        return self.demo.run_argv(self.step, self.args)

    def cancel(self) -> None:
        """Stop the demo: it is a process of its own, so it ends at once rather than at
        the next stage."""
        super().cancel()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def compute(self) -> dict:
        folder = Path(self.demo.folder)
        folder.mkdir(parents=True, exist_ok=True)
        self._say(self.demo.command(self.step, self.args))
        # The command the window showed, with paths in full; unbuffered, so its lines
        # arrive as it writes them rather than at the end.
        process = self._process = subprocess.Popen(
            self.argv,
            cwd=folder,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:  # type: ignore[union-attr]
                text = line.rstrip()
                if text:
                    self.lines.append(text)
                    self._say(text)
            code = process.wait()
        finally:
            if process.poll() is None:  # cancelled, or this thread failed
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
        if self._stop:
            raise Cancelled
        if code:
            last = self.lines[-1] if self.lines else ""
            raise RuntimeError(f"{self.step}: the demo stopped ({code}). {last}".strip())
        work = self.demo.work_folder(self.args)
        return {"demo": self.demo.name, "step": self.step, "folder": str(work)}


class DownloadJob(QThread):
    """Fetches a file (an HRTF set) in the background, or one member of an archive:
    ``progress(done, total)`` counts the bytes, ``done`` carries the path, and
    :meth:`cancel` stops it where it is, leaving no half file."""

    progress = Signal(int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, path, member: str | None = None) -> None:
        super().__init__()
        self.url = url
        self.path = Path(path)
        self.member = member
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def _step(self, done: int, total: int | None) -> None:
        if self._stop:
            raise Cancelled
        self.progress.emit(done, total or 0)

    def run(self) -> None:
        from acoustic_spaces.fetch import fetch

        try:
            path = fetch(self.url, self.path, member=self.member, progress=self._step)
        except Cancelled:
            return
        except Exception as exc:  # shown in the window rather than lost in the thread
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.done.emit(str(path))


class ExportJob(SimulationJob):
    """Writes the responses of a run into a folder
    (:func:`acoustic_spaces.results.write_results`, as ``acoustic-spaces simulate``
    does); ``done`` carries the report. Cancelled, it stops before the next file and
    leaves the folder as it was."""

    kind = "export"

    def __init__(self, result, folder) -> None:
        super().__init__(result.room)
        self.result = result
        self.folder = folder

    def compute(self):
        from acoustic_spaces.results import write_results

        return write_results(self.result, self.folder, progress=self._say)


__all__ = [
    "Cancelled",
    "DemoJob",
    "DownloadJob",
    "ExportJob",
    "MapJob",
    "MovementJob",
    "SimulationJob",
    "TuneJob",
]
