"""Simulations, coverage maps, exports, demo runs and downloads in the background, so
the editor stays responsive while rays fly.

The compiled tracer releases Python's lock while it works, so the window keeps
drawing. A cancelled run stops at the next stage (the early or late part of the next
source); its result is dropped.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from acoustic_spaces.engine.long_term import simulate
from acoustic_spaces.engine.sensitivity import coverage_map
from acoustic_spaces.i18n import _
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
    #: The result before its spread from the absorption uncertainty is drawn, to show
    #: meanwhile; ``done`` then carries it with the spread.
    traced = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        room: Room,
        settings: Simulation | None = None,
        threads: int | None = None,
        groups: np.ndarray | None = None,
        materials: list[str] | None = None,
    ):
        super().__init__()
        self.room = room
        self.settings = settings
        self.threads = threads
        #: Face groups to count reflections off (``simulate(groups=...)``), and the
        #: materials they are of, for the spread from their absorption uncertainty.
        self.groups = groups
        self.materials = materials
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def _say(self, message: str) -> None:
        if self._stop:
            raise Cancelled
        self.progress.emit(message)

    def compute(self):
        result = simulate(
            self.room,
            settings=self.settings,
            threads=self.threads,
            progress=self._say,
            groups=self.groups,
        )
        from acoustic_spaces.engine.run_size import worker_threads
        from acoustic_spaces.engine.uncertainty import has_uncertainty, with_spread

        settings = self.settings or self.room.scene.simulation
        draws = settings.uncertainty.draws
        if not draws or not self.materials or not has_uncertainty(self.room.scene, self.materials):
            return result
        self.traced.emit(result)
        self._say(_("Drawing the spread from the absorption uncertainty…"))
        return with_spread(
            result,
            self.room.scene,
            self.groups,
            self.materials,
            draws=draws,
            seed=settings.seed or 0,
            progress=self._say,
            one_sided=settings.uncertainty.one_sided,
            workers=worker_threads(
                settings.performance.threads if self.threads is None else self.threads
            ),
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
        if self._stop and not self._complete(result):
            self.cancelled.emit()
        else:
            self.done.emit(result)

    def _complete(self, result) -> bool:
        """Whether ``result`` is whole although the job was cancelled, so that the job
        is done rather than cancelled (a demo that had run every step)."""
        return False


class MapJob(SimulationJob):
    """Runs :func:`acoustic_spaces.engine.sensitivity.coverage_map` (also for an
    insertion loss, in weather classes and with the map's spread);
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


#: How long the editor waits, once a demo has said that every step has run (or has
#: closed its output), for its process to end (s): past it the editor stops the process,
#: as everything it writes is complete.
DEMO_EXIT_S = 30.0
#: How long the editor reads what is left of a demo's output once its process has ended
#: (s): past it the editor no longer waits for a process the demo started that still
#: holds the output, and on Linux and macOS stops it.
DEMO_DRAIN_S = 2.0
#: How long a demo's processes have to end after SIGTERM before SIGKILL, and then
#: before the editor no longer waits for them (s).
DEMO_KILL_S = 5.0
#: How often the editor looks at a demo's process while no line comes (s).
_DEMO_POLL_S = 0.2


class DemoJob(SimulationJob):
    """Runs a step of a demo (:mod:`acoustic_spaces.demos`) in a process of its own, as
    :meth:`Demo.run_argv` gives it (``python NAME_demo.py STEP``, or ``python -m
    acoustic_spaces.demos.scene_demo STEP`` for the demo of any scene; an example's
    script that is no demo through :mod:`acoustic_spaces.demos.run_script`): what the
    demo prints becomes progress messages, and Cancel stops it and what it started where
    they are. The run ends when the demo's process has ended. A demo that said every step
    has run (:data:`~acoustic_spaces.demos.DONE`) but whose process does not end within
    :data:`DEMO_EXIT_S` is stopped, and the editor no longer waits for a process it
    started that still holds its output :data:`DEMO_DRAIN_S` after it ended (on Linux
    and macOS it stops that too); the run is done all the same, and so it is when the
    process fails after that line. ``done`` carries the demo, the step, the folder the
    demo wrote into, and ``stopped``: why the editor stopped the demo's processes
    (``"not ending"``, ``"output held"``, ``"cancelled"`` after every step had run) or
    None, with the demo's ``pid``, how long the editor waits for it to end
    (``waited_s``) and its ``exit_code`` where it failed after every step had run (else
    0)."""

    kind = "demo"

    def __init__(self, demo, step: str, args) -> None:
        super().__init__(None)
        self.demo = demo
        self.step = step
        self.args = args
        #: Every line the demo printed, in order.
        self.lines: list[str] = []

    @property
    def argv(self) -> list[str]:
        """The command as a list (:meth:`Demo.run_argv`). An example's script that is no
        demo prints no :data:`~acoustic_spaces.demos.DONE` line of its own: it runs
        through :mod:`acoustic_spaces.demos.run_script`, which prints it once the script
        has ended well."""
        argv = self.demo.run_argv(self.step, self.args)
        if self.demo.stepless:
            return [argv[0], "-m", "acoustic_spaces.demos.run_script", *argv[1:]]
        return argv

    def compute(self) -> dict:
        from acoustic_spaces.demos import DONE

        folder = Path(self.demo.folder)
        folder.mkdir(parents=True, exist_ok=True)
        self._say(self.demo.command(self.step, self.args))
        # The command the window showed, with paths in full; unbuffered, so its lines
        # arrive as it writes them rather than at the end. Its output passes as UTF-8,
        # whatever the system's code page (on Windows a pipe takes the ANSI code page,
        # which may lack the letters of a folder's name); a byte that is no UTF-8 (from
        # ffmpeg, which writes into the same pipe) reads as a replacement character. It
        # reads nothing (nor does what it starts), and on Linux and macOS it starts a
        # session of its own, so that stopping it stops what it started too.
        process = subprocess.Popen(
            self.argv,
            cwd=folder,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=sys.platform != "win32",
        )
        lines: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(
            target=_read_lines, args=(process.stdout, lines), name="demo output", daemon=True
        ).start()
        try:
            stopped, done = self._follow(process, lines, DONE)
        except BaseException:  # this thread failed: the demo goes too
            _stop_processes(process)
            raise
        if stopped is not None:
            _stop_processes(process)
        if stopped == "cancelled" and not done:
            raise Cancelled
        if stopped == "not ending" and not done:
            raise RuntimeError(
                f"{self.step}: the demo closed its output, but its process had not ended "
                f"{DEMO_EXIT_S:.0f} s later; the editor stopped it."
            )
        # Its own exit code, where it ended by itself: a failure once every step had run
        # (a library failing as the process ends) leaves what it wrote whole.
        code = process.returncode if stopped in (None, "output held") else 0
        if code and not done:
            last = self.lines[-1] if self.lines else ""
            raise RuntimeError(f"{self.step}: the demo stopped ({code}). {last}".strip())
        work = self.demo.work_folder(self.args)
        return {
            "demo": self.demo.name,
            "step": self.step,
            "folder": str(work),
            "stopped": stopped,
            "pid": process.pid,
            "waited_s": DEMO_EXIT_S,
            "exit_code": code or 0,
        }

    def _complete(self, result) -> bool:
        # compute() gives a result only when what the demo wrote is whole (a Cancel
        # before every step had run raises Cancelled), whenever Cancel came.
        return result is not None

    def _follow(
        self, process: subprocess.Popen, lines: queue.SimpleQueue, done_line: str
    ) -> tuple[str | None, bool]:
        """Pass on the demo's lines from ``lines`` until its process has ended and its
        output with it; whether it said every step has run (a line beginning with
        ``done_line``), and why its processes must be stopped: None (they ended), or
        ``"cancelled"``, ``"not ending"`` (no end :data:`DEMO_EXIT_S` after that line or
        after the output closed) or ``"output held"`` (the output still open
        :data:`DEMO_DRAIN_S` after the process ended)."""
        done_at: float | None = None
        closed_at: float | None = None
        ended_at: float | None = None
        told = False

        def take(line: str | None) -> None:
            """Note a line of the demo's output, or its end (None), and pass it on."""
            nonlocal done_at, closed_at
            if line is None:
                closed_at = time.monotonic()
            elif line.strip():
                text = line.rstrip()
                self.lines.append(text)
                if done_at is None and text.startswith(done_line):
                    done_at = time.monotonic()
                self._say(text)

        try:
            while not self._stop:
                if closed_at is None:
                    try:
                        take(lines.get(timeout=_DEMO_POLL_S))
                    except queue.Empty:
                        pass
                else:
                    time.sleep(_DEMO_POLL_S)
                # After every line too: output that never pauses runs out the time as
                # well as silence does.
                now = time.monotonic()
                if ended_at is None and process.poll() is not None:
                    ended_at = now
                if ended_at is not None:
                    if closed_at is None and now - ended_at > DEMO_DRAIN_S:
                        # What is read already is passed on; what a process the demo
                        # started still writes is not waited for.
                        for _line in range(lines.qsize()):
                            take(lines.get_nowait())
                            if closed_at is not None:
                                break
                        if closed_at is None:
                            return "output held", done_at is not None
                    if closed_at is not None:
                        return None, done_at is not None
                    continue
                finished = done_at if done_at is not None else closed_at
                if finished is None:
                    continue
                if now - finished > DEMO_EXIT_S:
                    return "not ending", done_at is not None
                if not told and now - finished > DEMO_DRAIN_S:
                    told = True
                    self._say(
                        _("waiting up to {seconds:.0f} s for the demo's process to end").format(
                            seconds=DEMO_EXIT_S
                        )
                    )
        except Cancelled:
            pass
        return "cancelled", done_at is not None


def _read_lines(stream, lines: queue.SimpleQueue) -> None:
    """Put each line of ``stream`` in ``lines``, then None once it ends (or can be read
    no more), and close it. A demo's output is read so, in a thread of its own, so that
    the editor follows the demo's process whatever holds the output open."""
    try:
        for line in stream:
            lines.put(line)
    except (OSError, ValueError):
        pass
    finally:
        lines.put(None)
        try:
            stream.close()
        except OSError:
            pass


def _group_alive(process: subprocess.Popen) -> bool:
    """Whether a process of the demo's group is left (Linux and macOS): the demo itself
    is reaped first, so that its group can empty."""
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _stop_processes(process: subprocess.Popen) -> None:
    """Stop the demo's process and those it started: on Linux and macOS its whole
    session's group, SIGTERM and then, for what has not ended :data:`DEMO_KILL_S`
    later, SIGKILL; on Windows the demo's tree (taskkill), while the demo is there to
    find it by. The editor waits no longer than that: a process stuck in the kernel ends
    when it can."""
    if sys.platform == "win32":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
            )
    else:
        for sent in (signal.SIGTERM, signal.SIGKILL):
            if not _group_alive(process):
                break
            try:
                os.killpg(process.pid, sent)
            except (ProcessLookupError, PermissionError):
                break
            deadline = time.monotonic() + DEMO_KILL_S
            while time.monotonic() < deadline and _group_alive(process):
                time.sleep(0.05)
    try:
        process.wait(timeout=DEMO_KILL_S)
    except subprocess.TimeoutExpired:
        pass


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
    "DEMO_DRAIN_S",
    "DEMO_EXIT_S",
    "DEMO_KILL_S",
    "Cancelled",
    "DemoJob",
    "DownloadJob",
    "ExportJob",
    "MapJob",
    "MovementJob",
    "SimulationJob",
    "TuneJob",
]
