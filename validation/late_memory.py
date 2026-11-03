"""Measure the memory the late part takes against the estimate that refuses or notes a
run (``engine.run_size``), on Linux.

    python validation/late_memory.py [--parts late whole] [--lengths 30 120 480]
                                     [--examples stadium_concert orchestra]
                                     [--device cpu]

Each run goes in a process of its own, and its peak is the most memory the process held
(``VmHWM``, reset through ``/proc/self/clear_refs``) above what it held as the run
started:

- ``late``: the late part of the first source of the shoebox example alone, for
  responses of ``--lengths`` seconds, on 1, 4 and 64 threads (64 run as the cores),
  with vector-based and random scattering; 2,000 rays, as on the CPU the rays cost time,
  not memory. ``--device`` gpu or both traces them there (``late.device``; the GPU's
  driver takes memory of its own, which the estimate leaves out).
- ``whole``: whole simulations of ``--examples`` (``stadium_concert`` at 8 s and
  ``orchestra`` at 6 s) on 4 threads, with ``late.batches`` 10 (the default) and 1.

How much the memory allocator keeps of what a run frees moves the peaks a little: with
glibc, ``MALLOC_MMAP_THRESHOLD_=131072`` fixes the size from which it hands large
arrays back.

Prints a Markdown table: the estimate, the measured peak and how far apart they are.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import subprocess
import sys
from pathlib import Path

MiB = 2**20
THREADS = (1, 4, 64)
WHOLE = {"stadium_concert": 8.0, "orchestra": 6.0}


def _status(key: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise SystemExit(f"/proc/self/status has no {key}: this needs Linux")


def _reset() -> int:
    """Sets the peak to the memory held now, and returns that."""
    gc.collect()
    Path("/proc/self/clear_refs").write_text("5")
    return _status("VmRSS")


def _measure(
    part: str, name: str, length: float, setting: str, threads: int, device: str = "cpu"
) -> dict:
    """One run's estimate and peak in bytes: ``setting`` is the scattering method for
    ``late``, the batches for ``whole``."""
    from acoustic_spaces import Room, simulate
    from acoustic_spaces.engine import run_size
    from acoustic_spaces.examples import example_path

    room = Room.from_yaml(example_path(name))
    settings = room.scene.simulation
    settings.ir_length_s = length
    sources = [s for s in room.scene.sources if s.in_use]
    listeners = [s for s in room.scene.listeners if s.in_use]
    if part == "whole":
        settings.late.batches = int(setting)
        need = run_size.simulation_need(room, settings, sources, listeners, threads)
        before = _reset()
        simulate(room, threads=threads)
        return {"estimate": need.bytes, "peak": _status("VmHWM") - before}
    settings.scattering.method = setting
    settings.late.rays = 2000
    settings.late.device = device
    sources = sources[:1]
    need = run_size.simulation_need(room, settings, sources, listeners, threads)
    late = importlib.import_module("acoustic_spaces.engine.late")
    traced = late.stochastic_energy
    peaks = []

    def measured(*args, **kwargs):
        before = _reset()
        found = traced(*args, **kwargs)
        peaks.append(_status("VmHWM") - before)
        return found

    late.stochastic_energy = measured
    simulate(room, sources=[sources[0].name], threads=threads)
    return {"estimate": need.bytes, "peak": max(peaks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", choices=("late", "whole"), default=["late", "whole"])
    parser.add_argument("--lengths", nargs="+", type=float, default=[30.0, 120.0, 480.0])
    parser.add_argument("--examples", nargs="+", choices=list(WHOLE), default=list(WHOLE))
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print(json.dumps(_measure(*json.loads(args.one))))
        return
    runs = []
    if "late" in args.parts:
        for method in ("vector", "random"):
            for length in args.lengths:
                runs += [("late", "shoebox", length, method, t, args.device) for t in THREADS]
    if "whole" in args.parts:
        runs += [("whole", n, WHOLE[n], b, 4) for n in args.examples for b in ("10", "1")]
    print("| Run | Response (s) | Setting | Threads | Estimate (MiB) | Peak (MiB) | Apart (MiB) |")
    print("|-----|-------------:|---------|--------:|---------------:|-----------:|------------:|")
    for run in runs:
        done = subprocess.run(
            [sys.executable, __file__, "--one", json.dumps(run)],
            check=True,
            capture_output=True,
            text=True,
        )
        found = json.loads(done.stdout.splitlines()[-1])
        estimate, peak = found["estimate"] / MiB, found["peak"] / MiB
        part, name, length, setting, threads = run[:5]
        setting = f"batches {setting}" if part == "whole" else f"{setting} scattering"
        if part == "late" and args.device != "cpu":
            setting += f", {args.device}"
        print(
            f"| {name} ({part}) | {length:g} | {setting} | {threads} | {estimate:,.1f} | "
            f"{peak:,.1f} | {peak - estimate:+,.1f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
