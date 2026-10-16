"""How fast the CPU and the GPU are on this computer, and what ``auto`` picks from it.

``acoustic-spaces tune`` (:func:`tune`) times the late stage on the CPU, on the GPU and
on both at once, for several response lengths, and the wave solver on the CPU and on
the GPU, on one scene. It keeps the fastest for each in the user's cache folder, for
the computer (its GPU and CPU thread count):

- the late stage's ``auto`` (:mod:`.ray_device`) takes the choice for the run's
  response length: the one for the shortest timed length at least as long, else the
  longest. Without ``tune``, its trial keeps a choice there too, for its range of
  lengths (up to 0.5, 1, 2, 4 s and so on).
- the wave solver's ``auto`` compares the timed costs for the run's step count: per
  grid cell, a set-up and a cost per step on each device. Without ``tune``, it takes a
  hardware GPU when wgpu finds one.

A software GPU (such as lavapipe or SwiftShader) is timed like any other, but ``auto``
never takes it.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from acoustic_spaces.engine import gpu

#: Response lengths ``tune`` times the late stage at (s).
LENGTHS = (0.5, 1.0, 2.0, 4.0)
#: Response lengths ``tune`` times the wave solver at (s): two, to tell its set-up from
#: its cost per step.
WAVE_LENGTHS = (0.05, 0.2)
#: The example scene ``tune`` times when given none.
SCENE = "concert_hall"
#: How much faster the GPU, or both together, must be for ``auto`` to leave the CPU.
MARGIN = 1.1
_DEVICES = ("cpu", "gpu", "both")


def cache_file() -> Path:
    """Where the choices are kept: ``acoustic-spaces/devices.json`` in the user's cache
    folder."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "acoustic-spaces" / "devices.json"


def machine() -> str:
    """The computer the choices are kept for: its GPU and CPU thread count."""
    return f"{gpu.gpu_name()} | {os.cpu_count()} CPU threads"


def _read_all() -> dict:
    try:
        saved = json.loads(cache_file().read_text())
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


def saved() -> dict:
    """What is kept for this computer: ``{"late": {length: choice}, "wave": costs}``."""
    entry = _read_all().get(machine())
    return entry if isinstance(entry, dict) else {}


def _keep(stage: str, update: Callable[[dict], dict]) -> None:
    everything = _read_all()
    mine = everything.get(machine())
    mine = mine if isinstance(mine, dict) else {}
    current = mine.get(stage)
    mine[stage] = update(current if isinstance(current, dict) else {})
    everything[machine()] = mine
    path = cache_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(everything, indent=1))
    except OSError:
        pass  # a read-only home: the choice is simply not kept


def length_key(max_time: float) -> str:
    """The range of response lengths ``max_time`` (s) falls in, by its upper end: a
    power of two seconds (0.5, 1, 2, 4, ...)."""
    return f"{2.0 ** math.ceil(math.log2(max(float(max_time), 1e-3))):g}"


def late_choices() -> dict[float, dict]:
    """The late stage's choices kept for this computer, by range of response lengths
    (its upper end, s)."""
    entries = saved().get("late")
    found = {}
    for key, entry in (entries if isinstance(entries, dict) else {}).items():
        try:
            if entry["device"] in _DEVICES:
                found[float(key)] = entry | {
                    "cpu_rate": float(entry["cpu_rate"]),
                    "gpu_rate": float(entry["gpu_rate"]),
                }
        except (KeyError, TypeError, ValueError):
            continue  # an entry from another version, or edited by hand
    return found


def late_entry(max_time: float) -> dict | None:
    """The late stage's choice kept for exactly the range of ``max_time``, or None."""
    return late_choices().get(float(length_key(max_time)))


def late_choice(max_time: float) -> dict | None:
    """The late stage's choice for responses ``max_time`` long (s): the one kept for the
    shortest range at least that long, else for the longest; None when none is kept.
    It has ``device``, ``cpu_rate`` and ``gpu_rate`` (rays per second; only their ratio
    counts) and ``from`` (``tune`` or ``trial``)."""
    entries = late_choices()
    if not entries:
        return None
    wanted = float(length_key(max_time))
    longer = [length for length in entries if length >= wanted]
    return entries[min(longer) if longer else max(entries)]


def keep_late(max_time: float, choice: dict) -> None:
    """Keep the late stage's ``choice`` for the range of ``max_time``."""
    _keep("late", lambda late: late | {length_key(max_time): choice})


def wave_costs() -> dict | None:
    """The wave solver's timed costs, ``{"cpu": [a, b], "gpu": [a, b]}``: seconds per
    grid cell to set up (a) and per step (b); None when not timed."""
    costs = saved().get("wave")
    try:
        return {
            device: [float(costs[device][0]), float(costs[device][1])] for device in ("cpu", "gpu")
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def keep_wave(costs: dict) -> None:
    """Keep the wave solver's ``costs`` (see :func:`wave_costs`)."""
    _keep("wave", lambda _: costs)


def wave_prefers_gpu(steps: int) -> bool | None:
    """Whether the GPU steps a grid ``steps`` times clearly faster than the CPU did when
    timed; None when not timed."""
    costs = wave_costs()
    if costs is None:
        return None
    (cpu_a, cpu_b), (gpu_a, gpu_b) = costs["cpu"], costs["gpu"]
    return MARGIN * (gpu_a + steps * gpu_b) <= cpu_a + steps * cpu_b


def wave_pick(costs: dict) -> str:
    """What the wave solver's ``auto`` takes with ``costs``, in words."""
    (cpu_a, cpu_b), (gpu_a, gpu_b) = costs["cpu"], costs["gpu"]
    # The GPU for ``steps`` with steps * slope >= start.
    slope, start = cpu_b - MARGIN * gpu_b, MARGIN * gpu_a - cpu_a
    if slope == 0:
        return "the GPU" if start <= 0 else "the CPU"
    at = start / slope
    if slope > 0:
        return "the GPU" if at <= 1 else f"the GPU from {math.ceil(at):,} steps on, else the CPU"
    return "the CPU" if at < 1 else f"the GPU up to {math.floor(at):,} steps, else the CPU"


def _fastest(seconds: dict[str, float]) -> str:
    """The fastest of the devices timed, the CPU unless another is clearly faster."""
    best = min(seconds, key=seconds.__getitem__)
    return "cpu" if seconds["cpu"] <= MARGIN * seconds[best] else best


@dataclass
class LateTiming:
    """The late stage at one response length: seconds on each device, and the choice."""

    length: float
    seconds: dict[str, float]
    device: str


@dataclass
class WaveTiming:
    """The wave solver: seconds on each device at each step count, and the fitted costs
    (see :func:`wave_costs`)."""

    cells: int
    steps: list[int]
    seconds: dict[str, list[float]]
    costs: dict[str, list[float]]


@dataclass
class Tuning:
    """What :func:`tune` timed and kept."""

    scene: str
    machine: str
    rays: int
    late: list[LateTiming] = field(default_factory=list)
    wave: WaveTiming | None = None
    wave_skipped: str = ""
    software: bool = False
    file: Path = field(default_factory=lambda: cache_file())

    def summary(self) -> str:
        """The timings as a table, and what ``auto`` will pick."""
        lines = [
            f"{self.machine}",
            f"late stage, {self.rays:,} rays per band ({self.scene}):",
            f"  {'length':>8} {'cpu':>9} {'gpu':>9} {'both':>9}   auto",
        ]
        for timing in self.late:
            cells = " ".join(f"{timing.seconds[d]:7.2f} s" for d in _DEVICES)
            lines.append(f"  {timing.length:6g} s {cells}   {timing.device}")
        if self.wave is not None:
            wave = self.wave
            lines += [
                f"wave solver, {wave.cells:,} cells:",
                f"  {'steps':>8} {'cpu':>9} {'gpu':>9}",
            ]
            for i, steps in enumerate(wave.steps):
                cells = " ".join(f"{wave.seconds[d][i]:7.2f} s" for d in ("cpu", "gpu"))
                lines.append(f"  {steps:8,} {cells}")
            lines.append(f"  auto: {wave_pick(wave.costs)}")
        elif self.wave_skipped:
            lines.append(f"wave solver not timed: {self.wave_skipped}")
        if self.software:
            lines.append(
                "this GPU is a software renderer: auto keeps the CPU whatever the timings say"
            )
        lines.append(f"kept in {self.file}")
        return "\n".join(lines)


def _no_gpu() -> ValueError:
    try:
        import wgpu  # noqa: F401
    except ImportError:
        return ValueError(
            'tune compares the CPU with a GPU through wgpu: pip install "acoustic-spaces[gpu]"'
        )
    return ValueError(
        "tune compares the CPU with a GPU, but wgpu finds none: install a Vulkan, Metal or "
        "Direct3D 12 driver"
    )


def tune(
    room,
    *,
    rays: int | None = None,
    lengths: Sequence[float] = LENGTHS,
    wave_lengths: Sequence[float] | None = WAVE_LENGTHS,
    threads: int | None = None,
    say: Callable[[str], None] | None = None,
) -> Tuning:
    """Time the late stage and the wave solver of ``room`` (a loaded
    :class:`~acoustic_spaces.room.Room`) on the CPU and the GPU, keep the fastest
    for this computer and return the timings.

    The late stage traces the scene's first source to its listeners with the scene's
    settings, ``rays`` per band (default: the scene's), once for each response length
    of ``lengths`` (s) on each device. The wave solver runs the scene's wave settings
    (enabled or not) for the two ``wave_lengths``; None leaves it out. ``threads``
    defaults to the scene's.
    """
    import numpy as np

    from acoustic_spaces.engine import late as late_stage
    from acoustic_spaces.engine import ray_device
    from acoustic_spaces.engine.model import TraceModel
    from acoustic_spaces.engine.simulate import TIME_STEP
    from acoustic_spaces.spatial.directions import fibonacci_sphere

    say = say or (lambda _: None)
    if gpu._adapter() is None:
        raise _no_gpu()
    scene = room.scene
    if not scene.sources or not scene.listeners:
        raise ValueError("tune needs a scene with a source and a listener")
    settings = scene.simulation
    late_cfg, early_cfg = settings.late, settings.early
    rays = int(rays or late_cfg.rays)
    if rays < 1:
        raise ValueError("tune needs at least one ray")
    if not lengths or min(lengths) <= 0:
        raise ValueError("tune needs positive response lengths")
    threads = settings.performance.threads if threads is None else int(threads)
    model = TraceModel.from_room(room, settings)
    c = room.speed_of_sound()
    air = room.air_attenuation(model.bands)
    position = np.asarray(scene.sources[0].position, dtype=float)
    listeners = np.array([listener.position for listener in scene.listeners], dtype=float)
    directions = fibonacci_sphere(late_cfg.directions)
    result = Tuning(scene.name, machine(), rays, software=not gpu.hardware_gpu())

    def late_run(plan: ray_device.Plan, length: float, count: int = rays):
        radius = late_cfg.receiver_radius
        if radius is None:
            radius = late_stage.receiver_radius(room.volume_m3, count, length, c)
        started = time.perf_counter()
        late = late_stage.stochastic_energy(
            model,
            position,
            listeners,
            rays=count,
            max_time=length,
            time_step=TIME_STEP,
            speed_of_sound=c,
            air=air,
            reception=late_cfg.reception,
            covered_order=early_cfg.max_order,
            radius=radius,
            directions=directions,
            splitting=late_cfg.splitting,
            threads=threads,
            device=plan,
        )
        if late.plan is not None and late.plan.device != plan.device:
            raise ValueError(f"tune needs a scene the GPU can trace: {late.plan.reason}")
        return time.perf_counter() - started

    say("late stage: starting the GPU")
    late_run(ray_device.Plan("gpu"), 0.01, 64)  # compiles the shaders outside the timing
    for length in sorted(set(float(x) for x in lengths)):
        seconds = {}
        for device in ("cpu", "gpu"):
            say(f"late stage, {length:g} s responses: {device}")
            seconds[device] = late_run(ray_device.Plan(device), length)
        cpu_rate, gpu_rate = rays / seconds["cpu"], rays / seconds["gpu"]
        say(f"late stage, {length:g} s responses: both")
        seconds["both"] = late_run(ray_device.Plan("both", cpu_rate, gpu_rate), length)
        device = _fastest(seconds)
        result.late.append(LateTiming(length, seconds, device))
        keep_late(
            length,
            {
                "device": device,
                "cpu_rate": cpu_rate,
                "gpu_rate": gpu_rate,
                "from": "tune",
                "length": length,
                "rays": rays,
                "seconds": seconds,
            },
        )

    if wave_lengths is not None:
        try:
            result.wave = _time_wave(room, model, position, listeners, threads, wave_lengths, say)
        except ValueError as exc:
            result.wave_skipped = str(exc)
        else:
            keep_wave(result.wave.costs | {"from": "tune", "cells": result.wave.cells})
    return result


def _time_wave(room, model, position, listeners, threads, lengths, say) -> WaveTiming:
    """The wave solver timed on each device at the two ``lengths``, and its costs."""
    from acoustic_spaces.engine import wave as wave_stage

    settings = room.scene.simulation
    names = [listener.name for listener in room.scene.listeners]
    short, long = sorted(float(x) for x in lengths)[:2]
    if not short < long:
        raise ValueError("the wave solver is timed at two different lengths")

    def run(device: str, length: float) -> tuple[float, int, int]:
        marks: list[float] = []

        def mark(message: str) -> None:
            # Its first message comes just before the first step.
            if not marks and " steps" in message:
                marks.append(time.perf_counter())

        _, info = wave_stage.wave_responses(
            model,
            position,
            listeners,
            names,
            max_band_hz=settings.wave.max_band_hz,
            length_s=length,
            sample_rate=settings.sample_rate,
            speed_of_sound=room.speed_of_sound(),
            points_per_wavelength=settings.wave.points_per_wavelength,
            threads=threads,
            device=device,
            scheme=settings.wave.scheme,
            say=mark,
        )
        return time.perf_counter() - marks[0], int(info["wave_steps"]), int(info["wave_cells"])

    say("wave solver: starting the GPU")
    run("gpu", short)  # compiles the shaders outside the timing
    seconds: dict[str, list[float]] = {}
    steps: list[int] = []
    cells = 0
    for device in ("cpu", "gpu"):
        seconds[device] = []
        for length in (short, long):
            say(f"wave solver, {length:g} s: {device}")
            took, steps_run, cells = run(device, length)
            seconds[device].append(took)
            if len(steps) < 2:
                steps.append(steps_run)
    costs = {}
    for device, (first, second) in seconds.items():
        per_step = max((second - first) / (cells * max(steps[1] - steps[0], 1)), 1e-15)
        costs[device] = [max(first / cells - steps[0] * per_step, 0.0), per_step]
    return WaveTiming(cells, steps, seconds, costs)


__all__ = [
    "LENGTHS",
    "SCENE",
    "Tuning",
    "cache_file",
    "keep_late",
    "keep_wave",
    "late_choice",
    "late_choices",
    "late_entry",
    "length_key",
    "machine",
    "saved",
    "tune",
    "wave_costs",
    "wave_pick",
    "wave_prefers_gpu",
]
