"""Where the late stage traces its rays: on the CPU, on a GPU, or on both at once.

``simulation.late.device`` picks it:

- ``cpu`` (the default): the compiled core, on ``simulation.performance.threads``.
- ``gpu``: :mod:`acoustic_spaces.engine.gpu_tracer`, through WebGPU (the ``gpu`` extra).
- ``both``: the two share the rays. Each takes the next share of them as soon as it is
  done with the last, so the faster one traces more.
- ``auto``: whichever of the three is fastest on this computer. A short trial traces a
  few thousand rays on the scene's own model on the CPU and on the GPU and keeps the
  fastest; it takes the CPU, the safe choice, when wgpu finds no GPU or only a
  software one, or when neither the GPU nor both together would be clearly faster.
  The choice is saved for the computer. With the trial switched off
  (:func:`set_trial`, ``--no-device-trial``), ``auto`` takes the saved choice, or the
  CPU when there is none.

What only the CPU traces (sound through partitions and openings, the face groups of
material calibration) runs there, whatever the setting.

The two agree statistically, not ray for ray (the GPU works in single precision and
draws its own random numbers). With ``cpu`` or ``gpu`` a run gives the same result
every time; with ``both`` which rays each one traces depends on how fast each goes,
so results vary from run to run within the noise of the ray count.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from acoustic_spaces.engine import gpu

#: The settings of ``simulation.late.device``.
Device = Literal["cpu", "gpu", "both", "auto"]
DEVICES = ("cpu", "gpu", "both", "auto")
#: The command-line switch (``acoustic-spaces simulate`` and the editor) for set_trial.
TRIAL_FLAG = "--no-device-trial"
#: The trial: rays, and the response length they are traced for (s), at most.
TRIAL_RAYS = 8192
TRIAL_TIME = 0.25
#: How much faster the GPU, or both together, must be in the trial for auto to leave
#: the CPU; and how much the CPU must add to the GPU for auto to share the rays.
MARGIN = 1.2
CPU_SHARE = 1.1
#: Parts of the rays each one takes at a time when they share them: about this many
#: per device (fewer and larger for the GPU, which works best on many rays at once).
CPU_PARTS = 12
GPU_PARTS = 4

_trial = True
_lock = threading.Lock()
#: Trial results of this session: per model and band count, the CPU's and the GPU's rays
#: per second.
_rates: dict[tuple, tuple[float, float]] = {}


def set_trial(enabled: bool) -> None:
    """Switch ``auto``'s trial on or off for this session (off: the saved choice)."""
    global _trial
    _trial = bool(enabled)


def trial_enabled() -> bool:
    return _trial


@dataclass(frozen=True)
class Plan:
    """Where rays go: ``device`` is cpu, gpu or both; ``cpu_rate`` and ``gpu_rate``
    (rays per second, from the trial; 1 for unknown) share them; ``reason`` says why,
    when it differs from the setting."""

    device: str
    cpu_rate: float = 1.0
    gpu_rate: float = 1.0
    reason: str = ""


def _cache_file() -> Path:
    """Where the trial's choice is kept for the computer."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "acoustic-spaces" / "late-device.json"


def _machine() -> str:
    return f"{gpu.gpu_name()} | {os.cpu_count()} CPU threads"


def saved_choice() -> dict | None:
    """The last trial's choice and speeds on this computer, or None."""
    try:
        return json.loads(_cache_file().read_text()).get(_machine())
    except (OSError, ValueError, AttributeError):
        return None


def _save(choice: dict) -> None:
    path = _cache_file()
    try:
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        saved[_machine()] = choice
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=1))
    except OSError:
        pass  # a read-only home: the choice is simply not kept


def _unsupported(values: dict) -> str:
    """Why only the CPU can trace this run, or ''."""
    if np.asarray(values["transmission"]).size:
        return "sound through partitions and openings is traced on the CPU"
    if values["groups"]:
        return "face groups (material calibration) are traced on the CPU"
    return ""


def _check_gpu(setting: str) -> None:
    """For a device setting that names the GPU: raises ValueError without one."""
    if gpu._adapter() is not None:
        return
    try:
        import wgpu  # noqa: F401
    except ImportError:
        raise ValueError(
            f"simulation.late.device is {setting}, which needs wgpu: pip install "
            '"acoustic-spaces[gpu]"'
        ) from None
    raise ValueError(
        f"simulation.late.device is {setting}, but wgpu finds no GPU: install a Vulkan, "
        "Metal or Direct3D 12 driver, or set it to cpu or auto"
    )


def _cpu_threads(values: dict) -> int:
    return int(values["threads"]) or os.cpu_count() or 1


def trial(values: dict) -> tuple[float, float]:
    """Rays per second on the CPU and on the GPU for a run like ``values`` (the
    arguments of ``_native.trace_stochastic``): a few thousand rays traced on each, for
    a short response, past the run's own random streams."""
    from acoustic_spaces import _native
    from acoustic_spaces.engine import gpu_tracer

    short = values | {
        "rays": TRIAL_RAYS,
        "first_ray": 1 << 40,
        "max_time": min(float(values["max_time"]), TRIAL_TIME),
        "split_target": np.zeros(0),
    }
    # Compile the shader and start the GPU outside the timing.
    gpu_tracer.trace_stochastic(**(short | {"rays": 64, "max_time": 0.01}))
    started = time.perf_counter()
    gpu_tracer.trace_stochastic(**short)
    on_gpu = time.perf_counter() - started
    started = time.perf_counter()
    _native.trace_stochastic(**short)
    on_cpu = time.perf_counter() - started
    return TRIAL_RAYS / max(on_cpu, 1e-9), TRIAL_RAYS / max(on_gpu, 1e-9)


def _decide(cpu_rate: float, gpu_rate: float, threads: int) -> str:
    """The fastest of cpu, gpu and both by the trial's speeds, the CPU on a close call.
    Sharing leaves one CPU thread to keep the GPU busy."""
    both = cpu_rate * (threads - 1) / threads + gpu_rate if threads > 1 else gpu_rate
    if both >= MARGIN * cpu_rate and both >= CPU_SHARE * gpu_rate:
        return "both"
    if gpu_rate >= MARGIN * cpu_rate:
        return "gpu"
    return "cpu"


def choose(setting: str, values: dict) -> Plan:
    """Where to trace a run with the arguments ``values`` of
    ``_native.trace_stochastic`` for the device ``setting``."""
    if setting not in DEVICES:
        choices = ", ".join(DEVICES)
        raise ValueError(f"the late stage's device must be one of {choices}, not {setting!r}")
    if setting == "cpu":
        return Plan("cpu")
    if setting in ("gpu", "both"):
        _check_gpu(setting)
    reason = _unsupported(values)
    if reason:
        return Plan("cpu", reason=reason)
    if setting == "gpu":
        return Plan("gpu")
    if setting == "both":
        # Shares by the speeds of a trial, if there was one; else even ones.
        saved = saved_choice() or {"cpu_rate": 1.0, "gpu_rate": 1.0}
        rates = _rates.get(_key(values), (saved["cpu_rate"], saved["gpu_rate"]))
        return Plan("both", *rates)
    # auto
    if not gpu.hardware_gpu():
        return Plan("cpu", reason="no hardware GPU")
    threads = _cpu_threads(values)
    if not _trial:
        saved = saved_choice()
        if not saved:
            return Plan("cpu", reason="no trial on this computer yet")
        return Plan(saved["device"], saved["cpu_rate"], saved["gpu_rate"], "the saved choice")
    key = _key(values)
    with _lock:
        if key not in _rates:
            _rates[key] = trial(values)
    cpu_rate, gpu_rate = _rates[key]
    device = _decide(cpu_rate, gpu_rate, threads)
    _save({"device": device, "cpu_rate": cpu_rate, "gpu_rate": gpu_rate})
    return Plan(device, cpu_rate, gpu_rate, "the trial")


def _key(values: dict) -> tuple:
    """The model and band count of a run: one trial for each in a session (so not again
    for every run of the same room in the editor)."""
    bvh = values["bvh"]
    return (int(bvh.face_count), float(bvh.scale), len(np.asarray(values["air"])))


def trace(plan: Plan, values: dict) -> tuple:
    """``_native.trace_stochastic(**values)`` where ``plan`` says."""
    from acoustic_spaces import _native
    from acoustic_spaces.engine import gpu_tracer

    if plan.device == "gpu":
        try:
            return gpu_tracer.trace_stochastic(**values)
        except gpu_tracer.Unsupported:
            return _native.trace_stochastic(**values)
    if plan.device == "both":
        return _shared(plan, values)
    return _native.trace_stochastic(**values)


def _shared(plan: Plan, values: dict) -> tuple:
    """The rays of ``values`` shared between the CPU and the GPU: each takes the next
    part as it finishes the one before. A part traced on the CPU carries its own share
    of the energy, so its results (and its expected ray energies) are scaled to the
    whole run's."""
    from acoustic_spaces import _native
    from acoustic_spaces.engine import gpu_tracer

    count, first = int(values["rays"]), int(values["first_ray"])
    target = np.asarray(values["split_target"], float)
    threads = _cpu_threads(values)
    total = plan.cpu_rate + plan.gpu_rate
    sizes = {
        "cpu": max(256, math.ceil(count * plan.cpu_rate / total / CPU_PARTS)),
        "gpu": max(4096, math.ceil(count * plan.gpu_rate / total / GPU_PARTS)),
    }
    cursor = [0]
    lock = threading.Lock()
    parts: list[tuple[float, tuple]] = []
    errors: list[BaseException] = []

    def take(device: str) -> tuple[int, int]:
        with lock:
            lo = cursor[0]
            cursor[0] = hi = min(count, lo + sizes[device])
        return lo, hi

    def on_cpu(lo: int, hi: int) -> None:
        m = hi - lo
        part = _native.trace_stochastic(
            **values
            | {
                "rays": m,
                "first_ray": first + lo,
                "split_target": target * (count / m),
                "threads": max(1, threads - 1),
            }
        )
        parts.append((m / count, part))

    def work(device: str) -> None:
        try:
            while not errors:
                lo, hi = take(device)
                if lo >= hi:
                    return
                if device == "cpu":
                    on_cpu(lo, hi)
                    continue
                try:
                    part = gpu_tracer.trace_stochastic(
                        **values | {"rays": hi - lo, "first_ray": first + lo}, total=count
                    )
                except gpu_tracer.Unsupported:
                    on_cpu(lo, hi)
                    continue
                parts.append((1.0, part))
        except BaseException as error:  # noqa: BLE001 - handed to the caller below
            errors.append(error)

    helper = threading.Thread(target=work, args=("gpu",), name="acoustic-spaces-gpu")
    helper.start()
    work("cpu")
    helper.join()
    if errors:
        raise errors[0]
    scales = [scale for scale, _ in parts]
    results = [part for _, part in parts]

    def total_of(i: int, scaled: bool = True):
        return sum((s if scaled else 1.0) * r[i] for s, r in zip(scales, results, strict=True))

    return (
        total_of(0),
        total_of(1),
        total_of(2),
        int(total_of(3, scaled=False)),
        total_of(4),
        total_of(5, scaled=False),
        int(total_of(6, scaled=False)),
    )


__all__ = [
    "DEVICES",
    "TRIAL_FLAG",
    "Device",
    "Plan",
    "choose",
    "saved_choice",
    "set_trial",
    "trace",
    "trial",
    "trial_enabled",
]
