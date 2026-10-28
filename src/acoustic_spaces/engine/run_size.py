"""How big a run is: the threads it works on and the memory its late part needs.

What grows without bound with the settings is the late part's energy: one array per
listener (or map point) and band, as long as the response, and at a listener one more
per arrival direction. Every source's is kept to the end of the run, and while one is
traced the tracer needs more: on the CPU a copy for each thread that sums rays (at most
``_native.COPY_MEMORY`` together, one at least), on a GPU its buffers. So a run needs
memory in proportion to ``simulation.ir_length_s``, the listeners and
``late.directions``, the map points (``coverage.spacing``) and the sources. On the CPU
the rays cost time, not memory. A run whose late part would need more memory than the
computer has is refused before it starts (:func:`check`), and
:meth:`~acoustic_spaces.room.Room.warnings` notes one that would need more than half of
it. A map is checked again once its first source point is traced, with the early
paths found there at every point (:func:`held_bytes`). The estimate takes the threads
and the devices of the computer it runs on: with ``late.device`` gpu, both or auto
(where there is a hardware GPU), the worst of the CPU, the GPU and the two together.

Threads above the computer's cores run as its cores: more would only take turns on
them, each with its copy (:func:`worker_threads`).
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from acoustic_spaces.materials.bands import band_centres

if TYPE_CHECKING:
    from acoustic_spaces.room import Room
    from acoustic_spaces.scene.model import Simulation

#: Bytes of one value of the late part's arrays.
VALUE_BYTES = 8

# Where Linux keeps its control groups, which set a container's memory limit.
_CGROUPS = Path("/sys/fs/cgroup")


def machine_memory() -> int | None:
    """Bytes of memory this computer has, or the limit of the container the run is in
    where that is lower; None where neither can be read."""
    total = _windows_memory() if sys.platform == "win32" else _physical_memory()
    try:
        groups = Path("/proc/self/cgroup").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        groups = ""
    limit = _container_limit(groups)
    if limit is not None:
        total = limit if total is None else min(total, limit)
    return total if total else None


def _container_limit(groups: str, root: Path = _CGROUPS) -> int | None:
    """The lowest memory limit (bytes) on this process's control groups and the groups
    above them, from ``/proc/self/cgroup``'s ``groups``: a container's may be on any of
    them, not only on the root (``memory.max`` in version 2, ``memory.limit_in_bytes``
    in version 1's memory controller); None where none is set."""
    limit = None
    for line in groups.splitlines():
        fields = line.split(":", 2)
        if len(fields) < 3:
            continue
        number, controllers, path = fields
        if number == "0" and not controllers:
            base, name = root, "memory.max"
        elif "memory" in controllers.split(","):
            base, name = root / "memory", "memory.limit_in_bytes"
        else:
            continue
        parts = [part for part in path.split("/") if part]
        for depth in range(len(parts) + 1):
            try:
                text = base.joinpath(*parts[:depth], name).read_text(encoding="ascii").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if text.isdigit():
                limit = int(text) if limit is None else min(limit, int(text))
    return limit


def _physical_memory() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _windows_memory() -> int | None:
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("free_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("free_virtual", ctypes.c_ulonglong),
            ("free_extended_virtual", ctypes.c_ulonglong),
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
    return int(status.total) if kernel.GlobalMemoryStatusEx(ctypes.byref(status)) else None


def worker_threads(threads: int) -> int:
    """The threads a run works on for ``threads`` asked (``performance.threads``): as
    many as the computer has cores for 0 or more."""
    cores = os.cpu_count() or 1
    return cores if threads <= 0 else min(int(threads), cores)


@dataclass(frozen=True)
class Need:
    """Bytes of memory a run's late part needs, what for and what to change for less."""

    bytes: float
    what: str
    advice: str

    def note(self, memory: int) -> str:
        """What the run needs against ``memory`` bytes (the computer's)."""
        than = "more than" if self.bytes > memory else "over half of"
        return (
            f"{self.what} needs about {_size(self.bytes)} of memory, {than} this "
            f"computer's {_size(memory)}; {self.advice}"
        )


def _size(count: float) -> str:
    gib = count / 2.0**30
    return f"{gib:,.1f} GiB" if gib >= 1.0 else f"{count / 2.0**20:,.0f} MiB"


def held_bytes(value: object) -> float:
    """Bytes the arrays in ``value`` take (in lists, tuples and dataclasses, each array
    once, with its header), such as a map's early paths at every point."""
    seen: set[int] = set()

    def size(item: object) -> float:
        if item is None or id(item) in seen:
            return 0.0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            return sys.getsizeof(item) + (0.0 if item.base is None else size(item.base))
        if isinstance(item, (list, tuple)):
            return sys.getsizeof(item) + sum(size(v) for v in item)
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return sum(size(getattr(item, f.name)) for f in dataclasses.fields(item))
        return 0.0

    return size(value)


def check(need: Need, memory: int | None = None) -> None:
    """Raises ValueError when ``need`` is more than ``memory`` bytes (by default the
    computer's, :func:`machine_memory`)."""
    memory = machine_memory() if memory is None else memory
    if memory is not None and need.bytes > memory:
        raise ValueError(need.note(memory))


def warning(need: Need, memory: int | None = None) -> str | None:
    """A note when ``need`` is more than half of ``memory`` bytes (by default the
    computer's), else None."""
    memory = machine_memory() if memory is None else memory
    if memory is None or need.bytes <= 0.5 * memory:
        return None
    return need.note(memory)


def late_call(
    results: float,
    bands: int,
    groups: int,
    *,
    splitting: bool,
    threads: int,
    rays: int,
    devices: Sequence[str] = ("cpu",),
) -> float:
    """Bytes one run of the late part's tracer takes at its peak, whose arrays take
    ``results`` bytes in ``bands`` bands traced in ``groups`` groups of bands
    (:func:`~acoustic_spaces.engine.late.band_groups`): the arrays the groups are joined
    in, the first batch's while the second is traced (``splitting``), and a group's
    own, on the worst of ``devices``: on the CPU with its threads' copies (at most
    ``_native.COPY_MEMORY`` together, one at least), on a GPU with its buffers
    (:func:`~acoustic_spaces.engine.gpu_tracer.run_memory`), or on both with the sums of
    the parts and a part on each."""
    group = results / groups
    joined = results if groups > 1 else 0.0
    first = group if splitting else 0.0
    workers = worker_threads(threads)
    from acoustic_spaces import _native

    def cpu(workers: int) -> float:
        copies = min(workers, max(1, int(_native.COPY_MEMORY // max(group, 1.0))))
        return (1 + copies) * group

    peaks = [cpu(workers)]
    if "gpu" in devices or "both" in devices:
        from acoustic_spaces.engine.gpu_tracer import run_memory

        on_gpu = run_memory(rays, bands // groups, group)
        peaks.append(on_gpu)
        if "both" in devices:
            peaks.append(group + cpu(max(1, workers - 1)) + on_gpu)
    return joined + first + max(peaks)


def late_devices(setting: str | None) -> tuple[str, ...]:
    """Where the late part may trace its rays on this computer for ``late.device``:
    always on the CPU, which traces what a GPU declines; on the GPU for gpu; and on
    both too for both, and for auto where there is a hardware GPU."""
    from acoustic_spaces.engine import gpu, tuning

    device = tuning.resolve_device(setting)
    if device == "cpu" or (device == "auto" and not gpu.hardware_gpu()):
        return ("cpu",)
    return ("cpu", "gpu") if device == "gpu" else ("cpu", "gpu", "both")


def _band_groups(room: Room, settings: Simulation) -> int:
    """How many groups of bands the late tracer traces ``room``'s surfaces in."""
    from acoustic_spaces.engine.late import group_bands

    used = np.unique(room.face_materials[room.face_materials >= 0])
    if len(used) == 0:
        return 1
    materials = room.scene.materials
    coefficients = [materials[room.material_names[k]].coefficients(settings.bands) for k in used]
    scattering = settings.scattering
    found = group_bands(
        np.array([a for a, _ in coefficients]),
        np.array([s for _, s in coefficients]),
        sized=scattering.from_size,
        vector=scattering.method == "vector",
        by_band=scattering.by_band,
    )
    return len(found)


def simulation_need(
    room: Room,
    settings: Simulation,
    sources: Sequence,
    listeners: Sequence,
    threads: int,
    groups: int = 0,
) -> Need:
    """The memory the late part of a simulation of ``sources`` at ``listeners`` needs:
    the energy of every source, kept to the end; that of a line or surface source's
    points summed as each is traced; and while one is traced, the tracer's own
    (:func:`late_call`, on the devices :func:`late_devices` gives). Then, for a source
    with a feed of its own, a listener's copy at a time while its response takes it (four
    arrays as large at most). ``groups`` face groups (``simulate(groups=...)``) add
    their moments to each listener's arrays."""
    from acoustic_spaces.engine.late import DIRECTION_STEP, MOMENT_STEP
    from acoustic_spaces.engine.simulate import TIME_STEP, is_fed, response_length

    bands = len(band_centres(settings.bands))
    elements = [len(source.elements()) for source in sources] or [1]
    starts = [point for source in sources for point in source.elements()]
    ends = [listener.position for listener in listeners]
    length = response_length(room, settings.ir_length_s, starts or [[0, 0, 0]], ends)
    count = max(1, len(listeners))
    slots = math.ceil(length / TIME_STEP - 1e-9)
    late = settings.late
    noun = "listener" if count == 1 else "listeners"
    summed = max(elements) > 1
    held = max(0, len(sources) - 1) + int(summed)
    # Each fed source's responses: its late part copied for a listener at a time.
    feeding = 4.0 / count if any(is_fed(source) for source in sources) else 0.0
    what = f"the late part of a simulation at {count} {noun} ({length:.4g} s long"
    advice = "lower simulation.ir_length_s, the listeners or the sources"
    if late.method == "statistical":
        results = float(count * bands * slots * VALUE_BYTES)
        need = max((held + 1) * results, (len(sources) + feeding) * results)
        return Need(need, what + ")", advice)
    directions = late.directions
    values = slots + directions * math.ceil(length / DIRECTION_STEP - 1e-9)
    if groups > 0:
        terms = 1 + groups + groups * (groups + 1) // 2
        values += terms * math.ceil(length / MOMENT_STEP - 1e-9)
    results = float(count * bands * values * VALUE_BYTES)
    call = late_call(
        results,
        bands,
        _band_groups(room, settings),
        splitting=late.splitting,
        threads=threads,
        rays=late.rays,
        devices=late_devices(late.device),
    )
    what += f", {directions} arrival directions)"
    advice = (
        "lower simulation.ir_length_s, simulation.late.directions, the listeners or the sources"
    )
    need = max(held * results + call, (len(sources) + feeding) * results)
    return Need(need, what, advice)


def map_need(
    room: Room,
    settings: Simulation,
    sources: Sequence,
    points: int,
    corners: np.ndarray,
    grid: float,
    threads: int,
) -> Need:
    """The memory a coverage map of about ``points`` points within ``corners`` (those
    of its extent) needs: ``grid`` bytes while its grid is laid out, and with stochastic
    ray tracing the late energy at every point from each point of every source, kept to
    the end, and while one is traced, the tracer's own (:func:`late_call`; maps are
    traced on the CPU)."""
    from acoustic_spaces.engine.late import MAP_COARSE_STEP, MAP_FINE_BINS, MAP_FINE_STEP
    from acoustic_spaces.engine.simulate import response_length

    what = f"a coverage map of about {points:,} points"
    advice = "raise coverage.spacing, or lower simulation.ir_length_s"
    if settings.late.method == "statistical":
        return Need(grid, what, advice)
    bands = len(band_centres(settings.bands))
    starts = [point for source in sources for point in source.elements()]
    length = response_length(room, settings.ir_length_s, starts or [[0, 0, 0]], corners)
    rest = max(0.0, length - MAP_FINE_STEP * MAP_FINE_BINS)
    slots = MAP_FINE_BINS + math.ceil(rest / MAP_COARSE_STEP - 1e-9)
    results = float(points * bands * slots * VALUE_BYTES)
    call = late_call(
        results,
        bands,
        _band_groups(room, settings),
        splitting=settings.late.splitting,
        threads=threads,
        rays=settings.late.rays,
    )
    elements = sum(len(source.elements()) for source in sources)
    what += f" ({length:.4g} s long)"
    return Need(max(grid, max(0, elements - 1) * results + call), what, advice)


__all__ = [
    "Need",
    "check",
    "held_bytes",
    "late_call",
    "late_devices",
    "machine_memory",
    "map_need",
    "simulation_need",
    "warning",
    "worker_threads",
]
