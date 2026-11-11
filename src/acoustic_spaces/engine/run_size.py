"""How big a run is: the threads it works on and the memory its late part needs.

What grows without bound with the settings is the late part's energy: one array per
listener (or map point) and band, as long as the response, and at a listener one more
per arrival direction. Every source's is kept to the end of the run, and while one is
traced the tracer needs more: on the CPU a copy for each thread that sums rays (at most
``_native.COPY_MEMORY`` together, one at least), on a GPU its buffers. So a run needs
memory in proportion to ``simulation.ir_length_s``, the listeners and
``late.directions``, the map points (``coverage.spacing``) and the sources: those traced
for their late part, as a source or a listener whose late part is a mix of others' (a
walk's places between the traced ones, ``simulate``'s ``late_from``) holds none. On the CPU
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
from collections.abc import Callable, Collection, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from acoustic_spaces.materials.bands import band_centres

if TYPE_CHECKING:
    from acoustic_spaces.room import Room
    from acoustic_spaces.scene.model import Simulation

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")

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


def each_on_threads(
    function: Callable[[_Item], _Result], items: Iterable[_Item], threads: int
) -> list[_Result]:
    """``function`` of each of ``items``, in their order, on at most ``threads`` threads
    at once, so that at most ``threads`` of the calls are in progress at any time. With
    one thread (or one item) the calls run one after another in the calling thread,
    with no pool."""
    todo = list(items)
    if threads <= 1 or len(todo) <= 1:
        return [function(item) for item in todo]
    with ThreadPoolExecutor(min(int(threads), len(todo))) as pool:
        return list(pool.map(function, todo))


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
    estimates: float = 0.0,
) -> float:
    """Bytes one run of the late part's tracer takes at its peak, whose arrays take
    ``results`` bytes in ``bands`` bands traced in ``groups`` groups of bands
    (:func:`~acoustic_spaces.engine.late.band_groups`), and the energy as each batch of
    rays gives it ``estimates``: the arrays the groups are joined in, the sums of the
    runs before while the next is traced (``splitting``: a first batch, or batches
    traced in runs), a group's estimates, and a group's own arrays, on the worst of
    ``devices``: on the CPU with its threads' copies (at most ``_native.COPY_MEMORY``
    together, one at least), on a GPU with its buffers
    (:func:`~acoustic_spaces.engine.gpu_tracer.run_memory`), or on both with the sums of
    the parts and a part on each."""
    group = results / groups
    joined = results + estimates if groups > 1 else 0.0
    first = group + estimates / groups if splitting else 0.0
    workers = worker_threads(threads)
    from acoustic_spaces import _native

    def cpu(workers: int) -> float:
        # The first thread's copy takes the others' sums and is the call's result.
        return min(workers, max(1, int(_native.COPY_MEMORY // max(group, 1.0)))) * group

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
    mixed: Collection[str] = (),
) -> Need:
    """The memory the late part of a simulation of ``sources`` at ``listeners`` needs:
    the energy of every source, kept to the end; that of a line or surface source's
    points summed as each is traced; and while one is traced, the tracer's own
    (:func:`late_call`, on the devices :func:`late_devices` gives). Then, for a source
    with a feed of its own, a listener's copy at a time while its response takes it (four
    arrays as large at most). ``groups`` face groups (``simulate(groups=...)``) add
    their moments to each listener's arrays, and batches of rays (``late.batches``)
    the energy as each gives it.

    ``mixed`` names the sources and the listeners whose late part is a mix of others'
    (``simulate``'s ``late_from``, with the stochastic ray tracer): such a source is not
    traced and keeps no energy, and a point source's energy is traced and kept at the
    other listeners alone (a line's or a surface's is traced at those and mixed for
    every listener as its points are summed).

    Its advice gives what the run would need with each setting it may lower changed
    on its own (:func:`_lowered`), the least first."""
    stochastic = settings.late.method != "statistical"
    # The same for every setting lowered: the bands' groups and the devices.
    tracer = (
        (_band_groups(room, settings), late_devices(settings.late.device)) if stochastic else None
    )

    def found(settings: Simulation, sources: Sequence, listeners: Sequence) -> tuple[float, str]:
        return _simulation_bytes(room, settings, sources, listeners, threads, groups, tracer, mixed)

    def bytes_of(settings: Simulation, sources: Sequence, listeners: Sequence) -> float:
        return found(settings, sources, listeners)[0]

    need, what = found(settings, sources, listeners)
    options = sorted(
        (bytes_of(*changed), label) for label, changed in _lowered(settings, sources, listeners)
    )
    sizes = [f"{_size(size)} with {label}" for size, label in options]
    advice = ", ".join(sizes[:-1]) + (" or " if len(sizes) > 1 else "") + sizes[-1]
    return Need(need, what, f"it would need about {advice}")


def _lowered(settings: Simulation, sources: Sequence, listeners: Sequence) -> list:
    """The settings a simulation's late part may lower, each changed on its own: as
    (label, (settings, sources, listeners)). ``simulation.ir_length_s`` and
    ``late.directions`` halved, ``late.batches`` 1, and half the listeners and the
    sources (the first of them), where there are more than one; a walk's places are
    sources or listeners of their own."""
    late = settings.late
    length = settings.ir_length_s / 2.0

    def with_late(**update) -> Simulation:
        return settings.model_copy(update={"late": late.model_copy(update=update)})

    found: list = [
        (
            f"simulation.ir_length_s {length:g}",
            (settings.model_copy(update={"ir_length_s": length}), sources, listeners),
        )
    ]
    if late.method != "statistical":
        if late.directions > 1:
            fewer = late.directions // 2
            found.append(
                (
                    f"simulation.late.directions {fewer}",
                    (with_late(directions=fewer), sources, listeners),
                )
            )
        if late.batches > 1:
            found.append(
                (
                    "simulation.late.batches 1 (no standard errors)",
                    (with_late(batches=1), sources, listeners),
                )
            )
    if len(listeners) > 1:
        half = len(listeners) // 2
        found.append(
            (f"{half} of the {len(listeners)} listeners", (settings, sources, listeners[:half]))
        )
    if len(sources) > 1:
        half = len(sources) // 2
        found.append(
            (
                f"{half} of the {len(sources)} sources (a walk's places count as sources)",
                (settings, sources[:half], listeners),
            )
        )
    return found


def _simulation_bytes(
    room: Room,
    settings: Simulation,
    sources: Sequence,
    listeners: Sequence,
    threads: int,
    groups: int,
    tracer: tuple[int, tuple[str, ...]] | None,
    mixed: Collection[str] = (),
) -> tuple[float, str]:
    """:func:`simulation_need`'s bytes and what they are for; ``tracer`` is how many
    groups of bands the late tracer traces in and its devices (:func:`late_devices`),
    None for the statistical late method, which gives everyone a late part of their own
    whatever ``mixed`` names."""
    from acoustic_spaces.engine.late import BATCH_STEP, DIRECTION_STEP, MOMENT_STEP
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
    # Each fed source's responses: its late part copied for a listener at a time.
    feeding = 4.0 / count if any(is_fed(source) for source in sources) else 0.0
    what = f"the late part of a simulation at {count} {noun} ({length:.4g} s long"
    if tracer is None:
        held = max(0, len(sources) - 1) + int(max(elements) > 1)
        results = float(count * bands * slots * VALUE_BYTES)
        need = max((held + 1) * results, (len(sources) + feeding) * results)
        return need, what + ")"
    band_groups, devices = tracer
    directions = late.directions
    values = slots + directions * math.ceil(length / DIRECTION_STEP - 1e-9)
    if groups > 0:
        terms = 1 + groups + groups * (groups + 1) // 2
        values += terms * math.ceil(length / MOMENT_STEP - 1e-9)
    results = float(count * bands * values * VALUE_BYTES)
    coarse = math.ceil(length / BATCH_STEP - 1e-9)
    estimates = float(count * bands * late.batches * coarse * VALUE_BYTES)
    if late.batches < 2:
        estimates = 0.0
    # The share of the listeners that the late part is traced at: those whose own is
    # not a mix of others'.
    traced = sum(1 for listener in listeners if listener.name not in mixed) or count
    share = traced / count
    call = late_call(
        share * results,
        bands,
        band_groups,
        # Batches traced in runs hold their sums while the next run is traced.
        splitting=late.splitting or late.batches > 1,
        threads=threads,
        rays=late.rays,
        devices=devices,
        estimates=share * estimates,
    )
    results += estimates
    what += f", {directions} arrival directions)"
    # What each source traced keeps to the end, in units of one source's energy at every
    # listener (``results``), and what the run takes besides the other sources' while it
    # is traced: so many units and bytes.
    kept: list[float] = []
    besides: list[tuple[float, float]] = []
    copied = 0.0
    for source, points in zip(sources, elements, strict=False):
        if source.name in mixed:
            continue
        if points == 1:
            # A point source's energy stays at the listeners it was traced at.
            kept.append(share)
            besides.append((0.0, call))
        else:
            # A line's or a surface's: its points' sum at every listener, and after the
            # tracer each point's energy as traced with what it mixes to for them all.
            kept.append(1.0)
            besides.append((1.0, max(call, (share + 1.0) * results if share < 1.0 else 0.0)))
        if is_fed(source) and (points > 1 or share == 1.0):
            copied += kept[-1]
    while_traced = [
        (sum(kept) - one + units) * results + more
        for one, (units, more) in zip(kept, besides, strict=True)
    ]
    # Where another source's late part is a mix of theirs, the sources' energy stays for
    # it beside the copies that their own responses take with a feed.
    if not any(source.name in mixed for source in sources):
        copied = 0.0
    need = max(max(while_traced or [call]), (sum(kept) + copied + feeding) * results)
    return need, what


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
    "each_on_threads",
    "held_bytes",
    "late_call",
    "late_devices",
    "machine_memory",
    "map_need",
    "simulation_need",
    "warning",
    "worker_threads",
]
