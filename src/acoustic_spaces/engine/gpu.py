"""The wave solver's time steps on a GPU, through WebGPU.

:class:`GpuWaveSolver` takes a grid that the C++ solver
(:class:`acoustic_spaces._native.WaveSolver`) has set up, with its air, boundaries,
branches, matched layer, source and probes, and a speed of sound per cell where the
grid has one (outdoors in wind and temperature), and steps it with one WGSL compute shader on
any GPU that wgpu reaches: Vulkan (AMD, Intel, NVIDIA, and Mesa's software lavapipe),
Metal or Direct3D 12. It needs the optional ``gpu`` extra (``pip install
"acoustic-spaces[gpu]"``). Its arithmetic is the CPU's, in single precision, so the two
agree to rounding.

Each time step is one dispatch: one thread for each cell of the grid (the interior air
cells take the plain update and the rest return), then one for each boundary cell, each
matched-layer cell and each probe. The two pressure fields swap from one step to the
next, and so do the two sets of branch velocities, by the step's parity.
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import Literal

import numpy as np

#: Where the wave solver runs: ``"auto"`` takes a GPU when wgpu finds a hardware one
#: (and ``acoustic-spaces tune`` did not find the CPU faster for the run's length).
Device = Literal["cpu", "gpu", "auto"]

#: Threads per workgroup, and time steps per command buffer.
_GROUP = 256
_BATCH = 512
#: Cell updates per submission: about a second's work on an integrated GPU, well inside
#: the watchdog that resets a GPU whose job runs too long (10 s on Linux's amdgpu).
_SUBMIT_CELL_STEPS = 2**26
#: Workgroups per dispatch dimension WebGPU always allows.
_MAX_GROUPS = 65535

_SHADER = """
struct Grid {
    cells: u32,
    sy: u32,
    sz: u32,
    specials: u32,
    layers: u32,
    probes: u32,
    nx: u32,
    ny: u32,
    special_at: u32,
    state_at: u32,
    layer_at: u32,
    coefficient_at: u32,
    probe_at: u32,
    probe_cells_at: u32,
    branches: u32,
    layer_state_at: u32,
    link_at: u32,
    offset_at: u32,
    wide: u32,
    scaled: u32,
    scale_at: u32,
    layer_scale_at: u32,
    axial: f32,
    diagonal: f32,
    centre: f32,
    total: f32,
}

struct Step {
    index: u32,
    odd: u32,
    source: f32,
}

@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<uniform> now: Step;
// The pressure a step before, overwritten with the next; and the current pressure.
@group(0) @binding(2) var<storage, read_write> next: array<f32>;
@group(0) @binding(3) var<storage, read> current: array<f32>;
// The layout: cell types (four to a word), then the tables the offsets in `grid` give.
@group(0) @binding(4) var<storage, read> table: array<u32>;
// Two sets of branch velocities, then the matched layer's state.
@group(0) @binding(5) var<storage, read_write> state: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

fn number(at: u32) -> f32 {
    return bitcast<f32>(table[at]);
}

fn interior(i: u32) {
    if (i < grid.sz || i + grid.sz >= grid.cells) {
        return;
    }
    if (((table[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu) != 1u) {
        return;
    }
    let sy = grid.sy;
    let sz = grid.sz;
    let sum = current[i - 1u] + current[i + 1u] + current[i - sy] + current[i + sy] +
              current[i - sz] + current[i + sz];
    // With a speed of sound per cell, the cell's links weigh s times as much (the
    // square of its speed over the grid's) and its centre 2 - total * s, as on the CPU.
    var scaled = grid.scaled != 0u;
    var s = 1.0;
    if (scaled) {
        s = number(grid.scale_at + i);
    }
    if (grid.wide == 0u) {
        if (scaled) {
            next[i] = (2.0 - grid.total * s) * current[i] + (s * grid.axial) * sum - next[i];
        } else {
            next[i] = grid.axial * sum - next[i];
        }
        return;
    }
    let edges = current[i - 1u - sy] + current[i + 1u - sy] + current[i - 1u + sy] +
                current[i + 1u + sy] + current[i - 1u - sz] + current[i + 1u - sz] +
                current[i - 1u + sz] + current[i + 1u + sz] + current[i - sy - sz] +
                current[i + sy - sz] + current[i - sy + sz] + current[i + sy + sz];
    if (scaled) {
        next[i] = (2.0 - grid.total * s) * current[i] + (s * grid.axial) * sum +
                  (s * grid.diagonal) * edges - next[i];
        return;
    }
    next[i] = grid.centre * current[i] + grid.axial * sum + grid.diagonal * edges - next[i];
}

fn special(k: u32) {
    let at = grid.special_at + 9u * k;
    let i = table[at];
    let first = table[at + 3u];
    let last = table[at + 4u];
    // Its links: a direction (whose offset wraps round as a u32) and a weight each.
    var sum = 0.0;
    for (var l = table[at + 1u]; l < table[at + 2u]; l++) {
        let e = grid.link_at + 2u * l;
        sum += number(e + 1u) * current[i + table[grid.offset_at + table[e]]];
    }
    // The branch velocities a step before (then the next) and now.
    let before_at = select(0u, grid.branches, now.odd != 0u);
    let now_at = grid.branches - before_at;
    var flow = 0.0;
    for (var j = first; j < last; j++) {
        let c = grid.state_at + 4u * j;
        flow += number(c) * (number(c + 1u) * state[now_at + j] -
                             number(c + 2u) * state[before_at + j]);
    }
    let previous = next[i];
    let value = (number(at + 5u) * current[i] + sum - number(at + 6u) * previous - flow +
                 number(at + 8u) * now.source) * number(at + 7u);
    next[i] = value;
    for (var j = first; j < last; j++) {
        let c = grid.state_at + 4u * j;
        state[before_at + j] += (value - previous) * number(c + 3u) +
                                (number(c + 1u) * state[now_at + j] -
                                 number(c + 2u) * state[before_at + j]);
    }
}

fn layer(k: u32) {
    let i = table[grid.layer_at + k];
    var at = array<u32, 3>(i % grid.nx, (i / grid.nx) % grid.ny, i / grid.sz);
    var strides = array<u32, 3>(1u, grid.sy, grid.sz);
    var starts = array<u32, 3>(0u, grid.nx + 1u, grid.nx + grid.ny + 2u);
    let s = grid.layer_state_at + 9u * k;
    // With a speed of sound per cell, the cell's push weighs its scale, as on the CPU.
    var scale = 1.0;
    if (grid.scaled != 0u) {
        scale = number(grid.layer_scale_at + k);
    }
    var total = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let stride = strides[axis];
        // The centre's then the face's keep and push, at the cell and the one above.
        let c = grid.coefficient_at + 4u * (starts[axis] + at[axis]);
        let below = number(c + 2u) * state[s + 3u + 2u * axis] -
                    number(c + 3u) * (current[i] - current[i - stride]);
        let above = number(c + 6u) * state[s + 4u + 2u * axis] -
                    number(c + 7u) * (current[i + stride] - current[i]);
        state[s + 3u + 2u * axis] = below;
        state[s + 4u + 2u * axis] = above;
        var part = number(c) * state[s + axis] - number(c + 1u) * (above - below);
        if (grid.scaled != 0u) {
            part = number(c) * state[s + axis] - (number(c + 1u) * scale) * (above - below);
        }
        state[s + axis] = part;
        total += part;
    }
    next[i] = total;
}

fn record(k: u32) {
    let first = table[grid.probe_at + k];
    let last = table[grid.probe_at + k + 1u];
    var value = 0.0;
    for (var c = first; c < last; c++) {
        let e = grid.probe_cells_at + 2u * c;
        value += number(e + 1u) * current[table[e]];
    }
    out[now.index * grid.probes + k] = value;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    var t = id.x + id.y * groups.x * 256u;
    if (t < grid.cells) {
        interior(t);
        return;
    }
    t -= grid.cells;
    if (t < grid.specials) {
        special(t);
        return;
    }
    t -= grid.specials;
    if (t < grid.layers) {
        layer(t);
        return;
    }
    t -= grid.layers;
    if (t < grid.probes) {
        record(t);
    }
}
"""


def _wgpu(what: str = "the wave solver"):
    try:
        import wgpu
    except ImportError as error:  # pragma: no cover - depends on the install
        raise RuntimeError(
            f'{what} needs wgpu to run on a GPU: pip install "acoustic-spaces[gpu]"'
        ) from error
    return wgpu


def _not_the_vulkan_loader(record: logging.LogRecord) -> bool:
    """Whether wgpu's log ``record`` is anything but a note of the Vulkan loader's, or the
    line after it naming the Vulkan instance it is about."""
    message = record.getMessage()
    return "[Loader Message" not in message and not message.startswith(
        "\tobjects: (type: INSTANCE,"
    )


def _primary_backends() -> None:
    """Have wgpu use Vulkan, Metal or Direct3D 12 only, never its OpenGL backend: setting
    that up through EGL aborts the whole process (a Rust panic Python cannot catch) when
    the editor's 3D view already holds an OpenGL context. On Windows, Direct3D 12's
    shader compiler (FXC) also goes without its optimizer (wgpu's debug flag), which
    miscompiled the ray tracer's vegetation: rays scattered in it came out where they
    had started. The driver optimizes the shader anyway, and the compile takes no
    longer. (Only flags every wgpu from 0.31 on knows: an unknown one sets nothing.)
    The debug flag also has wgpu pass on the Vulkan loader's notes, such as finding no
    driver on a computer with Direct3D 12 only, which say nothing to a user: those are
    left out of wgpu's log."""
    flags = ["Debug"] if sys.platform == "win32" else ["Default"]
    if sys.platform == "win32":
        logging.getLogger("wgpu").addFilter(_not_the_vulkan_loader)
    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        set_instance_extras(backends=["Primary"], flags=flags)
    except (ImportError, KeyError, RuntimeError):  # another backend, or set up already
        pass


@functools.lru_cache(maxsize=1)
def _adapter():
    """The GPU wgpu picks (a hardware one first), or None."""
    try:
        import wgpu
    except ImportError:
        return None
    _primary_backends()
    try:
        return wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception:  # no driver, or none wgpu can use
        return None


@functools.lru_cache(maxsize=1)
def _device():
    adapter = _adapter()
    if adapter is None:
        raise RuntimeError("wgpu finds no GPU")
    names = ("max-storage-buffer-binding-size", "max-buffer-size")
    return adapter.request_device_sync(
        label="acoustic-spaces",
        required_limits={name: adapter.limits[name] for name in names},
    )


def gpu_name() -> str | None:
    """The GPU the wave solver would run on, as ``"<device> (<backend>, <type>)"``, or
    None when wgpu is missing or finds none."""
    adapter = _adapter()
    if adapter is None:
        return None
    info = adapter.info
    return f"{info['device']} ({info['backend_type']}, {info['adapter_type']})"


def hardware_gpu() -> bool:
    """Whether wgpu finds a hardware GPU (not a software renderer such as lavapipe)."""
    adapter = _adapter()
    return adapter is not None and adapter.info["adapter_type"] != "CPU"


def use_gpu(device: Device, steps: int | None = None) -> bool:
    """Whether the wave solver runs on the GPU for the setting ``device``, for a run of
    ``steps`` steps. ``auto`` takes a hardware GPU: when ``acoustic-spaces tune`` timed
    both on this computer, only if it steps that many times clearly faster (see
    :mod:`acoustic_spaces.engine.tuning`)."""
    if device == "cpu":
        return False
    if device == "gpu":
        if _adapter() is None:
            try:
                import wgpu  # noqa: F401
            except ImportError:
                raise ValueError(
                    "simulation.wave.device is gpu, which needs wgpu: pip install "
                    '"acoustic-spaces[gpu]"'
                ) from None
            raise ValueError(
                "simulation.wave.device is gpu, but wgpu finds no GPU: install a Vulkan, "
                "Metal or Direct3D 12 driver, or set it to cpu or auto"
            )
        return True
    if device == "auto":
        if not hardware_gpu():
            return False
        from acoustic_spaces.engine import tuning

        faster = None if steps is None else tuning.wave_prefers_gpu(steps)
        return True if faster is None else faster
    raise ValueError(f"the wave solver's device must be cpu, gpu or auto, not {device!r}")


#: The directions of the C++ solver's neighbours (fdtd.hpp): six axial, twelve edge.
_STEPS = (
    (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
    (-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0), (-1, 0, -1), (1, 0, -1),
    (-1, 0, 1), (1, 0, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1), (0, 1, 1),
)  # fmt: skip


def _words(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def _tables(layout: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    """The layout packed as 32-bit words, and where each table starts."""
    n = int(np.prod(layout["shape"]))
    types = np.zeros(4 * ((n + 3) // 4), np.uint8)
    types[:n] = layout["types"]
    special = layout["special"]
    start = layout["state_start"]
    links = layout["link_start"]
    specials = np.column_stack(
        [
            special.astype(np.uint32),
            links[:-1].astype(np.uint32),
            links[1:].astype(np.uint32),
            start[:-1].astype(np.uint32),
            start[1:].astype(np.uint32),
            _words(layout["coef_a"]),
            _words(layout["coef_b"]),
            _words(layout["coef_inv"]),
            _words(layout["source"]),
        ]
    )
    branch = layout["state_branch"]
    states = np.column_stack(
        [
            layout["state_weight"],
            layout["branch_now"][branch],
            layout["branch_before"][branch],
            layout["branch_gain"][branch],
        ]
    )
    coefficients = [
        np.column_stack(
            [
                layout[f"keep_{axis}"],
                layout[f"push_{axis}"],
                layout[f"keep_face_{axis}"],
                layout[f"push_face_{axis}"],
            ]
        ).reshape(-1)
        for axis in "xyz"
        if len(layout[f"keep_{axis}"])
    ]
    nx, ny, _ = (int(v) for v in layout["shape"])
    steps = np.array(_STEPS[: len(layout["weights"])], np.int64)
    offsets = (steps @ np.array([1, nx, nx * ny])).astype(np.int32).view(np.uint32)
    link_table = np.column_stack(
        [layout["link_direction"].astype(np.uint32), _words(layout["link_weight"])]
    )
    probes = np.column_stack(
        [layout["probe_cells"].astype(np.uint32), _words(layout["probe_weights"])]
    )
    parts = {
        "types": types.view(np.uint32),
        "special_at": specials.reshape(-1),
        "state_at": _words(states.reshape(-1)),
        "layer_at": layout["layer_cells"].astype(np.uint32),
        "coefficient_at": _words(np.concatenate(coefficients))
        if coefficients
        else np.zeros(0, np.uint32),
        "probe_at": layout["probe_start"].astype(np.uint32),
        "probe_cells_at": probes.reshape(-1),
        "link_at": link_table.reshape(-1),
        "offset_at": offsets,
        "scale_at": _words(layout["scale"]),
        "layer_scale_at": _words(layout["layer_scale"]),
    }
    offsets, at = {}, 0
    for name, words in parts.items():
        offsets[name] = at
        at += len(words)
    if at >= 2**32:
        raise ValueError("the wave grid is too large for the GPU's 32-bit indices")
    return np.concatenate(list(parts.values())), offsets


class GpuWaveSolver:
    """Steps the grid of ``solver`` (a fresh or reset
    :class:`acoustic_spaces._native.WaveSolver`) on the GPU: :meth:`run` does what the
    solver's ``run`` does, from the GPU's own state."""

    def __init__(self, solver) -> None:
        wgpu = _wgpu()
        self._device = device = _device()
        layout = solver.layout()
        nx, ny, nz = (int(v) for v in layout["shape"])
        n = nx * ny * nz
        table, offsets = _tables(layout)
        self._probes = len(layout["probe_start"]) - 1
        branches = len(layout["state_branch"])
        layers = len(layout["layer_cells"])
        weights = layout["weights"]
        grid = np.zeros(28, np.uint32)
        grid[:22] = [
            n,
            nx,
            nx * ny,
            len(layout["special"]),
            layers,
            self._probes,
            nx,
            ny,
            offsets["special_at"],
            offsets["state_at"],
            offsets["layer_at"],
            offsets["coefficient_at"],
            offsets["probe_at"],
            offsets["probe_cells_at"],
            branches,
            2 * branches,
            offsets["link_at"],
            offsets["offset_at"],
            int(layout["stencil"][0]) == 19,
            int(layout["scaled"][0]),
            offsets["scale_at"],
            offsets["layer_scale_at"],
        ]
        diagonal = weights[6] if len(weights) > 6 else 0.0
        grid[22:26] = _words(
            np.array([weights[0], diagonal, layout["centre"][0], layout["total"][0]])
        )
        self._cells = n
        threads = n + len(layout["special"]) + layers + self._probes
        groups = -(-threads // _GROUP)
        self._groups = (min(groups, _MAX_GROUPS), -(-groups // _MAX_GROUPS))

        limit = device.limits["max-storage-buffer-binding-size"]
        largest = max(4 * n, table.nbytes)
        if largest > limit:
            raise ValueError(
                f"the wave grid needs GPU buffers of {largest / 2**20:,.0f} MiB, and this GPU "
                f"binds at most {limit / 2**20:,.0f} MiB: lower max_band_hz or "
                "points_per_wavelength, or run the wave solver on the CPU"
            )
        usage = wgpu.BufferUsage
        storage = usage.STORAGE | usage.COPY_DST
        self._grid = device.create_buffer_with_data(data=grid, usage=usage.UNIFORM)
        self._table = device.create_buffer_with_data(data=table, usage=usage.STORAGE)
        self._fields = [device.create_buffer(size=4 * n, usage=storage) for _ in range(2)]
        self._state = device.create_buffer(
            size=4 * max(1, 2 * branches + 9 * layers), usage=storage
        )
        self._stride = max(16, device.limits["min-uniform-buffer-offset-alignment"])
        self._steps = device.create_buffer(
            size=self._stride * _BATCH, usage=usage.UNIFORM | usage.COPY_DST
        )

        def entry(binding, kind, dynamic=False):
            return {
                "binding": binding,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": kind, "has_dynamic_offset": dynamic},
            }

        kinds = wgpu.BufferBindingType
        self._layout = device.create_bind_group_layout(
            entries=[
                entry(0, kinds.uniform),
                entry(1, kinds.uniform, dynamic=True),
                entry(2, kinds.storage),
                entry(3, kinds.read_only_storage),
                entry(4, kinds.read_only_storage),
                entry(5, kinds.storage),
                entry(6, kinds.storage),
            ]
        )
        self._pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            compute={"module": device.create_shader_module(code=_SHADER), "entry_point": "main"},
        )
        self._done = 0  # steps run so far: their parity says which field is which

    @property
    def probes(self) -> int:
        return self._probes

    def run(self, signal: np.ndarray) -> np.ndarray:
        """Advances one step per sample of ``signal`` (added at the source) and returns
        the pressure at every probe before each step (probes, steps)."""
        wgpu = _wgpu()
        device, queue = self._device, self._device.queue
        signal = np.ascontiguousarray(signal, dtype=np.float32).reshape(-1)
        steps = len(signal)
        if steps == 0:
            return np.zeros((self._probes, 0))
        size = 4 * max(1, steps * self._probes)
        out = device.create_buffer(
            size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

        def bind(next_field, current_field):
            buffers = [
                self._grid,
                self._steps,
                next_field,
                current_field,
                self._table,
                self._state,
                out,
            ]
            entries = []
            for binding, buffer in enumerate(buffers):
                resource = {"buffer": buffer, "offset": 0, "size": buffer.size}
                if binding == 1:
                    resource["size"] = 16
                entries.append({"binding": binding, "resource": resource})
            return device.create_bind_group(layout=self._layout, entries=entries)

        first, second = self._fields
        groups = (bind(first, second), bind(second, first))
        words = self._stride // 4
        # The steps go in submissions of bounded work: a GPU's watchdog resets it when
        # one runs too long, and every step after is lost.
        per_submit = max(1, min(_BATCH, _SUBMIT_CELL_STEPS // self._cells))
        for begin in range(0, steps, _BATCH):
            count = min(_BATCH, steps - begin)
            index = np.arange(begin, begin + count, dtype=np.uint32)
            entries = np.zeros((count, words), np.uint32)
            entries[:, 0] = index
            entries[:, 1] = (self._done + index) & 1
            entries[:, 2] = _words(signal[begin : begin + count])
            queue.write_buffer(self._steps, 0, entries)
            for start in range(0, count, per_submit):
                encoder = device.create_command_encoder()
                compute = encoder.begin_compute_pass()
                compute.set_pipeline(self._pipeline)
                for k in range(start, min(start + per_submit, count)):
                    compute.set_bind_group(0, groups[entries[k, 1]], [k * self._stride])
                    compute.dispatch_workgroups(*self._groups)
                compute.end()
                queue.submit([encoder.finish()])
        self._done += steps
        recorded = np.frombuffer(queue.read_buffer(out), np.float32)[: steps * self._probes]
        return recorded.reshape(steps, self._probes).T.astype(np.float64)


__all__ = ["Device", "GpuWaveSolver", "gpu_name", "hardware_gpu", "use_gpu"]
