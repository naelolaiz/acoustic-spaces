"""Temporary: what the GPU late tracer does on this runner's adapter, against the CPU.
Prints as it goes and gives up on a run after a few hundred rounds."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_native import _unit_cube  # noqa: E402
from test_tracer_reference import REFERENCE, reference_arguments  # noqa: E402

from acoustic_spaces import _native  # noqa: E402
from acoustic_spaces.engine import gpu, gpu_tracer  # noqa: E402

ROUNDS = 300


class _Numpy:
    """numpy for gpu_tracer, printing each round's counters."""

    def __init__(self) -> None:
        self.rounds = 0
        self.started = time.perf_counter()

    def __getattr__(self, name):
        return getattr(np, name)

    def frombuffer(self, buffer, dtype=float, *args, **kwargs):
        out = np.frombuffer(buffer, dtype, *args, **kwargs)
        if len(out) == gpu_tracer._COUNTERS:
            self.rounds += 1
            if self.rounds <= 12 or self.rounds % 50 == 0:
                took = time.perf_counter() - self.started
                print(f"    round {self.rounds}: counters {out[:5].tolist()} at {took:.1f} s")
            if self.rounds >= ROUNDS:
                raise RuntimeError(f"still going after {ROUNDS} rounds")
        return out


def _run(label: str, values: dict):
    print(f"{label}:", flush=True)
    started = time.perf_counter()
    on_cpu = _native.trace_stochastic(**values)
    print(f"  cpu {time.perf_counter() - started:.2f} s", flush=True)
    gpu_tracer.np = _Numpy()
    started = time.perf_counter()
    try:
        on_gpu = gpu_tracer.trace_stochastic(**values)
    except Exception as error:  # noqa: BLE001
        print(f"  gpu failed: {error!r}", flush=True)
        return on_cpu, None
    finally:
        gpu_tracer.np = np
    print(f"  gpu {time.perf_counter() - started:.2f} s", flush=True)
    for i, name in ((0, "energy"), (1, "directional")):
        a, b = on_cpu[i], on_gpu[i]
        per = a.reshape(a.shape[0], -1).sum(1), b.reshape(b.shape[0], -1).sum(1)
        print(f"  {name} per listener cpu {per[0]} gpu {per[1]}")
        nonzero = np.abs(a) > 1e-6 * np.abs(a).max(initial=0.0)
        if nonzero.any():
            rel = np.abs(b[nonzero] / a[nonzero] - 1.0)
            print(f"  {name} bins {nonzero.sum()}, relative difference max {rel.max():.3g}")
            print(f"  {name} bins where gpu is 0: {(b[nonzero] == 0).sum()}")
    for i, name in ((3, "escaped"), (6, "traced")):
        print(f"  {name} cpu {on_cpu[i]} gpu {on_gpu[i]}")
    for i, name in ((4, "ray_energy"), (5, "reflections")):
        print(f"  {name} sum cpu {np.sum(on_cpu[i]):.6g} gpu {np.sum(on_gpu[i]):.6g}")
    return on_cpu, on_gpu


def main() -> None:
    adapter = gpu._adapter()
    print("adapter:", None if adapter is None else adapter.info)
    if adapter is None:
        return
    device = gpu._device()
    print("storage binding limit:", device.limits["max-storage-buffer-binding-size"])
    with np.load(REFERENCE) as stored:
        data = dict(stored)
    vertices, faces = _unit_cube()

    values = reference_arguments(_native, data, "random_detector")
    print(
        "random_detector flags:",
        {
            k: values[k]
            for k in (
                "vector_scattering",
                "diffuse_rain",
                "specular_rain",
                "covered_order",
                "rays",
                "max_time",
            )
        },
    )
    print("split target size:", np.asarray(values["split_target"]).size)
    values |= {
        "directions": data["directions"],
        "bvh": _native.Bvh(vertices * 120.0 - 55.0, faces),
        "absorption": np.full((len(faces), 3), 0.2),
        "scattering": np.full((len(faces), 3), 0.3),
        "receiver_radius": np.full(3, 1.0),
        "rays": 4000,
        "max_time": 0.12,
    }
    _run("direct sound, 4000 rays", values)
    _run("direct sound, no directions", values | {"directions": np.zeros((0, 3))})

    values = reference_arguments(_native, data, "random_detector")
    top = np.all(vertices[faces][:, :, 2] == 1.0, axis=1)
    walls = len(faces) - int(top.sum())
    values |= {
        "bvh": _native.Bvh(vertices * 20.0 - (4.0, 5.0, 0.0), faces[~top]),
        "absorption": np.full((walls, 3), 0.1),
        "scattering": np.full((walls, 3), 0.5),
        "rays": 4000,
    }
    _run("open box, 4000 rays", values)
    _run("open box, no splitting", values | {"split_target": np.zeros(0)})

    for name in ("vector", "random_detector"):
        values = reference_arguments(_native, data, name)
        _run(f"reference {name}, {values['rays']} rays", values)
        _run(f"reference {name}, no splitting", values | {"split_target": np.zeros(0)})


if __name__ == "__main__":
    main()
