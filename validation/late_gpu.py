"""Time the late stage's ray tracing on the CPU, on a GPU and on both, and compare the
room parameters each gives.

    python validation/late_gpu.py [--scenes validation/bras/CR3/scene.yaml ...]
                                  [--rays 20000] [--devices cpu gpu both]

Needs the ``gpu`` extra (wgpu) and a GPU driver: Vulkan, Metal or Direct3D 12. Each
scene is simulated with its own settings once per device (``simulation.late.device``),
and once more on the CPU with the next seed, which shows how much two runs differ by
the ray count's noise alone. The settings come first, then a Markdown table with aligned
columns: where the late rays went, the late stage's time (summed over the sources), how
many times faster than on the CPU it was, and how far T30, EDT and C80 are from the
CPU's, averaged over every response and octave band. The GPU is started and its shader
compiled before the first scene, so those one-off costs are left out.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, simulate
from acoustic_spaces.engine import gpu, ray_device
from acoustic_spaces.engine.model import TraceModel

ROOT = Path(__file__).resolve().parent.parent
SCENES = [
    ROOT / "validation" / "bras" / "CR2" / "scene.yaml",
    ROOT / "validation" / "bras" / "CR3" / "scene.yaml",
    ROOT / "examples" / "concert_hall" / "scene.yaml",
    ROOT / "examples" / "church" / "scene.yaml",
]
COLUMNS = (
    "Scene",
    "Faces",
    "Rays",
    "Device",
    "Traced on",
    "Late (s)",
    "Speed-up",
    "T30 diff",
    "EDT diff",
    "C80 diff (dB)",
)
PARAMETERS = ("t30", "edt", "c80")


def _settings(pairs: list[tuple[str, str]]) -> None:
    """Prints one setting per line, the values lined up."""
    width = max(len(label) for label, _ in pairs) + 1
    for label, value in pairs:
        print(f"{label + ':':<{width}} {value}")


def _table(rows: list[tuple[str, ...]]) -> str:
    """A Markdown table whose columns line up in a terminal too: text on the left,
    numbers on the right."""
    text = {0, 3, 4}
    widths = [max(len(row[i]) for row in [COLUMNS, *rows]) for i in range(len(COLUMNS))]

    def line(cells) -> str:
        padded = [
            cell.ljust(width) if i in text else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = ["-" * w if i in text else "-" * (w - 1) + ":" for i, w in enumerate(widths)]
    return "\n".join([line(COLUMNS), line(rule), *(line(row) for row in rows)])


def _run(room: Room, device: str, rays: int | None, seed_step: int = 0):
    """The late stage's time (s), where its rays went and the parameters (name: array of
    responses by bands) of one simulation."""
    settings = room.scene.simulation.model_copy(deep=True)
    settings.late.device = device
    if rays:
        settings.late.rays = rays
    settings.seed = (1 if settings.seed is None else settings.seed) + seed_step
    messages: list[str] = []
    result = simulate(room, settings=settings, progress=messages.append)
    late_s = sum(info.get("late_s", 0.0) for info in result.details.values())
    said = [m.split("late rays traced on ", 1)[1] for m in messages if "late rays traced on" in m]
    where = said[0] if said else "the CPU"
    found = list(result.parameters().values())
    parameters = {name: np.array([getattr(p, name) for p in found], float) for name in PARAMETERS}
    return late_s, where, parameters, settings.late.rays


def _differences(found: dict, cpu: dict) -> tuple[str, str, str]:
    """Mean absolute differences from the CPU's: T30 and EDT in %, C80 in dB."""
    with np.errstate(invalid="ignore", divide="ignore"):
        t30 = np.nanmean(np.abs(found["t30"] / cpu["t30"] - 1.0))
        edt = np.nanmean(np.abs(found["edt"] / cpu["edt"] - 1.0))
        c80 = np.nanmean(np.abs(found["c80"] - cpu["c80"]))
    return f"{t30:.1%}", f"{edt:.1%}", f"{c80:.2f}"


def _warm_up(room: Room) -> None:
    """Starts the GPU and compiles the tracer's shader as the scene needs it (for its
    bands, and what it has of the outdoors), in a run of a few rays."""
    _run(room, "gpu", 256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", type=Path, default=SCENES)
    parser.add_argument("--rays", type=int, default=None, help="late rays (default: the scene's)")
    parser.add_argument(
        "--devices", nargs="+", choices=ray_device.DEVICES, default=["cpu", "gpu", "both"]
    )
    args = parser.parse_args()
    devices = [d for d in args.devices if d != "cpu"]

    name = gpu.gpu_name()
    if name is None:
        raise SystemExit('wgpu finds no GPU: pip install "acoustic-spaces[gpu]" and a driver')
    rays = f"{args.rays:,} per band" if args.rays else "each scene's own"
    _settings(
        [
            ("Machine", f"{platform.machine()}, {platform.system()}"),
            ("CPU", f"{os.cpu_count()} threads, all used"),
            ("GPU", name),
            ("Late rays", rays),
            ("Devices", ", ".join(["cpu", "cpu (next seed)", *devices])),
        ]
    )
    print()
    rows = []
    for path in args.scenes:
        room = Room.from_yaml(path)
        if room.scene.simulation.late.method != "stochastic_ray_tracing":
            print(f"{path.parent.name}: skipped, its late stage traces no rays", flush=True)
            continue
        _warm_up(room)
        faces = f"{TraceModel.from_room(room, room.scene.simulation).bvh.face_count:,}"
        cpu_s, _, cpu, count = _run(room, "cpu", args.rays)
        runs = [("cpu", (cpu_s, "the CPU", cpu, count))]
        runs.append(("cpu (next seed)", _run(room, "cpu", args.rays, seed_step=1)))
        runs += [(device, _run(room, device, args.rays)) for device in devices]
        for label, (late_s, where, found, count) in runs:
            differences = ("-",) * 3 if label == "cpu" else _differences(found, cpu)
            rows.append(
                (
                    path.parent.name,
                    faces,
                    f"{count:,}",
                    label,
                    where,
                    f"{late_s:.2f}",
                    f"{cpu_s / max(late_s, 1e-9):.2f}x",
                    *differences,
                )
            )
        print(f"{path.parent.name} done", flush=True)
    print()
    print(_table(rows))


if __name__ == "__main__":
    main()
