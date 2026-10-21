"""Trace the late rays of outdoor scenes on the CPU and on a GPU, and compare what
reaches the listeners.

    python validation/outdoor_gpu.py [--scenes examples/open_air_stage/scene.yaml ...]
                                     [--rays 20000] [--devices cpu gpu both]

Needs the ``gpu`` extra (wgpu) and a GPU driver: Vulkan, Metal or Direct3D 12. The
scenes bend their rays in wind and temperature (the open-air stage), send them through
vegetation (the tree belt) and over a terrain (the hillside), which the GPU traces as
the CPU does. Each scene is simulated with its own settings once per device
(``simulation.late.device``), and once more on the CPU with the next seed, which shows
how much two runs differ by the ray count's noise alone. The settings come first, then
a Markdown table with aligned columns: where the late rays went, the late stage's time
(summed over the sources), how many times faster than on the CPU it was, and how far
the late sound's level per band and its centre time are from the CPU's, averaged over
every listener and band. The next seed also redraws the eddies of a scene with
turbulence (the open-air stage), which move the levels in the wind's shadow by several
dB on their own; the GPU run shares the CPU's eddies, so its differences are the rays'
alone.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, simulate
from acoustic_spaces.engine import gpu, gpu_tracer, ray_device
from acoustic_spaces.engine.model import TraceModel

ROOT = Path(__file__).resolve().parent.parent
SCENES = [
    ROOT / "examples" / "open_air_stage" / "scene.yaml",
    ROOT / "examples" / "tree_belt" / "scene.yaml",
    ROOT / "examples" / "hillside" / "scene.yaml",
]
COLUMNS = (
    "Scene",
    "Faces",
    "Rays",
    "Device",
    "Traced on",
    "Late (s)",
    "Speed-up",
    "Level diff (dB)",
    "Centre time diff (ms)",
)


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
    """The late stage's time (s), where its rays went, the late sound's level per band
    (dB) and centre time (s) per response, and the ray count of one simulation."""
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
    levels, centres = [], []
    for response in result.responses.values():
        energy = response.late
        t = (np.arange(energy.shape[1]) + 0.5) * response.time_step
        total = energy.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            levels.append(10.0 * np.log10(total))
            centres.append((energy * t).sum(axis=1) / total)
    return late_s, where, (np.array(levels), np.array(centres)), settings.late.rays


def _differences(found: tuple, cpu: tuple) -> tuple[str, str]:
    """Mean absolute differences from the CPU's: the level in dB and the centre time in
    ms, over the listeners and bands that both runs put late sound in."""
    both = np.isfinite(found[0]) & np.isfinite(cpu[0])
    level = np.abs(found[0][both] - cpu[0][both])
    centre = np.abs(found[1][both] - cpu[1][both]) * 1e3
    return f"{level.mean():.2f}", f"{centre.mean():.1f}"


def _warm_up(room: Room) -> None:
    """Starts the GPU and compiles the tracer's shader for the scene's band count."""
    model = TraceModel.from_room(room, room.scene.simulation)
    gpu._device()
    gpu_tracer._pipelines(len(model.bands))


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
            differences = ("-",) * 2 if label == "cpu" else _differences(found, cpu)
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
