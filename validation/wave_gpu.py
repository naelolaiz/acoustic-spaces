"""Time the wave solver on a GPU against the CPU, and check that the two agree.

    python validation/wave_gpu.py [--rooms studio concert_hall church] [--band 125]
                                  [--length 0.5]

Needs the ``gpu`` extra (wgpu) and a GPU driver: Vulkan, Metal or Direct3D 12. Mesa's
lavapipe, a Vulkan driver that runs on the CPU, works too, only slowly. For each example
room, the wave bands of the first source to every listener are computed on the CPU (all
its cores) and on the GPU. The settings come first, then a Markdown table with aligned
columns: the grid (its cells, their size and the stencil), the time its set-up took (on
the CPU either way), the time the time steps took on each (on the GPU, its buffers and
shader included), how many times faster the GPU stepped, and the largest difference
between the two runs, relative to the largest pressure. The GPU is started before the
first room, so that one-off cost is left out.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room
from acoustic_spaces.engine import gpu
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.wave import wave_responses

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
#: Grid cells per wavelength of the highest band, and the stencil (see wave_responses).
POINTS = 8.0
SCHEME = "auto"
COLUMNS = (
    "Room",
    "Cells",
    "Cell (cm)",
    "Stencil",
    "Steps",
    "Set-up (s)",
    "CPU steps (s)",
    "GPU steps (s)",
    "Speed-up",
    "Largest difference",
)


def _time(model, room: Room, band: int, length: float, device: str):
    """Seconds of set-up and of time steps, the responses and the run's details."""
    source = np.array(room.scene.sources[0].position)
    listeners = room.scene.listeners
    # The solver says what it runs just before its first time step.
    marks: list[float] = []
    started = time.perf_counter()
    responses, info = wave_responses(
        model,
        source,
        np.array([listener.position for listener in listeners]),
        [listener.name for listener in listeners],
        max_band_hz=band,
        length_s=length,
        sample_rate=room.scene.simulation.sample_rate,
        speed_of_sound=room.speed_of_sound(),
        points_per_wavelength=POINTS,
        scheme=SCHEME,
        device=device,
        say=lambda _: marks.append(time.perf_counter()),
    )
    done = time.perf_counter()
    stepping = marks[0] if marks else started
    return stepping - started, done - stepping, np.stack([r.signals for r in responses]), info


def _table(rows: list[tuple[str, ...]]) -> str:
    """A Markdown table whose columns line up in a terminal too: text on the left,
    numbers on the right."""
    widths = [max(len(row[i]) for row in [COLUMNS, *rows]) for i in range(len(COLUMNS))]

    def line(cells) -> str:
        padded = [
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = ["-" * widths[0], *("-" * (width - 1) + ":" for width in widths[1:])]
    return "\n".join([line(COLUMNS), line(rule), *(line(row) for row in rows)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rooms", nargs="+", default=["studio", "concert_hall", "church"])
    parser.add_argument("--band", type=int, default=125, help="highest wave band (Hz)")
    parser.add_argument("--length", type=float, default=0.5, help="response length (s)")
    args = parser.parse_args()

    name = gpu.gpu_name()
    if name is None:
        raise SystemExit('wgpu finds no GPU: pip install "acoustic-spaces[gpu]" and a driver')
    print(f"{platform.machine()}, {platform.system()}, {os.cpu_count()} CPU threads")
    print(f"GPU: {name}")
    print(
        f"Wave bands up to {args.band} Hz, {args.length:g} s responses, {POINTS:g} points "
        f"per wavelength, scheme {SCHEME} (19-point stencil, 7-point round an open model), "
        "first source to every listener; the CPU on all its threads"
    )
    print()
    gpu._device()  # start the GPU once, outside the timings
    rows = []
    for room_name in args.rooms:
        room = Room.from_yaml(EXAMPLES / room_name / "scene.yaml")
        model = TraceModel.from_room(room)
        setup_s, cpu_s, cpu, info = _time(model, room, args.band, args.length, "cpu")
        _, gpu_s, on_gpu, _ = _time(model, room, args.band, args.length, "gpu")
        difference = np.abs(cpu - on_gpu).max() / np.abs(cpu).max()
        rows.append(
            (
                room_name,
                f"{info['wave_cells']:,.0f}",
                f"{100 * info['wave_cell_m']:.1f}",
                f"{int(info['wave_stencil'])}-point",
                f"{int(info['wave_steps']):,}",
                f"{setup_s:.2f}",
                f"{cpu_s:.2f}",
                f"{gpu_s:.2f}",
                f"{cpu_s / gpu_s:.2f}x",
                f"{difference:.1e}",
            )
        )
        print(f"{room_name} done", flush=True)
    print()
    print(_table(rows))


if __name__ == "__main__":
    main()
