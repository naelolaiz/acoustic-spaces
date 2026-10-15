"""Time the wave solver on a GPU against the CPU, and check that the two agree.

    python validation/wave_gpu.py [--rooms studio concert_hall church] [--band 125]
                                  [--length 0.5]

Needs the ``gpu`` extra (wgpu) and a GPU driver: Vulkan, Metal or Direct3D 12. Mesa's
lavapipe, a Vulkan driver that runs on the CPU, works too, only slowly. For each example
room, the wave bands of the first source to every listener are computed on the CPU (all
its cores) and on the GPU, and printed as a Markdown table: the grid, the time each
took (the grid's set-up included, which is the same on both), and the largest
difference between the two runs, relative to the largest pressure.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room
from acoustic_spaces.engine.gpu import gpu_name
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.wave import wave_responses

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _time(model, room: Room, band: int, length: float, device: str):
    source = np.array(room.scene.sources[0].position)
    listeners = room.scene.listeners
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
        device=device,
    )
    seconds = time.perf_counter() - started
    return seconds, np.stack([r.signals for r in responses]), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rooms", nargs="+", default=["studio", "concert_hall", "church"])
    parser.add_argument("--band", type=int, default=125, help="highest wave band (Hz)")
    parser.add_argument("--length", type=float, default=0.5, help="response length (s)")
    args = parser.parse_args()

    name = gpu_name()
    if name is None:
        raise SystemExit('wgpu finds no GPU: pip install "acoustic-spaces[gpu]" and a driver')
    print(f"{platform.machine()}, {platform.system()}, {os.cpu_count()} CPU threads")
    print(f"GPU: {name}")
    print()
    print("| Room | Cells | Steps | CPU (s) | GPU (s) | Speed-up | Largest difference |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for room_name in args.rooms:
        room = Room.from_yaml(EXAMPLES / room_name / "scene.yaml")
        model = TraceModel.from_room(room)
        cpu_s, cpu, info = _time(model, room, args.band, args.length, "cpu")
        gpu_s, gpu, _ = _time(model, room, args.band, args.length, "gpu")
        steps = int(info["wave_steps"])
        difference = np.abs(cpu - gpu).max() / np.abs(cpu).max()
        print(
            f"| {room_name} | {info['wave_cells']:,.0f} | {steps:,} | {cpu_s:.1f} | "
            f"{gpu_s:.1f} | {cpu_s / gpu_s:.2f} | {difference:.1e} |"
        )


if __name__ == "__main__":
    main()
