#!/usr/bin/env python3
"""A/B the archway's jamb step at different early coherence settings.

The jamb step at x = 7.6 m is the known issue: at 250 and 500 Hz the energy model
(``simulation.early.coherent_max_band_hz: 125``, the default) steps by about 1 dB where
the reflection off the 40 cm jamb appears; with those bands in pressure it is smooth.
This script measures the three settings on the same points:

    energy        coherent_max_band_hz: 125   (default)
    coherent250   coherent_max_band_hz: 250
    coherent500   coherent_max_band_hz: 500

For each it prints the 1 cm and 1 mm steps in every band, the arrival's early energy,
and (unless ``--no-simulate``) the ISO 3382-1 parameters at the listener, with the time
of a short ``simulate`` run.  Use ``--threads``/``--late-device`` to include the
processor/device dimension; the edge fade itself runs on the CPU.

    python validation/archway_ab.py
    python validation/archway_ab.py --json archway_ab.json --threads 1
    python validation/archway_ab.py --late-device gpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import early_paths, simulate
from acoustic_spaces.engine.coherent import total_energy

SCENE = Path(__file__).resolve().parent.parent / "examples" / "archway" / "scene.yaml"
LISTENER = "behind the wall"
SOURCE = "singer"
#: (y before, y after) pairs 1 cm and 1 mm apart across the two steps.
PAIRS = (
    ("1 cm, y=6.14", 6.135, 6.145),
    ("1 mm, y=6.14", 6.1395, 6.1405),
    ("1 cm, y=7.33", 7.325, 7.335),
    ("1 mm, y=7.33", 7.3295, 7.3305),
)
MODES = {"energy": 125, "coherent250": 250, "coherent500": 500}


def _with_mode(scene: Scene, coherent: int, threads: int = 0) -> Scene:
    out = scene.model_copy(deep=True)
    out.simulation.early.coherent_max_band_hz = coherent
    out.simulation.performance.threads = threads
    return out


def _listener_at(scene: Scene, y: float):
    listener = next(item for item in scene.listeners if item.name == LISTENER)
    moved = listener.model_copy(update={"position": (7.6, y, 1.2)})
    return scene.model_copy(update={"listeners": [moved]}), moved


def steps(scene: Scene, coherent: int, threads: int = 0) -> list[dict]:
    """The level change across each pair, per band (dB) and dB(A)."""
    base = _with_mode(scene, coherent, threads)
    out = []
    for name, before, after in PAIRS:
        levels = []
        for y in (before, after):
            moved_scene, moved = _listener_at(base, y)
            paths = early_paths(moved_scene, SOURCE, moved.name)
            levels.append(total_energy(paths))
        delta = 10.0 * np.log10(levels[1] / levels[0])
        out.append(
            {
                "pair": name,
                "before": before,
                "after": after,
                "step_db": [float(value) for value in delta],
                "max_abs_db": float(np.abs(delta).max()),
            }
        )
        print(f"  {name}: step {np.round(delta, 3).tolist()} dB, max {np.abs(delta).max():.3f}")
    return out


def parameters(scene: Scene, coherent: int, args) -> dict:
    settings = scene.simulation.model_copy(deep=True)
    settings.early.coherent_max_band_hz = coherent
    settings.late.rays = args.rays
    settings.ir_length_s = args.length
    settings.seed = args.seed
    settings.performance.threads = args.threads
    if args.late_device is not None:
        settings.late.device = args.late_device
    room = Room.from_scene(_with_mode(scene, coherent))
    started = time.perf_counter()
    result = simulate(room, settings=settings)
    elapsed = time.perf_counter() - started
    response = result[SOURCE, LISTENER]
    params = response.parameters()
    levels = {
        "edt": [float(value) for value in params.edt],
        "t30": [float(value) for value in params.t30],
        "c80": [float(value) for value in params.c80],
        "d50": [float(value) for value in params.d50],
        "g": [float(value) for value in params.g],
    }
    print(
        f"  simulate {elapsed:.2f} s: C80 250/500 "
        f"{params.c80[2]:.2f}/{params.c80[3]:.2f} dB, G "
        f"{params.g[2]:.2f}/{params.g[3]:.2f} dB, T30 "
        f"{params.t30[2]:.3f}/{params.t30[3]:.3f} s"
    )
    return {"seconds": elapsed, "levels": levels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the results here")
    parser.add_argument("--rays", type=int, default=12_000, help="late rays per band")
    parser.add_argument("--length", type=float, default=1.5, help="response length (s)")
    parser.add_argument("--seed", type=int, default=7, help="simulation seed")
    parser.add_argument("--threads", type=int, default=0, help="performance.threads")
    parser.add_argument(
        "--late-device",
        choices=("cpu", "gpu", "both", "auto"),
        help="simulation.late.device for the parameter A/B",
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help="only the early-energy steps; skip the parameter simulations",
    )
    args = parser.parse_args()

    scene = Scene.from_yaml(SCENE)
    report = {"scene": str(SCENE), "listener": LISTENER, "modes": {}}
    for name, coherent in MODES.items():
        print(f"\n== {name} (coherent_max_band_hz: {coherent})")
        found = {
            "coherent_max_band_hz": coherent,
            "threads": args.threads,
            "steps": steps(scene, coherent, args.threads),
        }
        if not args.no_simulate:
            found["parameters"] = parameters(scene, coherent, args)
        report["modes"][name] = found
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
