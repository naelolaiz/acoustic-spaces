"""What each part of the scattering model does in BRAS's rooms: the T20 of the total
sound energy in a room without listeners, with each of three scattering models.

    python validation/bras_room_energy.py [--scenes CR2 CR3] [--materials fitted]
                                          [--rays 20000] [--seed 0]
                                          [--results bras.json ...]

For each room, rays leave its first loudspeaker and are traced as
:func:`pyroomacoustics_crosscheck.global_decay` traces them, with the scene's absorption
and air: with random scattering as this package had it up to version 2 (one ray for all
bands, turned with the bands' mean coefficient as probability and each band re-weighted
by its own), with vector-based scattering, and with vector-based scattering and the
scattering of each surface's size (this package's default since version 3), the last
two each band with rays of its own where the bands' coefficients differ. T20 is fitted
to each band's decay from 5 to 25 dB down. The script prints them per octave band with
the measured T20 (averaged over the room's source and receiver pairs) and Eyring's
formula with the same absorption and air; with ``--results`` (the JSON of
``bras_validation.py`` for the same rooms and materials) also the simulated responses'
T20 and how far it is from the room energy's with the default model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bras_validation import _plain, load_scene  # noqa: E402
from pyroomacoustics_crosscheck import decay_time, global_decay  # noqa: E402

from acoustic_spaces import Room  # noqa: E402
from acoustic_spaces.materials import OCTAVE_BANDS_HZ  # noqa: E402

MODELS = {
    "random": "random scattering (version 2)",
    "vector": "vector-based scattering",
    "size": "vector-based with surface size (default)",
}


def global_t20(t: np.ndarray, level: np.ndarray) -> np.ndarray:
    """T20 of a global decay: the fit from where it is 5 dB down to where it is 25 dB down."""
    t20 = np.full(level.shape[1], np.nan)
    for b in range(level.shape[1]):
        below = np.flatnonzero(level[:, b] <= -5.0), np.flatnonzero(level[:, b] <= -25.0)
        if len(below[1]):
            t20[b] = decay_time(t, level[:, b], t[below[0][0]], t[below[1][0]])
    return t20


def averaged_t20(results: dict, side: str) -> np.ndarray:
    """The T20 of a room's source and receiver pairs in ``bras_validation.py``'s JSON,
    averaged (``side``: measured or simulated)."""
    values = [
        [np.nan if v is None else v for v in pair["parameters"]["t20"][side]]
        for pair in results["pairs"].values()
    ]
    return np.nanmean(np.array(values, dtype=float), axis=0)


def row(label: str, values: np.ndarray) -> str:
    return f"| {label} | " + " | ".join(f"{v:.2f}" for v in values) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", default=["CR2", "CR3"])
    parser.add_argument("--materials", choices=["fitted", "initial"], default="fitted")
    parser.add_argument("--rays", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results", type=Path, nargs="+", default=[], help="bras_validation.py's JSON files"
    )
    parser.add_argument("--json", type=Path, help="write the numbers to this JSON file")
    args = parser.parse_args()

    results = {}
    for path in args.results:
        for entry in json.loads(path.read_text()):
            if entry["materials"] == args.materials:
                results[entry["scene"]] = entry
    out = {}
    labels = [f"{f} Hz" if f < 1000 else f"{f // 1000} kHz" for f in OCTAVE_BANDS_HZ]
    print("| T20 (s) | " + " | ".join(labels) + " |")
    print("|---|" + "---:|" * len(labels))
    for name in args.scenes:
        room = Room.from_scene(load_scene(name, args.materials, None, None))
        found = {}
        for model in MODELS:
            t, level = global_decay(room, model, rays=args.rays, seed=args.seed)
            found[model] = global_t20(t, level)
        eyring = room.reverberation_time("eyring", air_absorption=True)
        entry = results.get(name)
        if entry is not None:
            print(row(f"{name} measured", averaged_t20(entry, "measured")))
        for model, label in MODELS.items():
            print(row(f"{name} room energy, {label}", found[model]))
        print(row(f"{name} Eyring", eyring))
        out[name] = {model: _plain(found[model]) for model in MODELS}
        out[name]["eyring"] = _plain(eyring)
        if entry is not None:
            simulated = averaged_t20(entry, "simulated")
            off = 100.0 * (simulated / found["size"] - 1.0)
            print(row(f"{name} simulated responses", simulated))
            print(
                f"| {name} simulated against room energy (default, %) | "
                + " | ".join(f"{v:+.1f}" for v in off)
                + " |"
            )
            out[name]["simulated"] = _plain(simulated)
    if args.json is not None:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
