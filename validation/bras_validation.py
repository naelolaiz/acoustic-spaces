"""Validation against measured rooms: the seminar room and the chamber music hall of BRAS.

    python validation/bras_validation.py [--scenes CR2 CR3] [--materials fitted]
                                         [--rays 100000] [--diffraction first_order]
                                         [--json results.json] [--plots DIR]
    python validation/bras_validation.py --from-json results.json [--plots DIR]

The BRAS benchmark (Aspöck et al., 2020, DOI 10.14279/depositonce-6726.3) measured
rooms whose geometry and surfaces it describes; ``validation/bras/`` holds two of them
as scenes, with the parameters of their measured responses (see the README there). Each
scene is simulated as it stands, and every source and listener pair compared with the
response measured there, band by band in just noticeable differences
(:func:`acoustic_spaces.compare.compare_parameters`). Both sides are analysed by this
package. ``--materials initial`` takes BRAS's first estimates of the surfaces instead of
the absorption it fitted to the measured reverberation time, which is the blind case.

The script prints a Markdown table per room: the measured and simulated parameters
averaged over the ten pairs, and the mean difference in JNDs; then how many of the
pair-and-band differences from 125 Hz to 4 kHz are within one JND. ``--plots`` draws the
averages with the JND around the measured ones.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.compare import JND, SUMMARY_BANDS_HZ, compare_parameters
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import RoomParameters
from acoustic_spaces.scene.model import Material

HERE = Path(__file__).resolve().parent / "bras"
#: The parameters BRAS measured, and those the responses give besides (G has no
#: free-field reference in the measurements).
PARAMETERS = ("edt", "t20", "t30", "c50", "c80", "d50", "ts")
LABELS = {
    "edt": "EDT (s)",
    "t20": "T20 (s)",
    "t30": "T30 (s)",
    "c50": "C50 (dB)",
    "c80": "C80 (dB)",
    "d50": "D50",
    "ts": "Ts (s)",
}
SUMMARY = np.isin(OCTAVE_BANDS_HZ, SUMMARY_BANDS_HZ)


def _band(f: int) -> str:
    return f"{f} Hz" if f < 1000 else f"{f // 1000} kHz"


def load_scene(name: str, materials: str, rays: int | None, diffraction: str | None) -> Scene:
    scene = Scene.from_yaml(HERE / name / "scene.yaml")
    if materials != "fitted":
        scene.materials = {
            key: Material(file=HERE / name / "materials" / materials / f"{key}.csv")
            for key in scene.materials
        }
    if rays is not None:
        scene.simulation.late.rays = rays
    if diffraction is not None:
        scene.simulation.early.diffraction = diffraction
    return scene


def _values(values: list) -> np.ndarray:
    """A list from JSON, with NaN where it has null."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _plain(values: np.ndarray) -> list[float | None]:
    """For JSON: None where a value is missing."""
    return [float(v) if np.isfinite(v) else None for v in values]


def measured_parameters(entry: dict) -> RoomParameters:
    return RoomParameters(**{k: _values(v) for k, v in entry.items()})


def validate(name: str, materials: str, rays: int | None, diffraction: str | None) -> dict:
    measured = json.loads((HERE / name / "measured.json").read_text(encoding="utf-8"))
    scene = load_scene(name, materials, rays, diffraction)
    room = Room.from_scene(scene)
    start = time.perf_counter()
    result = simulate(room)
    seconds = time.perf_counter() - start
    pairs = {}
    for key, entry in measured["pairs"].items():
        source, listener = key.split()
        simulated = result[source, listener].parameters()
        pairs[key] = compare_parameters(simulated, measured_parameters(entry)).as_dict()
    out = {
        "scene": name,
        "name": scene.name,
        "materials": materials,
        "rays": scene.simulation.late.rays,
        "diffraction": scene.simulation.early.diffraction,
        "seconds": round(seconds, 1),
        "volume_m3": room.volume_m3,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "pairs": pairs,
        "average": {},
    }
    for p in PARAMETERS:
        stack = {
            side: np.array([_values(pair["parameters"][p][side]) for pair in pairs.values()])
            for side in ("simulated", "measured", "difference_jnd")
        }
        middle = stack["difference_jnd"][:, SUMMARY]
        compared = np.isfinite(middle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a band no pair has: NaN
            mean = {side: np.nanmean(values, axis=0) for side, values in stack.items()}
            out["average"][p] = {
                **{side: _plain(values) for side, values in mean.items()},
                "within_jnd": int(np.count_nonzero(np.abs(middle[compared]) <= 1.0)),
                "compared": int(np.count_nonzero(compared)),
                "mean_abs_jnd": float(np.abs(middle[compared]).mean()) if compared.any() else None,
            }
    return out


def table(result: dict) -> str:
    bands = result["bands_hz"]
    lines = [
        f"{result['name']}: {result['materials']} absorption, {result['rays']} rays, "
        f"diffraction {result['diffraction']}, simulated in {result['seconds']:.0f} s",
        "",
        "| | " + " | ".join(_band(f) for f in bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    for p in PARAMETERS:
        avg = result["average"][p]
        fmt = ".1f" if "dB" in LABELS[p] else ".2f"
        for side, label in (("measured", "measured"), ("simulated", "simulated")):
            cells = ["-" if v is None else f"{v:{fmt}}" for v in avg[side]]
            lines.append(f"| {LABELS[p]} {label} | " + " | ".join(cells) + " |")
        cells = ["-" if v is None else f"{v:+.1f}" for v in avg["difference_jnd"]]
        lines.append(f"| {LABELS[p]} JNDs | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| | within 1 JND, 125 Hz to 4 kHz | mean abs. difference (JNDs) |",
        "|---|---:|---:|",
    ]
    for p in PARAMETERS:
        avg = result["average"][p]
        mean = avg["mean_abs_jnd"]
        share = f"{avg['within_jnd']} of {avg['compared']}"
        lines.append(f"| {LABELS[p]} | {share} | {'-' if mean is None else f'{mean:.1f}'} |")
    return "\n".join(lines)


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = result["bands_hz"]
    x = np.arange(len(bands))
    shown = ("edt", "t20", "c80", "d50")
    fig, axes = plt.subplots(1, len(shown), figsize=(12, 3.2), layout="constrained")
    for ax, p in zip(axes, shown, strict=True):
        avg = result["average"][p]
        measured = _values(avg["measured"])
        step, relative = JND[p]
        jnd = step * np.abs(measured) if relative else np.full(len(bands), step)
        ax.fill_between(x, measured - jnd, measured + jnd, color="C1", alpha=0.2, label="1 JND")
        ax.plot(x, measured, "s-", color="C1", label="measured")
        ax.plot(x, _values(avg["simulated"]), "o-", color="C0", label="simulated")
        ax.set_xticks(x, [f"{f // 1000}k" if f >= 1000 else str(f) for f in bands], fontsize=7)
        ax.set_xlabel("octave band (Hz)", fontsize=7)
        ax.set_title(LABELS[p])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    title = f"{result['name']}, {result['materials']} absorption: averages of ten responses"
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", default=["CR2", "CR3"])
    parser.add_argument("--materials", choices=["fitted", "initial"], default="fitted")
    parser.add_argument("--rays", type=int, help="late rays (default: the scene's)")
    parser.add_argument("--diffraction", choices=["none", "direct", "first_order"])
    parser.add_argument("--json", type=Path, help="write every number to this file")
    parser.add_argument("--plots", type=Path, help="draw a PNG per room into this folder")
    parser.add_argument(
        "--from-json", type=Path, help="print and plot the results of an earlier --json run"
    )
    args = parser.parse_args()
    if args.from_json:
        results = json.loads(args.from_json.read_text(encoding="utf-8"))
    else:
        results = [
            validate(name, args.materials, args.rays, args.diffraction) for name in args.scenes
        ]
    for result in results:
        print(table(result) + "\n", flush=True)
        if args.plots:
            args.plots.mkdir(parents=True, exist_ok=True)
            plot(result, args.plots / f"bras_{result['scene']}_{result['materials']}.png")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
