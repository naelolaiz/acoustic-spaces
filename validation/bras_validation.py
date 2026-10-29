"""Validation against measured rooms: the coupled rooms, seminar room, chamber music hall
and auditorium of BRAS.

    python validation/bras_validation.py [--scenes CR1_DoorAngle1 CR2 CR3 CR4]
                                         [--materials fitted]
                                         [--rays 50000] [--diffraction first_order]
                                         [--bands octave] [--json results.json]
                                         [--plots DIR] [--calibrate [SOURCE ...]]
                                         [--calibration-rays 10000]
    python validation/bras_validation.py --from-json results.json [--plots DIR]

The BRAS benchmark (Aspöck et al., 2020, DOI 10.14279/depositonce-6726.3) measured
rooms whose geometry and surfaces it describes; ``validation/bras/`` holds four of them
as scenes, the coupled rooms once per angle of the door between them, with the
parameters of their measured responses (see the README there). Each scene is simulated
as it stands, and every source and listener pair compared with the response measured
there, band by band in just noticeable differences
(:func:`acoustic_spaces.compare.compare_parameters`). Both sides are analysed by this
package. ``--materials initial`` takes BRAS's first estimates of the surfaces instead of
the absorption it fitted to the measured reverberation time, which is the blind case.
``--bands third_octave`` simulates in third-octave bands, with BRAS's third-octave
surface data as published, instead of its octave averages; the parameters are compared
in octave bands either way.

``--calibrate`` first fits the absorption of every surface to the measured T30, EDT and
C80 (:func:`acoustic_spaces.calibration.calibrate`, with ``--calibration-rays`` rays),
starting from the ``--materials`` set, then simulates the room with what it found. With
sources named, only their responses are fitted, and the averages are over the other
source's: responses the fit did not see.

The script prints a Markdown table per room: the measured and simulated parameters
averaged over its pairs (ten; four in the coupled rooms), and the mean difference in
JNDs; then how many of the pair-and-band differences from 125 Hz to 4 kHz are within one
JND. G has no free-field reference in the measurements, so it is compared from seat to
seat: each pair's G less the mean over the room's pairs, against each pair's measured
level less its mean, by the rms difference over the pairs and how many are within one
JND (1 dB). Where the late rays were traced in batches, the summary's last column is
the largest standard error of a simulated parameter over the pairs from 125 Hz to 4 kHz,
in JNDs: how much the simulation itself would change with another seed. ``--plots``
draws the averages with the JND around the measured ones.
Progress goes to standard error: each stage of each room's simulation as it starts, with
the seconds since the room started.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.calibration import calibrate
from acoustic_spaces.compare import JND, SUMMARY_BANDS_HZ, compare_parameters, in_jnds
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import RoomParameters
from acoustic_spaces.scene.model import Material

HERE = Path(__file__).resolve().parent / "bras"
#: The parameters BRAS measured, and those the responses give besides (G, which has no
#: free-field reference in the measurements, is compared from seat to seat).
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
#: Scenes whose loudspeakers stand in different rooms (CR1: LS1 in the reverberation
#: chamber, LS2 in the laboratory with the microphones). An average over all their pairs
#: would mix two decays, so their tables and plots give each source's averages.
COUPLED = ("CR1_DoorAngle1", "CR1_DoorAngle3")


def _band(f: int) -> str:
    return f"{f} Hz" if f < 1000 else f"{f // 1000} kHz"


def load_scene(
    name: str,
    materials: str,
    rays: int | None,
    diffraction: str | None,
    bands: str | None = None,
) -> Scene:
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
    if bands is not None:
        scene.simulation.bands = bands
    return scene


def _values(values: list) -> np.ndarray:
    """A list from JSON, with NaN where it has null."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _plain(values: np.ndarray) -> list[float | None]:
    """For JSON: None where a value is missing."""
    return [float(v) if np.isfinite(v) else None for v in values]


def measured_parameters(entry: dict) -> RoomParameters:
    """The parameters of a measured pair (its level aside)."""
    return RoomParameters(**{k: _values(v) for k, v in entry.items() if k != "level_db"})


def _averages(pairs: dict) -> dict:
    """Per parameter, the pairs' mean measured and simulated values and differences, and
    how many differences from 125 Hz to 4 kHz are within one JND."""
    out = {}
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
            out[p] = {
                **{side: _plain(values) for side, values in mean.items()},
                "within_jnd": int(np.count_nonzero(np.abs(middle[compared]) <= 1.0)),
                "compared": int(np.count_nonzero(compared)),
                "mean_abs_jnd": float(np.abs(middle[compared]).mean()) if compared.any() else None,
            }
        errors = [pair["parameters"][p].get("standard_error_jnd") for pair in pairs.values()]
        if all(e is not None for e in errors):
            errors = np.array([_values(e) for e in errors])[:, SUMMARY]
            known = errors[np.isfinite(errors)]
            out[p]["largest_standard_error_jnd"] = float(known.max()) if known.size else None
    return out


def _seat_to_seat(levels: dict) -> dict:
    """G less the room's mean over its pairs, simulated and measured (``levels``: per
    pair, the simulated G and the measured level, whose means are taken off the same
    way), their difference in dB (1 JND) and its rms over the pairs, and how many
    differences from 125 Hz to 4 kHz are within one JND."""
    simulated = np.array([g for g, _ in levels.values()])
    measured = np.array([level for _, level in levels.values()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a band no pair has: NaN
        simulated = simulated - np.nanmean(simulated, axis=0)
        measured = measured - np.nanmean(measured, axis=0)
        difference = simulated - measured
        rms = np.sqrt(np.nanmean(difference**2, axis=0))
    middle = difference[:, SUMMARY]
    compared = np.isfinite(middle)
    return {
        "pairs": {
            key: {"simulated": _plain(s), "measured": _plain(m), "difference_db": _plain(d)}
            for key, s, m, d in zip(levels, simulated, measured, difference, strict=True)
        },
        "rms_difference_db": _plain(rms),
        "within_jnd": int(np.count_nonzero(np.abs(middle[compared]) <= 1.0)),
        "compared": int(np.count_nonzero(compared)),
        "mean_abs_db": float(np.abs(middle[compared]).mean()) if compared.any() else None,
    }


def calibrated_scene(
    name: str, scene: Scene, measured: dict, sources: list[str], rays: int
) -> tuple[Scene, dict]:
    """``scene`` with the absorption fitted to the measured responses of ``sources`` (all
    when empty), traced with ``rays`` rays; and what the fit found."""
    pairs = {
        tuple(key.split()): measured_parameters(entry)
        for key, entry in measured["pairs"].items()
        if not sources or key.split()[0] in sources
    }
    traced = scene.model_copy(deep=True)
    traced.simulation.late.rays = rays
    start = time.perf_counter()
    found = calibrate(traced, pairs, progress=lambda message: print(message, flush=True))
    seconds = time.perf_counter() - start
    fitted = found.fitted_scene()
    fitted.simulation = scene.simulation
    bras = load_scene(name, "fitted", None, None)
    report = found.as_dict()
    report["seconds"] = round(seconds, 1)
    report["calibration_rays"] = rays
    report["sources"] = sorted({s for s, _ in pairs})
    for material, entry in report["materials"].items():
        entry["bras_fitted"] = _plain(np.array(bras.materials[material].coefficients()[0]))
    return fitted, report


def validate(
    name: str,
    materials: str,
    rays: int | None,
    diffraction: str | None,
    bands: str | None = None,
    calibrate_on: list[str] | None = None,
    calibration_rays: int = 10_000,
) -> dict:
    measured = json.loads((HERE / name / "measured.json").read_text(encoding="utf-8"))
    scene = load_scene(name, materials, rays, diffraction, bands)
    calibration = None
    if calibrate_on is not None:
        scene, calibration = calibrated_scene(name, scene, measured, calibrate_on, calibration_rays)
    room = Room.from_scene(scene)
    start = time.perf_counter()

    def progress(message: str) -> None:
        print(
            f"{name} at {time.perf_counter() - start:.0f} s: {message}", file=sys.stderr, flush=True
        )

    result = simulate(room, progress=progress)
    seconds = time.perf_counter() - start
    progress("done")
    pairs = {}
    levels = {}
    for key, entry in measured["pairs"].items():
        source, listener = key.split()
        response = result[source, listener]
        simulated = response.parameters()
        pairs[key] = compare_parameters(simulated, measured_parameters(entry)).as_dict()
        errors = response.standard_errors()
        if errors is not None:
            for p, values in in_jnds(simulated, errors).items():
                if p in pairs[key]["parameters"]:
                    pairs[key]["parameters"][p]["standard_error_jnd"] = _plain(values)
        levels[key] = (np.asarray(simulated.g, dtype=float), _values(entry["level_db"]))
    out = {
        "scene": name,
        "name": scene.name,
        "materials": materials,
        "rays": scene.simulation.late.rays,
        "diffraction": scene.simulation.early.diffraction,
        "simulated_bands": scene.simulation.bands,
        "seconds": round(seconds, 1),
        "volume_m3": room.volume_m3,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "pairs": pairs,
        "g_seat_to_seat": _seat_to_seat(levels),
    }
    if calibration is not None:
        out["calibration"] = calibration
        fitted = set(calibration["sources"])
        held_out = {k: v for k, v in pairs.items() if k.split()[0] not in fitted}
        if held_out:
            pairs = held_out
            out["averaged_over"] = sorted({k.split()[0] for k in held_out})
            out["g_seat_to_seat"] = _seat_to_seat({k: levels[k] for k in held_out})
    out["average"] = _averages(pairs)
    return out


def calibration_table(result: dict) -> str:
    """The fitted absorption beside BRAS's, and the fit's errors before and after."""
    found = result["calibration"]
    bands = found["bands_hz"]
    lines = [
        f"Calibrated on {', '.join(found['sources'])} from the {result['materials']} "
        f"absorption: {found['rounds']} rounds of {found['calibration_rays']} rays, "
        f"{found['seconds']:.0f} s",
        "",
        "| absorption | " + " | ".join(_band(f) for f in bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    for material, entry in found["materials"].items():
        short = material.removeprefix(f"mat_{result['scene'].partition('_')[0]}_")
        for key, label in (
            ("initial", "start"),
            ("absorption", "calibrated"),
            ("uncertainty", "+-"),
            ("bras_fitted", "BRAS fitted"),
        ):
            cells = ["-" if v is None or v > 9 else f"{v:.2f}" for v in entry[key]]
            lines.append(f"| {short} {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| mean abs. error (JNDs) | " + " | ".join(_band(f) for f in bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    for p in found["parameters"]:
        for which in ("before", "after"):
            cells = ["-" if v is None else f"{v:.1f}" for v in found["errors_jnd"][which][p]]
            lines.append(f"| {LABELS[p]} {which} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _groups(result: dict) -> dict[str, dict]:
    """The averages a table or plot shows: over the pairs averaged (``""``), or per
    source where the sources stand in different rooms."""
    if result["scene"] not in COUPLED:
        return {"": result["average"]}
    shown = result.get("averaged_over")
    sources = sorted({k.split()[0] for k in result["pairs"]})
    return {
        source: _averages({k: v for k, v in result["pairs"].items() if k.split()[0] == source})
        for source in sources
        if shown is None or source in shown
    }


def table(result: dict) -> str:
    bands = result["bands_hz"]
    absorption = result["materials"]
    if "calibration" in result:
        absorption += " absorption, calibrated"
        if "averaged_over" in result:
            absorption += f"; averages over {', '.join(result['averaged_over'])}, not fitted"
    else:
        absorption += " absorption"
    lines = [
        f"{result['name']}: {absorption}, {result['rays']} rays per "
        f"band, {result.get('simulated_bands', 'octave').replace('_', '-')} bands, "
        f"diffraction {result['diffraction']}, simulated in {result['seconds']:.0f} s",
        "",
        "| | " + " | ".join(_band(f) for f in bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    for p in PARAMETERS:
        fmt = ".1f" if "dB" in LABELS[p] else ".2f"
        for source, averages in _groups(result).items():
            avg = averages[p]
            label = f"{LABELS[p]} {source}".rstrip()
            for side in ("measured", "simulated"):
                cells = ["-" if v is None else f"{v:{fmt}}" for v in avg[side]]
                lines.append(f"| {label} {side} | " + " | ".join(cells) + " |")
            cells = ["-" if v is None else f"{v:+.1f}" for v in avg["difference_jnd"]]
            lines.append(f"| {label} JNDs | " + " | ".join(cells) + " |")
    errors = all("largest_standard_error_jnd" in result["average"][p] for p in PARAMETERS)
    lines += [
        "",
        "| | within 1 JND, 125 Hz to 4 kHz | mean abs. difference (JNDs) |"
        + (" largest standard error (JNDs) |" if errors else ""),
        "|---|---:|---:|" + ("---:|" if errors else ""),
    ]
    for p in PARAMETERS:
        avg = result["average"][p]
        mean = avg["mean_abs_jnd"]
        share = f"{avg['within_jnd']} of {avg['compared']}"
        line = f"| {LABELS[p]} | {share} | {'-' if mean is None else f'{mean:.1f}'} |"
        if errors:
            largest = avg["largest_standard_error_jnd"]
            line += f" {'-' if largest is None else f'{largest:.2f}'} |"
        lines.append(line)
    seats = result.get("g_seat_to_seat")
    if seats is not None:
        mean = seats["mean_abs_db"]
        share = f"{seats['within_jnd']} of {seats['compared']}"
        lines.append(
            f"| G, seat to seat (dB) | {share} | {'-' if mean is None else f'{mean:.1f}'} |"
            + (" |" if errors else "")
        )
        cells = ["-" if v is None else f"{v:.1f}" for v in seats["rms_difference_db"]]
        lines += [
            "",
            "| G less the room's mean, simulated - measured | "
            + " | ".join(_band(f) for f in bands)
            + " |",
            "|---|" + "---:|" * len(bands),
            "| rms over the pairs (dB) | " + " | ".join(cells) + " |",
        ]
    return "\n".join(lines)


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = result["bands_hz"]
    x = np.arange(len(bands))
    shown = ("edt", "t20", "c80", "d50")
    groups = _groups(result)
    fig, axes = plt.subplots(1, len(shown), figsize=(12, 3.2), layout="constrained")
    for ax, p in zip(axes, shown, strict=True):
        for k, (source, averages) in enumerate(groups.items()):
            avg = averages[p]
            measured = _values(avg["measured"])
            step, relative = JND[p]
            jnd = step * np.abs(measured) if relative else np.full(len(bands), step)
            line = "-" if k == 0 else "--"
            name = f"{source} " if source else ""
            band = None if k else "1 JND"
            ax.fill_between(x, measured - jnd, measured + jnd, color="C1", alpha=0.2, label=band)
            ax.plot(x, measured, "s" + line, color="C1", label=f"{name}measured")
            ax.plot(x, _values(avg["simulated"]), "o" + line, color="C0", label=f"{name}simulated")
        ax.set_xticks(x, [f"{f // 1000}k" if f >= 1000 else str(f) for f in bands], fontsize=7)
        ax.set_xlabel("octave band (Hz)", fontsize=7)
        ax.set_title(LABELS[p])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    count = sum(
        k.split()[0] in result.get("averaged_over", [k.split()[0]]) for k in result["pairs"]
    )
    averages = f"averages of {count} responses"
    if len(groups) > 1:
        averages = f"averages of {count // len(groups)} responses per source"
    title = f"{result['name']}, {result['materials']} absorption: {averages}"
    if "calibration" in result:
        title = (
            f"{result['name']}, calibrated from the {result['materials']} absorption: "
            f"{averages}" + (", not fitted" if "averaged_over" in result else "")
        )
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", default=["CR1_DoorAngle1", "CR2", "CR3", "CR4"])
    parser.add_argument("--materials", choices=["fitted", "initial"], default="fitted")
    parser.add_argument("--rays", type=int, help="late rays (default: the scene's)")
    parser.add_argument("--diffraction", choices=["none", "direct", "first_order"])
    parser.add_argument(
        "--bands", choices=["octave", "third_octave"], help="the bands to simulate in"
    )
    parser.add_argument(
        "--calibrate",
        nargs="*",
        metavar="SOURCE",
        help="fit the absorption to the measured responses first (of these sources only)",
    )
    parser.add_argument(
        "--calibration-rays", type=int, default=10_000, help="late rays while fitting"
    )
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
            validate(
                name,
                args.materials,
                args.rays,
                args.diffraction,
                args.bands,
                args.calibrate,
                args.calibration_rays,
            )
            for name in args.scenes
        ]
    for result in results:
        if "calibration" in result:
            print(calibration_table(result) + "\n", flush=True)
        print(table(result) + "\n", flush=True)
        if args.plots:
            args.plots.mkdir(parents=True, exist_ok=True)
            bands = result.get("simulated_bands", "octave")
            suffix = "" if bands == "octave" else f"_{bands}"
            if "calibration" in result:
                suffix += "_calibrated"
            name = f"bras_{result['scene']}_{result['materials']}{suffix}.png"
            plot(result, args.plots / name)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
