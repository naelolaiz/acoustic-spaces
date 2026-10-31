"""Validation of binaural responses against measured ones: IACC in BRAS's seminar room,
chamber music hall and auditorium.

    python validation/bras_binaural.py path/to/BRAS [--scenes CR2 CR3 CR4]
                                       [--sources LS3 LS4 LS5 LS6 LS7]
                                       [--materials fitted] [--rays 50000]
                                       [--diffraction first_order] [--seeds 10]
                                       [--json results.json]
    python validation/bras_binaural.py --from-json results.json

BRAS measured the responses of five QSC K8 loudspeakers, LS3 to LS7, one at a time, at
FABIAN, a head and torso simulator facing LS7 (``validation/bras/CRx/binaural.json``,
made by ``prepare.py`` with the setup and the measured IACC_E and IACC_L of each). Each
loudspeaker is simulated in the room as ``bras_validation.py`` simulates the room, with
the K8's measured directivity (``validation/bras/QSC_K8.csv``), and the response
rendered through FABIAN's head-related impulse responses with its head straight on the
torso, every 2 degrees: BRAS's ``2 Source and receiver descriptions/FABIAN HRIRs/
CSV-files`` (``Fabian HRIRs.zip`` in ``2_source_and_receiver_descriptions-
FABIAN_HRIRs.zip``, unpacked where it lies; only its two ``HATO_0`` files are read).
The late part of a rendered response is noise shaped by the energy and the arrival
directions the rays found, drawn anew for each of ``--seeds`` seeds: IACC_E and IACC_L
are the mean over them, with their standard deviation, compared with the measured
ones in JNDs (0.075, ISO 3382-1, annex B).

The script prints a Markdown table per room: the measured and simulated IACCs averaged
over its loudspeakers, the mean difference in JNDs, and how many of the loudspeaker and
band differences from 125 Hz to 4 kHz are within one JND. Each loudspeaker is simulated
on its own; with ``--json`` its numbers are written as soon as they are found, and a
run started again with the same file goes on from there. Progress goes to standard
error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bras_validation import HERE, _band, _plain, _values, load_scene  # noqa: E402

from acoustic_spaces import Room, simulate  # noqa: E402
from acoustic_spaces.compare import JND, SUMMARY_BANDS_HZ  # noqa: E402
from acoustic_spaces.materials import OCTAVE_BANDS_HZ  # noqa: E402
from acoustic_spaces.parameters import interaural_cross_correlation  # noqa: E402
from acoustic_spaces.scene.model import Listener, Scene, Source  # noqa: E402
from acoustic_spaces.spatial.hrtf import HRTF  # noqa: E402
from acoustic_spaces.spatial.render import Binaural  # noqa: E402

ROOMS = ("CR2", "CR3", "CR4")
NAMES = ("iacc_e", "iacc_l")
LABELS = {"iacc_e": "IACC_E", "iacc_l": "IACC_L"}
#: The grid step of FABIAN's responses as read (degrees; BRAS gives every degree).
FABIAN_STEP_DEG = 2
SUMMARY = np.isin(OCTAVE_BANDS_HZ, SUMMARY_BANDS_HZ)


def fabian(bras: Path, step: int = FABIAN_STEP_DEG) -> HRTF:
    """FABIAN's head-related impulse responses with the head straight on the torso, every
    ``step`` degrees of azimuth and elevation."""
    folder = bras / "2 Source and receiver descriptions" / "FABIAN HRIRs" / "CSV-files"
    ears = []
    for ear in "LR":
        rows = {}
        with (folder / f"HATO_0_1x1_64442_HRIR_{ear}_top_pole.csv").open(encoding="utf-8") as f:
            for line in f:
                label, _, rest = line.partition(",")  # A000E-90
                azimuth, elevation = int(label[1:4]), int(label[5:8])
                pole = abs(elevation) == 90
                if elevation % step or (azimuth if pole else azimuth % step):
                    continue  # each pole once
                rows[azimuth, elevation] = np.array(rest.split(","), dtype=float)
        ears.append(rows)
    keys = sorted(ears[0])
    # Top pole (BRAS documentation, figure 3b): azimuth from +x towards +y (the left),
    # elevation up from the horizontal plane.
    azimuth, elevation = np.radians(np.array(keys, dtype=float)).T
    directions = np.stack(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        axis=1,
    )
    hrir = np.stack([np.stack([ears[0][k], ears[1][k]]) for k in keys])
    return HRTF(directions, hrir, 44_100.0, "FABIAN, BRAS (head straight)")


def loudspeaker_scene(
    name: str, loudspeaker: str, materials: str, rays: int | None, diffraction: str | None
) -> tuple[Scene, dict]:
    """The room with one loudspeaker and FABIAN in it, as measured, and the setup of the
    room's binaural measurements (``binaural.json``)."""
    setup = json.loads((HERE / name / "binaural.json").read_text(encoding="utf-8"))
    scene = load_scene(name, materials, rays, diffraction)
    source = setup["sources"][loudspeaker]
    scene.sources = [
        Source.model_validate(
            {
                "name": loudspeaker,
                "position": source["position"],
                "orientation": {"yaw": source["yaw"], "pitch": source["pitch"]},
                "directivity": {"file": str(HERE / "QSC_K8.csv")},
            }
        )
    ]
    head = setup["listener"]
    scene.listeners = [
        Listener.model_validate(
            {
                "name": head["name"],
                "position": head["position"],
                "orientation": {"yaw": head["yaw"], "pitch": head["pitch"]},
            }
        )
    ]
    return scene, setup


def simulate_source(
    name: str,
    loudspeaker: str,
    hrtf: HRTF,
    materials: str,
    rays: int | None,
    diffraction: str | None,
    seeds: int,
) -> dict:
    """IACC_E and IACC_L of one loudspeaker's simulated response, over ``seeds`` seeds of
    the rendering, against the measured ones."""
    scene, setup = loudspeaker_scene(name, loudspeaker, materials, rays, diffraction)
    head = setup["listener"]
    start = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"{name} {loudspeaker} at {elapsed:.0f} s: {message}", file=sys.stderr, flush=True)

    response = simulate(Room.from_scene(scene), progress=progress)[loudspeaker, head["name"]]
    simulated_s = time.perf_counter() - start
    progress("rendering")
    binaural = Binaural(hrtf)
    rate = int(setup["sample_rate"])
    found: dict[str, list] = {n: [] for n in NAMES}
    for seed in range(seeds):
        ir = response.spatial_impulse_response(binaural, rate, seed=seed)
        for n, values in interaural_cross_correlation(ir, rate).items():
            found[n].append(values)
    out: dict = {
        "simulated_s": round(simulated_s, 1),
        "rendered_s": round(time.perf_counter() - start - simulated_s, 1),
    }
    for n in NAMES:
        runs = np.array(found[n])
        mean = runs.mean(axis=0)
        measured = _values(setup["iacc"][loudspeaker][n])
        out[n] = {
            "simulated": _plain(mean),
            "sd": _plain(runs.std(axis=0, ddof=1) if seeds > 1 else np.full(len(mean), np.nan)),
            "measured": _plain(measured),
            "difference_jnd": _plain((mean - measured) / JND[n][0]),
        }
    return out


def _averages(sources: dict) -> dict:
    """Per IACC, the loudspeakers' mean measured and simulated values, the mean of the
    render's standard deviations and of the differences, and how many differences from
    125 Hz to 4 kHz are within one JND."""
    out = {}
    for n in NAMES:
        stack = {
            side: np.array([_values(entry[n][side]) for entry in sources.values()])
            for side in ("simulated", "sd", "measured", "difference_jnd")
        }
        middle = stack["difference_jnd"][:, SUMMARY]
        compared = np.isfinite(middle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a band no loudspeaker has
            mean = {side: np.nanmean(values, axis=0) for side, values in stack.items()}
        out[n] = {
            **{side: _plain(values) for side, values in mean.items()},
            "within_jnd": int(np.count_nonzero(np.abs(middle[compared]) <= 1.0)),
            "compared": int(np.count_nonzero(compared)),
            "mean_abs_jnd": float(np.abs(middle[compared]).mean()) if compared.any() else None,
        }
    return out


def validate(
    bras: Path | None,
    name: str,
    loudspeakers: list[str] | None,
    materials: str,
    rays: int | None,
    diffraction: str | None,
    seeds: int,
    done: dict | None = None,
    save=None,
    hrtf: list | None = None,
) -> dict:
    """A room's loudspeakers, those in ``done`` (an earlier run's) kept; ``save`` is
    called with the room's result after each. ``hrtf`` holds FABIAN's responses once
    read (they are read when first needed)."""
    setup = json.loads((HERE / name / "binaural.json").read_text(encoding="utf-8"))
    scene = load_scene(name, materials, rays, diffraction)
    settings = {
        "materials": materials,
        "rays": scene.simulation.late.rays,
        "diffraction": scene.simulation.early.diffraction,
        "seeds": seeds,
    }
    if done is not None and any(done[k] != v for k, v in settings.items()):
        raise SystemExit(f"{name}: the earlier run in the JSON file had other settings")
    result = done or {
        "scene": name,
        "name": scene.name,
        **settings,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "sources": {},
    }
    hrtf = [] if hrtf is None else hrtf
    for loudspeaker in loudspeakers or list(setup["sources"]):
        if loudspeaker in result["sources"]:
            continue
        if not hrtf:
            if bras is None:
                raise SystemExit("FABIAN's responses are read from the BRAS folder: give it")
            hrtf.append(fabian(bras))
        result["sources"][loudspeaker] = simulate_source(
            name, loudspeaker, hrtf[0], materials, rays, diffraction, seeds
        )
        result["average"] = _averages(result["sources"])
        if save is not None:
            save(result)
    return result


def table(result: dict) -> str:
    bands = result["bands_hz"]
    sources = result["sources"]
    seconds = sum(s["simulated_s"] for s in sources.values())
    lines = [
        f"{result['name']}: {result['materials']} absorption, {result['rays']} rays per band, "
        f"diffraction {result['diffraction']}; {len(sources)} loudspeakers "
        f"({', '.join(sources)}), simulated in {seconds:.0f} s, each rendered with "
        f"{result['seeds']} seeds",
        "",
        "| | " + " | ".join(_band(f) for f in bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    for n in NAMES:
        avg = result["average"][n]
        for side, label in (("measured", "measured"), ("simulated", "simulated")):
            cells = ["-" if v is None else f"{v:.2f}" for v in avg[side]]
            lines.append(f"| {LABELS[n]} {label} | " + " | ".join(cells) + " |")
        cells = ["-" if v is None else f"{v:.2f}" for v in avg["sd"]]
        lines.append(f"| {LABELS[n]} sd over seeds | " + " | ".join(cells) + " |")
        cells = ["-" if v is None else f"{v:+.1f}" for v in avg["difference_jnd"]]
        lines.append(f"| {LABELS[n]} JNDs | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| | within 1 JND, 125 Hz to 4 kHz | mean abs. difference (JNDs) |",
        "|---|---:|---:|",
    ]
    for n in NAMES:
        avg = result["average"][n]
        mean = avg["mean_abs_jnd"]
        share = f"{avg['within_jnd']} of {avg['compared']}"
        lines.append(f"| {LABELS[n]} | {share} | {'-' if mean is None else f'{mean:.1f}'} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bras", type=Path, nargs="?", help="folder of the unpacked BRAS archives")
    parser.add_argument("--scenes", nargs="+", default=list(ROOMS), choices=ROOMS)
    parser.add_argument("--sources", nargs="+", help="loudspeakers (default: LS3 to LS7)")
    parser.add_argument("--materials", choices=["fitted", "initial"], default="fitted")
    parser.add_argument("--rays", type=int, help="late rays (default: the scene's)")
    parser.add_argument("--diffraction", choices=["none", "direct", "first_order"])
    parser.add_argument("--seeds", type=int, default=10, help="seeds of the rendering")
    parser.add_argument("--json", type=Path, help="write every number to this file")
    parser.add_argument("--from-json", type=Path, help="print the results of an earlier run")
    args = parser.parse_args()
    if args.from_json:
        results = json.loads(args.from_json.read_text(encoding="utf-8"))
    else:
        if args.bras is None:
            parser.error("give the BRAS folder, or --from-json")
        results = []
        if args.json and args.json.exists():
            results = json.loads(args.json.read_text(encoding="utf-8"))
        earlier = {r["scene"]: r for r in results}
        hrtf: list = []

        def save(result: dict) -> None:
            if args.json:
                by_scene = {r["scene"]: r for r in results} | {result["scene"]: result}
                text = json.dumps(list(by_scene.values()), indent=1)
                args.json.write_text(text + "\n", encoding="utf-8")

        for name in args.scenes:
            result = validate(
                args.bras,
                name,
                args.sources,
                args.materials,
                args.rays,
                args.diffraction,
                args.seeds,
                earlier.get(name),
                save,
                hrtf,
            )
            results = [r for r in results if r["scene"] != name] + [result]
    for result in results:
        if result["sources"]:
            print(table(result) + "\n", flush=True)


if __name__ == "__main__":
    main()
