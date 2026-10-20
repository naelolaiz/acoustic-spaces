"""Validation against outdoor measurements: a wind turbine heard 178, 535 and 845 m away.

    python validation/outdoor_propagation.py [--cases 1 2 ... 10] [--data DIR]
                                             [--still] [--json results.json]
                                             [--plots DIR]

The Hannover wind turbine propagation data (Könecke, Hörmeyer, Bohne and Rolfes, 2021,
DOI 10.25835/0012136, CC BY-SA 3.0) measured a wind turbine with microphones 1.7 m
above a field, 178, 535 and 845 m away on a line, in ten 5-minute spells of different
weather, and published the loss from the nearest microphone to the two others in the
third-octave bands from 80 Hz to 2 kHz, with the inputs of the authors' parabolic
equation model and its result. The script downloads the ten cases into ``--data``
(by default ``~/.cache/acoustic_spaces/hannover``) the first time; the data is not
part of this package.

Each case becomes an outdoor scene as the authors modelled it: the turbine as three
point sources of equal power 71, 119 and 167 m up (across the rotor, after Barlas et
al., 2017), a flat ground of Miki's impedance with a flow resistivity of 200 kPa s/m²,
the air's temperature and humidity measured 53 m up, and their effective speed of
sound, c(z) = a0 + alog ln(1 + z / z0) + alin z, as a measured atmosphere: the log term
as the wind (towards the microphones where alog is positive, away from them where it
is negative), the linear one as the temperature. The scene runs in third-octave bands;
the three sources' energies add at each microphone, and the loss from the 178 m one is
compared with the measured one band by band. ``Overall`` is the energy sum of the
fifteen band losses, as the data has it. ``--still`` leaves the atmosphere out.

Cases 9 and 10 are strong upwind: there the published model puts the 845 m microphone
in a deep shadow (50 dB overall where 27 dB was measured), the gap that the authors put
down to turbulence, which no model here has.

The script prints a Markdown table: per case the measured, published and simulated
overall losses, and the mean absolute difference from the measured losses over the
bands, of the published model and of this package. ``--plots`` draws the band losses
of every case (needs matplotlib). Progress goes to standard error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.air import SOUND_SPEED_0C
from acoustic_spaces.materials import THIRD_OCTAVE_BANDS_HZ

#: Where the ten cases are published (CKAN resources of DOI 10.25835/0012136).
DATASET = "https://data.uni-hannover.de/dataset/ab6c544e-677a-4780-ba8a-0df805574696/resource/"
RESOURCES = {
    1: "7b72bc05-cfc9-44ff-86ae-a5ce5080c924",
    2: "70bd5f2d-75b3-476e-9d9c-7cd381bfb976",
    3: "265df405-965e-404c-aa62-e720464fee9e",
    4: "c9ee335d-44af-4f00-b003-0fa7c0f568a2",
    5: "0369c1b3-6863-480d-a9a3-04f722b7ff0b",
    6: "c631cec7-ec63-4290-a3bb-615fc440873f",
    7: "751a78fa-143c-48a2-aebe-d069814d2dc4",
    8: "936f756f-7690-4fc3-9ee1-07a5e1712e8b",
    9: "d488b934-0189-48e8-8c3a-0c1fe42259c9",
    10: "47cabb9e-285d-42c2-9e9d-e2e38a088f77",
}
#: The weather of each case, as the data's readme names it.
WEATHER = {
    1: "light downwind",
    2: "light downwind",
    3: "cross/downwind",
    4: "cross/downwind",
    5: "cross/upwind",
    6: "cross/upwind",
    7: "light upwind",
    8: "light upwind",
    9: "strong upwind",
    10: "strong upwind",
}
#: Heights of the measured atmosphere's rows (m): the profile's log term changes fastest
#: near the ground.
HEIGHTS = np.concatenate([[0.0], np.geomspace(0.01, 1000.0, 120)])


def fetch(case: int, folder: Path) -> Path:
    """The case's file in ``folder``, downloaded if it is not there yet."""
    path = folder / f"case{case:02d}.mat"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        url = f"{DATASET}{RESOURCES[case]}/download/case{case:02d}.mat"
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    return path


def load(path: Path):
    from scipy.io import loadmat

    return loadmat(path, squeeze_me=True, struct_as_record=False)


def scene_of(case: int, inputs, still: bool = False) -> Scene:
    """The outdoor scene of a case's model inputs."""
    heights = np.asarray(inputs.hs, dtype=float)
    ranges = np.asarray(inputs.R, dtype=float)
    hr = float(inputs.hr)
    a0, alog, alin, z0 = (float(inputs.a0), float(inputs.alog), float(inputs.alin), float(inputs.z0))
    # The speed of sound without the log term, as a temperature; the log term as wind.
    speed = a0 + alin * HEIGHTS
    temperature = 273.15 * ((speed / SOUND_SPEED_0C) ** 2 - 1.0)
    wind = abs(alog) * np.log1p(HEIGHTS / z0)
    data = {
        "name": f"Hannover case {case}",
        "geometry": {
            "ground": {
                "extent": [[-60.0, -60.0], [float(ranges.max()) + 60.0, 60.0]],
                "sky": float(heights.max()) + 40.0,
            }
        },
        "materials": {
            "field": {
                "impedance": {"model": "miki", "flow_resistivity": float(inputs.sigma_G) / 1000.0}
            }
        },
        "assignments": {"ground": "field"},
        "sources": [
            {"name": f"{h:g} m up", "position": [0.0, 0.0, float(h)]} for h in heights
        ],
        "listeners": [{"name": f"{r:g} m", "position": [float(r), 0.0, hr]} for r in ranges],
        "air": {"temperature_c": float(inputs.T), "humidity_percent": float(inputs.Hu)},
        "simulation": {
            "bands": "third_octave",
            "seed": 1,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
            "late": {"rays": 2000},
            "ir_length_s": 0.3,
        },
    }
    if not still:
        data["atmosphere"] = {
            "table": [[float(z), float(t), float(u)] for z, t, u in zip(HEIGHTS, temperature, wind)],
            "wind": {"direction_deg": 0.0 if alog >= 0.0 else 180.0},
        }
    return Scene.model_validate(data)


def run_case(case: int, folder: Path, still: bool) -> dict:
    content = load(fetch(case, folder))
    inputs, measured, model = content["Inputparameter"], content["MeasurementsLoss"], content["ModelLoss"]
    scene = scene_of(case, inputs, still)
    result = simulate(Room.from_scene(scene))
    listeners = [lis.name for lis in scene.listeners]
    energy = np.zeros((len(listeners), len(THIRD_OCTAVE_BANDS_HZ)))
    for key in result.responses:
        energy[listeners.index(key[1])] += result[key].energy().sum(axis=1)
    frequencies = np.asarray(measured.Frequency, dtype=float)
    frequencies[frequencies == 1200.0] = 1250.0  # the data's name for the 1.25 kHz band
    columns = [int(np.argmin(np.abs(np.asarray(THIRD_OCTAVE_BANDS_HZ) - f))) for f in frequencies]
    loss = (10.0 * np.log10(energy[0] / energy[1:]))[:, columns].T  # (bands, 2)
    measured_loss = np.asarray(measured.Spectra, dtype=float)
    model_loss = np.asarray(model.Spectra, dtype=float)
    return {
        "weather": WEATHER[case],
        "bands_hz": frequencies.tolist(),
        "distances_m": [float(r) for r in np.asarray(inputs.R, dtype=float)[1:]],
        "measured": measured_loss.T.tolist(),
        "measured_std": np.asarray(measured.Spectra_Std, dtype=float).T.tolist(),
        "published_model": model_loss.T.tolist(),
        "simulated": loss.T.tolist(),
        "overall": {
            "measured": np.asarray(measured.Overall, dtype=float).tolist(),
            "published_model": np.asarray(model.Overall, dtype=float).tolist(),
            "simulated": overall(loss).tolist(),
        },
        "mean_abs_difference": {
            "published_model": np.abs(model_loss - measured_loss).mean(axis=0).tolist(),
            "simulated": np.abs(loss - measured_loss).mean(axis=0).tolist(),
        },
    }


def overall(loss: np.ndarray) -> np.ndarray:
    """The energy sum of band losses (bands, n), as the data's ``Overall``."""
    return 10.0 * np.log10(np.sum(10.0 ** (loss / 10.0), axis=0))


def table(results: dict) -> str:
    lines = [
        "| Case | Weather | Overall, measured | published model | this package "
        "| Mean abs. difference, published model | this package |",
        "|---|---|---|---|---|---|---|",
    ]
    for case, r in results.items():
        o, d = r["overall"], r["mean_abs_difference"]

        def pair(values):
            return " / ".join(f"{v:.1f}" for v in values)

        lines.append(
            f"| {case} | {r['weather']} | {pair(o['measured'])} | {pair(o['published_model'])} "
            f"| {pair(o['simulated'])} | {pair(d['published_model'])} | {pair(d['simulated'])} |"
        )
    return "\n".join(lines)


def plot(results: dict, folder: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    for case, r in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        f = np.asarray(r["bands_hz"])
        for k, ax in enumerate(axes):
            measured, std = np.asarray(r["measured"][k]), np.asarray(r["measured_std"][k])
            ax.fill_between(f, measured - std, measured + std, color="0.85", label="measured ± std")
            ax.plot(f, measured, "k-", label="measured")
            ax.plot(f, r["published_model"][k], "--", label="published model")
            ax.plot(f, r["simulated"][k], "o-", ms=3, label="acoustic_spaces")
            ax.set_xscale("log")
            ax.set_title(f"case {case}, 178 to {r['distances_m'][k]:g} m")
            ax.set_xlabel("third-octave band (Hz)")
        axes[0].set_ylabel("loss (dB)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f"hannover_case{case:02d}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, nargs="+", default=list(RESOURCES))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.home() / ".cache" / "acoustic_spaces" / "hannover",
        help="where the cases are kept (downloaded there if missing)",
    )
    parser.add_argument("--still", action="store_true", help="leave the atmosphere out")
    parser.add_argument("--json", type=Path, help="write every number to this file")
    parser.add_argument("--plots", type=Path, help="draw a PNG per case into this folder")
    args = parser.parse_args()
    results = {}
    start = time.time()
    for case in args.cases:
        print(f"case {case} ({time.time() - start:.0f} s)", file=sys.stderr)
        results[case] = run_case(case, args.data, args.still)
    print(table(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    if args.plots:
        plot(results, args.plots)


if __name__ == "__main__":
    main()
