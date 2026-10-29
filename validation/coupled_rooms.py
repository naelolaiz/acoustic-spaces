"""BRAS's coupled rooms against the diffuse theory of two rooms joined by an opening.

    python validation/coupled_rooms.py [--scene CR1_DoorAngle1] [--materials fitted]
                                       [--opening 0.1] [--results results.json]
                                       [--json numbers.json]

BRAS's CR1 (``validation/bras/``) is a laboratory and a reverberation chamber joined by
a door ajar. BRAS fitted the absorption to each room on its own by Eyring's formula,
with each room's volume and surface as its material file gives them (mat_CR1.txt:
the laboratory 102.3 m³ and 194.7 m², the chamber 121.4 m³ and 175.8 m²). In the
diffuse theory of two coupled rooms (Kuttruff, Room Acoustics) the energy
densities w1, w2 of rooms of volumes V1, V2 and absorption areas A1, A2, joined by an
opening of area S, follow

    V1 dw1/dt = c/4 (-(A1 + S) w1 + S w2),    V2 dw2/dt = c/4 (S w1 - (A2 + S) w2),

and the sound of a source in the chamber dies away in the laboratory, once the faster
of the two decays has gone, as the slower one, in the time 6 ln 10 / rate.

In the model with the door at 4.1 degrees the rooms meet only where the leaf leaves a
gap: between its free end and the jamb, 4 cm across (from the leaf's corner at x 0.607,
y -1.468 m to the jamb's at 0.571, -1.488 m) over the door's 2.14 m, and over the
wedge of air between the leaf and the wall, open above the leaf (0.01 m²): about
0.1 m² in all, the default ``--opening``.

Per octave band from 125 Hz to 2 kHz the script prints each room's Eyring time with the
scene's absorption and air (the chamber's paintedConcrete, the laboratory's other
materials, over BRAS's surfaces and volumes), the slower decay of the two rooms joined
by ``--opening``, the measured T20 of LS1 (in the chamber) at the laboratory's
microphones, averaged over the two, and the opening that gives it in the theory; with
``--results`` (the JSON of ``bras_validation.py`` for the same scene and materials) also
the simulated T20 and the opening that gives that.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bras_validation import HERE, load_scene  # noqa: E402

from acoustic_spaces import Room  # noqa: E402

#: Each room's volume (m³) and surface (m²) as BRAS fitted its absorption to it.
LABORATORY = (102.3, 194.7)
CHAMBER = (121.4, 175.8)
#: The chamber's one material; the laboratory has the others.
CHAMBER_MATERIAL = "mat_CR1_paintedConcrete"
#: The source in the chamber, heard in the laboratory.
SOURCE = "LS1"
BANDS_HZ = (125, 250, 500, 1000, 2000)


def eyring_area(alpha: np.ndarray, surface: float, volume: float, air: np.ndarray) -> np.ndarray:
    """The absorption area (m²) of Eyring's formula, the air's 4 m V included."""
    return -surface * np.log1p(-np.clip(alpha, 0.0, 1.0 - 1e-12)) + 4.0 * air * volume


def slow_time(a1: float, a2: float, opening: float, c: float) -> float:
    """The slower decay time (s) of the two rooms (absorption areas ``a1``, ``a2``, m²)
    joined by ``opening`` (m²)."""
    v1, v2 = LABORATORY[0], CHAMBER[0]
    rates = np.array([[-(a1 + opening) / v1, opening / v1], [opening / v2, -(a2 + opening) / v2]])
    slowest = float(np.min(-np.linalg.eigvals(c / 4.0 * rates).real))
    return 6.0 * math.log(10.0) / slowest


def opening_for(time: float, a1: float, a2: float, c: float) -> float:
    """The opening (m²) whose slower decay takes ``time`` (s), by bisection; nan when
    no opening from 1 cm² to 20 m² gives it."""
    low, high = 1e-4, 20.0
    if not slow_time(a1, a2, high, c) <= time <= slow_time(a1, a2, low, c):
        return math.nan
    for _ in range(60):
        middle = math.sqrt(low * high)
        if slow_time(a1, a2, middle, c) > time:
            low = middle
        else:
            high = middle
    return math.sqrt(low * high)


def mean_t20(pairs: dict, bands: list[int], value) -> np.ndarray:
    """The T20 of :data:`SOURCE`'s pairs, averaged over them, in :data:`BANDS_HZ`."""
    rows = [value(entry) for name, entry in pairs.items() if name.split()[0] == SOURCE]
    columns = [bands.index(f) for f in BANDS_HZ]
    return np.mean(np.array(rows, dtype=float)[:, columns], axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene", default="CR1_DoorAngle1", choices=["CR1_DoorAngle1", "CR1_DoorAngle3"]
    )
    parser.add_argument("--materials", default="fitted", choices=["fitted", "initial"])
    parser.add_argument("--opening", type=float, default=0.1, help="the opening (m²)")
    parser.add_argument("--results", type=Path, help="bras_validation.py's JSON for the scene")
    parser.add_argument("--json", type=Path, help="write the numbers to this JSON file")
    args = parser.parse_args(argv)

    room = Room.from_scene(load_scene(args.scene, args.materials, None, None))
    centres = list(room.band_centres)
    columns = [centres.index(f) for f in BANDS_HZ]
    areas = room.area_by_material()
    alpha = {
        name: np.array(room.scene.materials[name].coefficients("octave")[0])[columns]
        for name in areas
    }
    lab = [name for name in areas if name != CHAMBER_MATERIAL]
    lab_alpha = sum(areas[n] * alpha[n] for n in lab) / sum(areas[n] for n in lab)
    air = room.air_attenuation()[columns]
    c = room.speed_of_sound()
    a1 = eyring_area(lab_alpha, LABORATORY[1], LABORATORY[0], air)
    a2 = eyring_area(alpha[CHAMBER_MATERIAL], CHAMBER[1], CHAMBER[0], air)
    t_lab = 24.0 * math.log(10.0) * LABORATORY[0] / c / a1
    t_chamber = 24.0 * math.log(10.0) * CHAMBER[0] / c / a2

    joined = np.array([slow_time(x, y, args.opening, c) for x, y in zip(a1, a2, strict=True)])
    measured_file = json.loads((HERE / args.scene / "measured.json").read_text())
    measured = mean_t20(measured_file["pairs"], measured_file["bands_hz"], lambda e: e["t20"])
    out: dict = {
        "scene": args.scene,
        "materials": args.materials,
        "bands_hz": list(BANDS_HZ),
        "eyring_laboratory_s": t_lab.tolist(),
        "eyring_chamber_s": t_chamber.tolist(),
        "opening_m2": args.opening,
        "joined_s": joined.tolist(),
        "measured_t20_s": measured.tolist(),
        "measured_opening_m2": [
            opening_for(t, x, y, c) for t, x, y in zip(measured, a1, a2, strict=True)
        ],
    }
    rows = [
        ("chamber alone, Eyring (s)", t_chamber, "{:8.2f}"),
        ("laboratory alone, Eyring (s)", t_lab, "{:8.2f}"),
        (f"joined by {args.opening:g} m² (s)", joined, "{:8.2f}"),
        (f"measured T20 of {SOURCE} (s)", measured, "{:8.2f}"),
        ("opening that gives it (m²)", np.array(out["measured_opening_m2"]), "{:8.2f}"),
    ]
    if args.results:
        result = json.loads(args.results.read_text())[0]
        simulated = mean_t20(
            result["pairs"],
            result["bands_hz"],
            lambda e: [np.nan if v is None else v for v in e["parameters"]["t20"]["simulated"]],
        )
        out["simulated_t20_s"] = simulated.tolist()
        out["simulated_opening_m2"] = [
            opening_for(t, x, y, c) for t, x, y in zip(simulated, a1, a2, strict=True)
        ]
        rows += [
            (f"simulated T20 of {SOURCE} (s)", simulated, "{:8.2f}"),
            ("opening that gives it (m²)", np.array(out["simulated_opening_m2"]), "{:8.3f}"),
        ]
    print(f"{args.scene}, {args.materials} materials: two diffuse rooms joined by an opening\n")
    print(f"{'':32}" + "".join(f"{f:>8}" for f in BANDS_HZ))
    for label, values, form in rows:
        print(f"{label:32}" + "".join(form.format(v) for v in values))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
