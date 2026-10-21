"""Check the levels over a long term: how the weather classes and the periods of the
day combine, and the meteorological correction that gives against ISO 9613-2's.

    python validation/long_term.py [--json results.json]

A scene's ``long_term`` runs it once in each weather class and combines the runs (see
:mod:`acoustic_spaces.engine.long_term`). The scene here is a source 1 m over grass
(Miki's impedance, 200 kPa s/m²), the same sound power in every octave band, and
listeners 1.5 m up from 50 to 800 m away along the wind, in four weather classes:
downwind and upwind (the ``windy`` preset, 5 m/s 10 m up, towards the listeners or
away from them), still air (``neutral``) and a clear night's inversion
(``clear_night``), with shares of the day, the evening and the night as a weather
statistic might give them: the inversion holds half the night and none of the day.
Two checks:

- **The arithmetic.** The engine's period levels, long-term levels and L_den against
  the same formulas worked here from each class's own run: the energy mean over the
  classes by their shares (ISO 1996-2, CNOSSOS-EU), over the periods by their hours,
  and L_den with 5 dB in the evening and 10 dB at night (Directive 2002/49/EC). They
  agree to rounding.
- **The meteorological correction.** ISO 9613-2 predicts the level in conditions
  favourable to propagation, downwind or under a moderate ground-based inversion such
  as clear nights bring (its clause 5), and takes the long-term level from it as L_LT =
  L_DW - C_met, with C_met = C_0 (1 - 10 (h_s + h_r) / d_p) beyond d_p = 10 (h_s + h_r)
  and 0 nearer, C_0 from 0 to 5 dB by the local weather statistics. The engine's C_met
  is the A-weighted level in the favourable classes (downwind and the inversion,
  combined by their shares of the whole day) less the long-term one; the script fits
  C_0 to it and gives the difference. This is a check of plausibility, not of
  accuracy: the standard's correction is an empirical fit, and the engine's comes from
  the classes' own physics (validated against a fast field program and a
  Crank-Nicholson parabolic equation).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acoustic_spaces import Room, Scene  # noqa: E402
from acoustic_spaces.engine import long_term  # noqa: E402

ZS, ZR = 1.0, 1.5
RANGES = (50.0, 100.0, 200.0, 400.0, 800.0)
PERIODS = ("day", "evening", "night")
HOURS = np.array([12.0, 4.0, 8.0])
PENALTIES = np.array([0.0, 5.0, 10.0])
#: The weather classes favourable to propagation, ISO 9613-2's reference condition.
FAVOURABLE = ("downwind", "inversion")
#: The weather classes and their shares of the day, the evening and the night.
CLASSES = [
    {"name": "downwind", "share": {"day": 0.30, "evening": 0.25, "night": 0.15}},
    {
        "name": "upwind",
        "share": {"day": 0.30, "evening": 0.25, "night": 0.15},
        "wind": {"direction_deg": 180.0},
    },
    {"name": "still", "share": {"day": 0.40, "evening": 0.30, "night": 0.20}, "preset": "neutral"},
    {
        "name": "inversion",
        "share": {"day": 0.0, "evening": 0.20, "night": 0.50},
        "preset": "clear_night",
    },
]


def scene() -> Scene:
    far = max(RANGES)
    return Scene.model_validate(
        {
            "name": "long term",
            "geometry": {"ground": {"extent": [[-60, -60], [far + 60, 60]], "sky": 40}},
            "materials": {
                "grass": {"impedance": {"model": "miki", "flow_resistivity": 200}},
            },
            "default_material": "grass",
            "sources": [{"name": "s", "position": [0, 0, ZS]}],
            "listeners": [{"name": f"{d:g} m", "position": [d, 0, ZR]} for d in RANGES],
            "atmosphere": {"preset": "windy", "wind": {"direction_deg": 0.0}},
            "simulation": {
                "ir_length_s": far / 340.0 + 0.3,
                "seed": 1,
                "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
                "late": {"rays": 500},
            },
            "long_term": {"classes": CLASSES},
        }
    )


def energy_mean(levels, weights) -> np.ndarray:
    """10 lg Σ w 10^(L/10) over the first axis, written out here."""
    levels = np.asarray(levels, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1, *[1] * (levels.ndim - 1))
    terms = np.where(weights > 0.0, weights * 10.0 ** (levels / 10.0), 0.0)
    return 10.0 * np.log10(terms.sum(axis=0))


def iso_correction(distance: float, c0: float) -> float:
    """ISO 9613-2 eq. (21)-(22): C_met for the source and listener heights."""
    near = 10.0 * (ZS + ZR)
    return 0.0 if distance <= near else c0 * (1.0 - near / distance)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    model = scene()
    result = long_term.simulate(Room.from_scene(model), threads=None)
    levels = result.long_term
    seconds = time.perf_counter() - started
    power = model.sources[0].power_db("octave")
    shares = np.array([[c["share"][p] for p in PERIODS] for c in CLASSES])
    names = [c["name"] for c in CLASSES]
    print(
        f"{len(names)} weather classes, {result.details['long_term']['runs']:g} runs "
        f"(the downwind class is the scene's own weather), {seconds:.0f} s\n"
    )

    # The arithmetic, worked here from each class's own run.
    worst = 0.0
    rows = []
    for d in RANGES:
        key = ("s", f"{d:g} m")
        per_class = np.array([levels.runs[n].responses[key].levels(power) for n in names])
        per_period = np.array([energy_mean(per_class, shares[:, p]) for p in range(3)])
        mean = energy_mean(per_period, HOURS / HOURS.sum())
        period_a = np.array([long_term.a_weighted(level) for level in per_period])
        den = 10.0 * math.log10(np.sum(HOURS * 10.0 ** ((period_a + PENALTIES) / 10.0)) / 24.0)
        worst = max(
            worst,
            float(np.max(np.abs(per_period - levels.period_levels[key]))),
            float(np.max(np.abs(mean - levels.levels[key]))),
            abs(den - levels.rating_level[key]),
        )
        class_a = [float(long_term.a_weighted(level)) for level in per_class]
        # The favourable classes together, by their shares of the whole day.
        whole_day = shares @ HOURS / HOURS.sum()
        favourable = [names.index(n) for n in FAVOURABLE]
        weights = whole_day[favourable] / whole_day[favourable].sum()
        level_f = float(energy_mean(np.array(class_a)[favourable], weights))
        rows.append(
            {
                "distance_m": d,
                "class_a_db": dict(zip(names, class_a, strict=True)),
                "period_a_db": dict(zip(PERIODS, period_a.tolist(), strict=True)),
                "long_term_a_db": float(levels.levels_a[key]),
                "l_den_db": float(levels.rating_level[key]),
                "favourable_a_db": level_f,
                "c_met_db": level_f - float(levels.levels_a[key]),
            }
        )
    print("The level in each weather class, each period and over the long term (dB(A))\n")
    header = "".join(f"{n:>11}" for n in names) + "".join(f"{p:>9}" for p in PERIODS)
    print(f"{'':>8}{header}{'long term':>11}{'L_den':>8}")
    for row in rows:
        cells = "".join(f"{row['class_a_db'][n]:11.1f}" for n in names)
        cells += "".join(f"{row['period_a_db'][p]:9.1f}" for p in PERIODS)
        print(
            f"{row['distance_m']:6.0f} m{cells}{row['long_term_a_db']:11.1f}{row['l_den_db']:8.1f}"
        )
    print(f"\nthe engine against the formulas worked here: {worst:.1e} dB at most\n")

    # The meteorological correction against ISO 9613-2's, C_0 fitted by least squares.
    distances = np.array(RANGES)
    shape = np.array([iso_correction(d, 1.0) for d in distances])
    engine = np.array([row["c_met_db"] for row in rows])
    c0 = float(shape @ engine / (shape @ shape))
    iso = c0 * shape
    print("The meteorological correction C_met = L_A,favourable - L_A,long term (dB)\n")
    print(f"{'':>8}{'engine':>9}{'ISO 9613-2':>12}{'difference':>12}")
    for d, e, i in zip(distances, engine, iso, strict=True):
        print(f"{d:6.0f} m{e:9.2f}{i:12.2f}{e - i:+12.2f}")
    spread = float(np.max(np.abs(engine - iso)))
    print(f"\nC_0 fitted: {c0:.2f} dB (ISO 9613-2: 0 to 5 dB); within {spread:.2f} dB of it")

    if args.json:
        data = {
            "arithmetic_max_db": worst,
            "c0_db": c0,
            "c_met_max_difference_db": spread,
            "rows": rows,
            "seconds": seconds,
        }
        args.json.write_text(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
