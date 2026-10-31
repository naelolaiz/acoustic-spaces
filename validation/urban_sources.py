"""Check the pass-by demos' siren and road vehicles against rules and data published
apart from the models they come from.

    python validation/urban_sources.py [--json results.json]

- **California's rule.** The police car's siren (:mod:`acoustic_spaces.demos.siren`),
  wailing and yelping, heard 3.0 m away from 0 to 50 degrees off its axis, as the
  engine gives it with the siren's directivity table, over a ground that takes all the
  sound (a free field), against the least levels of California's two classes of siren
  (California Code of Regulations, title 13, section 1028, table I); the octave band its
  level on the axis is highest in, which the rule asks to be 1 or 2 kHz; and how far the
  wail's level falls within its cycle, which the rule allows down to 10 dB(A) under the
  table's (the level in 125 ms, A-weighted per octave band).
- **The Swiss rule.** The two tones against the rule for priority vehicles' two-tone
  horns (Verordnung über die technischen Anforderungen an Strassenfahrzeuge, SR 741.41,
  annex 11, the values of DIN 14610): their base frequencies, the ratio between them and
  the cycle; the level on the axis 2 m away in a free field, which the rule asks to be
  116 to 129 dB(A) on the test bench; and the level 7.00 m in front of the police car,
  0.5, 1.0 and 1.5 m up, its siren 1.5 m up as in the street example, over the street's
  asphalt, as the engine gives it, which the rule asks to be 100 to 115 dB(A) on the
  vehicle.
- **The car's table.** The car's sound power by Table F-1 of CNOSSOS-EU as amended in
  2021 (:mod:`acoustic_spaces.demos.road`) and by the table of Directive (EU) 2015/996,
  which the car of acoustic-spaces 4.0.1 had, from 30 to 130 km/h.

What is not checked: no measured pass-by of a vehicle is compared here. Measured
pass-bys of 75 medium heavy vehicles in France found the 2015 table's levels 3 to 6
dB(A) low (Euronoise 2018, paper 210); the paper gives its measured levels in figures
only.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.demos import road, siren
from acoustic_spaces.directivity import write_directivity_csv
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import A_WEIGHTING_DB, a_weighted

RATE = 48_000
#: California's table I: the angles off the siren's axis (degrees), 3.0 m away, and the
#: least A-weighted level of each class there (dB(A)).
TABLE_I_DEG = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
CLASS_A = (120.0, 119.0, 118.0, 117.0, 115.0, 113.0)
CLASS_B = (115.0, 114.0, 113.0, 112.0, 110.0, 108.0)
TABLE_I_M = 3.0
#: The wail may fall this far under table I within its cycle (dB(A)).
WAIL_DIP_DB = 10.0
#: The Swiss rule's two tones: base frequencies (Hz), the ratio's tolerance about 4:3,
#: the cycle (s), the level on the bench 2 m away and on the vehicle 7.00 m in front
#: (dB(A)), and the heights of the microphone there (m).
SWISS_HZ = (360.0, 630.0)
SWISS_RATIO = (-0.03, 0.07)
SWISS_CYCLE_S = (2.5, 3.5)
SWISS_BENCH = (116.0, 129.0)
SWISS_BENCH_M = 2.0
SWISS_VEHICLE = (100.0, 115.0)
SWISS_VEHICLE_M = 7.0
SWISS_HEIGHTS_M = (0.5, 1.0, 1.5)
#: The siren's height on the police car (m), as in the street example.
SIREN_HEIGHT_M = 1.5
#: Directive (EU) 2015/996's Table F-1, category 1, as acoustic-spaces 4.0.1's car had it:
#: rolling noise A_R and B_R, propulsion noise A_P and B_P, per octave band (dB).
ROLLING_2015 = (
    (79.7, 85.7, 84.5, 90.2, 97.3, 93.9, 84.1, 74.3),
    (30.0, 41.5, 38.9, 25.7, 32.5, 37.2, 39.0, 40.0),
)
PROPULSION_2015 = (
    (94.5, 89.2, 88.0, 85.9, 84.2, 86.9, 83.3, 76.1),
    (-1.3, 7.2, 7.7, 8.0, 8.0, 8.0, 8.0, 8.0),
)
SPEEDS_KMH = (30.0, 50.0, 70.0, 90.0, 110.0, 130.0)


def engine_levels(listeners: list[tuple[str, tuple]], ground: dict, folder: Path) -> dict:
    """The siren 1.5 m up at the origin facing +x, its directivity table written as the
    street's is, heard at ``listeners`` (name, position) over ``ground`` (a material):
    each listener's octave-band level per 0 dB of sound power in each band (dB), as the
    engine gives it."""
    table = write_directivity_csv(
        folder / "siren.csv", siren.directivity(), siren.directivity_comment("check")
    )
    scene = Scene.model_validate(
        {
            "geometry": {"ground": {"extent": [[-20, -20], [20, 20]], "sky": 10}},
            "materials": {"ground": ground},
            "default_material": "ground",
            "sources": [
                {
                    "name": "siren",
                    "position": [0, 0, SIREN_HEIGHT_M],
                    "directivity": {"file": str(table)},
                }
            ],
            "listeners": [{"name": name, "position": list(at)} for name, at in listeners],
            "simulation": {
                "ir_length_s": 0.2,
                "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
                "late": {"rays": 2000},
                "seed": 1,
            },
        }
    )
    results = simulate(Room.from_scene(scene))
    return {
        name: results[("siren", name)].levels(np.zeros(len(A_WEIGHTING_DB))).tolist()
        for name, _ in listeners
    }


def check_california(folder: Path) -> dict:
    """The wail and the yelp against table I, the band they are loudest in, and the
    wail's dip within its cycle."""
    listeners = []
    for angle in TABLE_I_DEG:
        a = math.radians(angle)
        listeners.append(
            (f"{angle:g}", (TABLE_I_M * math.cos(a), TABLE_I_M * math.sin(a), SIREN_HEIGHT_M))
        )
    free = engine_levels(listeners, {"absorption": [1.0] * 8, "scattering": 0.0}, folder)
    out: dict = {}
    print(f"{'degrees off the axis':22}" + "".join(f"{a:>7g}" for a in TABLE_I_DEG))
    print(f"{'class A (least)':22}" + "".join(f"{v:7.0f}" for v in CLASS_A))
    print(f"{'class B (least)':22}" + "".join(f"{v:7.0f}" for v in CLASS_B))
    for pattern in ("wail", "yelp"):
        power = siren.siren_power(pattern, RATE)
        heard = [a_weighted(np.array(free[f"{a:g}"]) + power) for a in TABLE_I_DEG]
        meets = {
            name: sum(h < least for h, least in zip(heard, table, strict=True))
            for name, table in (("A", CLASS_A), ("B", CLASS_B))
        }
        axis = np.array(free["0"]) + power
        loudest = int(np.argmax(axis))
        out[pattern] = {
            "levels_db_a": heard,
            "points_under_class_a": meets["A"],
            "points_under_class_b": meets["B"],
            "loudest_band_hz": OCTAVE_BANDS_HZ[loudest],
        }
        print(f"{pattern + ', dB(A)':22}" + "".join(f"{v:7.1f}" for v in heard))
        print(
            f"  under class A at {meets['A']} of 6 angles, under class B at {meets['B']}; "
            f"loudest on the axis at {OCTAVE_BANDS_HZ[loudest]:g} Hz "
            f"({axis[loudest]:.1f} dB)"
        )
    dip = wail_dip()
    out["wail_dip_db"] = dip
    lowest = out["wail"]["levels_db_a"][0] + dip
    print(
        f"the wail's level in 125 ms falls {-dip:.1f} dB(A) under its average within its "
        f"cycle: {lowest:.1f} dB(A) on the axis at its lowest, the rule's least "
        f"{CLASS_A[0] - WAIL_DIP_DB:g} (class A) and {CLASS_B[0] - WAIL_DIP_DB:g} (class B)"
    )
    return out


def wail_dip() -> float:
    """How far the wail's A-weighted level in 125 ms windows falls under its level over
    a whole cycle (dB, negative), with its octave bands A-weighted."""
    from acoustic_spaces.synthesis import split_bands

    period = 60.0 / siren.WAIL_PER_MINUTE
    x = siren.siren_sound("wail", RATE, period)
    bands = split_bands(x, RATE)
    gains = 10.0 ** (np.array(A_WEIGHTING_DB) / 10.0)
    window = int(0.125 * RATE)
    count = len(x) // window
    energy = (bands[:, : count * window] ** 2).reshape(len(gains), count, window).mean(axis=2)
    weighted = gains @ energy
    return float(10.0 * np.log10(weighted.min() / weighted.mean()))


def check_swiss(folder: Path) -> dict:
    """The two tones against the Swiss rule: frequencies, ratio, cycle, and the level on
    the bench and on the vehicle."""
    low, high = siren.TWO_TONE_HZ
    ratio = high / low / (4.0 / 3.0) - 1.0
    print(
        f"base frequencies {low:g} and {high:.1f} Hz (rule {SWISS_HZ[0]:g} to "
        f"{SWISS_HZ[1]:g}), ratio 4:3 {100 * ratio:+.1f} % (rule {100 * SWISS_RATIO[0]:+g} "
        f"to {100 * SWISS_RATIO[1]:+g} %), cycle {siren.TWO_TONE_PERIOD_S:g} s (rule "
        f"{SWISS_CYCLE_S[0]:g} to {SWISS_CYCLE_S[1]:g}), no pause"
    )
    power = siren.siren_power("two_tone", RATE)
    listeners = [("bench", (SWISS_BENCH_M, 0.0, SIREN_HEIGHT_M))] + [
        (f"{h:g} m", (SWISS_VEHICLE_M, 0.0, h)) for h in SWISS_HEIGHTS_M
    ]
    free = engine_levels(listeners[:1], {"absorption": [1.0] * 8, "scattering": 0.0}, folder)
    bench = a_weighted(np.array(free["bench"]) + power)
    print(
        f"on the bench, {SWISS_BENCH_M:g} m away on the axis in a free field: {bench:.1f} "
        f"dB(A) (rule {SWISS_BENCH[0]:g} to {SWISS_BENCH[1]:g})"
    )
    asphalt = {"preset": "concrete_floor", "scattering": 0.05}
    road_levels = engine_levels(listeners[1:], asphalt, folder)
    vehicle = {}
    for name, _ in listeners[1:]:
        vehicle[name] = a_weighted(np.array(road_levels[name]) + power)
        print(
            f"on the vehicle, {SWISS_VEHICLE_M:g} m in front, {name} up, over asphalt: "
            f"{vehicle[name]:.1f} dB(A) (rule {SWISS_VEHICLE[0]:g} to {SWISS_VEHICLE[1]:g})"
        )
    return {"ratio_off": ratio, "bench_db_a": bench, "vehicle_db_a": vehicle}


def power_2015(speed_kmh: float) -> np.ndarray:
    """The car's sound power per octave band (dB) by 2015/996's table at 20 °C."""
    v = max(speed_kmh, road.MIN_SPEED_KMH)
    rolling = np.array(ROLLING_2015[0]) + np.array(ROLLING_2015[1]) * math.log10(v / 70.0)
    propulsion = np.array(PROPULSION_2015[0]) + np.array(PROPULSION_2015[1]) * (v - 70.0) / 70.0
    return 10.0 * np.log10(10.0 ** (rolling / 10.0) + 10.0 ** (propulsion / 10.0))


def check_car() -> dict:
    """The car's sound power by the 2021 table and by 2015/996's, from 30 to 130 km/h."""
    out = {}
    print(f"{'km/h':>6} {'2021 dB':>8} {'dB(A)':>6} {'2015 dB':>8} {'dB(A)':>6} {'more':>6}")
    for v in SPEEDS_KMH:
        new = road.vehicle_power("1", v)
        old = power_2015(v)
        old_db = float(10.0 * np.log10(np.sum(10.0 ** (old / 10.0))))
        old_a = a_weighted(old)
        out[f"{v:g}"] = {
            "2021_db": new.sound_power_db,
            "2021_db_a": new.sound_power_a_db,
            "2015_db": old_db,
            "2015_db_a": old_a,
        }
        print(
            f"{v:6g} {new.sound_power_db:8.1f} {new.sound_power_a_db:6.1f} {old_db:8.1f} "
            f"{old_a:6.1f} {new.sound_power_a_db - old_a:+6.1f}"
        )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        print("The siren, 3.0 m away, against California's table I (13 CCR 1028)")
        california = check_california(folder)
        print("\nThe two tones against the Swiss rule (SR 741.41, annex 11)")
        swiss = check_swiss(folder)
    print("\nThe car's sound power, CNOSSOS-EU's Table F-1 of 2021 and of 2015/996, at 20 °C")
    car = check_car()
    print(f"\n{time.perf_counter() - started:.0f} s")
    if args.json:
        args.json.write_text(
            json.dumps({"california": california, "swiss": swiss, "car": car}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
