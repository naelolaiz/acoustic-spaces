"""Check the street's road works (:mod:`acoustic_spaces.demos.plant`) against the data
and rules they come from, and against the EU's limits for outdoor machines.

    python validation/plant_sources.py [--json results.json]

- **Defra's rows.** Each machine's octave levels 10 m away, A-weighted and added,
  against the row's own A-weighted level (Defra rounds both to whole decibels).
- **Back 10 m away, in the engine.** Each machine at its height over a hard ground (a
  ground that takes no sound), its reflection added by its energy, heard 10 m away 1.5 m
  up: the level the engine gives against the row's. The sound power is the row's level
  plus 28 dB, which a point over hard ground radiating into a half sphere gives back.
- **The EU's limits.** Each machine's A-weighted sound power against the most Stage II
  of Directive 2000/14/EC (Article 12) lets a new machine of its kind have, where the
  directive sets one: the generator (its electric power taken as 6 kW, its 6 kVA) and
  the dumper (a wheeled one, 75 kW). The directive measures a machine in a test cycle
  of its own, not at work on a site, and Defra's dumper is the loudest moment of a
  drive-by; the excavator's limit, 103 dB(A) for its 121 kW, is for the excavator
  without its breaker. Epiroc gives its MB 1650, a 1650 kg breaker as Defra's, a
  guaranteed sound power of 121 dB(A) (by 2000/14/EC, measured to EN ISO 3744), beside
  the 118 dB(A) of Defra's row.
- **The alarm on its axis.** The reversing alarm 1.2 m up facing +x in a free field (a
  ground that takes all the sound), heard 1.2 m in front of it on its axis and 1.2 m
  behind it: the A-weighted level while it sounds, against SAE J994 Type B's 107 dB(A)
  in front, and as much under that behind it as the siren's directivity has it (11 dB).

What is not checked: no measured site is compared here. Defra's levels are averages of
the machines it measured, "indications" of what such a machine sends; the made sounds'
time structure (the blows, the firing, the beeps' rate) is assumed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.demos import plant, siren
from acoustic_spaces.directivity import write_directivity_csv
from acoustic_spaces.parameters import a_weighted

RATE = 48_000
#: The microphone's height over the ground 10 m from a machine (m; Defra gives none).
MIC_M = 1.5
#: The kinds of the EU's limits and the power each machine is taken at (kW).
LIMITS = {"generator": ("generator", 6.0), "dumper": ("dumper", 75.0)}
#: Epiroc's guaranteed sound power for its MB 1650 (dB(A)).
MB_1650_DB_A = 121.0
#: How much quieter the siren's directivity is behind it than on its axis (dB).
BEHIND_DB = 11.0
HARD = {"absorption": [0.0] * 8, "scattering": 0.0}
FREE = {"absorption": [1.0] * 8, "scattering": 0.0}


def heard(source: dict, listeners: list[tuple[str, tuple]], ground: dict) -> dict[str, float]:
    """The A-weighted level at each of ``listeners`` (name, position) of ``source`` (a
    scene source) over ``ground`` (a material), the reflection added by its energy, as
    the engine gives it (dB(A))."""
    scene = Scene.model_validate(
        {
            "geometry": {"ground": {"extent": [[-30, -30], [30, 30]], "sky": 10}},
            "materials": {"ground": ground},
            "default_material": "ground",
            "sources": [source],
            "listeners": [{"name": name, "position": list(at)} for name, at in listeners],
            "simulation": {
                "sample_rate": RATE,
                "ir_length_s": 0.2,
                "early": {
                    "method": "image_sources",
                    "max_order": 1,
                    "diffraction": "none",
                    "coherent_max_band_hz": None,
                },
                "late": {"rays": 2000},
                "seed": 1,
            },
        }
    )
    results = simulate(Room.from_scene(scene))
    level = np.asarray(source["level_db"], dtype=float)
    return {
        name: float(a_weighted(results[(source["name"], name)].levels(level)))
        for name, _ in listeners
    }


def check_rows() -> dict:
    """Each row's A-weighted sum against its own level, and the machine back 10 m away
    over a hard ground in the engine."""
    out = {}
    print(f"{'machine':<10} {'row':>6} {'A-sum':>6} {'engine':>7} {'L_WA':>6}")
    for machine in plant.MACHINES.values():
        source = {
            "name": machine.name,
            "position": [0.0, 0.0, machine.height_m],
            "level_db": [float(v) for v in machine.sound_power()],
        }
        engine = heard(source, [("10 m", (plant.DEFRA_AT_M, 0.0, MIC_M))], HARD)["10 m"]
        out[machine.name] = {
            "row": f"table {machine.table}, row {machine.row}",
            "row_db_a": machine.level_10m_db_a,
            "a_sum_db_a": round(plant.defra_a_sum(machine), 2),
            "engine_10m_db_a": round(engine, 2),
            "sound_power_a_db": round(machine.sound_power_a(), 2),
        }
        v = out[machine.name]
        print(
            f"{machine.name:<10} {v['row_db_a']:6.1f} {v['a_sum_db_a']:6.1f} "
            f"{v['engine_10m_db_a']:7.1f} {v['sound_power_a_db']:6.1f}"
        )
    return out


def check_limits() -> dict:
    """The machines' A-weighted sound power against Stage II of 2000/14/EC."""
    out = {}
    for name, (kind, power) in LIMITS.items():
        machine = plant.MACHINES[name]
        limit = plant.eu_limit_db_a(kind, power)
        out[name] = {
            "sound_power_a_db": round(machine.sound_power_a(), 2),
            "limit_db_a": limit,
            "over_db": round(machine.sound_power_a() - limit, 2),
        }
        print(f"{name}: {machine.sound_power_a():.1f} dB(A), limit {limit} dB(A)")
    breaker = plant.BREAKER.sound_power_a()
    out["breaker"] = {
        "sound_power_a_db": round(breaker, 2),
        "mb_1650_guaranteed_db_a": MB_1650_DB_A,
        "excavator_limit_db_a": plant.eu_limit_db_a("excavator", plant.BREAKER.power_kw),
    }
    print(f"breaker: {breaker:.1f} dB(A), Epiroc's MB 1650 guaranteed {MB_1650_DB_A:g} dB(A)")
    return out


def check_alarm(folder: Path) -> dict:
    """The alarm in a free field, 1.2 m in front of it on its axis and 1.2 m behind."""
    table = write_directivity_csv(
        folder / "siren.csv", siren.directivity(), siren.directivity_comment("check")
    )
    z = plant.ALARM_HEIGHT_M
    source = {
        "name": plant.ALARM,
        "position": [0.0, 0.0, z],
        "level_db": [float(v) for v in plant.alarm_beep_power(RATE)],
        "directivity": {"file": str(table)},
    }
    at = plant.ALARM_AT_M
    levels = heard(source, [("front", (at, 0.0, z)), ("behind", (-at, 0.0, z))], FREE)
    out = {
        "front_db_a": round(levels["front"], 2),
        "behind_db_a": round(levels["behind"], 2),
        "j994_type_b_db_a": plant.ALARM_ON_AXIS_DB_A,
        "duty": round(plant.alarm_duty(), 4),
        "sound_power_a_db": round(float(a_weighted(plant.alarm_power(RATE))), 2),
        "beep_sound_power_a_db": round(float(a_weighted(plant.alarm_beep_power(RATE))), 2),
    }
    print(
        f"alarm while it sounds: {levels['front']:.1f} dB(A) 1.2 m in front (J994 Type B "
        f"{plant.ALARM_ON_AXIS_DB_A:g}), {levels['behind']:.1f} behind"
    )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    print("Defra's rows: level 10 m away (dB(A)), the octaves' A-weighted sum, the engine's")
    rows = check_rows()
    print("\nThe EU's limits (Directive 2000/14/EC, Stage II)")
    limits = check_limits()
    print()
    with tempfile.TemporaryDirectory() as tmp:
        alarm = check_alarm(Path(tmp))
    print(f"\n{time.perf_counter() - started:.0f} s")
    if args.json:
        data = {"rows": rows, "limits": limits, "alarm": alarm}
        args.json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
