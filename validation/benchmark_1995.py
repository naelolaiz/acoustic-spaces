"""The 1995 benchmark cases for outdoor sound: the engine against a fast field program.

    python validation/benchmark_1995.py [--quick] [--far] [--json results.json]

Attenborough, Taherzadeh, Bass, Di, Raspet, Becker, Güdesen, Chrestman, Daigle,
L'Espérance, Gabillet, Gilbert, Li, White, Naz, Noble and van Hoof, "Benchmark cases
for outdoor sound propagation models", J. Acoust. Soc. Am. 97 (1995) 173-191, set four
cases for comparing models of outdoor propagation: a source 5 m and a receiver 1 m over
a ground of flow resistivity 200 kPa s/m² (Delany and Bazley's impedance), at 10, 100
and 1000 Hz, up to 10 km away, in still air (case 1), in air whose sound speed rises
0.1 m/s with every metre of height (case 2: downward refraction, as downwind or under
an inversion), falls as much (case 3: upward refraction, upwind or over sun-warmed
ground), and case 3 with turbulence (case 4). The published reference solutions came
from fast field programs. This script solves cases 1 to 3 with this package's own
(:mod:`validation.ffp`, written apart from the engine and checked against the exact
solution for the ground) and compares the engine with it:

- the parabolic equation that gives the engine's sound over the open ground in the
  bands up to 500 Hz, and in a shadow up to 2 kHz (:mod:`acoustic_spaces.engine.parabolic`),
  at the benchmark's pure tones, 100 m to 10 km;
- whole outdoor scenes, the level relative to free field per octave band from 63 Hz to
  1 kHz, 100 m to 2 km (``--far``: to 10 km, which takes long in the shadow of case 3),
  each case a scene with the gradient as a measured atmosphere, a wind along the line
  of the listeners, the air at 19.6 °C so that the speed of sound at the ground is
  343 m/s, and the engine's default settings;
- the same with the gradient of 0.1/s stopping at 100 m, 10 m/s more there than at the
  ground, as a strong wind blows: the weather the engine is meant for.

The benchmark's profiles go on with the height without end; the engine's atmosphere
stops changing 1000 m up, and the fast field program follows it there: above, the air
is homogeneous and nothing comes back down. Case 4 is not run: the benchmark's
turbulence has a Gaussian spectrum, the engine's eddies a von Kármán one, checked in
:mod:`validation.turbulence`. Levels are in dB re free field at the same distance. The
pure tones carry no air absorption (the parabolic equation has none: the engine takes
the absorption off the straight distance afterwards, which the level relative to free
field cancels); the whole scenes take the air's (ISO 9613-1 at 19.6 °C and 50 %),
which the engine's rays carry along their own paths and the fast field program is
run without, a difference of the path lengths' only. In a shadow deeper than
``FLOOR_DB`` neither is compared: the fast field program's layers scatter a little
sound there (see :mod:`validation.ffp`), and nothing is heard that deep anyway.
``--quick`` stops at 1 km and leaves 1 kHz out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffp import pressure  # noqa: E402

from acoustic_spaces import Room, Scene  # noqa: E402
from acoustic_spaces.air import SOUND_SPEED_0C  # noqa: E402
from acoustic_spaces.engine import insertion  # noqa: E402
from acoustic_spaces.engine.diffraction import band_frequencies  # noqa: E402
from acoustic_spaces.engine.parabolic import march  # noqa: E402
from acoustic_spaces.engine.refracted import (  # noqa: E402
    WAVE_MIN_HEADROOM_M,
    WAVE_TOP_SLOPE,
)
from acoustic_spaces.engine.refraction import _TOP_M  # noqa: E402
from acoustic_spaces.materials import OCTAVE_BANDS_HZ  # noqa: E402
from acoustic_spaces.materials.ground import surface_impedance  # noqa: E402
from acoustic_spaces.parameters import FILTER_ORDER, OCTAVE_RATIO, octave_midbands  # noqa: E402

#: Source and receiver heights (m), the ground's flow resistivity (kPa s/m²).
ZS, ZR = 5.0, 1.0
SIGMA = 200.0
#: The speed of sound at the ground (m/s) and the air temperature that gives it.
C0 = 343.0
TEMPERATURE_C = 273.15 * ((C0 / SOUND_SPEED_0C) ** 2 - 1.0)
#: The cases: name, the sound speed's gradient (1/s) and the height it stops at (m).
CASES = {
    "case 1, still air": (0.0, _TOP_M),
    "case 2, downward refraction 0.1/s": (0.1, _TOP_M),
    "case 3, upward refraction 0.1/s": (-0.1, _TOP_M),
    "0.1/s up to 100 m": (0.1, 100.0),
    "-0.1/s up to 100 m": (-0.1, 100.0),
}
TONES = (10.0, 100.0, 1000.0)
RANGES = (100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
#: The octave bands the whole scenes are compared in, and the slices of the 1 kHz band
#: the fast field program is averaged over (the engine averages the interference of
#: its rays over the band exactly; the parabolic equation's bands take its own 8).
BANDS = (63, 125, 250, 500, 1000)
RAY_BAND_SLICES = 24
#: The engine reads the bands up to this one through its band filters, which let in a
#: little of the neighbouring bands; the program is read through the same filters there
#: (as in :mod:`validation.insertion_loss`), from the band below 63 Hz to 1 kHz, past
#: which those filters let nothing through that counts.
FILTERED_MAX_BAND_HZ = 500
#: Below this level re free field (a deep shadow) nothing is compared.
FLOOR_DB = -60.0


def speed_of(gradient: float, cap: float):
    """The effective speed of sound (m/s) at heights z: linear up to ``cap``."""
    return lambda z: C0 + gradient * np.minimum(np.asarray(z, dtype=float), cap)


def impedance_of(frequency, c: float = C0) -> np.ndarray:
    """The ground's normalised impedance, time as exp(-i omega t)."""
    return np.conj(
        surface_impedance(
            np.asarray(frequency, dtype=float), SIGMA, "delany_bazley", speed_of_sound=c
        )
    )


def free_field_db(p: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """|p| as a level (dB) over the free field's at the direct distance."""
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.abs(p) * np.hypot(ranges, ZS - ZR))


def check_tones(cases: dict, tones=TONES, ranges=RANGES) -> dict:
    """The engine's parabolic equation against the fast field program, pure tones."""
    ranges = np.asarray(ranges, dtype=float)
    top = max(ZS + WAVE_MIN_HEADROOM_M, WAVE_TOP_SLOPE * ranges.max())
    print("The parabolic equation against the fast field program: pure tones, the")
    print("level re free field (dB) and the parabolic equation's difference from it\n")
    print(f"{'':36}" + "".join(f"{f'{r:g} m':>9}" for r in ranges))
    out = {}
    start = time.time()
    for name, (gradient, cap) in cases.items():
        speed = speed_of(gradient, cap)
        for f in tones:
            z = complex(impedance_of(f))
            reference = free_field_db(pressure(f, speed, ZS, ranges, ZR, z), ranges)
            wave = march([f], speed, lambda r, z=z: z, ZS, ranges, np.full(len(ranges), ZR), top)[0]
            engine = free_field_db(wave, ranges)
            miss = np.where(reference >= FLOOR_DB, engine - reference, np.nan)
            out[f"{name}, {f:g} Hz"] = {
                "ffp_db": reference.tolist(),
                "pe_minus_ffp_db": miss.tolist(),
            }
            label = f"{name}, {f:g} Hz"
            print(f"{label:36}" + "".join(f"{v:9.2f}" for v in reference) + "  ffp")
            print(
                f"{'':36}"
                + "".join("        -" if np.isnan(v) else f"{v:+9.2f}" for v in miss)
                + "  pe - ffp"
            )
    print(f"\n(-: below {FLOOR_DB:g} dB; {time.time() - start:.0f} s)\n")
    return out


def scene_of(gradient: float, cap: float, ranges) -> Scene:
    """A benchmark case as an outdoor scene: the gradient as a wind along the line of
    the listeners (towards them where it is positive)."""
    ranges = [float(r) for r in ranges]
    rows = [[0.0, TEMPERATURE_C, 0.0], [cap, TEMPERATURE_C, abs(gradient) * cap]]
    if cap < _TOP_M:
        rows.append([_TOP_M, TEMPERATURE_C, abs(gradient) * cap])
    data = {
        "name": "benchmark",
        "geometry": {
            "ground": {"extent": [[-50.0, -50.0], [max(ranges) + 50.0, 50.0]], "sky": 50.0}
        },
        "materials": {
            "field": {
                "impedance": {"model": "delany_bazley", "flow_resistivity": SIGMA},
                "scattering": 0.0,
            }
        },
        "default_material": "field",
        "sources": [{"name": "s", "position": [0.0, 0.0, ZS]}],
        "listeners": [{"name": f"{r:g} m", "position": [r, 0.0, ZR]} for r in ranges],
        "air": {"temperature_c": TEMPERATURE_C, "humidity_percent": 50.0},
        "simulation": {
            "ir_length_s": 0.5 if max(ranges) <= 2000.0 else 4.0,
            "seed": 1,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
            "late": {"rays": 200},
        },
    }
    if gradient != 0.0:
        data["atmosphere"] = {
            "table": rows,
            "wind": {"direction_deg": 0.0 if gradient > 0.0 else 180.0},
        }
    return Scene.model_validate(data)


def _filter_gains(frequencies: np.ndarray) -> np.ndarray:
    """Squared gains (octave bands, n) of the engine's Butterworth band filters at these
    frequencies (:func:`acoustic_spaces.parameters.octave_filter_gains`)."""
    fm = octave_midbands()[:, None]
    half = OCTAVE_RATIO**0.5
    width = half - 1.0 / half
    return 1.0 / (1.0 + ((frequencies / fm - fm / frequencies) / width) ** (2 * FILTER_ORDER))


def _frequency_plan(bands) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    """The frequencies the program runs at for whole scenes, the width of the slice
    each stands for, and where each octave band's own slices are: the band below 63 Hz
    and the bands up to 1 kHz, which the filters of the bands up to 500 Hz let in a
    little of, and 1 kHz in finer slices for the rays' band average when it is compared."""
    slices, widths = band_frequencies(bands="octave")
    many, many_widths = band_frequencies(RAY_BAND_SLICES, bands="octave")
    plan = [(slices[0] / OCTAVE_RATIO, widths[0] / OCTAVE_RATIO)]
    for band in range(list(OCTAVE_BANDS_HZ).index(1000) + 1):
        fine = OCTAVE_BANDS_HZ[band] >= 1000 and 1000 in bands
        plan.append((many[band], many_widths[band]) if fine else (slices[band], widths[band]))
    members, start = [], len(plan[0][0])
    for f, _ in plan[1:]:
        members.append(slice(start, start + len(f)))
        start += len(f)
    frequencies = np.concatenate([f for f, _ in plan])
    slice_widths = np.concatenate([np.full(len(f), w) for f, w in plan])
    return frequencies, slice_widths, members


def check_scenes(cases: dict, ranges, bands=BANDS) -> dict:
    """Whole scenes: the engine's level relative to free field per octave band against
    the fast field program averaged over the band, and read through the engine's band
    filters in the bands it reads so."""
    ranges = np.asarray(ranges, dtype=float)
    columns = [list(OCTAVE_BANDS_HZ).index(b) for b in bands]
    frequencies, slice_widths, members = _frequency_plan(bands)
    gains = _filter_gains(frequencies) * slice_widths  # (octave bands, n)
    print("Whole scenes against the fast field program: the level relative to free field")
    print("per octave band (dB), the engine's difference from it, and from it read through")
    print(f"the engine's band filters in the bands up to {FILTERED_MAX_BAND_HZ} Hz\n")
    out = {}
    start = time.time()
    for name, (gradient, cap) in cases.items():
        speed = speed_of(gradient, cap)
        scene = scene_of(gradient, cap, ranges)
        result = insertion.simulate(Room.from_scene(scene))
        engine = np.array(
            [result[("s", lis.name)].relative_to_free_field()[columns] for lis in scene.listeners]
        )  # (ranges, bands)
        energy = np.array(
            [
                np.abs(pressure(f, speed, ZS, ranges, ZR, complex(impedance_of(f)))) ** 2
                for f in frequencies
            ]
        ) * (ranges**2 + (ZS - ZR) ** 2)  # (n, ranges)
        reference = np.zeros_like(engine)
        filtered = np.zeros_like(engine)
        for j, band in enumerate(columns):
            reference[:, j] = 10.0 * np.log10(energy[members[band]].mean(axis=0))
            if OCTAVE_BANDS_HZ[band] <= FILTERED_MAX_BAND_HZ:
                weights = gains[band][:, None]
                filtered[:, j] = 10.0 * np.log10((weights * energy).sum(axis=0) / weights.sum())
            else:
                filtered[:, j] = reference[:, j]
        heard = reference >= FLOOR_DB
        miss = np.where(heard, engine - reference, np.nan)
        miss_filtered = np.where(heard, engine - filtered, np.nan)
        out[name] = {
            "bands_hz": list(bands),
            "ranges_m": ranges.tolist(),
            "ffp_db": reference.tolist(),
            "engine_minus_ffp_db": miss.tolist(),
            "engine_minus_filtered_ffp_db": miss_filtered.tolist(),
        }
        print(name + f" ({time.time() - start:.0f} s)")
        print(f"{'':12}" + "".join(f"{b:>8}" for b in bands))
        for i, r in enumerate(ranges):
            print(f"{f'{r:g} m':12}" + "".join(f"{v:8.2f}" for v in reference[i]) + "  ffp")
            for row, label in ((miss, "engine - ffp"), (miss_filtered, "through the filters")):
                print(
                    f"{'':12}"
                    + "".join("       -" if np.isnan(v) else f"{v:+8.2f}" for v in row[i])
                    + f"  {label}"
                )
        print()
    print(f"(-: below {FLOOR_DB:g} dB; {time.time() - start:.0f} s)\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="to 1 km, without 1 kHz")
    parser.add_argument("--far", action="store_true", help="whole scenes to 10 km (slow)")
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args(argv)
    ranges = [r for r in RANGES if r <= 1000.0] if args.quick else RANGES
    tones = [f for f in TONES if f < 1000.0] if args.quick else TONES
    near = [r for r in ranges if args.far or r <= 2000.0]
    bands = [b for b in BANDS if b < 1000] if args.quick else BANDS
    results = {
        "tones": check_tones(CASES, tones, ranges),
        "scenes": check_scenes(CASES, near, bands),
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
