"""The open ground's 1 and 2 kHz bands in wind and temperature gradients, with each
choice of ``simulation.early.open_ground_frequencies``, against a fast field program.

    python validation/open_ground_frequencies.py [--case NAME] [--choice 24|8|rays]
        [--cache DIR] [--part K/N] [--json results.json]

Over the open ground in the weather the engine takes its 1 and 2 kHz octave bands from
the parabolic equation at 24 frequencies an octave (``24``, the default), at the early
paths' own 8 (``8``), or from the rays where one reaches the listener short of half the
way to the shadow's start (``rays``). Each case is a scene (:func:`scene_of`): a source
over grass (Delany and Bazley, 200 kPa s/m²) or a rigid ground, in a weather preset that
bends sound, the listeners 1.5 and 4 m up along the wind or against it; the engine's
level is that of its early paths (:func:`~acoustic_spaces.engine.coherent.total_energy`)
re free field, so the late part, which the setting does not change, leaves it alone. The
reference is the fast field program (:mod:`validation.ffp`, written apart from the
engine), its top 100 m up (with 200 m the levels at 2 kHz from a source 10 m up 200 m to
1 km away, downwind and under an inversion, are the same to 0.001 dB), the band's energy
averaged over 24 equal slices; the cases of ``DENSER_CASES`` also over 48, the
reference's own sampling. Points deeper than ``FLOOR_DB`` below free field in the
program are not compared. The upward cases stop at ``UPWARD_RANGES``' last, 500 m, where
those bands lie 37 to 102 dB below free field (upwind, deeper than ``FLOOR_DB``), to
keep the program's time down.

The table: |engine - program| over every compared point of every case at 1 and 2 kHz
(dB, mean, nine in ten, largest), for each choice; per case the largest; and the
largest change of the 4 and 8 kHz bands (``ABOVE``) with 8 or ``rays`` against 24's,
over every point and over those within ``FLOOR_DB`` of free field with 24, since in the
shadow they deepen from 1 and 2 kHz. ``--cache``
keeps each case's reference in a directory and takes it from there; with ``--part K/N``
a run finds only the K-th of every N references (K from 0), so that N runs side by
side share the work, and a last run without it reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_1995 import FLOOR_DB  # noqa: E402
from ffp import pressure  # noqa: E402

from acoustic_spaces import Room, Scene, simulate  # noqa: E402
from acoustic_spaces.engine.coherent import total_energy  # noqa: E402
from acoustic_spaces.engine.diffraction import band_frequencies  # noqa: E402
from acoustic_spaces.materials.ground import surface_impedance  # noqa: E402
from acoustic_spaces.scene import Atmosphere  # noqa: E402

#: The bands compared (their index among the octave bands) and the choices.
BANDS = {1000: 4, 2000: 5}
#: The bands above, which the choice moves only where the shadow deepens them from
#: the 1 and 2 kHz levels: reported as their change against 24's.
ABOVE = {4000: 6, 8000: 7}
CHOICES = (24, 8, "rays")
#: The listeners' heights (m), the ranges along the ground (m) downwind and under an
#: inversion, and in the upward cases (wind against the sound, a sunny day).
HEIGHTS = (1.5, 4.0)
RANGES = (50.0, 100.0, 200.0, 500.0, 1000.0)
UPWARD_RANGES = (50.0, 100.0, 200.0, 500.0)
#: The grounds: grass's flow resistivity (kPa s/m²), or None for a rigid ground.
GROUNDS = {"grass": 200.0, "rigid": None}
GROUND_C = 20.0
#: The program's top (m) and its slices of a band.
TOP_M = 100.0
SLICES = 24
#: The weathers: preset, along the wind (1) or against it (-1).
WEATHERS = {
    "windy downwind": ("windy", 1.0),
    "windy upwind": ("windy", -1.0),
    "sunny_day": ("sunny_day", 1.0),
    "clear_night": ("clear_night", 1.0),
}
#: The cases: weather, ground, source height (m); every weather over grass with sources
#: 0.5, 2 and 5 m up and over a rigid ground with one 2 m up, and a source 10 m up over
#: grass downwind and under an inversion.
CASES = {
    f"{weather}, {ground}, {zs:g} m": (weather, ground, zs)
    for weather in WEATHERS
    for ground, heights in (("grass", (0.5, 2.0, 5.0)), ("rigid", (2.0,)))
    for zs in heights
}
CASES.update(
    {
        f"{weather}, grass, 10 m": (weather, "grass", 10.0)
        for weather in ("windy downwind", "clear_night")
    }
)
#: The cases whose reference is also found over 48 slices.
DENSER_CASES = ("windy downwind, grass, 10 m",)
CACHE: dict = {"dir": None, "part": None}


def points_of(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(n,) the ranges (m) and the heights (m) of a case's listeners."""
    weather, _ground, _zs = CASES[name]
    ranges = UPWARD_RANGES if WEATHERS[weather][1] < 0 or weather == "sunny_day" else RANGES
    r = np.array([x for _ in HEIGHTS for x in ranges])
    z = np.array([h for h in HEIGHTS for _ in ranges])
    return r, z


def speed_of(weather: str):
    """The effective speed of sound (m/s) at heights z in ``weather``, the way the
    sound goes."""
    preset, along = WEATHERS[weather]
    atmosphere = Atmosphere(preset=preset, wind={"direction_deg": 0.0})
    towards = np.array([along, 0.0])

    def speed(z):
        heights = np.atleast_1d(np.asarray(z, dtype=float))
        return np.asarray(atmosphere.sound_speed(heights, GROUND_C, towards), float)

    return speed


def reference(name: str, slices: int = SLICES) -> dict[int, list[float]]:
    """The program's level re free field (dB) at a case's listeners, per band: its
    energy averaged over ``slices`` equal slices of the band."""
    key = f"{name} {slices}".replace(" ", "_").replace(",", "")
    path = None if CACHE["dir"] is None else Path(CACHE["dir"]) / f"{key}.json"
    if path is not None and path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    weather, ground, zs = CASES[name]
    speed = speed_of(weather)
    c0 = float(speed(0.0)[0])
    r, z = points_of(name)
    direct2 = r**2 + (zs - z) ** 2
    out = {}
    for centre, band in BANDS.items():
        energy = np.zeros(len(r))
        for f in band_frequencies(slices)[0][band]:
            sigma = GROUNDS[ground]
            impedance = None
            if sigma is not None:
                impedance = complex(
                    np.conj(
                        surface_impedance(np.asarray(f), sigma, "delany_bazley", speed_of_sound=c0)
                    )
                )
            p = pressure(float(f), speed, zs, r, z, impedance, top=TOP_M)
            energy += np.abs(p) ** 2 * direct2 / slices
        out[centre] = (10.0 * np.log10(energy)).tolist()
    if path is not None:
        path.write_text(json.dumps(out))
    return out


def scene_of(name: str, choice) -> Scene:
    """A case as a scene: the source at the origin, the listeners along x, the wind
    along x (or against it), the early paths with ``choice``."""
    weather, ground, zs = CASES[name]
    preset, along = WEATHERS[weather]
    r, z = points_of(name)
    sigma = GROUNDS[ground]
    material = (
        {"absorption": [0.0] * 8, "scattering": 0.0}
        if sigma is None
        else {
            "impedance": {"model": "delany_bazley", "flow_resistivity": sigma},
            "scattering": 0.0,
        }
    )
    return Scene.model_validate(
        {
            "name": name,
            "geometry": {
                "ground": {"extent": [[-50.0, -50.0], [r.max() + 50.0, 50.0]], "sky": 150.0}
            },
            "materials": {"field": material},
            "default_material": "field",
            "sources": [{"name": "s", "position": [0.0, 0.0, zs]}],
            "listeners": [
                {"name": f"l{k}", "position": [float(x), 0.0, float(h)]}
                for k, (x, h) in enumerate(zip(r, z, strict=True))
            ],
            "air": {"temperature_c": GROUND_C, "humidity_percent": 50.0},
            "atmosphere": {
                "preset": preset,
                "wind": {"direction_deg": 0.0 if along > 0 else 180.0},
            },
            "simulation": {
                "seed": 1,
                "early": {
                    "method": "image_sources",
                    "max_order": 1,
                    "diffraction": "none",
                    "open_ground_frequencies": choice,
                },
                "late": {"rays": 100},
                "ir_length_s": 3.2,
                "performance": {"threads": 1},
            },
        }
    )


def engine(name: str, choice) -> dict:
    """The engine's early level re free field (dB) at a case's listeners, per band, and
    the processor time (s) it took."""
    scene = scene_of(name, choice)
    started = time.process_time()
    result = simulate(Room.from_scene(scene), threads=1)
    seconds = time.process_time() - started
    out: dict = {"seconds": seconds}
    levels = []
    for listener in scene.listeners:
        response = result["s", listener.name]
        free = np.asarray(response.free_field, dtype=float)
        levels.append(10.0 * np.log10(total_energy(response.early) / free))
    levels_array = np.array(levels)
    for centre, band in {**BANDS, **ABOVE}.items():
        out[centre] = levels_array[:, band].tolist()
    return out


def summary(misses: list[float]) -> str:
    a = np.abs(np.asarray(misses))
    return f"{a.mean():.2f}, {np.percentile(a, 90):.2f}, {a.max():.2f}"


def check(names, choices) -> dict:
    """Every case's reference and engine levels; the table."""
    out: dict = {}
    for name in names:
        if CACHE["part"] is not None:
            k, n = CACHE["part"]
            if list(CASES).index(name) % n != k:
                continue
        started = time.time()
        ffp = reference(name)
        if name in DENSER_CASES:
            denser = reference(name, 2 * SLICES)
            out.setdefault("denser", {})[name] = {
                c: (np.array(ffp[c]) - np.array(denser[c])).tolist() for c in BANDS
            }
        if CACHE["part"] is not None:
            print(f"{name}: reference ({time.time() - started:.0f} s)", flush=True)
            continue
        row = {"ffp": ffp}
        for choice in choices:
            row[str(choice)] = engine(name, choice)
        out[name] = row
        line = f"{name:34s}"
        for choice in choices:
            worst = 0.0
            for centre in BANDS:
                miss = np.array(row[str(choice)][centre]) - np.array(ffp[centre])
                kept = np.array(ffp[centre]) >= FLOOR_DB
                if kept.any():
                    worst = max(worst, float(np.abs(miss[kept]).max()))
            line += f"  {choice}: {worst:5.2f}"
        print(line + f"  ({time.time() - started:.0f} s)", flush=True)
    if CACHE["part"] is not None:
        return out
    print("\n|engine - program| at 1 and 2 kHz (dB: mean, nine in ten, largest) over the")
    print("points compared, and the engine's processor time (s)")
    table = {}
    for choice in choices:
        misses = []
        seconds = 0.0
        for name in out:
            if name == "denser":
                continue
            row = out[name]
            seconds += row[str(choice)]["seconds"]
            for centre in BANDS:
                ffp = np.array(row["ffp"][centre])
                miss = np.array(row[str(choice)][centre]) - ffp
                misses.extend(miss[ffp >= FLOOR_DB].tolist())
        table[str(choice)] = {"misses": len(misses), "seconds": seconds}
        print(f"  {choice!s:5s} {summary(misses)}   {len(misses)} points   {seconds:.0f} s")
        table[str(choice)]["summary"] = summary(misses)
    out["table"] = table
    for choice in choices:
        if choice == 24 or 24 not in choices:
            continue
        for floor, where in ((-np.inf, "every point"), (FLOOR_DB, "within FLOOR_DB")):
            worst = (0.0, "", 0.0)
            for name in out:
                if name not in CASES:
                    continue
                for centre in ABOVE:
                    base = np.array(out[name]["24"][centre])
                    shift = np.abs(np.array(out[name][str(choice)][centre]) - base)
                    shift[base < floor] = 0.0
                    k = int(shift.argmax())
                    if shift[k] > worst[0]:
                        worst = (float(shift[k]), f"{name}, {centre} Hz", float(base[k]))
            print(
                f"  4 and 8 kHz with {choice} less with 24, {where}: up to {worst[0]:.2f} dB"
                f" ({worst[1]}, at {worst[2]:.0f} dB with 24)"
            )
    for name, shift in out.get("denser", {}).items():
        print(
            f"  the reference over 24 less over 48 slices, {name}: "
            + ", ".join(f"{c} Hz up to {np.abs(shift[c]).max():.2f}" for c in BANDS)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", help="these cases only (repeatable)")
    parser.add_argument(
        "--choice", action="append", help="these choices only (24, 8 or rays; repeatable)"
    )
    parser.add_argument("--cache", help="keep each case's reference in this directory")
    parser.add_argument(
        "--part",
        help="with --cache: find only the K-th of every N references (K/N, K from 0)",
    )
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)
        CACHE["dir"] = args.cache
    if args.part:
        if not args.cache:
            parser.error("--part needs --cache")
        k, n = (int(v) for v in args.part.split("/"))
        CACHE["part"] = (k, n)
    names = [n for n in CASES if not args.case or n in args.case]
    choices = [c if c == "rays" else int(c) for c in (args.choice or CHOICES)]
    results = check(names, choices)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
