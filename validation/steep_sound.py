"""Steep sound over the open ground in wind and temperature gradients: the narrow- and
the wide-angle parabolic equation against a fast field program.

    python validation/steep_sound.py [--set first|further] [--case NAME] [--only WHAT]
        [--cache DIR] [--json results.json]

The engine's runs of the parabolic equation give the listeners that the reflection off
the ground reaches more than ``parabolic.STEEP_DEG`` (15) degrees up the wide-angle
equation's level, and the flatter ones the narrow-angle one's. Each case is a scene
(:func:`scene_of`): a source over a ground in a weather that bends sound, and a line of
listeners at one height along the wind or against it, ``POINTS`` of them from
``refracted.WAVE_MIN_RANGE`` times the heights added (``NEAREST_M`` at least), where the
parabolic equation takes over from the rays, to ``FARTHEST_M``, and two more ``BESIDE``
of the range either side of the one the reflection reaches at 15 degrees. The engine's
level is that of its early paths
(:func:`~acoustic_spaces.engine.coherent.total_energy`) re free field, the 1 and 2 kHz
bands left to the rays short of the shadow, which changes nothing in the bands compared.
The reference is
the fast field program (:mod:`validation.ffp`, written apart from the engine) with the
gradient going on to ``TOP_M`` up, the band's energy averaged over the frequencies the
engine's equation takes in the band
(:func:`~acoustic_spaces.engine.diffraction.band_frequencies`: 8 an octave band, 3 a
third octave), so that the two differ by the equation alone. Points deeper than
``FLOOR_DB`` below free field in the program are not compared.

Each case runs as the engine has it (``engine``), and with the narrow-angle equation at
every point (``parabolic.STEEP_DEG`` set to 90 degrees) at its range step, half and a
quarter of it (``narrow``, ``half``, ``quarter``). The first set (``FIRST``) is the
``windy`` preset down- and upwind over grass; the further set (``FURTHER``) other
weathers, a hard ground and other heights, in the bands of ``FURTHER_OCTAVES`` and
``FURTHER_THIRDS``.

The tables: |level - program| (dB, mean and largest) at the flatter points with the
narrow-angle equation and at the steep ones with each; the same at the flatter points
from ``NEAR_DEG`` to 15 degrees up; and the jump at 15 degrees, the engine's level less
the program's just steeper less the same just flatter, with the five largest and each
side's own distance from the program there. ``--only steps`` runs the range step's own
check instead: one listener (:data:`STEPS_CASE`), the 63 Hz band at the narrow-angle
equation's step, at one wavelength and at a 32nd of the step.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ffp import pressure  # noqa: E402

from acoustic_spaces import Room, Scene, simulate  # noqa: E402
from acoustic_spaces.engine import parabolic, refracted  # noqa: E402
from acoustic_spaces.engine.coherent import total_energy  # noqa: E402
from acoustic_spaces.engine.diffraction import band_frequencies  # noqa: E402
from acoustic_spaces.materials.ground import surface_impedance  # noqa: E402
from acoustic_spaces.scene import Atmosphere  # noqa: E402

#: The listeners of a case: this many, spaced evenly in the logarithm of the range from
#: the nearest the parabolic equation gives (``NEAREST_M`` at least) to ``FARTHEST_M``,
#: and one ``BESIDE`` of the range nearer and farther than where the reflection off the
#: ground comes in at ``parabolic.STEEP_DEG``.
POINTS = 16
NEAREST_M = 10.0
FARTHEST_M = 1000.0
BESIDE = 0.01
#: Points whose level in the program is deeper than this re free field are not compared.
FLOOR_DB = -15.0
#: The flatter points this steep or steeper (degrees) are also reported apart.
NEAR_DEG = 10.0
#: The grounds: flow resistivity (kPa s/m², Delany and Bazley).
GROUNDS = {"grass": 200.0, "hard": 2000.0}
GROUND_C = 20.0
#: The program's top (m): the gradient goes on to there, with the same air above.
TOP_M = 400.0
#: The octave bands (index among the octave bands) and third octaves (index among the
#: third-octave bands) compared in the first set, the third octaves only from the
#: sources ``THIRDS_FROM_M`` up or higher, and those of the further set.
OCTAVES = {63: 0, 125: 1, 250: 2, 500: 3}
THIRDS = {50: 0, 63: 1, 80: 2, 100: 3, 125: 4, 160: 5, 200: 6, 250: 7}
THIRDS_FROM_M = 10.0
FURTHER_OCTAVES = {63: 0, 125: 1}
FURTHER_THIRDS = {80: 2, 100: 3, 125: 4, 160: 5}
#: The weathers: the atmosphere, and whether the sound goes along the wind (1) or
#: against it (-1).
WEATHERS = {
    "windy downwind": ({"preset": "windy"}, 1.0),
    "windy upwind": ({"preset": "windy"}, -1.0),
    "10 m/s downwind": ({"preset": "windy", "wind": {"speed_m_s": 10.0}}, 1.0),
    "10 m/s upwind": ({"preset": "windy", "wind": {"speed_m_s": 10.0}}, -1.0),
    "clear_night": ({"preset": "clear_night"}, 1.0),
    "sunny_day, 3 m/s downwind": ({"preset": "sunny_day", "wind": {"speed_m_s": 3.0}}, 1.0),
}
#: The first set: weather, ground, source and listener heights (m).
FIRST = {
    f"{weather}, grass, {zs:g} m to {zr:g} m": (weather, "grass", zs, zr)
    for weather in ("windy downwind", "windy upwind")
    for zs, zr in ((5.0, 1.5), (10.0, 1.5), (30.0, 1.5), (120.0, 1.5), (1.5, 30.0))
}
#: The further set: every weather with three pairs of a ground and heights.
FURTHER = {
    f"{weather}, {ground}, {zs:g} m to {zr:g} m": (weather, ground, zs, zr)
    for weather in WEATHERS
    for ground, zs, zr in (("grass", 20.0, 1.5), ("grass", 40.0, 4.0), ("hard", 60.0, 4.0))
}
CASES = {**FIRST, **FURTHER}
#: The engine's runs of a case: the narrow-angle equation's range step as a share of
#: its own, or None for the engine as it is (the wide-angle equation at steep points).
RUNS = {"engine": None, "narrow": 1.0, "half": 0.5, "quarter": 0.25}
#: The range step's own check: a case's weather, ground and heights, the listener's
#: range (m), the band, and the steps (wavelengths at the highest of the band's
#: frequencies: 86 Hz in the 63 Hz band).
STEPS_CASE = ("windy downwind", "grass", 1.0, 1.5, 200.0, 63)
STEPS = (5.0, 1.0, 5.0 / 32.0)
#: The engine's own limit (degrees), kept while :func:`narrow_angle` moves it.
STEEP_DEG = parabolic.STEEP_DEG
CACHE: dict = {"dir": None}


def steep_from(zs: float, zr: float) -> float:
    """The range (m) within which the reflection off the ground reaches the listener
    more steeply than ``STEEP_DEG``."""
    return (zs + zr) / math.tan(math.radians(STEEP_DEG))


def points_of(name: str) -> np.ndarray:
    """(n,) the ranges (m) of a case's listeners, increasing."""
    _weather, _ground, zs, zr = CASES[name]
    nearest = max(NEAREST_M, 1.001 * refracted.WAVE_MIN_RANGE * (zs + zr))
    line = np.geomspace(nearest, FARTHEST_M, POINTS)
    edge = steep_from(zs, zr)
    return np.sort(np.concatenate([line, [edge * (1.0 - BESIDE), edge * (1.0 + BESIDE)]]))


def atmosphere_of(weather: str) -> dict:
    """The scene's atmosphere for ``weather``: the wind along x, or against it."""
    spec, along = WEATHERS[weather]
    wind = {**spec.get("wind", {}), "direction_deg": 0.0 if along > 0 else 180.0}
    return {**spec, "wind": wind}


def speed_of(weather: str):
    """The effective speed of sound (m/s) at heights z in ``weather``, the way the
    sound goes (towards x)."""
    atmosphere = Atmosphere(**atmosphere_of(weather))
    towards = np.array([1.0, 0.0])

    def speed(z):
        heights = np.atleast_1d(np.asarray(z, dtype=float))
        return np.asarray(atmosphere.sound_speed(heights, GROUND_C, towards), float)

    return speed


def bands_of(name: str, resolution: str) -> dict[int, int]:
    """The bands compared in a case: centre (Hz) to index."""
    _weather, _ground, zs, _zr = CASES[name]
    if name in FIRST:
        if resolution == "octave":
            return OCTAVES
        return THIRDS if zs >= THIRDS_FROM_M else {}
    return FURTHER_OCTAVES if resolution == "octave" else FURTHER_THIRDS


def program(weather, ground, zs, zr, ranges, resolution, bands, top=TOP_M) -> dict:
    """The program's level re free field (dB) at ``ranges``, per band of ``bands``, at
    the engine's frequencies of the band."""
    speed = speed_of(weather)
    c0 = float(speed(0.0)[0])
    r = np.asarray(ranges, dtype=float)
    direct2 = r**2 + (zs - zr) ** 2
    frequencies = band_frequencies(bands=resolution)[0]
    out = {}
    for centre, band in bands.items():
        energy = np.zeros(len(r))
        for f in frequencies[band]:
            impedance = complex(
                np.conj(
                    surface_impedance(
                        np.asarray(f), GROUNDS[ground], "delany_bazley", speed_of_sound=c0
                    )
                )
            )
            p = pressure(float(f), speed, zs, r, zr, impedance, top=top)
            energy += np.abs(p) ** 2 * direct2 / frequencies.shape[1]
        out[centre] = (10.0 * np.log10(energy)).tolist()
    return out


def reference(name: str, resolution: str) -> dict[int, list[float]]:
    """The program's level re free field (dB) at a case's listeners, per band."""
    bands = bands_of(name, resolution)
    if not bands:
        return {}
    key = f"{name} {resolution}".replace(" ", "_").replace(",", "").replace("/", "")
    path = None if CACHE["dir"] is None else Path(CACHE["dir"]) / f"{key}.json"
    weather, ground, zs, zr = CASES[name]
    ranges = points_of(name)
    asked = {"ranges": ranges.tolist(), "top": TOP_M, "bands": sorted(bands)}
    if path is not None and path.exists():
        kept = json.loads(path.read_text())
        if kept["asked"] == asked:
            return {int(k): v for k, v in kept["levels"].items()}
    out = program(weather, ground, zs, zr, ranges, resolution, bands)
    if path is not None:
        path.write_text(json.dumps({"asked": asked, "levels": out}))
    return out


def scene_for(weather, ground, zs, zr, ranges, resolution) -> Scene:
    """A source at the origin ``zs`` up over ``ground`` in ``weather``, the listeners
    ``zr`` up along x at ``ranges``."""
    return Scene.model_validate(
        {
            "name": "steep sound",
            "geometry": {
                "ground": {
                    "extent": [[-50.0, -50.0], [float(max(ranges)) + 50.0, 50.0]],
                    "sky": max(zs, zr) + 30.0,
                }
            },
            "materials": {
                "field": {
                    "impedance": {"model": "delany_bazley", "flow_resistivity": GROUNDS[ground]},
                    "scattering": 0.0,
                }
            },
            "default_material": "field",
            "sources": [{"name": "s", "position": [0.0, 0.0, zs]}],
            "listeners": [
                {"name": f"l{k}", "position": [float(x), 0.0, zr]} for k, x in enumerate(ranges)
            ],
            "air": {"temperature_c": GROUND_C, "humidity_percent": 50.0},
            "atmosphere": atmosphere_of(weather),
            "simulation": {
                "seed": 1,
                "bands": resolution,
                "early": {
                    "method": "image_sources",
                    "max_order": 1,
                    "diffraction": "none",
                    "open_ground_frequencies": "rays",
                },
                "late": {"rays": 100},
                "ir_length_s": 3.2,
                "performance": {"threads": 1},
            },
        }
    )


def scene_of(name: str, resolution: str) -> Scene:
    """A case as a scene."""
    weather, ground, zs, zr = CASES[name]
    return scene_for(weather, ground, zs, zr, points_of(name), resolution)


@contextlib.contextmanager
def narrow_angle(step: float | None, wavelengths: float | None = None):
    """The narrow-angle equation at every point, its range step ``step`` times its own
    (or ``wavelengths`` of the highest frequency, with no least step); None: the engine
    as it is."""
    kept = parabolic.STEEP_DEG, parabolic.RANGE_STEP_WAVELENGTHS, parabolic.RANGE_STEP_MIN_M
    try:
        if step is not None or wavelengths is not None:
            parabolic.STEEP_DEG = 90.0
        if step is not None:
            parabolic.RANGE_STEP_WAVELENGTHS = kept[1] * step
            parabolic.RANGE_STEP_MIN_M = kept[2] * step
        if wavelengths is not None:
            parabolic.RANGE_STEP_WAVELENGTHS = wavelengths
            parabolic.RANGE_STEP_MIN_M = 0.0
        yield
    finally:
        parabolic.STEEP_DEG, parabolic.RANGE_STEP_WAVELENGTHS, parabolic.RANGE_STEP_MIN_M = kept


def levels(scene: Scene, bands: dict[int, int]) -> dict[int, list[float]]:
    """The engine's early level re free field (dB) at the scene's listeners, per band."""
    result = simulate(Room.from_scene(scene), threads=1)
    rows = []
    for listener in scene.listeners:
        response = result["s", listener.name]
        free = np.asarray(response.free_field, dtype=float)
        rows.append(10.0 * np.log10(total_energy(response.early) / free))
    array = np.array(rows)
    return {centre: array[:, band].tolist() for centre, band in bands.items()}


def run_case(name: str) -> dict:
    """A case's ranges, the program's levels and the engine's with each run."""
    _weather, _ground, zs, zr = CASES[name]
    r = points_of(name)
    row: dict = {"ranges": r.tolist(), "steep_from": steep_from(zs, zr)}
    for resolution in ("octave", "third_octave"):
        bands = bands_of(name, resolution)
        if not bands:
            continue
        part: dict = {"program": reference(name, resolution)}
        for run, step in RUNS.items():
            with narrow_angle(step):
                part[run] = levels(scene_of(name, resolution), bands)
        row[resolution] = part
    return row


def gather(out: dict, names, resolution: str, bands=None):
    """Every compared point of ``names`` in ``resolution``: the case, the band, the
    range, the angle (degrees) the reflection comes in at, the program's level and
    each run's level less it."""
    rows = []
    for name in names:
        if name not in out or resolution not in out[name]:
            continue
        _weather, _ground, zs, zr = CASES[name]
        part = out[name][resolution]
        r = np.array(out[name]["ranges"])
        angle = np.degrees(np.arctan2(zs + zr, r))
        for centre in part["program"]:
            if bands is not None and int(centre) not in bands:
                continue
            ffp = np.array(part["program"][centre])
            for k in np.flatnonzero(ffp >= FLOOR_DB):
                miss = {run: part[run][centre][k] - ffp[k] for run in RUNS}
                rows.append((name, int(centre), float(r[k]), float(angle[k]), float(ffp[k]), miss))
    return rows


def stat(values) -> str:
    a = np.abs(np.asarray(list(values), dtype=float))
    return "none" if a.size == 0 else f"{a.mean():.2f} ({a.max():.2f}) [{a.size}]"


def jumps(out: dict, names, resolution: str, bands=None) -> list:
    """The jump at 15 degrees in each case and band: (engine - program) just steeper
    less the same just flatter, where both are compared; with the case, the band and
    each side's own miss."""
    rows = []
    for name in names:
        if name not in out or resolution not in out[name]:
            continue
        part = out[name][resolution]
        r = np.array(out[name]["ranges"])
        edge = out[name]["steep_from"]
        inner = int(np.argmin(np.abs(r - edge * (1.0 - BESIDE))))
        outer = int(np.argmin(np.abs(r - edge * (1.0 + BESIDE))))
        for centre in part["program"]:
            if bands is not None and int(centre) not in bands:
                continue
            ffp = np.array(part["program"][centre])
            if ffp[inner] < FLOOR_DB or ffp[outer] < FLOOR_DB:
                continue
            steep = part["engine"][centre][inner] - ffp[inner]
            flat = part["engine"][centre][outer] - ffp[outer]
            rows.append((steep - flat, name, int(centre), steep, flat))
    return rows


def report(out: dict) -> dict:
    """The tables, printed and returned."""
    table: dict = {}
    limit = STEEP_DEG

    def lines(title, names, resolution, bands=None):
        rows = gather(out, names, resolution, bands)
        if not rows:
            return
        flat = [row for row in rows if row[3] <= limit]
        steep = [row for row in rows if row[3] > limit]
        near = [row for row in flat if row[3] >= NEAR_DEG]
        entry = {
            "flatter, narrow-angle": stat(row[5]["engine"] for row in flat),
            "steeper, narrow-angle": stat(row[5]["narrow"] for row in steep),
            "steeper, half its step": stat(row[5]["half"] for row in steep),
            "steeper, a quarter": stat(row[5]["quarter"] for row in steep),
            "steeper, wide-angle": stat(row[5]["engine"] for row in steep),
            f"{NEAR_DEG:g} to {limit:g} degrees, narrow-angle": stat(
                row[5]["engine"] for row in near
            ),
            f"{NEAR_DEG:g} to {limit:g} degrees, half its step": stat(
                row[5]["half"] for row in near
            ),
        }
        table[title] = entry
        print(f"\n{title}: |level - program| in dB, mean (largest) [points]")
        for key, value in entry.items():
            print(f"  {key:38s} {value}")

    def jump_lines(title, names, resolution, bands=None):
        rows = jumps(out, names, resolution, bands)
        if not rows:
            return
        sizes = np.abs([row[0] for row in rows])
        order = np.argsort(-sizes)[:5]
        top = [rows[k] for k in order]
        entry = {
            "jumps": len(rows),
            "mean": float(sizes.mean()),
            "largest": float(sizes.max()),
            "largest at": f"{top[0][1]}, {top[0][2]} Hz",
            "five largest": [
                {"jump": row[0], "case": row[1], "band": row[2], "steep": row[3], "flat": row[4]}
                for row in top
            ],
        }
        table[title] = entry
        print(
            f"\n{title}: {len(rows)} jumps at {limit:g} degrees, mean {sizes.mean():.2f} dB,"
            f" largest {sizes.max():.2f} dB ({entry['largest at']})"
        )
        print(
            "  at the five largest the wide-angle side is within"
            f" {max(abs(row[3]) for row in top):.2f} dB of the program, the narrow-angle side"
            f" up to {max(abs(row[4]) for row in top):.2f} dB off"
        )

    first = [name for name in FIRST if name in out]
    further = [name for name in FURTHER if name in out]
    lines("first set, octaves 63 to 500 Hz", first, "octave")
    lines(
        f"first set, third octaves 50 to 250 Hz, sources {THIRDS_FROM_M:g} m up and higher",
        first,
        "third_octave",
    )
    lines("further set, octaves 63 and 125 Hz", further, "octave")
    lines("further set, third octaves 80 to 160 Hz", further, "third_octave")
    jump_lines("both sets, octaves", first + further, "octave")
    jump_lines("both sets, third octaves", first + further, "third_octave")
    return table


def check_steps() -> dict:
    """The 63 Hz band of one listener with the narrow-angle equation at each of
    ``STEPS``, less the program's (dB)."""
    weather, ground, zs, zr, at, centre = STEPS_CASE
    bands = {centre: OCTAVES[centre]}
    ffp = program(weather, ground, zs, zr, [at], "octave", bands)[centre][0]
    out = {"program": ffp}
    scene = scene_for(weather, ground, zs, zr, [at], "octave")
    for wavelengths in STEPS:
        with narrow_angle(None, wavelengths):
            out[f"{wavelengths:g}"] = levels(scene, bands)[centre][0] - ffp
    steps = ", ".join(f"{w:g}: {out[f'{w:g}']:+.2f} dB" for w in STEPS)
    print(
        f"\n{weather}, {ground}, {zs:g} m to {zr:g} m, {at:g} m away, {centre} Hz:"
        f" the program {ffp:.2f} dB re free field; the narrow-angle equation less it, in"
        f" steps of (wavelengths) {steps}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))
    parser.add_argument("--set", choices=("first", "further"), help="this set only")
    parser.add_argument("--case", action="append", help="these cases only (repeatable)")
    parser.add_argument("--only", choices=("steps",), help="the range step's own check only")
    parser.add_argument("--cache", help="keep each case's reference in this directory")
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    if args.cache:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
        CACHE["dir"] = args.cache
    out: dict = {}
    if args.only == "steps":
        out["steps"] = check_steps()
    else:
        names = list(FIRST if args.set == "first" else FURTHER if args.set == "further" else CASES)
        if args.case:
            unknown = [name for name in args.case if name not in CASES]
            if unknown:
                parser.error(f"unknown case {unknown[0]!r}; the cases: {', '.join(CASES)}")
            names = args.case
        for name in names:
            started = time.time()
            out[name] = run_case(name)
            print(f"{name}  ({time.time() - started:.0f} s)", flush=True)
            if args.json:
                Path(args.json).write_text(json.dumps(out, indent=1))
        out["table"] = report(out)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
