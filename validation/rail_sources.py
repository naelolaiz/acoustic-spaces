"""Check the pass-by demos' trains (:mod:`acoustic_spaces.demos.rail`) against Schall
03's own way of reckoning their level, and against the EU's limits for new trains.

    python validation/rail_sources.py [--json results.json]

- **Schall 03's sum, in a free field.** A train going past as the railway demo plays it:
  each of its sources' front point walking along the track in the engine
  (:func:`~acoustic_spaces.walkthrough.simulate_walk`, every STEP_M), with Schall 03's
  directivity, and its other points behind it as far as they are behind the front
  (:func:`~acoustic_spaces.demos.passby.trailed`), heard where EN ISO 3095 measures a
  pass-by (7.5 m from the track's centre 1.2 m above the rail top, and 25 m 3.5 m above
  it) over a ground that takes all the sound; its sound exposure level (SEL) against
  Schall 03's own sum over the same stretch of track: each unit's sound power per metre
  of track (equation 1) in pieces of PIECE_M, each piece a point with its directivity
  (equation 8) heard over a sphere (equation 11) through the air of Tabelle 17 (equation
  12, 10 °C and 70 %, as the scene's air is set here), one train an hour (L_pAeq,1h,
  with the SEL 10 lg 3600 dB over it). Schall 03's sum needs none of what the demo
  assumes: neither where a unit's sources are along it, nor how its points share them.
- **Over the railway example's ground.** The same pass over the example's track bed
  (ballast, an impedance) and its meadow (grass), without its barrier, as the engine
  gives it, against Schall 03 with its ground: the solid angle of the ground's
  reflection near the source (D_Ω, equation 9) and the ground's attenuation (A_gr,B,
  equation 14, which does not depend on the frequency), the source 0.2 m over the ground
  more than over the rail top. The two reckon the ground in different ways; their
  difference is reported, not checked.
- **The TSI's limits.** The S-Bahn (an electric multiple unit), the electric locomotive,
  a coach and a freight wagon of the demo, each alone at 80 km/h over the example's
  ground: their level over the pass (L_pAeq,Tp, the time Tp being the unit's length over
  its speed) 7.5 m from the track 1.2 m above the rail top, as the engine gives it and
  as Schall 03 gives it, beside the limits the EU sets for new trains at that place and
  speed (Commission Regulation (EU) No 1304/2014, the TSI Noise, table 4). Those limits
  are for new units on a track of low roughness (EN ISO 3095:2013, 6.2); Schall 03's
  levels are for units of the fleet as it ran in Germany on rails in average condition.
  Schall 03's being over a limit says that, not that either is wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene
from acoustic_spaces.demos import rail
from acoustic_spaces.demos.passby import Trail, heard_together, trailed
from acoustic_spaces.directivity import write_directivity_csv
from acoustic_spaces.parameters import A_WEIGHTING_DB
from acoustic_spaces.walkthrough import pass_by, simulate_walk

#: Where the walk of each front point starts and ends along the track (m): the
#: listeners are across its middle.
WALK_M = (-100.0, 100.0)
#: How far apart the engine simulates the front points (m), and the pieces of track of
#: Schall 03's sum (m).
STEP_M = 1.0
PIECE_M = 0.1
#: The rail top over the ground (m), as in the railway example.
RAIL_M = 0.2
#: EN ISO 3095's places for a pass-by: (name, from the track's centre, over the rail
#: top) (m).
PLACES = (("7.5 m", 7.5, 1.2), ("25 m", 25.0, 3.5))
#: Schall 03's Tabelle 17: the air's attenuation per octave band, 63 Hz to 8 kHz, at
#: 10 °C and 70 % (dB per 1000 m).
AIR_DB_KM = (0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0)
#: The trains checked against Schall 03's sum, and their speeds (km/h).
CHECKED = (("s_bahn", 80.0), ("regional", 160.0))
#: The TSI Noise's limits at 80 km/h 7.5 m from the track (dB, L_pAeq,Tp), and the units
#: of the demo's trains they are checked with: (what, limit, unit).
TSI = (
    ("electric multiple unit (the S-Bahn)", 80.0, rail.TRAINS["s_bahn"].units[0]),
    ("electric locomotive", 84.0, rail.TRAINS["regional"].units[0]),
    ("coach", 79.0, rail.TRAINS["regional"].units[1]),
    ("freight wagon", 83.0, rail.TRAINS["freight"].units[1]),
)
TSI_KMH = 80.0
FREE = {"absorption": [1.0] * 8, "scattering": 0.0}


def lines_a(train: rail.Train, speed_kmh: float) -> dict[float, np.ndarray]:
    """The train's A-weighted sound power per metre of track with one train an hour, per
    octave band, by the height of its sources (dB): its units' by equation 1, added."""
    out: dict[float, np.ndarray] = {}
    for unit in train.units:
        for m, level in unit.line_levels_a(speed_kmh).items():
            height = (rail.TRAM_SOURCES if unit.kind.tram else rail.SOURCES)[m].height_m
            out[height] = out.get(height, 0.0) + 10.0 ** (np.asarray(level) / 10.0)
    return {h: 10.0 * np.log10(e) for h, e in sorted(out.items())}


def schall03_sel(
    train: rail.Train, speed_kmh: float, across_m: float, up_m: float, ground: bool
) -> np.ndarray:
    """(8,) the A-weighted SEL per octave band of one pass of ``train`` at ``speed_kmh``,
    ``across_m`` from the track and ``up_m`` over the rail top, by Schall 03: the track
    from WALK_M[0] to WALK_M[1] in pieces of PIECE_M, each a point source of its power
    per metre plus 10 lg PIECE_M, with D_I, A_div and A_atm, and with ``ground`` D_Ω and
    A_gr,B too."""
    x = np.arange(WALK_M[0] + PIECE_M / 2, WALK_M[1], PIECE_M)
    air = np.asarray(AIR_DB_KM) / 1000.0
    total = np.zeros(8)
    for height, level in lines_a(train, speed_kmh).items():
        dz = up_m - height
        d = np.sqrt(x**2 + across_m**2 + dz**2)
        sin2 = (across_m**2 + dz**2) / d**2
        # Equation 8, D_I, and equation 11, A_div; equation 12, A_atm, band by band.
        heard = (10.0 * np.log10(0.22 + 1.27 * sin2) - 10.0 * np.log10(4.0 * math.pi * d**2))[
            :, None
        ] - air[None, :] * d[:, None]
        if ground:
            hg, hr = RAIL_M + height, RAIL_M + up_m
            omega = 10.0 * math.log10(
                1.0 + (across_m**2 + (hg - hr) ** 2) / (across_m**2 + (hg + hr) ** 2)
            )
            hm = (hg + hr) / 2.0
            a_gr = np.maximum(4.8 - hm / d * (34.0 + 600.0 / d), 0.0)
            heard = heard + omega - a_gr[:, None]
        hour = (
            level
            + 10.0 * np.log10(PIECE_M)
            + 10.0 * np.log10(np.sum(10.0 ** (heard / 10.0), axis=0))
        )
        total += 10.0 ** (hour / 10.0)
    return 10.0 * np.log10(total) + 10.0 * math.log10(3600.0)


def scene_of(train: rail.Train, speed_kmh: float, ground: bool, table: Path) -> dict:
    """The train's front points walking along the track at ``speed_kmh``, a source a
    height, and the listeners at PLACES, over the railway example's ground (no barrier)
    or one that takes all the sound, the air at Tabelle 17's 10 °C and 70 %."""
    points = rail.train_points(train, speed_kmh)
    sources = []
    for height, (suffix, _) in rail.PARTS.items():
        mine = [p for p in points if p.height_m == height]
        if not mine:
            continue
        z = RAIL_M + height
        sources.append(
            {
                "name": f"train{suffix}",
                "position": [WALK_M[0], 0.0, z],
                "level_db": [float(v) for v in mine[0].power_db],
                "directivity": {"file": str(table)},
                "walk": {
                    "to": [[WALK_M[1], 0.0, z]],
                    "speed_m_s": speed_kmh / 3.6,
                    "face_forward": True,
                },
            }
        )
    if ground:
        geometry = {
            "ground": {
                "extent": [[-150, -40], [150, 40]],
                "sky": 20,
                "zones": [
                    {"name": "track_bed", "polygon": [[-150, -2], [150, -2], [150, 2], [-150, 2]]}
                ],
            }
        }
        materials = {"meadow": {"impedance": "grass"}, "ballast": {"impedance": "gravel"}}
        assignments = {"ground": "meadow", "track_bed": "ballast"}
    else:
        geometry = {"ground": {"extent": [[-150, -40], [150, 40]], "sky": 20}}
        materials, assignments = {"meadow": FREE}, {"ground": "meadow"}
    return {
        "geometry": geometry,
        "materials": materials,
        "assignments": assignments,
        "air": {"temperature_c": 10.0, "humidity_percent": 70.0},
        "sources": sources,
        "listeners": [
            {"name": name, "position": [0.0, across, RAIL_M + up]} for name, across, up in PLACES
        ],
        "simulation": {
            "ir_length_s": 0.3,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
            "late": {"method": "stochastic_ray_tracing", "rays": 2000},
            "seed": 1,
        },
    }


def engine_passes(
    train: rail.Train, speed_kmh: float, ground: bool, folder: Path
) -> dict[str, object]:
    """Each place's pass of ``train`` (:class:`~acoustic_spaces.walkthrough.PassBy`) as
    the engine gives it: each height's front point walked alone, its other points that
    much later and louder or quieter (:func:`~acoustic_spaces.demos.passby.trailed`),
    the heights' energies added."""
    table = write_directivity_csv(folder / "train.csv", rail.directivity(), "check")
    data = scene_of(train, speed_kmh, ground, table)
    v = speed_kmh / 3.6
    points = rail.train_points(train, speed_kmh)
    trails = {}
    for height, (suffix, _) in rail.PARTS.items():
        mine = [p for p in points if p.height_m == height]
        if mine:
            behind = tuple(
                ((p.behind_m - mine[0].behind_m) / v, None, p.power_db - mine[0].power_db)
                for p in mine[1:]
            )
            trails[f"train{suffix}"] = Trail(mine[0].behind_m / v, behind)
    longest = max(t.length_s for t in trails.values())
    out = {}
    for name, _, _ in PLACES:
        passes = []
        for part, trail in trails.items():
            alone = dict(data, sources=[s for s in data["sources"] if s["name"] == part])
            alone["listeners"] = [lis for lis in data["listeners"] if lis["name"] == name]
            room = Room.from_scene(Scene.model_validate(alone))
            walked = simulate_walk(room, part, listener=name, step_m=STEP_M)
            passes.append(trailed(pass_by(walked), trail, longest))
        out[name] = heard_together(passes)
    return out


def a_sum(levels_a: np.ndarray) -> float:
    return float(10.0 * np.log10(np.sum(10.0 ** (np.asarray(levels_a) / 10.0))))


def check_sums(folder: Path) -> dict:
    """The engine's pass against Schall 03's sum, in a free field and over the ground."""
    out: dict = {}
    for ground in (False, True):
        where = "ground" if ground else "free_field"
        print("over the railway example's ground" if ground else "in a free field")
        for key, speed in CHECKED:
            train = rail.TRAINS[key]
            passes = engine_passes(train, speed, ground, folder)
            for name, across, up in PLACES:
                engine = passes[name].exposure + np.asarray(A_WEIGHTING_DB)
                schall = schall03_sel(train, speed, across, up, ground)
                diff = engine - schall
                out.setdefault(where, {}).setdefault(key, {})[name] = {
                    "speed_kmh": speed,
                    "engine_sel_a_db": a_sum(engine),
                    "schall03_sel_a_db": a_sum(schall),
                    "band_differences_db": [round(float(v), 2) for v in diff],
                }
                print(
                    f"  {train.short} at {speed:g} km/h, {name}: SEL {a_sum(engine):.1f} dB(A) "
                    f"in the engine, {a_sum(schall):.1f} by Schall 03 "
                    f"({a_sum(engine) - a_sum(schall):+.2f}); octave bands "
                    + " ".join(f"{v:+.1f}" for v in diff)
                )
    return out


def check_tsi(folder: Path) -> dict:
    """Each unit alone at TSI_KMH over the ground: L_pAeq,Tp 7.5 m from the track."""
    out = {}
    name, across, up = PLACES[0]
    for what, limit, unit in TSI:
        train = rail.Train(what, (unit,), TSI_KMH, what)
        tp = unit.length_m / (TSI_KMH / 3.6)
        engine = engine_passes(train, TSI_KMH, True, folder)[name].exposure_a
        schall = a_sum(schall03_sel(train, TSI_KMH, across, up, True))
        under = 10.0 * math.log10(tp)
        out[what] = {
            "tp_s": tp,
            "engine_db_a": engine - under,
            "schall03_db_a": schall - under,
            "limit_db_a": limit,
        }
        print(
            f"  {what} ({unit.length_m:g} m, Tp {tp:.2f} s): L_pAeq,Tp {engine - under:.1f} "
            f"dB(A) in the engine, {schall - under:.1f} by Schall 03; the limit {limit:g}"
        )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        print("A train's pass in the engine and by Schall 03's sum, SEL (dB(A))")
        sums = check_sums(folder)
        print(f"\nEach unit alone at {TSI_KMH:g} km/h, 7.5 m from the track, against the TSI Noise")
        tsi = check_tsi(folder)
    print(f"\n{time.perf_counter() - started:.0f} s")
    if args.json:
        args.json.write_text(json.dumps({"sums": sums, "tsi": tsi}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
