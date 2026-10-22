"""Check sound over a terrain (``geometry.ground.terrain``) against closed forms and the
exact solution for a wedge, and time it on a large one.

    python validation/terrain.py [--json results.json]

The terrain is a height grid sampled over the ground's extent and cut into triangles
(:mod:`acoustic_spaces.geometry.terrain`, :func:`acoustic_spaces.geometry.outdoor.
terrain_box`): the early sound reflects off the mean plane of the ground
(:mod:`acoustic_spaces.engine.terrain`), bends over its crests
(:mod:`acoustic_spaces.engine.crests`), and the late rays bounce off its triangles.
The script checks:

- a flat terrain (every height alike) against the flat ground: the direct sound and the
  one reflection off the ground, their distances and energies, and the level relative
  to free field of whole runs, which differ only in how the ground is cut into
  triangles;
- a plane sloping up 20 % against the image source in the tilted plane: the path's
  distance and energy are the image's, to rounding;
- a ridge, two planes meeting at 242 degrees (slopes of 0.6), with the source on one
  side and the listener in its shadow on the other: the engine's sound bent over the
  ridge (the wedge of the string pulled taut over the profile, by the uniform theory
  of diffraction with the reflections off both slopes in its terms) against the
  Biot-Tolstoy-Medwin solution for the rigid wedge (:mod:`edge_diffraction`), per
  octave band, at three pairs of positions;
- a smooth hill, 12 m high with a Gaussian profile, with the listener in its shadow:
  the sound over its crest against the thin screen's attenuation for the same path
  difference (Maekawa's 10 lg(3 + 20 N), ISO 9613-2 without ground reflections) over
  a soft ground, and against ISO 9613-2's 10 lg(3 + 10 N), which stands for a typical
  ground's reflections, over a rigid one (engineering formulas, not exact solutions:
  a rounded crest has none); and the level's steps as a listener climbs out of the
  shadow over the rigid ground (it must not jump);
- the cost on a large terrain, rolling hills 2 km across on a 20 m grid (10,000 cells)
  with a building: building the model, finding its diffracting edges and running the
  early paths and the late rays.

Levels are in dB re the direct sound in free field over the same distance, the energy
averaged over the engine's eight frequencies in each octave band, with the early paths
added in pressure in every band, as outdoors by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_diffraction import Edge, band_level, btm_first  # noqa: E402

from acoustic_spaces import Room, Scene, simulate  # noqa: E402
from acoustic_spaces.engine import early_paths  # noqa: E402
from acoustic_spaces.engine.coherent import total_energy  # noqa: E402
from acoustic_spaces.engine.crests import CREST  # noqa: E402
from acoustic_spaces.engine.diffraction import find_edges  # noqa: E402
from acoustic_spaces.engine.model import TraceModel  # noqa: E402
from acoustic_spaces.materials import OCTAVE_BANDS_HZ  # noqa: E402

#: The grid: 101 x 101 points 10 m apart, from (0, 0).
SIZE, SPACING = 101, 10.0
SLOPE = 0.6
RIDGE_X = 500.0


def heights(fn) -> list[list[float]]:
    xs = [k * SPACING for k in range(SIZE)]
    return [[float(fn(x, y)) for x in xs] for y in xs]


def scene(terrain, source, listener, absorption: float = 0.0, rays: int = 20000, **early) -> Scene:
    return Scene.model_validate(
        {
            "geometry": {"ground": {"terrain": terrain, "sky": 100}},
            "materials": {"ground": {"absorption": [absorption] * 8, "scattering": 0.0}},
            "default_material": "ground",
            "sources": [{"name": "s", "position": list(map(float, source))}],
            "listeners": [{"name": "l", "position": list(map(float, listener))}],
            "simulation": {
                "ir_length_s": 1.0,
                "seed": 1,
                "early": {
                    "method": "image_sources",
                    "max_order": 1,
                    "diffraction": "none",
                    **early,
                },
                "late": {"rays": rays},
            },
        }
    )


def _row(values) -> str:
    return "".join(f"{v:+8.2f}" for v in values)


def check_flat() -> dict:
    """A flat terrain against the flat ground."""
    source, listener = (100.0, 500.0, 2.0), (400.0, 500.0, 1.5)
    flat = Scene.model_validate(
        {
            **scene(
                {"heights": heights(lambda x, y: 0.0), "spacing": SPACING}, source, listener, 0.2
            ).to_data(),
        }
    )
    data = flat.to_data()
    data["geometry"]["ground"] = {"extent": [[0, 0], [1000, 1000]], "sky": 100}
    plain = Scene.model_validate(data)
    rooms = {"flat ground": Room.from_scene(plain), "flat terrain": Room.from_scene(flat)}
    paths = {name: early_paths(room) for name, room in rooms.items()}
    print("A flat terrain against the flat ground: the direct sound and the reflection")
    print(f"{'':16}{'distance (m)':>28}{'energy (63 Hz band)':>24}")
    for name, p in paths.items():
        print(
            f"{name:16}"
            + "".join(f"{d:14.9f}" for d in p.distance_m)
            + "".join(f"{e:12.4e}" for e in p.energy[:, 0])
        )
    gap = np.abs(paths["flat terrain"].distance_m - paths["flat ground"].distance_m).max()
    ratio = np.abs(paths["flat terrain"].energy / paths["flat ground"].energy - 1.0).max()
    print(f"the terrain's paths differ by {gap:.1e} m and {ratio:.1e} in energy")
    levels = {}
    for name, room in rooms.items():
        result = simulate(room)
        levels[name] = np.asarray(result[("s", "l")].relative_to_free_field())
    print(f"{'level re free field (dB)':26}" + "".join(f"{f:>8g}" for f in OCTAVE_BANDS_HZ))
    for name, level in levels.items():
        print(f"{name:26}{_row(level)}")
    print(f"{'terrain - ground':26}{_row(levels['flat terrain'] - levels['flat ground'])}")
    print()
    return {
        "distance_gap_m": float(gap),
        "energy_ratio_gap": float(ratio),
        "level_db": {k: v.tolist() for k, v in levels.items()},
    }


def check_slope() -> dict:
    """A plane sloping up 20 % against the image source in the tilted plane."""
    rows = []
    print("A plane sloping up 20 %: the reflection against the image source in it")
    print(f"{'source':>22}{'listener':>22}{'engine (m)':>14}{'exact (m)':>14}{'energy gap':>12}")
    normal = np.array([-0.2, 0.0, 1.0]) / math.hypot(0.2, 1.0)
    for source, listener in (
        ((100.0, 500.0, 24.0), (300.0, 500.0, 62.0)),
        ((200.0, 400.0, 45.0), (260.0, 560.0, 53.0)),
    ):
        room = Room.from_scene(
            scene({"heights": heights(lambda x, y: 0.2 * x), "spacing": SPACING}, source, listener)
        )
        paths = early_paths(room)
        assert paths.order.tolist() == [0, 1]
        s = np.array(source)
        image = s - 2.0 * (s @ normal) * normal
        exact = float(np.linalg.norm(image - listener))
        direct = float(np.linalg.norm(s - np.array(listener)))
        air = room.air_attenuation(TraceModel.from_room(room).bands)
        expected = paths.energy[0] * (direct / exact) ** 2 * np.exp(-air * (exact - direct))
        gap = float(np.abs(paths.energy[1] / expected - 1.0).max())
        print(
            f"{str(source):>22}{str(listener):>22}{paths.distance_m[1]:14.9f}{exact:14.9f}{gap:12.1e}"
        )
        rows.append(
            {
                "source": source,
                "listener": listener,
                "engine_m": float(paths.distance_m[1]),
                "exact_m": exact,
                "energy_ratio_gap": gap,
            }
        )
    print()
    return rows


def _level(room: Room, source, listener) -> np.ndarray:
    """The level (dB re the direct sound in free field over the same distance, without
    the air's attenuation) of the early sound at the listener, added in pressure in
    every band: over the crests, the path over them with the ground's reflections in
    its wave; in sight, the direct sound, the nearest crest's wave and the ground's
    reflections."""
    paths = early_paths(room)
    distance = math.dist(listener, source)
    # The air's attenuation comes off each path over its own length (the exact
    # solutions have none), so that a longer path over a crest is not charged for it.
    air = room.air_attenuation("octave")
    dry = replace(paths, energy=paths.energy * np.exp(air[None] * paths.distance_m[:, None]))
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(total_energy(dry) * 4.0 * np.pi * distance**2)


def check_ridge() -> list[dict]:
    """The sound bent over a ridge against the exact solution for the rigid wedge."""
    terrain = {"heights": heights(lambda x, y: 20.0 - SLOPE * abs(x - RIDGE_X)), "spacing": SPACING}
    open_angle = math.pi + 2.0 * math.atan(SLOPE)
    down = np.array([-1.0, 0.0, -SLOPE]) / math.hypot(1.0, SLOPE)  # down the near slope
    up = np.array([-SLOPE, 0.0, 1.0]) / math.hypot(1.0, SLOPE)  # out of it, into the air
    edge = Edge([RIDGE_X, 0.0, 20.0], [0.0, 1.0, 0.0], down, up, open_angle / math.pi)
    print(
        f"A ridge, two slopes of {SLOPE:g} meeting at {math.degrees(open_angle):.1f} degrees: the"
    )
    print("sound bent over it (dB re free field) against the exact solution for the wedge")
    print(f"{'':44}" + "".join(f"{f:>8g}" for f in OCTAVE_BANDS_HZ))
    rows = []
    for above_s, above_l, x_s, x_l, y_l in (
        (2.0, 1.5, 470.0, 530.0, 500.0),
        (1.0, 4.0, 460.0, 560.0, 500.0),
        (3.0, 1.5, 480.0, 540.0, 530.0),
    ):
        source = (x_s, 500.0, 20.0 - SLOPE * abs(x_s - RIDGE_X) + above_s)
        listener = (x_l, y_l, 20.0 - SLOPE * abs(x_l - RIDGE_X) + above_l)
        room = Room.from_scene(scene(terrain, source, listener, diffraction="direct"))
        assert len(find_edges(TraceModel.from_room(room))) == 0
        paths = early_paths(room)
        assert (paths.edge == CREST).all() and len(paths) == 1, paths.edge
        distance = float(np.linalg.norm(np.subtract(listener, source)))
        btm = band_level(btm_first(edge, np.array(source), np.array(listener)), distance)
        level = _level(room, source, listener)
        name = (
            f"{source[0]:g}, {source[2]:.1f} to {listener[0]:g}, {listener[1]:g}, {listener[2]:.1f}"
        )
        print(f"{name:44}{_row(btm)}  exact")
        print(f"{'':44}{_row(level - btm)}  engine - exact")
        rows.append(
            {
                "source": list(source),
                "listener": list(listener),
                "btm_db": btm.tolist(),
                "engine_minus_btm_db": (level - btm).tolist(),
            }
        )
    print()
    return rows


def check_hill() -> dict:
    """The sound over a smooth hill against the thin screen's attenuation, and the
    level's steps as a listener climbs out of the shadow."""
    top, width = 12.0, 15.0
    hill = {
        "heights": heights(lambda x, y: top * math.exp(-(((x - RIDGE_X) / width) ** 2))),
        "spacing": SPACING,
    }
    source, listener = (460.0, 500.0, 1.5), (540.0, 500.0, 1.5)
    levels = {}
    for ground, absorption in (("soft", 0.999), ("rigid", 0.0)):
        room = Room.from_scene(scene(hill, source, listener, absorption, diffraction="direct"))
        paths = early_paths(room)
        straight = paths[paths.order == 0]
        assert len(straight) == 1 and straight.edge[0] == CREST
        corners = straight.points[0]
        delta = np.linalg.norm(np.diff(corners, axis=0), axis=1).sum() - math.dist(source, listener)
        levels[ground] = _level(room, source, listener)
    c = room.speed_of_sound()
    fresnel = np.array([2.0 * delta * f / c for f in OCTAVE_BANDS_HZ])
    # Maekawa's thin screen, ISO 9613-2 with C2 = 40 (no ground reflections), and the
    # standard's C2 = 20 (a typical ground's reflections included).
    thin = -10.0 * np.log10(3.0 + 20.0 * fresnel)
    typical = -10.0 * np.log10(3.0 + 10.0 * fresnel)
    print(f"A smooth hill {top:g} m high: the sound over its crest (path")
    print(f"difference {delta:.2f} m) against the thin screen's attenuation")
    print(f"{'':26}" + "".join(f"{f:>8g}" for f in OCTAVE_BANDS_HZ))
    print(f"{'soft ground (dB)':26}{_row(levels['soft'])}")
    print(f"{'thin screen, no ground':26}{_row(thin)}  10 lg(3 + 20 N)")
    print(f"{'soft - thin':26}{_row(levels['soft'] - thin)}")
    print(f"{'rigid ground (dB)':26}{_row(levels['rigid'])}")
    print(f"{'ISO 9613-2, C2 = 20':26}{_row(typical)}  10 lg(3 + 10 N)")
    print(f"{'rigid - ISO':26}{_row(levels['rigid'] - typical)}")
    ups = np.arange(1.5, 30.0, 0.5)
    climb = []
    for z in ups:
        room = Room.from_scene(scene(hill, source, (540.0, 500.0, z), diffraction="direct"))
        climb.append(_level(room, source, (540.0, 500.0, z)))
    climb = np.array(climb)
    steps = np.diff(climb, axis=0)
    # Where the line of sight from the source clears the hill.
    x = np.linspace(source[0] + 0.5, listener[0], 2000)
    over = top * np.exp(-(((x - RIDGE_X) / width) ** 2))
    run = (listener[0] - source[0]) / (x - source[0])
    clear = float(np.max(source[2] + (over - source[2]) * run))
    print("climbing out of the shadow over the rigid ground in steps of 0.5 m, the level")
    print(f"{'changes by at most (dB)':26}{_row(np.abs(steps).max(axis=0))}")
    print(f"{'and falls by at most':26}{_row(np.maximum(-steps.min(axis=0), 0.0))}")
    print(f"the line of sight clears the hill from {clear:.1f} m up; the level (dB) at")
    for k in range(0, len(ups), 5):
        print(f"{ups[k]:>22.1f} m  {_row(climb[k])}")
    print()
    return {
        "path_difference_m": float(delta),
        "soft_db": levels["soft"].tolist(),
        "thin_screen_db": thin.tolist(),
        "rigid_db": levels["rigid"].tolist(),
        "iso_9613_2_db": typical.tolist(),
        "largest_step_db": np.abs(steps).max(axis=0).tolist(),
        "largest_fall_db": np.maximum(-steps.min(axis=0), 0.0).tolist(),
        "line_of_sight_clears_m": clear,
        "climb_heights_m": ups.tolist(),
        "climb_db": climb.tolist(),
    }


def check_cost() -> dict:
    """Building and running a scene over rolling hills 2 km across."""
    n, d = 101, 20.0
    xs = [k * d for k in range(n)]

    def hill(x: float, y: float) -> float:
        return 20.0 * math.sin(x / 150.0) * math.cos(y / 210.0) + 25.0

    rolling = [[hill(x, y) for x in xs] for y in xs]
    data = scene(
        {"heights": rolling, "spacing": d},
        (300.0, 1000.0, hill(300.0, 1000.0) + 2.0),
        (1500.0, 1100.0, hill(1500.0, 1100.0) + 1.5),
        0.1,
        diffraction="direct",
    ).to_data()
    data["geometry"]["buildings"] = [
        {
            "name": "hall",
            "footprint": [[900, 950], [960, 950], [960, 1000], [900, 1000]],
            "height": 12,
        }
    ]
    data["simulation"]["early"] = {"method": "ray_tracing", "max_order": 2, "diffraction": "direct"}
    started = time.perf_counter()
    room = Room.from_scene(Scene.model_validate(data))
    built = time.perf_counter() - started
    started = time.perf_counter()
    model = TraceModel.from_room(room)
    edges = find_edges(model)
    found = time.perf_counter() - started
    started = time.perf_counter()
    paths = early_paths(room)
    early = time.perf_counter() - started
    started = time.perf_counter()
    simulate(room)
    whole = time.perf_counter() - started
    faces = len(room.mesh.faces)
    print(f"Rolling hills 2 km across on a {d:g} m grid with a building: {faces} triangles")
    print(f"  the model built in {built:.1f} s, its {len(edges)} diffracting edges found in")
    print(f"  {found:.1f} s, {len(paths)} early paths in {early:.1f} s, a whole run with")
    print(f"  20,000 rays in {whole:.1f} s")
    print()
    return {
        "triangles": faces,
        "edges": len(edges),
        "paths": len(paths),
        "build_s": built,
        "edges_s": found,
        "early_s": early,
        "run_s": whole,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    results = {
        "flat": check_flat(),
        "slope": check_slope(),
        "ridge": check_ridge(),
        "hill": check_hill(),
        "cost": check_cost(),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
