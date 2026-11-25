"""Continuity outdoors: how much the level steps as one thing changes a little, behind
barriers and buildings, in still air and in wind: the rows of the Continuity table in
docs/validation.md.

    python validation/continuity.py [--only ROW[,ROW...]] [--atmospheres NAME[,NAME...]]
        [--json results.json]

The level is that of the early paths end to end
(:func:`~acoustic_spaces.engine.simulate.early_paths`, the coherent sum a simulation's
early part holds) in each octave band, with the scene's default settings. The scenes are
those of ``tests/test_barriers_in_wind.py``: a source 1 m up over hard ground (absorption
0.02, scattering 0), a barrier 4 m high 15 m away across the way (the x axis), the paths
over the edges (``rows: edges``) with first-order reflections off the ground before and
after each edge (``first_order``, ``both_sides``). The atmospheres (:data:`ATMOSPHERES`)
are still air and the ``windy`` preset (5 m/s) blowing along the way and against it.
Each row changes one thing in small steps and gives, in each atmosphere and each band,
the largest change of the level from one step to the next:

- ``wind``: the wind, from none to 0.001, 0.01 and 0.1 m/s along the way and against it
  (no preset), at listeners 40, 60 and 120 m away 1.5 m up and 100 m away 4 m up: the
  largest change from still air at each speed;
- ``top``: the barrier's top 1 mm higher, listeners 40 and 60 m away 1.5 m up;
- ``sight``: a listener 40 m away rising 1 cm at a time from 8.98 to 9.02 m up, through
  the line of sight over the top (9 m up there), and 0.2 mm across that line;
- ``roof``: a building 20 m wide from 25 to 37 m away behind the barrier, its roof
  rising 1 cm at a time from 6 to 6.3 m, listeners 60 and 120 m away 5 m off the axis,
  1.5 m up, with paths over two edges (``second_order``);
- ``slant``: the barrier, 40 m long, slanted at 63 degrees to the way through the same
  point, listeners 60 and 120 m away 1.5 m up going round 0.01 degrees at a time from
  0.40 to 0.60 degrees off the way;
- ``start``: a barrier 1.5 m high 2 m from a source 0.5 m up, a listener 1.5 m up 2 cm
  further, across where the parabolic equation starts (``WAVE_MIN_RANGE`` times the
  two heights added: 3 m);
- ``end``: the barrier ending on the axis (from y = -20 to 0 m), listeners 60 and 120 m
  away 1.5 m up, 1 mm across the shadow boundary of its end;
- ``low_end``: the same with a barrier 1 m high 30 m away, whose end lies below the line
  of sight seen from above, listeners 100 and 200 m away;
- ``roof_edge``: no barrier and a building 6 m high and 20 m wide from 15 to 25 m away,
  a source 1 m over its roof at (20, 0, 7), a listener 1.5 m up 10 cm across the shadow
  boundaries of the roof edge's direct sound (47.5 m) and of its ground reflection
  (62.5 m), and 2 cm across where the parabolic equation starts (32.75 m); and the two
  swapped. In wind also the step of the weather's effect, the level in wind less that
  in still air;
- ``roof_end``: the barrier ending on the axis and a building 6 m high and 20 m wide from
  40 to 50 m away, a listener 1.5 m over its roof at (45, y, 7.5) 1 mm across the shadow
  boundary of the barrier's end (and the two swapped); also the step of the weather's
  effect.

``--only`` runs some rows and ``--atmospheres`` some atmospheres (``wind`` takes its own),
so that several runs can share the work. On one core the whole takes about 50 minutes,
``roof`` and ``slant`` about 22 and 16 of them, nearly all in the two winds.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.refracted import WAVE_MIN_RANGE
from acoustic_spaces.engine.simulate import early_paths
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

#: Still air and the ``windy`` preset along the way (towards +x) and against it.
ATMOSPHERES = {
    "still": None,
    "downwind": {"preset": "windy", "wind": {"direction_deg": 0.0}},
    "upwind": {"preset": "windy", "wind": {"direction_deg": 180.0}},
}
SOURCE = (0.0, 0.0, 1.0)
BARRIER = {"name": "screen", "path": [[15.0, -20.0], [15.0, 20.0]], "height": 4.0}
#: The barrier ending on the x axis.
ENDING = {"name": "screen", "path": [[15.0, -20.0], [15.0, 0.0]], "height": 4.0}
#: Half a millimetre either side of a shadow boundary in the plane y = 0.
ACROSS = (-0.0005, 0.0005)


def scene(
    atmosphere=None,
    listeners=((40.0, 0.0, 1.5),),
    sources=(SOURCE,),
    barriers=(BARRIER,),
    buildings=(),
    **early,
) -> Scene:
    """The scene of ``tests/test_barriers_in_wind.py``, its ground reaching 30 m past
    the farthest point and the response long enough for it: sources named s (one) or
    s0, s1 ..., listeners l0, l1 ..."""
    far = max(max(abs(p[0]), abs(p[1])) for p in (*sources, *listeners))
    data = {
        "geometry": {
            "ground": {"extent": [[-60.0, -60.0], [max(far + 30.0, 90.0), 60.0]], "sky": 20.0},
            "barriers": list(barriers),
            "buildings": list(buildings),
        },
        "materials": {"hard": {"absorption": [0.02] * 8, "scattering": 0.0}},
        "assignments": {},
        "default_material": "hard",
        "sources": [
            {"name": "s" if len(sources) == 1 else f"s{k}", "position": list(p)}
            for k, p in enumerate(sources)
        ],
        "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
        "simulation": {
            "seed": 1,
            "bands": "octave",
            "early": {
                "method": "image_sources",
                "max_order": 1,
                "diffraction": "first_order",
                "diffraction_reflections": "both_sides",
                "rows": "edges",
                **early,
            },
            "late": {"rays": 20},
            "ir_length_s": far / 343.0 + 0.3,
        },
    }
    if atmosphere is not None:
        data["atmosphere"] = atmosphere
    return Scene.model_validate(data)


def levels(made: Scene) -> np.ndarray:
    """(pairs, bands) the level (dB) of the early paths between each source and each
    listener of ``made``, the listeners' index the faster."""
    room = Room.from_scene(made)
    out = []
    for source in made.sources:
        for listener in made.listeners:
            energy = total_energy(early_paths(room, source.name, listener.name, threads=1))
            out.append(10.0 * np.log10(np.maximum(energy, 1e-300)))
    return np.array(out)


def _largest(steps) -> list[float]:
    """(bands,) the largest of ``steps`` (..., bands) in each band, absolute, to six
    decimals, so that the table printed to three rounds them once."""
    steps = np.abs(np.asarray(steps))
    return steps.reshape(-1, steps.shape[-1]).max(axis=0).round(6).tolist()


def wind(atmospheres) -> dict:
    """The change from still air as the wind starts, along the way and against it."""
    listeners = ((40.0, 0.0, 1.5), (60.0, 0.0, 1.5), (120.0, 0.0, 1.5), (100.0, 0.0, 4.0))
    still = levels(scene(None, listeners))
    out = {"listeners": listeners, "still_db": still.round(4).tolist()}
    for speed in (0.001, 0.01, 0.1):
        changes = [
            levels(scene({"wind": {"speed_m_s": speed, "direction_deg": towards}}, listeners))
            - still
            for towards in (0.0, 180.0)
        ]
        out[f"{speed:g} m/s"] = {
            "steps_db": _largest(changes),
            "change_db": np.round(changes, 4).tolist(),
        }
    return out


def top(atmospheres) -> dict:
    """The barrier's top 1 mm higher."""
    listeners = ((40.0, 0.0, 1.5), (60.0, 0.0, 1.5))
    out = {}
    for name in atmospheres:
        found = [
            levels(scene(ATMOSPHERES[name], listeners, barriers=({**BARRIER, "height": h},)))
            for h in (4.0, 4.001)
        ]
        out[name] = {"steps_db": _largest(found[1] - found[0])}
    return out


def sight(atmospheres) -> dict:
    """A listener rising through the line of sight over the barrier's top."""
    heights = (8.98, 8.99, 9.0, 9.01, 9.02)
    pair = (9.0 - 1e-4, 9.0 + 1e-4)
    out = {}
    for name in atmospheres:
        found = levels(scene(ATMOSPHERES[name], [(40.0, 0.0, z) for z in (*heights, *pair)]))
        walk = found[: len(heights)]
        out[name] = {
            "steps_db": _largest(np.diff(walk, axis=0)),
            "across_the_line_db": _largest(found[-1] - found[-2]),
            "levels_db": walk.round(4).tolist(),
        }
    return out


def roof(atmospheres) -> dict:
    """A building's roof rising behind the barrier, 1 cm at a time."""
    listeners = ((60.0, 5.0, 1.5), (120.0, 5.0, 1.5))
    heights = [round(6.0 + 0.01 * k, 2) for k in range(31)]
    out = {}
    for name in atmospheres:
        found = np.array(
            [
                levels(
                    scene(
                        ATMOSPHERES[name],
                        listeners,
                        buildings=(
                            {
                                "name": "house",
                                "footprint": [[25, -10], [37, -10], [37, 10], [25, 10]],
                                "height": h,
                            },
                        ),
                        diffraction="second_order",
                    )
                )
                for h in heights
            ]
        )
        out[name] = {"steps_db": _largest(np.diff(found, axis=0))}
    return out


def slant(atmospheres) -> dict:
    """Listeners going round behind a barrier slanted to the way."""
    along = (20.0 * math.cos(math.radians(63.0)), 20.0 * math.sin(math.radians(63.0)))
    slanted = {
        "name": "screen",
        "path": [[15.0 - along[0], -along[1]], [15.0 + along[0], along[1]]],
        "height": 4.0,
    }
    bearings = [math.radians(0.40 + 0.01 * k) for k in range(21)]
    distances = (60.0, 120.0)
    listeners = [(d * math.cos(b), d * math.sin(b), 1.5) for d in distances for b in bearings]
    out = {}
    for name in atmospheres:
        found = levels(scene(ATMOSPHERES[name], listeners, barriers=(slanted,)))
        found = found.reshape(len(distances), len(bearings), -1)
        out[name] = {"steps_db": _largest(np.diff(found, axis=1))}
    return out


def start(atmospheres) -> dict:
    """A listener 2 cm further across where the parabolic equation starts."""
    source, height = (0.0, 0.0, 0.5), 1.5
    there = WAVE_MIN_RANGE * (source[2] + height)
    low = {"name": "screen", "path": [[2.0, -20.0], [2.0, 20.0]], "height": 1.5}
    listeners = [(there - 0.01, 0.0, height), (there + 0.01, 0.0, height)]
    out = {}
    for name in atmospheres:
        found = levels(scene(ATMOSPHERES[name], listeners, sources=(source,), barriers=(low,)))
        out[name] = {"steps_db": _largest(found[1] - found[0])}
    return out


def _across_an_end(atmospheres, barrier, distances) -> dict:
    out = {}
    for name in atmospheres:
        listeners = [(x, y, 1.5) for x in distances for y in ACROSS]
        found = levels(scene(ATMOSPHERES[name], listeners, barriers=(barrier,)))
        steps = found[1::2] - found[0::2]
        out[name] = {
            f"{x:g} m": {"steps_db": _largest(s)} for x, s in zip(distances, steps, strict=True)
        }
        out[name]["steps_db"] = _largest(steps)
    return out


def end(atmospheres) -> dict:
    """Across the shadow boundary of a barrier's end."""
    return _across_an_end(atmospheres, ENDING, (60.0, 120.0))


def low_end(atmospheres) -> dict:
    """Across the line of an end below the line of sight."""
    low = {"name": "screen", "path": [[30.0, -20.0], [30.0, 0.0]], "height": 1.0}
    return _across_an_end(atmospheres, low, (100.0, 200.0))


def _effects(found: dict, pairs) -> dict:
    """The level's steps between each pair of points, in each atmosphere, and in wind
    those of the weather's effect, the level less that in still air."""
    out = {}
    for name, level in found.items():
        steps = [level[b] - level[a] for a, b in pairs]
        out[name] = {"steps_db": [_largest(s) for s in steps]}
        if name != "still" and "still" in found:
            effect = level - found["still"]
            out[name]["effect_steps_db"] = [_largest(effect[b] - effect[a]) for a, b in pairs]
    return out


def roof_edge(atmospheres) -> dict:
    """Across a roof edge's shadow boundaries and where the equation starts."""
    house = {"name": "house", "footprint": [[15, -10], [25, -10], [25, 10], [15, 10]], "height": 6}
    over = (20.0, 0.0, 7.0)
    ground = [(x, 0.0, 1.5) for x in (47.5, 47.6, 62.5, 62.6, 32.74, 32.76)]
    pairs = ((0, 1), (2, 3), (4, 5))
    out = {"pairs": ["direct sound's boundary", "reflection's boundary", "the equation's start"]}
    for swapped in (False, True):
        found = {}
        for name in atmospheres:
            if swapped:
                made = scene(ATMOSPHERES[name], (over,), ground, barriers=(), buildings=(house,))
            else:
                made = scene(ATMOSPHERES[name], ground, (over,), barriers=(), buildings=(house,))
            found[name] = levels(made)
        out["swapped" if swapped else "from the roof"] = _effects(found, pairs)
    return out


def roof_end(atmospheres) -> dict:
    """Over a roof, across the shadow boundary of a barrier's end."""
    house = {"name": "house", "footprint": [[40, -10], [50, -10], [50, 10], [40, 10]], "height": 6}
    over = [(45.0, y, 7.5) for y in ACROSS]
    out = {}
    for swapped in (False, True):
        found = {}
        for name in atmospheres:
            if swapped:
                made = scene(ATMOSPHERES[name], (SOURCE,), over, (ENDING,), (house,))
            else:
                made = scene(ATMOSPHERES[name], over, (SOURCE,), (ENDING,), (house,))
            found[name] = levels(made)
        out["swapped" if swapped else "to the roof"] = _effects(found, ((0, 1),))
    return out


#: The rows, in the table's order.
ROWS = {
    "wind": wind,
    "top": top,
    "sight": sight,
    "roof": roof,
    "slant": slant,
    "start": start,
    "end": end,
    "low_end": low_end,
    "roof_edge": roof_edge,
    "roof_end": roof_end,
}


def _show(values) -> str:
    return " | ".join(f"{v:.3f}" for v in values)


def _print(name: str, found: dict, indent: str = "") -> None:
    """Every list of steps in ``found``, one line each, labelled by its keys."""
    for key, value in found.items():
        if isinstance(value, dict):
            _print(f"{name}, {key}", value, indent)
        elif key in ("steps_db", "across_the_line_db", "effect_steps_db"):
            rows = value if value and isinstance(value[0], list) else [value]
            for k, row in enumerate(rows):
                label = f"{name}, {key[:-3].replace('_', ' ')}" + (f" {k}" if len(rows) > 1 else "")
                print(f"| {indent}{label} | {_show(row)} | {max(row):.3f} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="these rows only, comma-separated: " + ", ".join(ROWS))
    parser.add_argument(
        "--atmospheres", default=",".join(ATMOSPHERES), help="these atmospheres only"
    )
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    names = args.only.split(",") if args.only else list(ROWS)
    atmospheres = args.atmospheres.split(",")
    for name in (*names, *atmospheres):
        if name not in ROWS and name not in ATMOSPHERES:
            parser.error(f"no row or atmosphere {name!r}")
    header = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ)
    print(f"| largest step (dB) | {header} | most |")
    print("|---|" + "---:|" * (len(OCTAVE_BANDS_HZ) + 1))
    results: dict = {}
    for name in names:
        began = time.perf_counter()
        results[name] = ROWS[name](atmospheres)
        results[name]["seconds"] = round(time.perf_counter() - began, 1)
        _print(name, results[name])
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
