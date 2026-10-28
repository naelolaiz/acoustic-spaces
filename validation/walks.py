"""Walk listeners and a car 1 cm at a time past the edges in the examples, and find where
the early level steps.

    python validation/walks.py [--json results.json] [--walks gallery archway street]

No exact solution covers a whole room or a street, so the check is that the early level
does not jump where a path appears or goes (the validation report's "Round a corner
indoors", "Along a street" and "Known issues of the edge fade"):

- ``gallery``: the visitor walks away from the guide and round the corner at (8, 8) m,
  1 cm at a time from y = 6 to 18 m along x = 3, 4 and 5.5 m (3,603 positions), with
  reflections up to third order and ``first_order``, as the example has them;
- ``archway``: the listener behind the wall walks along x = 7.6 m from y = 5.30 to
  7.80 m (251 positions), where the reflection off the arch's jamb appears;
- ``street``: the car drives along its lane from x = -55 to 70 m, every 10 cm, heard on
  the pavement, the balcony and in the garden; then 1 cm apart round each place where a
  step of 10 cm is among the four largest in some band or in dB(A) for that listener.
  Outdoors the paths add in pressure, so the level also swings smoothly as they go in
  and out of step: a step there is how far it stands out from the steps either side,
  ``|d - (d_before + d_after) / 2|``.

Each walk is made with the edge fade and without it (the early stage with the edge
diffraction alone). The level is each octave band's early energy, the street's also
A-weighted with the car's sound power. The tables give the largest step between
neighbours in each band, and for the street each place where one stands out by more
than 0.1 dB in some band but 8 kHz, where a centimetre is a quarter of a wavelength.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
STEP = 0.01
#: A-weighting of the octave bands, 63 Hz to 8 kHz (dB).
A_WEIGHTING = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])
#: Steps of 10 cm walked again 1 cm at a time, for each listener: the largest in each
#: band and in dB(A).
REFINED = 4
#: Steps that stand out by more than this in some band are listed (dB).
LISTED = 0.1


@contextmanager
def _fade(on: bool):
    """The early stage with the edge fade, or without it (``on`` False)."""
    # The engine package exports the function simulate under the module's name.
    simulate = importlib.import_module("acoustic_spaces.engine.simulate")
    kept = simulate.EdgeFade
    if not on:
        simulate.EdgeFade = lambda *args, **kwargs: None
    try:
        yield
    finally:
        simulate.EdgeFade = kept


def _levels(rooms) -> np.ndarray:
    return np.array([10.0 * np.log10(total_energy(paths)) for paths in rooms])


def gallery(fade: bool, lines=(3.0, 4.0, 5.5), span=(6.0, 18.0)) -> dict:
    """The visitor along each of ``lines`` (x), from y = ``span[0]`` to ``span[1]``."""
    room = Room.from_yaml(EXAMPLES / "gallery" / "scene.yaml")
    scene = room.scene
    visitor = scene.listeners[1]
    ys = np.round(np.arange(span[0], span[1] + STEP / 2, STEP), 2)
    levels = {}
    with _fade(fade):
        for x in lines:

            def paths(y, x=x):
                moved = visitor.model_copy(update={"position": (x, float(y), 1.6)})
                walked = scene.model_copy(update={"listeners": [moved]})
                return early_paths(Room.from_scene(walked, reuse=room), max_order=3)

            levels[x] = _levels(paths(y) for y in ys)
    return {"positions": ys, "lines": levels}


def archway(fade: bool, span=(5.30, 7.80)) -> dict:
    """The listener behind the wall along x = 7.6 m, from y = ``span[0]`` to ``span[1]``."""
    scene = Scene.from_yaml(EXAMPLES / "archway" / "scene.yaml")
    room = Room.from_scene(scene)
    listener = next(item for item in scene.listeners if item.name == "behind the wall")
    ys = np.round(np.arange(span[0], span[1] + STEP / 2, STEP), 2)

    def paths(y):
        moved = listener.model_copy(update={"position": (7.6, float(y), 1.2)})
        walked = Room.from_scene(scene.model_copy(update={"listeners": [moved]}), reuse=room)
        return early_paths(walked, "singer", moved.name)

    with _fade(fade):
        return {"positions": ys, "lines": {7.6: _levels(paths(y) for y in ys)}}


class _Street:
    """The car at a place on its lane, heard by each listener."""

    def __init__(self) -> None:
        self.scene = Scene.from_yaml(EXAMPLES / "street" / "scene.yaml")
        self.car = next(item for item in self.scene.sources if item.name == "car")
        self.listeners = [item.name for item in self.scene.listeners]
        self.power = 10.0 ** ((np.array(self.car.level_db) + A_WEIGHTING) / 10.0)

    def levels(self, x: float, listeners: list[str] | None = None) -> np.ndarray:
        """(listeners, bands + 1): each band's level and dB(A) last."""
        update = {"position": (float(x), -2.0, 0.05), "walk": None, "enabled": True}
        scene = self.scene.model_copy(update={"sources": [self.car.model_copy(update=update)]})
        out = []
        for name in listeners or self.listeners:
            energy = total_energy(early_paths(scene, "car", name))
            weighted = np.sum(energy * self.power)
            out.append(np.append(10.0 * np.log10(energy), 10.0 * np.log10(weighted)))
        return np.array(out)


def _stands_out(levels: np.ndarray) -> np.ndarray:
    """How far each step between neighbours stands out from the steps either side."""
    steps = np.diff(levels, axis=0)
    out = np.zeros_like(steps)
    out[1:-1] = np.abs(steps[1:-1] - 0.5 * (steps[:-2] + steps[2:]))
    return out


def street(fade: bool, places: list | None = None) -> dict:
    """The walk every 10 cm, and 1 cm apart at ``places`` (listener, x), by default
    round the largest steps of 10 cm."""
    walk = _Street()
    coarse = np.round(np.arange(-55.0, 70.0 + 0.05, 0.1), 2)
    with _fade(fade):
        levels = np.array([walk.levels(x) for x in coarse])  # (x, listener, band)
        if places is None:
            places = []
            for listener, name in enumerate(walk.listeners):
                steps = np.abs(np.diff(levels[:, listener], axis=0))
                for column in range(steps.shape[1]):
                    for i in np.argsort(steps[:, column])[-REFINED:]:
                        places.append((name, round(float(coarse[i]), 2)))
            places = sorted(set(places))
        rows = []
        for name, x in places:
            fine = np.round(x + np.arange(-STEP, 0.1 + 1.5 * STEP, STEP), 2)
            heard = np.array([walk.levels(at, [name])[0] for at in fine])
            out = _stands_out(heard)
            i = int(np.argmax(out[:, :-2].max(axis=1)))  # bands but 8 kHz and dB(A)
            rows.append(
                {
                    "listener": name,
                    "walked_from": x,
                    "x": float(fine[i + 1]),
                    "stands_out_db": out[i].tolist(),
                    "steps_db": np.diff(heard, axis=0)[i].tolist(),
                    "walked": fine[1:].tolist(),
                    "stands_out_walked_db": out.tolist(),
                }
            )
    return {
        "positions": coarse,
        "levels": levels,
        "listeners": walk.listeners,
        "places": places,
        "rows": rows,
    }


def _row(values) -> str:
    return " | ".join(f"{v:.2f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the numbers to this file")
    parser.add_argument(
        "--walks", nargs="+", default=["gallery", "archway", "street"], help="which walks"
    )
    args = parser.parse_args()
    header = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ)
    rule = "|---|" + "---:|" * len(OCTAVE_BANDS_HZ)
    results = {}
    for name in args.walks:
        walk = {"gallery": gallery, "archway": archway, "street": street}[name]
        results[name] = {}
        places = None
        for fade, label in ((True, "with the fade"), (False, "without")):
            start = time.perf_counter()
            found = walk(fade) if name != "street" else walk(fade, places)
            seconds = time.perf_counter() - start
            results[name][label] = {"seconds": round(seconds, 1)}
            if name != "street":
                steps = {
                    f"{x:g}": np.abs(np.diff(levels, axis=0))
                    for x, levels in found["lines"].items()
                }
                largest = np.max([s.max(axis=0) for s in steps.values()], axis=0)
                where = {
                    x: found["positions"][1:][s.argmax(axis=0)].tolist() for x, s in steps.items()
                }
                results[name][label].update(largest_step_db=largest.tolist(), at=where)
                continue
            places = found["places"]  # walked again without the fade
            results[name][label].update(
                listeners=found["listeners"],
                largest_step_10cm_db=np.abs(np.diff(found["levels"], axis=0)).max(axis=0).tolist(),
                places=found["rows"],
            )
        print(f"\n{name}")
        if name != "street":
            print(f"| largest step (dB) | {header} |\n{rule}")
            for label, row in results[name].items():
                print(f"| {label} | {_row(row['largest_step_db'])} |")
            continue
        print("| car at x (m) | listener | stands out (dB) | in | dB(A) | without the fade (dB) |")
        print("|---:|---|---:|---|---:|---:|")
        rows = zip(
            results[name]["with the fade"]["places"],
            results[name]["without"]["places"],
            strict=True,
        )
        for place, other in rows:
            out = np.array(place["stands_out_db"])
            band = int(np.argmax(out[:-2]))
            if out[band] <= LISTED:
                continue
            # Without the fade, the same step in the same band.
            plain = other["stands_out_walked_db"][other["walked"].index(place["x"])][band]
            print(
                f"| {place['x']:.2f} | {place['listener']} | {out[band]:.2f} "
                f"| {OCTAVE_BANDS_HZ[band]:g} Hz | {out[-1]:.2f} | {plain:.2f} |"
            )
    for name, rows in results.items():
        for label, row in rows.items():
            print(f"{name}, {label}: {row['seconds']:.0f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
