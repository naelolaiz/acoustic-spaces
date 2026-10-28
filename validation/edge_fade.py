"""Check that reflections fade in and out at edges, against exact solutions.

    python validation/edge_fade.py [--json results.json]

The edge diffraction bends the direct sound round an edge and, with ``first_order``, the
first-order reflections; the edge fade (:mod:`acoustic_spaces.engine.edge_fade`) gives
every other specular path the terms at the edges it passes. A vertical edge that stands
on a rigid ground is exact by images: the ground mirrors the edge into itself, so the
sound is that of the edge, infinite in height, from the source and from its image in
the ground, each the geometric paths that reach the listener plus the exact
Biot-Tolstoy-Medwin solution (:mod:`edge_diffraction`). The script moves a listener
across the edges' boundaries, 1 cm at a time, and compares the engine with that, with
and without the fade:

- the end of a thin screen on the ground, with ``diffraction: direct``: the reflection
  off the ground crosses the shadow boundary with the direct sound, and only the fade
  bends it;
- the corner of a building on the ground (a right-angled wedge), with ``direct`` and
  with ``first_order``: the listener passes the boundary of the reflection off one wall
  and then the shadow boundary of the corner, each with its copy off the ground. With
  ``first_order`` the edge diffraction bends every one of those paths itself (the
  reflections off the corner's own walls belong to its diffraction, and the ground's
  is the one reflection before or after the corner it allows), and the fade must add
  nothing to it;
- a strip 2 m wide (a thin screen standing on the ground), with ``direct`` and with
  ``first_order``: the listener, on the source's side, passes through the reflection
  off the strip, which a face that narrow gives only in part at low frequencies; the
  edge diffraction and the fade share the terms of the reflection itself (where its
  point lies off the strip, the far end's are the fade's), and only the fade has those
  of its copy off the ground;
- a gap of 4 m between two screens in line, on the ground, with ``direct``: the listener
  passes behind one screen's end, through the gap and behind the other's;
- a thin barrier 3 m high on the ground, with ``first_order`` and with ``both_sides``
  too: the listener rises behind it through the shadow boundaries of the direct sound
  and of its copy off the ground. The ground mirrors the barrier into a screen twice as
  high, so the exact sound is that of its top from the source and its image to the
  listener and its image: four terms, of which the edge diffraction takes three on one
  side and all four with ``both_sides``.

For the strip, the gap and the barrier the reference adds each edge's exact term (BTM)
to the geometric paths; the sound diffracted from one edge to the other is left out (at
the strip it changes the level by at most 0.1 dB at 63 to 250 Hz).

A thick barrier on the ground, 1 m wide and 3 m high, with ``second_order``, alone or
with ``both_sides``, is compared at three listeners in its shadow with the exact sound
over both its top corners (BTM of second order), by images alike: four paths over the
two corners, of which the edge diffraction follows only the one without the ground.

Everything is added in pressure in every band, as outdoors, and the comparison is of
the energy in each octave band relative to the direct sound in free field: the exact
solution's averaged over 96 frequencies in the band, the engine's as it averages its own
eight (each standing for its slice of the band, over which paths that arrive far apart
stop interfering). Eight frequencies alone would leave the exact level of several paths
a metre or more apart up to 2.5 dB off from 2 kHz up. The tables give the largest
difference from the exact solution over the whole line, and the largest step between
neighbouring listeners.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_diffraction import EXTENT, C, Edge, _room, btm_first, btm_second  # noqa: E402

from acoustic_spaces.engine import early as early_stage  # noqa: E402
from acoustic_spaces.engine.coherent import total_energy  # noqa: E402
from acoustic_spaces.engine.diffraction import (  # noqa: E402
    EdgeDiffraction,
    band_frequencies,
    find_edges,
)
from acoustic_spaces.engine.edge_fade import EdgeFade  # noqa: E402
from acoustic_spaces.engine.model import TraceModel  # noqa: E402
from acoustic_spaces.materials import OCTAVE_BANDS_HZ  # noqa: E402

#: Every band added in pressure, as outdoors.
BANDS = len(OCTAVE_BANDS_HZ)
#: The frequencies the exact solution is averaged over in each band, (bands, per band).
FREQUENCIES, _ = band_frequencies(96)
K = 2.0 * np.pi * FREQUENCIES.ravel() / C
#: Listeners every STEP metres along each line, which starts OFFSET metres along, so that
#: none lies exactly on a boundary, where the exact solution's terms are singular.
STEP = 0.01
OFFSET = 0.00317
#: The source and listener heights (m) over the ground.
SOURCE_Z, LISTENER_Z = 0.5, 1.5
#: The gap between the two screens (m), and the width of the strip (m).
GAP = 4.0
STRIP = 2.0
#: The height of the barriers on the ground (m), and the width of the thick one (m).
BARRIER = 3.0
THICK = 1.0
#: The ground's mirror.
MIRROR = np.array([1.0, 1.0, -1.0])


def ground() -> list:
    """The rigid ground, z = 0, facing up."""
    a = EXTENT
    return [[(-a, -a, 0.0), (a, -a, 0.0), (a, a, 0.0), (-a, a, 0.0)]]


def screen(y0: float, y1: float) -> list:
    """A thin screen in the plane x = 0 from y = ``y0`` to ``y1``, reaching down through
    the ground (so that it meets it with no edge) and far up."""
    a = EXTENT
    return [[(0.0, y0, -a), (0.0, y1, -a), (0.0, y1, a), (0.0, y0, a)]]


def corner() -> list:
    """A building filling x > 0, y > 0: its walls x = 0 and y = 0, facing out, through
    the ground and far up."""
    a = EXTENT
    return [
        [(0.0, 0.0, -a), (0.0, 0.0, a), (0.0, a, a), (0.0, a, -a)],  # x = 0, facing -x
        [(0.0, 0.0, -a), (a, 0.0, -a), (a, 0.0, a), (0.0, 0.0, a)],  # y = 0, facing -y
    ]


def barrier(width: float = 0.0) -> list:
    """A barrier BARRIER m high along y, reaching down through the ground: a thin screen
    in the plane x = 0, or a block from x = 0 to ``width`` with its faces facing out."""
    a, h = EXTENT, BARRIER
    if width == 0.0:
        return [[(0.0, -a, -a), (0.0, a, -a), (0.0, a, h), (0.0, -a, h)]]
    return [
        [(0.0, -a, h), (width, -a, h), (width, a, h), (0.0, a, h)],  # the top
        [(0.0, -a, -a), (0.0, -a, h), (0.0, a, h), (0.0, a, -a)],  # x = 0
        [(width, -a, -a), (width, a, -a), (width, a, h), (width, -a, h)],  # x = width
    ]


def _crossing_y(source, receiver) -> float:
    """Where along y the line from ``source`` to ``receiver`` crosses x = 0 (infinity
    where it does not)."""
    if source[0] * receiver[0] > 0.0 or source[0] == receiver[0]:
        return np.inf
    t = source[0] / (source[0] - receiver[0])
    return source[1] + t * (receiver[1] - source[1])


def _free(source, receiver) -> np.ndarray:
    r = float(np.linalg.norm(receiver - source))
    return np.exp(-1j * K * r) / r


def exact(case: str, source, receiver) -> np.ndarray:
    """The pressure at ``receiver`` per wave number (the direct sound e^(-jkr)/r), by
    images in the ground: for the source and its image, the geometric paths that reach
    the receiver and each edge's exact term."""
    out = np.zeros(len(K), dtype=complex)
    if case == "barrier":  # the thin barrier: its top, for the listener and its image too
        top = Edge([0, 0, BARRIER], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
        for src in (source, source * MIRROR):
            t = src[0] / (src[0] - receiver[0])  # where the line crosses x = 0
            if src[2] + t * (receiver[2] - src[2]) > BARRIER:
                out += _free(src, receiver)
            for rec in (receiver, receiver * MIRROR):
                out += btm_first(top, src, rec, K)
        return out
    for src in (source, source * MIRROR):
        if case == "screen end":  # the screen fills y > 0
            if not 0.0 < _crossing_y(src, receiver) < np.inf:
                out += _free(src, receiver)
            edge = Edge([0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], 2.0)
            out += btm_first(edge, src, receiver, K)
        elif case == "corner":  # the building fills x > 0, y > 0
            # The source sees the corner from x < 0, y > 0; the line is blocked where it
            # crosses x = 0 above y = 0.
            if not 0.0 < _crossing_y(src, receiver) < np.inf:
                out += _free(src, receiver)
            image = src * np.array([-1.0, 1.0, 1.0])  # off the wall x = 0
            if 0.0 < _crossing_y(image, receiver) < np.inf:
                out += _free(image, receiver)
            edge = Edge([0, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0], 1.5)
            out += btm_first(edge, src, receiver, K)
        elif case == "strip":  # a screen from y = -STRIP / 2 to STRIP / 2, seen from x < 0
            out += _free(src, receiver)
            image = src * np.array([-1.0, 1.0, 1.0])  # off the strip
            if abs(_crossing_y(image, receiver)) < STRIP / 2:
                out += _free(image, receiver)
            upper = Edge([0, STRIP / 2, 0], [0, 0, 1], [0, -1, 0], [1, 0, 0], 2.0)
            lower = Edge([0, -STRIP / 2, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], 2.0)
            out += btm_first(upper, src, receiver, K) + btm_first(lower, src, receiver, K)
        else:  # the gap: screens over y > GAP / 2 and y < -GAP / 2
            y = _crossing_y(src, receiver)
            if abs(y) < GAP / 2 or y == np.inf:
                out += _free(src, receiver)
            upper = Edge([0, GAP / 2, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], 2.0)
            lower = Edge([0, -GAP / 2, 0], [0, 0, 1], [0, -1, 0], [1, 0, 0], 2.0)
            out += btm_first(upper, src, receiver, K) + btm_first(lower, src, receiver, K)
    return out


def level(pressure: np.ndarray, distance: float) -> np.ndarray:
    """Octave band energies of ``pressure`` (per wave number) in dB re the direct sound
    over ``distance`` in free field."""
    energy = (np.abs(pressure) ** 2).reshape(FREQUENCIES.shape).mean(axis=1)
    return 10.0 * np.log10(energy * distance**2)


class Engine:
    """The engine's early paths from ``source`` in a model of ``quads``, with the edge
    diffraction in ``mode`` and with or without the fade."""

    def __init__(
        self, folder: Path, quads: list, source, mode: str, both_sides: bool = False
    ) -> None:
        self.source = np.asarray(source, dtype=float)
        room = _room(folder, "fade", quads, self.source, self.source + 1.0)
        self.model = TraceModel.from_room(room)
        self.edges = find_edges(self.model)
        self.c = room.speed_of_sound()
        self.air = np.zeros(BANDS)
        self.candidates = early_stage.exhaustive_candidates(self.model, self.source, 2)
        self.diffraction = EdgeDiffraction(
            self.model,
            self.source,
            speed_of_sound=self.c,
            air=self.air,
            mode=mode,
            edges=self.edges,
            coherent=BANDS,
            both_sides=both_sides,
        )
        self.fade = EdgeFade(self.diffraction, self.candidates)

    def level(self, receiver, fade: bool) -> np.ndarray:
        paths = early_stage.find_paths(
            self.model,
            self.source,
            receiver,
            self.candidates,
            speed_of_sound=self.c,
            air=self.air,
            max_delay_s=0.5,
            coherent=BANDS,
        )
        bent = self.diffraction.apply(receiver, paths, 0.5)
        if fade:
            bent = self.fade.join(bent, self.fade.terms(receiver, paths, 0.5, bent))
        paths = bent
        squared = float(np.sum((receiver - self.source) ** 2))
        return 10.0 * np.log10(total_energy(paths) * 4.0 * np.pi * squared)


#: The cases: name, the reference's case, the model, modes, source, and the line the
#: listener moves along (from, to).
CASES = [
    (
        "screen end, direct",
        "screen end",
        lambda: ground() + screen(0.0, EXTENT),
        "direct",
        (-4.0, -1.0, SOURCE_Z),
        ((4.0, -1.0, LISTENER_Z), (4.0, 3.0, LISTENER_Z)),
    ),
    (
        "building corner, direct",
        "corner",
        lambda: ground() + corner(),
        "direct",
        (-3.0, 2.0, SOURCE_Z),
        ((-7.0, -3.0, LISTENER_Z), (7.0, -3.0, LISTENER_Z)),
    ),
    (
        "building corner, first_order",
        "corner",
        lambda: ground() + corner(),
        "first_order",
        (-3.0, 2.0, SOURCE_Z),
        ((-7.0, -3.0, LISTENER_Z), (7.0, -3.0, LISTENER_Z)),
    ),
    (
        "2 m strip, direct",
        "strip",
        lambda: ground() + screen(-STRIP / 2, STRIP / 2),
        "direct",
        (-4.0, -1.5, SOURCE_Z),
        ((-3.0, -3.0, LISTENER_Z), (-3.0, 5.0, LISTENER_Z)),
    ),
    (
        "2 m strip, first_order",
        "strip",
        lambda: ground() + screen(-STRIP / 2, STRIP / 2),
        "first_order",
        (-4.0, -1.5, SOURCE_Z),
        ((-3.0, -3.0, LISTENER_Z), (-3.0, 5.0, LISTENER_Z)),
    ),
    (
        "4 m gap, direct",
        "gap",
        lambda: ground() + screen(GAP / 2, EXTENT) + screen(-EXTENT, -GAP / 2),
        "direct",
        (-4.0, 0.0, SOURCE_Z),
        ((4.0, -5.0, LISTENER_Z), (4.0, 5.0, LISTENER_Z)),
    ),
    (
        "barrier, first_order",
        "barrier",
        lambda: ground() + barrier(),
        "first_order",
        (-4.0, 0.0, SOURCE_Z),
        ((4.0, 0.0, 1.0), (4.0, 0.0, 8.0)),
    ),
    (
        "barrier, first_order, both_sides",
        "barrier",
        lambda: ground() + barrier(),
        "first_order, both_sides",
        (-4.0, 0.0, SOURCE_Z),
        ((4.0, 0.0, 1.0), (4.0, 0.0, 8.0)),
    ),
]


def run(case, folder: Path, step: float = STEP) -> dict:
    name, reference, quads, mode, source, (start, end) = case
    source = np.array(source)
    start, end = np.array(start), np.array(end)
    count = int(round(np.linalg.norm(end - start) / step)) + 1
    mode, *sides = mode.split(", ")
    engine = Engine(folder, quads(), source, mode, both_sides=bool(sides))
    rows = {"exact": [], "fade": [], "plain": []}
    along = (end - start) / np.linalg.norm(end - start)
    for t in np.linspace(0.0, 1.0, count):
        receiver = start + t * (end - start) + OFFSET * along
        distance = float(np.linalg.norm(receiver - source))
        rows["exact"].append(level(exact(reference, source, receiver), distance))
        rows["fade"].append(engine.level(receiver, True))
        rows["plain"].append(engine.level(receiver, False))
    exact_db = np.array(rows["exact"])
    out = {
        "case": name,
        "source": source.tolist(),
        "line": [start.tolist(), end.tolist()],
        "listeners": count,
        "exact_largest_step_db": np.abs(np.diff(exact_db, axis=0)).max(axis=0).tolist(),
    }
    for kind in ("fade", "plain"):
        found = np.array(rows[kind])
        out[kind] = {
            "largest_difference_db": np.abs(found - exact_db).max(axis=0).tolist(),
            "rms_difference_db": np.sqrt(np.mean((found - exact_db) ** 2, axis=0)).tolist(),
            "largest_step_db": np.abs(np.diff(found, axis=0)).max(axis=0).tolist(),
        }
    out["levels_db"] = {key: np.array(value).round(3).tolist() for key, value in rows.items()}
    return out


def thick_barrier(folder: Path) -> list[dict]:
    """The thick barrier on the ground with ``second_order``, alone and with
    ``both_sides``, with and without the fade, against the exact sound over both its top
    corners, by images, at three listeners in its shadow."""
    near = Edge([0, 0, BARRIER], [0, 1, 0], [1, 0, 0], [0, 0, 1], 1.5)  # facing the source
    far = Edge([THICK, 0, BARRIER], [0, 1, 0], [-1, 0, 0], [0, 0, 1], 1.5)
    source = np.array([-5.0, 0.0, SOURCE_Z])
    quads = ground() + barrier(THICK)
    engines = {
        mode: Engine(folder, quads, source, "second_order", both_sides=sides)
        for mode, sides in (("second_order", False), ("second_order, both_sides", True))
    }
    rows = []
    for z in (0.5, 1.5, 2.5):
        receiver = np.array([THICK + 5.0, 0.0, z])
        began = time.perf_counter()
        pressure = np.zeros(len(K), dtype=complex)
        for src in (source, source * MIRROR):
            for rec in (receiver, receiver * MIRROR):  # every path touches both at y = 0
                pressure += btm_second(near, far, src, rec, (0.0, 0.0), True, K)
        seconds = time.perf_counter() - began
        exact_db = level(pressure, float(np.linalg.norm(receiver - source)))
        row = {"receiver": receiver.tolist(), "exact_db": exact_db.tolist()}
        for mode, engine in engines.items():
            row[mode] = {
                kind: (engine.level(receiver, fade) - exact_db).tolist()
                for kind, fade in (("fade_minus_exact_db", True), ("plain_minus_exact_db", False))
            }
        row["btm_seconds"] = seconds
        rows.append(row)
    return rows


def _row(values) -> str:
    return " | ".join(f"{v:.2f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the numbers to this file")
    parser.add_argument("--step", type=float, default=STEP, help="metres between listeners")
    args = parser.parse_args()
    header = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ)
    rule = "|---|" + "---:|" * len(OCTAVE_BANDS_HZ)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            row = run(case, Path(tmp), args.step)
            results.append(row)
            print(f"\n{row['case']}: {row['listeners']} listeners {args.step * 100:g} cm apart")
            print(f"| (dB) | {header} |\n{rule}")
            for kind, label in (("fade", "with the fade"), ("plain", "without")):
                found = row[kind]
                print(f"| largest difference, {label} | {_row(found['largest_difference_db'])} |")
                print(f"| rms difference, {label} | {_row(found['rms_difference_db'])} |")
                print(f"| largest step, {label} | {_row(found['largest_step_db'])} |")
            print(f"| largest step, exact | {_row(row['exact_largest_step_db'])} |")
        thick = thick_barrier(Path(tmp))
        print(f"\nthick barrier {THICK:g} m wide, {BARRIER:g} m high, on the ground")
        print(f"| engine - exact (dB) | {header} |\n{rule}")
        for row in thick:
            print(f"| listener {row['receiver'][2]:g} m up: exact | {_row(row['exact_db'])} |")
            for mode in ("second_order", "second_order, both_sides"):
                for kind, label in (("fade", "with the fade"), ("plain", "without")):
                    found = row[mode][f"{kind}_minus_exact_db"]
                    print(f"| {mode}, {label} | {_row(found)} |")
        seconds = max(row["btm_seconds"] for row in thick)
        print(f"\n(BTM takes up to {seconds:.0f} s per listener)")
        results.append({"case": "thick barrier, second_order", "listeners": thick})
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
