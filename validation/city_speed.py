"""Time whole cities outdoors, and check that the edge diffraction's culling of the
reflections that cannot happen leaves every result as it was.

    python validation/city_speed.py [--sizes 10 20 30 60] [--check 4] [--threads 0]
                                    [--json results.json]

A city of n by n blocks 32 m apart, each a building of random width and depth (12 to
20 m) and height (6 to 30 m), with grass in the 12 m streets and 20 m of ground round
it; one source in a street crossing in the middle, 1 m up, and two listeners, three
blocks along the street 1.5 m up and five blocks across it 4 m up (fewer in small
cities). The simulation is
the outdoor default: early reflections to the third order traced from the source, the
edges diffracting the direct sound and the first-order reflections, the late part
traced with 50,000 rays per band.

Edge diffraction looks at every pair of an edge and a plane the source or a listener
faces, for a path that reflects off the plane before or after bending over the edge:
those pairs grow with the square of the buildings. Most cannot carry a path, as the
plane's faces lie nowhere near where a ray from the edge would meet the plane, and are
left out before the ray tests (:func:`acoustic_spaces.engine.diffraction._reflecting_pairs`).
For each size the script prints the time to build the room, to run the simulation and
how much of it went to edge diffraction (with the terms that make the other specular
paths fade at the edges), the share of pairs tested, and the peak memory where the
system reports it.

The check runs a smaller city with buildings at random angles (``--check`` blocks a
side), with paths reflecting on both sides of an edge, from two sources to two listeners,
once as the engine runs and once with every pair tested, and compares every path, its
energy and its spectrum bit for bit, on one thread and on several.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import diffraction, edge_fade
from acoustic_spaces.engine.simulate import early_paths

BLOCK, STREET = 20.0, 12.0


def city(n: int, seed: int = 1, turned: bool = False, both_sides: bool = False) -> Room:
    """The city of n by n blocks; with ``turned`` each building stands at a random angle
    round its block's middle, so that no two walls share a plane."""
    rng = np.random.default_rng(seed)
    pitch = BLOCK + STREET
    buildings = []
    for i in range(n):
        for j in range(n):
            width, depth = BLOCK * rng.uniform(0.6, 1.0, 2)
            height = float(rng.uniform(6.0, 30.0))
            corners = np.array([[0, 0], [width, 0], [width, depth], [0, depth]])
            if turned:
                angle = rng.uniform(0.0, np.pi)
                turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                corners = (corners - [width / 2, depth / 2]) @ turn.T + BLOCK / 2
            footprint = corners + [i * pitch, j * pitch]
            buildings.append(
                {"name": f"b{i}_{j}", "footprint": footprint.tolist(), "height": height}
            )
    mid = (n // 2) * pitch - STREET / 2
    room_left = n - n // 2 - 1  # blocks between the middle and the city's edge
    along, across = min(3, room_left), min(5, room_left)
    early = {"diffraction": "first_order"}
    if both_sides:
        early["diffraction_reflections"] = "both_sides"
    scene = {
        "geometry": {"ground": {"margin": 20.0}, "buildings": buildings},
        "materials": {
            "lawn": {"impedance": "grass"},
            "wall": {"absorption": [0.05] * 8, "scattering": 0.1},
        },
        "assignments": {"ground": "lawn", "buildings": "wall"},
        "sources": [
            {"name": "street", "position": [mid, mid, 1.0]},
            {"name": "roof", "position": [mid + pitch, mid + 4.0, 32.0]},
        ],
        "listeners": [
            {"name": "along", "position": [mid + along * pitch, mid, 1.5]},
            {"name": "across", "position": [mid, mid + across * pitch + 3.0, 4.0]},
        ],
        "simulation": {"seed": 1, "early": early},
    }
    return Room.from_scene(Scene.model_validate(scene))


class _Counted:
    """The pairs of an edge and a plane tested, all there were, and the time edge
    diffraction took (finding the source's sides of the edges, then each listener's,
    and the terms that make the other specular paths fade at the edges)."""

    def __init__(self) -> None:
        self.kept = self.all = 0
        self.seconds = 0.0
        self._culled = diffraction._reflecting_pairs
        self._init = diffraction.EdgeDiffraction.__init__
        self._apply = diffraction.EdgeDiffraction.apply
        self._terms = edge_fade.EdgeFade.terms
        self._join = edge_fade.EdgeFade.join

    def __enter__(self) -> _Counted:
        def counting(model, edges, bounds, end, planes, among, offset, threads=0):
            rows, cols = self._culled(model, edges, bounds, end, planes, among, offset, threads)
            self.kept += len(rows)
            self.all += len(among) * len(planes)
            return rows, cols

        def timed(method):
            def run(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return method(*args, **kwargs)
                finally:
                    self.seconds += time.perf_counter() - started

            return run

        diffraction._reflecting_pairs = counting
        diffraction.EdgeDiffraction.__init__ = timed(self._init)
        diffraction.EdgeDiffraction.apply = timed(self._apply)
        edge_fade.EdgeFade.terms = timed(self._terms)
        edge_fade.EdgeFade.join = timed(self._join)
        return self

    def __exit__(self, *exc) -> None:
        diffraction._reflecting_pairs = self._culled
        diffraction.EdgeDiffraction.__init__ = self._init
        diffraction.EdgeDiffraction.apply = self._apply
        edge_fade.EdgeFade.terms = self._terms
        edge_fade.EdgeFade.join = self._join


def _peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if platform.system() == "Darwin" else peak / 2**10


def time_city(n: int, threads: int) -> dict:
    started = time.perf_counter()
    room = city(n)
    built = time.perf_counter() - started
    with _Counted() as counted:
        started = time.perf_counter()
        result = simulate(room, sources=["street"], threads=threads)
        total = time.perf_counter() - started
    details = next(iter(result.details.values()))
    model_faces = len(room.mesh.faces)
    return {
        "blocks": n,
        "buildings": n * n,
        "faces": model_faces,
        "edges": int(details.get("edges", 0)),
        "build_s": built,
        "simulate_s": total,
        "early_s": float(details["early_s"]),
        "diffraction_s": counted.seconds,
        "late_s": float(details["late_s"]),
        "pairs_tested": counted.kept,
        "pairs": counted.all,
        "peak_memory_mb": _peak_memory_mb(),
    }


def _all_reflecting(model, edges, bounds, end, planes, among, offset, threads=0):
    rows, cols = np.nonzero(np.ones((len(among), len(planes)), dtype=bool))
    return rows.astype(np.int64), cols.astype(np.int64)


def _all_reaching(model, edges, bounds, end, planes, among, sides, *rest):
    rows, cols = np.nonzero(np.ones((len(sides[0]), len(planes)), dtype=bool))
    return rows.astype(np.int64), cols.astype(np.int64)


def _differences(a, b) -> list[str]:
    out = []
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            same = (
                x is not None
                and y is not None
                and x.shape == y.shape
                and np.array_equal(x, y, equal_nan=True)
            )
            if not same:
                out.append(field.name)
    return out


def check(n: int) -> dict:
    """Early paths with the culling and with every pair tested, bit for bit."""
    room = city(n, seed=7, turned=True, both_sides=True)
    culled, reaching = diffraction._reflecting_pairs, diffraction._reaching_pairs
    cases, differing, paths, fast_s, slow_s = 0, [], 0, 0.0, 0.0
    try:
        for source in ("street", "roof"):
            for listener in ("along", "across"):
                started = time.perf_counter()
                fast = early_paths(room, source, listener, threads=1)
                fast_s += time.perf_counter() - started
                many = early_paths(room, source, listener, threads=0)
                diffraction._reflecting_pairs = _all_reflecting
                diffraction._reaching_pairs = _all_reaching
                started = time.perf_counter()
                slow = early_paths(room, source, listener, threads=1)
                slow_s += time.perf_counter() - started
                diffraction._reflecting_pairs, diffraction._reaching_pairs = culled, reaching
                cases += 1
                paths += int(fast.diffracted.sum())
                for name in _differences(fast, slow) + _differences(many, slow):
                    differing.append(f"{source} to {listener}: {name}")
    finally:
        diffraction._reflecting_pairs, diffraction._reaching_pairs = culled, reaching
    return {
        "blocks": n,
        "cases": cases,
        "diffracted_paths": paths,
        "differing": differing,
        "culled_s": fast_s,
        "every_pair_s": slow_s,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10, 20, 30], help="blocks a side of each city"
    )
    parser.add_argument("--check", type=int, default=4, help="blocks a side of the check (0: none)")
    parser.add_argument("--threads", type=int, default=0, help="worker threads (0: every core)")
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)
    print(f"{platform.machine()}, {os.cpu_count()} cores, threads {args.threads or 'all'}\n")
    out: dict = {"cities": []}
    if args.check:
        print(f"Culling against every pair tested, {args.check} by {args.check} blocks turned:")
        found = check(args.check)
        out["check"] = found
        verdict = "bit-identical" if not found["differing"] else "DIFFERENT"
        print(
            f"  {found['cases']} source and listener pairs, {found['diffracted_paths']} "
            f"diffracted paths: {verdict}; early paths {found['culled_s']:.1f} s culled, "
            f"{found['every_pair_s']:.1f} s with every pair tested"
        )
        for line in found["differing"]:
            print("  differs:", line)
        print()
    header = (
        f"{'buildings':>9} {'faces':>7} {'edges':>7} {'build':>7} {'simulate':>9} "
        f"{'diffraction':>12} {'late':>6} {'pairs tested':>13} {'memory':>8}"
    )
    print(header)
    for n in sorted(args.sizes):
        row = time_city(n, args.threads)
        out["cities"].append(row)
        share = row["pairs_tested"] / max(row["pairs"], 1)
        memory = row["peak_memory_mb"]
        print(
            f"{row['buildings']:>9} {row['faces']:>7} {row['edges']:>7} "
            f"{row['build_s']:>6.1f}s {row['simulate_s']:>8.1f}s {row['diffraction_s']:>11.1f}s "
            f"{row['late_s']:>5.1f}s {100 * share:>12.1f}% "
            + (f"{memory:>6.0f}MB" if memory is not None else f"{'':>8}"),
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
    return 0 if not out.get("check", {}).get("differing") else 1


if __name__ == "__main__":
    raise SystemExit(main())
