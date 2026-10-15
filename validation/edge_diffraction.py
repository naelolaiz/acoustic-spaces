"""Check edge diffraction against the exact Biot-Tolstoy-Medwin (BTM) solution.

    python validation/edge_diffraction.py [--json results.json]

The BTM solution writes the sound diffracted by a rigid wedge as an integral along its
edge (Svensson, Fred and Vanderkooy, 1999): first order, the pressure at R from a source
at S whose direct sound is e^(-jkr)/r is

    -nu / (4 pi) * integral of beta(z) / (m l) e^(-jk (m + l)) dz

with m and l the distances from S and from R to the point z on the edge, nu = pi over
the wedge's open angle and beta the sum of four terms in the angles of S and R round the
edge. Second order, from S over one edge to a second and on to R,

    nu1 nu2 / (16 pi^2) * double integral of beta1 beta2 / (m l n) e^(-jk (m + l + n))

with l the distance between the two points on the edges, and half that when the edges
share a face, as the two corners of a thick wall do. The script integrates both along
edges 80 m long, sampled most densely where the shortest path touches them, each piece
integrated exactly for a phase that changes linearly along it, and tapered over the
last 12 m of each end so that the ends add nothing; it checks this against the UTD for a
half-plane, which is exact far from the edge, and the engine against it:

- a thin screen (a half-plane), first order: the engine's UTD and Maekawa's chart
  against BTM, which is exact for a half-plane;
- a thick barrier (two right-angled corners 0.3, 1 and 3 m apart, sharing the top face),
  second order: the engine against BTM, at two pairs of positions, one of them oblique;
- two thin screens in a row, second order;
- the level as the source rises through the plane of the barrier's top, where the path
  over the far corner alone appears, and as the listener does on the other side;
- the cost of the early stage with each mode, in a hall with alcoves.

Every comparison is of the energy in each octave band, averaged over the engine's eight
frequencies in the band, relative to the direct sound in free field over the same
distance. The engine's levels are given twice: with its paths and the terms of their
coefficients added in energy, and in the 63 and 125 Hz bands in pressure, as the
engine adds them there by default (``simulation.early.coherent_max_band_hz``).
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
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import EdgeDiffraction, band_frequencies, find_edges
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

C = 343.0
#: Half the length of the edges integrated over (m), and of the screens and barriers.
HALF_LENGTH = 40.0
#: The surfaces' extent in the engine's models (m): far enough for their other edges to
#: arrive after everything compared here.
EXTENT = 300.0
FREQUENCIES, _ = band_frequencies()
#: The bands in which the engine adds paths in pressure by default: 63 and 125 Hz.
COHERENT = 2
K = 2.0 * np.pi * FREQUENCIES.ravel() / C


# -- BTM ------------------------------------------------------------------------------


class Edge:
    """A straight edge through ``origin`` along ``direction``, with face 0 along
    ``along`` from it and the open side towards ``normal``, open over ``n`` pi."""

    def __init__(self, origin, direction, along, normal, n: float) -> None:
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.along = np.asarray(along, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.nu = 1.0 / n

    def frame(self, points):
        """Distance from the edge, angle round it from face 0, position along it."""
        v = np.asarray(points, dtype=float) - self.origin
        z = v @ self.direction
        w = v - np.multiply.outer(z, self.direction)
        theta = np.arctan2(w @ self.normal, w @ self.along)
        theta = np.where(theta < -1e-12, theta + 2.0 * np.pi, np.maximum(theta, 0.0))
        return np.linalg.norm(w, axis=-1), theta, z


def _samples(centre: float, du: float, taper: float = 0.3):
    """Points along an edge, z = centre + 0.05 sinh(u) for u in steps of ``du``, and
    their lengths, tapered to nothing over the last ``taper`` of each half."""
    top = math.asinh(HALF_LENGTH / 0.05)
    u = np.arange(-top, top + 0.5 * du, du)
    z = 0.05 * np.sinh(u)
    dz = 0.05 * np.cosh(u) * du
    x = np.clip((np.abs(z) / HALF_LENGTH - (1.0 - taper)) / taper, 0.0, 1.0)
    return centre + z, dz * np.cos(0.5 * np.pi * x) ** 2


def _beta(nu, argument, theta_s, theta_r):
    eta = np.arccosh(np.maximum(argument, 1.0))
    ch = np.cosh(nu * eta)
    out = 0.0
    for phi in (
        np.pi + theta_s + theta_r,
        np.pi + theta_s - theta_r,
        np.pi - theta_s + theta_r,
        np.pi - theta_s - theta_r,
    ):
        out = out + np.sin(nu * phi) / (ch - np.cos(nu * phi))
    return out


def btm_first(edge: Edge, source, receiver, k=K, du: float = 0.004) -> np.ndarray:
    """The diffracted pressure at ``receiver``, per wave number."""
    rs, ts, zs = edge.frame(source)
    rr, tr, zr = edge.frame(receiver)
    z, dz = _samples(zs + (zr - zs) * rs / (rs + rr), du)
    m, el = np.hypot(rs, z - zs), np.hypot(rr, z - zr)
    beta = _beta(edge.nu, (m * el + (z - zs) * (z - zr)) / (rs * rr), ts, tr)
    weight = -edge.nu / (4.0 * np.pi) * beta / (m * el) * dz
    slope = ((z - zs) / m + (z - zr) / el) * dz  # the path's change across each piece
    phase = np.exp(-1j * np.outer(k, m + el)) * np.sinc(np.outer(k, slope) / (2.0 * np.pi))
    return phase @ weight


def btm_second(
    e1: Edge, e2: Edge, source, receiver, centres, half: bool, k=K, du: float = 0.016
) -> np.ndarray:
    """The pressure at ``receiver`` over ``e1`` and then ``e2``; ``centres`` are where
    along each the shortest path touches them."""
    source, receiver = np.asarray(source, dtype=float), np.asarray(receiver, dtype=float)
    rs, ts, zs = e1.frame(source)
    rr, tr, zr = e2.frame(receiver)
    z1, d1 = _samples(centres[0], du)
    z2, d2 = _samples(centres[1], du)
    p1 = e1.origin + np.outer(z1, e1.direction)
    p2 = e2.origin + np.outer(z2, e2.direction)
    m = np.linalg.norm(p1 - source, axis=1)
    n = np.linalg.norm(p2 - receiver, axis=1)
    r21, t21, z21 = e1.frame(p2)  # the second edge's points round the first
    r12, t12, z12 = e2.frame(p1)
    factor = e1.nu * e2.nu / (16.0 * np.pi**2) * (0.5 if half else 1.0)
    out = np.zeros(len(k), dtype=complex)
    for begin in range(0, len(z1), 128):
        rows = slice(begin, begin + 128)
        gap = p2[None, :, :] - p1[rows, None, :]
        el = np.linalg.norm(gap, axis=2)
        a1 = m[rows, None] * el + (z1[rows, None] - zs) * (z1[rows, None] - z21[None, :])
        b1 = _beta(e1.nu, a1 / (rs * r21[None, :]), ts, t21[None, :])
        a2 = el * n[None, :] + (z2[None, :] - z12[rows, None]) * (z2[None, :] - zr)
        b2 = _beta(e2.nu, a2 / (r12[rows, None] * rr), t12[rows, None], tr)
        weight = factor * b1 * b2 / (m[rows, None] * el * n[None, :])
        weight *= d1[rows, None] * d2[None, :]
        v = gap / el[..., None]
        s1 = ((p1[rows] - source) @ e1.direction / m[rows])[:, None] - v @ e1.direction
        s2 = v @ e2.direction + ((p2 - receiver) @ e2.direction / n)[None, :]
        s1 = (s1 * d1[rows, None]).ravel() / (2.0 * np.pi)
        s2 = (s2 * d2[None, :]).ravel() / (2.0 * np.pi)
        path = (m[rows, None] + el + n[None, :]).ravel()
        phase = np.exp(-1j * np.outer(k, path))
        phase *= np.sinc(np.outer(k, s1)) * np.sinc(np.outer(k, s2))
        out += phase @ weight.ravel()
    return out


def band_level(pressure: np.ndarray, distance: float) -> np.ndarray:
    """Octave band energies of ``pressure`` (per wave number) in dB re the direct sound
    over ``distance`` in free field."""
    energy = (np.abs(pressure) ** 2).reshape(FREQUENCIES.shape).mean(axis=1)
    return 10.0 * np.log10(energy * distance**2)


# -- the engine -----------------------------------------------------------------------


def _room(folder: Path, name: str, quads: list, source, receiver) -> Room:
    lines, faces = [], []
    for quad in quads:
        base = len(lines)
        lines += [f"v {x} {y} {z}" for x, y, z in quad]
        faces.append("f " + " ".join(str(base + i + 1) for i in range(4)))
    obj = folder / f"{name}.obj"
    obj.write_text("\n".join(lines + ["g surfaces"] + faces) + "\n")
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": list(map(float, source))}],
                "listeners": [{"name": "l", "position": list(map(float, receiver))}],
            }
        )
    )


def screen_quads(x: float = 0.0, top: float = 0.0) -> list:
    """A screen in the plane x = ``x`` with its top edge along y at z = ``top``."""
    return [[(x, -EXTENT, -EXTENT), (x, EXTENT, -EXTENT), (x, EXTENT, top), (x, -EXTENT, top)]]


def barrier_quads(width: float) -> list:
    """A block from x = 0 to ``width`` with its top at z = 0, open underneath."""
    a, b, h = EXTENT, width, -EXTENT
    return [
        [(0, -a, 0), (b, -a, 0), (b, a, 0), (0, a, 0)],
        [(0, -a, h), (0, -a, 0), (0, a, 0), (0, a, h)],
        [(b, -a, h), (b, a, h), (b, a, 0), (b, -a, 0)],
        [(0, -a, h), (b, -a, h), (b, -a, 0), (0, -a, 0)],
        [(0, a, h), (0, a, 0), (b, a, 0), (b, a, h)],
    ]


def engine(room: Room, mode: str, model: TraceModel | None = None, edges=None, coherent: int = 0):
    """The engine's paths from the room's source to its listener, with their total
    level per band in dB re the direct sound in free field; in the lowest ``coherent``
    bands the paths and the terms of their coefficients add in pressure."""
    model = TraceModel.from_room(room) if model is None else model
    source = np.array(room.scene.sources[0].position)
    receiver = np.array(room.scene.listeners[0].position)
    air = np.zeros(len(OCTAVE_BANDS_HZ))
    c = room.speed_of_sound()
    paths = early_stage.find_paths(
        model, source, receiver, [], speed_of_sound=c, air=air, max_delay_s=0.5, coherent=coherent
    )
    if mode != "none":
        paths = EdgeDiffraction(
            model, source, speed_of_sound=c, air=air, mode=mode, edges=edges, coherent=coherent
        ).apply(receiver, paths, 0.5)
    squared = float(np.sum((receiver - source) ** 2))
    with np.errstate(divide="ignore"):
        level = 10.0 * np.log10(total_energy(paths) * 4.0 * np.pi * squared)
    return paths, level


# -- the checks -----------------------------------------------------------------------


def half_plane_reference() -> list[dict]:
    """BTM for a half-plane against the UTD, which is exact far from the edge."""
    from acoustic_spaces.engine.diffraction import boundary_angles, wedge_prefactor, wedge_terms

    edge = Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
    rows = []
    for source, receiver in (((-5, 0, -2), (5, 0, -1)), ((-5, 0.7, -2), (3, -1.2, -0.3))):
        source, receiver = np.array(source, float), np.array(receiver, float)
        rs, ts, zs = edge.frame(source)
        rr, tr, zr = edge.frame(receiver)
        zk = zs + (zr - zs) * rs / (rs + rr)
        r1, r2 = math.hypot(rs, zk - zs), math.hypot(rr, zr - zk)
        sin_beta = rs / r1
        lit = boundary_angles(np.array(2.0), np.array(tr), np.array(ts)) > 0
        terms = wedge_terms(2.0, tr, ts, K * r1 * r2 / (r1 + r2) * sin_beta**2, lit)
        utd = wedge_prefactor(2.0, K, sin_beta) * terms.sum(axis=-1)
        utd *= np.exp(-1j * K * (r1 + r2)) / math.sqrt(r1 * r2 * (r1 + r2))
        btm = btm_first(edge, source, receiver)
        distance = float(np.linalg.norm(receiver - source))
        rows.append(
            {
                "source": source.tolist(),
                "receiver": receiver.tolist(),
                "btm_db": band_level(btm, distance).tolist(),
                "utd_minus_btm_db": (
                    band_level(utd, distance) - band_level(btm, distance)
                ).tolist(),
            }
        )
    return rows


def thin_screen(folder: Path) -> list[dict]:
    """First order over a half-plane: the engine and Maekawa against BTM."""
    edge = Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
    rows = []
    for source, receiver in (
        ((-5.0, 0.0, -2.0), (5.0, 0.0, -1.0)),
        ((-2.0, 0.0, -0.2), (6.0, 0.0, -3.0)),
        ((-3.0, -1.0, -1.0), (3.0, 2.0, -0.5)),
    ):
        source, receiver = np.array(source), np.array(receiver)
        distance = float(np.linalg.norm(receiver - source))
        btm = band_level(btm_first(edge, source, receiver), distance)
        room = _room(folder, "screen", screen_quads(), source, receiver)
        _, level = engine(room, "direct")
        _, pressure = engine(room, "direct", coherent=COHERENT)
        point = np.array([0.0, _cross_y(source, receiver), 0.0])
        detour = float(np.linalg.norm(source - point) + np.linalg.norm(receiver - point))
        detour -= distance
        fresnel = 2.0 * detour * np.array(OCTAVE_BANDS_HZ) / C
        maekawa = -10.0 * np.log10(3.0 + 20.0 * fresnel)
        rows.append(
            {
                "source": source.tolist(),
                "receiver": receiver.tolist(),
                "btm_db": btm.tolist(),
                "engine_minus_btm_db": (level - btm).tolist(),
                "in_pressure_minus_btm_db": (pressure - btm).tolist(),
                "maekawa_minus_btm_db": (maekawa - btm).tolist(),
            }
        )
    return rows


def _cross_y(source, receiver) -> float:
    """Where along the screen's top edge (the y axis) the shortest path crosses it."""
    rs, rr = math.hypot(source[0], source[2]), math.hypot(receiver[0], receiver[2])
    return source[1] + (receiver[1] - source[1]) * rs / (rs + rr)


def _double_path(paths) -> tuple[np.ndarray, np.ndarray]:
    """The points on the two edges of the one path over two edges."""
    double = [p for p in paths.points if len(p) == 4]
    assert len(double) == 1, f"{len(double)} paths over two edges"
    return double[0][1], double[0][2]


def second_order(folder: Path) -> list[dict]:
    """Over a thick barrier and over two screens: the engine against BTM."""
    cases = []
    for width in (0.3, 1.0, 3.0):
        e1 = Edge([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], 1.5)
        e2 = Edge([width, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1], 1.5)
        for source, receiver in (
            ((-3.0, 0.0, -1.0), (width + 3.0, 0.0, -1.5)),
            ((-6.0, -2.0, -0.3), (width + 2.0, 3.0, -4.0)),
        ):
            cases.append(
                (
                    f"thick barrier, {width:g} m",
                    barrier_quads(width),
                    e1,
                    e2,
                    True,
                    source,
                    receiver,
                )
            )
    for gap, rise in ((1.0, 0.0), (2.0, 0.4)):
        e1 = Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
        e2 = Edge([gap, 0, rise], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
        name = f"two screens {gap:g} m apart" + (f", the second {rise:g} m higher" if rise else "")
        quads = screen_quads() + screen_quads(gap, rise)
        cases.append((name, quads, e1, e2, False, (-3.0, 0.0, -1.0), (gap + 3.0, 0.0, -1.5)))
    rows = []
    for name, quads, e1, e2, half, source, receiver in cases:
        source, receiver = np.array(source), np.array(receiver)
        room = _room(folder, "second", quads, source, receiver)
        paths, level = engine(room, "second_order")
        _, pressure = engine(room, "second_order", coherent=COHERENT)
        p1, p2 = _double_path(paths)
        centres = (float((p1 - e1.origin) @ e1.direction), float((p2 - e2.origin) @ e2.direction))
        began = time.perf_counter()
        btm = btm_second(e1, e2, source, receiver, centres, half)
        seconds = time.perf_counter() - began
        distance = float(np.linalg.norm(receiver - source))
        reference = band_level(btm, distance)
        rows.append(
            {
                "case": name,
                "source": source.tolist(),
                "receiver": receiver.tolist(),
                "btm_db": reference.tolist(),
                "engine_minus_btm_db": (level - reference).tolist(),
                "in_pressure_minus_btm_db": (pressure - reference).tolist(),
                "btm_seconds": seconds,
            }
        )
    return rows


def crossing(folder: Path) -> dict:
    """Levels as the source rises through the plane of a 1 m thick barrier's top on one
    side, and as the listener does on the other: the largest step between neighbouring
    positions 1 mm apart, with and without paths over two edges."""
    out = {}
    steps = np.linspace(-0.01, 0.01, 21)
    quads = barrier_quads(1.0)
    for who in ("source", "listener"):
        found = {}
        for mode, coherent in (
            ("first_order", 0),
            ("second_order", 0),
            ("second_order, in pressure", COHERENT),
        ):
            levels = []
            for dz in steps:
                source = np.array([-3.0, 0.0, -1.0 if who == "listener" else dz])
                receiver = np.array([4.0, 0.0, dz if who == "listener" else -1.5])
                room = _room(folder, "crossing", quads, source, receiver)
                _, level = engine(room, mode.split(",")[0], coherent=coherent)
                levels.append(level)
            levels = np.array(levels)
            with np.errstate(invalid="ignore"):
                jump = np.abs(np.diff(levels, axis=0))
            found[mode] = {
                "levels_db": levels.tolist(),
                "largest_step_db": np.where(np.isfinite(jump), jump, np.inf).max(axis=0).tolist(),
            }
        out[who] = found
    return out


def alcove_hall() -> Room:
    """A 30 x 20 x 10 m hall with four 3 m alcoves, 2 m deep, along each long wall:
    16 vertical edges that diffract, with the source and the listener in the first and
    the last alcove of one side, out of each other's sight."""
    plan = [(0.0, 0.0)]
    for side, y, inward in ((1, 0.0, -2.0), (-1, 20.0, 2.0)):
        xs = [4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5]
        xs = xs if side > 0 else xs[::-1]
        if side < 0:
            plan += [(30.0, 0.0), (30.0, 20.0)]
        for i in range(0, 8, 2):
            a, b = xs[i], xs[i + 1]
            plan += [(a, y), (a, y + inward), (b, y + inward), (b, y)]
    plan.append((0.0, 20.0))
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"floor_plan": plan, "height": 10.0},
                "materials": {"wall": {"absorption": [0.1] * 8, "scattering": 0.1}},
                "default_material": "wall",
                "sources": [{"name": "s", "position": [5.0, -1.5, 1.5]}],
                "listeners": [{"name": "l", "position": [24.0, -1.5, 1.2]}],
            }
        )
    )


def timing() -> dict:
    """The cost of the early stage (reflections up to the default order, half a second)
    in the hall with alcoves, with each diffraction mode."""
    room = alcove_hall()
    out = {"edges": len(find_edges(TraceModel.from_room(room)))}
    for mode in ("none", "first_order", "second_order"):
        early_paths(room, listener="l", diffraction=mode)  # a first run, without the one-off costs
        began = time.perf_counter()
        for _ in range(3):
            paths = early_paths(room, listener="l", diffraction=mode)
        out[mode + "_ms"] = 1000.0 * (time.perf_counter() - began) / 3
        out[mode + "_paths"] = len(paths)
    return out


def _row(values) -> str:
    return " | ".join(f"{v:+.2f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the numbers to this file")
    args = parser.parse_args()
    header = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ)
    rule = "|---|" + "---:|" * len(OCTAVE_BANDS_HZ)
    results: dict = {}

    results["half_plane"] = half_plane_reference()
    print(f"| BTM against the UTD for a half-plane (dB) | {header} |\n{rule}")
    for row in results["half_plane"]:
        print(f"| {row['source']} to {row['receiver']} | {_row(row['utd_minus_btm_db'])} |")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        results["thin_screen"] = thin_screen(folder)
        print(f"\n| thin screen, level re free field (dB) | {header} |\n{rule}")
        for row in results["thin_screen"]:
            where = f"{row['source']} to {row['receiver']}"
            print(f"| {where}: BTM | {_row(row['btm_db'])} |")
            print(f"| engine - BTM | {_row(row['engine_minus_btm_db'])} |")
            print(f"| engine in pressure - BTM | {_row(row['in_pressure_minus_btm_db'])} |")
            print(f"| Maekawa - BTM | {_row(row['maekawa_minus_btm_db'])} |")

        results["second_order"] = second_order(folder)
        print(f"\n| over two edges, level re free field (dB) | {header} |\n{rule}")
        for row in results["second_order"]:
            where = f"{row['case']}, {row['source']} to {row['receiver']}"
            print(f"| {where}: BTM | {_row(row['btm_db'])} |")
            print(f"| engine - BTM | {_row(row['engine_minus_btm_db'])} |")
            print(f"| engine in pressure - BTM | {_row(row['in_pressure_minus_btm_db'])} |")
        seconds = max(row["btm_seconds"] for row in results["second_order"])
        print(f"\n(BTM takes up to {seconds:.0f} s per case)")

        results["crossing"] = crossing(folder)
        print(f"\n| largest step over 1 mm (dB) | {header} |\n{rule}")
        for who, found in results["crossing"].items():
            for mode, row in found.items():
                print(f"| {who} through the top's plane, {mode} | {_row(row['largest_step_db'])} |")

    results["timing"] = timing()
    t = results["timing"]
    print(f"\nhall with alcoves, {t['edges']} edges, per source and listener:")
    for mode in ("none", "first_order", "second_order"):
        print(f"- {mode}: {t[mode + '_ms']:.0f} ms, {t[mode + '_paths']} paths")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
