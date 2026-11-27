"""Check edge diffraction against the exact Biot-Tolstoy-Medwin (BTM) solution.

    python validation/edge_diffraction.py [--json results.json] [--close-screens | --ceiling]
    python validation/edge_diffraction.py --wave-references

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
  and the same block absorbing 0.02, against BTM for the rigid one and the Weyl-van der
  Pol field's loss along its top (|F(w)|, by hand);
- two thin screens in a row, second order;
- two thin screens 0.5 to 4 m apart, second order: BTM to the second order and the engine,
  with ray tracing (the default) and with image sources, against a wave calculation
  that holds every order of diffraction and reflection (``validation/rows.py``'s: a
  two-dimensional run by finite differences on a 2 cm grid, turned into a point
  source's), whose levels are stored in ``close_screens_wave.json`` (63 to 500 Hz;
  ``--wave-references`` computes them again, a few minutes a case on one core, and
  ``--close-screens`` runs this check alone);
- a thin ceiling 1 m over two thin screens 2 m apart, whole, ending near the point where
  the path over both tops reflects off it, or a strip round that point, second order:
  the engine with both methods, by default and with every band in pressure, against
  the same wave calculation (``ceiling_wave.json``; ``--ceiling`` runs it alone);
- a thin strip, second order: the source and the listener in its shadow on either
  side, the paths over both its edges along each of its faces, against BTM;
- a thin strip standing free, 1, 2 and 4 m high: the listener crosses the line of sight
  over its top, where the direct sound is found over the top past its foot, against
  BTM over both edges and over both along each face; the engine with each mode, and
  the edge diffraction alone;
- a thin screen, a right-angled corner and a wedge open over 225 degrees, first order,
  with the listener on the shadow boundary and 10.5 degrees into the shadow: the whole
  early stage (``early_paths``) with every band added in energy and as by default,
  against BTM;
- the corner of a thin screen, first order: the listener walks past where the direct
  sound comes round one edge while the path over the other still touches it, against
  BTM over both edges, each ending at the corner; right-angled, and at 45, 60, 120,
  135.5, 150 and 170 degrees;
- a thin screen whose top bends by 2 to 14 degrees, away from the source and towards
  it, first order: the listener rises through the top's shadow boundary beside the
  bend and walks past it deep in the shadow, against BTM over both tops, each ending at
  the bend;
- the corners of solid prisms, where three edges meet (a block's, corners of 135 and 45
  degrees in plan, a thick barrier's end, buildings on a hard ground): the listener
  walks past where the paths over the edges reach their ends, against BTM over each
  edge from end to end;
- the level as the source rises through the plane of the barrier's top, where the path
  over the far corner alone appears, and as the listener does on the other side;
- the source and the listener in sight of each other just over a block's top, 10 and
  1 m wide: the whole early stage with paths over two edges and without, against BTM
  with the geometric waves, where the paths add in pressure; and how much the level
  differs between positions within a millimetre of the plane of the top, on both sides;
- two thin screens in a row at grazing incidence, against the closed form for two knife
  edges at high frequencies;
- the limits that stay: the listener behind a block with the source above the plane of
  its top, with paths over one edge alone, against BTM; screens and flat tops in a row
  with the source and the listener just above their tops, exactly level with them and
  just below; and the step in the bands that add in energy where the source or the
  listener crosses the plane of a block's top;
- the cost of the early stage with each mode, in a hall with alcoves.

The engine's levels are a simulation's with the scene's settings and the diffraction
mode each check names (``early_paths`` where named): the specular paths, the edge
diffraction and the terms that make paths fade at edges. Every comparison is of the
energy in each octave band, averaged over the engine's eight frequencies in the band
(BTM's at a shadow boundary, past a screen's corners of other angles, past the corners
of prisms and at the line of sight over a free strip over 96), relative to the direct
sound in free field over the same distance. Where a check says so, the engine's levels
are given twice: with its paths and the terms of their coefficients added in energy,
and in the 63 and 125 Hz bands in pressure, as the engine adds them there by default
(``simulation.early.coherent_max_band_hz``); the walks past corners and over a free
strip have every band in pressure. BTM takes the engine's speed of sound in the
scenes' air, at 20 °C.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene
from acoustic_spaces.air import speed_of_sound
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import band_frequencies, diffract, find_edges
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

#: The engine's speed of sound (m/s) in the scenes here, whose air is at the default 20 °C.
C = speed_of_sound(20.0)
#: Half the length of the edges integrated over (m), and of the screens and barriers.
HALF_LENGTH = 40.0
#: The surfaces' extent in the engine's models (m): far enough for their other edges to
#: arrive after everything compared here.
EXTENT = 300.0
FREQUENCIES, _ = band_frequencies()
#: The bands in which the engine adds paths in pressure by default: 63 and 125 Hz.
COHERENT = 2
#: The absorption of the absorbing blocks over two edges.
ABSORBING = 0.02
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


def _samples(centre: float, du: float, taper: float = 0.3, half: float = HALF_LENGTH):
    """Points along an edge, z = centre + 0.05 sinh(u) for u in steps of ``du`` up to
    ``half`` either way, and their lengths, tapered to nothing over the last ``taper`` of
    each half."""
    top = math.asinh(half / 0.05)
    u = np.arange(-top, top + 0.5 * du, du)
    z = 0.05 * np.sinh(u)
    dz = 0.05 * np.cosh(u) * du
    x = np.clip((np.abs(z) / half - (1.0 - taper)) / taper, 0.0, 1.0)
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


def btm_first(
    edge: Edge,
    source,
    receiver,
    k=K,
    du: float = 0.004,
    ends=(-np.inf, np.inf),
    half_length: float = HALF_LENGTH,
) -> np.ndarray:
    """The diffracted pressure at ``receiver``, per wave number, integrated over
    ``half_length`` either way of where the shortest path touches the edge in steps of
    ``du`` (see :func:`_samples`); over the part of the edge between ``ends`` (positions
    along it from its origin), where it ends there."""
    rs, ts, zs = edge.frame(source)
    rr, tr, zr = edge.frame(receiver)
    z, dz = _samples(zs + (zr - zs) * rs / (rs + rr), du, half=half_length)
    if np.isfinite(ends).any():  # each piece cut to the ends: past them, nothing
        lower, upper = np.maximum(z - dz / 2, ends[0]), np.minimum(z + dz / 2, ends[1])
        keep = upper > lower
        z, dz = 0.5 * (lower + upper)[keep], (upper - lower)[keep]
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
    """Octave band energies of ``pressure`` (per wave number, band by band, as many in
    each) in dB re the direct sound over ``distance`` in free field."""
    energy = (np.abs(pressure) ** 2).reshape(len(FREQUENCIES), -1).mean(axis=1)
    return 10.0 * np.log10(energy * distance**2)


# -- the engine -----------------------------------------------------------------------


def _room(
    folder: Path, name: str, quads: list, source, receiver, simulation=None, absorption=0.0
) -> Room:
    """A room of rigid ``quads`` (or of ``absorption`` in every band) with one source and
    one listener, with the scene's ``simulation`` settings where given."""
    lines, faces = [], []
    for quad in quads:
        base = len(lines)
        lines += [f"v {x} {y} {z}" for x, y, z in quad]
        faces.append("f " + " ".join(str(base + i + 1) for i in range(4)))
    obj = folder / f"{name}.obj"
    obj.write_text("\n".join(lines + ["g surfaces"] + faces) + "\n")
    scene = {
        "geometry": {"file": str(obj)},
        "materials": {"hard": {"absorption": [absorption] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": list(map(float, source))}],
        "listeners": [{"name": "l", "position": list(map(float, receiver))}],
    }
    if simulation is not None:
        scene["simulation"] = simulation
    return Room.from_scene(Scene.model_validate(scene))


def screen_quads(x: float = 0.0, top: float = 0.0) -> list:
    """A screen in the plane x = ``x`` with its top edge along y at z = ``top``."""
    return [[(x, -EXTENT, -EXTENT), (x, EXTENT, -EXTENT), (x, EXTENT, top), (x, -EXTENT, top)]]


def barrier_quads(width: float, run: float = 0) -> list:
    """A block from x = 0 to ``width`` with its top at z = 0, open underneath, its front
    face falling from the top's edge at x = 0 to z = -EXTENT at x = -``run``: upright for
    ``run`` 0, and at 45 degrees for ``run`` EXTENT, that edge then open over 225
    degrees."""
    a, b, h, r = EXTENT, width, -EXTENT, -run
    return [
        [(0, -a, 0), (b, -a, 0), (b, a, 0), (0, a, 0)],
        [(r, -a, h), (0, -a, 0), (0, a, 0), (r, a, h)],
        [(b, -a, h), (b, a, h), (b, a, 0), (b, -a, 0)],
        [(r, -a, h), (b, -a, h), (b, -a, 0), (0, -a, 0)],
        [(r, a, h), (0, a, 0), (b, a, 0), (b, a, h)],
    ]


def engine(
    room: Room,
    mode: str,
    model: TraceModel | None = None,
    edges=None,
    coherent: int = 0,
    fade: bool = True,
):
    """The engine's paths from the room's source to its listener, with their total
    level per band in dB re the direct sound in free field; in the lowest ``coherent``
    bands the paths and the terms of their coefficients add in pressure. As a
    simulation has them with the scene's early-stage settings and diffraction in
    ``mode``, without the air: the specular paths of the sequences of reflections it
    tests (by default those its rays find, with the planes of the edges' faces), the
    edge diffraction and the terms that make paths fade at edges. Without ``fade``, the
    direct sound and the edge diffraction alone."""
    model = TraceModel.from_room(room) if model is None else model
    source = np.array(room.scene.sources[0].position)
    receiver = np.array(room.scene.listeners[0].position)
    air = np.zeros(len(OCTAVE_BANDS_HZ))
    c = room.speed_of_sound()
    candidates = []
    if fade:
        early = room.scene.simulation.early
        simulate = importlib.import_module("acoustic_spaces.engine.simulate")
        candidates = simulate._candidates(
            model,
            source,
            early.method,
            early.max_order,
            early.rays,
            room.scene.simulation.performance.threads,
            simulate._Edges(model),
            mode,
        )
    paths = early_stage.find_paths(
        model,
        source,
        receiver,
        candidates,
        speed_of_sound=c,
        air=air,
        max_delay_s=0.5,
        coherent=coherent,
    )
    if mode != "none":
        paths = diffract(
            model,
            source,
            receiver,
            paths,
            speed_of_sound=c,
            air=air,
            max_delay_s=0.5,
            mode=mode,
            edges=edges,
            coherent=coherent,
            candidates=candidates if fade else None,
        )
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
    """The points on the two edges of the one path over two edges (of the edge
    diffraction alone: products of the fade's terms run over two edges too)."""
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
        _, level = engine(room, "second_order")
        _, pressure = engine(room, "second_order", coherent=COHERENT)
        p1, p2 = _double_path(engine(room, "second_order", fade=False)[0])
        centres = (float((p1 - e1.origin) @ e1.direction), float((p2 - e2.origin) @ e2.direction))
        began = time.perf_counter()
        btm = btm_second(e1, e2, source, receiver, centres, half)
        seconds = time.perf_counter() - began
        distance = float(np.linalg.norm(receiver - source))
        reference = band_level(btm, distance)
        row = {
            "case": name,
            "source": source.tolist(),
            "receiver": receiver.tolist(),
            "btm_db": reference.tolist(),
            "engine_minus_btm_db": (level - reference).tolist(),
            "in_pressure_minus_btm_db": (pressure - reference).tolist(),
            "btm_seconds": seconds,
        }
        if half:
            # The block's faces absorbing 0.02: along its top the first corner's wave is
            # the Weyl-van der Pol one, below the rigid block's by |F(w)| over the top.
            room = _room(folder, "second", quads, source, receiver, absorption=ABSORBING)
            _, soft = engine(room, "second_order")
            _, soft_pressure = engine(room, "second_order", coherent=COHERENT)
            row["absorbing_minus_btm_db"] = (soft - reference).tolist()
            row["absorbing_in_pressure_minus_btm_db"] = (soft_pressure - reference).tolist()
            row["ground_wave_db"] = ground_wave(ABSORBING, float(np.linalg.norm(p2 - p1)))
        rows.append(row)
    return rows


#: Two thin screens close together, against the wave calculation: the gap (m), the top
#: of the second screen (the first's is at z = 0), and the heights of the source, 3 m
#: before the first, and of the listener, 3 m past the second.
CLOSE_SCREENS = (
    (0.5, 0.0, -1.0, -1.5),
    (1.0, 0.0, -1.0, -1.5),
    (2.0, 0.0, -1.0, -1.5),
    (4.0, 0.0, -1.0, -1.5),
    (1.0, 0.0, -0.3, -0.5),
    (1.0, 0.0, -3.0, -3.0),
    (1.0, -0.4, -1.0, -1.5),
    (2.0, 0.4, -1.0, -1.5),
)
#: The wave calculation's levels for :data:`CLOSE_SCREENS` (:func:`close_screen_references`).
WAVE_STORED = Path(__file__).with_name("close_screens_wave.json")


def _close_name(gap: float, rise: float, zs: float, zl: float) -> str:
    name = f"two screens {gap:g} m apart"
    if rise:
        name += f", the second {abs(rise):g} m {'higher' if rise > 0 else 'lower'}"
    if (zs, zl) != (-1.0, -1.5):
        name += f", source and listener {-zs:g} and {-zl:g} m below the first top"
    return name


def close_screen_references() -> None:
    """Runs the wave calculation for :data:`CLOSE_SCREENS` and stores its levels in
    :data:`WAVE_STORED`, with the places of its source's and listener's cells as the
    engine's model has them: each screen one cell thick, its plane at the cell's
    middle."""
    import rows  # the wave calculation; it imports this module

    h = rows.GRID_M
    stored = {}
    for gap, rise, zs, zl in CLOSE_SCREENS:
        started = time.time()
        solids = [(0.0, h, -math.inf, 0.0), (gap, gap + h, -math.inf, rise)]
        frame = dict(
            x_range=(-4.0, gap + 4.0), z_range=(min(zs, zl, 0.0) - 2.0, max(rise, 0.0) + 2.0)
        )
        source, listener = (-3.0 + 0.5 * h, rows._cell(zs)), (gap + 3.0 + 0.5 * h, rows._cell(zl))
        # By reciprocity, from the listener's place.
        traces, dt, at, heard = rows.fdtd(solids, listener, [source], **frame)
        free = rows.fdtd([], listener, [source], **frame)[0]
        distance = math.hypot(at[0] - heard[0][0], at[1] - heard[0][1])
        pressure = rows.point_source(traces[:, 0], free[:, 0], distance, dt)
        stored[_close_name(gap, rise, zs, zl)] = {
            "source": [heard[0][0] - 0.5 * h, 0.0, heard[0][1]],
            "listener": [at[0] - 0.5 * h, 0.0, at[1]],
            "wave_db": [round(float(v), 3) for v in rows.band_levels(pressure, distance)],
        }
        print(f"{_close_name(gap, rise, zs, zl)} ({time.time() - started:.0f} s)", flush=True)
        WAVE_STORED.write_text(json.dumps(stored, indent=1) + "\n")


def _passes_above(a, b, x: float, top: float) -> bool:
    """Whether the line from ``a`` to ``b`` passes above ``top`` at ``x``."""
    t = (x - a[0]) / (b[0] - a[0])
    return bool(a[2] + t * (b[2] - a[2]) > top)


def close_screens(folder: Path) -> list[dict]:
    """Two thin screens close together (:data:`CLOSE_SCREENS`): BTM to the second order
    (over both tops, and over one where an end sees past the other screen's top) and the
    engine with ``second_order``, by default (the 63 and 125 Hz bands in pressure) with
    ray tracing and with image sources, against the wave calculation."""
    stored = json.loads(WAVE_STORED.read_text())
    out = []
    bands = len(next(iter(stored.values()))["wave_db"])
    for gap, rise, zs, zl in CLOSE_SCREENS:
        name = _close_name(gap, rise, zs, zl)
        entry = stored[name]
        source, receiver = np.array(entry["source"]), np.array(entry["listener"])
        wave = np.array(entry["wave_db"])
        e1 = Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
        e2 = Edge([gap, 0, rise], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
        quads = screen_quads() + screen_quads(gap, rise)
        room = _room(folder, "close", quads, source, receiver)
        p1, p2 = _double_path(engine(room, "second_order", fade=False)[0])
        centres = (float((p1 - e1.origin) @ e1.direction), float((p2 - e2.origin) @ e2.direction))
        btm = btm_second(e1, e2, source, receiver, centres, False)
        # Where one end sees past the other screen's top to an edge, BTM's first order
        # over that edge too: with the second screen 0.4 m lower the listener sees the
        # first top 2.5 cm above the second.
        if _passes_above(e1.origin, receiver, gap, rise):
            btm = btm + btm_first(e1, source, receiver)
        if _passes_above(source, e2.origin, 0.0, 0.0):
            btm = btm + btm_first(e2, source, receiver)
        distance = float(np.linalg.norm(receiver - source))
        row = {
            "case": name,
            "source": source.tolist(),
            "receiver": receiver.tolist(),
            "wave_db": wave.tolist(),
            "btm_minus_wave_db": (band_level(btm, distance)[:bands] - wave).tolist(),
        }
        for method in ("ray_tracing", "image_sources"):
            settings = {"early": {"method": method}}
            room = _room(folder, "close", quads, source, receiver, settings)
            level = engine(room, "second_order", coherent=COHERENT)[1]
            row[f"{method}_minus_wave_db"] = (level[:bands] - wave).tolist()
        out.append(row)
    return out


#: A thin ceiling 1 m over two thin screens 2 m apart (tops at z = 0, x = -1 and 1),
#: the source 2 m before the first and the listener 2 m past the second (x = -3 and 3),
#: 3 m below the tops, against the wave calculation: the path over both tops reflects
#: off the ceiling at x = 0, and the ceiling runs from ``start`` to ``end`` in the wave
#: calculation's grid (each a cell boundary there, 1 cm past the place the engine has:
#: see :func:`ceiling_references`).
CEILING = (
    ("whole", -math.inf, math.inf),
    ("ending 1.01 m past the reflection", -math.inf, 1.02),
    ("ending 0.25 m past it", -math.inf, 0.26),
    ("ending 1 cm past it", -math.inf, 0.02),
    ("ending 1 cm short of it", -math.inf, 0.0),
    ("ending 0.25 m short of it", -math.inf, -0.24),
    ("ending 0.99 m short of it", -math.inf, -0.98),
    ("a strip 0.5 m wide round it", -0.24, 0.26),
    ("a strip 1 m wide round it", -0.48, 0.52),
    ("a strip 2 m wide round it", -0.98, 1.02),
)
#: The ceiling's height over the tops (m).
CEILING_M = 1.0
#: The wave calculation's levels for :data:`CEILING` (:func:`ceiling_references`).
CEILING_STORED = Path(__file__).with_name("ceiling_wave.json")


def ceiling_references() -> None:
    """Runs the wave calculation for :data:`CEILING` and stores its levels in
    :data:`CEILING_STORED`, the places as the engine's model has them: each screen and
    the ceiling one cell thick, their planes at the cells' middles (the ceiling's at
    z = 1.01 m, its ends half a cell before their cells' boundaries)."""
    import rows  # the wave calculation; it imports this module

    h = rows.GRID_M
    stored = json.loads(CEILING_STORED.read_text()) if CEILING_STORED.exists() else {}
    frame = dict(x_range=(-4.0, 4.0), z_range=(-5.0, CEILING_M + 2.0))
    source, listener = (-3.0 + 0.5 * h, rows._cell(-3.0)), (3.0 + 0.5 * h, rows._cell(-3.0))
    free = None
    for name, start, end in CEILING:
        started = time.time()
        solids = [
            (-1.0, -1.0 + h, -math.inf, 0.0),
            (1.0, 1.0 + h, -math.inf, 0.0),
            (start, end, CEILING_M, CEILING_M + h),
        ]
        # By reciprocity, from the listener's place.
        traces, dt, at, heard = rows.fdtd(solids, listener, [source], **frame)
        if free is None:
            free = rows.fdtd([], listener, [source], **frame)[0]
        distance = math.hypot(at[0] - heard[0][0], at[1] - heard[0][1])
        pressure = rows.point_source(traces[:, 0], free[:, 0], distance, dt)
        stored[name] = {
            "source": [heard[0][0] - 0.5 * h, 0.0, heard[0][1]],
            "listener": [at[0] - 0.5 * h, 0.0, at[1]],
            "ceiling": [start - 0.5 * h, end - 0.5 * h, CEILING_M + 0.5 * h],
            "wave_db": [round(float(v), 3) for v in rows.band_levels(pressure, distance)],
        }
        print(f"ceiling {name} ({time.time() - started:.0f} s)", flush=True)
        CEILING_STORED.write_text(json.dumps(stored, indent=1) + "\n")


def ceiling(folder: Path) -> list[dict]:
    """A thin ceiling over two thin screens (:data:`CEILING`), whole, ending near the
    point the path over both tops reflects at or a strip round it: the engine with
    ``second_order``, by default (the 63 and 125 Hz bands in pressure) and with every
    band in pressure, with ray tracing and with image sources, against the wave
    calculation. The reflection off the ceiling is stopped by both screens, and the
    edge fade's product of the two tops' terms stands in for the path over both by way
    of the ceiling, as much of it as the ceiling gives back there."""
    stored = json.loads(CEILING_STORED.read_text())
    out = []
    for name, _, _ in CEILING:
        entry = stored[name]
        source, receiver = np.array(entry["source"]), np.array(entry["listener"])
        wave = np.array(entry["wave_db"])
        bands = len(wave)
        start, end, height = entry["ceiling"]
        start, end = max(start, -EXTENT), min(end, EXTENT)
        quads = [[(start, -EXTENT, height), (start, EXTENT, height), (end, EXTENT, height),
                  (end, -EXTENT, height)]]  # fmt: skip
        quads += screen_quads(-1.0) + screen_quads(1.0)
        row = {"case": name, "wave_db": wave.tolist()}
        for method in ("ray_tracing", "image_sources"):
            settings = {"early": {"method": method}}
            room = _room(folder, "ceiling", quads, source, receiver, settings)
            model = TraceModel.from_room(room)
            for coherent, tag in ((COHERENT, ""), (len(OCTAVE_BANDS_HZ), "_in_pressure")):
                level = engine(room, "second_order", model=model, coherent=coherent)[1]
                row[f"{method}{tag}_minus_wave_db"] = (level[:bands] - wave).tolist()
        out.append(row)
    return out


#: A thin strip (:func:`strip_quads`) and the source and the listener in its shadow on
#: either side of it.
STRIP = (
    ((-3.0, 0.0, -0.5), (3.0, 0.0, -1.5)),
    ((-2.0, 0.0, -1.0), (4.0, 0.0, -1.2)),
    ((-1.0, 0.0, -0.3), (1.5, 0.0, -1.9)),
)
STRIP_WIDTH = 2.0


def strip_quads(width: float = STRIP_WIDTH) -> list:
    """A thin strip in the plane x = 0 from z = -``width`` to its top at z = 0."""
    return [[(0, -EXTENT, -width), (0, EXTENT, -width), (0, EXTENT, 0), (0, -EXTENT, 0)]]


def thin_strip(folder: Path) -> list[dict]:
    """A thin strip 2 m wide, the source and the listener in its shadow on either side
    (:data:`STRIP`): BTM over its top and its foot and over both, along each face of the
    strip (with half the coefficient at grazing incidence, each), against the whole early
    stage with paths over two edges and without, in pressure at 63 and 125 Hz as by
    default. Along one face alone the paths over both edges would leave the field with a
    step where the listener crosses the strip's plane past its far edge: the first
    edge's wave on one face is the negative of that on the other."""
    top = Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
    foot = Edge([0, 0, -STRIP_WIDTH], [0, 1, 0], [0, 0, 1], [1, 0, 0], 2.0)
    rows = []
    for source, receiver in STRIP:
        source, receiver = np.array(source), np.array(receiver)
        first = btm_first(top, source, receiver) + btm_first(foot, source, receiver)
        along = btm_second(top, foot, source, receiver, (0.0, 0.0), True)
        along += btm_second(foot, top, source, receiver, (0.0, 0.0), True)
        distance = float(np.linalg.norm(receiver - source))
        reference = band_level(first + 2.0 * along, distance)
        row = {
            "source": source.tolist(),
            "receiver": receiver.tolist(),
            "btm_db": reference.tolist(),
            "btm_along_one_face_db": band_level(first + along, distance).tolist(),
            "btm_without_the_paths_over_both_db": band_level(first, distance).tolist(),
        }
        for mode in ("second_order", "first_order"):
            simulation = {"early": {"diffraction": mode}}
            room = _room(folder, "strip", strip_quads(), source, receiver, simulation)
            row[f"{mode}_minus_btm_db"] = (whole_early_stage(room) - reference).tolist()
        rows.append(row)
    return rows


#: The free-standing strips of :func:`strip_sight`: their height (m), and how far below
#: the top the source stands 5 m before them, and so how far above it the line of sight
#: over the top passes 5 m behind them.
STRIP_SIGHT = ((1.0, 0.5), (2.0, 1.0), (4.0, 1.0))
#: Where the listener stands behind them: how far above the line of sight (m).
SIGHT_OFFSETS = (-0.3, -1e-4, 1e-4, 0.3)


def strip_sight(folder: Path) -> list[dict]:
    """A thin strip standing free (:func:`strip_quads`), 1, 2 and 4 m high
    (:data:`STRIP_SIGHT`), the source 5 m before it and the listener 5 m past it round
    the line of sight over its top (:data:`SIGHT_OFFSETS`): BTM over its top and its
    foot and over both along each face, with the direct sound above the line, against
    the engine (every band in pressure) with each mode, and with the edge diffraction
    alone. Past the foot, where the direct sound is found over the top, the edge
    diffraction leaves the foot's incident term to the terms that make paths fade at
    edges: the step across the line, 0.2 mm, and the largest difference from BTM. BTM
    is averaged over 96 frequencies in each band, as the engine lets each of its eight
    stand for its slice of the band (at those eight BTM's level 0.3 m above the line
    behind the strip 4 m high is 0.18 dB above its level at 32 in the 4 kHz band)."""
    k = 2.0 * np.pi * band_frequencies(96)[0].ravel() / C
    top = Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
    rows = []
    for height, below in STRIP_SIGHT:
        foot = Edge([0, 0, -height], [0, 1, 0], [0, 0, 1], [1, 0, 0], 2.0)
        source = np.array([-5.0, 0.0, -below])
        exact, found = [], {}
        for offset in SIGHT_OFFSETS:
            receiver = np.array([5.0, 0.0, below + offset])
            distance = float(np.linalg.norm(receiver - source))
            # Sampled finely: the top's terms are on their boundary.
            pressure = btm_first(top, source, receiver, k, du=0.001)
            pressure = pressure + btm_first(foot, source, receiver, k, du=0.001)
            for e1, e2 in ((top, foot), (foot, top)):
                pressure = pressure + 2.0 * btm_second(e1, e2, source, receiver, (0, 0), True, k)
            if offset > 0.0:
                pressure = pressure + np.exp(-1j * k * distance) / distance
            exact.append(band_level(pressure, distance))
            for mode in ("direct", "first_order", "second_order"):
                simulation = {"early": {"diffraction": mode}}
                room = _room(folder, "sight", strip_quads(height), source, receiver, simulation)
                for fade in (True, False):
                    level = engine(room, mode, coherent=8, fade=fade)[1]
                    found.setdefault((mode, fade), []).append(level)
        exact = np.array(exact)
        row = {
            "height": height,
            "source": source.tolist(),
            "receivers": [[5.0, 0.0, below + offset] for offset in SIGHT_OFFSETS],
            "btm_db": exact.tolist(),
            "btm_step_db": (exact[2] - exact[1]).tolist(),
        }
        for (mode, fade), levels in found.items():
            levels = np.array(levels)
            row[mode if fade else f"{mode}, the edge diffraction alone"] = {
                "step_db": (levels[2] - levels[1]).tolist(),
                "largest_difference_db": np.abs(levels - exact).max(axis=0).tolist(),
            }
        rows.append(row)
    return rows


def ground_wave(absorption: float, distance: float) -> list[float]:
    """20 lg |F(w)| (dB) in each octave band, at its centre: the Weyl-van der Pol field
    over ``distance`` m of a locally reacting surface at grazing incidence, 2 F(w) times
    the free one, against a rigid one's twice, with w = (1 + j) / 2 sqrt(k distance) beta
    and the real admittance beta whose random-incidence absorption (Paris's formula) is
    ``absorption``."""
    from scipy.optimize import brentq
    from scipy.special import wofz

    def paris(z: float) -> float:  # of a real normalised impedance z
        return 8.0 / z * (1.0 + 1.0 / (1.0 + z) - 2.0 / z * math.log(1.0 + z))

    beta = 1.0 / brentq(lambda z: paris(z) - absorption, 1.0, 1e9)
    k = 2.0 * math.pi * np.array(OCTAVE_BANDS_HZ, dtype=float) / C
    w = 0.5 * (1.0 + 1.0j) * np.sqrt(k * distance) * beta
    return (20.0 * np.log10(np.abs(1.0 + 1.0j * math.sqrt(math.pi) * w * wofz(w)))).tolist()


def shadow_boundary(folder: Path) -> list[dict]:
    """First order at the shadow boundary of a thin screen, of a right-angled corner
    (the block, 300 m wide) and of a wedge open over 225 degrees (the block with its
    front face at 45 degrees), with the source 5 m in front of the edge and 2 m below it
    and the listener 5 m behind it, on the boundary (1 mm into the shadow) and 10.5
    degrees into the shadow: the whole early stage (``early_paths`` with the default
    settings), with every band added in energy (``coherent_max_band_hz: null``) and as
    by default (63 and 125 Hz in pressure), against BTM averaged over 96 frequencies in
    each band. The engine's levels are re the direct sound in free field in the same air
    (the air's absorption over the direct distance taken off)."""
    k = 2.0 * np.pi * band_frequencies(96)[0].ravel() / C
    source = np.array([-5.0, 0.0, -2.0])
    cases = (
        (
            "thin screen",
            screen_quads(),
            Edge([0, 0, 0], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0),
        ),
        (
            "right-angled corner",
            barrier_quads(EXTENT),
            Edge([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], 1.5),
        ),
        (
            "wedge open over 225 degrees",
            barrier_quads(EXTENT, run=EXTENT),
            Edge([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], 1.25),
        ),
    )
    rows = []
    for name, quads, edge in cases:
        # The shadow boundary, the line from the source over the edge, is at z = 2.
        for where, z in (("on the shadow boundary", 1.999), ("10.5 degrees into the shadow", 1.0)):
            receiver = np.array([5.0, 0.0, z])
            distance = float(np.linalg.norm(receiver - source))
            exact = band_level(btm_first(edge, source, receiver, k), distance)
            row = {"case": name, "where": where, "receiver": receiver.tolist()}
            row["btm_db"] = exact.tolist()
            for key, simulation in (
                ("in_energy", {"early": {"coherent_max_band_hz": None}}),
                ("default", None),
            ):
                room = _room(folder, "boundary", quads, source, receiver, simulation)
                energy = total_energy(early_paths(room))
                energy *= np.exp(room.air_attenuation() * distance)
                level = 10.0 * np.log10(energy * 4.0 * np.pi * distance**2)
                row[key + "_minus_btm_db"] = (level - exact).tolist()
            rows.append(row)
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


#: Where the source and the listener stand just over a block's top, in sight of each
#: other: the block's width, how far the source stands before it and the listener past
#: it, and their heights above the plane of the top (m).
IN_SIGHT = (
    (10.0, 10.0, 5.0, (0.01, 0.01)),
    (10.0, 10.0, 5.0, (0.5, 0.5)),
    (10.0, 10.0, 5.0, (0.5, 0.05)),
    (10.0, 10.0, 5.0, (1.0, 1.0)),
    (10.0, 10.0, 5.0, (2.0, 2.0)),
    (1.0, 3.0, 3.0, (0.05, 0.05)),
    (1.0, 3.0, 3.0, (0.3, 0.3)),
)
#: The bands in which the scenes of :func:`in_sight` add paths in pressure: to 500 Hz.
IN_SIGHT_COHERENT = 4


def whole_early_stage(room: Room) -> np.ndarray:
    """The level of the room's whole early stage (specular paths, edge diffraction and
    the terms that make paths fade at edges, as a simulation has them) from its source
    to its listener, per band in dB re the direct sound in free field, without the
    air."""
    source = np.array(room.scene.sources[0].position)
    receiver = np.array(room.scene.listeners[0].position)
    distance = float(np.linalg.norm(receiver - source))
    energy = total_energy(early_paths(room)) * np.exp(room.air_attenuation() * distance)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(energy * 4.0 * np.pi * distance**2)


def btm_over_block(width: float, source, receiver, k=K) -> tuple[np.ndarray, np.ndarray]:
    """The pressure at ``receiver`` from ``source`` (both at y = 0) over the block of
    :func:`barrier_quads`, ``width`` m wide: the geometric waves (the direct sound and
    its reflection off the top), the paths over each of the top's corners and the path
    over both; and the same without the path over both."""
    source, receiver = np.asarray(source, dtype=float), np.asarray(receiver, dtype=float)
    e1 = Edge([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], 1.5)
    e2 = Edge([width, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1], 1.5)
    run = receiver[0] - source[0]
    first = np.zeros(len(k), dtype=complex)
    # The line of sight clears the top where it passes above both corners.
    above = [source[2] + (receiver[2] - source[2]) * (x - source[0]) / run for x in (0.0, width)]
    if min(above) > 0.0:
        distance = float(np.linalg.norm(receiver - source))
        first += np.exp(-1j * k * distance) / distance
    if source[2] > 0.0 and receiver[2] > 0.0:
        at = source[0] + run * source[2] / (source[2] + receiver[2])
        if 0.0 <= at <= width:
            distance = math.hypot(run, source[2] + receiver[2])
            first += np.exp(-1j * k * distance) / distance
    if receiver[2] > 0.0:  # the near corner sees the listener
        first += btm_first(e1, source, receiver, k)
    if source[2] > 0.0:
        first += btm_first(e2, source, receiver, k)
    return first + btm_second(e1, e2, source, receiver, (0.0, 0.0), True, k), first


def in_sight(folder: Path) -> list[dict]:
    """The source and the listener in sight of each other just over a block's top, a
    little above its plane on either side (:data:`IN_SIGHT`): BTM against the engine's
    whole early stage with paths over two edges and without, in the bands where the
    scene adds paths in pressure (to 500 Hz). The source's wave passes both corners
    there, each takes its terms off it, and the path over both gives back what they
    take twice."""
    rows = []
    for width, before, past, (hs, hr) in IN_SIGHT:
        source, receiver = np.array([-before, 0.0, hs]), np.array([width + past, 0.0, hr])
        distance = float(np.linalg.norm(receiver - source))
        total, first = btm_over_block(width, source, receiver)
        reference = band_level(total, distance)
        row = {
            "width": width,
            "source": source.tolist(),
            "receiver": receiver.tolist(),
            "btm_db": reference.tolist(),
            "btm_without_the_path_over_both_db": band_level(first, distance).tolist(),
        }
        for mode in ("second_order", "first_order"):
            simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": 500}}
            room = _room(folder, "sight", barrier_quads(width), source, receiver, simulation)
            row[f"{mode}_minus_btm_db"] = (whole_early_stage(room) - reference).tolist()
        rows.append(row)
    return rows


def round_the_plane(folder: Path, heights=(-1e-3, -1e-6, 1e-6, 1e-3)) -> list[dict]:
    """The whole early stage's level (``second_order``, as in :func:`in_sight`) for every
    pair of ``heights`` of the source and the listener round the plane of a block's
    top, for the two blocks of :data:`IN_SIGHT`: the largest difference between two
    pairs in each band, the same with the pairs that have either exactly in the plane,
    and the largest difference between a pair and the same heights swapped."""
    rows = []
    simulation = {"early": {"diffraction": "second_order", "coherent_max_band_hz": 500}}
    for width, before, past in ((10.0, 10.0, 5.0), (1.0, 3.0, 3.0)):
        every = sorted({*heights, 0.0})
        levels = np.array(
            [
                [
                    whole_early_stage(
                        _room(
                            folder,
                            "round",
                            barrier_quads(width),
                            [-before, 0.0, hs],
                            [width + past, 0.0, hr],
                            simulation,
                        )
                    )
                    for hr in every
                ]
                for hs in every
            ]
        )
        off = [i for i, h in enumerate(every) if h != 0.0]
        beside = levels[np.ix_(off, off)]
        rows.append(
            {
                "width": width,
                "heights": list(every),
                "largest_difference_db": np.ptp(beside, axis=(0, 1)).tolist(),
                "with_the_plane_db": np.ptp(levels, axis=(0, 1)).tolist(),
                "swapped_db": np.abs(beside - beside.transpose(1, 0, 2)).max(axis=(0, 1)).tolist(),
            }
        )
    return rows


#: Two screens in a row and where the source and the listener stand to graze both tops:
#: the source before the first, the screens apart, the listener past the second (m).
GRAZING = (10.0, 10.0, 5.0)


def grazing_two_screens(folder: Path, heights=(1e-3, -1e-3)) -> dict:
    """Two thin screens in a row with their tops on one line, and the source and the
    listener ``heights`` above that line (below it for negative ones), :data:`GRAZING`:
    the engine's level over both against the field of two knife edges at grazing
    incidence at high frequencies, 1/4 + atan(sqrt(a c / (b (a + b + c)))) / (2 pi) of
    the free field for the source a before the first edge, the edges b apart and the
    listener c past the second (1/3 for equal distances; one edge alone leaves 1/2, two
    far apart 1/4)."""
    a, b, c = GRAZING
    limit = 0.25 + math.atan(math.sqrt(a * c / (b * (a + b + c)))) / (2.0 * math.pi)
    out = {"distances": list(GRAZING), "closed_form_db": 20.0 * math.log10(limit), "rows": []}
    quads = screen_quads() + screen_quads(b, 0.0)
    for h in heights:
        row = {"height": h}
        for mode in ("second_order", "first_order"):
            simulation = {"early": {"diffraction": mode}}
            room = _room(folder, "grazing", quads, [-a, 0.0, h], [b + c, 0.0, h], simulation)
            row[f"{mode}_db"] = whole_early_stage(room).tolist()
        out["rows"].append(row)
    return out


#: Where the source stands above the plane of a 10 m block's top, 10 m before it, and
#: the listener below it, 5 m past the block (m).
BEHIND = ((0.01, -0.5), (0.01, -5.0), (0.5, -2.0), (2.0, -2.0))


def behind_a_block(folder: Path) -> list[dict]:
    """The listener behind a 10 m block, below the plane of its top, and the source
    above that plane (:data:`BEHIND`): BTM against the engine's whole early stage with
    paths over two edges and without, in the bands where the scene adds paths in
    pressure (to 500 Hz). The far corner's path alone takes the source's wave as if the
    near corner were not there; the path over both takes off what the near corner
    does."""
    rows = []
    for hs, hr in BEHIND:
        source, receiver = np.array([-10.0, 0.0, hs]), np.array([15.0, 0.0, hr])
        distance = float(np.linalg.norm(receiver - source))
        reference = band_level(btm_over_block(10.0, source, receiver)[0], distance)
        row = {
            "source": source.tolist(),
            "receiver": receiver.tolist(),
            "btm_db": reference.tolist(),
        }
        for mode in ("second_order", "first_order"):
            simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": 500}}
            room = _room(folder, "behind", barrier_quads(10.0), source, receiver, simulation)
            row[f"{mode}_minus_btm_db"] = (whole_early_stage(room) - reference).tolist()
        rows.append(row)
    return rows


def _moved(quads: list, by: float) -> list:
    return [[(x + by, y, z) for x, y, z in quad] for quad in quads]


#: Screens and flat tops of one height in a row, by name: their faces and how far past
#: x = 0 the listener stands (the source stands 10 m before it).
IN_A_ROW = {
    "two screens 10 m apart": (screen_quads() + screen_quads(10.0), 15.0),
    "three screens 5 m apart": (screen_quads() + screen_quads(5.0) + screen_quads(10.0), 15.0),
    "one top 10 m wide": (barrier_quads(10.0), 15.0),
    "two tops 5 m wide, 5 m apart": (barrier_quads(5.0) + _moved(barrier_quads(5.0), 10.0), 20.0),
    "a 10 m top with a 2 cm slot": (
        barrier_quads(4.99) + _moved(barrier_quads(4.99), 5.01),
        15.0,
    ),
    "three tops 3 m wide, 2 m apart": (
        barrier_quads(3.0) + _moved(barrier_quads(3.0), 5.0) + _moved(barrier_quads(3.0), 10.0),
        18.0,
    ),
    "a top 5 m wide, then a screen": (barrier_quads(5.0) + screen_quads(10.0), 15.0),
    "a screen, then a top 5 m wide": (screen_quads() + _moved(barrier_quads(5.0), 5.0), 15.0),
}


def in_a_row(folder: Path, heights=(1e-2, 0.0, -1e-5, -1e-2), names=None) -> list[dict]:
    """Screens and flat tops of one height in a row (:data:`IN_A_ROW`, or those of
    ``names``), the source and the listener both ``heights`` above the plane of the
    tops: the whole early stage's level with paths over two edges and without, with
    the paths in pressure to 500 Hz and in energy in every band. No reference: the
    levels show where the positions agree and where they do not. The rays that look for
    the paths start a tenth of a millimetre off the edges here
    (:func:`~acoustic_spaces.engine.diffraction.ray_offset`), so a height of -1e-5 m is
    within their reach below the plane."""
    rows = []
    for name in names or IN_A_ROW:
        quads, past = IN_A_ROW[name]
        for h in heights:
            row = {"scene": name, "height": h}
            for mode in ("second_order", "first_order"):
                for coherent in (500, None):
                    simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": coherent}}
                    room = _room(folder, "row", quads, [-10.0, 0.0, h], [past, 0.0, h], simulation)
                    key = f"{mode}_{'pressure_to_500' if coherent else 'energy'}_db"
                    with np.errstate(divide="ignore"):
                        row[key] = whole_early_stage(room).tolist()
            rows.append(row)
    return rows


def across_the_plane(folder: Path, other: float = 0.5, step: float = 1e-3) -> list[dict]:
    """The source, then the listener, ``step`` below and above the plane of a 10 m
    block's top with the other ``other`` m above it: the whole early stage's level
    (``second_order``) with every band in energy and with the paths in pressure to 500
    Hz, and the step between the two heights. Above the plane the source has an image
    in the top, which the bands that add in energy count as a wave of its own."""
    rows = []
    for who in ("source", "listener"):
        for coherent in (None, 500):
            simulation = {
                "early": {"diffraction": "second_order", "coherent_max_band_hz": coherent}
            }
            levels = []
            for h in (-step, step):
                ends = ([-10.0, 0.0, h], [15.0, 0.0, other])
                if who == "listener":
                    ends = ([-10.0, 0.0, other], [15.0, 0.0, h])
                room = _room(folder, "across", barrier_quads(10.0), *ends, simulation)
                levels.append(whole_early_stage(room))
            rows.append(
                {
                    "crossing": who,
                    "bands": "pressure to 500 Hz" if coherent else "energy",
                    "below_db": levels[0].tolist(),
                    "above_db": levels[1].tolist(),
                    "step_db": (levels[1] - levels[0]).tolist(),
                }
            )
    return rows


def screen_corner(
    folder: Path, angle: float = 90.0, turned: float = 0.0, ys=None, per_band: int | None = None
) -> dict:
    """The corner of a thin screen in the plane x = 0, at the origin: the screen lies
    between its top edge, which runs along -y turned by ``turned`` degrees towards -z,
    and its side edge, ``angle`` degrees further round (by default the quarter plane
    y < 0, z < 0). The listener walks along y at ``ys`` (by default 1 cm at a time past
    where the direct sound comes round the side edge of the quarter plane while the
    path over the top edge still touches it near the corner). The engine (``direct``,
    every band in pressure) against BTM over both edges, each from the corner on, at
    the engine's frequencies or ``per_band`` in each band; the sound diffracted from one
    edge to the other is left out."""
    k = K if per_band is None else 2.0 * np.pi * band_frequencies(per_band)[0].ravel() / C
    # Rounded, so that a right angle is one exactly.
    first, second = math.radians(turned), math.radians(turned + angle)
    along_top = np.round([0.0, -math.cos(first), -math.sin(first)], 15)
    along_side = np.round([0.0, -math.cos(second), -math.sin(second)], 15)
    into_top = np.round([0.0, math.sin(first), -math.cos(first)], 15)  # into the screen
    into_side = np.round([0.0, -math.sin(second), math.cos(second)], 15)
    # A rhombus whose far edges lie EXTENT from the corner's edges.
    reach = EXTENT / math.sin(math.radians(angle))
    corners = [
        reach * (along_top + along_side),
        reach * along_side,
        0 * along_top,
        reach * along_top,
    ]
    quads = [[tuple(float(v) + 0.0 for v in corner) for corner in corners]]
    top = Edge([0, 0, 0], along_top, into_top, [1, 0, 0], 2.0)
    side = Edge([0, 0, 0], along_side, into_side, [1, 0, 0], 2.0)
    source = np.array([-5.0, -3.0, -1.0])
    if ys is None:
        ys = np.arange(2.5, 4.0 + 1e-9, 0.01) + 0.00317  # none exactly on a boundary
    found, exact = [], []
    for y in ys:
        receiver = np.array([5.0, y, -2.0])
        _, level = engine(_room(folder, "corner", quads, source, receiver), "direct", coherent=8)
        found.append(level)
        pressure = btm_first(top, source, receiver, k, ends=(0.0, np.inf))
        pressure = pressure + btm_first(side, source, receiver, k, ends=(0.0, np.inf))
        t = source[0] / (source[0] - receiver[0])  # where the line crosses x = 0
        crossing = source + t * (receiver - source)
        distance = float(np.linalg.norm(receiver - source))
        if not (crossing @ into_top > 0.0 and crossing @ into_side > 0.0):
            pressure = pressure + np.exp(-1j * k * distance) / distance
        exact.append(band_level(pressure, distance))
    found, exact = np.array(found), np.array(exact)
    steps = np.abs(np.diff(found, axis=0))
    return {
        "angle_deg": angle,
        "turned_deg": turned,
        "source": source.tolist(),
        "line": [[5.0, float(ys[0]), -2.0], [5.0, float(ys[-1]), -2.0]],
        "largest_step_db": steps.max(axis=0).tolist(),
        "largest_step_at_y": ys[1:][steps.argmax(axis=0)].tolist(),
        "exact_largest_step_db": np.abs(np.diff(exact, axis=0)).max(axis=0).tolist(),
        "largest_difference_db": np.abs(found - exact).max(axis=0).tolist(),
    }


#: The corners of :func:`other_corners`: the angle between the screen's two edges and how
#: far its top edge is turned from level (degrees). The second has one edge upright; the
#: last three are turns of a flat screen's outline by 44.5, 30 and 10 degrees.
CORNERS = (
    (45.0, 0.0),
    (45.0, 45.0),
    (60.0, 0.0),
    (120.0, 0.0),
    (135.5, 0.0),
    (150.0, 0.0),
    (170.0, 0.0),
)


def other_corners(folder: Path) -> list[dict]:
    """:func:`screen_corner` for corners that are not right-angled (:data:`CORNERS`), the
    listener walking from (5, -1, -2) to (5, 6, -2), 5 cm at a time, across the shadow
    boundaries of both edges. BTM is averaged over 96 frequencies in each band: along
    this walk the paths over the two edges come to differ in length by the speed of
    sound over the spacing of the engine's eight frequencies in a band (0.49 m at
    8 kHz), where eight of them alias, while the engine lets each frequency stand for
    its slice of the band."""
    ys = np.arange(-1.0, 6.0 + 1e-9, 0.05) + 0.00317  # none exactly on a boundary
    return [screen_corner(folder, angle, turned, ys, per_band=96) for angle, turned in CORNERS]


def bent_quads(bend: float) -> list:
    """Two screens below z = 0 that meet along the z axis: one in the plane x = 0 on the
    side y < 0, the other on the side y > 0, turned from that plane about the z axis by
    ``bend`` degrees (towards +x for a positive bend)."""
    a = EXTENT
    s, c = math.sin(math.radians(bend)), math.cos(math.radians(bend))
    return [
        [(0, -a, -a), (0, 0, -a), (0, 0, 0), (0, -a, 0)],
        [(0, 0, -a), (a * s, a * c, -a), (a * s, a * c, 0), (0, 0, 0)],
    ]


def bent_screen_levels(folder: Path, bend: float, source, receivers):
    """The levels at ``receivers`` behind :func:`bent_quads`: the engine's (``direct``,
    every band in pressure) and BTM's over both top edges, each from the joint on, with
    the direct sound where neither screen hides it; the sound diffracted from one edge
    to the other is left out."""
    s, c = math.sin(math.radians(bend)), math.cos(math.radians(bend))
    edges = (
        Edge([0, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], 2.0),
        Edge([0, 0, 0], [s, c, 0], [0, 0, -1], [c, -s, 0], 2.0),
    )
    source = np.asarray(source, dtype=float)
    found, exact = [], []
    for receiver in np.asarray(receivers, dtype=float):
        room = _room(folder, "bent", bent_quads(bend), source, receiver)
        found.append(engine(room, "direct", coherent=8)[1])
        pressure = sum(btm_first(edge, source, receiver, ends=(0.0, np.inf)) for edge in edges)
        hidden = False
        for edge in edges:  # where the line crosses each screen's plane: on the screen?
            across = (receiver - source) @ edge.normal
            if across != 0.0:
                t = (edge.origin - source) @ edge.normal / across
                at = source + t * (receiver - source) - edge.origin
                hidden |= 0.0 < t < 1.0 and at @ edge.direction > 0.0 and at @ edge.along > 0.0
        distance = float(np.linalg.norm(receiver - source))
        if not hidden:
            pressure = pressure + np.exp(-1j * K * distance) / distance
        exact.append(band_level(pressure, distance))
    return np.array(found), np.array(exact)


#: The bends of :func:`bent_screen` (degrees): away from the source, then towards it.
BENDS = (2.0, 5.0, 10.0, 14.0, -2.0, -5.0, -10.0, -14.0)


def bent_screen(folder: Path) -> list[dict]:
    """A screen whose top bends by a few degrees (:data:`BENDS`, :func:`bent_quads`),
    with the source at (-5, -0.5, -1): the listener rises through the top's shadow
    boundary 1 cm at a time, from z = 0.9 to 1.1 at (5, -0.5, z), where the line to the
    source passes the top 0.5 m from the bend, and at (5, 0.3, z), where it does 0.1 m
    from it; and walks deep in the shadow past the bend, from (5, -1, -2) to (5, 3, -2)
    5 cm at a time. The engine against BTM (:func:`bent_screen_levels`)."""
    source = (-5.0, -0.5, -1.0)
    heights = np.arange(0.90, 1.10 + 1e-9, 0.01) + 0.00137  # none exactly on a boundary
    lines = {
        "boundary": [(5.0, -0.5, z) for z in heights],
        "boundary_near": [(5.0, 0.3, z) for z in heights],
        "shadow": [(5.0, y, -2.0) for y in np.arange(-1.0, 3.0 + 1e-9, 0.05) + 0.00317],
    }
    rows = []
    for bend in BENDS:
        row: dict = {"bend_deg": bend, "source": list(source)}
        for name, line in lines.items():
            found, exact = bent_screen_levels(folder, bend, source, line)
            row[name] = {
                "line": [list(line[0]), list(line[-1])],
                "largest_step_db": np.abs(np.diff(found, axis=0)).max(axis=0).tolist(),
                "exact_largest_step_db": np.abs(np.diff(exact, axis=0)).max(axis=0).tolist(),
                "largest_difference_db": np.abs(found - exact).max(axis=0).tolist(),
            }
        rows.append(row)
    return rows


class Prism:
    """A rigid convex prism: a convex polygon ``plan`` (x, y) from z = ``z0`` up to ``z1``,
    open underneath unless ``bottom``, in free field; or, with ``ground``, standing on a
    rigid ground at z = 0 (BTM then takes the prism doubled about the ground, from -z1 to
    z1, with the source and its image)."""

    def __init__(self, plan, z0: float = -EXTENT, z1: float = 0.0, bottom=False, ground=False):
        plan = np.asarray(plan, dtype=float)
        x, y = plan[:, 0], plan[:, 1]
        if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0.0:
            plan = plan[::-1]  # counter-clockwise seen from above
        self.plan, self.z0, self.z1, self.bottom, self.ground = plan, z0, z1, bottom, ground

    def _sides(self):
        return [(self.plan[i], self.plan[(i + 1) % len(self.plan)]) for i in range(len(self.plan))]

    def polygons(self, z0: float, z1: float, bottom: bool) -> list:
        """The faces: the top, the bottom where there is one, and the sides."""
        out = [[(x, y, z1) for x, y in self.plan]]
        if bottom:
            out.append([(x, y, z0) for x, y in self.plan[::-1]])
        for a, b in self._sides():
            out.append([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)])
        return out

    def planes(self, z0: float, z1: float, bottom: bool) -> list:
        """(outward normal, offset) of each face, in the order of :meth:`polygons`."""
        out = [(np.array([0.0, 0.0, 1.0]), z1)]
        if bottom:
            out.append((np.array([0.0, 0.0, -1.0]), -z0))
        for a, b in self._sides():
            d = (b - a) / np.linalg.norm(b - a)
            normal = np.array([d[1], -d[0], 0.0])
            out.append((normal, float(normal[:2] @ a)))
        return out

    def edges(self, z0: float, z1: float, bottom: bool) -> list[tuple[Edge, float]]:
        """Each edge as a wedge with its length: those of the top (and of the bottom),
        right-angled, and the upright ones, open over 360 degrees less the plan's angle."""
        out = []
        for a, b in self._sides():
            length = float(np.linalg.norm(b - a))
            d = np.array([(b - a)[0], (b - a)[1], 0.0]) / length
            inward = np.array([-d[1], d[0], 0.0])
            out.append((Edge([a[0], a[1], z1], d, inward, [0.0, 0.0, 1.0], 1.5), length))
            if bottom:
                out.append((Edge([b[0], b[1], z0], -d, inward, [0.0, 0.0, -1.0], 1.5), length))
        for i, here in enumerate(self.plan):
            before, after = self.plan[i - 1], self.plan[(i + 1) % len(self.plan)]
            d_out = (after - here) / np.linalg.norm(after - here)
            d_in = (before - here) / np.linalg.norm(before - here)
            interior = float(np.arccos(np.clip(d_out @ d_in, -1.0, 1.0)))
            normal = np.array([d_out[1], -d_out[0], 0.0])
            along = [d_out[0], d_out[1], 0.0]
            edge = Edge(
                [here[0], here[1], z0], [0.0, 0.0, 1.0], along, normal, 2.0 - interior / np.pi
            )
            out.append((edge, z1 - z0))
        return out

    def _through(self, a, b, z0: float, z1: float, bottom: bool) -> bool:
        """Whether the open segment from ``a`` to ``b`` passes through the prism."""
        d, low, high = b - a, 0.0, 1.0
        planes = self.planes(z0, z1, bottom)
        if not bottom:
            planes = planes + [(np.array([0.0, 0.0, -1.0]), -z0)]
        for normal, offset in planes:
            across, room = normal @ d, offset - normal @ a
            if abs(across) < 1e-15:
                if room < 0.0:
                    return False
                continue
            t = room / across
            low, high = (low, min(high, t)) if across > 0.0 else (max(low, t), high)
        return low < high - 1e-12

    def geometric(self, source, receiver, z0: float, z1: float, bottom: bool, k) -> np.ndarray:
        """The direct sound unless the prism hides it, and the reflections off its faces
        (a convex body has no others)."""
        pressure = np.zeros(len(k), dtype=complex)
        if not self._through(source, receiver, z0, z1, bottom):
            distance = float(np.linalg.norm(receiver - source))
            pressure += np.exp(-1j * k * distance) / distance
        for (normal, offset), polygon in zip(
            self.planes(z0, z1, bottom), self.polygons(z0, z1, bottom), strict=True
        ):
            if normal @ source - offset <= 0.0 or normal @ receiver - offset <= 0.0:
                continue
            image = source - 2.0 * (normal @ source - offset) * normal
            hit = image + (offset - normal @ image) / (normal @ (receiver - image)) * (
                receiver - image
            )
            corners = np.array(polygon)
            sides = np.roll(corners, -1, axis=0) - corners
            if all(
                np.cross(e, hit - c) @ normal >= -1e-12 for e, c in zip(sides, corners, strict=True)
            ):
                distance = float(np.linalg.norm(receiver - image))
                pressure += np.exp(-1j * k * distance) / distance
        return pressure

    def btm(self, source, receiver, k=K) -> np.ndarray:
        """The pressure at ``receiver``: the geometric paths and BTM first order over each
        edge from end to end (the sound diffracted from one edge to another left out),
        relative to the direct sound in free field over the same distance, in dB."""
        source, receiver = np.asarray(source, dtype=float), np.asarray(receiver, dtype=float)
        if self.ground:  # the prism doubled about the ground, the source and its image
            shapes = [(-self.z1, self.z1, True)]
            sources = [source, source * np.array([1.0, 1.0, -1.0])]
        else:
            shapes, sources = [(self.z0, self.z1, self.bottom)], [source]
        pressure = np.zeros(len(k), dtype=complex)
        for z0, z1, bottom in shapes:
            for s in sources:
                pressure = pressure + self.geometric(s, receiver, z0, z1, bottom, k)
                for edge, length in self.edges(z0, z1, bottom):
                    limit = np.pi / edge.nu
                    if edge.frame(s)[1] > limit or edge.frame(receiver)[1] > limit:
                        continue  # one of them is behind the edge's faces
                    pressure = pressure + btm_first(edge, s, receiver, k, ends=(0.0, length))
        return band_level(pressure, float(np.linalg.norm(receiver - source)))

    def seen(self, point) -> tuple:
        """Which of the edges BTM takes see ``point``."""
        shapes = [(-self.z1, self.z1, True)] if self.ground else [(self.z0, self.z1, self.bottom)]
        return tuple(
            bool(edge.frame(np.asarray(point, dtype=float))[1] <= np.pi / edge.nu)
            for shape in shapes
            for edge, _ in self.edges(*shape)
        )

    def room(self, folder: Path, source, receiver) -> Room:
        """The engine's room: the prism's faces, with a rigid ground where it stands on one."""
        polygons = self.polygons(self.z0, self.z1, self.bottom)
        if self.ground:
            e = EXTENT
            polygons = polygons + [[(-e, -e, 0.0), (e, -e, 0.0), (e, e, 0.0), (-e, e, 0.0)]]
        lines, faces = [], []
        for polygon in polygons:
            base = len(lines)
            lines += [f"v {x} {y} {z}" for x, y, z in polygon]
            faces.append("f " + " ".join(str(base + i + 1) for i in range(len(polygon))))
        obj = folder / "prism.obj"
        obj.write_text("\n".join(lines + ["g surfaces"] + faces) + "\n")
        scene = {
            "geometry": {"file": str(obj)},
            "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
            "default_material": "hard",
            "sources": [{"name": "s", "position": list(map(float, source))}],
            "listeners": [{"name": "l", "position": list(map(float, receiver))}],
        }
        return Room.from_scene(Scene.model_validate(scene))


def _walk(start, end, step: float, offset=(0.00317, 0.00513, 0.00731)) -> np.ndarray:
    """Points from ``start`` to ``end`` about ``step`` apart, all moved by ``offset`` so
    that none lies exactly on a boundary."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = int(round(np.linalg.norm(end - start) / step))
    return start + np.outer(np.arange(count + 1) / count, end - start) + np.asarray(offset)


#: The solid block of :func:`prism_corners` in plan (m), its top at z = 0, open underneath
#: and reaching far down.
BLOCK_PLAN = [(0.0, -10.0), (2.0, -10.0), (2.0, 0.0), (0.0, 0.0)]


def _corner_walks() -> list[tuple]:
    """The walks of :func:`prism_corners`: name, prism, source, listener points, mode."""
    block, beside = Prism(BLOCK_PLAN), (0.0, 0.00317, 0.0)
    house = [(0.0, -10.0), (6.0, -10.0), (6.0, 0.0), (0.0, 0.0)]
    return [
        (
            "a block, 0.3 m over its top",
            block,
            (1, 3, 1),
            _walk((4, -8, 0.3), (4, 2, 0.3), 0.05, beside),
            "direct",
        ),
        (
            "a block, 1.8 m over its top",
            block,
            (1, 3, 1),
            _walk((4, -8, 1.8), (4, 2, 1.8), 0.05, beside),
            "direct",
        ),
        (
            "a block, both 1 m under its top",
            block,
            (1, 3, -1),
            _walk((4, -8, -1), (4, 2, -1), 0.05),
            "direct",
        ),
        (
            "a block, both just under its top",
            block,
            (1, 3, -0.3),
            _walk((4, -8, -0.2), (4, 2, -0.2), 0.05),
            "direct",
        ),
        (
            "a block, diagonally past its corner",
            block,
            (1, 3, 1),
            _walk((2.5, -7, 0.5), (7, 3, 0.5), 0.05),
            "direct",
        ),
        (
            "corners of 135 degrees",
            Prism([(0, -10), (2, -10), (2, -1), (1, 0), (0, 0)]),
            (1, 3, 1),
            _walk((4, -8, 0.3), (4, 2, 0.3), 0.05),
            "direct",
        ),
        (
            "a corner of 45 degrees, over the top",
            Prism([(0, 0), (6, 0), (0, 6)]),
            (2, 1, 1),
            _walk((8, -4, 0.4), (8, 4, 0.4), 0.05),
            "direct",
        ),
        (
            "a corner of 45 degrees, under the top",
            Prism([(0, 0), (6, 0), (0, 6)]),
            (3, -3, -1),
            _walk((9, -2, -1), (5, 5, -1), 0.05),
            "direct",
        ),
        (
            "round a 0.5 m thick barrier's end",
            Prism([(0, -30), (0.5, -30), (0.5, 0), (0, 0)]),
            (-5, -5, -1.5),
            _walk((-6, 2, -1), (6, 2, -1), 0.05),
            "direct",
        ),
        (
            "a 3 m building on a hard ground, 1.5 m up",
            Prism(house, z0=0.0, z1=3.0, ground=True),
            (3, 5, 1.0),
            _walk((10, -9, 1.5), (10, 3, 1.5), 0.05),
            "first_order",
        ),
        (
            "the same, 3.5 m up over the roof",
            Prism(house, z0=0.0, z1=3.0, ground=True),
            (3, 5, 4.0),
            _walk((10, -9, 3.5), (10, 3, 3.5), 0.05),
            "first_order",
        ),
        (
            "a 9 m building, above its roof",
            Prism([(-30, 6), (10, 6), (10, 16), (-30, 16)], z0=0.0, z1=9.0, ground=True),
            (-55, 0, 4.05),
            _walk((-5, -4, 10), (15, -4, 10), 0.1),
            "first_order",
        ),
        (
            "along a 7 m house's front, 7.5 m away",
            Prism([(-66, 20), (-54, 20), (-54, 28), (-66, 28)], z0=0.0, z1=7.0, ground=True),
            (0.00317, 7.5, 1.4),
            _walk((-150, 0, 0.2), (30, 0, 0.2), 1.0),
            "first_order",
        ),
    ]


def prism_corners(folder: Path) -> list[dict]:
    """The corners of solid prisms, where three edges meet (a block's, corners of 135
    and 45 degrees in plan, a thick barrier's end, buildings on a hard ground): the
    listener walks past where the paths over the edges reach their ends. The engine
    (every band in pressure; ``direct`` in free field, ``first_order`` on the ground)
    against BTM over each edge from end to end and the geometric paths, the sound
    diffracted from one edge to another left out, averaged over 96 frequencies in each
    band. The steps leave out those where an edge comes into or goes out of sight of the
    listener, where first-order BTM steps itself."""
    k = 2.0 * np.pi * band_frequencies(96)[0].ravel() / C
    out = []
    for name, prism, source, walk, mode in _corner_walks():
        source = np.asarray(source, dtype=float)
        found = np.array([engine(prism.room(folder, source, r), mode, coherent=8)[1] for r in walk])
        exact = np.array([prism.btm(source, r, k) for r in walk])
        seen = [prism.seen(r) for r in walk]
        steady = np.array([a == b for a, b in zip(seen[:-1], seen[1:], strict=True)])
        steps = np.abs(np.diff(found, axis=0))[steady]
        difference = np.abs(found - exact)
        out.append(
            {
                "walk": name,
                "mode": mode,
                "source": source.tolist(),
                "line": [walk[0].tolist(), walk[-1].tolist()],
                "points": len(walk),
                "largest_step_db": steps.max(axis=0).tolist(),
                "exact_largest_step_db": np.abs(np.diff(exact, axis=0))[steady]
                .max(axis=0)
                .tolist(),
                "largest_difference_db": difference.max(axis=0).tolist(),
                "largest_difference_at": walk[difference.argmax(axis=0)].tolist(),
            }
        )
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
    """The cost of the early stage (reflections up to the default order, within the default
    2 s response) in the hall with alcoves, with each diffraction mode."""
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


def _print_close_screens(rows: list[dict]) -> None:
    bands = len(rows[0]["wave_db"])
    header = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ[:bands])
    print(f"\n| two screens close together, against the waves (dB) | {header} |")
    print("|---|" + "---:|" * bands)
    for row in rows:
        print(f"| {row['case']}: wave calculation | {_row(row['wave_db'])} |")
        print(f"| BTM to second order - waves | {_row(row['btm_minus_wave_db'])} |")
        for method in ("ray_tracing", "image_sources"):
            print(f"| engine, {method} - waves | {_row(row[f'{method}_minus_wave_db'])} |")


def _print_ceiling(rows: list[dict]) -> None:
    bands = len(rows[0]["wave_db"])
    header = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ[:bands])
    print(f"\n| a ceiling 1 m over two screens, against the waves (dB) | {header} |")
    print("|---|" + "---:|" * bands)
    for row in rows:
        print(f"| ceiling {row['case']}: wave calculation | {_row(row['wave_db'])} |")
        for method in ("ray_tracing", "image_sources"):
            print(f"| engine, {method} - waves | {_row(row[f'{method}_minus_wave_db'])} |")
            pressure = row[f"{method}_in_pressure_minus_wave_db"]
            print(f"| engine, {method}, in pressure - waves | {_row(pressure)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the numbers to this file")
    parser.add_argument(
        "--close-screens", action="store_true", help="only two screens against the waves"
    )
    parser.add_argument(
        "--ceiling", action="store_true", help="only a ceiling over two screens against the waves"
    )
    parser.add_argument(
        "--wave-references", action="store_true", help="run the wave calculation again"
    )
    args = parser.parse_args()
    if args.wave_references:
        close_screen_references()
        ceiling_references()
        return
    header = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ)
    rule = "|---|" + "---:|" * len(OCTAVE_BANDS_HZ)
    results: dict = {}
    if args.close_screens:
        with tempfile.TemporaryDirectory() as tmp:
            results["close_screens"] = close_screens(Path(tmp))
        _print_close_screens(results["close_screens"])
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2))
        return
    if args.ceiling:
        with tempfile.TemporaryDirectory() as tmp:
            results["ceiling"] = ceiling(Path(tmp))
        _print_ceiling(results["ceiling"])
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2))
        return

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
            if "ground_wave_db" in row:
                print(f"| absorbing {ABSORBING:g} - BTM | {_row(row['absorbing_minus_btm_db'])} |")
                soft = row["absorbing_in_pressure_minus_btm_db"]
                print(f"| absorbing {ABSORBING:g} in pressure - BTM | {_row(soft)} |")
                print(f"| 20 lg abs F(w) over the top | {_row(row['ground_wave_db'])} |")
        seconds = max(row["btm_seconds"] for row in results["second_order"])
        print(f"\n(BTM takes up to {seconds:.0f} s per case)")

        results["close_screens"] = close_screens(folder)
        _print_close_screens(results["close_screens"])

        results["ceiling"] = ceiling(folder)
        _print_ceiling(results["ceiling"])

        results["thin_strip"] = thin_strip(folder)
        print(f"\n| a thin strip, level re free field (dB) | {header} |\n{rule}")
        for row in results["thin_strip"]:
            print(f"| {row['source']} to {row['receiver']}: BTM | {_row(row['btm_db'])} |")
            print(f"| BTM along one face | {_row(row['btm_along_one_face_db'])} |")
            alone = row["btm_without_the_paths_over_both_db"]
            print(f"| BTM without the paths over both | {_row(alone)} |")
            for mode in ("second_order", "first_order"):
                print(f"| {mode} - BTM | {_row(row[f'{mode}_minus_btm_db'])} |")

        results["strip_sight"] = strip_sight(folder)
        print(f"\n| the line of sight over a free strip (dB) | {header} |\n{rule}")
        for row in results["strip_sight"]:
            print(f"| {row['height']:g} m high: BTM 0.1 mm above | {_row(row['btm_db'][2])} |")
            print(f"| BTM's step across the line | {_row(row['btm_step_db'])} |")
            for mode in ("direct", "first_order", "second_order"):
                for key in (mode, f"{mode}, the edge diffraction alone"):
                    print(f"| {key}: step | {_row(row[key]['step_db'])} |")
                    print(f"| largest difference | {_row(row[key]['largest_difference_db'])} |")

        results["shadow_boundary"] = shadow_boundary(folder)
        print(f"\n| at a shadow boundary, level re free field (dB) | {header} |\n{rule}")
        for row in results["shadow_boundary"]:
            print(f"| {row['case']}, {row['where']}: BTM | {_row(row['btm_db'])} |")
            print(f"| engine in energy - BTM | {_row(row['in_energy_minus_btm_db'])} |")
            print(f"| engine by default - BTM | {_row(row['default_minus_btm_db'])} |")

        results["screen_corner"] = corner = screen_corner(folder)
        print(f"\n| a screen's corner, direct (dB) | {header} |\n{rule}")
        print(f"| largest step, engine | {_row(corner['largest_step_db'])} |")
        print(f"| largest step, BTM | {_row(corner['exact_largest_step_db'])} |")
        print(f"| largest difference | {_row(corner['largest_difference_db'])} |")

        results["other_corners"] = other_corners(folder)
        print(f"\n| corners of other angles, direct (dB) | {header} |\n{rule}")
        for row in results["other_corners"]:
            what = f"{row['angle_deg']:g} degrees"
            if row["turned_deg"]:
                what += f", turned {row['turned_deg']:g}"
            print(f"| {what}: largest step, engine | {_row(row['largest_step_db'])} |")
            print(f"| largest step, BTM | {_row(row['exact_largest_step_db'])} |")
            print(f"| largest difference | {_row(row['largest_difference_db'])} |")

        results["bent_screen"] = bent_screen(folder)
        print(f"\n| a bent screen, direct (dB) | {header} |\n{rule}")
        for row in results["bent_screen"]:
            for name in ("boundary", "boundary_near", "shadow"):
                line = row[name]
                what = f"{row['bend_deg']:+g} degrees, {name.replace('_', ' ')}"
                print(f"| {what}: largest step, engine | {_row(line['largest_step_db'])} |")
                print(f"| largest step, BTM | {_row(line['exact_largest_step_db'])} |")
                print(f"| largest difference | {_row(line['largest_difference_db'])} |")

        results["prism_corners"] = corners = prism_corners(folder)
        print(f"\n| the corners of solid prisms (dB) | {header} |\n{rule}")
        for row in corners:
            print(
                f"| {row['walk']} ({row['mode']}): largest step | {_row(row['largest_step_db'])} |"
            )
            print(f"| largest step, BTM | {_row(row['exact_largest_step_db'])} |")
            print(f"| largest difference | {_row(row['largest_difference_db'])} |")

        results["crossing"] = crossing(folder)
        print(f"\n| largest step over 1 mm (dB) | {header} |\n{rule}")
        for who, found in results["crossing"].items():
            for mode, row in found.items():
                print(f"| {who} through the top's plane, {mode} | {_row(row['largest_step_db'])} |")

        results["in_sight"] = in_sight(folder)
        low = " | ".join(f"{f:g}" for f in OCTAVE_BANDS_HZ[:IN_SIGHT_COHERENT])
        print(f"\n| just over a block's top, level re free field (dB) | {low} |")
        print("|---|" + "---:|" * IN_SIGHT_COHERENT)
        for row in results["in_sight"]:
            heights = f"{row['source'][2]:g} and {row['receiver'][2]:g} m above"
            print(f"| {row['width']:g} m wide, {heights}: BTM | {_row(row['btm_db'][:4])} |")
            without = row["btm_without_the_path_over_both_db"][:IN_SIGHT_COHERENT]
            print(f"| BTM without the path over both | {_row(without)} |")
            for mode in ("second_order", "first_order"):
                difference = row[f"{mode}_minus_btm_db"][:IN_SIGHT_COHERENT]
                print(f"| {mode} - BTM | {_row(difference)} |")

        results["round_the_plane"] = round_the_plane(folder)
        print(f"\n| within 1 mm of the plane of a block's top (dB) | {header} |\n{rule}")
        for row in results["round_the_plane"]:
            wide = f"{row['width']:g} m wide"
            print(f"| {wide}: largest difference | {_row(row['largest_difference_db'])} |")
            print(f"| with either in the plane | {_row(row['with_the_plane_db'])} |")
            print(f"| heights swapped | {_row(row['swapped_db'])} |")

        results["grazing_two_screens"] = grazing = grazing_two_screens(folder)
        print(f"\n| two screens at grazing incidence, level (dB) | {header} |\n{rule}")
        print(f"(two knife edges at high frequencies: {grazing['closed_form_db']:.2f} dB)")
        for row in grazing["rows"]:
            for mode in ("second_order", "first_order"):
                where = f"{row['height'] * 1000:+g} mm from the line of the tops, {mode}"
                print(f"| {where} | {_row(row[f'{mode}_db'])} |")

        results["behind_a_block"] = behind_a_block(folder)
        print(f"\n| behind a 10 m block, source above its top's plane (dB) | {low} |")
        print("|---|" + "---:|" * IN_SIGHT_COHERENT)
        for row in results["behind_a_block"]:
            heights = f"{row['source'][2]:g} m above and {-row['receiver'][2]:g} m below"
            print(f"| {heights}: BTM | {_row(row['btm_db'][:IN_SIGHT_COHERENT])} |")
            for mode in ("second_order", "first_order"):
                difference = row[f"{mode}_minus_btm_db"][:IN_SIGHT_COHERENT]
                print(f"| {mode} - BTM | {_row(difference)} |")

        results["in_a_row"] = in_a_row(folder)
        print(f"\n| edges in a row, both ends at one height, level (dB) | {header} |\n{rule}")
        for row in results["in_a_row"]:
            print(f"| {row['scene']}, {row['height'] * 1000:+g} mm from the tops' plane | |")
            for key, values in row.items():
                if key.endswith("_db"):
                    print(f"| {key[:-3].replace('_', ' ')} | {_row(values)} |")

        results["across_the_plane"] = across_the_plane(folder)
        print(
            f"\n| across the plane of a 10 m top, other end 0.5 m above (dB) | {header} |\n{rule}"
        )
        for row in results["across_the_plane"]:
            for part in ("below", "above", "step"):
                label = f"{row['crossing']} 1 mm {part}" if part != "step" else "step"
                print(f"| {label}, {row['bands']} | {_row(row[f'{part}_db'])} |")

    results["timing"] = timing()
    t = results["timing"]
    print(f"\nhall with alcoves, {t['edges']} edges, per source and listener:")
    for mode in ("none", "first_order", "second_order"):
        print(f"- {mode}: {t[mode + '_ms']:.0f} ms, {t[mode + '_paths']} paths")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
