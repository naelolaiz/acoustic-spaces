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
  and the same block absorbing 0.02, against BTM for the rigid one and the Weyl-van der
  Pol field's loss along its top (|F(w)|, by hand);
- two thin screens in a row, second order;
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
- the corners of a solid block, where three right-angled edges meet, first order: the
  listener walks past where the paths over its edges reach their ends, against BTM over
  each edge from end to end;
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

Every comparison is of the energy in each octave band, averaged over the engine's eight
frequencies in the band (BTM's at a shadow boundary over 96), relative to the direct
sound in free field over the same distance. The engine's levels are given twice: with
its paths and the terms of their coefficients added in energy, and in the 63 and 125 Hz
bands in pressure, as the engine adds them there by default
(``simulation.early.coherent_max_band_hz``). BTM takes the engine's speed of sound in
the scenes' air, at 20 °C.
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
from acoustic_spaces.air import speed_of_sound
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import EdgeDiffraction, band_frequencies, find_edges
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
    the paths start 1e-6 of the model's size off the edges (0.6 mm here), so a height of
    -1e-5 m is within their reach below the plane."""
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


#: The solid block of :func:`block_corner` (m): x, y and z from and to, open underneath
#: and reaching far down, so that only the edges of its top and of its upright corners
#: diffract within the time compared.
BLOCK = np.array([(0.0, 2.0), (-10.0, 0.0), (-EXTENT, 0.0)])


def _block_quads() -> list:
    (x0, x1), (y0, y1), (z0, z1) = BLOCK
    return [
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
        [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
    ]


def _block_edges() -> list[tuple[Edge, float]]:
    """The block's edges but those of its open bottom, each a right-angled wedge with
    its length."""
    out = []
    for axis in range(3):
        a, b = (i for i in range(3) if i != axis)
        for va, sa in ((BLOCK[a, 0], -1), (BLOCK[a, 1], 1)):
            for vb, sb in ((BLOCK[b, 0], -1), (BLOCK[b, 1], 1)):
                if (a == 2 and sa < 0) or (b == 2 and sb < 0):
                    continue
                origin = np.zeros(3)
                origin[axis], origin[a], origin[b] = BLOCK[axis, 0], va, vb
                direction, along, normal = np.zeros(3), np.zeros(3), np.zeros(3)
                direction[axis], along[b], normal[a] = 1.0, -sb, sa
                edge = Edge(origin, direction, along, normal, 1.5)
                out.append((edge, float(BLOCK[axis, 1] - BLOCK[axis, 0])))
    return out


def _block_geometric(source, receiver) -> np.ndarray:
    """The direct sound unless the block hides it, and the reflections off its faces."""
    pressure = np.zeros(len(K), complex)
    d = receiver - source
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (BLOCK[:, 0] - source) / d, (BLOCK[:, 1] - source) / d
    inside = np.where(d == 0.0, (source >= BLOCK[:, 0]) & (source <= BLOCK[:, 1]), True)
    low = np.max(np.where(d == 0.0, 0.0, np.minimum(a, b)), initial=0.0)
    high = np.min(np.where(d == 0.0, 1.0, np.maximum(a, b)), initial=1.0)
    if not (inside.all() and low < high):
        distance = float(np.linalg.norm(d))
        pressure += np.exp(-1j * K * distance) / distance
    for axis, side in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)):
        plane, sign = BLOCK[axis, side], 1.0 if side else -1.0
        if (source[axis] - plane) * sign <= 0.0 or (receiver[axis] - plane) * sign <= 0.0:
            continue
        image = source.copy()
        image[axis] = 2.0 * plane - source[axis]
        hit = image + (plane - image[axis]) / (receiver[axis] - image[axis]) * (receiver - image)
        others = [i for i in range(3) if i != axis]
        if all(BLOCK[i, 0] <= hit[i] <= BLOCK[i, 1] for i in others):
            distance = float(np.linalg.norm(receiver - image))
            pressure += np.exp(-1j * K * distance) / distance
    return pressure


def block_corner(folder: Path, step: float = 0.05) -> list[dict]:
    """The corners of a solid block, where three right-angled edges meet: the listener
    walks along y past the block's side, ``step`` at a time, at two heights above its
    top, past where the paths over its edges reach their ends. The engine (``direct``,
    every band in pressure) against BTM over each of the block's edges from end to end
    and the geometric paths; the sound diffracted from one edge to another is left out."""
    source = np.array([1.0, 3.0, 1.0])
    edges = _block_edges()
    ys = np.arange(-8.0, 2.0 + 1e-9, step) + 0.00317  # none exactly on a boundary
    out = []
    for height in (0.3, 1.8):
        found, exact = [], []
        for y in ys:
            receiver = np.array([4.0, y, height])
            room = _room(folder, "block", _block_quads(), source, receiver)
            found.append(engine(room, "direct", coherent=8)[1])
            pressure = _block_geometric(source, receiver)
            for edge, length in edges:
                if edge.frame(source)[1] > 1.5 * np.pi or edge.frame(receiver)[1] > 1.5 * np.pi:
                    continue  # one of them is behind the edge's faces
                pressure = pressure + btm_first(edge, source, receiver, ends=(0.0, length))
            exact.append(band_level(pressure, float(np.linalg.norm(receiver - source))))
        found, exact = np.array(found), np.array(exact)
        difference = np.abs(found - exact)
        steps = np.abs(np.diff(found, axis=0))
        out.append(
            {
                "height": height,
                "source": source.tolist(),
                "line": [[4.0, float(ys[0]), height], [4.0, float(ys[-1]), height]],
                "largest_difference_db": difference.max(axis=0).tolist(),
                "largest_difference_at_y": ys[difference.argmax(axis=0)].tolist(),
                "largest_step_db": steps.max(axis=0).tolist(),
                "largest_step_at_y": ys[1:][steps.argmax(axis=0)].tolist(),
                "exact_largest_step_db": np.abs(np.diff(exact, axis=0)).max(axis=0).tolist(),
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
            if "ground_wave_db" in row:
                print(f"| absorbing {ABSORBING:g} - BTM | {_row(row['absorbing_minus_btm_db'])} |")
                soft = row["absorbing_in_pressure_minus_btm_db"]
                print(f"| absorbing {ABSORBING:g} in pressure - BTM | {_row(soft)} |")
                print(f"| 20 lg abs F(w) over the top | {_row(row['ground_wave_db'])} |")
        seconds = max(row["btm_seconds"] for row in results["second_order"])
        print(f"\n(BTM takes up to {seconds:.0f} s per case)")

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

        results["block_corner"] = block = block_corner(folder)
        print(f"\n| a block's corners, direct (dB) | {header} |\n{rule}")
        for row in block:
            steps = _row(row["largest_step_db"])
            print(f"| {row['height']} m up: largest step, engine | {steps} |")
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
