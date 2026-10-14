"""Edge diffraction: sound that bends around the edges of the room, after the uniform
theory of diffraction (UTD) for a rigid wedge (Kouyoumjian and Pathak, 1974).

The edges that diffract are those where two faces meet at an angle that opens more than
180 degrees towards the room (the corner of a pillar, a balcony front, the end of a
wall) and the free edges of open surfaces (panels and screens). Corners of the room
itself open less than 180 degrees towards the room, cast no shadow and are left out, as
are the slight bends between the facets of a curved surface.

Sound from a source reaches a listener over an edge along the shortest path that touches
it, at the point where the path makes equal angles with the edge on both sides. The
diffracted wave splits into terms that each belong to one geometric wave: the direct
sound, or its reflection off either face of the edge. Where that wave arrives, its terms
are added to it coherently, frequency by frequency, which changes its energy per band; on
the far side of its boundary, where the edge cuts the wave off, the terms make a new path.
At the boundary itself both sides carry a quarter of the wave's energy, so levels stay
continuous where the direct sound or a reflection disappears behind an edge.

With ``first_order`` diffraction the same is done for the first-order reflections: sound
that reflects off a plane and then bends over an edge, and sound that bends over an edge
and then reflects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import TraceModel, source_weights
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

#: Faces that bend away from each other by less than this (degrees) make no
#: diffracting edge: the facets of a curved surface, for example.
MIN_BEND_DEG = 15.0
#: Frequencies per octave band at which the diffracted waves are evaluated, one in the
#: middle of each equal slice of the band; energies are averaged over the band.
FREQUENCIES_PER_BAND = 8
#: A diffracted path is dropped when its energy is this far below that of the direct
#: sound in free field, in every band.
MIN_RELATIVE_ENERGY = 1e-4
#: Diffraction settings: none, of the direct sound only, or of the direct sound and
#: first-order reflections.
DIFFRACTION_MODES = ("none", "direct", "first_order")

#: Below this argument the Fresnel integrals use their power series, above it their
#: asymptotic expansion.
_SERIES_LIMIT = 3.5
#: Distances (relative to the model size) for points on an edge or a plane.
_TOLERANCE = 1e-9
#: Rays start this far (relative to the model size) off the edge, into the open side.
_OFFSET = 1e-6
#: Angles (radians) this close to a shadow or reflection boundary count as on it; which
#: side is then decided by whether the specular path was found.
_ON_BOUNDARY = 1e-9


# -- the transition function --------------------------------------------------------


def fresnel_tail(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``1/2 - C(z)`` and ``1/2 - S(z)`` for z >= 0, where C and S are the Fresnel
    integrals of cos(pi t^2 / 2) and sin(pi t^2 / 2) from 0 to z.

    Power series below z = 3.5 and the asymptotic expansion above, good to about 1e-8
    where they meet and better elsewhere; computing the tails directly keeps their
    accuracy for large z.
    """
    z = np.asarray(z, dtype=float)
    c = np.empty_like(z)
    s = np.empty_like(z)
    small = z < _SERIES_LIMIT
    if np.any(small):
        x = z[small]
        t = 0.5 * np.pi * x**2
        # C = x sum over even m, S = x sum over odd m, of (-1)^(m // 2) t^m / (m! (2m + 1)).
        terms = int(np.ceil(_series_terms(float(t.max()))))
        term = np.ones_like(x)
        sum_c = np.ones_like(x)
        sum_s = np.zeros_like(x)
        for m in range(1, terms):
            term *= t * (1.0 / m)
            coefficient = (-1.0 if (m // 2) % 2 else 1.0) / (2 * m + 1)
            if m % 2:
                sum_s += coefficient * term
            else:
                sum_c += coefficient * term
        c[small] = 0.5 - x * sum_c
        s[small] = 0.5 - x * sum_s
    large = ~small
    if np.any(large):
        f, g = _auxiliary(z[large])
        angle = 0.5 * np.pi * z[large] ** 2
        c[large] = g * np.cos(angle) - f * np.sin(angle)
        s[large] = f * np.cos(angle) + g * np.sin(angle)
    return c, s


def _series_terms(t: float) -> int:
    """Terms of the power series until t^m / m! falls below 1e-18 (at least 5)."""
    m, term = 0, 1.0
    while m < 5 or term >= 1e-18:
        m += 1
        term *= t / m
    return m + 1


def _auxiliary(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The auxiliary functions f(z) and g(z) of the Fresnel integrals (Abramowitz and
    Stegun 7.3.27-28), from their asymptotic series, for z >= 3.5."""
    y = np.pi * z**2
    inv = 1.0 / y**2
    f_sum, g_sum = np.ones_like(z), 1.0 / y
    f_term, g_term = np.ones_like(z), 1.0 / y
    for m in range(1, 7):
        f_term = -f_term * (4 * m - 3) * (4 * m - 1) * inv
        g_term = -g_term * (4 * m - 1) * (4 * m + 1) * inv
        f_sum += f_term
        g_sum += g_term
    return f_sum / (np.pi * z), g_sum / (np.pi * z)


def transition(x: np.ndarray) -> np.ndarray:
    """The UTD transition function F(X) = 2j sqrt(X) e^(jX) times the integral of
    e^(-j t^2) from sqrt(X) to infinity, for X >= 0. It rises from 0 at X = 0 to 1 for
    large X, where the diffracted wave no longer depends on the shadow boundary."""
    x = np.asarray(x, dtype=float)
    z = np.sqrt(2.0 * x / np.pi)
    out = np.empty(x.shape, dtype=complex)
    small = z < _SERIES_LIMIT
    if np.any(small):
        c, s = fresnel_tail(z[small])
        root = np.sqrt(x[small])
        tail = math.sqrt(0.5 * np.pi) * (c - 1j * s)
        out[small] = 2j * root * np.exp(1j * x[small]) * tail
    large = ~small
    if np.any(large):
        f, g = _auxiliary(z[large])
        out[large] = np.sqrt(2.0 * np.pi * x[large]) * (f + 1j * g)
    return out


# -- the wedge coefficient ----------------------------------------------------------


def wedge_terms(
    n: np.ndarray,
    phi: np.ndarray,
    phi_source: np.ndarray,
    kl: np.ndarray,
    lit: np.ndarray | None = None,
) -> np.ndarray:
    """The four cotangent terms of the UTD wedge coefficient, without its prefactor.

    ``n`` is the open angle of the wedge over pi; ``phi`` and ``phi_source`` are the
    angles of the listener and source around the edge, from face 0; ``kl`` is the wave
    number times the distance parameter L. Arrays broadcast; the result has a last axis
    of 4: the two terms of the incident wave, then those of the reflections off face n
    and off face 0. Each term is singular on its shadow or reflection boundary, where
    it is replaced by its limit; exactly on a boundary, ``lit`` (a last axis of 3: the
    incident wave, the reflections off face 0 and off face n) says which side's limit
    to take (default: the lit side).
    """
    difference, total = phi - phi_source, phi + phi_source
    eps = np.stack(
        [
            _boundary_distance(n, difference, 1.0),
            _boundary_distance(n, difference, -1.0),
            _boundary_distance(n, total, 1.0),
            _boundary_distance(n, total, -1.0),
        ],
        axis=-1,
    )
    if lit is None:
        side = 1.0
    else:
        side = np.where(np.asarray(lit)[..., [0, 0, 2, 1]], 1.0, -1.0)
    return _cot_transition(np.asarray(n)[..., None], eps, np.asarray(kl)[..., None], side)


def boundary_angles(n: np.ndarray, phi: np.ndarray, phi_source: np.ndarray) -> np.ndarray:
    """(..., 3) angles from the listener to the boundaries of the incident wave and of
    the reflections off face 0 and face n, positive on their lit sides."""
    total = phi + phi_source
    return np.stack(
        [np.pi - np.abs(phi - phi_source), np.pi - total, total - (2.0 * n - 1.0) * np.pi],
        axis=-1,
    )


def _boundary_distance(n: np.ndarray, beta: np.ndarray, sigma: float) -> np.ndarray:
    """The angle epsilon between a term's argument and its nearest singularity: the
    term is cot(eps / 2n) F(2 kL sin^2(eps / 2)), positive on the lit side."""
    N = np.round((beta + sigma * np.pi) / (2.0 * np.pi * n))
    return np.pi + sigma * (beta - 2.0 * np.pi * n * N)


def _cot_transition(n, eps, kl, side) -> np.ndarray:
    n, eps, kl, side = np.broadcast_arrays(n, eps, kl, side)
    x = 2.0 * kl * np.sin(0.5 * eps) ** 2
    near = np.abs(eps) < _ON_BOUNDARY
    with np.errstate(divide="ignore", invalid="ignore"):
        value = transition(x) / np.tan(eps / (2.0 * n))
    # Near the boundary: n (sqrt(2 pi kL) sgn(eps) - 2 kL eps e^(j pi/4)) e^(j pi/4).
    root = np.sqrt(2.0 * np.pi * kl) * side
    limit = n * (root - 2.0 * kl * eps * np.exp(0.25j * np.pi)) * np.exp(0.25j * np.pi)
    return np.where(near, limit, value)


def wedge_prefactor(n: np.ndarray, k: np.ndarray, sin_beta: np.ndarray) -> np.ndarray:
    """-e^(-j pi/4) / (2n sqrt(2 pi k) sin beta0), the factor in front of the terms."""
    return -np.exp(-0.25j * np.pi) / (2.0 * n * np.sqrt(2.0 * np.pi * k) * sin_beta)


def band_frequencies(per_band: int = FREQUENCIES_PER_BAND) -> tuple[np.ndarray, np.ndarray]:
    """(bands, per_band) frequencies at the middle of equal slices of each octave band
    (from its centre over the square root of 2 to its centre times it), and the width
    of the slices per band (bands,)."""
    centre = np.asarray(OCTAVE_BANDS_HZ, dtype=float)
    low, width = centre / math.sqrt(2.0), centre / math.sqrt(2.0) / per_band
    return low[:, None] + width[:, None] * (np.arange(per_band) + 0.5), width


# -- edges --------------------------------------------------------------------------


@dataclass
class Edges:
    """The diffracting edges of a model, as straight segments.

    Each edge is a wedge between face 0 and face n that is open over ``wedge`` times
    pi radians (between 1 and 2; exactly 2 for the free edge of a surface, where face 0
    and face n are the two sides of the same face). Angles around an edge are measured
    from face 0 towards its ``normal``, through the open side.
    """

    start: np.ndarray  # (E, 3)
    direction: np.ndarray  # (E, 3) unit vectors along the edge
    length: np.ndarray  # (E,)
    along: np.ndarray  # (E, 3) unit vectors in face 0, across the edge, away from it
    normal: np.ndarray  # (E, 3) unit normals of face 0 on the open side
    wedge: np.ndarray  # (E,) open angle over pi
    face: np.ndarray  # (E, 2) face 0 and face n
    plane: np.ndarray  # (E, 2) their planes
    reflection: np.ndarray  # (E, 2, bands) specular pressure reflection factors

    def __len__(self) -> int:
        return len(self.length)

    @property
    def end(self) -> np.ndarray:
        return self.start + self.direction * self.length[:, None]

    def bisector(self) -> np.ndarray:
        """(E, 3) unit vectors from each edge into the middle of its open side."""
        half = 0.5 * np.pi * self.wedge[:, None]
        return np.cos(half) * self.along + np.sin(half) * self.normal

    def frame(self, index: np.ndarray, points: np.ndarray):
        """Position along edge ``index`` of each point, distance from it and angle around
        it from face 0, and whether the point lies in the open side of the wedge."""
        v = points - self.start[index]
        t = np.einsum("ij,ij->i", v, self.direction[index])
        w = v - t[:, None] * self.direction[index]
        rho = np.linalg.norm(w, axis=1)
        phi = np.arctan2(
            np.einsum("ij,ij->i", w, self.normal[index]),
            np.einsum("ij,ij->i", w, self.along[index]),
        )
        phi = np.where(phi < -1e-9, phi + 2.0 * np.pi, np.maximum(phi, 0.0))
        inside = phi <= np.pi * self.wedge[index] + 1e-9
        return t, rho, phi, inside


def find_edges(model: TraceModel, min_bend_deg: float = MIN_BEND_DEG) -> Edges:
    """The edges of ``model`` that diffract (see the module notes); collinear pieces of
    one edge between the same two planes are joined."""
    mesh = model.room.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = mesh.faces
    first = faces.reshape(-1)
    second = faces[:, [1, 2, 0]].reshape(-1)
    opposite = faces[:, [2, 0, 1]].reshape(-1)
    owner = np.repeat(np.arange(len(faces)), 3)
    lo, hi = np.minimum(first, second), np.maximum(first, second)
    order = np.argsort(lo * len(vertices) + hi, kind="stable")
    _, starts, uses = np.unique(
        (lo * len(vertices) + hi)[order], return_index=True, return_counts=True
    )
    normals = model.plane_normal[model.face_plane]  # into the room when oriented

    def geometry(half: np.ndarray):
        p0, p1 = vertices[lo[half]], vertices[hi[half]]
        length = np.linalg.norm(p1 - p0, axis=1)
        e = (p1 - p0) / np.maximum(length, 1e-300)[:, None]
        v = vertices[opposite[half]] - p0
        a = v - np.einsum("ij,ij->i", v, e)[:, None] * e
        a /= np.maximum(np.linalg.norm(a, axis=1), 1e-300)[:, None]
        return p0, p1, e, a

    parts = []
    # Two faces: a wedge, if it opens far enough beyond flat towards the room.
    pair = starts[uses == 2]
    h0, h1 = order[pair], order[pair + 1]
    swap = model.face_plane[owner[h0]] > model.face_plane[owner[h1]]
    h0, h1 = np.where(swap, h1, h0), np.where(swap, h0, h1)
    p0, p1, e, a0 = geometry(h0)
    _, _, _, a1 = geometry(h1)
    face0, face1 = owner[h0], owner[h1]
    theta = np.arccos(np.clip(np.einsum("ij,ij->i", a0, a1), -1.0, 1.0))
    side = np.einsum("ij,ij->i", normals[face0], a1)
    keep = theta <= np.pi - math.radians(min_bend_deg)
    if model.oriented:
        keep &= side < 0.0  # face n lies behind face 0: the solid is inside the angle
    normal = np.where((side > 0.0)[:, None], -normals[face0], normals[face0])
    parts.append((p0, p1, e, a0, normal, (2.0 * np.pi - theta) / np.pi, face0, face1, keep))
    # One face: the free edge of a surface, a wedge open all round.
    single = order[starts[uses == 1]]
    p0, p1, e, a0 = geometry(single)
    face0 = owner[single]
    parts.append(
        (
            p0,
            p1,
            e,
            a0,
            normals[face0],
            np.full(len(single), 2.0),
            face0,
            face0,
            np.ones(len(single), dtype=bool),
        )
    )
    p0, p1, e, along, normal, wedge, face0, face1 = (
        np.concatenate([part[i][part[8]] for part in parts]) for i in range(8)
    )
    return _join(model, p0, p1, e, along, normal, wedge, face0, face1)


def _join(model, p0, p1, e, along, normal, wedge, face0, face1) -> Edges:
    """Join collinear pieces between the same faces' planes and materials."""
    bands = model.absorption.shape[1]
    if len(p0) == 0:
        return Edges(
            start=np.zeros((0, 3)),
            direction=np.zeros((0, 3)),
            length=np.zeros(0),
            along=np.zeros((0, 3)),
            normal=np.zeros((0, 3)),
            wedge=np.zeros(0),
            face=np.zeros((0, 2), dtype=np.int64),
            plane=np.zeros((0, 2), dtype=np.int64),
            reflection=np.zeros((0, 2, bands)),
        )
    size = float(np.ptp(model.room.mesh.vertices, axis=0).max()) or 1.0
    tol = 1e-6 * size
    # One direction per line: its largest component positive.
    flip = np.take_along_axis(e, np.abs(e).argmax(axis=1)[:, None], 1)[:, 0] < 0
    e = np.where(flip[:, None], -e, e)
    p0, p1 = np.where(flip[:, None], p1, p0), np.where(flip[:, None], p0, p1)
    t0 = np.einsum("ij,ij->i", p0, e)
    t1 = np.einsum("ij,ij->i", p1, e)
    foot = p0 - t0[:, None] * e  # the point of the line nearest the origin
    materials = model.room.face_materials
    key = np.column_stack(
        [
            model.face_plane[face0],
            model.face_plane[face1],
            materials[face0],
            materials[face1],
            np.round(np.column_stack([e, along, normal, wedge]) / 1e-6),
            np.round(foot / tol),
        ]
    ).astype(np.int64)
    group = np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)
    chosen, begin, finish = [], [], []
    for i in np.lexsort((t0, group)):
        if chosen and group[chosen[-1]] == group[i] and t0[i] <= finish[-1] + tol:
            finish[-1] = max(finish[-1], t1[i])
        else:
            chosen.append(i)
            begin.append(t0[i])
            finish.append(t1[i])
    chosen = np.array(chosen, dtype=np.int64)
    begin, finish = np.array(begin), np.array(finish)
    faces = np.column_stack([face0[chosen], face1[chosen]]).astype(np.int64)
    return Edges(
        start=foot[chosen] + begin[:, None] * e[chosen],
        direction=e[chosen],
        length=finish - begin,
        along=along[chosen],
        normal=normal[chosen],
        wedge=wedge[chosen],
        face=faces,
        plane=model.face_plane[faces].astype(np.int64),
        reflection=np.sqrt((1.0 - model.absorption[faces]) * (1.0 - model.scattering[faces])),
    )


# -- diffracted paths ---------------------------------------------------------------


class EdgeDiffraction:
    """Diffraction of one source's sound at the edges of a model, for any number of
    listeners: :meth:`apply` adds it to the specular paths of one listener.

    ``mode`` is ``"direct"`` (the direct sound only) or ``"first_order"`` (also the
    first-order reflections, before or after the edge). ``air`` is the energy
    attenuation per metre per band; ``directivity`` and ``orientation`` are the source's,
    as for :func:`~acoustic_spaces.engine.early.find_paths`.
    """

    def __init__(
        self,
        model: TraceModel,
        source: np.ndarray,
        *,
        speed_of_sound: float,
        air: np.ndarray,
        directivity: Directivity | None = None,
        orientation: np.ndarray | None = None,
        mode: str = "first_order",
        edges: Edges | None = None,
    ) -> None:
        if mode not in DIFFRACTION_MODES[1:]:
            raise ValueError(f"unknown diffraction mode {mode!r}")
        self.model = model
        self.edges = find_edges(model) if edges is None else edges
        self.source = np.asarray(source, dtype=float)
        self.speed_of_sound = float(speed_of_sound)
        self.air = np.asarray(air, dtype=float)
        self.directivity, self.orientation = directivity, orientation
        self.mode = mode
        self.size = float(np.ptp(model.room.mesh.vertices, axis=0).max()) or 1.0
        frequencies, self.slice_hz = band_frequencies()
        self.wavenumber = 2.0 * np.pi * frequencies / self.speed_of_sound  # (B, M)

        # The source and its images in the planes it faces; the source side of a
        # diffracted path is one of them, the listener side the listener or its image.
        planes = self._facing(self.source) if mode == "first_order" else np.zeros(0, np.int64)
        self.source_planes = np.concatenate([[-1], planes]).astype(np.int64)
        self.source_points = np.vstack(
            [self.source, model.mirror(np.broadcast_to(self.source, (len(planes), 3)), planes)]
        )
        edge, side = self._pairs(len(self.source_planes))
        t, rho, phi, inside = self.edges.frame(edge, self.source_points[side])
        keep = inside & (rho > _TOLERANCE * self.size)
        keep &= self.source_planes[side] != self.edges.plane[edge, 0]
        keep &= self.source_planes[side] != self.edges.plane[edge, 1]
        self._source_side = (edge[keep], side[keep], t[keep], rho[keep], phi[keep])

    def _facing(self, point: np.ndarray) -> np.ndarray:
        """Planes that ``point`` lies in front of (on either side when not oriented)."""
        distance = self.model.plane_normal @ point - self.model.plane_offset
        tolerance = _TOLERANCE * self.size
        front = distance > tolerance if self.model.oriented else np.abs(distance) > tolerance
        return np.flatnonzero(front)

    def _pairs(self, sides: int) -> tuple[np.ndarray, np.ndarray]:
        edge = np.repeat(np.arange(len(self.edges)), sides)
        side = np.tile(np.arange(sides), len(self.edges))
        return edge, side

    def apply(
        self, listener: np.ndarray, paths: EarlyReflections, max_delay_s: float
    ) -> EarlyReflections:
        """``paths`` (the specular paths from the source to ``listener``) with the
        diffracted sound up to ``max_delay_s``: specular paths near the edges' shadow
        and reflection boundaries change their energy, and paths over the edges into the
        shadows are added."""
        listener = np.asarray(listener, dtype=float)
        if len(self.edges) == 0:
            return paths
        combos = self._combos(listener, max_delay_s)
        if combos is None:
            return paths
        return self._evaluate(listener, paths, combos, max_delay_s)

    def _combos(self, listener: np.ndarray, max_delay_s: float):
        """Source and listener sides of every edge whose shortest path over the edge
        touches it and is unobstructed."""
        edges = self.edges
        edge, side, t_s, rho_s, phi_s = self._source_side
        tolerance = _TOLERANCE * self.size

        # Listener sides: the listener, for every source side; and its images, for the
        # source itself.
        planes = self._facing(listener) if self.mode == "first_order" else np.zeros(0, np.int64)
        all_edges = np.arange(len(edges))
        t_l, rho_l, phi_l, in_l = edges.frame(all_edges, np.broadcast_to(listener, (len(edges), 3)))
        ok = in_l[edge] & (rho_l[edge] > tolerance)
        parts = [
            (
                edge[ok],
                self.source_points[side[ok]],
                np.broadcast_to(listener, (int(ok.sum()), 3)),
                self.source_planes[side[ok]],
                np.where(self.source_planes[side[ok]] >= 0, 1, 0),
                t_s[ok],
                rho_s[ok],
                phi_s[ok],
                t_l[edge[ok]],
                rho_l[edge[ok]],
                phi_l[edge[ok]],
            )
        ]
        if len(planes):
            direct = self.source_planes[side] < 0
            e0, ts0, rs0, ps0 = edge[direct], t_s[direct], rho_s[direct], phi_s[direct]
            images = self.model.mirror(np.broadcast_to(listener, (len(planes), 3)), planes)
            pe = np.repeat(e0, len(planes))
            pp = np.tile(np.arange(len(planes)), len(e0))
            t_i, rho_i, phi_i, in_i = edges.frame(pe, images[pp])
            plane = planes[pp]
            ok = in_i & (rho_i > tolerance)
            ok &= (plane != edges.plane[pe, 0]) & (plane != edges.plane[pe, 1])
            reps = np.repeat(np.arange(len(e0)), len(planes))[ok]
            parts.append(
                (
                    pe[ok],
                    np.broadcast_to(self.source, (int(ok.sum()), 3)),
                    images[pp[ok]],
                    plane[ok],
                    np.full(int(ok.sum()), 2),
                    ts0[reps],
                    rs0[reps],
                    ps0[reps],
                    t_i[ok],
                    rho_i[ok],
                    phi_i[ok],
                )
            )
        (edge, src, rcv, plane, kind, t_s, rho_s, phi_s, t_r, rho_r, phi_r) = (
            np.concatenate([p[i] for p in parts]) for i in range(11)
        )

        # The point on the edge where the path makes equal angles with it on both sides.
        t_k = t_s + (t_r - t_s) * rho_s / (rho_s + rho_r)
        r1 = np.hypot(rho_s, t_k - t_s)
        r2 = np.hypot(rho_r, t_r - t_k)
        ok = (t_k >= 0.0) & (t_k < edges.length[edge])
        ok &= (r1 + r2) / self.speed_of_sound <= max_delay_s
        if not ok.any():
            return None
        sel = np.flatnonzero(ok)
        edge, src, rcv, plane, kind = edge[sel], src[sel], rcv[sel], plane[sel], kind[sel]
        t_k, r1, r2 = t_k[sel], r1[sel], r2[sel]
        phi_s, phi_r, rho_s = phi_s[sel], phi_r[sel], rho_s[sel]
        point = edges.start[edge] + t_k[:, None] * edges.direction[edge]

        # Unobstructed, with the reflection on the expected plane before or after the
        # edge. Rays start just off the edge, in the middle of its open side.
        start = point + _OFFSET * self.size * edges.bisector()[edge]
        reflected = np.full(len(edge), -1, dtype=np.int64)
        bounce = np.zeros((len(edge), 3))
        ok = np.ones(len(edge), dtype=bool)
        for end, image_kind in ((self.source, 1), (listener, 2)):
            ends = np.broadcast_to(end, (len(edge), 3))
            via = kind == image_kind
            plain = ~via
            if plain.any():
                ok[plain] &= _clear(self.model, start[plain], ends[plain])
            if via.any():
                target = src[via] if image_kind == 1 else rcv[via]
                hit, where, face = _reflect(self.model, start[via], target, plane[via], ends[via])
                ok[via] &= hit
                bounce[via], reflected[via] = where, face
        if not ok.any():
            return None
        sel = np.flatnonzero(ok)
        return {
            "edge": edge[sel],
            "source_side": src[sel],
            "plane": plane[sel],
            "kind": kind[sel],
            "point": point[sel],
            "bounce": bounce[sel],
            "face": reflected[sel],
            "r1": r1[sel],
            "r2": r2[sel],
            "rho_s": rho_s[sel],
            "phi_s": phi_s[sel],
            "phi_r": phi_r[sel],
        }

    def _evaluate(self, listener, paths: EarlyReflections, c: dict, max_delay_s: float):
        model, edges = self.model, self.edges
        edge, kind = c["edge"], c["kind"]
        r1, r2 = c["r1"], c["r2"]
        n = edges.wedge[edge]
        k = self.wavenumber[None, :, :]  # (1, B, M)
        distance = r1 + r2
        bands = model.absorption.shape[1]

        # The pressure of each term: the wedge coefficient, spreading, air, the
        # source's directivity and the reflections on the way.
        sin_beta = np.clip(c["rho_s"] / r1, 1e-12, 1.0)
        length_param = r1 * r2 / distance * sin_beta**2
        # Which side of each wave's boundary the listener is on, and that wave's
        # specular path, if it was found. A term counts on the lit side of its boundary
        # when the path is there, and on the dark side when it is not; where the two
        # disagree (the path is blocked by something else, lies beyond the highest
        # order, or passes round the end of the edge) the term is left out.
        found = self._specular(listener, paths, c)
        angle = boundary_angles(n, c["phi_r"], c["phi_s"])
        lit = (angle > _ON_BOUNDARY) | ((np.abs(angle) <= _ON_BOUNDARY) & (found >= 0))
        terms = wedge_terms(
            n[:, None, None],
            c["phi_r"][:, None, None],
            c["phi_s"][:, None, None],
            k * length_param[:, None, None],
            lit[:, None, None, :],
        )
        first = np.where((kind == 1)[:, None], c["bounce"], c["point"])
        departure = first - self.source
        departure /= np.maximum(np.linalg.norm(departure, axis=1), 1e-300)[:, None]
        weight = source_weights(self.directivity, self.orientation, departure)  # (C, B)
        amplitude = np.sqrt(weight * np.exp(-np.outer(distance, self.air)) / (4.0 * np.pi))
        amplitude /= np.sqrt(r1 * r2 * distance)[:, None]
        on_plane = c["face"] >= 0
        if on_plane.any():
            face = c["face"][on_plane]
            amplitude[on_plane] *= np.sqrt(
                (1.0 - model.absorption[face]) * (1.0 - model.scattering[face])
            )
        common = (
            amplitude[:, :, None]
            * wedge_prefactor(n[:, None, None], k, sin_beta[:, None, None])
            * np.exp(-1j * k * distance[:, None, None])
        )  # (C, B, M)
        reflection = edges.reflection[edge]  # (C, 2, B)
        waves = (
            np.stack(
                [
                    terms[..., 0] + terms[..., 1],
                    terms[..., 3] * reflection[:, 0, :, None],
                    terms[..., 2] * reflection[:, 1, :, None],
                ],
                axis=1,
            )
            * common[:, None]
        )  # (C, 3, B, M): incident, off face 0, off face n

        field = np.sum(waves * (lit == (found >= 0))[:, :, None, None], axis=1)  # (C, B, M)
        energy = np.mean(np.abs(field) ** 2, axis=2)  # (C, B)

        # The diffracted wave arrives on its own, but interferes with the specular
        # paths of its lit terms, which arrive at almost the same time near the
        # boundaries: those paths take the cross terms. On the boundary a path's energy
        # and the diffracted wave's then add up to a quarter of the path's on its own,
        # as the diffracted wave alone carries on the dark side.
        out = paths
        rows, groups = np.nonzero(lit & (found >= 0))
        if len(rows):
            targets = found[rows, groups]
            phase = np.exp(-1j * self.wavenumber[None] * paths.distance_m[targets, None, None])
            specular = np.sqrt(paths.energy[targets])[:, :, None] * phase
            # Each sample stands for its slice of the band: averaged over the slice, the
            # interference of two waves that arrive delta metres apart fades by sinc.
            delta = distance[rows] - paths.distance_m[targets]
            fade = np.sinc(np.outer(delta / self.speed_of_sound, self.slice_hz))
            cross = 2.0 * fade * np.mean(np.real(specular * np.conj(field[rows])), axis=2)
            total = paths.energy.copy()
            np.add.at(total, targets, cross)
            out = _replace_energy(paths, np.maximum(total, 0.0))

        reference = 1.0 / (4.0 * np.pi * float(np.sum((listener - self.source) ** 2)))
        keep = energy.max(axis=1) >= MIN_RELATIVE_ENERGY * reference
        keep &= distance / self.speed_of_sound <= max_delay_s
        if not keep.any():
            return out
        idx = np.flatnonzero(keep)
        last = np.where((kind == 2)[:, None], c["bounce"], c["point"])[idx]
        arrival = last - listener
        arrival /= np.maximum(np.linalg.norm(arrival, axis=1), 1e-300)[:, None]
        points, faces = [], []
        for i in idx:
            middle = [c["point"][i]]
            if kind[i] == 1:
                middle = [c["bounce"][i], c["point"][i]]
            elif kind[i] == 2:
                middle = [c["point"][i], c["bounce"][i]]
            points.append(np.vstack([self.source, *middle, listener]))
            faces.append(np.array([c["face"][i]] if kind[i] else [], dtype=np.int64))
        new = EarlyReflections(
            order=np.where(kind[idx] > 0, 1, 0).astype(np.int64),
            delay_s=distance[idx] / self.speed_of_sound,
            distance_m=distance[idx],
            direction=arrival,
            energy=energy[idx].reshape(len(idx), bands),
            points=points,
            faces=faces,
            edge=edge[idx].astype(np.int64),
        )
        return EarlyReflections.concatenate([out, new], bands)

    def _specular(self, listener: np.ndarray, paths: EarlyReflections, c: dict) -> np.ndarray:
        """(C, 3) index in ``paths`` of the specular path of each term's wave (the
        incident wave and the reflections off face 0 and face n), or -1."""
        model, edges = self.model, self.edges
        count = len(c["edge"])
        out = np.full((count, 3), -1, dtype=np.int64)
        specular = np.flatnonzero(paths.edge < 0)
        if len(specular) == 0:
            return out
        images = listener + paths.direction[specular] * paths.distance_m[specular, None]
        orders = paths.order[specular]
        kind, plane = c["kind"], c["plane"]
        base = np.broadcast_to(self.source, (count, 3)).copy()
        has_plane = kind > 0
        base[has_plane] = model.mirror(base[has_plane], plane[has_plane])
        tolerance = 1e-6 * self.size
        for column, wedge_face in ((0, None), (1, 0), (2, 1)):
            if wedge_face is None:
                image, order = base, has_plane.astype(np.int64)
            else:
                face_plane = edges.plane[c["edge"], wedge_face]
                image = base.copy()
                # Reflection off the wedge face after the plane (source side image),
                # or before it (listener side image).
                after = kind != 2
                image[after] = model.mirror(base[after], face_plane[after])
                before = kind == 2
                if before.any():
                    first = model.mirror(
                        np.broadcast_to(self.source, (int(before.sum()), 3)), face_plane[before]
                    )
                    image[before] = model.mirror(first, plane[before])
                order = has_plane.astype(np.int64) + 1
            out[:, column] = _match(image, order, images, orders, specular, tolerance)
        return out


def _match(queries, query_order, images, orders, index, tolerance, chunk: int = 4096):
    """For each query point, the index of the image of the same order within
    ``tolerance`` of it, or -1."""
    out = np.full(len(queries), -1, dtype=np.int64)
    for begin in range(0, len(queries), chunk):
        q = queries[begin : begin + chunk]
        d = np.linalg.norm(q[:, None, :] - images[None, :, :], axis=2)
        d = np.where(query_order[begin : begin + chunk, None] == orders[None, :], d, np.inf)
        best = np.argmin(d, axis=1)
        hit = d[np.arange(len(q)), best] <= tolerance
        out[begin : begin + chunk] = np.where(hit, index[best], -1)
    return out


def _clear(model: TraceModel, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Whether nothing lies between each start and end point."""
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1e-300)[:, None]
    _, face = model.intersect(start, direction, length * (1.0 - 1e-9), np.full(len(start), -1))
    return face < 0


def _reflect(model: TraceModel, start, image, plane, end):
    """Rays from ``start`` towards ``image`` (the mirror image of ``end`` in ``plane``):
    whether they first hit a face of that plane and see ``end`` from there; the points
    and faces they hit."""
    delta = image - start
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1e-300)[:, None]
    t, face = model.intersect(start, direction, length, np.full(len(start), -1))
    hit = (face >= 0) & (model.face_plane[np.maximum(face, 0)] == plane)
    where = start + direction * np.where(hit, t, 0.0)[:, None]
    delta = end - where
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1e-300)[:, None]
    _, blocked = model.intersect(where, direction, length * (1.0 - 1e-9), np.where(hit, face, -1))
    return hit & (blocked < 0), where, np.where(hit, face, -1)


def _replace_energy(paths: EarlyReflections, energy: np.ndarray) -> EarlyReflections:
    return EarlyReflections(
        order=paths.order,
        delay_s=paths.delay_s,
        distance_m=paths.distance_m,
        direction=paths.direction,
        energy=energy,
        points=paths.points,
        faces=paths.faces,
        edge=paths.edge,
    )


def diffract(
    model: TraceModel,
    source: np.ndarray,
    listener: np.ndarray,
    paths: EarlyReflections,
    *,
    speed_of_sound: float,
    air: np.ndarray,
    max_delay_s: float,
    directivity: Directivity | None = None,
    orientation: np.ndarray | None = None,
    mode: str = "first_order",
    edges: Edges | None = None,
) -> EarlyReflections:
    """``paths`` from one source to one listener with the sound diffracted at the
    model's edges added; see :class:`EdgeDiffraction`."""
    return EdgeDiffraction(
        model,
        source,
        speed_of_sound=speed_of_sound,
        air=air,
        directivity=directivity,
        orientation=orientation,
        mode=mode,
        edges=edges,
    ).apply(listener, paths, max_delay_s)


__all__ = [
    "DIFFRACTION_MODES",
    "FREQUENCIES_PER_BAND",
    "MIN_BEND_DEG",
    "MIN_RELATIVE_ENERGY",
    "EdgeDiffraction",
    "Edges",
    "band_frequencies",
    "diffract",
    "find_edges",
    "fresnel_tail",
    "transition",
    "wedge_prefactor",
    "wedge_terms",
]
