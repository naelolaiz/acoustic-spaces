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

Each specular path interferes with the terms of its own wave only, as it adds to the
other paths in energy. In the diffracted path the terms of all the waves add up
coherently, as they must deep in the shadow, except near each wave's boundary, where
they stand in for the geometric wave they continue and add in energy too: a wave adds
coherently by the magnitude of its transition function, 0 on its boundary and 1 well
off it. Coherent all the way, levels would step by up to 4 dB where one wave's boundary
is crossed while another's terms are there; in energy all the way, the shadow of a
screen would be 3 dB deeper than Maekawa's chart. As it is, deep behind the corner of
an L-shaped room the 63 Hz band is 0.7 dB below the coherent sum, the 125 Hz band 0.4
dB, and from 500 Hz up the two agree within 0.1 dB.

With ``first_order`` diffraction the same is done for the first-order reflections: sound
that reflects off a plane and then bends over an edge, and sound that bends over an edge
and then reflects.

With ``second_order`` the direct sound also bends over two edges in turn: over a thick
wall or a pillar, round both of its corners. Its shortest path touches each edge where
the path makes equal angles with it on both sides. The wave from the first edge reaches
the second as a wave of its own, which the second bends again: where the path runs along
a face between them (the top of a thick wall), that wave already holds the face's
reflection, so the second edge's coefficient is halved. Near the first edge's
boundaries its wave still curves like the source's, not like a wave from the edge, and
the second edge's distance parameter takes that curvature where the wave reaches it;
seen from the listener, the same holds for the second edge's wave at the first. Each
edge takes the other's curvature by how much nearer the other's boundary is, so the
path is the same both ways round. The waves of the two edges add coherently by the
product of their transition functions, and near either edge's boundary the path
interferes with the path over the other edge alone, which takes its cross term, as a
specular path does at one edge. Where the source's wave would reach the listener past
both edges the path over two edges is left out: it is then a correction of a
correction.

Openings are not there for diffraction: the edges of a doorway are the free edges of the
wall round it, and diffracted paths pass through openings as through air. Outdoors, the
open sky is an opening too, but the edges it meets diffract nothing (the ground goes on
past them), nor do edges that lie on the ground, which closes the corner of a wall or a
barrier standing on it (see :mod:`acoustic_spaces.geometry.outdoor`). Any other
surface stops them, even one that lets some sound through: the sound diffracted
through a wall is left out, as it is far weaker than the sound that passes it straight.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.coherent import ground_factors, reflection_phases
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import MAX_CROSSINGS, TraceModel, source_weights
from acoustic_spaces.geometry.outdoor import ON_GROUND_M, on_ground
from acoustic_spaces.materials.bands import BANDS_PER_OCTAVE, Resolution, band_centres

#: Faces that bend away from each other by less than this (degrees) make no
#: diffracting edge: the facets of a curved surface, for example.
MIN_BEND_DEG = 15.0
#: Frequencies per octave band at which the diffracted waves are evaluated, one in the
#: middle of each equal slice of the band; energies are averaged over the band. A
#: third-octave band has a third as many, rounded up.
FREQUENCIES_PER_BAND = 8
#: A diffracted path is dropped when its energy is this far below that of the direct
#: sound in free field, in every band.
MIN_RELATIVE_ENERGY = 1e-4
#: Diffraction settings: none, of the direct sound only, of the direct sound and
#: first-order reflections, or also of the direct sound over two edges in turn.
DIFFRACTION_MODES = ("none", "direct", "first_order", "second_order")

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
#: Below this cosine to a plane's normal (0.6 degrees from grazing) a path is never
#: culled for missing that plane's faces (see :func:`_reflecting_pairs`).
_CULL_GRAZING = 0.01


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


def band_frequencies(
    per_band: int | None = None, bands: Resolution = "octave"
) -> tuple[np.ndarray, np.ndarray]:
    """(bands, per_band) frequencies at the middle of equal slices of each octave band
    (from its centre over the square root of 2 to its centre times it) or third-octave
    band (over and times the sixth root of 2), and the width of the slices per band
    (bands,). ``per_band`` defaults to :data:`FREQUENCIES_PER_BAND` per octave."""
    per_octave = BANDS_PER_OCTAVE[bands]
    if per_band is None:
        per_band = math.ceil(FREQUENCIES_PER_BAND / per_octave)
    half = 2.0 ** (0.5 / per_octave)
    centre = np.asarray(band_centres(bands), dtype=float)
    low = centre / half
    width = low * (1.0 if per_octave == 1 else half**2 - 1.0) / per_band
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

    def _across(self, index: np.ndarray, points: np.ndarray):
        v = points - self.start[index]
        t = np.einsum("ij,ij->i", v, self.direction[index])
        return t, v - t[:, None] * self.direction[index]

    def distances(self, index: np.ndarray, points: np.ndarray):
        """Position along edge ``index`` of each point and distance from it, as
        :meth:`frame` has them."""
        t, w = self._across(index, points)
        return t, np.linalg.norm(w, axis=1)

    def frame(self, index: np.ndarray, points: np.ndarray):
        """Position along edge ``index`` of each point, distance from it and angle around
        it from face 0, and whether the point lies in the open side of the wedge."""
        t, w = self._across(index, points)
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
    solid = ~model.transparent  # openings are holes here
    if model.sky is not None:
        # An outdoor scene's sky is an opening, but no hole: the ground goes on past it.
        solid |= model.sky
    solid = np.flatnonzero(solid)
    faces = mesh.faces[solid]
    first = faces.reshape(-1)
    second = faces[:, [1, 2, 0]].reshape(-1)
    opposite = faces[:, [2, 0, 1]].reshape(-1)
    owner = np.repeat(solid, 3)
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
        # Face n lies behind face 0: the solid is inside the angle. Between faces with a
        # room on both sides (a bent panel), the side that opens wider diffracts.
        keep &= (side < 0.0) | model.plane_two_sided[model.face_plane[face0]]
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
    if model.sky is not None:
        parts = [(*part[:8], part[8] & _outdoors(model, part)) for part in parts]
    p0, p1, e, along, normal, wedge, face0, face1 = (
        np.concatenate([part[i][part[8]] for part in parts]) for i in range(8)
    )
    return _join(model, p0, p1, e, along, normal, wedge, face0, face1)


def _outdoors(model: TraceModel, part) -> np.ndarray:
    """Which edges of an outdoor scene diffract: not those the sky meets, where the
    ground goes on and the air is open, nor those on the ground, where it closes the
    corner of a wall or barrier that stands on it, nor the inside corners of buildings.

    A building is open underneath, so there is air on both sides of its walls and the
    model cannot tell its outside corners from its inside ones; but its faces point out
    of it, and an edge between two of them is an outside corner when the side ``part``
    takes as open (``normal``) is the side a face points to."""
    p0, p1, normal, face0, face1 = part[0], part[1], part[4], part[6], part[7]
    layout = model.room.outdoor
    if layout.terrain is None:
        grounded = on_ground(p0, model.ground) & on_ground(p1, model.ground)
    else:
        # Edges at or below the terrain are buried in it, its own ridges included: the
        # sound over the terrain's crests is :mod:`acoustic_spaces.engine.crests`'s.
        grounded = (p0[:, 2] <= layout.ground_at(p0) + ON_GROUND_M) & (
            p1[:, 2] <= layout.ground_at(p1) + ON_GROUND_M
        )
    building = model.room.building_faces()
    out = model.room.mesh.face_normals()[face0]
    inside = building[face0] & building[face1] & (np.einsum("ij,ij->i", out, normal) < 0.0)
    return ~(model.sky[face0] | model.sky[face1] | grounded | inside)


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
    kept = 1.0 - model.absorption[faces]
    if model.face_impedance is not None or model.ground_part is not None:
        kept[model.factor_faces(faces)] = 1.0  # their reflection factor comes in on the way
    return Edges(
        start=foot[chosen] + begin[:, None] * e[chosen],
        direction=e[chosen],
        length=finish - begin,
        along=along[chosen],
        normal=normal[chosen],
        wedge=wedge[chosen],
        face=faces,
        plane=model.face_plane[faces].astype(np.int64),
        reflection=np.sqrt(kept * (1.0 - model.scattering[faces])),
    )


# -- diffracted paths ---------------------------------------------------------------


class EdgeDiffraction:
    """Diffraction of one source's sound at the edges of a model, for any number of
    listeners: :meth:`apply` adds it to the specular paths of one listener.

    ``mode`` is ``"direct"`` (the direct sound only), ``"first_order"`` (also the
    first-order reflections, before or after the edge) or ``"second_order"`` (also the
    direct sound over two edges in turn); with ``both_sides`` the paths over one edge
    may also reflect both before and after it (the four paths over a barrier on the
    ground). ``air`` is the energy
    attenuation per metre per band; ``directivity`` and ``orientation`` are the source's,
    as for :func:`~acoustic_spaces.engine.early.find_paths`. In the lowest ``coherent``
    bands the terms of every wave add coherently, the faces of the edges and the planes
    on the way reflect with the phase of their reflection factors, and the diffracted
    paths carry their pressure (``EarlyReflections.spectrum``) to add to the specular
    paths in pressure (:mod:`~acoustic_spaces.engine.coherent`) instead of through cross
    terms. Only the pairs of an edge and a plane that a reflection between them can take
    are tested (:func:`_reflecting_pairs`, on ``threads`` threads, 0 for every core).
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
        coherent: int = 0,
        both_sides: bool = False,
        threads: int = 0,
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
        self.coherent = int(coherent)
        self.reflections = mode != "direct"
        self.both_sides = bool(both_sides) and self.reflections
        self.two_edges = mode == "second_order"
        self.threads = int(threads)
        self._edge_pair_cache: np.ndarray | None = None
        self.size = float(np.ptp(model.room.mesh.vertices, axis=0).max()) or 1.0
        frequencies, self.slice_hz = band_frequencies(bands=model.bands)
        self.wavenumber = 2.0 * np.pi * frequencies / self.speed_of_sound  # (B, M)

        # The source and its images in the planes it faces; the source side of a
        # diffracted path is one of them, the listener side the listener or its image.
        planes = self._facing(self.source) if self.reflections else np.zeros(0, np.int64)
        self.source_planes = np.concatenate([[-1], planes]).astype(np.int64)
        self.source_points = np.vstack(
            [self.source, model.mirror(np.broadcast_to(self.source, (len(planes), 3)), planes)]
        )
        # Only the source's images that a path from each edge can reflect towards.
        self._bounds = _plane_bounds(model) if self.reflections else None
        rows, cols = self._reflecting(self.source, planes, np.arange(len(self.edges)))
        edge = np.concatenate([np.arange(len(self.edges)), rows])
        side = np.concatenate([np.zeros(len(self.edges), np.int64), cols + 1])
        order = np.lexsort((side, edge))
        edge, side = edge[order], side[order]
        t, rho, phi, inside = self.edges.frame(edge, self.source_points[side])
        keep = inside & (rho > _TOLERANCE * self.size)
        keep &= self.source_planes[side] != self.edges.plane[edge, 0]
        keep &= self.source_planes[side] != self.edges.plane[edge, 1]
        self._source_side = (edge[keep], side[keep], t[keep], rho[keep], phi[keep])

    def _face_factors(
        self, edge: np.ndarray, sin_beta: np.ndarray, phi_source: np.ndarray
    ) -> np.ndarray:
        """(C, 2, B, M) what the reflections off face 0 and face n of ``edge`` take: their
        specular share and, in the coherent bands, the phase of their reflection factor
        at the angle the wave from ``phi_source`` (at ``sin_beta`` to the edge) meets
        them at the edge, which a reflection on its boundary has."""
        reflection = self.edges.reflection[edge][..., None]
        out = np.broadcast_to(reflection, (*reflection.shape[:3], self.wavenumber.shape[1]))
        out = out.astype(complex)
        ground = self.model.factor_faces(self.edges.face[edge])  # (C, 2)
        if self.coherent or ground.any():
            n = self.edges.wedge[edge]
            sines = np.column_stack([np.sin(phi_source), np.sin(n * np.pi - phi_source)])
            cosines = np.asarray(sin_beta)[:, None] * np.abs(sines)
            for side in (0, 1):
                if self.coherent:
                    out[:, side, : self.coherent] *= reflection_phases(
                        self.model, self.edges.face[edge, side], cosines[:, side], self.coherent
                    )
                rows = np.flatnonzero(ground[:, side])
                if len(rows):  # a surface with an impedance: its plane wave's factor
                    out[rows, side] *= ground_factors(
                        self.model,
                        self.edges.face[edge[rows], side],
                        cosines[rows, side],
                        None,
                        spherical=False,
                    )
        return out

    def _coherent_share(self, share: np.ndarray) -> np.ndarray:
        """``share`` (C, W, B, M) with the waves fully coherent in the coherent bands."""
        share[:, :, : self.coherent] = 1.0
        return share

    def _spectrum(self, pressure: np.ndarray, distance: np.ndarray) -> np.ndarray | None:
        """(C, coherent, M) the paths' pressure (C, B, M) in the coherent bands relative
        to their delay, over its level in each band; None without coherent bands."""
        if not self.coherent:
            return None
        k = self.wavenumber[: self.coherent]
        out = pressure[:, : self.coherent] * np.exp(1j * k[None] * distance[:, None, None])
        level = np.sqrt(np.mean(np.abs(out) ** 2, axis=2, keepdims=True))
        return np.where(level > 0.0, out / np.where(level > 0.0, level, 1.0), 1.0)

    def _facing(self, point: np.ndarray) -> np.ndarray:
        """Planes that ``point`` lies in front of (on either side of two-sided ones)."""
        distance = self.model.plane_normal @ point - self.model.plane_offset
        return np.flatnonzero(self.model.in_front(distance, _TOLERANCE * self.size))

    def _reaching(self, listener, planes, row_edge, t_s, rho_s, max_delay_s):
        """For source sides on edges ``row_edge`` at ``t_s``, ``rho_s``, the images of
        ``listener`` in ``planes`` that a path from the edge can reflect towards and whose
        path over the edge may touch it within ``max_delay_s`` (:func:`_reaching_pairs`):
        rows and columns, by row and then plane."""
        if len(planes) == 0 or len(row_edge) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        used = np.unique(row_edge)
        return _reaching_pairs(
            self.model,
            self.edges,
            self._bounds,
            listener,
            planes,
            used,
            (np.searchsorted(used, row_edge), t_s, rho_s),
            self.speed_of_sound * max_delay_s,
            _TOLERANCE * self.size,
            _OFFSET * self.size,
            self.threads,
        )

    def _reflecting(self, end: np.ndarray, planes: np.ndarray, among: np.ndarray):
        """The pairs of an edge of ``among`` and a plane of ``planes`` that a path from
        the edge to ``end`` can reflect off, as rows into both (:func:`_reflecting_pairs`)."""
        if len(planes) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        offset = _OFFSET * self.size
        return _reflecting_pairs(
            self.model, self.edges, self._bounds, end, planes, among, offset, self.threads
        )

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
        single = None if combos is None else self._single(listener, paths, combos)
        double = self._double(listener, max_delay_s, single) if self.two_edges else None
        if single is None and double is None:
            return paths
        # The direct sound in free field; a listener on the source counts as just off it.
        squared = max(float(np.sum((listener - self.source) ** 2)), (_TOLERANCE * self.size) ** 2)
        floor = MIN_RELATIVE_ENERGY / (4.0 * np.pi * squared)
        parts = [paths if single is None else single["paths"]]
        for found in (single, double):
            if found is not None:
                parts.append(self._emit(listener, found, floor, max_delay_s))
        return EarlyReflections.concatenate(parts, self.model.absorption.shape[1])

    def _combos(self, listener: np.ndarray, max_delay_s: float):
        """Source and listener sides of every edge whose shortest path over the edge
        touches it and is unobstructed."""
        edges = self.edges
        edge, side, t_s, rho_s, phi_s = self._source_side
        tolerance = _TOLERANCE * self.size

        # Listener sides: the listener, for every source side; its images, for the
        # source itself; and, with both_sides, its images for the source's images too.
        # ``kind`` says which: 0 nothing reflected, 1 before the edge (``plane``), 2
        # after it (``plane``), 3 both (``plane`` before, ``plane2`` after).
        planes = self._facing(listener) if self.reflections else np.zeros(0, np.int64)
        all_edges = np.arange(len(edges))
        c = self.speed_of_sound
        t_l, rho_l, phi_l, in_l = edges.frame(all_edges, np.broadcast_to(listener, (len(edges), 3)))
        # Each part keeps the rows whose path touches the edge within max_delay_s:
        # (edge, source side, listener side, plane, kind, plane2, phi_s, phi_r, rho_s,
        # t_k, r1, r2).
        rows = np.flatnonzero(in_l[edge] & (rho_l[edge] > tolerance))
        e = edge[rows]
        kind = np.where(self.source_planes[side[rows]] >= 0, 1, 0)
        ok, t_k, r1, r2 = _reach(
            kind, edges.length[e], t_s[rows], rho_s[rows], t_l[e], rho_l[e], c, max_delay_s
        )
        rows, e = rows[ok], e[ok]
        parts = [
            (
                e,
                self.source_points[side[rows]],
                np.broadcast_to(listener, (len(rows), 3)),
                self.source_planes[side[rows]],
                kind[ok],
                np.full(len(rows), -1, dtype=np.int64),
                phi_s[rows],
                phi_l[e],
                rho_s[rows],
                t_k[ok],
                r1[ok],
                r2[ok],
            )
        ]
        if len(planes):
            images = self.model.mirror(np.broadcast_to(listener, (len(planes), 3)), planes)
            # Images of the listener for the source itself (kind 2) and, with both_sides,
            # for the source's images too (kind 3, a reflection on each side): those that
            # a path from the edge can reflect towards and that may touch the edge in time.
            groups = [(2, np.flatnonzero(self.source_planes[side] < 0))]
            if self.both_sides:
                groups.append((3, np.flatnonzero(self.source_planes[side] >= 0)))
            at = np.concatenate([g[1] for g in groups])
            found, cols = self._reaching(
                listener, planes, edge[at], t_s[at], rho_s[at], max_delay_s
            )
            split = np.searchsorted(found, np.cumsum([len(g[1]) for g in groups]))
            for (image_kind, _), reps, pp in zip(
                groups, np.split(found, split[:-1]), np.split(cols, split[:-1]), strict=True
            ):
                rows = at[reps]
                pe = edge[rows]
                plane = planes[pp]
                t_i, rho_i = edges.distances(pe, images[pp])
                ok = (rho_i > tolerance) & (plane != edges.plane[pe, 0])
                ok &= plane != edges.plane[pe, 1]
                reach, t_k, r1, r2 = _reach(
                    image_kind,
                    edges.length[pe],
                    t_s[rows],
                    rho_s[rows],
                    t_i,
                    rho_i,
                    c,
                    max_delay_s,
                )
                sel = np.flatnonzero(ok & reach)
                _, _, phi_i, in_i = edges.frame(pe[sel], images[pp[sel]])
                sel, phi_i = sel[in_i], phi_i[in_i]
                rows = rows[sel]
                n = len(sel)
                if image_kind == 2:
                    src = np.broadcast_to(self.source, (n, 3))
                    before, after = plane[sel], np.full(n, -1, dtype=np.int64)
                else:
                    src = self.source_points[side[rows]]
                    before, after = self.source_planes[side[rows]], plane[sel]
                parts.append(
                    (
                        pe[sel],
                        src,
                        images[pp[sel]],
                        before,
                        np.full(n, image_kind),
                        after,
                        phi_s[rows],
                        phi_i,
                        rho_s[rows],
                        t_k[sel],
                        r1[sel],
                        r2[sel],
                    )
                )
        (edge, src, rcv, plane, kind, plane2, phi_s, phi_r, rho_s, t_k, r1, r2) = (
            np.concatenate([p[i] for p in parts]) for i in range(12)
        )
        if len(edge) == 0:
            return None
        point = edges.start[edge] + t_k[:, None] * edges.direction[edge]

        # Unobstructed, with the reflection on the expected plane before or after the
        # edge. Rays start just off the edge, in the middle of its open side, and just
        # inside its ends (where a plane they may be meant to reflect off begins).
        offset = _OFFSET * self.size
        along = np.clip(t_k, offset, edges.length[edge] - offset) - t_k
        start = point + along[:, None] * edges.direction[edge] + offset * edges.bisector()[edge]
        reflected = np.full(len(edge), -1, dtype=np.int64)
        bounce = np.zeros((len(edge), 3))
        reflected2 = np.full(len(edge), -1, dtype=np.int64)  # after the edge, kind 3
        bounce2 = np.zeros((len(edge), 3))
        ok = np.ones(len(edge), dtype=bool)
        for end, image_kind in ((self.source, 1), (listener, 2)):
            ends = np.broadcast_to(end, (len(edge), 3))
            via = (kind == image_kind) | (kind == 3)
            plain = ~via
            if plain.any():
                ok[plain] &= _clear(self.model, start[plain], ends[plain])
            if via.any():
                target = src[via] if image_kind == 1 else rcv[via]
                through = plane[via]
                if image_kind == 2:
                    through = np.where(kind[via] == 3, plane2[via], through)
                hit, where, face = _reflect(self.model, start[via], target, through, ends[via])
                ok[via] &= hit
                rows = np.flatnonzero(via)
                second = (kind[via] == 3) & (image_kind == 2)
                bounce[rows[~second]], reflected[rows[~second]] = where[~second], face[~second]
                bounce2[rows[second]], reflected2[rows[second]] = where[second], face[second]
        if not ok.any():
            return None
        sel = np.flatnonzero(ok)
        return {
            "edge": edge[sel],
            "source_side": src[sel],
            "plane": plane[sel],
            "plane2": plane2[sel],
            "kind": kind[sel],
            "point": point[sel],
            "bounce": bounce[sel],
            "face": reflected[sel],
            "bounce2": bounce2[sel],
            "face2": reflected2[sel],
            "r1": r1[sel],
            "r2": r2[sel],
            "rho_s": rho_s[sel],
            "phi_s": phi_s[sel],
            "phi_r": phi_r[sel],
        }

    def _single(self, listener, paths: EarlyReflections, c: dict) -> dict:
        """The paths over one edge: each wave's terms and their energy, and ``paths`` with
        the cross terms of the specular paths that border them."""
        model, edges = self.model, self.edges
        edge, kind = c["edge"], c["kind"]
        r1, r2 = c["r1"], c["r2"]
        n = edges.wedge[edge]
        k = self.wavenumber[None, :, :]  # (1, B, M)
        distance = r1 + r2

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
        kl = k * length_param[:, None, None]
        before = (kind == 1) | (kind == 3)  # a reflection before the edge
        first = np.where(before[:, None], c["bounce"], c["point"])
        departure = first - self.source
        departure /= np.maximum(np.linalg.norm(departure, axis=1), 1e-300)[:, None]
        # A coherent array near the edge: its sound where it meets the edge.
        weight = source_weights(
            self.directivity, self.orientation, departure, len(self.air), distances=r1
        )  # (C, B)
        amplitude = np.sqrt(weight * np.exp(-np.outer(distance, self.air)) / (4.0 * np.pi))
        amplitude /= np.sqrt(r1 * r2 * distance)[:, None]
        # The reflections on the way: before the edge or after it ("face", "bounce"),
        # and after it as well for the paths that reflect on both sides ("face2").
        legs = (
            (c["face"], c["bounce"], before),
            (c["face2"], c["bounce2"], np.zeros(len(kind), dtype=bool)),
        )
        for faces, _, _ in legs:
            on_plane = faces >= 0
            if on_plane.any():
                face = faces[on_plane]
                kept = 1.0 - model.absorption[face]
                if model.face_impedance is not None or model.ground_part is not None:
                    kept[model.factor_faces(face)] = 1.0  # their reflection factor comes below
                amplitude[on_plane] *= np.sqrt(kept * (1.0 - model.scattering[face]))
        common = (
            amplitude[:, :, None]
            * wedge_prefactor(n[:, None, None], k, sin_beta[:, None, None])
            * np.exp(-1j * k * distance[:, None, None])
        )  # (C, B, M)
        for faces, bounces, via in legs:
            on_plane = faces >= 0
            grounds = on_plane.any() and model.factor_faces(faces[on_plane]).any()
            if not ((self.coherent or grounds) and on_plane.any()):
                continue
            # The phase of the reflection off the plane before or after the edge.
            rows = np.flatnonzero(on_plane)
            face = faces[rows]
            # The leg between the plane and the source or the listener: the one between
            # it and the edge vanishes where the path reflects at an end of the edge.
            bounce = bounces[rows]
            via_source = via[rows]
            incoming = np.where(
                via_source[:, None], bounce - self.source[None], listener[None] - bounce
            )
            normal = model.plane_normal[model.face_plane[face]]
            cosine = np.abs(np.einsum("ij,ij->i", normal, incoming)) / np.maximum(
                np.linalg.norm(incoming, axis=1), 1e-300
            )
            if self.coherent:
                common[rows, : self.coherent] *= reflection_phases(
                    model, face, cosine, self.coherent
                )
            ground = model.factor_faces(face)
            if ground.any():
                # A ground's spherical wave, from the image of the source or the listener
                # to the edge.
                leg = np.where(via_source, r1[rows], r2[rows])
                edge_point = c["point"][rows]
                ends = (
                    np.where(via_source[:, None], self.source[None], edge_point),
                    bounce,
                    np.where(via_source[:, None], edge_point, listener[None]),
                )
                common[rows[ground]] *= ground_factors(
                    model,
                    face[ground],
                    cosine[ground],
                    leg[ground],
                    ends=tuple(p[ground] for p in ends),
                )
        factors = self._face_factors(edge, sin_beta, c["phi_s"])
        waves = _waves(n, c["phi_r"], c["phi_s"], kl, lit, factors)
        waves *= common[:, None]  # (C, 3, B, M): incident, off face 0, off face n

        # The waves' terms add up coherently, as one diffracted wave, except in the
        # transition zone round each wave's boundary: there its terms stand in for the
        # geometric wave they continue and add in energy, as the specular paths do. A
        # wave adds coherently by |F(X)| of its boundary, 0 on it and 1 well off it.
        counted = lit == (found >= 0)
        coherence = _coherence(self._coherent_share(_share(kl, angle)))
        energy = _energy(waves * counted[:, :, None, None], coherence)

        # The diffracted wave arrives on its own, but interferes with the specular
        # paths of its lit terms, which arrive at almost the same time near the
        # boundaries: each of those paths takes the cross term with its own wave's
        # terms. On the boundary a path's energy and the diffracted wave's then add up
        # to a quarter of the path's on its own, as the diffracted wave alone carries on
        # the dark side.
        out = paths
        rows, groups = np.nonzero(lit & (found >= 0))
        if len(rows):
            targets = found[rows, groups]
            phase = np.exp(-1j * self.wavenumber[None] * paths.distance_m[targets, None, None])
            specular = np.sqrt(paths.energy[targets])[:, :, None] * phase
            fade = self._fade(distance[rows] - paths.distance_m[targets])
            own = waves[rows, groups]
            cross = 2.0 * fade * np.mean(np.real(specular * np.conj(own)), axis=2)
            # The waves of other edges that border the same path continue it too, as round
            # a narrow face: each pair adds its cross term to the path as well, half on
            # each wave's.
            size = np.bincount(targets, minlength=targets.max(initial=-1) + 1)
            if (size**2).sum() <= 1 << 16:
                i, j = _in_groups(targets, targets)
                keep = rows[i] != rows[j]
                i, j = i[keep], j[keep]
                if len(i):
                    pair = np.mean(np.real(own[i] * np.conj(own[j])), axis=2)
                    np.add.at(
                        cross, i, self._fade(distance[rows[i]] - distance[rows[j]]) * pair
                    )
            else:
                # A path many edges border (a curved surface): the waves' sum
                # stands in for each pair, which keeps the work linear in them.
                unique, inverse = np.unique(targets, return_inverse=True)
                total = np.zeros((len(unique), *own.shape[1:]), dtype=complex)
                np.add.at(total, inverse, own)
                rest = total[inverse] - own
                cross += np.mean(np.real(own * np.conj(rest)), axis=2)
            # In the coherent bands the two add in pressure instead.
            cross[:, : self.coherent] = 0.0
            total, energy = _take(paths.energy, targets, cross, energy, rows)
            out = _replace_energy(paths, total)

        def corners(i: int) -> list[np.ndarray]:
            if kind[i] == 1:
                return [c["bounce"][i], c["point"][i]]
            if kind[i] == 2:
                return [c["point"][i], c["bounce"][i]]
            if kind[i] == 3:
                return [c["bounce"][i], c["point"][i], c["bounce2"][i]]
            return [c["point"][i]]

        def faces(i: int) -> np.ndarray:
            if kind[i] == 3:
                return np.array([c["face"][i], c["face2"][i]], dtype=np.int64)
            return np.array([c["face"][i]] if kind[i] else [], dtype=np.int64)

        last = np.where((kind == 2)[:, None], c["bounce"], c["point"])
        last = np.where((kind == 3)[:, None], c["bounce2"], last)

        return {
            "c": c,
            "waves": waves,
            "counted": counted,
            "coherence": coherence,
            "energy": energy,
            "spectrum": self._spectrum((waves * counted[:, :, None, None]).sum(axis=1), distance),
            "distance": distance,
            "paths": out,
            "order": np.where(kind == 3, 2, np.where(kind > 0, 1, 0)).astype(np.int64),
            "edge": edge,
            "last": last,
            "corners": corners,
            "faces": faces,
        }

    def _double(self, listener: np.ndarray, max_delay_s: float, single: dict | None):
        """The direct sound over two edges in turn (see the module notes), as
        :meth:`_single` gives the paths over one edge; the cross terms with those paths go
        to their energies in ``single``."""
        model, edges = self.model, self.edges
        tolerance = _TOLERANCE * self.size
        edge_s, side_s, _, rho_s, phi_s = self._source_side
        direct = self.source_planes[side_s] < 0
        seen = np.zeros(len(edges), dtype=bool)
        seen[edge_s[direct]] = True
        rho_from, phi_from = np.zeros(len(edges)), np.zeros(len(edges))
        rho_from[edge_s[direct]] = rho_s[direct]
        phi_from[edge_s[direct]] = phi_s[direct]
        _, rho_to, phi_to, inside = edges.frame(
            np.arange(len(edges)), np.broadcast_to(listener, (len(edges), 3))
        )
        heard = inside & (rho_to > tolerance)
        pairs = self._edge_pairs()
        pairs = pairs[seen[pairs[:, 0]] & heard[pairs[:, 1]]]
        if len(pairs) == 0:
            return None
        e1, e2 = pairs[:, 0], pairs[:, 1]
        d1, d2 = edges.direction[e1], edges.direction[e2]
        t1, t2, ok = _shortest(edges.start[e1], d1, edges.start[e2], d2, self.source, listener)
        ok &= (t1 >= 0.0) & (t1 <= edges.length[e1]) & (t2 >= 0.0) & (t2 <= edges.length[e2])
        p1 = edges.start[e1] + t1[:, None] * d1
        p2 = edges.start[e2] + t2[:, None] * d2
        r0 = np.linalg.norm(p1 - self.source, axis=1)
        r1 = np.linalg.norm(p2 - p1, axis=1)
        r2 = np.linalg.norm(listener - p2, axis=1)
        ok &= (r1 > tolerance) & ((r0 + r1 + r2) / self.speed_of_sound <= max_delay_s)
        # Each edge lies in the open side of the other, where the path touches it.
        _, rho_21, phi_21, in_21 = edges.frame(e1, p2)
        _, rho_12, phi_12, in_12 = edges.frame(e2, p1)
        ok &= in_21 & (rho_21 > tolerance) & in_12 & (rho_12 > tolerance)
        # Unobstructed, with rays from just off each edge, as for one edge.
        offset = _OFFSET * self.size
        bisector = edges.bisector()
        starts = [
            p
            + (np.clip(t, offset, edges.length[e] - offset) - t)[:, None] * edges.direction[e]
            + offset * bisector[e]
            for p, t, e in ((p1, t1, e1), (p2, t2, e2))
        ]
        ends = np.broadcast_to(self.source, p1.shape), np.broadcast_to(listener, p2.shape)
        for a, b in ((starts[0], ends[0]), (starts[0], starts[1]), (starts[1], ends[1])):
            rows = np.flatnonzero(ok)
            if len(rows) == 0:
                return None
            ok[rows] = _clear(model, a[rows], b[rows])
        sel = np.flatnonzero(ok)
        if len(sel) == 0:
            return None
        e1, e2, d1, d2, p1, p2 = e1[sel], e2[sel], d1[sel], d2[sel], p1[sel], p2[sel]
        r0, r1, r2 = r0[sel], r1[sel], r2[sel]
        phi_21, rho_21, phi_12, rho_12 = phi_21[sel], rho_21[sel], phi_12[sel], rho_12[sel]
        n1, n2 = edges.wedge[e1], edges.wedge[e2]
        phi_s1, phi_r2 = phi_from[e1], phi_to[e2]
        sin1 = np.clip(rho_from[e1] / r0, 1e-12, 1.0)
        sin2 = np.clip(rho_12 / r1, 1e-12, 1.0)
        distance = r0 + r1 + r2
        k = self.wavenumber[None, :, :]  # (1, B, M)
        # The path from the first edge to the second runs along face 0 or face n of the
        # first (it leaves along it) and of the second (grazing incidence there).
        leave = _on_faces(phi_21, rho_21, n1, tolerance)
        graze = _on_faces(phi_12, rho_12, n2, tolerance)

        # Which side of each edge's boundaries the path is on, and the path over the
        # other edge alone that each geometric wave continues there.
        has1, has2 = self._alone(single, e1, e2, phi_s1, phi_12, leave, graze)
        angle1 = boundary_angles(n1, phi_21, phi_s1)
        angle2 = boundary_angles(n2, phi_r2, phi_12)
        lit1 = (angle1 > _ON_BOUNDARY) | ((np.abs(angle1) <= _ON_BOUNDARY) & (has1 >= 0))
        lit2 = (angle2 > _ON_BOUNDARY) | ((np.abs(angle2) <= _ON_BOUNDARY) & (has2 >= 0))
        # A face's reflection that is one wave with the incident wave shares its side,
        # also exactly on their common boundary.
        for lit, faces in ((lit1, leave), (lit2, graze)):
            lit[:, 1:] = np.where(faces, lit[:, :1], lit[:, 1:])

        # The distance parameters: the source's wave at the first edge, and at the second
        # the first edge's wave, whose radii of curvature are r1 across the path and
        # r0 + r1 along it, seen in the plane through the path and the second edge.
        v = (p2 - p1) / r1[:, None]
        t_across = np.cross(d1, v)
        t_across /= np.linalg.norm(t_across, axis=1)[:, None]
        q = d2 - np.einsum("ij,ij->i", d2, v)[:, None] * v
        q /= np.maximum(np.linalg.norm(q, axis=1), 1e-300)[:, None]
        qa = np.einsum("ij,ij->i", q, t_across) ** 2
        along = r0 + r1
        rho_e = 1.0 / (qa / r1 + (1.0 - qa) / along)
        length1 = r0 * r1 / (r0 + r1) * sin1**2
        length2 = r2 * (rho_e + r2) * r1 * along * sin2**2 / (rho_e * (r1 + r2) * (along + r2))
        # Near the first edge's boundaries its wave is still curved like the source's
        # where it reaches the second edge, not like a wave from the edge: from the
        # phase of the edge's coefficient across the path, 1/R = 1/r1 - psi'' / (k (r1
        # sin b1)^2), between 1/r1 and 1/(r0 + r1). Seen from the listener the same holds
        # for the second edge's wave at the first. The second edge's distance parameter
        # takes the first curvature and the first edge's the second, each by its share of
        # the two, so that the path is the same both ways round.
        kl1, kl2 = k * length1[:, None, None], k * length2[:, None, None]
        factors1 = self._face_factors(e1, sin1, phi_s1)
        factors2 = self._face_factors(e2, sin2, phi_12)
        forward = _phase_curvature(n1, phi_21, phi_s1, kl1, factors1, angle1)
        back_angle = boundary_angles(n2, phi_12, phi_r2)
        back_kl = k * (r2 * r1 / (r2 + r1) * sin2**2)[:, None, None]
        backward = _phase_curvature(n2, phi_12, phi_r2, back_kl, factors2, back_angle)
        flat = 1.0 / r1[:, None, None]
        bend1 = flat - np.clip(
            flat - forward / (k * ((r1 * sin1) ** 2)[:, None, None]),
            1.0 / along[:, None, None],
            flat,
        )
        bend2 = flat - np.clip(
            flat - backward / (k * ((r1 * sin2) ** 2)[:, None, None]),
            1.0 / (r1 + r2)[:, None, None],
            flat,
        )
        # In an edge's transition zone its wave is curved like the one it continues, at
        # least by 1 - |F(X)| of the nearest boundary; all of it on the boundary, where
        # the phase alone does not show it. Each edge's distance parameter takes the
        # other's bend by how much nearer the other's boundary is: all of it on that
        # boundary, where the path continues the path over one edge, and none on its own.
        near1 = _share(kl1, angle1).min(axis=1)
        near2 = _share(kl2, angle2).min(axis=1)
        bend1 = np.maximum(bend1, (1.0 - near1) * (flat - 1.0 / along[:, None, None]))
        bend2 = np.maximum(bend2, (1.0 - near2) * (flat - 1.0 / (r1 + r2)[:, None, None]))
        part = np.where(near1 + near2 > 0.0, near2 / np.maximum(near1 + near2, 1e-300), 0.5)
        radius2 = 1.0 / (flat - part * bend1)
        radius1 = 1.0 / (flat - (1.0 - part) * bend2)
        kl2 = (
            kl2
            * radius2
            * (r1 + r2)[:, None, None]
            / (r1[:, None, None] * (radius2 + r2[:, None, None]))
        )
        kl1 = (
            kl1
            * radius1
            * (r1 + r0)[:, None, None]
            / (r1[:, None, None] * (radius1 + r0[:, None, None]))
        )
        first = _waves(n1, phi_21, phi_s1, kl1, lit1, factors1)
        # Along a face, the incident wave and its reflection off the face are one wave.
        _merge(first, leave)
        second = _waves(n2, phi_r2, phi_12, kl2, lit2, factors2)
        # At grazing incidence they are one wave too, which already holds the face's
        # reflection: half the coefficient.
        _merge(second, graze)
        second *= np.where(graze.any(axis=1), 0.5, 1.0)[:, None, None, None]

        departure = (p1 - self.source) / r0[:, None]
        weight = source_weights(
            self.directivity, self.orientation, departure, len(self.air), distances=r0
        )
        amplitude = np.sqrt(weight * np.exp(-np.outer(distance, self.air)) / (4.0 * np.pi))
        amplitude /= np.sqrt(r0 * r1 * (r0 + r1))[:, None]
        common = (
            amplitude[:, :, None]
            * np.sqrt(rho_e / (r2 * (rho_e + r2)))[:, None, None]
            * wedge_prefactor(n1[:, None, None], k, sin1[:, None, None])
            * wedge_prefactor(n2[:, None, None], k, sin2[:, None, None])
            * np.exp(-1j * k * distance[:, None, None])
        )  # (C, B, M)
        waves = first[:, :, None] * second[:, None, :] * common[:, None, None]  # (C, 3, 3, B, M)
        counted = (lit1 == (has1 >= 0))[:, :, None] & (lit2 == (has2 >= 0))[:, None, :]
        # Each edge's waves add coherently by the product of their transition functions'
        # magnitudes, the rest in energy: between the waves of the first edge, A (C, 3,
        # 3, B, M), between those of the second, B, and between two pairs of waves the
        # product of the two.
        coherence1 = _coherence(self._coherent_share(_share(kl1, angle1)))
        coherence2 = _coherence(self._coherent_share(_share(kl2, angle2)))
        dark = ~lit1[:, :, None] & ~lit2[:, None, :] & counted
        alone = waves * dark[..., None, None]
        energy = np.einsum(
            "cikbm,cjlbm,cijbm,cklbm->cbm", coherence1, coherence2, alone, np.conj(alone)
        ).real.mean(axis=2)
        # A wave that reaches the second edge from the source (or its reflection) does
        # not reach the listener past both edges: where the first edge's wave is lit, the
        # path continues the path over the second edge alone, its waves at the second
        # edge add to that path's as they do to each other, and it takes the cross term
        # with them; and likewise where the second edge's wave is lit. Where both are,
        # the path is left out.
        others = None if single is None else single["waves"] * single["counted"][:, :, None, None]
        for lit, has, at_first in ((lit1, has1, True), (lit2, has2, False)):
            rows, groups = np.nonzero(lit & (has >= 0))
            if len(rows) == 0:
                continue
            if at_first:
                own = (
                    waves[rows, groups]
                    * (counted & ~lit2[:, None, :])[rows, groups][..., None, None]
                )
                between = coherence2[rows]
            else:
                own = (
                    waves[rows, :, groups]
                    * (counted & ~lit1[:, :, None])[rows, :, groups][..., None, None]
                )
                between = coherence1[rows]
            # own (R, 3, B, M): this path's waves at the other edge; between their coherence.
            np.add.at(
                energy,
                rows,
                np.einsum("rjlbm,rjbm,rlbm->rbm", between, own, np.conj(own)).real.mean(axis=2),
            )
            targets = has[rows, groups]
            partner = others[targets]
            fade = self._fade(distance[rows] - single["distance"][targets])
            product = np.einsum("rjlbm,rjbm,rlbm->rbm", between, partner, np.conj(own)).real
            cross = 2.0 * fade * np.mean(product, axis=2)
            cross[:, : self.coherent] = 0.0
            single["energy"], energy = _take(single["energy"], targets, cross, energy, rows)

        # In the coherent bands all the waves the path carries add in pressure, and add
        # to the paths over one edge in the impulse response.
        carried = counted & ~(lit1[:, :, None] & lit2[:, None, :])
        pressure = (waves * carried[..., None, None]).sum(axis=(1, 2))
        energy[:, : self.coherent] = np.mean(np.abs(pressure[:, : self.coherent]) ** 2, axis=2)
        return {
            "energy": energy,
            "spectrum": self._spectrum(pressure, distance),
            "distance": distance,
            "order": np.zeros(len(e1), dtype=np.int64),
            "edge": e1,
            "last": p2,
            "corners": lambda i: [p1[i], p2[i]],
            "faces": lambda i: np.zeros(0, dtype=np.int64),
        }

    def _alone(self, single, e1, e2, phi_1, phi_2, leave, graze):
        """(C, 3) each for the waves at the first edge and at the second (the incident
        wave and the reflections off face 0 and face n): the index in ``single`` of the
        path over the other edge alone that the geometric wave continues, or -1."""
        count = len(e1)
        has1 = np.full((count, 3), -1, dtype=np.int64)
        has2 = np.full((count, 3), -1, dtype=np.int64)
        if single is None:
            return has1, has2
        c, plane, n = single["c"], self.edges.plane, self.edges.wedge
        planes = len(self.model.plane_normal) + 1

        def key(edge, kind, through):
            return (np.asarray(edge) * 4 + kind) * planes + np.asarray(through) + 1

        known = key(c["edge"], c["kind"], c["plane"])
        order = np.argsort(known)
        ranked = known[order]

        def find(edge, kind, through):
            wanted = key(edge, kind, through)
            at = np.minimum(np.searchsorted(ranked, wanted), len(ranked) - 1)
            return np.where(ranked[at] == wanted, order[at], -1)

        # The source's wave reaches the second edge, or its reflection off a face of the
        # first does (the source in front of that face).
        has1[:, 0] = find(e2, 0, -1)
        has1[:, 1] = np.where((phi_1 < np.pi) & ~leave[:, 0], find(e2, 1, plane[e1, 0]), -1)
        front_n = (phi_1 > (n[e1] - 1.0) * np.pi) & ~leave[:, 1]
        has1[:, 2] = np.where(front_n, find(e2, 1, plane[e1, 1]), -1)
        # The first edge's wave reaches the listener, or off a face of the second.
        has2[:, 0] = find(e1, 0, -1)
        has2[:, 1] = np.where((phi_2 < np.pi) & ~graze[:, 0], find(e1, 2, plane[e2, 0]), -1)
        front_n = (phi_2 > (n[e2] - 1.0) * np.pi) & ~graze[:, 1]
        has2[:, 2] = np.where(front_n, find(e1, 2, plane[e2, 1]), -1)
        return has1, has2

    def _edge_pairs(self) -> np.ndarray:
        """(P, 2) ordered pairs of edges each with part of the other in its open side,
        not on one line."""
        if self._edge_pair_cache is None:
            edges = self.edges
            count = len(edges)
            tolerance = _TOLERANCE * self.size
            found = []
            for i in range(count):
                others = np.delete(np.arange(count), i)
                index = np.full(len(others), i)
                parts = [edges.frame(index, end[others]) for end in (edges.start, edges.end)]
                seen = np.zeros(len(others), dtype=bool)
                line = np.ones(len(others), dtype=bool)
                for _, rho, _, inside in parts:
                    seen |= inside & (rho > tolerance)
                    line &= rho <= tolerance
                found.append(np.column_stack([index, others])[seen & ~line])
            pairs = np.concatenate(found) if found else np.zeros((0, 2), dtype=np.int64)
            # Both ways round.
            mark = np.zeros((count, count), dtype=bool)
            mark[pairs[:, 0], pairs[:, 1]] = True
            self._edge_pair_cache = pairs[mark[pairs[:, 1], pairs[:, 0]]].astype(np.int64)
        return self._edge_pair_cache

    def _fade(self, delta_m: np.ndarray) -> np.ndarray:
        """Each sample stands for its slice of the band: averaged over the slice, the
        interference of two waves that arrive ``delta_m`` metres apart fades by sinc."""
        return np.sinc(np.outer(delta_m / self.speed_of_sound, self.slice_hz))

    def _emit(self, listener, found: dict, floor: float, max_delay_s: float) -> EarlyReflections:
        """The paths in ``found`` (from :meth:`_single` or :meth:`_double`) that carry
        at least ``floor`` in some band and arrive by ``max_delay_s``."""
        energy, distance = found["energy"], found["distance"]
        bands = self.model.absorption.shape[1]
        keep = (energy.max(axis=1) >= floor) & (distance / self.speed_of_sound <= max_delay_s)
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            return EarlyReflections.empty(bands)
        arrival = found["last"][idx] - listener
        arrival /= np.maximum(np.linalg.norm(arrival, axis=1), 1e-300)[:, None]
        spectrum = found["spectrum"]
        return EarlyReflections(
            order=found["order"][idx],
            delay_s=distance[idx] / self.speed_of_sound,
            distance_m=distance[idx],
            direction=arrival,
            energy=energy[idx].reshape(len(idx), bands),
            points=[np.vstack([self.source, *found["corners"](i), listener]) for i in idx],
            faces=[found["faces"](i) for i in idx],
            edge=found["edge"][idx].astype(np.int64),
            spectrum=None if spectrum is None else spectrum[idx],
        )

    def _specular(self, listener: np.ndarray, paths: EarlyReflections, c: dict) -> np.ndarray:
        """(C, 3) index in ``paths`` of the specular path of each term's wave (the
        incident wave and the reflections off face 0 and face n), or -1."""
        model, edges = self.model, self.edges
        count = len(c["edge"])
        out = np.full((count, 3), -1, dtype=np.int64)
        # A path through a wall is not the geometric wave a term continues.
        specular = np.flatnonzero((paths.edge < 0) & (paths.crossings == 0))
        if len(specular) == 0:
            return out
        images = listener + paths.direction[specular] * paths.distance_m[specular, None]
        orders = paths.order[specular]
        kind, plane, plane2 = c["kind"], c["plane"], c["plane2"]
        base = np.broadcast_to(self.source, (count, 3)).copy()
        has_plane = kind > 0
        base[has_plane] = model.mirror(base[has_plane], plane[has_plane])
        both = kind == 3  # and off the listener's plane last
        tolerance = 1e-6 * self.size
        for column, wedge_face in ((0, None), (1, 0), (2, 1)):
            if wedge_face is None:
                image, order = base.copy(), has_plane.astype(np.int64)
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
            if both.any():
                image[both] = model.mirror(image[both], plane2[both])
                order = order + both.astype(np.int64)
            out[:, column] = _match(image, order, images, orders, specular, tolerance)
        # A reflection off a face needs the wave to come from in front of it: on a free
        # edge both faces' images are the same point, and only one side is in front.
        n = edges.wedge[c["edge"]]
        out[:, 1] = np.where(c["phi_s"] < np.pi, out[:, 1], -1)
        out[:, 2] = np.where(c["phi_s"] > (n - 1.0) * np.pi, out[:, 2], -1)
        return out


def _waves(n, phi, phi_source, kl, lit, reflection) -> np.ndarray:
    """(C, 3, B, M) the terms of the incident wave and of the reflections off face 0 and
    face n of C edges, for wave numbers times distance parameters ``kl`` (C, B, M); their
    faces reflect by ``reflection`` (C, 2, B, M)."""
    terms = wedge_terms(
        n[:, None, None], phi[:, None, None], phi_source[:, None, None], kl, lit[:, None, None, :]
    )
    return np.stack(
        [
            terms[..., 0] + terms[..., 1],
            terms[..., 3] * reflection[:, 0],
            terms[..., 2] * reflection[:, 1],
        ],
        axis=1,
    )


def _on_faces(phi, rho, n, tolerance) -> np.ndarray:
    """(C, 2) whether points ``phi`` round edges open over ``n`` pi, ``rho`` from them,
    lie on the plane of face 0 or of face n (only face 0 for both)."""
    out = np.column_stack([phi * rho <= tolerance, (np.pi * n - phi) * rho <= tolerance])
    out[:, 1] &= ~out[:, 0]
    return out


def _merge(waves: np.ndarray, faces: np.ndarray) -> None:
    """Add the reflection off face 0 or face n into the incident wave where ``faces``
    (C, 2) says the two are one wave, in place."""
    for face in (0, 1):
        rows = np.flatnonzero(faces[:, face])
        waves[rows, 0] += waves[rows, face + 1]
        waves[rows, face + 1] = 0.0


def _coherence(share: np.ndarray) -> np.ndarray:
    """(C, W, W, B, M) how waves (C, W, B, M) that add coherently by ``share`` cohere:
    the product of their shares, 1 for a wave with itself."""
    out = share[:, :, None] * share[:, None, :]
    index = np.arange(share.shape[1])
    out[:, index, index] = 1.0
    return out


def _share(kl: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """(C, 3, B, M) |F(X)| of each wave's boundary, ``angle`` (C, 3) away."""
    x = 2.0 * kl[:, None] * np.sin(0.5 * angle)[..., None, None] ** 2
    return np.abs(transition(x))


def _energy(waves: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    """(C, B) energy of waves (C, W, B, M) that cohere by ``coherence`` (C, W, W, B, M),
    averaged over each band's frequencies."""
    return np.einsum("cijbm,cibm,cjbm->cbm", coherence, waves, np.conj(waves)).real.mean(axis=2)


def _in_groups(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every pair (i, j) of an entry of ``a`` and one of ``b`` in the same group (the
    values of both are groups)."""
    order = np.argsort(b, kind="stable")
    lo = np.searchsorted(b[order], a, side="left")
    count = np.searchsorted(b[order], a, side="right") - lo
    i = np.repeat(np.arange(len(a)), count)
    within = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    return i, order[np.repeat(lo, count) + within]


def _take(energy, targets, cross, row_energy, rows):
    """``energy`` with each cross term added to that of its target path, and
    ``row_energy`` of the paths that took them. Where the cross terms take more than a
    path has, the rest comes off the paths that took it, in proportion, so the sum stays
    the same; unless they have less still (two of them nearly cancelling each other)."""
    total = energy.copy()
    np.add.at(total, targets, cross)
    taken = np.maximum(-cross, 0.0)
    owed = np.zeros_like(total)
    np.add.at(owed, targets, taken)
    short = np.maximum(-total, 0.0)[targets]
    row_energy = row_energy.copy()
    np.add.at(row_energy, rows, -short * taken / np.maximum(owed[targets], 1e-300))
    return np.maximum(total, 0.0), np.maximum(row_energy, 0.0)


def _phase_curvature(n, phi, phi_source, kl, reflection, angle) -> np.ndarray:
    """(C, B, M) the second derivative of the phase of an edge's diffracted wave (all its
    terms) with respect to the angle round the edge, at ``phi``: by finite differences
    on the side of the nearest boundary (``angle`` (C, 3) away) that ``phi`` is on,
    where the diffracted wave alone jumps; closer to it than two steps, as it is two
    steps away, so that it changes smoothly up to the boundary."""
    rows = np.arange(len(phi))
    nearest = np.argmin(np.abs(angle), axis=1)
    near = angle[rows, nearest]
    clear = np.abs(near)[:, None, None]
    step = 0.05 / np.sqrt(1.0 + kl)
    # How each boundary's angle changes with phi: the incident wave's, then the
    # reflections' off face 0 and face n.
    slope = np.column_stack([-np.sign(phi - phi_source), -np.ones(len(phi)), np.ones(len(phi))])
    away = np.where(near >= 0.0, 1.0, -1.0) * slope[rows, nearest]
    away = np.where(away == 0.0, 1.0, away)[:, None, None]
    centre = phi[:, None, None] + away * (np.maximum(clear, 2.0 * step) - clear)
    values = []
    for offset in (-1.0, 0.0, 1.0):
        at = centre + offset * away * step
        terms = wedge_terms(n[:, None, None], at, phi_source[:, None, None], kl)
        values.append(
            terms[..., 0]
            + terms[..., 1]
            + terms[..., 3] * reflection[:, 0]
            + terms[..., 2] * reflection[:, 1]
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        bend = (np.angle(values[2] / values[1]) - np.angle(values[1] / values[0])) / step**2
    return np.nan_to_num(bend, nan=0.0, posinf=0.0, neginf=0.0)


def _shortest(a, d1, b, d2, source, listener, iterations: int = 50):
    """Where along the lines a + t1 d1 and b + t2 d2 the shortest path from ``source``
    over the first line and the second to ``listener`` touches them, and whether that
    was found: the length is convex in (t1, t2), and Newton's method with a step halved
    until the length does not grow finds its minimum."""

    def legs(t1, t2):
        p1 = a + t1[:, None] * d1
        p2 = b + t2[:, None] * d2
        return p1 - source, p2 - p1, listener - p2

    def length(t1, t2):
        return sum(np.linalg.norm(x, axis=1) for x in legs(t1, t2))

    middle = 0.5 * (source + listener)
    t1 = np.einsum("ij,ij->i", middle - a, d1)
    t2 = np.einsum("ij,ij->i", middle - b, d2)
    cross = np.einsum("ij,ij->i", d1, d2)
    for _ in range(iterations):
        u, v, w = legs(t1, t2)
        r0, r1, r2 = (np.maximum(np.linalg.norm(x, axis=1), 1e-300) for x in (u, v, w))
        u1, v1 = np.einsum("ij,ij->i", d1, u) / r0, np.einsum("ij,ij->i", d1, v) / r1
        v2, w2 = np.einsum("ij,ij->i", d2, v) / r1, np.einsum("ij,ij->i", d2, w) / r2
        g1, g2 = u1 - v1, v2 - w2
        # Lines that meet draw the path to where they meet, and the curvature there
        # grows without bound (beyond the largest number: no step is solved for, and the
        # path, with no length between the lines, is turned down).
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            h11 = (1.0 - u1**2) / r0 + (1.0 - v1**2) / r1
            h22 = (1.0 - v2**2) / r1 + (1.0 - w2**2) / r2
            h12 = -(cross - v1 * v2) / r1
            det = h11 * h22 - h12**2
            solve = det > 1e-12 * np.maximum(h11 * h22, 1e-300)
            s1 = np.where(solve, -(h22 * g1 - h12 * g2) / det, -g1 / np.maximum(h11, 1e-12))
            s2 = np.where(solve, -(h11 * g2 - h12 * g1) / det, -g2 / np.maximum(h22, 1e-12))
        now = r0 + r1 + r2
        scale = np.ones(len(t1))
        for _ in range(30):
            worse = length(t1 + scale * s1, t2 + scale * s2) > now * (1.0 + 1e-15)
            if not worse.any():
                break
            scale = np.where(worse, 0.5 * scale, scale)
        t1, t2 = t1 + scale * s1, t2 + scale * s2
        if np.all(np.maximum(np.abs(g1), np.abs(g2)) < 1e-12):
            break
    u, v, w = legs(t1, t2)
    r0, r1, r2 = (np.maximum(np.linalg.norm(x, axis=1), 1e-300) for x in (u, v, w))
    g1 = np.einsum("ij,ij->i", d1, u) / r0 - np.einsum("ij,ij->i", d1, v) / r1
    g2 = np.einsum("ij,ij->i", d2, v) / r1 - np.einsum("ij,ij->i", d2, w) / r2
    return t1, t2, np.maximum(np.abs(g1), np.abs(g2)) < 1e-8


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


def _plane_bounds(model: TraceModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, 3) the lowest and highest corner of the box round each plane's faces that are
    not openings, where a reflection off the plane can be (empty boxes, lowest above
    highest, for planes of openings only), and (P,) how far those faces' corners stray
    from the plane at most (faces join a plane within a tolerance)."""
    count = model.plane_count
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    spread = np.zeros(count)
    solid = np.flatnonzero(~model.transparent)
    corners = np.asarray(model.room.mesh.vertices, dtype=float)[model.room.mesh.faces[solid]]
    corners = corners.reshape(-1, 3)
    plane = np.repeat(model.face_plane[solid], 3)
    np.minimum.at(low, plane, corners)
    np.maximum.at(high, plane, corners)
    along = np.einsum("ij,ij->i", corners, model.plane_normal[plane])
    np.maximum.at(spread, plane, np.abs(along - model.plane_offset[plane]))
    return low, high, spread


def _reflecting_pairs(
    model: TraceModel,
    edges: Edges,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray],
    end: np.ndarray,
    planes: np.ndarray,
    among: np.ndarray,
    offset: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The pairs of an edge of ``among`` and a plane of ``planes`` (which ``end`` lies in
    front of) for which a path from the edge to ``end`` can reflect off one of the
    plane's faces, as rows into ``among`` and ``planes``, by edge and then plane: the
    others cannot, and the ray tests of :func:`_reflect` would turn them down.

    The rays start within ``offset`` of a point of the edge (twice that by a very short
    edge), as :meth:`EdgeDiffraction._combos` starts them, and head for ``end``'s image.
    One that starts behind the plane (on the far side from ``end``, beyond where its
    faces stray from it) does not reach it. One from the rest of the edge meets the plane
    at the shadow of its starting point cast from the image, which lies between the
    shadows of the ends of that part of the edge, to within ``4 offset`` plus twice the
    faces' stray, times one plus one over the smaller of the cosines of the rays from
    the image to those ends to the plane's normal: the reflection can only be on a face
    whose box takes in some of that stretch. Pairs whose rays graze the plane, or whose
    image lies too near it for those bounds, are all kept (the native
    ``reflecting_pairs`` does the work)."""
    from acoustic_spaces import _native

    if len(among) == 0 or len(planes) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    low, high, spread = (np.ascontiguousarray(b[planes]) for b in bounds)
    rows, cols = _native.reflecting_pairs(
        np.ascontiguousarray(edges.start[among]),
        np.ascontiguousarray(edges.end[among]),
        np.ascontiguousarray(model.plane_normal[planes], dtype=float),
        np.ascontiguousarray(model.plane_offset[planes], dtype=float),
        low,
        high,
        spread,
        np.asarray(end, dtype=float),
        float(offset),
        _CULL_GRAZING,
        int(threads),
    )
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def _reaching_pairs(
    model: TraceModel,
    edges: Edges,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray],
    end: np.ndarray,
    planes: np.ndarray,
    among: np.ndarray,
    sides: tuple[np.ndarray, np.ndarray, np.ndarray],
    max_distance: float,
    tolerance: float,
    offset: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """For source sides ``sides`` (rows of an edge of ``among``, the position along it and
    the distance from it), the planes of ``planes`` that :func:`_reflecting_pairs` keeps
    for the edge and whose image of ``end`` makes a path over the edge that touches it
    (:func:`_reach`) no further than ``max_distance``, by a little more than the
    exact tests, with the image more than half ``tolerance`` off the edge's line: rows
    into ``sides`` and into ``planes``, by row and then plane."""
    from acoustic_spaces import _native

    low, high, spread = (np.ascontiguousarray(b[planes]) for b in bounds)
    row_edge, t_s, rho_s = sides
    rows, cols = _native.reaching_pairs(
        np.ascontiguousarray(edges.start[among]),
        np.ascontiguousarray(edges.end[among]),
        np.ascontiguousarray(model.plane_normal[planes], dtype=float),
        np.ascontiguousarray(model.plane_offset[planes], dtype=float),
        low,
        high,
        spread,
        np.asarray(end, dtype=float),
        float(offset),
        _CULL_GRAZING,
        np.ascontiguousarray(row_edge, dtype=np.int64),
        np.ascontiguousarray(t_s, dtype=float),
        np.ascontiguousarray(rho_s, dtype=float),
        float(max_distance),
        float(tolerance),
        float(offset),
        int(threads),
    )
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def _reach(kind, length, t_s, rho_s, t_r, rho_r, speed_of_sound, max_delay_s):
    """Where along the edge a path from a source side at ``t_s``, ``rho_s`` to a listener
    side at ``t_r``, ``rho_r`` makes equal angles with it on both sides, the lengths on
    either side, and whether that point lies on the edge and the path ends by
    ``max_delay_s``."""
    t_k = t_s + (t_r - t_s) * rho_s / (rho_s + rho_r)
    r1 = np.hypot(rho_s, t_k - t_s)
    r2 = np.hypot(rho_r, t_r - t_k)
    # At an end of the edge that lies in the plane, the path reflecting there before the
    # edge and the one reflecting after it are the same path: only the first counts at
    # the start of the edge, only the second at its end.
    ok = np.where(kind == 2, (t_k > 0.0) & (t_k <= length), (t_k >= 0.0) & (t_k < length))
    ok &= (kind != 3) | ((t_k > 0.0) & (t_k < length))  # at an end, one of those two
    ok &= (r1 + r2) / speed_of_sound <= max_delay_s
    return ok, t_k, r1, r2


def _first_solid(model: TraceModel, start, direction, length, ignore, through=None):
    """How far along each ray from ``start`` the first face within ``length`` that is not
    an opening (nor one of ``through``, a mask over the faces) lies, and that face, or
    -1."""
    n = len(start)
    face = np.full(n, -1, dtype=np.int64)
    at = np.zeros(n)
    travelled = np.zeros(n)
    skip = np.broadcast_to(np.asarray(ignore, dtype=np.int64), (n,)).copy()
    transparent = model.transparent if through is None else model.transparent | through
    alive = np.arange(n)
    for _ in range(MAX_CROSSINGS + 1):
        if len(alive) == 0:
            break
        t, hit = model.intersect(
            start[alive] + direction[alive] * travelled[alive, None],
            direction[alive],
            length[alive] - travelled[alive],
            skip[alive],
        )
        found = hit >= 0
        rows, hit = alive[found], hit[found]
        travelled[rows] += t[found]
        solid = ~transparent[hit]
        face[rows[solid]] = hit[solid]
        at[rows[solid]] = travelled[rows[solid]]
        alive = rows[~solid]
        skip[alive] = hit[~solid]
    return at, face


def _clear(model: TraceModel, start: np.ndarray, end: np.ndarray, through=None) -> np.ndarray:
    """Whether nothing but openings (and the faces of ``through``, a mask over the
    faces) lies between each start and end point."""
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1e-300)[:, None]
    _, face = _first_solid(model, start, direction, length * (1.0 - 1e-9), -1, through)
    return face < 0


def _reflect(model: TraceModel, start, image, plane, end):
    """Rays from ``start`` towards ``image`` (the mirror image of ``end`` in ``plane``):
    whether they first hit a face of that plane and see ``end`` from there; the points
    and faces they hit."""
    delta = image - start
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1e-300)[:, None]
    t, face = _first_solid(model, start, direction, length, -1)
    hit = (face >= 0) & (model.face_plane[np.maximum(face, 0)] == plane)
    where = start + direction * np.where(hit, t, 0.0)[:, None]
    delta = end - where
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length, 1e-300)[:, None]
    _, blocked = _first_solid(
        model, where, direction, length * (1.0 - 1e-9), np.where(hit, face, -1)
    )
    return hit & (blocked < 0), where, np.where(hit, face, -1)


def _replace_energy(paths: EarlyReflections, energy: np.ndarray) -> EarlyReflections:
    return dataclasses.replace(paths, energy=energy)


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
    coherent: int = 0,
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
        coherent=coherent,
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
