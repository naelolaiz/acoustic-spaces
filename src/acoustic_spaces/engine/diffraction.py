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

In the bands where the paths add in energy, each specular path interferes with the
terms of its own wave only, as it adds to the other paths in energy: each term counts in
two waves (:mod:`~acoustic_spaces.engine.clusters`), the geometric wave it continues, by
c, and the edge's diffracted wave, by sqrt(1 - c^2). On the lit side of its boundary c
is the term's coherence with the geometric wave over the band; on the dark side, where
the term stands in for the wave, c = ``sqrt(1 - |F|^2)`` with F its transition
function, 0 well off the boundary. So in the diffracted path the terms of all the waves add up
coherently, as they must deep in the shadow, except near each wave's boundary, where
they add in energy too. Coherent all the way, levels would step by up to 4 dB where one
wave's boundary is crossed while another's terms are there; in energy all the way, the
shadow of a screen would be 3 dB deeper than Maekawa's chart. As it is, deep behind the
corner of an L-shaped room the 63 Hz band is 0.7 dB below the coherent sum, the 125 Hz
band 0.4 dB, and from 500 Hz up the two agree within 0.1 dB.

With ``first_order`` diffraction the same is done for the first-order reflections: sound
that reflects off a plane and then bends over an edge, and sound that bends over an edge
and then reflects. Where the path touches the edge closer to that plane than the rays
that test it start from the edge, it counts by where the path itself reflects: at that
very point where it lies in the plane, off a face that lies there (the ground an upright
edge stands on), not off a face of the plane elsewhere (a roof level with a barrier's
top).

With ``second_order`` the direct sound also bends over two edges in turn: over a thick
wall or a pillar, round both of its corners. Its shortest path touches each edge where
the path makes equal angles with it on both sides. The wave from the first edge reaches
the second as a wave of its own, which the second bends again: where the path runs along
a face between them (the top of a thick wall), that wave already holds the face's
reflection, so the second edge's coefficient is halved; as it is where the path only
leaves the first edge along a face, which ends before the second (a roof's corner,
then a barrier level with the roof), or only reaches the second along one, so that
the path is the same both ways round. Along a face that absorbs, it is
a point source's wave along a locally reacting surface at grazing incidence (Weyl-van der
Pol): both edges take the face as rigid, and the path the ground wave's loss F(w) over
the way along it. Near the first edge's boundaries its wave still curves like the
source's, not like a wave from the edge, and the second edge's distance parameter takes
that curvature where the wave reaches it; seen from the listener, the same holds for the
second edge's wave at the first. Each edge takes the other's curvature by how much
nearer the other's boundary is, so the path is the same both ways round. In the bands
where the paths add in energy, each term splits at each edge as at one: near the first
edge's boundary it is part of the wave of the path over the second edge alone, near the
second's of the path over the first alone, and well off both the waves of the two
edges add coherently. Where the source's wave reaches the listener past both edges, the
paths over each edge alone both take their terms off it, and the path over both gives
back what they take twice: over a block's top that the source and the listener both
just clear, the direct sound and its reflection off the top less the two corners' terms
come to nothing, and the path over both is all there is, half the direct sound. Only
with the source or the listener exactly in the plane of a face that the path runs along
at the edge next to them (the first edge's, or the second's) is it left out, where the
paths over one edge count once already (:meth:`_in_front`).

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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.clusters import Clusters
from acoustic_spaces.engine.coherent import ground_factors, ground_wave_factors, reflection_phases
from acoustic_spaces.engine.early import EarlyReflections, _face_at
from acoustic_spaces.engine.model import MAX_CROSSINGS, TraceModel, source_weights
from acoustic_spaces.geometry.cells import _components
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
#: How far past a free end of an edge (:func:`end_fades`) the paths over the edge's
#: line are followed, as the argument of the Fresnel integral along the edge at the
#: lowest frequency (:func:`along_edge`); their share tapers to nothing from
#: FADE_TAPER on.
FADE_REACH = 3.0
FADE_TAPER = 2.0
#: Paths nearer the edge's line in angle than this sine reach no further past its ends
#: than at this angle.
_FADE_MIN_SINE = 0.25
#: An edge goes on in another one at an end within this angle (degrees) of its line,
#: and meets it at a right-angled corner within this angle of square; a face whose
#: normal lies within _CLOSING_DEG of the edge's line closes the end it holds.
_COLLINEAR_DEG = 1.0
_CLOSING_DEG = 5.0
#: An edge this close to vertical (the cosine of its angle to the vertical) is upright.
_UPRIGHT = 1.0 - 1e-6
#: The four terms of the wedge coefficient (:func:`wedge_terms`) belong to these waves:
#: the incident wave (0), the reflection off face 0 (1) and that off face n (2).
_TERM_WAVE = np.array([0, 0, 2, 1])
#: :func:`_weighted_tail` integrates by Gauss-Legendre quadrature (these nodes and
#: weights, on 0 to 1) where the smaller of its arguments, times sqrt(pi / 2), is at
#: most _TAIL_SWITCH, and by Gauss-Laguerre quadrature beyond.
_TAIL_SWITCH = 2.5
_LEGENDRE = (
    0.5 * (np.polynomial.legendre.leggauss(12)[0] + 1.0),
    0.5 * np.polynomial.legendre.leggauss(12)[1],
)
_LAGUERRE = np.polynomial.laguerre.laggauss(12)


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
    eps = _term_angles(n, phi, phi_source)
    if lit is None:
        side = 1.0
    else:
        side = np.where(np.asarray(lit)[..., _TERM_WAVE], 1.0, -1.0)
    return _cot_transition(np.asarray(n)[..., None], eps, np.asarray(kl)[..., None], side)


def _term_angles(n: np.ndarray, phi: np.ndarray, phi_source: np.ndarray) -> np.ndarray:
    """(..., 4) the angle of each of the four terms of :func:`wedge_terms` from its
    nearest singularity (:func:`_boundary_distance`)."""
    difference, total = phi - phi_source, phi + phi_source
    return np.stack(
        [
            _boundary_distance(n, difference, 1.0),
            _boundary_distance(n, difference, -1.0),
            _boundary_distance(n, total, 1.0),
            _boundary_distance(n, total, -1.0),
        ],
        axis=-1,
    )


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
    #: (E, 2) how far the paths over each edge fade at its start and end, (E, 2, 2) the
    #: edge it meets at a corner there in the plane of face 0 and of face n, or -1, and
    #: (E, 2) whether a face across it closes the end (end_fades)
    fade: np.ndarray | None = None
    corner: np.ndarray | None = None
    closed: np.ndarray | None = None

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
    geometry, owner, h0, h1, single, theta, side, keep = _wedges(model, min_bend_deg)
    normals = model.plane_normal[model.face_plane]  # into the room when oriented
    parts = []
    # Two faces: a wedge, if it opens far enough beyond flat towards the room.
    p0, p1, e, a0 = geometry(h0)
    face0, face1 = owner[h0], owner[h1]
    normal = np.where((side > 0.0)[:, None], -normals[face0], normals[face0])
    parts.append((p0, p1, e, a0, normal, (2.0 * np.pi - theta) / np.pi, face0, face1, keep))
    # One face: the free edge of a surface, a wedge open all round.
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
    edges = _join(model, p0, p1, e, along, normal, wedge, face0, face1)
    edges.fade, edges.corner, edges.closed = end_fades(model, edges)
    return edges


def sheets(model: TraceModel, min_bend_deg: float = MIN_BEND_DEG) -> np.ndarray:
    """(F,) the sheet each face of ``model`` belongs to, a label: the faces that
    :func:`find_edges` looks at, joined where two of them share an edge of the mesh that
    is no inside corner, one that bends by less than ``min_bend_deg`` (a curved barrier's
    facets, a flat wall's triangles) or that diffracts (a barrier's bend, a building's
    corners and the edges of its roof): one barrier, one building, one pillar each. An
    inside corner (a pillar's foot on the floor, where two walls of a room meet) parts
    them. In a model that is not oriented, which says not which side of a face is solid,
    no bend can be told from an inside corner, and only those of less than
    ``min_bend_deg`` join."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    _, owner, h0, h1, _, theta, _, keep = _wedges(model, min_bend_deg)
    join = theta > np.pi - math.radians(min_bend_deg)
    if model.oriented:
        join |= keep
    count = len(model.face_plane)
    graph = coo_matrix(
        (np.ones(int(join.sum())), (owner[h0][join], owner[h1][join])), shape=(count, count)
    )
    return connected_components(graph, directed=False)[1]


def _wedges(
    model: TraceModel, min_bend_deg: float
) -> tuple[
    Callable, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """The edges of the mesh that two of ``model``'s solid faces share, as
    :func:`_halves` gives them, with the half on the face of the lower plane first; the
    angle between the two faces, which side of the first face's normal the second lies
    on, and whether the edge diffracts: it opens far enough beyond flat towards the room.
    Also :func:`_halves`' geometry, the faces of the halves and the halves of the edges
    of one face."""
    geometry, owner, h0, h1, single = _halves(model)
    normals = model.plane_normal[model.face_plane]  # into the room when oriented
    swap = model.face_plane[owner[h0]] > model.face_plane[owner[h1]]
    h0, h1 = np.where(swap, h1, h0), np.where(swap, h0, h1)
    a0, a1 = geometry(h0)[3], geometry(h1)[3]
    theta = np.arccos(np.clip(np.einsum("ij,ij->i", a0, a1), -1.0, 1.0))
    side = np.einsum("ij,ij->i", normals[owner[h0]], a1)
    keep = theta <= np.pi - math.radians(min_bend_deg)
    if model.oriented:
        # Face n lies behind face 0: the solid is inside the angle. Between faces with a
        # room on both sides (a bent panel), the side that opens wider diffracts.
        keep &= (side < 0.0) | model.plane_two_sided[model.face_plane[owner[h0]]]
    return geometry, owner, h0, h1, single, theta, side, keep


def _halves(model: TraceModel) -> tuple[Callable, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The edges of the mesh of ``model``'s solid faces (with an outdoor scene's sky, which
    is an opening but no hole: the ground goes on past it), as halves, one for each
    corner of a face: a function of halves that gives the ends of each, its direction and
    the unit vector across it in its face, away from it; the face of each half; the two
    halves of each edge that two faces share, and the halves of the edges of one face."""
    mesh = model.room.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    solid = ~model.transparent  # openings are holes here
    if model.sky is not None:
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

    def geometry(half: np.ndarray):
        p0, p1 = vertices[lo[half]], vertices[hi[half]]
        length = np.linalg.norm(p1 - p0, axis=1)
        e = (p1 - p0) / np.maximum(length, 1e-300)[:, None]
        v = vertices[opposite[half]] - p0
        a = v - np.einsum("ij,ij->i", v, e)[:, None] * e
        a /= np.maximum(np.linalg.norm(a, axis=1), 1e-300)[:, None]
        return p0, p1, e, a

    pair = starts[uses == 2]
    return geometry, owner, order[pair], order[pair + 1], order[starts[uses == 1]]


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


def end_fades(model: TraceModel, edges: Edges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(E, 2) how far the paths over each edge fade at its start and at its end
    (:func:`along_edge`), (E, 2, 2) the edge each meets at a corner there in the plane of
    its face 0 and of its face n, or -1, and (E, 2) whether a face across the edge closes
    the end, where the edge goes on as its image.

    A path over an edge comes from the stretch of it round the point it touches, about a
    Fresnel zone long; near an end only part of that stretch is there. An end the edge
    goes on from, in a collinear edge or as its own image in a face across it (the
    ground under a barrier's end, the wall a balcony front ends at), cuts nothing off:
    0, the paths stop there as on an endless edge. At a right-angled corner, where
    another edge meets it within 1 degree of square, the paths fade halfway (1/2): past
    the end of one edge the point a path touches reaches the other, and the two halves
    add up as the edges of a quarter plane do in the Kirchhoff approximation, which is
    the product of two Fresnel integrals only when they are square. Elsewhere (a free
    end, one another edge goes on from round a bend, or a corner of another angle) they
    fade all the way (1), and each term of the edge is the edge's own part of it, as the
    Biot-Tolstoy-Medwin integral over the finite edge has it near the path's point
    (:func:`_cumulative`); so are those of every edge that meets it there.
    """
    from scipy.spatial import cKDTree

    count = len(edges)
    neighbour = np.full((2 * count, 2), -1, dtype=np.int64)
    if count == 0:
        return np.ones((0, 2)), neighbour.reshape(0, 2, 2), np.zeros((0, 2), dtype=bool)
    size = float(np.ptp(model.room.mesh.vertices, axis=0).max()) or 1.0
    tol = 1e-6 * size
    points = np.concatenate([edges.start, edges.end])  # the starts, then the ends
    outward = np.concatenate([-edges.direction, edges.direction])
    which = np.concatenate([np.arange(count), np.arange(count)])
    collinear = np.zeros(2 * count, dtype=bool)
    bend = np.zeros(2 * count, dtype=bool)
    corner = np.zeros(2 * count, dtype=bool)
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        other = which[a] != which[b]
        a, b = a[other], b[other]
        # The cosine of the turn from one edge into the other at their common end: none
        # (it goes on), a right angle, or another (edges doubling back do not meet).
        turn = -np.einsum("ij,ij->i", outward[a], outward[b])
        aligned = math.cos(math.radians(_COLLINEAR_DEG))
        straight = turn >= aligned
        sharp = np.abs(turn) <= math.sin(math.radians(_COLLINEAR_DEG))
        slight = ~straight & ~sharp & (turn > -aligned)
        for flags, mask in ((collinear, straight), (bend, slight), (corner, sharp)):
            np.logical_or.at(flags, a, mask)
            np.logical_or.at(flags, b, mask)
        # Where one of the edges that meet at a point is on its own, all of them are.
        own = bend.copy()
        np.logical_or.at(bend, a, own[b])
        np.logical_or.at(bend, b, own[a])
        # The edge met at a corner in the plane of each face: one with a face there too.
        for one, two in ((a[sharp], b[sharp]), (b[sharp], a[sharp])):
            for side in (0, 1):
                plane = edges.plane[which[one], side]
                shared = (edges.plane[which[two], 0] == plane) | (
                    edges.plane[which[two], 1] == plane
                )
                neighbour[one[shared], side] = which[two[shared]]
    sides = edges.bisector()
    closed = _closed_ends(model, points, outward, np.concatenate([sides, sides]), tol)
    fade = np.where(collinear | closed, 0.0, np.where(bend, 1.0, np.where(corner, 0.5, 1.0)))
    # Only the edges of thin screens fade, and where they meet others only where every
    # one of those is one: at the corners of solid wedges (a box's, where three edges
    # meet) the halves do not add up as BTM has them (a fade of every edge there came
    # out up to 1.25 dB above BTM at 63 and 125 Hz near a block's corner), and their
    # paths stop at the ends as on an endless edge.
    thin = np.isclose(edges.wedge, 2.0)
    met = np.ones(2 * count, dtype=bool)
    if len(pairs):
        np.logical_and.at(met, a[sharp | slight], thin[which[b[sharp | slight]]])
        np.logical_and.at(met, b[sharp | slight], thin[which[a[sharp | slight]]])
    fade = np.where(thin[which] & met, fade, 0.0)
    closed &= thin[which]
    neighbour[fade != 0.5] = -1
    return (
        fade.reshape(2, count).T.copy(),
        neighbour.reshape(2, count, 2).transpose(1, 0, 2).copy(),
        (closed & ~collinear).reshape(2, count).T.copy(),
    )


def _closed_ends(model: TraceModel, points, outward, sides, tol) -> np.ndarray:
    """Whether each end (``points``, the edge leaving it against ``outward``, its open
    side towards ``sides``) lies on a face across the edge's line that reflects towards
    the edge, where the edge goes on as its image: the ground or a solid face that goes
    on from the end into the edge's open side. At a convex corner (of a box) the faces
    across the edge lie behind it, and the end does not close."""
    closing = math.cos(math.radians(_CLOSING_DEG))
    into = -outward  # along the edge, away from its end
    closed = np.zeros(len(points), dtype=bool)
    if model.ground is not None:
        layout = model.room.outdoor
        if layout is not None and layout.terrain is not None:
            grounded = points[:, 2] <= layout.ground_at(points) + ON_GROUND_M
        else:
            grounded = on_ground(points, model.ground)
        closed |= grounded & (into[:, 2] >= closing)
    mesh = model.room.mesh
    vertices = np.asarray(mesh.vertices, dtype=float)
    solid = np.flatnonzero(~model.transparent)
    planes = model.face_plane[solid]
    order = np.argsort(planes, kind="stable")
    ranked = planes[order]
    normal, offset = model.plane_normal, model.plane_offset
    for begin in range(0, len(points), 512):
        rows = np.arange(begin, min(begin + 512, len(points)))
        along = points[rows] @ normal.T - offset[None, :]
        across = np.abs(into[rows] @ normal.T) >= closing
        hit_rows, hit_planes = np.nonzero(across & (np.abs(along) <= tol))
        for row, plane in zip(rows[hit_rows], hit_planes, strict=True):
            if closed[row]:
                continue
            # The face must reflect towards the edge (its front, or either side).
            if model.oriented and not model.plane_two_sided[plane]:
                if float(normal[plane] @ into[row]) <= 0.0:
                    continue
            lo, hi = np.searchsorted(ranked, [plane, plane + 1])
            faces = solid[order[lo:hi]]
            if not len(faces):
                continue
            side = sides[row] - float(sides[row] @ normal[plane]) * normal[plane]
            side /= max(float(np.linalg.norm(side)), 1e-300)
            beside = points[row] + 100.0 * tol * side
            if _on_triangles(beside, vertices[mesh.faces[faces]], tol):
                closed[row] = True
    return closed


def _on_triangles(point: np.ndarray, triangles: np.ndarray, tol: float) -> bool:
    """Whether ``point``, in the triangles' plane, lies on any of ``triangles`` (T, 3,
    3), their edges included."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, point[None, :] - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01**2
    ok = denominator > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (d11 * d20 - d01 * d21) / denominator
        w = (d00 * d21 - d01 * d20) / denominator
    scale = np.sqrt(np.maximum(d00, d11))
    eps = tol / np.maximum(scale, 1e-300)
    return bool(np.any(ok & (v >= -eps) & (w >= -eps) & (v + w <= 1.0 + eps)))


def _fade_reach(r1, r2, sin_beta, k_min: float) -> np.ndarray:
    """How far (m) past a fading end of an edge a path touching its line at distances
    ``r1`` and ``r2`` from the source and listener sides, at ``sin_beta`` to it, is
    followed: FADE_REACH over the Fresnel integral's argument per metre at ``k_min``."""
    spread = np.sqrt(np.pi * r1 * r2 / ((r1 + r2) * k_min))
    return FADE_REACH * spread / np.maximum(sin_beta, _FADE_MIN_SINE)


def near_fading_end(fades, t_k, length, r1, r2, sin_beta, k_min: float):
    """(C,) whether each path over an edge touches its line within reach of an end that
    fades (``fades`` (C, 2), :func:`end_fades`), and (C,) that reach
    (:func:`_fade_reach`). There its terms count whether their paths are found or not."""
    reach = _fade_reach(r1, r2, sin_beta, k_min)
    near = (fades[:, 0] > 0.0) & (t_k < reach)
    near |= (fades[:, 1] > 0.0) & (length - t_k < reach)
    return near, reach


def _fresnel_tail_share(u: np.ndarray) -> np.ndarray:
    """The integral of exp(-j pi t^2 / 2) from ``u`` >= 0 to infinity over that along the
    whole line: erfc(u sqrt(pi / 2) e^(j pi / 4)) / 2."""
    tail_c, tail_s = fresnel_tail(u)
    return (tail_c - 1j * tail_s) / (1.0 - 1j)


def _weighted_tail(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The integral of exp(-j pi t^2 / 2) / (t^2 + v^2) from ``u`` >= 0 to infinity over
    that along the whole line, for ``v`` >= 0 (shaped like ``u``): 1/2 at u = 0, the
    Fresnel integral's share (:func:`_fresnel_tail_share`) for large v, and for v = 0
    nothing beyond any u > 0.

    With q = u sqrt(pi / 2), p = v sqrt(pi / 2) and f the Fresnel integral's share:
    where the smaller of p and q is at most _TAIL_SWITCH, from W(x, y), the integral of
    exp(-j x^2 (1 + s^2)) / (1 + s^2) over s from 0 to y / x <= 1 (Owen's T function at
    a complex argument, by Gauss-Legendre quadrature: its phase turns by y^2 at most),
    1/2 - W(p, q) / (2 pi f(p)) for q <= p and, as W(p, q) + W(q, p) =
    pi (f(p) + f(q) - 2 f(p) f(q)), (W(q, p) / pi - f(q) (1 - 2 f(p))) / (2 f(p)) for
    q > p. Elsewhere along the path of steepest descent from q, t^2 = q^2 - j s, by
    Gauss-Laguerre quadrature. Good to about 1e-8, as the Fresnel integrals are.
    """
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    q, p = math.sqrt(0.5 * np.pi) * u, math.sqrt(0.5 * np.pi) * v
    out = np.empty(u.shape, dtype=complex)

    def owen(x, y):
        ratio = y / np.maximum(x, 1e-300)
        s2 = (ratio[:, None] * _LEGENDRE[0]) ** 2
        return ratio * ((np.exp(-1j * (x**2)[:, None] * (1.0 + s2)) / (1.0 + s2)) @ _LEGENDRE[1])

    far = np.minimum(p, q) > _TAIL_SWITCH
    low = ~far & (q <= p)
    if low.any():
        out[low] = 0.5 - owen(p[low], q[low]) / (2.0 * np.pi * _fresnel_tail_share(v[low]))
    high = ~far & (q > p)
    if high.any():
        whole = _fresnel_tail_share(v[high])
        rest = _fresnel_tail_share(u[high]) * (1.0 - 2.0 * whole)
        out[high] = (owen(q[high], p[high]) / np.pi - rest) / (2.0 * whole)
    if far.any():
        square = (q[far] ** 2)[:, None] - 1j * _LAGUERRE[0]
        along = (-0.5j / (np.sqrt(square) * (square + (p[far] ** 2)[:, None]))) @ _LAGUERRE[1]
        phase = np.exp(-1j * (q[far] ** 2 + p[far] ** 2))
        out[far] = phase * p[far] * along / (2.0 * np.pi * _fresnel_tail_share(v[far]))
    return out


def _cumulative(u: np.ndarray, boundary: np.ndarray | None = None) -> np.ndarray:
    """The integral of exp(-j pi t^2 / 2) from minus infinity to ``u`` over its total: how
    much of a path over an endless edge the edge gives up to ``u`` past the point the
    path touches, in the argument of the Fresnel integral along it.

    With ``boundary`` (v >= 0, shaped like ``u``), the same of exp(-j pi t^2 / 2) /
    (t^2 + v^2): the integrand of the Biot-Tolstoy-Medwin solution along the edge near
    that point, for one of its four terms, the term's shadow or reflection boundary
    being v away in the same argument (:func:`term_boundaries`). Near its boundary a term
    comes from the edge within about v of the point only: on the boundary from the point
    itself (0 before it, 1 past it), so that a finite edge's term keeps all of its step
    there while the point lies on the edge and has none once it lies past an end. Far
    from the boundary the weight is flat, which leaves the Fresnel integral."""
    u = np.asarray(u, dtype=float)
    if boundary is None:
        tail = _fresnel_tail_share(np.abs(u))
    else:
        tail = _weighted_tail(np.abs(u), boundary)
    return np.where(u >= 0.0, 1.0 - tail, tail)


def term_boundaries(n: np.ndarray, eps: np.ndarray, kl: np.ndarray) -> np.ndarray:
    """(C, 4, B, M) how far each of the four terms of the wedge coefficient is from its
    boundary, as an argument of the Fresnel integral along the edge (:func:`_cumulative`),
    for edges open over ``n`` (C,) pi, the terms' angles ``eps`` (C, 4,
    :func:`_term_angles`) and wave numbers times distance parameters ``kl`` (C, B, M).

    A term of the Biot-Tolstoy-Medwin integrand is sin(nu phi) / (cosh(nu eta) - cos(nu
    phi)) with nu = 1 / n, phi = eps and eta = d sin^2(beta) / L at d along the edge from
    the path's point: to second order in eta, 1 / (eta^2 + w^2) with
    w = 2 n |sin(eps / 2n)|. The Fresnel integral's argument is eta sqrt(kL / pi) there."""
    width = 2.0 * n[:, None] * np.abs(np.sin(eps / (2.0 * n[:, None])))
    return width[:, :, None, None] * np.sqrt(kl / np.pi)[:, None]


def along_edge(
    t_k, length, scale, fade, reach, inside=None, closed=None, alone=None, boundary=None
) -> np.ndarray:
    """(C, W, B, M) the share of each path over an edge that the edge, from 0 to
    ``length`` (C,) along its line, gives the path touching the line at ``t_k`` (C,):
    the Fresnel integral of the path's phase along the edge over that along the whole
    line, ``scale`` (C, B, M) being the integral's argument per metre along it
    (sin^2 beta sqrt(k / (pi L))). Each end counts by its ``fade`` (C, 2 or C, W, 2,
    :func:`end_fades`) as that integral, and by the rest as a hard stop where ``inside``
    (C, W, 2, for each of W waves and each end) turns false: by default (W = 1, and then
    (C, B, M)) where the point leaves the edge. A free end gives 1/2, the share 1 well
    inside and 0 well past it. Within ``reach`` (C, :func:`_fade_reach`) of an end it
    tapers, from FADE_TAPER on, to 1 inside the edge and to 0 past it. Where a face
    closes one end (``closed``, C, 2) the edge goes on as its image in the face, as far
    as the image of its other end, which fades as that end does.

    At the ends that ``alone`` (C, W, 2) marks, where nothing completes the edge's terms
    (:func:`end_fades`), the integral is weighted for each wave's ``boundary`` (C, W, B,
    M; the W waves are then the four terms of :func:`term_boundaries`), as
    :func:`_cumulative` has it."""
    t_k, length = np.asarray(t_k, float), np.asarray(length, float)

    def window(distance):
        start = FADE_TAPER / FADE_REACH * reach
        x = np.clip((distance - start) / np.maximum(reach - start, 1e-300), 0.0, 1.0)
        return (0.5 * (1.0 + np.cos(np.pi * x)))[:, None, None, None]

    before, after = t_k < 0.0, t_k > length
    squeeze = inside is None
    if squeeze:
        inside = np.stack([~before, ~after], axis=-1)[:, None, :]
    if fade.ndim == 2:
        fade = fade[:, None, :]

    def beyond(distance, end):
        """(C, W or 1, B, M) what the line gives further than ``distance`` (C,) from the
        point the path touches, as the end ``end`` of the edge counts it."""
        u = distance[:, None, None] * scale
        plain = _cumulative(-u)[:, None]
        if alone is None or not alone[..., end].any():
            return plain
        own = alone[..., end]
        out = np.broadcast_to(plain, boundary.shape).copy()
        out[own] = _cumulative(-np.broadcast_to(u[:, None], boundary.shape)[own], boundary[own])
        return out

    w_a, w_b = fade[..., 0, None, None], fade[..., 1, None, None]  # (C, W, 1, 1)
    # What each end takes off a whole line's 1: its integral's part and its hard stop's.
    off_a = w_a * beyond(t_k, 0) + (1.0 - w_a) * ~inside[..., 0, None, None]
    off_b = w_b * beyond(length - t_k, 1) + (1.0 - w_b) * ~inside[..., 1, None, None]
    on = ~(before | after)
    out = 1.0 - off_a * np.where(on[:, None, None, None], window(t_k), 1.0)
    out = out - off_b * np.where(on[:, None, None, None], window(length - t_k), 1.0)
    if closed is not None:
        # The image of the far end, beyond the end a face closes.
        for end, image, far in ((0, -length, w_b), (1, 2.0 * length, w_a)):
            mirrored = closed[:, end] & on
            if not mirrored.any():
                continue
            gap = np.abs(image - t_k)
            off = far * beyond(gap, 1 - end) * mirrored[:, None, None, None]
            out = out - off * window(gap)
    past = np.where(before, -t_k, np.where(after, t_k - length, 0.0))
    out = out * np.where(on[:, None, None, None], 1.0, window(past))
    return out[:, 0] if squeeze else out


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
        #: The paths :meth:`apply` returned last and the waves of their pieces.
        self.ledger: tuple[EarlyReflections, Clusters] | None = None
        #: The paths :meth:`apply` returned last and its paths over edges in a row
        #: (:meth:`_in_line`) that carry too little to be among them, or None.
        self.unheard: tuple[EarlyReflections, EarlyReflections] | None = None
        self.size = float(np.ptp(model.room.mesh.vertices, axis=0).max()) or 1.0
        frequencies, self.slice_hz = band_frequencies(bands=model.bands)
        self.wavenumber = 2.0 * np.pi * frequencies / self.speed_of_sound  # (B, M)
        self.k_min = float(self.wavenumber.min())
        if self.edges.fade is None or self.edges.corner is None or self.edges.closed is None:
            self.edges.fade, self.edges.corner, self.edges.closed = end_fades(model, self.edges)
        self.fades, self.corners = self.edges.fade, self.edges.corner
        self.closed = self.edges.closed

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

    def _continued(self, kl, angle, lit, bands: slice) -> np.ndarray:
        """:func:`_continued` in ``bands``, for ``kl`` there."""
        return _continued(
            kl, angle, lit, self.wavenumber[bands], self.slice_hz[bands], self.speed_of_sound
        )

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
        edges = self.edges
        if np.any(self.fades > 0.0):
            # Past the ends that fade, as far as any path's fade reaches.
            half = 0.5 * self.speed_of_sound * max_delay_s
            reach = _fade_reach(half, half, 0.0, self.k_min)
            before, after = (np.where(self.fades[:, i] > 0.0, reach, 0.0) for i in (0, 1))
            edges = dataclasses.replace(
                edges,
                start=edges.start - before[:, None] * edges.direction,
                length=edges.length + before + after,
            )
            t_s = t_s + before[row_edge]
        return _reaching_pairs(
            self.model,
            edges,
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
        shadows are added. In the bands where the paths add in energy, the waves of
        their terms (:mod:`~acoustic_spaces.engine.clusters`) are kept in
        :attr:`ledger` with the paths returned, for the edge fade to add to."""
        listener = np.asarray(listener, dtype=float)
        self.ledger = self.unheard = None
        if len(self.edges) == 0:
            return paths
        combos = self._combos(listener, max_delay_s)
        if combos is not None:
            combos = self._in_front(listener, paths, combos)
        clusters = self.clusters(listener, paths)
        first = len(paths)
        single = None if combos is None else self._single(listener, paths, combos, clusters, first)
        if single is not None:
            first += len(single["distance"])
        double = None
        if self.two_edges:
            double = self._double(listener, max_delay_s, single, clusters, first)
        if single is None and double is None:
            if clusters is not None:
                self.ledger = (paths, clusters)
            return paths
        count = first + (0 if double is None else len(double["distance"]))
        starts = (len(paths), first)
        if clusters is not None:
            energy, has = clusters.resolve(count)
            paths = self.settled(paths, energy[: len(paths)], has[: len(paths)])
            for found, start in zip((single, double), starts, strict=True):
                if found is not None:
                    rows = slice(start, start + len(found["distance"]))
                    found["energy"][:, self.coherent :] = energy[rows]
        # The direct sound in free field; a listener on the source counts as just off it.
        squared = max(float(np.sum((listener - self.source) ** 2)), (_TOLERANCE * self.size) ** 2)
        floor = MIN_RELATIVE_ENERGY / (4.0 * np.pi * squared)
        parts, ids = [paths], [np.arange(len(paths))]
        quiet = np.zeros(0, dtype=np.int64)
        for found, start in zip((single, double), starts, strict=True):
            if found is not None:
                emitted, idx = self._emit(listener, found, floor, max_delay_s)
                parts.append(emitted)
                ids.append(start + idx)
                if found is single:
                    quiet = np.setdiff1d(single["in_line"], idx)
        out = EarlyReflections.concatenate(parts, self.model.absorption.shape[1])
        if len(quiet):
            # Paths over edges in a row whose terms cancel in the bands that add in
            # energy are still there for the edge fade.
            self.unheard = (out, self._emit(listener, single, floor, max_delay_s, quiet)[0])
        if clusters is not None:
            # Where each piece's path is among those returned (sorted by arrival).
            ids = np.concatenate(ids)
            delay = np.concatenate([p.delay_s for p in parts])
            mapping = np.full(count, -1, dtype=np.int64)
            mapping[ids[np.argsort(delay, kind="stable")]] = np.arange(len(ids))
            clusters.renumber(mapping)
            self.ledger = (out, clusters)
        return out

    def clusters(self, listener: np.ndarray, paths: EarlyReflections) -> Clusters | None:
        """The waves of the specular paths ``paths`` at ``listener`` in the bands where
        the paths add in energy, each path its own; None without such bands."""
        late = slice(self.coherent, None)
        if self.coherent >= self.model.absorption.shape[1]:
            return None
        out = Clusters(self.slice_hz[late], self.speed_of_sound, self.size)
        rows = np.flatnonzero((paths.edge < 0) & (paths.crossings == 0))
        if len(rows):
            energy = paths.energy[rows][:, late]
            distance = paths.distance_m[rows]
            phase = np.exp(-1j * self.wavenumber[None, late] * distance[:, None, None])
            out.add(
                rows,
                out.geometric(_image_sources(listener, paths)[rows], paths.order[rows]),
                np.sqrt(energy)[:, :, None] * phase,
                distance,
                anchor=True,
                own=energy,
            )
        return out

    def settled(self, paths: EarlyReflections, energy: np.ndarray, has: np.ndarray):
        """``paths`` with the ``energy`` (n, bands where the paths add in energy) the
        waves give the paths that ``has`` says are in them."""
        if not has.any():
            return paths
        total = paths.energy.copy()
        total[np.ix_(np.flatnonzero(has), np.arange(self.coherent, total.shape[1]))] = energy[has]
        return _replace_energy(paths, total)

    def _held(self, edge: np.ndarray, *planes: np.ndarray) -> np.ndarray:
        """(n, 2) whether the start and the end of each of ``edge`` (n,) lie in one of the
        ``planes`` (each (n,), -1 for none) its path reflects off."""
        ends = np.stack([self.edges.start[edge], self.edges.end[edge]], axis=1)  # (n, 2, 3)
        out = np.zeros((len(edge), 2), dtype=bool)
        for plane in planes:
            normal = self.model.plane_normal[np.maximum(plane, 0)]
            off = np.einsum("nei,ni->ne", ends, normal)
            off -= self.model.plane_offset[np.maximum(plane, 0), None]
            out |= (plane >= 0)[:, None] & (np.abs(off) <= _TOLERANCE * self.size)
        return out

    def _in_front(self, listener: np.ndarray, paths: EarlyReflections, c: dict) -> dict | None:
        """``c`` (:meth:`_combos`) without the paths whose listener side lies behind a
        face of their edge, or None where none is left. A listener side exactly in the
        plane of face 0 or face n, seen along the edge, beyond the face's end, with the
        source side in that plane too (:func:`_past_face`: a building's wall or a thick
        barrier's top in line with the source and the listener) lies behind the face
        where the straight way between them, the incident wave's path, is not found, as
        just off the plane on that side, inside the wedge, where the edge has no path.
        That is the side the way runs on in the wind
        (:meth:`~acoustic_spaces.engine.refracted.Refraction._edge_side`). The wave along
        the face is then the path over the edge at its other end, which the listener side
        lies in front of: the paths over the two edges are one ray there, and both
        together doubled its pressure. A listener on the face itself (on the wall, or at
        its far corner) has only the side in front of it, and its paths stay."""
        hearer = np.broadcast_to(listener, (len(c["edge"]), 3))
        tolerance = _TOLERANCE * self.size
        past = _past_face(
            self.model, self.edges, c["edge"], c["phi_r"], c["phi_s"], hearer, tolerance
        )
        rows = np.flatnonzero(past)
        if not len(rows):
            return c
        found = self._specular(listener, paths, _subset(c, rows))[0]
        keep = np.ones(len(c["edge"]), dtype=bool)
        keep[rows[found[:, 0] < 0]] = False
        if keep.all():
            return c
        return _subset(c, keep) if keep.any() else None

    def _in_line(self, c: dict, missing, angle, image, order):
        """Edges in a row on a straight way whose path is ``missing`` (C,) although the
        paths over them are found, among the paths ``c`` over one edge
        (:meth:`_combos`): the rows that count on the lit side of their incident wave's
        boundary, the one row of each such way that carries the wave itself, all the
        rows on those ways, and (C,) whether a row's incident wave is one of those
        waves. ``angle`` (C,) is the listener side's from the boundary, ``image``
        (C, 3) and ``order`` (C,) the wave's image of the source and its order.

        The rays that look for a path over an edge start just off the edge, and pass
        what the straight way between the source and the listener sides does not: two
        screens, with the source and the listener exactly level with both tops (or
        within the rays' reach below that line), have a path over each top and over
        both, but no direct sound. Each path over one edge then stood for all the
        direct sound on the dark side of its boundary, and the two together gave it
        twice. The paths that the rays find are those of a straight way that is clear,
        and the level is the one just above the line: the wave is there, and every
        edge on the way takes its term off it, on the lit side of its boundary, by the
        share its edge gives (a path near an end of its edge that fades, or past it).
        One row carries the wave, as its term on the dark side, where the term and the
        wave are one, and the rest of the wave where its share is not all of it
        (:meth:`_single`). It is one that touches its edge; which of several such
        rows it is changes nothing, and it is the one with the lowest index. A way
        that touches none of the edges it passes (it passes beside the ends of all of
        them) is left as it is: something else hides its path, whose edges stand for
        the wave. Edges that meet where the way passes (a corner) are one place, and a
        way with one place only is left as it is too: there the edges' terms on the
        dark side add up to the wave less their lit terms. So is a reflected way with
        the reflection between two of its edges (a wall beside two screens): the terms
        that make reflections fade at edges take the product of the two edges' terms
        on the dark side there, the level just below the line
        (:mod:`~acoustic_spaces.engine.edge_fade`)."""
        reach = 2.0 * _OFFSET * self.size
        t_k, length = c["t_k"], self.edges.length[c["edge"]]
        across = np.abs(angle) * c["rho_s"] * c["r2"] / (c["r1"] + c["r2"])
        rows = np.flatnonzero(missing & (angle <= _ON_BOUNDARY) & (across <= reach))
        of_way = np.zeros(len(t_k), dtype=bool)
        if len(rows) < 2:
            return rows[:0], rows[:0], rows[:0], of_way
        wave = np.column_stack([np.round(image / (1e-6 * self.size)), order])
        groups = np.unique(wave, axis=0, return_inverse=True)[1].ravel()
        group = groups[rows]
        past = (t_k[rows] < 0.0) | (t_k[rows] > length[rows])
        by = np.lexsort((c["edge"][rows], past, group))
        rows, group, past = rows[by], group[by], past[by]
        first = np.concatenate([[True], group[1:] != group[:-1]])
        lead = np.flatnonzero(first)[np.cumsum(first) - 1]
        apart = np.linalg.norm(c["point"][rows] - c["point"][rows[lead]], axis=1) > reach
        # The ways with edges in a row, one of which touches its edge: their rows.
        several = np.zeros(int(groups.max()) + 1, dtype=bool)
        several[group[apart & ~past[lead]]] = True
        # Not with a reflection between two of the way's edges (the paths over them
        # reflect on different sides of their edges).
        kind = c["kind"][rows]
        several[group[kind != kind[lead]]] = False
        clear = several[group]
        of_way = several[groups]
        return (
            np.sort(rows[clear & ~first]),
            np.sort(rows[clear & first]),
            np.sort(rows[clear]),
            of_way,
        )

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
        past = (self.fades[e], self.k_min, self._held(e, self.source_planes[side[rows]]))
        ok, t_k, r1, r2 = _reach(
            kind, edges.length[e], t_s[rows], rho_s[rows], t_l[e], rho_l[e], c, max_delay_s, past
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
                held = self._held(pe, plane, self.source_planes[side[rows]])
                reach, t_k, r1, r2 = _reach(
                    image_kind,
                    edges.length[pe],
                    t_s[rows],
                    rho_s[rows],
                    t_i,
                    rho_i,
                    c,
                    max_delay_s,
                    (self.fades[pe], self.k_min, held),
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
        # A path that touches the line of its edge within the tolerance of an end
        # touches the end: which side of it the rounding leaves it on decides nothing.
        length = edges.length[edge]
        t_k = np.where(np.abs(t_k) <= tolerance, 0.0, t_k)
        t_k = np.where(np.abs(t_k - length) <= tolerance, length, t_k)
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
                # Where the edge's point lies no farther from the plane the path is to
                # reflect off than the ray starts from the point, the ray's start does not
                # stand for the point: the ray can meet a face of that plane (a roof level
                # with a barrier's top) where the path does not reflect. The path reflects
                # where the line from the point to the image crosses the plane, at the
                # point itself where that lies in the plane (the ground an upright edge
                # stands on), and counts if a face of the plane lies there. From a point
                # beyond the plane no line crosses it: the point, off every face, is tried.
                at_edge = point[via]
                normal = self.model.plane_normal[through]
                height = np.einsum("ij,ij->i", normal, at_edge) - self.model.plane_offset[through]
                beyond = np.einsum("ij,ij->i", normal, target) - self.model.plane_offset[through]
                across = (height * beyond < 0.0) & (np.abs(height) > tolerance)
                share = np.where(across, height / np.where(across, height - beyond, 1.0), 0.0)
                crossing = at_edge + share[:, None] * (target - at_edge)
                for i in np.flatnonzero(hit & (np.abs(height) <= offset)):
                    hit[i] = _face_at(self.model, int(through[i]), crossing[i], tolerance) >= 0
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
            "listener_side": rcv[sel],
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
            "t_k": t_k[sel],
        }

    def _single(self, listener, paths: EarlyReflections, c: dict, clusters, start: int) -> dict:
        """The paths over one edge: each wave's terms and their energy in the bands where
        the terms add in pressure; in the others, their pieces go to ``clusters`` (if
        any), as rows from ``start``."""
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
        found, images, orders, front = self._specular(listener, paths, c)
        phi_r = c["phi_r"]
        angle = boundary_angles(n, phi_r, c["phi_s"])
        # The paths near an end of their edge that fades (their shares are below).
        t_k, length = c["t_k"], edges.length[edge]
        fades = self.fades[edge]
        near, reach = near_fading_end(fades, t_k, length, r1, r2, sin_beta, self.k_min)
        # Edges in a row on one straight way that is not found: all but one count on the
        # lit side of its boundary (_in_line), the listener side mirrored in it there,
        # and that one carries the wave; for the edges of that wave it is there.
        in_line, carrier, on_ways, of_way = self._in_line(
            c, found[:, 0] < 0, angle[:, 0], images[:, 0], orders[:, 0]
        )
        there = found >= 0
        there[of_way, 0] = True
        was_lit = angle[in_line] > _ON_BOUNDARY
        if len(in_line):
            phi_r = phi_r.copy()
            mirror = 2.0 * angle[in_line, 0] * np.sign(phi_r[in_line] - c["phi_s"][in_line])
            phi_r[in_line] += mirror
            angle = boundary_angles(n, phi_r, c["phi_s"])
        self._own_paths(listener, c, found, angle[:, 0] < -_ON_BOUNDARY)
        lit = (angle > _ON_BOUNDARY) | ((np.abs(angle) <= _ON_BOUNDARY) & (found >= 0))
        # The incident wave's term only: a face along the way has its reflection on
        # its boundary too, on the side it has in the row that carries the wave.
        lit[in_line, 0] |= np.abs(angle[in_line, 0]) <= _ON_BOUNDARY
        moved = lit[in_line] & ~was_lit
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
        # Near an end that fades, the share of the path the edge gives (along_edge).
        fading, shares = np.flatnonzero(near), None
        if near.any():
            spread = np.sqrt(k / (np.pi * length_param[fading, None, None]))
            scale = sin_beta[fading, None, None] ** 2 * spread
            fade, inside = self._inside(c, fading, there)
            # Each of the coefficient's four terms: at an end where nothing completes it
            # (end_fades gives 1), the edge's own part of the term, by its boundary.
            eps = _term_angles(n[fading], phi_r[fading], c["phi_s"][fading])
            shares = along_edge(
                t_k[fading],
                length[fading],
                scale,
                fade[:, _TERM_WAVE],
                reach[fading],
                inside[:, _TERM_WAVE],
                self.closed[edge[fading]],
                np.repeat((fades[fading] == 1.0)[:, None, :], 4, axis=1),
                term_boundaries(n[fading], eps, kl[fading]),
            )
        # The cosine each reflection on the way meets its face at, where its phase is
        # that face's absorption's (EarlyReflections.cosines).
        leg_cosines = np.full((2, len(kind)), np.nan)
        # The rows that reflect off a ground on the way and each term's factor (below).
        on_ground = []
        for place, (faces, bounces, via) in enumerate(legs):
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
            leg_cosines[place, rows[~ground]] = cosine[~ground]
            if ground.any():
                # A ground's spherical wave, from the image of the source or the listener
                # to the edge. Near its boundary, though, a term continues its geometric
                # wave, which reflects off the ground as the whole path's wave does (its
                # length, and the grounds round the reflection between the points before
                # and after it): for the incident wave those on either side of the edge,
                # for a face's reflection, which reflects at the edge on its boundary,
                # the edge itself. Each term takes that factor by 1 - |F(X)| of its
                # boundary, all of it on the boundary, where the wave and the term then
                # join, and the edge's farther off.
                g = rows[ground]
                leg = np.where(via_source, r1[rows], r2[rows])[ground]
                edge_point = c["point"][g]
                source = np.broadcast_to(self.source, edge_point.shape)
                hearer = np.broadcast_to(listener, edge_point.shape)
                twice = (kind[g] == 3)[:, None]
                ahead = via_source[ground][:, None]
                # The point on the far side of the edge from this leg's reflection.
                beyond = np.where(
                    ahead,
                    np.where(twice, c["bounce2"][g], hearer),
                    np.where(twice, c["bounce"][g], source),
                )
                at = bounce[ground]
                own = (np.where(ahead, source, edge_point), at, np.where(ahead, edge_point, hearer))
                whole = (np.where(ahead, source, beyond), at, np.where(ahead, beyond, hearer))
                args = (model, face[ground], cosine[ground])
                from_edge = ground_factors(*args, leg, ends=own)
                faces_whole = ground_factors(*args, distance[g], ends=own)
                incident_whole = ground_factors(*args, distance[g], ends=whole)
                geometric = np.stack([incident_whole, faces_whole, faces_whole], axis=1)
                kept_near = 1.0 - _share(kl[g], angle[g])
                on_ground.append(
                    (g, kept_near * geometric + (1.0 - kept_near) * from_edge[:, None])
                )
        factors = self._face_factors(edge, sin_beta, c["phi_s"])
        waves = _waves(n, phi_r, c["phi_s"], kl, lit, factors, fading, shares)
        # The row that carries the wave of a way with edges in a row (_in_line): its
        # term on the dark side is half the wave, by the share its edge gives; the other
        # edges take theirs off all of the wave, so it carries the rest of it too:
        # 1 - share of the wave, which is twice the whole term on the boundary.
        held = np.flatnonzero(np.isin(fading, carrier))
        if len(held):
            row = fading[held]
            term = np.argmin(np.abs(eps[held, :2]), axis=1)
            rest = 1.0 - shares[held, term]  # (R, B, M)
            wave = -2.0 * n[row, None, None] * np.sqrt(2.0 * np.pi * kl[row])
            waves[row, 0] += rest * wave * np.exp(0.25j * np.pi)
        waves *= common[:, None]  # (C, 3, B, M): incident, off face 0, off face n
        for g, factor in on_ground:
            waves[g] *= factor

        # The waves' terms add up coherently, as one diffracted wave, except in the
        # transition zone round each wave's boundary: there its terms stand in for the
        # geometric wave they continue. In the bands where the paths add in energy a
        # term continues its geometric wave by c and is part of the diffracted wave by
        # s = sqrt(1 - c^2) (_continued, :mod:`~acoustic_spaces.engine.clusters`); in
        # the others all of it adds in pressure.
        # Near an end that fades, each wave's term counts whether its path is found or not:
        # it fades out past the end, and the end's own edge takes the path round it
        # (end_fades), on the dark side where the path round the end is found and on the
        # lit side where its reflection point leaves the face past the end, or lies on
        # another face in the face's plane.
        counted = (lit == (found >= 0)) | near[:, None]
        counted[in_line] |= moved
        energy = np.zeros((len(edge), self.wavenumber.shape[0]))
        pressure = (waves * counted[:, :, None, None]).sum(axis=1)
        energy[:, : self.coherent] = np.mean(np.abs(pressure[:, : self.coherent]) ** 2, axis=2)
        if clusters is not None:
            late = slice(self.coherent, None)
            carried = (waves * counted[:, :, None, None])[:, :, late]
            kept = self._continued(kl[:, late], angle, lit, late)
            share = np.sqrt(np.maximum(1.0 - kept**2, 0.0))
            # The geometric wave of each term: its specular path where that was found.
            points = images.copy()
            hit = found >= 0
            points[hit] = _image_sources(listener, paths)[found[hit]]
            rows, cols = np.nonzero(counted)
            # On a way with edges in a row (_in_line), a face along the way has its
            # reflection on its boundary too: one wave with the incident wave.
            along = np.zeros_like(counted)
            along[on_ways, 1:] = np.abs(angle[on_ways, 1:]) <= _ON_BOUNDARY
            wave = np.where(along[rows, cols], 0, cols)
            keys = clusters.geometric(points[rows, wave], orders[rows, wave])
            # A face the wave comes from behind has no geometric wave of its own.
            alone = ~front[rows, cols]
            keys[alone] = clusters.private(int(alone.sum()))
            clusters.add(start + rows, keys, kept[rows, cols] * carried[rows, cols], distance[rows])
            clusters.add(
                start + np.arange(len(edge)),
                clusters.diffracted(c["source_side"], c["listener_side"]),
                np.sum(share * carried, axis=1),
                distance,
            )

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

        def cosines(i: int) -> np.ndarray:
            return leg_cosines[: 2 if kind[i] == 3 else 1 if kind[i] else 0, i].copy()

        last = np.where((kind == 2)[:, None], c["bounce"], c["point"])
        last = np.where((kind == 3)[:, None], c["bounce2"], last)

        return {
            "c": c,
            "waves": waves,
            "counted": counted,
            "energy": energy,
            "spectrum": self._spectrum(pressure, distance),
            "distance": distance,
            "order": np.where(kind == 3, 2, np.where(kind > 0, 1, 0)).astype(np.int64),
            "edge": edge,
            "last": last,
            "corners": corners,
            "faces": faces,
            "cosines": cosines,
            "in_line": on_ways,
        }

    def _double(self, listener, max_delay_s: float, single: dict | None, clusters, start: int):
        """The direct sound over two edges in turn (see the module notes), as
        :meth:`_single` gives the paths over one edge, with its pieces as rows from
        ``start``."""
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
        # A path that touches the line of an edge within the tolerance of an end touches
        # the end, as a path over one edge does (_combos).
        t1 = np.where(np.abs(t1) <= tolerance, 0.0, t1)
        t1 = np.where(np.abs(t1 - edges.length[e1]) <= tolerance, edges.length[e1], t1)
        t2 = np.where(np.abs(t2) <= tolerance, 0.0, t2)
        t2 = np.where(np.abs(t2 - edges.length[e2]) <= tolerance, edges.length[e2], t2)
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
        # Whether the path touches each edge at an end of it that fades (end_fades), and
        # whether the edge stops hard there: an upright edge at a corner (_inside).
        at_end = []
        for t, e in ((t1[sel], e1), (t2[sel], e2)):
            at = np.column_stack([t == 0.0, t == edges.length[e]]) & (self.fades[e] > 0.0)
            hard = (at & np.isclose(self.fades[e], 0.5)).any(axis=1)
            hard &= np.abs(edges.direction[e, 2]) >= _UPRIGHT
            at_end.append((at.any(axis=1) & ~hard, hard))
        r0, r1, r2 = r0[sel], r1[sel], r2[sel]
        phi_21, rho_21, phi_12, rho_12 = phi_21[sel], rho_21[sel], phi_12[sel], rho_12[sel]
        n1, n2 = edges.wedge[e1], edges.wedge[e2]
        phi_s1, phi_r2 = phi_from[e1].copy(), phi_to[e2].copy()
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
        # The rays that look for the path over one edge start just off it, so within
        # their reach of the incident wave's boundary at the other edge (two screens,
        # the source a fraction of a millimetre below the line of their tops) that path
        # can be there on the dark side, or missing on the lit one. The side is then the
        # one that agrees with it: the source (at the first edge) or the listener (at
        # the second) counts as mirrored in the boundary, that little way across.
        reach = 2.0 * offset
        for angle, rho_end, rho_edge, has, moved, fixed in (
            (angle1, rho_from[e1], rho_21, has1, phi_s1, phi_21),
            (angle2, rho_to[e2], rho_12, has2, phi_r2, phi_12),
        ):
            a = angle[:, 0]
            across = np.abs(a) * rho_end * rho_edge / (rho_end + rho_edge)
            wrong = (np.abs(a) > _ON_BOUNDARY) & (across <= reach)
            wrong &= (a > 0.0) != (has[:, 0] >= 0)
            moved[wrong] += 2.0 * a[wrong] * np.sign(moved[wrong] - fixed[wrong])
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
        # The wave along a face between the two edges (see _along_faces).
        loss = self._along_faces(factors1, factors2, e1, e2, leave, graze, r1)
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
        # On a boundary of each (the source, both edges and the listener in the plane of
        # the face they share) it is as just off the second's, the listener side moved
        # off it, as _in_front decides the side there.
        near1 = _share(kl1, angle1).min(axis=1)
        near2 = _share(kl2, angle2).min(axis=1)
        bend1 = np.maximum(bend1, (1.0 - near1) * (flat - 1.0 / along[:, None, None]))
        bend2 = np.maximum(bend2, (1.0 - near2) * (flat - 1.0 / (r1 + r2)[:, None, None]))
        part = np.where(near1 + near2 > 0.0, near2 / np.maximum(near1 + near2, 1e-300), 1.0)
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
        second = _waves(n2, phi_r2, phi_12, kl2, lit2, factors2)
        # A path over two edges stops at the ends of its edges: it is there on the edge
        # and gone past its end. At an end that fades (end_fades) it counts by half,
        # midway between the two, as a path over one edge does there: so the paths
        # over the two pieces of a bent edge, at their joint, are one path, not two.
        # At an end where an upright edge stops hard (_inside: its top at a screen's
        # corner, or its foot under a screen that hangs free) the edge takes nothing
        # off a wave that passes level with that end, as just past it: its incident
        # wave's term is left out, and the terms of its faces' reflections stay (just
        # past the end, the path over that edge alone has them, fading out; level with
        # the end that path may not be found).
        for waves, (half, hard) in ((first, at_end[0]), (second, at_end[1])):
            waves[half] *= 0.5
            waves[hard, 0] = 0.0
        # Along a face, the incident wave and its reflection off the face are one wave.
        _merge(first, leave)
        # At grazing incidence they are one wave too, which already holds the face's
        # reflection: half the coefficient.
        _merge(second, graze)
        # With the other edge in the plane of a face on the far side of this edge from
        # the face (two flat roofs of one height, a slot in a top), the term of the
        # face's reflection takes off the incident wave's what it would add, at every
        # angle: one wave too, so that the two cancel in the bands that add in energy as
        # they do in pressure.
        _merge(first, _beyond_faces(phi_21, rho_21, n1, tolerance))
        _merge(second, _beyond_faces(phi_12, rho_12, n2, tolerance))
        # Half the coefficient once, the same both ways round: also where the path
        # leaves the first edge along a face that ends before the second (a roof's
        # corner, then a screen level with the roof), and the face's reflection stays
        # behind where the face ends.
        on_face = leave.any(axis=1) | graze.any(axis=1)
        second *= np.where(on_face, 0.5, 1.0)[:, None, None, None]

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
            * loss
        )  # (C, B, M)
        waves = first[:, :, None] * second[:, None, :] * common[:, None, None]  # (C, 3, 3, B, M)
        counted = (lit1 == (has1 >= 0))[:, :, None] & (lit2 == (has2 >= 0))[:, None, :]
        # Where the first edge's wave is lit, the path continues the path over the second
        # edge alone, and its waves at the second edge are one with that path's; and
        # likewise where the second edge's wave is lit. Where both are, both paths over
        # one edge take their terms off the wave they pass on (the source's, also where
        # something else hides the source from the listener), and this path gives back
        # what they take twice. Not with the source exactly in the plane of a face of
        # the first edge that the path leaves along, or the listener in that of a face
        # of the second that it arrives along (whether or not the face reaches the other
        # edge), the incident wave on its boundary there: the paths over one edge count
        # once then (_in_front), and this one is left out.
        # In the coherent bands all the waves the path carries add in pressure, and add
        # to the paths over one edge in the impulse response.
        exact = ((np.abs(angle1[:, 0]) <= _ON_BOUNDARY) & leave.any(axis=1)) | (
            (np.abs(angle2[:, 0]) <= _ON_BOUNDARY) & graze.any(axis=1)
        )
        carried = counted & ~(lit1[:, :, None] & lit2[:, None, :] & exact[:, None, None])
        pressure = (waves * carried[..., None, None]).sum(axis=(1, 2))
        energy = np.zeros((len(e1), self.wavenumber.shape[0]))
        energy[:, : self.coherent] = np.mean(np.abs(pressure[:, : self.coherent]) ** 2, axis=2)
        if clusters is not None:
            # In the others each term (i, j) continues the geometric wave i at the first
            # edge or is part of its diffracted wave, and likewise at the second: four
            # pieces, as one edge's two (:mod:`~acoustic_spaces.engine.clusters`).
            late = slice(self.coherent, None)
            t = (waves * carried[..., None, None])[..., late, :]
            c1 = self._continued(kl1[:, late], angle1, lit1, late)
            c2 = self._continued(kl2[:, late], angle2, lit2, late)
            s1 = np.sqrt(np.maximum(1.0 - c1**2, 0.0))
            s2 = np.sqrt(np.maximum(1.0 - c2**2, 0.0))
            # The source and its images in the faces of the first edge, the listener and
            # its images in those of the second, and whether the waves come from in
            # front of the faces they reflect off.
            plane1, plane2 = edges.plane[e1], edges.plane[e2]
            count = len(e1)
            src = [np.broadcast_to(self.source, (count, 3))]
            rcv = [np.broadcast_to(listener, (count, 3))]
            for face in (0, 1):
                src.append(model.mirror(src[0], plane1[:, face]))
                rcv.append(model.mirror(rcv[0], plane2[:, face]))
            yes = np.ones(count, dtype=bool)
            front1 = [yes, phi_s1 < np.pi, phi_s1 > (n1 - 1.0) * np.pi]
            front2 = [yes, phi_12 < np.pi, phi_12 > (n2 - 1.0) * np.pi]
            rows = start + np.arange(count)

            def keyed(keys, ok):
                keys[~ok] = clusters.private(int((~ok).sum()))
                return keys

            for i in range(3):
                for j in range(3):
                    image = src[i] if j == 0 else model.mirror(src[i], plane2[:, j - 1])
                    order = int(i > 0) + int(j > 0)
                    clusters.add(
                        rows,
                        keyed(clusters.geometric(image, order), front1[i] & front2[j]),
                        c1[:, i] * c2[:, j] * t[:, i, j],
                        distance,
                    )
            # Bent at the first edge and on past the second (j), or on past the first and
            # bent at the second (i): where the two are one wave (both from the source to
            # the listener), a term is in it once, by the energy of both its pieces.
            bent1 = [keyed(clusters.diffracted(src[0], rcv[j]), front2[j]) for j in range(3)]
            bent2 = [keyed(clusters.diffracted(src[i], rcv[0]), front1[i]) for i in range(3)]
            first_part = s1[:, :, None] * c2[:, None, :]  # (C, 3, 3, B, M)
            second_part = c1[:, :, None] * s2[:, None, :]
            for i in range(3):
                for j in range(3):
                    same = np.flatnonzero(bent1[j] == bent2[i])
                    if len(same):
                        both = first_part[same, i, j] ** 2 + second_part[same, i, j] ** 2
                        first_part[same, i, j] = np.sqrt(both)
                        second_part[same, i, j] = 0.0
            for j in range(3):
                clusters.add(
                    rows, bent1[j], np.sum(first_part[:, :, j] * t[:, :, j], axis=1), distance
                )
            for i in range(3):
                clusters.add(rows, bent2[i], np.sum(second_part[:, i] * t[:, i], axis=1), distance)
            clusters.add(
                rows,
                clusters.private(count),
                np.einsum("cibm,cjbm,cijbm->cbm", s1, s2, t),
                distance,
            )
        return {
            "energy": energy,
            "spectrum": self._spectrum(pressure, distance),
            "distance": distance,
            "order": np.zeros(len(e1), dtype=np.int64),
            "edge": e1,
            "last": p2,
            "corners": lambda i: [p1[i], p2[i]],
            "faces": lambda i: np.zeros(0, dtype=np.int64),
            "cosines": lambda i: np.zeros(0),
        }

    def _along_faces(self, factors1, factors2, e1, e2, leave, graze, distance) -> np.ndarray:
        """(C, B, M) the boundary loss F(w) of the paths over two edges that run along a
        face between them (``leave`` at the first edge ``e1``, ``graze`` at the second
        ``e2``, (C, 2)), 1 for the others; and the factors ``factors1`` and ``factors2``
        (C, 2, B, M) of the reflections off those faces set in place to the specular
        share of their scattering alone, as on a rigid face. The first edge's wave runs
        to the second along the face at grazing incidence, where over a locally reacting
        face it is the spherical wave's (Weyl-van der Pol), 2 F(w) = 1 + Q times the
        free one over the way ``distance`` (C,) along it
        (:func:`~acoustic_spaces.engine.coherent.ground_wave_factors`), and over a rigid one
        twice the free one; not the plane wave's, whose R = -1 at grazing incidence
        would cancel it. So the edges' coefficients take the face as rigid, and the
        whole path F(w), once for a plane both edges' faces lie in, in every band. The
        way is the whole way from one edge to the other, also where the face covers
        only part of it (it ends before the other edge, which merely lies in its plane)."""
        edges, model = self.edges, self.model
        loss = np.ones((len(e1), *self.wavenumber.shape), dtype=complex)

        def plane_normal(edge, faces):
            plane = np.where(faces[:, 0], edges.plane[edge, 0], edges.plane[edge, 1])
            return np.where(faces.any(axis=1)[:, None], model.plane_normal[plane], 0.0)

        same = np.abs(np.einsum("ij,ij->i", plane_normal(e1, leave), plane_normal(e2, graze)))
        once = graze & (same < 1.0 - 1e-9)[:, None]  # not the plane the first edge leaves on
        for factors, edge, faces, lose in (
            (factors1, e1, leave, leave),
            (factors2, e2, graze, once),
        ):
            rows, side = np.nonzero(faces)
            face = edges.face[edge[rows], side]
            factors[rows, side] = np.sqrt(1.0 - model.scattering[face])[..., None]
            rows, side = np.nonzero(lose)
            if len(rows):
                face = edges.face[edge[rows], side]
                loss[rows] *= 0.5 * (1.0 + ground_wave_factors(model, face, distance[rows]))
        return loss

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
        # Not the paths past the ends of their edges, which only fade out there.
        known = np.where((c["t_k"] < 0.0) | (c["t_k"] > self.edges.length[c["edge"]]), -1, known)
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

    def _emit(self, listener, found: dict, floor: float, max_delay_s: float, rows=None):
        """The paths in ``found`` (from :meth:`_single` or :meth:`_double`) that carry
        at least ``floor`` in some band (or those of ``rows``, whatever they carry) and
        arrive by ``max_delay_s``, and their rows there."""
        energy, distance = found["energy"], found["distance"]
        bands = self.model.absorption.shape[1]
        keep = distance / self.speed_of_sound <= max_delay_s
        if rows is None:
            keep &= energy.max(axis=1) >= floor
        else:
            keep &= np.isin(np.arange(len(distance)), rows)
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            return EarlyReflections.empty(bands), idx
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
            cosines=[] if spectrum is None else [found["cosines"](i) for i in idx],
        ), idx

    def _inside(self, c: dict, rows: np.ndarray, there: np.ndarray):
        """For paths ``rows`` of ``c``, (R, 3, 2) how far each wave (the incident wave and
        the reflections off face 0 and face n) fades at the edge's start and its end and
        whether it lies inside them, for the hard part of the fade (:func:`along_edge`);
        ``there`` (C, 3) is whether each wave's specular path is found
        (:meth:`_specular`), or counts as found (:meth:`_in_line`).

        At a right-angled corner, where the wave's line (from the source side, mirrored in
        the face for a reflection, to the listener side) passes through the plane of the
        face the two edges share, the face cuts the wave off at both of them: the wave
        lies inside where it passes on this edge's side of the other edge, whose own term
        round the end then appears or goes, and the two halves of their fades add up as
        the edges of a quarter plane do in the Kirchhoff approximation. Where the line
        meets the other edge's line, on that edge's boundary, its term takes the side the
        wave's path decides (found or not, as in :meth:`_single`), and so does this: the
        face cuts the incident wave off where its path is not found, and reflects a
        reflection where its path is. Where the line does not pass through the face's
        plane, nothing cuts the wave off there and its term fades all the way.
        Elsewhere the wave lies inside where the point the path touches lies on the
        edge."""
        model, edges = self.model, self.edges
        edge = c["edge"][rows]
        t_k, length = c["t_k"][rows], edges.length[edge]
        fade = np.repeat(self.fades[edge][:, None, :], 3, axis=1)
        inside = np.empty((len(rows), 3, 2), dtype=bool)
        inside[:, :, 0] = (t_k >= 0.0)[:, None]
        inside[:, :, 1] = (t_k <= length)[:, None]
        source, listener = c["source_side"][rows], c["listener_side"][rows]
        seen = np.where(c["phi_s"][rows] < np.pi, 0, 1)  # the face the source side sees
        for end in (0, 1):
            corner = np.isclose(self.fades[edge, end], 0.5)
            if not corner.any():
                continue
            point = edges.start[edge] if end == 0 else edges.end[edge]
            into = edges.direction[edge] if end == 0 else -edges.direction[edge]
            for wave, side in ((0, seen), (1, np.zeros_like(seen)), (2, np.ones_like(seen))):
                other = self.corners[edge, end, side]
                use = np.flatnonzero(corner & (other >= 0))
                if not len(use):
                    continue
                plane = edges.plane[edge[use], side[use]]
                start = source[use] if wave == 0 else model.mirror(source[use], plane)
                normal, offset = model.plane_normal[plane], model.plane_offset[plane]
                line = listener[use] - start
                across = np.einsum("ij,ij->i", normal, line)
                ahead = offset - np.einsum("ij,ij->i", normal, start)
                with np.errstate(divide="ignore", invalid="ignore"):
                    at = ahead / across
                through = (np.abs(across) > 0.0) & (at > 0.0) & (at < 1.0)
                fade[use[~through], wave, end] = 1.0
                use, at, start = use[through], at[through], start[through]
                # Where a vertical edge meets one that is not, the one that is not fades
                # all the way and the vertical one stops hard: the two still add up, and
                # each stays continuous where the other's term appears, as paths over a
                # barrier's top and paths round its end take different weather.
                upright = np.abs(edges.direction[edge[use], 2]) >= _UPRIGHT
                beside = np.abs(edges.direction[other[use], 2]) >= _UPRIGHT
                fade[use[beside & ~upright], wave, end] = 1.0
                fade[use[upright & ~beside], wave, end] = 0.0
                crossing = start + at[:, None] * line[through]
                # Across the other edge's line, in the plane, towards this edge.
                along = edges.direction[other[use]]
                towards = into[use] - np.einsum("ij,ij->i", into[use], along)[:, None] * along
                side_of = np.einsum("ij,ij->i", crossing - point[use], towards)
                # On the other edge's boundary, as that edge has it.
                phi_from = edges.frame(other[use], start)[2]
                phi_to = edges.frame(other[use], listener[use])[2]
                angle = boundary_angles(edges.wedge[other[use]], phi_to, phi_from)[:, 0]
                hit = there[rows[use], wave]
                inside[use, wave, end] = np.where(
                    np.abs(angle) <= _ON_BOUNDARY, hit if wave else ~hit, side_of > 0.0
                )
        return fade, inside

    def _specular(self, listener: np.ndarray, paths: EarlyReflections, c: dict):
        """(C, 3) index in ``paths`` of the specular path of each term's wave (the
        incident wave and the reflections off face 0 and face n), or -1; (C, 3, 3) the
        image of the source each wave comes from and (C, 3) its order; and (C, 3)
        whether the wave comes from in front of its face (always for the incident
        wave)."""
        model, edges = self.model, self.edges
        count = len(c["edge"])
        out = np.full((count, 3), -1, dtype=np.int64)
        points = np.zeros((count, 3, 3))
        reflections = np.zeros((count, 3), dtype=np.int64)
        # A path through a wall is not the geometric wave a term continues.
        specular = np.flatnonzero((paths.edge < 0) & (paths.crossings == 0))
        images = _image_sources(listener, paths)[specular]
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
            points[:, column], reflections[:, column] = image, order
            if len(specular):
                out[:, column] = _match(image, order, images, orders, specular, tolerance)
        # A reflection off a face needs the wave to come from in front of it: on a free
        # edge both faces' images are the same point, and only one side is in front.
        n = edges.wedge[c["edge"]]
        front = np.column_stack(
            [np.ones(count, dtype=bool), c["phi_s"] < np.pi, c["phi_s"] > (n - 1.0) * np.pi]
        )
        out = np.where(front, out, -1)
        return out, points, reflections, front

    def _own_paths(self, listener, c: dict, found, dark) -> None:
        """Set to -1, in ``found`` (C, 3, from :meth:`_specular`), the specular path of
        the incident wave of each term on the ``dark`` side of its boundary (C,) where it
        is not the term's geometric wave, as the edge fade decides it. A reflection off a
        plane before the edge (kind 1) and one after it (kind 2) come from the same image
        of the source, but the path is the dark term's only where the line through it
        (from that image to the listener, or from the source to the listener's image)
        comes nearest to the edge on the term's side of the plane: past it (kind 1) or
        before it (kind 2). Behind a barrier on a rigid ground the term of the path over
        the top with the reflection off the ground after it was left out where the
        reflection before it was found."""
        model, edges, kind = self.model, self.edges, c["kind"]
        rows = np.flatnonzero(dark & ((kind == 1) | (kind == 2)) & (found[:, 0] >= 0))
        # An edge along the plane's normal is its own image in it: there the term's point
        # lies on the edge, on its side of the plane.
        normal = model.plane_normal[c["plane"][rows]]
        tilted = np.abs(np.einsum("ij,ij->i", normal, edges.direction[c["edge"][rows]]))
        rows = rows[tilted <= 1.0 - 1e-9]
        if not len(rows):
            return
        edge, plane = c["edge"][rows], c["plane"][rows]
        normal, offset = model.plane_normal[plane], model.plane_offset[plane]
        before = kind[rows] == 2
        start = np.broadcast_to(self.source, (len(rows), 3)).copy()
        end = np.broadcast_to(listener, (len(rows), 3)).copy()
        start[~before] = model.mirror(start[~before], plane[~before])
        end[before] = model.mirror(end[before], plane[before])
        closest = _closest(start, end, edges, edge)
        # The side of the plane away from the image across it; on the plane itself, the
        # leg after it.
        image = np.where(before[:, None], end, start)
        away = np.sign(offset - np.einsum("ij,ij->i", normal, image))
        at = (np.einsum("ij,ij->i", normal, closest) - offset) * away
        mine = np.where(before, at > 0.0, at >= 0.0)
        found[rows[~mine], 0] = -1


def _past_face(model: TraceModel, edges: Edges, edge, phi, phi_source, hearer, tolerance):
    """(C,) whether the listener side of each path over ``edge`` (C,), at ``phi`` (C,)
    round it from face 0, lies exactly in the plane of face 0 or face n seen along the
    edge (phi 0 or n pi) with the source side, at ``phi_source``, in the plane of that
    face too, and the path's listener ``hearer`` (C, 3) is not on that face of the edge
    (:func:`_on_own_face`, within ``tolerance``) but beyond it. The straight way between
    the source and the listener sides then runs in the face's plane past the face;
    where its path is not found the listener side lies behind the face
    (:meth:`EdgeDiffraction._in_front`). A listener on the face (its edges included)
    has only the side in front of it. Another face in that plane, such as the wall of
    a second building in line with the first, lies beyond the edge's face as the open
    ground does. The listener itself is tested, not its image where a reflection
    follows the edge: mirrored with the face in a plane across it (the ground under a
    wall), it lies on the face's image exactly where it lies on the face."""
    pi, on, n = np.pi, _ON_BOUNDARY, edges.wedge[edge]
    first = (phi <= on) & ((phi_source <= on) | (np.abs(phi_source - pi) <= on))
    last = (phi >= n * pi - on) & (
        (phi_source >= n * pi - on) | (np.abs(phi_source - (n - 1.0) * pi) <= on)
    )
    out = first | last
    for row in np.flatnonzero(out):
        side = 0 if first[row] else 1
        out[row] = not _on_own_face(model, edges, int(edge[row]), side, hearer[row], tolerance)
    return out


def _on_own_face(
    model: TraceModel, edges: Edges, edge: int, side: int, point: np.ndarray, tolerance: float
) -> bool:
    """Whether ``point`` lies on face 0 (``side`` 0) or face n (1) of ``edge``, its edges
    included, within ``tolerance``: in the face's plane, on the solid faces there that
    the face is joined to through shared corners (the triangles of a wall or a roof),
    not on another face in that plane (the wall of another building in line with it)."""
    face = int(edges.face[edge, side])
    if face < 0:
        face = int(edges.face[edge, 0])
    plane = int(model.face_plane[face])
    gap = float(model.plane_normal[plane] @ point) - float(model.plane_offset[plane])
    if abs(gap) > tolerance:
        return False
    faces = np.asarray(model.room.mesh.faces)
    candidates = np.flatnonzero((model.face_plane == plane) & ~model.transparent)
    if face not in candidates:
        candidates = np.append(candidates, face)
    corners = faces[candidates].reshape(-1)
    triangle = np.repeat(np.arange(len(candidates)), faces.shape[1])
    order = np.argsort(corners, kind="stable")
    shared = corners[order][1:] == corners[order][:-1]
    joined = _components(len(candidates), triangle[order][:-1][shared], triangle[order][1:][shared])
    own = candidates[joined == joined[np.flatnonzero(candidates == face)[0]]]
    vertices = np.asarray(model.room.mesh.vertices, dtype=float)
    return _on_triangles(np.asarray(point, dtype=float), vertices[faces[own]], tolerance)


def _subset(part: dict, keep: np.ndarray) -> dict:
    """The rows ``keep`` of each array in ``part``."""
    return {key: value[keep] for key, value in part.items()}


def _closest(start, end, edges, edge) -> np.ndarray:
    """(n, 3) the points of the lines from ``start`` to ``end`` (n, 3) nearest to the
    lines of ``edges`` ``edge`` (n,); ``start`` where they are parallel."""
    line = end - start
    u = edges.direction[edge]
    w = start - edges.start[edge]
    along = np.einsum("ij,ij->i", line, u)
    square = np.einsum("ij,ij->i", line, line)
    across = square - along**2  # |line|^2 sin^2 of the angle between them
    usable = across > 1e-12 * np.maximum(square, 1e-300)
    s = (along * np.einsum("ij,ij->i", u, w) - np.einsum("ij,ij->i", line, w)) / np.where(
        usable, across, 1.0
    )
    return start + np.where(usable, s, 0.0)[:, None] * line


def _waves(n, phi, phi_source, kl, lit, reflection, rows=None, shares=None) -> np.ndarray:
    """(C, 3, B, M) the terms of the incident wave and of the reflections off face 0 and
    face n of C edges, for wave numbers times distance parameters ``kl`` (C, B, M); their
    faces reflect by ``reflection`` (C, 2, B, M). The edges ``rows`` give each of their
    four terms by ``shares`` (R, 4, B, M, :func:`along_edge`)."""
    terms = wedge_terms(
        n[:, None, None], phi[:, None, None], phi_source[:, None, None], kl, lit[:, None, None, :]
    )
    if shares is not None:
        terms[rows] *= np.moveaxis(shares, 1, -1)
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


def _beyond_faces(phi, rho, n, tolerance) -> np.ndarray:
    """(C, 2) whether points ``phi`` round edges open over ``n`` pi, ``rho`` from them,
    lie in the plane of face 0 or of face n on the far side of the edge from the face."""
    return np.column_stack(
        [
            np.abs(phi - np.pi) * rho <= tolerance,
            np.abs(phi - (n - 1.0) * np.pi) * rho <= tolerance,
        ]
    )


def _merge(waves: np.ndarray, faces: np.ndarray) -> None:
    """Add the reflection off face 0 or face n into the incident wave where ``faces``
    (C, 2) says the two are one wave, in place."""
    for face in (0, 1):
        rows = np.flatnonzero(faces[:, face])
        waves[rows, 0] += waves[rows, face + 1]
        waves[rows, face + 1] = 0.0


def _share(kl: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """(C, 3, B, M) |F(X)| of each wave's boundary, ``angle`` (C, 3) away."""
    x = 2.0 * kl[:, None] * np.sin(0.5 * angle)[..., None, None] ** 2
    return np.abs(transition(x))


def _continued(kl, angle, lit, wavenumber, slice_hz, speed_of_sound) -> np.ndarray:
    """(C, 3, B, M) how much of each wave's term continues the geometric wave of its
    boundary, ``angle`` (C, 3) away, on its ``lit`` side or not, in bands of wave
    numbers ``wavenumber`` (B, M) that stand for slices ``slice_hz`` (B,) wide, for those
    times the distance parameter ``kl`` (C, B, M); the rest of it, s = sqrt(1 - c^2), is
    part of the edge's diffracted wave. On the lit side, where the term adds to the wave,
    c is its coherence with it over the band: the magnitude of the band's mean of
    exp(-jkd) sinc(w d / speed_of_sound), with d = X / k how much farther the term goes
    than the wave (X the argument of its transition function, k d near the boundary) and
    w the slice. On the dark side, where it stands in for the wave, which is gone,
    s = |F(X)|, 0 on the boundary and 1 well off it. The same at each of the band's
    frequencies on the lit side."""
    x = 2.0 * kl[:, None] * np.sin(0.5 * angle)[..., None, None] ** 2
    fade = np.sinc(slice_hz[:, None] * (x / wavenumber) / speed_of_sound)
    coherence = np.abs(np.mean(np.exp(-1j * x) * fade, axis=-1, keepdims=True))
    dark = np.sqrt(np.maximum(1.0 - np.abs(transition(x)) ** 2, 0.0))
    return np.where(np.asarray(lit)[..., None, None], np.minimum(coherence, 1.0), dark)


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


def _reach(kind, length, t_s, rho_s, t_r, rho_r, speed_of_sound, max_delay_s, past=None):
    """Where along the edge a path from a source side at ``t_s``, ``rho_s`` to a listener
    side at ``t_r``, ``rho_r`` makes equal angles with it on both sides, the lengths on
    either side, and whether that point lies on the edge and the path ends by
    ``max_delay_s``. With ``past`` (the fades of the edges' ends, (n, 2), the lowest
    wave number, and (n, 2) whether each end lies in a plane the path reflects off), the
    point may also lie on an end that fades or past it, within the reach of its fade
    (:func:`_fade_reach`)."""
    t_k = t_s + (t_r - t_s) * rho_s / (rho_s + rho_r)
    r1 = np.hypot(rho_s, t_k - t_s)
    r2 = np.hypot(rho_r, t_r - t_k)
    # At an end of the edge that lies in the plane, the path reflecting there before the
    # edge and the one reflecting after it are the same path: only the first counts at
    # the start of the edge, only the second at its end.
    ok = np.where(kind == 2, (t_k > 0.0) & (t_k <= length), (t_k >= 0.0) & (t_k < length))
    ok &= (kind != 3) | ((t_k > 0.0) & (t_k < length))  # at an end, one of those two
    if past is not None:
        # An end that fades and lies in no such plane is part of its fade, like the
        # line on either side of it.
        fade, k_min, held = past
        reach = _fade_reach(r1, r2, rho_s / np.maximum(r1, 1e-300), k_min)
        at_start = (t_k < 0.0) | ((t_k == 0.0) & ~held[:, 0])
        at_end = (t_k > length) | ((t_k == length) & ~held[:, 1])
        ok |= (fade[:, 0] > 0.0) & at_start & (t_k >= -reach)
        ok |= (fade[:, 1] > 0.0) & at_end & (t_k <= length + reach)
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


def _image_sources(listener: np.ndarray, paths: EarlyReflections) -> np.ndarray:
    """(n, 3) the image of the source of each of the specular ``paths`` at ``listener``."""
    return listener + paths.direction * paths.distance_m[:, None]


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
