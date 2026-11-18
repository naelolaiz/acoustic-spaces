"""Early reflections: exact specular paths with their delay, direction and band energies.

Two ways to find the candidate image sources, then one exact check:

- ``ray_traced_candidates``: deterministic specular rays from the source; every sequence
  of planes a ray hits becomes a candidate. The cost grows with the number of rays, not
  with the number of surfaces, so this suits large models (the default).
- ``exhaustive_candidates``: every sequence of planes up to the maximum order, pruned to
  images in front of the next plane. Complete, but grows as planes ** order.

``find_paths`` then traces each candidate back from the listener: the path must hit the
expected planes in order and reach the source unobstructed. Paths found are exact
whichever way they were proposed. A path through the edge where two planes meet (a
corner of a box, with the listener and source placed symmetrically) is kept once. A
path that reflects exactly at the foot of a wall standing on the surface (or in a
corner, at the foot of a wall standing in it), or on the line where such walls meet,
and goes on behind them, crosses them there as it does just to one side of the point:
of the two sides, one that agrees with the walls the legs' own intersection tests meet
and where the surfaces it reflects off go on, then the one where it crosses fewer
(past a wall's free end, none; see :func:`_crossed_here`). No path reflects off two
coincident planes (a wall drawn as one face for each room) at one point, and openings
are holes: no path reflects off one.

A path may pass straight through surfaces that let sound through (partitions, doors,
openings): its energy is then multiplied by their transmission, band by band.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces import products
from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.coherent import (
    ground_factors,
    reflection_phases,
    sample_frequencies,
)
from acoustic_spaces.engine.model import (
    MAX_CROSSINGS,
    PLANE_TOLERANCE,
    TraceModel,
    source_weights,
)
from acoustic_spaces.engine.turbulence import Wander

#: Distance (metres) below which a point counts as lying on a plane.
_ON_PLANE = 1e-9
#: Tolerance, relative to the model size, within which a point lies in a plane (a mesh's
#: faces need not lie in theirs to rounding): for a path reflecting where planes meet.
_EDGE_TOLERANCE = 1e-7
#: A ray traced on the CPU meets no face nearer its start than this share of the diagonal
#: of the model's box (``Bvh::min_distance``).
_RAY_START = 1e-9
#: Upper bound on candidates for the exhaustive search.
MAX_CANDIDATES = 2_000_000


def ray_start(mesh) -> float:
    """Distance (metres) from its start within which a ray traced on the CPU meets no
    face of ``mesh`` (:data:`_RAY_START` of the diagonal of its box,
    ``TraceModel.bvh.min_distance``): the path traced back from a listener misses a
    reflection whose last leg is shorter, and the rays of a source nearer a surface go
    through it."""
    return _RAY_START * math.hypot(*np.ptp(mesh.vertices, axis=0).tolist())


@dataclass
class EarlyReflections:
    """Paths from one source to one listener: specular (order 0 is the direct sound),
    and diffracted at an edge when edge diffraction is on."""

    order: np.ndarray  # (n,) reflections along the path
    delay_s: np.ndarray  # (n,)
    distance_m: np.ndarray  # (n,)
    direction: np.ndarray  # (n, 3) unit vectors from the listener towards the arrival
    energy: np.ndarray  # (n, bands) time-integrated intensity, unit-energy source
    points: list[np.ndarray] = field(default_factory=list)  # every corner, source first
    faces: list[np.ndarray] = field(default_factory=list)  # faces hit, source side first
    #: (n,) the edge a path bends over (an index of the model's diffracting edges, see
    #: :func:`~acoustic_spaces.engine.diffraction.find_edges`), -1 for specular paths,
    #: -2 for a path over the crests of the terrain
    #: (:mod:`~acoustic_spaces.engine.crests`), or -3 for the path that carries the
    #: sound of the solver for whole rows (:mod:`~acoustic_spaces.engine.row_paths`).
    edge: np.ndarray | None = None
    #: (n,) how many surfaces that let sound through a path passes, openings aside.
    crossings: np.ndarray | None = None
    #: (n, C, M) in the lowest C bands, where the paths add in pressure (see
    #: :mod:`~acoustic_spaces.engine.coherent`): each path's pressure at M frequencies
    #: in each band (:func:`~acoustic_spaces.engine.coherent.sample_frequencies`),
    #: relative to its delay and to its level in the band, which its energy gives (the
    #: mean of its squared magnitude over a band's frequencies is 1). None when the
    #: paths add in energy in every band.
    spectrum: np.ndarray | None = None
    #: (n,) which of the uncorrelated points of a line or surface source each path comes
    #: from (outdoors, where those paths carry a spectrum): in the coherent bands the
    #: paths of one point add in pressure, those of different points in energy.
    point: np.ndarray | None = None
    #: Outdoors in turbulence, how the eddies delay the paths at random
    #: (:class:`~acoustic_spaces.engine.turbulence.Wander`): on average two paths'
    #: pressures add with their coherence factor. None without turbulence, and at a
    #: moment of it (:meth:`frozen`).
    wander: Wander | None = None
    #: (n, bands) at a moment of turbulence, each path's delay (s) in each band beyond
    #: ``delay_s``, its delay in the highest band (the eddies part paths near the
    #: ground less at low frequencies); None otherwise.
    lag: np.ndarray | None = None
    #: Where the paths carry a :attr:`spectrum`: for each path, the cosine of the angle
    #: at which it meets the face of each of its reflections (as :attr:`faces`), whose
    #: phase its spectrum takes from that face's absorption
    #: (:func:`~acoustic_spaces.engine.coherent.reflection_phases`), so that re-weighting
    #: for other absorption can turn it (:func:`~acoustic_spaces.engine.reweight.
    #: turned_paths`); NaN for a reflection whose phase comes from elsewhere (a ground's
    #: impedance). Empty without a spectrum.
    cosines: list[np.ndarray] = field(default_factory=list)
    #: (n,) in the bands above the coherent ones, the phase factor a rendered response
    #: gives each path beyond the one held from those bands
    #: (:func:`~acoustic_spaces.engine.coherent.with_own_phases`); None outside one.
    own_phase: np.ndarray | None = None
    #: (n, C, K) where a path carries the field the parabolic equation gives over the open
    #: ground (outdoors in wind or temperature gradients, out of the shadow): in each
    #: coherent band its pressure, unnormalised, relative to its delay, at the frequencies
    #: of the layout (:func:`~acoustic_spaces.engine.fine.fine_frequencies`) the equation
    #: ran at there (in the lowest band down to an octave below it, in its top band
    #: through the upper crossover too where a response is rendered or the band filters
    #: read it); NaN where a path has none. Its band's shape is drawn from these and
    #: its neighbours' (:func:`~acoustic_spaces.engine.coherent.shaped`); its level stays
    #: the one its energy gives. None when no path carries one.
    fine: np.ndarray | None = None
    #: (n,) the delay (s) past :attr:`delay_s` whose phase :attr:`fine`'s samples follow
    #: on average (:func:`~acoustic_spaces.engine.fine.reference_delay`), relative to
    #: which they are interpolated; 0 for paths without them. None with :attr:`fine`.
    fine_delay: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.edge is None:
            self.edge = np.full(len(self.order), -1, dtype=np.int64)
        if self.crossings is None:
            self.crossings = np.zeros(len(self.order), dtype=np.int64)
        if self.point is None:
            self.point = np.zeros(len(self.order), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, index) -> EarlyReflections:
        """The paths ``index`` picks (a slice, indices or a boolean mask)."""
        rows = np.arange(len(self))[index]
        return EarlyReflections(
            order=self.order[rows],
            delay_s=self.delay_s[rows],
            distance_m=self.distance_m[rows],
            direction=self.direction[rows],
            energy=self.energy[rows],
            points=[self.points[i] for i in rows] if self.points else [],
            faces=[self.faces[i] for i in rows] if self.faces else [],
            edge=self.edge[rows],
            crossings=self.crossings[rows],
            spectrum=None if self.spectrum is None else self.spectrum[rows],
            point=self.point[rows],
            wander=None if self.wander is None else self.wander[rows],
            lag=None if self.lag is None else self.lag[rows],
            cosines=[self.cosines[i] for i in rows] if self.cosines else [],
            own_phase=None if self.own_phase is None else self.own_phase[rows],
            fine=None if self.fine is None else self.fine[rows],
            fine_delay=None if self.fine_delay is None else self.fine_delay[rows],
        )

    @property
    def coherent_bands(self) -> int:
        """How many of the lowest bands the paths add in pressure in."""
        return 0 if self.spectrum is None else self.spectrum.shape[1]

    def frozen(self, seed: int) -> EarlyReflections:
        """The paths at a moment of the turbulent air drawn with ``seed``: each
        delayed in each band as the eddies have it then, adding in pressure (the same
        paths without turbulence)."""
        if self.wander is None:
            return self
        roots = self.wander_roots()
        return self.at_moment(roots, _draws(roots, seed))

    def wander_roots(self) -> list:
        """How the eddies delay the paths, band by band, as
        :meth:`~acoustic_spaces.engine.turbulence.Wander.roots` (empty without
        turbulence)."""
        if self.wander is None:
            return []
        from acoustic_spaces.materials.bands import band_centres, resolution_of

        centres = band_centres(resolution_of(self.energy.shape[1]))
        return self.wander.roots(centres, self.point)

    def at_moment(self, roots: list, draws: list[np.ndarray]) -> EarlyReflections:
        """The paths delayed as :meth:`~acoustic_spaces.engine.turbulence.Wander.delays`
        has it for the normal variables ``draws`` (one array per group of ``roots``,
        :meth:`wander_roots`)."""
        if self.wander is None:
            return self
        # No path arrives before it leaves the source, however the eddies draw.
        delays = np.maximum(self.delay_s[:, None] + Wander.delays(roots, draws, len(self)), 0.0)
        return dataclasses.replace(
            self,
            delay_s=delays[:, -1],
            wander=None,
            lag=delays - delays[:, -1:],
        )

    @property
    def diffracted(self) -> np.ndarray:
        """(n,) True for paths that bend over an edge or the terrain's crests, and for
        the one that carries the sound of the solver for whole rows."""
        return self.edge != -1

    @classmethod
    def empty(cls, bands: int) -> EarlyReflections:
        return cls(
            order=np.zeros(0, dtype=np.int64),
            delay_s=np.zeros(0),
            distance_m=np.zeros(0),
            direction=np.zeros((0, 3)),
            energy=np.zeros((0, bands)),
        )

    @classmethod
    def concatenate(cls, parts: list[EarlyReflections], bands: int) -> EarlyReflections:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty(bands)
        shaped = [p.spectrum for p in parts if p.spectrum is not None]
        spectrum = None
        if shaped:
            # Paths without a spectrum have no phase of their own.
            spectrum = np.concatenate(
                [
                    np.ones((len(p), *shaped[0].shape[1:]), dtype=complex)
                    if p.spectrum is None
                    else p.spectrum
                    for p in parts
                ]
            )
        out = cls(
            order=np.concatenate([p.order for p in parts]),
            delay_s=np.concatenate([p.delay_s for p in parts]),
            distance_m=np.concatenate([p.distance_m for p in parts]),
            direction=np.concatenate([p.direction for p in parts]),
            energy=np.concatenate([p.energy for p in parts]),
            points=[x for p in parts for x in p.points],
            faces=[x for p in parts for x in p.faces],
            edge=np.concatenate([p.edge for p in parts]),
            crossings=np.concatenate([p.crossings for p in parts]),
            spectrum=spectrum,
            point=np.concatenate([p.point for p in parts]),
            wander=Wander.stacked([p.wander for p in parts], [len(p) for p in parts]),
            lag=_lags(parts, bands),
            cosines=_cosines(parts),
            fine=_fines(parts),
            fine_delay=_fine_delays(parts),
        )
        return out.sorted()

    def sorted(self) -> EarlyReflections:
        """The same paths in order of arrival."""
        return self[np.argsort(self.delay_s, kind="stable")]


def _cosines(parts: list[EarlyReflections]) -> list[np.ndarray]:
    """The parts' :attr:`~EarlyReflections.cosines` one after the other (NaN for the
    reflections of paths without them); empty when no part has them."""
    if all(not p.cosines for p in parts):
        return []
    out = []
    for p in parts:
        if p.cosines:
            out.extend(p.cosines)
        elif p.faces:
            out.extend(np.full(len(f), np.nan) for f in p.faces)
        else:
            out.extend(np.zeros(0) for _ in range(len(p)))
    return out


def _fines(parts: list[EarlyReflections]) -> np.ndarray | None:
    """The parts' :attr:`~EarlyReflections.fine` one above the other (NaN for paths
    without one); None when no part has one."""
    fines = [p.fine for p in parts if p.fine is not None]
    if not fines:
        return None
    shape = fines[0].shape[1:]
    return np.concatenate(
        [
            np.full((len(p), *shape), np.nan, dtype=complex) if p.fine is None else p.fine
            for p in parts
        ]
    )


def _fine_delays(parts: list[EarlyReflections]) -> np.ndarray | None:
    """The parts' :attr:`~EarlyReflections.fine_delay` one after the other (zero for
    paths without one); None when no part has one."""
    if all(p.fine_delay is None for p in parts):
        return None
    return np.concatenate(
        [np.zeros(len(p)) if p.fine_delay is None else p.fine_delay for p in parts]
    )


def _lags(parts: list[EarlyReflections], bands: int) -> np.ndarray | None:
    """The parts' :attr:`~EarlyReflections.lag` one above the other (zero for paths
    without one); None when no part has one."""
    if all(p.lag is None for p in parts):
        return None
    return np.concatenate([np.zeros((len(p), bands)) if p.lag is None else p.lag for p in parts])


def _draws(roots: list, seed: int) -> list[np.ndarray]:
    """Normal variables for each group of ``roots`` (see
    :meth:`EarlyReflections.at_moment`), drawn with ``seed``."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(len(rows)) for rows, _ in roots]


def ray_traced_candidates(
    model: TraceModel, source: np.ndarray, max_order: int, rays: int, threads: int = 0
) -> list[np.ndarray]:
    """Plane sequences hit by ``rays`` specular rays, which pass through surfaces that
    let sound through (forking where those also reflect); entry k-1 has shape (n, k)."""
    from acoustic_spaces import _native

    if max_order <= 0:
        return []
    passing = np.zeros(0, dtype=np.uint8)
    if model.transmission is not None:
        # 0: reflects, 1: reflects and lets sound through, 2: lets it through only.
        through = np.any(model.transmission > 0.0, axis=1)
        passing = np.where(model.partition, 1, np.where(through, 2, 0)).astype(np.uint8)
    faces = _native.specular_sequences(
        model.bvh,
        np.asarray(source, dtype=np.float64),
        int(rays),
        int(max_order),
        int(threads),
        passing,
    )
    planes = np.where(faces >= 0, model.face_plane[np.maximum(faces, 0)], -1)
    out = []
    for k in range(1, max_order + 1):
        prefix = planes[:, :k]
        prefix = prefix[np.all(prefix >= 0, axis=1)]
        out.append(
            _distinct_rows(prefix, model.plane_count) if len(prefix) else np.zeros((0, k), np.int64)
        )
    return out


def _distinct_rows(rows: np.ndarray, base: int) -> np.ndarray:
    """The distinct rows of ``rows`` (n, k), whole numbers from 0 to ``base`` - 1, in
    ascending order: what ``np.unique(rows, axis=0)`` gives, through one number per row
    (each row read as a number in base ``base``) where those fit in 64 bits."""
    k = rows.shape[1]
    if base**k > np.iinfo(np.int64).max:
        return np.unique(rows, axis=0)
    key = rows[:, 0].astype(np.int64)
    for column in range(1, k):
        key = key * base + rows[:, column]
    key = np.unique(key)
    out = np.empty((len(key), k), dtype=rows.dtype)
    for column in range(k - 1, -1, -1):
        key, out[:, column] = np.divmod(key, base)
    return out


def exhaustive_candidates(
    model: TraceModel, source: np.ndarray, max_order: int, limit: int = MAX_CANDIDATES
) -> list[np.ndarray]:
    """Every plane sequence whose images stay in front of the next plane (outdoors, not
    the planes of the open sky, which reflects nothing)."""
    normals, offsets = model.plane_normal, model.plane_offset
    sky = None
    if model.sky is not None:
        sky = np.ones(model.plane_count, dtype=bool)
        np.logical_and.at(sky, model.face_plane, model.sky)
    seqs = np.zeros((1, 0), dtype=np.int64)
    images = np.asarray(source, dtype=np.float64)[None, :]
    out = []
    total = 0
    for _ in range(max_order):
        # Starting from empty arrays keeps one (0, k) entry for an order with no sequence
        # left (a single reflecting plane leaves none from the second order on).
        new_seqs = [np.zeros((0, seqs.shape[1] + 1), dtype=np.int64)]
        new_images = [np.zeros((0, 3))]
        chunk = max(1, 4_000_000 // max(1, model.plane_count))
        for start in range(0, len(seqs), chunk):
            s, img = seqs[start : start + chunk], images[start : start + chunk]
            distance = products.product(img, normals.T) - offsets  # (n, planes)
            ok = model.in_front(distance, _ON_PLANE)
            if s.shape[1]:
                ok[np.arange(len(s)), s[:, -1]] = False
            if sky is not None:
                ok[:, sky] = False
            rows, cols = np.nonzero(ok)
            new_seqs.append(np.column_stack([s[rows], cols]))
            new_images.append(model.mirror(img[rows], cols))
        seqs = np.concatenate(new_seqs).astype(np.int64)
        images = np.concatenate(new_images)
        total += len(seqs)
        if total > limit:
            raise ValueError(
                f"more than {limit} image sources up to order {len(out) + 1}; lower "
                "max_order or use the ray_tracing early method"
            )
        out.append(seqs)
    return out


def _perpendicular(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether the planes ``a`` and ``b`` are perpendicular: reflections off them commute."""
    dot = np.einsum("ij,ij->i", model.plane_normal[a], model.plane_normal[b])
    return np.abs(dot) < 1e-9


def _image_points(
    model: TraceModel, seqs: np.ndarray, chain: list, source, listener, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """(n, k + 2, 3) the points of the paths from ``source`` to ``listener`` off the plane
    sequences ``seqs`` (n, k) as the image method places them, back from the listener
    towards each of the source's images ``chain[j]`` (off the first j planes) in turn,
    and (n,) whether each such path can be: it meets every plane from the front (farther
    than ``tolerance`` from it), crossing it on the way to its image, but in the corner
    of two perpendicular planes, where it reflects off both at one point. Where a path
    cannot be, its points are not its."""
    n, k = seqs.shape
    points = np.empty((n, k + 2, 3))
    points[:, 0], points[:, k + 1] = source, listener
    possible = np.ones(n, dtype=bool)
    current = np.broadcast_to(listener, (n, 3)).copy()
    for j in range(k, 0, -1):
        plane = seqs[:, j - 1]
        normal = model.plane_normal[plane]
        near = np.einsum("ij,ij->i", normal, current) - model.plane_offset[plane]
        far = np.einsum("ij,ij->i", normal, chain[j]) - model.plane_offset[plane]
        two_sided = model.plane_two_sided[plane]
        front = np.where(two_sided, np.abs(near) > tolerance, near > tolerance)
        # In the corner where it meets the plane of the next reflection, perpendicular
        # to it, the path reflects off both at one point.
        corner = np.zeros(n, dtype=bool)
        if j < k:
            behind = np.where(two_sided, np.abs(far) > tolerance, far < -tolerance)
            corner = (np.abs(near) <= tolerance) & behind & _perpendicular(model, plane, seqs[:, j])
        possible &= (front & (near * far < 0.0)) | corner
        t = near / np.where(possible, near - far, 1.0)
        current = current + t[:, None] * (chain[j] - current)
        points[:, j] = current
    return points, possible


def _height_over_plane(model: TraceModel, points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """The distance (m) of ``points`` (..., 3) from the planes ``planes`` (...), the two
    broadcast together, negative behind a plane (against its normal)."""
    normals = model.plane_normal[planes]
    return np.einsum("...k,...k->...", points, normals) - model.plane_offset[planes]


def _height_over_face(model: TraceModel, points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The distance (m) of ``points`` (..., 3) from the planes the faces ``faces`` (...)
    themselves lie in (:attr:`TraceModel.face_planes`), the two broadcast together,
    negative behind one (against its plane's normal)."""
    normal, offset = model.face_planes
    return np.einsum("...k,...k->...", points, normal[faces]) - offset[faces]


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``np.cross`` of 3-vectors (..., 3), without its overhead."""
    u0, u1, u2 = u[..., 0], u[..., 1], u[..., 2]
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0], axis=-1)


def _over(model: TraceModel, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether each of ``points`` (p, 3) projected onto the face of ``faces`` (p,) beside
    it lies on it (p,), its edges included: on the triangle, or no farther from it than
    the least distance at which a ray meets a face (``model.bvh.min_distance``, a
    billionth of the model's diagonal). The model's tolerance for planes is no room past
    an edge: a face a path meets where it ends is another one's, however near its
    plane."""
    mesh = model.room.mesh
    tri = mesh.vertices[mesh.faces[faces]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = _cross(b - a, c - a)
    area2 = np.einsum("ij,ij->i", n, n)
    # Barycentric coordinates of the points projected onto the triangles.
    w_a = np.einsum("ij,ij->i", _cross(c - b, points - b), n) / area2
    w_b = np.einsum("ij,ij->i", _cross(a - c, points - c), n) / area2
    w_c = 1.0 - w_a - w_b
    # Beside the triangle: how far the projected point lies from the nearest side.
    at = w_a[:, None] * a + w_b[:, None] * b + w_c[:, None] * c
    beside = np.full(len(at), np.inf)
    for p, q in ((a, b), (b, c), (c, a)):
        side = q - p
        along = np.einsum("ij,ij->i", at - p, side) / np.maximum(
            np.einsum("ij,ij->i", side, side), 1e-300
        )
        gap = at - p - np.clip(along, 0.0, 1.0)[:, None] * side
        beside = np.minimum(beside, np.linalg.norm(gap, axis=1))
    return ((w_a >= 0.0) & (w_b >= 0.0) & (w_c >= 0.0)) | (beside <= model.bvh.min_distance)


def _faces_at(
    model: TraceModel, plane: int, point: np.ndarray, tolerance: float, own: bool = True
) -> np.ndarray:
    """The faces of ``plane`` that ``point`` lies on (edges included, :func:`_over`), in
    ascending order: it lies within ``tolerance`` of the plane or, with ``own``, of the
    face's own plane (see :attr:`TraceModel.face_planes`). Openings are holes: no point
    lies on one."""
    height = abs(model.plane_normal[plane] @ point - model.plane_offset[plane])
    if height > tolerance + (2.0 * model.plane_spread[plane] if own else 0.0):
        return np.empty(0, dtype=np.int64)
    candidates = np.flatnonzero((model.face_plane == plane) & ~model.transparent)
    if height > tolerance:  # off the plane, but maybe on a face that is not in it
        normal, offset = model.face_planes
        near = np.abs(normal[candidates] @ point - offset[candidates]) <= tolerance
        candidates = candidates[near]
    if len(candidates) == 0:
        return candidates
    at = np.broadcast_to(point, (len(candidates), 3))
    return candidates[_over(model, candidates, at)]


def _face_at(
    model: TraceModel, plane: int, point: np.ndarray, tolerance: float, own: bool = True
) -> int:
    """The first of the faces of ``plane`` that ``point`` lies on (:func:`_faces_at`),
    or -1."""
    faces = _faces_at(model, plane, point, tolerance, own)
    return int(faces[0]) if len(faces) else -1


def _goes_on(
    model: TraceModel, plane: int, point: np.ndarray, way: np.ndarray, tolerance: float
) -> bool:
    """Whether a face of ``plane`` that ``point`` lies on (:func:`_faces_at`) goes on
    from it along ``way`` (in the plane, of any length): ``way`` leaves the face across
    none of its edges that the point lies within ``tolerance`` of, or beyond (a way
    along an edge, to the plane tolerance, stays on the face). With no way (zero) a
    face the point lies on is enough."""
    faces = _faces_at(model, plane, point, tolerance)
    if len(faces) == 0:
        return False
    mesh = model.room.mesh
    corner = mesh.vertices[mesh.faces[faces]]  # (f, 3, 3)
    side = np.roll(corner, -1, axis=1) - corner  # each edge, from its first corner
    side /= np.maximum(np.linalg.norm(side, axis=2, keepdims=True), 1e-300)
    inward = np.roll(corner, -2, axis=1) - corner  # to the corner across the edge
    inward -= np.einsum("fej,fej->fe", inward, side)[..., None] * side
    inward /= np.maximum(np.linalg.norm(inward, axis=2, keepdims=True), 1e-300)
    depth = np.einsum("fej,fej->fe", point - corner, inward)
    out = np.einsum("fej,j->fe", inward, way) < -PLANE_TOLERANCE * np.linalg.norm(way, axis=0)
    return bool(np.any(~np.any((depth <= tolerance) & out, axis=1)))


#: Points looked up at once in a plane's grid of faces (:func:`_faces_under`).
_CHUNK = 1024
#: A face whose box covers more cells of its plane's grid than this is tried for every
#: point (:func:`_face_grid`).
_GRID_CELLS = 64


def _face_grid(model: TraceModel, plane: int, tolerance: float):
    """The faces of ``plane`` binned for :func:`_faces_under` (worked out once for each
    plane and ``tolerance``, in :attr:`TraceModel.face_grids`): each face's box, grown by
    more than the room :func:`_over` leaves round its edges (the intersection test's
    least distance, no more than ``tolerance``: the triangle scaled about its centroid by
    1 + 3 ``tolerance`` / its least height holds it, and that moves no corner farther
    along an axis than 3 ``tolerance`` / least height times the box's largest side) and
    by ``tolerance`` and twice the plane's spread, then doubled and added the model's
    rounding (:attr:`TraceModel.height_error`); and a grid of square cells across the two
    axes the plane faces least, listing the faces whose boxes overlap each (those that
    overlap more than :data:`_GRID_CELLS` of them, or with no finite box, listed apart,
    for every point)."""
    key = (int(plane), float(tolerance))
    grid = model.face_grids.get(key)
    if grid is not None:
        return grid
    faces = np.flatnonzero(model.face_plane == plane)
    mesh = model.room.mesh
    tri = mesh.vertices[mesh.faces[faces]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = _cross(b - a, c - a)
    longest = np.linalg.norm(np.stack([b - a, c - b, a - c]), axis=2).max(axis=0)
    least = np.sqrt(np.einsum("ij,ij->i", n, n)) / np.maximum(longest, 1e-300)
    low, high = tri.min(axis=1), tri.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        grow = 3.0 * tolerance / least * (high - low).max(axis=1)
    grow = np.where(np.isfinite(grow), grow, np.inf)
    grow = 2.0 * (grow + tolerance + 2.0 * model.plane_spread[plane]) + model.height_error
    low, high = low - grow[:, None], high + grow[:, None]
    axes = np.delete(np.arange(3), np.argmax(np.abs(model.plane_normal[plane])))
    finite = np.isfinite(grow)
    start, cell, shape = np.zeros(2), 1.0, (0, 0)
    first = np.zeros(2, dtype=np.int64)
    listed = np.zeros(0, dtype=np.int64)
    apart = np.flatnonzero(~finite)
    if finite.any():
        start = low[finite][:, axes].min(axis=0)
        span = high[finite][:, axes].max(axis=0) - start
        size = (high - low)[finite][:, axes].max(axis=1)
        cell = max(
            float(np.median(size)),
            float(np.sqrt(span[0] * span[1] / (4 * np.count_nonzero(finite)))),
            float(np.finfo(float).tiny),
        )
        shape = tuple(int(x) for x in np.maximum(np.ceil(span / cell), 1))
        lo = np.clip(np.floor((low[:, axes] - start) / cell), 0, np.array(shape) - 1)
        hi = np.clip(np.floor((high[:, axes] - start) / cell), 0, np.array(shape) - 1)
        lo = np.where(finite[:, None], lo, 0).astype(np.int64)
        count = np.where(finite[:, None], hi, 0).astype(np.int64) - lo + 1
        cells = count.prod(axis=1)
        binned = finite & (cells <= _GRID_CELLS)
        apart = np.flatnonzero(~binned)
        which = np.flatnonzero(binned)
        each = np.repeat(which, cells[which])
        step = np.arange(len(each)) - np.repeat(
            np.cumsum(cells[which]) - cells[which], cells[which]
        )
        at = (lo[each, 0] + step // count[each, 1]) * shape[1] + lo[each, 1] + step % count[each, 1]
        order = np.argsort(at, kind="stable")
        listed = each[order]
        first = np.searchsorted(at[order], np.arange(shape[0] * shape[1] + 1))
    grid = (faces, low, high, axes, start, cell, shape, first, listed, apart)
    model.face_grids[key] = grid
    return grid


def _faces_under(model: TraceModel, plane: int, points: np.ndarray, tolerance: float):
    """Pairs of the ``points`` (m, 3) and faces of ``plane``, as (p,) rows of ``points``
    and (p,) faces, among which are all those where a point projected onto the face lies
    on it (edges included, :func:`_over`), within ``tolerance`` and twice the plane's
    spread of the face's own plane: those whose grown boxes hold the point (see
    :func:`_face_grid`; ``tolerance`` is no less than the intersection test's least
    distance)."""
    faces, low, high, axes, start, cell, shape, first, listed, apart = _face_grid(
        model, plane, tolerance
    )
    m = len(points)
    index = np.floor((points[:, axes] - start) / cell)
    inside = np.all((index >= 0) & (index < np.array(shape)), axis=1)
    at = np.where(inside, index[:, 0] * shape[1] + index[:, 1], 0).astype(np.int64)
    count = np.where(inside, first[at + 1] - first[at], 0)
    rows = np.repeat(np.arange(m), count)
    step = np.arange(len(rows)) - np.repeat(np.cumsum(count) - count, count)
    local = listed[np.repeat(first[at], count) + step]
    if len(apart):
        rows = np.concatenate([rows, np.repeat(np.arange(m), len(apart))])
        local = np.concatenate([local, np.tile(apart, m)])
    held = np.all((low[local] <= points[rows]) & (points[rows] <= high[local]), axis=1)
    return rows[held], faces[local[held]]


def _height_near(model: TraceModel, points, planes, tolerance: float) -> np.ndarray:
    """The distance (m) of ``points`` (..., 3) from the planes ``planes`` (...), the two
    broadcast together, as :func:`_height_over_plane` gives it; but where a plane's faces
    do not all lie in it (:attr:`TraceModel.plane_spread`) and the point lies within
    ``tolerance`` and twice that spread of it, the one of least size of that and the
    point's distances from the own planes (:attr:`TraceModel.face_planes`) of the plane's
    faces it lies over (edges included, :func:`_over`; of equal ones the first face's):
    a point on any face of the plane lies on it, and its side is taken from the faces
    there."""
    height = _height_over_plane(model, points, planes)
    spread = model.plane_spread[planes]
    near = np.nonzero((spread > 0.0) & (np.abs(height) <= tolerance + 2.0 * spread))
    if len(near[0]) == 0:
        return height
    points = np.broadcast_to(points, (*height.shape, 3))[near]
    planes = np.broadcast_to(planes, height.shape)[near]
    merged = height[near]
    normal, offset = model.face_planes
    for plane in np.unique(planes):
        on = np.flatnonzero(planes == plane)
        for part in np.array_split(on, -(-len(on) // _CHUNK)):
            rows, faces = _faces_under(model, plane, points[part], tolerance)
            at = points[part][rows]
            own = np.einsum("ij,ij->i", at, normal[faces]) - offset[faces]
            over = _over(model, faces, at)
            rows, faces, own = rows[over], faces[over], own[over]
            order = np.lexsort((faces, np.abs(own), rows))
            rows, own = rows[order], own[order]
            least = np.flatnonzero(np.diff(rows, prepend=-1) != 0)
            rows, own = part[rows[least]], own[least]
            closer = np.abs(own) < np.abs(merged[rows])
            height[tuple(i[rows[closer]] for i in near)] = own[closer]
    return height


def find_paths(
    model: TraceModel,
    source: np.ndarray,
    listener: np.ndarray,
    candidates: list[np.ndarray],
    *,
    speed_of_sound: float,
    air: np.ndarray,
    max_delay_s: float,
    directivity: Directivity | None = None,
    orientation: np.ndarray | None = None,
    coherent: int = 0,
    max_partitions: int | None = None,
) -> EarlyReflections:
    """Validate candidate plane sequences for one listener; add the direct sound.

    Each path's energy is weighted by the source's ``directivity`` (default: omni) in
    the direction it leaves the source; ``orientation`` has the source's forward, left
    and up axes as columns. In the lowest ``coherent`` bands each path also carries the
    phase of its reflections (:mod:`~acoustic_spaces.engine.coherent`). With
    ``max_partitions``, a reflected path that passes through more partitions
    (:attr:`TraceModel.partition`) is left out, as the stochastic late tracer leaves it
    to the late part (ray-traced candidates find no such path but where planes are
    shared, as the floor of rooms in a row).
    """
    source = np.asarray(source, dtype=np.float64)
    listener = np.asarray(listener, dtype=np.float64)
    bands = model.absorption.shape[1]

    tolerance = _EDGE_TOLERANCE * float(np.ptp(model.room.mesh.vertices, axis=0).max())
    counted = np.column_stack([~model.transparent, model.partition])
    parts = []
    direct = np.zeros((1, 0), dtype=np.int64)
    for seqs in [direct, *candidates]:
        if len(seqs) == 0:
            continue
        k = seqs.shape[1]
        # Image chain: level j is the source mirrored across the first j planes.
        chain = [np.broadcast_to(source, (len(seqs), 3)).copy()]
        for j in range(k):
            chain.append(model.mirror(chain[-1], seqs[:, j]))
        distance = np.linalg.norm(chain[-1] - listener, axis=1)
        keep = distance / speed_of_sound <= max_delay_s
        seqs, chain, distance = seqs[keep], [c[keep] for c in chain], distance[keep]
        n = len(seqs)
        if n == 0:
            continue

        # Trace back from the listener towards each image in turn.
        current = np.broadcast_to(listener, (n, 3)).copy()
        valid = np.ones(n, dtype=bool)
        faces = np.full((n, k), -1, dtype=np.int64)
        points = [None] * (k + 1)
        points[k] = current.copy()
        ignore = np.full(n, -1, dtype=np.int64)
        through = np.ones((n, bands))
        # Per path: the faces crossed that are not openings, and the partitions.
        crossings = np.zeros((n, 2), dtype=np.int64)
        # The reflections at the current point (from reflection ``first`` on, more than
        # one in a corner) and the point the path came to them from.
        before = current.copy()
        first = np.full(n, k - 1, dtype=np.int64)
        for j in range(k - 1, -1, -1):
            plane = seqs[:, j]
            # A reflection point on an edge already lies on the next plane's face too: the
            # path reflects off both at the same point (a corner), so there is nothing to
            # trace. One in that plane off its faces is no corner: the path goes on to it.
            # Nor is one beside the plane (farther than the least distance at which a ray
            # meets a face) from which the path does not go on to a face of it: on the
            # side of the image it heads for, or meeting the plane past its faces.
            at = np.full(n, -1, dtype=np.int64)
            if j < k - 1:
                near = np.flatnonzero(valid)
                height = _height_near(model, current[near], plane[near], tolerance)
                there = _height_over_plane(model, chain[j + 1][near], plane[near])
                beside = np.abs(height) > model.bvh.min_distance
                held = (np.abs(height) <= tolerance) & (~beside | (height * there < 0.0))
                for i, h, t, far in zip(
                    near[held], height[held], there[held], beside[held], strict=True
                ):
                    if far:
                        crossing = current[i] + h / (h - t) * (chain[j + 1][i] - current[i])
                        if _face_at(model, plane[i], crossing, tolerance) < 0:
                            continue
                    at[i] = _face_at(model, plane[i], current[i], tolerance)
            corner = at >= 0
            trace = valid & ~corner
            start = current
            delta = chain[j + 1] - current
            length = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(length, 1e-300)[:, None]
            face, current, share, crossed = _towards_plane(
                model, current, direction, length, ignore, plane, trace, tolerance, counted
            )
            through *= share
            crossings += crossed
            hit = face >= 0
            if j < k - 1:
                # The path leaves the reflections at ``start`` for this plane.
                rows = np.flatnonzero(trace & hit)
                planes = np.where(np.arange(j + 1, k) <= first[rows, None], seqs[rows, j + 1 :], -1)
                turns, share, crossed = _turn(
                    model,
                    planes,
                    faces[rows, j + 1 :],
                    before[rows],
                    start[rows],
                    current[rows],
                    tolerance,
                    counted,
                )
                valid[rows] &= turns
                through[rows] *= share
                crossings[rows] += crossed
            before[trace], first[trace] = start[trace], j
            valid &= hit | corner
            face[corner] = at[corner]
            faces[:, j] = face
            points[j] = current.copy()
            ignore = face
        delta = source - current
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1e-300)[:, None]
        share, crossed = model.passes(
            current, direction, length * (1.0 - 1e-9), ignore, counted=counted
        )
        through *= share
        crossings += crossed
        if k:
            # The path leaves the last reflections for the source.
            rows = np.flatnonzero(valid & np.any(through > 0.0, axis=1))
            planes = np.where(np.arange(k) <= first[rows, None], seqs[rows], -1)
            ends = np.broadcast_to(source, (len(rows), 3))
            turns, share, crossed = _turn(
                model, planes, faces[rows], before[rows], current[rows], ends, tolerance, counted
            )
            valid[rows] &= turns
            through[rows] *= share
            crossings[rows] += crossed
        valid &= np.any(through > 0.0, axis=1)
        if max_partitions is not None and k > 0:
            valid &= crossings[:, 1] <= max_partitions
        if not valid.any():
            continue

        idx = np.flatnonzero(valid)
        # Plane sequences that commute (perpendicular planes) make the same mirror off
        # the same surfaces: one path. Other mirrors' images of a source near where their
        # planes meet lie as near each other, and are paths of their own; so are paths
        # off other surfaces that make the same mirror (in two rooms as deep either side
        # of a partition: off the partition and one room's end wall, or the other's).
        # Two paths with one mirror off the same surfaces in another order still count
        # as one (from the fourth order on in such rooms).
        key = np.hstack(
            [_mirrors(model, seqs[idx], tolerance), _surfaces(model, seqs[idx], tolerance)]
        )
        _, unique = np.unique(key, axis=0, return_index=True)
        idx = idx[np.sort(unique)]
        found_points, found_faces = [p[idx] for p in points], faces[idx]
        _apart(
            model,
            seqs[idx],
            found_points,
            found_faces,
            crossings[idx],
            source,
            listener,
            tolerance,
            counted,
        )
        parts.append(
            assemble_paths(
                model,
                source,
                listener,
                found_points,
                found_faces,
                distance[idx],
                through[idx],
                crossings[idx, 0],
                speed_of_sound=speed_of_sound,
                air=air,
                directivity=directivity,
                orientation=orientation,
                coherent=coherent,
            )
        )
    found = EarlyReflections.concatenate(parts, bands)
    if model.room.outdoor is not None and model.room.outdoor.terrain is not None:
        from acoustic_spaces.engine.terrain import terrain_paths

        found = terrain_paths(
            model,
            found,
            source,
            listener,
            speed_of_sound=speed_of_sound,
            air=air,
            max_delay_s=max_delay_s,
            directivity=directivity,
            orientation=orientation,
            coherent=coherent,
        )
    return found


def _apart(
    model: TraceModel, seqs, points: list, faces, crossings, source, listener, tolerance, counted
) -> None:
    """The paths off the plane sequences ``seqs`` (n, k) through ``points`` (k + 1 arrays of
    (n, 3), the listener last) and ``faces`` (n, k), as :func:`find_paths` has found them,
    where they reflect off two planes at one point that the search joined into a corner
    (a reflection point within ``tolerance``, the model's tolerance for planes, of the next
    plane) and the image method places that path's points apart (farther than the
    intersection test's least distance, or the rounding of the coordinates where that is
    larger): in place, those points, in the order of the planes or in another that gives
    the same image (one that swaps only perpendicular planes), and the faces they lie on,
    where that path can be, each point lies on a face of its plane and its legs cross the
    faces the path's ``crossings`` (n, 2) count (of ``counted``, as :func:`find_paths`
    counts them), nothing solid. The corner lies up to the tolerance from where the path
    meets each plane; the share of sound the path lets through stays the corner's."""
    n, k = faces.shape
    if k < 2 or n == 0:
        return
    joined = np.zeros(n, dtype=bool)
    for j in range(k - 1):
        joined |= np.all(points[j] == points[j + 1], axis=1)
    if not joined.any():
        return
    least = max(model.bvh.min_distance, model.height_error)
    orders = [(o, np.argsort(o)) for o in itertools.permutations(range(k))]
    for i in np.flatnonzero(joined):
        seq = seqs[i]
        for order, where in orders:
            swapped = [(a, b) for a in range(k) for b in range(a + 1, k) if where[a] > where[b]]
            if not all(_perpendicular(model, seq[[a]], seq[[b]])[0] for a, b in swapped):
                continue
            planes = seq[list(order)][None, :]
            chain = [np.asarray(source, dtype=np.float64)[None, :]]
            for j in range(k):
                chain.append(model.mirror(chain[-1], planes[:, j]))
            exact, possible = _image_points(model, planes, chain, source, listener, least)
            legs = np.linalg.norm(np.diff(exact[0, 1:-1], axis=0), axis=1)
            if not possible[0] or np.any(legs <= least):
                continue
            on = [_face_at(model, int(planes[0, j]), exact[0, j + 1], tolerance) for j in range(k)]
            if min(on) < 0 or not _clear(model, exact[0], on, crossings[i], least, counted):
                continue
            for j in range(k):
                points[j][i] = exact[0, j + 1]
            faces[i] = on
            break


def _clear(model: TraceModel, points, faces, crossings, least, counted) -> bool:
    """Whether the legs between ``points`` (k + 2, 3: the source, the reflection points
    off ``faces`` and the listener) let sound through and cross the faces ``crossings``
    counts (of ``counted``), each traced from its start to ``least`` short of its end."""
    crossed = np.zeros_like(crossings)
    for j in range(len(points) - 1):
        way = points[j + 1] - points[j]
        length = float(np.linalg.norm(way))
        if length <= 2.0 * least:
            continue
        ignore = faces[j - 1] if j else -1
        share, hit = model.passes(points[j], way / length, length - least, ignore, counted=counted)
        if not np.any(share > 0.0):
            return False
        crossed += hit[0]
    return bool(np.array_equal(crossed, crossings))


def _mirrors(model: TraceModel, seqs: np.ndarray, tolerance: float) -> np.ndarray:
    """(n, 12) where the planes of each sequence of ``seqs`` (n, k), mirroring in turn,
    take four points (a corner of the model's box and the three its largest extent from
    it along the axes), in steps of ``tolerance``: the same for the sequences that make
    one mirror, wherever a source lies."""
    vertices = model.room.mesh.vertices
    size = float(np.ptp(vertices, axis=0).max())
    marks = vertices.min(axis=0) + size * np.vstack([np.zeros(3), np.eye(3)])
    moved = np.tile(marks, (len(seqs), 1))
    for j in range(seqs.shape[1]):
        moved = model.mirror(moved, np.repeat(seqs[:, j], 4))
    return np.round(moved.reshape(len(seqs), 12) / tolerance)


def _surfaces(model: TraceModel, seqs: np.ndarray, tolerance: float) -> np.ndarray:
    """(n, k) the surfaces each sequence of ``seqs`` (n, k) reflects off, whatever their
    order and however often: numbers of the planes' own mirrors (the two sides of a
    partition are one surface), greatest first, -1 for no further one."""
    planes = np.arange(model.plane_count)
    _, surface = np.unique(_mirrors(model, planes[:, None], tolerance), axis=0, return_inverse=True)
    met = np.sort(surface.reshape(-1)[seqs], axis=1)
    met[:, 1:][met[:, 1:] == met[:, :-1]] = -1
    return -np.sort(-met, axis=1)


def _along_at_planes(model: TraceModel, source, listener, planes: np.ndarray) -> np.ndarray:
    """(k, n, 3) for each of the k reflections of n paths off the planes ``planes`` (n, k)
    from ``source`` to ``listener`` a vector along the path where it meets that
    reflection's plane, for the angle between the two: the way the path leaves the
    reflection, from the source's image in the planes towards the listener, turned back
    by each plane after it. The image gives the angles wherever the reflection points
    lie: a point that the search took onto a plane, or two reflections it joined into a
    corner, lie up to its tolerance for planes from where the path meets the plane, and
    a leg about as short heads elsewhere than the path."""
    n, k = planes.shape
    image = np.broadcast_to(np.asarray(source, dtype=np.float64), (n, 3)).copy()
    for j in range(k):
        image = model.mirror(image, planes[:, j])
    way = np.asarray(listener, dtype=np.float64) - image
    along = np.zeros((k, n, 3))
    for j in range(k - 1, -1, -1):
        along[j] = way
        normal = model.plane_normal[planes[:, j]]
        way = way - 2.0 * np.einsum("ij,ij->i", normal, way)[:, None] * normal
    return along


def assemble_paths(
    model: TraceModel,
    source: np.ndarray,
    listener: np.ndarray,
    points: list[np.ndarray],
    faces: np.ndarray,
    distance: np.ndarray,
    through: np.ndarray,
    crossings: np.ndarray,
    *,
    speed_of_sound: float,
    air: np.ndarray,
    directivity: Directivity | None = None,
    orientation: np.ndarray | None = None,
    coherent: int = 0,
) -> EarlyReflections:
    """Paths of one order k from ``source`` to ``listener`` with their energies: their
    reflection points ``points`` (k + 1 arrays of (n, 3), the listener last), the faces
    they reflect off (n, k), their lengths (n,), the share of sound the surfaces they
    pass let through (n, bands) and how many they cross (n,), as :func:`find_paths`
    finds them."""
    n, k = faces.shape
    bands = model.absorption.shape[1]
    dist = distance
    energy = np.exp(-np.outer(dist, air)) / (4.0 * np.pi * dist[:, None] ** 2)
    energy *= through
    spectrum = None
    if coherent:
        spectrum = np.ones((n, *sample_frequencies(bands, coherent).shape), complex)
        cosines = np.full((n, k), np.nan)
    # Reflections off surfaces with an impedance (a ground's) take its spherical
    # wave's reflection factor, in every band: its energy averaged over each band,
    # and in the coherent bands its pressure; on a ground with zones, those of the
    # grounds round the reflection.
    grounds = bool(k) and model.factor_faces(faces).any()
    factor = None
    before = np.broadcast_to(source, (n, 3))
    if coherent or grounds:
        normals = model.plane_normal[model.face_plane[faces]]
        along = _along_at_planes(model, source, listener, model.face_plane[faces])
    for j in range(k):
        incoming = points[j] - before
        energy *= model.reflection(faces[:, j], incoming, j + 1, impedance=True)
        if coherent or grounds:
            cosine = np.abs(np.einsum("ij,ij->i", normals[:, j], along[j])) / np.maximum(
                np.linalg.norm(along[j], axis=1), 1e-300
            )
            if coherent:
                spectrum *= reflection_phases(model, faces[:, j], cosine, coherent)
                own = ~model.factor_faces(faces[:, j]) if grounds else slice(None)
                cosines[own, j] = cosine[own]
            if grounds:
                following = points[j + 1] if j + 1 < k else listener
                ends = (before, points[j], np.broadcast_to(following, before.shape))
                q = ground_factors(model, faces[:, j], cosine, dist, ends=ends)
                factor = q if factor is None else factor * q
        before = points[j]
    if factor is not None:
        gain = np.mean(np.abs(factor) ** 2, axis=2)  # (n, bands)
        energy *= gain
        if coherent:
            level = np.sqrt(gain[:, :coherent, None])
            spectrum *= np.where(
                level > 0.0, factor[:, :coherent] / np.where(level > 0.0, level, 1.0), 1.0
            )
    first = points[0] if k else np.broadcast_to(listener, (n, 3))
    departure = first - source
    departure /= np.maximum(np.linalg.norm(departure, axis=1), 1e-300)[:, None]
    energy *= source_weights(directivity, orientation, departure, energy.shape[1], distances=dist)
    last = points[k - 1] if k else np.broadcast_to(source, (n, 3))
    arrival = last - listener
    arrival /= np.maximum(np.linalg.norm(arrival, axis=1), 1e-300)[:, None]
    path_points = [
        np.vstack([source, *[points[j][i] for j in range(k)], listener]) for i in range(n)
    ]
    return EarlyReflections(
        order=np.full(n, k, dtype=np.int64),
        delay_s=dist / speed_of_sound,
        distance_m=dist,
        direction=arrival,
        energy=energy,
        points=path_points,
        faces=[faces[i].copy() for i in range(n)],
        crossings=crossings,
        spectrum=spectrum,
        cosines=[cosines[i].copy() for i in range(n)] if coherent else [],
    )


def _towards_plane(model, start, direction, length, ignore, plane, trace, tolerance, counted):
    """For the rays from ``start`` (n, 3) along ``direction`` up to ``length`` where
    ``trace`` holds: the face of ``plane`` (n,) each meets first, passing straight
    through openings and faces of other planes that let sound through, or -1; where it
    meets it (the start where it does not); the share of sound that passes the faces on
    the way (n, bands); and how many of them it crossed of each column of ``counted``
    (F, m), (n, m). A hit on another face exactly where it meets the expected plane
    counts as on the plane (whether the path goes on across that face there,
    :func:`_turn` says): one within ``tolerance`` of the plane, in which the image the
    ray heads for is mirrored, not of the own plane of one of its faces (where they do
    not all lie in the plane, :attr:`TraceModel.plane_spread`), over a face of it
    (:func:`_face_at`), on a ray whose two ends lie on either side of the plane (one
    that does not cross the plane reflects nowhere off it) and that meets that face's
    own plane on a face of the plane: a ray that met another face beside the line where
    the two join meets it past the edge."""
    n = len(start)
    ends = start + direction * np.asarray(length, dtype=float)[:, None]
    across = _height_over_plane(model, start, plane) * _height_over_plane(model, ends, plane) < 0.0
    bands = model.absorption.shape[1]
    transparent = model.transparent
    face = np.full(n, -1, dtype=np.int64)
    end = start.copy()
    share = np.ones((n, bands))
    crossed = np.zeros((n, counted.shape[1]), dtype=np.int64)
    position = start.copy()
    remaining = np.asarray(length, dtype=float).copy()
    skip = np.asarray(ignore, dtype=np.int64).copy()
    alive = np.flatnonzero(trace)
    for _ in range(MAX_CROSSINGS + 1):
        if len(alive) == 0:
            break
        t, hit = model.intersect(position[alive], direction[alive], remaining[alive], skip[alive])
        found = hit >= 0
        rows, hit, t = alive[found], hit[found], t[found]
        position[rows] += direction[rows] * t[:, None]
        remaining[rows] -= t
        on_plane = (model.face_plane[hit] == plane[rows]) & ~transparent[hit]
        normal, offset = model.face_planes
        for i in np.flatnonzero(~on_plane & across[rows]):
            row = rows[i]
            other = _face_at(model, plane[row], position[row], tolerance, own=False)
            if other < 0:
                continue
            height = float(np.einsum("k,k->", normal[other], position[row])) - offset[other]
            slope = float(np.einsum("k,k->", normal[other], direction[row]))
            if slope == 0.0:
                continue
            crossing = position[row] - (height / slope) * direction[row]
            if _face_at(model, plane[row], crossing, tolerance) >= 0:
                hit[i], on_plane[i] = other, True
        face[rows[on_plane]] = hit[on_plane]
        end[rows[on_plane]] = position[rows[on_plane]]
        rows, hit = rows[~on_plane], hit[~on_plane]
        if model.transmission is None:
            break
        through = model.transmission[hit]
        share[rows] *= through
        crossed[rows] += counted[hit]
        skip[rows] = hit
        alive = rows[np.any(through > 0.0, axis=1)]
    return face, end, share, crossed


def _crossed_here(model, reflecting, before, point, after, tolerance, near):
    """For a path that reflects off the planes ``reflecting`` (r,) at ``point``, coming
    from ``before`` and going on to ``after``: the faces it crosses right at the point,
    where neither leg's intersection test meets them, as a wall standing on the floor
    where the path reflects off the floor, or the walls meeting there. Returns them (k,),
    those on the leg arriving at the point first, and (k,) whether the leg arriving at
    it is the one that crosses each.

    The planes the path crosses near the point are those within ``tolerance`` of it that
    ``before`` and ``after`` lie on either side of, both farther than the intersection
    test's least distance (``before`` may be a reflection less than ``tolerance`` from
    the point, in a corner), and that it does not reflect off there; of coincident ones
    (within the model's plane tolerance; a wall drawn as one face for each room) only
    one counts, as the intersection test meets only one of them beside the point. Those
    right at the point are the ones that the leg crossing them meets within ``near``
    (2,) of it: the distance from its end within which the leg that arrives at the point
    meets no face, and from its start for the leg that leaves it; where the point lies
    on the side of ``after`` already, that leg is the one arriving. The legs' own tests
    meet the others (see :func:`_meets_face`).

    Which faces the path crosses right at the point is decided as on one side of it, the
    path moved off it along all the surfaces it meets there: across the plane of
    incidence for one surface (or coincident ones), along the line where two meet; not
    at all where three or more meet. Those surfaces are the planes ``reflecting`` and
    every other plane with a face (not an opening) within ``tolerance`` of the point
    that is not one of those crossed near it or parallel to one, nor crossed by the path
    (see :func:`_on_either_side`), as the floor beside a reflection in its corner with a
    wall; each faces the side of whichever of ``before`` and ``after`` is farther from
    it. On each side, a plane crossed is covered where its faces cover it beside the
    point that way (see :func:`_cover`). The side taken is one where those the legs'
    tests meet are covered as they meet them, then one where the faces the path
    reflects off go on ``tolerance`` / 2 along that way (see :func:`_on_faces`), then
    the one with fewer covered right at the point; their faces are the ones crossed.

    Where the faces the path reflects off go on past the point on one side of a face's
    plane only (see :func:`_side_reflected`), it reflects on that side, so the leg from
    the other side crosses that face; elsewhere the leg arriving where the point lies
    on the side of ``after`` already.

    Distances from a plane are those of :func:`_height_near` (from the faces a point lies
    over, where they are not all in the plane)."""
    least = model.bvh.min_distance
    planes = np.arange(model.plane_count)
    normals = model.plane_normal
    height, below, above = _height_near(
        model, np.stack([point, before, after])[:, None], planes, tolerance
    )
    arriving = (point - before) / max(np.linalg.norm(point - before), 1e-300)
    leaving = (after - point) / max(np.linalg.norm(after - point), 1e-300)
    arrives = height * above > 0.0
    slope = np.abs(np.where(arrives, normals @ arriving, normals @ leaving))
    crossing = _on_either_side(below, above, least)
    crossing[reflecting] = False
    here = crossing & (np.abs(height) <= np.where(arrives, near[0], near[1]) * slope)
    none = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    if not here.any():
        return none
    # Those right at the point first, so that of coincident planes one of them counts.
    crossed: list[int] = []
    for q in np.concatenate([np.flatnonzero(here), np.flatnonzero(crossing & ~here)]):
        if abs(height[q]) <= tolerance and all(
            abs(normals[q] @ normals[c]) < 1.0 - PLANE_TOLERANCE for c in crossed
        ):
            crossed.append(int(q))
    at_point = here[crossed]
    # The surfaces the path meets there without crossing them, and the way along them.
    others = [
        q
        for q in np.flatnonzero(np.abs(height) <= tolerance)
        if q not in reflecting
        and all(abs(normals[q] @ normals[c]) < 1.0 - PLANE_TOLERANCE for c in crossed)
        and _face_at(model, q, point, tolerance) >= 0
    ]
    others = np.array(others, dtype=np.int64)
    others = others[~_on_either_side(below[others], above[others], tolerance)]
    surfaces = np.concatenate([reflecting, others])
    farther = np.where(
        np.abs(below[surfaces]) >= np.abs(above[surfaces]), below[surfaces], above[surfaces]
    )
    front = normals[surfaces] * np.sign(farther)[:, None]
    _, size, axes = np.linalg.svd(front)
    rank = np.count_nonzero(size > PLANE_TOLERANCE)
    along = np.zeros(3)
    if rank < 3:
        along = axes[2] if rank == 2 else np.cross(axes[0], arriving)
        along /= max(np.linalg.norm(along), 1e-300)
    step = tolerance / 2
    cover = np.array([_cover(model, q, point, front, along, step) for q in crossed])
    traced = [
        _meets_face(
            model,
            q,
            point,
            -arriving if arrives[q] else leaving,
            abs(height[q]) / max(slope[q], 1e-300) + least,
        )
        for q in np.array(crossed)[~at_point]
    ]
    choice = min(
        (
            not np.array_equal(cover[~at_point, side] >= 0, traced),
            not _on_faces(model, reflecting, point + (1 - 2 * side) * step * along),
            np.count_nonzero(cover[at_point, side] >= 0),
            side,
        )
        for side in (0, 1)
    )
    faces = cover[at_point, choice[-1]]
    faces = faces[faces >= 0]
    arrive = arrives[model.face_plane[faces]]
    for i, face in enumerate(faces):
        side = _side_reflected(model, reflecting, model.face_plane[face], point, after, tolerance)
        if side:
            arrive[i] = side > 0
    order = np.argsort(~arrive, kind="stable")
    return faces[order], arrive[order]


def _cover(model, plane, point, front, along, step):
    """The faces of ``plane`` (-1 for none; not openings) that cover it beside ``point``
    (moved onto it) right in front of surfaces facing ``front`` (s, 3), on the side of
    ``along`` and on the other. In ``plane``, the ways in front of all the surfaces make
    a wedge, to which a surface parallel to ``plane`` (within the model's plane
    tolerance) adds no side; the point tested lies ``step`` from ``point`` in the middle
    of it, on both sides. Where all the surfaces face the same way in ``plane`` (one
    surface, or coincident ones), the wedge is half of it, and the points tested lie
    also ``step`` to either side of that one: the one on the side of ``along`` for that
    side and the other for the other; where ``along`` (unit, or zero) runs along the
    way to neither side beyond the plane tolerance, both for each side, covered (by the
    face under the first) where faces cover both. With no wedge, or one with nothing in
    it, nothing covers it."""
    normal = model.plane_normal[plane]
    away = front - np.outer(front @ normal, normal)
    size = np.linalg.norm(away, axis=1)
    away = away[size > PLANE_TOLERANCE] / size[size > PLANE_TOLERANCE, None]
    if len(away) == 0:
        return -1, -1
    # In front of each surface lies half of ``plane``: the ways within a quarter turn of
    # the way in front. Where those halves overlap, with angles from the first way in
    # front, is the wedge: from a quarter turn short of the largest angle to a quarter
    # turn past the smallest.
    across = np.cross(normal, away[0])
    angle = np.arctan2(away @ across, away @ away[0])
    low, high = angle.max() - np.pi / 2, angle.min() + np.pi / 2
    if low >= high:
        return -1, -1
    middle = (low + high) / 2
    up = np.cos(middle) * away[0] + np.sin(middle) * across
    point = point - (normal @ point - model.plane_offset[plane]) * normal
    near = model.bvh.min_distance
    if np.abs(angle).max() > PLANE_TOLERANCE:
        face = _face_at(model, plane, point + step * up, near)
        return face, face
    if across @ along < 0.0:
        across = -across
    faces = [_face_at(model, plane, point + step * (up + way), near) for way in (across, -across)]
    if abs(across @ along) <= PLANE_TOLERANCE:
        both = faces[0] if min(faces) >= 0 else -1
        return both, both
    return faces[0], faces[1]


def _meets_face(model, plane, start, direction, reach) -> bool:
    """Whether the intersection test meets a face (not an opening) of ``plane``, or of a
    plane coincident with it (within the model's plane tolerance), on the way from
    ``start`` along ``direction`` (unit) up to ``reach``."""
    normal = model.plane_normal[plane]
    skip = -1
    for _ in range(MAX_CROSSINGS + 1):
        t, face = model.intersect(start[None], direction[None], np.array([reach]), np.array([skip]))
        if face[0] < 0:
            return False
        if abs(model.plane_normal[model.face_plane[face[0]]] @ normal) >= 1.0 - PLANE_TOLERANCE:
            if not model.transparent[face[0]]:
                return True
        start, reach, skip = start + t[0] * direction, reach - t[0], face[0]
    return False


def _on_faces(model, planes, point) -> bool:
    """Whether the foot of ``point`` on each of ``planes`` (r,) lies on a face of it
    (edges included, to the intersection test's least distance; not an opening)."""
    normals = model.plane_normal[planes]
    heights = normals @ point - model.plane_offset[planes]
    near = model.bvh.min_distance
    return all(
        _face_at(model, r, point - h * n, near) >= 0
        for r, h, n in zip(planes, heights, normals, strict=True)
    )


def _side_reflected(model, reflecting, plane, point, after, tolerance) -> int:
    """On which side of ``plane`` a path reflects off the planes ``reflecting`` (r,) at
    ``point``, which lies in ``plane`` as well, going on to ``after``, judged by where
    their faces go on (see :func:`_on_faces`): 1 where they go on only on the side of
    ``after``, -1 only on the other side, 0 on both or neither. The points tested lie
    ``tolerance`` from ``point`` either way along all of ``reflecting``, as directly off
    ``plane`` as that allows; 0 too where that leaves them no farther from ``plane``
    than the intersection test's least distance (as where ``plane`` holds the line the
    planes meet in)."""
    normal = model.plane_normal[plane]
    towards = normal * np.sign(normal @ after - model.plane_offset[plane])
    normals = model.plane_normal[reflecting]
    way = towards - normals.T @ np.linalg.lstsq(normals.T, towards, rcond=None)[0]
    size = np.linalg.norm(way)
    if tolerance * size <= model.bvh.min_distance:
        return 0
    way /= size
    return int(_on_faces(model, reflecting, point + tolerance * way)) - int(
        _on_faces(model, reflecting, point - tolerance * way)
    )


#: Room for rounding where an end of one arc is tested against another arc
#: (:func:`_arcs_meet`).
_ARC_ROUNDING = 1e-9


def _arcs_meet(across: np.ndarray, least: np.ndarray, use: np.ndarray) -> np.ndarray:
    """(p,) whether some unit vector d of a plane has ``across[p, k]`` . d >=
    ``least[p, k]`` for every k that ``use`` (p, K) marks, where ``across`` (p, K, 2) are
    unit vectors: each such bound leaves an arc of the unit circle (all of it up to -1,
    none of it above 1), and the arcs of a row have a direction in common. Where they
    have one and are not each the whole circle, what they have in common ends, going
    round to the left, where one of the arcs does, so those ends are the directions
    tried."""
    least = np.where(use, least, -np.inf)
    cut = (least > -1.0) & (least <= 1.0)
    along = np.where(cut, least, 0.0)
    aside = np.sqrt(1.0 - along**2)
    ends = np.stack(
        [
            along * across[..., 0] - aside * across[..., 1],
            along * across[..., 1] + aside * across[..., 0],
        ],
        axis=-1,
    )
    reach = (
        ends[:, :, None, 0] * across[:, None, :, 0] + ends[:, :, None, 1] * across[:, None, :, 1]
    )
    held = np.all(reach >= least[:, None, :] - _ARC_ROUNDING, axis=2)
    return ~np.any(least > 1.0, axis=1) & (np.any(cut & held, axis=1) | ~np.any(cut, axis=1))


def _inner_corner(model, planes, fronts, point, tolerance) -> np.ndarray:
    """(n,) whether the faces of ``planes`` (n, r; -1 for none; two or more a row, which
    meet at ``point`` (n, 3)) make a corner there that a path reflects in, off all of
    them at that one point, where ``fronts`` (n, r, 3) are the planes' normals towards
    the side the path keeps to (zero for a plane the path runs in, which has no such
    side). A path that reflects off a plane near the point does so on the path's side of
    each of the other planes that have a side. A plane with no face there and one past
    any of those others makes no such corner: at an outer edge, the plane of the wall
    under a roof's edge has its faces below that edge and none over it, where sound from
    above the roof would meet it, so that a path through the edge is no reflection off
    the two. A plane with no face round the point leaves the corner one.

    The faces are looked for on the circle of twice ``tolerance`` about the point in the
    plane (the point itself may lie ``tolerance`` to either side of the other planes):
    the directions from the point in which that circle lies on a triangle of the plane
    (edges included to the intersection test's least distance), on the path's side of
    every other plane by more than twice that distance, or past one of them by as much
    (:func:`_arcs_meet`). Another plane counts where the circle reaches that far to
    both sides of it: not one nearly parallel to the plane."""
    least = model.bvh.min_distance
    radius, clear = 2.0 * tolerance, 2.0 * least
    valid = planes >= 0
    safe = np.maximum(planes, 0)
    normals = model.plane_normal[safe]
    side = np.sign(np.einsum("nrk,nrk->nr", fronts, normals))
    towards = normals * side[:, :, None]
    # The point's own height over each plane, towards the path.
    height = side * _height_over_plane(model, point[:, None], safe)
    mesh = model.room.mesh
    inner = np.ones(len(planes), dtype=bool)
    for a in range(planes.shape[1]):
        # Along plane a, each other plane's normal towards the path: across the line the
        # two planes meet in.
        normal = normals[:, a]
        across = towards - np.einsum("nrk,nk->nr", towards, normal)[:, :, None] * normal[:, None]
        size = np.linalg.norm(across, axis=2)
        bounds = valid & (radius * size > clear + np.abs(height))
        rows = np.flatnonzero(valid[:, a] & bounds.any(axis=1))
        if len(rows) == 0:
            continue
        normal, size, bounds = normal[rows], np.where(bounds, size, 1.0)[rows], bounds[rows]
        across = across[rows] / size[:, :, None]
        # Two axes of plane a at the point, and in them the other planes' normals and how
        # far along each a direction lies clear of that plane, before it and past it.
        ahead = across[np.arange(len(rows)), np.argmax(bounds, axis=1)]
        beside = _cross(normal, ahead)
        lines = np.stack(
            [np.einsum("mrk,mk->mr", across, ahead), np.einsum("mrk,mk->mr", across, beside)],
            axis=-1,
        )
        before = (clear - height[rows]) / (radius * size)
        past = (clear + height[rows]) / (radius * size)
        # The faces of plane a near the point: their corners in those axes, their edges'
        # normals into them, and how far along each a direction lies on the face.
        own = planes[rows, a]
        at, faces = [], []
        for plane in np.unique(own):
            held = np.flatnonzero(own == plane)
            for part in np.array_split(held, -(-len(held) // _CHUNK)):
                found, under = _faces_under(model, plane, point[rows[part]], tolerance)
                solid = ~model.transparent[under]
                at.append(part[found[solid]])
                faces.append(under[solid])
        at, faces = np.concatenate(at), np.concatenate(faces)
        corners = mesh.vertices[mesh.faces[faces]] - point[rows[at]][:, None]
        corners = np.stack(
            [
                np.einsum("pvk,pk->pv", corners, ahead[at]),
                np.einsum("pvk,pk->pv", corners, beside[at]),
            ],
            axis=-1,
        )
        edge = np.roll(corners, -1, axis=1) - corners
        third = np.roll(corners, -2, axis=1) - corners
        inward = np.stack([-edge[..., 1], edge[..., 0]], axis=-1)
        turn = np.sign(inward[..., 0] * third[..., 0] + inward[..., 1] * third[..., 1])
        inward *= (turn / np.maximum(np.linalg.norm(edge, axis=2), 1e-300))[..., None]
        on = (inward[..., 0] * corners[..., 0] + inward[..., 1] * corners[..., 1] - least) / radius
        on[np.any(turn == 0.0, axis=1)] = 2.0  # a face without area: in no direction
        every = np.ones((len(at), 3), dtype=bool)
        front = _arcs_meet(
            np.concatenate([inward, lines[at]], axis=1),
            np.concatenate([on, before[at]], axis=1),
            np.concatenate([every, bounds[at]], axis=1),
        )
        behind = np.zeros(len(at), dtype=bool)
        for b in range(planes.shape[1]):
            behind |= bounds[at, b] & _arcs_meet(
                np.concatenate([inward, -lines[at, b : b + 1]], axis=1),
                np.concatenate([on, past[at, b : b + 1]], axis=1),
                np.ones((len(at), 4), dtype=bool),
            )
        faces_before = np.zeros(len(rows), dtype=bool)
        faces_before[at[front]] = True
        faces_past = np.zeros(len(rows), dtype=bool)
        faces_past[at[behind]] = True
        inner[rows[faces_past & ~faces_before]] = False
    return inner


def _turn(model, planes, faces, before, point, after, tolerance, counted):
    """Paths that reflect off ``planes`` (n, r; -1 for none; more than one in a corner),
    off their faces ``faces`` (n, r), at ``point`` (n, 3), coming from ``before`` (n, 3)
    and going on to ``after`` (n, 3):
    (n,) whether each turns back off every one of those planes, as a reflection does,
    rather than going on across one: in a corner, ``before`` and ``after`` on either
    side of it, both farther from it than the intersection test's least distance (not
    ``tolerance``, within which the corner's point may lie off it); anywhere, the leg
    from the point to ``before`` or to ``after`` crossing it farther from both of its
    ends than the intersection test's least distance (in a corner the point may lie past
    a plane the path reflects off just before, and the leg's own test does not meet the
    face it leaves, or meets another face of that plane as one it passes through). Nor
    does a path turn off two coincident planes (within the model's plane tolerance; a
    wall drawn as one face for each room) at one point: it would go on through the wall
    there, or come back as off one of them. Nor does it turn off planes whose faces make
    no corner on its side of them (see :func:`_inner_corner`), as at an outer edge of a
    building: its side of each plane is that of the farther of ``before`` and ``after``,
    where that lies farther from the plane than the least distance, and none otherwise.
    Also the share of sound (n, bands) and the crossings (n, m) of the faces each
    crosses right at the point (see :func:`_crossed_at`). The leg leaving the point
    meets no face nearer to it than the intersection test's least distance; the one
    arriving meets every face on its way but one right where it meets the plane
    (:func:`_towards_plane`), and the point then lies in that face's plane.

    Sides and distances are those of the plane that each of ``faces`` itself lies in
    (:attr:`TraceModel.face_planes`; its plane where the face lies in it): the face the
    leg arriving met, or the face of the plane under the point (:func:`_face_at`) where
    the leg met another face there or the point is a corner. None within
    :attr:`TraceModel.height_error` of a plane (the rounding of the model's coordinates)
    counts as farther than the least distance, as for a leg that runs that close along
    the plane."""
    least = model.bvh.min_distance
    valid = planes >= 0
    safe = np.maximum(planes, 0)
    normals = model.plane_normal[safe]
    own = np.where(valid, faces, 0)
    here, below, above = _height_over_face(model, np.stack([point, before, after])[:, :, None], own)
    corner = np.count_nonzero(valid, axis=1) > 1
    off = max(least, model.height_error)
    across = corner[:, None] & _on_either_side(below, above, off)
    for end, there in ((before, below), (after, above)):
        way = end - point
        way /= np.maximum(np.linalg.norm(way, axis=1), 1e-300)[:, None]
        slope = np.abs(np.einsum("nrk,nk->nr", model.face_planes[0][own], way))
        across |= _on_either_side(here, there, np.maximum(least * slope, model.height_error))
    coincident = np.abs(normals @ normals.transpose(0, 2, 1)) >= 1.0 - PLANE_TOLERANCE
    coincident &= valid[:, :, None] & valid[:, None, :] & (planes[:, :, None] != planes[:, None, :])
    turns = ~np.any(valid & across, axis=1) & ~np.any(coincident, axis=(1, 2))
    rows = np.flatnonzero(corner & turns)
    if len(rows):
        farther = np.where(np.abs(below) >= np.abs(above), below, above)[rows]
        sides = np.where(valid[rows] & (np.abs(farther) > off), np.sign(farther), 0.0)
        fronts = model.face_planes[0][own[rows]] * sides[:, :, None]
        turns[rows] = _inner_corner(model, planes[rows], fronts, point[rows], tolerance)
    near = np.full((len(point), 2), least)
    share, crossed, _, _ = _crossed_at(
        model, planes, before, point, after, tolerance, counted, near
    )
    return turns, share, crossed


def _on_either_side(below, above, tolerance) -> np.ndarray:
    """Whether two points at heights ``below`` and ``above`` over a plane (any shape,
    negative behind it) lie on either side of it, both farther from it than
    ``tolerance``."""
    far = np.minimum(np.abs(below), np.abs(above)) > tolerance
    return far & (below * above < 0.0)


def _crossed_at(model, planes, before, point, after, tolerance, counted, near):
    """For paths that reflect off ``planes`` (n, r; -1 for none) at ``point`` (n, 3),
    coming from ``before`` (n, 3) and going on to ``after`` (n, 3): the faces each
    crosses right at the point, where neither leg's intersection test meets them (see
    :func:`_crossed_here`, with ``near`` (n, 2)), looked for where a chord across the
    turn, from ``tolerance`` back along the leg arriving to ``tolerance`` along the one
    leaving, but no farther than either leg's other end, meets a face: a chord that ran
    on past ``before`` could pass a face that the path crosses at the point beyond the
    face's end, as round the end of a barrier's arm. Returns the share of sound they
    let through (n, bands; 0 where one lets nothing through), how many of them each
    crosses of each column of ``counted`` (F, m), (n, m), the first of them on the way
    (n,), -1 for none, and whether the leg arriving at the point is the one that
    crosses it (n,)."""
    n = len(point)
    bands = model.absorption.shape[1]
    arriving = point - before
    arriving /= np.maximum(np.linalg.norm(arriving, axis=1), 1e-300)[:, None]
    leaving = after - point
    leaving /= np.maximum(np.linalg.norm(leaving, axis=1), 1e-300)[:, None]
    back = np.minimum(tolerance, np.linalg.norm(point - before, axis=1))[:, None]
    ahead = np.minimum(tolerance, np.linalg.norm(after - point, axis=1))[:, None]
    start = point - back * arriving
    chord = point + ahead * leaving - start
    length = np.linalg.norm(chord, axis=1)
    direction = chord / np.maximum(length, 1e-300)[:, None]
    share = np.ones((n, bands))
    crossed = np.zeros((n, counted.shape[1]), dtype=np.int64)
    first = np.full(n, -1, dtype=np.int64)
    first_arrives = np.zeros(n, dtype=bool)
    _, face = model.intersect(start, direction, length, np.full(n, -1, dtype=np.int64))
    for row in np.flatnonzero(face >= 0):
        reflecting = planes[row][planes[row] >= 0]
        faces, arrives = _crossed_here(
            model, reflecting, before[row], point[row], after[row], tolerance, near[row]
        )
        if len(faces) == 0:
            continue
        share[row] *= 0.0 if model.transmission is None else model.transmission[faces].prod(axis=0)
        crossed[row] += counted[faces].sum(axis=0)
        first[row], first_arrives[row] = faces[0], arrives[0]
    return share, crossed, first, first_arrives
