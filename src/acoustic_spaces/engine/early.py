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
path that reflects exactly at the foot of a wall standing on the surface, and goes on
to the wall's other side, crosses the wall there as it does beside the foot. Openings
are holes: no path reflects off one.

A path may pass straight through surfaces that let sound through (partitions, doors,
openings): its energy is then multiplied by their transmission, band by band.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.coherent import (
    ground_factors,
    reflection_phases,
    sample_frequencies,
)
from acoustic_spaces.engine.model import MAX_CROSSINGS, TraceModel, source_weights
from acoustic_spaces.engine.turbulence import Wander

#: Distance (metres) below which a point counts as lying on a plane.
_ON_PLANE = 1e-9
#: Tolerance, relative to the model size, for a path reflecting exactly on an edge.
_EDGE_TOLERANCE = 1e-7
#: Upper bound on candidates for the exhaustive search.
MAX_CANDIDATES = 2_000_000


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
    #: or -2 for a path over the crests of the terrain
    #: (:mod:`~acoustic_spaces.engine.crests`).
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
        """(n,) True for paths that bend over an edge or the terrain's crests."""
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
        out.append(np.unique(prefix, axis=0) if len(prefix) else np.zeros((0, k), np.int64))
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
        new_seqs, new_images = [], []
        chunk = max(1, 4_000_000 // max(1, model.plane_count))
        for start in range(0, len(seqs), chunk):
            s, img = seqs[start : start + chunk], images[start : start + chunk]
            ok = model.in_front(img @ normals.T - offsets, _ON_PLANE)  # (n, planes)
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


def _height_over_plane(model: TraceModel, points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """(n,) the distance (m) of each of ``points`` (n, 3) from the plane of ``planes``
    with the same row index, negative behind it (against its normal)."""
    normals = model.plane_normal[planes]
    return np.einsum("ij,ij->i", points, normals) - model.plane_offset[planes]


def _face_at(model: TraceModel, plane: int, point: np.ndarray, tolerance: float) -> int:
    """A face of ``plane`` that ``point`` lies on (edges included), or -1. Openings are
    holes: no point lies on one."""
    if abs(model.plane_normal[plane] @ point - model.plane_offset[plane]) > tolerance:
        return -1
    candidates = np.flatnonzero((model.face_plane == plane) & ~model.transparent)
    if len(candidates) == 0:
        return -1
    tri = model.room.mesh.triangles[candidates]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    area2 = np.einsum("ij,ij->i", n, n)
    # Barycentric coordinates of the point projected onto each triangle.
    w_a = np.einsum("ij,ij->i", np.cross(c - b, point - b), n) / area2
    w_b = np.einsum("ij,ij->i", np.cross(a - c, point - c), n) / area2
    w_c = 1.0 - w_a - w_b
    edge = np.sqrt(area2) / np.maximum(
        np.linalg.norm(np.stack([b - a, c - b, a - c]), axis=2).max(axis=0), 1e-300
    )  # twice the area over the longest edge: a height of the triangle
    slack = tolerance / np.maximum(edge, 1e-300)
    inside = np.flatnonzero(np.min(np.stack([w_a, w_b, w_c]), axis=0) >= -slack)
    return int(candidates[inside[0]]) if len(inside) else -1


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
            # A reflection point on an edge already lies on the next plane too: the path
            # reflects off both at the same point (a corner), so there is nothing to trace.
            corner = valid & (np.abs(_height_over_plane(model, current, plane)) <= tolerance)
            if j == k - 1:
                corner[:] = False
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
                    model, planes, before[rows], start[rows], current[rows], tolerance, counted
                )
                valid[rows] &= turns
                through[rows] *= share
                crossings[rows] += crossed
            before[trace], first[trace] = start[trace], j
            valid &= hit | corner
            for i in np.flatnonzero(corner):
                face[i] = _face_at(model, plane[i], current[i], tolerance)
            valid &= face >= 0
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
                model, planes, before[rows], current[rows], ends, tolerance, counted
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
        # Plane sequences that commute (perpendicular planes) give the same image.
        _, unique = np.unique(np.round(chain[-1][idx] / tolerance), axis=0, return_index=True)
        idx = idx[np.sort(unique)]
        parts.append(
            assemble_paths(
                model,
                source,
                listener,
                [p[idx] for p in points],
                faces[idx],
                distance[idx],
                through[idx],
                crossings[idx, 0],
                speed_of_sound=speed_of_sound,
                air=air,
                directivity=directivity,
                orientation=orientation,
                coherent=coherent,
                tolerance=tolerance,
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
    tolerance: float = 0.0,
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
    for j in range(k):
        incoming = points[j] - before
        energy *= model.reflection(faces[:, j], incoming, j + 1, impedance=True)
        if coherent or grounds:
            # At a corner the path reflects where it reflected last: it meets the
            # second plane at the angle it leaves it.
            after = points[j + 1] - points[j]
            corner = np.linalg.norm(incoming, axis=1) <= tolerance
            incoming = np.where(corner[:, None], after, incoming)
            normal = model.plane_normal[model.face_plane[faces[:, j]]]
            cosine = np.abs(np.einsum("ij,ij->i", normal, incoming)) / np.maximum(
                np.linalg.norm(incoming, axis=1), 1e-300
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
    :func:`_turn` says)."""
    n = len(start)
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
        for i in np.flatnonzero(~on_plane):
            other = _face_at(model, plane[rows[i]], position[rows[i]], tolerance)
            if other >= 0:
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


def _covers_both_sides(
    model: TraceModel, plane: int, point: np.ndarray, front: np.ndarray, tolerance: float
) -> bool:
    """Whether faces of ``plane`` cover it on both sides of ``point``, which lies in it,
    ``tolerance`` away along it and towards ``front`` (away from the surfaces a path
    reflects off at ``point``): the path goes through them there, as at the foot of a
    wall standing on the floor it reflects off, and not past their edge, as at the foot
    of an outside corner."""
    normal = model.plane_normal[plane]
    point = point - (normal @ point - model.plane_offset[plane]) * normal
    up = front - (front @ normal) * normal
    size = np.linalg.norm(up)
    if size <= 1e-300:
        return False
    up /= size
    along = np.cross(normal, up)
    near = model.bvh.min_distance
    return all(
        _face_at(model, plane, point + tolerance * (up + side * along), near) >= 0
        for side in (-1.0, 1.0)
    )


def _turn(model, planes, before, point, after, tolerance, counted):
    """Paths that reflect off ``planes`` (n, r; -1 for none; more than one in a corner)
    at ``point`` (n, 3), coming from ``before`` (n, 3) and going on to ``after`` (n, 3):
    (n,) whether each turns back off every one of those planes, as a reflection does,
    rather than going on across one (``before`` and ``after`` on either side of it,
    both farther than ``tolerance``); and the share of sound (n, bands) and the
    crossings (n, m) of the faces each crosses right at the point (see
    :func:`_crossed_at`). The leg leaving the point meets no face nearer to it than the
    intersection test's least distance; the one arriving meets every face on its way
    but one right where it meets the plane (:func:`_towards_plane`), and the point then
    lies in that face's plane."""
    n = len(point)
    turns = np.ones(n, dtype=bool)
    # One reflection turns back by itself; in a corner the path may go on across one.
    corner = np.flatnonzero(np.count_nonzero(planes >= 0, axis=1) > 1)
    for column in planes[corner].T:
        rows, plane = corner[column >= 0], column[column >= 0]
        turns[rows] &= ~_on_either_side(model, plane, before[rows], after[rows], tolerance)
    near = np.full((n, 2), model.bvh.min_distance)
    share, crossed, _, _ = _crossed_at(
        model, planes, before, point, after, tolerance, counted, near
    )
    return turns, share, crossed


def _on_either_side(model, plane, before, after, tolerance) -> np.ndarray:
    """(n,) whether ``before`` and ``after`` (n, 3) lie on either side of ``plane`` (n,),
    both farther from it than ``tolerance``."""
    below = _height_over_plane(model, before, plane)
    above = _height_over_plane(model, after, plane)
    far = np.minimum(np.abs(below), np.abs(above)) > tolerance
    return far & (below * above < 0.0)


def _crossed_at(model, planes, before, point, after, tolerance, counted, near):
    """For paths that reflect off ``planes`` (n, r; -1 for none) at ``point`` (n, 3),
    coming from ``before`` (n, 3) and going on to ``after`` (n, 3): the faces of other
    planes that each crosses right at the point, where neither leg's intersection test
    meets them, as a wall standing on the floor where the path reflects off the floor.
    Those are the faces whose plane ``before`` and ``after`` lie on either side of, both
    farther than ``tolerance``, that cover it on both sides of the point (see
    :func:`_covers_both_sides`), and that the leg crossing their plane meets within
    ``near`` (n, 2) of the point: the distance from its end within which the leg that
    arrives at the point meets no face, and from its start for the leg that leaves it.
    Returns the share of sound they let through (n, bands; 0 where one lets nothing
    through), how many of them each crosses of each column of ``counted`` (F, m), (n,
    m), the first of them on the way that is not an opening (n,), -1 for none, and
    whether the leg arriving at the point is the one that crosses it (n,)."""
    n = len(point)
    bands = model.absorption.shape[1]
    # A chord across the turn, just in front of the planes the path reflects off, finds
    # the faces there inside them, not on their edge as the path meets them.
    arriving = point - before
    arriving /= np.maximum(np.linalg.norm(arriving, axis=1), 1e-300)[:, None]
    leaving = after - point
    leaving /= np.maximum(np.linalg.norm(leaving, axis=1), 1e-300)[:, None]
    start = point - tolerance * arriving
    chord = point + tolerance * leaving - start
    length = np.linalg.norm(chord, axis=1)
    direction = chord / np.maximum(length, 1e-300)[:, None]
    share = np.ones((n, bands))
    crossed = np.zeros((n, counted.shape[1]), dtype=np.int64)
    first = np.full(n, -1, dtype=np.int64)
    first_arrives = np.zeros(n, dtype=bool)
    transparent = model.transparent
    skip = np.full(n, -1, dtype=np.int64)
    alive = np.arange(n)
    for _ in range(MAX_CROSSINGS + 1):
        if len(alive) == 0:
            break
        t, face = model.intersect(start[alive], direction[alive], length[alive], skip[alive])
        found = face >= 0
        rows, face, t = alive[found], face[found], t[found]
        plane = model.face_plane[face]
        here = _on_either_side(model, plane, before[rows], after[rows], tolerance)
        # The leg that crosses the plane: the one arriving where the point lies on the
        # side of ``after`` already.
        height = _height_over_plane(model, point[rows], plane)
        arrives = height * _height_over_plane(model, after[rows], plane) > 0.0
        leg = np.where(arrives[:, None], arriving[rows], leaving[rows])
        slope = np.abs(np.einsum("ij,ij->i", leg, model.plane_normal[plane]))
        here &= np.abs(height) <= np.where(arrives, near[rows, 0], near[rows, 1]) * slope
        here &= ~np.any(planes[rows] == plane[:, None], axis=1)
        for i in np.flatnonzero(here):
            front = leaving[rows[i]] - arriving[rows[i]]
            here[i] = _covers_both_sides(model, plane[i], point[rows[i]], front, tolerance)
        at, passed = rows[here], face[here]
        share[at] *= 0.0 if model.transmission is None else model.transmission[passed]
        crossed[at] += counted[passed]
        solid = (first[at] < 0) & ~transparent[passed]
        first[at[solid]] = passed[solid]
        first_arrives[at[solid]] = arrives[here][solid]
        start[rows] += direction[rows] * t[:, None]
        length[rows] -= t
        skip[rows] = face
        alive = rows
    return share, crossed, first, first_arrives
