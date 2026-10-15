"""The room as the propagation methods see it: a BVH, planes and per-face coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_spaces.directivity import PATTERNS, Directivity
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.materials.bands import Resolution, band_centres
from acoustic_spaces.room import UNASSIGNED, Room

#: Faces whose normals and offsets agree to this (unit normal components, metres) share
#: a plane, and so share image sources.
PLANE_TOLERANCE = 1e-5

#: The lowest reflection order that adds the scattering of a surface's size, for each
#: edge diffraction setting (see :attr:`TraceModel.size_from_order`).
_SIZE_FROM_ORDER = {"none": 1, "direct": 2, "first_order": 3}

#: Pressure pattern a + (1 - a) cos(theta) for each named source directivity (the same
#: as :data:`acoustic_spaces.directivity.PATTERNS`).
DIRECTIVITY_A = PATTERNS

#: A straight line through more surfaces that let sound through than this counts as
#: blocked.
MAX_CROSSINGS = 64


def source_weights(
    directivity: Directivity | None,
    orientation: np.ndarray | None,
    directions: np.ndarray,
    bands: int = len(OCTAVE_BANDS_HZ),
) -> np.ndarray:
    """Energy weights (n, bands) of a source's directivity for unit vectors (n, 3) in
    room coordinates; ``orientation`` has the source's forward, left and up axes as
    columns (default: the room's). Without a directivity, ``bands`` ones."""
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    if directivity is None:
        return np.ones((len(directions), bands))
    local = directions if orientation is None else directions @ np.asarray(orientation)
    return directivity.energy(local)


@dataclass
class TraceModel:
    room: Room
    bvh: object  # acoustic_spaces._native.Bvh
    #: (F,) plane index of every face.
    face_plane: np.ndarray
    #: (P, 3) unit plane normals, pointing into the room when ``oriented`` (and the plane
    #: is not two-sided).
    plane_normal: np.ndarray
    #: (P,) plane offsets: points x on plane p satisfy normal[p] . x == offset[p].
    plane_offset: np.ndarray
    #: True when the rooms are known (a closed model): each face's normal then points
    #: into the room on its side.
    oriented: bool
    #: (F, bands) random-incidence absorption and scattering coefficients.
    absorption: np.ndarray
    scattering: np.ndarray
    #: (P,) planes with a room on both sides of some face: a partition between two
    #: rooms or a free-standing panel; every plane when the model is not ``oriented``.
    plane_two_sided: np.ndarray | None = None
    #: (F, bands) share of sound each face lets through (see
    #: :meth:`acoustic_spaces.scene.Material.transmission_coefficients`), or None.
    transmission: np.ndarray | None = None
    #: (F, 2) the room on the front (the side the mesh's face normal points to) and the
    #: back of each face (an index into ``room_volumes``), or -1.
    face_room: np.ndarray | None = None
    #: (R,) volume of each room (m3).
    room_volumes: np.ndarray | None = None
    #: (F, 2) width and length (m) of the surface each face belongs to (see
    #: :func:`surface_sizes`), when the scattering of a surface's size is added; else
    #: None.
    surface_size: np.ndarray | None = None
    #: Vector-based scattering (the scene's ``simulation.scattering.method``).
    vector_scattering: bool = True
    #: With vector-based scattering, each band traced with rays of its own where the
    #: bands scatter differently (``simulation.scattering.by_band``).
    by_band: bool = True
    #: The bands the coefficients (and so every energy traced) are in.
    bands: Resolution = "octave"
    #: Reflections of this order and higher add the scattering of the surface's size.
    #: Below it, the early stage's edge diffraction accounts for the size of surfaces:
    #: its terms add coherently to the direct sound and the reflections off the faces of
    #: an edge (with ``direct`` diffraction), and to the reflections of first-order
    #: reflections off them too (``first_order``), so those waves keep their full
    #: specular share.
    size_from_order: int = 1

    @classmethod
    def from_room(cls, room: Room, settings=None) -> TraceModel:
        """The model of ``room`` with the simulation ``settings`` (default: the scene's)."""
        from acoustic_spaces import _native

        mesh = room.mesh
        if np.any(room.face_materials == UNASSIGNED):
            raise ValueError("every surface needs a material; see Room.problems()")
        bvh = _native.Bvh(mesh.vertices, mesh.faces)

        normals = mesh.face_normals()
        oriented = len(room.air_cells) > 0
        face_room = room.face_room if oriented else np.full((len(mesh.faces), 2), -1, np.int64)
        air = face_room >= 0
        back = air[:, 1] & ~air[:, 0]
        # A room on both sides, or the inside unknown: one canonical normal direction for
        # the face's geometric plane.
        both = air[:, 0] == air[:, 1]
        flip = np.take_along_axis(normals, np.abs(normals).argmax(axis=1)[:, None], 1) < 0
        normals = np.where((back | (both & flip[:, 0]))[:, None], -normals, normals)
        offsets = np.einsum("ij,ij->i", normals, mesh.triangles[:, 0])
        keys = np.round(np.column_stack([normals, offsets]) / PLANE_TOLERANCE).astype(np.int64)
        _, first, face_plane = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        face_plane = face_plane.reshape(-1)
        two_sided = np.zeros(len(first), dtype=bool)
        np.logical_or.at(two_sided, face_plane, both)

        materials = room.scene.materials
        simulation = room.scene.simulation if settings is None else settings
        coefficients = [materials[n].coefficients(simulation.bands) for n in room.material_names]
        absorption = np.array([a for a, _ in coefficients])
        scattering = np.array([s for _, s in coefficients])
        size = None
        if simulation.scattering.from_size:
            size = np.ascontiguousarray(surface_sizes(mesh.vertices, mesh.faces, face_plane))
        return cls(
            room=room,
            bvh=bvh,
            face_plane=face_plane,
            plane_normal=normals[first],
            plane_offset=offsets[first],
            oriented=oriented,
            absorption=np.ascontiguousarray(absorption[room.face_materials]),
            scattering=np.ascontiguousarray(scattering[room.face_materials]),
            surface_size=size,
            vector_scattering=simulation.scattering.method == "vector",
            by_band=simulation.scattering.by_band,
            bands=simulation.bands,
            size_from_order=_SIZE_FROM_ORDER[simulation.early.diffraction],
            plane_two_sided=two_sided,
            transmission=room.face_transmission(simulation.bands),
            face_room=face_room,
            room_volumes=room.room_volumes if oriented else np.zeros(0),
        )

    def __post_init__(self) -> None:
        if self.plane_two_sided is None:
            self.plane_two_sided = np.full(self.plane_count, not self.oriented)
        if self.face_room is None:
            self.face_room = np.full((len(self.face_plane), 2), -1, dtype=np.int64)
        if self.room_volumes is None:
            self.room_volumes = np.zeros(0)
        if self.transmission is not None:
            self.transmission = np.ascontiguousarray(self.transmission, dtype=np.float64)

    @property
    def transparent(self) -> np.ndarray:
        """(F,) faces that let all sound through in every band: openings."""
        if self.transmission is None:
            return np.zeros(len(self.face_plane), dtype=bool)
        return np.all(self.transmission >= 1.0, axis=1)

    def in_front(self, distance: np.ndarray, tolerance: float) -> np.ndarray:
        """Whether points at signed ``distance`` (..., P) from every plane, farther than
        ``tolerance``, can see it: in front of it, or on either side of a two-sided
        plane."""
        return np.where(self.plane_two_sided, np.abs(distance) > tolerance, distance > tolerance)

    def passes(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        lengths: np.ndarray,
        ignore: np.ndarray,
        transparent_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(n, bands) share of sound that travels from each origin ``lengths`` along its
        direction (unit): the product of the transmission of the faces it crosses, 0
        where a face lets nothing through; and (n,) how many of those faces it crosses
        that are not openings. With ``transparent_only``, only openings let sound
        through."""
        origins = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.ascontiguousarray(directions, dtype=np.float64).reshape(-1, 3)
        n = len(origins)
        bands = self.absorption.shape[1]
        factor = np.ones((n, bands))
        crossings = np.zeros(n, dtype=np.int64)
        lengths = np.broadcast_to(np.asarray(lengths, dtype=np.float64), (n,)).copy()
        skip = np.broadcast_to(np.asarray(ignore, dtype=np.int64), (n,)).copy()
        start = origins.copy()
        alive = np.arange(n)
        transparent = self.transparent
        for _ in range(MAX_CROSSINGS + 1):
            if len(alive) == 0:
                break
            t, face = self.intersect(start[alive], directions[alive], lengths[alive], skip[alive])
            hit = face >= 0
            rows, face, t = alive[hit], face[hit], t[hit]
            if self.transmission is None:
                factor[rows] = 0.0
                break
            through = self.transmission[face]
            if transparent_only:
                through = np.where(transparent[face][:, None], through, 0.0)
            factor[rows] *= through
            crossings[rows] += ~transparent[face]
            start[rows] += directions[rows] * t[:, None]
            lengths[rows] -= t
            skip[rows] = face
            alive = rows[np.any(through > 0.0, axis=1)]
        else:
            factor[alive] = 0.0
        return factor, crossings

    @property
    def band_hz(self) -> tuple[int, ...]:
        """Centre frequencies (Hz) of the model's bands."""
        return band_centres(self.bands)

    @property
    def plane_count(self) -> int:
        return len(self.plane_offset)

    def mirror(self, points: np.ndarray, planes: np.ndarray) -> np.ndarray:
        """Mirror each point across the plane with the same row index."""
        n = self.plane_normal[planes]
        distance = np.einsum("ij,ij->i", n, points) - self.plane_offset[planes]
        return points - 2.0 * distance[:, None] * n

    def reflection(self, faces: np.ndarray, incoming: np.ndarray, order: int) -> np.ndarray:
        """(n, bands) specular energy share of reflections off ``faces`` (n,) of sound
        that travelled ``incoming`` (n, 3) from the source or the previous reflection,
        the ``order``-th reflection of its path: (1 - absorption) (1 - scattering), with
        the scattering of the surface's size when that applies, as the stochastic
        tracer reflects its rays."""
        faces = np.asarray(faces, dtype=np.int64)
        scattering = self.scattering[faces]
        if self.surface_size is not None and order >= self.size_from_order and len(faces):
            incoming = np.asarray(incoming, dtype=float).reshape(-1, 3)
            distance = np.linalg.norm(incoming, axis=1)
            normal = self.plane_normal[self.face_plane[faces]]
            cosine = np.abs(np.einsum("ij,ij->i", normal, incoming)) / np.maximum(distance, 1e-300)
            width, length = self.surface_size[faces].T
            scattering = size_scattering(
                width,
                length,
                distance,
                cosine,
                self.room.speed_of_sound(),
                scattering,
                self.band_hz,
            )
        return (1.0 - self.absorption[faces]) * (1.0 - scattering)

    def intersect(
        self, origins: np.ndarray, directions: np.ndarray, t_max: np.ndarray, ignore: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.bvh.intersect(
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(directions, dtype=np.float64),
            np.ascontiguousarray(t_max, dtype=np.float64),
            np.ascontiguousarray(ignore, dtype=np.int64),
        )


def size_scattering(
    width: np.ndarray,
    length: np.ndarray,
    distance: np.ndarray,
    cosine: np.ndarray,
    speed_of_sound: float,
    scattering: np.ndarray,
    frequencies=OCTAVE_BANDS_HZ,
) -> np.ndarray:
    """Scattering coefficients (n, bands) of reflections off surfaces ``width`` by
    ``length`` metres (n,), with what their limited size scatters added to their own
    ``scattering`` (n, bands) at the bands' centre ``frequencies`` (Hz), after
    Christensen and Rindel (2005), as in ODEON.

    Below f_w = c d / (2 (w cos)^2) the specular share falls as f / f_w, and below
    f_l = c d / (2 l^2) as f / f_l as well, d being the distance the sound travelled to
    the surface (from the source or the previous reflection) and cos the cosine of its
    angle of incidence. A large surface reflects specularly down to low frequencies; a
    small one, a distant one or one met at a grazing angle scatters.
    """
    across = np.asarray(width, float) * np.maximum(np.asarray(cosine, float), 1e-3)
    # c d, never 0: a reflection right where the last one was (a corner) is specular.
    reach = np.maximum(speed_of_sound * np.asarray(distance, float), 1e-300)[:, None]
    f = np.asarray(frequencies, float)
    specular = np.minimum(1.0, f * 2.0 * across[:, None] ** 2 / reach) * np.minimum(
        1.0, f * 2.0 * np.asarray(length, float)[:, None] ** 2 / reach
    )
    return 1.0 - (1.0 - scattering) * specular


def surface_sizes(vertices: np.ndarray, faces: np.ndarray, face_plane: np.ndarray) -> np.ndarray:
    """(F, 2) width and length (m) of the surface each face belongs to.

    A surface is the faces of one plane that share edges, whatever their materials (a
    window in a wall is part of the wall). Its width and length are those of the
    rectangle with its area and perimeter; a surface rounder than a square (whose
    perimeter is too short for any rectangle of its area) counts as a square.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(faces)
    if n == 0:
        return np.zeros((0, 2))
    # Edges as sorted vertex pairs, one row per face corner.
    pairs = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(n), 3)
    edge_ids, edge = np.unique(pairs, axis=0, return_inverse=True)
    edge = edge.reshape(-1)
    # Join faces of one plane across shared edges (label propagation with jumps).
    order = np.lexsort((face_plane[owner], edge))
    e, f = edge[order], owner[order]
    same = (e[1:] == e[:-1]) & (face_plane[f[1:]] == face_plane[f[:-1]])
    a, b = f[:-1][same], f[1:][same]
    label = np.arange(n)
    while len(a):
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    triangles = vertices[faces]
    face_area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    area = np.bincount(label, weights=face_area, minlength=n)
    # An edge is on a surface's boundary when only one of the surface's faces has it.
    key = edge.astype(np.int64) * n + label[owner]
    keys, count = np.unique(key, return_counts=True)
    boundary = keys[count == 1]
    edge_length = np.linalg.norm(vertices[edge_ids[:, 0]] - vertices[edge_ids[:, 1]], axis=1)
    perimeter = np.bincount(boundary % n, weights=edge_length[boundary // n], minlength=n)
    quarter = perimeter / 4.0
    spread = np.sqrt(np.maximum(quarter**2 - area, 0.0))
    side = np.sqrt(area)
    width = np.where(quarter**2 > area, quarter - spread, side)
    length = np.where(quarter**2 > area, quarter + spread, side)
    size = np.column_stack([width[label], length[label]])
    return np.maximum(size, 1e-6)
