"""The vertical plane through a source and a listener outdoors.

What stands on the way between the two, seen from above
(:meth:`~acoustic_spaces.geometry.outdoor.Layout.obstacles_along`), which of the early
paths lie in that plane (the specular paths off the ground and the roofs alone, and the
paths over the tops of the barriers and buildings that cross it), and the ground along
it: what a calculation in two dimensions along the way holds, and so what it stands
for among the early paths. Shared by the weather's effect on the early paths
(:mod:`~acoustic_spaces.engine.refracted`) and by the solver for whole rows in still air
(:mod:`~acoustic_spaces.engine.row_paths`).
"""

from __future__ import annotations

import threading
from typing import cast

import numpy as np

from acoustic_spaces.engine.early import _EDGE_TOLERANCE, EarlyReflections
from acoustic_spaces.engine.model import TraceModel, _ground_faces
from acoustic_spaces.geometry.outdoor import GROUND, SKY, Layout, line_crossings

#: The grounds along the parabolic equation's way are looked up this often (m).
GROUND_STEP_M = 0.25
#: A listener this close (m) to a barrier's line or a building's wall, seen from above,
#: stands on it: the parabolic equation takes it past it, where nothing below its top
#: gets through, so the paths over it keep straight legs there.
ON_OBSTACLE_M = 1e-6
#: The kinds of the faces the parabolic equation's runs hold: barriers and buildings.
_LAID = frozenset({"barriers", "buildings"})
#: Marks, in ``EarlyReflections.edge``, the path that carries the sound of the solver for
#: whole rows (:mod:`~acoustic_spaces.engine.row_paths`).
ROW = -3
#: The labels of the ground and the sky.
_OPEN = frozenset({(GROUND,), (SKY,)})


class Section:
    """The vertical plane through the source at ``source`` and each listener of an
    outdoor scene (``model``): its obstacles, its paths and its ground. ``edges``: the
    model's diffracting edges, as the paths over them number them (found when needed if
    not given)."""

    def __init__(self, model: TraceModel, source: np.ndarray, edges=None) -> None:
        from acoustic_spaces.engine.diffraction import band_frequencies

        room = model.room
        if room.outdoor is None or model.ground is None:
            raise ValueError("the vertical plane through a source and a listener: outdoors")
        self.model = model
        self.source = np.asarray(source, dtype=float)
        self.ground = float(model.ground)
        self.zs = max(float(self.source[2]) - self.ground, 0.0)
        self.bands = model.absorption.shape[1]
        self.flat = _ground_faces(room)
        self.tolerance = _EDGE_TOLERANCE * float(np.ptp(room.mesh.vertices, axis=0).max())
        self.frequencies = band_frequencies(bands=model.bands)[0]  # (bands, M)
        self.zones = bool(room.outdoor.zones)
        self.layout: Layout = room.outdoor
        #: Frequencies and admittances at other frequencies than the early paths', by
        #: what asks for them (:meth:`_admittance_at`).
        self._dense: dict = {}
        self._admittances: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._objects: np.ndarray | None = None
        self._edges = edges
        self._laid: np.ndarray | None = None
        self._roofs: np.ndarray | None = None

    def _ground_face(self, points: np.ndarray) -> np.ndarray:
        """(n,) the face of the ground under each of ``points`` (n, 3), or -1."""
        start = np.asarray(points, dtype=float) + np.array([0.0, 0.0, 0.01])
        down = np.broadcast_to(np.array([0.0, 0.0, -1.0]), start.shape)
        _, face = self.model.intersect(
            start, down, np.full(len(start), 0.02), np.full(len(start), -1)
        )
        return np.where((face >= 0) & self.flat[np.maximum(face, 0)], face, -1)

    def _above(self, found) -> tuple:
        """The obstacles ``found`` ((near, far, top), as
        :meth:`~acoustic_spaces.geometry.outdoor.Layout.obstacles_along` gives them) with
        their tops above the ground, as :func:`~acoustic_spaces.engine.parabolic.march`
        takes them, to the nanometre (so that mirrored ways share their runs)."""
        return tuple(
            (round(near, 9), round(far, 9), round(top - self.ground, 9))
            for near, far, top in found
            if top > self.ground
        )

    def _laid_edges(self) -> np.ndarray:
        """(edges,) bool: whether each of the model's edges is a barrier's or a
        building's."""
        if self._laid is None:
            mesh = self.model.room.mesh
            laid = np.array([_kind(label) in _LAID for label in mesh.labels] or [False])
            faces = laid[mesh.face_labels]
            edge = self._edge_set().face
            self._laid = faces[edge[:, 0]] & faces[edge[:, 1]]
        return self._laid

    def _in_plane(self, paths: EarlyReflections, i: int, towards, reach: float, walls=None) -> bool:
        """Whether the path ``i`` of ``paths`` goes over edges that all cross the vertical
        plane through the source and the listener between them, none vertical, and
        reflects off nothing but the ground and the buildings' roofs (and the faces
        ``walls``, (faces,) bool, where given)."""
        over = cast(np.ndarray, paths.edge)
        if over[i] < 0:
            return False
        faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
        level = self.flat | self._roof_faces()
        if len(faces) and not np.all((level if walls is None else level | walls)[faces]):
            return False
        edges = self._edge_set()
        # Only the barriers' and buildings' edges: the parabolic equation's runs leave the
        # model file's objects out.
        laid = self._laid_edges()
        if not laid[over[i]]:
            return False
        if not self._across(edges, np.array([over[i]]), towards, reach)[0]:
            return False
        # Any other corner on an edge (the second of two) lies on one across the plane.
        tolerance = self.tolerance
        for corner in paths.points[i][1:-1]:
            on = _on_edges(edges, corner, tolerance)
            if (
                len(on)
                and over[i] not in on
                and not (self._across(edges, on, towards, reach) & laid[on]).any()
            ):
                return False
        return True

    def _roof_faces(self) -> np.ndarray:
        """(faces,) bool: the buildings' roofs, their faces facing straight up, which the
        parabolic equation's runs hold as rigid floors."""
        if self._roofs is None:
            mesh = self.model.room.mesh
            kinds = np.array([_kind(label) == "buildings" for label in mesh.labels] or [False])
            normal = self.model.plane_normal[self.model.face_plane]
            self._roofs = kinds[mesh.face_labels] & (normal[:, 2] >= 1.0 - 1e-9)
        return self._roofs

    def _roof_plane(self, plane: int) -> bool:
        """Whether the plane ``plane`` holds a building's roof."""
        faces = self.model.face_plane == plane
        return bool(np.any(self._roof_faces() & faces))

    def _across(self, edges, index: np.ndarray, towards, reach: float) -> np.ndarray:
        """Whether each of the edges ``index`` is not vertical and crosses the way from
        the source along ``towards`` (seen from above) short of ``reach``."""
        out = np.zeros(len(index), dtype=bool)
        for n, e in enumerate(index):
            if abs(float(edges.direction[e][2])) >= 1.0 - 1e-9:
                continue
            ends = np.array([edges.start[e][:2], edges.end[e][:2]])
            out[n] = bool(line_crossings(self.source[:2], towards, reach, ends, closed=False))
        return out

    def _edge_set(self):
        """The model's diffracting edges, as the paths over them number them."""
        if self._edges is None:
            from acoustic_spaces.engine.diffraction import find_edges

            self._edges = find_edges(self.model)
        return self._edges

    def _blocked(self, listener: np.ndarray) -> bool:
        """Whether a face of the model file's objects stands in the straight line from
        the source to ``listener``."""
        if self._objects is None:
            mesh = self.model.room.mesh
            laid = np.array(
                [_kind(label) in _LAID or tuple(label) in _OPEN for label in mesh.labels] or [True]
            )
            self._objects = ~laid[mesh.face_labels]
        return self._meets(listener, self._objects)

    def _meets(self, listener: np.ndarray, faces: np.ndarray) -> bool:
        """Whether the straight line from the source to ``listener`` meets one of the
        ``faces`` ((F,) bool), by the model's intersection test."""
        if not faces.any():
            return False
        start = self.source.copy()
        delta = np.asarray(listener, dtype=float) - start
        left = float(np.linalg.norm(delta))
        direction = delta / max(left, 1e-300)
        skip = -1
        for _ in range(64):
            t, face = self.model.intersect(
                start[None], direction[None], np.array([left]), np.array([skip])
            )
            if face[0] < 0:
                return False
            if faces[face[0]]:
                return True
            start = start + direction * float(t[0])
            left -= float(t[0])
            skip = int(face[0])
        return False

    def _admittance_at(self, face: int, band: int, frequencies: np.ndarray) -> np.ndarray:
        """(M,) the admittance of ``face`` as :meth:`_admittance` gives it, at other
        ``frequencies`` (Hz) of ``band`` than the early paths'."""
        index = (face, band, len(frequencies), float(frequencies[0]))
        with self._lock:
            if index not in self._dense:
                model = self.model
                if model.impedance_faces(np.array([face]))[0]:
                    row = int(cast(np.ndarray, model.face_impedance)[face])
                    grazing = (
                        model.impedance_extended is not None and model.impedance_extended.any()
                    )
                    value = model.impedances[row].admittance(
                        frequencies, model.room.speed_of_sound(), cosine=0.0 if grazing else None
                    )
                else:
                    from acoustic_spaces.engine.impedance import fit_admittance

                    value = fit_admittance(model.absorption[face], self.bands, None)(frequencies)
                self._dense[index] = np.asarray(value, dtype=complex)
            return self._dense[index]

    def _admittance(self, face: int) -> np.ndarray:
        """(bands, M) the normalised admittance (time as exp(j omega t)) of ``face``: an
        impedance's at grazing incidence (which differs with the extended reaction)."""
        with self._lock:
            if face not in self._admittances:
                model = self.model
                if model.impedance_faces(np.array([face]))[0]:
                    held = cast(np.ndarray, model.face_impedance)[face]
                    value = cast(np.ndarray, model.grazing_admittance)[held]
                else:
                    from acoustic_spaces.engine.impedance import fit_admittance

                    value = fit_admittance(model.absorption[face], self.bands, None)(
                        self.frequencies
                    )
                self._admittances[face] = np.asarray(value, dtype=complex)
            return self._admittances[face]

    def _here(self) -> int:
        """The face of the ground under the source (the first flat one off the ground)."""
        here = self._ground_face(np.array([[*self.source[:2], self.ground]]))[0]
        return int(here) if here >= 0 else int(np.flatnonzero(self.flat)[0])

    def _zone_faces(self, points: np.ndarray) -> np.ndarray:
        """The faces of the ground's zones at the horizontal ``points`` (n, 2), held to
        the layout's extent; the source's own (:meth:`_here`) outside every zone."""
        layout = self.layout
        x0, y0, x1, y1 = layout.extent
        parts = cast(np.ndarray, self.model.part_face)
        found = parts[layout.zone_at(np.clip(points, [x0, y0], [x1, y1])) + 1]
        return np.where(found < 0, self._here(), found)

    def _grounds_along(self, direction: np.ndarray, reach: float):
        """A function giving the face of the ground (its material) at a distance along
        ``direction`` from the source, up to ``reach``."""
        here = self._here()
        if not self.zones or self.model.ground_part is None:
            return lambda r: here
        # The grounds every GROUND_STEP_M along the way, looked up once.
        along = np.arange(0.0, reach + 2.0 * GROUND_STEP_M, GROUND_STEP_M)
        faces = self._zone_faces(self.source[:2] + along[:, None] * direction)
        last = len(faces) - 1
        return lambda r: int(faces[min(int(round(r / GROUND_STEP_M)), last)])


def _kind(label) -> str | None:
    """The kind of the faces of ``label``, its last part, as the outdoor geometry writes
    it (a building's or a barrier's own name, or a model file's, may be anything)."""
    return label[-1] if len(label) else None


def _on_edges(edges, point: np.ndarray, tolerance: float) -> np.ndarray:
    """The indices of the ``edges`` that ``point`` (3,) lies on, within ``tolerance``."""
    index = np.arange(len(edges.length))
    points = np.broadcast_to(point, (len(index), 3))
    return np.flatnonzero(_edge_distance(edges, index, points) <= tolerance)


def _edge_distance(edges, index: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n,) the distance (m) from each of ``points`` (n, 3) to the ``edges`` ``index``
    (n,), between their ends."""
    start = edges.start[index]
    direction = edges.direction[index]
    along = np.clip(np.einsum("ij,ij->i", points - start, direction), 0.0, edges.length[index])
    return np.linalg.norm(points - start - direction * along[:, None], axis=1)


def _at(cut: tuple, reach: float) -> bool:
    """Whether a barrier or a building's wall of ``cut`` ((near, far, top)) stands within
    ``ON_OBSTACLE_M`` of ``reach`` m along the way: the march there gives a listener the
    field past it, none below its top."""
    return any(min(abs(near - reach), abs(far - reach)) <= ON_OBSTACLE_M for near, far, _ in cut)
