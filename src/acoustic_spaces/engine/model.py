"""The room as the propagation methods see it: a BVH, planes and per-face coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acoustic_spaces.room import UNASSIGNED, Room

#: Faces whose normals and offsets agree to this (unit normal components, metres) share
#: a plane, and so share image sources.
PLANE_TOLERANCE = 1e-5

#: Pressure pattern a + (1 - a) cos(theta) for each source directivity.
DIRECTIVITY_A = {
    "omni": 1.0,
    "subcardioid": 0.7,
    "cardioid": 0.5,
    "supercardioid": 0.37,
    "hypercardioid": 0.25,
    "figure8": 0.0,
}


def directivity_weight(a: float, forward: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Energy weight of a first-order pattern, averaging 1 over the sphere."""
    p = a + (1.0 - a) * (np.asarray(directions) @ np.asarray(forward))
    return p**2 / (a**2 + (1.0 - a) ** 2 / 3.0)


@dataclass
class TraceModel:
    room: Room
    bvh: object  # acoustic_spaces._native.Bvh
    #: (F,) plane index of every face.
    face_plane: np.ndarray
    #: (P, 3) unit plane normals, pointing into the room when ``oriented``.
    plane_normal: np.ndarray
    #: (P,) plane offsets: points x on plane p satisfy normal[p] . x == offset[p].
    plane_offset: np.ndarray
    #: True when the inside of the room is known for every face (closed, consistent model).
    oriented: bool
    #: (F, bands) random-incidence absorption and scattering coefficients.
    absorption: np.ndarray
    scattering: np.ndarray

    @classmethod
    def from_room(cls, room: Room) -> TraceModel:
        from acoustic_spaces import _native

        mesh = room.mesh
        if np.any(room.face_materials == UNASSIGNED):
            raise ValueError("every surface needs a material; see Room.problems()")
        bvh = _native.Bvh(mesh.vertices, mesh.faces)

        normals = mesh.face_normals()
        inward = room.report.normals_inward
        oriented = inward is not None
        if oriented:
            normals = normals if inward else -normals
        else:
            # Unknown inside: give each geometric plane one canonical normal direction.
            flip = np.take_along_axis(normals, np.abs(normals).argmax(axis=1)[:, None], 1) < 0
            normals = np.where(flip, -normals, normals)
        offsets = np.einsum("ij,ij->i", normals, mesh.triangles[:, 0])
        keys = np.round(np.column_stack([normals, offsets]) / PLANE_TOLERANCE).astype(np.int64)
        _, first, face_plane = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        face_plane = face_plane.reshape(-1)

        materials = room.scene.materials
        absorption = np.array([materials[n].absorption for n in room.material_names])
        scattering = np.array([materials[n].scattering for n in room.material_names])
        return cls(
            room=room,
            bvh=bvh,
            face_plane=face_plane,
            plane_normal=normals[first],
            plane_offset=offsets[first],
            oriented=oriented,
            absorption=np.ascontiguousarray(absorption[room.face_materials]),
            scattering=np.ascontiguousarray(scattering[room.face_materials]),
        )

    @property
    def plane_count(self) -> int:
        return len(self.plane_offset)

    def mirror(self, points: np.ndarray, planes: np.ndarray) -> np.ndarray:
        """Mirror each point across the plane with the same row index."""
        n = self.plane_normal[planes]
        distance = np.einsum("ij,ij->i", n, points) - self.plane_offset[planes]
        return points - 2.0 * distance[:, None] * n

    def intersect(
        self, origins: np.ndarray, directions: np.ndarray, t_max: np.ndarray, ignore: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.bvh.intersect(
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(directions, dtype=np.float64),
            np.ascontiguousarray(t_max, dtype=np.float64),
            np.ascontiguousarray(ignore, dtype=np.int64),
        )
