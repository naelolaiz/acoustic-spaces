"""Triangle mesh with per-face labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoomMesh:
    """Triangles of a room model.

    ``labels`` holds, for each distinct combination of names found in the source file,
    the candidate names in priority order (for OBJ: material, group, object). Each face
    points at one entry through ``face_labels``.
    """

    vertices: np.ndarray  # (V, 3) float64, metres
    faces: np.ndarray  # (F, 3) int64 indices into vertices
    face_labels: np.ndarray  # (F,) int64 indices into labels
    labels: list[tuple[str, ...]]

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64).reshape(-1, 3)
        self.face_labels = np.ascontiguousarray(self.face_labels, dtype=np.int64).reshape(-1)
        if len(self.face_labels) != len(self.faces):
            raise ValueError("face_labels must have one entry per face")

    @property
    def triangles(self) -> np.ndarray:
        """(F, 3, 3) corner coordinates of every face."""
        return self.vertices[self.faces]

    def face_areas(self) -> np.ndarray:
        tri = self.triangles
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    def face_normals(self) -> np.ndarray:
        """Unit normals following the right-hand rule on the vertex order."""
        tri = self.triangles
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, length, out=np.zeros_like(n), where=length > 0)

    def cells(self):
        """The rooms and other cells the faces enclose
        (:class:`~acoustic_spaces.geometry.cells.MeshCells`), worked out once."""
        from acoustic_spaces.geometry.cells import mesh_cells

        cached = self.__dict__.get("_cells")
        if cached is None or cached[0] is not self.vertices or cached[1] is not self.faces:
            cached = (self.vertices, self.faces, mesh_cells(self.vertices, self.faces))
            self.__dict__["_cells"] = cached
        return cached[2]

    def label_names(self) -> list[str]:
        """Every distinct name that appears in any label, in first-seen order."""
        seen: dict[str, None] = {}
        for label in self.labels:
            for name in label:
                seen.setdefault(name, None)
        return list(seen)

    @staticmethod
    def concatenate(parts: list[RoomMesh]) -> RoomMesh:
        if not parts:
            raise ValueError("no geometry to combine")
        vertices, faces, face_labels = [], [], []
        index: dict[tuple[str, ...], int] = {}
        offset = 0
        for part in parts:
            remap = np.array([index.setdefault(lab, len(index)) for lab in part.labels])
            vertices.append(part.vertices)
            faces.append(part.faces + offset)
            face_labels.append(remap[part.face_labels] if len(remap) else part.face_labels)
            offset += len(part.vertices)
        labels = list(index)
        return RoomMesh(
            np.concatenate(vertices), np.concatenate(faces), np.concatenate(face_labels), labels
        )
