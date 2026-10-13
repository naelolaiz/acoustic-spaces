"""Room geometry: CAD import, cleanup and checks.

Every importer returns a :class:`RoomMesh`: triangles in metres with +z up, and for
each triangle the names it carried in the source file (material, group, object or
layer) so the scene's ``assignments`` can map them to acoustic materials.
"""

from acoustic_spaces.geometry.checks import (
    MeshReport,
    check_mesh,
    clean_mesh,
    distance_to_surface,
    points_inside,
)
from acoustic_spaces.geometry.io import SUPPORTED_SUFFIXES, load_mesh
from acoustic_spaces.geometry.mesh import RoomMesh

__all__ = [
    "SUPPORTED_SUFFIXES",
    "MeshReport",
    "RoomMesh",
    "check_mesh",
    "clean_mesh",
    "distance_to_surface",
    "load_mesh",
    "points_inside",
]
