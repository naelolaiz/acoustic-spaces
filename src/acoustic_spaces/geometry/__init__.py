"""Room geometry: CAD import, cleanup and checks.

Every importer returns a :class:`RoomMesh`: triangles in metres with +z up, and for
each triangle the names it carried in the source file (material, group, object or
layer) so the scene's ``assignments`` can map them to acoustic materials. Simple rooms
can also be built from dimensions (:func:`box_mesh`, :func:`extruded_mesh`).
"""

from acoustic_spaces.geometry.builders import box_mesh, extruded_mesh, plan_problem
from acoustic_spaces.geometry.cells import MeshCells, mesh_cells
from acoustic_spaces.geometry.checks import (
    MeshReport,
    check_mesh,
    clean_mesh,
    distance_to_surface,
    points_inside,
    room_of,
)
from acoustic_spaces.geometry.io import SUPPORTED_SUFFIXES, UNIT_SCALE, load_mesh
from acoustic_spaces.geometry.mesh import RoomMesh

__all__ = [
    "SUPPORTED_SUFFIXES",
    "UNIT_SCALE",
    "MeshCells",
    "MeshReport",
    "RoomMesh",
    "box_mesh",
    "check_mesh",
    "clean_mesh",
    "distance_to_surface",
    "extruded_mesh",
    "load_mesh",
    "mesh_cells",
    "plan_problem",
    "points_inside",
    "room_of",
]
