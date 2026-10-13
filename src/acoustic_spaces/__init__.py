"""acoustic_spaces: open-source room acoustics simulation and auralisation.

The package is organised as a pipeline:

- ``scene``: the scene model (geometry, materials, sources, listeners) and its YAML schema
- ``geometry``, ``materials``: CAD import and checks, the material database
- ``room``: a scene with its geometry loaded, the input of every simulation
- ``viz``: 3D view (optional, needs PyVista)
- ``engine``: propagation methods writing a directional reflectogram (M2)
- ``spatial``: binaural, Ambisonics and loudspeaker encoders (M3)
- ``render``: offline and real-time convolution (M4)
"""

from __future__ import annotations

from acoustic_spaces._version import __version__
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Listener, Material, Scene, Source

__all__ = ["Listener", "Material", "Room", "Scene", "Source", "__version__", "native_available"]


def native_available() -> bool:
    """Return True when the compiled core (``acoustic_spaces._native``) can be imported."""
    try:
        from acoustic_spaces import _native  # noqa: F401
    except ImportError:
        return False
    return True
