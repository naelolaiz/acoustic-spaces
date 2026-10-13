"""acoustic_spaces: open-source room acoustics simulation and auralisation.

The package is organised as a pipeline:

- ``scene``: the scene model (geometry, materials, sources, listeners) and its YAML schema
- ``engine``: propagation methods writing a directional reflectogram (M2)
- ``spatial``: binaural, Ambisonics and loudspeaker encoders (M3)
- ``render``: offline and real-time convolution (M4)

Only the scene model and the compiled-core smoke test exist in this milestone.
"""

from __future__ import annotations

from acoustic_spaces._version import __version__
from acoustic_spaces.scene import Listener, Material, Scene, Source

__all__ = ["Listener", "Material", "Scene", "Source", "__version__", "native_available"]


def native_available() -> bool:
    """Return True when the compiled core (``acoustic_spaces._native``) can be imported."""
    try:
        from acoustic_spaces import _native  # noqa: F401
    except ImportError:
        return False
    return True
