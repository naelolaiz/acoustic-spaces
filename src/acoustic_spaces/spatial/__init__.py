"""Spatial outputs: binaural, Ambisonics and loudspeaker impulse responses.

Every format turns the arrival directions of a simulated response into channels:

- :class:`Ambisonics`: AmbiX (ACN channel order, SN3D normalisation), orders 0 to 7;
- :class:`Loudspeakers`: vector base amplitude panning onto a named layout (stereo, 5.0,
  7.0, 5.0.4, 7.0.4) or any list of speakers;
- :class:`Binaural`: the two ears, with HRTFs from a SOFA file or a spherical head model;
- :class:`Mono`: the omnidirectional response.

Early reflections keep their exact direction. The late part comes from the arrival
directions the stochastic ray tracer records (or from all around, for the statistical
tail). :func:`spatial_impulse_response` (also ``Response.spatial_impulse_response``)
renders a response in a format.
"""

from acoustic_spaces.spatial.ambisonics import MAX_ORDER, Ambisonics, real_spherical_harmonics
from acoustic_spaces.spatial.directions import fibonacci_sphere, from_angles, to_angles, to_head
from acoustic_spaces.spatial.hrtf import HRTF, read_sofa, spherical_head
from acoustic_spaces.spatial.loudspeakers import LAYOUTS, Loudspeakers, Speaker
from acoustic_spaces.spatial.render import (
    Binaural,
    Format,
    Mono,
    from_output,
    spatial_impulse_response,
)

__all__ = [
    "HRTF",
    "LAYOUTS",
    "MAX_ORDER",
    "Ambisonics",
    "Binaural",
    "Format",
    "Loudspeakers",
    "Mono",
    "Speaker",
    "fibonacci_sphere",
    "from_angles",
    "from_output",
    "read_sofa",
    "real_spherical_harmonics",
    "spatial_impulse_response",
    "spherical_head",
    "to_angles",
    "to_head",
]
