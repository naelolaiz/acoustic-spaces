"""Aircraft noise by ECAC Doc 29, *Report on Standard Method of Computing Noise
Contours around Civil Airports* (5th edition, endorsed 19 May 2026): the noise of a
single movement of an aircraft at receivers on the ground, from its NPD curves along a
flight path built of segments.

- :mod:`.tables` reads the aircraft, NPD curves, spectral classes and fixed-point
  profiles from tables laid out as the ANP database's (Volume 2 Appendix G);
- :mod:`.path` builds the flight path's segments (Volume 2 section 3.6);
- :mod:`.noise` gives each segment's sound exposure and maximum level and the event's
  (Volume 2 chapter 4);
- :mod:`.performance` gives an aircraft's fixed-point profile from its procedural
  steps, its engine and aerodynamic coefficients and its weight (Volume 2 Appendix B);
- :mod:`.volume3` holds Volume 3's three hypothetical reference aircraft: Part 1's
  routes, receptors and reference sound exposure levels, and Part 2's procedures,
  aerodromes and reference flight profiles;
- :mod:`.anp` reads a copy of the ANP database that the user has downloaded.

The ANP database's own aircraft are not part of this package: its terms (EASA's ANP
Database Terms and Conditions, issue 3, 2024) allow their use only for noise contours
of an average fleet and traffic and forbid passing them on. The readers of
:mod:`.tables`, :mod:`.performance` and :mod:`.anp` take tables with the fields Volume
2 Appendix G lists, as Volume 3's workbook lays them out; they have not been tried on a
download of the database.
"""

from __future__ import annotations

from . import anp, performance
from .noise import Event, SegmentLevels, event, impedance_db, segment_levels
from .path import Segment, Track, flight_path
from .tables import (
    Aircraft,
    NpdCurves,
    ProfilePoint,
    SpectralClass,
    read_aircraft,
    read_fixed_point_profiles,
    read_npd,
    read_spectral_classes,
)

__all__ = [
    "Aircraft",
    "Event",
    "NpdCurves",
    "ProfilePoint",
    "Segment",
    "SegmentLevels",
    "SpectralClass",
    "Track",
    "anp",
    "event",
    "flight_path",
    "impedance_db",
    "performance",
    "read_aircraft",
    "read_fixed_point_profiles",
    "read_npd",
    "read_spectral_classes",
    "segment_levels",
]
