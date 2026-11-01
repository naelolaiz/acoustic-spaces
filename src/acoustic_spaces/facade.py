"""The sound outdoors heard in a room behind a closed window: a façade's level
difference by EN ISO 12354-3, for the pass-by demos' houses.

EN ISO 12354-3:2017 gives a façade's standardized level difference, from the level 2 m
in front of it to the room behind it at the reference reverberation time T0 = 0.5 s,
per band as

    D_2m,nT = R' + ΔL_fs + 10 lg(V / (6 T0 S)),

R' the façade's apparent sound reduction index, S its area as the room sees it, V the
room's volume, ΔL_fs what the façade's shape adds (none for a flat one). 2 m in front of
a façade the level is 3 dB over the free field's, its reflection added (ISO 1996-2:2017,
the correction for a microphone 0.5 to 2 m in front of a reflecting façade); a listener
of a scene without the house stands in that free field, so the room's level is

    L_room = L_free + 3 - D_2m,nT.

Here the closed window is the façade's only path: R' S is the window's R over its own
area (the area cancels), so the room hears L_free + 3 - R - 10 lg(V / (6 T0 S_window)).
The wall round it, a masonry one, lets through far less, and is left out; so are the
window's frame and seals and any vent, so the room is as quiet as the glazing lets it
be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.parameters import OCTAVE_BANDS_HZ

#: EN ISO 12354-3's reference reverberation time of a room (s): a furnished dwelling's.
T0_S = 0.5
#: How much louder the sound is 2 m in front of a reflecting façade than in the free
#: field (dB; ISO 1996-2:2017).
FACADE_REFLECTION_DB = 3.0
#: AGC Glass Europe's THERMOBEL 4-16-4, double glazing of two 4 mm panes 16 mm apart: its
#: sound reduction index per octave band from 125 Hz to 4 kHz (dB), measured by EN ISO
#: 10140 on a pane of 1.23 by 1.48 m, Rw (C; Ctr) = 30 (-1; -4) dB, "not better than
#: +/- 1 dB" (AGC, *Performances: Planibel, Stratobel, Stratophone, Thermobel*, 9000625
#: EN, 05/18).
GLAZING_R_DB = (20.5, 16.8, 25.7, 36.4, 41.4, 36.5)
#: The pane AGC measured (m): its width and height.
PANE_M = (1.23, 1.48)


def glazing_r_db() -> np.ndarray:
    """(8,) the double glazing's sound reduction index in the 8 octave bands, 63 Hz to
    8 kHz (dB): AGC's measured values (:data:`GLAZING_R_DB`); at 63 Hz the 125 Hz value
    less the mass law's 6 dB an octave, as below the panes' resonance on the air
    between them (some 210 Hz for this unit) they move as one; at 8 kHz the 4 kHz
    value, where the panes' coincidence dip is (near 3 kHz for 4 mm glass), above which
    the index rises again."""
    low = GLAZING_R_DB[0] + 20.0 * math.log10(OCTAVE_BANDS_HZ[0] / OCTAVE_BANDS_HZ[1])
    return np.array([low, *GLAZING_R_DB, GLAZING_R_DB[-1]])


@dataclass(frozen=True)
class Facade:
    """A room behind a façade whose closed window is its only way in: the window's sound
    reduction index per octave band (``reduction_db``, 8 bands from 63 Hz), its area
    and the room's volume, at the reference reverberation time :data:`T0_S`."""

    what: str
    reduction_db: tuple[float, ...]
    window_m2: float
    volume_m3: float

    def __post_init__(self) -> None:
        if len(self.reduction_db) != len(OCTAVE_BANDS_HZ):
            raise ValueError("give the window's sound reduction index in the 8 octave bands")
        if not (self.window_m2 > 0.0 and self.volume_m3 > 0.0):
            raise ValueError("the window's area and the room's volume are positive")

    @property
    def room_term_db(self) -> float:
        """EN ISO 12354-3's 10 lg(V / (6 T0 S)) of the room and the window (dB)."""
        return 10.0 * math.log10(self.volume_m3 / (6.0 * T0_S * self.window_m2))

    def level_difference(self) -> np.ndarray:
        """(8,) the standardized level difference D_2m,nT per octave band (dB)."""
        return np.asarray(self.reduction_db, dtype=float) + self.room_term_db

    def indoor_db(self) -> np.ndarray:
        """(8,) the room's level less the free field's outside, per octave band (dB):
        3 - D_2m,nT."""
        return FACADE_REFLECTION_DB - self.level_difference()

    def gains(self) -> np.ndarray:
        """(8,) the amplitude gain per octave band from the free field outside to the
        room."""
        return 10.0 ** (self.indoor_db() / 20.0)


#: The pass-by demos' house: a living room of 50 m³ (5 by 4 m, 2.5 m high) behind a
#: masonry wall with one window of the double glazing AGC measured, as large as its
#: pane (1.82 m²), closed.
HOUSE = Facade(
    "a living room of 50 m³ behind a closed window of 4-16-4 double glazing, 1.82 m²",
    tuple(round(float(v), 2) for v in glazing_r_db()),
    PANE_M[0] * PANE_M[1],
    50.0,
)


#: ISO 717-1's reference values for airborne sound insulation in octave bands, 125 Hz to
#: 2 kHz (dB), and its spectra No. 1 (pink noise, C) and No. 2 (urban traffic, Ctr), A-
#: weighted and normalised to 0 dB overall.
REFERENCE_DB = (36.0, 45.0, 52.0, 55.0, 56.0)
SPECTRUM_C_DB = (-21.0, -14.0, -8.0, -5.0, -4.0)
SPECTRUM_CTR_DB = (-14.0, -10.0, -7.0, -4.0, -6.0)


def weighted(reduction_db) -> tuple[int, int, int]:
    """ISO 717-1's single numbers of a sound reduction index in the five octave bands
    125 Hz to 2 kHz: (Rw, C, Ctr), the reference curve shifted in 1 dB steps as high as
    it goes with its unfavourable deviations summing to 10 dB at most, Rw its value at
    500 Hz, and the adaptation terms against the two spectra, rounded to whole dB."""
    r = np.asarray(reduction_db, dtype=float)
    if r.shape != (5,):
        raise ValueError("give the index in the five octave bands from 125 Hz to 2 kHz")
    reference = np.asarray(REFERENCE_DB)
    shift = math.floor(float(np.min(r - reference))) - 1
    while np.sum(np.maximum(reference + shift + 1 - r, 0.0)) <= 10.0 + 1e-9:
        shift += 1
    rw = int(round(reference[2] + shift))

    def term(spectrum) -> int:
        x = -10.0 * math.log10(float(np.sum(10.0 ** ((np.asarray(spectrum) - r) / 10.0))))
        return int(round(x - rw))

    return rw, term(SPECTRUM_C_DB), term(SPECTRUM_CTR_DB)


__all__ = [
    "FACADE_REFLECTION_DB",
    "GLAZING_R_DB",
    "HOUSE",
    "PANE_M",
    "REFERENCE_DB",
    "SPECTRUM_CTR_DB",
    "SPECTRUM_C_DB",
    "T0_S",
    "Facade",
    "glazing_r_db",
    "weighted",
]
