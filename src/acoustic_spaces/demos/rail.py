"""The trains and trams of the pass-by demos: their sound power as Schall 03 gives it,
and a sound made to it. Nothing of the simulation uses it; a demo gives each train's
sources their levels and plays their sound.

Schall 03 (2014), Germany's method for the noise of railways and tramways (Anlage 2 of
the 16. BImSchV, Bundesgesetzblatt 2014 I, p. 2269), gives every vehicle unit of a
category Fz (:data:`CATEGORIES`: a locomotive, a coach, a wagon, a multiple unit, a
tram) its sources m (:data:`SOURCES`, :data:`TRAM_SOURCES`): rolling noise from the
rails' and the wheels' roughness at the rail top, aerodynamic noise at the pantograph
(5 m above the rail top), the roof (4 m) and the bogies (0 m), and the noise of its
fans and of its traction. Each source's A-weighted sound power per metre of track, with
one unit an hour going by at v km/h, in the octave band f from 63 Hz to 8 kHz, is
(equation 1):

    L_W'A,f = a_A + Δa_f + 10 lg(n / n_0) + b_f lg(v / 100) + c1_f

with a_A the source's A-weighted level and Δa_f its share in band f, both at 100 km/h
on sleepers in ballast with rails in average condition (the data sheets, Beiblatt 1 for
trains, 2 for trams); n / n_0 the unit's axles over its category's (rolling noise
alone); b_f the speed factor of the source's kind (:data:`SPEED_FACTORS`, Tabelle 6, and
:data:`TRAM_SPEED_FACTORS`, Tabelle 14); and c1 the corrections for other tracks
(:data:`TRACKS`, Tabelle 7 and 15). Left out: the corrections c2 for rails ground or
damped (Tabelle 8), K for bridges, curves and squeal, and the rule that has tramways
reckoned at 50 km/h at least where switches, junctions and stops make them louder: a
unit here goes at its speed on plain track. The octave shares Δa_f are whole dB: a
row's eight bands add up to its a_A within 0.5 dB.

Every source radiates into the whole sphere, louder across the track than along it
(equation 8):

    D_I = 10 lg(0.22 + 1.27 sin² δ) dB

with δ the angle between the ray and the track (:func:`directivity_db`), and the
ground's reflection added apart (equation 9), as a simulation does. D_I averages
10 lg(0.22 + 1.27 · 2/3) = +0.28 dB over the sphere, where a simulation's directivity
averages 0 dB (it only shares the sound power out between directions): each source
gets DIRECTIVITY_DB more sound power in the scene and the pattern
(:func:`directivity`) as much less, to be as loud every way as Schall 03 has it.

A unit going by at v once an hour leaves on each metre of track the energy its source
sends while it crosses that metre, so the unit's own sound power is the level per metre
and hour plus 10 lg(3600 s · v / 1 m) = 10 lg(1000 v) dB, v in km/h (:func:`unit_power`).

A train here is units one behind the other (:class:`Unit`, :data:`TRAINS`), each unit's
sources at points along it (:func:`train_points`): those at the rail top at its bogies,
evenly from 15 % to 85 % of its length (a coach's two bogies 18.5 m apart, as a UIC
coach's pivots are), those on the roof (4 m) over its cars, and the pantograph (5 m) at
its middle. The units' lengths and the points' places are assumptions; Schall 03 gives
only each unit's sound power per metre of track.

:func:`point_sound` makes each point's sound, dry: noise with the octave spectrum of its
sources. Its time structure (steady noise) is made up; its spectrum per octave band is
Schall 03's.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces.demos import Option, indoors, pipeline
from acoustic_spaces.demos.audio import read, resampled
from acoustic_spaces.demos.audio import write as write_wav
from acoustic_spaces.demos.passby import (
    FADE_S,
    Trail,
    Vehicle,
    car_input,
    car_of,
    faded,
    forced_speed,
    kmh,
    pass_s,
    speed_of,
)
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ


@dataclass(frozen=True)
class Row:
    """A source m of a data sheet: its A-weighted level a_A (dB, per metre of track with
    one unit an hour at 100 km/h) and its octave shares Δa_f, 63 Hz to 8 kHz (dB)."""

    m: int
    a_a_db: float
    shares_db: tuple[int, ...]


@dataclass(frozen=True)
class Category:
    """A vehicle category Fz of Schall 03 (Tabelle 3 and 12) and its data sheet
    (Beiblatt 1 and 2): its name in English and in the method's German, its reference
    number of axles n_0, its rolling noise for each kind of brakes (the brakes roughen
    the wheels: cast-iron blocks most), and its other sources."""

    key: int
    name: str
    german: str
    axles: int
    rolling: dict[str, tuple[Row, ...]]
    others: tuple[Row, ...]

    @property
    def tram(self) -> bool:
        """Whether it is a tramway's (Fz 21 to 23)."""
        return self.key >= 21

    def rows(self, brakes: str = "") -> tuple[Row, ...]:
        """Its sources with ``brakes`` (the first kind its sheet has, if empty)."""
        kinds = list(self.rolling)
        brakes = brakes or kinds[0]
        if brakes not in self.rolling:
            raise ValueError(
                f"Schall 03's Fz {self.key} ({self.name}) has brakes {', '.join(kinds)}, "
                f"not {brakes!r}"
            )
        return self.rolling[brakes] + self.others


#: Schall 03's vehicle categories and their data sheets (Beiblatt 1 for trains, Fz 1 to
#: 10; Beiblatt 2 for trams, Fz 21 to 23). Brakes: "cast_iron_blocks", "composite_blocks",
#: "axle_discs", "wheel_discs" and, for high-speed trains, the one kind their sheet has.
#: Freight wagons' sources 3 and 4 (4 m up) are tank wagons' bodies, which ring with the
#: rolling noise.
CATEGORIES: dict[int, Category] = {
    1: Category(
        1,
        "high-speed power car",
        "HGV-Triebkopf",
        4,
        {
            "": (
                Row(1, 62.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 51.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(5, 43.0, (-30, -21, -13, -9, -6, -4, -9, -17)),
            Row(6, 46.0, (-28, -21, -12, -9, -6, -4, -9, -17)),
            Row(7, 35.0, (-15, -8, -6, -6, -8, -14, -21, -32)),
            Row(8, 62.0, (-35, -24, -10, -5, -5, -8, -15, -26)),
            Row(9, 54.0, (-30, -22, -5, -4, -7, -11, -17, -26)),
            Row(11, 50.0, (-32, -24, -5, -4, -8, -12, -18, -29)),
        ),
    ),
    2: Category(
        2,
        "high-speed intermediate or driving trailer",
        "HGV-Mittel-/Steuerwagen",
        4,
        {
            "axle_discs_absorbers": (
                Row(1, 62.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 51.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(6, 29.0, (-21, -18, -15, -12, -5, -4, -10, -18)),
            Row(7, 35.0, (-15, -8, -6, -6, -8, -14, -21, -32)),
            Row(8, 44.0, (-35, -24, -13, -4, -5, -7, -14, -25)),
        ),
    ),
    3: Category(
        3,
        "high-speed multiple unit",
        "HGV-Triebzug",
        32,
        {
            "axle_and_wheel_discs": (
                Row(1, 73.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 62.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(5, 41.0, (-30, -21, -13, -9, -6, -4, -9, -17)),
            Row(6, 44.0, (-27, -21, -12, -8, -5, -5, -11, -19)),
            Row(7, 45.0, (-16, -9, -7, -7, -7, -9, -12, -19)),
            Row(8, 56.0, (-35, -24, -13, -4, -5, -7, -14, -25)),
            Row(9, 62.0, (-35, -24, -10, -5, -5, -8, -15, -26)),
            Row(11, 53.0, (-32, -24, -5, -4, -8, -12, -18, -29)),
        ),
    ),
    4: Category(
        4,
        "high-speed tilting train",
        "HGV-Neigezug",
        28,
        {
            "axle_discs_absorbers": (
                Row(1, 72.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 61.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(5, 41.0, (-30, -21, -13, -9, -6, -4, -9, -17)),
            Row(6, 47.0, (-28, -21, -12, -8, -5, -5, -11, -19)),
            Row(7, 44.0, (-16, -9, -7, -7, -7, -9, -12, -19)),
            Row(8, 52.0, (-35, -24, -13, -4, -5, -7, -14, -25)),
            Row(9, 59.0, (-35, -24, -10, -5, -5, -8, -15, -26)),
            Row(11, 49.0, (-32, -24, -5, -4, -8, -12, -18, -29)),
        ),
    ),
    5: Category(
        5,
        "electric multiple unit or S-Bahn",
        "E-Triebzug und S-Bahn",
        10,
        {
            "axle_discs": (
                Row(1, 71.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 60.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
            "wheel_discs": (
                Row(1, 69.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 58.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(5, 43.0, (-30, -21, -13, -9, -6, -4, -9, -17)),
            Row(6, 44.0, (-29, -22, -11, -7, -5, -5, -12, -20)),
            Row(7, 44.0, (-16, -9, -6, -6, -7, -11, -15, -22)),
            Row(8, 48.0, (-35, -24, -13, -4, -5, -7, -14, -25)),
            Row(9, 55.0, (-35, -24, -10, -5, -5, -8, -15, -26)),
            Row(11, 45.0, (-32, -24, -5, -4, -8, -12, -18, -29)),
        ),
    ),
    6: Category(
        6,
        "diesel multiple unit",
        "V-Triebzug",
        6,
        {
            "axle_discs": (
                Row(1, 69.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 58.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(6, 32.0, (-21, -18, -15, -12, -5, -4, -10, -18)),
            Row(7, 38.0, (-16, -9, -7, -7, -7, -9, -13, -20)),
            Row(8, 47.0, (-35, -24, -13, -4, -5, -7, -14, -25)),
            Row(9, 55.0, (-44, -17, -10, -5, -5, -7, -13, -20)),
            Row(10, 42.0, (-12, -5, -4, -8, -12, -20, -30, -30)),
            Row(11, 57.0, (-25, -16, -9, -5, -5, -8, -12, -20)),
        ),
    ),
    7: Category(
        7,
        "electric locomotive",
        "E-Lok",
        4,
        {
            "cast_iron_blocks": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 71.0, (-40, -30, -22, -9, -3, -5, -15, -26)),
            ),
            "discs": (
                Row(1, 66.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 55.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(5, 43.0, (-30, -21, -13, -9, -6, -4, -9, -17)),
            Row(6, 49.0, (-29, -22, -12, -8, -5, -5, -10, -18)),
            Row(7, 40.0, (-15, -8, -6, -6, -8, -14, -21, -32)),
            Row(8, 61.0, (-28, -19, -6, -4, -6, -10, -14, -23)),
            Row(9, 54.0, (-30, -22, -5, -4, -7, -11, -17, -26)),
            Row(11, 50.0, (-32, -24, -5, -4, -8, -12, -18, -29)),
        ),
    ),
    8: Category(
        8,
        "diesel locomotive",
        "V-Lok",
        4,
        {
            "cast_iron_blocks": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 71.0, (-40, -30, -22, -9, -3, -5, -15, -26)),
            ),
        },
        (
            Row(6, 40.0, (-24, -20, -14, -13, -6, -4, -7, -14)),
            Row(7, 40.0, (-15, -8, -6, -6, -8, -14, -21, -32)),
            Row(8, 60.0, (-44, -17, -10, -5, -5, -7, -13, -20)),
            Row(10, 47.0, (-12, -5, -4, -8, -12, -20, -30, -30)),
            Row(11, 62.0, (-25, -16, -9, -5, -5, -8, -12, -20)),
        ),
    ),
    9: Category(
        9,
        "passenger coach",
        "Reisezugwagen",
        4,
        {
            "cast_iron_blocks": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 71.0, (-40, -30, -22, -9, -3, -5, -15, -26)),
            ),
            "axle_discs": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 56.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (
            Row(6, 29.0, (-21, -18, -15, -12, -5, -4, -10, -18)),
            Row(7, 40.0, (-15, -8, -6, -6, -8, -14, -21, -32)),
            Row(8, 44.0, (-35, -24, -13, -4, -5, -7, -14, -25)),
        ),
    ),
    10: Category(
        10,
        "freight wagon",
        "Güterwagen",
        4,
        {
            "cast_iron_blocks": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 71.0, (-40, -30, -22, -9, -3, -5, -15, -26)),
                Row(3, 57.0, (-29, -20, -19, -6, -5, -5, -17, -26)),
                Row(4, 61.0, (-28, -19, -18, -5, -4, -7, -17, -26)),
            ),
            "composite_blocks": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 58.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
                Row(3, 57.0, (-29, -20, -19, -6, -5, -5, -17, -26)),
                Row(4, 48.0, (-28, -19, -18, -5, -4, -7, -17, -26)),
            ),
            "axle_discs": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 56.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
                Row(3, 57.0, (-29, -20, -19, -6, -5, -5, -17, -26)),
                Row(4, 46.0, (-28, -19, -18, -5, -4, -7, -17, -26)),
            ),
            "wheel_discs": (
                Row(1, 67.0, (-50, -40, -24, -8, -3, -6, -11, -30)),
                Row(2, 61.0, (-50, -40, -25, -9, -4, -4, -11, -23)),
            ),
        },
        (Row(7, 40.0, (-15, -8, -6, -6, -8, -14, -21, -32)),),
    ),
    21: Category(
        21,
        "low-floor tram",
        "Straßenbahn-Niederflurfahrzeug",
        8,
        {
            "": (
                Row(1, 63.0, (-34, -25, -20, -10, -2, -7, -12, -20)),
                Row(2, 63.0, (-34, -25, -20, -10, -2, -7, -12, -20)),
            ),
        },
        (Row(4, 39.0, (-26, -15, -11, -8, -5, -6, -10, -11)),),
    ),
    22: Category(
        22,
        "high-floor tram",
        "Straßenbahn-Hochflurfahrzeug",
        8,
        {
            "": (
                Row(1, 63.0, (-32, -23, -17, -11, -2, -7, -12, -19)),
                Row(2, 63.0, (-32, -23, -17, -11, -2, -7, -12, -19)),
            ),
        },
        (Row(3, 39.0, (-26, -15, -11, -8, -5, -6, -10, -11)),),
    ),
    23: Category(
        23,
        "metro car",
        "U-Bahn-Fahrzeug",
        8,
        {
            "": (
                Row(1, 60.0, (-34, -25, -13, -9, -4, -6, -10, -17)),
                Row(2, 60.0, (-34, -25, -13, -9, -4, -6, -10, -17)),
            ),
        },
        (Row(3, 39.0, (-26, -15, -11, -8, -5, -6, -10, -11)),),
    ),
}


@dataclass(frozen=True)
class Source:
    """A source m of Schall 03 (Tabelle 5 for trains, 13 for trams): what it is, how
    high above the rail top (m) and its kind, which sets its speed factor."""

    what: str
    height_m: float
    kind: str


#: Tabelle 5: the sources of a train.
SOURCES: dict[int, Source] = {
    1: Source("rolling noise from the rails' roughness", 0.0, "rolling"),
    2: Source("rolling noise from the wheels' roughness", 0.0, "rolling"),
    3: Source("a tank wagon's body ringing with the rails' roughness", 4.0, "rolling"),
    4: Source("a tank wagon's body ringing with the wheels' roughness", 4.0, "rolling"),
    5: Source("the pantograph's head", 5.0, "aerodynamic"),
    6: Source("the pantograph's foot and the grilles of the roof's fans", 4.0, "aerodynamic"),
    7: Source("the air round the bogies", 0.0, "aerodynamic"),
    8: Source("the fans of the cooling and air conditioning on the roof", 4.0, "fans"),
    9: Source("the fans of the cooling and air conditioning under the floor", 0.0, "fans"),
    10: Source("the exhaust", 4.0, "traction"),
    11: Source("the motors and the gearboxes", 0.0, "traction"),
}
#: Tabelle 13: the sources of a tram (3 under the floor of high-floor trams and metro
#: cars, 4 on the roof of low-floor trams).
TRAM_SOURCES: dict[int, Source] = {
    1: Source("running noise from the rails' roughness", 0.0, "rolling"),
    2: Source(
        "running noise from the wheels' roughness, the motors and the gearboxes", 0.0, "rolling"
    ),
    3: Source("converters, compressors and air conditioning under the floor", 0.0, "fans"),
    4: Source("converters, compressors and air conditioning on the roof", 4.0, "fans"),
}
#: Tabelle 6: the speed factor b of a train's sources of each kind, per octave band.
SPEED_FACTORS: dict[str, tuple[float, ...]] = {
    "rolling": (-5.0, -5.0, -5.0, 0.0, 10.0, 25.0, 25.0, 25.0),
    "aerodynamic": (50.0,) * 8,
    "fans": (-10.0,) * 8,
    "traction": (20.0,) * 8,
}
#: Tabelle 14: the speed factor b of a tram's sources: running noise of low- and
#: high-floor trams (Fz 21, 22), of metro cars (Fz 23), and the fans.
TRAM_SPEED_FACTORS: dict[str, tuple[float, ...]] = {
    "rolling": (0.0, 0.0, -5.0, 5.0, 20.0, 15.0, 15.0, 20.0),
    "metro": (15.0, 10.0, 20.0, 20.0, 30.0, 25.0, 25.0, 20.0),
    "fans": (-10.0,) * 8,
}
#: The speed the data sheets hold (km/h).
REFERENCE_SPEED_KMH = 100.0
#: How much a low-floor tram's roof source (m 4) is louder with air conditioning (dB;
#: Beiblatt 2, Fz 21).
AIR_CONDITIONING_DB = 8.0


@dataclass(frozen=True)
class Track:
    """A kind of track and Schall 03's corrections c1 for it (dB per octave band): to
    the rolling noise (sources 1 and 2), and, on a train's track, to every source at the
    rail top (1, 2, 7, 9 and 11) for the sound it reflects other than ballast does."""

    what: str
    rolling_db: tuple[float, ...] = (0.0,) * 8
    reflection_db: tuple[float, ...] = (0.0,) * 8


#: Tabelle 7: the tracks of trains. Sleepers in ballast are the data sheets' own.
TRACKS: dict[str, Track] = {
    "ballast": Track("sleepers in ballast"),
    "slab": Track("slab track", (0.0, 0.0, 0.0, 7.0, 3.0, 0.0, 0.0, 0.0), (1.0,) * 8),
    "slab_absorber": Track(
        "slab track with absorbers",
        (0.0, 0.0, 0.0, 7.0, 3.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, -2.0, -2.0, -3.0, 0.0, 0.0),
    ),
    "level_crossing": Track(
        "a level crossing", (0.0, 0.0, 0.0, 8.0, 4.0, 0.0, 0.0, 0.0), (1.0,) * 8
    ),
}
#: Tabelle 15: the tracks of trams, against sleepers in ballast ("street" is also a slab
#: track's, as Schall 03 has the two alike).
TRAM_TRACKS: dict[str, Track] = {
    "ballast": Track("sleepers in ballast"),
    "street": Track("track set in the street", (2.0, 3.0, 2.0, 5.0, 8.0, 4.0, 2.0, 1.0)),
    "grass_low": Track(
        "grassed track, the grass low", (-2.0, -4.0, -3.0, -1.0, -1.0, -1.0, -1.0, -3.0)
    ),
    "grass_high": Track(
        "grassed track, the grass up to the rail top",
        (1.0, -1.0, -3.0, -4.0, -4.0, -7.0, -7.0, -5.0),
    ),
}


def _source(category: Category, m: int) -> Source:
    return (TRAM_SOURCES if category.tram else SOURCES)[m]


def _factors(category: Category, kind: str) -> tuple[float, ...]:
    if not category.tram:
        return SPEED_FACTORS[kind]
    if kind == "rolling" and category.key == 23:
        return TRAM_SPEED_FACTORS["metro"]
    return TRAM_SPEED_FACTORS[kind]


@dataclass(frozen=True)
class Unit:
    """A vehicle unit of a train: its category Fz (:data:`CATEGORIES`), its brakes, its
    axles (its category's reference number unless given), its length (m), and for a
    freight wagon whether it is a tank wagon, for a low-floor tram whether it has air
    conditioning."""

    category: int
    length_m: float
    brakes: str = ""
    axles: int | None = None
    tank: bool = False
    air_conditioning: bool = False

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Schall 03 has no vehicle category Fz {self.category}; it has "
                f"{', '.join(str(k) for k in CATEGORIES)}"
            )
        CATEGORIES[self.category].rows(self.brakes)
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError(
                f"a unit's length is a positive number of metres, not {self.length_m!r}"
            )
        if self.axles is not None and self.axles < 1:
            raise ValueError(f"a unit has an axle at least, not {self.axles}")

    @property
    def kind(self) -> Category:
        return CATEGORIES[self.category]

    @property
    def axle_count(self) -> int:
        return self.kind.axles if self.axles is None else int(self.axles)

    def line_levels_a(self, speed_kmh: float, track: str = "ballast") -> dict[int, np.ndarray]:
        """Each of its sources m: L_W'A,f (dB(A) per octave band, 63 Hz to 8 kHz), the
        A-weighted sound power per metre of track with one unit an hour going by at
        ``speed_kmh`` on ``track`` (:data:`TRACKS`, or :data:`TRAM_TRACKS` for a tram):
        equation 1."""
        kind = self.kind
        tracks = TRAM_TRACKS if kind.tram else TRACKS
        if track not in tracks:
            raise ValueError(
                f"Schall 03 has tracks {', '.join(tracks)} for "
                f"{'trams' if kind.tram else 'trains'}, not {track!r}"
            )
        if not math.isfinite(speed_kmh) or speed_kmh <= 0.0:
            raise ValueError(f"a unit's speed is a positive number of km/h, not {speed_kmh!r}")
        lay = tracks[track]
        out = {}
        for row in kind.rows(self.brakes):
            source = _source(kind, row.m)
            if source.kind == "rolling" and row.m in (3, 4) and not kind.tram and not self.tank:
                continue
            level = row.a_a_db + np.asarray(row.shares_db, dtype=float)
            level += np.asarray(_factors(kind, source.kind)) * math.log10(
                speed_kmh / REFERENCE_SPEED_KMH
            )
            if row.m in (1, 2):
                level += 10.0 * math.log10(self.axle_count / kind.axles)
                level += np.asarray(lay.rolling_db)
            if not kind.tram and row.m in (1, 2, 7, 9, 11):
                level += np.asarray(lay.reflection_db)
            if kind.key == 21 and row.m == 4 and self.air_conditioning:
                level += AIR_CONDITIONING_DB
            out[row.m] = level
        return out


#: How much a source's sound power in the scene is over Schall 03's level, as its
#: directivity D_I averages that much over the sphere (dB).
DIRECTIVITY_DB = 10.0 * math.log10(0.22 + 1.27 * 2.0 / 3.0)


def directivity_db(angle_deg) -> np.ndarray:
    """Schall 03's directivity D_I (equation 8) at ``angle_deg`` between the ray and the
    track (dB)."""
    s = np.sin(np.radians(np.asarray(angle_deg, dtype=float)))
    return 10.0 * np.log10(0.22 + 1.27 * s**2)


def directivity():
    """A train's directivity (:class:`~acoustic_spaces.directivity.Balloon`, octave
    bands, alike in all): :func:`directivity_db` round the way it goes, which the
    source faces."""
    from acoustic_spaces.directivity import GRID_STEP_DEG, Balloon, _from_angles, _grid

    elevation, azimuth = _grid(GRID_STEP_DEG)
    ways = _from_angles(azimuth[None, :], elevation[:, None])
    along = np.degrees(np.arccos(np.clip(ways[..., 0], -1.0, 1.0)))
    energy = 10.0 ** (directivity_db(along) / 10.0)
    table = np.repeat(energy[..., None], len(OCTAVE_BANDS_HZ), axis=2)
    return Balloon.from_table(table, "train")


def unit_power(level_a_db: np.ndarray, speed_kmh: float) -> np.ndarray:
    """A moving unit's sound power per octave band (dB re 1 pW, not weighted) from
    Schall 03's level per metre and hour ``level_a_db`` (A-weighted, per band) at
    ``speed_kmh``, with DIRECTIVITY_DB."""
    from acoustic_spaces.parameters import A_WEIGHTING_DB

    return (
        np.asarray(level_a_db, dtype=float)
        + 10.0 * math.log10(1000.0 * speed_kmh)
        - np.asarray(A_WEIGHTING_DB)
        + DIRECTIVITY_DB
    )


@dataclass(frozen=True)
class Train:
    """A train: its units from the front, and what it is in a few words."""

    title: str
    units: tuple[Unit, ...]
    speed_kmh: float = field(default=100.0)
    #: What it is in a few words, for the stills ("A regional train").
    short: str = ""
    #: The base curve of the ground's vibration the FTA's general method gives this kind
    #: of train (:data:`acoustic_spaces.vibration.CURVES`; the manual's table 6-9):
    #: locomotive-hauled trains the locomotive-powered curve, multiple units and trams
    #: the rapid transit one.
    curve: str = "locomotive"

    @property
    def length_m(self) -> float:
        return float(sum(unit.length_m for unit in self.units))


#: Some of Schall 03's trains (Tabelle 4) and a tram, with their usual top speeds. The
#: units' lengths are assumptions: an electric locomotive 19.0 m (a Vectron's 18.98 m),
#: a coach 26.4 m (a UIC type X coach's), a freight wagon of four axles 17.8 m (0.225
#: axles a metre, the wagons the EU's rule for the noise of rolling stock is normalised
#: to: TSI NOI, Regulation (EU) No 1304/2014), the S-Bahn a class 423 multiple unit of
#: 67.4 m and ten axles, the tram 30.0 m.
TRAINS: dict[str, Train] = {
    "s_bahn": Train(
        "an S-Bahn: a four-car electric multiple unit",
        (Unit(5, 67.4, "wheel_discs"),),
        120.0,
        "An S-Bahn",
        "rapid_transit",
    ),
    "regional": Train(
        "a regional train: an electric locomotive and five coaches",
        (Unit(7, 19.0, "discs"), *(Unit(9, 26.4, "axle_discs"),) * 5),
        160.0,
        "A regional train",
    ),
    "intercity": Train(
        "an intercity train: an electric locomotive and twelve coaches",
        (Unit(7, 19.0, "discs"), *(Unit(9, 26.4, "axle_discs"),) * 12),
        200.0,
        "An intercity train",
    ),
    "freight": Train(
        "a freight train: an electric locomotive and 24 wagons with composite blocks",
        (Unit(7, 19.0, "discs"), *(Unit(10, 17.8, "composite_blocks"),) * 24),
        100.0,
        "A freight train",
    ),
    "tram": Train(
        "a low-floor tram with air conditioning",
        (Unit(21, 30.0, air_conditioning=True),),
        50.0,
        "A low-floor tram",
        "rapid_transit",
    ),
}
#: Where along a unit its first and last bogie are (shares of its length).
BOGIES_AT = (0.15, 0.85)


@dataclass(frozen=True)
class Point:
    """A point of a train where some of its sources are: ``behind_m`` behind its front,
    ``height_m`` above the rail top, of unit ``unit`` (from the front), with the sound
    power of its share of those sources per octave band (dB re 1 pW, not weighted, with
    DIRECTIVITY_DB)."""

    behind_m: float
    height_m: float
    unit: int
    power_db: np.ndarray = field(repr=False)


def _places(unit: Unit, height_m: float) -> np.ndarray:
    """Where along ``unit`` (shares of its length) its sources at ``height_m`` are: at
    its bogies at the rail top (two, or one each two axles), over its cars on the roof
    (one each two bogies), the pantograph in its middle."""
    bogies = max(2, unit.axle_count // 2)
    if height_m == 0.0:
        return np.linspace(*BOGIES_AT, bogies)
    if height_m >= 5.0:
        return np.array([0.5])
    cars = max(1, bogies // 2)
    return (np.arange(cars) + 0.5) / cars


def train_points(train: Train, speed_kmh: float, track: str = "ballast") -> list[Point]:
    """The points of ``train`` going at ``speed_kmh`` on ``track``, each unit's sources
    at each height shared out evenly between its points there: by height, then from the
    front."""
    points: list[Point] = []
    front = 0.0
    for i, unit in enumerate(train.units):
        heights: dict[float, np.ndarray] = {}
        for m, level in unit.line_levels_a(speed_kmh, track).items():
            height = _source(unit.kind, m).height_m
            heights[height] = heights.get(height, 0.0) + 10.0 ** (level / 10.0)
        for height, energy in heights.items():
            places = _places(unit, height)
            share = 10.0 * np.log10(energy / len(places))
            power = unit_power(share, speed_kmh)
            points.extend(Point(front + float(p) * unit.length_m, height, i, power) for p in places)
        front += unit.length_m
    return sorted(points, key=lambda p: (p.height_m, p.behind_m))


def train_power_db(points: list[Point]) -> np.ndarray:
    """(8,) the sound power of ``points`` together, per octave band (dB re 1 pW)."""
    return 10.0 * np.log10(sum(10.0 ** (p.power_db / 10.0) for p in points))


def point_sound(
    power_db: np.ndarray,
    sample_rate: int,
    duration_s: float,
    *,
    reference_db: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """A point's sound, dry and mono, ``duration_s`` long at ``sample_rate``: noise with
    the octave spectrum of ``power_db``, at an RMS of SOUND_RMS for a point of
    ``reference_db`` (``power_db`` itself unless given), so that the points of a train
    keep their sound powers against one another."""
    from acoustic_spaces.demos.road import SOUND_RMS, _to_bands

    if duration_s <= 0.0:
        raise ValueError("a train's sound needs a positive duration")
    n = int(round(duration_s * sample_rate))
    noise = np.random.default_rng(seed).standard_normal(n)
    x = _to_bands(noise, sample_rate, np.asarray(power_db, dtype=float))
    reference = power_db if reference_db is None else reference_db
    share = np.sum(10.0 ** (np.asarray(power_db, dtype=float) / 10.0)) / np.sum(
        10.0 ** (np.asarray(reference, dtype=float) / 10.0)
    )
    # The bands overlap at their edges: the RMS is set on the whole.
    return x * (SOUND_RMS * math.sqrt(float(share)) / float(np.sqrt(np.mean(x**2))))


# --------------------------------------------------------------------------- the demo


#: How each height's sources are named after the train's source, and what they are.
PARTS: dict[float, tuple[str, str]] = {
    0.0: ("", "its wheels, bogies, motors and the fans under its floor, at the rail top"),
    4.0: ("_roof", "its roof: the fans and the pantograph's foot, 4 m up"),
    5.0: ("_pantograph", "its pantograph's head, 5 m up"),
}
#: What ``--horn`` lets a train's horn do: stay off, sound its notes in blasts one after
#: the other, or sound them together, as a chord, in the same blasts.
HORN_PLAYS = ("off", "blasts", "chord")


class TrainVehicle(Vehicle):
    """A train (or a tram) going past in a pass-by demo: the example's source of its name
    at the rail top and, along the same walk, the sources of its name with ``_roof`` (4
    m up) and ``_pantograph`` (5 m up) (:data:`PARTS`), each facing the way it goes, with
    Schall 03's directivity. Each plays its points (:func:`train_points`): the train's
    front point at that height along its walk, the others behind it as far as they are
    behind the front (:class:`~acoustic_spaces.demos.passby.Trail`), each with its own
    sound (:func:`point_sound`) at its own sound power. The train is the option of the
    vehicle's name (one of ``trains``, :data:`TRAINS`), on ``track`` (the option
    ``--track`` when there are ``tracks`` to choose). With a ``horn``, ``--horn
    blasts`` sounds it too: the source of its name with ``_horn``, at its front
    TRAIN_HORN_HEIGHT_M above the rail top along the same walk
    (:data:`~acoustic_spaces.demos.horn.TRAIN_HORN`, with the measured siren's
    directivity, facing the way it goes), off by default; ``--horn chord`` sounds its
    notes together (:data:`~acoustic_spaces.demos.horn.TRAIN_CHORD`)."""

    demo_name = "passby"
    go, goes, going, way = "run", "runs", "passing", "the track"
    sound_step = "sound"

    def __init__(
        self,
        name: str,
        noun: str,
        trains: tuple[str, ...],
        track: str,
        tracks: tuple[str, ...],
        script: str,
        horn: bool = False,
    ) -> None:
        from acoustic_spaces.demos.horn import TRAIN_CHORD, TRAIN_HORN, TRAIN_HORN_HEIGHT_M
        from acoustic_spaces.demos.siren import Horn

        self.name, self.noun, self.trains = name, noun, trains
        #: Its horn (:class:`~acoustic_spaces.demos.siren.Horn`), or None.
        self.horn = (
            Horn(
                f"{name}_horn",
                f"{name}_horn.wav",
                f"at its front, {TRAIN_HORN_HEIGHT_M:g} m above the rail top",
                f"{TRAIN_HORN.on_axis_db:g} dB(C) {TRAIN_HORN.at_m:g} m ahead (EN 15153-2's most)",
                kind=TRAIN_HORN,
                chord=TRAIN_CHORD,
            )
            if horn
            else None
        )
        self.track, self.tracks, self.script = track, tracks, script
        self.title = TRAINS[trains[0]].title if len(trains) == 1 else f"a {noun}"
        self.steady = f"Schall 03 gives a {noun}'s sound power at one steady speed"
        self.sound_file, self.info_file = f"{name}.wav", f"{name}.json"
        self.own_speed = len(trains) > 1

    def heights(self) -> dict[str, float]:
        """Its sources, by name, and their heights above the rail top (m)."""
        return {self.name + suffix: height for height, (suffix, _) in PARTS.items()}

    def options(self) -> list[Option]:
        out = []
        if len(self.trains) > 1:
            out.append(
                Option(
                    self.name,
                    "the train that runs past: "
                    + "; ".join(f"{key}, {TRAINS[key].title}" for key in self.trains),
                    self.trains[0],
                    "choice",
                    self.trains,
                    steps=(self.sound_step, "scene", "walk"),
                )
            )
        if len(self.tracks) > 1:
            out.append(
                Option(
                    "track",
                    "the track it runs on, for Schall 03's corrections to its sound power: "
                    + "; ".join(f"{key}, {TRACKS[key].what}" for key in self.tracks),
                    self.track,
                    "choice",
                    self.tracks,
                    steps=(self.sound_step, "scene", "walk"),
                )
            )
        if self.horn is not None:
            from acoustic_spaces.demos.horn import TRAIN_HORN, TRAIN_HORN_HEIGHT_M

            kind = TRAIN_HORN
            high, low = kind.hz
            out.append(
                Option(
                    "horn",
                    f"the {self.noun}'s warning horn: off; blasts, its high note "
                    f"({high:g} Hz) and its low note ({low:g} Hz) in turn, "
                    f"{kind.blast_s:g} s each, every {kind.every_s:g} s, each "
                    f"{kind.on_axis_db:g} dB(C) {kind.at_m:g} m ahead (the most EN "
                    "15153-2 allows); or chord, the two notes together in the same "
                    f"blasts, {kind.on_axis_db:g} dB(C) together (EN 15153-2 allows it); "
                    f"from its front {TRAIN_HORN_HEIGHT_M:g} m above the rail top",
                    HORN_PLAYS[0],
                    "choice",
                    HORN_PLAYS,
                    steps=(self.sound_step, "scene", "walk"),
                )
            )
        out.append(
            Option(
                "indoors",
                indoors.help_of(self.noun),
                indoors.OFF,
                "choice",
                indoors.choices(self.noun),
                steps=("walk", "video"),
            )
        )
        return out

    def house(self, args: argparse.Namespace) -> indoors.House | None:
        """The house at each listener ``--indoors`` asks for, or None outdoors."""
        try:
            return indoors.House.of(getattr(args, "indoors", None), self.noun)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from None

    def wheels(self, args: argparse.Namespace) -> indoors.Wheels:
        """What shakes the ground: the train's base curve at its speed, and its wheels,
        its points at the rail top (its bogies); a freight train's wagons
        FREIGHT_WAGONS_DB under the curve, its locomotive on it."""
        from acoustic_spaces import vibration

        info = pipeline.read_json(args.work / self.name / self.info_file)
        train = TRAINS[info["train"]]
        points = info["parts"][self.name]
        wagons = vibration.FREIGHT_WAGONS_DB if info["train"] == "freight" else 0.0
        return indoors.Wheels(
            train.curve,
            float(info["speed_kmh"]),
            tuple(float(p["behind_m"]) for p in points),
            tuple(0.0 if p["unit"] == 0 else wagons for p in points),
        )

    def horn_on(self, args: argparse.Namespace) -> bool:
        """Whether its horn sounds: ``--horn blasts`` or ``--horn chord``."""
        return self.horn is not None and getattr(args, "horn", None) in HORN_PLAYS[1:]

    def horn_of(self, args: argparse.Namespace) -> str:
        """How its horn sounds (one of :data:`HORN_PLAYS`): ``--horn``, or off."""
        return str(args.horn) if self.horn_on(args) else HORN_PLAYS[0]

    def train_of(self, args: argparse.Namespace) -> str:
        """Which train runs: the option of its name, or its first."""
        return getattr(args, self.name, None) or self.trains[0]

    def own_speed_kmh(self, args: argparse.Namespace) -> float | None:
        """The train's own speed (:data:`TRAINS`), whatever its walk's, when there are
        trains to choose; one alone (the tram) goes as its walk goes."""
        return TRAINS[self.train_of(args)].speed_kmh if self.own_speed else None

    def track_of(self, args: argparse.Namespace) -> str:
        """The track it runs on: ``--track``, or its own."""
        return getattr(args, "track", None) or self.track

    def check(self, args: argparse.Namespace) -> None:
        if self.train_of(args) not in self.trains:
            raise SystemExit(
                f"error: --{self.name} is one of {', '.join(self.trains)}, "
                f"not {self.train_of(args)!r}"
            )
        if self.track_of(args) not in (self.tracks or (self.track,)):
            raise SystemExit(
                f"error: --track is one of {', '.join(self.tracks)}, not {self.track_of(args)!r}"
            )
        horn = getattr(args, "horn", None)
        if self.horn is not None and horn is not None and horn not in HORN_PLAYS:
            raise SystemExit(f"error: --horn is one of {', '.join(HORN_PLAYS)}, not {horn!r}")
        self.house(args)

    def _parts_of(self, keys: tuple[str, ...]) -> tuple[str, ...]:
        """Its sources that the trains ``keys`` have points at."""
        there = {
            p.height_m
            for key in keys
            for p in train_points(TRAINS[key], REFERENCE_SPEED_KMH, self.track)
        }
        return tuple(part for part, height in self.heights().items() if height in there)

    def parts(self, args: argparse.Namespace) -> tuple[str, ...]:
        horn = (self.horn.part,) if self.horn is not None and self.horn_on(args) else ()
        return (*self._parts_of((self.train_of(args),)), *horn)

    def every_part(self) -> tuple[str, ...]:
        horn = (self.horn.part,) if self.horn is not None else ()
        return (*self._parts_of(self.trains), *horn)

    def info(self, args: argparse.Namespace, scene, speed_kmh: float) -> dict:
        """What its sound is made of, at ``speed_kmh`` in ``scene``: the train, the
        track, the speed, the pass's length and rate, each of its sources' points (how
        far behind the front, of which unit, their sound power per octave band), and
        the sound power of the whole train; with its horn on, the horn's sound power,
        averaged over its blasts and pauses."""
        from acoustic_spaces.parameters import a_weighted

        key, track = self.train_of(args), self.track_of(args)
        train = TRAINS[key]
        points = train_points(train, speed_kmh, track)
        parts = {}
        for part, height in self.heights().items():
            mine = [p for p in points if p.height_m == height]
            if mine:
                parts[part] = [
                    {
                        "behind_m": round(p.behind_m, 4),
                        "unit": p.unit,
                        "level_db": [round(float(v), 2) for v in p.power_db],
                    }
                    for p in mine
                ]
        total = train_power_db(points)
        rate = int(scene.simulation.sample_rate)
        horn: dict = {}
        if self.horn is not None:
            horn["horn"] = self.horn_of(args)
            if self.horn_on(args):
                level = self.horn.power(horn["horn"], rate)
                horn["horn_level_db"] = [round(float(v), 1) for v in level]
                horn["horn_sound_power_a_db"] = round(a_weighted(level), 2)
        return {
            "train": key,
            "track": track,
            "speed_kmh": round(speed_kmh, 6),
            "rate": rate,
            "duration_s": round(pass_s(car_of(scene, self), forced_speed(args, self)), 6),
            "length_m": round(train.length_m, 4),
            "units": len(train.units),
            "parts": parts,
            "sound_power_db": round(float(10.0 * np.log10(np.sum(10.0 ** (total / 10.0)))), 2),
            "sound_power_a_db": round(a_weighted(total), 2),
            **horn,
        }

    def levels(self, info: dict) -> dict[str, list[float]]:
        own = {part: points[0]["level_db"] for part, points in info["parts"].items()}
        if self.horn is not None and "horn_level_db" in info:
            own[self.horn.part] = info["horn_level_db"]
        return own

    def edit(self, source: dict, info: dict, args: argparse.Namespace) -> None:
        """Its front point's sound power at that height, and Schall 03's directivity in a
        CSV table beside the demo's scene, facing the way it goes; its horn's, the
        measured siren's."""
        from acoustic_spaces.directivity import write_directivity_csv

        super().edit(source, info, args)
        if self.horn is not None and source["name"] == self.horn.part:
            from acoustic_spaces.demos import siren

            table = args.scene.with_name(f"{args.scene.stem}_{self.horn.part}.csv")
            write_directivity_csv(
                table, siren.directivity(), siren.directivity_comment(self.script)
            )
            named = Path(os.path.relpath(table, args.source.parent)).as_posix()
            source["directivity"] = {"file": named}
            source.setdefault("walk", {})["face_forward"] = True
            return
        table = args.scene.with_name(f"{args.scene.stem}_{self.name}.csv")
        comment = (
            f"A {self.noun}'s directivity by Schall 03 (equation 8), 10 lg(0.22 + 1.27 sin² δ) "
            "dB round the track,\nless its mean over the sphere; written by " + self.script
        )
        write_directivity_csv(table, directivity(), comment)
        named = Path(os.path.relpath(table, args.source.parent)).as_posix()
        source["directivity"] = {"file": named}
        source.setdefault("walk", {})["face_forward"] = True

    def sound_files(self, args: argparse.Namespace) -> tuple[str, ...]:
        return tuple(f"{part}.wav" for part in self.parts(args))

    def sound_path(self, args: argparse.Namespace, part: str) -> Path:
        return args.work / self.name / f"{part}.wav"

    def made(self, args: argparse.Namespace, rate: int, duration_s: float, part: str = ""):
        """The front point's made sound at the height of ``part``."""
        return self._sounds(args, rate, part or self.name, first=True)

    def _sounds(self, args: argparse.Namespace, rate: int, part: str, first: bool = False):
        path = self.sound_path(args, part)
        if not path.exists():
            return None
        x, r = read(path)
        x = x[:, None] if x.ndim == 1 else x
        columns = [resampled(x[:, k], r, rate) for k in range(1 if first else x.shape[1])]
        return columns[0] if first else columns

    def make(self, args: argparse.Namespace, info: dict, folder: Path) -> str:
        rate, duration = info["rate"], info["duration_s"]
        said = []
        for i, (part, points) in enumerate(info["parts"].items()):
            reference = np.asarray(points[0]["level_db"])
            columns = [
                point_sound(
                    np.asarray(p["level_db"]),
                    rate,
                    duration,
                    reference_db=reference,
                    seed=1000 * i + k,
                )
                for k, p in enumerate(points)
            ]
            write_wav(folder / f"{part}.wav", np.column_stack(columns), rate)
            said.append(f"{len(points)} at {self.heights()[part]:g} m")
        out = (
            f"{duration:.1f} s of {TRAINS[info['train']].title} at "
            f"{kmh(info['speed_kmh'])} km/h, its points {', '.join(said)}, "
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power"
        )
        if self.horn is not None and "horn_level_db" in info:
            plays = info.get("horn", HORN_PLAYS[1])
            write_wav(folder / self.horn.file, self.horn.sound(plays, rate, duration), rate)
            chord = " of its chord" if plays == "chord" else ""
            out += (
                f"; its horn's blasts{chord}, {info['horn_sound_power_a_db']:.1f} dB(A) of "
                "sound power on average"
            )
        return out

    def trail(self, args: argparse.Namespace, part: str, played, rate: int) -> Trail:
        """Its points at the height of ``part`` behind the front one, each its made sound
        (or, for a recording ``--input`` gives the part, the recording itself, each from
        another place in it), faded as the front one's, as far behind as it is; its
        horn, at its front, nothing."""
        if self.horn is not None and part == self.horn.part:
            return Trail()
        info = pipeline.read_json(args.work / self.name / self.info_file)
        points = info["parts"][part]
        speed = info["speed_kmh"] / 3.6
        first = np.asarray(points[0]["level_db"], dtype=float)
        n = len(played)
        if car_input(args, self, part) is None:
            columns = self._sounds(args, rate, part)
            if columns is None or len(columns) != len(points):
                raise SystemExit(f"error: {self.sound_path(args, part)} is not this train's")
            sounds = [faded(c[:n], rate) for c in columns[1:]]
        else:
            # A recording: each point plays it from its own place on, round to its start.
            raw = np.asarray(played, dtype=float)
            sounds = []
            for k, p in enumerate(points[1:], start=1):
                share = np.sum(10.0 ** (np.asarray(p["level_db"]) / 10.0)) / np.sum(
                    10.0 ** (first / 10.0)
                )
                sounds.append(np.roll(raw, k * n // len(points)) * math.sqrt(float(share)))
        behind = tuple(
            (
                (p["behind_m"] - points[0]["behind_m"]) / speed,
                y,
                np.asarray(p["level_db"], dtype=float) - first,
            )
            for p, y in zip(points[1:], sounds, strict=True)
        )
        return Trail(points[0]["behind_m"] / speed, behind)

    def fade_s(self, part: str) -> float:
        """Its horn's blasts rise and fall of themselves (the first is not faded in);
        its other sources fade as a pass's do."""
        if self.horn is not None and part == self.horn.part:
            return self.horn.fade_s
        return FADE_S

    def sounding(
        self, args: argparse.Namespace, part: str, played, rate: int, at: np.ndarray
    ) -> np.ndarray | None:
        """Its horn is heard around each place as loud as its blasts play there, as the
        fire engine's (:meth:`~acoustic_spaces.demos.siren.EmergencyVehicle.sounding`);
        its other sources as loud all along."""
        from acoustic_spaces.demos.passby import blocks_loudness

        if self.horn is None or part != self.horn.part:
            return None
        return blocks_loudness(np.asarray(played, dtype=float), rate, at, self.horn.rms)

    def input_gain(
        self, args: argparse.Namespace, part: str, x: np.ndarray, rate: int
    ) -> float | None:
        if self.horn is not None and part == self.horn.part:
            return self.horn.input_gain(x, rate, self.horn_of(args))
        return super().input_gain(args, part, x, rate)

    def plays(self, args: argparse.Namespace, speed: str) -> str:
        return f"{TRAINS[self.train_of(args)].title} {speed}made to Schall 03's sound power"

    def feeds(self, args: argparse.Namespace, speed: str) -> dict[str, str]:
        out = {}
        for part in self.parts(args):
            if self.horn is not None and part == self.horn.part:
                chord = ", its notes together" if self.horn_of(args) == "chord" else ""
                out[part] = f"its horn {self.horn.place}, in blasts{chord}"
                continue
            what = PARTS[self.heights()[part]][1]
            out[part] = f"{self.plays(args, speed)}: {what}" if part == self.name else f"{what}"
        return out

    def played_by(self, args: argparse.Namespace) -> str:
        speed = f"{round(speed_of(args, self), 4):g} km/h"
        horn = {"off": "", "blasts": " horn", "chord": " horn chord"}[self.horn_of(args)]
        return f"{self.name} {self.train_of(args)} {speed} {self.track_of(args)}{horn}"

    def facts(self, args: argparse.Namespace, info: dict) -> dict:
        train = TRAINS[info["train"]]
        lay = (TRAM_TRACKS if train.units[0].kind.tram else TRACKS)[info["track"]].what
        about = (
            f"{train.short}, {info['length_m']:.0f} m long, at "
            f"{kmh(info['speed_kmh'])} km/h on {lay},\n"
            f"{info['sound_power_a_db']:.1f} dB(A) of sound power (Schall 03)"
        )
        horn = {}
        if "horn" in info:
            horn["horn"] = info["horn"]
        if "horn_sound_power_a_db" in info:
            horn["horn_sound_power_a_db"] = info["horn_sound_power_a_db"]
            chord = "'s chord" if info.get("horn") == "chord" else ""
            about += f", its horn{chord} in blasts {info['horn_sound_power_a_db']:.1f} dB(A)"
        return {
            "train": info["train"],
            "track": info["track"],
            "length_m": info["length_m"],
            "sound_power_a_db": info["sound_power_a_db"],
            **horn,
            "about": about,
        }

    def credit(self, facts: dict) -> str:
        return self.credits(facts, {})

    def credits(self, facts: dict, files: dict[str, Path | None]) -> str:
        given = [f"{part}: {file.name}" for part, file in files.items() if file is not None]
        horn = ""
        fade = f"faded in and out over {FADE_S:g} s"
        if self.horn is not None and facts.get("horn") in HORN_PLAYS[1:]:
            # Its blasts rise and fall of themselves, unfaded (:meth:`fade_s`).
            fade += ", not the horn"
            if files.get(self.horn.part) is None:
                horn = f"; {self.horn.credit(facts['horn'])}"
        if given:
            return f"Playing {'; '.join(given)}{horn}, {fade}"
        return (
            f"{self.noun.capitalize()}'s sound made up: noise at each bogie, roof and "
            "pantograph with the octave spectrum of Schall 03's sources there "
            f"(16. BImSchV, Anlage 2, 2014) at {kmh(facts['speed_kmh'])} km/h{horn}; {fade}"
        )

    def texts(self) -> dict[str, str]:
        noun, name = self.noun, self.name
        return {
            "speed": f"how fast the {noun} runs along its walk (km/h; m/s times 3.6), which "
            "sets how long its pass and its sound last (the walk's length, and the "
            f"{noun}'s, over the speed) and its sound power; by default "
            + (
                f"the {noun}'s own, whatever its walk's: "
                + ", ".join(f"{key} {TRAINS[key].speed_kmh:g}" for key in self.trains)
                + " km/h"
                if self.own_speed
                else "as its walk in the scene goes"
            ),
            "input": f"a WAV file a source of the {noun} plays instead of its made sound, "
            f"as {name}=WAV for its wheels and bogies (or {name}_roof=WAV, "
            f"{name}_pantograph=WAV): a dry recording, mixed down to mono, at least as "
            "long as the pass, played at each of its points at that height from another "
            f"place in it; the captions' levels are Schall 03's {noun} at --speed"
            + (
                f"; {name}_horn=WAV for its horn (its blasts as loud as the made horn's, "
                "whatever its pauses)"
                if self.horn is not None
                else ""
            ),
            "sound": f"demo/{name}/{name}.wav and the files of its roof and pantograph: "
            f"the {noun}'s sound at each of its points, made to Schall 03",
            "header": f"# The example with its {noun} running past alone, at the speed and with "
            "the sound\n# power of the pass-by demo (Schall 03), written by passby_demo.py.\n",
        }


#: The railway example's train, and the street's tram.
TRAIN_VEHICLE = TrainVehicle(
    "train",
    "train",
    ("regional", "s_bahn", "intercity", "freight"),
    "ballast",
    tuple(TRACKS),
    "acoustic_spaces.demos.rail",
    horn=True,
)
TRAM_VEHICLE = TrainVehicle("tram", "tram", ("tram",), "street", (), "acoustic_spaces.demos.rail")

__all__ = [
    "AIR_CONDITIONING_DB",
    "HORN_PLAYS",
    "PARTS",
    "BOGIES_AT",
    "CATEGORIES",
    "DIRECTIVITY_DB",
    "REFERENCE_SPEED_KMH",
    "SOURCES",
    "SPEED_FACTORS",
    "TRACKS",
    "TRAINS",
    "TRAIN_VEHICLE",
    "TRAM_VEHICLE",
    "TRAM_SOURCES",
    "TRAM_SPEED_FACTORS",
    "TRAM_TRACKS",
    "Category",
    "Point",
    "Row",
    "Source",
    "Track",
    "Train",
    "TrainVehicle",
    "Unit",
    "directivity",
    "directivity_db",
    "point_sound",
    "train_points",
    "train_power_db",
    "unit_power",
]
