"""Ground-borne vibration from trains, trams and construction plant, and the rumble it
makes in a building, by the US Federal Transit Administration's general method.

FTA, *Transit Noise and Vibration Impact Assessment Manual*, FTA Report No. 0123
(September 2018): the General Vibration Assessment of its section 6.4 for trains and
trams (:func:`ground_level`, :func:`indoor_db`, :data:`NOISE_DB`), the construction
vibration of its section 7.2 for plant (:func:`plant_ppv`, :func:`plant_level`), and
the criteria of its tables 6-2 to 6-4 and 7-5.

A vibration level here is the manual's: the rms vibration velocity in VdB, decibels
re 1 micro-inch per second (25.4 nm/s). Distances are in metres, the manual's feet
converted, and speeds in km/h. The method gives the overall level at a building and
the A-weighted level of the sound its room surfaces radiate, not their spectra. The
manual reads its base curves as "the upper range of vibration levels from
well-maintained systems", which ground-borne vibration rarely exceeds "by more than 1
or 2 dB unless there are extenuating circumstances such as wheel- or running-surface
defects" (section 6.4, step 3).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

#: A foot and an inch (m), and a mile (km).
FOOT_M = 0.3048
INCH_M = 0.0254
MILE_KM = 1.609344
#: The reference of a vibration level in VdB: 1 micro-inch per second (m/s).
VELOCITY_REFERENCE_M_S = 1e-6 * INCH_M
#: Section 5.2: "the threshold of perception for humans is approximately 65 VdB".
PERCEPTION_VDB = 65.0
#: How near and how far from the track the manual draws its base curves (figure 6-4):
#: 10 to 300 ft from the track's centreline (m), the diagonal distance for a subway.
NEAREST_M, FARTHEST_M = 10.0 * FOOT_M, 300.0 * FOOT_M


@dataclass(frozen=True)
class Curve:
    """A base curve of the ground's vibration (table 6-10): the overall rms vibration
    level at the ground's surface ``D`` feet from the track, a0 + a1 x + a2 x^2 + a3 x^3
    with x = log10(D), for vehicles in good condition at ``reference_kmh`` on welded
    track at grade, with normal propagation through soil; drawn from 10 to 300 ft
    (:data:`NEAREST_M`, :data:`FARTHEST_M`), and fitted with these equations only
    there."""

    what: str
    coefficients: tuple[float, float, float, float]
    reference_kmh: float

    def level(
        self, distance_m: float, speed_kmh: float | None = None, extend: bool = False
    ) -> float:
        """The level (VdB) ``distance_m`` from the track, at ``speed_kmh`` (eq. 6-4) or
        at the curve's own speed; a ValueError nearer or farther than the manual draws
        the curve, unless ``extend``: then nearer it holds its value at
        :data:`NEAREST_M`, and farther it goes on falling as it falls at
        :data:`FARTHEST_M` (straight on in log10 of the distance; the cubics themselves
        turn back up there)."""
        inside = NEAREST_M * (1 - 1e-9) <= distance_m <= FARTHEST_M * (1 + 1e-9)
        if not inside and not extend:
            raise ValueError(
                f"the manual draws its curves from {NEAREST_M:.2f} to {FARTHEST_M:.1f} m "
                f"(10 to 300 ft) from the track, not {distance_m:g} m"
            )
        a0, a1, a2, a3 = self.coefficients
        x = math.log10(min(max(distance_m, NEAREST_M), FARTHEST_M) / FOOT_M)
        level = a0 + a1 * x + a2 * x**2 + a3 * x**3
        if distance_m > FARTHEST_M:
            slope = a1 + 2.0 * a2 * x + 3.0 * a3 * x**2
            level += slope * math.log10(distance_m / FARTHEST_M)
        if speed_kmh is not None:
            level += speed_db(speed_kmh, self.reference_kmh)
        return level


#: Table 6-9 and 6-10: the three base curves. The locomotive-powered curve is for trains
#: hauled by diesel or electric locomotives (intercity, commuter and freight trains),
#: the rapid transit curve for heavy and light rail vehicles, electric multiple units and
#: streetcars, the rubber-tyred curve for a bus on a smooth road.
CURVES: dict[str, Curve] = {
    "locomotive": Curve(
        "locomotive-powered passenger or freight trains",
        (92.28, 14.81, -14.17, 1.65),
        50.0 * MILE_KM,
    ),
    "rapid_transit": Curve(
        "rapid transit or light rail vehicles",
        (85.88, -1.06, -2.32, -0.87),
        50.0 * MILE_KM,
    ),
    "rubber_tyred": Curve(
        "rubber-tyred vehicles",
        (66.08, 34.28, -30.25, 5.40),
        30.0 * MILE_KM,
    ),
}


def speed_db(speed_kmh: float, reference_kmh: float) -> float:
    """Eq. 6-4: the change in vibration level with speed, 20 log10(speed / reference)."""
    return 20.0 * math.log10(speed_kmh / reference_kmh)


#: Table 6-11, vehicle parameters (dB; one of them, the greatest that applies): a
#: primary suspension stiffer than a vertical resonance at 15 Hz, resilient wheels, and
#: worn wheels or wheels with flats.
VEHICLES: dict[str, float] = {
    "good": 0.0,
    "stiff_suspension": 8.0,
    "resilient_wheels": 0.0,
    "worn_wheels": 10.0,
}
#: Table 6-11, track conditions (dB; the greatest that applies, with special trackwork
#: near): welded track in good condition, jointed track, worn or corrugated track, and
#: an uneven road for rubber-tyred vehicles.
TRACK_CONDITIONS: dict[str, float] = {
    "good": 0.0,
    "jointed": 5.0,
    "worn": 10.0,
    "uneven_road": 5.0,
}
#: Table 6-11 and 6-12: special trackwork (turnouts and crossovers) adds 10 dB within
#: 100 ft of it and 5 dB between 100 and 200 ft, nothing farther.
SPECIAL_TRACKWORK_DB = ((100.0 * FOOT_M, 10.0), (200.0 * FOOT_M, 5.0))
#: Track treatments (dB; one of them): table 6-11's floating slab, ballast mats and
#: high-resilience fasteners, and table 6-12's resiliently supported ties (low-vibration
#: track).
TREATMENTS: dict[str, float] = {
    "none": 0.0,
    "floating_slab": -15.0,
    "ballast_mats": -10.0,
    "resilient_fasteners": -5.0,
    "resilient_ties": -10.0,
}
#: Table 6-12, the type of transit structure (dB; one of them): at grade, elevated and
#: in open cut against tie and ballast at grade; a subway's station, cut and cover and
#: rock-based tunnel against a bored tunnel in soil, which the rapid transit curve
#: assumes for a subway.
STRUCTURES: dict[str, float] = {
    "at_grade": 0.0,
    "elevated": -10.0,
    "open_cut": 0.0,
    "subway": 0.0,
    "subway_station": -5.0,
    "subway_cut_and_cover": -3.0,
    "subway_rock": -15.0,
}
#: Table 6-12, ground-borne propagation: geologic conditions that promote efficient
#: propagation in soil add 10 dB.
EFFICIENT_SOIL_DB = 10.0
#: Table 6-12, propagation in a rock layer: its gain over soil 50, 100, 150 and 200 ft
#: away (dB). Between them it is interpolated, and held at the ends nearer and farther.
ROCK_DB = (
    (50.0 * FOOT_M, 2.0),
    (100.0 * FOOT_M, 4.0),
    (150.0 * FOOT_M, 6.0),
    (200.0 * FOOT_M, 9.0),
)
#: The ground the vibration travels through.
GROUNDS = ("normal", "efficient", "rock")


@dataclass(frozen=True)
class Line:
    """What the manual adjusts its base curve for, at the source and along the path
    (tables 6-11 and 6-12): the vehicles, the track, its treatment and structure, and
    the ground."""

    vehicles: str = "good"
    track: str = "good"
    #: The distance from the nearest turnout or crossover (m), or None for none near.
    special_trackwork_m: float | None = None
    treatment: str = "none"
    structure: str = "at_grade"
    ground: str = "normal"

    def __post_init__(self) -> None:
        for name, value, known in (
            ("vehicles", self.vehicles, VEHICLES),
            ("track", self.track, TRACK_CONDITIONS),
            ("treatment", self.treatment, TREATMENTS),
            ("structure", self.structure, STRUCTURES),
            ("ground", self.ground, GROUNDS),
        ):
            if value not in known:
                raise ValueError(f"unknown {name} {value!r}; use one of {', '.join(known)}")

    def source_db(self) -> float:
        """Table 6-11's adjustment for the vehicles, the track and its treatment (dB):
        the vehicles' and the track's added, but only the greater of the two when the
        track is worn or corrugated, as the manual says."""
        vehicles = VEHICLES[self.vehicles]
        track = TRACK_CONDITIONS[self.track]
        if self.special_trackwork_m is not None:
            near = [db for within, db in SPECIAL_TRACKWORK_DB if self.special_trackwork_m <= within]
            track = max(track, near[0] if near else 0.0)
        if self.track == "worn" and vehicles > 0.0:
            conditions = max(vehicles, track)
        else:
            conditions = vehicles + track
        return conditions + TREATMENTS[self.treatment]

    def path_db(self, distance_m: float) -> float:
        """Table 6-12's adjustment for the structure and the ground, ``distance_m`` from
        the track (dB)."""
        if self.ground == "efficient":
            ground = EFFICIENT_SOIL_DB
        elif self.ground == "rock":
            ground = float(
                np.interp(distance_m, [d for d, _ in ROCK_DB], [db for _, db in ROCK_DB])
            )
        else:
            ground = 0.0
        return STRUCTURES[self.structure] + ground


def ground_level(
    curve: str, distance_m: float, speed_kmh: float, line: Line | None = None
) -> float:
    """The overall vibration level (VdB) at the ground's surface ``distance_m`` from the
    track, of vehicles on the base curve ``curve`` (:data:`CURVES`) at ``speed_kmh``,
    with the adjustments of ``line`` (none by default): what reaches a building's
    foundation."""
    line = line or Line()
    found = CURVES[curve].level(distance_m, speed_kmh)
    return found + line.source_db() + line.path_db(distance_m)


#: Table 6-12, the coupling loss of a building's foundation to the ground (dB).
COUPLINGS: dict[str, float] = {
    "wood_frame": -5.0,
    "masonry_1_2": -7.0,
    "masonry_3_4": -10.0,
    "large_masonry_piles": -10.0,
    "large_masonry_footings": -13.0,
    "on_rock": 0.0,
}
#: Section 6.4, step 2c: where a building's structure is not known, the manual's
#: conservative change from the vibration outdoors to indoors (dB), instead of its
#: coupling loss, floors and amplification: a light wood-frame house's on its first
#: floor and on its second and third, a small masonry building's and a large building's.
GENERIC: dict[str, float] = {
    "light_wood_frame": 3.0,
    "small_masonry": 3.0,
    "large": 0.0,
}
#: A light wood-frame house's on its second and third floors (dB).
LIGHT_WOOD_FRAME_UPPER_DB = 6.0
#: What each building of :data:`COUPLINGS` and :data:`GENERIC` is, in words.
BUILDINGS: dict[str, str] = {
    "wood_frame": "a wood-frame house",
    "masonry_1_2": "a masonry building of one or two storeys",
    "masonry_3_4": "a masonry building of three or four storeys",
    "large_masonry_piles": "a large masonry building on piles",
    "large_masonry_footings": "a large masonry building on spread footings",
    "on_rock": "a building founded on rock",
    "light_wood_frame": "a light wood-frame house, its structure not known",
    "small_masonry": "a small masonry building, its structure not known",
    "large": "a large building, its structure not known",
}
#: Table 6-13: the amplification by the resonances of floors, walls and ceilings (dB).
FLOOR_AMPLIFICATION_DB = 6.0
#: Table 6-13: the attenuation per floor (dB) from the first suspended floor up to the
#: fifth floor above grade, and from there up to the tenth.
PER_FLOOR_DB = (-2.0, -1.0)


def floors_db(floor: int) -> float:
    """Table 6-13's attenuation (dB) up to ``floor``, counted from 1 above the
    basement, as the manual's first floor assumes one: 2 dB a floor up to the fifth,
    1 dB a floor from there up to the tenth."""
    if not 1 <= floor <= 10:
        raise ValueError(f"the manual gives floors 1 to 10 above grade, not {floor}")
    low, high = PER_FLOOR_DB
    return low * min(floor, 5) + high * max(floor - 5, 0)


def indoor_db(building: str, floor: int = 1) -> float:
    """The change in vibration level (dB) from the ground outside ``building`` to its
    ``floor``. For one of :data:`COUPLINGS`, its foundation's coupling loss, the
    attenuation up to the floor and the floors' amplification (tables 6-12 and 6-13):
    the first floor of a wood-frame house gives -5 - 2 + 6 = -1 dB, as the manual's own
    example says. For one of :data:`GENERIC`, the manual's conservative change where
    the structure is not known; a light wood-frame house's is given for its first three
    floors only."""
    if building in GENERIC:
        if building != "light_wood_frame" or floor == 1:
            return GENERIC[building]
        if floor in (2, 3):
            return LIGHT_WOOD_FRAME_UPPER_DB
        raise ValueError(f"the manual gives a light wood-frame house's floors 1 to 3, not {floor}")
    if building not in COUPLINGS:
        raise ValueError(f"unknown building {building!r}; use one of {', '.join(BUILDINGS)}")
    return COUPLINGS[building] + floors_db(floor) + FLOOR_AMPLIFICATION_DB


#: Table 6-14: the A-weighted sound level radiated by a room's surfaces (dB(A)) against
#: their vibration level, by the frequency where the ground's vibration peaks: below 30
#: Hz (most surface track, subways in sandy soil, track on vibration isolation), 30 to
#: 60 Hz (most subways, surface track on very stiff clayey soil) or above 60 Hz
#: (subways founded in rock or very stiff clayey soil).
NOISE_DB: dict[str, float] = {"low": -50.0, "mid": -35.0, "high": -20.0}
#: The manual's unweighted sound pressure level of the rumble against the room surfaces'
#: vibration level (dB): "approximately 5 dB" less (section 6.4, step 2d).
UNWEIGHTED_DB = -5.0
#: Where each peak of :data:`NOISE_DB` puts the rumble (Hz): the third octave whose
#: A-weighting is the table's adjustment less :data:`UNWEIGHTED_DB` to within 1.1 dB (25
#: Hz -44.7 dB, 50 Hz -30.2, 125 Hz -16.1), as the manual's note to its table 5-5 says
#: the A-weighted level comes from A-weighting the vibration's spectrum and taking off
#: 5 dB for the room.
RUMBLE_HZ: dict[str, float] = {"low": 25.0, "mid": 50.0, "high": 125.0}


def noise_level(vibration_vdb: float, peak: str = "low") -> float:
    """The ground-borne noise (dB(A)) of a room whose surfaces vibrate at
    ``vibration_vdb``, the ground's vibration peaking at ``peak`` (:data:`NOISE_DB`)."""
    if peak not in NOISE_DB:
        raise ValueError(f"unknown peak {peak!r}; use one of {', '.join(NOISE_DB)}")
    return vibration_vdb + NOISE_DB[peak]


#: Table 6-2: the events a day above which they are frequent, and from which occasional.
FREQUENT_ABOVE, OCCASIONAL_FROM = 70, 30


def events(per_day: float) -> str:
    """Table 6-2: ``per_day`` events are frequent above 70 a day, occasional from 30 to
    70, infrequent under 30."""
    if per_day > FREQUENT_ABOVE:
        return "frequent"
    return "occasional" if per_day >= OCCASIONAL_FROM else "infrequent"


@dataclass(frozen=True)
class Criterion:
    """An impact criterion: the indoor vibration (VdB) and ground-borne noise (dB(A),
    or None where the manual gives none) above which there is impact."""

    vibration_vdb: float
    noise_dba: float | None


#: Table 6-3, by land use: 1 buildings where vibration would interfere with what is done
#: inside (moderately sensitive equipment such as optical microscopes), 2 residences and
#: buildings where people normally sleep, 3 institutions used mostly by day; and by
#: events (:func:`events`).
CRITERIA: dict[int, dict[str, Criterion]] = {
    1: {kind: Criterion(65.0, None) for kind in ("frequent", "occasional", "infrequent")},
    2: {
        "frequent": Criterion(72.0, 35.0),
        "occasional": Criterion(75.0, 38.0),
        "infrequent": Criterion(80.0, 43.0),
    },
    3: {
        "frequent": Criterion(75.0, 40.0),
        "occasional": Criterion(78.0, 43.0),
        "infrequent": Criterion(83.0, 48.0),
    },
}
#: Table 6-4, special buildings: frequent events, then occasional or infrequent ones.
SPECIAL_BUILDINGS: dict[str, tuple[Criterion, Criterion]] = {
    "concert_hall": (Criterion(65.0, 25.0), Criterion(65.0, 25.0)),
    "tv_studio": (Criterion(65.0, 25.0), Criterion(65.0, 25.0)),
    "recording_studio": (Criterion(65.0, 25.0), Criterion(65.0, 25.0)),
    "auditorium": (Criterion(72.0, 30.0), Criterion(80.0, 38.0)),
    "theatre": (Criterion(72.0, 35.0), Criterion(80.0, 43.0)),
}


def criterion(use: int | str, kind: str) -> Criterion:
    """The criterion for the land use ``use`` (1, 2 or 3, :data:`CRITERIA`, or a
    special building, :data:`SPECIAL_BUILDINGS`) and events of ``kind``."""
    if isinstance(use, str):
        frequent, other = SPECIAL_BUILDINGS[use]
        return frequent if kind == "frequent" else other
    return CRITERIA[use][kind]


#: Section 6.2, step 3c: a freight train's locomotives count as infrequent events and its
#: wagons, which pass for minutes, as frequent ones, the locomotives "often" 3 to 8 dB
#: higher; the wagons are taken FREIGHT_WAGONS_DB under the curve, the least of that.
FREIGHT_WAGONS_DB = -3.0


@dataclass(frozen=True)
class Plant:
    """A piece of construction plant's vibration 25 ft away (table 7-4): its peak
    particle velocity (in/s) and its rms vibration level (VdB), which the manual takes
    12 dB under the peak (a crest factor of 4)."""

    what: str
    ppv_in_s: float
    level_vdb: float


#: Table 7-4: averages over many sites and soils.
PLANT: dict[str, Plant] = {
    "pile_driver_impact_upper": Plant("an impact pile driver, upper range", 1.518, 112.0),
    "pile_driver_impact": Plant("an impact pile driver, typical", 0.644, 104.0),
    "pile_driver_sonic_upper": Plant("a sonic pile driver, upper range", 0.734, 105.0),
    "pile_driver_sonic": Plant("a sonic pile driver, typical", 0.170, 93.0),
    "clam_shovel": Plant("a clam shovel drop (slurry wall)", 0.202, 94.0),
    "hydromill_soil": Plant("a hydromill (slurry wall) in soil", 0.008, 66.0),
    "hydromill_rock": Plant("a hydromill (slurry wall) in rock", 0.017, 75.0),
    "vibratory_roller": Plant("a vibratory roller", 0.210, 94.0),
    "hoe_ram": Plant("a hoe ram (a breaker on an excavator)", 0.089, 87.0),
    "large_bulldozer": Plant("a large bulldozer", 0.089, 87.0),
    "caisson_drilling": Plant("caisson drilling", 0.089, 87.0),
    "loaded_trucks": Plant("loaded trucks", 0.076, 86.0),
    "jackhammer": Plant("a jackhammer", 0.035, 79.0),
    "small_bulldozer": Plant("a small bulldozer", 0.003, 58.0),
}
#: Where table 7-4 gives the plant's vibration (m): 25 ft.
PLANT_REFERENCE_M = 25.0 * FOOT_M


def plant_ppv(plant: str, distance_m: float) -> float:
    """Eq. 7-2: the peak particle velocity (in/s) ``distance_m`` from ``plant``
    (:data:`PLANT`), PPV (25 ft / D)^1.5, a point source with normal propagation."""
    return PLANT[plant].ppv_in_s * (PLANT_REFERENCE_M / distance_m) ** 1.5


def plant_level(plant: str, distance_m: float) -> float:
    """Eq. 7-3: the rms vibration level (VdB) ``distance_m`` from ``plant``, Lv - 30
    log10(D / 25 ft)."""
    return PLANT[plant].level_vdb - 30.0 * math.log10(distance_m / PLANT_REFERENCE_M)


@dataclass(frozen=True)
class Damage:
    """A category of building and the peak particle velocity (in/s) from which plant
    may damage it (table 7-5), with the rms level the manual gives for it (VdB)."""

    what: str
    ppv_in_s: float
    level_vdb: float


#: Table 7-5: construction vibration damage criteria.
DAMAGE: dict[str, Damage] = {
    "I": Damage("reinforced-concrete, steel or timber (no plaster)", 0.5, 102.0),
    "II": Damage("engineered concrete and masonry (no plaster)", 0.3, 98.0),
    "III": Damage("non-engineered timber and masonry buildings", 0.2, 94.0),
    "IV": Damage("buildings extremely susceptible to vibration damage", 0.12, 90.0),
}


def mm_s(in_s: float) -> float:
    """A velocity in inches a second, in millimetres a second."""
    return in_s * INCH_M * 1000.0


def level_of(ppv_in_s: float, crest_db: float = 12.0) -> float:
    """The rms vibration level (VdB) of a peak particle velocity (in/s), ``crest_db``
    under it: the manual's tables 7-4 and 7-5 take a crest factor of 4 (12 dB)."""
    return 20.0 * math.log10(ppv_in_s / 1e-6) - crest_db


def distance_for(
    level: Callable[[float], float], limit: float, near_m: float, far_m: float
) -> float | None:
    """The distance (m) between ``near_m`` and ``far_m`` beyond which ``level`` (of the
    distance, falling with it) stays at or under ``limit``: ``near_m`` when it is under
    all along, None when it is over all along."""
    if level(near_m) <= limit:
        return near_m
    if level(far_m) > limit:
        return None
    near, far = math.log(near_m), math.log(far_m)
    for _ in range(60):
        middle = 0.5 * (near + far)
        if level(math.exp(middle)) > limit:
            near = middle
        else:
            far = middle
    return math.exp(far)


def horizontal_distance(
    point: Sequence[float], path: Sequence[Sequence[float]], closed: bool = False
) -> float:
    """The distance (m) in plan from ``point`` (x, y[, z]) to the nearest point of
    ``path``, a line through (x, y[, z]) points, or the outline of a footprint when
    ``closed``."""
    return float(horizontal_distances([point], path, closed)[0])


def horizontal_distances(points, path, closed: bool = False) -> np.ndarray:
    """(n,) :func:`horizontal_distance` of each of ``points`` (n, 2 or 3)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))[:, None, :2]
    corners = np.asarray(path, dtype=float)[:, :2]
    if closed:
        corners = np.vstack([corners, corners[:1]])
    a, b = corners[:-1], corners[1:]
    ab = b - a
    lengths = np.einsum("ij,ij->i", ab, ab)
    # A leg of no length (a point repeated, a ring closed on its first corner) is that
    # point.
    t = np.divide(
        np.einsum("nij,ij->ni", p - a, ab),
        lengths,
        out=np.zeros((len(p), len(ab))),
        where=lengths > 0.0,
    )
    nearest = a + np.clip(t, 0.0, 1.0)[..., None] * ab
    ends = np.broadcast_to(corners[-1:], (len(p), 1, 2))
    return np.min(np.linalg.norm(np.concatenate([nearest, ends], axis=1) - p, axis=2), axis=1)


__all__ = [
    "BUILDINGS",
    "COUPLINGS",
    "CRITERIA",
    "CURVES",
    "DAMAGE",
    "FARTHEST_M",
    "FREIGHT_WAGONS_DB",
    "GENERIC",
    "NEAREST_M",
    "NOISE_DB",
    "PERCEPTION_VDB",
    "PLANT",
    "RUMBLE_HZ",
    "SPECIAL_BUILDINGS",
    "UNWEIGHTED_DB",
    "Criterion",
    "Curve",
    "Damage",
    "Line",
    "Plant",
    "criterion",
    "distance_for",
    "events",
    "floors_db",
    "ground_level",
    "horizontal_distance",
    "horizontal_distances",
    "indoor_db",
    "level_of",
    "mm_s",
    "noise_level",
    "plant_level",
    "plant_ppv",
    "speed_db",
]
