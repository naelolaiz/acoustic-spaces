"""Flight performance by Doc 29 Volume 2 Appendix B: an aircraft's fixed-point profile
(its height, true airspeed and corrected net thrust along the ground track) from its
procedural steps, its engine and aerodynamic coefficients and its weight, at an
aerodrome's elevation, temperature and pressure and into a headwind.

As Appendix B does, the calculation runs in feet, knots and pounds, and its profile
points (:class:`Point`) are in those units: :func:`as_profile` gives them in the metres
of :class:`~acoustic_spaces.doc29.ProfilePoint`, ready for
:func:`~acoustic_spaces.doc29.flight_path`.

The aircraft's coefficients are read from tables laid out as the ANP database's
(Volume 2 Appendix G3): :func:`read_performance` reads its aircraft, jet engine,
propeller engine, aerodynamic coefficients and default weights tables, and
:func:`read_departure_steps` and :func:`read_approach_steps` its procedural steps.

The aircraft flies straight: the bank angles of turns (Appendix B5) are not part of the
calculation.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .tables import FOOT_M, KNOT_M_S, ProfilePoint, _column, _key, read_rows

#: The standard gravitational acceleration (ft/s²) and a knot (ft/s), Appendix B2.
G_FT_S2 = 32.174
KNOT_FT_S = 1.68781
#: The standard sea-level pressure (inHg) Appendix B3 keeps, and the lapse rate of the
#: temperature (°F/ft), the standard sea-level temperature (°R) and the exponent of its
#: pressure ratio.
SEA_LEVEL_INHG = 29.92
LAPSE_F_PER_FT = 0.003566
SEA_LEVEL_RANKINE = 518.67
PRESSURE_EXPONENT = 5.256
#: The breakpoint temperature (°C) of eq. B-11, when the high-temperature jet
#: coefficients are not given.
BREAKPOINT_C = 30.0
#: The engine-out climb gradients (%) of eq. B-13, by number of engines.
ENGINE_OUT_GRADIENT = {2: 1.2, 3: 1.5, 4: 1.7}
#: The longest transition segment (ft) where the thrust changes (B6.1.6, B7).
TRANSITION_FT = 1000.0
#: The headwind (kt) the coefficients are given for (Appendix G3.1).
REFERENCE_HEADWIND_KT = 8.0


@dataclass(frozen=True)
class Aerodrome:
    """Where and in what air the aircraft flies (Appendix B3): the aerodrome's
    elevation (ft above mean sea level), its temperature (°C), its pressure reduced to
    sea level (QNH, inHg) and the headwind (kt). The defaults are the conditions of the
    ANP data (Appendix G3.1): a sea-level aerodrome in the standard atmosphere, into
    an 8 kt headwind."""

    elevation_ft: float = 0.0
    temperature_c: float = 15.0
    pressure_inhg: float = SEA_LEVEL_INHG
    headwind_kt: float = REFERENCE_HEADWIND_KT

    def theta(self, altitude_ft: float) -> float:
        """The temperature ratio at ``altitude_ft`` (ft above mean sea level; eq. B-3)."""
        fahrenheit = self.temperature_c * 9.0 / 5.0 + 32.0
        lapse = LAPSE_F_PER_FT * (altitude_ft - self.elevation_ft)
        return (459.67 + fahrenheit - lapse) / SEA_LEVEL_RANKINE

    def delta(self, altitude_ft: float) -> float:
        """The pressure ratio at ``altitude_ft`` (eq. B-4)."""
        base = (self.pressure_inhg / SEA_LEVEL_INHG) ** (1.0 / PRESSURE_EXPONENT)
        return float((base - LAPSE_F_PER_FT * altitude_ft / SEA_LEVEL_RANKINE) ** PRESSURE_EXPONENT)

    def sigma(self, altitude_ft: float) -> float:
        """The density ratio at ``altitude_ft`` (eq. B-5)."""
        return self.delta(altitude_ft) / self.theta(altitude_ft)

    def pressure_altitude(self, altitude_ft: float) -> float:
        """The pressure altitude (ft) at ``altitude_ft`` (eq. B-6)."""
        ratio = self.delta(altitude_ft) ** (1.0 / PRESSURE_EXPONENT)
        return SEA_LEVEL_RANKINE / LAPSE_F_PER_FT * (1.0 - ratio)

    def temperature(self, altitude_ft: float) -> float:
        """The air's temperature (°C) at ``altitude_ft`` (eq. B-1, B-2)."""
        fahrenheit = self.temperature_c * 9.0 / 5.0 + 32.0
        return (fahrenheit - LAPSE_F_PER_FT * (altitude_ft - self.elevation_ft) - 32.0) * 5.0 / 9.0


#: The conditions the approach procedures' idle steps are given for (Appendix B7.1.2,
#: B7.1.4): the standard atmosphere at a sea-level aerodrome.
STANDARD = Aerodrome()


@dataclass(frozen=True)
class JetCoefficients:
    """A thrust rating's jet engine coefficients (Appendix G3.2): E (lb), F (lb/kt),
    GA (lb/ft), GB (lb/ft²) and H (lb/°C) of eq. B-9."""

    e: float
    f: float
    ga: float
    gb: float
    h: float

    def thrust(
        self, calibrated_kt: float, pressure_altitude_ft: float, temperature_c: float
    ) -> float:
        """The corrected net thrust per engine (lb; eq. B-9, B-10)."""
        h = pressure_altitude_ft
        return (
            self.e + self.f * calibrated_kt + self.ga * h + self.gb * h * h + self.h * temperature_c
        )


@dataclass(frozen=True)
class PropellerCoefficients:
    """A thrust rating's propeller coefficients (Appendix G3.3): the propeller's
    efficiency and the installed net propulsive power (hp)."""

    efficiency: float
    power_hp: float


@dataclass(frozen=True)
class Flaps:
    """A flap setting's aerodynamic coefficients (Appendix G3.4): B (ft/lb), the
    take-off ground roll's; C or D (kt/√lb), the take-off's or the landing's speed; and
    R, the ratio of drag to lift. B, C and D are given only for the flaps of take-off
    and landing (None for the others)."""

    b: float | None
    speed: float | None
    r: float


@dataclass(frozen=True)
class Performance:
    """What the performance calculation needs of an aircraft (Appendix G2 to G3.5)."""

    name: str
    engine_type: str
    engines: int
    #: Its greatest landing weight and its engines' greatest static thrust at sea
    #: level, each (lb).
    max_landing_weight_lb: float
    max_static_thrust_lb: float
    #: Its jet engine coefficients and its propeller coefficients, by the key of the
    #: thrust rating (:func:`rating_key`): "maxtakeoff", "maxclimb", "idleapproach",
    #: and those with the suffix "hitemp", above the breakpoint temperature.
    jet: dict[str, JetCoefficients] = field(default_factory=dict)
    propeller: dict[str, PropellerCoefficients] = field(default_factory=dict)
    #: Its flaps' coefficients, by (operation "A" or "D", flap identifier).
    flaps: dict[tuple[str, str], Flaps] = field(default_factory=dict)
    #: Its default weights (lb), by (operation "A" or "D", stage length).
    weights: dict[tuple[str, str], float] = field(default_factory=dict)

    def flap(self, operation: str, name: str) -> Flaps:
        """The coefficients of the flap setting ``name`` on arrival ("A") or departure
        ("D")."""
        try:
            return self.flaps[(operation, name.strip())]
        except KeyError:
            raise KeyError(
                f"{self.name}: no aerodynamic coefficients for {operation} flaps {name!r}"
            ) from None


def rating_key(name: str) -> str:
    """The key of the thrust rating ``name``: its letters and digits in lower case, so
    that "MaxTakeoff", "MaxTakeOff" and "MaxTake-off" are one, with "HighTemp" written
    "hitemp"."""
    return _key(name).replace("hightemp", "hitemp")


@dataclass(frozen=True)
class Point:
    """A point of a fixed-point profile as Appendix B gives it: the distance along the
    ground track (ft; from the start of roll on departure, to touchdown on approach,
    negative before it), the height above the aerodrome (ft), the true airspeed (kt)
    and the corrected net thrust per engine (lb)."""

    distance_ft: float
    height_ft: float
    speed_kt: float
    thrust_lb: float


def as_profile(points: Sequence[Point]) -> tuple[ProfilePoint, ...]:
    """``points`` as the profile points (m, m/s) of a flight path."""
    return tuple(
        ProfilePoint(
            distance_m=p.distance_ft * FOOT_M,
            height_m=p.height_ft * FOOT_M,
            speed_m_s=p.speed_kt * KNOT_M_S,
            power=p.thrust_lb,
        )
        for p in points
    )


@dataclass(frozen=True)
class DepartureStep:
    """A step of a departure procedure (Appendix G3.6): its type ("Takeoff", "Climb",
    "Accelerate", "Level" or "Level-Accelerate"), its thrust rating, its flaps, the
    height it climbs to (ft above the aerodrome; a climb's), its rate of climb (ft/min)
    or its share of the thrust that accelerates (%, its accel percentage) and the
    calibrated airspeed it accelerates to (kt; an acceleration's), and its length (ft;
    a level step's)."""

    kind: str
    thrust: str
    flap: str
    altitude_ft: float | None = None
    climb_rate_ft_min: float | None = None
    speed_kt: float | None = None
    accelerating_percent: float | None = None
    distance_ft: float | None = None


@dataclass(frozen=True)
class ApproachStep:
    """A step of an approach procedure (Appendix G3.7): its type ("Descend",
    "Descend-Decel", "Descend-Idle", "Level", "Level-Decel", "Level-Idle", "Land" or
    "Decelerate"), its flaps, the height (ft above the aerodrome) and calibrated
    airspeed (kt) it starts at, its descent angle (°), the touchdown roll (ft; a
    landing's), its length (ft) and the thrust it starts with (% of the greatest static
    thrust; a deceleration's on the runway)."""

    kind: str
    flap: str
    altitude_ft: float | None = None
    speed_kt: float | None = None
    angle_deg: float | None = None
    touchdown_roll_ft: float | None = None
    distance_ft: float | None = None
    thrust_percent: float | None = None


def _number(text: str) -> float | None:
    text = text.strip()
    if not text or text in ("-", "n/a", "N/A"):
        return None
    return float(text)


def _optional(names: Sequence[str], *wanted: str) -> str | None:
    try:
        return _column(names, *wanted)
    except KeyError:
        return None


def read_performance(
    aircraft: str | Path,
    jet: str | Path | None = None,
    propeller: str | Path | None = None,
    aerodynamic: str | Path | None = None,
    weights: str | Path | None = None,
) -> dict[str, Performance]:
    """The aircraft of the aircraft table ``aircraft`` with their coefficients from the
    jet engine coefficients table ``jet``, the propeller engine coefficients table
    ``propeller``, the aerodynamic coefficients table ``aerodynamic`` and the default
    weights table ``weights`` (each optional), by identifier."""
    rows = read_rows(aircraft)
    if not rows:
        return {}
    c = list(rows[0].keys())
    name = _column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")
    found = {
        row[name]: Performance(
            name=row[name],
            engine_type=row[_column(c, "Engine Type")],
            engines=int(float(row[_column(c, "Number of Engines")])),
            max_landing_weight_lb=float(
                row[
                    _column(
                        c,
                        "Maximum Gross Landing Weight",
                        "Max Gross Landing Weight",
                        "Max Gross Weight Landing",
                    )
                ]
            ),
            max_static_thrust_lb=float(
                row[_column(c, "Maximum Sea Level Static Thrust", "Max Sea Level Static Thrust")]
            ),
        )
        for row in rows
    }
    for row, c in _rows(jet):
        plane = found.get(row[_column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")])
        if plane is not None:
            plane.jet[rating_key(row[_column(c, "Thrust Rating")])] = JetCoefficients(
                e=float(row[_column(c, "E")]),
                f=float(row[_column(c, "F")]),
                ga=float(row[_column(c, "Ga", "GA")]),
                gb=float(row[_column(c, "Gb", "GB")]),
                h=float(row[_column(c, "H")]),
            )
    for row, c in _rows(propeller):
        plane = found.get(row[_column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")])
        if plane is not None:
            plane.propeller[rating_key(row[_column(c, "Thrust Rating")])] = PropellerCoefficients(
                efficiency=float(row[_column(c, "Propeller Efficiency", "Efficiency")]),
                power_hp=float(row[_column(c, "Installed Net Propulsive Power", "Power")]),
            )
    for row, c in _rows(aerodynamic):
        plane = found.get(row[_column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")])
        if plane is not None:
            mode = row[_column(c, "Operation Type", "Operation Mode", "Op Mode", "Op Type")]
            r = _number(row[_column(c, "R")])
            if r is None:
                continue
            plane.flaps[(mode[:1].upper(), row[_column(c, "Flap Identifier", "Flap ID")])] = Flaps(
                b=_number(row[_column(c, "B")]),
                speed=_number(row[_column(c, "C/D", "C", "D")]),
                r=r,
            )
    for row, c in _rows(weights):
        plane = found.get(row[_column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")])
        if plane is not None:
            operation = _optional(c, "Operation", "Op Mode")
            key = (
                row[operation][:1].upper() if operation else "D",
                row[_column(c, "Stage Length")],
            )
            plane.weights[key] = float(row[_column(c, "Weight")])
    return found


def _rows(path: str | Path | None) -> list[tuple[dict[str, str], list[str]]]:
    if path is None:
        return []
    rows = read_rows(path)
    return [(row, list(row.keys())) for row in rows]


def read_departure_steps(
    path: str | Path,
) -> dict[tuple[str, str, str], tuple[DepartureStep, ...]]:
    """The departure procedures of the departure procedural steps table ``path``, by
    (aircraft, profile identifier, stage length), each its steps in order."""
    rows = read_rows(path)
    if not rows:
        return {}
    c = list(rows[0].keys())
    name = _column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")
    profile = _column(c, "Profile Identifier", "Profile ID")
    stage = _column(c, "Stage Length")
    number = _column(c, "Step Number")
    altitude = _optional(c, "End Point Altitude", "Altitude")
    rate = _optional(c, "Rate of Climb")
    speed = _optional(c, "End Point CAS", "CAS")
    accel = _optional(c, "Accel Percentage", "Accel")
    distance = _optional(c, "Distance")
    grouped: dict[tuple[str, str, str], list[tuple[int, DepartureStep]]] = {}
    for row in rows:
        step = DepartureStep(
            kind=row[_column(c, "Step Type")],
            thrust=row[_column(c, "Thrust Rating")],
            flap=row[_column(c, "Flap Identifier", "Flap ID")],
            altitude_ft=_number(row[altitude]) if altitude else None,
            climb_rate_ft_min=_number(row[rate]) if rate else None,
            speed_kt=_number(row[speed]) if speed else None,
            accelerating_percent=_number(row[accel]) if accel else None,
            distance_ft=_number(row[distance]) if distance else None,
        )
        key = (row[name], row[profile], row[stage])
        grouped.setdefault(key, []).append((int(float(row[number])), step))
    return {
        key: tuple(s for _, s in sorted(steps, key=lambda x: x[0]))
        for key, steps in grouped.items()
    }


def read_approach_steps(path: str | Path) -> dict[tuple[str, str], tuple[ApproachStep, ...]]:
    """The approach procedures of the approach procedural steps table ``path``, by
    (aircraft, profile identifier), each its steps in order."""
    rows = read_rows(path)
    if not rows:
        return {}
    c = list(rows[0].keys())
    name = _column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")
    profile = _column(c, "Profile Identifier", "Profile ID")
    number = _column(c, "Step Number")
    columns = {
        "altitude_ft": _optional(c, "Start Altitude"),
        "speed_kt": _optional(c, "Start CAS"),
        "angle_deg": _optional(c, "Descent Angle"),
        "touchdown_roll_ft": _optional(c, "Touchdown Roll"),
        "distance_ft": _optional(c, "Distance"),
        "thrust_percent": _optional(c, "Start Thrust"),
    }
    grouped: dict[tuple[str, str], list[tuple[int, ApproachStep]]] = {}
    for row in rows:
        values = {key: _number(row[col]) if col else None for key, col in columns.items()}
        step = ApproachStep(
            kind=row[_column(c, "Step Type")],
            flap=row[_column(c, "Flap Identifier", "Flap ID")],
            **values,
        )
        grouped.setdefault((row[name], row[profile]), []).append((int(float(row[number])), step))
    return {
        key: tuple(s for _, s in sorted(steps, key=lambda x: x[0]))
        for key, steps in grouped.items()
    }


class _Engines:
    """An aircraft's thrust (Appendix B4) in the air of an aerodrome."""

    def __init__(self, aircraft: Performance, air: Aerodrome, weight_lb: float, departure: bool):
        self.aircraft = aircraft
        self.air = air
        self.weight = weight_lb
        self.departure = departure
        self.n = aircraft.engines

    def rated(self, rating: str, calibrated_kt: float, true_kt: float, altitude_ft: float) -> float:
        """The corrected net thrust per engine (lb) at the thrust rating ``rating``, at a
        calibrated airspeed and a true airspeed (kt) and an altitude (ft above mean sea
        level): eq. B-9 to B-11 for jet coefficients, eq. B-12 for a propeller's."""
        key = rating_key(rating)
        propeller = self.aircraft.propeller.get(key)
        if propeller is not None:
            return (
                326.0
                * propeller.efficiency
                * propeller.power_hp
                / true_kt
                / self.air.delta(altitude_ft)
            )
        coefficients = self.aircraft.jet.get(key)
        if coefficients is None:
            raise KeyError(
                f"{self.aircraft.name}: no engine coefficients for the thrust rating {rating!r}"
            )
        h = self.air.pressure_altitude(altitude_ft)
        t = self.air.temperature(altitude_ft)
        thrust = coefficients.thrust(calibrated_kt, h, t)
        high = self.aircraft.jet.get(key + "hitemp")
        if high is not None:
            return min(thrust, high.thrust(calibrated_kt, h, t))
        if self.departure:
            c = coefficients
            hot = c.f * calibrated_kt + (c.e + c.h * BREAKPOINT_C) * (1.0 - 0.006 * t) / (
                1.0 - 0.006 * BREAKPOINT_C
            )
            return min(thrust, hot)
        return thrust

    def minimum(self, calibrated_kt: float, altitude_ft: float, r: float) -> float:
        """The minimum reduced thrust per engine (lb; eq. B-13), of a deep cutback."""
        if self.n < 2:
            raise ValueError(
                f"{self.aircraft.name}: the minimum reduced thrust needs two engines or more"
            )
        gradient = ENGINE_OUT_GRADIENT.get(self.n, ENGINE_OUT_GRADIENT[4])
        k = 1.01 if calibrated_kt <= 200.0 else 0.95
        load = self.weight / self.air.delta(altitude_ft)
        return load / (self.n - 1) * (math.sin(math.atan(0.01 * gradient)) / k + r)

    def thrust(
        self,
        step: DepartureStep,
        calibrated_kt: float,
        true_kt: float,
        altitude_ft: float,
        r: float,
    ) -> float:
        """The thrust per engine (lb) on the departure step ``step``."""
        if _is_minimum(step.thrust):
            return self.minimum(calibrated_kt, altitude_ft, r)
        return self.rated(step.thrust, calibrated_kt, true_kt, altitude_ft)

    def surplus(self, thrust_lb: float, altitude_ft: float, r: float) -> float:
        """The thrust left over for climbing and accelerating, as a fraction of the
        weight: N·CNT/(W/δ) - R (eq. B-21, B-24, in straight flight)."""
        return self.n * thrust_lb / (self.weight / self.air.delta(altitude_ft)) - r


def _is_minimum(rating: str) -> bool:
    key = rating_key(rating)
    return key.startswith("minimum") or "reduced" in key


def _kind(name: str) -> str:
    return _key(name)


def departure(
    aircraft: Performance,
    steps: Sequence[DepartureStep],
    weight_lb: float,
    aerodrome: Aerodrome = STANDARD,
    runway_gradient: float = 0.0,
) -> list[Point]:
    """The fixed-point profile (Appendix B6) of a departure of ``aircraft`` at
    ``weight_lb`` (lb) by the procedure ``steps`` from ``aerodrome``, its runway's
    gradient ``runway_gradient`` (positive uphill)."""
    air = aerodrome
    engines = _Engines(aircraft, air, weight_lb, departure=True)
    n, w = aircraft.engines, air.headwind_kt
    elevation = air.elevation_ft
    points: list[Point] = []
    previous: DepartureStep | None = None
    for step in steps:
        kind = _kind(step.kind)
        flaps = aircraft.flap("D", step.flap)
        if kind == "takeoff":
            if flaps.b is None or flaps.speed is None:
                raise ValueError(
                    f"{aircraft.name}: flaps {step.flap!r} have no take-off coefficients"
                )
            calibrated = flaps.speed * math.sqrt(weight_lb)
            speed = calibrated / math.sqrt(air.sigma(elevation))
            thrust = engines.thrust(step, calibrated, speed, elevation, flaps.r)
            standing = engines.thrust(step, 0.0, speed, elevation, flaps.r)
            load = weight_lb / air.delta(elevation)
            roll = flaps.b * air.theta(elevation) * load**2 / (n * thrust)
            roll *= (calibrated - w) ** 2 / (calibrated - REFERENCE_HEADWIND_KT) ** 2
            if runway_gradient:
                # Eq. B-18's average acceleration, its speed in ft/s.
                a = (KNOT_FT_S * speed) ** 2 / (2.0 * roll)
                roll *= a / (a - G_FT_S2 * runway_gradient)
            points += [Point(0.0, 0.0, 0.0, standing), Point(roll, 0.0, speed, thrust)]
            previous = step
            continue
        if not points:
            raise ValueError("a departure starts with its take-off step")
        start = points[-1]
        alt1 = elevation + start.height_ft
        if kind == "climb":
            if step.altitude_ft is None:
                raise ValueError("a climb step needs its end point altitude")
            if step.altitude_ft <= start.height_ft:
                # An acceleration ended above it (B6.1.3).
                continue
            alt2 = elevation + step.altitude_ft
            calibrated = start.speed_kt * math.sqrt(air.sigma(alt1))
            speed = calibrated / math.sqrt(air.sigma(alt2))
            # At the middle of the step: its height, and the true airspeed there of its
            # calibrated airspeed (Volume 3's reference profiles take that, not the
            # rms of the two ends' that Appendix B6.1.2 gives for propellers).
            middle = (alt1 + alt2) / 2.0
            mean = calibrated / math.sqrt(air.sigma(middle))
            thrust = engines.thrust(step, calibrated, speed, alt2, flaps.r)
            k = 1.01 if calibrated <= 200.0 else 0.95
            gamma = math.asin(
                k
                * engines.surplus(
                    engines.thrust(step, calibrated, mean, middle, flaps.r), middle, flaps.r
                )
            )
            gamma *= (calibrated - REFERENCE_HEADWIND_KT) / (calibrated - w)
            end = Point(
                start.distance_ft + (step.altitude_ft - start.height_ft) / math.tan(gamma),
                step.altitude_ft,
                speed,
                thrust,
            )
        elif kind in ("accelerate", "levelaccelerate"):
            end = _accelerate(step, kind == "levelaccelerate", start, flaps, engines)
        elif kind == "level":
            if step.distance_ft is None:
                raise ValueError("a level step needs its distance")
            load = weight_lb / air.delta(alt1)
            thrust = flaps.r / n * load
            end = Point(
                start.distance_ft + step.distance_ft, start.height_ft, start.speed_kt, thrust
            )
        else:
            raise ValueError(f"unknown departure step type {step.kind!r}")
        if (
            kind == "level"
            or previous is None
            or rating_key(previous.thrust) != rating_key(step.thrust)
        ):
            points.append(_transition(kind, start, end, step, flaps, engines))
        points.append(end)
        previous = step
    return points


def _accelerate(
    step: DepartureStep, level: bool, start: Point, flaps: Flaps, engines: _Engines
) -> Point:
    """The end of an accelerating step (Appendix B6.1.3, B6.1.5)."""
    air = engines.air
    if step.speed_kt is None:
        raise ValueError("an accelerate step needs its end point CAS")
    calibrated = step.speed_kt
    alt1 = air.elevation_ft + start.height_ft
    v1 = start.speed_kt

    def run(alt2: float) -> tuple[float, float, float]:
        """The step's length (ft, into 8 kt), its climb gradient and its end's altitude
        for the end altitude ``alt2``."""
        v2 = calibrated / math.sqrt(air.sigma(alt2))
        middle = (alt1 + alt2) / 2.0
        mean = math.sqrt((v1**2 + v2**2) / 2.0)
        thrust = engines.thrust(step, mean * math.sqrt(air.sigma(middle)), mean, middle, flaps.r)
        a_max = G_FT_S2 * engines.surplus(thrust, middle, flaps.r)
        if level:
            gradient = 0.0
        elif step.accelerating_percent is not None:
            gradient = a_max / G_FT_S2 * (1.0 - step.accelerating_percent / 100.0)
        elif step.climb_rate_ft_min is not None:
            gradient = step.climb_rate_ft_min / (60.0 * KNOT_FT_S * mean)
        else:
            raise ValueError("an accelerate step needs its rate of climb or accel percentage")
        if gradient > 0.0 and a_max - gradient * G_FT_S2 < 0.02 * G_FT_S2:
            gradient = a_max / G_FT_S2 - 0.02
            if gradient < 0.01:
                raise ValueError("not enough thrust to accelerate and climb as the step says")
        if a_max - gradient * G_FT_S2 <= 0.0:
            raise ValueError("not enough thrust to accelerate as the step says")
        length = 0.95 * KNOT_FT_S**2 * (v2**2 - v1**2) / (2.0 * (a_max - gradient * G_FT_S2))
        return length, gradient, alt1 + length * gradient / 0.95

    if level:
        length, _, alt2 = run(alt1)
    else:
        alt2 = alt1 + 250.0
        for _ in range(100):
            length, _, estimate = run(alt2)
            done = abs(estimate - alt2) <= 1.0
            alt2 = estimate
            if done:
                break
        else:
            raise ValueError("the accelerate step's height did not converge")
    v2 = calibrated / math.sqrt(air.sigma(alt2))
    mean = math.sqrt((v1**2 + v2**2) / 2.0)
    w = air.headwind_kt
    length *= (mean - w) / (mean - REFERENCE_HEADWIND_KT)
    thrust = engines.thrust(step, calibrated, v2, alt2, flaps.r)
    return Point(start.distance_ft + length, alt2 - air.elevation_ft, v2, thrust)


def _transition(
    kind: str, start: Point, end: Point, step: DepartureStep, flaps: Flaps, engines: _Engines
) -> Point:
    """The point where the thrust has changed, at the start of a departure step
    (Appendix B6.1.6): at most 1000 ft on, the step's gradient climbed."""
    air = engines.air
    length = end.distance_ft - start.distance_ft
    run = min(length / 2.0, TRANSITION_FT)
    height = start.height_ft + (end.height_ft - start.height_ft) / length * run
    altitude = air.elevation_ft + height
    if kind == "climb":
        speed = start.speed_kt * math.sqrt(
            air.sigma(air.elevation_ft + start.height_ft) / air.sigma(altitude)
        )
    elif kind == "level":
        speed = start.speed_kt
    else:
        speed = math.sqrt(start.speed_kt**2 + (end.speed_kt**2 - start.speed_kt**2) / length * run)
    if kind == "level":
        thrust = end.thrust_lb
    else:
        thrust = engines.thrust(
            step, speed * math.sqrt(air.sigma(altitude)), speed, altitude, flaps.r
        )
    return Point(start.distance_ft + run, height, speed, thrust)


_DESCENTS = ("descend", "descenddecel", "descendidle")
_LEVELS = ("level", "leveldecel", "levelidle")
_IDLE = ("descendidle", "levelidle")


def arrival(
    aircraft: Performance,
    steps: Sequence[ApproachStep],
    weight_lb: float | None = None,
    aerodrome: Aerodrome = STANDARD,
) -> list[Point]:
    """The fixed-point profile (Appendix B7) of an arrival of ``aircraft`` at
    ``weight_lb`` (lb; by default 90 % of its greatest landing weight, as Appendix B7
    takes) by the procedure ``steps`` at ``aerodrome``."""
    air = aerodrome
    weight = 0.9 * aircraft.max_landing_weight_lb if weight_lb is None else weight_lb
    engines = _Engines(aircraft, air, weight, departure=False)
    n, w = aircraft.engines, air.headwind_kt
    elevation = air.elevation_ft
    kinds = [_kind(s.kind) for s in steps]
    if "land" not in kinds:
        raise ValueError("an approach procedure needs its land step")
    landing = kinds.index("land")
    land = steps[landing]

    # On the runway (B7.1.5, B7.1.6): touchdown, then the decelerations.
    flaps = aircraft.flap("A", land.flap)
    if flaps.speed is None:
        raise ValueError(f"{aircraft.name}: flaps {land.flap!r} have no landing speed coefficient")
    descents = [s for s, k in zip(steps[:landing], kinds[:landing], strict=True) if k in _DESCENTS]
    if not descents or descents[-1].angle_deg is None:
        raise ValueError("the land step needs a descent before it")
    gamma = math.radians(descents[-1].angle_deg)
    calibrated = flaps.speed * math.sqrt(weight)
    load = weight / air.delta(elevation)
    thrust = load / n * (flaps.r - math.sin(gamma) / 1.03)
    thrust += 1.03 * load * math.sin(gamma) * (w - REFERENCE_HEADWIND_KT) / (n * calibrated)
    touchdown = Point(0.0, 0.0, calibrated / math.sqrt(air.sigma(elevation)), thrust)
    rolls = [
        s
        for s, k in zip(steps[landing + 1 :], kinds[landing + 1 :], strict=True)
        if k == "decelerate"
    ]
    if not rolls:
        raise ValueError("the land step needs a decelerate step after it")
    sigma = air.sigma(elevation)

    def rolling(distance_ft: float, roll: ApproachStep) -> Point:
        return Point(
            distance_ft,
            0.0,
            (roll.speed_kt or 0.0) / math.sqrt(sigma),
            (roll.thrust_percent or 0.0) * aircraft.max_static_thrust_lb / 100.0,
        )

    distance = land.touchdown_roll_ft or 0.0
    ground = [touchdown, rolling(distance, rolls[0])]
    for j, roll in enumerate(rolls):
        if not roll.distance_ft:
            # The last deceleration, of no length, gives no point.
            break
        if j + 1 == len(rolls):
            raise ValueError("a decelerate step of some length needs another after it")
        distance += roll.distance_ft
        ground.append(rolling(distance, rolls[j + 1]))

    # In the air (B7.1.1 to B7.1.4), from the last step back.
    flown: list[list[Point]] = []
    after = touchdown
    for i in range(landing - 1, -1, -1):
        step, kind = steps[i], kinds[i]
        nxt, nxt_kind = steps[i + 1], kinds[i + 1]
        points = _approach_step(step, kind, nxt, nxt_kind, after, aircraft, engines)
        flown.append(points)
        after = points[0]
    return [p for points in reversed(flown) for p in points] + ground


def _approach_step(
    step: ApproachStep,
    kind: str,
    nxt: ApproachStep,
    nxt_kind: str,
    after: Point,
    aircraft: Performance,
    engines: _Engines,
) -> list[Point]:
    """The points of the airborne approach step ``step`` (its first point and, where
    the thrust changes at its end, a second one), ``after`` the first point of the step
    after it."""
    air = engines.air
    n, w = aircraft.engines, air.headwind_kt
    elevation = air.elevation_ft
    if step.altitude_ft is None or step.speed_kt is None:
        raise ValueError(f"the {step.kind} step needs its start altitude and CAS")
    flaps = aircraft.flap("A", step.flap)
    height = step.altitude_ft
    calibrated = step.speed_kt
    speed = calibrated / math.sqrt(air.sigma(elevation + height))
    if kind in _DESCENTS:
        if step.angle_deg is None:
            raise ValueError(f"the {step.kind} step needs its descent angle")
        gamma = math.radians(step.angle_deg)
        if kind == "descendidle":
            height, speed = _idle_descent(step, nxt, nxt_kind, after, gamma, air)
        length = (height - after.height_ft) / math.tan(gamma)
        alone = (
            nxt_kind in _IDLE
            if kind == "descendidle"
            else (
                nxt_kind == kind
                and nxt.flap.strip() == step.flap.strip()
                and nxt.angle_deg == step.angle_deg
            )
        )
    elif kind in _LEVELS:
        if step.distance_ft is None:
            raise ValueError(f"the {step.kind} step needs its distance")
        gamma = 0.0
        length = step.distance_ft
        if kind == "levelidle":
            length = _idle_level(step, nxt, after, air)
            if nxt.altitude_ft is not None and nxt.altitude_ft != step.altitude_ft:
                warnings.warn(
                    f"a Level-Idle step at {step.altitude_ft} ft is followed by a step "
                    f"starting at {nxt.altitude_ft} ft (Appendix B7.1.4 asks for the same)",
                    stacklevel=3,
                )
        alone = (
            nxt_kind in _IDLE
            if kind == "levelidle"
            else (nxt_kind == kind and nxt.flap.strip() == step.flap.strip())
        )
    else:
        raise ValueError(f"unknown approach step type {step.kind!r}")
    altitude = elevation + height
    load = engines.weight / air.delta(altitude)
    if kind == "descend":
        thrust = load / n * (flaps.r - math.sin(gamma) / 1.03)
        thrust += 1.03 * load * math.sin(gamma) * (w - REFERENCE_HEADWIND_KT) / (n * calibrated)
    elif kind == "descenddecel":
        # The speeds along the path over the ground, the headwind w/cos(gamma) of them
        # (eq. B-41 as Volume 3's reference profiles have it).
        path = (height - after.height_ft) / math.sin(gamma)
        along = w / math.cos(gamma)
        a = KNOT_FT_S**2 * ((after.speed_kt - along) ** 2 - (speed - along) ** 2) / (2.0 * path)
        thrust = load / n * (flaps.r * math.cos(gamma) - math.sin(gamma) + a / G_FT_S2)
    elif kind == "level":
        thrust = load / n * flaps.r
    elif kind == "leveldecel":
        a = KNOT_FT_S**2 * ((after.speed_kt - w) ** 2 - (speed - w) ** 2) / (2.0 * length)
        thrust = load / n * (flaps.r + a / G_FT_S2)
    else:
        thrust = engines.rated(
            "IdleApproach", speed * math.sqrt(air.sigma(altitude)), speed, altitude
        )
    first = Point(after.distance_ft - length, height, speed, thrust)
    if alone:
        return [first]
    run = min(length / 2.0, TRANSITION_FT)
    end_height = after.height_ft + run * math.tan(gamma) if kind in _DESCENTS else height
    if kind == "level":
        end_speed = speed
    else:
        end_speed = math.sqrt(after.speed_kt**2 + run / length * (speed**2 - after.speed_kt**2))
    end_altitude = elevation + end_height
    if kind in ("descend", "descenddecel"):
        end_thrust = air.delta(altitude) / air.delta(end_altitude) * thrust
    elif kind in _IDLE:
        end_thrust = engines.rated(
            "IdleApproach", end_speed * math.sqrt(air.sigma(end_altitude)), end_speed, end_altitude
        )
    else:
        end_thrust = thrust
    return [first, Point(after.distance_ft - run, end_height, end_speed, end_thrust)]


def _idle_descent(
    step: ApproachStep, nxt: ApproachStep, nxt_kind: str, after: Point, gamma: float, air: Aerodrome
) -> tuple[float, float]:
    """The height (ft) and true airspeed (kt) a Descend-Idle step starts at (Appendix
    B7.1.2): its deceleration that of the standard atmosphere at a sea-level
    aerodrome."""
    assert step.altitude_ft is not None and step.speed_kt is not None
    height = step.altitude_ft
    speed = step.speed_kt / math.sqrt(air.sigma(air.elevation_ft + height))
    w = KNOT_FT_S * air.headwind_kt
    sin, cos = math.sin(gamma), math.cos(gamma)

    def ground(tas_ft_s: float) -> float:
        return math.sqrt(tas_ft_s**2 - (w * sin) ** 2) - w * cos

    next_height = nxt.altitude_ft if nxt.altitude_ft is not None else 0.0
    next_cas = (
        nxt.speed_kt
        if nxt.speed_kt is not None
        else after.speed_kt * math.sqrt(air.sigma(air.elevation_ft + after.height_ft))
    )
    next_isa = KNOT_FT_S * next_cas / math.sqrt(STANDARD.sigma(next_height))
    first_isa = KNOT_FT_S * step.speed_kt / math.sqrt(STANDARD.sigma(height))
    path = (height - next_height) / sin
    deceleration = (ground(next_isa) ** 2 - ground(first_isa) ** 2) / (2.0 * path)
    next_actual = KNOT_FT_S * after.speed_kt
    if nxt_kind in ("level", "levelidle", "land"):
        # Eq. B-58, the deceleration signed as eq. B-57 gives it (negative).
        path = (height - after.height_ft) / sin
        tas = math.sqrt(
            (math.sqrt(ground(next_actual) ** 2 - 2.0 * path * deceleration) + w * cos) ** 2
            + (w * sin) ** 2
        )
        return height, tas / KNOT_FT_S
    if deceleration == 0.0:
        return height, speed
    path = (ground(next_actual) ** 2 - ground(KNOT_FT_S * speed) ** 2) / (2.0 * deceleration)
    return after.height_ft + path * sin, speed


def _idle_level(step: ApproachStep, nxt: ApproachStep, after: Point, air: Aerodrome) -> float:
    """The length (ft) of a Level-Idle step (Appendix B7.1.4): its deceleration that of
    the standard atmosphere at a sea-level aerodrome (eq. B-73, B-74)."""
    assert (
        step.altitude_ft is not None and step.speed_kt is not None and step.distance_ft is not None
    )
    length = step.distance_ft
    next_cas = (
        nxt.speed_kt
        if nxt.speed_kt is not None
        else after.speed_kt * math.sqrt(air.sigma(air.elevation_ft + after.height_ft))
    )
    w = air.headwind_kt
    cas = step.speed_kt

    def squares(sigma: float) -> float:
        return (KNOT_FT_S * (next_cas / math.sqrt(sigma) - w)) ** 2 - (
            KNOT_FT_S * (cas / math.sqrt(sigma) - w)
        ) ** 2

    standard = squares(STANDARD.sigma(step.altitude_ft))
    if standard == 0.0:
        return length
    deceleration = standard / (2.0 * length)
    return squares(air.sigma(air.elevation_ft + step.altitude_ft)) / (2.0 * deceleration)
