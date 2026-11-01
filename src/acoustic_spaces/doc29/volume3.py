"""The reference cases of Doc 29 Volume 3 Part 1 (5th edition, 2026): three
hypothetical aircraft that rely on no ANP data, a rear-engined jet (JETF), a
wing-engined jet (JETW) and a propeller aircraft (PROP), each arriving and departing
on a straight and a curved route at a hypothetical aerodrome, and the sound exposure
levels Volume 3 gives at its receptors.

The tables are Volume 3's (its Appendix A and B, tables A-1 and A-6 to A-12 and B-1, as
its workbook gives them), read as the ANP database's are.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from .noise import Event, event, impedance_db
from .path import Segment, Track, flight_path, lowest_approach_node
from .tables import (
    Aircraft,
    NpdCurves,
    ProfilePoint,
    SpectralClass,
    read_aircraft,
    read_fixed_point_profiles,
    read_npd,
    read_rows,
    read_spectral_classes,
)

#: Where the tables are.
DATA = Path(__file__).with_name("volume3")


@cache
def aircraft() -> dict[str, Aircraft]:
    """The three reference aircraft (table A-1), by identifier."""
    return read_aircraft(DATA / "aircraft.csv")


@cache
def npd() -> dict[tuple[str, str, str], NpdCurves]:
    """Their NPD curves (table A-7), by (aircraft, "SEL" or "LAmax", "A" or "D")."""
    return read_npd(DATA / "npd.csv")


@cache
def spectral_classes() -> dict[str, SpectralClass]:
    """Their spectral classes (table A-8), by identifier."""
    return read_spectral_classes(DATA / "spectral_classes.csv")


@cache
def profiles() -> dict[tuple[str, str], tuple[ProfilePoint, ...]]:
    """Their fixed-point profiles (table A-6), by (aircraft, "A" or "D")."""
    return {
        (name, mode): points
        for (name, mode, _, _), points in read_fixed_point_profiles(
            DATA / "fixed_point_profiles.csv"
        ).items()
    }


@cache
def routes() -> dict[str, tuple[tuple[float, float], ...]]:
    """The four routes' ground tracks (table A-11; m), by identifier: "AC" and "AS"
    the curved and straight arrivals, "DC" and "DS" the departures."""
    found: dict[str, list[tuple[float, float]]] = {}
    for row in read_rows(DATA / "routes.csv"):
        found.setdefault(row["route_ID"], []).append(
            (float(row["X coordinate (m)"]), float(row["Y coordinate (m)"]))
        )
    return {key: tuple(points) for key, points in found.items()}


@cache
def runway() -> tuple[tuple[float, float], tuple[float, float]]:
    """The runway's start of roll and end (table A-10; m)."""
    row = read_rows(DATA / "runway.csv")[0]
    return (
        (float(row["SOR X coordinate (m)"]), float(row["SOR Y coordinate (m)"])),
        (float(row["End X coordinate (m)"]), float(row["End Y coordinate (m)"])),
    )


@cache
def receptors() -> dict[str, tuple[float, float, float]]:
    """The 18 receptors (table A-12; m), by identifier."""
    return {
        row["receptor_ID"]: (
            float(row["X coordinate (m)"]),
            float(row["Y coordinate (m)"]),
            float(row["Height (m)"]),
        )
        for row in read_rows(DATA / "receptors.csv")
    }


@cache
def sel_results() -> dict[tuple[str, str], float]:
    """Volume 3's sound exposure levels (table B-1; dB(A)), by (case, receptor)."""
    return {
        (row["case_ID"], row["receptor_ID"]): float(row["SEL(dB)"])
        for row in read_rows(DATA / "sel_results.csv")
    }


@cache
def impedance() -> float:
    """The impedance adjustment at the aerodrome's temperature and pressure (dB; table
    A-9, the standard atmosphere's: 0.074 dB)."""
    row = read_rows(DATA / "meteorological.csv")[0]
    pressure_kpa = float(row["Pressure (inHg)"]) * 3.386389
    return impedance_db(float(row["Temperature (degC)"]), pressure_kpa)


@dataclass(frozen=True)
class Case:
    """A reference case: an aircraft, an arrival ("A") or a departure ("D"), and a
    route, curved ("C") or straight ("S"); its identifier is Volume 3's, "JETFDC" for
    instance."""

    aircraft: str
    operation: str
    route: str

    @classmethod
    def named(cls, name: str) -> Case:
        """The case of Volume 3's identifier ``name``."""
        return cls(aircraft=name[:4], operation=name[4], route=name[5])

    @property
    def name(self) -> str:
        return f"{self.aircraft}{self.operation}{self.route}"

    @property
    def departure(self) -> bool:
        return self.operation == "D"

    def track(self) -> Track:
        """Its ground track: a departure's from the start of roll, an arrival's from the
        landing threshold, the runway's start (where the arrivals land, as the
        departures leave)."""
        sor, _ = runway()
        return Track.through(routes()[self.operation + self.route], sor)

    def segments(self) -> list[Segment]:
        """Its flight path's segments."""
        profile = profiles()[(self.aircraft, self.operation)]
        return flight_path(
            profile,
            self.track(),
            departure=self.departure,
            threshold_m=0.0 if self.departure else lowest_approach_node(profile),
        )

    def event(self, points: np.ndarray) -> Event:
        """Its levels at ``points`` (x, y, z in m, one a row)."""
        plane = aircraft()[self.aircraft]
        curves = npd()
        return event(
            plane,
            curves[(plane.npd, "SEL", self.operation)],
            curves[(plane.npd, "LAmax", self.operation)],
            self.segments(),
            points,
            impedance(),
        )


#: The twelve cases, as Volume 3's table 3-1 lists them.
CASES = tuple(
    Case(aircraft=a, operation=o, route=r)
    for a in ("JETF", "JETW", "PROP")
    for o in ("A", "D")
    for r in ("C", "S")
)
