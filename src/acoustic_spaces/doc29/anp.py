"""Doc 29 with the aircraft of a copy of the ANP database (Volume 2 Appendix G), the
international Aircraft Noise and Performance database, that the user has downloaded from
EASA under its terms.

This package ships no ANP data: the database's terms (EASA's ANP Database Terms and
Conditions, issue 3, 2024) allow its use only for noise contours of an average fleet and
traffic and forbid passing it on, so each user downloads their own copy and accepts
those terms. :func:`read_database` reads the copy's tables from a folder, by the names
of their files: "Aircraft", "NPD", "Spectral classes", "Fixed point profiles", "Jet
engine coefficients", "Propeller engine coefficients", "Aerodynamic coefficients",
"Default weights", "Departure procedural steps" and "Approach procedural steps", in any
case, with spaces or underscores, as CSV or text files separated by commas or
semicolons.

The readers take the fields Volume 2 Appendix G lists, as Volume 3's workbook lays them
out for its hypothetical aircraft; they have not been tried on a download of the
database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .performance import (
    STANDARD,
    Aerodrome,
    ApproachStep,
    DepartureStep,
    Performance,
    arrival,
    as_profile,
    departure,
    read_approach_steps,
    read_departure_steps,
    read_performance,
)
from .tables import (
    Aircraft,
    NpdCurves,
    ProfilePoint,
    SpectralClass,
    _key,
    read_aircraft,
    read_fixed_point_profiles,
    read_npd,
    read_spectral_classes,
)

#: The tables, each with the words its file's name holds (its case, spaces and
#: punctuation aside), the first that matches taken.
TABLES = {
    "npd": ("npd",),
    "spectral_classes": ("spectral",),
    "fixed_point_profiles": ("fixedpoint",),
    "jet": ("jetengine", "jetcoef"),
    "propeller": ("propeller",),
    "aerodynamic": ("aerodynamic",),
    "weights": ("weight",),
    "departure_steps": ("departureprocedur", "departurestep"),
    "approach_steps": ("approachprocedur", "approachstep"),
    "aircraft": ("aircraft",),
}
SUFFIXES = (".csv", ".txt")


def find_tables(folder: str | Path) -> dict[str, Path]:
    """The files of the ANP tables in ``folder`` and its subfolders, by the keys of
    :data:`TABLES`; a table without its file is left out."""
    files = sorted(
        p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES
    )
    found: dict[str, Path] = {}
    for path in files:
        stem = _key(path.stem)
        if "substitut" in stem:
            continue
        for table, words in TABLES.items():
            if any(word in stem for word in words):
                if table in found:
                    raise ValueError(
                        f"two files for the {table} table: {found[table].name} and {path.name}"
                    )
                found[table] = path
                break
    return found


@dataclass(frozen=True)
class Database:
    """The tables of an ANP database (Volume 2 Appendix G), as the readers of
    :mod:`.tables` and :mod:`.performance` give them."""

    aircraft: dict[str, Aircraft]
    npd: dict[tuple[str, str, str], NpdCurves]
    spectral_classes: dict[str, SpectralClass]
    fixed_point_profiles: dict[tuple[str, str, str, str], tuple[ProfilePoint, ...]]
    performance: dict[str, Performance]
    departure_procedures: dict[tuple[str, str, str], tuple[DepartureStep, ...]]
    approach_procedures: dict[tuple[str, str], tuple[ApproachStep, ...]]

    def curves(self, aircraft: str, operation: str) -> tuple[NpdCurves, NpdCurves]:
        """The SEL and LAmax curves of ``aircraft`` on arrival ("A") or departure ("D")."""
        npd = self.aircraft[aircraft].npd
        return self.npd[(npd, "SEL", operation)], self.npd[(npd, "LAmax", operation)]

    def profile(
        self,
        aircraft: str,
        operation: str,
        profile: str,
        stage: str = "1",
        aerodrome: Aerodrome = STANDARD,
    ) -> tuple[ProfilePoint, ...]:
        """The profile ``profile`` of ``aircraft`` on arrival ("A") or departure ("D"), in
        metres: from its procedural steps where the database has them and the
        aircraft's coefficients (a departure at the default weight of its stage
        length ``stage``), else its fixed-point profile."""
        plane = self.performance.get(aircraft)
        if operation == "D" and (aircraft, profile, stage) in self.departure_procedures:
            if plane is None:
                raise KeyError(f"{aircraft}: no performance coefficients")
            steps = self.departure_procedures[(aircraft, profile, stage)]
            return as_profile(departure(plane, steps, plane.weights[("D", stage)], aerodrome))
        if operation == "A" and (aircraft, profile) in self.approach_procedures:
            if plane is None:
                raise KeyError(f"{aircraft}: no performance coefficients")
            approach = self.approach_procedures[(aircraft, profile)]
            return as_profile(arrival(plane, approach, None, aerodrome))
        return self.fixed_point_profiles[(aircraft, operation, profile, stage)]


def read_database(folder: str | Path) -> Database:
    """The ANP database whose tables are the files in ``folder`` (:func:`find_tables`);
    the aircraft table is needed, the others are empty when their files are missing."""
    files = find_tables(folder)
    if "aircraft" not in files:
        raise FileNotFoundError(f"no aircraft table in {folder}")

    def path(table: str) -> Path | None:
        return files.get(table)

    npd, spectral, fixed = path("npd"), path("spectral_classes"), path("fixed_point_profiles")
    departures, approaches = path("departure_steps"), path("approach_steps")
    return Database(
        aircraft=read_aircraft(files["aircraft"]),
        npd=read_npd(npd) if npd else {},
        spectral_classes=read_spectral_classes(spectral) if spectral else {},
        fixed_point_profiles=read_fixed_point_profiles(fixed) if fixed else {},
        performance=read_performance(
            files["aircraft"],
            jet=path("jet"),
            propeller=path("propeller"),
            aerodynamic=path("aerodynamic"),
            weights=path("weights"),
        ),
        departure_procedures=read_departure_steps(departures) if departures else {},
        approach_procedures=read_approach_steps(approaches) if approaches else {},
    )
