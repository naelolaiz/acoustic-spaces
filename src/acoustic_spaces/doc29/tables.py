"""Doc 29's aircraft data read from tables laid out as the ANP database's (Doc 29
Volume 2 Appendix G, as Volume 3's workbook lays out its reference aircraft): the
aircraft, their NPD curves, their spectral classes and their fixed-point profiles.

A table is a CSV file with a header row; its separator may be a comma or a semicolon.
Columns are found by their names, whatever their case, spaces and punctuation, and a
length, a speed or a distance between NPD points is read in the unit its column's name
gives: ``(m)`` or ``(ft)``, ``(m/s)`` or ``(kt)``.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: A foot (m), a knot (m/s) and a nautical mile (m).
FOOT_M = 0.3048
KNOT_M_S = 1852.0 / 3600.0
NAUTICAL_MILE_M = 1852.0
#: The distances of the NPD tables (ft), 200 to 25000 ft (Volume 2 Appendix G).
NPD_FEET = (200.0, 400.0, 630.0, 1000.0, 2000.0, 4000.0, 6300.0, 10000.0, 16000.0, 25000.0)
#: The third-octave bands of a spectral class (Hz), 50 Hz to 10 kHz.
SPECTRAL_BANDS_HZ = (
    50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
)  # fmt: skip


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """The rows of the CSV table ``path``, each a dict from its column's name to its cell
    (stripped)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    header = [cell.strip() for cell in next(reader)]
    return [
        {name: cell.strip() for name, cell in zip(header, row, strict=False)}
        for row in reader
        if any(cell.strip() for cell in row)
    ]


def _column(names: Iterable[str], *wanted: str) -> str:
    """The first of ``names`` whose name, its case and punctuation aside, starts with
    one of ``wanted``."""
    names = list(names)
    for want in wanted:
        for name in names:
            if _key(name).startswith(_key(want)):
                return name
    raise KeyError(f"no column {wanted[0]!r} among {names}")


def _length_m(name: str) -> float:
    """The metres in one unit of the column ``name``: (m) or (ft)."""
    return FOOT_M if "(ft" in name.lower() else 1.0


def _speed_m_s(name: str) -> float:
    """The m/s in one unit of the column ``name``: (m/s) or (kt)."""
    return KNOT_M_S if "(kt" in name.lower() else 1.0


@dataclass(frozen=True)
class Aircraft:
    """An aircraft of the ANP database's aircraft table (Volume 2 Appendix G2): what
    the noise calculation needs of it."""

    #: Its identifier, its description, its engines ("Jet", "Turboprop" or "Piston")
    #: and how many it has.
    name: str
    description: str
    engine_type: str
    engines: int
    #: The identifier of its NPD curves, and the power parameter they are given against.
    npd: str
    power_parameter: str
    #: The identifiers of its spectral classes on approach and on departure.
    approach_spectrum: str
    departure_spectrum: str
    #: Its lateral directivity, "Fuselage", "Wing" or "Prop": the engine installation
    #: correction of Volume 2 section 4.5.3.
    installation: str

    @property
    def jet(self) -> bool:
        """Whether its engines are jets (the start-of-roll directivity of jets)."""
        return self.engine_type.lower().startswith("jet")


def read_aircraft(path: str | Path) -> dict[str, Aircraft]:
    """The aircraft of the aircraft table ``path``, by identifier."""
    rows = read_rows(path)
    if not rows:
        return {}
    c = rows[0].keys()
    name = _column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")
    found = {}
    for row in rows:
        found[row[name]] = Aircraft(
            name=row[name],
            description=row[_column(c, "Description")],
            engine_type=row[_column(c, "Engine Type")],
            engines=int(float(row[_column(c, "Number of Engines")])),
            npd=row[_column(c, "NPD Identifier", "NPD ID", "NPD_ID")],
            power_parameter=row[_column(c, "Power Parameter")],
            approach_spectrum=row[_column(c, "Approach Spectral Class")],
            departure_spectrum=row[_column(c, "Departure Spectral Class")],
            installation=row[_column(c, "Lateral Directivity")],
        )
    return found


@dataclass(frozen=True)
class NpdCurves:
    """One set of NPD curves (noise-power-distance, Volume 2 section 4.2): a level
    (dB) at each of :data:`NPD_FEET` for each power, interpolated linearly in power
    (eq. 4-3) and in the logarithm of the distance (eq. 4-4), and extrapolated from the
    last two curves or distances beyond them (eq. 4-5)."""

    powers: tuple[float, ...]
    #: The levels, a row for each power (as many as :attr:`powers`), a column for each
    #: distance.
    levels: tuple[tuple[float, ...], ...]

    def level(self, power: np.ndarray | float, distance_m: np.ndarray | float) -> np.ndarray:
        """The level at ``power`` and ``distance_m`` (m), each an array or a number."""
        powers = np.asarray(self.powers, dtype=float)
        table = np.asarray(self.levels, dtype=float)
        x = np.log10(np.asarray(distance_m, dtype=float) / FOOT_M)
        xs = np.log10(NPD_FEET)
        j = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
        t = (x - xs[j]) / (xs[j + 1] - xs[j])
        p = np.asarray(power, dtype=float)
        if len(powers) == 1:
            row = table[0]
            return np.asarray(row[j] + t * (row[j + 1] - row[j]))
        i = np.clip(np.searchsorted(powers, p) - 1, 0, len(powers) - 2)
        u = (p - powers[i]) / (powers[i + 1] - powers[i])
        low = table[i, j] + t * (table[i, j + 1] - table[i, j])
        high = table[i + 1, j] + t * (table[i + 1, j + 1] - table[i + 1, j])
        return np.asarray(low + u * (high - low))


def read_npd(path: str | Path) -> dict[tuple[str, str, str], NpdCurves]:
    """The NPD curves of the NPD table ``path``, by (NPD identifier, noise descriptor,
    operation mode): ("JETF", "SEL", "D") for instance, the descriptor "SEL" or "LAmax"
    and the mode "A" (arrival) or "D" (departure)."""
    rows = read_rows(path)
    if not rows:
        return {}
    c = list(rows[0].keys())
    name = _column(c, "Aircraft Identifier", "NPD Identifier", "NPD ID", "NPD_ID")
    metric = _column(c, "Noise Descriptor", "Noise Metric")
    mode = _column(c, "Operation Mode", "Op Mode")
    power = _column(c, "Power Setting")
    distances = [_column(c, f"L_{int(feet)}", f"L{int(feet)}") for feet in NPD_FEET]
    grouped: dict[tuple[str, str, str], list[tuple[float, tuple[float, ...]]]] = {}
    for row in rows:
        key = (row[name], _descriptor(row[metric]), row[mode][:1].upper())
        levels = tuple(float(row[d]) for d in distances)
        grouped.setdefault(key, []).append((float(row[power]), levels))
    return {
        key: NpdCurves(
            powers=tuple(p for p, _ in sorted(curves)),
            levels=tuple(levels for _, levels in sorted(curves)),
        )
        for key, curves in grouped.items()
    }


def _descriptor(text: str) -> str:
    key = _key(text)
    if key in ("sel", "lae"):
        return "SEL"
    if key in ("lamax", "lasmax", "lmax"):
        return "LAmax"
    return text


@dataclass(frozen=True)
class SpectralClass:
    """A spectral class (Volume 2 Appendix D, G): the spectrum, in third octaves of
    :data:`SPECTRAL_BANDS_HZ`, of an aircraft's sound 1000 ft (305 m) under it, its
    level 70 dB at 1 kHz."""

    name: str
    mode: str
    description: str
    levels: tuple[float, ...]


def read_spectral_classes(path: str | Path) -> dict[str, SpectralClass]:
    """The spectral classes of the table ``path``, by identifier."""
    rows = read_rows(path)
    if not rows:
        return {}
    c = list(rows[0].keys())
    name = _column(c, "Spectral Class Identifier", "Spectral Class ID")
    bands = [_column(c, f"L_{hz}Hz", f"L{hz}") for hz in SPECTRAL_BANDS_HZ]
    return {
        row[name]: SpectralClass(
            name=row[name],
            mode=row[_column(c, "Operation Mode", "Op Mode")],
            description=row[_column(c, "Description")],
            levels=tuple(float(row[b]) for b in bands),
        )
        for row in rows
    }


@dataclass(frozen=True)
class ProfilePoint:
    """A point of a fixed-point profile: the distance along the ground track (m; from
    the start of roll on departure, to touchdown on approach, negative before it), the
    height above the runway (m), the true airspeed (m/s) and the power (the NPD
    curves' parameter, per engine)."""

    distance_m: float
    height_m: float
    speed_m_s: float
    power: float


def read_fixed_point_profiles(
    path: str | Path,
) -> dict[tuple[str, str, str, str], tuple[ProfilePoint, ...]]:
    """The fixed-point profiles of the table ``path``, by (aircraft, operation mode "A"
    or "D", profile identifier, stage length), each its points in order."""
    rows = read_rows(path)
    if not rows:
        return {}
    c = list(rows[0].keys())
    name = _column(c, "Aircraft Identifier", "ACFT_ID", "Aircraft ID")
    mode = _column(c, "Operation Mode", "Op Mode")
    profile = _column(c, "Profile Identifier", "Profile ID")
    stage = _column(c, "Stage Length")
    number = _column(c, "Point Number")
    distance = _column(c, "Distance")
    height = _column(c, "Altitude", "Height")
    speed = _column(c, "True Airspeed", "Speed")
    power = _column(c, "Corrected Net Thrust", "Power")
    grouped: dict[tuple[str, str, str, str], list[tuple[int, ProfilePoint]]] = {}
    for row in rows:
        key = (row[name], row[mode][:1].upper(), row[profile], row[stage])
        point = ProfilePoint(
            distance_m=float(row[distance]) * _length_m(distance),
            height_m=float(row[height]) * _length_m(height),
            speed_m_s=float(row[speed]) * _speed_m_s(speed),
            power=float(row[power]),
        )
        grouped.setdefault(key, []).append((int(float(row[number])), point))
    return {key: tuple(p for _, p in sorted(points)) for key, points in grouped.items()}


def points_m(rows: Sequence[dict[str, str]], x: str, y: str) -> list[tuple[float, float]]:
    """The points (m) of the columns ``x`` and ``y`` of ``rows``, in the unit of their
    names."""
    return [(float(r[x]) * _length_m(x), float(r[y]) * _length_m(y)) for r in rows]
