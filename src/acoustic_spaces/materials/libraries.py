"""Material libraries of other room acoustics programs: CATT-Acoustic and ODEON.

A scene's material can take its coefficients from an entry of such a library, with
``file`` naming the library and ``entry`` the material in it, and
``acoustic-spaces materials --file`` lists a library's entries.

CATT-Acoustic writes materials in its geometry and library files as
``ABS name = <a125 a250 a500 a1k a2k a4k> L <s125 s250 s500 s1k s2k s4k> {r g b}``:
absorption and scattering in percent from 125 Hz to 4 kHz, with values at 8 and 16 kHz
optional. ``L <...>`` and the colour may be left out, ``;`` starts a comment, and
``ABS name = other`` gives a material the coefficients of another in the same file.
Scattering that CATT estimates itself (``L <estimate(...)>``) is not read.

ODEON's library (``Material.Li8``) has each material on two lines: its number and
description, then eight absorption coefficients from 63 Hz to 8 kHz. Libraries of six
values (125 Hz to 4 kHz) are read too. ODEON gives scattering to surfaces, not to
materials, so these entries have none.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

#: Centre frequencies (Hz) of CATT-Acoustic's bands, 125 Hz to 16 kHz.
CATT_BANDS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
#: ODEON's bands, 63 Hz to 8 kHz; a library of six values starts at 125 Hz.
ODEON_BANDS_HZ = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

_CATT = re.compile(
    r"\bABS\s+(?P<name>[^\s=;]+)\s*=\s*"
    r"(?:<(?P<absorption>[^>]*)>|(?P<alias>[^\s<{;]+))"
    r"(?:\s*L\s*<(?P<scattering>[^>]*)>)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LibraryMaterial:
    """One material of a library."""

    #: Its name: CATT's material name, or ODEON's description.
    name: str
    #: ODEON's material number; None in CATT libraries.
    number: int | None
    #: Centre frequencies (Hz) of the values.
    frequencies: tuple[float, ...]
    #: Random-incidence absorption coefficients, from 0 to 1.
    absorption: tuple[float, ...]
    #: Scattering coefficients at the same frequencies, or None.
    scattering: tuple[float, ...] | None = None

    @property
    def label(self) -> str:
        """How the material is named in messages: with its number, if it has one."""
        return self.name if self.number is None else f"{self.number} {self.name}"


def _text(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")  # ODEON is a Windows program


def _numbers(text: str) -> list[float] | None:
    """The numbers of a line, or None when it holds anything else."""
    try:
        values = [float(t.replace(",", ".")) for t in text.split()]
    except ValueError:
        return None
    return values if values and all(math.isfinite(v) for v in values) else None


def read_library(path: str | Path) -> list[LibraryMaterial]:
    """Every material of a CATT-Acoustic or ODEON library, in file order.

    The format is found from the contents: ``ABS`` definitions make a CATT file, and
    numbered descriptions followed by a line of coefficients an ODEON one.
    """
    path = Path(path)
    text = _text(path)
    without_comments = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    if _CATT.search(without_comments):
        return _read_catt(path, without_comments)
    materials = _read_odeon(path, text)
    if not materials:
        raise ValueError(
            f"{path}: no materials found; expected CATT-Acoustic ABS definitions or an "
            "ODEON library (a material's number and description, then its coefficients)"
        )
    return materials


def _read_catt(path: Path, text: str) -> list[LibraryMaterial]:
    found: dict[str, LibraryMaterial] = {}
    aliases: list[tuple[str, str]] = []
    for match in _CATT.finditer(text):
        name = match["name"]
        if match["alias"] is not None:
            aliases.append((name, match["alias"]))
            continue
        absorption = _numbers(match["absorption"])
        if absorption is None or not 6 <= len(absorption) <= len(CATT_BANDS_HZ):
            raise ValueError(
                f"{path}: material {name!r}: expected 6 to 8 absorption values in percent "
                f"(125 Hz to 4, 8 or 16 kHz), got <{match['absorption'].strip()}>"
            )
        # Scattering that CATT estimates (estimate(...)) is no list of numbers: none.
        scattering = None if match["scattering"] is None else _numbers(match["scattering"])
        if scattering is not None and not 6 <= len(scattering) <= len(CATT_BANDS_HZ):
            raise ValueError(
                f"{path}: material {name!r}: expected 6 to 8 scattering values in percent, "
                f"got <{match['scattering'].strip()}>"
            )
        found[name.lower()] = LibraryMaterial(
            name=name,
            number=None,
            frequencies=CATT_BANDS_HZ[: len(absorption)],
            absorption=tuple(v / 100.0 for v in absorption),
            scattering=None if scattering is None else _held(scattering, len(absorption)),
        )
    for name, other in aliases:
        if other.lower() not in found:
            raise ValueError(
                f"{path}: material {name!r} is {other!r}, which the file does not define "
                "(it may come from a CATT library: give that file instead)"
            )
        found[name.lower()] = _renamed(found[other.lower()], name)
    return list(found.values())


def _held(percent: list[float], count: int) -> tuple[float, ...]:
    """Scattering in percent as shares at as many bands as the absorption: cut, or the
    last value held."""
    values = [v / 100.0 for v in percent]
    return tuple((values + values[-1:] * count)[:count])


def _renamed(material: LibraryMaterial, name: str) -> LibraryMaterial:
    return LibraryMaterial(
        name, material.number, material.frequencies, material.absorption, material.scattering
    )


def _read_odeon(path: Path, text: str) -> list[LibraryMaterial]:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    out = []
    k = 0
    while k < len(lines) - 1:
        head = re.match(r"(-?\d+)\s+(.*\S)", lines[k])
        values = _numbers(lines[k + 1])
        if head is None or _numbers(lines[k]) is not None or values is None:
            k += 1
            continue
        if len(values) not in (6, len(ODEON_BANDS_HZ)):
            raise ValueError(
                f"{path}: material {head[1]} {head[2]!r}: expected 8 absorption values "
                f"(63 Hz to 8 kHz) or 6 (125 Hz to 4 kHz), got {len(values)}"
            )
        bands = ODEON_BANDS_HZ if len(values) == len(ODEON_BANDS_HZ) else ODEON_BANDS_HZ[1:7]
        out.append(LibraryMaterial(head[2], int(head[1]), bands, tuple(values)))
        k += 2
    return out


def find_material(path: str | Path, entry: str | int) -> LibraryMaterial:
    """The material of a library named ``entry``: a name or description (any case), or
    an ODEON material's number."""
    materials = read_library(path)
    key = str(entry).strip()
    for material in materials:
        if material.name.lower() == key.lower() or str(material.number) == key:
            return material
    known = ", ".join(m.label for m in materials[:12])
    more = f" and {len(materials) - 12} more" if len(materials) > 12 else ""
    raise ValueError(f"{path}: no material {key!r}; it has {known}{more}")


__all__ = [
    "CATT_BANDS_HZ",
    "ODEON_BANDS_HZ",
    "LibraryMaterial",
    "find_material",
    "read_library",
]
