"""Reading coefficient tables from files.

The format is the one the BRAS benchmark uses for its surface descriptions: a CSV with
three rows (centre frequencies in Hz, absorption coefficients, scattering
coefficients), at any band resolution. The scattering row is optional. Commas,
semicolons or whitespace separate values; lines starting with ``#`` are ignored.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

_SPLIT = re.compile(r"[,;\s]+")


def read_coefficients_csv(path: str | Path) -> tuple[list[float], list[float], list[float] | None]:
    """Return (frequencies, absorption, scattering or None) as read from ``path``."""
    path = Path(path)
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [t for t in _SPLIT.split(line) if t]
            try:
                values = [float(t) for t in tokens]
            except ValueError:
                if not rows:
                    continue  # a text header line
                raise ValueError(f"{path}:{lineno}: expected numbers, got {line!r}") from None
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{path}:{lineno}: expected finite numbers, got {line!r}")
            rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"{path}: expected rows of frequencies and absorption coefficients")
    frequencies, absorption = rows[0], rows[1]
    scattering = rows[2] if len(rows) > 2 else None
    for name, row in (("absorption", absorption), ("scattering", scattering)):
        if row is not None and len(row) != len(frequencies):
            raise ValueError(f"{path}: {len(row)} {name} values for {len(frequencies)} frequencies")
    return frequencies, absorption, scattering
