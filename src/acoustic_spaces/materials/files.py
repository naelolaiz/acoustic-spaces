"""Reading coefficient tables from files.

The format is the one the BRAS benchmark uses for its surface descriptions: a CSV with
three rows (centre frequencies in Hz, rising, then absorption coefficients and
scattering coefficients, a value per band in each), at any band resolution. The
scattering row is optional. Commas, semicolons or whitespace separate values; lines
with no value (``,,,`` as spreadsheets write them) or starting with ``#`` are ignored,
and so are lines of text before the first row.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

_SPLIT = re.compile(r"[,;\s]+")
#: The layout a coefficient CSV is expected in, for the messages that refuse one.
_LAYOUT = (
    "expected one row of centre frequencies in Hz, rising, one of absorption "
    "coefficients and optionally one of scattering coefficients, a value per band in "
    "each (turn a table laid out in columns into rows)"
)


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
            if not tokens:
                continue
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
    if len(rows) > 3:
        raise ValueError(f"{path}: {len(rows)} rows of numbers; {_LAYOUT}")
    frequencies, absorption = rows[0], rows[1]
    if frequencies[0] <= 0 or any(
        b <= a for a, b in zip(frequencies, frequencies[1:], strict=False)
    ):
        shown = ", ".join(f"{f:g}" for f in frequencies[:6])
        shown += ", ..." if len(frequencies) > 6 else ""
        raise ValueError(
            f"{path}: the first row ({shown}) is not one of rising frequencies; {_LAYOUT}"
        )
    scattering = rows[2] if len(rows) > 2 else None
    for name, row in (("absorption", absorption), ("scattering", scattering)):
        if row is not None and len(row) != len(frequencies):
            raise ValueError(f"{path}: {len(row)} {name} values for {len(frequencies)} frequencies")
    return frequencies, absorption, scattering
