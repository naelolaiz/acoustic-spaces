"""Built-in material database.

The first data set is the materials table shipped with pyroomacoustics (MIT
license, see ``data/LICENSE-pyroomacoustics.txt``): 90 absorption entries in octave
bands from 125 Hz and 7 scattering entries. That table does not name its original
source for most entries; check values that matter for a project against the
manufacturer's data or a measured database such as BRAS.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

from acoustic_spaces.materials.bands import to_octave_bands

Kind = Literal["absorption", "scattering"]


@dataclass(frozen=True)
class MaterialEntry:
    id: str
    kind: Kind
    category: str
    description: str
    frequencies: tuple[float, ...]
    coefficients: tuple[float, ...]
    source: str

    def octave_bands(self) -> tuple[list[float], list[int]]:
        """Coefficients on ``OCTAVE_BANDS_HZ`` and the octaves that had to be extrapolated."""
        return to_octave_bands(self.frequencies, self.coefficients)


class MaterialDatabase:
    def __init__(self, entries: list[MaterialEntry]) -> None:
        self._entries = {(e.kind, e.id): e for e in entries}

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self, kind: Kind | None = None) -> list[MaterialEntry]:
        return [e for (k, _), e in self._entries.items() if kind is None or k == kind]

    def get(self, id: str, kind: Kind = "absorption") -> MaterialEntry:
        try:
            return self._entries[(kind, id)]
        except KeyError:
            names = [e.id for e in self.entries(kind)]
            close = difflib.get_close_matches(id, names, n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise KeyError(f"no {kind} material named {id!r}{hint}") from None

    def search(self, text: str = "", kind: Kind | None = None) -> list[MaterialEntry]:
        words = text.lower().split()
        return [
            e
            for e in self.entries(kind)
            if all(w in f"{e.id} {e.category} {e.description}".lower() for w in words)
        ]


@lru_cache(maxsize=1)
def builtin_database() -> MaterialDatabase:
    data = json.loads(
        resources.files("acoustic_spaces.materials")
        .joinpath("data/pyroomacoustics_materials.json")
        .read_text(encoding="utf-8")
    )
    centre = data["center_freqs"]
    entries = []
    for kind in ("absorption", "scattering"):
        for category, materials in data[kind].items():
            for id, item in materials.items():
                coeffs = tuple(float(c) for c in item["coeffs"])
                entries.append(
                    MaterialEntry(
                        id=id,
                        kind=kind,
                        category=category,
                        description=item.get("description", ""),
                        frequencies=tuple(float(f) for f in centre[: len(coeffs)]),
                        coefficients=coeffs,
                        source=f"pyroomacoustics materials database: {id}",
                    )
                )
    return MaterialDatabase(entries)
