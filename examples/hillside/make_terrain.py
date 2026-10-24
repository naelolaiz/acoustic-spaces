"""Writes hill.asc, the terrain of this example: an Esri ASCII grid of heights.

A ridge 16 to 20 m high running east to west across a square 400 m a side, its crest
along y = 0, its sides falling off as a Gaussian 35 m wide; 41 x 41 heights 10 m
apart, from (-200, -200). Run it: python make_terrain.py (it writes beside itself).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acoustic_spaces.geometry.terrain import HeightGrid, write_asc

SPACING, SIZE = 10.0, 41


def main() -> None:
    x = -200.0 + SPACING * np.arange(SIZE)
    y = -200.0 + SPACING * np.arange(SIZE)
    heights = (16.0 + 4.0 * np.cos(x[None, :] / 70.0)) * np.exp(-((y[:, None] / 35.0) ** 2))
    out = Path(__file__).resolve().with_name("hill.asc")
    write_asc(out, HeightGrid((-200.0, -200.0), SPACING, np.round(heights, 2)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
