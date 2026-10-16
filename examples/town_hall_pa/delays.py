"""The town hall's speech transmission index with and without its delay loudspeakers.

    python delays.py [scene.yaml]

Maps the STI over the seats twice, from the two columns alone and from all four
loudspeakers, and prints its median and its lowest value at the front, in the middle and
at the back of the hall.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acoustic_spaces.engine import coverage_map

#: Parts of the hall by distance from the stage wall (x, in metres).
ZONES = {"front, to 10 m": (0, 10), "middle, 10 to 18 m": (10, 18), "back, from 18 m": (18, 28)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", nargs="?", default=Path(__file__).with_name("scene.yaml"))
    args = parser.parse_args()
    columns = coverage_map(args.scene, sources=["column_left", "column_right"]).get()
    everything = coverage_map(args.scene).get()
    x = everything.grid.points[:, 0]
    print(f"{'STI (median, lowest)':20} {'columns alone':>15} {'with the delays':>17}")
    for zone, (start, end) in ZONES.items():
        seats = (x >= start) & (x < end)
        cells = [
            f"{np.nanmedian(m.value('sti')[seats]):.2f}, {np.nanmin(m.value('sti')[seats]):.2f}"
            for m in (columns, everything)
        ]
        print(f"{zone:20} {cells[0]:>15} {cells[1]:>17}")


if __name__ == "__main__":
    main()
