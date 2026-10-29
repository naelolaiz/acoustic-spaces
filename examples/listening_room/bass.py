"""Bass from a loudspeaker against the front wall and from one a metre out.

At low frequencies the reflection off the wall behind a loudspeaker adds to its direct
sound in pressure. From a loudspeaker against the wall it arrives at once and in
phase, and raises the bass; from one a metre out it arrives 6 ms late, half a period
at 80 Hz, and takes from the 63 Hz octave. This prints G at the sofa in the lowest
octave bands from both, with the early paths adding in pressure up to 125 Hz (the
default, ``simulation.early.coherent_max_band_hz``) and in energy, as they did up to
version 2, where the wall makes no difference; for the whole response and for its
early part alone.

    python bass.py [--rays 20000]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, simulate

HERE = Path(__file__).resolve().parent
BANDS = (63, 125, 250)


def levels(room: Room, coherent: int | None, rays: int) -> dict[str, tuple[np.ndarray, ...]]:
    """G of the whole response and of its early part, per source, in ``BANDS``."""
    sim = room.scene.simulation
    settings = sim.model_copy(
        update={
            "early": sim.early.model_copy(update={"coherent_max_band_hz": coherent}),
            "late": sim.late.model_copy(update={"rays": rays}),
        }
    )
    out = {}
    for response in simulate(room, settings=settings):
        early = dataclasses.replace(response, late=np.zeros_like(response.late))
        bands = slice(0, len(BANDS))
        out[response.source] = (response.parameters().g[bands], early.parameters().g[bands])
    return out


#: Its name in the editor's Run demo window.
TITLE = "Bass against the wall and a metre out"


def parser() -> argparse.ArgumentParser:
    """Its arguments, which the editor's Run demo window asks for too."""
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument("--rays", type=int, default=20_000, help="late rays per band")
    return found


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    room = Room.from_yaml(HERE / "scene.yaml")
    header = "".join(f"{f'{hz} Hz':>9}" for hz in BANDS)
    print(f"{'G at the sofa (dB)':<34}{header}{header}")
    print(f"{'':<34}{'whole response':^{9 * len(BANDS)}}{'early part':^{9 * len(BANDS)}}")
    for coherent, title in ((125, "in pressure up to 125 Hz"), (None, "in energy")):
        print(title)
        for source, (whole, early) in levels(room, coherent, args.rays).items():
            row = "".join(f"{g:9.1f}" for g in (*whole, *early))
            print(f"  {source:<32}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
