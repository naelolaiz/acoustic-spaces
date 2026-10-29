"""How far each array's far-field pattern would be off at each listener, which the scene
avoids by adding up the array's elements one by one.

    python near_field.py [scene.yaml]

A coherent array's far-field pattern, sent out from one point, holds beyond about the
array's length squared over the wavelength: 61 m at 1 kHz for the main arrays, 4.6 m
long. Nearer, its sound has not formed that pattern yet, so the scene has each array
add up its elements one by one, each from where it is, with its own distance, direction
and feed (``near_field: true`` on the array's directivity). This script prints, for
each such array and listener, how much louder (dB) the far-field pattern would make the
direct sound than the elements added up, in each octave band; a dash marks a band
the source's filter leaves out (``eq_db`` below -30 dB, as the subwoofers' upper bands).
The ground and the reflections are left out: they come to both alike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acoustic_spaces import Room
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ


def differences(room: Room) -> dict[str, dict[str, np.ndarray]]:
    """For each array source that adds up its elements and each listener in use, the far-field
    pattern's level above the elements' sum (dB per octave band, NaN where the source's
    filter leaves the band out)."""
    out = {}
    for source in (item for item in room.scene.sources if item.in_use):
        balloon = room.directivity(source.name)
        if getattr(balloon, "near", None) is None:
            continue
        turn = np.array(source.orientation.rotation())
        silent = source.feed_db() - source.gain_db < -30.0
        out[source.name] = {}
        for listener in (item for item in room.scene.listeners if item.in_use):
            # The listener in the source's frame (x forward, y left, z up).
            where = turn.T @ (np.asarray(listener.position) - np.asarray(source.position))
            far = balloon.energy((where / np.linalg.norm(where))[None])[0]
            exact = balloon.near_energy(where[None])[0]
            out[source.name][listener.name] = np.where(silent, np.nan, 10 * np.log10(far / exact))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", nargs="?", default=Path(__file__).with_name("scene.yaml"))
    args = parser.parse_args(argv)
    bands = " ".join(f"{f if f < 1000 else f'{f // 1000}k':>5}" for f in OCTAVE_BANDS_HZ)
    for source, rows in differences(Room.from_yaml(args.scene)).items():
        print(f"\n{source + ', far field above the elements added up (dB)'}")
        print(f"  {'':20} {bands}")
        for listener, db in rows.items():
            print(
                f"  {listener:20} " + " ".join("    -" if np.isnan(d) else f"{d:5.1f}" for d in db)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
