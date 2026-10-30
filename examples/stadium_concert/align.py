"""When and how loud each loudspeaker of the stadium arrives at each listener.

    python align.py [scene.yaml] [--rays N]

For every listener it prints when each loudspeaker's direct sound arrives (its feed's
delay included), its level, and the level of all its sound there with the reflections
and the reverberation; the level of all of them together; and the loudspeakers that
would be heard as an echo. The rule of thumb for music used here, on the direct sounds
in the bands from 500 Hz to 4 kHz, where echoes are heard most: a loudspeaker that comes
more than 50 ms after the first one within 10 dB of the loudest, and is itself within
10 dB of the loudest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np

from acoustic_spaces import Room, simulate

#: The octave bands the echo check reads (500 Hz to 4 kHz), the window after the first
#: strong arrival that the ear fuses with it (s), and how far below the loudest a
#: loudspeaker may be and still count (dB).
ECHO_BANDS = slice(3, 7)
ECHO_WINDOW_S = 0.050
ECHO_RANGE_DB = 10.0


class Arrival(NamedTuple):
    source: str
    #: When its direct sound arrives, its feed's delay included (s).
    time_s: float
    #: Its direct sound's level in each octave band (dB; -inf where something blocks it).
    direct: np.ndarray
    #: The level of all its sound in each octave band (dB).
    levels: np.ndarray
    #: Whether it would be heard as an echo of the first strong arrival.
    echo: bool = False


def total(levels: np.ndarray) -> float:
    """The level of the bands (dB) together."""
    with np.errstate(divide="ignore"):
        return float(10.0 * np.log10(np.sum(10.0 ** (np.asarray(levels) / 10.0))))


def arrivals(
    room: Room, rays: int | None = None, seed: int | None = None
) -> dict[str, list[Arrival]]:
    """Each listener's arrivals, earliest first (``rays`` per band, or the scene's; a
    ``seed`` repeats a run's random rays), of the sources and listeners in use."""
    settings = room.scene.simulation
    if rays:
        settings = settings.model_copy(
            update={"late": settings.late.model_copy(update={"rays": rays})}
        )
    if seed is not None:
        settings = settings.model_copy(update={"seed": seed})
    result = simulate(room, settings=settings)
    out = {}
    # Those left out of simulations (``enabled: false``) have no responses.
    for listener in (item for item in room.scene.listeners if item.in_use):
        rows = []
        for source in (item for item in room.scene.sources if item.in_use):
            response = result[source.name, listener.name]
            power = source.power_db()
            # The direct sound is the path without reflections, of a source of unit power
            # with its feed (gain and EQ) already in the energy.
            early = response.early
            energy = early.energy[early.order == 0].sum(axis=0)
            with np.errstate(divide="ignore"):
                direct = power - source.feed_db() + 10.0 * np.log10(energy)
            rows.append((source.name, response.direct_delay_s, direct, response.levels(power)))
        rows.sort(key=lambda row: row[1])
        mid = {row[0]: total(row[2][ECHO_BANDS]) for row in rows}
        loud = max(mid.values()) - ECHO_RANGE_DB
        first = next(time for name, time, *_ in rows if mid[name] >= loud)
        out[listener.name] = [
            Arrival(*row, echo=row[1] - first > ECHO_WINDOW_S and mid[row[0]] >= loud)
            for row in rows
        ]
    return out


#: Its name in the editor's Run demo window.
TITLE = "When and how loud each loudspeaker arrives"


def parser() -> argparse.ArgumentParser:
    """Its arguments, which the editor's Run demo window asks for too."""
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument(
        "scene", nargs="?", default=Path(__file__).with_name("scene.yaml"), help="the scene"
    )
    found.add_argument("--rays", type=int, help="rays per band (fewer is quicker)")
    return found


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for listener, rows in arrivals(Room.from_yaml(args.scene), args.rays).items():
        everything = total([total(row.levels) for row in rows])
        print(f"\n{listener}: {everything:.1f} dB from all of them")
        print(f"  {'':18} {'arrives':>9} {'direct':>8} {'in all':>8}")
        for row in rows:
            mark = "  echo" if row.echo else ""
            print(
                f"  {row.source:18} {1000 * row.time_s:6.1f} ms {total(row.direct):5.1f} dB"
                f" {total(row.levels):5.1f} dB{mark}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
