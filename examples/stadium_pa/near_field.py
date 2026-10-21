"""How far the arrays' far-field patterns are from the exact sum of their boxes at each
listener.

    python near_field.py [scene.yaml]

A coherent array in a scene sends out its far-field pattern from one point, which holds
beyond about the array's length squared over the wavelength: 61 m at 1 kHz for the main
arrays, 4.6 m long, so the scene's warnings name the listeners nearer than that. This
script adds up each array's elements one by one in free field, each with its own
distance, direction and feed, and prints how much louder (dB) the far-field pattern is
than that sum at each listener, in each octave band. The ground and the reflections are
left out: they come to both alike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acoustic_spaces import Room
from acoustic_spaces.directivity import FirstOrder, load_directivity, read_array
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ


class _Array:
    """A source's array, in the source's frame (x forward, y left, z up)."""

    def __init__(self, path: Path) -> None:
        spec = read_array(path)
        elements = spec.layout()
        self.positions = np.array([e.position for e in elements], dtype=float)
        self.turns = np.array([e.orientation.rotation() for e in elements], dtype=float)
        self.gains = np.array([10 ** (e.gain_db / 20) * (-1 if e.invert else 1) for e in elements])
        self.delays = np.array([e.delay_ms / 1000 for e in elements])
        self.element = load_directivity(spec.element)

    def amplitudes(self, directions: np.ndarray) -> np.ndarray:
        """(elements, bands) each element's amplitude towards its direction (elements, 3):
        first-order patterns with their sign, others by magnitude."""
        local = np.einsum("nk,nkj->nj", directions, self.turns)
        if isinstance(self.element, FirstOrder):
            a = np.asarray(self.element.a)
            return a + (1.0 - a) * local[:, :1]
        return np.sqrt(self.element.energy(local))

    def energy(self, where: np.ndarray, far: bool, c: float) -> np.ndarray:
        """(bands,) the elements' energy at ``where`` (source frame), each band's mean over
        its frequencies: from each element exactly, or from the far-field pattern."""
        if far:
            r = np.linalg.norm(where)
            towards = np.repeat((where / r)[None], len(self.positions), axis=0)
            spread = np.full(len(self.positions), r)
            # How much sooner each element is heard than the array's centre (s).
            lead = self.positions @ (where / r) / c
        else:
            offsets = where[None] - self.positions
            spread = np.linalg.norm(offsets, axis=1)
            towards = offsets / spread[:, None]
            lead = -spread / c
        amplitude = self.amplitudes(towards) * (self.gains / spread)[:, None]
        out = np.zeros(len(OCTAVE_BANDS_HZ))
        for b, centre in enumerate(OCTAVE_BANDS_HZ):
            low, high = centre / np.sqrt(2), centre * np.sqrt(2)
            # Fine enough for the largest difference of arrival between two elements.
            span = np.ptp(lead - self.delays) + 1e-4
            f = np.linspace(low, high, max(400, int(20 * (high - low) * span)))
            phase = np.exp(2j * np.pi * f[:, None] * (lead - self.delays)[None])
            out[b] = np.mean(np.abs(phase @ amplitude[:, b]) ** 2)
        return out


def differences(room: Room) -> dict[str, dict[str, np.ndarray]]:
    """For each array source and listener, the far-field pattern's level above the exact
    sum of the elements (dB per octave band)."""
    c = room.speed_of_sound()
    out = {}
    for source in room.scene.sources:
        spec = source.directivity
        if spec is None or spec.file is None or spec.file.suffix.lower() not in (".yaml", ".yml"):
            continue
        array = _Array(spec.file)
        turn = np.array(source.orientation.rotation())
        out[source.name] = {}
        for listener in room.scene.listeners:
            where = turn.T @ (np.asarray(listener.position) - np.asarray(source.position))
            far, exact = (array.energy(where, flag, c) for flag in (True, False))
            out[source.name][listener.name] = 10 * np.log10(far / exact)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", nargs="?", default=Path(__file__).with_name("scene.yaml"))
    args = parser.parse_args(argv)
    bands = " ".join(f"{f if f < 1000 else f'{f // 1000}k':>5}" for f in OCTAVE_BANDS_HZ)
    for source, rows in differences(Room.from_yaml(args.scene)).items():
        print(f"\n{source + ', far field above exact (dB)':43}")
        print(f"  {'':20} {bands}")
        for listener, db in rows.items():
            print(f"  {listener:20} " + " ".join(f"{d:5.1f}" for d in db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
