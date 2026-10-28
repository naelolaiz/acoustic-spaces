"""The walks of validation/walks.py, 1 cm at a time past the edges in the examples: the
early level steps by no more than it does now, with a margin (the validation report's
"Round a corner indoors", "Along a street" and "Known issues of the edge fade")."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import walks  # noqa: E402


def _largest_steps(walk: dict) -> np.ndarray:
    """The largest step between neighbours in each band, over every line of ``walk``."""
    steps = [np.abs(np.diff(levels, axis=0)).max(axis=0) for levels in walk["lines"].values()]
    return np.max(steps, axis=0)


def test_the_archway_walk() -> None:
    """Behind the archway's wall, the whole walk (x = 7.6 m, y = 5.30 to 7.80 m): at most
    0.57 and 0.41 dB at 63 and 125 Hz, where a thick end cuts reflections off (a known
    issue), 0.08 dB from 250 Hz to 2 kHz, and 0.16 dB above, where the level swings
    smoothly over a few centimetres."""
    steps = _largest_steps(walks.archway(True))
    assert (steps < [0.65, 0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]).all(), steps


@pytest.mark.parametrize("span", [(7.9, 9.9), (13.2, 13.5)])
def test_the_gallery_walk(span: tuple[float, float]) -> None:
    """Round the gallery's corner, where paths over it come and go, and where the step at
    125 Hz is largest on the whole walk (x = 5.5 m): at most 0.06 dB but 0.10 dB at
    125 Hz and 0.07 and 0.13 dB at 4 and 8 kHz."""
    steps = _largest_steps(walks.gallery(True, lines=(5.5,), span=span))
    assert (steps < [0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]).all(), steps


@pytest.mark.parametrize("x", [-41.39, -14.04, -0.69, 60.80])
def test_the_street_walk(x: float) -> None:
    """The car 1 cm at a time round the largest steps in the garden's early level, where
    a path over the barrier's top or a reflection off it reaches an end of the top (a
    known issue): the step there stands out from the steps either side by at most
    1.42 dB in a band below 8 kHz and 0.53 dB(A)."""
    street = walks._Street()
    heard = [street.levels(at, ["garden"])[0] for at in np.round(x + np.arange(-3, 3) * 0.01, 2)]
    out = walks._stands_out(np.array(heard))
    assert out[:, :-2].max() < 1.5, out.max(axis=0)
    assert out[:, -1].max() < 0.6, out.max(axis=0)
