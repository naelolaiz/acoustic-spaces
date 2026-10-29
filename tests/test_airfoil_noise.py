"""Airfoil self-noise (:mod:`acoustic_spaces.airfoil_noise`) against the two sample
outputs of Brooks, Pope and Marcolini's code (NASA RP-1218, tables D2 and D3)."""

import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import airfoil_noise as bpm

TABLES = json.loads((Path(__file__).parent / "reference" / "bpm_tables.json").read_text())
F = np.array(bpm.FREQUENCIES_HZ)
#: The air of the report's code: its kinematic viscosity (m²/s) and speed of sound (m/s).
AIR = {"viscosity": 1.4529e-5, "sound_speed": 340.46}


def _total(*levels) -> np.ndarray:
    return 10.0 * np.log10(sum(10.0 ** (level / 10.0) for level in levels))


def test_one_untripped_segment_as_table_d2() -> None:
    # A 30.48 cm chord, 45.72 cm of span, 1.22 m away at 90 degrees, 1.516 degrees
    # into a 71.3 m/s flow, its boundary layers untripped.
    table = np.array(TABLES["D2"])
    np.testing.assert_array_equal(table[:, 0], F)
    args = (F, 0.3048, 0.4572, 1.22, 90.0, 90.0, 1.516, 71.3)
    p, s, a = bpm.tbl_te(*args, trip=bpm.UNTRIPPED, **AIR)
    laminar = bpm.lbl_vs(*args, trip=bpm.UNTRIPPED, **AIR)
    ours = np.column_stack([p, s, a, laminar, _total(p, s, a, laminar)])
    # The table's levels are rounded to 0.001 dB, from the code's single precision.
    np.testing.assert_allclose(ours, table[:, [1, 2, 3, 4, 7]], atol=0.01)


def test_ten_tripped_segments_and_a_rounded_tip_as_table_d3() -> None:
    # Ten segments of 15.24 cm chord, 3.05 cm span each, at 5.4 degrees, tripped; and
    # the rounded tip's vortex.
    table = np.array(TABLES["D3"])
    n = 10
    args = (F, [0.1524] * n, [0.0305] * n, [1.22] * n, [90.0] * n, [90.0] * n, [5.4] * n)
    p, s, a = bpm.tbl_te(*args, [71.3] * n, trip=bpm.TRIPPED, **AIR)
    summed = [10.0 * np.log10(np.sum(10.0 ** (x / 10.0), axis=0)) for x in (p, s, a)]
    np.testing.assert_allclose(np.column_stack(summed), table[:, 1:4], atol=0.01)

    def tip(alpha: float) -> np.ndarray:
        return bpm.tip(F, 0.1524, 1.22, 90.0, 90.0, alpha, 71.3, rounded=True, sound_speed=340.46)

    # The table's tip column is the code's equations at 7.668 degrees, not at the 7.7 of
    # its input (0.19 dB apart at 100 Hz); the code as printed gives 7.7's levels, so the
    # table may come from another version of it.
    np.testing.assert_allclose(tip(7.668), table[:, 6], atol=0.01)
    assert np.max(np.abs(tip(7.7) - table[:, 6])) == pytest.approx(0.19, abs=0.01)
    np.testing.assert_allclose(_total(*summed, tip(7.668)), table[:, 7], atol=0.01)


def test_directivity_and_stall() -> None:
    # Both directivities are 1 at 90 degrees from the chord and from the span.
    assert bpm.directivity_high(0.3, 90.0, 90.0) == pytest.approx(1.0)
    assert bpm.directivity_low(0.3, 90.0, 90.0) == pytest.approx(1.0)
    # Moving towards the observer (theta 180 degrees, ahead of the leading edge) is
    # louder than away, at the same angle from the chord.
    assert bpm.directivity_high(0.3, 135.0, 90.0) > bpm.directivity_high(0.3, 45.0, 90.0)
    # The flow separates past 23.43 M + 4.651 degrees, or 12.5.
    assert not bpm.stalled(6.9, 0.1) and bpm.stalled(7.0, 0.1)
    assert not bpm.stalled(12.4, 0.5) and bpm.stalled(12.6, 0.5)


def test_a_sharp_trailing_edge_is_quiet_and_levels_floor() -> None:
    args = (F, 0.3, 0.1, 1.0, 90.0, 90.0, 4.0, 60.0)
    p, s, a = bpm.tbl_te(*args, trip=bpm.TRIPPED, **AIR)
    assert np.all(np.isfinite(p)) and np.all(a >= bpm.FLOOR_DB)
    blunt = bpm.bluntness(*args, 0.002, 14.0, trip=bpm.TRIPPED, **AIR)
    thinner = bpm.bluntness(*args, 0.001, 14.0, trip=bpm.TRIPPED, **AIR)
    # A thinner edge sheds smaller vortices: quieter and higher.
    assert np.max(thinner) < np.max(blunt)
    assert F[np.argmax(thinner)] > F[np.argmax(blunt)]
