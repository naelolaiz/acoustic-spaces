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


# The values below come from the report's code (its appendix D listing), transcribed line
# by line apart from this package and run in double precision, for cases its sample
# outputs D2 and D3 leave out; the transcription gives their TBL-TE and tip columns to
# within 0.01 dB.


def _bands(levels, *hz: float) -> np.ndarray:
    return np.asarray(levels)[..., np.searchsorted(F, hz)]


def test_a_lightly_tripped_boundary_layer_takes_the_untripped_suction_fits() -> None:
    """THICK: a light trip makes delta_0 and delta*_0 0.6 of a heavy trip's (lines 40
    and 56), and its suction side takes the untripped fits (the ELSE of line 70): at
    0.3048 m and 71.3 m/s (Rc 1.4958e6), delta*_0 = 0.6 x 2.6634e-3 m, times
    0.0162 x 10^(0.3066 alpha) past 7.5 degrees. The heavy trip's fits would give
    1.35 times as thick at 8 degrees and 0.32 times at 12."""
    nu = AIR["viscosity"]
    for alpha, expected in ((8.0, 7.3434e-3), (12.0, 1.23679e-1)):
        _, dstar_s, _ = bpm.thicknesses(0.3048, 71.3, alpha, bpm.LIGHTLY_TRIPPED, nu)
        assert dstar_s == pytest.approx(expected, rel=1e-4)
    alpha = np.array([0.0, 4.0, 8.0, 12.0, 16.0])
    light = bpm.thicknesses(0.3048, 71.3, alpha, bpm.LIGHTLY_TRIPPED, nu)
    heavy = bpm.thicknesses(0.3048, 71.3, alpha, bpm.TRIPPED, nu)
    # The pressure side's thickness and displacement thickness are 0.6 of a heavy
    # trip's at every angle; the suction side's too up to 5 degrees, where both trips
    # share one fit.
    np.testing.assert_allclose(light[0] / heavy[0], 0.6, rtol=1e-12)
    np.testing.assert_allclose(light[2] / heavy[2], 0.6, rtol=1e-12)
    np.testing.assert_allclose(light[1][:2] / heavy[1][:2], 0.6, rtol=1e-12)


def test_a_stalled_segment_as_the_reports_code() -> None:
    """TBLTE lines 0253-0262: past gamma_0 = 23.43 M + 4.651 = 7.376 degrees (M 0.1163),
    stalled though under 12.5 degrees, each side is 10 lg(delta*_s M^5 L D_l / r^2)
    and the angle of attack's part is the A curve for 3 Rc (A'(3 Rc): a_0 0.943, its
    ratio 0.924, where Rc's own are 0.636 and 0.340) at |lg(St_s / St_2)|, plus K2,
    with the low-frequency directivity D_l = 0.5581 (D_h is 0.4308 here). A 0.0508 m
    chord, 0.4572 m of span, 1.22 m away at 60 degrees from the chord and 75 from the
    span, 10 degrees into 39.6 m/s, tripped: Rc 1.3846e5, K1 134.141, K2 133.881,
    delta*_s 9.898 mm."""
    m = 39.6 / AIR["sound_speed"]
    assert bpm.stalled(10.0, m)
    args = (F, 0.0508, 0.4572, 1.22, 60.0, 75.0, 10.0, 39.6)
    p, s, a = bpm.tbl_te(*args, trip=bpm.TRIPPED, **AIR)
    np.testing.assert_allclose(p, -74.422, atol=0.01)
    np.testing.assert_allclose(s, -74.422, atol=0.01)
    expected = [42.194, 59.398, 46.465, 25.403, -18.678]
    np.testing.assert_allclose(_bands(a, 100, 800, 4000, 12500, 40000), expected, atol=0.01)


def test_a_low_reynolds_number_segment_as_the_reports_code() -> None:
    """TBLTE and THICK below their lowest breaks: a 0.0254 m chord 4 degrees into
    39.6 m/s (Rc 6.923e4) takes K1 = -4.31 lg Rc + 156.3 = 135.438 (Rc < 2.47e5),
    a_0 = 0.57 and b_0 = 0.30 (Rc < 9.52e4), Delta K1 = -alpha (5.29 - 1.43 lg
    R_delta*p) (R_delta*p <= 5000), and tripped, delta*_0 = 0.0601 Rc^-0.114 c
    (Rc <= 3e5). 0.4572 m of span, 1.22 m away at 90 degrees."""
    nu = AIR["viscosity"]
    _, dstar_s, dstar_p = bpm.thicknesses(0.0254, 39.6, 4.0, bpm.TRIPPED, nu)
    assert (dstar_s, dstar_p) == pytest.approx((8.0078e-4, 3.0005e-4), rel=1e-4)
    expected = {
        # Pressure side, suction side, angle of attack's part at 2, 5 and 12.5 kHz.
        bpm.TRIPPED: [[9.943, 35.534, 40.122], [44.593, 48.760, 33.212], [32.145, 49.761, -20.177]],
        bpm.UNTRIPPED: [
            [-21.210, 25.170, 37.806],
            [35.433, 47.656, 40.430],
            [-10.967, 48.732, 25.940],
        ],
    }
    args = (F, 0.0254, 0.4572, 1.22, 90.0, 90.0, 4.0, 39.6)
    for trip, levels in expected.items():
        ours = bpm.tbl_te(*args, trip=trip, **AIR)
        np.testing.assert_allclose(_bands(ours, 2000, 5000, 12500), levels, atol=0.01)


def test_a_blunt_trailing_edge_as_the_reports_code() -> None:
    """BLUNT and G5COMP: a trailing edge on a 0.3048 m chord at 0 degrees in 71.3 m/s,
    tripped (delta*_avg 2.6634 mm), 0.4572 m of span, 1.22 m away at 90 degrees. 2.5 mm
    thick with Psi = 14 degrees: h / delta*_avg 0.9387, St'''_peak 0.1206 (3.44 kHz),
    G4 141.423 dB. 1.1 mm thick with Psi = 0: h / delta*_avg 0.4130, remapped for
    Psi = 0 to 0.5941 (line 0115), St'''_peak 0.1421 (9.21 kHz), G4 150.779 dB."""
    # Thickness (m) and Psi (degrees): the loudest band (Hz), and its level and those of
    # the two bands either side.
    cases = {
        (0.0025, 14.0): (3150.0, [42.073, 55.807, 70.031, 67.116, 52.042]),
        (0.0011, 0.0): (10000.0, [69.794, 76.229, 77.688, 62.484, 45.808]),
    }
    args = (F, 0.3048, 0.4572, 1.22, 90.0, 90.0, 0.0, 71.3)
    for (h, psi), (peak_hz, levels) in cases.items():
        ours = bpm.bluntness(*args, h, psi, trip=bpm.TRIPPED, **AIR)
        k = int(np.argmax(ours))
        assert F[k] == peak_hz
        np.testing.assert_allclose(ours[k - 2 : k + 3], levels, atol=0.01)


def test_a_square_tip_as_the_reports_code() -> None:
    """TIPNOIS lines 0050-0054: a square tip's vortex is (0.0378 + 0.0095 alpha) c
    across past 2 degrees (1.300 cm at 5 degrees on a 0.1524 m chord) and
    (0.023 + 0.0169 alpha) c up to it (0.737 cm at 1.5 degrees), with M_max =
    (1 + 0.036 alpha) M; in 71.3 m/s, 1.22 m away at 90 degrees."""
    expected = {
        5.0: (3150.0, [46.794, 54.754, 53.682, 40.108]),
        1.5: (5000.0, [33.045, 47.009, 48.354, 40.866]),
    }
    for alpha, (peak_hz, levels) in expected.items():
        ours = bpm.tip(F, 0.1524, 1.22, 90.0, 90.0, alpha, 71.3, rounded=False, sound_speed=340.46)
        assert F[np.argmax(ours)] == peak_hz
        np.testing.assert_allclose(_bands(ours, 1000, 3150, 5000, 16000), levels, atol=0.01)
