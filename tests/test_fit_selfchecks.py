"""Fitted walls: the fit's self-checks, read from cut_stats.

On each golden case (tests/reference/fit_golden.npz) and on the three cases of test_fit_cases.py,
every self-check of the fit (K1 to K9, the wall balance and the partition) reports no failed
comparison, and its largest residual stays below one thousandth of its threshold: a check that
throws in every build must leave legitimate fits that far from throwing.
"""

import numpy as np
import pytest
from test_fit_cases import case_mesh, run_fit
from test_fit_golden import DATA, NAMES

from acoustic_spaces import _native

CHECKS = [
    "check_wall",
    "check_volume",
    "check_copy",
    "check_rows",
    "check_open_symmetry",
    "check_open",
    "check_piece_sides",
    "check_owners",
    "check_conduits",
]


def golden_stats(name):
    g = DATA[name + "/grid"]
    nx, ny, nz, materials = (int(x) for x in DATA[name + "/shape"])
    cut = _native.CutFit(
        DATA[name + "/vertices"],
        DATA[name + "/faces"],
        DATA[name + "/materials"],
        materials,
        g[0:3],
        float(g[3]),
        nx,
        ny,
        nz,
        g[4:7],
        0,
        2,
        False,
    )
    return cut.cut_stats


def case_stats(name):
    v, f, mats, materials, _ = case_mesh(name)
    return run_fit(v, f, mats, materials).cut_stats


def assert_checks_pass(stats):
    for key in CHECKS:
        assert stats[key + "_failed"] == 0, key
        assert stats[key] < 1e-3, key
    # The balance and the partition are relative residuals with the threshold 1e-9.
    assert stats["balance"] < 1e-12
    assert stats["partition"] < 1e-12
    assert stats["owner_near_band"] == 0
    assert stats["sliver_near_gate"] == 0
    assert stats["conduit_near_band"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_golden_cases_pass_every_check(name):
    # A refused case (the golden set has two) stops before the later checks, whose fields stay zero.
    assert_checks_pass(golden_stats(name))


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_exact_cases_pass_every_check(name):
    stats = case_stats(name)
    assert stats["refused"] == 0
    assert_checks_pass(stats)
    assert np.isfinite(stats["check_wall"])
