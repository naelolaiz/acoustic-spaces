"""Fitted walls: the arithmetic the fit rests on, tested in this platform's compiled code.

`_native._fit_selftest` runs the checks of cpp/include/acoustic_spaces/cut_selftest.hpp: the wide
integers' primitives against their portable code, the 128-bit classes against a reference in
32-bit limbs, exact division, the correctly rounded conversions to double, the compiler's
conversion of 64-bit integers of 2^53 and more to double against the correctly rounded one, the
stencil's quotient and the double-double operations against exact values. It also reports the
arithmetic guard: the package's own build must be one in which the fit runs (IEEE doubles, no
contraction of a product and a sum, no reassociation, rounding to nearest, 64-bit integers
converted to double rounded to nearest).
"""

import pytest

from acoustic_spaces import _native

CHECKS = [
    "mul64",
    "zero_counts",
    "wide_integers",
    "exact_division",
    "conversion",
    "int64_conversion",
    "quotient",
    "dd_sum",
    "dd_product",
    "dd_quotient",
    "dd_sqrt",
]


@pytest.mark.parametrize("seed", [1, 2])
def test_every_check_has_cases_and_no_mismatch(seed):
    out = _native._fit_selftest(4000, seed)
    for name in CHECKS:
        assert out[name + "_cases"] > 0, name
        assert out[name + "_mismatches"] == 0, name


def test_the_package_build_passes_the_arithmetic_guard():
    out = _native._fit_selftest(1, 1)
    assert out["arithmetic_constant"] is True
    assert out["probe_failed"] == ""
    assert out["rounding_to_nearest"] is True
