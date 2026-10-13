import math

import numpy as np
import pytest

from acoustic_spaces.parameters import decay_curve, from_energy, from_impulse_response
from acoustic_spaces.synthesis import REFERENCE_ENERGY_10M

DT = 1e-3


def _exponential(t60: float, length: float, onset: float = 0.0) -> np.ndarray:
    """Energy per slot of a decay e^(-k t) starting at ``onset``, integrated per slot."""
    k = 6.0 * math.log(10.0) / t60
    edges = np.maximum(np.arange(int(round(length / DT)) + 1) * DT - onset, 0.0)
    return (np.exp(-k * edges[:-1]) - np.exp(-k * edges[1:])) / k


@pytest.mark.parametrize("t60", [0.4, 1.0, 2.5])
def test_decay_times_of_an_exponential(t60: float) -> None:
    # Truncated at 2 s: the long decay needs the tail estimate to come out right.
    energy = np.tile(_exponential(t60, 2.0), (8, 1))
    p = from_energy(energy, DT, onset_s=0.0)
    for value in (p.edt, p.t20, p.t30):
        np.testing.assert_allclose(value, t60, rtol=0.02)


def test_energy_ratios_of_an_exponential() -> None:
    t60, onset = 1.0, 0.0105
    k = 6.0 * math.log(10.0) / t60
    energy = np.tile(_exponential(t60, 3.0, onset), (8, 1))
    p = from_energy(energy, DT, onset_s=onset)
    early = 1.0 - math.exp(-k * 0.08)
    np.testing.assert_allclose(p.c80, 10 * math.log10(early / (1 - early)), atol=0.05)
    np.testing.assert_allclose(p.d50, 1.0 - math.exp(-k * 0.05), atol=0.005)
    np.testing.assert_allclose(p.ts, 1.0 / k, rtol=0.01)


def test_strength_against_free_field() -> None:
    reference = 1.0 / (4 * math.pi * 100)
    energy = np.zeros((8, 100))
    energy[:, 10] = reference * 10  # ten times the free field energy at 10 m
    p = from_energy(energy, DT, onset_s=0.0101, reference_energy=reference)
    np.testing.assert_allclose(p.g, 10.0, atol=1e-9)
    assert np.all(np.isnan(p.t30))  # a single impulse has no decay to fit


def test_decay_curve_starts_at_zero_and_falls() -> None:
    curve = decay_curve(_exponential(1.0, 0.5), DT)
    assert curve[0] == pytest.approx(0.0)
    assert np.all(np.diff(curve) < 0)
    assert curve[300] == pytest.approx(-18.0, abs=0.2)


def test_from_impulse_response_of_decaying_noise() -> None:
    fs, t60 = 48_000, 0.8
    rng = np.random.default_rng(3)
    t = np.arange(int(1.5 * fs)) / fs
    ir = rng.standard_normal(len(t)) * np.exp(-3.0 * math.log(10.0) * t / t60)
    ir = np.concatenate([np.zeros(480), ir])
    p = from_impulse_response(ir, fs, REFERENCE_ENERGY_10M)
    np.testing.assert_allclose(p.t30[2:], t60, rtol=0.1)
    assert np.all(np.isfinite(p.g))


def test_tail_estimate_ignores_a_late_spike() -> None:
    # With few rays, one strong ray can land in the last few milliseconds of a response
    # that stops long before it has decayed. That must not read as a decay that stopped.
    t60 = 2.5
    k = 6.0 * math.log(10.0) / t60
    energy = _exponential(t60, 0.8)
    energy[790] += 0.03 * energy.sum()
    p = from_energy(energy[None, :], DT, onset_s=0.0, reference_energy=1.0 / k)
    assert abs(p.g[0]) < 0.3  # total energy, against the untruncated decay's 1 / k
    assert p.ts[0] == pytest.approx(1.0 / k, rel=0.2)
