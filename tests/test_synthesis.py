import numpy as np
import pytest

from acoustic_spaces.synthesis import (
    band_energy_share,
    band_gains,
    combine_bands,
    dirac_sequence,
    place_impulses,
    shape_noise,
    smooth_energy,
    split_bands,
)

FS = 48_000


def test_band_gains_sum_to_one() -> None:
    gains = band_gains(4096, FS)
    assert gains.shape == (8, 2049)
    np.testing.assert_allclose(gains.sum(axis=0), 1.0, atol=1e-12)
    assert gains.min() >= 0.0


def test_split_and_combine_reconstruct() -> None:
    signal = np.random.default_rng(1).standard_normal(5000)
    bands = split_bands(signal, FS)
    np.testing.assert_allclose(bands.sum(axis=0), signal, atol=1e-10)
    same = np.broadcast_to(signal, (8, len(signal)))
    np.testing.assert_allclose(combine_bands(same, FS), signal, atol=1e-10)


def test_band_shares_cover_the_spectrum() -> None:
    share = band_energy_share(FS)
    assert share.shape == (8,)
    assert np.all(share > 0)
    # The 8 kHz band runs to Nyquist, so it holds most of a white spectrum.
    assert share.argmax() == 7


def test_place_impulses_integer_and_fractional() -> None:
    out = place_impulses(np.array([10 / FS]), np.full((1, 8), 0.5), 64, FS)
    np.testing.assert_allclose(out[:, 10], 0.5, atol=1e-12)
    assert np.abs(np.delete(out, 10, axis=1)).max() < 1e-12
    frac = place_impulses(np.array([20.5 / FS]), np.ones((1, 8)), 64, FS)
    # The windowed sinc rolls off in the last octave below Nyquist only.
    assert 0.93 < (frac[0] ** 2).sum() <= 1.0
    assert frac[0, 20] == pytest.approx(frac[0, 21])


def test_dirac_sequence_has_a_pulse_in_every_slot() -> None:
    rng = np.random.default_rng(0)
    seq = dirac_sequence(FS, FS, 200.0, 343.0, rng, 1e-3)
    per_slot = np.abs(seq).reshape(-1, 48).sum(axis=1)
    assert per_slot.min() >= 1
    assert set(np.unique(seq)) <= {-1.0, 0.0, 1.0}
    assert np.abs(seq[-4800:]).sum() > np.abs(seq[:4800]).sum()  # density grows with t^2


def test_shape_noise_sets_slot_energy() -> None:
    rng = np.random.default_rng(0)
    noise = dirac_sequence(4800, FS, 100.0, 343.0, rng, 1e-3)
    energy = np.outer(np.linspace(1, 2, 8), np.exp(-np.arange(100) / 20))
    shaped = shape_noise(energy, 1e-3, noise, FS)
    got = (shaped**2).reshape(8, 100, 48).sum(axis=2)
    np.testing.assert_allclose(got, energy, rtol=1e-12)


@pytest.mark.parametrize("rate", [22_050, 44_100, 88_200])
def test_no_slot_is_silent_where_slots_are_not_whole_samples(rate: int) -> None:
    # 1 ms is 44.1 samples at 44.1 kHz. In a hall the impulses are sparse for 200 ms,
    # and every slot of shape_noise still holds one.
    rng = np.random.default_rng(0)
    noise = dirac_sequence(rate, rate, 20_000.0, 343.0, rng, 1e-3)
    slot = np.floor(np.arange(rate) / (rate * 1e-3) + 1e-9).astype(np.int64)
    assert np.bincount(slot, weights=np.abs(noise)).min() >= 1
    energy = np.outer(np.linspace(1, 2, 8), np.exp(-np.arange(1000) / 200))
    shaped = shape_noise(energy, 1e-3, noise, rate)
    got = np.array([np.bincount(slot, weights=band**2) for band in shaped])
    np.testing.assert_allclose(got, energy, rtol=1e-12)


def test_smooth_energy_keeps_the_total() -> None:
    energy = np.zeros((2, 500))
    energy[:, 300] = 1.0
    smoothed = smooth_energy(energy, 1e-3, 0.1, window_s=0.01)
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0)
    assert np.count_nonzero(smoothed[0]) == 11
    early = np.zeros(500)
    early[20] = 1.0
    np.testing.assert_array_equal(smooth_energy(early[None], 1e-3, 0.1)[0], early)
