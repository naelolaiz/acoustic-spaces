"""The sound a source sends each way (:mod:`acoustic_spaces.emission`): what it sends
in a direction, on average and in bands."""

import numpy as np
import pytest

from acoustic_spaces.directivity import _from_angles, _grid, _sphere
from acoustic_spaces.emission import Mixture

RATE = 48_000


def _tone(f: float = 500.0, seconds: float = 1.0) -> Mixture:
    """A tone louder ahead than behind (as a sine), and a cosine of it sent alike every
    way."""
    elevation, azimuth = _grid(10.0)
    ways = _from_angles(azimuth[None, :], elevation[:, None])
    t = np.arange(int(seconds * RATE)) / RATE
    signals = np.array([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    weights = np.stack([1.0 + 0.8 * ways[..., 0], np.full(ways.shape[:2], 0.5)], axis=-1)
    return Mixture(signals, weights, RATE)


def test_what_is_sent_each_way() -> None:
    sent = _tone()
    ahead = sent.sent(-10, 100, np.tile([1.0, 0.0, 0.0], (110, 1)))
    assert np.all(ahead[:10] == 0.0)
    np.testing.assert_allclose(
        ahead[10:], 1.8 * sent.signals[0, :100] + 0.5 * sent.signals[1, :100]
    )
    behind = sent.sent(0, 100, np.tile([-1.0, 0.0, 0.0], (100, 1)))
    np.testing.assert_allclose(behind, 0.2 * sent.signals[0, :100] + 0.5 * sent.signals[1, :100])
    assert np.all(
        sent.sent(sent.length - 5, sent.length + 5, np.tile([0, 0, 1.0], (10, 1)))[5:] == 0
    )


def test_on_average_and_in_bands() -> None:
    sent = _tone()
    ways = _sphere(100_000)
    power = np.mean(((1.0 + 0.8 * ways[:, 0]) ** 2 + 0.25) / 2.0)
    assert np.mean(sent.averaged() ** 2) == pytest.approx(power, rel=1e-3)
    energy = sent.directivity().energy([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    # 500 Hz is the fourth octave band.
    assert energy[0, 3] / energy[1, 3] == pytest.approx((1.8**2 + 0.25) / (0.2**2 + 0.25), rel=1e-3)
    louder = sent.scaled(2.0)
    np.testing.assert_allclose(louder.averaged(), 2.0 * sent.averaged())


def test_what_an_emission_needs() -> None:
    elevation, azimuth = _grid(30.0)
    with pytest.raises(ValueError, match="one per signal"):
        Mixture(np.ones((2, 10)), np.ones((len(elevation), len(azimuth), 1)), RATE)
    with pytest.raises(ValueError, match="at least two elevations"):
        Mixture(np.ones((1, 10)), np.ones((1, len(azimuth), 1)), RATE)
    with pytest.raises(ValueError, match="finite"):
        Mixture(np.full((1, 10), np.nan), np.ones((len(elevation), len(azimuth), 1)), RATE)
