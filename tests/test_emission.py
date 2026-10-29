"""The sound a source sends each way (:mod:`acoustic_spaces.emission`): what it sends
in a direction, on average and in bands."""

import numpy as np
import pytest

from acoustic_spaces.directivity import _from_angles, _grid, _sphere
from acoustic_spaces.emission import Mixture, Periodic, Sum

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


def _periodic(**kw) -> Periodic:
    """One period (10 ms, 64 samples) of a tone sent louder ahead, 1 + 0.5 cos θ."""
    elevation, azimuth = _grid(10.0)
    ways = _from_angles(azimuth[None, :], elevation[:, None])
    t = np.arange(64) / 64 * 0.01
    table = (1.0 + 0.5 * ways[..., 0])[..., None] * np.cos(2 * np.pi * 100.0 * t)
    return Periodic(table, 0.01, RATE, RATE, **kw)


def test_a_sound_that_repeats() -> None:
    sent = _periodic()
    n = np.arange(-10, 990)
    tone = np.cos(2 * np.pi * 100.0 * n / RATE)
    ahead = sent.sent(-10, 990, np.tile([1.0, 0.0, 0.0], (1000, 1)))
    assert np.all(ahead[:10] == 0.0)
    # A cubic through four of 64 samples a period reads a tone to 1e-4.
    np.testing.assert_allclose(ahead[10:], 1.5 * tone[10:], atol=2e-4)
    behind = sent.sent(0, 990, np.tile([-1.0, 0.0, 0.0], (990, 1)))
    np.testing.assert_allclose(behind, 0.5 * tone[10:], atol=1e-4)
    assert np.all(sent.sent(RATE - 5, RATE + 5, np.tile([0, 0, 1.0], (10, 1)))[5:] == 0)


def test_a_sound_that_turns_with_its_rotor() -> None:
    # Three blades turning at 20 turns a second about y: heard from an azimuth phi
    # round the axis, the third harmonic of the turn a share phi / 2 pi of it later.
    turn = 1.0 / 20.0
    axes = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elevation, azimuth = _grid(10.0)
    own = _from_angles(azimuth[None, :], elevation[:, None])
    t = np.arange(64) / 64 * turn / 3
    # Held a share of a turn later, so every azimuth holds the same.
    table = np.hypot(own[..., 0], own[..., 1])[..., None] * np.cos(2 * np.pi * 60.0 * t)
    sent = Periodic(table, turn / 3, RATE, RATE, axes=axes, turn_s=turn)
    rng = np.random.default_rng(0)
    ways = rng.normal(size=(500, 3))
    ways /= np.linalg.norm(ways, axis=1, keepdims=True)
    mine = ways @ axes.T
    phi = np.arctan2(mine[:, 1], mine[:, 0])
    times = np.arange(1000, 1500) / RATE
    exact = np.hypot(mine[:, 0], mine[:, 1]) * np.cos(
        2 * np.pi * 60.0 * (times - phi / (2 * np.pi) * turn)
    )
    # Bilinear between directions 10 degrees apart, of a pattern that is sin θ.
    np.testing.assert_allclose(sent.sent(1000, 1500, ways), exact, atol=5e-3)
    # The other way round, the delay changes sign.
    other = Periodic(table, turn / 3, RATE, RATE, axes=axes, turn_s=-turn)
    exact = np.hypot(mine[:, 0], mine[:, 1]) * np.cos(
        2 * np.pi * 60.0 * (times + phi / (2 * np.pi) * turn)
    )
    np.testing.assert_allclose(other.sent(1000, 1500, ways), exact, atol=5e-3)


def test_a_sound_sent_from_off_the_source_s_place() -> None:
    centre = np.array([-6.0, 0.5, 1.0])
    sent = _periodic(centre=centre, speed=60.0, sound_speed=340.0)
    plain = _periodic()
    ways = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    for way in ways:
        lead = way @ centre / (340.0 - 60.0 * way[0])
        shifted = plain.sent(2000, 2400, np.tile(way, (400, 1)))
        n = np.arange(2000, 2400) + lead * RATE
        exact = (1.0 + 0.5 * way[0]) * np.cos(2 * np.pi * 100.0 * n / RATE)
        np.testing.assert_allclose(sent.sent(2000, 2400, np.tile(way, (400, 1))), exact, atol=2e-4)
        assert not np.allclose(shifted, exact, atol=1e-2)


def test_a_periodic_sound_on_average_and_in_bands() -> None:
    sent = _periodic()
    power = np.mean((1.0 + 0.5 * _sphere(100_000)[:, 0]) ** 2) / 2.0
    assert np.mean(sent.averaged() ** 2) == pytest.approx(power, rel=1e-3)
    energy = sent.directivity().energy([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    # 100 Hz is in the second octave band.
    assert energy[0, 1] / energy[1, 1] == pytest.approx(9.0, rel=1e-3)
    np.testing.assert_allclose(sent.scaled(3.0).averaged(), 3.0 * sent.averaged(), atol=1e-12)


def test_a_sum_of_emissions() -> None:
    tone, period = _tone(), _periodic()
    both = Sum((tone, period))
    ways = np.tile([0.0, 0.6, 0.8], (300, 1))
    np.testing.assert_allclose(
        both.sent(100, 400, ways), tone.sent(100, 400, ways) + period.sent(100, 400, ways)
    )
    assert both.length == RATE and both.sample_rate == RATE
    np.testing.assert_allclose(both.averaged(), tone.averaged() + period.averaged())
    np.testing.assert_allclose(
        both.scaled(2.0).sent(100, 400, ways), 2.0 * both.sent(100, 400, ways)
    )
    # Their energies add in every band and direction.
    ahead = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        both._powers("octave", ahead),
        tone._powers("octave", ahead) + period._powers("octave", ahead),
    )
    with pytest.raises(ValueError, match="share a sample rate"):
        Sum((tone, Periodic(period.table, 0.01, 44_100, 10)))
    with pytest.raises(ValueError, match="at least one"):
        Sum(())


def test_what_a_periodic_sound_needs() -> None:
    table = np.ones((19, 36, 8))
    with pytest.raises(ValueError, match="whole number of periods"):
        Periodic(table, 0.01, RATE, 10, turn_s=0.025)
    with pytest.raises(ValueError, match="rotation"):
        Periodic(table, 0.01, RATE, 10, axes=2.0 * np.eye(3))
    with pytest.raises(ValueError, match="slower than sound"):
        Periodic(table, 0.01, RATE, 10, speed=400.0)
    with pytest.raises(ValueError, match="four samples"):
        Periodic(np.ones((19, 36, 3)), 0.01, RATE, 10)
