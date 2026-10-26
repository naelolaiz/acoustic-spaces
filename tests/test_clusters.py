"""The waves that are one wave in the bands where the early paths add in energy: each
wave's energy is that of its pieces' pressures added up, averaged over the slices of
the band, and goes to the rows the pieces belong to."""

from __future__ import annotations

import numpy as np

from acoustic_spaces.engine import clusters as clusters_module
from acoustic_spaces.engine.clusters import Clusters, _pair_shares, _rule_size, _slice_shares
from acoustic_spaces.engine.diffraction import band_frequencies

C = 343.0
SLICE_HZ = band_frequencies()[1]
BANDS, PER_BAND = len(SLICE_HZ), 8


def _pieces(rng, count: int, spread_m: float) -> tuple[np.ndarray, np.ndarray]:
    pressure = rng.normal(size=(count, BANDS, PER_BAND))
    pressure = pressure + 1j * rng.normal(size=pressure.shape)
    return pressure, 10.0 + rng.uniform(0.0, spread_m, count)


def _pairwise(pressure: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(n, bands) each piece's own energy and half its cross term with each other piece,
    pair by pair: the slices fade two pieces' interference by sinc."""
    fade = np.sinc((distance[:, None] - distance[None, :])[..., None] / C * SLICE_HZ)
    products = np.einsum("ibm,jbm->ijb", pressure, np.conj(pressure)).real / PER_BAND
    return (fade * products).sum(axis=1)


def test_a_wave_s_energy_is_its_pieces_added_up_pair_by_pair() -> None:
    """Within 1e-10 of the pairs' sum, for pieces that arrive together or up to 2 km
    apart: pair by pair, as a wave of few pieces takes them, and by the rule, whose work
    grows with the pieces, not their pairs."""
    rng = np.random.default_rng(1)
    clusters = Clusters(SLICE_HZ, C, 10.0)
    for spread in (0.0, 0.01, 1.0, 100.0, 2000.0):
        pressure, distance = _pieces(rng, 40, spread)
        exact = _pairwise(pressure, distance)
        shares = clusters._shares(pressure, distance, np.zeros(40, dtype=np.int64))
        assert np.abs(shares - exact).max() <= 1e-10 * np.abs(exact).max()
        assert (shares.sum(axis=0) >= 0.0).all()
        relative = (distance - 0.5 * (distance.min() + distance.max())) / C
        for b, width in enumerate(SLICE_HZ):
            panels, points = _rule_size(width, np.ptp(distance) / C)
            wave = np.zeros(40)
            rule = _slice_shares(pressure[:, b], relative, wave, width, int(panels), int(points))
            pairs = _pair_shares(pressure[:, b], relative, wave, width)
            for shares in (rule, pairs):
                assert np.abs(shares - exact[:, b]).max() <= 1e-10 * np.abs(exact).max()


def test_a_wave_too_big_to_take_at_once_is_taken_a_few_pieces_at_a_time(monkeypatch) -> None:
    """A curved surface's event of 200,000 terms would take gigabytes at once."""
    rng = np.random.default_rng(7)
    pressure, distance = _pieces(rng, 300, 2000.0)
    exact = _pairwise(pressure, distance)[:, 3]
    relative = (distance - 0.5 * (distance.min() + distance.max())) / C
    panels, points = (int(v) for v in _rule_size(SLICE_HZ[3], np.ptp(distance) / C))
    monkeypatch.setattr(clusters_module, "_VALUES", 7 * points)
    shares = _slice_shares(pressure[:, 3], relative, np.zeros(300), SLICE_HZ[3], panels, points)
    assert panels > 1 and np.abs(shares - exact).max() <= 1e-10 * np.abs(exact).max()


def test_a_wave_of_many_pieces_arriving_together_is_their_coherent_sum() -> None:
    """The facets of a curved surface give events of hundreds of thousands of terms."""
    rng = np.random.default_rng(2)
    count = 20_000
    pressure, _ = _pieces(rng, count, 0.0)
    clusters = Clusters(SLICE_HZ, C, 10.0)
    clusters.add(np.arange(count), np.zeros(count, dtype=np.int64), pressure, 10.0)
    energy, has = clusters.resolve(count)
    coherent = np.mean(np.abs(pressure.sum(axis=0)) ** 2, axis=1)
    assert has.all() and (energy >= 0.0).all()
    np.testing.assert_allclose(energy.sum(axis=0), coherent, rtol=1e-5)


def test_the_specular_path_takes_the_cross_terms_of_its_wave() -> None:
    """Its pieces keep their own energy; where the cross terms take more than the path
    has, the rest comes off them in proportion to their energy."""
    rng = np.random.default_rng(3)
    pressure, distance = _pieces(rng, 6, 0.5)
    clusters = Clusters(SLICE_HZ, C, 10.0)
    keys = np.zeros(6, dtype=np.int64)
    clusters.add([0], keys[:1], pressure[:1], distance[:1], anchor=True)
    clusters.add(np.arange(1, 6), keys[1:], pressure[1:], distance[1:])
    energy, _ = clusters.resolve(6)
    whole = _pairwise(pressure, distance).sum(axis=0)
    own = np.mean(np.abs(pressure) ** 2, axis=2)
    left = whole - own[1:].sum(axis=0)
    rtol = 1e-6  # the pieces are kept in single precision
    np.testing.assert_allclose(energy.sum(axis=0), whole, rtol=rtol)
    enough = left >= 0.0
    np.testing.assert_allclose(energy[0, enough], left[enough], rtol=rtol, atol=1e-6)
    np.testing.assert_allclose(energy[1:, enough], own[1:, enough], rtol=rtol)
    assert (energy[0, ~enough] == 0.0).all() and (~enough).any()
    share = energy[1:, ~enough] / own[1:, ~enough]
    np.testing.assert_allclose(share, np.broadcast_to(share[:1], share.shape), rtol=rtol)


def test_the_other_waves_share_their_energy_none_below_nothing() -> None:
    rng = np.random.default_rng(4)
    pressure, distance = _pieces(rng, 12, 3.0)
    clusters = Clusters(SLICE_HZ, C, 10.0)
    clusters.add(np.arange(12), np.zeros(12, dtype=np.int64), pressure, distance)
    energy, has = clusters.resolve(12)
    whole = _pairwise(pressure, distance).sum(axis=0)
    assert has.all() and (energy >= 0.0).all()
    np.testing.assert_allclose(energy.sum(axis=0), whole, rtol=1e-6)


def test_a_wave_of_one_piece_keeps_its_energy_exactly() -> None:
    rng = np.random.default_rng(5)
    pressure, distance = _pieces(rng, 3, 1.0)
    clusters = Clusters(SLICE_HZ, C, 10.0)
    keys = clusters.geometric(np.eye(3), [0, 1, 2])
    own = np.mean(np.abs(pressure) ** 2, axis=2)
    clusters.add(np.arange(3), keys, pressure, distance, anchor=True, own=own)
    clusters.alone([1], own[1:2])
    energy, has = clusters.resolve(4)
    assert np.array_equal(energy[[0, 2]], own[[0, 2]]) and has.tolist() == [1, 1, 1, 0]
    assert np.array_equal(energy[1], 2.0 * own[1])


def test_rows_left_out_are_given_nothing() -> None:
    """A piece whose row is left out (a path under the floor) is part of its wave, but
    its energy goes nowhere; in the wave of a specular path, the path keeps the cross
    terms with it."""
    rng = np.random.default_rng(6)
    pressure, distance = _pieces(rng, 4, 0.2)
    clusters = Clusters(SLICE_HZ, C, 10.0)
    keys = np.zeros(4, dtype=np.int64)
    clusters.add([0], keys[:1], pressure[:1], distance[:1], anchor=True)
    clusters.add([1, 2, 3], keys[1:], pressure[1:], distance[1:])
    clusters.renumber(np.array([0, 1, -1, 2]))
    energy, has = clusters.resolve(3)
    whole = _pairwise(pressure, distance).sum(axis=0)
    own = np.mean(np.abs(pressure) ** 2, axis=2)
    left = whole - own[1:].sum(axis=0)
    enough = left >= 0.0
    assert has.tolist() == [True, True, True] and enough.any()
    np.testing.assert_allclose(energy[0, enough], left[enough], rtol=1e-6, atol=1e-6)


def test_images_this_close_are_one_wave_and_orders_apart_are_not() -> None:
    clusters = Clusters(SLICE_HZ, C, 10.0)
    image = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 1e-9], [1.0, 2.0, 3.0]])
    keys = clusters.geometric(image, [1, 1, 3])
    assert keys[0] == keys[1] != keys[2]
    bent = clusters.diffracted(image[:2], image[::-1][:2])
    assert bent[0] == bent[1] and bent[0] not in keys
    assert len(set(clusters.private(3)) | set(clusters.private(2))) == 5
