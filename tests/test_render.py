import math
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.render import Convolver, auralize, convolve, play, power_gains, resample
from acoustic_spaces.spatial import Binaural, Loudspeakers, Mono

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


def test_convolve_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(70_000)
    h = rng.standard_normal((5_000, 3))
    y = convolve(x, h)
    assert y.shape == (len(x) + len(h) - 1, 3)
    for c in range(3):
        np.testing.assert_allclose(y[:, c], np.convolve(x, h[:, c]), atol=1e-9)
    stereo = rng.standard_normal((300, 2))
    y = convolve(stereo, h[:40, 0])
    assert y.shape == (339, 2)
    np.testing.assert_allclose(y[:, 1], np.convolve(stereo[:, 1], h[:40, 0]), atol=1e-12)
    np.testing.assert_allclose(convolve([1.0, 2.0], [0.5, 1.0]), [0.5, 2.0, 2.0])


def test_convolve_rejects_two_multichannel_inputs() -> None:
    with pytest.raises(ValueError, match="mono response"):
        convolve(np.zeros((10, 2)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        convolve(np.zeros(0), np.zeros(5))


@pytest.mark.parametrize(
    ("rate_in", "rate_out"),
    [(44_100, 48_000), (48_000, 44_100), (96_000, 48_000), (48_000, 47_999.5)],
)
def test_resample_keeps_tones(rate_in: float, rate_out: float) -> None:
    t = np.arange(int(rate_in)) / rate_in
    tones = (1_000.0, 15_000.0)
    x = np.stack([np.sin(2 * math.pi * f * t) for f in tones], axis=1)
    y = resample(x, rate_in, rate_out)
    assert y.shape == (math.ceil(len(x) * rate_out / rate_in), 2)
    t = np.arange(len(y)) / rate_out
    middle = slice(len(y) // 10, len(y) * 9 // 10)  # away from the edges
    for c, f in enumerate(tones):
        np.testing.assert_allclose(y[middle, c], np.sin(2 * math.pi * f * t[middle]), atol=1e-3)


def test_resample_filters_what_the_new_rate_cannot_hold() -> None:
    t = np.arange(96_000) / 96_000
    y = resample(np.sin(2 * math.pi * 30_000 * t), 96_000, 44_100)
    assert np.abs(y[2_000:-2_000]).max() < 1e-4
    np.testing.assert_array_equal(resample(np.arange(4.0), 48_000, 48_000), np.arange(4.0))


def test_native_fft_matches_numpy() -> None:
    from acoustic_spaces import _native

    rng = np.random.default_rng(2)
    for n in (4, 32, 4096):
        x = rng.standard_normal(n)
        np.testing.assert_allclose(_native.real_fft(x), np.fft.rfft(x), atol=1e-11)
        np.testing.assert_allclose(_native.inverse_real_fft(np.fft.rfft(x), n), x, atol=1e-13)
    with pytest.raises(ValueError, match="power of two"):
        _native.real_fft(np.zeros(12))


@pytest.mark.parametrize("block", [64, 1024])
def test_convolver_matches_offline_convolution(block: int) -> None:
    rng = np.random.default_rng(3)
    ir = rng.standard_normal((1_000, 2, 3))  # taps, inputs, outputs
    x = rng.standard_normal((3_000, 2))
    convolver = Convolver(ir, block)
    assert (convolver.inputs, convolver.outputs, convolver.taps) == (2, 3, 1_000)
    y = convolver.run(x)
    expected = sum(convolve(x[:, i], ir[:, i, :]) for i in range(2))
    assert y.dtype == np.float32
    assert y.shape == expected.shape
    np.testing.assert_allclose(y, expected, atol=1e-5 * np.abs(expected).max())


@pytest.mark.parametrize(
    ("growth", "max_block"), [(1, 8192), (2, 8192), (8, 8192), (4, 64), (16, 1 << 20)]
)
def test_partitions_of_every_size_add_up(growth: int, max_block: int) -> None:
    from acoustic_spaces import _native

    rng = np.random.default_rng(6)
    ir = rng.standard_normal((1, 2, 3_000)).astype(np.float32)
    x = rng.standard_normal((8_000, 1)).astype(np.float32)
    convolver = _native.Convolver(ir, 16, growth, max_block)
    starts = [start for start, _, _ in convolver.segments]
    assert starts[0] == 0
    for (start, size, parts), nxt in zip(convolver.segments, [*starts[1:], 3_000], strict=True):
        assert start in (0, 2 * size) and size <= max(16, max_block)
        assert (parts - 1) * size < nxt - start <= parts * size
    y = np.concatenate([convolver.process(b) for b in x.reshape(-1, 16, 1)])
    expected = convolve(x[:, 0], ir[0].T)[: len(x)]
    np.testing.assert_allclose(y, expected, atol=1e-5 * np.abs(expected).max())


def test_long_responses_get_longer_partitions() -> None:
    segments = Convolver(np.ones((144_000, 2)), 128).segments
    assert segments == [(0, 128, 16), (2_048, 1_024, 14), (16_384, 8_192, 16)]


def test_convolver_streams_and_resets() -> None:
    rng = np.random.default_rng(4)
    ir = rng.standard_normal((500, 2))
    x = rng.standard_normal(256 * 4).astype(np.float32)
    convolver = Convolver(ir, 256)
    first = np.concatenate([convolver.process(b) for b in x.reshape(4, 256)])
    np.testing.assert_allclose(first, convolve(x, ir)[: len(x)], atol=1e-4)
    convolver.reset()
    np.testing.assert_array_equal(convolver.process(x[:256]), first[:256])


def test_convolver_writes_into_an_array_it_is_given() -> None:
    rng = np.random.default_rng(7)
    ir = rng.standard_normal((300, 2))
    x = rng.standard_normal(256 * 3).astype(np.float32)
    plain, into = Convolver(ir, 256), Convolver(ir, 256)
    out = np.empty((256, 2), np.float32)
    for b in x.reshape(3, 256):
        assert into.process(b, out=out) is out
        np.testing.assert_array_equal(out, plain.process(b))
    read_only = np.empty((256, 2), np.float32)
    read_only.flags.writeable = False
    for bad in (
        np.empty((256, 2)),
        np.empty((2, 256), np.float32).T,
        read_only,
        np.empty((128, 2), np.float32),
    ):
        with pytest.raises(ValueError, match="out"):
            into.process(x[:256], out=bad)


def test_convolver_rejects_bad_settings() -> None:
    from acoustic_spaces import _native

    with pytest.raises(ValueError, match="power of two"):
        Convolver(np.ones(10), 100)
    with pytest.raises(ValueError, match="growth"):
        _native.Convolver(np.ones((1, 1, 10), np.float32), 64, 3)
    with pytest.raises(ValueError, match="taps"):
        Convolver(np.ones((2, 2, 2, 2)), 64)
    convolver = Convolver(np.ones((10, 2)), 64)
    with pytest.raises(ValueError, match="block"):
        convolver.process(np.zeros(32))
    with pytest.raises(ValueError, match="channel"):
        convolver.run(np.zeros((100, 2)))


@pytest.fixture(scope="module")
def result():
    from acoustic_spaces.engine import simulate
    from acoustic_spaces.room import Room

    room = Room.from_yaml(SHOEBOX)
    settings = room.scene.simulation.model_copy(deep=True)
    settings.late.rays = 1_000
    settings.ir_length_s = 0.3
    return simulate(room, settings=settings)


def test_auralize_places_each_source(result) -> None:
    fs = result.settings.sample_rate
    click = np.zeros(100)
    click[10] = 1.0
    mix = auralize(result, {"talker": click}, fmt=Binaural())
    ir = result["talker", "seat-a"].spatial_impulse_response(Binaural(), fs)
    assert mix.shape == (len(ir) + 99, 2)
    np.testing.assert_allclose(mix[10 : 10 + len(ir)], ir, atol=1e-12)
    # A signal at another rate is resampled first.
    slow = np.zeros(50)
    slow[5] = 1.0
    layout = Loudspeakers.layout("5.0")
    assert auralize(result, {"talker": (slow, fs // 2)}, "seat-a", layout).shape[1] == 5
    with pytest.raises(ValueError, match="no response from 'piano'"):
        auralize(result, {"piano": click})
    with pytest.raises(ValueError, match="no signals"):
        auralize(result, {})


def test_auralize_keeps_the_sources_power(two_sources) -> None:
    fs = two_sources.settings.sample_rate
    click = np.zeros(100)
    click[10] = 1.0
    talker = two_sources["talker", "seat-a"].spatial_impulse_response(Mono(), fs)
    radio = two_sources["radio", "seat-a"].spatial_impulse_response(Mono(), fs)
    n = max(len(talker), len(radio))
    expected = np.zeros((n + 99, 1))
    expected[10 : 10 + len(talker)] += talker
    expected[10 : 10 + len(radio)] += 0.1 * radio  # 20 dB less sound power
    mix = auralize(two_sources, {"talker": click, "radio": click})
    np.testing.assert_allclose(mix, expected, atol=1e-12)
    # Alone, or all at one power, a source plays as its response is.
    alone = auralize(two_sources, {"radio": click})
    np.testing.assert_allclose(alone[10 : 10 + len(radio)], radio, atol=1e-12)
    same = auralize(two_sources, {"talker": click, "radio": click}, equal_power=True)
    expected[10 : 10 + len(radio)] += 0.9 * radio
    np.testing.assert_allclose(same, expected, atol=1e-12)
    assert power_gains({"a": 100.0, "b": 94.0}, ["a", "b", "c"]) == pytest.approx(
        {"a": 1.0, "b": 10 ** (-6 / 20)}
    )
    assert power_gains(two_sources.room.scene.sources, ["radio"]) == {"radio": 1.0}


def test_play_runs_a_signal_through_the_response(fake_sounddevice) -> None:
    rng = np.random.default_rng(5)
    ir = rng.standard_normal((300, 2)) * 0.1
    signal = rng.standard_normal(1_000)
    report = play(ir, 48_000, signal, block=128, gain=0.5)
    played = np.concatenate(fake_sounddevice.played)
    expected = 0.5 * convolve(signal, ir)
    assert report.frames == len(played) == 1_408  # 11 blocks hold signal and tail
    assert report.xruns == 1
    assert report.load > 0
    np.testing.assert_allclose(played[: len(expected)], expected, atol=1e-5)
    np.testing.assert_allclose(played[len(expected) :], 0.0, atol=1e-6)


def test_play_live_input_for_a_while(fake_sounddevice) -> None:
    ir = np.array([0.0, 1.0, 0.5])
    fake_sounddevice.live_input = np.ones((1_000, 1), np.float32)
    report = play(ir, 1_000, block=64, seconds=0.5)
    played = np.concatenate(fake_sounddevice.played)
    assert report.frames == len(played) == 512
    np.testing.assert_allclose(played[:4, 0], [0.0, 1.0, 1.5, 1.5], atol=1e-6)


def test_play_without_sounddevice_says_how_to_install(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(ImportError, match="measure"):
        play(np.ones(4), 48_000, np.ones(10))
