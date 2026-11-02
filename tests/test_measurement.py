import math
import sys

import numpy as np
import pytest

from acoustic_spaces import measurement
from acoustic_spaces.measurement import Sweep, deconvolve, measure, remove_latency
from acoustic_spaces.parameters import from_impulse_response, octave_bands

FS = 48_000


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    n_fft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, n_fft) * np.fft.rfft(b, n_fft), n_fft)[:n]


def _room(t60: float = 0.6, length_s: float = 1.0, delay: int = 240) -> np.ndarray:
    """A direct sound followed by exponentially decaying noise."""
    rng = np.random.default_rng(7)
    t = np.arange(int(length_s * FS)) / FS
    ir = 0.05 * rng.standard_normal(len(t)) * np.exp(-3.0 * math.log(10.0) * t / t60)
    ir[:delay] = 0.0
    ir[delay] = 1.0
    return ir


def _sweep(**kwargs) -> Sweep:
    settings = dict(sample_rate=FS, duration_s=1.5, silence_s=1.2, f_start=40.0)
    return Sweep(**{**settings, **kwargs})


def test_sweep_is_synchronized_faded_and_scaled() -> None:
    sweep = _sweep(level_db=-6.0)
    # Synchronized: f_start * L is a whole number of cycles.
    cycles = sweep.f_start * sweep.rate
    assert cycles == pytest.approx(round(cycles))
    x = sweep.sweep()
    assert len(x) == sweep.sweep_length
    assert abs(x[0]) < 1e-3 and abs(x[-1]) < 1e-3
    assert np.max(np.abs(x)) == pytest.approx(10 ** (-6 / 20), rel=1e-3)
    signal = _sweep(repeats=3).signal()
    assert len(signal) == 3 * sweep.period
    np.testing.assert_array_equal(signal[sweep.period : sweep.period + len(x)], x)


def test_sweep_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        Sweep(sample_rate=FS, f_stop=30_000.0)
    with pytest.raises(ValueError):
        Sweep(repeats=0)


def test_sweep_save_and_load(tmp_path) -> None:
    from acoustic_spaces.audio import read_wav

    sweep = _sweep(repeats=2)
    sweep.save(tmp_path / "sweep.wav")
    assert Sweep.load(tmp_path / "sweep.wav") == sweep
    data, rate = read_wav(tmp_path / "sweep.wav")
    assert rate == FS
    np.testing.assert_allclose(data[:, 0], sweep.signal(), atol=1e-7)


def test_deconvolution_recovers_a_room_despite_distortion_and_noise() -> None:
    sweep = _sweep(repeats=2)
    room = _room()
    x = sweep.signal()
    driven = x + 0.1 * x**2 + 0.05 * x**3  # a distorting loudspeaker, then the room
    rng = np.random.default_rng(1)
    recording = _convolve(driven, room)[: len(x)] + 1e-4 * rng.standard_normal(len(x))
    ir = deconvolve(recording, sweep, length_s=1.0)[:, 0]
    assert ir.shape == (FS,)
    assert np.argmax(np.abs(ir)) == 240
    # Inside the swept band its error energy is under 0.1 % of the room's in each band.
    for band in (2, 4, 6):  # 250 Hz, 1 kHz, 4 kHz
        want = octave_bands(room, FS)[band]
        got = octave_bands(ir, FS)[band]
        assert np.sum((got - want) ** 2) < 1e-3 * np.sum(want**2)
    np.testing.assert_allclose(
        from_impulse_response(ir, FS).t30[2:], from_impulse_response(room, FS).t30[2:], rtol=0.03
    )


def test_harmonic_distortion_lands_before_the_linear_response() -> None:
    sweep = _sweep()
    x = sweep.signal()
    delay = FS // 2  # the linear response lands here
    late = np.concatenate([np.zeros(delay), x])
    linear = deconvolve(late, sweep, length_s=1.0)[:, 0]
    distorted = deconvolve(late + 0.3 * late**2, sweep, length_s=1.0)[:, 0]
    peak = np.abs(linear).max()
    assert np.argmax(np.abs(distorted)) == delay
    difference = np.abs(distorted - linear)
    second = delay - int(round(sweep.harmonic_delay_s(2) * FS))
    assert np.argmax(difference) == pytest.approx(second, abs=2)
    assert difference[second - 1 : second + 2].max() > 0.05 * peak
    assert difference[delay - 100 :].max() < 1e-3 * peak


@pytest.mark.parametrize("power", [3, 2])
def test_repeats_keep_the_next_sweeps_distortion_out_of_a_long_response(power: int) -> None:
    # A loudspeaker distorting before a room that is a unit impulse. Harmonic n of the
    # next sweep lands L ln n before its linear response, a period on: from a lag of the
    # silence (n = f_stop / f_start) to L ln 2 short of the period. A response asked for
    # longer than the silence was cut only at the 2nd harmonic's peak, and took in the
    # others: 0.64 % of the linear peak for x^3, 1.3 % for x^2.
    sweep = Sweep(
        sample_rate=FS, duration_s=1.0, f_start=20.0, f_stop=20_000.0, silence_s=0.5, repeats=2
    )
    x = sweep.signal()
    y = x + 0.2 * x**power
    after = int(0.02 * FS)
    floor = np.abs(deconvolve(y, sweep)[after:, 0]).max()  # 0.06 % and 0.09 %
    long = deconvolve(y, sweep, length_s=2.0)[:, 0]
    assert np.abs(long[after:]).max() <= floor
    assert len(long) == round(sweep.silence_s * FS)


def test_one_sweep_gives_the_response_the_recording_holds() -> None:
    # Frequency f of the response at a lag needs the recording to run on L ln(f / f_start)
    # past it: a recording that stops with the excitation holds the whole band up to the
    # silence, and one that goes on for a second, up to a second more.
    sweep = _sweep()
    room = _room(t60=2.0, length_s=3.0)
    x = sweep.signal()
    heard = _convolve(x, room)
    silence = round(sweep.silence_s * FS)
    assert len(deconvolve(heard[: len(x)], sweep, length_s=2.0)) == silence
    assert len(deconvolve(heard[: len(x) + FS // 2], sweep, length_s=2.0)) == silence + FS // 2
    ir = deconvolve(heard[: len(x) + FS], sweep, length_s=2.0)[:, 0]
    assert len(ir) == 2 * FS
    # Past the silence its error energy is under 0.1 % of the room's, as before it.
    tail = slice(silence, None)
    for band in (2, 4, 6, 7):  # 250 Hz, 1 kHz, 4 kHz, 8 kHz
        want = octave_bands(room[: len(ir)], FS)[band, tail]
        got = octave_bands(ir, FS)[band, tail]
        assert np.sum((got - want) ** 2) < 1e-3 * np.sum(want**2)


def test_deconvolution_takes_a_reference_read_from_a_wav_file() -> None:
    # read_wav gives a mono loopback recording as (samples, 1). A short, narrow sweep
    # keeps what the old code made of it (a spectrum per sample) small.
    sweep = _sweep(duration_s=0.05, silence_s=0.05, f_start=1000.0, f_stop=2000.0)
    recording = sweep.signal()
    expected = deconvolve(recording, sweep)
    got = deconvolve(recording, sweep, reference=sweep.sweep()[:, None])
    np.testing.assert_array_equal(got, expected)
    with pytest.raises(ValueError, match="one channel"):
        deconvolve(recording, sweep, reference=np.zeros((10, 2)))


def test_repeats_average_out_noise() -> None:
    rng = np.random.default_rng(2)
    floors = []
    for repeats in (1, 4):
        sweep = _sweep(repeats=repeats)
        noise = 1e-3 * rng.standard_normal(len(sweep.signal()))
        floors.append(np.mean(deconvolve(noise, sweep, length_s=1.0) ** 2))
    assert 10 * math.log10(floors[0] / floors[1]) == pytest.approx(6.0, abs=1.0)


def test_remove_latency_drops_the_loopback_column() -> None:
    responses = np.zeros((100, 2))
    responses[30, 0] = 1.0  # room, 30 samples in
    responses[10, 1] = 1.0  # loopback, 10 samples of latency
    aligned, lag = remove_latency(responses, 1)
    assert lag == 10
    assert aligned.shape == (100, 1)
    assert np.argmax(aligned[:, 0]) == 20


def _fake_device(room: np.ndarray, latency: int):
    """A play_record that sends the signal through ``room`` to input 1 and straight
    back on input 2, both late by ``latency`` samples."""

    def play_record(signal, sample_rate, outputs, inputs, device):
        assert sample_rate == FS and list(outputs) == [1]
        late = np.concatenate([np.zeros(latency), signal])[: len(signal)]
        channels = {1: _convolve(late, room)[: len(signal)], 2: late}
        return np.column_stack([channels[k] for k in inputs])

    return play_record


def test_measure_removes_the_latency_of_the_interface() -> None:
    sweep = _sweep()
    result = measure(
        sweep, inputs=[1], loopback=2, length_s=1.0, play_record=_fake_device(_room(), 96)
    )
    assert result.latency_s == pytest.approx(96 / FS)
    assert result.impulse_responses.shape == (FS, 1)
    assert result.recording.shape == (len(sweep.signal()), 2)
    assert np.argmax(np.abs(result.impulse_responses[:, 0])) == 240


def test_measure_without_sounddevice_says_how_to_install(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(ImportError, match=r'pip install "acoustic-spaces\[measure\]"'):
        measurement.list_devices()
