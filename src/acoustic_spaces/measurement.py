"""Measuring impulse responses with exponential sine sweeps.

The usual chain is: generate an excitation (:class:`Sweep`), play it through a
loudspeaker while recording one or more microphones, deconvolve the recording into an
impulse response, then analyse it with :mod:`acoustic_spaces.parameters`, the same
code that analyses simulated responses.

- :class:`Sweep` is a synchronized exponential sweep (Novak et al., 2015): its
  harmonic distortion lands in separate impulse responses *before* the linear one,
  so deconvolution leaves the linear response clean. The excitation repeats the sweep
  with silence after each one, long enough for the room to decay.
- :func:`deconvolve` divides spectra with Kirkeby regularisation outside the swept band,
  then averages the repeats, which lowers background noise by 3 dB per doubling.
- :func:`measure` plays and records through ``sounddevice`` (``pip install
  "acoustic-spaces[measure]"``) and returns the deconvolved responses. A loopback
  channel (an output wired straight back to an input) removes the latency of the
  audio interface.

Rules of thumb: make the silence after each sweep longer than the reverberation time,
and the sweep itself about ten times longer than the reverberation time if distortion
products must not overlap the linear response at all.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Regularisation inside and outside the swept band, relative to the peak power.
_EPS_IN, _EPS_OUT = 1e-8, 1.0


def _ramp(n: int) -> np.ndarray:
    """A raised-cosine fade-in of ``n`` samples."""
    return np.sin(0.5 * math.pi * (np.arange(n) + 0.5) / n) ** 2


@dataclass(frozen=True)
class Sweep:
    """An exponential sine sweep excitation, repeated with silence after each sweep."""

    sample_rate: int = 48_000
    duration_s: float = 6.0
    f_start: float = 20.0
    f_stop: float = 20_000.0
    level_db: float = -6.0  # peak level, dB re full scale
    silence_s: float = 3.0  # after each sweep, for the room to decay
    repeats: int = 1
    fade_in_s: float = 0.05
    fade_out_s: float = 0.005

    def __post_init__(self) -> None:
        if not 0 < self.f_start < self.f_stop < self.sample_rate / 2:
            raise ValueError("need 0 < f_start < f_stop < sample_rate / 2")
        if self.duration_s <= 0 or self.silence_s < 0 or self.repeats < 1:
            raise ValueError("need duration_s > 0, silence_s >= 0 and repeats >= 1")

    @property
    def rate(self) -> float:
        """L of x(t) = sin(2 pi f_start L exp(t / L)), rounded so that harmonics stay in
        phase with the fundamental (the synchronized sweep)."""
        ratio = math.log(self.f_stop / self.f_start)
        cycles = max(1, round(self.f_start * self.duration_s / ratio))
        return cycles / self.f_start

    @property
    def sweep_length(self) -> int:
        """Samples in one sweep (the duration after synchronization)."""
        return int(round(self.rate * math.log(self.f_stop / self.f_start) * self.sample_rate))

    @property
    def period(self) -> int:
        """Samples from one sweep's start to the next."""
        return self.sweep_length + int(round(self.silence_s * self.sample_rate))

    def harmonic_delay_s(self, order: int) -> float:
        """How far before the linear response the ``order``-th harmonic's response lands."""
        return self.rate * math.log(order)

    def sweep(self) -> np.ndarray:
        """One sweep, faded in and out."""
        t = np.arange(self.sweep_length) / self.sample_rate
        x = np.sin(2.0 * math.pi * self.f_start * self.rate * np.exp(t / self.rate))
        n_in = min(int(round(self.fade_in_s * self.sample_rate)), len(x) // 2)
        n_out = min(int(round(self.fade_out_s * self.sample_rate)), len(x) // 2)
        if n_in:
            x[:n_in] *= _ramp(n_in)
        if n_out:
            x[-n_out:] *= _ramp(n_out)[::-1]
        return x * 10.0 ** (self.level_db / 20.0)

    def signal(self) -> np.ndarray:
        """The whole excitation: every repeat, each followed by its silence."""
        out = np.zeros(self.period * self.repeats)
        one = self.sweep()
        for k in range(self.repeats):
            out[k * self.period : k * self.period + len(one)] = one
        return out

    def save(self, path: str | Path) -> None:
        """Write the excitation as a float WAV and its settings next to it as JSON."""
        from acoustic_spaces.audio import write_wav

        path = Path(path)
        write_wav(path, self.signal(), self.sample_rate)
        path.with_suffix(".json").write_text(
            json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Sweep:
        """Read the settings of a saved excitation (its .json, or the .wav next to it)."""
        data = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
        return cls(**data)


def _band_regularisation(n_fft: int, sweep: Sweep) -> np.ndarray:
    """Small inside the swept band, large outside, with half-octave transitions."""
    f = np.fft.rfftfreq(n_fft, 1.0 / sweep.sample_rate)
    log_f = np.log2(np.maximum(f, 1e-6))
    below = np.clip((np.log2(sweep.f_start) - log_f) / 0.5, 0.0, 1.0)
    above = np.clip((log_f - np.log2(sweep.f_stop)) / 0.5, 0.0, 1.0)
    out_of_band = np.sin(0.5 * math.pi * np.maximum(below, above)) ** 2
    return _EPS_IN + (_EPS_OUT - _EPS_IN) * out_of_band


def deconvolve(
    recording: np.ndarray,
    sweep: Sweep,
    length_s: float | None = None,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Impulse responses (samples, channels) from a recording of ``sweep.signal()``.

    ``recording`` is (samples,) or (samples, channels), starting when playback started.
    Each repeat is deconvolved and the repeats are averaged. ``length_s`` defaults to
    the silence after each sweep. ``reference`` replaces the generated sweep, for
    instance with a loopback recording of one sweep.
    """
    recording = np.asarray(recording, dtype=float)
    if recording.ndim == 1:
        recording = recording[:, None]
    excitation = sweep.sweep() if reference is None else np.asarray(reference, dtype=float)
    length = int(round((sweep.silence_s if length_s is None else length_s) * sweep.sample_rate))
    length = max(1, min(length, sweep.period - int(sweep.harmonic_delay_s(2) * sweep.sample_rate)))

    needed = sweep.period * sweep.repeats
    if len(recording) < needed:
        recording = np.pad(recording, ((0, needed - len(recording)), (0, 0)))
    n_fft = 1 << int(math.ceil(math.log2(len(recording) + len(excitation))))
    x = np.fft.rfft(excitation, n_fft)
    power = np.abs(x) ** 2
    inverse = np.conj(x) / (power + _band_regularisation(n_fft, sweep) * power.max())
    full = np.fft.irfft(np.fft.rfft(recording, n_fft, axis=0) * inverse[:, None], n_fft, axis=0)

    out = np.zeros((length, recording.shape[1]))
    for k in range(sweep.repeats):
        start = k * sweep.period
        out += full[start : start + length]
    return out / sweep.repeats


@dataclass
class Measurement:
    """Measured impulse responses, one column per microphone."""

    impulse_responses: np.ndarray  # (samples, channels)
    sample_rate: int
    recording: np.ndarray  # (samples, channels), raw
    sweep: Sweep
    #: Round-trip latency removed using the loopback channel (s), or None.
    latency_s: float | None = None


PlayRecord = Callable[[np.ndarray, int, Sequence[int], Sequence[int], object], np.ndarray]


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:  # OSError: PortAudio library missing
        raise ImportError(
            'playing and recording needs sounddevice: pip install "acoustic-spaces[measure]"'
        ) from exc
    return sounddevice


def play_and_record(
    signal: np.ndarray,
    sample_rate: int,
    outputs: Sequence[int],
    inputs: Sequence[int],
    device: object = None,
) -> np.ndarray:
    """Play ``signal`` on the output channels and record the inputs (1-based channels).

    Returns (samples, len(inputs)), as long as the signal.
    """
    sd = _sounddevice()
    out = np.repeat(np.asarray(signal, dtype=np.float32)[:, None], len(outputs), axis=1)
    recorded = sd.playrec(
        out,
        samplerate=sample_rate,
        input_mapping=list(inputs),
        output_mapping=list(outputs),
        device=device,
        blocking=True,
    )
    return np.asarray(recorded, dtype=float)


def list_devices() -> str:
    """The audio devices sounddevice sees, as text."""
    return str(_sounddevice().query_devices())


def measure(
    sweep: Sweep,
    *,
    inputs: Sequence[int] = (1,),
    output: int = 1,
    loopback: int | None = None,
    device: object = None,
    length_s: float | None = None,
    play_record: PlayRecord | None = None,
) -> Measurement:
    """Play ``sweep`` on one output, record ``inputs`` and deconvolve them.

    Channels are 1-based. With ``loopback`` (an input wired to the output), the latency
    of the audio interface is measured on it and removed from every response.
    ``play_record`` replaces the audio device, for tests or other audio back ends.
    """
    play_record = play_record or play_and_record
    channels = list(inputs) + ([loopback] if loopback is not None else [])
    recording = play_record(sweep.signal(), sweep.sample_rate, [output], channels, device)
    recording = np.asarray(recording, dtype=float).reshape(len(recording), len(channels))
    responses = deconvolve(recording, sweep, length_s=length_s)
    latency = None
    if loopback is not None:
        responses, lag = remove_latency(responses, len(channels) - 1)
        latency = lag / sweep.sample_rate
    return Measurement(
        impulse_responses=responses,
        sample_rate=sweep.sample_rate,
        recording=recording,
        sweep=sweep,
        latency_s=latency,
    )


def remove_latency(responses: np.ndarray, loopback: int) -> tuple[np.ndarray, int]:
    """Move responses (samples, channels) earlier by the delay of a loopback channel.

    ``loopback`` is the 0-based column of the loopback response, which is dropped. Returns
    the other columns, padded with zeros at the end, and the delay in samples.
    """
    responses = np.asarray(responses, dtype=float)
    lag = int(np.argmax(np.abs(responses[:, loopback])))
    others = np.delete(responses, loopback, axis=1)
    return np.vstack([others[lag:], np.zeros((lag, others.shape[1]))]), lag


__all__ = [
    "Measurement",
    "Sweep",
    "deconvolve",
    "list_devices",
    "measure",
    "play_and_record",
    "remove_latency",
]
