"""Rendering: impulse responses applied to dry signals (auralisation).

- :func:`convolve`: offline convolution of a recording with an impulse response (FFT
  overlap-add in float64), for any number of response channels.
- :func:`resample`: band-limited sample-rate conversion, so a 44.1 kHz recording can go
  through a 48 kHz response.
- :func:`auralize`: the sources of a simulation each playing their own signal, heard at
  one listener in a spatial format (mono, binaural, Ambisonics, loudspeakers), summed.
  Outdoors in turbulence the early paths change as the eddies drift past: a new
  moment of them every :data:`MOMENTS_PER_CORRELATION`-th of their correlation time (at
  most every :data:`MOMENT_MIN_S`), each path's delay going from one moment's to the
  next's, so the sound flickers as it does in gusty air.
- :class:`Convolver`: the compiled real-time convolver, block by block. It divides the
  response into partitions that grow along it, from the block size up to 8192 frames,
  and multiplies in the frequency domain, so the only latency is the block itself;
  nothing allocates while it runs.
- :func:`play`: a signal, or live input, through a response on the sound card (needs the
  ``measure`` extra: sounddevice); or through any block processor with one input, such
  as :class:`~acoustic_spaces.headtracking.HeadTrackedBinaural`.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

if TYPE_CHECKING:
    from acoustic_spaces.engine.simulate import SimulationResult
    from acoustic_spaces.headtracking import HeadTrackedBinaural
    from acoustic_spaces.spatial import Format

#: Smallest FFT of the offline overlap-add.
_MIN_FFT = 1 << 14
#: Zero crossings of the resampling filter on each side, and its Kaiser window's beta.
_RESAMPLE_ZEROS = 32
_RESAMPLE_BETA = 9.0
#: Most filter phases resampling keeps; finer ratios round the position to these.
_MAX_PHASES = 4096
#: Output samples resampled at once when the rates have no short ratio.
_RESAMPLE_CHUNK = 1 << 15
#: In turbulence, the moments of the eddies a signal goes through: this many per
#: correlation time, and at most one every MOMENT_MIN_S seconds.
MOMENTS_PER_CORRELATION = 16
MOMENT_MIN_S = 0.05


def _pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty (samples,) or (samples, channels) array")
    return x


def convolve(signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Full linear convolution: len(signal) + len(ir) - 1 samples.

    ``signal`` and ``ir`` are (samples,) or (samples, channels). A mono signal goes
    through every channel of the response (a dry source through a binaural or Ambisonics
    response); a multichannel signal goes through a mono response channel by channel.
    The result is 1-D only when both inputs are.
    """
    x, h = _as_2d(signal, "signal"), _as_2d(ir, "ir")
    if x.shape[1] > 1 and h.shape[1] > 1:
        raise ValueError(
            "a multichannel signal needs a mono response; mix the signal down, or use "
            "auralize for several sources"
        )
    n, m = len(x), len(h)
    length = n + m - 1
    nfft = min(_pow2(length), max(_pow2(4 * m), _MIN_FFT))
    step = nfft - m + 1
    spectrum = np.fft.rfft(h, nfft, axis=0)
    out = np.zeros((length, max(x.shape[1], h.shape[1])))
    for start in range(0, n, step):
        chunk = np.fft.rfft(x[start : start + step], nfft, axis=0)
        y = np.fft.irfft(chunk * spectrum, nfft, axis=0)
        stop = min(start + nfft, length)
        out[start:stop] += y[: stop - start]
    if np.ndim(signal) == 1 and np.ndim(ir) == 1:
        return out[:, 0]
    return out


def _kaiser_sinc(t: np.ndarray, cutoff: float) -> np.ndarray:
    """Low-pass taps at offsets ``t`` (input samples), ``cutoff`` as a share of Nyquist."""
    half = _RESAMPLE_ZEROS / cutoff
    window = np.i0(_RESAMPLE_BETA * np.sqrt(np.clip(1.0 - (t / half) ** 2, 0.0, 1.0)))
    return cutoff * np.sinc(cutoff * t) * window / np.i0(_RESAMPLE_BETA)


def resample(signal: np.ndarray, rate_in: float, rate_out: float) -> np.ndarray:
    """``signal`` (samples,) or (samples, channels) at ``rate_out``.

    Polyphase windowed-sinc interpolation (Kaiser, 32 zero crossings each side), with
    the cutoff just below the lower Nyquist frequency so nothing aliases.
    """
    x = np.asarray(signal, dtype=np.float64)
    if rate_in <= 0 or rate_out <= 0:
        raise ValueError("sample rates must be positive")
    if math.isclose(rate_in, rate_out) or len(x) == 0:
        return x.copy()
    # Output sample k sits at input position k * step / phases.
    ratio = (Fraction(rate_in) / Fraction(rate_out)).limit_denominator(1 << 20)
    exact = ratio.denominator <= _MAX_PHASES
    step, phases = (ratio.numerator, ratio.denominator) if exact else (0, _MAX_PHASES)
    cutoff = 0.97 * min(1.0, rate_out / rate_in)
    reach = int(math.ceil(_RESAMPLE_ZEROS / cutoff))
    offsets = np.arange(-reach + 1, reach + 1)  # taps relative to floor(position)
    frac = np.arange(phases) / phases
    table = _kaiser_sinc(offsets[None, :] - frac[:, None], cutoff)  # (phases, taps)

    two_d = x.reshape(len(x), -1)
    count = int(math.ceil(len(x) * rate_out / rate_in))
    out = np.zeros((count, two_d.shape[1]))
    for c in range(two_d.shape[1]):
        padded = np.concatenate([np.zeros(reach), two_d[:, c], np.zeros(reach + 1)])
        # Row r holds the taps around input sample r - 1.
        rows = sliding_window_view(padded, len(offsets))
        if exact:
            # Outputs k0, k0 + phases, ... share phase p and step `step` samples apart, so
            # each phase is one matrix-vector product.
            inverse = pow(step, -1, phases)
            for p in range(phases):
                k0 = p * inverse % phases
                if k0 >= count:
                    continue
                n = len(range(k0, count, phases))
                first = (k0 * step - p) // phases + 1
                out[k0::phases, c] = rows[first : first + (n - 1) * step + 1 : step] @ table[p]
            continue
        for start in range(0, count, _RESAMPLE_CHUNK):
            k = np.arange(start, min(start + _RESAMPLE_CHUNK, count))
            position = k * (rate_in / rate_out)
            whole = np.floor(position).astype(np.int64)
            phase = np.round((position - whole) * phases).astype(np.int64)
            whole, phase = whole + phase // phases, phase % phases
            out[k, c] = np.einsum("kt,kt->k", table[phase], rows[whole + 1])
    return out[:, 0] if x.ndim == 1 else out


def power_gains(sources, names) -> dict[str, float]:
    """Amplitude gain of each source in ``names`` for its sound power: 1 for the loudest
    of them, 10^(-ΔL/20) for one ΔL dB below it. ``sources`` are scene sources, or a
    mapping of name to sound power level (dB re 1 pW); names it lacks get no gain."""
    if not isinstance(sources, Mapping):
        sources = {source.name: source.sound_power_db for source in sources}
    levels = {name: float(sources[name]) for name in names if name in sources}
    if not levels:
        return {}
    loudest = max(levels.values())
    return {name: 10.0 ** ((level - loudest) / 20.0) for name, level in levels.items()}


def auralize(
    result: SimulationResult,
    signals: Mapping[str, np.ndarray | tuple[np.ndarray, float]],
    listener: str | None = None,
    fmt: Format | None = None,
    *,
    sample_rate: int | None = None,
    seed: int | None = None,
    equal_power: bool = False,
) -> np.ndarray:
    """Every source in ``signals`` playing its signal, heard at ``listener``.

    ``signals`` maps source names to a signal at ``sample_rate``, or to a
    (signal, rate) pair, resampled when the rates differ; multichannel signals are mixed
    down, as sources are points. ``fmt`` is a format of :mod:`acoustic_spaces.spatial`
    (default mono, through the listener's microphone when it has a pattern:
    :func:`~acoustic_spaces.spatial.for_listener`); ``listener`` may be left out when
    the result has one listener.
    Returns (samples, channels) at ``sample_rate`` (default: the simulation's), with the
    sources at the levels the simulation gives them: the responses carry each source's
    feed (gain, equalizer, delay) but are per unit of sound power, so each signal is
    scaled by its source's sound power (``level_db``) against the loudest source's in
    ``signals`` (:func:`power_gains`). ``equal_power`` leaves that out, as if every
    source had the loudest one's power.
    """
    from acoustic_spaces.spatial import Mono, for_listener

    sample_rate = int(sample_rate or result.settings.sample_rate)
    listeners = sorted({lis for _, lis in result.responses})
    if listener is None:
        if len(listeners) != 1:
            raise ValueError(f"pick a listener: {', '.join(listeners)}")
        listener = listeners[0]
    items = {item.name: item for item in result.room.scene.listeners}
    fmt = for_listener(fmt or Mono(), items.get(listener))
    if not signals:
        raise ValueError("no signals to auralize")
    mix = np.zeros((0, fmt.channels))
    gains = {} if equal_power else power_gains(result.room.scene.sources, signals)
    for source, value in signals.items():
        if (source, listener) not in result.responses:
            sources = sorted({s for s, lis in result.responses if lis == listener})
            raise ValueError(f"no response from {source!r} at {listener!r}; sources: {sources}")
        signal, rate = value if isinstance(value, tuple) else (value, sample_rate)
        dry = _as_2d(signal, f"signal of {source!r}").mean(axis=1) * gains.get(source, 1.0)
        dry = resample(dry, rate, sample_rate)
        response = result[source, listener]
        if flickers(response):
            n = int(math.ceil(response.length_s * sample_rate))
            gusts = gusts_for(response, (len(dry) + n) / sample_rate, seed)
            wet = through_eddies(dry, response, fmt, sample_rate, seed, gusts, 0, len(dry) + n - 1)
        else:
            ir = response.spatial_impulse_response(fmt, sample_rate, seed)
            wet = convolve(dry, ir)
        if len(wet) > len(mix):
            mix = np.concatenate([mix, np.zeros((len(wet) - len(mix), fmt.channels))])
        mix[: len(wet)] += wet
    return mix


def flickers(response) -> bool:
    """Whether ``response``'s early paths change as eddies drift past (outdoors in
    turbulence, without the wave solver's bands). Anything without early paths does
    not."""
    early = getattr(response, "early", None)
    return getattr(early, "wander", None) is not None and getattr(response, "wave", None) is None


def gusts_for(response, duration_s: float, seed: int | None = None):
    """The eddies' moments (:class:`~acoustic_spaces.engine.turbulence.Gusts`) for
    ``duration_s`` seconds of sound through ``response``'s turbulence, drawn with
    ``seed`` (default the response's)."""
    from acoustic_spaces.engine.turbulence import Gusts

    correlation = response.early.wander.correlation_s
    step = max(correlation / MOMENTS_PER_CORRELATION, MOMENT_MIN_S)
    count = int(math.ceil(duration_s / step)) + 2
    return Gusts(correlation, step, count, response.seed if seed is None else seed)


def through_eddies(
    dry: np.ndarray, response, fmt, sample_rate: int, seed, gusts, begin: int, end: int
) -> np.ndarray:
    """(end - begin, channels) samples ``begin`` to ``end`` of the mono ``dry`` signal
    played through ``response`` in ``fmt``: its early paths each through its own delay,
    which the eddies ``gusts`` has move from one moment to the next (taken between the
    moments on either side, as :func:`acoustic_spaces.doppler.render` takes them with
    the one place), its late part as it is. Crossfading the moments' impulse responses
    instead would mix each path at two delays a few microseconds apart, which cancels
    the treble half-way through each crossfade."""
    from acoustic_spaces import doppler
    from acoustic_spaces.engine.early import EarlyReflections

    out = np.zeros((end - begin, fmt.channels))
    quiet = dataclasses.replace(response, early=EarlyReflections.empty(response.late.shape[0]))
    _add_window(out, dry, quiet.spatial_impulse_response(fmt, sample_rate, seed), begin, end, None)
    out += doppler.render(
        dry,
        [response],
        np.zeros(1),
        fmt,
        sample_rate,
        end,
        source_walks=False,
        planes=np.zeros(0, dtype=np.int64),
        step_s=0.0,
        gusts=gusts,
        seed=seed,
        span=(begin, end),
    )
    return out


def _add_window(out, dry, ir, lo: int, hi: int, weight, origin: int | None = None) -> None:
    """Adds samples ``lo`` to ``hi`` of ``dry`` convolved with ``ir``, times ``weight``
    (or 1), to ``out``, whose first sample is sample ``origin`` (default ``lo``)."""
    origin = lo if origin is None else origin
    first = max(lo - len(ir) + 1, 0)
    piece = dry[first:hi]
    if len(piece) == 0:
        return
    wet = convolve(piece, ir)[lo - first : hi - first]
    if wet.ndim == 1:
        wet = wet[:, None]
    if weight is not None:
        wet = wet * weight[: len(wet), None]
    out[lo - origin : lo - origin + len(wet)] += wet


class Convolver:
    """Real-time convolution through ``ir``, one block of ``block`` frames at a time.

    ``ir`` is (taps,), (taps, outputs) for one input, or (taps, inputs, outputs), where
    each output sums its inputs, each through its own response. ``block`` is a power of
    two, and the only latency. The response is split into segments whose partitions
    grow along it (see :attr:`segments`), and the work of the long partitions is spread
    over the blocks in between, so the cost per block stays even and grows slowly with
    the response's length: a 3 s, 16-channel response in blocks of 128 frames takes
    about 0.3 ms of the 2.7 ms each block lasts at 48 kHz, on one current x86-64 core.
    """

    def __init__(self, ir: np.ndarray, block: int = 256) -> None:
        from acoustic_spaces import _native

        h = np.asarray(ir, dtype=np.float32)
        if h.ndim == 1:
            h = h[:, None, None]
        elif h.ndim == 2:
            h = h[:, None, :]
        if h.ndim != 3 or 0 in h.shape:
            raise ValueError("ir must be (taps,), (taps, outputs) or (taps, inputs, outputs)")
        if block < 2 or block & (block - 1):
            raise ValueError("block must be a power of two")
        self._native = _native.Convolver(np.ascontiguousarray(h.transpose(1, 2, 0)), int(block))

    @property
    def inputs(self) -> int:
        return self._native.inputs

    @property
    def outputs(self) -> int:
        return self._native.outputs

    @property
    def taps(self) -> int:
        return self._native.taps

    @property
    def block(self) -> int:
        return self._native.block

    @property
    def segments(self) -> list[tuple[int, int, int]]:
        """(first tap, partition length, partitions) of each segment of the response."""
        return list(self._native.segments)

    def process(self, block: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """One block, (block,) or (block, inputs), to (block, outputs) float32.

        Given ``out``, a writable C-contiguous float32 (block, outputs) array, the block
        is written into it and nothing new is allocated for it, as a sound card's
        callback needs (a float32 C-contiguous ``block`` is not copied either).
        """
        x = np.asarray(block, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        x = np.ascontiguousarray(x)
        if out is None:
            return self._native.process(x)
        if out.dtype != np.float32 or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError("out must be a writable C-contiguous float32 array")
        self._native.process_into(x, out)
        return out

    def run(self, signal: np.ndarray) -> np.ndarray:
        """A whole signal, (samples,) or (samples, inputs), through the convolver:
        (samples + taps - 1, outputs), as :func:`convolve` gives it."""
        x = np.asarray(signal, dtype=np.float32)
        x = x[:, None] if x.ndim == 1 else x
        if x.ndim != 2 or x.shape[1] != self.inputs:
            raise ValueError(f"signal must have {self.inputs} channel(s)")
        length = len(x) + self.taps - 1
        blocks = -(-length // self.block)
        padded = np.zeros((blocks * self.block, self.inputs), np.float32)
        padded[: len(x)] = x
        out = np.concatenate(
            [self.process(padded[k * self.block : (k + 1) * self.block]) for k in range(blocks)]
        )
        return out[:length]

    def reset(self) -> None:
        """Forget all past input."""
        self._native.reset()


@dataclass
class Playback:
    """How a :func:`play` run went."""

    frames: int
    #: Blocks the sound card reported an underflow or overflow for.
    xruns: int
    #: Longest time spent convolving one block, as a share of the block's duration.
    load: float


def play(
    ir: np.ndarray | Convolver | HeadTrackedBinaural,
    sample_rate: int,
    signal: np.ndarray | None = None,
    *,
    block: int = 256,
    gain: float = 1.0,
    device: object = None,
    seconds: float | None = None,
    stop: threading.Event | None = None,
) -> Playback:
    """Play ``signal`` through ``ir`` (samples, outputs) on the sound card, convolving in
    real time; without a signal, the sound card's first input goes through it live.

    ``ir`` may also be a processor that works block by block, as a :class:`Convolver`
    or :class:`~acoustic_spaces.headtracking.HeadTrackedBinaural` does; its own block
    size then replaces ``block``. Playback ends with the signal's reverberation tail;
    live input runs for ``seconds``, until ``stop`` is set, or until Ctrl-C. ``signal``
    must be at ``sample_rate``.
    """
    from acoustic_spaces.measurement import _sounddevice

    sd = _sounddevice()
    convolver = ir if hasattr(ir, "process") else Convolver(ir, block)
    block = convolver.block
    if convolver.inputs != 1:
        raise ValueError("play takes a response with one input")
    dry = None if signal is None else _as_2d(signal, "signal").mean(axis=1).astype(np.float32)
    total = None if dry is None else len(dry) + convolver.taps - 1
    if dry is None and seconds is not None:
        total = int(round(seconds * sample_rate))
    budget = block / sample_rate
    state = {"frames": 0, "xruns": 0, "load": 0.0}
    done = threading.Event()
    scratch = np.zeros(block, np.float32)
    # A processor that takes ``out`` writes each block into this one buffer, so the
    # sound card's callback allocates nothing for it.
    wet = np.zeros((block, convolver.outputs), np.float32)
    into = "out" in inspect.signature(convolver.process).parameters

    def render(frames_in: np.ndarray, outdata: np.ndarray, status) -> None:
        began = time.perf_counter()
        if status:
            state["xruns"] += 1
        processed = convolver.process(frames_in, out=wet) if into else convolver.process(frames_in)
        np.multiply(processed, gain, out=outdata)
        state["load"] = max(state["load"], (time.perf_counter() - began) / budget)
        state["frames"] += block
        if (total is not None and state["frames"] >= total) or (stop and stop.is_set()):
            raise sd.CallbackStop

    def output_callback(outdata, frames, _time, status) -> None:
        start = state["frames"]
        scratch[:] = 0.0
        if start < len(dry):
            piece = dry[start : start + frames]
            scratch[: len(piece)] = piece
        render(scratch, outdata, status)

    def duplex_callback(indata, outdata, frames, _time, status) -> None:
        render(indata[:, 0], outdata, status)

    common = {
        "samplerate": sample_rate,
        "blocksize": block,
        "dtype": "float32",
        "device": device,
        "finished_callback": done.set,
    }
    if dry is not None:
        stream = sd.OutputStream(channels=convolver.outputs, callback=output_callback, **common)
    else:
        stream = sd.Stream(channels=(1, convolver.outputs), callback=duplex_callback, **common)
    with stream:
        try:
            while not done.wait(0.1):  # a timeout keeps Ctrl-C working on Windows
                pass
        except KeyboardInterrupt:
            pass
    return Playback(state["frames"], state["xruns"], state["load"])


__all__ = ["Convolver", "Playback", "auralize", "convolve", "play", "power_gains", "resample"]
