"""Audio the demos measure and cut: WAV files, levels as meters gate and weight them,
lining signals up, and the moments between phrases.

Levels are gated in the manner of a loudness meter, though not to the letter of ITU-R
BS.1770 (:func:`gated_rms` says how): over the 400 ms blocks that are playing, so that a
pause or a fade does not pull a level down. The meters (:func:`level_db`,
:func:`loudness_db`) add up the channels' powers as BS.1770 does, each weighted by where
it plays from (:func:`channel_weights`). ``a_weighted`` gives the dB(A) the captions
state, ``k_weighted`` the loudness the tours match their stops by: the K weighting keeps
the bass that the A weighting all but leaves out, which far places and the back of a
stage keep little else of.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def read(path: str | Path) -> tuple[np.ndarray, int]:
    """A WAV file as (samples, channels) and its rate."""
    from acoustic_spaces.audio import read_wav

    return read_wav(path)


def write(path: str | Path, data: np.ndarray, rate: int) -> None:
    """``data`` as a 32-bit float WAV, its folder made."""
    from acoustic_spaces.audio import write_wav

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, data, rate)


def mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def stereo(x: np.ndarray) -> np.ndarray:
    """``x`` (samples, channels) as two channels: a mono signal doubled, more cut."""
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def db(level: float) -> float:
    return 20.0 * math.log10(level + 1e-12)


def gated_rms(x: np.ndarray, rate: int) -> float:
    """RMS of ``x`` mixed down to mono (:func:`mono`, the mean of its channels, as a feed
    made of it plays) over the 400 ms blocks that are playing: louder than 70 dB below
    the loudest block, then than 20 dB below the mean of those. BS.1770 gates alike but
    differently in the details: its blocks overlap by 75 %, its first gate is at -70
    LUFS, absolute, and its second 10 LU below the mean. No weighting is applied here;
    the callers weight first. The meters (:func:`level_db`, :func:`loudness_db`) do not
    mix down: they add up the channels' powers."""
    return _gated(mono(x)[:, None], rate, np.ones(1))


def channels_rms(x: np.ndarray, rate: int) -> float:
    """The RMS of the channels of ``x`` (samples, channels) together: their mean square,
    over the 400 ms blocks in which they play, gated on the channels' summed power as
    BS.1770 gates several channels (and as :func:`gated_rms` gates one), so that a
    channel that pauses while the others play counts its pause. The level that the
    left and right channels of one recording share one gain from: the mean of the
    channels' powers, so that two channels alike give the level of one (the meters,
    :func:`level_db` and :func:`loudness_db`, take BS.1770's sum instead)."""
    x = np.atleast_2d(np.asarray(x, dtype=float).T).T
    return _gated(x, rate, np.full(x.shape[1], 1.0 / x.shape[1]))


def _gated(x: np.ndarray, rate: int, weights: np.ndarray) -> float:
    """The root of the mean squares of the channels of ``x`` (samples, channels), each
    times its weight (``weights``, one per channel) and added up, over the blocks
    :func:`gated_rms` keeps, the gates on that weighted sum in each block."""
    block = int(0.4 * rate)
    n = len(x) // block
    if n == 0:
        return float(np.sqrt(np.mean(x**2, axis=0) @ weights))
    power = (x[: n * block].reshape(n, block, -1) ** 2).mean(axis=1) @ weights
    on = power[power > 1e-7 * power.max()]
    if on.size == 0:
        return 0.0
    on = on[on > on.mean() * 0.01]
    return float(np.sqrt(on.mean()))


def a_gain(freq) -> np.ndarray:
    """The A weighting's gain in amplitude (IEC 61672) at ``freq`` (Hz), 1 at 1 kHz."""
    f2 = np.asarray(freq, dtype=float) ** 2
    gain = (12194.0**2 * f2**2) / (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    )
    return gain * 10.0 ** (2.0 / 20.0)  # 0 dB at 1 kHz


def a_weighted(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` (samples, channels) through the A weighting (IEC 61672), by FFT."""
    n = 1 << math.ceil(math.log2(len(x)))  # a length the FFT is fast at
    gain = a_gain(np.fft.rfftfreq(n, 1.0 / rate))
    rows = np.ascontiguousarray(np.atleast_2d(x.T))  # a channel per row
    return np.fft.irfft(np.fft.rfft(rows, n) * gain, n)[:, : len(x)].T


def k_weighted(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` (samples, channels) through the K weighting of ITU-R BS.1770, by FFT: its
    two filters as pyloudnorm designs them for any sample rate, a +4 dB high shelf at
    1.5 kHz (Q 0.71) and a second-order high-pass at 38 Hz (Q 0.5), close to the
    standard's coefficients at 48 kHz."""
    n = 1 << math.ceil(math.log2(len(x)))
    z = np.exp(-1j * np.pi * np.fft.rfftfreq(n, 0.5))  # z^-1 at each bin
    w, a = 2.0 * np.pi * 1500.0 / rate, 10.0 ** (4.0 / 40.0)
    alpha, cos = math.sin(w) / math.sqrt(2.0), math.cos(w)
    shelf = (
        a * ((a + 1) + (a - 1) * cos + 2 * math.sqrt(a) * alpha)
        - 2 * a * ((a - 1) + (a + 1) * cos) * z
        + a * ((a + 1) + (a - 1) * cos - 2 * math.sqrt(a) * alpha) * z**2
    ) / (
        (a + 1)
        - (a - 1) * cos
        + 2 * math.sqrt(a) * alpha
        + 2 * ((a - 1) - (a + 1) * cos) * z
        + ((a + 1) - (a - 1) * cos - 2 * math.sqrt(a) * alpha) * z**2
    )
    w = 2.0 * np.pi * 38.0 / rate
    alpha, cos = math.sin(w), math.cos(w)  # Q = 0.5
    high_pass = (1 + cos) / 2 * (1 - z) ** 2 / ((1 + alpha) - 2 * cos * z + (1 - alpha) * z**2)
    rows = np.ascontiguousarray(np.atleast_2d(x.T))
    return np.fft.irfft(np.fft.rfft(rows, n) * shelf * high_pass, n)[:, : len(x)].T


#: ITU-R BS.1770-4's weight G_i of a loudspeaker less than 30 degrees above or below
#: the horizon and 60 to 120 degrees to one side (Annex 3, Table 4): +1.5 dB.
SIDE_WEIGHT = 1.41


def channel_weights(fmt) -> np.ndarray | None:
    """The weight of each channel of the format ``fmt`` (:mod:`acoustic_spaces.spatial`)
    in the meters' sum of the channels' powers (:func:`level_db`), or None when every
    channel weighs 1: mono, binaural (the two ears, as BS.1770's left and right) and
    microphones, one capsule or a pair. Loudspeakers weigh as ITU-R BS.1770-4 weighs
    them by where they stand (Annex 3, Table 4): :data:`SIDE_WEIGHT` less than 30
    degrees above or below the horizon and 60 to 120 degrees to a side (the surrounds
    of 5.0 at 110 degrees, the side surrounds of 7.0 at 90), 1 elsewhere, and 0 for LFE
    channels, which BS.1770 leaves out (Annex 1). Ambisonics (AmbiX) is measured on W
    alone, the sound pressure: its other channels feed no loudspeaker but carry the
    field's gradients, which add to W or take from it by where the sound comes from."""
    from acoustic_spaces.spatial import Ambisonics, Loudspeakers

    if isinstance(fmt, Ambisonics):
        return np.eye(fmt.channels)[0]
    if isinstance(fmt, Loudspeakers):

        def weight(speaker) -> float:
            if speaker.lfe:
                return 0.0
            azimuth = abs((speaker.azimuth + 180.0) % 360.0 - 180.0)
            side = abs(speaker.elevation) < 30.0 and 60.0 <= azimuth <= 120.0
            return SIDE_WEIGHT if side else 1.0

        return np.array([weight(s) for s in fmt.speakers])
    return None


def _summed_rms(x: np.ndarray, rate: int, weights: np.ndarray | None) -> float:
    """The root of the gated sum of the channels' mean squares of ``x`` (samples,
    channels), each times its weight (``weights``; None weighs each 1)."""
    x = np.atleast_2d(np.asarray(x, dtype=float).T).T
    g = np.ones(x.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if g.shape != (x.shape[1],):
        raise ValueError(
            f"the meters got {g.size} channel weights for {x.shape[1]} channels: give one "
            "weight per channel (channel_weights of the signal's format)"
        )
    return _gated(x, rate, g)


def level_db(x: np.ndarray, rate: int, weights: np.ndarray | None = None) -> float:
    """The gated A-weighted level of ``x`` (samples, channels), dB re full scale, its
    channels added up as ITU-R BS.1770-4 eq. 2 adds them: the sum of their mean squares,
    each times its weight G_i (``weights``, :func:`channel_weights` of its format; None
    weighs each 1), gated on that sum as :func:`gated_rms` gates. So two channels alike
    read 3.01 dB above one alone, as two unlike channels of the same power do, and a
    delay between them changes nothing."""
    return db(_summed_rms(a_weighted(x, rate), rate, weights))


def loudness_db(x: np.ndarray, rate: int, weights: np.ndarray | None = None) -> float:
    """The gated K-weighted level of ``x`` (samples, channels), dB re full scale, its
    channels' powers added up as :func:`level_db` adds them: BS.1770's loudness without
    its -0.691 dB and gated as :func:`gated_rms` gates, so a loudness in the manner of
    LUFS, not a BS.1770 measurement."""
    return db(_summed_rms(k_weighted(x, rate), rate, weights))


def lag_of(x: np.ndarray, reference: np.ndarray, rate: int, most_s: float) -> int:
    """Samples by which ``x`` lags ``reference`` (both mono), within ``most_s``."""
    n = 1 << math.ceil(math.log2(len(x) + len(reference)))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(reference, n)), n)
    most = int(most_s * rate)
    lags = np.r_[np.arange(0, most + 1), np.arange(-most, 0)]
    return int(lags[np.argmax(np.abs(c[lags]))])


def shifted(x: np.ndarray, samples: int, length: int) -> np.ndarray:
    """``x`` started ``samples`` earlier (later if negative), cut or padded to ``length``."""
    out = np.zeros((length,) + x.shape[1:])
    if samples >= 0:
        part = x[samples : samples + length]
        out[: len(part)] = part
    else:
        part = x[: max(0, length + samples)]
        out[-samples : -samples + len(part)] = part
    return out


def envelope(x: np.ndarray, rate: int, step_s: float = 0.1) -> np.ndarray:
    """RMS of ``x`` over 0.5 s windows every ``step_s``; one value, the RMS of all of it,
    for a signal shorter than one window (0 when it is empty)."""
    m = mono(x).astype(float)
    step, window = int(step_s * rate), int(0.5 * rate)
    if len(m) < window:  # one window, cut to the signal: silence when there is none
        return np.array([float(np.sqrt(np.mean(m**2))) if len(m) else 0.0])
    power = np.cumsum(np.r_[0.0, m**2])
    starts = np.arange(0, max(1, len(m) - window), step)
    return np.sqrt((power[starts + window] - power[starts]) / window)


def quiet_moment(env: np.ndarray, step_s: float, near_s: float, within_s: float) -> float:
    """The time (s) of the quietest moment within ``within_s`` of ``near_s``: the
    middle of the quietest 0.5 s window of :func:`envelope`."""
    lo = max(0, int((near_s - within_s) / step_s))
    hi = min(len(env) - 1, int((near_s + within_s) / step_s))
    if hi <= lo:
        return near_s
    return (lo + int(np.argmin(env[lo : hi + 1]))) * step_s + 0.25


def resampled(x: np.ndarray, rate: int, rate_out: int) -> np.ndarray:
    """``x`` at ``rate_out`` (unchanged when already there)."""
    if rate == rate_out:
        return x
    from acoustic_spaces.render import resample

    return resample(x, rate, rate_out)


def decoded(path: Path) -> tuple[np.ndarray, int]:
    """A WAV file read, or any other decoded by ffmpeg (32-bit float, its own channels)."""
    if path.suffix.lower() == ".wav":
        return read(path)
    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"error: ffmpeg is needed to decode {path.name}")
    # Into a folder of its own: the file's may be read-only (an installed example's).
    with tempfile.TemporaryDirectory() as folder:
        wav = Path(folder) / "decoded.wav"
        command = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(wav)]
        subprocess.run(command, check=True)
        return read(wav)


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


__all__ = [
    "a_gain",
    "a_weighted",
    "channel_weights",
    "db",
    "decoded",
    "envelope",
    "gated_rms",
    "k_weighted",
    "lag_of",
    "level_db",
    "loudness_db",
    "mmss",
    "mono",
    "quiet_moment",
    "read",
    "resampled",
    "shifted",
    "stereo",
    "write",
]
