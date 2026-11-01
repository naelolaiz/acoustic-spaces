"""Audio the demos measure and cut: WAV files, levels as meters gate and weight them,
lining signals up, and the moments between phrases.

Levels are gated in the manner of a loudness meter, though not to the letter of ITU-R
BS.1770 (:func:`gated_rms` says how): over the 400 ms blocks that are playing, so that a
pause or a fade does not pull a level down. ``a_weighted`` gives the dB(A) the captions
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
    """RMS over the 400 ms blocks that are playing: louder than 70 dB below the loudest
    block, then than 20 dB below the mean of those. BS.1770 gates alike but differently
    in the details: its blocks overlap by 75 %, its first gate is at -70 LUFS, absolute,
    and its second 10 LU below the mean. No weighting is applied here; the callers
    weight first."""
    return _gated(mono(x)[:, None], rate)


def channels_rms(x: np.ndarray, rate: int) -> float:
    """The RMS of the channels of ``x`` (samples, channels) together: their mean square,
    over the 400 ms blocks in which they play, gated on the channels' summed power as
    BS.1770 gates several channels (and as :func:`gated_rms` gates one), so that a
    channel that pauses while the others play counts its pause. The level that the
    left and right channels of one recording share one gain from."""
    return _gated(np.atleast_2d(np.asarray(x, dtype=float).T).T, rate)


def _gated(x: np.ndarray, rate: int) -> float:
    """The RMS (samples, channels) over the blocks :func:`gated_rms` keeps, the gates on
    the mean over the channels of each block's mean square."""
    block = int(0.4 * rate)
    n = len(x) // block
    if n == 0:
        return float(np.sqrt(np.mean(x**2)))
    power = (x[: n * block].reshape(n, block, -1) ** 2).mean(axis=(1, 2))
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


def level_db(x: np.ndarray, rate: int) -> float:
    """The gated A-weighted level of ``x`` (:func:`gated_rms`), dB re full scale."""
    return db(gated_rms(a_weighted(x, rate), rate))


def loudness_db(x: np.ndarray, rate: int) -> float:
    """The gated K-weighted level of ``x`` (:func:`gated_rms`), dB re full scale: a
    loudness in the manner of LUFS, not a BS.1770 measurement."""
    return db(gated_rms(k_weighted(x, rate), rate))


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
    """RMS of ``x`` over 0.5 s windows every ``step_s``."""
    m = mono(x)
    step, window = int(step_s * rate), int(0.5 * rate)
    power = np.cumsum(np.r_[0.0, m.astype(float) ** 2])
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
