"""Head-related transfer functions: SOFA files and a spherical head model.

An :class:`HRTF` holds head-related impulse responses (left and right ear) for a set of
directions in head coordinates (x forward, y left, z up). Responses are looked up by
nearest direction.

``read_sofa`` reads SimpleFreeFieldHRIR files (the AES69 convention most HRTF databases
use); it needs h5py (the ``sofa`` extra). ``spherical_head`` needs no
data: Brown and Duda's (1998) rigid sphere with ears slightly behind the middle, which
gives the interaural time and level differences of a head but no pinna cues, so front
and back, up and down are hard to tell apart with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acoustic_spaces._extras import install_hint
from acoustic_spaces.spatial.directions import fibonacci_sphere, from_angles


@dataclass
class HRTF:
    directions: np.ndarray  # (m, 3) unit vectors in head coordinates
    hrir: np.ndarray  # (m, 2, taps): left, right
    sample_rate: float
    name: str = "hrtf"

    def __post_init__(self) -> None:
        directions = np.atleast_2d(np.asarray(self.directions, dtype=float))
        self.directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        self.hrir = np.asarray(self.hrir, dtype=float)
        if self.hrir.ndim != 3 or self.hrir.shape[1] != 2:
            raise ValueError("hrir must be (directions, 2, taps)")
        if len(self.hrir) != len(self.directions):
            raise ValueError("one pair of responses per direction")

    @property
    def taps(self) -> int:
        return self.hrir.shape[2]

    def nearest(self, directions: np.ndarray) -> np.ndarray:
        """Index of the nearest measured direction for each of (dirs, 3)."""
        v = np.atleast_2d(np.asarray(directions, dtype=float))
        return np.argmax(v @ self.directions.T, axis=1)

    def resampled(self, sample_rate: float) -> HRTF:
        """The same responses at another sample rate (band-limited, by FFT).

        The responses are filters: their frequency response, and so their gain, stays
        the same at every rate.
        """
        if math.isclose(sample_rate, self.sample_rate):
            return self
        n = self.taps
        m = max(1, int(round(n * sample_rate / self.sample_rate)))
        spectrum = np.fft.rfft(self.hrir, n, axis=-1)
        bins = m // 2 + 1
        if bins <= spectrum.shape[-1]:
            spectrum = spectrum[..., :bins]
        else:
            pad = np.zeros((*spectrum.shape[:-1], bins - spectrum.shape[-1]), complex)
            spectrum = np.concatenate([spectrum, pad], axis=-1)
        hrir = np.fft.irfft(spectrum, m, axis=-1)
        return HRTF(self.directions, hrir, float(sample_rate), self.name)


def spherical_head(
    sample_rate: float = 48_000,
    *,
    radius: float = 0.0875,
    ear_azimuth: float = 100.0,
    speed_of_sound: float = 343.0,
    directions: int = 2048,
) -> HRTF:
    """Brown and Duda's spherical head model on a grid of ``directions`` directions.

    Each ear sees a one-pole, one-zero head-shadow filter that boosts high frequencies
    up to 6 dB towards the source and cuts them up to 20 dB on the far side, and the
    path length around the sphere (Woodworth's formula). Responses start 1 ms early,
    so the earliest ear still has room for the filter's ringing.
    """
    grid = fibonacci_sphere(directions)
    taps = 1 << int(math.ceil(math.log2(0.004 * sample_rate)))
    f = np.fft.rfftfreq(taps, 1.0 / sample_rate)
    omega = 2.0 * math.pi * f
    omega0 = speed_of_sound / radius
    alpha_min, theta_min = 0.1, math.radians(150.0)
    lead = 0.001
    hrir = np.empty((directions, 2, taps))
    for ear, sign in enumerate((1.0, -1.0)):
        axis = from_angles(sign * ear_azimuth, 0.0)
        theta = np.arccos(np.clip(grid @ axis, -1.0, 1.0))  # angle from the ear
        alpha = (1 + alpha_min / 2) + (1 - alpha_min / 2) * np.cos(theta / theta_min * math.pi)
        delay = np.where(
            theta < math.pi / 2,
            -radius / speed_of_sound * np.cos(theta),
            radius / speed_of_sound * (theta - math.pi / 2),
        )
        shadow = (1 + 1j * alpha[:, None] * omega / (2 * omega0)) / (1 + 1j * omega / (2 * omega0))
        phase = np.exp(-1j * omega * (lead + delay[:, None]))
        hrir[:, ear] = np.fft.irfft(shadow * phase, taps, axis=-1)
    return HRTF(grid, hrir, float(sample_rate), "spherical head")


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return _text(value.item()) if value.size == 1 else ""
    return str(value)


def _positions(dataset) -> np.ndarray:
    """Positions (n, 3) of a SOFA variable as Cartesian coordinates.

    A variable that changes with the measurements as well ([R C M], say) gives those of
    the first measurement.
    """
    values = np.asarray(dataset[()], dtype=float)
    if values.ndim == 3:
        values = values[..., 0]
    values = values.reshape(-1, 3)
    kind = _text(dataset.attrs.get("Type", "cartesian")).lower()
    if kind == "spherical":
        units = _text(dataset.attrs.get("Units", "degree")).lower()
        az, el, r = values[:, 0], values[:, 1], values[:, 2]
        if "degree" not in units:
            az, el = np.degrees(az), np.degrees(el)
        return from_angles(az, el) * r[:, None]
    return values


def read_sofa(path: str | Path) -> HRTF:
    """HRTF from a SOFA file with head-related impulse responses for two ears."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(f"reading SOFA files needs h5py: {install_hint('sofa')}") from exc
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "Data.IR" not in f or "SourcePosition" not in f:
            raise ValueError(f"{path}: not a SOFA file with impulse responses")
        ir = np.asarray(f["Data.IR"][()], dtype=float)  # (measurements, receivers, taps)
        if ir.ndim != 3 or ir.shape[1] != 2:
            raise ValueError(f"{path}: expected responses for 2 ears, got shape {ir.shape}")
        rate = float(np.asarray(f["Data.SamplingRate"][()]).ravel()[0])
        count = len(ir)

        def per_measurement(name: str, default: list[float]) -> np.ndarray:
            """(count, 3): one value for all measurements ([I C]) or one each ([M C])."""
            values = _positions(f[name]) if name in f else np.array([default], dtype=float)
            if len(values) == count:
                return values
            return np.repeat(values[:1], count, axis=0)

        relative = per_measurement("SourcePosition", [1.0, 0.0, 0.0]) - per_measurement(
            "ListenerPosition", [0.0, 0.0, 0.0]
        )
        # Head axes from the listener's view and up vectors (SOFA default: x and z), for
        # each measurement, as a file may turn the head instead of moving the source.
        view = per_measurement("ListenerView", [1.0, 0.0, 0.0])
        up = per_measurement("ListenerUp", [0.0, 0.0, 1.0])
        view = view / np.linalg.norm(view, axis=1, keepdims=True)
        up = up - np.sum(up * view, axis=1, keepdims=True) * view
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        left = np.cross(up, view)
        directions = np.stack(
            [np.sum(relative * axis, axis=1) for axis in (view, left, up)], axis=1
        )
        # Receiver 0 is the left ear unless the file places it on the right (y < 0).
        if "ReceiverPosition" in f:
            ears = _positions(f["ReceiverPosition"])
            if len(ears) >= 2 and ears[0, 1] < ears[1, 1]:
                ir = ir[:, ::-1]
        if "Data.Delay" in f:
            delay = np.rint(np.asarray(f["Data.Delay"][()], dtype=float)).astype(int)
            delay = np.broadcast_to(delay.reshape(-1, 2), (len(ir), 2))
            if delay.any():
                shifted = np.zeros((*ir.shape[:2], ir.shape[2] + int(delay.max())))
                for m in range(len(ir)):
                    for r in range(2):
                        shifted[m, r, delay[m, r] : delay[m, r] + ir.shape[2]] = ir[m, r]
                ir = shifted
        name = _text(f.attrs.get("Title", path.stem)) or path.stem
    return HRTF(directions, ir, rate, name)


__all__ = ["HRTF", "read_sofa", "spherical_head"]
