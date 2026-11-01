"""Head-related transfer functions: SOFA files and a spherical head model.

An :class:`HRTF` holds head-related impulse responses (left and right ear) for a set of
directions in head coordinates (x forward, y left, z up). Responses between the
measured directions are interpolated (:meth:`HRTF.interpolate`): the three nearest are
aligned at their onsets, mixed, and delayed by the mix of their onsets, so the
interaural time difference changes smoothly with the direction.

``read_sofa`` reads SimpleFreeFieldHRIR files (the AES69 convention most HRTF databases
use); it needs h5py (the ``sofa`` extra). ``spherical_head`` needs no
data: Brown and Duda's (1998) rigid sphere with ears slightly behind the middle, which
gives the interaural time and level differences of a head but no pinna cues, so front
and back, up and down are hard to tell apart with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, overload

import numpy as np

from acoustic_spaces._extras import install_hint
from acoustic_spaces.spatial.directions import fibonacci_sphere, from_angles

#: Finding the onsets: how many times finer than a sample the cross-correlation with the
#: minimum-phase responses is sampled, and the floor of their magnitudes (below the
#: peak) for the minimum phase.
_ONSET_STEPS = 8
_FLOOR_DB = -100.0
#: Directions interpolated in one go, and directions times triangles searched in one go
#: (memory).
_CHUNK = 256
_SEARCH = 1 << 20


@dataclass
class HRTF:
    directions: np.ndarray  # (m, 3) unit vectors in head coordinates
    hrir: np.ndarray  # (m, 2, taps): left, right
    sample_rate: float
    name: str = "hrtf"
    _aligned: tuple | None = field(default=None, init=False, repr=False, compare=False)
    _triangles: tuple | None = field(default=None, init=False, repr=False, compare=False)

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

    def onsets(self) -> np.ndarray:
        """(m, 2) onset of each response (samples, fractions included): the lag at which
        it best matches its minimum-phase version, which starts at once (see
        :func:`_onsets`)."""
        return self._alignment()[0]

    def _alignment(self) -> tuple[np.ndarray, np.ndarray]:
        """The onsets, and the spectra (m, 2, bins) of the responses moved to start
        there, on an FFT twice the responses long. The samples before an onset wrap
        round to the end of the FFT, so delaying the response again brings them back."""
        if self._aligned is None:
            onsets = _onsets(self.hrir)
            n_fft = 2 * self.taps
            spectra = np.fft.rfft(self.hrir, n_fft, axis=-1)
            spectra *= np.exp(2j * np.pi * np.fft.rfftfreq(n_fft) * onsets[..., None])
            self._aligned = (onsets, spectra)
        return self._aligned

    @overload
    def interpolate(
        self, directions: np.ndarray, *, onsets: Literal[False] = False
    ) -> np.ndarray: ...

    @overload
    def interpolate(
        self, directions: np.ndarray, *, onsets: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def interpolate(
        self, directions: np.ndarray, *, onsets: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Responses (dirs, 2, taps) for head directions (dirs, 3), between the measured
        ones; with ``onsets``, also the onset (dirs, 2) each starts at (samples,
        fractions included).

        The measured directions are triangulated (the faces of their convex hull, as
        for VBAP). Each direction mixes the responses at the corners of the triangle it
        falls in, aligned at their onsets, with its barycentric weights, and the mix is
        delayed by the same mix of the onsets, to a fraction of a sample: that mix is
        the onset given. A measured direction gives its own response, and the weights,
        and with them the responses and the interaural time difference, change smoothly
        with the direction. Where no triangle covers a direction (a set that leaves part
        of the sphere out), the nearest response is used, and its onset.
        """
        v = np.atleast_2d(np.asarray(directions, dtype=float))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        if len(self.directions) < 4:
            nearest = self.nearest(v)
            if onsets:
                return self.hrir[nearest], self.onsets()[nearest]
            return self.hrir[nearest]
        measured, spectra = self._alignment()
        n_fft = spectra.shape[-1] * 2 - 2
        frequency = np.fft.rfftfreq(n_fft)  # cycles per sample
        out = np.empty((len(v), 2, self.taps))
        delays = np.empty((len(v), 2))
        for start in range(0, len(v), _CHUNK):
            index, weights = self._corners(v[start : start + _CHUNK])
            spectrum = np.einsum("kj,kjeb->keb", weights, spectra[index])
            delay = np.einsum("kj,kje->ke", weights, measured[index])
            spectrum *= np.exp(-2j * np.pi * frequency * delay[..., None])
            out[start : start + _CHUNK] = np.fft.irfft(spectrum, n_fft, axis=-1)[..., : self.taps]
            delays[start : start + _CHUNK] = delay
        return (out, delays) if onsets else out

    def onsets_at(self, directions: np.ndarray) -> np.ndarray:
        """The onsets (dirs, 2) that :meth:`interpolate` gives for head directions
        (dirs, 3) (samples, fractions included), without mixing the responses."""
        v = np.atleast_2d(np.asarray(directions, dtype=float))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        if len(self.directions) < 4:
            return self.onsets()[self.nearest(v)]
        measured, _ = self._alignment()
        delays = np.empty((len(v), 2))
        for start in range(0, len(v), _CHUNK):
            index, weights = self._corners(v[start : start + _CHUNK])
            delays[start : start + _CHUNK] = np.einsum("kj,kje->ke", weights, measured[index])
        return delays

    def _corners(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The measured directions (k, 3) to mix for unit directions (k, 3), and their
        weights (k, 3)."""
        faces, inverse, around, real = self._triangulation()
        nearest = np.argmax(v @ self.directions.T, axis=1)
        # The triangles round the nearest direction first, then all of them.
        candidates = around[nearest]
        g = np.einsum("kc,ktcd->ktd", v, inverse[candidates])
        g[candidates < 0] = -np.inf
        best = np.argmax(g.min(axis=2), axis=1)
        face = candidates[np.arange(len(v)), best]
        weights = g[np.arange(len(v)), best]
        missed = np.flatnonzero(weights.min(axis=1) < -1e-9)
        step = max(1, _SEARCH // len(faces))
        for start in range(0, len(missed), step):
            some = missed[start : start + step]
            g = np.einsum("kc,tcd->ktd", v[some], inverse)
            best = np.argmax(g.min(axis=2), axis=1)
            face[some] = best
            weights[some] = g[np.arange(len(some)), best]
        index = faces[face]
        made_up = index >= real
        weights[made_up] = 0.0
        index[made_up] = 0
        total = weights.sum(axis=1, keepdims=True)
        uncovered = (weights.min(axis=1) < -1e-9) | (total[:, 0] <= 1e-12)
        weights = np.maximum(weights, 0.0) / np.where(uncovered[:, None], 1.0, total)
        index[uncovered] = nearest[uncovered, None]
        weights[uncovered] = [1.0, 0.0, 0.0]
        return index, weights

    def _triangulation(self):
        """Triangles (t, 3) of the measured directions, the inverse of each one's corner
        matrix (t, 3, 3), the triangles round each direction (m, n) padded with -1, and
        how many of the corners are measured directions (the rest are made up)."""
        if self._triangles is None:
            from scipy.spatial import ConvexHull, QhullError

            points = self.directions
            try:
                faces = ConvexHull(points).simplices
            except QhullError:
                # All in one plane (a ring of directions): two made-up directions on
                # either side of it, which give their weight to the others.
                normal = np.linalg.svd(points - points.mean(axis=0))[2][-1]
                points = np.vstack([points, normal, -normal])
                faces = ConvexHull(points).simplices
            # Triangles through the centre (a set covering half the sphere, cut along
            # its rim) cover no direction.
            faces = faces[np.abs(np.linalg.det(points[faces])) > 1e-12]
            inverse = np.linalg.inv(points[faces])
            real = len(self.directions)
            count = np.bincount(faces.ravel(), minlength=len(points))[:real]
            around = np.full((real, max(int(count.max()), 1)), -1)
            filled = np.zeros(real, dtype=int)
            for t, face in enumerate(faces):
                for corner in face[face < real]:
                    around[corner, filled[corner]] = t
                    filled[corner] += 1
            self._triangles = (faces, inverse, around, real)
        return self._triangles

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


def _onsets(hrir: np.ndarray) -> np.ndarray:
    """(m, 2) where each of the responses (m, 2, taps) starts (samples, fractions
    included): the lag of the peak of its cross-correlation with its minimum-phase
    version (from the folded real cepstrum), which has the same magnitude and starts at
    once. Unlike the first sample near the peak, this holds for a pulse delayed by a
    fraction of a sample, whose ringing before it rises with the fraction: responses
    aligned at a threshold of it come apart by a sample or two, and mixing them cancels
    the high frequencies (up to 13 dB at 8 kHz for the spherical head)."""
    taps = hrir.shape[-1]
    n = 1 << int(math.ceil(math.log2(4 * taps)))
    spectra = np.fft.rfft(hrir, n, axis=-1)
    magnitude = np.abs(spectra)
    floor = magnitude.max(axis=-1, keepdims=True) * 10.0 ** (_FLOOR_DB / 20.0)
    cepstrum = np.fft.irfft(np.log(np.maximum(magnitude, np.maximum(floor, 1e-300))), n)
    fold = np.zeros(n)
    fold[0], fold[1 : n // 2], fold[n // 2] = 1.0, 2.0, 1.0
    minimum = np.exp(np.fft.rfft(cepstrum * fold, n))
    fine = n * _ONSET_STEPS
    onsets = np.empty(hrir.shape[:-1])
    flat = onsets.reshape(-1)
    cross = (spectra * np.conj(minimum)).reshape(-1, spectra.shape[-1])
    for start in range(0, len(cross), _CHUNK):
        lagged = np.fft.irfft(cross[start : start + _CHUNK], fine, axis=-1)
        # Lags from -taps to the response's length, then the peak to a fraction of a step.
        lagged = np.roll(lagged, taps * _ONSET_STEPS, axis=-1)[:, : 2 * taps * _ONSET_STEPS]
        peak = np.clip(np.argmax(lagged, axis=-1), 1, lagged.shape[-1] - 2)
        rows = np.arange(len(lagged))
        a, b, c = lagged[rows, peak - 1], lagged[rows, peak], lagged[rows, peak + 1]
        bend = a - 2.0 * b + c
        shift = np.where(bend < 0.0, 0.5 * (a - c) / np.where(bend < 0.0, bend, -1.0), 0.0)
        flat[start : start + _CHUNK] = (peak + shift) / _ONSET_STEPS - taps
    return onsets


def _delayed(ir: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """Responses (m, 2, taps) delayed by (m, 2) samples, fractions included, and made
    longer by the largest delay."""
    taps = ir.shape[2] + int(math.ceil(delay.max()))
    n_fft = 2 * taps
    whole = np.floor(delay).astype(int)
    fraction = delay - whole
    spectrum = np.fft.rfft(ir, n_fft, axis=-1)
    frequency = np.fft.rfftfreq(n_fft)
    shifted = np.fft.irfft(spectrum * np.exp(-2j * np.pi * frequency * fraction[..., None]), n_fft)
    out = np.zeros((*ir.shape[:2], taps))
    for (m, r), start in np.ndenumerate(whole):
        out[m, r, start:] = shifted[m, r, : taps - start]
    return out


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
            delay = np.asarray(f["Data.Delay"][()], dtype=float)
            delay = np.broadcast_to(delay.reshape(-1, 2), (len(ir), 2))
            if delay.any():
                ir = _delayed(ir, delay)
        name = _text(f.attrs.get("Title", path.stem)) or path.stem
    return HRTF(directions, ir, rate, name)


__all__ = ["HRTF", "read_sofa", "spherical_head"]
