"""The sound a source sends each way, for a walk with Doppler.

A source's directivity (:mod:`acoustic_spaces.directivity`) weighs each path's energy
per octave band, and every path carries the same signal. The sound of a rotor changes
with the direction in more than its band levels: each harmonic of the blades has a
pattern and a phase of its own, and the broadband noise has another. An
:class:`Emission` gives the signal itself in every direction, in the source's own frame
(x forward, y left, z up, as :class:`acoustic_spaces.scene.Orientation` turns it), and a
walk with Doppler (:func:`acoustic_spaces.doppler.render`) plays down each early path
the sound sent the way the path leaves the source, at the time it leaves.

What the early paths carry then comes from the emission alone: each path's share of
the source's directivity is taken out of its filter. The directivity still gives the
levels the simulation reports and the late part, which plays the emission's sound
averaged over the directions (:meth:`Emission.averaged`). For the two to agree, give
the source the emission's own directivity (:meth:`Emission.directivity`, written for a
scene by :func:`acoustic_spaces.directivity.write_directivity_csv`) and, for an
emission in pascals 1 m away, its sound power (:meth:`Emission.sound_power_db`).

- :class:`Mixture` is made of signals that do not correlate with one another
  (harmonics, as a cosine and a sine each, and noises in separate bands), each with an
  amplitude in every direction on the grid of :func:`acoustic_spaces.directivity.on_grid`.
- :class:`Periodic` is a sound that repeats, such as a rotor's tones: one period of it
  in every direction, which keeps a long walk's emission small.
- :class:`Sum` sends several emissions at once, such as a helicopter's two rotors.
- :class:`Faded` fades one in and out, so that it does not start or stop at once.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.directivity import (
    GRID_STEP_DEG,
    Balloon,
    _cell,
    _from_angles,
    _grid,
    _unit,
    on_grid,
    sphere_mean,
)
from acoustic_spaces.materials.bands import Resolution, band_centres

#: Samples of the signals' spectra, one stretch after another, for their band powers;
#: also how many samples a periodic sound is read in at a time.
_SEGMENT = 1 << 16


class Emission:
    """The signal a source sends each way (see the module's description)."""

    #: Samples per second.
    sample_rate: int

    @property
    def length(self) -> int:
        """How many samples it sends."""
        raise NotImplementedError

    def sent(self, start: int, stop: int, directions: np.ndarray) -> np.ndarray:
        """(stop - start,) the samples ``start`` to ``stop`` of the sound sent the ways
        ``directions`` ((stop - start, 3) unit vectors in the source's frame, one for
        each sample); nothing before 0 or from :attr:`length` on."""
        raise NotImplementedError

    def averaged(self) -> np.ndarray:
        """(length,) a signal with the emission's power spectrum averaged over the
        sphere: what the late part carries."""
        raise NotImplementedError

    def scaled(self, gain: float) -> Emission:
        """This emission ``gain`` times as loud in amplitude."""
        raise NotImplementedError

    def directivity(self, bands: Resolution = "octave", name: str = "") -> Balloon:
        """The emission's directivity: its energy in each band and direction, on the
        grid ``GRID_STEP_DEG`` apart that measured balloons use, over the sphere's
        mean."""
        return Balloon.from_table(self._grid_powers(bands), name)

    def sound_power_db(
        self, density: float, sound_speed: float, bands: Resolution = "octave"
    ) -> np.ndarray:
        """The sound power level (dB re 1 pW) in each band of an emission that sends
        the far field's pressure 1 m from the source (Pa): its mean square over the
        sphere times 4π m² over the air's impedance, ``density`` times
        ``sound_speed``."""
        watts = 4.0 * np.pi * sphere_mean(self._grid_powers(bands)) / (density * sound_speed)
        return 10.0 * np.log10(np.maximum(watts, 1e-30) / 1e-12)

    def _grid_powers(self, bands: Resolution) -> np.ndarray:
        """(elevations, azimuths, bands) the mean square sent each way of the grid
        ``GRID_STEP_DEG`` apart, in each band."""
        elevation, azimuth = _grid(GRID_STEP_DEG)
        ways = _from_angles(azimuth[None, :], elevation[:, None]).reshape(-1, 3)
        return self._powers(bands, ways).reshape(len(elevation), len(azimuth), -1)

    def _powers(self, bands: Resolution, directions: np.ndarray) -> np.ndarray:
        """(n, bands) the mean square of the sound sent the ways ``directions`` (n, 3),
        in each band."""
        raise NotImplementedError


def _band_powers(spectra: np.ndarray, gains: np.ndarray, n: int) -> np.ndarray:
    """(..., bands) the mean square in each band of ``n`` samples whose real FFT is
    ``spectra`` (..., n // 2 + 1), shared out by ``gains`` (n // 2 + 1, bands)."""
    power = np.abs(spectra) ** 2 / float(n) ** 2
    # Each bin but the first (and the last, for an even n) stands for two frequencies.
    power[..., 1 : (n + 1) // 2] *= 2.0
    return power @ gains


@dataclass(frozen=True)
class Mixture(Emission):
    """Signals that do not correlate with one another, each sent with an amplitude of
    its own in every direction."""

    #: (K, samples) the signals.
    signals: np.ndarray = dataclasses.field(repr=False)
    #: (elevations, azimuths, K) each signal's amplitude on the grid of
    #: :func:`~acoustic_spaces.directivity.on_grid` (elevations from -90 to 90 degrees,
    #: azimuths from 0 up to 360, each evenly spaced).
    weights: np.ndarray = dataclasses.field(repr=False)
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        signals = np.atleast_2d(np.asarray(self.signals, dtype=float))
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 3 or weights.shape[-1] != len(signals):
            raise ValueError(
                f"weights must be (elevations, azimuths, {len(signals)}), one per signal; "
                f"got {weights.shape}"
            )
        if weights.shape[0] < 2 or weights.shape[1] < 1:
            raise ValueError("weights need at least two elevations and one azimuth")
        if not (np.all(np.isfinite(signals)) and np.all(np.isfinite(weights))):
            raise ValueError("signals and weights must be finite")
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

    @property
    def length(self) -> int:
        return self.signals.shape[1]

    def sent(self, start: int, stop: int, directions: np.ndarray) -> np.ndarray:
        out = np.zeros(stop - start)
        lo, hi = max(start, 0), min(stop, self.length)
        if lo >= hi:
            return out
        amplitudes = on_grid(self.weights, np.asarray(directions)[lo - start : hi - start])
        out[lo - start : hi - start] = np.einsum("nk,kn->n", amplitudes, self.signals[:, lo:hi])
        return out

    def averaged(self) -> np.ndarray:
        # Signals that do not correlate add their powers: each at its mean power.
        return np.sqrt(sphere_mean(self.weights**2)) @ self.signals

    def scaled(self, gain: float) -> Mixture:
        return dataclasses.replace(self, weights=self.weights * float(gain))

    def _powers(self, bands: Resolution, directions: np.ndarray) -> np.ndarray:
        size = min(self.length, _SEGMENT)
        # The band gains add up to one: they share each frequency's power out.
        gains = synthesis.band_gains(size, self.sample_rate, len(band_centres(bands))).T
        starts = range(0, self.length - size + 1, size)
        powers = sum(
            _band_powers(np.fft.rfft(self.signals[:, s : s + size], axis=1), gains, size)
            for s in starts
        ) / len(starts)
        return on_grid(self.weights**2 @ powers, directions)


@dataclass(frozen=True)
class Periodic(Emission):
    """A sound that repeats, sent each way: one period of it in every direction of a
    grid like that of :func:`~acoustic_spaces.directivity.on_grid`, in a frame of its
    own (``axes``), read between its samples by a cubic through four of them and between
    the directions bilinearly.

    The sound of a rotor turns with its blades. In hover, heard from an azimuth ``phi``
    round the axis it is the sound heard from azimuth 0 a share ``phi / 2 pi`` of a turn
    before, and its phase turns that fast between the grid's directions. The table holds
    each direction's sound from that share of a turn (``turn_s``) later, which makes the
    directions differ only as slowly as the rotor's pattern, and :meth:`sent` puts the
    delay back. It also takes into account where the sound comes from (``centre``): from
    there, the sound sent in a direction ``d`` leaves ``d . centre / (c - v d_x)``
    later than it would from the source's place, for a source moving at ``v`` along
    its x through still air (``speed``, ``sound_speed``), and the table holds it as sent
    from the centre.
    """

    #: (elevations, azimuths, N) one period in each direction of the grid, in the frame
    #: ``axes``, N samples evenly spread over it; each direction's shifted as above.
    table: np.ndarray = dataclasses.field(repr=False)
    #: How long a period lasts (s).
    period_s: float = 1.0
    sample_rate: int = 48_000
    #: How many samples it sends.
    samples: int = 0
    #: (3, 3) the table's frame: its x, y and z as rows, in the source's frame.
    axes: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3), repr=False)
    #: How long one turn takes (s), a whole number of periods: positive for a turn from
    #: the table's x towards its y, negative the other way, 0 for a sound that does not
    #: turn.
    turn_s: float = 0.0
    #: (3,) where the sound comes from, in the source's frame (m).
    centre: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    #: How fast the source moves along its x through the air (m/s).
    speed: float = 0.0
    #: The speed of sound (m/s).
    sound_speed: float = 343.0

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=float)
        axes = np.asarray(self.axes, dtype=float)
        centre = np.asarray(self.centre, dtype=float)
        if table.ndim != 3 or table.shape[0] < 2 or table.shape[1] < 1 or table.shape[2] < 4:
            raise ValueError(
                "the table must be (elevations, azimuths, samples), with at least two "
                f"elevations, one azimuth and four samples a period; got {table.shape}"
            )
        if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-9):
            raise ValueError("axes must be a rotation: three unit rows at right angles")
        if centre.shape != (3,):
            raise ValueError(f"the centre must be a point (3,); got {centre.shape}")
        if not (np.all(np.isfinite(table)) and np.all(np.isfinite(centre))):
            raise ValueError("the table and the centre must be finite")
        if not (self.period_s > 0.0 and self.sound_speed > abs(self.speed) and self.samples >= 0):
            raise ValueError(
                "the period must be positive, the source slower than sound and the "
                "samples at least none"
            )
        turns = self.turn_s / self.period_s
        if abs(turns - round(turns)) > 1e-9 * max(1.0, abs(turns)):
            raise ValueError("a turn must last a whole number of periods")
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "centre", centre)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "samples", int(self.samples))

    @property
    def length(self) -> int:
        return self.samples

    def sent(self, start: int, stop: int, directions: np.ndarray) -> np.ndarray:
        out = np.zeros(stop - start)
        lo, hi = max(start, 0), min(stop, self.length)
        ways = _unit(np.asarray(directions)[lo - start : hi - start])
        for a in range(0, max(hi - lo, 0), _SEGMENT):
            v = ways[a : a + _SEGMENT]
            times = np.arange(lo + a, lo + a + len(v)) / self.sample_rate
            out[lo - start + a : lo - start + a + len(v)] = self._read(times, v)
        return out

    def _read(self, times: np.ndarray, ways: np.ndarray) -> np.ndarray:
        """The sound sent at ``times`` (n,) the ways ``ways`` (n, 3), unit vectors in
        the source's frame."""
        own = ways @ self.axes.T
        if self.turn_s:
            times = times - np.arctan2(own[:, 1], own[:, 0]) / (2 * np.pi) * self.turn_s
        if np.any(self.centre):
            times = times + (ways @ self.centre) / (self.sound_speed - self.speed * ways[:, 0])
        index, weights = self._taps(times)
        i0, j0, j1, ti, tj = _cell(self.table.shape, own)
        a, b, c, d = (
            np.einsum("nt,nt->n", self.table[i[:, None], j[:, None], index], weights)
            for i, j in ((i0, j0), (i0, j1), (i0 + 1, j0), (i0 + 1, j1))
        )
        return (1 - ti) * ((1 - tj) * a + tj * b) + ti * ((1 - tj) * c + tj * d)

    def _taps(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, 4) the table's samples round each time and (n, 4) their weights: a
        cubic through the four."""
        n = self.table.shape[-1]
        x = times / self.period_s % 1.0 * n
        whole = np.floor(x)
        f = x - whole
        weights = np.stack(
            [
                -f * (f - 1.0) * (f - 2.0) / 6.0,
                (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0,
                -(f + 1.0) * f * (f - 2.0) / 2.0,
                (f + 1.0) * f * (f - 1.0) / 6.0,
            ],
            axis=-1,
        )
        return (whole.astype(np.int64)[:, None] + np.arange(-1, 3)) % n, weights

    def averaged(self) -> np.ndarray:
        n = self.table.shape[-1]
        spectra = np.fft.rfft(self.table, axis=-1)
        # Each harmonic at its power over the sphere, in the phase of its mean.
        power = sphere_mean(np.abs(spectra) ** 2)
        period = np.fft.irfft(np.sqrt(power) * np.exp(1j * np.angle(sphere_mean(spectra))), n)
        out = np.empty(self.length)
        for a in range(0, self.length, _SEGMENT):
            times = np.arange(a, min(a + _SEGMENT, self.length)) / self.sample_rate
            index, weights = self._taps(times)
            out[a : a + len(times)] = np.sum(period[index] * weights, axis=1)
        return out

    def scaled(self, gain: float) -> Periodic:
        return dataclasses.replace(self, table=self.table * float(gain))

    def _powers(self, bands: Resolution, directions: np.ndarray) -> np.ndarray:
        n = self.table.shape[-1]
        # The FFT of one period holds its harmonics: bin k at k periods a second.
        gains = synthesis.band_gains(n, n / self.period_s, len(band_centres(bands))).T
        powers = _band_powers(np.fft.rfft(self.table, axis=-1), gains, n)
        return on_grid(powers, _unit(directions) @ self.axes.T)


@dataclass(frozen=True)
class Sum(Emission):
    """Emissions sent at once from the same source, such as a helicopter's rotors and
    their broadband noise: what they send adds up. Each part's sound is taken as not
    correlating with the others' when they are averaged and put in bands."""

    #: The emissions, all at one sample rate.
    parts: tuple[Emission, ...] = ()

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if not parts:
            raise ValueError("a sum needs at least one emission")
        if len({part.sample_rate for part in parts}) > 1:
            raise ValueError("the emissions of a sum must share a sample rate")
        object.__setattr__(self, "parts", parts)

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self.parts[0].sample_rate

    @property
    def length(self) -> int:
        return max(part.length for part in self.parts)

    def sent(self, start: int, stop: int, directions: np.ndarray) -> np.ndarray:
        out = np.zeros(stop - start)
        for part in self.parts:
            out += part.sent(start, stop, directions)
        return out

    def averaged(self) -> np.ndarray:
        out = np.zeros(self.length)
        for part in self.parts:
            out[: part.length] += part.averaged()
        return out

    def scaled(self, gain: float) -> Sum:
        return Sum(tuple(part.scaled(gain) for part in self.parts))

    def _powers(self, bands: Resolution, directions: np.ndarray) -> np.ndarray:
        powers = [part._powers(bands, directions) for part in self.parts]
        return np.sum(powers, axis=0)


@dataclass(frozen=True)
class Faded(Emission):
    """An emission's first ``length`` samples, faded in over the first ``fade`` and out
    over the last, by a squared sine, so that it does not start or stop at once. Its
    bands and its directivity are the emission's own."""

    part: Emission = dataclasses.field(repr=False)
    fade: int = 0
    #: How many samples of it are sent.
    length_samples: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.length_samples <= self.part.length:
            raise ValueError(f"a faded emission sends 1 to {self.part.length} samples")
        if not 0 <= 2 * self.fade <= self.length_samples:
            raise ValueError("an emission fades in and out over half its length at most")

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self.part.sample_rate

    @property
    def length(self) -> int:
        return self.length_samples

    def _ramp(self, start: int, stop: int) -> np.ndarray:
        """(stop - start,) the fade's gain at the samples ``start`` to ``stop``."""
        k = np.arange(start, stop)
        inside = (k >= 0) & (k < self.length_samples)
        if self.fade == 0:
            return inside.astype(float)
        edge = np.minimum(k, self.length_samples - 1 - k)
        ramp = np.sin(0.5 * np.pi * np.clip(edge / max(self.fade - 1, 1), 0.0, 1.0)) ** 2
        return np.where(inside, ramp, 0.0)

    def sent(self, start: int, stop: int, directions: np.ndarray) -> np.ndarray:
        return self.part.sent(start, stop, directions) * self._ramp(start, stop)

    def averaged(self) -> np.ndarray:
        return self.part.averaged()[: self.length_samples] * self._ramp(0, self.length_samples)

    def scaled(self, gain: float) -> Faded:
        return Faded(self.part.scaled(gain), self.fade, self.length_samples)

    def _powers(self, bands: Resolution, directions: np.ndarray) -> np.ndarray:
        return self.part._powers(bands, directions)


__all__ = ["Emission", "Faded", "Mixture", "Periodic", "Sum"]
