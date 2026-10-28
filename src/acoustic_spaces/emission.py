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
the source the emission's own directivity (:meth:`Mixture.directivity`).

:class:`Mixture` is an emission made of signals that do not correlate with one another
(harmonics, as a cosine and a sine each, and noises in separate bands), each with an
amplitude in every direction on the grid of :func:`acoustic_spaces.directivity.on_grid`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from acoustic_spaces import synthesis
from acoustic_spaces.directivity import Balloon, on_grid, sphere_mean
from acoustic_spaces.materials.bands import Resolution, band_centres

#: Samples of the signals' spectra, one stretch after another, for their band powers.
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

    def directivity(self, bands: Resolution = "octave", name: str = "") -> Balloon:
        """The emission's directivity: its energy in each band and direction on its
        grid, over the sphere's mean."""
        count = len(band_centres(bands))
        size = min(self.length, _SEGMENT)
        # The band gains add up to one: they share each frequency's power out.
        gains = synthesis.band_gains(size, self.sample_rate, count).T
        powers = np.zeros((len(self.signals), count))
        for start in range(0, self.length - size + 1, size):
            spectra = np.abs(np.fft.rfft(self.signals[:, start : start + size], axis=1)) ** 2
            powers += spectra @ gains
        return Balloon.from_table(self.weights**2 @ powers, name)


__all__ = ["Emission", "Mixture"]
