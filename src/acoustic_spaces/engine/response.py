"""The simulated response from one source to one listener."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces import parameters as iso3382
from acoustic_spaces import synthesis
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.wave import WaveResponse

#: The synthesized late part is smoothed over 10 ms from this long after the direct sound.
SMOOTH_AFTER_S = 0.05


@dataclass
class Response:
    """Early specular paths plus late energy per band and time slot, and the lowest
    bands from the wave solver when it ran (``wave``), which then replace the rays'.

    Energies are time-integrated intensity for a source of unit energy (the direct sound
    at r metres carries 1 / (4 pi r^2)); times are measured from emission.
    """

    source: str
    listener: str
    early: EarlyReflections
    late: np.ndarray  # (bands, slots)
    time_step: float  # s, slot width of ``late``
    length_s: float
    #: Arrival time of the direct sound (s), whether or not the direct path is blocked.
    direct_delay_s: float
    #: Early paths end here (s, from emission); the late part covers everything else.
    transition_s: float
    volume_m3: float | None
    speed_of_sound: float
    #: Seed for the noise of the synthesized late part.
    seed: int = 0
    #: The listener's head axes (forward, left, up) as columns, in room coordinates.
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    #: Arrival directions of the late energy (room coordinates) and the late energy
    #: split between them (directions, bands, slots of ``direction_step``). Without
    #: them the late part arrives evenly from all directions.
    late_directions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    late_directional: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    direction_step: float = 0.01
    #: The lowest octave bands from the wave solver, or None.
    wave: WaveResponse | None = None

    @property
    def bands(self) -> int:
        return self.late.shape[0]

    def energy(self, time_step: float | None = None) -> np.ndarray:
        """Echogram: energy per band and time slot (bands, slots), early paths included."""
        step = self.time_step if time_step is None else time_step
        slots = int(math.ceil(self.length_s / step - 1e-9))
        if math.isclose(step, self.time_step):
            out = self.late[:, :slots].copy()
            if out.shape[1] < slots:
                out = np.pad(out, ((0, 0), (0, slots - out.shape[1])))
        else:
            # Spread each late slot evenly over the slots it overlaps.
            edges = np.arange(self.late.shape[1] + 1) * self.time_step
            cumulative = np.concatenate(
                [np.zeros((self.bands, 1)), np.cumsum(self.late, axis=1)], axis=1
            )
            new_edges = np.arange(slots + 1) * step
            out = np.diff(
                np.array([np.interp(new_edges, edges, row) for row in cumulative]), axis=1
            )
        idx = (self.early.delay_s / step).astype(np.int64)
        keep = idx < slots
        for b in range(self.bands):
            np.add.at(out[b], idx[keep], self.early.energy[keep, b])
        if self.wave is not None:
            out[: self.wave.bands] = self.wave.energy(step, slots)
        return out

    def parameters(self) -> iso3382.RoomParameters:
        """ISO 3382-1 parameters per octave band, straight from the energies."""
        reference = 1.0 / (4.0 * math.pi * 100.0)  # free field at 10 m
        return iso3382.from_energy(
            self.energy(), self.time_step, self.direct_delay_s, reference_energy=reference
        )

    def impulse_response(self, sample_rate: int = 48_000, seed: int | None = None) -> np.ndarray:
        """Mono impulse response; the direct sound at r metres is an impulse of height 1 / r.

        Early paths become fractional-delay impulses filtered per band; the late energy
        drives a random-sign Dirac sequence with the density of reflections in the room.
        From 50 ms after the direct sound the late energy is smoothed over 10 ms first,
        so a single strong ray late in the tail does not become a click. The wave
        solver's bands, when it ran, take the place of the lowest ones: through the
        crossover the early paths make way for them in pressure and the noise, which
        does not interfere with them, in energy (``WaveResponse.high_pass``).
        """
        n = int(math.ceil(self.length_s * sample_rate))
        scale = synthesis.ENERGY_TO_IR
        band_signals = synthesis.place_impulses(
            self.early.delay_s, np.sqrt(scale * self.early.energy), n, sample_rate
        )
        noise_signals = np.zeros_like(band_signals)
        if np.any(self.late > 0):
            rng = np.random.default_rng(self.seed if seed is None else seed)
            noise = synthesis.dirac_sequence(
                n, sample_rate, self.volume_m3, self.speed_of_sound, rng, self.time_step
            )
            late = synthesis.smooth_energy(
                self.late, self.time_step, self.direct_delay_s + SMOOTH_AFTER_S
            )
            noise_signals = synthesis.shape_noise(scale * late, self.time_step, noise, sample_rate)
        if self.wave is None:
            return synthesis.combine_bands(band_signals + noise_signals, sample_rate)
        # The wave bands take the early paths' place in pressure, the noise's in energy.
        early = self.wave.high_pass(synthesis.combine_bands(band_signals, sample_rate), sample_rate)
        noise = synthesis.combine_bands(noise_signals, sample_rate)
        late = self.wave.high_pass(noise, sample_rate, late=True)
        return early + late + self.wave.field(sample_rate, n)[:, 0]

    def spatial_impulse_response(
        self, fmt, sample_rate: int = 48_000, seed: int | None = None
    ) -> np.ndarray:
        """Impulse responses (samples, channels) in a format of
        :mod:`acoustic_spaces.spatial`: ``Ambisonics(order)``, ``Loudspeakers.layout(name)``,
        ``Binaural(hrtf)`` or ``Mono()``."""
        from acoustic_spaces.spatial.render import spatial_impulse_response

        return spatial_impulse_response(self, fmt, sample_rate, seed)


__all__ = ["Response"]
