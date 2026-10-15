"""The simulated response from one source to one listener."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces import parameters as iso3382
from acoustic_spaces import synthesis
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.wave import WaveResponse
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, band_centres, resolution_of, to_octaves

#: The synthesized late part is smoothed over 10 ms from this long after the direct sound.
SMOOTH_AFTER_S = 0.05
#: Stage support is given for listeners at most this far from the source (m); ISO 3382-1
#: measures it at 1 m.
STAGE_DISTANCE_M = 1.5


@dataclass
class Response:
    """Early specular paths plus late energy per band and time slot, and the lowest
    bands from the wave solver when it ran (``wave``), which then replace the rays'.
    The bands are the 8 octave bands or the 24 third-octave bands the simulation ran in;
    room parameters are given in octave bands either way.

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
    #: Unit vector from the listener towards the source (room coordinates); None takes
    #: the direct path's, or the listener's forward axis without one.
    source_direction: np.ndarray | None = None
    #: The energy gain per band of the source's feed (its gain and equalizer), which the
    #: energies include, or None for none. G is taken against the same source in free
    #: field, so the feed does not change it.
    feed: np.ndarray | None = None
    #: With face groups (``simulate(groups=...)``): the moments of the late energy's
    #: reflection counts off each group (terms, bands, slots of ``moment_step``), for
    #: :meth:`reweighted`; else empty.
    late_moments: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    moment_step: float = 0.0

    @property
    def bands(self) -> int:
        return self.late.shape[0]

    @property
    def band_hz(self) -> tuple[int, ...]:
        """Centre frequencies (Hz) of the bands: octave or third-octave bands."""
        return band_centres(resolution_of(self.bands))

    def energy(self, time_step: float | None = None) -> np.ndarray:
        """Echogram: energy per band and time slot (bands, slots), early paths included."""
        return self._echogram(time_step)

    def octave_energy(self, time_step: float | None = None) -> np.ndarray:
        """Echogram in octave bands (8, slots): each octave the mean of its three
        third-octave bands, when the simulation ran in those."""
        return to_octaves(self._echogram(time_step))

    def weighted_energy(
        self, axis: np.ndarray, exponent: int = 2, time_step: float | None = None
    ) -> np.ndarray:
        """Echogram of a figure of eight facing ``axis`` (a unit vector, room
        coordinates): each arrival's energy times the cosine of its angle to the axis,
        taken without its sign, to the power ``exponent``. With 2 that is the energy the
        figure of eight picks up; with 1 the magnitude of its product with the pressure,
        as LFC takes it. Late energy without directions counts as arriving evenly from
        all of them."""
        return self._echogram(time_step, np.asarray(axis, dtype=float), exponent)

    def _echogram(
        self, time_step: float | None, axis: np.ndarray | None = None, exponent: int = 2
    ) -> np.ndarray:
        step = self.time_step if time_step is None else time_step
        slots = int(math.ceil(self.length_s / step - 1e-9))
        late = self.late if axis is None else self.late * self._late_weights(axis, exponent)
        if math.isclose(step, self.time_step):
            out = late[:, :slots].copy()
            if out.shape[1] < slots:
                out = np.pad(out, ((0, 0), (0, slots - out.shape[1])))
        else:
            # Spread each late slot evenly over the slots it overlaps.
            edges = np.arange(late.shape[1] + 1) * self.time_step
            cumulative = np.concatenate(
                [np.zeros((self.bands, 1)), np.cumsum(late, axis=1)], axis=1
            )
            new_edges = np.arange(slots + 1) * step
            out = np.diff(
                np.array([np.interp(new_edges, edges, row) for row in cumulative]), axis=1
            )
        early = self.early.energy
        if axis is not None:
            early = early * (np.abs(self.early.direction @ axis) ** exponent)[:, None]
        idx = (self.early.delay_s / step).astype(np.int64)
        keep = idx < slots
        for b in range(self.bands):
            np.add.at(out[b], idx[keep], early[keep, b])
        if self.wave is not None:
            out[: self.wave.bands] = self.wave.energy(step, slots, axis, exponent)
        return out

    def _late_weights(self, axis: np.ndarray, exponent: int) -> np.ndarray:
        """Per band and slot of ``late``, the mean |cos|^exponent of the late energy's
        directions to ``axis``: 1/3 or 1/2 for energy from all directions alike."""
        even = 1.0 / (exponent + 1.0)
        shape = self.late.shape
        energy = self.late_directional
        if len(self.late_directions) == 0 or energy.size == 0:
            return np.full(shape, even)
        gain = np.abs(self.late_directions @ axis) ** exponent
        total = energy.sum(axis=0)
        mean = np.full(total.shape, even)
        np.divide(np.tensordot(gain, energy, axes=1), total, out=mean, where=total > 0)
        coarse = (np.arange(shape[1]) * self.time_step / self.direction_step + 1e-9).astype(int)
        inside = coarse < mean.shape[1]
        return np.where(inside, mean[:, np.minimum(coarse, mean.shape[1] - 1)], even)

    def lateral_axis(self) -> np.ndarray:
        """The axis of a figure of eight with its null on the source (ISO 3382-1): level,
        across the direction of the source, towards the listener's left side of it."""
        towards = self.source_direction
        if towards is None:
            direct = np.flatnonzero((self.early.order == 0) & ~self.early.diffracted)
            towards = self.early.direction[direct[0]] if len(direct) else self.orientation[:, 0]
        side = np.cross([0.0, 0.0, 1.0], towards)
        if np.linalg.norm(side) < 1e-9:  # the source straight above or below
            side = self.orientation[:, 1]
        return side / np.linalg.norm(side)

    def fed(self, gain_db: Sequence[float] | np.ndarray, delay_s: float = 0.0) -> Response:
        """This response with the source's feed: ``gain_db`` per band (dB, the bands of
        the response) on every energy, and every arrival ``delay_s`` later."""
        gain = 10.0 ** (np.asarray(gain_db, dtype=float) / 10.0)
        if gain.shape != (self.bands,):
            raise ValueError(f"expected {self.bands} gains, one per band")
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("the delay must be a finite time, 0 or more")
        early = dataclasses.replace(
            self.early, delay_s=self.early.delay_s + delay_s, energy=self.early.energy * gain
        )
        directional = self.late_directional
        if directional.size:
            directional = _delayed(directional * gain[None, :, None], delay_s / self.direction_step)
        wave = None if self.wave is None else self.wave.fed(gain[: self.wave.bands], delay_s)
        moments = self.late_moments
        if moments.size:
            moments = _delayed(moments * gain[None, :, None], delay_s / self.moment_step)
        return dataclasses.replace(
            self,
            early=early,
            late_moments=moments,
            late=_delayed(self.late * gain[:, None], delay_s / self.time_step),
            length_s=self.length_s + delay_s,
            direct_delay_s=self.direct_delay_s + delay_s,
            transition_s=self.transition_s + delay_s,
            late_directional=directional,
            wave=wave,
            feed=gain if self.feed is None else self.feed * gain,
        )

    def _reference(self) -> np.ndarray:
        """Energy per octave band of the same source, fed alike, at 10 m in free field."""
        feed = np.ones(self.bands) if self.feed is None else self.feed
        return to_octaves(feed) / (4.0 * math.pi * 100.0)

    def reweighted(self, face_group: np.ndarray, ratio: np.ndarray) -> Response:
        """This response for other absorption, without tracing again
        (:mod:`acoustic_spaces.engine.reweight`): ``face_group`` the group of each face
        (-1 for none) as traced, ``ratio`` the change per reflection off each group,
        ln((1 - new) / (1 - old)) per group and band (:func:`~acoustic_spaces.engine.
        reweight.log_ratio`). The early paths re-weight exactly, the late part from its
        moments, which the response needs (``simulate(groups=...)``)."""
        from acoustic_spaces.engine import reweight

        ratio = np.asarray(ratio, dtype=float)
        if ratio.shape[1] != self.bands:
            raise ValueError(f"expected a ratio per group and each of {self.bands} bands")
        groups = ratio.shape[0]
        if self.late_moments.shape[0] != 1 + groups + groups * (groups + 1) // 2:
            raise ValueError(
                f"the response has no late moments of {groups} groups; "
                "simulate it with the same groups"
            )
        if self.wave is not None:
            raise ValueError("the wave solver's bands do not re-weight; trace without it")
        early = reweight.reweighted_paths(
            self.early, reweight.path_counts(self.early, face_group, groups), ratio
        )
        exponent = reweight.moment_exponent(self.late_moments, ratio)
        late = self.late * reweight.slot_factor(
            exponent, self.late.shape[1], self.time_step, self.moment_step
        )
        directional = self.late_directional
        if directional.size:
            factor = reweight.slot_factor(
                exponent, directional.shape[2], self.direction_step, self.moment_step
            )
            directional = directional * factor[None]
        return dataclasses.replace(self, early=early, late=late, late_directional=directional)

    def parameters(self) -> iso3382.RoomParameters:
        """ISO 3382-1 parameters per octave band, straight from the energies.

        Those of a figure of eight with its null on the source (LF, LFC, LJ) and the
        decay times of figures of eight along the listener's axes come from the
        directions the energy arrives from; stage support only for a listener within
        ``STAGE_DISTANCE_M`` of the source. IACC needs a binaural response:
        :func:`~acoustic_spaces.parameters.interaural_cross_correlation`.
        """
        reference = self._reference()  # free field at 10 m
        side = self.lateral_axis()

        def weighted(axis, exponent=2):
            return to_octaves(self.weighted_energy(axis, exponent))

        return iso3382.from_energy(
            self.octave_energy(),
            self.time_step,
            self.direct_delay_s,
            reference_energy=reference,
            stage=self.direct_delay_s * self.speed_of_sound <= STAGE_DISTANCE_M,
            lateral=(weighted(side), weighted(side, 1)),
            directional=[weighted(self.orientation[:, k]) for k in range(3)],
        )

    def levels(
        self,
        level_db: float | Sequence[float] = 94.0,
        parameters: iso3382.RoomParameters | None = None,
    ) -> np.ndarray:
        """Sound pressure level per octave band (dB SPL) of a source of sound power
        ``level_db`` (dB re 1 pW, the feed's gain included): one value spread evenly over
        the octave bands, or one per octave band
        (:meth:`~acoustic_spaces.scene.Source.power_db`). It is the band's sound power
        plus its G, less 31 dB, the level of a source of 0 dB at 10 m in free field."""
        params = self.parameters() if parameters is None else parameters
        power = np.asarray(level_db, dtype=float)
        if power.ndim == 0:
            power = np.full(
                len(OCTAVE_BANDS_HZ), float(power) - 10.0 * math.log10(len(OCTAVE_BANDS_HZ))
            )
        elif power.shape != (len(OCTAVE_BANDS_HZ),):
            raise ValueError("give the sound power in all or per octave band")
        return power + params.g + 10.0 * math.log10(1.0 / (4.0 * math.pi * 100.0))

    def speech(
        self,
        level_db: float | Sequence[float] = 94.0,
        noise_db: float | Sequence[float] | None = None,
        parameters: iso3382.RoomParameters | None = None,
        talker: iso3382.Talker = "male",
    ) -> iso3382.SpeechParameters:
        """STI, STIPA and U50 for speech from a source of sound power ``level_db`` (dB
        re 1 pW: in all, spread evenly over the octave bands, or per octave band), with
        background noise ``noise_db`` (dB SPL, one value or one per band), as the
        coverage maps take them, weighted for a male or female ``talker``.
        ``parameters`` are this response's, when already at hand."""
        params = self.parameters() if parameters is None else parameters
        level = self.levels(level_db, params)
        signal = np.where(np.isfinite(level), level, -100.0)
        energy = self.octave_energy()
        sti, stipa = (
            iso3382.speech_transmission_index(
                energy,
                self.time_step,
                signal_db=signal,
                noise_db=noise_db,
                stipa=short,
                talker=talker,
            )
            for short in (False, True)
        )
        u50 = iso3382.useful_to_detrimental(params.d50, signal, noise_db)
        return iso3382.SpeechParameters(sti=sti, stipa=stipa, u50=u50, signal_db=level)

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


def _delayed(x: np.ndarray, slots: float) -> np.ndarray:
    """``x`` with its last axis (time slots) ``slots`` later, a fraction of a slot
    shared between the two slots it falls between; as long as it needs to be."""
    if slots <= 0:
        return x
    whole = int(math.floor(slots))
    part = slots - whole
    pad = [(0, 0)] * (x.ndim - 1)
    out = np.pad(x, [*pad, (whole, 1)]) * (1.0 - part)
    out[..., 1:] += np.pad(x, [*pad, (whole, 0)]) * part
    return out


__all__ = ["Response"]
