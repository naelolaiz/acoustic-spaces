"""The simulated response from one source to one listener."""

from __future__ import annotations

import dataclasses
import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from acoustic_spaces import parameters as iso3382
from acoustic_spaces import synthesis
from acoustic_spaces.engine import coherent
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.wave import WaveResponse
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, band_centres, resolution_of, to_octaves

if TYPE_CHECKING:
    from acoustic_spaces.engine.reweight import Redraws

#: The synthesized late part is smoothed over 10 ms from this long after the direct sound.
SMOOTH_AFTER_S = 0.05
#: Stage support is measured this far from the source (m): ISO 3382-1:2009, annex C (after
#: Gade's ST1). It is given only for listeners within ``STAGE_TOLERANCE_M`` of it: its
#: first 10 ms hold the direct sound, whose energy goes as 1 / r^2, so a listener r m away
#: reads it up to 20 lg r dB off, 0.9 dB at 0.9 m and 0.8 dB at 1.1 m.
STAGE_DISTANCE_M = 1.0
STAGE_TOLERANCE_M = 0.1
#: The fields of a :class:`Response` that hold its late part, each of which may be given
#: as a :class:`Deferred`.
LATE_ARRAYS = ("late", "late_directional", "late_moments", "late_batches")


@dataclass(frozen=True)
class Deferred:
    """A late array that a :class:`Response` makes when it is read instead of holding
    it: what ``make()`` gives. A walk's places between those traced for their
    late part have theirs so, each mixed from the traced ones on either side
    (``simulate(late_from=...)``), which are held once however many places read them."""

    make: Callable[[], np.ndarray]

    def then(self, change: Callable[[np.ndarray], np.ndarray]) -> Deferred:
        """``change`` of this array, made when it is read."""
        make = self.make
        return Deferred(lambda: change(make()))

    def field(self) -> np.ndarray:
        """This, as a :class:`Response` takes it for one of its ``LATE_ARRAYS``: typed
        as the array the field reads as."""
        return cast(np.ndarray, self)


@dataclass
class Response:
    """Early specular paths plus late energy per band and time slot, and the lowest
    bands from the wave solver when it ran (``wave``), which then replace the rays'.
    The bands are the 8 octave bands or the 24 third-octave bands the simulation ran in;
    room parameters are given in octave bands either way.

    Energies are time-integrated intensity for a source of unit energy (the direct sound
    at r metres carries 1 / (4 pi r^2)); times are measured from emission.

    The late part's arrays (``late``, ``late_directional``, ``late_moments`` and
    ``late_batches``) may each be given as a :class:`Deferred`: the response then keeps
    that, and reading the field makes the array, which a read of the field again gets
    until that of another response is made.
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
    #: Volume of the listener's room (m3; None in a model that is not closed): how
    #: densely the synthesized late part's reflections come.
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
    #: How much later the feed makes every arrival (s), which the times include.
    feed_delay_s: float = 0.0
    #: With face groups (``simulate(groups=...)``): the moments of the late energy's
    #: reflection counts off each group (terms, bands, slots of ``moment_step``), for
    #: :meth:`reweighted`; else empty.
    late_moments: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    moment_step: float = 0.0
    #: With the late rays traced in batches (``simulation.late.batches`` of 2 or more):
    #: ``late`` as each batch gives it (batches, bands, slots of ``batch_step``, a whole
    #: number of ``time_step``), each an estimate of its own and their mean ``late``, for
    #: :meth:`standard_errors`; else empty.
    late_batches: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    batch_step: float = 0.01
    #: Outdoors: the energy per band of the same source at the listener in free field
    #: (its direct sound over the straight line, with the air's attenuation and the
    #: source's directivity, without the feed), for :meth:`relative_to_free_field`; else
    #: None.
    free_field: np.ndarray | None = None
    #: The early part as the echogram reads its coherent bands, with what it came from.
    _pressure: tuple | None = field(default=None, init=False, repr=False, compare=False)
    #: What the coherent bands' echograms share with those of the other draws of the
    #: absorption the response was re-weighted for (:meth:`reweighted` with ``redraws``).
    _same_delays: coherent.SameDelays | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __getstate__(self) -> dict:
        """What a pickle or a copy of the response holds: its attributes, with the late
        arrays as they read (one given as a :class:`Deferred` made)."""
        return {**vars(self), **{name: getattr(self, name) for name in LATE_ARRAYS}}

    @property
    def bands(self) -> int:
        return self.late.shape[0]

    @property
    def coherent_bands(self) -> int:
        """How many of the lowest bands the early paths add in pressure in
        (:mod:`~acoustic_spaces.engine.coherent`); the wave solver's take the place of
        those it covers."""
        return self.early.coherent_bands

    def _early_pressure(self) -> dict:
        """The :class:`~acoustic_spaces.engine.coherent.EarlyPressure` of each point
        source's paths, as far as they were made, for
        :func:`~acoustic_spaces.engine.coherent.coherent_energy`."""
        early = self.early
        key = (early, early.energy, early.delay_s, early.direction, early.spectrum, early.point)
        if self._pressure is None or any(
            a is not b for a, b in zip(self._pressure[0], key, strict=True)
        ):
            self._pressure = (key, {})
        return self._pressure[1]

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
        # A path before its emission (none: see EarlyReflections.at_moment) would count
        # in the first slot, as what a measured response's filters spread before it.
        idx = np.maximum(self.early.delay_s / step, 0.0).astype(np.int64)
        keep = idx < slots
        count = self.coherent_bands
        for b in range(count, self.bands):
            np.add.at(out[b], idx[keep], early[keep, b])
        if count and len(self.early):
            # The lowest bands read from the early part's pressure, as from a measured
            # response (or the wave solver's).
            out[:count] += coherent.coherent_energy(
                self.early,
                step,
                slots,
                axis,
                exponent,
                start=self.direct_delay_s,
                cache=self._early_pressure(),
                kept=self._same_delays,
            )
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

        def late(name: str, step: float, gains: np.ndarray, empty: bool = False) -> np.ndarray:
            """The late array ``name`` (slots of ``step`` s) times ``gains`` and
            ``delay_s`` later, one without values left as it is unless ``empty``; one
            made when it is read (:class:`Deferred`) is fed when it is read."""

            def feed(x: np.ndarray) -> np.ndarray:
                return _delayed(x * gains, delay_s / step) if empty or x.size else x

            held = vars(self)[name]
            return held.then(feed).field() if isinstance(held, Deferred) else feed(held)

        directional = late("late_directional", self.direction_step, gain[None, :, None])
        wave = None if self.wave is None else self.wave.fed(gain[: self.wave.bands], delay_s)
        moments = late("late_moments", self.moment_step, gain[None, :, None])
        batches = late("late_batches", self.batch_step, gain[None, :, None])
        return dataclasses.replace(
            self,
            early=early,
            late_moments=moments,
            late_batches=batches,
            late=late("late", self.time_step, gain[:, None], empty=True),
            length_s=self.length_s + delay_s,
            direct_delay_s=self.direct_delay_s + delay_s,
            transition_s=self.transition_s + delay_s,
            late_directional=directional,
            wave=wave,
            feed=gain if self.feed is None else self.feed * gain,
            feed_delay_s=self.feed_delay_s + delay_s,
        )

    def _reference(self) -> np.ndarray:
        """Energy per octave band of the same source, fed alike, at 10 m in free field."""
        feed = np.ones(self.bands) if self.feed is None else self.feed
        return to_octaves(feed) / (4.0 * math.pi * 100.0)

    def reweighted(
        self,
        face_group: np.ndarray,
        ratio: np.ndarray,
        absorption: tuple[np.ndarray, np.ndarray] | None = None,
        admittance_of=None,
        redraws: Redraws | None = None,
    ) -> Response:
        """This response for other absorption, without tracing again
        (:mod:`acoustic_spaces.engine.reweight`): ``face_group`` the group of each face
        (-1 for none) as traced, ``ratio`` the change per reflection off each group,
        ln((1 - new) / (1 - old)) per group and band (:func:`~acoustic_spaces.engine.
        reweight.log_ratio`), and ``absorption`` each group's absorption (groups, bands)
        as traced and as ``ratio`` changes it, (old, new). The early paths re-weight
        exactly (but for reflections off a ground with an impedance or with zones, whose
        factor is not 1 - absorption: see the module), the late part from its moments,
        which the response needs (``simulate(groups=...)``). Where the early paths add in pressure
        (:mod:`~acoustic_spaces.engine.coherent`), the phase of their reflections
        depends on the absorption too, and turns with it
        (:func:`~acoustic_spaces.engine.reweight.turned_paths`, with
        ``admittance_of`` if given): that needs ``absorption``, unless no path reflects
        off a group whose absorption changes. ``redraws`` (a
        :class:`~acoustic_spaces.engine.reweight.Redraws` of this response's early paths),
        for one of many draws of the absorption, holds what does not depend on the draw;
        the response is the same with it as without it."""
        from acoustic_spaces.engine import reweight

        ratio = np.asarray(ratio, dtype=float)
        if ratio.ndim != 2 or ratio.shape[1] != self.bands:
            raise ValueError(f"expected a ratio per group and each of {self.bands} bands")
        groups = ratio.shape[0]
        if self.late_moments.shape[0] != 1 + groups + groups * (groups + 1) // 2:
            raise ValueError(
                f"the response has no late moments of {groups} groups; "
                "simulate it with the same groups"
            )
        if self.wave is not None:
            raise ValueError("the wave solver's bands do not re-weight; trace without it")
        if redraws is None:
            counts = reweight.path_counts(self.early, face_group, groups)
        else:
            counts = redraws.counts(self.early, face_group, groups)
        early = reweight.reweighted_paths(self.early, counts, ratio)
        if self.early.spectrum is not None:
            if absorption is not None:
                old, new = (np.asarray(a, dtype=float) for a in absorption)
                if old.shape != ratio.shape or new.shape != ratio.shape:
                    raise ValueError(
                        f"expected the absorption of {groups} groups in {self.bands} bands"
                    )
                turn = reweight.turned_paths if redraws is None else redraws.turned
                early = turn(early, face_group, old, new, admittance_of)
            elif len(counts) and np.any(counts[:, np.any(ratio != 0.0, axis=1)] > 0):
                raise ValueError(
                    "the early paths add in pressure in the lowest bands, where the phase "
                    "of a reflection depends on the absorption: give the absorption as "
                    "traced and as re-weighted for"
                )
        exponent = reweight.moment_exponent(self.late_moments, ratio)
        factor = reweight.slot_factor(
            exponent, self.late.shape[1], self.time_step, self.moment_step
        )
        late = self.late * factor
        batches = self.late_batches
        if batches.size:
            batches = batches * reweight.slot_factor(
                exponent, batches.shape[2], self.batch_step, self.moment_step
            )
        directional = self.late_directional
        if directional.size:
            factor = reweight.slot_factor(
                exponent, directional.shape[2], self.direction_step, self.moment_step
            )
            directional = directional * factor[None]
        out = dataclasses.replace(
            self, early=early, late=late, late_directional=directional, late_batches=batches
        )
        if redraws is not None:
            out._same_delays = redraws.delays
        return out

    def parameters(self, energy: np.ndarray | None = None) -> iso3382.RoomParameters:
        """ISO 3382-1 parameters per octave band, straight from the energies.

        Those of a figure of eight with its null on the source (LF, LFC, LJ) and the
        decay times of figures of eight along the listener's axes come from the
        directions the energy arrives from; stage support only for a listener 0.9 to
        1.1 m from the source (``STAGE_DISTANCE_M`` and ``STAGE_TOLERANCE_M``), as the
        standard measures it 1 m away. IACC needs a binaural response:
        :func:`~acoustic_spaces.parameters.interaural_cross_correlation`. Outdoors (with
        :attr:`free_field`), G of a band that does not decay as a room's does is that of
        the energy the response holds. ``energy``: its :meth:`octave_energy`, when already
        at hand.
        """
        energy = self.octave_energy() if energy is None else energy
        side = self.lateral_axis()

        def weighted(axis, exponent=2):
            return to_octaves(self.weighted_energy(axis, exponent))

        return self._parameters(
            energy,
            lateral=(weighted(side), weighted(side, 1)),
            directional=[weighted(self.orientation[:, k]) for k in range(3)],
        )

    def _parameters(self, energy: np.ndarray, **directions) -> iso3382.RoomParameters:
        """:meth:`parameters` from the echogram ``energy`` in octave bands, with those
        from the directions of the energy given in ``directions`` (else NaN)."""
        return iso3382.from_energy(
            energy,
            self.time_step,
            self.direct_delay_s,
            reference_energy=self._reference(),  # free field at 10 m
            stage=self._at_stage_distance(),
            open_air=self.free_field is not None,
            **directions,
        )

    def _at_stage_distance(self) -> bool:
        """Whether the listener is close enough to 1 m from the source for stage support:
        the direct sound's path (its time, the feed's delay taken off) within
        ``STAGE_TOLERANCE_M`` of ``STAGE_DISTANCE_M``, to a nanometre, which a distance
        taken through a delay and back can miss by."""
        distance = (self.direct_delay_s - self.feed_delay_s) * self.speed_of_sound
        return abs(distance - STAGE_DISTANCE_M) <= STAGE_TOLERANCE_M + 1e-9

    def standard_errors(self) -> iso3382.RoomParameters | None:
        """The standard error of each of :meth:`parameters` from the late rays' batches
        (``simulation.late.batches``): one standard deviation of how much it would
        change between runs with other random seeds and as many rays. A jackknife: the
        parameters again without each of the n batches in turn, whose spread, times
        (n - 1) / n under the root, is the standard error. The early paths and the wave
        solver's bands do not depend on the seed, and add none. Those from the
        directions the late energy arrives from (LF, LFC, LJ and the decay times along
        the listener's axes) are NaN, as IACC is; so is a parameter some batches leave
        undefined or infinite (a clarity without late energy). None without 2 batches or more.

        The batches hold the energy in slots of :attr:`batch_step`; without a batch,
        each slot of :attr:`time_step` keeps its share of its slot of
        :attr:`batch_step`, so a strong arrival in one of them (a ray through the
        listener long after the rest, in a room whose walls keep a few going) leaves
        with the batch it came in."""
        values = [self._parameters(energy) for energy in self._without_each_batch()]
        if not values:
            return None
        return iso3382.RoomParameters(
            **{
                f.name: _jackknife([getattr(v, f.name) for v in values])
                for f in dataclasses.fields(iso3382.RoomParameters)
            }
        )

    def _without_each_batch(self):
        """The echogram in octave bands without each batch of late rays in turn (the
        mean of the others in its place), for a jackknife; none without 2 batches."""
        batches = self.late_batches
        n = len(batches)
        if n < 2:
            return
        echogram = self.energy()
        late = self.late[:, : echogram.shape[1]]
        step = max(1, round(self.batch_step / self.time_step))
        mean = batches.mean(axis=0)
        for k in range(n):
            # The late energy without batch k (the mean of the others), as a factor on it.
            factor = np.ones_like(mean)
            np.divide(n * mean - batches[k], (n - 1) * mean, out=factor, where=mean > 0)
            factor = np.repeat(factor, step, axis=1)[:, : late.shape[1]]
            energy = echogram.copy()
            energy[:, : factor.shape[1]] += late[:, : factor.shape[1]] * (factor - 1.0)
            if self.wave is not None:
                energy[: self.wave.bands] = echogram[: self.wave.bands]
            yield to_octaves(np.maximum(energy, 0.0))

    def relative_to_free_field(self) -> np.ndarray:
        """The level relative to free field per octave band (dB): how much louder (or,
        negative, quieter) the source is here than the same source the same distance
        away in free field, which the air between them attenuates as here. The ground,
        the weather, buildings, barriers and vegetation make the difference; with the
        sign turned, it is the excess attenuation of outdoor acoustics. It compares the
        energy the response holds (nothing is extrapolated past its end, as G does for
        a room's decay) with the free field's; -inf where no sound arrives, NaN in every
        band of a response without :attr:`free_field`."""
        if self.free_field is None:
            return np.full(len(OCTAVE_BANDS_HZ), math.nan)
        return self._relative_to_free_field(self.octave_energy())

    def _relative_to_free_field(self, energy: np.ndarray) -> np.ndarray:
        """:meth:`relative_to_free_field` of the echogram ``energy`` in octave bands."""
        if self.free_field is None:
            return np.full(len(OCTAVE_BANDS_HZ), math.nan)
        held = energy.sum(axis=1) / self._reference()  # re 10 m free field
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10.0 * np.log10(held / (to_octaves(self.free_field) * (4.0 * math.pi * 100.0)))

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

    def levels_or_held(
        self,
        level_db: float | Sequence[float] = 94.0,
        parameters: iso3382.RoomParameters | None = None,
    ) -> np.ndarray:
        """:meth:`levels`, but in a band whose G is undefined (the few rays that reach a
        far place may carry no energy in it, or too little of its decay to read), the
        level of the energy the response holds from the direct sound's arrival, as
        outdoors: -inf where none arrives. The level a listener is given in
        ``simulate``'s report, the editor's results, the demos' captions and each place
        of :func:`~acoustic_spaces.walkthrough.pass_by`; the coverage maps and
        :meth:`speech` leave such a band undefined."""
        params = self.parameters() if parameters is None else parameters
        return self._levels_or_held(level_db, params)

    def _levels_or_held(
        self, level_db, params: iso3382.RoomParameters, energy: np.ndarray | None = None
    ) -> np.ndarray:
        """:meth:`levels_or_held` from the parameters ``params`` of the echogram
        ``energy`` in octave bands (default the response's own)."""
        if not np.isfinite(params.g).all():
            energy = self.octave_energy() if energy is None else energy
            first = min(int(self.direct_delay_s / self.time_step), energy.shape[1] - 1)
            with np.errstate(divide="ignore"):
                held = 10.0 * np.log10(energy[:, first:].sum(axis=1) / self._reference())
            params = dataclasses.replace(params, g=np.where(np.isfinite(params.g), params.g, held))
        return self.levels(level_db, params)

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
        coverage maps take them, weighted for a male or female ``talker``; the STI and
        STIPA are NaN when a band from 125 Hz up has no level (its G is undefined).
        ``parameters`` are this response's, when already at hand."""
        params = self.parameters() if parameters is None else parameters
        return self._speech(self.octave_energy(), level_db, noise_db, params, talker)

    def _speech(
        self, energy: np.ndarray, level_db, noise_db, params: iso3382.RoomParameters, talker
    ) -> iso3382.SpeechParameters:
        """:meth:`speech` from the echogram ``energy`` in octave bands and its
        parameters ``params``."""
        level = self.levels(level_db, params)
        signal = np.where(np.isfinite(level), level, -100.0)
        # No STI without the level of each of its bands, 125 Hz to 8 kHz.
        sti, stipa = (
            iso3382.speech_transmission_index(
                energy,
                self.time_step,
                signal_db=signal,
                noise_db=noise_db,
                stipa=short,
                talker=talker,
            )
            if np.isfinite(level[1:]).all()
            else math.nan
            for short in (False, True)
        )
        u50 = iso3382.useful_to_detrimental(params.d50, signal, noise_db)
        return iso3382.SpeechParameters(sti=sti, stipa=stipa, u50=u50, signal_db=level)

    def levels_and_speech(
        self,
        level_db: float | Sequence[float] = 94.0,
        noise_db: float | Sequence[float] | None = None,
        talker: iso3382.Talker = "male",
        parameters: iso3382.RoomParameters | None = None,
        energy: np.ndarray | None = None,
    ) -> iso3382.Levels:
        """The levels and speech results at the listener for a source of sound power
        ``level_db`` (dB re 1 pW, in all or per octave band), as a run's report gives
        them: :meth:`levels_or_held` and its A-weighted sum; the STI, STIPA and U50 of
        :meth:`speech` with background noise ``noise_db`` for ``talker``; and
        :meth:`relative_to_free_field`, with its A-weighted value (the A-weighted level
        less that of the same source in free field). ``parameters`` are this
        response's, and ``energy`` its :meth:`octave_energy`, when already at hand."""
        params = self.parameters(energy) if parameters is None else parameters
        energy = self.octave_energy() if energy is None else energy
        return self._levels_and_speech(energy, level_db, noise_db, talker, params)

    def _levels_and_speech(
        self, energy: np.ndarray, level_db, noise_db, talker, params: iso3382.RoomParameters
    ) -> iso3382.Levels:
        """:meth:`levels_and_speech` from the echogram ``energy`` in octave bands and
        its parameters ``params``."""
        spl = self._levels_or_held(level_db, params, energy)
        speech = self._speech(energy, level_db, noise_db, params, talker)
        relative = self._relative_to_free_field(energy)
        total = iso3382.a_weighted(spl)
        # In a band no sound arrives in (a level of -inf) the level in free field
        # cannot be read back from the level: the A-weighted value is then NaN.
        with np.errstate(invalid="ignore"):
            relative_a = total - iso3382.a_weighted(spl - relative)
        return iso3382.Levels(
            spl=spl,
            spl_a=total,
            sti=speech.sti,
            stipa=speech.stipa,
            u50=speech.u50,
            re_free_field=relative,
            re_free_field_a=relative_a,
        )

    def level_errors(
        self,
        level_db: float | Sequence[float] = 94.0,
        noise_db: float | Sequence[float] | None = None,
        talker: iso3382.Talker = "male",
    ) -> iso3382.Levels | None:
        """The standard error of each of :meth:`levels_and_speech` from the late rays'
        batches, by the jackknife of :meth:`standard_errors`: NaN where some batch
        leaves the value undefined or infinite, None without 2 batches or more. Where
        G is defined, a band's level is its G and the source's power, so its error is
        G's; where it is not, the level is that of the energy held
        (:meth:`levels_or_held`) and has an error of its own."""
        values = [
            self._levels_and_speech(energy, level_db, noise_db, talker, self._parameters(energy))
            for energy in self._without_each_batch()
        ]
        if not values:
            return None
        errors: dict[str, Any] = {}
        for f in dataclasses.fields(iso3382.Levels):
            error = _jackknife([getattr(v, f.name) for v in values])
            errors[f.name] = error if error.ndim else float(error)
        return iso3382.Levels(**errors)

    def impulse_response(self, sample_rate: int = 48_000, seed: int | None = None) -> np.ndarray:
        """Mono impulse response; the direct sound at r metres is an impulse of height 1 / r.

        Early paths become fractional-delay impulses filtered per band, in the lowest
        bands with the phase of their reflections
        (:mod:`~acoustic_spaces.engine.coherent`), in turbulence as at one moment
        (:meth:`moment`); the late energy drives a random-sign
        Dirac sequence with the density of reflections in the room.
        From 50 ms after the direct sound the late energy is smoothed over 10 ms first,
        so a single strong ray late in the tail does not become a click. The wave
        solver's bands, when it ran, take the place of the lowest ones: through the
        crossover the early paths make way for them in pressure and the noise, which
        does not interfere with them, in energy (``WaveResponse.high_pass``).
        """
        n = int(math.ceil(self.length_s * sample_rate))
        scale = synthesis.ENERGY_TO_IR
        paths = coherent.with_own_phases(self.moment(seed), self.seed if seed is None else seed)
        band_signals, quadrature = coherent.band_impulses(paths, n, sample_rate)
        low = coherent.signal(paths, n, sample_rate)[:, 0]
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
            return (
                synthesis.combine_bands(band_signals + noise_signals, sample_rate, quadrature) + low
            )
        # The wave bands take the early paths' place in pressure, the noise's in energy.
        early = synthesis.combine_bands(band_signals, sample_rate, quadrature) + low
        early = self.wave.high_pass(early, sample_rate)
        noise = synthesis.combine_bands(noise_signals, sample_rate)
        late = self.wave.high_pass(noise, sample_rate, late=True)
        return early + late + self.wave.field(sample_rate, n)[:, 0]

    def moment(self, seed: int | None = None) -> EarlyReflections:
        """The early paths at a moment of the turbulent air, drawn with ``seed``
        (default the response's): outdoors in turbulence the eddies delay each path at
        random, and an impulse response is one moment of them, whose levels scatter
        about the echogram's average. Without turbulence, the early paths."""
        return self.early.frozen(self.seed if seed is None else seed)

    def spatial_impulse_response(
        self, fmt, sample_rate: int = 48_000, seed: int | None = None, *, threads: int = 1
    ) -> np.ndarray:
        """Impulse responses (samples, channels) in a format of
        :mod:`acoustic_spaces.spatial`: ``Ambisonics(order)``, ``Loudspeakers.layout(name)``,
        ``Binaural(hrtf)`` or ``Mono()``, its channels rendered on ``threads`` threads
        (the same result with any number; at most the computer's cores)."""
        from acoustic_spaces.spatial.render import spatial_impulse_response

        return spatial_impulse_response(self, fmt, sample_rate, seed, threads=threads)


class _LateArray:
    """One of a :class:`Response`'s ``LATE_ARRAYS``, kept under the field's name among
    the response's own attributes: read, it is the array given, or what the
    :class:`Deferred` given makes. The array made last is kept with a weak reference to
    its :class:`Deferred`, which keeps none of what it is made from and lets the array
    go with it, so that reading the field of one response again and again makes it
    once."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._last: tuple[weakref.ref[Deferred], np.ndarray] | None = None

    def __get__(self, response: Response | None, owner: type | None = None) -> np.ndarray:
        # Without a response (the class asked), there is no such attribute, as for a
        # dataclass's field without a default value or with a factory for one.
        held = None if response is None else vars(response).get(self.name)
        if held is None:
            raise AttributeError(self.name)
        if not isinstance(held, Deferred):
            return held
        last = self._last
        if last is not None and last[0]() is held:
            return last[1]
        made = held.make()
        self._last = (weakref.ref(held, self._forget), made)
        return made

    def _forget(self, gone: weakref.ref[Deferred]) -> None:
        last = self._last
        if last is not None and last[0] is gone:
            self._last = None

    def __set__(self, response: Response, value: np.ndarray | Deferred) -> None:
        vars(response)[self.name] = value


for _name in LATE_ARRAYS:
    setattr(Response, _name, _LateArray(_name))


def _jackknife(replicas) -> np.ndarray:
    """The standard error of a value from its ``replicas`` (n, ...), each found without
    one of n batches: their spread, times (n - 1) / n under the root. NaN where a
    replica is undefined or infinite (no spread to a clarity of inf)."""
    replicas = np.array(replicas, dtype=float)
    replicas[~np.isfinite(replicas)] = math.nan
    n = len(replicas)
    spread = ((replicas - replicas.mean(axis=0)) ** 2).sum(axis=0)
    return np.sqrt((n - 1) / n * spread)


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


__all__ = ["LATE_ARRAYS", "Deferred", "Response"]
