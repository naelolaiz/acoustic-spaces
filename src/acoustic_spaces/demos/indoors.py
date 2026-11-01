"""A pass-by heard inside a house standing at each listener, as far from the track as
the listener: the train's sound through the house's closed window and the rumble the
ground's vibration makes in its room, or either alone (``--indoors``).

The window: each part of the train is simulated and played outdoors as the pass-by
demo plays it (:mod:`acoustic_spaces.demos.passby`), then brought indoors band by band
by the façade's level difference (:data:`acoustic_spaces.facade.HOUSE`, EN ISO
12354-3): its levels per octave band, and its sound through a zero-phase filter that
changes each octave band's energy by as much
(:func:`~acoustic_spaces.synthesis.matched_gains`). The sound keeps the direction it
comes from outdoors; the room's own reverberation is not added.

The rumble: by the FTA's general method (:mod:`acoustic_spaces.vibration`), the ground
outside shakes at each moment as the train's base curve says at the distance of its
nearest wheels, as if it ran alongside that far away: when it runs past, the manual's
level for the track's distance; as it comes and goes, less. The house, a small masonry
building whose structure is not known, adds the manual's 3 dB on its floors, and its
room's surfaces radiate the rumble at the level of table 6-14 for vibration peaking
below 30 Hz (the railway's vibration.py's defaults). Its sound is noise in the third
octave at 25 Hz (:data:`~acoustic_spaces.vibration.RUMBLE_HZ`), made to that
A-weighted level at each moment, the same at every point of the room (its wavelength,
14 m, is longer than the room), so the same in each ear (:func:`pressure_gains`). Its
level in the mix is the engine's own: a sound pressure in a mix is the feeds' RMS
(:data:`~acoustic_spaces.demos.road.SOUND_RMS`) over the loudest part's sound power,
as a point source's direct sound at r metres is an impulse of height 1 / r and its
level its sound power less 10 lg(4 π r²)
(:meth:`~acoustic_spaces.engine.response.Response.levels`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces import vibration
from acoustic_spaces.facade import HOUSE, Facade

#: What ``--indoors`` hears without a house: the pass outdoors, as before.
OFF = "off"
#: The rumble's name among a pass's parts.
RUMBLE = "rumble"
#: The house the pass-by demos put at each listener (the railway's vibration.py's
#: defaults): a small masonry building, its structure not known, on its first floor;
#: the vibration peaking below 30 Hz, as the manual says it does for most surface track.
BUILDING, FLOOR, PEAK = "small_masonry", 1, "low"
#: The rumble's third octave: from 2^(-1/6) to 2^(1/6) of its centre.
THIRD = 2.0 ** (1.0 / 6.0)
#: How long the rumble fades in and out at its very start and end (s).
FADE_S = 1.0
#: How finely the rumble's level is worked out along the pass (s).
STEP_S = 0.05


def choices(noun: str) -> tuple[str, ...]:
    """What ``--indoors`` takes for a vehicle called ``noun``: off; the vehicle and the
    rumble; the vehicle through the window alone; the rumble alone."""
    return (OFF, f"{noun}_and_{RUMBLE}", noun, RUMBLE)


def help_of(noun: str, away: str = "the track") -> str:
    """The help of ``--indoors`` for a vehicle called ``noun`` (or the road works),
    the house standing as far from ``away`` as the listener."""
    off, both, alone, rumble = choices(noun)
    return (
        f"where the {noun} is heard: {off}, outdoors at each listener; or inside a house "
        f"standing there, as far from {away}: {both}, the {noun} through the house's "
        "closed window (4-16-4 double glazing, EN ISO 12354-3) and the rumble its "
        "ground-borne vibration makes in the room (the FTA's general method); "
        f"{alone}, through the window alone; {rumble}, the rumble alone"
    )


@dataclass(frozen=True)
class Wheels:
    """What shakes the ground as a vehicle goes along its walk: its base curve
    (:data:`~acoustic_spaces.vibration.CURVES`) at ``speed_kmh``, and its wheels, each
    ``behind_m`` behind its front and ``offset_db`` under the curve (a freight train's
    wagons, :data:`~acoustic_spaces.vibration.FREIGHT_WAGONS_DB`)."""

    curve: str
    speed_kmh: float
    behind_m: tuple[float, ...]
    offset_db: tuple[float, ...]

    @property
    def length_s(self) -> float:
        """How long after its front its last wheels pass a place (s)."""
        return max(self.behind_m) / (self.speed_kmh / 3.6)


@dataclass(frozen=True)
class House:
    """A house at each listener of a pass, and what is heard in it: the vehicle's sound
    through its window (``airborne``), the rumble of its floors and walls
    (``rumble``), or both."""

    airborne: bool
    rumble: bool
    facade: Facade = HOUSE
    building: str = BUILDING
    floor: int = FLOOR
    peak: str = PEAK
    line: vibration.Line = field(default_factory=vibration.Line)

    @classmethod
    def of(cls, choice: str | None, noun: str) -> House | None:
        """The house ``--indoors`` asks for (``choice``, one of :func:`choices`), or None
        for none."""
        off, both, alone, rumble = choices(noun)
        if choice in (None, off):
            return None
        if choice not in (both, alone, rumble):
            raise ValueError(f"--indoors is one of {', '.join(choices(noun))}, not {choice!r}")
        return cls(airborne=choice != rumble, rumble=choice != alone)

    @property
    def indoor_vdb(self) -> float:
        """How much more the house's floor shakes than the ground outside (dB)."""
        return vibration.indoor_db(self.building, self.floor)

    def through(self, passing):
        """``passing`` (a :class:`~acoustic_spaces.walkthrough.PassBy` outdoors) in the
        room: each band's level changed by the façade's
        :meth:`~acoustic_spaces.facade.Facade.indoor_db`."""
        from acoustic_spaces.parameters import a_weighted
        from acoustic_spaces.walkthrough import PassBy

        levels = np.asarray(passing.levels, dtype=float) + self.facade.indoor_db()
        levels_a = np.array([a_weighted(row) for row in levels])
        return PassBy(passing.source, passing.listener, passing.times, levels, levels_a)

    def heard(self, wet: np.ndarray, rate: int) -> np.ndarray:
        """``wet`` (samples, channels), the vehicle's sound outdoors, as the room hears
        it: through a zero-phase filter that changes each octave band's energy by the
        façade's gain (:func:`~acoustic_spaces.synthesis.matched_gains`)."""
        from acoustic_spaces.synthesis import equalized, matched_gains

        return equalized(wet, rate, matched_gains(self.facade.gains(), rate))

    def ground_vdb(self, wheels: Wheels, path, at: Sequence[float], times) -> np.ndarray:
        """(N,) the ground's vibration outside the house at ``at`` (a position) at each
        of ``times`` (s into the walk) as ``wheels`` go along ``path`` (the walk's
        :class:`~acoustic_spaces.walkthrough.Knots`, of the vehicle's front): the curve
        at the distance in plan of each wheel that is on the walk then (its front's
        place ``behind_m`` earlier, between the walk's start and end), with the line's
        adjustments and the wheel's offset, the greatest of them; -inf when none is.
        Nearer than 10 ft the curve holds, farther than 300 ft it falls on as it falls
        there (:meth:`~acoustic_spaces.vibration.Curve.level`)."""
        curve = vibration.CURVES[wheels.curve]
        times = np.asarray(times, dtype=float)
        speed = wheels.speed_kmh / 3.6
        positions = np.asarray(path.positions, dtype=float)
        knots = np.asarray(path.times, dtype=float)
        here = np.asarray(at, dtype=float)[:2]
        out = np.full(len(times), -np.inf)
        source = self.line.source_db()
        for behind, offset in zip(wheels.behind_m, wheels.offset_db, strict=True):
            when = times - behind / speed
            on = (when >= knots[0]) & (when <= knots[-1])
            if not np.any(on):
                continue
            x = np.interp(when[on], knots, positions[:, 0])
            y = np.interp(when[on], knots, positions[:, 1])
            distances = np.hypot(x - here[0], y - here[1])
            levels = np.array(
                [
                    curve.level(float(d), wheels.speed_kmh, extend=True)
                    + source
                    + self.line.path_db(float(d))
                    for d in distances
                ]
            )
            out[on] = np.maximum(out[on], levels + offset)
        return out

    def rumble_a_db(self, ground_vdb: np.ndarray) -> np.ndarray:
        """The rumble in the room (dB(A)) of the ground's vibration outside
        (``ground_vdb``, VdB): on the house's floor :attr:`indoor_vdb` more, and by
        table 6-14 for its peak."""
        return np.asarray(ground_vdb, dtype=float) + self.indoor_vdb + vibration.NOISE_DB[self.peak]

    def rumble_pass(self, source: str, listener: str, times, levels_a_db):
        """The rumble as a :class:`~acoustic_spaces.walkthrough.PassBy`, its A-weighted
        level ``levels_a_db`` at ``times``: its energy held in the lowest octave band,
        less that band's A-weighting, so that the parts' energies add band by band
        (:func:`~acoustic_spaces.demos.passby.heard_together`) to the right A-weighted
        level; its other bands silent. Only its A-weighted levels mean anything."""
        from acoustic_spaces.parameters import A_WEIGHTING_DB
        from acoustic_spaces.walkthrough import PassBy

        a = np.asarray(levels_a_db, dtype=float)
        levels = np.full((len(a), len(A_WEIGHTING_DB)), -np.inf)
        levels[:, 0] = a - A_WEIGHTING_DB[0]
        return PassBy(source, listener, np.asarray(times, dtype=float), levels, a)

    def rumble_sound(
        self, times, levels_a_db, rate: int, n: int, loudest_db: float, seed: int = 0
    ) -> np.ndarray:
        """(n,) the rumble's sound at ``rate``: noise in the third octave of its peak
        (:data:`~acoustic_spaces.vibration.RUMBLE_HZ`, :func:`third_octave_noise`), at
        the A-weighted level ``levels_a_db`` at ``times`` (s; linear in energy between
        them), in the mix's units against parts whose loudest has a sound power of
        ``loudest_db`` (:func:`mean_square`), faded in and out over FADE_S at its ends.
        Noise this narrow swells and fades at random by a decibel or so over a few
        seconds, so the whole is scaled to the A-weighted energy its levels give."""
        x = third_octave_noise(vibration.RUMBLE_HZ[self.peak], rate, n, seed)
        at = np.arange(n) / rate
        with np.errstate(divide="ignore"):
            energy = np.interp(at, times, 10.0 ** (np.asarray(levels_a_db) / 10.0))
        fade = min(int(FADE_S * rate), n // 2)
        ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
        envelope = np.sqrt(mean_square(energy, loudest_db))
        envelope[:fade] *= ramp
        envelope[n - fade :] *= ramp[::-1]
        x = x * envelope
        got = a_mean_square(x, rate)
        return x * math.sqrt(float(np.mean(envelope**2)) / got) if got > 0.0 else x

    def words(self) -> str:
        """What the stills say of the house, in lines short enough for them."""
        window = "through its closed window (EN ISO 12354-3)"
        rumble = "the rumble of its floors and walls (FTA)"
        if self.airborne and self.rumble:
            return f"In a house here, {window}\nand {rumble}"
        return f"In a house here, {window}" if self.airborne else f"In a house here: {rumble}"


@dataclass(frozen=True)
class Plant:
    """A source of a scene where nothing moves that shakes the ground: construction
    plant (the road works' breaker and dumper), its row of the FTA's table 7-4
    (:data:`~acoustic_spaces.vibration.PLANT`), where it stands."""

    source: str
    row: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class StillHouses:
    """A house at each listener of a scene where nothing moves (the road works'), as far
    from the sources as the listener, and what is heard in it (``house``, as
    ``--indoors`` asks, ``choice``): the scene's sound through the house's window, the
    rumble the ``plant``'s vibration makes in its room, or both. ``listeners`` are the
    listeners' positions, ``loudest_db`` the sound power of the loudest source that
    plays (dB re 1 pW), ``fmt`` the mixes' format.

    The plant's vibration at a house is each machine's level at its distance in plan
    (:func:`~acoustic_spaces.vibration.plant_level`, eq. 7-3), their energies added,
    and the manual compares plant's level with its limits as it is, with no adjustment
    for the building (section 7.2, step 4); its rumble is table 6-14's for vibration
    peaking below 30 Hz, as the trains'. A breaker's blows, 5 to 11 a second, come
    faster than noise in the third octave at 25 Hz, under 6 Hz wide, can swell and
    fade, so its rumble is steady."""

    house: House
    choice: str
    plant: tuple[Plant, ...]
    listeners: dict[str, tuple[float, float, float]]
    loudest_db: float
    fmt: object

    def ground_vdb(self, listener: str) -> float:
        """The ground's vibration at ``listener``'s house (VdB), -inf with no plant."""
        here = np.asarray(self.listeners[listener], dtype=float)[:2]
        energy = 0.0
        for machine in self.plant:
            distance = float(np.hypot(*(np.asarray(machine.position[:2]) - here)))
            energy += 10.0 ** (vibration.plant_level(machine.row, distance) / 10.0)
        return 10.0 * math.log10(energy) if energy > 0.0 else -math.inf

    def rumble_a_db(self, listener: str) -> float:
        """The rumble in ``listener``'s house (dB(A))."""
        return self.ground_vdb(listener) + vibration.NOISE_DB[self.house.peak]

    def info(self, listener: str) -> dict:
        """What ``listener``'s mix keeps of its house (``post``, how the mix was made;
        with the rumble, the vibration and its rumble)."""
        out: dict = {"post": f"indoors {self.choice}"}
        if self.house.rumble:
            out["vibration_vdb"] = round(self.ground_vdb(listener), 2)
            out["rumble_a_db"] = round(self.rumble_a_db(listener), 2)
        return out

    def post(self, mix, x: np.ndarray, rate: int) -> np.ndarray:
        """``x`` (samples, channels), ``mix``'s sound outdoors in the engine's units, as
        the house hears it: through its window, or none of it for the rumble alone; and
        the rumble, steady over the whole mix, faded in and out (:meth:`House.
        rumble_sound`), in every channel as a pressure (:func:`pressure_gains`)."""
        x = np.atleast_2d(np.asarray(x, dtype=float).T).T
        out = self.house.heard(x, rate) if self.house.airborne else np.zeros_like(x)
        if self.house.rumble:
            n = len(x)
            level = self.rumble_a_db(mix.listener)
            ends = [0.0, n / rate]
            names = sorted(self.listeners)
            seed = names.index(mix.listener) if mix.listener in names else 0
            sound = self.house.rumble_sound(ends, [level, level], rate, n, self.loudest_db, seed)
            out = out + sound[:, None] * pressure_gains(self.fmt)[None, :]
        return out

    def report(self, report: dict) -> dict:
        """``report`` (parameters.json) as heard in the houses: each response's levels
        per octave band through the window (:meth:`House.through`'s), its A-weighted
        level from them; none for the rumble alone."""
        from acoustic_spaces.parameters import a_weighted

        indoor = self.house.facade.indoor_db()
        responses = []
        for entry in report.get("responses", []):
            entry = dict(entry)
            bands = entry.get("spl_db")
            if not self.house.airborne:
                entry["spl_a_db"] = None
            elif bands is not None and all(v is not None for v in bands):
                levels = np.asarray(bands, dtype=float) + indoor
                entry["spl_db"] = [round(float(v), 2) for v in levels]
                entry["spl_a_db"] = round(a_weighted(levels), 2)
            responses.append(entry)
        return {**report, "responses": responses}

    def pictures(self) -> dict:
        """What the stills say of the houses (:class:`~acoustic_spaces.demos.
        scene_pictures.Pictures`'s ``indoors``)."""
        out: dict = {"airborne": self.house.airborne, "about": self.house.words()}
        if self.house.rumble:
            out["vibration_vdb"] = {name: self.ground_vdb(name) for name in self.listeners}
            out["rumble_a_db"] = {name: self.rumble_a_db(name) for name in self.listeners}
        return out


def third_octave_noise(centre_hz: float, rate: int, n: int, seed: int = 0) -> np.ndarray:
    """(n,) noise of one third octave, from 2^(-1/6) to 2^(1/6) of ``centre_hz``, flat in
    it and nothing outside, made in the frequency domain (so one period of it, without
    the edges a filter would leave), at an A-weighted mean square of 1."""
    f = np.fft.rfftfreq(n, 1.0 / rate)
    band = (f >= centre_hz / THIRD) & (f < centre_hz * THIRD)
    if not np.any(band):
        raise ValueError(f"{n} samples at {rate} Hz are too short for a band at {centre_hz:g} Hz")
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(len(f), dtype=complex)
    spectrum[band] = np.exp(2j * np.pi * rng.random(int(band.sum())))
    x = np.fft.irfft(spectrum, n)
    return x / math.sqrt(a_mean_square(x, rate))


def a_mean_square(x: np.ndarray, rate: int) -> float:
    """The A-weighted mean square of ``x`` (n,), one period of a periodic sound, from
    its spectrum (Parseval), so without the steps a filter would see at its ends."""
    from acoustic_spaces.demos.audio import a_gain

    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / rate)
    power = np.abs(np.fft.rfft(x) * a_gain(f)) ** 2
    power[1 : (n + 1) // 2] *= 2.0  # both sides of the spectrum, but 0 Hz and Nyquist
    return float(np.sum(power)) / n**2


def mean_square(energy, loudest_db: float):
    """The mean square in a mix of a sound pressure of ``energy`` (p² re (20 µPa)², a
    level's 10^(L/10)), its parts' loudest of sound power ``loudest_db`` (dB re 1 pW):
    SOUND_RMS² 4π energy / 10^(loudest_db / 10), as a part of that sound power fed at
    SOUND_RMS gives its direct sound at r metres the mean square SOUND_RMS² / r² and
    the level loudest_db - 10 lg(4 π r²)."""
    from acoustic_spaces.demos.road import SOUND_RMS

    return SOUND_RMS**2 * 4.0 * math.pi * np.asarray(energy) / 10.0 ** (loudest_db / 10.0)


def pressure_gains(fmt) -> np.ndarray:
    """(channels,) what each channel of the format ``fmt`` picks up of a sound pressure
    that is the same all over the room: all of it in each ear, a mono or W channel and
    a microphone's omnidirectional part (its gradient part none); a loudspeaker layout
    1/√n in each of its n main speakers, as it plays sound from straight above or
    below, and none in its LFE."""
    from acoustic_spaces.spatial.ambisonics import Ambisonics
    from acoustic_spaces.spatial.loudspeakers import Loudspeakers
    from acoustic_spaces.spatial.microphone import Microphone

    if isinstance(fmt, Ambisonics):
        out = np.zeros(fmt.channels)
        out[0] = 1.0
        return out
    if isinstance(fmt, Loudspeakers):
        mains = [not s.lfe for s in fmt.speakers]
        return np.where(mains, 1.0 / math.sqrt(sum(mains)), 0.0)
    if isinstance(fmt, Microphone):
        if fmt.a is None:
            return np.asarray(fmt.weights, dtype=float)
        steady, scaled = fmt.parts()
        low = float(fmt.a_in(8)[0])
        return np.asarray(steady.weights, dtype=float) + low * np.asarray(scaled.weights)
    return np.ones(fmt.channels)


def grid(end_s: float) -> np.ndarray:
    """The times the rumble's level is worked out at: from 0 to ``end_s``, STEP_S
    apart."""
    return np.linspace(0.0, end_s, max(int(math.ceil(end_s / STEP_S)) + 1, 2))


__all__ = [
    "BUILDING",
    "FLOOR",
    "OFF",
    "PEAK",
    "RUMBLE",
    "House",
    "Plant",
    "StillHouses",
    "Wheels",
    "a_mean_square",
    "choices",
    "grid",
    "help_of",
    "mean_square",
    "pressure_gains",
    "third_octave_noise",
]
