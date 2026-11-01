"""A pass-by heard inside the houses of a scene (``--indoors``): at each listener that
stands in front of a building's wall, 0.5 to 2 m from it (:func:`facing`, the positions
ISO 1996-2 counts as in front of a reflecting façade), the train's sound through the
house's closed window and the rumble the ground's vibration makes in its room, or
either alone. Every other listener hears the pass outdoors.

The window: each part of the train is simulated and played at the listener in front of
the house as the pass-by demo plays it (:mod:`acoustic_spaces.demos.passby`), the
façade's reflection simulated with the rest, then brought indoors band by band by the
façade's level difference (:data:`acoustic_spaces.facade.HOUSE`, EN ISO 12354-3), from
the level 2 m in front of it to the room: its levels per octave band, and its sound
through a zero-phase filter that changes each octave band's energy by as much
(:func:`~acoustic_spaces.synthesis.matched_gains`). The sound keeps the direction it
comes from outdoors; the room's own reverberation is not added.

The rumble: by the FTA's general method (:mod:`acoustic_spaces.vibration`), the ground
under the room shakes at each moment as the train's base curve says at the distance in
plan from its nearest wheels to the room's wall (the point of the house's wall nearest
the listener): when it runs past, the manual's level for that
wall's distance from the track; as it comes and goes, less. The house, a small masonry
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

What is felt: outdoors the ground shakes as the curve says at the listener; people
feel it from about 65 VdB (:data:`~acoustic_spaces.vibration.PERCEPTION_VDB`) but do
not hear it, the manual's ground-borne noise being the rumble of a building's floors
and walls. In a house the floor shakes. :func:`shaker_sound` makes that vibration a
track for a tactile transducer (a bass shaker): noise in the rumble's third octave at
the vibration's velocity at each moment.
"""

from __future__ import annotations

import math
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
#: Where a listener stands in front of a building's wall (m from it, in plan): ISO
#: 1996-2:2017's positions in front of a reflecting façade, 0.5 to 2 m, where its
#: reflection adds to the sound.
FACADE_M = (0.5, 2.0)
#: What the walls' distance may be off by for a listener to stand in front of one (m).
FACADE_TOLERANCE_M = 1e-6


def choices(noun: str) -> tuple[str, ...]:
    """What ``--indoors`` takes for a vehicle called ``noun``: off; the vehicle and the
    rumble; the vehicle through the window alone; the rumble alone."""
    return (OFF, f"{noun}_and_{RUMBLE}", noun, RUMBLE)


def help_of(noun: str) -> str:
    """The help of ``--indoors`` for a vehicle called ``noun`` (or the road works)."""
    off, both, alone, rumble = choices(noun)
    return (
        f"where the {noun} is heard: {off}, outdoors at each listener; or, at each "
        "listener 0.5 to 2 m in front of a building's wall, inside it: "
        f"{both}, the {noun} through the house's closed window (4-16-4 double glazing, "
        "EN ISO 12354-3) and the rumble its ground-borne vibration makes in the room "
        f"(the FTA's general method); {alone}, through the window alone; {rumble}, the "
        "rumble alone; the other listeners hear it outdoors"
    )


@dataclass(frozen=True)
class Facing:
    """The house a listener stands in front of: the scene's building ``name``, the
    point of its wall nearest the listener in plan (m), the room's, and how far the
    listener is from it (m)."""

    name: str
    wall: tuple[float, float]
    distance_m: float

    def distances(self, points) -> np.ndarray:
        """(n,) how far each of ``points`` (n, 2 or 3) is in plan from the room's wall
        (m), where the ground's vibration from there reaches the room. A long
        building's nearest corner may be far nearer, as a row of houses' end is to works
        along its street, but the room is behind the listener."""
        p = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
        return np.linalg.norm(p - np.asarray(self.wall), axis=1)


def facing(scene, position) -> Facing | None:
    """The building of the outdoor ``scene`` whose wall a listener at ``position``
    stands in front of: FACADE_M from it in plan, square in front of it (not off a
    corner, where ISO 1996-2's façade position does not hold), outside its footprint
    and lower than its roof; the nearest such, or None."""
    from acoustic_spaces.geometry.io import UNIT_SCALE
    from acoustic_spaces.geometry.outdoor import in_polygon

    geometry = scene.geometry
    if geometry.ground is None:
        return None
    scale = UNIT_SCALE[geometry.units or "m"]
    here = np.asarray(position, dtype=float)
    ground = scale * geometry.ground.height
    terrain = geometry.ground.height_grid()
    if terrain is not None:
        ground = float(terrain.scaled(scale, ground).height_at(here[None, :2])[0])
    found: Facing | None = None
    least, most = FACADE_M
    for building in geometry.all_buildings():
        footprint = scale * np.asarray(building.footprint, dtype=float)
        if in_polygon(here[None, :2], footprint)[0] or here[2] - ground >= scale * building.height:
            continue
        wall = vibration.nearest_points(here[None], footprint, closed=True)[0]
        if np.min(np.linalg.norm(footprint[:, :2] - wall, axis=1)) <= FACADE_TOLERANCE_M:
            continue  # off a corner: in front of no wall
        distance = float(np.linalg.norm(wall - here[:2]))
        near = least - FACADE_TOLERANCE_M <= distance <= most + FACADE_TOLERANCE_M
        if near and (found is None or distance < found.distance_m):
            found = Facing(str(building.name), (float(wall[0]), float(wall[1])), distance)
    return found


def facings(scene, names) -> dict[str, Facing | None]:
    """The house each of the listeners ``names`` of ``scene`` stands in front of
    (:func:`facing`), or None."""
    where = {lis.name: lis.position for lis in scene.listeners}
    return {name: facing(scene, where[name]) for name in names}


def houses_heard(scene, names, house: House | None, noun: str, scene_name: str) -> dict:
    """:func:`facings` of the listeners ``names`` where ``--indoors`` hears the ``noun``
    in a house (``house``), none outdoors; SystemExit when no listener stands in front
    of a house."""
    if house is None:
        return {}
    found = facings(scene, names)
    if not any(found.values()):
        raise SystemExit(
            f"error: --indoors hears the {noun} in a house that a listener stands "
            f"{FACADE_M[0]:g} to {FACADE_M[1]:g} m in front of; no listener of "
            f"{scene_name} does"
        )
    return found


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

    def ground_vdb(self, wheels: Wheels, path, where, times) -> np.ndarray:
        """:func:`ground_vdb` with the house's line."""
        return ground_vdb(wheels, path, where, times, self.line)

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
        with np.errstate(divide="ignore"):
            energy = 10.0 ** (np.asarray(levels_a_db, dtype=float) / 10.0)
        return enveloped(x, times, mean_square(energy, loudest_db), rate, a_mean_square)

    def words(self, name: str = "house") -> str:
        """What the stills say of the house (the building ``name``) at a listener in
        front of it, in lines short enough for them."""
        window = "through a closed window (EN ISO 12354-3)"
        rumble = "the rumble of the floors and walls (FTA)"
        inside = f"Inside the {name.replace('_', ' ')} here"
        if self.airborne and self.rumble:
            return f"{inside}, {window}\nand {rumble}"
        return f"{inside}, {window}" if self.airborne else f"{inside}: {rumble}"


def ground_vdb(
    wheels: Wheels, path, where, times, line: vibration.Line | None = None
) -> np.ndarray:
    """(N,) the ground's vibration at ``where`` (a position, or the house a listener
    stands in front of, :class:`Facing`) at each of ``times`` (s into the walk) as
    ``wheels`` go along ``path`` (the walk's :class:`~acoustic_spaces.walkthrough.Knots`,
    of the vehicle's front): the curve at the distance in plan of each wheel that is on
    the walk then (its front's place ``behind_m`` earlier, between the walk's start and
    end) from the position, or from the room's wall, with the ``line``'s
    adjustments (none by default) and the wheel's offset, the greatest of them; -inf
    when none is. Nearer than 10 ft the curve holds, farther than 300 ft it falls on as
    it falls there (:meth:`~acoustic_spaces.vibration.Curve.level`)."""
    line = vibration.Line() if line is None else line
    curve = vibration.CURVES[wheels.curve]
    times = np.asarray(times, dtype=float)
    speed = wheels.speed_kmh / 3.6
    positions = np.asarray(path.positions, dtype=float)
    knots = np.asarray(path.times, dtype=float)
    out = np.full(len(times), -np.inf)
    source = line.source_db()
    for behind, offset in zip(wheels.behind_m, wheels.offset_db, strict=True):
        when = times - behind / speed
        on = (when >= knots[0]) & (when <= knots[-1])
        if not np.any(on):
            continue
        x = np.interp(when[on], knots, positions[:, 0])
        y = np.interp(when[on], knots, positions[:, 1])
        if isinstance(where, Facing):
            distances = where.distances(np.column_stack([x, y]))
        else:
            here = np.asarray(where, dtype=float)[:2]
            distances = np.hypot(x - here[0], y - here[1])
        levels = curve.levels(distances, wheels.speed_kmh) + source + line.path_db(distances)
        out[on] = np.maximum(out[on], levels + offset)
    return out


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
    """The listeners of a scene where nothing moves (the road works'), each in a house
    or outdoors: at a listener in front of a building's wall (``inside``, the house
    each stands in front of, :func:`facing`), what ``--indoors`` asks (``house`` and its
    ``choice``; None: every listener outdoors): the scene's sound through the house's
    window, the rumble the ``plant``'s vibration makes in its room, or both; at the
    others the scene outdoors. ``listeners`` are the listeners' positions, ``loudest_db``
    the sound power of the loudest source that plays (dB re 1 pW), ``fmt`` the mixes'
    format.

    The plant's vibration at a house is each machine's level at its distance in plan
    from the room's wall behind the listener (:func:`~acoustic_spaces.vibration.plant_level`, eq.
    7-3), their energies added, and the manual compares plant's level with its limits
    as it is, with no adjustment for the building (section 7.2, step 4); its rumble is
    table 6-14's for vibration peaking below 30 Hz, as the trains'. Outdoors the ground
    shakes as each machine's level at its distance from the listener says. A breaker's
    blows, 5 to 11 a second, come faster than noise in the third octave at 25 Hz, under
    6 Hz wide, can swell and fade, so its rumble, and what is felt, is steady."""

    house: House | None
    choice: str
    plant: tuple[Plant, ...]
    listeners: dict[str, tuple[float, float, float]]
    inside: dict[str, Facing | None]
    loudest_db: float
    fmt: object

    def house_of(self, listener: str) -> Facing | None:
        """The house ``listener`` is heard in, or None outdoors."""
        return None if self.house is None else self.inside.get(listener)

    def ground_vdb(self, listener: str) -> float:
        """The ground's vibration at ``listener``'s house, or under it outdoors (VdB);
        -inf with no plant."""
        where = self.house_of(listener)
        here = np.asarray(self.listeners[listener], dtype=float)[:2]
        energy = 0.0
        for machine in self.plant:
            at = np.asarray(machine.position[:2], dtype=float)
            if where is None:
                distance = float(np.hypot(*(at - here)))
            else:
                distance = float(where.distances(at[None])[0])
            # Nearer than 10 ft, where the trains' curves start, the level holds.
            distance = max(distance, vibration.NEAREST_M)
            energy += 10.0 ** (vibration.plant_level(machine.row, distance) / 10.0)
        return 10.0 * math.log10(energy) if energy > 0.0 else -math.inf

    def rumble_a_db(self, listener: str) -> float:
        """The rumble in ``listener``'s house (dB(A))."""
        peak = PEAK if self.house is None else self.house.peak
        return self.ground_vdb(listener) + vibration.NOISE_DB[peak]

    def info(self, listener: str) -> dict:
        """What ``listener``'s mix keeps of its house (``post``, how the mix was made;
        with the rumble, the vibration and its rumble); nothing outdoors."""
        if self.house_of(listener) is None:
            return {}
        out: dict = {"post": f"indoors {self.choice}"}
        if self.house is not None and self.house.rumble:
            out["vibration_vdb"] = round(self.ground_vdb(listener), 2)
            out["rumble_a_db"] = round(self.rumble_a_db(listener), 2)
        return out

    def post(self, mix, x: np.ndarray, rate: int) -> np.ndarray:
        """``x`` (samples, channels), ``mix``'s sound at its listener in the engine's
        units, as the house it stands in front of hears it: through its window, or none
        of it for the rumble alone; and the rumble, steady over the whole mix, faded in
        and out (:meth:`House.rumble_sound`), in every channel as a pressure
        (:func:`pressure_gains`). Outdoors ``x`` as it is."""
        if self.house is None or self.house_of(mix.listener) is None:
            return x
        x = np.atleast_2d(np.asarray(x, dtype=float).T).T
        out = self.house.heard(x, rate) if self.house.airborne else np.zeros_like(x)
        if self.house.rumble:
            n = len(x)
            level = self.rumble_a_db(mix.listener)
            ends = [0.0, n / rate]
            sound = self.house.rumble_sound(
                ends, [level, level], rate, n, self.loudest_db, self.seed(mix.listener)
            )
            out = out + sound[:, None] * pressure_gains(self.fmt)[None, :]
        return out

    def seed(self, listener: str) -> int:
        """The seed of ``listener``'s noise: its place among the listeners' names."""
        names = sorted(self.listeners)
        return names.index(listener) if listener in names else 0

    def shaker(self, listener: str, rate: int, n: int) -> np.ndarray:
        """(n,) what is felt at ``listener`` over ``n`` samples at ``rate``, steady, as
        :func:`shaker_sound` makes it (m/s); silence with no plant."""
        level = self.ground_vdb(listener)
        return shaker_sound([0.0, n / rate], [level, level], rate, n, PEAK, self.seed(listener))

    def report(self, report: dict) -> dict:
        """``report`` (parameters.json) as heard where each listener is: the responses
        at a listener in a house with their levels per octave band through the window
        (:meth:`House.through`'s), the A-weighted level from them, none for the rumble
        alone; the others as they are."""
        from acoustic_spaces.parameters import a_weighted

        if self.house is None:
            return report
        indoor = self.house.facade.indoor_db()
        responses = []
        for entry in report.get("responses", []):
            entry = dict(entry)
            bands = entry.get("spl_db")
            if self.house_of(str(entry.get("listener"))) is None:
                pass
            elif not self.house.airborne:
                entry["spl_a_db"] = None
            elif bands is not None and all(v is not None for v in bands):
                levels = np.asarray(bands, dtype=float) + indoor
                entry["spl_db"] = [round(float(v), 2) for v in levels]
                entry["spl_a_db"] = round(a_weighted(levels), 2)
            else:
                # A band unknown outdoors leaves the level inside unknown too.
                entry["spl_db"] = None
                entry["spl_a_db"] = None
            responses.append(entry)
        return {**report, "responses": responses}

    def pictures(self) -> dict:
        """What the stills say of each listener (:class:`~acoustic_spaces.demos.
        scene_pictures.Pictures`'s ``indoors``): for those in a house, what is heard
        there (``airborne``, ``about``) and its rumble; for every listener, how hard the
        ground or the house's floor shakes."""
        inside = [name for name in self.listeners if self.house_of(name) is not None]
        out: dict = {"airborne": self.house is None or self.house.airborne, "about": {}}
        if self.house is not None:
            for name in inside:
                facing = self.house_of(name)
                assert facing is not None
                out["about"][name] = self.house.words(facing.name)
        if self.plant:
            out["vibration_vdb"] = {name: self.ground_vdb(name) for name in self.listeners}
        if self.house is not None and self.house.rumble:
            out["rumble_a_db"] = {name: self.rumble_a_db(name) for name in inside}
        return out


#: The name of what is felt among a listener's files: ``<listener>.shaker.wav``.
SHAKER = "shaker"


def write_shakers(folder, shakers: dict[str, np.ndarray], rate: int) -> dict:
    """``<listener>.shaker.wav`` in ``folder`` for each of ``shakers`` (what is felt
    there, m/s; :func:`shaker_sound`), one gain for all, the loudest peak at -1 dBFS;
    what their full scale is (a velocity) and the third octave they are in, for the
    demo's JSON; nothing without shakers, or with nothing felt. As with dBFS, a sample
    of 1 is ``full_scale_mm_s``, and a track's RMS level in dBFS plus
    ``full_scale_vdb`` is its vibration level in VdB (an RMS level)."""
    from acoustic_spaces.demos.audio import write

    peak = max((float(np.abs(x).max(initial=0.0)) for x in shakers.values()), default=0.0)
    if peak <= 0.0:
        return {}
    full = peak / 10.0 ** (-1.0 / 20.0)
    for name, x in shakers.items():
        write(folder / f"{name}.{SHAKER}.wav", np.asarray(x) / full, rate)
    return {
        "full_scale_mm_s": round(full * 1e3, 6),
        "full_scale_vdb": round(20.0 * math.log10(full / vibration.VELOCITY_REFERENCE_M_S), 2),
        "hz": vibration.RUMBLE_HZ[PEAK],
    }


def tour_shaker(
    work, stops, folder, seats: dict, shaker: dict, crossfade_s: float, fade_out_s: float
) -> None:
    """``tour.shaker.wav`` in ``work``: what is felt along the tour, each stop's
    ``<listener>.shaker.wav`` (in ``folder``) as the tour plays its mix
    (:func:`~acoustic_spaces.demos.pipeline.tour_audio`), at their full scale
    (``shaker``, :func:`write_shakers`'s)."""
    from acoustic_spaces.demos import pipeline
    from acoustic_spaces.demos.audio import write

    x = pipeline.tour_audio(
        stops, folder, seats, crossfade_s=crossfade_s, fade_out_s=fade_out_s, kind=SHAKER
    )
    write(work / f"tour.{SHAKER}.wav", x[:, 0], seats["rate"])
    print(
        f"wrote {work / f'tour.{SHAKER}.wav'}: what is felt, for a bass shaker, "
        f"{shaker['full_scale_mm_s']:.3g} mm/s at full scale"
    )


def felt(vdb: float) -> str:
    """Whether people feel a vibration of ``vdb`` (VdB), in words: from
    :data:`~acoustic_spaces.vibration.PERCEPTION_VDB`."""
    return "felt" if vdb >= vibration.PERCEPTION_VDB else "too little to feel"


def shaking_line(vdb: float, floor: bool, rumble_a_db: float | None = None) -> str:
    """The stills' line of a vibration of ``vdb`` (VdB) at most, of a house's ``floor``
    or of the ground, and of its rumble (dB(A)) when it is heard."""
    where = "Floor" if floor else "Ground"
    line = f"{where} shaking {vdb:.0f} VdB at most, {felt(vdb)}"
    return line if rumble_a_db is None else f"{line}; its rumble {rumble_a_db:.1f} dB(A)"


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


def enveloped(x: np.ndarray, times, mean_squares, rate: int, measure) -> np.ndarray:
    """``x`` (n,), noise of a ``measure`` (a mean square of samples at ``rate``) of 1, at
    the mean squares ``mean_squares`` at ``times`` (s; linear between them), faded in and
    out over FADE_S at its ends. Noise this narrow swells and fades at random by a
    decibel or so over a few seconds, so the whole is scaled to the energy its envelope
    gives."""
    n = len(x)
    at = np.arange(n) / rate
    envelope = np.sqrt(np.interp(at, np.asarray(times, dtype=float), np.asarray(mean_squares)))
    fade = min(int(FADE_S * rate), n // 2)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 2
    envelope[:fade] *= ramp
    envelope[n - fade :] *= ramp[::-1]
    x = x * envelope
    got = measure(x, rate)
    return x * math.sqrt(float(np.mean(envelope**2)) / got) if got > 0.0 else x


def shaker_sound(times, levels_vdb, rate: int, n: int, peak: str = PEAK, seed: int = 0):
    """(n,) what is felt at a listener, for a tactile transducer (a bass shaker): noise
    in the third octave the vibration peaks in (:data:`~acoustic_spaces.vibration.
    RUMBLE_HZ`, :func:`third_octave_noise`), its RMS the velocity (m/s) of the vibration
    levels ``levels_vdb`` (VdB re 1 micro-inch/s) at ``times`` (s; linear in energy
    between them), faded in and out over FADE_S at its ends (:func:`enveloped`)."""
    x = third_octave_noise(vibration.RUMBLE_HZ[peak], rate, n, seed)
    x = x / math.sqrt(float(np.mean(x**2)))
    with np.errstate(divide="ignore"):
        energy = vibration.VELOCITY_REFERENCE_M_S**2 * 10.0 ** (
            np.asarray(levels_vdb, dtype=float) / 10.0
        )
    return enveloped(x, times, energy, rate, plain_mean_square)


def plain_mean_square(x: np.ndarray, rate: int) -> float:
    """The mean square of ``x`` (n,), unweighted."""
    return float(np.mean(np.asarray(x, dtype=float) ** 2))


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
    "FACADE_M",
    "FACADE_TOLERANCE_M",
    "FLOOR",
    "OFF",
    "PEAK",
    "RUMBLE",
    "SHAKER",
    "Facing",
    "House",
    "Plant",
    "StillHouses",
    "Wheels",
    "a_mean_square",
    "choices",
    "enveloped",
    "facing",
    "facings",
    "felt",
    "grid",
    "ground_vdb",
    "help_of",
    "mean_square",
    "plain_mean_square",
    "pressure_gains",
    "shaker_sound",
    "shaking_line",
    "third_octave_noise",
    "tour_shaker",
    "write_shakers",
]
