"""The levels over a long term, outdoors: the scene run once in each weather class.

The weather changes from day to day, and the level over a year is not the level in one
weather: it is the energy mean of the levels in each weather, each weighted by the share
of the time it holds (ISO 1996-2, CNOSSOS-EU, Nord2000),

    L = 10 lg Σ p_k 10^(L_k / 10).

The scene's ``long_term`` (:class:`~acoustic_spaces.scene.LongTerm`) gives the weather
classes with their shares, and the periods of the day: by default the day (12 h), the
evening (4 h) and the night (8 h), each with its own shares of the classes, as the
inversions of a clear night do not come by day. The results give, at every listener
and every map point:

- the level in each class, per octave band and A-weighted;
- the level in each period, the mean over the classes with the period's shares;
- the long-term level, the mean over the periods by their hours (the day's L_eq), per
  octave band, and relative to free field;
- the rating level: each period's A-weighted level with its penalty added, 5 dB in the
  evening and 10 dB at night by default, averaged by the hours, which is L_den,

    L_den = 10 lg (1/24) (12 10^(L_day/10) + 4 10^((L_evening + 5)/10) + 8 10^((L_night + 10)/10)).

:func:`simulate` and :func:`coverage_map` are
:func:`~acoustic_spaces.engine.insertion.simulate` and
:func:`~acoustic_spaces.engine.insertion.coverage_map` (and so the plain ones) when the
scene has no ``long_term``. With one, the scene is run as it is, for its impulse
responses, and once in each weather class, all with the same seed, so that every run
traces the same rays; a class whose weather is the scene's own takes the scene's run. The
insertion loss (``simulation.insertion_loss_of``) is taken in every class, and the
long-term levels give it too: the long-term level without the parts less that with them.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces.engine import coverage as coverage_stage
from acoustic_spaces.engine import insertion
from acoustic_spaces.engine.simulate import SimulationResult, _as_room
from acoustic_spaces.parameters import A_WEIGHTING_DB
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Coverage, Scene
from acoustic_spaces.scene.model import Atmosphere, LongTerm, Simulation, WeatherClass

Key = tuple[str, str]


@dataclass
class LongTermLevels:
    """The levels over a long term at every (source, listener) of a run: the weather
    classes and the periods of the day, and the levels combined over them (dB SPL for
    the sources' sound power; ``_a`` are A-weighted)."""

    classes: tuple[str, ...]
    periods: tuple[str, ...]
    #: (classes, periods) the share of each period's time each class holds.
    shares: np.ndarray
    #: (periods,) hours a day, and the penalty (dB) each period gets in the rating.
    hours: np.ndarray
    penalties: np.ndarray
    #: The run in each class, by name; the scene's own run where the class's weather is
    #: the scene's.
    runs: dict[str, SimulationResult]
    #: (classes, bands) and (classes,) per (source, listener).
    class_levels: dict[Key, np.ndarray] = field(default_factory=dict)
    class_a: dict[Key, np.ndarray] = field(default_factory=dict)
    #: (periods, bands) and (periods,).
    period_levels: dict[Key, np.ndarray] = field(default_factory=dict)
    period_a: dict[Key, np.ndarray] = field(default_factory=dict)
    #: (bands,) the long-term level, and its A-weighted level.
    levels: dict[Key, np.ndarray] = field(default_factory=dict)
    levels_a: dict[Key, float] = field(default_factory=dict)
    #: The rating level (dB(A)): the periods' levels with their penalties.
    rating_level: dict[Key, float] = field(default_factory=dict)
    #: (bands,) the long-term level relative to free field, and A-weighted.
    re_free_field: dict[Key, np.ndarray] = field(default_factory=dict)
    re_free_field_a: dict[Key, float] = field(default_factory=dict)
    #: (bands,) the long-term insertion loss, and A-weighted, where the runs give one.
    insertion_loss: dict[Key, np.ndarray] = field(default_factory=dict)
    insertion_loss_a: dict[Key, float] = field(default_factory=dict)

    @property
    def rating(self) -> str:
        """What the rating level is called: ``L_den`` with the default periods, else
        ``L_R`` (a whole-day rating level of ISO 1996-1)."""
        return rating_name(self.periods, self.hours, self.penalties)


def rating_name(periods, hours, penalties) -> str:
    """``L_den`` for the day, evening and night of 12, 4 and 8 hours with penalties of
    0, 5 and 10 dB (Directive 2002/49/EC), else ``L_R``."""
    den = (
        tuple(periods) == ("day", "evening", "night")
        and np.allclose(hours, [12.0, 4.0, 8.0])
        and np.allclose(penalties, [0.0, 5.0, 10.0])
    )
    return "L_den" if den else "L_R"


# -- the arithmetic -------------------------------------------------------------------


def energy_mean(levels, weights, axis: int = 0) -> np.ndarray:
    """10 lg Σ w 10^(L/10) over ``axis`` of ``levels``, with ``weights`` (n,) along it:
    -inf where no sound comes in any weighted term, NaN where a weighted level is NaN.
    A weight of zero takes nothing from its level, be it NaN."""
    levels = np.asarray(levels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    shape = [1] * levels.ndim
    shape[axis] = -1
    weights = weights.reshape(shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        energy = np.where(weights == 0.0, 0.0, weights * 10.0 ** (levels / 10.0))
        return 10.0 * np.log10(energy.sum(axis=axis))


def period_levels(class_levels, shares) -> np.ndarray:
    """(periods, ...) the level in each period: the energy mean of ``class_levels``
    (classes, ...) with ``shares`` (classes, periods)."""
    shares = np.asarray(shares, dtype=float)
    return np.stack([energy_mean(class_levels, shares[:, p]) for p in range(shares.shape[1])])


def mean_level(levels, hours) -> np.ndarray:
    """(...) the long-term level: the energy mean of the periods' ``levels``
    (periods, ...) by their ``hours``."""
    hours = np.asarray(hours, dtype=float)
    return energy_mean(levels, hours / hours.sum())


def rating_level(levels_a, hours, penalties) -> np.ndarray:
    """(...) the rating level: the periods' A-weighted ``levels_a`` (periods, ...), each
    with its penalty added, averaged by the hours (L_den with 12/4/8 h and 0/5/10 dB)."""
    levels_a = np.asarray(levels_a, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    shape = [1] * levels_a.ndim
    shape[0] = -1
    return mean_level(levels_a + penalties.reshape(shape), hours)


def a_weighted(levels) -> np.ndarray:
    """(...) the A-weighted level of ``levels`` (..., bands) per octave band."""
    weighted = np.asarray(levels, dtype=float) + np.array(A_WEIGHTING_DB)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return 10.0 * np.log10(np.sum(10.0 ** (weighted / 10.0), axis=-1))


# -- the runs -------------------------------------------------------------------------


def _seeded(settings: Simulation) -> Simulation:
    """The settings with a seed of their own, so that every class's run traces the same
    rays."""
    if settings.seed is not None:
        return settings
    seed = int(np.random.SeedSequence().generate_state(1, np.uint32)[0])
    return settings.model_copy(update={"seed": seed})


def _weather_of(scene: Scene) -> tuple:
    """What tells one weather from another between runs: the atmosphere whole, and the
    temperature at the ground."""
    atmosphere = scene.atmosphere or Atmosphere()
    return atmosphere.model_dump(mode="json"), scene.air.temperature_c


def _spec(room: Room, long_term: LongTerm | None) -> LongTerm | None:
    spec = room.scene.long_term if long_term is None else long_term
    if spec is not None and room.outdoor is None:
        raise ValueError("the levels over a long term are for outdoor scenes (geometry.ground)")
    return spec


def _runs(room: Room, spec: LongTerm, say, run: Callable[[Room, str], object]) -> tuple:
    """``run(room, "")`` the scene as it is, then ``run`` it in every class whose weather
    is not the scene's own, which the classes with the scene's weather take: the
    scene's own result, the result in each class by name, and how many runs were
    made."""
    own = run(room, "")
    runs = {}
    extra = 0
    # The scene's own weather, written out as a class's is (a preset by its values).
    plain = _weather_of(room.scene.under(WeatherClass(share=1.0)))
    for name, given in zip(spec.class_names(), spec.classes, strict=True):
        scene = room.scene.under(given)
        if _weather_of(scene) == plain:
            runs[name] = own
            continue
        say(f"Long term: the scene in the weather class {name}")
        runs[name] = run(Room.from_scene(scene, reuse=room), name)
        extra += 1
    return own, runs, extra + 1


def combine(spec: LongTerm, runs: dict[str, SimulationResult], scene: Scene) -> LongTermLevels:
    """The levels over the long term from the run in each class (by name), for the
    scene's sources' sound power."""
    names = spec.class_names()
    shares, hours, penalties = spec.shares(), spec.hours(), spec.penalties()
    out = LongTermLevels(
        tuple(names), tuple(spec.period_names()), shares, hours, penalties, dict(runs)
    )
    sources = {s.name: s for s in scene.sources}
    for key in runs[names[0]].responses:
        power = sources[key[0]].power_db("octave") if key[0] in sources else 94.0
        responses = [runs[name].responses[key] for name in names]
        per_class = np.array([r.levels(power) for r in responses])  # (classes, bands)
        per_period = period_levels(per_class, shares)
        mean = mean_level(per_period, hours)
        period_a = a_weighted(per_period)
        out.class_levels[key], out.class_a[key] = per_class, a_weighted(per_class)
        out.period_levels[key], out.period_a[key] = per_period, period_a
        out.levels[key], out.levels_a[key] = mean, float(a_weighted(mean))
        out.rating_level[key] = float(rating_level(period_a, hours, penalties))
        if all(r.free_field is not None for r in responses):
            # The free field of each class (the air differs with the ground's
            # temperature), combined the same way.
            free = per_class - np.array([r.relative_to_free_field() for r in responses])
            free_mean = mean_level(period_levels(free, shares), hours)
            out.re_free_field[key] = mean - free_mean
            out.re_free_field_a[key] = float(a_weighted(mean) - a_weighted(free_mean))
        losses = [runs[name].insertion_loss.get(key) for name in names]
        if all(loss is not None for loss in losses):
            bare = mean_level(period_levels(per_class + np.array(losses), shares), hours)
            out.insertion_loss[key] = bare - mean
            out.insertion_loss_a[key] = float(a_weighted(bare) - a_weighted(mean))
    return out


def simulate(
    room: Room | Scene | str | Path,
    *,
    long_term: LongTerm | None = None,
    without: Sequence[str] | None = None,
    settings: Simulation | None = None,
    sources: Sequence[str] | None = None,
    listeners: Sequence[str] | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
    groups: np.ndarray | None = None,
) -> SimulationResult:
    """:func:`~acoustic_spaces.engine.insertion.simulate`, and the levels over a long
    term when the scene has weather classes (``long_term``, or ``long_term`` given
    here): the result is the scene's own weather's, with ``long_term`` holding the run
    in each class and the levels combined (:class:`LongTermLevels`). ``groups`` are for
    the run of the scene as it is."""
    room = _as_room(room)
    spec = _spec(room, long_term)
    common = {"without": without, "sources": sources, "listeners": listeners, "threads": threads}
    if spec is None:
        return insertion.simulate(
            room, settings=settings, progress=progress, groups=groups, **common
        )
    say = progress or (lambda _: None)
    started = time.perf_counter()
    settings = _seeded(settings or room.scene.simulation)
    names = spec.class_names()
    say(f"Long term: the scene as it is, then in the weather classes {', '.join(names)}")

    def run(where: Room, name: str) -> SimulationResult:
        return insertion.simulate(
            where,
            settings=settings,
            progress=progress,
            groups=groups if name == "" else None,
            **common,
        )

    own, runs, made = _runs(room, spec, say, run)
    levels = combine(spec, runs, room.scene)
    details = dict(own.details)
    details["long_term"] = {
        "classes": float(len(names)),
        "runs": float(made),
        "seconds": time.perf_counter() - started,
    }
    return dataclasses.replace(own, details=details, long_term=levels)


def coverage_map(
    room: Room | Scene | str | Path,
    *,
    long_term: LongTerm | None = None,
    without: Sequence[str] | None = None,
    settings: Coverage | None = None,
    simulation: Simulation | None = None,
    sources: Sequence[str] | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> coverage_stage.CoverageResult:
    """:func:`~acoustic_spaces.engine.insertion.coverage_map`, and the levels over a
    long term when the scene has weather classes: every map gains ``long_term`` (dB per
    octave band, the long-term level) and ``rating_level`` (dB(A): L_den with the default
    periods). The maps are the scene's own weather's, and every class's run shares
    their points."""
    room = _as_room(room)
    spec = _spec(room, long_term)
    common = {"without": without, "sources": sources, "threads": threads}
    if spec is None:
        return insertion.coverage_map(
            room, settings=settings, simulation=simulation, progress=progress, **common
        )
    say = progress or (lambda _: None)
    started = time.perf_counter()
    simulation = _seeded(simulation or room.scene.simulation)
    names = spec.class_names()
    say(f"Long term: the map as it is, then in the weather classes {', '.join(names)}")
    first: list[coverage_stage.CoverageResult] = []

    def run(where: Room, name: str) -> coverage_stage.CoverageResult:
        result = insertion.coverage_map(
            where,
            settings=first[0].settings if first else settings,
            simulation=simulation,
            progress=progress,
            grid=first[0].grid if first else None,
            **common,
        )
        if not first:
            first.append(result)
        return result

    own, runs, made = _runs(room, spec, say, run)
    shares, hours, penalties = spec.shares(), spec.hours(), spec.penalties()

    def with_levels(name: str | None) -> coverage_stage.CoverageMap:
        kept = own.combined if name is None else own.maps[name]
        maps = [runs[c].combined if name is None else runs[c].maps[name] for c in names]
        per_class = np.array([m.values["spl"] for m in maps])  # (classes, points, bands)
        per_period = period_levels(per_class, shares)
        values = {
            **kept.values,
            "long_term": mean_level(per_period, hours),
            "rating_level": rating_level(a_weighted(per_period), hours, penalties),
        }
        return dataclasses.replace(kept, values=values)

    maps = {name: with_levels(name) for name in own.maps}
    combined = with_levels(None) if own.combined is not None else None
    details = dict(own.details)
    details["long_term"] = {
        "classes": float(len(names)),
        "runs": float(made),
        "seconds": time.perf_counter() - started,
    }
    return dataclasses.replace(own, maps=maps, combined=combined, details=details)


__all__ = [
    "LongTermLevels",
    "a_weighted",
    "combine",
    "coverage_map",
    "energy_mean",
    "mean_level",
    "period_levels",
    "rating_level",
    "rating_name",
    "simulate",
]
