"""The uncertainty of a coverage map from its materials' uncertain absorption.

A listener's spread is a Monte Carlo over the absorption that re-weights its traced
response for each draw (:mod:`~acoustic_spaces.engine.uncertainty`). A map keeps no
response at its points to re-weight (no count of the reflections off each material, no
batches of rays), so its spread comes from the law of propagation of uncertainty
instead (GUM, JCGM 100:2008, 5.1.2), with the sensitivities found numerically (5.1.3,
note 2). The inputs are those of the listeners' draws: one deviate per material, which
its bands share, the materials independent of one another. For each material the map is
traced again twice, with its absorption where a deviate of plus and of minus one
standard deviation puts it
(:func:`~acoustic_spaces.engine.uncertainty.absorption_nodes`: its own plus and minus
its standard uncertainty, held to a draw's limits), on the same points and with the
same seed, so that the runs trace the same rays. Half the difference of the two maps is
that material's part of the uncertainty of each value at each point, and the root of
the sum of the parts' squares the value's combined standard uncertainty
(:class:`~acoustic_spaces.engine.coverage.MapSpread`). A map with M such materials is
run 2 M + 1 times.

The runs are whole map runs, in the weather classes and without the parts an insertion
loss names where the map has those (:func:`~acoustic_spaces.engine.long_term.
coverage_map`), so every value a map holds gets its uncertainty.

It is first order: where a value bends over a material's uncertainty, as a decay time
does in a room with little absorption for a tolerance of 0.1, the standard uncertainty
comes out low. In a box of 8 by 6 by 3.5 m with three such tolerances, against the
Monte Carlo of 2 000 draws of a listener at one of the map's points: the A-weighted
level's 4 % low, the STI's 3 %, a band's level's up to 8 %, T30's up to 13 % and EDT's
and Ts's up to a quarter (at 63 Hz); with standard uncertainties of 0.02 to 0.03, all
within 7 %. A material whose absorption lies within its uncertainty of 0, or of the
most a draw takes, has its draws held there, which the two runs follow in part only.
Not counted: the rays' random error (between two seeds of the lecture hall example's
map a material's part differs by 2 % of its size for the A-weighted level and 11 % for
T30, root mean square over the points), and what a listener's spread does not count
either (the scattering, the method, a ground's absorption uncertainty beside its
impedance). The interval and the one-sided bounds take the result as normal
(:meth:`~acoustic_spaces.engine.coverage.CoverageMap.spread_of`), where a listener's
are its draws' quantiles.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from acoustic_spaces.engine import coverage as coverage_stage
from acoustic_spaces.engine import long_term as long_term_stage
from acoustic_spaces.engine.reweight import reweighted_materials, with_absorption
from acoustic_spaces.engine.simulate import _as_room
from acoustic_spaces.engine.uncertainty import absorption_nodes, assumed, held, uncertainty_of
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Coverage, Scene
from acoustic_spaces.scene.model import LongTerm, Simulation

#: What the source of a material reads in a run of the spread.
_SOURCE = "absorption moved by its standard uncertainty, for a map's spread"


def spread_materials(room: Room) -> list[str]:
    """The materials whose absorption a map's spread moves: those on ``room``'s
    surfaces that have an uncertainty a spread takes
    (:func:`~acoustic_spaces.engine.uncertainty.uncertainty_of`) and absorb less than
    everything in some band."""
    return [n for n in reweighted_materials(room) if uncertainty_of(room.scene, n) is not None]


def coverage_map(
    room: Room | Scene | str | Path,
    *,
    spread: bool | None = None,
    long_term: LongTerm | None = None,
    without: Sequence[str] | None = None,
    settings: Coverage | None = None,
    simulation: Simulation | None = None,
    sources: Sequence[str] | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> coverage_stage.CoverageResult:
    """:func:`~acoustic_spaces.engine.long_term.coverage_map`, and with ``spread``
    (default: the map settings' ``spread``) the uncertainty of every value of every
    map from the materials' absorption uncertainty (:attr:`~acoustic_spaces.engine.
    coverage.CoverageMap.spread`): the map is run twice more for each material of
    :func:`spread_materials`, on the same points with the same seed. The one-sided
    bounds' probability is ``simulation.uncertainty.one_sided``. A map none of whose
    materials has an uncertainty is run once and has no spread."""
    room = _as_room(room)
    scene = room.scene
    settings = settings or scene.coverage or Coverage()
    wanted = settings.spread if spread is None else spread
    materials = spread_materials(room) if wanted else []
    # One seed for every run, so that they trace the same rays.
    simulation = simulation or scene.simulation
    if materials:
        simulation = long_term_stage._seeded(simulation)

    def run(
        of: Room, settings: Coverage, grid: coverage_stage.MapGrid | None = None
    ) -> coverage_stage.CoverageResult:
        return long_term_stage.coverage_map(
            of,
            long_term=long_term,
            without=without,
            settings=settings,
            simulation=simulation,
            sources=sources,
            threads=threads,
            progress=progress,
            grid=grid,
        )

    if not materials:
        return run(room, settings)
    say = progress or (lambda _: None)
    started = time.perf_counter()
    runs = 2 * len(materials)
    say(
        f"Spread: the map as it is, then {runs} times more, with the absorption of each of "
        f"{', '.join(materials)} one standard uncertainty up and down"
    )
    own = run(room, settings)
    keys = [*own.maps, None]
    parts: dict[str | None, dict[str, dict[str, np.ndarray]]] = {key: {} for key in keys}
    for k, name in enumerate(materials):
        moved = []
        for n, absorption in enumerate(absorption_nodes(scene, name)):
            say(
                f"Spread: the map with the absorption of {name} one standard uncertainty "
                f"{'down' if n else 'up'} (run {2 * k + n + 1} of {runs})"
            )
            other = with_absorption(scene, [name], [absorption], _SOURCE, digits=None)
            moved.append(run(Room.from_scene(other, reuse=room), own.settings, own.grid))
        for key in keys:
            up, down = (_map(result, key) for result in moved)
            if up is not None and down is not None:
                parts[key][name] = {
                    value: 0.5 * (_finite(up.values[value]) - _finite(down.values[value]))
                    for value in up.values
                }

    def with_spread(kept: coverage_stage.CoverageMap, key: str | None):
        each = parts[key]
        standard = {}
        for value, at in kept.values.items():
            squares = sum(part[value] ** 2 for part in each.values())
            standard[value] = np.where(np.isfinite(at), np.sqrt(squares), np.nan)
        found = coverage_stage.MapSpread(
            standard=standard,
            materials=each,
            one_sided=simulation.uncertainty.one_sided,
            assumed=assumed(scene, materials),
            held=held(scene, materials),
        )
        return dataclasses.replace(kept, spread=found)

    details = dict(own.details)
    details["spread"] = {
        "materials": float(len(materials)),
        "runs": float(runs),
        "seconds": time.perf_counter() - started,
    }
    return dataclasses.replace(
        own,
        maps={name: with_spread(kept, name) for name, kept in own.maps.items()},
        combined=None if own.combined is None else with_spread(own.combined, None),
        details=details,
    )


def _map(result: coverage_stage.CoverageResult, key: str | None):
    """The map of source ``key`` of ``result``, or that of all its sources together."""
    return result.combined if key is None else result.maps[key]


def _finite(values: np.ndarray) -> np.ndarray:
    """``values`` with NaN where they are undefined or infinite (a band without sound)."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


__all__ = ["coverage_map", "spread_materials"]
