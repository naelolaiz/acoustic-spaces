"""The insertion loss of an outdoor scene's buildings, barriers and vegetation.

The scene is run as it is and again without them (:meth:`~acoustic_spaces.scene.Scene.
without`), with the same settings and seed, so that the two runs' rays start alike;
the insertion loss is the level without them less the level with them, per octave band
(and A-weighted, for the sound power of each source). A barrier that takes 10 dB off has
an insertion loss of 10 dB; a building that sends sound back towards the listener can
have a negative one.

``simulation.insertion_loss_of`` names what to take away; :func:`simulate` and
:func:`coverage_map` read it unless given ``without``, and are :func:`~acoustic_spaces.
engine.simulate` and :func:`~acoustic_spaces.engine.coverage.coverage_map` when it names
nothing.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from acoustic_spaces.engine import coverage as coverage_stage
from acoustic_spaces.engine.response import Response
from acoustic_spaces.engine.simulate import SimulationResult, _as_room
from acoustic_spaces.engine.simulate import simulate as simulate_stage
from acoustic_spaces.parameters import A_WEIGHTING_DB, a_weighted
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Coverage, Scene
from acoustic_spaces.scene.model import Simulation


def _names(room: Room, settings: Simulation, without: Sequence[str] | None) -> list[str]:
    names = list(settings.insertion_loss_of or []) if without is None else list(without)
    if names and room.outdoor is None:
        raise ValueError("an insertion loss is for outdoor scenes (geometry.ground)")
    return names


def _seeded(settings: Simulation) -> Simulation:
    """The settings with a seed of their own, so that both runs trace the same rays, and
    without ``insertion_loss_of``."""
    seed = settings.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, np.uint32)[0])
    return settings.model_copy(update={"seed": seed, "insertion_loss_of": None})


def bare_room(room: Room, names: Sequence[str]) -> Room:
    """The room without the buildings, barriers and vegetation of these names."""
    return Room.from_scene(room.scene.without(names))


def _loss(bare: Response, kept: Response) -> np.ndarray:
    """The insertion loss per octave band (dB) from the energy each response holds:
    nothing is extrapolated past their ends, as G does for a room's decay. Infinite where
    only one of them holds any, NaN where neither does."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(bare.octave_energy().sum(axis=1) / kept.octave_energy().sum(axis=1))


def simulate(
    room: Room | Scene | str | Path,
    *,
    without: Sequence[str] | None = None,
    settings: Simulation | None = None,
    sources: Sequence[str] | None = None,
    listeners: Sequence[str] | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
    groups: np.ndarray | None = None,
) -> SimulationResult:
    """:func:`~acoustic_spaces.engine.simulate`, and the insertion loss of the
    buildings, barriers and vegetation named in ``without`` (default:
    ``simulation.insertion_loss_of``) at every listener: the result is the scene's as it
    is, with ``removed`` and ``insertion_loss`` (dB per octave band, the level without
    them less the level with them; NaN where either level is undefined). ``groups`` are
    for the run of the scene as it is."""
    room = _as_room(room)
    settings = settings or room.scene.simulation
    names = _names(room, settings, without)
    if not names:
        return simulate_stage(
            room,
            settings=settings,
            sources=sources,
            listeners=listeners,
            threads=threads,
            progress=progress,
            groups=groups,
        )
    say = progress or (lambda _: None)
    started = time.perf_counter()
    settings = _seeded(settings)
    say(f"Insertion loss: the scene as it is, then without {', '.join(names)}")
    with_parts = simulate_stage(
        room,
        settings=settings,
        sources=sources,
        listeners=listeners,
        threads=threads,
        progress=progress,
        groups=groups,
    )
    bare = bare_room(room, names)
    say(f"Insertion loss: the scene without {', '.join(names)}")
    without_parts = simulate_stage(
        bare,
        settings=settings,
        sources=sources,
        listeners=listeners,
        threads=threads,
        progress=progress,
    )
    loss = {
        key: _loss(without_parts.responses[key], response)
        for key, response in with_parts.responses.items()
    }
    details = dict(with_parts.details)
    details["insertion_loss"] = {
        "parts": float(len(names)),
        "seconds": time.perf_counter() - started,
    }
    return dataclasses.replace(
        with_parts, details=details, removed=tuple(names), insertion_loss=loss
    )


def coverage_map(
    room: Room | Scene | str | Path,
    *,
    without: Sequence[str] | None = None,
    settings: Coverage | None = None,
    simulation: Simulation | None = None,
    sources: Sequence[str] | None = None,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> coverage_stage.CoverageResult:
    """:func:`~acoustic_spaces.engine.coverage.coverage_map`, and the insertion loss of
    the buildings, barriers and vegetation named in ``without`` (default:
    ``simulation.insertion_loss_of``) at every map point: each map gains
    ``insertion_loss`` (dB per octave band) and ``insertion_loss_a`` (dB(A), for the
    sources' sound power), the level without them less the level with them. Both runs
    share the map's points, those of the scene as it is."""
    room = _as_room(room)
    simulation = simulation or room.scene.simulation
    names = _names(room, simulation, without)
    run = coverage_stage.coverage_map
    if not names:
        return run(
            room,
            settings=settings,
            simulation=simulation,
            sources=sources,
            threads=threads,
            progress=progress,
        )
    say = progress or (lambda _: None)
    started = time.perf_counter()
    simulation = _seeded(simulation)
    say(f"Insertion loss: the map as it is, then without {', '.join(names)}")
    with_parts = run(
        room,
        settings=settings,
        simulation=simulation,
        sources=sources,
        threads=threads,
        progress=progress,
    )
    say(f"Insertion loss: the map without {', '.join(names)}")
    without_parts = run(
        bare_room(room, names),
        settings=with_parts.settings,
        simulation=simulation,
        sources=sources,
        threads=threads,
        progress=progress,
        grid=with_parts.grid,
    )

    def with_loss(
        kept: coverage_stage.CoverageMap, bare: coverage_stage.CoverageMap
    ) -> coverage_stage.CoverageMap:
        # Both maps hold the same free field at the same points, so the difference of
        # their levels relative to it is that of the energies they hold.
        loss = bare.values["re_free_field"] - kept.values["re_free_field"]
        spl = kept.values["spl"]
        values = {
            **kept.values,
            "insertion_loss": loss,
            "insertion_loss_a": _a_weighted_rows(spl + loss) - _a_weighted_rows(spl),
        }
        return dataclasses.replace(kept, values=values)

    maps = {
        name: with_loss(kept, without_parts.maps[name]) for name, kept in with_parts.maps.items()
    }
    combined = with_parts.combined
    if combined is not None:
        combined = with_loss(combined, without_parts.combined)
    details = dict(with_parts.details)
    details["insertion_loss"] = {
        "parts": float(len(names)),
        "seconds": time.perf_counter() - started,
    }
    return dataclasses.replace(
        with_parts, maps=maps, combined=combined, details=details, removed=tuple(names)
    )


def a_weighted_loss(spl: np.ndarray, loss: np.ndarray) -> float:
    """The A-weighted insertion loss (dB) from the levels with the parts per octave band
    (dB SPL) and their insertion loss per band: the A-weighted level without them less
    that with them."""
    spl = np.asarray(spl, dtype=float)
    return float(a_weighted(spl + np.asarray(loss, dtype=float)) - a_weighted(spl))


def _a_weighted_rows(levels: np.ndarray) -> np.ndarray:
    """:func:`~acoustic_spaces.parameters.a_weighted` of every row of (points, bands)."""
    weighted = np.asarray(levels, dtype=float) + np.array(A_WEIGHTING_DB)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.sum(10.0 ** (weighted / 10.0), axis=1))


__all__ = ["a_weighted_loss", "bare_room", "coverage_map", "simulate"]
