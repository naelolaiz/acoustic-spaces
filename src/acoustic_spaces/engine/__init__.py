"""Propagation: early specular paths, the late tail, and the responses they make.

The usual entry point is :func:`simulate`::

    from acoustic_spaces.engine import simulate

    result = simulate("scene.yaml")
    response = result["talker", "seat-a"]
    print(response.parameters().table())
    ir = response.impulse_response(48_000)

:func:`coverage_map` gives the same parameters on a grid over the audience area.
"""

from acoustic_spaces.engine.coverage import (
    CoverageMap,
    CoverageResult,
    MapGrid,
    coverage_map,
    map_grid,
)
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.late import LateEnergy
from acoustic_spaces.engine.model import DIRECTIVITY_A, TraceModel
from acoustic_spaces.engine.particles import Particles, trace_particles
from acoustic_spaces.engine.response import Response
from acoustic_spaces.engine.simulate import (
    TIME_STEP,
    SimulationResult,
    early_paths,
    mixing_time,
    simulate,
)

__all__ = [
    "DIRECTIVITY_A",
    "TIME_STEP",
    "CoverageMap",
    "CoverageResult",
    "EarlyReflections",
    "LateEnergy",
    "MapGrid",
    "Particles",
    "Response",
    "SimulationResult",
    "TraceModel",
    "coverage_map",
    "early_paths",
    "map_grid",
    "mixing_time",
    "simulate",
    "trace_particles",
]
