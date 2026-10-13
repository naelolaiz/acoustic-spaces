"""Propagation: early specular paths, the late tail, and the responses they make.

The usual entry point is :func:`simulate`::

    from acoustic_spaces.engine import simulate

    result = simulate("scene.yaml")
    response = result["talker", "seat-a"]
    print(response.parameters().table())
    ir = response.impulse_response(48_000)
"""

from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.late import LateEnergy
from acoustic_spaces.engine.model import DIRECTIVITY_A, TraceModel
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
    "EarlyReflections",
    "LateEnergy",
    "Response",
    "SimulationResult",
    "TraceModel",
    "early_paths",
    "mixing_time",
    "simulate",
]
