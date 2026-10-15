"""The wave solver stepped on a GPU through WebGPU, against the CPU.

The comparisons run wherever wgpu finds a GPU, software ones included (CI uses Mesa's
lavapipe on Linux), and are skipped elsewhere.
"""

from pathlib import Path

import numpy as np
import pytest
from test_wave import STENCILS, _courant, _drive, _solver

from acoustic_spaces import Room, _native
from acoustic_spaces.engine import gpu
from acoustic_spaces.engine.impedance import fit_admittance
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.wave import COURANT, LAYER_CELLS, LAYER_DAMPING, wave_responses
from acoustic_spaces.scene.model import WaveStage

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
C = 343.0
needs_gpu = pytest.mark.skipif(gpu.gpu_name() is None, reason="wgpu finds no GPU")


def _runs(make, signal, parts):
    """The probes' pressures from a CPU solver and from a GPU one on the same grid, both
    run in the pieces of ``signal`` that ``parts`` bounds."""
    cpu, gpu_solver = make(), gpu.GpuWaveSolver(make())
    pieces = list(zip(parts[:-1], parts[1:], strict=True))
    a = np.concatenate([cpu.run(signal[lo:hi]) for lo, hi in pieces], axis=1)
    b = np.concatenate([gpu_solver.run(signal[lo:hi]) for lo, hi in pieces], axis=1)
    return a, b


@needs_gpu
@STENCILS
def test_gpu_steps_a_room_with_resonant_walls_as_the_cpu_does(stencil: int) -> None:
    cell = 0.1
    rate = C / (_courant(stencil) * cell)
    fit = fit_admittance([0.1, 0.3, 0.6], 8, rate)
    assert len(fit.conductance) > 0

    def make():
        solver, _ = _solver(
            (3.0, 2.2, 1.7),
            lambda mesh: [0.02] * len(mesh.faces),
            [0.8, 0.9, 0.7],
            [[2.1, 1.3, 1.1], [1.0, 1.6, 0.4]],
            cell=cell,
            branches=[(0, fit), (3, fit), (5, fit)],
            stencil=stencil,
        )
        return solver

    assert make().branch_states > 0
    # Pieces of odd and even lengths, so the fields swap both ways between runs.
    a, b = _runs(make, _drive(rate, cell, 1601), [0, 377, 1000, 1601])
    assert np.abs(a).max() > 0.0
    assert np.abs(a - b).max() < 1e-5 * np.abs(a).max()


@needs_gpu
def test_gpu_steps_the_matched_layer_round_an_open_model_as_the_cpu_does() -> None:
    h = 0.3
    vertices = np.array([[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]], float)
    bvh = _native.Bvh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.int64))
    source = np.array([2.0, 2.0, 1.5])
    listeners = np.array([[4.0, 3.0, 1.2], [1.0, 1.0, 0.6]])
    points = np.vstack([source, listeners])
    low = np.minimum(vertices.min(0), points.min(0) - 2 * h) - h * (2.0 + LAYER_CELLS)
    high = np.maximum(vertices.max(0), points.max(0) + 2 * h)
    shape = np.ceil((high - low) / h).astype(int) + 2 + LAYER_CELLS

    def make():
        return _native.WaveSolver(
            bvh,
            np.zeros(2, np.int64),
            np.array([0.05]),
            np.zeros((0, 3)),
            np.zeros(0, np.int64),
            low + np.array([0.0137, 0.0219, 0.0291]) * h,
            h,
            *map(int, shape),
            source,
            listeners,
            layer=LAYER_CELLS,
            layer_damping=LAYER_DAMPING,
        )

    assert make().layer_cells > 0
    rate = C / (COURANT * h)
    a, b = _runs(make, _drive(rate, h, 600), [0, 301, 600])
    assert np.abs(a - b).max() < 1e-5 * np.abs(a).max()


@needs_gpu
def test_wave_responses_on_the_gpu_match_the_cpu() -> None:
    room = Room.from_yaml(SHOEBOX)
    model = TraceModel.from_room(room)
    source = np.array(room.scene.sources[0].position)
    listener = np.array(room.scene.listeners[0].position)
    common = {"max_band_hz": 63, "length_s": 0.3, "sample_rate": 48_000, "speed_of_sound": C}
    said: list[str] = []
    cpu, _ = wave_responses(model, source, [listener], ["l"], **common)
    on_gpu, info = wave_responses(
        model, source, [listener], ["l"], device="gpu", say=said.append, **common
    )
    assert info["wave_gpu"] == 1.0
    assert gpu.gpu_name() in said[0]
    a, b = cpu[0].signals, on_gpu[0].signals
    assert np.abs(a - b).max() < 1e-5 * np.abs(a).max()


def test_the_device_setting_picks_the_cpu_or_the_gpu(monkeypatch) -> None:
    assert WaveStage().device == "cpu"
    assert not gpu.use_gpu("cpu")
    with pytest.raises(ValueError, match="cpu, gpu or auto"):
        gpu.use_gpu("tpu")
    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    assert not gpu.use_gpu("auto")
    assert gpu.gpu_name() is None
    with pytest.raises(ValueError, match="simulation.wave.device is gpu"):
        gpu.use_gpu("gpu")
