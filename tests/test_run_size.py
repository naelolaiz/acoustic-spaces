"""How big a run is (engine/run_size.py): the threads it works on, its memory, and the
runs refused or warned about for it."""

import os
from pathlib import Path

import numpy as np
import pytest
from test_tracer_reference import REFERENCE, reference_arguments

from acoustic_spaces import Room, _native, simulate
from acoustic_spaces.engine import run_size
from acoustic_spaces.engine.coverage import coverage_map
from acoustic_spaces.engine.gpu_tracer import run_memory
from acoustic_spaces.engine.late import band_groups
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.scene import Coverage

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
MiB = 2**20


def test_the_computers_memory_is_read() -> None:
    memory = run_size.machine_memory()
    assert memory is not None and memory > 256 * MiB


def test_threads_above_the_cores_run_as_the_cores(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert [run_size.worker_threads(t) for t in (0, 1, 4, 6, 7, 100)] == [6, 1, 4, 6, 6, 6]


def test_the_tracer_runs_threads_above_the_cores_as_the_cores() -> None:
    """The sum over threads depends on how many there are, so asking for more than the
    cores gives what the cores do only where the tracer takes no more than them."""
    with np.load(REFERENCE) as stored:
        values = reference_arguments(_native, dict(stored), "vector")
    cores = os.cpu_count() or 1
    found = _native.trace_stochastic(**(values | {"threads": cores}))
    for threads in (0, 3 * cores + 1):
        more = _native.trace_stochastic(**(values | {"threads": threads}))
        np.testing.assert_array_equal(more[0], found[0])
        np.testing.assert_array_equal(more[1], found[1])


def test_a_run_past_the_memory_is_refused_and_one_over_half_noted() -> None:
    need = run_size.Need(3 * 2**30, "a run", "lower this or that")
    with pytest.raises(ValueError) as refused:
        run_size.check(need, 2 * 2**30)
    assert str(refused.value) == (
        "a run needs about 3.0 GiB of memory, more than this computer's 2.0 GiB; lower this or that"
    )
    run_size.check(need, 4 * 2**30)
    assert run_size.warning(need, 4 * 2**30) == (
        "a run needs about 3.0 GiB of memory, over half of this computer's 4.0 GiB; "
        "lower this or that"
    )
    assert run_size.warning(need, 6 * 2**30) is None


def _call(results: float, threads: int, devices=("cpu",), groups: int = 1) -> float:
    return run_size.late_call(
        results, 8, groups, splitting=True, threads=threads, rays=10_000, devices=devices
    )


def test_the_tracers_memory_takes_the_computers_cores_and_devices(monkeypatch) -> None:
    """On the CPU, a copy per thread up to the cores, within one budget of copies; on a
    GPU or on both, the worst of the devices."""
    budget = _native.COPY_MEMORY
    small = 4 * MiB
    for cores in (2, 16, 128):
        monkeypatch.setattr(os, "cpu_count", lambda cores=cores: cores)
        on_cpu = [_call(small, threads) for threads in (1, cores, 4 * cores, 0)]
        # The results, the first batch's and a copy per thread.
        assert on_cpu[0] == 3 * small
        assert on_cpu[1:] == [(2 + min(cores, budget // small)) * small] * 3
        # Results larger than the budget: one copy, whatever the threads.
        large = 2 * budget
        assert _call(large, cores) == _call(large, 1) == 3 * large
        on_gpu = _call(small, cores, ("cpu", "gpu"))
        assert on_gpu == max(on_cpu[1], small + run_memory(10_000, 8, small))
        assert _call(small, cores, ("cpu", "gpu", "both")) >= on_gpu
    # Band by band: the joined arrays, and a band's own at a time.
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert _call(8 * small, 4, groups=8) == 8 * small + (2 + 4) * small


@pytest.mark.parametrize("method", ["vector", "random"])
def test_the_bands_are_grouped_as_the_tracer_groups_them(method) -> None:
    room = Room.from_yaml(SHOEBOX)
    settings = room.scene.simulation
    settings.scattering.method = method
    traced = band_groups(TraceModel.from_room(room, settings))
    assert run_size._band_groups(room, settings) == len(traced)


def test_a_simulation_too_large_for_the_memory_is_refused(monkeypatch) -> None:
    room = Room.from_yaml(SHOEBOX)
    monkeypatch.setattr(run_size, "machine_memory", lambda: MiB)
    with pytest.raises(ValueError, match=r"more than this computer's 1 MiB; lower simulation"):
        simulate(room)


def test_a_map_too_fine_for_the_memory_is_refused(monkeypatch) -> None:
    room = Room.from_yaml(SHOEBOX)
    monkeypatch.setattr(run_size, "machine_memory", lambda: 64 * MiB)
    refused = r"coverage map of about 4,800 points .*; raise coverage.spacing"
    with pytest.raises(ValueError, match=refused):
        coverage_map(room, settings=Coverage(spacing=0.1))


def test_the_editor_notes_a_run_over_half_the_memory(monkeypatch) -> None:
    room = Room.from_yaml(SHOEBOX)
    scene = room.scene
    sources = [s for s in scene.sources if s.in_use]
    listeners = [s for s in scene.listeners if s.in_use]
    need = run_size.simulation_need(room, scene.simulation, sources, listeners, 0)
    monkeypatch.setattr(run_size, "machine_memory", lambda: int(1.5 * need.bytes))
    notes = [w for w in room.warnings() if "of memory" in w]
    assert len(notes) == 1 and "over half of this computer's" in notes[0]
    monkeypatch.setattr(run_size, "machine_memory", lambda: int(3 * need.bytes))
    assert not [w for w in room.warnings() if "of memory" in w]
