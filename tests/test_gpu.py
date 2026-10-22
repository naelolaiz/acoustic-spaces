"""The wave solver stepped, and rays searched and traced, on a GPU through WebGPU, against
the CPU.

The comparisons run wherever wgpu finds a GPU, software ones included (CI uses Mesa's
lavapipe on Linux), and are skipped elsewhere.
"""

import platform
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from test_native import _stochastic_args, _unit_cube, cube_with_seats
from test_tracer_reference import REFERENCE, reference_arguments
from test_vegetation import _big_box, _slab
from test_wave import STENCILS, _courant, _drive, _solver, _speeds, _warming

from acoustic_spaces import Room, _native
from acoustic_spaces.engine import gpu, gpu_rays, gpu_tracer, late, ray_device
from acoustic_spaces.engine.impedance import fit_admittance
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.wave import COURANT, LAYER_CELLS, LAYER_DAMPING, wave_responses
from acoustic_spaces.scene.model import WaveStage

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
OPEN_AIR = Path(__file__).resolve().parent.parent / "examples" / "open_air_stage" / "scene.yaml"
C = 343.0
needs_gpu = pytest.mark.skipif(gpu.gpu_name() is None, reason="wgpu finds no GPU")


def _tolerance(software: float, hardware: float) -> float:
    """The share of the peak a GPU's pressures may stray from the CPU's: a software
    renderer rounds much as the CPU does, a hardware GPU less so (its fused multiply-adds
    and divisions round otherwise), and GitHub's Intel Mac runners stray the most."""
    intel_mac = sys.platform == "darwin" and platform.machine() == "x86_64"
    return hardware if intel_mac or gpu.hardware_gpu() else software


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
    # GPUs round differently from the CPU: in this nearly closed room the slow drift the
    # walls' branches leave picks that up, by the end 1.1e-5 of the peak on Apple's GPU
    # and 7.9e-4 on an AMD Radeon 680M and on the GPU of GitHub's Intel Mac runners (the
    # same on every run), so hardware GPUs get the wider margin. A field swapped the
    # wrong way between runs would differ by far more than either.
    assert np.abs(a - b).max() < _tolerance(3e-5, 2e-3) * np.abs(a).max()


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
    assert np.abs(a - b).max() < _tolerance(1e-5, 1e-4) * np.abs(a).max()


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
    assert np.abs(a - b).max() < _tolerance(1e-5, 1e-4) * np.abs(a).max()


@needs_gpu
@STENCILS
def test_gpu_steps_a_speed_of_sound_per_cell_as_the_cpu_does(stencil: int) -> None:
    """A room whose speed of sound falls from the grid's at the ceiling to 0.6 of it at
    the floor: the interior, the walls (with their branches) and the source take the
    cells' scale as the CPU does."""
    cell = 0.1
    rate = C / (_courant(stencil) * cell)
    fit = fit_admittance([0.1, 0.3, 0.6], 8, rate)

    def make():
        solver, _ = _solver(
            (3.0, 2.2, 1.7),
            lambda mesh: [0.02] * len(mesh.faces),
            [0.8, 0.9, 0.7],
            [[2.1, 1.3, 1.1], [1.0, 1.6, 0.4]],
            cell=cell,
            branches=[(0, fit), (3, fit)],
            stencil=stencil,
            speed=lambda centres: np.clip(0.6 + 0.4 * centres[:, 2] / 1.7, 0.6, 1.0),
        )
        return solver

    assert make().layout()["scaled"][0] == 1
    a, b = _runs(make, _drive(rate, cell, 1201), [0, 577, 1201])
    assert np.abs(a).max() > 0.0
    assert np.abs(a - b).max() < _tolerance(3e-5, 2e-3) * np.abs(a).max()


@needs_gpu
def test_gpu_steps_the_matched_layer_with_a_speed_of_sound_per_cell_as_the_cpu_does() -> None:
    """A plate in the open, the speed of sound half the grid's at the bottom of the grid
    and all of it 2.5 m up: the matched layer's push takes each cell's scale as the CPU
    does."""
    h = 0.3
    vertices = np.array([[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]], float)
    bvh = _native.Bvh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.int64))
    source = np.array([2.0, 2.0, 1.5])
    listeners = np.array([[4.0, 3.0, 1.2], [1.0, 1.0, 0.6]])
    points = np.vstack([source, listeners])
    low = np.minimum(vertices.min(0), points.min(0) - 2 * h) - h * (2.0 + LAYER_CELLS)
    high = np.maximum(vertices.max(0), points.max(0) + 2 * h)
    shape = np.ceil((high - low) / h).astype(int) + 2 + LAYER_CELLS
    origin = low + np.array([0.0137, 0.0219, 0.0291]) * h

    def speed(centres):
        return np.clip(0.5 + 0.5 * (centres[:, 2] - low[2]) / 2.5, 0.5, 1.0)

    def make():
        return _native.WaveSolver(
            bvh,
            np.zeros(2, np.int64),
            np.array([0.05]),
            np.zeros((0, 3)),
            np.zeros(0, np.int64),
            origin,
            h,
            *map(int, shape),
            source,
            listeners,
            layer=LAYER_CELLS,
            layer_damping=LAYER_DAMPING,
            speed=_speeds(speed, origin, h, shape),
        )

    solver = make()
    assert solver.layer_cells > 0 and solver.layout()["scaled"][0] == 1
    rate = C / (COURANT * h)
    a, b = _runs(make, _drive(rate, h, 600), [0, 301, 600])
    assert np.abs(a - b).max() < _tolerance(1e-5, 1e-4) * np.abs(a).max()


@needs_gpu
def test_wave_responses_in_wind_and_temperature_run_on_the_gpu() -> None:
    """Air warming with height (the speed of sound from 337 to 402 m/s): the GPU steps
    the grid, with its speed of sound per cell, as the CPU does."""
    room = _warming([[0, 10, 0], [60, 130, 0]])
    model = TraceModel.from_room(room)
    source = np.array([0.0, 0.0, 30.0])
    listener = np.array([60.0, 0.0, 30.0])
    common = {
        "max_band_hz": 63,
        "length_s": 0.25,
        "sample_rate": 48_000,
        "speed_of_sound": room.speed_of_sound(),
    }
    said: list[str] = []
    cpu, _ = wave_responses(model, source, [listener], ["l"], **common)
    on_gpu, info = wave_responses(
        model, source, [listener], ["l"], device="gpu", say=said.append, **common
    )
    assert info["wave_gpu"] == 1.0
    assert info["wave_speed_min_m_s"] == pytest.approx(337.3, abs=0.1)
    assert gpu.gpu_name() in said[0] and "sound at 337 to 402 m/s" in said[0]
    a, b = cpu[0].signals, on_gpu[0].signals
    assert np.abs(a - b).max() < _tolerance(1e-5, 1e-4) * np.abs(a).max()


@needs_gpu
def test_gpu_steps_in_submissions_short_enough_for_the_gpus_watchdog(monkeypatch) -> None:
    """A big grid's steps go to the GPU a few at a time, since a GPU whose job runs too
    long (10 s on Linux's amdgpu) is reset and loses every step after; the pieces step
    the grid exactly as one long submission does."""
    cell = 0.1
    rate = C / (_courant(7) * cell)

    def make():
        solver, _ = _solver(
            (3.0, 2.2, 1.7),
            lambda mesh: [0.02] * len(mesh.faces),
            [0.8, 0.9, 0.7],
            [[2.1, 1.3, 1.1], [1.0, 1.6, 0.4]],
            cell=cell,
        )
        return solver

    signal = _drive(rate, cell, 700)
    solvers = gpu.GpuWaveSolver(make()), gpu.GpuWaveSolver(make())
    queue = type(solvers[0]._device.queue)
    submit = queue.submit
    submits: list[int] = []

    def counted(self, buffers):
        submits.append(1)
        return submit(self, buffers)

    monkeypatch.setattr(queue, "submit", counted)
    whole = solvers[0].run(signal)
    once = len(submits)
    monkeypatch.setattr(gpu, "_SUBMIT_CELL_STEPS", 37 * solvers[1]._cells)
    split = solvers[1].run(signal)
    # 700 steps: a batch of 512 and one of 188, sent whole or 37 steps at a time.
    assert len(submits) - once == once + -(-512 // 37) + -(-188 // 37) - 2
    np.testing.assert_array_equal(split, whole)


def test_the_device_setting_picks_the_cpu_or_the_gpu(monkeypatch) -> None:
    assert WaveStage().device is None  # this computer's default
    assert not gpu.use_gpu("cpu")
    with pytest.raises(ValueError, match="cpu, gpu or auto"):
        gpu.use_gpu("tpu")
    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    assert not gpu.use_gpu("auto")
    assert gpu.gpu_name() is None
    with pytest.raises(ValueError, match="simulation.wave.device is gpu"):
        gpu.use_gpu("gpu")


def test_wgpu_leaves_its_opengl_backend_out(monkeypatch) -> None:
    """Vulkan, Metal or Direct3D 12 only: setting up wgpu's OpenGL backend through EGL
    aborts the process when the editor's 3D view holds an OpenGL context."""
    asked = []
    extras = types.ModuleType("wgpu.backends.wgpu_native.extras")
    extras.set_instance_extras = lambda backends: asked.append(backends)
    for name in ("wgpu", "wgpu.backends", "wgpu.backends.wgpu_native"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, extras.__name__, extras)
    gpu._primary_backends()
    assert asked == [["Primary"]]

    def too_late(backends):
        raise RuntimeError("Instance already exists")

    extras.set_instance_extras = too_late
    gpu._primary_backends()  # wgpu keeps the instance it has


@needs_gpu
def test_gpu_ray_search_finds_what_the_cpu_does() -> None:
    rng = np.random.default_rng(6)
    vertices, faces = cube_with_seats(rng)
    # Off the origin, where single precision has fewer digits to spare.
    vertices = vertices * 20.0 + np.array([100.0, -40.0, 3.0])
    bvh = _native.Bvh(vertices, faces)
    on_gpu = gpu_rays.GpuBvh(bvh)
    n = 4000
    origins = vertices[:8].min(axis=0) + rng.uniform(0.02, 0.98, (n, 3)) * 20.0
    directions = rng.standard_normal((n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    none = np.full(n, -1)
    t, face = bvh.intersect(origins, directions, np.full(n, np.inf), none)
    t_gpu, face_gpu = on_gpu.intersect(origins, directions, np.inf, none)
    assert np.all(face_gpu >= 0)  # inside a closed box
    same = face_gpu == face
    assert same.mean() > 0.995
    np.testing.assert_allclose(t_gpu[same], t[same], rtol=1e-4, atol=1e-5 * on_gpu.scale)
    # From where those rays hit, skipping that face, on to the next one and towards a
    # point: the queries of a tracer's reflections and of its diffuse rain.
    starts = origins + directions * t[:, None]
    bounce = -directions
    t, face_next = bvh.intersect(starts, bounce, np.full(n, np.inf), face)
    t_gpu, face_gpu = on_gpu.intersect(starts, bounce, np.inf, face)
    assert np.mean(face_gpu == face_next) > 0.995
    targets = origins[::-1]
    distance = np.linalg.norm(targets - starts, axis=1)
    towards = (targets - starts) / distance[:, None]
    blocked = bvh.occluded(starts, towards, distance, face)
    blocked_gpu = on_gpu.occluded(starts, towards, distance, face)
    assert 0 < blocked.sum() < n
    assert np.mean(blocked_gpu == blocked) > 0.995
    # A ray limited short of every face finds none.
    t_gpu, face_gpu = on_gpu.intersect(origins[:5], directions[:5], 1e-3, -1)
    assert np.all(face_gpu == -1) and np.all(np.isinf(t_gpu))


@needs_gpu
def test_gpu_ray_search_meets_the_faces_main_found() -> None:
    with np.load(REFERENCE) as stored:
        data = dict(stored)
    bvh = _native.Bvh(data["hall_vertices"], data["hall_faces"])
    on_gpu = gpu_rays.GpuBvh(bvh)
    origins, directions = (data[f"rays_{k}"].astype(float) for k in ("origins", "directions"))
    t, face = on_gpu.intersect(origins, directions, data["rays_t_max"], data["rays_ignore"])
    same = face == data["rays_face"]
    assert same.mean() > 0.99
    hit = same & (face >= 0)
    assert hit.sum() > 0.8 * len(face)
    np.testing.assert_allclose(t[hit], data["rays_t"][hit], rtol=1e-4, atol=1e-5 * on_gpu.scale)


@needs_gpu
def test_the_ray_search_timing_script_runs(monkeypatch, capsys) -> None:
    script = Path(__file__).resolve().parent.parent / "validation" / "rays_gpu.py"
    argv = ["rays_gpu.py", "--scenes", str(SHOEBOX), "--rays", "3000"]
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert "3,000 reflections and 3,000 shadow tests per scene" in out
    row = next(line for line in out.splitlines() if line.startswith("| shoebox"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert len(cells) == 9 and cells[6].endswith("x")
    # The same face and the same shadow for nearly every ray.
    assert float(cells[7].rstrip("%")) > 99.0 and float(cells[8].rstrip("%")) > 99.0


# -- the late stage's tracer ------------------------------------------------------------

#: Stretches of the response (ms) the late energy is compared over.
STRETCHES = ((0, 20), (20, 50), (50, 100), (100, 200), (200, 400))
#: Bounds on the differences from main in units of the CPU's spread: the CPU itself, on
#: other seeds, reaches 6.6 at most and 0.4 on average over a run (the spread, from
#: eight seeds, is rough and the few rays through a small listener are far from normal).
Z_MAX = 10.0
Z_MEAN = 1.0


def _reference() -> dict[str, np.ndarray]:
    with np.load(REFERENCE) as stored:
        return dict(stored)


def _stretches(energy: np.ndarray) -> np.ndarray:
    """Energy per listener, band and stretch of the response."""
    return np.stack([energy[..., lo:hi].sum(-1) for lo, hi in STRETCHES], -1)


@needs_gpu
def test_gpu_tracer_meets_the_direct_sound_as_the_cpu_does() -> None:
    """In a room too large for any ray to reach a wall within the response, all either
    counts is the rays from the source through the listeners: the same rays, with the
    same directivity weights, on both."""
    data = _reference()
    values = reference_arguments(_native, data, "random_detector")
    vertices, faces = _unit_cube()
    values |= {
        "directions": data["directions"],
        "bvh": _native.Bvh(vertices * 120.0 - 55.0, faces),
        "absorption": np.full((len(faces), 3), 0.2),
        "scattering": np.full((len(faces), 3), 0.3),
        "receiver_radius": np.full(3, 1.0),
        "rays": 100_000,
        "max_time": 0.12,
    }
    on_cpu = _native.trace_stochastic(**values)
    on_gpu = gpu_tracer.trace_stochastic(**values)
    energy = on_cpu[0]
    assert np.all(energy.sum((1, 2)) > 0)  # through each listener's sphere
    np.testing.assert_allclose(on_gpu[0], energy, rtol=1e-4, atol=1e-6 * energy.max())
    np.testing.assert_allclose(on_gpu[1], on_cpu[1], rtol=1e-4, atol=1e-6 * on_cpu[1].max())
    assert on_gpu[6] == on_cpu[6] == values["rays"]
    assert on_gpu[3] == on_cpu[3] == 0  # out of time, not out of the room


@needs_gpu
def test_gpu_tracer_counts_the_rays_that_escape_as_the_cpu_does() -> None:
    """A 20 m box without a ceiling: as many rays get out on both, within the noise."""
    values = reference_arguments(_native, _reference(), "random_detector")
    vertices, faces = _unit_cube()
    top = np.all(vertices[faces][:, :, 2] == 1.0, axis=1)
    walls = len(faces) - int(top.sum())
    values |= {
        "bvh": _native.Bvh(vertices * 20.0 - (4.0, 5.0, 0.0), faces[~top]),
        "absorption": np.full((walls, 3), 0.1),
        "scattering": np.full((walls, 3), 0.5),
        "rays": 20_000,
    }
    rays = values["rays"]
    on_cpu = _native.trace_stochastic(**values)[3] / rays
    on_gpu = gpu_tracer.trace_stochastic(**values)[3] / rays
    assert 0.2 < on_cpu < 0.95
    # Four times the spread of the difference between two runs.
    assert abs(on_gpu - on_cpu) < 4 * np.sqrt(2 * on_cpu * (1 - on_cpu) / rays)


@needs_gpu
def test_gpu_tracer_sums_the_reflections_of_many_rays() -> None:
    """A 10 cm box with hard walls: each ray meets them some 250 times in the weight
    window's first 50 ms, 12 million additions to that window's sums in all, more than
    its fixed point held when each could bring as much as one ray's whole energy."""
    values = reference_arguments(_native, _reference(), "random_detector")
    vertices, faces = _unit_cube()
    values |= {
        "bvh": _native.Bvh(vertices * 0.1, faces),
        "source": np.array([0.05, 0.05, 0.05]),
        "directivity_a": np.zeros(0),
        "directivity_table": np.zeros((0, 0, 3)),
        "absorption": np.zeros((len(faces), 3)),
        "scattering": np.full((len(faces), 3), 0.5),
        "receivers": np.array([[0.03, 0.065, 0.04]]),
        "receiver_radius": np.full(1, 0.01),
        "covered_time": np.zeros(0),
        "rays": 50_000,
        "max_time": 0.05,
    }
    on_cpu = _native.trace_stochastic(**values)
    on_gpu = gpu_tracer.trace_stochastic(**values)
    assert on_cpu[5][0] > 200 * values["rays"]
    np.testing.assert_allclose(on_gpu[4], on_cpu[4], rtol=0.01)  # the rays' energy
    np.testing.assert_allclose(on_gpu[5], on_cpu[5], rtol=0.01)  # and reflections


@needs_gpu
@pytest.mark.parametrize("name", ["vector", "vector_split", "random_detector", "secondary"])
def test_gpu_tracer_agrees_with_main_within_the_noise(name: str) -> None:
    """The late energy traced on the GPU, with four times the rays, against main's
    stored result of the same run: the difference per listener, band and stretch of the
    response in units of the spread of the CPU's result over seeds."""
    data = _reference()
    values = reference_arguments(_native, data, name)
    expected = _stretches(data[f"late_{name}_energy"])
    spread = np.std(
        [_stretches(_native.trace_stochastic(**values | {"seed": 1000 + s})[0]) for s in range(8)],
        axis=0,
        ddof=1,
    )
    more = 4
    values |= {"rays": more * values["rays"], "split_target": values["split_target"] / more}
    found = _stretches(gpu_tracer.trace_stochastic(**values)[0])
    noise = np.maximum(spread * np.sqrt(1 + 1 / more), 1e-4 * np.abs(expected))
    z = (found - expected) / np.maximum(noise, 1e-300)
    assert np.abs(z).max() < Z_MAX, np.round(z, 2)
    assert abs(z.mean()) < Z_MEAN, np.round(z, 2)


@needs_gpu
def test_the_late_stage_traces_where_its_device_says() -> None:
    room = Room.from_yaml(SHOEBOX)
    model = TraceModel.from_room(room, room.scene.simulation)
    source = np.array(room.scene.sources[0].position, float)
    listeners = np.array([listener.position for listener in room.scene.listeners], float)
    common = {
        "rays": 6000,
        "max_time": 0.5,
        "time_step": 1e-3,
        "speed_of_sound": C,
        "air": room.air_attenuation(model.bands),
        "covered_order": 2,
        "radius": 0.5,
        "seed": 5,
    }
    on_cpu = late.stochastic_energy(model, source, listeners, **common)
    assert on_cpu.plan == ray_device.Plan("cpu")
    for device in ("gpu", "both"):
        found = late.stochastic_energy(model, source, listeners, **common, device=device)
        assert found.plan.device == device and found.traced_rays > 0
        # Within three times the spread of the difference over seeds (5 %).
        assert found.energy.sum() == pytest.approx(on_cpu.energy.sum(), rel=0.15)


# -- outdoors: rays bent by the atmosphere, and through vegetation ------------------------


def _profile(gradient: float, wind: float) -> dict:
    """The tracer's bending arguments for an atmosphere whose speed of sound in still air
    grows by ``gradient`` m/s per metre up to 100 m and stays put above, with a wind of
    ``wind`` m/s 10 m up growing as the log of the height, blowing along x."""
    z = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 1000.0])
    still = 340.0 + gradient * np.minimum(z, 100.0)
    speed = wind * np.log1p(z / 0.1) / np.log1p(10.0 / 0.1)
    return {
        "atmosphere": np.column_stack([z, still, speed]),
        "wind_direction": np.array([1.0, 0.0]),
        "ground": 0.0,
        "chord_sagitta": 0.02,
    }


@needs_gpu
@pytest.mark.parametrize("gradient, wind", [(0.3, 0.0), (-0.3, 0.0), (0.0, 6.0), (-0.1, 4.0)])
def test_gpu_chords_follow_the_cpu_s(gradient: float, wind: float) -> None:
    """One chord of each of 4000 rays from random places in random directions, half of
    them leaving a plane they must not meet again: the GPU's chords turn where the
    CPU's do, point the same way and stretch as far to a part in a thousand; and cut to
    a common length, their ends and times are the CPU's to the millimetre."""
    bending = _profile(gradient, wind)
    rng = np.random.default_rng(3)
    n = 4000
    origins = rng.uniform([-50.0, -50.0, 0.0], [50.0, 50.0, 60.0], (n, 3))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    budget = rng.uniform(1.0, 400.0, n)
    away = np.zeros((n, 3))
    away[n // 2 :] = rng.normal(size=(n - n // 2, 3))
    away[n // 2 :] /= np.linalg.norm(away[n // 2 :], axis=1, keepdims=True)
    on_cpu = _native.bent_chords(origins, directions, budget, away, *bending.values())
    vertices, faces = _unit_cube()
    tracer = gpu_tracer.GpuTracer(_native.Bvh(vertices * 200.0 - 100.0, faces))
    on_gpu = tracer.bent_chords(origins, directions, budget, away, *bending.values())
    ends, tangents, arcs, times, turned = on_cpu
    assert arcs.max() > 300.0 and (gradient == 0.0 or turned[n // 2 :].any())
    assert (on_gpu[4] == turned).all()
    assert np.linalg.norm(on_gpu[1] - tangents, axis=1).max() < 1e-4
    # A chord ends where its sagitta would grow past the tolerance, and that end moves
    # a little with the rounding of the turn gathered along it (most where the wind
    # shears hard just above the ground); so the arcs agree to a part in a thousand
    # for all but a few, and along a common arc the two chords are the same curve.
    stretch = np.abs(on_gpu[2] - arcs) / (1.0 + arcs)
    assert np.quantile(stretch, 0.99) < 1e-3 and stretch.max() < 2e-2
    common = 0.999 * np.minimum(on_gpu[2], arcs)
    on_cpu = _native.bent_chords(origins, directions, common, away, *bending.values())
    on_gpu = tracer.bent_chords(origins, directions, common, away, *bending.values())
    assert np.allclose(on_cpu[2], common, rtol=0.0, atol=1e-6 + 1e-6 * common)
    assert np.allclose(on_gpu[2], common, rtol=0.0, atol=1e-4 + 1e-6 * common)
    assert np.linalg.norm(on_gpu[0] - on_cpu[0], axis=1).max() < 1e-3
    assert np.abs(on_gpu[3] - on_cpu[3]).max() < 1e-5


def _open_field(bands: int = 2, rays: int = 200_000, **kwargs) -> dict:
    """The tracer's arguments for a field 400 m across under a sky 200 m up: the ground
    reflects, the sides and the sky absorb all; a source 2 m up in the middle and
    listeners 100 m off downwind, upwind and across the wind, and 60 m up."""
    vertices, faces = _unit_cube()
    bottom = np.all(vertices[faces][:, :, 2] == 0.0, axis=1)
    absorption = np.ones((12, bands))
    absorption[bottom] = 0.05
    scattering = np.zeros((12, bands))
    scattering[bottom] = 0.3
    args = _stochastic_args(rays)
    args.update(
        bvh=_native.Bvh(vertices * (400.0, 400.0, 200.0) - (200.0, 200.0, 0.0), faces),
        source=np.array([0.0, 0.0, 2.0]),
        absorption=absorption,
        scattering=scattering,
        air=np.array([1e-4, 1e-3])[:bands],
        receivers=np.array([[100.0, 0, 2], [-100.0, 0, 2], [0, 100.0, 2], [0, 0, 60.0]]),
        receiver_radius=np.full(4, 8.0),
        max_time=1.0,
        time_step=5e-3,
        diffuse_rain=True,
    )
    args.update(kwargs)
    return args


def _outdoors(**values):
    """The GPU's trace of a run whose rays the air bends or that has vegetation; None on
    Direct3D 12, which leaves those to the CPU, having checked that it says so."""
    if gpu_tracer.outdoors_declined(True, True):
        with pytest.raises(gpu_tracer.Unsupported, match="Direct3D 12"):
            gpu_tracer.trace_stochastic(**values)
        return None
    return gpu_tracer.trace_stochastic(**values)


def _arrivals(energy: np.ndarray, time_step: float) -> np.ndarray:
    """Per listener and band, the energy and the mean time it arrives at (s)."""
    t = (np.arange(energy.shape[2]) + 0.5) * time_step
    total = energy.sum(2)
    return np.stack([total, (energy * t).sum(2) / np.maximum(total, 1e-300)], -1)


@needs_gpu
@pytest.mark.parametrize("gradient, wind", [(0.3, 0.0), (-0.3, 0.0), (-0.1, 6.0)])
def test_gpu_tracer_bends_the_rays_as_the_cpu_does(gradient: float, wind: float) -> None:
    """Sound bent down (a warmer sky, and downwind) reaches listeners 100 m off, and bent
    up (a cooler sky, and upwind) leaves them in a shadow: the GPU's energy per listener
    and band, and the mean time it arrives at, are the CPU's within the noise."""
    values = _open_field() | _profile(gradient, wind)
    on_gpu = _outdoors(**values)
    if on_gpu is None:
        return
    on_cpu = _native.trace_stochastic(**values)
    step = values["time_step"]
    cpu, found = _arrivals(on_cpu[0], step), _arrivals(on_gpu[0], step)
    assert np.all(cpu[..., 0] > 0.0)
    np.testing.assert_allclose(found[..., 0], cpu[..., 0], rtol=0.06)
    np.testing.assert_allclose(found[..., 1], cpu[..., 1], atol=4e-3)
    # The rays that bend up out of the field leave it, and both count them.
    assert on_gpu[3] == pytest.approx(on_cpu[3], rel=0.03)
    # Their energy over time, which the weight window reads.
    np.testing.assert_allclose(on_gpu[4], on_cpu[4], rtol=0.03, atol=1e-3 * on_cpu[4].max())


def _crown(absorption, scattering, asymmetry) -> dict:
    """A spheroid of foliage 30 m across and 20 m high, between the source and the first
    listener of :func:`_big_box`."""
    return {
        "plants": np.array([[1, 0, 25.0, 0.0, 0.0, 15.0, 10.0, 0]], dtype=float),
        "plant_outline": np.zeros((0, 2)),
        "plant_absorption": np.atleast_2d(absorption).astype(float),
        "plant_scattering": np.atleast_2d(scattering).astype(float),
        "plant_asymmetry": np.atleast_2d(asymmetry).astype(float),
        "plant_upright": np.array([0], dtype=np.uint8),
    }


@needs_gpu
def test_gpu_tracer_loses_what_the_plants_absorb_as_the_cpu_does() -> None:
    """A slab of absorbing plants 30 m thick, and a crown of them: the same rays on the
    GPU with and without the plants (absorbing draws no random numbers), so Beer and
    Lambert's law holds exactly, as on the CPU."""
    rate = np.array([0.02, 0.05])
    slab = _slab(rate, [0, 0], [0, 0])
    through = _outdoors(**_big_box(2, 50_000), **slab)
    if through is None:
        return
    through = through[0].sum(axis=2)
    free = gpu_tracer.trace_stochastic(**_big_box(2, 50_000))[0].sum(axis=2)
    assert through[0] / free[0] == pytest.approx(np.exp(-30 * rate), rel=3e-3)
    np.testing.assert_allclose(through[1:], free[1:], rtol=1e-6)
    crown = _crown(rate, [0, 0], [0, 0])
    on_cpu = _native.trace_stochastic(**_big_box(2, 50_000), **crown)[0].sum(axis=2)
    on_gpu = gpu_tracer.trace_stochastic(**_big_box(2, 50_000), **crown)[0].sum(axis=2)
    assert 0.3 < on_cpu[0, 0] / free[0, 0] < 0.9
    np.testing.assert_allclose(on_gpu, on_cpu, rtol=3e-3)


@needs_gpu
def test_gpu_tracer_scatters_in_foliage_and_stems_as_the_cpu_does() -> None:
    """Foliage that scatters forward and rains straight to the listeners, and stems that
    turn the rays sideways and rain what their next legs carry through the spheres: the
    GPU's energy per listener and band is the CPU's within the noise."""
    big = {"receiver_radius": np.full(3, 10.0), "diffuse_rain": True}
    foliage = _slab([0.0, 0.01], [0.03, 0.02], [0.0, 0.6])
    on_gpu = _outdoors(**_big_box(2, 300_000, **big), **foliage)
    if on_gpu is None:
        return
    on_cpu = _native.trace_stochastic(**_big_box(2, 300_000, **big), **foliage)[0].sum(axis=2)
    np.testing.assert_allclose(on_gpu[0].sum(axis=2), on_cpu, rtol=0.03)
    crown = _crown([0.0, 0.01], [0.03, 0.02], [0.0, 0.6])
    on_cpu = _native.trace_stochastic(**_big_box(2, 300_000, **big), **crown)[0].sum(axis=2)
    on_gpu = gpu_tracer.trace_stochastic(**_big_box(2, 300_000, **big), **crown)[0].sum(axis=2)
    np.testing.assert_allclose(on_gpu, on_cpu, rtol=0.03)
    stems = _slab([0.0], [0.03], [-0.3], True)
    beside = {
        "receivers": np.array([[0.0, 50.0, 0.0], [25.0, 30.0, 0.0]]),
        "receiver_radius": np.array([10.0, 10.0]),
        "diffuse_rain": True,
    }
    on_cpu = _native.trace_stochastic(**_big_box(1, 300_000, **beside), **stems)[0].sum(axis=2)
    on_gpu = gpu_tracer.trace_stochastic(**_big_box(1, 300_000, **beside), **stems)[0].sum(axis=2)
    np.testing.assert_allclose(on_gpu, on_cpu, rtol=0.03)
    # Detector spheres alone, with the stems' scattering: the same within the noise.
    spheres = {"receivers": beside["receivers"], "receiver_radius": beside["receiver_radius"]}
    on_cpu = _native.trace_stochastic(**_big_box(1, 300_000, **spheres), **stems)[0].sum(axis=2)
    on_gpu = gpu_tracer.trace_stochastic(**_big_box(1, 300_000, **spheres), **stems)[0].sum(axis=2)
    np.testing.assert_allclose(on_gpu, on_cpu, rtol=0.03)


@needs_gpu
def test_gpu_tracer_bends_rays_through_plants_as_the_cpu_does() -> None:
    """Both at once: a crown of scattering foliage in a field whose sound bends down."""
    crown = _crown([0.0, 0.01], [0.03, 0.02], [0.0, 0.6])
    crown["plants"][0, 2:5] = (50.0, 0.0, 10.0)
    values = _open_field() | _profile(0.3, 0.0) | crown
    on_gpu = _outdoors(**values)
    if on_gpu is None:
        return
    on_cpu = _native.trace_stochastic(**values)
    step = values["time_step"]
    cpu, found = _arrivals(on_cpu[0], step), _arrivals(on_gpu[0], step)
    np.testing.assert_allclose(found[..., 0], cpu[..., 0], rtol=0.06)
    np.testing.assert_allclose(found[..., 1], cpu[..., 1], atol=4e-3)


@needs_gpu
def test_the_late_stage_traces_outdoor_scenes_on_the_gpu() -> None:
    """The open-air stage in the wind: the GPU bends its late rays, and the result is
    the CPU's within the noise; on Direct3D 12 the CPU traces them, and says why."""
    room = Room.from_yaml(OPEN_AIR)
    model = TraceModel.from_room(room, room.scene.simulation)
    source = np.array(room.scene.sources[0].position, float)
    listeners = np.array([listener.position for listener in room.scene.listeners], float)
    common = {
        "rays": 8000,
        "max_time": 1.0,
        "time_step": 1e-3,
        "speed_of_sound": room.speed_of_sound(),
        "air": room.air_attenuation(model.bands),
        "covered_order": 1,
        "radius": 3.0,
        "seed": 5,
    }
    on_cpu = late.stochastic_energy(model, source, listeners, **common)
    assert on_cpu.plan == ray_device.Plan("cpu")
    found = late.stochastic_energy(model, source, listeners, **common, device="gpu")
    declined = gpu_tracer.outdoors_declined(True, True)
    expected = ray_device.Plan("cpu", reason=declined) if declined else ray_device.Plan("gpu")
    assert found.plan == expected
    assert found.traced_rays > 0
    assert found.energy.sum() == pytest.approx(on_cpu.energy.sum(), rel=0.15)
    np.testing.assert_allclose(found.energy.sum((1, 2)), on_cpu.energy.sum((1, 2)), rtol=0.25)


@needs_gpu
def test_the_late_stage_timing_script_runs(monkeypatch, capsys) -> None:
    script = Path(__file__).resolve().parent.parent / "validation" / "late_gpu.py"
    argv = ["late_gpu.py", "--scenes", str(SHOEBOX), "--rays", "1500", "--devices", "gpu"]
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert "1,500 per band" in out
    rows = [line for line in out.splitlines() if line.startswith("| shoebox")]
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
    assert [row[3] for row in cells] == ["cpu", "cpu (next seed)", "gpu"]
    assert [row[4] for row in cells] == ["the CPU", "the CPU", "the GPU"]
    assert all(len(row) == 10 and row[6].endswith("x") for row in cells)
    # Within the noise of 1,500 rays: as far from the CPU's as another seed is, or not
    # much further.
    t30 = [float(row[7].rstrip("%")) for row in cells[1:]]
    assert t30[1] < max(3 * t30[0], 5.0)


@needs_gpu
def test_the_outdoor_timing_script_runs(monkeypatch, capsys) -> None:
    script = Path(__file__).resolve().parent.parent / "validation" / "outdoor_gpu.py"
    argv = ["outdoor_gpu.py", "--scenes", str(OPEN_AIR), "--rays", "2000", "--devices", "gpu"]
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert "2,000 per band" in out
    rows = [line for line in out.splitlines() if line.startswith("| open_air_stage")]
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
    assert [row[3] for row in cells] == ["cpu", "cpu (next seed)", "gpu"]
    declined = gpu_tracer.outdoors_declined(True, True)
    on_gpu = f"the CPU ({declined})" if declined else "the GPU"
    assert [row[4] for row in cells] == ["the CPU", "the CPU", on_gpu]
    assert all(len(row) == 9 and row[6].endswith("x") for row in cells)
    # Within the noise of 2,000 rays: as far from the CPU's as another seed is, or not
    # much further.
    level = [float(row[7]) for row in cells[1:]]
    assert level[1] < max(3 * level[0], 1.0)


@needs_gpu
def test_tune_times_each_device_and_auto_takes_its_choices(capsys, device_choices) -> None:
    from acoustic_spaces.cli import main
    from acoustic_spaces.engine import tuning

    room = Room.from_yaml(SHOEBOX)
    # Without keep, nothing is saved (the editor saves on OK).
    result = tuning.tune(room, rays=100, lengths=[0.1], wave_lengths=None, keep=False)
    assert [timing.length for timing in result.late] == [0.1] and result.wave is None
    assert not device_choices.exists()
    argv = ["tune", str(SHOEBOX), "--rays", "200", "--lengths", "0.1", "0.3", "--default", "auto"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "late stage, 200 rays per band (Shoebox" in out and str(device_choices) in out
    assert "default device is now auto" in out and tuning.default_device() == "auto"
    assert "wave solver, " in out and "auto: the " in out
    # A software GPU (lavapipe, SwiftShader) is timed, but auto keeps the CPU.
    assert ("software renderer" in out) == (not gpu.hardware_gpu())
    late = tuning.late_choices()
    assert list(late) == [0.125, 0.5]
    assert all(entry["from"] == "tune" and entry["rays"] == 200 for entry in late.values())
    assert set(late[0.5]["seconds"]) == {"cpu", "gpu", "both"}
    assert tuning.wave_costs() is not None
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "late stage up to 0.125 s:" in out and "(tune)" in out and "wave solver: the " in out
