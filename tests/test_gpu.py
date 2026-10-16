"""The wave solver stepped, and rays searched and traced, on a GPU through WebGPU, against
the CPU.

The comparisons run wherever wgpu finds a GPU, software ones included (CI uses Mesa's
lavapipe on Linux), and are skipped elsewhere.
"""

import runpy
from pathlib import Path

import numpy as np
import pytest
from test_native import _unit_cube, cube_with_seats
from test_tracer_reference import REFERENCE, reference_arguments
from test_wave import STENCILS, _courant, _drive, _solver

from acoustic_spaces import Room, _native
from acoustic_spaces.engine import gpu, gpu_rays, gpu_tracer, late, ray_device
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
    # Apple's GPU rounds differently from the CPU: in this nearly closed room the slow
    # drift the walls' branches leave picks that up, 1.1e-5 of the peak by the end. A
    # field swapped the wrong way between runs would differ by far more.
    assert np.abs(a - b).max() < 3e-5 * np.abs(a).max()


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
    assert WaveStage().device is None  # this computer's default
    assert not gpu.use_gpu("cpu")
    with pytest.raises(ValueError, match="cpu, gpu or auto"):
        gpu.use_gpu("tpu")
    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    assert not gpu.use_gpu("auto")
    assert gpu.gpu_name() is None
    with pytest.raises(ValueError, match="simulation.wave.device is gpu"):
        gpu.use_gpu("gpu")


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
