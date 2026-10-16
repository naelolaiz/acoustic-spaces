"""Where the late stage traces its rays: the choice of device, ``auto``'s trial and the
CPU and the GPU sharing the rays. None of these needs a GPU: test_gpu.py traces on one."""

from pathlib import Path

import numpy as np
import pytest
from test_tracer_reference import REFERENCE, late_results, reference_arguments

from acoustic_spaces import Room, _native
from acoustic_spaces.engine import gpu, gpu_tracer, ray_device, tuning

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(REFERENCE) as stored:
        return dict(stored)


def _arguments(reference, name: str) -> dict:
    return reference_arguments(_native, reference, name)


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A computer with a hardware GPU (as far as ray_device can tell), a cache of its
    own (conftest's device_choices) and no trial run yet."""
    monkeypatch.setattr(gpu, "_adapter", lambda: object())
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: True)
    monkeypatch.setattr(gpu, "gpu_name", lambda: "Test GPU (Vulkan, DiscreteGPU)")
    monkeypatch.setattr(ray_device, "_rates", {})
    monkeypatch.setattr(ray_device, "_trial", True)
    return tmp_path


def test_the_cpu_traces_as_before(reference) -> None:
    values = _arguments(reference, "vector")
    plan = ray_device.choose("cpu", values)
    assert plan == ray_device.Plan("cpu")
    found = late_results(ray_device.trace(plan, values))
    expected = late_results(_native.trace_stochastic(**values))
    for key, value in expected.items():
        np.testing.assert_array_equal(found[key], value, err_msg=key)


def test_gpu_settings_need_a_gpu(reference, monkeypatch) -> None:
    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    values = _arguments(reference, "vector")
    for setting in ("gpu", "both"):
        with pytest.raises(ValueError, match=f"device is {setting}"):
            ray_device.choose(setting, values)
    with pytest.raises(ValueError, match="cpu, gpu, both, auto"):
        ray_device.choose("tpu", values)
    # auto never fails: without a GPU it takes the CPU.
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: False)
    assert ray_device.choose("auto", values) == ray_device.Plan("cpu", reason="no hardware GPU")


def test_what_only_the_cpu_traces_stays_there(reference, machine) -> None:
    rooms = _arguments(reference, "rooms_first")
    for setting in ("gpu", "both", "auto"):
        plan = ray_device.choose(setting, rooms)
        assert plan.device == "cpu" and "partitions" in plan.reason
    plan = ray_device.choose("gpu", _arguments(reference, "vector") | {"groups": 2})
    assert plan.device == "cpu" and "face groups" in plan.reason


@pytest.mark.parametrize(
    "cpu_rate, gpu_rate, threads, device",
    [
        (100.0, 10.0, 16, "cpu"),  # the GPU adds too little
        (100.0, 50.0, 16, "both"),  # 1.4 times the CPU
        (10.0, 100.0, 16, "gpu"),  # the CPU would add under a tenth
        (100.0, 110.0, 1, "cpu"),  # a close call: the safe choice
        (100.0, 130.0, 1, "gpu"),
    ],
)
def test_auto_keeps_the_fastest_and_the_cpu_on_a_close_call(
    cpu_rate, gpu_rate, threads, device
) -> None:
    assert ray_device._decide(cpu_rate, gpu_rate, threads) == device


def test_auto_runs_a_trial_once_and_saves_its_choice(reference, machine, monkeypatch) -> None:
    trials = []

    def trial(values):
        trials.append(values["rays"])
        return 100.0, 80.0

    monkeypatch.setattr(ray_device, "trial", trial)
    values = _arguments(reference, "vector") | {"threads": 8}
    plan = ray_device.choose("auto", values)
    assert plan == ray_device.Plan("both", 100.0, 80.0, "the trial")
    assert ray_device.choose("auto", values) == plan
    assert len(trials) == 1  # once per model and range of lengths in a session
    saved = tuning.late_choice(values["max_time"])
    assert saved == {"device": "both", "cpu_rate": 100.0, "gpu_rate": 80.0, "from": "trial"}
    assert "Test GPU" in (machine / "devices.json").read_text()
    assert list(tuning.late_choices()) == [0.5]  # the run is 0.4 s long
    # Without the trial, auto takes the saved choice; "both" shares by its speeds.
    ray_device.set_trial(False)
    assert not ray_device.trial_enabled()
    assert ray_device.choose("auto", values) == ray_device.Plan(
        "both", 100.0, 80.0, "the saved choice"
    )
    monkeypatch.setattr(ray_device, "_rates", {})
    assert ray_device.choose("both", values) == ray_device.Plan("both", 100.0, 80.0)
    (machine / "devices.json").unlink()
    plan = ray_device.choose("auto", values)
    assert plan == ray_device.Plan("cpu", reason="no trial or tune on this computer yet")
    assert len(trials) == 1


def test_choices_are_kept_by_range_of_response_lengths(machine) -> None:
    keys = [tuning.length_key(t) for t in (0.1, 0.3, 0.5, 0.51, 1.0, 3.5, 7.0)]
    assert keys == ["0.125", "0.5", "0.5", "1", "1", "4", "8"]
    assert tuning.late_choice(1.0) is None and tuning.late_choices() == {}
    for length, device in [(0.5, "cpu"), (2.0, "gpu")]:
        tuning.keep_late(length, {"device": device, "cpu_rate": 1, "gpu_rate": 2, "from": "tune"})
    # The shortest range at least as long, else the longest.
    picked = [tuning.late_choice(t)["device"] for t in (0.2, 0.5, 1.0, 2.0, 6.0)]
    assert picked == ["cpu", "cpu", "gpu", "gpu", "gpu"]
    assert tuning.late_entry(1.0) is None and tuning.late_entry(1.5)["device"] == "gpu"


def test_auto_takes_what_tune_kept_without_a_trial(reference, machine, monkeypatch) -> None:
    trials = []

    def trial(values):
        trials.append(values["max_time"])
        return 100.0, 10.0

    monkeypatch.setattr(ray_device, "trial", trial)
    values = _arguments(reference, "vector")  # 0.4 s
    tuning.keep_late(0.5, {"device": "gpu", "cpu_rate": 1.0, "gpu_rate": 3.0, "from": "tune"})
    tuned = ray_device.Plan("gpu", 1.0, 3.0, "acoustic-spaces tune")
    assert ray_device.choose("auto", values) == tuned and not trials
    assert ray_device.choose("both", values) == ray_device.Plan("both", 1.0, 3.0)
    # A length tune did not time: the trial, for the run's own length, kept for its range.
    longer = values | {"max_time": 1.5}
    assert ray_device.choose("auto", longer) == ray_device.Plan("cpu", 100.0, 10.0, "the trial")
    assert trials == [1.5] and tuning.late_entry(1.5)["from"] == "trial"
    # Without the trial, the nearest kept choice.
    ray_device.set_trial(False)
    kept = ray_device.choose("auto", values | {"max_time": 3.0})
    assert kept == ray_device.Plan("cpu", 100.0, 10.0, "the saved choice")
    assert ray_device.choose("auto", values) == tuned and trials == [1.5]


def test_a_plan_given_is_followed(reference, machine, monkeypatch) -> None:
    values = _arguments(reference, "vector")
    plan = ray_device.Plan("both", 2.0, 1.0)
    assert ray_device.choose(plan, values) == plan
    rooms = ray_device.choose(ray_device.Plan("gpu"), _arguments(reference, "rooms_first"))
    assert rooms.device == "cpu" and "partitions" in rooms.reason
    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    with pytest.raises(ValueError, match="device is gpu"):
        ray_device.choose(ray_device.Plan("gpu"), values)
    assert ray_device.choose(ray_device.Plan("cpu"), values) == ray_device.Plan("cpu")


def test_the_wave_solver_takes_the_faster_for_its_step_count(machine, monkeypatch) -> None:
    assert tuning.wave_costs() is None and tuning.wave_prefers_gpu(100) is None
    assert gpu.use_gpu("auto", 100)  # a hardware GPU, nothing timed
    # The GPU costs more to set up and less per step: the GPU from 124 steps on.
    tuning.keep_wave({"cpu": [0.0, 1e-8], "gpu": [1e-6, 1e-9], "from": "tune"})
    assert tuning.wave_pick(tuning.wave_costs()) == "the GPU from 124 steps on, else the CPU"
    assert not gpu.use_gpu("auto", 100) and gpu.use_gpu("auto", 200)
    assert gpu.use_gpu("gpu", 100) and not gpu.use_gpu("cpu", 200)
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: False)
    assert not gpu.use_gpu("auto", 200)  # never a software GPU
    for cpu, gpu_costs, pick in [
        ([0.0, 1e-8], [0.0, 1e-9], "the GPU"),
        ([0.0, 1e-9], [1e-6, 1e-8], "the CPU"),
        ([1e-6, 1e-9], [0.0, 3e-9], "the GPU up to 434 steps, else the CPU"),
    ]:
        assert tuning.wave_pick({"cpu": cpu, "gpu": gpu_costs}) == pick


def test_a_damaged_file_keeps_nothing(machine) -> None:
    (machine / "devices.json").write_text("not json")
    assert tuning.saved() == {} and tuning.late_choice(1.0) is None
    tuning.keep_late(1.0, {"device": "tpu", "cpu_rate": 1, "gpu_rate": 1})
    tuning.keep_late(2.0, {"device": "gpu", "cpu_rate": "fast"})
    tuning.keep_wave({"cpu": [1.0]})
    assert tuning.late_choices() == {} and tuning.wave_costs() is None


def test_the_trial_times_both_devices(reference, monkeypatch) -> None:
    calls = []

    def fake(device):
        def trace(**values):
            calls.append((device, values["rays"], values["max_time"], values["first_ray"]))

        return trace

    monkeypatch.setattr(_native, "trace_stochastic", fake("cpu"))
    monkeypatch.setattr(gpu_tracer, "trace_stochastic", fake("gpu"))
    cpu_rate, gpu_rate = ray_device.trial(_arguments(reference, "vector_split"))
    assert cpu_rate > 0 and gpu_rate > 0
    # A warm-up on the GPU, then the same run on each: a few rays, past the run's own,
    # for the run's own length.
    assert [c[0] for c in calls] == ["gpu", "gpu", "cpu"]
    assert calls[1][1:] == calls[2][1:] == (ray_device.TRIAL_RAYS, 0.4, 1 << 40)


@pytest.mark.parametrize("name", ["vector", "vector_split", "secondary"])
def test_shared_rays_add_up_to_one_run(reference, monkeypatch, name) -> None:
    """With every share traced on the CPU (the GPU declining them), sharing gives what
    one run on the CPU does: each share's energy, ray energies and weight window
    scaled to the whole run."""

    def decline(**_):
        raise gpu_tracer.Unsupported("not here")

    monkeypatch.setattr(gpu_tracer, "trace_stochastic", decline)
    values = _arguments(reference, name)
    plan = ray_device.Plan("both", cpu_rate=1.0, gpu_rate=3.0)
    found = late_results(ray_device.trace(plan, values))
    expected = late_results(_native.trace_stochastic(**values))
    for key, value in expected.items():
        np.testing.assert_allclose(found[key], value, rtol=1e-9, atol=1e-300, err_msg=key)


def test_simulate_says_where_the_late_rays_went(monkeypatch) -> None:
    from acoustic_spaces.engine.simulate import simulate

    monkeypatch.setattr(gpu, "hardware_gpu", lambda: False)
    room = Room.from_yaml(SHOEBOX)
    sim = room.scene.simulation
    for device, said in [("cpu", None), ("auto", "the CPU (no hardware GPU)")]:
        late = sim.late.model_copy(update={"rays": 300, "device": device})
        messages: list[str] = []
        settings = sim.model_copy(update={"ir_length_s": 0.3, "late": late})
        simulate(room, settings=settings, progress=messages.append)
        where = [m for m in messages if "late rays traced on" in m]
        assert where == ([f"talker: late rays traced on {said}"] if said else [])
