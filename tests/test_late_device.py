"""Where the late stage traces its rays: the choice of device, ``auto``'s trial and the
CPU and the GPU sharing the rays. None of these needs a GPU: test_gpu.py traces on one."""

from pathlib import Path

import numpy as np
import pytest
from test_tracer_reference import REFERENCE, late_results, reference_arguments

from acoustic_spaces import Room, _native
from acoustic_spaces.engine import gpu, gpu_tracer, ray_device

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
    own and no trial run yet."""
    monkeypatch.setattr(gpu, "_adapter", lambda: object())
    monkeypatch.setattr(gpu, "hardware_gpu", lambda: True)
    monkeypatch.setattr(gpu, "gpu_name", lambda: "Test GPU (Vulkan, DiscreteGPU)")
    monkeypatch.setattr(ray_device, "_cache_file", lambda: tmp_path / "late-device.json")
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
    assert len(trials) == 1  # once per model in a session
    saved = ray_device.saved_choice()
    assert saved == {"device": "both", "cpu_rate": 100.0, "gpu_rate": 80.0}
    assert "Test GPU" in (machine / "late-device.json").read_text()
    # Without the trial, auto takes the saved choice; "both" shares by its speeds.
    ray_device.set_trial(False)
    assert not ray_device.trial_enabled()
    assert ray_device.choose("auto", values) == ray_device.Plan(
        "both", 100.0, 80.0, "the saved choice"
    )
    monkeypatch.setattr(ray_device, "_rates", {})
    assert ray_device.choose("both", values) == ray_device.Plan("both", 100.0, 80.0)
    (machine / "late-device.json").unlink()
    plan = ray_device.choose("auto", values)
    assert plan == ray_device.Plan("cpu", reason="no trial on this computer yet")
    assert len(trials) == 1


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
    # A warm-up on the GPU, then the same short run on each, past the run's own rays.
    assert [c[0] for c in calls] == ["gpu", "gpu", "cpu"]
    assert calls[1][1:] == calls[2][1:] == (ray_device.TRIAL_RAYS, 0.25, 1 << 40)


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
