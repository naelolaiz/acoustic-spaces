from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.cli import main
from acoustic_spaces.render import convolve
from acoustic_spaces.spatial import Mono
from acoustic_spaces.walkthrough import (
    CROSSFADE_S,
    LATE_STEP_M,
    STEP_M,
    WalkPoints,
    WalkResponses,
    crossfade_weights,
    late_mixing,
    render_walk,
    simulate_walk,
    walk_points,
)

SHOEBOX_OBJ = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "shoebox.obj"
GALLERY = Path(__file__).resolve().parent.parent / "examples" / "gallery_walk" / "scene.yaml"
QUICK = {
    "ir_length_s": 0.3,
    "seed": 3,
    "early": {"max_order": 2},
    "late": {"rays": 2000},
}


def _scene(*, sources, listeners, simulation=None) -> Scene:
    return Scene.model_validate(
        {
            "geometry": {"file": str(SHOEBOX_OBJ)},
            "materials": {"m": {"absorption": [0.2] * 8, "scattering": 0.3}},
            "default_material": "m",
            "sources": sources,
            "listeners": listeners,
            "simulation": simulation or QUICK,
        }
    )


def _walker(walk, *, position=(1.0, 1.0, 1.5), **extra) -> dict:
    return {"name": "w", "position": list(position), "walk": walk, **extra}


def test_walk_points_follow_the_path_facing_forward() -> None:
    walk = {"to": [[3.0, 1.0, 1.5], [3.0, 2.0, 1.5]], "speed_m_s": 1.5}
    scene = _scene(sources=[{"name": "s", "position": [6, 4, 1.5]}], listeners=[_walker(walk)])
    points = walk_points(scene.listeners[0], 0.25)
    assert len(points.times) == 13
    assert points.times[-1] == pytest.approx(3.0 / 1.5)
    np.testing.assert_allclose(np.diff(points.times), 0.25 / 1.5)
    np.testing.assert_allclose(points.positions[8], [3.0, 1.0, 1.5])
    np.testing.assert_allclose(points.positions[-1], [3.0, 2.0, 1.5])
    # Along +x, then (from the corner on) along +y.
    np.testing.assert_allclose(points.yaws[:8], 0.0)
    np.testing.assert_allclose(points.yaws[8:], 90.0)
    still = scene.listeners[0].model_copy(
        update={"walk": scene.listeners[0].walk.model_copy(update={"face_forward": False})}
    )
    assert walk_points(still).yaws is None
    # A step that does not divide the path: a little shorter, the same distance apart.
    uneven = walk_points(scene.listeners[0], 0.7)
    assert len(uneven.times) == 6
    np.testing.assert_allclose(np.diff(uneven.times), 3.0 / 5 / 1.5)


def test_a_walk_needs_somewhere_to_go_and_a_point_source() -> None:
    sources = [{"name": "s", "position": [6, 4, 1.5]}]
    with pytest.raises(ValidationError, match="at least 1"):
        _scene(sources=sources, listeners=[_walker({"to": []})])
    with pytest.raises(ValidationError, match="greater than 0"):
        _scene(sources=sources, listeners=[_walker({"to": [[2, 1, 1.5]], "speed_m_s": 0})])
    line = {
        "name": "line",
        "position": [2, 2, 1.5],
        "line_m": 2.0,
        "walk": {"to": [[3, 3, 1.5]]},
    }
    with pytest.raises(ValidationError, match="only a point source walks"):
        _scene(sources=[line], listeners=[{"name": "l", "position": [6, 4, 1.2]}])


def test_crossfade_weights_add_up_to_one() -> None:
    times = np.array([0.0, 0.2, 0.4, 0.45, 1.0])
    t = np.linspace(-0.1, 1.5, 3001)
    weights = crossfade_weights(times, t, 0.05)
    assert weights.shape == (5, len(t))
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)
    assert weights.min() >= 0.0 and weights.max() <= 1.0
    # Each place has the sound to itself at its own time, and the crossfade to the next
    # is centred half-way (shortened where places are closer than the crossfade).
    for k, time in enumerate(times):
        assert crossfade_weights(times, [time], 0.05)[k, 0] == pytest.approx(1.0)
    half = crossfade_weights(times, [0.1, 0.425], 0.05)
    np.testing.assert_allclose(half[:, 0], [0.5, 0.5, 0, 0, 0])
    np.testing.assert_allclose(half[:, 1], [0, 0, 0.5, 0.5, 0])


def test_late_mixing_interpolates_between_traced_places() -> None:
    distances = np.arange(0.0, 5.01, 0.25)
    mixing = late_mixing(distances, 2.0)
    traced = sorted(set(range(len(distances))) - set(mixing))
    # 5 m in steps of at most 2 m: 4 places, 5/3 m apart (rounded to the nearest place).
    assert traced[0] == 0 and traced[-1] == len(distances) - 1 and len(traced) == 4
    for k, parts in mixing.items():
        (a, wa), (b, wb) = sorted(parts.items())
        assert a < k < b and a in traced and b in traced
        assert wa + wb == pytest.approx(1.0) and wa > 0 and wb > 0
        assert wa * distances[a] + wb * distances[b] == pytest.approx(distances[k])
    with pytest.raises(ValueError, match="positive"):
        late_mixing(distances, 0.0)


def test_late_from_mixes_the_late_part_of_listeners() -> None:
    listeners = [
        {"name": "a", "position": [2.0, 2.0, 1.5]},
        {"name": "b", "position": [6.0, 4.0, 1.5]},
        {"name": "c", "position": [4.0, 3.0, 1.5]},
    ]
    room = Room.from_scene(
        _scene(sources=[{"name": "s", "position": [1, 5, 1.5]}], listeners=listeners)
    )
    own = simulate(room)
    mixed = simulate(room, late_from={"c": {"a": 0.25, "b": 0.75}})
    for name in ("a", "b"):
        np.testing.assert_allclose(mixed["s", name].late, own["s", name].late, rtol=1e-9)
    np.testing.assert_allclose(
        mixed["s", "c"].late, 0.25 * own["s", "a"].late + 0.75 * own["s", "b"].late, rtol=1e-9
    )
    np.testing.assert_allclose(
        mixed["s", "c"].late_directional,
        0.25 * own["s", "a"].late_directional + 0.75 * own["s", "b"].late_directional,
        rtol=1e-9,
    )
    # The early sound stays its own.
    np.testing.assert_array_equal(mixed["s", "c"].early.delay_s, own["s", "c"].early.delay_s)
    with pytest.raises(ValueError, match="no listener or source here"):
        simulate(room, late_from={"d": {"a": 1.0}})
    with pytest.raises(ValueError, match="which is no listener traced for its own"):
        simulate(room, late_from={"c": {"b": 1.0}, "b": {"a": 1.0}})
    with pytest.raises(ValueError, match="positive"):
        simulate(room, late_from={"c": {"a": 0.0}})


def test_late_from_mixes_the_late_part_of_sources() -> None:
    sources = [
        {"name": "a", "position": [2.0, 2.0, 1.5]},
        {"name": "b", "position": [6.0, 4.0, 1.5]},
        {"name": "c", "position": [4.0, 3.0, 1.5]},
    ]
    room = Room.from_scene(
        _scene(sources=sources, listeners=[{"name": "l", "position": [1, 5, 1.2]}])
    )
    own = simulate(room)
    mixed = simulate(room, late_from={"c": {"a": 0.5, "b": 0.5}})
    np.testing.assert_allclose(mixed["a", "l"].late, own["a", "l"].late, rtol=1e-9)
    np.testing.assert_allclose(
        mixed["c", "l"].late, 0.5 * (own["a", "l"].late + own["b", "l"].late), rtol=1e-9
    )
    assert "rays" not in mixed.details["c"] and mixed.details["a"]["rays"] > 0
    assert mixed["c", "l"].direct_delay_s == own["c", "l"].direct_delay_s


def test_a_walking_listener() -> None:
    walk = {"to": [[5.0, 1.0, 1.5], [5.0, 4.0, 1.5]], "speed_m_s": 2.0}
    sources = [{"name": "s", "position": [7.0, 5.0, 1.5]}, {"name": "t", "position": [1, 5, 1]}]
    room = Room.from_scene(_scene(sources=sources, listeners=[_walker(walk)]))
    walked = simulate_walk(room, step_m=0.5, late_step_m=2.0)
    assert walked.name == "w" and walked.kind == "listeners"
    assert len(walked.names) == 15 and walked.duration_s == pytest.approx(3.5)
    assert walked.late_places == (0, 3, 7, 10, 14)
    assert len(walked.result) == 2 * 15
    c = room.speed_of_sound()
    for k, name in enumerate(walked.names):
        assert name == f"w @ {walked.points.times[k]:.2f} s"
        distance = np.linalg.norm(walked.points.positions[k] - [7.0, 5.0, 1.5])
        assert walked.result["s", name].direct_delay_s == pytest.approx(distance / c)
    every = simulate_walk(room, step_m=0.5, late_step_m=None)
    assert every.late_places == tuple(range(15))


def test_a_walking_source_is_heard_at_a_listener() -> None:
    sources = [
        {"name": "w", "position": [1.0, 1.0, 1.5], "walk": {"to": [[3.0, 1.0, 1.5]]}},
        {"name": "still", "position": [7.0, 5.0, 1.5]},
    ]
    listeners = [{"name": "l", "position": [4, 4, 1.2]}, {"name": "m", "position": [6, 2, 1.2]}]
    room = Room.from_scene(_scene(sources=sources, listeners=listeners))
    with pytest.raises(ValueError, match="pick a listener: l, m"):
        simulate_walk(room)
    walked = simulate_walk(room, listener="m", step_m=1.0)
    assert walked.kind == "sources" and len(walked.names) == 3
    assert {key for key in walked.result.responses} == {
        *((name, "m") for name in walked.names),
        ("still", "m"),
    }
    out = render_walk(walked, {"w": np.ones(100), "still": np.ones(100)})
    assert out.shape[1] == 1 and np.isfinite(out).all()
    with pytest.raises(ValueError, match="no source 'x'"):
        render_walk(walked, {"x": np.ones(10)})


def test_walk_errors() -> None:
    sources = [{"name": "s", "position": [7.0, 5.0, 1.5]}]
    out = _walker({"to": [[9.0, 1.0, 1.5]]})
    room = Room.from_scene(_scene(sources=sources, listeners=[out]))
    with pytest.raises(ValueError, match=r"leaves the room at \(8\.\d\d, 1\.00, 1\.50\) m"):
        simulate_walk(room)
    still = Room.from_scene(
        _scene(sources=sources, listeners=[{"name": "l", "position": [1, 1, 1]}])
    )
    with pytest.raises(ValueError, match="no listener or source walks"):
        simulate_walk(still)
    with pytest.raises(ValueError, match="'l' has no walk"):
        simulate_walk(still, "l")
    with pytest.raises(ValueError, match="no listener or source named 'x'"):
        simulate_walk(still, "x")
    two = [_walker({"to": [[2, 2, 1.5]]}), {**_walker({"to": [[3, 3, 1.5]]}), "name": "v"}]
    both = Room.from_scene(_scene(sources=sources, listeners=two))
    with pytest.raises(ValueError, match=r"more than one walks \(w, v\)"):
        simulate_walk(both)


@dataclass
class _Response:
    ir: np.ndarray
    seed: int = 0

    def spatial_impulse_response(self, fmt, rate, seed):
        return self.ir[:, None]


@dataclass
class _Settings:
    sample_rate: int = 1000
    seed: int | None = 1


@dataclass
class _Result:
    responses: dict
    settings: _Settings = field(default_factory=_Settings)

    def __getitem__(self, key):
        return self.responses[key]

    def __iter__(self):
        return iter(self.responses.values())


def _fake_walk(irs, times) -> WalkResponses:
    names = tuple(f"w{k}" for k in range(len(irs)))
    result = _Result(
        {("s", name): _Response(np.asarray(ir, float)) for name, ir in zip(names, irs, strict=True)}
    )
    points = WalkPoints(np.asarray(times, float), np.zeros((len(times), 3)), None)
    return WalkResponses("w", "listeners", points, names, result)


def test_one_response_everywhere_is_a_plain_convolution() -> None:
    rng = np.random.default_rng(1)
    ir = rng.standard_normal(64)
    walk = _fake_walk([ir] * 6, np.linspace(0.0, 1.0, 6))
    dry = rng.standard_normal(1500)
    out = render_walk(walk, {"s": dry}, Mono())
    np.testing.assert_allclose(out[:, 0], convolve(dry, ir[:, None])[:, 0], atol=1e-12)


def test_crossfades_are_continuous() -> None:
    # Each place a pure gain: the result is the dry signal times the crossfaded gains,
    # changing by at most one gain step per crossfade, with no jump.
    gains = [1.0, 0.25, 2.0, 0.5]
    times = [0.0, 0.3, 0.6, 0.9]
    walk = _fake_walk([[g] for g in gains], times)
    rate = 1000
    out = render_walk(walk, {"s": np.ones(1500)}, crossfade_s=0.05)[:, 0]
    expected = np.array(gains) @ crossfade_weights(times, np.arange(1500) / rate, 0.05)
    np.testing.assert_allclose(out, expected, atol=1e-12)
    # A raised cosine: at most pi / 2 times the mean slope of a linear crossfade.
    assert np.abs(np.diff(out)).max() <= np.pi / 2 * 1.75 / 50 + 1e-9
    # (FFT convolution: equal to rounding.)
    assert out[0] == pytest.approx(1.0) and out[-1] == pytest.approx(0.5)


def test_render_walk_with_the_example(tmp_path: Path) -> None:
    room = Room.from_yaml(GALLERY)
    walker = room.scene.listeners[0]
    assert walker.walk is not None and walk_points(walker).times.size == 101
    short = walker.walk.model_copy(update={"to": [(15.0, 4.0, 1.6)]})
    scene = room.scene.model_copy(update={"listeners": [walker.model_copy(update={"walk": short})]})
    room = Room.from_scene(scene, reuse=room)
    sim = room.scene.simulation
    settings = sim.model_copy(
        update={
            "ir_length_s": 0.3,
            "seed": 1,
            "early": sim.early.model_copy(update={"max_order": 1}),
            "late": sim.late.model_copy(update={"rays": 1000}),
        }
    )
    walked = simulate_walk(room, step_m=0.5, settings=settings)
    assert len(walked.names) == 5 and walked.late_places == (0, 4)
    rate = settings.sample_rate
    dry = np.random.default_rng(0).standard_normal(rate)
    out = render_walk(walked, {"guide": dry}, Mono())
    assert out.shape[0] > rate and out.shape[1] == 1
    assert np.isfinite(out).all() and np.abs(out).max() > 0


def test_cli_walkthrough(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from acoustic_spaces import cli
    from acoustic_spaces.audio import read_wav, write_wav

    assert (cli.STEP_M, cli.LATE_STEP_M, cli.CROSSFADE_S) == (STEP_M, LATE_STEP_M, CROSSFADE_S)
    scene = _scene(
        sources=[{"name": "s", "position": [7.0, 5.0, 1.5]}],
        listeners=[_walker({"to": [[3.0, 1.0, 1.5]], "speed_m_s": 2.0})],
    )
    path = tmp_path / "scene.yaml"
    scene.to_yaml(path)
    dry = tmp_path / "dry.wav"
    write_wav(dry, np.random.default_rng(0).standard_normal((8000, 1)) * 0.1, 8000)
    out = tmp_path / "walk.wav"
    args = ["walkthrough", str(path), f"s={dry}", "--out", str(out), "--step", "1"]
    assert main([*args, "--format", "binaural", "--normalize"]) == 0
    data, rate = read_wav(out)
    assert data.shape[1] == 2 and rate == 48000
    assert "the late part traced at 2 of them" in capsys.readouterr().err
    assert main([*args, "--late-step", "0"]) == 0
    assert "traced at 3 of them" in capsys.readouterr().err
    assert main(["inspect", str(path)]) == 0
    assert "walks 2.0 m to (3.00, 1.00, 1.50) in 1.0 s" in capsys.readouterr().out
    assert main(["walkthrough", str(path), "x", "--out", str(out)]) == 1
    assert "expected SOURCE=WAV" in capsys.readouterr().err
    # A video it could not save is reported before simulating.
    assert main([*args, "--video", str(tmp_path / "walk.avi")]) == 1
    err = capsys.readouterr().err
    assert "gif or .mp4" in err and "traced" not in err
