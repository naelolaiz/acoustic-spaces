from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

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
    STEP_DEG,
    STEP_M,
    WalkPoints,
    WalkResponses,
    crossfade_weights,
    knots,
    late_mixing,
    movers,
    render_movement,
    render_walk,
    save_movement,
    simulate_movement,
    simulate_walk,
    walk_points,
    with_moving,
)

SHOEBOX_OBJ = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "shoebox.obj"
GALLERY = Path(__file__).resolve().parent.parent / "examples" / "gallery" / "scene.yaml"
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
    # Along +x, then along +y, turning through the corner at 90 degrees per second: over
    # one second, 1.5 m, but no further than half-way along the second leg, 0.5 m.
    np.testing.assert_allclose(points.yaws[:7], 0.0)
    np.testing.assert_allclose(points.yaws[6:11], [0.0, 22.5, 45.0, 67.5, 90.0])
    np.testing.assert_allclose(points.yaws[10:], 90.0)
    still = scene.listeners[0].model_copy(
        update={"walk": scene.listeners[0].walk.model_copy(update={"face_forward": False})}
    )
    assert walk_points(still).yaws is None
    # A step that does not divide the path: a little shorter, the same distance apart.
    uneven = walk_points(scene.listeners[0], 0.7)
    assert len(uneven.times) == 6
    np.testing.assert_allclose(np.diff(uneven.times), 3.0 / 5 / 1.5)


def test_a_walker_turns_the_shorter_way_round_at_its_pace() -> None:
    # From 135 degrees to -135: through 180, a quarter turn, not three quarters.
    walk = {"to": [[2.0, 2.0, 1.5], [1.0, 1.0, 1.5]], "speed_m_s": 1.0, "turn_deg_s": 180}
    sources = [{"name": "s", "position": [6, 4, 1.5]}]
    scene = _scene(sources=sources, listeners=[_walker(walk, position=(3.0, 1.0, 1.5))])
    points = walk_points(scene.listeners[0], 0.25)
    assert len(points.times) == 13  # the corner is place 6
    np.testing.assert_allclose(points.yaws[[0, 6, 12]], [135.0, 180.0, -135.0])
    assert np.all(np.abs(np.diff(np.unwrap(np.radians(points.yaws)))) <= np.radians(45.0))
    # The turn takes half a second, a quarter of the way along each leg either side.
    turning = np.flatnonzero(~np.isclose(np.abs(points.yaws), 135.0))
    assert points.times[turning[-1]] - points.times[turning[0]] < 0.5


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
    nowhere = _scene(sources=sources, listeners=[_walker({"to": [[1.0, 1.0, 1.5]] * 2})])
    with pytest.raises(ValueError, match="the walk of 'w' goes nowhere"):
        walk_points(nowhere.listeners[0])


def test_keyframes_move_and_turn_evenly_between_them() -> None:
    """A walk along keyframes goes from the walker's position to each keyframe by its
    time, moving and turning evenly between them: it faces as a keyframe says, turning
    the shorter way round, or as before where a keyframe says nothing (face_forward has
    no say), and it may turn or wait on the spot. Its places along a leg are every
    step_m and every step_deg of its turn, and at every keyframe."""
    frames = [
        {"time_s": 2.0, "position": [3.0, 1.0, 1.5]},
        {"time_s": 3.0, "position": [3.0, 1.0, 1.5], "orientation": {"yaw": 170.0}},
        {"time_s": 5.0, "position": [3.0, 3.0, 1.5], "orientation": {"yaw": -170.0}},
    ]
    walker = _walker({"keyframes": frames}, orientation={"yaw": 90.0})
    scene = _scene(sources=[{"name": "s", "position": [6, 4, 1.5]}], listeners=[walker])
    item = scene.listeners[0]
    path = knots(item)
    np.testing.assert_allclose(path.times, [0.0, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(path.positions[0], [1.0, 1.0, 1.5])
    assert path.duration_s == 5.0 and path.length_m == pytest.approx(4.0)
    assert path.top_speed == pytest.approx(1.0)
    position, angles = path.at(1.0)  # half-way along, facing as it did
    np.testing.assert_allclose(position, [2.0, 1.0, 1.5])
    np.testing.assert_allclose(angles, [90.0, 0.0, 0.0])
    position, angles = path.at(2.5)  # on the spot, half-way round from 90 to 170
    np.testing.assert_allclose(position, [3.0, 1.0, 1.5])
    np.testing.assert_allclose(angles, [130.0, 0.0, 0.0])
    position, angles = path.at(4.0)  # half-way up, from 170 to -170 the shorter way round
    np.testing.assert_allclose(position, [3.0, 2.0, 1.5])
    np.testing.assert_allclose(angles, [180.0, 0.0, 0.0])
    np.testing.assert_allclose(path.at(-1.0)[0], [1.0, 1.0, 1.5])  # at its ends outside
    np.testing.assert_allclose(path.at(9.0)[1], [-170.0, 0.0, 0.0])
    positions, angles = path.at([0.0, 5.0])
    assert positions.shape == (2, 3) and angles.shape == (2, 3)
    points = walk_points(item, 0.5, 20.0)  # 80 degrees on the spot: 4 steps of 20
    expected = [0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]
    np.testing.assert_allclose(points.times, expected)
    assert points.angles.shape == (13, 3)
    np.testing.assert_allclose(points.yaws, points.angles[:, 0])
    np.testing.assert_allclose(points.yaws[4:9], [90.0, 110.0, 130.0, 150.0, 170.0])
    assert walk_points(item).times.size == len(walk_points(item, STEP_M, STEP_DEG).times)
    from acoustic_spaces.scene import Keyframe, Walk

    waiting = [
        Keyframe(time_s=4.0, position=(1, 1, 1.5)),
        Keyframe(time_s=5.0, position=(1, 2, 1.5)),
    ]
    paused = item.model_copy(update={"walk": Walk(keyframes=waiting)})
    np.testing.assert_allclose(walk_points(paused, 0.5).times, [0.0, 4.0, 4.5, 5.0])  # none waiting
    # The schema: one kind of path, keyframes in order of time, after the start.
    with pytest.raises(ValidationError, match="not both"):
        Walk(to=[(1.0, 1.0, 1.0)], keyframes=[Keyframe(time_s=1.0, position=(1.0, 1.0, 1.0))])
    with pytest.raises(ValidationError, match="in order of time"):
        Walk(keyframes=[Keyframe(time_s=2.0, position=(1.0, 1.0, 1.0))] * 2)
    with pytest.raises(ValidationError, match="greater than 0"):
        Keyframe(time_s=0.0, position=(1.0, 1.0, 1.0))
    with pytest.raises(ValidationError, match="at least 1"):
        Walk(keyframes=[])
    assert Walk(keyframes=[Keyframe(time_s=1.0, position=(1.0, 1.0, 1.0))]).enabled
    # A walk along keyframes is simulated place by place as one along to is.
    room = Room.from_scene(scene)
    walked = simulate_walk(room, step_m=0.5, late_step_m=2.0)
    assert walked.names[5] == "w #5 @ 2.25 s" and len(walked.names) == 13
    turned = walked.result.room.scene.listeners[6]
    assert (turned.orientation.yaw, turned.position) == (130.0, (3.0, 1.0, 1.5))
    # The places on the spot take the late part of the one traced there.
    assert walked.late_places == (0, 4, 12)
    out = render_walk(walked, {"s": np.ones(100)})
    assert np.isfinite(out).all() and np.abs(out).max() > 0


def test_a_run_picks_which_walks_move() -> None:
    """with_moving turns the walks it names on and every other one off, for one run of
    the same scene; None keeps the scene's own, and a name without a walk is refused."""
    scene = Room.from_yaml(GALLERY).scene
    assert with_moving(scene, None) is scene
    still = with_moving(scene, [])
    assert movers(still) == [] and still.listeners[2].walk.to == scene.listeners[2].walk.to
    assert [item.name for _, item in movers(with_moving(still, ["visitor"]))] == ["visitor"]
    with pytest.raises(ValueError, match="'in sight' has no walk to move along"):
        with_moving(scene, ["in sight"])


def test_several_movers_on_one_clock() -> None:
    """simulate_movement puts every mover (a walk that is enabled, in use) at each
    moment of its walk as a copy, one per distinct place, in one simulation with the
    still sources and listeners; render_movement plays the signals through it as heard
    at every listener in use, a still pair through its one response; save_movement
    writes a WAV per listener with one gain for all."""
    sources = [
        {
            "name": "s",
            "position": [7.0, 5.0, 1.5],
            "walk": {"to": [[7.0, 2.0, 1.5]], "speed_m_s": 1.0},
        },
        {"name": "still", "position": [1.0, 5.0, 1.0], "level_db": 100.0},
    ]
    frames = [
        {"time_s": 2.0, "position": [3.0, 1.0, 1.5]},
        {"time_s": 3.0, "position": [3.0, 1.0, 1.5], "orientation": {"yaw": 90.0}},
    ]
    listeners = [
        _walker({"keyframes": frames}),
        {"name": "seat", "position": [5.0, 4.0, 1.2]},
        {
            "name": "off",
            "position": [6.0, 4.0, 1.2],
            "walk": {"to": [[6.0, 1.0, 1.2]], "enabled": False},
        },
    ]
    room = Room.from_scene(_scene(sources=sources, listeners=listeners))
    assert [(kind, item.name) for kind, item in movers(room.scene)] == [
        ("listeners", "w"),
        ("sources", "s"),
    ]
    movement = simulate_movement(room, step_m=1.0, step_deg=45.0, late_step_m=2.0)
    # The moments: w every metre, then every 45 degrees on the spot; s every metre.
    np.testing.assert_allclose(movement.times, [0.0, 1.0, 2.0, 2.5, 3.0])
    assert movement.duration_s == 3.0 and movement.places == 10
    names = movement.copies["listeners", "w"]
    assert len(names) == 5 and names[0] == "w #0 @ 0.00 s" and names[3] == "w #3 @ 2.50 s"
    assert movement.key("s", "w", 3) == ("s #3 @ 2.50 s", "w #3 @ 2.50 s")
    assert movement.key("still", "seat", 4) == ("still", "seat")
    assert movement.moves("s", "seat") and not movement.moves("still", "seat")
    keys = set(movement.result.responses)
    assert {listener for _, listener in keys} == {"seat", "off", *names}
    assert {source for source, _ in keys} == {"still", *movement.copies["sources", "s"]}
    assert len(keys) == 6 * 7
    # The late part every 2 m along each walk: 2 of w's 5 places, 3 of s's.
    assert movement.late_places == 5
    # Every listener in use hears the movement, through the same length.
    rate = room.scene.simulation.sample_rate
    out = render_movement(movement, {"s": np.ones(100), "still": np.ones(100)}, Mono())
    assert list(out) == ["w", "seat", "off"]
    assert all(x.shape == out["w"].shape and np.isfinite(x).all() for x in out.values())
    assert all(np.abs(x).max() > 0 for x in out.values())
    # A still pair plays through its one response, the source's power applied once.
    alone = render_movement(movement, {"still": np.ones(100)}, Mono())
    seed = room.scene.simulation.seed
    ir = movement.result["still", "seat"].spatial_impulse_response(Mono(), rate, seed)
    plain = convolve(np.ones(100), ir)
    np.testing.assert_allclose(alone["seat"][: len(plain)], plain)
    assert not alone["seat"][len(plain) :].any()
    # Powers: s (94 dB) plays 6 dB below still (100 dB) unless equal_power.
    quiet = render_movement(movement, {"s": np.ones(100), "still": np.zeros(100)}, Mono())
    equal = render_movement(
        movement, {"s": np.ones(100), "still": np.zeros(100)}, Mono(), equal_power=True
    )
    np.testing.assert_allclose(quiet["seat"], 10.0 ** (-6.0 / 20.0) * equal["seat"])
    with pytest.raises(ValueError, match="no source 'x'"):
        render_movement(movement, {"x": np.ones(10)})
    with pytest.raises(ValueError, match="no signals"):
        render_movement(movement, {})
    # Nothing moves, or Doppler beside another mover, is refused.
    stills = Room.from_scene(_scene(sources=sources[1:], listeners=listeners[1:]))
    with pytest.raises(ValueError, match="nothing moves"):
        simulate_movement(stills)
    fast = {**sources[0], "walk": {**sources[0]["walk"], "doppler": True}}
    with pytest.raises(ValueError, match="'s' walks with Doppler, which moves alone: leave w"):
        simulate_movement(Room.from_scene(_scene(sources=[fast, sources[1]], listeners=listeners)))
    with pytest.raises(ValueError, match=r"leaves the room at \(8\.\d\d, 1\.00, 1\.50\)"):
        gone = [_walker({"to": [[9.0, 1.0, 1.5]]}), *listeners[1:]]
        simulate_movement(Room.from_scene(_scene(sources=sources, listeners=gone)))
    with pytest.raises(ValueError, match="step_m must be positive"):
        simulate_movement(room, step_m=0.0)


def test_a_walk_with_doppler_moves_alone_on_the_clock() -> None:
    """A walk with Doppler moving alone plays in a movement as in a walk of its own: at
    each listener, the same places, each early path through its changing delay and the
    late part crossfaded, so render_movement gives what render_walk does; a listener
    that the walk does not reach hears it too, from the same simulation."""
    from acoustic_spaces.spatial import Binaural

    car = {
        "name": "car",
        "position": [1.0, 4.0, 1.0],
        "walk": {"to": [[7.0, 4.0, 1.0]], "speed_m_s": 20.0, "doppler": True},
    }
    seat = {"name": "seat", "position": [4.0, 1.0, 1.2], "orientation": {"yaw": 90.0}}
    room = Room.from_scene(_scene(sources=[car], listeners=[seat]))
    movement = simulate_movement(room, step_m=0.5)
    assert movement.doppler
    walked = simulate_walk(room, "car", step_m=0.5)
    np.testing.assert_allclose(movement.times, walked.points.times)
    signal = np.random.default_rng(4).standard_normal(int(0.3 * room.scene.simulation.sample_rate))
    for fmt in (Mono(), Binaural()):
        moved = render_movement(movement, {"car": signal}, fmt)["seat"]
        np.testing.assert_allclose(moved, render_walk(walked, {"car": signal}, fmt), atol=1e-12)
    # Facing +y, the seat hears the car start on its left (x < 4) and end on its right.
    ears = render_movement(movement, {"car": signal}, Binaural())["seat"]
    third = len(signal) // 3
    start, end = (ears[:third] ** 2).sum(axis=0), (ears[2 * third : 3 * third] ** 2).sum(axis=0)
    assert start[0] > 1.5 * start[1] and end[1] > 1.2 * end[0]
    # Faster than sound, or with the wave solver, as a walk of its own.
    supersonic = {**car, "walk": {**car["walk"], "speed_m_s": 400.0}}
    with pytest.raises(ValueError, match="as fast as sound"):
        simulate_movement(Room.from_scene(_scene(sources=[supersonic], listeners=[seat])))


def test_save_movement_writes_a_file_per_listener(tmp_path: Path) -> None:
    from acoustic_spaces.audio import read_wav

    rendered = {"w": np.full((10, 1), 0.5), "a seat": np.full((10, 1), -0.25)}
    paths, gain = save_movement(rendered, tmp_path / "out" / "walk.wav", 1000, normalize=True)
    assert paths == {
        "w": tmp_path / "out" / "walk_w.wav",
        "a seat": tmp_path / "out" / "walk_a_seat.wav",
    }
    assert gain == pytest.approx(10.0 ** (-1.0 / 20.0) / 0.5)
    loud, rate = read_wav(paths["w"])
    assert rate == 1000 and np.abs(loud).max() == pytest.approx(10.0 ** (-1.0 / 20.0), abs=1e-6)
    assert np.abs(read_wav(paths["a seat"])[0]).max() == pytest.approx(0.5 * 10 ** (-1 / 20), 1e-5)
    paths, gain = save_movement({"w": rendered["w"]}, tmp_path / "one.wav", 1000)
    assert paths == {"w": tmp_path / "one.wav"} and gain == 1.0
    # Names that would give one file, also where a file system ignores case.
    alike = {"A/B": rendered["w"], "A_B": rendered["a seat"], "a_b": rendered["w"]}
    paths, _ = save_movement(alike, tmp_path / "alike" / "walk.wav", 1000)
    assert [p.name for p in paths.values()] == ["walk_A_B.wav", "walk_A_B-2.wav", "walk_a_b-3.wav"]
    assert len(list((tmp_path / "alike").iterdir())) == 3
    with pytest.raises(ValueError, match="nothing rendered"):
        save_movement({}, tmp_path / "none.wav", 1000)


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
    # Places where a traced one is (a walker turning on the spot) take its late part.
    mixing = late_mixing(np.array([0.0, 0.0, 1.0, 2.0, 2.0, 2.0]), 2.0)
    assert mixing == {1: {0: 1.0}, 2: {0: 0.5, 3: 0.5}, 4: {3: 1.0}, 5: {3: 1.0}}


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
        assert name == f"w #{k} @ {walked.points.times[k]:.2f} s"
        distance = np.linalg.norm(walked.points.positions[k] - [7.0, 5.0, 1.5])
        assert walked.result["s", name].direct_delay_s == pytest.approx(distance / c)
    every = simulate_walk(room, step_m=0.5, late_step_m=None)
    assert every.late_places == tuple(range(15))
    # The statistical method gives every place its own late part.
    statistical = room.scene.simulation.model_copy(deep=True)
    statistical.late.method = "statistical"
    assert simulate_walk(room, step_m=0.5, settings=statistical).late_places == tuple(range(15))


def test_places_less_than_10_ms_apart_have_names_of_their_own() -> None:
    # 2 m at 30 m/s in steps of 0.25 m: 8.3 ms apart, two places at 0.03 s.
    walk = {"to": [[3.0, 1.0, 1.5]], "speed_m_s": 30.0}
    sources = [{"name": "s", "position": [7.0, 5.0, 1.5]}]
    room = Room.from_scene(_scene(sources=sources, listeners=[_walker(walk)]))
    walked = simulate_walk(room, late_step_m=1.0)
    assert len(set(walked.names)) == len(walked.names) == 9
    assert walked.late_places == (0, 4, 8) and len(walked.result) == 9


def test_a_walking_source_is_heard_at_a_listener() -> None:
    sources = [
        {"name": "w", "position": [1.0, 1.0, 1.5], "walk": {"to": [[3.0, 1.0, 1.5]]}},
        {"name": "still", "position": [7.0, 5.0, 1.5], "level_db": 100.0},
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
    # The walker (94 dB, under its own name at every place) plays 6 dB below "still".
    alone = {"w": np.ones(100), "still": np.zeros(100)}
    equal = render_walk(walked, alone, equal_power=True)
    np.testing.assert_allclose(render_walk(walked, alone), 10.0 ** (-6.0 / 20.0) * equal)
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
    with pytest.raises(ValueError, match=r"more than one walks \(w, v\): pick one, or play them"):
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
    # No scene sources: every signal plays at the one power.
    room: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(scene=SimpleNamespace(sources=[]))
    )

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
    walker = next(lis for lis in room.scene.listeners if lis.name == "visitor")
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
    # --hrtf is for binaural output: the format without --format, refused with another.
    hrtf = ["--hrtf", str(tmp_path / "missing.sofa")]
    assert main([*args, *hrtf]) == 1
    err = capsys.readouterr().err
    assert "error: " in err and "traced" not in err
    assert main([*args, "--format", "mono", *hrtf]) == 1
    assert "error: --hrtf is for binaural output, not mono" in capsys.readouterr().err
    assert main(["inspect", str(path)]) == 0
    assert "walks 2.0 m to (3.00, 1.00, 1.50) in 1.0 s" in capsys.readouterr().out
    assert main(["walkthrough", str(path), "x", "--out", str(out)]) == 1
    assert "expected SOURCE=WAV" in capsys.readouterr().err
    # A video it could not save is reported before simulating.
    assert main([*args, "--video", str(tmp_path / "walk.avi")]) == 1
    err = capsys.readouterr().err
    assert "gif or .mp4" in err and "traced" not in err


def test_cli_walkthrough_of_every_mover(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``walkthrough --all`` plays every walk that is enabled on one clock and writes a
    WAV per listener in use, or the one of ``--listener``; ``inspect`` describes a walk
    along keyframes and one that is off."""
    from acoustic_spaces.audio import read_wav, write_wav

    frames = [
        {"time_s": 1.0, "position": [3.0, 1.0, 1.5]},
        {"time_s": 1.5, "position": [3.0, 1.0, 1.5], "orientation": {"yaw": 90.0}},
    ]
    scene = _scene(
        sources=[
            {
                "name": "s",
                "position": [7.0, 5.0, 1.5],
                "walk": {"to": [[7.0, 3.0, 1.5]], "speed_m_s": 1.0},
            },
            {"name": "t", "position": [1.0, 5.0, 1.0]},
        ],
        listeners=[
            _walker({"keyframes": frames}),
            {"name": "seat", "position": [5.0, 4.0, 1.2]},
            {
                "name": "v",
                "position": [6.0, 4.0, 1.2],
                "walk": {"to": [[6, 1, 1.2]], "enabled": False},
            },
        ],
    )
    path = tmp_path / "scene.yaml"
    scene.to_yaml(path)
    dry = tmp_path / "dry.wav"
    write_wav(dry, np.random.default_rng(0).standard_normal((8000, 1)) * 0.1, 8000)
    out = tmp_path / "all" / "walk.wav"
    args = ["walkthrough", str(path), f"s={dry}", f"t={dry}", "--out", str(out), "--step", "1"]
    assert main([*args, "--all", "--turn-step", "45", "--normalize"]) == 0
    captured = capsys.readouterr()
    assert "w, s move for 2.0 s: responses at 11 places, the late part traced at 4" in captured.err
    files = {name: out.with_name(f"walk_{name}.wav") for name in ("w", "seat", "v")}
    assert all(f"wrote {file}: {name}," in captured.out for name, file in files.items())
    assert not out.exists()
    data = {name: read_wav(file) for name, file in files.items()}
    assert all(rate == 48000 and x.shape[1] == 1 for x, rate in data.values())
    assert max(np.abs(x).max() for x, _ in data.values()) == pytest.approx(10 ** (-1 / 20), 1e-3)
    assert main([*args, "--all", "--listener", "seat"]) == 0
    assert read_wav(out)[0].shape == data["seat"][0].shape
    # Without --all, two walks are too many for one walkthrough; --levels is for one.
    assert main(args) == 1
    assert (
        "more than one walks (w, s): pick one, or play them together (--all"
        in capsys.readouterr().err
    )
    assert main([*args, "--walker", "v"]) == 1
    assert "the walk of 'v' is off (enabled: false)" in capsys.readouterr().err
    assert main([*args, "--all", "--levels", str(tmp_path / "levels.csv")]) == 1
    assert "--levels is for one walk" in capsys.readouterr().err
    assert main([*args, "--all", "--listener", "nobody"]) == 1
    assert "no listener named 'nobody' in use; the scene has w, seat, v" in capsys.readouterr().err
    assert main(["inspect", str(path)]) == 0
    text = capsys.readouterr().out
    assert "moves through 2 keyframe(s) 2.0 m to (3.00, 1.00, 1.50) in 1.5 s" in text
    assert "walks 3.0 m to (6.00, 1.00, 1.20) in 2.5 s (the walk is off: it stands still)" in text


def test_a_walk_takes_the_threads_of_the_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from acoustic_spaces import walkthrough
    from acoustic_spaces.audio import write_wav

    asked = []

    def stop(room, *, threads, **_):
        asked.append(threads)
        raise ValueError("stopped")

    monkeypatch.setattr(walkthrough, "simulate", stop)
    scene = _scene(
        sources=[{"name": "s", "position": [7.0, 5.0, 1.5]}],
        listeners=[_walker({"to": [[3.0, 1.0, 1.5]]})],
    )
    with pytest.raises(ValueError, match="stopped"):
        simulate_walk(Room.from_scene(scene))
    path = tmp_path / "scene.yaml"
    scene.to_yaml(path)
    dry = tmp_path / "dry.wav"
    write_wav(dry, np.zeros((480, 1)), 48_000)
    args = ["walkthrough", str(path), f"s={dry}", "--out", str(tmp_path / "walk.wav")]
    assert main(args) == 1
    assert main([*args, "--threads", "2"]) == 1
    assert "error: stopped" in capsys.readouterr().err
    # None: simulate takes the scene's simulation.performance.threads, as for simulate.
    assert asked == [None, None, 2]
    with pytest.raises(SystemExit):
        main([*args, "--threads", "-1"])
    assert "--threads" in capsys.readouterr().err
