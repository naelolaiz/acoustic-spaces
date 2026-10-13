import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import TraceModel, early_paths
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.parameters import from_impulse_response
from acoustic_spaces.synthesis import REFERENCE_ENERGY_10M

SHOEBOX_OBJ = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "shoebox.obj"
SIZE = np.array([8.0, 6.0, 3.0])


def _scene(absorption, scattering, *, source, listeners, simulation=None) -> Scene:
    return Scene.model_validate(
        {
            "geometry": {"file": str(SHOEBOX_OBJ)},
            "materials": {"m": {"absorption": absorption, "scattering": scattering}},
            "default_material": "m",
            "sources": [{"name": "s", "position": source}],
            "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
            "simulation": simulation or {},
        }
    )


def _analytic_images(source: np.ndarray, max_order: int) -> dict[tuple, int]:
    """Image sources of a box [0, L]: position -> reflection order."""
    out = {}
    for n in itertools.product(range(-max_order, max_order + 1), repeat=3):
        for p in itertools.product((0, 1), repeat=3):
            n_, p_ = np.array(n), np.array(p)
            order = int(np.abs(2 * n_ - p_).sum())
            if order <= max_order:
                pos = 2 * n_ * SIZE + (1 - 2 * p_) * source
                out[tuple(np.round(pos, 6))] = order
    return out


@pytest.mark.parametrize("method", ["image_sources", "ray_tracing"])
@pytest.mark.parametrize(
    "source, listener",
    [
        ((2.1, 2.3, 1.7), (5.9, 3.7, 1.2)),
        # Symmetric placement: some paths reflect exactly on edges and corners.
        ((2.0, 3.0, 1.5), (6.0, 3.0, 1.2)),
        ((4.0, 3.0, 1.5), (4.0, 3.0, 1.6)),
    ],
)
def test_early_paths_match_analytic_image_sources(method, source, listener) -> None:
    alpha, scattering = 0.1, 0.2
    room = Room.from_scene(_scene([alpha] * 8, scattering, source=source, listeners=[listener]))
    paths = early_paths(room, max_order=4, method=method)

    src, lis = np.array(source), np.array(listener)
    images = _analytic_images(src, 4)
    assert len(paths) == len(images)  # 1 + 6 + 18 + 38 + 66
    np.testing.assert_array_equal(np.bincount(paths.order), [1, 6, 18, 38, 66])
    c = room.speed_of_sound()
    air = room.air_attenuation()
    for k in range(len(paths)):
        image = lis + paths.direction[k] * paths.distance_m[k]
        key = tuple(np.round(image, 6))
        assert images[key] == paths.order[k]
        distance = np.linalg.norm(np.array(key) - lis)
        assert paths.delay_s[k] == pytest.approx(distance / c, abs=1e-12)
        expected = (
            np.exp(-air * distance)
            / (4 * math.pi * distance**2)
            * ((1 - alpha) * (1 - scattering)) ** paths.order[k]
        )
        np.testing.assert_allclose(paths.energy[k], expected, rtol=1e-9)
        assert len(paths.points[k]) == paths.order[k] + 2


def test_exhaustive_search_has_a_limit() -> None:
    room = Room.from_scene(_scene([0.1] * 8, 0.1, source=(2, 2, 1), listeners=[(5, 4, 1)]))
    model = TraceModel.from_room(room)
    with pytest.raises(ValueError, match="image sources"):
        early_stage.exhaustive_candidates(model, np.array([2.0, 2.0, 1.0]), 8, limit=1000)


def _diffuse_box(simulation: dict) -> Room:
    return Room.from_scene(
        _scene(
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3],
            1.0,
            source=(2.1, 2.3, 1.7),
            listeners=[(5.9, 3.7, 1.2), (6.5, 1.1, 2.1)],
            simulation={"seed": 7, "ir_length_s": 2.5, **simulation},
        )
    )


@pytest.fixture(scope="module")
def diffuse_result():
    room = _diffuse_box({"late": {"rays": 20_000}})
    return room, simulate(room)


def test_decay_matches_eyring_in_a_diffuse_box(diffuse_result) -> None:
    room, result = diffuse_result
    eyring = room.reverberation_time("eyring", air_absorption=True)
    for response in result:
        ratio = response.parameters().t30 / eyring
        # Ray tracing decays a little slower than Eyring as absorption grows (spread of
        # path lengths between reflections, Kuttruff), never faster.
        assert np.all(ratio > 0.97), ratio
        assert np.all(ratio < 1.1), ratio


def test_strength_matches_barron_in_a_diffuse_box(diffuse_result) -> None:
    room, result = diffuse_result
    eyring = room.reverberation_time("eyring", air_absorption=True)
    for response in result:
        r = response.direct_delay_s * room.speed_of_sound()
        # Barron's revised theory, relative to 10 m free field.
        reflected = 31200 * eyring / room.volume_m3 * np.exp(-0.04 * r / eyring)
        barron = 10 * np.log10(100 / r**2 + reflected)
        np.testing.assert_allclose(response.parameters().g, barron, atol=1.0)


def test_impulse_response_carries_the_same_parameters(diffuse_result) -> None:
    _, result = diffuse_result
    response = result["s", "l0"]
    ir = response.impulse_response(48_000, seed=1)
    assert ir.shape == (int(2.5 * 48_000),)
    # The direct sound peaks at its delay with height about 1 / r.
    r = response.direct_delay_s * response.speed_of_sound
    assert np.argmax(np.abs(ir)) == pytest.approx(response.direct_delay_s * 48_000, abs=1)
    p_ir = from_impulse_response(ir, 48_000, REFERENCE_ENERGY_10M)
    p = response.parameters()
    np.testing.assert_allclose(p_ir.g[3:], p.g[3:], atol=1.0)
    np.testing.assert_allclose(p_ir.t30[3:], p.t30[3:], rtol=0.1)
    assert abs(ir).max() == pytest.approx(1 / r, rel=0.3)


def test_transition_time_does_not_change_the_total(diffuse_result) -> None:
    _, by_order = diffuse_result
    by_time = simulate(_diffuse_box({"late": {"rays": 20_000}, "transition_ms": 20.0}))
    for key, response in by_order.responses.items():
        other = by_time[key]
        assert len(other.early) < len(response.early)
        np.testing.assert_allclose(other.parameters().g, response.parameters().g, atol=0.15)


def test_reception_modes_agree_in_a_diffuse_box(diffuse_result) -> None:
    _, rain = diffuse_result
    for reception in ("secondary_sources", "detector"):
        result = simulate(_diffuse_box({"late": {"rays": 20_000, "reception": reception}}))
        for key, response in rain.responses.items():
            a, b = response.parameters(), result[key].parameters()
            np.testing.assert_allclose(b.g, a.g, atol=0.5)
            np.testing.assert_allclose(b.t30, a.t30, rtol=0.08)


def test_statistical_tail_follows_eyring() -> None:
    room = _diffuse_box({"late": {"method": "statistical"}})
    result = simulate(room)
    eyring = room.reverberation_time("eyring", air_absorption=True)
    for response in result:
        assert response.transition_s == pytest.approx(
            response.direct_delay_s + math.sqrt(room.volume_m3) * 1e-3
        )
        np.testing.assert_allclose(response.parameters().t30, eyring, rtol=0.03)


def test_seed_and_threads_give_identical_results() -> None:
    settings = {"late": {"rays": 3000}, "ir_length_s": 0.5}
    a = simulate(_diffuse_box(settings), threads=1)
    b = simulate(_diffuse_box(settings), threads=3)
    for key in a.responses:
        # Each ray has its own random stream; only the order of summation differs.
        np.testing.assert_allclose(a[key].late, b[key].late, rtol=1e-12, atol=0)


def test_simulate_checks_the_room() -> None:
    scene = _scene([0.1] * 8, 0.1, source=(2, 2, 1), listeners=[(5, 4, 9)])
    with pytest.raises(ValueError, match="outside the room"):
        simulate(scene)
    scene = _scene([0.1] * 8, 0.1, source=(2, 2, 1), listeners=[(5, 4, 1)])
    with pytest.raises(ValueError, match="unknown listener"):
        simulate(scene, listeners=["nobody"])
