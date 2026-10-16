import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import TraceModel, early_paths
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import late as late_stage
from acoustic_spaces.parameters import from_impulse_response
from acoustic_spaces.scene.model import Scattering
from acoustic_spaces.synthesis import REFERENCE_ENERGY_10M

SHOEBOX_OBJ = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "shoebox.obj"
SHOEBOX_YAML = SHOEBOX_OBJ.with_name("scene.yaml")
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
    # Without the scattering of the walls' size, every reflection keeps the same share.
    plain = {"scattering": {"from_size": False}}
    scene = _scene([alpha] * 8, scattering, source=source, listeners=[listener], simulation=plain)
    room = Room.from_scene(scene)
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


def _diffuse_box(simulation: dict, scattering: float = 1.0) -> Room:
    return Room.from_scene(
        _scene(
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3],
            scattering,
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


def test_splitting_leaves_a_diffuse_box_alone(diffuse_result) -> None:
    # Every ray carries about the expected energy here, so the weight window hardly acts,
    # and each ray keeps its random stream with or without it.
    _, split = diffuse_result
    plain = simulate(_diffuse_box({"late": {"rays": 20_000, "splitting": False}}))
    assert 20_000 < split.details["s"]["traced_rays"] < 22_000
    assert plain.details["s"]["traced_rays"] == 20_000
    for key, response in split.responses.items():
        a, b = response.parameters(), plain[key].parameters()
        np.testing.assert_allclose(b.g, a.g, atol=0.05)
        np.testing.assert_allclose(b.t30, a.t30, rtol=0.01)


#: Late energy (dB, 100 ms windows from 0.5 to 2.5 s) at 500 Hz, 1 kHz and 2 kHz in the
#: shoebox example, from 1 million rays with splitting; 2 million without agree within
#: 0.3 to 0.9 dB.
SHOEBOX_TAIL_DB = np.array(
    [
        [-31.1, -33.9, -36.2, -38.9, -41.3, -43.5, -46.1, -48.1, -50.4, -52.8,
         -54.8, -56.8, -58.9, -60.2, -63.5, -65.7, -66.5, -67.9, -70.4, -69.2],
        [-33.6, -36.5, -38.9, -41.6, -44.2, -46.4, -49.2, -51.3, -53.8, -56.2,
         -58.1, -60.2, -62.4, -63.7, -67.3, -69.7, -70.4, -72.4, -74.8, -73.5],
        [-35.9, -39.0, -41.6, -44.4, -47.3, -49.6, -52.8, -54.9, -57.8, -60.4,
         -62.5, -64.7, -66.9, -68.6, -72.4, -75.1, -76.1, -79.2, -81.3, -81.2],
    ]
)  # fmt: skip


def test_splitting_keeps_a_sparse_tail_together() -> None:
    # Absorbing ceiling, smooth walls: late energy rides on the few rays that keep away
    # from the ceiling. Without splitting, 10k rays miss the tail by 5 to 17 dB. (The
    # reference levels are of random scattering without the walls' size, as before 3.0.)
    room = Room.from_yaml(SHOEBOX_YAML)
    room.scene.simulation.scattering = Scattering(method="random", from_size=False)
    late = late_stage.stochastic_energy(
        TraceModel.from_room(room),
        np.array([2.0, 3.0, 1.5]),
        np.array([[6.0, 3.0, 1.2]]),
        rays=10_000,
        max_time=2.5,
        time_step=1e-3,
        speed_of_sound=room.speed_of_sound(),
        air=room.air_attenuation(),
        covered_order=3,
        radius=0.5,
        seed=1,
    )
    windows = late.energy[0].reshape(8, 25, 100).sum(axis=-1)[3:6, 5:]
    error = np.abs(10 * np.log10(windows) - SHOEBOX_TAIL_DB)
    assert np.mean(np.median(error, axis=1)) < 4.0


#: Absorption of the example scene's materials: painted concrete walls, carpet on the
#: floor and an absorbing tile ceiling.
EXAMPLE_ABSORPTION = {
    "walls": [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02],
    "floor": [0.03, 0.03, 0.09, 0.25, 0.31, 0.33, 0.44, 0.44],
    "ceiling": [0.49, 0.49, 0.53, 0.53, 0.75, 0.92, 0.99, 0.99],
}


def _box(absorption: dict[str, list[float]], scatter: float, **simulation) -> Room:
    """The example's 8 x 6 x 3 m box with the given absorption per surface and
    scattering coefficient ``scatter``."""
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": SIZE.tolist()},
                "materials": {
                    name: {"absorption": alpha, "scattering": scatter}
                    for name, alpha in absorption.items()
                },
                "assignments": {name: name for name in absorption},
                "sources": [{"name": "s", "position": [2.0, 3.0, 1.5]}],
                "listeners": [
                    {"name": f"l{k}", "position": p}
                    for k, p in enumerate([[6.0, 3.5, 1.2], [5.0, 1.5, 1.2], [3.5, 4.8, 1.2]])
                ],
                "simulation": {"seed": 3, **simulation},
            }
        )
    )


def _specular_box_decay(absorption: dict[str, list[float]], t: np.ndarray, room: Room):
    """Sound energy left at times ``t`` in the box with specular walls (times, bands).

    A specular ray travelling in direction d meets the walls facing axis k c |d_k| / L_k
    times a second, so each direction decays at its own fixed rate; image sources fill
    space evenly, so a listener receives the average over all directions.
    """
    n = 20_000
    z = 1.0 - (2.0 * np.arange(n) + 1.0) / n
    phi = math.pi * (3.0 - math.sqrt(5.0)) * np.arange(n)
    r = np.sqrt(1.0 - z * z)
    d = np.abs(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    walls, floor, ceiling = (
        -np.log1p(-np.array(absorption[k])) for k in ("walls", "floor", "ceiling")
    )
    # Nepers per metre: half the vertical crossings end on the floor, half on the ceiling.
    rate = np.outer(d[:, 0] / SIZE[0] + d[:, 1] / SIZE[1], walls) + np.outer(
        d[:, 2] / SIZE[2], (floor + ceiling) / 2
    )
    c = room.speed_of_sound()
    left = np.column_stack([np.exp(-c * np.outer(t, band)).mean(axis=1) for band in rate.T])
    return left * np.exp(-c * np.outer(t, room.air_attenuation()))


def _slopes(t: np.ndarray, energy: np.ndarray, start: float, stop: float) -> np.ndarray:
    """Decay rate (dB/s) of each band (columns) between two times."""
    inside = (t >= start) & (t <= stop)
    return np.polyfit(t[inside], 10.0 * np.log10(energy[inside]), 1)[0]


@pytest.mark.parametrize("absorption", ["uniform", "example"])
def test_specular_box_decays_as_its_directions_predict(absorption) -> None:
    # Specular walls make the decay far from Eyring's, but exactly predictable. With the
    # example's absorption, rays parallel to the floor never meet the absorbing ceiling:
    # T30 is many times Eyring's estimate, and the stochastic tail must follow.
    if absorption == "uniform":
        areas = {"walls": 84.0, "floor": 48.0, "ceiling": 48.0}
        mean = sum(a * np.array(EXAMPLE_ABSORPTION[n]) for n, a in areas.items()) / 180.0
        alpha = dict.fromkeys(areas, list(mean))
        length, window, bands = 1.2, (0.15, 1.0), slice(0, 7)  # 8 kHz is gone by 1 s
    else:
        alpha = EXAMPLE_ABSORPTION
        length, window, bands = 2.0, (0.3, 1.8), slice(0, 8)
    # Specular walls: no scattering from their size either.
    room = _box(
        alpha, 0.0, late={"rays": 20_000}, scattering={"from_size": False}, ir_length_s=length
    )
    step = 0.05
    # Every listener sees the same decay; their sum is steadier than each one.
    simulated = sum(response.energy(step) for response in simulate(room)).T[:, bands]
    t = (np.arange(len(simulated)) + 0.5) * step
    expected = _specular_box_decay(alpha, t, room)[:, bands]
    ratio = _slopes(t, simulated, *window) / _slopes(t, expected, *window)
    np.testing.assert_allclose(ratio, 1.0, atol=0.1)


def test_example_box_decays_as_eyring_says_once_its_surfaces_diffuse() -> None:
    # The same absorption as the example, but every reflection diffuse: sound reaches all
    # surfaces alike, and Eyring holds within its own error for uneven absorption (it
    # reads about 5 % long at 63 and 125 Hz, where the ceiling absorbs almost alone).
    room = _box(EXAMPLE_ABSORPTION, 1.0, late={"rays": 20_000}, ir_length_s=2.0)
    response = simulate(room, listeners=["l0"])["s", "l0"]
    ratio = response.parameters().t30 / room.reverberation_time("eyring", air_absorption=True)
    assert np.all((ratio > 0.9) & (ratio < 1.1)), ratio


def test_statistical_tail_follows_eyring() -> None:
    room = _diffuse_box({"late": {"method": "statistical"}})
    result = simulate(room)
    eyring = room.reverberation_time("eyring", air_absorption=True)
    for response in result:
        assert response.transition_s == pytest.approx(
            response.direct_delay_s + math.sqrt(room.volume_m3) * 1e-3
        )
        np.testing.assert_allclose(response.parameters().t30, eyring, rtol=0.03)


@pytest.mark.parametrize("scattering", [0.3, 1.0])
def test_statistical_and_stochastic_methods_agree(scattering) -> None:
    # The shoebox example's box with the same absorption on every surface, so the field
    # is about diffuse. Over both listeners, every band and seeds 3, 5, 7 and 11, the
    # statistical G is within 0.62 dB of the stochastic one at scattering 0.3 (vector-based
    # scattering, with the walls' size; 0.23 dB with random scattering) and within 0.43
    # dB at 1.0 (before its late part held the scattered share of the early reflections
    # and the paths above the early order, it was up to 0.9 and 1.8 dB low). Its C80 is
    # up to 1.3 and 2.2 dB higher, from 1 kHz up, where the tracer decays up to 7 %
    # slower than Eyring; the statistical C80 is Barron's within 0.03 dB. With the
    # example's own materials (absorbing ceiling, smooth walls) the tail at 0.3 is far
    # from diffuse and C80 differs by up to 9 dB at 4 and 8 kHz, beyond any statistical
    # tail.
    results = {
        method: simulate(_diffuse_box({"late": {"method": method, "rays": 20_000}}, scattering))
        for method in ("stochastic_ray_tracing", "statistical")
    }
    for key, response in results["stochastic_ray_tracing"].responses.items():
        statistical = results["statistical"][key]
        a, b = response.parameters(), statistical.parameters()
        np.testing.assert_allclose(b.g, a.g, atol=0.7)
        np.testing.assert_allclose(b.c80, a.c80, atol=2.5)
        assert np.all(statistical.late >= 0)


def test_statistical_energy_keeps_barrons_total_around_early_paths() -> None:
    # A reflection 5 ms after the direct sound brings more than Barron's decay delivers
    # by then: the late part waits until the decay has caught up, and the total stays.
    t60, volume, c, direct = 1.0, 100.0, 343.0, 0.0105
    rate = 6.0 * math.log(10.0) / t60
    total = c / volume / rate * math.exp(-rate * direct)
    delay = np.array([direct, direct + 0.005])
    paths = early_stage.EarlyReflections(
        order=np.array([0, 1]),
        delay_s=delay,
        distance_m=c * delay,
        direction=np.zeros((2, 3)),
        energy=np.array([[1.0], [0.1 * total]]),  # the direct sound is not reflected
    )
    late = late_stage.statistical_energy(
        np.array([t60]),
        volume,
        np.array([direct]),
        [paths],
        max_time=5.0,
        time_step=1e-3,
        speed_of_sound=c,
    ).energy[0, 0]
    caught_up = int((direct - math.log(0.9) / rate) / 1e-3)  # slot where it has 0.1 * total
    assert np.all(late >= 0)
    assert np.all(late[:caught_up] == 0) and late[caught_up + 1] > 0
    assert late.sum() + 0.1 * total == pytest.approx(total, rel=1e-9)


def test_seed_and_threads_give_identical_results() -> None:
    settings = {"late": {"rays": 3000}, "ir_length_s": 0.5}
    a = simulate(_diffuse_box(settings), threads=1)
    b = simulate(_diffuse_box(settings), threads=3)
    for key in a.responses:
        # Each ray has its own random stream; only the order of summation differs.
        np.testing.assert_allclose(a[key].late, b[key].late, rtol=1e-12, atol=0)


def test_performance_settings_leave_results_unchanged() -> None:
    settings = {"late": {"rays": 3000}, "ir_length_s": 0.5}
    room = _diffuse_box(settings)
    assert room.scene.simulation.performance.threads == 0
    assert room.scene.simulation.performance.ray_tree == "sah"
    a = simulate(room)
    b = simulate(_diffuse_box({**settings, "performance": {"threads": 2, "ray_tree": "median"}}))
    for key in a.responses:
        np.testing.assert_allclose(a[key].late, b[key].late, rtol=1e-12, atol=0)


def test_simulate_checks_the_room() -> None:
    scene = _scene([0.1] * 8, 0.1, source=(2, 2, 1), listeners=[(5, 4, 9)])
    with pytest.raises(ValueError, match="outside the room"):
        simulate(scene)
    scene = _scene([0.1] * 8, 0.1, source=(2, 2, 1), listeners=[(5, 4, 1)])
    with pytest.raises(ValueError, match="unknown listener"):
        simulate(scene, listeners=["nobody"])
    with pytest.raises(ValueError, match="no listener"):
        simulate(scene, listeners=[])


def test_a_single_ray_still_traces() -> None:
    # Splitting traces a first batch before the rest; with one ray, that is all of them.
    result = simulate(_diffuse_box({"late": {"rays": 1}, "ir_length_s": 0.2}))
    assert result.details["s"]["rays"] == 1
    for response in result:
        assert np.all(np.isfinite(response.late)) and np.all(response.late >= 0)


def test_particles_spread_reflect_and_fade() -> None:
    from acoustic_spaces.engine import trace_particles
    from acoustic_spaces.room import Room

    room = Room.from_yaml(Path(__file__).resolve().parent.parent / "examples/shoebox/scene.yaml")
    particles = trace_particles(room, count=500, duration=0.1, frames=21, seed=3)
    assert particles.positions.shape == (21, 500, 3)
    lo, hi = np.array(room.report.bounds_min), np.array(room.report.bounds_max)
    assert np.all((particles.positions >= lo - 1e-9) & (particles.positions <= hi + 1e-9))
    # Before any reflection, particles sit on a sphere of radius c t around the source.
    source = np.array(room.scene.sources[0].position)
    first = particles.order[1] == 0
    radius = np.linalg.norm(particles.positions[1, first] - source, axis=1)
    np.testing.assert_allclose(radius, room.speed_of_sound() * particles.times[1])
    assert np.all(np.diff(particles.order, axis=0) >= 0)
    assert np.all(np.diff(particles.energy, axis=0) <= 1e-15)
    assert particles.order[-1].mean() > 5 and particles.energy[-1].mean() < 0.5
    # A line source's points each send their share, on spheres round each of them.
    scene = room.scene.model_copy(deep=True)
    scene.sources[0] = scene.sources[0].model_copy(update={"line_m": 2.0, "spacing_m": 0.5})
    line = trace_particles(Room.from_scene(scene), count=400, duration=0.1, frames=21)
    points = scene.sources[0].elements()
    first = line.order[1] == 0
    owner = np.arange(400)[first] % len(points)
    radius = np.linalg.norm(line.positions[1, first] - points[owner], axis=1)
    np.testing.assert_allclose(radius, room.speed_of_sound() * line.times[1])
    with pytest.raises(ValueError, match="no source 'piano'"):
        trace_particles(room, "piano")
    with pytest.raises(ValueError, match="band"):
        trace_particles(room, band=8)
