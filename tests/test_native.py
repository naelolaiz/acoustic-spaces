import math

import numpy as np
import pytest

from acoustic_spaces import native_available
from acoustic_spaces.spatial.directions import fibonacci_sphere


def test_native_module_imports() -> None:
    assert native_available(), "compiled core failed to build or install"


@pytest.mark.parametrize("n, m", [(1, 1), (8, 3), (257, 64)])
def test_convolve_direct_matches_numpy(n: int, m: int) -> None:
    from acoustic_spaces import _native

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n)
    ir = rng.standard_normal(m)

    out = _native.convolve_direct(signal, ir)

    assert out.shape == (n + m - 1,)
    np.testing.assert_allclose(out, np.convolve(signal, ir), rtol=1e-12, atol=1e-12)


def test_convolve_direct_rejects_empty() -> None:
    from acoustic_spaces import _native

    with pytest.raises(ValueError):
        _native.convolve_direct(np.zeros(0), np.ones(4))


def _unit_cube():
    vertices = np.array(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)], dtype=float
    )
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)
    return vertices, faces


def test_bvh_intersect_hits_misses_and_ignores() -> None:
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    bvh = _native.Bvh(vertices, faces)
    assert bvh.face_count == 12
    origins = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0], [0.5, 0.5, 0.0]])
    directions = np.array([[1.0, 0, 0], [0, 0, -1.0], [1.0, 0, 0], [0, 0, 1.0]])
    t, face = bvh.intersect(origins, directions, np.full(4, 10.0), np.full(4, -1))
    np.testing.assert_allclose(t[:2], 0.5)
    assert face[2] == -1 and np.isinf(t[2])
    # Starting on the floor and ignoring nothing: the floor itself is too close to count.
    assert t[3] == pytest.approx(1.0)
    # t_max cuts hits beyond it.
    t, face = bvh.intersect(origins[:1], directions[:1], np.array([0.4]), np.array([-1]))
    assert face[0] == -1


def test_bvh_ray_through_an_edge_hits() -> None:
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    bvh = _native.Bvh(vertices, faces)
    # Straight at the vertical edge x = 1, y = 1, and through the cube's corner vertex.
    origins = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    directions = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]) / np.sqrt([[2.0], [3.0]])
    t, face = bvh.intersect(origins, directions, np.full(2, 10.0), np.full(2, -1))
    assert np.all(face >= 0)
    np.testing.assert_allclose(t, [np.sqrt(0.5), np.sqrt(0.75)])


def cube_with_seats(rng: np.random.Generator):
    """The unit cube with 500 small triangles scattered inside, like seats in a hall."""
    vertices, faces = _unit_cube()
    corners = rng.uniform(0.1, 0.9, (500, 1, 3)) + rng.uniform(-0.02, 0.02, (500, 3, 3))
    vertices = np.vstack([vertices, corners.reshape(-1, 3)])
    faces = np.vstack([faces, 8 + np.arange(1500, dtype=np.int64).reshape(-1, 3)])
    return vertices, faces


def test_bvh_splits_find_the_same_hits() -> None:
    from acoustic_spaces import _native

    rng = np.random.default_rng(3)
    vertices, faces = cube_with_seats(rng)
    sah, median = (_native.Bvh(vertices, faces, split=s) for s in ("sah", "median"))
    assert sah.face_count == median.face_count == 512
    n = 5000
    origins = rng.uniform(0.0, 1.0, (n, 3))
    directions = rng.standard_normal((n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    args = (origins, directions, np.full(n, 10.0), np.full(n, -1, dtype=np.int64))
    t_sah, face_sah = sah.intersect(*args)
    t_median, face_median = median.intersect(*args)
    assert np.all(face_sah >= 0)  # inside a closed box
    # The same distances; a face may differ only where two meet at that point.
    np.testing.assert_array_equal(t_sah, t_median)
    assert np.mean(face_sah == face_median) > 0.999
    # No limit at all finds the same (a missed box must not pass for one entered then).
    t_open, face_open = sah.intersect(origins, directions, np.full(n, np.inf), args[3])
    np.testing.assert_array_equal(t_open, t_sah)
    np.testing.assert_array_equal(face_open, face_sah)
    with pytest.raises(ValueError, match="split"):
        _native.Bvh(vertices, faces, split="octree")


def test_bvh_layout_holds_the_tree_and_every_face_once() -> None:
    from acoustic_spaces import _native

    vertices, faces = cube_with_seats(np.random.default_rng(4))
    bvh = _native.Bvh(vertices, faces)
    assert bvh.scale == pytest.approx(np.sqrt(3.0))
    layout = bvh.layout()
    boxes = layout["boxes"].reshape(-1, 6)
    first, count = layout["first"], layout["count"]
    assert len(boxes) == len(first) == len(count)
    # Every face sits in one leaf, and the triangles are the mesh's, in leaf order.
    np.testing.assert_array_equal(np.sort(layout["faces"]), np.arange(len(faces)))
    assert count.sum() == len(faces)
    edges = layout["triangles"].reshape(-1, 3, 3)
    corners = vertices[faces[layout["faces"]]]
    np.testing.assert_allclose(edges[:, 0], corners[:, 0])
    np.testing.assert_allclose(edges[:, 1], corners[:, 1] - corners[:, 0], atol=1e-12)
    np.testing.assert_allclose(edges[:, 2], corners[:, 2] - corners[:, 0], atol=1e-12)
    # A leaf's triangles lie in its box; an inner node's children lie in its box.
    lo, hi = boxes[:, :3], boxes[:, 3:]
    for node in np.flatnonzero(count > 0):
        inside = corners[first[node] : first[node] + count[node]].reshape(-1, 3)
        assert np.all(inside >= lo[node]) and np.all(inside <= hi[node])
    for node in np.flatnonzero(count == 0):
        children = [first[node], first[node] + 1]
        assert np.all(lo[children] >= lo[node]) and np.all(hi[children] <= hi[node])


def test_bvh_occluded_agrees_with_the_nearest_hit() -> None:
    from acoustic_spaces import _native

    rng = np.random.default_rng(5)
    vertices, faces = cube_with_seats(rng)
    bvh = _native.Bvh(vertices, faces)
    n = 3000
    origins = rng.uniform(0.0, 1.0, (n, 3))
    targets = rng.uniform(0.0, 1.0, (n, 3))
    distance = np.linalg.norm(targets - origins, axis=1)
    directions = (targets - origins) / distance[:, None]
    ignore = rng.integers(-1, len(faces), n)
    t, _ = bvh.intersect(origins, directions, distance, ignore)
    blocked = bvh.occluded(origins, directions, distance, ignore)
    assert blocked.dtype == bool and 0 < blocked.sum() < n
    np.testing.assert_array_equal(blocked, np.isfinite(t))
    with pytest.raises(ValueError, match="same length"):
        bvh.occluded(origins, directions, distance[:-1], ignore)


def test_specular_sequences_stay_inside_a_closed_box() -> None:
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    bvh = _native.Bvh(vertices, faces)
    origin = np.array([0.3, 0.4, 0.6])
    seq = _native.specular_sequences(bvh, origin, 500, 6, 1, np.zeros(0, np.uint8))
    assert seq.shape == (500, 6)
    assert seq.min() >= 0 and seq.max() < 12
    # Consecutive hits are never on the same face.
    assert np.all(seq[:, 1:] != seq[:, :-1])


def test_specular_sequences_keep_every_branch_of_a_ray() -> None:
    from acoustic_spaces import _native

    # Three nested boxes that reflect and let sound through, inside one that only
    # reflects: a branch of the one ray forks at each of their faces it meets until it
    # has passed through MAX_FORKS of them, giving C(order + MAX_FORKS, MAX_FORKS) rows,
    # and every one is kept.
    vertices, faces = _unit_cube()
    sizes = (1.0, 2.0, 3.0, 50.0)
    bvh = _native.Bvh(
        np.vstack([(2.0 * vertices - 1.0) * s for s in sizes]),
        np.vstack([faces + 8 * k for k in range(len(sizes))]),
    )
    passing = np.ones(12 * len(sizes), np.uint8)
    passing[-12:] = 0
    for order in (3, 10):
        rows = _native.specular_sequences(bvh, np.array([0.1, 0.2, 0.3]), 1, order, 1, passing)
        assert len(rows) == math.comb(order + _native.MAX_FORKS, _native.MAX_FORKS)


def _stochastic_args(rays: int = 2000) -> dict:
    """The tracer's arguments for the unit cube, one band, one listener."""
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    return dict(
        bvh=_native.Bvh(vertices, faces),
        source=np.array([0.3, 0.4, 0.6]),
        directivity_a=np.zeros(0),
        orientation=np.eye(3),
        directivity_table=np.zeros((0, 0, 1)),
        absorption=np.full((12, 1), 0.3),
        scattering=np.full((12, 1), 0.5),
        air=np.zeros(1),
        surface_size=np.zeros((0, 2)),
        band_hz=np.zeros(0),
        vector_scattering=False,
        size_from_order=1,
        transmission=np.zeros((0, 0)),
        face_room=np.zeros((0, 2), dtype=np.int64),
        source_room=-1,
        receivers=np.array([[0.7, 0.6, 0.4]]),
        directions=np.zeros((0, 3)),
        rays=rays,
        max_time=0.05,
        time_step=1e-3,
        dir_time_step=1e-2,
        speed_of_sound=343.0,
        receiver_radius=np.array([0.2]),
        diffuse_rain=True,
        specular_rain=False,
        covered_order=-1,
        covered_time=np.zeros(0),
        seed=1,
        threads=0,
        window_time_step=0.01,
        first_ray=0,
        split_target=np.zeros(0),
        split_ratio=4.0,
        max_split=8,
        max_family=256,
        room_level=np.zeros(0),
        face_group=np.zeros(0, dtype=np.int64),
        groups=0,
        moment_time_step=0.01,
    )


def _trace_stochastic(rays: int, **settings):
    """Stochastic ray tracing in the unit cube, one band, one listener."""
    from acoustic_spaces import _native

    energy, _, _, _, ray_energy, reflections, traced = _native.trace_stochastic(
        **{**_stochastic_args(rays), **settings}
    )
    return energy[0, 0], ray_energy / reflections, traced


def test_weight_window_keeps_the_expected_energy() -> None:
    reference, mean, traced = _trace_stochastic(100_000)
    assert traced == 100_000
    # Far below the rays' energy every ray splits until its family is full; far above,
    # nearly all are thinned out by roulette. Either way the energy stays the same.
    for rays, scale, families, tolerance in ((500, 1e-2, 64, 0.12), (50_000, 20.0, 256, 0.06)):
        target = mean * 100_000 / rays * scale
        energy, _, traced = _trace_stochastic(rays, split_target=target, max_family=families)
        assert energy.sum() == pytest.approx(reference.sum(), rel=tolerance)
        if scale < 1:
            assert traced == rays * families
        else:
            assert traced < 1.1 * rays


def test_a_vector_reflections_specular_share_goes_where_the_specular_paths_go() -> None:
    # A 2 m square reflector in free field, the source 2 m in front of it: each ray
    # reflects once at most. Random scattering sends a reflection's specular share
    # exactly specular, so a listener out of the reflection's specular reach hears the
    # direct sound and the scattered share (the rain) alone. Vector-based scattering
    # counts that share on a leg of its own along the specular direction, not on its
    # blended ray (bent towards the normal, which once carried it there): from the same
    # rays the listener hears the same. In reach, the specular share is the same on
    # average.
    from acoustic_spaces import _native

    vertices = np.array([[0.0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    # Out of reach at y = 2.5 (the specular paths reach |y| < 2 at x = 2), and in reach.
    listeners = np.array([[2.0, 2.5, 0.0], [3.0, 0.5, 0.0]])

    def energy(vector: bool) -> np.ndarray:
        args = _stochastic_args(200_000) | {
            "bvh": _native.Bvh(vertices, faces),
            "source": np.array([2.0, 0.0, 0.0]),
            "absorption": np.full((2, 1), 0.2),
            "scattering": np.full((2, 1), 0.5),
            "vector_scattering": vector,
            "receivers": listeners,
            "receiver_radius": np.array([0.3, 0.3]),
            "threads": 2,
        }
        return _native.trace_stochastic(**args)[0][:, 0].sum(axis=1)

    vector, random = energy(True), energy(False)
    assert vector[0] == pytest.approx(random[0], rel=1e-9)
    assert vector[1] == pytest.approx(random[1], rel=0.005)


def test_the_tracer_counts_reflections_off_face_groups() -> None:
    # Two groups: the cube's floor and ceiling, and its walls. Each arrival is summed
    # times each group's reflection count and their products, per bin of 10 ms.
    from acoustic_spaces import _native

    groups = np.array([1] * 8 + [0] * 4, dtype=np.int64)
    args = {"face_group": groups, "groups": 2, "max_time": 0.05}
    energy, _, moments, *_ = _native.trace_stochastic(**{**_stochastic_args(), **args})
    assert moments.shape == (1, 1 + 2 + 3, 1, 5)
    plain = _native.trace_stochastic(**{**_stochastic_args(), "max_time": 0.05})[0]
    np.testing.assert_array_equal(energy, plain)  # counting changes nothing
    total, n0, n1, n00, n01, n11 = moments[0, :, 0]
    np.testing.assert_allclose(total, energy[0, 0].reshape(5, 10).sum(axis=1), rtol=1e-12)
    # Means grow with time (a reflection every 0.67 m in a unit cube); variances >= 0.
    mean = (n0 + n1) / total
    assert np.all(np.diff(mean) > 0) and mean[-1] > 15
    assert np.all(n00 * total >= n0**2 * (1 - 1e-12)) and np.all(n11 * total >= n1**2 * (1 - 1e-12))
    # The walls are twice the floor and ceiling: twice the reflections.
    assert n1.sum() / n0.sum() == pytest.approx(2.0, rel=0.05)
    assert n01.sum() > 0
    with pytest.raises(ValueError, match="face_group"):
        _native.trace_stochastic(**{**_stochastic_args(), "face_group": groups[:5], "groups": 2})
    with pytest.raises(ValueError, match="face_group"):
        _native.trace_stochastic(**{**_stochastic_args(), "face_group": groups + 1, "groups": 2})


@pytest.mark.parametrize(
    "setting", [{"split_ratio": 1.0}, {"max_split": 1}, {"max_family": 0}, {"window_time_step": 0}]
)
def test_weight_window_rejects_bad_settings(setting) -> None:
    with pytest.raises(ValueError, match="bad settings"):
        _trace_stochastic(10, **setting)


def test_scatter_kernels_places_filters_at_impulses() -> None:
    from acoustic_spaces import _native

    rng = np.random.default_rng(1)
    kernels = rng.standard_normal((3, 2, 5))
    positions = np.array([0, 4, 4, 18], dtype=np.int64)
    index = np.array([2, 0, 1, 2], dtype=np.int64)
    amplitudes = np.array([1.0, -0.5, 2.0, 3.0])
    out = _native.scatter_kernels(20, positions, index, amplitudes, kernels)
    want = np.zeros((2, 25))
    for p, i, a in zip(positions, index, amplitudes, strict=True):
        want[:, p : p + 5] += a * kernels[i]
    np.testing.assert_allclose(out, want[:, :20])  # past the end is dropped
    with pytest.raises(ValueError, match="out of range"):
        _native.scatter_kernels(20, positions, index + 1, amplitudes, kernels)


def _trace_map(rays: int, **changes):
    """Rays from near the floor of the unit cube onto one map slab higher up; walls
    that absorb everything, so only the direct sound reaches it."""
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    args = dict(
        bvh=_native.Bvh(vertices, faces),
        source=np.array([0.5, 0.5, 0.2]),
        directivity_a=np.zeros(0),
        orientation=np.eye(3),
        directivity_table=np.zeros((0, 0, 1)),
        absorption=np.ones((12, 1)),
        scattering=np.zeros((12, 1)),
        air=np.zeros(1),
        surface_size=np.zeros((0, 2)),
        band_hz=np.zeros(0),
        vector_scattering=False,
        size_from_order=1,
        transmission=np.zeros((0, 0)),
        face_room=np.zeros((0, 2), dtype=np.int64),
        source_room=-1,
        origin=np.array([0.4, 0.4]),
        cell=0.2,
        point=np.zeros((1, 1, 1), dtype=np.int64),
        low=np.array([0.7]),
        high=np.array([0.8]),
        volume=np.array([0.2 * 0.2 * 0.1]),
        map_room=np.full(1, -1, dtype=np.int64),
        onset=np.array([0.5 / 343.0]),
        covered_time=np.zeros(1),
        fine_step=1e-4,
        fine_bins=20,
        coarse_step=1e-3,
        rays=rays,
        max_time=0.01,
        speed_of_sound=343.0,
        covered_order=-1,
        seed=5,
        threads=0,
        window_time_step=0.01,
        first_ray=0,
        split_target=np.zeros(0),
        split_ratio=4.0,
        max_split=8,
        max_family=256,
        room_level=np.zeros(0),
    )
    return _native.trace_map(**{**args, **changes})


def test_rays_out_of_an_open_model_escape() -> None:
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    top = np.all(vertices[faces][:, :, 2] == 1.0, axis=1)
    walls = len(faces) - int(top.sum())
    changes = {
        "bvh": _native.Bvh(vertices, faces[~top]),
        "absorption": np.ones((walls, 1)),
        "scattering": np.zeros((walls, 1)),
    }
    _, escaped, _, _, traced = _trace_map(200_000, **changes)
    # The share of directions through the open top: its solid angle from the source,
    # 0.8 m below the middle of the 1 m square, over 4 pi.
    share = math.asin(0.25 / (0.25 + 0.64)) / math.pi
    assert traced == 200_000
    assert escaped / traced == pytest.approx(share, rel=0.03)


def test_trace_map_counts_the_track_length_through_a_slab() -> None:
    energy, escaped, _, _, traced = _trace_map(400_000)
    assert energy.shape == (1, 1, 20 + 8)
    assert traced == 400_000 and escaped == 0
    # A track-length estimate of the direct sound: 1 / (4 pi r^2) averaged over the slab.
    g = (np.arange(20) + 0.5) / 20
    x, y, z = np.meshgrid(0.4 + 0.2 * g, 0.4 + 0.2 * g, 0.7 + 0.1 * g, indexing="ij")
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.2) ** 2
    assert energy.sum() == pytest.approx(np.mean(1 / (4 * np.pi * r2)), rel=0.04)
    # Each crossing counts at its middle: 0.5 m to 0.6 m away, so within 0.35 ms of the
    # onset at 0.5 m, mostly in the second 0.1 ms bin.
    arrived = np.flatnonzero(energy[0, 0])
    assert arrived.min() == 0 and arrived.max() <= 3
    assert np.argmax(energy[0, 0]) == 1


def test_trace_map_skips_what_the_early_part_covers() -> None:
    covered, *_ = _trace_map(20_000, covered_order=0, covered_time=np.array([1.0]))
    assert covered.sum() == 0.0
    after, *_ = _trace_map(20_000, covered_order=0, covered_time=np.array([0.56 / 343.0]))
    assert 0 < after.sum() < _trace_map(20_000)[0].sum()


def test_trace_map_counts_the_rays_of_a_points_room_alone() -> None:
    # Every ray stays in the source's room 0: a point in room 1 gets none of them, one in
    # room 0 or in any (-1) all of them, ray for ray.
    anywhere, *_ = _trace_map(20_000, source_room=0)
    own, *_ = _trace_map(20_000, source_room=0, map_room=np.zeros(1, dtype=np.int64))
    other, *_ = _trace_map(20_000, source_room=0, map_room=np.ones(1, dtype=np.int64))
    assert anywhere.sum() > 0.0 and other.sum() == 0.0
    np.testing.assert_array_equal(own, anywhere)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"origin": np.zeros(3)}, "origin must be"),
        ({"map_room": np.zeros(2, dtype=np.int64)}, "bad map"),
        ({"low": np.zeros(0), "high": np.zeros(0), "volume": np.zeros(0)}, "no map points"),
        ({"volume": np.zeros(1)}, "bad map"),
        ({"high": np.array([0.7])}, "bad map"),
        ({"point": np.full((1, 1, 1), 3, dtype=np.int64)}, "bad map"),
        ({"cell": 0.0}, "bad map"),
    ],
)
def test_trace_map_rejects_a_bad_map(change, message) -> None:
    with pytest.raises(ValueError, match=message):
        _trace_map(10, **change)


def test_a_worker_that_fails_raises_instead_of_ending_the_process() -> None:
    """An exception that left a worker thread ended the process (std::terminate)."""
    from acoustic_spaces import _native

    for threads, at in ((4, 0), (4, 999), (1, 3)):
        with pytest.raises(RuntimeError, match="a worker failed"):
            _native._parallel_for_raises(1_000, threads, at)
    _native._parallel_for_raises(1_000, 4, 1_000)  # past the last item: none fails


def _hard_arrivals(directions: np.ndarray, rng, random: int) -> np.ndarray:
    """Arrivals that test the nearest direction's table: ``random`` random ones at
    lengths from 1e-30 to 1e30, the cube map's cell edges (u or v a multiple of 1/4 on
    each face), the cube's edges and corners, the directions themselves and their
    opposites, the points halfway between two directions (ties) or three, and those
    moved by a few units in the last place (near-ties), all at several lengths."""
    arrivals = [rng.standard_normal((random, 3)) * 10.0 ** rng.uniform(-30, 30, (random, 1))]
    grid = np.linspace(-1.0, 1.0, 9)
    u, v = np.meshgrid(grid, rng.uniform(-1.0, 1.0, 40))
    edges = np.meshgrid(grid, grid)
    for a, b in ((u, v), (v, u), edges):
        flat = np.column_stack([np.ones(a.size), a.reshape(-1), b.reshape(-1)])
        for axis in range(3):
            for sign in (1.0, -1.0):
                arrivals.append(np.roll(flat * [sign, 1.0, 1.0], axis, axis=1))
    corners = np.array(np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]))
    arrivals.append(corners.reshape(3, -1).T)
    arrivals += [directions, -directions]
    # Pairs and triples of near directions (all pairs of a small set).
    n = len(directions)
    near = np.argsort(-(directions @ directions.T), axis=1)[:, 1:4] if n > 1 else np.zeros((1, 0))
    for k in range(near.shape[1]):
        arrivals.append(directions + directions[near[:, k]])
    if near.shape[1] >= 2:
        arrivals.append(directions + directions[near[:, 0]] + directions[near[:, 1]])
    if n <= 64:
        i, j = np.triu_indices(n, 1)
        arrivals.append(directions[i] + directions[j])
    hard = np.vstack(arrivals[1:])
    moved = [hard * s for s in (1.0, 3.0, 1e-12, 1e12)]
    for steps in (1, 2, 5):
        moved.append(np.nextafter(hard, rng.choice([-np.inf, np.inf], hard.shape)))
        moved.append(hard + 2.0**-52 * steps * rng.standard_normal(hard.shape) * np.abs(hard))
    return np.ascontiguousarray(np.vstack([arrivals[0], *moved]))


def _direction_sets(rng) -> dict:
    ring = np.column_stack(
        [np.cos(np.arange(12) * math.pi / 6), np.sin(np.arange(12) * math.pi / 6)]
    )
    axes = np.vstack([np.eye(3), -np.eye(3)])
    cube = np.array([p for p in np.ndindex(3, 3, 3) if p != (1, 1, 1)], dtype=float) - 1.0
    random = rng.standard_normal((64, 3))
    return {
        "fibonacci 64": fibonacci_sphere(64),
        "random 64": random / np.linalg.norm(random, axis=1, keepdims=True),
        "one": fibonacci_sphere(1),
        "two": fibonacci_sphere(2),
        "three": fibonacci_sphere(3),
        "ring in xy": np.column_stack([ring, np.zeros(12)]),
        "ring in yz": np.column_stack([np.zeros(12), ring]),
        "axes": axes,
        "cube's 26": cube / np.linalg.norm(cube, axis=1, keepdims=True),
        "twice the same 32": np.vstack([fibonacci_sphere(32), fibonacci_sphere(32)]),
        "64 the same": np.tile([[0.6, 0.0, 0.8]], (64, 1)),
        "lengths 0.5 to 2": fibonacci_sphere(64) * rng.uniform(0.5, 2.0, (64, 1)),
        "long": fibonacci_sphere(64) * 1e10,
        "short": fibonacci_sphere(64) * 1e-10,
        "3000": fibonacci_sphere(3000),
    }


def test_the_nearest_directions_table_finds_what_a_scan_of_all_does() -> None:
    """The tracer's nearest direction to each arrival through its cube map (the first of
    the strictly largest dot product) is the one a scan of all directions finds, for
    every arrival tried: random, on the map's cell edges and the cube's edges and
    corners, between directions (ties) and next to that (near-ties), at any length."""
    from acoustic_spaces import _native

    rng = np.random.default_rng(7)
    checked = 0
    for name, directions in _direction_sets(rng).items():
        directions = np.ascontiguousarray(directions)
        arrivals = _hard_arrivals(directions, rng, 2_000_000 if name == "fibonacci 64" else 100_000)
        if len(directions) > 1000:
            arrivals = arrivals[:: max(1, len(arrivals) // 200_000)]
        looked, scanned, served, kept = _native._nearest_directions(arrivals, directions)
        assert np.array_equal(looked, scanned), name
        assert 1 <= kept <= 4
        checked += len(arrivals)
        # The table serves all of these finite arrivals within its lengths (2^-200 to
        # 2^200), except where it is not made: for one or two directions and for 64 the
        # same, whose lists would hold most of them.
        within = np.all(np.isfinite(arrivals), axis=1) & (np.abs(arrivals).max(axis=1) >= 2.0**-200)
        if name in ("one", "two", "64 the same"):
            assert not served.any(), name
        else:
            assert np.array_equal(served == 1, within), name
    assert checked > 4_000_000


def _angles(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The angles between the rows of p and of q, by atan2 as the core's table takes them."""
    return np.arctan2(np.linalg.norm(np.cross(p, q), axis=-1), np.sum(p * q, axis=-1))


def _table_slack(directions: np.ndarray) -> tuple:
    """Per cube map cell (6 x 8 x 8, numbered as the core's DirectionTable numbers them)
    and per direction, how far its upper bound U = |d| cos(max(0, theta - rho')) lies above
    the bar the table lists by, the largest lower bound L = |d| cos(min(pi, theta + rho'))
    less 1e-9 N (rho' the cell's corner radius plus 1e-9, N the longest direction's
    length); and each cell's centre (unit), rho' and largest L."""
    side, cells = 0.25, np.arange(384)
    face, u0, v0 = cells // 64, -1.0 + side * ((cells // 8) % 8), -1.0 + side * (cells % 8)

    def on_face(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        s = np.where(face % 2 == 0, 1.0, -1.0)
        out = np.column_stack([s, u, v])
        out[face // 2 == 1] = out[face // 2 == 1][:, [1, 0, 2]]
        out[face // 2 == 2] = out[face // 2 == 2][:, [1, 2, 0]]
        return out

    centre = on_face(u0 + 0.5 * side, v0 + 0.5 * side)
    centre /= np.linalg.norm(centre, axis=1, keepdims=True)
    rho = np.max(
        [_angles(centre, on_face(u0 + du, v0 + dv)) for du in (0, side) for dv in (0, side)],
        axis=0,
    )
    rho = rho + 1e-9
    length = np.linalg.norm(directions, axis=1)
    theta = _angles(centre[:, None, :], directions[None, :, :])
    low = length * np.cos(np.minimum(math.pi, theta + rho[:, None]))
    high = length * np.cos(np.maximum(0.0, theta - rho[:, None]))
    best_low = low.max(axis=1)
    slack = high - (best_low - 1e-9 * length.max())[:, None]
    return slack, centre, rho, best_low


def test_the_nearest_directions_table_lists_each_direction_its_margins_take() -> None:
    """Each cell's list of the nearest directions' table holds every direction whose upper
    bound is at least the cell's largest lower bound less 1e-9 N, with the cell's radius
    taken 1e-9 rad wider (the rule its comment argues from), and no direction far below
    that. A direction placed only 0.25e-9 N above that bar, at 1.2 rad past the widened
    radius, must be listed: a table made without either margin, or with a cell radius
    short by more than about 2e-9 rad, leaves it out."""
    from acoustic_spaces import _native

    rng = np.random.default_rng(11)
    random = rng.standard_normal((64, 3))
    for directions in (
        fibonacci_sphere(64),
        random / np.linalg.norm(random, axis=1, keepdims=True),
        fibonacci_sphere(64) * rng.uniform(0.5, 2.0, (64, 1)),
    ):
        start, index = _native._nearest_directions_lists(np.ascontiguousarray(directions))
        assert len(start) == 385
        slack = _table_slack(directions)[0]
        for cell in range(384):
            listed = np.zeros(len(directions), dtype=bool)
            listed[index[start[cell] : start[cell + 1]]] = True
            assert np.all(listed[slack[cell] > 1e-12]), cell
            assert np.all(slack[cell][listed] >= -1e-12), cell
            assert np.all(np.diff(index[start[cell] : start[cell + 1]]) > 0), cell

    directions = fibonacci_sphere(64)
    _, centre, rho, best_low = _table_slack(directions)
    turn = 1.2
    for cell in range(0, 384, 7):
        c = centre[cell]
        across = np.cross(c, [0.0, 0.0, 1.0] if abs(c[2]) < 0.9 else [1.0, 0.0, 0.0])
        across /= np.linalg.norm(across)
        angle = rho[cell] + turn
        # Its upper bound is its length times cos(turn); its length is N.
        length = best_low[cell] / (math.cos(turn) + 0.75e-9)
        placed = np.vstack([directions, length * (math.cos(angle) * c + math.sin(angle) * across)])
        slack = _table_slack(placed)[0][cell]
        assert 0.2e-9 * length < slack[64] < 0.3e-9 * length
        start, index = _native._nearest_directions_lists(placed)
        assert len(start) == 385
        assert 64 in index[start[cell] : start[cell + 1]], cell


def test_the_nearest_directions_table_falls_back_to_a_scan_of_all() -> None:
    """Arrivals that are zero, NaN, infinite, too long or too short, and directions with
    no table (NaN, infinite or zero ones, too many), take the scan of all directions."""
    from acoustic_spaces import _native

    directions = fibonacci_sphere(64)
    with np.errstate(over="ignore", invalid="ignore"):
        arrivals = np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0],
                [np.nan, 0.5, 0.2],
                [0.5, np.nan, 0.2],
                [0.1, 0.2, np.nan],
                [np.inf, 0.0, 0.0],
                [1.0, -np.inf, 0.0],
                [np.inf, np.inf, -np.inf],
                [1e300, 1e-300, 3e299],
                [5e-324, 0.0, -5e-324],
                [1e-70, -3e-70, 2e-70],
            ]
        )
    looked, scanned, served, _ = _native._nearest_directions(arrivals, directions)
    assert np.array_equal(looked, scanned)
    assert not served.any()
    rng = np.random.default_rng(3)
    arrivals = rng.standard_normal((20_000, 3))
    for change in (np.nan, np.inf, 0.0):
        odd = directions.copy()
        odd[17] = change
        looked, scanned, served, _ = _native._nearest_directions(arrivals, odd)
        assert np.array_equal(looked, scanned) and not served.any()
    for many in (fibonacci_sphere(4096), fibonacci_sphere(4097)):
        looked, scanned, served, _ = _native._nearest_directions(arrivals[:2_000], many)
        assert np.array_equal(looked, scanned)
        assert served.all() == (len(many) <= 4096)
    with pytest.raises(ValueError, match="at least one direction"):
        _native._nearest_directions(arrivals[:2], np.zeros((0, 3)))


def test_the_nearest_directions_tables_are_shared_between_threads_and_bounded() -> None:
    """Python threads that trace at once with directions of their own (the binding lets
    go of the GIL) each find their own nearest directions, while the tables they share
    are made, kept and dropped, four at most."""
    from concurrent.futures import ThreadPoolExecutor

    from acoustic_spaces import _native

    rng = np.random.default_rng(5)
    sets = [fibonacci_sphere(n) for n in (16, 32, 48, 64, 80, 96, 112, 128)]
    arrivals = rng.standard_normal((20_000, 3))

    def run(k: int) -> bool:
        same = True
        for turn in range(12):
            directions = sets[(k + turn) % len(sets)]
            looked, scanned, served, kept = _native._nearest_directions(arrivals, directions)
            same &= bool(np.array_equal(looked, scanned) and served.all() and kept <= 4)
        return same

    with ThreadPoolExecutor(8) as pool:
        assert all(pool.map(run, range(8)))


def test_the_nearest_directions_table_leaves_the_tracer_s_results_as_a_scan_of_all() -> None:
    """The tracer with 64 directions (through their table) sums each arrival where it does
    with the same directions and a zero one after them, which no arrival is nearer to and
    which leaves them without a table (a scan of all): every result the same."""
    from acoustic_spaces import _native

    directions = fibonacci_sphere(64)
    args = _stochastic_args(3_000) | {
        "absorption": np.full((12, 1), 0.1),
        "max_time": 0.2,
        "vector_scattering": True,
        "specular_rain": True,
        "diffuse_rain": False,
        "face_group": np.arange(12, dtype=np.int64) // 4,
        "groups": 3,
    }
    fast = _native.trace_stochastic(**args | {"directions": directions})
    plain = _native.trace_stochastic(**args | {"directions": np.vstack([directions, np.zeros(3)])})
    assert fast[1].shape[1] == 64 and plain[1].shape[1] == 65
    assert fast[1].sum() > 0.0 and not plain[1][:, 64].any()
    np.testing.assert_array_equal(fast[1], plain[1][:, :64])
    for a, b in zip(fast[:1] + fast[2:], plain[:1] + plain[2:], strict=True):
        np.testing.assert_array_equal(a, b)
