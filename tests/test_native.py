import math

import numpy as np
import pytest

from acoustic_spaces import native_available


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


@pytest.mark.parametrize(
    "change, message",
    [
        ({"origin": np.zeros(3)}, "origin must be"),
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
