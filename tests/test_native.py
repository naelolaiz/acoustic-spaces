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


def test_specular_sequences_stay_inside_a_closed_box() -> None:
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    bvh = _native.Bvh(vertices, faces)
    seq = _native.specular_sequences(bvh, np.array([0.3, 0.4, 0.6]), 500, 6, 1)
    assert seq.shape == (500, 6)
    assert seq.min() >= 0 and seq.max() < 12
    # Consecutive hits are never on the same face.
    assert np.all(seq[:, 1:] != seq[:, :-1])


def _trace_stochastic(rays: int, **settings):
    """Stochastic ray tracing in the unit cube, one band, one listener."""
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    args = dict(
        bvh=_native.Bvh(vertices, faces),
        source=np.array([0.3, 0.4, 0.6]),
        directivity_a=1.0,
        forward=np.array([1.0, 0.0, 0.0]),
        absorption=np.full((12, 1), 0.3),
        scattering=np.full((12, 1), 0.5),
        air=np.zeros(1),
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
    )
    energy, _, _, ray_energy, reflections, traced = _native.trace_stochastic(**{**args, **settings})
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
