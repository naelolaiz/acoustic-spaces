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
