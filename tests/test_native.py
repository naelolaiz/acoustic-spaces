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
