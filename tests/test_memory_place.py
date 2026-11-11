"""Whether this computer's linear algebra and functions round by where an array lies in
memory, and whether a fit of an admittance is the same wherever its arrays lie.

For the Intel Mac job, where the fit with its own forward differences was not the fit
with ``least_squares``' own, bit for bit, in three of five cases
(``test_wave.test_the_fit_is_the_one_least_squares_own_differences_give``), while it was
on Linux, on Windows and on an Apple silicon Mac. To be taken out again once the
cause is known.
"""

import numpy as np
import pytest

from acoustic_spaces.engine import impedance
from acoustic_spaces.engine.impedance import fit_admittance

_FITS = [
    ([0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05], 8, None),
    ([0.07, 0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.48], 8, None),
    (np.repeat([0.07, 0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.48], 3).tolist(), 24, None),
    ([0.07, 0.07, 0.31, 0.49], 8, 5512.5),
    ([0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2], 24, 3472.0),
]
#: (bands, frequencies, branches) of those fits.
_SIZES = [(8, 352, 10), (24, 373, 26), (4, 225, 6), (9, 214, 11)]
_OFFSETS = (0, 8, 16, 24, 32, 40, 48, 56)


def _computer() -> str:
    """The NumPy and SciPy of this run, their linear algebra and the processor's
    features NumPy uses."""
    import scipy

    out = [f"numpy {np.__version__}, scipy {scipy.__version__}"]
    try:
        build = np.show_config(mode="dicts")["Build Dependencies"]
        out.append(f"blas {build['blas'].get('name')}, lapack {build['lapack'].get('name')}")
    except Exception as error:
        out.append(f"no build information ({error})")
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features

        wanted = ("FMA3", "AVX2", "AVX512F", "AVX512_SKX", "ASIMD")
        out.append("cpu " + " ".join(f"{k}={features.get(k)}" for k in wanted))
    except Exception as error:
        out.append(f"no processor features ({error})")
    return "; ".join(out)


def _placed(values: np.ndarray, offset: int) -> np.ndarray:
    """A copy of ``values`` (float64 or complex128), in C order, whose data begin
    ``offset`` bytes past a multiple of 64."""
    values = np.asarray(values)
    width = values.dtype.itemsize // 8
    raw = np.empty(values.size * width + 16)
    first = ((-raw.ctypes.data) % 64 + offset) // 8
    out = raw[first : first + values.size * width].view(values.dtype).reshape(values.shape)
    out[...] = values
    assert out.ctypes.data % 64 == offset % 64 and out.flags.c_contiguous
    return out


def _by_place(function, *arrays, place: int) -> str | None:
    """None when ``function`` of the arrays gives the same bits with the array numbered
    ``place`` at each of ``_OFFSETS`` (the others at a multiple of 64), else the groups
    of offsets that give the same."""
    results: dict[bytes, list[int]] = {}
    for offset in _OFFSETS:
        placed = [_placed(a, offset if k == place else 0) for k, a in enumerate(arrays)]
        out = np.ascontiguousarray(function(*placed))
        results.setdefault(out.tobytes(), []).append(offset)
    return None if len(results) == 1 else str(sorted(results.values()))


def test_results_do_not_depend_on_where_an_array_lies() -> None:
    """The products, sums and functions a fit is made of give the same bits wherever in
    memory their arrays begin."""
    rng = np.random.default_rng(11)
    found = []

    def check(name, function, *arrays, places=None):
        for place in range(len(arrays)) if places is None else places:
            groups = _by_place(function, *arrays, place=place)
            if groups is not None:
                shapes = ", ".join(str(a.shape) for a in arrays)
                found.append(f"{name} [{shapes}], array {place} moved: {groups}")

    for _ in range(3):
        for bands, frequencies, branches in _SIZES:
            for unknowns in (1 + branches, 1 + 2 * branches):
                rows = bands + (branches if unknowns > 1 + branches else 0)
                weights = rng.uniform(0.0, 1.0, (bands, frequencies))
                v = rng.uniform(0.0, 1.0, frequencies) * 10.0 ** rng.uniform(-6, 0, frequencies)
                jac = rng.normal(size=(unknowns, rows))  # as the fit lays it, used as jac.T
                f = rng.normal(size=rows) * 1e-3
                p = rng.normal(size=unknowns)
                check("matrix @ vector", lambda w, x: w @ x, weights, v)
                check("jac.T.T.dot(f)", lambda j, x: j.T.T.dot(x), jac, f)
                check("jac.T.dot(p)", lambda j, x: j.T.dot(x), jac, p)
                check("f @ f", lambda x: np.array(x @ x), f)
                check("norm(p)", lambda x: np.array(np.linalg.norm(x)), p)
                check("jac.T * p", lambda j, x: j.T * x, jac, p)
                for name, function in (
                    ("exp", np.exp),
                    ("log", np.log),
                    ("arctan", np.arctan),
                    ("sin", np.sin),
                    ("cos", np.cos),
                    ("sqrt", np.sqrt),
                ):
                    check(name, function, v)
                    check(name + " of the unknowns", function, np.abs(p) + 0.5)
                gamma = rng.uniform(1e-6, 1.0, (branches, 1))
                q = rng.uniform(0.5, 10.0, (branches, 1))
                w = rng.uniform(0.2, 5.0, (1, frequencies))
                denominator = 1.0 + 1j * q * (w - 1.0 / w)
                check("conductance / denominator", lambda g, d: g / d, gamma, denominator)
                check("1j * q * w", lambda a, b: 1j * a * b, q, w - 1.0 / w)
                check("sum of the branches", lambda d: d.sum(axis=0), gamma / denominator)
                check(
                    "numerator * conductance / denominator",
                    lambda n, g, d: n * g / d,
                    1j * np.sin(w),
                    gamma,
                    denominator,
                )
    assert not found, _computer() + "\n" + "\n".join(found)


def test_a_decomposition_is_the_same_whatever_else_is_in_memory() -> None:
    """The singular value decomposition ``least_squares`` takes of the differences gives
    the same bits whatever small arrays were made and kept before it."""
    from scipy.linalg import svd

    rng = np.random.default_rng(12)
    found = []
    for bands, _, branches in _SIZES:
        for unknowns in (1 + branches, 1 + 2 * branches):
            rows = bands + (branches if unknowns > 1 + branches else 0)
            a = rng.normal(size=(rows + unknowns, unknowns))
            kept = []
            for name, function in (
                (
                    "scipy svd",
                    lambda m: np.concatenate(
                        [part.ravel() for part in svd(m, full_matrices=False)]
                    ),
                ),
                (
                    "numpy svd",
                    lambda m: np.concatenate(
                        [part.ravel() for part in np.linalg.svd(m, full_matrices=False)]
                    ),
                ),
                ("m.T.dot(m[:, 0])", lambda m: m.T.dot(np.array(m[:, 0]))),
                ("m.copy().T.dot(m[:, 0])", lambda m: m.copy().T.dot(np.array(m[:, 0]))),
            ):
                results = set()
                for turn in range(24):
                    kept.append(np.empty(1 + (5 * turn) % 23))
                    results.add(function(a).tobytes())
                if len(results) > 1:
                    found.append(f"{name} of {a.shape}: {len(results)} results in 24 turns")
    assert not found, _computer() + "\n" + "\n".join(found)


@pytest.mark.parametrize("own", [True, False], ids=["least_squares' differences", "the fit's"])
@pytest.mark.parametrize(("absorption", "total", "rate"), _FITS)
def test_a_fit_is_the_same_whatever_else_is_in_memory(
    monkeypatch, absorption, total: int, rate, own: bool
) -> None:
    """A fit gives the same bits when small arrays are made and kept while it runs, which
    moves where its own arrays lie: with ``least_squares``' own forward differences and
    with the fit's."""
    import scipy.optimize

    real = scipy.optimize.least_squares
    misses = impedance._Misses.__call__
    slopes = impedance._Misses.slopes

    def with_its_own_differences(fun, x0, jac=None, **options):
        return real(fun, x0, **options)

    def fit(every: int):
        kept = []

        def keeping(function):
            def call(self, x, free):
                if every and len(kept) % every == 0:
                    kept.append(np.empty(1 + (7 * len(kept)) % 29))
                else:
                    kept.append(None)
                return function(self, x, free)

            return call

        with monkeypatch.context() as patch:
            patch.setattr(impedance._Misses, "__call__", keeping(misses))
            patch.setattr(impedance._Misses, "slopes", keeping(slopes))
            if own:
                patch.setattr(scipy.optimize, "least_squares", with_its_own_differences)
            impedance._fit.cache_clear()
            out = fit_admittance(absorption, total, rate)
        impedance._fit.cache_clear()
        return out

    plain = fit(0)
    found = []
    for every in (1, 2, 3, 5):
        other = fit(every)
        for name in ("beta", "conductance", "quality", "frequency_hz", "achieved"):
            a, b = (np.asarray(getattr(v, name), dtype=float) for v in (plain, other))
            if a.shape != b.shape or a.tobytes() != b.tobytes():
                found.append(f"an array kept at every {every}. call: {name} differs")
                break
    assert not found, _computer() + "\n" + "\n".join(found)


def _one_unknown_at_a_time(misses, x: np.ndarray, free: bool) -> np.ndarray:
    """As in ``test_wave``: the misses once more for each unknown, moved by its step."""
    steps = np.diag(impedance._forward_steps(x, misses.lower[: len(x)], misses.upper[: len(x)]))
    at = misses(x, free)
    out = np.empty((len(x), len(at)))
    for k in range(len(x)):
        moved = x + steps[k]
        out[k] = (misses(moved, free) - at) / (moved[k] - x[k])
    return out.T


@pytest.mark.parametrize(("absorption", "total", "rate"), _FITS)
def test_three_ways_to_take_the_differences_give_one_fit(
    monkeypatch, absorption, total: int, rate
) -> None:
    """The fit with ``least_squares``' own forward differences, with the same
    differences taken one unknown at a time by a function given to it, and with the
    fit's own: the same bits. Which of the three part says whether the differences
    differ or what ``least_squares`` makes of the same ones."""
    import scipy.optimize

    real = scipy.optimize.least_squares

    def own(fun, x0, jac=None, **options):
        return real(fun, x0, **options)

    def one_at_a_time(fun, x0, jac=None, **options):
        return real(fun, x0, jac=lambda x, free: _one_unknown_at_a_time(fun, x, free), **options)

    fits = {}
    for name, way in (
        ("least_squares' own", own),
        ("one at a time", one_at_a_time),
        ("the fit's", real),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(scipy.optimize, "least_squares", way)
            impedance._fit.cache_clear()
            fits[name] = fit_admittance(absorption, total, rate)
        impedance._fit.cache_clear()

    def same(a, b) -> bool:
        for name in ("beta", "conductance", "quality", "frequency_hz", "achieved"):
            u, v = (np.asarray(getattr(w, name), dtype=float) for w in (a, b))
            if u.shape != v.shape or u.tobytes() != v.tobytes():
                return False
        return True

    names = list(fits)
    found = [
        f"{a} and {b} differ"
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if not same(fits[a], fits[b])
    ]
    assert not found, _computer() + "\n" + "; ".join(found)
