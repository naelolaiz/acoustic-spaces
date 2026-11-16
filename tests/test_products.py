import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from acoustic_spaces.products import add_product, dot, dots, product


def _complex(rng, *shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _forms(rng) -> dict:
    """Operands (a, b, conj_b) of every form the products take, among them those of the
    call sites they serve (numbered as in the audit of the products that can thread)."""
    weights = rng.standard_normal((256, 4))
    windows = sliding_window_view(rng.standard_normal(5000), 33)
    pressure = _complex(rng, 40, 8)
    return {
        "1-D @ 1-D": (rng.standard_normal(50), rng.standard_normal(50), False),
        "1-D @ 2-D": (rng.standard_normal(7), _complex(rng, 7, 9), False),
        "2-D @ 1-D": (_complex(rng, 9, 7), rng.standard_normal(7), False),
        "transposed a (#1)": (weights[:200].T, _complex(rng, 200, 300), False),
        "column slice a (#3)": (rng.random((60, 8))[:, 2:], rng.random((8, 70))[2:], False),
        "transposed a, strided vector (#9)": (
            _complex(rng, 63, 700).T,
            rng.random((63, 3))[:, 1],
            False,
        ),
        "rows @ vector (#10)": (_complex(rng, 500, 12), rng.random(12), False),
        "overlapping windows, step 7 (#24)": (windows[3:4000:7], rng.standard_normal(33), False),
        "overlapping windows, step 1": (windows[:300], rng.standard_normal(33), False),
        "b conjugated, transposed (#7)": (pressure, pressure.T, True),
        "Fortran-ordered rows @ conjugated vector": (
            np.asfortranarray(_complex(rng, 30, 8)),
            _complex(rng, 8),
            True,
        ),
        "real @ conjugated complex": (rng.random((5, 7)), _complex(rng, 7, 4), True),
        "real vector @ conjugated complex vector": (rng.random(7), _complex(rng, 7), True),
        "Fortran-ordered b": (rng.random((7, 9)), np.asfortranarray(rng.random((9, 11))), False),
        "negative strides": (_complex(rng, 6, 5)[::-1], rng.random((5, 8))[:, ::-1], False),
        "stack @ matrix (#14)": (rng.random((3, 5, 6)), rng.random((6, 4)), False),
        "matrix @ stack (#12, #21)": (rng.random((5, 6)), _complex(rng, 3, 6, 4), False),
        "stacks that broadcast": (rng.random((2, 1, 5, 6)), _complex(rng, 3, 6, 4), False),
        "1-D @ stack": (rng.random(6), rng.random((3, 6, 4)), False),
        "integers": (  # some above 2**24, which float32 would round
            (np.arange(12).reshape(3, 4) - 5) * (2**40 + 1),
            np.arange(8, dtype=np.uint8).reshape(4, 2),
            False,
        ),
        "float32 and complex64": (
            rng.random((3, 4)).astype(np.float32),
            _complex(rng, 4, 5).astype(np.complex64),
            False,
        ),
        "lists": ([[1.5, 2.0], [3.0, -4.25]], [0.5, 2.0], False),
        "no terms": (np.zeros((3, 0)), np.zeros((0, 4)), False),
        "stack @ matrix, no terms": (np.ones((2, 3, 0)), np.ones((0, 4)), False),
        "no rows": (np.zeros((0, 3)), _complex(rng, 3, 4), False),
    }


FORMS = list(_forms(np.random.default_rng(0)))


def _as_doubles(x) -> np.ndarray:
    x = np.asarray(x)
    return x.astype(complex if x.dtype.kind == "c" else float)


def _kernel(a, b, out=None, rows=None, conj_b=False) -> np.ndarray:
    """The compiled kernel on C-contiguous float64 / complex128 copies of the operands,
    stack by stack, as matmul broadcasts them: the bits every form must give."""
    from acoustic_spaces import _native

    a, b = _as_doubles(a), _as_doubles(b)
    if conj_b and a.dtype.kind == "f" and b.dtype.kind == "c":  # a @ conj(b) for a real a
        b, conj_b = b.conj(), False
    shape = np.matmul(np.ones(a.shape), np.ones(b.shape)).shape
    a2 = a[None, :] if a.ndim == 1 else a
    b2 = b[:, None] if b.ndim == 1 else b
    batch = np.broadcast_shapes(a2.shape[:-2], b2.shape[:-2])
    a2 = np.broadcast_to(a2, (*batch, *a2.shape[-2:]))
    b2 = np.broadcast_to(b2, (*batch, *b2.shape[-2:]))
    dtype = complex if "c" in (a.dtype.kind, b.dtype.kind) else float
    result = np.empty((*batch, a2.shape[-2], b2.shape[-1]), dtype)
    if out is not None:
        result[...] = np.asarray(out).reshape(result.shape)
    for index in np.ndindex(*batch):
        mine = np.ascontiguousarray(result[index])
        _native.fixed_product(
            np.ascontiguousarray(a2[index]),
            np.ascontiguousarray(b2[index]),
            mine,
            rows,
            conj_b,
            out is not None,
        )
        result[index] = mine
    return result.reshape(shape)


def _same_bits(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.mark.parametrize("form", FORMS)
def test_product_is_matmul_on_every_operand_form(form) -> None:
    """Within rounding of @ on each form (the bound of the kernel's test against @:
    twice gamma = (k + 2) u / (1 - (k + 2) u) times the sum of the terms' magnitudes),
    with @'s shape, float64 or complex128."""
    a, b, conj_b = _forms(np.random.default_rng(0))[form]
    a, b = np.asarray(a), np.asarray(b)
    got = product(a, b, conj_b=conj_b)
    want = _as_doubles(a) @ _as_doubles(b.conj() if conj_b else b)
    assert np.shape(got) == want.shape
    assert np.asarray(got).dtype == (complex if "c" in (a.dtype.kind, b.dtype.kind) else float)
    k = a.shape[-1]
    u = 2.0**-53
    bound = 2.0 * (k + 2) * u / (1.0 - (k + 2) * u) * (np.abs(a) @ np.abs(b))
    assert np.all(np.abs(np.real(got) - np.real(want)) <= bound)
    assert np.all(np.abs(np.imag(got) - np.imag(want)) <= bound)


@pytest.mark.parametrize("form", FORMS)
def test_product_gives_the_kernel_s_bits_in_every_layout(form) -> None:
    """Whatever the strides, the transposed form or the copies taken, and over rows given
    as indices or as a mask: the bits of the kernel on contiguous operands."""
    a, b, conj_b = _forms(np.random.default_rng(0))[form]
    assert _same_bits(product(a, b, conj_b=conj_b), _kernel(a, b, conj_b=conj_b))
    k = np.shape(a)[-1]
    mask = np.random.default_rng(1).random(k) < 0.5
    want = _kernel(a, b, rows=np.flatnonzero(mask), conj_b=conj_b)
    assert _same_bits(product(a, b, rows=mask, conj_b=conj_b), want)
    assert _same_bits(product(a, b, rows=list(np.flatnonzero(mask)), conj_b=conj_b), want)


@pytest.mark.parametrize("form", FORMS)
def test_add_product_adds_into_out_in_place(form) -> None:
    """Added to out's own values in ascending k: the kernel's bits with add; into out
    itself, also where it is a view (here of every other column of a larger array)."""
    a, b, conj_b = _forms(np.random.default_rng(0))[form]
    shape = np.shape(product(a, b))
    dtype = np.asarray(product(a, b)).dtype
    rng = np.random.default_rng(2)
    whole = rng.standard_normal((*shape, 2)).astype(dtype)
    before = whole.copy()
    out = whole[..., 0]
    assert add_product(out, a, b, conj_b=conj_b) is out
    assert _same_bits(out, _kernel(a, b, before[..., 0], conj_b=conj_b))
    assert _same_bits(whole[..., 1], before[..., 1])


def test_add_product_adds_into_a_column_and_a_transposed_out() -> None:
    """The view of #9 (one column of the impulse response's spectra) and a transposed
    out: the same bits as on contiguous copies, the other columns untouched."""
    rng = np.random.default_rng(3)
    spectra, weights = _complex(rng, 63, 700), rng.random((63, 3))
    out = _complex(rng, 1000, 3)
    before = out.copy()
    add_product(out[:700, 1], spectra.T, weights[:, 1])
    assert _same_bits(out[:700, 1], _kernel(spectra.T, weights[:, 1], before[:700, 1]))
    assert _same_bits(out[:, [0, 2]], before[:, [0, 2]])
    assert _same_bits(out[700:], before[700:])
    a, b = rng.random((5, 7)), _complex(rng, 7, 4)
    out = _complex(rng, 4, 5)
    before = out.copy()
    add_product(out.T, a, b)
    assert _same_bits(out.T, _kernel(a, b, before.T))


def test_add_product_reads_operands_that_share_out_s_memory_as_they_were() -> None:
    """out += out @ out, with b a view into out, and on stacks with an operand that is
    another slice of out (reversed, or one matrix broadcast over all): the operands as
    they were before any element of out changed."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((6, 6))
    want = _kernel(x, x, x)
    add_product(x, x, x)
    assert _same_bits(x, want)
    y = _complex(rng, 4, 8)
    want = _kernel(y[:, :4], y[:, 4:], y[:, :4])
    add_product(y[:, :4], y[:, :4], y[:, 4:])
    assert _same_bits(y[:, :4], want)
    z, w = rng.standard_normal((3, 4, 4)), _complex(rng, 3, 4, 4).real
    for operands in (lambda o: (o[::-1], w), lambda o: (o[0], w), lambda o: (w, o[::-1])):
        out = z.copy()
        want = _kernel(*operands(z.copy()), z)
        add_product(out, *operands(out))
        assert _same_bits(out, want)


def test_products_split_among_threads_give_the_same_bits() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((4, 100)), _complex(rng, 100, 333)
    for x in (a, a[:1], a[0]):  # one row: the threads split the columns of one sum each
        one = product(x, b)
        for threads in (0, 2, 3, 8, np.int64(4)):
            assert _same_bits(product(x, b, threads=threads), one)


def _unaligned(x) -> np.ndarray:
    """A copy of x whose data starts one byte past an aligned address."""
    x = np.asarray(x)
    out = np.zeros(x.nbytes + 1, np.uint8)[1:].view(x.dtype).reshape(x.shape)
    out[...] = x
    return out


def _packed(x) -> np.ndarray:
    """x as the field of a packed record array: its strides not whole elements."""
    x = np.asarray(x)
    records = np.zeros(x.shape, dtype=[("x", x.dtype), ("tag", np.int32)])
    records["x"] = x
    return records["x"]


def test_products_take_unaligned_and_packed_operands() -> None:
    """Operands and outs that are not aligned, or whose strides are not whole elements,
    as the compiled core cannot read them: the kernel's bits on contiguous copies."""
    rng = np.random.default_rng(7)
    a, b, c = rng.standard_normal((5, 8)), _complex(rng, 8, 6), rng.standard_normal(8)
    for put in (_unaligned, _packed):
        assert not put(a).flags.aligned or put(a).strides[1] % 8
        assert _same_bits(product(put(a), b), _kernel(a, b))
        assert _same_bits(product(a, put(b)), _kernel(a, b))
        assert _same_bits(product(put(a), put(c)), _kernel(a, c))
        start = _complex(rng, 5, 6)
        out = put(start)
        add_product(out, a, b)
        assert _same_bits(out, _kernel(a, b, start))
        assert dot(put(c), put(c)) == dot(c, c)
        assert dots(put(c), c) == dots(c, c)


# Where long double is wider than double (x86-64 Linux and macOS; on Windows and Apple
# silicon it is a double, and taken).
_LONG_DOUBLE = pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= 52, reason="long double is a double here"
)


@pytest.mark.parametrize(
    "call, error, message",
    [
        (lambda: product(np.ones((2, 2), bool), np.ones((2, 2))), TypeError, "bool"),
        (lambda: product(np.array([["a"]]), np.ones((1, 1))), TypeError, "<U1"),
        pytest.param(
            lambda: product(np.ones((2, 2), np.longdouble), np.ones(2)),
            TypeError,
            "double precision",
            marks=_LONG_DOUBLE,
        ),
        pytest.param(
            lambda: product(np.ones(2), np.ones((2, 2), np.clongdouble)),
            TypeError,
            "double precision",
            marks=_LONG_DOUBLE,
        ),
        (lambda: product(np.ones((2, 3)), np.ones((2, 3))), ValueError, "do not match"),
        (lambda: product(np.ones(3), 2.0), ValueError, "at least one axis"),
        (lambda: product(np.ones((2, 3)), np.ones((3, 2)), rows=[0.5]), TypeError, "rows"),
        (lambda: product(np.ones((2, 3)), np.ones((3, 2)), rows=[True]), ValueError, "mask"),
        (lambda: product(np.ones((2, 3)), np.ones((3, 2)), rows=[2, 1]), ValueError, "ascending"),
        (lambda: product(np.ones((2, 3)), np.ones((3, 2)), rows=[3]), ValueError, "ascending"),
        (lambda: add_product([[0.0]], np.ones((1, 1)), np.ones((1, 1))), TypeError, "numpy"),
        (lambda: add_product(np.zeros(2), np.ones(2), np.ones((2, 2), complex)), TypeError, "out"),
        (lambda: add_product(np.zeros(3), np.ones(2), np.ones((2, 2))), ValueError, "shape"),
        (
            lambda: add_product(np.broadcast_to(0.0, (2,)), np.ones(2), np.ones((2, 2))),
            ValueError,
            "read-only",
        ),
        (
            lambda: product(np.ones((2, 2)), np.ones((2, 2)), threads=-1),
            ValueError,
            "threads must be",
        ),
        (
            lambda: product(np.ones((2, 2)), np.ones((2, 2)), threads=2.0),
            TypeError,
            "threads must be",
        ),
        (
            lambda: add_product(np.ones(2), np.ones(2), np.ones((2, 2)), threads=True),
            TypeError,
            "threads must be",
        ),
        (lambda: dot(np.ones(2, complex), np.ones(2)), TypeError, "real"),
        (lambda: dot(np.ones(2), np.ones(3)), ValueError, "one length"),
        (lambda: dots(np.ones((2, 2)), np.ones((2, 2))), ValueError, "one length"),
    ],
)
def test_products_refuse_what_they_cannot_take(call, error, message) -> None:
    with pytest.raises(error, match=message):
        call()


def test_dot_and_dots_give_the_kernel_s_bits_on_any_vectors() -> None:
    """Strided, integer, single-precision and list vectors: the kernel's bits on their
    contiguous float64 values, within rounding of @ (bound as for product)."""
    from acoustic_spaces import _native

    rng = np.random.default_rng(6)
    x, y = rng.standard_normal(2 * 13_762)[::2], rng.standard_normal(13_762)
    for u, v in [
        (x, y),
        (np.arange(-20, 17), np.arange(37, dtype=np.uint16)),
        (x.astype(np.float32), list(y)),
    ]:
        cu, cv = np.array(u, dtype=float), np.array(v, dtype=float)
        assert dot(u, v) == _native.fixed_dot(cu, cv)
        assert dots(u, v) == _native.fixed_dots(cu, cv)
        bound = 2.0 * len(cu) * 2.0**-53 / (1.0 - len(cu) * 2.0**-53) * float(abs(cu) @ abs(cv))
        assert abs(dot(u, v) - float(cu @ cv)) <= bound
    assert isinstance(dot(x, y), float) and all(isinstance(s, float) for s in dots(x, y))
