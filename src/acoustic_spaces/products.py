"""Matrix and dot products whose sums are added in a fixed order.

numpy's ``@`` hands its sums to the linear-algebra library, which splits them among as
many threads as it is given and adds the parts in an order that depends on that number
and on the CPU. The products here are the compiled core's (``fixed_sums.hpp``), which
add each sum in an order its definition fixes, whatever the threads, the CPU, the
compiler or the platform, so a result has the same bits everywhere.

:func:`product` and :func:`add_product`: element ``[i, j]`` of ``a @ b`` is
``a[i, 0] * b[0, j]``, plus ``a[i, 1] * b[1, j]``, and so on for k ascending (or for the
``rows`` given), each term a rounded product added to the sum so far; +0.0 with no
terms; :func:`add_product` starts from ``out[i, j]`` instead. Complex terms are their
real and imaginary parts written out, each product rounded and then added, the parts
summed separately. :func:`dot` adds ``x[i] * y[i]`` in eight sums, one for each
``i mod 8``, and adds the eight in pairs.

The operands are those of ``@``: 1-D or 2-D, or stacks of matrices whose leading axes
broadcast; float64 or complex128, of any strides. Narrower floats and complex numbers
are taken as their (exact) float64 and complex128 values, integers as the nearest
float64 (exact up to 2**53). The result is always float64 or complex128. Booleans,
numbers wider than a double and anything else are refused with a TypeError.

The bits of every result are fixed, but for the sign and payload of a NaN: which of
two NaNs an operation passes on is the hardware's and the compiler's choice, so a NaN
result is only NaN.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

_FLOAT = np.dtype(np.float64)
_COMPLEX = np.dtype(np.complex128)


def product(
    a: ArrayLike,
    b: ArrayLike,
    *,
    rows: ArrayLike | None = None,
    conj_b: bool = False,
    threads: int = 1,
) -> np.ndarray:
    """``a @ b`` (or ``a @ conj(b)`` with ``conj_b``), each element's terms added in
    ascending k from the first; over only the k of ``rows`` (ascending indices into the
    summed axis, or a boolean mask over it) if given. Split among ``threads`` threads
    (an integer, 0: all cores) by the result's columns or rows, with the same bits for
    any number."""
    a, b = _operand(a, "a"), _operand(b, "b")
    shape = _result_shape(a, b)
    out = np.empty(shape, dtype=_COMPLEX if _complex(a, b) else _FLOAT)
    _products(out, a, b, rows, conj_b, False, threads)
    return out[()] if out.ndim == 0 else out


def add_product(
    out: np.ndarray,
    a: ArrayLike,
    b: ArrayLike,
    *,
    rows: ArrayLike | None = None,
    conj_b: bool = False,
    threads: int = 1,
) -> np.ndarray:
    """``out += a @ b`` in place, each element's terms added in ascending k to its
    value in ``out``; ``rows``, ``conj_b`` and ``threads`` as for :func:`product`.
    ``out`` (an array of any strides, or a view into one) has the shape of ``a @ b``
    and is complex128 if ``a`` or ``b`` is complex, float64 otherwise. Returns ``out``."""
    a, b = _operand(a, "a"), _operand(b, "b")
    if not isinstance(out, np.ndarray):
        raise TypeError("add_product: out must be a numpy array")
    want = _COMPLEX if _complex(a, b) else _FLOAT
    if out.dtype != want:
        raise TypeError(f"add_product: out must be {want} for these operands, not {out.dtype}")
    shape = _result_shape(a, b)
    if out.shape != shape:
        raise ValueError(f"add_product: out has shape {out.shape}, a @ b has {shape}")
    if not out.flags.writeable:
        raise ValueError("add_product: out is read-only")
    _products(out, a, b, rows, conj_b, True, threads)
    return out


def dot(x: ArrayLike, y: ArrayLike) -> float:
    """``x @ y`` of two real vectors of one length, in eight sums: sum L adds
    ``x[i] * y[i]`` for i mod 8 == L in ascending i from +0.0; the result is
    ``((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))``."""
    return float(_compiled().fixed_dot(*_vectors(x, y, "dot")))


def dots(x: ArrayLike, y: ArrayLike) -> tuple[float, float, float]:
    """``(x @ x, y @ y, x @ y)``, each as :func:`dot`, in one call."""
    xx, yy, xy = _compiled().fixed_dots(*_vectors(x, y, "dots"))
    return float(xx), float(yy), float(xy)


def _compiled():
    """The compiled core, imported when first called for, as everywhere in the package."""
    from acoustic_spaces import _native

    return _native


def _threads(threads) -> int:
    """``threads`` as a count for the compiled core: an integer, 0 or more."""
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)):
        raise TypeError(f"products: threads must be an integer, not {threads!r}")
    if threads < 0:
        raise ValueError(f"products: threads must be 0 (all cores) or more, not {threads}")
    return int(threads)


def _operand(x: ArrayLike, name: str) -> np.ndarray:
    """``x`` as a float64 or complex128 array, a view of it where it is one already and
    the compiled core can read it (see :func:`_readable`), else a copy."""
    array = np.asarray(x)
    if array.dtype == _FLOAT or array.dtype == _COMPLEX:
        return array if _readable(array) else array.copy()
    kind = array.dtype.kind
    if kind == "c" and np.can_cast(array.dtype, _COMPLEX):
        return array.astype(_COMPLEX, copy=False)
    if kind in "iu" or (kind == "f" and np.can_cast(array.dtype, _FLOAT)):
        return array.astype(_FLOAT, copy=False)
    raise TypeError(
        f"products: {name} must hold integers, floats or complex numbers of at most double "
        f"precision, not {array.dtype}"
    )


def _vectors(x: ArrayLike, y: ArrayLike, name: str):
    """Two real vectors of one length as contiguous float64 arrays."""
    x, y = _operand(x, "x"), _operand(y, "y")
    if x.dtype == _COMPLEX or y.dtype == _COMPLEX:
        raise TypeError(f"{name}: x and y must be real; product takes complex vectors")
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(
            f"{name}: x and y must be vectors of one length, not {x.shape} and {y.shape}"
        )
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _complex(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == _COMPLEX or b.dtype == _COMPLEX


def _result_shape(a: np.ndarray, b: np.ndarray) -> tuple[int, ...]:
    """The shape of ``a @ b`` (ValueError where ``@`` would raise one)."""
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("products: a and b must have at least one axis each, as for @")
    k_a = a.shape[-1]
    k_b = b.shape[0] if b.ndim == 1 else b.shape[-2]
    if k_a != k_b:
        raise ValueError(f"products: a {a.shape} and b {b.shape} do not match as for a @ b")
    batch = () if a.ndim <= 2 and b.ndim <= 2 else np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m = () if a.ndim == 1 else (a.shape[-2],)
    n = () if b.ndim == 1 else (b.shape[-1],)
    return (*batch, *m, *n)


def _products(out, a, b, rows, conj_b, add, threads) -> None:
    """Into ``out`` (the shape of ``a @ b``): the product, added to it with ``add``.
    Operands that share memory with ``out`` are copied first, so that they are read as
    they were before any element of ``out`` changed. A real ``a`` times a conjugated
    complex ``b`` is ``a`` times a copy of ``b``'s conjugate (the compiled product
    conjugates ``b`` only for a complex ``a``)."""
    threads = _threads(threads)
    k = a.shape[-1]
    terms = None if rows is None else _rows(rows, k)
    if np.may_share_memory(out, a):
        a = a.copy()
    if np.may_share_memory(out, b):
        b = b.copy()
    if conj_b and a.dtype == _FLOAT and b.dtype == _COMPLEX:
        b, conj_b = np.conjugate(b), False
    # As (..., m, k) @ (..., k, n) into (..., m, n): views with an axis of one.
    if b.ndim == 1:
        b, out = b[:, None], out[..., None]
    if a.ndim == 1:
        a, out = a[None, :], out[..., None, :]
    batch = out.shape[:-2]
    if not batch:
        _matrix(out, a, b, terms, conj_b, add, threads)
    elif b.ndim == 2:  # a stack of matrices times one matrix: one product of all their rows
        rows_of = np.reshape(
            np.broadcast_to(a, (*batch, *a.shape[-2:])), (math.prod(batch) * a.shape[-2], k)
        )
        _into(_matrix, out, (-1, out.shape[-1]), rows_of, b, terms, conj_b, add, threads)
    else:
        a = np.broadcast_to(a, (*batch, *a.shape[-2:]))
        b = np.broadcast_to(b, (*batch, *b.shape[-2:]))
        for index in np.ndindex(*batch):
            _matrix(out[index], a[index], b[index], terms, conj_b, add, threads)


def _into(run, out, shape, a, b, terms, conj_b, add, threads) -> None:
    """``run`` on ``out`` seen as ``shape``: on a view if one exists, else on a copy that
    is written back."""
    try:
        view = np.reshape(out, shape, copy=False)
    except ValueError:
        view = None
    if view is not None:
        run(view, a, b, terms, conj_b, add, threads)
        return
    copy = np.reshape(out, shape) if add else np.empty(shape, dtype=out.dtype)
    run(copy, a, b, terms, conj_b, add, threads)
    out[...] = np.reshape(copy, out.shape)


def _rows(rows: ArrayLike, k: int) -> np.ndarray:
    """``rows`` as contiguous int64 indices into an axis of ``k``; a boolean mask of
    ``k`` values gives the indices where it is True."""
    rows = np.asarray(rows)
    if rows.dtype == np.bool_:
        if rows.shape != (k,):
            raise ValueError(f"products: rows as a boolean mask must have {k} values, one per term")
        return np.flatnonzero(rows)
    if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu"):
        raise TypeError("products: rows must be a 1-D array of indices or a boolean mask")
    return np.ascontiguousarray(rows, dtype=np.int64)


def _matrix(out, a, b, terms, conj_b, add, threads):
    """One product of matrices into the matrix ``out``, in whichever of two equal forms
    copies least: ``out = a @ b``, or ``out.T = b.T @ a.T``, whose sums are the same
    terms in the same order (``x * y`` and ``x + y`` are the same in IEEE arithmetic as
    ``y * x`` and ``y + x``; the sign and payload of a NaN result may differ). The
    compiled loop runs along the result's columns, which must be adjacent in memory, as
    must ``b``'s; an operand or result whose columns are not is copied (and the result
    written back). Of matrix times vector (one column), the compiled loop adds each
    row's sum by itself, and is slow where ``a``'s rows are not adjacent in memory;
    there the transposed form makes the vector's length the loop's."""
    m, n = out.shape
    k = a.shape[1]
    direct = _cost(b, out) + (m * k if n == 1 and m > 1 and not _adjacent(a) else 0)
    transposed = _cost(a.T, out.T) + (n * k if m == 1 and n > 1 and not _adjacent(b.T) else 0)
    if not conj_b and transposed < direct:
        a, b, out = b.T, a.T, out.T
    if not _adjacent(b):
        b = np.ascontiguousarray(b)
    if _adjacent(out) and _readable(out):
        _compiled().fixed_product(a, b, out, terms, conj_b, add, threads)
        return
    result = out.copy(order="C") if add else np.empty(out.shape, dtype=out.dtype)
    _compiled().fixed_product(a, b, result, terms, conj_b, add, threads)
    out[...] = result


def _readable(x: np.ndarray) -> bool:
    """Whether the compiled core can take ``x`` as it is: aligned, and its strides whole
    elements."""
    return x.flags.aligned and all(s % x.itemsize == 0 for s in x.strides)


def _adjacent(x: np.ndarray) -> bool:
    """Whether the columns of the matrix ``x`` are adjacent in memory."""
    return x.shape[1] <= 1 or x.shape[0] == 0 or x.strides[1] == x.itemsize


def _cost(b: np.ndarray, out: np.ndarray) -> int:
    """Values copied to run ``out = a @ b`` as it stands."""
    return (0 if _adjacent(b) else b.size) + (0 if _adjacent(out) and _readable(out) else out.size)
