// SPDX-License-Identifier: GPL-3.0-or-later
//
// Matrix products and dot products whose sums are added in an order fixed by their
// definition, so that their bits do not depend on the number of threads, the CPU, the
// compiler or the platform (acoustic_spaces/products.py calls them).
//
// fixed_product: out (m, n) = a (m, k) @ b (k, n). Element out[i, j] is
//
//     acc = a[i, k0] * b[k0, j];  acc = acc + a[i, k1] * b[k1, j];  ...
//
// over k0 < k1 < ... (0, 1, ..., k - 1, or the given rows in their order), +0.0 for no
// terms; with `add`, acc starts from out[i, j] and adds every term. Each term is rounded
// once as a product and then added: plain IEEE double operations, never fused (the
// project builds with -ffp-contract=off, and MSVC with /fp:precise does not contract).
// Complex numbers are pairs of doubles; their terms are
//
//     real x complex     (a * br, a * bi)
//     complex x real     (ar * b, ai * b)
//     complex x complex  (ar * br - ai * bi, ar * bi + ai * br)
//     with conj_b        (ar * br + ai * bi, ai * br - ar * bi)
//
// each product rounded, then the subtraction or addition, and the real and imaginary
// parts are summed separately, each in the order above. Not std::complex arithmetic,
// which checks for infinities and NaNs in a library routine. conj_b is for a complex a
// and b only (a real a times a conjugated b is not defined here; the binding refuses it).
//
// Every finite or infinite result, and whether a result is NaN, is fixed by these
// operations. The sign and payload bits of a NaN result are not: IEEE 754 leaves open
// which NaN operand an operation passes on, and the hardware passes on its first
// operand's, which a compiler may put either way round in a vector loop and in its
// scalar remainder, so they differ with the blocks, the threads and the compiler.
//
// The loops keep a block of out in cache, run over the terms four at a time with the
// four added one after the other into each element, and leave the columns innermost,
// where the compilers vectorise them element by element. A product with one column
// (matrix times vector) instead adds each row's sum by itself, four rows at a time.
// Threads split the output's columns (or its rows, when the columns are few), never
// the terms: every element gets its terms in the same order whatever the blocks and
// the threads.
//
// out must not share memory with a or b (the loops read a and b while they write out),
// and every operand must be aligned to a double.
//
// fixed_dot: x . y over n, in eight partial sums: sum L adds x[i] * y[i] for the i with
// i mod 8 == L in ascending order, starting from +0.0, and the result is
// ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)). Eight sums, so that any CPU can
// keep them in its vector registers; the order does not depend on their width.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "acoustic_spaces/ray_tracer.hpp"

namespace acoustic_spaces {

// A product's operands and result. Strides count elements (a complex number is one);
// b's and out's columns are adjacent (their column stride is one element).
struct ProductOperands {
    std::size_t m = 0, k = 0, n = 0;
    const double *a = nullptr;
    std::ptrdiff_t a_row = 0, a_col = 0;  // a[i, k] at a + (i * a_row + k * a_col) * width
    bool a_complex = false;
    const double *b = nullptr;
    std::ptrdiff_t b_row = 0;  // b[k, j] at b + (k * b_row + j) * width
    bool b_complex = false;
    double *out = nullptr;
    std::ptrdiff_t out_row = 0;  // out[i, j] at out + (i * out_row + j) * width
    // The terms: k = rows[0], rows[1], ... (terms of them), or k = 0, 1, ... for a null
    // rows (so an empty list of rows is a non-null pointer with terms 0).
    const std::int64_t *rows = nullptr;
    std::size_t terms = 0;
    bool conj_b = false;
    bool add = false;
};

namespace detail {

// The kinds of operands: a and b real or complex, b conjugated.
enum class ProductKind { real, real_complex, complex_real, complex, complex_conj };

inline ProductKind product_kind(const ProductOperands &p) {
    if (!p.a_complex) return p.b_complex ? ProductKind::real_complex : ProductKind::real;
    if (!p.b_complex) return ProductKind::complex_real;
    return p.conj_b ? ProductKind::complex_conj : ProductKind::complex;
}

// One term a[i, k] * b[k, j] of kind K into (re, im) (re only for real x real), as the
// definition at the top of this file writes it; a and b point at the two values.
template <ProductKind K>
inline void fixed_term(const double *a, const double *b, double &re, double &im) {
    if constexpr (K == ProductKind::real) {
        re = a[0] * b[0];
    } else if constexpr (K == ProductKind::real_complex) {
        re = a[0] * b[0];
        im = a[0] * b[1];
    } else if constexpr (K == ProductKind::complex_real) {
        re = a[0] * b[0];
        im = a[1] * b[0];
    } else if constexpr (K == ProductKind::complex) {
        re = a[0] * b[0] - a[1] * b[1];
        im = a[0] * b[1] + a[1] * b[0];
    } else {
        re = a[0] * b[0] + a[1] * b[1];
        im = a[1] * b[0] - a[0] * b[1];
    }
}

// Columns [0, len) of one output row: o[j] (from the start value) plus T terms in order,
// a's T values and b's T rows given. Start: Fresh, the first term; else o[j].
// real x real, and real x complex as pairs of doubles (len counts doubles).
template <int T, bool Fresh>
inline void fixed_real_run(double *__restrict o, const double *av,
                           const double *const *bk, std::size_t len) {
    const double a0 = av[0];
    const double *__restrict b0 = bk[0];
    if constexpr (T == 1) {
        for (std::size_t j = 0; j < len; ++j) {
            if constexpr (Fresh) {
                o[j] = a0 * b0[j];
            } else {
                o[j] = o[j] + a0 * b0[j];
            }
        }
    } else {
        const double a1 = av[1], a2 = av[2], a3 = av[3];
        const double *__restrict b1 = bk[1];
        const double *__restrict b2 = bk[2];
        const double *__restrict b3 = bk[3];
        for (std::size_t j = 0; j < len; ++j) {
            double s;
            if constexpr (Fresh) {
                s = a0 * b0[j];
            } else {
                s = o[j] + a0 * b0[j];
            }
            s = s + a1 * b1[j];
            s = s + a2 * b2[j];
            s = s + a3 * b3[j];
            o[j] = s;
        }
    }
}

// complex x real: a's values as (re, im) pairs, b real, o complex pairs; len counts
// complex elements.
template <int T, bool Fresh>
inline void fixed_complex_real_run(double *__restrict o, const double *av,
                                   const double *const *bk, std::size_t len) {
    for (std::size_t j = 0; j < len; ++j) {
        double re, im;
        {
            const double b = bk[0][j];
            if constexpr (Fresh) {
                re = av[0] * b;
                im = av[1] * b;
            } else {
                re = o[2 * j] + av[0] * b;
                im = o[2 * j + 1] + av[1] * b;
            }
        }
        for (int t = 1; t < T; ++t) {
            const double b = bk[t][j];
            re = re + av[2 * t] * b;
            im = im + av[2 * t + 1] * b;
        }
        o[2 * j] = re;
        o[2 * j + 1] = im;
    }
}

// complex x complex (conjugating b with Conj); len counts complex elements.
template <int T, bool Fresh, bool Conj>
inline void fixed_complex_run(double *__restrict o, const double *av,
                              const double *const *bk, std::size_t len) {
    constexpr ProductKind K = Conj ? ProductKind::complex_conj : ProductKind::complex;
    for (std::size_t j = 0; j < len; ++j) {
        double re, im;
        fixed_term<K>(av, bk[0] + 2 * j, re, im);
        if constexpr (!Fresh) {
            re = o[2 * j] + re;
            im = o[2 * j + 1] + im;
        }
        for (int t = 1; t < T; ++t) {
            double term_re, term_im;
            fixed_term<K>(av + 2 * t, bk[t] + 2 * j, term_re, term_im);
            re = re + term_re;
            im = im + term_im;
        }
        o[2 * j] = re;
        o[2 * j + 1] = im;
    }
}

// Terms [r, r + T) into rows [i0, i1) and columns [j0, j1) of out.
template <int T, bool Fresh>
inline void fixed_product_step(const ProductOperands &p, std::size_t r, std::size_t i0,
                               std::size_t i1, std::size_t j0, std::size_t j1) {
    const std::size_t aw = p.a_complex ? 2 : 1, bw = p.b_complex ? 2 : 1;
    const std::size_t ow = (p.a_complex || p.b_complex) ? 2 : 1;
    std::array<std::ptrdiff_t, 4> ks{};
    std::array<const double *, 4> bk{};
    for (int t = 0; t < T; ++t) {
        ks[t] = p.rows ? static_cast<std::ptrdiff_t>(p.rows[r + t])
                       : static_cast<std::ptrdiff_t>(r + t);
        bk[t] = p.b + (ks[t] * p.b_row + static_cast<std::ptrdiff_t>(j0)) *
                          static_cast<std::ptrdiff_t>(bw);
    }
    const std::size_t len = j1 - j0;
    for (std::size_t i = i0; i < i1; ++i) {
        double *o = p.out + (static_cast<std::ptrdiff_t>(i) * p.out_row +
                             static_cast<std::ptrdiff_t>(j0)) *
                                static_cast<std::ptrdiff_t>(ow);
        std::array<double, 8> av{};
        for (int t = 0; t < T; ++t) {
            const double *at = p.a + (static_cast<std::ptrdiff_t>(i) * p.a_row + ks[t] * p.a_col) *
                                         static_cast<std::ptrdiff_t>(aw);
            av[aw * static_cast<std::size_t>(t)] = at[0];
            if (aw == 2) av[2 * static_cast<std::size_t>(t) + 1] = at[1];
        }
        if (!p.a_complex && !p.b_complex) {
            fixed_real_run<T, Fresh>(o, av.data(), bk.data(), len);
        } else if (!p.a_complex) {  // real x complex: b and out as doubles
            fixed_real_run<T, Fresh>(o, av.data(), bk.data(), 2 * len);
        } else if (!p.b_complex) {
            fixed_complex_real_run<T, Fresh>(o, av.data(), bk.data(), len);
        } else if (p.conj_b) {
            fixed_complex_run<T, Fresh, true>(o, av.data(), bk.data(), len);
        } else {
            fixed_complex_run<T, Fresh, false>(o, av.data(), bk.data(), len);
        }
    }
}

// Rows [i0, i0 + R) of a product with one column (n == 1), each row's sum over the
// `terms` terms in order by itself, the R sums side by side.
template <ProductKind K, std::size_t R>
inline void fixed_column_rows(const ProductOperands &p, std::size_t i0, std::size_t terms) {
    constexpr bool cplx = K != ProductKind::real;
    constexpr std::ptrdiff_t aw = (K == ProductKind::real || K == ProductKind::real_complex) ? 1 : 2;
    constexpr std::ptrdiff_t bw = (K == ProductKind::real || K == ProductKind::complex_real) ? 1 : 2;
    constexpr std::ptrdiff_t ow = cplx ? 2 : 1;
    std::array<const double *, R> a{};
    std::array<double *, R> o{};
    std::array<double, R> re{}, im{};
    for (std::size_t q = 0; q < R; ++q) {
        const auto i = static_cast<std::ptrdiff_t>(i0 + q);
        a[q] = p.a + i * p.a_row * aw;
        o[q] = p.out + i * p.out_row * ow;
    }
    const auto key = [&](std::size_t r) {
        return p.rows ? static_cast<std::ptrdiff_t>(p.rows[r]) : static_cast<std::ptrdiff_t>(r);
    };
    std::size_t r = 0;
    if (p.add) {
        for (std::size_t q = 0; q < R; ++q) {
            re[q] = o[q][0];
            if constexpr (cplx) im[q] = o[q][1];
        }
    } else {
        const std::ptrdiff_t k = key(0);
        for (std::size_t q = 0; q < R; ++q) {
            fixed_term<K>(a[q] + k * p.a_col * aw, p.b + k * p.b_row * bw, re[q], im[q]);
        }
        r = 1;
    }
    for (; r < terms; ++r) {
        const std::ptrdiff_t k = key(r);
        const double *bk = p.b + k * p.b_row * bw;
        for (std::size_t q = 0; q < R; ++q) {
            double term_re = 0.0, term_im = 0.0;
            fixed_term<K>(a[q] + k * p.a_col * aw, bk, term_re, term_im);
            re[q] = re[q] + term_re;
            if constexpr (cplx) im[q] = im[q] + term_im;
        }
    }
    for (std::size_t q = 0; q < R; ++q) {
        o[q][0] = re[q];
        if constexpr (cplx) o[q][1] = im[q];
    }
}

// Rows [i0, i1) of a product with one column, four rows at a time.
template <ProductKind K>
inline void fixed_column_part(const ProductOperands &p, std::size_t i0, std::size_t i1,
                              std::size_t terms) {
    std::size_t i = i0;
    for (; i + 4 <= i1; i += 4) fixed_column_rows<K, 4>(p, i, terms);
    for (; i < i1; ++i) fixed_column_rows<K, 1>(p, i, terms);
}

// Rows [i0, i1) and columns [j0, j1) of the product, one block of out at a time.
inline void fixed_product_part(const ProductOperands &p, std::size_t i0, std::size_t i1,
                               std::size_t j0, std::size_t j1) {
    const std::size_t ow = (p.a_complex || p.b_complex) ? 2 : 1;
    const std::size_t terms = p.rows ? p.terms : p.k;
    if (p.n == 1 && terms > 0) {
        switch (product_kind(p)) {
            case ProductKind::real:
                return fixed_column_part<ProductKind::real>(p, i0, i1, terms);
            case ProductKind::real_complex:
                return fixed_column_part<ProductKind::real_complex>(p, i0, i1, terms);
            case ProductKind::complex_real:
                return fixed_column_part<ProductKind::complex_real>(p, i0, i1, terms);
            case ProductKind::complex:
                return fixed_column_part<ProductKind::complex>(p, i0, i1, terms);
            case ProductKind::complex_conj:
                return fixed_column_part<ProductKind::complex_conj>(p, i0, i1, terms);
        }
    }
    if (terms == 0) {
        if (!p.add) {
            for (std::size_t i = i0; i < i1; ++i) {
                double *o = p.out + static_cast<std::ptrdiff_t>(i) * p.out_row *
                                        static_cast<std::ptrdiff_t>(ow);
                std::fill(o + j0 * ow, o + j1 * ow, 0.0);
            }
        }
        return;
    }
    // Blocks of up to 8 rows by about 2048 doubles (16 KiB) of out.
    constexpr std::size_t kRows = 8;
    const std::size_t rows = std::min(kRows, i1 - i0);
    const std::size_t width = std::max<std::size_t>(32, 2048 / (rows * ow));
    for (std::size_t jb = j0; jb < j1; jb += width) {
        const std::size_t je = std::min(j1, jb + width);
        for (std::size_t ib = i0; ib < i1; ib += kRows) {
            const std::size_t ie = std::min(i1, ib + kRows);
            if (p.add) {
                fixed_product_step<1, false>(p, 0, ib, ie, jb, je);
            } else {
                fixed_product_step<1, true>(p, 0, ib, ie, jb, je);
            }
            std::size_t r = 1;
            for (; r + 4 <= terms; r += 4) fixed_product_step<4, false>(p, r, ib, ie, jb, je);
            for (; r < terms; ++r) fixed_product_step<1, false>(p, r, ib, ie, jb, je);
        }
    }
}

}  // namespace detail

// The product as defined at the top of this file, on `threads` threads (0: all cores),
// which split the output's columns, or its rows when there are fewer than 32 columns
// for each thread. Every element's bits are the same for any number of threads.
inline void fixed_product(const ProductOperands &p, std::size_t threads) {
    if (p.m == 0 || p.n == 0) return;
    constexpr std::size_t kColumns = 32;
    const std::size_t units = (p.n + kColumns - 1) / kColumns;
    // (worker_threads asks the system for the cores: not for one thread.)
    if (threads != 1) threads = std::max<std::size_t>(1, worker_threads(threads));
    if (threads == 1) {
        detail::fixed_product_part(p, 0, p.m, 0, p.n);
    } else if (units >= threads || p.m < 2) {
        parallel_for(units, threads, [&](std::size_t begin, std::size_t end, std::size_t) {
            detail::fixed_product_part(p, 0, p.m, begin * kColumns,
                                       std::min(p.n, end * kColumns));
        });
    } else {
        parallel_for(p.m, threads, [&](std::size_t begin, std::size_t end, std::size_t) {
            detail::fixed_product_part(p, begin, end, 0, p.n);
        });
    }
}

// x . y over n values, in the eight partial sums defined at the top of this file.
inline double fixed_dot(const double *x, const double *y, std::size_t n) {
    std::array<double, 8> s{};
    const std::size_t whole = n - n % 8;
    for (std::size_t i = 0; i < whole; i += 8) {
        for (std::size_t q = 0; q < 8; ++q) s[q] = s[q] + x[i + q] * y[i + q];
    }
    for (std::size_t i = whole; i < n; ++i) s[i - whole] = s[i - whole] + x[i] * y[i];
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

// x . x, y . y and x . y, each as fixed_dot, in three passes (measured three times
// faster than one pass with the 24 sums).
inline std::array<double, 3> fixed_dots(const double *x, const double *y, std::size_t n) {
    return {fixed_dot(x, x, n), fixed_dot(y, y, n), fixed_dot(x, y, n)};
}

}  // namespace acoustic_spaces
