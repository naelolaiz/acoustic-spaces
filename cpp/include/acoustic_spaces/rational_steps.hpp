// SPDX-License-Identifier: GPL-3.0-or-later
//
// The range steps of the wide-angle parabolic equation (acoustic_spaces/engine/wide_angle.py):
// for each row (one frequency),
//
//     psi <- (M + b_P L)^-1 (M + a_P L) ... (M + b_1 L)^-1 (M + a_1 L) psi,
//
// with M and L tridiagonal over the heights (the finite differences' weights and the
// operator) and a_j, b_j the factors of the step's rational approximation. Each M + b_j L
// is factored once, by Gaussian elimination without pivoting (the Thomas algorithm), and
// every step reuses the factors. The complex products are written out, as the compilers'
// own check every one for infinities and NaNs.
//
// The rows are marched in batches of up to four, height by height together: each row's
// sums and products are the ones it would take alone, in the same order, so a row's field
// does not depend on the rows it shares a batch with. With SSE2 (any x86-64) two rows
// share a vector register, element by element. A batch's diagonals lie by height, the
// rows side by side, and the field is eliminated and substituted back in its own place.
// Of diagonals of the equation's form (see compact_form) the steps keep one value where
// several are the same.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define ACOUSTIC_SPACES_HAS_SSE2 1
#endif

#include "acoustic_spaces/ray_tracer.hpp"

namespace acoustic_spaces {

class RationalSteps {
   public:
    using Complex = std::complex<double>;

    // m and l: rows x 3 x points, the lower diagonal (its first entry unused), the
    // diagonal and the upper diagonal (its last entry unused); a and b: rows x terms.
    RationalSteps(std::size_t rows, std::size_t points, std::size_t terms, const Complex *m,
                  const Complex *l, const Complex *a, const Complex *b, std::size_t threads)
        : rows_(rows), points_(points), terms_(terms) {
        if (points == 0) throw std::invalid_argument("RationalSteps: no heights");
        compact_ = compact_form(m, l);
        // Batches of 4, 2 and 1 rows, no wider than leaves every thread one.
        std::size_t widest = 4;
        while (widest > 1 && rows / widest < worker_threads(threads)) widest /= 2;
        for (std::size_t row = 0; row < rows;) {
            std::size_t width = widest;
            while (width > rows - row) width /= 2;
            batches_.push_back(Batch{row, width});
            row += width;
        }
        diagonals_.resize(rows * points * record());
        sides_.resize(compact_ ? rows * points * 2 : 0);
        ends_.resize(compact_ ? rows * kEnds * 2 : 0);
        factors_.resize(rows * terms * 4);
        inverse_.resize(rows * terms * points * 2);
        std::atomic<bool> singular{false};
        parallel_for(batches_.size(), threads,
                     [&](std::size_t begin, std::size_t end, std::size_t) {
                         for (std::size_t k = begin; k < end; ++k) {
                             const Batch &batch = batches_[k];
                             lay(batch, m, l, a, b);
                             bool factored;
                             if (batch.width == 4) {
                                 factored = factor<Rows<Pair, 2>>(batch);
                             } else if (batch.width == 2) {
                                 factored = factor<Rows<Pair, 1>>(batch);
                             } else {
                                 factored = factor<Rows<One, 1>>(batch);
                             }
                             if (!factored) singular = true;
                         }
                     });
        if (singular) throw std::runtime_error("RationalSteps: a step's matrix is singular");
    }

    std::size_t rows() const { return rows_; }
    std::size_t points() const { return points_; }
    std::size_t terms() const { return terms_; }

    // One step on psi (rows x points, by row) in place.
    void apply(Complex *psi, std::size_t threads) const {
        parallel_for(batches_.size(), threads,
                     [&](std::size_t begin, std::size_t end, std::size_t) {
                         // A batch's field by height: its rows' real parts, then their
                         // imaginary parts.
                         std::vector<double> field(points_ * 2 * kWidest);
                         for (std::size_t k = begin; k < end; ++k) {
                             const Batch &batch = batches_[k];
                             if (batch.width == 4) {
                                 march<Rows<Pair, 2>>(batch, psi, field.data());
                             } else if (batch.width == 2) {
                                 march<Rows<Pair, 1>>(batch, psi, field.data());
                             } else {
                                 march<Rows<One, 1>>(batch, psi, field.data());
                             }
                         }
                     });
    }

   private:
    // Values this small (300 dB under the field's) are set to zero: the field's tails
    // would otherwise decay into subnormal numbers, which processors work through tens
    // of times slower.
    static constexpr double kTiny = 1e-150;
    static constexpr std::size_t kWidest = 4;  // rows of the widest batch
    // What a step reads next is asked for this many bytes ahead of where it reads: at
    // the highest frequencies a batch's diagonals, pivots and field are larger than a
    // core's own caches, and the processor does not by itself fetch ahead across the
    // pages of several arrays at once.
    static constexpr std::size_t kAheadBytes = 4096;

    static void fetch(const double *at) {
#if defined(ACOUSTIC_SPACES_HAS_SSE2)
        _mm_prefetch(reinterpret_cast<const char *>(at), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(at);
#else
        (void)at;
#endif
    }
    // Asks for ``doubles`` values from ``at`` on.
    static void fetch(const double *at, std::size_t doubles) {
        for (std::size_t k = 0; k < doubles; k += 8) fetch(at + k);
    }

    // One value of one row.
    struct One {
        static constexpr std::size_t kRows = 1;
        double v;
        static One load(const double *p) { return {*p}; }
        void store(double *p) const { *p = v; }
        friend One operator+(One x, One y) { return {x.v + y.v}; }
        friend One operator-(One x, One y) { return {x.v - y.v}; }
        friend One operator*(One x, One y) { return {x.v * y.v}; }
        friend One operator/(One x, One y) { return {x.v / y.v}; }
        One operator-() const { return {-v}; }
        // Whether the value is above zero and finite.
        bool regular() const { return v > 0.0 && v <= std::numeric_limits<double>::max(); }
        // Zero for both parts of a complex value whose parts are both tiny.
        static void flush(One &re, One &im) {
            if (std::abs(re.v) < kTiny && std::abs(im.v) < kTiny) re.v = im.v = 0.0;
        }
    };

    // One value of each of two rows, worked on element by element: in a vector register
    // with SSE2, else as two numbers.
    struct Pair {
        static constexpr std::size_t kRows = 2;
#ifdef ACOUSTIC_SPACES_HAS_SSE2
        __m128d v;
        static Pair load(const double *p) { return {_mm_loadu_pd(p)}; }
        void store(double *p) const { _mm_storeu_pd(p, v); }
        friend Pair operator+(Pair x, Pair y) { return {_mm_add_pd(x.v, y.v)}; }
        friend Pair operator-(Pair x, Pair y) { return {_mm_sub_pd(x.v, y.v)}; }
        friend Pair operator*(Pair x, Pair y) { return {_mm_mul_pd(x.v, y.v)}; }
        friend Pair operator/(Pair x, Pair y) { return {_mm_div_pd(x.v, y.v)}; }
        Pair operator-() const { return {_mm_xor_pd(_mm_set1_pd(-0.0), v)}; }
        // Whether both rows' values are above zero and finite.
        bool regular() const {
            const __m128d largest = _mm_set1_pd(std::numeric_limits<double>::max());
            return _mm_movemask_pd(_mm_and_pd(_mm_cmpgt_pd(v, _mm_setzero_pd()),
                                              _mm_cmple_pd(v, largest))) == 3;
        }
        static void flush(Pair &re, Pair &im) {
            const __m128d sign = _mm_set1_pd(-0.0), tiny = _mm_set1_pd(kTiny);
            const __m128d both = _mm_and_pd(_mm_cmplt_pd(_mm_andnot_pd(sign, re.v), tiny),
                                            _mm_cmplt_pd(_mm_andnot_pd(sign, im.v), tiny));
            re.v = _mm_andnot_pd(both, re.v);
            im.v = _mm_andnot_pd(both, im.v);
        }
#else
        double v[2];
        static Pair load(const double *p) { return {{p[0], p[1]}}; }
        void store(double *p) const {
            p[0] = v[0];
            p[1] = v[1];
        }
        friend Pair operator+(Pair x, Pair y) { return {{x.v[0] + y.v[0], x.v[1] + y.v[1]}}; }
        friend Pair operator-(Pair x, Pair y) { return {{x.v[0] - y.v[0], x.v[1] - y.v[1]}}; }
        friend Pair operator*(Pair x, Pair y) { return {{x.v[0] * y.v[0], x.v[1] * y.v[1]}}; }
        friend Pair operator/(Pair x, Pair y) { return {{x.v[0] / y.v[0], x.v[1] / y.v[1]}}; }
        Pair operator-() const { return {{-v[0], -v[1]}}; }
        bool regular() const { return One{v[0]}.regular() && One{v[1]}.regular(); }
        static void flush(Pair &re, Pair &im) {
            for (std::size_t k = 0; k < 2; ++k) {
                if (std::abs(re.v[k]) < kTiny && std::abs(im.v[k]) < kTiny) re.v[k] = im.v[k] = 0.0;
            }
        }
#endif
    };

    // One complex value of each row of a batch of count times V's rows: the real parts
    // and the imaginary parts, the rows in the batch's order.
    template <typename V, std::size_t count>
    struct Rows {
        static constexpr std::size_t kRows = count * V::kRows;
        V re[count], im[count];

        // From the rows' real parts and then their imaginary parts at ``p``.
        static Rows load(const double *p) {
            Rows out;
            for (std::size_t k = 0; k < count; ++k) {
                out.re[k] = V::load(p + k * V::kRows);
                out.im[k] = V::load(p + kRows + k * V::kRows);
            }
            return out;
        }
        void store(double *p) const {
            for (std::size_t k = 0; k < count; ++k) {
                re[k].store(p + k * V::kRows);
                im[k].store(p + kRows + k * V::kRows);
            }
        }
        friend Rows operator+(const Rows &x, const Rows &y) {
            Rows out;
            for (std::size_t k = 0; k < count; ++k) {
                out.re[k] = x.re[k] + y.re[k];
                out.im[k] = x.im[k] + y.im[k];
            }
            return out;
        }
        friend Rows operator-(const Rows &x, const Rows &y) {
            Rows out;
            for (std::size_t k = 0; k < count; ++k) {
                out.re[k] = x.re[k] - y.re[k];
                out.im[k] = x.im[k] - y.im[k];
            }
            return out;
        }
        // Each row's product, written out.
        friend Rows operator*(const Rows &x, const Rows &y) {
            Rows out;
            for (std::size_t k = 0; k < count; ++k) {
                out.re[k] = x.re[k] * y.re[k] - x.im[k] * y.im[k];
                out.im[k] = x.re[k] * y.im[k] + x.im[k] * y.re[k];
            }
            return out;
        }
        Rows flushed() const {
            Rows out = *this;
            for (std::size_t k = 0; k < count; ++k) V::flush(out.re[k], out.im[k]);
            return out;
        }
        // Each row's reciprocal into ``out``: the conjugate over the squared size. False,
        // with ``out`` left incomplete, if a row's squared size is zero or not finite.
        bool inverted(Rows &out) const {
            for (std::size_t k = 0; k < count; ++k) {
                const V size = re[k] * re[k] + im[k] * im[k];
                if (!size.regular()) return false;
                out.re[k] = re[k] / size;
                out.im[k] = -im[k] / size;
            }
            return true;
        }
    };

    // Rows ``first`` to ``first + width`` are marched together.
    struct Batch {
        std::size_t first, width;
    };

    // Whether the rows' diagonals have the equation's form: each diagonal of M has one
    // value at all its heights but the lowest (the finite differences' weights, the
    // ground's row aside), and L's upper diagonal at a height above the lowest is its
    // lower diagonal two heights on (both come from the height between). The steps then
    // keep M's values once, L's diagonal, and for its other two one side diagonal: at
    // height j the lower diagonal's value of height j + 1, which above the lowest heights
    // is the upper diagonal's of height j - 1.
    bool compact_form(const Complex *m, const Complex *l) const {
        if (points_ < 3) return false;
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t d = 0; d < 3; ++d) {
                const Complex *diagonal = m + (row * 3 + d) * points_;
                // The upper diagonal's last entry is not used.
                const std::size_t end = d == 2 ? points_ - 1 : points_;
                for (std::size_t i = 2; i < end; ++i) {
                    if (std::memcmp(diagonal + i, diagonal + 1, sizeof(Complex)) != 0) return false;
                }
            }
            const Complex *lower = l + row * 3 * points_, *upper = lower + 2 * points_;
            if (std::memcmp(upper + 1, lower + 3, (points_ - 3) * sizeof(Complex)) != 0) {
                return false;
            }
        }
        return true;
    }

    // The doubles a row takes at each height in ``diagonals_``: M's three diagonals and
    // L's three, or L's diagonal alone with ``compact_``, each a real and an imaginary
    // part.
    std::size_t record() const { return compact_ ? 2 : 12; }

    // With ``compact_``, a row has 6 more values: M's diagonal and upper diagonal at the
    // lowest height, its lower diagonal, diagonal and upper diagonal above it, and L's
    // upper diagonal at the lowest height.
    static constexpr std::size_t kEnds = 6;

    // Lays a batch's diagonals and factors out as its steps read them: by height (by
    // term), each value's real parts of the rows and then its imaginary parts.
    void lay(const Batch &batch, const Complex *m, const Complex *l, const Complex *a,
             const Complex *b) {
        const std::size_t width = batch.width, per = record();
        double *diagonals = &diagonals_[batch.first * points_ * per];
        // data(), not [], since factors_ is empty when there are no terms.
        double *factors = factors_.data() + batch.first * terms_ * 4;
        for (std::size_t w = 0; w < width; ++w) {
            const std::size_t row = batch.first + w;
            const Complex *of_m = m + row * 3 * points_, *of_l = l + row * 3 * points_;
            const auto put = [&](double *slots, std::size_t slot, Complex value) {
                slots[(2 * slot) * width + w] = value.real();
                slots[(2 * slot + 1) * width + w] = value.imag();
            };
            if (compact_) {
                double *sides = &sides_[batch.first * points_ * 2];
                for (std::size_t i = 0; i < points_; ++i) {
                    put(diagonals + i * per * width, 0, of_l[points_ + i]);
                    // The top height has no lower diagonal over it.
                    put(sides + i * 2 * width, 0,
                        i + 1 < points_ ? of_l[i + 1] : of_l[2 * points_ + i - 1]);
                }
                double *ends = &ends_[batch.first * kEnds * 2];
                put(ends, 0, of_m[points_]);
                put(ends, 1, of_m[2 * points_]);
                for (std::size_t d = 0; d < 3; ++d) put(ends, 2 + d, of_m[d * points_ + 1]);
                put(ends, 5, of_l[2 * points_]);
            } else {
                for (std::size_t i = 0; i < points_; ++i) {
                    for (std::size_t d = 0; d < 3; ++d) {
                        put(diagonals + i * per * width, d, of_m[d * points_ + i]);
                        put(diagonals + i * per * width, 3 + d, of_l[d * points_ + i]);
                    }
                }
            }
            for (std::size_t term = 0; term < terms_; ++term) {
                put(factors + term * 4 * width, 0, a[row * terms_ + term]);
                put(factors + term * 4 * width, 1, b[row * terms_ + term]);
            }
        }
    }

    // Where a batch's inverse pivots of one term lie: by height, the rows' real parts
    // and then their imaginary parts.
    std::size_t pivots(const Batch &batch, std::size_t term) const {
        return (batch.first * terms_ + term * batch.width) * points_ * 2;
    }

    // The diagonals of M + c L of a batch's rows, R one value of each row, from M's and
    // L's by height.
    template <typename R>
    struct Diagonals {
        static constexpr std::size_t kPer = 12 * R::kRows;               // doubles at each height
        static constexpr std::size_t kAhead = kAheadBytes / (8 * kPer);  // heights
        const double *at;

        Diagonals(const RationalSteps &steps, const Batch &batch)
            : at(&steps.diagonals_[batch.first * steps.points_ * 12]) {}
        // Diagonal ``d`` (0 lower, 1 centre, 2 upper) at height i.
        R over(std::size_t i, std::size_t d, const R &c) const {
            const double *here = at + i * kPer;
            return R::load(here + 2 * d * R::kRows) + c * R::load(here + 2 * (3 + d) * R::kRows);
        }
        // Diagonal ``d`` (1 or 2) at height 0.
        R first(std::size_t d, const R &c) const { return over(0, d, c); }
        // Asks for what a step up from height i will read, of n heights, and a step
        // down.
        void ahead(std::size_t i, std::size_t n) const {
            fetch(at + std::min(i + kAhead, n - 1) * kPer, kPer);
        }
        void behind(std::size_t i) const { fetch(at + (i > kAhead ? i - kAhead : 0) * kPer, kPer); }
    };

    // The same with ``compact_``: from L's diagonal and side diagonal by height and the
    // kept values.
    template <typename R>
    struct CompactDiagonals {
        static constexpr std::size_t kPer = 2 * R::kRows;
        static constexpr std::size_t kAhead = kAheadBytes / (8 * kPer);
        const double *centre, *side;
        R lowest[2];     // M's diagonal and upper diagonal at height 0
        R above[3];      // its three diagonals at the other heights
        R lowest_upper;  // L's upper diagonal at height 0

        CompactDiagonals(const RationalSteps &steps, const Batch &batch)
            : centre(&steps.diagonals_[batch.first * steps.points_ * 2]),
              side(&steps.sides_[batch.first * steps.points_ * 2]) {
            const double *ends = &steps.ends_[batch.first * kEnds * 2];
            for (std::size_t k = 0; k < 2; ++k) lowest[k] = R::load(ends + 2 * k * R::kRows);
            for (std::size_t d = 0; d < 3; ++d) above[d] = R::load(ends + 2 * (2 + d) * R::kRows);
            lowest_upper = R::load(ends + 2 * 5 * R::kRows);
        }
        // Diagonal ``d`` at a height i above 0: L's lower diagonal is the side diagonal
        // of the height below, its upper diagonal that of the height above.
        R over(std::size_t i, std::size_t d, const R &c) const {
            return above[d] + c * R::load(d == 1 ? centre + i * kPer : side + (i + d - 1) * kPer);
        }
        R first(std::size_t d, const R &c) const {
            return lowest[d - 1] + c * (d == 1 ? R::load(centre) : lowest_upper);
        }
        void ahead(std::size_t i, std::size_t n) const {
            fetch(centre + std::min(i + kAhead, n - 1) * kPer, kPer);
            fetch(side + std::min(i + kAhead, n - 1) * kPer, kPer);
        }
        void behind(std::size_t i) const {
            fetch(side + (i > kAhead ? i - kAhead : 0) * kPer, kPer);
        }
    };

    // Factors M + b L for the rows of a batch (R one value of each of them) and every
    // term's b: the inverses of the pivots. False if a matrix is singular.
    template <typename R>
    bool factor(const Batch &batch) {
        for (std::size_t term = 0; term < terms_; ++term) {
            const double *factors = &factors_[(batch.first * terms_ + term * R::kRows) * 4];
            const R b = R::load(factors + 2 * R::kRows);
            double *inverse = &inverse_[pivots(batch, term)];
            bool regular;
            if (compact_) {
                regular = invert_pivots(CompactDiagonals<R>(*this, batch), b, inverse);
            } else {
                regular = invert_pivots(Diagonals<R>(*this, batch), b, inverse);
            }
            if (!regular) return false;
        }
        return true;
    }

    // The inverses of the pivots of M + b L, from the diagonals D of a batch's rows R.
    // False if a row's matrix is singular.
    template <template <typename> class D, typename R>
    bool invert_pivots(const D<R> &diagonals, const R &b, double *inverse) const {
        const std::size_t slot = 2 * R::kRows;
        R inverted{};
        if (!diagonals.first(1, b).inverted(inverted)) return false;
        inverted.store(inverse);
        for (std::size_t i = 1; i < points_; ++i) {
            // The upper diagonal over the pivot, both of the height below.
            const R upper = i > 1 ? diagonals.over(i - 1, 2, b) : diagonals.first(2, b);
            const R above = upper * inverted;
            const R pivot = diagonals.over(i, 1, b) - diagonals.over(i, 0, b) * above;
            if (!pivot.inverted(inverted)) return false;
            inverted.store(inverse + i * slot);
        }
        return true;
    }

    // One step on the rows of a batch, R one value of each of them.
    template <typename R>
    void march(const Batch &batch, Complex *psi, double *field) const {
        const std::size_t width = R::kRows, n = points_;
        for (std::size_t w = 0; w < width; ++w) {
            const Complex *row = psi + (batch.first + w) * n;
            for (std::size_t i = 0; i < n; ++i) {
                field[(2 * i) * width + w] = row[i].real();
                field[(2 * i + 1) * width + w] = row[i].imag();
            }
        }
        if (compact_) {
            steps(CompactDiagonals<R>(*this, batch), batch, field);
        } else {
            steps(Diagonals<R>(*this, batch), batch, field);
        }
        for (std::size_t w = 0; w < width; ++w) {
            Complex *row = psi + (batch.first + w) * n;
            for (std::size_t i = 0; i < n; ++i) {
                row[i] = Complex(field[(2 * i) * width + w], field[(2 * i + 1) * width + w]);
            }
        }
    }

    // Every term's step on a batch's field, D the batch's diagonals of R.
    template <template <typename> class D, typename R>
    void steps(const D<R> &diagonals, const Batch &batch, double *field) const {
        for (std::size_t term = 0; term < terms_; ++term) {
            const double *factors = &factors_[(batch.first * terms_ + term * R::kRows) * 4];
            step(diagonals, R::load(factors), R::load(factors + 2 * R::kRows),
                 &inverse_[pivots(batch, term)], field);
        }
    }

    // One term's step on a batch's field (by height, the rows' real parts and then
    // their imaginary parts), with the term's factors a and b and inverse pivots.
    template <template <typename> class D, typename R>
    void step(const D<R> &diagonals, const R &a, const R &b, const double *inverse,
              double *field) const {
        const std::size_t n = points_, slot = 2 * R::kRows;
        // How many heights ahead the pivots and the field are asked for.
        const std::size_t ahead = kAheadBytes / (8 * slot);
        // (M + a L) psi, eliminated forwards as it is formed, in psi's place: ``here``
        // and ``below`` keep psi at the height being formed and at the one under it.
        R here = R::load(field), below = here, previous = here;
        {
            R d = diagonals.first(1, a) * here;
            if (n > 1) d = d + diagonals.first(2, a) * R::load(field + slot);
            previous = (d * R::load(inverse)).flushed();
            previous.store(field);
        }
        for (std::size_t i = 1; i + 1 < n; ++i) {
            diagonals.ahead(i, n);
            fetch(inverse + std::min(i + ahead, n - 1) * slot, slot);
            fetch(field + std::min(i + ahead, n - 1) * slot, slot);
            below = here;
            here = R::load(field + i * slot);
            R d = diagonals.over(i, 1, a) * here;
            d = d + diagonals.over(i, 0, a) * below;
            d = d - diagonals.over(i, 0, b) * previous;
            d = d + diagonals.over(i, 2, a) * R::load(field + (i + 1) * slot);
            previous = (d * R::load(inverse + i * slot)).flushed();
            previous.store(field + i * slot);
        }
        if (n > 1) {
            const std::size_t i = n - 1;
            below = here;
            here = R::load(field + i * slot);
            R d = diagonals.over(i, 1, a) * here;
            d = d + diagonals.over(i, 0, a) * below;
            d = d - diagonals.over(i, 0, b) * previous;
            previous = (d * R::load(inverse + i * slot)).flushed();
            previous.store(field + i * slot);
        }
        // Back substitution.
        for (std::size_t i = n - 1; i-- > 1;) {
            const std::size_t behind = i > ahead ? i - ahead : 0;
            diagonals.behind(i);
            fetch(inverse + behind * slot, slot);
            fetch(field + behind * slot, slot);
            const R over = diagonals.over(i, 2, b) * R::load(inverse + i * slot);
            previous = (R::load(field + i * slot) - over * previous).flushed();
            previous.store(field + i * slot);
        }
        if (n > 1) {
            const R over = diagonals.first(2, b) * R::load(inverse);
            previous = (R::load(field) - over * previous).flushed();
            previous.store(field);
        }
    }

    std::size_t rows_, points_, terms_;
    bool compact_ = false;
    std::vector<Batch> batches_;
    // By batch (rows side by side): the diagonals by height (with ``compact_`` L's
    // diagonal, its side diagonal in ``sides_`` and the kept values in ``ends_``), the
    // factors a and b by term, and the inverse pivots by term and height.
    std::vector<double> diagonals_, sides_, ends_, factors_, inverse_;
};

}  // namespace acoustic_spaces
