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

#pragma once

#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "acoustic_spaces/ray_tracer.hpp"

namespace acoustic_spaces {

class RationalSteps {
   public:
    using Complex = std::complex<double>;

    // m and l: rows x 3 x points, the lower diagonal (its first entry unused), the
    // diagonal and the upper diagonal (its last entry unused); a and b: rows x terms.
    RationalSteps(std::size_t rows, std::size_t points, std::size_t terms, const Complex *m,
                  const Complex *l, const Complex *a, const Complex *b, std::size_t threads)
        : rows_(rows),
          points_(points),
          terms_(terms),
          m_(m, m + rows * 3 * points),
          l_(l, l + rows * 3 * points),
          a_(a, a + rows * terms),
          b_(b, b + rows * terms),
          inverse_(rows * terms * points) {
        if (points == 0) throw std::invalid_argument("RationalSteps: no heights");
        std::atomic<bool> singular{false};
        parallel_for(rows * terms, threads, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t k = begin; k < end; ++k) {
                if (!factor(k / terms_, k % terms_)) singular = true;
            }
        });
        if (singular) throw std::runtime_error("RationalSteps: a step's matrix is singular");
    }

    std::size_t rows() const { return rows_; }
    std::size_t points() const { return points_; }
    std::size_t terms() const { return terms_; }

    // One step on psi (rows x points, by row) in place.
    void apply(Complex *psi, std::size_t threads) const {
        parallel_for(rows_, threads, [&](std::size_t begin, std::size_t end, std::size_t) {
            std::vector<Complex> work(points_);
            for (std::size_t row = begin; row < end; ++row) {
                for (std::size_t term = 0; term < terms_; ++term) {
                    step(row, term, psi + row * points_, work);
                }
            }
        });
    }

   private:
    // Values this small (300 dB under the field's) are set to zero: the field's tails
    // would otherwise decay into subnormal numbers, which processors work through tens
    // of times slower.
    static constexpr double kTiny = 1e-150;
    static Complex flush(Complex x) {
        return std::abs(x.real()) < kTiny && std::abs(x.imag()) < kTiny ? Complex(0.0, 0.0) : x;
    }

    static Complex mul(Complex x, Complex y) {
        return {x.real() * y.real() - x.imag() * y.imag(),
                x.real() * y.imag() + x.imag() * y.real()};
    }

    // The diagonals of M + c L at height i: lower (i > 0), centre, upper (i < points - 1).
    Complex lower(const Complex *m, const Complex *l, Complex c, std::size_t i) const {
        return m[i] + mul(c, l[i]);
    }
    Complex centre(const Complex *m, const Complex *l, Complex c, std::size_t i) const {
        return m[points_ + i] + mul(c, l[points_ + i]);
    }
    Complex upper(const Complex *m, const Complex *l, Complex c, std::size_t i) const {
        return m[2 * points_ + i] + mul(c, l[2 * points_ + i]);
    }

    // Factors M + b L for one row and term: the inverses of the pivots.
    bool factor(std::size_t row, std::size_t term) {
        const Complex *m = &m_[row * 3 * points_];
        const Complex *l = &l_[row * 3 * points_];
        const Complex b = b_[row * terms_ + term];
        Complex *inverse = &inverse_[(row * terms_ + term) * points_];
        Complex above{0.0, 0.0};  // the upper diagonal over the previous pivot
        for (std::size_t i = 0; i < points_; ++i) {
            Complex pivot = centre(m, l, b, i);
            if (i > 0) pivot -= mul(lower(m, l, b, i), above);
            const double size = std::norm(pivot);
            if (!(size > 0.0) || !std::isfinite(size)) return false;
            inverse[i] = Complex(pivot.real() / size, -pivot.imag() / size);
            if (i + 1 < points_) above = mul(upper(m, l, b, i), inverse[i]);
        }
        return true;
    }

    void step(std::size_t row, std::size_t term, Complex *psi, std::vector<Complex> &y) const {
        const Complex *m = &m_[row * 3 * points_];
        const Complex *l = &l_[row * 3 * points_];
        const Complex a = a_[row * terms_ + term];
        const Complex b = b_[row * terms_ + term];
        const Complex *inverse = &inverse_[(row * terms_ + term) * points_];
        const std::size_t n = points_;
        // (M + a L) psi, eliminated forwards as it is formed.
        Complex previous{0.0, 0.0};
        for (std::size_t i = 0; i < n; ++i) {
            Complex d = mul(centre(m, l, a, i), psi[i]);
            if (i > 0) {
                d += mul(lower(m, l, a, i), psi[i - 1]);
                d -= mul(lower(m, l, b, i), previous);
            }
            if (i + 1 < n) d += mul(upper(m, l, a, i), psi[i + 1]);
            previous = flush(mul(d, inverse[i]));
            y[i] = previous;
        }
        // Back substitution.
        psi[n - 1] = y[n - 1];
        for (std::size_t i = n - 1; i-- > 0;) {
            psi[i] = flush(y[i] - mul(mul(upper(m, l, b, i), inverse[i]), psi[i + 1]));
        }
    }

    std::size_t rows_, points_, terms_;
    std::vector<Complex> m_, l_, a_, b_;
    std::vector<Complex> inverse_;
};

}  // namespace acoustic_spaces
