// SPDX-License-Identifier: GPL-3.0-or-later
//
// Real FFT of a power-of-two length for real-time use: every table and buffer is made in
// the constructor, so transforms never allocate. A real transform of length n runs as a
// complex radix-2 transform of length n / 2 plus a pass that separates even and odd
// samples (the usual "two for one" trick).

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace acoustic_spaces {

template <typename T>
class RealFft {
public:
    using Complex = std::complex<T>;

    explicit RealFft(std::size_t n) : n_(n), half_(n / 2) {
        if (n < 4 || (n & (n - 1)) != 0) {
            throw std::invalid_argument("RealFft: length must be a power of two, at least 4");
        }
        const double pi = std::acos(-1.0);
        twiddle_.resize(half_ / 2);
        for (std::size_t k = 0; k < twiddle_.size(); ++k) {
            const double a = -2.0 * pi * static_cast<double>(k) / static_cast<double>(half_);
            twiddle_[k] = Complex(static_cast<T>(std::cos(a)), static_cast<T>(std::sin(a)));
        }
        split_.resize(half_ + 1);
        for (std::size_t k = 0; k <= half_; ++k) {
            const double a = -2.0 * pi * static_cast<double>(k) / static_cast<double>(n_);
            split_[k] = Complex(static_cast<T>(std::cos(a)), static_cast<T>(std::sin(a)));
        }
        reversed_.resize(half_);
        std::size_t bits = 0;
        while ((std::size_t{1} << bits) < half_) ++bits;
        for (std::size_t i = 0; i < half_; ++i) {
            std::size_t r = 0;
            for (std::size_t b = 0; b < bits; ++b) r |= ((i >> b) & 1U) << (bits - 1 - b);
            reversed_[i] = r;
        }
        work_.resize(half_);
    }

    std::size_t size() const { return n_; }
    std::size_t bins() const { return half_ + 1; }

    // n real samples to n / 2 + 1 bins, unscaled.
    void forward(const T *in, Complex *out) {
        for (std::size_t m = 0; m < half_; ++m) work_[m] = Complex(in[2 * m], in[2 * m + 1]);
        transform(false);
        const T half = static_cast<T>(0.5);
        for (std::size_t k = 0; k <= half_; ++k) {
            const Complex a = work_[k % half_];
            const Complex b = std::conj(work_[(half_ - k) % half_]);
            const Complex even = (a + b) * half;
            const Complex odd = Complex((a.imag() - b.imag()) * half, (b.real() - a.real()) * half);
            out[k] = even + multiply(split_[k], odd);
        }
    }

    // n / 2 + 1 bins to n real samples, scaled by 1 / n (the inverse of forward).
    void inverse(const Complex *in, T *out) {
        const T half = static_cast<T>(0.5);
        for (std::size_t k = 0; k < half_; ++k) {
            const Complex a = in[k];
            const Complex b = std::conj(in[half_ - k]);
            const Complex even = (a + b) * half;
            const Complex odd = multiply((a - b) * half, std::conj(split_[k]));
            work_[k] = even + Complex(-odd.imag(), odd.real());
        }
        transform(true);
        const T scale = static_cast<T>(1) / static_cast<T>(half_);
        for (std::size_t m = 0; m < half_; ++m) {
            out[2 * m] = work_[m].real() * scale;
            out[2 * m + 1] = work_[m].imag() * scale;
        }
    }

private:
    std::size_t n_, half_;
    std::vector<Complex> twiddle_;  // e^(-2 pi i k / (n / 2))
    std::vector<Complex> split_;    // e^(-2 pi i k / n)
    std::vector<std::size_t> reversed_;
    std::vector<Complex> work_;

    // Plain complex product: operator* also handles infinities and NaNs (C99 Annex G),
    // which costs a branch per product and keeps loops from vectorizing.
    static Complex multiply(const Complex &a, const Complex &b) {
        return Complex(a.real() * b.real() - a.imag() * b.imag(),
                       a.real() * b.imag() + a.imag() * b.real());
    }

    // In-place complex FFT of work_ (length n / 2); `backward` conjugates the twiddles.
    void transform(bool backward) {
        for (std::size_t i = 0; i < half_; ++i) {
            if (i < reversed_[i]) std::swap(work_[i], work_[reversed_[i]]);
        }
        for (std::size_t len = 2; len <= half_; len <<= 1) {
            const std::size_t step = half_ / len;
            for (std::size_t start = 0; start < half_; start += len) {
                for (std::size_t j = 0; j < len / 2; ++j) {
                    const Complex w = backward ? std::conj(twiddle_[j * step]) : twiddle_[j * step];
                    const Complex u = work_[start + j];
                    const Complex v = multiply(work_[start + j + len / 2], w);
                    work_[start + j] = u + v;
                    work_[start + j + len / 2] = u - v;
                }
            }
        }
    }
};

}  // namespace acoustic_spaces
