// SPDX-License-Identifier: GPL-3.0-or-later
//
// Real FFT of a power-of-two length for real-time use: every table and buffer is made in
// the constructor, so transforms never allocate. A real transform of length n runs as a
// complex radix-2 transform of length n / 2 plus a pass that separates even and odd
// samples (the usual "two for one" trick). Spectra come as interleaved std::complex or
// as separate real and imaginary arrays; the work arrays are kept split the same way,
// with each stage's twiddles in a row, so the butterflies vectorize.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
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
        // Stage with butterflies of length L (2, 4, ..., n / 2) uses e^(-2 pi i j / L),
        // j < L / 2, stored from index L / 2 - 1.
        twiddle_re_.resize(half_);
        twiddle_im_.resize(half_);
        for (std::size_t len = 2; len <= half_; len <<= 1) {
            for (std::size_t j = 0; j < len / 2; ++j) {
                const double a = -2.0 * pi * static_cast<double>(j) / static_cast<double>(len);
                twiddle_re_[len / 2 - 1 + j] = static_cast<T>(std::cos(a));
                twiddle_im_[len / 2 - 1 + j] = static_cast<T>(std::sin(a));
            }
        }
        split_re_.resize(half_ + 1);
        split_im_.resize(half_ + 1);
        for (std::size_t k = 0; k <= half_; ++k) {
            const double a = -2.0 * pi * static_cast<double>(k) / static_cast<double>(n_);
            split_re_[k] = static_cast<T>(std::cos(a));
            split_im_[k] = static_cast<T>(std::sin(a));
        }
        reversed_.resize(half_);
        std::size_t bits = 0;
        while ((std::size_t{1} << bits) < half_) ++bits;
        for (std::size_t i = 0; i < half_; ++i) {
            std::size_t r = 0;
            for (std::size_t b = 0; b < bits; ++b) r |= ((i >> b) & 1U) << (bits - 1 - b);
            reversed_[i] = r;
        }
        re_.resize(half_);
        im_.resize(half_);
    }

    std::size_t size() const { return n_; }
    std::size_t bins() const { return half_ + 1; }

    // n real samples to n / 2 + 1 bins, unscaled.
    void forward(const T *in, T *out_re, T *out_im) {
        for (std::size_t m = 0; m < half_; ++m) {
            re_[reversed_[m]] = in[2 * m];
            im_[reversed_[m]] = in[2 * m + 1];
        }
        transform(false);
        const T half = static_cast<T>(0.5);
        for (std::size_t k = 0; k <= half_; ++k) {
            const std::size_t a = k == half_ ? 0 : k;
            const std::size_t c = k == 0 ? 0 : half_ - k;
            const T even_re = (re_[a] + re_[c]) * half;
            const T even_im = (im_[a] - im_[c]) * half;
            const T odd_re = (im_[a] + im_[c]) * half;
            const T odd_im = (re_[c] - re_[a]) * half;
            out_re[k] = even_re + split_re_[k] * odd_re - split_im_[k] * odd_im;
            out_im[k] = even_im + split_re_[k] * odd_im + split_im_[k] * odd_re;
        }
    }

    // n / 2 + 1 bins to n real samples, scaled by 1 / n (the inverse of forward).
    void inverse(const T *in_re, const T *in_im, T *out) {
        const T half = static_cast<T>(0.5);
        for (std::size_t k = 0; k < half_; ++k) {
            const std::size_t c = half_ - k;
            const T even_re = (in_re[k] + in_re[c]) * half;
            const T even_im = (in_im[k] - in_im[c]) * half;
            const T diff_re = (in_re[k] - in_re[c]) * half;
            const T diff_im = (in_im[k] + in_im[c]) * half;
            // odd = diff * conj(split); z = even + i * odd
            const T odd_re = diff_re * split_re_[k] + diff_im * split_im_[k];
            const T odd_im = diff_im * split_re_[k] - diff_re * split_im_[k];
            re_[reversed_[k]] = even_re - odd_im;
            im_[reversed_[k]] = even_im + odd_re;
        }
        transform(true);
        const T scale = static_cast<T>(1) / static_cast<T>(half_);
        for (std::size_t m = 0; m < half_; ++m) {
            out[2 * m] = re_[m] * scale;
            out[2 * m + 1] = im_[m] * scale;
        }
    }

    // The same with interleaved complex bins.
    void forward(const T *in, Complex *out) {
        scratch_.resize(2 * bins());
        forward(in, scratch_.data(), scratch_.data() + bins());
        for (std::size_t k = 0; k < bins(); ++k) out[k] = Complex(scratch_[k], scratch_[bins() + k]);
    }

    void inverse(const Complex *in, T *out) {
        scratch_.resize(2 * bins());
        for (std::size_t k = 0; k < bins(); ++k) {
            scratch_[k] = in[k].real();
            scratch_[bins() + k] = in[k].imag();
        }
        inverse(scratch_.data(), scratch_.data() + bins(), out);
    }

private:
    std::size_t n_, half_;
    std::vector<T> twiddle_re_, twiddle_im_;  // per stage, see the constructor
    std::vector<T> split_re_, split_im_;      // e^(-2 pi i k / n)
    std::vector<std::size_t> reversed_;
    std::vector<T> re_, im_;  // the complex work array, split
    std::vector<T> scratch_;  // only for the interleaved interface

    // In-place complex FFT of (re_, im_) from bit-reversed to natural order;
    // `backward` conjugates the twiddles.
    void transform(bool backward) {
        T *re = re_.data();
        T *im = im_.data();
        const T sign = backward ? T(-1) : T(1);
        for (std::size_t len = 2; len <= half_; len <<= 1) {
            const std::size_t h = len / 2;
            const T *wr = twiddle_re_.data() + h - 1;
            const T *wi = twiddle_im_.data() + h - 1;
            for (std::size_t start = 0; start < half_; start += len) {
                T *ar = re + start;
                T *ai = im + start;
                T *br = ar + h;
                T *bi = ai + h;
                for (std::size_t j = 0; j < h; ++j) {
                    const T w_im = sign * wi[j];
                    const T tr = wr[j] * br[j] - w_im * bi[j];
                    const T ti = wr[j] * bi[j] + w_im * br[j];
                    br[j] = ar[j] - tr;
                    bi[j] = ai[j] - ti;
                    ar[j] += tr;
                    ai[j] += ti;
                }
            }
        }
    }
};

}  // namespace acoustic_spaces
