// SPDX-License-Identifier: GPL-3.0-or-later
//
// Uniformly partitioned convolution (overlap-save with a frequency-domain delay line)
// for real-time rendering. Each block of `block` frames costs two FFTs per channel plus
// one complex multiply-add per partition of the impulse response, independent of how
// long the response is otherwise; output is ready as soon as the block it belongs to
// has come in (no latency beyond the block). Nothing allocates after construction.
//
// A convolver holds a matrix of impulse responses, one per input and output: output o
// is the sum over inputs i of input i convolved with response (i, o).

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "acoustic_spaces/fft.hpp"

namespace acoustic_spaces {

template <typename T>
class PartitionedConvolver {
public:
    using Complex = std::complex<T>;

    // `ir` is (inputs, outputs, taps), row-major; `block` is a power of two.
    PartitionedConvolver(const T *ir, std::size_t inputs, std::size_t outputs,
                         std::size_t taps, std::size_t block)
        : inputs_(inputs),
          outputs_(outputs),
          taps_(taps),
          block_(block),
          partitions_(block == 0 ? 0 : (taps + block - 1) / block),
          fft_(check(inputs, outputs, taps, block)),
          bins_(block + 1) {
        const std::size_t filters = inputs_ * outputs_ * partitions_ * bins_;
        filters_re_.assign(filters, T(0));
        filters_im_.assign(filters, T(0));
        spectrum_.assign(bins_, Complex(0, 0));
        std::vector<T> segment(2 * block_);
        for (std::size_t i = 0; i < inputs_; ++i) {
            for (std::size_t o = 0; o < outputs_; ++o) {
                const T *response = ir + (i * outputs_ + o) * taps_;
                for (std::size_t p = 0; p < partitions_; ++p) {
                    std::fill(segment.begin(), segment.end(), T(0));
                    const std::size_t start = p * block_;
                    const std::size_t count = std::min(block_, taps_ - start);
                    std::copy(response + start, response + start + count, segment.begin());
                    fft_.forward(segment.data(), spectrum_.data());
                    const std::size_t at = filter(i, o, p);
                    for (std::size_t k = 0; k < bins_; ++k) {
                        filters_re_[at + k] = spectrum_[k].real();
                        filters_im_[at + k] = spectrum_[k].imag();
                    }
                }
            }
        }
        spectra_re_.assign(inputs_ * partitions_ * bins_, T(0));
        spectra_im_.assign(inputs_ * partitions_ * bins_, T(0));
        history_.assign(inputs_ * 2 * block_, T(0));
        sum_re_.assign(bins_, T(0));
        sum_im_.assign(bins_, T(0));
        time_.assign(2 * block_, T(0));
    }

    std::size_t inputs() const { return inputs_; }
    std::size_t outputs() const { return outputs_; }
    std::size_t taps() const { return taps_; }
    std::size_t block() const { return block_; }
    std::size_t partitions() const { return partitions_; }

    // One block: `in` is (block, inputs) and `out` (block, outputs), both interleaved.
    void process(const T *in, T *out) {
        for (std::size_t i = 0; i < inputs_; ++i) {
            T *history = history_.data() + i * 2 * block_;
            std::copy(history + block_, history + 2 * block_, history);
            for (std::size_t f = 0; f < block_; ++f) history[block_ + f] = in[f * inputs_ + i];
            fft_.forward(history, spectrum_.data());
            const std::size_t at = spectrum(i, head_);
            for (std::size_t k = 0; k < bins_; ++k) {
                spectra_re_[at + k] = spectrum_[k].real();
                spectra_im_[at + k] = spectrum_[k].imag();
            }
        }
        for (std::size_t o = 0; o < outputs_; ++o) {
            std::fill(sum_re_.begin(), sum_re_.end(), T(0));
            std::fill(sum_im_.begin(), sum_im_.end(), T(0));
            for (std::size_t i = 0; i < inputs_; ++i) {
                for (std::size_t p = 0; p < partitions_; ++p) {
                    const std::size_t x = spectrum(i, (head_ + partitions_ - p) % partitions_);
                    const std::size_t h = filter(i, o, p);
                    multiply_add(spectra_re_.data() + x, spectra_im_.data() + x,
                                 filters_re_.data() + h, filters_im_.data() + h);
                }
            }
            for (std::size_t k = 0; k < bins_; ++k) spectrum_[k] = Complex(sum_re_[k], sum_im_[k]);
            fft_.inverse(spectrum_.data(), time_.data());
            for (std::size_t f = 0; f < block_; ++f) out[f * outputs_ + o] = time_[block_ + f];
        }
        head_ = (head_ + 1) % partitions_;
    }

    // Forget all past input.
    void reset() {
        std::fill(spectra_re_.begin(), spectra_re_.end(), T(0));
        std::fill(spectra_im_.begin(), spectra_im_.end(), T(0));
        std::fill(history_.begin(), history_.end(), T(0));
        head_ = 0;
    }

private:
    std::size_t inputs_, outputs_, taps_, block_, partitions_;
    RealFft<T> fft_;
    std::size_t bins_;
    // Spectra are kept as separate real and imaginary parts, so the multiply-add over
    // partitions, where nearly all the time goes, vectorizes.
    std::vector<T> filters_re_, filters_im_;  // (inputs, outputs, partitions, bins)
    std::vector<T> spectra_re_, spectra_im_;  // (inputs, partitions, bins): recent input
    std::vector<T> history_;                  // (inputs, 2 * block): last two input blocks
    std::vector<T> sum_re_, sum_im_;
    std::vector<Complex> spectrum_;
    std::vector<T> time_;
    std::size_t head_ = 0;  // slot of the newest input spectrum

    void multiply_add(const T *xr, const T *xi, const T *hr, const T *hi) {
        T *sr = sum_re_.data();
        T *si = sum_im_.data();
        for (std::size_t k = 0; k < bins_; ++k) {
            sr[k] += xr[k] * hr[k] - xi[k] * hi[k];
            si[k] += xr[k] * hi[k] + xi[k] * hr[k];
        }
    }

    static std::size_t check(std::size_t inputs, std::size_t outputs, std::size_t taps,
                             std::size_t block) {
        if (inputs == 0 || outputs == 0 || taps == 0) {
            throw std::invalid_argument("PartitionedConvolver: empty impulse response");
        }
        if (block < 2 || (block & (block - 1)) != 0) {
            throw std::invalid_argument("PartitionedConvolver: block must be a power of two");
        }
        return 2 * block;
    }

    std::size_t filter(std::size_t i, std::size_t o, std::size_t p) const {
        return ((i * outputs_ + o) * partitions_ + p) * bins_;
    }
    std::size_t spectrum(std::size_t i, std::size_t slot) const {
        return (i * partitions_ + slot) * bins_;
    }
};

}  // namespace acoustic_spaces
