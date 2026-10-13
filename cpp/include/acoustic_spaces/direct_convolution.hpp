// SPDX-License-Identifier: GPL-3.0-or-later
//
// Direct (time-domain) convolution. It is O(N*M): convolve_direct is the reference the
// partitioned real-time convolver is tested against, and scatter_kernels places short
// filters (HRIRs) at sparse impulses.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace acoustic_spaces {

// Full linear convolution: the result has signal.size() + ir.size() - 1 samples.
template <typename T>
std::vector<T> convolve_direct(const T *signal, std::size_t signal_size, const T *ir,
                               std::size_t ir_size) {
    if (signal_size == 0 || ir_size == 0) {
        throw std::invalid_argument("convolve_direct: inputs must not be empty");
    }
    std::vector<T> out(signal_size + ir_size - 1, T(0));
    for (std::size_t i = 0; i < signal_size; ++i) {
        const T x = signal[i];
        T *dst = out.data() + i;
        for (std::size_t j = 0; j < ir_size; ++j) {
            dst[j] += x * ir[j];
        }
    }
    return out;
}

// Short kernels added at given samples: for each i,
// out[c][positions[i] + t] += amplitudes[i] * kernels[index[i]][c][t].
// kernels is (count, channels, taps) and the result (channels, length); whatever falls
// past `length` is dropped.
inline std::vector<double> scatter_kernels(std::size_t length, const std::int64_t *positions,
                                           const std::int64_t *index, const double *amplitudes,
                                           std::size_t n, const double *kernels,
                                           std::size_t count, std::size_t channels,
                                           std::size_t taps) {
    std::vector<double> out(channels * length, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (positions[i] < 0 || index[i] < 0 || static_cast<std::size_t>(index[i]) >= count) {
            throw std::invalid_argument("scatter_kernels: position or kernel index out of range");
        }
        const auto start = static_cast<std::size_t>(positions[i]);
        if (start >= length) continue;
        const std::size_t span = std::min(taps, length - start);
        const double a = amplitudes[i];
        const double *kernel = kernels + static_cast<std::size_t>(index[i]) * channels * taps;
        for (std::size_t c = 0; c < channels; ++c) {
            double *dst = out.data() + c * length + start;
            const double *src = kernel + c * taps;
            for (std::size_t t = 0; t < span; ++t) dst[t] += a * src[t];
        }
    }
    return out;
}

}  // namespace acoustic_spaces
