// SPDX-License-Identifier: GPL-3.0-or-later
//
// Direct (time-domain) convolution. It is O(N*M) and only meant as the
// reference that the partitioned real-time convolver is tested against.

#pragma once

#include <cstddef>
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

}  // namespace acoustic_spaces
