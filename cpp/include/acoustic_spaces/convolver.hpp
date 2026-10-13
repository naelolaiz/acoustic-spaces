// SPDX-License-Identifier: GPL-3.0-or-later
//
// Partitioned convolution for real-time rendering, without added latency: output is
// ready as soon as the block it belongs to has come in. Nothing allocates and nothing
// blocks after construction, and no threads are involved.
//
// The response is cut into segments whose partitions grow along it (non-uniform
// partitioning, as in Gardner 1995 and Wefers 2015). The head is convolved by uniformly
// partitioned overlap-save with a frequency-domain delay line at the caller's block
// size. Each later segment uses blocks `growth` times larger than the one before and
// starts two of its own blocks into the response, so the result of a block is needed
// only one block after its input is complete: its work (input FFT, multiply-adds,
// inverse FFTs) is spread evenly over the caller's blocks in between. A long response
// then costs a few times what its head would cost alone, instead of growing with its
// length: a 3 s response at 48 kHz in blocks of 128 frames takes about a tenth of the
// time per block that uniform partitions take.
//
// A convolver holds a matrix of impulse responses, one per input and output: output o
// is the sum over inputs i of input i convolved with response (i, o).

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "acoustic_spaces/fft.hpp"

namespace acoustic_spaces {

template <typename T>
class PartitionedConvolver {
public:
    // `ir` is (inputs, outputs, taps), row-major. `block`, `growth` and `max_block` are
    // powers of two: partitions grow `growth` times per segment up to `max_block` frames
    // (larger FFTs would make single steps of work too long); growth 1 keeps every
    // partition at `block` (uniform partitioning).
    PartitionedConvolver(const T *ir, std::size_t inputs, std::size_t outputs,
                         std::size_t taps, std::size_t block, std::size_t growth = 8,
                         std::size_t max_block = 8192)
        : inputs_(inputs), outputs_(outputs), taps_(taps), block_(block) {
        check(inputs, outputs, taps, block, growth, max_block);
        // Segment 0 runs from 0 to 2 * block * growth; segment k > 0, with blocks of
        // block * growth^k, from two of its blocks to two of the next segment's. The
        // last one runs to the end.
        std::size_t size = block;
        std::size_t start = 0;
        while (start < taps) {
            const std::size_t next = size * growth;
            const bool last = growth == 1 || next > max_block || 2 * next >= taps;
            const std::size_t stop = last ? taps : 2 * next;
            segments_.emplace_back(ir, inputs, outputs, taps, start, stop, size, block);
            start = stop;
            size = next;
        }
    }

    std::size_t inputs() const { return inputs_; }
    std::size_t outputs() const { return outputs_; }
    std::size_t taps() const { return taps_; }
    std::size_t block() const { return block_; }
    std::size_t segments() const { return segments_.size(); }
    std::size_t segment_start(std::size_t k) const { return segments_.at(k).start; }
    std::size_t segment_block(std::size_t k) const { return segments_.at(k).size; }
    std::size_t segment_partitions(std::size_t k) const { return segments_.at(k).partitions; }

    // One block: `in` is (block, inputs) and `out` (block, outputs), both interleaved.
    void process(const T *in, T *out) {
        std::fill(out, out + block_ * outputs_, T(0));
        for (auto &segment : segments_) segment.process(in, out, block_);
    }

    // Forget all past input.
    void reset() {
        for (auto &segment : segments_) segment.reset();
    }

private:
    // Uniformly partitioned overlap-save over taps [start, stop) with blocks of `size`.
    // Segment 0 (start 0, size equal to the caller's block) works synchronously; later
    // ones (start = 2 * size) compute a block while the next one comes in.
    struct Segment {
        std::size_t start, size, partitions, bins, steps, units;
        std::size_t inputs, outputs;
        RealFft<T> fft;
        std::vector<T> filters_re, filters_im;  // (inputs, outputs, partitions, bins)
        std::vector<T> spectra_re, spectra_im;  // (inputs, partitions, bins): recent input
        std::vector<T> history;                 // (inputs, 2 * size): last two input blocks
        std::vector<T> sum_re, sum_im;
        std::vector<T> time;
        std::vector<T> results;  // (2, outputs, size): written while the other is read
        std::size_t head = 0;    // slot of the newest input spectrum
        std::size_t calls = 0;   // caller's blocks so far

        Segment(const T *ir, std::size_t n_inputs, std::size_t n_outputs, std::size_t taps,
                std::size_t first, std::size_t stop, std::size_t length, std::size_t block)
            : start(first),
              size(length),
              partitions((stop - first + length - 1) / length),
              bins(length + 1),
              steps(length / block),
              units(n_outputs * (n_inputs * partitions + 1)),
              inputs(n_inputs),
              outputs(n_outputs),
              fft(2 * length) {
            const std::size_t count = inputs * outputs * partitions * bins;
            filters_re.assign(count, T(0));
            filters_im.assign(count, T(0));
            std::vector<T> segment(2 * size);
            for (std::size_t i = 0; i < inputs; ++i) {
                for (std::size_t o = 0; o < outputs; ++o) {
                    const T *response = ir + (i * outputs + o) * taps;
                    for (std::size_t p = 0; p < partitions; ++p) {
                        std::fill(segment.begin(), segment.end(), T(0));
                        const std::size_t from = start + p * size;
                        const std::size_t n = std::min(size, stop - from);
                        std::copy(response + from, response + from + n, segment.begin());
                        const std::size_t at = filter(i, o, p);
                        fft.forward(segment.data(), filters_re.data() + at, filters_im.data() + at);
                    }
                }
            }
            spectra_re.assign(inputs * partitions * bins, T(0));
            spectra_im.assign(inputs * partitions * bins, T(0));
            history.assign(inputs * 2 * size, T(0));
            sum_re.assign(bins, T(0));
            sum_im.assign(bins, T(0));
            time.assign(2 * size, T(0));
            results.assign(2 * outputs * size, T(0));
        }

        void reset() {
            std::fill(spectra_re.begin(), spectra_re.end(), T(0));
            std::fill(spectra_im.begin(), spectra_im.end(), T(0));
            std::fill(history.begin(), history.end(), T(0));
            std::fill(results.begin(), results.end(), T(0));
            head = 0;
            calls = 0;
        }

        // Adds this segment's share of one caller block to `out`.
        void process(const T *in, T *out, std::size_t block) {
            const std::size_t step = calls % steps;
            const std::size_t cycle = calls / steps;
            if (start == 0) {
                take(in, 0, block);
                transform_input();
                for (std::size_t u = 0; u < units; ++u) run(u, 0);
                add(out, 0, 0, block);
            } else {
                // Input block `cycle - 1` is complete: transform it, then do this step's
                // share of its work. Its result is read during the next cycle.
                if (cycle >= 1) {
                    if (step == 0) transform_input();
                    const std::size_t begin = units * step / steps;
                    const std::size_t end = units * (step + 1) / steps;
                    for (std::size_t u = begin; u < end; ++u) run(u, (cycle - 1) % 2);
                }
                take(in, step * block, block);
                if (cycle >= 2) add(out, cycle % 2, step * block, block);
            }
            ++calls;
        }

        // Copies a caller block into the newest input block at `offset`, shifting the
        // history first when a new block begins.
        void take(const T *in, std::size_t offset, std::size_t block) {
            for (std::size_t i = 0; i < inputs; ++i) {
                T *h = history.data() + i * 2 * size;
                if (offset == 0) std::copy(h + size, h + 2 * size, h);
                for (std::size_t f = 0; f < block; ++f) h[size + offset + f] = in[f * inputs + i];
            }
        }

        void transform_input() {
            head = (head + 1) % partitions;
            for (std::size_t i = 0; i < inputs; ++i) {
                const std::size_t at = slot(i, head);
                fft.forward(history.data() + i * 2 * size, spectra_re.data() + at,
                            spectra_im.data() + at);
            }
        }

        // Unit u of a block's work: for each output, a multiply-add per input and
        // partition, then the inverse FFT into result buffer `which`.
        void run(std::size_t u, std::size_t which) {
            const std::size_t per_output = inputs * partitions + 1;
            const std::size_t o = u / per_output;
            const std::size_t r = u % per_output;
            if (r == 0) {
                std::fill(sum_re.begin(), sum_re.end(), T(0));
                std::fill(sum_im.begin(), sum_im.end(), T(0));
            }
            if (r < per_output - 1) {
                const std::size_t i = r / partitions;
                const std::size_t p = r % partitions;
                const std::size_t x = slot(i, (head + partitions - p) % partitions);
                const std::size_t h = filter(i, o, p);
                multiply_add(spectra_re.data() + x, spectra_im.data() + x,
                             filters_re.data() + h, filters_im.data() + h);
                return;
            }
            fft.inverse(sum_re.data(), sum_im.data(), time.data());
            std::copy(time.begin() + static_cast<std::ptrdiff_t>(size), time.end(),
                      results.begin() + static_cast<std::ptrdiff_t>((which * outputs + o) * size));
        }

        void add(T *out, std::size_t which, std::size_t offset, std::size_t block) const {
            for (std::size_t o = 0; o < outputs; ++o) {
                const T *r = results.data() + (which * outputs + o) * size + offset;
                for (std::size_t f = 0; f < block; ++f) out[f * outputs + o] += r[f];
            }
        }

        void multiply_add(const T *xr, const T *xi, const T *hr, const T *hi) {
            T *sr = sum_re.data();
            T *si = sum_im.data();
            for (std::size_t k = 0; k < bins; ++k) {
                sr[k] += xr[k] * hr[k] - xi[k] * hi[k];
                si[k] += xr[k] * hi[k] + xi[k] * hr[k];
            }
        }

        std::size_t filter(std::size_t i, std::size_t o, std::size_t p) const {
            return ((i * outputs + o) * partitions + p) * bins;
        }
        std::size_t slot(std::size_t i, std::size_t s) const {
            return (i * partitions + s) * bins;
        }
    };

    std::size_t inputs_, outputs_, taps_, block_;
    std::vector<Segment> segments_;

    static bool power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

    static void check(std::size_t inputs, std::size_t outputs, std::size_t taps,
                      std::size_t block, std::size_t growth, std::size_t max_block) {
        if (inputs == 0 || outputs == 0 || taps == 0) {
            throw std::invalid_argument("PartitionedConvolver: empty impulse response");
        }
        if (block < 2 || !power_of_two(block)) {
            throw std::invalid_argument("PartitionedConvolver: block must be a power of two");
        }
        if (!power_of_two(growth) || !power_of_two(max_block)) {
            throw std::invalid_argument(
                "PartitionedConvolver: growth and max_block must be powers of two");
        }
    }
};

}  // namespace acoustic_spaces
