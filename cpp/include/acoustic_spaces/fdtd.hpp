// SPDX-License-Identifier: GPL-3.0-or-later
//
// Wave-based room acoustics: the standard rectilinear finite-difference time-domain
// scheme (the second-order leapfrog on a cubic grid, at the Courant limit 1/sqrt(3)),
// with frequency-independent locally reacting boundaries on the voxelised room.
//
// The grid's cells are air when they can be reached from the source without crossing
// a face of the model, so closed rooms, open models and thin screens all work. Between
// two cell centres, a crossed face is a boundary of that face's specific admittance
// beta (normal velocity = beta * p / (rho c)). The update of a cell with K neighbours
// it connects to (no face between them) and boundary faces of admittances beta_f is
// the finite-volume one:
//
//   p+ = ((2 - K L^2) p + L^2 sum(connected neighbours) - (1 - g) p-) / (1 + g),
//   g = L / 2 * sum(beta_f),  L = 1 / sqrt(3),
//
// (across a thin screen, the air on the other side is a neighbour it does not connect
// to)
//
// which is passive (stable) for beta >= 0 and is the plain scheme for K = 6, beta = 0.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define ACOUSTIC_SPACES_HAS_MXCSR 1
#endif

#include "acoustic_spaces/bvh.hpp"
#include "acoustic_spaces/ray_tracer.hpp"
#include "acoustic_spaces/vec3.hpp"

namespace acoustic_spaces {

// Threads meeting between time steps: each spins briefly, then yields.
class StepBarrier {
public:
    explicit StepBarrier(std::size_t count) : count_(count) {}

    void wait() {
        const std::size_t generation = generation_.load(std::memory_order_acquire);
        if (waiting_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            waiting_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        for (unsigned spin = 0; generation_.load(std::memory_order_acquire) == generation;
             ++spin) {
            if (spin > 2000) std::this_thread::yield();
        }
    }

private:
    const std::size_t count_;
    std::atomic<std::size_t> waiting_{0};
    std::atomic<std::size_t> generation_{0};
};

// Flushes denormal floats to zero while it lives (on x86, where they are slow): the
// edge of a spreading wave holds ever smaller values.
class FlushDenormals {
public:
    FlushDenormals() {
#ifdef ACOUSTIC_SPACES_HAS_MXCSR
        saved_ = _mm_getcsr();
        _mm_setcsr(saved_ | 0x8040u);  // flush to zero, denormals are zero
#endif
    }
    ~FlushDenormals() {
#ifdef ACOUSTIC_SPACES_HAS_MXCSR
        _mm_setcsr(saved_);
#endif
    }
    FlushDenormals(const FlushDenormals &) = delete;
    FlushDenormals &operator=(const FlushDenormals &) = delete;

#ifdef ACOUSTIC_SPACES_HAS_MXCSR
private:
    unsigned saved_ = 0;
#endif
};

class WaveSolver {
public:
    // The grid has nx x ny x nz cells of `cell` metres, cell 0 starting at `origin`, with
    // x fastest; the outermost layer is never air. `admittance` holds the specific
    // admittance of every face of `bvh`; the grid's edge, which only an open model's air
    // reaches, has `edge_admittance`. The source and every probe take the air cells
    // round them that they can see, weighted trilinearly.
    WaveSolver(const Bvh &bvh, const double *admittance, const Vec3 &origin, double cell,
               std::size_t nx, std::size_t ny, std::size_t nz, const Vec3 &source,
               const std::vector<Vec3> &probes, double edge_admittance, std::size_t threads)
        : origin_(origin), cell_(cell), nx_(nx), ny_(ny), nz_(nz), sy_(nx), sz_(nx * ny) {
        if (nx < 3 || ny < 3 || nz < 3) throw std::invalid_argument("the grid is too small");
        if (!(cell > 0.0)) throw std::invalid_argument("cell must be positive");
        threads_ = threads ? threads : std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t n = nx * ny * nz;

        // Faces crossed between neighbouring cell centres, per axis (-1: none).
        std::array<std::vector<std::int32_t>, 3> blocked;
        for (auto &b : blocked) b.assign(n, -1);
        parallel_for(n, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                const std::size_t x = i % nx, y = (i / nx) % ny, z = i / sz_;
                const std::size_t at[3] = {x, y, z}, size[3] = {nx, ny, nz};
                const Vec3 c = centre(i);
                for (int a = 0; a < 3; ++a) {
                    if (at[a] + 1 >= size[a]) continue;
                    const Vec3 dir{a == 0 ? 1.0 : 0.0, a == 1 ? 1.0 : 0.0, a == 2 ? 1.0 : 0.0};
                    const Hit hit = bvh.intersect(c, dir, cell, -1);
                    if (hit.found()) blocked[a][i] = static_cast<std::int32_t>(hit.face);
                }
            }
        });

        // Air: every cell reachable from the cells the source sees.
        type_.assign(n, kSolid);
        std::vector<std::size_t> queue;
        for (const auto &[i, w] : visible_cells(bvh, source, nullptr)) {
            (void)w;
            if (type_[i] == kSolid) {
                type_[i] = kInterior;
                queue.push_back(i);
            }
        }
        if (queue.empty()) {
            throw std::invalid_argument("the source sees no cell of the wave grid");
        }
        while (!queue.empty()) {
            const std::size_t i = queue.back();
            queue.pop_back();
            for (int a = 0; a < 3; ++a) {
                const std::size_t stride = a == 0 ? 1 : (a == 1 ? sy_ : sz_);
                for (int side = 0; side < 2; ++side) {
                    if (!open_link(blocked, i, a, side)) continue;
                    const std::size_t j = side ? i + stride : i - stride;
                    if (type_[j] == kSolid) {
                        type_[j] = kInterior;
                        queue.push_back(j);
                    }
                }
            }
        }

        // Boundary cells: fewer than six air neighbours, or a face between them.
        const double half_courant = 0.5 / std::sqrt(3.0);
        for (std::size_t i = 0; i < n; ++i) {
            if (type_[i] == kSolid) continue;
            ++air_;
            int k = 0;
            std::uint8_t links = 0;  // bit 2a + side: connected to that neighbour
            double beta = 0.0;
            for (int a = 0; a < 3; ++a) {
                const std::size_t stride = a == 0 ? 1 : (a == 1 ? sy_ : sz_);
                for (int side = 0; side < 2; ++side) {
                    const std::size_t j = side ? i + stride : i - stride;
                    if (edge(j)) {
                        beta += edge_admittance;
                        open_ = true;
                        continue;
                    }
                    std::int32_t face = side ? blocked[a][i] : blocked[a][j];
                    if (face >= 0 && !side) {
                        // blocked[a][j] is the face nearest j: of a wall thinner than a
                        // cell, the other side's. This cell meets the face nearest it.
                        const Vec3 back{a == 0 ? -1.0 : 0.0, a == 1 ? -1.0 : 0.0,
                                        a == 2 ? -1.0 : 0.0};
                        const Hit hit = bvh.intersect(centre(i), back, cell, -1);
                        if (hit.found()) face = static_cast<std::int32_t>(hit.face);
                    }
                    if (face >= 0) {
                        beta += admittance[face];
                    } else {
                        ++k;
                        links = static_cast<std::uint8_t>(links | (1u << (2 * a + side)));
                    }
                }
            }
            if (k < 6 || beta > 0.0) add_special(i, k, links, half_courant * beta);
        }
        boundary_ = special_.size();

        // The source's cells (made special, so one update adds the source).
        for (const auto &[i, w] : visible_cells(bvh, source, &type_)) {
            std::size_t k = special_index(i);
            if (k == npos) k = add_special(i, 6, kAllLinks, 0.0);
            source_[k] += static_cast<float>(w);
        }
        for (const Vec3 &p : probes) {
            auto cells = visible_cells(bvh, p, &type_);
            probe_cells_.push_back({});
            probe_weights_.push_back({});
            for (const auto &[i, w] : cells) {
                probe_cells_.back().push_back(i);
                probe_weights_.back().push_back(w);
            }
        }
        u0_.assign(n, 0.0f);
        u1_.assign(n, 0.0f);
        threads_ = std::max<std::size_t>(1, std::min(threads_, air_ / 20000 + 1));
    }

    std::size_t air_cells() const { return air_; }
    std::size_t boundary_cells() const { return boundary_; }
    bool open() const { return open_; }
    std::size_t probes() const { return probe_cells_.size(); }
    bool probe_found(std::size_t p) const { return !probe_cells_.at(p).empty(); }
    std::size_t threads() const { return threads_; }

    // Advances `steps` steps, adding signal[k] at the source on step k, and writes every
    // probe's pressure before each step to out (probes x steps).
    void run(const float *signal, std::size_t steps, double *out) {
        const std::size_t t_count = threads_;
        const std::size_t n = u0_.size(), specials = special_.size();
        StepBarrier barrier(t_count);
        float *u0 = u0_.data(), *u1 = u1_.data();
        auto work = [&](std::size_t t) {
            FlushDenormals flush;
            const std::size_t ib = n * t / t_count, ie = n * (t + 1) / t_count;
            const std::size_t sb = specials * t / t_count, se = specials * (t + 1) / t_count;
            const std::uint8_t *type = type_.data();
            const std::int64_t *index = special_.data();
            const float *ca = coef_a_.data(), *cb = coef_b_.data(), *ci = coef_inv_.data();
            const std::uint8_t *links = links_.data();
            const float *src = source_.data();
            const std::size_t sy = sy_, sz = sz_;
            // L^2 = 1 / 3, rounded down: rounded up, six of it make more than 2 and the
            // highest mode of a very large grid could grow.
            const float third = std::nextafter(1.0f / 3.0f, 0.0f);
            float *a = u0, *b = u1;  // a: the previous step, then the next; b: the current
            for (std::size_t step = 0; step < steps; ++step) {
                if (t == 0) record(b, step, steps, out);
                for (std::size_t i = std::max<std::size_t>(ib, sz); i < std::min(ie, n - sz);
                     ++i) {
                    if (type[i] != kInterior) continue;
                    a[i] = third * (b[i - 1] + b[i + 1] + b[i - sy] + b[i + sy] + b[i - sz] +
                                    b[i + sz]) -
                           a[i];
                }
                const float s = signal[step];
                for (std::size_t k = sb; k < se; ++k) {
                    const std::size_t i = static_cast<std::size_t>(index[k]);
                    const unsigned m = links[k];
                    const float sum = ((m & 1u) ? b[i - 1] : 0.0f) + ((m & 2u) ? b[i + 1] : 0.0f) +
                                      ((m & 4u) ? b[i - sy] : 0.0f) +
                                      ((m & 8u) ? b[i + sy] : 0.0f) +
                                      ((m & 16u) ? b[i - sz] : 0.0f) +
                                      ((m & 32u) ? b[i + sz] : 0.0f);
                    a[i] = (ca[k] * b[i] + third * sum - cb[k] * a[i] + src[k] * s) * ci[k];
                }
                barrier.wait();
                std::swap(a, b);
            }
        };
        {
            std::vector<std::thread> pool;
            for (std::size_t t = 1; t < t_count; ++t) pool.emplace_back(work, t);
            work(0);
            for (auto &thread : pool) thread.join();
        }
        if (steps % 2 == 1) std::swap(u0_, u1_);
    }

    void reset() {
        std::fill(u0_.begin(), u0_.end(), 0.0f);
        std::fill(u1_.begin(), u1_.end(), 0.0f);
    }

private:
    static constexpr std::uint8_t kSolid = 0, kInterior = 1, kSpecial = 2;
    static constexpr std::uint8_t kAllLinks = 63;
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    Vec3 centre(std::size_t i) const {
        const std::size_t x = i % nx_, y = (i / nx_) % ny_, z = i / sz_;
        return {origin_.x + (static_cast<double>(x) + 0.5) * cell_,
                origin_.y + (static_cast<double>(y) + 0.5) * cell_,
                origin_.z + (static_cast<double>(z) + 0.5) * cell_};
    }

    bool edge(std::size_t i) const {
        const std::size_t x = i % nx_, y = (i / nx_) % ny_, z = i / sz_;
        return x == 0 || y == 0 || z == 0 || x + 1 == nx_ || y + 1 == ny_ || z + 1 == nz_;
    }

    // Whether cell i connects to its neighbour along axis a (side 1: +, 0: -) through
    // no face; the edge layer never connects.
    bool open_link(const std::array<std::vector<std::int32_t>, 3> &blocked, std::size_t i,
                   int a, int side) const {
        const std::size_t stride = a == 0 ? 1 : (a == 1 ? sy_ : sz_);
        const std::size_t j = side ? i + stride : i - stride;
        if (edge(j)) return false;
        return (side ? blocked[a][i] : blocked[a][j]) < 0;
    }

    // The cells round point p (up to eight) whose centres p sees, with trilinear weights
    // that sum to one; only air cells when `air` is given.
    std::vector<std::pair<std::size_t, double>> visible_cells(
        const Bvh &bvh, const Vec3 &p, const std::vector<std::uint8_t> *air) const {
        std::vector<std::pair<std::size_t, double>> out;
        const double q[3] = {(p.x - origin_.x) / cell_ - 0.5, (p.y - origin_.y) / cell_ - 0.5,
                             (p.z - origin_.z) / cell_ - 0.5};
        const double size[3] = {static_cast<double>(nx_), static_cast<double>(ny_),
                                static_cast<double>(nz_)};
        long long base[3];
        double frac[3];
        for (int a = 0; a < 3; ++a) {
            if (!(q[a] >= 1.0 && q[a] <= size[a] - 3.0)) return out;  // outside the grid
            base[a] = static_cast<long long>(std::floor(q[a]));
            frac[a] = q[a] - static_cast<double>(base[a]);
        }
        double total = 0.0;
        for (int corner = 0; corner < 8; ++corner) {
            double w = 1.0;
            std::size_t i = 0, stride = 1;
            for (int a = 0; a < 3; ++a) {
                const int up = (corner >> a) & 1;
                w *= up ? frac[a] : 1.0 - frac[a];
                i += static_cast<std::size_t>(base[a] + up) * stride;
                stride *= a == 0 ? nx_ : (a == 1 ? ny_ : 1);
            }
            if (w <= 0.0) continue;
            if (air != nullptr && (*air)[i] == kSolid) continue;
            const Vec3 to = centre(i) - p;
            const double distance = norm(to);
            if (distance > 0.0 && bvh.intersect(p, to / distance, distance, -1).found()) continue;
            out.emplace_back(i, w);
            total += w;
        }
        if (total < 1e-9) {
            out.clear();
            return out;
        }
        for (auto &cell_weight : out) cell_weight.second /= total;
        return out;
    }

    std::size_t add_special(std::size_t i, int k, std::uint8_t links, double g) {
        type_[i] = kSpecial;
        special_.push_back(static_cast<std::int64_t>(i));
        links_.push_back(links);
        coef_a_.push_back(static_cast<float>(2.0 - k / 3.0));
        coef_b_.push_back(static_cast<float>(1.0 - g));
        coef_inv_.push_back(static_cast<float>(1.0 / (1.0 + g)));
        source_.push_back(0.0f);
        return special_.size() - 1;
    }

    std::size_t special_index(std::size_t i) const {
        if (type_[i] != kSpecial) return npos;
        const auto it = std::find(special_.begin(), special_.end(), static_cast<std::int64_t>(i));
        return it == special_.end() ? npos : static_cast<std::size_t>(it - special_.begin());
    }

    void record(const float *field, std::size_t step, std::size_t steps, double *out) const {
        for (std::size_t p = 0; p < probe_cells_.size(); ++p) {
            double value = 0.0;
            for (std::size_t k = 0; k < probe_cells_[p].size(); ++k) {
                value += probe_weights_[p][k] * field[probe_cells_[p][k]];
            }
            out[p * steps + step] = value;
        }
    }

    Vec3 origin_;
    double cell_;
    std::size_t nx_, ny_, nz_, sy_, sz_;
    std::size_t threads_ = 1;
    std::size_t air_ = 0, boundary_ = 0;
    bool open_ = false;
    std::vector<std::uint8_t> type_;
    std::vector<std::int64_t> special_;
    std::vector<std::uint8_t> links_;
    std::vector<float> coef_a_, coef_b_, coef_inv_, source_;
    std::vector<std::vector<std::size_t>> probe_cells_;
    std::vector<std::vector<double>> probe_weights_;
    std::vector<float> u0_, u1_;
};

}  // namespace acoustic_spaces
