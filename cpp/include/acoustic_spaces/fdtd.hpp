// SPDX-License-Identifier: GPL-3.0-or-later
//
// Wave-based room acoustics: the standard rectilinear finite-difference time-domain
// scheme (the second-order leapfrog on a cubic grid, at the Courant limit 1/sqrt(3)),
// with frequency-dependent locally reacting boundaries on the voxelised room and, round
// an open model, a perfectly matched layer.
//
// The grid's cells are air when they can be reached from the source without crossing
// a face of the model, so closed rooms, open models and thin screens all work. Between
// two cell centres, a crossed face is a boundary of that face's material, whose specific
// admittance (normal velocity times rho c over pressure) is a constant and parallel
// mass-resistance-stiffness branches,
//
//   Y(s) = beta + sum_j 1 / (M_j s + R_j + S_j / s),
//
// passive for beta, M, R, S >= 0. A branch's normalised velocity w (rho c times the
// velocity) follows M w'' + R w' + S w = p', centred in time:
//
//   m (w+ - 2 w + w-) + r (w+ - w-) / 2 + q (w+ + w-) / 2 = (p+ - p-) / 2,
//
// with m = M / T, r = R and q = S T for the time step T. A cell with n neighbours it
// connects to (no face between them) and boundary faces f is updated finite-volume
// style, L = 1 / sqrt(3):
//
//   p+ - 2 p + p- = L^2 (sum(connected neighbours) - n p)
//                   - L / 2 sum_f (beta_f (p+ - p-) + sum_j (w_j+ - w_j-)),
//
// which, solved together with the branches, is
//
//   p+ = ((2 - n L^2) p + L^2 sum(connected neighbours) - (1 - g) p- - L / 2 sum c_j)
//        / (1 + g),
//   g = L / 2 sum_f (beta_f + sum_j 1 / (2 a_j)),  a_j = m_j + r_j / 2 + q_j / 2,
//   c_j = (2 m_j w_j - (2 m_j + q_j) w_j-) / a_j,  then w_j+ = w_j- + (p+ - p-) / (2 a_j) + c_j
//
// (across a thin screen, the air on the other side is a neighbour it does not connect
// to). Its energy only falls, and it is the plain scheme for n = 6 and no faces.
//
// The matched layer: the `layer` cells inside the grid's edge, where an open model's air
// reaches them, are updated in the pressure-velocity form the leapfrog is equivalent to,
// with the pressure split by axis and each part damped along its axis (Berenger's split
// field), the damping growing as the square of the depth into the layer. Each layer cell
// keeps its own copy of the velocity on its six faces; both cells of a face update it
// alike. At the layer's inner side the damping is zero, where the two forms agree, and
// behind it the grid's edge holds no pressure.

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
    // x fastest; the outermost layer is never air. Face f of `bvh` is of material
    // face_material[f], of `materials`: beta[material], and the branches whose
    // branch_material is that material, branches holding m, r and q of each (see above).
    // The grid's edge, which only an open model's air reaches, has `edge_admittance`, or
    // with `layer` > 0 a matched layer of that many cells inside it, damped up to
    // `layer_damping` (s T / 2, s the damping rate) at the edge. The source and every
    // probe take the air cells round them that they can see, weighted trilinearly.
    WaveSolver(const Bvh &bvh, const std::int64_t *face_material, const double *beta,
               std::size_t materials, const double *branches, const std::int64_t *branch_material,
               std::size_t branch_count, const Vec3 &origin, double cell, std::size_t nx,
               std::size_t ny, std::size_t nz, const Vec3 &source,
               const std::vector<Vec3> &probes, double edge_admittance, std::size_t layer,
               double layer_damping, std::size_t threads)
        : origin_(origin), cell_(cell), nx_(nx), ny_(ny), nz_(nz), sy_(nx), sz_(nx * ny),
          layer_(layer) {
        if (nx < 3 || ny < 3 || nz < 3) throw std::invalid_argument("the grid is too small");
        if (!(cell > 0.0)) throw std::invalid_argument("cell must be positive");
        if (2 * layer + 3 > std::min({nx, ny, nz})) {
            throw std::invalid_argument("the matched layer leaves no grid inside it");
        }
        if (!(layer_damping >= 0.0)) throw std::invalid_argument("negative layer damping");
        for (std::size_t f = 0; f < bvh.face_count(); ++f) {
            if (face_material[f] < 0 || static_cast<std::size_t>(face_material[f]) >= materials) {
                throw std::invalid_argument("a face has no material");
            }
        }
        // Each material's branches, as a_j and the coefficients of c_j.
        std::vector<std::vector<std::int32_t>> material_branches(materials);
        for (std::size_t m = 0; m < materials; ++m) {
            if (!(beta[m] >= 0.0)) throw std::invalid_argument("negative admittance");
        }
        for (std::size_t j = 0; j < branch_count; ++j) {
            const double m = branches[3 * j], r = branches[3 * j + 1], q = branches[3 * j + 2];
            const double a = m + 0.5 * r + 0.5 * q;
            if (!(m >= 0.0 && r >= 0.0 && q >= 0.0 && a > 0.0)) {
                throw std::invalid_argument("a branch needs mass, resistance and stiffness >= 0");
            }
            if (branch_material[j] < 0 || static_cast<std::size_t>(branch_material[j]) >= materials) {
                throw std::invalid_argument("a branch has no material");
            }
            material_branches[static_cast<std::size_t>(branch_material[j])].push_back(
                static_cast<std::int32_t>(j));
            branch_now_.push_back(static_cast<float>(2.0 * m / a));
            branch_before_.push_back(static_cast<float>((2.0 * m + q) / a));
            branch_gain_.push_back(static_cast<float>(0.5 / a));
            branch_a_.push_back(a);
        }
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

        // The matched layer's air, which must meet no face.
        for (std::size_t i = 0; i < n && layer_ > 0; ++i) {
            if (type_[i] == kSolid || depth(i) > layer_) continue;
            for (int a = 0; a < 3; ++a) {
                const std::size_t stride = a == 0 ? 1 : (a == 1 ? sy_ : sz_);
                for (int side = 0; side < 2; ++side) {
                    const std::size_t j = side ? i + stride : i - stride;
                    if (!edge(j) && !open_link(blocked, i, a, side)) {
                        throw std::invalid_argument("the model reaches into the matched layer");
                    }
                }
            }
            type_[i] = kLayer;
            layer_cells_.push_back(static_cast<std::int64_t>(i));
            ++air_;
            open_ = true;
        }
        if (!layer_cells_.empty()) {
            layer_damping_ = layer_damping;
            for (int a = 0; a < 3; ++a) layer_coefficients(a);
            layer_state_.assign(9 * layer_cells_.size(), 0.0f);
        }

        // Boundary cells: fewer than six air neighbours, or a face between them.
        const double half_courant = 0.5 / std::sqrt(3.0);
        state_start_.push_back(0);
        for (std::size_t i = 0; i < n; ++i) {
            if (type_[i] != kInterior) continue;
            ++air_;
            int k = 0;
            std::uint8_t links = 0;  // bit 2a + side: connected to that neighbour
            double beta_sum = 0.0;
            std::array<std::int32_t, 6> crossed{};  // materials of the faces crossed,
            std::array<int, 6> count{};             // and how many faces of each
            std::size_t kinds = 0;
            for (int a = 0; a < 3; ++a) {
                const std::size_t stride = a == 0 ? 1 : (a == 1 ? sy_ : sz_);
                for (int side = 0; side < 2; ++side) {
                    const std::size_t j = side ? i + stride : i - stride;
                    if (edge(j)) {
                        beta_sum += edge_admittance;
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
                        const auto m = static_cast<std::int32_t>(face_material[face]);
                        beta_sum += beta[m];
                        std::size_t kind = 0;
                        while (kind < kinds && crossed[kind] != m) ++kind;
                        if (kind == kinds) crossed[kinds++] = m;
                        ++count[kind];
                    } else {
                        ++k;
                        links = static_cast<std::uint8_t>(links | (1u << (2 * a + side)));
                    }
                }
            }
            // One set of branch velocities per material the cell meets, standing for
            // each of its faces of that material.
            double g = beta_sum;
            for (std::size_t kind = 0; kind < kinds; ++kind) {
                const auto &own = material_branches[static_cast<std::size_t>(crossed[kind])];
                for (const std::int32_t j : own) {
                    g += count[kind] * (0.5 / branch_a_[static_cast<std::size_t>(j)]);
                    state_branch_.push_back(j);
                    state_faces_.push_back(static_cast<float>(count[kind]));
                }
            }
            if (k < 6 || g > 0.0) {
                add_special(i, k, links, half_courant * g);
            } else {
                state_branch_.resize(state_start_.back());
                state_faces_.resize(state_start_.back());
            }
        }
        boundary_ = special_.size();

        // The source's cells (made special, so one update adds the source).
        for (const auto &[i, w] : visible_cells(bvh, source, &type_)) {
            if (type_[i] == kLayer) {
                throw std::invalid_argument("the source is in the matched layer");
            }
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
        w0_.assign(state_branch_.size(), 0.0f);
        w1_.assign(state_branch_.size(), 0.0f);
        threads_ = std::max<std::size_t>(1, std::min(threads_, air_ / 20000 + 1));
    }

    std::size_t air_cells() const { return air_; }
    std::size_t boundary_cells() const { return boundary_; }
    std::size_t branch_states() const { return state_branch_.size(); }
    std::size_t layer_cells() const { return layer_cells_.size(); }
    bool open() const { return open_; }
    std::size_t probes() const { return probe_cells_.size(); }
    bool probe_found(std::size_t p) const { return !probe_cells_.at(p).empty(); }
    std::size_t threads() const { return threads_; }

    // Advances `steps` steps, adding signal[k] at the source on step k, and writes every
    // probe's pressure before each step to out (probes x steps).
    void run(const float *signal, std::size_t steps, double *out) {
        const std::size_t t_count = threads_;
        const std::size_t n = u0_.size(), specials = special_.size();
        const std::size_t layers = layer_cells_.size();
        StepBarrier barrier(t_count);
        float *u0 = u0_.data(), *u1 = u1_.data();
        float *w0 = w0_.data(), *w1 = w1_.data();
        auto work = [&](std::size_t t) {
            FlushDenormals flush;
            const std::size_t ib = n * t / t_count, ie = n * (t + 1) / t_count;
            const std::size_t sb = specials * t / t_count, se = specials * (t + 1) / t_count;
            const std::size_t lb = layers * t / t_count, le = layers * (t + 1) / t_count;
            const std::uint8_t *type = type_.data();
            const std::int64_t *index = special_.data();
            const float *ca = coef_a_.data(), *cb = coef_b_.data(), *ci = coef_inv_.data();
            const std::uint8_t *links = links_.data();
            const float *src = source_.data();
            const std::size_t *start = state_start_.data();
            const std::int32_t *branch = state_branch_.data();
            const float *faces = state_faces_.data();
            const float *now = branch_now_.data(), *before = branch_before_.data();
            const float *gain = branch_gain_.data();
            const std::size_t sy = sy_, sz = sz_;
            // L^2 = 1 / 3, rounded down: rounded up, six of it make more than 2 and the
            // highest mode of a very large grid could grow.
            const float third = std::nextafter(1.0f / 3.0f, 0.0f);
            const float half_courant = 0.5f * std::sqrt(third);
            float *a = u0, *b = u1;  // a: the previous step, then the next; b: the current
            float *wa = w0, *wb = w1;  // branch velocities, the same way
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
                    float flow = 0.0f;
                    for (std::size_t j = start[k]; j < start[k + 1]; ++j) {
                        const std::size_t br = static_cast<std::size_t>(branch[j]);
                        flow += faces[j] * (now[br] * wb[j] - before[br] * wa[j]);
                    }
                    const float previous = a[i];
                    const float next = (ca[k] * b[i] + third * sum - cb[k] * previous -
                                        half_courant * flow + src[k] * s) *
                                       ci[k];
                    a[i] = next;
                    for (std::size_t j = start[k]; j < start[k + 1]; ++j) {
                        const std::size_t br = static_cast<std::size_t>(branch[j]);
                        wa[j] += (next - previous) * gain[br] + (now[br] * wb[j] - before[br] * wa[j]);
                    }
                }
                for (std::size_t c = lb; c < le; ++c) update_layer(c, a, b);
                barrier.wait();
                std::swap(a, b);
                std::swap(wa, wb);
            }
        };
        {
            std::vector<std::thread> pool;
            for (std::size_t t = 1; t < t_count; ++t) pool.emplace_back(work, t);
            work(0);
            for (auto &thread : pool) thread.join();
        }
        if (steps % 2 == 1) {
            std::swap(u0_, u1_);
            std::swap(w0_, w1_);
        }
    }

    void reset() {
        std::fill(u0_.begin(), u0_.end(), 0.0f);
        std::fill(u1_.begin(), u1_.end(), 0.0f);
        std::fill(w0_.begin(), w0_.end(), 0.0f);
        std::fill(w1_.begin(), w1_.end(), 0.0f);
        std::fill(layer_state_.begin(), layer_state_.end(), 0.0f);
    }

private:
    static constexpr std::uint8_t kSolid = 0, kInterior = 1, kSpecial = 2, kLayer = 3;
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

    // Cells from cell i to the grid's edge, the nearest way (0: on the edge).
    std::size_t depth(std::size_t i) const {
        const std::size_t x = i % nx_, y = (i / nx_) % ny_, z = i / sz_;
        return std::min({x, y, z, nx_ - 1 - x, ny_ - 1 - y, nz_ - 1 - z});
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

    // The matched layer's coefficients along axis a: (1 - d) / (1 + d) and L / (1 + d)
    // for the damping d of each cell's centre and of the face below it.
    void layer_coefficients(int a) {
        const std::size_t size = a == 0 ? nx_ : (a == 1 ? ny_ : nz_);
        const double layer = static_cast<double>(layer_);
        const double courant = std::sqrt(static_cast<double>(std::nextafter(1.0f / 3.0f, 0.0f)));
        auto damping = [&](double from_edge) {
            const double into = std::max(0.0, layer + 0.5 - from_edge) / layer;
            return layer_damping_ * into * into;
        };
        auto &centre_keep = layer_keep_[2 * a], &face_keep = layer_keep_[2 * a + 1];
        auto &centre_push = layer_push_[2 * a], &face_push = layer_push_[2 * a + 1];
        for (std::size_t x = 0; x <= size; ++x) {
            const double c = static_cast<double>(x), last = static_cast<double>(size - 1);
            const double dc = damping(std::min(c, last - c));
            const double df = damping(std::min(c - 0.5, last - c + 0.5));
            centre_keep.push_back(static_cast<float>((1.0 - dc) / (1.0 + dc)));
            centre_push.push_back(static_cast<float>(courant / (1.0 + dc)));
            face_keep.push_back(static_cast<float>((1.0 - df) / (1.0 + df)));
            face_push.push_back(static_cast<float>(courant / (1.0 + df)));
        }
    }

    // One step of matched-layer cell c: its six face velocities, then the parts of its
    // pressure, from the current pressures b, into a.
    void update_layer(std::size_t c, float *a, const float *b) {
        const std::size_t i = static_cast<std::size_t>(layer_cells_[c]);
        const std::size_t at[3] = {i % nx_, (i / nx_) % ny_, i / sz_};
        float *state = layer_state_.data() + 9 * c;  // p_x, p_y, p_z, then u- and u+ per axis
        float total = 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            const std::size_t stride = axis == 0 ? 1 : (axis == 1 ? sy_ : sz_);
            const std::size_t x = at[axis];
            const float *fk = layer_keep_[2 * axis + 1].data(), *fp = layer_push_[2 * axis + 1].data();
            float &below = state[3 + 2 * axis], &above = state[4 + 2 * axis];
            below = fk[x] * below - fp[x] * (b[i] - b[i - stride]);
            above = fk[x + 1] * above - fp[x + 1] * (b[i + stride] - b[i]);
            float &part = state[axis];
            part = layer_keep_[2 * axis][x] * part - layer_push_[2 * axis][x] * (above - below);
            total += part;
        }
        a[i] = total;
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

    // Makes cell i special, with the branch velocities added since the last one.
    std::size_t add_special(std::size_t i, int k, std::uint8_t links, double g) {
        type_[i] = kSpecial;
        special_.push_back(static_cast<std::int64_t>(i));
        links_.push_back(links);
        coef_a_.push_back(static_cast<float>(2.0 - k / 3.0));
        coef_b_.push_back(static_cast<float>(1.0 - g));
        coef_inv_.push_back(static_cast<float>(1.0 / (1.0 + g)));
        source_.push_back(0.0f);
        state_start_.push_back(state_branch_.size());
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
    std::size_t layer_ = 0;
    double layer_damping_ = 0.0;
    std::size_t threads_ = 1;
    std::size_t air_ = 0, boundary_ = 0;
    bool open_ = false;
    std::vector<std::uint8_t> type_;
    std::vector<std::int64_t> special_;
    std::vector<std::uint8_t> links_;
    std::vector<float> coef_a_, coef_b_, coef_inv_, source_;
    // Branch velocities: special cell k has states state_start_[k] to state_start_[k + 1],
    // each a branch of a material and the number of the cell's faces it stands for.
    std::vector<std::size_t> state_start_;
    std::vector<std::int32_t> state_branch_;
    std::vector<float> state_faces_;
    std::vector<float> branch_now_, branch_before_, branch_gain_;  // 2m/a, (2m + q)/a, 1/(2a)
    std::vector<double> branch_a_;
    std::vector<float> w0_, w1_;
    // The matched layer: its cells, their state (see update_layer) and, per axis and
    // grid coordinate, the coefficients of the centres and of the faces below them.
    std::vector<std::int64_t> layer_cells_;
    std::vector<float> layer_state_;
    std::array<std::vector<float>, 6> layer_keep_, layer_push_;
    std::vector<std::vector<std::size_t>> probe_cells_;
    std::vector<std::vector<double>> probe_weights_;
    std::vector<float> u0_, u1_;
};

}  // namespace acoustic_spaces
