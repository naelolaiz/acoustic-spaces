// SPDX-License-Identifier: GPL-3.0-or-later
//
// Wave-based room acoustics: compact explicit finite-difference time-domain schemes on a
// cubic grid, with frequency-dependent locally reacting boundaries on the voxelised room
// and, round an open model, a perfectly matched layer.
//
// Two stencils, each at its Courant limit L (the distance sound travels in a time step,
// in cells):
//
// - the standard rectilinear one: 7 points, the six axial neighbours weighted g = 1/3,
//   L = 1/sqrt(3); waves along the axes travel up to 1.7 % slow at 8 cells per
//   wavelength, along the face diagonals 0.44 %, along the body diagonals not at all;
// - the interpolated isotropic one: 19 points, the axial neighbours weighted 1/4 and
//   the twelve edge neighbours 1/8, L = sqrt(3/4); 0.67 % slow at 8 cells per
//   wavelength, within 0.03 % the same in every direction.
//
// Either way a cell's update is a weighted graph Laplacian over its links to its
// neighbours, p+ - 2 p + p- = sum_k g_k (p_k - p) - (boundary terms), its weights
// rounded down so that the highest mode cannot grow.
//
// The grid's cells are air when they can be reached from the source without crossing
// a face of the model, so closed rooms, open models and thin screens all work. A link
// whose segment between cell centres crosses a face is cut: it is part of the
// boundary, of the material of the face it crosses nearest the cell, with weight
// w_k = g_k / (2 L). (A cell's cut links through a plane wall weigh L / 2 together, as
// the one face of the 7-point stencil does.) The material's specific admittance (normal
// velocity times rho c over pressure) is a constant and parallel
// mass-resistance-stiffness branches,
//
//   Y(s) = beta + sum_j 1 / (M_j s + R_j + S_j / s),
//
// passive for beta, M, R, S >= 0. A branch's normalised velocity w (rho c times the
// velocity) follows M w'' + R w' + S w = p', centred in time:
//
//   m (w+ - 2 w + w-) + r (w+ - w-) / 2 + q (w+ + w-) / 2 = (p+ - p-) / 2,
//
// with m = M / T, r = R and q = S T for the time step T. With W_f the weight of the
// cell's cut links of material f, the cell's update is finite-volume style
//
//   p+ - 2 p + p- = sum_k g_k (p_k - p) - sum_f W_f (beta_f (p+ - p-) + sum_j (w_j+ - w_j-)),
//
// which, solved together with the branches, is
//
//   p+ = ((2 - G) p + sum_k g_k p_k - (1 - g) p- - sum_f W_f sum_j c_j) / (1 + g),
//   G = sum_k g_k,  g = sum_f W_f (beta_f + sum_j 1 / (2 a_j)),  a_j = m_j + r_j / 2 + q_j / 2,
//   c_j = (2 m_j w_j - (2 m_j + q_j) w_j-) / a_j,  then w_j+ = w_j- + (p+ - p-) / (2 a_j) + c_j.
//
// Cutting a link removes it from the sum. Of the 19-point stencil's edge links, one
// whose far end lies behind a wall across one of its two axes is folded instead: it
// links the cell to its axial neighbour along the other axis, where the far end's image
// in the wall lies, as a ghost point behind a plane wall would; behind walls across both
// axes it is removed, and blocked between the two (a convex edge) it is split between
// them. Without folding, the cells along a wall would be stiffer across the wall than
// along it, which moves a small room's modes by a percent. The weights are then averaged
// between each pair of cells so that the Laplacian stays symmetric: the scheme's
// energy only falls, and it is stable while the Laplacian's largest eigenvalue stays
// within 4, which it does exactly for planar walls (by their images) and did on every
// voxelised room tried. With no edge links, the 7-point stencil folds nothing.
//
// The matched layer (7-point stencil only): the `layer` cells inside the grid's edge,
// where an open model's air reaches them, are updated in the pressure-velocity form the
// leapfrog is equivalent to, with the pressure split by axis and each part damped along
// its axis (Berenger's split field), the damping growing as the square of the depth into
// the layer. Each layer cell keeps its own copy of the velocity on its six faces; both
// cells of a face update it alike. At the layer's inner side the damping is zero, where
// the two forms agree, and behind it the grid's edge holds no pressure.

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

// The interior update over cells lo to hi, all interior: the 7-point stencil's, and
// the 19-point one's. No test per cell, so the compiler can vectorise them.
inline void update_interior(float *__restrict a, const float *__restrict b, std::size_t lo,
                            std::size_t hi, std::size_t sy, std::size_t sz, float axial) {
    for (std::size_t i = lo; i < hi; ++i) {
        const float sum = b[i - 1] + b[i + 1] + b[i - sy] + b[i + sy] + b[i - sz] + b[i + sz];
        a[i] = axial * sum - a[i];
    }
}

inline void update_interior(float *__restrict a, const float *__restrict b, std::size_t lo,
                            std::size_t hi, std::size_t sy, std::size_t sz, float axial,
                            float diagonal, float centre) {
    for (std::size_t i = lo; i < hi; ++i) {
        const float sum = b[i - 1] + b[i + 1] + b[i - sy] + b[i + sy] + b[i - sz] + b[i + sz];
        const float edges = b[i - 1 - sy] + b[i + 1 - sy] + b[i - 1 + sy] + b[i + 1 + sy] +
                            b[i - 1 - sz] + b[i + 1 - sz] + b[i - 1 + sz] + b[i + 1 + sz] +
                            b[i - sy - sz] + b[i + sy - sz] + b[i - sy + sz] + b[i + sy + sz];
        a[i] = centre * b[i] + axial * sum + diagonal * edges - a[i];
    }
}

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
    // x fastest; the outermost layer is never air. `stencil` is 7 or 19 (see above).
    // Face f of `bvh` is of material face_material[f], of `materials`: beta[material],
    // and the branches whose branch_material is that material, branches holding m, r and
    // q of each (see above). The grid's edge, which only an open model's air reaches, has
    // `edge_admittance`, or with `layer` > 0 a matched layer of that many cells inside
    // it, damped up to `layer_damping` (s T / 2, s the damping rate) at the edge. The
    // source and every probe take the air cells round them that they can see, weighted
    // trilinearly.
    WaveSolver(const Bvh &bvh, const std::int64_t *face_material, const double *beta,
               std::size_t materials, const double *branches, const std::int64_t *branch_material,
               std::size_t branch_count, const Vec3 &origin, double cell, std::size_t nx,
               std::size_t ny, std::size_t nz, const Vec3 &source,
               const std::vector<Vec3> &probes, double edge_admittance, std::size_t layer,
               double layer_damping, std::size_t threads, std::size_t stencil = 7)
        : origin_(origin), cell_(cell), nx_(nx), ny_(ny), nz_(nz), sy_(nx), sz_(nx * ny),
          layer_(layer), stencil_(stencil) {
        if (stencil != 7 && stencil != 19) {
            throw std::invalid_argument("the stencil has 7 or 19 points");
        }
        if (layer > 0 && stencil != 7) {
            throw std::invalid_argument("the matched layer needs the 7-point stencil");
        }
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
        set_up_stencil();
        threads_ = threads ? threads : std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t n = nx * ny * nz;

        // Links crossing a face, each pair of neighbours once: bit b of cut[i] is set
        // when the link to i's neighbour along the b-th direction of the + half crosses
        // one.
        // Only cells a face passes near can have such links, so only they cast rays.
        const std::vector<std::uint8_t> near = near_faces(bvh);
        std::vector<std::uint16_t> cut(n, 0);
        parallel_for(n, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                if (!near[i]) continue;
                const long long at[3] = {static_cast<long long>(i % nx),
                                         static_cast<long long>((i / nx) % ny),
                                         static_cast<long long>(i / sz_)};
                const long long size[3] = {static_cast<long long>(nx), static_cast<long long>(ny),
                                           static_cast<long long>(nz)};
                const Vec3 c = centre(i);
                for (int k = 0; k < directions_; ++k) {
                    if (bit_[k] < 0) continue;
                    bool inside = true;
                    for (int a = 0; a < 3; ++a) {
                        const long long to = at[a] + kSteps[k][a];
                        inside = inside && to >= 0 && to < size[a];
                    }
                    if (!inside) continue;
                    if (bvh.intersect(c, unit(k), length(k), -1).found()) {
                        cut[i] = static_cast<std::uint16_t>(cut[i] | (1u << bit_[k]));
                    }
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
            for (int k = 0; k < directions_; ++k) {
                if (blocked(cut, i, k)) continue;
                const std::size_t j = neighbour(i, k);
                if (type_[j] == kSolid) {
                    type_[j] = kInterior;
                    queue.push_back(j);
                }
            }
        }

        // The matched layer's air, which must meet no face.
        for (std::size_t i = 0; i < n && layer_ > 0; ++i) {
            if (type_[i] == kSolid || depth(i) > layer_) continue;
            for (int k = 0; k < 6; ++k) {
                if (!edge(neighbour(i, k)) && is_cut(cut, i, k)) {
                    throw std::invalid_argument("the model reaches into the matched layer");
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

        // Boundary cells: a link cut or at the grid's edge (mark 1), or a neighbour of
        // one whose folded links reach it (mark 2), whose weights averaging changes.
        std::vector<std::uint8_t> mark(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            if (type_[i] != kInterior) continue;
            for (int k = 0; k < directions_ && !mark[i]; ++k) mark[i] = blocked(cut, i, k);
        }
        std::array<double, 18> wi{}, wj{};
        for (std::size_t i = 0; i < n; ++i) {
            if (mark[i] != 1) continue;
            raw_weights(cut, i, wi.data());
            for (int k = 0; k < directions_; ++k) {
                const std::size_t j = neighbour(i, k);
                if (wi[k] != exact_[k] && type_[j] == kInterior && !mark[j]) mark[j] = 2;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (type_[i] != kInterior) continue;
            ++air_;
            if (!mark[i]) continue;
            raw_weights(cut, i, wi.data());
            std::array<double, 18> weight{};
            double beta_sum = 0.0;
            std::array<std::int32_t, 18> crossed{};  // materials of the cut links,
            std::array<double, 18> crossed_weight{};  // and their weight
            std::size_t kinds = 0;
            // A cut link's boundary weight goes to the face it crosses nearest the cell;
            // an edge link's, to the wall behind each of its steps that one is behind.
            auto meet = [&](int k, double w) {
                const std::size_t j = neighbour(i, k);
                if (edge(j)) {
                    beta_sum += w * edge_admittance;
                    open_ = true;
                    return;
                }
                Hit hit = bvh.intersect(centre(i), unit(k), length(k), -1);
                if (!hit.found()) hit = bvh.intersect(centre(j), unit(opposite(k)), length(k), -1);
                if (!hit.found()) return;  // grazing a face's edge: as if rigid
                const auto m = static_cast<std::int32_t>(face_material[hit.face]);
                beta_sum += w * beta[m];
                std::size_t kind = 0;
                while (kind < kinds && crossed[kind] != m) ++kind;
                if (kind == kinds) crossed[kinds++] = m;
                crossed_weight[kind] += w;
            };
            for (int k = 0; k < directions_; ++k) {
                const std::size_t j = neighbour(i, k);
                if (!edge(j) && type_[j] != kSolid && type_[j] != kLayer && mark[j]) {
                    raw_weights(cut, j, wj.data());
                    weight[k] = 0.5 * (wi[k] + wj[opposite(k)]);
                } else if (!edge(j) && type_[j] != kSolid) {
                    weight[k] = 0.5 * (wi[k] + exact_[k]);
                }
                if (!blocked(cut, i, k)) continue;
                if (k < 6) {
                    meet(k, boundary_[k]);
                    continue;
                }
                const auto [ka, kb] = steps_of(k);
                const bool behind_a = blocked(cut, i, ka), behind_b = blocked(cut, i, kb);
                if (behind_a) meet(ka, boundary_[k]);
                if (behind_b) meet(kb, boundary_[k]);
                if (!behind_a && !behind_b) meet(k, boundary_[k]);
            }
            // One set of branch velocities per material the cell meets, standing for its
            // cut links of that material.
            double g = beta_sum;
            for (std::size_t kind = 0; kind < kinds; ++kind) {
                const auto &own = material_branches[static_cast<std::size_t>(crossed[kind])];
                for (const std::int32_t j : own) {
                    g += crossed_weight[kind] * (0.5 / branch_a_[static_cast<std::size_t>(j)]);
                    state_branch_.push_back(j);
                    state_weight_.push_back(static_cast<float>(crossed_weight[kind]));
                }
            }
            add_special(i, weight.data(), g);
        }
        boundary_cells_ = special_.size();

        // The source's cells (made special, so one update adds the source).
        for (const auto &[i, w] : visible_cells(bvh, source, &type_)) {
            if (type_[i] == kLayer) {
                throw std::invalid_argument("the source is in the matched layer");
            }
            std::size_t k = special_index(i);
            if (k == npos) k = add_special(i, exact_.data(), 0.0);
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
        // The runs of consecutive interior cells, which the update streams through.
        for (std::size_t i = 0; i < n;) {
            if (type_[i] != kInterior) {
                ++i;
                continue;
            }
            run_begin_.push_back(i);
            while (i < n && type_[i] == kInterior) ++i;
            run_end_.push_back(i);
        }
        u0_.assign(n, 0.0f);
        u1_.assign(n, 0.0f);
        w0_.assign(state_branch_.size(), 0.0f);
        w1_.assign(state_branch_.size(), 0.0f);
        threads_ = std::max<std::size_t>(1, std::min(threads_, air_ / 20000 + 1));
    }

    std::size_t air_cells() const { return air_; }
    std::size_t boundary_cells() const { return boundary_cells_; }
    std::size_t branch_states() const { return state_branch_.size(); }
    std::size_t layer_cells() const { return layer_cells_.size(); }
    bool open() const { return open_; }
    std::size_t probes() const { return probe_cells_.size(); }
    bool probe_found(std::size_t p) const { return !probe_cells_.at(p).empty(); }
    std::size_t stencil() const { return stencil_; }
    std::size_t threads() const { return threads_; }

    // Hands each array of the grid's layout to put(name, vector), for stepping the grid
    // elsewhere (on a GPU: engine/gpu.py); the grid's state is not among them.
    template <typename Put>
    void layout(Put &&put) const {
        put("shape", std::vector<std::size_t>{nx_, ny_, nz_});
        put("stencil", std::vector<std::size_t>{stencil_});
        put("weights", std::vector<float>(weight_.begin(), weight_.begin() + directions_));
        put("centre", std::vector<float>{centre_});
        put("types", type_);
        put("special", special_);
        put("link_start", link_start_);
        put("link_direction", link_direction_);
        put("link_weight", link_weight_);
        put("coef_a", coef_a_);
        put("coef_b", coef_b_);
        put("coef_inv", coef_inv_);
        put("source", source_);
        put("state_start", state_start_);
        put("state_branch", state_branch_);
        put("state_weight", state_weight_);
        put("branch_now", branch_now_);
        put("branch_before", branch_before_);
        put("branch_gain", branch_gain_);
        put("layer_cells", layer_cells_);
        // Per axis, the centres' then the faces' coefficients (see layer_coefficients).
        static const char *const keep[6] = {"keep_x", "keep_face_x", "keep_y",
                                            "keep_face_y", "keep_z", "keep_face_z"};
        static const char *const push[6] = {"push_x", "push_face_x", "push_y",
                                            "push_face_y", "push_z", "push_face_z"};
        for (int k = 0; k < 6; ++k) {
            put(keep[k], layer_keep_[k]);
            put(push[k], layer_push_[k]);
        }
        std::vector<std::size_t> probe_start{0}, probe_cells;
        std::vector<double> probe_weights;
        for (std::size_t p = 0; p < probe_cells_.size(); ++p) {
            probe_cells.insert(probe_cells.end(), probe_cells_[p].begin(), probe_cells_[p].end());
            probe_weights.insert(probe_weights.end(), probe_weights_[p].begin(),
                                 probe_weights_[p].end());
            probe_start.push_back(probe_cells.size());
        }
        put("probe_start", probe_start);
        put("probe_cells", probe_cells);
        put("probe_weights", probe_weights);
    }

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
            const std::size_t *run_begin = run_begin_.data(), *run_end = run_end_.data();
            const std::size_t runs = run_begin_.size();
            // This thread's first run: the first to end past ib.
            const std::size_t first_run = static_cast<std::size_t>(
                std::upper_bound(run_end_.begin(), run_end_.end(), ib) - run_end_.begin());
            const std::int64_t *index = special_.data();
            const float *ca = coef_a_.data(), *cb = coef_b_.data(), *ci = coef_inv_.data();
            const std::size_t *link_start = link_start_.data();
            const std::uint8_t *link_direction = link_direction_.data();
            const float *link_weight = link_weight_.data();
            const float *src = source_.data();
            const std::size_t *start = state_start_.data();
            const std::int32_t *branch = state_branch_.data();
            const float *state_weight = state_weight_.data();
            const float *now = branch_now_.data(), *before = branch_before_.data();
            const float *gain = branch_gain_.data();
            const std::ptrdiff_t *offset = offset_.data();
            const std::size_t sy = sy_, sz = sz_;
            const bool wide = stencil_ == 19;
            const float axial = weight_[0], diagonal = weight_[6], centre = centre_;
            float *a = u0, *b = u1;  // a: the previous step, then the next; b: the current
            float *wa = w0, *wb = w1;  // branch velocities, the same way
            for (std::size_t step = 0; step < steps; ++step) {
                if (t == 0) record(b, step, steps, out);
                for (std::size_t r = first_run; r < runs && run_begin[r] < ie; ++r) {
                    const std::size_t lo = std::max(run_begin[r], ib);
                    const std::size_t hi = std::min(run_end[r], ie);
                    if (wide) {
                        update_interior(a, b, lo, hi, sy, sz, axial, diagonal, centre);
                    } else {
                        update_interior(a, b, lo, hi, sy, sz, axial);
                    }
                }
                const float s = signal[step];
                for (std::size_t k = sb; k < se; ++k) {
                    const std::size_t i = static_cast<std::size_t>(index[k]);
                    float sum = 0.0f;
                    for (std::size_t l = link_start[k]; l < link_start[k + 1]; ++l) {
                        const std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) + offset[link_direction[l]];
                        sum += link_weight[l] * b[j];
                    }
                    float flow = 0.0f;
                    for (std::size_t j = start[k]; j < start[k + 1]; ++j) {
                        const std::size_t br = static_cast<std::size_t>(branch[j]);
                        flow += state_weight[j] * (now[br] * wb[j] - before[br] * wa[j]);
                    }
                    const float previous = a[i];
                    const float next =
                        (ca[k] * b[i] + sum - cb[k] * previous - flow + src[k] * s) * ci[k];
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
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
    // The directions to a cell's neighbours: the six axial ones (2 a + side, side 1 the
    // + side), then the twelve edge ones, 6 + 4 p + sa + 2 sb for the pair of axes p
    // ((x, y), (x, z), (y, z)) and the signs sa, sb of its steps along them (1: +).
    static constexpr int kSteps[18][3] = {
        {-1, 0, 0},  {1, 0, 0},  {0, -1, 0},  {0, 1, 0},  {0, 0, -1}, {0, 0, 1},
        {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},  {1, 1, 0},  {-1, 0, -1}, {1, 0, -1},
        {-1, 0, 1},  {1, 0, 1},  {0, -1, -1}, {0, 1, -1}, {0, -1, 1},  {0, 1, 1}};

    static int opposite(int k) { return k < 6 ? k ^ 1 : 6 + 4 * ((k - 6) / 4) + 3 - (k - 6) % 4; }

    // The axial directions of edge direction k's two steps.
    static std::array<int, 2> steps_of(int k) {
        static constexpr int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        const int p = (k - 6) / 4, sa = (k - 6) % 2, sb = ((k - 6) / 2) % 2;
        return {2 * pairs[p][0] + sa, 2 * pairs[p][1] + sb};
    }

    // The stencil's weights (exact, and rounded down for the update), boundary weights,
    // offsets and the bits of the + half's directions in the cut mask.
    void set_up_stencil() {
        directions_ = stencil_ == 7 ? 6 : 18;
        const double axial = stencil_ == 7 ? 1.0 / 3.0 : 0.25;
        const double diagonal = stencil_ == 7 ? 0.0 : 0.125;
        const double courant = std::sqrt(axial + 4.0 * diagonal);  // sum g_k d_x^2 = 2 L^2
        int bits = 0;
        for (int k = 0; k < 18; ++k) {
            exact_[k] = k < directions_ ? (k < 6 ? axial : diagonal) : 0.0;
            weight_[k] = down(exact_[k]);
            boundary_[k] = exact_[k] / (2.0 * courant);
            offset_[k] = kSteps[k][0] + kSteps[k][1] * static_cast<std::ptrdiff_t>(sy_) +
                         kSteps[k][2] * static_cast<std::ptrdiff_t>(sz_);
            const int *d = kSteps[k];
            const bool plus = d[2] > 0 || (d[2] == 0 && (d[1] > 0 || (d[1] == 0 && d[0] > 0)));
            bit_[k] = k < directions_ && plus ? bits++ : -1;
        }
        double total = 0.0;
        for (int k = 0; k < directions_; ++k) total += exact_[k];
        centre_ = static_cast<float>(2.0 - total);
    }

    // The largest float not above w.
    static float down(double w) {
        float f = static_cast<float>(w);
        if (static_cast<double>(f) > w) f = std::nextafter(f, 0.0f);
        return f;
    }

    Vec3 unit(int k) const {
        const int *d = kSteps[k];
        const double scale = 1.0 / std::sqrt(static_cast<double>(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
        return {d[0] * scale, d[1] * scale, d[2] * scale};
    }

    double length(int k) const { return cell_ * (k < 6 ? 1.0 : std::sqrt(2.0)); }

    std::size_t neighbour(std::size_t i, int k) const {
        return static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) + offset_[k]);
    }

    // Whether the link from cell i (not on the grid's edge) along direction k crosses a
    // face.
    bool is_cut(const std::vector<std::uint16_t> &cut, std::size_t i, int k) const {
        if (bit_[k] >= 0) return (cut[i] >> bit_[k]) & 1u;
        return (cut[neighbour(i, k)] >> bit_[opposite(k)]) & 1u;
    }

    // Whether cell i has no link along direction k: it is cut, or reaches the grid's edge.
    bool blocked(const std::vector<std::uint16_t> &cut, std::size_t i, int k) const {
        return edge(neighbour(i, k)) || is_cut(cut, i, k);
    }

    // The weights of air cell i's links, folded (see above), before averaging.
    void raw_weights(const std::vector<std::uint16_t> &cut, std::size_t i, double *w) const {
        for (int k = 0; k < 18; ++k) w[k] = 0.0;
        for (int k = 0; k < directions_; ++k) {
            if (!blocked(cut, i, k)) {
                w[k] += exact_[k];
                continue;
            }
            if (k < 6 || edge(neighbour(i, k))) continue;
            const auto [ka, kb] = steps_of(k);
            const bool behind_a = blocked(cut, i, ka), behind_b = blocked(cut, i, kb);
            if (behind_a && behind_b) continue;
            if (behind_a) {
                w[kb] += exact_[k];
            } else if (behind_b) {
                w[ka] += exact_[k];
            } else {
                w[ka] += 0.5 * exact_[k];
                w[kb] += 0.5 * exact_[k];
            }
        }
    }

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

    // The cells some face passes within the cube of half-width one cell round their
    // centre, which holds the links of the + half of every stencil: those within its
    // bounding box, grown by a cell, that are no further from its plane than the
    // cube's corners. Every cell when the faces' boxes hold too many cells to visit.
    std::vector<std::uint8_t> near_faces(const Bvh &bvh) const {
        const std::size_t n = nx_ * ny_ * nz_;
        const double reach = cell_ * (1.0 + 1e-9), radius = std::sqrt(3.0) * reach;
        const double low[3] = {origin_.x, origin_.y, origin_.z};
        const std::size_t size[3] = {nx_, ny_, nz_};
        std::vector<std::uint8_t> near(n, 0);
        std::size_t visits = 0;
        for (std::size_t f = 0; f < bvh.face_count(); ++f) {
            const std::array<Vec3, 3> p = bvh.corners(f);
            std::size_t from[3] = {}, to[3] = {}, count = 1;
            for (int a = 0; a < 3; ++a) {
                const double lo = std::min({p[0][a], p[1][a], p[2][a]}) - reach;
                const double hi = std::max({p[0][a], p[1][a], p[2][a]}) + reach;
                // Centres at low + (x + 0.5) cell within [lo, hi].
                const double first = std::ceil((lo - low[a]) / cell_ - 0.5);
                const double last = std::floor((hi - low[a]) / cell_ - 0.5);
                if (last < 0.0 || first > static_cast<double>(size[a] - 1)) {
                    count = 0;
                    break;
                }
                from[a] = static_cast<std::size_t>(std::max(first, 0.0));
                to[a] = std::min(static_cast<std::size_t>(last), size[a] - 1);
                count *= to[a] - from[a] + 1;
            }
            if (count == 0) continue;
            visits += count;
            if (visits > 8 * n) return std::vector<std::uint8_t>(n, 1);
            const Vec3 &normal = bvh.normal(f);
            for (std::size_t z = from[2]; z <= to[2]; ++z) {
                for (std::size_t y = from[1]; y <= to[1]; ++y) {
                    for (std::size_t x = from[0]; x <= to[0]; ++x) {
                        const std::size_t i = x + y * sy_ + z * sz_;
                        if (std::abs(dot(normal, centre(i) - p[0])) <= radius) near[i] = 1;
                    }
                }
            }
        }
        return near;
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

    // Makes cell i special, with the weights w of its links along each direction and
    // the branch velocities added since the last one.
    std::size_t add_special(std::size_t i, const double *w, double g) {
        type_[i] = kSpecial;
        special_.push_back(static_cast<std::int64_t>(i));
        double total = 0.0;
        for (int k = 0; k < directions_; ++k) {
            if (!(w[k] > 0.0)) continue;
            total += w[k];
            link_direction_.push_back(static_cast<std::uint8_t>(k));
            link_weight_.push_back(down(w[k]));
        }
        link_start_.push_back(link_direction_.size());
        coef_a_.push_back(static_cast<float>(2.0 - total));
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
    std::size_t stencil_ = 7;
    double layer_damping_ = 0.0;
    std::size_t threads_ = 1;
    std::size_t air_ = 0, boundary_cells_ = 0;
    bool open_ = false;
    // The stencil: its directions, their weights (exact, and as the update uses them),
    // the weight of a cut link as a boundary, the offset to the neighbour, the bit of
    // the + half's directions in the cut mask (-1: the - half), and the centre's weight.
    int directions_ = 6;
    std::array<double, 18> exact_{}, boundary_{};
    std::array<float, 18> weight_{};
    std::array<std::ptrdiff_t, 18> offset_{};
    std::array<int, 18> bit_{};
    float centre_ = 0.0f;
    std::vector<std::uint8_t> type_;
    std::vector<std::int64_t> special_;
    // Special cell k's links: link_start_[k] to link_start_[k + 1], each a direction and
    // its weight.
    std::vector<std::size_t> link_start_{0};
    std::vector<std::uint8_t> link_direction_;
    std::vector<float> link_weight_;
    std::vector<float> coef_a_, coef_b_, coef_inv_, source_;
    // Branch velocities: special cell k has states state_start_[k] to state_start_[k + 1],
    // each a branch of a material and the weight of the cell's cut links it stands for.
    std::vector<std::size_t> state_start_{0};
    std::vector<std::int32_t> state_branch_;
    std::vector<float> state_weight_;
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
    std::vector<std::size_t> run_begin_, run_end_;
    std::vector<float> u0_, u1_;
};

}  // namespace acoustic_spaces
