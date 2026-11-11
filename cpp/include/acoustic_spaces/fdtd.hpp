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
// neighbours, p+ - 2 p + p- = sum_k g_k (p_k - p) - (boundary terms). At the Courant
// limit its highest modes lie on the edge of stability even in exact arithmetic (the
// 7-point stencil's checkerboard and the 19-point's modes alternating along two axes,
// p+ + 2 p + p- = 0, under which a kick grows linearly), and single precision's
// rounding can take them over it. So the update takes the Laplacian times 1 - 2^-18,
// each link weighing g_k (1 - 2^-18) rounded down and the centre 2 less their sum:
// sound goes 2^-19 slower, and those modes lie 2^-16 inside the limit and, kicked
// once, swing within 1 / sin of their angle a step, 256 times the kick (259 over
// 300,000 steps in single precision). The uniform pressure stays as a closed room's
// should: what the source pushes in stays.
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
//
// The speed of sound may change from cell to cell (outdoors, in wind and temperature
// that change with height): each cell's ratio r of its speed to the grid's, at most 1,
// the grid's time step being set by the fastest. The wave equation p_tt = c^2 lap p then
// weighs each cell's links by s = r^2 (its row of the Laplacian, so the update stays
// similar to a symmetric one, with eigenvalues s times the Laplacian's at most, and
// stable), its boundary terms by r (the boundary's admittance is relative to the local
// rho c, and a cut link's term is its weight over 2 L, L the local Courant number r times
// the grid's), the source by s, and in the matched layer the pressure's update from the
// velocities by s. With every r 1 the scheme is the one above, to the bit.
//
// The air may move (outdoors, in the wind): each cell's horizontal velocity, as shifts
// along x and y in cells per step. Before each step both of the scheme's pressures, the
// current one and the one before, are carried along with the air for a step, d q / d t =
// -v . grad q, and then the still air's step follows. In air moving as a whole this is
// the still air's scheme in the frame of the air, exactly but for the carrying: the
// wave equation of a uniformly moving medium, (d/dt + v . grad)^2 p = c^2 lap p, whose
// sound goes at c + v downwind and c - v upwind whichever way it turned on its way.
// Where the wind changes with height, the layers of air slide over one another; the
// coupling of the sound with the wind's shear, (u . grad) v in the linearised Euler
// equations, is left out (in a stratified wind it is a share of about |dv/dz| / omega of
// the wind's effect). The carrying is the strong-stability-preserving Runge-Kutta
// method of third order over the gradient's central differences, of fourth order where
// two cells either side are air, second where one is; next to a face, the upwind
// difference where the wind blows into it, none where it blows out of it. Over its
// small steps (|v| T / h is under 0.1 for wind) it loses a little of the shortest
// waves and no more, so the scheme stays stable. The surfaces' branch velocities, the
// source and the probes stay where they are, and the matched layer's air is still.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <unordered_map>
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
                            std::size_t hi, std::size_t sy, std::size_t sz, float axial,
                            float centre) {
    for (std::size_t i = lo; i < hi; ++i) {
        const float sum = b[i - 1] + b[i + 1] + b[i - sy] + b[i + sy] + b[i - sz] + b[i + sz];
        a[i] = centre * b[i] + axial * sum - a[i];
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

// The same, each cell's links weighed by its scale[i] (the square of its speed of sound
// over the grid's) and its centre by 2 - total scale[i], total the stencil's weights
// summed (2 and 3 less the margin). With scale 1 the update is the one above, to the bit.
inline void update_interior(float *__restrict a, const float *__restrict b,
                            const float *__restrict scale, std::size_t lo, std::size_t hi,
                            std::size_t sy, std::size_t sz, float axial, float total) {
    for (std::size_t i = lo; i < hi; ++i) {
        const float s = scale[i];
        const float sum = b[i - 1] + b[i + 1] + b[i - sy] + b[i + sy] + b[i - sz] + b[i + sz];
        a[i] = (2.0f - total * s) * b[i] + (s * axial) * sum - a[i];
    }
}

inline void update_interior(float *__restrict a, const float *__restrict b,
                            const float *__restrict scale, std::size_t lo, std::size_t hi,
                            std::size_t sy, std::size_t sz, float axial, float diagonal,
                            float total) {
    for (std::size_t i = lo; i < hi; ++i) {
        const float s = scale[i];
        const float sum = b[i - 1] + b[i + 1] + b[i - sy] + b[i + sy] + b[i - sz] + b[i + sz];
        const float edges = b[i - 1 - sy] + b[i + 1 - sy] + b[i - 1 + sy] + b[i + 1 + sy] +
                            b[i - 1 - sz] + b[i + 1 - sz] + b[i - 1 + sz] + b[i + 1 + sz] +
                            b[i - sy - sz] + b[i + sy - sz] + b[i - sy + sz] + b[i + sy + sz];
        a[i] = (2.0f - total * s) * b[i] + (s * axial) * sum + (s * diagonal) * edges - a[i];
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
    // source and every probe take the four air cells a side round them, weighted by a
    // cubic through their centres; next to a surface along the grid fewer along the axis
    // across it; next to another surface or to the source the cells round them that they
    // see, weighted trilinearly (see reading_cells). `speed`, when
    // given, holds each cell's speed of sound over the grid's (in (0, 1], x fastest; see
    // above); without it the speed is the same everywhere.
    WaveSolver(const Bvh &bvh, const std::int64_t *face_material, const double *beta,
               std::size_t materials, const double *branches, const std::int64_t *branch_material,
               std::size_t branch_count, const Vec3 &origin, double cell, std::size_t nx,
               std::size_t ny, std::size_t nz, const Vec3 &source,
               const std::vector<Vec3> &probes, double edge_admittance, std::size_t layer,
               double layer_damping, std::size_t threads, std::size_t stencil = 7,
               const float *speed = nullptr, const float *flow_x = nullptr,
               const float *flow_y = nullptr, std::size_t fitted = 0)
        : origin_(origin), cell_(cell), nx_(nx), ny_(ny), nz_(nz), sy_(nx), sz_(nx * ny),
          layer_(layer), stencil_(stencil) {
        fitted_ = fitted;
        // 19-point: no stable fitted variant that helps was found (prototype). The fit is
        // used only when an experimental variant is named: "area" (unstable) or "volume".
        const char *v19 = std::getenv("ACOUSTIC_SPACES_FIT19");
        if (stencil == 19 && !(v19 && (std::strcmp(v19, "volume") == 0 || std::strcmp(v19, "area") == 0))) {
            fitted_ = 0;
        }
        if (v19) fit19_volume_ = std::strcmp(v19, "volume") == 0;
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
        threads_ = worker_threads(threads);
        const std::size_t n = nx * ny * nz;
        if (speed != nullptr) {
            scale_.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                const double r = static_cast<double>(speed[i]);
                if (!(r > 0.0 && r <= 1.0)) {
                    throw std::invalid_argument(
                        "each cell's speed of sound over the grid's lies in (0, 1]");
                }
                scale_[i] = static_cast<float>(r * r);
            }
        }

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
            reach_sides(i, layer_);
        }
        if (!layer_cells_.empty()) {
            layer_damping_ = layer_damping;
            for (int a = 0; a < 3; ++a) layer_coefficients(a);
            layer_state_.assign(9 * layer_cells_.size(), 0.0f);
            for (std::size_t c = 0; c < layer_cells_.size() && !scale_.empty(); ++c) {
                layer_scale_.push_back(scale_[static_cast<std::size_t>(layer_cells_[c])]);
            }
        }

        // PROTOTYPE (fitted > 0): each air cell's share of air and each link's open share,
        // sampled fitted x fitted (x fitted) per cell (see fit_boundary).
        std::vector<std::uint8_t> candidate;
        if (fitted_ > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            const char *sampled = std::getenv("ACOUSTIC_SPACES_FIT_SAMPLED");
            fit_exact_ = !(sampled && std::strcmp(sampled, "1") == 0);
            if (fit_exact_) {
                // Round 2: exact clipping; special only the cells that differ from the interior.
                candidate.assign(n, 0);
                for (const std::size_t i : fit_boundary_exact(bvh, face_material)) {
                    if (i >= n || type_[i] != kInterior) continue;
                    bool differs = std::abs(fit_volume(i) - 1.0) > 1e-12;
                    for (int k = 0; k < 6 && !differs; ++k) differs = std::abs(axial_area(cut, i, k) - 1.0) > 1e-12;
                    for (int k = 6; k < 18 && !differs; ++k) differs = diag_weight(i, k) > 0.0;
                    const auto wt = fit_wall_.find(i);
                    if (wt != fit_wall_.end()) for (const auto &e : wt->second) differs = differs || e.second > 0.0;
                    candidate[i] = differs ? 1 : 3;  // 3: touched, as the interior
                }
            } else {
                candidate = fit_boundary(bvh, near, cut);
            }
            fit_stats_[7] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        auto raw = [&](std::size_t i, double *w) {
            if (fitted_ > 0) {
                raw_fitted(cut, i, w);
            } else {
                raw_weights(cut, i, w);
            }
        };
        // Boundary cells: a link cut or at the grid's edge (mark 1), or a neighbour of
        // one whose folded links reach it (mark 2), whose weights averaging changes.
        std::vector<std::uint8_t> mark(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            if (type_[i] != kInterior) continue;
            if (fitted_ > 0 && fit_exact_ && candidate[i]) {
                mark[i] = candidate[i] == 1;
                continue;
            }
            for (int k = 0; k < directions_ && !mark[i]; ++k) mark[i] = blocked(cut, i, k);
            if (fitted_ > 0 && candidate[i]) mark[i] = 1;
        }
        std::array<double, 18> wi{}, wj{};
        for (std::size_t i = 0; i < n; ++i) {
            if (mark[i] != 1) continue;
            raw(i, wi.data());
            for (int k = 0; k < directions_; ++k) {
                const std::size_t j = neighbour(i, k);
                if (((fitted_ > 0 && !fit_exact_) || wi[k] != exact_[k]) && type_[j] == kInterior && !mark[j]) mark[j] = 2;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (type_[i] != kInterior) continue;
            ++air_;
            if (!mark[i]) continue;
            raw(i, wi.data());
            std::array<double, 18> weight{};
            double beta_sum = 0.0;
            std::array<std::int32_t, 18> crossed{};  // materials of the cut links,
            std::array<double, 18> crossed_weight{};  // and their weight
            std::size_t kinds = 0;
            // A cut link's boundary weight goes to the face it crosses nearest the cell;
            // an edge link's, to the wall behind each of its steps that one is behind.
            double met = 0.0;  // the boundary weight handed to meet
            // PROTOTYPE (fitted): the walls met, by direction (up to 3), and each meet's
            // material (-1: the grid's edge) and weight.
            std::array<Vec3, 3> wall_normal{};
            std::size_t walls = 0;
            bool many_walls = false;
            std::array<std::int32_t, 40> entry_material{};
            std::array<double, 40> entry_weight{};
            std::array<std::size_t, 40> entry_wall{};
            std::size_t entries = 0;
            auto record = [&](Vec3 nrm, const Vec3 &dir, std::int32_t m, double w) {
                if (fitted_ == 0) return;
                if (dot(nrm, dir) < 0.0) nrm = nrm * -1.0;
                std::size_t g = 0;
                while (g < walls && dot(wall_normal[g], nrm) < 0.999) ++g;
                if (g == walls) {
                    if (walls == 3) {
                        many_walls = true;
                        g = 0;
                    } else {
                        wall_normal[walls++] = nrm;
                    }
                }
                if (entries < entry_weight.size()) {
                    entry_material[entries] = m;
                    entry_weight[entries] = w;
                    entry_wall[entries++] = g;
                } else {
                    many_walls = true;
                }
            };
            auto meet = [&](int k, double w) {
                met += w;
                const std::size_t j = neighbour(i, k);
                if (edge(j)) {
                    beta_sum += w * edge_admittance;
                    record(unit(k) * (1.0 / norm(unit(k))), unit(k), -1, w);
                    open_ = true;
                    reach_sides(j, 0);
                    return;
                }
                Hit hit = bvh.intersect(centre(i), unit(k), length(k), -1);
                if (!hit.found()) hit = bvh.intersect(centre(j), unit(opposite(k)), length(k), -1);
                if (!hit.found()) return;  // grazing a face's edge: as if rigid
                const auto m = static_cast<std::int32_t>(face_material[hit.face]);
                record(bvh.normal(static_cast<std::size_t>(hit.face)), unit(k), m, w);
                beta_sum += w * beta[m];
                std::size_t kind = 0;
                while (kind < kinds && crossed[kind] != m) ++kind;
                if (kind == kinds) crossed[kinds++] = m;
                crossed_weight[kind] += w;
            };
            for (int k = 0; k < directions_; ++k) {
                const std::size_t j = neighbour(i, k);
                if (!edge(j) && type_[j] != kSolid && type_[j] != kLayer && mark[j]) {
                    raw(j, wj.data());
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
            if (fitted_ > 0 && fit_exact_ && stencil_ == 7 && diag_mode_ > 0) {
                for (int k = 6; k < 18; ++k) weight[k] = diag_weight(i, k);
            }
            double volume = 1.0;
            if (fitted_ > 0) {
                // The wall's area in the cell: the open faces' area vector, closed by it.
                Vec3 net{0.0, 0.0, 0.0};
                for (int k = 0; k < 6; ++k) net = net + unit(k) * axial_area(cut, i, k);
                const double area = norm(net);
                const double plane = 0.5 * std::sqrt(exact_[0] + 4.0 * exact_[6]);  // L / 2
                const double fit = plane * area;
                // Walls in more than one direction (a corner, an edge of the room, the
                // grid's edge): |net| is not their area. The wall vector -net is the sum of
                // each wall's area times its normal; solve for the areas (least squares).
                std::array<double, 3> wall_area{};
                bool solved = false;
                if (met > 0.0 && walls >= 2 && !many_walls) {
                    const Vec3 W = net * -1.0;
                    double M[3][4] = {};
                    for (std::size_t a = 0; a < walls; ++a) {
                        for (std::size_t b = 0; b < walls; ++b) M[a][b] = dot(wall_normal[a], wall_normal[b]);
                        M[a][3] = dot(wall_normal[a], W);
                    }
                    solved = true;
                    for (std::size_t c = 0; c < walls && solved; ++c) {  // Gauss-Jordan
                        std::size_t p = c;
                        for (std::size_t r = c + 1; r < walls; ++r)
                            if (std::abs(M[r][c]) > std::abs(M[p][c])) p = r;
                        if (std::abs(M[p][c]) < 1e-6) { solved = false; break; }
                        for (int q = 0; q < 4; ++q) std::swap(M[c][q], M[p][q]);
                        for (std::size_t r = 0; r < walls; ++r) {
                            if (r == c) continue;
                            const double f = M[r][c] / M[c][c];
                            for (int q = 0; q < 4; ++q) M[r][q] -= f * M[c][q];
                        }
                    }
                    for (std::size_t a = 0; a < walls && solved; ++a) {
                        wall_area[a] = M[a][3] / M[a][a];
                        if (wall_area[a] < -1e-6) solved = false;
                        wall_area[a] = std::max(wall_area[a], 0.0);
                    }
                    if (solved) fit_stats_[8] += 1.0;
                }
                if (solved) {
                    // each wall's fitted weight, shared among its meets by their weights
                    std::array<double, 3> wall_met{};
                    for (std::size_t e = 0; e < entries; ++e) wall_met[entry_wall[e]] += entry_weight[e];
                    beta_sum = 0.0;
                    crossed_weight.fill(0.0);
                    for (std::size_t e = 0; e < entries; ++e) {
                        const std::size_t g = entry_wall[e];
                        const double w = plane * wall_area[g] * entry_weight[e] / wall_met[g];
                        const std::int32_t m = entry_material[e];
                        if (m < 0) {
                            beta_sum += w * edge_admittance;
                            continue;
                        }
                        beta_sum += w * beta[m];
                        std::size_t kind = 0;
                        while (kind < kinds && crossed[kind] != m) ++kind;
                        if (kind == kinds) crossed[kinds++] = m;
                        crossed_weight[kind] += w;
                    }
                } else if (met > 0.0) {
                    const double scale = fit / met;
                    beta_sum *= scale;
                    for (std::size_t kind = 0; kind < kinds; ++kind) crossed_weight[kind] *= scale;
                } else if (fit > 1e-9) {
                    Hit hit = bvh.intersect(centre(i), net * (-1.0 / area), 1.5 * cell_, -1);
                    if (hit.found()) {
                        const auto m = static_cast<std::int32_t>(face_material[hit.face]);
                        beta_sum += fit * beta[m];
                        std::size_t kind = 0;
                        while (kind < kinds && crossed[kind] != m) ++kind;
                        if (kind == kinds) crossed[kinds++] = m;
                        crossed_weight[kind] += fit;
                    }
                }
                if (fit_exact_ && !fit_sampled_.count(i)) {
                    // Round 2: the wall's exact area in the cell, per material.
                    beta_sum = 0.0;
                    kinds = 0;
                    crossed_weight.fill(0.0);
                    const auto wt = fit_wall_.find(i);
                    if (wt != fit_wall_.end()) {
                        for (const auto &[m, a] : wt->second) {
                            const double w = plane * a;
                            if (m < 0) {
                                beta_sum += w * edge_admittance;
                                continue;
                            }
                            beta_sum += w * beta[m];
                            std::size_t kind = 0;
                            while (kind < kinds && crossed[kind] != m) ++kind;
                            if (kind == kinds) crossed[kinds++] = m;
                            crossed_weight[kind] += w;
                        }
                    }
                }
                volume = fit_volume(i);
                fit_stats_[4] = std::min(fit_stats_[4], volume);
                fit_stats_[5] = std::max(fit_stats_[5], volume);
                double total = 0.0;
                for (int k = 0; k < link_directions(); ++k) total += weight[k];
                // 7-point: sum of the weights within 2 V bounds the largest eigenvalue by
                // 4 (proof: (p_i - p_j)^2 <= 2 p_i^2 + 2 p_j^2). 19-point: no such bound;
                // a cell is kept no stiffer for its volume than an interior one (3).
                const double limit = stencil_ == 7 ? 2.0 : 3.0;
                if (total > limit * volume) {
                    fit_stats_[2] += 1.0;
                    fit_stats_[3] = std::max(fit_stats_[3], total / (limit * volume));
                    volume = total / limit;
                }
            }
            // One set of branch velocities per material the cell meets, standing for its
            // cut links of that material.
            double g = beta_sum;
            const double r = std::sqrt(cell_scale(i));
            for (std::size_t kind = 0; kind < kinds; ++kind) {
                const auto &own = material_branches[static_cast<std::size_t>(crossed[kind])];
                for (const std::int32_t j : own) {
                    g += crossed_weight[kind] * (0.5 / branch_a_[static_cast<std::size_t>(j)]);
                    state_branch_.push_back(j);
                    state_weight_.push_back(static_cast<float>(r * crossed_weight[kind] / volume));
                }
            }
            add_special(i, weight.data(), g, volume);
        }
        boundary_cells_ = special_.size();
        fit_stats_[13] = static_cast<double>(special_.size());

        // The source's cells (made special, so one update adds the source): none of the
        // cells round it in the matched layer, whichever cells it then takes.
        for (const auto &[i, w] : visible_cells(bvh, source, &type_)) {
            if (type_[i] == kLayer) {
                throw std::invalid_argument("the source is in the matched layer");
            }
        }
        for (const auto &[i, w] : reading_cells(bvh, cut, source, true)) {
            std::size_t k = special_index(i);
            if (k == npos) k = add_special(i, exact_.data(), 0.0);
            source_[k] += static_cast<float>(cell_scale(i) * w / special_volume_[k]);
        }
        for (const Vec3 &p : probes) {
            // (kNear cells away to within rounding counts as that far.)
            const bool near = norm(p - source) < (kNear - 1e-9) * cell;
            auto cells = reading_cells(bvh, cut, p, !near);
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
        if ((flow_x == nullptr) != (flow_y == nullptr)) {
            throw std::invalid_argument("the air's motion needs both of its components");
        }
        if (flow_x != nullptr) set_up_flow(cut, flow_x, flow_y);
        threads_ = std::max<std::size_t>(1, std::min(threads_, air_ / 20000 + 1));
    }

    std::size_t air_cells() const { return air_; }
    std::size_t boundary_cells() const { return boundary_cells_; }
    std::size_t branch_states() const { return state_branch_.size(); }
    std::size_t layer_cells() const { return layer_cells_.size(); }
    bool open() const { return open_; }
    // Whether the air reaches each side of the grid (-x, +x, -y, +y, -z, +z): its edge
    // there, or the matched layer inside it.
    const std::array<bool, 6> &open_sides() const { return open_sides_; }
    bool flowing() const { return !flow_mask_.empty(); }
    std::size_t probes() const { return probe_cells_.size(); }
    bool probe_found(std::size_t p) const { return !probe_cells_.at(p).empty(); }
    std::size_t stencil() const { return stencil_; }
    // PROTOTYPE: fitted boundary's figures: candidate cells, sample rays, cells whose
    // volume was raised for stability, the largest raise, smallest and largest volume,
    // face area dropped (cells), set-up seconds.
    const std::array<double, 16> &fit_stats() const { return fit_stats_; }
    std::size_t threads() const { return threads_; }
    // PROTOTYPE: per special cell, its index, volume (as used), wall area and six open areas.
    std::vector<double> fit_cells() const {
        std::vector<double> out;
        for (std::size_t k = 0; k < special_.size(); ++k) {
            const std::size_t i = static_cast<std::size_t>(special_[k]);
            out.push_back(static_cast<double>(i));
            out.push_back(special_volume_[k]);
            double w = 0.0;
            const auto wt = fit_wall_.find(i);
            if (wt != fit_wall_.end()) for (const auto &e : wt->second) w += e.second;
            out.push_back(w);
            for (int d = 0; d < 6; ++d) {
                const std::size_t key = i * 18 + static_cast<std::size_t>(d);
                double a = -1.0;
                if (fit_done_.count(key)) {
                    const auto it = fit_area_.find(key);
                    a = it == fit_area_.end() ? 0.0 : it->second;
                }
                out.push_back(a);
            }
        }
        return out;
    }

    // Hands each array of the grid's layout to put(name, vector), for stepping the grid
    // elsewhere (on a GPU: engine/gpu.py); the grid's state is not among them.
    template <typename Put>
    void layout(Put &&put) const {
        put("shape", std::vector<std::size_t>{nx_, ny_, nz_});
        put("stencil", std::vector<std::size_t>{stencil_});
        put("weights", std::vector<float>(weight_.begin(), weight_.begin() + directions_));
        put("centre", std::vector<float>{centre_});
        put("total", std::vector<float>{total_});
        put("scaled", std::vector<std::size_t>{scale_.empty() ? 0u : 1u});
        put("scale", scale_);
        put("layer_scale", layer_scale_);
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
        put("flow_x", flow_x_);
        put("flow_y", flow_y_);
        put("flow_mask", flow_mask_);
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
            const float *scale = scale_.empty() ? nullptr : scale_.data();
            const float total = total_;
            float *a = u0, *b = u1;  // a: the previous step, then the next; b: the current
            float *wa = w0, *wb = w1;  // branch velocities, the same way
            const bool flowing = !flow_mask_.empty();
            for (std::size_t step = 0; step < steps; ++step) {
                if (t == 0) record(b, step, steps, out);
                if (flowing) {
                    // Both pressures carried along with the air (b's last pass waits for
                    // two barriers, by when thread 0 has recorded it).
                    carry(b, ib, ie, barrier);
                    carry(a, ib, ie, barrier);
                }
                for (std::size_t r = first_run; r < runs && run_begin[r] < ie; ++r) {
                    const std::size_t lo = std::max(run_begin[r], ib);
                    const std::size_t hi = std::min(run_end[r], ie);
                    if (scale != nullptr && wide) {
                        update_interior(a, b, scale, lo, hi, sy, sz, axial, diagonal, total);
                    } else if (scale != nullptr) {
                        update_interior(a, b, scale, lo, hi, sy, sz, axial, total);
                    } else if (wide) {
                        update_interior(a, b, lo, hi, sy, sz, axial, diagonal, centre);
                    } else {
                        update_interior(a, b, lo, hi, sy, sz, axial, centre);
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
        std::fill(flow_t1_.begin(), flow_t1_.end(), 0.0f);
        std::fill(flow_t2_.begin(), flow_t2_.end(), 0.0f);
        std::fill(u0_.begin(), u0_.end(), 0.0f);
        std::fill(u1_.begin(), u1_.end(), 0.0f);
        std::fill(w0_.begin(), w0_.end(), 0.0f);
        std::fill(w1_.begin(), w1_.end(), 0.0f);
        std::fill(layer_state_.begin(), layer_state_.end(), 0.0f);
    }

private:
    static constexpr std::uint8_t kSolid = 0, kInterior = 1, kSpecial = 2, kLayer = 3;
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
    static constexpr double kMargin = 1.0 / 262144.0;  // 2^-18 (see above)
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

    // The stencil's weights (exact, and lowered for the update), boundary weights,
    // offsets and the bits of the + half's directions in the cut mask.
    void set_up_stencil() {
        directions_ = stencil_ == 7 ? 6 : 18;
        const double axial = stencil_ == 7 ? 1.0 / 3.0 : 0.25;
        const double diagonal = stencil_ == 7 ? 0.0 : 0.125;
        const double courant = std::sqrt(axial + 4.0 * diagonal);  // sum g_k d_x^2 = 2 L^2
        int bits = 0;
        for (int k = 0; k < 18; ++k) {
            exact_[k] = k < directions_ ? (k < 6 ? axial : diagonal) : 0.0;
            weight_[k] = lowered(exact_[k]);
            boundary_[k] = exact_[k] / (2.0 * courant);
            offset_[k] = kSteps[k][0] + kSteps[k][1] * static_cast<std::ptrdiff_t>(sy_) +
                         kSteps[k][2] * static_cast<std::ptrdiff_t>(sz_);
            const int *d = kSteps[k];
            const bool plus = d[2] > 0 || (d[2] == 0 && (d[1] > 0 || (d[1] == 0 && d[0] > 0)));
            bit_[k] = k < directions_ && plus ? bits++ : -1;
        }
        // Both exact in single precision, so that the centre and the links sum to 2.
        double total = 0.0;
        for (int k = 0; k < directions_; ++k) total += weight_[k];
        centre_ = static_cast<float>(2.0 - total);
        total_ = static_cast<float>(total);
    }

    // Moving air: each cell's shifts along x and y (cells per step; none in the matched
    // layer or out of the air), and which of its neighbours along them the gradient may
    // take: bits 0 to 3 for x (the one below, above, two below, two above), 4 to 7 for y.
    void set_up_flow(const std::vector<std::uint16_t> &cut, const float *fx, const float *fy) {
        const std::size_t n = nx_ * ny_ * nz_;
        flow_x_.assign(n, 0.0f);
        flow_y_.assign(n, 0.0f);
        flow_mask_.assign(n, 0);
        std::vector<std::uint8_t> near(n, 0);  // bits 0 to 3: the neighbour along -x, +x, -y, +y
        auto moving = [&](std::size_t i) { return type_[i] == kInterior || type_[i] == kSpecial; };
        for (std::size_t i = 0; i < n; ++i) {
            if (!moving(i)) continue;
            if (!(std::isfinite(fx[i]) && std::isfinite(fy[i]) && std::abs(fx[i]) < 0.5f &&
                  std::abs(fy[i]) < 0.5f)) {
                throw std::invalid_argument("the air must move less than half a cell per step");
            }
            flow_x_[i] = fx[i];
            flow_y_[i] = fy[i];
            for (int k = 0; k < 4; ++k) {
                const std::size_t j = neighbour(i, k);
                if (!blocked(cut, i, k) && type_[j] != kSolid) near[i] |= std::uint8_t(1u << k);
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (!moving(i)) continue;
            unsigned m = 0;
            for (int axis = 0; axis < 2; ++axis) {
                const int lo = 2 * axis, hi = 2 * axis + 1;
                const unsigned shift = 4u * static_cast<unsigned>(axis);
                const bool below = near[i] & (1u << lo), above = near[i] & (1u << hi);
                if (below) m |= 1u << shift;
                if (above) m |= 2u << shift;
                if (below && (near[neighbour(i, lo)] & (1u << lo))) m |= 4u << shift;
                if (above && (near[neighbour(i, hi)] & (1u << hi))) m |= 8u << shift;
            }
            flow_mask_[i] = static_cast<std::uint8_t>(m);
        }
        flow_t1_.assign(n, 0.0f);
        flow_t2_.assign(n, 0.0f);
    }

    // q's change along an axis (stride s) at cell i, over a cell, from its neighbours
    // m allows (bits as in set_up_flow, shifted down to the axis): fourth-order central
    // differences, second order, or next to a face the upwind difference for a shift
    // into it and none for one out of it.
    static float gradient(const float *q, std::size_t i, std::size_t s, unsigned m, float shift) {
        const bool below = m & 1u, above = m & 2u;
        if (below && above) {
            if ((m & 12u) == 12u) {
                return (8.0f * (q[i + s] - q[i - s]) - (q[i + 2 * s] - q[i - 2 * s])) *
                       (1.0f / 12.0f);
            }
            return 0.5f * (q[i + s] - q[i - s]);
        }
        if (shift > 0.0f) return below ? q[i] - q[i - s] : 0.0f;
        return above ? q[i + s] - q[i] : 0.0f;
    }

    // -v . grad q at cell i, over a step.
    float carried(const float *q, std::size_t i) const {
        const float sx = flow_x_[i], sy = flow_y_[i];
        const unsigned m = flow_mask_[i];
        return -(sx * gradient(q, i, 1, m & 15u, sx) + sy * gradient(q, i, sy_, m >> 4, sy));
    }

    // Carries q (cells ib to ie on this thread) along with the air for a step: the
    // strong-stability-preserving Runge-Kutta method of third order, through the two
    // stages' fields; cells the air does not move keep their values, to the bit.
    void carry(float *q, std::size_t ib, std::size_t ie, StepBarrier &barrier) {
        float *t1 = flow_t1_.data(), *t2 = flow_t2_.data();
        const std::uint8_t *type = type_.data();
        const float *fx = flow_x_.data(), *fy = flow_y_.data();
        auto still = [&](std::size_t i) { return fx[i] == 0.0f && fy[i] == 0.0f; };
        for (std::size_t i = ib; i < ie; ++i) {
            if (type[i] == kSolid) continue;
            t1[i] = still(i) ? q[i] : q[i] + carried(q, i);
        }
        barrier.wait();
        for (std::size_t i = ib; i < ie; ++i) {
            if (type[i] == kSolid) continue;
            t2[i] = still(i) ? q[i] : 0.75f * q[i] + 0.25f * (t1[i] + carried(t1, i));
        }
        barrier.wait();
        for (std::size_t i = ib; i < ie; ++i) {
            if (type[i] == kSolid || still(i)) continue;
            q[i] = (1.0f / 3.0f) * q[i] + (2.0f / 3.0f) * (t2[i] + carried(t2, i));
        }
        barrier.wait();
    }

    // Cell i's s: the square of its speed of sound over the grid's (1 without a field).
    double cell_scale(std::size_t i) const {
        return scale_.empty() ? 1.0 : static_cast<double>(scale_[i]);
    }

    // The largest float not above w.
    static float down(double w) {
        float f = static_cast<float>(w);
        if (static_cast<double>(f) > w) f = std::nextafter(f, 0.0f);
        return f;
    }

    // A link's weight w as the update takes it: lowered by the stability margin (see
    // above), rounded down.
    static float lowered(double w) { return down(w * (1.0 - kMargin)); }

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

    // Marks as reached by the air (open_sides) the sides of the grid that cell i lies
    // within `within` cells of.
    void reach_sides(std::size_t i, std::size_t within) {
        const std::size_t at[3] = {i % nx_, (i / nx_) % ny_, i / sz_};
        const std::size_t size[3] = {nx_, ny_, nz_};
        for (std::size_t a = 0; a < 3; ++a) {
            if (at[a] <= within) open_sides_[2 * a] = true;
            if (at[a] + 1 + within >= size[a]) open_sides_[2 * a + 1] = true;
        }
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
        const double courant = std::sqrt(static_cast<double>(weight_[0]));  // as the interior's
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
            if (layer_scale_.empty()) {
                part = layer_keep_[2 * axis][x] * part - layer_push_[2 * axis][x] * (above - below);
            } else {
                part = layer_keep_[2 * axis][x] * part -
                       (layer_push_[2 * axis][x] * layer_scale_[c]) * (above - below);
            }
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

    // A reading takes this many cells a side where it can (see reading_cells), and a
    // probe nearer the source than kNear cells the eight round it.
    static constexpr int kReach = 4;
    static constexpr double kNear = 3.0;

    // The cells that stand for point p, with weights that sum to one: the source spreads
    // over them, and a probe reads them. With `cubic`, the kReach cells a side round p
    // (two to either side of it along each axis), weighted along each axis by Lagrange's
    // cubic through their centres. Halfway between two centres they read a wave of eight
    // cells per wavelength along the axis 0.85 % low, where the two cells round p,
    // weighted linearly, read it 7.6 % low (9/8 cos(kh/2) - 1/8 cos(3kh/2) against
    // cos(kh/2)), and the level of a point source's sound twelve cells from it changes
    // with the places of the source and of the probe within their cells by 0.02 dB on
    // the 19-point grid (docs/validation.md, Free field).
    //
    // A cell is p's when it is air outside the matched layer and p sees its centre. A
    // layer of that box none of whose cells is p's lies behind a face along the grid (a
    // floor, a wall, the layer's start), and the box goes without it: along that axis
    // the reading is then of the two cells round p, weighted linearly, or of the one on
    // p's side, and along the other two still the cubic. So a point low over flat ground
    // keeps the cubic along the ground. All the cells left must be p's, with no face
    // between neighbours; next to a face at an angle to the grid, an edge or an opening
    // they are not, and the reading is of the cells round p that it sees, weighted
    // trilinearly (visible_cells), as without `cubic`. The reading never reaches past
    // the box round p: rows of four cells pushed to p's side of a surface were tried and
    // are not taken, as they reach a cell further to that side, and where the field
    // there is not smooth (the source's own cells, the edges of an opening) the level
    // stepped by up to 0.7 dB from one probe to the next.
    //
    // The source takes the cubic. Within a few cells of it its sound is not that of a
    // point, and a probe nearer it than kNear cells reads without `cubic`. The table of
    // docs/validation.md (Free field) gives the level and the gradient by the distance
    // from the source, and with kNear changed they come out further off at the places
    // that decide it: with every probe taking the cubic (kNear 0), the 19-point grid's
    // level two cells from the source is up to 2.1 dB off where it is 0.6 dB, and the
    // 7-point grid's gradient three cells from it 6.8 dB where it is 1.7 dB; with the
    // cubic only from four cells on (kNear 4), the 19-point grid's level in the 125 Hz
    // band three cells from the source is 0.28 dB off where it is 0.05 dB. (The 7-point
    // grid's level 2.5 cells from the source is closer with the cubic, 0.35 dB off
    // against 0.83 dB.)
    std::vector<std::pair<std::size_t, double>> reading_cells(
        const Bvh &bvh, const std::vector<std::uint16_t> &cut, const Vec3 &p, bool cubic) const {
        if (!cubic) return visible_cells(bvh, p, &type_);
        const double q[3] = {(p.x - origin_.x) / cell_ - 0.5, (p.y - origin_.y) / cell_ - 0.5,
                             (p.z - origin_.z) / cell_ - 0.5};
        const std::size_t size[3] = {nx_, ny_, nz_};
        long long start[3];
        double weight[3][kReach];
        for (int a = 0; a < 3; ++a) {
            // The box lies in the grid, and the two cells round p inside its edge cells
            // (as visible_cells has them).
            if (!(q[a] >= 1.0 && q[a] < static_cast<double>(size[a]) - 3.0)) {
                return visible_cells(bvh, p, &type_);
            }
            start[a] = static_cast<long long>(std::floor(q[a])) - (kReach / 2 - 1);
            const double x = q[a] - static_cast<double>(start[a]);
            for (int j = 0; j < kReach; ++j) {
                double w = 1.0;
                for (int m = 0; m < kReach; ++m) {
                    if (m != j) w *= (x - m) / static_cast<double>(j - m);
                }
                weight[a][j] = w;
            }
        }
        // Which of the box's cells are p's.
        auto index = [&](int x, int y, int z) {
            return static_cast<std::size_t>(start[0] + x) +
                   static_cast<std::size_t>(start[1] + y) * sy_ +
                   static_cast<std::size_t>(start[2] + z) * sz_;
        };
        bool own[kReach][kReach][kReach];
        for (int z = 0; z < kReach; ++z) {
            for (int y = 0; y < kReach; ++y) {
                for (int x = 0; x < kReach; ++x) {
                    const std::size_t i = index(x, y, z);
                    bool seen = type_[i] == kInterior || type_[i] == kSpecial;
                    if (seen) {
                        const Vec3 to = centre(i) - p;
                        const double distance = norm(to);
                        seen = !(distance > 0.0 &&
                                 bvh.intersect(p, to / distance, distance, -1).found());
                    }
                    own[z][y][x] = seen;
                }
            }
        }
        // The box goes without a layer none of whose cells is p's (found again until no
        // layer goes, as a layer counts over the layers the other axes keep).
        bool keep[3][kReach];
        for (auto &axis : keep) std::fill(std::begin(axis), std::end(axis), true);
        for (bool changed = true; changed;) {
            changed = false;
            for (int a = 0; a < 3; ++a) {
                const int b = (a + 1) % 3, c = (a + 2) % 3;
                for (int j = 0; j < kReach; ++j) {
                    if (!keep[a][j]) continue;
                    bool any = false;
                    int at[3];
                    at[a] = j;
                    for (at[b] = 0; at[b] < kReach && !any; ++at[b]) {
                        for (at[c] = 0; at[c] < kReach && !any; ++at[c]) {
                            any = keep[b][at[b]] && keep[c][at[c]] && own[at[2]][at[1]][at[0]];
                        }
                    }
                    if (!any) {
                        keep[a][j] = false;
                        changed = true;
                    }
                }
            }
        }
        // Along each axis: all four cells and the cubic; without an outer one, the two
        // round p, weighted linearly; without one of those two, the other.
        int from[3], count[3];
        for (int a = 0; a < 3; ++a) {
            const int inner = kReach / 2 - 1;
            if (keep[a][0] && keep[a][1] && keep[a][2] && keep[a][3]) {
                from[a] = 0;
                count[a] = kReach;
            } else if (keep[a][inner] && keep[a][inner + 1]) {
                from[a] = inner;
                count[a] = 2;
                const double t = q[a] - std::floor(q[a]);
                weight[a][inner] = 1.0 - t;
                weight[a][inner + 1] = t;
            } else if (keep[a][inner] || keep[a][inner + 1]) {
                from[a] = keep[a][inner] ? inner : inner + 1;
                count[a] = 1;
                weight[a][from[a]] = 1.0;
            } else {
                return visible_cells(bvh, p, &type_);
            }
        }
        // Every cell left must be p's, with no face between neighbours: next to a face
        // at an angle to the grid, an edge or an opening they are not.
        std::vector<std::pair<std::size_t, double>> out;
        out.reserve(static_cast<std::size_t>(count[0] * count[1] * count[2]));
        for (int z = from[2]; z < from[2] + count[2]; ++z) {
            for (int y = from[1]; y < from[1] + count[1]; ++y) {
                for (int x = from[0]; x < from[0] + count[0]; ++x) {
                    const std::size_t i = index(x, y, z);
                    if (!own[z][y][x] || (x + 1 < from[0] + count[0] && is_cut(cut, i, 1)) ||
                        (y + 1 < from[1] + count[1] && is_cut(cut, i, 3)) ||
                        (z + 1 < from[2] + count[2] && is_cut(cut, i, 5))) {
                        return visible_cells(bvh, p, &type_);
                    }
                    const double w = weight[0][x] * weight[1][y] * weight[2][z];
                    if (w != 0.0) out.emplace_back(i, w);
                }
            }
        }
        return out;
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
    // With a speed field its links weigh s times as much, its boundary terms g (and its
    // branch velocities' weights, set by the caller) r times.
    // (PROTOTYPE: `volume`, the cell's share of air, divides its links and boundary terms;
    // the caller divides its branch velocities' weights.)
    std::size_t add_special(std::size_t i, const double *w, double g, double volume = 1.0) {
        type_[i] = kSpecial;
        special_.push_back(static_cast<std::int64_t>(i));
        special_volume_.push_back(volume);
        const double s = cell_scale(i) / volume, r = std::sqrt(cell_scale(i)) / volume;
        double total = 0.0, taken = 0.0;
        for (int k = 0; k < link_directions(); ++k) {
            if (!(w[k] > 0.0)) continue;
            total += w[k];
            link_direction_.push_back(static_cast<std::uint8_t>(k));
            link_weight_.push_back(lowered(s * w[k]));
            taken += static_cast<double>(link_weight_.back());
        }
        link_start_.push_back(link_direction_.size());
        // (PROTOTYPE: with a volume, the centre is 2 less the links as taken, so that the
        // uniform pressure stays; without, as before, to the bit.)
        coef_a_.push_back(static_cast<float>(volume != 1.0 ? 2.0 - taken : 2.0 - s * total * (1.0 - kMargin)));
        coef_b_.push_back(static_cast<float>(1.0 - r * g));
        coef_inv_.push_back(static_cast<float>(1.0 / (1.0 + r * g)));
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

    // ---- PROTOTYPE: fitted (cut-cell) boundary ------------------------------------------
    // Air for the fit: the matched layer's cells too (no face reaches them).
    bool fit_air(std::size_t i) const { return type_[i] == kInterior || type_[i] == kLayer; }

    // Whether segment a-b crosses no face.
    bool sees(const Bvh &bvh, const Vec3 &a, const Vec3 &b) const {
        const Vec3 d = b - a;
        const double l = norm(d);
        if (l < 1e-12) return true;
        ++fit_rays_;
        return !bvh.intersect(a, d / l, l, -1).found();
    }

    // The air cell point x (in cell c's cube) belongs to: c when c is air and its centre
    // sees x; otherwise the nearest of c's axial neighbours, then of its edge neighbours,
    // that is air and sees x; npos when none.
    std::size_t owner(const Bvh &bvh, const Vec3 &x, std::size_t c) const {
        if (fit_air(c) && sees(bvh, centre(c), x)) return c;
        if (edge(c)) return npos;
        for (int pass = 0; pass < 2; ++pass) {
            std::size_t best = npos;
            double best_d = 1e300;
            for (int k = pass == 0 ? 0 : 6; k < (pass == 0 ? 6 : 18); ++k) {
                const std::size_t j = neighbour(c, k);
                if (!fit_air(j)) continue;
                const double d = norm(centre(j) - x);
                if (d < best_d && sees(bvh, centre(j), x)) {
                    best = j;
                    best_d = d;
                }
            }
            if (best != npos) return best;
        }
        return npos;
    }

    int direction_to(std::size_t a, std::size_t b) const {
        const std::ptrdiff_t d = static_cast<std::ptrdiff_t>(b) - static_cast<std::ptrdiff_t>(a);
        for (int k = 0; k < 18; ++k) {
            if (offset_[k] == d) return k;
        }
        return -1;
    }

    // Samples, for every cell a face passes near (and the cells round those), the share
    // of its cube each air cell owns (slivers of cells whose centre lies behind a face go
    // to a neighbour, see owner), the open share of each face between two cubes, by the
    // owners of the air on its two sides, and on the 19-point stencil each edge link's
    // open share: the share of the unit square across its middle that both its cells
    // see. Returns the candidate cells (air ones become boundary cells).
    std::vector<std::uint8_t> fit_boundary(const Bvh &bvh, const std::vector<std::uint8_t> &near,
                                           const std::vector<std::uint16_t> &cut) {
        const std::size_t n = nx_ * ny_ * nz_;
        const int S = static_cast<int>(fitted_);
        const double unit_volume = 1.0 / (static_cast<double>(S) * S * S);
        const double unit_area = 1.0 / (static_cast<double>(S) * S);
        std::vector<std::uint8_t> candidate(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            if (!near[i] || edge(i)) continue;
            candidate[i] = 1;
            for (int k = 0; k < 18; ++k) {
                const std::size_t j = neighbour(i, k);
                if (!edge(j) && !candidate[j]) candidate[j] = 2;
            }
        }
        // A near cell whose centre sees its cube's 26 corners, edge middles and face
        // middles is taken as clear (PROTOTYPE shortcut: a face poking into the cube
        // between those points is missed; production would clip the faces to the cube).
        std::vector<std::uint8_t> clear(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            if (!near[i] || edge(i) || !fit_air(i)) continue;
            bool all = true;
            for (int a = -1; a <= 1 && all; ++a) {
                for (int b = -1; b <= 1 && all; ++b) {
                    for (int d = -1; d <= 1 && all; ++d) {
                        if (a == 0 && b == 0 && d == 0) continue;
                        all = sees(bvh, centre(i), centre(i) + Vec3{0.5 * a, 0.5 * b, 0.5 * d} * cell_);
                    }
                }
            }
            clear[i] = all;
        }
        auto is_near = [&](std::size_t i) { return near[i] && !clear[i]; };
        fit_stats_[4] = 1e300;
        auto add_area = [&](std::size_t a, int k, double v) {
            fit_area_[a * 18 + static_cast<std::size_t>(k)] += v;
        };
        for (std::size_t c = 0; c < n; ++c) {
            if (!candidate[c]) continue;
            fit_stats_[0] += 1.0;
            const Vec3 o = centre(c);
            // the cube's air
            if (!is_near(c)) {
                if (fit_air(c)) fit_volume_[c] += 1.0;
            } else {
                for (int a = 0; a < S; ++a) {
                    for (int b = 0; b < S; ++b) {
                        for (int d = 0; d < S; ++d) {
                            const Vec3 x = o + Vec3{((a + 0.5) / S - 0.5) * cell_,
                                                    ((b + 0.5) / S - 0.5) * cell_,
                                                    ((d + 0.5) / S - 0.5) * cell_};
                            const std::size_t w = owner(bvh, x, c);
                            if (w != npos) fit_volume_[w] += unit_volume;
                        }
                    }
                }
            }
            // its faces (each once)
            for (int k = 0; k < 6; ++k) {
                const std::size_t j = neighbour(c, k);
                if (edge(j) || (candidate[j] && j < c)) continue;
                if (fit_air(c)) fit_done_[c * 18 + static_cast<std::size_t>(k)] = 1;
                if (fit_air(j)) fit_done_[j * 18 + static_cast<std::size_t>(opposite(k))] = 1;
                if (!is_near(c) && !is_near(j)) {
                    if (fit_air(c) && fit_air(j)) {
                        add_area(c, k, 1.0);
                        add_area(j, opposite(k), 1.0);
                    }
                    continue;
                }
                const int axis = k / 2;
                const Vec3 e = unit(k);
                const Vec3 mid = o + e * (0.5 * cell_);
                for (int a = 0; a < S; ++a) {
                    for (int b = 0; b < S; ++b) {
                        double u[3] = {0.0, 0.0, 0.0};
                        u[(axis + 1) % 3] = ((a + 0.5) / S - 0.5) * cell_;
                        u[(axis + 2) % 3] = ((b + 0.5) / S - 0.5) * cell_;
                        const Vec3 y = mid + Vec3{u[0], u[1], u[2]};
                        const std::size_t o1 = owner(bvh, y - e * (1e-6 * cell_), c);
                        if (o1 == npos) continue;
                        const std::size_t o2 = owner(bvh, y + e * (1e-6 * cell_), j);
                        if (o2 == npos || o2 == o1) continue;
                        const int kk = direction_to(o1, o2);
                        if (kk < 0 || kk >= 6) {
                            fit_stats_[6] += unit_area;
                            continue;
                        }
                        add_area(o1, kk, unit_area);
                        add_area(o2, opposite(kk), unit_area);
                        fit_done_[o1 * 18 + static_cast<std::size_t>(kk)] = 1;
                        fit_done_[o2 * 18 + static_cast<std::size_t>(opposite(kk))] = 1;
                    }
                }
            }
            // the 19-point stencil's edge links (each once)
            if (stencil_ != 19 || !fit_air(c)) continue;
            for (int k = 6; k < 18; ++k) {
                const std::size_t j = neighbour(c, k);
                if (edge(j) || !fit_air(j) || (candidate[j] && j < c)) continue;
                if (blocked(cut, c, k)) continue;
                double share = 1.0;
                if (is_near(c) || is_near(j)) {
                    const auto [ka, kb] = steps_of(k);
                    const Vec3 ua = unit(ka), ub = unit(kb);
                    const Vec3 across = (ua - ub) * (1.0 / std::sqrt(2.0));
                    Vec3 third{1.0, 1.0, 1.0};
                    third = third - Vec3{std::abs(ua.x) + std::abs(ub.x), std::abs(ua.y) + std::abs(ub.y),
                                         std::abs(ua.z) + std::abs(ub.z)};
                    const Vec3 mid = (centre(c) + centre(j)) * 0.5;
                    int seen = 0;
                    for (int a = 0; a < S; ++a) {
                        for (int b = 0; b < S; ++b) {
                            const Vec3 y = mid + across * (((a + 0.5) / S - 0.5) * cell_) +
                                           third * (((b + 0.5) / S - 0.5) * cell_);
                            if (sees(bvh, centre(c), y) && sees(bvh, centre(j), y)) ++seen;
                        }
                    }
                    share = seen * unit_area;
                }
                fit_area_[c * 18 + static_cast<std::size_t>(k)] = share;
                fit_area_[j * 18 + static_cast<std::size_t>(opposite(k))] = share;
                fit_done_[c * 18 + static_cast<std::size_t>(k)] = 1;
                fit_done_[j * 18 + static_cast<std::size_t>(opposite(k))] = 1;
            }
        }
        // air cells that only received slivers keep their own cube
        for (auto &[c, v] : fit_volume_) {
            if (!candidate[c]) v += 1.0;
        }
        fit_stats_[1] = static_cast<double>(fit_rays_);
        return candidate;
    }

    // ---- PROTOTYPE (round 2): exact clipping -------------------------------------------
    // Cell-local coordinates: u = (x - centre) / cell, the cube [-1/2, 1/2]^3.
    struct FitPolygon {
        std::vector<Vec3> v;
        int tag;  // 0-5: the cube's face along direction k; 6 + p: plane p
    };
    using FitPolyhedron = std::vector<FitPolygon>;
    static constexpr double kFitEps = 1e-11;   // on a plane, in cells
    static constexpr double kFitTiny = 1e-13;  // an area or a volume of nothing, in cells

    static Vec3 poly_vector(const std::vector<Vec3> &v) {
        Vec3 s{0.0, 0.0, 0.0};
        for (std::size_t i = 1; i + 1 < v.size(); ++i) s = s + cross(v[i] - v[0], v[i + 1] - v[0]);
        return s * 0.5;
    }
    static double poly_area(const std::vector<Vec3> &v) { return v.size() < 3 ? 0.0 : norm(poly_vector(v)); }

    // Polygon v clipped to n.x >= d (points within kFitEps of the plane kept, and listed in on).
    static std::vector<Vec3> clip_poly(const std::vector<Vec3> &v, const Vec3 &n, double d,
                                       std::vector<Vec3> *on) {
        std::vector<Vec3> out;
        const std::size_t m = v.size();
        for (std::size_t i = 0; i < m; ++i) {
            const Vec3 &a = v[i], &b = v[(i + 1) % m];
            const double sa = dot(n, a) - d, sb = dot(n, b) - d;
            if (sa >= -kFitEps) {
                out.push_back(a);
                if (on && std::abs(sa) <= kFitEps) on->push_back(a);
            }
            if ((sa > kFitEps && sb < -kFitEps) || (sa < -kFitEps && sb > kFitEps)) {
                const Vec3 p = a + (b - a) * (sa / (sa - sb));
                out.push_back(p);
                if (on) on->push_back(p);
            }
        }
        return out;
    }

    // Convex polyhedron P clipped to n.x >= d, capped on the plane by a polygon tagged tag.
    static FitPolyhedron clip_polyhedron(const FitPolyhedron &P, const Vec3 &n, double d, int tag) {
        FitPolyhedron out;
        std::vector<Vec3> on;
        for (const auto &f : P) {
            auto c = clip_poly(f.v, n, d, &on);
            if (c.size() >= 3 && poly_area(c) > kFitTiny) out.push_back({std::move(c), f.tag});
        }
        if (out.empty()) return out;
        std::vector<Vec3> cap;
        for (const Vec3 &p : on) {
            bool dup = false;
            for (const Vec3 &q : cap) dup = dup || norm(p - q) < 1e-9;
            if (!dup) cap.push_back(p);
        }
        if (cap.size() >= 3) {
            Vec3 m{0.0, 0.0, 0.0};
            for (const Vec3 &p : cap) m = m + p;
            m = m / static_cast<double>(cap.size());
            const Vec3 t = std::abs(n.x) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
            const Vec3 e1 = normalized(cross(n, t)), e2 = cross(n, e1);
            std::sort(cap.begin(), cap.end(), [&](const Vec3 &a, const Vec3 &b) {
                return std::atan2(dot(a - m, e2), dot(a - m, e1)) < std::atan2(dot(b - m, e2), dot(b - m, e1));
            });
            if (poly_area(cap) > kFitTiny) out.push_back({std::move(cap), tag});
        }
        return out;
    }

    static Vec3 vertex_mean(const FitPolyhedron &P) {
        Vec3 m{0.0, 0.0, 0.0};
        std::size_t c = 0;
        for (const auto &f : P) {
            for (const Vec3 &p : f.v) {
                m = m + p;
                ++c;
            }
        }
        return c ? m / static_cast<double>(c) : m;
    }

    static double poly_volume(const FitPolyhedron &P) {
        const Vec3 m = vertex_mean(P);
        double v = 0.0;
        for (const auto &f : P) {
            const Vec3 a = poly_vector(f.v);
            v += std::abs(dot(a, f.v[0] - m)) / 3.0;
        }
        return v;
    }

    static FitPolyhedron unit_cube() {
        FitPolyhedron P;
        for (int k = 0; k < 6; ++k) {
            const int a = k / 2, b = (a + 1) % 3, c = (a + 2) % 3;
            const double s = (k % 2) ? 0.5 : -0.5;
            std::vector<Vec3> v;
            const double corner[4][2] = {{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}};
            for (const auto &q : corner) {
                double u[3];
                u[a] = s;
                u[b] = q[0];
                u[c] = q[1];
                v.push_back({u[0], u[1], u[2]});
            }
            P.push_back({v, k});
        }
        return P;
    }

    // Area of the part of triangle t (coplanar with convex polygon F, unit normal n) inside F.
    static double covered(const std::vector<Vec3> &F, const Vec3 &n, std::vector<Vec3> t) {
        Vec3 m{0.0, 0.0, 0.0};
        for (const Vec3 &p : F) m = m + p;
        m = m / static_cast<double>(F.size());
        for (std::size_t i = 0; i < F.size() && t.size() >= 3; ++i) {
            const Vec3 &a = F[i], &b = F[(i + 1) % F.size()];
            Vec3 q = cross(n, b - a);
            if (dot(q, m - a) < 0.0) q = q * -1.0;
            t = clip_poly(t, q, dot(q, a), nullptr);
        }
        return poly_area(t);
    }

    struct FitTri {
        std::array<Vec3, 3> p;  // global
        std::int32_t material;
    };
    struct FitPlane {
        Vec3 n;
        double d;  // local: n.u = d
        std::vector<std::size_t> tris;  // into the cube's triangles
    };
    struct FitCube {
        bool exact = true;
        std::vector<FitPlane> planes;
        std::vector<FitTri> tris;
        std::array<std::vector<std::size_t>, 6> membrane;  // triangles on each face of the cube
        std::vector<std::uint32_t> sign;  // per piece: bit p set on plane p's + side
        std::vector<std::int64_t> region;  // per piece: global region
    };
    struct FitRegion {
        std::size_t cube;
        double volume = 0.0;
        std::int64_t owner = -1;  // a cell; -1 none; -2 the grid's edge
        bool fixed = false;
        std::vector<std::pair<std::int32_t, double>> wall;  // material (-1: grid's edge), area
    };

    static void add_wall(std::vector<std::pair<std::int32_t, double>> &w, std::int32_t m, double a) {
        if (!(a > 0.0)) return;
        for (auto &e : w) {
            if (e.first == m) {
                e.second += a;
                return;
            }
        }
        w.push_back({m, a});
    }

    Vec3 to_local(std::size_t c, const Vec3 &x) const { return (x - centre(c)) / cell_; }

    // The region of cube c holding local point u (-1 when no piece has its signs).
    std::int64_t region_at(const FitCube &C, const Vec3 &u) const {
        std::uint32_t s = 0;
        for (std::size_t p = 0; p < C.planes.size(); ++p) {
            if (dot(C.planes[p].n, u) - C.planes[p].d >= 0.0) s |= 1u << p;
        }
        for (std::size_t q = 0; q < C.sign.size(); ++q) {
            if (C.sign[q] == s) return C.region[q];
        }
        return -1;
    }

    // Exact fit (see NOTES of round 2): volumes, open face areas and wall areas per material
    // of the cells round every cube a face reaches into. Returns the cells it touched.
    std::vector<std::size_t> fit_boundary_exact(const Bvh &bvh, const std::int64_t *face_material) {
        const std::size_t n = nx_ * ny_ * nz_;
        const int S = static_cast<int>(fitted_);
        static constexpr double jitter[3] = {0.41421356237309515, 0.7320508075688772, 0.2360679774997898};
        const char *env_max = std::getenv("ACOUSTIC_SPACES_FIT_MAXPLANES");
        const char *env_diag = std::getenv("ACOUSTIC_SPACES_FIT_DIAG");
        diag_mode_ = env_diag ? std::atoi(env_diag) : 2;
        if (stencil_ != 7) diag_mode_ = 0;
        fit_stats_[8] = 0.0;
        const std::size_t max_planes = env_max ? std::strtoul(env_max, nullptr, 10) : 8;
        fit_stats_[4] = 1e300;
        // 1. The faces in each cube (by plane distance and clipping).
        std::unordered_map<std::size_t, FitCube> cubes;
        const double low[3] = {origin_.x, origin_.y, origin_.z};
        const std::size_t size[3] = {nx_, ny_, nz_};
        for (std::size_t f = 0; f < bvh.face_count(); ++f) {
            const std::array<Vec3, 3> p = bvh.corners(f);
            const Vec3 nrm = bvh.normal(f);
            std::size_t from[3], to[3];
            bool any = true;
            for (int a = 0; a < 3; ++a) {
                const double lo = std::min({p[0][a], p[1][a], p[2][a]});
                const double hi = std::max({p[0][a], p[1][a], p[2][a]});
                const double first = std::floor((lo - low[a]) / cell_ - 1e-9);
                const double last = std::floor((hi - low[a]) / cell_ + 1e-9);
                if (last < 0.0 || first > static_cast<double>(size[a] - 1)) { any = false; break; }
                from[a] = static_cast<std::size_t>(std::max(first, 0.0));
                to[a] = std::min(static_cast<std::size_t>(last), size[a] - 1);
            }
            if (!any) continue;
            for (std::size_t z = from[2]; z <= to[2]; ++z)
                for (std::size_t y = from[1]; y <= to[1]; ++y)
                    for (std::size_t x = from[0]; x <= to[0]; ++x) {
                        const std::size_t i = x + y * sy_ + z * sz_;
                        const Vec3 c = centre(i);
                        if (std::abs(dot(nrm, c - p[0])) > 0.8661 * cell_) continue;
                        std::vector<Vec3> t = {to_local(i, p[0]), to_local(i, p[1]), to_local(i, p[2])};
                        for (int k = 0; k < 6 && t.size() >= 3; ++k) {
                            Vec3 e = unit(k);
                            t = clip_poly(t, e * -1.0, -0.5, nullptr);  // -e.u >= -1/2
                        }
                        if (t.size() < 3 || poly_area(t) <= kFitTiny) continue;
                        if (edge(i)) continue;
                        FitCube &C = cubes[i];
                        const std::size_t ti = C.tris.size();
                        C.tris.push_back({p, static_cast<std::int32_t>(face_material[f])});
                        // On a face of the cube?
                        int on_face = -1;
                        const double d0 = dot(nrm, to_local(i, p[0]));
                        for (int k = 0; k < 6; ++k) {
                            const Vec3 e = unit(k);
                            if (std::abs(std::abs(dot(nrm, e)) - 1.0) < 1e-12 &&
                                std::abs(d0 - dot(nrm, e) * 0.5) < kFitEps) on_face = k;
                        }
                        if (on_face >= 0) {
                            C.membrane[static_cast<std::size_t>(on_face)].push_back(ti);
                            continue;
                        }
                        Vec3 pn = nrm;
                        double pd = d0;
                        const double lead = std::abs(pn.x) > 1e-12 ? pn.x : (std::abs(pn.y) > 1e-12 ? pn.y : pn.z);
                        if (lead < 0.0) { pn = pn * -1.0; pd = -pd; }
                        std::size_t q = 0;
                        while (q < C.planes.size() &&
                               !(dot(C.planes[q].n, pn) > 1.0 - 1e-12 && std::abs(C.planes[q].d - pd) < kFitEps)) ++q;
                        if (q == C.planes.size()) C.planes.push_back({pn, pd, {}});
                        C.planes[q].tris.push_back(ti);
                    }
        }
        // 2. Each cube's pieces and regions.
        std::vector<FitRegion> regions;
        std::unordered_map<std::size_t, std::int64_t> whole;  // cube without faces -> its region
        auto whole_region = [&](std::size_t j) -> std::int64_t {
            if (edge(j)) return -2;
            auto it = whole.find(j);
            if (it != whole.end()) return it->second;
            FitRegion r;
            r.cube = j;
            r.volume = 1.0;
            if (fit_air(j)) { r.owner = static_cast<std::int64_t>(j); r.fixed = true; }
            regions.push_back(r);
            whole[j] = static_cast<std::int64_t>(regions.size() - 1);
            return whole[j];
        };
        std::map<std::pair<std::size_t, std::size_t>, std::int64_t> pseudo;  // (cube, owner) -> region
        auto pseudo_region = [&](std::size_t c, std::size_t o) -> std::int64_t {
            auto it = pseudo.find({c, o});
            if (it != pseudo.end()) return it->second;
            FitRegion r;
            r.cube = c;
            r.owner = static_cast<std::int64_t>(o);
            r.fixed = true;
            regions.push_back(r);
            pseudo[{c, o}] = static_cast<std::int64_t>(regions.size() - 1);
            return pseudo[{c, o}];
        };
        std::vector<std::size_t> order;
        for (auto &kv : cubes) order.push_back(kv.first);
        std::sort(order.begin(), order.end());
        for (const std::size_t c : order) {
            FitCube &C = cubes[c];
            fit_stats_[0] += 1.0;
            if (C.planes.size() > max_planes) C.exact = false;
            if (!C.exact) {
                fit_stats_[10] += 1.0;
                // Fallback: S^3 samples at irrational offsets, owners by visibility.
                const double uv = 1.0 / (static_cast<double>(S) * S * S);
                for (int a = 0; a < S; ++a)
                    for (int b = 0; b < S; ++b)
                        for (int d = 0; d < S; ++d) {
                            const Vec3 x = centre(c) + Vec3{((a + jitter[0]) / S - 0.5) * cell_,
                                                            ((b + jitter[1]) / S - 0.5) * cell_,
                                                            ((d + jitter[2]) / S - 0.5) * cell_};
                            const std::size_t w = owner(bvh, x, c);
                            if (w != npos) regions[static_cast<std::size_t>(pseudo_region(c, w))].volume += uv;
                        }
                continue;
            }
            fit_stats_[9] += 1.0;
            std::vector<FitPolyhedron> pieces{unit_cube()};
            std::vector<std::uint32_t> sign{0u};
            for (std::size_t p = 0; p < C.planes.size(); ++p) {
                const Vec3 pn = C.planes[p].n;
                const double pd = C.planes[p].d;
                std::vector<FitPolyhedron> next;
                std::vector<std::uint32_t> nsign;
                for (std::size_t q = 0; q < pieces.size(); ++q) {
                    double lo = 1e300, hi = -1e300;
                    for (const auto &f : pieces[q])
                        for (const Vec3 &v : f.v) {
                            const double s = dot(pn, v) - pd;
                            lo = std::min(lo, s);
                            hi = std::max(hi, s);
                        }
                    if (lo < -kFitEps && hi > kFitEps) {
                        FitPolyhedron plus = clip_polyhedron(pieces[q], pn, pd, 6 + static_cast<int>(p));
                        FitPolyhedron minus = clip_polyhedron(pieces[q], pn * -1.0, -pd, 6 + static_cast<int>(p));
                        if (!plus.empty() && poly_volume(plus) > kFitTiny) {
                            next.push_back(std::move(plus));
                            nsign.push_back(sign[q] | (1u << p));
                        }
                        if (!minus.empty() && poly_volume(minus) > kFitTiny) {
                            next.push_back(std::move(minus));
                            nsign.push_back(sign[q]);
                        }
                    } else {
                        next.push_back(std::move(pieces[q]));
                        nsign.push_back(hi > kFitEps ? sign[q] | (1u << p) : sign[q]);
                    }
                }
                pieces = std::move(next);
                sign = std::move(nsign);
            }
            // Facets on each plane: covered (a wall), open or partly covered (joined).
            const std::size_t m = pieces.size();
            std::vector<std::size_t> parent(m);
            for (std::size_t q = 0; q < m; ++q) parent[q] = q;
            std::function<std::size_t(std::size_t)> root = [&](std::size_t q) {
                while (parent[q] != q) q = parent[q] = parent[parent[q]];
                return q;
            };
            std::vector<std::vector<std::pair<std::int32_t, double>>> piece_wall(m);
            for (std::size_t a = 0; a < m; ++a) {
                for (const auto &f : pieces[a]) {
                    if (f.tag < 6) continue;
                    const std::size_t p = static_cast<std::size_t>(f.tag - 6);
                    if (!(sign[a] & (1u << p))) continue;  // each facet once, from its + side
                    std::size_t b = m;
                    for (std::size_t q = 0; q < m; ++q)
                        if (sign[q] == (sign[a] ^ (1u << p))) b = q;
                    const double area = poly_area(f.v);
                    double cov = 0.0;
                    std::vector<std::pair<std::int32_t, double>> mats;
                    for (const std::size_t ti : C.planes[p].tris) {
                        const auto &T = C.tris[ti];
                        const double ca = covered(f.v, C.planes[p].n,
                                                  {to_local(c, T.p[0]), to_local(c, T.p[1]), to_local(c, T.p[2])});
                        cov += ca;
                        add_wall(mats, T.material, ca);
                    }
                    const double r = cov / area;
                    if (r < 1e-9) {
                        if (b < m) parent[root(a)] = root(b);
                        continue;
                    }
                    for (const auto &e : mats) {
                        add_wall(piece_wall[a], e.first, e.second);
                        if (b < m) add_wall(piece_wall[b], e.first, e.second);
                    }
                    if (r < 1.0 - 1e-9) {
                        fit_stats_[12] += 1.0;  // partly covered: joined
                        if (b < m) parent[root(a)] = root(b);
                    }
                }
            }
            // Regions: one per joined set of pieces; the one round the centre is the cell's.
            std::map<std::size_t, std::int64_t> made;
            C.sign = sign;
            C.region.assign(m, -1);
            for (std::size_t q = 0; q < m; ++q) {
                const std::size_t r = root(q);
                if (!made.count(r)) {
                    FitRegion R;
                    R.cube = c;
                    regions.push_back(R);
                    made[r] = static_cast<std::int64_t>(regions.size() - 1);
                }
                const std::int64_t g = made[r];
                C.region[q] = g;
                FitRegion &R = regions[static_cast<std::size_t>(g)];
                R.volume += poly_volume(pieces[q]);
                for (const auto &e : piece_wall[q]) add_wall(R.wall, e.first, e.second);
            }
            if (fit_air(c)) {
                const std::int64_t g = region_at(C, Vec3{0.0, 0.0, 0.0});
                if (g >= 0) {
                    regions[static_cast<std::size_t>(g)].owner = static_cast<std::int64_t>(c);
                    regions[static_cast<std::size_t>(g)].fixed = true;
                }
            }
        }
        // 3. The faces between cubes, each once, as one arrangement seen from both sides.
        struct Link {
            std::int64_t a, b;  // regions (-2: the grid's edge)
            std::size_t ca, cb;  // their cubes
            int k;  // direction from ca to cb
            double open;
            std::vector<std::pair<std::int32_t, double>> wall;  // on the face
        };
        std::vector<Link> links;
        std::vector<std::size_t> touched;
        for (const std::size_t c : order) {
            const FitCube &C = cubes[c];
            for (int k = 0; k < 6; ++k) {
                const std::size_t j = neighbour(c, k);
                auto jt = cubes.find(j);
                if (jt != cubes.end() && j < c) continue;
                const FitCube *J = jt == cubes.end() ? nullptr : &jt->second;
                const Vec3 e = unit(k);
                touched.push_back(c);
                touched.push_back(j);
                if (fit_air(c)) fit_done_[c * 18 + static_cast<std::size_t>(k)] = 1;
                if (!edge(j) && fit_air(j)) fit_done_[j * 18 + static_cast<std::size_t>(opposite(k))] = 1;
                // the square, in c's local coordinates
                const int a = k / 2, b = (a + 1) % 3, d = (a + 2) % 3;
                std::vector<Vec3> sq;
                const double corner[4][2] = {{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}};
                for (const auto &q : corner) {
                    double u[3];
                    u[a] = (k % 2) ? 0.5 : -0.5;
                    u[b] = q[0];
                    u[d] = q[1];
                    sq.push_back({u[0], u[1], u[2]});
                }
                const std::vector<std::size_t> &mem = C.membrane[static_cast<std::size_t>(k)];
                if (!C.exact || (J && !J->exact)) {
                    // sampled square (irrational offsets)
                    const double ua = 1.0 / (static_cast<double>(S) * S);
                    std::map<std::pair<std::int64_t, std::int64_t>, std::size_t> at;
                    for (int s1 = 0; s1 < S; ++s1)
                        for (int s2 = 0; s2 < S; ++s2) {
                            double u[3];
                            u[a] = (k % 2) ? 0.5 : -0.5;
                            u[b] = (s1 + jitter[b]) / S - 0.5;
                            u[d] = (s2 + jitter[d]) / S - 0.5;
                            const Vec3 y = centre(c) + Vec3{u[0], u[1], u[2]} * cell_;
                            const Vec3 ym = y - e * (1e-6 * cell_), yp = y + e * (1e-6 * cell_);
                            std::int64_t r1, r2;
                            if (C.exact) r1 = region_at(C, to_local(c, ym));
                            else { const std::size_t o = owner(bvh, ym, c); r1 = o == npos ? -1 : pseudo_region(c, o); }
                            if (edge(j)) r2 = -2;
                            else if (!J) r2 = whole_region(j);
                            else if (J->exact) r2 = region_at(*J, to_local(j, yp));
                            else { const std::size_t o = owner(bvh, yp, j); r2 = o == npos ? -1 : pseudo_region(j, o); }
                            if (r1 == -1 && r2 == -1) continue;
                            const bool wall = !sees(bvh, ym, yp);
                            auto key = std::make_pair(r1, r2);
                            if (!at.count(key)) {
                                at[key] = links.size();
                                links.push_back({r1, r2, c, j, k, 0.0, {}});
                            }
                            Link &L = links[at[key]];
                            if (wall) add_wall(L.wall, 0, ua);  // material found below
                            else L.open += ua;
                        }
                    // a sampled face's wall takes the material of the first face on it
                    for (auto &kv : at) {
                        Link &L = links[kv.second];
                        if (L.wall.empty()) continue;
                        std::int32_t mat = 0;
                        Hit hit = bvh.intersect(centre(c), e, cell_, -1);
                        if (hit.found()) mat = static_cast<std::int32_t>(face_material[hit.face]);
                        L.wall[0].first = mat;
                    }
                    continue;
                }
                // exact: the square cut by every plane of either cube that crosses it
                std::vector<std::pair<Vec3, double>> cutters;
                for (const auto &P : C.planes) cutters.push_back({P.n, P.d});
                if (J) for (const auto &P : J->planes) cutters.push_back({P.n, P.d + dot(P.n, e)});
                std::vector<std::vector<Vec3>> parts{sq};
                for (const auto &[pn, pd] : cutters) {
                    std::vector<std::vector<Vec3>> next;
                    for (auto &pt : parts) {
                        double lo = 1e300, hi = -1e300;
                        for (const Vec3 &v : pt) {
                            lo = std::min(lo, dot(pn, v) - pd);
                            hi = std::max(hi, dot(pn, v) - pd);
                        }
                        if (lo < -kFitEps && hi > kFitEps) {
                            auto p1 = clip_poly(pt, pn, pd, nullptr), p2 = clip_poly(pt, pn * -1.0, -pd, nullptr);
                            if (poly_area(p1) > kFitTiny) next.push_back(std::move(p1));
                            if (poly_area(p2) > kFitTiny) next.push_back(std::move(p2));
                        } else {
                            next.push_back(std::move(pt));
                        }
                    }
                    parts = std::move(next);
                }
                for (const auto &pt : parts) {
                    Vec3 m{0.0, 0.0, 0.0};
                    for (const Vec3 &v : pt) m = m + v;
                    m = m / static_cast<double>(pt.size());
                    const std::int64_t r1 = region_at(C, m);
                    std::int64_t r2;
                    if (edge(j)) r2 = -2;
                    else if (!J) r2 = whole_region(j);
                    else r2 = region_at(*J, m - e);
                    const double area = poly_area(pt);
                    Link L{r1, r2, c, j, k, area, {}};
                    double cov = 0.0;
                    for (const std::size_t ti : mem) {
                        const auto &T = C.tris[ti];
                        const double ca = covered(pt, e, {to_local(c, T.p[0]), to_local(c, T.p[1]), to_local(c, T.p[2])});
                        cov += ca;
                        add_wall(L.wall, T.material, ca);
                    }
                    L.open = area - cov > 1e-9 ? area - cov : 0.0;  // rounding of a covered face: shut
                    links.push_back(std::move(L));
                }
            }
        }
        // 4. Owners of the regions no cell's centre lies in: the owner of the neighbouring
        // region they share the most open face with (passes over the whole set at once).
        for (int pass = 0; pass < 6; ++pass) {
            std::map<std::int64_t, std::map<std::int64_t, double>> offer;
            for (const Link &L : links) {
                if (!(L.open > 0.0) || L.a < 0 || L.b < 0) continue;
                const FitRegion &A = regions[static_cast<std::size_t>(L.a)], &B = regions[static_cast<std::size_t>(L.b)];
                if (A.owner < 0 && B.owner >= 0) offer[L.a][B.owner] += L.open;
                if (B.owner < 0 && A.owner >= 0) offer[L.b][A.owner] += L.open;
            }
            if (offer.empty()) break;
            for (auto &[r, by] : offer) {
                std::int64_t best = -1;
                double most = -1.0;
                for (auto &[o, area] : by) {
                    if (area > most + 1e-12) { most = area; best = o; }
                    else if (std::abs(area - most) <= 1e-12) fit_stats_[11] += 1.0;  // tie: lowest index
                }
                regions[static_cast<std::size_t>(r)].owner = best;
                if (std::getenv("ACOUSTIC_SPACES_FIT_DEBUG")) {
                    const std::size_t q = regions[static_cast<std::size_t>(r)].cube;
                    std::fprintf(stderr, "pass %d region %lld cube (%zu %zu %zu) vol %.4f -> owner (%zu %zu %zu) area %.4f\n", pass,
                                 (long long)r, q % nx_, (q / nx_) % ny_, q / sz_, regions[static_cast<std::size_t>(r)].volume,
                                 (std::size_t)best % nx_, ((std::size_t)best / nx_) % ny_, (std::size_t)best / sz_, most);
                }
            }
        }
        // 5. Volumes, open areas and walls of the owners.
        auto key = [](std::int64_t o, int k) { return static_cast<std::size_t>(o) * 18 + static_cast<std::size_t>(k); };
        std::unordered_map<std::size_t, std::uint8_t> own_counted;
        for (const FitRegion &R : regions) {
            if (R.owner < 0) continue;
            const std::size_t o = static_cast<std::size_t>(R.owner);
            fit_volume_[o] += R.volume;
            if (R.cube == o) own_counted[o] = 1;
            for (const auto &e : R.wall) add_wall(fit_wall_[o], e.first, e.second);
            const auto ct = cubes.find(R.cube);
            if (ct != cubes.end() && !ct->second.exact) fit_sampled_[o] = 1;
            touched.push_back(o);
        }
        for (auto &[o, v] : fit_volume_) {
            if (!own_counted.count(o) && !cubes.count(o)) v += 1.0;  // its own cube, without faces
        }
        for (const Link &L : links) {
            const std::int64_t oa = L.a >= 0 ? regions[static_cast<std::size_t>(L.a)].owner : L.a;
            const std::int64_t ob = L.b >= 0 ? regions[static_cast<std::size_t>(L.b)].owner : L.b;
            const bool sampled = !cubes[L.ca].exact || (cubes.count(L.cb) && !cubes[L.cb].exact);
            if (oa >= 0 && sampled) fit_sampled_[static_cast<std::size_t>(oa)] = 1;
            if (ob >= 0 && sampled) fit_sampled_[static_cast<std::size_t>(ob)] = 1;
            for (const auto &e : L.wall) {
                if (oa >= 0) add_wall(fit_wall_[static_cast<std::size_t>(oa)], e.first, e.second);
                if (ob >= 0) add_wall(fit_wall_[static_cast<std::size_t>(ob)], e.first, e.second);
            }
            if (!(L.open > 0.0)) continue;
            if (oa >= 0 && ob == -2) {
                add_wall(fit_wall_[static_cast<std::size_t>(oa)], -1, L.open);
                continue;
            }
            if (oa < 0 || ob < 0) {
                if ((oa >= 0) != (ob >= 0)) fit_stats_[6] += L.open;  // open to nothing
                continue;
            }
            if (oa == ob) continue;
            const int kk = direction_to(static_cast<std::size_t>(oa), static_cast<std::size_t>(ob));
            if (kk < 0 || kk >= 6) {
                // owners along an edge diagonal (a sliver given to one, open to the other):
                // a diagonal link of the 7-point stencil (env ACOUSTIC_SPACES_FIT_DIAG: 0 drop,
                // 1 g A / 2 (two-point flux, n.d / |d|^2), 2 g A: area over the normal distance, as the
                // axial links have it; the default)
                if (kk >= 6 && diag_mode_ > 0 && type_[static_cast<std::size_t>(oa)] == kInterior &&
                    type_[static_cast<std::size_t>(ob)] == kInterior) {
                    const double f = diag_mode_ == 1 ? 0.5 : 1.0;
                    fit_area_[key(oa, kk)] += f * L.open;
                    fit_area_[key(ob, opposite(kk))] += f * L.open;
                    fit_done_[key(oa, kk)] = 1;
                    fit_done_[key(ob, opposite(kk))] = 1;
                    fit_stats_[8] += L.open;
                    continue;
                }
                fit_stats_[6] += L.open;
                continue;
            }
            // a face to the matched layer stays whole (the layer's update takes it so)
            fit_area_[key(oa, kk)] += L.open;
            fit_area_[key(ob, opposite(kk))] += L.open;
            fit_done_[key(oa, kk)] = 1;
            fit_done_[key(ob, opposite(kk))] = 1;
        }
        // A face no cube with faces touches, of an owner that took a sliver: its own whole
        // face (or the grid's edge) was never visited above; add it.
        for (auto &[k2, flag] : fit_done_) {
            (void)flag;
            const std::size_t o = k2 / 18;
            const int k = static_cast<int>(k2 % 18);
            if (k >= 6) continue;
            const std::size_t j = neighbour(o, k);
            if (cubes.count(o) || cubes.count(j)) continue;
            if (edge(j)) {
                add_wall(fit_wall_[o], -1, 1.0);
            } else if (fit_air(j)) {
                fit_area_[k2] += 1.0;
            }
        }
        // B4 (on; env ACOUSTIC_SPACES_FIT_SLIVER=0 turns it off): a cell that took a sliver of thickness f
        // through its face k (sliver volume over the open area between them) keeps the
        // volume 1 + f, and the link on its other side (opposite k) is scaled by
        // w(lam) = 3 lam / ((lam + 1/2)(lam + 1)), lam = 1 + f (<= 1: the bound holds), which
        // puts the wall in place to fourth order in 1D (NOTES of round 2).
        const char *env_sliver = std::getenv("ACOUSTIC_SPACES_FIT_SLIVER");
        if (!(env_sliver && std::strcmp(env_sliver, "0") == 0)) {
            std::map<std::pair<std::size_t, int>, std::pair<double, double>> sl;  // (o, k) -> volume, area
            for (const FitRegion &R : regions) {
                if (R.owner < 0 || R.cube == static_cast<std::size_t>(R.owner)) continue;
                const int k = direction_to(static_cast<std::size_t>(R.owner), R.cube);
                if (k >= 0 && k < 6) sl[{static_cast<std::size_t>(R.owner), k}].first += R.volume;
            }
            for (const Link &L : links) {
                if (L.a < 0 || L.b < 0 || !(L.open > 0.0)) continue;
                const FitRegion &A = regions[static_cast<std::size_t>(L.a)], &Bq = regions[static_cast<std::size_t>(L.b)];
                if (A.owner < 0 || A.owner != Bq.owner) continue;
                const std::size_t o = static_cast<std::size_t>(A.owner);
                const FitRegion &other = A.cube == o ? Bq : A;
                if ((A.cube == o) == (Bq.cube == o)) continue;
                const int k = direction_to(o, other.cube);
                if (k >= 0 && k < 6) sl[{o, k}].second += L.open;
            }
            for (const auto &[ok, va] : sl) {
                if (!(va.second > 1e-9)) continue;
                const double lam = 1.0 + va.first / va.second;
                const double w = 3.0 * lam / ((lam + 0.5) * (lam + 1.0));
                const std::size_t o = ok.first;
                const int k = opposite(ok.second);
                const std::size_t m = neighbour(o, k);
                const double before = axial_area_raw(o, k, m);
                fit_area_[key(static_cast<std::int64_t>(o), k)] = before * w;
                fit_area_[key(static_cast<std::int64_t>(m), opposite(k))] = before * w;
                fit_done_[key(static_cast<std::int64_t>(o), k)] = 1;
                fit_done_[key(static_cast<std::int64_t>(m), opposite(k))] = 1;
                fit_stats_[14] += 1.0;
            }
        }
        // Faces to the matched layer: whole, as the layer's update has them.
        for (auto &[k2, v] : fit_done_) {
            const std::size_t i = k2 / 18;
            const int k = static_cast<int>(k2 % 18);
            if (k < 6 && type_[i] == kInterior && type_[neighbour(i, k)] == kLayer) fit_area_[k2] = 1.0;
        }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        return touched;
    }
    std::unordered_map<std::size_t, std::vector<std::pair<std::int32_t, double>>> fit_wall_;
    std::unordered_map<std::size_t, std::uint8_t> fit_sampled_;
    int diag_mode_ = 0;
    // 7-point fitted: a cell's diagonal links (g times their stored share), 0 elsewhere.
    double diag_weight(std::size_t i, int k) const {
        if (!fit_exact_ || diag_mode_ == 0 || stencil_ != 7) return 0.0;
        const auto it = fit_area_.find(i * 18 + static_cast<std::size_t>(k));
        return it == fit_area_.end() ? 0.0 : exact_[0] * it->second;
    }
    double axial_area_raw(std::size_t o, int k, std::size_t m) const {
        const std::size_t key = o * 18 + static_cast<std::size_t>(k);
        if (fit_done_.count(key)) {
            const auto it = fit_area_.find(key);
            return it == fit_area_.end() ? 0.0 : it->second;
        }
        return (!edge(m) && fit_air(m)) ? 1.0 : 0.0;
    }
    int link_directions() const { return (fitted_ > 0 && fit_exact_ && diag_mode_ > 0) ? 18 : directions_; }
    // ---- end round 2 -----------------------------------------------------------------------

    double fit_volume(std::size_t i) const {
        const auto it = fit_volume_.find(i);
        return it == fit_volume_.end() ? 1.0 : it->second;
    }

    // The open area of air cell i's cube face along axial direction k (fitted).
    double axial_area(const std::vector<std::uint16_t> &cut, std::size_t i, int k) const {
        const std::size_t key = i * 18 + static_cast<std::size_t>(k);
        if (fit_done_.count(key)) {
            const auto it = fit_area_.find(key);
            return it == fit_area_.end() ? 0.0 : it->second;
        }
        return blocked(cut, i, k) ? 0.0 : 1.0;
    }

    // The weights of air cell i's links before averaging, fitted: axial links by their
    // faces' open area; on the 19-point stencil an open edge link by its share, and a
    // blocked one folded as raw_weights does, by the open area of the axial face it
    // folds onto.
    void raw_fitted(const std::vector<std::uint16_t> &cut, std::size_t i, double *w) const {
        if (stencil_ == 19 && fit19_volume_) {
            // Variant: the voxel weights, every link of a cell with less than a whole cube
            // of air scaled by the smaller of its two cells' shares (at most 1).
            raw_weights(cut, i, w);
            const double vi = std::min(1.0, fit_volume(i));
            for (int k = 0; k < directions_; ++k) {
                if (w[k] > 0.0) w[k] *= std::min(vi, std::min(1.0, fit_volume(neighbour(i, k))));
            }
            return;
        }
        for (int k = 0; k < 18; ++k) w[k] = 0.0;
        for (int k = 0; k < 6; ++k) w[k] += exact_[k] * axial_area(cut, i, k);
        for (int k = 6; k < directions_; ++k) {
            const std::size_t key = i * 18 + static_cast<std::size_t>(k);
            if (!blocked(cut, i, k)) {
                double share = 1.0;
                if (fit_done_.count(key)) {
                    const auto it = fit_area_.find(key);
                    share = it == fit_area_.end() ? 0.0 : it->second;
                }
                w[k] += exact_[k] * share;
                continue;
            }
            if (edge(neighbour(i, k))) continue;
            const auto [ka, kb] = steps_of(k);
            const bool behind_a = blocked(cut, i, ka), behind_b = blocked(cut, i, kb);
            if (behind_a && behind_b) continue;
            if (behind_a) {
                w[kb] += exact_[k] * axial_area(cut, i, kb);
            } else if (behind_b) {
                w[ka] += exact_[k] * axial_area(cut, i, ka);
            } else {
                w[ka] += 0.5 * exact_[k] * axial_area(cut, i, ka);
                w[kb] += 0.5 * exact_[k] * axial_area(cut, i, kb);
            }
        }
    }

    std::size_t fitted_ = 0;
    bool fit19_volume_ = false;
    std::unordered_map<std::size_t, double> fit_volume_, fit_area_;
    std::unordered_map<std::size_t, std::uint8_t> fit_done_;
    std::vector<double> special_volume_;
    std::array<double, 16> fit_stats_{};
    bool fit_exact_ = false;
    mutable std::size_t fit_rays_ = 0;
    // ---- end PROTOTYPE ---------------------------------------------------------------------

    Vec3 origin_;
    double cell_;
    std::size_t nx_, ny_, nz_, sy_, sz_;
    std::size_t layer_ = 0;
    std::size_t stencil_ = 7;
    double layer_damping_ = 0.0;
    std::size_t threads_ = 1;
    std::size_t air_ = 0, boundary_cells_ = 0;
    bool open_ = false;
    std::array<bool, 6> open_sides_{};
    // The stencil: its directions, their weights (exact, and as the update uses them),
    // the weight of a cut link as a boundary, the offset to the neighbour, the bit of
    // the + half's directions in the cut mask (-1: the - half), and the centre's weight.
    int directions_ = 6;
    std::array<double, 18> exact_{}, boundary_{};
    std::array<float, 18> weight_{};
    std::array<std::ptrdiff_t, 18> offset_{};
    std::array<int, 18> bit_{};
    float centre_ = 0.0f;
    float total_ = 0.0f;  // the update's weights summed
    // With a speed field: the square of each cell's speed of sound over the grid's, and
    // of each matched-layer cell's; both empty without one.
    std::vector<float> scale_, layer_scale_;
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
    // Moving air: each cell's shifts and gradient neighbours (see set_up_flow), and the
    // carrying's two stages; all empty in still air.
    std::vector<float> flow_x_, flow_y_, flow_t1_, flow_t2_;
    std::vector<std::uint8_t> flow_mask_;
};

}  // namespace acoustic_spaces
