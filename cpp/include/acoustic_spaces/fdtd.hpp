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
// boundary, of the material of the face it crosses nearest the cell (a centre on a
// face lies on its side towards +x, else +y, else +z: see kTie), with weight
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
// two cells either side are air, second where one is. Over its small steps (|v| T / h
// is under 0.1 for the weather's wind) it loses a little of the shortest waves. A
// velocity of half a cell per step or more along an axis is refused, which is the
// carrying's own bound and no promise for the two steps together: where the wind
// changes or ends they keep no energy of their own. The surfaces' branch velocities,
// the source and the probes stay where they are, and the matched layer's air is still;
// at its border the moving air meets still air, where the wind comes out of the layer
// with the layer's own sound and leaves into it. In the open, sound has been seen to
// grow there at winds beyond the weather's (from 0.09 cells per step on, a wind at 45
// degrees to the grid; not at 0.07), so a run in moving air is to be watched
// (mean_square()).
//
// That equation keeps the sound's energy, (|q_t|^2 - c^2 <q, lap q> - |v . grad q|^2) / 2,
// when the carrying operator A = v . grad is antisymmetric, which on the grid it is when
// no wind crosses a face of the model and no cell has a net flux of air. A wind that
// blows through a solid is neither: where it comes out of a face it carries the face's
// own sound back into the air, where it goes in it takes sound away, and a room whose
// walls it leaves more than it enters grows without bound. So where the given
// velocities meet a face, or change along a row of cells, the wind is put on the cells'
// faces and made to flow round the solids (set_up_round): the given wind u0 less a
// correction k grad lambda, k the given speed at the face, with lambda (0 in the matched
// layer) solved so that every cell's faces sum to zero with none through a face of the
// model. That is the flow of least change, each face's change weighed by 1 / k, as
// mass-consistent wind models make it. A wind that only changes with height is itself
// k grad (the distance along the wind), so a closed room keeps still air and a dead
// end nearly still air whatever the wind's profile, and far from the solids the wind is
// the given one.
// It has no wake and no eddies behind a solid, and at a sharp edge it is fast, the
// faster the finer the grid. The carrying operator is then made of pairs of opposite
// sign only: u / 2 over each open face, and for every three cells in a row joined by two
// open faces the loop (first, second): a, (second, third): a, (first, third): -a, with a
// the two faces' winds over 24. Where the wind is the same along the row that is the
// fourth-order central difference; where it changes, the operator is of second order.
// It is taken as differences of neighbours, so a uniform pressure is carried to the bit,
// and is antisymmetric to within each cell's net flux, which is the solve's and single
// precision's rounding. Where the given velocities meet no face and are the same along
// each row (an open field over flat ground), they are taken as they are, cell by cell,
// by the central differences: the same operator, but next to the matched layer, where
// those are of second order.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
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
    // above); without it the speed is the same everywhere. The probes come in groups of
    // `probes_per_listener`, the first of each a listener's own point and the others
    // round it: a source on a face lies on a side of it chosen by the listeners its air
    // reaches (see place_on_face), and each probe is in sight of its group's listener
    // or not (see probe_in_sight).
    WaveSolver(const Bvh &bvh, const std::int64_t *face_material, const double *beta,
               std::size_t materials, const double *branches, const std::int64_t *branch_material,
               std::size_t branch_count, const Vec3 &origin, double cell, std::size_t nx,
               std::size_t ny, std::size_t nz, const Vec3 &source,
               const std::vector<Vec3> &probes, double edge_admittance, std::size_t layer,
               double layer_damping, std::size_t threads, std::size_t stencil = 7,
               const float *speed = nullptr, const float *flow_x = nullptr,
               const float *flow_y = nullptr, std::size_t probes_per_listener = 1)
        : origin_(origin), cell_(cell), nx_(nx), ny_(ny), nz_(nz), sy_(nx), sz_(nx * ny),
          layer_(layer), stencil_(stencil) {
        if (stencil != 7 && stencil != 19) {
            throw std::invalid_argument("the stencil has 7 or 19 points");
        }
        if (probes_per_listener == 0 || probes.size() % probes_per_listener != 0) {
            throw std::invalid_argument("the probes come in whole groups, one per listener");
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
        // Of the cells a face's plane passes that near, found once for their links: those
        // whose centre lies on a face (see kTie), and those within the reach of crossing's
        // short casts of a face (see blind). Every other centre is an End with neither.
        std::vector<std::uint8_t> centre_ends(n, 0);  // bit 0 on a face, bit 1 within reach
        parallel_for(n, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                if (near[i] & kNearPlane) centre_ends[i] = ends_at(bvh, centre(i));
            }
        });
        const auto centre_end = [&](std::size_t i) {
            return End{centre(i), {}, (centre_ends[i] & 1) != 0, (centre_ends[i] & 2) != 0};
        };
        std::vector<std::uint16_t> cut(n, 0);
        parallel_for(n, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                if (!near[i]) continue;
                const End from = centre_end(i);
                const long long at[3] = {static_cast<long long>(i % nx),
                                         static_cast<long long>((i / nx) % ny),
                                         static_cast<long long>(i / sz_)};
                const long long size[3] = {static_cast<long long>(nx), static_cast<long long>(ny),
                                           static_cast<long long>(nz)};
                for (int k = 0; k < directions_; ++k) {
                    if (bit_[k] < 0) continue;
                    bool inside = true;
                    long long to[3];
                    for (int a = 0; a < 3; ++a) {
                        to[a] = at[a] + kSteps[k][a];
                        inside = inside && to[a] >= 0 && to[a] < size[a];
                    }
                    if (!inside) continue;
                    // (The neighbour's centre as centre() has it, from its coordinates.)
                    const std::uint8_t ends = centre_ends[neighbour(i, k)];
                    const End there{
                        centre_at(to[0], to[1], to[2]), {}, (ends & 1) != 0, (ends & 2) != 0};
                    if (crossing(bvh, from, there, unit(k), length(k)) >= 0) {
                        cut[i] = static_cast<std::uint16_t>(cut[i] | (1u << bit_[k]));
                    }
                }
            }
        });

        // Air: every cell reachable from the cells the source sees (for a source on a
        // face, see place_on_face).
        static_assert(kSolid == 0, "flood marks the cells still 0");
        source_on_face_ = on_face(bvh, source);
        if (source_on_face_) {
            place_on_face(bvh, cut, source, probes, probes_per_listener);
        } else {
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
            flood(cut, queue, type_, kInterior);
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
                    reach_sides(j, 0);
                    return;
                }
                // (Cast from i along the link, else from j back, as the + half's cut is.)
                const End from = centre_end(i), to = centre_end(j);
                std::int64_t face = crossing(bvh, from, to, unit(k), length(k));
                if (face < 0) face = crossing(bvh, to, from, unit(opposite(k)), length(k));
                if (face < 0) return;  // grazing a face's edge: as if rigid
                const auto m = static_cast<std::int32_t>(face_material[face]);
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
            const double r = std::sqrt(cell_scale(i));
            for (std::size_t kind = 0; kind < kinds; ++kind) {
                const auto &own = material_branches[static_cast<std::size_t>(crossed[kind])];
                for (const std::int32_t j : own) {
                    g += crossed_weight[kind] * (0.5 / branch_a_[static_cast<std::size_t>(j)]);
                    state_branch_.push_back(j);
                    state_weight_.push_back(static_cast<float>(r * crossed_weight[kind]));
                }
            }
            add_special(i, weight.data(), g);
        }
        boundary_cells_ = special_.size();

        // The source's cells (made special, so one update adds the source): none of the
        // cells round it in the matched layer, whichever cells it then takes.
        for (const auto &[i, w] : visible_cells(bvh, source, &type_, source_lean_)) {
            if (type_[i] == kLayer) {
                throw std::invalid_argument("the source is in the matched layer");
            }
        }
        for (const auto &[i, w] : reading_cells(bvh, cut, source, true, source_lean_)) {
            std::size_t k = special_index(i);
            if (k == npos) k = add_special(i, exact_.data(), 0.0);
            source_[k] += static_cast<float>(cell_scale(i) * w);
        }
        std::vector<End> probe_ends;  // each probe, and its lean (see kTie)
        Vec3 group_lean{};            // the lean of the listener of the group, if it lies on a face
        bool group_on_face = false;
        for (std::size_t k = 0; k < probes.size(); ++k) {
            const Vec3 &p = probes[k];
            const bool listener = k % probes_per_listener == 0;
            // (kNear cells away to within rounding counts as that far.)
            const bool near = norm(p - source) < (kNear - 1e-9) * cell;
            // A listener on faces (see kTie) reads with the first of its leans (see leans)
            // that has air, the rule's sides, or where those have none (a listener on a
            // ceiling, in a room's corner, on a pillar's edge) other sides; on an edge or in
            // a corner, the widest of those (see widest). The other probes of its group
            // take its lean, as they move with it (a point round a listener on a pillar's
            // edge, on one of its faces, leans with the listener into the middle of the air
            // round the edge); a probe on faces that has no air with it, or round a
            // listener on no face, chooses its own lean as a listener does.
            Vec3 lean{};
            const bool on = on_face(bvh, p);
            if (listener) group_on_face = on;
            if (!listener && group_on_face) lean = group_lean;
            if (on &&
                (listener || !group_on_face || reading_cells(bvh, cut, p, !near, lean).empty())) {
                const Leans candidates = leans(bvh, p);
                const auto has_air = [&](std::size_t c) {
                    return !reading_cells(bvh, cut, p, !near, candidates.all[c]).empty();
                };
                lean = widest(bvh, p, candidates, has_air).second;
            }
            if (listener) group_lean = lean;
            const auto cells = reading_cells(bvh, cut, p, !near, lean);
            probe_ends.push_back(End{p, lean, true});
            probe_cells_.push_back({});
            probe_weights_.push_back({});
            for (const auto &[i, w] : cells) {
                probe_cells_.back().push_back(i);
                probe_weights_.back().push_back(w);
            }
        }
        // Whether each probe's listener sees it, each with the lean its reading takes: a
        // probe across a surface from its listener (a screen, a balcony's front) can have
        // air of its own, and a difference across the surface is no gradient there.
        for (std::size_t k = 0; k < probes.size(); ++k) {
            const std::size_t l = k - k % probes_per_listener;
            probe_in_sight_.push_back(k == l || crossing(bvh, probe_ends[l], probe_ends[k]) < 0);
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
    // In moving air: whether the wind was made to flow round the model's solids (see the
    // notes above), the fastest it then moves through a face between two cells (cells
    // per step; else the largest of the given velocities' components along x and y), and
    // the sweeps its solve took.
    bool wind_round() const { return round_; }
    double wind_fastest() const { return fastest_; }
    std::size_t wind_sweeps() const { return sweeps_; }
    // In moving air, the pressure squared over the grid's air and the steps of the last
    // run(), on average (the pressure before each step, as the probes read it); 0 in
    // still air, where it is not summed.
    double mean_square() const { return mean_square_; }
    std::size_t probes() const { return probe_cells_.size(); }
    bool probe_found(std::size_t p) const { return !probe_cells_.at(p).empty(); }
    // Whether probe p's listener (the first probe of its group) sees it, each point with
    // the lean its reading takes (see kTie); a listener sees its own point.
    bool probe_in_sight(std::size_t p) const { return probe_in_sight_.at(p) != 0; }
    // Whether the source lies on a face (see kTie), and so sends sound into the space
    // on one side of it only (see place_on_face).
    bool source_on_face() const { return source_on_face_; }
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
        static const char *const wind[3] = {"wind_x", "wind_y", "wind_z"};
        static const char *const loop[3] = {"loop_x", "loop_y", "loop_z"};
        for (int d = 0; d < 3; ++d) {
            put(wind[d], wind_[d]);
            put(loop[d], loop_[d]);
        }
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
        std::vector<double> squares(t_count, 0.0);  // each thread's sum, in moving air
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
            double squared = 0.0;
            for (std::size_t step = 0; step < steps; ++step) {
                if (t == 0) record(b, step, steps, out);
                if (flowing) {
                    // Both pressures carried along with the air (b's last pass waits for
                    // two barriers, by when thread 0 has recorded it), and b's squares
                    // summed on the way, for mean_square().
                    squared += carry(b, ib, ie, barrier);
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
            squares[t] = squared;
        };
        {
            std::vector<std::thread> pool;
            for (std::size_t t = 1; t < t_count; ++t) pool.emplace_back(work, t);
            work(0);
            for (auto &thread : pool) thread.join();
        }
        mean_square_ = 0.0;
        if (steps > 0 && air_ > 0) {
            for (const double sum : squares) mean_square_ += sum;
            mean_square_ /= static_cast<double>(steps) * static_cast<double>(air_);
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
        mean_square_ = 0.0;
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
    // layer or out of the air). Where they meet no face of the model and are the same
    // along each row of cells, each cell is carried by its own (the plain way): which of
    // its neighbours along x and y the gradient may take is kept in flow_mask_, bits 0 to
    // 3 for x (the one below, above, two below, two above), 4 to 7 for y. Otherwise the
    // wind is made to flow round the solids (set_up_round).
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
            fastest_ = std::max({fastest_, std::abs(fx[i]), std::abs(fy[i])});
            for (int k = 0; k < 4; ++k) {
                const std::size_t j = neighbour(i, k);
                if (!blocked(cut, i, k) && type_[j] != kSolid) near[i] |= std::uint8_t(1u << k);
            }
        }
        // Plain: every cell the air moves along an axis has both its neighbours along it,
        // moving as it does where they move at all.
        bool plain = true;
        for (std::size_t i = 0; i < n && plain; ++i) {
            if (!moving(i)) continue;
            for (int axis = 0; axis < 2 && plain; ++axis) {
                const float *f = axis == 0 ? flow_x_.data() : flow_y_.data();
                if (f[i] == 0.0f) continue;
                for (int k = 2 * axis; k < 2 * axis + 2; ++k) {
                    const std::size_t j = neighbour(i, k);
                    plain = plain && (near[i] & (1u << k)) && (!moving(j) || f[j] == f[i]);
                }
            }
        }
        for (std::size_t i = 0; i < n && plain; ++i) {
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
        if (!plain) set_up_round(cut);
        flow_t1_.assign(n, 0.0f);  // (after the wind's solve, which takes room of its own)
        flow_t2_.assign(n, 0.0f);
    }

    // The wind round the solids (see the notes above), from the cells' given velocities
    // in flow_x_ and flow_y_ (which it lets go of): the wind on each cell's faces towards
    // +x, +y and +z (wind_; 0 on a face it does not cross), each row triple's loop
    // (loop_), and in flow_mask_ whether the air moves the cell at all.
    //
    // A face is open to the wind between two cells of air that no face of the model
    // parts, one of them at least outside the matched layer; without a matched layer
    // (the grid's edge is then a wall to the wind), only between cells two or more from
    // the edge, so that a carried cell's row of five lies on the grid. lambda solves
    //   sum over a cell's open faces of k (lambda - lambda') = -(the given wind's net flux)
    // by conjugate gradients on the diagonal's scale, in double precision, to a net flux
    // of 1e-9 of the fastest given wind. Its sums are taken layer by layer along z, so
    // the wind is the same whatever the number of threads.
    void set_up_round(const std::vector<std::uint16_t> &cut) {
        const std::size_t n = nx_ * ny_ * nz_;
        const std::size_t stride[3] = {1, sy_, sz_};
        auto moving = [&](std::size_t i) { return type_[i] == kInterior || type_[i] == kSpecial; };
        std::vector<float> speed(n, 0.0f);  // the given speed, 0 where the air does not move
        std::vector<std::uint8_t> faces(n, 0);  // bit d: the face towards +d is open
        const float *given[2] = {flow_x_.data(), flow_y_.data()};
        const std::size_t least = layer_ > 0 ? 1 : 2;
        parallel_for(n, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                if (type_[i] == kSolid || depth(i) < least) continue;
                speed[i] = moving(i) ? std::hypot(given[0][i], given[1][i]) : 0.0f;
                for (int d = 0; d < 3; ++d) {
                    const std::size_t j = i + stride[d];
                    if (type_[j] == kSolid || depth(j) < least || blocked(cut, i, 2 * d + 1)) continue;
                    if (moving(i) || moving(j)) faces[i] |= std::uint8_t(1u << d);
                }
            }
        });
        // A face's weight k and the given wind through it: the two cells' mean, or the
        // moving one's beside the matched layer (whose cells hold 0 here).
        auto mean = [&](const float *f, std::size_t i, std::size_t j) {
            const double sum = static_cast<double>(f[i]) + static_cast<double>(f[j]);
            return moving(i) && moving(j) ? 0.5 * sum : sum;
        };
        auto through = [&](std::size_t i, int d) {
            return d < 2 ? mean(given[d], i, i + stride[d]) : 0.0;
        };
        std::vector<double> lambda(n, 0.0), residual(n, 0.0), direction(n, 0.0), image(n, 0.0);
        std::vector<float> diagonal(n, 0.0f);
        std::vector<double> slab(nz_, 0.0);
        // term(i) summed (or its largest) over the cells, layer by layer along z.
        auto over = [&](bool largest, auto &&term) {
            parallel_for(nz_, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
                for (std::size_t z = begin; z < end; ++z) {
                    double total = 0.0;
                    for (std::size_t i = z * sz_; i < (z + 1) * sz_; ++i) {
                        if (!moving(i)) continue;
                        const double v = term(i);
                        total = largest ? std::max(total, v) : total + v;
                    }
                    slab[z] = total;
                }
            });
            double total = 0.0;
            for (const double v : slab) total = largest ? std::max(total, v) : total + v;
            return total;
        };
        // The right side, minus the given wind's net flux out of each cell, and the
        // diagonal.
        const double top = over(true, [&](std::size_t i) {
            double flux = 0.0, weights = 0.0;
            for (int d = 0; d < 3; ++d) {
                const std::size_t s = stride[d];
                if (faces[i] & (1u << d)) {
                    flux += through(i, d);
                    weights += mean(speed.data(), i, i + s);
                }
                if (faces[i - s] & (1u << d)) {
                    flux -= through(i - s, d);
                    weights += mean(speed.data(), i - s, i);
                }
            }
            residual[i] = weights > 0.0 ? -flux : 0.0;
            diagonal[i] = static_cast<float>(weights);
            return static_cast<double>(speed[i]);
        });
        auto scaled = [&](std::size_t i) {
            return diagonal[i] > 0.0f ? residual[i] / static_cast<double>(diagonal[i]) : 0.0;
        };
        double carried_over = over(false, [&](std::size_t i) {
            direction[i] = scaled(i);
            return residual[i] * direction[i];
        });
        const double enough = 1e-9 * top;
        double left = over(true, [&](std::size_t i) { return std::abs(residual[i]); });
        const std::size_t most = 1000 + 50 * (nx_ + ny_ + nz_);
        while (left > enough) {
            if (++sweeps_ > most) {
                throw std::runtime_error("the wind round the model's solids did not settle");
            }
            const double bent = over(false, [&](std::size_t i) {
                double sum = 0.0;
                for (int d = 0; d < 3; ++d) {
                    const std::size_t s = stride[d];
                    if (faces[i] & (1u << d)) {
                        sum += mean(speed.data(), i, i + s) * (direction[i] - direction[i + s]);
                    }
                    if (faces[i - s] & (1u << d)) {
                        sum += mean(speed.data(), i - s, i) * (direction[i] - direction[i - s]);
                    }
                }
                image[i] = sum;
                return direction[i] * sum;
            });
            if (!(bent > 0.0)) break;
            const double step = carried_over / bent;
            left = over(true, [&](std::size_t i) {
                lambda[i] += step * direction[i];
                residual[i] -= step * image[i];
                return std::abs(residual[i]);
            });
            const double next = over(false, [&](std::size_t i) { return residual[i] * scaled(i); });
            const double turn = next / carried_over;
            carried_over = next;
            over(false, [&](std::size_t i) {
                direction[i] = scaled(i) + turn * direction[i];
                return 0.0;
            });
        }
        residual = {};
        direction = {};
        image = {};
        // The faces' winds, and the fastest of them.
        for (int d = 0; d < 3; ++d) wind_[d].assign(n, 0.0f);
        parallel_for(nz_, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t z = begin; z < end; ++z) {
                double fastest = 0.0;
                for (std::size_t i = z * sz_; i < (z + 1) * sz_; ++i) {
                    for (int d = 0; d < 3; ++d) {
                        if (!(faces[i] & (1u << d))) continue;
                        const std::size_t j = i + stride[d];
                        const double u =
                            through(i, d) - mean(speed.data(), i, j) * (lambda[j] - lambda[i]);
                        wind_[d][i] = static_cast<float>(u);
                        fastest = std::max(fastest, std::abs(u));
                    }
                }
                slab[z] = fastest;
            }
        });
        fastest_ = 0.0f;
        for (const double v : slab) fastest_ = std::max(fastest_, static_cast<float>(v));
        // A wind that nowhere reaches a millionth of the given one (a closed room's) is
        // none: still air, to the bit.
        if (static_cast<double>(fastest_) <= 1e-6 * top) {
            fastest_ = 0.0f;
            for (int d = 0; d < 3; ++d) std::fill(wind_[d].begin(), wind_[d].end(), 0.0f);
        }
        if (!(fastest_ < 0.5f)) {
            throw std::invalid_argument(
                "the air must move less than half a cell per step, and the wind round the "
                "model's solids moves " + std::to_string(fastest_) + " at its fastest");
        }
        lambda = {};
        flow_x_ = {};
        flow_y_ = {};
        // The loops, and the cells the air moves at all.
        for (int d = 0; d < 3; ++d) loop_[d].assign(n, 0.0f);
        parallel_for(n, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                for (int d = 0; d < 3; ++d) {
                    if (!(faces[i] & (1u << d)) || !(faces[i + stride[d]] & (1u << d))) continue;
                    loop_[d][i] = (wind_[d][i] + wind_[d][i + stride[d]]) * (1.0f / 24.0f);
                }
            }
        });
        parallel_for(n, threads_, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                if (!moving(i) || depth(i) < 2) continue;  // none next to the edge
                bool moved = false;
                for (int d = 0; d < 3; ++d) {
                    const std::size_t s = stride[d];
                    moved = moved || wind_[d][i] != 0.0f || wind_[d][i - s] != 0.0f ||
                            loop_[d][i] != 0.0f || loop_[d][i - s] != 0.0f ||
                            loop_[d][i - 2 * s] != 0.0f;
                }
                flow_mask_[i] = moved ? 1 : 0;
            }
        });
        round_ = true;
    }

    // q's change along an axis (stride s) at cell i, over a cell, from its neighbours
    // m allows (bits as in set_up_flow, shifted down to the axis): fourth-order central
    // differences, or second order. (The plain way's cells have both neighbours along
    // every axis the air moves them; the one-sided differences are for none.)
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

    // -v . grad q at cell i, over a step, the plain way.
    float carried(const float *q, std::size_t i) const {
        const float sx = flow_x_[i], sy = flow_y_[i];
        const unsigned m = flow_mask_[i];
        return -(sx * gradient(q, i, 1, m & 15u, sx) + sy * gradient(q, i, sy_, m >> 4, sy));
    }

    // The same with the wind round the solids: over each axis the two faces' winds times
    // the differences across them, halved, and the three loops the cell is part of.
    float carried_round(const float *q, std::size_t i) const {
        const std::size_t stride[3] = {1, sy_, sz_};
        const float c = q[i];
        float sum = 0.0f;
        for (int d = 0; d < 3; ++d) {
            const std::size_t s = stride[d];
            const float *w = wind_[d].data(), *l = loop_[d].data();
            const float above = q[i + s], below = q[i - s];
            sum += 0.5f * (w[i] * (above - c) + w[i - s] * (c - below)) +
                   l[i] * (above - q[i + 2 * s]) + l[i - s] * (above - below) +
                   l[i - 2 * s] * (q[i - 2 * s] - below);
        }
        return -sum;
    }

    // Carries q (cells ib to ie on this thread) along with the air for a step: the
    // strong-stability-preserving Runge-Kutta method of third order, through the two
    // stages' fields; cells the air does not move keep their values, to the bit.
    // Returns the sum of those cells' squares as they came.
    double carry(float *q, std::size_t ib, std::size_t ie, StepBarrier &barrier) {
        if (round_) {
            const std::uint8_t *moved = flow_mask_.data();
            return carry(
                q, ib, ie, barrier, [&](std::size_t i) { return !moved[i]; },
                [&](const float *p, std::size_t i) { return carried_round(p, i); });
        }
        const float *fx = flow_x_.data(), *fy = flow_y_.data();
        return carry(
            q, ib, ie, barrier, [&](std::size_t i) { return fx[i] == 0.0f && fy[i] == 0.0f; },
            [&](const float *p, std::size_t i) { return carried(p, i); });
    }

    template <typename Still, typename Carried>
    double carry(float *q, std::size_t ib, std::size_t ie, StepBarrier &barrier, Still &&still,
                 Carried &&carried) {
        float *t1 = flow_t1_.data(), *t2 = flow_t2_.data();
        const std::uint8_t *type = type_.data();
        double squares = 0.0;
        for (std::size_t i = ib; i < ie; ++i) {
            if (type[i] == kSolid) continue;
            squares += static_cast<double>(q[i]) * static_cast<double>(q[i]);
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
            // (1/3) q + (2/3) (t2 + D t2), written from q so that a uniform pressure
            // stays to the bit: those two weights add to 1 + 2^-25 in single precision,
            // which let a closed room's mean grow.
            q[i] = q[i] + (2.0f / 3.0f) * ((t2[i] + carried(t2, i)) - q[i]);
        }
        barrier.wait();
        return squares;
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
        return centre_at(static_cast<long long>(i % nx_), static_cast<long long>((i / nx_) % ny_),
                         static_cast<long long>(i / sz_));
    }

    // The centre of the cell at grid coordinates x, y, z.
    Vec3 centre_at(long long x, long long y, long long z) const {
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

    // Of each cell, kNearFace when some face passes within the cube of half-width one cell
    // round its centre, which holds the links of the + half of every stencil: the cells
    // within its bounding box, grown by a cell, that are no further from its plane than
    // the cube's corners; and kNearPlane too when that face's plane passes within twice
    // the reach of crossing's short casts (see blind) of the centre, as a face the centre
    // lies on does (see kTie). Both for every cell when the faces' boxes hold too many
    // cells to visit.
    static constexpr std::uint8_t kNearFace = 1, kNearPlane = 2;

    std::vector<std::uint8_t> near_faces(const Bvh &bvh) const {
        const std::size_t n = nx_ * ny_ * nz_;
        const double reach = cell_ * (1.0 + 1e-9), radius = std::sqrt(3.0) * reach;
        const double tie = 2.0 * blind(bvh);
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
            if (visits > 8 * n) return std::vector<std::uint8_t>(n, kNearFace | kNearPlane);
            const Vec3 &normal = bvh.normal(f);
            for (std::size_t z = from[2]; z <= to[2]; ++z) {
                for (std::size_t y = from[1]; y <= to[1]; ++y) {
                    for (std::size_t x = from[0]; x <= to[0]; ++x) {
                        const std::size_t i = x + y * sy_ + z * sz_;
                        const double off = std::abs(dot(normal, centre(i) - p[0]));
                        if (off <= radius) {
                            near[i] |= off <= tie ? kNearFace | kNearPlane : kNearFace;
                        }
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
    //
    // p, on faces, takes `lean` (see kTie), and its place among the centres (see grid_place).
    std::vector<std::pair<std::size_t, double>> reading_cells(const Bvh &bvh,
                                                              const std::vector<std::uint16_t> &cut,
                                                              const Vec3 &p, bool cubic,
                                                              const Vec3 &lean = {}) const {
        if (!cubic) return visible_cells(bvh, p, &type_, lean);
        const GridPlace g = grid_place(bvh, p, lean);
        const double *q = g.q;
        const std::size_t size[3] = {nx_, ny_, nz_};
        long long below[3], start[3];  // the centre at or below p (on p's side of a tie)
        double weight[3][kReach];
        for (int a = 0; a < 3; ++a) {
            // The box lies in the grid, and the two cells round p inside its edge cells
            // (as visible_cells has them).
            below[a] = static_cast<long long>(std::floor(q[a]));
            if (g.tied[a] && g.side[a].sign < 0) --below[a];
            if (!(q[a] >= 1.0 && q[a] < static_cast<double>(size[a]) - 3.0 && below[a] >= 1)) {
                return visible_cells(bvh, p, &type_, lean);
            }
            start[a] = below[a] - (kReach / 2 - 1);
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
                    if (seen) seen = crossing(bvh, End{p, lean}, End{centre(i)}) < 0;
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
                const double t = q[a] - static_cast<double>(below[a]);
                weight[a][inner] = 1.0 - t;
                weight[a][inner + 1] = t;
            } else if (keep[a][inner] || keep[a][inner + 1]) {
                from[a] = keep[a][inner] ? inner : inner + 1;
                count[a] = 1;
                weight[a][from[a]] = 1.0;
            } else {
                return visible_cells(bvh, p, &type_, lean);
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
                        return visible_cells(bvh, p, &type_, lean);
                    }
                    const double w = weight[0][x] * weight[1][y] * weight[2][z];
                    if (w != 0.0) out.emplace_back(i, w);
                }
            }
        }
        return out;
    }

    // Ties: a point within kTie cells of a face's plane, its foot on the face (a cell
    // centre, the source or a probe), lies on the face. It then counts as lying an
    // infinitesimal step off it: along its lean, a direction the source and the probes
    // may have (see leans; none for a cell centre), then a far smaller step along +x, a
    // smaller one still along +y and one along +z. The first of those steps not parallel
    // to the face (a lean whose part along the face's normal is at most 1e-9, an axis
    // along which the normal's part is at most 1e-12, counts as parallel) says which side
    // of the face the point lies on, and all of them together where on the face it lies:
    // inside it, or past one of its edges (see crossing). Without a lean this is the
    // rule: a point on a face lies on the side of it that x grows into (else y, else z),
    // the same for every face whichever way its normal points, as if every face lay an
    // infinitesimal step further along -x (then -y, then -z) than it does. So a wall
    // through a plane of centres acts as one just below it on every link, every sight
    // line and either stencil, and a cell on the wall's plane is air on the side above
    // it. Only the source and the probes take leans (see place_on_face and the probes in
    // the constructor).
    static constexpr double kTie = 1e-10;

    // One end of a segment for crossing: the point; its lean (see kTie; zero for none, a
    // unit vector otherwise); whether a face may pass through it; and whether a face may
    // pass within the reach of crossing's short casts of it (see blind). Each false only
    // where that is known not to be so, which spares the search for those faces.
    struct End {
        Vec3 at;
        Vec3 lean{};
        bool ties = true;
        bool close = true;
    };

    // How far from an end crossing looks for faces the cast along a segment cannot
    // count (see crossing): its min_distance, and kTie cells more.
    double blind(const Bvh &bvh) const { return bvh.min_distance() + kTie * cell_; }

    // Of a cell centre (or any point) p: bit 0, whether it lies on a face (see kTie); bit
    // 1, whether a face's plane passes within blind of it, its leaf of the tree's box
    // grown that much holding p (each_face_near), as every face crossing's short casts
    // from p meet does.
    std::uint8_t ends_at(const Bvh &bvh, const Vec3 &p) const {
        const double tie = kTie * cell_;
        std::uint8_t out = 0;
        bvh.each_face_near(p, blind(bvh) + tie, [&](std::size_t f, double offset) {
            out |= 2;
            if (std::abs(offset) <= tie && holds(bvh, f, p)) out |= 1;
        });
        return out;
    }

    // Whether a face lies within blind of p: its plane within blind and kTie cells, and its
    // foot on that plane within as much of the face.
    bool within_blind(const Bvh &bvh, const Vec3 &p) const {
        const double reach = blind(bvh) + kTie * cell_;
        bool any = false;
        bvh.each_face_near(
            p, reach, [&](std::size_t f, double) { any = any || bvh.holds_foot(f, p, reach); });
        return any;
    }

    // The side of a face with unit normal n that a point on its plane lies on with
    // `lean` (see kTie): the sign of an offset from the plane, +1 or -1, and the step that
    // puts it there: 0 its lean, 1 to 3 the steps along x, y and z.
    struct Side {
        int step;
        int sign;
    };

    static Side side_of(const Vec3 &n, const Vec3 &lean) {
        const double along = dot(n, lean);
        if (std::abs(along) > 1e-9) return {0, along > 0.0 ? 1 : -1};
        if (std::abs(n.x) > 1e-12) return {1, n.x > 0.0 ? 1 : -1};
        if (std::abs(n.y) > 1e-12) return {2, n.y > 0.0 ? 1 : -1};
        return {3, n.z > 0.0 ? 1 : -1};
    }

    // Whether p lies on a face (see kTie).
    bool on_face(const Bvh &bvh, const Vec3 &p) const {
        bool any = false;
        each_face_through(bvh, p, [&](std::size_t) { any = true; });
        return any;
    }

    // Calls visit(face) for every face p lies on (see kTie), in no particular order.
    template <typename Visit>
    void each_face_through(const Bvh &bvh, const Vec3 &p, Visit visit) const {
        bvh.each_face_near(p, kTie * cell_, [&](std::size_t f, double) {
            if (holds(bvh, f, p)) visit(f);
        });
    }

    // Whether face f, whose plane passes within kTie cells of p, holds p's foot: past none
    // of its edges (see edges).
    bool holds(const Bvh &bvh, std::size_t f, const Vec3 &p) const {
        return Bvh::holds_foot(edges(bvh, f, p, [&](std::size_t g) { return bvh.offset(g, p); }));
    }

    // Whether p lies on a face (see kTie) of face f's plane.
    bool on_plane(const Bvh &bvh, std::size_t f, const Vec3 &p) const {
        bool on = false;
        each_face_through(bvh, p, [&](std::size_t g) {
            on = on || !(norm(cross(bvh.normal(f), bvh.normal(g))) > 1e-9);
        });
        return on;
    }

    // Whether the segment from a to b meets face g's plane within kTie cells of face f's
    // plane and f's within kTie cells of g's: |D| / |fa - fb| and |D| / |ga - gb| at most
    // kTie, D = fa gb - ga fb from the ends' offsets fa, fb from f's plane and ga, gb from
    // g's, an offset within kTie cells counting as 0 (the segment meets that plane at that
    // end, as the test of a point on a face has it). The same numbers, to the bit, for
    // (f, g) and (g, f): so where a segment passes through the edge two faces share, both
    // read it on the edge or neither, whatever the angle between the segment and each
    // face; and a segment from an end on g's plane, more than kTie cells from f's, is on
    // the edge at neither, as that end is on g's face by the test of a point (see edges).
    bool tied_pair(const Bvh &bvh, std::size_t f, std::size_t g, const Vec3 &a,
                   const Vec3 &b) const {
        const double tie = kTie * cell_;
        const auto level = [tie](double s) { return std::abs(s) <= tie ? 0.0 : s; };
        const double fa = level(bvh.offset(f, a)), fb = level(bvh.offset(f, b));
        const double ga = level(bvh.offset(g, a)), gb = level(bvh.offset(g, b));
        const double d = std::abs(fa * gb - ga * fb);
        return d <= tie * std::min(std::abs(fa - fb), std::abs(ga - gb));
    }

    // The side of face g's plane both ends of a segment lie on by the rule, where one of
    // them lies on a face of that plane (its step's side, see kTie; the other's by its
    // offset, or its step's too): +1 above, -1 below; 0 where they lie on different
    // sides, or neither lies on the plane. So a segment from a cell on a box's floor to a
    // cell past its side wall, which meets the side wall on its edge with the floor,
    // meets it above the floor, as its end on the floor lies above it.
    int side_of_segment(const Bvh &bvh, std::size_t g, const End &a, const End &b) const {
        const double tie = kTie * cell_;
        // A segment in g's plane (both ends within kTie cells of it): both ends by their
        // steps, on a face or not (as crossing has them).
        const bool in_plane =
            std::abs(bvh.offset(g, a.at)) <= tie && std::abs(bvh.offset(g, b.at)) <= tie;
        bool tied = false;
        const auto rule = [&](const End &e) {
            const double s = bvh.offset(g, e.at);
            if (s == 0.0 ||
                (std::abs(s) <= tie && (in_plane || (e.ties && on_plane(bvh, g, e.at))))) {
                tied = true;
                return side_of(bvh.normal(g), e.lean).sign;
            }
            return s > 0.0 ? 1 : -1;
        };
        const int ra = rule(a), rb = rule(b);
        return tied && ra == rb ? ra : 0;
    }

    // Of p, a point of face f's plane: its coordinates along the plane, and on which of the
    // face's edges it lies (see Bvh::EdgeCoordinates). offset(g) is the distance from face
    // g's plane the caller has for p (for a point where a segment meets f's plane, from the
    // segment's ends, see crossing); side(g), given for such a point, the side of g's plane
    // both ends lie on by the rule where one lies on it (see side_of_segment), or 0; and
    // close(g), given for a segment that crosses f's plane between its ends (both more
    // than kTie cells off it: a hit of the segment's cast, or a meet holds_meet reads), whether
    // the segment meets g's plane within kTie cells of f's and f's within kTie cells of
    // g's (one test for both faces, see tied_pair), in place of offset(g).
    // p lies on an edge within kTie cells of its line, and only there; but within kEdge
    // (barycentric) of the edge, where a ray's hit widens the face (see Bvh::hit_triangle),
    // and more than kTie / 2 cells from the line (any distance for a segment's meet with
    // side given, or with close given where another face continues the edge), the other
    // faces decide:
    // - Where another face continues the edge at an angle (not in f's plane, its plane
    //   holding the edge's ends, its foot holding p: the next face of a closed object), p
    //   lies on the edge where it lies on that face's plane (within kTie cells by
    //   offset(g), or close(g): so a segment through the edge is a tie at both faces or
    //   at neither, whatever its angle to them and the rounding where it is kTie), and not
    //   where it does not, however close to the line. So a link along the next face's
    //   plane meets the edge where its ends lie on that plane by the test of a point on a
    //   face (see kTie), and on the side of the edge the ends lie on where they do not: the
    //   two tests are one. A segment both of whose ends lie on one side of that plane by
    //   the rule (side(g)) meets f off the edge, on that side of it.
    // - Otherwise (an edge between two faces of one plane, or a screen's rim) the edge is
    //   read at p moved onto f's plane and onto the planes within kTie cells of p (by
    //   offset(g), or close(g)) of the faces through an end of the edge (not in f's plane,
    //   their foot holding p): near a box's corner, p tied to both planes of the corner's other
    //   faces lies on the corner, so on the diagonal of the box's face through it too. p lies on a
    //   rim within kTie cells of its line, or within kTie / 2 there; on an edge between two faces
    //   of one plane, which bounds no surface, only within kTie / 2 cells there, and on one side of
    //   it otherwise, as both faces have it.
    template <typename Offset, typename Side = std::nullptr_t, typename Close = std::nullptr_t>
    Bvh::EdgeCoordinates edges(const Bvh &bvh, std::size_t f, const Vec3 &p, Offset offset,
                               Side side = nullptr, Close close = nullptr) const {
        constexpr bool segment = !std::is_same_v<Side, std::nullptr_t>;
        const double tie = kTie * cell_;
        Bvh::EdgeCoordinates c = bvh.edges(f, p, tie);
        if (!c.area) return c;
        const Vec3 &n = bvh.normal(f);
        const std::array<Vec3, 3> corner = bvh.corners(f);
        bool continued[3] = {false, false, false};
        constexpr bool paired = !std::is_same_v<Close, std::nullptr_t>;
        for (int e = 0; e < 3; ++e) {
            if (!c.near(e)) continue;
            const double distance = c.distance(e);
            const bool line = !(distance > 0.5 * tie);
            if (line && !segment && !paired) continue;
            const std::array<Vec3, 2> ends = bvh.edge_ends(f, e);
            const Vec3 &opposite = corner[e == 0 ? 1 : (e == 1 ? 2 : 0)];
            bool next = false, on = false;
            int sided = 0;  // +1: the segment's ends on f's side of g's plane, -1: on the other
            bvh.each_face_near(p, distance + tie, [&](std::size_t g, double) {
                if (g == f || (next && (on || sided != 0))) return;
                if (!(norm(cross(n, bvh.normal(g))) > 1e-9)) return;  // in f's plane
                if (!(std::abs(bvh.offset(g, ends[0])) <= tie &&
                      std::abs(bvh.offset(g, ends[1])) <= tie)) {
                    return;
                }
                if (!bvh.holds_foot(g, p, distance + tie)) return;
                next = true;
                int s = 0;
                bool near = std::abs(offset(g)) <= tie;
                if constexpr (segment) s = side(g);
                if constexpr (paired) near = close(g);
                if (s != 0 && near) {
                    sided = (s > 0) == (bvh.offset(g, opposite) > 0.0) ? 1 : -1;
                } else {
                    on = on || near;
                }
            });
            continued[e] = next;
            if (sided != 0) {  // off the edge, on the side of it the ends' sides give
                c.on[e] = false;
                const double at =
                    std::max(std::abs(c.at[e]), std::numeric_limits<double>::denorm_min());
                c.at[e] = sided > 0 ? at : -at;
            } else if (next && (!line || paired)) {
                c.on[e] = on;
            }
        }
        for (int e = 0; e < 3; ++e) {
            if (!c.near(e) || continued[e]) continue;
            const double distance = c.distance(e);
            if (!(distance > 0.5 * tie)) continue;
            const std::array<Vec3, 2> ends = bvh.edge_ends(f, e);
            // The least move of p onto f's plane and the tied planes: an orthonormal basis
            // q of their normals, and the move's parts along it.
            Vec3 q[3];
            double part[3];
            std::size_t planes = 0;
            const auto onto = [&](const Vec3 &m, double s) {
                Vec3 v = m;
                double r = -s;
                for (std::size_t k = 0; k < planes; ++k) {
                    const double along = dot(m, q[k]);
                    v = v - q[k] * along;
                    r -= part[k] * along;
                }
                const double length = norm(v);
                if (planes == 3 || !(length > 1e-6)) return;  // in the planes' span
                q[planes] = v / length;
                part[planes++] = r / length;
            };
            onto(n, bvh.offset(f, p));
            bool tied = false, inner = false;
            const Vec3 middle = (ends[0] + ends[1]) * 0.5;
            bvh.each_face_near(p, distance + tie, [&](std::size_t g, double) {
                if (g == f) return;
                if (!(norm(cross(n, bvh.normal(g))) > 1e-9)) {  // in f's plane: across the edge?
                    inner = inner || (std::abs(bvh.offset(g, middle)) <= tie &&
                                      bvh.holds_foot(g, middle, tie));
                    return;
                }
                const double s = offset(g);
                if constexpr (paired) {
                    if (!close(g)) return;
                } else if (!(std::abs(s) <= tie)) {
                    return;
                }
                if (!(std::abs(bvh.offset(g, ends[0])) <= tie ||
                      std::abs(bvh.offset(g, ends[1])) <= tie))
                    return;
                if (!bvh.holds_foot(g, p, distance + tie)) return;
                tied = true;
                onto(bvh.normal(g), s);
            });
            Vec3 move{};
            for (std::size_t k = 0; k < planes; ++k) move = move + q[k] * part[k];
            if (tied) c.at[e] += dot(c.gradient[e], move);
            double at = std::abs(c.at[e]) / c.steep(e);
            if (!tied && !inner) {
                // A rim: p's distance from its line across f's plane, from the line's own
                // corners (not p's barycentric coordinates, whose rounding varies with p
                // along the line): so the links of a row of centres along a rim kTie cells
                // from it are all cut or all not, as for a face's plane.
                const Vec3 across = cross(n, ends[1] - ends[0]);
                at = std::abs(dot(across, p - ends[0])) / norm(across);
                c.on[e] = at <= tie;
            }
            c.on[e] = inner ? at <= 0.5 * tie : (c.on[e] || at <= 0.5 * tie);
        }
        return c;
    }

    // The leans a point p on faces takes in turn: for p on faces of one plane and on edges
    // of them that no face of the plane continues past (a screen's free rim or corner),
    // first the sum, normalised, of the directions along the plane away from those edges;
    // then for each choice of sides of the planes of the faces through p (of the first
    // eight planes; the rule's sides first, then fewest sides changed from the rule's),
    // the sum of those planes' unit normals towards the chosen sides, normalised, where
    // it puts p on the chosen side of every one of those faces (no lean, for the rule's
    // sides, where that sum does not); then the 26 directions from a cell to its
    // neighbours, which reach choices the sums miss. A point in a room's corner, on a
    // pillar's edge or between a slab's faces so has a lean into each part of the space
    // round it that the faces through it bound, and on an edge it leans away from both
    // faces, as a point just off the edge sees the cells the edge's point itself sees
    // past the faces (with no lean, the rule's step along x would put it over one face,
    // which hides cells past the other).
    struct Leans {
        std::vector<Vec3> all;
        bool edge = false;  // p lies on faces of two planes or more
    };

    Leans leans(const Bvh &bvh, const Vec3 &p) const {
        std::vector<std::size_t> faces;
        each_face_through(bvh, p, [&](std::size_t f) { faces.push_back(f); });
        std::sort(faces.begin(), faces.end());
        std::vector<Vec3> planes;  // each plane's normal towards the side the rule gives
        std::vector<std::size_t> plane_of;
        for (const std::size_t f : faces) {
            const Vec3 &n = bvh.normal(f);
            const Vec3 m = side_of(n, {}).sign > 0 ? n : -n;
            std::size_t j = 0;
            while (j < planes.size() && dot(planes[j], m) < 1.0 - 1e-9) ++j;
            if (j == planes.size()) planes.push_back(m);
            plane_of.push_back(j);
        }
        Leans leans;
        leans.edge = planes.size() > 1;
        std::vector<Vec3> &out = leans.all;
        // On faces of one plane, on an edge no face of that plane continues past (a screen's
        // free rim or corner): first, along the plane away from the faces, into the middle
        // of the air round the edge (the sum of those edges' directions; its side of the
        // plane is the rule's), as a point just past the edge sees round it.
        if (planes.size() == 1) {
            const auto at_point = [&](std::size_t f) {
                return edges(bvh, f, p, [&](std::size_t g) { return bvh.offset(g, p); });
            };
            std::vector<Vec3> free;
            for (const std::size_t f : faces) {
                Vec3 away[3];
                const std::size_t count = Bvh::edges_out(at_point(f), away);
                for (std::size_t e = 0; e < count; ++e) {
                    bool continued = false;
                    for (const std::size_t g : faces) {
                        continued =
                            continued || (g != f && Bvh::holds_towards(at_point(g), &away[e], 1));
                    }
                    bool known = false;
                    for (const Vec3 &d : free) known = known || dot(d, away[e]) > 1.0 - 1e-9;
                    if (!continued && !known) free.push_back(away[e]);
                }
            }
            Vec3 sum{};
            for (const Vec3 &d : free) sum = sum + d;
            if (norm(sum) > 1e-9) out.push_back(sum / norm(sum));
        }
        const std::size_t choices = std::min<std::size_t>(planes.size(), 8);
        for (std::size_t changed = 0; changed <= choices; ++changed) {
            if (changed == 1 && out.empty()) out.push_back(Vec3{});  // the rule's own steps
            for (unsigned mask = 0; mask < (1u << choices); ++mask) {
                std::size_t count = 0;
                for (unsigned m = mask; m != 0; m &= m - 1) ++count;
                if (count != changed) continue;
                const auto flipped = [&](std::size_t j) {
                    return j < choices && ((mask >> j) & 1u) != 0;
                };
                Vec3 sum{};
                for (std::size_t j = 0; j < planes.size(); ++j) {
                    sum = sum + (flipped(j) ? -planes[j] : planes[j]);
                }
                const double length = norm(sum);
                if (!(length > 1e-9)) continue;
                const Vec3 lean = sum / length;
                bool fits = true;
                for (std::size_t k = 0; k < faces.size() && fits; ++k) {
                    const Vec3 &n = bvh.normal(faces[k]);
                    const int rule = side_of(n, {}).sign;
                    fits = side_of(n, lean).sign == (flipped(plane_of[k]) ? -rule : rule);
                }
                if (fits) out.push_back(lean);
            }
        }
        if (out.empty()) out.push_back(Vec3{});
        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    if (x == 0 && y == 0 && z == 0) continue;
                    out.push_back(normalized(Vec3{static_cast<double>(x), static_cast<double>(y),
                                                  static_cast<double>(z)}));
                }
            }
        }
        return leans;
    }

    // Which of 64 directions spread evenly round p (bit k for the k-th) p sees with
    // `lean`: towards the point 1e-7 cells off p along each (see widest).
    std::uint64_t view(const Bvh &bvh, const Vec3 &p, const Vec3 &lean) const {
        constexpr int kCount = 64;
        const double golden = 3.141592653589793 * (3.0 - std::sqrt(5.0));
        std::uint64_t seen = 0;
        for (int k = 0; k < kCount; ++k) {
            const double z = 1.0 - (2.0 * k + 1.0) / kCount, r = std::sqrt(1.0 - z * z);
            const Vec3 d{r * std::cos(golden * k), r * std::sin(golden * k), z};
            if (crossing(bvh, End{p, lean}, End{p + d * (1e-7 * cell_)}) < 0) {
                seen |= std::uint64_t{1} << k;
            }
        }
        return seen;
    }

    // Of `leans` of p, the first for which usable(k) holds, k its index in leans.all; for
    // p on faces of two planes or more (an edge or a corner), then in turn each later
    // usable one that sees every direction round p the one before sees and more (see
    // view): a point on a pillar's edge leans into the middle of the air round the edge,
    // and sees the cells the edge's point itself sees past both faces, where leaning
    // across one face's plane it sees fewer. Whether one is usable, and the lean.
    template <typename Usable>
    std::pair<bool, Vec3> widest(const Bvh &bvh, const Vec3 &p, const Leans &leans,
                                 Usable usable) const {
        const Vec3 *chosen = nullptr;
        std::uint64_t seen = 0;
        for (std::size_t k = 0; k < leans.all.size(); ++k) {
            const Vec3 &lean = leans.all[k];
            if (chosen && !leans.edge) break;
            if (!usable(k)) continue;
            if (!chosen) {
                chosen = &lean;
                if (leans.edge) seen = view(bvh, p, lean);
                continue;
            }
            const std::uint64_t more = view(bvh, p, lean);
            if ((more & seen) == seen && more != seen) {
                chosen = &lean;
                seen = more;
            }
        }
        return {chosen != nullptr, chosen ? *chosen : Vec3{}};
    }

    // Whether face f holds the point where the segment from a to b meets its plane (see
    // kTie): each end moved by its lean times a small e, then both by the rule's steps
    // along x, y and z, which go to 0 first, then e (as for the same points moved off by
    // a hair, e, the rule then deciding only for the cell centres). sa and sb are the
    // ends' offsets from the plane, 0 for an end on the face; `meet` the point the
    // unmoved segment meets the plane at (computed here if null). The meet point moves
    // along the plane by, in order of size: the terms in e and e^2 of its move with the
    // ends; then, for the step along x, the terms in 1/e, 1, e, e^2 of its move with that
    // step (where the segment lies in the plane with one end on either side, by their
    // leans or one by its lean and the other by that step, the term in 1/e runs along the
    // segment); then those of y and z. The first that leaves the line of an edge the meet
    // lies on says whether it goes into the face (holds_towards). An end with offset 0 (sa
    // or sb, an end crossing has on the plane) is where the segment meets it, to the bit.
    bool holds_meet(const Bvh &bvh, std::size_t f, const Vec3 *meet, const End &a, double sa,
                    const End &b, double sb) const {
        // The meet's coordinates on the face, its offsets from other faces' planes from the
        // ends' (see edges): lambda1, where on the segment it lies. Where neither end lies
        // within kTie cells of f's plane (a segment that crosses it between its ends), its
        // tie with an edge another face continues is read by the test of both faces
        // (tied_pair), as for a hit of the segment's cast.
        const double tie = kTie * cell_;
        const bool between = std::abs(sa) > tie && std::abs(sb) > tie;
        const auto coordinates = [&](const Vec3 &at, double lambda1) {
            const auto offset = [&](std::size_t g) {
                // (At an end, that end's own offset to the bit, as the test of a point has it.)
                const double ga = bvh.offset(g, a.at), gb = bvh.offset(g, b.at);
                return lambda1 == 1.0 ? gb : ga + lambda1 * (gb - ga);
            };
            const auto side = [&](std::size_t g) { return side_of_segment(bvh, g, a, b); };
            if (between) {
                return edges(bvh, f, at, offset, side,
                             [&](std::size_t g) { return tied_pair(bvh, f, g, a.at, b.at); });
            }
            return edges(bvh, f, at, offset, side);
        };
        // A meet inside the face, off its edges, is held whatever the steps.
        if (meet && sb != sa && Bvh::holds_inside(coordinates(*meet, sa / (sa - sb)))) return true;
        const Vec3 &n = bvh.normal(f);
        // Series in e, its terms in e^-1, 1, e and e^2 at [0] to [3].
        const auto along = [&](const Vec3 &lean) {
            const double part = dot(n, lean);
            return std::abs(part) > 1e-9 ? part : 0.0;  // as side_of counts it
        };
        const double l1 = along(b.lean) - along(a.lean);
        const auto times = [](const double *x, const double *y, double *out) {
            for (int p = 0; p < 4; ++p) out[p] = 0.0;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4 && i + j - 1 < 4; ++j) {
                    if (i + j - 1 >= 0) out[i + j - 1] += x[i] * y[j];
                }
            }
        };
        const Vec3 spread = b.lean - a.lean;
        const Vec3 axes[3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        // The meet's moves (steps, kSteps of them) for the ends at pa and pb, qa and qb off
        // the plane, and where on the segment it lies (lambda); false for a segment
        // parallel to the plane.
        // The planes both ends lie on (faces of them, see kTie): across them the ends' places
        // are the rule's (on the plane, then their steps), not their coordinates', so the
        // segment's part across them (rounding, or less than kTie cells) is dropped: two
        // points tied to a box's corner are one point, and the meet moves by their leans
        // and steps alone.
        Vec3 across[3];
        std::size_t planes = 0;
        if (a.ties && b.ties) {
            each_face_through(bvh, b.at, [&](std::size_t g) {
                if (planes == 3 || !(std::abs(bvh.offset(g, a.at)) <= tie) ||
                    !on_plane(bvh, g, a.at))
                    return;
                Vec3 m = bvh.normal(g);
                for (std::size_t k = 0; k < planes; ++k) m = m - across[k] * dot(m, across[k]);
                if (norm(m) > 1e-6) across[planes++] = m / norm(m);
            });
        }
        const auto flat = [&](Vec3 v) {
            for (std::size_t k = 0; k < planes; ++k) v = v - across[k] * dot(v, across[k]);
            return v;
        };
        constexpr std::size_t kSteps = 14;
        const auto series = [&](const Vec3 &pa, double qa, const Vec3 &pb, double qb, Vec3 *steps,
                                double *lambda) {
            const double l0 = qb - qa;
            double inv[4] = {0.0, 0.0, 0.0, 0.0};  // 1 / (the offset of b' from a' along n)
            if (l0 != 0.0) {
                inv[1] = 1.0 / l0;
                inv[2] = -l1 / (l0 * l0);
                inv[3] = l1 * l1 / (l0 * l0 * l0);
            } else if (l1 != 0.0) {
                inv[0] = 1.0 / l1;
            } else {
                return false;  // parallel to the plane: no meet
            }
            const double from_a[4] = {0.0, -qa, -along(a.lean), 0.0};
            times(from_a, inv, lambda);
            const Vec3 span = flat(pb - pa);
            // Each term, each of its components 0 where it is a sum that cancels to within
            // rounding of its parts (a segment along the lean: the meet stays put; a meet
            // that runs along an edge while it moves fast along another axis), which no
            // direction must be read from.
            std::size_t count = 0;
            const auto put = [&](const Vec3 &x, const Vec3 &y, const Vec3 &z) {
                const auto part = [](double a, double b, double c) {
                    const double sum = a + b + c;
                    return std::abs(sum) > 1e-9 * (std::abs(a) + std::abs(b) + std::abs(c)) ? sum
                                                                                            : 0.0;
                };
                steps[count++] =
                    Vec3{part(x.x, y.x, z.x), part(x.y, y.y, z.y), part(x.z, y.z, z.z)};
            };
            // The meet's move with the ends: a.lean e + lambda (span + spread e), its terms
            // in e and e^2.
            put(a.lean, span * lambda[2], spread * lambda[1]);
            put(span * lambda[3], spread * lambda[2], Vec3{});
            // Its move with a step d of both ends: d - (span + spread e) (n.d) inv.
            for (const Vec3 &d : axes) {
                const double part = dot(n, d);
                if (!(std::abs(part) > 1e-12)) {
                    steps[count++] = d;
                    continue;
                }
                put(span * (-part * inv[0]), Vec3{}, Vec3{});
                put(d, span * (-part * inv[1]), spread * (-part * inv[0]));
                put(span * (-part * inv[2]), spread * (-part * inv[1]), Vec3{});
                put(span * (-part * inv[3]), spread * (-part * inv[2]), Vec3{});
            }
            return true;
        };
        Vec3 steps[kSteps];
        double lambda[4];  // where on the segment the meet lies, from a
        if (!series(a.at, sa, b.at, sb, steps, lambda)) return false;
        // A meet at an end on the plane (offset 0) is that end, to the bit: its coordinates and
        // offsets then are the end's own, as the test of a point has them.
        if (sb == 0.0 && sa != 0.0) lambda[1] = 1.0;
        const Vec3 at = meet ? *meet
                             : (lambda[1] == 1.0 ? b.at : a.at + (b.at - a.at) * lambda[1]) +
                                   spread * lambda[0];
        // The same with either end moved along an axis by 16 times the rounding of the
        // largest coordinate of the ends (an end on the plane along it only): a step those
        // moves can turn round is decided by the rounding of the ends' places (the inputs'
        // coordinates, a triangle's corners and the cells' centres each lie within an ulp
        // or two of their places), not by the ends, and is taken as along the edges (see
        // Bvh::holds_towards). So the side of an edge a segment that passes through it
        // passes is the same on both faces that share the edge.
        Vec3 moved[12 * kSteps];
        std::size_t variants = 0;
        const auto largest = [](const Vec3 &v) {
            return std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)});
        };
        const double wiggle = 16.0 * std::numeric_limits<double>::epsilon() *
                              std::max({largest(a.at), largest(b.at), cell_});
        for (int end = 0; end < 2; ++end) {
            for (const Vec3 &d : axes) {
                for (const double sign : {-1.0, 1.0}) {
                    const double q = end == 0 ? sa : sb;
                    const Vec3 step = (q == 0.0 ? d - n * dot(n, d) : d) * (sign * wiggle);
                    const double moved_q = q == 0.0 ? 0.0 : q + dot(n, step);
                    double unused[4];
                    const bool ok = end == 0 ? series(a.at + step, moved_q, b.at, sb,
                                                      moved + variants * kSteps, unused)
                                             : series(a.at, sa, b.at + step, moved_q,
                                                      moved + variants * kSteps, unused);
                    if (ok) ++variants;
                }
            }
        }
        return Bvh::holds_towards(coordinates(at, lambda[1]), steps, kSteps, moved, variants);
    }

    // The face nearest a that the segment from a to b crosses, or -1. A face is crossed
    // when a and b lie on different sides of it and the segment meets it, an end on a
    // face lying an infinitesimal step off it (see kTie). So a segment in a face's plane
    // crosses it only from a point on the face to a point on its plane on the other side,
    // and a segment from a point on a face's edge crosses the face only where it leaves
    // that point into the face. A segment of length zero (a source or a probe at a cell
    // centre) crosses a face through its point where its two ends lie on different sides
    // of it (see kTie).
    std::int64_t crossing(const Bvh &bvh, const End &a, const End &b) const {
        const Vec3 to = b.at - a.at;
        const double distance = norm(to);
        if (!(distance > 0.0)) return crossing(bvh, a, b, Vec3{}, 0.0);
        return crossing(bvh, a, b, to / distance, distance);
    }

    // The same, given the direction from a to b and their distance (a link's; for a
    // distance of zero, any direction).
    std::int64_t crossing(const Bvh &bvh, const End &a, const End &b, const Vec3 &dir,
                          double distance) const {
        const double tie = kTie * cell_;
        // A face the segment crosses at t from a, the n-th found; of several, the nearest,
        // and of those at the same t the first found.
        struct Candidate {
            std::int64_t face = -1;
            double t = 0.0;
            std::size_t n = 0;
            bool before(const Candidate &o) const {
                return face >= 0 && (o.face < 0 || t < o.t || (t == o.t && n < o.n));
            }
        };
        std::size_t found = 0;
        const auto offer = [&](Candidate &c, std::size_t f, double t) {
            const Candidate x{static_cast<std::int64_t>(f), t, found++};
            if (x.before(c)) c = x;
        };
        // The nearest face the tie rule says the segment crosses; of the segment's cast's
        // hits that no tie touches (see meets), the nearest (loose) and the nearest whose meet
        // lies inside its face (held); and whether a tie touched any face (tied).
        Candidate ruled, loose, held;
        bool tied = false;
        const auto take = [&](std::size_t f, double t) { offer(ruled, f, t); };
        // Whether face f, whose plane lies `offset` from end e, passes through e.
        const auto through_end = [&](const End &e, std::size_t f, double offset) {
            return e.ties && std::abs(offset) <= tie && holds(bvh, f, e.at);
        };
        // Faces through neither end: those the segment meets between its ends. The
        // segment's cast (Bvh::each_hit from a, hits in (min_distance, the segment's
        // length)) counts them, with their distances from a.
        // A hit of it with an end within kTie cells of the face's plane, or its meet within
        // kTie cells of the line of one of the face's edges, is a tie: the rule decides it.
        // The cast cannot count the faces within its min_distance of a, or at (or by
        // rounding, past) b. Short casts over blind each side of an end close to faces (see
        // End) add those, each face once, as ties: whatever their angle to the segment, the
        // segment's ends lie on different sides of them (a face whose plane passes within
        // kTie of an end outside the face too: its plane meets the segment's line on the
        // segment). A segment of length zero meets none.
        // A face the segment's line meets at t from a; `cast` for a hit of the segment's cast.
        const auto meets = [&](std::size_t f, double t, bool cast) {
            const double sa = bvh.offset(f, a.at), sb = bvh.offset(f, b.at);
            const Vec3 meet = a.at + dir * t;
            // A hit of the segment's cast with neither end within kTie cells of the face's plane
            // and the meet on none of its edges (see Bvh::near_edge): no tie touches it.
            // (On an edge another face continues, by the test of both faces, tied_pair: so
            // a segment through that edge is a tie at both faces or at neither.)
            if (cast && std::abs(sa) > tie && std::abs(sb) > tie) {
                const Bvh::EdgeCoordinates c = edges(
                    bvh, f, meet,
                    [&](std::size_t g) {
                        const double ga = bvh.offset(g, a.at), gb = bvh.offset(g, b.at);
                        return ga + t / distance * (gb - ga);
                    },
                    nullptr, [&](std::size_t g) { return tied_pair(bvh, f, g, a.at, b.at); });
                if (!Bvh::near_edge(c)) {
                    offer(loose, f, t);
                    if (Bvh::holds_foot(c)) offer(held, f, t);
                    return;
                }
            }
            tied = true;
            if (through_end(a, f, sa) || through_end(b, f, sb)) return;  // below
            const Vec3 &n = bvh.normal(f);
            // An end within kTie cells of f's plane lies on it, on the side its step gives,
            // where the segment meets the plane within kTie cells of it, or lies in the
            // plane (both ends within kTie cells of it, as through() has the other end):
            // the segment meets the plane there. (An end the segment leaves at a slant
            // keeps its offset: a plate's plane 0.5e-10 cells above a centre past its edge,
            // at a slope of 1e-5, meets the centre's link 5e-6 cells off, on the plate.)
            // on_plane_face: the end lies on another face of that plane (a face's other
            // triangle).
            const bool in_plane = std::abs(sa) <= tie && std::abs(sb) <= tie;
            bool on_plane_face[2] = {false, false};
            const auto level = [&](const End &e, double s, bool &on) {
                if (!(std::abs(s) <= tie)) return s;
                if (!in_plane && !(std::abs(s) * distance <= tie * std::abs(sa - sb))) return s;
                if (e.ties) {
                    each_face_through(bvh, e.at, [&](std::size_t g) {
                        on = on || (g != f && !(norm(cross(n, bvh.normal(g))) > 1e-9));
                    });
                }
                return 0.0;
            };
            const double qa = level(a, sa, on_plane_face[0]), qb = level(b, sb, on_plane_face[1]);
            const bool above_a = qa != 0.0 ? qa > 0.0 : side_of(n, a.lean).sign > 0;
            const bool above_b = qb != 0.0 ? qb > 0.0 : side_of(n, b.lean).sign > 0;
            if (above_a == above_b) return;
            // A segment that meets the face inside it meets it; through an edge (a link
            // along a pillar's face in a plane of centres, past the end of the face next to
            // it; a segment from a leaned point along a face's plane to a point past the
            // next face's edge) the ends' steps say whether it does (see holds_meet); from
            // an end on the plane, where the end's steps take the meet. Between two faces of
            // f's plane, with both ends on them, it meets f's plane on the faces.
            const bool level_ends = qa == sa && qb == sb;
            if (!(qa == qb && on_plane_face[0] && on_plane_face[1]) &&
                !holds_meet(bvh, f, level_ends ? &meet : nullptr, a, qa, b, qb)) {
                return;
            }
            take(f, t);
        };
        if (distance > 0.0) {
            bvh.each_hit(a.at, dir, distance, [&](std::size_t f, double t) { meets(f, t, true); });
            const double reach = blind(bvh), least = bvh.min_distance();
            // Whether the cast along the segment left face f out.
            const auto left_out = [&](std::size_t f) {
                const double t = bvh.hit(f, a.at, dir);
                return !(t > least && t < distance);
            };
            if (a.close) {
                bvh.each_hit(a.at - dir * reach, dir, 2.0 * reach, [&](std::size_t f, double t) {
                    if (left_out(f)) meets(f, t - reach, false);
                });
            }
            if (b.close) {
                bvh.each_hit(b.at - dir * reach, dir, 2.0 * reach, [&](std::size_t f, double t) {
                    if (left_out(f)) meets(f, distance + (t - reach), false);
                });
            }
        }
        // Faces through an end: found from the end itself, since the segment's line can
        // meet their plane anywhere (far off, for a segment nearly in it).
        const auto through = [&](const End &e, const End &other, double t) {
            if (!e.ties) return;
            each_face_through(bvh, e.at, [&](std::size_t f) {
                tied = true;
                const Vec3 &n = bvh.normal(f);
                const Side at_e = side_of(n, e.lean);
                // The other end off the plane, on the other side; or on it, on the other
                // side by its step (both ends then on the plane: the segment meets it where
                // their steps put it, see holds_meet).
                const double s = bvh.offset(f, other.at);
                const bool off = std::abs(s) > tie;
                if (off ? (at_e.sign > 0) == (s > 0.0) : at_e.sign == side_of(n, other.lean).sign) {
                    return;
                }
                if (holds_meet(bvh, f, nullptr, e, 0.0, other, off ? s : 0.0)) take(f, t);
            });
        };
        through(a, b, 0.0);
        through(b, a, distance);
        // Where no tie touches any face and neither end lies within blind of a face (where
        // the segment's cast drops hits), the choice comes from that cast alone: its hits
        // with the face's edges widened as its hit test widens them (by kEdge, barycentric),
        // which a segment through an edge two faces share needs. Otherwise the faces the
        // rule finds and, of the others, those the segment meets inside them: so a segment
        // that passes exactly through an edge of one face (a tie) and just past the corner
        // of another, or one just over the edge of a face whose neighbour runs just under
        // an end, is judged by the rule alone. (Whether an end lies within blind of a face
        // matters only where a hit lies past its face's edge: only then is it looked for.)
        const bool differ = loose.face != held.face || loose.n != held.n;
        const bool strict = tied || (differ && ((a.close && within_blind(bvh, a.at)) ||
                                                (b.close && within_blind(bvh, b.at))));
        const Candidate &cast = strict ? held : loose;
        return (cast.before(ruled) ? cast : ruled).face;
    }

    // The cells round point p (up to eight) whose centres p sees, with trilinear weights
    // that sum to one; only air cells when `air` is given. p, on faces, takes `lean` (see
    // crossing).
    std::vector<std::pair<std::size_t, double>> visible_cells(const Bvh &bvh, const Vec3 &p,
                                                              const std::vector<std::uint8_t> *air,
                                                              const Vec3 &lean = {}) const {
        return seen_cells(bvh, p, lean,
                          [air](std::size_t i) { return air == nullptr || (*air)[i] != kSolid; });
    }

    // Where p lies among the cell centres along each axis, for its weights: q, its grid
    // coordinate (in cells from the first centre). A point on faces (see kTie), or with a
    // lean (a point round a listener on a face, which moves with it), within kTie cells
    // of a plane of centres across an axis lies on that plane (tied, and q the plane's
    // coordinate) and an infinitesimal step off it, to the side that its lean, then the
    // rule's steps, put it on, as for a face in that plane (side, see side_of): the centre
    // on the plane is the near one along that axis, and the one past the plane on that
    // side has a weight of the size of that step (see seen_cells); the cubic's box lies
    // round that side (see reading_cells). So such a point has the weights the same point
    // moved along its lean has, in the limit, whichever way its coordinate rounds.
    struct GridPlace {
        double q[3];
        bool tied[3] = {false, false, false};
        Side side[3] = {};
    };

    GridPlace grid_place(const Bvh &bvh, const Vec3 &p, const Vec3 &lean) const {
        GridPlace g{{(p.x - origin_.x) / cell_ - 0.5, (p.y - origin_.y) / cell_ - 0.5,
                     (p.z - origin_.z) / cell_ - 0.5}};
        bool near = false;
        for (const double q : g.q) near = near || std::abs(q - std::round(q)) <= kTie;
        if (!near || (!(norm(lean) > 0.0) && !on_face(bvh, p))) return g;
        const Vec3 axes[3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        for (int a = 0; a < 3; ++a) {
            const double plane = std::round(g.q[a]);
            if (std::abs(g.q[a] - plane) > kTie) continue;
            g.q[a] = plane;
            g.tied[a] = true;
            g.side[a] = side_of(axes[a], lean);
        }
        return g;
    }

    // The same, of the cells for which keep(i) holds. Of a point on faces (see
    // grid_place), a cell past a plane of centres the point is tied to has a weight
    // smaller than any of finite size, and those past more such planes, or past one by
    // a smaller step, smaller still; only the seen cells whose weights are of the largest
    // size seen are kept (the limit of the weights of the point moved off the face).
    template <typename Keep>
    std::vector<std::pair<std::size_t, double>> seen_cells(const Bvh &bvh, const Vec3 &p,
                                                           const Vec3 &lean, Keep keep) const {
        std::vector<std::pair<std::size_t, double>> out;
        const GridPlace g = grid_place(bvh, p, lean);
        const double *q = g.q;
        const double size[3] = {static_cast<double>(nx_), static_cast<double>(ny_),
                                static_cast<double>(nz_)};
        long long base[3];
        double frac[3];
        for (int a = 0; a < 3; ++a) {
            if (!(q[a] >= 1.0 && q[a] <= size[a] - 3.0)) return out;  // outside the grid
            base[a] = static_cast<long long>(std::floor(q[a]));
            if (g.tied[a] && g.side[a].sign < 0 && --base[a] < 1) return out;
            frac[a] = q[a] - static_cast<double>(base[a]);
        }
        // The seen cells, with their weights and how small those are: 0 for a finite
        // weight, else the sum over the tied planes the cell lies past of 4^step for the
        // step that puts p off that plane (step 0, its lean, at most three times; a larger
        // sum is a smaller weight).
        std::pair<std::size_t, double> seen[8];
        int smallness[8], count = 0, least = 0;
        for (int corner = 0; corner < 8; ++corner) {
            double w = 1.0;
            int small = 0;
            std::size_t i = 0, stride = 1;
            for (int a = 0; a < 3; ++a) {
                const int up = (corner >> a) & 1;
                if (!g.tied[a]) {
                    w *= up ? frac[a] : 1.0 - frac[a];
                } else if (up == (g.side[a].sign > 0 ? 1 : 0)) {  // past the plane
                    small += 1 << (2 * g.side[a].step);
                    if (g.side[a].step == 0) w *= std::abs(lean[a]);
                }
                i += static_cast<std::size_t>(base[a] + up) * stride;
                stride *= a == 0 ? nx_ : (a == 1 ? ny_ : 1);
            }
            if (w <= 0.0) continue;
            if (!keep(i)) continue;
            if (crossing(bvh, End{p, lean}, End{centre(i)}) >= 0) continue;
            if (count == 0 || small < least) least = small;
            seen[count] = {i, w};
            smallness[count++] = small;
        }
        double total = 0.0;
        for (int k = 0; k < count; ++k) {
            if (smallness[k] != least) continue;
            out.push_back(seen[k]);
            total += seen[k].second;
        }
        // The cut-off: a point whose seen cells all weigh under 1e-9 sees none; but
        // not a point on faces, whose cells are those of the point moved a hair off the faces
        // along its lean, renormalised over the cells it sees, however small their weights.
        if (least == 0 && total < 1e-9 && !on_face(bvh, p)) {
            out.clear();
            return out;
        }
        for (auto &cell_weight : out) cell_weight.second /= total;
        return out;
    }

    // Marks with `value` every cell still marked 0 in `mark` that the cells in `queue`
    // (marked already) reach through links that are not blocked; empties the queue.
    void flood(const std::vector<std::uint16_t> &cut, std::vector<std::size_t> &queue,
               std::vector<std::uint8_t> &mark, std::uint8_t value) const {
        while (!queue.empty()) {
            const std::size_t i = queue.back();
            queue.pop_back();
            for (int k = 0; k < directions_; ++k) {
                if (blocked(cut, i, k)) continue;
                const std::size_t j = neighbour(i, k);
                if (mark[j] == 0) {
                    mark[j] = value;
                    queue.push_back(j);
                }
            }
        }
    }

    // The air (type_) and the lean (source_lean_) of a source on faces (see kTie). Its
    // air is the one it sees with the rule's sides, unless that reaches no listener: then
    // that of the first of its other leans (see leans) whose air does. A listener on the
    // plane of a face through the source does not count for this, as from either side of
    // that face it can see the source's air (the outside of a room's wall, the inside of
    // a pillar): where only such listeners are reached, the one air that reaches them is
    // taken, and a source whose airs on different sides reach them is refused. A source
    // whose air reaches no listener takes the first air it sees (the errors that follow
    // say so), and one that sees no cell with any lean is refused; a lean whose cells lie
    // in two airs sees none. A listener reaches an
    // air when it sees a cell of it with one of its own leans. Of the leans that see
    // that air, the source takes the first, or on an edge or in a corner the widest (see
    // widest).
    void place_on_face(const Bvh &bvh, const std::vector<std::uint16_t> &cut, const Vec3 &source,
                       const std::vector<Vec3> &probes, std::size_t per_listener) {
        const std::size_t n = nx_ * ny_ * nz_;
        const double tie = kTie * cell_;
        // The airs the source's cells lie in, each flooded once: region[i] is 1 + the
        // index of the air of cell i, 0 for none. Each lean sees cells of the same 3 x 3 x 3
        // round the source (two along an axis it is not tied across; along a tied one, the
        // centre on the plane and the one past it on the lean's side), so there are at most
        // 27, which `airs` holds a bit each of.
        std::vector<std::uint8_t> region(n, 0);
        std::uint8_t regions = 0;
        struct Option {
            Vec3 lean;
            unsigned airs;  // bit r for air r
        };
        std::vector<Option> options;  // the leans that see cells, each set of airs once
        const Leans candidates = leans(bvh, source);
        std::vector<unsigned> airs_of;  // of each candidate
        for (const Vec3 &lean : candidates.all) {
            unsigned airs = 0;
            for (const auto &[i, w] : visible_cells(bvh, source, nullptr, lean)) {
                (void)w;
                if (region[i] == 0) {
                    region[i] = ++regions;
                    std::vector<std::size_t> queue{i};
                    flood(cut, queue, region, regions);
                }
                airs |= 1u << (region[i] - 1);
            }
            // A lean whose cells lie in two airs is none: one source feeds one air.
            if ((airs & (airs - 1)) != 0) airs = 0;
            airs_of.push_back(airs);
            bool known = airs == 0;
            for (const Option &o : options) known = known || o.airs == airs;
            if (!known) options.push_back({lean, airs});
        }
        if (options.empty()) {
            throw std::invalid_argument(
                "the source lies on a surface of the model and sees no cell of the wave grid "
                "on any side of it: move it off the surface into the air");
        }
        std::vector<std::size_t> faces;
        each_face_through(bvh, source, [&](std::size_t f) { faces.push_back(f); });
        // Whether listener l (probe l of the groups) sees a cell of `airs`, and whether
        // it lies on the plane of a face through the source.
        const auto reaches = [&](std::size_t l, unsigned airs) {
            const auto in = [&](std::size_t i) {
                return region[i] != 0 && ((airs >> (region[i] - 1)) & 1u) != 0;
            };
            const Vec3 &p = probes[l];
            if (!seen_cells(bvh, p, Vec3{}, in).empty()) return true;
            if (!on_face(bvh, p)) return false;
            for (const Vec3 &lean : leans(bvh, p).all) {
                if (!seen_cells(bvh, p, lean, in).empty()) return true;
            }
            return false;
        };
        const auto beside = [&](std::size_t l) {
            for (const std::size_t f : faces) {
                if (std::abs(bvh.offset(f, probes[l])) <= tie) return true;
            }
            return false;
        };
        const Option *chosen = nullptr;
        for (const Option &o : options) {
            for (std::size_t l = 0; l < probes.size() && !chosen; l += per_listener) {
                if (!beside(l) && reaches(l, o.airs)) chosen = &o;
            }
            if (chosen) break;
        }
        if (!chosen) {
            const Option *first = nullptr;  // the first that reaches a listener beside it
            for (const Option &o : options) {
                bool reached = false;
                for (std::size_t l = 0; l < probes.size() && !reached; l += per_listener) {
                    reached = beside(l) && reaches(l, o.airs);
                }
                if (!reached) continue;
                if (first) {
                    throw std::invalid_argument(
                        "the source lies on a surface of the model, and the only listeners "
                        "its sound reaches lie on that surface too, where they hear it from "
                        "either side: move the source or the listeners off the surface");
                }
                first = &o;
            }
            chosen = first ? first : &options.front();
        }
        const unsigned airs = chosen->airs;
        const auto same_air = [&](std::size_t k) { return airs_of[k] == airs; };
        source_lean_ = widest(bvh, source, candidates, same_air).second;
        type_.assign(n, kSolid);
        for (std::size_t i = 0; i < n; ++i) {
            if (region[i] != 0 && ((airs >> (region[i] - 1)) & 1u) != 0) {
                type_[i] = kInterior;
            }
        }
    }

    // Makes cell i special, with the weights w of its links along each direction and
    // the branch velocities added since the last one.
    // With a speed field its links weigh s times as much, its boundary terms g (and its
    // branch velocities' weights, set by the caller) r times.
    std::size_t add_special(std::size_t i, const double *w, double g) {
        type_[i] = kSpecial;
        special_.push_back(static_cast<std::int64_t>(i));
        const double s = cell_scale(i), r = std::sqrt(s);
        double total = 0.0;
        for (int k = 0; k < directions_; ++k) {
            if (!(w[k] > 0.0)) continue;
            total += w[k];
            link_direction_.push_back(static_cast<std::uint8_t>(k));
            link_weight_.push_back(lowered(s * w[k]));
        }
        link_start_.push_back(link_direction_.size());
        coef_a_.push_back(static_cast<float>(2.0 - s * total * (1.0 - kMargin)));
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
    double mean_square_ = 0.0;
    std::vector<std::vector<std::size_t>> probe_cells_;
    std::vector<std::vector<double>> probe_weights_;
    std::vector<std::size_t> run_begin_, run_end_;
    std::vector<std::uint8_t> probe_in_sight_;
    // The source lies on a face (see kTie); and the lean it takes there.
    bool source_on_face_ = false;
    Vec3 source_lean_{};
    std::vector<float> u0_, u1_;
    // Moving air: each cell's shifts and gradient neighbours (see set_up_flow), and the
    // carrying's two stages; all empty in still air. With the wind round the solids
    // (round_; see set_up_round) the shifts are let go of for the wind on the cells'
    // faces towards +x, +y and +z and the loops of each three cells in a row from the
    // cell on, and flow_mask_ says which cells the air moves.
    std::vector<float> flow_x_, flow_y_, flow_t1_, flow_t2_;
    std::vector<std::uint8_t> flow_mask_;
    std::array<std::vector<float>, 3> wind_, loop_;
    bool round_ = false;
    float fastest_ = 0.0f;
    std::size_t sweeps_ = 0;
};

}  // namespace acoustic_spaces
