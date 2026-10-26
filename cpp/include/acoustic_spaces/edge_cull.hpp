// SPDX-License-Identifier: GPL-3.0-or-later
//
// Which pairs of a diffracting edge and a reflecting plane can carry a path that
// reflects off the plane between the edge and an end point (the source or the listener),
// as acoustic_spaces/engine/diffraction.py tests them before its ray tests: the others
// are turned down by those tests, so leaving them out changes no result and spares the
// work, which otherwise grows as the edges times the planes.
//
// The ray tests start within `offset` of a point of the edge (twice that by a very
// short edge) and head for the end's mirror image in the plane. A ray that starts behind
// the plane (on the far side from the end, further than the plane's faces stray from it)
// does not reach it, while the image lies beyond them. The rest of the edge, clipped
// where it passes `guard` behind the plane, casts a shadow on the plane from the image;
// a ray from that part meets the plane on that shadow, to within (4 offset + 2 stray)
// times one plus one over the smaller of the cosines of the rays from the image to the
// ends of that part to the plane's normal, so the reflection can only be on a face whose
// box takes in some of that stretch. Pairs whose rays graze the plane (that cosine below
// `grazing`), or whose image lies too near the plane for those bounds to hold, are kept.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "acoustic_spaces/ray_tracer.hpp"
#include "acoustic_spaces/vec3.hpp"

namespace acoustic_spaces {

// A reflecting plane as the culling sees it: the normal on the end's side, how far
// along it the plane lies, and the box of its faces that are not openings (empty, low
// above high, for none) with how far those faces stray from the plane.
struct CullPlane {
    Vec3 normal;
    double offset = 0.0;
    Vec3 low, high;
    double spread = 0.0;
};

// The planes as seen from an end point, and the test of an edge against one of them.
class PlaneCull {
   public:
    PlaneCull(const std::vector<CullPlane> &planes, const Vec3 &end, double offset, double grazing)
        : seen_(planes.size()), grazing_(grazing) {
        for (std::size_t j = 0; j < planes.size(); ++j) {
            const CullPlane &p = planes[j];
            const double distance = dot(p.normal, end) - p.offset;
            const double side = distance < 0.0 ? -1.0 : 1.0;
            Seen &s = seen_[j];
            // Turned towards the end, so that the end lies in front of the plane.
            s.normal = p.normal * side;
            s.offset = p.offset * side;
            s.far = std::abs(distance);
            s.image = end - p.normal * (2.0 * distance);
            s.low = p.low;
            s.high = p.high;
            s.guard = 2.0 * offset + p.spread;
            s.pad = 4.0 * offset + 2.0 * p.spread;
            s.reached = s.far > p.spread;
            s.usable = s.far >= 4.0 * s.guard;
        }
    }

    std::size_t size() const { return seen_.size(); }

    // The end's mirror image in plane j.
    const Vec3 &image(std::size_t j) const { return seen_[j].image; }

    // Whether a path from the edge from a to b can reflect off plane j on its way to the
    // end.
    bool keeps(const Vec3 &a, const Vec3 &b, std::size_t j) const {
        const Seen &s = seen_[j];
        const double da = dot(s.normal, a) - s.offset;
        const double db = dot(s.normal, b) - s.offset;
        if (da < -s.guard && db < -s.guard) return !s.reached;  // behind the plane
        if (!s.usable) return true;
        // The part of the edge no further than guard behind the plane.
        const double across = da - db;
        Vec3 xa = a, xb = b;
        if (da < -s.guard) xa = a + (b - a) * ((da + s.guard) / across);
        if (db < -s.guard) xb = b - (b - a) * (-(db + s.guard) / across);
        const double ea = std::max(da, -s.guard), eb = std::max(db, -s.guard);
        const Vec3 ga = s.image - xa, gb = s.image - xb;
        const double cosine = std::min((ea + s.far) / std::sqrt(std::max(dot(ga, ga), 1e-300)),
                                       (eb + s.far) / std::sqrt(std::max(dot(gb, gb), 1e-300)));
        if (cosine < grazing_) return true;
        // Their shadows on the plane, cast from the image.
        const Vec3 pa = xa + ga * (ea / (ea + s.far));
        const Vec3 pb = xb + gb * (eb / (eb + s.far));
        const double margin = s.pad * (1.0 + 1.0 / cosine);
        for (int k = 0; k < 3; ++k) {
            if (std::min(pa[k], pb[k]) > s.high[k] + margin) return false;
            if (std::max(pa[k], pb[k]) < s.low[k] - margin) return false;
        }
        return true;
    }

   private:
    struct Seen {
        Vec3 normal, image, low, high;
        double offset = 0.0, far = 0.0, guard = 0.0, pad = 0.0;
        bool reached = false, usable = false;
    };
    std::vector<Seen> seen_;
    double grazing_;
};

using Pairs = std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>;

// Runs emit(i, rows, cols) for i in [0, n) on up to `threads` threads, each appending its
// pairs, and joins them in the order of i.
template <typename Emit>
Pairs pairs_in_order(std::size_t n, std::size_t threads, Emit emit) {
    threads = std::max<std::size_t>(1, std::min(worker_threads(threads), n));
    std::vector<Pairs> found(threads);
    parallel_for(n, threads, [&](std::size_t begin, std::size_t stop, std::size_t worker) {
        for (std::size_t i = begin; i < stop; ++i)
            emit(i, found[worker].first, found[worker].second);
    });
    Pairs out;
    for (const Pairs &f : found) {
        out.first.insert(out.first.end(), f.first.begin(), f.first.end());
        out.second.insert(out.second.end(), f.second.begin(), f.second.end());
    }
    return out;
}

// The pairs of an edge (from starts[i] to ends[i]) and a plane that a path from the edge
// to `end` can reflect off, by edge and then plane: (edge rows, plane columns).
inline Pairs reflecting_pairs(const std::vector<Vec3> &starts, const std::vector<Vec3> &ends,
                              const std::vector<CullPlane> &planes, const Vec3 &end, double offset,
                              double grazing, std::size_t threads) {
    const PlaneCull cull(planes, end, offset, grazing);
    return pairs_in_order(starts.size(), threads, [&](std::size_t i, auto &rows, auto &cols) {
        for (std::size_t j = 0; j < cull.size(); ++j) {
            if (cull.keeps(starts[i], ends[i], j)) {
                rows.push_back(static_cast<std::int64_t>(i));
                cols.push_back(static_cast<std::int64_t>(j));
            }
        }
    });
}

// The source sides of paths over an edge that reflect off a plane after it on their way
// to `end`, with that plane: for each row, an edge (row_edge, into starts and ends) with
// the source side's position along it and distance from it (t_s, rho_s), the planes that
// reflecting_pairs keeps whose image of `end` makes a path over the edge that touches it
// (where it makes equal angles with the edge on both sides) and is no longer than
// max_distance, all to within `slack` (m), with the image more than half `tolerance`
// off the edge's line: a superset of those the exact tests take. By row and then plane:
// (rows, plane columns).
inline Pairs reaching_pairs(const std::vector<Vec3> &starts, const std::vector<Vec3> &ends,
                            const std::vector<CullPlane> &planes, const Vec3 &end, double offset,
                            double grazing, const std::vector<std::int64_t> &row_edge,
                            const std::vector<double> &t_s, const std::vector<double> &rho_s,
                            double max_distance, double tolerance, double slack,
                            std::size_t threads) {
    const PlaneCull cull(planes, end, offset, grazing);
    // The planes each edge keeps, edge by edge.
    std::vector<std::size_t> count(starts.size(), 0);
    const Pairs kept =
        pairs_in_order(starts.size(), threads, [&](std::size_t i, auto &rows, auto &cols) {
            for (std::size_t j = 0; j < cull.size(); ++j) {
                if (cull.keeps(starts[i], ends[i], j)) {
                    rows.push_back(static_cast<std::int64_t>(i));
                    cols.push_back(static_cast<std::int64_t>(j));
                }
            }
        });
    for (const std::int64_t i : kept.first) ++count[static_cast<std::size_t>(i)];
    std::vector<std::size_t> first(starts.size() + 1, 0);
    for (std::size_t i = 0; i < starts.size(); ++i) first[i + 1] = first[i] + count[i];
    return pairs_in_order(row_edge.size(), threads, [&](std::size_t r, auto &rows, auto &cols) {
        const std::size_t i = static_cast<std::size_t>(row_edge[r]);
        const Vec3 a = starts[i], along = ends[i] - starts[i];
        const double length = norm(along);
        const Vec3 direction = along / std::max(length, 1e-300);
        for (std::size_t k = first[i]; k < first[i + 1]; ++k) {
            const std::size_t j = static_cast<std::size_t>(kept.second[k]);
            const Vec3 v = cull.image(j) - a;
            const double t_r = dot(v, direction);
            const double rho_r = norm(v - direction * t_r);
            if (rho_r <= 0.5 * tolerance) continue;
            const double t_k = t_s[r] + (t_r - t_s[r]) * rho_s[r] / (rho_s[r] + rho_r);
            if (t_k < -slack || t_k > length + slack) continue;
            const double r1 = std::hypot(rho_s[r], t_k - t_s[r]);
            const double r2 = std::hypot(rho_r, t_r - t_k);
            if (r1 + r2 > max_distance + slack) continue;
            rows.push_back(static_cast<std::int64_t>(r));
            cols.push_back(static_cast<std::int64_t>(j));
        }
    });
}

}  // namespace acoustic_spaces
