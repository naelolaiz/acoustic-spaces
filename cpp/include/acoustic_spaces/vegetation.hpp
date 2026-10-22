// SPDX-License-Identifier: GPL-3.0-or-later
//
// Vegetation outdoors, as acoustic_spaces/engine/vegetation.py has it: volumes that rays
// pass through, filled with a medium that absorbs and scatters sound. A volume is an
// upright prism (a polygon raised from one height to another: a belt of trees, a hedge)
// or a spheroid with a vertical axis (a lone tree's crown); each holds one medium, and
// where volumes overlap their media add up. A medium has, per band, the absorption and
// the scattering per metre of the energy, and the mean cosine of the angle it scatters
// through. An upright medium (stems) takes them times the horizontal share of a ray's
// direction: a ray going straight up runs along the stems and meets none.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "acoustic_spaces/vec3.hpp"

namespace acoustic_spaces {

struct Vegetation {
    struct Shape {
        int kind = 0;  // 0: prism, 1: spheroid
        std::size_t medium = 0;
        // Prism: from z0 to z1 over the polygon of `count` corners in `outline` from
        // corner `first` on.
        double z0 = 0.0, z1 = 0.0;
        std::size_t first = 0, count = 0;
        // Spheroid: its centre, horizontal and vertical semi-axes.
        Vec3 centre{};
        double radius = 0.0, half_height = 0.0;
        // A box round it.
        Vec3 lo{}, hi{};
    };

    // A part of a segment inside a shape: from t0 to t1 metres along it.
    struct Span {
        double t0, t1;
        std::size_t medium;
    };

    std::size_t bands = 0;
    std::vector<Shape> shapes;
    std::vector<double> outline;                            // x, y of every prism's corners
    std::vector<double> absorption, scattering, asymmetry;  // media x bands
    std::vector<std::uint8_t> upright;                      // per medium

    std::size_t media() const { return upright.size(); }

    // Fills each shape's box; false when the shapes or media do not fit together.
    bool prepare() {
        const std::size_t m = media();
        if (bands == 0 || absorption.size() != m * bands || scattering.size() != m * bands ||
            asymmetry.size() != m * bands) {
            return false;
        }
        for (std::size_t i = 0; i < m * bands; ++i) {
            if (!(absorption[i] >= 0.0) || !(scattering[i] >= 0.0) ||
                !(std::abs(asymmetry[i]) < 1.0)) {
                return false;
            }
        }
        const std::size_t corners = outline.size() / 2;
        for (Shape &s : shapes) {
            if (s.medium >= m) return false;
            if (s.kind == 0) {
                if (s.count < 3 || s.first + s.count > corners || !(s.z1 > s.z0)) return false;
                s.lo = {outline[2 * s.first], outline[2 * s.first + 1], s.z0};
                s.hi = {s.lo.x, s.lo.y, s.z1};
                for (std::size_t k = s.first; k < s.first + s.count; ++k) {
                    s.lo.x = std::min(s.lo.x, outline[2 * k]);
                    s.lo.y = std::min(s.lo.y, outline[2 * k + 1]);
                    s.hi.x = std::max(s.hi.x, outline[2 * k]);
                    s.hi.y = std::max(s.hi.y, outline[2 * k + 1]);
                }
            } else if (s.kind == 1) {
                if (!(s.radius > 0.0) || !(s.half_height > 0.0)) return false;
                const Vec3 axes{s.radius, s.radius, s.half_height};
                s.lo = s.centre - axes;
                s.hi = s.centre + axes;
            } else {
                return false;
            }
        }
        return true;
    }

    // How much of a ray going along unit d a medium meets per metre: its horizontal
    // share for stems, all of it otherwise.
    double share(std::size_t medium, const Vec3 &d) const {
        return upright[medium] ? std::hypot(d.x, d.y) : 1.0;
    }

    // Whether (x, y) lies inside the polygon of prism s (crossing number).
    bool inside(const Shape &s, double x, double y) const {
        bool in = false;
        for (std::size_t i = 0, j = s.count - 1; i < s.count; j = i++) {
            const double xi = outline[2 * (s.first + i)], yi = outline[2 * (s.first + i) + 1];
            const double xj = outline[2 * (s.first + j)], yj = outline[2 * (s.first + j) + 1];
            if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) in = !in;
        }
        return in;
    }

    // Adds to `out` the parts of the segment from p along unit d, t in [0, length],
    // inside each shape.
    void spans(const Vec3 &p, const Vec3 &d, double length, std::vector<Span> &out,
               std::vector<double> &cuts) const {
        out.clear();
        const Vec3 q = p + d * length;
        const Vec3 near = min(p, q), far = max(p, q);
        for (const Shape &s : shapes) {
            if (far.x < s.lo.x || far.y < s.lo.y || far.z < s.lo.z || near.x > s.hi.x ||
                near.y > s.hi.y || near.z > s.hi.z) {
                continue;
            }
            if (s.kind == 1) {
                const Vec3 axes{s.radius, s.radius, s.half_height};
                const Vec3 a{(p.x - s.centre.x) / axes.x, (p.y - s.centre.y) / axes.y,
                             (p.z - s.centre.z) / axes.z};
                const Vec3 e{d.x / axes.x, d.y / axes.y, d.z / axes.z};
                const double aa = dot(e, e), b = dot(a, e), c = dot(a, a) - 1.0;
                const double disc = b * b - aa * c;
                if (!(disc > 0.0)) continue;
                const double root = std::sqrt(disc);
                const double t0 = std::max((-b - root) / aa, 0.0);
                const double t1 = std::min((-b + root) / aa, length);
                if (t1 > t0) out.push_back({t0, t1, s.medium});
                continue;
            }
            // The prism: the slab between its heights, then the polygon's crossings.
            double ta = 0.0, tb = length;
            if (std::abs(d.z) < 1e-12) {
                if (!(p.z > s.z0 && p.z < s.z1)) continue;
            } else {
                double a = (s.z0 - p.z) / d.z, b = (s.z1 - p.z) / d.z;
                if (a > b) std::swap(a, b);
                ta = std::max(ta, a);
                tb = std::min(tb, b);
            }
            if (!(tb > ta)) continue;
            cuts.clear();
            cuts.push_back(ta);
            for (std::size_t i = 0; i < s.count; ++i) {
                const std::size_t k = s.first + i, l = s.first + (i + 1) % s.count;
                const double ax = outline[2 * k], ay = outline[2 * k + 1];
                const double ex = outline[2 * l] - ax, ey = outline[2 * l + 1] - ay;
                const double denom = d.x * ey - d.y * ex;
                if (std::abs(denom) < 1e-12) continue;
                const double wx = ax - p.x, wy = ay - p.y;
                const double t = (wx * ey - wy * ex) / denom;
                const double u = (wx * d.y - wy * d.x) / denom;
                if (u >= 0.0 && u < 1.0 && t > ta && t < tb) cuts.push_back(t);
            }
            cuts.push_back(tb);
            std::sort(cuts.begin() + 1, cuts.end() - 1);
            for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
                const double t0 = cuts[i], t1 = cuts[i + 1];
                if (!(t1 > t0)) continue;
                const double middle = 0.5 * (t0 + t1);
                if (!inside(s, p.x + d.x * middle, p.y + d.y * middle)) continue;
                // Join it to the part before it when they meet.
                if (!out.empty() && out.back().medium == s.medium && out.back().t1 == t0) {
                    out.back().t1 = t1;
                } else {
                    out.push_back({t0, t1, s.medium});
                }
            }
        }
    }

    // Per band, the absorption and scattering together met along the segment from p
    // along unit d for `length` metres (nepers of energy) into tau.
    void depth(const Vec3 &p, const Vec3 &d, double length, double *tau,
               std::vector<Span> &scratch, std::vector<double> &cuts) const {
        std::fill(tau, tau + bands, 0.0);
        spans(p, d, length, scratch, cuts);
        for (const Span &s : scratch) {
            const double metres = (s.t1 - s.t0) * share(s.medium, d);
            const double *a = absorption.data() + s.medium * bands;
            const double *b = scattering.data() + s.medium * bands;
            for (std::size_t k = 0; k < bands; ++k) tau[k] += (a[k] + b[k]) * metres;
        }
    }
};

}  // namespace acoustic_spaces
