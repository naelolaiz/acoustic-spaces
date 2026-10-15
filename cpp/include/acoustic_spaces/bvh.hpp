// SPDX-License-Identifier: GPL-3.0-or-later
//
// Triangle bounding volume hierarchy for ray queries against a room model.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "acoustic_spaces/vec3.hpp"

namespace acoustic_spaces {

struct Hit {
    double t = std::numeric_limits<double>::infinity();
    std::int64_t face = -1;
    bool found() const { return face >= 0; }
};

class Bvh {
public:
    // vertices: n_vertices x 3, faces: n_faces x 3 indices into vertices (row-major).
    Bvh(const double *vertices, std::size_t n_vertices, const std::int64_t *faces,
        std::size_t n_faces) {
        if (n_faces == 0) {
            throw std::invalid_argument("Bvh: the mesh has no faces");
        }
        tri_.reserve(n_faces);
        for (std::size_t f = 0; f < n_faces; ++f) {
            Vec3 p[3];
            for (std::size_t k = 0; k < 3; ++k) {
                const std::int64_t index = faces[3 * f + k];
                if (index < 0 || static_cast<std::size_t>(index) >= n_vertices) {
                    throw std::invalid_argument("Bvh: face refers to a missing vertex");
                }
                const double *v = vertices + 3 * static_cast<std::size_t>(index);
                p[k] = Vec3{v[0], v[1], v[2]};
            }
            Triangle t;
            t.v0 = p[0];
            t.e1 = p[1] - p[0];
            t.e2 = p[2] - p[0];
            const Vec3 n = cross(t.e1, t.e2);
            const double len = norm(n);
            t.normal = len > 0.0 ? n / len : Vec3{0.0, 0.0, 0.0};
            tri_.push_back(t);
        }
        order_.resize(n_faces);
        for (std::size_t i = 0; i < n_faces; ++i) {
            order_[i] = static_cast<std::uint32_t>(i);
        }
        std::vector<Vec3> centroids(n_faces);
        std::vector<Box> boxes(n_faces);
        for (std::size_t i = 0; i < n_faces; ++i) {
            const Triangle &t = tri_[i];
            const Vec3 a = t.v0, b = t.v0 + t.e1, c = t.v0 + t.e2;
            boxes[i] = Box{min(a, min(b, c)), max(a, max(b, c))};
            centroids[i] = (a + b + c) / 3.0;
        }
        Box all;
        for (const Box &b : boxes) all.grow(b);
        scale_ = n_faces > 0 ? std::max(norm(all.hi - all.lo), 1e-9) : 1.0;
        // Pad the boxes so that rays grazing an edge or a vertex still reach its faces.
        const Vec3 pad{1e-9 * scale_, 1e-9 * scale_, 1e-9 * scale_};
        for (Box &b : boxes) {
            b.lo = b.lo - pad;
            b.hi = b.hi + pad;
        }
        nodes_.reserve(2 * n_faces);
        build(0, n_faces, boxes, centroids);
    }

    std::size_t face_count() const { return tri_.size(); }

    // Unit normal of a face, following the right-hand rule on its vertex order.
    const Vec3 &normal(std::size_t face) const { return tri_[face].normal; }

    // The corners of a face, in its vertex order.
    std::array<Vec3, 3> corners(std::size_t face) const {
        const Triangle &t = tri_[face];
        return {t.v0, t.v0 + t.e1, t.v0 + t.e2};
    }

    // Length of the model's bounding box diagonal; used to scale tolerances.
    double scale() const { return scale_; }

    // Nearest hit with t in (t_min, t_max); `ignore` is a face to skip (the one a
    // reflected ray starts on), or -1.
    Hit intersect(const Vec3 &origin, const Vec3 &dir, double t_max,
                  std::int64_t ignore = -1) const {
        Hit hit;
        hit.t = t_max;
        traverse(origin, dir, ignore, [&](std::size_t face, double t) {
            if (t < hit.t) {
                hit.t = t;
                hit.face = static_cast<std::int64_t>(face);
            }
            return false;
        }, hit.t);
        if (!hit.found()) {
            hit.t = std::numeric_limits<double>::infinity();
        }
        return hit;
    }

    // True when any face lies between origin and origin + t_max * dir.
    bool occluded(const Vec3 &origin, const Vec3 &dir, double t_max,
                  std::int64_t ignore = -1) const {
        bool any = false;
        double limit = t_max;
        traverse(origin, dir, ignore, [&](std::size_t, double) {
            any = true;
            return true;
        }, limit);
        return any;
    }

private:
    struct Triangle {
        Vec3 v0, e1, e2, normal;
    };
    struct Box {
        Vec3 lo{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
        Vec3 hi{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()};
        void grow(const Box &b) {
            lo = min(lo, b.lo);
            hi = max(hi, b.hi);
        }
    };
    struct Node {
        Box box;
        std::uint32_t first = 0;  // leaf: first index into order_; inner: right child
        std::uint32_t count = 0;  // leaf: number of faces; inner: 0
    };

    static constexpr std::size_t kLeafSize = 4;

    std::vector<Triangle> tri_;
    std::vector<std::uint32_t> order_;
    std::vector<Node> nodes_;
    double scale_ = 1.0;

    std::size_t build(std::size_t begin, std::size_t end, const std::vector<Box> &boxes,
                      const std::vector<Vec3> &centroids) {
        const std::size_t index = nodes_.size();
        nodes_.emplace_back();
        Box box, cbox;
        for (std::size_t i = begin; i < end; ++i) {
            box.grow(boxes[order_[i]]);
            cbox.grow(Box{centroids[order_[i]], centroids[order_[i]]});
        }
        nodes_[index].box = box;
        const std::size_t n = end - begin;
        const Vec3 extent = cbox.hi - cbox.lo;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;
        if (n <= kLeafSize || extent[axis] <= 0.0) {
            nodes_[index].first = static_cast<std::uint32_t>(begin);
            nodes_[index].count = static_cast<std::uint32_t>(n);
            return index;
        }
        const std::size_t mid = begin + n / 2;
        std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                         order_.begin() + static_cast<std::ptrdiff_t>(mid),
                         order_.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::uint32_t a, std::uint32_t b) {
                             return centroids[a][axis] < centroids[b][axis];
                         });
        build(begin, mid, boxes, centroids);  // left child is index + 1
        const std::size_t right = build(mid, end, boxes, centroids);
        nodes_[index].first = static_cast<std::uint32_t>(right);
        nodes_[index].count = 0;
        return index;
    }

    static bool slab(const Box &b, const Vec3 &origin, const Vec3 &inv, double t_max) {
        double t0 = 0.0, t1 = t_max;
        for (int a = 0; a < 3; ++a) {
            const double ta = (b.lo[a] - origin[a]) * inv[a];
            const double tb = (b.hi[a] - origin[a]) * inv[a];
            // fmin/fmax drop the NaN of 0 * inf, for a ray lying in a slab boundary.
            t0 = std::fmax(t0, std::fmin(ta, tb));
            t1 = std::fmin(t1, std::fmax(ta, tb));
        }
        return t0 <= t1;
    }

    // Moller-Trumbore, two-sided; returns t or a negative value for a miss. Edges are
    // widened by kEdge (barycentric) so that a ray through a shared edge or vertex hits
    // a face on one side or the other instead of slipping between them.
    static constexpr double kEdge = 1e-9;

    double hit_triangle(const Triangle &tri, const Vec3 &origin, const Vec3 &dir) const {
        const Vec3 p = cross(dir, tri.e2);
        const double det = dot(tri.e1, p);
        if (std::abs(det) < 1e-300) return -1.0;
        const double inv_det = 1.0 / det;
        const Vec3 s = origin - tri.v0;
        const double u = dot(s, p) * inv_det;
        if (u < -kEdge || u > 1.0 + kEdge) return -1.0;
        const Vec3 q = cross(s, tri.e1);
        const double v = dot(dir, q) * inv_det;
        if (v < -kEdge || u + v > 1.0 + kEdge) return -1.0;
        return dot(tri.e2, q) * inv_det;
    }

    // Calls visit(face, t) for faces hit with t in (t_min, limit); visit returns true to
    // stop. `limit` shrinks as the caller records closer hits.
    template <typename Visit>
    void traverse(const Vec3 &origin, const Vec3 &dir, std::int64_t ignore, Visit visit,
                  double &limit) const {
        const double t_min = 1e-9 * scale_;
        const Vec3 inv{1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z};
        std::array<std::uint32_t, 64> stack{};
        std::size_t top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node &node = nodes_[stack[--top]];
            if (!slab(node.box, origin, inv, limit)) continue;
            if (node.count > 0) {
                for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
                    const std::uint32_t face = order_[i];
                    if (static_cast<std::int64_t>(face) == ignore) continue;
                    const double t = hit_triangle(tri_[face], origin, dir);
                    if (t > t_min && t < limit && visit(face, t)) return;
                }
            } else {
                const std::uint32_t self =
                    static_cast<std::uint32_t>(&node - nodes_.data());
                if (top + 2 > stack.size()) {
                    throw std::runtime_error("Bvh: traversal stack overflow");
                }
                stack[top++] = node.first;  // right
                stack[top++] = self + 1;    // left
            }
        }
    }
};

}  // namespace acoustic_spaces
