// SPDX-License-Identifier: GPL-3.0-or-later
//
// Triangle bounding volume hierarchy for ray queries against a room model: split by
// the surface area heuristic, traversed nearest child first.

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

// How the tree splits its faces: by the surface area heuristic (fastest to trace), or
// at the middle face along the longest side of their centres (the split before it).
enum class BvhSplit : std::uint8_t { kSah = 0, kMedian = 1 };

struct Hit {
    double t = std::numeric_limits<double>::infinity();
    std::int64_t face = -1;
    bool found() const { return face >= 0; }
};

class Bvh {
public:
    // vertices: n_vertices x 3, faces: n_faces x 3 indices into vertices (row-major).
    Bvh(const double *vertices, std::size_t n_vertices, const std::int64_t *faces,
        std::size_t n_faces, BvhSplit split = BvhSplit::kSah)
        : split_(split) {
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
        if (n_faces > 0) {
            lower_ = all.lo;
            upper_ = all.hi;
        }
        // Pad the boxes so that rays grazing an edge or a vertex still reach its faces.
        const Vec3 pad{1e-9 * scale_, 1e-9 * scale_, 1e-9 * scale_};
        for (Box &b : boxes) {
            b.lo = b.lo - pad;
            b.hi = b.hi + pad;
        }
        nodes_.reserve(2 * n_faces);
        nodes_.emplace_back();
        build(0, 0, n_faces, 0, boxes, centroids);
        // Keep the triangles in leaf order, so a leaf's are next to each other in memory.
        std::vector<Triangle> sorted(n_faces);
        place_.resize(n_faces);
        for (std::size_t i = 0; i < n_faces; ++i) {
            sorted[i] = tri_[order_[i]];
            place_[order_[i]] = static_cast<std::uint32_t>(i);
        }
        tri_ = std::move(sorted);
    }

    std::size_t face_count() const { return tri_.size(); }

    // Unit normal of a face, following the right-hand rule on its vertex order.
    const Vec3 &normal(std::size_t face) const { return tri_[place_[face]].normal; }

    // How far p lies from a face's plane, along its normal.
    double offset(std::size_t face, const Vec3 &p) const {
        const Triangle &t = tri_[place_[face]];
        return dot(t.normal, p - t.v0);
    }

    // The corners of a face, in its vertex order.
    std::array<Vec3, 3> corners(std::size_t face) const {
        const Triangle &t = tri_[place_[face]];
        return {t.v0, t.v0 + t.e1, t.v0 + t.e2};
    }

    // Length of the model's bounding box diagonal; used to scale tolerances.
    double scale() const { return scale_; }

    // Hits closer than this to a ray's origin are not reported, so that a ray leaving a
    // face does not meet it again where it starts.
    double min_distance() const { return 1e-9 * scale_; }

    // Whether p lies inside the model's bounding box, grown by `pad` on every side.
    bool encloses(const Vec3 &p, double pad) const {
        return p.x >= lower_.x - pad && p.x <= upper_.x + pad && p.y >= lower_.y - pad &&
               p.y <= upper_.y + pad && p.z >= lower_.z - pad && p.z <= upper_.z + pad;
    }

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

    // Calls visit(face, t) for every face hit with t in (min_distance(), t_max), in no
    // particular order.
    template <typename Visit>
    void each_hit(const Vec3 &origin, const Vec3 &dir, double t_max, Visit visit) const {
        double limit = t_max;
        traverse(
            origin, dir, -1,
            [&](std::size_t face, double t) {
                visit(face, t);
                return false;
            },
            limit);
    }

    // t of the ray's hit on a face, by the arithmetic of intersect and each_hit (which keep
    // it when it lies in (min_distance(), t_max)); negative for a miss.
    double hit(std::size_t face, const Vec3 &origin, const Vec3 &dir) const {
        return hit_triangle(tri_[place_[face]], origin, dir);
    }

    // Calls visit(face, offset) for every face whose plane passes within `tolerance` of
    // p (`offset` from it, along the normal) and that lies in a leaf of the tree whose box,
    // grown by `tolerance`, holds p, in no particular order: every face within `tolerance`
    // of p, and some further off.
    template <typename Visit>
    void each_face_near(const Vec3 &p, double tolerance, Visit visit) const {
        std::array<std::uint32_t, kMaxDepth + 4> stack{};
        std::size_t top = 0;
        std::uint32_t current = 0;
        for (;;) {
            const Node &node = nodes_[current];
            const Box &box = node.box;
            bool inside = true;
            for (int a = 0; a < 3; ++a) {
                inside = inside && p[a] >= box.lo[a] - tolerance && p[a] <= box.hi[a] + tolerance;
            }
            if (inside && node.count > 0) {
                for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
                    const Triangle &tri = tri_[i];
                    const double offset = dot(tri.normal, p - tri.v0);
                    if (std::abs(offset) <= tolerance) {
                        visit(static_cast<std::size_t>(order_[i]), offset);
                    }
                }
            } else if (inside) {
                if (top == stack.size()) {
                    throw std::runtime_error("Bvh: traversal stack overflow");
                }
                stack[top++] = node.first + 1;
                current = node.first;
                continue;
            }
            if (top == 0) return;
            current = stack[--top];
        }
    }

    // Of p, a point of a face's plane: its barycentric coordinates, each 0 on one edge (at:
    // edge 0 runs from the face's first corner to its third, edge 1 from the first to the
    // second, edge 2 from the second to the third, see corners), their gradients along the
    // plane (gradient), whose lengths are the inverse of the triangle's heights (steep,
    // found only where it may matter), and whether p lies on each edge (on): within
    // `edge_tie` (a distance along the plane) of the edge's line, and within kEdge
    // (barycentric) of the edge, inside or outside it; a caller may decide `on` otherwise
    // (see WaveSolver::edges). `area` false for a face of no area, the rest then unset.
    struct EdgeCoordinates {
        bool area = false;
        double at[3] = {};
        Vec3 gradient[3];
        bool on[3] = {false, false, false};
        double steep(int e) const { return norm(gradient[e]); }
        // How far p lies from edge e's line, along the plane.
        double distance(int e) const { return std::abs(at[e]) / steep(e); }
        // Whether p lies within kEdge (barycentric) of edge e, inside or outside it: where
        // a ray's hit counts as on the face (see hit_triangle).
        bool near(int e) const { return std::abs(at[e]) <= kEdge; }
        // Whether p lies past edge e, outside the face, and not on it.
        bool out(int e) const { return at[e] < 0.0 && !on[e]; }
    };

    EdgeCoordinates edges(std::size_t face, const Vec3 &p, double edge_tie) const {
        EdgeCoordinates c = edge_coordinates(tri_[place_[face]], p);
        for (int e = 0; e < 3 && c.area; ++e) {
            c.on[e] = c.at[e] <= kEdge && std::abs(c.at[e]) <= edge_tie * c.steep(e);
        }
        return c;
    }

    // The two ends of edge e of a face (see EdgeCoordinates).
    std::array<Vec3, 2> edge_ends(std::size_t face, int e) const {
        const std::array<Vec3, 3> v = corners(face);
        return e == 0
                   ? std::array<Vec3, 2>{v[0], v[2]}
                   : (e == 1 ? std::array<Vec3, 2>{v[0], v[1]} : std::array<Vec3, 2>{v[1], v[2]});
    }

    // Whether the foot of p on a face's plane lies in the face or on one of its edges: past
    // none of them (see EdgeCoordinates).
    static bool holds_foot(const EdgeCoordinates &c) {
        return c.area && !c.out(0) && !c.out(1) && !c.out(2);
    }

    // The same, with `on` within `edge_tie` of each edge's line (see edges).
    bool holds_foot(std::size_t face, const Vec3 &p, double edge_tie) const {
        return holds_foot(edges(face, p, edge_tie));
    }

    // Whether a face holds the point p + e_0 d[0] + e_1 d[1] + ... of its plane, for
    // infinitesimal e_0 >> e_1 >> ... > 0 (p's foot, with its coordinates c, the d's along
    // the plane): where the foot lies on an edge (see EdgeCoordinates), the first d that
    // leaves the edge's line says whether the point goes into the face or out of it; along
    // all of them the point stays on the edge, and is held. A foot inside the face and on
    // none of its edges is held; one outside it and on none of its edges is not (the face
    // is not widened here as a ray's hit widens it).
    // A d whose part across an edge is within 1e-9 of the sizes of its products (rounding)
    // is taken as along that edge; so is one whose part across an edge one of `variants`
    // sets of the d's (count each, one after another, from slightly other inputs) turns
    // round, or makes as large.
    static bool holds_towards(const EdgeCoordinates &c, const Vec3 *d, std::size_t count,
                              const Vec3 *variants = nullptr, std::size_t many = 0) {
        if (!holds_foot(c)) return false;
        for (int e = 0; e < 3; ++e) {
            if (!c.on[e]) continue;
            for (std::size_t i = 0; i < count; ++i) {
                const double length = norm(d[i]);
                if (!(length > 0.0)) continue;
                const Vec3 &grad = c.gradient[e];
                const double g = dot(grad, d[i]);
                // Rounding: 1e-9 of the sum of the products' sizes (of the parts of d across
                // the edge, not of all of d: a step can be large along the edge).
                double spread = 1e-9 * (std::abs(grad.x * d[i].x) + std::abs(grad.y * d[i].y) +
                                        std::abs(grad.z * d[i].z));
                for (std::size_t v = 0; v < many; ++v) {
                    spread =
                        std::max(spread, std::abs(dot(c.gradient[e], variants[v * count + i]) - g));
                }
                if (std::abs(g) <= spread) continue;  // along the edge
                if (g < 0.0) return false;
                break;
            }
        }
        return true;
    }

    // Whether p lies on one of the face's edges (see EdgeCoordinates).
    static bool near_edge(const EdgeCoordinates &c) {
        return c.area && (c.on[0] || c.on[1] || c.on[2]);
    }

    // Whether holds_towards holds p whatever the d's: its foot inside the face and on none
    // of its edges.
    static bool holds_inside(const EdgeCoordinates &c) { return holds_foot(c) && !near_edge(c); }

    // The edges of a face that p, a point of its plane, lies on (see EdgeCoordinates): for
    // each, the unit vector along the plane from the edge away from the face, in out; how
    // many.
    static std::size_t edges_out(const EdgeCoordinates &c, Vec3 *out) {
        std::size_t count = 0;
        for (int e = 0; e < 3 && c.area; ++e) {
            if (c.on[e]) out[count++] = -c.gradient[e] / c.steep(e);
        }
        return count;
    }

    // The tree as flat arrays, for tracing elsewhere (on a GPU), through
    // emit(name, std::vector): per node its box ("boxes": lo then hi), "first" and
    // "count" as in Node; per place in leaf order the face there ("faces") and its first
    // corner and two edges ("triangles": v0, e1, e2).
    template <typename Emit>
    void layout(Emit emit) const {
        std::vector<double> boxes(6 * nodes_.size());
        std::vector<std::uint32_t> first(nodes_.size()), count(nodes_.size());
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const Node &node = nodes_[i];
            for (int a = 0; a < 3; ++a) {
                boxes[6 * i + static_cast<std::size_t>(a)] = node.box.lo[a];
                boxes[6 * i + 3 + static_cast<std::size_t>(a)] = node.box.hi[a];
            }
            first[i] = node.first;
            count[i] = node.count;
        }
        std::vector<double> triangles(9 * tri_.size());
        for (std::size_t i = 0; i < tri_.size(); ++i) {
            const Triangle &t = tri_[i];
            const Vec3 corner_and_edges[3] = {t.v0, t.e1, t.e2};
            for (std::size_t k = 0; k < 3; ++k) {
                for (int a = 0; a < 3; ++a) {
                    triangles[9 * i + 3 * k + static_cast<std::size_t>(a)] = corner_and_edges[k][a];
                }
            }
        }
        emit("boxes", boxes);
        emit("first", first);
        emit("count", count);
        emit("faces", order_);
        emit("triangles", triangles);
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
        // Half the surface area; 0 for an empty box.
        double area() const {
            const Vec3 d = hi - lo;
            if (!(d.x >= 0.0)) return 0.0;
            return d.x * d.y + d.y * d.z + d.z * d.x;
        }
    };
    struct Node {
        Box box;
        std::uint32_t first = 0;  // leaf: first index into tri_; inner: left child, then right
        std::uint32_t count = 0;  // leaf: number of faces; inner: 0
    };

    // Most faces in a leaf, unless they cannot be told apart; bins of the split search;
    // deepest node, which keeps the traversal stack within its size.
    static constexpr std::size_t kLeafSize = 4;
    static constexpr std::size_t kBins = 16;
    static constexpr std::size_t kMaxDepth = 60;

    std::vector<Triangle> tri_;          // in leaf order
    std::vector<std::uint32_t> order_;   // face at each place in tri_
    std::vector<std::uint32_t> place_;   // place in tri_ of each face
    std::vector<Node> nodes_;
    double scale_ = 1.0;
    Vec3 lower_, upper_;  // corners of the model's bounding box
    BvhSplit split_ = BvhSplit::kSah;

    // Splits faces order_[begin, end) at the binned surface area heuristic's best plane,
    // or makes them a leaf when testing them all is cheaper (one box test costing about
    // one triangle test); or, with BvhSplit::kMedian, at the middle face along the
    // longest side of their centres, leaves of up to kLeafSize.
    void build(std::size_t index, std::size_t begin, std::size_t end, std::size_t depth,
               const std::vector<Box> &boxes, const std::vector<Vec3> &centroids) {
        Box box, cbox;
        for (std::size_t i = begin; i < end; ++i) {
            box.grow(boxes[order_[i]]);
            cbox.grow(Box{centroids[order_[i]], centroids[order_[i]]});
        }
        nodes_[index].box = box;
        nodes_[index].first = static_cast<std::uint32_t>(begin);
        nodes_[index].count = static_cast<std::uint32_t>(end - begin);
        const std::size_t n = end - begin;
        if (n <= 1 || depth >= kMaxDepth) return;
        const Vec3 extent = cbox.hi - cbox.lo;
        if (split_ == BvhSplit::kMedian) {
            int axis = 0;
            if (extent.y > extent.x) axis = 1;
            if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;
            if (n <= kLeafSize || extent[axis] <= 0.0) return;
            const std::size_t mid = begin + n / 2;
            std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                             order_.begin() + static_cast<std::ptrdiff_t>(mid),
                             order_.begin() + static_cast<std::ptrdiff_t>(end),
                             [&](std::uint32_t a, std::uint32_t b) {
                                 return centroids[a][axis] < centroids[b][axis];
                             });
            return split_at(index, begin, mid, end, depth, boxes, centroids);
        }
        const auto bin = [&](std::uint32_t face, int axis) {
            const double k = static_cast<double>(kBins) / extent[axis];
            return std::min(kBins - 1,
                            static_cast<std::size_t>((centroids[face][axis] - cbox.lo[axis]) * k));
        };
        double best = std::numeric_limits<double>::infinity();
        int best_axis = -1;
        std::size_t best_bin = 0;
        for (int axis = 0; axis < 3; ++axis) {
            if (!(extent[axis] > 0.0)) continue;
            std::array<Box, kBins> bin_box{};
            std::array<std::size_t, kBins> bin_count{};
            for (std::size_t i = begin; i < end; ++i) {
                const std::size_t b = bin(order_[i], axis);
                ++bin_count[b];
                bin_box[b].grow(boxes[order_[i]]);
            }
            // Area and count of everything above each plane, then below.
            std::array<double, kBins> above_area{};
            std::array<std::size_t, kBins> above_count{};
            Box sweep;
            std::size_t count = 0;
            for (std::size_t b = kBins - 1; b > 0; --b) {
                sweep.grow(bin_box[b]);
                count += bin_count[b];
                above_area[b] = sweep.area();
                above_count[b] = count;
            }
            sweep = Box{};
            count = 0;
            for (std::size_t b = 0; b + 1 < kBins; ++b) {
                sweep.grow(bin_box[b]);
                count += bin_count[b];
                if (count == 0 || above_count[b + 1] == 0) continue;
                const double cost = sweep.area() * static_cast<double>(count) +
                                    above_area[b + 1] * static_cast<double>(above_count[b + 1]);
                if (cost < best) {
                    best = cost;
                    best_axis = axis;
                    best_bin = b;
                }
            }
        }
        if (best_axis < 0) return;  // all centroids in one point
        const double split_cost = box.area() + best;
        if (n <= kLeafSize && split_cost >= box.area() * static_cast<double>(n)) return;
        const auto split = std::partition(
            order_.begin() + static_cast<std::ptrdiff_t>(begin),
            order_.begin() + static_cast<std::ptrdiff_t>(end),
            [&](std::uint32_t face) { return bin(face, best_axis) <= best_bin; });
        split_at(index, begin, static_cast<std::size_t>(split - order_.begin()), end, depth,
                 boxes, centroids);
    }

    // Makes node `index` an inner node over faces order_[begin, mid) and [mid, end).
    void split_at(std::size_t index, std::size_t begin, std::size_t mid, std::size_t end,
                  std::size_t depth, const std::vector<Box> &boxes,
                  const std::vector<Vec3> &centroids) {
        const std::size_t left = nodes_.size();
        nodes_.emplace_back();
        nodes_.emplace_back();
        nodes_[index].first = static_cast<std::uint32_t>(left);
        nodes_[index].count = 0;
        build(left, begin, mid, depth + 1, boxes, centroids);
        build(left + 1, mid, end, depth + 1, boxes, centroids);
    }

    // Where the ray enters the box, if it does before t_max; else -1 (not infinity, which
    // would pass for an entry when t_max is infinite too). A ray lying in a slab's
    // boundary gives 0 * inf = NaN there, which the comparisons drop (as fmin and fmax
    // would, but those are library calls on most compilers).
    static double slab(const Box &b, const Vec3 &origin, const Vec3 &inv, double t_max) {
        double t0 = 0.0, t1 = t_max;
        for (int a = 0; a < 3; ++a) {
            const double ta = (b.lo[a] - origin[a]) * inv[a];
            const double tb = (b.hi[a] - origin[a]) * inv[a];
            const double t_in = ta < tb ? ta : tb;   // tb if either is NaN
            const double t_out = ta < tb ? tb : ta;  // ta if either is NaN
            t0 = t_in > t0 ? t_in : t0;
            t1 = t_out < t1 ? t_out : t1;
        }
        return t0 <= t1 ? t0 : -1.0;
    }

    // Moller-Trumbore, two-sided; returns t or a negative value for a miss. Edges are
    // widened by kEdge (barycentric) so that a ray through a shared edge or vertex hits
    // a face on one side or the other instead of slipping between them.
    static constexpr double kEdge = 1e-9;

    static double hit_triangle(const Triangle &tri, const Vec3 &origin, const Vec3 &dir) {
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

    // Of p, a point of a triangle's plane: its coordinates (see EdgeCoordinates), `on` unset.
    static EdgeCoordinates edge_coordinates(const Triangle &tri, const Vec3 &p) {
        EdgeCoordinates c;
        const Vec3 h = cross(tri.normal, tri.e2);
        const double det = dot(tri.e1, h);
        if (std::abs(det) < 1e-300) return c;
        const double inv_det = 1.0 / det;
        const Vec3 gu = h * inv_det, gv = cross(tri.e1, tri.normal) * inv_det;
        const Vec3 s = p - tri.v0;
        const double u = dot(s, gu), v = dot(s, gv);
        c.area = true;
        c.at[0] = u;
        c.at[1] = v;
        c.at[2] = 1.0 - u - v;
        c.gradient[0] = gu;
        c.gradient[1] = gv;
        c.gradient[2] = -(gu + gv);
        return c;
    }

    // Calls visit(face, t) for faces hit with t in (t_min, limit); visit returns true to
    // stop. `limit` shrinks as the caller records closer hits. Of two children the ray
    // enters, the nearer goes first; the other waits on the stack and is dropped if a
    // hit closer than its box turns up meanwhile.
    template <typename Visit>
    void traverse(const Vec3 &origin, const Vec3 &dir, std::int64_t ignore, Visit visit,
                  double &limit) const {
        const double t_min = min_distance();
        const Vec3 inv{1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z};
        struct Pending {
            std::uint32_t node;
            double entry;
        };
        // Not zeroed: only stack[0, top) is read, each written by a push in this call.
        std::array<Pending, kMaxDepth + 4> stack;
        std::size_t top = 0;
        if (!(slab(nodes_[0].box, origin, inv, limit) >= 0.0)) return;
        std::uint32_t current = 0;
        for (;;) {
            const Node &node = nodes_[current];
            if (node.count > 0) {
                for (std::uint32_t i = node.first; i < node.first + node.count; ++i) {
                    const std::uint32_t face = order_[i];
                    if (static_cast<std::int64_t>(face) == ignore) continue;
                    const double t = hit_triangle(tri_[i], origin, dir);
                    if (t > t_min && t < limit && visit(face, t)) return;
                }
            } else {
                const std::uint32_t left = node.first, right = node.first + 1;
                const double to_left = slab(nodes_[left].box, origin, inv, limit);
                const double to_right = slab(nodes_[right].box, origin, inv, limit);
                const bool in_left = to_left >= 0.0, in_right = to_right >= 0.0;
                if (in_left && in_right) {
                    if (top == stack.size()) {
                        throw std::runtime_error("Bvh: traversal stack overflow");
                    }
                    const bool left_first = to_left <= to_right;
                    stack[top++] = left_first ? Pending{right, to_right} : Pending{left, to_left};
                    current = left_first ? left : right;
                    continue;
                }
                if (in_left || in_right) {
                    current = in_left ? left : right;
                    continue;
                }
            }
            do {
                if (top == 0) return;
                --top;
            } while (!(stack[top].entry <= limit));
            current = stack[top].node;
        }
    }
};

}  // namespace acoustic_spaces
