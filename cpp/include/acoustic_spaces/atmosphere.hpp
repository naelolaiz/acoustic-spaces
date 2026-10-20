// SPDX-License-Identifier: GPL-3.0-or-later
//
// Rays bent by an outdoor atmosphere, as acoustic_spaces/engine/refraction.py has them:
// the speed of sound in still air and the wind's speed depend on the height above the
// ground only, linear between nodes, constant above the last node and below the
// ground. Along a horizontal direction e a ray travels at the effective speed
// c(z) = c_T(z) + u(z) (e . w), w the way the wind blows. In each layer between two
// nodes it is then an arc of a circle in its vertical plane: it keeps its ray parameter
// p = cos(elevation) / c (Snell's law), and its elevation turns at -p dc/dz per metre.
//
// The tracer follows such a ray in chords, straight segments it tests against the
// BVH: each one stays within `sagitta` of the arc (a curve that turns by an angle phi
// over a length s strays at most s phi / 4 from its chord).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "acoustic_spaces/vec3.hpp"

namespace acoustic_spaces {

namespace detail {

// sin(x) / x, 1 at 0.
inline double sinc(double x) {
    return std::abs(x) < 1e-4 ? 1.0 - x * x / 6.0 : std::sin(x) / x;
}

// log(1 + x) / x, 1 at 0.
inline double log_ratio(double x) {
    return std::abs(x) < 1e-8 ? 1.0 - 0.5 * x : std::log1p(x) / x;
}

}  // namespace detail

struct Atmosphere {
    std::vector<double> height;  // m above the ground: 0 first, then increasing
    std::vector<double> still;   // m/s: the speed of sound in still air at each height
    std::vector<double> wind;    // m/s: the wind's speed at each height
    double wind_x = 1.0, wind_y = 0.0;  // the horizontal unit vector the wind blows along
    double ground = 0.0;                // z of the ground (m)
    double sagitta = 0.02;              // m: how far a chord may stray from its arc

    // Filled by prepare(), per bucket of the wind's cosine to a direction (one without
    // wind), at each node: the horizontal distance (m) from where the ray that grazes
    // the ground touches it to the node's height, infinite above where that ray turns
    // back down (or where the speed never falls below the ground's: no shadow); and the
    // integral of dz / c from the ground up (s / m * m).
    std::size_t buckets = 0;
    std::vector<double> reach, slow;  // buckets x nodes

    static constexpr std::size_t kWindBuckets = 401;

    bool active() const { return height.size() >= 2; }
    std::size_t nodes() const { return height.size(); }

    bool valid() const {
        const std::size_t n = nodes();
        if (n < 2 || still.size() != n || wind.size() != n || height[0] != 0.0 ||
            !(sagitta > 0.0) || !(std::abs(std::hypot(wind_x, wind_y) - 1.0) < 1e-6)) {
            return false;
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0 && !(height[i] > height[i - 1])) return false;
            if (!(still[i] > 0.0) || !std::isfinite(wind[i])) return false;
            if (!(still[i] - std::abs(wind[i]) > 0.0)) return false;
        }
        return true;
    }

    // The cosine of the wind to the horizontal direction (x, y) (unit, or zero).
    double along(double x, double y) const { return x * wind_x + y * wind_y; }

    // The effective speed at node i along a direction at `along` to the wind.
    double speed(std::size_t i, double along) const { return still[i] + wind[i] * along; }

    // The effective speed at z (m above the ground).
    double speed_at(double z, double along) const {
        const std::size_t n = nodes();
        if (!(z > 0.0)) return speed(0, along);
        if (z >= height[n - 1]) return speed(n - 1, along);
        const auto i = static_cast<std::size_t>(
            std::upper_bound(height.begin(), height.end(), z) - height.begin() - 1);
        const double t = (z - height[i]) / (height[i + 1] - height[i]);
        return speed(i, along) + t * (speed(i + 1, along) - speed(i, along));
    }

    // The time (s) sound takes along the straight line from a to b.
    double straight_time(const Vec3 &a, const Vec3 &b) const {
        const Vec3 v = b - a;
        const double length = norm(v);
        if (!(length > 0.0)) return 0.0;
        const double flat = std::hypot(v.x, v.y);
        const std::size_t m = bucket(flat > 0.0 ? along(v.x / flat, v.y / flat) : 0.0);
        const double za = a.z - ground, zb = b.z - ground;
        if (std::abs(zb - za) <= 1e-9 * length) {
            return length / speed_at(0.5 * (za + zb), cosine(m));
        }
        return length * (slowness_at(m, zb) - slowness_at(m, za)) / (zb - za);
    }

    void prepare() {
        const std::size_t n = nodes();
        const bool windy = std::any_of(wind.begin(), wind.end(), [](double u) { return u != 0.0; });
        buckets = windy ? kWindBuckets : 1;
        const double inf = std::numeric_limits<double>::infinity();
        reach.assign(buckets * n, inf);
        slow.assign(buckets * n, 0.0);
        for (std::size_t m = 0; m < buckets; ++m) {
            const double along = cosine(m);
            double *time = slow.data() + m * n;
            for (std::size_t i = 0; i + 1 < n; ++i) {
                const double ca = speed(i, along), cb = speed(i + 1, along);
                time[i + 1] =
                    time[i] + (height[i + 1] - height[i]) * detail::log_ratio((cb - ca) / ca) / ca;
            }
            const double c0 = speed(0, along), p = 1.0 / c0;
            double *row = reach.data() + m * n;
            row[0] = 0.0;
            double x = 0.0;
            for (std::size_t i = 0; i + 1 < n; ++i) {
                const double ca = speed(i, along), cb = speed(i + 1, along);
                // The grazing ray turns back down before the next node, or never rises.
                if (!(cb < c0)) break;
                const double sa = std::sqrt(std::max(1.0 - (p * ca) * (p * ca), 0.0));
                const double sb = std::sqrt(std::max(1.0 - (p * cb) * (p * cb), 0.0));
                x += p * (ca + cb) * (height[i + 1] - height[i]) / (sa + sb);
                row[i + 1] = x;
            }
        }
    }

    // Whether b lies in the shadow of upward refraction as seen from a: past where the
    // ray that grazes the ground could take sound from one to the other.
    bool shadowed(const Vec3 &a, const Vec3 &b) const {
        if (buckets == 0) return false;
        const double dx = b.x - a.x, dy = b.y - a.y;
        const double flat = std::hypot(dx, dy);
        if (!(flat > 0.0)) return false;
        const std::size_t m = bucket(along(dx / flat, dy / flat));
        return flat > reach_at(m, a.z - ground) + reach_at(m, b.z - ground);
    }

private:
    // The bucket of the wind's cosine `along`, and the cosine of bucket m.
    std::size_t bucket(double along) const {
        if (buckets < 2) return 0;
        const double t = 0.5 * (std::clamp(along, -1.0, 1.0) + 1.0);
        return static_cast<std::size_t>(std::lround(t * static_cast<double>(buckets - 1)));
    }
    double cosine(std::size_t m) const {
        if (buckets < 2) return 0.0;
        return -1.0 + 2.0 * static_cast<double>(m) / static_cast<double>(buckets - 1);
    }

    // The integral of dz / c from the ground up to z (negative below it), in bucket m.
    double slowness_at(std::size_t m, double z) const {
        const std::size_t n = nodes();
        const double along = cosine(m);
        if (!(z > 0.0)) return z / speed(0, along);
        const double *time = slow.data() + m * n;
        if (z >= height[n - 1]) return time[n - 1] + (z - height[n - 1]) / speed(n - 1, along);
        const auto i = static_cast<std::size_t>(
            std::upper_bound(height.begin(), height.end(), z) - height.begin() - 1);
        const double ca = speed(i, along);
        const double cz = ca + (speed(i + 1, along) - ca) * (z - height[i]) /
                                   (height[i + 1] - height[i]);
        return time[i] + (z - height[i]) * detail::log_ratio((cz - ca) / ca) / ca;
    }

    double reach_at(std::size_t m, double z) const {
        const std::size_t n = nodes();
        const double *row = reach.data() + m * n;
        if (!(z > 0.0)) return 0.0;
        if (z >= height[n - 1]) return row[n - 1];
        const auto i = static_cast<std::size_t>(
            std::upper_bound(height.begin(), height.end(), z) - height.begin() - 1);
        if (std::isinf(row[i + 1])) return row[i + 1];
        const double t = (z - height[i]) / (height[i + 1] - height[i]);
        return row[i] + t * (row[i + 1] - row[i]);
    }
};

// A ray bent by the atmosphere: where it is, which way it goes, and its arc through the
// layers.
class BentRay {
public:
    struct Chord {
        Vec3 start, end;
        double arc = 0.0;   // m along the arc
        double time = 0.0;  // s along the arc
        // The chord stops where the ray turns back towards the plane it left (`away`).
        bool turned = false;
    };

    BentRay(const Atmosphere &air, const Vec3 &pos, const Vec3 &dir) : air_(&air), pos_(pos) {
        const double flat = std::hypot(dir.x, dir.y);
        if (flat > 0.0) {
            ex_ = dir.x / flat;
            ey_ = dir.y / flat;
        }
        along_ = air.along(ex_, ey_);
        theta_ = std::atan2(dir.z, flat);
        z_ = pos.z - air.ground;
        p_ = std::cos(theta_) / air.speed_at(z_, along_);
        const std::size_t n = air.nodes();
        int layer = 0;
        if (z_ < 0.0) {
            layer = -1;
        } else if (z_ >= air.height[n - 1]) {
            layer = static_cast<int>(n) - 1;
        } else {
            layer = static_cast<int>(
                std::upper_bound(air.height.begin(), air.height.end(), z_) - air.height.begin() - 1);
        }
        if (layer >= 0 && z_ == air.height[static_cast<std::size_t>(layer)] && theta_ < 0.0) {
            --layer;
        }
        enter(layer);
    }

    const Vec3 &position() const { return pos_; }
    Vec3 direction() const {
        const double flat = std::cos(theta_);
        return {flat * ex_, flat * ey_, std::sin(theta_)};
    }
    double speed() const { return speed_in(z_); }

    // Follows the arc for at most `budget` metres, as far as one chord may go. With
    // `away` (the unit normal of a plane the ray leaves, pointing the way it goes), it
    // stops where the ray turns back towards that plane.
    Chord chord(double budget, const Vec3 *away) {
        Chord out;
        out.start = pos_;
        double turn = 0.0;
        for (std::size_t n = 0; n < kMaxPieces && out.arc < budget; ++n) {
            const Piece piece = step(budget - out.arc, out.arc, turn, away);
            out.arc += piece.length;
            out.time += piece.time;
            turn += piece.turn;
            if (piece.turned) {
                out.turned = true;
                break;
            }
            if (!(piece.length > 0.0) || !piece.boundary) break;
        }
        out.end = pos_;
        return out;
    }

private:
    static constexpr std::size_t kMaxPieces = 1 << 16;

    struct Piece {
        double length = 0.0, time = 0.0, turn = 0.0;
        bool boundary = false, turned = false;
    };

    const Atmosphere *air_;
    Vec3 pos_;
    double ex_ = 0.0, ey_ = 0.0, along_ = 0.0;
    double theta_ = 0.0, p_ = 0.0, z_ = 0.0;
    // The layer the ray is in: i between nodes i and i + 1, -1 below the ground, nodes - 1
    // above the last node; its bottom and top, and the speed there as c + g (z - base).
    int layer_ = 0;
    double lo_ = 0.0, hi_ = 0.0, base_ = 0.0, c_ = 0.0, g_ = 0.0;

    void enter(int i) {
        const double inf = std::numeric_limits<double>::infinity();
        const auto n = static_cast<int>(air_->nodes());
        layer_ = i;
        g_ = 0.0;
        if (i < 0) {
            lo_ = -inf;
            hi_ = base_ = 0.0;
            c_ = air_->speed(0, along_);
        } else if (i + 1 >= n) {
            lo_ = base_ = air_->height[static_cast<std::size_t>(n - 1)];
            hi_ = inf;
            c_ = air_->speed(static_cast<std::size_t>(n - 1), along_);
        } else {
            const auto k = static_cast<std::size_t>(i);
            lo_ = base_ = air_->height[k];
            hi_ = air_->height[k + 1];
            c_ = air_->speed(k, along_);
            g_ = (air_->speed(k + 1, along_) - c_) / (hi_ - lo_);
        }
    }
    double speed_in(double z) const { return c_ + g_ * (z - base_); }

    // One piece of arc in the current layer, at most `limit` metres, within the chord's
    // tolerance after `arc` metres and `turn` radians of it.
    Piece step(double limit, double arc, double turn, const Vec3 *away) {
        const double inf = std::numeric_limits<double>::infinity();
        double kappa = 0.0, length = inf, theta_end = theta_;
        int next = layer_;
        // Where the ray leaves the layer. A ray that only just touches a node's height
        // goes on in the layer it heads into; one that neither layer takes (level on the
        // axis of a duct) goes on level.
        for (int attempt = 0; attempt < 3; ++attempt) {
            kappa = g_ * p_;
            if (std::abs(kappa) * (hi_ - lo_) < 1e-12) kappa = 0.0;
            next = layer_;
            theta_end = theta_;
            length = inf;
            if (kappa == 0.0) {
                const double sine = std::sin(theta_);
                if (sine > 0.0 && std::isfinite(hi_)) {
                    length = (hi_ - z_) / sine;
                    next = layer_ + 1;
                } else if (sine < 0.0 && std::isfinite(lo_)) {
                    length = (lo_ - z_) / sine;
                    next = layer_ - 1;
                }
            } else {
                const bool up = theta_ > 0.0 || (theta_ == 0.0 && kappa < 0.0);
                const double q_hi = p_ * speed_in(hi_), q_lo = p_ * speed_in(lo_);
                if (up && q_hi < 1.0) {
                    theta_end = std::acos(q_hi);  // on up into the next layer
                    next = layer_ + 1;
                } else if (up) {
                    theta_end = -std::acos(std::min(q_lo, 1.0));  // turns back down
                    next = layer_ - 1;
                } else if (q_lo < 1.0) {
                    theta_end = -std::acos(q_lo);  // on down into the layer below
                    next = layer_ - 1;
                } else {
                    theta_end = std::acos(std::min(q_hi, 1.0));  // turns back up
                    next = layer_ + 1;
                }
                length = (theta_ - theta_end) / kappa;
            }
            if (length > 0.0) break;
            enter(next);
            if (attempt == 2) {
                kappa = 0.0;
                length = inf;
                theta_end = theta_;
                next = layer_;
            }
        }

        // The chord's tolerance: (arc + s) (turn + |kappa| s) <= 4 sagitta.
        const double room = std::max(4.0 * air_->sagitta - arc * turn, 0.0);
        const double k = std::abs(kappa);
        double tolerance = inf;
        if (k > 0.0) {
            const double b = turn + k * arc;
            tolerance = 2.0 * room / (b + std::sqrt(b * b + 4.0 * k * room));
        } else if (turn > 0.0) {
            tolerance = room / turn;
        }
        Piece piece;
        double s = std::min({length, limit, tolerance});
        piece.boundary = s == length && std::isfinite(length);
        // Where the ray turns back towards the plane it left.
        if (away != nullptr && kappa != 0.0 && away->z != 0.0) {
            const double a = away->x * ex_ + away->y * ey_, nz = away->z;
            const auto facing = [&](double theta) {
                return a * std::cos(theta) + nz * std::sin(theta);
            };
            const double theta_s = piece.boundary ? theta_end : theta_ - kappa * s;
            if (facing(theta_) > 0.0 && facing(theta_s) <= 0.0) {
                const double at = (theta_ - std::atan(-a / nz)) / kappa;
                if (at <= s) {
                    s = std::max(at, 0.0);
                    piece.boundary = false;
                    piece.turned = true;
                }
            }
        }
        if (!(s > 0.0)) return piece;
        const double theta_new = piece.boundary ? theta_end : theta_ - kappa * s;
        const double phi = theta_ - theta_new;
        const double chord = s * detail::sinc(0.5 * phi);
        const double mean = 0.5 * (theta_ + theta_new);
        const double flat = chord * std::cos(mean);
        const double rise = chord * std::sin(mean);
        double z_end = z_ + rise;
        if (piece.boundary) z_end = next > layer_ ? hi_ : lo_;
        // The travel time by Simpson's rule (the speed is linear in height in a layer).
        double mid = 0.5 * (z_ + z_end);
        if (kappa != 0.0) {
            mid = z_ + 0.5 * s * detail::sinc(0.25 * phi) * std::sin(theta_ - 0.25 * phi);
        }
        piece.length = s;
        piece.time = s / 6.0 * (1.0 / speed_in(z_) + 4.0 / speed_in(mid) + 1.0 / speed_in(z_end));
        piece.turn = std::abs(phi);
        pos_ = pos_ + Vec3{flat * ex_, flat * ey_, rise};
        theta_ = theta_new;
        z_ = z_end;
        if (piece.boundary) enter(next);
        return piece;
    }
};

}  // namespace acoustic_spaces
