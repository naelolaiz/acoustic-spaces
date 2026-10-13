// SPDX-License-Identifier: GPL-3.0-or-later
//
// Minimal 3-D vector for the geometric acoustics kernels.

#pragma once

#include <algorithm>
#include <cmath>

namespace acoustic_spaces {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
};

inline double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double norm(const Vec3 &a) { return std::sqrt(dot(a, a)); }

inline Vec3 normalized(const Vec3 &a) {
    const double n = norm(a);
    return n > 0.0 ? a / n : a;
}

inline Vec3 min(const Vec3 &a, const Vec3 &b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

inline Vec3 max(const Vec3 &a, const Vec3 &b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

// Mirror direction d about a plane with unit normal n.
inline Vec3 reflect(const Vec3 &d, const Vec3 &n) { return d - n * (2.0 * dot(d, n)); }

}  // namespace acoustic_spaces
