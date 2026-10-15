// SPDX-License-Identifier: GPL-3.0-or-later
//
// Geometric acoustics kernels on top of the BVH:
//
// - specular_sequences: which faces deterministic specular rays hit, in order. The
//   early-reflection finder turns these into image sources and checks them exactly.
// - trace_stochastic: energy-based stochastic ray tracing with diffuse rain, giving
//   energy per receiver, octave band and time slot (and, optionally, per arrival
//   direction) for the late part of the response, and optionally over a coverage map.
//
// Units: a source of unit energy; results are time-integrated intensity, so the
// direct sound at distance r contributes 1 / (4 pi r^2).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "acoustic_spaces/bvh.hpp"
#include "acoustic_spaces/vec3.hpp"

namespace acoustic_spaces {

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kMaxBands = 32;

// xoshiro256** seeded through splitmix64: one independent stream per ray, so results
// do not depend on how rays are split between threads.
class Rng {
public:
    Rng(std::uint64_t seed, std::uint64_t stream) {
        std::uint64_t x = seed ^ (0x9E3779B97F4A7C15ULL * (stream + 1));
        for (auto &s : state_) {
            x += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s = z ^ (z >> 31);
        }
    }

    // Uniform in [0, 1).
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

    // A generator for a new, independent stream (for a ray split off this one).
    Rng split() { return Rng(next(), next()); }

private:
    std::array<std::uint64_t, 4> state_{};

    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    std::uint64_t next() {
        const std::uint64_t result = rotl(state_[1] * 5, 7) * 9;
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }
};

inline Vec3 uniform_sphere(Rng &rng) {
    const double z = 1.0 - 2.0 * rng.uniform();
    const double phi = 2.0 * kPi * rng.uniform();
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    return {r * std::cos(phi), r * std::sin(phi), z};
}

// Point i of n on a Fibonacci lattice: near-uniform, deterministic directions.
inline Vec3 fibonacci_sphere(std::size_t i, std::size_t n) {
    const double golden = kPi * (3.0 - std::sqrt(5.0));
    const double z = 1.0 - (2.0 * static_cast<double>(i) + 1.0) / static_cast<double>(n);
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double phi = golden * static_cast<double>(i);
    return {r * std::cos(phi), r * std::sin(phi), z};
}

// Lambertian (cosine-weighted) direction on the hemisphere around unit normal n.
inline Vec3 cosine_hemisphere(const Vec3 &n, Rng &rng) {
    const double u1 = rng.uniform();
    const double phi = 2.0 * kPi * rng.uniform();
    const double r = std::sqrt(u1);
    const Vec3 helper = std::abs(n.x) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
    const Vec3 t = normalized(cross(n, helper));
    const Vec3 b = cross(n, t);
    return normalized(t * (r * std::cos(phi)) + b * (r * std::sin(phi)) +
                      n * std::sqrt(std::max(0.0, 1.0 - u1)));
}

// The scattering coefficients (bands) of a reflection off a surface `width` by `length`
// metres, with what its limited size scatters added to the surface's own (Christensen
// and Rindel, 2005): below f_w = c d / (2 (w cos)^2) the specular share falls as f /
// f_w, and below f_l = c d / (2 l^2) as f / f_l too, d the distance the sound travelled
// to the surface and cos the cosine of its angle of incidence.
inline void size_scattering(double width, double length, double distance, double cosine,
                            double speed_of_sound, std::size_t bands, const double *band_hz,
                            const double *surface, double *out) {
    const double across = width * std::max(cosine, 1e-3);
    const double f_width = speed_of_sound * distance / (2.0 * across * across);
    const double f_length = speed_of_sound * distance / (2.0 * length * length);
    for (std::size_t b = 0; b < bands; ++b) {
        const double specular =
            std::min(1.0, band_hz[b] / f_width) * std::min(1.0, band_hz[b] / f_length);
        out[b] = 1.0 - (1.0 - surface[b]) * specular;
    }
}

// How a source shares its power between directions: energy weights per band that
// average 1 over the sphere. A first-order pattern a + (1 - a) cos(theta) per band (none:
// omni), times a table of weights on a grid of elevations (-90 to 90 degrees) and
// azimuths (0 up to 360) in the source's frame, read bilinearly (none: 1).
struct Directivity {
    std::vector<double> a;      // bands; empty for omni
    std::vector<double> table;  // elevations x azimuths x bands; may be empty
    std::size_t elevations = 0, azimuths = 0;
    Vec3 forward{1.0, 0.0, 0.0}, left{0.0, 1.0, 0.0}, up{0.0, 0.0, 1.0};

    void energy_weights(const Vec3 &dir, std::size_t bands, double *out) const {
        const double x = dot(dir, forward);
        for (std::size_t b = 0; b < bands; ++b) {
            if (a.empty()) {
                out[b] = 1.0;
                continue;
            }
            const double p = a[b] + (1.0 - a[b]) * x;
            out[b] = p * p / (a[b] * a[b] + (1.0 - a[b]) * (1.0 - a[b]) / 3.0);
        }
        if (table.empty()) return;
        const double el = std::asin(std::clamp(dot(dir, up), -1.0, 1.0)) * 180.0 / kPi;
        double az = std::atan2(dot(dir, left), x) * 180.0 / kPi;
        if (az < 0.0) az += 360.0;
        const double fi = (el + 90.0) / (180.0 / static_cast<double>(elevations - 1));
        const double fj = az / (360.0 / static_cast<double>(azimuths));
        const auto i0 = static_cast<std::size_t>(
            std::clamp(std::floor(fi), 0.0, static_cast<double>(elevations - 2)));
        const double floor_j = std::floor(fj);
        const std::size_t j0 = static_cast<std::size_t>(floor_j) % azimuths;
        const std::size_t j1 = (j0 + 1) % azimuths;
        const double ti = fi - static_cast<double>(i0), tj = fj - floor_j;
        const auto at = [&](std::size_t i, std::size_t j, std::size_t b) {
            return table[(i * azimuths + j) * bands + b];
        };
        for (std::size_t b = 0; b < bands; ++b) {
            out[b] *= (1.0 - ti) * ((1.0 - tj) * at(i0, j0, b) + tj * at(i0, j1, b)) +
                      ti * ((1.0 - tj) * at(i0 + 1, j0, b) + tj * at(i0 + 1, j1, b));
        }
    }
};

// Runs body(begin, end, worker) over [0, n) on up to `threads` threads (0: all cores).
template <typename Body>
void parallel_for(std::size_t n, std::size_t threads, Body body) {
    if (threads == 0) {
        threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    threads = std::max<std::size_t>(1, std::min(threads, n));
    if (threads == 1) {
        body(std::size_t{0}, n, std::size_t{0});
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const std::size_t chunk = (n + threads - 1) / threads;
    for (std::size_t w = 0; w < threads; ++w) {
        const std::size_t begin = w * chunk;
        const std::size_t end = std::min(n, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([=, &body] { body(begin, end, w); });
    }
    for (auto &t : pool) t.join();
}

// Face indices hit by `rays` specular rays from `source` (Fibonacci directions), up to
// `max_order` reflections each; rays x max_order, -1 once a ray leaves the model.
inline std::vector<std::int64_t> specular_sequences(const Bvh &bvh, const Vec3 &source,
                                                    std::size_t rays, std::size_t max_order,
                                                    std::size_t threads) {
    std::vector<std::int64_t> out(rays * max_order, -1);
    const double far = 1e6 * bvh.scale();
    parallel_for(rays, threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t i = begin; i < end; ++i) {
            Vec3 pos = source;
            Vec3 dir = fibonacci_sphere(i, rays);
            std::int64_t last = -1;
            for (std::size_t k = 0; k < max_order; ++k) {
                const Hit hit = bvh.intersect(pos, dir, far, last);
                if (!hit.found()) break;
                out[i * max_order + k] = hit.face;
                pos = pos + dir * hit.t;
                dir = reflect(dir, bvh.normal(static_cast<std::size_t>(hit.face)));
                last = hit.face;
            }
        }
    });
    return out;
}

struct StochasticSettings {
    std::size_t rays = 10000;
    double max_time = 2.0;          // s
    double time_step = 1e-3;        // s, energy histogram resolution
    double dir_time_step = 1e-2;    // s, directional histogram resolution
    double window_time_step = 5e-2;  // s, resolution of split_target and ray_energy
    double speed_of_sound = 343.0;  // m/s
    std::vector<double> receiver_radius;  // m, detector sphere per receiver
    // How reflected energy reaches the receivers:
    // - diffuse_rain: the scattered part of every reflection radiates straight to each
    //   visible receiver (Lambert); specular legs are caught by the detector spheres.
    // - specular_rain (with diffuse_rain): the specular part radiates the same way, as
    //   the secondary sources of ODEON's late method; the spheres are then not used.
    // - neither: classic ray tracing, every leg caught by the detector spheres.
    bool diffuse_rain = true;
    bool specular_rain = false;
    // Where a reflected ray goes. Vector-based (ODEON's): in the direction of the
    // specular direction and a Lambert direction added with weights 1 - s and s, s the
    // reflection's scattering coefficient averaged over the bands; the ray keeps all the
    // reflected energy, and the detector spheres count the specular share of it on the
    // next leg (the rain covers the rest). Otherwise random: specularly or, with s as
    // probability, in a Lambert direction, the bands re-weighted so each keeps its own
    // share.
    bool vector_scattering = false;
    // With StochasticScene::surface_size: reflections of this order and higher add the
    // scattering of the surface's limited size (lower orders are left to the early
    // stage's edge diffraction).
    int size_from_order = 1;
    // Specular-only paths with at most this many reflections, arriving at receiver k no
    // later than covered_time[k] (s, from emission), are left out: the early-reflection
    // finder computes those exactly. -1 keeps everything.
    int covered_order = -1;
    std::vector<double> covered_time;  // per receiver; empty: no time limit
    // Rays use random streams first_ray .. first_ray + rays - 1, and each carries
    // 1 / rays of the source energy.
    std::uint64_t first_ray = 0;
    // Weight window (variance reduction): the expected band-mean energy of a ray per
    // window of window_time_step. After a reflection, a ray above split_ratio times it
    // is split into copies that share its energy; one below 1 / split_ratio times it
    // survives with probability energy / expected, with its energy raised to match. Both
    // keep every expectation unchanged; they spend the rays where the energy is. Empty:
    // off. A ray and its copies never number more than max_family.
    std::vector<double> split_target;
    double split_ratio = 4.0;
    std::size_t max_split = 8;
    std::size_t max_family = 256;
    std::uint64_t seed = 0;
    std::size_t threads = 0;
};

// A coverage map: a regular grid of columns in x and y over the audience area. Each
// column holds up to `layers` map points (a balcony above the stalls), each a slab of
// air between low and high (z). Every leg of every ray adds its track length through a
// slab over the slab's volume, a track-length estimate of the energy there, whatever
// the reception mode; the receivers above are not needed.
struct MapGrid {
    double x0 = 0.0, y0 = 0.0;  // corner of column (0, 0), m
    double cell = 1.0;          // column width, m
    std::size_t nx = 0, ny = 0, layers = 1;
    std::vector<std::int64_t> point;    // ny x nx x layers: map point index, -1 for none
    std::vector<double> low, high;      // per point: the slab's bottom and top, m
    std::vector<double> volume;         // per point: the slab's volume inside the room, m^3
    std::vector<double> onset;          // per point: arrival of the direct sound, s
    std::vector<double> covered_time;   // per point, as StochasticSettings::covered_time
    // Time bins count from each point's onset: fine_bins of fine_step, then coarse_step.
    double fine_step = 1e-3, coarse_step = 1e-2;
    std::size_t fine_bins = 100;

    std::size_t points() const { return low.size(); }

    std::size_t bins(double max_time) const {
        const double rest = std::max(0.0, max_time - fine_step * static_cast<double>(fine_bins));
        return fine_bins + static_cast<std::size_t>(std::ceil(rest / coarse_step));
    }

    // Bin of an arrival `after` seconds after the onset (bins() and up: none).
    std::size_t bin(double after) const {
        after = std::max(after, 0.0);
        const double fine = fine_step * static_cast<double>(fine_bins);
        if (after < fine) return static_cast<std::size_t>(after / fine_step);
        return fine_bins + static_cast<std::size_t>((after - fine) / coarse_step);
    }

    bool valid() const {
        const std::size_t n = points();
        if (!(cell > 0.0) || !(fine_step > 0.0) || !(coarse_step > 0.0) || layers == 0 ||
            point.size() != nx * ny * layers || high.size() != n || volume.size() != n ||
            onset.size() != n || covered_time.size() != n) {
            return false;
        }
        for (std::size_t m = 0; m < n; ++m) {
            if (!(high[m] > low[m]) || !(volume[m] > 0.0)) return false;
        }
        return std::all_of(point.begin(), point.end(), [n](std::int64_t m) {
            return m >= -1 && m < static_cast<std::int64_t>(n);
        });
    }
};

struct StochasticScene {
    const Bvh *bvh = nullptr;
    Vec3 source;
    Directivity directivity;
    std::size_t bands = 0;
    const double *absorption = nullptr;  // faces x bands
    const double *scattering = nullptr;  // faces x bands
    const double *air = nullptr;         // bands, energy attenuation per metre
    // Optional: the width and length (m) of the surface of each face (faces x 2) and the
    // centre frequency of each band (Hz), for the scattering of a surface's limited size.
    const double *surface_size = nullptr;
    const double *band_hz = nullptr;
    std::vector<Vec3> receivers;
    std::vector<Vec3> directions;  // unit vectors for directional bins; may be empty
    MapGrid map;                   // no points: no map
};

struct StochasticResult {
    std::size_t bins = 0, dir_bins = 0, windows = 0;
    std::vector<double> energy;       // receivers x bands x bins
    std::vector<double> directional;  // receivers x directions x bands x dir_bins
    // windows: the sum and count of the band-mean energy of rays at each reflection, so
    // ray_energy / reflections is the expected energy of a ray in the window.
    std::vector<double> ray_energy, reflections;
    std::size_t map_bins = 0;
    std::vector<double> map_energy;  // map points x bands x map_bins
    std::size_t escaped_rays = 0;
    std::size_t traced_rays = 0;  // rays including the copies split off
};

namespace detail {

// Everything a ray carries from one leg to the next.
struct RayState {
    Vec3 pos, dir;
    std::array<double, kMaxBands> energy;
    // Per band: the share of the energy the detector spheres count on this leg, and the
    // share that has only ever reflected specularly (which the early stage computes
    // exactly for the paths it covers).
    std::array<double, kMaxBands> seen, alone;
    double travelled;
    int order;
    std::int64_t last;   // face of the last reflection, ignored by the next intersection
    Rng rng;
};

struct Accumulator {
    const StochasticScene &scene;
    const StochasticSettings &settings;
    StochasticResult &out;

    std::size_t nearest_direction(const Vec3 &arrival) const {
        std::size_t best = 0;
        double best_dot = -2.0;
        for (std::size_t d = 0; d < scene.directions.size(); ++d) {
            const double v = dot(arrival, scene.directions[d]);
            if (v > best_dot) {
                best_dot = v;
                best = d;
            }
        }
        return best;
    }

    // Adds energy[b] * factor * air(distance) arriving from `arrival` at `distance`.
    void add(std::size_t receiver, const double *energy, double factor, double distance,
             const Vec3 &arrival) {
        const double time = distance / settings.speed_of_sound;
        const auto bin = static_cast<std::size_t>(time / settings.time_step);
        if (bin >= out.bins) return;
        const std::size_t bands = scene.bands;
        double *row = out.energy.data() + receiver * bands * out.bins;
        double *drow = nullptr;
        std::size_t dbin = 0;
        if (!scene.directions.empty()) {
            dbin = std::min(out.dir_bins - 1,
                            static_cast<std::size_t>(time / settings.dir_time_step));
            const std::size_t d = nearest_direction(arrival);
            drow = out.directional.data() +
                   (receiver * scene.directions.size() + d) * bands * out.dir_bins;
        }
        for (std::size_t b = 0; b < bands; ++b) {
            const double e = energy[b] * factor * std::exp(-scene.air[b] * distance);
            row[b * out.bins + bin] += e;
            if (drow) drow[b * out.dir_bins + dbin] += e;
        }
    }

    // Lowest slab bottom and highest slab top of the map.
    double map_bottom = 0.0, map_top = 0.0;

    // Adds a leg that leaves `pos` along `dir` (unit) for `length` metres, `travelled`
    // metres from the source at `pos`, to every map slab it passes through.
    void add_map(const Vec3 &pos, const Vec3 &dir, double length, double travelled,
                 const double *energy, const double *alone, int order) {
        const MapGrid &map = scene.map;
        constexpr double kFlat = 1e-12;
        const double inf = std::numeric_limits<double>::infinity();
        double t0 = 0.0, t1 = length;
        // Keeps the part of the leg where p + d t lies in [lo, hi].
        const auto clip = [&](double p, double d, double lo, double hi) {
            if (std::abs(d) < kFlat) return p >= lo && p <= hi;
            double a = (lo - p) / d, b = (hi - p) / d;
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
            return t0 < t1;
        };
        const double width = map.cell * static_cast<double>(map.nx);
        const double depth = map.cell * static_cast<double>(map.ny);
        if (!clip(pos.z, dir.z, map_bottom, map_top) ||
            !clip(pos.x, dir.x, map.x0, map.x0 + width) ||
            !clip(pos.y, dir.y, map.y0, map.y0 + depth)) {
            return;
        }
        // Walk the columns the leg crosses (Amanatides and Woo).
        const auto first = [&](double p, double d, double origin, std::size_t n) {
            const double i = std::floor((p + d * t0 - origin) / map.cell);
            return static_cast<std::int64_t>(std::clamp(i, 0.0, static_cast<double>(n - 1)));
        };
        std::int64_t ix = first(pos.x, dir.x, map.x0, map.nx);
        std::int64_t iy = first(pos.y, dir.y, map.y0, map.ny);
        const auto next = [&](double p, double d, double origin, std::int64_t i) {
            if (std::abs(d) < kFlat) return inf;
            const double edge = origin + map.cell * static_cast<double>(i + (d > 0.0 ? 1 : 0));
            return (edge - p) / d;
        };
        double next_x = next(pos.x, dir.x, map.x0, ix);
        double next_y = next(pos.y, dir.y, map.y0, iy);
        const double step_x = std::abs(dir.x) < kFlat ? inf : map.cell / std::abs(dir.x);
        const double step_y = std::abs(dir.y) < kFlat ? inf : map.cell / std::abs(dir.y);
        const std::int64_t sx = dir.x > 0.0 ? 1 : -1, sy = dir.y > 0.0 ? 1 : -1;
        const double c = settings.speed_of_sound;
        const std::size_t bands = scene.bands;
        double t = t0;
        for (std::size_t guard = 0; guard < map.nx + map.ny + 2; ++guard) {
            const double t_out = std::min({next_x, next_y, t1});
            const std::size_t column =
                (static_cast<std::size_t>(iy) * map.nx + static_cast<std::size_t>(ix)) *
                map.layers;
            for (std::size_t layer = 0; layer < map.layers; ++layer) {
                const std::int64_t m = map.point[column + layer];
                if (m < 0) continue;
                const auto k = static_cast<std::size_t>(m);
                double a = t, b = t_out;
                if (std::abs(dir.z) < kFlat) {
                    if (pos.z < map.low[k] || pos.z > map.high[k]) continue;
                } else {
                    double za = (map.low[k] - pos.z) / dir.z, zb = (map.high[k] - pos.z) / dir.z;
                    if (za > zb) std::swap(za, zb);
                    a = std::max(a, za);
                    b = std::min(b, zb);
                }
                if (!(b > a)) continue;
                const double distance = travelled + 0.5 * (a + b);
                const bool covered =
                    order <= settings.covered_order && distance / c <= map.covered_time[k];
                const std::size_t bin = map.bin(distance / c - map.onset[k]);
                if (bin >= out.map_bins) continue;
                const double factor = (b - a) / map.volume[k];
                double *row = out.map_energy.data() + k * bands * out.map_bins;
                for (std::size_t band = 0; band < bands; ++band) {
                    // The specular-only share of a covered path is the early stage's.
                    const double share = covered ? 1.0 - alone[band] : 1.0;
                    row[band * out.map_bins + bin] +=
                        energy[band] * share * factor * std::exp(-scene.air[band] * distance);
                }
            }
            if (t_out >= t1) break;
            t = t_out;
            if (next_x <= next_y) {
                ix += sx;
                next_x += step_x;
                if (ix < 0 || ix >= static_cast<std::int64_t>(map.nx)) break;
            } else {
                iy += sy;
                next_y += step_y;
                if (iy < 0 || iy >= static_cast<std::int64_t>(map.ny)) break;
            }
        }
    }
};

}  // namespace detail

inline StochasticResult trace_stochastic(const StochasticScene &scene,
                                         const StochasticSettings &settings) {
    if (scene.bvh == nullptr || scene.bands == 0 || scene.bands > kMaxBands ||
        (scene.surface_size != nullptr && scene.band_hz == nullptr)) {
        throw std::invalid_argument("trace_stochastic: bad scene");
    }
    if (settings.rays == 0 || settings.time_step <= 0.0 || settings.max_time <= 0.0 ||
        settings.speed_of_sound <= 0.0 ||
        settings.receiver_radius.size() != scene.receivers.size() ||
        std::any_of(settings.receiver_radius.begin(), settings.receiver_radius.end(),
                    [](double r) { return !(r > 0.0); }) ||
        settings.dir_time_step <= 0.0 || settings.window_time_step <= 0.0 ||
        !(settings.split_ratio > 1.0) || settings.max_split < 2 || settings.max_family < 1 ||
        (!settings.covered_time.empty() &&
         settings.covered_time.size() != scene.receivers.size())) {
        throw std::invalid_argument("trace_stochastic: bad settings");
    }
    const MapGrid &map = scene.map;
    const bool has_map = map.points() > 0;
    if (has_map && !map.valid()) throw std::invalid_argument("trace_stochastic: bad map");
    const Bvh &bvh = *scene.bvh;
    const std::size_t bands = scene.bands;
    const std::size_t n_rec = scene.receivers.size();
    const std::size_t n_dirs = scene.directions.size();
    const double c = settings.speed_of_sound;
    const double max_length = c * settings.max_time;
    std::vector<double> sphere_volume(n_rec);
    for (std::size_t k = 0; k < n_rec; ++k) {
        const double r = settings.receiver_radius[k];
        sphere_volume[k] = 4.0 / 3.0 * kPi * r * r * r;
    }
    const double offset = 1e-7 * bvh.scale();


    std::size_t threads = settings.threads;
    if (threads == 0) threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    threads = std::max<std::size_t>(1, std::min(threads, settings.rays));
    double map_bottom = 0.0, map_top = 0.0;
    if (has_map) {
        map_bottom = *std::min_element(map.low.begin(), map.low.end());
        map_top = *std::max_element(map.high.begin(), map.high.end());
        // Every thread sums its own copy of the map: keep them within kMapMemory.
        constexpr double kMapMemory = 256.0 * 1024.0 * 1024.0;
        const double copy = 8.0 * static_cast<double>(map.points() * bands) *
                            static_cast<double>(map.bins(settings.max_time));
        threads = std::min(threads, std::max<std::size_t>(
                                        1, static_cast<std::size_t>(kMapMemory / copy)));
    }

    auto make_result = [&] {
        StochasticResult res;
        res.bins = static_cast<std::size_t>(std::ceil(settings.max_time / settings.time_step));
        res.dir_bins =
            static_cast<std::size_t>(std::ceil(settings.max_time / settings.dir_time_step));
        res.energy.assign(n_rec * bands * res.bins, 0.0);
        if (n_dirs > 0) res.directional.assign(n_rec * n_dirs * bands * res.dir_bins, 0.0);
        res.windows =
            static_cast<std::size_t>(std::ceil(settings.max_time / settings.window_time_step));
        res.ray_energy.assign(res.windows, 0.0);
        res.reflections.assign(res.windows, 0.0);
        if (has_map) {
            res.map_bins = map.bins(settings.max_time);
            res.map_energy.assign(map.points() * bands * res.map_bins, 0.0);
        }
        return res;
    };
    std::vector<StochasticResult> partial(threads);
    for (auto &p : partial) p = make_result();

    parallel_for(settings.rays, threads, [&](std::size_t begin, std::size_t end,
                                             std::size_t worker) {
        StochasticResult &out = partial[worker];
        detail::Accumulator acc{scene, settings, out};
        acc.map_bottom = map_bottom;
        acc.map_top = map_top;
        std::array<double, kMaxBands> reflected{}, rain{}, scatter{}, counted{};

        std::vector<detail::RayState> stack;

        for (std::size_t ray = begin; ray < end; ++ray) {
            Rng first_rng(settings.seed, settings.first_ray + ray);
            const Vec3 first_dir = uniform_sphere(first_rng);
            std::array<double, kMaxBands> start{};
            scene.directivity.energy_weights(first_dir, bands, start.data());
            double loudest = 0.0;
            for (std::size_t b = 0; b < bands; ++b) {
                start[b] /= static_cast<double>(settings.rays);
                loudest = std::max(loudest, start[b]);
            }
            if (loudest <= 0.0) continue;  // the source sends nothing this way
            const double cutoff = loudest * 1e-14;
            stack.clear();
            std::array<double, kMaxBands> whole{};
            whole.fill(1.0);
            stack.push_back(
                detail::RayState{scene.source, first_dir, start, whole, whole, 0.0, 0, -1, first_rng});
            std::size_t family = 1;

            while (!stack.empty()) {
                detail::RayState ray_state = stack.back();
                stack.pop_back();
                ++out.traced_rays;
                auto &[pos, dir, energy, seen, alone, travelled, order, last, rng] = ray_state;

                while (travelled < max_length) {
                    const Hit hit = bvh.intersect(pos, dir, max_length - travelled, last);
                    const double segment = hit.found() ? hit.t : max_length - travelled;

                    if (!settings.specular_rain) {
                        for (std::size_t k = 0; k < n_rec; ++k) {
                            const Vec3 rel = scene.receivers[k] - pos;
                            const double s = dot(rel, dir);
                            const double d2 = dot(rel, rel) - s * s;
                            const double r = settings.receiver_radius[k];
                            if (d2 >= r * r) continue;
                            const double half = std::sqrt(r * r - d2);
                            const double t0 = std::max(s - half, 0.0);
                            const double t1 = std::min(s + half, segment);
                            if (t1 <= t0) continue;
                            const double distance = travelled + std::clamp(s, t0, t1);
                            const bool covered =
                                order <= settings.covered_order &&
                                (settings.covered_time.empty() ||
                                 distance / c <= settings.covered_time[k]);
                            // The specular-only share of a covered path is the early
                            // stage's.
                            double any = 0.0;
                            for (std::size_t b = 0; b < bands; ++b) {
                                counted[b] =
                                    energy[b] * std::max(seen[b] - (covered ? alone[b] : 0.0), 0.0);
                                any += counted[b];
                            }
                            if (any <= 0.0) continue;
                            acc.add(k, counted.data(), (t1 - t0) / sphere_volume[k], distance,
                                    -dir);
                        }
                    }
                    if (has_map) {
                        acc.add_map(pos, dir, segment, travelled, energy.data(), alone.data(),
                                    order);
                    }
                    if (!hit.found()) {
                        if (travelled + segment < max_length) ++out.escaped_rays;
                        break;
                    }

                    travelled += hit.t;
                    pos = pos + dir * hit.t;
                    const auto face = static_cast<std::size_t>(hit.face);
                    Vec3 n = bvh.normal(face);
                    if (dot(n, dir) > 0.0) n = -n;  // face the side the ray came from
                    ++order;
                    last = hit.face;

                    const double *alpha = scene.absorption + face * bands;
                    const double *surface = scene.scattering + face * bands;
                    if (scene.surface_size != nullptr && order >= settings.size_from_order) {
                        size_scattering(scene.surface_size[2 * face], scene.surface_size[2 * face + 1],
                                        hit.t, -dot(n, dir), c, bands, scene.band_hz, surface,
                                        scatter.data());
                    } else {
                        std::copy(surface, surface + bands, scatter.begin());
                    }
                    double peak = 0.0, mean = 0.0, p = 0.0;
                    for (std::size_t b = 0; b < bands; ++b) {
                        mean += energy[b];
                        reflected[b] = energy[b] * (1.0 - alpha[b]);
                        rain[b] = reflected[b] * scatter[b];
                        peak = std::max(peak, reflected[b]);
                        p += scatter[b];
                        alone[b] *= 1.0 - scatter[b];  // after this reflection
                    }
                    p /= static_cast<double>(bands);
                    const std::size_t window =
                        std::min(out.windows - 1,
                                 static_cast<std::size_t>(travelled / c / settings.window_time_step));
                    out.ray_energy[window] += mean / static_cast<double>(bands);
                    out.reflections[window] += 1.0;

                    if (settings.diffuse_rain || settings.specular_rain) {
                        const Vec3 origin = pos + n * offset;
                        for (std::size_t k = 0; k < n_rec; ++k) {
                            const Vec3 v = scene.receivers[k] - origin;
                            const double dist = std::max(norm(v), settings.receiver_radius[k]);
                            const double cos_theta = dot(n, v) / dist;
                            if (cos_theta <= 0.0) continue;
                            const Vec3 toward = v / norm(v);
                            if (bvh.occluded(origin, toward, norm(v), last)) continue;
                            const double factor = cos_theta / (kPi * dist * dist);
                            if (!settings.specular_rain) {
                                acc.add(k, rain.data(), factor, travelled + dist, -toward);
                                continue;
                            }
                            // All reflected energy radiates, but the specular-only share of
                            // a path the early stage computes exactly stays out.
                            const bool covered =
                                order <= settings.covered_order &&
                                (settings.covered_time.empty() ||
                                 (travelled + dist) / c <= settings.covered_time[k]);
                            for (std::size_t b = 0; b < bands; ++b) {
                                counted[b] = reflected[b] * (covered ? 1.0 - alone[b] : 1.0);
                            }
                            acc.add(k, counted.data(), factor, travelled + dist, -toward);
                        }
                    }
                    if (peak < cutoff) break;

                    if (settings.vector_scattering) {
                        // One ray for the mixture: all the reflected energy, in between the
                        // specular and a Lambert direction. The rain covers the scattered
                        // share of the next leg.
                        const Vec3 mixed = reflect(dir, n) * (1.0 - p) + cosine_hemisphere(n, rng) * p;
                        dir = mixed / norm(mixed);
                        for (std::size_t b = 0; b < bands; ++b) {
                            energy[b] = reflected[b];
                            seen[b] = settings.diffuse_rain ? 1.0 - scatter[b] : 1.0;
                        }
                    } else if (rng.uniform() < p) {
                        // Diffuse, re-weighted per band so each band's expected energy
                        // follows its own scattering coefficient.
                        for (std::size_t b = 0; b < bands; ++b) {
                            energy[b] = rain[b] / p;
                            seen[b] = settings.diffuse_rain ? 0.0 : 1.0;  // the rain covered it
                            alone[b] = 0.0;
                        }
                        dir = cosine_hemisphere(n, rng);
                    } else {
                        // Specular: the whole ray is still specular-only if it was before.
                        for (std::size_t b = 0; b < bands; ++b) {
                            energy[b] = (reflected[b] - rain[b]) / (1.0 - p);
                            seen[b] = 1.0;
                            alone[b] = alone[b] > 0.0 ? 1.0 : 0.0;
                        }
                        dir = reflect(dir, n);
                    }

                    if (settings.split_target.empty()) continue;
                    const double expected =
                        settings.split_target[std::min(window, settings.split_target.size() - 1)];
                    if (!(expected > 0.0)) continue;
                    double now = 0.0;
                    for (std::size_t b = 0; b < bands; ++b) now += energy[b];
                    now /= static_cast<double>(bands);
                    if (now > settings.split_ratio * expected && family < settings.max_family) {
                        const auto copies = static_cast<std::size_t>(std::min<double>(
                            {std::ceil(now / expected), static_cast<double>(settings.max_split),
                             static_cast<double>(settings.max_family - family + 1)}));
                        if (copies >= 2) {
                            for (std::size_t b = 0; b < bands; ++b) {
                                energy[b] /= static_cast<double>(copies);
                            }
                            for (std::size_t i = 1; i < copies; ++i) {
                                stack.push_back(ray_state);
                                stack.back().rng = rng.split();
                            }
                            family += copies - 1;
                        }
                    } else if (now * settings.split_ratio < expected) {
                        const double survive = now / expected;
                        if (rng.uniform() >= survive) break;
                        for (std::size_t b = 0; b < bands; ++b) energy[b] /= survive;
                    }
                }
            }
        }
    });

    StochasticResult total = make_result();
    for (const auto &p : partial) {
        for (std::size_t i = 0; i < total.energy.size(); ++i) total.energy[i] += p.energy[i];
        for (std::size_t i = 0; i < total.directional.size(); ++i) {
            total.directional[i] += p.directional[i];
        }
        for (std::size_t i = 0; i < total.windows; ++i) {
            total.ray_energy[i] += p.ray_energy[i];
            total.reflections[i] += p.reflections[i];
        }
        for (std::size_t i = 0; i < total.map_energy.size(); ++i) {
            total.map_energy[i] += p.map_energy[i];
        }
        total.escaped_rays += p.escaped_rays;
        total.traced_rays += p.traced_rays;
    }
    return total;
}

}  // namespace acoustic_spaces
