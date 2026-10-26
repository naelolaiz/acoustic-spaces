// SPDX-License-Identifier: GPL-3.0-or-later
//
// Geometric acoustics kernels on top of the BVH:
//
// - specular_sequences: which faces deterministic specular rays hit, in order. The
//   early-reflection finder turns these into image sources and checks them exactly.
// - trace_stochastic: energy-based stochastic ray tracing with diffuse rain, giving
//   energy per receiver, octave band and time slot (and, optionally, per arrival
//   direction) for the late part of the response, and optionally over a coverage map.
//   Outdoors its rays may bend in a refracting atmosphere (atmosphere.hpp), and plants
//   absorb and scatter them (vegetation.hpp).
//
// Units: a source of unit energy; results are time-integrated intensity, so the
// direct sound at distance r contributes 1 / (4 pi r^2).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "acoustic_spaces/atmosphere.hpp"
#include "acoustic_spaces/bvh.hpp"
#include "acoustic_spaces/vec3.hpp"
#include "acoustic_spaces/vegetation.hpp"

namespace acoustic_spaces {

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kMaxBands = 32;
// Most groups of faces whose reflections the tracer counts (StochasticScene::face_group).
constexpr std::size_t kMaxGroups = 16;

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

    // A generator for a new, independent stream (for a ray split off this one). The two
    // draws are sequenced: as arguments of one call, compilers took them in different
    // orders (GCC and MSVC as here, Clang the other way round).
    Rng split() {
        const std::uint64_t stream = next();
        const std::uint64_t seed = next();
        return Rng(seed, stream);
    }

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
// An exception that left a thread would end the process (std::terminate): the first a
// worker throws is kept and thrown here once every thread has finished, as is one from
// starting a thread, after the threads already started have finished.
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
    std::exception_ptr error;
    std::mutex error_mutex;
    const auto run = [&](std::size_t begin, std::size_t end, std::size_t w) {
        try {
            body(begin, end, w);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const std::size_t chunk = (n + threads - 1) / threads;
    try {
        for (std::size_t w = 0; w < threads; ++w) {
            const std::size_t begin = w * chunk;
            const std::size_t end = std::min(n, begin + chunk);
            if (begin >= end) break;
            pool.emplace_back(run, begin, end, w);
        }
    } catch (...) {
        for (auto &t : pool) t.join();
        throw;
    }
    for (auto &t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

// How a face treats a specular ray in specular_sequences: it reflects it, it reflects
// it and lets it through (a partition), or it only lets it through (an opening).
enum class Passing : std::uint8_t { kReflects = 0, kBoth = 1, kThrough = 2 };

// Faces that let sound through, crossed on one path of specular_sequences: only that
// many of those that also reflect (each one forks the path), and only that many in all.
constexpr std::size_t kMaxForks = 2;
constexpr std::size_t kMaxCrossings = 64;
// Rows one ray of specular_sequences may give, its forks included.
constexpr std::size_t kMaxRows = 64;

// Face indices hit by `rays` specular rays from `source` (Fibonacci directions), up to
// `max_order` reflections each; one row of max_order per ray, -1 once a ray leaves the
// model. With `passing` (one Passing per face), a ray passes straight through faces
// that let sound through: past an opening, and both through and off a partition, each
// branch giving its own row (the reflections only, not the crossings).
inline std::vector<std::int64_t> specular_sequences(const Bvh &bvh, const Vec3 &source,
                                                    std::size_t rays, std::size_t max_order,
                                                    std::size_t threads,
                                                    const std::vector<std::uint8_t> &passing = {}) {
    std::vector<std::vector<std::int64_t>> rows(rays);
    const double far = 1e6 * bvh.scale();
    struct Branch {
        Vec3 pos, dir;
        std::int64_t last;
        std::size_t forks, crossings;
        std::vector<std::int64_t> faces;
    };
    parallel_for(rays, threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        std::vector<Branch> stack;
        for (std::size_t i = begin; i < end; ++i) {
            auto &out = rows[i];
            stack.clear();
            stack.push_back(Branch{source, fibonacci_sphere(i, rays), -1, 0, 0, {}});
            while (!stack.empty() && out.size() < kMaxRows * max_order) {
                Branch branch = std::move(stack.back());
                stack.pop_back();
                while (branch.faces.size() < max_order) {
                    const Hit hit = bvh.intersect(branch.pos, branch.dir, far, branch.last);
                    if (!hit.found()) break;
                    const auto face = static_cast<std::size_t>(hit.face);
                    const auto kind = passing.empty() ? Passing::kReflects
                                                      : static_cast<Passing>(passing[face]);
                    branch.pos = branch.pos + branch.dir * hit.t;
                    branch.last = hit.face;
                    if (kind != Passing::kReflects) {
                        if (++branch.crossings > kMaxCrossings) break;
                        if (kind == Passing::kThrough) continue;
                        if (branch.forks < kMaxForks) {
                            stack.push_back(branch);
                            ++stack.back().forks;
                        }
                    }
                    branch.faces.push_back(hit.face);
                    branch.dir = reflect(branch.dir, bvh.normal(face));
                }
                branch.faces.resize(max_order, -1);
                out.insert(out.end(), branch.faces.begin(), branch.faces.end());
            }
        }
    });
    std::vector<std::int64_t> out;
    for (const auto &r : rows) out.insert(out.end(), r.begin(), r.end());
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
    // With StochasticScene::face_room: the weight window's expected energy in each room
    // relative to the source's room (split_target is the source room's). A ray that
    // meets a surface letting sound through into a room with a level then always goes
    // on both ways: a copy carries the share that passes through, the ray the share
    // that reflects; the window keeps the copies in each room near that room's level.
    // Without one, or in the first batch (no split_target), the ray goes one way or
    // the other, at random in proportion to the two shares.
    std::vector<double> room_level;
    // With StochasticScene::face_group: resolution of StochasticResult::moments (s).
    double moment_time_step = 1e-2;
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
    // Optional: the share of sound each face lets through (faces x bands), and the room
    // on the front (the side its normal points to) and on the back of each face (faces
    // x 2, -1 for none). A ray passes into the room on the far side; with no room
    // there (the outside), what passes counts as absorbed. Without face_room, every
    // face lets its share through. source_room is the room the source is in.
    const double *transmission = nullptr;
    const std::int64_t *face_room = nullptr;
    std::int64_t source_room = -1;
    // Optional: the group of each face (faces, -1 for none) out of `groups` (at most
    // kMaxGroups). The tracer counts the reflections each ray has had off each group and
    // sums, per receiver, the energy times those counts and their products in
    // StochasticResult::moments: enough to re-weight the response for new absorption.
    const std::int64_t *face_group = nullptr;
    std::size_t groups = 0;
    std::vector<Vec3> receivers;
    std::vector<Vec3> directions;  // unit vectors for directional bins; may be empty
    MapGrid map;                   // no points: no map
    // Optional, outdoors: the atmosphere bends the rays, followed in chords (see
    // atmosphere.hpp; prepared). Times then follow the speed along each ray and the air
    // absorbs over its length; the rain from a reflection goes straight, in the time
    // the line takes, and none reaches a receiver in the shadow of upward refraction.
    const Atmosphere *atmosphere = nullptr;
    // Optional, outdoors: plants (prepared, with these bands) that absorb along the rays
    // and scatter them. The rain on the way to a receiver loses what the plants take.
    const Vegetation *vegetation = nullptr;
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
    // With face groups: per receiver, the energy arriving in each moment bin times 1,
    // each group's reflection count n_i, and each product n_i n_j (i <= j), in that
    // order (moment_terms of them): receivers x moment_terms x bands x moment_bins.
    std::size_t moment_bins = 0, moment_terms = 0;
    std::vector<double> moments;
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
    int order;           // reflections so far (passing through a face is none)
    std::int64_t last;   // face of the last reflection, ignored by the next intersection
    Rng rng;
    std::int64_t room;   // the room the ray is in (StochasticScene::face_room), or -1
    // Reflections so far off each face group (StochasticScene::face_group).
    std::array<std::uint16_t, kMaxGroups> hits{};
    // In an atmosphere: metres along the ray so far (travelled is then the speed of
    // sound times the time), and, while `leaving`, the normal of the face the ray left
    // (pointing the way it went), which it cannot meet again until it turns back.
    double path = 0.0;
    Vec3 away{};
    bool leaving = false;
};

// The share of sound per band that travels from `origin` along `dir` (unit) for
// `length` metres: 1 in the open, the product of the transmission of the faces it
// crosses; false (and no share) where a face lets nothing through.
inline bool passing_share(const StochasticScene &scene, const Vec3 &origin, const Vec3 &dir,
                          double length, std::int64_t ignore, double *share) {
    const Bvh &bvh = *scene.bvh;
    const std::size_t bands = scene.bands;
    std::fill(share, share + bands, 1.0);
    if (scene.transmission == nullptr) return !bvh.occluded(origin, dir, length, ignore);
    Vec3 pos = origin;
    for (std::size_t k = 0; k <= kMaxCrossings; ++k) {
        const Hit hit = bvh.intersect(pos, dir, length, ignore);
        if (!hit.found()) return true;
        const double *tau = scene.transmission + static_cast<std::size_t>(hit.face) * bands;
        bool any = false;
        for (std::size_t b = 0; b < bands; ++b) {
            share[b] *= tau[b];
            any = any || share[b] > 0.0;
        }
        if (!any) return false;
        pos = pos + dir * hit.t;
        length -= hit.t;
        ignore = hit.face;
    }
    return false;
}

// A straight piece of a ray through plants: where along it the ray scatters, and how
// much of each band reaches any point of it. Scatterings are drawn at one rate for all
// bands, each medium's scattering averaged over the bands weighted by the ray's energy;
// the bands are then weighted by what they meet against that rate (w_b(s) = exp(S(s) -
// tau_b(s)), S the drawn rate and tau_b the band's absorption and scattering, both
// integrated to s), so each keeps its expected energy.
struct Track {
    std::vector<Vegetation::Span> spans;
    std::vector<double> cuts;
    std::vector<double> edges;    // where the rates change, from 0 to the piece's end
    std::vector<double> drawn;    // at each edge: the drawn rate integrated so far
    std::vector<double> met;      // edges x bands: tau_b at each edge
    std::vector<double> rate;     // per medium: the drawn rate per metre of it
    std::vector<double> chance;   // per span: its medium's drawn rate times its share
    std::size_t bands = 0;
    std::size_t scattered = 0;    // the piece between edges collide last scattered in

    // The piece from p along unit d for `length` metres, for a ray with `energy` per
    // band; false where it meets no plants.
    bool build(const Vegetation &plants, const Vec3 &p, const Vec3 &d, double length,
               const double *energy) {
        plants.spans(p, d, length, spans, cuts);
        if (spans.empty()) return false;
        bands = plants.bands;
        const std::size_t media = plants.media();
        double total = 0.0;
        for (std::size_t b = 0; b < bands; ++b) total += energy[b];
        rate.assign(media, 0.0);
        for (std::size_t m = 0; m < media && total > 0.0; ++m) {
            const double *s = plants.scattering.data() + m * bands;
            for (std::size_t b = 0; b < bands; ++b) rate[m] += energy[b] * s[b];
            rate[m] /= total;
        }
        chance.resize(spans.size());
        edges.assign({0.0, length});
        for (std::size_t i = 0; i < spans.size(); ++i) {
            chance[i] = rate[spans[i].medium] * plants.share(spans[i].medium, d);
            edges.push_back(spans[i].t0);
            edges.push_back(spans[i].t1);
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        drawn.assign(edges.size(), 0.0);
        met.assign(edges.size() * bands, 0.0);
        for (std::size_t e = 0; e + 1 < edges.size(); ++e) {
            const double t0 = edges[e], t1 = edges[e + 1], middle = 0.5 * (t0 + t1);
            double here = 0.0;
            double *next = met.data() + (e + 1) * bands;
            std::copy(met.data() + e * bands, next, next);
            for (std::size_t i = 0; i < spans.size(); ++i) {
                const Vegetation::Span &s = spans[i];
                if (!(s.t0 <= middle && middle <= s.t1)) continue;
                const double metres = (t1 - t0) * plants.share(s.medium, d);
                here += chance[i] * (t1 - t0);
                const double *a = plants.absorption.data() + s.medium * bands;
                const double *c = plants.scattering.data() + s.medium * bands;
                for (std::size_t b = 0; b < bands; ++b) next[b] += (a[b] + c[b]) * metres;
            }
            drawn[e + 1] = drawn[e] + here;
        }
        return true;
    }

    // Where along the piece the ray scatters, or infinity where it goes through. Draws
    // from rng only where it may scatter.
    double collide(Rng &rng) {
        if (!(drawn.back() > 0.0)) return std::numeric_limits<double>::infinity();
        const double target = -std::log(1.0 - rng.uniform());
        if (target >= drawn.back()) return std::numeric_limits<double>::infinity();
        const auto e = static_cast<std::size_t>(
            std::upper_bound(drawn.begin(), drawn.end(), target) - drawn.begin() - 1);
        scattered = e;
        const double f = (target - drawn[e]) / (drawn[e + 1] - drawn[e]);
        return std::min(edges[e] + f * (edges[e + 1] - edges[e]), edges[e + 1]);
    }

    // The weight w_b of each band at s metres along the piece.
    void weight(double s, double *w) const {
        const auto found = std::upper_bound(edges.begin(), edges.end(), s);
        const auto after = static_cast<std::size_t>(found - edges.begin());
        const std::size_t e = std::min(after == 0 ? 0 : after - 1, edges.size() - 2);
        const double f = std::clamp((s - edges[e]) / (edges[e + 1] - edges[e]), 0.0, 1.0);
        const double sampled = drawn[e] + f * (drawn[e + 1] - drawn[e]);
        const double *a = met.data() + e * bands, *b = a + bands;
        for (std::size_t k = 0; k < bands; ++k) w[k] = std::exp(sampled - a[k] - f * (b[k] - a[k]));
    }

    // The medium the ray scatters in where collide had it scatter: one of those there,
    // at random in proportion to their drawn rates.
    std::size_t medium(Rng &rng) const {
        const double middle = 0.5 * (edges[scattered] + edges[scattered + 1]);
        const auto there = [&](std::size_t i) {
            return spans[i].t0 <= middle && middle <= spans[i].t1 && chance[i] > 0.0;
        };
        double total = 0.0;
        std::size_t only = 0, count = 0;
        for (std::size_t i = 0; i < spans.size(); ++i) {
            if (!there(i)) continue;
            total += chance[i];
            only = i;
            ++count;
        }
        if (count <= 1) return spans[only].medium;
        double pick = rng.uniform() * total;
        for (std::size_t i = 0; i < spans.size(); ++i) {
            if (!there(i)) continue;
            only = i;
            pick -= chance[i];
            if (pick < 0.0) break;
        }
        return spans[only].medium;
    }
};

// Turns over which the stems' rain averages the leg they send through a receiver's
// sphere.
constexpr std::size_t kStemTurns = 12;

// Henyey and Greenstein's phase function: the share per steradian of what foliage
// scatters that goes off at cos(angle) mu to the way the sound went, mean cosine g.
inline double henyey_greenstein(double mu, double g) {
    const double q = 1.0 + g * g - 2.0 * g * mu;
    return (1.0 - g * g) / (4.0 * kPi * q * std::sqrt(q));
}

// The cosine of an angle drawn from henyey_greenstein with mean cosine g.
inline double henyey_greenstein_cosine(double g, Rng &rng) {
    const double u = rng.uniform();
    if (std::abs(g) < 1e-6) return 1.0 - 2.0 * u;
    const double s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
    return std::clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
}

// The wrapped Cauchy distribution's density at angle phi, mean cosine rho: how stems
// share out the angles through which they turn sound sideways.
inline double wrapped_cauchy(double phi, double rho) {
    return (1.0 - rho * rho) / (2.0 * kPi * (1.0 + rho * rho - 2.0 * rho * std::cos(phi)));
}

// An angle drawn from wrapped_cauchy with mean cosine rho.
inline double wrapped_cauchy_angle(double rho, Rng &rng) {
    const double u = rng.uniform();
    return 2.0 * std::atan((1.0 - rho) / (1.0 + rho) * std::tan(kPi * (u - 0.5)));
}

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

    // Adds energy[b] * factor * air(distance) arriving from `arrival` at `distance`, from
    // a ray with `hits` reflections off each face group; the air absorbs over
    // `air_distance` instead where that is not negative.
    void add(std::size_t receiver, const double *energy, double factor, double distance,
             const Vec3 &arrival, const std::uint16_t *hits, double air_distance = -1.0) {
        const double time = distance / settings.speed_of_sound;
        const auto bin = static_cast<std::size_t>(time / settings.time_step);
        if (bin >= out.bins) return;
        const std::size_t bands = scene.bands;
        double *moments = nullptr;
        std::array<double, 1 + kMaxGroups + kMaxGroups * (kMaxGroups + 1) / 2> terms{};
        if (out.moment_terms > 0) {
            const std::size_t mbin = std::min(
                out.moment_bins - 1, static_cast<std::size_t>(time / settings.moment_time_step));
            moments = out.moments.data() +
                      receiver * out.moment_terms * bands * out.moment_bins + mbin;
            const std::size_t groups = scene.groups;
            terms[0] = 1.0;
            std::size_t t = 1 + groups;
            for (std::size_t i = 0; i < groups; ++i) {
                terms[1 + i] = static_cast<double>(hits[i]);
                for (std::size_t j = i; j < groups; ++j) {
                    terms[t++] = static_cast<double>(hits[i]) * static_cast<double>(hits[j]);
                }
            }
        }
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
        const double absorbing = air_distance < 0.0 ? distance : air_distance;
        for (std::size_t b = 0; b < bands; ++b) {
            const double e = energy[b] * factor * std::exp(-scene.air[b] * absorbing);
            row[b * out.bins + bin] += e;
            if (drow) drow[b * out.dir_bins + dbin] += e;
            if (moments != nullptr && e != 0.0) {
                for (std::size_t t = 0; t < out.moment_terms; ++t) {
                    moments[(t * bands + b) * out.moment_bins] += e * terms[t];
                }
            }
        }
    }

    // Lowest slab bottom and highest slab top of the map.
    double map_bottom = 0.0, map_top = 0.0;

    // Adds a leg that leaves `pos` along `dir` (unit) for `length` metres, `travelled`
    // metres from the source at `pos`, to every map slab it passes through. A chord of a
    // bent ray adds `rate` to travelled per metre, and the air absorbs from `path` on,
    // `stretch` metres per metre of chord (path negative: as travelled).
    // Through plants, `track` (from pos along dir) weights the bands at each point.
    void add_map(const Vec3 &pos, const Vec3 &dir, double length, double travelled,
                 const double *energy, const double *alone, int order, double rate = 1.0,
                 double path = -1.0, double stretch = 1.0, const Track *track = nullptr) {
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
                const double middle = 0.5 * (a + b);
                const double distance = travelled + rate * middle;
                const double absorbing = path < 0.0 ? distance : path + stretch * middle;
                const bool covered =
                    order <= settings.covered_order && distance / c <= map.covered_time[k];
                const std::size_t bin = map.bin(distance / c - map.onset[k]);
                if (bin >= out.map_bins) continue;
                const double factor = (b - a) / map.volume[k];
                double *row = out.map_energy.data() + k * bands * out.map_bins;
                std::array<double, kMaxBands> weights{};
                if (track != nullptr) track->weight(middle, weights.data());
                for (std::size_t band = 0; band < bands; ++band) {
                    // The specular-only share of a covered path is the early stage's.
                    const double share = covered ? 1.0 - alone[band] : 1.0;
                    if (track == nullptr) {
                        row[band * out.map_bins + bin] +=
                            energy[band] * share * factor * std::exp(-scene.air[band] * absorbing);
                    } else {
                        row[band * out.map_bins + bin] += energy[band] * weights[band] * share *
                                                          factor *
                                                          std::exp(-scene.air[band] * absorbing);
                    }
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
        std::any_of(settings.room_level.begin(), settings.room_level.end(),
                    [](double v) { return !(v >= 0.0); }) ||
        (!settings.covered_time.empty() &&
         settings.covered_time.size() != scene.receivers.size()) ||
        (scene.groups > 0 && !(settings.moment_time_step > 0.0))) {
        throw std::invalid_argument("trace_stochastic: bad settings");
    }
    if (scene.groups > kMaxGroups || (scene.groups > 0 && scene.face_group == nullptr)) {
        throw std::invalid_argument("trace_stochastic: bad face groups");
    }
    const MapGrid &map = scene.map;
    const bool has_map = map.points() > 0;
    if (has_map && !map.valid()) throw std::invalid_argument("trace_stochastic: bad map");
    const Atmosphere *atmosphere = scene.atmosphere;
    if (atmosphere != nullptr && (!atmosphere->valid() || atmosphere->buckets == 0)) {
        throw std::invalid_argument("trace_stochastic: bad atmosphere");
    }
    const Vegetation *plants = scene.vegetation;
    if (plants != nullptr && plants->bands != scene.bands) {
        throw std::invalid_argument("trace_stochastic: bad vegetation");
    }
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
    const std::size_t moment_terms =
        scene.groups > 0 ? 1 + scene.groups + scene.groups * (scene.groups + 1) / 2 : 0;
    const std::size_t moment_bins =
        moment_terms > 0
            ? static_cast<std::size_t>(std::ceil(settings.max_time / settings.moment_time_step))
            : 0;
    if (moment_terms > 0) {
        // Every thread sums its own copy of the moments, too.
        constexpr double kMomentMemory = 256.0 * 1024.0 * 1024.0;
        const double copy = 8.0 * static_cast<double>(n_rec * moment_terms * bands) *
                            static_cast<double>(moment_bins);
        threads = std::min(threads, std::max<std::size_t>(
                                        1, static_cast<std::size_t>(kMomentMemory / copy)));
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
        res.moment_terms = moment_terms;
        res.moment_bins = moment_bins;
        res.moments.assign(n_rec * moment_terms * bands * moment_bins, 0.0);
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
        std::array<double, kMaxBands> reflected{}, rain{}, scatter{}, counted{}, through{},
            share{}, seen_before{}, alone_before{}, weights{}, depth{};
        // A piece of a ray through plants, and room to find what a line meets in them.
        detail::Track track;
        std::vector<Vegetation::Span> line_spans;
        std::vector<double> line_cuts;
        const double inf = std::numeric_limits<double>::infinity();
        // The weight window's level of a room relative to the source's.
        const auto level = [&](std::int64_t room) {
            if (room < 0 || static_cast<std::size_t>(room) >= settings.room_level.size()) {
                return settings.room_level.empty() ? 1.0 : 0.0;
            }
            return settings.room_level[static_cast<std::size_t>(room)];
        };
        // The receivers' detector spheres along a straight piece of a ray: `segment`
        // metres from `from` along `along` (unit). On a chord of a bent ray, travelled
        // grows by `rate` per metre and the air absorbs from `path` on, `stretch` metres
        // per metre (path negative: as travelled). Through plants, `plant_track` (the
        // piece's) weights the bands.
        const auto spheres = [&](const Vec3 &from, const Vec3 &along, double segment,
                                 const detail::RayState &state, double rate, double path,
                                 double stretch, const detail::Track *plant_track) {
            for (std::size_t k = 0; k < n_rec; ++k) {
                const Vec3 rel = scene.receivers[k] - from;
                const double s = dot(rel, along);
                const double d2 = dot(rel, rel) - s * s;
                const double r = settings.receiver_radius[k];
                if (d2 >= r * r) continue;
                const double half = std::sqrt(r * r - d2);
                const double t0 = std::max(s - half, 0.0);
                const double t1 = std::min(s + half, segment);
                if (t1 <= t0) continue;
                const double nearest = std::clamp(s, t0, t1);
                const double distance = state.travelled + rate * nearest;
                const bool covered = state.order <= settings.covered_order &&
                                     (settings.covered_time.empty() ||
                                      distance / c <= settings.covered_time[k]);
                // The specular-only share of a covered path is the early stage's.
                std::array<double, kMaxBands> passed;
                if (plant_track != nullptr) plant_track->weight(nearest, passed.data());
                double any = 0.0;
                for (std::size_t b = 0; b < bands; ++b) {
                    counted[b] = state.energy[b] *
                                 std::max(state.seen[b] - (covered ? state.alone[b] : 0.0), 0.0);
                    if (plant_track != nullptr) counted[b] *= passed[b];
                    any += counted[b];
                }
                if (any <= 0.0) continue;
                acc.add(k, counted.data(), (t1 - t0) / sphere_volume[k], distance, -along,
                        state.hits.data(), path < 0.0 ? -1.0 : path + stretch * nearest);
            }
        };

        // What stems at state.pos send each receiver as rain: the detector sphere's count
        // of the leg the ray goes on along after they turn it, on average over the turns
        // (mean cosine g[b] per band) instead of for the one turn drawn. The leg keeps the
        // slope of state.dir, so it crosses a sphere for the turns in one range, by
        // quadrature over the sine of the angle from its middle (the chord grows from
        // nothing at the ends). Each turn's leg ends at the first face in the sphere, and
        // the plants and faces that let sound through on the way take their share.
        const auto stem_rain = [&](const detail::RayState &state, const double *g) {
            const Vec3 &p = state.pos, &d = state.dir;
            const double flat = std::hypot(d.x, d.y);
            if (!(flat > 0.0)) return;
            const double heading = std::atan2(d.y, d.x);
            for (std::size_t k = 0; k < n_rec; ++k) {
                const double r = settings.receiver_radius[k];
                const Vec3 v = scene.receivers[k] - p;
                const double vv = dot(v, v), across = std::hypot(v.x, v.y);
                // A leg heading a meets the sphere ahead where v . leg > sqrt(|v|^2 - r^2),
                // v . leg = flat * across * cos(a - bearing) + d.z * v.z: every heading
                // from inside, none or a range round the bearing from outside.
                double middle = 0.0, half = kPi;
                if (vv > r * r) {
                    const double need = std::sqrt(vv - r * r) - d.z * v.z;
                    const double most = flat * across;
                    if (need >= most) continue;
                    if (need > -most) {
                        middle = std::atan2(v.y, v.x);
                        half = std::acos(need / most);
                    }
                }
                const bool round = half >= kPi;
                for (std::size_t j = 0; j < detail::kStemTurns; ++j) {
                    const double x = (static_cast<double>(j) + 0.5) / detail::kStemTurns;
                    double heading_j, width;  // the leg's heading and its share of the range
                    if (round) {
                        heading_j = middle + kPi * (2.0 * x - 1.0);
                        width = 2.0 * kPi / detail::kStemTurns;
                    } else {
                        const double psi = kPi * (x - 0.5);
                        heading_j = middle + half * std::sin(psi);
                        width = half * std::cos(psi) * kPi / detail::kStemTurns;
                    }
                    const Vec3 leg{flat * std::cos(heading_j), flat * std::sin(heading_j), d.z};
                    const double along = dot(v, leg);
                    const double d2 = vv - along * along;
                    if (d2 >= r * r) continue;
                    const double chord = std::sqrt(r * r - d2);
                    const double t0 = std::max(along - chord, 0.0);
                    double t1 = along + chord;
                    if (t1 <= t0) continue;
                    if (!detail::passing_share(scene, p, leg, t0, -1, share.data())) continue;
                    const Hit wall = bvh.intersect(p + leg * t0, leg, t1 - t0, -1);
                    if (wall.found()) t1 = t0 + wall.t;
                    const double nearest = std::clamp(along, t0, t1);
                    const Vec3 there = p + leg * nearest;
                    if (atmosphere != nullptr && atmosphere->shadowed(p, there)) continue;
                    plants->depth(p, leg, nearest, depth.data(), line_spans, line_cuts);
                    const double turn = heading_j - heading;
                    double any = 0.0;
                    for (std::size_t b = 0; b < bands; ++b) {
                        counted[b] = state.energy[b] * detail::wrapped_cauchy(turn, g[b]) * width *
                                     share[b] * std::exp(-depth[b]);
                        any += counted[b];
                    }
                    if (!(any > 0.0)) continue;
                    const double reach =
                        atmosphere == nullptr ? nearest : c * atmosphere->straight_time(p, there);
                    acc.add(k, counted.data(), (t1 - t0) / sphere_volume[k],
                            state.travelled + reach, -leg, state.hits.data(),
                            atmosphere == nullptr ? -1.0 : state.path + nearest);
                }
            }
        };

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
            stack.push_back(detail::RayState{scene.source, first_dir, start, whole, whole, 0.0, 0,
                                             -1, first_rng, scene.source_room, {}, 0.0, {},
                                             false});
            std::size_t family = 1;
            // The weight window after a reflection or a scattering in window `window`:
            // splits the ray (its copies go on the stack) or plays roulette with it;
            // false when it dies.
            const auto weigh = [&](detail::RayState &state, std::size_t window) {
                if (settings.split_target.empty()) return true;
                const double expected =
                    settings.split_target[std::min(window, settings.split_target.size() - 1)] *
                    level(state.room);
                if (!(expected > 0.0)) return true;
                double now = 0.0;
                for (std::size_t b = 0; b < bands; ++b) now += state.energy[b];
                now /= static_cast<double>(bands);
                if (now > settings.split_ratio * expected && family < settings.max_family) {
                    const auto copies = static_cast<std::size_t>(std::min<double>(
                        {std::ceil(now / expected), static_cast<double>(settings.max_split),
                         static_cast<double>(settings.max_family - family + 1)}));
                    if (copies >= 2) {
                        for (std::size_t b = 0; b < bands; ++b) {
                            state.energy[b] /= static_cast<double>(copies);
                        }
                        for (std::size_t i = 1; i < copies; ++i) {
                            stack.push_back(state);
                            stack.back().rng = state.rng.split();
                        }
                        family += copies - 1;
                    }
                } else if (now * settings.split_ratio < expected) {
                    const double survive = now / expected;
                    if (state.rng.uniform() >= survive) return false;
                    for (std::size_t b = 0; b < bands; ++b) state.energy[b] /= survive;
                }
                return true;
            };

            while (!stack.empty()) {
                detail::RayState ray_state = stack.back();
                stack.pop_back();
                ++out.traced_rays;
                auto &[pos, dir, energy, seen, alone, travelled, order, last, rng, room, hits,
                       path, away, leaving] = ray_state;

                while (travelled < max_length) {
                    Hit hit;
                    double leg = 0.0;  // metres the sound went to the face it meets
                    // Where on its piece (track) the ray scatters in plants, if it does.
                    double scatter_at = inf;
                    if (atmosphere == nullptr) {
                        hit = bvh.intersect(pos, dir, max_length - travelled, last);
                        double segment = hit.found() ? hit.t : max_length - travelled;
                        const detail::Track *plant_track = nullptr;
                        if (plants != nullptr &&
                            track.build(*plants, pos, dir, segment, energy.data())) {
                            plant_track = &track;
                            scatter_at = track.collide(rng);
                            if (hit.found() && scatter_at > hit.t - offset) scatter_at = inf;
                            segment = std::min(segment, scatter_at);
                        }
                        if (!settings.specular_rain) {
                            spheres(pos, dir, segment, ray_state, 1.0, -1.0, 1.0, plant_track);
                        }
                        if (has_map) {
                            acc.add_map(pos, dir, segment, travelled, energy.data(), alone.data(),
                                        order, 1.0, -1.0, 1.0, plant_track);
                        }
                        if (scatter_at < inf) {
                            travelled += segment;
                            pos = pos + dir * segment;
                        } else if (!hit.found()) {
                            // Out of time, or out of the model: then no face lies anywhere
                            // ahead. (The search above only looks as far as the ray can go.)
                            if (!bvh.occluded(pos, dir, std::numeric_limits<double>::infinity(),
                                              last)) {
                                ++out.escaped_rays;
                            }
                            break;
                        } else {
                            travelled += hit.t;
                            pos = pos + dir * hit.t;
                            leg = hit.t;
                            if (plant_track != nullptr) {
                                track.weight(hit.t, weights.data());
                                for (std::size_t b = 0; b < bands; ++b) energy[b] *= weights[b];
                            }
                        }
                    } else {
                        // Chord by chord along the bent ray, to the first face it meets.
                        BentRay bent(*atmosphere, pos, dir);
                        while (travelled < max_length) {
                            const BentRay before = bent;
                            const double budget =
                                1.05 * (max_length - travelled) / c * bent.speed() + offset;
                            const BentRay::Chord chord = bent.chord(budget, leaving ? &away : nullptr);
                            const Vec3 span = chord.end - chord.start;
                            const double length = norm(span);
                            if (length > 0.0) {
                                const Vec3 along = span / length;
                                const Hit met = bvh.intersect(chord.start, along, length, last);
                                double segment = met.found() ? met.t : length;
                                const double rate = c * chord.time / length;
                                const double stretch = chord.arc / length;
                                const detail::Track *plant_track = nullptr;
                                if (plants != nullptr &&
                                    track.build(*plants, chord.start, along, segment, energy.data())) {
                                    plant_track = &track;
                                    scatter_at = track.collide(rng);
                                    if (met.found() && scatter_at > met.t - offset) scatter_at = inf;
                                    segment = std::min(segment, scatter_at);
                                }
                                if (!settings.specular_rain) {
                                    spheres(chord.start, along, segment, ray_state, rate, path, stretch,
                                            plant_track);
                                }
                                if (has_map) {
                                    acc.add_map(chord.start, along, segment, travelled, energy.data(),
                                                alone.data(), order, rate, path, stretch, plant_track);
                                }
                                if (scatter_at < inf) {
                                    // It scatters in the plants, going the ray's way there.
                                    travelled += rate * segment;
                                    path += stretch * segment;
                                    pos = chord.start + along * segment;
                                    BentRay there = before;
                                    there.chord(chord.arc * segment / length, leaving ? &away : nullptr);
                                    dir = there.direction();
                                    break;
                                }
                                if (plant_track != nullptr) {
                                    track.weight(segment, weights.data());
                                    for (std::size_t b = 0; b < bands; ++b) energy[b] *= weights[b];
                                }
                                if (met.found()) {
                                    travelled += rate * met.t;
                                    path += stretch * met.t;
                                    leg += stretch * met.t;
                                    pos = chord.start + along * met.t;
                                    // The ray's own direction where it meets the face.
                                    BentRay there = before;
                                    there.chord(chord.arc * met.t / length, leaving ? &away : nullptr);
                                    dir = there.direction();
                                    hit = met;
                                    break;
                                }
                            }
                            travelled += c * chord.time;
                            path += chord.arc;
                            leg += chord.arc;
                            pos = chord.end;
                            dir = bent.direction();
                            // Out of the model: it left the scene, as through the open sky.
                            if (!bvh.encloses(pos, offset)) break;
                            if (chord.turned) {
                                leaving = false;  // back towards the face it left
                                last = -1;
                            } else if (!(length > 0.0)) {
                                break;
                            }
                        }
                        if (scatter_at == inf && !hit.found()) {
                            if (!bvh.occluded(pos, dir, std::numeric_limits<double>::infinity(),
                                              last)) {
                                ++out.escaped_rays;
                            }
                            break;
                        }
                    }

                    if (scatter_at < inf) {
                        // The ray scatters in the plants at pos: in medium m, its bands
                        // weighted by what they met on the way and by m's scattering
                        // against the rate the scattering was drawn at.
                        const std::size_t m = track.medium(rng);
                        track.weight(scatter_at, weights.data());
                        const double *mu = plants->scattering.data() + m * bands;
                        const double *g = plants->asymmetry.data() + m * bands;
                        double mean = 0.0, kept = 0.0, mean_g = 0.0;
                        for (std::size_t b = 0; b < bands; ++b) {
                            energy[b] *= weights[b];
                            mean += energy[b];
                            energy[b] *= mu[b] / track.rate[m];
                            kept += energy[b];
                            mean_g += energy[b] * g[b];
                            alone[b] = 0.0;
                        }
                        const std::size_t window =
                            std::min(out.windows - 1,
                                     static_cast<std::size_t>(travelled / c / settings.window_time_step));
                        if (room == scene.source_room) {
                            out.ray_energy[window] += mean / static_cast<double>(bands);
                            out.reflections[window] += 1.0;
                        }
                        if (!(kept > 0.0)) break;
                        mean_g /= kept;
                        last = -1;
                        leaving = false;
                        const bool rains = settings.diffuse_rain || settings.specular_rain;
                        if (plants->upright[m]) {
                            // Stems turn it sideways, keeping its slope, and send each
                            // receiver what its next leg carries through the receiver's
                            // sphere, on average over the turns.
                            if (rains) stem_rain(ray_state, g);
                            const double turn = detail::wrapped_cauchy_angle(mean_g, rng);
                            const double cos_turn = std::cos(turn), sin_turn = std::sin(turn);
                            dir = normalized(Vec3{dir.x * cos_turn - dir.y * sin_turn,
                                                  dir.x * sin_turn + dir.y * cos_turn, dir.z});
                            const double drawn = detail::wrapped_cauchy(turn, mean_g);
                            for (std::size_t b = 0; b < bands; ++b) {
                                energy[b] *= detail::wrapped_cauchy(turn, g[b]) / drawn;
                                seen[b] = rains ? 0.0 : 1.0;  // the rain covered it
                            }
                        } else {
                            // Foliage sends its share straight to each receiver as the phase
                            // function has it, and the ray goes on in a direction drawn
                            // from it.
                            for (std::size_t k = 0; rains && k < n_rec; ++k) {
                                const Vec3 v = scene.receivers[k] - pos;
                                const double gap = norm(v);
                                if (!(gap > 0.0)) continue;
                                if (atmosphere != nullptr &&
                                    atmosphere->shadowed(pos, scene.receivers[k])) {
                                    continue;
                                }
                                const Vec3 toward = v / gap;
                                if (!detail::passing_share(scene, pos, toward, gap, -1,
                                                           share.data())) {
                                    continue;
                                }
                                plants->depth(pos, toward, gap, depth.data(), line_spans, line_cuts);
                                const double cosine = dot(dir, toward);
                                for (std::size_t b = 0; b < bands; ++b) {
                                    counted[b] = energy[b] * detail::henyey_greenstein(cosine, g[b]) *
                                                 share[b] * std::exp(-depth[b]);
                                }
                                const double dist = std::max(gap, settings.receiver_radius[k]);
                                const double reach =
                                    atmosphere == nullptr
                                        ? dist
                                        : c * atmosphere->straight_time(pos, scene.receivers[k]);
                                acc.add(k, counted.data(), 1.0 / (dist * dist), travelled + reach,
                                        -toward, hits.data(), atmosphere == nullptr ? -1.0 : path + gap);
                            }
                            const double cosine = detail::henyey_greenstein_cosine(mean_g, rng);
                            const double azimuth = 2.0 * kPi * rng.uniform();
                            const Vec3 helper =
                                std::abs(dir.x) < 0.9 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
                            const Vec3 t = normalized(cross(dir, helper));
                            const Vec3 u = cross(dir, t);
                            const double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
                            dir = normalized(dir * cosine + (t * std::cos(azimuth) +
                                                             u * std::sin(azimuth)) * sine);
                            const double drawn = detail::henyey_greenstein(cosine, mean_g);
                            for (std::size_t b = 0; b < bands; ++b) {
                                energy[b] *= detail::henyey_greenstein(cosine, g[b]) / drawn;
                                seen[b] = rains ? 0.0 : 1.0;  // the rain covered it
                            }
                        }
                        double peak = 0.0;
                        for (std::size_t b = 0; b < bands; ++b) peak = std::max(peak, energy[b]);
                        if (peak < cutoff) break;
                        if (!weigh(ray_state, window)) break;
                        continue;
                    }

                    const auto face = static_cast<std::size_t>(hit.face);
                    Vec3 n = bvh.normal(face);
                    const bool from_back = dot(n, dir) > 0.0;
                    if (from_back) n = -n;  // face the side the ray came from
                    last = hit.face;
                    // The ray leaves the face on the side it came from (when it reflects).
                    away = n;
                    leaving = true;

                    const double *alpha = scene.absorption + face * bands;
                    // What passes through into the room beyond, and what reflects.
                    const double *tau =
                        scene.transmission ? scene.transmission + face * bands : nullptr;
                    std::int64_t beyond = room;
                    if (tau != nullptr && scene.face_room != nullptr) {
                        beyond = scene.face_room[2 * face + (from_back ? 0 : 1)];
                    }
                    const bool passes =
                        tau != nullptr && (scene.face_room == nullptr || beyond >= 0);
                    double passed = 0.0, kept = 0.0;
                    for (std::size_t b = 0; b < bands; ++b) {
                        through[b] = passes ? energy[b] * tau[b] : 0.0;
                        passed += through[b];
                        kept += energy[b] * (1.0 - alpha[b]);
                    }
                    if (passed > 0.0 && !(kept > 0.0)) {
                        // An opening: the ray goes on into the next room, as it was.
                        std::copy(through.begin(), through.begin() + bands, energy.begin());
                        room = beyond;
                        away = -n;
                        continue;
                    }
                    std::copy(seen.begin(), seen.begin() + bands, seen_before.begin());
                    std::copy(alone.begin(), alone.begin() + bands, alone_before.begin());
                    ++order;
                    const std::int64_t group =
                        scene.groups > 0 ? scene.face_group[face] : std::int64_t{-1};
                    if (group >= 0) ++hits[static_cast<std::size_t>(group)];

                    const double *surface = scene.scattering + face * bands;
                    if (scene.surface_size != nullptr && order >= settings.size_from_order) {
                        size_scattering(scene.surface_size[2 * face], scene.surface_size[2 * face + 1],
                                        leg, -dot(n, dir), c, bands, scene.band_hz, surface,
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
                    // The weight window's expectation is that of rays in the source's room.
                    if (room == scene.source_room) {
                        out.ray_energy[window] += mean / static_cast<double>(bands);
                        out.reflections[window] += 1.0;
                    }

                    if (settings.diffuse_rain || settings.specular_rain) {
                        const Vec3 origin = pos + n * offset;
                        for (std::size_t k = 0; k < n_rec; ++k) {
                            const Vec3 v = scene.receivers[k] - origin;
                            const double dist = std::max(norm(v), settings.receiver_radius[k]);
                            const double cos_theta = dot(n, v) / dist;
                            if (cos_theta <= 0.0) continue;
                            if (atmosphere != nullptr &&
                                atmosphere->shadowed(origin, scene.receivers[k])) {
                                continue;
                            }
                            const Vec3 toward = v / norm(v);
                            // Through the surfaces on the way that let sound through.
                            if (!detail::passing_share(scene, origin, toward, norm(v), last,
                                                       share.data())) {
                                continue;
                            }
                            if (plants != nullptr) {
                                plants->depth(origin, toward, norm(v), depth.data(), line_spans,
                                              line_cuts);
                                for (std::size_t b = 0; b < bands; ++b) share[b] *= std::exp(-depth[b]);
                            }
                            const double factor = cos_theta / (kPi * dist * dist);
                            // In an atmosphere: the time the straight line takes.
                            const double reach =
                                atmosphere == nullptr
                                    ? dist
                                    : c * atmosphere->straight_time(origin, scene.receivers[k]);
                            // All reflected energy radiates with secondary sources, but the
                            // specular-only share of a path the early stage computes
                            // exactly stays out.
                            const bool covered =
                                settings.specular_rain && order <= settings.covered_order &&
                                (settings.covered_time.empty() ||
                                 (travelled + reach) / c <= settings.covered_time[k]);
                            for (std::size_t b = 0; b < bands; ++b) {
                                const double radiated =
                                    settings.specular_rain
                                        ? reflected[b] * (covered ? 1.0 - alone[b] : 1.0)
                                        : rain[b];
                                counted[b] = radiated * share[b];
                            }
                            acc.add(k, counted.data(), factor, travelled + reach, -toward,
                                    hits.data(), atmosphere == nullptr ? -1.0 : path + norm(v));
                        }
                    }
                    if (passed > 0.0) {
                        if (!settings.split_target.empty() && level(beyond) > 0.0 &&
                            family < settings.max_family) {
                            // Both ways: a copy goes on through, on the leg it was on.
                            stack.push_back(detail::RayState{pos, dir, through, seen_before,
                                                             alone_before, travelled, order - 1,
                                                             last, rng.split(), beyond, hits,
                                                             path, -n, true});
                            if (group >= 0) --stack.back().hits[static_cast<std::size_t>(group)];
                            ++family;
                        } else {
                            const double q = passed / (passed + kept);
                            if (rng.uniform() < q) {
                                for (std::size_t b = 0; b < bands; ++b) energy[b] = through[b] / q;
                                std::copy(seen_before.begin(), seen_before.begin() + bands,
                                          seen.begin());
                                std::copy(alone_before.begin(), alone_before.begin() + bands,
                                          alone.begin());
                                --order;
                                if (group >= 0) --hits[static_cast<std::size_t>(group)];
                                room = beyond;
                                away = -n;
                                continue;
                            }
                            for (std::size_t b = 0; b < bands; ++b) {
                                reflected[b] /= 1.0 - q;
                                rain[b] /= 1.0 - q;
                            }
                            peak /= 1.0 - q;
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

                    if (!weigh(ray_state, window)) break;
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
        for (std::size_t i = 0; i < total.moments.size(); ++i) total.moments[i] += p.moments[i];
        total.escaped_rays += p.escaped_rays;
        total.traced_rays += p.traced_rays;
    }
    return total;
}

}  // namespace acoustic_spaces
