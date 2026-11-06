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
// - gather_first_scattering: the scattered sound of the first reflections, gathered from
//   each receiver, for the share of it trace_stochastic leaves out (gather_ratio).
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
#include <string>
#include <thread>
#include <utility>
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

// The worker threads for `threads` asked: as many as this computer has cores for 0 or
// more, as more would only take turns on them, each with its own buffers.
inline std::size_t worker_threads(std::size_t threads) {
    const std::size_t cores = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    return threads == 0 ? cores : std::min(threads, cores);
}

// Runs body(begin, end, worker) over [0, n) on up to `threads` threads (0: all cores;
// see worker_threads). An exception that left a thread would end the process
// (std::terminate): the first a worker throws is kept and thrown here once every
// thread has finished, as is one from starting a thread, after the threads already
// started have finished.
template <typename Body>
void parallel_for(std::size_t n, std::size_t threads, Body body) {
    threads = std::max<std::size_t>(1, std::min(worker_threads(threads), n));
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
// many partitions, those that also reflect (each one forks the path), and only that
// many in all. The stochastic tracer leaves to the late part every path that reflects
// and passes through more partitions than the early stage follows.
constexpr std::size_t kMaxForks = 2;
constexpr std::size_t kMaxCrossings = 64;

// Bytes the stochastic tracer's threads' copies of the results (the energy at the
// receivers, over a coverage map and the moments) may take together: fewer threads sum
// them where more copies would not fit (one at least).
constexpr double kCopyMemory = 256.0 * 1024.0 * 1024.0;

// Face indices hit by `rays` specular rays from `source` (Fibonacci directions), up to
// `max_order` reflections each; one row of max_order per ray, -1 once a ray leaves the
// model. With `passing` (one Passing per face), a ray passes straight through faces
// that let sound through: past an opening, and both through and off a partition, each
// branch giving its own row (the reflections only, not the crossings). A branch forks
// at each partition it meets until it has passed through kMaxForks of them, so one ray
// gives at most C(max_order + kMaxForks, kMaxForks) rows (10 at order 3), all kept.
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
            while (!stack.empty()) {
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
    // With diffuse_rain alone, in still air: above 0, the rain of a reflection straight
    // from the source is shared with gather_first_scattering, which looks for those
    // reflections from each receiver in gather_ratio times as many directions as there
    // are rays. Where the rays land densely (near the source) the rain keeps most of it,
    // where the receiver's directions do (near the receiver) the gather (the balance
    // heuristic: detail::rain_share), so a ray landing next to a receiver no longer
    // brings it a burst. 0: the rain keeps it all; infinity: the gather takes it all.
    double gather_ratio = 0.0;
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
// column holds up to `layers` map points (a balcony above the stalls, the rooms on either
// side of a wall that crosses it), each a slab of air between low and high (z). Every
// leg of every ray adds its track length through a slab over the slab's volume, a
// track-length estimate of the energy there, whatever the reception mode; the receivers
// above are not needed. A point with a room counts the legs of rays in that room alone
// (with StochasticScene::face_room, or one room the rays never leave: source_room).
struct MapGrid {
    double x0 = 0.0, y0 = 0.0;  // corner of column (0, 0), m
    double cell = 1.0;          // column width, m
    std::size_t nx = 0, ny = 0, layers = 1;
    std::vector<std::int64_t> point;    // ny x nx x layers: map point index, -1 for none
    std::vector<double> low, high;      // per point: the slab's bottom and top, m
    std::vector<double> volume;         // per point: the slab's volume in its room's air, m^3
    std::vector<std::int64_t> room;     // per point: the room it lies in, -1 for any
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
            room.size() != n || onset.size() != n || covered_time.size() != n) {
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
    std::array<std::uint32_t, kMaxGroups> hits{};
    // In an atmosphere: metres along the ray so far (travelled is then the speed of
    // sound times the time), and, while `leaving`, the normal of the face the ray left
    // (pointing the way it went), which it cannot meet again until it turns back.
    double path = 0.0;
    Vec3 away{};
    bool leaving = false;
    // Partitions passed through (faces that let sound through and also reflect, not
    // openings). Once more than the early stage follows (kMaxForks) and a reflection,
    // the path is no longer the early stage's: alone drops to zero.
    std::size_t partitions = 0;
    // The specular share of a vector reflection (vector_scattering), on its own along
    // the exact specular direction: the detector spheres and the maps count it to the
    // first face that reflects it, and nothing else happens to it (no rain, splitting or
    // plants' scattering; it goes on through openings and partitions as their share).
    bool detect_only = false;
};

// A ray that passes through a partition: one more, and once past those the early stage
// follows, a path that has reflected is no longer the early stage's.
inline void pass_partition(RayState &state) {
    if (++state.partitions > kMaxForks && state.order > 0) state.alone.fill(0.0);
}

// The share of sound per band that travels from `origin` along `dir` (unit) for
// `length` metres: 1 in the open, the product of the transmission of the faces it
// crosses; false (and no share) where a face lets nothing through, or, as for a ray,
// where it would let the sound out into no room (StochasticScene::face_room). With
// `crossed`, also how far along the last face it crosses lies (0: none).
inline bool passing_share(const StochasticScene &scene, const Vec3 &origin, const Vec3 &dir,
                          double length, std::int64_t ignore, double *share,
                          double *crossed = nullptr) {
    const Bvh &bvh = *scene.bvh;
    const std::size_t bands = scene.bands;
    std::fill(share, share + bands, 1.0);
    if (crossed != nullptr) *crossed = 0.0;
    if (scene.transmission == nullptr) return !bvh.occluded(origin, dir, length, ignore);
    Vec3 pos = origin;
    double along = 0.0;
    for (std::size_t k = 0; k <= kMaxCrossings; ++k) {
        const Hit hit = bvh.intersect(pos, dir, length, ignore);
        if (!hit.found()) {
            if (crossed != nullptr) *crossed = along;
            return true;
        }
        const auto face = static_cast<std::size_t>(hit.face);
        if (scene.face_room != nullptr) {
            const bool from_back = dot(bvh.normal(face), dir) > 0.0;
            if (scene.face_room[2 * face + (from_back ? 0 : 1)] < 0) return false;
        }
        const double *tau = scene.transmission + face * bands;
        bool any = false;
        for (std::size_t b = 0; b < bands; ++b) {
            share[b] *= tau[b];
            any = any || share[b] > 0.0;
        }
        if (!any) return false;
        pos = pos + dir * hit.t;
        length -= hit.t;
        along += hit.t;
        ignore = hit.face;
    }
    return false;
}

// The rain's share of the scattered sound of a reflection straight from the source,
// with gather_ratio `ratio` (StochasticSettings::gather_ratio): the balance heuristic
// between the rays, which land cos_in / r_in^2 densely there (r_in from the source),
// and the receiver's gather, ratio cos_out / r_out^2 (r_out from the receiver). The
// gather takes the rest.
inline double rain_share(double cos_in, double r_in, double cos_out, double r_out, double ratio) {
    if (std::isinf(ratio)) return 0.0;
    const double rays = cos_in * r_out * r_out;
    const double gathered = ratio * cos_out * r_in * r_in;
    return rays + gathered > 0.0 ? rays / (rays + gathered) : 1.0;
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
    // Where add() sums the receivers' energy, directional energy and moments: out's own
    // arrays, or arrays of their shapes elsewhere (gather_first_scattering's caller's).
    double *energy_sum = out.energy.data();
    double *directional_sum = out.directional.data();
    double *moment_sum = out.moments.data();

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
             const Vec3 &arrival, const std::uint32_t *hits, double air_distance = -1.0) {
        const double time = distance / settings.speed_of_sound;
        const auto bin = static_cast<std::size_t>(time / settings.time_step);
        if (bin >= out.bins) return;
        const std::size_t bands = scene.bands;
        double *moments = nullptr;
        std::array<double, 1 + kMaxGroups + kMaxGroups * (kMaxGroups + 1) / 2> terms{};
        if (out.moment_terms > 0) {
            const std::size_t mbin = std::min(
                out.moment_bins - 1, static_cast<std::size_t>(time / settings.moment_time_step));
            moments = moment_sum + receiver * out.moment_terms * bands * out.moment_bins + mbin;
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
        double *row = energy_sum + receiver * bands * out.bins;
        double *drow = nullptr;
        std::size_t dbin = 0;
        if (!scene.directions.empty()) {
            dbin = std::min(out.dir_bins - 1,
                            static_cast<std::size_t>(time / settings.dir_time_step));
            const std::size_t d = nearest_direction(arrival);
            drow = directional_sum +
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

    // Narrows [t0, t1] (metres along `dir`, unit, from `pos`) to the part inside the box
    // that holds the map's slabs; false when none of it is.
    bool clip_to_map(const Vec3 &pos, const Vec3 &dir, double &t0, double &t1) const {
        const MapGrid &map = scene.map;
        constexpr double kFlat = 1e-12;
        const auto clip = [&](double p, double d, double lo, double hi) {
            if (std::abs(d) < kFlat) return p >= lo && p <= hi;
            double a = (lo - p) / d, b = (hi - p) / d;
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
            return t0 < t1;
        };
        return clip(pos.z, dir.z, map_bottom, map_top) &&
               clip(pos.x, dir.x, map.x0, map.x0 + map.cell * static_cast<double>(map.nx)) &&
               clip(pos.y, dir.y, map.y0, map.y0 + map.cell * static_cast<double>(map.ny));
    }

    // Whether a leg from `pos` along `dir` (unit) for `length` metres passes through the
    // box that holds the map's slabs.
    bool crosses_map(const Vec3 &pos, const Vec3 &dir, double length) const {
        double t0 = 0.0, t1 = length;
        return clip_to_map(pos, dir, t0, t1);
    }

    // Adds a leg that leaves `pos` along `dir` (unit) for `length` metres, `travelled`
    // metres from the source at `pos`, to every map slab it passes through, of a point in
    // the ray's `room` (RayState::room) or in any. A leg ends at a face: it lies in one room.
    // A chord of a bent ray adds `rate` to travelled per metre, and the air absorbs from
    // `path` on, `stretch` metres per metre of chord (path negative: as travelled).
    // Through plants, `track` (from pos along dir) weights the bands at each point.
    // The leg counts the share of `state`'s energy the detector spheres would without
    // rain (maps have none): a vector reflection's scattered share on its blended leg and
    // its specular share on its detect-only leg.
    void add_map(const Vec3 &pos, const Vec3 &dir, double length, const RayState &state,
                 double rate = 1.0, double path = -1.0, double stretch = 1.0,
                 const Track *track = nullptr) {
        const double travelled = state.travelled;
        const int order = state.order;
        const std::int64_t room = state.room;
        const double *energy = state.energy.data();
        // A blended leg's scattered share never is the early stage's (see spheres).
        const bool blended = settings.vector_scattering && order > 0 && !state.detect_only;
        const MapGrid &map = scene.map;
        constexpr double kFlat = 1e-12;
        const double inf = std::numeric_limits<double>::infinity();
        double t0 = 0.0, t1 = length;
        if (!clip_to_map(pos, dir, t0, t1)) return;
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
                if (room >= 0 && map.room[k] >= 0 && map.room[k] != room) continue;
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
                if (distance > c * settings.max_time) continue;  // in the overrun
                const double absorbing = path < 0.0 ? distance : path + stretch * middle;
                const bool covered = !blended && order <= settings.covered_order &&
                                     distance / c <= map.covered_time[k];
                const std::size_t bin = map.bin(distance / c - map.onset[k]);
                if (bin >= out.map_bins) continue;
                const double factor = (b - a) / map.volume[k];
                double *row = out.map_energy.data() + k * bands * out.map_bins;
                std::array<double, kMaxBands> weights{};
                if (track != nullptr) track->weight(middle, weights.data());
                for (std::size_t band = 0; band < bands; ++band) {
                    // The specular-only share of a covered path is the early stage's.
                    const double share =
                        std::max(state.seen[band] - (covered ? state.alone[band] : 0.0), 0.0);
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

namespace detail {

// Throws std::invalid_argument, its message starting with `name`, for a scene or
// settings the stochastic tracer cannot run.
inline void check_stochastic(const StochasticScene &scene, const StochasticSettings &settings,
                             const std::string &name) {
    if (scene.bvh == nullptr || scene.bands == 0 || scene.bands > kMaxBands ||
        (scene.surface_size != nullptr && scene.band_hz == nullptr)) {
        throw std::invalid_argument(name + ": bad scene");
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
        (scene.groups > 0 && !(settings.moment_time_step > 0.0)) ||
        !(settings.gather_ratio >= 0.0)) {
        throw std::invalid_argument(name + ": bad settings");
    }
    if (settings.gather_ratio > 0.0 &&
        (!settings.diffuse_rain || settings.specular_rain || scene.atmosphere != nullptr)) {
        throw std::invalid_argument(name +
                                    ": gather_ratio needs diffuse rain alone, in still air");
    }
    if (scene.groups > kMaxGroups || (scene.groups > 0 && scene.face_group == nullptr)) {
        throw std::invalid_argument(name + ": bad face groups");
    }
    if (scene.map.points() > 0 && !scene.map.valid()) {
        throw std::invalid_argument(name + ": bad map");
    }
    if (scene.atmosphere != nullptr &&
        (!scene.atmosphere->valid() || scene.atmosphere->buckets == 0)) {
        throw std::invalid_argument(name + ": bad atmosphere");
    }
    if (scene.vegetation != nullptr && scene.vegetation->bands != scene.bands) {
        throw std::invalid_argument(name + ": bad vegetation");
    }
}

// The moments' terms: 1, each group's n_i and each n_i n_j (i <= j); none without groups.
inline std::size_t moment_terms(const StochasticScene &scene) {
    return scene.groups > 0 ? 1 + scene.groups + scene.groups * (scene.groups + 1) / 2 : 0;
}

// The sizes of the stochastic tracer's results for the scene and settings (bins,
// dir_bins, windows, map_bins, moment_terms and moment_bins), with no arrays.
inline StochasticResult result_shape(const StochasticScene &scene,
                                     const StochasticSettings &settings) {
    StochasticResult res;
    res.bins = static_cast<std::size_t>(std::ceil(settings.max_time / settings.time_step));
    res.dir_bins = static_cast<std::size_t>(std::ceil(settings.max_time / settings.dir_time_step));
    res.windows =
        static_cast<std::size_t>(std::ceil(settings.max_time / settings.window_time_step));
    if (scene.map.points() > 0) res.map_bins = scene.map.bins(settings.max_time);
    res.moment_terms = moment_terms(scene);
    res.moment_bins = res.moment_terms > 0 ? static_cast<std::size_t>(std::ceil(
                                                 settings.max_time / settings.moment_time_step))
                                           : 0;
    return res;
}

// The stochastic tracer's results for the scene and settings, all zero.
inline StochasticResult zero_result(const StochasticScene &scene,
                                    const StochasticSettings &settings) {
    const std::size_t n_rec = scene.receivers.size(), n_dirs = scene.directions.size();
    const std::size_t bands = scene.bands;
    StochasticResult res = result_shape(scene, settings);
    res.energy.assign(n_rec * bands * res.bins, 0.0);
    if (n_dirs > 0) res.directional.assign(n_rec * n_dirs * bands * res.dir_bins, 0.0);
    res.ray_energy.assign(res.windows, 0.0);
    res.reflections.assign(res.windows, 0.0);
    if (scene.map.points() > 0) {
        res.map_energy.assign(scene.map.points() * bands * res.map_bins, 0.0);
    }
    res.moments.assign(n_rec * res.moment_terms * bands * res.moment_bins, 0.0);
    return res;
}

}  // namespace detail

inline StochasticResult trace_stochastic(const StochasticScene &scene,
                                         const StochasticSettings &settings) {
    detail::check_stochastic(scene, settings, "trace_stochastic");
    const MapGrid &map = scene.map;
    const bool has_map = map.points() > 0;
    const Atmosphere *atmosphere = scene.atmosphere;
    const Vegetation *plants = scene.vegetation;
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

    double map_bottom = 0.0, map_top = 0.0;
    if (has_map) {
        map_bottom = *std::min_element(map.low.begin(), map.low.end());
        map_top = *std::max_element(map.high.begin(), map.high.end());
    }
    // A ray's legs go on this far past its time (in metres the sound goes), without
    // reflecting there, so that a detector sphere or map cell it crosses as its time runs
    // out takes the whole chord, booked at the time of its middle, if that is in time:
    // half the longest chord of a sphere or of a map cell, twice over for the bent rays'
    // slower or faster sound.
    double overrun = 0.0;
    for (const double r : settings.receiver_radius) overrun = std::max(overrun, r);
    for (std::size_t k = 0; k < map.points(); ++k) {
        const double height = map.high[k] - map.low[k];
        overrun = std::max(overrun, 0.5 * std::sqrt(2.0 * map.cell * map.cell + height * height));
    }
    overrun *= 2.0;
    const std::size_t moment_terms = detail::moment_terms(scene);
    const std::size_t moment_bins =
        moment_terms > 0
            ? static_cast<std::size_t>(std::ceil(settings.max_time / settings.moment_time_step))
            : 0;
    // Every thread sums its own copy of the results (at the receivers, over the map and
    // the moments): keep the copies within kCopyMemory together, one thread at least.
    const double bins = std::ceil(settings.max_time / settings.time_step);
    const double dir_bins = n_dirs > 0 ? std::ceil(settings.max_time / settings.dir_time_step) : 0.0;
    const double copy =
        8.0 * static_cast<double>(bands) *
        (static_cast<double>(n_rec) * (bins + static_cast<double>(n_dirs) * dir_bins) +
         (has_map ? static_cast<double>(map.points() * map.bins(settings.max_time)) : 0.0) +
         static_cast<double>(n_rec * moment_terms * moment_bins));
    std::size_t threads = std::max<std::size_t>(
        1, std::min(worker_threads(settings.threads), settings.rays));
    if (copy > 0.0) {
        threads = std::min(threads, std::max<std::size_t>(
                                        1, static_cast<std::size_t>(kCopyMemory / copy)));
    }

    std::vector<StochasticResult> partial(threads);
    for (auto &p : partial) p = detail::zero_result(scene, settings);

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
                if (distance > max_length) continue;  // in the overrun: after the time
                // The specular-only share of a covered path is the early stage's. A vector
                // reflection's blended leg counts only the scattered share (its specular
                // share is on its detect-only leg), which never is.
                const bool covered = state.order <= settings.covered_order &&
                                     (settings.covered_time.empty() ||
                                      distance / c <= settings.covered_time[k]) &&
                                     !(settings.vector_scattering && state.order > 0 &&
                                       !state.detect_only);
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

        // Whether a straight line from `from` along `along` (unit) crosses a receiver's
        // sphere ahead, before the ray's time runs out `travelled` metres in.
        const auto reaches = [&](const Vec3 &from, const Vec3 &along, double travelled) {
            for (std::size_t k = 0; k < n_rec; ++k) {
                const Vec3 rel = scene.receivers[k] - from;
                const double s = dot(rel, along);
                const double d2 = dot(rel, rel) - s * s;
                const double r = settings.receiver_radius[k];
                if (d2 >= r * r) continue;
                const double half = std::sqrt(r * r - d2);
                if (s + half > 0.0 && s - half < max_length - travelled) return true;
            }
            return false;
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
                if (!ray_state.detect_only) ++out.traced_rays;
                auto &[pos, dir, energy, seen, alone, travelled, order, last, rng, room, hits,
                       path, away, leaving, partitions, detect_only] = ray_state;

                while (travelled < max_length) {
                    Hit hit;
                    double leg = 0.0;  // metres the sound went to the face it meets
                    // Where on its piece (track) the ray scatters in plants, if it does.
                    double scatter_at = inf;
                    if (atmosphere == nullptr) {
                        const double left = max_length - travelled;
                        hit = bvh.intersect(pos, dir, left + overrun, last);
                        double segment = hit.found() ? hit.t : left + overrun;
                        const detail::Track *plant_track = nullptr;
                        if (plants != nullptr &&
                            track.build(*plants, pos, dir, segment, energy.data())) {
                            plant_track = &track;
                            if (!detect_only) scatter_at = track.collide(rng);
                            if (hit.found() && scatter_at > hit.t - offset) scatter_at = inf;
                            segment = std::min(segment, scatter_at);
                        }
                        if (!settings.specular_rain) {
                            spheres(pos, dir, segment, ray_state, 1.0, -1.0, 1.0, plant_track);
                        }
                        if (has_map) {
                            acc.add_map(pos, dir, segment, ray_state, 1.0, -1.0, 1.0, plant_track);
                        }
                        if (scatter_at < inf ? scatter_at > left : hit.found() && hit.t > left) {
                            break;  // in the overrun
                        }
                        if (scatter_at < inf) {
                            travelled += segment;
                            pos = pos + dir * segment;
                        } else if (!hit.found()) {
                            // Out of time, or out of the model: then no face lies anywhere
                            // ahead. (The search above only looks as far as the ray can go.)
                            if (!detect_only &&
                                !bvh.occluded(pos, dir, std::numeric_limits<double>::infinity(),
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
                        const double end = max_length + overrun;
                        while (travelled < end) {
                            const BentRay before = bent;
                            const double budget =
                                1.05 * (end - travelled) / c * bent.speed() + offset;
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
                                    if (!detect_only) scatter_at = track.collide(rng);
                                    if (met.found() && scatter_at > met.t - offset) scatter_at = inf;
                                    segment = std::min(segment, scatter_at);
                                }
                                if (!settings.specular_rain) {
                                    spheres(chord.start, along, segment, ray_state, rate, path, stretch,
                                            plant_track);
                                }
                                if (has_map) {
                                    acc.add_map(chord.start, along, segment, ray_state, rate,
                                                path, stretch, plant_track);
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
                        if ((scatter_at < inf || hit.found()) && travelled > max_length) {
                            break;  // in the overrun
                        }
                        if (scatter_at == inf && !hit.found()) {
                            if (!detect_only &&
                                !bvh.occluded(pos, dir, std::numeric_limits<double>::infinity(),
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
                    if (detect_only) {
                        // Its share of a partition goes on through; the face reflects the rest.
                        if (!(passed > 0.0)) break;
                        std::copy(through.begin(), through.begin() + bands, energy.begin());
                        detail::pass_partition(ray_state);
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
                        // After this reflection; past the partitions the early stage
                        // follows, none of the path is the early stage's.
                        alone[b] = partitions > kMaxForks ? 0.0 : alone[b] * (1.0 - scatter[b]);
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
                        // A reflection straight from the source (through openings and
                        // partitions, not plants) shares its rain with the receivers'
                        // gather (gather_first_scattering).
                        const bool first =
                            settings.gather_ratio > 0.0 && order == 1 && alone_before[0] > 0.0;
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
                            const double kept =
                                first ? detail::rain_share(-dot(n, dir), travelled,
                                                           dot(n, v) / norm(v), norm(v),
                                                           settings.gather_ratio)
                                      : 1.0;
                            for (std::size_t b = 0; b < bands; ++b) {
                                const double radiated =
                                    settings.specular_rain
                                        ? reflected[b] * (covered ? 1.0 - alone[b] : 1.0)
                                        : rain[b];
                                counted[b] = radiated * share[b] * kept;
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
                                                             path, -n, true, partitions});
                            if (group >= 0) --stack.back().hits[static_cast<std::size_t>(group)];
                            detail::pass_partition(stack.back());
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
                                detail::pass_partition(ray_state);
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
                        // specular and a Lambert direction. The detector spheres and the
                        // maps count its specular share along the exact specular direction,
                        // on a leg of its own (when a sphere lies that way, the air bends it
                        // or there are maps), and on this one its scattered share unless the
                        // rain covers that.
                        const Vec3 specular = reflect(dir, n);
                        if (!settings.specular_rain &&
                            (atmosphere != nullptr || reaches(pos, specular, travelled) ||
                             (has_map && acc.crosses_map(pos, specular, max_length - travelled)))) {
                            stack.push_back(ray_state);
                            detail::RayState &share_leg = stack.back();
                            share_leg.dir = specular;
                            share_leg.detect_only = true;
                            for (std::size_t b = 0; b < bands; ++b) {
                                share_leg.energy[b] = reflected[b];
                                share_leg.seen[b] = 1.0 - scatter[b];
                            }
                        }
                        const Vec3 mixed = specular * (1.0 - p) + cosine_hemisphere(n, rng) * p;
                        dir = mixed / norm(mixed);
                        for (std::size_t b = 0; b < bands; ++b) {
                            energy[b] = reflected[b];
                            seen[b] = settings.diffuse_rain ? 0.0 : scatter[b];
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

    // The first thread's copy takes the others' sums, each freed once added, so the run
    // holds no copy more than its threads'. The same sums as from zero: a copy only ever
    // adds to its zeros, so it holds no negative zero, and 0 + x is x.
    StochasticResult total = std::move(partial[0]);
    for (std::size_t w = 1; w < partial.size(); ++w) {
        StochasticResult &p = partial[w];
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
        p = StochasticResult{};
    }
    return total;
}

// The scattered sound of the reflections straight from the source (through openings and
// partitions, not plants) that trace_stochastic's rain leaves to the receivers with
// settings.gather_ratio (above 0): from each receiver, `samples` directions of a
// Fibonacci lattice turned by `turn` (a rotation, 3 x 3 by rows; a uniformly random
// one leaves every time bin's estimate unbiased), each standing for 4 pi / samples
// steradians. Along each it finds the faces the receiver sees, through those that let
// sound through as the rain's line does, and adds what each sends it: the source's
// sound landing there straight, the face's scattered share of it radiating as Lambert
// has it, times the gather's share (1 - detail::rain_share) and the solid angle's patch
// of face (distance^2 / cosine). It adds into `energy`, `directional` and `moments`,
// laid out as trace_stochastic's (detail::result_shape), and takes the lattice's
// directions first, first + stride, ... only, so that `stride` calls (first 0 to
// stride - 1) add up to the whole lattice. The receivers are shared between threads,
// each summing its own, so the result does not depend on how many there are.
inline void gather_first_scattering(const StochasticScene &scene,
                                    const StochasticSettings &settings, std::size_t samples,
                                    const double *turn, std::size_t first, std::size_t stride,
                                    double *energy, double *directional, double *moments) {
    detail::check_stochastic(scene, settings, "gather_first_scattering");
    if (samples == 0 || turn == nullptr || !(settings.gather_ratio > 0.0) ||
        scene.map.points() > 0 || stride == 0 || first >= stride) {
        throw std::invalid_argument(
            "gather_first_scattering: needs directions, a turn, a gather_ratio above 0, "
            "first below stride and no map");
    }
    StochasticResult out = detail::result_shape(scene, settings);
    const Bvh &bvh = *scene.bvh;
    const Vegetation *plants = scene.vegetation;
    const std::size_t bands = scene.bands;
    const std::size_t n_rec = scene.receivers.size();
    const double c = settings.speed_of_sound;
    const double max_length = c * settings.max_time;
    // As far as an arrival may come from and still fall in the last time bin.
    const double horizon = static_cast<double>(out.bins) * settings.time_step * c;
    const double offset = 1e-7 * bvh.scale();
    const double solid = 4.0 * kPi / static_cast<double>(samples);
    std::vector<Vec3> lattice(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        const Vec3 f = fibonacci_sphere(i, samples);
        lattice[i] = {turn[0] * f.x + turn[1] * f.y + turn[2] * f.z,
                      turn[3] * f.x + turn[4] * f.y + turn[5] * f.z,
                      turn[6] * f.x + turn[7] * f.y + turn[8] * f.z};
    }
    parallel_for(n_rec, settings.threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        detail::Accumulator acc{scene, settings, out};
        acc.energy_sum = energy;
        acc.directional_sum = directional;
        acc.moment_sum = moments;
        std::array<double, kMaxBands> through{}, from_source{}, weights{}, scatter{}, counted{},
            depth{};
        std::array<std::uint32_t, kMaxGroups> hits{};
        std::vector<Vegetation::Span> line_spans;
        std::vector<double> line_cuts;
        for (std::size_t k = begin; k < end; ++k) {
            const Vec3 receiver = scene.receivers[k];
            const double radius = settings.receiver_radius[k];
            // What face `f` at x (normal towards the receiver, d metres away along unit
            // w) sends the receiver, through faces that let `through` of it pass.
            const auto add = [&](std::size_t f, const Vec3 &x, const Vec3 &normal, double d,
                                 const Vec3 &w) {
                const double *alpha = scene.absorption + f * bands;
                if (std::all_of(alpha, alpha + bands, [](double a) { return !(a < 1.0); })) return;
                const double cos_out = -dot(normal, w);
                const Vec3 back = scene.source - x;
                const double r_in = norm(back);
                if (!(cos_out > 0.0) || !(r_in > 0.0) || r_in > max_length) return;
                const double cos_in = dot(normal, back) / r_in;
                if (!(cos_in > 0.0)) return;  // the source is on the face's other side
                const Vec3 origin = x + normal * offset;
                const Vec3 v = receiver - origin;
                const double gap = norm(v);
                const double dist = std::max(gap, radius);
                const double cos_theta = dot(normal, v) / dist;
                if (!(gap > 0.0) || cos_theta <= 0.0) return;
                // The source's sound on its straight way there, as a ray's.
                const Vec3 out_dir = back * (-1.0 / r_in);
                double crossed = 0.0;
                if (!detail::passing_share(scene, scene.source, out_dir, r_in - offset, -1,
                                           from_source.data(), &crossed)) {
                    return;
                }
                scene.directivity.energy_weights(out_dir, bands, weights.data());
                const double *surface = scene.scattering + f * bands;
                if (scene.surface_size != nullptr && 1 >= settings.size_from_order) {
                    // As a ray's leg, from the last face it passed through.
                    size_scattering(scene.surface_size[2 * f], scene.surface_size[2 * f + 1],
                                    r_in - crossed, cos_in, c, bands, scene.band_hz, surface,
                                    scatter.data());
                } else {
                    std::copy(surface, surface + bands, scatter.begin());
                }
                double any = 0.0;
                for (std::size_t b = 0; b < bands; ++b) {
                    counted[b] = weights[b] * from_source[b] * (1.0 - alpha[b]) * scatter[b] *
                                 through[b];
                    any += counted[b];
                }
                if (!(any > 0.0)) return;
                if (plants != nullptr) {
                    plants->depth(scene.source, out_dir, r_in, depth.data(), line_spans,
                                  line_cuts);
                    for (std::size_t b = 0; b < bands; ++b) counted[b] *= std::exp(-depth[b]);
                    plants->depth(origin, v / gap, gap, depth.data(), line_spans, line_cuts);
                    for (std::size_t b = 0; b < bands; ++b) counted[b] *= std::exp(-depth[b]);
                }
                const double share =
                    1.0 - detail::rain_share(cos_in, r_in, dot(normal, v) / gap, gap,
                                             settings.gather_ratio);
                const double factor = cos_in / (4.0 * kPi * r_in * r_in) *
                                      cos_theta / (kPi * dist * dist) * d * d / cos_out *
                                      solid * share;
                hits.fill(0);
                const std::int64_t group =
                    scene.groups > 0 ? scene.face_group[f] : std::int64_t{-1};
                if (group >= 0) hits[static_cast<std::size_t>(group)] = 1;
                acc.add(k, counted.data(), factor, r_in + dist, v * (-1.0 / gap), hits.data());
            };
            // i += stride, stopping at the end however large stride is.
            for (std::size_t i = first; i < samples;
                 i = samples - i > stride ? i + stride : samples) {
                const Vec3 &w = lattice[i];
                std::fill(through.begin(), through.begin() + bands, 1.0);
                Vec3 pos = receiver;
                std::int64_t ignore = -1;
                double d = 0.0;
                // The faces the receiver sees this way: the first, and those beyond the
                // faces that let sound through, as many as the rain's line crosses.
                for (std::size_t j = 0; j <= kMaxCrossings; ++j) {
                    const Hit hit = bvh.intersect(pos, w, horizon - d, ignore);
                    if (!hit.found()) break;
                    d += hit.t;
                    pos = pos + w * hit.t;
                    const auto f = static_cast<std::size_t>(hit.face);
                    const Vec3 face_normal = bvh.normal(f);
                    add(f, pos, dot(face_normal, w) > 0.0 ? -face_normal : face_normal, d, w);
                    if (scene.transmission == nullptr) break;
                    // The rain's line goes the other way, from the face to the receiver.
                    if (scene.face_room != nullptr &&
                        scene.face_room[2 * f + (dot(face_normal, w) < 0.0 ? 0 : 1)] < 0) {
                        break;
                    }
                    const double *tau = scene.transmission + f * bands;
                    bool any = false;
                    for (std::size_t b = 0; b < bands; ++b) {
                        through[b] *= tau[b];
                        any = any || through[b] > 0.0;
                    }
                    if (!any) break;
                    ignore = hit.face;
                }
            }
        }
    });
}

}  // namespace acoustic_spaces
