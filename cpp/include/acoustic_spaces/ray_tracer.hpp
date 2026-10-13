// SPDX-License-Identifier: GPL-3.0-or-later
//
// Geometric acoustics kernels on top of the BVH:
//
// - specular_sequences: which faces deterministic specular rays hit, in order. The
//   early-reflection finder turns these into image sources and checks them exactly.
// - trace_stochastic: energy-based stochastic ray tracing with diffuse rain, giving
//   energy per receiver, octave band and time slot (and, optionally, per arrival
//   direction) for the late part of the response.
//
// Units: a source of unit energy; results are time-integrated intensity, so the
// direct sound at distance r contributes 1 / (4 pi r^2).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

struct StochasticScene {
    const Bvh *bvh = nullptr;
    Vec3 source;
    Directivity directivity;
    std::size_t bands = 0;
    const double *absorption = nullptr;  // faces x bands
    const double *scattering = nullptr;  // faces x bands
    const double *air = nullptr;         // bands, energy attenuation per metre
    std::vector<Vec3> receivers;
    std::vector<Vec3> directions;  // unit vectors for directional bins; may be empty
};

struct StochasticResult {
    std::size_t bins = 0, dir_bins = 0, windows = 0;
    std::vector<double> energy;       // receivers x bands x bins
    std::vector<double> directional;  // receivers x directions x bands x dir_bins
    // windows: the sum and count of the band-mean energy of rays at each reflection, so
    // ray_energy / reflections is the expected energy of a ray in the window.
    std::vector<double> ray_energy, reflections;
    std::size_t escaped_rays = 0;
    std::size_t traced_rays = 0;  // rays including the copies split off
};

namespace detail {

// Everything a ray carries from one leg to the next.
struct RayState {
    Vec3 pos, dir;
    std::array<double, kMaxBands> energy;
    double travelled;
    int order;
    bool specular_only;  // no diffuse reflection yet
    bool detectable;     // the detector spheres count this leg
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
};

}  // namespace detail

inline StochasticResult trace_stochastic(const StochasticScene &scene,
                                         const StochasticSettings &settings) {
    if (scene.bvh == nullptr || scene.bands == 0 || scene.bands > kMaxBands) {
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
    const Bvh &bvh = *scene.bvh;
    const std::size_t bands = scene.bands;
    const std::size_t faces = bvh.face_count();
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

    // Probability of a diffuse bounce per face: the mean scattering coefficient.
    std::vector<double> diffuse_probability(faces, 0.0);
    for (std::size_t f = 0; f < faces; ++f) {
        double sum = 0.0;
        for (std::size_t b = 0; b < bands; ++b) sum += scene.scattering[f * bands + b];
        diffuse_probability[f] = sum / static_cast<double>(bands);
    }

    std::size_t threads = settings.threads;
    if (threads == 0) threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    threads = std::max<std::size_t>(1, std::min(threads, settings.rays));

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
        return res;
    };
    std::vector<StochasticResult> partial(threads);
    for (auto &p : partial) p = make_result();

    parallel_for(settings.rays, threads, [&](std::size_t begin, std::size_t end,
                                             std::size_t worker) {
        StochasticResult &out = partial[worker];
        detail::Accumulator acc{scene, settings, out};
        std::array<double, kMaxBands> reflected{}, rain{};

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
            stack.push_back(detail::RayState{scene.source, first_dir, start, 0.0, 0, true,
                                             true, -1, first_rng});
            std::size_t family = 1;

            while (!stack.empty()) {
                detail::RayState ray_state = stack.back();
                stack.pop_back();
                ++out.traced_rays;
                auto &[pos, dir, energy, travelled, order, specular_only, detectable, last,
                       rng] = ray_state;

                while (travelled < max_length) {
                    const Hit hit = bvh.intersect(pos, dir, max_length - travelled, last);
                    const double segment = hit.found() ? hit.t : max_length - travelled;

                    if (detectable && !settings.specular_rain) {
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
                                specular_only && order <= settings.covered_order &&
                                (settings.covered_time.empty() ||
                                 distance / c <= settings.covered_time[k]);
                            if (covered) continue;
                            acc.add(k, energy.data(), (t1 - t0) / sphere_volume[k], distance,
                                    -dir);
                        }
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
                    const double *scatter = scene.scattering + face * bands;
                    double peak = 0.0, mean = 0.0;
                    for (std::size_t b = 0; b < bands; ++b) {
                        mean += energy[b];
                        reflected[b] = energy[b] * (1.0 - alpha[b]);
                        rain[b] = reflected[b] * scatter[b];
                        peak = std::max(peak, reflected[b]);
                    }
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
                            // The specular part of a path the early stage computes exactly
                            // stays out; only its scattered part radiates.
                            const bool covered =
                                specular_only && order <= settings.covered_order &&
                                (settings.covered_time.empty() ||
                                 (travelled + dist) / c <= settings.covered_time[k]);
                            const bool all = settings.specular_rain && !covered;
                            acc.add(k, all ? reflected.data() : rain.data(),
                                    cos_theta / (kPi * dist * dist), travelled + dist, -toward);
                        }
                    }
                    if (peak < cutoff) break;

                    // Specular or diffuse continuation, re-weighted per band so each band's
                    // expected energy follows its own scattering coefficient.
                    const double p = diffuse_probability[face];
                    if (rng.uniform() < p) {
                        for (std::size_t b = 0; b < bands; ++b) energy[b] = rain[b] / p;
                        dir = cosine_hemisphere(n, rng);
                        specular_only = false;
                        detectable = !settings.diffuse_rain;  // the rain covered this leg
                    } else {
                        for (std::size_t b = 0; b < bands; ++b) {
                            energy[b] = (reflected[b] - rain[b]) / (1.0 - p);
                        }
                        dir = reflect(dir, n);
                        detectable = true;
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
        total.escaped_rays += p.escaped_rays;
        total.traced_rays += p.traced_rays;
    }
    return total;
}

}  // namespace acoustic_spaces
