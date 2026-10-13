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

// First-order pattern a + (1 - a) cos(theta); energy weights average to 1 over the sphere.
struct Directivity {
    double a = 1.0;
    Vec3 forward{1.0, 0.0, 0.0};

    double energy_weight(const Vec3 &dir) const {
        const double p = a + (1.0 - a) * dot(dir, forward);
        return p * p / (a * a + (1.0 - a) * (1.0 - a) / 3.0);
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
    std::size_t bins = 0, dir_bins = 0;
    std::vector<double> energy;       // receivers x bands x bins
    std::vector<double> directional;  // receivers x directions x bands x dir_bins
    std::size_t escaped_rays = 0;
};

namespace detail {

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
        settings.dir_time_step <= 0.0 ||
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
        return res;
    };
    std::vector<StochasticResult> partial(threads);
    for (auto &p : partial) p = make_result();

    parallel_for(settings.rays, threads, [&](std::size_t begin, std::size_t end,
                                             std::size_t worker) {
        StochasticResult &out = partial[worker];
        detail::Accumulator acc{scene, settings, out};
        std::array<double, kMaxBands> energy{}, reflected{}, rain{};

        for (std::size_t ray = begin; ray < end; ++ray) {
            Rng rng(settings.seed, ray);
            Vec3 dir = uniform_sphere(rng);
            const double start = scene.directivity.energy_weight(dir) /
                                 static_cast<double>(settings.rays);
            energy.fill(0.0);
            for (std::size_t b = 0; b < bands; ++b) energy[b] = start;
            const double cutoff = start * 1e-14;

            Vec3 pos = scene.source;
            double travelled = 0.0;
            int order = 0;
            bool specular_only = true;
            bool detectable = true;
            std::int64_t last = -1;

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
                        acc.add(k, energy.data(), (t1 - t0) / sphere_volume[k], distance, -dir);
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
                double peak = 0.0;
                for (std::size_t b = 0; b < bands; ++b) {
                    reflected[b] = energy[b] * (1.0 - alpha[b]);
                    rain[b] = reflected[b] * scatter[b];
                    peak = std::max(peak, reflected[b]);
                }

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
            }
        }
    });

    StochasticResult total = make_result();
    for (const auto &p : partial) {
        for (std::size_t i = 0; i < total.energy.size(); ++i) total.energy[i] += p.energy[i];
        for (std::size_t i = 0; i < total.directional.size(); ++i) {
            total.directional[i] += p.directional[i];
        }
        total.escaped_rays += p.escaped_rays;
    }
    return total;
}

}  // namespace acoustic_spaces
