// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings for the C++ core (acoustic_spaces._native).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "acoustic_spaces/bvh.hpp"
#include "acoustic_spaces/direct_convolution.hpp"
#include "acoustic_spaces/ray_tracer.hpp"

namespace nb = nanobind;
namespace as = acoustic_spaces;

namespace {

using InArray = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using Points = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Faces = nb::ndarray<const std::int64_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Indices = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Hands a std::vector to NumPy without copying; the array owns the vector.
template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T> &&data, std::vector<std::size_t> shape) {
    auto *owned = new std::vector<T>(std::move(data));
    nb::capsule owner(owned, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
    return nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.data(), owner);
}

as::Vec3 vec3(const InArray &a, const char *name) {
    if (a.shape(0) != 3) throw std::invalid_argument(std::string(name) + " must have 3 values");
    return {a(0), a(1), a(2)};
}

std::vector<as::Vec3> points(const Points &p) {
    std::vector<as::Vec3> out(p.shape(0));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = {p(i, 0), p(i, 1), p(i, 2)};
    return out;
}

OutArray convolve_direct(InArray signal, InArray ir) {
    auto *result = new std::vector<double>();
    nb::capsule owner(result,
                      [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    {
        nb::gil_scoped_release release;
        *result = as::convolve_direct(signal.data(), signal.shape(0), ir.data(), ir.shape(0));
    }
    return OutArray(result->data(), {result->size()}, owner);
}

nb::tuple intersect(const as::Bvh &bvh, Points origins, Points directions, InArray t_max,
                    Indices ignore) {
    const std::size_t n = origins.shape(0);
    if (directions.shape(0) != n || t_max.shape(0) != n || ignore.shape(0) != n) {
        throw std::invalid_argument("intersect: inputs must have the same length");
    }
    std::vector<double> t(n);
    std::vector<std::int64_t> face(n);
    {
        nb::gil_scoped_release release;
        as::parallel_for(n, 0, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                const as::Vec3 o{origins(i, 0), origins(i, 1), origins(i, 2)};
                const as::Vec3 d{directions(i, 0), directions(i, 1), directions(i, 2)};
                const as::Hit hit = bvh.intersect(o, d, t_max(i), ignore(i));
                t[i] = hit.t;
                face[i] = hit.face;
            }
        });
    }
    return nb::make_tuple(to_numpy(std::move(t), {n}), to_numpy(std::move(face), {n}));
}

nb::ndarray<nb::numpy, std::int64_t> specular_sequences(const as::Bvh &bvh, InArray source,
                                                        std::size_t rays, std::size_t max_order,
                                                        std::size_t threads) {
    const as::Vec3 s = vec3(source, "source");
    std::vector<std::int64_t> out;
    {
        nb::gil_scoped_release release;
        out = as::specular_sequences(bvh, s, rays, max_order, threads);
    }
    return to_numpy(std::move(out), {rays, max_order});
}

nb::tuple trace_stochastic(const as::Bvh &bvh, InArray source, double directivity_a,
                           InArray forward, Matrix absorption, Matrix scattering, InArray air,
                           Points receivers, Points directions, std::size_t rays,
                           double max_time, double time_step, double dir_time_step,
                           double speed_of_sound, InArray receiver_radius, bool diffuse_rain,
                           bool specular_rain,
                           int covered_order, InArray covered_time, std::uint64_t seed,
                           std::size_t threads, double window_time_step,
                           std::uint64_t first_ray, InArray split_target, double split_ratio,
                           std::size_t max_split, std::size_t max_family) {
    const std::size_t faces = bvh.face_count();
    const std::size_t bands = air.shape(0);
    if (absorption.shape(0) != faces || absorption.shape(1) != bands ||
        scattering.shape(0) != faces || scattering.shape(1) != bands) {
        throw std::invalid_argument("trace_stochastic: coefficients must be faces x bands");
    }
    as::StochasticScene scene;
    scene.bvh = &bvh;
    scene.source = vec3(source, "source");
    scene.directivity.a = directivity_a;
    scene.directivity.forward = as::normalized(vec3(forward, "forward"));
    scene.bands = bands;
    scene.absorption = absorption.data();
    scene.scattering = scattering.data();
    scene.air = air.data();
    scene.receivers = points(receivers);
    scene.directions = points(directions);

    as::StochasticSettings settings;
    settings.rays = rays;
    settings.max_time = max_time;
    settings.time_step = time_step;
    settings.dir_time_step = dir_time_step;
    settings.window_time_step = window_time_step;
    settings.speed_of_sound = speed_of_sound;
    settings.receiver_radius.assign(receiver_radius.data(),
                                    receiver_radius.data() + receiver_radius.shape(0));
    settings.diffuse_rain = diffuse_rain;
    settings.specular_rain = specular_rain;
    settings.covered_order = covered_order;
    settings.covered_time.assign(covered_time.data(),
                                 covered_time.data() + covered_time.shape(0));
    settings.seed = seed;
    settings.threads = threads;
    settings.first_ray = first_ray;
    settings.split_target.assign(split_target.data(),
                                 split_target.data() + split_target.shape(0));
    settings.split_ratio = split_ratio;
    settings.max_split = max_split;
    settings.max_family = max_family;

    as::StochasticResult result;
    {
        nb::gil_scoped_release release;
        result = as::trace_stochastic(scene, settings);
    }
    const std::size_t n_rec = scene.receivers.size();
    const std::size_t n_dirs = scene.directions.size();
    return nb::make_tuple(
        to_numpy(std::move(result.energy), {n_rec, bands, result.bins}),
        to_numpy(std::move(result.directional), {n_rec, n_dirs, bands, result.dir_bins}),
        result.escaped_rays, to_numpy(std::move(result.ray_energy), {result.windows}),
        to_numpy(std::move(result.reflections), {result.windows}), result.traced_rays);
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "Compiled core of acoustic_spaces.";

    m.def("convolve_direct", &convolve_direct, nb::arg("signal"), nb::arg("ir"),
          "Full linear convolution of two 1-D float64 arrays (reference implementation).");

    nb::class_<as::Bvh>(m, "Bvh", "Bounding volume hierarchy over a triangle mesh.")
        .def(
            "__init__",
            [](as::Bvh *self, Points vertices, Faces faces) {
                new (self) as::Bvh(vertices.data(), vertices.shape(0), faces.data(),
                                   faces.shape(0));
            },
            nb::arg("vertices"), nb::arg("faces"))
        .def_prop_ro("face_count", &as::Bvh::face_count)
        .def("intersect", &intersect, nb::arg("origins"), nb::arg("directions"),
             nb::arg("t_max"), nb::arg("ignore"),
             "Nearest hit per ray: (distance, face), with (inf, -1) for a miss.");

    m.def("specular_sequences", &specular_sequences, nb::arg("bvh"), nb::arg("source"),
          nb::arg("rays"), nb::arg("max_order"), nb::arg("threads") = 0,
          "Faces hit by deterministic specular rays, rays x max_order, -1 padded.");

    m.def("trace_stochastic", &trace_stochastic, nb::arg("bvh"), nb::arg("source"),
          nb::arg("directivity_a"), nb::arg("forward"), nb::arg("absorption"),
          nb::arg("scattering"), nb::arg("air"), nb::arg("receivers"), nb::arg("directions"),
          nb::arg("rays"), nb::arg("max_time"), nb::arg("time_step"), nb::arg("dir_time_step"),
          nb::arg("speed_of_sound"), nb::arg("receiver_radius"), nb::arg("diffuse_rain"),
          nb::arg("specular_rain"),
          nb::arg("covered_order"), nb::arg("covered_time"), nb::arg("seed"),
          nb::arg("threads"), nb::arg("window_time_step"), nb::arg("first_ray"),
          nb::arg("split_target"), nb::arg("split_ratio"), nb::arg("max_split"),
          nb::arg("max_family"),
          "Stochastic ray tracing: (energy[r, band, bin], directional[r, dir, band, bin], "
          "escaped rays, ray_energy[window], reflections[window], traced rays); "
          "ray_energy / reflections is the mean band-mean energy of a ray per window.");
}
