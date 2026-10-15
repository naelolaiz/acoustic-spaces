// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings for the C++ core (acoustic_spaces._native).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "acoustic_spaces/bvh.hpp"
#include "acoustic_spaces/convolver.hpp"
#include "acoustic_spaces/direct_convolution.hpp"
#include "acoustic_spaces/fdtd.hpp"
#include "acoustic_spaces/fft.hpp"
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

using Kernels = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using Spectrum =
    nb::ndarray<const std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using FloatResponses = nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using FloatBlock = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Convolver = as::PartitionedConvolver<float>;

nb::ndarray<nb::numpy, std::complex<double>> real_fft(InArray x) {
    as::RealFft<double> fft(x.shape(0));
    std::vector<std::complex<double>> out(fft.bins());
    fft.forward(x.data(), out.data());
    return to_numpy(std::move(out), {fft.bins()});
}

nb::ndarray<nb::numpy, double> inverse_real_fft(Spectrum bins, std::size_t n) {
    as::RealFft<double> fft(n);
    if (bins.shape(0) != fft.bins()) {
        throw std::invalid_argument("inverse_real_fft: expected n / 2 + 1 bins");
    }
    std::vector<double> out(n);
    fft.inverse(bins.data(), out.data());
    return to_numpy(std::move(out), {n});
}

nb::ndarray<nb::numpy, float> convolver_process(Convolver &convolver, FloatBlock block) {
    if (block.shape(0) != convolver.block() || block.shape(1) != convolver.inputs()) {
        throw std::invalid_argument("Convolver.process: expected a (block, inputs) array");
    }
    std::vector<float> out(convolver.block() * convolver.outputs());
    {
        nb::gil_scoped_release release;
        convolver.process(block.data(), out.data());
    }
    return to_numpy(std::move(out), {convolver.block(), convolver.outputs()});
}

nb::ndarray<nb::numpy, double> scatter_kernels(std::size_t length, Indices positions,
                                               Indices index, InArray amplitudes,
                                               Kernels kernels) {
    const std::size_t n = positions.shape(0);
    if (index.shape(0) != n || amplitudes.shape(0) != n) {
        throw std::invalid_argument("scatter_kernels: inputs must have the same length");
    }
    const std::size_t channels = kernels.shape(1);
    std::vector<double> out;
    {
        nb::gil_scoped_release release;
        out = as::scatter_kernels(length, positions.data(), index.data(), amplitudes.data(), n,
                                  kernels.data(), kernels.shape(0), channels, kernels.shape(2));
    }
    return to_numpy(std::move(out), {channels, length});
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

nb::ndarray<nb::numpy, bool> occluded(const as::Bvh &bvh, Points origins, Points directions,
                                      InArray t_max, Indices ignore) {
    const std::size_t n = origins.shape(0);
    if (directions.shape(0) != n || t_max.shape(0) != n || ignore.shape(0) != n) {
        throw std::invalid_argument("occluded: inputs must have the same length");
    }
    std::vector<std::uint8_t> blocked(n);
    {
        nb::gil_scoped_release release;
        as::parallel_for(n, 0, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t i = begin; i < end; ++i) {
                const as::Vec3 o{origins(i, 0), origins(i, 1), origins(i, 2)};
                const as::Vec3 d{directions(i, 0), directions(i, 1), directions(i, 2)};
                blocked[i] = bvh.occluded(o, d, t_max(i), ignore(i)) ? 1 : 0;
            }
        });
    }
    auto *owned = new std::vector<std::uint8_t>(std::move(blocked));
    nb::capsule owner(owned,
                      [](void *p) noexcept { delete static_cast<std::vector<std::uint8_t> *>(p); });
    // NumPy reads the 0 and 1 bytes as its bools.
    const std::size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, bool>(reinterpret_cast<bool *>(owned->data()), 1, shape, owner);
}

nb::dict bvh_layout(const as::Bvh &bvh) {
    nb::dict out;
    bvh.layout([&](const char *name, const auto &values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        out[name] = to_numpy(std::vector<T>(values), {values.size()});
    });
    return out;
}

using Flags = nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, std::int64_t> specular_sequences(const as::Bvh &bvh, InArray source,
                                                        std::size_t rays, std::size_t max_order,
                                                        std::size_t threads, Flags passing) {
    const as::Vec3 s = vec3(source, "source");
    if (passing.shape(0) != 0 && passing.shape(0) != bvh.face_count()) {
        throw std::invalid_argument("specular_sequences: passing must have one per face");
    }
    for (std::size_t f = 0; f < passing.shape(0); ++f) {
        if (passing(f) > 2) throw std::invalid_argument("specular_sequences: passing is 0, 1 or 2");
    }
    const std::vector<std::uint8_t> flags(passing.data(), passing.data() + passing.shape(0));
    std::vector<std::int64_t> out;
    {
        nb::gil_scoped_release release;
        out = as::specular_sequences(bvh, s, rays, max_order, threads, flags);
    }
    const std::size_t width = std::max<std::size_t>(max_order, 1);
    return to_numpy(std::move(out), {out.size() / width, max_order});
}

using FaceRooms = nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// A source's directivity (see as::Directivity) over `bands` bands: the first-order
// pattern's a per band (empty: omni), a table of weights (elevations x azimuths x bands;
// empty: none) and the source's frame (forward, left and up in the columns).
as::Directivity source_directivity(const std::string &fn, std::size_t bands,
                                   InArray directivity_a, Matrix orientation,
                                   Kernels directivity_table) {
    if (directivity_a.shape(0) != 0 && directivity_a.shape(0) != bands) {
        throw std::invalid_argument(fn + ": directivity_a must have one per band");
    }
    if (orientation.shape(0) != 3 || orientation.shape(1) != 3) {
        throw std::invalid_argument(fn + ": orientation must be 3 x 3");
    }
    const std::size_t table_size = directivity_table.size();
    if (table_size != 0 &&
        (directivity_table.shape(0) < 2 || directivity_table.shape(2) != bands)) {
        throw std::invalid_argument(
            fn + ": directivity_table must be elevations (2 or more) x azimuths x bands");
    }
    as::Directivity directivity;
    directivity.a.assign(directivity_a.data(), directivity_a.data() + directivity_a.shape(0));
    if (table_size != 0) {
        directivity.table.assign(directivity_table.data(),
                                 directivity_table.data() + table_size);
        directivity.elevations = directivity_table.shape(0);
        directivity.azimuths = directivity_table.shape(1);
    }
    // Columns: the source's forward, left and up axes in room coordinates.
    directivity.forward = {orientation(0, 0), orientation(1, 0), orientation(2, 0)};
    directivity.left = {orientation(0, 1), orientation(1, 1), orientation(2, 1)};
    directivity.up = {orientation(0, 2), orientation(1, 2), orientation(2, 2)};
    return directivity;
}

// The room, source and directivity part of a stochastic scene.
as::StochasticScene stochastic_scene(const char *name, const as::Bvh &bvh, InArray source,
                                     InArray directivity_a, Matrix orientation,
                                     Kernels directivity_table, Matrix absorption,
                                     Matrix scattering, InArray air, Matrix surface_size,
                                     InArray band_hz, Matrix transmission, FaceRooms face_room,
                                     std::int64_t source_room) {
    const std::string fn = name;
    const std::size_t faces = bvh.face_count();
    const std::size_t bands = air.shape(0);
    if (absorption.shape(0) != faces || absorption.shape(1) != bands ||
        scattering.shape(0) != faces || scattering.shape(1) != bands) {
        throw std::invalid_argument(fn + ": coefficients must be faces x bands");
    }
    if (surface_size.shape(1) != 2 ||
        (surface_size.shape(0) != 0 &&
         (surface_size.shape(0) != faces || band_hz.shape(0) != bands))) {
        throw std::invalid_argument(
            fn + ": surface_size must be faces x 2 (or 0 x 2), with band_hz one per band");
    }
    as::StochasticScene scene;
    scene.bvh = &bvh;
    scene.source = vec3(source, "source");
    scene.directivity =
        source_directivity(fn, bands, directivity_a, orientation, directivity_table);
    scene.bands = bands;
    scene.absorption = absorption.data();
    scene.scattering = scattering.data();
    scene.air = air.data();
    if (surface_size.shape(0) != 0) {
        scene.surface_size = surface_size.data();
        scene.band_hz = band_hz.data();
    }
    if (transmission.size() != 0) {
        if (transmission.shape(0) != faces || transmission.shape(1) != bands) {
            throw std::invalid_argument(fn + ": transmission must be faces x bands (or empty)");
        }
        scene.transmission = transmission.data();
    }
    if (face_room.shape(1) != 2 || (face_room.shape(0) != 0 && face_room.shape(0) != faces)) {
        throw std::invalid_argument(fn + ": face_room must be faces x 2 (or 0 x 2)");
    }
    if (face_room.shape(0) != 0) {
        scene.face_room = face_room.data();
        for (std::size_t i = 0; i < 2 * faces; ++i) {
            if (face_room.data()[i] < -1) throw std::invalid_argument(fn + ": bad face_room");
        }
    }
    scene.source_room = source_room;
    return scene;
}

std::vector<double> to_vector(const InArray &a) { return {a.data(), a.data() + a.shape(0)}; }

// The rays trace_stochastic sends from its source: for each of the random streams
// first_ray .. first_ray + rays - 1, the direction its ray leaves in (rays x 3) and the
// source's energy weights that way (rays x bands, before sharing the energy out).
nb::tuple ray_starts(InArray directivity_a, Matrix orientation, Kernels directivity_table,
                     std::size_t bands, std::size_t rays, std::uint64_t first_ray,
                     std::uint64_t seed) {
    if (bands == 0 || bands > as::kMaxBands) {
        throw std::invalid_argument("ray_starts: 1 to 32 bands");
    }
    const as::Directivity directivity =
        source_directivity("ray_starts", bands, directivity_a, orientation, directivity_table);
    std::vector<double> directions(3 * rays), weights(rays * bands);
    {
        nb::gil_scoped_release release;
        for (std::size_t i = 0; i < rays; ++i) {
            as::Rng rng(seed, first_ray + i);
            const as::Vec3 d = as::uniform_sphere(rng);
            directions[3 * i] = d.x;
            directions[3 * i + 1] = d.y;
            directions[3 * i + 2] = d.z;
            directivity.energy_weights(d, bands, weights.data() + i * bands);
        }
    }
    return nb::make_tuple(to_numpy(std::move(directions), {rays, 3}),
                          to_numpy(std::move(weights), {rays, bands}));
}

// Ray count, timing, seed, threads and weight window of a stochastic run.
as::StochasticSettings stochastic_settings(std::size_t rays, double max_time,
                                           double speed_of_sound, int covered_order,
                                           std::uint64_t seed, std::size_t threads,
                                           double window_time_step, std::uint64_t first_ray,
                                           InArray split_target, double split_ratio,
                                           std::size_t max_split, std::size_t max_family,
                                           bool vector_scattering, int size_from_order,
                                           InArray room_level) {
    as::StochasticSettings settings;
    settings.rays = rays;
    settings.max_time = max_time;
    settings.window_time_step = window_time_step;
    settings.speed_of_sound = speed_of_sound;
    settings.covered_order = covered_order;
    settings.seed = seed;
    settings.threads = threads;
    settings.first_ray = first_ray;
    settings.split_target = to_vector(split_target);
    settings.split_ratio = split_ratio;
    settings.max_split = max_split;
    settings.max_family = max_family;
    settings.vector_scattering = vector_scattering;
    settings.size_from_order = size_from_order;
    settings.room_level = to_vector(room_level);
    return settings;
}

nb::tuple trace_stochastic(const as::Bvh &bvh, InArray source, InArray directivity_a,
                           Matrix orientation, Kernels directivity_table, Matrix absorption,
                           Matrix scattering, InArray air, Matrix surface_size,
                           InArray band_hz, bool vector_scattering, int size_from_order,
                           Matrix transmission, FaceRooms face_room, std::int64_t source_room,
                           Points receivers, Points directions, std::size_t rays,
                           double max_time, double time_step, double dir_time_step,
                           double speed_of_sound, InArray receiver_radius, bool diffuse_rain,
                           bool specular_rain,
                           int covered_order, InArray covered_time, std::uint64_t seed,
                           std::size_t threads, double window_time_step,
                           std::uint64_t first_ray, InArray split_target, double split_ratio,
                           std::size_t max_split, std::size_t max_family, InArray room_level,
                           Indices face_group, std::size_t groups, double moment_time_step) {
    as::StochasticScene scene =
        stochastic_scene("trace_stochastic", bvh, source, directivity_a, orientation,
                         directivity_table, absorption, scattering, air, surface_size, band_hz,
                         transmission, face_room, source_room);
    const std::size_t bands = scene.bands;
    scene.receivers = points(receivers);
    scene.directions = points(directions);
    if (groups > 0) {
        if (groups > as::kMaxGroups || face_group.shape(0) != absorption.shape(0)) {
            throw std::invalid_argument(
                "trace_stochastic: face_group must have one per face, with at most 16 groups");
        }
        const auto n = static_cast<std::int64_t>(groups);
        for (std::size_t i = 0; i < face_group.shape(0); ++i) {
            if (face_group.data()[i] < -1 || face_group.data()[i] >= n) {
                throw std::invalid_argument("trace_stochastic: bad face_group");
            }
        }
        scene.face_group = face_group.data();
        scene.groups = groups;
    }

    as::StochasticSettings settings = stochastic_settings(
        rays, max_time, speed_of_sound, covered_order, seed, threads, window_time_step,
        first_ray, split_target, split_ratio, max_split, max_family, vector_scattering,
        size_from_order, room_level);
    settings.time_step = time_step;
    settings.dir_time_step = dir_time_step;
    settings.receiver_radius = to_vector(receiver_radius);
    settings.diffuse_rain = diffuse_rain;
    settings.specular_rain = specular_rain;
    settings.covered_time = to_vector(covered_time);
    settings.moment_time_step = moment_time_step;

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
        to_numpy(std::move(result.moments),
                 {n_rec, result.moment_terms, bands, result.moment_bins}),
        result.escaped_rays, to_numpy(std::move(result.ray_energy), {result.windows}),
        to_numpy(std::move(result.reflections), {result.windows}), result.traced_rays);
}

using MapIndex = nb::ndarray<const std::int64_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

nb::tuple trace_map(const as::Bvh &bvh, InArray source, InArray directivity_a,
                    Matrix orientation, Kernels directivity_table, Matrix absorption,
                    Matrix scattering, InArray air, Matrix surface_size, InArray band_hz,
                    bool vector_scattering, int size_from_order, Matrix transmission,
                    FaceRooms face_room, std::int64_t source_room, InArray origin, double cell,
                    MapIndex point, InArray low, InArray high, InArray volume, InArray onset,
                    InArray covered_time, double fine_step, std::size_t fine_bins,
                    double coarse_step, std::size_t rays, double max_time,
                    double speed_of_sound, int covered_order, std::uint64_t seed,
                    std::size_t threads, double window_time_step, std::uint64_t first_ray,
                    InArray split_target, double split_ratio, std::size_t max_split,
                    std::size_t max_family, InArray room_level) {
    as::StochasticScene scene =
        stochastic_scene("trace_map", bvh, source, directivity_a, orientation,
                         directivity_table, absorption, scattering, air, surface_size, band_hz,
                         transmission, face_room, source_room);
    const std::size_t bands = scene.bands;
    if (origin.shape(0) != 2) throw std::invalid_argument("trace_map: origin must be (x, y)");
    as::MapGrid &map = scene.map;
    map.x0 = origin(0);
    map.y0 = origin(1);
    map.cell = cell;
    map.ny = point.shape(0);
    map.nx = point.shape(1);
    map.layers = point.shape(2);
    map.point.assign(point.data(), point.data() + point.size());
    map.low = to_vector(low);
    map.high = to_vector(high);
    map.volume = to_vector(volume);
    map.onset = to_vector(onset);
    map.covered_time = to_vector(covered_time);
    map.fine_step = fine_step;
    map.fine_bins = fine_bins;
    map.coarse_step = coarse_step;
    if (map.points() == 0) throw std::invalid_argument("trace_map: no map points");

    // No receivers: rain and detector spheres cost nothing, every leg reaches the map.
    as::StochasticSettings settings = stochastic_settings(
        rays, max_time, speed_of_sound, covered_order, seed, threads, window_time_step,
        first_ray, split_target, split_ratio, max_split, max_family, vector_scattering,
        size_from_order, room_level);
    settings.time_step = max_time;
    settings.dir_time_step = max_time;
    settings.diffuse_rain = false;

    as::StochasticResult result;
    {
        nb::gil_scoped_release release;
        result = as::trace_stochastic(scene, settings);
    }
    return nb::make_tuple(
        to_numpy(std::move(result.map_energy), {map.points(), bands, result.map_bins}),
        result.escaped_rays, to_numpy(std::move(result.ray_energy), {result.windows}),
        to_numpy(std::move(result.reflections), {result.windows}), result.traced_rays);
}

void make_wave_solver(as::WaveSolver *self, const as::Bvh &bvh, Indices face_material,
                      InArray beta, Matrix branches, Indices branch_material, InArray origin,
                      double cell, std::size_t nx, std::size_t ny, std::size_t nz,
                      InArray source, Points probes, double edge_admittance, std::size_t layer,
                      double layer_damping, std::size_t threads, std::size_t stencil) {
    if (face_material.shape(0) != bvh.face_count()) {
        throw std::invalid_argument("WaveSolver: one material per face");
    }
    const std::size_t count = branch_material.shape(0);
    if (branches.shape(0) != count || (count > 0 && branches.shape(1) != 3)) {
        throw std::invalid_argument("WaveSolver: m, r and q of each branch, and its material");
    }
    const std::vector<as::Vec3> probe_points = points(probes);
    const as::Vec3 o = vec3(origin, "origin"), s = vec3(source, "source");
    {
        nb::gil_scoped_release release;
        new (self) as::WaveSolver(bvh, face_material.data(), beta.data(), beta.shape(0),
                                  branches.data(), branch_material.data(), count, o, cell, nx, ny,
                                  nz, s, probe_points, edge_admittance, layer, layer_damping,
                                  threads, stencil);
    }
}

using Signal = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, double> wave_run(as::WaveSolver &solver, Signal signal) {
    const std::size_t steps = signal.shape(0);
    std::vector<double> out(solver.probes() * steps);
    {
        nb::gil_scoped_release release;
        solver.run(signal.data(), steps, out.data());
    }
    return to_numpy(std::move(out), {solver.probes(), steps});
}

nb::dict wave_layout(const as::WaveSolver &solver) {
    nb::dict out;
    solver.layout([&](const char *name, const auto &values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        out[name] = to_numpy(std::vector<T>(values), {values.size()});
    });
    return out;
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "Compiled core of acoustic_spaces.";

    m.def("convolve_direct", &convolve_direct, nb::arg("signal"), nb::arg("ir"),
          "Full linear convolution of two 1-D float64 arrays (reference implementation).");

    m.def("real_fft", &real_fft, nb::arg("x"),
          "FFT of a real signal of power-of-two length: n / 2 + 1 bins (as numpy.fft.rfft).");
    m.def("inverse_real_fft", &inverse_real_fft, nb::arg("bins"), nb::arg("n"),
          "Inverse of real_fft (as numpy.fft.irfft).");

    nb::class_<Convolver>(m, "Convolver",
                          "Non-uniformly partitioned real-time convolution (float32).")
        .def(
            "__init__",
            [](Convolver *self, FloatResponses ir, std::size_t block, std::size_t growth,
               std::size_t max_block) {
                new (self) Convolver(ir.data(), ir.shape(0), ir.shape(1), ir.shape(2), block,
                                     growth, max_block);
            },
            nb::arg("ir"), nb::arg("block"), nb::arg("growth") = 8, nb::arg("max_block") = 8192,
            "ir: (inputs, outputs, taps) float32; block: frames per call, a power of two; "
            "partitions grow `growth` times per segment up to `max_block` (growth 1: "
            "uniform).")
        .def("process", &convolver_process, nb::arg("block"),
             "One block (block, inputs) in, (block, outputs) out.")
        .def("reset", &Convolver::reset, "Forget all past input.")
        .def_prop_ro("inputs", &Convolver::inputs)
        .def_prop_ro("outputs", &Convolver::outputs)
        .def_prop_ro("taps", &Convolver::taps)
        .def_prop_ro("block", &Convolver::block)
        .def_prop_ro(
            "segments",
            [](const Convolver &c) {
                std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> out;
                for (std::size_t k = 0; k < c.segments(); ++k) {
                    out.emplace_back(c.segment_start(k), c.segment_block(k),
                                     c.segment_partitions(k));
                }
                return out;
            },
            "(first tap, block, partitions) of each segment of the response.");

    m.def("scatter_kernels", &scatter_kernels, nb::arg("length"), nb::arg("positions"),
          nb::arg("index"), nb::arg("amplitudes"), nb::arg("kernels"),
          "Adds amplitudes[i] * kernels[index[i]] (channels x taps) at sample positions[i]; "
          "returns (channels, length).");

    nb::class_<as::Bvh>(m, "Bvh", "Bounding volume hierarchy over a triangle mesh.")
        .def(
            "__init__",
            [](as::Bvh *self, Points vertices, Faces faces, const std::string &split) {
                if (split != "sah" && split != "median") {
                    throw std::invalid_argument("Bvh: split must be 'sah' or 'median'");
                }
                new (self) as::Bvh(vertices.data(), vertices.shape(0), faces.data(),
                                   faces.shape(0),
                                   split == "sah" ? as::BvhSplit::kSah : as::BvhSplit::kMedian);
            },
            nb::arg("vertices"), nb::arg("faces"), nb::arg("split") = "sah",
            "Builds the tree: split \"sah\" (by the surface area heuristic) or \"median\".")
        .def_prop_ro("face_count", &as::Bvh::face_count)
        .def("intersect", &intersect, nb::arg("origins"), nb::arg("directions"),
             nb::arg("t_max"), nb::arg("ignore"),
             "Nearest hit per ray: (distance, face), with (inf, -1) for a miss.")
        .def("occluded", &occluded, nb::arg("origins"), nb::arg("directions"),
             nb::arg("t_max"), nb::arg("ignore"),
             "Per ray, whether any face lies within t_max of its origin (ignoring face "
             "ignore[i]).")
        .def_prop_ro("scale", &as::Bvh::scale)
        .def("layout", &bvh_layout,
             "The tree as flat arrays (see Bvh::layout in bvh.hpp), for tracing elsewhere: "
             "boxes (nodes x 6), first, count, faces and triangles (faces x 9).");

    nb::class_<as::WaveSolver>(
        m, "WaveSolver",
        "Finite-difference time-domain solver on a cubic grid over a mesh (see fdtd.hpp).")
        .def("__init__", &make_wave_solver, nb::arg("bvh"), nb::arg("face_material"),
             nb::arg("beta"), nb::arg("branches"), nb::arg("branch_material"), nb::arg("origin"),
             nb::arg("cell"), nb::arg("nx"), nb::arg("ny"), nb::arg("nz"), nb::arg("source"),
             nb::arg("probes"), nb::arg("edge_admittance") = 1.0, nb::arg("layer") = 0,
             nb::arg("layer_damping") = 0.0, nb::arg("threads") = 0, nb::arg("stencil") = 7,
             "face_material: each face's material; beta: each material's constant admittance; "
             "branches (n, 3): m, r and q of each mass-resistance-stiffness branch, of "
             "branch_material; layer: cells of matched layer inside the grid's edge; "
             "stencil: 7 or 19 points.")
        .def_prop_ro("air_cells", &as::WaveSolver::air_cells)
        .def_prop_ro("boundary_cells", &as::WaveSolver::boundary_cells)
        .def_prop_ro("branch_states", &as::WaveSolver::branch_states)
        .def_prop_ro("layer_cells", &as::WaveSolver::layer_cells)
        .def_prop_ro("open", &as::WaveSolver::open)
        .def_prop_ro("stencil", &as::WaveSolver::stencil)
        .def_prop_ro("threads", &as::WaveSolver::threads)
        .def("probe_found", &as::WaveSolver::probe_found, nb::arg("probe"))
        .def("run", &wave_run, nb::arg("signal"),
             "Advances one step per sample of signal (added at the source); returns the "
             "pressure at every probe before each step, probes x steps.")
        .def("reset", &as::WaveSolver::reset, "Silence the grid.")
        .def("layout", &wave_layout,
             "The grid's layout as arrays (see WaveSolver::layout in fdtd.hpp), for "
             "stepping it elsewhere.");

    m.def("specular_sequences", &specular_sequences, nb::arg("bvh"), nb::arg("source"),
          nb::arg("rays"), nb::arg("max_order"), nb::arg("threads"), nb::arg("passing"),
          "Faces hit by deterministic specular rays, one row of max_order per ray, -1 "
          "padded. passing (one per face, or empty): 0 reflects, 1 reflects and lets rays "
          "through (each ray forks, adding rows), 2 only lets them through.");

    m.def("trace_stochastic", &trace_stochastic, nb::arg("bvh"), nb::arg("source"),
          nb::arg("directivity_a"), nb::arg("orientation"), nb::arg("directivity_table"),
          nb::arg("absorption"),
          nb::arg("scattering"), nb::arg("air"), nb::arg("surface_size"), nb::arg("band_hz"),
          nb::arg("vector_scattering"), nb::arg("size_from_order"), nb::arg("transmission"),
          nb::arg("face_room"), nb::arg("source_room"), nb::arg("receivers"),
          nb::arg("directions"),
          nb::arg("rays"), nb::arg("max_time"), nb::arg("time_step"), nb::arg("dir_time_step"),
          nb::arg("speed_of_sound"), nb::arg("receiver_radius"), nb::arg("diffuse_rain"),
          nb::arg("specular_rain"),
          nb::arg("covered_order"), nb::arg("covered_time"), nb::arg("seed"),
          nb::arg("threads"), nb::arg("window_time_step"), nb::arg("first_ray"),
          nb::arg("split_target"), nb::arg("split_ratio"), nb::arg("max_split"),
          nb::arg("max_family"), nb::arg("room_level"), nb::arg("face_group"),
          nb::arg("groups"), nb::arg("moment_time_step"),
          "Stochastic ray tracing: (energy[r, band, bin], directional[r, dir, band, bin], "
          "moments[r, term, band, moment bin], escaped rays, ray_energy[window], "
          "reflections[window], traced rays); "
          "ray_energy / reflections is the mean band-mean energy of a ray per window. The "
          "source radiates with a first-order pattern per band (directivity_a; empty: omni) "
          "times directivity_table (elevations x azimuths x bands; empty: none), in the "
          "frame whose forward, left and up axes are the columns of orientation. Rays "
          "reflect with vector-based scattering or, without vector_scattering, specularly or "
          "diffusely at random; surface_size (faces x 2: width and length of each face's "
          "surface, m; 0 x 2: none) with band_hz (centre frequency per band) adds the "
          "scattering of a surface's limited size from reflection order size_from_order on. "
          "transmission (faces x bands, or empty) is the share of sound each face lets "
          "through into the room on its far side, face_room (faces x 2, or 0 x 2) the room "
          "on the front and back of each face (-1: none), source_room the source's, and "
          "room_level (one per room, or empty) each room's weight window level relative to "
          "the source's room. With groups > 0, face_group (one per face, -1 for none) "
          "puts faces in groups, and moments sums per receiver and bin of moment_time_step "
          "the energy times 1, the reflections a ray has had off each group, n_i, and their "
          "products n_i n_j (i <= j); else it is empty.");

    m.def("ray_starts", &ray_starts, nb::arg("directivity_a"), nb::arg("orientation"),
          nb::arg("directivity_table"), nb::arg("bands"), nb::arg("rays"), nb::arg("first_ray"),
          nb::arg("seed"),
          "The rays trace_stochastic sends from its source, for the random streams "
          "first_ray .. first_ray + rays - 1: (directions[ray, 3], weights[ray, band]), the "
          "source's energy weights in each direction (averaging 1 over the sphere), "
          "directivity_a, orientation and directivity_table as for trace_stochastic.");

    m.def("trace_map", &trace_map, nb::arg("bvh"), nb::arg("source"), nb::arg("directivity_a"),
          nb::arg("orientation"), nb::arg("directivity_table"), nb::arg("absorption"),
          nb::arg("scattering"), nb::arg("air"), nb::arg("surface_size"), nb::arg("band_hz"),
          nb::arg("vector_scattering"), nb::arg("size_from_order"), nb::arg("transmission"),
          nb::arg("face_room"), nb::arg("source_room"), nb::arg("origin"), nb::arg("cell"),
          nb::arg("point"), nb::arg("low"), nb::arg("high"), nb::arg("volume"),
          nb::arg("onset"), nb::arg("covered_time"), nb::arg("fine_step"),
          nb::arg("fine_bins"), nb::arg("coarse_step"), nb::arg("rays"), nb::arg("max_time"),
          nb::arg("speed_of_sound"), nb::arg("covered_order"), nb::arg("seed"),
          nb::arg("threads"), nb::arg("window_time_step"), nb::arg("first_ray"),
          nb::arg("split_target"), nb::arg("split_ratio"), nb::arg("max_split"),
          nb::arg("max_family"), nb::arg("room_level"),
          "Stochastic ray tracing onto a coverage map: (energy[point, band, bin], escaped "
          "rays, ray_energy[window], reflections[window], traced rays). point (ny x nx x "
          "layers) holds the map point of each column of width cell from origin (x, y), "
          "-1 for none; each point is a slab of air from low to high (z) of the given "
          "volume. Bins count from each point's onset: fine_bins of fine_step, then "
          "coarse_step. Other arguments as for trace_stochastic.");
}
