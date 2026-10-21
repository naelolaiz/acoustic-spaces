// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings for the C++ core (acoustic_spaces._native).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cmath>
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
#include "acoustic_spaces/edge_cull.hpp"
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

std::vector<as::CullPlane> cull_planes(const Points &normals, const InArray &offsets,
                                       const Points &low, const Points &high,
                                       const InArray &spread) {
    const std::size_t count = normals.shape(0);
    if (offsets.shape(0) != count || low.shape(0) != count || high.shape(0) != count ||
        spread.shape(0) != count) {
        throw std::invalid_argument("an offset, a box and a spread per plane");
    }
    std::vector<as::CullPlane> planes(count);
    for (std::size_t j = 0; j < count; ++j) {
        planes[j].normal = {normals(j, 0), normals(j, 1), normals(j, 2)};
        planes[j].offset = offsets(j);
        planes[j].low = {low(j, 0), low(j, 1), low(j, 2)};
        planes[j].high = {high(j, 0), high(j, 1), high(j, 2)};
        planes[j].spread = spread(j);
    }
    return planes;
}

nb::tuple pairs_to_numpy(as::Pairs &&out) {
    const std::size_t n = out.first.size();
    return nb::make_tuple(to_numpy(std::move(out.first), {n}),
                          to_numpy(std::move(out.second), {n}));
}

nb::tuple reflecting_pairs(Points starts, Points ends, Points normals, InArray offsets,
                           Points low, Points high, InArray spread, InArray end, double offset,
                           double grazing, std::size_t threads) {
    if (ends.shape(0) != starts.shape(0)) {
        throw std::invalid_argument("reflecting_pairs: one end per start of an edge");
    }
    const std::vector<as::Vec3> a = points(starts), b = points(ends);
    const std::vector<as::CullPlane> planes = cull_planes(normals, offsets, low, high, spread);
    const as::Vec3 e = vec3(end, "end");
    as::Pairs out;
    {
        nb::gil_scoped_release release;
        out = as::reflecting_pairs(a, b, planes, e, offset, grazing, threads);
    }
    return pairs_to_numpy(std::move(out));
}

nb::tuple reaching_pairs(Points starts, Points ends, Points normals, InArray offsets,
                         Points low, Points high, InArray spread, InArray end, double offset,
                         double grazing, Indices row_edge, InArray t_s, InArray rho_s,
                         double max_distance, double tolerance, double slack,
                         std::size_t threads) {
    if (ends.shape(0) != starts.shape(0)) {
        throw std::invalid_argument("reaching_pairs: one end per start of an edge");
    }
    const std::size_t rows = row_edge.shape(0);
    if (t_s.shape(0) != rows || rho_s.shape(0) != rows) {
        throw std::invalid_argument("reaching_pairs: a position and a distance per row");
    }
    std::vector<std::int64_t> edge(row_edge.data(), row_edge.data() + rows);
    for (const std::int64_t i : edge) {
        if (i < 0 || static_cast<std::size_t>(i) >= starts.shape(0)) {
            throw std::invalid_argument("reaching_pairs: row_edge out of range");
        }
    }
    const std::vector<as::Vec3> a = points(starts), b = points(ends);
    const std::vector<as::CullPlane> planes = cull_planes(normals, offsets, low, high, spread);
    const as::Vec3 e = vec3(end, "end");
    const std::vector<double> ts(t_s.data(), t_s.data() + rows);
    const std::vector<double> rs(rho_s.data(), rho_s.data() + rows);
    as::Pairs out;
    {
        nb::gil_scoped_release release;
        out = as::reaching_pairs(a, b, planes, e, offset, grazing, edge, ts, rs, max_distance,
                                 tolerance, slack, threads);
    }
    return pairs_to_numpy(std::move(out));
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

// The atmosphere of an outdoor stochastic run, into `air`: rows of (height above the
// ground, speed of sound in still air, wind speed) and the way the wind blows (x, y).
// False with no rows (None): the rays go straight.
bool atmosphere(const char *name, const Points &rows, const InArray &wind_direction,
                double ground, double chord_sagitta, as::Atmosphere &air) {
    if (!rows.is_valid() || rows.shape(0) == 0) return false;
    const std::string fn = name;
    if (!wind_direction.is_valid() || wind_direction.shape(0) != 2) {
        throw std::invalid_argument(fn + ": wind_direction must be (x, y)");
    }
    for (std::size_t i = 0; i < rows.shape(0); ++i) {
        air.height.push_back(rows(i, 0));
        air.still.push_back(rows(i, 1));
        air.wind.push_back(rows(i, 2));
    }
    air.wind_x = wind_direction(0);
    air.wind_y = wind_direction(1);
    air.ground = ground;
    air.sagitta = chord_sagitta;
    if (!air.valid()) {
        throw std::invalid_argument(
            fn + ": atmosphere rows must start at height 0 and rise, with speeds above the "
                 "wind's, and wind_direction a unit vector");
    }
    air.prepare();
    return true;
}

// The plants of an outdoor stochastic run, into `out` (see vegetation.hpp): rows of 8
// per volume and medium in it (kind 0, medium, low, high, first corner in outline,
// corners, 0, 0 for an upright prism; kind 1, medium, x, y, z, horizontal and vertical
// semi-axes, 0 for a spheroid), the prisms' corners (x, y), and each medium's
// absorption, scattering and mean cosine per band and whether it stands upright.
// False with no rows (None): no plants.
bool vegetation(const char *name, std::size_t bands, const Matrix &rows, const Matrix &outline,
                const Matrix &absorption, const Matrix &scattering, const Matrix &asymmetry,
                const Flags &upright, as::Vegetation &out) {
    if (!rows.is_valid() || rows.shape(0) == 0) return false;
    const std::string fn = name;
    const auto bad = [&](const char *what) {
        return std::invalid_argument(fn + ": " + what);
    };
    if (!outline.is_valid() || !absorption.is_valid() || !scattering.is_valid() ||
        !asymmetry.is_valid() || !upright.is_valid()) {
        throw bad("plants need plant_outline, plant_absorption, plant_scattering, "
                  "plant_asymmetry and plant_upright");
    }
    const std::size_t media = upright.shape(0);
    if (rows.shape(1) != 8 || outline.shape(1) != 2 || absorption.shape(0) != media ||
        scattering.shape(0) != media || asymmetry.shape(0) != media ||
        absorption.shape(1) != bands || scattering.shape(1) != bands ||
        asymmetry.shape(1) != bands) {
        throw bad("plants must have rows of 8, plant_outline rows of (x, y), and one row "
                  "of plant_absorption, plant_scattering and plant_asymmetry per medium, "
                  "with one value per band");
    }
    out.bands = bands;
    out.outline.assign(outline.data(), outline.data() + outline.size());
    out.absorption.assign(absorption.data(), absorption.data() + absorption.size());
    out.scattering.assign(scattering.data(), scattering.data() + scattering.size());
    out.asymmetry.assign(asymmetry.data(), asymmetry.data() + asymmetry.size());
    out.upright.assign(upright.data(), upright.data() + media);
    const auto whole = [&](double v) {
        if (!(v >= 0.0) || v != std::floor(v) || v > 1e12) throw bad("bad plants row");
        return static_cast<std::size_t>(v);
    };
    for (std::size_t i = 0; i < rows.shape(0); ++i) {
        as::Vegetation::Shape shape;
        shape.kind = static_cast<int>(whole(rows(i, 0)));
        shape.medium = whole(rows(i, 1));
        if (shape.kind == 0) {
            shape.z0 = rows(i, 2);
            shape.z1 = rows(i, 3);
            shape.first = whole(rows(i, 4));
            shape.count = whole(rows(i, 5));
        } else {
            shape.centre = {rows(i, 2), rows(i, 3), rows(i, 4)};
            shape.radius = rows(i, 5);
            shape.half_height = rows(i, 6);
        }
        out.shapes.push_back(shape);
    }
    if (!out.prepare()) {
        throw bad("plants must be prisms of 3 or more corners between rising heights or "
                  "spheroids of positive size, each with a medium, absorption and scattering "
                  "not negative and mean cosines between -1 and 1");
    }
    return true;
}

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
                           Indices face_group, std::size_t groups, double moment_time_step,
                           Points atmosphere_rows, InArray wind_direction, double ground,
                           double chord_sagitta, Matrix plants, Matrix plant_outline,
                           Matrix plant_absorption, Matrix plant_scattering,
                           Matrix plant_asymmetry, Flags plant_upright) {
    as::StochasticScene scene =
        stochastic_scene("trace_stochastic", bvh, source, directivity_a, orientation,
                         directivity_table, absorption, scattering, air, surface_size, band_hz,
                         transmission, face_room, source_room);
    as::Atmosphere bending;
    if (atmosphere("trace_stochastic", atmosphere_rows, wind_direction, ground, chord_sagitta,
                   bending)) {
        scene.atmosphere = &bending;
    }
    const std::size_t bands = scene.bands;
    as::Vegetation greenery;
    if (vegetation("trace_stochastic", bands, plants, plant_outline, plant_absorption,
                   plant_scattering, plant_asymmetry, plant_upright, greenery)) {
        scene.vegetation = &greenery;
    }
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

// One chord of each of n rays bent by an atmosphere (see atmosphere.hpp): from
// origins along directions (unit), at most budget metres of arc, stopping where a ray
// turns back towards the plane of normal away (zero: none). Returns the chord ends (n x
// 3), the rays' directions there (n x 3), the metres of arc and seconds along it (n),
// and whether each turned back (n).
nb::tuple bent_chords(Points origins, Points directions, InArray budget, Points away,
                      Points atmosphere_rows, InArray wind_direction, double ground,
                      double chord_sagitta) {
    const std::size_t n = origins.shape(0);
    if (directions.shape(0) != n || budget.shape(0) != n || away.shape(0) != n) {
        throw std::invalid_argument("bent_chords: one origin, direction, budget and away per ray");
    }
    as::Atmosphere air;
    if (!atmosphere("bent_chords", atmosphere_rows, wind_direction, ground, chord_sagitta, air)) {
        throw std::invalid_argument("bent_chords: no atmosphere");
    }
    std::vector<double> ends(3 * n), tangents(3 * n), arcs(n), times(n);
    std::vector<std::uint8_t> turned(n);
    {
        nb::gil_scoped_release release;
        for (std::size_t i = 0; i < n; ++i) {
            const as::Vec3 o{origins(i, 0), origins(i, 1), origins(i, 2)};
            const as::Vec3 d{directions(i, 0), directions(i, 1), directions(i, 2)};
            const as::Vec3 a{away(i, 0), away(i, 1), away(i, 2)};
            const bool leaving = a.x != 0.0 || a.y != 0.0 || a.z != 0.0;
            as::BentRay ray(air, o, d);
            const as::BentRay::Chord chord = ray.chord(budget(i), leaving ? &a : nullptr);
            const as::Vec3 t = ray.direction();
            ends[3 * i] = chord.end.x;
            ends[3 * i + 1] = chord.end.y;
            ends[3 * i + 2] = chord.end.z;
            tangents[3 * i] = t.x;
            tangents[3 * i + 1] = t.y;
            tangents[3 * i + 2] = t.z;
            arcs[i] = chord.arc;
            times[i] = chord.time;
            turned[i] = chord.turned ? 1 : 0;
        }
    }
    return nb::make_tuple(to_numpy(std::move(ends), {n, 3}), to_numpy(std::move(tangents), {n, 3}),
                          to_numpy(std::move(arcs), {n}), to_numpy(std::move(times), {n}),
                          to_numpy(std::move(turned), {n}));
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
                    std::size_t max_family, InArray room_level, Points atmosphere_rows,
                    InArray wind_direction, double ground, double chord_sagitta, Matrix plants,
                    Matrix plant_outline, Matrix plant_absorption, Matrix plant_scattering,
                    Matrix plant_asymmetry, Flags plant_upright) {
    as::StochasticScene scene =
        stochastic_scene("trace_map", bvh, source, directivity_a, orientation,
                         directivity_table, absorption, scattering, air, surface_size, band_hz,
                         transmission, face_room, source_room);
    as::Atmosphere bending;
    if (atmosphere("trace_map", atmosphere_rows, wind_direction, ground, chord_sagitta,
                   bending)) {
        scene.atmosphere = &bending;
    }
    const std::size_t bands = scene.bands;
    as::Vegetation greenery;
    if (vegetation("trace_map", bands, plants, plant_outline, plant_absorption,
                   plant_scattering, plant_asymmetry, plant_upright, greenery)) {
        scene.vegetation = &greenery;
    }
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

using Speeds = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void make_wave_solver(as::WaveSolver *self, const as::Bvh &bvh, Indices face_material,
                      InArray beta, Matrix branches, Indices branch_material, InArray origin,
                      double cell, std::size_t nx, std::size_t ny, std::size_t nz,
                      InArray source, Points probes, double edge_admittance, std::size_t layer,
                      double layer_damping, std::size_t threads, std::size_t stencil,
                      std::optional<Speeds> speed, std::optional<Speeds> flow_x,
                      std::optional<Speeds> flow_y) {
    if (face_material.shape(0) != bvh.face_count()) {
        throw std::invalid_argument("WaveSolver: one material per face");
    }
    const std::size_t count = branch_material.shape(0);
    if (branches.shape(0) != count || (count > 0 && branches.shape(1) != 3)) {
        throw std::invalid_argument("WaveSolver: m, r and q of each branch, and its material");
    }
    if (speed && speed->shape(0) != nx * ny * nz) {
        throw std::invalid_argument("WaveSolver: one speed per cell of the grid");
    }
    if (flow_x.has_value() != flow_y.has_value()) {
        throw std::invalid_argument("WaveSolver: flow_x and flow_y go together");
    }
    if (flow_x && (flow_x->shape(0) != nx * ny * nz || flow_y->shape(0) != nx * ny * nz)) {
        throw std::invalid_argument("WaveSolver: one shift of the air per cell of the grid");
    }
    const std::vector<as::Vec3> probe_points = points(probes);
    const as::Vec3 o = vec3(origin, "origin"), s = vec3(source, "source");
    {
        nb::gil_scoped_release release;
        new (self) as::WaveSolver(bvh, face_material.data(), beta.data(), beta.shape(0),
                                  branches.data(), branch_material.data(), count, o, cell, nx, ny,
                                  nz, s, probe_points, edge_admittance, layer, layer_damping,
                                  threads, stencil, speed ? speed->data() : nullptr,
                                  flow_x ? flow_x->data() : nullptr,
                                  flow_y ? flow_y->data() : nullptr);
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
             nb::arg("speed").none() = nb::none(), nb::arg("flow_x").none() = nb::none(),
             nb::arg("flow_y").none() = nb::none(),
             "face_material: each face's material; beta: each material's constant admittance; "
             "branches (n, 3): m, r and q of each mass-resistance-stiffness branch, of "
             "branch_material; layer: cells of matched layer inside the grid's edge; "
             "stencil: 7 or 19 points; speed (nx ny nz,) float32, x fastest: each cell's "
             "speed of sound over the grid's, in (0, 1], or None for the same everywhere; "
             "flow_x, flow_y (nx ny nz,) float32: the air's velocity in cells per step, "
             "each under half, or None for still air.")
        .def_prop_ro("air_cells", &as::WaveSolver::air_cells)
        .def_prop_ro("boundary_cells", &as::WaveSolver::boundary_cells)
        .def_prop_ro("branch_states", &as::WaveSolver::branch_states)
        .def_prop_ro("layer_cells", &as::WaveSolver::layer_cells)
        .def_prop_ro("open", &as::WaveSolver::open)
        .def_prop_ro("flowing", &as::WaveSolver::flowing)
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

    m.def("reflecting_pairs", &reflecting_pairs, nb::arg("starts"), nb::arg("ends"),
          nb::arg("normals"), nb::arg("offsets"), nb::arg("low"), nb::arg("high"),
          nb::arg("spread"), nb::arg("end"), nb::arg("offset"), nb::arg("grazing"),
          nb::arg("threads"),
          "The pairs of an edge (starts to ends, n x 3) and a plane (normals, m x 3, and "
          "offsets) that a path from the edge to end can reflect off, by edge and then plane: "
          "(edge rows[k], plane columns[k]). low and high (m x 3) box each plane's faces "
          "that are not openings, spread (m) is how far those faces stray from the plane; "
          "the ray tests start within offset of the edge, and pairs whose rays meet the "
          "plane at a cosine below grazing are all kept. threads 0: all cores.");

    m.def("reaching_pairs", &reaching_pairs, nb::arg("starts"), nb::arg("ends"),
          nb::arg("normals"), nb::arg("offsets"), nb::arg("low"), nb::arg("high"),
          nb::arg("spread"), nb::arg("end"), nb::arg("offset"), nb::arg("grazing"),
          nb::arg("row_edge"), nb::arg("t_s"), nb::arg("rho_s"), nb::arg("max_distance"),
          nb::arg("tolerance"), nb::arg("slack"), nb::arg("threads"),
          "For rows of an edge (row_edge) with a source side at t_s along it and rho_s from "
          "it, the planes reflecting_pairs keeps whose image of end makes a path over the "
          "edge that touches it and is no longer than max_distance, to within slack, with "
          "the image more than tolerance / 2 off the edge's line: (rows[k], plane "
          "columns[k]), by row and then plane. Other arguments as for reflecting_pairs.");

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
          nb::arg("atmosphere").none() = nb::none(), nb::arg("wind_direction").none() = nb::none(),
          nb::arg("ground") = 0.0, nb::arg("chord_sagitta") = 0.02,
          nb::arg("plants").none() = nb::none(), nb::arg("plant_outline").none() = nb::none(),
          nb::arg("plant_absorption").none() = nb::none(),
          nb::arg("plant_scattering").none() = nb::none(),
          nb::arg("plant_asymmetry").none() = nb::none(),
          nb::arg("plant_upright").none() = nb::none(),
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
          "products n_i n_j (i <= j); else it is empty. Outdoors, atmosphere (rows of height "
          "above the ground, speed of sound in still air and wind speed, from height 0 up; "
          "None: straight rays) with wind_direction (x, y, unit) and ground (z) bends the "
          "rays, followed in chords within chord_sagitta (m) of their arcs. Plants (None: "
          "none) absorb and scatter the rays: rows of 8 per volume and medium in it (kind "
          "0, medium, low, high, first corner in plant_outline, corners, 0, 0 for a polygon "
          "of corners (x, y) raised from low to high; kind 1, medium, x, y, z, horizontal "
          "and vertical semi-axes, 0 for a spheroid), filled with media whose absorption "
          "and scattering per metre (energy) and mean cosine of scattering per band are "
          "the rows of plant_absorption, plant_scattering and plant_asymmetry (media x "
          "bands); plant_upright (one per medium) marks stems, met by the horizontal share "
          "of a ray and scattering it sideways, keeping its slope: with rain, they send "
          "each receiver what the ray's next leg carries through its sphere, averaged over "
          "the turns.");

    m.def("ray_starts", &ray_starts, nb::arg("directivity_a"), nb::arg("orientation"),
          nb::arg("directivity_table"), nb::arg("bands"), nb::arg("rays"), nb::arg("first_ray"),
          nb::arg("seed"),
          "The rays trace_stochastic sends from its source, for the random streams "
          "first_ray .. first_ray + rays - 1: (directions[ray, 3], weights[ray, band]), the "
          "source's energy weights in each direction (averaging 1 over the sphere), "
          "directivity_a, orientation and directivity_table as for trace_stochastic.");

    m.def("bent_chords", &bent_chords, nb::arg("origins"), nb::arg("directions"),
          nb::arg("budget"), nb::arg("away"), nb::arg("atmosphere"), nb::arg("wind_direction"),
          nb::arg("ground"), nb::arg("chord_sagitta"),
          "One chord of each ray bent by an atmosphere (as trace_stochastic's): from origins "
          "(n x 3) along directions (n x 3, unit), at most budget (n) metres of arc, stopping "
          "where a ray turns back towards the plane whose normal is its row of away (n x 3; "
          "zeros: none). Returns (ends[n, 3], directions at the ends[n, 3], arc lengths[n] "
          "(m), times[n] (s), turned[n]).");

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
          nb::arg("atmosphere").none() = nb::none(), nb::arg("wind_direction").none() = nb::none(),
          nb::arg("ground") = 0.0, nb::arg("chord_sagitta") = 0.02,
          nb::arg("plants").none() = nb::none(), nb::arg("plant_outline").none() = nb::none(),
          nb::arg("plant_absorption").none() = nb::none(),
          nb::arg("plant_scattering").none() = nb::none(),
          nb::arg("plant_asymmetry").none() = nb::none(),
          nb::arg("plant_upright").none() = nb::none(),
          "Stochastic ray tracing onto a coverage map: (energy[point, band, bin], escaped "
          "rays, ray_energy[window], reflections[window], traced rays). point (ny x nx x "
          "layers) holds the map point of each column of width cell from origin (x, y), "
          "-1 for none; each point is a slab of air from low to high (z) of the given "
          "volume. Bins count from each point's onset: fine_bins of fine_step, then "
          "coarse_step. Other arguments as for trace_stochastic.");
}
