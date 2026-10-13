// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings for the C++ core (acoustic_spaces._native).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <vector>

#include "acoustic_spaces/direct_convolution.hpp"

namespace nb = nanobind;

namespace {

using InArray = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

OutArray convolve_direct(InArray signal, InArray ir) {
    auto *result = new std::vector<double>();
    nb::capsule owner(result,
                      [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    {
        nb::gil_scoped_release release;
        *result = acoustic_spaces::convolve_direct(signal.data(), signal.shape(0), ir.data(),
                                                   ir.shape(0));
    }
    return OutArray(result->data(), {result->size()}, owner);
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "Compiled core of acoustic_spaces.";

    m.def("convolve_direct", &convolve_direct, nb::arg("signal"), nb::arg("ir"),
          "Full linear convolution of two 1-D float64 arrays (reference implementation).");
}
