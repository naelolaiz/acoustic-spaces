// Runs the fit's arithmetic tests (cut_selftest.hpp) in a build of the header alone, for compilers
// and settings that the package does not build with. Prints each count and exits with 1 if any
// check has a mismatch or the arithmetic guard refuses the build.
// Usage: fit_selftest [CASES [SEED]]
#include <cstdio>
#include <cstdlib>

#include "acoustic_spaces/cut_selftest.hpp"

int main(int argc, char **argv) {
    const std::size_t cases = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
    const std::uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
    int status = 0;
    for (const auto &[name, count] : acoustic_spaces::cut_selftest::run(cases, seed)) {
        std::printf("%s %llu\n", name.c_str(), static_cast<unsigned long long>(count));
        if (name.size() > 11 && name.compare(name.size() - 11, 11, "_mismatches") == 0 && count != 0) status = 1;
    }
    const auto g = acoustic_spaces::cut_selftest::guard();
    std::printf("arithmetic_constant %d\nprobe_failed '%s'\nrounding_to_nearest %d\n", g.arithmetic_constant ? 1 : 0,
                g.probe_failed.c_str(), g.rounding_to_nearest ? 1 : 0);
    if (!g.arithmetic_constant || !g.probe_failed.empty() || !g.rounding_to_nearest) status = 1;
    return status;
}
