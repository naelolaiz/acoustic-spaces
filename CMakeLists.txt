cmake_minimum_required(VERSION 3.18...3.31)
project(acoustic_spaces LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Header-only C++ core: real-time DSP lives here, free of any Python dependency,
# so it can later be reused by a standalone player or an audio plugin.
add_library(acoustic_spaces_core INTERFACE)
target_include_directories(acoustic_spaces_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

# NOMINSIZE: nanobind otherwise builds with -Os, which made the tracer and the
# convolver several times slower.
nanobind_add_module(_native NB_STATIC NOMINSIZE cpp/bindings/native_module.cpp)
target_link_libraries(_native PRIVATE acoustic_spaces_core)

if(MSVC)
  target_compile_options(_native PRIVATE /W4)
else()
  # No multiply-adds fused where the source has none (Apple's clang fuses them by default
  # on Apple silicon): every platform rounds alike, and the wave solver with a speed of
  # sound per cell stays, with every speed the grid's, the solver without one to the bit.
  target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()

# For CI's sanitizer job (GCC or Clang): out-of-bounds and use-after-free reads and writes,
# and undefined behaviour, stop the process. Python then needs the sanitizers' runtimes
# preloaded; see CONTRIBUTING.md.
option(ACOUSTIC_SPACES_SANITIZE "Build the native module with ASan and UBSan" OFF)
if(ACOUSTIC_SPACES_SANITIZE)
  target_compile_options(
    _native PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer -g
  )
  target_link_options(_native PRIVATE -fsanitize=address,undefined)
endif()

install(TARGETS _native LIBRARY DESTINATION acoustic_spaces)

# The example scenes and their scripts ship inside the package; acoustic_spaces.examples
# finds them.
install(
  DIRECTORY examples/
  DESTINATION acoustic_spaces/example_scenes
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.obj"
  PATTERN "*.csv"
  PATTERN "*.sofa"
  PATTERN "*.py"
  PATTERN "*.sh"
  PATTERN "*.geo"
  PATTERN "*.li8"
  PATTERN "*.ifc"
  PATTERN "*.dae"
  PATTERN "*.asc"
  PATTERN "*.geojson"
)
