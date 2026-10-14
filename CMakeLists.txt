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
  target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
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
)
