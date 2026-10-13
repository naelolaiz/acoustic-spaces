# acoustic_spaces

Open-source room acoustics simulator and auralisation toolkit, in the spirit of
CATT-Acoustic and ODEON: load a 3D model of a space, place sources and listeners,
compute impulse responses, and listen to the result in binaural, Ambisonics or
loudspeaker formats.

**Status: pre-alpha.** This repository currently holds the project foundations
(package layout, compiled core, scene file schema, CI). The simulation engine,
spatial encoders and real-time convolver follow milestone by milestone.

## Design in one paragraph

Python is the main language; a small C++ core (bound with nanobind) carries the
parts that have hard real-time deadlines. Every simulation runs in two stages with
selectable methods: deterministic ray tracing (or image sources) for the early
reflections and stochastic processing for the late reverberation. Both write a
format-independent *directional reflectogram*, from which binaural, Ambisonics and
loudspeaker impulse responses are encoded, so one simulation serves every output
format.

## Install from source

Requirements: Python 3.10+, a C++17 compiler and CMake 3.18+.

```sh
git clone https://github.com/naelolaiz/acoustic_spaces_simulator
cd acoustic_spaces_simulator
python -m pip install -e .[dev]
```

Check the installation and validate the example scene:

```sh
acoustic-spaces info
acoustic-spaces validate examples/shoebox/scene.yaml
pytest
```

## Scene files

A scene is a YAML file that names the geometry, the materials, which surface gets
which material, and where the sources and listeners are. See
[`examples/shoebox/scene.yaml`](examples/shoebox/scene.yaml) and the schema in
[`src/acoustic_spaces/scene/model.py`](src/acoustic_spaces/scene/model.py).
Frequency-dependent data uses eight octave bands from 63 Hz to 8 kHz.

## Repository layout

```
src/acoustic_spaces/   Python package
  scene/               scene model and YAML schema
  cli.py               command-line interface
cpp/include/           header-only C++ core (no Python dependency)
cpp/bindings/          nanobind module exposing the core as acoustic_spaces._native
tests/                 pytest suite, run on Linux, macOS and Windows in CI
examples/              example scenes
```

## Roadmap

- M0 Foundations: this.
- M1 Scene and geometry: CAD import (OBJ first), material database, 3D view.
- M2 Mono simulation: ray-traced early reflections, stochastic tail, IR synthesis, ISO 3382 parameters.
- M3 Spatial outputs: binaural (SOFA HRTFs), Ambisonics up to 3rd order, loudspeaker layouts.
- M4 Rendering: offline convolution in Python, real-time partitioned convolver in C++.
- M5 v1 release: validation report, documentation, wheels.

## Contributing

Pull requests are welcome. Run `ruff check . && ruff format --check . && pytest`
before opening one; CI runs the same checks on all three platforms.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
