# acoustic_spaces

Open-source room acoustics simulator and auralisation toolkit, in the spirit of
CATT-Acoustic and ODEON: load a 3D model of a space, place sources and listeners,
compute impulse responses, and listen to the result in binaural, Ambisonics or
loudspeaker formats.

![The church example in 3D: the stone nave with its early reflection paths from the altar to the front of the nave, coloured by order up to the third](docs/images/church_paths.png)

**Status: 3.2** (see the [roadmap](docs/roadmap.md)).

## What it does

- **Rooms** from CAD and BIM models (OBJ, DXF, glTF, STL, IFC, Collada, SketchUp), from
  their dimensions or from a floor plan, with a material database and the material
  libraries of CATT-Acoustic and ODEON, walls and doors between rooms, audience seating,
  and directional, line, surface and array sources; a leak finder shows where a model
  is open: [scene files](docs/scenes.md).
- **Simulation**, in octave or third-octave bands: ray-traced early reflections with
  edge diffraction, added in pressure at low frequencies, a stochastic late tail and,
  for the lowest bands, a wave solver, both on the CPU or a GPU; ISO 3382-1 parameters,
  the STI and echo criteria for every source and listener:
  [simulation](docs/simulation.md).
- **Coverage maps** of the parameters and of the speech transmission index over the
  audience: [coverage maps](docs/coverage-maps.md).
- **Listening**: impulse responses in mono, binaural, Ambisonics (AmbiX) and loudspeaker
  formats as WAV files; dry recordings run through them offline or in real time on the
  sound card, head-tracked on headphones, or heard along a walk through the room:
  [spatial audio and listening](docs/spatial-audio.md). Three [demos](docs/demos.md),
  binaural for headphones, each with its scripts: Beethoven's 8th symphony from 61
  anechoic tracks in the concert hall example, a choir and an organ in the church, and a
  band with its sound system in the stadium; the
  [playlist](https://www.youtube.com/playlist?list=PLWz5JDREfod8) has their videos.
- **Measuring real rooms** with sine sweeps, comparing them with their model in just
  noticeable differences, and fitting the model's absorption to them:
  [measurements](docs/measurements.md).
- **A graphical editor** that does all of this in one window, starting from the bundled
  [example rooms](docs/examples.md); reflection paths and sound particles in 3D, animated
  and exported to Blender: [graphical editor](docs/editor.md).
- **Validation** against theory, against pyroomacoustics and against measured rooms,
  with the numbers and the scripts to reproduce them: the
  [validation report](docs/validation.md) says how far the results can be trusted.

## Install

```sh
pip install acoustic-spaces                # simulation, rendering and the command line
pip install "acoustic-spaces[viz,gui]"     # with the 3D view and the graphical editor
```

PyPI and every [GitHub release](https://github.com/naelolaiz/acoustic-spaces/releases)
have wheels for Linux (x86-64), macOS (Apple silicon and Intel) and Windows (x86-64),
CPython 3.10 to 3.14; elsewhere pip builds the package from source, which needs a C++17
compiler.
From a source checkout, with [uv](https://docs.astral.sh/uv/), a C++17 compiler and
CMake:

```sh
git clone https://github.com/naelolaiz/acoustic-spaces
cd acoustic-spaces
uv sync --extra viz --extra gui
```

[Getting started](docs/getting-started.md) lists the extras, the same with pip, and a
tour of the command line and the Python API.

## Quick start

```sh
acoustic-spaces examples                                   # the example rooms
acoustic-spaces simulate examples/shoebox/scene.yaml --out results --plot
acoustic-spaces view examples/gallery/scene.yaml --paths --listener "round the corner"
acoustic-spaces                                            # alone: the graphical editor
```

```python
from acoustic_spaces import Room, simulate

room = Room.from_yaml("examples/shoebox/scene.yaml")
response = simulate(room)["talker", "seat-a"]
print(response.parameters().table())
```

## Documentation

- [Getting started](docs/getting-started.md): installing, the command line and the
  Python API.
- [Graphical editor](docs/editor.md): editing scenes, drawing rooms, results and
  listening, export to Blender.
- [Example scenes](docs/examples.md): the bundled rooms and what each shows.
- [Scene files](docs/scenes.md): geometry, materials, walls and openings, audience
  seating, sources and directivity.
- [Outdoor scenes](docs/outdoor.md): the ground and the open sky, buildings and
  barriers.
- [Simulation](docs/simulation.md): early reflections, the late part, scattering,
  frequency bands, edge diffraction, low frequencies and the wave solver.
- [Coverage maps](docs/coverage-maps.md): parameters and the STI over the audience area.
- [Spatial audio and listening](docs/spatial-audio.md): binaural, Ambisonics and
  loudspeaker outputs, convolution, real-time playback and head tracking.
- [Measurements](docs/measurements.md): measuring impulse responses, comparing them with
  simulations, and fitting materials to them.
- [Validation report](docs/validation.md): how the results compare with theory, another
  simulator and measured rooms.
- [Roadmap](docs/roadmap.md),
  [changelog](https://github.com/naelolaiz/acoustic-spaces/blob/main/CHANGELOG.md)
  and [repository layout](docs/development.md).

The same pages, with an API reference generated from the docstrings, build into a
website with Sphinx (see
[CONTRIBUTING.md](https://github.com/naelolaiz/acoustic-spaces/blob/main/CONTRIBUTING.md)).

## Design in one paragraph

Python is the main language; a C++ core (bound with nanobind) carries the parts that
must be fast: the ray tracer, the wave solver and the real-time convolver. Every
simulation runs in two stages with selectable methods: deterministic ray tracing (or
image sources) for the early reflections and stochastic processing for the late
reverberation. Both write a format-independent *directional reflectogram*, from which
binaural, Ambisonics and loudspeaker impulse responses are encoded, so one simulation
serves every output format. For the lowest bands a wave solver can take over; it
records the pressure and its gradient, so the bass keeps its direction (to first order)
in every format.

## Contributing

Pull requests are welcome. Run `ruff check . && ruff format --check . && pytest -W error`
before opening one; CI runs the tests on all three platforms and ruff on Linux.
[CONTRIBUTING.md](https://github.com/naelolaiz/acoustic-spaces/blob/main/CONTRIBUTING.md)
covers the development setup, the documentation, the validation scripts and releases.

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/naelolaiz/acoustic-spaces/blob/main/LICENSE).

The built-in material table comes from
[pyroomacoustics](https://github.com/LCAV/pyroomacoustics) (MIT, see
[`materials/data/LICENSE-pyroomacoustics.txt`](https://github.com/naelolaiz/acoustic-spaces/blob/main/src/acoustic_spaces/materials/data/LICENSE-pyroomacoustics.txt)).
That table does not cite a source for most entries; check values that matter
against manufacturer data or a measured database.
