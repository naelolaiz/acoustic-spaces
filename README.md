# acoustic_spaces

Open-source room acoustics simulator and auralisation toolkit, in the spirit of
CATT-Acoustic and ODEON: load a 3D model of a space, place sources and listeners,
compute impulse responses, and listen to the result in binaural, Ambisonics or
loudspeaker formats.

**Status: pre-alpha.** Scenes can be built and checked: CAD import, a material
database, statistical reverberation estimates and a 3D view. The simulation engine,
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
python -m pip install -e ".[dev,viz]"     # viz adds the 3D view (PyVista)
```

Check the installation and look at the example scene:

```sh
acoustic-spaces info
acoustic-spaces validate examples/shoebox/scene.yaml   # errors exit with status 1
acoustic-spaces inspect examples/shoebox/scene.yaml    # geometry, materials, RT estimates
acoustic-spaces view examples/shoebox/scene.yaml       # 3D window; --screenshot out.png
acoustic-spaces materials carpet                       # search the material database
pytest
```

From Python:

```python
from acoustic_spaces.room import Room
from acoustic_spaces.viz import show_room

room = Room.from_yaml("examples/shoebox/scene.yaml")
print(room.report.summary())
print(room.problems(), room.warnings())
print(room.reverberation_time("eyring"))  # per octave band, seconds
show_room(room)
```

## Scene files

A scene is a YAML file that names the geometry, the materials, which surface gets
which material, and where the sources and listeners are. See
[`examples/shoebox/scene.yaml`](examples/shoebox/scene.yaml) and the schema in
[`src/acoustic_spaces/scene/model.py`](src/acoustic_spaces/scene/model.py).
Frequency-dependent data uses eight octave bands from 63 Hz to 8 kHz.

### Geometry

| Format | Read with | Names used for material assignment |
| --- | --- | --- |
| OBJ | built-in reader | `usemtl` material, `g` group, `o` object |
| DXF | [ezdxf](https://ezdxf.mozman.de/) (3DFACE, MESH, polyface, block references) | layer |
| glTF / GLB | [trimesh](https://trimesh.org/) | material, node and mesh names |
| STL, PLY, OFF | trimesh | file name |

OBJ is the recommended exchange format: SketchUp, Rhino, Revit and Blender all
export it with material names. Set `geometry.units` (`m`, `cm`, `mm`, `in`, `ft`)
and, when the file is +y up (Blender's OBJ default, glTF always), `geometry.up_axis: y`.

Polygons are triangulated, duplicate vertices merged and degenerate faces dropped.
`validate` then reports surfaces without a material and sources or listeners
outside the room; it warns about models that are not closed (rays would escape)
and assignments that match nothing. Normals may point either way: absorption is
applied on both sides.

### Materials

Each entry in `materials` takes coefficients from one of:

- `preset: <id>`: the built-in database (`acoustic-spaces materials` lists it);
- `file: <csv>`: a table of centre frequencies, absorption and optionally
  scattering, one row each, at any band resolution (the layout the
  [BRAS](https://doi.org/10.14279/depositonce-6726.3) benchmark uses);
- `absorption: [...]` with eight octave values, which also overrides a preset or file.

`scattering` is a number, eight values, or the id of a built-in scattering entry.
Data in other bands is averaged onto octaves; octaves with no data (63 Hz in most
tables) take the nearest octave's value and the material's `source` says which.
Surfaces get a material through `assignments` (any geometry name to a material),
by having the same name as a material, or through `default_material`.

The BRAS room models are distributed as SketchUp files only; export them to OBJ
from SketchUp to use them here.

## Repository layout

```
src/acoustic_spaces/   Python package
  scene/               scene model and YAML schema
  geometry/            CAD import, triangulation, mesh cleanup and checks
  materials/           material database and coefficient files
  room.py              a scene with its geometry loaded: materials per face, checks
  viz.py               3D view (optional, PyVista)
  cli.py               command-line interface
cpp/include/           header-only C++ core (no Python dependency)
cpp/bindings/          nanobind module exposing the core as acoustic_spaces._native
tests/                 pytest suite, run on Linux, macOS and Windows in CI
examples/              example scenes
```

## Roadmap

- M0 Foundations: package layout, compiled core, scene schema, CI. Done.
- M1 Scene and geometry: CAD import, material database, 3D view. Done.
- M2 Mono simulation: ray-traced early reflections, stochastic tail, IR synthesis, ISO 3382 parameters.
- M3 Spatial outputs: binaural (SOFA HRTFs), Ambisonics up to 3rd order, loudspeaker layouts.
- M4 Rendering: offline convolution in Python, real-time partitioned convolver in C++.
- M5 v1 release: validation report, documentation, wheels.

## Contributing

Pull requests are welcome. Run `ruff check . && ruff format --check . && pytest`
before opening one; CI runs the same checks on all three platforms.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

The built-in material table comes from
[pyroomacoustics](https://github.com/LCAV/pyroomacoustics) (MIT, see
[`materials/data/LICENSE-pyroomacoustics.txt`](src/acoustic_spaces/materials/data/LICENSE-pyroomacoustics.txt)).
That table does not cite a source for most entries; check values that matter
against manufacturer data or a measured database.
