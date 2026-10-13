# acoustic_spaces

Open-source room acoustics simulator and auralisation toolkit, in the spirit of
CATT-Acoustic and ODEON: load a 3D model of a space, place sources and listeners,
compute impulse responses, and listen to the result in binaural, Ambisonics or
loudspeaker formats.

**Status: pre-alpha.** Scenes can be built, checked and simulated in mono: CAD
import, a material database, ray-traced early reflections, a stochastic late tail,
impulse responses as WAV files and ISO 3382-1 parameters. Spatial encoders and the
real-time convolver follow milestone by milestone.

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
acoustic-spaces view examples/shoebox/scene.yaml --paths --max-order 2  # reflection paths
acoustic-spaces materials carpet                       # search the material database
acoustic-spaces simulate examples/shoebox/scene.yaml --out results --plot
pytest
```

`simulate` writes one impulse response per source and listener
(`results/talker__seat-a.wav`, 32-bit float), `parameters.json` with every ISO 3382-1
parameter per octave band, and with `--plot` an echogram and decay curves as PNG. It
prints the parameters as it goes:

```
talker -> seat-a: direct sound at 11.7 ms, 63 early paths
             63 Hz  125 Hz  250 Hz  500 Hz   1 kHz   2 kHz   4 kHz   8 kHz
EDT (s)      1.811   1.797   1.703   1.419   1.306   1.200   0.960   0.532
T30 (s)      2.953   2.918   2.840   2.369   2.280   2.101   1.648   0.849
C80 (dB)       1.3     1.3     1.9     3.0     3.9     4.7     6.0     9.4
G (dB)        22.8    22.8    22.1    21.0    19.3    18.1    16.9    15.5
...
```

From Python:

```python
from acoustic_spaces import Room, simulate
from acoustic_spaces.audio import write_wav
from acoustic_spaces.engine import early_paths
from acoustic_spaces.viz import plot_response, show_room

room = Room.from_yaml("examples/shoebox/scene.yaml")
print(room.report.summary())
print(room.problems(), room.warnings())
print(room.reverberation_time("eyring"))  # per octave band, seconds
show_room(room, paths=early_paths(room, max_order=2))

result = simulate(room)
response = result["talker", "seat-a"]
print(response.parameters().table())
write_wav("talker__seat-a.wav", response.impulse_response(48_000), 48_000)
plot_response(response, "talker__seat-a.png")
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

## Simulation

Each source and listener pair is simulated in two parts that add up without a
crossfade.

**Early reflections** (`simulation.early`) are exact specular paths up to
`max_order` reflections, with delay, arrival direction and energy per band. Candidate
image sources come from deterministic specular rays (`method: ray_tracing`, the
default; cost grows with `rays`, not with the number of surfaces) or from every
sequence of planes (`method: image_sources`; complete, but grows as planes to the
power of the order). Every candidate is then traced back from the listener, so the
paths found are exact either way, including paths that reflect exactly on an edge
or in a corner.

**Late reverberation** (`simulation.late`) is either:

- `stochastic_ray_tracing` (default): `rays` rays leave the source and reflect
  specularly or diffusely by each surface's scattering coefficient, losing energy
  to absorption and air. Specular paths the early part already computed are left
  out. How energy reaches the listener is set by `reception`:
  - `diffuse_rain` (default): the scattered part of every reflection radiates to
    each listener it can see; specular rays count when they cross a detector
    sphere (`receiver_radius`, 0.5 to 1 m by default and kept clear of walls).
  - `secondary_sources`: all reflected energy radiates that way, as in ODEON's late
    method. Much smoother tails, but it treats specular energy as diffuse: in the
    example box (smooth walls, absorbing ceiling) EDT comes out about 20 % shorter.
  - `detector`: classic ray tracing, everything counted by the sphere.
- `statistical`: an exponential decay at the Eyring reverberation time, at the
  level of Barron's revised theory, from `transition_ms` after the direct sound
  (by default the mixing time, the square root of the volume in milliseconds).

The two parts hand over by reflection order by default. Set `transition_ms` to
keep early paths only that long after the direct sound.

**Impulse responses** are built per octave band: early paths become fractional-delay
impulses, the late energy shapes a Dirac sequence with the density of reflections in
the room, and a zero-phase crossover bank sums the bands. The direct sound at r
metres is an impulse of height 1/r, so levels compare between files and the energy
of the direct sound at 10 m (0.01) is the reference for strength G.

**Parameters** follow ISO 3382-1 per octave band: EDT, T20, T30, C50, C80, D50, Ts
and G. `Response.parameters()` computes them from the simulated energies;
`acoustic_spaces.parameters.from_impulse_response` does the same for any mono IR.
A response that ends before it has decayed is extended by its fitted decay rate.
Air absorption follows ISO 9613-1 for the scene's temperature, humidity and pressure.

### Validation so far

The tests check, on Linux, macOS and Windows:

- an 8 x 6 x 3 m box against analytic image sources up to 4th order (1, 6, 18, 38
  and 66 paths), for both early methods and for symmetric placements where paths
  hit edges and corners: positions, delays and energies agree to rounding;
- T30 in a box with fully diffusing surfaces against Eyring with air absorption:
  within 10 %, and slightly longer as absorption grows, as Kuttruff's correction
  for the spread of free path lengths predicts;
- G in the same box against Barron's revised theory: within 1 dB;
- every ISO 3382-1 parameter on synthetic exponential decays, and the air
  absorption against the ISO 9613-1 table.

Comparison with BRAS measurements needs the BRAS models exported to OBJ (see above).

## Repository layout

```
src/acoustic_spaces/   Python package
  scene/               scene model and YAML schema
  geometry/            CAD import, triangulation, mesh cleanup and checks
  materials/           material database and coefficient files
  room.py              a scene with its geometry loaded: materials per face, checks
  engine/              early reflections, late reverberation, responses, simulate()
  air.py               ISO 9613-1 air absorption and speed of sound
  synthesis.py         octave filter bank and impulse response synthesis
  parameters.py        ISO 3382-1 parameters
  audio.py             WAV files
  viz.py               3D view, reflection paths and response plots (optional)
  cli.py               command-line interface
cpp/include/           header-only C++ core: BVH, ray tracer, convolution (no Python)
cpp/bindings/          nanobind module exposing the core as acoustic_spaces._native
tests/                 pytest suite, run on Linux, macOS and Windows in CI
examples/              example scenes
```

## Roadmap

- M0 Foundations: package layout, compiled core, scene schema, CI. Done.
- M1 Scene and geometry: CAD import, material database, 3D view. Done.
- M2 Mono simulation: ray-traced early reflections, stochastic tail, IR synthesis, ISO 3382 parameters. In review.
- M2b IR measurement: exponential sine sweeps, playback and recording, deconvolution, parameters of measured IRs.
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
