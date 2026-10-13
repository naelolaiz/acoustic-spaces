# acoustic_spaces

Open-source room acoustics simulator and auralisation toolkit, in the spirit of
CATT-Acoustic and ODEON: load a 3D model of a space, place sources and listeners,
compute impulse responses, and listen to the result in binaural, Ambisonics or
loudspeaker formats.

**Status: 1.0, beta.** Scenes can be built, checked, viewed and simulated: CAD import
or rooms from dimensions, a material database, ray-traced early reflections, a
stochastic late tail, impulse responses in mono, binaural, Ambisonics (AmbiX) and
loudspeaker formats as WAV files, ISO 3382-1 parameters, and animations of sound
particles. Dry recordings can be run through the responses offline or in real time on
the sound card, and real rooms can be measured with sine sweeps and analysed the same
way. The [validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md)
says how far the results can be trusted. A [graphical editor](#graphical-editor) does all
of this in one window, starting from the bundled example rooms.

## Design in one paragraph

Python is the main language; a small C++ core (bound with nanobind) carries the
parts that have hard real-time deadlines. Every simulation runs in two stages with
selectable methods: deterministic ray tracing (or image sources) for the early
reflections and stochastic processing for the late reverberation. Both write a
format-independent *directional reflectogram*, from which binaural, Ambisonics and
loudspeaker impulse responses are encoded, so one simulation serves every output
format.

## Install

Every release has wheels for Linux, macOS and Windows (CPython 3.10 to 3.14), with the
compiled core inside, on its
[GitHub release page](https://github.com/naelolaiz/acoustic_spaces_simulator/releases):
`pip install` the wheel for your platform, with the extras below in brackets if you
want them. Until the package is published on PyPI, everything else installs from source.

The documentation (this README, the validation report and an API reference generated
from the docstrings) builds with Sphinx; see CONTRIBUTING.md.

## Install from source

Requirements: Python 3.10+, a C++17 compiler and CMake 3.18+.

```sh
git clone https://github.com/naelolaiz/acoustic_spaces_simulator
cd acoustic_spaces_simulator
```

With [uv](https://docs.astral.sh/uv/), `uv sync` builds the package into `.venv` together
with the test tools (pytest, ruff), and `--extra` adds the optional parts:

```sh
uv sync --extra viz                    # viz: the 3D view (PyVista) and plots (matplotlib)
uv sync --extra viz --extra measure    # measure: sounddevice, for sweeps and playback
uv sync --extra gui                    # gui: the graphical editor (Qt, PySide6)
uv sync --all-extras                   # everything, with sofa (h5py) and docs
```

`uv sync` installs exactly what it is asked for, so name every extra each time. It
rebuilds the compiled core when the C++ sources change.

With pip, in a virtual environment:

```sh
python -m pip install -e ".[dev,viz]"     # viz adds the 3D view (PyVista)
python -m pip install -e ".[measure]"     # sounddevice: sweeps, real-time playback
python -m pip install -e ".[sofa]"        # h5py, to read HRTFs from SOFA files
python -m pip install -e ".[gui]"         # the graphical editor (Qt, PySide6)
```

Check the installation and look at the example scene. The commands run in the environment
you installed into: activate it (`source .venv/bin/activate`, or `.venv\Scripts\activate`
on Windows), or with uv put `uv run` in front of each.

```sh
acoustic-spaces info
acoustic-spaces validate examples/shoebox/scene.yaml   # errors exit with status 1
acoustic-spaces inspect examples/shoebox/scene.yaml    # geometry, materials, RT estimates
acoustic-spaces view examples/shoebox/scene.yaml       # 3D window; --screenshot out.png
acoustic-spaces view examples/shoebox/scene.yaml --paths --max-order 2  # reflection paths
acoustic-spaces animate examples/lecture_hall/scene.yaml               # sound particles
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb --paths --particles  # Blender
acoustic-spaces materials carpet                       # search the material database
acoustic-spaces directivity hypercardioid --plot polar.png  # a source directivity
acoustic-spaces simulate examples/shoebox/scene.yaml --out results --plot
python -m pytest                                       # the test suite
```

`animate` shows sound particles leaving a source and spreading through the room, each
reflecting off the walls (specularly, or diffusely with the surface's scattering
coefficient) and losing the absorbed share of its energy at every hit, coloured by
level or by reflection order. It opens a window with a time slider, or saves a GIF
(`--out particles.gif`) or, with `imageio[ffmpeg]` installed, an MP4 video.

`simulate` writes impulse responses for every source and listener in each format the
scene's `outputs` list (`results/talker__seat-a.wav` in mono, `.binaural.wav`,
`.ambix3.wav`, `.5.0.wav` and so on, 32-bit float), `parameters.json` with every
ISO 3382-1 parameter per octave band, and with `--plot` an echogram, decay curves and a
map of arrival directions as PNG. `--output ambisonics:3` (repeatable) picks formats on
the command line. It prints the parameters as it goes:

```
talker -> seat-a: direct sound at 11.7 ms, 63 early paths
             63 Hz  125 Hz  250 Hz  500 Hz   1 kHz   2 kHz   4 kHz   8 kHz
EDT (s)      1.811   1.797   1.703   1.419   1.306   1.200   0.960   0.532
T30 (s)      2.953   2.918   2.840   2.369   2.280   2.101   1.648   0.849
C80 (dB)       1.3     1.3     1.9     3.0     3.9     4.7     6.0     9.4
G (dB)        22.8    22.8    22.1    21.0    19.3    18.1    16.9    15.5
...
```

T30 comes out several times the Sabine and Eyring estimates that `inspect` prints, and
that is the room, not a fault. Its absorption sits on the ceiling and the carpet while
the walls are hard and smooth (scattering 0.05), so sound running between the walls meets
little absorption and dies away slowly. Sabine and Eyring assume a diffuse field, where
sound meets every surface alike; `inspect` says when a room is far from that and adds the
Arau-Puchades estimate, which allows for absorption that faces one way. With every surface
fully diffusing, the same room decays in Eyring's 0.37 s at 1 kHz; with every surface
scattering 0.3, in about 0.6 s. The
[validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md)
checks both kinds of room.

From Python:

```python
from acoustic_spaces import Room, simulate
from acoustic_spaces.audio import write_wav
from acoustic_spaces.engine import early_paths, trace_particles
from acoustic_spaces.spatial import Ambisonics, Binaural, Loudspeakers
from acoustic_spaces.viz import animate_particles, plot_directions, plot_response, show_room

room = Room.from_yaml("examples/shoebox/scene.yaml")
print(room.report.summary())
print(room.problems(), room.warnings())
print(room.reverberation_time("eyring"))  # per octave band, seconds
show_room(room, paths=early_paths(room, max_order=2))
animate_particles(room, trace_particles(room, duration=0.1), "particles.gif")

result = simulate(room)
response = result["talker", "seat-a"]
print(response.parameters().table())
write_wav("talker__seat-a.wav", response.impulse_response(48_000), 48_000)
plot_response(response, "talker__seat-a.png")

binaural = response.spatial_impulse_response(Binaural(), 48_000)  # (samples, 2)
ambix = response.spatial_impulse_response(Ambisonics(order=3), 48_000)  # 16 channels
surround = response.spatial_impulse_response(Loudspeakers.layout("5.0"), 48_000)
plot_directions(response, "talker__seat-a.directions.png")
```

## Graphical editor

The editor needs the `gui` extra: `uv sync --extra gui` in a source checkout, then
`uv run acoustic-spaces edit`, or `pip install "acoustic-spaces[gui]"`.

```sh
acoustic-spaces edit                              # a new box room
acoustic-spaces edit examples/studio/scene.yaml   # or open a scene file
```

![The editor with the concert hall example: the scene tree, the room with its early reflection paths, the source's properties and the echogram of a run](docs/images/editor.png)

The editor works on the same scene files as the command line:

- **File > Open example** opens the bundled rooms. Saving one asks where to keep your
  copy and takes its model file along. **New box room** and **New from model file**
  (OBJ, STL, PLY, OFF, glTF or DXF) start from scratch.
- The **scene tree** lists the geometry and its surfaces, the materials, sources,
  listeners, simulation settings and output formats. Selecting an entry, there or by
  clicking it in the 3D view, shows its form under **Properties**. Every change is
  checked against the scene schema; a refused change says why under **Messages**, and
  every change can be undone.
- **Place source** and **Place listener**, then click a surface: on a floor the new
  point stands at ear or mouth height, on a wall or ceiling it goes half a metre into
  the room. Surfaces take materials from their form, or from a new material picked from
  the database (**Edit > Add material**).
- **Simulation > Run** (F5) works in the background. The **Results** panel shows the
  ISO 3382-1 parameters, the echogram and decay curves and the arrival directions of
  any source and listener, and draws their reflection paths in the room. **Export**
  writes the same files as `simulate`; **Listen** plays a dry WAV file through the
  response on the sound card (binaural when the scene has a binaural output; needs the
  `measure` extra).
- A source's form sets its directivity: a pattern, or a SOFA or CSV file of measured
  data. It shows the directivity index per band, opens the polar patterns, and the
  selected source shows its balloon in the 3D view.
- The status bar keeps the volume and the Eyring reverberation time at 1 kHz up to date
  as you edit.
- **File > Export to Blender** writes the room with the reflection paths on show, and
  **File > Export particle animation to Blender** the sound particles leaving the
  selected source (see below).

### Exporting to Blender

`acoustic-spaces export` writes a scene as glTF 2.0 (`.glb`) for Blender or another 3D
tool; the editor does the same from its File menu.

```sh
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb              # the room
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb --paths      # + reflection paths
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb --particles  # + particle animation
```

In Blender, use **File > Import > glTF 2.0 (.glb/.gltf)** and pick the file. The axes are
the scene's (+z up, metres).

- Each surface is an object named as in the editor, with a see-through material per
  scene material; the material's custom properties hold its absorption and scattering
  per octave band. Sources (red) and listeners (blue) are spheres turned the way they
  face, with an arrow along the aim.
- `--paths` adds the early reflection paths from `--source` to `--listener` up to
  `--max-order`, one object of edges per order. Edges do not render: to render them,
  use Object > Convert > Curve and give the curve a bevel depth.
- `--particles` adds the particles of `animate` as small spheres that move with them and
  shrink with their level, gone 60 dB down. The animation runs 40 times slower than the
  sound (`--slow-motion`): the default 100 ms take 4 s, 96 frames at 24 fps. If the
  timeline ends sooner, set its end frame, then press play. With the default 1000
  particles (`--count`) the file is about 3 MB.

### Example scenes

| Example | What it is |
| --- | --- |
| `shoebox` | An 8 × 6 × 3 m room from an OBJ file, the simplest case every method must get right. |
| `lecture_hall` | A fan-shaped lecture hall built from its floor plan, no CAD file needed. |
| `concert_hall` | A 32 × 20 × 15 m shoebox hall with a raised stage and upholstered seats; two sources on stage, three seats in the stalls. |
| `studio` | A 6 × 4.5 × 3 m control room with side panels, a diffuser and stereo loudspeakers at ±30°. |
| `church` | A stone nave with an apse, 18 m high, with about six seconds of reverberation. |

`acoustic-spaces examples` lists them with where they are installed, and
`acoustic-spaces examples --copy my-rooms` copies them to a folder you can edit. From
Python, `acoustic_spaces.examples.example_path("church")` gives a scene file.

## Scene files

A scene is a YAML file that names the geometry, the materials, which surface gets
which material, and where the sources and listeners are. See
[`examples/shoebox/scene.yaml`](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/examples/shoebox/scene.yaml) and the schema in
[`src/acoustic_spaces/scene/model.py`](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/src/acoustic_spaces/scene/model.py).
Frequency-dependent data uses eight octave bands from 63 Hz to 8 kHz.

### Geometry

| Format | Read with | Names used for material assignment |
| --- | --- | --- |
| OBJ | built-in reader | `usemtl` material, `g` group, `o` object |
| DXF | [ezdxf](https://ezdxf.mozman.de/) (3DFACE, MESH, polyface, block references) | layer |
| glTF / GLB | [trimesh](https://trimesh.org/) | material, node and mesh names |
| STL, PLY, OFF | trimesh | file name |

Simple rooms need no CAD file: `geometry: {box: [8, 6, 3]}` builds a rectangular
room from the origin, and a floor plan is raised to a height (see
[`examples/lecture_hall/scene.yaml`](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/examples/lecture_hall/scene.yaml)):

```yaml
geometry:
  floor_plan: [[0, -4], [0, 4], [14, 8], [14, -8]]  # corners, in order, any shape
  height: 6
assignments:
  floor: audience
  ceiling: plaster
  wall_1: wood       # the edge from the first corner to the second
  walls: concrete    # every other wall
```

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

### Sources and directivity

A source has a position, an orientation (`yaw`, `pitch` and `roll` in degrees; all zero
faces +x with +z up), a sound power level (`level_db`, dB re 1 pW) and a
`directivity`:

```yaml
sources:
  - name: talker
    position: [2.0, 3.0, 1.5]
    orientation: {yaw: 30}
    directivity: cardioid   # or omni, subcardioid, supercardioid, hypercardioid, figure8
  - name: monitor
    position: [0.7, 1.3, 1.2]
    directivity:            # a pattern, or the a of a + (1 - a) cos(theta), per octave band
      pattern: [omni, omni, omni, subcardioid, subcardioid, cardioid, cardioid, 0.3]
  - name: loudspeaker
    position: [0.7, 3.3, 1.2]
    directivity: {file: speaker.sofa}   # measured data: SOFA or CSV
```

- A directivity says where the sound power goes, not how much: its weights average 1
  over the sphere in every band. It shapes the direct sound, each early path (by the
  direction it leaves the source in), the rays of the late part and the particles of
  `animate`.
- **SOFA** files follow the AES69 `FreeFieldDirectivityTF` convention (spectra measured
  by microphones around the source, as published for loudspeakers and instruments), or
  hold impulse responses in `Data.IR`. Directions are the microphones' positions as the
  source sees them (`SourceView` and `SourceUp`). `measurement: 3` picks one
  measurement, for example one note of an instrument; by default all are averaged.
  Reading SOFA files needs the `sofa` extra (h5py).
- A **CSV** table has the header `azimuth, elevation` followed by frequencies in Hz,
  then one row per direction with levels in dB (any reference; only differences
  matter). Azimuth turns from the front towards the source's left and elevation is
  positive upwards, in degrees. Any set of directions works: they are interpolated
  onto a 5 degree grid. Bands are averaged into octaves, and octaves with no data copy
  the nearest one.
- CLF (Common Loudspeaker Format) files are not read: their specification is not
  public. Convert them to SOFA or CSV.

`acoustic-spaces directivity speaker.sofa --plot polar.png` prints the directivity
index and front-to-back ratio per octave band and draws the polar patterns;
`acoustic-spaces inspect` lists every source's directivity index.

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

  In rooms with absorption in one place and smooth walls, most late energy rides on
  the few rays that keep clear of the absorber, and the tail breaks up where those
  run out. With `splitting` (on by default), a first tenth of the rays measures how
  much energy a ray usually carries over time; the rest are split into copies when
  they carry more than 4 times that and thinned out by Russian roulette when they
  carry less than a quarter (a weight window). The expected result is unchanged; in
  the example box the late tail (0.5 to 2.5 s) misses a 1-million-ray reference by
  0.5 to 2 dB per band instead of 1 to 8 dB at the default 100k rays (1 to 4 dB
  instead of 4 to 17 dB at 10k), for about 1.5 times the run time.
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
`acoustic_spaces.parameters.from_impulse_response` does the same for any mono IR (see
[Measuring impulse responses](#measuring-impulse-responses)). A response that ends
before it has decayed is extended by its fitted decay rate.
Air absorption follows ISO 9613-1 for the scene's temperature, humidity and pressure.

### Validation

The tests check the simulation against theory on Linux, macOS and Windows: exact image
sources in a box, Eyring's reverberation time and Barron's revised theory in a diffusing
box, ISO 3382-1 parameters of known decays, spatial encoders against closed forms, the
convolvers against direct convolution, and sweep measurements of synthetic rooms. A
cross-check against pyroomacoustics in four rooms agrees within 6 % on T30 where both
programs model the same physics, and shows where their scattering models part ways.
The [validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md)
has the details, the numbers and the scripts to reproduce them.

## Spatial outputs

The scene's `outputs` list says which formats `simulate` writes (mono by default):

```yaml
outputs:
  - mono
  - binaural                              # spherical head model
  - {format: binaural, hrtf: kemar.sofa}  # measured HRTFs
  - {format: ambisonics, order: 3}        # AmbiX: ACN order, SN3D, orders 0 to 7
  - {format: loudspeakers, layout: "5.1"} # stereo, 5.0, 7.0, 5.0.4, 7.0.4, and .1 with LFE
  - format: loudspeakers                  # or any layout, in channel order
    speakers: [{name: L, azimuth: 30}, {name: R, azimuth: -30}, {name: Top, azimuth: 0, elevation: 60}]
```

Directions are those of the listener's head as its `orientation` turns it: azimuth
counts to the left, elevation up, as in SOFA and AmbiX. Every early reflection keeps
its exact arrival direction. For the late part the stochastic tracer sorts arriving
energy into 64 directions (`simulation.late.directions`) every 10 ms; each impulse of
the synthesized tail then comes from one of them, drawn in proportion to that energy.
The statistical tail arrives evenly from all around.

- **Ambisonics** channels are the real spherical harmonics with SN3D normalisation, so
  W is the mono response exactly.
- **Loudspeakers** get vector base amplitude panning (VBAP, unit power). Layouts with
  height but nothing below the listener get an imaginary speaker straight down whose
  share is dropped. Named layouts carry their WAV channel mask; the LFE channel of the
  `.1` layouts stays silent, leaving bass management to the playback system.
- **Binaural** uses the HRTF of the nearest measured direction for every reflection and
  for each late direction. Without a SOFA file it uses Brown and Duda's spherical head
  model: interaural time and level differences, but no pinna cues. `read_sofa` reads
  SimpleFreeFieldHRIR files, which most HRTF databases (for example those listed by
  [sofaconventions.org](https://www.sofaconventions.org)) provide; it needs h5py.

Levels match the mono response: an arrival from straight ahead has the same height in
W, in the speaker it is panned to, and, below the head's influence, at both ears.

## Listening: convolution and real-time playback

Run a dry (anechoic) recording through a simulated or measured response:

```sh
acoustic-spaces convolve speech.wav results/talker__seat-a.binaural.wav --out wet.wav
acoustic-spaces auralize results talker=speech.wav piano=piano.wav --format 5.0 --out mix.wav
acoustic-spaces play results/talker__seat-a.binaural.wav speech.wav   # real time
acoustic-spaces play results/talker__seat-a.binaural.wav --live      # microphone in
```

`convolve` takes any response WAV, with any number of channels; `auralize` reads a
`simulate` folder and mixes every source's own signal as heard at one `--listener`, in
one of the formats it wrote (`mono`, `binaural`, `ambix3`, `5.0`, ...), keeping the
levels the simulation gives each source. Dry signals are mixed down to mono, as
sources are points, and resampled to the response's rate. The results are 32-bit float
WAV files; `--normalize` scales them to a -1 dBFS peak.

`play` convolves in real time on the sound card (it needs the `measure` extra). The
compiled convolver adds no latency beyond one `--block` of frames (256 by default, 5.3
ms at 48 kHz), and nothing allocates memory while it runs. It cuts the response into
partitions that grow along it, from the block size up to 8192 frames, and spreads the
work of the long ones evenly over the blocks in between (non-uniform partitioning), so
long and multichannel responses stay cheap: a 3 s, 16-channel third-order Ambisonics
response in blocks of 128 frames takes about 0.3 ms of the 2.7 ms each block lasts, on
one current x86-64 core, where uniform partitions took 3.6 ms. By default the response
is scaled to unit energy so the result plays about as loud as the dry signal;
`--absolute` keeps the simulated level.

From Python:

```python
from acoustic_spaces.render import Convolver, auralize, convolve, play, resample
from acoustic_spaces.spatial import Binaural

mix = auralize(result, {"talker": (speech, 44_100)}, "seat-a", Binaural())  # (samples, 2)
wet = convolve(speech_48k, binaural)  # offline, float64
convolver = Convolver(binaural, block=256)  # block by block, float32
out = convolver.process(next_block)  # (256, 2)
play(binaural, 48_000, speech_48k)  # through the sound card
```

## Measuring impulse responses

The same parameters can be measured in a real room with an exponential sine sweep,
a loudspeaker, a measurement microphone and an audio interface:

```sh
acoustic-spaces devices                                # pick the interface
acoustic-spaces measure --out hall --device 3 --input 1 2 --loopback 4 \
    --duration 10 --silence 4 --repeats 3 --plot
```

`measure` plays the sweep on `--output` (1 by default), records every `--input`,
deconvolves the recording and prints the parameters of each microphone. It writes
`sweep.wav` and its settings, the raw `recording.wav`, the impulse responses in
`ir.wav` (one channel per input), `measurement.json` and, with `--plot`, a level and
decay plot per input. A `--loopback` input wired straight to the output removes the
latency of the interface, so arrival times are true times of flight.

Any other player and recorder works too: write the excitation, play it while
recording, then deconvolve and analyse:

```sh
acoustic-spaces sweep sweep.wav --duration 10 --silence 4 --repeats 3
# ... play sweep.wav and record recording.wav, starting together ...
acoustic-spaces deconvolve recording.wav --sweep sweep.json --out ir.wav
acoustic-spaces analyze ir.wav --plot ir.png --json ir.json
```

How it works:

- The sweep is synchronized (Novak et al., 2015), so harmonic distortion of the
  loudspeaker lands before the linear response and is cut away. Make the silence
  after each sweep longer than the reverberation time; repeats are averaged, each
  doubling lowering the background noise by 3 dB.
- Deconvolution divides spectra, regularised outside the swept band.
- `analyze` filters with IEC 61260-1 octave filters (Butterworth, order 3, zero
  phase), finds where each band sinks into the background noise (Lundeby et al.,
  1995), cuts it there, subtracts the noise and adds the missing tail (ISO 3382-1,
  5.3.4). Decay times are left out when a band's decay does not clear the noise by
  enough: 20 dB for EDT, 35 dB for T20, 45 dB for T30.
- G needs the source's free-field level: `--free-field ff.wav --distance 3` uses a
  measurement of the same source in free field (or with reflections windowed out),
  and `--reference-energy 0.01` suits responses made by `simulate`.

From Python:

```python
from acoustic_spaces.measurement import Sweep, deconvolve, measure
from acoustic_spaces.parameters import from_impulse_response

sweep = Sweep(duration_s=10, silence_s=4, repeats=3)
result = measure(sweep, inputs=[1, 2], loopback=4, device=3)
for ir in result.impulse_responses.T:
    print(from_impulse_response(ir, sweep.sample_rate).table())
```

## Repository layout

```
src/acoustic_spaces/   Python package
  scene/               scene model and YAML schema
  geometry/            CAD import, triangulation, mesh cleanup and checks
  materials/           material database and coefficient files
  directivity.py       source directivity: patterns per band, SOFA and CSV balloons
  room.py              a scene with its geometry loaded: materials per face, checks
  engine/              early reflections, late reverberation, responses, simulate()
  air.py               ISO 9613-1 air absorption and speed of sound
  synthesis.py         octave filter bank and impulse response synthesis
  spatial/             Ambisonics, VBAP loudspeaker layouts, HRTFs, spatial rendering
  parameters.py        ISO 3382-1 parameters, octave filters, noise floor detection
  render.py            convolution, resampling, auralisation, real-time playback
  measurement.py       sine sweeps, playback and recording, deconvolution
  audio.py             WAV files
  viz.py               3D view, reflection paths, particle animations, response plots
  gltf.py              export to glTF (.glb) for Blender: room, paths, particle animation
  results.py           writing a simulation's responses and parameters to a folder
  examples.py          the example scenes that ship with the package
  gui/                 the graphical editor (Qt)
  cli.py               command-line interface
cpp/include/           header-only C++ core: BVH, ray tracer, FFT, partitioned
                       convolver (no Python, so a plugin or player can reuse it)
cpp/bindings/          nanobind module exposing the core as acoustic_spaces._native
tests/                 pytest suite, run on Linux, macOS and Windows in CI
examples/              example scenes, installed with the package
```

## Roadmap

v1, version 1.0.0:

- [x] M0 Foundations: package layout, compiled core, scene schema, CI.
- [x] M1 Scene and geometry: CAD import, material database, 3D view.
- [x] M2 Mono simulation: ray-traced early reflections, stochastic tail, IR synthesis, ISO 3382 parameters.
- [x] M2b IR measurement: exponential sine sweeps, playback and recording, deconvolution, parameters of measured IRs.
- [x] M3 Spatial outputs: binaural (SOFA HRTFs or a spherical head), Ambisonics (AmbiX) up to 7th order, VBAP loudspeaker layouts.
- [x] M4 Rendering: offline convolution and auralisation in Python, real-time partitioned convolver in C++, playback.
- [x] M5 v1 release: validation report, documentation, wheels and release workflow.

v2:

- [x] A graphical editor: scenes, materials, sources and listeners, runs, results and listening.
- [x] Export to Blender: rooms, reflection paths and particle animations as glTF.
- [x] Source directivity per octave band: patterns, and measured data from SOFA or CSV files.
- [ ] CLF loudspeaker files, once their specification is available.
- [ ] Editing geometry in the editor: drawing rooms and moving points by dragging.
- [ ] Head tracking.
- [ ] Diffraction.
- [ ] A wave-based solver for low frequencies.
- [ ] Coverage maps.

## Contributing

Pull requests are welcome. Run `ruff check . && ruff format --check . && pytest`
before opening one; CI runs the same checks on all three platforms.
[CONTRIBUTING.md](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/CONTRIBUTING.md)
covers the development setup, the documentation, the validation scripts and releases.

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/LICENSE).

The built-in material table comes from
[pyroomacoustics](https://github.com/LCAV/pyroomacoustics) (MIT, see
[`materials/data/LICENSE-pyroomacoustics.txt`](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/src/acoustic_spaces/materials/data/LICENSE-pyroomacoustics.txt)).
That table does not cite a source for most entries; check values that matter
against manufacturer data or a measured database.
