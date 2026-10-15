# acoustic_spaces

Open-source room acoustics simulator and auralisation toolkit, in the spirit of
CATT-Acoustic and ODEON: load a 3D model of a space, place sources and listeners,
compute impulse responses, and listen to the result in binaural, Ambisonics or
loudspeaker formats.

**Status: 2.0.** Rooms come from CAD models, from their dimensions or from a floor plan
drawn in the editor, with a material database and directional sources. Simulations
combine ray-traced early reflections, edge diffraction, a stochastic late tail and,
for the lowest bands, a wave solver, and give impulse responses in mono, binaural,
Ambisonics (AmbiX) and loudspeaker formats as WAV files, ISO 3382-1 parameters, and
coverage maps of them and of the speech transmission index over the audience. Dry
recordings can be run through the responses offline or in real time on the sound card,
head-tracked on headphones; real rooms can be measured with sine sweeps and compared
with their model, whose absorption can then be fitted to the measurements. Reflection paths and sound particles can be watched in 3D, animated,
and exported to Blender. The
[validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md)
says how far the results can be trusted. A [graphical editor](#graphical-editor) does
all of this in one window, starting from the bundled example rooms. To hear it,
[listen to Beethoven's 8th symphony](https://www.youtube.com/watch?v=c0-5F-iNhIE) played by 61
anechoic tracks on the stage of the concert hall example, on headphones ([how it was
made](#an-orchestra-in-the-concert-hall)).

![The church example in 3D: the stone nave with its early reflection paths from the altar to the front of the nave, coloured by order up to the third](docs/images/church_paths.png)

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
`pip install` the wheel for your platform, with the extras below in brackets after its
file name if you want them, such as
`pip install "acoustic_spaces-2.0.0-cp312-cp312-win_amd64.whl[viz,gui]"`. The package is
not on PyPI, so `pip install acoustic-spaces` does not find it; between releases, install
from source.

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
uv sync --extra gpu                    # gpu: the wave solver on a GPU (wgpu, WebGPU)
uv sync --all-extras                   # everything, with sofa (h5py), sketchup and docs
```

`uv sync` installs exactly what it is asked for, so name every extra each time. It
rebuilds the compiled core when the C++ sources change.

With pip, in a virtual environment:

```sh
python -m pip install -e ".[dev,viz]"     # viz adds the 3D view (PyVista)
python -m pip install -e ".[measure]"     # sounddevice: sweeps, real-time playback
python -m pip install -e ".[sofa]"        # h5py, to read HRTFs from SOFA files
python -m pip install -e ".[sketchup]"    # openskp, to read SketchUp files (experimental)
python -m pip install -e ".[gui]"         # the graphical editor (Qt, PySide6)
python -m pip install -e ".[gpu]"         # wgpu: the wave solver on a GPU (WebGPU)
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
acoustic-spaces view examples/gallery/scene.yaml --paths --listener "round the corner"  # diffraction
acoustic-spaces animate examples/lecture_hall/scene.yaml               # sound particles
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb --paths --particles  # Blender
acoustic-spaces materials carpet                       # search the material database
acoustic-spaces directivity hypercardioid --plot polar.png  # a source directivity
acoustic-spaces simulate examples/shoebox/scene.yaml --out results --plot
acoustic-spaces map examples/lecture_hall/scene.yaml --plot sti.png --parameter sti
python -m pytest                                       # the test suite
```

`animate` shows sound particles leaving a source and spreading through the room, each
reflecting off the walls (turned from the specular direction by as much as the surface
scatters; see [Scattering](#scattering)) and losing the absorbed share of its energy at
every hit, coloured by level or by reflection order. It opens a window with a time
slider, or saves a GIF (`--out particles.gif`) or, with `imageio[ffmpeg]` installed, an
MP4 video.

![Sound particles leaving the soloist of the concert hall example, coloured by level at 1 kHz, over the first 90 ms](docs/images/particles.gif)

`simulate` writes impulse responses for every source and listener in each format the
scene's `outputs` list (`results/talker__seat-a.wav` in mono, `.binaural.wav`,
`.ambix3.wav`, `.5.0.wav` and so on, 32-bit float), `parameters.json` with the room
acoustic parameters of every response per octave band, its STI and its echo criteria,
and with `--plot` an echogram, decay curves and a map of arrival directions as PNG.
`--output ambisonics:3` (repeatable) picks formats on the command line. It prints the
parameters as it goes:

```
talker -> seat-a: direct sound at 11.7 ms, 63 early paths
             63 Hz  125 Hz  250 Hz  500 Hz   1 kHz   2 kHz   4 kHz   8 kHz
EDT (s)      0.720   0.722   0.710   0.746   0.907   1.010   0.964   0.585
T30 (s)      0.715   0.895   0.984   1.224   1.267   1.260   1.130   0.774
C80 (dB)       5.4     5.5     6.1     6.3     5.9     5.6     5.8     8.9
G (dB)        21.8    21.7    20.8    20.0    18.7    17.9    17.1    15.7
LF           0.307   0.307   0.298   0.309   0.334   0.350   0.352   0.335
IACC_E       0.992   0.976   0.880   0.600   0.303   0.687   0.578   0.431
T30_X (s)    0.713   0.799   1.017   1.195   1.243   1.210   1.122   0.756
T30_Z (s)    0.714   0.818   0.925   1.117   1.226   1.215   1.158   0.769
...
STI 0.65 (STIPA 0.66), echo criterion 0.67 for speech (echo above 1.0), 0.67 for music (echo above 1.8)
```

With `--plot`, each response gets an echogram (the late energy per millisecond and the
early paths by order) with the decay curves of every band, and a map of where the sound
comes from:

![Echogram and decay curves of a seat in the middle of the concert hall's stalls](docs/images/response.png)

![Arrival directions at the same seat: early reflections by delay and level, and the late energy per direction](docs/images/directions.png)

From 500 Hz up, T30 comes out two to four and a half times the Sabine and Eyring
estimates that `inspect` prints, and that is the room, not a fault. Its absorption sits
on the ceiling and the carpet while the walls are hard and smooth (scattering 0.05), so
sound running between the walls meets little absorption and dies away slowly. Sabine and
Eyring assume a diffuse field, where sound meets every surface alike; `inspect` says
when a room is far from that and adds the Arau-Puchades estimate, which allows for
absorption that faces one way. With every surface fully diffusing, the same room decays
in Eyring's 0.37 s at 1 kHz; with every surface scattering 0.3, in about 0.55 s. The
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
`uv run acoustic-spaces edit`, or a release wheel installed with `[gui]` after its file
name.

```sh
acoustic-spaces edit                              # a new box room
acoustic-spaces edit examples/studio/scene.yaml   # or open a scene file
```

![The editor with the concert hall example: the scene tree, the room with its early reflection paths, the source's properties and the echogram of a run](docs/images/editor.png)

The editor works on the same scene files as the command line:

- **File > Open example** opens the bundled rooms. Saving one asks where to keep your
  copy and takes its model file along. **New box room**, **New room from a floor plan**
  and **New from model file** (OBJ, STL, PLY, OFF, glTF, DXF or SketchUp) start from
  scratch.
- The **Plan** tab, next to the 3D view, shows the room from above and draws rooms (see
  below).
- The **scene tree** lists the geometry and its surfaces, the materials, sources,
  listeners, simulation settings and output formats. Selecting an entry, there or by
  clicking it in the 3D view, shows its form under **Properties**. Every change is
  checked against the scene schema; a refused change says why under **Messages**, and
  every change can be undone.
- **Place source** and **Place listener**, then click a surface: on a floor the new
  point stands at ear or mouth height, on a wall or ceiling it goes half a metre into
  the room. In the plan, the click places it above the floor there. Sources and
  listeners move by dragging, in the 3D view or the plan, and keep their height above
  the floor (up a raked floor too); with Shift in the 3D view they move up and down.
  Surfaces take materials from their form, or from a new material picked from the
  database (**Edit > Add material**).
- **Simulation > Run coverage map** (F6) maps every source over the audience area (see
  [Coverage maps](#coverage-maps)). The **Coverage map** panel shows any parameter
  and band from above, for one source or all of them together, and **Map in 3D** lays
  it over the room; **Save** writes the picture or every map as JSON. The scene tree's
  **Coverage map** entry sets the surfaces, height, spacing, rays and background noise.

  ![The editor with the lecture hall example and its speech transmission index map laid over the room](docs/images/editor_map.png)

- **Simulation > Run** (F5) works in the background. The **Results** panel shows the
  ISO 3382-1 parameters, the echogram and decay curves and the arrival directions of
  any source and listener, and draws their reflection paths in the room. **Export**
  writes the same files as `simulate`; **Listen** plays a dry WAV file through the
  response on the sound card (binaural when the scene has a binaural output; needs the
  `measure` extra), or head-tracked with **Head tracking** ticked (see
  [Head tracking](#head-tracking)).
- Change a material's absorption after a run and the results follow at once: the run
  counted each ray's reflections off each material, and is re-weighted for the new
  values instead of traced again (as `calibrate` does). The panel says so; **Run**
  traces afresh. Other changes, and runs with the statistical method, third-octave
  bands or the wave solver, leave the results marked out of date until the next run.
- A source's form sets its directivity: a pattern, or a SOFA or CSV file of measured
  data. It shows the directivity index per band, opens the polar patterns, and the
  selected source shows its balloon in the 3D view.
- The status bar keeps the volume and the Eyring reverberation time at 1 kHz up to date
  as you edit.
- **File > Export to Blender** writes the room with the reflection paths on show, and
  **File > Export particle animation to Blender** the sound particles leaving the
  selected source (see below).

### Drawing rooms

![The plan of the gallery example while a corner is dragged, with the lengths of the two walls it moves](docs/images/plan.png)

The **Plan** tab shows the room from above on a grid in metres. In a floor plan or box
room every corner is a handle:

- Drag a corner to move it; the two walls it joins show their lengths as it moves. A
  box becomes a floor plan with the same walls, so its materials stay where they were.
- Double-click a wall to add a corner there, or right-click a corner to remove it.
  Clicking a wall selects that surface, to give it a material.
- **Draw room** (or **Edit > Draw floor plan**) draws a new floor plan: click its corners
  in turn and close it on the first one, with a double-click or with Enter. Backspace
  takes the last corner back and Esc stops. The new room keeps the old one's height
  (change it under **Geometry**). Sources and listeners that would end up outside move
  inside, keeping their arrangement. Over a model file, the editor asks before
  replacing it.
- Corners and points snap to the grid chosen under **Snap** (10 cm unless changed);
  hold Shift to place them freely. A move that would make walls cross is refused, with
  the reason under **Messages**.
- The wheel zooms, dragging empty space pans and **Fit** shows the whole room again.

Rooms from model files show their outline, and their sources and listeners move the
same way. Every change can be undone.

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
| `gallery` | An L-shaped gallery: one listener sees the guide, the other stands round the corner and hears the guide by [edge diffraction](#edge-diffraction). |
| `archway` | A bar and a dining room joined by an archway in a wall 40 cm thick: a diner behind the wall hears the singer over both corners of the arch's jamb (`second_order` diffraction). |
| `orchestra` | The concert hall with a full orchestra on its stage, one source for each of the 61 tracks of an anechoic recording, and the scripts that [mix them and make a video](#an-orchestra-in-the-concert-hall). |

`acoustic-spaces examples` lists them with where they are installed, and
`acoustic-spaces examples --copy my-rooms` copies them to a folder you can edit. From
Python, `acoustic_spaces.examples.example_path("church")` gives a scene file.

## Scene files

A scene is a YAML file that names the geometry, the materials, which surface gets
which material, and where the sources and listeners are. See
[`examples/shoebox/scene.yaml`](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/examples/shoebox/scene.yaml) and the schema in
[`src/acoustic_spaces/scene/model.py`](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/src/acoustic_spaces/scene/model.py).
Frequency-dependent data uses eight octave bands from 63 Hz to 8 kHz, or 24
third-octave bands from 50 Hz to 10 kHz (see [Frequency bands](#frequency-bands)).

### Geometry

| Format | Read with | Names used for material assignment |
| --- | --- | --- |
| OBJ | built-in reader | `usemtl` material, `g` groups, `o` object |
| DXF | [ezdxf](https://ezdxf.mozman.de/) (3DFACE, MESH, polyface, block references) | layer |
| glTF / GLB | [trimesh](https://trimesh.org/) | material, node and mesh names |
| STL, PLY, OFF | trimesh | file name |
| SketchUp (`.skp`), experimental | [openskp](https://github.com/iamahsanmehmood/openskp), the `sketchup` extra | material |

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
Without `units`, a DXF file is read in the units its header records (`$INSUNITS`),
and any other file in metres.

SketchUp files can also be read as they are, with the `sketchup` extra: openskp reads
them without SketchUp, in metres. That reader is young. Faces are matched to materials
by colour, so give each material its own colour; a face painted differently on its two
sides takes its front's material; and some files do not read at all (the BRAS
auditorium, for one). An OBJ export from SketchUp is then the way.

Polygons are triangulated, duplicate vertices merged and degenerate faces dropped,
and faces that meet along an edge are turned to agree with each other (each part of
the model keeps the orientation most of its area has), so a few faces flipped by the
exporter do not spoil the volume or the inside checks. Where a face's edge runs past
the corners of faces on its other side (a T-junction, which SketchUp stitches with
faces of zero area), the edge is split at those corners, so the model stays closed.
A model with no faces left is refused. `validate` then reports surfaces without a
material, sources or listeners outside the room and listeners within 1 cm of a source;
it warns about models that are not closed (rays would escape) and assignments that
match nothing. Normals may point either way: the room is the space the faces close off,
found from the faces themselves (see [Walls, doors and openings](#walls-doors-and-openings)),
and a panel standing in the room, with air on both sides, reflects and absorbs on both.

### Materials

Each entry in `materials` takes coefficients from one of:

- `preset: <id>`: the built-in database (`acoustic-spaces materials` lists it);
- `file: <csv>`: a table of centre frequencies, absorption and optionally
  scattering, one row each, at any band resolution (the layout the
  [BRAS](https://doi.org/10.14279/depositonce-6726.3) benchmark uses);
- `absorption: [...]` with eight octave values or 24 third-octave values, which also
  overrides a preset or file.

`scattering` is a number, eight or 24 values, or the id of a built-in scattering entry.
File data with more than one value in some octave (third-octave tables) stays in
third-octave bands; other data is averaged onto octaves. Bands with no data (63 Hz in
most tables) take the nearest band's value and the material's `source` says which.
Surfaces get a material through `assignments` (any geometry name to a material),
by having the same name as a material, or through `default_material`.

Two rooms of the BRAS benchmark, converted from its SketchUp models, are in
[`validation/bras`](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/validation/bras)
as scenes with their surface data and measurements.

### Walls, doors and openings

A model can hold several rooms. Sound passes between them through surfaces whose
material lets some through: `transmission`, the share that passes (1 for an opening),
or `transmission_loss`, the sound reduction index R in dB (the share is 10^(-R/10)),
one value or one per band.

```yaml
materials:
  partition:                 # 12 cm of brick
    absorption: [0.02, 0.02, 0.03, 0.04, 0.05, 0.07, 0.07, 0.07]
    transmission_loss: [30, 35, 40, 46, 52, 57, 60, 60]
  door:
    absorption: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    transmission_loss: 25
  doorway: {transmission: 1}  # an opening: all of it passes
assignments:
  wall_between: partition
  door_leaf: door
  door_hole: doorway
```

- What passes through counts as absorbed on the side it comes from, so the absorption
  must be at least the transmission in every band. A material without absorption
  values, preset or file absorbs what it lets through: an opening absorbs 1.
- The rooms are found from the faces: every space the faces close off on all sides is
  one, whichever way the normals point. The space inside a thick wall or a column, with
  rooms all round it, is solid; it is air when a source or listener stands in it, and
  so is any space it opens into through surfaces that let sound through. The faces
  between two rooms have a room on both sides and reflect and absorb in both.
  `inspect` lists the rooms ("closed, 2 rooms of 48.00, 72.00 m3"), the editor too,
  and the volume is theirs together.
- Early reflections pass through those surfaces, keeping the share that passes at each;
  through openings they pass unchanged. The rays of the late part that hit a partition
  go on as two: the reflection, and a copy behind it with the share that passes, so even
  a 60 dB wall has rays behind it. Each room's rays are thinned and split for the level
  that room is expected to reach (from the rooms' power balance in a diffuse field),
  so a quiet room behind a wall is traced as finely as the loud one. Sound goes straight
  through, in the direction it came.
- Coverage maps cover the audience areas of every room and find at each point what a
  listener there would. The statistical late method follows coupled-room theory: each
  room's diffuse field decays as the energy the rooms exchange through the faces
  between them has it, with Eyring's absorption in each, so a room behind a partition
  holds less and a dry room opening into a live one decays in two slopes. The
  statistical reverberation times of `inspect` count each face between two rooms on
  both its sides, as for one room.
- For diffraction an opening is a hole: the edges of a doorway are the free edges of
  the wall round it, and sound bends round them into the next room. Diffraction through
  a wall is left out, as it is far weaker than what passes straight through.
- The wave solver lets its air flow through openings; it does not model walls that let
  part of the sound through, and says so. Particles in animations pass through
  openings, and through partitions as often as the sound does.
- A listener in a room that no surface joins to the source's room is warned about.
- In the editor, a material's **Lets through** chooses nothing, part of the sound (the
  sound reduction index per band in the table) or everything (an opening).

The tests check the level behind a 30 and a 50 dB partition between two rooms against
diffuse-field theory (within 0.3 dB), two rooms joined by an opening the size of the
wall against one room of both their sizes, and the double-sloped decay of a dry room
opening into a live one against the two-room model (within 1 dB down to -40 dB). The
statistical method follows that model in both rooms, and maps over both rooms match
listeners at their points.

### Audience seating

Sound that reaches a listener low over rows of seats loses much more than the seats'
absorption says, as it passes heads and seat backs on the way. Mark the seating's
material with `audience: true` (in the editor, **Audience seating**):

```yaml
materials:
  seats: {preset: audience_wooden_chairs_1_m2, scattering: 0.7, audience: true}
```

- The attenuation is the fit of Jurkiewicz, Rummler, Green, Wulfrank and Kahle
  (["Impact of grazing incidence attenuation on early reflection design"](https://www.ioa.org.uk/system/files/publications/Y.%20Jurkiewicz%2C%20N.%20Rummler%20%2C%20E.%20Green%20%2C%20T.%20Wulfrank%20and%20E.%20Kahle%20IMPACT%20OF%20GRAZING%20INCIDENCE%20ATTENUATION%20ON%20EARLY%20REFLECTION%20DESIGN.pdf),
  Proc. IOA 47 Pt. 1, 2025) to scale-model measurements over a seated audience: sound
  arriving at an elevation φ above the audience, having run d metres over it, keeps
  min(1, (3.40 sin²φ + 2.17 sin φ + 0.02)^((d - 1)/10)) of its energy. That is 17 dB
  less at grazing incidence after 11 m, 0.6 dB per metre at 5°, and nothing lost from
  17.8° up.
- It applies to the last leg of every early path (the direct sound, reflections and
  diffracted sound) to listeners and map points up to 2 m above audience surfaces: φ
  is the leg's elevation above the surface under the listener, and d how far the leg
  runs over audience surfaces no more than 2 m higher than the listener's ears are
  above theirs, so the stalls do not shade a balcony seat.
- Only in the 500 Hz to 2 kHz octaves (third octaves from 400 Hz to 2.5 kHz), the bands
  of the fit, where it hardly changed with frequency. The seat dip lower down depends on
  the seats themselves, and the fit has no data higher up. The late part is left as it
  is, and the statistical method does not make up what the seats take from the early
  paths. Leaving out the leg from the source to the first reflection, as the fit does,
  errs on the side of too little attenuation.
- In a flat lecture hall with the talker at 1.6 m, it takes 10 dB off the direct sound
  at the back row in those bands and 1.3 dB off C50 there.

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
  the nearest one; in third-octave runs each octave's pattern holds for its thirds.
- CLF (Common Loudspeaker Format) files are not read: their specification is not
  public. Convert them to SOFA or CSV.

`acoustic-spaces directivity speaker.sofa --plot polar.png` prints the directivity
index and front-to-back ratio per octave band and draws the polar patterns;
`acoustic-spaces inspect` lists every source's directivity index.

A coherent array, such as a column loudspeaker or a line array, is a source whose
directivity is the sum of its elements' sound with its phase:

```yaml
sources:
  - name: column
    position: [0.5, 3.0, 2.0]
    directivity:
      array:
        element: cardioid    # each element's pattern, along its aim (default omni)
        count: 8             # a straight column along the up axis, centred...
        spacing_m: 0.15      # ...and this far apart
        steer_deg: -8        # delays that tilt the beam 8 degrees down
  - name: flown
    position: [6.0, 3.0, 6.0]
    directivity: {file: line_array.yaml}   # the same fields in a YAML file
```

```yaml
# line_array.yaml: elements one by one, in the source's frame (x forward, y left, z up)
element: {file: cabinet.sofa}   # or a pattern, as for a source
elements:
  - {position: [0, 0, 0.0], orientation: {pitch: 0}}
  - {position: [0, 0, -0.35], orientation: {pitch: -3}, gain_db: -1, delay_ms: 0.05}
  - {position: [-0.02, 0, -0.70], orientation: {pitch: -7}, gain_db: -2}
  - {position: [0, 0, -1.05], invert: true}   # polarity inverted
```

- The elements' pressures add up with their path differences, their feeds' gains,
  delays and polarities, the steering, and each element's pattern along its aim
  (first-order patterns with their sign, measured ones by magnitude). The energy is
  the exact mean over each third-octave band, on a grid of 1 to 5 degrees fine enough
  for the beam, and with the room's speed of sound. A column of 8 takes about a second.
- The result is the array's **far-field** pattern, a balloon like a measured one, so
  the source stays one point: right beyond about the array's length squared over the
  wavelength, L² f / c (16 m for a 2 m array at 1.4 kHz). Closer to a long array, its
  sound spreads as a line's and its beam has not formed yet; a `line_m` source is the
  better model there. The room's warnings name every listener nearer to an array than
  that distance at 1 kHz: `validate`, `inspect`, `simulate` and `map` print them, and
  the editor lists them under **Properties** when the scene itself is selected. Higher
  up the far field starts farther still:

  | array length | far field from, at 1 kHz | at 4 kHz |
  |---|---|---|
  | 0.5 m | 0.7 m | 2.9 m |
  | 1 m | 2.9 m | 11.7 m |
  | 2 m | 11.7 m | 46.6 m |
- As for any source, `level_db` is the array's sound power in all; the pattern only
  says where it goes. A single element's array has the element's own pattern.
- The tests check a column against the analytic sums of a line array (the
  directivity index within 0.05 dB in every band, the vertical pattern within 0.3
  dB), steering against delays given element by element, a dipole made of two
  elements in opposite polarity, and splayed cardioids against their exact sum.
- In the editor, pick the YAML file with "Measured data or an array from a file…";
  `acoustic-spaces directivity line_array.yaml --plot polar.png` shows its pattern.

How loud a source is, and how it is fed:

```yaml
sources:
  - name: talker
    position: [2.0, 3.0, 1.5]
    level_db: 78            # sound power in all, dB re 1 pW
    spectrum: speech_male   # or flat (default) or speech_female
  - name: delay-speaker
    position: [12.0, 3.0, 2.5]
    level_db: [95, 98, 100, 100, 100, 99, 97, 92]   # or per band (8 or 24 values)
    gain_db: -6                        # gain of the feed
    eq_db: [0, 0, -3, 0, 0, 0, 2, 4]   # equalizer per band, dB
    delay_ms: 18.5                     # plays this long after emission starts
```

- `level_db` is the sound power level, in all or per band. A level in all is spread
  by `spectrum`: evenly over the octave bands (`flat`), or as the long-term spectrum of
  male or female speech (IEC 60268-16), for which the STI then takes the male or female
  weighting. The standard's normal voice, 60 dB(A) at 1 m, is about 78 dB of sound
  power with either speech spectrum and an omnidirectional source. It gives no level
  at 63 Hz, nor at 125 Hz for female speech; those bands are taken 12 dB below the next
  one up, an assumption that does not reach the STI.
- `gain_db` and `eq_db` are the gain and equalizer of the source's feed. Both act on
  the sound power and on the impulse responses, so a mix made with `auralize` hears
  them. G stays the room's: it is measured against the same source, fed the same way,
  in free field.
- `delay_ms` delays everything the source sends. Its impulse responses start that much
  later, and in the coverage map of all sources together its sound arrives that much
  later, as when delay loudspeakers are aligned with a stage.
- `simulate` reports the sound pressure level per octave band and A-weighted
  (`spl_db` and `spl_a_db` in `parameters.json`), the editor's results show it per
  band, and `inspect` lists each source's sound power. In free field, the level 4 m from
  a source comes out as its sound power less 10 log10(4 π r²) = 23 dB and the air's
  absorption over the 4 m, to within 0.01 dB.

Line and surface sources, such as a road, a row of loudspeakers fed alike, a choir
or a vibrating wall:

```yaml
sources:
  - name: traffic
    position: [0.0, 20.0, 0.5]
    orientation: {yaw: 90}   # the line runs along the source's left axis: here -x
    line_m: 40
  - name: panel
    position: [6.0, 0.05, 1.5]
    orientation: {yaw: 90}   # facing +y, into the room
    surface_m: [2.0, 1.2]    # width along the left axis, height along the up axis
    spacing_m: 0.25          # at most this far apart (default 0.5 m)
    directivity: lambert     # a vibrating surface's: in proportion to the cosine
```

- A line or surface is made of uncorrelated point sources, at most `spacing_m` apart,
  each in the middle of an equal share of it. They share the source's sound power and
  its rays equally, each with a seed of its own, and keep its directivity, orientation
  and feed. A source is made of at most 400 of them; their sound adds up in energy, as
  for incoherent sources such as traffic or a crowd. Close to a long line, the level
  falls about 3 dB per doubling of distance rather than a point source's 6 dB (3.3 dB
  between 2 and 4 m from a 40 m line in the tests).
- `lambert` radiates as a vibrating surface, in proportion to the cosine of the angle
  to the forward axis and nothing behind (a directivity index of 6 dB).
- All of a line or surface must be inside the room; `inspect` lists its length or size
  and its number of points. The wave solver takes point sources only. The editor
  draws lines and surfaces in the 3D view and the plan, and the Blender export has
  them too.

![Polar patterns of the per-band directivity above, from omni in the bass to a narrow pattern at 8 kHz](docs/images/directivity.png)

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
or in a corner. Sound that bends round edges is added to them (see
[Edge diffraction](#edge-diffraction)).

**Late reverberation** (`simulation.late`) is either:

- `stochastic_ray_tracing` (default): `rays` rays per band (50,000 by default)
  leave the source and reflect off the surfaces as [Scattering](#scattering) says,
  losing energy to absorption and air.
  Specular paths the early part already computed are left out. How energy reaches the
  listener is set by `reception`:
  - `diffuse_rain` (default): the scattered part of every reflection radiates to
    each listener it can see; the rest counts when a ray crosses a detector
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
  carry less than a quarter (a weight window). The expected result is unchanged; with
  random scattering, the example box's late tail (0.5 to 2.5 s) misses a 1-million-ray
  reference by 0.5 to 2 dB per band instead of 1 to 8 dB at 100k rays (1 to 4 dB
  instead of 4 to 17 dB at 10k), for about 1.5 times the run time.
- `statistical`: the reflected energy of Barron's revised theory, an exponential
  decay at the Eyring reverberation time T from the direct sound on, less what the
  early paths carry. Early paths end `transition_ms` after the direct sound (by
  default the mixing time, the square root of the volume in milliseconds), so the
  late part holds the scattered share of every early reflection, the orders above
  `max_order` before the transition and everything after it. The total is Barron's,
  c T / (13.8 V) for a unit source: the Sabine form 4 / A with Eyring's absorption
  area, which reads 0.2 dB above the classical 4 (1 - α) / (α S) at a mean
  absorption of 0.1 and 0.8 dB above at 0.3.

The two parts hand over by reflection order by default. Set `transition_ms` to
keep early paths only that long after the direct sound.

**Impulse responses** are built per band: early paths become fractional-delay
impulses, the late energy shapes a Dirac sequence with the density of reflections in
the room, and a zero-phase crossover bank sums the bands. The direct sound at r
metres is an impulse of height 1/r, so levels compare between files and the energy
of the direct sound at 10 m (0.01) is the reference for strength G.

**Parameters** follow ISO 3382-1 per octave band: EDT, T20, T30, C50, C80, D50, Ts
and G. `Response.parameters()` computes them from the simulated energies;
`acoustic_spaces.parameters.from_impulse_response` does the same for any mono IR (see
[Measuring impulse responses](#measuring-impulse-responses)). A response that ends
before it has decayed is extended by its fitted decay rate; one that ends before it
has decayed by 20 dB leaves too much to that estimate, and all its parameters are NaN.
Air absorption follows ISO 9613-1 for the scene's temperature, humidity and pressure.

The directions the sound arrives from give more of them:

- LF, LFC and LJ (ISO 3382-1, annex A), from a figure of eight with its null on the
  source: the share of the early energy that arrives from the sides, which widens the
  source, and the level of the late lateral sound, which envelops the listener.
- IACC early and late (annex B), from the binaural response when the scene writes one.
- Stage support ST early (Gade's ST1) and ST late (annex C) for listeners within 1.5 m
  of a source, where musicians hear themselves and each other.
- T30 of figures of eight facing forward, left and up from the listener: where sound
  keeps running between two hard walls, the decay along that axis is longer.
- STI and STIPA (IEC 60268-16) and U50, at the level the source's `level_db` gives and
  the background noise of `coverage.noise_db`, as the coverage maps take them; and the
  echo criteria of Dietsch and Kraak for speech and music.

The echogram adds the arrivals in energy, as ODEON and CATT do. A figure of eight on
the centre line of a symmetric room hears the two side walls' reflections arrive
together with opposite signs, and they cancel in the rendered Ambisonics response.
LF read from that response with `analyze --ambisonic` is then lower than the
echogram's; off the centre line the two agree on average, band by band within the
interference of reflections close in time.

### Scattering

A surface's scattering coefficient (ISO 17497-1) is the share of the reflected energy
that leaves it diffusely rather than specularly. `simulation.scattering` says how rays
carry it out:

```yaml
simulation:
  scattering:
    method: vector     # or random
    from_size: true    # small, distant and grazed surfaces scatter more
    by_band: true      # each band follows its own scattering (vector only)
```

- `vector` (default): every reflection sends the ray in the direction of the sum of its
  specular direction and a random Lambert direction, weighted by the scattering
  coefficient, as ODEON and pyroomacoustics do. The ray keeps all the reflected energy:
  the diffuse rain sends the scattered share to the listeners, and the detector spheres
  count only the rest. Where the bands' coefficients differ, each band is traced with
  rays of its own, so a wall that scatters the bass and reflects the treble does both;
  one direction for all bands (pyroomacoustics takes their mean) made the example box's
  treble decay twice as fast as it should. For the same rays per band, tracing band by
  band takes 3 to 5 times as long as one ray for all bands; at the default 50,000 per
  band a simulation takes 1.5 to 4 times as long as version 2's 100,000 shared rays.
  `by_band: false` goes back to one ray for all bands, turned by their mean scattering.
- `random`: the ray reflects specularly or, with the coefficient as probability, in a
  Lambert direction. This was the only model up to version 2. With smooth walls a few
  rays then keep running between them, and the decay comes out longer.
- `from_size` (default on): a surface also scatters for its size, after Christensen and
  Rindel (2005), as in ODEON. The specular share falls below f_w = c d / (2 (w cos θ)²)
  and again below f_l = c d / (2 l²), where w and l are the surface's width and length,
  d the distance the sound came from and θ its angle of incidence. A large wall reflects
  specularly down to low frequencies, while a small panel, a distant one or one met at a
  grazing angle scatters. Each flat surface (the faces of one plane joined at their
  edges, of any material) counts as the rectangle with its area and perimeter. Early
  reflections lose that share too and the late part picks it up. Reflections that
  [edge diffraction](#edge-diffraction) already completes keep their own coefficient:
  the first two orders with `first_order` diffraction (the default) or `second_order`,
  the first with `direct`.

In the two measured rooms of the [validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md),
the default model brings the reverberation time of a chamber music hall from 20 to 32 %
long down to 7 to 20 %, and keeps a seminar room within 2 % from 500 Hz to 4 kHz.
Without the surfaces' sizes, neither model fits both rooms.

### Frequency bands

Simulations run in the eight octave bands from 63 Hz to 8 kHz by default, or in the 24
third-octave bands from 50 Hz to 10 kHz:

```yaml
simulation:
  bands: third_octave   # or octave (default)
```

- Everything that depends on frequency follows the bands: materials, air absorption,
  source directivity, the scattering of a surface's size, the rays (`rays` per band)
  and the filter bank the impulse response is built with. With the wave solver on, it
  computes the third octaves of the octaves up to `max_band_hz`.
- Materials hold 8 or 24 values. An octave value holds for each of its three third
  octaves, and an octave simulation takes the mean of a third-octave material's three.
- Room parameters, the STI, coverage maps, plots and the editor's results stay in
  octave bands: each octave's energy is the mean of its three thirds'. A response's
  `energy()` has all 24 bands and `octave_energy()` the octaves.
- Third octaves pay where coefficients change within an octave: absorbers narrower
  than an octave (a Helmholtz resonator, a slot absorber), and the 8 kHz octave, across
  which the air absorbs about twice as much at 10 kHz as at 6.3 kHz. Its decay is then
  no single slope, and in the shoebox example its T30 comes out 8 % longer in third
  octaves. In the measured seminar room of the
  [validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md)
  they bring the reverberation time at 250 Hz and 8 kHz from 13 and 17 % short to
  within 4 %. They trace three times the rays and take about three times as long.
- `acoustic-spaces simulate --bands third_octave` sets them for one run, and the
  editor's simulation page has them. A material with third-octave data shows its 24
  values in the editor, and `inspect` lists its octave averages.

### Edge diffraction

Sound bends round the edges of walls, screens and balconies: a listener round a corner
still hears the source, quieter and duller, and the level does not drop off a cliff
where a wall cuts a path off. `simulation.early.diffraction` adds this to the early
part with the uniform theory of diffraction (UTD, Kouyoumjian and Pathak), averaged over
each octave band:

- `first_order` (default): the direct sound and the first-order reflections bend round
  edges, with the reflection before or after the edge.
- `second_order`: as `first_order`, and the direct sound bends round two edges in turn:
  over a thick wall or barrier, round both corners of a doorway's jamb, over two screens
  in a row.
- `direct`: only the direct sound.
- `none`: specular paths only.

The edges are found in the model: corners where two faces meet with more than 195° of
open air round them (the inside corner of an L-shaped room, the front of a stage or a
balcony) and the free edges of thin screens. Each source and listener get the shortest
path over each edge they can both see. In the shadow of the edge that path carries the
diffracted sound; on the lit side, near the boundary, it changes the energy of the
specular path it borders. Across a boundary the level is continuous: the tests cross one
in an L-shaped room in millimetre steps, with first-order reflections, and the level
changes by less than 0.25 dB where specular paths alone drop by more than 5 dB; where a
reflection off one of the corner's walls ends and the direct sound carries on, by less
than 0.05 dB. The
diffraction coefficient matches the exact solutions for a half-plane and for wedges
(within 0.25 dB), and a thin screen matches Maekawa's chart within 1.5 dB.

Behind a thick wall or barrier, whose two corners share its top, no edge sees both the
source and the listener: `first_order` finds nothing there, or only a way round one
corner of a doorway. `second_order` follows the shortest path over each pair of edges
that see each other. The first edge's wave reaches the second as the UTD has it, with
the second edge's distance parameter from the curvature of that wave where it gets
there, since near the first edge's boundaries it is still curved like the source's.
Where one of the two edges' waves is lit, the path continues a path over the other edge
alone and shares its energy with it, so the level is continuous where a path over one
edge appears: as the listener rises past the top of a barrier, it changes by at most
0.8 dB at 63 Hz and 0.1 dB from 1 kHz up. Against the exact solution (BTM, in the
[validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md#edge-diffraction-against-the-exact-solution)),
the level over barriers 0.3 to 3 m thick is within 0.35 dB from 500 Hz up and 2 dB at
63 Hz, and over two screens in a row within 0.75 dB and 2.7 dB.

```sh
acoustic-spaces simulate examples/gallery/scene.yaml --out gallery
acoustic-spaces view examples/gallery/scene.yaml --paths --listener "round the corner"
acoustic-spaces simulate examples/gallery/scene.yaml --out plain --diffraction none
acoustic-spaces view examples/archway/scene.yaml --paths --listener "behind the wall"
acoustic-spaces simulate examples/archway/scene.yaml --out archway --diffraction first_order
```

![Early level along a walk round the corner of the gallery example, with and without diffraction](docs/images/diffraction.png)

The figure walks round the corner of the `gallery` example, away from the guide. Without
diffraction (dotted) the direct sound and each reflection stop at the line where the
corner hides them; with it, the level falls smoothly, by more at high frequencies.
Paths with two or more reflections still stop at their boundaries: with `max_order: 3`
in the gallery, the early energy still steps by up to 3 dB where the reflections off the
long wall go out of sight. The late part is unchanged, since its
scattered rays already find their way round corners.

`view --paths` and the Blender export draw diffracted paths in their own colour, the
response's `early.diffracted` marks them, and `simulate` reports how many there are. The
cost grows with the number of edges: in a hall with about 100 edges, `first_order` adds
about 17 ms per source and listener. `second_order` costs more, as it looks at pairs of
edges: in a hall with 16 edges, the early stage takes 200 ms per source and listener
with it, 120 ms with `first_order` and 100 ms without diffraction.

### Low frequencies: the wave solver

Rays treat sound as rays, which holds while a room is large next to the wavelength. At
low frequencies it is not: a small room rings at its modes, so the bass booms at one
seat and vanishes at the next, and sound bends round everything. With
`simulation.wave`, the lowest octave bands come from the sound field itself:

```yaml
simulation:
  wave:
    enabled: true
    max_band_hz: 125           # the highest band it computes: 63, 125 (default), 250 or 500
    points_per_wavelength: 8   # grid cells per wavelength at the top of that band
    scheme: auto               # the grid's stencil: auto (default), isotropic or standard
    device: cpu                # where it steps the grid: cpu (default), gpu or auto
```

- The sound field is computed with finite differences in time on a cubic grid over the
  room (an explicit leapfrog scheme at its stability limit), in the compiled core, on
  all cores. Up to the 125 Hz band the cells are 20 cm.
- Waves on the grid travel a little slower the shorter they are. In closed rooms the
  grid takes the interpolated isotropic stencil (19 points): 0.67 % slow at 8 cells per
  wavelength, the same in every direction within 0.03 %. Round open models it takes
  the standard one (7 points), the only one with the matched layer below: up to 1.7 %
  slow, along the grid's axes. `scheme` picks one or the other.
- With `device: gpu` the grid is stepped on the GPU instead, through WebGPU: Vulkan,
  Metal or Direct3D 12, so AMD, NVIDIA, Intel and Apple GPUs alike, with the `gpu`
  extra (wgpu); `auto` takes a hardware GPU when wgpu finds one, and the CPU otherwise.
  The grid is laid out on the CPU either way, and the two agree to within about 1e-5 of
  the largest pressure (single-precision rounding). `acoustic-spaces info` names the GPU
  it finds.
- Surfaces absorb as their materials do in each band: each material becomes a locally
  reacting surface whose admittance changes with frequency (a constant and resonant
  mass-spring-damper branches), fitted so that its random-incidence absorption matches
  the material's in every wave band. One run covers all the bands.
- Closed rooms, open models and thin screens all work. Round an open model the grid
  grows by a perfectly matched layer of 8 cells, which takes in the sound leaving it:
  what comes back is 57 to 74 dB down, against 7 to 14 dB from the grid's edge alone.
- The listener records the pressure and its gradient, so the low end keeps its
  direction in Ambisonics, binaural and loudspeaker outputs (to first order).
- The result replaces the rays from 22 Hz up to the top of the highest wave band (177
  Hz for 125), through the same crossovers the impulse response synthesis uses. The
  parameters of those bands come from it too.

```sh
acoustic-spaces simulate examples/concert_hall/scene.yaml --out hall --wave
acoustic-spaces simulate examples/studio/scene.yaml --out studio --wave --wave-band 250
acoustic-spaces simulate examples/church/scene.yaml --out church --wave --wave-device gpu
```

The editor's Simulation page has the same switch, and where to run it. Each octave
higher costs about 16 times as much: on 4 cores, up to the 125 Hz band, the studio takes
0.4 s, the concert hall (9,400 m³, 3.5 s responses) 15 s, and the church (7 s
responses) 28 s per source; 8 s and 19 s with `scheme: standard`, at the cost of its
dispersion. Grids of more than 40 million cells are refused. `python
validation/wave_gpu.py` times your GPU against your CPU; Mesa's lavapipe, a Vulkan
driver that runs on the CPU (and what CI tests with), takes about 9 times as long as
the CPU solver.

![Transfer function of a 7 x 5 x 3 m box from the sum of its modes, the wave solver and rays](docs/images/wave.png)

In a 7 x 5 x 3 m box with 15 % absorption, the wave solver follows the sum of the box's
modes from 22 Hz up; rays give the average level instead. In the 63 and 125 Hz bands,
G is within 1 dB of the modes and T20 within 5 %; rays are up to 5.3 dB and 12 % off.

What it leaves out: surfaces react locally, so their absorption changes with the angle
only as an impedance's does, and absorb at most 95 % at random incidence; next to a
much more absorbing band, a band absorbs somewhat more than its material says (the
fit reports how much, as `wave_absorption_miss`); near the top of the highest band the
grid's walls absorb a little less than their admittance (0.025 less at 500 Hz with 14
cells per wavelength); walls on the grid lie within half a cell of the model's, which can move a small
room's modes by a few percent (more `points_per_wavelength` moves them less); waves on
the grid travel slower the shorter they are, at the top of the highest band with 8
points per wavelength by 0.67 % in closed rooms and up to 1.7 % round open models
(along the grid's axes, not at all along its diagonals), which nothing corrects, so the
modes there come out up to that much low;
sources radiate equally in all directions in the wave bands; and air absorption,
negligible there, is left out.

### Validation

The tests check the simulation against theory on Linux, macOS and Windows: exact image
sources in a box, Eyring's reverberation time and Barron's revised theory in a diffusing
box, ISO 3382-1 parameters of known decays, spatial encoders against closed forms, the
convolvers against direct convolution, and sweep measurements of synthetic rooms. A
cross-check against pyroomacoustics in four rooms agrees within 5 % on T30 with the same
scattering model, from purely specular to fully diffusing surfaces.
Against measurements in two rooms of the BRAS benchmark, a seminar room comes out
within 2 % on reverberation time from 500 Hz to 4 kHz and within one JND on clarity, and
a chamber music hall within about one JND on clarity and 7 to 20 % long on reverberation
time. Edge diffraction
is checked against the exact solutions for a half-plane and for wedges, against
Maekawa's chart, and over two edges against the exact BTM solution. The
wave solver matches the frequencies of a box's modes within 0.25 % and their decay rates
within 2 % with walls whose admittance changes with frequency, the reflection and
absorption of walls, and the free-field level within 0.35 dB, and round an open model
what comes back from the grid's edge is 57 dB down or more. Sound
through a partition reaches the level behind it that diffuse-field theory gives within
0.3 dB, and coupled rooms decay in the two slopes of the two-room model.
The [validation report](https://github.com/naelolaiz/acoustic_spaces_simulator/blob/main/docs/validation.md)
has the details, the numbers and the scripts to reproduce them.

## Coverage maps

A coverage map gives the parameters on a grid over the audience area, the way CATT and
ODEON draw them: strength G and sound pressure level, C50, C80, D50, EDT, T20, T30 and
Ts per octave band, and the A-weighted level and speech transmission index.

```sh
acoustic-spaces map examples/lecture_hall/scene.yaml --plot sti.png --parameter sti
acoustic-spaces map examples/concert_hall/scene.yaml --spacing 0.5 --json map.json \
    --screenshot c80.png --parameter c80 --band 1000
```

![Coverage map of the speech transmission index in the lecture hall example, seen from above](docs/images/coverage.png)

The map covers the surfaces named in `coverage.surfaces` (by default those whose names
contain "audience", else the floor) with square columns `spacing` metres wide, with a
point at the listening `height` above the surface in each; a balcony over the stalls
gets a second level of points. Columns along a slanted wall keep the part over the
surface. The early part is exact at every point, as at a listener. The late part comes
from one ray tracing run per source in which every ray adds its path length through a
slab of air around each point (a track-length estimate), so a map of hundreds of
points costs about as much as a few listeners: the concert hall example (480 points,
two sources) takes about 15 s. In the shoebox example, full simulations at map points
give G within 0.3 dB of the map and T30 within 5 %. With the statistical late method
each point gets the late part a listener there would, and G agrees within 0.01 dB.

Each source radiates its sound power per band (its `level_db`, `spectrum`, `gain_db`
and `eq_db`; see [Sources and directivity](#sources-and-directivity)), which sets the
sound pressure levels. The STI (IEC 60268-16, with the male weighting, or the female
one for a `speech_female` source) takes that as the speech level, with auditory masking
and the reception threshold, and `noise_db` as the background noise. In the map of all
sources together, each source's sound arrives its `delay_ms` later. `map` prints the range of every
parameter, `--plot` saves the top view, `--screenshot` the 3D view and `--json` every
value of every point. Settings live in the scene:

```yaml
coverage:
  surfaces: [audience_floor]   # default: names with "audience", else "floor"
  height: 1.2                  # listening height above the surface, m
  spacing: 1.0                 # distance between points, m
  rays: 50000                  # rays per source and band (default: simulation.late.rays)
  noise_db: 35                 # background noise for the STI, dB SPL (or one per band)
```

From Python:

```python
from acoustic_spaces.engine import coverage_map
from acoustic_spaces.viz import plot_map, show_room

result = coverage_map("examples/concert_hall/scene.yaml")
both = result.get()  # all sources together; result.get("soloist") for one
c80 = both.value("c80", band=4)  # one value per point, 1 kHz
plot_map(both, "c80.png", parameter="c80", room=result.room)
show_room(result.room, coverage=both, parameter="sti")
```

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
  share is dropped. A gap of more than 90 degrees between the speakers on the horizon,
  as behind the listener in 5.x.4, gets imaginary speakers on the horizon that the two
  speakers either side play, so arrivals there pan as in 5.0. Named layouts carry their WAV channel mask; the LFE channel of the
  `.1` layouts stays silent, leaving bass management to the playback system.
- **Binaural** gives every reflection and each late direction an HRTF interpolated
  between the measured directions: the measured directions are triangulated as for
  VBAP, and the responses at the corners of the triangle a direction falls in are
  aligned at their onsets, mixed with its barycentric weights and delayed by the same
  mix of their onsets, so the interaural time difference follows the direction smoothly
  even with a coarse set. Without a SOFA file it uses Brown and Duda's spherical head
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

### Dry recordings to try

These anechoic recordings are free to download, and each one tests something different:

- **BRAS anechoic audio** ([BRAS](https://doi.org/10.14279/depositonce-6726.3), folder
  `4 Additional data/Audio/AnechoicAudio`; CC BY-SA 4.0; WAV, 32-bit float, 44.1 kHz):
  noise pulses and a string quartet with one instrument per channel. Next to them,
  `Audio/Auralization` holds the authors' renderings through the measured BRIRs of each
  room (quartet channels 1 to 4 from LS3 to LS6, the noise pulses from LS7, all heard at
  MP6), so a simulation of the same room, sources and seat can be heard side by side
  with the real one. The noise pulses make the decay and its colour easy to hear.
- **Beethoven, Symphony No. 8** ([TU Berlin](https://doi.org/10.14279/depositonce-6729.2);
  Böhm, Ackermann and Weinzierl; CC BY 4.0; WAV, 32-bit float, 48 kHz): movements
  1, 2 and 4 of a full orchestra, 61 mono tracks per movement (zips of 590 to 740 MB),
  for `auralize` with one source per instrument on a stage, as in the
  [`orchestra` example](#an-orchestra-in-the-concert-hall).
- **Speech**: Cicero's [3rd Catiline Oration](https://doi.org/10.14279/depositonce-8536)
  in Italian, Latin and German, and Demosthenes' [1st Olynthic
  Oration](https://doi.org/10.14279/depositonce-8537) in German (TU Berlin; CC BY-NC-SA
  4.0, non-commercial; mono WAV, 48 kHz), for clarity, intelligibility and echoes in
  lecture rooms and churches.
- **University of Iowa [Musical Instrument
  Samples](https://theremin.music.uiowa.edu/MIS.html)** (free for any use; AIFF, 16-bit
  44.1 kHz, or 24-bit 96 kHz for those recorded since 2011): single notes of orchestral
  instruments, all but the piano recorded in an anechoic chamber. Short attacks
  (xylophone, marimba, pizzicato) make discrete echoes and any click at the convolver's
  partition boundaries easy to hear.
- **Aalto anechoic symphonic recordings**
  ([Aalto University](https://research.cs.aalto.fi/acoustics/virtual-acoustics/research/acoustic-measurement-and-analysis/85-anechoic-recordings.html);
  Pätynen, Pulkki and Lokki, 2008; free for academic research; MP3, PCM on request):
  orchestral excerpts recorded one instrument at a time.

The commands read WAV files only (16, 24 or 32-bit PCM, or float), so convert AIFF, FLAC
or MP3 first, for example with `ffmpeg -i note.aiff -c:a pcm_f32le note.wav`. A dry file
at another rate is resampled to the response's, with a note. A file with several
channels is averaged to mono, so split a multitrack file into one file per source
first; channels that are present but silent also lower the level (the BRAS files have
five channels, of which the noise pulses use one and the quartet four):

```python
from acoustic_spaces.audio import read_wav, write_wav

quartet, rate = read_wav("StringQuartet_4channel.wav")
for n in range(4):
    write_wav(f"quartet_{n + 1}.wav", quartet[:, n], rate)
```

A unit impulse at the response's rate checks the convolution itself: `convolve` gives
the response back to float precision.

### An orchestra in the concert hall

[![Beethoven 8 in a virtual concert hall: the orchestra on the stage of the concert hall and a seat in the middle of the stalls, with the first reflections from a violin to it](docs/images/orchestra_video.png)](https://www.youtube.com/watch?v=c0-5F-iNhIE)

[The video](https://www.youtube.com/watch?v=c0-5F-iNhIE) plays the first movement of the
Beethoven recording above as heard from the middle of the stalls of the concert hall
example, binaurally (use headphones). Each of the 61 tracks is one source on the stage,
in German seating: violins on both sides of the conductor, the winds and brass on
risers, every player facing the conductor with a directivity that narrows above 1 kHz,
horn bells pointing back. The `orchestra` example is that scene, with four seats from
the conductor's podium to the back of the hall, and the scripts that made the video:

```sh
acoustic-spaces examples --copy . orchestra && cd orchestra
unzip movement1.zip -d movement1  # from https://doi.org/10.14279/depositonce-6729.2
acoustic-spaces simulate scene.yaml --out results
bash auralize.sh results movement1/anechoic stalls-middle binaural
python make_images.py --seat stalls-middle --movement 1  # cover, seating, thumbnail
bash make_video.sh beethoven8_movement1_stalls-middle.binaural.wav  # needs ffmpeg
```

`auralize.sh` maps each source to its track and calls `auralize`; the mix is
normalized to -1 dBFS. All sources have the same power, so the balance between the
instruments is the recording's own. `make_images.py` draws the pictures with Pillow
(`viz` extra) and `make_video.sh` puts a scrolling spectrogram of the mix under the
cover. Movements 2 and 4 work the same way.

### Head tracking

With headphones and a head tracker, a binaural rendering stays put in the room while the
head turns. `play --head-tracking` takes an Ambisonics response (`--output
ambisonics:3` writes `.ambix3.wav`), runs the dry signal through it in real time, turns
the sound field against the head once per block and decodes it to the ears:

```sh
acoustic-spaces simulate examples/church/scene.yaml --out church --output ambisonics:3
acoustic-spaces play church/organ__nave-middle.ambix3.wav organ.wav --head-tracking
```

The head's orientation comes from OSC messages on UDP port 9000 (`--osc-port`) of this
computer (`--osc-host 0.0.0.0` takes them from the network too, as from a phone): `/ypr`
with yaw, pitch and roll in degrees, `/quaternion` (or `/quaternions`, `/quat`) with w,
x, y and z, or `/yaw`, `/pitch` and `/roll` on their own, under any prefix
(`/SceneRotator/ypr` too). Yaw turns left, pitch looks up and roll tilts the head to the
right; `--invert yaw,pitch` flips the axes a tracker counts the other way round. Each
turn crossfades over one block, so the head moves without clicks, and the latency is
still the block.

The decoder fits the HRTF (a SOFA file with `--hrtf`, or the spherical head model) with
the spherical harmonics up to the response's order. Above 600 Hz per order it fits the
magnitudes only (MagLS, after Schörkhuber, Zaunschirm and Höldrich, 2018): with the
spherical head at third order, every octave from 125 Hz to 16 kHz keeps its level within
0.1 dB, and the level difference between the ears is within 0.5 dB rms over all
directions, where a plain least-squares fit (`--decoder ls`) is 16 dB short at 16 kHz.

In the editor, tick **Head tracking** in the Results panel: the yaw and pitch sliders
turn the head by hand, a tracker sending to the **OSC port** turns it as you move
(**Centre** takes your current orientation as straight ahead), and **Listen** plays
through the head-tracked rendering.

From Python:

```python
from acoustic_spaces.headtracking import HeadTrackedBinaural, OscReceiver
from acoustic_spaces.render import play
from acoustic_spaces.spatial import Ambisonics

ambix = response.spatial_impulse_response(Ambisonics(order=3), 48_000)
renderer = HeadTrackedBinaural(ambix, 48_000, block=256)
renderer.orientation.look(yaw=30)  # by hand, or let a tracker turn it:
with OscReceiver(renderer.orientation, port=9000):
    play(renderer, 48_000, speech_48k)
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
  5.3.4). Digital silence at the end of a file (zero padding, or the shift that
  removes latency) is not taken for noise: the noise floor is looked for before it.
  Decay times are left out when a band's decay does not clear the noise by enough:
  20 dB for EDT, 35 dB for T20, 45 dB for T30.
- G needs the source's free-field level: `--free-field ff.wav --distance 3` uses a
  measurement of the same source in free field (or with reflections windowed out),
  and `--reference-energy 0.01` suits responses made by `simulate`.
- `analyze` also prints the STI and STIPA of the room alone and the echo criteria.
  `--lateral 2` takes channel 2 as a figure of eight with its null on the source, for
  LF, LFC and LJ; `--ambisonic` reads first-order AmbiX (W, Y, Z, X) facing the source,
  for those and the T30 of X, Y and Z; `--binaural` takes two channels as the ears and
  adds IACC; `--stage` adds stage support for a response measured 1 m from the source.

![analyze on a measured response: the level with each band's noise floor from where the decay meets it, and the decay curves with the noise taken out](docs/images/measured.png)

The response above was measured in a seminar room of the BRAS benchmark (Aspöck et al.,
DOI 10.14279/depositonce-6726.3, CC BY-SA 4.0, which the picture keeps): room CR2,
loudspeaker 1 to microphone 1.

From Python:

```python
from acoustic_spaces.measurement import Sweep, deconvolve, measure
from acoustic_spaces.parameters import from_impulse_response

sweep = Sweep(duration_s=10, silence_s=4, repeats=3)
result = measure(sweep, inputs=[1, 2], loopback=4, device=3)
for ir in result.impulse_responses.T:
    print(from_impulse_response(ir, sweep.sample_rate).table())
```

## Comparing with measurements

A model is checked against the room it stands for by comparing its parameters with
those measured there. `compare` analyses a simulated and a measured impulse response
the same way, each at its own sample rate, and shows band by band how far apart they
are in just noticeable differences (JNDs, ISO 3382-1:2009, annex A): 5 % of the
measured EDT, T20 and T30, 1 dB of C50, C80 and G, 0.05 of D50 and 10 ms of Ts.

```sh
acoustic-spaces simulate hall.yaml --out sim
acoustic-spaces compare sim/talker__seat-a.wav hall/ir.wav --measured-channel 2 \
    --simulated-reference-energy 0.01 --measured-free-field ff.wav --measured-distance 3 \
    --plot compare.png --json compare.json
```

```
simulated: talker__seat-a.wav, channel 1, 48000 Hz, direct sound at 10.1 ms
measured:  ir.wav, channel 2, 48000 Hz, direct sound at 10.0 ms

                  63 Hz  125 Hz  250 Hz  500 Hz   1 kHz   2 kHz   4 kHz   8 kHz
...
T30 (s)   sim     1.533   1.372   1.267   1.111   1.030   1.004   0.899   0.761
          meas        -   1.252   1.214   1.029   1.040   0.959   0.852   0.731
          JNDs        -    +1.9*   +0.9    +1.6*   -0.2    +0.9    +1.1*   +0.8
C50 (dB)  sim      -1.6     2.0    -2.0    -1.9     1.5    -0.1     0.3     1.8
          meas     -1.9     0.7     2.2    -1.2     1.8     0.5     1.3     2.3
          JNDs     +0.2    +1.3*   -4.1*   -0.7    -0.4    -0.6    -1.1*   -0.5
...
JNDs: simulated minus measured, in just noticeable differences; * beyond one
missing:
  T20 at 63 Hz: the measured decay clears the noise by 34 dB, T20 needs 35 dB
  T30 at 63 Hz: the measured decay clears the noise by 34 dB, T30 needs 45 dB

            within 1 JND   mean |JNDs|
EDT (s)           4 of 8           2.0
...
```

- Each JNDs row is the simulated value minus the measured one; `*` marks the bands
  more than one JND apart, where the difference can be heard.
- The summary gives, per parameter, how many of the compared bands are within one JND,
  and the mean absolute difference in JNDs from 125 Hz to 4 kHz.
- A band where either side has no value shows `-` and is listed with the reason, never
  counted as a difference of zero: a response that decays by less than 20 dB, a decay
  time whose range does not clear the background noise, or G without a reference.
- Each file has its own channel (`--simulated-channel`, `--measured-channel`, 1 by
  default) and its own reference for G, given as for `analyze`:
  `--simulated-reference-energy 0.01` suits responses made by `simulate`, and the
  measured side takes `--measured-free-field` with `--measured-distance`, or
  `--measured-reference-energy`. G is compared only when both sides have one.
- `--filters` and `--no-noise` apply to both files. `--json` writes every value and
  reason, and `--plot` overlays the two decays in each octave band (viz extra).

From Python, `compare_impulse_responses` does the same, and `compare_parameters`
compares two sets of parameters already computed:

```python
from acoustic_spaces.audio import read_wav
from acoustic_spaces.compare import compare_impulse_responses

sim, sim_rate = read_wav("sim/talker__seat-a.wav")
meas, meas_rate = read_wav("hall/ir.wav")
comparison = compare_impulse_responses(
    sim[:, 0], sim_rate, meas[:, 1], meas_rate, simulated_reference=0.01
)
print(comparison.table())
t30 = comparison["t30"]
print(t30.difference_jnd, t30.within_jnd, t30.mean_abs_jnd, t30.missing)
```

## Fitting materials to measurements

Absorption from tables seldom matches a room as built. `calibrate` adjusts the
absorption of the scene's materials, octave band by octave band, until the simulated
T30, EDT and C80 at each measured source and listener pair come as close as they can to
those measured, in JNDs:

```sh
acoustic-spaces calibrate hall.yaml \
    --measured talker seat-a hall/ir_a.wav --measured talker seat-b hall/ir_b.wav \
    -o hall_fitted.yaml --json calibration.json
```

```
absorption     63 Hz  125 Hz  250 Hz  500 Hz   1 kHz   2 kHz   4 kHz   8 kHz
floor           0.09    0.14    0.23    0.33    0.40    0.46    0.47    0.46
  was           0.13    0.20    0.33    0.46    0.52    0.59    0.59    0.59
  +-            0.36    0.20    0.23    0.08    0.08    0.04    0.08    0.02
...
mean |error|   63 Hz  125 Hz  250 Hz  500 Hz   1 kHz   2 kHz   4 kHz   8 kHz   (JND)
T30 before      5.86    5.84    5.36    5.78    5.32    5.14    5.77    5.06
T30 after       0.04    0.07    0.07    0.11    0.08    0.08    0.06    0.09
...
```

- Each `--measured SOURCE LISTENER FILE` is an impulse response measured where the
  scene has that source and listener: a WAV (its first channel, or `--channel`), or a
  report written by `analyze --json`. `--measured-json` takes parameters already
  measured, as `{"pairs": {"SOURCE LISTENER": {"t30": [...], "edt": [...]}}}` with eight
  octave values each (null where missing), the format of `validation/bras/*/measured.json`.
- Every material on a surface is fitted (at most 16), or those `--materials` names;
  `--parameters` picks others to match (T20, C50, D50, Ts, G).
- The table gives each material's fitted absorption, what it started from (`was`), and
  `+-`: how far it could move, the other materials following it, before the fit worsens
  by one JND. Where that is a tenth or more, the measurements hardly decide it. `-o`
  writes the scene with the fitted absorption (a material given in third octaves keeps
  its shape within each octave), and `--json` every number.
- The materials keep their proportions unless the measurements ask otherwise.
  Measurements fix the total absorption best. Telling two materials apart needs
  responses that hear them differently, and surfaces that the sound meets alike (the
  floor and ceiling of a flat room) trade absorption freely. So the fit may scale every
  material by a common factor, but moving one by a factor of 2 more than the others
  costs as much as missing a parameter by one JND. With measurements that cannot tell
  the materials apart, it is the fit of one factor per band, like fitting the
  reverberation time with Eyring's formula, but with the rays of the model itself.
- It does not trace the room for every trial. Each round traces it once, counting each
  ray's reflections off each material, and the responses for other absorption come from
  re-weighting those rays (`acoustic_spaces.engine.reweight`); the next round traces
  again at the absorption found, until it moves by less than 0.01 or `--rounds` (4) run
  out. A round costs about one simulation of the measured sources at the scene's rays
  (`--rays` overrides them).
- It needs stochastic ray tracing in octave bands, without the wave solver.
- On the two measured rooms of the validation report, fitted to one loudspeaker
  position from tabulated absorption, the responses at the other position come within
  about one JND: as close as with the absorption BRAS fitted with Eyring's formula, or
  closer.

From Python:

```python
from acoustic_spaces.calibration import calibrate
from acoustic_spaces.parameters import from_impulse_response

measured = {("talker", "seat-a"): from_impulse_response(ir_a, 48000)}
found = calibrate("hall.yaml", measured)
print(found.table())
found.fitted_scene().to_yaml("hall_fitted.yaml")
```

## Repository layout

```
src/acoustic_spaces/   Python package
  scene/               scene model and YAML schema
  geometry/            CAD import, triangulation, mesh cleanup and checks
  materials/           material database and coefficient files
  directivity.py       source directivity: patterns per band, SOFA and CSV balloons
  room.py              a scene with its geometry loaded: materials per face, checks
  engine/              early reflections, edge diffraction, late reverberation, the
                       wave solver, responses, simulate(), coverage maps
  air.py               ISO 9613-1 air absorption and speed of sound
  synthesis.py         octave filter bank and impulse response synthesis
  spatial/             Ambisonics, VBAP loudspeaker layouts, HRTFs, spatial rendering,
                       sound field rotation, binaural decoding of Ambisonics
  parameters.py        ISO 3382-1 parameters, octave filters, noise floor detection
  render.py            convolution, resampling, auralisation, real-time playback
  headtracking.py      head-tracked binaural playback, OSC head trackers
  measurement.py       sine sweeps, playback and recording, deconvolution
  compare.py           simulated against measured parameters, in JNDs (ISO 3382-1)
  audio.py             WAV files
  viz.py               3D view, reflection paths, particle animations, response plots
  gltf.py              export to glTF (.glb) for Blender: room, paths, particle animation
  results.py           writing a simulation's responses and parameters to a folder
  examples.py          the example scenes that ship with the package
  gui/                 the graphical editor (Qt)
  cli.py               command-line interface
cpp/include/           header-only C++ core: BVH, ray tracer, wave solver (FDTD), FFT,
                       partitioned convolver (no Python, so a plugin or player can
                       reuse it)
cpp/bindings/          nanobind module exposing the core as acoustic_spaces._native
tests/                 pytest suite, run on Linux, macOS and Windows in CI
examples/              example scenes and their scripts, installed with the package
validation/            the longer checks behind the validation report; bras/ has the
                       measured rooms of the BRAS benchmark
docs/                  the Sphinx documentation, the validation report and the pictures
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

v2, version 2.0.0 (version 1.1.0 had the first three):

- [x] A graphical editor: scenes, materials, sources and listeners, runs, results and listening.
- [x] Export to Blender: rooms, reflection paths and particle animations as glTF.
- [x] Source directivity per octave band: patterns, and measured data from SOFA or CSV files.
- [ ] CLF loudspeaker files, once their specification is available.
- [x] Head tracking: binaural playback of Ambisonics that turns with the head, from OSC
  head trackers or by hand.
- [x] A wave-based solver for low frequencies: room modes from finite differences in
  time, joined to the rays.
- [x] Coverage maps: parameters and STI over the audience area.
- [x] Edge diffraction (UTD) of the direct sound and first-order reflections.
- [x] Editing geometry in the editor: drawing floor plans, dragging corners, sources and
  listeners.

v3, version 3.0.0, in progress (scene files may change; the changelog will say how):

- [x] Comparing a simulation with a measurement in just noticeable differences
  (`acoustic-spaces compare`).
- [x] Validation against measured rooms: two rooms of the BRAS benchmark.
- [x] SketchUp files read directly (experimental, the `sketchup` extra).
- [x] More parameters: LF, LFC, LJ, IACC, ST early and late, U50, STIPA, echo criteria,
  directional decay.
- [x] Vector-based scattering, and scattering that follows surface size and frequency.
- [x] Third-octave bands.
- [x] Sources with their own spectra, gain, EQ and delay; speech spectra for the STI.
- [x] Line and surface sources.
- [x] Loudspeaker arrays with phase (their far-field patterns).
- [x] Sound through walls and between coupled rooms.
- [x] Fitting absorption to measured parameters (`acoustic-spaces calibrate`).
- [x] Wave solver walls that absorb as their materials do in every band, in one run, and
  a matched layer round open models.
- [x] A GPU path for the wave solver through WebGPU (AMD, NVIDIA, Intel and Apple GPUs
  alike).
- [x] A wave solver stencil with less than half the dispersion, the same in every
  direction.
- [x] Second-order edge diffraction: the direct sound over two edges, checked against
  the exact solution.
- [ ] Walkthroughs with moving sources and listeners; IFC and Collada import; CATT and
  ODEON material libraries.

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
