# User manual

This manual is a path through the documentation, from a first simulation of a plain
room to the advanced workflows. Each chapter names an example scene to try and a command
or an editor menu to try it with, and links to the page that describes it in full; the
pages themselves are the reference, so the manual repeats none of their detail. Read it
in order the first time, or jump to a chapter from the contents.

The examples come with the package. `acoustic-spaces examples --copy examples` copies
them into a folder `examples` that you can edit, so the commands below run as written,
from an installed copy as from a source checkout. The [example scenes](examples.md) page
lists them all, in the groups the editor's **File > Open example** shows.

## Contents

1. [First steps](#first-steps): installing, a first simulation, looking at the room,
   the editor and listening to the result.
2. [Building a scene](#building-a-scene): the geometry, from dimensions, a floor plan or
   a CAD or BIM model; materials; sources and listeners; rooms joined by walls and
   doors; audience seating; the air and the seed.
3. [Simulating rooms](#simulating-rooms): how a run works and its settings, sound round
   corners, low frequencies and the wave solver, how far a result may be off, coverage
   maps, and speed and memory.
4. [Listening in space](#listening-in-space): binaural, Ambisonics, loudspeakers and
   microphones, mixing several sources, real-time playback and head tracking, walks
   through a scene and moving sources.
5. [Advanced workflows](#advanced-workflows): sound systems, outdoor noise, demos,
   measuring a real room and fitting its model, long runs on another computer, and
   scripting in Python.
6. [How far to trust the results](#how-far-to-trust-the-results): the validation
   report and the known issues.
7. [Reference](#reference): every page of the documentation.

## First steps

### Installing

```sh
pip install "acoustic-spaces[viz,gui]"    # with the 3D view and the graphical editor
acoustic-spaces info                      # the version, the compiled core and the GPU
```

The other extras (`measure` for the sound card, `sofa`, `gpu`, `ifc`, `sketchup`) and
installing from a source checkout are in [Getting started](getting-started.md#install).

### A first simulation

The `shoebox` example is an 8 × 6 × 3 m room from an OBJ file, with a talker and one
seat: the simplest case every method must get right.

```sh
acoustic-spaces examples --copy examples
acoustic-spaces validate examples/shoebox/scene.yaml   # the scene's checks
acoustic-spaces inspect examples/shoebox/scene.yaml    # geometry, materials, reverberation estimates
acoustic-spaces simulate examples/shoebox/scene.yaml --out results --plot
```

`simulate` prints the ISO 3382-1 parameters per octave band, the level, the STI and the
echo criteria, and writes the impulse responses in each format the scene lists, with
`parameters.json` and, with `--plot`, the echogram and the arrival directions. The
printout, why its T30 is longer than the Sabine and Eyring estimates, and which file
holds what are in [Getting started](getting-started.md#install-from-source) and
[Where the results go](getting-started.md#where-the-results-go).

### Looking at the room

```sh
acoustic-spaces view examples/shoebox/scene.yaml --paths --max-order 2     # reflection paths in 3D
acoustic-spaces animate examples/lecture_hall/scene.yaml                   # sound particles
acoustic-spaces export examples/concert_hall/scene.yaml hall.glb --paths   # for Blender
```

These need the `viz` extra. The particles are described under
[Getting started](getting-started.md#install-from-source), and the export under
[Exporting to Blender](editor.md#exporting-to-blender).

### The editor

`acoustic-spaces` alone opens the [graphical editor](editor.md) on a new box room, and
`acoustic-spaces edit examples/studio/scene.yaml` on a scene file. Open an example with
**File > Open example**, run it with **Simulation > Run** (F5) and pick a source and a
listener in the **Results** panel; **Listen** plays a dry recording through the
response (with the `measure` extra). The editor works on the same scene files as the
command line, so everything below can be done in either.

### Listening to the result

A dry (anechoic) recording run through the binaural response is what the seat sounds
like on headphones. [Dry recordings to try](spatial-audio.md#dry-recordings-to-try)
lists free ones; with one saved as `speech.wav`:

```sh
acoustic-spaces convolve speech.wav results/talker__seat-a.binaural.wav --out wet.wav
```

The binaural output uses a spherical head model unless the scene names a SOFA file;
`acoustic-spaces hrtf` lists free measured sets to download
([HRTF sets to download](spatial-audio.md#hrtf-sets-to-download)).

## Building a scene

### Starting a scene

A scene is a YAML file: the geometry, the materials, which surface gets which, and where
the sources and listeners are ([Scene files](scenes.md)). Start from a copy of an
example, or in the editor from **File > New box room**, **New room from a floor plan**
or **New from model file**. A rectangular room needs only its dimensions
(`geometry: {box: [8, 6, 3]}`), and the `lecture_hall` example is raised from its floor
plan, no CAD file needed ([Geometry](scenes.md#geometry)). The editor's **Plan** tab
draws rooms from above ([Drawing rooms](editor.md#drawing-rooms)).

### Models from CAD and BIM files

OBJ is the recommended exchange format; DXF, glTF, STL, IFC, Collada and SketchUp files
are read too, with their units and up axis ([Geometry](scenes.md#geometry)). Try
`rehearsal_room` (a Collada file in inches, as SketchUp exports it) and `bim_office` (an
IFC model read as one room, with the `ifc` extra; [BIM models](scenes.md#bim-models-ifc)).
A model that is not closed lets rays escape; `leaky_classroom` has two such mistakes for
the leak finder to find ([Finding leaks](scenes.md#finding-leaks)):

```sh
acoustic-spaces validate examples/leaky_classroom/scene.yaml --leaks
```

### Materials

Each surface gets a material: absorption and scattering per band, from the built-in
database (`acoustic-spaces materials carpet` searches it), values of your own, or the
material libraries of CATT-Acoustic and ODEON, as in the `material_libraries` example.
A material can say how far its absorption may be off, which the simulation turns into
an uncertainty of every parameter ([Materials](scenes.md#materials)).

### Sources and listeners

A source has a position, an orientation, a directivity (a pattern, a measured balloon
or an array of elements), a sound power or a level, and a spectrum; it can also be a
line or a surface ([Sources and directivity](scenes.md#sources-and-directivity)).
`acoustic-spaces directivity hypercardioid --plot polar.png` shows a pattern. Try
`studio` (stereo loudspeakers at ±30°) and `workshop` (a conveyor as a line source and
a machine casing as a surface source). Sources and listeners sit in groups, and any of
them can be left out of simulations without being deleted
([Groups, and sources left out](scenes.md#groups-and-sources-left-out)).

### Rooms, walls and doors

A model can hold several rooms, with walls that let part of the sound through, doors
and openings: `apartment` hears a television through a wall and a door, closed or open
([Walls, doors and openings](scenes.md#walls-doors-and-openings)). Sound that runs low
over rows of seats loses more than their absorption says; a material marked
`audience: true` adds that loss, as the seats of `town_hall_pa` are
([Audience seating](scenes.md#audience-seating)).

### The air and the seed

The air's temperature, humidity and pressure set the speed of sound and the air's
absorption, and `simulation.ir_length_s` the responses' length. The rays are drawn anew
on each run unless the scene sets `simulation.seed`, with which the CPU's results
repeat from run to run
([Air, response length and seed](scenes.md#air-response-length-and-seed)).

## Simulating rooms

### How a simulation works

Each source and listener pair is simulated in two parts: the early reflections, exact
specular paths found by ray tracing or image sources, and the late reverberation, from
stochastic ray tracing by default. Their settings are under `simulation` in the scene
and in the editor's **Simulation** entry: [Simulation](simulation.md), with how surfaces
[scatter](simulation.md#scattering) and when to use
[third-octave bands](simulation.md#frequency-bands).

### Sound round corners

Sound that bends round an edge is added to the reflections. In `gallery` one listener
stands round the corner from the guide, and in `archway` a diner hears a singer over
both corners of an archway's jamb in a wall 40 cm thick ([Edge diffraction](simulation.md#edge-diffraction)):

```sh
acoustic-spaces view examples/gallery/scene.yaml --paths --listener "round the corner"
```

### Low frequencies

In the lowest bands the early reflections add in pressure, with their phases, so a
loudspeaker near a wall gains or loses bass, as `listening_room` shows
([Early reflections in pressure](simulation.md#low-frequencies-early-reflections-in-pressure)).
For the modes of a small room a wave solver can take over the lowest bands, on the CPU
or a GPU, as in `room_modes`
([The wave solver](simulation.md#low-frequencies-the-wave-solver)).

### How far a result may be off

With the late rays in batches (10 by default) every parameter gets a standard error:
how much it would change on a run with another seed. `simulate` also propagates the
materials' absorption uncertainty (plus or minus 0.1 for a built-in material unless the
scene says otherwise) and the rays' error into each parameter's combined uncertainty and
95 % interval, by Monte Carlo (GUM Supplement 1), for runs in octave bands without the
wave solver. It prints the largest of both after
the parameters and keeps them all in `parameters.json`; the editor gives them in the
results' tips ([Simulation](simulation.md), [Materials](scenes.md#materials)).

### Coverage maps

A coverage map gives the parameters, the A-weighted level and the STI on a grid over the
audience area ([Coverage maps](coverage-maps.md)); in the editor, **Simulation > Run
coverage map** (F6).

```sh
acoustic-spaces map examples/lecture_hall/scene.yaml --plot sti.png --parameter sti
```

### Speed and memory

Rays, bands, the wave solver and maps set how long a run takes and how much memory it
needs. The late stage and the wave solver can run on a GPU with the `gpu` extra, and
`acoustic-spaces tune` finds the fastest device on a computer
([Performance](simulation.md#performance), [Memory](simulation.md#memory),
[Tuning the devices for a computer](simulation.md#tuning-the-devices-for-a-computer)).

## Listening in space

### Spatial formats and microphones

The scene's `outputs` list picks the formats: mono, binaural (spherical head or a SOFA
file), Ambisonics up to the seventh order, and loudspeaker layouts from stereo to 7.1.4
or of your own; `simulate --output` picks them for one run. A listener can also pick up
its mono output through a microphone, a first-order pattern or a stereo pair
([Spatial outputs](spatial-audio.md#spatial-outputs),
[Microphones](spatial-audio.md#microphones)).

### Mixing several sources

`acoustic-spaces auralize` mixes every source's own dry signal as heard at one listener,
in any format the simulation wrote
([Listening](spatial-audio.md#listening-convolution-and-real-time-playback)). The
`orchestra` example takes this to 61 sources, one for each track of an anechoic
recording of a symphony ([An orchestra in the concert hall](demos.md#an-orchestra-in-the-concert-hall)).

### Real-time playback and head tracking

`acoustic-spaces play` runs a recording, or the microphone with `--live`, through a
response on the sound card in real time (the `measure` extra). With a head tracker and
an Ambisonics response, `--head-tracking` keeps the sound in place while the head turns
([Head tracking](spatial-audio.md#head-tracking)).

### Walking through a scene

A listener or a source can follow a [walk](scenes.md#walks), drawn in the editor's
**Movement** panel, and be heard as it goes. In `gallery` a visitor walks away from the
guide and round the corner ([Walkthroughs](spatial-audio.md#walkthroughs)); a fast
source, such as a passing car, also drops in pitch
([Moving sources and Doppler](spatial-audio.md#moving-sources-and-doppler)).

## Advanced workflows

### Sound systems

Loudspeaker arrays add up box by box, with each feed's gain, equalizer and delay. The
examples go from simple to large, each with scripts that print what a system designer
checks ([Example scenes](examples.md#sound-systems)):

- `town_hall_pa`: two steered columns for speech and two delay loudspeakers;
  `delays.py` maps the STI with and without the delays.
- `concert_stage`: two line arrays, subwoofers, fills and the stage's monitors;
  `align.py`, `feedback.py` (gain before feedback) and `near_field.py`.
- `stadium_concert`: a touring system with delay towers and the same monitors, mapped
  over the pitch and the stands.

### Outdoor noise

A scene is outdoors when its geometry has a ground ([Outdoor scenes](outdoor.md); the
editor's **File > New outdoor scene**, [outdoor scenes in the editor](editor.md#outdoor-scenes)).
The outdoor examples add one effect at a time:

- `street`: buildings and a noise barrier, whose insertion loss
  `simulate --without noise_barrier` gives
  ([Buildings and barriers](outdoor.md#buildings-and-barriers)).
- `roadside`: the ground's impedance, grass against a paved yard
  ([Ground impedance](outdoor.md#ground-impedance)).
- `tree_belt`: foliage and trunks ([Vegetation](outdoor.md#vegetation)).
- `hillside`: a terrain from a grid of heights and houses from a map
  ([Terrain](outdoor.md#terrain), [Buildings from a map](outdoor.md#buildings-from-a-map)).
- `open_air_stage`: wind, downwind and upwind, and turbulence
  ([Wind and temperature](outdoor.md#wind-and-temperature),
  [Turbulence](outdoor.md#turbulence)).
- `lakeside_concert`: weather zones over a lake and a village
  ([Weather zones](outdoor.md#weather-zones)).
- `railway`, `helicopter_flyover` and `airport`: trains by Schall 03, a helicopter from
  its rotors and aircraft by ECAC Doc 29
  ([Passing vehicles](outdoor.md#passing-vehicles),
  [Aircraft by ECAC Doc 29](outdoor.md#aircraft-by-ecac-doc-29)).

What a listener gets outdoors (the level relative to free field, insertion losses,
levels over a long term) is in [Results outdoors](outdoor.md#results-outdoors) and
[Levels over a long term](outdoor.md#levels-over-a-long-term); how much traffic and
works shake a building, in [Vibration through the ground](outdoor.md#vibration-through-the-ground).

### Demos

`acoustic-spaces demo my_room/scene.yaml` (or **Simulation > Run demo…**) has each
source play its recording, mixes what every listener hears and makes a video that tours
them ([A demo of any scene](demos.md#a-demo-of-any-scene)). Several examples come with
demos of their own, run step by step ([Running a demo](demos.md#running-a-demo)); each
shows one space and one kind of sound, not the simulator as a whole.

### Measuring a real room and fitting its model

`acoustic-spaces measure` records sine sweeps through a loudspeaker and a microphone and
analyses them; `compare` sets a simulated response beside a measured one in just
noticeable differences; `calibrate` fits the materials' absorption to the measurements
([Measurements](measurements.md)). Without a sound card, the `measurement` example
records a sweep in a simulated room, and the `fitting` example makes up measurements of
a seminar room to compare and fit.

### Long runs on another computer

`acoustic-spaces pack` (**File > Pack for another computer**) puts a scene and every
file it names in one zip, which every command that reads a scene takes in its place. A
container image runs it on a GPU server over SSH or as a cloud job
([Running on another computer](remote.md)).

### Scripting in Python

Everything the command line does is in the Python package: `Room.from_yaml` reads a
scene, `simulate` runs it, and each response gives its parameters and its impulse
responses in every format. The short tour is at the end of
[Getting started](getting-started.md#install-from-source), and every class and function
is in the [API reference](api.md). `acoustic_spaces.examples.example_path("church")`
gives an example's scene file.

## How far to trust the results

The [validation report](validation.md) compares the results with theory, with
pyroomacoustics and with measured rooms (BRAS), with the numbers and the scripts to
reproduce them; its [known issues](validation.md#known-issues) say what is still wrong. Read it
before relying on a result for a decision.

## Reference

- [Getting started](getting-started.md): installing, the command line and the Python API,
  where the results go, the language.
- [Graphical editor](editor.md): editing scenes, drawing rooms, outdoor scenes, export
  to Blender.
- [Example scenes](examples.md): every bundled scene and what it shows.
- [Scene files](scenes.md): geometry, materials, walls and openings, seating, sources,
  walks, air.
- [Simulation](simulation.md): the methods and their settings, diffraction, low
  frequencies, performance.
- [Spatial audio and listening](spatial-audio.md): output formats, convolution,
  real-time playback, walkthroughs, Doppler, head tracking.
- [Coverage maps](coverage-maps.md): parameters and the STI over the audience.
- [Outdoor scenes](outdoor.md): ground, terrain, buildings, weather, vehicles, aircraft,
  vibration.
- [Demos](demos.md): every demo, its recordings and options.
- [Measurements](measurements.md): measuring, comparing and fitting.
- [Running on another computer](remote.md): packs, the container image, servers and
  cloud jobs.
- [Validation report](validation.md) and [API reference](api.md).
- [Roadmap](roadmap.md), [development](development.md),
  [translating](translating.md) and the [changelog](changelog.md).
