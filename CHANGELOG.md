# Changelog

## 1.1.0

The graphical editor, export to Blender and source directivity.

**Graphical editor** (`acoustic-spaces edit`, or `acoustic-spaces-editor`; the `gui` extra)

- A Qt window with the room in 3D, a scene tree, a form for each item and a message log.
  Every change is checked against the scene schema and can be undone.
- Sources and listeners placed by clicking a floor or wall; surfaces given materials,
  materials added from the database and their coefficients edited per octave band.
- New rooms as a box or from a model file, box and floor plan dimensions, model units
  and up axis, simulation settings and output formats, all from forms.
- Simulations in the background with progress and cancel; results with ISO 3382-1
  parameters, echograms, decay curves, arrival directions and reflection paths in 3D;
  export to the files `simulate` writes; listening to a WAV file through a response.
- The results show the source and listener selected in the scene, and stay on them
  after another run. The selected point has arrows to its front and to its top, so
  yaw, pitch and roll all show in 3D.

**Export to Blender**

- `acoustic-spaces export` and the editor's File menu write a scene as glTF 2.0
  (`.glb`) for Blender: the surfaces with their materials (absorption and scattering as
  custom properties), sources and listeners facing their way, the early reflection
  paths by order, and sound particles as an animation in slow motion.

**Source directivity**

- Directivity per octave band: a pattern name or a first-order `a` for each band, or
  measured data from a SOFA file (AES69 `FreeFieldDirectivityTF`, or impulse responses)
  or a CSV table of levels by azimuth and elevation, interpolated over the sphere.
- It applies to the direct sound, the early paths, the rays of the late part (in the
  compiled tracer) and the sound particles, in the source's own frame including roll.
- `acoustic-spaces directivity` prints a directivity's index and front-to-back ratio
  per band and draws its polar patterns (`--plot`); `inspect` lists each source's
  index. The editor picks a pattern or a file, shows the index and the polar patterns,
  and draws the selected source's balloon in 3D. The control room example's monitors
  narrow from omni in the bass to supercardioid at 8 kHz.
- `acoustic_spaces.directivity` has the patterns, balloons and file readers;
  `Room.directivities` holds each source's.

**Examples**

- New example rooms: a concert hall, a control room and a church.
- The examples ship inside the package. `acoustic-spaces examples` lists them and
  copies them (`--copy DIR`); `acoustic_spaces.examples` finds them from Python.

**Other changes**

- `uv sync` installs the test tools (a `dev` dependency group: pytest, h5py, ruff), so
  `uv run pytest` works in a fresh checkout, and rebuilds the compiled core when the C++
  sources change. The README shows the uv and pip setups side by side, and messages about
  a missing extra name `uv sync --extra ...` as well as pip.
- `inspect` adds the Arau-Puchades reverberation time, which allows for absorption that
  faces one way (also `Room.reverberation_time("arau_puchades")`), and a note when the
  absorption is too uneven for Sabine and Eyring to hold, as in the example shoebox,
  whose simulated T30 is several times their estimates.
- Tests check the decay of a box with specular walls against its closed form, with even
  and with uneven absorption, and the example's absorption with diffusing walls against
  Eyring.
- Saved scene files refer to model, material and HRTF files under their folder by
  relative paths, so the folder can move, and leave out unset fields.
- `acoustic_spaces.results.write_results` writes a simulation's responses and
  `parameters.json`, as `simulate` does.

## 1.0.0

The first release: everything the v1 plan set out, from a CAD model to a room you can
listen to.

**Scenes and geometry**

- Scene files in YAML with a checked schema: geometry, materials, sources (with
  directivity), listeners (with orientation), air, simulation settings and outputs.
- Geometry from OBJ (with material groups), STL, PLY, OFF, glTF/GLB and DXF (with
  layers), or built from dimensions: a box or a floor plan raised to a height.
- Mesh checks and cleanup: closed, consistently oriented, inside or outside, distance of
  sources and listeners to the walls.
- A built-in database of absorption and scattering coefficients per octave band, plus
  CSV coefficient files in any band layout.
- A 3D view of the room with reflection paths, and animations of sound particles saved
  as GIF or MP4.

**Simulation**

- Early reflections as exact specular paths, found by deterministic ray tracing or by
  image sources, then traced back from the listener.
- A stochastic late tail by ray tracing (diffuse rain, secondary sources or a detector
  sphere) with weight-window splitting, or a statistical tail from Eyring and Barron.
- A compiled core (C++, nanobind) with a BVH; simulations run on every core.
- Impulse responses per octave band, ISO 3382-1 parameters (EDT, T20, T30, C50, C80, D50,
  Ts, G) and ISO 9613-1 air absorption.

**Spatial outputs**

- Binaural (HRTFs from SOFA files or a spherical head model), Ambisonics (AmbiX, orders
  0 to 7) and loudspeaker layouts by VBAP, from stereo to 7.1.4 (silent LFE channel).

**Listening**

- Offline convolution and auralisation of dry recordings, with resampling.
- A real-time convolver in C++ with non-uniform partitions, and playback or live input
  through the sound card (`sounddevice`).

**Measurement**

- Exponential sine sweeps, playback and recording, deconvolution with a loopback channel,
  and the same parameters from measured impulse responses (noise compensated).

**Validation and release**

- A validation report: the automated checks against theory, a cross-check against
  pyroomacoustics in four rooms, and a benchmark of the real-time convolver
  (`docs/validation.md`, scripts in `validation/`).
- Documentation built with Sphinx; wheels for Linux, macOS and Windows on CPython 3.10
  to 3.14; tagged releases publish to GitHub releases and, once set up, to PyPI.
