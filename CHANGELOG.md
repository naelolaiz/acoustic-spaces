# Changelog

## Unreleased

**Comparing with measurements** (`acoustic-spaces compare`, `acoustic_spaces.compare`)

- A simulated and a measured impulse response, each at its own sample rate, compared by
  their ISO 3382-1 parameters band by band: both values, the difference, and the
  difference in just noticeable differences (ISO 3382-1:2009, table A.1), marked where
  it is more than one; per parameter, the bands within one JND and the mean absolute
  difference from 125 Hz to 4 kHz.
- Bands that cannot be compared (a decay of less than 20 dB, a decay time the noise
  leaves out, G without a reference on both sides) are listed with the reason, not
  counted as zero.
- A channel and a reference for G per file, `--json`, and `--plot` with the two decays
  in each octave band (`plot_comparison` in `acoustic_spaces.viz`).

**More parameters**

- LF, LFC and LJ (ISO 3382-1, annex A) from the directions the energy arrives from,
  with a figure of eight whose null points at the source; IACC early and late
  (annex B) from the binaural response when one is written; stage support ST early and
  ST late (annex C) for listeners within 1.5 m of a source; and the T30 of figures of
  eight facing forward, left and up from the listener.
- STIPA next to the STI (IEC 60268-16), U50, and the echo criteria of Dietsch and Kraak
  for speech and music. `parameters.json` has them for every response, at the source's
  level and the coverage noise, and `simulate` prints them.
- `analyze --lateral CH`, `--ambisonic` (first-order AmbiX) and `--binaural` give the
  same from measured responses, and `--stage` the stage support; `analyze` prints the
  STI, STIPA and echo criteria of each channel. `compare` takes LF, LFC and the
  directional decay times where both responses have them.
- The editor's results table shows the further parameters where a response has them.

**Validation against measured rooms**

- Two rooms of the BRAS benchmark, a seminar room and a chamber music hall, as scenes in
  `validation/bras`, with their surface data and the parameters of their measured
  responses (CC BY-SA 4.0, like BRAS). `validation/bras_validation.py` compares
  simulations with them; the validation report has the results.

**Examples**

- `orchestra`: an anechoic recording of Beethoven's 8th symphony (TU Berlin, CC BY 4.0)
  on the stage of the concert hall, one source for each of its 61 tracks, with scripts
  that mix a movement at a seat, draw pictures of the hall and the seating, and make a
  video of the mix. The README links the video made with them.
- Scripts in the example folders (`.py`, `.sh`) are installed with the scenes.

**SketchUp files** (experimental)

- `.skp` models read directly with the `sketchup` extra, through openskp (MIT), which
  needs no SketchUp: faces labelled by their material, matched by colour. The editor
  opens them too. OBJ exported from SketchUp stays the recommended path.

**Smoother binaural**

- HRTFs are interpolated between the measured directions instead of taking the nearest:
  barycentric weights on a triangulation of the measured directions (as in VBAP), with
  the responses aligned at their onsets and the mix delayed by the mixed onsets, to a
  fraction of a sample. Reflections between measured directions get an interaural time
  difference between theirs, so moving a source or the listener's head in a scene no
  longer makes it jump in steps. A set measured round the horizon only works too.
- Delays in SOFA files (`Data.Delay`) keep their fractions of a sample.
- scipy is a new dependency (for the triangulation).

**Geometry**

- Where a face's edge runs past the corners of faces on its other side (a T-junction,
  which SketchUp stitches with faces of zero area), cleaning splits the edge at those
  corners, so the model stays closed.
- Rooms touching along an edge, with four faces on it, count as closed when the faces
  pair up. Edges shared by an odd number of faces still leave a model open, and the
  report and the warning count them.

**Head tracking**

- Turning the sound field is exact and about three times faster (0.3 ms instead of
  0.8 ms per orientation at seventh order): each order's rotation follows from the one
  below by Ivanic and Ruedenberg's recursion instead of a least-squares fit on a grid of
  directions.

**Fixes**

- The hint for a missing extra names the packages to install. It said
  `pip install "acoustic-spaces[extra]"`, which looks for the package on PyPI, where it
  is not.

## 2.0.0

Coverage maps, edge diffraction, drawing rooms in the editor, head tracking and a
wave-based solver for low frequencies.

Simulations now include edge diffraction by default, so rooms with convex corners or
thin screens sound and measure differently than in 1.1.0. Set
`simulation.early.diffraction: none` (or `--diffraction none`) for purely specular
early reflections as before. The wave solver is off unless switched on.

**Coverage maps** (`acoustic-spaces map`, the editor's Simulation > Run coverage map)

- Room parameters on a grid over the audience area: SPL and G, C50, C80, D50, EDT, T20,
  T30 and Ts per octave band, the A-weighted level and the speech transmission index
  (IEC 60268-16, with masking, reception threshold and background noise).
- Maps for each source and for all sources together, from the sources' sound power.
- Several levels of seating under one another (stalls and balcony); columns along a
  slanted wall keep the part over the seating.
- The late part is one ray tracing run per source with a track-length estimate through
  the slab of air at each point, so a map costs about as much as a few listeners.
- A top view with the values (`plot_map`, `--plot`), coloured tiles in the 3D view
  (`plot_room(coverage=...)`, `--screenshot`) and every value as JSON (`--json`).
- The scene's `coverage:` settings (surfaces, height, spacing, rays, noise), with a form
  in the editor; the concert hall, lecture hall and church examples set them.

**Edge diffraction** (`simulation.early.diffraction`, `--diffraction`)

- Sound bends round edges with the uniform theory of diffraction (UTD), averaged over
  each octave band: the direct sound and first-order reflections (`first_order`, the
  default), the direct sound only (`direct`), or none.
- Edges are found in the model: convex corners with more than 195° of open air and the
  free edges of thin screens; collinear pieces of one edge are joined.
- In the shadow a diffracted path carries the sound; on the lit side the specular path
  next to the boundary takes the interference, so levels stay continuous across shadow
  and reflection boundaries.
- Simulations, coverage maps, `view --paths`, the Blender export and the editor (a
  Diffraction choice on the Simulation page) all use it; diffracted paths are drawn in
  their own colour and counted in the results.
- A new example, `gallery`: an L-shaped room with a listener round the corner.

**Drawing rooms in the editor**

- A Plan tab next to the 3D view: the room from above on a grid, with every corner of a
  floor plan or box room as a handle. Drag a corner to move it (a box becomes a floor
  plan with the same walls), double-click a wall to add one, right-click to remove one.
- Draw room (and File > New room from a floor plan): click the corners of a new floor
  plan. Sources and listeners left outside move inside; over a model file the editor
  asks first.
- Sources and listeners move by dragging, in the plan and in the 3D view, keeping their
  height above the floor; Shift moves them up and down in the 3D view.
- Snapping to a grid (Shift for none), wall lengths while dragging, and refused moves
  when walls would cross.
- `plan_problem` in `acoustic_spaces.geometry`: floor plans whose walls cross, touch or
  run back on themselves are refused with the reason.

**Head tracking** (`play --head-tracking`, the editor's Head tracking controls)

- Binaural playback of an Ambisonics response that turns with the listener's head, in
  real time: the sound field is turned once per block, crossfading from the previous
  orientation, then decoded to the ears; the latency is still one block.
- Head trackers send OSC over UDP: `/ypr`, `/quaternion` (`/quaternions`, `/quat`),
  `/yaw`, `/pitch` or `/roll` under any prefix; `--invert` for trackers that count an
  axis the other way round, and a centre to take as straight ahead.
- In the editor, the Results panel's Head tracking controls turn the head by hand or
  from a tracker, and Listen plays head-tracked.
- `binaural_decoder` in `acoustic_spaces.spatial.decoder`: MagLS (or least squares)
  filters from Ambisonics of any order to the ears, from any HRTF.
- `FieldRotation` and rotation helpers in `acoustic_spaces.spatial.rotation`: matrices
  that turn Ambisonics sound fields of any order.
- `render.play` takes any block processor, such as `HeadTrackedBinaural`.

**Wave-based solver for low frequencies** (`simulation.wave`, `simulate --wave`)

- The lowest octave bands come from the sound field itself instead of rays: finite
  differences in time on a cubic grid over the room (the standard leapfrog scheme at
  its stability limit), in the compiled core, on all cores.
- Room modes and sound bending round everything at low frequencies; the response
  joins the ray-traced one through the synthesis's crossovers, from 22 Hz up to the top
  of the highest wave band (`max_band_hz`: 63, 125 by default, 250 or 500).
- Surfaces absorb as their materials do in each band (locally reacting, with the
  admittance whose random-incidence absorption is the material's); bands with
  different coefficients get runs of their own on the same grid.
- Closed rooms, open models (the air at the edge of the grid absorbs) and thin
  screens; a listener outside the source's air is refused with its name.
- First-order directions from the pressure gradient round the listener, so
  Ambisonics, binaural and loudspeaker outputs keep the low end's direction.
- The editor's Simulation page has a Wave solver switch and its highest band;
  `simulate --wave/--no-wave` and `--wave-band` override the scene.
- Checked against theory: the modes of a box and their decay rates (within 0.5 %),
  the reflection off a duct end and the attenuation along absorbing duct walls, the
  free-field level (within 0.35 dB) and a box's modal response
  (`validation/wave_solver.py`).

**Also**

- `speech_transmission_index` and `modulation_transfer` in `acoustic_spaces.parameters`.
- Parameters from energy responses are computed about twice as fast.

A code review before the release fixed 38 bugs in the simulation, the parameters, the
wave solver, the spatial outputs, the geometry import, the command line and the editor.
Against 1.1.0, G with the statistical late method rises by up to 1.8 dB, and a response
that decays less than 20 dB has no parameters instead of wrong ones.

A scene's `geometry.units` now defaults to the units a DXF file records, and to metres
for other files, as before.

**Fixes: simulation and parameters**

- The statistical late method keeps Barron's total energy: its tail starts at the
  direct sound and takes off the early paths, instead of starting at the transition
  and losing the scattered share of the early reflections. Coverage maps use the same
  tail.
- A response whose decay has not reached -20 dB has every parameter NaN, as T20 and
  T30 already had, instead of values read from the estimated tail.
- The noise floor is looked for before trailing digital silence (zero padding, the
  zeros `remove_latency` leaves), which made T30 read up to 70 % long.
- Mono WAV columns, as `read_wav` gives them, work in `deconvolve` and the parameter
  analysis; an empty impulse response has NaN parameters.
- A run with one late ray, and `simulate` with no listener, give a result or a plain
  error instead of failing deep inside.
- Coverage maps no longer fail on points whose response ends within 100 ms of the
  direct sound.

**Fixes: diffraction**

- A specular path interferes with its own wave's diffraction terms only, which keeps
  the level continuous across reflection boundaries (it stepped by up to 2.1 dB).
- Paths that reflect off the floor or ceiling where an edge ends are kept once, not
  twice or never.
- A listener or map point on the source no longer divides by zero.

**Fixes: wave solver**

- The wave bands start at emission; they arrived one grid step early.
- The listener's velocity sits on the pressure's samples (it led by half a step).
- The rays' late noise joins the wave bands in energy, which removes a 3 dB dip at
  the crossover.
- The source adds no net air, which had left a static pressure and a floor about 50 dB
  down on the wave bands' decay.
- A wall thinner than a cell gives each side its own face.
- The grid runs at a multiple of a low output rate instead of with coarser cells, and
  listeners outside the grid's air are refused before the runs.

**Fixes: spatial audio**

- HRTFs keep their gain when resampled: a 96 kHz SOFA file used at 48 kHz played 6 dB
  quiet.
- SOFA files that turn the head for each measurement, or give the ears in spherical
  coordinates, are read with the right directions and ears.
- 5.0.4 and 5.1.4 keep arrivals from behind on the horizon, panned as in 5.0; they went
  up to the rear height speakers.
- A head path given to `HeadTrackedBinaural.run` is not inverted by the tracker's
  `invert`.
- Ambisonic encoding refuses directions of zero length instead of giving NaN.

**Fixes: geometry, scene files and the command line**

- Faces of a closed model that point the wrong way are turned to agree with their
  neighbours, so the inside test no longer puts sources outside the room.
- DXF files are read in the units their header records when the scene gives none, and
  3DFACE triangles no longer get a fourth corner at the origin.
- Polygons that repeat a corner are triangulated correctly: an L-shaped floor had 8 m²
  instead of 5 m².
- OBJ files with several groups on one line, lines continued with `\`, or coordinates
  that are not finite numbers are read or refused properly.
- A model whose faces all have no area is refused with a clear error.
- NaN and infinity are refused in scene files and coefficient CSV files.
- A listener within 1 cm of a source is reported as a problem instead of giving a NaN
  response.
- Command-line overrides are checked as the scene file is, and `--sample-rate` now
  reaches the wave solver. `analyze`, `deconvolve` and `sweep` report unreadable inputs
  as errors and create their output folders.

**Fixes: editor**

- Choosing a material for one wall assigns it to that wall, not to every surface that
  shares a name with it.
- The 3D view stays lit after it redraws the room, and keeps its camera while the room
  is edited.
- Typing in a position, orientation or box field changes only that value; the others
  were rounded to two decimals. Humidity shows a tenth of a percent.
- A custom loudspeaker list is shown as it is, without a 5.0 layout added on any change.
- Results of a run whose scene was closed are dropped, and saving an example elsewhere
  keeps its results current.
- One recording plays at a time.
- Box and floor-plan fields show the units they are in, and boxes may be larger than
  1000 units.
- Save As adds `.yaml` to a bare name, asking before it replaces a file, and opening
  another scene cancels placing a point in the 3D view.

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
