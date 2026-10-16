# Changelog

## 3.0.0

Validation against two measured rooms, third-octave bands, vector-based scattering,
sources with their own spectra and line, surface and array sources, sound through walls
and between coupled rooms, fitting materials to measurements, a GPU path for the wave
solver, edge diffraction over two edges, early reflections in pressure at low
frequencies, IFC and Collada models, material libraries, a leak finder and walkthroughs.

Simulations scatter by vector and by the size of each surface, and trace rays of their
own for each band, so rooms with smooth walls decay faster than in 2.0.0 and closer to
measurements (see **Scattering** for version 2's settings). Early reflections add in
pressure at 63 and 125 Hz, and the wave solver's walls absorb as their materials do in
every band, which changes results in those bands. Scene files of 2.0.0 load unchanged.

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
  simulations with them; the validation report has the results. It prints how far each
  room has got on standard error.
- The cross-check against pyroomacoustics averages five runs of pyroomacoustics per room
  (`--pra-runs`), which draws its rays afresh on every run, and shows the spread between
  them. One run had made its C80 in the lecture hall look up to 3.5 dB lower than this
  package's; the mean of five is 0.3 to 1.7 dB lower.

**Scattering** (changes results: see below for version 2's)

- Vector-based scattering, as in ODEON: every reflection sends a ray in the direction of
  the sum of its specular direction and a random Lambert direction, weighted by the
  surface's scattering coefficient, instead of specularly or, with the coefficient as
  probability, diffusely. The diffuse rain sends the scattered share of each reflection
  to the listeners and the detector spheres count only the rest, so nothing is counted
  twice. Where the bands' coefficients differ, each band is traced with rays of its own,
  so a wall can scatter the bass and reflect the treble. Particles in `animate` and the
  editor move the same way.
- Scattering from the size of a surface (Christensen and Rindel, 2005, as in ODEON): a
  surface that is small for the wavelength, far from where the sound came from or met at
  a grazing angle scatters more than its coefficient says. Each flat surface (the faces
  of one plane joined at their edges, of any material) counts as the rectangle with its
  area and perimeter. Early reflections lose that share too, and the late part takes it
  up. Reflections that edge diffraction already completes keep their own coefficient:
  the first two orders with `first_order` diffraction (the default), the first with
  `direct`.
- `simulation.scattering` picks the model: `method: vector` (default) or `random`,
  `from_size: true` (default) or `false`, and `by_band` (below); the editor's simulation
  page has all three.
- `simulation.late.rays` (and `coverage.rays`) now count the rays per octave band, 50,000
  by default instead of 100,000 shared by all bands. The church, concert hall and
  orchestra examples trace 25,000 per band. A simulation takes about 1.5 to 4 times as
  long as before; `simulation.scattering.by_band: false` turns one ray for all bands by
  their mean scattering instead, which is faster but scatters the treble like the bass.
- Rooms with smooth walls decay faster than with version 2: in the measured chamber
  music hall of BRAS the reverberation time came out 20 to 32 % long and now 7 to 20 %.
  The shoebox example's T30 at 1 kHz drops from 2.4 s to 1.3 s. Scene files need no
  change; to get version 2's results, set
  `simulation: {scattering: {method: random, from_size: false}, late: {rays: 100000}}`.

**Third-octave bands**

- `simulation.bands: third_octave` runs a simulation in the 24 third-octave bands from
  50 Hz to 10 kHz instead of the eight octave bands: materials, air absorption, source
  directivity, the scattering of surface size, the rays (`rays` per band), the wave
  solver's bands and the impulse response's filter bank all follow them. It traces
  three times the rays and takes about three times as long. `simulate --bands` and the
  editor's simulation page set it.
- Materials take 8 or 24 absorption and scattering values. An octave value holds for
  its three third octaves; an octave run takes the mean of a material's three.
- Coefficient files with more than one value in some octave (third-octave tables, as in
  BRAS) keep their third octaves instead of being averaged onto octaves, so such a
  material's `absorption` now has 24 values; `Material.coefficients(bands)` gives them
  in either bands. Octave runs average them as before. A band with no data now copies
  the nearest third octave rather than the nearest octave: a table that starts at
  100 Hz gives the 63 Hz octave its 100 Hz value instead of the 125 Hz octave's mean.
- Room parameters, the STI, coverage maps and plots stay in octave bands, each octave
  the mean of its three thirds' energies. `Response.energy()` has the 24 bands and
  `Response.octave_energy()` the octaves; `Response.band_hz` names them.
- The editor shows a material with third-octave data in 24 columns, and `inspect` its
  octave averages.
- In the BRAS seminar room, third-octave bands bring the reverberation time at 250 Hz
  and 8 kHz from 13 and 17 % short to within 4 % of the measurement; the validation
  report has both rooms (`bras_validation.py --bands third_octave`).

**Sources**

- `level_db` takes the sound power per band (8 or 24 values) as well as in all, and
  `spectrum` spreads a level in all evenly (`flat`, as before) or as male or female
  speech (IEC 60268-16). The STI takes the female weighting for `speech_female`
  sources, in responses and coverage maps; `speech_transmission_index` has a `talker`.
- `gain_db`, `eq_db` (per band) and `delay_ms` feed a source: the gain and equalizer act
  on its sound power and its impulse responses, the delay on every arrival, and the
  coverage map of all sources together takes each source's delay. G does not change
  with the feed (`Response.fed` applies one to a response).
- Sound pressure levels: `simulate` prints them and writes `spl_db` and `spl_a_db` for
  each response, `Response.levels` gives them for any sound power, the editor's results
  have an SPL row, and `inspect` lists each source's sound power per band. The editor's
  source page sets the spectrum, gain, equalizer and delay.
- Line and surface sources: `line_m` or `surface_m` (and `spacing_m`) make a source of
  uncorrelated point sources that share its power and rays, in responses, coverage
  maps, early paths and particles. A `lambert` directivity radiates as a vibrating
  surface. The editor sets the shape and size, and draws them in the 3D view and the
  plan; the Blender export and `inspect` show them; `directivity lambert` prints the
  pattern's index.
- Coherent arrays: `directivity: {array: ...}`, or a YAML file with the same fields,
  gives a source the far-field pattern of elements fed alike, with their positions,
  aims, gains, delays and polarities, a column's count and spacing, and electronic
  steering (`steer_deg`). The energy is the exact mean over each third-octave band, with
  the room's speed of sound (`array_directivity`, `read_array`). The editor and
  `acoustic-spaces directivity` take array files.
- The pattern is the array's far field, from about its length squared over the
  wavelength. The room warns about every listener nearer than that at 1 kHz, naming
  the array and the distance (`array_length`, `far_field_distance`).

**Walls, doors and openings**

- Materials take `transmission` (the share that passes, 1 for an opening) or
  `transmission_loss` (the sound reduction index R in dB), one value or one per band.
  What passes counts as absorbed on its side, so the absorption must be at least the
  transmission; without absorption values it is the transmission.
- A model can hold several rooms, found from the faces themselves: every space they
  close off, whichever way the normals point. Faces between two rooms, and panels
  standing in a room, reflect and absorb on both sides. The inside of a thick wall or a
  column is solid unless a source or listener stands in it. `inspect`, `validate` and
  the editor list the rooms; `Room.room_volumes`, `Room.room_of` and `Room.face_room`
  give them to scripts.
- Early reflections pass through partitions with the share that passes, and through
  openings unchanged; late rays go on behind a partition as a copy carrying that share,
  with a weight window per room from the rooms' diffuse power balance. Coverage maps
  cover every room. The statistical late method follows coupled-room theory room by
  room; the statistical reverberation times count both sides of the faces between
  rooms.
- Openings are holes for diffraction (the edges of a doorway diffract), for the wave
  solver's air and for particles; particles pass partitions as often as the sound does.
  The wave solver refuses walls that let part of the sound through.
- A warning for listeners in a room no surface joins to the source's.
- In the editor, a material's **Lets through** (nothing, part of the sound with R per
  band in the table, or everything) and the rooms under the scene's Properties.
- Checked against diffuse-field theory: the level behind 30 and 50 dB partitions within
  0.3 dB, an opening the size of the wall the same as one room, and the double slope of
  coupled rooms within 1 dB of the two-room model; the statistical method and coverage
  maps across rooms against the same model, the rays and listeners.

**Audience seating**

- `audience: true` on a material: sound that reaches a listener low over it loses more
  in the 500 Hz to 2 kHz bands, after the 2025 fit of Jurkiewicz et al. to scale-model
  measurements (`acoustic_spaces.engine.grazing`). It applies to the last leg of the
  early paths to listeners and map points over the seats. The editor's **Audience
  seating** sets it, and `inspect` says so.

**Fitting materials to measurements** (`acoustic-spaces calibrate`, `acoustic_spaces.calibration`)

- The absorption of a scene's materials fitted, octave band by octave band, to the T30,
  EDT and C80 (or other parameters) measured at its source and listener pairs, in JNDs:
  from impulse response WAVs, `analyze --json` reports or a JSON of parameters. It
  prints the fitted absorption, what it started from and how far the measurements leave
  each value open, with the errors before and after, and writes the fitted scene.
- The materials keep their proportions unless the measurements tell them apart: a
  material moving by a factor of 2 more than the others costs one JND.
- Each round traces the room once, and the fit re-weights the rays traced for other
  absorption (`acoustic_spaces.engine.reweight`): the tracer counts each ray's
  reflections off each of up to 16 groups of faces (`simulate(..., groups=...)`,
  `Response.late_moments`, `Response.reweighted`).
- In the editor, changing a material's absorption after a run updates the results at
  once by re-weighting that run's rays (`reweight.reweighted_result`); the results
  panel says so, and **Run** traces afresh.
- Checked on the simulator's own responses: absorption is recovered within 0.05, and a
  shoebox's walls within 0.015 from 500 Hz up. Also checked on the two BRAS rooms
  (`bras_validation.py --calibrate`), fitted to one loudspeaker position from BRAS's
  initial absorption estimates. On the other position's responses, both rooms come to
  about one JND. That is as close as BRAS's own fitted absorption gets in the seminar
  room, and closer in the hall: T30 within 0.5 JNDs against 1.8.

**Wave solver: walls that follow their materials, and open models** (changes results in
the wave bands)

- Each material becomes a locally reacting surface whose admittance changes with
  frequency: a constant and resonant mass-spring-damper branches, fitted so that its
  random-incidence absorption, counted through the analysis's band filters, matches the
  material's in every wave band (`acoustic_spaces.engine.impedance`). One run covers
  all the wave bands, where version 2 ran one for each set of coefficients, and the
  absorption changes smoothly from band to band instead of stepping. The largest miss of
  the fit is in the details as `wave_absorption_miss`; `wave_runs` is gone.
- Round an open model the grid grows by a perfectly matched layer of 8 cells, which
  takes in the sound leaving it: what comes back is 57 to 74 dB below the sound, where
  the grid's edge alone sent back 7 to 14 dB below it.
- `WaveResponse.signals` holds the one run (samples, 4), and `WaveResponse.bands` is a
  field; `band_runs` is gone. `_native.WaveSolver` takes each face's material, the
  materials' constant admittances and their branches, and the matched layer.
- Checked in `validation/wave_solver.py`: modes of a box with frequency-dependent walls
  (frequencies within 0.25 % of perturbation theory, decay rates within 2 %), the
  normal-incidence absorption of a duct's end (within 0.004 of the grid's theory), the
  matched layer against a grid too large for anything to come back, and the run count.

**Wave solver on a GPU** (`simulation.wave.device`, `acoustic_spaces.engine.gpu`)

- `device: gpu` steps the wave solver's grid on a GPU through WebGPU (wgpu: Vulkan,
  Metal or Direct3D 12, on any vendor's GPU), with the new `gpu` extra; `auto` takes a
  hardware GPU when there is one. The C++ core still lays the grid out, and one WGSL
  compute shader steps it, one dispatch per time step; the runs agree with the CPU's to
  within about 1e-5 of the largest pressure. `cpu` stays the default.
- `simulate --wave-device`, a choice on the editor's Simulation page, the GPU's name in
  `acoustic-spaces info` and in the progress line, and `wave_gpu` and `wave_steps` in
  the details.
- `validation/wave_gpu.py` times a GPU against the CPU on the example rooms. CI runs
  the comparisons on Mesa's software Vulkan driver on Linux.

**Wave solver: less dispersion, and faster** (`simulation.wave.scheme`; changes results
in the wave bands of closed rooms)

- A second stencil, the interpolated isotropic one: 19 points (the six axial neighbours
  and the twelve edge neighbours) at its Courant limit, √(3/4). Waves on it are 0.67 %
  slow at 8 cells per wavelength, within 0.03 % the same in every direction, where the
  standard 7-point stencil is up to 1.74 % slow along the grid's axes. `auto`, the
  default, takes it in closed rooms and the standard one round open models, since only
  that one has the matched layer; `isotropic` with an open model is an error.
- Edge links that reach behind a wall are folded onto the axial link to the far end's
  image in the wall, so a plane wall reflects as it does with the 7-point stencil and
  the cells along it are as stiff across it as along it. A cut link's losses go to the
  wall it crosses.
- The CPU update streams through runs of interior cells, which the compiler
  vectorises: 1.7 times as fast with the 7-point stencil, 2.4 times with the 19-point
  one, with the same results to the bit. The grid's set-up casts rays along the links
  only from cells near a face, about 3 times as fast. The concert hall's wave bands
  take about as long with the 19-point stencil as they did with the 7-point one.
- `_native.WaveSolver` takes `stencil` (7 or 19) and reports it, the details have
  `wave_stencil`, and the GPU kernel steps both stencils.
- Checked in `validation/wave_solver.py`: the speed of sound along an axis and along a
  diagonal from the modes of a rigid box, within 2e-5 of each stencil's dispersion
  relation, and every other check with both stencils.

**Edge diffraction over two edges** (`simulation.early.diffraction: second_order`)

- `second_order` adds to `first_order` the direct sound over two edges in turn: over a
  thick wall or barrier, whose two corners share its top, round both corners of a
  doorway's jamb, or over two screens in a row, where `first_order` finds nothing or
  only the way round one corner. Each source and listener get the shortest path over
  each pair of edges that see each other (`acoustic-spaces view --paths` draws it with
  its two corners).
- The wave of the first edge reaches the second as the UTD has it, with the second
  edge's distance parameter from the curvature of that wave where it gets there: near
  the first edge's boundaries it is still curved like the source's, which the textbook
  formula for parallel edges misses by up to 6 dB at low frequencies. Along a face the
  two edges share, the incident wave and its reflection off the face are one wave, and
  the second edge's coefficient is halved.
- Where the first edge's wave is lit at the second edge (or the second's at the
  listener), the path continues a path over one edge and takes its cross term with
  that path, as a path over one edge does with the specular path it borders: across the
  plane of a barrier's top the level changes by at most 0.8 dB at 63 Hz and 0.1 dB from
  1 kHz up, where `first_order` drops from the path over one corner to nothing.
- Checked against the exact Biot-Tolstoy-Medwin solution in
  `validation/edge_diffraction.py`: over barriers 0.3, 1 and 3 m thick within 0.35 dB
  from 500 Hz up and 2 dB at 63 Hz, and over two thin screens in a row within 0.75 dB
  from 500 Hz up and 2.7 dB at 63 Hz. The validation report has the tables.
- The early stage of a hall with 16 edges that diffract takes 170 ms per source and
  listener with `second_order`, 120 ms with `first_order` and 90 ms without
  diffraction. `first_order` stays the default.

**Early reflections in pressure at low frequencies** (`simulation.early.coherent_max_band_hz`;
changes results at 63 and 125 Hz)

- Up to the 125 Hz band by default (the 63 and 125 Hz octaves, or the six third octaves
  up to 160 Hz), the early paths add in pressure, not in energy. Each carries the phase
  of every reflection on its way, that of a locally reacting surface with the
  material's absorption (R = (cos θ - Y) / (cos θ + Y), with the admittance fitted as
  for the wave solver), and a path over an edge the whole complex UTD coefficient, whose
  terms then add to the specular paths as the theory has them. Above those bands each
  path keeps the phase it has at their top, so the bands join without a step.
- The echogram reads those bands' early energy from the paths' pressure, through the
  band filters of the analysis of measured responses; coverage maps and the material
  fit do the same. The parameters of a simulated impulse response and of its echogram
  now agree there: in a 7.3 x 5.1 x 3 m box, G within 0.01 dB and C50 within 0.05 dB,
  where adding the paths in energy left them up to 2.1 and 1.6 dB apart.
- Next to a hard floor the direct sound's pressure doubles at low frequencies (+6 dB
  where energy gave +3 dB), and an absorbing floor at grazing incidence takes from it;
  a loudspeaker a metre out from a wall loses bass at 63 Hz.
- Over edges, against the exact solution (`validation/edge_diffraction.py`), the levels
  at 63 and 125 Hz come within 0.05 dB over a thin screen (0.9 dB in energy) and within
  0.6 dB over two edges (2.7 dB), and step by at most 0.01 dB across a barrier's top
  (0.8 dB).
- The late part still adds in energy, and so do the uncorrelated points of a line or
  surface source. With the wave solver up to that band or higher, its bands take their
  place.
- `null` adds the paths in energy in every band, as before. `--coherent-band` sets it
  for one run of `simulate` or `map`, and the editor's simulation page has it ("In
  pressure up to").
- `EarlyReflections.spectrum` holds each path's pressure at the frequencies the edge
  diffraction is evaluated at, relative to its level and delay;
  `acoustic_spaces.engine.coherent` has the rest.

**Leak finder** (`validate --leaks`, `view --leaks`, `acoustic_spaces.engine.leaks`)

- Rays traced from the sources with diffuse reflections show where a model that is not
  closed lets them escape. Each gap is reported with the rays that escaped through it,
  its area and size from the open edges round it, where the rays crossed it and the
  surfaces next to it. Openings to the outside (surfaces that let sound through) are
  told apart from leaks, and a source outside a closed model is named.
- `view --leaks` and the editor's **View > Show leaks** draw the open edges, the
  escaping rays and a marker on each gap (`plot_leaks` in `acoustic_spaces.viz`); the
  warning about a model that is not closed says how to find its holes.

**Material libraries of CATT-Acoustic and ODEON**

- A material takes its coefficients from an entry of a CATT-Acoustic file (`ABS`
  lines, in a geometry or library file, with scattering) or an ODEON library
  (`Material.Li8`): `file` names the library and `entry` the material, by name or by
  ODEON's number. `acoustic-spaces materials --file` lists a library's materials, and
  the editor's **Edit > Add material from a library** adds one.
- ODEON's own `Material.Li8` is read as ODEON writes it (checked against the library
  from odeon.dk, 198 materials): the product name of `{Product name*=...}` names a
  material and `{Category=...}` groups the listing, and materials from ODEON's Material
  calculator, with its settings and third-octave values on their lines, are read by
  their octave values.

**IFC and Collada models**

- IFC files (BIM models from Revit, ArchiCAD and others) with the `ifc` extra
  (IfcOpenShell, LGPL): every building element with its openings cut, or, with
  `geometry.space`, one room from its space boundaries, a closed room with a face per
  surface (from the space's solid where the file has no boundaries). Faces are named by
  their element's name, type, materials and IFC class, and the file's units are used.
  The editor's geometry panel picks the room.
- Collada (`.dae`, SketchUp's export among others), read directly: triangles, polylists
  and polygons, nodes with their transformations and instanced nodes (SketchUp's
  components), labelled by material and geometry name, in the unit and with the up axis
  the file records.
- DXF polygon meshes and closed 3D polylines are read too.
- Assignments and material names match surface names whatever their case.
- `acoustic-spaces examples` and the editor's examples menu say which extra an example
  needs when it is not installed.

**Walkthroughs** (`acoustic-spaces walkthrough`, `acoustic_spaces.walkthrough`)

- A listener or a point source walks through the room along the corners of its `walk`
  in the scene, at a walking pace, facing the way it goes and turning through corners
  at 90 degrees per second (`turn_deg_s`). Responses are simulated
  every 25 cm along the way; the late part, which changes slowly, is traced every 2 m
  and mixed in between, so the gallery's walk of 101 places costs six times one
  response.
- Dry signals play through the response of the place the walker is nearest to, with
  50 ms crossfades from each place to the next, in any of the output formats.
  `--video` saves the room and the walker going along its path (with an arrow and
  its yaw for the way it faces) as a GIF or MP4, in time with the sound (`animate_walk` in `acoustic_spaces.viz`); an MP4 carries the
  sound itself when it has one or two channels (mono, stereo, binaural).
- The viz extra installs imageio and imageio-ffmpeg, which MP4 videos need (for
  `animate` too); before, they had to be installed by hand.
- `simulate(..., late_from=...)` gives listeners or sources the late part of others,
  mixed, which is how walks trace fewer late parts.

**Examples**

- `gallery_walk`: a visitor walks away from the guide in the L-shaped gallery and round
  the corner while the guide talks.
- `bim_office`: an office from an IFC model in millimetres, with a window and a door,
  read as one room from its space boundaries; `make_model.py` writes the model.
- `rehearsal_room`: a rehearsal room under a pitched roof from a Collada file in inches,
  as SketchUp exports it; `make_model.py` writes the model.
- `material_libraries`: a seminar room with materials from a CATT-Acoustic file and an
  ODEON library.
- `leaky_classroom`: a classroom model with a missing window pane and a ceiling 3 cm
  short of a wall, for the leak finder.
- `listening_room`: a living room with a loudspeaker against the front wall or a metre
  out from it. `bass.py` prints G at the sofa in the lowest bands from both, with the
  early paths in pressure and in energy: 5.7 dB apart at 63 Hz in the early part in
  pressure, within 0.7 dB in energy.
- `archway`: a bar and a dining room joined by an archway in a wall 40 cm thick,
  with `second_order` diffraction: a diner behind the wall hears the singer over both
  corners of the arch's jamb.
- `town_hall_pa`: a sound system for speech in a reverberant town hall, two columns
  (coherent arrays) beside the stage and two delay loudspeakers with their own gain,
  equalizer and delay, over seats marked as audience. `delays.py` maps the STI with and
  without the delays: the back rows' median goes from 0.52 to 0.59, the front's and the
  middle's down by 0.04.
- `workshop`: noise from a conveyor (a line source), a machine's casing (a surface
  source) and a compressor, each with its own sound power spectrum, mapped as the
  A-weighted level over the floor.
- `apartment`: a living room and a bedroom with a wall and a door between them; the
  television on the sofa at 73 dB(A) is 45 dB(A) in bed with the door closed, 63 dB(A)
  with it open.
- `room_modes`: the bass of a small bare room from the wave solver, in third-octave
  bands, on the CPU or a GPU, against rays alone.
- `fitting`: a seminar room that measured longer than designed, as the absorbing panels
  of its back wall were left out. `make_measurements.py` makes up the measurements by
  simulating the room as built; `compare` shows the gap, and `calibrate --materials
  panels` finds the plain wall from 500 Hz up.
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
- The volume and the inside checks no longer depend on which way the normals point: a
  model with flipped faces has its volume, and points near a flipped face count as
  inside. A model counts as closed when every face has a room on at least one side.

**Head tracking**

- Turning the sound field is exact and about three times faster (0.3 ms instead of
  0.8 ms per orientation at seventh order): each order's rotation follows from the one
  below by Ivanic and Ruedenberg's recursion instead of a least-squares fit on a grid of
  directions.

**Faster ray tracing, and performance settings** (`simulation.performance`)

- Ray queries are 7 to 8 times faster, and the late stage with them: the model's
  bounding volume hierarchy is split by the surface area heuristic instead of at the
  median, walked nearest box first, and its box test no longer calls the maths
  library. One band of 50,000 rays in BRAS's chamber music hall takes 5.7 s instead of
  41 s on four cores. Rays meet the same surfaces at the same distances; where a ray
  meets two faces at exactly the same point (a shared edge), it may now take the other.
- `simulation.performance.threads` (default 0: every core) and `ray_tree` (`sah`, the
  default, or `median` for comparison) in scenes, under Performance on the editor's
  Simulation page, and as `--threads` and `--ray-tree` for `simulate`, `map` and
  `calibrate` (`--threads` for `view` and `export`). Without `--threads` they take the
  scene's setting; `info` prints the number of CPU cores.
- The ray searches run on a GPU too, through WebGPU (`gpu` extra): the first part of
  tracing rays there. `validation/rays_gpu.py` times them against the CPU on rays like
  the late tracer's and checks that both find the same faces. `Bvh.layout()` hands the
  tree to it, and `Bvh.occluded` tests many rays at once on the CPU.
- `validation/wave_gpu.py` times the grid's set-up and the time steps apart (the
  speed-up is that of the steps), starts the GPU before timing, prints its settings
  and each grid's cell size and stencil, and lines up its table in a terminal.
- The late stage's rays can be traced on a GPU (`simulation.late.device`: `cpu`, the
  default, `gpu`, `both` or `auto`; `acoustic_spaces.engine.gpu_tracer` and
  `ray_device`). One WGSL compute shader traces what the CPU does, in single precision
  and with random numbers of its own, so the two agree statistically. `both` shares the
  rays between the CPU and the GPU as each is free; `auto` times a short trial on both
  and keeps the fastest, or the CPU when wgpu finds no hardware GPU or nothing is
  clearly faster, and saves its choice for the computer. Sound through partitions and
  calibration's face groups stay on the CPU. `simulate --late-device`, "Run on" on the
  editor's Simulation page, and `--no-device-trial` (`simulate`, `edit`) to take the
  saved choice without a trial. `validation/late_gpu.py` times the late stage on each
  device and compares the parameters it gives.
- Tests hold the ray tracer to results stored from main before these changes
  (`tests/reference/tracer.npz`): the trees' faces and distances, specular paths, and
  every kind of late run on the CPU (to single precision), and statistically on a GPU.

**Fixes**

- The late stage's count of rays that left an open model (`escaped_rays` in the run
  details) was always 0: a ray whose search found no face within the distance left to
  it was taken for one that had run out of time. A ray that finds none is now checked
  for a face anywhere ahead, on the CPU and on the GPU.
- Runs that split rays (the late stage's weight window, sound passing both ways
  through a partition) came out differently when the core was built with Clang, as on
  macOS, than with GCC or MSVC: the two numbers that seed a copy's random stream were
  drawn in an order the compiler chose. They are now drawn in the order GCC and MSVC
  used, so Linux and Windows give the same results as before and macOS gives those too.
- The sound diffracted by the free edge of a thin screen left out the term of the
  reflection off the screen's back wherever the reflection off its front reached the
  listener: up to 0.9 dB too little there, at high frequencies.
- The hint for a missing extra names the packages to install. It said
  `pip install "acoustic-spaces[extra]"`, which looks for the package on PyPI, where it
  is not.
- Arrays, whose patterns are worked out in third octaves, were shown band by band as if
  those were octaves: `inspect`, `directivity` and the editor listed 24 directivity
  indices under the 8 octave bands, the polar plots drew the lowest eight third octaves
  labelled as octaves, and the 3D view's balloon showed 125 Hz for 1 kHz. They show
  octaves now. Simulations were not affected.
- The editor's Stop button for playback was easy to miss: it sat at the end of the
  Results panel's bar, out of sight once another panel was on show, and the toolbar's
  Cancel, which only cancels a run, had the same stop icon. While a recording plays, the
  status bar now says so, with **Stop playback**, which is also in the toolbar and the
  Simulation menu; Cancel has an icon of its own, and Esc stops playback when nothing is
  being placed or drawn. Moving a source or listener while the sound played looked as if
  it changed what plays, which it does not: the status bar marks the sound as from
  before the latest changes, and the log says how to hear them.
- Picking another source or listener, or ticking Head tracking, while a recording played
  in the editor left the sound as it was. The playback now goes on from the same point
  with the response picked (the run has them all), at the same level, so the
  difference between listeners stays audible (lowered only if it would clip).
- The editor's Listen played mono when the scene wrote no binaural output, while
  head-tracked listening is always binaural, so the two sounded nothing alike (the
  reverberation above all), and each set its own level. Listen plays binaurally now,
  through the spherical head unless the scene names an HRTF, and both play the dry
  signal at half its level through the response's omnidirectional energy (Listen
  lowered only where it would clip), so ticking Head tracking changes only how the
  head turns the sound.

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
