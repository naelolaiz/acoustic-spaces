# Roadmap

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

v3, version 3.0.0:

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
- [x] Audience seating: the attenuation of sound that skims the seats.
- [x] Fitting absorption to measured parameters (`acoustic-spaces calibrate`); in the
  editor, results follow a change of absorption at once.
- [x] Wave solver walls that absorb as their materials do in every band, in one run, and
  a matched layer round open models.
- [x] A GPU path for the wave solver through WebGPU (AMD, NVIDIA, Intel and Apple GPUs
  alike).
- [x] Ray tracing 7 to 8 times faster, and the late stage's rays on a GPU too, alone or
  sharing them with the CPU.
- [x] A wave solver stencil with less than half the dispersion, the same in every
  direction.
- [x] Second-order edge diffraction: the direct sound over two edges, checked against
  the exact solution.
- [x] Early reflections that add in pressure at low frequencies, with the phase of every
  reflection and of the sound over edges.
- [x] A leak finder: where rays escape from a model that is not closed.
- [x] Materials from CATT-Acoustic and ODEON material libraries.
- [x] IFC (BIM) and Collada import.
- [x] Walkthroughs with moving sources and listeners.
- [x] Binaural rendering with HRTFs interpolated between the measured directions.

Version 3.1.0:

- [x] Timing the CPU and the GPU on each computer and running on the faster one
  (`acoustic-spaces tune`).

Version 3.2.0:

- [x] On PyPI: `pip install acoustic-spaces`.

Version 3.2.1:

- [x] Wheels for Intel Macs.

Version 3.2.2:

- [x] Fix: the editor no longer aborts on looking for a GPU (wgpu's OpenGL backend).

- [x] Translation supported. Spanish ready on catalogs and editor.

Version 3.2.3:

- [x] Added Catalan translation.

- [x] Added Italian translation.

- [x] Added German translation.

- [x] Added Dutch translation.

v4, version 4.0.0: outdoor acoustics (ground, wind, temperature, turbulence, vegetation,
terrain and buildings from a map, weather classes and long-term levels, Doppler, whole
cities, on the GPU), reflections that fade at edges, coherent arrays, groups, movement
along keyframes with every mover on one clock, demos of any scene and of the examples,
HRTF sets to download, the ISO 3382-2 checks and packs to run on another computer, in
the sections below.

Edge diffraction:

- [x] Reflections that fade in and out at the edges that cut them off: every specular
  path, of any order, takes the uniform theory of diffraction's term of its own
  boundary at each edge it passes, checked against the exact solution past a screen's
  end, a building's corner, a strip and a gap (within 0.2 dB), and walked round the
  gallery's corner 1 cm at a time.
- [x] Paths that two events stop at once, as a reflection past the end of its face with
  an edge in its way: followed where one or two edges explain the events, so the
  street's garden no longer steps by up to 1.9 dB in one band (0.63 dB(A)) where the
  second event clears.
- [x] The terms of one event at different edges in the bands that add in energy
  (indoors from 250 Hz): they take their cross terms with each other, as one wave, so
  behind the `archway`'s wall the level steps by up to 1.4 dB where the reflection off
  the arch's jamb appears, instead of 4.4 dB.
- [ ] Reflections cut off by a thick end: behind the archway's wall the level steps by
  0.57 dB at 63 Hz where they go from hitting the jamb's end face to its far face, a
  face of the end's other corner. This and the fade's other known issues are listed in
  [Validation](validation.md#known-issues-of-the-edge-fade).
- [ ] Terms at the corners of edges, which the UTD has none for: each term weighted by the
  part of its Fresnel zone that the edge covers, as BTM's integral over a finite edge
  has it, and kept where its path appears round the edge's end. Past a thin screen's
  corner the edge diffraction steps by 2.8 dB at 63 Hz, and along the `street` the
  level by up to 1.42 dB in a band (0.55 dB(A)).
- [ ] Reflections on paths over two edges (`second_order`): behind a thick barrier on a
  hard ground the engine has one of the four paths over it, 1.7 to 5.7 dB low from
  500 Hz up.

Parameters:

- [x] The checks of ISO 3382-2 on every T20 and T30: the non-linearity ξ of the line it
  is read from and the curvature of the decay, amber in the editor past the standard's
  limits.

Outdoor acoustics, in steps:

- [x] O1 Outdoor scenes: flat ground under an open sky, buildings, barriers and model
  files on it; responses as long as the distances need; coverage maps over the ground.
- [x] O2 Ground impedance (grass, soil, snow) and the spherical-wave ground reflection;
  mixed ground in zones, weighted by the Fresnel zone; the early paths in pressure in
  every band outdoors; the wave solver fitted to the impedance.
- [x] O3 Wind and temperature gradients: curved rays and shadow zones; a parabolic
  equation for the sound over the ground below 500 Hz and in the shadows; validated
  against measurements of a wind turbine up to 845 m away.
- [x] O4 Atmospheric turbulence: the coherence of the paths (the ground dip filling
  in), sound scattered into shadows, and impulse responses, renders and walkthroughs
  that flicker as the eddies drift past; checked against a parabolic equation through
  random eddies.
- [x] O5 Vegetation: belts of trees, hedges and lone trees as volumes of foliage and
  stems that absorb and scatter sound, along the early paths and the late rays; checked
  against a separate particle simulation.
- [x] O6 The wave solver in wind and temperature: a speed of sound per cell (the
  effective one along the horizontal from the source) in the grid, its walls, its source
  and its matched layer; checked against the parabolic equation at 63 and 125 Hz. The
  flow itself (the linearised Euler equations) stays out (O18 adds moving air).
- [x] O7 Outdoor results: the level relative to free field and the insertion loss of
  buildings, barriers and vegetation (a second run without them), at listeners and on
  maps, in the report, on the command line and in the editor; checked against the
  exact solutions for a half-plane and for a ground.
- [x] O20 Sound reflected off the ground both before and after bending over an edge
  (the fourth of the four paths over a barrier on a hard ground):
  `simulation.early.diffraction_reflections: both_sides`, on the command line and in
  the editor; the barrier's insertion loss is then within 0.15 dB of the exact
  solution. `one_side` stays the default, so existing scenes give the same results.
- [x] O8 Foliage from the leaves' size and density: `foliage: {leaf_area_per_m3,
  leaf_width}` works out what the leaves scatter and absorb in each band, and how far
  forward, from thin rigid discs turned every way alike; checked against the disc's
  closed forms at both ends and a wall's boundary-layer absorption, the late rays
  through such leaves against the particle simulation, and a wood of them compared
  with ISO 9613-2's dense-foliage table. Presets and measured rates stay as they were.
- [x] O9 Terrain and buildings from a map: `ground.terrain` (heights on a grid from an
  Esri ASCII grid, x y z points or the scene, `ground.map_origin`) and
  `geometry.buildings_file` (GeoJSON footprints raised to a property's height, named,
  some left out); buildings, barriers, vegetation and zones stand on the terrain. The
  early sound reflects off the mean plane of the ground's profile (the standards'
  ground reflection) and bends over the crests the line of sight cuts through, one UTD
  wedge per crest with both slopes' reflections in its terms, the equivalent edge and
  thick-screen factor for a rounded top, and the nearest crest's wave in sight, so the
  level is continuous across the edge of a hill's shadow; the late rays bounce off the
  triangles. A ridge is within 0.2 dB of the exact wedge in every band, a flat terrain
  and a slope give the flat ground's and the image's paths, a smooth hill over soft
  ground lets 1.4 to 1.8 dB more past than the thin screen (its crest is a wedge open
  wider than a half-plane), over rigid ground the flanks' reflections add. Scenes
  without a terrain are unchanged.
- [x] O10 Outdoor scenes in the editor: **File > New outdoor scene** (a meadow to build
  on), buildings, barriers, vegetation (a belt, a hedge or a tree) and ground zones
  drawn in the plan from the Geometry form, the ground's extent set there, a terrain
  from a file and the buildings of a map (asking which properties hold their heights
  and names), each with a button to remove it; the plan shows the vegetation and the
  zones. Saved scenes refer to their terrain and map by relative paths.
- [x] O11 A fast field program and the 1995 benchmark cases: `validation/ffp.py`, the
  exact solution through layers of constant sound speed, within 0.001 dB of the exact
  ground solution in still air, and `validation/benchmark_1995.py`, the benchmark's
  still air and 0.1/s gradients down and up, and the same stopping 100 m up, at 10, 100
  and 1000 Hz to 10 km through the engine's parabolic equation and as whole scenes per
  band. With the gradient stopping 100 m up, as the weather's do, the parabolic
  equation is within 0.35 dB of the program to 1 km downwind and within 3 dB at 10 km
  at 100 Hz (11 dB at 1 kHz, 47 dB down), and parts from it beyond 1 to 2 km with the
  benchmark's gradient going on to a kilometre up (O21's wide-angle parabolic equation
  carries that); whole scenes are within 0.11 dB per band to 1 km downwind in every
  case and upwind with the benchmark's gradient, 1.5 dB at 1 km upwind with the
  gradient stopping 100 m up, 44 dB down. The comparison found and fixed a notch at the join of the
  parabolic equation's bands and the rays' (see the changelog).
- [x] O12 Outdoor scenes on the GPU: the GPU traces the rays bent by wind and
  temperature through the same chords as the CPU (ray for ray: the same turns and the
  same curve to the millimetre, in single precision without trigonometry), the rays
  through vegetation (foliage and stems, up to 16 media) and the open sky, and steps the
  wave solver's grid with its speed of sound per cell; `simulation.late.device` and
  `simulation.wave.device: gpu`, `both` and `auto` take outdoor scenes now. What
  reaches each listener in each band, and when, agrees within the noise of the ray
  count (`validation/outdoor_gpu.py`).
- [x] Bent rays and vegetation on the GPU on Direct3D 12 (Windows): the shader
  compiler's (FXC's) optimizer got the vegetation wrong, so it goes without it on
  Windows; and Windows CI's software GPU (WARP) lost what the bent rays' code wrote
  after returning early from a function, which it no longer does. Each now agrees with
  the CPU there as on other GPUs (see
  [Validation](validation.md#outdoor-scenes-on-the-gpu)).
- [ ] Bent rays through vegetation on the GPU on Direct3D 12: FXC takes more than an
  hour over the tracer with both, so the CPU traces a scene with both there; the newer
  compiler (DXC) hung over the tracer's shaders in Windows CI.
- [x] O13 Eddies weaker with height: `turbulence.height_m` (the height the structure
  parameters hold at) and `falloff` (2/3 by default, the surface layer's; up to 4/3 in
  strong sunshine) make the eddies weaken with height from that height up, no nearer
  the ground than their outer scale, in the coherence of each pair of paths (Cn² at
  their mean height), the scattering into shadows and the shaking of the refraction; on
  the command line's `inspect` and in the editor. Checked against the parabolic
  equation through eddies strengthening right down to the ground
  (`validation/turbulence.py --profile`). Without `height_m` nothing changes.
- [x] O14 Grounds of extended reaction: `impedance.reaction: extended` gives each
  reflection the impedance of its own angle (sound travelling sideways inside the
  ground, Attenborough's models), which tells for soft snow, moss, forest floors and
  thin layers; the early paths at each reflection's angle, the parabolic equation at
  grazing incidence, the wave solver locally. Checked against the fast field program
  with the ground's reflection at every angle (`validation/ground_reflection.py
  --extended`). Not with `two_parameter`; `local` by default, as before.
- [x] O15 Weather that changes over the scene: `atmosphere.zones`, polygons in the plan
  with weather of their own (the lee of buildings, a lake under an inversion, a warm
  car park), taking what they leave out from the scene's; the parabolic equation and
  the wave solver go through them, the early paths over the mean profile along their
  way and the late rays through the mean over the scene. Checked against a
  Crank-Nicholson parabolic equation marched through the zones
  (`validation/weather_zones.py`); drawn and set in the editor. Without zones nothing
  changes.
- [x] O16 Levels over a long term: `long_term` runs the scene in weather classes with
  their shares of the day, the evening and the night, and gives each period's level,
  the long-term level and L_den (or L_R) at every listener and map point. Checked
  against the formulas and ISO 9613-2's meteorological correction
  (`validation/long_term.py`); set in the editor. Without it nothing changes.
- [x] O17 Moving sources and Doppler: `walk.doppler` plays each early path through a
  delay that changes as the source or the listener goes, and `pass_by` gives L_Amax,
  the sound exposure level and L_Aeq (`walkthrough --levels`). Checked against the
  exact field of a passing source over a rigid ground (`validation/doppler.py`).
  Without it nothing changes.
- [x] O18 Moving air in the wave solver: `simulation.wave.wind: moving` carries the
  sound along with the wind before each step, so sound a building sends back goes
  against the wind. Checked against the exact field in a uniform wind, a façade's echo
  against ray theory and the parabolic equation in the `windy` preset
  (`validation/wave_wind.py`). Runs on the CPU; the wind's shear term on the air's
  motion stays out. Without it nothing changes.
- [x] O19 Whole cities: edge diffraction skips the pairs of an edge and a wall or roof
  that no path can use before it traces a ray, on every core, with the same results:
  900 buildings simulate in about 2 s on four cores instead of 81 s, and 3,600 in
  about 22 s, where 1,600 ran out of memory (`validation/city_speed.py`); with the
  reflections that fade at edges of 4.0.0, about 17 and 56 s.
- [x] O21 A wide-angle parabolic equation: `simulation.early.parabolic_equation:
  wide_angle` gives the bands the parabolic equation gives from the split-step Padé
  method (Collins' RAM: ten terms, steps of three wavelengths, fourth-order differences
  in height, its own starting field), for gradients that bend sound through large
  angles over several kilometres. On the 1995 benchmark it is within 0.11 dB of the
  fast field program from 10 Hz to 1 kHz and to 10 km wherever the sound is within
  40 dB of free field, where the narrow-angle one strays by 3 to 20 dB; in the weather
  presets within 0.04 dB on average (the narrow-angle one 0.11). Three to four times
  the work; without it nothing changes.

Sources:

- [x] Coherent arrays added up element by element near their listeners
  (`near_field: true`): the direct sound, the reflections and the sound that reaches an
  edge, exact over each band and the pattern far away.
- [x] Sources and listeners in groups, and left out of a run without being removed
  (`group`, `enabled: false`); in the editor a tick and an eye per source, listener
  and group.

Examples:

- [x] A concert in a stadium with a sound system: main line arrays, subwoofers, fills,
  delay towers and delays timed to them, with the stage's monitors (`stadium_concert`,
  whose sound system plays alone with its stage and monitors left out); the stage
  alone with the singer's microphone's gain before feedback (`concert_stage`); every
  array added up box by box near its listeners; the system tuned at the mix position,
  as a system engineer tunes it; and the editor's examples in groups.
- [x] Demos with their scripts: a choir and an organ in the church, a band with its
  sound system in the stadium (one of three songs) and a song carried across a lake by
  a lawn concert's line arrays and subwoofers, in the evening's weather and in still
  air, and the orchestra in the concert hall, its four seats toured through three
  movements; each downloads its recordings, mixes them at every listener and makes a
  video that tours them. The stadium's pictures show where the crowd stands and sits,
  the empty seats, and how much each takes up.
- [x] Everything a demo plays in its example's scene, left out of the example's own
  simulations until ticked, so the editor shows it: the stadium's band and the singer's
  microphone, the church's choir, organ pipes and toured places, the lakeside's sound
  system and its listener behind the stage, the gallery's place to look round from.
- [x] A walk through a gallery, the fifth demo and the first of speech: a guide talks
  while a visitor walks away and round a corner, then turns their head in front of
  them (the `gallery` example).
- [x] One shared package behind the demo scripts (`acoustic_spaces.demos`): a demo
  describes its steps and options and gets the command line, the recordings, the
  simulation, the mixes, the tour, the stills and the video, and runs from the editor
  as well as from a terminal.
- [x] Free HRTF sets to download, with their licences, from the command line
  (`acoustic-spaces hrtf`) and from the editor.
- [x] A demo of any scene (`acoustic-spaces demo`): each source plays the WAV file it
  names, at its sound power, and every listener is heard in turn in a video drawn from
  the scene; every demo's stills say how the sound was made and the command that makes
  it again.
- [x] A recording of the user's for any source of any demo (`--input SOURCE=WAV`), at
  the level of the feed it replaces, and a table of the sources in Run demo…
- [x] A song of the user's own in the stadium and lakeside demos (`--music`, `--bpm`,
  `--credit`; the stadium's stems with `--stem NAME=WAV` and its first downbeat), nothing
  downloaded, with a file field, a number and a table of the stems in Run demo…
- [x] A recording for a scene's sound system in the demo of any scene: the scene's
  `recording` (a file of the user's, or a free song or talk downloaded the first time)
  and each source's `channel` of it (left, right or mono), on the scene's and the
  source's pages in the editor; every example with a stage, a sound system or a talker
  plays one out of the box, its talkers at the normal vocal effort of IEC 60268-16.

Movement:

- [x] Walks along timed keyframes (position and facing at each time, waits and turns on
  the spot), walks that can be switched off, and every mover at once on one clock in one
  simulation (`walkthrough --all`), heard at every listener, with a video of them all.
- [x] Editor: the Movement panel, with the keyframes of a source or listener in a table
  (no speed to set: it follows from the times), a timeline that can be made longer to
  add keyframes past the end, a preview of every mover in the 3D view and the plan,
  draggable keyframe handles, and the movement rendered in the background.
- [x] One scene still or moving: walks turned on and off rather than copies of a scene
  with and without them; the demo of any scene plays the walks that are on, with a
  moving picture, and `--moving` picks them for a run; the Run demo window ticks what
  moves and opens the Movement panel.
- [x] A car passing by with Doppler: a car in the `roadside` and `street` examples with
  CNOSSOS-EU's sound power, a pass-by demo for both with a sound made to that power and
  the level over time, the demo of any scene playing one walk with Doppler, and the
  Movement panel's **Doppler** tick.
- [x] Editing a car's walk as it is: a walk along points edited in the Movement panel
  (points, one speed), keyframes at a steady speed accepted by the pass-by demo,
  **Render movement…** heard binaurally and with Doppler, and units for every number in
  the panel and in Run demo, with the pass's length, speed and duration.
- [x] Faster walks outdoors: the places of a walking source share the parabolic
  equation's runs, each listener getting what a run of its own place would give it.
- [x] The sound a source sends each way, along a walk with Doppler: each early path
  carries what is sent the way it leaves the source (`acoustic_spaces.emission`).
- [x] A helicopter's rotors: their tones from the blades' steady loads, checked against
  Gutin's hover formula and Farassat's formulation in time, and Pegg's broadband noise,
  checked against his worked example (`acoustic_spaces.rotorcraft`).
- [x] A helicopter's broadband noise from each blade section's own noise (Brooks, Pope
  and Marcolini's airfoil self-noise), checked against its authors' sample outputs, or
  from Pegg's method, chosen per helicopter.
- [ ] A helicopter's sound within its certified level: the AS350 B3's overflight comes
  out 3.4 dB over with Pegg's broadband noise and 4.5 dB under with the blade
  sections' own, which leaves out the blades cutting through the wake of those ahead
  and through turbulence.
- [x] A helicopter flying over a meadow: the `helicopter_flyover` example and its
  flyover demo, the rotors trimmed at the helicopter's airspeed against the wind.
- [x] A helicopter crossing over the stadium concert: `helicopter_demo.py`, its pass
  added to the concert's mixes at their simulated levels, 150 m up halfway through the
  song, or 330 m as the rules of the air ask over a crowd without leave. The helicopter
  and its walk are in the stadium's scene, left out of the concert's simulations, to
  see and change in the editor.
- [x] Trucks, buses, motorcycles and a police car with its siren on, in the pass-by
  demos as the car is (`--vehicle`): CNOSSOS-EU's sound power of their category, and the
  siren's level and directivity as measured, checked against California's and the
  Swiss rules for sirens; the police car's own sound (CNOSSOS-EU's car) along with its
  siren, or the siren alone.
- [x] A fire engine in the street: a medium heavy vehicle (CNOSSOS-EU's category 2)
  with its siren's two tones as loud as the Swiss rule allows on a vehicle and an air
  horn in blasts as loud as the rule for vehicles' horns allows on its bench, both
  checked against those rules; the vehicle, the siren and the horn heard together, or
  without the horn or the vehicle.
- [ ] Trains and trams: Schall 03 by default, CNOSSOS-EU's railway noise to choose, in a
  `railway` example and a tram in the street.
- [ ] Construction plant (breakers, excavators, generators), car horns and reversing
  alarms.
- [ ] Aircraft taking off, landing and flying over.
- [ ] Ground-borne vibration from trains and construction plant, and the noise it makes
  in the rooms of a building.
- [ ] Movement: a path drawn in the plan click by click.

Editor:

- [x] A simpler Run demo window: the options that change what a demo plays, in words,
  and the step, the paths, what fine-tunes a run and the command under Advanced.
- [x] An example's scripts that are no demos (the singer's microphone's feedback, the
  loudspeakers' alignment) run from the Run demo window.
- [x] The Run demo window says what a run downloads and how much, above Run; the
  orchestra demo downloads its tracks itself, as the other demos do their recordings.
- [x] The listeners a demo plays, picked for a run: `--listeners` in every demo with
  several listeners, and a Listeners list to tick in the Run demo window.
- [x] The background task in the status bar says what runs, with Cancel beside
  it; messages that can be shown or hidden by level, filtered by item or words, and
  cleared, every other one shaded.
- [x] Markers sized from the spacing of the sources and listeners, smaller where they
  would overlap, and arrows that stop short of a marker ahead.
- [x] The wind drawn outdoors, in the 3D view and the plan: arrows the way it blows, as
  long as it is strong, the weather zones with their winds, and masts with the wind's
  profile up them; eyes for both in the scene tree, and an example to see them
  (`lakeside_concert`).
- [x] The particle export to Blender asks how much of the sound, how slowly and how many
  particles, outdoors at first enough to reach the farthest listener.
- [x] Simulation > Run demo…: the demos of the open scene (This scene, and the
  example's own for an example's scene), its options at the command line's defaults,
  run in the background with its messages in the panel; the source page's Plays row
  picks the WAV file each source plays, and the window's table of sources the file each
  plays instead in that run (`--input`).
- [x] Download HRTFs…: the free sets with their licences, fetched into the user's data
  folder and put in the scene's binaural output.
- [x] The Scene YAML panel: the scene as coloured YAML text that follows every edit, and
  edited text applied with the checks every edit goes through.

Running on other computers, in steps:

- [x] Packs (`acoustic-spaces pack`, **File > Pack for another computer**): a scene and
  every file it names in one zip, which every command that reads a scene takes, and a
  container image that runs them headless on a GPU through Vulkan.
- [ ] `acoustic-spaces remote`: send a pack to a server over SSH or to Hugging Face
  Jobs, follow its progress, cancel it and bring the results back.
- [ ] **Run remotely** in the editor, with the results opened from the folder that comes
  back.

Version 4.0.1:

- [x] The archway's 1.4 dB, where the reflection off the jamb is found: in the bands
  that add in energy the terms are kept as waves, whose energy is never below nothing,
  and the events a path gets past split like those that stop it, so the level steps
  there by at most 0.11 dB from 250 Hz to 1 kHz.
- [x] A term past the foot of its edge, where the edge stands on the plane of the next
  reflection, goes on as the term of the edge's image: behind the archway's wall the
  level steps by at most 0.08 dB from 250 Hz to 1 kHz, and no longer by 0.63 dB at
  125 Hz at y = 6.32 m.
- [x] Reflection phases from an admittance fitted to a material's absorption in every
  band, so that the level at 63 and 125 Hz no longer depends on how many bands add in
  pressure.
- [x] Rooms with a curved surface of many facets, as BRAS CR2 and CR3, run at their
  defaults instead of running out of memory, in octave and in third-octave bands.
- [x] The validation report's measured rooms (BRAS) and its cross-check with
  pyroomacoustics run again with this version, with their run times and memory.
- [x] The edge fade walked in CI: the `archway`'s whole line behind its wall, the
  `gallery` round its corner and the `street`'s garden round its four largest steps
  (`tests/test_walks.py`), with `validation/walks.py` for the whole walks. The edge
  diffraction and the fade checked against the exact solution past a thin screen's
  corner, behind a barrier on a hard ground with `first_order` (`one_side` and
  `both_sides`) and behind a thick barrier with `second_order`.

Known issues found by the review of 25 October 2026 (see
[Validation](validation.md#known-issues)), in the order they come next:

- [x] Paths through three partitions or more, which neither part counted: the late
  stage counts those that reflect, and the early stage leaves them to it.
- [x] Packs that keep an earlier one when writing fails, and result file names that
  can't be the same for two names.
- [x] Vertices that no face uses dropped, coordinates that are not finite refused in
  every format, and a solid standing on a surface (a block on the floor) found to be
  solid from several of its faces.
- [x] Sources and listeners inside a closed solid reported, and `transmission: 0` as a
  surface that lets nothing through.
- [x] The late part's rain kept from going out of the rooms and back in, as the rays
  are.
- [x] G against a free-field response that starts at its first sample: the free-field
  reference keeps what its band filters spread before it.
- [ ] The first milliseconds of a rendered response, where the band filters' spread
  before the first sample is lost: filters that don't spread before an arrival.
- [x] No path arriving before it leaves the source, and errors in the native code's
  threads raised in Python.
- [ ] The echogram's pairs of paths from 1 kHz up without adding energy where they
  cancel.
- [x] A warning for coefficients outside 0 to 1, and the wave solver's grid checked
  before the weather's fields and the first one let go round an open model.
- [x] The music echo criterion through one band-pass over its two octaves, as Dietsch
  and Kraak have it.
- [x] Limits on the size of a run: its memory estimated before it starts, for the
  computer's cores and devices, and a run past the computer's memory refused (threads
  above the cores run as the cores).
- [x] Results written whole or not at all, a response at a time, and `simulate`'s
  folder checked before its run.
- [x] The mixing time from the listener's room (the detector stays sized for all the
  rooms, which the rays spread over).
- [x] The real-time convolver writing each block into buffers made before playback starts.
- [x] A warning when the wave solver's grid is coarser than the default, with how slow
  its shortest waves travel, and that in each run's details.
- [x] The wave solver's buffers on a GPU all checked against what it binds, and its
  recording at the listeners kept once.
- [x] The late part's last milliseconds with the whole chords of the rays through the
  listeners' spheres and the map's cells.
- [x] Re-weighting for other absorption (the editor's results after an absorption
  change) turns the phases of the early reflections where they add in pressure, as
  tracing again has them.
- [x] The wave solver's highest modes kept inside its stability limit in single
  precision, with both stencils.
- [x] The review's lower-severity findings, each checked: faces given twice dropped,
  sources and listeners on a surface refused and listeners next to one warned of,
  `validate --strict`, YAML keys written twice refused, packs unpacked only with room
  for them and each file checked before it takes its name, the late part's smoothing
  kept after 50 ms, the tracer's reflection counts in 32 bits, and `from_impulse_response`
  and `wave_responses` checking what they are given. The others were not borne out, or
  only a direct caller of the native module can reach them.
- [x] CI: type checks, sanitizers for the native code and the tests' coverage.
- [ ] Type checks passing in the modules that did not when they began (listed in
  `pyproject.toml`), a few at a time.
- [ ] Collada polygons with their holes.
- [ ] Vector-based scattering with a lobe centred on the specular direction, so the late
  part carries the first orders' specular share as the exact paths do, checked against
  the BRAS rooms' decay.

Next version:

- [x] Uncertainty: each parameter's standard error, how much it would change between
  runs with other seeds, from the late rays traced in batches (a jackknife over them),
  in `parameters.json`, `simulate`'s output and the editor's results; checked against
  the spread over seeds.
- [x] EDT's line fit always from the onset, however the energy is summed.
- [x] The path over a barrier's top with the reflection off the ground behind it, kept
  where the reflection before the barrier is found: behind a barrier on a hard ground
  the level follows the exact one within 0.05 dB with `first_order` and `both_sides`,
  instead of stepping by 0.35 dB at 63 Hz. Checked against BRAS: the seminar room's and
  the hall's average differences from the measurement move by 0.15 JNDs at most in
  each band, but for the calibrated hall's (0.67), whose fit lands on other
  coefficients.
- [x] Models cleaned where an edge runs past a corner at which three faces meet: BRAS's
  auditorium (CR4) is closed.
- [x] More measured rooms: BRAS's coupled rooms (CR1, the door open by 4.1°) and
  auditorium (CR4) beside the seminar room and the hall, G from seat to seat in all four
  with the measured level's 3 dB question reported, IACC of FABIAN's binaural responses
  in three (`validation/bras_binaural.py`), and each run's largest standard error.
- [ ] BRAS's coupled rooms with the door open by 30.4°, whose model has no closed inside:
  the door leaf cuts some 2 cm into the wall at its hinge.
- [x] Random scattering traced band by band where the bands scatter differently, as
  vector-based scattering is: with one ray for all bands, re-weighted band by band, its
  decays in BRAS were noisy, and in the coupled rooms far too short.
- [x] Paths over edges that arrive together rendered with phases of their own, so they
  render their energy on average over seeds instead of adding in amplitude; the
  echogram's 63 and 125 Hz reading no longer takes their interference a second time
  from the band above.
- [x] Listeners as microphones: a first-order pattern, or a coincident pair (XY,
  Blumlein, MS) in two channels, that the mono output picks up through, aimed where the
  listener faces; chosen in the editor's listener form and drawn as balloons in the 3D
  view.
- [x] Microphone patterns that change with frequency: one per octave or third-octave
  band, for a capsule or a pair's capsules, rendered exactly from the same paths.
- [ ] A microphone's measured on-axis response and measured patterns (files), as
  sources have.
- [ ] Spaced pairs whose late parts correlate as real capsules' do at low frequencies.
- [x] Heavy vehicles, buses and motorcycles in the pass-by demos with CNOSSOS-EU's sound
  power of their category (a truck in the roadside, a bus and a motorcycle in the
  street), and a police car with its siren on; the car's sound power by the 2021
  amendment of CNOSSOS-EU's table, 2.5 dB(A) more at 50 km/h.
